"""
vizual/animate.py

Живая анимация модели Шеллинга в окне Taichi GUI с панелью ползунков.

Динамика — из core/ (numpy), кадр собирается в numpy + Pillow,
отображение — taichi.GUI. Запуск из корня проекта:

    .venv/Scripts/python -m vizual.animate --dim 2
    .venv/Scripts/python -m vizual.animate --dim 1 --length 400

Конвенции taichi.GUI (важно, тут легко обжечься):
  - res окна = (width, height); буфер кадра хранится как B[x, y] (оси
    перевернуты относимо привычной картинки (rows, cols));
  - get_cursor_pos возвращает нормированные координаты с НИЖНЕГО левого
    угла, y вверх. Мы рисуем в обычной системе «y от верха», поэтому
    my = (1 - cy) * H.

Ползунки внизу окна (мышь, перетаскивание):
    T   — порог терпимости (применяется сразу)
    p   — вероятность флипа (применяется сразу)
    rho — плотность занятости (применяется при отпускании / R)

Клавиши (английская раскладка):
    SPACE  — пауза / продолжить
    R      — перезапуск с текущими ползунками и новым seed
    LEFT/RIGHT — T (шаг 0.02),  UP/DOWN — p (шаг 0.01)
    [ / ]  — шагов симуляции на кадр (÷/× 2)
    U      — оверлей: подсветить несчастных агентов (2D)
    ESC    — выход

В 2D-режиме справа на панели — «искровые» графики (sparklines):
истории I(t) и happy(t) по кадрам, помогают увидеть выход на стационар.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# запуск и как скрипт (python vizual/animate.py): корень проекта — в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import taichi as ti
from PIL import Image, ImageDraw, ImageFont

from core.factories import make_schelling_1d, make_schelling_2d

ti.init(arch=ti.cpu)  # GUI работает и на cpu-архитектуре; динамика — numpy

# ------------------------------------------------------------ палитра
# индекс = тип + 1 (EMPTY = -1 -> 0). Фон тёмный, агенты — тёплые тона.
PALETTE = np.array([
    [22, 24, 31],    # EMPTY      — почти чёрный
    [76, 150, 255],  # тип 0      — синий
    [255, 170, 69],  # тип 1      — оранжевый
    [125, 255, 154], # тип 2      — зелёный
    [255, 107, 138], # тип 3      — розовый
], dtype=np.uint8)

PANEL_BG = (23, 26, 34)
TRACK_BG = (42, 46, 58)
ACCENT   = (76, 150, 255)
ACCENT_H = (120, 180, 255)   # акцент при наведении/перетаскивании
KNOB     = (235, 238, 245)
TEXT     = (207, 212, 224)

def _load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Шрифт с кириллицей. load_default(size) в новых Pillow — Aileron
    (латиница только), поэтому ищем системный TTF."""
    for name in ("arial.ttf", "segoeui.ttf", "calibri.ttf"):
        try:
            return ImageFont.truetype(name, 14)
        except OSError:
            continue
    return ImageFont.load_default(14)


_FONT = _load_font()


def render(grid: np.ndarray) -> np.ndarray:
    """grid (int8, EMPTY=-1) -> RGB-изображение (..., 3) uint8."""
    return PALETTE[np.clip(grid.astype(np.int64) + 1, 0, PALETTE.shape[0] - 1)]


def _upscale(img: np.ndarray, s: int) -> np.ndarray:
    """Целочисленный nearest-neighbor апскейл (чёткие пиксели клеток)."""
    if s <= 1:
        return img
    return np.repeat(np.repeat(img, s, axis=0), s, axis=1)


# ------------------------------------------------------------ кадр

class Frame:
    """Буфер кадра в конвенции taichi: shape (W, H, 3), индекс B[x, y],
    y отсчитывается снизу экрана. Внутри — обычные картинки (rows, cols),
    которые вставляются «верхом вверх» через blit."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.buf = np.zeros((width, height, 3), dtype=np.uint8)

    def blit(self, img: np.ndarray, y0: int) -> None:
        """img (rows, cols, 3), y0 — экранный y её НИЖНЕГО края."""
        img = np.ascontiguousarray(img)
        h = img.shape[0]
        self.buf[:, y0:y0 + h] = img[::-1].transpose(1, 0, 2)

    def fill(self, img: np.ndarray, y0: int) -> None:
        """img (W, h, 3), не зависящая от y, — без переворота."""
        self.buf[:, y0:y0 + img.shape[1]] = img

    def numpy(self) -> np.ndarray:
        """Готовый буфер для gui.set_image (shape == (W, H, 3))."""
        return self.buf


# ------------------------------------------------------------ ползунки

class Slider:
    """Горизонтальный ползунок. Координаты — пиксели панели,
    x вправо, y от её верха."""

    TRACK_H = 10
    KNOB_R = 9
    ROW_H = 30          # шаг между дорожками

    def __init__(self, label: str, vmin: float, vmax: float, value: float):
        self.label = label
        self.vmin, self.vmax = float(vmin), float(vmax)
        self.value = float(value)
        self.x0 = self.y0 = self.w = 0
        self.dragging = False
        self.just_released = False   # отпустили в этом кадре (для rho)

    def place(self, x: int, y: int, w: int) -> None:
        self.x0, self.y0, self.w = x, y, w

    def set_norm(self, t: float) -> None:
        t = min(max(t, 0.0), 1.0)
        self.value = self.vmin + t * (self.vmax - self.vmin)

    def norm(self) -> float:
        return (self.value - self.vmin) / (self.vmax - self.vmin)

    def knob_x(self) -> float:
        return self.x0 + self.w * self.norm()

    def in_row(self, mx: float, my: float) -> bool:
        return (self.x0 - self.KNOB_R - 2 <= mx <= self.x0 + self.w + self.KNOB_R + 2 and
                self.y0 - 8 <= my <= self.y0 + self.TRACK_H + 10)

    def on_mouse(self, mx: float, my: float, pressed: bool) -> None:
        """mx, my — пиксели панели (y от её верха)."""
        self.just_released = False
        if pressed and self.in_row(mx, my):
            self.dragging = True
        if self.dragging:
            self.set_norm((mx - self.x0) / self.w)
            if not pressed:
                self.dragging = False
                self.just_released = True

    def draw(self, d: ImageDraw.ImageDraw) -> None:
        active = self.dragging
        kx = self.knob_x()
        y0, y1 = self.y0, self.y0 + self.TRACK_H
        d.text((self.x0, self.y0 - 15),
               f"{self.label} = {self.value:.2f}", fill=TEXT, font=_FONT)
        d.rounded_rectangle([self.x0, y0, self.x0 + self.w, y1],
                            radius=self.TRACK_H // 2, fill=TRACK_BG)
        fill_col = ACCENT_H if active else ACCENT
        if kx > self.x0 + self.TRACK_H // 2:
            d.rounded_rectangle([self.x0, y0, kx, y1],
                                radius=self.TRACK_H // 2, fill=fill_col)
        cy = (y0 + y1) // 2
        r = self.KNOB_R
        d.ellipse([kx - r, cy - r - 1, kx + r, cy + r - 1],
                  fill=KNOB, outline=fill_col)


class Panel:
    """Нижняя панель: строка статуса + ползунки. Рисуется в обычную
    картинку (rows, cols); во Frame вставляется перевёрнутой по y —
    её «верх» оказывается у верхнего края панели на экране."""

    PAD = 14

    SPARK_W = 270     # ширина области искровых графиков
    SPARK_H = 40

    def __init__(self, height: int, sliders: list[Slider],
                 status_fn, history_fn=None) -> None:
        self.height = height
        self.sliders = sliders
        self.status_fn = status_fn
        self.history_fn = history_fn   # -> (I_hist, H_hist) или None

    def layout(self, width: int) -> None:
        self._w = width
        n = len(self.sliders)
        w = (width - self.PAD * (n + 1)) // n
        for i, s in enumerate(self.sliders):
            s.place(self.PAD + i * (w + self.PAD), self.PAD + 28, w)

    def on_mouse(self, mx: float, my_top: float, pressed: bool,
                 frame_h: int) -> None:
        """my_top — y в системе «от верха кадра». Панель — внизу."""
        in_panel = my_top >= frame_h - self.height
        if not in_panel:
            if not pressed:              # отпустили за пределами панели
                for s in self.sliders:
                    if s.dragging:
                        s.dragging = False
                        s.just_released = True
            return
        my_panel = my_top - (frame_h - self.height)
        for s in self.sliders:
            s.on_mouse(mx, my_panel, pressed)

    def on_mouse_norm(self, cx: float, cy: float, pressed: bool,
                      frame_h: int) -> None:
        """Нормированные координаты taichi (y от нижнего края) -> on_mouse."""
        self.on_mouse(cx * self._w, (1.0 - cy) * frame_h, pressed, frame_h)

    def _draw_sparkline(self, d: ImageDraw.ImageDraw) -> None:
        """Мини-графики I(t) и happy(t) в правом верхнем углу панели."""
        if self.history_fn is None:
            return
        i_hist, h_hist = self.history_fn()
        x1 = self._w - self.PAD - self.SPARK_W
        y0, y1 = 6, 6 + self.SPARK_H
        d.rectangle([x1, y0, self._w - self.PAD, y1], outline=TRACK_BG)
        for vals, col, name in ((i_hist, (120, 190, 255), "I"),
                                (h_hist, (255, 200, 110), "happy")):
            if len(vals) < 2:
                continue
            step = self.SPARK_W / max(len(vals) - 1, 1)
            pts = [(x1 + k * step,
                    y1 - min(max(v, 0.0), 1.0) * self.SPARK_H)
                   for k, v in enumerate(vals)]
            d.line(pts, fill=col)
            d.text((x1 + 4 + (0 if name == "I" else 40), y0 + 2),
                   name, fill=col, font=_FONT)

    def draw(self, frame: Frame) -> None:
        w = frame.width
        panel_img = Image.new("RGB", (w, self.height), PANEL_BG)
        d = ImageDraw.Draw(panel_img)
        d.text((self.PAD, 7), self.status_fn(), fill=TEXT, font=_FONT)
        for s in self.sliders:
            s.draw(d)
        self._draw_sparkline(d)
        frame.blit(np.asarray(panel_img), 0)


def status_fn(sim_box, mcs: dict, paused: bool, spf: int):
    def fn() -> str:
        sim = sim_box["sim"]
        iso = sim.observe("isolation")
        iso_s = f"{iso:.3f}" if np.isfinite(iso) else "nan"
        state = "PAUSE" if paused else "run"
        return (f"{state}  MCS {mcs['n']:>6} | I = {iso_s} | "
                f"happy = {sim.happiness():.3f} | "
                f"{spf} step/frame")
    return fn


# ------------------------------------------------------------ общий цикл

def _event_loop(gui, sim_box, sliders, rebuild, reset_hist, mcs,
                paused: bool, spf: int, panel: Panel, frame_h: int,
                on_key=None):
    """Клавиатура + мышь. Возвращает (running, paused, spf).

    on_key — необязательный обработчик дополнительных клавиш
    (например, 'u' — оверлей несчастных в 2D).
    """
    for e in gui.get_events():
        if on_key is not None and on_key(e.key):
            continue
        if e.key == ti.GUI.ESCAPE:
            gui.running = False
        elif e.key == ti.GUI.SPACE:
            paused = not paused
        elif e.key == "r":
            sim_box["sim"] = rebuild(None)
            reset_hist()
            mcs["n"] = 0
        elif e.key == ti.GUI.LEFT:
            sliders[0].set_norm(sliders[0].norm() - 0.02)
        elif e.key == ti.GUI.RIGHT:
            sliders[0].set_norm(sliders[0].norm() + 0.02)
        elif e.key == ti.GUI.DOWN:
            sliders[1].set_norm(sliders[1].norm() - 0.01)
        elif e.key == ti.GUI.UP:
            sliders[1].set_norm(sliders[1].norm() + 0.01)
        elif e.key == "[":
            spf = max(1, spf // 2)
        elif e.key == "]":
            spf = min(512, spf * 2)

    # мышь: taichi даёт (x, y) от нижнего левого угла, нормированные;
    # переворот y и перевод в пиксели — внутри panel.on_mouse_norm
    cx, cy = gui.get_cursor_pos()
    panel.on_mouse_norm(cx, cy, gui.is_pressed(ti.GUI.LMB), frame_h)

    if sliders[2].just_released:          # rho — применить перезапуском
        sim_box["sim"] = rebuild(None)
        reset_hist()
        mcs["n"] = 0

    sim = sim_box["sim"]
    sim.satisfaction.threshold = sliders[0].value
    sim.action.flip_prob = sliders[1].value
    return paused, spf


# ------------------------------------------------------------ 2D

def animate_2d(a) -> None:
    sim_box = {}
    sliders = [
        Slider("T", 0.0, 1.0, a.threshold),
        Slider("p", 0.0, 1.0, a.flip_prob),
        Slider("rho", 0.1, 0.95, a.density),
    ]

    def rebuild(seed):
        return make_schelling_2d(size=a.size, threshold=sliders[0].value,
                                 flip_prob=sliders[1].value,
                                 density=sliders[2].value,
                                 radius=a.radius, metric=a.metric,
                                 n_types=a.types, seed=seed)

    sim_box["sim"] = rebuild(a.seed)
    scale = max(1, 800 // a.size)
    W = a.size * scale
    panel_h = 104
    H = W + panel_h
    gui = ti.GUI("Schelling 2D", res=(W, H))   # res = (width, height)

    # истории метрик для искровых графиков (ограниченное окно)
    HIST_MAX = 600
    hist = {"I": [], "happy": []}
    overlay = {"unhappy": False}

    panel = Panel(panel_h, sliders, None,
                  history_fn=lambda: (hist["I"], hist["happy"]))
    panel.layout(W)

    frame = Frame(W, H)

    mcs = {"n": 0}
    paused = False
    spf = a.steps_per_frame

    def reset_hist():
        hist["I"].clear()
        hist["happy"].clear()

    def on_key(key):
        if key == "u":
            overlay["unhappy"] = not overlay["unhappy"]
            return True
        return False

    while gui.running:
        paused, spf = _event_loop(gui, sim_box, sliders, rebuild, reset_hist,
                                  mcs, paused, spf, panel, H, on_key=on_key)
        sim = sim_box["sim"]

        if not paused:
            sim.run(spf)
            mcs["n"] += spf
            hist["I"].append(sim.observe("isolation"))
            hist["happy"].append(sim.happiness())
            for k in hist:
                del hist[k][:-HIST_MAX]

        img = render(sim.grid)
        if overlay["unhappy"]:
            # подсветка несчастных агентов — динамика живёт на границах
            happy = sim.satisfaction(sim.counts(), sim.grid)
            mark = (sim.grid != -1) & ~happy
            img = img.copy()
            img[mark] = (245, 245, 250)

        frame.buf[...] = 0
        frame.blit(_upscale(img, scale), panel_h)
        panel.status_fn = status_fn(sim_box, mcs, paused, spf)
        panel.draw(frame)
        gui.set_image(frame.numpy())
        gui.show()


# ------------------------------------------------------------ 1D
# Сверху вниз на экране: полоса текущего кольца, под ней
# пространственно-временная диаграмма (время вниз, история перезаписывается),
# снизу — панель ползунков.

def animate_1d(a) -> None:
    sim_box = {}
    sliders = [
        Slider("T", 0.0, 1.0, a.threshold),
        Slider("p", 0.0, 1.0, a.flip_prob),
        Slider("rho", 0.1, 0.95, a.density),
    ]

    def rebuild(seed):
        return make_schelling_1d(length=a.length, threshold=sliders[0].value,
                                 flip_prob=sliders[1].value,
                                 density=sliders[2].value,
                                 radius=a.radius, n_types=a.types, seed=seed)

    sim_box["sim"] = rebuild(a.seed)
    L = a.length
    strip_h = 36
    hist_h = 520
    panel_h = 118
    H = strip_h + hist_h + panel_h
    W = L
    gui = ti.GUI("Schelling 1D", res=(W, H))

    hist = np.zeros((hist_h, L, 3), dtype=np.uint8)
    hist[...] = PALETTE[0]
    row = 0

    panel = Panel(panel_h, sliders, None)
    panel.layout(W)

    frame = Frame(W, H)

    mcs = {"n": 0}
    paused = False
    spf = a.steps_per_frame

    def reset_hist():
        nonlocal row
        hist[...] = PALETTE[0]
        row = 0

    while gui.running:
        paused, spf = _event_loop(gui, sim_box, sliders, rebuild, reset_hist,
                                  mcs, paused, spf, panel, H)
        sim = sim_box["sim"]

        if not paused:
            sim.run(spf)
            mcs["n"] += spf
            hist[row % hist_h] = render(sim.grid)
            row += 1

        frame.buf[...] = 0
        row_img = render(sim.grid)[:, None, :]                 # (L, 1, 3)
        frame.fill(np.repeat(row_img, strip_h, axis=1),  # полоса кольца
                   panel_h + hist_h)
        frame.blit(hist, panel_h)                              # spacetime
        panel.status_fn = status_fn(sim_box, mcs, paused, spf)
        panel.draw(frame)
        gui.set_image(frame.numpy())
        gui.show()


# ------------------------------------------------------------ CLI

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Анимация модели Шеллинга")
    p.add_argument("--dim", type=int, choices=[1, 2], default=2)
    p.add_argument("--size", type=int, default=160, help="сторона тора (2D)")
    p.add_argument("--length", type=int, default=300, help="длина кольца (1D)")
    p.add_argument("--threshold", type=float, default=1.0 / 3.0, help="порог T")
    p.add_argument("--flip-prob", type=float, default=0.0, help="вероятность флипа p")
    p.add_argument("--density", type=float, default=0.7, help="плотность занятости")
    p.add_argument("--radius", type=int, default=None,
                   help="радиус окна (умолч.: 2 в 1D, 1 в 2D)")
    p.add_argument("--metric", choices=["moore", "von_neumann"], default="moore")
    p.add_argument("--types", type=int, default=2, help="число типов K")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--steps-per-frame", type=int, default=1)
    a = p.parse_args(argv)
    if a.radius is None:
        a.radius = 2 if a.dim == 1 else 1
    return a


if __name__ == "__main__":
    # консоль Windows может быть не в UTF-8 — не падаем на русском докстринге
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = parse_args()
    print(__doc__)
    if args.dim == 2:
        animate_2d(args)
    else:
        animate_1d(args)
