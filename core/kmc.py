"""
core/kmc.py

Событийно-ориентированный движок (kinetic Monte Carlo, метод Гиллеспи,
"n-fold way") для той же конкурирующей динамики флип/переезд.

Зачем
-----
1. ТОЧНОЕ соответствие мастер-уравнению. Батч-MCS из core/simulation.py —
   аппроксимация непрерывно-временной цепи Маркова с генератором Q
   (см. отчёт, раздел 2). KMC сэмплирует эту цепь строго: интервалы
   между событиями ~ Exp(R), где R — полная интенсивность, поэтому
   траектории x(t) напрямую сравнимы с решением ẋ = Qx (mean-field и
   другие замыкания).
2. Скорость при N -> ∞. Стоимость одного события O(z) (локальное
   обновление counts и списка несчастных), без полных проходов по
   решётке. В замороженном состоянии KMC делает большие скачки по
   времени — асинхронная динамика не тратит шаги на «ничего не случилось».

Семантика (согласована с FlipOrMove)
------------------------------------
Каждый агент имеет два Пуассоновских процесса:
  - флип с интенсивностью r_f = p        (смена типа на случайный другой);
  - переезд с интенсивностью r_m = 1 - p (только если агент несчастлив;
    цель — случайная свободная клетка).
Единица времени: 1 MCS = интервал, за который каждый агент в среднем
совершает одно действие. Поэтому t_KMC сравнимо с номером MCS.

Алгоритм (n-fold way):
  R = p * N_agents + (1 - p) * N_unhappy
  dt ~ Exp(R);  с вероятностью p*N_agents/R — флип случайного агента,
  иначе — переезд случайного несчастного в случайную вакансию.
После события локально обновляются counts и флаги счастья в
окрестностях затронутых клеток — O(z).
"""

from __future__ import annotations

import numpy as np

from .policies import EMPTY


class KMCSimulation:
    """Гиллеспи-движок поверх той же конфигурации, что у Simulation.

    Параметры
    ---------
    grid : int8 массив, EMPTY = -1
    offsets : (z, dim) смещения соседей
    n_types : K
    threshold : порог терпимости T (та же политика ThresholdSatisfaction)
    flip_prob : p
    rng : np.random.Generator

    Атрибуты
    --------
    t : текущее время (в MCS-эквивалентах)
    n_events : число событий
    """

    def __init__(self, grid: np.ndarray, offsets: np.ndarray, n_types: int,
                 threshold: float, flip_prob: float,
                 rng: np.random.Generator | None = None):
        self.grid = grid
        self.offsets = np.asarray(offsets)
        self.n_types = int(n_types)
        self.threshold = float(threshold)
        self.flip_prob = float(flip_prob)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.shape = grid.shape
        self.t = 0.0
        self.n_events = 0

        self._counts = self._full_counts()
        self._happy = self._happy_mask(self._counts)
        # динамические списки (numpy-массивы flat-индексов + позиции для O(1) удаления)
        self._agents = _IndexSet(np.flatnonzero(grid != EMPTY))
        self._unhappy = _IndexSet(
            np.flatnonzero((grid != EMPTY) & ~self._happy))
        self._vacant = _IndexSet(np.flatnonzero(grid == EMPTY))

    # ----------------------------------------------------- базовые пересчёты

    def _full_counts(self) -> np.ndarray:
        counts = np.zeros((self.n_types,) + self.shape, dtype=np.int32)
        for off in self.offsets:
            shifted = np.roll(self.grid, shift=tuple(off),
                              axis=tuple(range(self.grid.ndim)))
            for k in range(self.n_types):
                counts[k] += (shifted == k)
        return counts

    def _happy_mask(self, counts: np.ndarray) -> np.ndarray:
        total = counts.sum(axis=0)
        occupied = self.grid != EMPTY
        own = np.zeros(self.shape, dtype=np.float64)
        np.divide(
            np.take_along_axis(counts, np.clip(self.grid, 0, None)[None],
                               axis=0)[0],
            total, out=own, where=(total > 0))
        return ~occupied | (own >= self.threshold)

    def _happy_at(self, flat: np.ndarray) -> np.ndarray:
        """Счастье отдельных клеток по кешированным counts — O(len(flat))."""
        cf = self._counts.reshape(self.n_types, -1)
        total = cf[:, flat].sum(axis=0)
        g = self.grid.flat[flat]
        own = cf[g, flat]
        # семантика ThresholdSatisfaction: доля своих считается по ЗАНЯТЫМ
        # соседям; при total == 0 доля = 0.0 (как в policies.py)
        frac = np.where(total > 0, own / np.maximum(total, 1), 0.0)
        return frac >= self.threshold

    # ----------------------------------------------------- локальное обновление

    def _apply_changes(self, cells: np.ndarray, old: np.ndarray,
                       new: np.ndarray) -> None:
        """Применить изменения типов клеток и обновить все кеши.

        cells — (m,) flat-индексы; old/new — (m,) типы до/после.
        """
        m = cells.size
        if m == 0:
            return
        grid_flat = self.grid.reshape(-1)
        counts_flat = self._counts.reshape(self.n_types, -1)

        # 1. counts: окрестности изменённых клеток — O(m*z)
        cells_multi = np.stack(np.unravel_index(cells, self.shape), axis=1)
        nbr = (cells_multi[:, None, :] + self.offsets[None, :, :]) \
            % np.array(self.shape)
        nbr_flat = np.ravel_multi_index(
            nbr.reshape(-1, self.grid.ndim).T, self.shape)
        n_cells = counts_flat.shape[1]
        idx_parts, w_parts = [], []
        for val, delta in ((np.repeat(old, self.offsets.shape[0]), -1),
                           (np.repeat(new, self.offsets.shape[0]), +1)):
            mask = val != EMPTY
            idx_parts.append(val[mask] * n_cells + nbr_flat[mask])
            w_parts.append(np.full(mask.sum(), delta, dtype=np.int32))
        if idx_parts:
            idx = np.concatenate(idx_parts)
            w = np.concatenate(w_parts)
            bc = np.bincount(idx, weights=w,
                             minlength=self.n_types * n_cells)
            counts_flat += bc.reshape(self.n_types, n_cells).astype(np.int32)

        # 2. сами клетки + списки агентов/вакансий
        for i in range(m):
            c, o, n = cells[i], old[i], new[i]
            grid_flat[c] = n
            if o == EMPTY:                    # заселение
                self._vacant.remove(c)
                self._agents.add(c)
            elif n == EMPTY:                  # освобождение
                self._agents.remove(c)
                self._vacant.add(c)
                # пустая клетка не участвует в динамике: убрать из
                # списка несчастных и вернуть нейтральный флаг счастья,
                # иначе «протухшая» запись в unhappy сломает списки
                self._unhappy.remove(c)
                self._happy.flat[c] = True

        # 3. счастье: пересчитать клетки окрестностей + сами клетки — O(m*z)
        affected = np.unique(np.concatenate([nbr_flat, cells]))
        affected = affected[self.grid.flat[affected] != EMPTY]
        if affected.size:
            was = self._happy.flat[affected]
            now = self._happy_at(affected)
            self._happy.flat[affected] = now
            flipped = affected[was != now]
            for c in flipped:
                if self._happy.flat[c]:
                    self._unhappy.remove(c)
                else:
                    self._unhappy.add(c)

    # ----------------------------------------------------- события

    def _do_flip(self) -> None:
        a = self._agents.random(self.rng)
        cur = int(self.grid.flat[a])
        shift = int(self.rng.integers(1, self.n_types))
        self._apply_changes(np.array([a]), np.array([cur]),
                            np.array([(cur + shift) % self.n_types]))

    def _do_move(self) -> None:
        src = self._unhappy.random(self.rng)
        dst = self._vacant.random(self.rng)
        typ = int(self.grid.flat[src])
        self._apply_changes(np.array([src, dst]),
                            np.array([typ, EMPTY]),
                            np.array([EMPTY, typ]))

    def event(self) -> float:
        """Одно событие Гиллеспи. Возвращает dt (или inf, если событий нет)."""
        n_a = self._agents.size
        n_u = self._unhappy.size
        rate_flip = self.flip_prob * n_a
        # канал переездов отключён, если свободных клеток нет
        rate_move = (1.0 - self.flip_prob) * n_u if self._vacant.size else 0.0
        R = rate_flip + rate_move
        if R <= 0.0 or n_a == 0:
            return np.inf                      # поглощающее состояние
        dt = self.rng.exponential(1.0 / R)
        if self.rng.random() < rate_flip / R:
            self._do_flip()
        else:
            self._do_move()
        self.t += dt
        self.n_events += 1
        return dt

    def run_until(self, t_end: float) -> None:
        """Крутить до времени t_end (в MCS-эквивалентах)."""
        while self.t < t_end:
            if not np.isfinite(self.event()):
                break

    # ----------------------------------------------------- наблюдаемые

    def counts(self) -> np.ndarray:
        return self._counts

    def happiness(self) -> float:
        if self._agents.size == 0:
            return float("nan")
        return 1.0 - self._unhappy.size / self._agents.size


class _IndexSet:
    """Множество flat-индексов с O(1) добавлением/удалением/случайным
    выбором: массив значений + массив позиций (swap-remove)."""

    def __init__(self, values: np.ndarray):
        cap = max(int(values.size) * 2, 16)
        # ёмкость выбирается вызывающим кодом через resize при надобности;
        # для агентов/вакансий суммарный размер ограничен числом клеток.
        self.cap = cap
        self.vals = np.full(cap, -1, dtype=np.int64)
        self.pos = None                         # ленивая инициализация
        self.size = int(values.size)
        self.vals[:self.size] = values
        self._init_pos()

    def _init_pos(self) -> None:
        mx = self.vals[:self.size].max(initial=-1) + 1
        self.pos = np.full(mx, -1, dtype=np.int64)
        self.pos[self.vals[:self.size]] = np.arange(self.size)

    def _grow_pos(self, v: int) -> None:
        if v >= self.pos.size:
            new = np.full(max(v + 1, 2 * self.pos.size), -1, dtype=np.int64)
            new[:self.pos.size] = self.pos
            self.pos = new

    def _grow_vals(self) -> None:
        if self.size >= self.cap:
            new = np.full(2 * self.cap, -1, dtype=np.int64)
            new[:self.size] = self.vals[:self.size]
            self.vals, self.cap = new, 2 * self.cap

    def add(self, v: int) -> None:
        self._grow_pos(v)
        if self.pos[v] >= 0:
            return
        self._grow_vals()
        self.vals[self.size] = v
        self.pos[v] = self.size
        self.size += 1

    def remove(self, v: int) -> None:
        p = self.pos[v]
        if p < 0:
            return
        last = self.vals[self.size - 1]
        self.vals[p] = last
        self.pos[last] = p
        self.pos[v] = -1
        self.size -= 1

    def random(self, rng: np.random.Generator) -> int:
        return int(self.vals[rng.integers(0, self.size)])
