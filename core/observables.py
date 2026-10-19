"""
core/observables.py

Метрики — чистые функции поверх состояния: читают grid и counts,
ничего не меняют. Каждая имеет сигнатуру (grid, counts, n_types).

Соглашение: EMPTY = -1 (импорт из policies, единая конвенция).

Реестр OBSERVABLES связывает строковое имя с функцией — через него
Simulation.observe() достаёт метрику по имени.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

from .policies import EMPTY

if TYPE_CHECKING:
    from .simulation import Simulation

# сигнатура любой метрики
ObsFn = Callable[[np.ndarray, np.ndarray, int], float | np.ndarray]


def isolation_index(grid: np.ndarray, counts: np.ndarray, n_types: int) -> float:
    """
    Глобальный индекс изоляции: суммарная доля 'своих' среди занятых
    соседей по всем агентам.

      ~0.5 — случайное перемешивание (при равных долях типов),
      1.0   — полная сегрегация.
    """
    agents = grid != EMPTY
    own = np.take_along_axis(counts, np.clip(grid, 0, None)[None], axis=0)[0]
    denom = counts.sum(axis=0)[agents].sum()   # всего занятых соседских связей
    if denom == 0:
        return float("nan")
    return float(own[agents].sum() / denom)


def happiness(sim: "Simulation") -> float:
    """Доля счастливых по определению satisfaction-политики модели.

    Не чистая функция поверх состояния, а свойство конструкции —
    threshold живёт в самой политике. Поэтому принимает Simulation,
    а не (grid, counts, n_types), и в реестр не входит.
    """
    happy = sim.satisfaction(sim.counts(), sim.grid)
    agents = sim.grid != EMPTY
    return float(happy[agents].mean())


def type_fractions(grid: np.ndarray, counts: np.ndarray,
                   n_types: int) -> np.ndarray:
    """Доли типов среди занятых клеток (контроль стационара флипов).

    Без агентов возвращает нули — согласовано с индексом изоляции (nan).
    """
    agents = grid[grid != EMPTY]
    if agents.size == 0:
        return np.zeros(n_types)
    return np.bincount(agents, minlength=n_types) / agents.size


# ------------------------------------------------------------ реестр


def duncan_dissimilarity(grid: np.ndarray, counts: np.ndarray,
                         n_types: int) -> float:
    """Индекс диссимилярности Данкана между типом 0 и типом 1.

    D = (1/2) * sum_x | n_0(x)/N_0 - n_1(x)/N_1 |,

    где в качестве «пространственных единиц» берутся ОКРЕСТНОСТИ клеток
    (n_k(x) — число соседей типа k у клетки x, т.е. готовый counts).
    D = 0 — идентичное распределение типов по окрестностям (смешение),
    D = 1 — полное разделение. Классика: Duncan & Duncan (1955).

    Определён только для K = 2 и непустых популяций обоих типов.
    """
    if n_types != 2:
        raise ValueError("индекс Данкана определён для двух типов")
    n0 = float(counts[0].sum())
    n1 = float(counts[1].sum())
    if n0 == 0 or n1 == 0:
        return float("nan")
    d = np.abs(counts[0] / n0 - counts[1] / n1).sum() / 2.0
    return float(d)


def structure_factor(grid: np.ndarray, counts: np.ndarray,
                     n_types: int) -> tuple[np.ndarray, np.ndarray]:
    """Структурный фактор S(q) поля типов — радиально усреднённый спектр.

    Поле: phi(x) = +1 для типа 1, -1 для типа 0, 0 для пустых (K=2);
    для K > 2 берётся индикатор типа 0 минус его среднее.
    S(q) = |FFT(phi)|^2 / N_agents^2, усреднённая по кольцам |q| = const.

    Рост S(q -> 0) со временем — сигнатура крупномасштабной сегрегации
    (укрупнение доменов); положение пика q* задаёт характерный размер
    домена R ~ 1/q*.

    Возвращает (q, S): центры колец и усреднённый спектр.
    """
    if n_types == 2:
        phi = np.where(grid == EMPTY, 0.0,
                       np.where(grid == 1, 1.0, -1.0)).astype(np.float64)
    else:
        phi = (grid == 0).astype(np.float64)
        phi -= phi[grid != EMPTY].mean()
        phi[grid == EMPTY] = 0.0
    n_agents = max(int((grid != EMPTY).sum()), 1)
    f = np.abs(np.fft.fftn(phi)) ** 2 / n_agents ** 2

    # радиальное усреднение по |q| (в единицах 2*pi / размер)
    q2 = np.zeros(grid.shape)
    for ax, n in enumerate(grid.shape):
        k = np.fft.fftfreq(n) * n          # целые волновые числа
        shape = [1] * grid.ndim
        shape[ax] = n
        q2 += k.reshape(shape) ** 2
    qr = np.round(np.sqrt(q2)).astype(int).ravel()
    maxq = int(qr.max())
    sums = np.bincount(qr, weights=f.ravel(), minlength=maxq + 1)
    nums = np.bincount(qr, minlength=maxq + 1)
    s = sums / np.maximum(nums, 1)
    q = np.arange(maxq + 1) * (2 * np.pi / max(grid.shape))
    return q, s


def cluster_stats(grid: np.ndarray, counts: np.ndarray,
                  n_types: int, connectivity: int = 8) -> dict:
    """Статистика доменов (кластеров одного типа).

    Кластер — связная компонента занятых клеток одного типа
    (4- или 8-связность). Возвращает словарь:
      n_clusters, mean_size, max_size, sizes (массив размеров).
    Характерный размер домена R ~ mean_size^(1/d) растёт как t^alpha
    (для кинетики Кавасаки alpha = 1/3 — Лифшиц–Слёзов).
    """
    from scipy import ndimage

    sizes = []
    structure = ndimage.generate_binary_structure(
        grid.ndim, 2 if connectivity == 8 else 1)
    for k in range(n_types):
        mask = grid == k
        lab, n = ndimage.label(mask, structure=structure)
        if n:
            sizes.append(np.bincount(lab.ravel())[1:])
    if not sizes:
        return {"n_clusters": 0, "mean_size": 0.0, "max_size": 0,
                "sizes": np.array([], dtype=int)}
    all_sizes = np.concatenate(sizes)
    return {"n_clusters": int(all_sizes.size),
            "mean_size": float(all_sizes.mean()),
            "max_size": int(all_sizes.max()),
            "sizes": all_sizes}


OBSERVABLES: dict[str, ObsFn] = {
    "isolation": isolation_index,
    "type_fractions": type_fractions,
    "duncan": duncan_dissimilarity,
    "structure_factor": structure_factor,
    "cluster_stats": cluster_stats,
}


def get(name: str) -> ObsFn:
    if name not in OBSERVABLES:
        raise KeyError(f"неизвестная метрика: {name}. Доступны: {sorted(OBSERVABLES)}")
    return OBSERVABLES[name]