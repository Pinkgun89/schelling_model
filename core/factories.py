"""
core/factories.py

Сборка готовых моделей
"""

from __future__ import annotations

import numpy as np

from .policies import EMPTY, FlipOrMove, ThresholdSatisfaction
from .simulation import Simulation
from . import topologies



"""
Проверка на условие: Окно соседей не должно пересекать себя

args:
    offset - массив формы, список целочисленных смещений от центральной ячейки 
    shape  - размер системы по каждому измерению 
"""
def _check_offsets(offsets: np.ndarray, shape: tuple[int, ...]) -> None:

    size = np.array(shape)
    folded = offsets % size
    flat = np.ravel_multi_index(folded.T, shape)
    if np.any(flat == 0) or np.unique(flat).size != flat.size:
        raise ValueError(
            "окно соседей пересекает себя"
        )


"""
Равномерная случайная расстановка при заданной плотности занятости.

args: 
    density - доля занятых клеток [0, 1] 
    n_types - количество типов [0, 127]

"""

def _random_grid(shape: tuple[int, ...], density: float, n_types: int,
                 rng: np.random.Generator) -> np.ndarray:
    
    grid = np.full(shape, EMPTY, dtype=np.int8)                 # заполняем поле пустыми клетками
    n_agents = int(round(density * grid.size))                  # считаем количетсво зантых клеток 
    cells = rng.permutation(grid.size)[:n_agents]               # выдаёт случайную перестановку индексов от 0 до n-1
    grid.flat[cells] = rng.integers(0, n_types, size=n_agents)  # выбранным клетком присваиваются случайные типы в [0, n_types]
    return grid


"""
модель на торе
args: 
    size        - 
    threshold   - агент доволен если по крайней мере threshold соседей свои 
    flip_prob   - вероятность смены типа
    density     - процент занятых клеток 
    radius      - размер окна 
    metric      - выбранная метрика 
    n_types     - количество типов 


"""


def make_schelling_2d(size: int, threshold: float = 1.0 / 3.0,
                      flip_prob: float = 0.0, density: float = 0.7,
                      radius: int = 1, metric: str = "moore",
                      n_types: int = 2, seed: int | None = None) -> Simulation:
    
    rng = np.random.default_rng(seed)
    offsets = topologies.torus_offsets(radius, metric)
    _check_offsets(offsets, (size, size)) 
    return Simulation(
        grid=_random_grid((size, size), density, n_types, rng),
        offsets=offsets,
        n_types=n_types,
        satisfaction=ThresholdSatisfaction(threshold),
        action=FlipOrMove(flip_prob, n_types),
        rng=rng,
    )

"""Одномерная модель на кольце """


def make_schelling_1d(length: int, threshold: float = 1.0 / 3.0,
                      flip_prob: float = 0.0, density: float = 0.7,
                      radius: int = 2, n_types: int = 2,
                      seed: int | None = None) -> Simulation:
    rng = np.random.default_rng(seed)
    offsets = topologies.ring_offsets(radius)
    _check_offsets(offsets, (length,))
    return Simulation(
        grid=_random_grid((length,), density, n_types, rng),
        offsets=offsets,
        n_types=n_types,
        satisfaction=ThresholdSatisfaction(threshold),
        action=FlipOrMove(flip_prob, n_types),
        rng=rng,
    )  