"""
experiments/common.py

Общие помощники для численных экспериментов: сборка быстрой симуляции
из стандартных фабрик и измерение стационарного индекса изоляции.

Единый протокол измерения I:
  - term_steps MCS на термализацию (отбрасываются);
  - meas_steps MCS на измерение, I усредняется по кадрам.
Это соответствует стандартной схеме Монте-Карло по спиновым системам.
"""

from __future__ import annotations

import numpy as np

from core import factories
from core.fast import FastSimulation


def make_fast_2d(size: int, threshold: float, flip_prob: float,
                 density: float = 0.7, radius: int = 1,
                 seed: int | None = None) -> FastSimulation:
    """FastSimulation с параметрами make_schelling_2d (та же фабрика,
    затем оборачивание — разделяем один источник начальных условий)."""
    base = factories.make_schelling_2d(
        size=size, threshold=threshold, flip_prob=flip_prob,
        density=density, radius=radius, seed=seed)
    return FastSimulation(grid=base.grid, offsets=base.offsets,
                          n_types=base.n_types,
                          satisfaction=base.satisfaction,
                          action=base.action, rng=base.rng)


def stationary_isolation(size: int, threshold: float, flip_prob: float,
                         density: float = 0.7, term_steps: int = 400,
                         meas_steps: int = 200,
                         seed: int | None = None) -> float:
    """Один прогон: термализация + средний I по кадрам измерения."""
    sim = make_fast_2d(size, threshold, flip_prob, density, seed=seed)
    sim.run(term_steps)
    vals = np.empty(meas_steps)
    for i in range(meas_steps):
        sim.step()
        vals[i] = sim.observe("isolation")
    return float(np.nanmean(vals))
