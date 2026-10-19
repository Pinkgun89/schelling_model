"""
core/simulation.py

Скелет симуляции: состояние решётки + фиксированный порядок шага.
Вся вариативность (геометрия, поведение) живёт снаружи и подставляется
через offsets и политики — этот класс про неё ничего не знает.

Шаг (MCS):
  1. по состоянию на t: маска счастья (satisfaction)
  2. действия агентов: политика решает по состоянию на t, применяет сразу
     (флипы -> переезды несчастных, не выбравших флип)
Все решения принимаются по состоянию на начало шага — без пересчёта.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .policies import EMPTY  # единая конвенция пустой клетки


@dataclass
class Simulation:
    grid: np.ndarray                                  # int8: EMPTY или 0..K-1
    offsets: np.ndarray                               # (n_offsets, dim)
    n_types: int                                      # K, задаётся явно
    satisfaction: Callable[[np.ndarray, np.ndarray], np.ndarray]  # (counts, grid) -> happy
    action: object                                    # .apply(grid, happy, rng)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng())

    # ------------------------------------------------------------ соседи

    def counts(self) -> np.ndarray:
        """counts[k, *shape] — число соседей типа k у каждой клетки.

        Геометрия живёт здесь и только здесь: каждое смещение — np.roll
        (периодические границы — тор/кольцо — бесплатно).
        """
        counts = np.zeros((self.n_types,) + self.grid.shape, dtype=np.int16)
        for off in self.offsets:
            shifted = np.roll(self.grid, shift=tuple(off),
                              axis=tuple(range(self.grid.ndim)))
            for k in range(self.n_types):
                counts[k] += (shifted == k)
        return counts

    # ------------------------------------------------------------ динамика

    def step(self) -> None:
        """Один MCS."""
        happy = self.satisfaction(self.counts(), self.grid)
        self.action.apply(self.grid, happy, self.rng)

    def run(self, n_steps: int) -> None:
        for _ in range(n_steps):
            self.step()

    # ------------------------------------------------------------ наблюдение

    def observe(self, name: str) -> float | np.ndarray:
        """Диспетчер метрик (читает состояние, не меняет его)."""
        from . import observables  # ленивый импорт, файл следующий
        fn = observables.get(name)
        return fn(self.grid, self.counts(), self.n_types)

    def happiness(self) -> float:
        """Доля счастливых агентов (зависит от satisfaction-политики)."""
        from . import observables
        return observables.happiness(self)