"""
core/policies.py

Вариативное поведение модели. Каждая политика — маленький объект с одним
методом apply; Simulation вызывает их в фиксированном порядке внутри step().

Конвенции:
  - grid меняется in-place, ничего не возвращается;
  - randomness только через переданный rng (воспроизводимость);
  - EMPTY = -1 обозначает пустую клетку.
"""

from __future__ import annotations

import numpy as np

EMPTY = -1


# удовлетворённость

class ThresholdSatisfaction:
    """
    Агент счастлив, если доля 'своих' среди ЗАНЯТЫХ соседей >= threshold.
    Пустые клетки не участвуют в динамике
    Агент без занятых соседей — счастлив (нет повода для переезда).
    """

    def __init__(self, threshold: float):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold должен быть в [0, 1]")
        self.threshold = float(threshold)

    def __call__(self, counts: np.ndarray, grid: np.ndarray) -> np.ndarray:
        total = counts.sum(axis=0)                    # занятые соседи у каждой клетки
        own = np.zeros(grid.shape, dtype=np.float64)
        occupied = grid != EMPTY
        np.divide(
            np.take_along_axis(counts, np.clip(grid, 0, None)[None], axis=0)[0],
            total,
            out=own, where=(total > 0),
        )
        happy = ~occupied | (own >= self.threshold)
        return happy


#  переезды

class RandomVacancyMove:
    """
    Несчастный агент переезжает в случайную свободную клетку.
    """

    def apply(self, grid: np.ndarray, movers: np.ndarray,
              rng: np.random.Generator) -> None:
        empty = np.flatnonzero(grid == EMPTY)
        if movers.size == 0 or empty.size == 0:
            return
        rng.shuffle(movers)
        rng.shuffle(empty)
        n_move = min(movers.size, empty.size)
        src, dst = movers[:n_move], empty[:n_move]
        types = grid.flat[src].copy()   # копия: при наложении src<->dst
        grid.flat[src] = EMPTY          # потеряли бы агентов
        grid.flat[dst] = types


# действие агента

class FlipOrMove:
    """
    Основная политика

    Каждый агент за шаг выбирает ОДНО действие:
      1) с вероятностью flip_prob меняет тип на случайный другой;
      2) иначе с вероятностью 1 - flip_prob, если несчастен, переезжает.

    Порядок внутри шага: сначала бросок монетки для всех, флипы,
    затем переезды оставшихся несчастных (типы у переезжающих — уже новые).
    """

    def __init__(self, flip_prob: float, n_types: int = 2):
        if not 0.0 <= flip_prob <= 1.0:
            raise ValueError("flip_prob должен быть в [0, 1]")
        if n_types < 2:
            raise ValueError("n_types должен быть >= 2")
        self.flip_prob = float(flip_prob)
        self.n_types = int(n_types)
        self.move = RandomVacancyMove()

    def apply(self, grid: np.ndarray, happy: np.ndarray,
              rng: np.random.Generator) -> None:
        agents = np.flatnonzero(grid != EMPTY)
        if agents.size == 0 or self.flip_prob == 1.0:
            if self.flip_prob == 1.0:
                self._flip(grid, agents, rng)
            return

        do_flip = rng.random(agents.size) < self.flip_prob
        self._flip(grid, agents[do_flip], rng)

        movers = agents[~do_flip]
        movers = movers[~happy.flat[movers]]     # несчастные из оставшихся
        self.move.apply(grid, movers, rng)

    def _flip(self, grid: np.ndarray, flippers: np.ndarray,
              rng: np.random.Generator) -> None:
        if flippers.size == 0:
            return
        cur = grid.flat[flippers].astype(np.int64)
        shift = rng.integers(1, self.n_types, size=flippers.size)
        grid.flat[flippers] = ((cur + shift) % self.n_types).astype(np.int8)