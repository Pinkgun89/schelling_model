"""
core/fast.py

Быстрый движок: та же модель, что и core/simulation.py, но с
ИНКРЕМЕНТАЛЬНЫМ обновлением counts.

Идея
----
Полный пересчёт counts стоит O(z * K * N) на шаг (z смещений, K типов,
N клеток) и не зависит от того, сколько агентов реально действовало.
Но за один MCS меняются только клетки, где был флип или переезд (m штук),
а counts[k, x] меняется лишь для x из окрестностей изменённых клеток —
это O(m * z) операций. В поздней (почти замороженной) фазе m << N, и
ускорение достигает десятков раз; даже в активной фазе выигрыш ~ z*K/2.

Совместимость
-------------
FastSimulation использует ТЕ ЖЕ объекты satisfaction и action и тот же
порядок обращений к rng, что и Simulation. Поэтому при одинаковом seed
траектории Simulation и FastSimulation совпадают БИТ-В-БИТ — это
используется как тест корректности (tests/test_fast.py).

Ограничение: политика должна менять grid только через переданный rng и
не зависеть от counts внутри apply (все наши политики таковы).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .policies import EMPTY
from .simulation import Simulation


@dataclass
class FastSimulation(Simulation):
    """Simulation с инкрементальным counts и кешем маски счастья."""

    def __post_init__(self):
        self._counts = self._full_counts()          # кеш counts
        self._shape = self.grid.shape
        self._offsets = np.asarray(self.offsets)    # (z, dim)

    # ------------------------------------------------------- counts

    def _full_counts(self) -> np.ndarray:
        """Эталонный полный пересчёт (как в Simulation.counts)."""
        counts = np.zeros((self.n_types,) + self.grid.shape, dtype=np.int32)
        for off in self.offsets:
            shifted = np.roll(self.grid, shift=tuple(off),
                              axis=tuple(range(self.grid.ndim)))
            for k in range(self.n_types):
                counts[k] += (shifted == k)
        return counts

    def counts(self) -> np.ndarray:
        """Возвращает кешированный counts (O(1) вне шага)."""
        return self._counts

    def _update_counts(self, old_vals: np.ndarray, new_vals: np.ndarray,
                       cells_multi: np.ndarray) -> None:
        """Локальное обновление counts вокруг изменённых клеток.

        old_vals/new_vals — (m,) типы до/после; cells_multi — (m, dim)
        мультииндексы изменённых клеток. Для каждой изменённой клетки c и
        каждого смещения off клетка x = c + off (mod размер) потеряла
        соседа типа old и приобрела соседа типа new.
        """
        m = cells_multi.shape[0]
        if m == 0:
            return
        shape = np.array(self._shape)
        # (m, 1, dim) + (1, z, dim) -> (m, z, dim) по модулю размера
        nbr = (cells_multi[:, None, :] + self._offsets[None, :, :]) % shape
        nbr_flat = np.ravel_multi_index(
            nbr.reshape(-1, self.grid.ndim).T, self._shape)          # (m*z,)
        old_rep = np.repeat(old_vals, self._offsets.shape[0])
        new_rep = np.repeat(new_vals, self._offsets.shape[0])

        counts_flat = self._counts.reshape(self.n_types, -1)
        n_cells = counts_flat.shape[1]
        # объединяем «−1 за старый тип» и «+1 за новый» в один bincount —
        # на порядок быстрее np.add.at при массовых изменениях
        idx_parts, w_parts = [], []
        for val, delta in ((old_rep, -1), (new_rep, +1)):
            mask = val != EMPTY
            idx_parts.append(val[mask] * n_cells + nbr_flat[mask])
            w_parts.append(np.full(mask.sum(), delta, dtype=np.int32))
        if idx_parts:
            idx = np.concatenate(idx_parts)
            w = np.concatenate(w_parts)
            bc = np.bincount(idx, weights=w,
                             minlength=self.n_types * n_cells)
            counts_flat += bc.reshape(self.n_types, n_cells).astype(np.int32)

    # ------------------------------------------------------- шаг

    def step(self) -> None:
        """Один MCS: те же действия, что в Simulation, но counts — диффом."""
        happy = self.satisfaction(self._counts, self.grid)
        before = self.grid.copy()                      # дешёвый memcpy O(N)
        self.action.apply(self.grid, happy, self.rng)

        changed = np.flatnonzero(self.grid != before)  # (m,)
        if changed.size == 0:
            return
        cells_multi = np.unravel_index(changed, self._shape)
        cells_multi = np.stack(cells_multi, axis=1)    # (m, dim)
        self._update_counts(before.flat[changed].astype(np.int64),
                            self.grid.flat[changed].astype(np.int64),
                            cells_multi)
