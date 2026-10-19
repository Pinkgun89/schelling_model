"""
experiments/benchmark.py

Сравнение трёх движков на одинаковой задаче:
  Simulation      — полный пересчёт counts каждый MCS, O(z*K*N);
  FastSimulation  — инкрементальный counts, O(m*z) на шаг;
  KMCSimulation   — события Гиллеспи, O(z) на событие (время в MCS-экв.).

Печатает время на MCS и ускорение относительно базового движка.
Запуск:  python -m experiments.benchmark --size 200 --steps 50
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from core import factories, topologies
from core.fast import FastSimulation
from core.kmc import KMCSimulation
from experiments.common import make_fast_2d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--T", type=float, default=1 / 3)
    ap.add_argument("--p", type=float, default=0.01)
    args = ap.parse_args()

    # 1. базовый движок
    sim = factories.make_schelling_2d(size=args.size, threshold=args.T,
                                      flip_prob=args.p, seed=0)
    t0 = time.perf_counter()
    sim.run(args.steps)
    t_base = time.perf_counter() - t0

    # 2. быстрый движок (тот же seed -> та же траектория, честное сравнение)
    fast = make_fast_2d(args.size, args.T, args.p, seed=0)
    t0 = time.perf_counter()
    fast.run(args.steps)
    t_fast = time.perf_counter() - t0

    # 3. KMC: гоняем до того же физического времени (steps MCS)
    kmc = KMCSimulation(
        grid=factories._random_grid((args.size, args.size), 0.7, 2,
                                    np.random.default_rng(0)),
        offsets=topologies.torus_offsets(1, "moore"),
        n_types=2, threshold=args.T, flip_prob=args.p,
        rng=np.random.default_rng(1))
    t0 = time.perf_counter()
    kmc.run_until(float(args.steps))
    t_kmc = time.perf_counter() - t0

    N = args.size ** 2
    print(f"система {args.size}x{args.size} (N = {N}), {args.steps} MCS")
    print("== активная фаза (старт из случайной расстановки) ==")
    print(f"Simulation      : {t_base:7.3f} c  ({1e3*t_base/args.steps:7.3f} мс/MCS)")
    print(f"FastSimulation  : {t_fast:7.3f} c  ({1e3*t_fast/args.steps:7.3f} мс/MCS)"
          f"  ускорение x{t_base/t_fast:.1f}")
    print(f"KMCSimulation   : {t_kmc:7.3f} c  ({1e3*t_kmc/args.steps:7.3f} мс/MCS-экв.,"
          f" {kmc.n_events} событий)")

    # контроль идентичности Simulation vs FastSimulation
    same = np.array_equal(sim.grid, fast.grid)
    print(f"траектории Simulation и FastSimulation совпадают: {same}")

    # --------------------------------------------------- замороженная фаза
    # докручиваем обе системы до поздней стадии и меряем там: событий
    # мало, и асимптотика движков проявляется полностью
    print("== замороженная фаза (после 1000 MCS прогрева) ==")
    sim.run(1000)
    fast.run(1000)
    kmc.run_until(kmc.t + 1000)

    t0 = time.perf_counter()
    sim.run(args.steps)
    t_base2 = time.perf_counter() - t0
    t0 = time.perf_counter()
    fast.run(args.steps)
    t_fast2 = time.perf_counter() - t0
    t0 = time.perf_counter()
    kmc.run_until(kmc.t + args.steps)
    t_kmc2 = time.perf_counter() - t0

    print(f"Simulation      : {1e3*t_base2/args.steps:7.3f} мс/MCS")
    print(f"FastSimulation  : {1e3*t_fast2/args.steps:7.3f} мс/MCS"
          f"  ускорение x{t_base2/t_fast2:.1f}")
    print(f"KMCSimulation   : {1e3*t_kmc2/args.steps:7.3f} мс/MCS-экв."
          f"  ускорение x{t_base2/t_kmc2:.1f}")


if __name__ == "__main__":
    main()
