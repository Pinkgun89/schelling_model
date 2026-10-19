"""
experiments/phase_diagram.py

Фазовая диаграмма I(T, p): средний стационарный индекс изоляции на
сетке параметров (порог T, вероятность флипа p). Прогоны по сидам
независимы — тривиальный параллелизм через multiprocessing.

Выход:
  figures/phase_diagram.png  — тепловая карта I(T, p)
  results/phase_diagram.npz  — сырые данные (T, p, I, статистика)

Запуск из корня проекта:
    python -m experiments.phase_diagram --size 64 --seeds 4
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from experiments.common import stationary_isolation

ROOT = Path(__file__).resolve().parent.parent


def _job(task):
    size, T, p, density, term, meas, seed = task
    return stationary_isolation(size, T, p, density, term, meas, seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--density", type=float, default=0.7)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--term", type=int, default=400, help="MCS термализации")
    ap.add_argument("--meas", type=int, default=200, help="MCS измерения")
    ap.add_argument("--procs", type=int, default=0,
                    help="число процессов (0 = все ядра)")
    args = ap.parse_args()

    Ts = np.linspace(0.20, 0.60, 9)          # порог терпимости
    ps = np.array([0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3])

    tasks = [(args.size, T, p, args.density, args.term, args.meas, seed)
             for T in Ts for p in ps for seed in range(args.seeds)]

    t0 = time.time()
    with Pool(processes=args.procs or None) as pool:
        out = pool.map(_job, tasks)
    dt = time.time() - t0

    out = np.array(out).reshape(len(Ts), len(ps), args.seeds)
    I_mean, I_std = out.mean(axis=-1), out.std(axis=-1)

    (ROOT / "results").mkdir(exist_ok=True)
    np.savez(ROOT / "results" / "phase_diagram.npz",
             T=Ts, p=ps, I_mean=I_mean, I_std=I_std, args=vars(args))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (ROOT / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    extent = [np.log10(max(ps[1], 1e-4)), 0, Ts[0], Ts[-1]]
    lp = np.where(ps > 0, np.log10(ps), np.log10(ps[1]) - 1)
    im = ax.imshow(I_mean, aspect="auto", origin="lower", cmap="viridis",
                   extent=[lp[0], lp[-1], Ts[0], Ts[-1]], vmin=0.5, vmax=1.0)
    ax.set_xticks(lp)
    ax.set_xticklabels([f"{p:g}" for p in ps])
    ax.set_xlabel("p (вероятность флипа)")
    ax.set_ylabel("T (порог терпимости)")
    ax.set_title(f"Стационарный I(T, p), {args.size}x{args.size}, "
                 f"{args.seeds} сид(а)")
    fig.colorbar(im, label="индекс изоляции I")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "phase_diagram.png", dpi=150)
    print(f"готово за {dt:.1f} c; figures/phase_diagram.png, "
          f"results/phase_diagram.npz")


if __name__ == "__main__":
    main()
