"""
experiments/growth.py

Кинетика укрупнения доменов: характерный размер R(t) из двух независимых
измерений —
  1. средний размер кластера (cluster_stats), R_c = sqrt(mean_size);
  2. пик структурного фактора S(q): R_s = 2*pi / q*.
Ожидание для кинетики с сохранением порядкового параметра (Кавасаки):
R ~ t^alpha, alpha = 1/3 (Лифшиц–Слёзов). Флипы (нелокальная смена
типа, сохраняющая доли) кластеров почти не трогают.

Выход:
  figures/growth.png   — R(t) в лог-лог осях + степенная подгонка
  results/growth.npz

Запуск:  python -m experiments.growth --size 128 --steps 400
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.common import make_fast_2d

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--T", type=float, default=0.35)
    ap.add_argument("--p", type=float, default=0.0)
    ap.add_argument("--density", type=float, default=0.7)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--every", type=int, default=4, help="кадр каждые N MCS")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sim = make_fast_2d(args.size, args.T, args.p, args.density,
                       seed=args.seed)

    ts, R_c, R_s = [], [], []
    n_frames = args.steps // args.every + 1
    for frame in range(n_frames):
        if frame:
            sim.run(args.every)
        t = frame * args.every + 1
        cs = sim.observe("cluster_stats")
        R_c.append(np.sqrt(max(cs["mean_size"], 1e-12)))
        q, s = sim.observe("structure_factor")
        s = s.copy()
        s[0] = 0.0                            # q=0 — константа, не несёт R
        qstar = q[1:][np.argmax(s[1:])]
        R_s.append(2 * np.pi / qstar if qstar > 0 else np.nan)
        ts.append(t)

    ts, R_c, R_s = map(np.array, (ts, R_c, R_s))

    # степенная подгонка R ~ t^alpha по логарифмам (поздняя стадия)
    def fit_alpha(R):
        mask = np.isfinite(R) & (R > 0) & (ts >= 8)
        a, _ = np.polyfit(np.log(ts[mask]), np.log(R[mask]), 1)
        return float(a)

    a_c, a_s = fit_alpha(R_c), fit_alpha(R_s)

    (ROOT / "results").mkdir(exist_ok=True)
    np.savez(ROOT / "results" / "growth.npz",
             t=ts, R_cluster=R_c, R_struct=R_s,
             alpha_cluster=a_c, alpha_struct=a_s, args=vars(args))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (ROOT / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(ts, R_c, "o-", ms=3, label=f"кластеры, α = {a_c:.3f}")
    ax.loglog(ts, R_s, "s-", ms=3, label=f"пик S(q), α = {a_s:.3f}")
    ax.loglog(ts, 0.9 * ts ** (1 / 3), "k:", lw=1, label="t^(1/3) (Лифшиц–Слёзов)")
    ax.set_xlabel("t (MCS)")
    ax.set_ylabel("характерный размер домена R")
    ax.set_title(f"Укрупнение доменов, {args.size}x{args.size}, "
                 f"T = {args.T}, p = {args.p:g}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "growth.png", dpi=150)
    print(f"готово: figures/growth.png  (α_cluster={a_c:.3f}, "
          f"α_struct={a_s:.3f})")


if __name__ == "__main__":
    main()
