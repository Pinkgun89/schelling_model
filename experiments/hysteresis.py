"""
experiments/hysteresis.py

Проверка метастабильности: скан порога T вверх и вниз С СОХРАНЕНИЕМ
состояния (каждая следующая точка стартует из конца предыдущей —
протокол «отжиг/закалка»). Петля гистерезиса в I(T) — признак перехода
первого рода / метастабильной сегрегации (гипотеза De Marco et al., 2023).

Выход:
  figures/hysteresis.png
  results/hysteresis.npz

Запуск:  python -m experiments.hysteresis --size 64 --p 0.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.common import make_fast_2d

ROOT = Path(__file__).resolve().parent.parent


def scan(sim, Ts: np.ndarray, equil: int, meas: int) -> np.ndarray:
    """Прогон по последовательности T без перезапуска; средний I в точке."""
    vals = np.empty(Ts.size)
    for i, T in enumerate(Ts):
        sim.satisfaction.threshold = float(T)
        sim.run(equil)
        acc = np.empty(meas)
        for j in range(meas):
            sim.step()
            acc[j] = sim.observe("isolation")
        vals[i] = np.nanmean(acc)
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--p", type=float, default=0.0)
    ap.add_argument("--density", type=float, default=0.7)
    ap.add_argument("--equil", type=int, default=300)
    ap.add_argument("--meas", type=int, default=100)
    ap.add_argument("--nT", type=int, default=17)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    T_up = np.linspace(0.20, 0.70, args.nT)
    T_dn = T_up[::-1]

    sim_up = make_fast_2d(args.size, T_up[0], args.p, args.density,
                          seed=args.seed)
    I_up = scan(sim_up, T_up, args.equil, args.meas)

    # ветвь вниз — из независимого старта при максимальном T
    sim_dn = make_fast_2d(args.size, T_dn[0], args.p, args.density,
                          seed=args.seed + 1)
    I_dn = scan(sim_dn, T_dn, args.equil, args.meas)

    (ROOT / "results").mkdir(exist_ok=True)
    np.savez(ROOT / "results" / "hysteresis.npz",
             T_up=T_up, I_up=I_up, T_dn=T_dn, I_dn=I_dn, args=vars(args))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (ROOT / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(T_up, I_up, "o-", label="T вверх (из перемешанного)")
    ax.plot(T_dn, I_dn, "s--", label="T вниз (из T = 0.7)")
    ax.set_xlabel("T (порог терпимости)")
    ax.set_ylabel("индекс изоляции I")
    ax.set_title(f"Гистерезис I(T), {args.size}x{args.size}, p = {args.p:g}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "hysteresis.png", dpi=150)
    print("готово: figures/hysteresis.png, results/hysteresis.npz")


if __name__ == "__main__":
    main()
