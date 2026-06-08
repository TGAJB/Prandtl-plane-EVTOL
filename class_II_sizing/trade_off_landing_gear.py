"""
trade_off_landing_gear.py
Offline landing-gear ARCHITECTURE trade study (was landing_gear.py sections 5-7).

The MTOW convergence loop sizes the gear and picks the weighted winner deterministically
inside class_II_sizing.mass_components.landing_gear_mass. This script is the human-facing
study around that same sizing: it sizes all five architectures at a representative landing
mass, prints the sizing + weighted-score tables, runs a Monte-Carlo sensitivity to find
the ROBUST winner (perturbing the weights and the uncertain metrics), names the weighted
and robust winners, and writes the whole-gear load-deflection (F-delta) plot.

Sizing physics, the concept models, scipy-SLSQP `size`, and the deterministic weighted
`score` all live in mass_components; the constants live in parameters. Nothing here is
called during MTOW convergence.

Run:  python class_II_sizing/trade_off_landing_gear.py     (needs numpy, scipy, matplotlib)
"""

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from class_II_sizing.mass_components import (
    effective_mass, v_limit, v_reserve, E_limit, E_reserve,
    size_all_architectures, metrics, score, score_table,
)
from parameters import G, H_L, D_EST, LIFT, WEIGHTS, GEAR_OPT_SEED

# Representative FULL landing mass [kg] at which the trade study sizes the gear. Set to the
# converged MTOW from mtow_sizing; the gear self-weight is only ~2% of MTOW.
MTOW_STUDY = 2198.58


# =====================================================================
#  Monte-Carlo sensitivity: how robust is the weighted winner?
# =====================================================================
def sensitivity(rows, trials=4000):
    """Perturb the criterion weights (+/-30%) and the uncertain metrics (per-criterion
    relative uncertainty in WEIGHTS) and count how often each architecture wins."""
    feas = [r for r in rows if r["feasible"]]
    if not feas:
        return {}
    wins = {r["name"]: 0 for r in feas}
    for _ in range(trials):
        w = {c: max(0.01, WEIGHTS[c][1] * (1 + random.gauss(0, 0.3))) for c in WEIGHTS}
        data = {}
        for r in feas:
            mv = metrics(r)
            for c, (_, _, u) in WEIGHTS.items():
                if u > 0:
                    mv[c] = mv[c] * (1 + random.gauss(0, u))
            data[r["name"]] = mv
        sc = score_table(data, w)
        wins[max(sc, key=sc.get)] += 1
    return {n: wins[n] / trials for n in wins}


# =====================================================================
#  Whole-gear load-deflection (F-delta) plot
# =====================================================================
def plot(rows, filename="fd_curves.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib not available - skipping plot)")
        return
    plt.figure(figsize=(9, 6))
    for r in rows:
        Fy = r["k"] * r["dy"]
        xs = [0, r["dy"] * 1000, r["dmax"] * 1000]
        ys = [0, Fy / 1000, r["Fmax"] / 1000]
        style = "-" if r["feasible"] else "--"
        plt.plot(xs, ys, style, lw=2, label=r["name"])
    plt.xlabel("stroke  delta  [mm]")
    plt.ylabel("total reaction  F  [kN]")
    plt.title("Whole-gear load-deflection (area under curve = energy absorbed)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filename, dpi=140)
    print("  F-delta plot written ->", filename)


# =====================================================================
#  Driver
# =====================================================================
def main():
    random.seed(GEAR_OPT_SEED)
    m_eff = effective_mass(MTOW_STUDY, H_L, D_EST, LIFT)

    print("=" * 64)
    print("DROP CONDITIONS (whole-gear basis)")
    print("=" * 64)
    print("  MTOW  = %.0f kg   M_eff = %.0f kg   (4 hinges across 2 skids)"
          % (MTOW_STUDY, m_eff))
    print("  v_L   = %.2f m/s   v_R = %.2f m/s" % (v_limit(), v_reserve()))
    print("  E_L   = %.0f J     E_R = %.0f J  (= 1.5 E_L)"
          % (E_limit(0, m_eff), E_reserve(0, m_eff)))
    print()

    rows = size_all_architectures(m_eff)

    print("=" * 64)
    print("SIZING  (scipy SLSQP: lightest whole gear meeting both drops)")
    print("=" * 64)
    print("%-20s %4s %9s %9s %7s %7s %7s" %
          ("concept", "feas", "mass[kg]", "SEA", "MS_e", "MS_R", "n_pk"))
    print("-" * 64)
    for r in rows:
        print("%-20s %4s %9.1f %9.0f %7.2f %7.2f %7.1f" %
              (r["name"], "Y" if r["feasible"] else "N",
               r["mass"], r["SEA"], r["MSe"], r["MSr"], r["npk"]))
    print()

    sc = score(rows)
    pw = sensitivity(rows)
    print("=" * 64)
    print("TRADE-OFF SCORE + SENSITIVITY")
    print("=" * 64)
    if not sc:
        print("  No concept passed screening - relax inputs/bounds.")
    else:
        print("%-20s %8s %12s" % ("concept", "score", "P(rank #1)"))
        print("-" * 42)
        for n in sorted(sc, key=sc.get, reverse=True):
            print("%-20s %8.3f %10.0f%%" % (n, sc[n], pw.get(n, 0) * 100))
        win = max(sc, key=sc.get)
        rob = max(pw, key=pw.get)
        print("\n  Weighted winner : %s" % win)
        print("  Robust winner   : %s  (%.0f%% of trials)" % (rob, pw[rob] * 100))
    print()
    plot(rows)


if __name__ == "__main__":
    main()
