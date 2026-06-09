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
    """Monte-Carlo robustness of the weighted winner.

    Each trial randomly perturbs the criterion weights (Gaussian +/-30%) AND the uncertain
    metrics (each by its per-criterion relative uncertainty from WEIGHTS), re-scores the
    architectures, and records which one wins. Returns {architecture: fraction of trials won}."""
    feasible_archs = []
    for arch in rows:
        if arch["feasible"]:
            feasible_archs.append(arch)
    if not feasible_archs:
        return {}

    # Tally of how many trials each architecture won; start every count at 0.
    win_count = {}
    for arch in feasible_archs:
        win_count[arch["name"]] = 0

    for _ in range(trials):
        # Perturb each criterion weight by a Gaussian +/-30% (floored at 0.01).
        perturbed_weights = {}
        for criterion in WEIGHTS:
            base_weight = WEIGHTS[criterion][1]
            noisy_weight = base_weight * (1 + random.gauss(0, 0.30))
            perturbed_weights[criterion] = max(0.01, noisy_weight)

        # Perturb each architecture's metrics by their per-criterion relative uncertainty.
        perturbed_metrics = {}
        for arch in feasible_archs:
            values = metrics(arch)
            for criterion in WEIGHTS:
                uncertainty = WEIGHTS[criterion][2]
                if uncertainty > 0:
                    values[criterion] = values[criterion] * (1 + random.gauss(0, uncertainty))
            perturbed_metrics[arch["name"]] = values

        # Re-score with the perturbed weights and metrics, then tally the winner.
        trial_scores = score_table(perturbed_metrics, perturbed_weights)
        winner = max(trial_scores, key=trial_scores.get)
        win_count[winner] = win_count[winner] + 1

    # Convert win counts to fractions of the total trials.
    win_fraction = {}
    for name in win_count:
        win_fraction[name] = win_count[name] / trials
    return win_fraction


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
    for arch in rows:
        force_at_knee = arch["k"] * arch["dy"]          # reaction at the elastic-limit stroke [N]
        strokes_mm = [0, arch["dy"] * 1000, arch["dmax"] * 1000]
        forces_kn  = [0, force_at_knee / 1000, arch["Fmax"] / 1000]
        style = "-" if arch["feasible"] else "--"
        plt.plot(strokes_mm, forces_kn, style, lw=2, label=arch["name"])
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
    for arch in rows:
        print("%-20s %4s %9.1f %9.0f %7.2f %7.2f %7.1f" %
              (arch["name"], "Y" if arch["feasible"] else "N",
               arch["mass"], arch["SEA"], arch["MSe"], arch["MSr"], arch["npk"]))
    print()

    scores = score(rows)
    win_fraction = sensitivity(rows)
    print("=" * 64)
    print("TRADE-OFF SCORE + SENSITIVITY")
    print("=" * 64)
    if not scores:
        print("  No concept passed screening - relax inputs/bounds.")
    else:
        print("%-20s %8s %12s" % ("concept", "score", "P(rank #1)"))
        print("-" * 42)
        for name in sorted(scores, key=scores.get, reverse=True):
            print("%-20s %8.3f %10.0f%%" % (name, scores[name], win_fraction.get(name, 0) * 100))
        weighted_winner = max(scores, key=scores.get)
        robust_winner = max(win_fraction, key=win_fraction.get)
        print("\n  Weighted winner : %s" % weighted_winner)
        print("  Robust winner   : %s  (%.0f%% of trials)" % (robust_winner, win_fraction[robust_winner] * 100))
    print()
    plot(rows)


if __name__ == "__main__":
    main()
