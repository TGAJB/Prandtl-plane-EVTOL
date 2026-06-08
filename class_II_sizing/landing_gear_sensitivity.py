"""
landing_gear_sensitivity.py
Sensitivity / robustness study of the landing-gear architecture trade study in
class_II_sizing.mass_components.

The gear sizer compares five energy-absorber architectures (cross-tube, metal leaf,
composite leaf, crushable hybrid, elastomeric) and selects one by the weighted
trade-off score. This module asks the harder question:

    Is the weighted winner the right choice ONLY at the nominal inputs, or across the
    whole plausible range of the uncertain drop / absorber assumptions?

Three complementary, well-established methods are used (most robust last):

  1. TORNADO (local one-at-a-time) - quick, readable ranking of which assumption moves the
     selected gear mass most when swept low->high with all others held nominal.

  2. SOBOL indices (variance-based GLOBAL sensitivity, Saltelli/Jansen estimators) -
     decomposes the variance of the chosen design's mass into the first-order and
     total-order contribution of each input. Sampling uses a scrambled Sobol' sequence.

  3. MONTE CARLO selection robustness (Latin Hypercube) - samples the full joint
     uncertainty space and records, for every sample, each architecture's mass and which
     one the sizer selects, showing HOW OFTEN the nominal winner is chosen.

The sizing random restarts are seeded per evaluation, so a given input vector maps to a
deterministic mass (parameter effects are isolated from restart noise). Because each
sample now runs the full scipy-SLSQP sizer (not a closed form), the sample counts are
kept modest.

Run:  python class_II_sizing/landing_gear_sensitivity.py
Output: PNG figures + a CSV summary in  class_II_sizing/sensitivity_plots/
"""

import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: save figures, do not open a window
import matplotlib.pyplot as plt
from scipy.stats import qmc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import class_II_sizing.mass_components as mc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "sensitivity_plots"

# Representative FULL landing mass [kg] at which the gear is sized. The gear self-weight is
# ~1-2% of MTOW, so sizing every sample at one fixed landing mass isolates the parameter
# effects cleanly (and is far cheaper than re-running the MTOW loop per sample). Set to the
# converged MTOW from mtow_sizing.
M_LANDING_KG = 2136.0

N_SOBOL = 32    # Sobol base sample size (total evals = N_SOBOL*(D+2)); power of 2 for balance
N_MC    = 150   # Latin-Hypercube Monte Carlo samples
SEED    = 12345

# Architecture names (match mass_components GEAR_OPT_BOUNDS / QUAL) and their reusability.
ARCHS = ["A cross-tube", "B metal leaf", "B' composite leaf",
         "F crushable hybrid", "E elastomeric"]
ARCH_COLOR = {
    "A cross-tube":       "#9e9e9e",
    "B metal leaf":       "#8d6e63",
    "B' composite leaf":  "#1e88e5",
    "F crushable hybrid": "#fb8c00",
    "E elastomeric":      "#8e24aa",
}
REUSABLE_AT_LIMIT = {"A cross-tube": False, "B metal leaf": True, "B' composite leaf": False,
                     "F crushable hybrid": False, "E elastomeric": True}

# Uncertain inputs: (mass_components attribute, low, high, nominal, label, unit).
# Scalar module globals of the new drop model; ranges are plausible engineering bounds.
PARAMS = [
    ("H_L",               0.20,    0.35,    0.25,    "limit drop height",     "m"),
    ("D_EST",             0.08,    0.25,    0.15,    "impact deflection",     "m"),
    ("LIFT",              0.00,    0.50,    0.00,    "rotor-lift credit",     "-"),
    ("N_LIMIT",           12.0,    25.0,    20.0,    "peak-decel cap",        "g"),
    ("ENVELOPE",          0.30,    0.50,    0.40,    "stroke envelope",       "m"),
    ("HONEYCOMB_STRESS",  1.5e6,   4.0e6,   2.5e6,   "honeycomb crush stress","Pa"),
    ("HONEYCOMB_DENSITY", 50.0,    120.0,   80.0,    "honeycomb density",     "kg/m^3"),
    ("CRUSH_STROKE_EFF",  0.60,    0.85,    0.75,    "crush usable fraction", "-"),
    ("ELASTO_FIXED_MASS", 0.8,     1.8,     1.2,     "elastomer fixed mass",  "kg"),
    ("ELASTO_MASS_PER_N", 6.0e-5,  1.5e-4,  1.0e-4,  "elastomer mass/load",   "kg/N"),
    ("SKID_RAIL_MASS",    10.0,    22.0,    15.0,    "skid rail mass",        "kg"),
]
PNAMES  = [p[0] for p in PARAMS]
PLABEL  = {p[0]: p[4] for p in PARAMS}
LO      = np.array([p[1] for p in PARAMS])
HI      = np.array([p[2] for p in PARAMS])
NOMINAL = {p[0]: p[3] for p in PARAMS}
D = len(PARAMS)


# ---------------------------------------------------------------------------
# Core evaluation: size every architecture at M_LANDING_KG for a given parameter set
# ---------------------------------------------------------------------------
def _apply(overrides):
    """Set the mass_components module globals from `overrides`; return the saved originals."""
    saved = {}
    for k, v in overrides.items():
        saved[k] = getattr(mc, k)
        setattr(mc, k, v)
    return saved


def _restore(saved):
    for k, v in saved.items():
        setattr(mc, k, v)


def evaluate(overrides):
    """Size all architectures at M_LANDING_KG with the given parameter overrides.

    Returns (masses, selected_arch, selected_mass) where masses maps arch -> mass [kg]
    (np.nan if that architecture is infeasible). The selected architecture is the
    deterministic weighted-score winner. Restarts are seeded so the output is repeatable."""
    saved = _apply(overrides)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            random.seed(SEED)                          # isolate parameter effect from restart noise
            m_eff = mc.effective_mass(M_LANDING_KG, mc.H_L, mc.D_EST, mc.LIFT)
            rows = mc.size_all_architectures(m_eff)
            sc = mc.score(rows)
        masses = {r["name"]: (r["mass"] if r["feasible"] else np.nan) for r in rows}
        if sc:
            sel = max(sc, key=sc.get)
            sel_mass = next(r["mass"] for r in rows if r["name"] == sel)
        else:
            sel, sel_mass = None, np.nan
        return masses, sel, sel_mass
    finally:
        _restore(saved)


def selected_mass(overrides):
    """Mass [kg] of the CHOSEN design for variance-based SA. A finite fallback keeps the
    output defined on the rare sample where no architecture is feasible."""
    _, _, sel_mass = evaluate(overrides)
    return sel_mass if np.isfinite(sel_mass) else 0.05 * M_LANDING_KG


def _overrides_from_row(row):
    return {PNAMES[j]: float(row[j]) for j in range(D)}


# ---------------------------------------------------------------------------
# Method 1 - Tornado (local one-at-a-time)
# ---------------------------------------------------------------------------
def tornado():
    _, _, base = evaluate(NOMINAL)
    rows = []
    for name, lo, hi, nom, label, unit in PARAMS:
        o = dict(NOMINAL); o[name] = lo
        _, _, m_lo = evaluate(o)
        o = dict(NOMINAL); o[name] = hi
        _, _, m_hi = evaluate(o)
        rows.append((label, m_lo - base, m_hi - base, abs(m_hi - m_lo)))
    rows.sort(key=lambda r: r[3])  # ascending swing -> largest at top of barh
    return base, rows


def plot_tornado(base, rows, path):
    labels = [r[0] for r in rows]
    lo = [r[1] for r in rows]
    hi = [r[2] for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(y, lo, color="#ef5350", label="input at low end")
    ax.barh(y, hi, color="#42a5f5", label="input at high end")
    ax.axvline(0.0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("change in SELECTED gear mass vs nominal [kg]")
    ax.set_title(f"Tornado: local sensitivity of the selected gear mass\n"
                 f"(nominal selected mass = {base:.1f} kg)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Method 2 - Sobol variance-based global sensitivity (Saltelli / Jansen)
# ---------------------------------------------------------------------------
def sobol():
    sampler = qmc.Sobol(d=2 * D, scramble=True, seed=SEED)
    base = sampler.random(N_SOBOL)              # (N, 2D) in [0,1]
    A01, B01 = base[:, :D], base[:, D:]
    A = A01 * (HI - LO) + LO
    B = B01 * (HI - LO) + LO

    def evalmat(M):
        return np.array([selected_mass(_overrides_from_row(M[i])) for i in range(M.shape[0])])

    fA, fB = evalmat(A), evalmat(B)
    var = np.var(np.concatenate([fA, fB]), ddof=1)
    S = np.zeros(D); ST = np.zeros(D)
    for i in range(D):
        ABi = A.copy(); ABi[:, i] = B[:, i]
        fABi = evalmat(ABi)
        S[i]  = np.mean(fB * (fABi - fA)) / var           # Saltelli 2010 first-order
        ST[i] = 0.5 * np.mean((fA - fABi) ** 2) / var      # Jansen total-order
    return S, ST, var


def plot_sobol(S, ST, var, path):
    order = np.argsort(ST)
    labels = [PLABEL[PNAMES[i]] for i in order]
    Si = np.clip(S[order], 0, None)
    STi = np.clip(ST[order], 0, None)
    y = np.arange(D)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(y + 0.2, STi, height=0.4, color="#7e57c2", label="total-order $S_{Ti}$ (incl. interactions)")
    ax.barh(y - 0.2, Si,  height=0.4, color="#26a69a", label="first-order $S_i$")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Sobol sensitivity index [-]")
    ax.set_title("Global (variance-based) sensitivity of the chosen gear mass\n"
                 f"(output std = {np.sqrt(var):.1f} kg; Sobol' sampling, N={N_SOBOL})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Method 3 - Monte Carlo selection robustness (Latin Hypercube)
# ---------------------------------------------------------------------------
def monte_carlo():
    lhs = qmc.LatinHypercube(d=D, seed=SEED)
    U = lhs.random(N_MC)
    X = U * (HI - LO) + LO
    mass = {a: np.full(N_MC, np.nan) for a in ARCHS}
    selected = []
    for i in range(N_MC):
        masses, sel, _ = evaluate(_overrides_from_row(X[i]))
        for a in ARCHS:
            mass[a][i] = masses.get(a, np.nan)
        selected.append(sel)
    selected = np.array(selected, dtype=object)
    return mass, selected


def plot_mc_distributions(mass, path):
    data, labels, colors = [], [], []
    for a in ARCHS:
        m = mass[a][np.isfinite(mass[a])]
        if m.size:
            data.append(m); labels.append(f"{a}\n({'reusable' if REUSABLE_AT_LIMIT[a] else 'yields@limit'})")
            colors.append(ARCH_COLOR[a])
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bp = ax.boxplot(data, vert=True, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("gear mass [kg]")
    ax.set_title(f"Architecture mass distributions under joint uncertainty (LHS, N={N_MC})\n"
                 "feasible samples only; lower = lighter")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_selection_frequency(mass, selected, path):
    feas_pct = {a: 100.0 * np.mean(np.isfinite(mass[a])) for a in ARCHS}
    sel_pct = {a: 100.0 * np.mean(selected == a) for a in ARCHS}
    y = np.arange(len(ARCHS))
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.barh(y + 0.2, [feas_pct[a] for a in ARCHS], height=0.4,
            color="#bdbdbd", label="feasible")
    ax.barh(y - 0.2, [sel_pct[a] for a in ARCHS], height=0.4,
            color=[ARCH_COLOR[a] for a in ARCHS], label="SELECTED")
    for i, a in enumerate(ARCHS):
        ax.text(sel_pct[a] + 1, i - 0.2, f"{sel_pct[a]:.0f}%", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(ARCHS)
    ax.set_xlabel("% of Monte-Carlo samples")
    ax.set_xlim(0, 105)
    ax.set_title("How often each architecture is feasible vs SELECTED\n"
                 f"(weighted trade-off score, N={N_MC})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_reusable_headtohead(mass, path):
    """Head-to-head of the two reusable absorbers: is the metal leaf lighter than the
    elastomer? (Empty if the metal leaf never reaches feasibility at this landing mass.)"""
    a, b = mass["B metal leaf"], mass["E elastomeric"]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    win = 100.0 * np.mean(a < b) if a.size else float("nan")
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.scatter(a, b, s=14, alpha=0.5, color="#8d6e63")
    if a.size:
        lim = [min(a.min(), b.min()) * 0.95, max(a.max(), b.max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1, label="equal mass")
        ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("B metal leaf gear mass [kg]")
    ax.set_ylabel("E elastomeric gear mass [kg]")
    ax.set_title("Reusable head-to-head: metal leaf vs elastomeric\n"
                 f"metal leaf is lighter in {win:.0f}% of samples (points above the line)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return win


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Landing-gear sensitivity study  (landing mass = {M_LANDING_KG:.0f} kg, "
          f"weighted trade-off score)")
    print(f"  inputs varied : {D}   |  Sobol N={N_SOBOL} (~{N_SOBOL*(D+2)} evals)  |  MC N={N_MC}")

    # 1) Tornado
    print("\n[1/3] Tornado (local one-at-a-time) ...")
    base, rows = tornado()
    plot_tornado(base, rows, OUT_DIR / "tornado.png")
    print(f"      nominal selected mass = {base:.1f} kg.  Top drivers (low->high swing):")
    for label, dlo, dhi, swing in reversed(rows[-5:]):
        print(f"        {label:<24s} swing {swing:5.1f} kg  [{dlo:+.1f}, {dhi:+.1f}]")

    # 2) Sobol global indices
    print("\n[2/3] Sobol variance-based global sensitivity ...")
    S, ST, var = sobol()
    plot_sobol(S, ST, var, OUT_DIR / "sobol_indices.png")
    idx = np.argsort(ST)[::-1]
    print(f"      output std = {np.sqrt(var):.1f} kg.  Most influential (total-order):")
    for i in idx[:5]:
        print(f"        {PLABEL[PNAMES[i]]:<24s} S_T={ST[i]:.2f}  S_1={S[i]:.2f}")

    # 3) Monte Carlo selection robustness
    print("\n[3/3] Monte Carlo selection robustness (LHS) ...")
    mass, selected = monte_carlo()
    plot_mc_distributions(mass, OUT_DIR / "architecture_distributions.png")
    plot_selection_frequency(mass, selected, OUT_DIR / "selection_frequency.png")
    win = plot_reusable_headtohead(mass, OUT_DIR / "reusable_headtohead.png")

    sel_pct = {a: 100.0 * np.mean(selected == a) for a in ARCHS}
    feas_pct = {a: 100.0 * np.mean(np.isfinite(mass[a])) for a in ARCHS}
    print("\n  Architecture          feasible%   selected%   median mass [kg]")
    for a in ARCHS:
        m = mass[a][np.isfinite(mass[a])]
        med = np.median(m) if m.size else float("nan")
        print(f"    {a:<19s} {feas_pct[a]:7.0f}   {sel_pct[a]:8.0f}   {med:10.1f}")

    nominal_winner = max(sel_pct, key=sel_pct.get)
    print(f"\n  ROBUSTNESS OF THE CHOICE:")
    print(f"    - '{nominal_winner}' SELECTED in {sel_pct[nominal_winner]:.0f}% of samples.")
    print(f"    - Among the reusable options, the metal leaf is lighter than the elastomer "
          f"in {win:.0f}% of samples.")

    # CSV summary
    import pandas as pd
    summ = pd.DataFrame({
        "param": PNAMES,
        "label": [PLABEL[p] for p in PNAMES],
        "low": LO, "high": HI, "nominal": [NOMINAL[p] for p in PNAMES],
        "sobol_S1": S, "sobol_ST": ST,
    })
    summ.to_csv(OUT_DIR / "sensitivity_summary.csv", index=False)
    print(f"\nFigures + CSV written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
