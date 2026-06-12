"""
stability_sensitivity.py
========================

STANDALONE sensitivity consult for the stability requirements.  NOT part of the
optimiser loop -- run it, read the graphs, and manually decide which currently
fixed parameters.py values are worth promoting to optimiser design variables.

WHAT IT DOES
------------
For a list of CORE DESIGN parameters (things the team sets that FEED the
computation -- NOT another department's calculated results), it sweeps each over
a plausible range and measures how the stability requirements respond, using two
well-established global-sensitivity methods (mirroring the proven structure of
class_II_sizing/landing_gear_sensitivity.py):

  1. TORNADO (local one-at-a-time) -- quick readable ranking of which parameter
     moves the number of failing requirements most when swept low->high.

  2. SOBOL indices (variance-based GLOBAL sensitivity, Saltelli/Jansen) -- for
     EACH requirement's signed margin AND for the aggregate "number of failing
     requirements", decomposes the output variance into each parameter's
     first-order and total-order contribution.  Scrambled Sobol' sampling.

The headline output is a heatmap of the total-order index for every
(parameter x requirement) pair: it shows at a glance which knob drives which
requirement (e.g. dihedral -> C_L_beta).

WHAT IS SWEPT  (edit the PARAMS list freely)
--------------------------------------------
Only core design inputs.  Deliberately EXCLUDED (aero-department results, not
design knobs): CL_alpha_fw/aw, downwash gradient, x_ac_*_cruise, dyn_pres_ratio,
I_v, oswald e, C_M_ac_*, section cl_alpha.

The MTOW is held at the cached converged value: the sign requirements are
essentially mass-independent and re-converging per sample would be far slower
(same choice the landing-gear study makes by fixing the landing mass).

Run:    python final_characteristics/stability_sensitivity.py
Output: PNG figures + a CSV in  final_characteristics/sensitivity_plots/stability/
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: save figures, do not open a window
import matplotlib.pyplot as plt
from scipy.stats import qmc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import final_characteristics.stability_eval as se

OUT_DIR = Path(__file__).resolve().parent / "sensitivity_plots" / "stability"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SOBOL = 16    # Sobol base sample size (total evals ~ N_SOBOL*(D+2)); power of 2
SEED    = 12345

# Core design parameters to sweep: (dotted_path, low, high, nominal, label, unit).
# dotted_path is fed straight to stability_eval.evaluate_stability(param_overrides).
# The pseudo-key "s_aft_to_s_total" is the aft/total wing-area split.
# Edit this list to add/remove candidates; ranges are plausible design bounds.
PARAMS = [
    # ---- Longitudinal geometry ----
    ("mass.x_cg_opt",                    2.8,   3.6,   3.311, "c.g. (cruise/VTOL)",      "m"),
    ("s_aft_to_s_total",                 0.30,  0.70,  0.50,  "aft/total wing-area split","-"),
    ("wing_geometry.stagger",            4.0,   6.0,   5.0,   "wing stagger",            "m"),
    ("wing_geometry.gap",                1.5,   2.8,   2.1,   "wing gap",                "m"),
    ("wing_geometry.x_LEMAC_fw",         0.5,   2.0,   1.0,   "front-wing LEMAC x",      "m"),
    ("wing_geometry.design_point",       650.0, 880.0, 760.0, "wing loading W/S",        "N/m^2"),
    ("wing_geometry.taper_fw",           0.30,  0.60,  0.45,  "front-wing taper",        "-"),
    # ---- Roll (dihedral / wing height / sweep) ----
    ("wing_geometry.dihedral_front_wing", 0.0,  6.0,   0.0,   "front-wing dihedral",     "deg"),
    ("wing_geometry.dihedral_aft_wing",   0.0,  6.0,   0.0,   "aft-wing dihedral",       "deg"),
    ("wing_geometry.z_w_fw",            -1.0,   0.0,  -0.5,   "front-wing height z",     "m"),
    ("wing_geometry.LE_sweep_fw",        0.0,  10.0,   0.0,   "front-wing LE sweep",     "deg"),
    # ---- Directional (vertical tail / winglet) ----
    ("tail_geometry.b_vert_tail",        1.2,   2.4,   1.6,   "vertical-tail span",      "m"),
    ("tail_geometry.x_vert_tail",        5.5,   7.5,   6.4,   "vertical-tail arm",       "m"),
    ("winglet_geometry.S_winglet",       1.0,   2.5,   1.57,  "winglet area",            "m^2"),
]
PNAMES  = [p[0] for p in PARAMS]
PLABEL  = {p[0]: p[4] for p in PARAMS}
LO      = np.array([p[1] for p in PARAMS])
HI      = np.array([p[2] for p in PARAMS])
NOMINAL = {p[0]: p[3] for p in PARAMS}
D = len(PARAMS)

REQ_NAMES = se.REQUIREMENT_NAMES
# The aggregate output is appended to the per-requirement outputs.
AGG_NAME = "n_failed"
OUTPUT_NAMES = REQ_NAMES + [AGG_NAME]


# ---------------------------------------------------------------------------
# Core evaluation: one parameter vector -> all outputs
# ---------------------------------------------------------------------------
def evaluate_outputs(overrides):
    """Run stability_eval for one override dict; return a dict of outputs.

    Outputs are every requirement's signed margin (<= 0 means satisfied) plus
    the aggregate count of failing requirements.
    """
    results = se.evaluate_stability(param_overrides=overrides)
    out = dict(results["margins"])  # name -> signed margin
    out[AGG_NAME] = float(sum(1 for ok in results["requirements"].values() if not ok))
    return out


def _overrides_from_row(row):
    return {PNAMES[j]: float(row[j]) for j in range(D)}


def _eval_matrix(matrix):
    """Evaluate every row of `matrix`; return {output_name: array(N)}."""
    n = matrix.shape[0]
    collected = {name: np.empty(n) for name in OUTPUT_NAMES}
    for i in range(n):
        out = evaluate_outputs(_overrides_from_row(matrix[i]))
        for name in OUTPUT_NAMES:
            collected[name][i] = out[name]
    return collected


# ---------------------------------------------------------------------------
# Method 1 - Tornado (local one-at-a-time) on the aggregate n_failed
# ---------------------------------------------------------------------------
def tornado():
    base = evaluate_outputs(dict(NOMINAL))[AGG_NAME]
    rows = []
    for name, lo, hi, nom, label, unit in PARAMS:
        o = dict(NOMINAL); o[name] = lo
        n_lo = evaluate_outputs(o)[AGG_NAME]
        o = dict(NOMINAL); o[name] = hi
        n_hi = evaluate_outputs(o)[AGG_NAME]
        rows.append((label, n_lo - base, n_hi - base, abs(n_hi - n_lo)))
    rows.sort(key=lambda r: r[3])  # ascending swing -> largest at top of barh
    return base, rows


def plot_tornado(base, rows, path):
    labels = [r[0] for r in rows]
    lo = [r[1] for r in rows]
    hi = [r[2] for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(y, lo, color="#ef5350", label="parameter at low end")
    ax.barh(y, hi, color="#42a5f5", label="parameter at high end")
    ax.axvline(0.0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("change in NUMBER OF FAILING requirements vs nominal")
    ax.set_title(f"Tornado: local sensitivity of requirement failures\n"
                 f"(nominal failing count = {base:.0f})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Method 2 - Sobol variance-based global sensitivity (Saltelli / Jansen)
# ---------------------------------------------------------------------------
def _sobol_indices(fA, fB, fAB_list):
    """First-order (Saltelli) and total-order (Jansen) indices for one output.

    fA, fB are arrays (N,); fAB_list is a list of D arrays (one per parameter).
    Returns (S, ST) arrays of length D. Guards the zero-variance case (a constant
    output, e.g. an always-infeasible VTOL requirement) by returning zeros.
    """
    var = np.var(np.concatenate([fA, fB]), ddof=1)
    S = np.zeros(D); ST = np.zeros(D)
    if var <= 1e-12:
        return S, ST  # output does not vary over the swept space
    for i in range(D):
        fABi = fAB_list[i]
        S[i]  = np.mean(fB * (fABi - fA)) / var          # Saltelli 2010 first-order
        ST[i] = 0.5 * np.mean((fA - fABi) ** 2) / var     # Jansen total-order
    return S, ST


def sobol():
    """Run the Sobol design once; return per-output S and ST matrices.

    Returns:
        S_mat, ST_mat: dicts {output_name: array(D)} of first/total-order indices.
    """
    sampler = qmc.Sobol(d=2 * D, scramble=True, seed=SEED)
    base01 = sampler.random(N_SOBOL)
    A = base01[:, :D] * (HI - LO) + LO
    B = base01[:, D:] * (HI - LO) + LO

    fA = _eval_matrix(A)
    fB = _eval_matrix(B)

    # Build the D cross-matrices AB_i (= A with column i replaced by B's column i).
    fAB = []  # fAB[i] is a dict {output: array(N)}
    for i in range(D):
        ABi = A.copy()
        ABi[:, i] = B[:, i]
        fAB.append(_eval_matrix(ABi))

    S_mat, ST_mat = {}, {}
    for name in OUTPUT_NAMES:
        fAB_list = [fAB[i][name] for i in range(D)]
        S, ST = _sobol_indices(fA[name], fB[name], fAB_list)
        S_mat[name] = S
        ST_mat[name] = ST
    return S_mat, ST_mat


def plot_sobol_aggregate(S_mat, ST_mat, path):
    """Bar chart of first/total-order indices for the aggregate n_failed output."""
    S = S_mat[AGG_NAME]; ST = ST_mat[AGG_NAME]
    order = np.argsort(ST)
    labels = [PLABEL[PNAMES[i]] for i in order]
    Si = np.clip(S[order], 0, None)
    STi = np.clip(ST[order], 0, None)
    y = np.arange(D)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(y + 0.2, STi, height=0.4, color="#7e57c2", label="total-order $S_{Ti}$")
    ax.barh(y - 0.2, Si,  height=0.4, color="#26a69a", label="first-order $S_i$")
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Sobol sensitivity index [-]")
    ax.set_title("Global sensitivity of the NUMBER OF FAILING requirements\n"
                 f"(Sobol' sampling, N={N_SOBOL})")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_heatmap(ST_mat, path):
    """Heatmap of total-order index for every (parameter x requirement) pair."""
    # rows = requirements (exclude the aggregate), cols = parameters
    matrix = np.array([np.clip(ST_mat[req], 0, None) for req in REQ_NAMES])  # (R, D)
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * D, 1.0 + 0.40 * len(REQ_NAMES)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(D)); ax.set_xticklabels([PLABEL[p] for p in PNAMES], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(REQ_NAMES))); ax.set_yticklabels(REQ_NAMES, fontsize=7)
    for r in range(len(REQ_NAMES)):
        for c in range(D):
            val = matrix[r, c]
            if val > 0.02:
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        color="white" if val < 0.6 else "black", fontsize=6)
    ax.set_title("Total-order Sobol index $S_{Ti}$ per (parameter x requirement)\n"
                 "bright = this parameter drives this requirement")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="$S_{Ti}$")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Stability sensitivity consult  ({D} parameters, {len(REQ_NAMES)} requirements)")
    print(f"  Sobol N={N_SOBOL}  (~{N_SOBOL * (D + 2)} evaluations)")

    # 1) Tornado on the aggregate
    print("\n[1/2] Tornado (local one-at-a-time) on number of failing requirements ...")
    base, rows = tornado()
    plot_tornado(base, rows, OUT_DIR / "tornado_failures.png")
    print(f"      nominal failing count = {base:.0f}.  Largest swings (low->high):")
    for label, dlo, dhi, swing in reversed(rows[-5:]):
        print(f"        {label:<26s} swing {swing:.0f}  [{dlo:+.0f}, {dhi:+.0f}]")

    # 2) Sobol per requirement + aggregate
    print("\n[2/2] Sobol variance-based global sensitivity ...")
    S_mat, ST_mat = sobol()
    plot_sobol_aggregate(S_mat, ST_mat, OUT_DIR / "sobol_aggregate.png")
    plot_heatmap(ST_mat, OUT_DIR / "sobol_heatmap.png")

    print("      Top driver (total-order) of each requirement that VARIES:")
    for req in REQ_NAMES:
        ST = ST_mat[req]
        if np.max(ST) <= 1e-9:
            continue  # constant requirement (e.g. always-infeasible VTOL OEI)
        i = int(np.argmax(ST))
        print(f"        {req:<18s} <- {PLABEL[PNAMES[i]]:<26s} (S_T={ST[i]:.2f})")

    # CSV summary: total-order index per (parameter x output). Uses the stdlib
    # csv module so there is no third-party dependency.
    import csv
    header = ["parameter", "label"] + [f"ST_{name}" for name in OUTPUT_NAMES]
    with open(OUT_DIR / "sobol_total_order.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for j, pname in enumerate(PNAMES):
            row = [pname, PLABEL[pname]] + [f"{ST_mat[name][j]:.6f}" for name in OUTPUT_NAMES]
            writer.writerow(row)

    print(f"\nFigures + CSV written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
