"""
stability_sensitivity.py
========================

STANDALONE sensitivity consult for the stability requirements AND the converged
MTOW.  NOT part of the optimiser loop -- run it, read the graphs, and manually
decide which currently fixed parameters.py values are worth promoting to
optimiser design variables.

WHAT IT DOES
------------
For a list of ROOT DESIGN parameters -- the bare, non-interdependent ORIGIN
inputs the team sets that cascade into the design (NOT another department's
calculated results, and NOT emergent quantities) -- it sweeps each over a
plausible range and measures how (a) every stability requirement and (b) the
converged MTOW respond, using two well-established global-sensitivity methods
(mirroring class_II_sizing/landing_gear_sensitivity.py):

  1. TORNADO (local one-at-a-time) -- quick readable ranking of which root moves
     an output most when swept low->high.  Produced for the number of failing
     requirements AND for the converged MTOW.

  2. SOBOL indices (variance-based GLOBAL sensitivity, Saltelli/Jansen) -- for
     EACH requirement's signed margin, for the converged MTOW, AND for the
     aggregate "number of failing requirements", decomposes the output variance
     into each root's first-order and total-order contribution.

The headline output is a heatmap of the total-order index for every
(root x output) pair -- it shows at a glance which knob drives which requirement
(e.g. dihedral -> C_L_beta) and which knobs drive the MTOW.

ROOTS ONLY  (edit the PARAMS list freely)
-----------------------------------------
Every swept entry is a true origin input.  Deliberately EXCLUDED:
  * aero-department results (CL_alpha_fw/aw, downwash, x_ac_*_cruise,
    dyn_pres_ratio, I_v, oswald e, C_M_ac_*, section cl_alpha), and
  * EMERGENT quantities -- notably the c.g. (mass.x_cg_opt), which is an OUTPUT
    of the component mass/position build-up (MMOI), not a root.  With the c.g.
    removed as an input the c.g.-envelope requirements still vary: they respond
    to the geometry roots through the moving envelope LIMITS, with the c.g. held
    at its design value.  (Limitation: the c.g. itself is not re-derived from
    component placement here -- envelope-only.  Recomputing it from the layout
    roots via MMOI is a possible future enhancement.)

MTOW IS NOW COUPLED  (single coupled sweep)
-------------------------------------------
Roots tagged with a converger global (the 7th tuple field) feed the MTOW
converger.  Each sample reconverges the MTOW for those roots (via the shared
stability_eval.converged_mtow, memoised on the converger sub-vector) and uses
that MTOW in the stability evaluation, so the MTOW output and the area-dependent
derivatives both respond.  A fresh converge runs the internal landing-gear SLSQP
(~25 s); the memoisation means only the converger-coupled columns trigger it, so
the cost is ~ (2 + n_converger_roots) * N_SOBOL reconverges.  Lower N_SOBOL (env
var STAB_SENS_N_SOBOL) for a quick look.

Run:    python final_characteristics/stability_sensitivity.py
        STAB_SENS_N_SOBOL=4 python final_characteristics/stability_sensitivity.py   # quick
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
# Sobol base sample size (total evals ~ N_SOBOL*(D+2)); power of 2. The MTOW
# coupling makes each converger-coupled sample cost ~25 s, so allow an env
# override for a quick look.
N_SOBOL = int(os.environ.get("STAB_SENS_N_SOBOL", "16"))
SEED    = 12345

# Root design parameters to sweep:
#   (dotted_path, low, high, nominal, label, unit, converger_global)
# dotted_path is fed straight to stability_eval.evaluate_stability(param_overrides).
# The pseudo-key "s_aft_to_s_total" is the aft/total wing-area split.
# converger_global is the class_II_sizing.mass_components module global to
#   override before reconverging the MTOW (None = MTOW-neutral root).
#
# Only ROOTS appear here; derived/emergent fields (tail S/AR/MAC, winglet AR,
# z_w_aw, b_aw, taper_aw, areas from W/S, and the c.g.) are excluded.  Edit
# freely; ROOT_DESIGN_VARIABLES.md lists further roots the team may toggle in
# (e.g. wing span b_fw, twists, z_vert_tail).
PARAMS = [
    # ---- Longitudinal / planform ----
    ("s_aft_to_s_total",                  0.30,  0.70,  0.50,  "aft/total wing-area split", "-",     None),
    ("wing_geometry.design_point",        650.0, 880.0, 760.0, "wing loading W/S",          "N/m^2", "WING_LOADING_N"),
    ("wing_geometry.taper_fw",            0.30,  0.60,  0.45,  "wing taper",                "-",     "TAPER_W"),
    ("wing_geometry.x_LEMAC_aw",          4.5,   6.5,   5.5982,"aft-wing LEMAC x (sets stagger)", "m", None),
    ("wing_geometry.gap",                 1.5,   2.8,   2.1,   "wing gap",                  "m",     None),
    ("wing_geometry.x_LEMAC_fw",          0.5,   2.0,   1.0,   "front-wing LEMAC x",        "m",     None),
    # ---- Roll (dihedral / wing height / sweep) ----
    ("wing_geometry.dihedral_front_wing", 0.0,   6.0,   0.0,   "front-wing dihedral",       "deg",   None),
    ("wing_geometry.dihedral_aft_wing",   0.0,   6.0,   0.0,   "aft-wing dihedral",         "deg",   None),
    ("wing_geometry.z_w_fw",             -0.9,  -0.6,  -0.75,  "front-wing height z",       "m",     None),
    ("wing_geometry.LE_sweep_fw",         0.0,  10.0,   0.0,   "front-wing LE sweep",       "deg",   None),
    ("wing_geometry.LE_sweep_aw",         0.0,  10.0,   0.0,   "aft-wing LE sweep",         "deg",   None),
    # ---- Directional (vertical tail / winglet) ----
    ("tail_geometry.b_vert_tail",         1.2,   2.4,   1.6,   "vertical-tail span",        "m",     None),
    ("tail_geometry.x_vert_tail",         5.5,   7.5,   6.4,   "vertical-tail arm",         "m",     None),
    ("tail_geometry.c_r_vert_tail",       1.2,   2.0,   1.6,   "v-tail root chord",         "m",     None),
    ("tail_geometry.c_t_vert_tail",       0.9,   1.6,   1.3,   "v-tail tip chord",          "m",     None),
    ("winglet_geometry.S_winglet",        1.0,   2.5,   1.57,  "winglet area",              "m^2",   None),
    ("winglet_geometry.b_winglet",        1.5,   2.7,   2.1,   "winglet span",              "m",     None),
    # ---- Fuselage (MTOW-coupled via Raymer mass) ----
    ("fuselage_geometry.fuselage_length", 6.0,   8.5,   7.0,   "fuselage length",           "m",     "L_FUS"),
    ("fuselage_geometry.d_fw",            1.6,   2.4,   2.0,   "fuselage width",            "m",     "FUSE_WIDTH"),
]
PNAMES    = [p[0] for p in PARAMS]
PLABEL    = {p[0]: p[4] for p in PARAMS}
LO        = np.array([p[1] for p in PARAMS])
HI        = np.array([p[2] for p in PARAMS])
NOMINAL   = {p[0]: p[3] for p in PARAMS}
# Map of swept root -> mass_components global to override when reconverging MTOW.
CONVERGER = {p[0]: p[6] for p in PARAMS if p[6]}
D = len(PARAMS)

REQ_NAMES = se.REQUIREMENT_NAMES
# The aggregate failing-count and the converged MTOW are extra outputs appended
# to the per-requirement margins.
AGG_NAME  = "n_failed"
MTOW_NAME = "mtow"
OUTPUT_NAMES = REQ_NAMES + [AGG_NAME, MTOW_NAME]
# Rows shown in the per-output heatmap: every requirement plus the MTOW (the
# aggregate n_failed is summarised separately in its own tornado/bar).
HEATMAP_ROWS = REQ_NAMES + [MTOW_NAME]


# ---------------------------------------------------------------------------
# Core evaluation: one parameter vector -> all outputs
# ---------------------------------------------------------------------------
def evaluate_outputs(overrides):
    """Run the coupled evaluation for one override dict; return a dict of outputs.

    Reconverges the MTOW for any converger-coupled roots in `overrides` (memoised
    in stability_eval), then evaluates the stability requirements at that MTOW.
    Outputs are every requirement's signed margin (<= 0 means satisfied), the
    aggregate count of failing requirements, and the converged MTOW [kg].
    """
    geometry_overrides = {
        CONVERGER[name]: value for name, value in overrides.items() if name in CONVERGER
    }
    mtow = se.converged_mtow(geometry_overrides)

    results = se.evaluate_stability(param_overrides=overrides, mtow=mtow)
    out = dict(results["margins"])  # name -> signed margin
    out[AGG_NAME] = float(sum(1 for ok in results["requirements"].values() if not ok))
    out[MTOW_NAME] = float(results["mtow"])
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
# Method 1 - Tornado (local one-at-a-time) for a chosen output
# ---------------------------------------------------------------------------
def tornado(output_name):
    """Local low->high swing of `output_name` for each root, vs the nominal."""
    base = evaluate_outputs(dict(NOMINAL))[output_name]
    rows = []
    for name, lo, hi, nom, label, unit, _conv in PARAMS:
        o = dict(NOMINAL); o[name] = lo
        v_lo = evaluate_outputs(o)[output_name]
        o = dict(NOMINAL); o[name] = hi
        v_hi = evaluate_outputs(o)[output_name]
        rows.append((label, v_lo - base, v_hi - base, abs(v_hi - v_lo)))
    rows.sort(key=lambda r: r[3])  # ascending swing -> largest at top of barh
    return base, rows


def plot_tornado(base, rows, path, xlabel, title):
    labels = [r[0] for r in rows]
    lo = [r[1] for r in rows]
    hi = [r[2] for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(y, lo, color="#ef5350", label="root at low end")
    ax.barh(y, hi, color="#42a5f5", label="root at high end")
    ax.axvline(0.0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
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
    """Heatmap of total-order index for every (root x output) pair.

    Rows are every requirement plus the converged MTOW; columns are the roots.
    The MTOW row lights up only on the converger-coupled roots.
    """
    matrix = np.array([np.clip(ST_mat[row], 0, None) for row in HEATMAP_ROWS])  # (R, D)
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * D, 1.0 + 0.40 * len(HEATMAP_ROWS)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(D)); ax.set_xticklabels([PLABEL[p] for p in PNAMES], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(HEATMAP_ROWS))); ax.set_yticklabels(HEATMAP_ROWS, fontsize=7)
    # Visually separate the MTOW row from the requirement rows.
    ax.axhline(len(REQ_NAMES) - 0.5, color="w", lw=1.5)
    for r in range(len(HEATMAP_ROWS)):
        for c in range(D):
            val = matrix[r, c]
            if val > 0.02:
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        color="white" if val < 0.6 else "black", fontsize=6)
    ax.set_title("Total-order Sobol index $S_{Ti}$ per (root x output)\n"
                 "bright = this root drives this requirement / the MTOW")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="$S_{Ti}$")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n_conv = len(CONVERGER)
    print(f"Stability + MTOW sensitivity consult  ({D} roots, {len(REQ_NAMES)} requirements)")
    print(f"  Sobol N={N_SOBOL}  (~{N_SOBOL * (D + 2)} evaluations)")
    print(f"  MTOW-coupled roots: {n_conv}  (~{(2 + n_conv) * N_SOBOL} MTOW reconverges @ ~25 s each;"
          f" set STAB_SENS_N_SOBOL to lower)")

    # 1) Tornado on the aggregate failing count AND on the converged MTOW.
    print("\n[1/2] Tornado (local one-at-a-time) ...")
    base_fail, rows_fail = tornado(AGG_NAME)
    plot_tornado(
        base_fail, rows_fail, OUT_DIR / "tornado_failures.png",
        xlabel="change in NUMBER OF FAILING requirements vs nominal",
        title=f"Tornado: local sensitivity of requirement failures\n"
              f"(nominal failing count = {base_fail:.0f})",
    )
    base_mtow, rows_mtow = tornado(MTOW_NAME)
    plot_tornado(
        base_mtow, rows_mtow, OUT_DIR / "tornado_mtow.png",
        xlabel="change in converged MTOW [kg] vs nominal",
        title=f"Tornado: local sensitivity of the converged MTOW\n"
              f"(nominal MTOW = {base_mtow:.1f} kg)",
    )
    print(f"      nominal failing count = {base_fail:.0f}, nominal MTOW = {base_mtow:.1f} kg.")
    print("      Largest MTOW swings (low->high):")
    for label, dlo, dhi, swing in reversed(rows_mtow[-5:]):
        print(f"        {label:<26s} swing {swing:7.1f} kg  [{dlo:+.1f}, {dhi:+.1f}]")

    # 2) Sobol per requirement + MTOW + aggregate
    print("\n[2/2] Sobol variance-based global sensitivity ...")
    S_mat, ST_mat = sobol()
    plot_sobol_aggregate(S_mat, ST_mat, OUT_DIR / "sobol_aggregate.png")
    plot_heatmap(ST_mat, OUT_DIR / "sobol_heatmap.png")

    print("      Top driver (total-order) of each output that VARIES:")
    for name in HEATMAP_ROWS:
        ST = ST_mat[name]
        if np.max(ST) <= 1e-9:
            continue  # constant output (e.g. always-infeasible VTOL OEI)
        i = int(np.argmax(ST))
        print(f"        {name:<18s} <- {PLABEL[PNAMES[i]]:<26s} (S_T={ST[i]:.2f})")

    # CSV summary: total-order index per (root x output). Uses the stdlib csv
    # module so there is no third-party dependency.
    import csv
    header = ["parameter", "label", "converger_global"] + [f"ST_{name}" for name in OUTPUT_NAMES]
    with open(OUT_DIR / "sobol_total_order.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for j, pname in enumerate(PNAMES):
            row = [pname, PLABEL[pname], CONVERGER.get(pname, "")] + \
                  [f"{ST_mat[name][j]:.6f}" for name in OUTPUT_NAMES]
            writer.writerow(row)

    print(f"\nFigures + CSV written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
