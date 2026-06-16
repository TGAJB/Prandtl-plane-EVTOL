"""
optimiser.py
============

Two-phase MTOW / stability design driver.

WHY THIS FILE EXISTS
--------------------
A converged MTOW on its own does not mean the design is acceptable: the stability
& control derivatives, the centre-of-gravity (c.g.) range, and the neutral point
all stem from that MTOW and the geometry, and they must each lie inside the ranges
the team has set. This file ties those together.

THE TWO-PHASE METHOD
--------------------
Phase 1 -- Acceptance check.
    Evaluate the current design against every stability requirement.
    All requirements pass  -> ACCEPTED, done.
    Any requirement fails  -> go to Phase 2.

Phase 2 -- Optimisation (only if Phase 1 fails).
    Run an NSGA-II optimiser (gradient-free GA). For each candidate set of design
    variables it (re)converges the MTOW, recomputes the derivatives / c.g.
    envelopes, and checks the same requirements, searching the design space until
    a converged design satisfies every requirement.

WHERE THE PHYSICS LIVES
-----------------------
All the evaluation and the GOAL definitions live in `stability_eval.py` (shared,
pymoo-free). This file holds ONLY:
  * the design-variable registry (which knobs the optimiser may turn),
  * the converger-with-geometry wiring (so the converged MTOW responds to the
    geometry design variables), and
  * the NSGA-II machinery.

The design-variable set below is informed by `stability_sensitivity.py` (the
standalone Sobol consult); edit it freely as the team decides what to tune.

Run:      python final_characteristics/optimiser.py
Requires: pip install pymoo
"""

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap (project root + vehicle_dynamics, same as stability_eval).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VEHICLE_DYNAMICS_DIR = PROJECT_ROOT / "final_characteristics" / "vehicle_dynamics"
for _path in (PROJECT_ROOT, VEHICLE_DYNAMICS_DIR):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

# Shared evaluation core + the single goal registry. converged_mtow lives in the
# shared core so the converger-override path is defined once (it calls the
# PRIVATE _solve_converged_mass directly when a geometry design var is active so
# the global converger cache is NOT polluted with the overridden geometry).
from final_characteristics.stability_eval import (
    REQUIREMENT_NAMES,
    converged_mtow,
    converged_state,          # {mtow, wing_feasible, wing_margin} from the detailed wing sizing
    evaluate_stability,
    _nominal_margin_scales,   # per-requirement |baseline margin| scales (shared with the objective)
)
# Matching diagram: supplies the W/S (wing-loading) search range (stall upper
# bound, fraction-of-stall lower bound) and the cruise/climb power-feasibility +
# cruise-CL constraints. Re-implemented parametrically (no hardcoded inputs, no
# qualitative selection) in final_characteristics/matching_diagram.py.
from final_characteristics import matching_diagram as _md

# Design-variable BOUNDS are derived from parameters.py (single source of truth),
# never hardcoded: the z_w_fw range is expressed as a chord/ground span and the
# VTOL-battery slide range from the (flat-wide) box length vs the fuselage.
import parameters as _p
from class_II_sizing.mtow_sizing import converged_mass as _converged_mass

_WG = _p.WingGeometry()
_TG = _p.TailGeometry()
# Front-wing height: placement range from the fuselage-centreline datum, set by the
# fuselage layout (parameters.Z_W_FW_MIN/MAX = [-0.9, -0.6] m, z up, -=below). Sorted
# so pymoo gets (low, high).
_ZWFW_BOUNDS = tuple(sorted((_p.Z_W_FW_MIN, _p.Z_W_FW_MAX)))
# Sliding-battery VTOL station: cruise station back to where the flat-wide box's
# aft face reaches the fuselage tail (so it stays inside the airframe).
_BATT_MASS = _converged_mass()["battery"]
_XBATT_VTOL_BOUNDS = _p.battery_vtol_slide_range(_BATT_MASS)
# Cruise-battery station: kept forward enough that the box stays inside the cabin
# and the cruise c.g. is balanceable; aft bound is the cruise station.
_XBATT_CRUISE_BOUNDS = (1.5, _p.X_BATT_CRUISE + 0.3)

# Aft/total wing-area split (S_aw / S_tot). Centred on the current equal-area
# box-wing (S_aw/S_tot from parameters.py, = 0.5) and constrained to +/-0.10 so
# both Prandtl box-wings keep a balanced, valid area share. It is a TRUE design
# lever: it feeds the aero wing-split (stability) AND, via the converger global
# mass_components.S_AFT_TO_S_TOTAL, the per-wing structural mass -> MTOW.
_S_AFT_BASELINE = _WG.S_aw / _WG.S_tot
_S_AFT_BOUNDS = (_S_AFT_BASELINE - 0.10, _S_AFT_BASELINE + 0.10)

# Vertical-tail span search range. It trades directional stability against tail
# structural MASS, so it is converger-coupled (see COUPLE_TAIL_MASS): a smaller
# tail lowers MTOW. Kept as an explicit, documented range (the matching diagram
# does not bound it).
_B_TAIL_BOUNDS = (1.4, 1.6)

# Wing loading W/S search range, straight from the matching diagram: the stall
# constraint sets the hard UPPER bound, a fraction of it the lower bound.
_WS_BOUNDS = _md.feasible_wing_loading_range()
# W/S is converger-coupled (sets wing AREA -> wing mass -> MTOW), so every
# distinct W/S reconverges the MTOW (~23 s). The converged MTOW depends only on the
# two converger-coupled levers, W/S and the aft/total split (see below), so we
# QUANTISE both to a grid before evaluating: the converged-MTOW cache then holds
# across the GA, and the mass area stays consistent with the stability area
# (both use the quantised value).
W_S_CACHE_STEP = 5.0    # [N/m^2] W/S quantisation grid (caches the reconverge); finer
                        # = more distinct MTOW levels -> a richer Pareto front (n_nds),
                        # at the cost of more first-touch reconverges (each cached).
# The aft/total split is now ALSO converger-coupled (it sets the per-wing
# structural mass via mass_components.S_AFT_TO_S_TOTAL), so it likewise reconverges
# the MTOW. Quantise it to a coarse grid so the converged-MTOW cache holds: the
# MTOW now keys on the (W/S, split) pair, and only the distinct grid points pay the
# reconverge. Its mass effect is second-order, so a coarse 0.05 grid is plenty.
S_AFT_CACHE_STEP = 0.05   # [-] aft/total-split quantisation grid (caches the reconverge)
# The vertical-tail span is also converger-coupled (sets S_TAIL/AR_T -> tail mass
# -> MTOW; see COUPLE_TAIL_MASS), so it likewise reconverges the MTOW. Quantise it
# to a grid so the converged-MTOW cache holds across the (W/S, split, tail) key.
B_TAIL_CACHE_STEP = 0.02   # [m] vertical-tail-span quantisation grid (caches the reconverge)
# The non-folding wingbox thickness t_nonfolding is a pure FEASIBILITY lever (it sets
# the torsional twist of the non-folding wing, checked against MAX_TWIST_NONFOLDING_DEG
# in size_wing); it does NOT change MTOW. It is still part of the converger cache key
# (size_wing reads it live), so quantise it to a 1 mm grid to cap the distinct
# reconverges to the 6 values across its [1 mm, 6 mm] range.
T_NONFOLD_CACHE_STEP = 0.001   # [m] non-folding-thickness quantisation grid
_TNONFOLD_BOUNDS = (0.001, 0.006)   # [m] = [1 mm, 6 mm]
# Hover-sized installed power [W] (baseline) -- reference for the matching-diagram
# power-feasibility constraint. It is ~7x the cruise/climb requirement and grows
# only weakly with MTOW, so the baseline value is an adequate, documented constant.
_P_INSTALLED_W = _converged_mass()["max_power_kw"] * 1000.0


def _quantize_wing_loading(w_s):
    """Snap W/S to the cache grid so identical wing areas reuse the converged MTOW.

    Clipped to the matching-diagram bounds AFTER snapping: round-to-nearest could
    otherwise push the top of the range past the stall limit (the upper bound is a
    hard matching-diagram constraint, not a pymoo G entry), so the clip guarantees
    the evaluated W/S never exceeds the stall limit nor drops below the floor.
    """
    q = round(w_s / W_S_CACHE_STEP) * W_S_CACHE_STEP
    return min(max(q, _WS_BOUNDS[0]), _WS_BOUNDS[1])


def _quantize_s_aft(s_aft):
    """Snap the aft/total split to the cache grid (mirrors _quantize_wing_loading).

    The same quantised value feeds BOTH the converger (per-wing structural mass /
    MTOW) and the aero wing-split + MMOI c.g. in stability_eval, keeping the mass
    area consistent with the stability area. Clipped to the split bounds AFTER
    snapping so it never leaves the reasonable Prandtl range."""
    q = round(s_aft / S_AFT_CACHE_STEP) * S_AFT_CACHE_STEP
    return min(max(q, _S_AFT_BOUNDS[0]), _S_AFT_BOUNDS[1])


def _quantize_b_tail(b_tail):
    """Snap the vertical-tail span to the cache grid (mirrors _quantize_wing_loading).

    The same quantised value feeds BOTH the converger (tail mass / MTOW) and the
    stability override (directional derivatives), so a Pareto design reproduces its
    reported MTOW exactly when its tail span is baked back into the sheet."""
    q = round(b_tail / B_TAIL_CACHE_STEP) * B_TAIL_CACHE_STEP
    return min(max(q, _B_TAIL_BOUNDS[0]), _B_TAIL_BOUNDS[1])


def _quantize_t_nonfolding(t):
    """Snap the non-folding wingbox thickness to the cache grid (mirrors
    _quantize_wing_loading). t_nonfolding does not change MTOW but is part of the
    converger cache key (size_wing reads it live for the torsion feasibility check),
    so quantising caps the distinct reconverges to its 6 grid points."""
    q = round(t / T_NONFOLD_CACHE_STEP) * T_NONFOLD_CACHE_STEP
    return min(max(q, _TNONFOLD_BOUNDS[0]), _TNONFOLD_BOUNDS[1])

# Couple the vertical-tail span to the MTOW converger (tail mass)? ON: b_vert_tail
# is the one layout lever with a real MASS effect, so leaving it off made the
# optimiser optimise a smaller tail for stability while still charging the baseline
# (bigger) tail mass to MTOW -- the reported MTOW then dropped when the optimal tail
# was baked into the sheet. The span is quantised (B_TAIL_CACHE_STEP) so the
# converged-MTOW cache absorbs the extra reconverges. See evaluate_design.
COUPLE_TAIL_MASS = True

# Phase 2 returns the chosen design but does NOT write it anywhere. The tuned
# design variables are consumed by the final_design/ runner, which serialises them
# to final_design/chosen_design.json (read by parameters.py at import) so every
# other module sees the design WITHOUT rewriting the parameters.py source.

# pymoo: the NSGA-II implementation used in Phase 2.
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination.default import DefaultMultiObjectiveTermination

# Phase-2 termination knobs. A convergence-based termination replaces a fixed
# generation count: the GA runs until the Pareto front stops moving (relative
# change < FTOL, sustained over TERMINATION_PERIOD generations) instead of always
# burning exactly N generations -- so it neither under-runs (stopping while the
# front is still improving) nor wastes evaluations after it has converged. The
# n_max_gen cap bounds the worst case.
TERMINATION_FTOL = 0.0025   # [-] relative change in normalised objective space
TERMINATION_PERIOD = 10     # generations the tolerance must hold before stopping
TERMINATION_MAX_GEN = 60    # hard cap on generations (safety bound)


# ===========================================================================
# 1. DESIGN-VARIABLE REGISTRY
# ===========================================================================
# Each entry maps a design-variable name to:
#   bounds    -- (low, high) search range
#   route     -- where the value goes when evaluating a design:
#                  "design_var:<key>"  -> stability_eval design_vars[<key>]
#                                         (recognised keys: s_aft_to_s_total)
#                  "override:<dotted>"  -> stability_eval param_overrides[<dotted>]
#   converger -- name of a class_II_sizing.mass_components module global to
#                override before reconverging the MTOW (None = does not affect the
#                converged mass). "WING_LOADING_N" (W/S) and "S_AFT_TO_S_TOTAL"
#                (aft/total area split) do today -- a var may carry BOTH a route and
#                a converger entry (the split does: aero AND mass).
#
# NOTE: the c.g. is NOT a design variable. It is an EMERGENT output of the
# component mass build-up, computed by MMOI (cruise + VTOL) inside
# stability_eval.evaluate_stability. The c.g.-envelope requirements remain
# constraints; promote the layout roots that MOVE the c.g. (e.g. x_LEMAC_fw,
# x_vert_tail, battery/payload stations) rather than tuning the c.g. directly.
#
# The seven tunable levers the team can manipulate (the c.g. is NOT one: it emerges
# from the layout + the sliding battery). All ranges come from parameters.py.
#
# DEFERRED: the rear-rotor SPANWISE lever (ETA_ROTOR_RW -> folded rotor further aft
# via the 120-deg wing fold). Analysis showed it moves the OEI window the WRONG way
# for cg_vtol_fwd (it pushes the window aft of the c.g.), and the sliding battery
# already seats the VTOL c.g. in the window, so it is omitted here. Re-add it once
# the fold-kinematic folded-rotor position is wired into MMOI + the OEI LP.
DESIGN_VARIABLES = {
    # front-wing LEMAC: nose (0) to the current 1.5 m station -- shifts the cruise
    # c.g. envelope (and, via the MMOI wing station, the c.g.).
    "x_LEMAC_fw":    {"bounds": (0.0, 1.5),            "route": "override:wing_geometry.x_LEMAC_fw", "converger": None},
    # sliding battery: cruise station (cruise c.g.) and aft VTOL-emergency station
    # (seats the VTOL-OEI c.g. in its envelope without disturbing cruise).
    "x_batt_cruise": {"bounds": _XBATT_CRUISE_BOUNDS,  "route": "design_var:x_batt_cruise",          "converger": None},
    "x_batt_vtol":   {"bounds": _XBATT_VTOL_BOUNDS,    "route": "design_var:x_batt_vtol",            "converger": None},
    # front-wing height z (3/4 to 2.4 root-chords above the ground datum).
    "z_w_fw":        {"bounds": _ZWFW_BOUNDS,          "route": "override:wing_geometry.z_w_fw",     "converger": None},
    # vertical-tail span (also feeds the MTOW converger via the tail mass, so
    # shrinking it trades directional stability against mass -- see evaluate_design).
    # converger=None here: the S_TAIL/AR_T coupling is wired separately in
    # evaluate_design (COUPLE_TAIL_MASS) because it sets TWO converger globals.
    "b_vert_tail":   {"bounds": _B_TAIL_BOUNDS,        "route": "override:tail_geometry.b_vert_tail", "converger": None},
    # wing loading W/S: sets the wing AREA -> wing mass -> converged MTOW
    # (converger-coupled via WING_LOADING_N) AND the stability reference area
    # (S_tot = mtow*g/design_point). The lever that makes MTOW a live objective.
    # Bounds from the matching diagram (stall upper limit; 0.6x lower). The override
    # route sets params.wing_geometry.design_point for stability; the converger
    # entry sets mass_components.WING_LOADING_N for the mass build-up -- BOTH are
    # needed because they are independent module copies of W/S.
    "design_point":  {"bounds": _WS_BOUNDS,            "route": "override:wing_geometry.design_point", "converger": "WING_LOADING_N"},
    # aft/total wing-area split (S_aw/S_tot): sizes the front vs aft box-wing area.
    # The design_var route feeds the aero wing-split (-> C_M_alpha, area ratios, cg
    # aft limit, MMOI c.g. via stability_eval); the converger entry sets
    # mass_components.S_AFT_TO_S_TOTAL so the per-wing STRUCTURAL mass -> MTOW
    # responds too. Range +/-0.10 around the equal-area baseline (Prandtl-valid).
    "s_aft_to_s_total": {"bounds": _S_AFT_BOUNDS,      "route": "design_var:s_aft_to_s_total",         "converger": "S_AFT_TO_S_TOTAL"},
    # non-folding wingbox skin thickness t_nonfolding [m], [1 mm, 6 mm]. CONVERGER-ONLY
    # (route kind "converger"): it touches neither the aero/stability params nor the
    # MTOW mass build-up -- it only sets the non-folding torsional twist that size_wing
    # checks against MAX_TWIST_NONFOLDING_DEG. The wing-feasibility constraint (see
    # DesignProblem._evaluate) is what gives the optimiser a reason to tune it.
    "t_nonfolding":   {"bounds": _TNONFOLD_BOUNDS,     "route": "converger:t_nonfolding",              "converger": "t_nonfolding"},
}
DESIGN_VARIABLE_NAMES = list(DESIGN_VARIABLES.keys())


def design_vector_to_dict(x):
    """Turn a pymoo design vector x into a named {var_name: value} dict."""
    return {name: float(x[index]) for index, name in enumerate(DESIGN_VARIABLE_NAMES)}


def default_design_vector():
    """Nominal starting design (mid-range, near the current sheet values)."""
    return {
        "x_LEMAC_fw":    1.5,
        "x_batt_cruise": _p.X_BATT_CRUISE,
        "x_batt_vtol":   0.5 * (_XBATT_VTOL_BOUNDS[0] + _XBATT_VTOL_BOUNDS[1]),
        "z_w_fw":        0.5 * (_ZWFW_BOUNDS[0] + _ZWFW_BOUNDS[1]),
        "b_vert_tail":   1.6,
        "design_point":  _p.WING_LOADING_N,   # current sheet W/S (760)
        "s_aft_to_s_total": _S_AFT_BASELINE,  # current equal-area split (0.5)
        "t_nonfolding":  _p.t_nonfolding,     # current non-folding wingbox thickness (1 mm)
    }


# ===========================================================================
# 2. CORE EVALUATION  (shared by Phase 1 and Phase 2)
# ===========================================================================
def evaluate_design(design_vars=None):
    """Evaluate one design: converge MTOW (geometry-aware), check all requirements.

    Routes each design variable to the converger and/or the stability evaluation,
    then returns the stability_eval results dict augmented with the design vector.
    """
    if design_vars is None:
        design_vars = default_design_vector()

    # Quantise W/S to the cache grid so identical wing areas reuse the converged
    # MTOW (the converger is the ~23 s cost). The same quantised value feeds BOTH
    # the converger (area/mass) and the stability override (S_tot), keeping them
    # consistent, and is what the report prints.
    design_vars = dict(design_vars)
    if "design_point" in design_vars:
        design_vars["design_point"] = _quantize_wing_loading(design_vars["design_point"])
    if "s_aft_to_s_total" in design_vars:
        design_vars["s_aft_to_s_total"] = _quantize_s_aft(design_vars["s_aft_to_s_total"])
    if "b_vert_tail" in design_vars:
        design_vars["b_vert_tail"] = _quantize_b_tail(design_vars["b_vert_tail"])
    if "t_nonfolding" in design_vars:
        design_vars["t_nonfolding"] = _quantize_t_nonfolding(design_vars["t_nonfolding"])

    # Split the design vars by where they go.
    param_overrides = {}
    stability_design_vars = {}
    geometry_overrides = {}
    for name, value in design_vars.items():
        spec = DESIGN_VARIABLES[name]
        kind, key = spec["route"].split(":", 1)
        if kind == "design_var":
            stability_design_vars[key] = value
        elif kind == "override":
            param_overrides[key] = value
        # kind == "converger": no stability/param route; it only feeds the converger
        # below via spec["converger"] (e.g. t_nonfolding -> feasibility, not MTOW).
        if spec["converger"]:
            geometry_overrides[spec["converger"]] = value

    # The vertical-tail span also drives the tail MASS in the converger (S_TAIL,
    # AR_T): shrinking it lowers MTOW. Coupling it (COUPLE_TAIL_MASS, on by default)
    # uses the SAME quantised b_vert_tail that stability sees, so the reported MTOW
    # reflects the optimal tail and reproduces exactly when the span is baked into
    # the sheet. (S_TAIL/AR_T mirror TailGeometry.S_vert_tail/AR_vert_tail.)
    if COUPLE_TAIL_MASS and "b_vert_tail" in design_vars:
        b = design_vars["b_vert_tail"]
        s_tail = (_TG.c_r_vert_tail + _TG.c_t_vert_tail) * b / 2.0
        geometry_overrides["S_TAIL"] = s_tail
        geometry_overrides["AR_T"] = b * b / s_tail

    # 1) Converged MTOW + detailed-wing feasibility (respond to converger-coupled vars).
    state = converged_state(geometry_overrides)
    mtow = state["mtow"]

    # 2) Stability requirements at that MTOW with the overrides applied.
    results = evaluate_stability(
        param_overrides=param_overrides,
        design_vars=stability_design_vars,
        mtow=mtow,
    )
    results["design_vars"] = design_vars
    # Carry the structural wing feasibility through so DesignProblem can turn it into a
    # constraint (wing_margin <= 0 means feasible) and the report can show it. A
    # wing-infeasible design is never "accepted" (folds into the Phase-1 accept test).
    results["wing_feasible"] = state["wing_feasible"]
    results["wing_margin"] = state["wing_margin"]
    results["accepted"] = bool(results.get("accepted", False) and state["wing_feasible"])
    return results


def print_design_report(results):
    """Print a readable summary of one evaluated design."""
    print("-" * 64)
    print(f"  Converged MTOW : {results['mtow']:.2f} kg")
    print("  Design variables:")
    for name, value in results["design_vars"].items():
        print(f"    {name:18s}: {value:.4f}")
    print("  Requirement checks (PASS / FAIL):")
    for name in REQUIREMENT_NAMES:
        status = "PASS" if results["requirements"][name] else "FAIL"
        print(f"    {name:18s}: {status}   (margin {results['margins'][name]:+.4f})")
    print(f"  Worst normalised margin (robustness): {results['worst_margin']:+.4f}  "
          f"(<=0 = all met with slack; ~-1 = baseline slack)")
    print(f"  Wing structure (bending+torsion): "
          f"{'FEASIBLE' if results.get('wing_feasible', True) else 'INFEASIBLE'}  "
          f"(margin {results.get('wing_margin', -1.0):+.4f}, <=0 = OK)")
    print(f"  Overall: {'ACCEPTED' if results['accepted'] else 'REJECTED'}")
    print("-" * 64)


# ===========================================================================
# 3. PHASE 1 -- single acceptance check
# ===========================================================================
def run_phase_1():
    """Evaluate the current/default design once and report accept/reject."""
    print("\n=== PHASE 1: acceptance check on the current design ===")
    results = evaluate_design(default_design_vector())
    print_design_report(results)
    return results["accepted"]


# ===========================================================================
# 4. PHASE 2 -- NSGA-II optimisation (pymoo)
# ===========================================================================
class DesignProblem(Problem):
    """pymoo problem wrapper around evaluate_design().

    n_var    = number of design variables
    n_obj    = 2  -> minimise (converged MTOW, worst NORMALISED stability margin)
                     With W/S (design_point) now a lever, MTOW is a LIVE objective
                     (a smaller wing lowers MTOW), so the front spreads in MTOW. The
                     worst normalised margin (each margin scaled by its baseline
                     magnitude, so robustness tracks RELATIVE slack) is the second
                     objective -> a real Pareto trade of mass against stability slack.
    n_constr = stability requirements + 2 matching-diagram feasibility constraints
               (cruise/climb power, cruise CL) + 1 wing-structure feasibility
               constraint (bending + non-folding torsion, driven by t_nonfolding);
               all in pymoo's "g <= 0 = OK" form, all normalised to ~O(1) so the
               constraint-violation pull is balanced across requirements (see _evaluate)
    xl / xu  = bounds from DESIGN_VARIABLES
    """

    def __init__(self):
        lower = [DESIGN_VARIABLES[name]["bounds"][0] for name in DESIGN_VARIABLE_NAMES]
        upper = [DESIGN_VARIABLES[name]["bounds"][1] for name in DESIGN_VARIABLE_NAMES]
        super().__init__(
            n_var=len(DESIGN_VARIABLE_NAMES),
            n_obj=2,                          # (MTOW, worst-case stability margin)
            n_constr=len(REQUIREMENT_NAMES) + 3,   # + matching-diagram power & CL + wing structure
            xl=np.array(lower),
            xu=np.array(upper),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # Per-requirement |baseline margin| scales (computed once, cached). The
        # SAME scales normalise the robustness objective; using them on the
        # constraints too keeps the two consistent and puts every requirement on a
        # common O(1) scale (see the CONSTRAINTS note below).
        scales = _nominal_margin_scales()

        objective_rows = []
        constraint_rows = []
        for x in X:
            results = evaluate_design(design_vector_to_dict(x))

            # OBJECTIVES (both minimised by pymoo):
            #   1) the converged MTOW.
            #   2) the worst NORMALISED stability margin -- each margin is divided
            #      by its baseline magnitude so the binding requirement is the one
            #      with the least RELATIVE slack (not the smallest absolute number,
            #      which would always be C_Y_p). Driving it down makes the design
            #      robustly stable where it actually has tension. total_violation
            #      would flatten to 0 across the feasible region; worst_margin keeps
            #      improving, so it is the objective that actually shapes the front.
            objective_rows.append([results["mtow"], results["worst_margin"]])

            # CONSTRAINTS (all "g(x) <= 0 means satisfied"):
            #   - each stability requirement's signed margin, NORMALISED by its
            #     |baseline margin| (the same scales the objective uses). The raw
            #     margins span ~O(100) (C_M_q) down to ~O(0.01) (C_Y_p) and the cg
            #     limits in metres (~O(0.1)); feeding them raw lets pymoo's
            #     constraint-violation aggregate be dominated by the large-magnitude
            #     derivatives (which already pass) and barely feel the cg-envelope
            #     margins that actually bind. Dividing by the baseline scale puts
            #     every requirement on a common ~O(1) RELATIVE-slack scale so the
            #     feasibility pull is balanced across them.
            #   - the matching-diagram feasibility constraints, already normalised
            #     to O(1) the same way:
            #       g_power = (required cruise/climb power - installed) / installed
            #       g_cl    = (cruise CL - CL_max_operational) / CL_max_operational
            #     Both are deeply slack for this hover-power-dominated eVTOL, but
            #     they encode the matching diagram so it stays enforced if a future
            #     design ever approaches the cruise/stall power or lift limits.
            w_s = results["design_vars"]["design_point"]
            g_power = (_md.required_power_W(w_s, results["mtow"]) - _P_INSTALLED_W) / _P_INSTALLED_W
            g_cl = (_md.cruise_lift_coefficient(w_s) - _p.CL_MAX_OPERATIONAL) / _p.CL_MAX_OPERATIONAL
            #   - the wing-structure feasibility constraint from the detailed sizing:
            #     g_wing = results["wing_margin"], already <=0 iff the wing is feasible
            #     (no bending overstress AND non-folding torsional twist within
            #     MAX_TWIST_NONFOLDING_DEG). The twist term is continuous in t_nonfolding,
            #     so it gives the GA a gradient to tune the thickness toward feasibility.
            g_wing = results["wing_margin"]
            constraint_rows.append(
                [results["margins"][name] / scales[name] for name in REQUIREMENT_NAMES]
                + [g_power, g_cl, g_wing]
            )

        out["F"] = np.array(objective_rows)
        out["G"] = np.array(constraint_rows)


class AftBatterySeededSampling(FloatRandomSampling):
    """Random initial population, but with the VTOL-emergency battery biased into
    the AFT end of its travel.

    cg_vtol_fwd_ok is the single binding requirement, and it is only satisfiable
    with the sliding VTOL battery near its aft travel bound -- the feasible region
    is a thin sliver there. Seeding x_batt_vtol into the aft fraction of its travel
    starts the population inside (or right next to) that corner, so the GA does not
    have to discover it by chance. Every OTHER design variable is left fully random,
    so global diversity is preserved where it actually matters.
    """
    AFT_FRACTION = 0.2   # seed x_batt_vtol within the aft 20% of its travel

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        X = super()._do(problem, n_samples, *args, random_state=random_state, **kwargs)
        i = DESIGN_VARIABLE_NAMES.index("x_batt_vtol")
        lo, hi = problem.xl[i], problem.xu[i]
        # Reuse the RNG pymoo threaded through so the seed stays reproducible.
        u = (random_state.random(n_samples) if random_state is not None
             else np.random.random(n_samples))
        X[:, i] = hi - u * self.AFT_FRACTION * (hi - lo)
        return X


def run_phase_2():
    """Run NSGA-II until it finds a converged, requirement-satisfying design."""
    print("\n=== PHASE 2: NSGA-II optimisation (design rejected in Phase 1) ===")

    problem = DesignProblem()

    # The converger-coupled levers (W/S, the aft/total split and the tail span)
    # reconverge the MTOW (~23 s), but all are quantised to a grid (W_S_CACHE_STEP /
    # S_AFT_CACHE_STEP / B_TAIL_CACHE_STEP), so the converged-MTOW cache holds: only
    # the distinct (W/S, split, tail) grid points pay the reconverge once, and every
    # other evaluation is cheap (~0.3 s). Seeding the VTOL battery aft (see
    # AftBatterySeededSampling) starts the search inside the thin VTOL-OEI feasible
    # corner instead of hunting for it.
    algorithm = NSGA2(
        pop_size=24,
        sampling=AftBatterySeededSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    # Convergence-based termination: stop once the Pareto front stops moving rather
    # than after a fixed generation count, with a hard generation cap as a safety
    # bound (see the TERMINATION_* knobs).
    termination = DefaultMultiObjectiveTermination(
        ftol=TERMINATION_FTOL,
        period=TERMINATION_PERIOD,
        n_max_gen=TERMINATION_MAX_GEN,
    )
    result = minimize(problem, algorithm, termination=termination, seed=1, verbose=True)

    pareto_x = result.X
    if pareto_x is None:
        print("NSGA-II did not return a feasible design. Some requirements may not be "
              "reachable with the current design-variable set (e.g. the VTOL OEI "
              "envelope, which needs a propulsion change, not geometry).")
        return None

    # Two objectives now -> result.X is the non-dominated (Pareto) set trading MTOW
    # against stability robustness. Report the whole front, then detail one pick.
    pareto_x = np.atleast_2d(pareto_x)
    pareto_f = np.atleast_2d(result.F)
    order = np.argsort(pareto_f[:, 0])   # by ascending MTOW

    print(f"\nNSGA-II Pareto front ({len(pareto_x)} non-dominated designs):")
    print("    rank   MTOW [kg]   worst norm. margin (<=0 = feasible)")
    for rank, i in enumerate(order):
        print(f"    #{rank:<3d}  {pareto_f[i, 0]:9.2f}   {pareto_f[i, 1]:+.4f}")

    # Representative pick: the lightest design that is also feasible (worst margin
    # <= 0); if none on the front are feasible, fall back to the most robust one.
    feasible = order[pareto_f[order, 1] <= 0.0]
    chosen = int(feasible[0]) if len(feasible) else int(np.argmin(pareto_f[:, 1]))

    label = "lightest feasible" if len(feasible) else "most robust (none feasible)"
    print(f"\nRepresentative design on the front ({label}):")
    best = evaluate_design(design_vector_to_dict(pareto_x[chosen]))
    print_design_report(best)
    print("\nThe chosen design is NOT written to parameters.py. Run it through the "
          "final_design/ runner to apply it (via final_design/chosen_design.json) and "
          "regenerate all department outputs.")
    return best


# ===========================================================================
# 5. ORCHESTRATION
# ===========================================================================
def main():
    if run_phase_1():
        print("\nDesign accepted in Phase 1 -- no optimisation needed.")
    else:
        run_phase_2()


if __name__ == "__main__":
    main()
