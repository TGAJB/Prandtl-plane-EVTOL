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
    evaluate_stability,
)

# Design-variable BOUNDS are derived from parameters.py (single source of truth),
# never hardcoded: the z_w_fw range is expressed as a chord/ground span and the
# VTOL-battery slide range from the (flat-wide) box length vs the fuselage.
import parameters as _p
from class_II_sizing.mtow_sizing import converged_mass as _converged_mass

_WG = _p.WingGeometry()
_TG = _p.TailGeometry()
_C_ROOT_FW = _WG.chord_fw_root                     # [m] front-wing root chord
_Z_GROUND = _p.Z_GEAR                              # [m] skid-bottom datum (centreline frame, z up)
# Front-wing height above the ground (skid) datum: bounded by (3/4 * root_chord)
# and (2.4 m - root_chord). Sorted so pymoo gets (low, high); in the body frame
# this is ~[-0.56, -0.055] m, bracketing the current -0.5 m.
_ZWFW_BOUNDS = tuple(sorted((
    _Z_GROUND + 0.75 * _C_ROOT_FW,     # 3/4 root-chord above ground
    _Z_GROUND + (2.4 - _C_ROOT_FW),    # 2.4 m - root chord above ground
)))
# Sliding-battery VTOL station: cruise station back to where the flat-wide box's
# aft face reaches the fuselage tail (so it stays inside the airframe).
_BATT_MASS = _converged_mass()["battery"]
_XBATT_VTOL_BOUNDS = _p.battery_vtol_slide_range(_BATT_MASS)
# Cruise-battery station: kept forward enough that the box stays inside the cabin
# and the cruise c.g. is balanceable; aft bound is the cruise station.
_XBATT_CRUISE_BOUNDS = (1.5, _p.X_BATT_CRUISE + 0.3)

# Couple the vertical-tail span to the MTOW converger (tail mass)? Off by default
# for speed (each coupled eval reconverges the MTOW, ~6 s). See evaluate_design.
COUPLE_TAIL_MASS = False

# pymoo: the NSGA-II implementation used in Phase 2.
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM


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
#                converged mass). Only "wing_loading" does today.
#
# NOTE: the c.g. is NOT a design variable. It is an EMERGENT output of the
# component mass build-up, computed by MMOI (cruise + VTOL) inside
# stability_eval.evaluate_stability. The c.g.-envelope requirements remain
# constraints; promote the layout roots that MOVE the c.g. (e.g. x_LEMAC_fw,
# x_vert_tail, battery/payload stations) rather than tuning the c.g. directly.
#
# The six tunable levers the team can manipulate (the c.g. is NOT one: it emerges
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
    "b_vert_tail":   {"bounds": (1.4, 1.6),            "route": "override:tail_geometry.b_vert_tail", "converger": None},
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

    # Split the design vars by where they go.
    param_overrides = {}
    stability_design_vars = {}
    geometry_overrides = {}
    for name, value in design_vars.items():
        spec = DESIGN_VARIABLES[name]
        kind, key = spec["route"].split(":", 1)
        if kind == "design_var":
            stability_design_vars[key] = value
        else:  # "override"
            param_overrides[key] = value
        if spec["converger"]:
            geometry_overrides[spec["converger"]] = value

    # The vertical-tail span also drives the tail MASS in the converger (S_TAIL,
    # AR_T); shrinking it lowers MTOW slightly. That coupling is left OFF by
    # default because it forces a ~6 s MTOW reconverge on every evaluation for a
    # tail-mass effect of only ~10-20 kg. Set COUPLE_TAIL_MASS = True to trade
    # speed for a true min-MTOW gradient on the tail span.
    if COUPLE_TAIL_MASS and "b_vert_tail" in design_vars:
        b = design_vars["b_vert_tail"]
        s_tail = (_TG.c_r_vert_tail + _TG.c_t_vert_tail) * b / 2.0
        geometry_overrides["S_TAIL"] = s_tail
        geometry_overrides["AR_T"] = b * b / s_tail

    # 1) Converged MTOW (responds to converger-coupled geometry vars).
    mtow = converged_mtow(geometry_overrides)

    # 2) Stability requirements at that MTOW with the overrides applied.
    results = evaluate_stability(
        param_overrides=param_overrides,
        design_vars=stability_design_vars,
        mtow=mtow,
    )
    results["design_vars"] = design_vars
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
    n_obj    = 1  -> minimise the converged MTOW
    n_constr = number of stability requirements (their signed margins; <= 0 = OK)
    xl / xu  = bounds from DESIGN_VARIABLES
    """

    def __init__(self):
        lower = [DESIGN_VARIABLES[name]["bounds"][0] for name in DESIGN_VARIABLE_NAMES]
        upper = [DESIGN_VARIABLES[name]["bounds"][1] for name in DESIGN_VARIABLE_NAMES]
        super().__init__(
            n_var=len(DESIGN_VARIABLE_NAMES),
            n_obj=1,                          # <-- bump to 2 + add a 2nd F below for true Pareto
            n_constr=len(REQUIREMENT_NAMES),
            xl=np.array(lower),
            xu=np.array(upper),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        objective_rows = []
        constraint_rows = []
        for x in X:
            results = evaluate_design(design_vector_to_dict(x))

            # OBJECTIVE: minimise the converged MTOW.
            # To add a 2nd objective (true multi-objective Pareto): append it here,
            # e.g. objectives = [results["mtow"], results["total_violation"]], and
            # set n_obj=2 above.
            objective_rows.append([results["mtow"]])

            # CONSTRAINTS: each requirement's signed margin is already in pymoo's
            # "g(x) <= 0 means satisfied" convention.
            constraint_rows.append([results["margins"][name] for name in REQUIREMENT_NAMES])

        out["F"] = np.array(objective_rows)
        out["G"] = np.array(constraint_rows)


def run_phase_2():
    """Run NSGA-II until it finds a converged, requirement-satisfying design."""
    print("\n=== PHASE 2: NSGA-II optimisation (design rejected in Phase 1) ===")

    problem = DesignProblem()

    # The five active levers do NOT reconverge the MTOW (COUPLE_TAIL_MASS=False),
    # so each evaluation is cheap (~0.3 s) and we can afford a healthy population
    # and generation count for robust exploration of the feasible region (the
    # VTOL-OEI constraint needs the sliding battery near its aft travel).
    algorithm = NSGA2(
        pop_size=24,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    result = minimize(problem, algorithm, termination=("n_gen", 20), seed=1, verbose=True)

    best_x = result.X
    if best_x is None:
        print("NSGA-II did not return a feasible design. Some requirements may not be "
              "reachable with the current design-variable set (e.g. the VTOL OEI "
              "envelope, which needs a propulsion change, not geometry).")
        return None

    best_x = np.atleast_2d(best_x)[0]
    print("\nBest design found by NSGA-II:")
    best = evaluate_design(design_vector_to_dict(best_x))
    print_design_report(best)
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
