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

# Shared evaluation core + the single goal registry.
from final_characteristics.stability_eval import (
    REQUIREMENT_NAMES,
    evaluate_stability,
)

# Converger pieces. We call the PRIVATE _solve_converged_mass directly when a
# geometry design var is active so the global converger cache is NOT polluted
# with the overridden geometry; the cached baseline is used otherwise.
from class_II_sizing.mtow_sizing import load_final_design_state
from class_II_sizing import mtow_sizing
import class_II_sizing.mass_components as mass_components

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
#                                         (recognised keys: x_cg, s_aft_to_s_total)
#                  "override:<dotted>"  -> stability_eval param_overrides[<dotted>]
#   converger -- name of a class_II_sizing.mass_components module global to
#                override before reconverging the MTOW (None = does not affect the
#                converged mass). Only "wing_loading" does today.
#
# The set + bounds below are SUGGESTIONS grounded in the Sobol consult
# (stability_sensitivity.py). Edit to match the team's chosen tunables.
DESIGN_VARIABLES = {
    "x_cg":             {"bounds": (2.8, 3.6),   "route": "design_var:x_cg",                           "converger": None},
    "s_aft_to_s_total": {"bounds": (0.30, 0.70), "route": "override:s_aft_to_s_total",                 "converger": None},
    "dihedral_fw":      {"bounds": (0.0, 6.0),   "route": "override:wing_geometry.dihedral_front_wing","converger": None},
    "dihedral_aw":      {"bounds": (0.0, 6.0),   "route": "override:wing_geometry.dihedral_aft_wing",  "converger": None},
    "x_LEMAC_fw":       {"bounds": (0.5, 2.0),   "route": "override:wing_geometry.x_LEMAC_fw",         "converger": None},
    "b_vert_tail":      {"bounds": (1.2, 2.4),   "route": "override:tail_geometry.b_vert_tail",        "converger": None},
    # wing_loading feeds BOTH the stability geometry (design_point) AND the MTOW
    # converger (WING_LOADING_N). Including it makes each evaluation reconverge
    # the MTOW (~25 s) -- see the performance note in converged_mtow().
    "wing_loading":     {"bounds": (650.0, 880.0), "route": "override:wing_geometry.design_point",     "converger": "WING_LOADING_N"},
}
DESIGN_VARIABLE_NAMES = list(DESIGN_VARIABLES.keys())


def design_vector_to_dict(x):
    """Turn a pymoo design vector x into a named {var_name: value} dict."""
    return {name: float(x[index]) for index, name in enumerate(DESIGN_VARIABLE_NAMES)}


def default_design_vector():
    """Nominal starting design (mid-range-ish, near the current sheet values)."""
    return {
        "x_cg":             3.311,  # current sheet x_cg_opt
        "s_aft_to_s_total": 0.50,
        "dihedral_fw":      0.0,
        "dihedral_aw":      0.0,
        "x_LEMAC_fw":       1.0,
        "b_vert_tail":      1.6,
        "wing_loading":     760.0,
    }


# ===========================================================================
# 2. CONVERGER WITH GEOMETRY OVERRIDES  (so MTOW responds to the design vars)
# ===========================================================================
# Memoise on the rounded geometry tuple so identical geometries are not
# reconverged. A fresh converge runs the internal landing-gear SLSQP sizing and
# costs ~25 s, so this is the dominant cost of Phase 2 whenever a converger-
# coupled design var (wing_loading) is active. Keep pop_size/n_gen small, or
# build a coarse MTOW(geometry) surrogate, for serious runs.
_MTOW_CACHE = {}


def converged_mtow(geometry_overrides):
    """Return the converged MTOW [kg] for the given converger-geometry overrides.

    geometry_overrides maps a mass_components module-global name to its value
    (e.g. {"WING_LOADING_N": 820.0}). Empty -> the cached baseline MTOW.

    We temporarily set the mass_components globals and call the converger's
    private _solve_converged_mass(), which does NOT touch the module-level cache
    that stability_eval reads, then restore the globals.
    """
    if not geometry_overrides:
        return load_final_design_state(force_recompute=False)["mtow"]

    key = tuple(sorted((name, round(value, 3)) for name, value in geometry_overrides.items()))
    if key in _MTOW_CACHE:
        return _MTOW_CACHE[key]

    saved = {name: getattr(mass_components, name) for name in geometry_overrides}
    try:
        for name, value in geometry_overrides.items():
            setattr(mass_components, name, value)
        state = mtow_sizing._solve_converged_mass(verbose=False)
        mtow = state["mtow"]
    finally:
        for name, value in saved.items():
            setattr(mass_components, name, value)

    _MTOW_CACHE[key] = mtow
    return mtow


# ===========================================================================
# 3. CORE EVALUATION  (shared by Phase 1 and Phase 2)
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
# 4. PHASE 1 -- single acceptance check
# ===========================================================================
def run_phase_1():
    """Evaluate the current/default design once and report accept/reject."""
    print("\n=== PHASE 1: acceptance check on the current design ===")
    results = evaluate_design(default_design_vector())
    print_design_report(results)
    return results["accepted"]


# ===========================================================================
# 5. PHASE 2 -- NSGA-II optimisation (pymoo)
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

    # pop_size / n_gen are intentionally small: when wing_loading is active each
    # evaluation reconverges the MTOW (~25 s). Raise these only with a surrogate
    # or after dropping the converger-coupled design var.
    algorithm = NSGA2(
        pop_size=12,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    result = minimize(problem, algorithm, termination=("n_gen", 10), seed=1, verbose=True)

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
# 6. ORCHESTRATION
# ===========================================================================
def main():
    if run_phase_1():
        print("\nDesign accepted in Phase 1 -- no optimisation needed.")
    else:
        run_phase_2()


if __name__ == "__main__":
    main()
