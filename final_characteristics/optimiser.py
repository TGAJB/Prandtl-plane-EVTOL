"""
optimiser.py
============

Two-phase MTOW / stability design driver.  TEMPLATE.

WHY THIS FILE EXISTS
--------------------
Our MTOW is found by a fixed-point converger (class_II_sizing/mtow_sizing.py):
it iterates the component mass build-up until the MTOW stops changing.  But a
converged MTOW on its own does not mean the design is acceptable: the stability
derivatives, the centre-of-gravity (c.g.) range, and the neutral point all stem
from that MTOW, and they must each lie inside ranges we have decided are
feasible.  This file ties those two ideas together.

THE TWO-PHASE METHOD
--------------------
Phase 1 -- Acceptance check.
    Run the existing converger, compute the derivatives / c.g. range / neutral
    point for the converged MTOW, and check every one against its allowed range.
    If ALL constraints pass  -> the design is ACCEPTED, we are done.
    If ANY constraint fails  -> go to Phase 2.

Phase 2 -- Optimisation (only if Phase 1 fails).
    Run an NSGA-II optimiser (multi-objective genetic algorithm, gradient-free).
    For each candidate set of tunable design variables it re-runs the converger,
    recomputes the derivatives / c.g. / neutral point, and checks the same
    constraints.  NSGA-II searches the design space until it finds variable
    values that give a converged MTOW AND satisfy every constraint.

WHAT IS STILL A PLACEHOLDER (clearly marked "DUMMY" throughout)
---------------------------------------------------------------
  * The exact tunable design variables and their bounds  (see DESIGN_VARIABLES).
  * The exact constraint ranges                          (see CONSTRAINTS).
  * The two analysis routines that turn (mtow, design vars) into derivatives and
    neutral-point / c.g. numbers.  Here they are dummy stand-ins
    (compute_all_derivatives, compute_neutral_point_and_cg) that return made-up
    numbers so the whole pipeline runs today.  They will be replaced by the real
    routines (cf. vehicle_dynamics/aircraft.py and stat_long_stab_anal.py).
  * The optimiser currently has ONE objective (minimise MTOW).  The hook for a
    second objective (true multi-objective Pareto search) is marked in
    DesignProblem._evaluate.
  * Feeding the tuned geometry back INTO the converger is also a marked TODO:
    the converger currently reads geometry from parameters.py / Fusion, not from
    the optimiser's design vector.

The code favours readability over efficiency on purpose: simple, explicit,
easy to follow and easy to fill in later.

Run with:  python final_characteristics/optimiser.py
Requires:  pip install pymoo
"""

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap.
# optimiser.py lives in final_characteristics/, so the project root is one level
# up.  We add it to sys.path so "class_II_sizing" resolves regardless of the
# current working directory.  This is the ONLY real cross-module wiring: we pull
# the converged MTOW from the existing converger, exactly like ppe.py does.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from class_II_sizing.mtow_sizing import load_final_design_state

# pymoo: the NSGA-II implementation used in Phase 2.
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM


# ===========================================================================
# 1. DESIGN-VARIABLE REGISTRY  (DUMMY -- confirm the real tunable set + bounds)
# ===========================================================================
# These are the variables the optimiser is allowed to change in Phase 2.
# Each entry is a name mapped to its (low, high) search bounds.  The order of
# this dictionary defines the order of the design vector "x" that pymoo passes
# around, so keep it stable.
#
# The values below are PLACEHOLDERS seeded from what already lives in the
# parameter sheet (vehicle_dynamics/vd_parameters.py) and stat_long_stab_anal.py.
# Replace the set and the bounds once the team has decided what we actually tune.
DESIGN_VARIABLES = {
    "S_aw_over_S_tot": (0.25, 0.75),   # DUMMY -- aft/total wing-area split (cf. stat_long_stab_anal.py)
    "stagger_m":       (3.0, 6.0),     # DUMMY -- wing stagger [m] (sheet default 5.0)
    "x_cg_opt_m":      (2.0, 3.5),     # DUMMY -- cruise c.g. location [m] (sheet default 2.8)
    "wing_loading":    (600.0, 900.0), # DUMMY -- design-point wing loading [N/m^2] (sheet default 760)
}

# Convenience: the variable names in a fixed order (= the order of vector x).
DESIGN_VARIABLE_NAMES = list(DESIGN_VARIABLES.keys())


def design_vector_to_dict(x):
    """Turn a plain pymoo design vector x (a list/array) into a named dict.

    This keeps the rest of the code readable: instead of x[2] we write
    design_vars["x_cg_opt_m"].
    """
    named = {}
    for index, name in enumerate(DESIGN_VARIABLE_NAMES):
        named[name] = float(x[index])
    return named


def default_design_vector():
    """A reasonable starting design vector (DUMMY values).

    Used by Phase 1, which evaluates the current/default design once.  These are
    just sensible mid-range numbers; replace with the true current design point.
    """
    return {
        "S_aw_over_S_tot": 0.50,  # DUMMY
        "stagger_m":       5.0,   # DUMMY
        "x_cg_opt_m":      2.8,   # DUMMY
        "wing_loading":    760.0, # DUMMY
    }


# ===========================================================================
# 2. CONSTRAINT REGISTRY  (DUMMY ranges -- confirm with the stability/control budget)
# ===========================================================================
# A constraint says: "this computed quantity must lie between low and high".
# Each entry is (name, getter, low, high) where:
#   name   -- a human-readable label
#   getter -- a function(results) that pulls the quantity out of the analysis
#             results dict (built in evaluate_design below)
#   low    -- minimum allowed value (use -np.inf for "no lower bound")
#   high   -- maximum allowed value (use +np.inf for "no upper bound")
#
# The bounds below are PLACEHOLDERS.  They are seeded from the sign conventions
# and values already present in the code:
#   * C_M_alpha < 0          -> stable (aircraft.py print_results sign note)
#   * static margin >= 0.05  -> stat_long_stab_anal.py:  SM = 0.05
#   * c.g. ahead of allowable aft c.g. (= x_np - SM*MAC)
#   * C_n_beta > 0, C_l_beta < 0, C_n_r < 0  -> lateral/directional stability
CONSTRAINTS = [
    # name                       getter                                    low        high
    ("C_M_alpha_stable",         lambda r: r["derivatives"]["C_M_alpha"],  -np.inf,   0.0),     # DUMMY: must be negative (stable)
    ("static_margin_min",        lambda r: r["np_cg"]["static_margin"],     0.05,     np.inf),  # DUMMY: SM >= 5%
    ("cg_aft_of_fwd_limit",      lambda r: r["np_cg"]["x_cg_aft"],         -np.inf,   np.inf),  # DUMMY: placeholder, see note below
    ("C_n_beta_stable",          lambda r: r["derivatives"]["C_n_beta"],    0.0,      np.inf),  # DUMMY: weathercock stability
    ("C_l_beta_stable",          lambda r: r["derivatives"]["C_l_beta"],   -np.inf,   0.0),     # DUMMY: dihedral effect
    ("C_n_r_damping",            lambda r: r["derivatives"]["C_n_r"],      -np.inf,   0.0),     # DUMMY: yaw damping
    # NOTE: a proper c.g.-range constraint compares the loading-diagram c.g.
    # extremes against the allowable aft c.g. (= x_np - SM*MAC).  Wire that in
    # once the real c.g. range is available; the placeholder above just keeps
    # the slot.
]


def check_all_constraints(results):
    """Check every constraint against the analysis results.

    Returns a dict:
        {
          "per_constraint": { name: True/False, ... },
          "all_passed":     True/False,
        }
    A constraint passes when its value lies within [low, high].
    """
    per_constraint = {}
    for name, getter, low, high in CONSTRAINTS:
        value = getter(results)
        passed = (low <= value <= high)
        per_constraint[name] = passed

    all_passed = all(per_constraint.values())
    return {"per_constraint": per_constraint, "all_passed": all_passed}


def constraint_margins(results):
    """Constraint values expressed as pymoo "g(x) <= 0" margins.

    pymoo treats a constraint as satisfied when its value is <= 0.  For a
    quantity q that must satisfy low <= q <= high we produce two margins:
        low  - q   (<= 0 means q >= low  : OK)
        q - high   (<= 0 means q <= high : OK)
    Unbounded sides (-inf / +inf) are skipped.  The returned list lines up with
    n_constr in DesignProblem.
    """
    margins = []
    for name, getter, low, high in CONSTRAINTS:
        value = getter(results)
        if low != -np.inf:
            margins.append(low - value)   # want low - value <= 0
        if high != np.inf:
            margins.append(value - high)  # want value - high <= 0
    return margins


def number_of_margins():
    """How many g(x) margins constraint_margins() will produce (for n_constr)."""
    count = 0
    for name, getter, low, high in CONSTRAINTS:
        if low != -np.inf:
            count += 1
        if high != np.inf:
            count += 1
    return count


# ===========================================================================
# 3a. DUMMY ANALYSIS FUNCTIONS  (stand-ins for the real, not-yet-final routines)
# ===========================================================================
# These two functions are the ONLY pieces of physics the optimiser needs.  They
# are deliberately fake right now: they return plausible-looking numbers built
# from the inputs so the full Phase-1 / Phase-2 pipeline runs end-to-end today.
#
# WHEN THE REAL CODE IS READY:
#   * Replace compute_all_derivatives    with the single routine that computes
#     ALL stability & control derivatives  (cf. aircraft.Aircraft.solve, which
#     populates params.stability / params.controls).
#   * Replace compute_neutral_point_and_cg with the routine that computes the
#     neutral point and c.g. range         (cf. stat_long_stab_anal.py, which
#     computes x_np and the allowable aft c.g. = x_np - SM*MAC).
# Keep the SAME return-dict keys so the constraint registry above keeps working.

def compute_all_derivatives(mtow, design_vars):
    """DUMMY: return all stability & control derivatives for this design.

    Replace with the real single derivative-computing routine.  The real version
    will build an Aircraft from the parameter sheet, apply `design_vars`, set its
    mass to `mtow`, run .solve(), and read params.stability / params.controls.

    Returns a flat dict of derivative-name -> value.
    """
    # --- DUMMY model: tie a few derivatives loosely to the design vars so the
    # --- optimiser has a non-flat landscape to move around in. NOT PHYSICAL.
    s_aw_ratio = design_vars["S_aw_over_S_tot"]
    x_cg = design_vars["x_cg_opt_m"]

    derivatives = {
        # Longitudinal
        "C_L_alpha":   5.0,                          # DUMMY [1/rad]
        "C_M_alpha":  -0.8 + 0.5 * (x_cg - 2.8),     # DUMMY: more aft c.g. -> less stable
        "C_M_q":      -12.0,                         # DUMMY
        # Lateral / directional
        "C_n_beta":    0.10 - 0.05 * (s_aw_ratio - 0.5),  # DUMMY
        "C_l_beta":   -0.08,                         # DUMMY
        "C_n_r":      -0.20,                         # DUMMY
    }
    return derivatives


def compute_neutral_point_and_cg(mtow, design_vars):
    """DUMMY: return neutral-point and c.g.-range quantities for this design.

    Replace with the real neutral-point / c.g.-range routine (cf.
    stat_long_stab_anal.py, which computes x_np from the wing-area distribution
    and the allowable aft c.g. = x_np - SM*MAC).

    Returns a dict of named quantities used by the constraint registry.
    """
    # --- DUMMY model. NOT PHYSICAL. ---
    static_margin_target = 0.05
    s_aw_ratio = design_vars["S_aw_over_S_tot"]
    x_cg = design_vars["x_cg_opt_m"]

    x_np = 3.0 + 0.5 * s_aw_ratio   # DUMMY: neutral point drifts aft with aft-wing area
    mac = 1.262                     # DUMMY: mean aerodynamic chord [m] (sheet value)

    static_margin = (x_np - x_cg) / mac          # DUMMY normalised static margin
    allowable_aft_cg = x_np - static_margin_target * mac

    np_cg = {
        "x_np":             x_np,               # [m] neutral point from nose
        "mac":              mac,                # [m]
        "static_margin":    static_margin,      # [-]
        "x_cg_fwd":         x_cg - 0.2,          # DUMMY forward c.g. extreme [m]
        "x_cg_aft":         x_cg + 0.2,          # DUMMY aft c.g. extreme [m]
        "allowable_aft_cg": allowable_aft_cg,   # [m]
    }
    return np_cg


# ===========================================================================
# 3b. CORE EVALUATION  (shared by Phase 1 and Phase 2)
# ===========================================================================
def evaluate_design(design_vars=None):
    """Evaluate one design: converge MTOW, run analysis, check constraints.

    Steps:
      1. Run the MTOW converger and read the converged mass.
      2. Compute all derivatives and the neutral-point / c.g. quantities for
         that mass and the candidate design variables.
      3. Check every constraint.

    Returns a results dict with the converged mtow, the analysis outputs, and
    the constraint verdict (including the overall "accepted" boolean).
    """
    if design_vars is None:
        design_vars = default_design_vector()

    # --- Step 1: converge the MTOW. ---------------------------------------
    # IMPORTANT (performance + correctness, marked TODO):
    #   To make the converger actually respond to `design_vars`, the tuned
    #   geometry (wing-area split, wing loading, ...) must be pushed into the
    #   converger's inputs here BEFORE calling it.  Today the converger reads its
    #   geometry from parameters.py / Fusion, so changing design_vars does NOT
    #   yet change the converged MTOW.
    #
    #   Because of that, we use the CACHED converged mass here
    #   (force_recompute=False).  A full reconverge costs ~25 s (it runs an
    #   internal landing-gear optimisation), and since the result is identical
    #   for every candidate right now, recomputing it 100s of times would only
    #   make Phase 2 take hours for no benefit.
    #
    #   WHEN GEOMETRY FEEDBACK IS WIRED IN: set the design_vars on the converger
    #   inputs above and switch this to force_recompute=True so each candidate
    #   gets its own fresh convergence.
    design_state = load_final_design_state(force_recompute=False)
    mtow = design_state["mtow"]

    # --- Step 2: run the (dummy) analysis. --------------------------------
    derivatives = compute_all_derivatives(mtow, design_vars)
    np_cg = compute_neutral_point_and_cg(mtow, design_vars)

    results = {
        "design_vars": design_vars,
        "mtow":        mtow,
        "derivatives": derivatives,
        "np_cg":       np_cg,
    }

    # --- Step 3: check the constraints. -----------------------------------
    verdict = check_all_constraints(results)
    results["constraints"] = verdict["per_constraint"]
    results["accepted"]    = verdict["all_passed"]

    return results


def print_design_report(results):
    """Print a readable summary of one evaluated design."""
    print("-" * 60)
    print(f"  Converged MTOW : {results['mtow']:.2f} kg")
    print("  Design variables:")
    for name, value in results["design_vars"].items():
        print(f"    {name:18s}: {value:.4f}")
    print("  Constraint checks (PASS / FAIL):")
    for name, passed in results["constraints"].items():
        status = "PASS" if passed else "FAIL"
        print(f"    {name:22s}: {status}")
    overall = "ACCEPTED" if results["accepted"] else "REJECTED"
    print(f"  Overall: {overall}")
    print("-" * 60)


# ===========================================================================
# 4. PHASE 1 -- single acceptance check
# ===========================================================================
def run_phase_1():
    """Evaluate the current/default design once and report accept/reject.

    Returns True if every constraint passes, False otherwise.
    """
    print("\n=== PHASE 1: acceptance check on the current design ===")
    results = evaluate_design(default_design_vector())
    print_design_report(results)
    return results["accepted"]


# ===========================================================================
# 5. PHASE 2 -- NSGA-II optimisation (pymoo)
# ===========================================================================
class DesignProblem(Problem):
    """pymoo problem wrapper around evaluate_design().

    n_var    = number of tunable design variables (DESIGN_VARIABLES)
    n_obj    = 1  -> minimise MTOW (single objective for now)
    n_constr = number of g(x) <= 0 margins (number_of_margins())
    xl / xu  = lower / upper bounds, taken from DESIGN_VARIABLES
    """

    def __init__(self):
        lower_bounds = [DESIGN_VARIABLES[name][0] for name in DESIGN_VARIABLE_NAMES]
        upper_bounds = [DESIGN_VARIABLES[name][1] for name in DESIGN_VARIABLE_NAMES]

        super().__init__(
            n_var=len(DESIGN_VARIABLE_NAMES),
            n_obj=1,                      # <-- bump to 2 to add a second objective (see below)
            n_constr=number_of_margins(),
            xl=np.array(lower_bounds),
            xu=np.array(upper_bounds),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        # pymoo hands us a whole population X (one row per candidate). We loop
        # row by row for clarity rather than vectorising.
        objective_rows = []
        constraint_rows = []

        for x in X:
            design_vars = design_vector_to_dict(x)
            results = evaluate_design(design_vars)

            # ----- OBJECTIVE(S) -----------------------------------------
            # Single objective for now: minimise the converged MTOW.
            objectives = [results["mtow"]]
            #
            # TO ADD A SECOND OBJECTIVE (true multi-objective Pareto search):
            #   1. compute the extra objective, e.g.
            #        second = results["derivatives"]["some_quantity"]
            #   2. append it:  objectives = [results["mtow"], second]
            #   3. set n_obj=2 in __init__ above.
            # NSGA-II will then return a Pareto front instead of a single best.
            objective_rows.append(objectives)

            # ----- CONSTRAINTS (g(x) <= 0) ------------------------------
            constraint_rows.append(constraint_margins(results))

        out["F"] = np.array(objective_rows)
        out["G"] = np.array(constraint_rows)


def run_phase_2():
    """Run NSGA-II until it finds a converged, constraint-satisfying design."""
    print("\n=== PHASE 2: NSGA-II optimisation (design rejected in Phase 1) ===")

    problem = DesignProblem()

    # NSGA-II configuration. pop_size and n_gen are DUMMY -- tune for the real
    # problem (bigger = more thorough but slower).
    algorithm = NSGA2(
        pop_size=20,                                  # DUMMY
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),              # DUMMY operator settings
        mutation=PM(eta=20),                          # DUMMY
        eliminate_duplicates=True,
    )

    result = minimize(
        problem,
        algorithm,
        termination=("n_gen", 20),                    # DUMMY: 20 generations
        seed=1,
        verbose=True,
    )

    # `result.X` is the best design vector (single objective) or the Pareto set
    # (multi objective). Re-evaluate the chosen design to print a full report.
    best_x = result.X
    if best_x is None:
        print("NSGA-II did not return a feasible design. "
              "Loosen the constraints or widen the design-variable bounds.")
        return None

    # For a single-objective run pymoo returns one vector; for multi-objective it
    # returns several. Handle both by taking the first row if it is 2-D.
    best_x = np.atleast_2d(best_x)[0]
    best_design_vars = design_vector_to_dict(best_x)

    print("\nBest design found by NSGA-II:")
    best_results = evaluate_design(best_design_vars)
    print_design_report(best_results)
    return best_results


# ===========================================================================
# 6. ORCHESTRATION
# ===========================================================================
def main():
    accepted = run_phase_1()

    if accepted:
        print("\nDesign accepted in Phase 1 -- no optimisation needed.")
    else:
        run_phase_2()


if __name__ == "__main__":
    main()
