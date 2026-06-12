"""
Determine the OEI-safe longitudinal CG envelope for VTOL hover trim.

The trim problem is treated as a linear-programming feasibility problem. For
each selected one-engine-inoperative case, the script finds the minimum and
maximum longitudinal CG location for which the remaining propellers can carry
the aircraft weight while balancing lateral moment.
"""

import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from parameters import (
    A_DISK,
    ETA_ROTOR_FW_IN,
    ETA_ROTOR_FW_OUT,
    ETA_ROTOR_RW,
    FM,
    N_PROP,
    RHO_ORIGIN,
    X_ROTOR_FW,
    X_ROTOR_RW,
    Z_ROTOR_FW,
    Z_ROTOR_RW,
    AircraftParameters,
)
from class_II_sizing.mtow_sizing import converged_mass


G = 9.80665  # [m/s^2]
LP_TOL = 1e-8


def max_thrust_per_engine_from_power(p_max_total_w):
    """
    Convert the installed power limit to a per-propeller static-thrust limit.

    Inverts the static-hover momentum-theory relation used by the class-II
    power sizing (energy.py / mtow_sizing.py): P = T^(3/2) / (FM sqrt(2 rho A)),
    evaluated at sea level. No engine max-thrust figure exists anywhere else in
    the code, so the installed power (design_state["max_power_kw"], which
    already carries POWER_SAFETY_FACTOR) is the authoritative capability limit.
    """
    p_per_prop = p_max_total_w / N_PROP
    return (p_per_prop * FM) ** (2.0 / 3.0) * (2.0 * RHO_ORIGIN * A_DISK) ** (1.0 / 3.0)


def load_design_parameters():
    """
    Return an AircraftParameters sheet synced to the converged design state.

    Fills the fields that the VD parameter sheet leaves as None from their
    authoritative sources elsewhere in the code:
      - mass.mtow            <- class-II converged MTOW (class_II_sizing/mtow_sizing.py)
      - propulsion x/y/z     <- the 4+2 rotor stations used by class_II_sizing/MMOI.py
                                (parameters.py Part 2: X_ROTOR_*, ETA_ROTOR_*, nose datum)
      - propulsion.P_max_SL  <- installed power from the class-II sizing
      - max_thrust_per_engine<- P_max_SL inverted through hover momentum theory

    Propeller numbering (1-based, matches the --failed-cases CLI):
      1 fw inboard stbd, 2 fw inboard port, 3 fw outboard stbd,
      4 fw outboard port, 5 rw stbd, 6 rw port.
    """
    params = AircraftParameters()
    design_state = converged_mass()

    params.mass.mtow = design_state["mtow"]

    span = design_state["wing_geom"]["span_m"]
    stations = [
        (X_ROTOR_FW, +ETA_ROTOR_FW_IN * span / 2.0, Z_ROTOR_FW),
        (X_ROTOR_FW, -ETA_ROTOR_FW_IN * span / 2.0, Z_ROTOR_FW),
        (X_ROTOR_FW, +ETA_ROTOR_FW_OUT * span / 2.0, Z_ROTOR_FW),
        (X_ROTOR_FW, -ETA_ROTOR_FW_OUT * span / 2.0, Z_ROTOR_FW),
        (X_ROTOR_RW, +ETA_ROTOR_RW * span / 2.0, Z_ROTOR_RW),
        (X_ROTOR_RW, -ETA_ROTOR_RW * span / 2.0, Z_ROTOR_RW),
    ]
    prop = params.propulsion
    for idx, (x, y, z) in enumerate(stations, start=1):
        setattr(prop, f"x_vtol_{idx}", x)
        setattr(prop, f"y_vtol_{idx}", y)
        setattr(prop, f"z_vtol_{idx}", z)

    prop.P_max_SL = design_state["max_power_kw"] * 1000.0
    prop.max_thrust_per_engine = max_thrust_per_engine_from_power(prop.P_max_SL)

    return params

# Tweak this value manually to change the VTOL OEI allowable CG envelope.
# The margin is applied inward from both feasible CG limits using the global MAC.
VTOL_STATIC_MARGIN_GLOBAL_MAC = 0.05


def propeller_coordinates(params):
    """Return VTOL propeller coordinates from the aircraft nose datum."""
    prop = params.propulsion
    x_nose = np.array(
        [
            prop.x_vtol_1,
            prop.x_vtol_2,
            prop.x_vtol_3,
            prop.x_vtol_4,
            prop.x_vtol_5,
            prop.x_vtol_6,
        ],
        dtype=float,
    )
    y = np.array(
        [
            prop.y_vtol_1,
            prop.y_vtol_2,
            prop.y_vtol_3,
            prop.y_vtol_4,
            prop.y_vtol_5,
            prop.y_vtol_6,
        ],
        dtype=float,
    )
    return x_nose, y


def derivation_coordinates(x_nose, y):
    """
    Return the propeller coordinates used in the derivation.

    The current AGENTS.md specifies the nose as the x datum and the aircraft
    symmetry line as the y datum. That coordinate system is already suitable for
    the moment-balance equations, so no rotation or translation is applied.
    """
    return x_nose.copy(), y.copy(), 0.0, 0.0


def side_capacity_diagnostic(failed_index, y_deriv, weight, t_max):
    """
    Return a quick lateral-balance capacity check for symmetric layouts.

    With y_CG = 0 and propellers at symmetric +/- y stations, the positive-y and
    negative-y sides must each supply W/2. If one side has insufficient capacity,
    the LP must be infeasible before longitudinal CG is even considered.
    """
    positive_side = np.where(y_deriv > LP_TOL)[0]
    negative_side = np.where(y_deriv < -LP_TOL)[0]
    if len(positive_side) == 0 or len(negative_side) == 0:
        return None

    positive_capacity = sum(
        t_max for idx in positive_side if idx != failed_index
    )
    negative_capacity = sum(
        t_max for idx in negative_side if idx != failed_index
    )
    required_per_side = 0.5 * weight

    return {
        "required_per_side": required_per_side,
        "positive_capacity": positive_capacity,
        "negative_capacity": negative_capacity,
        "positive_ok": positive_capacity >= required_per_side - LP_TOL,
        "negative_ok": negative_capacity >= required_per_side - LP_TOL,
    }


def default_failed_cases(x_deriv):
    """Use one representative propeller from the most forward and aft stations."""
    min_x = np.min(x_deriv)
    max_x = np.max(x_deriv)
    forward_case = int(np.where(np.isclose(x_deriv, min_x))[0][0])
    aft_case = int(np.where(np.isclose(x_deriv, max_x))[0][0])
    return sorted(set([forward_case, aft_case]))


def solve_with_scipy(c, a_eq, b_eq, bounds):
    """Solve a linear program with scipy.optimize.linprog."""
    try:
        from scipy.optimize import linprog
    except ImportError:
        return None

    result = linprog(
        c=c,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return {
            "success": False,
            "message": result.message,
            "objective": None,
            "thrust": None,
            "solver": "scipy.optimize.linprog",
        }
    return {
        "success": True,
        "message": result.message,
        "objective": float(result.fun),
        "thrust": result.x,
        "solver": "scipy.optimize.linprog",
    }


def solve_by_vertex_enumeration(c, a_eq, b_eq, bounds):
    """
    Fallback LP solver for this small bounded problem.

    The feasible set has six thrust variables and two equality constraints.
    Vertices are found by fixing enough variables at their bounds, then solving
    the remaining equality system.
    """
    n_var = len(c)
    n_eq = len(b_eq)
    n_fixed = n_var - n_eq
    best_objective = None
    best_thrust = None

    for fixed_indices in combinations(range(n_var), n_fixed):
        free_indices = [idx for idx in range(n_var) if idx not in fixed_indices]
        for fixed_at_upper in product([False, True], repeat=n_fixed):
            thrust = np.zeros(n_var)
            for idx, at_upper in zip(fixed_indices, fixed_at_upper):
                lower, upper = bounds[idx]
                thrust[idx] = upper if at_upper else lower

            rhs = b_eq - a_eq[:, fixed_indices] @ thrust[list(fixed_indices)]
            a_free = a_eq[:, free_indices]
            try:
                free_solution = np.linalg.solve(a_free, rhs)
            except np.linalg.LinAlgError:
                continue

            thrust[free_indices] = free_solution
            if not bounds_are_satisfied(thrust, bounds):
                continue
            if not np.allclose(a_eq @ thrust, b_eq, atol=1e-7, rtol=0.0):
                continue

            objective = float(c @ thrust)
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_thrust = thrust.copy()

    if best_thrust is None:
        return {
            "success": False,
            "message": "No feasible vertex found by fallback enumeration.",
            "objective": None,
            "thrust": None,
            "solver": "fallback vertex enumeration",
        }
    return {
        "success": True,
        "message": "Optimization succeeded.",
        "objective": best_objective,
        "thrust": best_thrust,
        "solver": "fallback vertex enumeration",
    }


def bounds_are_satisfied(thrust, bounds):
    """Check thrust lower and upper bounds with a small numerical tolerance."""
    for value, (lower, upper) in zip(thrust, bounds):
        if value < lower - LP_TOL or value > upper + LP_TOL:
            return False
    return True


def solve_linear_program(c, a_eq, b_eq, bounds):
    """Use SciPy when available; otherwise use the local vertex fallback."""
    result = solve_with_scipy(c, a_eq, b_eq, bounds)
    if result is not None:
        return result
    return solve_by_vertex_enumeration(c, a_eq, b_eq, bounds)


def solve_failure_case(failed_index, x_deriv, y_deriv, weight, t_max):
    """Solve min and max longitudinal CG for one failed propeller."""
    n_prop = len(x_deriv)
    bounds = [(0.0, float(t_max)) for _ in range(n_prop)]
    bounds[failed_index] = (0.0, 0.0)

    a_eq = np.vstack([np.ones(n_prop), y_deriv])
    b_eq = np.array([weight, 0.0])

    min_result = solve_linear_program(x_deriv, a_eq, b_eq, bounds)
    max_result = solve_linear_program(-x_deriv, a_eq, b_eq, bounds)

    if max_result["success"]:
        max_result = dict(max_result)
        max_result["objective"] = -max_result["objective"]

    return min_result, max_result


def minimum_required_t_max_for_oei(y_deriv, weight):
    """
    Estimate the minimum per-propeller thrust needed for lateral OEI balance.

    This assumes one common T_max for all propellers and the current symmetric
    +/- y layout.
    """
    positive_side_count = int(np.sum(y_deriv > LP_TOL))
    negative_side_count = int(np.sum(y_deriv < -LP_TOL))
    limiting_side_after_failure = min(positive_side_count, negative_side_count) - 1
    if limiting_side_after_failure <= 0:
        return None
    return 0.5 * weight / limiting_side_after_failure


def normalized_by_front_wing_mac(x_nose, params):
    """Convert a nose-datum CG location to (x - LEMAC_fw) / MAC_fw."""
    wing = params.wing_geometry
    return (x_nose - wing.x_LEMAC_fw) / wing.MAC_fw


def global_mac(params):
    """Return the global MAC used for static-margin offsets."""
    wing = params.wing_geometry
    if wing.MAC_ref is not None:
        return wing.MAC_ref
    return (wing.S_fw * wing.MAC_fw + wing.S_aw * wing.MAC_aw) / (wing.S_fw + wing.S_aw)


def cg_envelope_with_static_margin(envelope_min_nose, envelope_max_nose, params):
    """Apply the VTOL OEI static-margin buffer to the feasible CG envelope."""
    margin_distance = VTOL_STATIC_MARGIN_GLOBAL_MAC * global_mac(params)
    allowable_min_nose = envelope_min_nose + margin_distance
    allowable_max_nose = envelope_max_nose - margin_distance
    available_width = envelope_max_nose - envelope_min_nose
    suggested_static_margin = None

    if allowable_min_nose > allowable_max_nose + LP_TOL:
        suggested_static_margin = max(0.0, 0.5 * available_width / global_mac(params))

    return {
        "static_margin": VTOL_STATIC_MARGIN_GLOBAL_MAC,
        "global_mac": global_mac(params),
        "margin_distance": margin_distance,
        "x_min_nose": allowable_min_nose,
        "x_max_nose": allowable_max_nose,
        "x_min_mac": normalized_by_front_wing_mac(allowable_min_nose, params),
        "x_max_mac": normalized_by_front_wing_mac(allowable_max_nose, params),
        "is_valid": allowable_min_nose <= allowable_max_nose + LP_TOL,
        "suggested_static_margin": suggested_static_margin,
    }


def print_thrust_vector(thrust):
    """Print thrust in Newton for each propeller."""
    for idx, value in enumerate(thrust, start=1):
        print(f"      T{idx}: {value:10.2f} N")


def get_vtol_oei_cg_envelope(failed_cases=None, check_all=False):
    """
    Return the raw and static-margin VTOL OEI CG envelopes without printing.

    failed_cases uses 1-based propeller numbers, matching the command-line
    interface. The default checks one forward and one aft representative case.
    """
    params = load_design_parameters()
    x_nose, y_nominal = propeller_coordinates(params)
    x_deriv, y_deriv, origin_x, _ = derivation_coordinates(x_nose, y_nominal)

    weight = params.mass.mtow * G
    t_max = params.propulsion.max_thrust_per_engine

    if check_all:
        selected_failed_cases = list(range(len(x_deriv)))
    elif failed_cases is None:
        selected_failed_cases = default_failed_cases(x_deriv)
    else:
        selected_failed_cases = [case - 1 for case in failed_cases]

    case_results = []
    for failed_index in selected_failed_cases:
        min_result, max_result = solve_failure_case(
            failed_index=failed_index,
            x_deriv=x_deriv,
            y_deriv=y_deriv,
            weight=weight,
            t_max=t_max,
        )
        if min_result["success"] and max_result["success"]:
            case_results.append(
                {
                    "failed_propeller": failed_index + 1,
                    "x_min_deriv": min_result["objective"] / weight,
                    "x_max_deriv": max_result["objective"] / weight,
                }
            )

    if not case_results:
        return None

    envelope_min_deriv = max(result["x_min_deriv"] for result in case_results)
    envelope_max_deriv = min(result["x_max_deriv"] for result in case_results)
    envelope_min_nose = envelope_min_deriv + origin_x
    envelope_max_nose = envelope_max_deriv + origin_x
    allowable_envelope = cg_envelope_with_static_margin(
        envelope_min_nose,
        envelope_max_nose,
        params,
    )

    return {
        "case_results": case_results,
        "x_min_deriv": envelope_min_deriv,
        "x_max_deriv": envelope_max_deriv,
        "x_min_nose": envelope_min_nose,
        "x_max_nose": envelope_max_nose,
        "x_min_mac": normalized_by_front_wing_mac(envelope_min_nose, params),
        "x_max_mac": normalized_by_front_wing_mac(envelope_max_nose, params),
        "allowable_with_static_margin": allowable_envelope,
    }


def analyse_cg_envelope(failed_cases=None, check_all=False):
    """Run the OEI CG-envelope calculation and print the results."""
    params = load_design_parameters()
    x_nose, y_nominal = propeller_coordinates(params)
    x_deriv, y_deriv, origin_x, origin_y = derivation_coordinates(x_nose, y_nominal)

    weight = params.mass.mtow * G
    t_max = params.propulsion.max_thrust_per_engine

    if check_all:
        failed_cases = list(range(len(x_deriv)))
    elif failed_cases is None:
        failed_cases = default_failed_cases(x_deriv)
    else:
        failed_cases = [case - 1 for case in failed_cases]

    print("VTOL OEI longitudinal CG-envelope determination")
    print("------------------------------------------------")
    print(f"MTOW: {params.mass.mtow:.2f} kg")
    print(f"Weight: {weight:.2f} N")
    print(f"Maximum thrust per propeller: {t_max:.2f} N")
    print("Derivation coordinate system: nose x datum, aircraft centreline y datum")
    print(f"Coordinate offset applied: x = {origin_x:.3f} m, y = {origin_y:.3f} m")
    print(f"Failed propeller cases checked: {[idx + 1 for idx in failed_cases]}")
    required_t_max = minimum_required_t_max_for_oei(y_deriv, weight)
    if required_t_max is not None:
        print(
            "Minimum equal per-propeller T_max for lateral OEI balance: "
            f"{required_t_max:.2f} N"
        )
    print()

    case_results = []
    for failed_index in failed_cases:
        min_result, max_result = solve_failure_case(
            failed_index=failed_index,
            x_deriv=x_deriv,
            y_deriv=y_deriv,
            weight=weight,
            t_max=t_max,
        )

        print(f"Failed propeller {failed_index + 1}")
        diagnostic = side_capacity_diagnostic(failed_index, y_deriv, weight, t_max)
        if diagnostic is not None:
            print(
                "  Lateral balance capacity check: "
                f"required per side = {diagnostic['required_per_side']:.2f} N, "
                f"+y side capacity = {diagnostic['positive_capacity']:.2f} N, "
                f"-y side capacity = {diagnostic['negative_capacity']:.2f} N"
            )
        for label, result in [("Forward/minimum", min_result), ("Aft/maximum", max_result)]:
            if not result["success"]:
                print(f"  {label} limit infeasible: {result['message']}")
                continue

            x_cg_deriv = result["objective"] / weight
            x_cg_nose = x_cg_deriv + origin_x
            x_cg_mac = normalized_by_front_wing_mac(x_cg_nose, params)

            print(f"  {label} x_CG w.r.t. derivation origin: {x_cg_deriv:.4f} m")
            print(f"  {label} x_CG from nose datum: {x_cg_nose:.4f} m")
            print(f"  {label} x_CG w.r.t. LEMAC_fw / MAC_fw: {x_cg_mac:.4f}")
            print(f"  Solver: {result['solver']}")
            print("  Thrust distribution:")
            print_thrust_vector(result["thrust"])

        if min_result["success"] and max_result["success"]:
            case_results.append(
                {
                    "failed_propeller": failed_index + 1,
                    "x_min_deriv": min_result["objective"] / weight,
                    "x_max_deriv": max_result["objective"] / weight,
                }
            )
        print()

    if not case_results:
        print("No feasible OEI cases were found; no envelope intersection can be reported.")
        return None

    envelope_min_deriv = max(result["x_min_deriv"] for result in case_results)
    envelope_max_deriv = min(result["x_max_deriv"] for result in case_results)
    envelope_min_nose = envelope_min_deriv + origin_x
    envelope_max_nose = envelope_max_deriv + origin_x
    envelope_min_mac = normalized_by_front_wing_mac(envelope_min_nose, params)
    envelope_max_mac = normalized_by_front_wing_mac(envelope_max_nose, params)
    allowable_envelope = cg_envelope_with_static_margin(
        envelope_min_nose,
        envelope_max_nose,
        params,
    )

    print("Final OEI-safe longitudinal CG envelope")
    print("--------------------------------------")
    if envelope_min_deriv > envelope_max_deriv + LP_TOL:
        print("The individual failure envelopes do not overlap.")
    print(
        "Derivation coordinates: "
        f"{envelope_min_deriv:.4f} m <= x_CG <= {envelope_max_deriv:.4f} m"
    )
    print(
        "Nose datum: "
        f"{envelope_min_nose:.4f} m <= x_CG <= {envelope_max_nose:.4f} m"
    )
    print(
        "LEMAC_fw-normalized by MAC_fw: "
        f"{envelope_min_mac:.4f} <= x_CG/MAC_fw <= {envelope_max_mac:.4f}"
    )
    print()
    print("Allowable CG envelope with VTOL static margin")
    print("--------------------------------------------")
    print(
        f"Static margin: {allowable_envelope['static_margin']:.4f} "
        f"global MAC = {allowable_envelope['margin_distance']:.4f} m"
    )
    if not allowable_envelope["is_valid"]:
        print("The requested static margin is larger than the feasible CG envelope.")
        print(
            "A possible maximum value is approximately "
            f"{allowable_envelope['suggested_static_margin']:.4f} global MAC."
        )
    print(
        "Nose datum: "
        f"{allowable_envelope['x_min_nose']:.4f} m <= x_CG <= "
        f"{allowable_envelope['x_max_nose']:.4f} m"
    )
    print(
        "LEMAC_fw-normalized by MAC_fw: "
        f"{allowable_envelope['x_min_mac']:.4f} <= x_CG/MAC_fw <= "
        f"{allowable_envelope['x_max_mac']:.4f}"
    )

    return {
        "case_results": case_results,
        "x_min_deriv": envelope_min_deriv,
        "x_max_deriv": envelope_max_deriv,
        "x_min_nose": envelope_min_nose,
        "x_max_nose": envelope_max_nose,
        "x_min_mac": envelope_min_mac,
        "x_max_mac": envelope_max_mac,
        "allowable_with_static_margin": allowable_envelope,
    }


def parse_failed_cases(raw_cases):
    """Parse comma-separated 1-based failed propeller numbers."""
    if raw_cases is None:
        return None
    cases = []
    for raw_case in raw_cases.split(","):
        case = int(raw_case.strip())
        if case < 1 or case > 6:
            raise ValueError("Failed propeller cases must be between 1 and 6.")
        cases.append(case)
    return cases


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Determine the VTOL OEI-safe longitudinal CG envelope."
    )
    parser.add_argument(
        "--failed-cases",
        help="Comma-separated 1-based propeller failures to check, e.g. '1,3'.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all six single-propeller failure cases.",
    )
    args = parser.parse_args()

    analyse_cg_envelope(
        failed_cases=parse_failed_cases(args.failed_cases),
        check_all=args.all,
    )


if __name__ == "__main__":
    main()
