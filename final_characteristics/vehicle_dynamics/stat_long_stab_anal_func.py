"""
Callable longitudinal-stability and VTOL-OEI CG-envelope analysis.

This module is intentionally plot-free. It is meant to be called from loops in
other scripts while varying MTOW, wing-area split, and the VTOL/cruise CGs.
"""

import aircraft
import numpy as np

from parameters import CRUISE_STATIC_MARGIN, G, WingGeometry
from VTOL_cg_envelope_det import (
    LP_TOL,
    cg_envelope_with_static_margin,
    default_failed_cases,
    derivation_coordinates,
    load_design_parameters,
    normalized_by_front_wing_mac,
    propeller_coordinates,
    solve_failure_case,
)


# Defaults sourced from parameters.py (the master sheet) so the analysis stays
# in sync with the design: the cruise static margin and the aft/total wing-area
# split both come from there rather than being hardcoded here.
DEFAULT_CRUISE_STATIC_MARGIN = CRUISE_STATIC_MARGIN
DEFAULT_S_AFT_TO_S_TOTAL = WingGeometry.S_aw / WingGeometry.S_tot


def calc_wing_geom(ac, S, wing):
    """Compute wing MAC, aspect ratio, and exposed area for one wing."""
    if wing == 1:
        b = ac.params.wing_geometry.b_fw
        taper = ac.params.wing_geometry.taper_fw
        d = ac.params.fuselage_geometry.d_fw
    else:
        b = ac.params.wing_geometry.b_aw
        taper = ac.params.wing_geometry.taper_aw
        d = ac.params.fuselage_geometry.d_aw

    c_r = (2*S)/(b*(1 + taper))
    MAC = (2/3)*c_r*((1 + taper + taper**2)/(1 + taper))
    A = b**2/S

    # Trapezoidal exposed area; reduces exactly to S when the carry-through
    # width d = 0 (the current aft-wing case, d_aw = 0), so it is valid for both.
    S_e = ((b - d)/2) * (c_r*(1 - (1 - taper)*(d/b)) + taper*c_r)

    return MAC, A, S_e


def _as_real(value, name):
    """Return a real scalar, or raise if a computation became complex."""
    real_value = np.real_if_close(value, tol=1000)
    if np.iscomplexobj(real_value):
        raise ValueError(f"{name} became complex: {value}")
    return float(real_value)


def _check_ratio(s_aft_to_s_total):
    if not 0.0 < s_aft_to_s_total < 1.0:
        raise ValueError("s_aft_to_s_total must be between 0 and 1.")


def _update_aircraft_for_wing_split(ac, mtow, s_aft_to_s_total):
    """
    Update the aircraft parameter object for the requested MTOW and wing split.

    The total wing area follows the design point from the matching diagram:
    W/S = wing_geometry.design_point.
    """
    _check_ratio(s_aft_to_s_total)

    ac.params.mass.mtow = mtow
    S_tot = mtow * G / ac.params.wing_geometry.design_point
    S_aw = s_aft_to_s_total * S_tot
    S_fw = S_tot - S_aw

    MAC_fw, A_fw, S_e_fw = calc_wing_geom(ac, S_fw, 1)
    MAC_aw, A_aw, S_e_aw = calc_wing_geom(ac, S_aw, 2)
    MAC_global = (S_fw*MAC_fw + S_aw*MAC_aw)/S_tot

    wg = ac.params.wing_geometry
    wg.S_tot = S_tot
    wg.S_fw = S_fw
    wg.S_aw = S_aw
    wg.MAC_fw = MAC_fw
    wg.MAC_aw = MAC_aw
    wg.MAC_ref = MAC_global
    wg.A_fw = A_fw
    wg.A_aw = A_aw
    wg.S_e_fw = S_e_fw
    wg.S_e_aw = S_e_aw
    # Non-dimensionalisation references MUST track the live geometry, otherwise
    # every aircraft-level derivative is referenced to a stale area/span. The
    # dataclass seeds (S_ref=S_tot, b_ref=b_fw) are frozen at class-definition
    # time; refresh them here so they follow the converged/overridden geometry.
    wg.S_ref = S_tot
    wg.b_ref = wg.b_fw

    # Root/tip chords follow the SAME planform identity already used above in
    # calc_wing_geom (c_r = 2S/(b(1+taper)), c_t = taper*c_r). Refresh them from
    # the live area so consumers that read the chord directly -- e.g.
    # CM_alpha_front_wing/CM_alpha_aft_wing in aircraft.py, which form c_r/MAC --
    # stay consistent with the (possibly W/S-overridden) area instead of the
    # frozen sheet chords. This is planform GEOMETRY only; no aerodynamic formula
    # is introduced or altered.
    wg.chord_fw_root = (2.0 * S_fw) / (wg.b_fw * (1.0 + wg.taper_fw))
    wg.chord_fw_tip = wg.taper_fw * wg.chord_fw_root
    wg.chord_aw_root = (2.0 * S_aw) / (wg.b_aw * (1.0 + wg.taper_aw))
    wg.chord_aw_tip = wg.taper_aw * wg.chord_aw_root

    return {
        "S_tot": S_tot,
        "S_fw": S_fw,
        "S_aw": S_aw,
        "S_aft_to_S_total": s_aft_to_s_total,
        "MAC_fw": MAC_fw,
        "MAC_aw": MAC_aw,
        "MAC_global": MAC_global,
        "A_fw": A_fw,
        "A_aw": A_aw,
        "S_e_fw": S_e_fw,
        "S_e_aw": S_e_aw,
    }


def _vtol_oei_envelope_for_params(params, failed_cases=None, check_all=False):
    """Compute the VTOL OEI CG envelope using the supplied parameter object."""
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
        if not min_result["success"] or not max_result["success"]:
            raise ValueError(
                f"VTOL OEI case for failed propeller {failed_index + 1} is infeasible."
            )

        case_results.append(
            {
                "failed_propeller": failed_index + 1,
                "x_min_deriv": min_result["objective"] / weight,
                "x_max_deriv": max_result["objective"] / weight,
            }
        )

    x_min_deriv = max(result["x_min_deriv"] for result in case_results)
    x_max_deriv = min(result["x_max_deriv"] for result in case_results)
    x_min_nose = x_min_deriv + origin_x
    x_max_nose = x_max_deriv + origin_x
    allowable = cg_envelope_with_static_margin(x_min_nose, x_max_nose, params)

    if not allowable["is_valid"]:
        raise ValueError(
            "The VTOL static margin is larger than the feasible OEI CG envelope. "
            f"Suggested maximum static margin: {allowable['suggested_static_margin']:.4f}."
        )

    return {
        "case_results": case_results,
        "feasible": {
            "x_min_nose": x_min_nose,
            "x_max_nose": x_max_nose,
            "x_min_front_mac": normalized_by_front_wing_mac(x_min_nose, params),
            "x_max_front_mac": normalized_by_front_wing_mac(x_max_nose, params),
        },
        "allowable": {
            "x_min_nose": allowable["x_min_nose"],
            "x_max_nose": allowable["x_max_nose"],
            "x_min_front_mac": allowable["x_min_mac"],
            "x_max_front_mac": allowable["x_max_mac"],
            "static_margin": allowable["static_margin"],
            "margin_distance": allowable["margin_distance"],
        },
    }


def _assert_within_envelope(value, lower, upper, value_name, envelope_name):
    if value < lower - LP_TOL or value > upper + LP_TOL:
        raise ValueError(
            f"{value_name} = {value:.4f} m is outside the {envelope_name}: "
            f"{lower:.4f} m <= x_CG <= {upper:.4f} m."
        )


def analyse_configuration(
    cg_vtol_nose,
    cg_cruise_nose,
    mtow,
    s_aft_to_s_total=DEFAULT_S_AFT_TO_S_TOTAL,
    cruise_static_margin=DEFAULT_CRUISE_STATIC_MARGIN,
    failed_cases=None,
    check_all_oei=False,
):
    """
    Analyse one aircraft configuration.

    Arguments:
        cg_vtol_nose: CG in VTOL/folded-wing configuration, from nose [m].
        cg_cruise_nose: CG in folded-out/cruise configuration, from nose [m].
        mtow: maximum take-off mass [kg].
        s_aft_to_s_total: aft wing area fraction S_aw/S_tot [-].
        cruise_static_margin: static margin used for the cruise aft-CG limit,
            applied with respect to the global MAC [-].
        failed_cases: optional 1-based list of failed propeller numbers.
        check_all_oei: if True, check all six propeller failures.

    Returns only the requested stability and CG-envelope values.
    """
    params = load_design_parameters()
    physical = aircraft.Physical()
    fc = aircraft.FlightCondition()
    charts = aircraft.DatcomChartInputs()
    ac = aircraft.Aircraft(params, physical, fc, charts)

    wing = _update_aircraft_for_wing_split(ac, mtow, s_aft_to_s_total)
    wg = ac.params.wing_geometry
    aero = ac.params.aerodynamics

    downwash_grad = _as_real(ac.downwash_gradient(), "downwash gradient")
    aircraft_lift_curve_slope = _as_real(ac.CL_alpha_aircraft(), "CL_alpha_aircraft")
    CL_alpha_aw = ac.params.aerodynamics.CL_alpha_aw
    CL_2_CL_ratio = CL_alpha_aw * (1 - downwash_grad) / aircraft_lift_curve_slope
    aft_wing_volume = (wing["S_aw"] * wg.stagger) / (wing["S_tot"] * wing["MAC_global"])

    x_np_front_mac = (
        aero.x_ac_fw_cruise / wing["MAC_fw"]
        + CL_2_CL_ratio * aft_wing_volume * (wing["MAC_global"] / wing["MAC_fw"])
    )
    x_np_front_mac = _as_real(x_np_front_mac, "neutral point normalized by front MAC")
    x_np_nose = wg.x_LEMAC_fw + x_np_front_mac * wing["MAC_fw"]

    cruise_aft_cg_nose = x_np_nose - cruise_static_margin * wing["MAC_global"]
    cruise_forward_cg_nose = wg.x_LEMAC_fw + aero.x_ac_fw_cruise
    cruise_forward_cg_front_mac = (
        cruise_forward_cg_nose - wg.x_LEMAC_fw
    ) / wing["MAC_fw"]
    cruise_aft_cg_front_mac = (
        cruise_aft_cg_nose - wg.x_LEMAC_fw
    ) / wing["MAC_fw"]

    vtol_oei_envelope = _vtol_oei_envelope_for_params(
        ac.params,
        failed_cases=failed_cases,
        check_all=check_all_oei,
    )

    _assert_within_envelope(
        cg_vtol_nose,
        vtol_oei_envelope["allowable"]["x_min_nose"],
        vtol_oei_envelope["allowable"]["x_max_nose"],
        "cg_vtol_nose",
        "VTOL OEI allowable CG envelope",
    )
    _assert_within_envelope(
        cg_cruise_nose,
        cruise_forward_cg_nose,
        cruise_aft_cg_nose,
        "cg_cruise_nose",
        "cruise static-stability CG envelope",
    )

    cg_cruise_front_mac = (cg_cruise_nose - wg.x_LEMAC_fw) / wing["MAC_fw"]
    moment_arm_np_to_cg = ((cg_cruise_front_mac - x_np_front_mac) * wing["MAC_fw"]) / wing["MAC_global"]
    C_M_alpha = _as_real(
        moment_arm_np_to_cg * aircraft_lift_curve_slope,
        "C_M_alpha",
    )

    return {
        "C_M_alpha": C_M_alpha,
        "neutral_point": {
            "x_np_nose": x_np_nose,
            "x_np_front_mac": x_np_front_mac,
        },
        "cruise_allowable_aft_cg": {
            "x_aft_nose": cruise_aft_cg_nose,
            "x_max_front_mac": cruise_aft_cg_front_mac,
        },
        "vtol_oei_trimmable_cg_envelope": {
            "x_min_nose": vtol_oei_envelope["feasible"]["x_min_nose"],
            "x_max_nose": vtol_oei_envelope["feasible"]["x_max_nose"],
            "x_min_front_mac": vtol_oei_envelope["feasible"]["x_min_front_mac"],
            "x_max_front_mac": vtol_oei_envelope["feasible"]["x_max_front_mac"],
        },
        "vtol_oei_allowable_cg_envelope": {
            "x_min_nose": vtol_oei_envelope["allowable"]["x_min_nose"],
            "x_max_nose": vtol_oei_envelope["allowable"]["x_max_nose"],
            "x_min_front_mac": vtol_oei_envelope["allowable"]["x_min_front_mac"],
            "x_max_front_mac": vtol_oei_envelope["allowable"]["x_max_front_mac"],
        },
    }


def main():
    """Run a small example when this file is executed directly."""
    params = load_design_parameters()
    try:
        result = analyse_configuration(
            cg_vtol_nose=3.35,
            cg_cruise_nose=3.35,
            mtow=params.mass.mtow,
            s_aft_to_s_total=DEFAULT_S_AFT_TO_S_TOTAL,
        )
    except ValueError as error:
        print(f"Configuration rejected: {error}")
        print(
            "Design-state inputs: "
            f"MTOW = {params.mass.mtow:.1f} kg, "
            f"T_max per propeller = {params.propulsion.max_thrust_per_engine:.1f} N "
            f"(from P_max = {params.propulsion.P_max_SL / 1000:.1f} kW installed)"
        )
        return
    print(result)


if __name__ == "__main__":
    main()
