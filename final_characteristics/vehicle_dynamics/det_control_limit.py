"""
Determine the longitudinal controllability limit for a Prandtl-plane aircraft.

The main entry point is ``determine_control_limit``.  It searches the CG range
from the neutral point to the aircraft nose and checks whether front- and
aft-wing flapelevon deflections can simultaneously trim total lift and pitching
moment.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import aircraft


G = 9.80665
DEFAULT_N_DELTA = 41
DEFAULT_N_CG = 31
DEFAULT_LIFT_TOL_FRACTION = 0.01
DEFAULT_MOMENT_TOL_FRACTION = 1.0e-3

# ---------------------------------------------------------------------------
# PLACEHOLDERS - replace with Aircraft attributes when they become available.
#
# Expected future Aircraft inputs:
#   - maximum/minimum front-wing flapelevon deflection angles [deg]
#   - maximum/minimum aft-wing flapelevon deflection angles [deg]
#   - front-wing lift-curve slope w.r.t. flapelevon deflection [1/rad]
#   - aft-wing lift-curve slope w.r.t. flapelevon deflection [1/rad]
#
# These values are deliberately kept in this script, not in aircraft.py, so the
# temporary assumptions are obvious and easy to remove later.
# ---------------------------------------------------------------------------
PLACEHOLDER_DELTA_E_FW_MIN_DEG = -25.0
PLACEHOLDER_DELTA_E_FW_MAX_DEG = 25.0
PLACEHOLDER_DELTA_E_AW_MIN_DEG = -25.0
PLACEHOLDER_DELTA_E_AW_MAX_DEG = 25.0
PLACEHOLDER_CL_DELTA_E_FW = 0.0
PLACEHOLDER_CL_DELTA_E_AW = 0.0


def _require(value, name):
    if value is None:
        raise ValueError(f"Required aircraft parameter '{name}' is None.")
    return value


def _surface_positions(ac):
    wg = ac.params.wing_geometry
    aero = ac.params.aerodynamics
    x_fw = _require(wg.x_LEMAC_fw, "wing_geometry.x_LEMAC_fw") + _require(
        aero.x_ac_fw_cruise, "aerodynamics.x_ac_fw_cruise"
    )
    x_aw = _require(wg.x_LEMAC_aw, "wing_geometry.x_LEMAC_aw") + _require(
        aero.x_ac_aw_cruise, "aerodynamics.x_ac_aw_cruise"
    )
    return x_fw, x_aw


def _reference_values(ac):
    wg = ac.params.wing_geometry
    s_ref = wg.S_ref if wg.S_ref is not None else wg.S_tot
    mac_ref = wg.MAC_ref if wg.MAC_ref is not None else wg.MAC_fw
    return _require(s_ref, "wing_geometry.S_ref/S_tot"), _require(
        mac_ref, "wing_geometry.MAC_ref/MAC_fw"
    )


def _control_input_value(obj, names, fallback):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return {"value": value, "is_placeholder": False}
    return {"value": fallback, "is_placeholder": True}


def _control_surface_inputs(ac):
    """
    Read flapelevon limits and lift powers.

    The Aircraft class does not yet expose these values. Until it does, the
    clearly marked placeholders above are used. The candidate names below make
    this function automatically pick up common future attribute names without
    changing the rest of the controllability analysis.
    """
    return {
        "delta_fw_min_deg": _control_input_value(
            ac,
            ("delta_e_fw_min_deg", "flapelevon_fw_min_deg", "max_defl_fw_min_deg"),
            PLACEHOLDER_DELTA_E_FW_MIN_DEG,
        ),
        "delta_fw_max_deg": _control_input_value(
            ac,
            ("delta_e_fw_max_deg", "flapelevon_fw_max_deg", "max_defl_fw_max_deg"),
            PLACEHOLDER_DELTA_E_FW_MAX_DEG,
        ),
        "delta_aw_min_deg": _control_input_value(
            ac,
            ("delta_e_aw_min_deg", "flapelevon_aw_min_deg", "max_defl_aw_min_deg"),
            PLACEHOLDER_DELTA_E_AW_MIN_DEG,
        ),
        "delta_aw_max_deg": _control_input_value(
            ac,
            ("delta_e_aw_max_deg", "flapelevon_aw_max_deg", "max_defl_aw_max_deg"),
            PLACEHOLDER_DELTA_E_AW_MAX_DEG,
        ),
        "cl_delta_fw": _control_input_value(
            ac,
            ("CL_delta_e_fw", "CL_delta_flapelevon_fw", "cl_delta_e_fw"),
            PLACEHOLDER_CL_DELTA_E_FW,
        ),
        "cl_delta_aw": _control_input_value(
            ac,
            ("CL_delta_e_aw", "CL_delta_flapelevon_aw", "cl_delta_e_aw"),
            PLACEHOLDER_CL_DELTA_E_AW,
        ),
    }


def _base_lift_coefficients(ac, mtow, rho, airspeed):
    wg = ac.params.wing_geometry
    aero = ac.params.aerodynamics
    s_ref, _ = _reference_values(ac)
    q_inf = 0.5 * rho * airspeed**2
    cl_total = mtow * G / (q_inf * s_ref)
    try:
        cl_alpha_aircraft = ac.CL_alpha_aircraft()
    except (AttributeError, ValueError):
        s_fw = _require(wg.S_fw, "wing_geometry.S_fw")
        s_aw = _require(wg.S_aw, "wing_geometry.S_aw")
        eta_aw = _require(aero.dyn_pres_ratio_fw_to_aw, "aerodynamics.dyn_pres_ratio_fw_to_aw")
        cl_alpha_aircraft = (
            _require(aero.CL_alpha_fw, "aerodynamics.CL_alpha_fw") * s_fw
            + _require(aero.CL_alpha_aw, "aerodynamics.CL_alpha_aw") * eta_aw * s_aw
        ) / s_ref

    alpha_fw_eff = cl_total / cl_alpha_aircraft
    downwash = ac.downwash_gradient() * alpha_fw_eff
    alpha_aw_eff = alpha_fw_eff - downwash

    return {
        "q_inf": q_inf,
        "alpha_fw_eff": alpha_fw_eff,
        "downwash": downwash,
        "alpha_aw_eff": alpha_aw_eff,
        "cl_fw": ac.CL_alpha_front_wing() * alpha_fw_eff,
        "cl_aw": ac.CL_alpha_aft_wing() * alpha_aw_eff,
        "eta_aw": _require(aero.dyn_pres_ratio_fw_to_aw, "aerodynamics.dyn_pres_ratio_fw_to_aw"),
        "S_fw": _require(wg.S_fw, "wing_geometry.S_fw"),
        "S_aw": _require(wg.S_aw, "wing_geometry.S_aw"),
        "MAC_fw": _require(wg.MAC_fw, "wing_geometry.MAC_fw"),
        "MAC_aw": _require(wg.MAC_aw, "wing_geometry.MAC_aw"),
    }


def _lift_and_moment(ac, base, controls, x_cg, delta_fw_rad, delta_aw_rad):
    aero = ac.params.aerodynamics
    q = base["q_inf"]
    eta_aw = base["eta_aw"]
    x_fw, x_aw = _surface_positions(ac)

    cl_fw = base["cl_fw"] + controls["cl_delta_fw"]["value"] * delta_fw_rad
    cl_aw = base["cl_aw"] + controls["cl_delta_aw"]["value"] * delta_aw_rad
    lift_fw = q * base["S_fw"] * cl_fw
    lift_aw = q * eta_aw * base["S_aw"] * cl_aw
    lift_total = lift_fw + lift_aw

    cm_ac_fw = _require(aero.C_M_ac_fw, "aerodynamics.C_M_ac_fw")
    cm_ac_aw = _require(aero.C_M_ac_aw, "aerodynamics.C_M_ac_aw")
    moment_fw = lift_fw * (x_fw - x_cg) + q * base["S_fw"] * base["MAC_fw"] * cm_ac_fw
    moment_aw = lift_aw * (x_aw - x_cg) + q * eta_aw * base["S_aw"] * base["MAC_aw"] * cm_ac_aw
    return lift_total, moment_fw + moment_aw


def _plot_surface(x_grid, y_grid, z_grid, title, z_title, html_path):
    fig = go.Figure(
        data=[
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                colorscale="Cividis",
                colorbar=dict(title=z_title),
                hovertemplate=(
                    "delta_fw: %{x:.2f} deg<br>"
                    "delta_aw: %{y:.2f} deg<br>"
                    f"{z_title}: " + "%{z:.4g}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis=dict(title="Front flapelevon deflection [deg]"),
            yaxis=dict(title="Aft flapelevon deflection [deg]"),
            zaxis=dict(title=z_title),
            aspectmode="manual",
            aspectratio=dict(x=1.5, y=1.5, z=1.0),
        ),
        width=1200,
        height=850,
        margin=dict(l=10, r=10, b=10, t=50),
        template="plotly_white",
    )
    fig.write_html(html_path, include_plotlyjs=True, full_html=True)


def _find_best_trim_candidate(lift_grid, moment_grid, weight, moment_scale):
    lift_error = np.abs(lift_grid - weight)
    moment_error = np.abs(moment_grid)
    score = (lift_error / weight) ** 2 + (moment_error / moment_scale) ** 2
    best_index = np.unravel_index(np.argmin(score), score.shape)
    return {
        "index": best_index,
        "lift_error": lift_error[best_index],
        "moment_error": moment_error[best_index],
        "score": score[best_index],
    }


def determine_control_limit(
    ac,
    mtow,
    neutral_point_nose,
    rho,
    airspeed,
    n_delta=DEFAULT_N_DELTA,
    n_cg=DEFAULT_N_CG,
    lift_tolerance_fraction=DEFAULT_LIFT_TOL_FRACTION,
    moment_tolerance_fraction=DEFAULT_MOMENT_TOL_FRACTION,
    make_plots=True,
    plot_dir=None,
):
    """
    Return the controllability limit from the nose and from the front-wing LEMAC.

    The returned dictionary includes the limit, per-CG search results, and the
    baseline lift split used before control-surface deflections are applied.
    """
    if n_delta < 2:
        raise ValueError("n_delta must be at least 2.")
    if n_cg < 2:
        raise ValueError("n_cg must be at least 2.")
    if neutral_point_nose < 0.0:
        raise ValueError("neutral_point_nose must be measured from the nose and be non-negative.")

    ac.params.mass.mtow = mtow
    ac.fc.rho = rho
    ac.fc.tas = airspeed

    wg = ac.params.wing_geometry
    x_lemac_fw = _require(wg.x_LEMAC_fw, "wing_geometry.x_LEMAC_fw")
    mac_fw = _require(wg.MAC_fw, "wing_geometry.MAC_fw")
    weight = mtow * G
    base = _base_lift_coefficients(ac, mtow, rho, airspeed)
    controls = _control_surface_inputs(ac)
    _, mac_ref = _reference_values(ac)
    moment_scale = weight * mac_ref

    base_lift = (
        base["q_inf"] * base["S_fw"] * base["cl_fw"]
        + base["q_inf"] * base["eta_aw"] * base["S_aw"] * base["cl_aw"]
    )
    print(f"Baseline wing lift before deflections: {100.0 * base_lift / weight:.2f}% of MTOW.")

    delta_fw_deg = np.linspace(
        controls["delta_fw_min_deg"]["value"],
        controls["delta_fw_max_deg"]["value"],
        n_delta,
    )
    delta_aw_deg = np.linspace(
        controls["delta_aw_min_deg"]["value"],
        controls["delta_aw_max_deg"]["value"],
        n_delta,
    )
    delta_fw_grid_deg, delta_aw_grid_deg = np.meshgrid(delta_fw_deg, delta_aw_deg, indexing="xy")
    delta_fw_grid_rad = np.radians(delta_fw_grid_deg)
    delta_aw_grid_rad = np.radians(delta_aw_grid_deg)

    if plot_dir is None:
        plot_dir = Path(__file__).resolve().parent / "plots" / "control_limit"
    else:
        plot_dir = Path(plot_dir)
    if make_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    cg_envelope = np.linspace(neutral_point_nose, 0.0, n_cg)
    results = []
    controllable_results = []
    uncontrollable_results = []

    for cg_index, x_cg in enumerate(cg_envelope):
        lift_grid, moment_grid = _lift_and_moment(
            ac,
            base,
            controls,
            x_cg,
            delta_fw_grid_rad,
            delta_aw_grid_rad,
        )
        best = _find_best_trim_candidate(lift_grid, moment_grid, weight, moment_scale)
        best_i, best_j = best["index"]
        is_controllable = (
            best["lift_error"] <= lift_tolerance_fraction * weight
            and best["moment_error"] <= moment_tolerance_fraction * moment_scale
        )
        result = {
            "x_cg_nose": float(x_cg),
            "x_cg_front_mac": float((x_cg - x_lemac_fw) / mac_fw),
            "is_controllable": bool(is_controllable),
            "best_delta_fw_deg": float(delta_fw_grid_deg[best_i, best_j]),
            "best_delta_aw_deg": float(delta_aw_grid_deg[best_i, best_j]),
            "best_lift_N": float(lift_grid[best_i, best_j]),
            "best_moment_Nm": float(moment_grid[best_i, best_j]),
            "lift_error_fraction": float(best["lift_error"] / weight),
            "moment_error_fraction": float(best["moment_error"] / moment_scale),
        }
        results.append(result)
        if is_controllable:
            controllable_results.append(result)
        else:
            uncontrollable_results.append(result)

        if make_plots:
            cg_tag = f"cg_{cg_index:03d}_x_{x_cg:.3f}".replace(".", "p")
            _plot_surface(
                delta_fw_grid_deg,
                delta_aw_grid_deg,
                lift_grid,
                f"Total lift surface at x_cg = {x_cg:.3f} m",
                "L_total [N]",
                plot_dir / f"{cg_tag}_lift_surface.html",
            )
            _plot_surface(
                delta_fw_grid_deg,
                delta_aw_grid_deg,
                moment_grid,
                f"Pitching moment surface at x_cg = {x_cg:.3f} m",
                "M_total [N m]",
                plot_dir / f"{cg_tag}_moment_surface.html",
            )

    if controllable_results:
        control_limit = min(controllable_results, key=lambda item: item["x_cg_nose"])
        control_limit_nose = control_limit["x_cg_nose"]
        control_limit_front_mac = control_limit["x_cg_front_mac"]
    else:
        control_limit = None
        control_limit_nose = None
        control_limit_front_mac = None

    most_aft_uncontrollable = (
        max(uncontrollable_results, key=lambda item: item["x_cg_nose"])
        if uncontrollable_results
        else None
    )

    return {
        "control_limit_nose_m": control_limit_nose,
        "control_limit_front_mac": control_limit_front_mac,
        "control_limit_case": control_limit,
        "most_aft_uncontrollable_case": most_aft_uncontrollable,
        "results": results,
        "baseline": {
            "alpha_fw_eff_rad": float(base["alpha_fw_eff"]),
            "downwash_rad": float(base["downwash"]),
            "alpha_aw_eff_rad": float(base["alpha_aw_eff"]),
            "cl_fw": float(base["cl_fw"]),
            "cl_aw": float(base["cl_aw"]),
            "baseline_lift_fraction": float(base_lift / weight),
        },
        "control_surface_inputs": {
            "delta_fw_min_deg": float(controls["delta_fw_min_deg"]["value"]),
            "delta_fw_max_deg": float(controls["delta_fw_max_deg"]["value"]),
            "delta_aw_min_deg": float(controls["delta_aw_min_deg"]["value"]),
            "delta_aw_max_deg": float(controls["delta_aw_max_deg"]["value"]),
            "CL_delta_e_fw_per_rad": float(controls["cl_delta_fw"]["value"]),
            "CL_delta_e_aw_per_rad": float(controls["cl_delta_aw"]["value"]),
            "placeholder_fields": [
                name
                for name, control_input in controls.items()
                if control_input["is_placeholder"]
            ],
        },
        "plot_dir": str(plot_dir.resolve()) if make_plots else None,
    }


def main():
    params = aircraft.AircraftParameters()
    physical = aircraft.Physical()
    fc = aircraft.FlightCondition()
    charts = aircraft.DatcomChartInputs()
    ac = aircraft.Aircraft(params, physical, fc, charts)

    neutral_point = ac.params.wing_geometry.x_LEMAC_fw + 0.5 * ac.params.wing_geometry.stagger
    result = determine_control_limit(
        ac,
        mtow=ac.params.mass.mtow,
        neutral_point_nose=neutral_point,
        rho=ac.fc.rho,
        airspeed=ac.fc.tas,
    )
    print("Controllability limit from nose [m]:", result["control_limit_nose_m"])
    print("Controllability limit from front-wing LEMAC/MAC_fw [-]:", result["control_limit_front_mac"])
    print("Most aft uncontrollable case:", result["most_aft_uncontrollable_case"])


if __name__ == "__main__":
    main()
