"""
stability_eval.py
=================

Shared, pymoo-free stability evaluation core + the single source of truth for the
stability GOALS.

WHY THIS FILE EXISTS
--------------------
Two consumers need the exact same "given a design, does it meet the stability
requirements?" calculation:
  * optimiser.py            -> uses it inside the NSGA-II loop
  * stability_sensitivity.py -> uses it as a standalone graph consult

Keeping the evaluation and the goal list here means the goals are defined ONCE.
This module deliberately contains NO optimiser machinery (no pymoo) so importing
it is cheap and side-effect free.

WHAT IT COMPUTES
----------------
evaluate_stability(param_overrides, design_vars) builds an aircraft from the
converged design state, applies any parameter overrides / design variables, then:
  * runs aircraft.Aircraft.solve()  -> ALL stability & control derivatives
    (this is the same path main_aircraft() uses, but on OUR overridable params,
    so the derivatives RESPOND to the overrides),
  * mirrors stat_long_stab_anal_func.analyse_configuration() WITHOUT its
    raise-on-violation -> neutral point + cruise & VTOL-OEI c.g. envelopes as
    plain numbers,
  * checks every requirement in STABILITY_REQUIREMENTS and returns booleans,
    signed margins, and an aggregate total-violation score.

THE STABILITY GOALS (sign conventions, from the team)
-----------------------------------------------------
    C_M_alpha<0, C_L_q>0, C_M_q<0, C_Y_p<0, C_L_p<0, C_N_p<0,
    C_Y_beta<0, C_L_beta<0, C_N_beta>0, C_Y_r>0, C_L_r>0, C_N_r<0
plus the c.g.-envelope goals from analyse_configuration:
    cruise c.g. within [forward, aft] limit, VTOL-OEI c.g. within the allowable
    envelope (the aft cruise limit already enforces the NP-consistent C_M_alpha<0
    with static margin).

Readability over efficiency throughout.
"""

import sys
from dataclasses import fields
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap: project root (for class_II_sizing / parameters) and the
# vehicle_dynamics directory (because aircraft / stat_long_stab_anal_func /
# VTOL_cg_envelope_det import each other with bare names).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VEHICLE_DYNAMICS_DIR = PROJECT_ROOT / "final_characteristics" / "vehicle_dynamics"
for _path in (PROJECT_ROOT, VEHICLE_DYNAMICS_DIR):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from class_II_sizing.mtow_sizing import load_final_design_state
from class_II_sizing import mtow_sizing
import class_II_sizing.mass_components as mass_components

import aircraft
from stat_long_stab_anal_func import (
    DEFAULT_CRUISE_STATIC_MARGIN,
    DEFAULT_S_AFT_TO_S_TOTAL,
    _as_real,
    _update_aircraft_for_wing_split,
    _vtol_oei_envelope_for_params,
)
from VTOL_cg_envelope_det import load_design_parameters

# Large penalty used when the VTOL OEI envelope cannot be computed (an OEI case
# is infeasible). It pushes the allowable limits out of reach so both VTOL c.g.
# requirements fail with a big, finite violation.
_INFEASIBLE_PENALTY = 1.0e6  # [m]


# ===========================================================================
# 0. CONVERGED MTOW WITH GEOMETRY OVERRIDES  (so MTOW responds to the roots)
# ===========================================================================
# Both consumers (optimiser.py and stability_sensitivity.py) need the converged
# MTOW to RESPOND to converger-coupled geometry roots (wing loading, span,
# taper, fuselage size, ...). This helper lives here, in the shared pymoo-free
# core, so the converger-override path is defined ONCE.
#
# Memoise on the rounded geometry tuple so identical geometries are not
# reconverged. A fresh converge runs the internal landing-gear SLSQP sizing and
# costs ~25 s, so this dominates whenever a converger-coupled root is swept.
_MTOW_CACHE = {}


def converged_mtow(geometry_overrides):
    """Return the converged MTOW [kg] for the given converger-geometry overrides.

    geometry_overrides maps a class_II_sizing.mass_components module-global name
    to its value (e.g. {"WING_LOADING_N": 820.0}). Empty -> the cached baseline
    MTOW.

    We temporarily set the mass_components globals and call the converger's
    private _solve_converged_mass(), which does NOT touch the module-level cache
    that load_final_design_state() reads, then restore the globals.
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
# 1. THE GOAL REGISTRY  (single source of truth)
# ===========================================================================
# Each requirement is (name, getter, low, high): the value returned by getter
# must satisfy  low <= value <= high.  Use -inf / +inf for an open side.
#
#   * Derivative-sign goals read a value out of results["derivatives"].
#   * c.g.-envelope goals are written as a DIFFERENCE that must stay <= 0, so the
#     per-design envelope limit can live inside the getter while the bound stays
#     a constant.
STABILITY_REQUIREMENTS = [
    # ---- Derivative sign conventions ---------------------------------------
    ("C_M_alpha", lambda r: r["derivatives"]["C_M_alpha"], -np.inf, 0.0),   # < 0
    ("C_L_q",     lambda r: r["derivatives"]["C_L_q"],      0.0,     np.inf),  # > 0
    ("C_M_q",     lambda r: r["derivatives"]["C_M_q"],     -np.inf, 0.0),   # < 0
    ("C_Y_p",     lambda r: r["derivatives"]["C_Y_p"],     -np.inf, 0.0),   # < 0
    ("C_L_p",     lambda r: r["derivatives"]["C_L_p"],     -np.inf, 0.0),   # < 0
    ("C_N_p",     lambda r: r["derivatives"]["C_N_p"],     -np.inf, 0.0),   # < 0
    ("C_Y_beta",  lambda r: r["derivatives"]["C_Y_beta"],  -np.inf, 0.0),   # < 0
    ("C_L_beta",  lambda r: r["derivatives"]["C_L_beta"],  -np.inf, 0.0),   # < 0
    ("C_N_beta",  lambda r: r["derivatives"]["C_N_beta"],   0.0,     np.inf),  # > 0
    ("C_Y_r",     lambda r: r["derivatives"]["C_Y_r"],      0.0,     np.inf),  # > 0
    ("C_L_r",     lambda r: r["derivatives"]["C_L_r"],      0.0,     np.inf),  # > 0
    ("C_N_r",     lambda r: r["derivatives"]["C_N_r"],     -np.inf, 0.0),   # < 0

    # ---- Cruise c.g. inside the static-stability envelope (differences <= 0) -
    ("cg_cruise_fwd_ok", lambda r: r["np_cg"]["cruise_forward_cg_nose"] - r["np_cg"]["cg_cruise_nose"], -np.inf, 0.0),
    ("cg_cruise_aft_ok", lambda r: r["np_cg"]["cg_cruise_nose"] - r["np_cg"]["cruise_aft_cg_nose"],     -np.inf, 0.0),

    # ---- VTOL OEI c.g. inside the allowable envelope (differences <= 0) ------
    ("cg_vtol_fwd_ok",   lambda r: r["np_cg"]["vtol_allow_min_nose"] - r["np_cg"]["cg_vtol_nose"],      -np.inf, 0.0),
    ("cg_vtol_aft_ok",   lambda r: r["np_cg"]["cg_vtol_nose"] - r["np_cg"]["vtol_allow_max_nose"],      -np.inf, 0.0),
]

REQUIREMENT_NAMES = [name for name, _getter, _low, _high in STABILITY_REQUIREMENTS]


# ===========================================================================
# 2. PARAMETER OVERRIDES  (dotted paths -> the parameter sheet / charts)
# ===========================================================================
def _set_dotted(params, charts, dotted_path, value):
    """Apply one override.

    dotted_path is either:
        "charts.<field>"        -> set on the DatcomChartInputs object, or
        "<subobject>.<field>"   -> set on params.<subobject> (e.g.
                                   "wing_geometry.dihedral_front_wing",
                                   "mass.x_cg_opt", "tail_geometry.b_vert_tail").
    The pseudo-key "s_aft_to_s_total" is NOT a sheet field; it is handled
    separately by the wing-area split and must not reach this function.
    """
    head, _, field_name = dotted_path.partition(".")
    if head == "charts":
        setattr(charts, field_name, value)
    else:
        target = getattr(params, head)
        setattr(target, field_name, value)


def _resolve_dependents(params):
    """Recompute fields that are derived from other fields.

    Dataclass defaults compute these once at class-definition time, so when we
    override a primitive (e.g. tail chord) the dependent fields (tail area, AR,
    MAC, ...) would otherwise stay stale. Recompute the common ones so a swept
    parameter stays self-consistent.
    """
    wg = params.wing_geometry
    tg = params.tail_geometry
    wl = params.winglet_geometry
    fg = params.fuselage_geometry

    # Wing: the box-wing aft panel mirrors the front panel's span and taper (the
    # sheet defines b_aw = b_fw and taper_aw = taper_fw). calc_wing_geom() reads
    # these aft fields, so a swept taper_fw / b_fw must propagate or the aft wing
    # stays stale. (No-ops at the baseline where they already match.)
    wg.b_aw = wg.b_fw
    wg.taper_aw = wg.taper_fw

    # Wing: aft-wing vertical position tracks the front wing plus the gap.
    wg.z_w_aw = wg.z_w_fw + wg.gap

    # Vertical tail: area, taper, aspect ratio, MAC from root/tip chord and span.
    tg.S_vert_tail = ((tg.c_r_vert_tail + tg.c_t_vert_tail) * tg.b_vert_tail) / 2.0
    tg.taper_vert_tail = tg.c_t_vert_tail / tg.c_r_vert_tail
    tg.AR_vert_tail = tg.b_vert_tail ** 2 / tg.S_vert_tail
    tg.MAC_vert_tail = (2.0 / 3.0) * tg.c_r_vert_tail * (
        (1 + tg.taper_vert_tail + tg.taper_vert_tail ** 2) / (1 + tg.taper_vert_tail)
    )

    # Winglet: aspect ratio and joiner sweep.
    wl.AR_winglet = wl.b_winglet ** 2 / wl.S_winglet
    wl.LE_sweep_winglet = np.arctan(wg.stagger / wg.gap)

    # Fuselage projected side area carries the vertical-tail area.
    fg.side_area = 20.0 + tg.S_vert_tail
    fg.body_depth_at_wing = fg.d_fw


# ===========================================================================
# 3. THE NEUTRAL-POINT / C.G.-ENVELOPE PART
# ===========================================================================
# Mirrors stat_long_stab_anal_func.analyse_configuration() (the cruise part and
# the VTOL OEI part) but returns the envelope limits instead of raising, so the
# caller can form numeric margins. Operates on an aircraft that already has the
# overrides + wing split applied.
def _neutral_point_and_cg(ac, wing, cg_cruise_nose, cg_vtol_nose, cruise_static_margin):
    wg = ac.params.wing_geometry
    aero = ac.params.aerodynamics

    downwash_grad = _as_real(ac.downwash_gradient(), "downwash gradient")
    aircraft_lift_curve_slope = _as_real(ac.CL_alpha_aircraft(), "CL_alpha_aircraft")
    CL_alpha_aw = aero.CL_alpha_aw
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

    # NP-consistent C_M_alpha at the cruise c.g.
    cg_cruise_front_mac = (cg_cruise_nose - wg.x_LEMAC_fw) / wing["MAC_fw"]
    moment_arm = ((cg_cruise_front_mac - x_np_front_mac) * wing["MAC_fw"]) / wing["MAC_global"]
    C_M_alpha_np = _as_real(moment_arm * aircraft_lift_curve_slope, "C_M_alpha (NP)")

    # VTOL OEI allowable envelope (penalty when infeasible).
    try:
        vtol = _vtol_oei_envelope_for_params(ac.params)
        vtol_allow_min_nose = vtol["allowable"]["x_min_nose"]
        vtol_allow_max_nose = vtol["allowable"]["x_max_nose"]
        vtol_feasible = True
    except ValueError:
        vtol_allow_min_nose = +_INFEASIBLE_PENALTY
        vtol_allow_max_nose = -_INFEASIBLE_PENALTY
        vtol_feasible = False

    return {
        "C_M_alpha_np": C_M_alpha_np,
        "x_np_nose": x_np_nose,
        "x_np_front_mac": x_np_front_mac,
        "mac_global": wing["MAC_global"],
        "cruise_forward_cg_nose": cruise_forward_cg_nose,
        "cruise_aft_cg_nose": cruise_aft_cg_nose,
        "cg_cruise_nose": cg_cruise_nose,
        "cg_vtol_nose": cg_vtol_nose,
        "vtol_feasible": vtol_feasible,
        "vtol_allow_min_nose": vtol_allow_min_nose,
        "vtol_allow_max_nose": vtol_allow_max_nose,
    }


# ===========================================================================
# 4. THE MAIN ENTRY POINT
# ===========================================================================
def evaluate_stability(param_overrides=None, design_vars=None, mtow=None):
    """Evaluate the stability requirements for one design.

    Arguments:
        param_overrides: dict {dotted_path: value} of sheet/chart overrides
            (e.g. {"wing_geometry.dihedral_front_wing": 5.0}). The pseudo-key
            "s_aft_to_s_total" sets the aft/total wing-area split.
        design_vars: dict of higher-level design variables. Recognised keys:
            "s_aft_to_s_total", "x_cg" (sets both cruise & VTOL c.g.),
            "cg_cruise_nose", "cg_vtol_nose". Anything else is ignored here.
        mtow: optional converged MTOW [kg]. Defaults to the cached converged
            value (the sign requirements are essentially mass-independent, so the
            sensitivity consult holds it fixed for speed).

    Returns a dict:
        {mtow, derivatives, np_cg, requirements: {name: bool},
         margins: {name: signed worst-side margin}, total_violation: float}
    """
    param_overrides = dict(param_overrides) if param_overrides else {}
    design_vars = dict(design_vars) if design_vars else {}

    if mtow is None:
        mtow = load_final_design_state()["mtow"]

    # --- Resolve the wing-area split and the c.g. positions ---------------
    # design_vars take precedence; "s_aft_to_s_total" may also arrive via
    # param_overrides for convenience.
    s_aft = design_vars.get(
        "s_aft_to_s_total",
        param_overrides.pop("s_aft_to_s_total", DEFAULT_S_AFT_TO_S_TOTAL),
    )

    # --- Build the parameter sheet (converged MTOW + propulsion + inertias) -
    params = load_design_parameters()
    charts = aircraft.DatcomChartInputs()

    # --- Apply parameter overrides, then fix up co-dependent fields --------
    for dotted_path, value in param_overrides.items():
        _set_dotted(params, charts, dotted_path, value)
    _resolve_dependents(params)

    # c.g. positions: design_vars override the sheet default x_cg_opt.
    default_cg = params.mass.x_cg_opt
    cg_cruise_nose = design_vars.get("cg_cruise_nose", design_vars.get("x_cg", default_cg))
    cg_vtol_nose = design_vars.get("cg_vtol_nose", design_vars.get("x_cg", default_cg))
    cruise_static_margin = design_vars.get("cruise_static_margin", DEFAULT_CRUISE_STATIC_MARGIN)

    # --- Build the aircraft and apply the wing-area split ------------------
    physical = aircraft.Physical()
    fc = aircraft.FlightCondition()
    ac = aircraft.Aircraft(params, physical, fc, charts)
    wing = _update_aircraft_for_wing_split(ac, mtow, s_aft)

    # --- Derivatives (override-aware): solve() fills stability + controls --
    solved = ac.solve()
    derivatives = {}
    for group in (solved.stability, solved.controls):
        for f in fields(group):
            value = getattr(group, f.name)
            if value is not None:
                derivatives[f.name] = float(value)

    # --- Neutral point / c.g. envelopes ------------------------------------
    np_cg = _neutral_point_and_cg(ac, wing, cg_cruise_nose, cg_vtol_nose, cruise_static_margin)

    results = {"mtow": mtow, "derivatives": derivatives, "np_cg": np_cg}

    # --- Requirements: booleans, margins, aggregate violation --------------
    results["requirements"] = check_requirements(results)
    results["margins"] = requirement_margins(results)
    results["total_violation"] = total_violation(results)
    results["accepted"] = all(results["requirements"].values())
    return results


# ===========================================================================
# 5. REQUIREMENT HELPERS  (booleans + continuous outputs for the consult)
# ===========================================================================
def check_requirements(results):
    """Return {requirement_name: True/False}. True when low <= value <= high."""
    verdict = {}
    for name, getter, low, high in STABILITY_REQUIREMENTS:
        value = getter(results)
        verdict[name] = bool(low <= value <= high)
    return verdict


def requirement_margins(results):
    """Return {name: signed margin}, where margin <= 0 means SATISFIED.

    For a value v that must lie in [low, high] the margin is the worst (largest)
    of (low - v) and (v - high); open sides are skipped. This is the continuous
    output the sensitivity consult decomposes per requirement.
    """
    margins = {}
    for name, getter, low, high in STABILITY_REQUIREMENTS:
        value = getter(results)
        sides = []
        if low != -np.inf:
            sides.append(low - value)   # <= 0 means value >= low
        if high != np.inf:
            sides.append(value - high)  # <= 0 means value <= high
        margins[name] = float(max(sides)) if sides else 0.0
    return margins


def total_violation(results):
    """Aggregate scalar: sum of POSITIVE margins (0 when every requirement passes).

    This is the single continuous output the consult's aggregate Sobol/Tornado
    decomposes to rank which parameters drive overall feasibility.
    """
    margins = requirement_margins(results)
    return float(sum(max(0.0, m) for m in margins.values()))


# ===========================================================================
# 6. STANDALONE SMOKE TEST
# ===========================================================================
def _print_report(results):
    print(f"Converged MTOW : {results['mtow']:.2f} kg")
    print("Requirement checks (PASS / FAIL):")
    for name in REQUIREMENT_NAMES:
        passed = results["requirements"][name]
        margin = results["margins"][name]
        status = "PASS" if passed else "FAIL"
        print(f"  {name:18s}: {status}   (margin {margin:+.4f})")
    print(f"Total violation: {results['total_violation']:.4f}")
    print(f"Overall: {'ACCEPTED' if results['accepted'] else 'REJECTED'}")


if __name__ == "__main__":
    print("=== Baseline design ===")
    _print_report(evaluate_stability())

    print("\n=== With 5 deg dihedral on both wings (should help C_L_beta) ===")
    _print_report(evaluate_stability(param_overrides={
        "wing_geometry.dihedral_front_wing": 5.0,
        "wing_geometry.dihedral_aft_wing": 5.0,
    }))
