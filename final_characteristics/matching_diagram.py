"""
matching_diagram.py
===================

Cruise-configuration matching diagram (power loading W/P vs wing loading W/S) for
the Prandtl-plane eVTOL, re-implemented as an ITERATIVE, fully parametric module.

WHY THIS FILE EXISTS
--------------------
The original design point W/S = 760 N/m^2 was picked once, offline, from a
matching diagram with hardcoded inputs and a QUALITATIVE "top-right corner =
lightest" rule (Midterm report S8.3). That single number is only a feasibility
upper bound, not a proven optimum. This module:

  * computes the matching-diagram constraints from the CONVERGED design state and
    parameters.py (no hardcoded aero/mission numbers),
  * makes the aspect ratio respond to W/S (A = b^2 / S, S = W / (W/S)) instead of
    a frozen A,
  * uses the project's Prandtl box-wing Oswald relation (Rizzo 2007 = report
    eq 4.7, support_files/oswaldefficiency.py) for e -- the only aero relation
    invoked, and it is Prandtl-valid,
  * performs NO qualitative point selection. It exposes the feasible W/S range and
    the power-loading constraint so the optimiser can SEARCH W/S against the real
    objectives (MTOW, stability robustness) -- see optimiser.py.

CONSTRAINT EQUATIONS
--------------------
The stall / cruise / climb-rate / climb-gradient relations are the STANDARD ADSEE
power-loading forms (verified). Take-off and landing sizing are omitted: a VTOL
does them vertically, not on a runway (report S8.3).

  stall          : W/S <= 0.5 * rho_SL * V_stall^2 * CL_max           (vertical line)
  cruise         : W/P = eta_p * sigma^(3/4) * [ CD0 0.5 rho V^3 /(W/S)
                                                 + (W/S)/(pi A e 0.5 rho V) ]^-1
  climb rate     : W/P = eta_p / [ c + sqrt(W/S) sqrt(2/rho_SL)
                                       / (1.345 (A e)^(3/4) / CD0^(1/4)) ]
  climb gradient : W/P = eta_p / [ sqrt(W/S) (c/V + CD/CL)
                                       sqrt((2/rho_SL)(1/CL)) ]

All "W/P" values are referenced to sea level (power available lapses with
density), so they compare directly against the installed (hover-sized) power.

Run:  python final_characteristics/matching_diagram.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from parameters import (
    G,
    RHO_ORIGIN,
    CD0,
    CL_MAX_OPERATIONAL,
    STALL_SPEED_MAX,
    CLIMB_RATE,
    CLIMB_GRADIENT,
    CL_MAX_CLIMB,
    PROP_EFFICIENCY,
    V_CRUISE,
    WingGeometry,
    Mission,
)
from support_files.oswaldefficiency import oswald_efficiency
from class_II_sizing.mtow_sizing import load_final_design_state

# Footprint span and vertical gap are converged-geometry roots (not W/S driven).
SPAN_M = WingGeometry.b_fw   # [m] full span (footprint constraint)
GAP_M = WingGeometry.gap     # [m] vertical gap between the box wings (Rizzo h)
RHO_CRUISE = Mission.rho_cr  # [kg/m^3] ISA density at cruise altitude (single source)

# The matching diagram hard-bounds only the TOP of W/S (stall). The bottom is a
# practical floor for the optimiser search; 0.6 brackets the report's 570 & 760
# candidates and leaves room for an interior optimum.
WING_LOADING_LOWER_FRACTION = 0.6


def _converged_mtow():
    """Converged MTOW [kg] from the class-II sizing loop (cached)."""
    return load_final_design_state(verbose=False)["mtow"]


def _oswald():
    """Prandtl box-wing Oswald factor e from the Rizzo relation (report eq 4.7)."""
    return float(oswald_efficiency(GAP_M, SPAN_M))


# ---------------------------------------------------------------------------
# Geometry coupling: AR responds to W/S (this is the iterative refinement).
# ---------------------------------------------------------------------------
def wing_area(wing_loading, mtow=None):
    """Total reference wing area [m^2] = W / (W/S)."""
    if mtow is None:
        mtow = _converged_mtow()
    return mtow * G / wing_loading


def aspect_ratio(wing_loading, mtow=None, span=SPAN_M):
    """Total-aircraft aspect ratio b^2/S at a given W/S (S = W/(W/S))."""
    return span ** 2 / wing_area(wing_loading, mtow)


# ---------------------------------------------------------------------------
# Constraints (standard ADSEE forms).
# ---------------------------------------------------------------------------
def wing_loading_stall_limit(cl_max=CL_MAX_OPERATIONAL, v_stall=STALL_SPEED_MAX,
                             rho_sl=RHO_ORIGIN):
    """Stall constraint -> hard UPPER bound on W/S [N/m^2]."""
    return cl_max * 0.5 * rho_sl * v_stall ** 2


def power_loading_cruise(wing_loading, mtow=None, eta_p=PROP_EFFICIENCY, cd0=CD0,
                         v=V_CRUISE, rho_cr=RHO_CRUISE, rho_sl=RHO_ORIGIN):
    """Cruise-speed power-loading constraint W/P [N/W], referenced to sea level."""
    a = aspect_ratio(wing_loading, mtow)
    e = _oswald()
    rho_ratio = rho_cr / rho_sl
    parasite = (cd0 * 0.5 * rho_cr * v ** 3) / wing_loading
    induced = wing_loading / (np.pi * a * e * 0.5 * rho_cr * v)
    return eta_p * rho_ratio ** 0.75 / (parasite + induced)


def power_loading_climb_rate(wing_loading, mtow=None, eta_p=PROP_EFFICIENCY,
                             cd0=CD0, climb_rate=CLIMB_RATE, rho_sl=RHO_ORIGIN):
    """Climb-rate power-loading constraint W/P [N/W] (sized at sea level)."""
    a = aspect_ratio(wing_loading, mtow)
    e = _oswald()
    denom = climb_rate + (np.sqrt(wing_loading) * np.sqrt(2.0 / rho_sl)) / (
        1.345 * (a * e) ** 0.75 / cd0 ** 0.25
    )
    return eta_p / denom


def power_loading_climb_gradient(wing_loading, mtow=None, eta_p=PROP_EFFICIENCY,
                                 cd0=CD0, climb_gradient=CLIMB_GRADIENT,
                                 cl=CL_MAX_CLIMB, rho_sl=RHO_ORIGIN):
    """Climb-gradient power-loading constraint W/P [N/W] (sized at sea level)."""
    a = aspect_ratio(wing_loading, mtow)
    e = _oswald()
    cd = cd0 + cl ** 2 / (np.pi * a * e)
    return eta_p / (
        np.sqrt(wing_loading) * (climb_gradient + cd / cl)
        * np.sqrt((2.0 / rho_sl) * (1.0 / cl))
    )


def power_loading_limit(wing_loading, mtow=None):
    """Feasible-boundary W/P [N/W] = the LOWEST (most restrictive) constraint.

    No assumption about which constraint binds (this is the logic-error fix: the
    original script hardcoded climb-rate as binding at W/S_max).
    """
    return np.minimum.reduce([
        power_loading_cruise(wing_loading, mtow),
        power_loading_climb_rate(wing_loading, mtow),
        power_loading_climb_gradient(wing_loading, mtow),
    ])


def required_power_W(wing_loading, mtow=None):
    """Minimum installed power [W] for forward flight at this W/S = W / (W/P)_limit."""
    if mtow is None:
        mtow = _converged_mtow()
    return mtow * G / power_loading_limit(wing_loading, mtow)


def cruise_lift_coefficient(wing_loading, v=V_CRUISE, rho_cr=RHO_CRUISE):
    """Cruise CL = (W/S) / (0.5 rho_cr V^2) at the given W/S."""
    return wing_loading / (0.5 * rho_cr * v ** 2)


# ---------------------------------------------------------------------------
# Optimiser interface: feasible W/S range (no qualitative point selection).
# ---------------------------------------------------------------------------
def feasible_wing_loading_range(mtow=None, lower_fraction=WING_LOADING_LOWER_FRACTION):
    """(W_S_min, W_S_max) [N/m^2] for the optimiser search.

    Upper bound = stall limit (the matching diagram's only hard W/S bound). Lower
    bound = a practical fraction of it (power feasibility is non-binding for an
    eVTOL whose installed power is hover-sized; it is enforced separately as an
    optimiser constraint via required_power_W). The selection of the operating
    W/S is left to the optimiser, against MTOW + stability robustness.
    """
    w_s_max = wing_loading_stall_limit()
    return lower_fraction * w_s_max, w_s_max


def reference_design_point(mtow=None, n=400):
    """Deterministic reference points on the feasible boundary (for reporting only;
    NOT used to select the design -- the optimiser does that).

    Returns a dict with the stall-limited max-W/S point and the global power-loading
    peak (argmax of the feasible boundary), both found quantitatively.
    """
    if mtow is None:
        mtow = _converged_mtow()
    w_s_max = wing_loading_stall_limit()
    grid = np.linspace(0.05 * w_s_max, w_s_max, n)
    wp = power_loading_limit(grid, mtow)
    peak_i = int(np.argmax(wp))
    return {
        "mtow": mtow,
        "max_wing_loading": {"W_S": w_s_max,
                             "W_P": float(power_loading_limit(w_s_max, mtow))},
        "peak_power_loading": {"W_S": float(grid[peak_i]), "W_P": float(wp[peak_i])},
    }


# ===========================================================================
# Reporting + plot
# ===========================================================================
def _print_summary(mtow=None):
    if mtow is None:
        mtow = _converged_mtow()
    w_s_min, w_s_max = feasible_wing_loading_range(mtow)
    ref = reference_design_point(mtow)
    print("=== Matching diagram (cruise configuration) ===")
    print(f"  Converged MTOW            : {mtow:.1f} kg")
    print(f"  Oswald e (Rizzo box-wing) : {_oswald():.4f}")
    print(f"  Stall W/S upper bound     : {w_s_max:.1f} N/m^2")
    print(f"  Optimiser W/S search range: [{w_s_min:.1f}, {w_s_max:.1f}] N/m^2")
    print(f"  Reference points (quantitative, not a selection):")
    p = ref["max_wing_loading"]
    print(f"    max W/S (stall)         : W/S = {p['W_S']:.1f} N/m^2,  W/P = {p['W_P']:.4f} N/W"
          f"  (A = {aspect_ratio(p['W_S'], mtow):.2f}, CL_cr = {cruise_lift_coefficient(p['W_S']):.3f})")
    p = ref["peak_power_loading"]
    print(f"    peak W/P (boundary max) : W/S = {p['W_S']:.1f} N/m^2,  W/P = {p['W_P']:.4f} N/W")
    print(f"  Required vs installed power at max W/S:")
    state = load_final_design_state(verbose=False)
    p_installed = state["max_power_kw"] * 1000.0
    p_req = float(required_power_W(w_s_max, mtow))
    print(f"    P_required (cruise/climb): {p_req/1000:.1f} kW   "
          f"P_installed (hover-sized): {p_installed/1000:.1f} kW   "
          f"-> {'FEASIBLE' if p_req <= p_installed else 'INFEASIBLE'}")


def plot(mtow=None, ax=None, show=True):
    import matplotlib.pyplot as plt

    if mtow is None:
        mtow = _converged_mtow()
    w_s_max = wing_loading_stall_limit()
    grid = np.linspace(1.0, 1.25 * w_s_max, 400)

    wp_cruise = power_loading_cruise(grid, mtow)
    wp_climb_rate = power_loading_climb_rate(grid, mtow)
    wp_climb_grad = power_loading_climb_gradient(grid, mtow)
    wp_limit = np.minimum.reduce([wp_cruise, wp_climb_rate, wp_climb_grad])
    feasible = grid <= w_s_max

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.fill_between(grid[feasible], 0, wp_limit[feasible], color="green", alpha=0.35,
                    label="Feasible region")
    ax.axvline(w_s_max, color="black", linestyle="--", label="Stall-speed limit")
    ax.plot(grid, wp_cruise, label="Cruise-speed constraint")
    ax.plot(grid, wp_climb_rate, label="Climb-rate constraint")
    ax.plot(grid, wp_climb_grad, label="Climb-gradient constraint")

    ref = reference_design_point(mtow)
    for key in ("max_wing_loading", "peak_power_loading"):
        ax.scatter(ref[key]["W_S"], ref[key]["W_P"], s=90, facecolor="yellow",
                   edgecolor="black", zorder=5)

    ax.set_xlabel(r"Wing loading $W/S$ [N/m$^2$]")
    ax.set_ylabel(r"Power loading $W/P$ [N/W]")
    ax.set_title("Matching Diagram (converged, iterative)")
    ax.grid(True)
    ax.legend(loc="upper center")
    ax.set_ylim(0, 0.4)

    if created and show:
        plt.tight_layout()
        plt.show()
    return ax


def main():
    _print_summary()
    plot()


if __name__ == "__main__":
    main()
