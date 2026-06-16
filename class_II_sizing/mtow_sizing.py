"""
mtow_sizing.py
MTOW sizing orchestrator.
Imports constants, component mass functions, and Fusion 360 geometry.
Run this file to execute the MTOW iteration and view the convergence plot.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from parameters import (
    G, RHO_ORIGIN, A_DISK, V_HOVER, T_ELAPSED_VC, V_AVG_TO,
    FM, POWER_SAFETY_FACTOR, N_MOTOR, N_PROP, N_BLADES,
    M_PAYLOAD, WINGLET_MASS_FRAC, WingGeometry,t_nonfolding
)
from class_II_sizing.energy import battery_mass
from class_II_sizing.mass_components import (
    fuselage_mass, wing_geometry, wing_mass, landing_gear_mass, tail_mass,
    motor_mass, propeller_mass, hub_mass, misc_mass, hinge_mass,
    size_wing, USE_DETAILED_WING_SIZING,
)
from fusion_geometry import (
    USE_FUSION_PROP, M_BLADE_FUSION,
    USE_FUSION_HUB, M_HUB_FUSION,
)


BOUND_LOW = 1000
BOUND_HIGH = 5500
CONVERGENCE_TOLERANCE = 0.01

_FINAL_DESIGN_STATE = None
MTOW_FINAL = None
WING_SIZING_FINAL = None


def compute_mtow(mtow_kg, verbose=True):
    # Max installed power [kW] - sized to hover out-of-ground-effect.
    avg_thrust = mtow_kg * (G + (V_HOVER / T_ELAPSED_VC))
    v_hover_i = np.sqrt(avg_thrust / (2 * N_PROP * RHO_ORIGIN * A_DISK))
    v_avg_i = -V_AVG_TO / 2 + np.sqrt((V_AVG_TO / 2) ** 2 + v_hover_i ** 2)
    max_power_kw = (POWER_SAFETY_FACTOR * avg_thrust * (v_avg_i + V_AVG_TO) / FM) / 1000

    # Component masses.
    wing_geom = wing_geometry(mtow_kg)
    m_fuselage = fuselage_mass(mtow_kg)
    lg = landing_gear_mass(mtow_kg)
    m_lg = lg["m_gear"] if lg is not None else 0.03 * mtow_kg
    m_tail = tail_mass(mtow_kg)
    m_motors = motor_mass(max_power_kw)

    # Propellers/hubs are sized before the wing because the detailed wing sizing
    # carries the rotor weights as point loads on the wing.
    m_props = propeller_mass(max_power_kw, USE_FUSION_PROP, M_BLADE_FUSION)
    m_hubs = hub_mass(USE_FUSION_HUB, M_HUB_FUSION)

    # Wing + winglet structural mass. The detailed (FEA-style) size_wing() returns
    # the wing structure and the winglet mass SEPARATELY; the analytical fallback
    # splits its single total with the legacy WINGLET_MASS_FRAC so downstream
    # (MMOI) sees the same (wing, winglet) shape either way. m_wing is the pure
    # wing structure (no winglet); m_winglet is fed to MMOI's tip plates.
    def _analytical_wing_winglet():
        total = wing_mass(mtow_kg, wing_geom)
        return (1.0 - WINGLET_MASS_FRAC) * total, WINGLET_MASS_FRAC * total

    if USE_DETAILED_WING_SIZING:
        try:
            m_wing, m_winglet = size_wing(mtow_kg, wing_geom, m_props, t_nonfolding)
            if not (np.isfinite(m_wing) and np.isfinite(m_winglet)) or m_wing <= 0.0 or m_wing > 0.6 * mtow_kg:
                raise ValueError(f"implausible size_wing result ({m_wing:.1f} kg)")
        except Exception as exc:  # never let wing sizing break the converger/optimiser
            if verbose:
                print(f"  [size_wing fallback -> analytical wing_mass: {exc}]")
            m_wing, m_winglet = _analytical_wing_winglet()
    else:
        m_wing, m_winglet = _analytical_wing_winglet()

    m_batt = battery_mass(mtow_kg)
    m_misc = misc_mass(mtow_kg)
    m_hinge = hinge_mass(mtow_kg)

    mtow_new = (
        m_fuselage
        + m_wing
        + m_winglet
        + m_lg
        + m_tail
        + m_motors
        + m_props
        + m_hubs
        + M_PAYLOAD
        + m_batt
        + m_misc
        + m_hinge
    )

    prop_src = "Fusion" if USE_FUSION_PROP else "analytical"
    hub_src = "Fusion" if USE_FUSION_HUB else "not modelled"

    if verbose:
        wing_src = "detailed size_wing" if USE_DETAILED_WING_SIZING else "analytical"
        print(f"\n  Fuselage        : {m_fuselage:.2f} kg")
        print(f"  Wing  [{wing_src:18s}]: {m_wing:.2f} kg")
        print(f"  Winglet         : {m_winglet:.2f} kg")
        print(f"    Wing area     : {wing_geom['total_area_m2']:.2f} m^2")
        print(f"    Wing AR       : {wing_geom['aspect_ratio']:.2f}")
        print(
            f"    Root / tip / mean c (MAC): "
            f"{wing_geom['root_chord_m']:.2f} / {wing_geom['tip_chord_m']:.2f} / {wing_geom['mac_m']:.2f} m"
        )
        print(f"  Landing gear    : {m_lg:.2f} kg")
        print(f"    Root / tip c  : {wing_geom['root_chord_m']:.2f} / {wing_geom['tip_chord_m']:.2f} m")
        if lg is not None:
            print(f"  Landing gear    : {m_lg:.2f} kg  [{lg['arch']}]")
            print(f"                     params={lg['params']}  (score {lg['score']:.3f})")
        else:
            print(f"  Landing gear    : {m_lg:.2f} kg  (fallback 3% MTOW – no feasible architecture)")
        print(f"  Tail            : {m_tail:.2f} kg")
        print(f"  Motors          : {m_motors / N_MOTOR:.2f} kg/motor  ({N_MOTOR} motors)")
        print(f"  Blades [{prop_src:10s}]: {m_props:.2f} kg  ({N_PROP} rotors x {N_BLADES} blades)")
        print(f"  Hubs   [{hub_src:10s}]: {m_hubs:.2f} kg  ({N_PROP} rotors)")
        print(f"  Battery         : {m_batt:.2f} kg")
        print(f"  Payload         : {M_PAYLOAD:.2f} kg")
        print(f"  Miscellaneous   : {m_misc:.2f} kg")
        print(f"  Hinge           : {m_hinge:.2f} kg")
        print(f"  {'-' * 33}")
        print(f"  MTOW estimate   : {mtow_new:.2f} kg\n")

    return {
        "mtow": mtow_new,
        "fuselage": m_fuselage,
        "wing": m_wing,
        "winglet": m_winglet,
        "landing_gear": m_lg,
        "tail": m_tail,
        "motors": m_motors,
        "props": m_props,
        "hubs": m_hubs,
        "battery": m_batt,
        "payload": M_PAYLOAD,
        "misc": m_misc,
        "hinge": m_hinge,
        "wing_geom": wing_geom,
        "max_power_kw": max_power_kw,
    }


def _solve_converged_mass(verbose=True):
    guess = (BOUND_LOW + BOUND_HIGH) / 2
    count = 0
    converged = False
    final_breakdown = None

    while True:
        breakdown = compute_mtow(guess, verbose=verbose)
        guess_new = breakdown["mtow"]
        final_breakdown = breakdown
        count += 1

        if guess_new > BOUND_HIGH or guess_new < BOUND_LOW:
            if verbose:
                print("MTOW out of bounds - check your inputs.")
            break

        if np.abs(guess_new - guess) / guess < CONVERGENCE_TOLERANCE:
            converged = True
            break

        guess = guess + (guess_new - guess) / 2

    mtow_final = guess_new
    wing_sizing_final = (
        final_breakdown["wing_geom"] if final_breakdown is not None else wing_geometry(mtow_final)
    )

    if verbose and converged:
        print(f"Converged in {count} iterations.  Final MTOW: {guess_new:.2f} kg")
        if USE_FUSION_PROP:
            print(
                f"  Blade source : Fusion 360 | single blade {M_BLADE_FUSION:.4f} kg"
                f" | total {M_BLADE_FUSION * N_BLADES * N_PROP:.2f} kg"
            )
        else:
            print("  Blade source : analytical regression")
        if USE_FUSION_HUB:
            print(
                f"  Hub source   : Fusion 360 | single hub {M_HUB_FUSION:.4f} kg"
                f" | total {M_HUB_FUSION * N_PROP:.2f} kg"
            )
        else:
            print("  Hub source   : not modelled (set to zero)")

    return {
        "mtow_final_kg": mtow_final,
        "mtow": mtow_final,
        "wing_sizing_final": wing_sizing_final,
        "fuselage": final_breakdown["fuselage"],
        "wing": final_breakdown["wing"],
        "winglet": final_breakdown["winglet"],
        "landing_gear": final_breakdown["landing_gear"],
        "tail": final_breakdown["tail"],
        "motors": final_breakdown["motors"],
        "props": final_breakdown["props"],
        "hubs": final_breakdown["hubs"],
        "battery": final_breakdown["battery"],
        "payload": final_breakdown["payload"],
        "misc": final_breakdown["misc"],
        "hinge": final_breakdown["hinge"],
        "wing_geom": wing_sizing_final,
        "max_power_kw": final_breakdown["max_power_kw"],
    }


def _write_back_wing_geometry(wing_geom):
    """Push the converged per-wing planform back into the WingGeometry dataclass so
    aero/stability/MMOI consumers read the iteratively-SIZED geometry instead of the
    hand-typed seeds. Called only for the canonical converged design (here), NOT in
    the optimiser's hot converged_mtow path, to avoid mutating shared state per-eval.
    Module-level constants snapshotted at import (e.g. WING_SPAN) are unaffected."""
    front, aft = wing_geom["front"], wing_geom["aft"]
    span = wing_geom["span_m"]
    WingGeometry.S_fw = front["area_per_wing_m2"]
    WingGeometry.S_aw = aft["area_per_wing_m2"]
    WingGeometry.S_tot = wing_geom["total_area_m2"]
    WingGeometry.b_fw = span
    WingGeometry.b_aw = span
    WingGeometry.A_fw = front["aspect_ratio"]
    WingGeometry.A_aw = aft["aspect_ratio"]
    WingGeometry.chord_fw_root = front["root_chord_m"]
    WingGeometry.chord_fw_tip = front["tip_chord_m"]
    WingGeometry.chord_aw_root = aft["root_chord_m"]
    WingGeometry.chord_aw_tip = aft["tip_chord_m"]
    WingGeometry.MAC_fw = front["mac_m"]
    WingGeometry.MAC_aw = aft["mac_m"]


def load_final_design_state(force_recompute=False, verbose=False):
    global _FINAL_DESIGN_STATE, MTOW_FINAL, WING_SIZING_FINAL

    if force_recompute or _FINAL_DESIGN_STATE is None:
        _FINAL_DESIGN_STATE = _solve_converged_mass(verbose=verbose)
        MTOW_FINAL = _FINAL_DESIGN_STATE["mtow"]
        WING_SIZING_FINAL = _FINAL_DESIGN_STATE["wing_sizing_final"]
        _write_back_wing_geometry(WING_SIZING_FINAL)

    return _FINAL_DESIGN_STATE


def converged_mass(force_recompute=False, verbose=False):
    return load_final_design_state(force_recompute=force_recompute, verbose=verbose)


def main():
    converged_mass(force_recompute=True, verbose=True)


converged_mass(verbose=False)


if __name__ == "__main__":
    main()
