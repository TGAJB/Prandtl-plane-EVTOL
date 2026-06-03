"""
main_fusion.py
MTOW sizing orchestrator.
Imports constants, component mass functions, and Fusion 360 geometry.
Run this file to execute the MTOW iteration and view the convergence plot.
"""

import numpy as np
import matplotlib.pyplot as plt
#from fontTools.misc.symfont import printGreenPen

from parameters import (
    G, RHO_ORIGIN, A_DISK, V_HOVER, T_ELAPSED_VC, V_AVG_TO,
    FM, POWER_SAFETY_FACTOR, N_MOTOR, N_PROP, N_BLADES,
    M_PAYLOAD,
    E_AL, RHO_AL, SIGMA_ALLOW_AL,
)
from class_II_sizing.energy import battery_mass
from class_II_sizing.mass_components import (
    fuselage_mass, wing_geometry, wing_mass, landing_gear_mass, tail_mass,
    motor_mass, propeller_mass, hub_mass, misc_mass, hinge_mass,
)
from fusion_geometry import (
    USE_FUSION_PROP, M_BLADE_FUSION,
    USE_FUSION_HUB,  M_HUB_FUSION,
)


# MTOW computation

def compute_mtow(mtow_kg):

    # Max installed power [kW] - sized to hover out-of-ground-effect
    avg_thrust   = mtow_kg * (G + (V_HOVER / T_ELAPSED_VC))
    v_hover_i    = np.sqrt(avg_thrust / (2 * N_PROP * RHO_ORIGIN * A_DISK))
    v_avg_i      = -V_AVG_TO / 2 + np.sqrt((V_AVG_TO / 2) ** 2 + v_hover_i ** 2)
    max_power_kw = (POWER_SAFETY_FACTOR * avg_thrust * (v_avg_i + V_AVG_TO) / FM) / 1000

    # Component masses
    wing_geom   = wing_geometry(mtow_kg)
    m_fuselage  = fuselage_mass(mtow_kg)
    m_wing      = wing_mass(mtow_kg, wing_geom)
    m_lg        = landing_gear_mass(mtow_kg, E_AL, RHO_AL, SIGMA_ALLOW_AL)
    m_tail      = tail_mass(mtow_kg)
    m_motors    = motor_mass(max_power_kw)
    m_props     = propeller_mass(max_power_kw, USE_FUSION_PROP, M_BLADE_FUSION)
    m_hubs      = hub_mass(USE_FUSION_HUB, M_HUB_FUSION)
    m_batt      = battery_mass(mtow_kg)
    m_misc      = misc_mass(mtow_kg)
    m_hinge     = hinge_mass(mtow_kg)

    mtow_new = (m_fuselage + m_wing + m_lg + m_tail
                + m_motors + m_props + m_hubs
                + M_PAYLOAD + m_batt + m_misc + m_hinge)

    prop_src = "Fusion"       if USE_FUSION_PROP else "analytical"
    hub_src  = "Fusion"       if USE_FUSION_HUB  else "not modelled"

    print(f"\n  Fuselage        : {m_fuselage:.2f} kg")
    print(f"  Wing            : {m_wing:.2f} kg")
    print(f"    Wing area     : {wing_geom['total_area_m2']:.2f} m^2")
    print(f"    Wing AR       : {wing_geom['aspect_ratio']:.2f}")
    print(f"    Root / tip c  : {wing_geom['root_chord_m']:.2f} / {wing_geom['tip_chord_m']:.2f} m")
    print(f"  Landing gear    : {m_lg:.2f} kg")
    print(f"  Tail            : {m_tail:.2f} kg")
    print(f"  Motors          : {m_motors / N_MOTOR:.2f} kg/motor  ({N_MOTOR} motors)")
    print(f"  Blades [{prop_src:10s}]: {m_props:.2f} kg  ({N_PROP} rotors x {N_BLADES} blades)")
    print(f"  Hubs   [{hub_src:10s}]: {m_hubs:.2f} kg  ({N_PROP} rotors)")
    print(f"  Battery         : {m_batt:.2f} kg")
    print(f"  Payload         : {M_PAYLOAD:.2f} kg")
    print(f"  Miscellaneous   : {m_misc:.2f} kg")
    print(f"  Hinge           : {m_hinge:.2f} kg")
    print(f"  MTOW estimate   : {mtow_new:.2f} kg\n")

    return {
        "mtow": mtow_new,
        "fuselage": m_fuselage,
        "wing": m_wing,
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
        "max_power_kw": max_power_kw
    }



# MTOW iteration

BOUND_LOW  = 1000
BOUND_HIGH = 5500

def converged_mass():
    guess      = (BOUND_LOW + BOUND_HIGH) / 2
    history    = [guess]
    converged  = False
    count      = 0

    while True:
        guess_new = compute_mtow(guess)["mtow"]
        count += 1

        if guess_new > BOUND_HIGH or guess_new < BOUND_LOW:
            print("MTOW out of bounds - check your inputs.")
            break

        if np.abs(guess_new - guess) / guess < 0.01:
            converged = True

            mtow_final = compute_mtow(guess)["mtow"]
            fuselage_final = compute_mtow(guess)["fuselage"]
            wing_final = compute_mtow(guess)["wing"]
            landing_gear_final = compute_mtow(guess)["landing_gear"]
            tail_final = compute_mtow(guess)["tail"]
            motors_final = compute_mtow(guess)["motors"]
            props_final = compute_mtow(guess)["props"]
            hubs_final = compute_mtow(guess)["hubs"]
            battery_final = compute_mtow(guess)["battery"]
            payload_final = compute_mtow(guess)["payload"]
            misc_final = compute_mtow(guess)["misc"]
            hinge_final = compute_mtow(guess)["hinge"]
            max_power_kw_final = compute_mtow(guess)["max_power_kw"]

            return {
                "mtow": mtow_final,
                "fuselage": fuselage_final,
                "wing": wing_final,
                "landing_gear": landing_gear_final,
                "tail": tail_final,
                "motors": motors_final,
                "props": props_final,
                "hubs": hubs_final,
                "battery": battery_final,
                "payload": payload_final,
                "misc": misc_final,
                "hinge": hinge_final,
                "max_power_kw": max_power_kw_final

            }

            break

        guess = guess + (guess_new - guess) / 2
        history.append(guess)

    history.append(guess_new)

    if converged:
        print(f"Converged in {count} iterations.  Final MTOW: {guess_new:.2f} kg")

        if USE_FUSION_PROP:
            print(f"  Blade source : Fusion 360 | single blade {M_BLADE_FUSION:.4f} kg"
                  f" | total {M_BLADE_FUSION * N_BLADES * N_PROP:.2f} kg")
        else:
            print(f"  Blade source : analytical regression")
        if USE_FUSION_HUB:
            print(f"  Hub source   : Fusion 360 | single hub {M_HUB_FUSION:.4f} kg"
                  f" | total {M_HUB_FUSION * N_PROP:.2f} kg")
        else:
            print(f"  Hub source   : not modelled (set to zero)")

#use this to call the mtow final pookies
MTOW_FINAL = converged_mass()["mtow"]
WING_SIZING_FINAL = wing_geometry(MTOW_FINAL)


