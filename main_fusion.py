"""
main_fusion.py
MTOW sizing orchestrator.
Imports constants, component mass functions, and Fusion 360 geometry.
Run this file to execute the MTOW iteration and view the convergence plot.
"""

import numpy as np
import matplotlib.pyplot as plt

from constants import (
    G, RHO_ORIGIN, A_DISK, V_HOVER, T_ELAPSED_VC, V_AVG_TO,
    FM, POWER_SAFETY_FACTOR, N_MOTOR, N_PROP, N_BLADES,
    M_PAYLOAD,
)
from energy import battery_mass
from mass_components import (
    fuselage_mass, wing_mass, landing_gear_mass, tail_mass,
    motor_mass, propeller_mass, hub_mass, misc_mass, hinge_mass,
)
from fusion_geometry import (
    USE_FUSION_PROP, M_BLADE_FUSION,
    USE_FUSION_HUB,  M_HUB_FUSION,
)


# ── MTOW computation ───────────────────────────────────────────────────────────

def compute_mtow(mtow_kg):

    # Max installed power [kW] — sized to hover out-of-ground-effect
    avg_thrust   = mtow_kg * (G + (V_HOVER / T_ELAPSED_VC))
    v_hover_i    = np.sqrt(avg_thrust / (2 * N_PROP * RHO_ORIGIN * A_DISK))
    v_avg_i      = -V_AVG_TO / 2 + np.sqrt((V_AVG_TO / 2) ** 2 + v_hover_i ** 2)
    max_power_kw = (POWER_SAFETY_FACTOR * avg_thrust * (v_avg_i + V_AVG_TO) / FM) / 1000

    # Component masses
    m_fuselage = fuselage_mass(mtow_kg)
    m_wing     = wing_mass()
    m_lg       = landing_gear_mass(mtow_kg)
    m_tail     = tail_mass(mtow_kg)
    m_motors   = motor_mass(max_power_kw)
    m_props    = propeller_mass(max_power_kw, USE_FUSION_PROP, M_BLADE_FUSION)
    m_hubs     = hub_mass(USE_FUSION_HUB, M_HUB_FUSION)
    m_batt     = battery_mass(mtow_kg)
    m_misc     = misc_mass(mtow_kg)
    m_hinge    = hinge_mass(mtow_kg)

    mtow_new = (m_fuselage + m_wing + m_lg + m_tail
                + m_motors + m_props + m_hubs
                + M_PAYLOAD + m_batt + m_misc + m_hinge)

    prop_src = "Fusion"       if USE_FUSION_PROP else "analytical"
    hub_src  = "Fusion"       if USE_FUSION_HUB  else "not modelled"

    print(f"\n  Fuselage        : {m_fuselage:.2f} kg")
    print(f"  Wing            : {m_wing:.2f} kg")
    print(f"  Landing gear    : {m_lg:.2f} kg")
    print(f"  Tail            : {m_tail:.2f} kg")
    print(f"  Motors          : {m_motors / N_MOTOR:.2f} kg/motor  ({N_MOTOR} motors)")
    print(f"  Blades [{prop_src:10s}]: {m_props:.2f} kg  ({N_PROP} rotors x {N_BLADES} blades)")
    print(f"  Hubs   [{hub_src:10s}]: {m_hubs:.2f} kg  ({N_PROP} rotors)")
    print(f"  Battery         : {m_batt:.2f} kg")
    print(f"  Payload         : {M_PAYLOAD:.2f} kg")
    print(f"  Miscellaneous   : {m_misc:.2f} kg")
    print(f"  Hinge           : {m_hinge:.2f} kg")
    print(f"  {'─' * 33}")
    print(f"  MTOW estimate   : {mtow_new:.2f} kg\n")

    return mtow_new


# ── MTOW iteration ─────────────────────────────────────────────────────────────

BOUND_LOW  = 1000
BOUND_HIGH = 5500
guess      = (BOUND_LOW + BOUND_HIGH) / 2
history    = [guess]
converged  = False
count      = 0

while True:
    guess_new = compute_mtow(guess)
    count += 1

    if guess_new > BOUND_HIGH or guess_new < BOUND_LOW:
        print("MTOW out of bounds — check your inputs.")
        break

    if np.abs(guess_new - guess) / guess < 0.01:
        converged = True
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


# ── Convergence plot ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("MTOW Convergence", fontweight="bold")

iters = list(range(len(history)))

axes[0].plot(iters, history, marker="o", linewidth=2, color="steelblue")
axes[0].axhline(guess_new, color="red", linestyle="--", linewidth=1,
                label=f"Converged: {guess_new:.1f} kg")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("MTOW estimate [kg]")
axes[0].set_title("MTOW per Iteration")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

residuals = [abs(history[i+1] - history[i]) / history[i] * 100
             for i in range(len(history) - 1)]
axes[1].semilogy(range(1, len(history)), residuals, marker="s", linewidth=2, color="darkorange")
axes[1].axhline(1.0, color="red", linestyle="--", linewidth=1, label="1% threshold")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Relative change [%]")
axes[1].set_title("Relative Change per Iteration")
axes[1].legend()
axes[1].grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()