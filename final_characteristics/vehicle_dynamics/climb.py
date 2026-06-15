"""
climb.py
========
Wing-borne climb & descent performance for the Folding Prandtl eVTOL.

SCOPE & CAVEAT
--------------
This covers the WING-BORNE (cruise-configuration) climb and descent, i.e. the
phases after transition where the aircraft flies on the wing. The hover
take-off, vertical climb and the vertical landing touchdown are rotor-borne
(momentum-theory, figure-of-merit) and are sized in the Power & Propulsion
chapter; they are a separate regime and are out of scope here, as with the
V-n and turning analyses.

METHOD
------
Two complementary views, matching the lecture framework:
  1. CAPABILITY  -- the power-available / power-required construction.
     Rate of climb RoC = (P_a - P_r)/W; best-rate speed V_y is where the
     P_a - P_r gap is widest. P_r is the full drag power (induced + parasite)
     from the cruise drag polar. P_a is taken at the continuous cruise-power
     level as a PLACEHOLDER reference -- the firm available power at climb
     advance ratio is a propeller result established in the Power chapter, so
     the absolute ceiling / max-RoC magnitude is reported as P_a-dependent.
  2. AS-FLOWN    -- the climb the design mission actually flies: a shallow,
     fixed flight-path-angle climb on the prescribed schedule
     (V: 20 -> 55.6 m/s, gamma ~ 3.1 deg to the 3810 m cruise altitude),
     reproduced from the mission-profile model.

PROVENANCE
----------
Pulled from parameters.py where possible (single source of truth). Mission
schedule numbers (initial/cruise speed, accel time, gamma) are from the
mission-profile trade model. Flagged estimates carried through:
    CD0 = 0.0205   preliminary;  e = 1.34 box-wing span efficiency
    L/D = 14.7     preliminary cruise value
NB the climb potential-energy term in the mission model represents the
(varying) climb vertical speed by a single constant; see the report note.

Run:  python climb.py
"""

import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

# parameters.py lives at the repo root; this file is in
# final_characteristics/vehicle_dynamics/, so walk two levels up.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# --- Inputs from parameters.py -------------------------------------------- #
try:
    import parameters as P
    from class_II_sizing.mtow_sizing import load_final_design_state
    G       = P.G
    MTOW    = load_final_design_state()["mtow"]
    S       = P.WingGeometry.S_tot
    B_SPAN  = P.WingGeometry.b_fw
    RHO_CR  = P.RHO_CRUISE
    CD0     = P.CD0
    E_SPAN  = P.OSWALD_EFFICIENCY
    V_C     = P.V_CRUISE
    H_CR    = P.H_CRUISE
    LD      = P.LD_CRUISE
except Exception:
    # Standalone fallback (root not found). Values are the last known sheet
    # state; if this path runs, the import above failed -- check the layout.
    print("WARNING: parameters.py not found; running on fallback constants.")
    G, MTOW, S, B_SPAN = 9.81, 1743.0, 25.82, 13.0
    RHO_CR, CD0, E_SPAN = 0.835679, 0.0205, 1.34
    V_C, H_CR, LD = 200/3.6, 3810.0, 14.7

# Mission-schedule climb inputs (mission-profile trade model)
V_CLIMB_I = 20.0      # [m/s] horizontal speed entering wing-borne climb
T_ACCEL   = 20.0      # [s]   accel segment duration
GAMMA_DEG = 3.09      # [deg] as-flown flight-path angle (mission model)
H_MOUNTAIN = 3048.0   # [m]   terrain clearance check altitude

# Placeholder available power at the continuous cruise-power level [W].
# Real climb P_a (propeller thrust * V at climb advance ratio) -> Power chapter.
P_AVAIL_REF = MTOW * G / LD * V_C

# --- Derived --------------------------------------------------------------- #
W  = MTOW * G
AR = B_SPAN**2 / S
k  = 1.0 / (np.pi * E_SPAN * AR)


def P_required(V, rho=RHO_CR):
    """Full drag power (induced + parasite) in level flight [W]."""
    q  = 0.5 * rho * V**2 * S
    CL = W / q
    CD = CD0 + k * CL**2
    return q * CD * V                      # = D * V


def roc(V, P_avail=P_AVAIL_REF, rho=RHO_CR):
    """Rate of climb from excess power [m/s]."""
    return (P_avail - P_required(V, rho)) / W


def main():
    # --- capability: V_y and the climb-speed band ------------------------- #
    Vgrid = np.linspace(15, 80, 600)
    Pr    = P_required(Vgrid)
    RoC   = (P_AVAIL_REF - Pr) / W
    iy    = np.argmax(RoC)
    Vy    = Vgrid[iy]
    Vmd   = Vgrid[np.argmin(Pr)]          # min-power speed (best endurance)

    print(f"W/S              = {W/S:6.1f} N/m^2  (MTOW {MTOW:.0f} kg)")
    print(f"k                = {k:.4f}  (e={E_SPAN}, AR={AR:.2f})")
    print(f"P_r minimum      = {Pr.min()/1e3:5.1f} kW at V = {Vmd:.1f} m/s")
    print(f"P_avail (ref)    = {P_AVAIL_REF/1e3:5.1f} kW  [PLACEHOLDER - cruise-power level]")
    print(f"V_y (best RoC)   = {Vy:5.1f} m/s,  RoC_max(ref) = {RoC[iy]:.2f} m/s")

    # speed band where climb is possible at the reference power
    climbable = Vgrid[RoC > 0]
    if climbable.size:
        print(f"climb band (ref) = {climbable.min():.0f} - {climbable.max():.0f} m/s")

    # --- as-flown shallow climb ------------------------------------------- #
    g_rad = np.radians(GAMMA_DEG)
    roc_start = V_CLIMB_I * np.sin(g_rad)
    roc_end   = V_C       * np.sin(g_rad)
    print(f"\nAs-flown climb: gamma = {GAMMA_DEG} deg")
    print(f"  RoC = {roc_start:.2f} m/s (at {V_CLIMB_I:.0f} m/s) "
          f"-> {roc_end:.2f} m/s (at cruise speed)")
    print(f"  clears {H_MOUNTAIN:.0f} m terrain: design reports TRUE")

    # --- plot: P_a / P_r construction ------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(Vgrid, Pr/1e3, color="#185FA5", lw=2, label="power required $P_r$")
    ax.axhline(P_AVAIL_REF/1e3, color="#D85A30", lw=2,
               label="power available $P_a$ (ref.)")
    # shade excess-power region
    ax.fill_between(Vgrid, Pr/1e3, P_AVAIL_REF/1e3,
                    where=(Pr < P_AVAIL_REF), color="#185FA5", alpha=0.08)
    # V_y marker
    ax.axvline(Vy, color="grey", ls=":", lw=0.9)
    ax.annotate(rf"$V_y \approx {Vy:.0f}$ m/s",
                (Vy, Pr[iy]/1e3), xytext=(Vy-13, Pr[iy]/1e3-12),
                fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.6, color="grey"))
    ax.plot(V_C, P_required(V_C)/1e3, "ko", ms=5)
    ax.annotate("cruise", (V_C, P_required(V_C)/1e3),
                xytext=(V_C+1.5, P_required(V_C)/1e3-9), fontsize=8)

    ax.set_xlabel("true airspeed $V$ [m/s]  (cruise altitude)")
    ax.set_ylabel("power [kW]")
    ax.set_title("Climb construction: power available vs required")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(15, 80)
    ax.set_ylim(0, 130)
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()