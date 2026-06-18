"""
vn_diagram.py
=============
V-n (manoeuvre + gust) diagram for the Folding Prandtl eVTOL, cruise /
wing-borne flight regime.

SCOPE & CAVEAT
--------------
This is the WING-BORNE manoeuvre envelope (the fixed-wing cruise regime).
It does NOT cover the hover / transition phases, whose rotor and tilt loads
are a separate structural case. Presented as the cruise manoeuvre envelope,
not the complete VTOL structural envelope.

PROVENANCE OF INPUTS
--------------------
From parameters.py + the class-II converger (load_final_design_state), sourced
at runtime so the envelope tracks the design instead of fixed numbers:
    N_W = 3.5                positive limit load factor
    MTOW (converged)         from the MTOW loop; weight W = MTOW * g
    S_tot, b_fw              box-wing reference area and span -> W/S and AR = b^2/S
    rho = RHO_ORIGIN = 1.225 sea-level density (this is an EAS envelope)
    V_cruise = 200/3.6 = 55.6 m/s   design cruise (TAS at cruise altitude); placed
                             on the diagram as EAS via sqrt(rho_cr / rho_SL)

Given / updated by the team:
    V_dive = V_DIVE_FACTOR * V_cruise        (V_DIVE_FACTOR = 1.25 from
                                             parameters.py, CS-25.335 lower bound)
    CL_max = 1.66        ESTIMATE pending the real aero polar (was 2.0)

FLAGGED ASSUMPTIONS (confirm with loads / certification basis):
    N_neg = -1.5         negative limit load factor (typical; not in params)
    Ude_cruise = 15 m/s, Ude_dive = 7.5 m/s   gust velocities (params has 0)
    lift-curve slope a   estimated from AR (finite-wing correction)
    negative stall uses the same CL_max magnitude

Run:  python vn_diagram.py
"""

import sys
import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt

# Repo root (parameters.py + the optimizer's final-design output live here).
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- Inputs ---------------------------------------------------------------- #
# MTOW and wing geometry from the OPTIMIZER's final design state
# (characteristics.json); the rest from parameters.py.
try:
    import parameters as P
    _cs = json.loads((ROOT / "final_design/results/data/characteristics.json").read_text())
    G        = P.G
    MTOW     = _cs["mass"]["mtow"]                 # optimizer final MTOW
    N_POS    = P.N_W                              # positive limit load factor; CS-23-aligned,
                                                  #   which EASA SC-VTOL (small cat.) targets
    S_W      = _cs["wing_geometry"]["total_area_m2"]  # converged wing area [m^2]
    B_SPAN   = _cs["wing_geometry"]["span_m"]         # reference span [m]
    RHO      = P.RHO_ORIGIN                        # sea-level density (EAS envelope) [kg/m^3]
    RHO_CR   = P.RHO_CRUISE                         # cruise-altitude density [kg/m^3]
    V_C_TAS  = P.V_CRUISE                           # design cruise speed (TAS at altitude) [m/s]
    VD_FACTOR = P.V_DIVE_FACTOR                     # V_dive / V_cruise (CS-25.335 lower bound)
except Exception:
    G, MTOW          = 9.81, 1979.9
    N_POS            = 3.5
    S_W, B_SPAN      = 25.616, 13.0
    RHO              = 1.225
    RHO_CR           = 0.835679
    V_C_TAS          = 200 / 3.6
    VD_FACTOR        = 1.25

# This is a sea-level EAS structural envelope: the stall / manoeuvre speeds are
# computed at sea-level density. The design cruise speed is a TAS at cruise
# altitude, so it is converted to equivalent airspeed before being placed on the
# diagram for consistency (V_EAS = V_TAS * sqrt(rho_alt / rho_SL)).
V_CRUISE = V_C_TAS * np.sqrt(RHO_CR / RHO)   # cruise speed in EAS [m/s]

AR = B_SPAN**2 / S_W    # reference aspect ratio (box-wing benefit carried by e)

# Given / updated
V_DIVE  = VD_FACTOR * V_CRUISE   # dive speed [m/s]  (V_DIVE_FACTOR from parameters.py)
CL_MAX  = 1.686            # box-wing CL_max from XFLR5 (aero dept)

# Flagged assumptions / certification basis
# Basis: EASA SC-VTOL-01, small category (MTOM <= 3175 kg -> our converged
# MTOW qualifies). SC-VTOL is performance-based: VTOL.2215 requires critical
# loads be established over the manoeuvre + gust envelope (hence both are drawn
# here) but does NOT prescribe n; the values below are adopted from the
# CS-23/CS-25 conventions SC-VTOL is designed to be consistent with, pending
# the detailed Limit Flight Envelope (VTOL.2115).
N_NEG       = -1.5      # negative limit load factor; CS-25.333(b)-style
UDE_CRUISE  = 15.0      # cruise gust velocity [m/s] (cert-standard; params has 0)
UDE_DIVE    = 7.5       # dive gust velocity [m/s]   (cert-standard; params has 0)
OSWALD      = 0.85      # for the lift-curve slope estimate

W = MTOW * G            # weight [N]  (converged MTOW x g)

# Lift-curve slope, finite wing: a = a0 / (1 + a0/(pi e AR)) [per rad]
a0 = 2 * np.pi
A_SLOPE = a0 / (1 + a0 / (np.pi * OSWALD * AR))


# --- Characteristic speeds ------------------------------------------------- #
V_S = np.sqrt(W / (0.5 * RHO * CL_MAX * S_W))            # 1g stall
V_A = np.sqrt(N_POS * W / (0.5 * RHO * CL_MAX * S_W))    # corner / manoeuvre


# --- Curves ---------------------------------------------------------------- #
def n_stall_pos(V):
    return 0.5 * RHO * V**2 * CL_MAX * S_W / W

def n_stall_neg(V):
    return -0.5 * RHO * V**2 * CL_MAX * S_W / W

# Gust load factor: n = 1 +/- (0.5 rho V a Kg Ude) / (W/S)
_chord = np.sqrt(S_W / AR)
_mu = 2 * (W / S_W) / (RHO * 9.81 * A_SLOPE * _chord)
_Kg = 0.88 * _mu / (5.3 + _mu)

def n_gust(V, Ude, sign=+1):
    return 1 + sign * (0.5 * RHO * V * A_SLOPE * _Kg * Ude) / (W / S_W)


def main():
    print(f"V_S (1g stall) = {V_S:.1f} m/s")
    print(f"V_A (corner)   = {V_A:.1f} m/s")
    print(f"V_cruise       = {V_CRUISE:.1f} m/s")
    print(f"V_dive         = {V_DIVE:.1f} m/s")
    print(f"limit loads    = +{N_POS} / {N_NEG}")
    print(f"gust at cruise = +{n_gust(V_CRUISE,UDE_CRUISE):.2f} / {n_gust(V_CRUISE,UDE_CRUISE,-1):.2f}")
    print(f"gust at dive   = +{n_gust(V_DIVE,UDE_DIVE):.2f} / {n_gust(V_DIVE,UDE_DIVE,-1):.2f}")

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # ---- Manoeuvre envelope (solid) ----
    # positive stall curve from 0 to corner speed
    Vp = np.linspace(0, V_A, 200)
    ax.plot(Vp, n_stall_pos(Vp), color="#185FA5", lw=2)
    # flat positive limit from corner to dive
    ax.plot([V_A, V_DIVE], [N_POS, N_POS], color="#185FA5", lw=2)
    # right edge (dive line): from the positive limit down to where the
    # negative limit has ramped back to zero at V_D (CS-25.333(b) convention)
    ax.plot([V_DIVE, V_DIVE], [N_POS, 0], color="#185FA5", lw=2)
    # negative stall curve from 0 to the speed where it meets N_neg
    V_neg = np.sqrt(abs(N_NEG) * W / (0.5 * RHO * CL_MAX * S_W))
    Vn = np.linspace(0, V_neg, 200)
    ax.plot(Vn, n_stall_neg(Vn), color="#185FA5", lw=2)
    # flat negative limit from there to cruise, then ramp to 0 at dive
    ax.plot([V_neg, V_CRUISE], [N_NEG, N_NEG], color="#185FA5", lw=2)
    ax.plot([V_CRUISE, V_DIVE], [N_NEG, 0], color="#185FA5", lw=2,
            label="Manoeuvre envelope")

    # ---- Gust lines (dashed) ----
    ax.plot([0, V_CRUISE], [1, n_gust(V_CRUISE, UDE_CRUISE)], "--",
            color="#D85A30", lw=1.5)
    ax.plot([0, V_DIVE], [1, n_gust(V_DIVE, UDE_DIVE)], "--",
            color="#D85A30", lw=1.5, label="Gust lines")
    ax.plot([0, V_CRUISE], [1, n_gust(V_CRUISE, UDE_CRUISE, -1)], "--",
            color="#D85A30", lw=1.5)
    ax.plot([0, V_DIVE], [1, n_gust(V_DIVE, UDE_DIVE, -1)], "--",
            color="#D85A30", lw=1.5)
    # connect cruise-gust peaks to dive (gust envelope outline)
    ax.plot([V_CRUISE, V_DIVE],
            [n_gust(V_CRUISE, UDE_CRUISE), n_gust(V_DIVE, UDE_DIVE)], "--",
            color="#D85A30", lw=1.5)
    ax.plot([V_CRUISE, V_DIVE],
            [n_gust(V_CRUISE, UDE_CRUISE, -1), n_gust(V_DIVE, UDE_DIVE, -1)],
            "--", color="#D85A30", lw=1.5)

    # ---- Reference markers ----
    for V, lbl in [(V_S, "$V_S$"), (V_A, "$V_A$"),
                   (V_CRUISE, "$V_C$"), (V_DIVE, "$V_D$")]:
        ax.axvline(V, color="grey", ls=":", lw=0.8)
        ax.text(V, N_NEG - 0.35, lbl, ha="center", fontsize=9)

    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("equivalent airspeed $V$ [m/s]")
    ax.set_ylabel("load factor $n$ [-]")
    ax.set_title("V-n diagram (cruise / wing-borne regime)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, V_DIVE * 1.08)
    ax.set_ylim(N_NEG - 1, N_POS + 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()