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
From parameters.py (traceable):
    N_W = 3.5            positive limit load factor
    S_W = 30 m^2         wing area
    wing loading 760 N/m^2  -> W = 22 800 N (mass ~2324 kg, ~ MTOW 2314)
    V_cruise = 200/3.6 = 55.6 m/s   design cruise speed
    rho = 1.225, AR = 5.63

Given / updated by the team:
    V_dive = 1.6 * V_cruise = 88.9 m/s      (NB parameters.py still lists
                                             V_DIVE_FACTOR = 1.25 -- reconcile)
    CL_max = 1.66        ESTIMATE pending the real aero polar (was 2.0)

FLAGGED ASSUMPTIONS (confirm with loads / certification basis):
    N_neg = -1.5         negative limit load factor (typical; not in params)
    Ude_cruise = 15 m/s, Ude_dive = 7.5 m/s   gust velocities (params has 0)
    lift-curve slope a   estimated from AR (finite-wing correction)
    negative stall uses the same CL_max magnitude

Run:  python vn_diagram.py
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Inputs ---------------------------------------------------------------- #
# From parameters.py
N_POS   = 3.5            # positive limit load factor (N_W); CS-23-aligned,
                        #   which EASA SC-VTOL (small cat.) explicitly targets
S_W     = 30.0          # wing area [m^2]
WS_N    = 760.0         # wing loading [N/m^2]
RHO     = 1.225         # air density [kg/m^3]
AR      = 5.63          # wing aspect ratio
V_CRUISE = 200 / 3.6    # design cruise speed [m/s]

# Given / updated
V_DIVE  = 1.6 * V_CRUISE   # dive speed [m/s]  (team value; params says 1.25)
CL_MAX  = 1.66             # ESTIMATE pending aero polar (was 2.0)

# Flagged assumptions / certification basis
# Basis: EASA SC-VTOL-01, small category (MTOM <= 3175 kg -> our 2314 kg
# qualifies). SC-VTOL is performance-based: VTOL.2215 requires critical loads
# be established over the manoeuvre + gust envelope (hence both are drawn
# here) but does NOT prescribe n; the values below are adopted from the
# CS-23/CS-25 conventions SC-VTOL is designed to be consistent with, pending
# the detailed Limit Flight Envelope (VTOL.2115).
N_NEG       = -1.5      # negative limit load factor; CS-25.333(b)-style
UDE_CRUISE  = 15.0      # cruise gust velocity [m/s] (cert-standard; params has 0)
UDE_DIVE    = 7.5       # dive gust velocity [m/s]   (cert-standard; params has 0)
OSWALD      = 0.85      # for the lift-curve slope estimate

W = WS_N * S_W          # weight [N]

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
    # right edge (dive line) down to negative limit
    ax.plot([V_DIVE, V_DIVE], [N_POS, N_NEG], color="#185FA5", lw=2)
    # negative stall curve from 0 to the speed where it meets N_neg
    V_neg = np.sqrt(abs(N_NEG) * W / (0.5 * RHO * CL_MAX * S_W))
    Vn = np.linspace(0, V_neg, 200)
    ax.plot(Vn, n_stall_neg(Vn), color="#185FA5", lw=2)
    # flat negative limit from there to cruise, then ramp to 0 at dive
    ax.plot([V_neg, V_CRUISE], [N_NEG, N_NEG], color="#185FA5", lw=2)
    ax.plot([V_CRUISE, V_DIVE], [N_NEG, 0], color="#185FA5", lw=2,
            label="Manoeuvre envelope")

    # ---- Gust lines (dashed) ----
    Vg = np.array([0, V_CRUISE, V_DIVE])
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