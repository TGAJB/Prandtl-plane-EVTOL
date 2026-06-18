"""
turning.py
==========
Wing-borne turning-performance analysis for the Folding Prandtl eVTOL.

SCOPE & CAVEAT
--------------
Steady, level, coordinated turn in the WING-BORNE cruise regime. This is the
aerodynamic + structural turn envelope: the lift (C_L,max) boundary and the
structural limit load factor. The SUSTAINED-turn ceiling (whether the
propulsion system can supply the thrust to hold a given load factor without
decelerating) is set by the available propulsive thrust at cruise advance
ratio and is established in the Propulsion chapter -- it is NOT drawn here.
Hover / transition turning is a separate regime (rotor-differential yaw, no
coordinated-turn EOM) and is also out of scope, as with the V-n diagram.

CONDITION
---------
Drawn at CRUISE ALTITUDE in true airspeed (rho_cr), because turning
performance is a real-airspeed/real-altitude problem ("all turning
performance deteriorates with increasing altitude"). This is deliberately a
different condition from the V-n diagram, which is a sea-level (EAS)
structural envelope.

PROVENANCE
----------
All inputs pulled from parameters.py (single source of truth). Where a value
is flagged there as preliminary/estimate, it is flagged here too:
    CL_MAX = 1.686   box-wing CL_max from XFLR5 (aero dept)
    e = 1.34         box-wing span efficiency (e_hor_wings, updated 11-06-2026)
    CD0 = 0.0205     preliminary
    N_W = 3.5        positive limit load factor

Run:  python turning.py
"""

import sys
import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt

# Repo root (parameters.py + the optimizer's final-design output live here).
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# --- Inputs --------------------------------------------------------------- #
# MTOW and wing geometry come from the OPTIMIZER's final design state
# (final_design/results/data/characteristics.json); the aero/mission constants
# come from parameters.py. Import-or-fallback so it still runs standalone.
try:
    import parameters as P
    _cs = json.loads((ROOT / "final_design/results/data/characteristics.json").read_text())
    G       = P.G
    MTOW    = _cs["mass"]["mtow"]                 # optimizer final MTOW
    S       = _cs["wing_geometry"]["total_area_m2"]   # converged wing area
    B_SPAN  = _cs["wing_geometry"]["span_m"]
    RHO_CR  = P.RHO_CRUISE
    RHO_SL  = P.RHO_ORIGIN
    E_SPAN  = P.OSWALD_EFFICIENCY                 # = e_hor_wings = 1.34
    CD0     = P.CD0                               # preliminary
    N_LIM   = P.N_W                               # 3.5
    V_C     = P.V_CRUISE                          # design cruise (TAS)
except Exception:
    G, MTOW, S, B_SPAN = 9.81, 1979.9, 25.616, 13.0
    RHO_CR, RHO_SL     = 0.835679, 1.225
    E_SPAN, CD0, N_LIM = 1.34, 0.0205, 3.5
    V_C                = 200 / 3.6

CL_MAX = 1.686  # box-wing CL_max from XFLR5 (aero dept)

# --- Derived --------------------------------------------------------------- #
W   = MTOW * G
WS  = W / S
AR  = B_SPAN**2 / S                 # monoplane-equivalent reference AR;
                                    #   box-wing benefit carried by e > 1
k   = 1.0 / (np.pi * E_SPAN * AR)   # induced-drag factor


def v_stall(rho):
    """1g stall speed at density rho."""
    return np.sqrt(2 * WS / (rho * CL_MAX))


def n_lift(V, rho):
    """Aerodynamic (C_L,max) load-factor ceiling at speed V."""
    return 0.5 * rho * V**2 * CL_MAX * S / W


def turn_geometry(V, n):
    """Bank angle [deg], radius [m], turn rate [deg/s] for a level turn."""
    phi = np.degrees(np.arccos(1.0 / n))
    R   = V**2 / (G * np.sqrt(n**2 - 1.0))
    om  = np.degrees(G * np.sqrt(n**2 - 1.0) / V)
    return phi, R, om


def main():
    VS = v_stall(RHO_CR)
    VA = VS * np.sqrt(N_LIM)          # corner speed at cruise altitude

    print(f"--- cruise altitude (rho = {RHO_CR:.3f} kg/m^3), TAS ---")
    print(f"W/S        = {WS:6.1f} N/m^2   (MTOW {MTOW:.0f} kg, S {S:.2f} m^2)")
    print(f"k          = {k:.4f}   (e = {E_SPAN}, AR_ref = {AR:.2f})")
    print(f"V_S        = {VS:5.1f} m/s")
    print(f"V_A corner = {VA:5.1f} m/s")
    print(f"V_C cruise = {V_C:5.1f} m/s (TAS)")
    n_at_VC = min(n_lift(V_C, RHO_CR), N_LIM)
    print(f"n at V_C   = {n_at_VC:5.2f} "
          f"({'lift-capped' if n_lift(V_C, RHO_CR) < N_LIM else 'structural'})")

    phiA, RA, omA = turn_geometry(VA, N_LIM)
    print(f"\nTightest structural turn @ V_A: "
          f"phi={phiA:.1f} deg, R={RA:.0f} m, rate={omA:.1f} deg/s")
    phiC, RC, omC = turn_geometry(V_C, n_at_VC)
    print(f"Turn @ V_C (n={n_at_VC:.2f}):        "
          f"phi={phiC:.1f} deg, R={RC:.0f} m, rate={omC:.1f} deg/s")

    # --- turn diagram (n vs V) -------------------------------------------- #
    V_MAX = 1.12 * VA                 # trim: nothing of interest past the corner
    V = np.linspace(0, V_MAX, 400)
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # lift-limited branch up to where it meets the structural limit
    n_cl = n_lift(V, RHO_CR)
    mask = n_cl <= N_LIM
    ax.plot(V[mask], n_cl[mask], color="#185FA5", lw=2,
            label=r"$C_{L,\max}$ limit (lift)")
    # structural ceiling from corner speed onward
    ax.plot([VA, V[-1]], [N_LIM, N_LIM], color="#D85A30", lw=2,
            label=r"structural limit $n = %.1f$" % N_LIM)

    # reference speeds (V_C and V_A are within ~2 m/s, so stagger the labels)
    for Vx, lbl, dy in [(VS, "$V_S$", -0.25),
                        (V_C, "$V_C$", -0.25),
                        (VA, "$V_A$", -0.50)]:
        ax.axvline(Vx, color="grey", ls=":", lw=0.8)
        ax.text(Vx, dy, lbl, ha="center", fontsize=9)

    # operating point at cruise, annotation offset up-left into clear space
    ax.plot(V_C, n_at_VC, "ko", ms=5)
    ax.annotate(rf"cruise: $n={n_at_VC:.2f}$, $\phi={phiC:.0f}^\circ$, "
                rf"$R={RC:.0f}$ m",
                (V_C, n_at_VC), xytext=(V_C - 2.0, n_at_VC + 0.55),
                ha="right", fontsize=8,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="grey"))

    ax.axhline(1, color="black", lw=0.5, ls="--")
    ax.set_xlabel("true airspeed $V$ [m/s]  (cruise altitude)")
    ax.set_ylabel("load factor $n$ [-]")
    ax.set_title("Turn envelope (wing-borne, cruise altitude)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, V_MAX)
    ax.set_ylim(-0.6, N_LIM + 0.8)
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()