"""
MMOI.py
Aircraft mass-moment-of-inertia buildup for the Prandtl-plane eVTOL.

Each major component is approximated by a simple geometric shape, placed at the
component-CG coordinates defined in parameters.py ("MMOI component layout" section),
given its mass from the converged class-II breakdown (mtow_sizing.converged_mass),
and combined with the parallel axis theorem about the composite centre of gravity.

Axes / datum
------------
Datum at the nose tip: x positive aft, y positive starboard, z positive up, origin
on the fuselage centreline (right-handed). This frame is related to the standard
flight-dynamics body frame (x forward, z down) by a 180 deg rotation about y, under
which I_xx, I_yy, I_zz AND I_xz are numerically invariant - the values returned here
feed vd_parameters.MassProperties directly. Products of inertia are reported in the
flight-dynamics convention I_xz = sum(m * dx * dz) (= -I_tensor[0, 2]); for a
laterally symmetric layout I_xy = I_yz = 0.

Wing geometry approximation - trade study
-----------------------------------------
Three simple models were compared for one wing panel (b = WING_SPAN, lam = TAPER_W):

1. Rectangular prism, MAC x span x (TIP_TO_CHORD_W * MAC)  <- CHOSEN
   One box formula gives all three axes. Assumes uniform spanwise mass.
2. Slender spanwise rod: identical Ixx = Izz = m b^2 / 12 but drops the (small)
   local pitch term; no simpler in code once a box helper exists.
3. Span-tapered beam (mass per unit span ~ local chord):
   gyradius^2 = (b/2)^2 (1 + 3 lam) / (6 (1 + lam)) = 0.81x the uniform value at
   lam = 0.45, i.e. the prism OVERESTIMATES the wing-own spanwise term by ~19%.

The prism is retained: the 19% applies only to the wing-own term, which is small
next to the parallel-axis contributions of the tip plates at y = +/-b/2 and the
rotor pods at 30-70% semi-span, so the aircraft-level error is a few percent and
conservative (inertia slightly high = safe for control sizing). Real spar mass is
biased inboard by root bending but partly pushed back outboard by the Prandtl
tip-joiner loads, so uniform spanwise mass is not systematically wrong at class-II
fidelity.

Component shape assumptions (all documented at the build site below)
--------------------------------------------------------------------
fuselage      thin-walled cylinder shell (structure mass lives at the skin; a solid
              cylinder would halve the roll term; contents are separate components;
              nose/tail taper ignored)
wing x2       rectangular prism (see trade study); m_wing split 50/50 front/rear
              (asserted against AREA_SPLIT) after carving out WINGLET_MASS_FRAC
tip plates x2 flat plates in the x-z plane at y = +/- span/2, height H_GAP_WINGS,
              chord = wing tip chord; the real joiner runs diagonally over the
              stagger but its parallel-axis y^2 term dominates anyway
tail x2       flat plates rotated +/- V_ANGLE about x; S_TAIL is treated as ONE
              panel's area, following the tail_mass implementation (NOTE: the
              comment in parameters.py says "both panels" - stale, the code wins)
motors/hubs/  point masses, 1/6 of the group mass at each of the 6 rotor stations
hinge         (hinge = tilt hardware assumed co-located with the pods)
props         thin discs, normal z (hover orientation; the local disc term is <1%
              of its parallel-axis term, so tilt orientation is irrelevant)
battery       solid uniform-density box under the cabin floor
payload       solid uniform-density box (pax + luggage) in the cabin
landing gear  two slender rods along x at the skid rails (cross-tubes/legs folded
              into the rails)
misc          solid cylinder filling the fuselage (20%-MTOW systems/wiring/ECS
              assumed volume-distributed, hence solid rather than shell)

Known inconsistencies flagged (parameters.py is authoritative)
--------------------------------------------------------------
- vd_parameters.py assumes a 10 m fuselage vs L_FUS = 7.0 m; vd-derived seed
  coordinates in parameters.py were rescaled accordingly.
- All layout coordinates are PLACEHOLDER pending a real layout drawing.

Run standalone for the per-component breakdown table:
    python class_II_sizing/MMOI.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from parameters import (
    AREA_SPLIT, WING_SPAN, TIP_TO_CHORD_W,
    S_TAIL, AR_T, TAPER_TAIL, V_ANGLE,
    N_PROP, N_MOTOR, D_PROP,
    L_FUS, D_FUS, X_CG_FUS, Z_CG_FUS,
    X_WING_F, Z_WING_F, X_WING_R, Z_WING_R, H_GAP_WINGS, WINGLET_MASS_FRAC,
    X_TAIL, Z_TAIL_ROOT,
    ETA_ROTOR_FW_IN, ETA_ROTOR_FW_OUT, ETA_ROTOR_RW,
    X_ROTOR_FW, X_ROTOR_RW, Z_ROTOR_FW, Z_ROTOR_RW,
    L_BATT, W_BATT, H_BATT, X_BATT, Z_BATT,
    X_PAYLOAD, Z_PAYLOAD, L_PAYLOAD_BOX, W_PAYLOAD_BOX, H_PAYLOAD_BOX,
    X_GEAR, Y_GEAR_RAIL, Z_GEAR, L_SKID_RAIL,
    X_MISC, Z_MISC,
)
from class_II_sizing.mtow_sizing import converged_mass


# ---------------------------------------------------------------------------
# Shape helpers - all return the 3x3 inertia tensor [kg m^2] about the shape's
# own CG, in body-parallel axes.
# ---------------------------------------------------------------------------

def inertia_cylinder_shell(m, radius, length):
    """Thin-walled cylindrical shell, axis along x."""
    ix = m * radius**2
    iyz = m * (radius**2 / 2.0 + length**2 / 12.0)
    return np.diag([ix, iyz, iyz])


def inertia_solid_cylinder(m, radius, length):
    """Solid uniform cylinder, axis along x."""
    ix = m * radius**2 / 2.0
    iyz = m * (3.0 * radius**2 + length**2) / 12.0
    return np.diag([ix, iyz, iyz])


def inertia_solid_box(m, lx, ly, lz):
    """Solid uniform box; zero side lengths give plates and rods."""
    return np.diag([
        m * (ly**2 + lz**2) / 12.0,
        m * (lx**2 + lz**2) / 12.0,
        m * (lx**2 + ly**2) / 12.0,
    ])


def inertia_thin_disc(m, radius):
    """Thin uniform disc, normal along z."""
    return np.diag([m * radius**2 / 4.0, m * radius**2 / 4.0, m * radius**2 / 2.0])


def rotate_about_x(inertia, angle_rad):
    """Rotate an inertia tensor by angle_rad about the x axis (R I R^T)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    r = np.array([[1.0, 0.0, 0.0],
                  [0.0, c, -s],
                  [0.0, s, c]])
    return r @ inertia @ r.T


# ---------------------------------------------------------------------------
# Component table
# ---------------------------------------------------------------------------

def build_components(breakdown):
    """
    Assemble the per-piece component list from a converged_mass()/compute_mtow()
    breakdown dict. Every breakdown mass key is used exactly once, so the
    component masses sum back to breakdown["mtow"] by construction.

    Returns a list of dicts: {"name", "mass" [kg], "position" (3,) [m],
    "I_local" 3x3 [kg m^2]}.
    """
    # The 50/50 front/rear wing split and the 4+2 rotor layout below are
    # hard-wired to the current configuration.
    assert AREA_SPLIT == 0.5, "wing mass split assumes equal front/rear wing areas"
    assert N_PROP == 6 and N_MOTOR == 6, "rotor stations assume the 4 front + 2 rear layout"

    geom = breakdown["wing_geom"]
    mac = geom["mac_m"]
    span = geom["span_m"]
    c_tip = geom["tip_chord_m"]

    comps = []

    def add(name, mass, position, inertia_local):
        comps.append({
            "name": name,
            "mass": float(mass),
            "position": np.asarray(position, dtype=float),
            "I_local": inertia_local,
        })

    # -- Fuselage: thin-walled cylinder shell --
    m_fus = breakdown["fuselage"]
    add("fuselage", m_fus, (X_CG_FUS, 0.0, Z_CG_FUS),
        inertia_cylinder_shell(m_fus, D_FUS / 2.0, L_FUS))

    # -- Wings: rectangular prisms, tip-joiner fraction carved out first --
    m_wing_each = (1.0 - WINGLET_MASS_FRAC) * breakdown["wing"] / 2.0
    wing_local = inertia_solid_box(m_wing_each, mac, span, TIP_TO_CHORD_W * mac)
    add("wing_front", m_wing_each, (X_WING_F, 0.0, Z_WING_F), wing_local)
    add("wing_rear", m_wing_each, (X_WING_R, 0.0, Z_WING_R), wing_local)

    # -- Tip plates: vertical flat plates at mid-stagger, y = +/- span/2 --
    m_plate = WINGLET_MASS_FRAC * breakdown["wing"] / 2.0
    plate_local = inertia_solid_box(m_plate, c_tip, 0.0, H_GAP_WINGS)
    x_plate = (X_WING_F + X_WING_R) / 2.0
    z_plate = (Z_WING_F + Z_WING_R) / 2.0
    add("tip_plate_stbd", m_plate, (x_plate, span / 2.0, z_plate), plate_local)
    add("tip_plate_port", m_plate, (x_plate, -span / 2.0, z_plate), plate_local)

    # -- V-tail: two flat plates rotated +/- V_ANGLE about x. Panel geometry
    #    follows tail_mass: l_panel = sqrt(AR_T * S_TAIL) with S_TAIL per panel.
    #    Panel CG at the trapezoid spanwise centroid d from the root. --
    m_panel = breakdown["tail"] / 2.0
    l_panel = np.sqrt(AR_T * S_TAIL)
    c_mean = S_TAIL / l_panel
    d_cg = (l_panel / 3.0) * (1.0 + 2.0 * TAPER_TAIL) / (1.0 + TAPER_TAIL)
    v_rad = np.radians(V_ANGLE)
    panel_flat = inertia_solid_box(m_panel, c_mean, l_panel, 0.0)
    for side, sign in (("stbd", 1.0), ("port", -1.0)):
        add(f"tail_{side}", m_panel,
            (X_TAIL, sign * d_cg * np.cos(v_rad), Z_TAIL_ROOT + d_cg * np.sin(v_rad)),
            rotate_about_x(panel_flat, sign * v_rad))

    # -- Rotor stations: 4 front (2/side) + 2 rear (1/side). Motors, hubs and
    #    hinge hardware as point masses; props as thin discs (hover normal). --
    stations = [
        ("fw_in_stbd", X_ROTOR_FW, ETA_ROTOR_FW_IN * span / 2.0, Z_ROTOR_FW),
        ("fw_in_port", X_ROTOR_FW, -ETA_ROTOR_FW_IN * span / 2.0, Z_ROTOR_FW),
        ("fw_out_stbd", X_ROTOR_FW, ETA_ROTOR_FW_OUT * span / 2.0, Z_ROTOR_FW),
        ("fw_out_port", X_ROTOR_FW, -ETA_ROTOR_FW_OUT * span / 2.0, Z_ROTOR_FW),
        ("rw_stbd", X_ROTOR_RW, ETA_ROTOR_RW * span / 2.0, Z_ROTOR_RW),
        ("rw_port", X_ROTOR_RW, -ETA_ROTOR_RW * span / 2.0, Z_ROTOR_RW),
    ]
    m_pod = (breakdown["motors"] + breakdown["hubs"] + breakdown["hinge"]) / N_PROP
    m_prop = breakdown["props"] / N_PROP
    prop_local = inertia_thin_disc(m_prop, D_PROP / 2.0)
    for name, x, y, z in stations:
        add(f"pod_{name}", m_pod, (x, y, z), np.zeros((3, 3)))
        add(f"prop_{name}", m_prop, (x, y, z), prop_local)

    # -- Battery: solid underfloor box --
    m_batt = breakdown["battery"]
    add("battery", m_batt, (X_BATT, 0.0, Z_BATT),
        inertia_solid_box(m_batt, L_BATT, W_BATT, H_BATT))

    # -- Payload: solid cabin box --
    m_pay = breakdown["payload"]
    add("payload", m_pay, (X_PAYLOAD, 0.0, Z_PAYLOAD),
        inertia_solid_box(m_pay, L_PAYLOAD_BOX, W_PAYLOAD_BOX, H_PAYLOAD_BOX))

    # -- Landing gear: two skid rails as slender rods along x --
    m_rail = breakdown["landing_gear"] / 2.0
    rail_local = inertia_solid_box(m_rail, L_SKID_RAIL, 0.0, 0.0)
    add("gear_rail_stbd", m_rail, (X_GEAR, Y_GEAR_RAIL, Z_GEAR), rail_local)
    add("gear_rail_port", m_rail, (X_GEAR, -Y_GEAR_RAIL, Z_GEAR), rail_local)

    # -- Misc systems: solid cylinder filling the fuselage --
    m_misc = breakdown["misc"]
    add("misc", m_misc, (X_MISC, 0.0, Z_MISC),
        inertia_solid_cylinder(m_misc, D_FUS / 2.0, L_FUS))

    return comps


# ---------------------------------------------------------------------------
# CG and inertia
# ---------------------------------------------------------------------------

def compute_cg(components):
    """Composite CG [m] (nose datum) from the component table."""
    m_total = sum(c["mass"] for c in components)
    moment = sum(c["mass"] * c["position"] for c in components)
    return moment / m_total


def compute_inertia(components, about=None):
    """
    Total inertia tensor via the parallel axis theorem:
        I = sum( I_local + m * ((d . d) E3 - d (x) d) ),  d = r - about.

    `about` defaults to the composite CG. Returns a dict with the total mass,
    cg, the 3x3 tensor I, the diagonal moments and the products of inertia in
    the flight-dynamics convention Iab = sum(m da db) (= -I_tensor off-diagonal),
    so Ixz feeds vd_parameters.MassProperties.I_xz directly.
    """
    cg = compute_cg(components)
    ref = cg if about is None else np.asarray(about, dtype=float)

    inertia = np.zeros((3, 3))
    for c in components:
        d = c["position"] - ref
        inertia += c["I_local"] + c["mass"] * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    return {
        "mass": sum(c["mass"] for c in components),
        "cg": cg,
        "I": inertia,
        "Ixx": inertia[0, 0],
        "Iyy": inertia[1, 1],
        "Izz": inertia[2, 2],
        "Ixy": -inertia[0, 1],
        "Iyz": -inertia[1, 2],
        "Ixz": -inertia[0, 2],
    }


def aircraft_inertia(breakdown=None, verbose=False):
    """
    Top-level entry point: inertia of the converged design about its own CG.
    Pass a compute_mtow()/converged_mass() breakdown to evaluate a specific
    design point; defaults to the cached converged design.
    """
    if breakdown is None:
        breakdown = converged_mass()
    components = build_components(breakdown)
    result = compute_inertia(components)
    if verbose:
        _print_breakdown(components, result)
    return result


def as_mass_properties(result):
    """
    Map an aircraft_inertia() result onto vd_parameters.MassProperties field
    names. I_xx/I_yy/I_zz/I_xz are frame-invariant under the x-aft/z-up ->
    x-fwd/z-down flip, so they transfer directly. z_cg is given in THIS module's
    convention (nose datum, z up) - flip its sign for a z-down consumer.

    Usage:
        from dataclasses import replace
        params.mass = replace(params.mass, **as_mass_properties(res))
    """
    return {
        "I_xx": float(result["Ixx"]),
        "I_yy": float(result["Iyy"]),
        "I_zz": float(result["Izz"]),
        "I_xz": float(result["Ixz"]),
        "z_cg": float(result["cg"][2]),
    }


# ---------------------------------------------------------------------------
# Standalone report
# ---------------------------------------------------------------------------

def _print_breakdown(components, result):
    cg = result["cg"]
    print(f"\n{'Component':<16} {'m [kg]':>8} {'x':>6} {'y':>7} {'z':>6} "
          f"{'dIxx':>9} {'dIyy':>9} {'dIzz':>9}")
    print("-" * 80)
    for c in components:
        d = c["position"] - cg
        contrib = c["I_local"] + c["mass"] * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        x, y, z = c["position"]
        print(f"{c['name']:<16} {c['mass']:>8.1f} {x:>6.2f} {y:>7.2f} {z:>6.2f} "
              f"{contrib[0, 0]:>9.1f} {contrib[1, 1]:>9.1f} {contrib[2, 2]:>9.1f}")
    print("-" * 80)
    print(f"{'TOTAL':<16} {result['mass']:>8.1f}"
          f"{'':>22} {result['Ixx']:>9.1f} {result['Iyy']:>9.1f} {result['Izz']:>9.1f}")
    print(f"\nCG (nose datum, z up) : x = {cg[0]:.3f} m, y = {cg[1]:.3f} m, z = {cg[2]:.3f} m")
    print(f"Ixx = {result['Ixx']:.0f}, Iyy = {result['Iyy']:.0f}, "
          f"Izz = {result['Izz']:.0f}, Ixz = {result['Ixz']:.0f} kg m^2")

    # Nondimensional radii of gyration vs Raymer Table 16.1 typicals (GA twin:
    # R_x ~ 0.25-0.30). Tip plates and outboard rotor pods push R_x above the
    # conventional-configuration band - expected for a box wing.
    m = result["mass"]
    rx = np.sqrt(result["Ixx"] / m) / (WING_SPAN / 2.0)
    ry = np.sqrt(result["Iyy"] / m) / (L_FUS / 2.0)
    print(f"Gyradii: R_x = {rx:.3f} (b/2 ref), R_y = {ry:.3f} (L_fus/2 ref)")


if __name__ == "__main__":
    aircraft_inertia(verbose=True)
