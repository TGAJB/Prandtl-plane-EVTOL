"""
mass_components.py
One function per structural/propulsion component.
Each function returns mass in kg.
Functions that depend on MTOW take it as their first argument.
"""

import numpy as np
from parameters import (
    G, RHO_ORIGIN,
    L_FUS, PER_FUS_MAX, N_PAX,
    N_W, TAPER_W, TIP_TO_CHORD_W,
    SIGMA_ALLOW_CFRP, RHO_CFRP, T_SKIN_MIN_CFRP,
    S_TAIL, AR_T, TAPER_TAIL, TIP_TO_CHORD, V_ANGLE,
    F_REAR_WING, SIGMA_ALLOW_AL, RHO_AL, T_SKIN_MIN_AL,
    STRUCT_SF, C_N_TAIL_MAX, V_DIVE_FACTOR, V_CRUISE,
    N_PROP, N_MOTOR, N_BLADES, D_PROP, PM,
    WING_SPAN, AREA_SPLIT, WING_LOADING_N,
)


# ── Fuselage ───────────────────────────────────────────────────────────────────

def fuselage_mass(mtow_kg):
    return (0.453592
            * (14.86 * ((mtow_kg * 2.20462) ** 0.144) * ((L_FUS * 3.28084) ** 0.778)
               / ((PER_FUS_MAX * 3.28084) ** 0.778))
            * ((L_FUS * 3.28084) ** 0.383) * N_PAX ** 0.455)


# ── Wing (dynamic sizing from design-point W/S + Class II mass) ───────────────

def wing_geometry(mtow_kg):
    # -- Geometry from selected design point: S = W / (W/S) --
    weight_n = mtow_kg * G
    total_area_m2 = weight_n / WING_LOADING_N
    area_per_wing_m2 = total_area_m2 * AREA_SPLIT
    aspect_ratio = WING_SPAN ** 2 / total_area_m2
    aspect_ratio_per_wing = WING_SPAN ** 2 / area_per_wing_m2
    mean_chord_m = WING_SPAN / aspect_ratio_per_wing
    root_chord_m = 2 * area_per_wing_m2 / ((1 + TAPER_W) * WING_SPAN)
    tip_chord_m = TAPER_W * root_chord_m

    return {
        "weight_n": weight_n,
        "total_area_m2": total_area_m2,
        "area_per_wing_m2": area_per_wing_m2,
        "span_m": WING_SPAN,
        "aspect_ratio": aspect_ratio,
        "mean_chord_m": mean_chord_m,
        "root_chord_m": root_chord_m,
        "tip_chord_m": tip_chord_m,
    }


def wing_mass(mtow_kg, geometry=None):
    # -- Geometry (identical for front and rear wings) --
    if geometry is None:
        geometry = wing_geometry(mtow_kg)

    b_w    = geometry["span_m"]                         # full wing span [m]
    s      = b_w / 2                                    # semi-span [m]
    s_wing = geometry["area_per_wing_m2"]              # planform area per wing [m²]
    c_mean = geometry["mean_chord_m"]       # root chord [m]
    h_spar_mean = TIP_TO_CHORD_W * c_mean                   # spar depth at root [m]
    # no h_eff correction: wing is horizontal, bending is about the chord axis

    # -- Size front and rear wings independently (general for F_REAR_WING ≠ 0.5) --
    m_total = 0.0
    for f_lift in [(1.0 - F_REAR_WING), F_REAR_WING]:
        l_wing = f_lift * N_W * mtow_kg * G

        # Correct spar cap volume for elliptical lift distribution.
        # ∫₀ˢ M(y) dy = L_wing·s²/8  (derived analytically from elliptical l(y))
        # vol_caps = STRUCT_SF · L·s² / (8·σ·h)
        vol_caps = STRUCT_SF * l_wing * s**2 / (8 * SIGMA_ALLOW_CFRP * h_spar_mean)
        m_spar   = 1.4 * vol_caps * RHO_CFRP    # caps + web, CFRP

        # Skins: min-gauge CFRP (8-ply prepreg), upper + lower surface
        m_skin   = 2 * s_wing * T_SKIN_MIN_CFRP * RHO_CFRP

        # Primary fraction 0.76 recovers ribs + secondary structure
        m_total += (m_spar + m_skin) / 0.76

    return m_total


# ── Landing gear ───────────────────────────────────────────────────────────────

def landing_gear_mass(mtow_kg):
    return 0.03 * mtow_kg


# ── V-tail (physics-based cantilever sizing) ───────────────────────────────────

def tail_mass(mtow_kg):
    # -- Geometry --
    l_panel = 0.5 * np.sqrt(AR_T * S_TAIL)
    b_half  = l_panel * np.cos(np.radians(V_ANGLE))
    c_root  = S_TAIL / ((1 + TAPER_TAIL) * l_panel)
    h_spar  = TIP_TO_CHORD * c_root
    h_eff   = h_spar * np.cos(np.radians(V_ANGLE))

    # -- Load case 1: rear wing lift transferred through V-tail root --
    f_vert  = (F_REAR_WING * N_W * mtow_kg * G) / 2.0
    m_rear  = b_half * f_vert

    # -- Load case 2: V-tail own aero load at dive speed, max deflection --
    q_dive  = 0.5 * RHO_ORIGIN * (V_DIVE_FACTOR * V_CRUISE) ** 2
    f_aero  = q_dive * (S_TAIL / 2) * C_N_TAIL_MAX
    m_aero  = f_aero * l_panel / 2

    # -- Ultimate root moment --
    m_root  = STRUCT_SF * (m_rear + m_aero)

    # -- Spar (caps + web): Euler-Bernoulli cantilever --
    vol_caps = m_root * l_panel / (SIGMA_ALLOW_AL * h_eff)
    vol_spar = 1.4 * vol_caps          # web adds ~40% of cap volume
    m_spar   = vol_spar * RHO_AL

    # -- Skins: min-gauge governs --
    m_skin   = 2 * (S_TAIL / 2) * T_SKIN_MIN_AL * RHO_AL

    # -- Primary fraction 0.76 accounts for ribs + fittings (secondary structure) --
    m_panel  = (m_spar + m_skin) / 0.76

    return 2.0 * m_panel               # both panels


# ── Motors ─────────────────────────────────────────────────────────────────────

def motor_mass(max_power_kw):
    """Total mass of all motors [kg]."""
    m_single = 0.165 * ((max_power_kw * (1 + PM)) / N_MOTOR)
    return m_single * N_MOTOR


# ── Propellers ─────────────────────────────────────────────────────────────────

def propeller_mass(max_power_kw, use_fusion=False, m_blade_fusion=None):
    """Total propeller blade mass [kg] across all rotors."""
    if use_fusion and m_blade_fusion is not None:
        return m_blade_fusion * N_BLADES * N_PROP
    return N_PROP * 0.144 * ((D_PROP * (max_power_kw / N_PROP) * N_BLADES ** 0.5) ** 0.782)


def hub_mass(use_fusion=False, m_hub_fusion=None):
    """Total hub mass [kg] across all rotors."""
    if use_fusion and m_hub_fusion is not None:
        return m_hub_fusion * N_PROP
    return 0.0


# ── Miscellaneous & hinge ──────────────────────────────────────────────────────

def misc_mass(mtow_kg):
    return 0.20 * mtow_kg


def hinge_mass(mtow_kg):
    return 0.05 * mtow_kg
