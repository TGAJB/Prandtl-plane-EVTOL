"""
mass_components.py
One function per structural/propulsion component.
Each function returns mass in kg.
Functions that depend on MTOW take it as their first argument.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

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
    # landing gear
    L_EFF_MIN, L_EFF_MAX, N_REACT,
    V_Z_LIMIT, V_Z_RESERVE,
    N_LIMIT_LG, KAPPA_LG, GROUND_CLEARANCE, MU_DRAG,
    T_WALL_SKID, DO_SKID_MIN, DO_SKID_MAX,
    L_TRACK, L_SKID, K_FITTINGS,
)


# Fuselage

def fuselage_mass(mtow_kg):
    return (0.453592
            * (14.86 * ((mtow_kg * 2.20462) ** 0.144) * ((L_FUS * 3.28084) ** 0.778)
               / ((PER_FUS_MAX * 3.28084) ** 0.778))
            * ((L_FUS * 3.28084) ** 0.383) * N_PAX ** 0.455)


# Wing (dynamic sizing from design-point W/S + Class II mass)

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
        # ∫ M(y) dy = L_wing·s²/8  (derived analytically from elliptical l(y))
        # vol_caps = STRUCT_SF · L·s² / (8·σ·h)
        vol_caps = STRUCT_SF * l_wing * s**2 / (8 * SIGMA_ALLOW_CFRP * h_spar_mean)
        m_spar   = 1.4 * vol_caps * RHO_CFRP    # caps + web, CFRP

        # Skins: min-gauge CFRP (8-ply prepreg), upper + lower surface
        m_skin   = 2 * s_wing * T_SKIN_MIN_CFRP * RHO_CFRP

        # Primary fraction 0.76 recovers ribs + secondary structure
        m_total += (m_spar + m_skin) / 0.76

    return m_total


# Landing gear

def landing_gear_mass(mtow_kg, rho, sigma_allow=SIGMA_ALLOW_AL):
    """
    Physics-based skid landing gear mass [kg] via rigid-plastic energy-balance sizing.

    Sizes two CS-27/29 certification conditions (limit drop, V_Z_LIMIT; and
    reserve-energy drop, V_Z_RESERVE).  Cross-tube section is found by sweeping
    outer diameter and cantilever arm, with inner diameter derived analytically
    to minimise mass subject to:

        delta  ≤ GROUND_CLEARANCE  (plastic stroke from energy balance)
        n      ≤ N_LIMIT_LG        (peak load factor)
        t_wall ≥ T_WALL_SKID       (minimum manufacturable gauge)

    Plastic hinge model at cross-tube root (fuselage attachment):
        F_max = N_REACT · Mp / l_eff
        Mp    = (sigma_allow / sqrt(1 + MU_DRAG²)) · Z_p / STRUCT_SF

    The MU_DRAG term accounts for simultaneous horizontal drag bending
    (CS-27.725, μ = 0.5).  STRUCT_SF = 1.5 matches wing_mass / tail_mass.
    The governing (heavier) condition is returned; gear mass is iterated
    because it is part of the total drop mass.

    Returns
    -------
    (mass_kg, Do_m, Di_m, l_eff_m) — geometry is None if no feasible section.

    Parameters
    ----------
    mtow_kg     : float  Aircraft OEW + payload, excl. gear [kg]
    rho         : float  Density of skid tube material [kg/m³]
    sigma_allow : float  Allowable bending stress [Pa] (default SIGMA_ALLOW_AL)
    """
    n_cross    = N_REACT // 2
    conditions = [
        (V_Z_LIMIT,   'limit'),
        (V_Z_RESERVE, 'reserve'),
    ]

    def _size_condition(Do, Di, l_eff, Vz, m_total):
        """
        Rigid-plastic cantilever sizing for one (Do, Di, l_eff, condition).

        Cross-tube forms a plastic hinge at fuselage root:
            Mp    = sigma_eff · Z_p / STRUCT_SF   (biaxial + safety factor)
            F_max = N_REACT · Mp / l_eff
        Energy balance:
            F_max · δ = 0.5 · m · Vz²   (net force = F_max when KAPPA_LG = 1)
        """
        if Di <= 0 or Di >= Do or (Do - Di) / 2 < T_WALL_SKID:
            raise ValueError(f"Invalid Di={Di:.4f} m for Do={Do:.4f} m")
        Z_p       = (Do**3 - Di**3) / 6                         # plastic section modulus [m³]
        sigma_eff = sigma_allow / np.sqrt(1.0 + MU_DRAG**2)    # biaxial (vert + drag) reduction
        Mp        = sigma_eff * Z_p / STRUCT_SF                 # effective plastic moment [N·m]
        A     = (np.pi / 4) * (Do**2 - Di**2)  # cross-sectional area [m²]

        W         = m_total * G
        L         = KAPPA_LG * W
        F_max     = N_REACT * Mp / l_eff        # total collapse force [N]
        net_force = F_max - (W - L)
        if net_force <= 0:
            raise ValueError(f"Net restoring force {net_force:.1f} N ≤ 0: tube too weak")
        n     = F_max / W                        # load factor [-]
        delta = 0.5 * m_total * Vz**2 / net_force  # plastic stroke [m]

        m_gear = (n_cross * L_TRACK * A + 2 * L_SKID * A) * rho * K_FITTINGS
        return n, delta, m_gear

    def _find_governing(m_total):
        """Sweep Do and l_eff; compute optimal Di analytically; return governing tuple or None.

        Di is derived analytically rather than swept:
          - delta ≤ GROUND_CLEARANCE  →  Z_p ≥ Z_p_min  →  Di ≤ Di_delta
          - n ≤ N_LIMIT_LG           →  Z_p ≤ Z_p_max  →  Di ≥ Di_n
          - wall gauge               →  Di ≤ Do − 2·T_WALL_SKID
        Optimal (minimum mass) Di = min(Di_delta, Do − 2·T_WALL_SKID).
        """
        Do_arr   = np.linspace(DO_SKID_MIN, DO_SKID_MAX, 60)
        Leff_arr = np.linspace(L_EFF_MIN,   L_EFF_MAX,   20)

        W         = m_total * G
        sigma_eff = sigma_allow / np.sqrt(1.0 + MU_DRAG**2)

        best = {}
        for Vz, name in conditions:
            F_net_min = 0.5 * m_total * Vz**2 / GROUND_CLEARANCE
            F_max_min = F_net_min + (1.0 - KAPPA_LG) * W   # from delta constraint
            F_max_max = N_LIMIT_LG * W                       # from n constraint

            if F_max_min > F_max_max:
                continue

            for l_eff in Leff_arr:
                # F_max = N_REACT * sigma_eff * Z_p / (STRUCT_SF * l_eff)  →  Z_p = F * l * SF / (N * σ)
                Z_p_min = F_max_min * l_eff * STRUCT_SF / (N_REACT * sigma_eff)
                Z_p_max = F_max_max * l_eff * STRUCT_SF / (N_REACT * sigma_eff)

                for Do in Do_arr:
                    # Di_delta: largest Di that keeps delta ≤ GROUND_CLEARANCE (Z_p ≥ Z_p_min)
                    rhs_delta = Do**3 - 6.0 * Z_p_min
                    if rhs_delta <= 0.0:
                        continue
                    Di_delta = rhs_delta ** (1.0 / 3.0)

                    # Di_n: smallest Di that keeps n ≤ N_LIMIT_LG (Z_p ≤ Z_p_max)
                    rhs_n = Do**3 - 6.0 * Z_p_max
                    Di_n = rhs_n ** (1.0 / 3.0) if rhs_n > 0.0 else 0.0

                    # Wall thickness upper bound on Di
                    Di_wall = Do - 2.0 * T_WALL_SKID

                    # Optimal Di: maximum Di (minimum area) satisfying all bounds
                    Di = min(Di_delta, Di_wall)
                    if Di < max(Di_n, 0.0) or Di <= 0.0 or Di >= Do:
                        continue

                    try:
                        n, delta, mg = _size_condition(Do, Di, l_eff, Vz, m_total)
                    except ValueError:
                        continue
                    if n <= N_LIMIT_LG and delta <= GROUND_CLEARANCE:
                        if name not in best or mg < best[name][0]:
                            best[name] = (mg, Do, Di, l_eff, n, delta, name)

        if not best:
            return None
        return max(best.values(), key=lambda x: x[0])   # governing = heavier condition

    # Outer mass-convergence loop (gear mass appears in m_total it decelerates)
    m_gear = 0.03 * mtow_kg   # seed: 3% of aircraft mass
    for _ in range(50):
        result = _find_governing(mtow_kg + m_gear)
        if result is None:
            # No feasible tube in [DO_SKID_MIN, DO_SKID_MAX] for all three constraints.
            # With a simple cantilever model, stress scales as σ ~ n·W·L_EFF/(Do²·t).
            # The Do that keeps n ≤ N_LIMIT_LG is too small to carry the bending moment —
            # a known incompatibility when placeholder parameters are used.  Calibrate
            # L_EFF (should be the effective bent-section arm, NOT the full track half-span)
            # and T_WALL_SKID before trusting this result.
            warnings.warn(
                "landing_gear_mass: no feasible section found in Do sweep - "
                "returning class-1 fallback (3% MTOW).  "
                "Calibrate L_EFF and T_WALL_SKID in parameters.py.",
                stacklevel=2,
            )
            return 0.03 * mtow_kg, None, None, None
        m_new, Do_opt, Di_opt, Leff_opt = result[0], result[1], result[2], result[3]
        if abs(m_new - m_gear) / max(m_new, 1e-9) < 0.005:
            return m_new, Do_opt, Di_opt, Leff_opt
        m_gear = 0.4 * m_gear + 0.6 * m_new   # under-relaxation λ = 0.6

    return result[0], result[1], result[2], result[3]   # geometry-consistent best estimate


# V-tail (physics-based cantilever sizing)

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


# Motors

def motor_mass(max_power_kw):
    """Total mass of all motors [kg]."""
    m_single = 0.165 * ((max_power_kw * (1 + PM)) / N_MOTOR)
    return m_single * N_MOTOR


# Propellers

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


# Miscellaneous & hinge

def misc_mass(mtow_kg):
    return 0.20 * mtow_kg


def hinge_mass(mtow_kg):
    return 0.05 * mtow_kg
