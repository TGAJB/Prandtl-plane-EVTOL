"""
mass_components.py
One function per structural/propulsion component.
Each function returns mass in kg.
Functions that depend on MTOW take it as their first argument.
"""

import sys
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
    L_EFF, N_REACT,
    V_Z_LIMIT, V_Z_RESERVE, ETA_LG_LIMIT, ETA_LG_RESERVE,
    N_LIMIT_LG, KAPPA_LG, GROUND_CLEARANCE,
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
    s_wing = geometry["area_per_wing_m2"]              # planform area per wing [m?]
    c_mean = geometry["mean_chord_m"]       # root chord [m]
    h_spar_mean = TIP_TO_CHORD_W * c_mean                   # spar depth at root [m]
    # no h_eff correction: wing is horizontal, bending is about the chord axis

    # -- Size front and rear wings independently (general for F_REAR_WING ? 0.5) --
    m_total = 0.0
    for f_lift in [(1.0 - F_REAR_WING), F_REAR_WING]:
        l_wing = f_lift * N_W * mtow_kg * G

        # Correct spar cap volume for elliptical lift distribution.
        # ??? M(y) dy = L_wing?s?/8  (derived analytically from elliptical l(y))
        # vol_caps = STRUCT_SF ? L?s? / (8???h)
        vol_caps = STRUCT_SF * l_wing * s**2 / (8 * SIGMA_ALLOW_CFRP * h_spar_mean)
        m_spar   = 1.4 * vol_caps * RHO_CFRP    # caps + web, CFRP

        # Skins: min-gauge CFRP (8-ply prepreg), upper + lower surface
        m_skin   = 2 * s_wing * T_SKIN_MIN_CFRP * RHO_CFRP

        # Primary fraction 0.76 recovers ribs + secondary structure
        m_total += (m_spar + m_skin) / 0.76

    return m_total


# Landing gear

def landing_gear_mass(mtow_kg, E, rho, sigma_allow=SIGMA_ALLOW_AL):
    """
    Physics-based skid landing gear mass [kg] via energy-balance sizing.

    Sizes two CS-27/29 certification conditions (limit drop and reserve-energy
    drop), selects the governing (heavier) section, and iterates to a fixed
    point because gear mass contributes to the total mass being decelerated.

    Caveats
    -------
    (i)  Equal force split P = F_max / n_react is valid for a SYMMETRIC
         vertical drop only.  One-skid / banked landings require a
         statically-indeterminate or 6-DOF dynamic solve - stub left at
         _size_asymmetric() below.
    (ii) CFRP: set ETA_LG_RESERVE ? 0.50 (brittle fracture, no plastic
         plateau).  Do NOT use 0.60-0.80 for CFRP in the reserve condition.

    Parameters
    ----------
    mtow_kg     : float  Aircraft OEW + payload, excl. gear [kg]
    E           : float  Young's modulus of skid tube material [Pa]
    rho         : float  Density of skid tube material [kg/m?]
    sigma_allow : float  Allowable bending stress [Pa] (default SIGMA_ALLOW_AL)
    """
    n_react = N_REACT       # = 2 * n_cross; set via parameters.py
    n_cross = n_react // 2

    # Two certification conditions: (sink rate [m/s], efficiency [-], label)
    conditions = [
        (V_Z_LIMIT,   ETA_LG_LIMIT,   'limit'),
        (V_Z_RESERVE, ETA_LG_RESERVE, 'reserve'),
    ]

    def _size_condition(Do, Vz, eta, m_total):
        """
        Energy-balance + Euler-Bernoulli cantilever sizing for one (Do, condition).

        Quadratic stroke solver (CS-27/29 energy requirement):
            0.5?K?delta? - (W - L)?delta - 0.5?m?Vz? = 0   (positive root)
        Cantilever root stress:
            sigma = M?(Do/2)/I,  M = P?L_eff,  P = F_max/n_react (symmetric only)

        Stub: replace the equal-split P line with _size_asymmetric() for the
        one-skid / banked-landing case (statically-indeterminate analysis).
        """
        Di = Do - 2 * T_WALL_SKID
        if Di <= 0:
            raise ValueError(f"Wall {T_WALL_SKID} m too large for Do={Do:.4f} m")
        I = (np.pi / 64) * (Do**4 - Di**4)
        A = (np.pi / 4)  * (Do**2 - Di**2)

        W     = m_total * G
        L     = KAPPA_LG * W
        k_leg = 3 * E * I / L_EFF**3   # cantilever stiffness [N/m]
        K     = n_react * k_leg         # legs in parallel [N/m]

        # Analytic go/no-go: bracket must be > 0 (Eq. 5 in spec)
        bracket = eta * N_LIMIT_LG - 1 + KAPPA_LG
        if bracket <= 0:
            raise ValueError(
                f"Infeasible: eta*N_LIMIT_LG - 1 + KAPPA_LG = {bracket:.4f} <= 0. "
                "Increase eta, N_LIMIT_LG, or reduce Vz."
            )

        # Quadratic for stroke: 0.5?K?d? - (W-L)?d - 0.5?m?Vz? = 0
        a_q = 0.5 * K
        b_q = -(W - L)
        c_q = -0.5 * m_total * Vz**2
        delta = (-b_q + np.sqrt(b_q**2 - 4 * a_q * c_q)) / (2 * a_q)

        F_max = K * delta
        n     = F_max / W               # ground-reaction load factor
        P     = F_max / n_react         # per-leg force - symmetric drop only
        M     = P * L_EFF
        sigma = M * (Do / 2) / I
        MS    = sigma_allow / (STRUCT_SF * sigma) - 1

        # Mass: n_cross cross-tubes + 2 skid runners + fittings knockup
        m_gear = (n_cross * L_TRACK * A + 2 * L_SKID * A) * rho * K_FITTINGS
        return MS, n, delta, m_gear

    def _find_governing(m_total):
        """Sweep Do; return governing (m_gear, Do, MS, n, delta, cond) or None."""
        Do_arr = np.linspace(DO_SKID_MIN, DO_SKID_MAX, 200)
        best = {}
        for Vz, eta, name in conditions:
            feasible = []
            for Do in Do_arr:
                try:
                    MS, n, delta, mg = _size_condition(Do, Vz, eta, m_total)
                    if MS >= 0 and n <= N_LIMIT_LG and delta <= GROUND_CLEARANCE:
                        feasible.append((mg, Do, MS, n, delta, name))
                except ValueError:
                    continue
            if feasible:
                feasible.sort(key=lambda x: x[0])
                best[name] = feasible[0]
        if not best:
            return None
        return max(best.values(), key=lambda x: x[0])  # heavier = governing

    # Outer mass-convergence loop (gear mass appears in m_total it decelerates)
    m_gear = 0.03 * mtow_kg  # seed: 3 % of aircraft mass
    lam    = 0.6             # under-relaxation factor
    tol    = 0.005           # 0.5 % convergence tolerance on mass

    for _ in range(50):
        result = _find_governing(mtow_kg + m_gear)
        if result is None:
            # No feasible tube in [DO_SKID_MIN, DO_SKID_MAX] for all three constraints.
            # With a simple cantilever model, stress scales as sigma ~ n?W?L_EFF/(Do??t).
            # The Do that keeps n <= N_LIMIT_LG is too small to carry the bending moment -
            # a known incompatibility when placeholder parameters are used.  Calibrate
            # L_EFF (should be the effective bent-section arm, NOT the full track half-span)
            # and T_WALL_SKID before trusting this result.
            import warnings
            warnings.warn(
                "landing_gear_mass: no feasible section found in Do sweep - "
                "returning class-1 fallback (3% MTOW).  "
                "Calibrate L_EFF and T_WALL_SKID in parameters.py.",
                stacklevel=2,
            )
            return 0.03 * mtow_kg
        m_new = result[0]
        if abs(m_new - m_gear) / max(m_new, 1e-9) < tol:
            return m_new
        m_gear = (1 - lam) * m_gear + lam * m_new

    return m_gear   # best estimate if tolerance not met within 50 iterations


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
