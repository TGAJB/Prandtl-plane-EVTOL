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
    L_EFF_MIN, L_EFF_MAX, N_REACT, N_CROSS,
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
#
# Two CS-27/29 drop conditions, each with its OWN consistent mechanics. The section
# modulus AND the energy balance switch together as a pair (never independently):
#
#   RESERVE-ENERGY drop (V_Z_RESERVE) -> rigid-plastic hinge (_size_plastic)
#       The cross-tube forms a plastic hinge at the fuselage root. Force is the
#       constant plastic-collapse value, the F-delta curve is a rectangle, and the
#       absorption efficiency is eta ~ 1 by construction. Plastic modulus Z_p governs.
#       Ductile metals only (a plastic plateau requires yield).
#
#   LIMIT / no-yield drop (V_Z_LIMIT) -> elastic cantilever spring (_size_elastic)
#       The leg stays elastic; force ramps linearly with stroke, so the F-delta curve
#       is a triangle and it stores only 1/2 F_max*delta (eta ~ 0.5). Cantilever
#       stiffness 3EI/l^3 governs the stroke; a sigma = M c / I check governs strength.
#       All materials (metals AND CFRP) are sized on this condition.
#
# Sizing each condition over (Do, Di, l_eff) with the energy balance of the WRONG
# mechanics over-credits energy by ~2x and overstates strength, so the pairing is
# enforced in code. The governing (heavier) feasible section is returned.

# Gear-mass convergence seed / book-keeping (CHANGE 4).
GEAR_FRAC_SEED      = 0.03   # [-]  class-1 seed: gear ~ 3% of MTOW
INPUT_INCLUDES_GEAR = True   # True if mtow_kg already carries a gear allowance to remove;
                             # False if mtow_kg is the gear-excluded "rest" mass.


def _section_props(Do, Di):
    """Thin-walled round-tube section properties [SI].

    Z_p = (Do^3 - Di^3) / 6           plastic section modulus  (full-plasticity hinge)
    S   = pi (Do^4 - Di^4) / (32 Do)  elastic section modulus  (first-yield bending)
    I   = pi (Do^4 - Di^4) / 64       second moment of area    (3EI/l^3 stiffness, M c / I)
    A   = pi (Do^2 - Di^2) / 4        cross-sectional area      (mass)
    Note S < Z_p for any hollow section (elastic stores less than plastic).
    """
    if Di <= 0 or Di >= Do or (Do - Di) / 2 < T_WALL_SKID:
        raise ValueError(f"Invalid Di={Di:.4f} m for Do={Do:.4f} m (min wall {T_WALL_SKID} m)")
    Z_p = (Do**3 - Di**3) / 6.0
    S   = np.pi * (Do**4 - Di**4) / (32.0 * Do)
    I   = np.pi * (Do**4 - Di**4) / 64.0
    A   = (np.pi / 4.0) * (Do**2 - Di**2)
    return Z_p, S, I, A


def _gear_mass(A, rho):
    """Structural skid mass [kg]: N_CROSS cross-tubes (L_TRACK) + 2 runners (L_SKID),
    same tube section A, with a fittings/attachment knockup K_FITTINGS."""
    return (N_CROSS * L_TRACK * A + 2.0 * L_SKID * A) * rho * K_FITTINGS


def _size_plastic(Do, Di, l_eff, Vz, m_total, rho, sigma_allow, ductile=True):
    """RESERVE-ENERGY rigid-plastic hinge sizing for one section (CHANGE 1, 3).

    Mechanics (rectangular F-delta plateau, eta ~ 1):
        sigma_eff = sigma_allow / sqrt(1 + MU_DRAG^2)   biaxial (vertical + drag) reduction
        Mp        = sigma_eff * Z_p                     UNFACTORED plastic moment  [N.m]
        F_max     = N_REACT * Mp / l_eff                UNFACTORED collapse force  [N]
        net       = F_max - (W - L),  W = m g,  L = KAPPA_LG * W
        delta     = 0.5 m Vz^2 / net                    flat-plateau work-energy balance
        n         = F_max / W                           honest (unfactored) load factor

    CHANGE 3: STRUCT_SF is NOT divided into F_max. The force the occupants and the
    energy balance see is the real, unfactored plastic-hinge reaction. STRUCT_SF is
    applied separately as a section fracture/strength margin: it is carried in the
    ultimate-vs-yield basis of sigma_allow (ductile metals have sigma_ult/sigma_yield
    > STRUCT_SF), so the plastic hinge forms before fracture. A fiber-stress check of
    the kind used in the elastic branch is self-referential here (the hinge stress is
    sigma_eff by definition) and is therefore not applied.

    CHANGE 2: brittle materials (CFRP) have no yield plateau -> no plastic hinge.
    Calling this with ductile=False raises ValueError.
    """
    if not ductile:
        raise ValueError(
            "CFRP/brittle material has no plastic hinge (it fractures, no yield "
            "plateau); size it on the elastic limit condition only (_size_elastic)."
        )
    Z_p, S, I, A = _section_props(Do, Di)
    sigma_eff = sigma_allow / np.sqrt(1.0 + MU_DRAG**2)
    Mp        = sigma_eff * Z_p              # unfactored plastic moment [N.m]
    F_max     = N_REACT * Mp / l_eff         # unfactored collapse force  [N]

    W = m_total * G
    L = KAPPA_LG * W
    net = F_max - (W - L)
    if net <= 0:
        raise ValueError(f"Net restoring force {net:.1f} N <= 0: tube too weak")
    n     = F_max / W                        # honest load factor (no STRUCT_SF de-rating)
    delta = 0.5 * m_total * Vz**2 / net      # rectangular plateau, eta ~ 1

    return {
        "mechanics": "plastic",
        "n": n,
        "delta": delta,
        "F_max": F_max,
        "A": A,
        "m_gear": _gear_mass(A, rho),
        "strength_ok": True,                 # margin carried in sigma_allow ultimate basis
    }


def _size_elastic(Do, Di, l_eff, Vz, m_total, rho, sigma_allow, E):
    """LIMIT / no-yield elastic cantilever-spring sizing for one section (CHANGE 1).

    Mechanics (triangular F-delta, eta ~ 0.5):
        I      = pi (Do^4 - Di^4) / 64
        k_leg  = 3 E I / l_eff^3                cantilever tip stiffness per leg  [N/m]
        K      = N_REACT * k_leg                total landing-gear stiffness
        solve  0.5 K delta^2 = 0.5 m Vz^2 + (m g - L) delta   (work-energy, positive root)
               => delta = [ (mg - L) + sqrt((mg - L)^2 + K m Vz^2) ] / K
        F_max  = K * delta                      peak elastic reaction  [N]
        n      = F_max / W

    Strength (separate from the energy balance, since F_max here comes from stiffness
    and stroke, NOT from the section's strength):
        sigma = M c / I = (F_max / N_REACT) * l_eff * (Do/2) / I  <=  sigma_eff / STRUCT_SF
    With sigma_eff = sigma_allow / sqrt(1 + MU_DRAG^2). STRUCT_SF lives on this stress
    check, not inside F_max.
    """
    Z_p, S, I, A = _section_props(Do, Di)
    k_leg = 3.0 * E * I / l_eff**3
    K     = N_REACT * k_leg                  # total stiffness [N/m]

    W    = m_total * G
    L    = KAPPA_LG * W
    grav = W - L                             # (m g - L), positive work over the stroke
    # 0.5 K d^2 - grav d - 0.5 m Vz^2 = 0  ->  positive root
    disc  = grav**2 + K * m_total * Vz**2
    delta = (grav + np.sqrt(disc)) / K
    F_max = K * delta
    n     = F_max / W

    sigma_eff = sigma_allow / np.sqrt(1.0 + MU_DRAG**2)
    M_leg     = (F_max / N_REACT) * l_eff    # per-leg root bending moment [N.m]
    sigma     = M_leg * (Do / 2.0) / I       # sigma = M c / I
    strength_ok = sigma <= sigma_eff / STRUCT_SF

    return {
        "mechanics": "elastic",
        "n": n,
        "delta": delta,
        "F_max": F_max,
        "A": A,
        "m_gear": _gear_mass(A, rho),
        "sigma": sigma,
        "strength_ok": strength_ok,
    }


def landing_gear_mass(mtow_kg, rho, sigma_allow, E, ductile=True,
                      return_details=False, gear_seed_kg=None):
    """
    Physics-based skid landing-gear mass [kg] via condition-specific energy-balance sizing.

    Sizes the CS-27/29 drop conditions with their own consistent mechanics (CHANGE 1):
        - LIMIT (V_Z_LIMIT)    -> elastic cantilever spring  (_size_elastic), all materials
        - RESERVE (V_Z_RESERVE)-> rigid-plastic hinge        (_size_plastic), ductile metals only

    Ductile metals (Al, Ti) run BOTH branches; CFRP (ductile=False) runs the elastic
    branch only because it has no plastic hinge (CHANGE 2). The cross-tube section is
    found by sweeping outer diameter, inner diameter and effective bent-arm l_eff
    subject to:

        delta       <= GROUND_CLEARANCE   (stroke from the per-condition energy balance)
        n           <= N_LIMIT_LG         (peak load factor)
        strength_ok                       (elastic: sigma = M c / I <= sigma_eff/STRUCT_SF)
        t_wall      >= T_WALL_SKID         (minimum manufacturable gauge)

    The governing (heavier) feasible condition is returned. The gear is sized for the
    FULL landing mass (it decelerates itself too), so its mass is iterated; the non-gear
    mass is held fixed and MTOW is re-formed with the current gear each pass, removing
    the gear allowance exactly once (CHANGE 4).

    Returns
    -------
    (mass_kg, Do_m, Di_m, l_eff_m)         if return_details is False (geometry None on fail)
    dict with per-condition + governing info if return_details is True

    Parameters
    ----------
    mtow_kg     : float  Aircraft mass [kg]; gear allowance removed if INPUT_INCLUDES_GEAR.
    rho         : float  Skid tube material density [kg/m^3]
    sigma_allow : float  Allowable bending stress [Pa]
    E           : float  Young's modulus of the tube material [Pa] (elastic branch)
    ductile     : bool   True for metals (both branches); False for CFRP (elastic only)
    gear_seed_kg: float  Optional initial guess for the gear-mass iteration. The non-gear
                         mass removed from MTOW is always GEAR_FRAC_SEED*mtow_kg (a fixed
                         allowance, removed once), so the converged result is independent
                         of this starting guess; it only changes the iteration path.
    """
    # Condition -> mechanics pairing (the flag switches modulus AND balance together).
    conditions = [(V_Z_LIMIT, "limit", "elastic")]
    if ductile:
        conditions.append((V_Z_RESERVE, "reserve", "plastic"))

    def _size_condition(Do, Di, l_eff, Vz, m_total, mechanics):
        if mechanics == "elastic":
            return _size_elastic(Do, Di, l_eff, Vz, m_total, rho, sigma_allow, E)
        return _size_plastic(Do, Di, l_eff, Vz, m_total, rho, sigma_allow, ductile)

    def _find_governing(m_total):
        """Sweep Do, l_eff and Di; per condition keep the minimum-mass feasible section;
        return the governing (heavier) condition's result dict, or None.

        Di is swept (not derived analytically) because the two mechanics bound the
        section differently; minimum mass = minimum area = the lightest feasible Di."""
        Do_arr   = np.linspace(DO_SKID_MIN, DO_SKID_MAX, 40)
        Leff_arr = np.linspace(L_EFF_MIN,   L_EFF_MAX,   12)

        best = {}
        for Vz, name, mechanics in conditions:
            for Do in Do_arr:
                Di_hi = Do - 2.0 * T_WALL_SKID           # thinnest wall -> largest Di (lightest)
                if Di_hi <= 0.0:
                    continue
                Di_lo = max(0.1 * Do, 1e-3)
                if Di_lo >= Di_hi:
                    continue
                for Di in np.linspace(Di_lo, Di_hi, 30):
                    for l_eff in Leff_arr:
                        try:
                            res = _size_condition(Do, Di, l_eff, Vz, m_total, mechanics)
                        except ValueError:
                            continue
                        if (res["delta"] <= GROUND_CLEARANCE
                                and res["n"] <= N_LIMIT_LG
                                and res["strength_ok"]):
                            if name not in best or res["m_gear"] < best[name]["m_gear"]:
                                res.update(Do=Do, Di=Di, l_eff=l_eff, condition=name)
                                best[name] = res

        if not best:
            return None
        governing = max(best.values(), key=lambda r: r["m_gear"])
        governing = dict(governing)
        governing["per_condition"] = {k: v["m_gear"] for k, v in best.items()}
        return governing

    # ---- Outer mass-convergence loop (CHANGE 4): hold non-gear mass fixed, re-form MTOW ----
    # The gear allowance baked into mtow_kg is removed exactly ONCE here, using the fixed
    # GEAR_FRAC_SEED (not the iteration's starting guess), so m_rest - and hence the fixed
    # point - is independent of gear_seed_kg.
    m_rest = (mtow_kg - GEAR_FRAC_SEED * mtow_kg) if INPUT_INCLUDES_GEAR else mtow_kg
    m_gear = GEAR_FRAC_SEED * mtow_kg if gear_seed_kg is None else gear_seed_kg
    result = None
    for _ in range(50):
        m_total = m_rest + m_gear            # full landing mass, gear folded in exactly once
        result = _find_governing(m_total)
        if result is None:
            warnings.warn(
                "landing_gear_mass: no feasible section found in the (Do, Di, l_eff) "
                "sweep - returning class-1 fallback (3% MTOW). Check L_EFF, T_WALL_SKID "
                "and the per-material allowables in parameters.py.",
                stacklevel=2,
            )
            if return_details:
                return None
            return GEAR_FRAC_SEED * mtow_kg, None, None, None
        m_new = result["m_gear"]
        if abs(m_new - m_gear) / max(m_new, 1e-9) < 0.005:
            m_gear = m_new
            break
        m_gear = 0.4 * m_gear + 0.6 * m_new  # under-relax lambda = 0.6, converge on MASS

    if return_details:
        return result
    return result["m_gear"], result["Do"], result["Di"], result["l_eff"]


# ---------------------------------------------------------------------------
# STUB - asymmetric one-skid / 6-DOF landing load case.
# A real CS-27.727 sideload / one-gear-first condition needs a 6-DOF reaction
# solve (asymmetric vertical + drag + side loads, torsion of the cross-tube,
# unequal leg reactions). This is NOT modelled here; the symmetric two-skid
# drop above does not bound it. Do NOT treat the result above as covering it.
def landing_gear_asymmetric_stub(*args, **kwargs):
    raise NotImplementedError(
        "Asymmetric one-skid / 6-DOF landing load case is not implemented. "
        "Requires a 6-DOF reaction solve (side + drag + vertical, cross-tube "
        "torsion, unequal leg reactions); the symmetric drop does not bound it."
    )


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
