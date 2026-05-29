"""
main_fusion.py
MTOW sizing script — identical to main.py except propeller blade and hub
masses are sourced from Fusion 360 geometry (fusion/geometry.json).

Workflow:
    1. In Fusion 360, run (in any order):
         fusion/propeller_blade/propeller_blade.py  → writes "propeller_blade"
         fusion/hub/hub.py                          → writes "propeller_hub"
       Each script merges its entry into geometry.json without overwriting others.
       IMPORTANT: assign correct materials in Fusion first (CFRP for blade,
       aluminium for hub) so the exported masses reflect real densities.
    2. Run this file.  Missing entries fall back to analytical formulae.
"""

import json
import os
import numpy as np

# ── Fusion geometry import ─────────────────────────────────────────────────────

_GEOMETRY_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fusion", "geometry.json")

def _load_fusion_geometry(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

_FUSION_GEO = _load_fusion_geometry(_GEOMETRY_JSON)

if "propeller_blade" in _FUSION_GEO:
    M_BLADE_FUSION   = _FUSION_GEO["propeller_blade"]["mass_kg"]
    _USE_FUSION_PROP = True
    print(f"[Fusion] Single blade mass loaded: {M_BLADE_FUSION:.4f} kg")
else:
    M_BLADE_FUSION   = None
    _USE_FUSION_PROP = False
    print("[Fusion] WARNING: no 'propeller_blade' entry — falling back to analytical formula.")

if "propeller_hub" in _FUSION_GEO:
    M_HUB_FUSION    = _FUSION_GEO["propeller_hub"]["mass_kg"]
    _USE_FUSION_HUB = True
    print(f"[Fusion] Single hub mass loaded:   {M_HUB_FUSION:.4f} kg")
else:
    M_HUB_FUSION    = None
    _USE_FUSION_HUB = False
    print("[Fusion] WARNING: no 'propeller_hub' entry — hub mass set to zero.")

print(f"         Source: {_GEOMETRY_JSON}")

# ── Mission & payload constants ────────────────────────────────────────────────

M_PAYLOAD = 400.0      # [kg]  Fixed mission payload (4 pax + luggage)
W_PAYLOAD = 400.0      # [kg]  Fixed mission payload (4 pax + luggage)
W_CREW    = 0.0        # [kg]  0 if autonomous; add 85 if piloted

RANGE_M          = 200000.0    # [m]   Cruise range
V_CRUISE         = 200 / 3.6   # [m/s] Cruise speed
T_CRUISE         = 2194.7      # [s]   Cruise segment duration
T_TAKEOFF        = 5.0         # [s]   Takeoff segment duration
T_CLIMB          = 1221.7      # [s]   Climb segment duration
T_CLIMB_ACC      = 20.0        # [s]   Climb acceleration segment duration
T_DESCENT        = 311.945     # [s]   Descent segment duration
T_VERTICAL_CLIMB = 25.0        # [s]   Total vertical climb time
T_LANDING        = 71.47       # [s]   Total landing time

# ── Design assumptions ─────────────────────────────────────────────────────────

LD_CRUISE             = 14.7   # [-]   Box-wing cruise L/D
ETA_CRUISE            = 0.95   # [-]   Cruise total efficiency
ETA_POWERTRAIN_HOVER  = 0.95   # [-]   Hover powertrain efficiency
ETA_CLIMB             = 0.90   # [-]   Climb powertrain efficiency
FM                    = 0.73   # [-]   Rotor figure of merit
N_PROPS               = 6      # [-]   Number of rotors
PROP_DIAMETER         = 1.9    # [m]   Rotor diameter
T_W0                  = 1.29   # [-]   Initial thrust-to-weight ratio guess

# Battery
E_PACK_WH_KG = 300.0   # [Wh/kg] Pack-level specific energy
SOC_USABLE   = 0.80    # [-]     Usable state-of-charge
CONTINGENCY  = 1.05    # [-]     Energy contingency

# ── Shared constants ───────────────────────────────────────────────────────────

L_D    = 14.7
A_DISK = 2.84          # [m²]  per disk
PROP_EFF = 0.8
T_W    = 1.29
G      = 9.81          # [m/s²]
V_CR   = 200 / 3.6     # [m/s]
D_CR   = 121929.0741   # [m]
BATT_EFF  = 0.98
BATT_SE   = 300        # [Wh/kg]
L_FUS     = 7          # [m]
PER_FUS_MAX = 12       # [m]
N_PAX     = 4
S_W       = 30         # [m²]
N_W       = 3.5        # design load factor
AR_W      = 5.63       # wing aspect ratio
SKID_LEN  = 1          # [m]
S_TAIL    = 3.25       # [m²]
AR_T      = 1.23       # tail aspect ratio
TIP_TO_CHORD    = 0.1
TAIL_SWEEP_C4   = 15   # [deg]
TAIL_SAFETY_FACTOR = 1.5

# V-tail sizing constants ──────────────────────────────────────
V_ANGLE          = 45.0     # [deg]  V-tail dihedral from horizontal — verify with CAD
TAPER_TAIL       = 0.40     # [-]    chord taper ratio c_tip/c_root
F_REAR_WING      = 0.5     # [-]    rear Prandtl-wing lift fraction (from aero model; typ. 0.40-0.55)
# SIGMA_ALLOW_CFRP = 500e6  # [Pa]   CFRP UD compression allowable (superseded by Al below)
# RHO_CFRP         = 1550.0 # [kg/m3] CFRP density (superseded by Al below)
# T_PLY            = 0.125e-3# [m]   UD CFRP prepreg ply thickness (superseded)
# N_PLY_MIN        = 8       # [-]   min plies per skin, MIL-HDBK-17-3F (superseded)
SIGMA_ALLOW_AL   = 260e6    # [Pa]   2024-T3 Fcy B-basis, MMPDS-01 Table 3.2.3.0(e)
RHO_AL           = 2700.0   # [kg/m3] aluminium alloy density
T_SKIN_MIN_AL    = 1.2e-3   # [m]    minimum sheet thickness for Al primary structure
                             #        (manufacturing and damage tolerance floor; Niu 1988)
STRUCT_SF        = 1.5      # [-]    structural safety factor, limit -> ultimate (FAR/CS 25.303)
C_N_TAIL_MAX     = 1.2      # [-]    peak normal force coeff. at max control deflection
V_DIVE_FACTOR    = 1.25     # [-]    dive speed as fraction of cruise (conservative lower bound)

T_ELAPSED_VC    = 5    # [s]
INIT_ELEV       = 0    # [m]
FIN_ELEV        = 10   # [m]
V_AVG_TO        = 2    # [m/s]
V_HOVER         = 3    # [m/s]
RHO_ORIGIN      = 1.225  # [kg/m³]
POWER_SAFETY_FACTOR = 1.4
PM        = 0.5        # power margin
N_MOTOR   = 6
N_PROP    = 6
N_BLADES  = 5
D_PROP    = 1.9        # [m]


# ── MTOW computation ───────────────────────────────────────────────────────────

def COMPUTE_MTOW(MTOW):

    # ── Power calculations ──────────────────────────────────────────────────────

    AVG_THRUST = MTOW * (G + (V_HOVER / T_ELAPSED_VC))
    V_HOVER_I  = np.sqrt(AVG_THRUST / (2 * 6 * RHO_ORIGIN * A_DISK))
    V_AVG_I    = -V_AVG_TO / 2 + np.sqrt((V_AVG_TO / 2) ** 2 + V_HOVER_I ** 2)
    MAX_POWER  = (POWER_SAFETY_FACTOR * (AVG_THRUST * (V_AVG_I + V_AVG_TO)) / FM) / 1000

    # ── Energy / battery ───────────────────────────────────────────────────────

    def takeoff_power(mtow_kg, A_disk, Vs_avg, Vs_f, N_prop):
        thrust  = mtow_kg * (G + Vs_f / T_TAKEOFF)
        Vi_avg  = -(Vs_avg / 2) + np.sqrt((Vs_avg / 2) ** 2 + thrust / (2 * 1.225 * A_disk * N_prop))
        p_ideal = thrust * (Vi_avg + Vs_avg) / FM
        return p_ideal / ETA_POWERTRAIN_HOVER

    def vertical_climb_power(mtow_kg, A_disk, Vs, N_prop):
        thrust  = mtow_kg * (G + Vs / T_TAKEOFF)
        Vi_avg  = -(Vs / 2) + np.sqrt((Vs / 2) ** 2 + thrust / (2 * 1.225 * A_disk * N_prop))
        p_ideal = thrust * (Vi_avg + Vs) / FM
        return p_ideal / ETA_POWERTRAIN_HOVER

    def climb_acceleration_power(mtow_kg, V_i, V_f, Vs, acc_time):
        acc   = (V_f - V_i) / acc_time
        E_req = (0.5 * mtow_kg * (V_f ** 2 - V_i ** 2)
                 + mtow_kg * G * Vs * acc_time
                 + (0.5 * mtow_kg * G * acc * acc_time ** 2) / LD_CRUISE)
        return E_req / ETA_CLIMB

    def climb_power(mtow_kg, climb_angle_deg, V_climb_horizontal, Vs):
        thrust  = ((mtow_kg * G) * (np.cos(np.radians(climb_angle_deg)) / LD_CRUISE)
                   + (mtow_kg * G) * np.sin(np.radians(climb_angle_deg)))
        p_ideal = thrust * np.sqrt(V_climb_horizontal ** 2 + Vs ** 2)
        return p_ideal / ETA_CLIMB

    def cruise_power(mtow_kg):
        return mtow_kg * G * V_CRUISE / (LD_CRUISE * ETA_CRUISE)

    def landing_power(mtow_kg, A_disk, Vs_0, N_prop):
        thrust  = mtow_kg * (G + Vs_0 / T_LANDING)
        Vs_avg  = Vs_0 / 3
        Vi_avg  = -(Vs_avg / 2) + np.sqrt((Vs_avg / 2) ** 2 + thrust / (2 * 1.00583 * A_disk * N_prop))
        p_ideal = thrust * (Vi_avg - Vs_avg) / FM
        return p_ideal / ETA_POWERTRAIN_HOVER

    def mission_energy(mtow_kg):
        p_to     = takeoff_power(mtow_kg, A_disk=np.pi * (PROP_DIAMETER / 2) ** 2,
                                 Vs_avg=2, Vs_f=3, N_prop=6)
        p_vc     = vertical_climb_power(mtow_kg, A_disk=np.pi * (PROP_DIAMETER / 2) ** 2,
                                        Vs=3, N_prop=6)
        p_cl_acc = climb_acceleration_power(mtow_kg, V_i=20, V_f=V_CRUISE, Vs=3,
                                            acc_time=T_CLIMB_ACC)
        p_cl     = climb_power(mtow_kg, climb_angle_deg=3.09097,
                               V_climb_horizontal=V_CRUISE, Vs=3)
        p_c      = cruise_power(mtow_kg)
        p_l      = landing_power(mtow_kg, A_disk=np.pi * (PROP_DIAMETER / 2) ** 2,
                                 Vs_0=7.6, N_prop=6)
        return (p_to     * T_TAKEOFF
                + p_vc   * T_VERTICAL_CLIMB
                + p_c    * T_CRUISE
                + p_cl   * T_CLIMB
                + p_cl_acc
                + p_l    * T_LANDING)

    def battery_mass_from_mtow(mtow_kg):
        e_mission_Wh  = mission_energy(mtow_kg) / 3600.0
        e_installed_Wh = e_mission_Wh * CONTINGENCY / SOC_USABLE
        return e_installed_Wh / E_PACK_WH_KG

    # ── Component masses ────────────────────────────────────────────────────────

    M_F    = (0.453592
              * (14.86 * ((MTOW * 2.20462) ** 0.144) * ((L_FUS * 3.28084) ** 0.778)
                 / ((PER_FUS_MAX * 3.28084)) ** 0.778)
              * ((L_FUS * 3.28084) ** 0.383) * N_PAX ** 0.455)

    M_W    = (0.453592
              * 0.002933 * ((S_W * 3.28084 ** 2) ** 1.018)
              * (AR_W ** 2.473) * N_W ** 0.611)

    M_LG   = MTOW * 0.03

    # Roskam/Torenbeek transport regression (metallic, calibrated outside this weight class):
    # M_TAIL = (0.453592
    #           * ((1.68 * ((MTOW * 2.20462) ** 0.567) * ((S_TAIL * 3.28084 ** 2) ** 1.249)
    #               * (AR_T ** 0.482))
    #              / (639.95 * (TIP_TO_CHORD ** 0.747)
    #                 * (np.cos(TAIL_SWEEP_C4 * (np.pi / 180)) ** 0.882))))

    # Physics-based cantilever sizing.  Each V-tail panel is a cantilever fixed at the
    # fuselage root and loaded at the tip by the rear Prandtl-wing lift (dominant load case).
    # Two structural contributions: bending spar (Euler-Bernoulli beam theory) + min-gage skins.
    # Note: TIP_TO_CHORD is used here as the thickness-to-chord ratio (t/c = 0.10).

    # -- Geometry --
    # S_TAIL is the actual panel surface area (not planform).
    # AR_T = (actual span)^2 / S_TAIL, so l_panel is derived directly.
    _l_panel = 0.5 * np.sqrt(AR_T * S_TAIL)                               # actual panel span [m]
    _b_half  = _l_panel * np.cos(np.radians(V_ANGLE))                     # horizontal projection [m]
    _c_root  = S_TAIL / ((1 + TAPER_TAIL) * _l_panel)                     # root chord [m]
    _h_spar  = TIP_TO_CHORD * _c_root                                      # spar depth at root [m]

    # -- Spar caps: Euler-Bernoulli cantilever, sized to ultimate load --
    #
    # Load case 1 — rear wing lift at limit load (vector analysis gives M = b_half x F_vert;
    # cos terms cancel in the moment arm, but cos(V_ANGLE) re-enters via h_eff below).
    _F_vert_lim = (F_REAR_WING * N_W * MTOW * G) / 2.0                    # [N] per panel, limit
    _M_rear     = _b_half * _F_vert_lim                                    # [N.m]
    #
    # Load case 2 — V-tail own aerodynamic load at dive speed, max control deflection.
    # Normal force acts perpendicular to panel surface.  Cross-product analysis (same as
    # rear wing derivation) gives M_aero = F_aero * l_panel / 2 regardless of V_ANGLE,
    # because the aero force is perpendicular to the span vector throughout.
    _q_dive  = 0.5 * RHO_ORIGIN * (V_DIVE_FACTOR * V_CR) ** 2             # [Pa]
    _F_aero  = _q_dive * (S_TAIL / 2) * C_N_TAIL_MAX                      # [N] per panel
    _M_aero  = _F_aero * _l_panel / 2                                      # [N.m]
    #
    # Ultimate design moment (FAR/CS 25.303: ultimate = limit x 1.5)
    _M_root  = STRUCT_SF * (_M_rear + _M_aero)                             # [N.m]
    #
    # h_eff: flanges separated by h_spar in panel thickness direction (-sinG, 0, cosG).
    # Only the z-component resists bending about y-axis: h_eff = h_spar * cos(V_ANGLE).
    # Vol_flanges = integral of A_cap(x) dx = M_root * l / (sigma * h_eff)
    _h_eff    = _h_spar * np.cos(np.radians(V_ANGLE))                     # [m]
    _vol_caps = _M_root * _l_panel / (SIGMA_ALLOW_AL * _h_eff)              # [m3]
    _vol_spar = 1.4 * _vol_caps        # spar web ~40% of cap volume (thin-walled box beam)
    _m_spar   = _vol_spar * RHO_AL                                         # [kg] per panel

    # -- Skins: min-gage governs (shear sizing < 0.2 mm; min-gage governs) --
    # S_TAIL is the actual panel surface area, so no cos(V_ANGLE) correction needed.
    # Two skins (upper + lower) per panel.
    _m_skin   = 2 * (S_TAIL / 2) * T_SKIN_MIN_AL * RHO_AL                 # [kg] per panel

    # -- Ribs + fittings: secondary structure fraction derived from first principles --
    # Rib pitch governed by skin panel bending stress under ultimate aerodynamic pressure
    # (Timoshenko & Woinowsky-Krieger, Theory of Plates and Shells, 2nd ed. 1959, §4 Table 8):
    #   sigma_max = (3/4) * dp_ult * b_rib^2 / t_skin^2  (long plate, 4 sides simply supported)
    # Solving for b_max with dp_ult = q_dive*C_N_max*SF and sigma = sigma_allow gives b_max ≈ 0.31 m,
    # requiring 5 ribs over l_panel = 1.0 m (root, 0.25, 0.50, 0.75, tip).
    # All rib webs are minimum-gauge governed (shear stress << allowable at every station).
    # Explicit rib mass summation at each station gives ~3.5 kg secondary per panel vs ~12.5 kg primary,
    # yielding a primary fraction of 12.5/16.0 = 0.78. Conservative rounding: 0.76.
    _m_panel  = (_m_spar + _m_skin) / 0.76

    M_TAIL    = 2.0 * _m_panel                                             # [kg] both panels

    M_MOTOR     = 0.165 * ((MAX_POWER * (1 + PM)) / N_MOTOR)
    M_MOTOR_TOT = M_MOTOR * N_MOTOR

    # ── Propeller blade mass: Fusion 360 or analytical fallback ────────────────
    if _USE_FUSION_PROP:
        M_PROP     = M_BLADE_FUSION * N_BLADES   # blades on one rotor disc
        M_PROP_TOT = M_PROP * N_PROP             # all rotors
        _prop_src  = "Fusion"
    else:
        M_PROP     = 0.144 * ((D_PROP * (MAX_POWER / N_PROP) * N_BLADES ** 0.5) ** 0.782)
        M_PROP_TOT = N_PROP * M_PROP
        _prop_src  = "analytical"

    # ── Hub mass: Fusion 360 or zero ────────────────────────────────────────────
    if _USE_FUSION_HUB:
        M_HUB_TOT = M_HUB_FUSION * N_PROP       # one hub per rotor
        _hub_src  = "Fusion"
    else:
        M_HUB_TOT = 0.0
        _hub_src  = "not modelled"

    M_BATT  = battery_mass_from_mtow(MTOW)
    M_MISC  = 0.20 * MTOW
    M_HINGE = 0.05 * MTOW

    MTOW_NEW = (M_F + M_W + M_LG + M_TAIL
                + M_MOTOR_TOT + M_PROP_TOT + M_HUB_TOT
                + M_PAYLOAD + M_BATT + M_MISC + M_HINGE)

    print("")
    print(f"Fuselage:             {M_F:.2f} kg")
    print(f"Wing:                 {M_W:.2f} kg")
    print(f"Landing Gear:         {M_LG:.2f} kg")
    print(f"Tail:                 {M_TAIL:.2f} kg")
    print(f"Motor:                {M_MOTOR:.2f} kg  (per motor)")
    print(f"Blades [{_prop_src}]:   {M_PROP_TOT:.2f} kg  ({N_PROP} rotors x {N_BLADES} blades)")
    print(f"Hubs   [{_hub_src}]:    {M_HUB_TOT:.2f} kg  ({N_PROP} rotors)")
    print(f"Payload:              {M_PAYLOAD:.2f} kg")
    print(f"Battery:              {M_BATT:.2f} kg")
    print(f"Miscellaneous:        {M_MISC:.2f} kg")
    print(f"Hinge:                {M_HINGE:.2f} kg")
    print("")

    return MTOW_NEW


# ── Iterator ───────────────────────────────────────────────────────────────────

BOUND_LOW_MTOW  = 1000
BOUND_HIGH_MTOW = 5500
GUESS_OLD       = (BOUND_HIGH_MTOW + BOUND_LOW_MTOW) / 2

ITERATE = True
COUNT   = 0

while ITERATE:
    GUESS_NEW = COMPUTE_MTOW(GUESS_OLD)

    if GUESS_NEW > BOUND_HIGH_MTOW or GUESS_NEW < BOUND_LOW_MTOW:
        print("MTOW out of bounds — check your inputs.")
        ITERATE = False
    elif np.abs(GUESS_NEW - GUESS_OLD) / GUESS_OLD < 0.01:
        ITERATE = False
        print(f"Final MTOW: {GUESS_NEW:.2f} kg")
        if _USE_FUSION_PROP:
            print(f"  Blade source  : Fusion 360  |  single blade : {M_BLADE_FUSION:.4f} kg"
                  f"  |  total : {M_BLADE_FUSION * N_BLADES * N_PROP:.2f} kg")
        else:
            print(f"  Blade source  : analytical regression formula")
        if _USE_FUSION_HUB:
            print(f"  Hub source    : Fusion 360  |  single hub   : {M_HUB_FUSION:.4f} kg"
                  f"  |  total : {M_HUB_FUSION * N_PROP:.2f} kg")
        else:
            print(f"  Hub source    : not modelled (set to zero)")

    STEP      = (GUESS_NEW - GUESS_OLD) / 2
    GUESS_OLD = GUESS_OLD + STEP
    COUNT    += 1

print(f"Total Iterations: {COUNT}")
