"""
pressurization_sizing.py
Cabin-pressurization mass + energy penalty for the Prandtl-plane eVTOL MTOW loop.

Plugs into the fixed-point MTOW convergence loop with the signature

    pressurization_sizing(mtow, n_pax, t_above_cabin_alt_s, material, h_cruise, h_cabin)
        -> (m_press_kg, E_press_J, info_dict)

m_press_kg is added to the component-mass sum; E_press_J is added to the mission energy
(the battery-mass feedback is then closed by the existing loop). An indicative
dm_batt_equiv [kg] is reported in info_dict so the energy penalty can be read as
kilograms without re-running the loop; it mirrors the main model's SA504 cell-count
battery basis (energy x (1 + reserve) -> cells -> pack mass, smooth, no string round-up).

Method
  Part A - structural penalty: thin-walled circular pressure vessel (hoop stress), as an
           INCREMENT over the unpressurized min-gauge/flight-loads skin, plus a seal/cutout
           floor that is nonzero even when the thickness increment is zero.
  Part B - electric air-supply system (no bleed air on an eVTOL): FAR 25.831 fresh-air flow
           or leakage make-up, isentropic compressor power from ambient at cruise to cabin
           pressure, plus fixed valve/controller masses and ducting.

DESIGN NOTES (trades to revisit)
  * 12,500 ft (3810 m) is exactly the FAR 91.211 supplemental-oxygen threshold. An
    UNPRESSURIZED cabin + O2-mask alternative could delete most of this system mass -
    flag as a trade to revisit before freezing the configuration.
  * h_cabin is the key sweep variable: lowering the cabin altitude to 6,000 ft
    (premium comfort) raises the design differential by roughly 1.5-2x relative to the
    8,000 ft FAR 25.841 baseline, which scales the shell increment directly.
  * MTOW couples in only via the battery-energy feedback and the mass-fraction
    reporting; the geometry (and hence the shell + air-supply sizing) is pax-driven.

HARD ASSUMPTION: circular (or near-circular) pressure-cabin cross-section. Flat panels
carry the differential in BENDING (t ~ w*sqrt(dp/sigma)) and come out roughly an order
of magnitude heavier than a membrane-stressed circular shell; a warning is raised if a
non-circular flag is passed.

All quantities SI. Functions, not classes (repo convention).
"""

import math
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from parameters import (
    # battery pack figures (mirror the main model's SA504 cell-count basis; the older
    # E_pack [Wh/kg] x SOC form was superseded on this branch by energy.battery_mass)
    E_CELL_WH, M_CELL_KG, CELL_TO_PACK, RESERVE_FRAC,
    T_SKIN_MIN_AL, T_SKIN_MIN_CFRP,    # unpressurized min-gauge skins (baseline thickness)
)


# -- Physical constants (not placeholders) -----------------------------------
G_ISA     = 9.80665   # [m/s^2] standard gravity (ISA definition)
R_AIR     = 287.05    # [J/kg/K] specific gas constant, dry air
L_ISA     = 0.0065    # [K/m]   ISA tropospheric lapse rate
T0_ISA    = 288.15    # [K]     ISA sea-level temperature
P0_ISA    = 101325.0  # [Pa]    ISA sea-level pressure
CP_AIR    = 1005.0    # [J/kg/K] air specific heat at constant pressure
GAMMA_AIR = 1.4       # [-]     air ratio of specific heats

# -- Regulatory figures (cited, not placeholders) ----------------------------
H_CRUISE_REQ    = 3810.0     # [m]  12,500 ft - FAR 91.211 supplemental-O2 threshold (requirement)
H_CABIN_DEFAULT = 2440.0     # [m]  8,000 ft  - FAR 25.841 max cabin altitude (standard)
MDOT_PER_OCC    = 0.25 / 60  # [kg/s/occupant] FAR 25.831: 0.55 lb/min fresh air per occupant

# -- Design differential / shell ---------------------------------------------
K_RELIEF   = 1.10            # [-]  relief-valve tolerance on dp                  PLACEHOLDER
SF_PRESS   = 2.0             # [-]  safety factor on dp (FAR 25.365-style)        PLACEHOLDER
K_NONIDEAL = 1.6             # [-]  frames/joints/cutout-reinforcement factor     PLACEHOLDER
#    (1.3-2.0 covers non-ideal shell mass; 1.6 mid-range)

# -- Materials: working stress sigma_w [Pa] (incl. fatigue knockdown for pressure
#    cycles), density rho [kg/m^3], unpressurized baseline skin t_baseline [m].
#    Both sigma_w values are knockdown placeholders.                              PLACEHOLDER
MATERIALS = {
    "AL2024":  dict(sigma_w=100e6, rho=2780.0, t_baseline=T_SKIN_MIN_AL),
    "CFRP_QI": dict(sigma_w=200e6, rho=1600.0, t_baseline=T_SKIN_MIN_CFRP),
}

# -- Cabin geometry from pax (circular cross-section) -------------------------
N_ABREAST   = 2       # [-]  seats abreast                                        PLACEHOLDER
SEAT_WIDTH  = 0.55    # [m]  seat width incl. armrest                             PLACEHOLDER
CLEARANCE   = 0.50    # [m]  aisle + wall clearance across the section            PLACEHOLDER
SEAT_PITCH  = 0.95    # [m]  row pitch                                            PLACEHOLDER
L_EXTRA     = 1.00    # [m]  extra cylinder length (panel, baggage, systems)      PLACEHOLDER

# -- Seal / cutout floor -------------------------------------------------------
N_DOOR          = 2     # [-]  cabin doors                                        PLACEHOLDER
DOOR_W, DOOR_H  = 0.80, 1.20   # [m]  door cutout width x height                  PLACEHOLDER
WINDOWS_PER_PAX = 1     # [-]  windows per passenger                              PLACEHOLDER
WIN_W, WIN_H    = 0.30, 0.25   # [m]  window cutout width x height                PLACEHOLDER
K_SEAL          = 0.5   # [kg/m] seal + local doubler mass per cutout perimeter   PLACEHOLDER

# -- Air-supply system (electric, no bleed) ------------------------------------
N_CREW          = 1       # [-]   crew counted as occupants (FAR 25.831 flow)     PLACEHOLDER
T_CABIN         = 293.15  # [K]   cabin temperature for leakage density           PLACEHOLDER
ACH_LEAK        = 1.5     # [1/h] leakage air changes per hour (TBD)              PLACEHOLDER
ETA_IS          = 0.70    # [-]   compressor isentropic efficiency                PLACEHOLDER
ETA_MOTOR       = 0.85    # [-]   drive motor + electronics efficiency            PLACEHOLDER
P_SPECIFIC_COMP = 1500.0  # [W/kg] electric compressor specific power             PLACEHOLDER
M_COMP_MIN      = 2.0     # [kg]  floor mass of the smallest practical unit       PLACEHOLDER
M_OUTFLOW_VALVE = 1.5     # [kg]  outflow valve                                   PLACEHOLDER
M_RELIEF_VALVE  = 0.8     # [kg]  each of positive + negative relief valves       PLACEHOLDER
M_CONTROLLER    = 0.5     # [kg]  cabin-pressure controller                       PLACEHOLDER
DUCT_KG_PER_M   = 0.6     # [kg/m] distribution ducting                           PLACEHOLDER


# -- ISA atmosphere (troposphere) ----------------------------------------------

def isa_temperature(h_m):
    """ISA tropospheric temperature [K]: T(h) = 288.15 - 0.0065 h."""
    return T0_ISA - L_ISA * h_m


def isa_pressure(h_m):
    """ISA tropospheric pressure [Pa]: p(h) = p0 (T/T0)^(g/(R L))."""
    return P0_ISA * (isa_temperature(h_m) / T0_ISA) ** (G_ISA / (R_AIR * L_ISA))


# -- Part A: structural penalty -------------------------------------------------

def design_dp(h_cruise, h_cabin):
    """Design differential pressure [Pa]: dp = max(p(h_cabin) - p(h_cruise), 0) * k_relief.
    Zero when the cabin altitude is at or above the cruise altitude (no pressurization)."""
    return max(isa_pressure(h_cabin) - isa_pressure(h_cruise), 0.0) * K_RELIEF


def cabin_geometry(n_pax):
    """Pressure-cabin geometry from the pax count: circular cylinder of internal radius r
    (seats-abreast width + clearance) and length l_cyl (rows * pitch + extra), closed by
    two hemispherical end domes. Returns r, l_cyl, surfaces and volume."""
    r = (N_ABREAST * SEAT_WIDTH + CLEARANCE) / 2.0
    n_rows = math.ceil(n_pax / N_ABREAST)
    l_cyl = n_rows * SEAT_PITCH + L_EXTRA
    s_cyl = 2.0 * math.pi * r * l_cyl
    s_dome = 4.0 * math.pi * r**2                 # two hemispheres = one full sphere
    v_cab = math.pi * r**2 * l_cyl + (4.0 / 3.0) * math.pi * r**3
    return dict(r=r, l_cyl=l_cyl, s_cyl=s_cyl, s_dome=s_dome, v_cab=v_cab)


def shell_penalty(dp, geom, mat):
    """Pressurization shell mass increment [kg] over the unpressurized baseline skin.

    Hoop stress sizes the cylinder: t_cyl = SF * dp * r / sigma_w; the hemispherical dome
    carries half the membrane stress, t_dome = t_cyl / 2. Only the increment over the
    min-gauge/flight-loads baseline counts; k_nonideal covers frames, joints and cutout
    reinforcement. Returns (dm_shell, detail dict)."""
    t_cyl = SF_PRESS * dp * geom["r"] / mat["sigma_w"]
    t_dome = t_cyl / 2.0
    dt_cyl = max(t_cyl - mat["t_baseline"], 0.0)
    dt_dome = max(t_dome - mat["t_baseline"], 0.0)
    dm_shell = mat["rho"] * (geom["s_cyl"] * dt_cyl + geom["s_dome"] * dt_dome) * K_NONIDEAL
    return dm_shell, dict(t_cyl_req=t_cyl, t_dome_req=t_dome,
                          dt_cyl=dt_cyl, dt_dome=dt_dome)


def seal_mass(n_pax):
    """Seal/cutout floor mass [kg]: k_seal * total cutout perimeter (doors + windows).
    Nonzero even when the shell thickness increment is zero."""
    per_door = 2.0 * (DOOR_W + DOOR_H)
    per_win = 2.0 * (WIN_W + WIN_H)
    perimeter = N_DOOR * per_door + WINDOWS_PER_PAX * n_pax * per_win
    return K_SEAL * perimeter


# -- Part B: air supply (electric, no bleed) -------------------------------------

def air_supply(n_pax, v_cab, h_cruise, h_cabin, t_above_cabin_alt_s):
    """Electric cabin-air supply: mass flow, compressor power/mass, secondary hardware,
    ducting and mission energy. Returns a detail dict.

    Mass flow is the larger of the FAR 25.831 fresh-air requirement (0.55 lb/min per
    occupant, crew included) and the leakage make-up (cabin density * volume * ACH).
    Compressor power is isentropic compression from ambient at the ceiling to cabin
    pressure, divided by the isentropic and motor efficiencies. The compressor runs only
    above the cabin altitude, so E = P * t_above_cabin_alt_s."""
    p_amb = isa_pressure(h_cruise)
    t_amb = isa_temperature(h_cruise)
    p_cab = isa_pressure(h_cabin)

    mdot_occ = (n_pax + N_CREW) * MDOT_PER_OCC
    rho_cab = p_cab / (R_AIR * T_CABIN)
    mdot_leak = rho_cab * v_cab * ACH_LEAK / 3600.0
    mdot = max(mdot_occ, mdot_leak)

    pr = p_cab / p_amb
    if pr > 1.0:
        p_comp = (mdot * CP_AIR * t_amb * (pr ** ((GAMMA_AIR - 1.0) / GAMMA_AIR) - 1.0)
                  / (ETA_IS * ETA_MOTOR))
    else:
        p_comp = 0.0                                # cabin at/above ambient: nothing to do

    m_comp = max(p_comp / P_SPECIFIC_COMP, M_COMP_MIN)
    m_valves = M_OUTFLOW_VALVE + 2.0 * M_RELIEF_VALVE + M_CONTROLLER
    return dict(mdot=mdot, mdot_occ=mdot_occ, mdot_leak=mdot_leak,
                mdot_governing="occupants" if mdot_occ >= mdot_leak else "leakage",
                pressure_ratio=pr, p_comp_w=p_comp, m_comp=m_comp, m_valves=m_valves,
                e_press_j=p_comp * t_above_cabin_alt_s)


# -- Public entry ------------------------------------------------------------------

def pressurization_sizing(mtow, n_pax, t_above_cabin_alt_s, material,
                          h_cruise=H_CRUISE_REQ, h_cabin=H_CABIN_DEFAULT,
                          cross_section="circular"):
    """Cabin-pressurization mass [kg] and mission energy [J] penalties.

    Parameters
    ----------
    mtow : float                 trial MTOW [kg]; used ONLY for mass-fraction reporting
                                 (geometry is pax-driven, energy feeds the battery loop)
    n_pax : int                  passengers (crew added internally for the air flow)
    t_above_cabin_alt_s : float  mission time spent above the cabin altitude [s]
    material : str or dict       MATERIALS key ("AL2024" / "CFRP_QI") or a material dict
                                 with sigma_w [Pa], rho [kg/m^3], t_baseline [m]
    h_cruise : float             cruise altitude [m]; requirement = 3810 m (12,500 ft)
    h_cabin : float              cabin pressure altitude [m]; default 2440 m (8,000 ft);
                                 the key sweep parameter
    cross_section : str          must be "circular" (hard model assumption); a warning is
                                 raised otherwise and the membrane result is NOT valid

    Returns
    -------
    (m_press_kg, E_press_J, info_dict)
    """
    if cross_section != "circular":
        warnings.warn(
            "pressurization_sizing: model assumes a circular (membrane-stressed) section. "
            "Flat panels carry dp in BENDING (t ~ w*sqrt(dp/sigma)) and are roughly an "
            "order of magnitude heavier - this result is NOT valid for a non-circular "
            f"cabin (got cross_section={cross_section!r}).",
            stacklevel=2,
        )

    mat = MATERIALS[material] if isinstance(material, str) else material

    dp = design_dp(h_cruise, h_cabin)
    geom = cabin_geometry(n_pax)
    dm_shell, shell = shell_penalty(dp, geom, mat)
    m_seal = seal_mass(n_pax)
    supply = air_supply(n_pax, geom["v_cab"], h_cruise, h_cabin, t_above_cabin_alt_s)

    # ducting run: one supply line along the cabin plus the dome offsets
    duct_len = geom["l_cyl"] + 2.0 * geom["r"]
    m_duct = DUCT_KG_PER_M * duct_len

    m_press = dm_shell + m_seal + supply["m_comp"] + supply["m_valves"] + m_duct
    e_press = supply["e_press_j"]

    # Indicative battery-mass equivalent (the loop closes the real feedback). Mirrors
    # energy.battery_mass: deliverable = E x (1 + reserve), cells = deliverable/E_CELL,
    # pack = cell mass / cell-to-pack -- smooth (no whole-string round-up).
    e_press_wh = e_press / 3600.0
    n_cells_equiv = e_press_wh * (1.0 + RESERVE_FRAC) / E_CELL_WH
    dm_batt_equiv = n_cells_equiv * M_CELL_KG / CELL_TO_PACK

    info = dict(
        dp_pa=dp,
        p_cabin_pa=isa_pressure(h_cabin), p_ambient_pa=isa_pressure(h_cruise),
        geom=geom,
        **shell,
        dm_shell=dm_shell, m_seal=m_seal,
        m_comp=supply["m_comp"], m_valves=supply["m_valves"], m_duct=m_duct,
        duct_len_m=duct_len,
        mdot_kg_s=supply["mdot"], mdot_governing=supply["mdot_governing"],
        pressure_ratio=supply["pressure_ratio"], p_comp_w=supply["p_comp_w"],
        e_press_j=e_press, dm_batt_equiv_kg=dm_batt_equiv,
        m_press_kg=m_press, mass_fraction=m_press / mtow if mtow > 0 else float("nan"),
        material=material if isinstance(material, str) else "custom",
        h_cruise_m=h_cruise, h_cabin_m=h_cabin,
    )
    return m_press, e_press, info


# -- Standalone report --------------------------------------------------------------

T_ABOVE_CABIN_DEMO = 1200.0   # [s] demo mission time above cabin altitude         PLACEHOLDER


def main():
    # Follow the converged design (same pattern as the landing-gear sketch).
    import class_II_sizing.mtow_sizing as mtow_mod
    from parameters import N_PAX
    mtow_mod.load_final_design_state()
    mtow = mtow_mod.MTOW_FINAL

    print("=" * 72)
    print(f"PRESSURIZATION SIZING   (MTOW = {mtow:.0f} kg, {N_PAX} pax, "
          f"cruise {H_CRUISE_REQ:.0f} m = FL125)")
    print("=" * 72)
    header = ("material  h_cab[m]  dp[kPa]  dm_shell  m_seal  m_comp  m_valve  m_duct"
              "  m_press[kg]  P[W]  E[kJ]  dm_batt[kg]")
    print(header)
    print("-" * len(header))
    for mat in ("AL2024", "CFRP_QI"):
        for h_cab in (1830.0, H_CABIN_DEFAULT, 3000.0):
            m, e, i = pressurization_sizing(mtow, N_PAX, T_ABOVE_CABIN_DEMO, mat,
                                            h_cabin=h_cab)
            print(f"{mat:8s} {h_cab:8.0f} {i['dp_pa']/1000:8.2f} {i['dm_shell']:9.2f}"
                  f" {i['m_seal']:7.2f} {i['m_comp']:7.2f} {i['m_valves']:8.2f}"
                  f" {i['m_duct']:7.2f} {m:12.2f} {i['p_comp_w']:6.0f}"
                  f" {e/1000:6.1f} {i['dm_batt_equiv_kg']:11.3f}")
    print()
    m, e, i = pressurization_sizing(mtow, N_PAX, T_ABOVE_CABIN_DEMO, "AL2024")
    print(f"Baseline (AL2024, 8,000 ft cabin): m_press = {m:.1f} kg "
          f"({100*i['mass_fraction']:.2f}% MTOW), E_press = {e/1000:.0f} kJ, "
          f"flow governed by {i['mdot_governing']}")
    print("NOTE: FL125 is the FAR 91.211 O2 threshold - an unpressurized + O2-mask "
          "configuration would delete most of this mass (trade to revisit).")


if __name__ == "__main__":
    main()
