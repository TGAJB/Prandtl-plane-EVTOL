
"""
Electric Environmental Control System (E-ECS) sizing model
============================================================

Conceptual / preliminary-design sizing of the cabin pressurization &
conditioning system for the Folding Prandtl eVTOL.

Method adapted from:
    M. Fioriti & F. Di Fede, "A Design Model for Electric Environmental
    Control System in Aircraft Conceptual and Preliminary Design",
    Int. Review of Aerospace Engineering, Vol. 16 No. 2, 2023.
    https://doi.org/10.15866/irease.v16i2.23379

What this implements (the subset that matters for a 4-pax, low-altitude,
*pressurized* eVTOL):
    Module 1  - Cabin heat load          (paper Eq. 1-11)
    Module 1  - Required cabin air flow   (paper Eq. 12 + reg. minimum)
    Module 2  - Air-intake ram recovery   (paper Eq. 13-14)
    Module 2  - Dedicated compressor power(paper Eq. 15)
    Module 2  - Electric motor power      (paper Eq. 16)

What this deliberately OMITS (full air-cycle-machine cooling train: PHE,
SHE, ACM compressor/turbine, condenser, mixers, fan -- paper Eq. 17-34).
At a 12,500 ft cruise / 6,000 ft destination the compression temperature
rise is modest, so the ACM is not the sizing driver. The compressor outlet
temperature is reported so you can confirm this assumption; if it climbs
toward ~80-100 C you would need to add a heat exchanger.

KNOWN LIMITATIONS (conceptual stage -- defer to detailed design):
  - Intake ram recovery (Eq. 13-14) assumes CLEAN FREESTREAM flow at the
    flight Mach number, exactly as in the source paper. Our vehicle has six
    tilt-rotors, so the real intake will likely sit in a propeller
    slipstream. The ECS intake location relative to the rotor disks is TBD,
    so slipstream interaction is NOT modelled here. Expected impact:
      * Cruise (the pressurization SIZING case): negligible. At M ~ 0.17
        freestream ram recovery is already almost nothing (p1 ~ ambient),
        so any slipstream pressure boost is second-order. If the intake IS
        washed by a wake the true intake pressure is slightly higher, which
        makes this model mildly CONSERVATIVE (over-estimates beta_c / power).
      * Hover / takeoff / landing: freestream ram is zero (M ~ 0) and the
        real intake would see rotor downwash (possibly energized, possibly
        warm recirculated air). These phases are NOT pressurization-driven
        here (beta_c = 1.0, cabin p <= ambient), so the missing physics does
        not affect the current sizing. It WOULD matter for a future hot-day
        ground-cooling analysis, where recirculated rotor wash drives the
        intake air temperature and cooling load.
    Action: refine intake recovery once the intake position is fixed.

Equation numbers in comments refer to the paper.

Author: (your group) -- conceptual design phase
"""

from __future__ import annotations
from dataclasses import dataclass, field
from math import log, sqrt
import csv

# --------------------------------------------------------------------------- #
# Physical constants and air properties
# --------------------------------------------------------------------------- #
G0 = 9.80665           # gravity [m/s^2]  (not used directly, kept for clarity)
R_AIR = 287.05         # specific gas constant for air [J/(kg K)]
CP_AIR = 1004.0        # specific heat at constant pressure [J/(kg K)]
GAMMA = 1.4            # heat capacity ratio of air [-]
PR = 0.71             # Prandtl number of air [-]

# ISA sea-level reference
T0_ISA_SL = 288.15     # [K]
P0_ISA_SL = 101325.0   # [Pa]
LAPSE = 0.0065         # ISA temperature lapse rate [K/m]

FT2M = 0.3048          # feet -> metres


# --------------------------------------------------------------------------- #
# Atmosphere helper
# --------------------------------------------------------------------------- #
def isa(altitude_ft: float, isa_offset_K: float = 0.0) -> tuple[float, float, float]:
    """
    ISA troposphere (valid to 11 km / ~36 kft) with an optional ISA
    temperature offset (e.g. +25 for a hot day, -20 for a cold day).

    Returns (static_temperature [K], static_pressure [Pa], density [kg/m^3]).
    The temperature offset shifts T but, by convention here, pressure is
    computed from the *standard* temperature profile (offset applied to T only),
    which is how design-case 'hot/cold day' deltas are usually handled.
    """
    h = altitude_ft * FT2M
    T_std = T0_ISA_SL - LAPSE * h
    p = P0_ISA_SL * (T_std / T0_ISA_SL) ** (G0 / (LAPSE * R_AIR))
    T = T_std + isa_offset_K
    rho = p / (R_AIR * T)
    return T, p, rho


# --------------------------------------------------------------------------- #
# Avionics electrical-power build-up (component method)
# --------------------------------------------------------------------------- #
# Purpose: replace the single P_avionics_W placeholder with a traceable,
# bottom-up sum, mirroring the Class-I MASS build-up. Each row is one box
# from the electrical/comms block diagrams (report Figs 8.1, 8.4, 8.5).
#
# Two flags per component matter for the ECS heat balance:
#   qty                  : how many units
#   watts                : continuous electrical draw per unit [W]
#   liquid_cooled        : if True, the box rejects its heat to the radiator
#                          loop (report Fig. 8.2) and does NOT dump heat into
#                          the cabin/avionics-bay air -> excluded from the ECS
#                          cabin heat load. If False, its heat counts.
#
# The 'watts' values below are CONCEPTUAL-STAGE bands taken from public
# datasheets of comparable units (see source notes). Replace each with the
# real number from Control/Power once parts are selected. Sources sampled:
#   IMU/AHRS    : ~0.5-2 W   (Inertial Labs AHRS-10, IMU-P datasheets)
#   GNSS        : ~1-3 W     (typical aviation GNSS receiver)
#   ADS-B xpdr  : ~1.5-3 W   (uAvionix ping200X 1.5 W; L3Harris ~3 W)
#   LiDAR       : ~90 W      (scanning aviation LiDAR, datasheet)
#   Radar       : ~20-40 W   (small mmWave/avoidance radar module)
#   SATCOM term : ~100-200 W (aviation LEO/Ku-Ka terminal, transmit) <-- driver
#   Cellular    : ~10-20 W   (cellular modem + PA)
#   Cameras     : ~3-5 W ea  (interior + exterior feed cameras)
#   Flight comp : ~40-60 W ea(flight-grade computer; x2-3 for redundancy)
#   Air-data    : ~2-5 W     (pitot/AoA/baro heating excluded; sensing only)
#
# NOTE on the r_avionics 0.95 factor (paper Eq. 3): that accounts for the
# small fraction of electrical power radiated away as RF rather than heat.
# Do NOT also subtract transmit power by hand -- it is already covered.
# The liquid_cooled flag is a SEPARATE, larger effect (whole-box removal).

@dataclass
class AvionicsItem:
    name: str
    qty: int
    watts: float            # continuous electrical draw per unit [W]
    liquid_cooled: bool = False
    note: str = ""

    @property
    def electrical_W(self) -> float:
        return self.qty * self.watts

    @property
    def cabin_heat_W(self) -> float:
        """Heat dumped into cabin/bay air (0 if on the radiator loop)."""
        return 0.0 if self.liquid_cooled else self.electrical_W


def default_avionics() -> list[AvionicsItem]:
    """
    Build-up seeded from the report block diagrams. EDIT freely:
    set real wattages, quantities, and especially the liquid_cooled flags
    once Control/Power confirm which boxes sit on the Fig. 8.2 cooling loop.
    Defaults below assume the high-power computers and SATCOM are AIR-cooled
    into the bay (conservative for the ECS load); flip to liquid_cooled=True
    to see the cabin-load reduction.
    """
    return [
        AvionicsItem("Flight computer",      3, 50.0, False, "redundant set (RAMS)"),
        AvionicsItem("IMU",                  2, 2.0,  False, "redundant"),
        AvionicsItem("GNSS receiver",        2, 3.0,  False, "redundant"),
        AvionicsItem("Air-data (pitot/AoA/baro)", 1, 5.0, False, ""),
        AvionicsItem("ADS-B transponder",    1, 3.0,  False, ""),
        AvionicsItem("LiDAR",                1, 90.0, False, "scanning aviation LiDAR, datasheet"),
        AvionicsItem("Radar",                1, 30.0, False, "vehicle avoidance"),
        AvionicsItem("SATCOM terminal",      1, 150.0, False, "LEO link; main driver"),
        AvionicsItem("Cellular modem",       1, 15.0, False, "backup link"),
        AvionicsItem("Exterior cameras",     4, 4.0,  False, "remote-pilot feed"),
        AvionicsItem("Interior camera",      1, 4.0,  False, ""),
    ]


def sum_avionics(items: list[AvionicsItem], margin: float = 0.20) -> dict:
    """
    Returns total electrical draw, total heat dumped into cabin air, and the
    same with a design margin applied. `margin` is a contingency on top of the
    summed components (0.20 = +20%), consistent with conceptual-stage practice.
    """
    elec = sum(it.electrical_W for it in items)
    cabin = sum(it.cabin_heat_W for it in items)
    return {
        "electrical_W": elec,
        "electrical_W_margined": elec * (1.0 + margin),
        "cabin_heat_W": cabin,
        "cabin_heat_W_margined": cabin * (1.0 + margin),
        "margin": margin,
    }


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
@dataclass
class Aircraft:
    """
    Geometry & equipment inputs.

    Values tagged [CONST] come from the team shared constants file.
    Values tagged [REPORT] come from the midterm report.
    Values tagged [ASSUMED] still need an owner to confirm.

    NOTE -- geometry conflict to resolve with Structures:
      constants file gives L_FUS = 7.0 m, PER_FUS_MAX = 12.0 m
      midterm report Table 5.7 gives fuselage 9.0 x 2.0 x 2.0 m
    The ECS conduction term is sensitive to the effective radius, so pick
    one source. Defaults below use the report cross-section (2 m height ->
    r ~ 1.0 m) and the constants-file fuselage length (7.0 m). Set
    `use_perimeter_for_radius=True` to instead derive radius from PER_FUS_MAX.
    """
    n_pax: int = 4                  # [CONST] N_PAX = 4
    n_crew: int = 0                 # [CONST] autonomous -> W_CREW = 0
    n_cabin_crew: int = 0

    # Fuselage as a 3-layer cylinder (paper Fig. 2)
    cabin_length_m: float = 6.0     # [ASSUMED] conditioned cabin length;
                                    #   ~0.65-0.75 x fuselage length (paper)
    fuselage_length_m: float = 7.0  # [CONST] L_FUS = 7.0
    fuselage_perimeter_m: float = 12.0   # [CONST] PER_FUS_MAX = 12.0
    use_perimeter_for_radius: bool = False
    r_inner_m: float = 1.0          # [REPORT] ~half of 2.0 m cabin height
    t_trim_m: float = 0.005         # [ASSUMED] decorative/inner-skin thickness
    t_insul_m: float = 0.035        # [ASSUMED] microlite insulation (25-51 mm)
    t_outer_m: float = 0.002        # [ASSUMED] outer skin thickness
    k_trim: float = 0.3             # [ASSUMED] composite inner skin [W/(m K)]
    k_insul: float = 0.04           # [ASSUMED] microlite [W/(m K)]
    k_outer: float = 0.3            # [ASSUMED] composite outer skin [W/(m K)]

    # Transparent surfaces (windows + windscreen) for solar load (Eq. 11)
    window_area_m2: float = 1.5     # [ASSUMED] confirm with structures/cabin
    tau_glass: float = 0.8          # [ASSUMED] glass transmissivity
    f_rad: float = 0.5              # [ASSUMED] windows radiation factor (~0.5)

    # Electrical equipment (Eq. 3-5).
    # P_avionics_W can be set directly, OR computed from a component build-up
    # via the `avionics_items` list (recommended -- traceable & parametric).
    # If avionics_items is non-empty, the build-up's CABIN heat sum overrides
    # P_avionics_W (so liquid-cooled boxes are correctly excluded).
    P_avionics_W: float = 1500.0    # [FALLBACK] used only if avionics_items empty
    avionics_margin: float = 0.20   # [ASSUMED] contingency on the build-up
    r_avionics: float = 0.95        # [ASSUMED] avionic thermal-load fraction
    P_display_W: float = 8.4        # [ASSUMED] per-display power (9-inch LCD)
    r_light: float = 0.95           # [ASSUMED] lighting thermal-load fraction
    P_reading_light_W: float = 3.0  # [ASSUMED] per reading light
    P_floor_light_Wpm: float = 2.0  # [ASSUMED] floor strip [W/m]
    P_aisle_light_Wpm: float = 2.0  # [ASSUMED] aisle strip [W/m]
    n_floor_strips: int = 1
    n_aisle_strips: int = 1
    avionics_items: list = field(default_factory=default_avionics)

    @property
    def P_avionics_cabin_W(self) -> float:
        """
        Avionics electrical power that becomes CABIN heat. If a component
        build-up is present, use its margined cabin-heat sum (liquid-cooled
        boxes excluded); otherwise fall back to the flat P_avionics_W.
        """
        if self.avionics_items:
            return sum_avionics(self.avionics_items, self.avionics_margin)["cabin_heat_W_margined"]
        return self.P_avionics_W

    @property
    def r_inner_effective_m(self) -> float:
        """Internal radius used for the conduction model."""
        if self.use_perimeter_for_radius:
            return self.fuselage_perimeter_m / (2.0 * 3.141592653589793)
        return self.r_inner_m

    @property
    def r_outer_m(self) -> float:
        return self.r_inner_effective_m + self.t_trim_m + self.t_insul_m + self.t_outer_m


@dataclass
class Conditioning:
    """Targets and efficiencies (paper Tables III / V style)."""
    T_cab_C: float = 21.0           # desired cabin temp [C] (your REQ-SUB-66: 19-23)
    T_ecs_C: float = 12.0           # vent supply temp when COOLING [C]
    T_ecs_heat_C: float = 45.0      # vent supply temp when HEATING [C]
    min_fresh_air_kg_s_pax: float = 0.0042   # regulatory minimum (paper)
    cabin_pressure_bar: float = 0.8127       # target cabin pressure [bar]
                                             # 0.8127 bar ~ 8000 ft cabin alt (B787 value)

    # Intake + compressor + motor efficiencies (paper)
    eps_intake_ground: float = 1.0      # no ram on ground (Eq. 14)
    eta_is_c: float = 0.77              # compressor isentropic efficiency
    eta_mec_c: float = 0.95             # compressor mechanical efficiency
    eta_motor: float = 0.93             # electric motor efficiency

    G_sun_Wm2: float = 1000.0           # solar irradiance for sunny cases [W/m^2]


@dataclass
class Phase:
    """One mission design point."""
    name: str
    altitude_ft: float
    isa_offset_K: float
    mach: float
    mode: str = "cool"        # "cool" or "heat"
    sunny: bool = True        # apply solar load through windows?
    on_ground: bool = False   # if True, intake efficiency = 1 (no ram)
    E_prop_kWh: float = 0.0   # propulsive energy from battery sizing [kWh],
                              # optional -- used only for the comparison column


# --------------------------------------------------------------------------- #
# Module 1 -- Cabin heat load (paper Eq. 1-11)
# --------------------------------------------------------------------------- #
def metabolic_heat(ac: Aircraft) -> float:
    """Eq. 2: 130 W/crew, 70 W/pax."""
    return 130.0 * ac.n_crew + 70.0 * ac.n_pax


def avionics_heat(ac: Aircraft) -> float:
    """Eq. 3: r_avionics * (avionics electrical power that becomes cabin heat)."""
    return ac.r_avionics * ac.P_avionics_cabin_W


def display_heat(ac: Aircraft) -> float:
    """Eq. 4: one display per seat (pax + cabin crew)."""
    return (ac.n_pax + ac.n_cabin_crew) * ac.P_display_W


def lights_heat(ac: Aircraft) -> float:
    """Eq. 5."""
    reading = (ac.n_pax + ac.n_crew) * ac.P_reading_light_W
    strips = ac.cabin_length_m * (
        ac.P_floor_light_Wpm * ac.n_floor_strips
        + ac.P_aisle_light_Wpm * ac.n_aisle_strips
    )
    return ac.r_light * (reading + strips)


def fuselage_resistance(ac: Aircraft) -> float:
    """
    Eq. 7: multilayer cylindrical thermal resistance [K/W].
    R = sum_i  ln(r_{i+1}/r_i) / (2*pi*l_cabin*k_i)
    """
    radii = [
        ac.r_inner_effective_m,
        ac.r_inner_effective_m + ac.t_trim_m,
        ac.r_inner_effective_m + ac.t_trim_m + ac.t_insul_m,
        ac.r_outer_m,
    ]
    ks = [ac.k_trim, ac.k_insul, ac.k_outer]
    R = 0.0
    for i, k in enumerate(ks):
        R += log(radii[i + 1] / radii[i]) / (2.0 * 3.141592653589793 * ac.cabin_length_m * k)
    return R


def conduction_heat(ac: Aircraft, T_skin_ext_K: float, T_cab_K: float) -> float:
    """
    Eq. 6: conduction through the fuselage skin.
    Positive = heat INTO cabin (skin hotter than cabin). We use a simplified
    boundary: external skin at recovery temperature (flight) or ambient+solar
    proxy (ground). Sign is handled by the caller via the (ext - int) term.
    """
    R = fuselage_resistance(ac)
    return (T_skin_ext_K - T_cab_K) / R


def recovery_temperature(T_static_K: float, mach: float) -> float:
    """Eq. 9-10: T_skin,ext = T_amb (1 + f_r (gamma-1)/2 M^2), f_r = Pr^(1/3)."""
    f_r = PR ** (1.0 / 3.0)
    return T_static_K * (1.0 + f_r * (GAMMA - 1.0) / 2.0 * mach ** 2)


def solar_window_heat(ac: Aircraft, cond: Conditioning, sunny: bool) -> float:
    """Eq. 11: solar radiation through transparent surfaces."""
    if not sunny:
        return 0.0
    return ac.tau_glass * cond.G_sun_Wm2 * ac.f_rad * ac.window_area_m2


def total_heat_load(ac: Aircraft, cond: Conditioning, phase: Phase) -> dict:
    """
    Eq. 1: sum of all contributors. Returns a breakdown dict [W].
    Note: conduction can be negative in cold conditions (heat leaving cabin),
    which correctly turns the problem into a heating case.
    """
    T_static, _, _ = isa(phase.altitude_ft, phase.isa_offset_K)
    T_cab_K = cond.T_cab_C + 273.15

    if phase.on_ground:
        # On the ground the external skin is driven by ambient + sun; use a
        # simple proxy: skin ~ ambient (solar effect captured separately via
        # windows term; full ground skin balance Eq. 8 is omitted for Class-I).
        T_skin_ext = T_static
    else:
        T_skin_ext = recovery_temperature(T_static, phase.mach)

    q_met = metabolic_heat(ac)
    q_avi = avionics_heat(ac)
    q_dis = display_heat(ac)
    q_lig = lights_heat(ac)
    q_cond = conduction_heat(ac, T_skin_ext, T_cab_K)
    q_sun = solar_window_heat(ac, cond, phase.sunny)

    q_tot = q_met + q_avi + q_dis + q_lig + q_cond + q_sun
    return {
        "metabolic": q_met,
        "avionics": q_avi,
        "displays": q_dis,
        "lights": q_lig,
        "conduction": q_cond,
        "solar_windows": q_sun,
        "total": q_tot,
        "T_skin_ext_K": T_skin_ext,
    }


# --------------------------------------------------------------------------- #
# Module 1 -- required cabin air mass flow (paper Eq. 12 + reg. floor)
# --------------------------------------------------------------------------- #
def required_mass_flow(ac: Aircraft, cond: Conditioning, q_tot: float, mode: str) -> dict:
    """
    Eq. 12: m_ECS = q_tot / [cp (T_ECS - T_cab)].
    Then enforce the regulatory minimum fresh-air floor.
    """
    T_cab = cond.T_cab_C
    T_ecs = cond.T_ecs_C if mode == "cool" else cond.T_ecs_heat_C
    dT = T_ecs - T_cab
    if abs(dT) < 1e-6:
        m_thermal = 0.0
    else:
        # use |q_tot| / |dT|: a cooling load (q_tot>0) needs cold supply (dT<0),
        # a heating load (q_tot<0) needs hot supply (dT>0); magnitude is what sizes flow
        m_thermal = abs(q_tot) / (CP_AIR * abs(dT))

    n_occ = ac.n_pax + ac.n_crew + ac.n_cabin_crew
    m_min = cond.min_fresh_air_kg_s_pax * max(n_occ, ac.n_pax)
    m_ecs = max(m_thermal, m_min)
    return {
        "m_thermal": m_thermal,
        "m_min_regulation": m_min,
        "m_ecs": m_ecs,
        "driven_by": "regulation" if m_min >= m_thermal else "thermal",
    }


# --------------------------------------------------------------------------- #
# Module 2 -- intake ram recovery (paper Eq. 13-14)
# --------------------------------------------------------------------------- #
def intake_state(phase: Phase, cond: Conditioning) -> tuple[float, float]:
    """
    Eq. 13-14: total temperature and pressure at intake exit.
    Returns (T1_total [K], p1_total [Pa]).

    LIMITATION: assumes clean FREESTREAM flow at phase.mach (as in the source
    paper). Our six tilt-rotors mean the real intake may sit in a propeller
    slipstream; that interaction is NOT modelled (intake location TBD). Impact
    is negligible for the cruise pressurization sizing case (M ~ 0.17, ram
    already ~ nil) and irrelevant for hover/landing (not pressurization-driven
    here). See module docstring 'KNOWN LIMITATIONS'. Refine once intake
    position is fixed in detailed design.
    """
    T0, p0, _ = isa(phase.altitude_ft, phase.isa_offset_K)
    T1 = T0 * (1.0 + (GAMMA - 1.0) / 2.0 * phase.mach ** 2)            # Eq. 13
    eps = 1.0 if phase.on_ground else cond.eps_intake_ground          # ~1 at low M
    p1 = eps * p0 * (T1 / T0) ** (GAMMA / (GAMMA - 1.0))              # Eq. 14
    return T1, p1


# --------------------------------------------------------------------------- #
# Module 2 -- dedicated compressor + motor (paper Eq. 15-16)
# --------------------------------------------------------------------------- #
def compressor_power(m_ecs: float, T1_total: float, p1_total: float,
                     p_cabin_Pa: float, cond: Conditioning) -> dict:
    """
    Eq. 15: P_c = (m_ECS cp T1 / eta_is_c) [ beta_c^((g-1)/g) - 1 ]
    Eq. 16: P_motor = P_c / (eta_motor eta_mec)
    Also returns compressor exit temperature so the no-ACM assumption can be
    checked (adiabatic compression, including isentropic-efficiency penalty).
    """
    beta_c = max(p_cabin_Pa / p1_total, 1.0)          # never < 1 (no compression needed)
    exponent = (GAMMA - 1.0) / GAMMA

    P_c = (m_ecs * CP_AIR * T1_total / cond.eta_is_c) * (beta_c ** exponent - 1.0)  # Eq.15
    P_motor = P_c / (cond.eta_motor * cond.eta_mec_c)                                # Eq.16

    # Real (with-efficiency) compressor exit temperature:
    # T_out = T1 [1 + (1/eta_is)(beta^((g-1)/g) - 1)]
    T_out = T1_total * (1.0 + (1.0 / cond.eta_is_c) * (beta_c ** exponent - 1.0))

    return {
        "beta_c": beta_c,
        "P_c_W": P_c,
        "P_motor_W": P_motor,
        "T_compressor_exit_K": T_out,
        "T_compressor_exit_C": T_out - 273.15,
    }


# --------------------------------------------------------------------------- #
# Full single-phase evaluation
# --------------------------------------------------------------------------- #
def evaluate_phase(ac: Aircraft, cond: Conditioning, phase: Phase) -> dict:
    loads = total_heat_load(ac, cond, phase)
    # If conduction makes the net load negative, it's really a heating phase
    mode = phase.mode
    flow = required_mass_flow(ac, cond, loads["total"], mode)
    _, p_amb, _ = isa(phase.altitude_ft, phase.isa_offset_K)
    p_cabin = cond.cabin_pressure_bar * 1e5
    T1, p1 = intake_state(phase, cond)
    comp = compressor_power(flow["m_ecs"], T1, p1, p_cabin, cond)

    return {
        "phase": phase.name,
        "altitude_ft": phase.altitude_ft,
        "isa_offset_K": phase.isa_offset_K,
        "mach": phase.mach,
        "p_ambient_bar": p_amb / 1e5,
        "p_cabin_bar": cond.cabin_pressure_bar,
        "q_total_W": loads["total"],
        "q_breakdown": {k: v for k, v in loads.items()
                        if k not in ("total", "T_skin_ext_K")},
        "m_ecs_kg_s": flow["m_ecs"],
        "m_flow_driven_by": flow["driven_by"],
        "beta_c": comp["beta_c"],
        "P_compressor_W": comp["P_c_W"],
        "P_motor_W": comp["P_motor_W"],
        "T_comp_exit_C": comp["T_compressor_exit_C"],
        "E_prop_kWh": phase.E_prop_kWh,
    }


# --------------------------------------------------------------------------- #
# Segmented phase support (e.g. climb crossing into the pressurized regime)
# --------------------------------------------------------------------------- #
def make_altitude_sweep_phases(name: str, alt_start_ft: float, alt_end_ft: float,
                               isa_offset_K: float, mach: float, mode: str,
                               sunny: bool, total_duration_s: float,
                               E_prop_kWh: float, n_segments: int = 6) -> list:
    """
    Split a phase that changes altitude (climb or descent) into n equal-time
    sub-segments, each evaluated at its own mid-segment altitude. This captures
    the ramp of pressurization work as the aircraft crosses from ambient-cabin
    altitude up into the pressurized regime, instead of using a single point.

    Returns a list of (Phase, duration_s) for the sub-segments. The propulsive
    energy is distributed evenly across segments only so the per-segment rows
    carry a sensible share; the aggregator re-sums it to the original total.
    """
    seg_dt = total_duration_s / n_segments
    seg_Eprop = E_prop_kWh / n_segments
    step = (alt_end_ft - alt_start_ft) / n_segments
    segs = []
    for i in range(n_segments):
        mid_alt = alt_start_ft + step * (i + 0.5)
        segs.append((
            Phase(f"{name}[{i+1}/{n_segments}]", mid_alt, isa_offset_K, mach,
                  mode, sunny, False, seg_Eprop),
            seg_dt,
        ))
    return segs


def aggregate_segments(seg_results: list, group_name: str) -> dict:
    """
    Collapse a list of per-segment result dicts into one phase-level result,
    summing energy and propulsive energy and taking the peak motor power.
    Keeps the output table at one row per mission phase.
    """
    energy_Wh = sum(r["energy_Wh"] for r in seg_results)
    E_prop = sum(r["E_prop_kWh"] for r in seg_results)
    peak = max(seg_results, key=lambda r: r["P_motor_W"])
    # representative (peak-power) segment for the displayed conditions
    agg = dict(peak)
    agg["phase"] = group_name
    agg["energy_Wh"] = energy_Wh
    agg["E_prop_kWh"] = E_prop
    agg["duration_s"] = sum(r["duration_s"] for r in seg_results)
    agg["_segments"] = seg_results
    return agg


# --------------------------------------------------------------------------- #
# Mission-level driver: size on worst case, integrate for energy
# --------------------------------------------------------------------------- #
def size_mission(ac: Aircraft, cond: Conditioning,
                 phases: list) -> dict:
    """
    phases: list of entries, each either
        (Phase, duration_seconds)                       -- a single phase, or
        ("SEGMENTED", group_name, [(Phase, dt), ...])   -- a phase split into
                                                           altitude sub-segments
    Returns per-phase results, the sizing (max) electric power, and the total
    ECS energy [kWh] to fold into your battery budget.
    """
    results = []
    energy_J = 0.0
    for entry in phases:
        if entry[0] == "SEGMENTED":
            _, group_name, seg_list = entry
            seg_results = []
            for phase, dt_s in seg_list:
                r = evaluate_phase(ac, cond, phase)
                r["duration_s"] = dt_s
                r["energy_Wh"] = r["P_motor_W"] * dt_s / 3600.0
                energy_J += r["P_motor_W"] * dt_s
                seg_results.append(r)
            results.append(aggregate_segments(seg_results, group_name))
        else:
            phase, dt_s = entry
            r = evaluate_phase(ac, cond, phase)
            r["duration_s"] = dt_s
            r["energy_Wh"] = r["P_motor_W"] * dt_s / 3600.0
            energy_J += r["P_motor_W"] * dt_s
            results.append(r)

    P_size = max(r["P_motor_W"] for r in results)
    sizing_phase = max(results, key=lambda r: r["P_motor_W"])["phase"]
    ecs_total_kWh = energy_J / 3.6e6
    prop_total_kWh = sum(r["E_prop_kWh"] for r in results)
    return {
        "per_phase": results,
        "sizing_motor_power_W": P_size,
        "sizing_phase": sizing_phase,
        "total_energy_kWh": ecs_total_kWh,
        "prop_total_kWh": prop_total_kWh,
        "ecs_fraction_of_prop": (ecs_total_kWh / prop_total_kWh
                                 if prop_total_kWh > 0 else None),
    }


# --------------------------------------------------------------------------- #
# Example run -- Folding Prandtl, Nice -> Courchevel
# --------------------------------------------------------------------------- #
def _print_report(summary: dict) -> None:
    print("=" * 74)
    print("E-ECS SIZING REPORT  --  Folding Prandtl eVTOL")
    print("=" * 74)
    for r in summary["per_phase"]:
        print(f"\nPhase: {r['phase']}  "
              f"(alt {r['altitude_ft']:.0f} ft, ISA{r['isa_offset_K']:+.0f}, M{r['mach']:.2f})")
        print(f"  ambient p        : {r['p_ambient_bar']:.3f} bar"
              f"   cabin p: {r['p_cabin_bar']:.3f} bar   beta_c: {r['beta_c']:.3f}")
        print(f"  heat load total  : {r['q_total_W']:8.1f} W")
        for k, v in r["q_breakdown"].items():
            print(f"      {k:<16}: {v:8.1f} W")
        print(f"  ECS air mass flow: {r['m_ecs_kg_s']:.4f} kg/s "
              f"(driven by {r['m_flow_driven_by']})")
        print(f"  compressor power : {r['P_compressor_W']:8.1f} W")
        print(f"  MOTOR power      : {r['P_motor_W']:8.1f} W")
        print(f"  comp. exit temp  : {r['T_comp_exit_C']:6.1f} C "
              f"({'OK, no ACM needed' if r['T_comp_exit_C'] < 60 else 'WARN: consider heat exchanger'})")
        print(f"  duration         : {r['duration_s']:.0f} s "
              f"-> ECS energy {r['energy_Wh']:.1f} Wh ({r['energy_Wh']/1000:.4f} kWh)")
        if r["E_prop_kWh"] > 0:
            frac = (r["energy_Wh"] / 1000.0) / r["E_prop_kWh"] * 100.0
            print(f"  vs propulsion    : {r['E_prop_kWh']:.3f} kWh prop "
                  f"-> ECS is {frac:.2f}% of phase propulsive energy")
    print("\n" + "-" * 74)
    print(f"SIZING (peak) motor power : {summary['sizing_motor_power_W']:.1f} W "
          f"({summary['sizing_motor_power_W']/1000:.2f} kW)  "
          f"-> driven by '{summary['sizing_phase']}'")
    print(f"TOTAL ECS energy for mission: {summary['total_energy_kWh']:.3f} kWh")
    if summary.get("prop_total_kWh"):
        print(f"TOTAL propulsive energy     : {summary['prop_total_kWh']:.2f} kWh")
        print(f"ECS / propulsion            : "
              f"{summary['ecs_fraction_of_prop']*100:.2f} %")
    print("=" * 74)

    # Compact phase comparison table
    if summary.get("prop_total_kWh"):
        print("\nPHASE COMPARISON  (ECS motor energy vs battery propulsive energy)")
        print(f"{'phase':<22}{'ECS [kWh]':>12}{'prop [kWh]':>12}{'ECS % prop':>12}")
        for r in summary["per_phase"]:
            ecs_kwh = r["energy_Wh"] / 1000.0
            if r["E_prop_kWh"] > 0:
                pct = f"{ecs_kwh / r['E_prop_kWh'] * 100:.2f}"
            else:
                pct = "-"
            print(f"{r['phase']:<22}{ecs_kwh:>12.4f}{r['E_prop_kWh']:>12.3f}{pct:>12}")
        print("-" * 58)
        print(f"{'TOTAL':<22}{summary['total_energy_kWh']:>12.4f}"
              f"{summary['prop_total_kWh']:>12.3f}"
              f"{summary['ecs_fraction_of_prop']*100:>11.2f}%")


def _write_csv(summary: dict, path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "alt_ft", "isa_offset_K", "mach", "p_amb_bar",
                    "p_cab_bar", "beta_c", "q_total_W", "m_ecs_kg_s",
                    "P_compressor_W", "P_motor_W", "T_comp_exit_C",
                    "duration_s", "ECS_energy_Wh", "ECS_energy_kWh",
                    "E_prop_kWh", "ECS_pct_of_prop"])
        for r in summary["per_phase"]:
            ecs_kwh = r["energy_Wh"] / 1000.0
            pct = (ecs_kwh / r["E_prop_kWh"] * 100.0) if r["E_prop_kWh"] > 0 else ""
            w.writerow([r["phase"], r["altitude_ft"], r["isa_offset_K"], r["mach"],
                        f"{r['p_ambient_bar']:.4f}", f"{r['p_cabin_bar']:.4f}",
                        f"{r['beta_c']:.4f}", f"{r['q_total_W']:.1f}",
                        f"{r['m_ecs_kg_s']:.5f}", f"{r['P_compressor_W']:.1f}",
                        f"{r['P_motor_W']:.1f}", f"{r['T_comp_exit_C']:.1f}",
                        f"{r['duration_s']:.0f}", f"{r['energy_Wh']:.2f}",
                        f"{ecs_kwh:.5f}", f"{r['E_prop_kWh']:.3f}",
                        f"{pct:.2f}" if pct != "" else ""])


if __name__ == "__main__":
    ac = Aircraft()        # geometry now seeded from team constants (see class)
    cond = Conditioning()  # tweak targets / efficiencies

    # --- Avionics power build-up (component method) --------------------- #
    av = sum_avionics(ac.avionics_items, ac.avionics_margin)
    print("=" * 74)
    print("AVIONICS POWER BUILD-UP (edit watts / qty / liquid_cooled per box)")
    print("=" * 74)
    print(f"{'component':<26}{'qty':>4}{'W/unit':>9}{'elec W':>9}"
          f"{'cooling':>11}{'cabin W':>9}")
    for it in ac.avionics_items:
        cooling = "liquid" if it.liquid_cooled else "air->cabin"
        print(f"{it.name:<26}{it.qty:>4}{it.watts:>9.1f}{it.electrical_W:>9.1f}"
              f"{cooling:>11}{it.cabin_heat_W:>9.1f}")
    print("-" * 74)
    print(f"{'TOTAL electrical':<26}{'':>4}{'':>9}{av['electrical_W']:>9.1f}")
    print(f"{'  + '+str(int(ac.avionics_margin*100))+'% margin':<26}"
          f"{'':>4}{'':>9}{av['electrical_W_margined']:>9.1f}")
    print(f"{'TOTAL into cabin (heat)':<26}{'':>4}{'':>9}{'':>9}"
          f"{'':>11}{av['cabin_heat_W']:>9.1f}")
    print(f"{'  + margin -> ECS input':<26}{'':>4}{'':>9}{'':>9}"
          f"{'':>11}{av['cabin_heat_W_margined']:>9.1f}")
    print(f"\nECS avionics heat term used (x r_avionics {ac.r_avionics}): "
          f"{avionics_heat(ac):.1f} W")
    print("=" * 74 + "\n")

    # Mission phases aligned 1:1 with the BATTERY SIZING phases so ECS energy
    # adds into the same budget. Durations [s] from the team constants file;
    # propulsive energies [kWh] supplied from the battery sizing per phase.
    # Mapping choices (confirmed): cruise ISA-20; takeoff/v-climb at Nice SL,
    # landing at Courchevel 6000 ft; climb SEGMENTED from SL to 12500 ft so the
    # pressurization ramp is resolved (final-design fidelity).
    T_TAKEOFF        = 5.0       # [CONST]
    T_VERTICAL_CLIMB = 25.0      # [CONST]
    T_CLIMB          = 1221.7    # [CONST] (+ T_CLIMB_ACC 20 s folded in)
    T_CRUISE         = 2194.7    # [CONST]
    T_DESCENT        = 311.945   # [CONST]
    T_LANDING        = 71.47     # [CONST]

    # Phase(name, alt_ft, ISA_offset, Mach, mode, sunny, on_ground, E_prop_kWh)
    phases = [
        # takeoff -- vertical, at Nice sea level, hot sunny coastal
        (Phase("Takeoff",        0,     +25, 0.00, "cool", True,  True,  0.686),  T_TAKEOFF),
        # vertical climb -- still near SL, low speed
        (Phase("Vertical climb", 500,   +25, 0.02, "cool", True,  True,  3.28),   T_VERTICAL_CLIMB),
        # climb -- SEGMENTED from SL to cruise altitude so the pressurization
        # ramp is captured (single mid-point would zero it out). 6 sub-segments.
        ("SEGMENTED", "Climb",
         make_altitude_sweep_phases("Climb", 0, 12500, -20, 0.15, "heat",
                                    True, T_CLIMB, 37.78, n_segments=6)),
        # cruise over the Alps, cold
        (Phase("Cruise",         12500, -20, 0.17, "heat", False, False, 36.9),   T_CRUISE),
        # descent toward Courchevel
        (Phase("Descent",        9000,  -20, 0.12, "heat", False, False, 6.58),   T_DESCENT),
        # landing -- vertical, at Courchevel 6000 ft, cold
        (Phase("Landing",        6000,  -20, 0.05, "heat", False, True,  8.36),   T_LANDING),
    ]

    summary = size_mission(ac, cond, phases)
    _print_report(summary)
    _write_csv(summary, "ecs_results.csv")

    # --- Climb sub-segment breakdown (shows the pressurization ramp) ----- #
    for r in summary["per_phase"]:
        if r["phase"] == "Climb" and "_segments" in r:
            print("\nCLIMB SUB-SEGMENT BREAKDOWN (altitude ramp into pressurized regime)")
            print(f"{'segment':<14}{'alt [ft]':>10}{'beta_c':>9}"
                  f"{'motor [W]':>11}{'ECS [Wh]':>10}")
            for s in r["_segments"]:
                print(f"{s['phase']:<14}{s['altitude_ft']:>10.0f}{s['beta_c']:>9.3f}"
                      f"{s['P_motor_W']:>11.1f}{s['energy_Wh']:>10.2f}")
            print(f"{'  climb total':<14}{'':>10}{'':>9}"
                  f"{'':>11}{r['energy_Wh']:>10.2f}")
            break

    # --- Cabin-pressure sensitivity sweep ------------------------------- #
    # Cabin pressure is the dominant lever for a low-altitude eVTOL: it sets
    # beta_c in cruise, which sets the sizing power. Sweep cabin altitude and
    # report peak motor power + mission ECS energy so you can pick a target.
    print("\nCABIN-PRESSURE SENSITIVITY (sizing driver = cruise pressurization)")
    print(f"{'cabin alt [ft]':>15} {'cabin p [bar]':>14} "
          f"{'peak motor [W]':>15} {'ECS energy [kWh]':>17}")
    for cab_alt_ft in (0, 4000, 6000, 8000):
        _, p_cab, _ = isa(cab_alt_ft, 0.0)
        cond_s = Conditioning(cabin_pressure_bar=p_cab / 1e5)
        s = size_mission(ac, cond_s, phases)
        print(f"{cab_alt_ft:>15.0f} {p_cab/1e5:>14.3f} "
              f"{s['sizing_motor_power_W']:>15.1f} {s['total_energy_kWh']:>17.3f}")

    # --- Quick validation hook (paper B787 reference) -------------------- #
    # To run the paper's own validation: set ac to B787 geometry (62.8 m,
    # r~2.88 m, 280 pax) and a FL430 hot-day pressurized case; the model
    # should return a compressor power in the 67-75 kW band the paper cites.