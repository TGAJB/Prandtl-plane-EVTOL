# ============================================================================
# PROJECT PARAMETER SHEET
#
# Part 1 - module-level design constants (sizing, structures, propulsion, ...)
# Part 2 - Vehicle Dynamics parameter sheet (dataclasses, formerly
#          final_characteristics/vehicle_dynamics/vd_parameters.py)
#
# Shared quantities live in Part 1; the VD dataclasses default to them so
# there is a single source of truth per parameter.
# ============================================================================

import math
from dataclasses import dataclass, field

# Physical constants

G          = 9.81          # [m/s^2]  gravitational acceleration
RHO_ORIGIN = 1.225         # [kg/m^3] ISA sea-level air density

# Mission parameters

M_PAYLOAD        = 400.0      # [kg]  fixed payload (4 pax + luggage)
W_CREW           = 0.0        # [kg]  0 for autonomous; 85 if piloted
RANGE_M          = 200000.0   # [m]   design range
D_VERT_DESCENT   = 1676.0     # [m]   vertical descent distance
V_CRUISE         = 200 / 3.6  # [m/s] cruise speed
H_CRUISE         = 3810.0     # [m]   cruise altitude (consistent with RHO_CRUISE / T_CR below)
H_TRANSITION     = 0.0        # [m]   transition altitude                       PLACEHOLDER
RHO_CRUISE       = 0.835679   # [kg/m^3] ISA density at H_CRUISE
T_CR_ISA         = 263.385    # [K]   ISA temperature at H_CRUISE
T_SL_ISA         = 288.15     # [K]   ISA sea-level temperature
M_CR             = 0.17       # [-]   cruise Mach number
T_CRUISE         = 2194.7     # [s]   cruise segment
T_TAKEOFF        = 5.0        # [s]   takeoff segment
T_CLIMB          = 1221.7     # [s]   climb segment
T_CLIMB_ACC      = 20.0       # [s]   climb acceleration segment
T_DESCENT        = 311.945    # [s]   descent segment
T_VERTICAL_CLIMB = 25.0       # [s]   vertical climb segment
T_LANDING        = 71.47      # [s]   landing segment
T_ELAPSED_VC     = 5.0        # [s]   vertical climb elapsed time
V_AVG_TO         = 2.0        # [m/s] average takeoff vertical speed
V_HOVER          = 3.0        # [m/s] hover climb speed
V_I               = 20.0       # [m/s] initial climbing speed
VS_0             = D_VERT_DESCENT / T_DESCENT # [m/s] 

# Material parameters

SIGMA_ALLOW_CFRP = 500e6   # [Pa]   UD CFRP compression allowable, B-basis (MIL-HDBK-17-1F)
RHO_CFRP         = 1550.0  # [kg/m^3] CFRP density
E_CFRP           = 100 * 10 ** 9 # [Pa] CFRP young modulus (TO BE UPDATED)

SIGMA_ALLOW_AL = 260e6   # [Pa]   2024-T3 compression allowable (MMPDS-01)
RHO_AL         = 2700.0  # [kg/m^3] aluminium alloy density
E_AL           = 73.1 * 10 ** 9   # [Pa] aluminium young modulus

# Aerodynamic & propulsive parameters

ETA_CRUISE           = 0.95   # [-]   cruise total efficiency
ETA_POWERTRAIN_HOVER = 0.95   # [-]   hover powertrain efficiency
ETA_CLIMB            = 0.90   # [-]   climb powertrain efficiency
FM                   = 0.73   # [-]   rotor figure of merit
POWER_SAFETY_FACTOR  = 1.4    # [-]   power safety factor
PM                   = 0.5    # [-]   power margin
CL_MAX_OPERATIONAL   = 2.0    # [-]   operational upper lift coefficient limit 
CL_PLOT_MIN          = -0.5   # [-]   lower bound for drag-polar plotting
CL_PLOT_MAX          = 2.20   # [-]   upper bound for drag-polar plotting
DRAG_POLAR_N_POINTS  = 100    # [-]   number of points used for drag-polar plots
#   Preliminary values
LD_CRUISE            = 14.7   # [-]   box-wing cruise L/D (preliminary)
CD0                  = 0.0205 # [-]   zero-lift drag coefficient (preliminary)
OSWALD_EFFICIENCY    = 1.34   # [-]   Oswald efficiency factor (updated 11-06-2026)

# Wing structural parameters

NUMBER_OF_WINGS     = 2       # [-]     e.g. 1 for conventional, 2 for Prandtl/box-wing
AREA_SPLIT          = 0.5     # [-]     fraction of total area assigned to one wing
WING_SPAN           = 13.0    # [m]     span from the footprint constraint
WING_LOADING_N      = 760.0   # [N/m^2] selected design-point wing loading from matching diagram
TAPER_W             = 0.45    # [-]     wing chord taper ratio (c_tip / c_root)
TIP_TO_CHORD_W      = 0.17    # [-]     wing thickness-to-chord ratio
LE_SWEEP_W          = 0       # [rad]   wing leading edge sweep angle
DIHEDRAL            = 0       # [rad]   wing digedral angle
TWIST               = 0.05236 # [rad]   wing twist angle (3 deg) NOT FINAL
#    Class I parameters (OUTDATED - CLASS II AVAILABLE)
S_W         = 30.0            # [m^2]   class I total wing reference area 
AR_W        = 5.63            # [-]     class I wing aspect ratio 
#    Wing material: CFRP (standard for modern eVTOL primary structure)
T_SKIN_MIN_CFRP  = 1.0e-3  # [m]    minimum CFRP skin - 8 plies ?- 0.125 mm prepreg (MIL-HDBK-17-3F)

# V-tail structural parameters

V_ANGLE        = 25.0    # [deg]  V-tail dihedral from horizontal
TAPER_TAIL     = 0.40    # [-]    chord taper ratio (c_tip / c_root)
TIP_TO_CHORD   = 0.10    # [-]    thickness-to-chord ratio
F_REAR_WING    = 0.50    # [-]    rear Prandtl-wing lift fraction
N_W            = 3.5     # [-]    design limit load factor
T_SKIN_MIN_AL  = 2.0e-3  # [m]    minimum skin gauge (Niu 1988)
STRUCT_SF      = 1.5     # [-]    ultimate safety factor (FAR/CS 25.303)
C_N_TAIL_MAX   = 1.2     # [-]    peak normal force coefficient at max deflection
V_DIVE_FACTOR  = 1.25    # [-]    V_dive / V_cruise (FAR/CS 25.335 lower bound)
S_TAIL      = 3.25   # [m^2]  V-tail total panel area (both panels)
AR_T        = 1.23   # [-]   V-tail aspect ratio

# Propulsion geometry

N_PROP   = 6      # [-]  number of rotors
N_MOTOR  = 6      # [-]  number of motors
N_BLADES = 8      # [-]  blades per rotor
D_PROP   = 1.9    # [m]  rotor diameter
A_DISK   = 2.84   # [m^2] rotor disk area per rotor

# Battery -- Amprius SA504 cell-count basis (see battery_cells.py)
# The 20% energy reserve is the sole margin
# allowance is superseded by the reserve + whole-string layout round-up.

E_CELL_WH    = 37.57    # [Wh]  energy per cell (SA504 datasheet)
M_CELL_KG    = 0.0973   # [kg]  cell mass (SA504 datasheet, 97.3 g)
CELL_TO_PACK = 0.72     # [-]   pack mass fraction that is cells
E_AUX_KWH    = 2.79     # [kWh] auxiliary energy (aux_loads.py build-up)
RESERVE_FRAC = 0.20     # [-]   20% energy reserve on (mission + aux)
S_SERIES     = 235      # [-]   cells in series for the 800 V bus
E_CELL_DENS_WH_KG = E_CELL_WH / M_CELL_KG   # = 386 Wh/kg (derived, do not edit)

# Airframe geometry

L_FUS       = 10.0    # [m]   fuselage length
FUSE_WIDTH  = 2.0    # [m]   fuselage width (front view; placeholder)
PER_FUS_MAX = 12.0   # [m]   fuselage maximum perimeter
N_PAX       = 4      # [-]   passenger count
FAT_CYLINDER_SECTION = 0.4 #this is for mmoi calculations on what we assume is the cylinder (rest of fus is neglected)

# MMOI component layout (class_II_sizing/MMOI.py)
#
# Datum: nose tip. x positive aft, y positive starboard, z positive up, origin on the
# fuselage centreline. Seeds taken from the Vehicle Dynamics parameter sheet
# (Part 2 below) where available. All coordinates are component-CG
# locations pending a real layout drawing.

# -- Fuselage equivalent cylinder --
D_FUS    = FUSE_WIDTH     # [m]   equivalent-cylinder diameter (circular section assumed)
X_CG_FUS = 0.45 * L_FUS   # [m]   shell CG slightly fwd of mid-length (light tailcone)  PLACEHOLDER
Z_CG_FUS = 0.0            # [m]   shell CG on the centreline

# -- Wings (Prandtl pair) --
X_WING_F     = 1.5                       # [m]  front-wing CG ~ x_LEMAC (vd: 1.0) + 0.4*MAC  PLACEHOLDER
Z_WING_F     = -0.5                      # [m]  low-mounted front wing (vd z_w_fw)
WING_STAGGER = 5.0                       # [m]  longitudinal distance front -> rear wing (vd stagger)
H_GAP_WINGS  = 2.1                       # [m]  vertical gap between wing planes (vd gap); tip-plate height
X_WING_R     = X_WING_F + WING_STAGGER   # [m]  rear-wing CG station (derived)
Z_WING_R     = Z_WING_F + H_GAP_WINGS    # [m]  high rear wing (derived)
WINGLET_MASS_FRAC = 0.10                 # [-]  wing-mass fraction carved out for the two
                                         #      vertical tip joiners                      PLACEHOLDER

# -- V-tail --
X_TAIL      = 6.3   # [m]  panel-pair CG station (vd x_vert_tail = 8.5 scaled x 7/10)   PLACEHOLDER
Z_TAIL_ROOT = 0.5   # [m]  panel root height above centreline                           PLACEHOLDER

# -- Rotor / motor stations (4 on the front wing, 2 on the rear wing, symmetric) --
ETA_ROTOR_FW_IN  = 0.30  # [-]  inboard front rotor, fraction of semi-span (tip clears fuselage)  PLACEHOLDER
ETA_ROTOR_FW_OUT = 0.70  # [-]  outboard front rotor (2.6 m spacing > D_PROP, no disc overlap)    PLACEHOLDER
ETA_ROTOR_RW     = 0.50  # [-]  rear rotor, one per side                                          PLACEHOLDER
X_ROTOR_OFFSET   = 0.8   # [m]  pod CG ahead of the wing CG station (pylon + spinner)             PLACEHOLDER
X_ROTOR_FW = X_WING_F - X_ROTOR_OFFSET  # [m]  front-rotor station (derived)
X_ROTOR_RW = X_WING_R - X_ROTOR_OFFSET  # [m]  rear-rotor station (derived)
Z_ROTOR_FW = Z_WING_F                   # [m]  rotors carried at front-wing height
Z_ROTOR_RW = Z_WING_R                   # [m]  rotors carried at rear-wing height

# -- Optimal cruise CG (shared with the VD sheet, MassProperties.x_cg_opt) --
X_CG_OPT = 2.8  # [m]  optimal CG location during cruise

# -- Battery (underfloor box) --
L_BATT = 3.0    # [m]  box length ~ cabin floor length                                  PLACEHOLDER
W_BATT = 1.2    # [m]  box width between cabin floor beams                              PLACEHOLDER
H_BATT = 0.25   # [m]  underfloor bay depth                                             PLACEHOLDER
X_BATT = X_CG_OPT  # [m]  box mid-length at the cruise-optimal CG so the
                   #      heaviest item is CG-neutral
Z_BATT = -0.7   # [m]  below the cabin floor (floor ~ -0.5 m for the 2.0 m section)

# -- Payload (pax + luggage cabin box) --
X_PAYLOAD     = X_CG_OPT  # [m]  pax + luggage centred on the target CG
Z_PAYLOAD     = -0.2   # [m]  seated-occupant CG slightly below centreline
L_PAYLOAD_BOX = 2.0    # [m]  two seat rows                                             PLACEHOLDER
W_PAYLOAD_BOX = 1.4    # [m]  cabin width                                               PLACEHOLDER
H_PAYLOAD_BOX = 1.2    # [m]  seated height                                             PLACEHOLDER

# -- Landing gear (skid rails) --
X_GEAR      = 0.5 * L_FUS                  # [m]  rail mid-length under the cabin
Z_GEAR      = -(FUSE_WIDTH / 2 + 0.30)     # [m]  belly radius + 0.30 m static clearance
# Y_GEAR_RAIL (half-track) is derived from L_ARM further down, after the landing-gear
# trade-study constants that define it.

# -- Misc systems (20% MTOW: wiring, ECS, avionics, furnishings) --
X_MISC = X_CG_FUS   # [m]  smeared through the fuselage -> fuselage CG
Z_MISC = 0.0        # [m]  on the centreline


# Landing-gear drop trade study (CS-27.725 limit + 27.727 reserve)
#
# Skid gear modelled as 2 ground rails joined by 2 transverse cross-members; each
# member has 2 knees -> 4 "legs"/hinges across the 2 skids. Every architecture is an
# energy absorber sized for the two drops (stay ELASTIC at the limit, survive the
# reserve), and all capacities/masses are WHOLE-GEAR totals (per-member values scaled
# by the *_COUNT constants below). Used by mass_components.landing_gear_mass and the
# offline study class_II_sizing/trade_off_landing_gear.py.

# -- Whole-aircraft drop inputs (MTOW is supplied by the caller, not a constant) --
H_L      = 0.25      # [m]  limit drop height (>= 0.20 floor)                  PLACEHOLDER
D_EST    = 0.15      # [m]  estimated impact deflection (gear stroke); used    PLACEHOLDER
                     #       ONLY to size the effective drop mass M_EFF
LIFT     = 0.0       # [-]  rotor/lift credit ratio L (0 = conservative, no credit)
N_LIMIT  = 20.0      # [g]  max allowed peak deceleration                      PLACEHOLDER
ENVELOPE = 0.40      # [m]  available vertical stroke envelope                 PLACEHOLDER

# -- Materials: E [Pa], sy [Pa], rho [kg/m^3], eu [-] ductility, Gc [J/m^2] --
#    (MMPDS / MIL-HDBK-5 for metals; laminate datasheet for CFRP)
AL    = {"E": 71.7e9, "sy": 503e6,  "rho": 2810, "eu": 0.11,  "Gc": 0}
STEEL = {"E": 200e9,  "sy": 1200e6, "rho": 7850, "eu": 0.06,  "Gc": 0}
CFRP  = {"E": 70e9,   "sy": 600e6,  "rho": 1600, "eu": 0.015, "Gc": 1500}

# -- Gear architecture: how many parallel units form the full gear --
N_SKID          = 2      # [-]   longitudinal ground rails
# -- Skid-rail tube geometry (hollow aluminium round tube, density RHO_AL) --
D_SKID_RAIL     = 0.080  # [m]   rail outer diameter
T_SKID_RAIL     = 0.004  # [m]   rail wall thickness
L_SKID_RAIL     = 0.65 * L_FUS  # [m]  rail length ~ ground-contact footprint
CROSSTUBE_COUNT = 2      # [-]   transverse cross-tubes (front + rear)
HINGES_PER_TUBE = 2      # [-]   knees per cross-tube  -> 2*2 = 4 hinges total
LEAF_COUNT      = 2      # [-]   transverse leaf springs
HINGES_PER_LEAF = 2      # [-]
COMPOSITE_COUNT = 2      # [-]
CRUSH_COUNT     = 4      # [-]   one crush unit per leg
ELASTO_COUNT    = 4      # [-]   one elastomeric mount per leg

# -- Structural-model constants --
BC_FACTOR         = 3.0    # [-]   k = BC*EI/L^3 (3 = tip-loaded cantilever)
CROSS_SPAN        = 0.60   # [m]   skid track width under fuselage             PLACEHOLDER
FOOTPRINT_MAX_SPAN = 6.0   # [m]   max lateral gear track (spanwise footprint cap)
TUBE_HINGE_LEN    = 1.0    # [xD]  plastic-hinge length, tube knee
LEAF_HINGE_LEN    = 1.5    # [xt]  plastic-hinge length, leaf
LEAF_DEV_FACTOR   = 2.0    # [-]   developed leaf length = factor * L
COMP_CRUSH_FRAC   = 0.5    # [xL]  post-failure crush travel, composite        PLACEHOLDER
COMP_DELAM_FACTOR = 2.0    # [-]   delaminated length = factor * L
CRUSH_LEAF_B      = 0.08   # [m]   width of the elastic leaf in the hybrid
CRUSH_LEAF_L      = 0.45   # [m]   length of that leaf
HONEYCOMB_STRESS  = 2.5e6  # [Pa]  honeycomb crush plateau stress
HONEYCOMB_DENSITY = 80.0   # [kg/m^3] honeycomb core density
CRUSH_STROKE_EFF  = 0.75   # [-]   usable crush fraction before densification
ELASTO_FIXED_MASS = 1.2    # [kg]  mount housing/bracket fixed mass
ELASTO_MASS_PER_N = 1.0e-4 # [kg/N] mount mass vs peak load
# -- Discrete-mount load-path frame (elastomeric): CROSSTUBE_COUNT cross-tubes + ELASTO_COUNT
#    arms carrying the mounts down to the skids. Hollow aluminium tubes (density RHO_AL). --
D_FRAME_TUBE      = 0.060  # [m]   cross-tube / arm outer diameter
T_FRAME_TUBE      = 0.003  # [m]   cross-tube / arm wall thickness
L_ARM             = 0.45   # [m]   arm length, mount/hinge down to skid

# -- MMOI layout, gear half-track (see "MMOI component layout" section above) --
Y_GEAR_RAIL = (FUSE_WIDTH + 2 * L_ARM) / 2  # [m]  matches the whole_gear track FUSE_WIDTH + 2*L

# -- Trade-off scoring: qualitative scores 0..1 by engineering judgement (PLACEHOLDERS) --
QUAL = {
    "A cross-tube":       dict(tunable=0.3, cert_risk=0.1, cost=0.1),
    "B metal leaf":       dict(tunable=0.5, cert_risk=0.2, cost=0.2),
    "B' composite leaf":  dict(tunable=0.5, cert_risk=0.8, cost=0.5),
    "F crushable hybrid": dict(tunable=0.9, cert_risk=0.5, cost=0.5),
    "E elastomeric":      dict(tunable=0.4, cert_risk=0.4, cost=0.2),
}
# criterion -> (direction, base weight, relative uncertainty for sensitivity)
#   direction "min_ratio" (mass only) is magnitude-aware: score = lightest/value, so a 6x
#   heavier design scores ~0.16 (not just "worst in the feasible set" as plain min-max would).
#   Mass is the dominant weight so a much heavier gear cannot win on the soft criteria.
WEIGHTS = {
    "SEA":       ("max",       0.18, 0.10),
    "mass":      ("min_ratio", 0.45, 0.05),
    "npk":       ("min",       0.11, 0.10),
    "reusable":  ("max",       0.07, 0.00),
    "tunable":   ("max",       0.06, 0.20),
    "cert_risk": ("min",       0.09, 0.20),
    "cost":      ("min",       0.04, 0.20),
}

# -- Per-architecture SLSQP search config (x0 / bounds / varnames / material / scale) --
#    Keys match the concept-function map in mass_components and the QUAL table above.
#
#    The `bounds`/`x0` below are calibrated at GEAR_BOUNDS_REF_MASS. `scale` gives a
#    per-variable exponent so mass_components._scaled_gear_bounds grows each variable's
#    UPPER bound (and seed) by (m_eff / GEAR_BOUNDS_REF_MASS) ** scale_i, keeping the
#    study valid at any MTOW. Scale the ENERGY-bearing, stroke-neutral dimensions ~linearly
#    (leaf width, honeycomb area, elastomer stiffness) so elastic capacity tracks E_L ~ m_eff,
#    while leaving stroke / thickness / dimensionless variables fixed (exponent 0) so the
#    stroke envelope and load-factor cap are not blown. The cross-tube grows only modestly
#    and stays elastic-infeasible at high sink speeds by design (a bending tube is not an
#    elastic spring).
GEAR_OPT_SEED = 0          # [-]  seed for the SLSQP random restarts (reproducible sizing)
GEAR_OPT_RESTARTS = 48     # [-]  random multi-starts per architecture. Marginal designs
                           #       (e.g. a stiff metal leaf near the load-factor cap) sit on
                           #       a small feasible island; too few restarts make feasibility
                           #       and the optimum flicker, so this is set generously.
GEAR_BOUNDS_REF_MASS = 700.0  # [kg] effective mass at which the bounds/x0 below are tuned
GEAR_OPT_BOUNDS = {
    "A cross-tube":      dict(x0=[0.060, 0.004, 0.45],
                              bounds=[(0.03, 0.14), (0.002, 0.014), (0.30, 0.70)],
                              varnames=["D", "t", "L"], material=AL,
                              scale=[0.5, 0.0, 0.0]),
    "B metal leaf":      dict(x0=[0.080, 0.010, 0.45],
                              bounds=[(0.04, 0.18), (0.004, 0.030), (0.30, 0.70)],
                              varnames=["b", "t", "L"], material=STEEL,
                              scale=[1.0, 0.0, 0.0]),
    "B' composite leaf": dict(x0=[0.090, 0.012, 0.45],
                              bounds=[(0.04, 0.20), (0.004, 0.035), (0.30, 0.70)],
                              varnames=["b", "t", "L"], material=CFRP,
                              scale=[1.0, 0.0, 0.0]),
    "F crushable hybrid": dict(x0=[0.008, 0.003, 0.10],
                               bounds=[(0.003, 0.020), (0.0005, 0.012), (0.04, 0.25)],
                               varnames=["tleaf", "Ac", "sc"], material=STEEL,
                               scale=[0.5, 1.0, 0.5]),
    "E elastomeric":     dict(x0=[6e5, 0.08, 0.4],
                              bounds=[(1e5, 3e6), (0.03, 0.20), (0.3, 0.6)],
                              varnames=["kb", "dm", "loss"], material=AL,
                              scale=[1.0, 0.0, 0.0]),
}



# Miscellaneous design constants

gust_speed               = 0      # [m/s]  design gust speed
cabin_noise_req          = 0      # [dB]   cabin noise requirement
propulsion_type          = 0      # [-]    propulsion type flag
contingency_energy_reserve = 0    # [kWh]  energy reserve
vehicle_lifetime         = 0      # [years]
aircraft_person_proximity  = 0    # [m]
aircraft_building_proximity = 0   # [m]
rho_propeller_hub        = 2700.0 # [kg/m^3] hub material density (aluminium)
rho_propeller_blade      = 1550.0 # [kg/m^3] blade material density (CFRP)


# ============================================================================
# Part 2 - VEHICLE DYNAMICS PARAMETER SHEET
#
# Independent variables that describe the aircraft for the Vehicle Dynamics
# model (formerly final_characteristics/vehicle_dynamics/vd_parameters.py).
# Quantities shared with Part 1 default to the Part 1 constants so each
# parameter has a single source of truth. Angles in this sheet are in DEGREES
# unless noted; Part 1 wing angles are stored in radians and converted here.
# ============================================================================


@dataclass
class Mission:
    """
    Mission-level independent requirements relevant to the Vehicle Dynamics model.
    """

    transition_speed:    float = None        # [m/s]
    cruise_speed:        float = V_CRUISE    # [m/s]
    M_cr:                float = M_CR        # [-]
    wind_gust_speed:     float = None        # [m/s]

    transition_altitude: float = H_TRANSITION  # [m]
    cruise_altitude:     float = H_CRUISE      # [m]

    rho_cr:              float = RHO_CRUISE  # [kg/(m^3)]
    rho_SL:              float = RHO_ORIGIN  # [kg/(m^3)]

    T_cr:                float = T_CR_ISA    # [K]
    T_SL:                float = T_SL_ISA    # [K]

    minimum_range:       float = RANGE_M     # [m]


@dataclass
class WingGeometry:
    """
    Geometric layout of the horizontal lifting surfaces.

    For the Prandtl-plane / box-wing configuration, these parameters describe
    the position, size, and orientation of the front and aft horizontal wings.
    """

    design_point:         float = WING_LOADING_N  # [N/m^2]

    S_fw:                 float = 12.91  # [m^2]
    S_aw:                 float = 12.91  # [m^2]
    S_e_fw:               float = 9.685  # [m^2]
    S_e_aw:               float = 9.685  # [m^2]
    S_tot:                float = 25.82  # [m^2]

    b_fw:                 float = WING_SPAN  # [m]
    b_aw:                 float = WING_SPAN  # [m]

    A_fw:                 float = b_fw**2/S_fw  # [-]
    A_aw:                 float = b_aw**2/S_aw  # [-]

    gap:                  float = H_GAP_WINGS   # [m]
    stagger:              float = WING_STAGGER  # [m]

    MAC_fw:               float = 1.262  # [m]
    MAC_aw:               float = 1.262  # [m]

    taper_fw:             float = TAPER_W  # [-]
    taper_aw:             float = TAPER_W  # [-]

    chord_fw_root:        float = 1.66  # [m]
    chord_fw_tip:         float = 0.75  # [m]
    chord_aw_root:        float = 1.66  # [m]
    chord_aw_tip:         float = 0.75  # [m]

    LE_sweep_fw:          float = math.degrees(LE_SWEEP_W)  # [deg.]
    LE_sweep_aw:          float = math.degrees(LE_SWEEP_W)  # [deg.]

    dihedral_front_wing:  float = math.degrees(DIHEDRAL)  # [deg.]
    dihedral_aft_wing:    float = math.degrees(DIHEDRAL)  # [deg.]

    twist_fw:             float = math.degrees(TWIST)  # [deg.]
    twist_aw:             float = math.degrees(TWIST)  # [deg.]

    incidence_fw:         float = None  # [deg.]
    incidence_aw:         float = None  # [deg.]

    airfoil_fw:           str   = "NASA LANGLEY LS(1)-0417"  # [-]
    airfoil_aw:           str   = "NASA LANGLEY LS(1)-0417"  # [-]

    # ===== ADDED FOR DATCOM ==================================================
    # Reference quantities for non-dimensionalisation. EVERY aircraft-level derivative is referenced to these; they must match the EOM reference.
    S_ref:                float = None  # [m^2]
    b_ref:                float = WING_SPAN  # [m]
    MAC_ref:              float = 1.262  # [m]

    # Airfoil thickness and trailing-edge angle.
    # NOTE: the front/aft WING lift-curve slopes now use the aero department's
    # section slope (cl_alpha_fw / cl_alpha_aw) directly, so these are no longer
    # consumed by the lift methods. Retained as geometric descriptors only.
    t_c_fw:               float = TIP_TO_CHORD_W  # [-]
    t_c_aw:               float = TIP_TO_CHORD_W  # [-]
    te_angle_fw:          float = 18  # [deg.]
    te_angle_aw:          float = 18  # [deg.]

    # Wing vertical position relative to body centreline (z positive up,
    # matching the MMOI layout: front wing low at Z_WING_F, aft wing at
    # Z_WING_R = Z_WING_F + H_GAP_WINGS)
    z_w_fw:               float = Z_WING_F  # [m]
    z_w_aw:               float = Z_WING_R  # [m]

    x_LEMAC_fw:           float = 1 # [m] - very rough estimate


@dataclass
class TailGeometry:
    """
    Geometry and position of the vertical tail.

    NOTE: the VD model treats the tail as a single vertical fin; the V-tail
    structural constants in Part 1 (V_ANGLE, S_TAIL, AR_T, ...) describe the
    structural panel pair and are kept separate on purpose.
    """

    x_vert_tail:                float = 8.5  # [m]
    c_r_vert_tail:              float = 1.4  # [m]
    c_t_vert_tail:              float = 0.5  # [m]
    b_vert_tail:                float = 2.5  # [m]
    S_vert_tail:                float = ((c_r_vert_tail + c_t_vert_tail)*b_vert_tail)/2  # [m^2]
    AR_vert_tail:               float = b_vert_tail**2/S_vert_tail  # [-]
    LE_sweep_vert_tail:         float = 0.31  # [rad]
    airfoil_vert_tail:          str   = "NACA 0012"  # [-]

    taper_vert_tail:            float = 0.33  # [-]
    MAC_vert_tail:              float = 0.867  # [m]
    t_c_vert_tail:              float = 0.12  # [-]
    te_angle_vert_tail:         float = 14  # [deg.]
    z_vert_tail:                float = 1.0  # [m] vertical a.c. height (datum)


@dataclass
class WingletGeometry:
    """
    Geometry of the Prandtl-plane vertical joiners / winglets.

    These are not treated as conventional aft vertical tails. They are modelled
    as vertical side-force-producing panels at the wing tips, with their own
    effective aspect-ratio and local-flow corrections. Set enabled=True only
    after the geometry and chart/correction inputs have been supplied.
    """

    enabled:                    bool  = True  # keep existing model backward-compatible
    n_winglets:                 int   = 2      # usually left and right tip joiners

    S_winglet:                  float = 3.125  # [m^2] planform area of ONE winglet/joiner
    b_winglet:                  float = 2.5  # [m] vertical span/height of ONE winglet
    AR_winglet:                 float = b_winglet**2/S_winglet  # [-] aspect ratio of ONE winglet
    taper_winglet:              float = 1.0 # [-]
    LE_sweep_winglet:           float = 0.5  # [rad]

    x_ac_winglet:               float = 3.6  # [m] longitudinal aerodynamic-centre location
    z_ac_winglet:               float = 0.0  # [m] vertical aerodynamic-centre location

    airfoil_winglet:            str   = "NASA LANGLEY LS(1)-0417"  # [-]
    t_c_winglet:                float = 0.12  # [-]
    te_angle_winglet:           float = 10  # [deg.]


@dataclass
class FuselageGeometry:
    """
    Fuselage geometry relevant to aerodynamics and vehicle dynamics.
    """

    fuselage_length:   float = L_FUS       # [m]
    d_fw:              float = FUSE_WIDTH  # [m] - Fuselage is modelled as a tube for now
    d_aw:              float = 0     # [m]
    x_ac_fuselage:     float = None  # [m]

    side_area:           float = 20.0 + TailGeometry.S_vert_tail  # [m^2] projected side area S_Bs (Cn_beta)
    base_area:           float = 20.0  # [m^2] reference/base area S_B0 (CY_beta body)
    body_depth_at_wing:  float = FUSE_WIDTH  # [m]   d at the wing (sidewash); ~ diameter


@dataclass
class ControlSurfaceGeometry:
    """
    Control-surface geometry needed for the deflection (control) derivatives.
    Chord ratios feed the section-effectiveness charts (Sec 6.1.1.1); the span
    factors feed K_b / strip integration.
    """
    elevator_cf_c:       float = 0.3   # [-] flap-chord / wing-chord ratio
    elevator_Kb:         float = 0.46   # [-] flap-span factor (Fig 6.1.4.1)
    elevator_on_surface: str   = "aw"   # which wing carries the elevator ('fw' or 'aw')

    aileron_cf_c:        float = 0.3   # [-]
    aileron_eta_inner:   float = 0.35   # [-] inboard span station
    aileron_eta_outer:   float = 0.77   # [-] outboard span station

    rudder_cf_c:         float = 0.2   # [-]


@dataclass
class MassProperties:
    """
    Aircraft mass and inertia properties.

    mtow here is a seed value for standalone VD runs; the converged class-II
    design state (class_II_sizing/mtow_sizing.py) is authoritative.
    """

    mtow:         float = 2000.0     # [kg]
    oew:          float = None       # [kg]
    payload_mass: float = M_PAYLOAD  # [kg]

    x_cg_min:     float = None      # [m]
    x_cg_max:     float = None      # [m]
    x_cg_opt:     float = X_CG_OPT  # [m] - Optimal CG location during cruise
    z_cg:         float = None      # [m] - vertical CG (datum), used by moment arms

    I_xx:         float = None  # [kg m^2]
    I_yy:         float = None  # [kg m^2]
    I_zz:         float = None  # [kg m^2]
    I_xz:         float = None  # [kg m^2]


@dataclass
class AerodynamicCoefficients:
    """
    Aerodynamic characteristics required by the Vehicle Dynamics model.
    """
    # Airfoil curve
    cl_alpha_fw:                        float = 0.107  # [1/deg.]
    cl_alpha_aw:                        float = 0.107  # [1/deg.]
    cl_alpha_vert_tail:                 float = None  # [1/deg.]
    cl_alpha_winglet:                   float = None  # [1/deg.]

    # Lift curve
    CL_alpha_fw:                        float = 5.02484  # [1/rad.]
    CL_alpha_aw:                        float = 5.02484  # [1/rad.]
    CL_alpha_vert_tail:                 float = None  # [1/rad.]
    CL_alpha_winglet:                   float = None  # [1/rad.]

    # Maximum lift coefficients
    CL_max_clean:                       float = None  # [-]
    CL_max_hld:                         float = None  # [-]

    # Zero-lift drag coefficients
    CD0_fw_clean:                       float = None  # [-]
    CD0_aw_clean:                       float = None  # [-]

    CD0_fw_HLD:                         float = None  # [-]
    CD0_aw_HLD:                         float = None  # [-]

    CD0_vert_tail:                      float = None  # [-]
    CD0_winglet:                        float = None  # [-]
    CD0_fuselage:                       float = None  # [-]

    # Oswald efficiency factors
    e_hor_wings:                         float = OSWALD_EFFICIENCY  # [-] (DOES THE VALIDITY CHANGE WHEN HLDs ARE DEPLOYED?)
    e_vert_tail:                         float = None  # [-]
    e_winglet:                           float = None  # [-]

    # Aerodynamic moments
    C_M_ac_fw:                           float = -0.118  # [-]
    C_M_ac_aw:                           float = -0.118  # [-]
    C_M_ac_fuselage:                     float = None  # [-]

    # Downwash gradients
    downwash_gradient_fw_to_aw:         float = None  # [-]
    sidewash_gradient_fuselage_to_tail: float = None  # [-]

    # Dynamic pressure ratios
    dyn_pres_ratio_fw_to_aw:            float = 0.9  # [-]
    dyn_pres_ratio_fuselage_to_tail:    float = 0.95  # [-]

    # Flow speed ratios
    flow_speed_ratio_fw_to_aw:          float = None  # [-]
    flow_speed_ratio_fuselage_to_tail:  float = None  # [-]

    # Interference coefficients
    I_v:                                float = -1.75 # [-] - Vortex interference factor

    # Long. positions of aerodynamic centres
    x_ac_fw_cruise:                     float = 0.313  # [m] as seen from the LEMAC of the front wing
    x_ac_aw_cruise:                     float = 0.313  # [m] as seen from the LEMAC of the aft wing
    x_ac_fw_approach:                   float = None  # [m]
    x_ac_aw_approach:                   float = None  # [m]

    # Vert. positions of aerodynamic centres
    z_ac_fw_cruise:                     float = None  # [m]
    z_ac_aw_cruise:                     float = None  # [m]
    z_ac_fw_approach:                   float = None  # [m]
    z_ac_aw_approach:                   float = None  # [m]


@dataclass
class StabilityDerivatives:
    """
    Stability derivatives used in the Vehicle Dynamics model.
    """

    # AoA
    C_L_alpha: float = None  # [1/rad]
    C_M_alpha: float = None  # [1/rad]

    # AoA-rate (lag of downwash)
    C_L_alpha_dot: float = None  # [1/rad]
    C_M_alpha_dot: float = None  # [1/rad]

    # Pitch rate
    C_L_q:     float = None  # [1/rad]
    C_M_q:     float = None  # [1/rad]

    # Sideslip
    C_Y_beta:  float = None  # [1/rad]
    C_L_beta:  float = None  # [1/rad]
    C_N_beta:  float = None  # [1/rad]

    # Sideslip rate
    C_Y_beta_dot: float = None  # [1/rad]
    C_N_beta_dot: float = None  # [1/rad]

    # Roll rate
    C_Y_p:     float = None  # [1/rad]
    C_L_p:     float = None  # [1/rad]
    C_N_p:     float = None  # [1/rad]

    # Yaw rate
    C_Y_r:     float = None  # [1/rad]
    C_L_r:     float = None  # [1/rad]
    C_N_r:     float = None  # [1/rad]

    # X/Z force bridge (symmetric EOM)
    C_X_0:        float = None  # [-]
    C_Z_0:        float = None  # [-]
    C_X_u:        float = None  # [-]
    C_Z_u:        float = None  # [-]
    C_X_alpha:    float = None  # [1/rad]
    C_Z_alpha:    float = None  # [1/rad]
    C_Z_alpha_dot: float = None  # [1/rad]
    C_Z_q:        float = None  # [1/rad]


@dataclass
class ControlDerivatives:
    """
    Control-surface deflection derivatives.

    These represent control effectiveness in terms of force or moment
    coefficient change per control-surface deflection.
    """

    C_L_delta_e: float = None  # [1/rad]
    C_M_delta_e: float = None  # [1/rad]
    C_Z_delta_e: float = None  # [1/rad]
    C_X_delta_e: float = None  # [1/rad]

    C_Y_delta_r: float = None  # [1/rad]
    C_N_delta_r: float = None  # [1/rad]
    C_L_delta_r: float = None  # [1/rad]

    C_L_delta_a: float = None  # [1/rad]
    C_N_delta_a: float = None  # [1/rad]


@dataclass
class Propulsion:
    """
    Propulsion parameters relevant to Vehicle Dynamics.

    Power-required quantities should not be placed here, because they are
    dependent outputs of the aircraft performance model.
    """

    P_max_SL:               float = None  # [W]
    propulsive_efficiency:  float = None  # [-]

    n_engines:              int   = N_PROP  # [-] - Number of engines
    max_thrust_per_engine:  float = None  # [N]

    # Engine locations in CRUISE CONFIGURATION
    x_cr_1:               float = None  # [m]
    y_cr_1:               float = None  # [m]
    z_cr_1:               float = None  # [m]

    x_cr_2:               float = None  # [m]
    y_cr_2:               float = None  # [m]
    z_cr_2:               float = None  # [m]

    x_cr_3:               float = None  # [m]
    y_cr_3:               float = None  # [m]
    z_cr_3:               float = None  # [m]

    x_cr_4:               float = None  # [m]
    y_cr_4:               float = None  # [m]
    z_cr_4:               float = None  # [m]

    x_cr_5:               float = None  # [m]
    y_cr_5:               float = None  # [m]
    z_cr_5:               float = None  # [m]

    x_cr_6:               float = None  # [m]
    y_cr_6:               float = None  # [m]
    z_cr_6:               float = None  # [m]

    # Engine locations in VTOL CONFIGURATION
    x_vtol_1:               float = None  # [m]
    y_vtol_1:               float = None  # [m]
    z_vtol_1:               float = None  # [m]

    x_vtol_2:               float = None  # [m]
    y_vtol_2:               float = None  # [m]
    z_vtol_2:               float = None  # [m]

    x_vtol_3:               float = None  # [m]
    y_vtol_3:               float = None  # [m]
    z_vtol_3:               float = None  # [m]

    x_vtol_4:               float = None  # [m]
    y_vtol_4:               float = None  # [m]
    z_vtol_4:               float = None  # [m]

    x_vtol_5:               float = None  # [m]
    y_vtol_5:               float = None  # [m]
    z_vtol_5:               float = None  # [m]

    x_vtol_6:               float = None  # [m]
    y_vtol_6:               float = None  # [m]
    z_vtol_6:               float = None  # [m]

    # Mass
    propulsion_system_mass: float = None  # [kg] - MASS PER ENGINE


@dataclass
class StructuralLimits:
    """
    Structural limits relevant to Vehicle Dynamics.
    """
    ultimate_load_factor: float = None  # [-]


@dataclass
class AircraftParameters:
    """
    Complete independent parameter sheet for the Vehicle Dynamics aircraft model.
    """

    mission:           Mission                   = field(default_factory=Mission)
    wing_geometry:     WingGeometry              = field(default_factory=WingGeometry)
    tail_geometry:     TailGeometry              = field(default_factory=TailGeometry)
    winglet_geometry:  WingletGeometry           = field(default_factory=WingletGeometry)
    fuselage_geometry: FuselageGeometry          = field(default_factory=FuselageGeometry)
    control_surfaces:  ControlSurfaceGeometry    = field(default_factory=ControlSurfaceGeometry)
    mass:              MassProperties            = field(default_factory=MassProperties)
    aerodynamics:      AerodynamicCoefficients   = field(default_factory=AerodynamicCoefficients)
    stability:         StabilityDerivatives      = field(default_factory=StabilityDerivatives)
    controls:          ControlDerivatives        = field(default_factory=ControlDerivatives)
    propulsion:        Propulsion                = field(default_factory=Propulsion)
    structures:        StructuralLimits          = field(default_factory=StructuralLimits)



