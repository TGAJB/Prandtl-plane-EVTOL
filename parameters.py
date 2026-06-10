# Physical constants

G          = 9.81          # [m/s^2]  gravitational acceleration
RHO_ORIGIN = 1.225         # [kg/m^3] ISA sea-level air density

# Mission parameters

M_PAYLOAD        = 400.0      # [kg]  fixed payload (4 pax + luggage)
W_CREW           = 0.0        # [kg]  0 for autonomous; 85 if piloted
RANGE_M          = 200000.0   # [m]   design range
D_VERT_DESCENT   = 1676.0     # [m]   vertical descent distance
V_CRUISE         = 200 / 3.6  # [m/s] cruise speed
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
OSWALD_EFFICIENCY    = 1.26   # [-]   Oswald efficiency factor (preliminary)

# Wing structural parameters

NUMBER_OF_WINGS     = 2       # [-]     e.g. 1 for conventional, 2 for Prandtl/box-wing
AREA_SPLIT          = 0.5     # [-]     fraction of total area assigned to one wing
WING_SPAN           = 13      # [-]     span from the footprint constraint
WING_LOADING_N      = 760.0   # [N/m^2] selected design-point wing loading from matching diagram
TAPER_W             = 0.45    # [-]     wing chord taper ratio (c_tip / c_root)
TIP_TO_CHORD_W      = 0.12    # [-]     wing thickness-to-chord ratio
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
N_BLADES = 5      # [-]  blades per rotor
D_PROP   = 1.9    # [m]  rotor diameter
A_DISK   = 2.84   # [m^2] rotor disk area per rotor

# Battery

E_PACK_WH_KG = 300.0   # [Wh/kg] pack-level specific energy
SOC_USABLE   = 0.80    # [-]     usable state-of-charge fraction
CONTINGENCY  = 1.05    # [-]     energy contingency factor

# Airframe geometry

L_FUS       = 7.0    # [m]   fuselage length
FUSE_WIDTH  = 2.0    # [m]   fuselage width (front view; placeholder)
PER_FUS_MAX = 12.0   # [m]   fuselage maximum perimeter
N_PAX       = 4      # [-]   passenger count


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

hcruise                  = 0      # [m]    cruise altitude
htransition              = 0      # [m]    transition altitude
gust_speed               = 0      # [m/s]  design gust speed
cabin_noise_req          = 0      # [dB]   cabin noise requirement
propulsion_type          = 0      # [-]    propulsion type flag
contingency_energy_reserve = 0    # [kWh]  energy reserve
vehicle_lifetime         = 0      # [years]
aircraft_person_proximity  = 0    # [m]
aircraft_building_proximity = 0   # [m]
rho_propeller_hub        = 2700.0 # [kg/m^3] hub material density (aluminium)
rho_propeller_blade      = 1550.0 # [kg/m^3] blade material density (CFRP)



