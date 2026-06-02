# ── Physical constants ─────────────────────────────────────────────────────────

G          = 9.81          # [m/s²]  gravitational acceleration
RHO_ORIGIN = 1.225         # [kg/m³] ISA sea-level air density

# ── Mission parameters ─────────────────────────────────────────────────────────

M_PAYLOAD        = 400.0      # [kg]  fixed payload (4 pax + luggage)
W_CREW           = 0.0        # [kg]  0 for autonomous; 85 if piloted
RANGE_M          = 200000.0   # [m]   design range
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
D_VERT_LANDING   = 1676       # [m] vertical distance to land
VI = 20.0                     # [m/s] initial velocity pre-climb acceleration
VS_0 = D_VERT_LANDING / T_DESCENT # [m/s] landing phase initial velocity

# ── Aerodynamic & propulsive parameters ───────────────────────────────────────

LD_CRUISE            = 14.7   # [-]   box-wing cruise L/D
ETA_CRUISE           = 0.95   # [-]   cruise total efficiency
ETA_POWERTRAIN_HOVER = 0.95   # [-]   hover powertrain efficiency
ETA_CLIMB            = 0.90   # [-]   climb powertrain efficiency
FM                   = 0.73   # [-]   rotor figure of merit
POWER_SAFETY_FACTOR  = 1.4    # [-]   power safety factor
PM                   = 0.5    # [-]   power margin

# ── Propulsion geometry ────────────────────────────────────────────────────────

N_PROP   = 6      # [-]  number of rotors
N_MOTOR  = 6      # [-]  number of motors
N_BLADES = 5      # [-]  blades per rotor
D_PROP   = 1.9    # [m]  rotor diameter
A_DISK   = 2.84   # [m²] rotor disk area per rotor

# ── Battery ────────────────────────────────────────────────────────────────────

E_PACK_WH_KG = 300.0   # [Wh/kg] pack-level specific energy
SOC_USABLE   = 0.80    # [-]     usable state-of-charge fraction
CONTINGENCY  = 1.05    # [-]     energy contingency factor

# ── Airframe geometry ──────────────────────────────────────────────────────────

L_FUS       = 7.0    # [m]   fuselage length
PER_FUS_MAX = 12.0   # [m]   fuselage maximum perimeter
N_PAX       = 4      # [-]   passenger count
S_W         = 30.0   # [m²]  total wing reference area
AR_W        = 5.63   # [-]   wing aspect ratio
S_TAIL      = 3.25   # [m²]  V-tail total panel area (both panels)
AR_T        = 1.23   # [-]   V-tail aspect ratio

# ── Wing structural parameters ────────────────────────────────────────────────

TAPER_W          = 0.40    # [-]    wing chord taper ratio (c_tip / c_root)
TIP_TO_CHORD_W   = 0.12    # [-]    wing thickness-to-chord ratio

# Wing material: CFRP (standard for modern eVTOL primary structure)
SIGMA_ALLOW_CFRP = 500e6   # [Pa]   UD CFRP compression allowable, B-basis (MIL-HDBK-17-1F)
RHO_CFRP         = 1550.0  # [kg/m³] CFRP density
T_SKIN_MIN_CFRP  = 1.0e-3  # [m]    minimum CFRP skin — 8 plies × 0.125 mm prepreg (MIL-HDBK-17-3F)

# ── V-tail structural parameters ───────────────────────────────────────────────

V_ANGLE        = 45.0    # [deg]  V-tail dihedral from horizontal
TAPER_TAIL     = 0.40    # [-]    chord taper ratio (c_tip / c_root)
TIP_TO_CHORD   = 0.10    # [-]    thickness-to-chord ratio
F_REAR_WING    = 0.50    # [-]    rear Prandtl-wing lift fraction
N_W            = 3.5     # [-]    design limit load factor
SIGMA_ALLOW_AL = 260e6   # [Pa]   2024-T3 compression allowable (MMPDS-01)
RHO_AL         = 2700.0  # [kg/m³] aluminium alloy density
T_SKIN_MIN_AL  = 1.2e-3  # [m]    minimum skin gauge (Niu 1988)
STRUCT_SF      = 1.5     # [-]    ultimate safety factor (FAR/CS 25.303)
C_N_TAIL_MAX   = 1.2     # [-]    peak normal force coefficient at max deflection
V_DIVE_FACTOR  = 1.25    # [-]    V_dive / V_cruise (FAR/CS 25.335 lower bound)

# ── Miscellaneous design constants ─────────────────────────────────────────────

hcruise                  = 0      # [m]    cruise altitude
htransition              = 0      # [m]    transition altitude
gust_speed               = 0      # [m/s]  design gust speed
cabin_noise_req          = 0      # [dB]   cabin noise requirement
propulsion_type          = 0      # [-]    propulsion type flag
contingency_energy_reserve = 0    # [kWh]  energy reserve
vehicle_lifetime         = 0      # [years]
aircraft_person_proximity  = 0    # [m]
aircraft_building_proximity = 0   # [m]
rho_propeller_hub        = 2700.0 # [kg/m³] hub material density (aluminium)
rho_propeller_blade      = 1550.0 # [kg/m³] blade material density (CFRP)