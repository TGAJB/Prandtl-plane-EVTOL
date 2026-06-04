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

LD_CRUISE            = 14.7   # [-]   box-wing cruise L/D
ETA_CRUISE           = 0.95   # [-]   cruise total efficiency
ETA_POWERTRAIN_HOVER = 0.95   # [-]   hover powertrain efficiency
ETA_CLIMB            = 0.90   # [-]   climb powertrain efficiency
FM                   = 0.73   # [-]   rotor figure of merit
POWER_SAFETY_FACTOR  = 1.4    # [-]   power safety factor
PM                   = 0.5    # [-]   power margin
CD0                  = 0.0205 # [-]   zero-lift drag coefficient
OSWALD_EFFICIENCY    = 1.26   # [-]   Oswald efficiency factor
CL_MAX_OPERATIONAL   = 2.0    # [-]   operational upper lift coefficient limit
CL_PLOT_MIN          = -0.5   # [-]   lower bound for drag-polar plotting
CL_PLOT_MAX          = 2.20   # [-]   upper bound for drag-polar plotting
DRAG_POLAR_N_POINTS  = 100    # [-]   number of points used for drag-polar plots

# Wing structural parameters

NUMBER_OF_WINGS     = 2       # [-]     e.g. 1 for conventional, 2 for Prandtl/box-wing
AREA_SPLIT         = 0.5      # [-]     fraction of total area assigned to one wing
WING_SPAN          = 13       # [-]     span from the footprint constraint
WING_LOADING_N      = 760.0   # [N/m^2] selected design-point wing loading from matching diagram
TAPER_W          = 0.40       # [-]     wing chord taper ratio (c_tip / c_root)
TIP_TO_CHORD_W   = 0.12       # [-]     wing thickness-to-chord ratio
#    Class I parameters (OUTDATED - CLASS II AVAILABLE)
S_W         = 30.0            # [m^2]   class I total wing reference area 
AR_W        = 5.63            # [-]     class I wing aspect ratio 
#    Wing material: CFRP (standard for modern eVTOL primary structure)
T_SKIN_MIN_CFRP  = 1.0e-3  # [m]    minimum CFRP skin - 8 plies ?- 0.125 mm prepreg (MIL-HDBK-17-3F)

# V-tail structural parameters

V_ANGLE        = 45.0    # [deg]  V-tail dihedral from horizontal
TAPER_TAIL     = 0.40    # [-]    chord taper ratio (c_tip / c_root)
TIP_TO_CHORD   = 0.10    # [-]    thickness-to-chord ratio
F_REAR_WING    = 0.50    # [-]    rear Prandtl-wing lift fraction
N_W            = 3.5     # [-]    design limit load factor
T_SKIN_MIN_AL  = 1.2e-3  # [m]    minimum skin gauge (Niu 1988)
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
PER_FUS_MAX = 12.0   # [m]   fuselage maximum perimeter
N_PAX       = 4      # [-]   passenger count


# Landing Gear design constants

ETA = 0.003       # legacy placeholder - superseded by ETA_LG_* below
D_O_SKID = 0.06   # [m]  placeholder outer diameter (sizing sweeps DO_SKID_MIN->MAX)
D_I_SKID = 0.054  # [m]  placeholder inner diameter (sizing uses T_WALL_SKID)
L_EFF    = 1.2    # [m]  effective cantilever length of each skid leg
                  #       CALIBRATION REQUIRED: with the simple cantilever model,
                  #       sigma ~ n*W*L/(Do^2*t), so the Do that keeps n<=N_LIMIT_LG
                  #       is too small to carry the root bending moment for L~1 m.
                  #       Set L_EFF to the actual bent-section arm (often 0.2-0.5 m
                  #       for helicopter skid cross-tubes), not the full half-track.
N_REACT  = 4      # [-]  total reaction points = 2 * n_cross (2 cross-tubes)

# CS-27/29 certification conditions
V_Z_LIMIT        = 2.44   # [m/s]  limit-condition sink rate
V_Z_RESERVE      = 3.7    # [m/s]  reserve-energy sink rate
# Efficiency: keep ETA_LG_RESERVE = 0.50 for CFRP (brittle; no plastic plateau).
# For ductile metals (Al, Ti) the reserve value may be raised to 0.60-0.80.
ETA_LG_LIMIT     = 0.50   # [-]  elastic absorption efficiency (limit condition)
ETA_LG_RESERVE   = 0.50   # [-]  absorption efficiency (reserve); override per material

# Load-factor and stroke constraints
# N_LIMIT_LG is the landing-gear load-factor limit, NOT the wing limit N_W = 3.5.
# Typical skid-gear values: 4-8 g depending on aircraft category and cert basis.
N_LIMIT_LG       = 7.0    # [-]  landing-gear ultimate load-factor limit (placeholder)
KAPPA_LG         = 1.0    # [-]  lift fraction at touchdown; 1.0 = full rotor lift (powered VTOL landing)
GROUND_CLEARANCE = 0.30   # [m]  maximum allowable stroke (30 cm; covers reserve case)

# Tube geometry
T_WALL_SKID  = 0.003  # [m]  skid tube wall thickness (placeholder - refine from sizing)
DO_SKID_MIN  = 0.03   # [m]  outer-diameter sweep lower bound
DO_SKID_MAX  = 0.30   # [m]  outer-diameter sweep upper bound

# Frame geometry
L_TRACK    = 2.0   # [m]  lateral track width (cross-tube chord)
L_SKID     = 3.0   # [m]  skid runner length (each side)
K_FITTINGS = 1.3   # [-]  mass knockup factor for fittings and attachments



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



