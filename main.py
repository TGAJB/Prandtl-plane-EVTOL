import numpy as np
import scipy as sc
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from ambiance import Atmosphere

M_PAYLOAD = 400 #[kg]

# BATTERY CONSTANTS (NEEDS TO BE UPDATED AND ORGANIZED LATER)

W_PAYLOAD = 400.0    # [kg]  Fixed mission payload (4 pax + luggage)
W_CREW    = 0.0      # [kg]  0 if autonomous; add 85 if piloted

RANGE_M           = 200000.0     # [m]   Cruise range
V_CRUISE          = 200 / 3.6     # [m/s] Cruise speed
T_CRUISE          = 2194.7        # [s]   Cruise segment duration
T_TAKEOFF         = 5.0          # [s]   Takeoff segment duration
T_CLIMB           = 1221.7          # [s]   climb segment duration
T_CLIMB_ACC       = 20.0          # [s]   climb acceleration segment duration
T_DESCENT         = 311.945          # [s]   Descent segment duration
T_VERTICAL_CLIMB  = 25.0         # [s]   Total vertical climb time
T_LANDING         = 71.47         # [s]   Total landing time

# DESIGN ASSUMPTIONS  

# Aerodynamic & propulsive
LD_CRUISE             = 14.7      # [-]   Box-wing cruise L/D
ETA_CRUISE            = 0.95      # [-]   Cruise total efficiency
ETA_POWERTRAIN_HOVER  = 0.95      # [-]   Hover powertrain efficiency
ETA_CLIMB             = 0.90      # [-]   Climb powertrain efficiency
FM                    = 0.73      # [-]   Rotor figure of merit
N_PROPS               = 6         # [-]   Number of rotors
PROP_DIAMETER         = 1.9       # [m]   Rotor diameter
T_W0                  = 1.29       # [-]   Initial thrust-to-weight ratio guess for hover power

# Battery
E_PACK_WH_KG  = 300.0     # [Wh/kg] Pack-level specific energy
SOC_USABLE    = 0.80      # [-]     Usable state-of-charge
CONTINGENCY   = 1.05      # [-]     Energy contingency

# CONSTANTS

L_D = 14.7
A_DISK = 2.84 #[m^2] per disk
PROP_EFF = 0.8
FM = 0.73
T_W = 1.29
G = 9.81 #[m/s^2]
V_CR = 200/3.6 #[m/s]
D_CR = 121929.0741 #[m]
BATT_EFF = 0.98
BATT_SE = 300 #[Wh/kg]
L_FUS = 7 #[m]
PER_FUS_MAX = 12 #[m]
N_PAX = 4
S_W = 11.2 #[m^2]
N_W = 3.5 # design load factor
AR_W = 5.63 # aspect ratio
SKID_LEN = 1 #[m]
S_TAIL = 3.25 #[m^2]
AR_T = 1.23 # aspect ratio
TIP_TO_CHORD = 0.1
TAIL_SWEEP_C4 = 15 #[deg]
TAIL_SAFETY_FACTOR = 1.5 # this safety factor accounts for the tail carrying the second wing
T_ELAPSED_VC = 5 #[s]
INIT_ELEV = 0 #[m]
FIN_ELEV = 10 #[m]
V_AVG_TO = 2 #[m/s]
V_HOVER = 3 #[m/s]
RHO_ORIGIN = 1.225 #[kg/m^3]
POWER_SAFETY_FACTOR = 1.4
PM = 0.5 # power margin
N_MOTOR = 6
N_PROP = 6
N_BLADES = 5
D_PROP = 1.9 #[m]

def COMPUTE_MTOW(MTOW):

    # POWER CALCULATIONS

    AVG_THRUST = MTOW * (G + (V_HOVER / T_ELAPSED_VC))
    V_HOVER_I = np.sqrt(AVG_THRUST / (2 * 6 * RHO_ORIGIN * A_DISK))
    V_AVG_I = - V_AVG_TO / 2 + np.sqrt((V_AVG_TO / 2) ** 2 + V_HOVER_I ** 2)
    MAX_POWER = (POWER_SAFETY_FACTOR * (AVG_THRUST * (V_AVG_I + V_AVG_TO)) / (FM)) / 1000

    # BATTERY MASS CALCULATIONS

    def takeoff_power(mtow_kg, A_disk, Vs_avg, Vs_f, N_prop):

        thrust = mtow_kg*(G + Vs_f/T_TAKEOFF)
        Vi_avg = -(Vs_avg/2) + np.sqrt((Vs_avg/2)**2 + thrust/(2*1.225*A_disk*N_prop))
        p_ideal = thrust*(Vi_avg + Vs_avg)/FM
        return p_ideal/ETA_POWERTRAIN_HOVER

    def vertical_climb_power(mtow_kg, A_disk, Vs, N_prop):

        thrust = mtow_kg*(G + Vs/T_TAKEOFF)
        Vi_avg = -(Vs/2) + np.sqrt((Vs/2)**2 + thrust/(2*1.225*A_disk*N_prop))
        p_ideal = thrust*(Vi_avg + Vs)/FM
        return p_ideal/ETA_POWERTRAIN_HOVER

    def climb_acceleration_power(mtow_kg, V_i, V_f, Vs, acc_time):

        acc = (V_f - V_i)/(acc_time)
        E_req = 0.5*mtow_kg*(V_f**2 - V_i**2) + mtow_kg*G*Vs*acc_time + (0.5*mtow_kg*G*acc*acc_time**2)/(LD_CRUISE)
        return E_req/ETA_CLIMB

    def climb_power(mtow_kg, climb_angle_deg, V_climb_horizontal, Vs):

        thrust = ((mtow_kg*G)*((np.cos(np.radians(climb_angle_deg)))/(LD_CRUISE))) + ((mtow_kg*G)*(np.sin(np.radians(climb_angle_deg))))
        p_ideal = thrust * np.sqrt(V_climb_horizontal**2 + Vs**2)
        return p_ideal/ETA_CLIMB

    def cruise_power(mtow_kg):

        return mtow_kg * G * V_CRUISE / (LD_CRUISE * ETA_CRUISE)

    def landing_power(mtow_kg, A_disk, Vs_0, N_prop):

        thrust = mtow_kg*(G + Vs_0/T_LANDING)
        Vs_avg = Vs_0 / 3
        Vi_avg = -(Vs_avg/2) + np.sqrt((Vs_avg/2)**2 + thrust/(2*1.00583*A_disk*N_prop))
        p_ideal = thrust*(Vi_avg - Vs_avg)/FM
        return p_ideal/ETA_POWERTRAIN_HOVER

    def mission_energy(mtow_kg):

        p_to = takeoff_power(mtow_kg, A_disk=np.pi*(PROP_DIAMETER/2)**2, Vs_avg=2, Vs_f=3, N_prop=6)
        p_vc = vertical_climb_power(mtow_kg, A_disk=np.pi*(PROP_DIAMETER/2)**2, Vs=3, N_prop=6)
        p_cl_acc = climb_acceleration_power(mtow_kg, V_i=20, V_f=V_CRUISE, Vs=3, acc_time=T_CLIMB_ACC)
        p_cl = climb_power(mtow_kg, climb_angle_deg=3.09097, V_climb_horizontal=V_CRUISE, Vs=3)
        p_c = cruise_power(mtow_kg)
        p_l = landing_power(mtow_kg, A_disk=np.pi*(PROP_DIAMETER/2)**2, Vs_0=7.6, N_prop=6)

        return (p_to * T_TAKEOFF
                + p_vc * T_VERTICAL_CLIMB
                + p_c * T_CRUISE
                + p_cl * T_CLIMB
                + p_cl_acc
                + p_l * T_LANDING)

    def battery_mass_from_mtow(mtow_kg):
 
        e_mission_Wh = mission_energy(mtow_kg) / 3600.0
        e_installed_Wh = e_mission_Wh * CONTINGENCY / SOC_USABLE
        return e_installed_Wh / E_PACK_WH_KG

    # MASS CALCULATIONS

    M_F = (0.453592) * (14.86 * ((MTOW * 2.20462) ** 0.144) * ((L_FUS * 3.28084) ** 0.778) / (PER_FUS_MAX * 3.28084)) * ((L_FUS * 3.28084) ** 0.383) * N_PAX ** 0.455
    M_W = (0.453592) * 2 * 0.04674 * ((MTOW * 2.20462) ** 0.397) * ((S_W * 3.28084 ** 2) ** 0.36) * (N_W ** 0.397) * (AR_W ** 1.712)
    M_LG = MTOW * 0.07
    M_TAIL = TAIL_SAFETY_FACTOR * (0.453592) * ((1.68 * ((MTOW * 2.20462) ** 0.567) * ((S_TAIL * 3.28084 ** 2) ** 1.249) * (AR_T ** 0.482)) / (639.95 * (TIP_TO_CHORD ** 0.747) * (np.cos(TAIL_SWEEP_C4 * (np.pi / 180)) ** 0.882)))
    M_MOTOR = 0.165 * ((MAX_POWER * (1 + PM)) / (N_MOTOR))
    M_PROP = 0.144 * ((D_PROP * (MAX_POWER / N_PROP) * N_BLADES ** 0.5) ** 0.782)
    M_PROP_TOT = N_PROP * M_PROP
    M_BATT = battery_mass_from_mtow(MTOW)
    M_MISC = 0.20 * MTOW

    MTOW = M_F + M_W + M_LG + M_TAIL + M_MOTOR + M_PROP_TOT + M_PAYLOAD + M_BATT + M_MISC

    return MTOW

# ITERATOR

BOUND_LOW_MTOW = 1000
BOUND_HIGH_MTOW = 5500
GUESS_OLD = (BOUND_HIGH_MTOW + BOUND_LOW_MTOW) / 2

ITERATE = True
COUNT = 0

while ITERATE == True:
    GUESS_NEW = COMPUTE_MTOW(GUESS_OLD)

    if GUESS_NEW > BOUND_HIGH_MTOW or GUESS_NEW < BOUND_LOW_MTOW:
        print("ur model is cooked buddy")
        ITERATE = False
    elif np.abs(GUESS_NEW - GUESS_OLD) / GUESS_OLD < 0.01:
        ITERATE = False
        print("Final MTOW: " + str(GUESS_NEW))

    STEP = (GUESS_NEW - GUESS_OLD) / 2
    GUESS_OLD = GUESS_OLD + STEP
    COUNT += 1

print("Total Iterations: " + str(COUNT))
    




