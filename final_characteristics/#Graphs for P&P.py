#Graphs for P&P

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
sys.path.insert(0, os.path.join(ROOT, "class_II_sizing"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#from energy import takeoff_power, vertical_climb_power, climb_acceleration_power, climb_power, cruise_power, landing_power, mission_energy
from parameters import T_TAKEOFF, T_CLIMB_ACC, T_CLIMB, T_CRUISE, T_LANDING, V_AVG_TO, V_CRUISE, V_HOVER, N_PROP, ETA_POWERTRAIN_HOVER, ETA_CLIMB, ETA_CRUISE, LD_CRUISE,G, A_DISK, V_HOVER, RHO, C_D0, S, W_TO, W_PAY, W_PL, W_FUEL, T_TO, V_CLIMB, V_CRUISE, V_LANDING, H_CRUISE, H_LANDING, H_CLIMB, H_TAKEOFF, T_CLIMB_ACC, T_CRUISE, T_LANDING, N_PROP, ETA_POWERTRAIN_HOVER, ETA_CLIMB, ETA_CRUISE, LD_CRUISE, G, A_DISK, V_HOVER, RHO, C_D0, S, W_TO, W_PAY, W_PL, W_FUEL
from class_II_sizing.mtow_sizing import load_final_design_state
from energy import takeoff_power, vertical_climb_power, climb_acceleration_power, climb_power, cruise_power, landing_power, mission_energy  

design_state = load_final_design_state(verbose=False)
mtow= 1980

p_to     = takeoff_power(mtow_kg, a_disk=_A_DISK_PROP, vs_avg=2, vs_f=3, n_prop=N_PROP)
p_vc     = vertical_climb_power(mtow_kg, a_disk=_A_DISK_PROP, vs=3, n_prop=N_PROP)
p_cl_acc = climb_acceleration_power(mtow_kg, V_I, v_f=V_CRUISE, vs=3, acc_time=T_CLIMB_ACC)
p_cl     = climb_power(mtow_kg, climb_angle_deg=3.09097, v_climb_horizontal=V_CRUISE, vs=3)
p_c      = cruise_power(mtow_kg)
p_l      = landing_power(mtow_kg, a_disk=_A_DISK_PROP, vs_0=VS_0, n_prop=N_PROP)

plt.plot([T_TAKEOFF, T_CLIMB_ACC, T_CLIMB, T_CRUISE, T_LANDING], [p_to, p_vc, p_cl_acc, p_cl, p_c, p_l], 'o')
plt.title("Power vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Power (W)")
plt.show()

