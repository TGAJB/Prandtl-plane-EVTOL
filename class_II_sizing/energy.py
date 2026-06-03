"""
energy.py
Mission power segment functions, total mission energy, and battery mass.
All functions take mtow_kg as their first argument so they can be called
at any point in the MTOW iteration.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from parameters import (
    G, FM, RHO_ORIGIN,
    ETA_POWERTRAIN_HOVER, ETA_CLIMB, ETA_CRUISE, LD_CRUISE, V_CRUISE,
    T_TAKEOFF, T_LANDING, T_CRUISE, T_CLIMB, T_CLIMB_ACC, T_VERTICAL_CLIMB,
    D_PROP, N_PROP,
    E_PACK_WH_KG, SOC_USABLE, CONTINGENCY, VS_0, V_I
)

_A_DISK_PROP = np.pi * (D_PROP / 2) ** 2   # disk area from D_PROP


def takeoff_power(mtow_kg, a_disk, vs_avg, vs_f, n_prop):
    thrust  = mtow_kg * (G + vs_f / T_TAKEOFF)
    vi_avg  = -(vs_avg / 2) + np.sqrt((vs_avg / 2) ** 2 + thrust / (2 * 1.225 * a_disk * n_prop))
    p_ideal = thrust * (vi_avg + vs_avg) / FM
    return p_ideal / ETA_POWERTRAIN_HOVER


def vertical_climb_power(mtow_kg, a_disk, vs, n_prop):
    thrust  = mtow_kg * (G + vs / T_TAKEOFF)
    vi_avg  = -(vs / 2) + np.sqrt((vs / 2) ** 2 + thrust / (2 * 1.225 * a_disk * n_prop))
    p_ideal = thrust * (vi_avg + vs) / FM
    return p_ideal / ETA_POWERTRAIN_HOVER


def climb_acceleration_power(mtow_kg, v_i, v_f, vs, acc_time):
    acc   = (v_f - v_i) / acc_time
    e_req = (0.5 * mtow_kg * (v_f ** 2 - v_i ** 2)
             + mtow_kg * G * vs * acc_time
             + (0.5 * mtow_kg * G * acc * acc_time ** 2) / LD_CRUISE)
    return e_req / ETA_CLIMB


def climb_power(mtow_kg, climb_angle_deg, v_climb_horizontal, vs):
    thrust  = ((mtow_kg * G) * (np.cos(np.radians(climb_angle_deg)) / LD_CRUISE)
               + (mtow_kg * G) * np.sin(np.radians(climb_angle_deg)))
    p_ideal = thrust * np.sqrt(v_climb_horizontal ** 2 + vs ** 2)
    return p_ideal / ETA_CLIMB


def cruise_power(mtow_kg):
    return mtow_kg * G * V_CRUISE / (LD_CRUISE * ETA_CRUISE)


def landing_power(mtow_kg, a_disk, vs_0, n_prop):
    thrust  = mtow_kg * (G + vs_0 / T_LANDING)
    vs_avg  = vs_0 / 3
    vi_avg  = -(vs_avg / 2) + np.sqrt((vs_avg / 2) ** 2 + thrust / (2 * 1.00583 * a_disk * n_prop))
    p_ideal = thrust * (vi_avg - vs_avg) / FM
    return p_ideal / ETA_POWERTRAIN_HOVER


def mission_energy(mtow_kg):
    """Total mission energy [W·s] for a given MTOW estimate."""
    p_to     = takeoff_power(mtow_kg, a_disk=_A_DISK_PROP, vs_avg=2, vs_f=3, n_prop=N_PROP)
    p_vc     = vertical_climb_power(mtow_kg, a_disk=_A_DISK_PROP, vs=3, n_prop=N_PROP)
    p_cl_acc = climb_acceleration_power(mtow_kg, V_I, v_f=V_CRUISE, vs=3, acc_time=T_CLIMB_ACC)
    p_cl     = climb_power(mtow_kg, climb_angle_deg=3.09097, v_climb_horizontal=V_CRUISE, vs=3)
    p_c      = cruise_power(mtow_kg)
    p_l      = landing_power(mtow_kg, a_disk=_A_DISK_PROP, vs_0=VS_0, n_prop=N_PROP)
    return (p_to  * T_TAKEOFF
            + p_vc   * T_VERTICAL_CLIMB
            + p_c    * T_CRUISE
            + p_cl   * T_CLIMB
            + p_cl_acc
            + p_l    * T_LANDING)


def battery_mass(mtow_kg):
    """Installed battery mass [kg] required to complete the mission."""
    e_mission_wh   = mission_energy(mtow_kg) / 3600.0
    e_installed_wh = e_mission_wh * CONTINGENCY / SOC_USABLE
    return e_installed_wh / E_PACK_WH_KG
