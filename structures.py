from constants import *
from ppe import *
from control import *
from aerodynamics import *
from vehicle_dynamics import *
import numpy as np
import math


#propeller failure loads
#propeller structure
#propeller material selection
#critical loading areas
#feasible wing shape
#fuselage shape
#CG envelope
#MMOI
#MTOW
#OEW
#vehicle lifetime

def compute_mmoI(components, cg=(0, 0, 0)):
    """
    Computes aircraft mass moments of inertia.

    Parameters
    ----------
    components : list of dicts
        Each component must contain:
        {
            "mass": float,
            "position": (x, y, z),
            "I_local": (Ixx, Iyy, Izz)
        }

    cg : tuple
        Aircraft center of gravity (x, y, z)

    Returns
    -------
    dict
        Total Ixx, Iyy, Izz
    """

    Ixx_total = 0
    Iyy_total = 0
    Izz_total = 0

    x_cg, y_cg, z_cg = cg

    for comp in components:

        m = comp["mass"]
        x, y, z = comp["position"]

        # Relative position to aircraft CG
        dx = x - x_cg
        dy = y - y_cg
        dz = z - z_cg

        # Local inertia
        Ixx_local, Iyy_local, Izz_local = comp["I_local"]

        # Parallel axis theorem
        Ixx = Ixx_local + m * (dy**2 + dz**2)
        Iyy = Iyy_local + m * (dx**2 + dz**2)
        Izz = Izz_local + m * (dx**2 + dy**2)

        Ixx_total += Ixx
        Iyy_total += Iyy
        Izz_total += Izz

    return {
        "Ixx": Ixx_total,
        "Iyy": Iyy_total,
        "Izz": Izz_total
    }