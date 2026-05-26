from constants import *
from ppe import *
from control import *
from aerodynamics import *
from vehicle_dynamics import *
import numpy as np
import math
#inputs

#from PPE
#battery mass
#no. of propulsion systems
#location of propulsion systems
#type and length of wiring
#max thrust per engine
#propeller load cases
#propeller diametre

#from aerodyn
#wing design
#pos. of vertical wings
#pos. of vertical tail
#locations of control surfaces
#airfoil type
#wing load distribution
#fuselage shape and size

#from VD
#long. positions of horizontal wings
#gap
#stagger
#ultimate load factor

#from B&O
#no. of passengers 
# vehicle lifetime
# vehicle size envelope
# payload mass

#outputs
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