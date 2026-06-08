
# This script determines the static longitudinal stability for various wing area distributions.
# A value for C_M_alpha is found for each wing area distribution.

import aircraft

def calc_wing_geom(ac, wing):
    """ This function computes the geometry characteristics of a wing based
      on its wing area and static geometry parameters. """

    if wing == 1:
        S = ac.params.wing_geometry.S_fw
        b = ac.params.wing_geometry.b_fw
        lamda = ac.params.wing_geometry.taper_fw
        d = ac.params.fuselage_geometry.d_1        
    else:
        S = ac.params.wing_geometry.S_aw
        b = ac.params.wing_geometry.b_aw
        lamda = ac.params.wing_geometry.taper_aw
        d = 0
        S_e = S

    # Compute root chord
    c_r = (2*S)/(b*(1+lamda))
    # Compute MAC
    MAC = (2/3)*c_r*((1 + lamda + lamda**2)/(1 + lamda))
    # Compute aspect ratio
    A = b**2 / S
    # Compute exposed wing area
    if wing == 1:
        S_e = ((b-d)/2) * ( c_r*(1-(1-lamda)*(d/b)) + lamda*c_r )

    # Return parameters
    return MAC, A, S_e


def __main__():
    
    # Define static margin and search space
    SM = 0.05
    S_aw_S_tot_ratio = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
    # Create an aircraft instance
    ac = aircraft.Aircraft()
    S_tot = ac.mass * 9.81 * (1/ac.wing_geometry.design_point)

    # Loop over each possible combination of wing areas
    for dist in S_aw_S_tot_ratio:

        # Compute new geometric properties
        S_2 = dist*S_tot
        S_1 = S_tot - S_2
        d_1, MAC_1, A_1, S_e_1 = calc_wing_geom(ac, 1)
        d_2, MAC_2, A_2, S_e_2 = calc_wing_geom(ac, 2)
        MAC = (S_1 * MAC_1 + S_2 * MAC_2)/S_tot

        # Update geometric properties of aircraft instance
        ac.params.wing_geometry.S_aw = S_2
        ac.params.wing_geometry.S_fw = S_1
        ac.params.wing_geometry.MAC_aw = MAC_2
        ac.params.wing_geometry.MAC_fw = MAC_1
        ac.params.wing_geometry.A_aw = A_2
        ac.params.wing_geometry.A_fw = A_1
        ac.params.wing_geometry.S_e_aw = S_e_2
        ac.params.wing_geometry.S_e_fw = S_e_1

        # 