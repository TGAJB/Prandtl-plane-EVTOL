
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, cumulative_trapezoid
from parameters import *


def analyze_cantilever_distributed_load(L, w_func, plot=False, x_eval=None):
    """
    Calculates reactions and plots load, shear, and moment diagrams
    for a cantilever beam (fixed at x=0) with a distributed load function.

    Parameters:
    L (float): Length of the beam (m)
    w_func (callable): Function taking position 'x' and returning load magnitude (N/m).
    plot (bool): If True, displays the shear and moment diagrams.
    x_eval (float, optional): Specific location along the beam (0 to L) to evaluate the moment.

    Returns:
    float: The bending moment at x_eval (if provided), otherwise the maximum absolute moment.
    """

    # 1. Calculate Reaction Forces at the Wall (x=0)
    W, _ = quad(w_func, 0, L)

    moment_integrand = lambda x: x * w_func(x)
    M_wall, _ = quad(moment_integrand, 0, L)

    R_y = W

    if plot:
        print("-" * 35)
        print("WALL REACTIONS (at x = 0)")
        print("-" * 35)
        print(f"Vertical Reaction (R_y): {R_y:.2f} N (Upward)")
        print(f"Reaction Moment (M):     {M_wall:.2f} N·m (Counter-Clockwise)\n")

    # 2. Discretize the beam for array operations
    x = np.linspace(0, L, 1000)
    w_arr = np.array([w_func(xi) for xi in x])

    # 3. Calculate Shear (V)
    integral_w = cumulative_trapezoid(w_arr, x, initial=0)
    V = R_y - integral_w

    # 4. Calculate Bending Moment (M)
    M = -M_wall + cumulative_trapezoid(V, x, initial=0)

    if plot:
        # 5. Plot the Diagrams
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

        # Applied Load Diagram
        ax0.plot(x, -w_arr, color='green', linewidth=2)
        ax0.fill_between(x, -w_arr, 0, color='green', alpha=0.2)
        ax0.axhline(0, color='black', linewidth=1)
        ax0.set_ylabel('Distributed Load\n$w(x)$ (N/m)')
        ax0.set_title('Cantilever Beam Analysis (Fixed at $x=0$)')
        ax0.grid(True, linestyle='--', alpha=0.6)

        # Shear Diagram
        ax1.plot(x, V, color='blue', linewidth=2)
        ax1.fill_between(x, V, 0, color='blue', alpha=0.2)
        ax1.axhline(0, color='black', linewidth=1)
        ax1.set_ylabel('Shear Force\n$V$ (N)')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Moment Diagram
        ax2.plot(x, M, color='red', linewidth=2)
        ax2.fill_between(x, M, 0, color='red', alpha=0.2)
        ax2.axhline(0, color='black', linewidth=1)
        ax2.set_ylabel('Bending Moment\n$M$ (N·m)')
        ax2.set_xlabel('Position from wall, $x$ (m)')
        ax2.grid(True, linestyle='--', alpha=0.6)


        plt.tight_layout()
        # plt.show()

    return M, x


def calculate_Ibeam_moment_of_inertia(root_chord, thick_chord_ratio, b, t_f, t_w, return_area=False):

        h = root_chord * thick_chord_ratio - 2 * t_f

        h_inner = h - 2 * t_f

        I_x = (b * h ** 3) / 12 - ((b - t_w) * h_inner ** 3) / 12

        if return_area:
            return ((h - 2 * t_f) * t_w + 2 * t_f * b)

        return I_x


def calc_individual_rib_spacing(stress, buckling_coeff, young_mod, skin_thickness, poisson_ratio):
    num = buckling_coeff * (np.pi ** 2) * young_mod * skin_thickness ** 2
    den = 12 * stress * (1 - poisson_ratio ** 2)
    return np.sqrt(num / den)

def calc_total_rib_spacing(max_stress_beam, wing_length, root_chord, thick_chord_ratio, I, number_of_beams, M, x, buckling_coeff, young_mod, skin_thickness, poisson_ratio):
    #Using the rearranged critical buckling formula to find rib spacing
    curr_rib = calc_individual_rib_spacing(max_stress_beam, buckling_coeff, young_mod, skin_thickness, poisson_ratio)
    ribslst = []

    while curr_rib < wing_length:
        ribslst.append(curr_rib)
        Moment_at_point = float(np.interp(curr_rib, x, M))
        stress_beam = Moment_at_point * root_chord*thick_chord_ratio/2 /(I * number_of_beams)
        curr_rib += calc_individual_rib_spacing(stress_beam, buckling_coeff, young_mod, skin_thickness, poisson_ratio)

    ribslst.insert(0, 0)
    ribslst.append(wing_length)

    return ribslst



def calc_winglet_mass(root_chord):
    def winglet_lift(x):
        return front_wing_distribution - x * (front_wing_distribution + back_wing_distribution) / winglet_length

    ###PARAMETERS

    front_wing_distribution = 500
    back_wing_distribution = 300

    # Wing & Winglet dimensions
    thick_chord_ratio = TIP_TO_CHORD_W
    taper_ratio = 1

    wings_vertical_spacing = 2  # UPDATE THESE LATER
    wings_horizontal_spacing = 5  # UPDATE THESE LATER

    winglet_length = np.sqrt(wings_horizontal_spacing ** 2 + wings_vertical_spacing ** 2)  # ???
    winglet_area = 1.57  # From utku, UPDATE LATER
    winglet_skin_thickness = WINGLET_SKIN_THICKNESS

    ### Characteristics unique to the winglet, NOT USED ANYWHERE ELSE
    number_of_beams = 1
    rib_thickness = 0.001

    # I_BEAM
    flange_length = 0.01
    flange_thickness = 0.004  # This is what will impact the mmoi the most
    beam_thickness = 0.004  # m

    # Material characteristics
    allowable_stress = 278e6 * 0.8
    poisson_ratio = 0.33
    buckling_coeff = 4
    young_mod = 70e9
    density = 2700


    I = calculate_Ibeam_moment_of_inertia(root_chord, thick_chord_ratio, flange_length, flange_thickness, beam_thickness)

    M, x = analyze_cantilever_distributed_load(winglet_length, winglet_lift, plot=False)

    M_max = float(np.max(np.abs(M)))

    max_stress_beam = M_max * root_chord*thick_chord_ratio/2 /(I * number_of_beams)

    ribslst = calc_total_rib_spacing(max_stress_beam, winglet_length, root_chord, thick_chord_ratio, I, number_of_beams, M, x, buckling_coeff, young_mod, winglet_skin_thickness, poisson_ratio)

    surface_area = winglet_area/winglet_length

    rib_surface_area = surface_area/((taper_ratio - 1)*0.5 + 1)**2 * 0.6 #Assuming the rib area is 0.6 times the airfoil cross section due to holes & cutouts

    #ribs_mass = rib_surface_area * density * rib_thickness * len(ribslst)
    ribs_mass = 0

    for rib_pos in ribslst:
        point_taper = ((taper_ratio - 1)/winglet_length * rib_pos + 1)
        rib_volume = surface_area * point_taper**2 * rib_thickness
        ribs_mass += rib_volume * density

    print(ribs_mass)
    skin_mass = winglet_area * winglet_skin_thickness

    Ibeam_area = calculate_Ibeam_moment_of_inertia(root_chord, thick_chord_ratio, flange_length, flange_thickness, beam_thickness, return_area=True)
    Ibeam_mass = Ibeam_area * winglet_length * density

    print(f"The max stress is {max_stress_beam * 10**-6} Mpa")
    print(f"Number of ribs: {len(ribslst) + 2}")
    print(f"The total winglet mass is ({ribs_mass} + {skin_mass} + {Ibeam_mass} )* 2 = {(ribs_mass + skin_mass + Ibeam_mass)*2}")

    ###once the winglet dimensions have been done sizing


calc_winglet_mass(1.2)