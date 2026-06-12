
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, cumulative_trapezoid
from parameters import *




def calc_winglet_mass(root_chord):
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

            # Highlight the evaluated point if one was requested
            if x_eval is not None:
                eval_moment = np.interp(x_eval, x, M)
                ax2.plot(x_eval, eval_moment, 'ko', markersize=8)
                ax2.annotate(f"{eval_moment:.1f} N·m",
                             (x_eval, eval_moment),
                             textcoords="offset points",
                             xytext=(10, 10),
                             ha='left',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1))

            plt.tight_layout()
            # plt.show()

        # 6. Return Logic
        if x_eval is not None:
            # Validate that x_eval is within the beam's length
            if x_eval < 0 or x_eval > L:
                raise ValueError(f"x_eval ({x_eval}) must be between 0 and the beam length ({L})")

            # Interpolate the moment array to find the value at x_eval
            return float(np.interp(x_eval, x, M))
        else:
            # Default back to the maximum absolute moment
            return float(np.max(np.abs(M)))

    def calculate_area_moment_of_inertia(root_chord, thick_chord_ratio, b, t_f, t_w, return_area=False):

        h = root_chord * thick_cord_ratio - 2 * t_f

        h_inner = h - 2 * t_f

        I_x = (b * h ** 3) / 12 - ((b - t_w) * h_inner ** 3) / 12

        if return_area:
            return ((h - 2 * t_f) * t_w + 2 * t_f * b)

        return I_x
    def calc_rib_spacing(stress):
        num = buckling_coeff * (np.pi ** 2) * young_mod * winglet_skin_thickness ** 2
        den = 12 * stress * (1 - poisson_ratio ** 2)
        return np.sqrt(num / den)
    def winglet_lift(x):
        return front_wing_distribution - x * (front_wing_distribution + back_wing_distribution) / winglet_length

    ###PARAMETERS

    front_wing_distribution = 500
    back_wing_distribution = 300

    # Wing & Winglet dimensions
    thick_cord_ratio = TIP_TO_CHORD_W

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
    flange_thickness = 0.001  # This is what will impact the mmoi the most
    beam_thickness = 0.001  # m


    # Material characteristics
    allowable_stress = 278e6 * 0.8
    poisson_ratio = 0.33
    buckling_coeff = 4
    young_mod = 70e9
    density = 2700


    I = calculate_area_moment_of_inertia(root_chord, thick_cord_ratio, flange_length, flange_thickness, beam_thickness)

    M_max = analyze_cantilever_distributed_load(winglet_length, winglet_lift, plot=False)

    max_stress_beam = M_max * root_chord*thick_cord_ratio/2 /(I * number_of_beams)

    #Using the rearranged critical buckling formula to find rib spacing
    curr_rib = calc_rib_spacing(max_stress_beam)
    ribslst = []

    while curr_rib < winglet_length:
        ribslst.append(curr_rib)
        M = analyze_cantilever_distributed_load(winglet_length, winglet_lift, x_eval=curr_rib)
        stress_beam = M * root_chord*thick_cord_ratio/2 /(I * number_of_beams)
        curr_rib += calc_rib_spacing(stress_beam)

    ribs_num = len(ribslst) + 2
    rib_area = winglet_area/winglet_length * 0.6 #Assuming ribs are about

    ribs_mass = ribs_num * rib_area * rib_thickness * density
    skin_mass = winglet_area * winglet_skin_thickness

    Ibeam_area = calculate_area_moment_of_inertia(root_chord, thick_cord_ratio, flange_length, flange_thickness, beam_thickness, return_area=True)
    Ibeam_mass = Ibeam_area * winglet_length * density

    print(f"The max stress is {max_stress_beam * 10**-6} Mpa")
    print(f"Number of ribs: {len(ribslst) + 2}")
    print(f"The total winglet mass is ({ribs_mass} + {skin_mass} + {Ibeam_mass} )* 2 = {(ribs_mass + skin_mass + Ibeam_mass)*2}")

    ###once the winglet dimensions have been done sizing


calc_winglet_mass(1.2)