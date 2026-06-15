
# This script determines the static longitudinal stability for various wing area distributions.
# A value for C_M_alpha is found for each wing area distribution.

import aircraft
import numpy as np
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import plotly.graph_objects as go
from pathlib import Path
from VTOL_cg_envelope_det import (
    VTOL_STATIC_MARGIN_GLOBAL_MAC,
    get_vtol_oei_cg_envelope,
)
#from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

def calc_wing_geom(ac, S, wing):
    """ This function computes the geometry characteristics of a wing based
      on its wing area and static geometry parameters. """

    if wing == 1:
        #S = ac.params.wing_geometry.S_fw
        b = ac.params.wing_geometry.b_fw
        lamda = ac.params.wing_geometry.taper_fw
        d = ac.params.fuselage_geometry.d_fw        
    else:
        #S = ac.params.wing_geometry.S_aw
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


def main():
    
    # Define static margin and search space
    SM = 0.05
    S_aw_S_tot_ratio = np.linspace(0.25, 0.75, 11)
    #S_aw_S_tot_ratio = [0.50]
    # Create an aircraft instance
    params = aircraft.AircraftParameters()
    physical = aircraft.Physical()
    fc = aircraft.FlightCondition()
    dc = aircraft.DatcomChartInputs()
    ac = aircraft.Aircraft(params, physical, fc, dc)
    # The location of the aerodynamic centre will change with each new wing area.
    # The first wing area is used to derive the CG search space
    MAC_1 = ac.params.wing_geometry.MAC_fw
    MAC_2 = ac.params.wing_geometry.MAC_aw
    CG_lower = ac.params.aerodynamics.x_ac_fw_cruise/MAC_1
    CG_upper = (ac.params.aerodynamics.x_ac_fw_cruise + ac.params.wing_geometry.stagger)/MAC_1
    CG_envelope = np.linspace(CG_lower, CG_upper, 11)
    # Compute the total wing area based on the MTOW and design point
    S_tot = ac.params.mass.mtow * 9.81 * (1/ac.params.wing_geometry.design_point)

    # Loop over each possible combination of wing areas and record the location of the NP, and the C_M_alphas
    np_lst = [] # NP seen from the nose of the a/c in [m]
    allowable_aft_cg_lst = [] # Seen from the nose of the a/c in [m]
    C_M_alpha_outer_lst = [] # Normalised by the MAC
    cg_x_outer_lst = []  # CG locations from nose in [m]
    MAC_fw_lst = [] # Front-wing MAC for each wing-area distribution

    for dist in S_aw_S_tot_ratio:

        # Compute new geometric properties
        S_2 = dist*S_tot
        S_1 = S_tot - S_2
        MAC_1, A_1, S_e_1 = calc_wing_geom(ac, S_1, 1)
        MAC_2, A_2, S_e_2 = calc_wing_geom(ac, S_2, 2)
        MAC = (S_1 * MAC_1 + S_2 * MAC_2)/S_tot
        MAC_fw_lst.append(MAC_1)

        # Update geometric properties of aircraft instance
        ac.params.wing_geometry.S_aw = S_2
        ac.params.wing_geometry.S_fw = S_1
        ac.params.wing_geometry.MAC_aw = MAC_2
        ac.params.wing_geometry.MAC_fw = MAC_1
        ac.params.wing_geometry.A_aw = A_2
        ac.params.wing_geometry.A_fw = A_1
        ac.params.wing_geometry.S_e_aw = S_e_2
        ac.params.wing_geometry.S_e_fw = S_e_1

        # Parameters independent from CG-location
        CL_alpha_2 = ac.params.aerodynamics.CL_alpha_aw
        downwash_grad = ac.downwash_gradient()
        a = ac.CL_alpha_aircraft()
        CL_2_CL_ratio = CL_alpha_2 * (1 - downwash_grad) * (1/a)
        aft_wing_volume = (S_2 * ac.params.wing_geometry.stagger)/(S_tot * MAC)
        # Compute location of NP
        x_np_norm = ac.params.aerodynamics.x_ac_fw_cruise/MAC_1 + CL_2_CL_ratio * aft_wing_volume * (MAC/MAC_1)
        # NP-location is currently normalised by MAC_1; should eventually be plotted as a location from the a/c nose
        x_np = x_np_norm*MAC_1 + ac.params.wing_geometry.x_LEMAC_fw
        np_lst.append(x_np)
        # Compute the allowable most aft CG
        allowable_aft_cg = x_np - SM*MAC
        allowable_aft_cg_lst.append(allowable_aft_cg)

        # Loop over all CG-positions:
        C_M_alpha_inner_lst = []
        cg_x_inner_lst = []

        for cg in CG_envelope:
            x_cg = cg*MAC_1 + ac.params.wing_geometry.x_LEMAC_fw # As seen from the a/c nose
            cg_x_inner_lst.append(x_cg)

            # Quantify the stati c longitudinal stability by applying the change in lift at the NP
            l_cg_np = ((cg - x_np_norm) * MAC_1)/MAC
            C_M_alpha = l_cg_np * a
            C_M_alpha_inner_lst.append(C_M_alpha)

        C_M_alpha_outer_lst.append(C_M_alpha_inner_lst)
        cg_x_outer_lst.append(cg_x_inner_lst)

    
    # Convert lists to NumPy arrays
    np_arr = np.array(np_lst)
    allowable_aft_cg_arr = np.array(allowable_aft_cg_lst)
    C_M_alpha_arr = np.array(C_M_alpha_outer_lst)
    cg_x_arr = np.array(cg_x_outer_lst)
    MAC_fw_arr = np.array(MAC_fw_lst)
    allowable_aft_cg_fw_mac_arr = (
        (allowable_aft_cg_arr - ac.params.wing_geometry.x_LEMAC_fw) / MAC_fw_arr
    )

    #print("C_M_alpha_arr dtype:", C_M_alpha_arr.dtype)
    #print("Any complex values:", np.iscomplexobj(C_M_alpha_arr))

    #if np.iscomplexobj(C_M_alpha_arr):
    #    imag_max = np.max(np.abs(np.imag(C_M_alpha_arr)))
    #    print("Maximum imaginary part:", imag_max)

    # ==================================================
    # PLOTTING SECTION
    # ==================================================
    if 1:
        # PATH DEFINITION
        script_dir = Path(__file__).resolve().parent
        plot_dir = script_dir / "plots" / "stat_long_stab_vtol_oei"
        plot_dir.mkdir(parents=True, exist_ok=True)

        print("Saving plots to:", plot_dir)
        print("Folder exists:", plot_dir.exists())

        # ==================================================
        # Plot 1: Wing area distribution vs neutral point
        # ==================================================

        plt.figure(figsize=(7, 5))

        plt.plot(
            S_aw_S_tot_ratio,
            np_arr,
            marker="o",
            linestyle="-"
        )

        plt.xlabel(r"$S_{aw}/S_{tot}$ [-]")
        plt.ylabel(r"$x_{np}$ from nose [m]")
        plt.title("Neutral Point Location vs Wing Area Distribution")
        plt.grid(True)

        plt.savefig(
        plot_dir / "wing_area_dist_vs_np.png",
        dpi=300,
        bbox_inches="tight"
        )
        plt.close()
        #plt.show()


        # ==================================================
        # Plot 2: Wing area distribution vs most aft allowable CG
        # ==================================================

        plt.figure(figsize=(7, 5))

        plt.plot(
            S_aw_S_tot_ratio,
            allowable_aft_cg_arr,
            marker="o",
            linestyle="-"
        )

        plt.xlabel(r"$S_{aw}/S_{tot}$ [-]")
        plt.ylabel(r"$x_{cg,aft}$ from nose [m]")
        plt.title("Most Aft Allowable CG Location vs Wing Area Distribution")
        plt.grid(True)

        plt.savefig(
        plot_dir / "wing_area_dist_vs_aft_cg.png",
        dpi=300,
        bbox_inches="tight"
        )
        plt.close()
        #plt.show()

        # ==================================================
        # Plot 3: Wing area distribution vs allowable aft CG,
        # with VTOL OEI static-margin CG envelope
        # ==================================================

        vtol_envelope = get_vtol_oei_cg_envelope()

        plt.figure(figsize=(7, 5))

        plt.plot(
            S_aw_S_tot_ratio,
            allowable_aft_cg_fw_mac_arr,
            marker="o",
            linestyle="-",
            label=rf"$x_{{cg,aft}}$ with cruise SM = {SM:.2f}"
        )

        if vtol_envelope is not None:
            vtol_allowable = vtol_envelope["allowable_with_static_margin"]
            vtol_min = vtol_allowable["x_min_mac"]
            vtol_max = vtol_allowable["x_max_mac"]

            plt.axhline(
                y=vtol_min,
                color="tab:green",
                linestyle="--",
                linewidth=1.4,
                label=rf"VTOL OEI forward limit, SM = {VTOL_STATIC_MARGIN_GLOBAL_MAC:.2f}"
            )
            plt.axhline(
                y=vtol_max,
                color="tab:red",
                linestyle="--",
                linewidth=1.4,
                label=rf"VTOL OEI aft limit, SM = {VTOL_STATIC_MARGIN_GLOBAL_MAC:.2f}"
            )

            if not vtol_allowable["is_valid"]:
                print(
                    "VTOL OEI static-margin envelope is invalid. "
                    "Reduce VTOL_STATIC_MARGIN_GLOBAL_MAC in VTOL_cg_envelope_det.py."
                )
                print(
                    "Suggested maximum static margin:",
                    f"{vtol_allowable['suggested_static_margin']:.4f}",
                )
        else:
            print("VTOL OEI CG envelope is infeasible; no VTOL limits added to the plot.")

        plt.xlabel(r"$S_{aw}/S_{tot}$ [-]")
        plt.ylabel(r"$(x_{cg,aft} - x_{LEMAC,fw})/MAC_{fw}$ [-]")
        plt.title("Allowable Aft CG vs Wing Area Distribution")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            plot_dir / "wing_area_dist_vs_aft_cg_norm_with_vtol_oei.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


        # ==================================================
        # Plot 4: Wing area distribution vs allowable aft CG
        # from nose datum, with VTOL OEI static-margin CG envelope
        # ==================================================

        plt.figure(figsize=(7, 5))

        plt.plot(
            S_aw_S_tot_ratio,
            allowable_aft_cg_arr,
            marker="o",
            linestyle="-",
            label=rf"$x_{{cg,aft}}$ with cruise SM = {SM:.2f}"
        )

        if vtol_envelope is not None:
            vtol_allowable = vtol_envelope["allowable_with_static_margin"]
            vtol_min = vtol_allowable["x_min_nose"]
            vtol_max = vtol_allowable["x_max_nose"]

            plt.axhline(
                y=vtol_min,
                color="tab:green",
                linestyle="--",
                linewidth=1.4,
                label=rf"VTOL OEI forward limit, SM = {VTOL_STATIC_MARGIN_GLOBAL_MAC:.2f}"
            )
            plt.axhline(
                y=vtol_max,
                color="tab:red",
                linestyle="--",
                linewidth=1.4,
                label=rf"VTOL OEI aft limit, SM = {VTOL_STATIC_MARGIN_GLOBAL_MAC:.2f}"
            )

        plt.xlabel(r"$S_{aw}/S_{tot}$ [-]")
        plt.ylabel(r"$x_{cg,aft}$ from nose [m]")
        plt.title("Allowable Aft CG vs Wing Area Distribution")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            plot_dir / "wing_area_dist_vs_aft_cg_nose_with_vtol_oei.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()


        # ==================================================
        # Plot 5: 3D plot of C_M_alpha
        # ==================================================
        
        # Create grid for wing area distribution
        # Create grid for wing area distribution and normalized CG location
        S_dist_grid, CG_norm_grid = np.meshgrid(
            S_aw_S_tot_ratio,
            CG_envelope,
            indexing="ij"
        )

        # Grid arrays
        X = S_dist_grid              # Wing area distribution
        Y = CG_norm_grid             # CG location normalized by front-wing MAC
        Z = C_M_alpha_arr            # C_M_alpha

        # Flatten arrays for scatter points and projection lines
        S_flat = X.flatten()
        cg_flat = Y.flatten()
        CM_flat = Z.flatten()

        # --------------------------------------------------
        # 3D interpolated sheet
        # --------------------------------------------------

        surface = go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="Cividis",
            opacity=1.0,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="C_M_alpha [1/rad]",
                    side="right"
                ),
                thickness=18,
                len=0.75
            ),
            name="C_M_alpha surface",
            hovertemplate=(
                "S_aw/S_tot: %{x:.2f}<br>"
                "x_cg/MAC_fw: %{y:.2f}<br>"
                "C_M_alpha: %{z:.2f} 1/rad"
                "<extra></extra>"
            )
        )

        # --------------------------------------------------
        # Scatter points with value labels
        # --------------------------------------------------

        scatter = go.Scatter3d(
            x=S_flat,
            y=cg_flat,
            z=CM_flat,
            mode="markers+text",
            marker=dict(
                size=3,
                color=CM_flat,
                colorscale="Cividis",
                showscale=False,
                line=dict(
                    color="black",
                    width=1
                )
            ),
            # Leading space makes minus signs easier to see.
            # No plus signs are shown for positive values.
            text=[f" {value:.2f}" for value in CM_flat],
            textposition="top center",
            textfont=dict(
                size=10,
                color="black"
            ),
            name="Modelled points",
            hovertemplate=(
                "S_aw/S_tot: %{x:.2f}<br>"
                "x_cg/MAC_fw: %{y:.2f}<br>"
                "C_M_alpha: %{z:.2f} 1/rad"
                "<extra></extra>"
            )
        )

        # --------------------------------------------------
        # Dashed projection lines from each point to X-Y plane
        # --------------------------------------------------

        line_traces = []

        for x, y, z in zip(S_flat, cg_flat, CM_flat):
            line_traces.append(
                go.Scatter3d(
                    x=[x, x],
                    y=[y, y],
                    z=[0, z],
                    mode="lines",
                    line=dict(
                        color="black",
                        width=2,
                        dash="dash"
                    ),
                    showlegend=False,
                    hoverinfo="skip"
                )
            )

        # --------------------------------------------------
        # Create figure
        # --------------------------------------------------

        fig = go.Figure(data=[surface, scatter] + line_traces)

        fig.update_layout(
            title=dict(
                text="Static Longitudinal Stability: C_M_alpha",
                x=0.5
            ),
            scene=dict(
                xaxis=dict(
                    title="S_aw/S_tot [-]",
                    showgrid=True,
                    gridcolor="lightgray",
                    zeroline=True,
                    zerolinecolor="black"
                ),
                yaxis=dict(
                    title="x_cg/MAC_fw [-]",
                    showgrid=True,
                    gridcolor="lightgray",
                    zeroline=True,
                    zerolinecolor="black"
                ),
                zaxis=dict(
                    title="C_M_alpha [1/rad]",
                    showgrid=True,
                    gridcolor="lightgray",
                    zeroline=True,
                    zerolinecolor="black"
                ),

                # Similar to ax.set_box_aspect((2.0, 1.5, 1.0))
                aspectmode="manual",
                aspectratio=dict(
                    x=2.0,
                    y=1.5,
                    z=1.0
                ),

                camera=dict(
                    eye=dict(
                        x=1.6,
                        y=-1.8,
                        z=1.1
                    )
                )
            ),
            width=1300,
            height=900,
            margin=dict(
                l=10,
                r=10,
                b=10,
                t=50
            ),
            template="plotly_white",
            showlegend=False
        )

        # --------------------------------------------------
        # Store as interactive HTML file
        # --------------------------------------------------

        html_path = plot_dir / "CM_alpha_interactive_3D_plot.html"

        fig.write_html(
            html_path,
            include_plotlyjs=True,
            full_html=True
        )

        print("Saved interactive 3D plot to:")
        print(html_path.resolve())

        # Open the interactive plot immediately
        #fig.show()

        # ==================================================
        # Plot 6: C_M_alpha vs CG location for selected wing area distributions
        # ==================================================

        plt.figure(figsize=(9, 6))

        # Select first, third, fifth, seventh, ninth, and eleventh item
        selected_indices = [3, 4, 5, 6, 7]

        # Colour-blind friendly colour cycle
        colors = plt.cm.tab10(np.linspace(0, 1, len(selected_indices)))

        for color, idx in zip(colors, selected_indices):

            S_dist = S_aw_S_tot_ratio[idx]

            # Use normalized CG location instead of absolute CG location
            x_cg_values = CG_envelope

            CM_alpha_values = C_M_alpha_arr[idx, :]

            plt.plot(
                x_cg_values,
                CM_alpha_values,
                marker="o",
                linestyle="-",
                color=color,
                label=rf"$S_{{aw}}/S_{{tot}} = {S_dist:.2f}$"
            )

        # Add horizontal stability boundary
        plt.axhline(
            y=0,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=r"$C_{M_\alpha}=0$"
        )

        plt.xlabel(r"$x_{cg}/MAC_{fw}$ [-]")
        plt.ylabel(r"$C_{M_\alpha}$ [1/rad]")
        plt.title(r"$C_{M_\alpha}$ vs Normalized CG Location for Selected Wing Area Distributions")

        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            plot_dir / "cg_norm_vs_C_M_alpha.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


if __name__ == '__main__':
    main()

            
        
