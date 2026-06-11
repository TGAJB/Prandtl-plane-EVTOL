# This file contains all the independent variables that describe the aircraft in variable form.

import sys
from pathlib import Path

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Single source of truth for MTOW.
#
# MTOW is no longer a hardcoded constant. It is the CONVERGED value produced by
# the Class-II sizing converger (class_II_sizing/mtow_sizing.py), the same value
# that ppe.py consumes. We import it once here so that every reader of
# params.mass.mtow automatically sees the converged mass.
#
# Circular-import note: the MTOW chain
#   mtow_sizing -> energy, mass_components, parameters, fusion_geometry
# imports nothing from the vehicle_dynamics package, so this import is safe.
# mtow_sizing also caches its result in module globals, so the convergence runs
# at most once per session.
#
# vd_parameters.py lives in final_characteristics/vehicle_dynamics, so the
# project root is two levels up. We add it to sys.path so "class_II_sizing"
# resolves no matter what the current working directory is.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from class_II_sizing.mtow_sizing import load_final_design_state

_CONVERGED_MTOW = load_final_design_state()["mtow"]  # [kg] converged MTOW


@dataclass
class Mission:
    """
    Mission-level independent requirements relevant to the Vehicle Dynamics model.
    """

    transition_speed:    float = None  # [m/s]
    cruise_speed:        float = 200/3.6  # [m/s]
    M_cr:                float = 0.17   # [-]
    wind_gust_speed:     float = None  # [m/s]
    
    transition_altitude: float = None  # [m]
    cruise_altitude:     float = 3810.0  # [m]

    rho_cr:              float = 0.835679  # [kg/(m^3)]
    rho_SL:              float = 1.225  # [kg/(m^3)]

    T_cr:                float = 263.385 # [K]
    T_SL:                float = 288.15  # [K]

    minimum_range:       float = None  # [m]


@dataclass
class WingGeometry:
    """
    Geometric layout of the horizontal lifting surfaces.

    For the Prandtl-plane / box-wing configuration, these parameters describe
    the position, size, and orientation of the front and aft horizontal wings.
    """

    design_point:         int   = 760   # [N/m^2]

    S_fw:                 float = 12.91  # [m^2]
    S_aw:                 float = 12.91  # [m^2]
    S_e_fw:               float = 9.685  # [m^2]
    S_e_aw:               float = 9.685  # [m^2]
    S_tot:                float = 25.82  # [m^2]

    
    b_fw:                 float = 13.0  # [m]
    b_aw:                 float = 13.0  # [m]

    A_fw:                 float = b_fw**2/S_fw  # [-]
    A_aw:                 float = b_aw**2/S_aw  # [-]

    gap:                  float = 2.1  # [m]
    stagger:              float = 5.0   # [m]

    MAC_fw:               float = 1.262  # [m]
    MAC_aw:               float = 1.262  # [m]

    taper_fw:             float = 0.4  # [-]
    taper_aw:             float = 0.4  # [-]

    chord_fw_root:        float = 1.66  # [m]
    chord_fw_tip:         float = 0.75  # [m]
    chord_aw_root:        float = 1.66  # [m]
    chord_aw_tip:         float = 0.75  # [m]

    LE_sweep_fw:          float = 0  # [deg.]
    LE_sweep_aw:          float = 0  # [deg.]

    dihedral_front_wing:  float = 1.0  # [deg.]
    dihedral_aft_wing:    float = 1.0  # [deg.]

    twist_fw:             float = 0  # [deg.]
    twist_aw:             float = 0  # [deg.]

    incidence_fw:         float = None  # [deg.]
    incidence_aw:         float = None  # [deg.]

    airfoil_fw:           str   = "NASA LANGLEY LS(1)-0417"  # [-]
    airfoil_aw:           str   = "NASA LANGLEY LS(1)-0417"  # [-]

    # ===== ADDED FOR DATCOM ==================================================
    # Reference quantities for non-dimensionalisation. EVERY aircraft-level derivative is referenced to these; they must match the EOM reference.
    S_ref:                float = None  # [m^2]
    b_ref:                float = 13.0  # [m]
    MAC_ref:              float = 1.262  # [m]

    # Airfoil thickness and trailing-edge angle.
    # NOTE: the front/aft WING lift-curve slopes now use the aero department's
    # section slope (cl_alpha_fw / cl_alpha_aw) directly, so these are no longer
    # consumed by the lift methods. Retained as geometric descriptors only.
    t_c_fw:               float = 0.17  # [-]
    t_c_aw:               float = 0.17  # [-]
    te_angle_fw:          float = 18  # [deg.]
    te_angle_aw:          float = 18  # [deg.]

    # Wing vertical position relative to body centreline (+ is down)
    z_w_fw:               float = -0.5  # [m]
    z_w_aw:               float = 2.1   # [m]

    x_LEMAC_fw:           float = 1 # [m] - very rough estimate


@dataclass
class TailGeometry:
    """
    Geometry and position of the vertical tail.
    """

    x_vert_tail:                float = 8.5  # [m]
    c_r_vert_tail:              float = 1.4  # [m]
    c_t_vert_tail:              float = 0.5  # [m]
    b_vert_tail:                float = 2.5  # [m]
    S_vert_tail:                float = ((c_r_vert_tail + c_t_vert_tail)*b_vert_tail)/2  # [m^2]
    AR_vert_tail:               float = b_vert_tail**2/S_vert_tail  # [-]
    LE_sweep_vert_tail:         float = 0.31  # [rad]
    airfoil_vert_tail:          str   = "NACA 0012"  # [-]

    taper_vert_tail:            float = 0.33  # [-]
    MAC_vert_tail:              float = 0.867  # [m]
    t_c_vert_tail:              float = 0.12  # [-]
    te_angle_vert_tail:         float = 14  # [deg.]
    z_vert_tail:                float = 1.0  # [m] vertical a.c. height (datum)


@dataclass
class WingletGeometry:
    """
    Geometry of the Prandtl-plane vertical joiners / winglets.

    These are not treated as conventional aft vertical tails. They are modelled
    as vertical side-force-producing panels at the wing tips, with their own
    effective aspect-ratio and local-flow corrections. Set enabled=True only
    after the geometry and chart/correction inputs have been supplied.
    """

    enabled:                    bool  = True  # keep existing model backward-compatible
    n_winglets:                 int   = 2      # usually left and right tip joiners

    S_winglet:                  float = 3.125  # [m^2] planform area of ONE winglet/joiner
    b_winglet:                  float = 2.5  # [m] vertical span/height of ONE winglet
    AR_winglet:                 float = b_winglet**2/S_winglet  # [-] aspect ratio of ONE winglet
    taper_winglet:              float = 1.0 # [-]
    LE_sweep_winglet:           float = 0.5  # [rad]

    x_ac_winglet:               float = 3.6  # [m] longitudinal aerodynamic-centre location
    z_ac_winglet:               float = 0.0  # [m] vertical aerodynamic-centre location

    airfoil_winglet:            str   = "NASA LANGLEY LS(1)-0417"  # [-]
    t_c_winglet:                float = 0.12  # [-]
    te_angle_winglet:           float = 10  # [deg.]


@dataclass
class FuselageGeometry:
    """
    Fuselage geometry relevant to aerodynamics and vehicle dynamics.
    """

    fuselage_length:   float = 10.0  # [m]
    d_fw:              float = 2.0  # [m] - Fuselage is modelled as a tube for now
    d_aw:              float = 0     # [m]
    x_ac_fuselage:     float = None  # [m]

    side_area:           float = 20.0 + TailGeometry.S_vert_tail  # [m^2] projected side area S_Bs (Cn_beta)
    base_area:           float = 20.0  # [m^2] reference/base area S_B0 (CY_beta body)
    body_depth_at_wing:  float = 2.0  # [m]   d at the wing (sidewash); ~ diameter


@dataclass
class ControlSurfaceGeometry:
    """
    Control-surface geometry needed for the deflection (control) derivatives.
    Chord ratios feed the section-effectiveness charts (Sec 6.1.1.1); the span
    factors feed K_b / strip integration.
    """
    elevator_cf_c:       float = 0.3   # [-] flap-chord / wing-chord ratio
    elevator_Kb:         float = 0.46   # [-] flap-span factor (Fig 6.1.4.1)
    elevator_on_surface: str   = "aw"   # which wing carries the elevator ('fw' or 'aw')

    aileron_cf_c:        float = 0.3   # [-]
    aileron_eta_inner:   float = 0.35   # [-] inboard span station
    aileron_eta_outer:   float = 0.77   # [-] outboard span station

    rudder_cf_c:         float = 0.2   # [-]


@dataclass
class MassProperties:
    """
    Aircraft mass and inertia properties.
    """

    mtow:         float = field(default_factory=lambda: _CONVERGED_MTOW)  # [kg] converged value from mtow_sizing.py
    oew:          float = None  # [kg]
    payload_mass: int   = 400  # [kg]

    x_cg_min:     float = None  # [m]
    x_cg_max:     float = None  # [m]
    x_cg_opt:     float = 2.8  # [m] - Optimal CG location during cruise
    z_cg:         float = None  # [m] - vertical CG (datum), used by moment arms

    I_xx:         float = None  # [kg m^2]
    I_yy:         float = None  # [kg m^2]
    I_zz:         float = None  # [kg m^2]
    I_xz:         float = None  # [kg m^2]


@dataclass
class AerodynamicCoefficients:
    """
    Aerodynamic characteristics required by the Vehicle Dynamics model.
    """
    # Airfoil curve
    cl_alpha_fw:                        float = 0.107  # [1/deg.]
    cl_alpha_aw:                        float = 0.107  # [1/deg.]
    cl_alpha_vert_tail:                 float = None  # [1/deg.]
    cl_alpha_winglet:                   float = None  # [1/deg.]

    # Lift curve
    CL_alpha_fw:                        float = 5.02484  # [1/rad.]
    CL_alpha_aw:                        float = 5.02484  # [1/rad.]
    CL_alpha_vert_tail:                 float = None  # [1/rad.]
    CL_alpha_winglet:                   float = None  # [1/rad.]

    # Maximum lift coefficients
    CL_max_clean:                       float = None  # [-]
    CL_max_hld:                         float = None  # [-]

    # Zero-lift drag coefficients
    CD0_fw_clean:                       float = None  # [-]
    CD0_aw_clean:                       float = None  # [-]
    
    CD0_fw_HLD:                         float = None  # [-]
    CD0_aw_HLD:                         float = None  # [-]   
    
    CD0_vert_tail:                      float = None  # [-]
    CD0_winglet:                        float = None  # [-]
    CD0_fuselage:                       float = None  # [-]

    # Oswald efficiency factors
    e_hor_wings:                         float = 1.34  # [-] - Oswald efficiency factor (DOES THE VALIDITY CHANGE WHEN HLDs ARE DEPLOYED?)
    e_vert_tail:                         float = None  # [-]
    e_winglet:                           float = None  # [-]

    # Aerodynamic moments
    C_M_ac_fw:                           float = -0.118  # [-]
    C_M_ac_aw:                           float = -0.118  # [-]
    C_M_ac_fuselage:                     float = None  # [-]

    # Downwash gradients
    downwash_gradient_fw_to_aw:         float = None  # [-]
    sidewash_gradient_fuselage_to_tail: float = None  # [-]

    # Dynamic pressure ratios
    dyn_pres_ratio_fw_to_aw:            float = 0.9  # [-]
    dyn_pres_ratio_fuselage_to_tail:    float = 0.95  # [-]

    # Flow speed ratios
    flow_speed_ratio_fw_to_aw:          float = None  # [-]
    flow_speed_ratio_fuselage_to_tail:  float = None  # [-]

    # Interference coefficients
    I_v:                                float = -1.75 # [-] - Vortex interference factor

    # Long. positions of aerodynamic centres
    x_ac_fw_cruise:                     float = 0.313  # [m] as seen from the LEMAC of the front wing
    x_ac_aw_cruise:                     float = 0.313  # [m] as seen from the LEMAC of the aft wing
    x_ac_fw_approach:                   float = None  # [m]
    x_ac_aw_approach:                   float = None  # [m]

    # Vert. positions of aerodynamic centres
    z_ac_fw_cruise:                     float = None  # [m]
    z_ac_aw_cruise:                     float = None  # [m]
    z_ac_fw_approach:                   float = None  # [m]
    z_ac_aw_approach:                   float = None  # [m]


@dataclass
class StabilityDerivatives:
    """
    Stability derivatives used in the Vehicle Dynamics model.
    """

    # AoA
    C_L_alpha: float = None  # [1/rad]
    C_M_alpha: float = None  # [1/rad]

    # AoA-rate (lag of downwash)
    C_L_alpha_dot: float = None  # [1/rad]
    C_M_alpha_dot: float = None  # [1/rad]

    # Pitch rate
    C_L_q:     float = None  # [1/rad]
    C_M_q:     float = None  # [1/rad]

    # Sideslip
    C_Y_beta:  float = None  # [1/rad]
    C_L_beta:  float = None  # [1/rad]
    C_N_beta:  float = None  # [1/rad]

    # Sideslip rate
    C_Y_beta_dot: float = None  # [1/rad]
    C_N_beta_dot: float = None  # [1/rad]

    # Roll rate
    C_Y_p:     float = None  # [1/rad]
    C_L_p:     float = None  # [1/rad]
    C_N_p:     float = None  # [1/rad]

    # Yaw rate
    C_Y_r:     float = None  # [1/rad]
    C_L_r:     float = None  # [1/rad]
    C_N_r:     float = None  # [1/rad]

    # X/Z force bridge (symmetric EOM)
    C_X_0:        float = None  # [-]
    C_Z_0:        float = None  # [-]
    C_X_u:        float = None  # [-]
    C_Z_u:        float = None  # [-]
    C_X_alpha:    float = None  # [1/rad]
    C_Z_alpha:    float = None  # [1/rad]
    C_Z_alpha_dot: float = None  # [1/rad]
    C_Z_q:        float = None  # [1/rad]


@dataclass
class ControlDerivatives:
    """
    Control-surface deflection derivatives.

    These represent control effectiveness in terms of force or moment
    coefficient change per control-surface deflection.
    """

    C_L_delta_e: float = None  # [1/rad]
    C_M_delta_e: float = None  # [1/rad]
    C_Z_delta_e: float = None  # [1/rad]
    C_X_delta_e: float = None  # [1/rad]

    C_Y_delta_r: float = None  # [1/rad]
    C_N_delta_r: float = None  # [1/rad]
    C_L_delta_r: float = None  # [1/rad]

    C_L_delta_a: float = None  # [1/rad]
    C_N_delta_a: float = None  # [1/rad]


@dataclass
class Propulsion:
    """
    Propulsion parameters relevant to Vehicle Dynamics.

    Power-required quantities should not be placed here, because they are
    dependent outputs of the aircraft performance model.
    """

    P_max_SL:               float = None  # [W]
    propulsive_efficiency:  float = None  # [-]

    n_engines:              int   = 6  # [-] - Number of engines
    max_thrust_per_engine:  float = None  # [N]

    # Engine locations in CRUISE CONFIGURATION
    x_cr_1:               float = None  # [m]
    y_cr_1:               float = None  # [m]
    z_cr_1:               float = None  # [m]

    x_cr_2:               float = None  # [m]
    y_cr_2:               float = None  # [m]
    z_cr_2:               float = None  # [m]

    x_cr_3:               float = None  # [m]
    y_cr_3:               float = None  # [m]
    z_cr_3:               float = None  # [m]

    x_cr_4:               float = None  # [m]
    y_cr_4:               float = None  # [m]
    z_cr_4:               float = None  # [m]

    x_cr_5:               float = None  # [m]
    y_cr_5:               float = None  # [m]
    z_cr_5:               float = None  # [m]

    x_cr_6:               float = None  # [m]
    y_cr_6:               float = None  # [m]
    z_cr_6:               float = None  # [m]

    # Engine locations in VTOL CONFIGURATION
    x_vtol_1:               float = None  # [m]
    y_vtol_1:               float = None  # [m]
    z_vtol_1:               float = None  # [m]

    x_vtol_2:               float = None  # [m]
    y_vtol_2:               float = None  # [m]
    z_vtol_2:               float = None  # [m]

    x_vtol_3:               float = None  # [m]
    y_vtol_3:               float = None  # [m]
    z_vtol_3:               float = None  # [m]

    x_vtol_4:               float = None  # [m]
    y_vtol_4:               float = None  # [m]
    z_vtol_4:               float = None  # [m]

    x_vtol_5:               float = None  # [m]
    y_vtol_5:               float = None  # [m]
    z_vtol_5:               float = None  # [m]

    x_vtol_6:               float = None  # [m]
    y_vtol_6:               float = None  # [m]
    z_vtol_6:               float = None  # [m]

    # Mass
    propulsion_system_mass: float = None  # [kg] - MASS PER ENGINE


@dataclass
class StructuralLimits:
    """
    Structural limits relevant to Vehicle Dynamics.
    """
    ultimate_load_factor: float = None  # [-]


@dataclass
class AircraftParameters:
    """
    Complete independent parameter sheet for the Vehicle Dynamics aircraft model.
    """

    mission:           Mission                   = field(default_factory=Mission)
    wing_geometry:     WingGeometry              = field(default_factory=WingGeometry)
    tail_geometry:     TailGeometry              = field(default_factory=TailGeometry)
    winglet_geometry:  WingletGeometry           = field(default_factory=WingletGeometry)
    fuselage_geometry: FuselageGeometry          = field(default_factory=FuselageGeometry)
    control_surfaces:  ControlSurfaceGeometry    = field(default_factory=ControlSurfaceGeometry)
    mass:              MassProperties            = field(default_factory=MassProperties)
    aerodynamics:      AerodynamicCoefficients   = field(default_factory=AerodynamicCoefficients)
    stability:         StabilityDerivatives      = field(default_factory=StabilityDerivatives)
    controls:          ControlDerivatives        = field(default_factory=ControlDerivatives)
    propulsion:        Propulsion                = field(default_factory=Propulsion)
    structures:        StructuralLimits          = field(default_factory=StructuralLimits)

# testing