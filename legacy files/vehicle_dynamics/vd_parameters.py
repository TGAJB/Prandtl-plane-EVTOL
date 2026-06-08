# This file contains all the independent variables that describe the aircraft in variable form.

from dataclasses import dataclass, field


@dataclass
class Mission:
    """
    Mission-level independent requirements relevant to the Vehicle Dynamics model.
    """

    transition_speed:    float = None  # [m/s]
    cruise_speed:        float = 200/3.6  # [m/s]
    wind_gust_speed:     float = None  # [m/s]
    
    transition_altitude: float = None  # [m]
    cruise_altitude:     float = None  # [m]

    rho_cr:              float = None  # [kg/(m^3)]
    rho_SL:              float = 1.225  # [kg/(m^3)]

    minimum_range:       float = None  # [m]


@dataclass
class WingGeometry:
    """
    Geometric layout of the horizontal lifting surfaces.

    For the Prandtl-plane / box-wing configuration, these parameters describe
    the position, size, and orientation of the front and aft horizontal wings.
    """

    design_point:         int   = 760   # [N/m^2]

    S_fw:                 float = None  # [m^2]
    S_aw:                 float = None  # [m^2]
    S_e_fw:               float = None  # [m^2]
    S_e_aw:               float = None  # [m^2]
    S_tot:                float = None  # [m^2]

    
    b_fw:                 float = None  # [m]
    b_aw:                 float = None  # [m]

    A_fw:                 float = None  # [-]
    A_aw:                 float = None  # [-]

    gap:                  float = None  # [m]
    stagger:              float = None  # [m]

    MAC_fw:               float = None  # [m]
    MAC_aw:               float = None  # [m]

    taper_fw:             float = None  # [-]
    taper_aw:             float = None  # [-]

    chord_fw_root:        float = None  # [m]
    chord_fw_tip:         float = None  # [m]
    chord_aw_root:        float = None  # [m]
    chord_aw_tip:         float = None  # [m]

    LE_sweep_fw:          float = None  # [deg.]
    LE_sweep_aw:          float = None  # [deg.]

    dihedral_front_wing:  float = None  # [deg.]
    dihedral_aft_wing:    float = None  # [deg.]

    twist_fw:             float = None  # [deg.]
    twist_aw:             float = None  # [deg.]

    incidence_fw:         float = None  # [deg.]
    incidence_aw:         float = None  # [deg.]

    airfoil_fw:           str   = None  # [-]
    airfoil_aw:           str   = None  # [-]

    # ===== ADDED FOR DATCOM ==================================================
    # Reference quantities for non-dimensionalisation. EVERY aircraft-level derivative is referenced to these; they must match the EOM reference.
    S_ref:                float = None  # [m^2]
    b_ref:                float = None  # [m]
    MAC_ref:              float = None  # [m]

    # Exposed planform areas (fuselage carry-through removed) - interference terms
    S_exposed_fw:         float = None  # [m^2]
    S_exposed_aw:         float = None  # [m^2]

    # Airfoil thickness and trailing-edge angle.
    # NOTE: the front/aft WING lift-curve slopes now use the aero department's
    # section slope (cl_alpha_fw / cl_alpha_aw) directly, so these are no longer
    # consumed by the lift methods. Retained as geometric descriptors only.
    t_c_fw:               float = None  # [-]
    t_c_aw:               float = None  # [-]
    te_angle_fw:          float = None  # [deg.]
    te_angle_aw:          float = None  # [deg.]

    # Wing vertical position relative to body centreline (+ is down)
    z_w_fw:               float = None  # [m]
    z_w_aw:               float = None  # [m]



@dataclass
class TailGeometry:
    """
    Geometry and position of the vertical tail.
    """

    x_vert_tail:                float = None  # [m]
    S_vert_tail:                float = None  # [m^2]
    b_vert_tail:                float = None  # [m]
    AR_vert_tail:               float = None  # [-]
    LE_sweep_vert_tail:         float = None  # [rad]
    airfoil_vert_tail:          str   = None  # [-]

    taper_vert_tail:            float = None  # [-]
    MAC_vert_tail:              float = None  # [m]
    t_c_vert_tail:              float = None  # [-]
    te_angle_vert_tail:         float = None  # [deg.]
    z_vert_tail:                float = None  # [m] vertical a.c. height (datum)


@dataclass
class FuselageGeometry:
    """
    Fuselage geometry relevant to aerodynamics and vehicle dynamics.
    """

    fuselage_length:   float = None  # [m]
    d_fw:              float = None  # [m] - Fuselage is modelled as a tube for now
    d_aw:              float = 0     # [m]
    x_ac_fuselage:     float = None  # [m]

    side_area:           float = None  # [m^2] projected side area S_Bs (Cn_beta)
    base_area:           float = None  # [m^2] reference/base area S_B0 (CY_beta body)
    body_depth_at_wing:  float = None  # [m]   d at the wing (sidewash); ~ diameter


@dataclass
class ControlSurfaceGeometry:
    """
    Control-surface geometry needed for the deflection (control) derivatives.
    Chord ratios feed the section-effectiveness charts (Sec 6.1.1.1); the span
    factors feed K_b / strip integration.
    """
    elevator_cf_c:       float = None   # [-] flap-chord / wing-chord ratio
    elevator_Kb:         float = None   # [-] flap-span factor (Fig 6.1.4.1)
    elevator_on_surface: str   = "aw"   # which wing carries the elevator ('fw' or 'aw')

    aileron_cf_c:        float = None   # [-]
    aileron_eta_inner:   float = None   # [-] inboard span station
    aileron_eta_outer:   float = None   # [-] outboard span station

    rudder_cf_c:         float = None   # [-]


@dataclass
class MassProperties:
    """
    Aircraft mass and inertia properties.
    """

    mtow:         float = None  # [kg]
    oew:          float = None  # [kg]
    payload_mass: int   = 400  # [kg]

    x_cg_min:     float = None  # [m]
    x_cg_max:     float = None  # [m]
    x_cg_opt:     float = None  # [m] - Optimal CG location during cruise
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
    cl_alpha_fw:                        float = None  # [1/deg.]
    cl_alpha_aw:                        float = None  # [1/deg.]
    cl_alpha_vert_tail:                 float = None  # [1/deg.]

    # Lift curve
    CL_alpha_fw:                        float = None  # [1/rad.]
    CL_alpha_aw:                        float = None  # [1/rad.]
    CL_alpha_vert_tail:                 float = None  # [1/rad.]

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
    e_hor_wings:                         float = None  # [-] - Oswald efficiency factor (DOES THE VALIDITY CHANGE WHEN HLDs ARE DEPLOYED?)
    e_vert_tail:                         float = None  # [-]
    e_winglet:                           float = None  # [-]

    # Aerodynamic moments
    C_M_ac_fw:                           float = None  # [-]
    C_M_ac_aw:                           float = None  # [-]
    C_M_ac_fuselage:                     float = None  # [-]

    # Downwash gradients
    downwash_gradient_fw_to_aw:         float = None  # [-]
    sidewash_gradient_fuselage_to_tail: float = None  # [-]

    # Dynamic pressure ratios
    dyn_pres_ratio_fw_to_aw:            float = 0.9  # [-]
    dyn_pres_ratio_fuselage_to_tail:    float = None  # [-]

    # Flow speed ratios
    flow_speed_ratio_fw_to_aw:          float = None  # [-]
    flow_speed_ratio_fuselage_to_tail:  float = None  # [-]

    # Interference coefficients
    I_v:                                float = None #  [-] - Vortex interference factor

    # Long. positions of aerodynamic centres
    x_ac_fw_cruise:                     float = None  # [m]
    x_ac_aw_cruise:                     float = None  # [m]
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
    fuselage_geometry: FuselageGeometry          = field(default_factory=FuselageGeometry)
    control_surfaces:  ControlSurfaceGeometry    = field(default_factory=ControlSurfaceGeometry)
    mass:              MassProperties            = field(default_factory=MassProperties)
    aerodynamics:      AerodynamicCoefficients   = field(default_factory=AerodynamicCoefficients)
    stability:         StabilityDerivatives      = field(default_factory=StabilityDerivatives)
    controls:          ControlDerivatives        = field(default_factory=ControlDerivatives)
    propulsion:        Propulsion                = field(default_factory=Propulsion)
    structures:        StructuralLimits          = field(default_factory=StructuralLimits)

# testing