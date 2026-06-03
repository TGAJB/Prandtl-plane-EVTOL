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

    S_fw:                 float = None  # [m^2]
    S_aw:                 float = None  # [m^2]
    S_tot:                float = None  # [m^2]
    
    b_fw:                 float = None  # [m]
    b_aw:                 float = None  # [m]

    gap:                  float = None  # [m]
    stagger:              float = None  # [m]

    MAC_fw:               float = None  # [m]
    MAC_aw:               float = None  # [m]

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


@dataclass
class FuselageGeometry:
    """
    Fuselage geometry relevant to aerodynamics and vehicle dynamics.
    """

    fuselage_length:   float = None  # [m]
    fuselage_diameter: float = None  # [m]
    x_ac_fuselage:     float = None  # [m]


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

    I_xx:         float = None  # [kg m^2]
    I_yy:         float = None  # [kg m^2]
    I_zz:         float = None  # [kg m^2]
    I_xz:         float = None  # [kg m^2]


@dataclass
class AerodynamicCoefficients:
    """
    Aerodynamic characteristics required by the Vehicle Dynamics model.
    """

    # Lift curve
    CL_alpha_fw:                        float = None  # [1/deg.]
    CL_alpha_aw:                        float = None  # [1/deg.]
    CL_alpha_vert_tail:                 float = None  # [1/deg.]

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

    # Velocity ratios
    flow_speed_ratio_fw_to_aw:          float = None  # [-]
    flow_speed_ratio_fuselage_to_tail:  float = None  # [-]

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

    # Sideslip
    C_Y_beta:  float = None  # [1/rad]
    C_L_beta:  float = None  # [1/rad]
    C_N_beta:  float = None  # [1/rad]

    # Roll and yaw
    C_L_p:     float = None  # [1/rad]
    C_L_r:     float = None  # [1/rad]
    C_M_q:     float = None  # [1/rad]
    C_N_p:     float = None  # [1/rad]
    C_N_r:     float = None  # [1/rad]


@dataclass
class ControlDerivatives:
    """
    Control-surface deflection derivatives.

    These represent control effectiveness in terms of force or moment
    coefficient change per control-surface deflection.
    """

    C_L_delta_e: float = None  # [1/rad]
    C_M_delta_e: float = None  # [1/rad]

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
    mass:              MassProperties            = field(default_factory=MassProperties)
    aerodynamics:      AerodynamicCoefficients   = field(default_factory=AerodynamicCoefficients)
    stability:         StabilityDerivatives      = field(default_factory=StabilityDerivatives)
    controls:          ControlDerivatives        = field(default_factory=ControlDerivatives)
    propulsion:        Propulsion                = field(default_factory=Propulsion)
    structures:        StructuralLimits          = field(default_factory=StructuralLimits)