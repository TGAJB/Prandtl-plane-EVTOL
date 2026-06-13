# ============================================================================
# PROJECT PARAMETER SHEET
#
# Part 1 - Vehicle Dynamics parameter sheet (dataclasses, formerly
#          final_characteristics/vehicle_dynamics/vd_parameters.py).
#          For every quantity that appears in both parts, the DATACLASS FIELD
#          is the single source of truth.
# Part 2 - module-level design constants (sizing, structures, propulsion, ...).
#          Shared quantities are assigned FROM the Part 1 dataclasses; edit
#          them there, not here.
#
# Angles in the Part 1 sheet are in DEGREES unless noted; the Part 2 wing
# constants are in radians and converted from Part 1 with math.radians().
# ============================================================================

import math
import numpy as np
from dataclasses import dataclass, field


# ============================================================================
# Part 1 - VEHICLE DYNAMICS PARAMETER SHEET (single source of truth for
#          shared quantities)
# ============================================================================


@dataclass
class Mission:
    """
    Mission-level independent requirements relevant to the Vehicle Dynamics model.
    """

    transition_speed:    float = None      # [m/s]
    cruise_speed:        float = 200/3.6   # [m/s]
    M_cr:                float = 0.17      # [-]
    wind_gust_speed:     float = None      # [m/s]

    transition_altitude: float = 0.0       # [m]   PLACEHOLDER
    cruise_altitude:     float = 3810.0    # [m]   consistent with rho_cr / T_cr below

    rho_cr:              float = 0.835679  # [kg/(m^3)] ISA density at cruise_altitude
    rho_SL:              float = 1.225     # [kg/(m^3)] ISA sea-level density

    T_cr:                float = 263.385   # [K] ISA temperature at cruise_altitude
    T_SL:                float = 288.15    # [K] ISA sea-level temperature

    minimum_range:       float = 200000.0  # [m] design range


@dataclass
class WingGeometry:
    """
    Geometric layout of the horizontal lifting surfaces.

    For the Prandtl-plane / box-wing configuration, these parameters describe
    the position, size, and orientation of the front and aft horizontal wings.
    """

    design_point:         float = 760.0  # [N/m^2] selected wing loading from matching diagram

    S_fw:                 float = 12.91  # [m^2]
    S_aw:                 float = 12.91  # [m^2]
    S_e_fw:               float = 9.685  # [m^2]
    S_e_aw:               float = 9.685  # [m^2]
    S_tot:                float = 25.82  # [m^2]

    b_fw:                 float = 13.0  # [m] span from the footprint constraint
    b_aw:                 float = b_fw  # [m]

    A_fw:                 float = b_fw**2/S_fw  # [-]
    A_aw:                 float = b_aw**2/S_aw  # [-]

    gap:                  float = 2.1  # [m]
    stagger:              float = 5.0  # [m]

    MAC_fw:               float = 1.262  # [m]
    MAC_aw:               float = 1.262  # [m]

    taper_fw:             float = 0.45      # [-] (merged value; consistent with the chords below)
    taper_aw:             float = taper_fw  # [-]

    chord_fw_root:        float = 1.66  # [m]
    chord_fw_tip:         float = 0.75  # [m]
    chord_aw_root:        float = 1.66  # [m]
    chord_aw_tip:         float = 0.75  # [m]

    LE_sweep_fw:          float = 0.0  # [deg.]
    LE_sweep_aw:          float = 0.0  # [deg.]

    dihedral_front_wing:  float = 0.0  # [deg.]
    dihedral_aft_wing:    float = 0.0  # [deg.]

    twist_fw:             float = 3.0  # [deg.] NOT FINAL
    twist_aw:             float = 3.0  # [deg.] NOT FINAL

    incidence_fw:         float = None  # [deg.]
    incidence_aw:         float = None  # [deg.]

    airfoil_fw:           str   = "NASA LANGLEY LS(1)-0417"  # [-]
    airfoil_aw:           str   = "NASA LANGLEY LS(1)-0417"  # [-]

    # ===== ADDED FOR DATCOM ==================================================
    # Reference quantities for non-dimensionalisation. EVERY aircraft-level derivative is referenced to these; they must match the EOM reference.
    S_ref:                float = S_tot  # [m^2]
    b_ref:                float = b_fw  # [m]
    MAC_ref:              float = 1.262  # [m]

    # Airfoil thickness and trailing-edge angle.
    # NOTE: the front/aft WING lift-curve slopes now use the aero department's
    # section slope (cl_alpha_fw / cl_alpha_aw) directly, so these are no longer
    # consumed by the lift methods. Retained as geometric descriptors only.
    t_c_fw:               float = 0.17  # [-]
    t_c_aw:               float = 0.17  # [-]
    te_angle_fw:          float = 18  # [deg.]
    te_angle_aw:          float = 18  # [deg.]

    # Wing vertical position relative to body centreline (z positive up,
    # matching the MMOI layout: front wing low, aft wing one gap above it)
    z_w_fw:               float = -0.5         # [m]
    z_w_aw:               float = z_w_fw + gap  # [m]

    x_LEMAC_fw:           float = 1 # [m] - very rough estimate

    # ===== FOLDED-WING / VTOL LAYOUT ========================================
    fold_hinge_eta_fw:    float = 0.70  # [-] front-wing hinge station / semi-span; update from spanwise hinge layout
    fold_hinge_eta_aw:    float = 0.70  # [-] aft-wing hinge station / semi-span; update from spanwise hinge layout

    x_vtol_fw_fixed:      float = 1.6  # [m] front-wing fixed (unfolded) section CG -> cruise x_ac_fw
    x_vtol_fw_folded:     float = 4.70  # [m] front-wing FOLDED outer-section CG (drawing)
    x_vtol_aw_fixed:      float = 6.65  # [m] aft-wing fixed (unfolded) section CG -> cruise x_ac_aw
    x_vtol_aw_folded:     float = 8.72  # [m] aft-wing FOLDED outer-section CG (drawing)
    x_vtol_tip_plate:     float = 7.27  # [m] winglet / Prandtl tip-joiner CG in VTOL (drawing)

    x_vtol_rotor_fw_in:   float = 2.80  # [m] front inboard rotor pair CG (first prop)
    x_vtol_rotor_fw_out:  float = 4.80  # [m] front outboard rotor pair CG (second prop)
    x_vtol_rotor_rw:      float = 8.00  # [m] rear rotor pair CG (third prop)


@dataclass
class TailGeometry:
    """
    Geometry and position of the vertical tail.

    The Part 2 structural tail constants (TAPER_TAIL, TIP_TO_CHORD, S_TAIL,
    AR_T) are derived from this class; V_ANGLE and X_TAIL remain Part 2-only
    (V-tail mounting angle and CG-station placeholder).
    """

    n_fins:                     float = 2.0  # [-]
    x_vert_tail:                float = 6.4  # [m]
    c_r_vert_tail:              float = 1.6  # [m]
    c_t_vert_tail:              float = 1.3  # [m]
    b_vert_tail:                float = 1.6  # [m]
    S_vert_tail:                float = ((c_r_vert_tail + c_t_vert_tail)*b_vert_tail)/2  # [m^2]
    AR_vert_tail:               float = b_vert_tail**2/S_vert_tail  # [-]
    LE_sweep_vert_tail:         float = 0.31  # [rad]
    airfoil_vert_tail:          str   = "NACA 0012"  # [-]

    taper_vert_tail:            float = c_t_vert_tail/c_r_vert_tail  # [-]
    MAC_vert_tail:              float = (2/3)*c_r_vert_tail*((1 + taper_vert_tail + taper_vert_tail**2)/(1 + taper_vert_tail))  # [m]
    t_c_vert_tail:              float = 0.12  # [-]
    te_angle_vert_tail:         float = 14  # [deg.]
    z_vert_tail:                float = 0.68  # [m] vertical a.c. height (datum)


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

    S_winglet:                  float = 1.57  # [m^2] planform area of ONE winglet/joiner
    b_winglet:                  float = 2.1  # [m] vertical span/height of ONE winglet
    AR_winglet:                 float = b_winglet**2/S_winglet  # [-] aspect ratio of ONE winglet
    taper_winglet:              float = 1.0 # [-]
    LE_sweep_winglet:           float = np.arctan(WingGeometry.stagger/WingGeometry.gap)  # [rad]

    x_ac_winglet:               float = 4.125  # [m] longitudinal aerodynamic-centre location
    z_ac_winglet:               float = 0.68  # [m] vertical aerodynamic-centre location

    airfoil_winglet:            str   = "NASA LANGLEY LS(1)-0417"  # [-]
    t_c_winglet:                float = 0.17  # [-]
    te_angle_winglet:           float = 18  # [deg.]


@dataclass
class FuselageGeometry:
    """
    Fuselage geometry relevant to aerodynamics and vehicle dynamics.
    """

    fuselage_length:   float = 7.0  # [m]
    d_fw:              float = 2.0   # [m] - Fuselage is modelled as a tube for now
    d_aw:              float = 0     # [m]
    x_ac_fuselage:     float = None  # [m]

    base_area:           float = 12.8  # [m^2] reference/base area S_B0 (CY_beta body)
    side_area:           float = base_area + TailGeometry.S_vert_tail  # [m^2] projected side area S_Bs (Cn_beta)
    body_depth_at_wing:  float = d_fw  # [m]   d at the wing (sidewash); ~ diameter


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


def _converged_mtow_default():
    """
    Converged MTOW default for MassProperties.mtow.

    Deferred import on purpose: parameters.py is imported BY
    class_II_sizing/mtow_sizing.py, so importing the converger at module load
    here would be circular. The import only runs when a MassProperties() is
    actually instantiated, by which point mtow_sizing is fully imported and its
    converged state is cached, so this is cheap and safe.
    """
    from class_II_sizing.mtow_sizing import load_final_design_state
    return load_final_design_state()["mtow"]


@dataclass
class MassProperties:
    """
    Aircraft mass and inertia properties.

    mtow now defaults to the converged class-II design state
    (class_II_sizing/mtow_sizing.py), which is authoritative.
    """

    mtow:         float = field(default_factory=_converged_mtow_default)  # [kg] converged value from mtow_sizing.py
    oew:          float = None    # [kg]
    payload_mass: float = 400.0   # [kg] fixed payload (4 pax + luggage)

    x_cg_min:     float = None  # [m]
    x_cg_max:     float = None  # [m]
    x_cg_opt:     float = 3.311   # [m] - Optimal CG location during cruise
    z_cg:         float = -0.162  # [m] - vertical CG (datum), used by moment arms

    I_xx:         float = 11458.0  # [kg m^2]
    I_yy:         float = 9707.0  # [kg m^2]
    I_zz:         float = 18196.0  # [kg m^2]
    I_xz:         float = 1949.0  # [kg m^2]


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
    e_hor_wings:                         float = 1.34  # [-] (updated 11-06-2026) (DOES THE VALIDITY CHANGE WHEN HLDs ARE DEPLOYED?)
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
    x_ac_fw_cruise:                     float = 0.313 # [m] as seen from the LEMAC of the front wing
    x_ac_aw_cruise:                     float = 0.313 # [m] as seen from the LEMAC of the aft wing

    x_ac_fw:                            float = 1.60  # [m]
    x_ac_aw:                            float = 6.65  # [m]

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
    propulsive_efficiency:  float = 0.87  # [-]

    n_engines:              int   = 6  # [-] - Number of engines / rotors
    max_thrust_per_engine:  float = 5500  # [N]

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
    # x-stations from the folding-layout drawing via WingGeometry:
    # engines 1-2 = front inboard pair, 3-4 = front outboard pair, 5-6 = rear
    # pair. y/z remain to be filled from the layout (unchanged arms for now).
    x_vtol_1:               float = WingGeometry.x_vtol_rotor_fw_in   # [m]
    y_vtol_1:               float = None  # [m]
    z_vtol_1:               float = None  # [m]

    x_vtol_2:               float = WingGeometry.x_vtol_rotor_fw_in   # [m]
    y_vtol_2:               float = None  # [m]
    z_vtol_2:               float = None  # [m]

    x_vtol_3:               float = WingGeometry.x_vtol_rotor_fw_out  # [m]
    y_vtol_3:               float = None  # [m]
    z_vtol_3:               float = None  # [m]

    x_vtol_4:               float = WingGeometry.x_vtol_rotor_fw_out  # [m]
    y_vtol_4:               float = None  # [m]
    z_vtol_4:               float = None  # [m]

    x_vtol_5:               float = WingGeometry.x_vtol_rotor_rw      # [m]
    y_vtol_5:               float = None  # [m]
    z_vtol_5:               float = None  # [m]

    x_vtol_6:               float = WingGeometry.x_vtol_rotor_rw      # [m]
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


# ============================================================================
# Part 2 - MODULE-LEVEL DESIGN CONSTANTS
#
# Quantities shared with the VD sheet are assigned from the Part 1 dataclass
# fields; edit those there. Everything else is owned by this part.
# ============================================================================

# Physical constants

G          = 9.81             # [m/s^2]  gravitational acceleration
RHO_ORIGIN = Mission.rho_SL   # [kg/m^3] ISA sea-level air density

# Mission parameters

M_PAYLOAD        = MassProperties.payload_mass  # [kg]  fixed payload (4 pax + luggage)
W_CREW           = 0.0        # [kg]  0 for autonomous; 85 if piloted
RANGE_M          = Mission.minimum_range   # [m]   design range
D_VERT_DESCENT   = 1676.0     # [m]   vertical descent distance
V_CRUISE         = Mission.cruise_speed    # [m/s] cruise speed
H_CRUISE         = Mission.cruise_altitude # [m]   cruise altitude
H_TRANSITION     = Mission.transition_altitude  # [m] transition altitude
RHO_CRUISE       = Mission.rho_cr          # [kg/m^3] ISA density at H_CRUISE
T_CR_ISA         = Mission.T_cr            # [K]   ISA temperature at H_CRUISE
T_SL_ISA         = Mission.T_SL            # [K]   ISA sea-level temperature
M_CR             = Mission.M_cr            # [-]   cruise Mach number
T_CRUISE         = 2194.7     # [s]   cruise segment
T_TAKEOFF        = 5.0        # [s]   takeoff segment
T_CLIMB          = 1221.7     # [s]   climb segment
T_CLIMB_ACC      = 20.0       # [s]   climb acceleration segment
T_DESCENT        = 311.945    # [s]   descent segment
T_VERTICAL_CLIMB = 25.0       # [s]   vertical climb segment
T_LANDING        = 71.47      # [s]   landing segment
T_ELAPSED_VC     = 5.0        # [s]   vertical climb elapsed time
V_AVG_TO         = 2.0        # [m/s] average takeoff vertical speed
V_HOVER          = 3.0        # [m/s] hover climb speed
V_I               = 20.0       # [m/s] initial climbing speed
VS_0             = D_VERT_DESCENT / T_DESCENT # [m/s]

# Material parameters

SIGMA_ALLOW_CFRP = 500e6   # [Pa]   UD CFRP compression allowable, B-basis (MIL-HDBK-17-1F)
RHO_CFRP         = 1550.0  # [kg/m^3] CFRP density
E_CFRP           = 100 * 10 ** 9 # [Pa] CFRP young modulus (TO BE UPDATED)

SIGMA_ALLOW_AL = 260e6   # [Pa]   2024-T3 compression allowable (MMPDS-01)
RHO_AL         = 2700.0  # [kg/m^3] aluminium alloy density
E_AL           = 73.1 * 10 ** 9   # [Pa] aluminium young modulus

# Aerodynamic & propulsive parameters

ETA_CRUISE           = 0.95   # [-]   cruise total efficiency
ETA_POWERTRAIN_HOVER = 0.95   # [-]   hover powertrain efficiency
ETA_CLIMB            = 0.90   # [-]   climb powertrain efficiency
FM                   = 0.73   # [-]   rotor figure of merit
POWER_SAFETY_FACTOR  = 1.4    # [-]   power safety factor
PM                   = 0.5    # [-]   power margin
CL_MAX_OPERATIONAL   = 2.0    # [-]   operational upper lift coefficient limit
CL_PLOT_MIN          = -0.5   # [-]   lower bound for drag-polar plotting
CL_PLOT_MAX          = 2.20   # [-]   upper bound for drag-polar plotting
DRAG_POLAR_N_POINTS  = 100    # [-]   number of points used for drag-polar plots
#   Preliminary values
LD_CRUISE            = 14.7   # [-]   box-wing cruise L/D (preliminary)
CD0                  = 0.0205 # [-]   zero-lift drag coefficient (preliminary)
OSWALD_EFFICIENCY    = AerodynamicCoefficients.e_hor_wings  # [-] Oswald efficiency factor

# Longitudinal static-stability margins (fraction of the global MAC)
# Master values for the two CG-envelope analyses in final_characteristics/
# vehicle_dynamics; edit them here, not in those modules.
CRUISE_STATIC_MARGIN   = 0.05  # [-] cruise aft-CG static margin (stat_long_stab_anal_func.py)
VTOL_OEI_STATIC_MARGIN = 0.05  # [-] VTOL OEI CG-envelope static margin (VTOL_cg_envelope_det.py)

# Penalty applied when the VTOL-OEI c.g. envelope is infeasible (no allowable c.g.
# exists for a propeller-out case): the allowable limits are pushed out of reach
# so both VTOL c.g. requirements fail with a big, finite violation.
VTOL_INFEASIBLE_PENALTY = 1.0e6  # [m] (stability_eval.py)

# Inertia-estimation placeholders (Roskam class-I radii-of-gyration method,
# I = m*(Rbar*L_char/2)^2). Only used as a fallback when MMOI has not filled the
# real inertias; the MMOI build-up (class_II_sizing/MMOI.py) supersedes these.
INERTIA_RBAR_X        = 0.25  # [-] roll-inertia radius factor   (dyn_stab_analysis.py)
INERTIA_RBAR_Y        = 0.35  # [-] pitch-inertia radius factor  (dyn_stab_analysis.py)
INERTIA_RBAR_Z        = 0.40  # [-] yaw-inertia radius factor    (dyn_stab_analysis.py)
INERTIA_XZ_TO_XX_RATIO = 0.05  # [-] I_xz as a fraction of I_xx   (dyn_stab_analysis.py)

# Wing structural parameters

NUMBER_OF_WINGS     = 2       # [-]     e.g. 1 for conventional, 2 for Prandtl/box-wing
AREA_SPLIT          = WingGeometry.S_fw / WingGeometry.S_tot  # [-] fraction of total area assigned to one wing
WING_SPAN           = WingGeometry.b_fw          # [m]     span from the footprint constraint
WING_LOADING_N      = WingGeometry.design_point  # [N/m^2] selected design-point wing loading from matching diagram
TAPER_W             = WingGeometry.taper_fw      # [-]     wing chord taper ratio (c_tip / c_root)
TIP_TO_CHORD_W      = WingGeometry.t_c_fw        # [-]     wing thickness-to-chord ratio
LE_SWEEP_W          = math.radians(WingGeometry.LE_sweep_fw)         # [rad] wing leading edge sweep angle
DIHEDRAL            = math.radians(WingGeometry.dihedral_front_wing) # [rad] wing dihedral angle
TWIST               = math.radians(WingGeometry.twist_fw)            # [rad] wing twist angle NOT FINAL
#    Class I parameters (OUTDATED - CLASS II AVAILABLE)
S_W         = 30.0            # [m^2]   class I total wing reference area
AR_W        = 5.63            # [-]     class I wing aspect ratio
#    Wing material: CFRP (standard for modern eVTOL primary structure)
T_SKIN_MIN_CFRP  = 1.0e-3  # [m]    minimum CFRP skin - 8 plies ?- 0.125 mm prepreg (MIL-HDBK-17-3F)

# Vtail structural parameters
WINGLET_SKIN_THICKNESS = 0.005
WINGLET_BEAM_NUMBER = 1 #The number of beams used for the stress calculations
WINGLET_RIB_THICKNESS = 0.001
WINGLET_SPAR_FLANGE_LENGTH = 0.01
WINGLET_SPAR_FLANGE_THICKNESS = 0.004
WINGLET_SPAR_BEAM_THICKNESS = 0.004

# V-tail structural parameters
# (planform values derived from TailGeometry; V_ANGLE and X_TAIL are the
#  V-tail-specific quantities with no VD counterpart)

V_ANGLE        = 25.0    # [deg]  V-tail dihedral from horizontal
TAPER_TAIL     = TailGeometry.taper_vert_tail  # [-]  chord taper ratio (c_tip / c_root)
TIP_TO_CHORD   = TailGeometry.t_c_vert_tail    # [-]  thickness-to-chord ratio
F_REAR_WING    = 0.50    # [-]    rear Prandtl-wing lift fraction
N_W            = 3.5     # [-]    design limit load factor
T_SKIN_MIN_AL  = 2.0e-3  # [m]    minimum skin gauge (Niu 1988)
STRUCT_SF      = 1.5     # [-]    ultimate safety factor (FAR/CS 25.303)
C_N_TAIL_MAX   = 1.2     # [-]    peak normal force coefficient at max deflection
V_DIVE_FACTOR  = 1.25    # [-]    V_dive / V_cruise (FAR/CS 25.335 lower bound)
S_TAIL      = TailGeometry.S_vert_tail   # [m^2]  tail panel area (per panel, see mass_components.tail_mass)
AR_T        = TailGeometry.AR_vert_tail  # [-]    tail panel aspect ratio

# Propulsion geometry

N_PROP   = Propulsion.n_engines  # [-]  number of rotors
N_MOTOR  = 6      # [-]  number of motors
N_BLADES = 8      # [-]  blades per rotor
D_PROP   = 1.9    # [m]  rotor diameter
A_DISK   = 2.84   # [m^2] rotor disk area per rotor

# Battery -- Amprius SA504 cell-count basis (see battery_cells.py)
# The 20% energy reserve is the sole margin
# allowance is superseded by the reserve + whole-string layout round-up.

E_CELL_WH    = 37.57    # [Wh]  energy per cell (SA504 datasheet)
M_CELL_KG    = 0.0973   # [kg]  cell mass (SA504 datasheet, 97.3 g)
CELL_TO_PACK = 0.72     # [-]   pack mass fraction that is cells
E_AUX_KWH    = 2.79     # [kWh] auxiliary energy (aux_loads.py build-up)
RESERVE_FRAC = 0.20     # [-]   20% energy reserve on (mission + aux)
S_SERIES     = 235      # [-]   cells in series for the 800 V bus
E_CELL_DENS_WH_KG = E_CELL_WH / M_CELL_KG   # = 386 Wh/kg (derived, do not edit)

# Airframe geometry

L_FUS       = FuselageGeometry.fuselage_length  # [m]  fuselage length
FUSE_WIDTH  = FuselageGeometry.d_fw             # [m]  fuselage width (front view; placeholder)
PER_FUS_MAX = 12.0   # [m]   fuselage maximum perimeter
N_PAX       = 4      # [-]   passenger count
FAT_CYLINDER_SECTION = 0.4 #this is for mmoi calculations on what we assume is the cylinder (rest of fus is neglected)

# MMOI component layout (class_II_sizing/MMOI.py)
#
# Datum: nose tip. x positive aft, y positive starboard, z positive up, origin on the
# fuselage centreline. Seeds taken from the Vehicle Dynamics parameter sheet
# (Part 1 above) where available. All coordinates are component-CG
# locations pending a real layout drawing.

# -- Fuselage equivalent cylinder --
D_FUS    = FUSE_WIDTH     # [m]   equivalent-cylinder diameter (circular section assumed)
X_CG_FUS = 0.45 * L_FUS   # [m]   shell CG slightly fwd of mid-length (light tailcone)  PLACEHOLDER
Z_CG_FUS = 0.0            # [m]   shell CG on the centreline

# -- Wings (Prandtl pair) --
X_WING_F     = AerodynamicCoefficients.x_ac_fw                       # [m]  front-wing CG ~ x_LEMAC (vd: 1.0) + 0.4*MAC  PLACEHOLDER
Z_WING_F     = WingGeometry.z_w_fw       # [m]  low-mounted front wing
WING_STAGGER = WingGeometry.stagger      # [m]  longitudinal distance front -> rear wing
H_GAP_WINGS  = WingGeometry.gap          # [m]  vertical gap between wing planes; tip-plate height
X_WING_R     = AerodynamicCoefficients.x_ac_aw   # [m]  rear-wing CG station (derived)
Z_WING_R     = WingGeometry.z_w_aw       # [m]  high rear wing (= z_w_fw + gap)
WINGLET_MASS_FRAC = 0.10                 # [-]  wing-mass fraction carved out for the two
                                         #      vertical tip joiners                      PLACEHOLDER
# -- Winglets --
WINGLET_TAPER_RATIO = 1

# -- Hinge --
HINGE_THETA = -45 # [deg]
HINGE_PHI = 35.26438968 # [deg]
HINGED_WING_LENGTH = 4 # [m] The length of the portion of the wing that is hinged
FW_THRUSTER_POSITION_1 = (0.5, 0, 3) #The position of the thrusters in the forward wing w.r.t the wing axis (talk to Antonio if ur confused)
FW_THRUSTER_POSITION_2 = (0.5, 0, 6)


ETA_FOLD_HINGE_FW = WingGeometry.fold_hinge_eta_fw  # [-] update from spanwise hinge layout if needed
ETA_FOLD_HINGE_RW = WingGeometry.fold_hinge_eta_aw  # [-] update from spanwise hinge layout if needed

X_WING_F_FIXED_VTOL  = WingGeometry.x_vtol_fw_fixed
X_WING_R_FIXED_VTOL  = WingGeometry.x_vtol_aw_fixed
X_WING_F_FOLDED_VTOL = WingGeometry.x_vtol_fw_folded
X_WING_R_FOLDED_VTOL = WingGeometry.x_vtol_aw_folded
X_TIP_PLATE_VTOL     = WingGeometry.x_vtol_tip_plate

X_ROTOR_FW_IN_VTOL  = WingGeometry.x_vtol_rotor_fw_in   # [m]  front inboard pair (first prop)
X_ROTOR_FW_OUT_VTOL = WingGeometry.x_vtol_rotor_fw_out  # [m]  front outboard pair (second prop)
X_ROTOR_RW_VTOL     = WingGeometry.x_vtol_rotor_rw      # [m]  rear pair (third prop)

# -- V-tail --
X_TAIL      = TailGeometry.x_vert_tail   # [m]  panel-pair CG station (kept separate from TailGeometry.x_vert_tail,
                    #      which is the fin a.c. station)                                PLACEHOLDER
Z_TAIL_ROOT = TailGeometry.z_vert_tail   # [m]  panel root height above centreline                            PLACEHOLDER

# -- Rotor / motor stations (4 on the front wing, 2 on the rear wing, symmetric) --
ETA_ROTOR_FW_IN  = 0.30  # [-]  inboard front rotor, fraction of semi-span (tip clears fuselage)  PLACEHOLDER
ETA_ROTOR_FW_OUT = 0.70  # [-]  outboard front rotor (2.6 m spacing > D_PROP, no disc overlap)    PLACEHOLDER
ETA_ROTOR_RW     = 0.50  # [-]  rear rotor, one per side                                          PLACEHOLDER
X_ROTOR_OFFSET   = 0.8   # [m]  pod CG ahead of the wing CG station (pylon + spinner)             PLACEHOLDER
X_ROTOR_FW = X_WING_F - X_ROTOR_OFFSET  # [m]  front-rotor station (derived)
X_ROTOR_RW = X_WING_R - X_ROTOR_OFFSET  # [m]  rear-rotor station (derived)
Z_ROTOR_FW = Z_WING_F                   # [m]  rotors carried at front-wing height
Z_ROTOR_RW = Z_WING_R                   # [m]  rotors carried at rear-wing height

# -- Optimal cruise CG (from the VD sheet) --
X_CG_OPT = MassProperties.x_cg_opt  # [m]  optimal CG location during cruise

# -- Battery (underfloor box) --
L_BATT = 3.0    # [m]  box length ~ cabin floor length                                  PLACEHOLDER
W_BATT = 1.2    # [m]  box width between cabin floor beams                              PLACEHOLDER
H_BATT = 0.25   # [m]  underfloor bay depth                                             PLACEHOLDER
X_BATT = 2.7  # [m]  box mid-length at the cruise-optimal CG so the
                   #      heaviest item is CG-neutral
Z_BATT = -0.7   # [m]  below the cabin floor (floor ~ -0.5 m for the 2.0 m section)

# -- Payload (pax + luggage cabin box) --
X_PAYLOAD     = 2.7  # [m]  pax + luggage centred on the target CG
Z_PAYLOAD     = -0.2   # [m]  seated-occupant CG slightly below centreline
L_PAYLOAD_BOX = 2.0    # [m]  two seat rows                                             PLACEHOLDER
W_PAYLOAD_BOX = 1.4    # [m]  cabin width                                               PLACEHOLDER
H_PAYLOAD_BOX = 1.2    # [m]  seated height                                             PLACEHOLDER

# -- Landing gear (skid rails) --
X_GEAR      = 0.5 * L_FUS                  # [m]  rail mid-length under the cabin
Z_GEAR      = -(FUSE_WIDTH / 2 + 0.30)     # [m]  belly radius + 0.30 m static clearance
# Y_GEAR_RAIL (half-track) is derived from L_ARM further down, after the landing-gear
# trade-study constants that define it.

# -- Misc systems (20% MTOW: wiring, ECS, avionics, furnishings) --
X_MISC = X_CG_FUS   # [m]  smeared through the fuselage -> fuselage CG
Z_MISC = 0.0        # [m]  on the centreline


# Landing-gear drop trade study (CS-27.725 limit + 27.727 reserve)
#
# Skid gear modelled as 2 ground rails joined by 2 transverse cross-members; each
# member has 2 knees -> 4 "legs"/hinges across the 2 skids. Every architecture is an
# energy absorber sized for the two drops (stay ELASTIC at the limit, survive the
# reserve), and all capacities/masses are WHOLE-GEAR totals (per-member values scaled
# by the *_COUNT constants below). Used by mass_components.landing_gear_mass and the
# offline study class_II_sizing/trade_off_landing_gear.py.

# -- Whole-aircraft drop inputs (MTOW is supplied by the caller, not a constant) --
H_L      = 0.25      # [m]  limit drop height (>= 0.20 floor)                  PLACEHOLDER
D_EST    = 0.15      # [m]  estimated impact deflection (gear stroke); used    PLACEHOLDER
                     #       ONLY to size the effective drop mass M_EFF
LIFT     = 0.0       # [-]  rotor/lift credit ratio L (0 = conservative, no credit)
N_LIMIT  = 20.0      # [g]  max allowed peak deceleration                      PLACEHOLDER
ENVELOPE = 0.40      # [m]  available vertical stroke envelope                 PLACEHOLDER

# -- Materials: E [Pa], sy [Pa], rho [kg/m^3], eu [-] ductility, Gc [J/m^2] --
#    (MMPDS / MIL-HDBK-5 for metals; laminate datasheet for CFRP)
AL    = {"E": 71.7e9, "sy": 503e6,  "rho": 2810, "eu": 0.11,  "Gc": 0}
STEEL = {"E": 200e9,  "sy": 1200e6, "rho": 7850, "eu": 0.06,  "Gc": 0}
CFRP  = {"E": 70e9,   "sy": 600e6,  "rho": 1600, "eu": 0.015, "Gc": 1500}

# -- Gear architecture: how many parallel units form the full gear --
N_SKID          = 2      # [-]   longitudinal ground rails
# -- Skid-rail tube geometry (hollow aluminium round tube, density RHO_AL) --
D_SKID_RAIL     = 0.080  # [m]   rail outer diameter
T_SKID_RAIL     = 0.004  # [m]   rail wall thickness
L_SKID_RAIL     = 0.65 * L_FUS  # [m]  rail length ~ ground-contact footprint
CROSSTUBE_COUNT = 2      # [-]   transverse cross-tubes (front + rear)
HINGES_PER_TUBE = 2      # [-]   knees per cross-tube  -> 2*2 = 4 hinges total
LEAF_COUNT      = 2      # [-]   transverse leaf springs
HINGES_PER_LEAF = 2      # [-]
COMPOSITE_COUNT = 2      # [-]
CRUSH_COUNT     = 4      # [-]   one crush unit per leg
ELASTO_COUNT    = 4      # [-]   one elastomeric mount per leg

# -- Structural-model constants --
BC_FACTOR         = 3.0    # [-]   k = BC*EI/L^3 (3 = tip-loaded cantilever)
CROSS_SPAN        = 0.60   # [m]   skid track width under fuselage             PLACEHOLDER
FOOTPRINT_MAX_SPAN = 6.0   # [m]   max lateral gear track (spanwise footprint cap)
TUBE_HINGE_LEN    = 1.0    # [xD]  plastic-hinge length, tube knee
LEAF_HINGE_LEN    = 1.5    # [xt]  plastic-hinge length, leaf
LEAF_DEV_FACTOR   = 2.0    # [-]   developed leaf length = factor * L
COMP_CRUSH_FRAC   = 0.5    # [xL]  post-failure crush travel, composite        PLACEHOLDER
COMP_DELAM_FACTOR = 2.0    # [-]   delaminated length = factor * L
CRUSH_LEAF_B      = 0.08   # [m]   width of the elastic leaf in the hybrid
CRUSH_LEAF_L      = 0.45   # [m]   length of that leaf
HONEYCOMB_STRESS  = 2.5e6  # [Pa]  honeycomb crush plateau stress
HONEYCOMB_DENSITY = 80.0   # [kg/m^3] honeycomb core density
CRUSH_STROKE_EFF  = 0.75   # [-]   usable crush fraction before densification
ELASTO_FIXED_MASS = 1.2    # [kg]  mount housing/bracket fixed mass
ELASTO_MASS_PER_N = 1.0e-4 # [kg/N] mount mass vs peak load
# -- Discrete-mount load-path frame (elastomeric): CROSSTUBE_COUNT cross-tubes + ELASTO_COUNT
#    arms carrying the mounts down to the skids. Hollow aluminium tubes (density RHO_AL). --
D_FRAME_TUBE      = 0.060  # [m]   cross-tube / arm outer diameter
T_FRAME_TUBE      = 0.003  # [m]   cross-tube / arm wall thickness
L_ARM             = 0.45   # [m]   arm length, mount/hinge down to skid

# -- MMOI layout, gear half-track (see "MMOI component layout" section above) --
Y_GEAR_RAIL = (FUSE_WIDTH + 2 * L_ARM) / 2  # [m]  matches the whole_gear track FUSE_WIDTH + 2*L

# -- Trade-off scoring: qualitative scores 0..1 by engineering judgement (PLACEHOLDERS) --
QUAL = {
    "A cross-tube":       dict(tunable=0.3, cert_risk=0.1, cost=0.1),
    "B metal leaf":       dict(tunable=0.5, cert_risk=0.2, cost=0.2),
    "B' composite leaf":  dict(tunable=0.5, cert_risk=0.8, cost=0.5),
    "F crushable hybrid": dict(tunable=0.9, cert_risk=0.5, cost=0.5),
    "E elastomeric":      dict(tunable=0.4, cert_risk=0.4, cost=0.2),
}
# criterion -> (direction, base weight, relative uncertainty for sensitivity)
#   direction "min_ratio" (mass only) is magnitude-aware: score = lightest/value, so a 6x
#   heavier design scores ~0.16 (not just "worst in the feasible set" as plain min-max would).
#   Mass is the dominant weight so a much heavier gear cannot win on the soft criteria.
WEIGHTS = {
    "SEA":       ("max",       0.18, 0.10),
    "mass":      ("min_ratio", 0.45, 0.05),
    "npk":       ("min",       0.11, 0.10),
    "reusable":  ("max",       0.07, 0.00),
    "tunable":   ("max",       0.06, 0.20),
    "cert_risk": ("min",       0.09, 0.20),
    "cost":      ("min",       0.04, 0.20),
}

# -- Per-architecture SLSQP search config (x0 / bounds / varnames / material / scale) --
#    Keys match the concept-function map in mass_components and the QUAL table above.
#
#    The `bounds`/`x0` below are calibrated at GEAR_BOUNDS_REF_MASS. `scale` gives a
#    per-variable exponent so mass_components._scaled_gear_bounds grows each variable's
#    UPPER bound (and seed) by (m_eff / GEAR_BOUNDS_REF_MASS) ** scale_i, keeping the
#    study valid at any MTOW. Scale the ENERGY-bearing, stroke-neutral dimensions ~linearly
#    (leaf width, honeycomb area, elastomer stiffness) so elastic capacity tracks E_L ~ m_eff,
#    while leaving stroke / thickness / dimensionless variables fixed (exponent 0) so the
#    stroke envelope and load-factor cap are not blown. The cross-tube grows only modestly
#    and stays elastic-infeasible at high sink speeds by design (a bending tube is not an
#    elastic spring).
GEAR_OPT_SEED = 0          # [-]  seed for the SLSQP random restarts (reproducible sizing)
GEAR_OPT_RESTARTS = 48     # [-]  random multi-starts per architecture. Marginal designs
                           #       (e.g. a stiff metal leaf near the load-factor cap) sit on
                           #       a small feasible island; too few restarts make feasibility
                           #       and the optimum flicker, so this is set generously.
GEAR_BOUNDS_REF_MASS = 700.0  # [kg] effective mass at which the bounds/x0 below are tuned
GEAR_OPT_BOUNDS = {
    "A cross-tube":      dict(x0=[0.060, 0.004, 0.45],
                              bounds=[(0.03, 0.14), (0.002, 0.014), (0.30, 0.70)],
                              varnames=["D", "t", "L"], material=AL,
                              scale=[0.5, 0.0, 0.0]),
    "B metal leaf":      dict(x0=[0.080, 0.010, 0.45],
                              bounds=[(0.04, 0.18), (0.004, 0.030), (0.30, 0.70)],
                              varnames=["b", "t", "L"], material=STEEL,
                              scale=[1.0, 0.0, 0.0]),
    "B' composite leaf": dict(x0=[0.090, 0.012, 0.45],
                              bounds=[(0.04, 0.20), (0.004, 0.035), (0.30, 0.70)],
                              varnames=["b", "t", "L"], material=CFRP,
                              scale=[1.0, 0.0, 0.0]),
    "F crushable hybrid": dict(x0=[0.008, 0.003, 0.10],
                               bounds=[(0.003, 0.020), (0.0005, 0.012), (0.04, 0.25)],
                               varnames=["tleaf", "Ac", "sc"], material=STEEL,
                               scale=[0.5, 1.0, 0.5]),
    "E elastomeric":     dict(x0=[6e5, 0.08, 0.4],
                              bounds=[(1e5, 3e6), (0.03, 0.20), (0.3, 0.6)],
                              varnames=["kb", "dm", "loss"], material=AL,
                              scale=[1.0, 0.0, 0.0]),
}



# Miscellaneous design constants

gust_speed               = 0      # [m/s]  design gust speed
cabin_noise_req          = 0      # [dB]   cabin noise requirement
propulsion_type          = 0      # [-]    propulsion type flag
contingency_energy_reserve = 0    # [kWh]  energy reserve
vehicle_lifetime         = 0      # [years]
aircraft_person_proximity  = 0    # [m]
aircraft_building_proximity = 0   # [m]
rho_propeller_hub        = 2700.0 # [kg/m^3] hub material density (aluminium)
rho_propeller_blade      = 1550.0 # [kg/m^3] blade material density (CFRP)