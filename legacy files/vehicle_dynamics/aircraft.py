from vd_parameters import AircraftParameters
from dataclasses import dataclass
import numpy as np

# ---------------------------------------------------------------------------
# Constants (frozen; separate from the VD parameter sheet)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Physical:
    deg_per_rad:        float = 57.29577951308232
    gamma:              float = 1.4
    R:                  float = 287.05
    g:                  float = 9.80665
    mu:                 float = 1.789e-5


# ---------------------------------------------------------------------------
# Flight condition: a separate layer, NOT part of the design sheet.
# ---------------------------------------------------------------------------
@dataclass
class FlightCondition:
    mach:               float = None
    rho:                float = None
    tas:                float = None       # true airspeed [m/s]
    alpha:              float = 0.0        # [rad]
    gamma0:             float = 0.0        # [rad]
    config:             str = "cruise"     # selects x_ac_*_cruise vs *_approach

@dataclass
class DatcomChartInputs:
    """
    Empirical factors read from DATCOM design charts (digitise once, enter here).
    Each is tagged with the figure it comes from. These keep the methods
    parametrised rather than hardcoded; supply per surface where noted.
    """
    # Section lift-slope empirical ratio  cl_a / (cl_a)_theory   (Fig 4.1.1.2-8a)
    # fw/aw are no longer used: the wing section slopes come straight from the
    # aero department (cl_alpha_fw / cl_alpha_aw). Only _vt is still consumed,
    # for the vertical tail whose airfoil is not yet selected.
    section_slope_ratio_fw:   float = 1.0   # (unused; retained for reference)
    section_slope_ratio_aw:   float = 1.0   # (unused; retained for reference)
    section_slope_ratio_vt:   float = 1.0
 
    # Vertical-tail effective aspect ratio and other ratios (Figs 5.3.1.1-22a/b)
    vtail_Aeff_A:             float = None
    cyb_v_over_cyb_v_eff:     float = None
    cyb_v_eff:                float = None
 
    # CY_beta body interference K_i (Fig 5.2.1.1-7)
    cyb_Ki:                   float = None
 
    # Cl_beta wing-body chart terms (per wing) — Figs 5.1.2.1-27/28a/28b/29/30a/30b, 5.2.2.1-26
    clb_over_CL_sweep_fw:     float = None  # (Clb/CL)_Lambda_c/2   27
    clb_over_CL_sweep_aw:     float = None
    clb_KM_Lambda_fw:         float = None  # K_M_Lambda            28a
    clb_KM_Lambda_aw:         float = None
    clb_Kf_fw:                float = None  # K_f                   5.2.2.1-26
    clb_Kf_aw:                float = None
    clb_over_CL_AR_fw:        float = None  # (Clb/CL)_A            28b
    clb_over_CL_AR_aw:        float = None
    clb_over_dihedral_fw:     float = None  # Clb/Gamma             29
    clb_over_dihedral_aw:     float = None
    clb_KM_Gamma_fw:          float = None  # K_M_Gamma             30a
    clb_KM_Gamma_aw:          float = None
    clb_twist_fw:             float = None  # dClb/(theta tanLc/4)  30b
    clb_twist_aw:             float = None
 
    # Cn_beta wing-body (Eq 5.2.3.1-a)
    cnb_KN:                   float = None  # Fig 5.2.3.1-8
    cnb_KRl:                  float = None  # Fig 5.2.3.1-9
 
    # Roll damping (Eq 7.1.2.2-a, Fig 7.1.2.2-20)  (beta Clp/kappa)_CL0 per wing
    clp_param_fw:             float = None
    clp_param_aw:             float = None
 
    # Side-force-due-to-roll (Fig 7.1.2.1-9)  (CYp/CL)_CL0,M0  per wing
    cyp_over_CL_fw:           float = None
    cyp_over_CL_aw:           float = None
    K_CYp:                    float = 1.0   # clean-wing factor K
 
    # Rolling-moment-due-to-yaw (Fig 7.1.3.2-10) (Clr/CL)_CL0,M0 per wing
    clr_over_CL_fw:           float = None
    clr_over_CL_aw:           float = None
 
    # Yaw damping wing (Figs 7.1.3.3-6/7)
    cnr_over_CL2:             float = None
    cnr_over_CD0:             float = None
 
    # Section flap effectiveness (Figs 6.1.1.1-39/40) and 3-D ratio (6.1.4.1-14)
    elev_ad_theory:           float = None  # (a_delta)_theory
    elev_ad_ratio:            float = None  # a_delta/(a_delta)_theory
    elev_ad_3D_over_2D:       float = None  # (a_delta)_CL/(a_delta)_cl
    rudder_ad_theory:         float = None
    rudder_ad_ratio:          float = None
    aileron_ad_theory:        float = None
    aileron_ad_ratio:         float = None
    aileron_ad_full_chord:    float = None  # (a_delta)_{cf/c = 1}
    aileron_param:            float = None  # (beta Cl_delta'/kappa)  Fig 6.2.1.1-23
 
    # Sideslip-acceleration sidewash parts (Eq 7.4.4.4-b; Figs 7.4.4.4-6/22/26/42)
    sigma_beta_alpha:         float = None
    sigma_beta_gamma:         float = None
    sigma_beta_theta:         float = None
    sigma_beta_WB:            float = None


# ---------------------------------------------------------------------------
# Aircraft: holds the parameter sheet + the DATCOM methods.
# ---------------------------------------------------------------------------
class Aircraft:
    def __init__(self, params: AircraftParameters, physical: Physical, fc: FlightCondition, charts: DatcomChartInputs):
        self.params = params
        self.physical = physical
        self.fc = fc
        self.charts = charts

    def _require(self, value, name):
        """
        Helper method to check for missing DATCOM inputs.
        """
        if value is None:
            raise ValueError(f"DATCOM input '{name}' is None — add it to the sheet.")
        return value

    def beta(self):
        return np.sqrt(1 - self.fc.mach**2)
    
    def _ref(self):
        """Reference (S_ref, b_ref, MAC_ref); falls back to S_tot if S_ref unset."""
        wg = self.params.wing_geometry
        S_ref = wg.S_ref if wg.S_ref is not None else wg.S_tot
        b_ref = wg.b_ref if wg.b_ref is not None else wg.b_aw
        c_ref = wg.MAC_ref if wg.MAC_ref is not None else wg.MAC_fw
        return (self._require(S_ref, "wing_geometry.S_ref/S_tot"),
                self._require(b_ref, "wing_geometry.b_ref"),
                self._require(c_ref, "wing_geometry.MAC_ref"))    
    
    @staticmethod
    def _le_to_c2(le_sweep, A, taper):
        return np.arctan(np.tan(le_sweep) - (2/A)*((1 - taper)/(1 + taper)))

    @staticmethod
    def _le_to_c4(le_sweep, A, taper):
        return np.arctan(np.tan(le_sweep) - (1/A)*((1 - taper)/(1 + taper)))

    @staticmethod
    def _c4_to_c2(sweep_c4, A, taper):
        # quarter-chord -> mid-chord sweep (Sec 2.2.2)
        return np.arctan(np.tan(sweep_c4) - (1/A)*((1 - taper)/(1 + taper)))

    def _surface_lift_slope(self, A, sweep_c2, kappa):
        """
        Lifting-surface lift-curve slope (per rad). DATCOM 4.1.3.2 / chart 4.1.3.2-49.
        Shared kernel used by the wing methods and CL_alpha_vtail.
        """
        b = self.beta()
        return (2*np.pi*A) / (2 + np.sqrt(4 + ((A**2)*(b**2)/(kappa**2))
                                          * (1 + (np.tan(sweep_c2)**2)/(b**2))))

    def _reynolds(self, length):
        return self.fc.rho * self.fc.tas * length / self.physical.mu

    def _cfg(self, cruise_val, approach_val):
        """Pick cruise vs approach value per FlightCondition.config."""
        return cruise_val if self.fc.config == "cruise" else approach_val

    def x_ac_fw(self):
        ac = self.params.aerodynamics
        return self._require(self._cfg(ac.x_ac_fw_cruise, ac.x_ac_fw_approach), "x_ac_fw")
 
    def x_ac_aw(self):
        ac = self.params.aerodynamics
        return self._require(self._cfg(ac.x_ac_aw_cruise, ac.x_ac_aw_approach), "x_ac_aw")
 
    def x_cg(self):
        return self._require(self.params.mass.x_cg_opt, "mass.x_cg_opt")

    
    # =======================================================================
    # DATCOM methods: these are the core of the code, and should be as close to the original DATCOM as possible.
    # =======================================================================
    def wing_interference_factors(self, d, b):
        K_WB = ((d/b) + 1)**2
        return K_WB

    def nose_carryover(self, d, CL_alpha, S_e):
        K_N = (2*np.pi*((d/2)**2))/(CL_alpha*S_e)
        return K_N

    def CL_alpha_section(self, t_c, phi_te, ratio): # Used for the VERTICAL TAIL only
        """
        Section lift curve slope (per radian), estimated from airfoil geometry.
        NOTE: Retained for the vertical tail, whose airfoil is not yet chosen.
        The front/aft wing section slopes are supplied directly by the aero
        department (params.aerodynamics.cl_alpha_fw / cl_alpha_aw, in 1/deg),
        so the wing methods no longer call this.
        """
        cl_alpha_theory = 6.28 + 4.7*t_c*(1 + 0.00375*phi_te)
        return (1.05/self.beta()) * cl_alpha_theory * ratio
    
    def _kappa(self, t_c, phi_te_deg, ratio): # Used for the VERTICAL TAIL only
        return self.CL_alpha_section(t_c, phi_te_deg, ratio) / (2*np.pi/self.beta())

    def _kappa_from_section_slope(self, cl_alpha_per_deg):
        """
        Built from a KNOWN incompressible
        section lift-curve slope (per deg, from the aero department's airfoil).
        With Prandtl-Glauert correction, so beta cancels:
        """
        cl_alpha_rad = self._require(cl_alpha_per_deg, "aerodynamics.cl_alpha_*") * self.physical.deg_per_rad
        return cl_alpha_rad / (2*np.pi)
        
    def CL_alpha_front_wing(self):
        """
        Front wing lift curve slope (per radian).
        Section slope taken from the aero department (cl_alpha_fw, 1/deg).
        """
        wg = self.params.wing_geometry
        ac = self.params.aerodynamics

        S  = self._require(wg.S_fw, "wing_geometry.S_fw")
        b  = self._require(wg.b_fw, "wing_geometry.b_fw")
        LE_sweep_fw = self._require(wg.LE_sweep_fw, "wing_geometry.LE_sweep_fw")

        A  = b**2 / S
        k = self._kappa_from_section_slope(ac.cl_alpha_fw)
        sweep_c2_fw = self._le_to_c2(np.radians(LE_sweep_fw), A, wg.taper_fw)

        return (2*np.pi*A)/(2 + np.sqrt(4 + ((A**2)*(self.beta()**2)/(k**2))*(1 + (np.tan(sweep_c2_fw)**2)/self.beta()**2)))
    
    def CL_alpha_aft_wing(self):
        """
        Aft wing lift curve slope (per radian).
        Section slope taken from the aero department (cl_alpha_aw, 1/deg).
        """
        wg = self.params.wing_geometry
        ac = self.params.aerodynamics

        S  = self._require(wg.S_aw, "wing_geometry.S_aft")
        b  = self._require(wg.b_aw, "wing_geometry.b_aft")
        LE_sweep_aw = self._require(wg.LE_sweep_aw, "wing_geometry.LE_sweep_aft")

        A  = b**2 / S
        k = self._kappa_from_section_slope(ac.cl_alpha_aw)
        sweep_c2_aft = self._le_to_c2(np.radians(LE_sweep_aw), A, wg.taper_aw)

        return (2*np.pi*A)/(2 + np.sqrt(4 + ((A**2)*(self.beta()**2)/(k**2))*(1 + (np.tan(sweep_c2_aft)**2)/self.beta()**2)))
    
    def downwash_gradient(self):
        """
        Downwash gradient at the tail.
        """
        wg = self.params.wing_geometry

        S = self._require(wg.S_fw, "wing_geometry.S_fw")
        b = self._require(wg.b_fw, "wing_geometry.b_fw")
        x_ac_aw = self.x_ac_aw()
        x_cg = self.x_cg()

        l_h = x_ac_aw - x_cg
        A = b**2 / S
        sweep_c4_fw = self._le_to_c4(np.radians(wg.LE_sweep_fw), A, wg.taper_fw)

        K_A = (1/A) - (1/(1 + A**1.7))
        K_lambda = (10 - 3*wg.taper_fw) / 7
        K_H = (1 - (np.abs(wg.gap)/b))/(((2*l_h)/b)**(1/3))

        beta = self.beta()

        return 4.44*(K_A*K_lambda*K_H*(np.sqrt(np.cos(sweep_c4_fw))))**(1.19) * (1/beta)
    
    def CL_alpha_aircraft(self):
        """
        Aircraft lift curve slope (per radian). DATCOM 4.5.1.1-a, referenced to S_ref.
        Exposed areas read from the sheet (S_ex_fw / S_e_aw).
        Aft surface gets the wing-body interference K_WB only -- the nose
        carryover K_N applies to the forward surface alone.
        """
        # VARIABLE DEFINITIONS
        wg = self.params.wing_geometry
        fg = self.params.fuselage_geometry
        ac = self.params.aerodynamics

        S_1 = self._require(wg.S_fw, "wing_geometry.S_fw")
        S_2 = self._require(wg.S_aw, "wing_geometry.S_aw")
        S_e_1 = self._require(wg.S_e_fw, "wing_geometry.S_e_fw")
        S_e_2 = self._require(wg.S_e_aw, "wing_geometry.S_e_aw")
        b_1 = self._require(wg.b_fw, "wing_geometry.b_fw")
        d_1 = self._require(fg.d_fw, "fuselage_geometry.d_fw")
        A_2 = self._require(wg.A_aw, "wing_geometry.A_aw")

        CL_alpha_1 = self._require(ac.CL_alpha_fw, "aerodynamics.CL_alpha_fw")
        CL_alpha_2 = self._require(ac.CL_alpha_aw, "aerodynamics.CL_alpha_aw")
        downwash = self.downwash_gradient()
        q_2_inf_ratio = self._require(ac.dyn_pres_ratio_fw_to_aw, "aerodynamics.dyn_pres_ratio_fw_to_aw")
        I_v = self._require(ac.I_v, "aerodynamics.I_v")

        # COMPUTATIONS
        K_N = (np.pi*d_1**2)/(2*CL_alpha_1*S_e_1)
        wing_body_sum_1 = ((d_1/b_1) + 1)**2
        wing_body_sum_2 = 1
        K_WB_1 = 0.8*(d_1/b_1) + 1

        numer = CL_alpha_1 * (S_e_1/S_1) * CL_alpha_2 * q_2_inf_ratio * K_WB_1 * I_v * (0.5*b_2 - 0.5*d_2)
        denum = 2 * np.pi * A_2 * (0.5*b_1 - 0.5*d_1)
        CL_alpha_W2 = numer / denum

        left_term = (S_1/(S_1+S_2)) * CL_alpha_1 * (K_N + wing_body_sum_1) * (S_e_1/S_1)
        right_term = (S_2/(S_1+S_2)) * CL_alpha_2 * wing_body_sum_2 * q_2_inf_ratio * (S_e_2/S_1) + CL_alpha_W2
        a = left_term + right_term

        return a
    
    
    def CM_alpha_front_wing(self, n): # n is the selected location along the wing chord
        """
        Front wing pitching moment coefficient gradient (per radian).
        """
        wg = self.params.wing_geometry

        x_ac_fw = self.x_ac_fw()
        c_r = self._require(wg.chord_fw_root, "wing_geometry.chord_fw_root")
        MAC = self._require(wg.MAC_fw, "wing_geometry.MAC_fw")

        CL_alpha_fw = self.CL_alpha_front_wing()
        return (n - (x_ac_fw/c_r)) * CL_alpha_fw * (c_r/MAC)
    
    def CM_alpha_aft_wing(self, n): # n is the selected location along the wing chord
        """
        Aft wing pitching moment coefficient gradient (per radian).
        """
        wg = self.params.wing_geometry

        x_ac_aft = self.x_ac_aw()
        c_r = self._require(wg.chord_aw_root, "wing_geometry.chord_aw_root")
        MAC = self._require(wg.MAC_aw, "wing_geometry.MAC_aw")

        CL_alpha_aft = self.CL_alpha_aft_wing()
        return (n - (x_ac_aft/c_r)) * CL_alpha_aft * (c_r/MAC)
    
    def CM_alpha_aircraft(self):
        """
        Aircraft pitching moment coefficient gradient (per radian).
        DATCOM 4.5.2.1-a, linear small-angle form, referenced to S_ref / c_ref.
        """
        wg = self.params.wing_geometry
        ac = self.params.aerodynamics  

        S_ref, _, c_ref = self._ref()
        S_fw = self._require(wg.S_fw, "wing_geometry.S_fw")
        S_aw = self._require(wg.S_aw, "wing_geometry.S_aft")
        x_ac_fw = self.x_ac_fw()
        x_ac_aw = self.x_ac_aw()
        x_cg = self.x_cg()
        eta = self._require(ac.flow_speed_ratio_fw_to_aw, "aerodynamics.flow_speed_ratio_fw_to_aw")

        downwash = self.downwash_gradient()
        CL_alpha_fw = self.CL_alpha_front_wing()
        CL_alpha_aft = self.CL_alpha_aft_wing()

        front_term = -CL_alpha_fw * ((x_cg - x_ac_fw)/c_ref) * (S_fw/S_ref)
        aft_term = -CL_alpha_aft * ((x_cg - x_ac_aw)/c_ref) * (1 - downwash)*eta*(S_aw/S_ref)

        return front_term + aft_term
    
    def CL_q_front_wing(self):
        """
        Front wing lift coefficient gradient with respect to pitch rate (per rad/s).
        """
        wg = self.params.wing_geometry

        c_ref = self._require(wg.MAC_fw, "wing_geometry.MAC_fw")
        S_ref = self._require(wg.S_tot,   "wing_geometry.S_tot")
        x_cg  = self.x_cg()
        x_ac  = self.x_ac_fw()
        S_fw  = self._require(wg.S_fw, "wing_geometry.S_fw")
        eta   = 1.0                                   # front wing in freestream
        x_bar = (x_ac - x_cg) / c_ref                
        return (0.5 + 2.0*x_bar) * self.CL_alpha_front_wing() * (S_fw/S_ref) * eta

    def CL_q_aft_wing(self):
        """
        Aft wing lift coefficient gradient with respect to pitch rate (per rad/s).
        """
        wg = self.params.wing_geometry
        ac = self.params.aerodynamics  

        c_ref = self._require(wg.MAC_aw, "wing_geometry.MAC_aw")
        S_ref = self._require(wg.S_tot,   "wing_geometry.S_tot")
        x_cg  = self.x_cg()
        x_ac  = self.x_ac_aw()
        S_aw  = self._require(wg.S_aw, "wing_geometry.S_aw")
        eta   = self._require(ac.flow_speed_ratio_fw_to_aw, "aerodynamics.flow_speed_ratio_fw_to_aw")
        x_bar = (x_ac - x_cg) / c_ref             
        return (0.5 + 2.0*x_bar) * self.CL_alpha_aft_wing() * (S_aw/S_ref) * eta

    def CL_q_aircraft(self):
        """
        Aircraft lift coefficient gradient with respect to pitch rate (per rad/s). Box-wing symmetric sum — no extra term.
        """
        return self.CL_q_front_wing() + self.CL_q_aft_wing()
    
    def CM_q_front_wing(self):
        """
        Front wing pitching moment coefficient gradient with respect to pitch rate (per rad/s).
        """
        wg = self.params.wing_geometry

        c_ref = self._require(wg.MAC_fw, "wing_geometry.MAC_fw")
        S_fw = self._require(wg.S_fw,   "wing_geometry.S_fw")
        b_fw = self._require(wg.b_fw,   "wing_geometry.b_fw")
        x_cg  = self.x_cg()
        x_ac  = self.x_ac_fw()

        A = b_fw**2 / S_fw
        sweep_c4_fw = self._le_to_c4(np.radians(wg.LE_sweep_fw), A, wg.taper_fw)
        x_bar = (x_ac - x_cg) / c_ref

        first_term = (A*(0.5*x_bar + 2*(x_bar**2)))/(A + 2*np.cos(sweep_c4_fw))
        second_term = (1/24)*(((A**3)*(np.tan(sweep_c4_fw))**2)/(A + 6*np.cos(sweep_c4_fw)))
        third_term = 1/8

        # Low-speed section slope from the aero department (cl_alpha_fw, 1/deg -> 1/rad).
        # The 1/beta compressibility is applied separately via the (num/den) factor below.
        ac = self.params.aerodynamics
        cla_low = self._require(ac.cl_alpha_fw, "aerodynamics.cl_alpha_fw") * self.physical.deg_per_rad
        CM_q = -0.7*cla_low*np.cos(sweep_c4_fw)*(first_term + second_term + third_term)

        B = np.sqrt(1 - (self.fc.mach**2)*(np.cos(sweep_c4_fw)**2))

        num = (((A**3)*(np.tan(sweep_c4_fw))**2)/(A*B + 6*np.cos(sweep_c4_fw))) + (3/B)
        den = (((A**3)*(np.tan(sweep_c4_fw))**2)/(A + 6*np.cos(sweep_c4_fw))) + 3

        return (num/den)*CM_q
    
    def CM_q_aircraft(self):
        """
        Aircraft pitching moment coefficient gradient with respect to pitch rate (per rad/s).
        """

        wg = self.params.wing_geometry
        ac = self.params.aerodynamics

        S_aw = self._require(wg.S_aw, "wing_geometry.S_aft")
        S_ref, _, c_ref = self._ref()
        x_ac_aw = self.x_ac_aw()
        x_cg = self.x_cg()
        l_h = x_ac_aw - x_cg
        eta   = self._require(ac.flow_speed_ratio_fw_to_aw, "aerodynamics.flow_speed_ratio_fw_to_aw")

        delta = -2*self.CL_alpha_aft_wing()*eta*(S_aw/S_ref)*((l_h/c_ref)**2)
        return self.CM_q_front_wing() + delta
    
    def CL_alpha_dot_aircraft(self):
        """
        C_L_alpha_dot (per rad). DATCOM 7.4.4.1-a (aft surface; WB part ~ 0).
        """
        wg = self.params.wing_geometry
        fg = self.params.fuselage_geometry
        ac = self.params.aerodynamics

        S_ref, _, c_ref = self._ref()
        Se_aw = self._require(wg.S_e_aw, "wing_geometry.S_e_aw")
        d = self._require(fg.fuselage_diameter, "fuselage_geometry.fuselage_diameter")
        eta = self._require(ac.flow_speed_ratio_fw_to_aw, "aerodynamics.flow_speed_ratio_fw_to_aw")
        K_WB = self.wing_interference_factors(d, self._require(wg.b_aw, "wing_geometry.b_aw"))
        eps = self.downwash_gradient()
        arm = (self.x_ac_aw() - self.x_cg()) / c_ref

        return 2.0 * K_WB * (Se_aw/S_ref) * arm * eta * eps * self.CL_alpha_aft_wing()

    def CM_alpha_dot_aircraft(self):
        """
        C_M_alpha_dot (per rad). DATCOM 7.4.4.2-a (arm squared, negative).
        """
        wg = self.params.wing_geometry
        fg = self.params.fuselage_geometry
        ac = self.params.aerodynamics
        
        S_ref, _, c_ref = self._ref()
        Se_aw = self._require(wg.S_e_aw, "wing_geometry.S_e_aw")
        d = self._require(fg.fuselage_diameter, "fuselage_geometry.fuselage_diameter")
        eta = self._require(ac.flow_speed_ratio_fw_to_aw, "aerodynamics.flow_speed_ratio_fw_to_aw")
        K_WB = self.wing_interference_factors(d, self._require(wg.b_aw, "wing_geometry.b_aw"))
        eps = self.downwash_gradient()
        arm = (self.x_ac_aw() - self.x_cg()) / c_ref
        return -2.0 * K_WB * (Se_aw/S_ref) * arm**2 * eta * eps * self.CL_alpha_aft_wing()
    
    def CL_trim(self):
        """
        Trim lift coefficient C_W = W/(0.5 rho V^2 S_ref) = C_L0.
        """
        S_ref, _, _ = self._ref()
        m = self._require(self.params.mass.mtow, "mass.mtow")
        q = 0.5 * self._require(self.fc.rho, "fc.rho") * self._require(self.fc.tas, "fc.tas")**2
        return m * self.physical.g / (q * S_ref)
 
    def _aspect_ratio_aircraft(self):
        _, b_ref, _ = self._ref()
        S_ref, _, _ = self._ref()
        return b_ref**2 / S_ref
 
    def CD0_total(self):
        ac = self.params.aerodynamics
        parts = [ac.CD0_fw_clean, ac.CD0_aw_clean, ac.CD0_vert_tail, ac.CD0_winglet, ac.CD0_fuselage]
        return sum(p for p in parts if p is not None)
 
    def CD_trim(self):
        """
        Drag at trim CL: CD0 + CL^2/(pi A e).
        """
        ac = self.params.aerodynamics
        e = self._require(ac.e_hor_wings, "aerodynamics.e_hor_wings")
        CL = self.CL_trim()
        return self.CD0_total() + CL**2/(np.pi*self._aspect_ratio_aircraft()*e)
 
    def CX_0(self):
        return -self.CL_trim() * np.sin(self.fc.gamma0)
 
    def CZ_0(self):
        return -self.CL_trim() * np.cos(self.fc.gamma0)
 
    def CZ_alpha(self):
        return -(self.CL_alpha_aircraft() + self.CD_trim())
 
    def CX_alpha(self):
        ac = self.params.aerodynamics
        e = self._require(ac.e_hor_wings, "aerodynamics.e_hor_wings")
        return self.CL_trim() * (1 - 2*self.CL_alpha_aircraft()/(np.pi*self._aspect_ratio_aircraft()*e))
 
    def CZ_alpha_dot(self):
        return -self.CL_alpha_dot_aircraft()
 
    def CZ_q(self):
        return -self.CL_q_aircraft()
 
    def CX_u(self):
        """
        -2 CD - M dCD/dM ; for clean/prop subsonic dCD/dM ~ 0.
        """
        return -2.0 * self.CD_trim()
 
    def CZ_u(self):
        """
        -CL (2 - M^2)/(1 - M^2)  (Prandtl-Glauert).
        """
        M = self.fc.mach
        return -self.CL_trim() * (2 - M**2)/(1 - M**2)
 
    def section_flap_effectiveness(self, ad_theory, ad_ratio): # Placeholders for now
        """
        alpha_delta = (a_d/(a_d)theory)(a_d)theory  (§6.1.1.1).
        """
        return self._require(ad_ratio, "charts.ad_ratio") * self._require(ad_theory, "charts.ad_theory")
 
    def CL_delta_e(self):
        """
        Elevator lift power referenced to aircraft: (CL_a)_aw eta_aw (S_aw/S_ref) a_d3D K_b.
        """
        wg, ac, cs, ch = (self.params.wing_geometry, self.params.aerodynamics,
                          self.params.control_surfaces, self.charts)
        S_ref, _, _ = self._ref()
        on = cs.elevator_on_surface
        if on == "aw":
            CLa, S, eta = self.CL_alpha_aft_wing(), wg.S_aw, ac.flow_speed_ratio_fw_to_aw
        else:
            CLa, S, eta = self.CL_alpha_front_wing(), wg.S_fw, 1.0
        eta = self._require(eta, "elevator surface eta")
        a_d = self.section_flap_effectiveness(ch.elev_ad_theory, ch.elev_ad_ratio)
        a_d3D = a_d * ch.elev_ad_3D_over_2D
        Kb = self._require(cs.elevator_Kb, "control_surfaces.elevator_Kb")
        return CLa * a_d3D * Kb * eta * (self._require(S, "elevator surface area")/S_ref)
 
    def CZ_delta_e(self):
        return -self.CL_delta_e()
 
    def CM_delta_e(self):
        """
        C_m_delta_e = -CL_delta_e (l_h/c_ref); l_h = CG -> elevator-surface a.c.
        """
        cs = self.params.control_surfaces
        _, _, c_ref = self._ref()
        x_ac = self.x_ac_aw() if cs.elevator_on_surface == "aw" else self.x_ac_fw()
        l_h = (x_ac - self.x_cg())
        return -self.CL_delta_e() * (l_h/c_ref)
 
    def CX_delta_e(self):
        return 0.0
 
    # =======================================================================
    # LATERAL — vertical-tail prerequisites
    # =======================================================================
    def vtail_effective_AR(self):
        """
        A_eff
        """
        tg, ch = self.params.tail_geometry ,self.charts
        A_v = self._require(tg.AR_vert_tail, "tail_geometry.AR_vert_tail")

        return ch.vtail_Aeff_A*A_v
 
    def CL_alpha_vtail(self):
        """
        Vertical-tail lift slope at A_eff (§4.1.3.2).
        """
        tg, ch = self.params.tail_geometry, self.charts

        A = self.vtail_effective_AR()
        taper = self._require(tg.taper_vert_tail, "tail_geometry.taper_vert_tail")
        sweep_c4 = self._le_to_c4(self._require(tg.LE_sweep_vert_tail, "tail_geometry.LE_sweep_vert_tail"), A, taper)
        sweep_c2 = self._c4_to_c2(sweep_c4, A, taper)
        kappa = self._kappa(self._require(tg.t_c_vert_tail, "tail_geometry.t_c_vert_tail"),
                            self._require(tg.te_angle_vert_tail, "tail_geometry.te_angle_vert_tail"),
                            ch.section_slope_ratio_vt)
        return self._surface_lift_slope(A, sweep_c2, kappa)
 
    def sidewash_factor(self):
        """
        (1 + d_sigma/d_beta)(q_v/q_inf)  (Eq 5.4.1-a).
        """
        wg, tg, fg = self.params.wing_geometry, self.params.tail_geometry, self.params.fuselage_geometry
        S_ref, _, _ = self._ref()
        Sv = self._require(tg.S_vert_tail, "tail_geometry.S_vert_tail")
        b = self._require(wg.b_fw, "wing_geometry.b_fw")
        A = b**2 / self._require(wg.S_fw, "wing_geometry.S_fw")
        taper = self._require(wg.taper_fw, "wing_geometry.taper_fw")
        sweep_c4 = self._le_to_c4(np.radians(wg.LE_sweep_fw), A, taper)
        z_w = self._require(wg.z_w_fw, "wing_geometry.z_w_fw")
        d = self._require(fg.body_depth_at_wing, "fuselage_geometry.body_depth_at_wing")

        return (0.724
                + 3.06*(Sv/S_ref)/(1 + np.cos(sweep_c4))
                + 0.4*(z_w/d)
                + 0.009*A)
 
    def dCY_beta_vtail(self):
        """
        (Delta C_Y_beta)_V = -k (CL_a)_V (1+ds/db)(q_v/q_inf)(S_v/S_ref)  (Eq 5.3.1.1-b, per rad).
        """
        tg, wg, ch = self.params.tail_geometry, self.params.wing_geometry, self.charts

        S_v = self._require(tg.S_vert_tail, "tail_geometry.S_vert_tail")
        S_w = (self._require(wg.S_e_fw, "wing_geometry.S_e_fw") + self._require(wg.S_e_aw, "wing_geometry.S_e_aw"))

        return -ch.cyb_v_over_cyb_v_eff*ch.cyb_v_eff*((2*S_v)/S_w)
 
    # ----- vertical-tail moment arms about the CG --------------------------
    def _vtail_arms(self):
        """
        Return (l_p, z_p): longitudinal (+aft of CG) and vertical (+above CG) arms.
        """
        tg, m = self.params.tail_geometry, self.params.mass
        l_p = self._require(tg.x_vert_tail, "tail_geometry.x_vert_tail") - self.x_cg()
        z_cg = m.z_cg if m.z_cg is not None else 0.0
        z_p = self._require(tg.z_vert_tail, "tail_geometry.z_vert_tail") - z_cg

        return l_p, z_p
 
    # =======================================================================
    # LATERAL — side force due to sideslip
    # =======================================================================
    def CY_beta(self):
        """
        C_Y_beta = (C_Y_beta)_WB + (dC_Y_beta)_V  (Eq 5.6.1.1-a, per rad).
        """
        wg, fg, ch = self.params.wing_geometry, self.params.fuselage_geometry, self.charts
        S_ref, _, _ = self._ref()
        
        gam_fw = abs(self._require(wg.dihedral_front_wing, "wing_geometry.dihedral_front_wing"))
        gam_aw = abs(self._require(wg.dihedral_aft_wing, "wing_geometry.dihedral_aft_wing"))

        cyb_w = -0.0001 * (gam_fw + gam_aw) * self.physical.deg_per_rad

        Ki = self._require(ch.cyb_Ki, "charts.cyb_Ki")
        SB0 = self._require(fg.base_area, "fuselage_geometry.base_area")
        cyb_b = -2.0 * Ki * (SB0/S_ref)
        return (cyb_w + cyb_b) + self.dCY_beta_vtail()
 
    # =======================================================================
    # LATERAL — rolling moment due to sideslip
    # =======================================================================
    def _cl_beta_wing(self, which):
        """
        Per-wing wing-body Cl_beta (per DEGREE) for 'fw' or 'aw'  (Eq 5.2.2.1-a).
        """
        wg, ch = self.params.wing_geometry, self.charts
        _, b_ref, _ = self._ref()
        if which == "fw":
            S, b, taper, dih, tw = wg.S_fw, wg.b_fw, wg.taper_fw, wg.dihedral_front_wing, wg.twist_fw
            le, zw = wg.LE_sweep_fw, wg.z_w_fw
            clCL_s, KML, Kf = ch.clb_over_CL_sweep_fw, ch.clb_KM_Lambda_fw, ch.clb_Kf_fw
            clCL_A, clG, KMG = ch.clb_over_CL_AR_fw, ch.clb_over_dihedral_fw, ch.clb_KM_Gamma_fw
            tw_term = ch.clb_twist_fw
        else:
            S, b, taper, dih, tw = wg.S_aw, wg.b_aw, wg.taper_aw, wg.dihedral_aft_wing, wg.twist_aw
            le, zw = wg.LE_sweep_aw, wg.z_w_aw
            clCL_s, KML, Kf = ch.clb_over_CL_sweep_aw, ch.clb_KM_Lambda_aw, ch.clb_Kf_aw
            clCL_A, clG, KMG = ch.clb_over_CL_AR_aw, ch.clb_over_dihedral_aw, ch.clb_KM_Gamma_aw
            tw_term = ch.clb_twist_aw
        A = b**2/S
        d = self._require(self.params.fuselage_geometry.fuselage_diameter, "fuselage_geometry.fuselage_diameter")
        sweep_c4 = self._le_to_c4(np.radians(le), A, taper)
        CL = self.CL_trim()
        Gam = self._require(dih, "dihedral")           
        th = tw if tw is not None else 0.0             
        dClb_dG = -0.0005*np.sqrt(A)*(d/b_ref)**2                                  
        dClb_zw = (1.2*np.sqrt(A)/57.3)*(self._require(zw, "z_w")/b_ref)*(2*d/b_ref)  
        term_CL = CL*((self._require(clCL_s, "clb_over_CL_sweep")*self._require(KML, "clb_KM_Lambda")*self._require(Kf, "clb_Kf"))
                      + self._require(clCL_A, "clb_over_CL_AR"))
        term_G = Gam*(self._require(clG, "clb_over_dihedral")*self._require(KMG, "clb_KM_Gamma") + dClb_dG)
        term_tw = th*np.tan(sweep_c4)*self._require(tw_term, "clb_twist")

        return term_CL + term_G + dClb_zw + term_tw
 
    def Cl_beta(self):
        """
        C_l_beta (per rad). Box-wing: sum fw+aw wing-body parts + vertical tail (Eq 5.6.2.1-a).
        """
        clb_wb_deg = self._cl_beta_wing("fw") + self._cl_beta_wing("aw")
        clb_wb = clb_wb_deg * self.physical.deg_per_rad    
        _, b_ref, _ = self._ref()
        l_p, z_p = self._vtail_arms()
        a = self.fc.alpha
        vt = self.dCY_beta_vtail() * (z_p*np.cos(a) - l_p*np.sin(a))/b_ref

        return clb_wb + vt
 
    # =======================================================================
    # LATERAL — yawing moment due to sideslip
    # =======================================================================
    def Cn_beta(self):
        """
        C_n_beta (per rad). Wing-body (Eq 5.2.3.1-a, per deg) + vertical tail (Eq 5.6.3.1-b).
        """
        fg, ch = self.params.fuselage_geometry, self.charts
        S_ref, b_ref, _ = self._ref()
        KN = self._require(ch.cnb_KN, "charts.cnb_KN")
        KRl = self._require(ch.cnb_KRl, "charts.cnb_KRl")
        SBs = self._require(fg.side_area, "fuselage_geometry.side_area")
        lB = self._require(fg.fuselage_length, "fuselage_geometry.fuselage_length")
        cnb_wb_deg = -KN * KRl * (SBs/S_ref) * (lB/b_ref)
        cnb_wb = cnb_wb_deg * self.physical.deg_per_rad
        l_p, z_p = self._vtail_arms()
        a = self.fc.alpha
        vt = -self.dCY_beta_vtail() * (l_p*np.cos(a) + z_p*np.sin(a))/b_ref
        return cnb_wb + vt
 
    # =======================================================================
    # LATERAL — roll-rate derivatives
    # =======================================================================
    def _cl_p_wing(self, which):
        """
        Per-wing roll damping contribution (Eq 7.1.2.2-a).
        """
        wg, ch = self.params.wing_geometry, self.charts
        _, b_ref, _ = self._ref()
        if which == "fw":
            S, b, taper, le, dih, par, zw = (wg.S_fw, wg.b_fw, wg.taper_fw, wg.LE_sweep_fw,
                                             wg.dihedral_front_wing, ch.clp_param_fw, wg.z_w_fw)
        else:
            S, b, taper, le, dih, par, zw = (wg.S_aw, wg.b_aw, wg.taper_aw, wg.LE_sweep_aw,
                                             wg.dihedral_aft_wing, ch.clp_param_aw, wg.z_w_aw)
        A = b**2/S
        ac = self.params.aerodynamics
        cl_alpha_per_deg = ac.cl_alpha_fw if which == "fw" else ac.cl_alpha_aw
        kappa = self._kappa_from_section_slope(cl_alpha_per_deg)
        beta = self.beta()
        Gam = np.radians(self._require(dih, "dihedral"))
        z = self._require(zw, "z_w")
        dihedral_factor = 1 - 2*(z/(b/2))*np.sin(Gam) + 3*(z/(b/2))**2*np.sin(Gam)**2
        base = self._require(par, "charts.clp_param") * (kappa/beta) * dihedral_factor
        return base * (S/self._ref()[0]) * (b/b_ref)**2  
 
    def Cl_p(self):
        """
        C_l_p (per rad). Box-wing: sum of both wings.
        """
        return self._cl_p_wing("fw") + self._cl_p_wing("aw")
 
    def _cyp_over_cl_M(self, A, sweep_c4, base):
        """
        (CYp/CL)_CL0,M with compressibility (Eq 7.1.2.1-b).
        """
        B = np.sqrt(1 - self.fc.mach**2*np.cos(sweep_c4)**2)
        f = ((A + 4*np.cos(sweep_c4))/(A*B + 4*np.cos(sweep_c4))
             - (A*B + np.cos(sweep_c4))/(A + np.cos(sweep_c4)))
        return f * base
 
    def CY_p(self):
        """
        C_Y_p (per rad). DATCOM 7.1.2.1-a, summed over both wings.
        """
        wg, ch = self.params.wing_geometry, self.charts
        CL = self.CL_trim()
        K = ch.K_CYp
        total = 0.0
        for which in ("fw", "aw"):
            if which == "fw":
                S, b, taper, le, dih, zw, base = (wg.S_fw, wg.b_fw, wg.taper_fw, wg.LE_sweep_fw,
                                                  wg.dihedral_front_wing, wg.z_w_fw, ch.cyp_over_CL_fw)
            else:
                S, b, taper, le, dih, zw, base = (wg.S_aw, wg.b_aw, wg.taper_aw, wg.LE_sweep_aw,
                                                  wg.dihedral_aft_wing, wg.z_w_aw, ch.cyp_over_CL_aw)
            A = b**2/S
            sweep_c4 = self._le_to_c4(np.radians(le), A, taper)
            cyp_cl = self._cyp_over_cl_M(A, sweep_c4, self._require(base, "charts.cyp_over_CL"))
            Gam = np.radians(self._require(dih, "dihedral"))
            z = self._require(zw, "z_w")
            clp_G0 = self._require(ch.clp_param_fw if which == "fw" else ch.clp_param_aw, "clp_param")
            dCYp_G = (3*np.sin(Gam)*(1 - 2*(z/(b/2))*np.sin(Gam))) * clp_G0
            total += (K*(cyp_cl*CL) + dCYp_G) * (S/self._ref()[0])
        return total
 
    def _cnp_over_cl(self, A, sweep_c4, x_bar_over_c):
        """
        (Cnp/CL)_CL0,M with compressibility (Eq 7.1.2.3-b/-c).
        """
        L = sweep_c4
        base = -(1/6)*(A + 6*(A + np.cos(L))*((x_bar_over_c)*(np.tan(L)/A) + np.tan(L)**2/12))/(A + 4*np.cos(L))
        B = np.sqrt(1 - self.fc.mach**2*np.cos(L)**2)
        f = ((A + 4*np.cos(L))/(A*B + 4*np.cos(L))
             - (A*B + 0.5*(A*B + np.cos(L))*np.tan(L)**2)/(A + 0.5*(A + np.cos(L))*np.tan(L)**2))
        return f * base
 
    def Cn_p(self):
        """
        C_n_p (per rad). Clean-wing form + vertical tail (Eq 7.1.2.3-a, §7.4.2).
        """
        wg = self.params.wing_geometry
        _, b_ref, c_ref = self._ref()
        a = self.fc.alpha
        CL = self.CL_trim()
        Clp = self.Cl_p()

        total_wing = 0.0
        for which in ("fw", "aw"):
            if which == "fw":
                S, b, taper, le, x_ac = wg.S_fw, wg.b_fw, wg.taper_fw, wg.LE_sweep_fw, self.x_ac_fw()
            else:
                S, b, taper, le, x_ac = wg.S_aw, wg.b_aw, wg.taper_aw, wg.LE_sweep_aw, self.x_ac_aw()
            A = b**2/S
            sweep_c4 = self._le_to_c4(np.radians(le), A, taper)
            x_bar_c = (x_ac - self.x_cg())/c_ref
            cnp_cl = self._cnp_over_cl(A, sweep_c4, x_bar_c)
            total_wing += cnp_cl*CL*(S/self._ref()[0])
        cnp_wing = total_wing - Clp*np.tan(a)

        l_p, z_p = self._vtail_arms()
        cnp_vt = -(2/b_ref)*(l_p*np.cos(a) + z_p*np.sin(a))*((z_p*np.cos(a) - l_p*np.sin(a))/b_ref)*self.dCY_beta_vtail()
        return cnp_wing + cnp_vt
 
    # =======================================================================
    # LATERAL — yaw-rate derivatives
    # =======================================================================
    def CY_r(self):
        """
        C_Y_r (per rad). Vertical-tail dominated (§7.1.3.1).
        """
        _, b_ref, _ = self._ref()
        l_p, _ = self._vtail_arms()
        return -(2/b_ref) * l_p * self.dCY_beta_vtail()
 
    def _clr_over_cl_M(self, A, sweep_c4, base):
        """
        (Clr/CL)_CL0,M with compressibility (Eq 7.1.3.2-b).
        """
        L = sweep_c4
        B = np.sqrt(1 - self.fc.mach**2*np.cos(L)**2)
        num = 1 + A*(1 - B**2)/(2*B*(A*B + 2*np.cos(L))) + ((A*B + 2*np.cos(L))/(A*B + 4*np.cos(L)))*(np.tan(L)**2/8)
        den = 1 + ((A + 2*np.cos(L))/(A + 4*np.cos(L)))*(np.tan(L)**2/8)
        return (num/den) * base
 
    def Cl_r(self):
        """
        C_l_r (per rad). Clean linear range, summed over both wings (Eq 7.1.3.2-a).
        """
        wg, ch = self.params.wing_geometry, self.charts
        CL = self.CL_trim()
        total = 0.0
        for which in ("fw", "aw"):
            if which == "fw":
                S, b, taper, le, base = wg.S_fw, wg.b_fw, wg.taper_fw, wg.LE_sweep_fw, ch.clr_over_CL_fw
            else:
                S, b, taper, le, base = wg.S_aw, wg.b_aw, wg.taper_aw, wg.LE_sweep_aw, ch.clr_over_CL_aw
            A = b**2/S
            sweep_c4 = self._le_to_c4(np.radians(le), A, taper)
            clr_cl = self._clr_over_cl_M(A, sweep_c4, self._require(base, "charts.clr_over_CL"))
            total += CL*clr_cl*(S/self._ref()[0])
        return total
 
    def Cn_r(self):
        """
        C_n_r (per rad). Wing (Eq 7.1.3.3-a) + vertical tail (Eq 7.4.3.3-a).
        """
        ch = self.charts
        _, b_ref, _ = self._ref()
        CL = self.CL_trim()
        CD0 = self.CD0_total()
        cnr_wing = (self._require(ch.cnr_over_CL2, "charts.cnr_over_CL2")*CL**2
                    + self._require(ch.cnr_over_CD0, "charts.cnr_over_CD0")*CD0)
        l_p, z_p = self._vtail_arms()
        a = self.fc.alpha
        cnr_vt = (2/b_ref**2)*(l_p*np.cos(a) + z_p*np.sin(a))**2*self.dCY_beta_vtail()
        return cnr_wing + cnr_vt
 
    # =======================================================================
    # LATERAL — sideslip-acceleration derivatives
    # =======================================================================
    def CY_beta_dot(self):
        """
        C_Y_beta_dot (per rad). DATCOM 7.4.4.4-a.
        """
        wg, tg, ch = self.params.wing_geometry, self.params.tail_geometry, self.charts
        S_ref, b_ref, _ = self._ref()
        Sv = self._require(tg.S_vert_tail, "tail_geometry.S_vert_tail")
        aF = self.fc.alpha
        Gam = self._require(wg.dihedral_front_wing, "wing_geometry.dihedral_front_wing")
        th = wg.twist_fw if wg.twist_fw is not None else 0.0
        sigma_beta = (self._require(ch.sigma_beta_alpha, "charts.sigma_beta_alpha")*aF
                      + self._require(ch.sigma_beta_gamma, "charts.sigma_beta_gamma")/57.3*Gam
                      - self._require(ch.sigma_beta_theta, "charts.sigma_beta_theta")*np.radians(th)
                      + self._require(ch.sigma_beta_WB, "charts.sigma_beta_WB"))
        l_p, z_p = self._vtail_arms()
        return 2*self.CL_alpha_vtail()*sigma_beta*(Sv/S_ref)*(l_p*np.cos(aF) + z_p*np.sin(aF))/b_ref
 
    def Cn_beta_dot(self):
        """
        C_n_beta_dot (per rad). DATCOM 7.4.4.6-a.
        """
        _, b_ref, _ = self._ref()
        l_p, z_p = self._vtail_arms()
        aF = self.fc.alpha
        return -self.CY_beta_dot()*(l_p*np.cos(aF) + z_p*np.sin(aF))/b_ref
 
    # =======================================================================
    # LATERAL — aileron
    # =======================================================================
    def Cl_delta_a(self):
        """
        Aileron rolling effectiveness C_l_delta_a (per rad). §6.2.1.1.
        """
        wg, cs, ch = self.params.wing_geometry, self.params.control_surfaces, self.charts
        ac = self.params.aerodynamics
        # aileron assumed on the front wing; section slope from the aero department (cl_alpha_fw)
        kappa = self._kappa_from_section_slope(ac.cl_alpha_fw)
        beta = self.beta()
        Cld_prime = self._require(ch.aileron_param, "charts.aileron_param") * (kappa/beta)
        a_d = self.section_flap_effectiveness(ch.aileron_ad_theory, ch.aileron_ad_ratio)
        a_d_full = self._require(ch.aileron_ad_full_chord, "charts.aileron_ad_full_chord")
        return Cld_prime * (a_d/a_d_full)
 
    def Cn_delta_a(self):
        """
        Adverse yaw from aileron — small; left as an input/aero estimate (returns None-safe 0).
        """
        return 0.0
 
    # =======================================================================
    # LATERAL — rudder
    # =======================================================================
    def _rudder_effectiveness(self):
        ch = self.charts
        return self.section_flap_effectiveness(ch.rudder_ad_theory, ch.rudder_ad_ratio)
 
    def CY_delta_r(self):
        tg, ac = self.params.tail_geometry, self.params.aerodynamics
        S_ref, _, _ = self._ref()
        Sv = self._require(tg.S_vert_tail, "tail_geometry.S_vert_tail")
        eta_v = self._require(ac.flow_speed_ratio_fuselage_to_tail, "aerodynamics.flow_speed_ratio_fuselage_to_tail")
        return self.CL_alpha_vtail() * eta_v * (Sv/S_ref) * self._rudder_effectiveness()
 
    def Cn_delta_r(self):
        _, b_ref, _ = self._ref()
        l_p, _ = self._vtail_arms()
        return -self.CY_delta_r() * (l_p/b_ref)
 
    def Cl_delta_r(self):
        _, b_ref, _ = self._ref()
        _, z_p = self._vtail_arms()
        return self.CY_delta_r() * (z_p/b_ref)
 
    # =======================================================================
    # MASS & INERTIA
    # =======================================================================
    def mu_c(self):
        S_ref, _, c_ref = self._ref()
        return self._require(self.params.mass.mtow, "mass.mtow") / (self.fc.rho*S_ref*c_ref)
 
    def mu_b(self):
        S_ref, b_ref, _ = self._ref()
        return self._require(self.params.mass.mtow, "mass.mtow") / (self.fc.rho*S_ref*b_ref)
 
    def KY2(self):
        _, _, c_ref = self._ref()
        m = self.params.mass
        return self._require(m.I_yy, "mass.I_yy") / (self._require(m.mtow, "mass.mtow")*c_ref**2)
 
    def KX2(self):
        _, b_ref, _ = self._ref()
        m = self.params.mass
        return self._require(m.I_xx, "mass.I_xx") / (self._require(m.mtow, "mass.mtow")*b_ref**2)
 
    def KZ2(self):
        _, b_ref, _ = self._ref()
        m = self.params.mass
        return self._require(m.I_zz, "mass.I_zz") / (self._require(m.mtow, "mass.mtow")*b_ref**2)
 
    def KXZ(self):
        _, b_ref, _ = self._ref()
        m = self.params.mass
        return self._require(m.I_xz, "mass.I_xz") / (self._require(m.mtow, "mass.mtow")*b_ref**2)
 
    # =======================================================================
    # POINT ANALYSES — assemble coefficients at a given state
    # =======================================================================
    def CL(self, alpha, CL0=0.0):
        """Lift coefficient at angle of attack alpha [rad]."""
        return CL0 + self.CL_alpha_aircraft()*alpha
 
    def CD(self, CL):
        ac = self.params.aerodynamics
        e = self._require(ac.e_hor_wings, "aerodynamics.e_hor_wings")
        return self.CD0_total() + CL**2/(np.pi*self._aspect_ratio_aircraft()*e)
 
    def CM(self, alpha, q_hat=0.0, alpha_dot_hat=0.0, delta_e=0.0, CM0=0.0):
        return (CM0 + self.CM_alpha_aircraft()*alpha + self.CM_q_aircraft()*q_hat
                + self.CM_alpha_dot_aircraft()*alpha_dot_hat + self.CM_delta_e()*delta_e)
 
    def CY(self, beta, p_hat=0.0, r_hat=0.0, delta_r=0.0, beta_dot_hat=0.0):
        return (self.CY_beta()*beta + self.CY_p()*p_hat + self.CY_r()*r_hat
                + self.CY_delta_r()*delta_r + self.CY_beta_dot()*beta_dot_hat)
 
    def Cl(self, beta, p_hat=0.0, r_hat=0.0, delta_a=0.0, delta_r=0.0):
        """Rolling-moment coefficient (lowercase l)."""
        return (self.Cl_beta()*beta + self.Cl_p()*p_hat + self.Cl_r()*r_hat
                + self.Cl_delta_a()*delta_a + self.Cl_delta_r()*delta_r)
 
    def CN(self, beta, p_hat=0.0, r_hat=0.0, delta_a=0.0, delta_r=0.0, beta_dot_hat=0.0):
        """Yawing-moment coefficient."""
        return (self.Cn_beta()*beta + self.Cn_p()*p_hat + self.Cn_r()*r_hat
                + self.Cn_delta_a()*delta_a + self.Cn_delta_r()*delta_r
                + self.Cn_beta_dot()*beta_dot_hat)
 
    # =======================================================================
    # SOLVE — populate the OUTPUT snapshot dataclasses (one direction only)
    # =======================================================================
    def solve(self):
        s, c = self.params.stability, self.params.controls
        s.C_L_alpha = self.CL_alpha_aircraft()
        s.C_M_alpha = self.CM_alpha_aircraft()
        s.C_L_alpha_dot = self.CL_alpha_dot_aircraft()
        s.C_M_alpha_dot = self.CM_alpha_dot_aircraft()
        s.C_L_q = self.CL_q_aircraft()
        s.C_M_q = self.CM_q_aircraft()
        s.C_Y_beta = self.CY_beta()
        s.C_L_beta = self.Cl_beta()     
        s.C_N_beta = self.Cn_beta()
        s.C_Y_beta_dot = self.CY_beta_dot()
        s.C_N_beta_dot = self.Cn_beta_dot()
        s.C_Y_p = self.CY_p(); s.C_L_p = self.Cl_p(); s.C_N_p = self.Cn_p()
        s.C_Y_r = self.CY_r(); s.C_L_r = self.Cl_r(); s.C_N_r = self.Cn_r()
        s.C_X_0 = self.CX_0(); s.C_Z_0 = self.CZ_0()
        s.C_X_u = self.CX_u(); s.C_Z_u = self.CZ_u()
        s.C_X_alpha = self.CX_alpha(); s.C_Z_alpha = self.CZ_alpha()
        s.C_Z_alpha_dot = self.CZ_alpha_dot(); s.C_Z_q = self.CZ_q()
        c.C_L_delta_e = self.CL_delta_e(); c.C_M_delta_e = self.CM_delta_e()
        c.C_Z_delta_e = self.CZ_delta_e(); c.C_X_delta_e = self.CX_delta_e()
        c.C_Y_delta_r = self.CY_delta_r(); c.C_N_delta_r = self.Cn_delta_r(); c.C_L_delta_r = self.Cl_delta_r()
        c.C_L_delta_a = self.Cl_delta_a(); c.C_N_delta_a = self.Cn_delta_a()
        return self.params
    

if __name__ == "__main__":

    c2_sweep = Aircraft._le_to_c2(0, A= 5.63, taper=0.4)
    print(np.degrees(c2_sweep))