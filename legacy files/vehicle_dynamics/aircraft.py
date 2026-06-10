from vd_parameters import AircraftParameters
#from parameters import *
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
    mach:               float = 0.17
    rho:                float = None
    tas:                float = None       # true airspeed [m/s]
    alpha:              float = 0.0        # [rad]
    gamma0:             float = 0.0        # [rad]
    config:             str = "cruise"     # selects x_ac_*_cruise vs *_approach


# ---------------------------------------------------------------------------
# Aircraft: holds the parameter sheet + the DATCOM methods.
# ---------------------------------------------------------------------------
class Aircraft:
    def __init__(self, params: AircraftParameters, physical: Physical, fc: FlightCondition):
        self.params = params
        self.physical = physical
        self.fc = fc

    def _require(self, value, name):
        """
        Helper method to check for missing DATCOM inputs.
        """
        if value is None:
            raise ValueError(f"DATCOM input '{name}' is None — add it to the sheet.")
        return value

    def beta(self):
        return np.sqrt(1 - self.fc.mach**2)
    
    @staticmethod
    def _le_to_c2(le_sweep, A, taper):
        return np.arctan(np.tan(le_sweep) - (2/A)*((1 - taper)/(1 + taper)))

    @staticmethod
    def _le_to_c4(le_sweep, A, taper):
        return np.arctan(np.tan(le_sweep) - (1/A)*((1 - taper)/(1 + taper)))

    
    # =======================================================================
    # DATCOM methods: these are the core of the code, and should be as close to the original DATCOM as possible.
    # =======================================================================
    def CL_alpha_section(self, t_c, phi_te, ratio): # These are all placeholders for now, add them to param sheet after
        """
        Section lift curve slope (per radian).
        """
        cl_alpha_theory = 6.28 + 4.7*t_c*(0.00375*phi_te)
        return (1.05/self.beta()) * cl_alpha_theory * ratio
        
    def CL_alpha_front_wing(self):
        """
        Front wing lift curve slope (per radian).
        """
        wg = self.params.wing_geometry

        S  = self._require(wg.S_fw, "wing_geometry.S_fw")
        b  = self._require(wg.b_fw, "wing_geometry.b_fw")
        LE_sweep_fw = self._require(wg.LE_sweep_fw, "wing_geometry.LE_sweep_fw")

        A  = b**2 / S
        k = self.CL_alpha_section(t_c, phi_te, ratio)/(2*np.pi/self.beta())
        sweep_c2_fw = self._le_to_c2(LE_sweep_fw, A, wg.taper_fw)

        return (2*np.pi*A)/(2 + np.sqrt(4 + ((A**2)*(self.beta()**2)/(k**2))*(1 + (np.tan(sweep_c2_fw)**2)/self.beta()**2)))
    
    def CL_alpha_aft_wing(self): # Half chord sweep is a placeholeder, add it to param sheet after
        """
        Aft wing lift curve slope (per radian).
        """
        wg = self.params.wing_geometry

        S  = self._require(wg.S_aw, "wing_geometry.S_aft")
        b  = self._require(wg.b_aw, "wing_geometry.b_aft")
        LE_sweep_aw = self._require(wg.LE_sweep_aw, "wing_geometry.LE_sweep_aft")

        A  = b**2 / S
        k = self.CL_alpha_section(t_c, phi_te, ratio)/(2*np.pi/self.beta())
        sweep_c2_aft = self._le_to_c2(LE_sweep_aw, A, wg.taper_aw)

        return (2*np.pi*A)/(2 + np.sqrt(4 + ((A**2)*(self.beta()**2)/(k**2))*(1 + (np.tan(sweep_c2_aft)**2)/self.beta()**2)))
    
    def downwash_gradient(self):
        """
        Downwash gradient at the tail.
        """
        wg = self.params.wing_geometry

        S = self._require(wg.S_fw, "wing_geometry.S_fw")
        b = self._require(wg.b_fw, "wing_geometry.b_fw")
        A = b**2 / S
        sweep_c4_fw = self._le_to_c4(wg.LE_sweep_fw, A, wg.taper_fw)

        K_A = (1/A) - (1/(1 + A**1.7))
        K_lambda = (10 - 3*wg.taper_fw) / 7
        K_H = (1 - (np.abs(wg.gap)/b))/(((2*wg.stagger)/b)**(1/3))

        beta = self.beta()

        return 4.44*(K_A*K_lambda*K_H*(np.sqrt(np.cos(sweep_c4_fw))))**(1.19) * (1/beta)
    

    def CL_alpha_aircraft(self):  
        """
        Aircraft lift curve slope (per radian).
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
        b_2 = self._require(wg.b_aw, "wing_geometry.b_aw")
        d_2 = self._require(fg.d_aw, "fuselage_geometry.d_aw")
        A_2 = self._require(wg.A_aw, "wing_geometry.A_aw")

        CL_alpha_1 = self._require(ac.CL_alpha_fw, "aerodynamics.CL_alpha_fw")
        CL_alpha_2 = self._require(ac.CL_alpha_aw, "aerodynamics.CL_alpha_aw")
        downwash = self.downwash_gradient()
        q_2_inf_ratio = self._require(ac.dyn_pres_ratio_fw_to_aw, "aerodynamics.dyn_pres_ratio_fw_to_aw")
        I_v = self._require(ac.I_v, "aerodynamics.I_v")

        # COMPUTATIONS
        K_N = (np.pi*(d_1**2))/(2*CL_alpha_1*S_e_1)
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
        ac = self.params.aerodynamics

        x_ac_fw = self._require(ac.x_ac_fw_cruise, "aerodynamics.x_ac_fw")
        c_r = self._require(wg.chord_fw_root, "wing_geometry.chord_fw_root")
        MAC = self._require(wg.MAC_fw, "wing_geometry.MAC_fw")

        CL_alpha_fw = self.CL_alpha_front_wing()
        return (n - (x_ac_fw/c_r)) * CL_alpha_fw * (c_r/MAC)
    
    def CM_alpha_aft_wing(self, n): # n is the selected location along the wing chord
        """
        Aft wing pitching moment coefficient gradient (per radian).
        """
        wg = self.params.wing_geometry
        ac = self.params.aerodynamics

        x_ac_aft = self._require(ac.x_ac_aw_cruise, "aerodynamics.x_ac_aw")
        c_r = self._require(wg.chord_aw_root, "wing_geometry.chord_aw_root")
        MAC = self._require(wg.MAC_aw, "wing_geometry.MAC_aw")

        CL_alpha_aft = self.CL_alpha_aft_wing()
        return (n - (x_ac_aft/c_r)) * CL_alpha_aft * (c_r/MAC)
    
    def CM_alpha_aircraft(self):
        """
        Aircraft pitching moment coefficient gradient (per radian).
        """
        wg = self.params.wing_geometry
        ac = self.params.aerodynamics  
        m = self.params.mass

        S_fw = self._require(wg.S_fw, "wing_geometry.S_fw")
        S_aw = self._require(wg.S_aw, "wing_geometry.S_aft")
        MAC_fw = self._require(wg.MAC_fw, "wing_geometry.MAC_fw")
        MAC_aw = self._require(wg.MAC_aw, "wing_geometry.MAC_aw")
        x_ac_fw = self._require(ac.x_ac_fw_cruise, "aerodynamics.x_ac_fw")
        x_ac_aw = self._require(ac.x_ac_aw_cruise, "aerodynamics.x_ac_aw")
        x_cg = self._require(m.x_cg_opt, "mass.x_cg_opt")
        eta = self._require(ac.flow_speed_ratio_fw_to_aw, "aerodynamics.flow_speed_ratio_fw_to_aw")

        downwash = self.downwash_gradient()
        CL_alpha_fw = self.CL_alpha_front_wing()
        CL_alpha_aft = self.CL_alpha_aft_wing()

        front_term = -CL_alpha_fw * ((x_cg - x_ac_fw)/MAC_fw)
        aft_term = -CL_alpha_aft * ((x_cg - x_ac_aw)/MAC_fw)*(1 - downwash)*eta*(S_aw/S_fw)*(MAC_aw/MAC_fw)

        return front_term + aft_term
    
    def CL_q_front_wing(self):
        """
        Front wing lift coefficient gradient with respect to pitch rate (per rad/s).
        """
        wg = self.params.wing_geometry
        ac = self.params.aerodynamics  
        m = self.params.mass

        c_ref = self._require(wg.MAC_fw, "wing_geometry.MAC_fw")
        S_ref = self._require(wg.S_tot,   "wing_geometry.S_tot")
        x_cg  = self._require(m.x_cg_opt, "mass.x_cg_opt")
        x_ac  = self._require(ac.x_ac_fw_cruise, "aerodynamics.x_ac_fw_cruise")
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
        m = self.params.mass

        c_ref = self._require(wg.MAC_aw, "wing_geometry.MAC_aw")
        S_ref = self._require(wg.S_tot,   "wing_geometry.S_tot")
        x_cg  = self._require(m.x_cg_opt, "mass.x_cg_opt")
        x_ac  = self._require(ac.x_ac_aw_cruise, "aerodynamics.x_ac_aw_cruise")
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
        ac = self.params.aerodynamics  
        m = self.params.mass

        c_ref = self._require(wg.MAC_fw, "wing_geometry.MAC_fw")
        S_fw = self._require(wg.S_fw,   "wing_geometry.S_fw")
        b_fw = self._require(wg.b_fw,   "wing_geometry.b_fw")
        x_cg  = self._require(m.x_cg_opt, "mass.x_cg_opt")
        x_ac  = self._require(ac.x_ac_fw_cruise, "aerodynamics.x_ac_fw_cruise")

        A = b_fw**2 / S_fw
        sweep_c4_fw = self._le_to_c4(wg.LE_sweep_fw, A, wg.taper_fw)
        x_bar = (x_ac - x_cg) / c_ref

        first_term = (A*(0.5*x_bar + 2*(x_bar**2)))/(A + 2*np.cos(sweep_c4_fw))
        second_term = (1/24)*(((A**3)*(np.tan(sweep_c4_fw))**2)/(A + 6*np.cos(sweep_c4_fw)))
        third_term = 1/8

        CM_q = -0.7*self.CL_alpha_section(t_c, phi_te, ratio)*np.cos(sweep_c4_fw)*(first_term + second_term + third_term)

        B = np.sqrt(1 - (self.fc.mach**2)*(np.cos(sweep_c4_fw)**2))

        num = (((A**3)*(np.tan(sweep_c4_fw))**2)/(A*B + 6*np.cos(sweep_c4_fw))) + (3/B)
        den = (((A**3)*(np.tan(sweep_c4_fw))**2)/(A + 6*np.cos(sweep_c4_fw))) + 3

        return (num/den)*CM_q
    
    