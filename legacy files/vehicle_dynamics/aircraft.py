from vd_parameters import AircraftParameters
from parameters import *
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

        return 4.44*(K_A*K_lambda*K_H*(np.sqrt(np.cos(sweep_c4_fw))))**(1.19)
    
    def CL_alpha_aircraft(self, d, b, r, S_e_fw, S_e_aft, eta_h): # These are all placeholders for now, add them to param sheet after  
        """
        Aircraft lift curve slope (per radian).
        """
        wg = self.params.wing_geometry
        S_fw = self._require(wg.S_fw, "wing_geometry.S_fw")
        S_aft = self._require(wg.S_aw, "wing_geometry.S_aft")

        CL_alpha_fw = self.CL_alpha_front_wing()
        CL_alpha_aft = self.CL_alpha_aft_wing()
        downwash = self.downwash_gradient()

        K_WB = ((d/b) + 1)**2
        K_N_fw = (2*np.pi*(r**2))/(CL_alpha_fw*S_e_fw)
        K_N_aft = (2*np.pi*(r**2))/(CL_alpha_aft*S_e_aft)

        return CL_alpha_fw*(K_N_fw + K_WB)*(S_e_fw/S_fw) + CL_alpha_aft*K_N_aft*(1 - downwash)*eta_h*(S_aft/S_fw)*(S_e_aft/S_aft)
    
