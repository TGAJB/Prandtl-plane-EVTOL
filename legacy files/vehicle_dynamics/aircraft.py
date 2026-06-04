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
    
    # =======================================================================
    # DATCOM methods: these are the core of the code, and should be as close to the original DATCOM as possible.
    # =======================================================================
    def CL_alpha_section(self, t_c, phi_te, ratio): # These are all placeholders for now, add them to param sheet after
        """
        Section lift curve slope (per radian).
        """
        cl_alpha_theory = 6.28 + 4.7*t_c*(0.00375*phi_te)
        return (1.05/self.beta()) * cl_alpha_theory * ratio
        
    def CL_alpha_front_wing(self, sweep_c2_fw): # Half chord sweep is a placeholeder, add it to param sheet after
        """
        Front wing lift curve slope (per radian).
        """
        wg = self.params.wing_geometry
        S  = self._require(wg.S_fw, "wing_geometry.S_fw")
        b  = self._require(wg.b_fw, "wing_geometry.b_fw")
        A  = b**2 / S
        k = self.CL_alpha_section(t_c, phi_te, ratio)/(2*np.pi/self.beta())
        return (2*np.pi*A)/(2 + np.sqrt(4 + ((A**2)*(self.beta()**2)/(k**2))*(1 + (np.tan(sweep_c2_fw)**2)/self.beta()**2)))
    
    def CL_alpha_aft_wing(self, sweep_c2_aft): # Half chord sweep is a placeholeder, add it to param sheet after
        """
        Aft wing lift curve slope (per radian).
        """
        wg = self.params.wing_geometry
        S  = self._require(wg.S_aft, "wing_geometry.S_aft")
        b  = self._require(wg.b_aft, "wing_geometry.b_aft")
        A  = b**2 / S
        k = self.CL_alpha_section(t_c, phi_te, ratio)/(2*np.pi/self.beta())
        return (2*np.pi*A)/(2 + np.sqrt(4 + ((A**2)*(self.beta()**2)/(k**2))*(1 + (np.tan(sweep_c2_aft)**2)/self.beta()**2)))
    
    def downwash_gradient(self, sweep_c4_fw): # Quarter chord sweep is a placeholeder, add it to param sheet after
        """
        Downwash gradient at the tail.
        """
        wg = self.params.wing_geometry
        S = self._require(wg.S_fw, "wing_geometry.S_fw")
        b = self._require(wg.b_fw, "wing_geometry.b_fw")
        A = b**2 / S

        K_A = (1/A) - (1/(1 + A**1.7))
        K_lambda = (10 - 3*wg.taper_fw) / 7
        K_H = (1 - (np.abs(wg.gap)/b))/(((2*wg.stagger)/b)**(1/3))

        return 4.44*(K_A*K_lambda*K_H*(np.sqrt(np.cos(sweep_c4_fw))))**(1.19)