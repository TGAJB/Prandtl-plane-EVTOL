"""
aero_calibration.py  (READ-ONLY validation gate -- changes no model state)

Builds the box-wing surfaces parametrically in AeroSandbox from WingGeometry and
checks whether a headless VLM (+ NeuralFoil for 2D) REPRODUCES the frozen
XFLR5-sourced aero coefficients before we let it replace them in the loop.

It maps the way the DATCOM model consumes the values:
  * CL_alpha_fw/aw, x_ac_*_cruise  -> per-wing ISOLATED 3D VLM (each wing alone),
    x_ac measured from that wing's LEMAC (matches "as seen from the LEMAC").
  * cl_alpha (2D), C_M_ac           -> NeuralFoil on the LS(1)-0417 section at the
    cruise Reynolds/Mach.

Run:  python final_characteristics/aero_calibration.py
"""

import sys
from pathlib import Path

import numpy as np
import aerosandbox as asb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from parameters import WingGeometry, AerodynamicCoefficients, Mission, V_CRUISE, H_CRUISE

_WG = WingGeometry()
_AERO = AerodynamicCoefficients()

AIRFOIL = asb.Airfoil("ls417")          # LS(1)-0417 == the sheet airfoil
ALT = H_CRUISE
V = V_CRUISE
ATMO = asb.Atmosphere(altitude=ALT)
MACH = V / ATMO.speed_of_sound()


def _consistent_chords(S, b, taper):
    """Root/tip chord from the planform identity (the live, S-consistent values)."""
    c_root = (2.0 * S) / (b * (1.0 + taper))
    return c_root, taper * c_root


def _single_wing_airplane(S, b, taper, sweep_deg, dihedral_deg):
    """One isolated wing (symmetric) placed with its root LE at the origin, so the
    neutral point comes out measured from the LEMAC."""
    c_root, c_tip = _consistent_chords(S, b, taper)
    semis = b / 2.0
    x_tip = semis * np.tan(np.radians(sweep_deg))
    z_tip = semis * np.tan(np.radians(dihedral_deg))
    mac = (2.0 / 3.0) * c_root * (1 + taper + taper ** 2) / (1 + taper)
    wing = asb.Wing(
        name="wing",
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=c_root, twist=0.0, airfoil=AIRFOIL),
            asb.WingXSec(xyz_le=[x_tip, semis, z_tip], chord=c_tip, twist=0.0, airfoil=AIRFOIL),
        ],
    )
    # Reference x at the LEMAC (root LE = origin for an unswept wing) so x_np is
    # reported relative to the LEMAC, matching x_ac_*_cruise.
    airplane = asb.Airplane(name="iso", xyz_ref=[0.0, 0.0, 0.0],
                            s_ref=S, c_ref=mac, b_ref=b, wings=[wing])
    return airplane, mac


def isolated_wing_aero(S, b, taper, sweep_deg, dihedral_deg):
    airplane, mac = _single_wing_airplane(S, b, taper, sweep_deg, dihedral_deg)
    op = asb.OperatingPoint(atmosphere=ATMO, velocity=V, alpha=2.0)
    vlm = asb.VortexLatticeMethod(airplane=airplane, op_point=op)
    aero = vlm.run_with_stability_derivatives(alpha=True, beta=False, p=False, q=False, r=False)
    cl_alpha = float(aero["CLa"])               # dCL/dalpha [1/rad]
    # x_np is reported in the geometry frame; the root LE sits at x=0, so it is
    # already measured from the LEMAC (matches x_ac_*_cruise).
    x_ac_from_lemac = float(aero["x_np"])
    return {"CL_alpha": cl_alpha, "x_ac_from_lemac": x_ac_from_lemac, "MAC": mac,
            "AR": b ** 2 / S}


def section_2d():
    """2D section slope and moment from NeuralFoil at the cruise Reynolds number."""
    c_root, _ = _consistent_chords(_WG.S_fw, _WG.b_fw, _WG.taper_fw)
    mac = (2.0 / 3.0) * c_root * (1 + _WG.taper_fw + _WG.taper_fw ** 2) / (1 + _WG.taper_fw)
    Re = ATMO.density() * V * mac / ATMO.dynamic_viscosity()
    alphas = np.array([0.0, 2.0, 4.0])
    res = AIRFOIL.get_aero_from_neuralfoil(alpha=alphas, Re=Re, mach=MACH)
    cl = np.array(res["CL"]); cm = np.array(res["CM"])
    cl_alpha_per_deg = float(np.polyfit(alphas, cl, 1)[0])
    return {"Re": Re, "cl_alpha_per_deg": cl_alpha_per_deg, "Cm": float(np.mean(cm))}


def main():
    print("=== AeroSandbox VLM calibration vs frozen XFLR5 values ===")
    print(f"  cruise: V={V:.2f} m/s, alt={ALT:.0f} m, M={MACH:.3f}\n")

    fw = isolated_wing_aero(_WG.S_fw, _WG.b_fw, _WG.taper_fw, _WG.LE_sweep_fw, _WG.dihedral_front_wing)
    aw = isolated_wing_aero(_WG.S_aw, _WG.b_fw, _WG.taper_aw, _WG.LE_sweep_aw, _WG.dihedral_aft_wing)
    s2d = section_2d()

    def row(label, vlm_val, frozen, unit=""):
        d = (vlm_val - frozen) / frozen * 100 if frozen else float("nan")
        print(f"  {label:24s} VLM {vlm_val:9.4f}   XFLR5 {frozen:9.4f}   diff {d:+6.1f}% {unit}")

    print(f"  [front wing]  AR={fw['AR']:.2f}  MAC={fw['MAC']:.3f} m")
    row("CL_alpha_fw [1/rad]", fw["CL_alpha"], _AERO.CL_alpha_fw)
    row("x_ac_fw from LEMAC [m]", fw["x_ac_from_lemac"], _AERO.x_ac_fw_cruise)
    print(f"  [aft wing]   AR={aw['AR']:.2f}  MAC={aw['MAC']:.3f} m")
    row("CL_alpha_aw [1/rad]", aw["CL_alpha"], _AERO.CL_alpha_aw)
    row("x_ac_aw from LEMAC [m]", aw["x_ac_from_lemac"], _AERO.x_ac_aw_cruise)
    print(f"  [2D section LS(1)-0417]  Re={s2d['Re']:.3e}")
    row("cl_alpha [1/deg]", s2d["cl_alpha_per_deg"], _AERO.cl_alpha_fw)
    row("C_M_ac (section Cm)", s2d["Cm"], _AERO.C_M_ac_fw)


if __name__ == "__main__":
    main()
