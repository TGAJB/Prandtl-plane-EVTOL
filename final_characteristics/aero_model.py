"""
aero_model.py
=============

Live, geometry-driven aerodynamic provider. Rebuilds the box-wing surfaces in a
headless VLM (AeroSandbox) + NeuralFoil from the CURRENT WingGeometry and writes
the per-wing coefficients the DATCOM stability model consumes into
params.aerodynamics:

    CL_alpha_fw/aw   (3D wing lift-curve slope, 1/rad)   <- isolated-wing VLM
    x_ac_fw/aw_cruise (a.c. from the wing LEMAC, m)       <- isolated-wing VLM x_np
    cl_alpha_fw/aw   (2D section slope, 1/deg)            <- NeuralFoil
    C_M_ac_fw/aw     (section moment about the a.c.)       <- NeuralFoil

WHY
---
These four used to be frozen XFLR5 point values. Calibration
(aero_calibration.py) showed the VLM reproduces XFLR5 within ~2% on IDENTICAL
geometry, and that the frozen values had been computed for an inconsistent
S=15.67 m^2 wing rather than the converged/runtime geometry (S_fw=12.91 m^2 at
the baseline W/S). Sourcing them live from the actual geometry is therefore a
consistency CORRECTION, and -- crucially -- makes them respond to W/S, AR, taper,
sweep and dihedral instead of staying frozen.

NOT TOUCHED (per the Prandtl aero rule): the DATCOM derivative build in
aircraft.py (it consumes these inputs unchanged), the Rizzo box-wing Oswald e,
and the downwash gradient (computed inside aircraft.py). No single-wing/simple
relations are introduced -- the VLM models each actual wing planform.

INTEGRATION
-----------
stability_eval.evaluate_stability calls apply_vlm_aero(params) once, AFTER the
wing-area split (so the live S/chords are set) and BEFORE ac.solve(). Results are
memoised on a rounded per-wing geometry signature, so the optimiser (which
quantises W/S and moves few levers) mostly hits the cache; a cold isolated-wing
solve is ~1 s. Set USE_VLM_AERO = False to fall back to the frozen sheet values.
"""

import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from parameters import V_CRUISE, H_CRUISE

# AeroSandbox/NeuralFoil emit assorted numpy/tf deprecation chatter; silence it so
# it does not spam the optimiser loop.
warnings.filterwarnings("ignore")
import aerosandbox as asb

# Master switch: live VLM aero (True) vs the frozen sheet values (False).
USE_VLM_AERO = True

_ATMO = asb.Atmosphere(altitude=H_CRUISE)
_MACH = float(V_CRUISE / _ATMO.speed_of_sound())
_RE_PER_MAC = float(_ATMO.density() * V_CRUISE / _ATMO.dynamic_viscosity())  # Re = this * MAC

# Sheet airfoil name -> AeroSandbox database key.
_AIRFOIL_DB = {"NASA LANGLEY LS(1)-0417": "ls417"}

# Quantisation for the cache signature (so near-identical geometry reuses solves).
_S_STEP, _B_STEP, _TAPER_STEP, _ANG_STEP = 1e-3, 1e-3, 1e-4, 1e-3


@lru_cache(maxsize=64)
def _airfoil(sheet_name):
    return asb.Airfoil(_AIRFOIL_DB.get(sheet_name, "ls417"))


def _consistent_chords(area, span, taper):
    """Root/tip chord from the planform identity (S-consistent)."""
    c_root = (2.0 * area) / (span * (1.0 + taper))
    return c_root, taper * c_root


@lru_cache(maxsize=512)
def _isolated_wing_aero(area, span, taper, sweep_deg, dihedral_deg, sheet_airfoil):
    """3D (VLM) + 2D (NeuralFoil) coefficients for ONE wing alone, root LE at the
    origin so x_np is measured from the LEMAC. Cached on the rounded signature."""
    af = _airfoil(sheet_airfoil)
    c_root, c_tip = _consistent_chords(area, span, taper)
    semis = span / 2.0
    x_tip = semis * np.tan(np.radians(sweep_deg))
    z_tip = semis * np.tan(np.radians(dihedral_deg))
    mac = (2.0 / 3.0) * c_root * (1 + taper + taper ** 2) / (1 + taper)

    wing = asb.Wing(name="w", symmetric=True, xsecs=[
        asb.WingXSec(xyz_le=[0.0, 0.0, 0.0], chord=c_root, twist=0.0, airfoil=af),
        asb.WingXSec(xyz_le=[x_tip, semis, z_tip], chord=c_tip, twist=0.0, airfoil=af),
    ])
    airplane = asb.Airplane(name="iso", xyz_ref=[0.0, 0.0, 0.0],
                            s_ref=area, c_ref=mac, b_ref=span, wings=[wing])
    op = asb.OperatingPoint(atmosphere=_ATMO, velocity=V_CRUISE, alpha=2.0)
    vlm = asb.VortexLatticeMethod(airplane=airplane, op_point=op)
    a = vlm.run_with_stability_derivatives(alpha=True, beta=False, p=False, q=False, r=False)

    re = _RE_PER_MAC * mac
    s2d = af.get_aero_from_neuralfoil(alpha=np.array([0.0, 2.0, 4.0]), Re=re, mach=_MACH)
    cl_alpha_per_deg = float(np.polyfit([0.0, 2.0, 4.0], np.asarray(s2d["CL"]), 1)[0])
    c_m_ac = float(np.mean(np.asarray(s2d["CM"])))

    return {
        "CL_alpha": float(a["CLa"]),     # 1/rad
        "x_ac": float(a["x_np"]),        # m, from LEMAC
        "cl_alpha_per_deg": cl_alpha_per_deg,
        "C_M_ac": c_m_ac,
    }


def _sig(area, span, taper, sweep_deg, dihedral_deg):
    """Rounded geometry signature for cache reuse."""
    return (
        round(area / _S_STEP) * _S_STEP,
        round(span / _B_STEP) * _B_STEP,
        round(taper / _TAPER_STEP) * _TAPER_STEP,
        round(sweep_deg / _ANG_STEP) * _ANG_STEP,
        round(dihedral_deg / _ANG_STEP) * _ANG_STEP,
    )


def wing_aero(area, span, taper, sweep_deg, dihedral_deg, sheet_airfoil):
    """Public, cached per-wing aero from the live planform."""
    return _isolated_wing_aero(*_sig(area, span, taper, sweep_deg, dihedral_deg), sheet_airfoil)


def apply_vlm_aero(params):
    """Overwrite the frozen aero inputs in params.aerodynamics with live,
    geometry-driven VLM/NeuralFoil values. No-op if USE_VLM_AERO is False.

    Call AFTER the wing-area split (so wg.S_fw/S_aw and chords are live) and
    BEFORE ac.solve(). Leaves the DATCOM derivative build, the Rizzo Oswald e and
    the downwash gradient untouched.
    """
    if not USE_VLM_AERO:
        return
    wg = params.wing_geometry
    aero = params.aerodynamics

    fw = wing_aero(wg.S_fw, wg.b_fw, wg.taper_fw, wg.LE_sweep_fw,
                   wg.dihedral_front_wing, wg.airfoil_fw)
    aw = wing_aero(wg.S_aw, wg.b_aw, wg.taper_aw, wg.LE_sweep_aw,
                   wg.dihedral_aft_wing, wg.airfoil_aw)

    aero.CL_alpha_fw = fw["CL_alpha"]
    aero.x_ac_fw_cruise = fw["x_ac"]
    aero.cl_alpha_fw = fw["cl_alpha_per_deg"]
    aero.C_M_ac_fw = fw["C_M_ac"]

    aero.CL_alpha_aw = aw["CL_alpha"]
    aero.x_ac_aw_cruise = aw["x_ac"]
    aero.cl_alpha_aw = aw["cl_alpha_per_deg"]
    aero.C_M_ac_aw = aw["C_M_ac"]


def main():
    """Quick self-test: print the live coefficients at the baseline geometry."""
    from parameters import WingGeometry
    wg = WingGeometry()
    fw = wing_aero(wg.S_fw, wg.b_fw, wg.taper_fw, wg.LE_sweep_fw,
                   wg.dihedral_front_wing, wg.airfoil_fw)
    print("Live VLM aero at baseline geometry (front wing):")
    for k, v in fw.items():
        print(f"  {k:18s}: {v:.4f}")


if __name__ == "__main__":
    main()
