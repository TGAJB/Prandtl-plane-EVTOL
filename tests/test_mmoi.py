"""
Unit tests for the mass-moment-of-inertia buildup (class_II_sizing.MMOI).

Coverage:
  1. shape helpers - shell/box/disc closed forms, rotation identity + trace
  2. mass closure - component table sums back to the breakdown MTOW exactly
  3. rigid-body validity - positive moments + triangle inequalities
  4. lateral symmetry - y_cg = 0, Ixy = Iyz = 0, Ixz nonzero expected
  5. near-planar band - Izz vs Ixx + Iyy
  6. nondimensional gyradii - loose Raymer-style sanity bounds
  7. parallel-axis minimality - inertia about the CG is the minimum
  8. CG sanity - lands inside the fuselage near the vd target (2.8 m)

All quantities SI.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import class_II_sizing.MMOI as mmoi
from class_II_sizing.mtow_sizing import converged_mass
from parameters import WING_SPAN, L_FUS


@pytest.fixture(scope="module")
def breakdown():
    return converged_mass()


@pytest.fixture(scope="module")
def components(breakdown):
    return mmoi.build_components(breakdown)


@pytest.fixture(scope="module")
def res(components):
    return mmoi.compute_inertia(components)


# -- 1. Shape helpers ------------------------------------------------------

def test_cylinder_shell_closed_form():
    inertia = mmoi.inertia_cylinder_shell(10.0, 0.5, 4.0)
    assert inertia[0, 0] == pytest.approx(10.0 * 0.5**2)              # m R^2
    assert inertia[1, 1] == pytest.approx(10.0 * (0.5**2 / 2 + 4.0**2 / 12))
    assert inertia[1, 1] == inertia[2, 2]


def test_solid_box_closed_form():
    inertia = mmoi.inertia_solid_box(12.0, 1.0, 2.0, 3.0)
    assert inertia[0, 0] == pytest.approx(12.0 * (2.0**2 + 3.0**2) / 12)
    assert inertia[1, 1] == pytest.approx(12.0 * (1.0**2 + 3.0**2) / 12)
    assert inertia[2, 2] == pytest.approx(12.0 * (1.0**2 + 2.0**2) / 12)
    # degenerate box = slender rod along x
    rod = mmoi.inertia_solid_box(6.0, 2.0, 0.0, 0.0)
    assert rod[0, 0] == 0.0
    assert rod[1, 1] == pytest.approx(6.0 * 2.0**2 / 12)


def test_thin_disc_closed_form():
    inertia = mmoi.inertia_thin_disc(8.0, 1.0)
    assert inertia[2, 2] == pytest.approx(8.0 * 1.0**2 / 2)
    assert inertia[0, 0] == pytest.approx(inertia[2, 2] / 2)


def test_rotation_identity_and_trace():
    inertia = mmoi.inertia_solid_box(5.0, 1.0, 2.0, 3.0)
    assert np.allclose(mmoi.rotate_about_x(inertia, 0.0), inertia)
    rotated = mmoi.rotate_about_x(inertia, 0.7)
    assert np.trace(rotated) == pytest.approx(np.trace(inertia))


# -- 2. Mass closure -------------------------------------------------------

def test_mass_closure(breakdown, components):
    m_total = sum(c["mass"] for c in components)
    assert m_total == pytest.approx(breakdown["mtow"], rel=1e-6)


# -- 3. Rigid-body validity ------------------------------------------------

def test_moments_positive_and_triangle_inequalities(res):
    ixx, iyy, izz = res["Ixx"], res["Iyy"], res["Izz"]
    assert ixx > 0 and iyy > 0 and izz > 0
    assert ixx + iyy >= izz
    assert iyy + izz >= ixx
    assert izz + ixx >= iyy


# -- 4. Lateral symmetry ---------------------------------------------------

def test_lateral_symmetry(res):
    assert abs(res["cg"][1]) < 1e-9
    assert abs(res["Ixy"]) < 1e-6 * res["Izz"]
    assert abs(res["Iyz"]) < 1e-6 * res["Izz"]
    # high rear wing aft + low battery/gear forward -> nonzero product expected
    assert abs(res["Ixz"]) > 1.0


# -- 5. Near-planar band ---------------------------------------------------

def test_near_planar_band(res):
    # 1.0 exactly for a planar (z = const) distribution; the 2.1 m wing gap,
    # tail and underfloor masses pull it below - loose band
    ratio = res["Izz"] / (res["Ixx"] + res["Iyy"])
    assert 0.55 < ratio <= 1.0


# -- 6. Gyradii ------------------------------------------------------------

def test_gyradii_bands(res):
    # Loose Raymer-style sanity bounds. GA-twin typical R_x ~ 0.25-0.30; tip
    # plates and outboard rotor pods push a box wing above that band.
    m = res["mass"]
    rx = np.sqrt(res["Ixx"] / m) / (WING_SPAN / 2.0)
    ry = np.sqrt(res["Iyy"] / m) / (L_FUS / 2.0)
    assert 0.15 < rx < 0.50
    assert 0.3 < ry < 1.2


# -- 7. Parallel-axis minimality -------------------------------------------

def test_inertia_minimal_about_cg(components, res):
    about_nose = mmoi.compute_inertia(components, about=np.zeros(3))
    assert about_nose["Ixx"] > res["Ixx"]
    assert about_nose["Iyy"] > res["Iyy"]
    assert about_nose["Izz"] > res["Izz"]


# -- 8. CG sanity ----------------------------------------------------------

def test_cg_inside_fuselage(res):
    assert 2.0 < res["cg"][0] < 4.5   # near the vd x_cg_opt = 2.8 m


# -- API mapping ------------------------------------------------------------

def test_as_mass_properties_keys(res):
    mapped = mmoi.as_mass_properties(res)
    assert set(mapped) == {"I_xx", "I_yy", "I_zz", "I_xz", "z_cg"}
    assert mapped["I_xx"] == pytest.approx(res["Ixx"])
    assert mapped["z_cg"] == pytest.approx(res["cg"][2])
