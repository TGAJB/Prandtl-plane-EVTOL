"""
Unit tests for the structures mass blocks -- block IDs MT, ST-FUS, ST-WG, ST-SEC.

Entry points verified:
  ST-FUS  mass_components.fuselage_mass
  ST-WG   mass_components.wing_geometry, wing_mass
  ST-SEC  mass_components.tail_mass, hinge_mass, misc_mass
  MT      mtow_sizing.compute_mtow            (the mass-aggregation loop)

NB the decomposition table lists `battery_box_dimensions` under ST-SEC, but no
such function exists in the codebase at the time of writing; the secondary-mass
coverage here is tail/hinge/misc. Flag to structures if that block is expected.

compute_mtow runs a (seeded, deterministic) landing-gear optimiser, so each call
costs ~2 s; the MT states are built once in module-scoped fixtures and reused.

Catalogue coverage:
  U1/U5  Formula inspection / hand reference - regression and linear formulas
  U3     Null input        - zero MTOW behaviour
  U4     Extreme value     - 10 t MTOW stays finite and ordered
  U7     Scaling           - component masses rise with MTOW
  U8     Conservation      - converged MTOW equals the sum of its components
  U10    Output format     - masses are finite positive floats

All masses in kg.
"""

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import class_II_sizing.mass_components as mc
import class_II_sizing.mtow_sizing as mt
from parameters import G, WING_LOADING_N, AREA_SPLIT, TAPER_W, L_FUS, PER_FUS_MAX, N_PAX

MTOW_LO, MTOW_HI = 1700.0, 2200.0


# ===========================================================================
# ST-FUS -- fuselage
# ===========================================================================
def test_fuselage_formula():
    # Cessna-style regression (Raymer/GD form); recompute symbol-for-symbol.
    m = 2000.0
    expected = (0.453592
                * (14.86 * ((m * 2.20462) ** 0.144) * ((L_FUS * 3.28084) ** 0.778)
                   / ((PER_FUS_MAX * 3.28084) ** 0.778))
                * ((L_FUS * 3.28084) ** 0.383) * N_PAX ** 0.455)
    assert mc.fuselage_mass(m) == pytest.approx(expected)


def test_fuselage_zero_at_zero_mtow():
    assert mc.fuselage_mass(0.0) == pytest.approx(0.0)   # mass ~ MTOW^0.144 -> 0


def test_fuselage_monotonic_and_finite():
    assert mc.fuselage_mass(MTOW_HI) > mc.fuselage_mass(MTOW_LO) > 0.0
    big = mc.fuselage_mass(10_000.0)
    assert math.isfinite(big) and big > mc.fuselage_mass(2000.0)


# ===========================================================================
# ST-WG -- wing / winglet
# ===========================================================================
def test_wing_area_from_design_loading():
    # S = W / (W/S): area follows directly from the design wing loading.
    m = 2000.0
    geom = mc.wing_geometry(m)
    assert geom["total_area_m2"] == pytest.approx(m * G / WING_LOADING_N)


def test_wing_geometry_internal_consistency():
    geom = mc.wing_geometry(2000.0)
    # taper, area split and aspect ratio must be mutually consistent.
    assert geom["tip_chord_m"] == pytest.approx(TAPER_W * geom["root_chord_m"])
    assert geom["area_per_wing_m2"] == pytest.approx(geom["total_area_m2"] * AREA_SPLIT)
    assert geom["aspect_ratio"] == pytest.approx(geom["span_m"] ** 2 / geom["total_area_m2"])


def test_wing_mass_monotonic_and_finite():
    assert mc.wing_mass(MTOW_HI) > mc.wing_mass(MTOW_LO) > 0.0
    assert math.isfinite(mc.wing_mass(10_000.0))


# ===========================================================================
# ST-SEC -- secondary masses (tail, hinge, misc)
# ===========================================================================
def test_hinge_mass_linear():
    assert mc.hinge_mass(2000.0) == pytest.approx(0.05 * 2000.0)


def test_misc_mass_linear_with_offset():
    # 10% of MTOW plus a fixed 32 kg thermal-management allowance.
    assert mc.misc_mass(2000.0) == pytest.approx(0.10 * 2000.0 + 32.0)


def test_tail_mass_aero_floor_and_monotonic():
    # At zero MTOW the rear-wing transfer load vanishes but the dive-speed aero
    # load case remains, so the tail keeps a positive structural floor.
    assert mc.tail_mass(0.0) > 0.0
    assert mc.tail_mass(MTOW_HI) > mc.tail_mass(MTOW_LO)
    assert math.isfinite(mc.tail_mass(10_000.0))


# ===========================================================================
# MT -- MTOW aggregation loop
# ===========================================================================
@pytest.fixture(scope="module")
def mt_lo():
    return mt.compute_mtow(MTOW_LO, verbose=False)


@pytest.fixture(scope="module")
def mt_hi():
    return mt.compute_mtow(MTOW_HI, verbose=False)


# the component lines that must sum to the reported MTOW
_PARTS = ["fuselage", "wing", "landing_gear", "tail", "motors", "props",
          "hubs", "battery", "payload", "misc", "hinge"]


def test_mtow_equals_component_sum(mt_hi):
    # U8: the headline check -- the loop must not drop or double-count a mass.
    assert mt_hi["mtow"] == pytest.approx(sum(mt_hi[k] for k in _PARTS))


def test_mtow_components_finite_positive(mt_hi):
    for k in _PARTS:
        assert math.isfinite(mt_hi[k]) and mt_hi[k] > 0.0


def test_mtow_increases_with_input_mass(mt_lo, mt_hi):
    # Heavier input -> heavier structure/battery -> heavier estimate (gradient<1
    # is what makes the fixed-point iteration converge).
    assert mt_hi["mtow"] > mt_lo["mtow"]
