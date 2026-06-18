"""
Unit tests for the mission-energy and battery-sizing block
(class_II_sizing.energy) -- block ID EN.

Entry points verified: mission_energy(mtow) and battery_mass(mtow); the
per-segment power functions are exercised indirectly through them.

Catalogue coverage (Table: unit test type catalogue):
  U1  Formula inspection   - cruise_power matches W*V/(L/D*eta) symbol-for-symbol
  U2  Unit consistency     - mission_energy is an energy [W*s]; battery cell chain Wh->kg
  U3  Null input           - zero MTOW -> zero propulsive energy
  U4  Extreme value        - 10 t MTOW stays finite and ordered
  U5  Hand-computed ref    - cruise_power(1980 kg) against pencil value
  U7  Scaling (metamorphic)- cruise_power linear in mass; energy/battery monotonic
  U8  Conservation         - mission_energy equals the sum of its phase terms
  V   Reference cross-check - battery_mass(MTOW) reproduces the optimizer's stored pack

All quantities SI unless a kWh/Wh conversion is noted.
"""

import json
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import class_II_sizing.energy as en
from parameters import (
    G, V_CRUISE, LD_CRUISE, ETA_CRUISE, N_PROP, V_I, VS_0,
    T_TAKEOFF, T_LANDING, T_CRUISE, T_CLIMB, T_CLIMB_ACC, T_VERTICAL_CLIMB,
    E_CELL_WH, M_CELL_KG, CELL_TO_PACK, E_AUX_KWH, RESERVE_FRAC, S_SERIES,
)

MTOW_TEST = 1979.905760376435   # optimizer final design point


# ---------------------------------------------------------------------------
# U1 / U5 - cruise power: formula inspection and hand-computed reference
# ---------------------------------------------------------------------------
def test_cruise_power_formula():
    # P_cruise = W * V / (L/D * eta_cruise), recomputed independently.
    expected = MTOW_TEST * G * V_CRUISE / (LD_CRUISE * ETA_CRUISE)
    assert en.cruise_power(MTOW_TEST) == pytest.approx(expected)


def test_cruise_power_hand_reference():
    # Pencil check at the design MTOW: W = m*g, then W*V/(L/D*eta).
    W = MTOW_TEST * 9.81
    hand = W * (200.0 / 3.6) / (14.7 * 0.95)
    # tolerate the small differences between hand constants and parameters.py
    assert en.cruise_power(MTOW_TEST) == pytest.approx(hand, rel=1e-2)


# ---------------------------------------------------------------------------
# U7 - scaling (metamorphic)
# ---------------------------------------------------------------------------
def test_cruise_power_linear_in_mass():
    assert en.cruise_power(2 * MTOW_TEST) == pytest.approx(2 * en.cruise_power(MTOW_TEST))


def test_mission_energy_monotonic_in_mass():
    assert en.mission_energy(2200) > en.mission_energy(1980) > en.mission_energy(1700)


def test_battery_mass_monotonic_in_mass():
    # Smooth (no string round-up) so the comparison is not flattened by the step.
    m_lo = en.battery_mass(1700, round_to_strings=False)
    m_hi = en.battery_mass(2200, round_to_strings=False)
    assert m_hi > m_lo


# ---------------------------------------------------------------------------
# U3 - null input
# ---------------------------------------------------------------------------
def test_zero_mtow_zero_propulsive_energy():
    # Every power term carries an MTOW factor, so a zero aircraft costs no
    # propulsive energy (it must not pick up a spurious offset).
    assert en.mission_energy(0.0) == pytest.approx(0.0, abs=1e-6)


def test_zero_mtow_battery_is_aux_floor_only():
    # With no propulsion, the pack still has to carry the fixed aux energy
    # (+reserve, +string round-up), so the mass is a small positive floor.
    m = en.battery_mass(0.0)
    assert m > 0.0
    assert m < en.battery_mass(MTOW_TEST)


# ---------------------------------------------------------------------------
# U4 - extreme value
# ---------------------------------------------------------------------------
def test_extreme_mtow_finite_and_ordered():
    e = en.mission_energy(10_000.0)
    assert math.isfinite(e)
    assert e > en.mission_energy(MTOW_TEST)
    b = en.battery_mass(10_000.0)
    assert math.isfinite(b) and b > en.battery_mass(MTOW_TEST)


# ---------------------------------------------------------------------------
# U8 - conservation: mission energy == sum of its phase contributions
# ---------------------------------------------------------------------------
def test_mission_energy_equals_phase_sum():
    m = MTOW_TEST
    a = en._A_DISK_PROP
    p_to = en.takeoff_power(m, a_disk=a, vs_avg=2, vs_f=3, n_prop=N_PROP)
    p_vc = en.vertical_climb_power(m, a_disk=a, vs=3, n_prop=N_PROP)
    e_cl_acc = en.climb_acceleration_power(m, V_I, v_f=V_CRUISE, vs=3, acc_time=T_CLIMB_ACC)
    p_cl = en.climb_power(m, climb_angle_deg=3.09097, v_climb_horizontal=V_CRUISE, vs=3)
    p_c = en.cruise_power(m)
    p_l = en.landing_power(m, a_disk=a, vs_0=VS_0, n_prop=N_PROP)

    rebuilt = (p_to * T_TAKEOFF
               + p_vc * T_VERTICAL_CLIMB
               + p_c * T_CRUISE
               + p_cl * T_CLIMB
               + e_cl_acc
               + p_l * T_LANDING)
    assert en.mission_energy(m) == pytest.approx(rebuilt)


# ---------------------------------------------------------------------------
# U2 / U7 - battery cell-count chain
# ---------------------------------------------------------------------------
def test_battery_cell_chain_smooth():
    # Wh -> cells -> kg: (E_mission + E_aux)*(1+reserve)/E_cell * m_cell / cell_to_pack
    m = MTOW_TEST
    e_deliv_kwh = (en.mission_energy(m) / 3.6e6 + E_AUX_KWH) * (1.0 + RESERVE_FRAC)
    n_cells = e_deliv_kwh * 1000.0 / E_CELL_WH
    expected = n_cells * M_CELL_KG / CELL_TO_PACK
    assert en.battery_mass(m, round_to_strings=False) == pytest.approx(expected)


def test_battery_built_pack_is_whole_strings():
    # The built pack rounds cells UP to whole 235-cell strings, so the implied
    # cell count is an exact multiple of S_SERIES and >= the smooth count.
    m = MTOW_TEST
    built = en.battery_mass(m, round_to_strings=True)
    smooth = en.battery_mass(m, round_to_strings=False)
    assert built >= smooth
    n_built_cells = built * CELL_TO_PACK / M_CELL_KG
    assert n_built_cells == pytest.approx(round(n_built_cells))
    assert round(n_built_cells) % S_SERIES == 0


# ---------------------------------------------------------------------------
# V - reference cross-check against the optimizer's stored design
# ---------------------------------------------------------------------------
def test_battery_mass_matches_optimizer_output():
    # The optimizer's characteristics.json stores the converged pack mass; the
    # sizing chain here must reproduce it (regression/validation anchor).
    cs = json.loads((PROJECT_ROOT / "final_design/results/data/characteristics.json").read_text())
    mtow = cs["mass"]["mtow"]
    batt = cs["mass"]["battery"]
    assert en.battery_mass(mtow) == pytest.approx(batt, rel=5e-3)
