"""
Unit tests for the cabin-pressurization sizing module
(class_II_sizing.pressurization_sizing).

Coverage:
  1. ISA atmosphere - sea-level anchors, monotonic decrease
  2. design differential - zero at/above cruise, positive below, relief factor applied
  3. cabin geometry - pax-driven, sane surfaces/volume
  4. shell penalty - increment-over-baseline logic (zero at small dp, positive at large dp)
  5. air supply - occupant flow vs leakage, zero power without differential, E = P*t
  6. totals - seal floor keeps m_press > 0, battery equivalent, non-circular warning

All quantities SI.
"""

import sys
import warnings
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import class_II_sizing.pressurization_sizing as ps
from parameters import E_CELL_WH, M_CELL_KG, CELL_TO_PACK, RESERVE_FRAC

MTOW_TEST = 2200.0
N_PAX_TEST = 4
T_TEST = 1200.0


# ---------------------------------------------------------------------------
# 1 - ISA atmosphere
# ---------------------------------------------------------------------------
def test_isa_sea_level_anchors():
    assert ps.isa_temperature(0.0) == pytest.approx(288.15)
    assert ps.isa_pressure(0.0) == pytest.approx(101325.0)


def test_isa_monotonic():
    hs = [0.0, 1000.0, 2440.0, 3810.0, 5000.0]
    ps_vals = [ps.isa_pressure(h) for h in hs]
    assert all(a > b for a, b in zip(ps_vals, ps_vals[1:]))


# ---------------------------------------------------------------------------
# 2 - design differential
# ---------------------------------------------------------------------------
def test_dp_zero_when_unpressurized():
    # Cabin at or above cruise altitude -> no differential.
    assert ps.design_dp(3810.0, 3810.0) == 0.0
    assert ps.design_dp(3810.0, 5000.0) == 0.0


def test_dp_positive_and_relief_scaled():
    raw = ps.isa_pressure(2440.0) - ps.isa_pressure(3810.0)
    assert ps.design_dp(3810.0, 2440.0) == pytest.approx(raw * ps.K_RELIEF)
    # Lower cabin altitude -> larger differential (the key sweep direction).
    assert ps.design_dp(3810.0, 1830.0) > ps.design_dp(3810.0, 2440.0)


# ---------------------------------------------------------------------------
# 3 - cabin geometry
# ---------------------------------------------------------------------------
def test_cabin_geometry_pax_driven():
    g4 = ps.cabin_geometry(4)
    g6 = ps.cabin_geometry(6)
    assert g4["r"] > 0 and g4["l_cyl"] > 0 and g4["v_cab"] > 0
    # More pax -> more rows -> longer cabin; radius set by seats abreast, unchanged.
    assert g6["l_cyl"] > g4["l_cyl"]
    assert g6["r"] == pytest.approx(g4["r"])


# ---------------------------------------------------------------------------
# 4 - shell penalty (increment over baseline)
# ---------------------------------------------------------------------------
def test_shell_penalty_zero_at_small_dp():
    # At the FL125 / 8,000 ft design point the required hoop thickness is far below the
    # min-gauge baseline, so the increment must be exactly zero.
    geom = ps.cabin_geometry(N_PAX_TEST)
    dp = ps.design_dp(3810.0, 2440.0)
    dm, detail = ps.shell_penalty(dp, geom, ps.MATERIALS["AL2024"])
    assert detail["t_cyl_req"] < ps.MATERIALS["AL2024"]["t_baseline"]
    assert dm == 0.0


def test_shell_penalty_positive_at_large_dp():
    # At this small cabin radius even a transport-style 55 kPa differential stays below
    # the 2 mm Al min gauge (t = SF*dp*r/sigma_w ~ 0.9 mm), so the increment is zero
    # there too; the penalty must appear once t_required crosses the baseline
    # (dp > sigma_w*t_base/(SF*r) ~ 125 kPa for AL2024 at r = 0.8 m).
    geom = ps.cabin_geometry(N_PAX_TEST)
    mat = ps.MATERIALS["AL2024"]
    dm_55, _ = ps.shell_penalty(55e3, geom, mat)
    assert dm_55 == 0.0
    dm, detail = ps.shell_penalty(200e3, geom, mat)
    assert detail["t_cyl_req"] > mat["t_baseline"]
    assert dm > 0.0
    # Dome carries half the membrane stress.
    assert detail["t_dome_req"] == pytest.approx(detail["t_cyl_req"] / 2.0)


# ---------------------------------------------------------------------------
# 5 - air supply
# ---------------------------------------------------------------------------
def test_air_supply_flow_and_energy():
    geom = ps.cabin_geometry(N_PAX_TEST)
    s = ps.air_supply(N_PAX_TEST, geom["v_cab"], 3810.0, 2440.0, T_TEST)
    # FAR 25.831 occupant flow for this small cabin governs over leakage.
    assert s["mdot"] == pytest.approx((N_PAX_TEST + ps.N_CREW) * ps.MDOT_PER_OCC)
    assert s["mdot_governing"] == "occupants"
    assert s["p_comp_w"] > 0
    assert s["e_press_j"] == pytest.approx(s["p_comp_w"] * T_TEST)
    assert s["m_comp"] >= ps.M_COMP_MIN


def test_air_supply_zero_power_without_differential():
    geom = ps.cabin_geometry(N_PAX_TEST)
    s = ps.air_supply(N_PAX_TEST, geom["v_cab"], 3810.0, 3810.0, T_TEST)
    assert s["p_comp_w"] == 0.0
    assert s["e_press_j"] == 0.0
    assert s["m_comp"] == ps.M_COMP_MIN          # hardware floor remains


# ---------------------------------------------------------------------------
# 6 - totals / public entry
# ---------------------------------------------------------------------------
def test_totals_and_battery_equivalent():
    m, e, info = ps.pressurization_sizing(MTOW_TEST, N_PAX_TEST, T_TEST, "AL2024")
    # Seal/cutout floor keeps the mass nonzero even with a zero shell increment.
    assert info["dm_shell"] == 0.0
    assert info["m_seal"] > 0.0
    assert m == pytest.approx(info["dm_shell"] + info["m_seal"] + info["m_comp"]
                              + info["m_valves"] + info["m_duct"])
    assert m > 0.0
    assert e > 0.0
    # Mirrors energy.battery_mass (smooth cell-count chain, no string round-up).
    expected_dm = (e / 3600.0) * (1.0 + RESERVE_FRAC) / E_CELL_WH * M_CELL_KG / CELL_TO_PACK
    assert info["dm_batt_equiv_kg"] == pytest.approx(expected_dm)
    assert info["mass_fraction"] == pytest.approx(m / MTOW_TEST)


def test_material_table_and_custom_dict():
    m_al, _, _ = ps.pressurization_sizing(MTOW_TEST, N_PAX_TEST, T_TEST, "AL2024")
    custom = dict(sigma_w=100e6, rho=2780.0, t_baseline=ps.MATERIALS["AL2024"]["t_baseline"])
    m_custom, _, _ = ps.pressurization_sizing(MTOW_TEST, N_PAX_TEST, T_TEST, custom)
    assert m_custom == pytest.approx(m_al)


def test_non_circular_warns():
    with pytest.warns(UserWarning, match="circular"):
        ps.pressurization_sizing(MTOW_TEST, N_PAX_TEST, T_TEST, "AL2024",
                                 cross_section="flat_sided")
