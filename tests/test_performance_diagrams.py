"""
Unit tests for the wing-borne performance diagrams -- block ID VD.

Covers the four diagram scripts whose physics must be self-consistent:
  final_characteristics/vn_diagram.py                (V-n manoeuvre + gust)
  final_characteristics/vehicle_dynamics/turnperformance.py
  final_characteristics/vehicle_dynamics/climb.py
  class_II_sizing/payloadrange.py

Each is a runnable script (guarded by __main__), so it is loaded here by file
path and its module-level quantities / helper functions are checked directly;
main() (which only draws) is never called.

Catalogue coverage:
  U1  Formula inspection   - corner speed, stall speed, load-factor relations
  U5  Hand-computed ref    - n at the corner speed equals the structural limit
  U7  Scaling (metamorphic)- gust n rises with V and gust velocity; range falls
                             with mass; required power has an interior minimum
  U8  Consistency          - gust lines symmetric about n=1; payload curve anchored
"""

import importlib.util
import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load(mod_name, rel_path):
    """Import a runnable script by file path without executing its main()."""
    spec = importlib.util.spec_from_file_location(mod_name, PROJECT_ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


vn = _load("vd_vn", "final_characteristics/vn_diagram.py")
turn = _load("vd_turn", "final_characteristics/vehicle_dynamics/turnperformance.py")
climb = _load("vd_climb", "final_characteristics/vehicle_dynamics/climb.py")
prng = _load("vd_payloadrange", "class_II_sizing/payloadrange.py")


# ===========================================================================
# V-n DIAGRAM
# ===========================================================================
def test_vn_corner_speed_formula():
    # V_A = V_S * sqrt(n_max)  (corner speed definition).
    assert vn.V_A == pytest.approx(vn.V_S * math.sqrt(vn.N_POS))


def test_vn_load_factor_at_corner_equals_limit():
    # Hand reference: at the corner speed the lift-limited n hits the structural limit.
    assert vn.n_stall_pos(vn.V_A) == pytest.approx(vn.N_POS)


def test_vn_dive_speed_from_factor():
    assert vn.V_DIVE == pytest.approx(vn.VD_FACTOR * vn.V_CRUISE)


def test_vn_gust_increases_with_speed_and_velocity():
    # Gust load factor is linear in V and in U_de, so both raise it above 1g.
    assert vn.n_gust(vn.V_CRUISE, 15.0) > vn.n_gust(0.5 * vn.V_CRUISE, 15.0) > 1.0
    assert vn.n_gust(vn.V_CRUISE, 20.0) > vn.n_gust(vn.V_CRUISE, 15.0)


def test_vn_gust_symmetric_about_1g():
    up = vn.n_gust(vn.V_CRUISE, 15.0, +1)
    dn = vn.n_gust(vn.V_CRUISE, 15.0, -1)
    assert up + dn == pytest.approx(2.0)


# ===========================================================================
# TURN PERFORMANCE
# ===========================================================================
def test_turn_stall_speed_formula():
    assert turn.v_stall(turn.RHO_CR) == pytest.approx(
        math.sqrt(2 * turn.WS / (turn.RHO_CR * turn.CL_MAX)))


def test_turn_corner_lift_equals_structural_limit():
    VA = turn.v_stall(turn.RHO_CR) * math.sqrt(turn.N_LIM)
    assert turn.n_lift(VA, turn.RHO_CR) == pytest.approx(turn.N_LIM)


def test_turn_geometry_bank_angle():
    # n = 1/cos(phi): n=2 -> 60 deg, n=sqrt(2) -> 45 deg.
    phi2, _, _ = turn.turn_geometry(60.0, 2.0)
    assert phi2 == pytest.approx(60.0)
    phi45, _, _ = turn.turn_geometry(60.0, math.sqrt(2.0))
    assert phi45 == pytest.approx(45.0)


def test_turn_radius_formula():
    V, n = 55.0, 2.0
    _, R, _ = turn.turn_geometry(V, n)
    assert R == pytest.approx(V**2 / (turn.G * math.sqrt(n**2 - 1.0)))


# ===========================================================================
# CLIMB / DESCENT
# ===========================================================================
def test_climb_required_power_positive_at_cruise():
    pr = climb.P_required(climb.V_C)
    assert math.isfinite(pr) and pr > 0.0


def test_climb_required_power_has_interior_minimum():
    # P_r is U-shaped (induced term dominates low, parasite high), so the
    # minimum over the band is strictly below both endpoints -> V_y is locatable.
    vs = [20.0, 30.0, 37.0, 50.0, 70.0]
    prs = [climb.P_required(v) for v in vs]
    assert min(prs) < prs[0]
    assert min(prs) < prs[-1]


def test_climb_roc_definition():
    V = climb.V_C
    assert climb.roc(V) == pytest.approx((climb.P_AVAIL_REF - climb.P_required(V)) / climb.W)


# ===========================================================================
# PAYLOAD-RANGE
# ===========================================================================
def test_payload_range_lighter_flies_further():
    # Fixed pack: less mass -> less cruise power -> more range.
    assert prng.cruise_range_km(prng.MTOW_DESIGN - 200) > prng.cruise_range_km(prng.MTOW_DESIGN)


def test_payload_range_curve_anchored_and_monotonic():
    rows = prng.payload_range_curve()       # ordered 4 pax -> 0 pax
    assert rows[0][0] == 4
    # 4-pax point is anchored at the design range.
    assert rows[0][3] == pytest.approx(prng.DESIGN_RANGE_KM)
    # Range grows monotonically as passengers are shed.
    ranges = [r[3] for r in rows]
    assert all(b > a for a, b in zip(ranges, ranges[1:]))


# ===========================================================================
# U2 - unit consistency (EAS/TAS conversion done in the right units)
# ===========================================================================
def test_vn_eas_tas_conversion():
    # The V-n envelope is sea-level EAS; cruise is a cruise-altitude TAS placed
    # on it via V_EAS = V_TAS * sqrt(rho_alt / rho_SL). Recompute independently.
    expected = vn.V_C_TAS * math.sqrt(vn.RHO_CR / vn.RHO)
    assert vn.V_CRUISE == pytest.approx(expected)
    assert vn.V_CRUISE < vn.V_C_TAS          # EAS below TAS at altitude (rho < rho_SL)


# ===========================================================================
# U3 - null input (zero speed -> no aerodynamic load)
# ===========================================================================
def test_vn_zero_speed_no_load():
    assert vn.n_stall_pos(0.0) == pytest.approx(0.0)   # no dynamic pressure, no lift
    assert vn.n_gust(0.0, 15.0) == pytest.approx(1.0)  # no gust increment at V=0 -> 1 g


# ===========================================================================
# U4 - extreme value (large speeds stay finite and physically ordered)
# ===========================================================================
def test_diagrams_finite_at_extreme_speed():
    assert math.isfinite(vn.n_gust(500.0, 15.0))
    assert vn.n_stall_pos(500.0) > vn.n_stall_pos(100.0)        # monotone, no overflow
    pr_hi = climb.P_required(200.0)
    assert math.isfinite(pr_hi) and pr_hi > 0.0


# ===========================================================================
# U9 - input range / valid-envelope checks
# ===========================================================================
def test_vn_inputs_in_valid_range():
    # CL_max for the box wing should be a sane high-lift value (XFLR5 gave 1.686).
    assert 1.0 < vn.CL_MAX < 2.5
    # Characteristic speeds must be correctly ordered for a valid envelope.
    assert vn.V_S < vn.V_A < vn.V_DIVE
    assert vn.V_S < vn.V_CRUISE < vn.V_DIVE
