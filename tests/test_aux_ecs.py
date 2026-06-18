"""
Unit tests for the auxiliary-load and ECS sizing block -- block ID AX.

Entry points verified:
  final_characteristics.aux_loads.energy_breakdown   (aux electrical energy)
  final_characteristics.ecs_sizing.size_mission       (ECS power + energy)
plus the supporting ISA / compressor / avionics build-up functions they call.

Catalogue coverage:
  U1  Formula inspection   - ISA pressure/temperature anchors; recovery temp
  U2  Unit consistency     - aux energy W -> kWh chain (1/3600, 1/eta_dcdc)
  U3  Null input           - GROUND duty -> 0 energy; no differential -> no comp power
  U5  Hand-computed ref    - single known load energy by hand
  U7  Scaling (metamorphic)- recovery temp and beta_c rise with Mach / pressure ratio
  U8  Conservation         - phase energies and bus energies sum to the total;
                             ECS per-phase energy sums to the mission total

All quantities SI unless a kWh/Wh conversion is noted.
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import final_characteristics.aux_loads as ax
import final_characteristics.ecs_sizing as ecs


# ===========================================================================
# AUXILIARY ELECTRICAL LOADS  (aux_loads.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# U8 - conservation: phase split and bus split both reconstruct the total
# ---------------------------------------------------------------------------
def test_energy_breakdown_conserves_total():
    b = ax.energy_breakdown(ax.default_aux_loads())
    assert sum(b["phase_E"].values()) == pytest.approx(b["total_kWh"])
    assert b["bus_E"][28] + b["bus_E"][270] == pytest.approx(b["total_kWh"])
    assert b["total_kWh"] > 0.0


# ---------------------------------------------------------------------------
# U5 / U2 - single known load, energy by hand (incl. DC/DC loss, Wh->kWh)
# ---------------------------------------------------------------------------
def test_single_always_on_load_energy_by_hand():
    load = ax.AuxLoad("test", 28, 100.0, ax.ALWAYS)
    b = ax.energy_breakdown([load])
    total_seconds = sum(dt for _, dt in ax.PHASES)
    hand_kwh = 100.0 / 1000.0 / ax.ETA_DCDC * (total_seconds / 3600.0)
    assert b["total_kWh"] == pytest.approx(hand_kwh)
    # ...and it all lands on the 28 V bus.
    assert b["bus_E"][28] == pytest.approx(hand_kwh)
    assert b["bus_E"][270] == 0.0


def test_dcdc_loss_inflates_energy():
    # Delivered energy must exceed the ideal bus energy by exactly 1/eta_dcdc.
    load = ax.AuxLoad("test", 270, 1000.0, ax.ALWAYS)
    b = ax.energy_breakdown([load])
    total_seconds = sum(dt for _, dt in ax.PHASES)
    ideal_kwh = 1000.0 / 1000.0 * (total_seconds / 3600.0)
    assert b["total_kWh"] == pytest.approx(ideal_kwh / ax.ETA_DCDC)


# ---------------------------------------------------------------------------
# U3 - null: a ground-only load contributes no in-flight energy
# ---------------------------------------------------------------------------
def test_ground_only_load_contributes_zero():
    load = ax.AuxLoad("folding", 270, 4000.0, ax.GROUND)
    b = ax.energy_breakdown([load])
    assert b["total_kWh"] == pytest.approx(0.0)


def test_active_power_respects_duty():
    load = ax.AuxLoad("probe", 28, 100.0, ax.ICING)
    assert load.active_W("Cruise") == pytest.approx(100.0 * ax.ICING["Cruise"])
    assert load.active_W("Takeoff") == pytest.approx(0.0)   # ICING off at warm SL takeoff


# ===========================================================================
# ECS SIZING  (ecs_sizing.py)
# ===========================================================================

# ---------------------------------------------------------------------------
# U1 - ISA atmosphere anchors and monotonic decay
# ---------------------------------------------------------------------------
def test_isa_sea_level_anchors():
    T, p, _ = ecs.isa(0.0)
    assert T == pytest.approx(288.15)
    assert p == pytest.approx(101325.0)


def test_isa_pressure_monotonic():
    ps = [ecs.isa(h)[1] for h in (0, 4000, 8000, 12500)]
    assert all(a > b for a, b in zip(ps, ps[1:]))


def test_isa_offset_shifts_temperature_only():
    T0, p0, _ = ecs.isa(8000.0, 0.0)
    T1, p1, _ = ecs.isa(8000.0, +25.0)
    assert T1 - T0 == pytest.approx(25.0)
    assert p1 == pytest.approx(p0)   # pressure from the standard profile, unshifted


# ---------------------------------------------------------------------------
# U1 / U7 - recovery temperature: equals static at M=0, rises with Mach
# ---------------------------------------------------------------------------
def test_recovery_temperature():
    T = 250.0
    assert ecs.recovery_temperature(T, 0.0) == pytest.approx(T)
    assert ecs.recovery_temperature(T, 0.3) > T
    assert ecs.recovery_temperature(T, 0.5) > ecs.recovery_temperature(T, 0.3)


# ---------------------------------------------------------------------------
# U3 - compressor: no pressure deficit -> beta_c floored at 1, zero power
# ---------------------------------------------------------------------------
def test_compressor_no_power_without_pressure_ratio():
    cond = ecs.Conditioning()
    # cabin pressure at or below intake pressure -> nothing to compress.
    comp = ecs.compressor_power(0.05, T1_total=300.0, p1_total=101325.0,
                                p_cabin_Pa=80000.0, cond=cond)
    assert comp["beta_c"] == 1.0
    assert comp["P_c_W"] == pytest.approx(0.0)
    assert comp["P_motor_W"] == pytest.approx(0.0)


def test_compressor_motor_power_relationship():
    cond = ecs.Conditioning()
    comp = ecs.compressor_power(0.1, T1_total=260.0, p1_total=60000.0,
                                p_cabin_Pa=81270.0, cond=cond)
    assert comp["beta_c"] > 1.0
    assert comp["P_c_W"] > 0.0
    # Eq.16: motor power = compressor power / (eta_motor * eta_mec).
    assert comp["P_motor_W"] == pytest.approx(
        comp["P_c_W"] / (cond.eta_motor * cond.eta_mec_c))


# ---------------------------------------------------------------------------
# avionics build-up: margin and liquid-cooled exclusion
# ---------------------------------------------------------------------------
def test_sum_avionics_margin_and_liquid_cooling():
    items = [
        ecs.AvionicsItem("air-box", 1, 100.0, liquid_cooled=False),
        ecs.AvionicsItem("liquid-box", 1, 100.0, liquid_cooled=True),
    ]
    s = ecs.sum_avionics(items, margin=0.20)
    assert s["electrical_W"] == pytest.approx(200.0)            # both draw power
    assert s["cabin_heat_W"] == pytest.approx(100.0)            # only the air-cooled box heats the cabin
    assert s["electrical_W_margined"] == pytest.approx(200.0 * 1.20)
    assert s["cabin_heat_W_margined"] == pytest.approx(100.0 * 1.20)


# ---------------------------------------------------------------------------
# U8 - mission driver: sizing power is the per-phase peak; energy is the sum
# ---------------------------------------------------------------------------
def _simple_phases():
    return [
        (ecs.Phase("A", 0, +25, 0.00, "cool", True, True, 0.0), 600.0),
        (ecs.Phase("B", 12500, -20, 0.17, "heat", False, False, 0.0), 1800.0),
        (ecs.Phase("C", 6000, -20, 0.05, "heat", False, True, 0.0), 300.0),
    ]


def test_size_mission_power_is_peak_and_energy_is_sum():
    ac, cond = ecs.Aircraft(), ecs.Conditioning()
    summary = ecs.size_mission(ac, cond, _simple_phases())
    per_phase = summary["per_phase"]

    peak = max(r["P_motor_W"] for r in per_phase)
    assert summary["sizing_motor_power_W"] == pytest.approx(peak)

    energy_sum_kwh = sum(r["energy_Wh"] for r in per_phase) / 1000.0
    assert summary["total_energy_kWh"] == pytest.approx(energy_sum_kwh)
    assert summary["total_energy_kWh"] > 0.0


def test_size_mission_per_phase_energy_definition():
    # Each phase energy [Wh] must equal motor power [W] * duration [s] / 3600.
    ac, cond = ecs.Aircraft(), ecs.Conditioning()
    summary = ecs.size_mission(ac, cond, _simple_phases())
    for r in summary["per_phase"]:
        assert r["energy_Wh"] == pytest.approx(r["P_motor_W"] * r["duration_s"] / 3600.0)


# ---------------------------------------------------------------------------
# U9 - input range / model-validity checks
# ---------------------------------------------------------------------------
def test_duty_fractions_within_unit_range():
    # Every duty cycle is a fraction-active in [0, 1]; anything outside is a
    # data-entry error that would silently corrupt the energy build-up.
    for L in ax.default_aux_loads():
        for p, _ in ax.PHASES:
            assert 0.0 <= L.duty.get(p, 0.0) <= 1.0


def test_ecs_cruise_within_no_acm_regime():
    # The ECS model omits the air-cycle-machine cooling train; the module states
    # this holds while the compressor exit temperature stays modest (< ~60 C).
    # Confirm the sizing (cruise) case is inside that documented validity band.
    ac, cond = ecs.Aircraft(), ecs.Conditioning()
    cruise = ecs.Phase("Cruise", 12500, -20, 0.17, "heat", False, False, 0.0)
    r = ecs.evaluate_phase(ac, cond, cruise)
    assert r["T_comp_exit_C"] < 60.0


# ---------------------------------------------------------------------------
# U10 - output format / downstream interface contract
# ---------------------------------------------------------------------------
def test_energy_breakdown_output_contract():
    b = ax.energy_breakdown(ax.default_aux_loads())
    assert set(b) == {"phase_E", "bus_E", "total_kWh"}
    assert set(b["phase_E"]) == {p for p, _ in ax.PHASES}     # one entry per phase
    assert set(b["bus_E"]) == {28, 270}                       # the two buses
    assert isinstance(b["total_kWh"], float)


def test_size_mission_output_contract():
    ac, cond = ecs.Aircraft(), ecs.Conditioning()
    s = ecs.size_mission(ac, cond, _simple_phases())
    for key in ("per_phase", "sizing_motor_power_W", "sizing_phase",
                "total_energy_kWh", "prop_total_kWh"):
        assert key in s
    assert isinstance(s["sizing_motor_power_W"], float)
    assert isinstance(s["total_energy_kWh"], float)
    # Each per-phase row must carry the fields the report table / battery budget read.
    for r in s["per_phase"]:
        for fld in ("phase", "P_motor_W", "energy_Wh", "duration_s"):
            assert fld in r
