"""
Verification tests for the aerodynamics block.

Coverage follows the AE row in the report V&V matrix:
U1, U2, U3, U5, U7, U8, U9, U10.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from final_characteristics.aerodynamics.dragpolar import (  # noqa: E402
    DragPolarAnalysis,
    PowerCurveAnalysis,
)
import final_characteristics.matching_diagram as matching  # noqa: E402
from support_files import oswaldefficiency as oswald  # noqa: E402


def _load_module(name: str, relative_path: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def h_b_module():
    return _load_module(
        "h_b_lift_ratio_test",
        "final_characteristics/aerodynamics/h_b determination/h_b_lift_ratio.py",
    )


@pytest.fixture(scope="module")
def airfoil_module():
    return _load_module(
        "airfoil_characteristics_test",
        "final_characteristics/aerodynamics/airfoil_validation/airfoil_characteristics.py",
    )


def _synthetic_dragpolar():
    return DragPolarAnalysis(
        mass_kg=1000.0,
        gravity_m_s2=10.0,
        cruise_altitude_m=0.0,
        stall_density_kg_m3=1.25,
        wing_area_m2=20.0,
        cruise_speed_m_s=50.0,
        cl_max_operational=2.0,
        cl_min=0.0,
        cl_max=1.0,
        cd0=0.02,
        aspect_ratio=8.0,
        oswald_efficiency=0.8,
        n_points=6,
        velocity_m_s=np.array([10.0, 100.0]),
        power_required_w=np.array([100.0, 1000.0]),
        power_available_w=800.0,
    )


def test_ae_u1_1a_drag_polar_equations():
    analysis = _synthetic_dragpolar()
    results = analysis.compute()

    k_expected = 1.0 / (np.pi * analysis.aspect_ratio * analysis.oswald_efficiency)
    q_cruise = 0.5 * results["cruise_density_kg_m3"] * analysis.cruise_speed_m_s**2
    cl_cruise_expected = results["weight_n"] / (q_cruise * analysis.wing_area_m2)
    cd_cruise_expected = analysis.cd0 + k_expected * cl_cruise_expected**2

    assert results["induced_drag_factor"] == pytest.approx(k_expected)
    assert np.allclose(
        results["cd_values"],
        analysis.cd0 + k_expected * results["cl_values"] ** 2,
    )
    assert np.allclose(results["ld_values"], results["cl_values"] / results["cd_values"])
    assert results["cl_cruise"] == pytest.approx(cl_cruise_expected)
    assert results["cd_cruise"] == pytest.approx(cd_cruise_expected)
    assert results["ld_cruise"] == pytest.approx(cl_cruise_expected / cd_cruise_expected)


def test_ae_u1_2a_matching_diagram_equations(monkeypatch):
    monkeypatch.setattr(matching, "_oswald", lambda: 0.75)

    wing_loading = 500.0
    mtow = 1000.0
    stall_expected = 2.0 * 0.5 * 1.2 * 25.0**2
    area_expected = mtow * matching.G / wing_loading
    aspect_expected = matching.SPAN_M**2 / area_expected

    cruise = matching.power_loading_cruise(
        wing_loading,
        mtow=mtow,
        eta_p=0.85,
        cd0=0.03,
        v=60.0,
        rho_cr=0.9,
        rho_sl=1.2,
    )
    parasite = (0.03 * 0.5 * 0.9 * 60.0**3) / wing_loading
    induced = wing_loading / (np.pi * aspect_expected * 0.75 * 0.5 * 0.9 * 60.0)
    cruise_expected = 0.85 * (0.9 / 1.2) ** 0.75 / (parasite + induced)

    climb_rate = matching.power_loading_climb_rate(
        wing_loading,
        mtow=mtow,
        eta_p=0.85,
        cd0=0.03,
        climb_rate=4.0,
        rho_sl=1.2,
    )
    climb_rate_expected = 0.85 / (
        4.0
        + (np.sqrt(wing_loading) * np.sqrt(2.0 / 1.2))
        / (1.345 * (aspect_expected * 0.75) ** 0.75 / 0.03**0.25)
    )

    climb_gradient = matching.power_loading_climb_gradient(
        wing_loading,
        mtow=mtow,
        eta_p=0.85,
        cd0=0.03,
        climb_gradient=0.08,
        cl=1.5,
        rho_sl=1.2,
    )
    cd = 0.03 + 1.5**2 / (np.pi * aspect_expected * 0.75)
    climb_gradient_expected = 0.85 / (
        np.sqrt(wing_loading) * (0.08 + cd / 1.5) * np.sqrt((2.0 / 1.2) / 1.5)
    )

    monkeypatch.setattr(matching, "power_loading_cruise", lambda *_args, **_kwargs: 3.0)
    monkeypatch.setattr(matching, "power_loading_climb_rate", lambda *_args, **_kwargs: 2.0)
    monkeypatch.setattr(matching, "power_loading_climb_gradient", lambda *_args, **_kwargs: 4.0)

    assert matching.wing_loading_stall_limit(2.0, 25.0, 1.2) == pytest.approx(stall_expected)
    assert matching.wing_area(wing_loading, mtow=mtow) == pytest.approx(area_expected)
    assert matching.aspect_ratio(wing_loading, mtow=mtow) == pytest.approx(aspect_expected)
    assert cruise == pytest.approx(cruise_expected)
    assert climb_rate == pytest.approx(climb_rate_expected)
    assert climb_gradient == pytest.approx(climb_gradient_expected)
    assert matching.power_loading_limit(wing_loading, mtow=mtow) == pytest.approx(2.0)


def test_ae_u1_3a_rizzo_oswald_equation():
    heights = np.array([0.0, 2.1])
    span = 13.0
    k_expected = (0.44 + 0.9594 * heights / span) / (0.44 + 2.219 * heights / span)
    e_expected = 1.0 / k_expected

    assert np.allclose(oswald.induced_drag_factor(heights, span), k_expected)
    assert np.allclose(oswald.oswald_efficiency(heights, span), e_expected)
    assert oswald.oswald_efficiency(np.array([0.0]), span)[0] == pytest.approx(1.0)


def test_ae_u2_4a_unit_consistency():
    wing_loading = 500.0
    mtow = 1000.0
    area = matching.wing_area(wing_loading, mtow=mtow)
    h_over_b = 2.1 / 13.0
    results = _synthetic_dragpolar().compute()

    assert area == pytest.approx(mtow * matching.G / wing_loading)
    assert h_over_b == pytest.approx(0.1615384615)
    assert np.isfinite(results["cl_cruise"])
    assert np.isfinite(results["cd_cruise"])
    assert results["induced_drag_factor"] == pytest.approx(
        1.0 / (np.pi * 8.0 * 0.8)
    )


def test_ae_u3_5a_null_inputs():
    analysis = DragPolarAnalysis(
        mass_kg=0.0,
        gravity_m_s2=10.0,
        cruise_altitude_m=0.0,
        stall_density_kg_m3=1.25,
        wing_area_m2=20.0,
        cruise_speed_m_s=50.0,
        cl_max_operational=2.0,
        cl_min=0.0,
        cl_max=1.0,
        cd0=0.02,
        aspect_ratio=8.0,
        oswald_efficiency=0.8,
        n_points=6,
        velocity_m_s=np.array([10.0, 100.0]),
        power_required_w=np.array([100.0, 1000.0]),
        power_available_w=800.0,
    )
    results = analysis.compute()

    assert oswald.oswald_efficiency(np.array([0.0]), 13.0)[0] == pytest.approx(1.0)
    assert results["weight_n"] == 0.0
    assert results["stall_speed_m_s"] == 0.0
    assert results["cl_min_operational"] == 0.0
    assert results["cl_cruise"] == 0.0
    assert results["ld_cruise"] == 0.0


def test_ae_u5_6a_hand_computed_reference():
    power_curve = PowerCurveAnalysis(
        velocity_m_s=np.array([10.0, 20.0]),
        power_required_w=np.array([200.0, 400.0]),
        power_available_w=300.0,
    )
    results = _synthetic_dragpolar().compute()
    analysis = _synthetic_dragpolar()
    density_sl = DragPolarAnalysis.compute_isa_density(0.0)
    cl_cruise_expected = (
        1000.0 * 10.0 / (0.5 * density_sl * 50.0**2 * 20.0)
    )
    cd_cruise_expected = 0.02 + (1.0 / (np.pi * 8.0 * 0.8)) * cl_cruise_expected**2

    assert power_curve.get_max_speed() == pytest.approx(15.0)
    assert results["max_speed_m_s"] == pytest.approx(80.0)
    assert results["cl_cruise"] == pytest.approx(cl_cruise_expected)
    assert results["cd_cruise"] == pytest.approx(cd_cruise_expected)
    assert results["ld_cruise"] == pytest.approx(
        cl_cruise_expected / cd_cruise_expected
    )
    assert analysis.compute()["induced_drag_factor"] == pytest.approx(0.0497359197)


def test_ae_u5_7a_h_b_ratio_reference(h_b_module):
    data_dir = h_b_module.DEFAULT_DATA_DIR
    single_file, two_files = h_b_module.discover_polar_files(
        data_dir=data_dir,
        single_wing_file=None,
        two_wing_glob="Volde h =*.txt",
    )
    single = h_b_module.parse_xflr5_polar(single_file)
    polars = [h_b_module.parse_xflr5_polar(path) for path in two_files]
    rows = h_b_module.build_lift_ratio_rows(
        single,
        polars,
        alpha_deg=0.0,
        span_m=13.0,
        single_wing_factor=1.0,
    )
    row = next(row for row in rows if float(row["h_m"]) == pytest.approx(2.6))
    hand_ratio = row["CL_two_wings"] / row["CL_reference"]

    assert row["h_over_b"] == pytest.approx(0.2)
    assert row["CL_ratio_two_wings_over_main_wing"] == pytest.approx(hand_ratio)
    assert row["CL_ratio_two_wings_over_main_wing"] == pytest.approx(0.9520, rel=5e-5)


def test_ae_u7_8a_wing_loading_scaling():
    mtow = 1000.0
    area_low = matching.wing_area(500.0, mtow=mtow)
    area_high = matching.wing_area(1000.0, mtow=mtow)
    ar_low = matching.aspect_ratio(500.0, mtow=mtow)
    ar_high = matching.aspect_ratio(1000.0, mtow=mtow)

    assert area_high / area_low == pytest.approx(0.5)
    assert ar_high / ar_low == pytest.approx(2.0)


def test_ae_u7_9a_h_b_cl_recovery_monotonic(h_b_module):
    data_dir = h_b_module.DEFAULT_DATA_DIR
    single_file, two_files = h_b_module.discover_polar_files(
        data_dir=data_dir,
        single_wing_file=None,
        two_wing_glob="Volde h =*.txt",
    )
    single = h_b_module.parse_xflr5_polar(single_file)
    polars = [h_b_module.parse_xflr5_polar(path) for path in two_files]

    for alpha in (0.0, 4.0, 8.0):
        rows = h_b_module.build_lift_ratio_rows(
            single,
            polars,
            alpha_deg=alpha,
            span_m=13.0,
            single_wing_factor=1.0,
        )
        ratios = [float(row["CL_ratio_two_wings_over_main_wing"]) for row in rows]
        assert all(a < b for a, b in zip(ratios, ratios[1:]))


def test_ae_u8_10a_h_b_row_coverage(h_b_module):
    data_dir = h_b_module.DEFAULT_DATA_DIR
    single_file, two_files = h_b_module.discover_polar_files(
        data_dir=data_dir,
        single_wing_file=None,
        two_wing_glob="Volde h =*.txt",
    )
    single = h_b_module.parse_xflr5_polar(single_file)
    polars = [h_b_module.parse_xflr5_polar(path) for path in two_files]
    alpha_values = [0.0, 4.0, 8.0]
    rows = []
    for alpha in alpha_values:
        rows.extend(
            h_b_module.build_lift_ratio_rows(
                single,
                polars,
                alpha_deg=alpha,
                span_m=13.0,
                single_wing_factor=1.0,
            )
        )

    assert len(two_files) == 9
    assert len(rows) == len(alpha_values) * len(two_files)
    for alpha in alpha_values:
        files_for_alpha = {
            row["file"] for row in rows if float(row["alpha_deg"]) == alpha
        }
        assert len(files_for_alpha) == len(two_files)


def test_ae_u8_11a_airfoil_comparison_consistency(airfoil_module):
    reference = airfoil_module.load_xfoil_polar(
        airfoil_module.DEFAULT_AIRFOILTOOLS_PATH,
        airfoil_module.REFERENCE_LABEL,
    )
    candidate = airfoil_module.load_xfoil_polar(
        airfoil_module.DEFAULT_FLOW5_PATH,
        airfoil_module.CANDIDATE_LABEL,
    )
    rows, overlap = airfoil_module.build_point_comparison(reference, candidate)
    expected_count = int(
        np.count_nonzero((reference.alpha >= overlap[0]) & (reference.alpha <= overlap[1]))
    )
    row_zero = next(row for row in rows if row["alpha_deg"] == pytest.approx(0.0))

    assert len(rows) == expected_count
    assert row_zero["CL_diff_flow5_minus_airfoiltools"] == pytest.approx(
        row_zero["CL_flow5_interp"] - row_zero["CL_airfoiltools"]
    )
    assert row_zero["CD_diff_drag_counts"] == pytest.approx(
        10000.0
        * (row_zero["CD_flow5_interp"] - row_zero["CD_airfoiltools"])
    )


def test_ae_u9_12a_default_input_ranges(h_b_module, airfoil_module):
    power_curve = PowerCurveAnalysis()
    drag = _synthetic_dragpolar()
    results = drag.compute()

    assert np.all(np.diff(power_curve.velocity_m_s) > 0.0)
    assert drag.cl_min <= results["cl_cruise"] <= drag.cl_max

    _, two_files = h_b_module.discover_polar_files(
        data_dir=h_b_module.DEFAULT_DATA_DIR,
        single_wing_file=None,
        two_wing_glob="Volde h =*.txt",
    )
    h_values = [
        h_b_module.parse_vertical_gap_m(h_b_module.parse_xflr5_polar(path))
        for path in two_files
    ]
    assert min(h_values) <= h_b_module.DEFAULT_DESIGN_GAP_M <= max(h_values)

    reference = airfoil_module.load_xfoil_polar(
        airfoil_module.DEFAULT_AIRFOILTOOLS_PATH,
        airfoil_module.REFERENCE_LABEL,
    )
    candidate = airfoil_module.load_xfoil_polar(
        airfoil_module.DEFAULT_FLOW5_PATH,
        airfoil_module.CANDIDATE_LABEL,
    )
    alpha_min, alpha_max = airfoil_module._alpha_overlap(reference, candidate)
    assert alpha_min <= -2.0 < 6.0 <= alpha_max


def test_ae_u10_13a_output_contracts():
    results = _synthetic_dragpolar().compute()
    expected_keys = {
        "weight_n",
        "cruise_density_kg_m3",
        "stall_density_kg_m3",
        "max_speed_m_s",
        "stall_speed_m_s",
        "cl_min_operational",
        "cl_max_operational",
        "induced_drag_factor",
        "cl_values",
        "cd_values",
        "ld_values",
        "cl_cruise",
        "cd_cruise",
        "ld_cruise",
        "cl_at_max_ld",
        "cd_at_max_ld",
        "max_ld",
        "power_intersection_velocities_m_s",
    }
    wing_loading_range = matching.feasible_wing_loading_range(mtow=1000.0)

    assert expected_keys <= set(results)
    assert all(np.isfinite(results[key]) for key in ("cl_cruise", "cd_cruise", "ld_cruise"))
    assert isinstance(wing_loading_range, tuple)
    assert len(wing_loading_range) == 2
    assert wing_loading_range[0] < wing_loading_range[1]


def test_ae_u10_14a_apply_vlm_aero_output_contract(monkeypatch):
    fake_asb = types.ModuleType("aerosandbox")

    class FakeAtmosphere:
        def __init__(self, altitude):
            self.altitude = altitude

        def speed_of_sound(self):
            return 340.0

        def density(self):
            return 1.0

        def dynamic_viscosity(self):
            return 2.0e-5

    fake_asb.Atmosphere = FakeAtmosphere
    monkeypatch.setitem(sys.modules, "aerosandbox", fake_asb)
    sys.modules.pop("final_characteristics.aero_model", None)
    aero_model = importlib.import_module("final_characteristics.aero_model")

    class WingGeometry:
        S_fw = 10.0
        b_fw = 13.0
        taper_fw = 0.45
        LE_sweep_fw = 0.0
        dihedral_front_wing = 2.0
        airfoil_fw = "NASA LANGLEY LS(1)-0417"
        S_aw = 12.0
        b_aw = 13.0
        taper_aw = 0.45
        LE_sweep_aw = 0.0
        dihedral_aft_wing = 2.0
        airfoil_aw = "NASA LANGLEY LS(1)-0417"

    class Aerodynamics:
        pass

    class Params:
        wing_geometry = WingGeometry()
        aerodynamics = Aerodynamics()

    def fake_wing_aero(area, *_args):
        if area == 10.0:
            return {
                "CL_alpha": 4.1,
                "x_ac": 0.31,
                "cl_alpha_per_deg": 0.101,
                "C_M_ac": -0.11,
            }
        return {
            "CL_alpha": 4.3,
            "x_ac": 0.33,
            "cl_alpha_per_deg": 0.103,
            "C_M_ac": -0.12,
        }

    monkeypatch.setattr(aero_model, "wing_aero", fake_wing_aero)
    params = Params()
    aero_model.apply_vlm_aero(params)

    assert params.aerodynamics.CL_alpha_fw == pytest.approx(4.1)
    assert params.aerodynamics.x_ac_fw_cruise == pytest.approx(0.31)
    assert params.aerodynamics.cl_alpha_fw == pytest.approx(0.101)
    assert params.aerodynamics.C_M_ac_fw == pytest.approx(-0.11)
    assert params.aerodynamics.CL_alpha_aw == pytest.approx(4.3)
    assert params.aerodynamics.x_ac_aw_cruise == pytest.approx(0.33)
    assert params.aerodynamics.cl_alpha_aw == pytest.approx(0.103)
    assert params.aerodynamics.C_M_ac_aw == pytest.approx(-0.12)
