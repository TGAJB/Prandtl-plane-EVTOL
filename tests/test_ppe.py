"""
Unit tests for the propulsion performance estimation and aeroacoustics block -- block ID PP.

ppe.py implements:
  - Polar   : tabulated airfoil Cl/Cd with stall detection
  - Blade   : discretized blade geometry
  - bemt_hover  : hover BEMT (combined-inflow equation + momentum power)
  - bemt_cruise : forward-flight BEMT with iterated induced inflow
  - noise_spl   : JPL TR 32-1462 far-field SPL
  - pwl_regression : Wang et al. 2025 A-weighted sound power level
  - spl_to_dba  : overall SPL -> dBA via harmonic spectrum + A-weighting

Catalogue coverage (unit test type catalogue, Table 10.2):
  U1  Formula inspection   - CT, CP, J, eta (cruise) and FoM, P_hover (hover)
                             verified symbol-for-symbol against standard definitions
  U2  Unit consistency     - CT/CP/J dimensionless; FoM in (0,1]; Vtip in m/s;
                             blade-passage frequency in Hz
  U3  Null input           - V_axial=0 in cruise -> J=0, eta=0 (no spurious values)
  U4  Extreme value        - very high RPM stays finite and increases monotonically
  U5  Hand-computed ref    - hover P and cruise CT/CP/J/eta reproduced within 0.1%
  U7  Scaling (metamorphic)- doubling RPM ~quadruples hover thrust and ~octets power;
                             doubling shaft power raises takeoff PWL by 3.2*log10(2) dB
  U9  Input range          - design Mtip within [0,1]; disk loading inside eVTOL regime
  U10 Output format        - bemt_hover / bemt_cruise / pwl_regression / spl_to_dba
                             return the keys/types that downstream consumers expect

Test IDs follow the convention PP-U<n>-<seq><iter>:
  PP  = propulsion performance block
  U<n>= unit test type from the catalogue
  seq = sequential test number within this file
  iter= attempt letter (a = first attempt)
"""

import math
import sys
from pathlib import Path

import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import final_characteristics.ppe as ppe

# ---------------------------------------------------------------------------
# Shared fixture: a Blade instance with the module-level design parameters.
# All BEMT tests use this blade so inputs are consistent and realistic.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def blade():
    return ppe.Blade()


# ===========================================================================
# U1 -- Formula Inspection
# ===========================================================================

def test_bemt_cruise_CT_definition(blade):
    """PP-U1-1a: CT = T / (rho * n^2 * D^4) symbol-for-symbol."""
    rho = ppe.RHO
    n_rpm, V, pitch = 2000, 55.5, 35.0
    res = ppe.bemt_cruise(blade, n_rpm, V, pitch)
    n_rev = n_rpm / 60.0
    D = 2 * blade.R
    CT_hand = res["T"] / (rho * n_rev**2 * D**4)
    assert res["CT"] == pytest.approx(CT_hand, rel=1e-6)


def test_bemt_cruise_CP_definition(blade):
    """PP-U1-2a: CP = P / (rho * n^3 * D^5) symbol-for-symbol."""
    rho = ppe.RHO
    n_rpm, V, pitch = 2000, 55.5, 35.0
    res = ppe.bemt_cruise(blade, n_rpm, V, pitch)
    n_rev = n_rpm / 60.0
    D = 2 * blade.R
    CP_hand = res["P"] / (rho * n_rev**3 * D**5)
    assert res["CP"] == pytest.approx(CP_hand, rel=1e-6)


def test_bemt_cruise_J_definition(blade):
    """PP-U1-3a: J = V_axial / (n * D) symbol-for-symbol."""
    n_rpm, V, pitch = 2000, 55.5, 35.0
    res = ppe.bemt_cruise(blade, n_rpm, V, pitch)
    n_rev = n_rpm / 60.0
    D = 2 * blade.R
    J_hand = V / (n_rev * D)
    assert res["J"] == pytest.approx(J_hand, rel=1e-6)


def test_bemt_cruise_eta_definition(blade):
    """PP-U1-4a: eta = CT * J / CP when CP > 0 and J > 0."""
    n_rpm, V, pitch = 2000, 55.5, 35.0
    res = ppe.bemt_cruise(blade, n_rpm, V, pitch)
    assert res["CP"] > 0 and res["J"] > 0
    eta_hand = res["CT"] * res["J"] / res["CP"]
    assert res["eta"] == pytest.approx(eta_hand, rel=1e-6)


def test_bemt_hover_power_momentum_formula(blade):
    """PP-U1-5a: hover P = T^1.5 / sqrt(2*rho*A) / FOM_ASSUMED (momentum theory)."""
    rho = ppe.RHO
    res = ppe.bemt_hover(blade, n_rpm=1500, collective_deg=20)
    T = res["T"]
    P_hand = T**1.5 / math.sqrt(2 * rho * blade.A) / ppe.FOM_ASSUMED
    assert res["P"] == pytest.approx(P_hand, rel=1e-3)


# ===========================================================================
# U2 -- Unit Consistency
# ===========================================================================

def test_cruise_coefficients_dimensionless(blade):
    """PP-U2-1a: CT, CP, J, eta are dimensionless and in physically reasonable ranges."""
    res = ppe.bemt_cruise(blade, n_rpm=2000, V_axial=55.5, collective_deg=35)
    # CT for this rotor should be in [0, 0.5] (typical propeller range)
    assert 0.0 < res["CT"] < 0.5
    # CP similarly bounded (power coefficient)
    assert 0.0 < res["CP"] < 1.0
    # J (advance ratio) at cruise is usually 0.5 -- 2.0
    assert 0.0 < res["J"] < 5.0
    # Propulsive efficiency must be sub-unity
    assert 0.0 < res["eta"] < 1.0


def test_hover_fom_in_unit_interval(blade):
    """PP-U2-2a: FoM (figure of merit) is in (0, 1]."""
    res = ppe.bemt_hover(blade, n_rpm=1500, collective_deg=20)
    assert 0.0 < res["FoM"] <= 1.0


def test_vtip_in_ms(blade):
    """PP-U2-3a: Vtip [m/s] lies between 0 and the speed of sound."""
    res = ppe.bemt_hover(blade, n_rpm=1500, collective_deg=20)
    assert 0.0 < res["Vtip"] < ppe.SOUND_SPEED


def test_blade_passage_frequency_in_hz(blade):
    """PP-U2-4a: f1 = B * n_rpm/60 is in Hz and positive for n_rpm > 0."""
    n_rpm = 1500
    f1 = blade.B * n_rpm / 60.0
    assert f1 > 0.0
    # For 8 blades at 1500 rpm: f1 = 8 * 25 = 200 Hz
    assert f1 == pytest.approx(200.0)


# ===========================================================================
# U3 -- Null Input Test
# ===========================================================================

def test_cruise_zero_axial_speed_gives_zero_J_eta(blade):
    """PP-U3-1a: V_axial=0 -> J=0 and eta=0 (no spurious advance/efficiency)."""
    res = ppe.bemt_cruise(blade, n_rpm=1500, V_axial=0.0, collective_deg=30)
    assert res["J"] == pytest.approx(0.0)
    assert res["eta"] == pytest.approx(0.0)


# ===========================================================================
# U4 -- Extreme Value Test
# ===========================================================================

def test_hover_high_rpm_finite(blade):
    """PP-U4-1a: very high RPM (near tip-Mach limit) yields finite, positive outputs."""
    # 5000 rpm -> Vtip ≈ 497 m/s, well above M_TIP_MAX but still a valid BEMT call
    res = ppe.bemt_hover(blade, n_rpm=5000, collective_deg=10)
    assert math.isfinite(res["T"])
    assert math.isfinite(res["P"])
    assert res["T"] > 0.0
    assert res["P"] > 0.0


def test_hover_thrust_monotonic_with_rpm(blade):
    """PP-U4-2a: thrust increases monotonically with RPM at constant pitch."""
    T_values = [ppe.bemt_hover(blade, n_rpm=n, collective_deg=20)["T"]
                for n in (800, 1200, 1600, 2000)]
    assert all(a < b for a, b in zip(T_values, T_values[1:]))


# ===========================================================================
# U5 -- Hand-Computed Reference
# ===========================================================================

def test_hover_power_hand_check(blade):
    """PP-U5-1a: hover P matches T^1.5/(sqrt(2*rho*A)*FoM) within 0.1%."""
    rho = ppe.RHO
    n_rpm, pitch = 1500, 20
    res = ppe.bemt_hover(blade, n_rpm, pitch)
    P_hand = res["T"]**1.5 / math.sqrt(2 * rho * blade.A) / ppe.FOM_ASSUMED
    assert res["P"] == pytest.approx(P_hand, rel=1e-3)


def test_cruise_coefficient_hand_check(blade):
    """PP-U5-2a: cruise CT, CP, J, eta all match their definitions within 0.1%."""
    rho = ppe.RHO
    n_rpm, V, pitch = 2000, 55.5, 35.0
    res = ppe.bemt_cruise(blade, n_rpm, V, pitch)
    n_rev = n_rpm / 60.0
    D = 2 * blade.R
    CT_hand = res["T"] / (rho * n_rev**2 * D**4)
    CP_hand = res["P"] / (rho * n_rev**3 * D**5)
    J_hand  = V / (n_rev * D)
    eta_hand = CT_hand * J_hand / CP_hand
    assert res["CT"]  == pytest.approx(CT_hand,  rel=1e-3)
    assert res["CP"]  == pytest.approx(CP_hand,  rel=1e-3)
    assert res["J"]   == pytest.approx(J_hand,   rel=1e-3)
    assert res["eta"] == pytest.approx(eta_hand, rel=1e-3)


# ===========================================================================
# U7 -- Scaling / Metamorphic Tests
# ===========================================================================

def test_hover_thrust_scales_with_rpm_squared(blade):
    """PP-U7-1a: doubling RPM approximately quadruples hover thrust (T ~ Vtip^2)."""
    T_lo = ppe.bemt_hover(blade, n_rpm=1000, collective_deg=20)["T"]
    T_hi = ppe.bemt_hover(blade, n_rpm=2000, collective_deg=20)["T"]
    assert T_hi / T_lo == pytest.approx(4.0, rel=1e-2)


def test_hover_power_scales_with_rpm_cubed(blade):
    """PP-U7-2a: doubling RPM approximately octets hover power (P ~ Vtip^3)."""
    P_lo = ppe.bemt_hover(blade, n_rpm=1000, collective_deg=20)["P"]
    P_hi = ppe.bemt_hover(blade, n_rpm=2000, collective_deg=20)["P"]
    assert P_hi / P_lo == pytest.approx(8.0, rel=1e-2)


def test_pwl_takeoff_ps_scaling(blade):
    """PP-U7-3a: doubling shaft power raises takeoff PWL by 3.2*log10(2) dB."""
    res = ppe.bemt_hover(blade, n_rpm=1500, collective_deg=20)
    res_double = dict(res)
    res_double["P"] = res["P"] * 2.0

    pwl1 = ppe.pwl_regression(res,        blade, stage="takeoff")
    pwl2 = ppe.pwl_regression(res_double, blade, stage="takeoff")
    expected_delta = 3.2 * np.log10(2.0)
    assert (pwl2 - pwl1) == pytest.approx(expected_delta, abs=1e-3)


# ===========================================================================
# U9 -- Input Range Check
# ===========================================================================

def test_design_tip_mach_within_limit(blade):
    """PP-U9-1a: tip Mach at the sized hover point is within the model's valid range [0, M_TIP_MAX]."""
    res = ppe.bemt_hover(blade, n_rpm=1500, collective_deg=20)
    Mt = res["Vtip"] / ppe.SOUND_SPEED
    assert 0.0 < Mt <= ppe.M_TIP_MAX


def test_design_disk_loading_in_evtol_range(blade):
    """PP-U9-2a: disk loading at the design hover thrust is in the eVTOL regime (50-600 kg/m^2)."""
    DL_kgm2 = ppe.T_HOVER_PROP / (blade.A * 9.81)
    assert 50.0 <= DL_kgm2 <= 600.0


def test_pwl_regression_rejects_invalid_stage(blade):
    """PP-U9-3a: pwl_regression raises ValueError for an unrecognised stage name."""
    res = ppe.bemt_hover(blade, n_rpm=1500, collective_deg=20)
    with pytest.raises(ValueError):
        ppe.pwl_regression(res, blade, stage="invalid")


# ===========================================================================
# U10 -- Output Format / Interface Contract
# ===========================================================================

def test_bemt_hover_output_contract(blade):
    """PP-U10-1a: bemt_hover returns a dict with all keys that downstream consumers read."""
    res = ppe.bemt_hover(blade, n_rpm=1500, collective_deg=20)
    required = {"T", "P", "P_bemt", "FoM", "FoM_bemt", "CT", "Vtip", "n_rpm", "alpha_max"}
    assert required.issubset(res.keys())
    assert isinstance(res["T"], float)
    assert isinstance(res["P"], float)
    assert isinstance(res["FoM"], float)


def test_bemt_cruise_output_contract(blade):
    """PP-U10-2a: bemt_cruise returns a dict with all keys that downstream consumers read."""
    res = ppe.bemt_cruise(blade, n_rpm=2000, V_axial=55.5, collective_deg=35)
    required = {"T", "P", "CT", "CP", "J", "eta", "Vtip", "n_rpm", "alpha_max"}
    assert required.issubset(res.keys())
    assert isinstance(res["CT"], float)
    assert isinstance(res["J"], float)
    assert isinstance(res["eta"], float)


def test_pwl_regression_returns_float(blade):
    """PP-U10-3a: pwl_regression returns a single scalar (float or numpy scalar)."""
    res = ppe.bemt_cruise(blade, n_rpm=2000, V_axial=55.5, collective_deg=35)
    pwl = ppe.pwl_regression(res, blade, stage="cruise")
    assert math.isfinite(float(pwl))


def test_spl_to_dba_output_contract(blade):
    """PP-U10-4a: spl_to_dba returns (float dBA, list of band tuples)."""
    res = ppe.bemt_cruise(blade, n_rpm=2000, V_axial=55.5, collective_deg=35)
    dba, table = ppe.spl_to_dba(res, blade, overall_spl_db=85.0)
    assert math.isfinite(float(dba))
    assert isinstance(table, list)
    # Each band tuple has (center_freq, band_spl, band_spl_A)
    assert len(table) > 0
    assert all(len(row) == 3 for row in table)


# ===========================================================================
# NO BLOCK — Noise functions in ppe.py
# (JPL TR 32-1462 noise_spl, Wang 2025 pwl_regression,
#  pwl_to_spl_directional, spl_to_dba)
#
# Test IDs: NO-U<n>-<seq>a
# Catalogue coverage:
#   U1  Formula inspection  - pwl_regression matches Wang 2025 Eq. 13 & 14
#   U2  Unit consistency    - A-weight at 1 kHz = 0 dB; f1 in Hz; noise_spl in dB
#   U3  Null input          - P=0 or Vtip=0 -> NaN (documented graceful return)
#   U4  Extreme value       - very high power stays finite
#   U5  Hand-computed ref   - takeoff PWL reproduced symbol-for-symbol within 0.01 dB
#   U7  Scaling             - doubling observer distance drops SPL by 20*log10 factor
#   U9  Input range         - design Mt in JPL interpolation range [0.2,1.0];
#                             theta in directivity table range [20,180] deg
#   U10 Output format       - noise_spl and pwl_to_spl_directional return finite scalars
# ===========================================================================

# ---------------------------------------------------------------------------
# NO-U1 -- Formula Inspection: pwl_regression matches Wang 2025 Eq. 13 & 14
# ---------------------------------------------------------------------------

def test_pwl_regression_takeoff_formula(blade):
    """NO-U1-1a: takeoff PWL matches Wang 2025 Eq. 13 symbol-for-symbol."""
    res = ppe.bemt_hover(blade, n_rpm=1500, collective_deg=20)
    D = 2 * blade.R
    B = blade.B
    Ps = res["P"] / ppe.FOM_ASSUMED
    Mt = res["Vtip"] / ppe.SOUND_SPEED
    Nprop = ppe.N_PROPS
    pwl_hand = (3.2 * np.log10(Ps)
                - 20.3 * np.log10(D)
                + 60.0 * np.log10(Mt)
                - 2.9 * B
                + 10.0 * np.log10(Nprop)
                + 124.1)
    assert ppe.pwl_regression(res, blade, stage="takeoff") == pytest.approx(pwl_hand, abs=1e-4)


def test_pwl_regression_cruise_formula(blade):
    """NO-U1-2a: cruise PWL matches Wang 2025 Eq. 14 symbol-for-symbol."""
    res = ppe.bemt_cruise(blade, n_rpm=2000, V_axial=55.5, collective_deg=35)
    D = 2 * blade.R
    B = blade.B
    Mt = res["Vtip"] / ppe.SOUND_SPEED
    Nprop = ppe.N_PROPS
    pwl_hand = (14.1 * np.log10(D)
                + 60.0 * np.log10(Mt)
                + 0.5 * B
                + 10.0 * np.log10(Nprop)
                + 102.2)
    assert ppe.pwl_regression(res, blade, stage="cruise") == pytest.approx(pwl_hand, abs=1e-4)


# ---------------------------------------------------------------------------
# NO-U2 -- Unit Consistency
# ---------------------------------------------------------------------------

def test_a_weighting_at_1kHz_is_zero(blade):
    """NO-U2-3a: A-weighting gain at 1000 Hz = 0.0 dB (IEC 61672 standard reference)."""
    idx = list(ppe._AW_FREQ).index(1000)
    assert ppe._AW_GAIN[idx] == pytest.approx(0.0)


def test_noise_spl_returns_dB_scalar(blade):
    """NO-U2-4a: noise_spl returns a finite scalar in dB (not an array or dBA)."""
    res = ppe.bemt_hover(blade, n_rpm=1500, collective_deg=20)
    spl = ppe.noise_spl(res, blade, r_ft=50.0, theta_deg=90.0)
    assert math.isfinite(float(spl))
    # dB value is positive (acoustic SPL above 0 dB for a powered propeller)
    assert float(spl) > 0.0


def test_blade_passage_frequency_positive_hz(blade):
    """NO-U2-5a: blade-passage frequency f1 = B*n/60 is positive and in Hz."""
    n_rpm = 1500
    f1 = blade.B * n_rpm / 60.0
    assert f1 > 0.0
    assert f1 == pytest.approx(200.0)   # 8 blades * 1500 rpm / 60 = 200 Hz


# ---------------------------------------------------------------------------
# NO-U3 -- Null Input Test
# ---------------------------------------------------------------------------

def test_pwl_zero_power_returns_nan(blade):
    """NO-U3-6a: P=0 (Ps=0) -> pwl_regression returns NaN (logged in model docs)."""
    res = ppe.bemt_hover(blade, n_rpm=1500, collective_deg=20)
    res_zero = dict(res)
    res_zero["P"] = 0.0
    result = ppe.pwl_regression(res_zero, blade, stage="takeoff")
    assert math.isnan(float(result))


def test_pwl_zero_vtip_returns_nan(blade):
    """NO-U3-7a: Vtip=0 (Mt=0) -> pwl_regression returns NaN."""
    res = ppe.bemt_hover(blade, n_rpm=1500, collective_deg=20)
    res_zero = dict(res)
    res_zero["Vtip"] = 0.0
    result = ppe.pwl_regression(res_zero, blade, stage="cruise")
    assert math.isnan(float(result))


# ---------------------------------------------------------------------------
# NO-U4 -- Extreme Value Test
# ---------------------------------------------------------------------------

def test_pwl_very_high_power_stays_finite(blade):
    """NO-U4-8a: shaft power of 1 GW (extreme) gives finite PWL in dBA."""
    res = ppe.bemt_hover(blade, n_rpm=1500, collective_deg=20)
    res_big = dict(res)
    res_big["P"] = 1e9   # 1 GW — physically absurd but numerically valid
    pwl = ppe.pwl_regression(res_big, blade, stage="takeoff")
    assert math.isfinite(float(pwl))
    # PWL must be larger than the design point (more power -> more noise)
    pwl_design = ppe.pwl_regression(res, blade, stage="takeoff")
    assert float(pwl) > float(pwl_design)


# ---------------------------------------------------------------------------
# NO-U5 -- Hand-Computed Reference
# ---------------------------------------------------------------------------

def test_pwl_takeoff_hand_computed(blade):
    """NO-U5-9a: takeoff PWL matches a full pencil calculation within 0.01 dB.

    Hand calc (Wang 2025 Eq. 13) at n=1500 rpm, pitch=20 deg, hover:
      Ps = 54 582.8 W,  D = 1.9 m,  Mt = 0.4351,  B = 8,  Nprop = 6
      PWL = 3.2*log10(54582.8) - 20.3*log10(1.9) + 60*log10(0.4351)
            - 2.9*8 + 10*log10(6) + 124.1
          = 15.159 - 5.659 - 21.687 - 23.2 + 7.782 + 124.1
          = 96.494 dBA
    """
    res = ppe.bemt_hover(blade, n_rpm=1500, collective_deg=20)
    pwl = ppe.pwl_regression(res, blade, stage="takeoff")
    assert float(pwl) == pytest.approx(96.494, abs=0.01)


# ---------------------------------------------------------------------------
# NO-U7 -- Scaling Tests
# ---------------------------------------------------------------------------

def test_noise_spl_distance_scaling(blade):
    """NO-U7-10a: doubling observer distance reduces noise_spl by 20*log10((r2-1)/(r1-1)) dB."""
    res = ppe.bemt_cruise(blade, n_rpm=2000, V_axial=55.5, collective_deg=35)
    r1, r2 = 50.0, 100.0    # ft
    spl1 = ppe.noise_spl(res, blade, r_ft=r1, theta_deg=90.0)
    spl2 = ppe.noise_spl(res, blade, r_ft=r2, theta_deg=90.0)
    expected_drop = 20.0 * np.log10((r2 - 1.0) / (r1 - 1.0))
    assert (spl1 - spl2) == pytest.approx(expected_drop, abs=1e-3)


def test_pwl_to_spl_directional_inverse_square(blade):
    """NO-U7-11a: doubling r drops directional SPL by 10*log10(4)≈6.02 dB (hemisphere)."""
    res = ppe.bemt_cruise(blade, n_rpm=2000, V_axial=55.5, collective_deg=35)
    pwl = ppe.pwl_regression(res, blade, stage="cruise")
    r1, r2 = 15.0, 30.0   # m
    spl1 = ppe.pwl_to_spl_directional(pwl, r_m=r1, theta_deg=120.0)
    spl2 = ppe.pwl_to_spl_directional(pwl, r_m=r2, theta_deg=120.0)
    expected_drop = 10.0 * np.log10((r2 / r1) ** 2)
    assert (spl1 - spl2) == pytest.approx(expected_drop, abs=1e-2)


# ---------------------------------------------------------------------------
# NO-U9 -- Input Range Check
# ---------------------------------------------------------------------------

def test_design_mtip_within_jpl_interpolation_range(blade):
    """NO-U9-12a: design Mtip lies within the JPL model's interpolation table [0.2, 1.0]."""
    res = ppe.bemt_hover(blade, n_rpm=1500, collective_deg=20)
    Mt = res["Vtip"] / ppe.SOUND_SPEED
    assert 0.2 <= Mt <= 1.0


def test_observer_theta_within_directivity_range(blade):
    """NO-U9-13a: default observer angle (NOISE_OBS_THETA) is within the JPL directivity
    table range [20, 180] deg so no extrapolation occurs."""
    assert 20.0 <= ppe.NOISE_OBS_THETA <= 180.0


# ---------------------------------------------------------------------------
# NO-U10 -- Output Format / Interface Contract
# ---------------------------------------------------------------------------

def test_noise_spl_output_is_finite_scalar(blade):
    """NO-U10-14a: noise_spl returns a single finite float in dB."""
    res = ppe.bemt_hover(blade, n_rpm=1500, collective_deg=20)
    spl = ppe.noise_spl(res, blade,
                        r_ft=ppe.NOISE_OBS_DIST_M * ppe.FT_PER_M,
                        theta_deg=ppe.NOISE_OBS_THETA)
    assert math.isfinite(float(spl))
    assert not isinstance(spl, (list, np.ndarray))


def test_pwl_to_spl_directional_output_is_finite_scalar(blade):
    """NO-U10-15a: pwl_to_spl_directional returns a single finite float in dBA."""
    res = ppe.bemt_cruise(blade, n_rpm=2000, V_axial=55.5, collective_deg=35)
    pwl = ppe.pwl_regression(res, blade, stage="cruise")
    spl = ppe.pwl_to_spl_directional(pwl, r_m=ppe.NOISE_OBS_DIST_M,
                                     theta_deg=ppe.NOISE_OBS_THETA)
    assert math.isfinite(float(spl))
    assert not isinstance(spl, (list, np.ndarray))
