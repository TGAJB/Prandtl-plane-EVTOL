"""
Unit tests for the condition-specific skid landing-gear sizing
(class_II_sizing.mass_components). One test per revision change:

  1. elastic (triangular) vs plastic (rectangular) energy balance
  2. CFRP restricted to the elastic branch; plastic branch forbidden
  3. STRUCT_SF removed from the plastic collapse force -> honest load factor n
  4. gear is not double-counted in the mass loop; seed-independent fixed point
  5. a feasible section exists with the calibrated bent-arm L_EFF (~0.4 m)

All quantities are SI.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import class_II_sizing.mass_components as mc
from parameters import (
    RHO_AL, SIGMA_ALLOW_AL, E_AL,
    RHO_CFRP, SIGMA_ALLOW_CFRP, E_CFRP,
    N_REACT, G, MU_DRAG, STRUCT_SF,
    N_LIMIT_LG, GROUND_CLEARANCE,
)

# A representative section that runs in BOTH branches without raising.
SECTION = dict(Do=0.12, Di=0.10, l_eff=0.40)
VZ = 2.44
M_TOTAL = 2500.0


# ---------------------------------------------------------------------------
# CHANGE 1 - the two mechanics use their own energy balance.
# ---------------------------------------------------------------------------
def test_elastic_triangular_vs_plastic_rectangular_balance():
    Do, Di, l = SECTION["Do"], SECTION["Di"], SECTION["l_eff"]

    re = mc._size_elastic(Do, Di, l, VZ, M_TOTAL, RHO_AL, SIGMA_ALLOW_AL, E_AL)
    _, _, I, _ = mc._section_props(Do, Di)
    K = N_REACT * 3.0 * E_AL * I / l**3

    delta_e = re["delta"]
    F_e = re["F_max"]

    # Closed-form triangular stroke: delta ~ Vz*sqrt(m/K) when the gravity term is
    # small (it is, because K m Vz^2 >> (m g)^2 for a stiff leg).
    assert delta_e == pytest.approx(VZ * np.sqrt(M_TOTAL / K), rel=0.05)

    # Triangular work-energy balance is satisfied exactly: 1/2 K d^2 = 1/2 m Vz^2 + (mg)d.
    grav = M_TOTAL * G  # KAPPA_LG = 0 -> L = 0
    assert 0.5 * F_e * delta_e == pytest.approx(0.5 * M_TOTAL * VZ**2 + grav * delta_e)

    # The bug this catches: switching the section modulus to elastic but keeping the
    # PLASTIC (flat-plateau) balance delta = 1/2 m Vz^2 / F_max underpredicts the stroke
    # by ~2x, because a triangle stores half the energy of a rectangle at equal peak force.
    delta_plastic_balance = 0.5 * M_TOTAL * VZ**2 / F_e
    assert delta_e / delta_plastic_balance == pytest.approx(2.0, abs=0.2)

    # Plastic branch: stroke really is 1/2 m Vz^2 / net_force (rectangular plateau).
    rp = mc._size_plastic(Do, Di, l, VZ, M_TOTAL, RHO_AL, SIGMA_ALLOW_AL)
    net = rp["F_max"] - M_TOTAL * G  # net = F_max - (W - L), L = 0
    assert rp["delta"] == pytest.approx(0.5 * M_TOTAL * VZ**2 / net)
    assert rp["mechanics"] == "plastic" and re["mechanics"] == "elastic"


# ---------------------------------------------------------------------------
# CHANGE 2 - CFRP only goes through the elastic branch.
# ---------------------------------------------------------------------------
def test_cfrp_forbidden_in_plastic_runs_in_elastic():
    Do, Di, l = SECTION["Do"], SECTION["Di"], SECTION["l_eff"]

    # CFRP has no plastic hinge -> the plastic sizer must refuse it.
    with pytest.raises(ValueError, match="no plastic hinge"):
        mc._size_plastic(Do, Di, l, VZ, M_TOTAL, RHO_CFRP, SIGMA_ALLOW_CFRP, ductile=False)

    # CFRP runs fine in the elastic branch.
    re = mc._size_elastic(Do, Di, l, VZ, M_TOTAL, RHO_CFRP, SIGMA_ALLOW_CFRP, E_CFRP)
    assert re["mechanics"] == "elastic" and re["F_max"] > 0.0

    # Ductile metal runs the plastic branch without raising.
    rp = mc._size_plastic(Do, Di, l, VZ, M_TOTAL, RHO_AL, SIGMA_ALLOW_AL, ductile=True)
    assert rp["mechanics"] == "plastic"

    # In the full sizer, ductile=False must NEVER touch the plastic branch.
    original = mc._size_plastic

    def _boom(*a, **k):
        raise AssertionError("plastic branch must not be used for CFRP")

    mc._size_plastic = _boom
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # CFRP elastic-only may be infeasible -> fallback
            mc.landing_gear_mass(2500.0, RHO_CFRP, SIGMA_ALLOW_CFRP, E_CFRP, ductile=False)
    finally:
        mc._size_plastic = original


# ---------------------------------------------------------------------------
# CHANGE 3 - STRUCT_SF is no longer baked into the plastic collapse force.
# ---------------------------------------------------------------------------
def test_struct_sf_removed_from_plastic_force_raises_n():
    Do, Di, l = SECTION["Do"], SECTION["Di"], SECTION["l_eff"]

    rp = mc._size_plastic(Do, Di, l, VZ, M_TOTAL, RHO_AL, SIGMA_ALLOW_AL)
    n_new = rp["n"]

    # Old code used the de-rated moment Mp = sigma_eff * Z_p / STRUCT_SF inside F_max,
    # so its reported n was a factor STRUCT_SF too low.
    Z_p = (Do**3 - Di**3) / 6.0
    sigma_eff = SIGMA_ALLOW_AL / np.sqrt(1.0 + MU_DRAG**2)
    F_old = N_REACT * (sigma_eff * Z_p / STRUCT_SF) / l
    n_old = F_old / (M_TOTAL * G)

    assert n_new / n_old == pytest.approx(STRUCT_SF, rel=1e-9)


# ---------------------------------------------------------------------------
# CHANGE 4 - no gear double-count; seed-independent fixed point.
# ---------------------------------------------------------------------------
def test_mass_loop_no_inflation_and_seed_independent():
    mtow = 2500.0
    seen = []

    orig_p, orig_e = mc._size_plastic, mc._size_elastic

    def rec_p(Do, Di, l, Vz, m_total, *a, **k):
        seen.append(m_total)
        return orig_p(Do, Di, l, Vz, m_total, *a, **k)

    def rec_e(Do, Di, l, Vz, m_total, *a, **k):
        seen.append(m_total)
        return orig_e(Do, Di, l, Vz, m_total, *a, **k)

    mc._size_plastic, mc._size_elastic = rec_p, rec_e
    try:
        m_final, *_ = mc.landing_gear_mass(mtow, RHO_AL, SIGMA_ALLOW_AL, E_AL)
    finally:
        mc._size_plastic, mc._size_elastic = orig_p, orig_e

    # Pass 1 sizes the FULL landing mass with the gear folded in exactly once:
    # m_total = (mtow - seed) + seed == mtow.  No ~1.03*MTOW inflation.
    assert seen[0] == pytest.approx(mtow)

    # Fixed point is independent of the iteration starting guess.
    finals = [
        mc.landing_gear_mass(mtow, RHO_AL, SIGMA_ALLOW_AL, E_AL, gear_seed_kg=s)[0]
        for s in (5.0, 50.0, 300.0)
    ]
    for f in finals:
        assert f == pytest.approx(m_final, rel=5e-3)

    # Starting the loop AT the fixed point reproduces it (converged to < 0.5%).
    m_at_fp = mc.landing_gear_mass(mtow, RHO_AL, SIGMA_ALLOW_AL, E_AL,
                                   gear_seed_kg=m_final)[0]
    assert m_at_fp == pytest.approx(m_final, rel=5e-3)


# ---------------------------------------------------------------------------
# CHANGE 5 - calibrated bent-arm L_EFF yields a feasible plastic-tube section.
# ---------------------------------------------------------------------------
def test_long_bent_arm_makes_plastic_tube_heavier():
    m_total = 2500.0

    base = mc._governing_plastic_tube(m_total, RHO_AL, SIGMA_ALLOW_AL)
    assert base is not None, "no feasible tube with the calibrated bent-arm"
    assert 0.20 <= base["l_eff"] <= 0.60       # bent-arm, NOT the full track half-span
    assert base["n"] <= N_LIMIT_LG + 1e-9
    assert base["delta"] <= GROUND_CLEARANCE + 1e-9

    # An over-long arm (the old, mis-set 1.2 m "half-track") drives a heavier tube: a smaller
    # collapse force per unit Z_p needs a larger section.
    saved_min, saved_max = mc.L_EFF_MIN, mc.L_EFF_MAX
    mc.L_EFF_MIN = mc.L_EFF_MAX = 1.2
    try:
        long = mc._governing_plastic_tube(m_total, RHO_AL, SIGMA_ALLOW_AL)
    finally:
        mc.L_EFF_MIN, mc.L_EFF_MAX = saved_min, saved_max
    assert long is None or long["m_gear"] > base["m_gear"]


# ---------------------------------------------------------------------------
# Architecture trade study: every feasible gear architecture is sized and the
# selected one honours the objective (default: lightest reusable-at-limit gear).
# ---------------------------------------------------------------------------
def test_architecture_trade_study_and_selection():
    mtow = 2500.0
    d = mc.landing_gear_mass(mtow, RHO_AL, SIGMA_ALLOW_AL, E_AL, return_details=True)
    assert d is not None

    # the selected gear satisfies the drop constraints
    assert d["n"] <= N_LIMIT_LG + 1e-9
    assert d["delta"] <= GROUND_CLEARANCE + 1e-9

    archs = {r["arch"]: r for r in d["all_architectures"]}
    assert set(archs) == {"plastic_tube", "metal_spring", "composite_spring", "two_stage"}

    # the plastic tube is always feasible for ductile metal, with a bent-arm in the sweep
    # band, and it is NOT reusable (it yields at the limit drop)
    pt = archs["plastic_tube"]
    assert pt["feasible"]
    assert 0.20 <= pt["geom"]["l_eff"] <= 0.60
    assert not pt["reusable_limit"]

    # the GFRP leaf spring is fully reusable (elastic at BOTH drops)
    cs = archs["composite_spring"]
    assert cs["feasible"] and cs["reusable_limit"] and cs["reusable_reserve"]

    # default objective 'prefer_reusable' -> lightest gear that is elastic at the limit drop
    reusable = [r for r in d["all_architectures"] if r.get("feasible") and r["reusable_limit"]]
    assert reusable, "expected at least one reusable architecture"
    assert d["reusable_limit"] is True
    assert d["m_gear"] == pytest.approx(min(r["m_gear"] for r in reusable), rel=1e-6)


def test_min_mass_objective_picks_lightest_overall(monkeypatch):
    monkeypatch.setattr(mc, "GEAR_OBJECTIVE", "min_mass")
    d = mc.landing_gear_mass(2500.0, RHO_AL, SIGMA_ALLOW_AL, E_AL, return_details=True)
    feasible = [r["m_gear"] for r in d["all_architectures"] if r.get("feasible")]
    assert d["m_gear"] == pytest.approx(min(feasible), rel=1e-6)
