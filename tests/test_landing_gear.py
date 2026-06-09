"""
Unit tests for the multi-architecture skid landing-gear drop trade study
(class_II_sizing.mass_components). The gear is sized for the CS-27.725 limit and
27.727 reserve drops; five energy-absorber architectures are sized with scipy SLSQP and
the MTOW loop uses the weighted-trade-off winner.

Coverage:
  1. effective drop mass (CS-27.725(b)) - lift credit reduces M_eff
  2. drop velocities - reserve is sqrt(1.5) x limit
  3. concept models return sane whole-gear dicts; whole_gear scaling
  4. size() returns a feasible design that honours all four drop constraints
  5. landing_gear_mass: deterministic, returns the weighted-score winner, sane mass

All quantities are SI.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import class_II_sizing.mass_components as mc
from parameters import (
    G, H_L, D_EST, LIFT, N_LIMIT, ENVELOPE, FOOTPRINT_MAX_SPAN,
    N_SKID, D_SKID_RAIL, T_SKID_RAIL, L_SKID_RAIL, RHO_AL, GEAR_OPT_BOUNDS,
    D_FRAME_TUBE, T_FRAME_TUBE, L_ARM, CROSSTUBE_COUNT, CROSS_SPAN, ELASTO_COUNT,
)

ARCH_NAMES = set(GEAR_OPT_BOUNDS)            # the five architecture names
MTOW_TEST = 700.0                            # mass at which the placeholder bounds were tuned


# ---------------------------------------------------------------------------
# 1 - effective drop mass (CS-27.725(b))
# ---------------------------------------------------------------------------
def test_effective_mass_lift_credit():
    m = 2000.0
    # No lift credit -> the whole weight is reacted, M_eff == MTOW.
    assert mc.effective_mass(m, H_L, D_EST, 0.0) == pytest.approx(m)
    # Full lift credit reduces the effective drop mass below MTOW (but stays positive).
    m_full = mc.effective_mass(m, H_L, D_EST, 1.0)
    assert 0.0 < m_full < m
    # Closed form: W_e = W (h + (1-L) d)/(h + d).
    assert m_full == pytest.approx(m * H_L / (H_L + D_EST))


# ---------------------------------------------------------------------------
# 2 - drop velocities
# ---------------------------------------------------------------------------
def test_drop_velocities():
    assert mc.v_limit() == pytest.approx(np.sqrt(2 * G * H_L))
    assert mc.v_reserve() == pytest.approx(np.sqrt(1.5) * mc.v_limit())


# ---------------------------------------------------------------------------
# 3 - concept models + whole_gear scaling
# ---------------------------------------------------------------------------
def test_concepts_return_sane_dicts():
    for name, cfg in GEAR_OPT_BOUNDS.items():
        r = mc._GEAR_CONCEPTS[name](cfg["x0"], cfg["material"])
        for key in ("k", "Ue", "Up", "Fmax", "mass", "dy", "dmax", "reusable", "track_m"):
            assert key in r, f"{name} missing {key}"
        assert r["mass"] > 0 and r["k"] > 0 and r["Fmax"] > 0, name
        assert r["Ue"] >= 0 and r["Up"] >= 0, name
        assert r["dmax"] >= r["dy"] >= 0, name
        assert r["track_m"] > 0, name


def test_whole_gear_scaling():
    count = 3
    r = mc.whole_gear(k=1.0, Ue=2.0, Up=3.0, Fmax=4.0, mass1=5.0,
                      dy=0.10, dmax=0.20, count=count, reusable=True, track_m=2.5)
    # Parallel members add: stiffness, energy, load and mass scale by count.
    assert r["k"] == count * 1.0
    assert r["Ue"] == count * 2.0
    assert r["Up"] == count * 3.0
    assert r["Fmax"] == count * 4.0
    # Whole-gear mass = members + 2 skid rails + the shared skeleton (2 cross-tubes + 4 legs).
    assert r["mass"] == count * 5.0 + N_SKID * mc.skid_rail_mass() + mc.mount_frame_mass()
    # Members deflect together, so strokes do NOT scale; track is passed through.
    assert r["dy"] == 0.10 and r["dmax"] == 0.20
    assert r["track_m"] == 2.5
    assert r["reusable"] is True


def test_skid_rail_mass():
    # Closed form: hollow aluminium tube, m = rho * pi/4 (D^2 - d^2) * L.
    d = D_SKID_RAIL - 2 * T_SKID_RAIL
    expected = RHO_AL * np.pi / 4 * (D_SKID_RAIL**2 - d**2) * L_SKID_RAIL
    assert mc.skid_rail_mass() > 0
    assert mc.skid_rail_mass() == pytest.approx(expected)


def test_mount_frame_mass():
    # Closed form: 2 cross-tubes (CROSS_SPAN each) + 4 arms (L_ARM each), hollow Al tubes.
    d = D_FRAME_TUBE - 2 * T_FRAME_TUBE
    length_total = CROSSTUBE_COUNT * CROSS_SPAN + ELASTO_COUNT * L_ARM
    expected = RHO_AL * np.pi / 4 * (D_FRAME_TUBE**2 - d**2) * length_total
    assert mc.mount_frame_mass() > 0
    assert mc.mount_frame_mass() == pytest.approx(expected)


def test_elastomeric_includes_frame_and_rails():
    # The elastomeric whole-gear mass must include the 4 mounts, the 2 skid rails AND the
    # load-path skeleton (2 cross-tubes + 4 legs) -- not just mounts + rails. The skeleton now
    # comes from whole_gear (shared by every architecture), not from elastomeric itself.
    cfg = GEAR_OPT_BOUNDS["E elastomeric"]
    r = mc.elastomeric(cfg["x0"], cfg["material"])
    kb, dm, _ = cfg["x0"]
    fmax = kb * dm
    mounts = ELASTO_COUNT * (mc.ELASTO_FIXED_MASS + mc.ELASTO_MASS_PER_N * fmax)
    rails = N_SKID * mc.skid_rail_mass()
    assert r["mass"] == pytest.approx(mounts + rails + mc.mount_frame_mass())


def test_skeleton_on_all_architectures():
    # Every architecture sits on the same 2-cross-tube + 4-leg skeleton + 2 skid rails, so each
    # whole-gear mass must exceed (members + rails + skeleton) lower bound, i.e. include the
    # skeleton. Check the skeleton is present by reconstructing the non-member floor.
    floor = N_SKID * mc.skid_rail_mass() + mc.mount_frame_mass()
    for name, cfg in GEAR_OPT_BOUNDS.items():
        r = mc._GEAR_CONCEPTS[name](cfg["x0"], cfg["material"])
        assert r["mass"] > floor, f"{name} mass should include rails + skeleton"


def test_track_within_footprint():
    # Every feasible sized architecture must fit the spanwise footprint cap.
    m_eff = mc.effective_mass(MTOW_TEST, H_L, D_EST, LIFT)
    rows = mc.size_all_architectures(m_eff)
    for r in rows:
        assert r["track_m"] > 0, r["name"]
        if r["feasible"]:
            assert r["track_m"] <= FOOTPRINT_MAX_SPAN + 1e-6, r["name"]


# ---------------------------------------------------------------------------
# 4 - size() returns a feasible design honouring the four drop constraints
# ---------------------------------------------------------------------------
def test_size_feasible_designs_honour_constraints():
    m_eff = mc.effective_mass(MTOW_TEST, H_L, D_EST, LIFT)
    rows = mc.size_all_architectures(m_eff)
    feas = [r for r in rows if r["feasible"]]
    assert feas, "expected at least one feasible architecture at the tuned mass"
    for r in feas:
        # g1/g2: elastic at limit and survive the reserve -> non-negative margins.
        assert r["MSe"] >= -1e-6, r["name"]
        assert r["MSr"] >= -1e-6, r["name"]
        # g3: peak deceleration cap.  g4: stroke fits the envelope.
        assert r["npk"] <= N_LIMIT + 1e-6, r["name"]
        assert r["dmax"] <= ENVELOPE + 1e-6, r["name"]
        # g5: lateral track fits the spanwise footprint cap.
        assert r["track_m"] <= FOOTPRINT_MAX_SPAN + 1e-6, r["name"]


# ---------------------------------------------------------------------------
# 5 - landing_gear_mass: deterministic weighted-winner selection, sane mass
# ---------------------------------------------------------------------------
def test_landing_gear_mass_selects_weighted_winner():
    mtow = 2136.0
    d = mc.landing_gear_mass(mtow)
    assert d is not None
    assert d["arch"] in ARCH_NAMES
    assert len(d["all_architectures"]) == len(GEAR_OPT_BOUNDS)
    assert 0.0 < d["m_gear"] < 0.20 * mtow          # gear is a small fraction of MTOW

    # The chosen architecture is the deterministic weighted-score winner.
    sc = mc.score(d["all_architectures"])
    assert d["arch"] == max(sc, key=sc.get)
    assert d["m_gear"] == pytest.approx(
        next(r["mass"] for r in d["all_architectures"] if r["name"] == d["arch"]))


def test_landing_gear_mass_is_deterministic():
    a = mc.landing_gear_mass(2136.0)
    b = mc.landing_gear_mass(2136.0)
    assert a["arch"] == b["arch"]
    assert a["m_gear"] == pytest.approx(b["m_gear"])


def test_landing_gear_mass_scalar_return():
    d = mc.landing_gear_mass(2136.0, return_details=True)
    m = mc.landing_gear_mass(2136.0, return_details=False)
    assert m == pytest.approx(d["m_gear"])
