"""
System-level V&V tests for the integrated class-II design pipeline.

These act on the whole converged pipeline rather than a single block, covering
two of the system test types from the catalogue:

  S2  Regression  - the converged design (MTOW + component breakdown) is frozen
                    as a baseline; a fresh run must reproduce it within tolerance,
                    so an inadvertent change to any formula/constant is caught.
  S4  Stress      - degraded / off-design inputs (battery at 80% beginning-of-life
                    energy density; +100 kg payload) must keep the pipeline finite,
                    physical, and moving in the expected direction.

NB S1 (sensitivity) is covered by the global Sobol study in the stability chapter;
S3 (Fusion 360 whole-aircraft validation) is owned by the MI/CAD team and needs the
assembled-aircraft reference, so it is not in this automated suite.

All masses in kg.
"""

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import class_II_sizing.energy as energy
import class_II_sizing.mtow_sizing as mt


# ---------------------------------------------------------------------------
# S2 - regression baseline of the converged design
# ---------------------------------------------------------------------------
# Frozen from the converged class-II pipeline. If a change is intended, update
# these and note it; an unintended shift fails the test.
_BASELINE = {
    "mtow": 1743.1,
    "fuselage": 90.9,
    "wing": 111.0,
    "tail": 23.41,
    "battery": 476.36,
    "landing_gear": 40.46,
    "misc": 207.87,
    "hinge": 87.93,
    "motors": 193.2,
    "props": 70.25,
}


def test_converged_design_matches_baseline():
    r = mt.converged_mass()
    for key, expected in _BASELINE.items():
        assert r[key] == pytest.approx(expected, rel=1e-2), (
            f"{key} drifted: {r[key]:.2f} vs baseline {expected}"
        )


# ---------------------------------------------------------------------------
# S4 - stress: degraded and off-design inputs
# ---------------------------------------------------------------------------
def test_stress_degraded_battery_energy_density(monkeypatch):
    # Battery at 80% of beginning-of-life energy density -> each cell stores less,
    # so the pack (and MTOW) must grow, stay finite, and not crash the pipeline.
    base = mt.compute_mtow(2000.0, verbose=False)
    monkeypatch.setattr(energy, "E_CELL_WH", 0.8 * energy.E_CELL_WH)
    deg = mt.compute_mtow(2000.0, verbose=False)
    assert math.isfinite(deg["mtow"])
    assert deg["battery"] > base["battery"]      # degraded cells -> heavier pack
    assert deg["mtow"] > base["mtow"]


def test_stress_off_design_heavier_payload(monkeypatch):
    # Off-design heavier payload must propagate to a heavier, finite estimate.
    base = mt.compute_mtow(2000.0, verbose=False)
    monkeypatch.setattr(mt, "M_PAYLOAD", mt.M_PAYLOAD + 100.0)
    heavy = mt.compute_mtow(2000.0, verbose=False)
    assert math.isfinite(heavy["mtow"])
    assert heavy["mtow"] > base["mtow"]


def test_stress_outputs_stay_physical(monkeypatch):
    # Under the degraded-battery case every component mass must remain positive
    # and finite (no NaN/negative leaking through the integrated pipeline).
    monkeypatch.setattr(energy, "E_CELL_WH", 0.8 * energy.E_CELL_WH)
    r = mt.compute_mtow(2000.0, verbose=False)
    for k in ("fuselage", "wing", "tail", "battery", "motors", "props", "misc", "hinge"):
        assert math.isfinite(r[k]) and r[k] > 0.0
