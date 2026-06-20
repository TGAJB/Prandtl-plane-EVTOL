"""
V2 cross-model validation.

Per the validation catalogue (V2, "Cross-Model Comparison"), independent
modelling approaches applied to the same design point should be mutually
consistent. Here the class-II analytical models (energy.py, mass_components.py)
are checked against the design stored by the independent optimiser pipeline in
final_design/results/data/characteristics.json. Agreement across two separate
implementations validates both.

This needs no external higher-fidelity tool (CAD/CFD/FEM); those comparisons are
the detailed-design-phase part of V2 and are not in this automated suite.
"""

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import class_II_sizing.energy as energy
import class_II_sizing.mass_components as mc

_CS = json.loads((PROJECT_ROOT / "final_design/results/data/characteristics.json").read_text())
_MTOW = _CS["mass"]["mtow"]


def test_battery_mass_cross_model():
    # Class-II cell-count chain vs the optimiser's stored pack mass.
    assert energy.battery_mass(_MTOW) == pytest.approx(_CS["mass"]["battery"], rel=1e-2)


def test_wing_area_cross_model():
    # Class-II wing geometry vs the optimiser's converged wing area.
    geom = mc.wing_geometry(_MTOW)
    assert geom["total_area_m2"] == pytest.approx(_CS["wing_geometry"]["total_area_m2"], rel=1e-2)


def test_aspect_ratio_cross_model():
    geom = mc.wing_geometry(_MTOW)
    assert geom["aspect_ratio"] == pytest.approx(_CS["wing_geometry"]["aspect_ratio"], rel=1e-2)


def test_span_cross_model():
    geom = mc.wing_geometry(_MTOW)
    assert geom["span_m"] == pytest.approx(_CS["wing_geometry"]["span_m"], rel=1e-3)
