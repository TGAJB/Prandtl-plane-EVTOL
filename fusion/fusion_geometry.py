"""
fusion_geometry.py
Loads Fusion 360 exported geometry (fusion/geometry.json) and exposes
per-component mass values used by main_fusion.py.

Workflow:
    1. In Fusion 360, run (in any order):
         fusion/propeller_blade/propeller_blade.py  → writes "propeller_blade"
         fusion/hub/hub.py                          → writes "propeller_hub"
       Each script merges its entry into geometry.json without overwriting others.
       Assign correct materials first: CFRP for blade, aluminium for hub.
    2. Run main_fusion.py. Missing entries fall back to analytical formulae.
"""

import json
import os

_GEOMETRY_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fusion", "geometry.json"
)


def _load(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


_geo = _load(_GEOMETRY_JSON)

# ── Propeller blade ────────────────────────────────────────────────────────────

if "propeller_blade" in _geo:
    M_BLADE_FUSION   = _geo["propeller_blade"]["mass_kg"]
    USE_FUSION_PROP  = True
    print(f"[Fusion] Single blade mass loaded: {M_BLADE_FUSION:.4f} kg")
else:
    M_BLADE_FUSION   = None
    USE_FUSION_PROP  = False
    print("[Fusion] WARNING: no 'propeller_blade' entry — falling back to analytical formula.")

# ── Hub ────────────────────────────────────────────────────────────────────────

if "propeller_hub" in _geo:
    M_HUB_FUSION  = _geo["propeller_hub"]["mass_kg"]
    USE_FUSION_HUB = True
    print(f"[Fusion] Single hub mass loaded:   {M_HUB_FUSION:.4f} kg")
else:
    M_HUB_FUSION  = None
    USE_FUSION_HUB = False
    print("[Fusion] WARNING: no 'propeller_hub' entry — hub mass set to zero.")

print(f"         Source: {_GEOMETRY_JSON}")