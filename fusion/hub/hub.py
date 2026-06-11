"""
hub.py  ·  Fusion 360 Script
Parametric propeller hub: annular disk with shaft bore and blade-root sockets.

Geometry (three operations):
  1. Solid outer cylinder  — full hub body
  2. Shaft bore cut        — through-hole for motor shaft
  3. Blade-root socket cuts — N_BLADES circular cutouts around the rim
     (each socket circle is centred on the rim; only the inward half is cut)

Parameters read from fusion_params.py:
  N_BLADES, HUB_OUTER_RADIUS_M, HUB_INNER_RADIUS_M,
  HUB_HEIGHT_M, HUB_SOCKET_RADIUS_M

Coordinate convention (matches propeller_blade.py):
  Hub disk lies in the XY plane.  Shaft axis along +Z.
  Blade 0 points along +Y (0°); subsequent blades spaced 360°/N_BLADES.

Run via: Tools → Add-ins → Scripts and Add-ins → Scripts → [+] → select this
         folder → Run
"""

import adsk.core
import adsk.fusion
import traceback
import importlib.util
import json
import math
import os


# ── Parameter file loading ─────────────────────────────────────────────────────

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PARAMS_PATH = os.path.join(_SCRIPT_DIR, "..", "fusion_params.py")

def _load_params(path):
    try:
        spec = importlib.util.spec_from_file_location("fusion_params", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None

_P = _load_params(_PARAMS_PATH)

def _get(attr, default):
    if _P is not None and hasattr(_P, attr):
        return getattr(_P, attr)
    return default


# ── Iterable constants ─────────────────────────────────────────────────────────
# Values come from fusion_params.py; inline defaults are fallbacks only.

N_BLADES             = int(_get("N_BLADES",             8))
HUB_OUTER_RADIUS_M   = _get("HUB_OUTER_RADIUS_M",      0.191)   # [m]  = blade HUB_RADIUS_M
HUB_INNER_RADIUS_M   = _get("HUB_INNER_RADIUS_M",      0.025)   # [m]  motor shaft bore
HUB_HEIGHT_M         = _get("HUB_HEIGHT_M",             0.160)   # [m]  axial thickness
HUB_SOCKET_RADIUS_M  = _get("HUB_SOCKET_RADIUS_M",     0.030)   # [m]  blade-root cutout radius
RHO_PROPELLER_HUB    = _get("RHO_PROPELLER_HUB",       2700.0)  # [kg/m³]  aluminium

GEOMETRY_JSON = os.path.join(_SCRIPT_DIR, "..", "geometry.json")


# ── Unit helper ────────────────────────────────────────────────────────────────

def cm(x_m):
    """Metres → centimetres (Fusion 360 internal unit)."""
    return x_m * 100.0


# ── Hub solid ──────────────────────────────────────────────────────────────────

def build_hub(root, comp):
    """
    Builds the propeller hub as a solid with bore and blade-root sockets.
    Returns the initial extrude feature (the body is named 'propeller_hub').
    """
    sketches = comp.sketches
    extrudes = comp.features.extrudeFeatures
    xy_plane = root.xYConstructionPlane
    origin   = adsk.core.Point3D.create(0.0, 0.0, 0.0)

    # ── 1. Outer cylinder ──────────────────────────────────────────────────────
    sk_outer = sketches.add(xy_plane)
    sk_outer.sketchCurves.sketchCircles.addByCenterRadius(
        origin, cm(HUB_OUTER_RADIUS_M)
    )
    ext_outer = extrudes.createInput(
        sk_outer.profiles.item(0),
        adsk.fusion.FeatureOperations.NewBodyFeatureOperation
    )
    ext_outer.setDistanceExtent(
        False, adsk.core.ValueInput.createByReal(cm(HUB_HEIGHT_M))
    )
    hub_feat = extrudes.add(ext_outer)
    hub_feat.bodies.item(0).name = "propeller_hub"

    # ── 2. Shaft bore ──────────────────────────────────────────────────────────
    sk_bore = sketches.add(xy_plane)
    sk_bore.sketchCurves.sketchCircles.addByCenterRadius(
        origin, cm(HUB_INNER_RADIUS_M)
    )
    ext_bore = extrudes.createInput(
        sk_bore.profiles.item(0),
        adsk.fusion.FeatureOperations.CutFeatureOperation
    )
    ext_bore.setDistanceExtent(
        False, adsk.core.ValueInput.createByReal(cm(HUB_HEIGHT_M))
    )
    extrudes.add(ext_bore)

    # ── 3. Blade-root sockets ──────────────────────────────────────────────────
    # Each socket circle is centred on the hub outer rim at the blade's angular
    # position.  The half of the circle inside the hub is cut away to form the
    # root pocket; the outer half lies outside the solid and removes nothing.
    for i in range(N_BLADES):
        theta = 2.0 * math.pi * i / N_BLADES
        cx    = HUB_OUTER_RADIUS_M * math.cos(theta)
        cy    = HUB_OUTER_RADIUS_M * math.sin(theta)

        sk_sock = sketches.add(xy_plane)
        sk_sock.sketchCurves.sketchCircles.addByCenterRadius(
            adsk.core.Point3D.create(cm(cx), cm(cy), 0.0),
            cm(HUB_SOCKET_RADIUS_M)
        )
        ext_sock = extrudes.createInput(
            sk_sock.profiles.item(0),
            adsk.fusion.FeatureOperations.CutFeatureOperation
        )
        ext_sock.setDistanceExtent(
            False, adsk.core.ValueInput.createByReal(cm(HUB_HEIGHT_M))
        )
        extrudes.add(ext_sock)

    return hub_feat


# ── Mass properties export (merges into geometry.json) ────────────────────────

def export_mass_properties(comp, out_path, density_override=None):
    """
    Adds this script's body entries to geometry.json without overwriting
    entries written by other Fusion scripts (blade, fuselage, etc.).
    density_override: when given, mass_kg = volume_m3 * density_override,
    so the exported value is correct regardless of the material assigned in Fusion.
    """
    existing = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = {}

    for body in comp.bRepBodies:
        props  = body.physicalProperties
        vol_m3 = props.volume / 1.0e6
        mass_kg = vol_m3 * density_override if density_override is not None else props.mass
        existing[body.name] = {
            "mass_kg":         mass_kg,
            "volume_m3":       vol_m3,
            "surface_area_m2": props.area   / 1.0e4,
            "cog_x_m":         props.centerOfMass.x / 100.0,
            "cog_y_m":         props.centerOfMass.y / 100.0,
            "cog_z_m":         props.centerOfMass.z / 100.0,
        }

    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)

    return existing


# ── Entry point ────────────────────────────────────────────────────────────────

def run(context):
    ui = None
    try:
        app  = adsk.core.Application.get()
        ui   = app.userInterface
        des  = adsk.fusion.Design.cast(app.activeProduct)
        root = des.rootComponent

        build_hub(root, root)

        results = export_mass_properties(root, GEOMETRY_JSON,
                                         density_override=RHO_PROPELLER_HUB)

        lines = [
            "Hub built.",
            "Mass properties merged into:",
            f"  {GEOMETRY_JSON}",
            "",
            "All components in geometry.json:",
        ]
        for name, props in results.items():
            lines.append(
                f"  {name}:  {props['mass_kg']:.4f} kg  |  "
                f"V = {props['volume_m3'] * 1e6:.1f} cm³"
            )
        lines += [
            "",
            "Next: re-run class_II_sizing.mtow_sizing.py to update MTOW with hub mass.",
        ]
        ui.messageBox("\n".join(lines))

    except Exception:
        if ui:
            ui.messageBox("Build failed:\n" + traceback.format_exc())
