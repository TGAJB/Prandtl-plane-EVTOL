"""
fuselage.py  ·  Fusion 360 Script
Parametric eVTOL fuselage pod and landing skids.
Run via: Tools → Add-ins → Scripts and Add-ins → Scripts → [+] → select this
         folder → Run

Units going IN:  SI (metres)
Units Fusion uses: centimetres internally
All conversions happen in to_fusion_units() — nowhere else.
"""

import adsk.core
import adsk.fusion
import traceback
import json
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Sizing parameters (SI, metres) ────────────────────────────────────────────
# Edit values here, then re-run the script to rebuild geometry.

PARAMS_SI = {
    "fus_length_m":     7.0,      # total fuselage length
    "fus_radius_m":     0.95,     # max cross-section radius (half of 1.9m perimeter / pi)
    "nose_length_m":    1.4,      # length of nose cone section
    "tail_taper_m":     1.8,      # length of tail taper section
    "fus_floor_drop_m": 0.15,     # how far the floor sits below centreline
    "skid_length_m":    1.0,      # landing skid fore-aft length
    "skid_radius_m":    0.04,     # skid tube radius
    "skid_offset_y_m":  0.70,     # lateral offset of each skid from CL
}

# ── Unit conversion boundary ─────────────────────────────────────────────────

def to_fusion_units(p):
    """All SI → cm conversion lives here. Geometry code reads fp, never p."""
    return {k: v * 100 for k, v in p.items()}   # all values here are lengths in m

fp = to_fusion_units(PARAMS_SI)

# ── Fusion geometry helpers ───────────────────────────────────────────────────

def point3d(x, y, z):
    return adsk.core.Point3D.create(x, y, z)

def make_ellipse_sketch(sketches, plane, rx, ry):
    """Draw an ellipse sketch on the given plane, centred at origin."""
    sk = sketches.add(plane)
    sk.sketchCurves.sketchEllipses.add(
        point3d(0, 0, 0),
        point3d(rx, 0, 0),
        point3d(0, ry, 0)
    )
    return sk

def make_circle_sketch(sketches, plane, cx, cy, r):
    sk = sketches.add(plane)
    sk.sketchCurves.sketchCircles.addByCenterRadius(point3d(cx, cy, 0), r)
    return sk

# ── Main build function ───────────────────────────────────────────────────────

def build_fuselage(root, comp):
    """
    Strategy: three loft profiles at nose tip, max section, and tail end.
    The loft tool interpolates between them — Fusion handles the blend.

    x = fuselage station (0 = nose tip, positive aft)
    y = lateral (left = negative)
    z = vertical (up = positive)
    """
    sketches   = comp.sketches
    planes     = comp.constructionPlanes
    loft_feats = comp.features.loftFeatures

    fus_len    = fp["fus_length_m"]
    r_max      = fp["fus_radius_m"]
    nose_len   = fp["nose_length_m"]
    tail_len   = fp["tail_taper_m"]
    mid_start  = nose_len
    mid_end    = fus_len - tail_len

    # Helper: offset plane perpendicular to X axis at station x_pos
    def x_plane(x_pos):
        plane_input = planes.createInput()
        plane_input.setByOffset(
            root.yZConstructionPlane,
            adsk.core.ValueInput.createByReal(x_pos)
        )
        return planes.add(plane_input)

    # ── Profile 0: nose tip — small ellipse (not a point, loft needs a profile)
    plane_nose = x_plane(0)
    sk0 = sketches.add(plane_nose)
    sk0.sketchCurves.sketchEllipses.add(
        point3d(0, 0, 0),
        point3d(r_max * 0.08, 0, 0),
        point3d(0, r_max * 0.05, 0)
    )

    # ── Profile 1: max section — full ellipse
    plane_mid_s = x_plane(mid_start)
    sk1 = sketches.add(plane_mid_s)
    sk1.sketchCurves.sketchEllipses.add(
        point3d(0, 0, 0),
        point3d(r_max, 0, 0),
        point3d(0, r_max * 0.72, 0)   # slightly flattened vertically
    )

    # ── Profile 2: end of constant section — same as max section
    plane_mid_e = x_plane(mid_end)
    sk2 = sketches.add(plane_mid_e)
    sk2.sketchCurves.sketchEllipses.add(
        point3d(0, 0, 0),
        point3d(r_max, 0, 0),
        point3d(0, r_max * 0.72, 0)
    )

    # ── Profile 3: tail tip — small ellipse
    plane_tail = x_plane(fus_len)
    sk3 = sketches.add(plane_tail)
    sk3.sketchCurves.sketchEllipses.add(
        point3d(0, 0, 0),
        point3d(r_max * 0.18, 0, 0),
        point3d(0, r_max * 0.10, 0)
    )

    # ── Loft the four profiles ────────────────────────────────────────────────
    loft_input = loft_feats.createInput(
        adsk.fusion.FeatureOperations.NewBodyFeatureOperation
    )

    for sk in [sk0, sk1, sk2, sk3]:
        prof = sk.profiles.item(0)
        loft_input.loftSections.add(prof)

    loft_input.isSolid = True
    fuselage_body = loft_feats.add(loft_input)
    fuselage_body.bodies.item(0).name = "fuselage"

    return fuselage_body


def build_skids(root, comp):
    """
    Two cylindrical tube skids swept along a straight path.
    Each skid is a circle profile swept along a line at ±skid_offset_y.
    """
    sketches   = comp.sketches
    planes     = comp.constructionPlanes
    sweeps     = comp.features.sweepFeatures
    extrudes   = comp.features.extrudeFeatures

    r_skid  = fp["skid_radius_m"]
    y_off   = fp["skid_offset_y_m"]
    sk_len  = fp["skid_length_m"]
    z_skid  = -(fp["fus_radius_m"] * 0.72 + fp["fus_floor_drop_m"] * 100 + r_skid)

    fus_len = fp["fus_length_m"]
    x_start = fus_len * 0.20
    x_end   = fus_len * 0.75

    for side, y_sign in [("port", -1), ("starboard", 1)]:
        y_pos = y_sign * y_off

        # Circle profile on a plane at x_start
        plane_input = planes.createInput()
        plane_input.setByOffset(
            root.yZConstructionPlane,
            adsk.core.ValueInput.createByReal(x_start)
        )
        skid_plane = planes.add(plane_input)

        sk = sketches.add(skid_plane)
        sk.sketchCurves.sketchCircles.addByCenterRadius(
            point3d(y_pos, z_skid, 0), r_skid
        )

        prof = sk.profiles.item(0)

        # Extrude along X for sk_len
        ext_input = extrudes.createInput(
            prof,
            adsk.fusion.FeatureOperations.NewBodyFeatureOperation
        )
        ext_input.setDistanceExtent(
            False,
            adsk.core.ValueInput.createByReal(x_end - x_start)
        )
        skid_feat = extrudes.add(ext_input)
        skid_feat.bodies.item(0).name = f"skid_{side}"


def export_mass_properties(comp, out_path):
    """
    Merges body mass properties into geometry.json so entries from other
    Fusion scripts (blade, hub, etc.) are preserved.
    """
    existing = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = {}

    for body in comp.bRepBodies:
        props = body.physicalProperties
        existing[body.name] = {
            "mass_kg":         props.mass,
            "volume_m3":       props.volume / 1.0e6,
            "surface_area_m2": props.area   / 1.0e4,
            "cog_x_m":         props.centerOfMass.x / 100.0,
            "cog_y_m":         props.centerOfMass.y / 100.0,
            "cog_z_m":         props.centerOfMass.z / 100.0,
        }

    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2)

    return existing


# ── Entry point ───────────────────────────────────────────────────────────────

def run(context):
    ui = None
    try:
        app  = adsk.core.Application.get()
        ui   = app.userInterface
        des  = adsk.fusion.Design.cast(app.activeProduct)
        root = des.rootComponent

        # Build geometry
        build_fuselage(root, root)
        build_skids(root, root)

        # Export mass properties — writes to fusion/geometry.json
        out_path = os.path.join(_SCRIPT_DIR, "..", "geometry.json")
        results  = export_mass_properties(root, out_path)

        # Report in the Fusion console
        lines = ["Mass properties written to " + out_path, ""]
        for name, props in results.items():
            lines.append(f"{name}: {props['mass_kg']:.2f} kg")
        ui.messageBox("\n".join(lines))

    except:
        if ui:
            ui.messageBox("Build failed:\n" + traceback.format_exc())