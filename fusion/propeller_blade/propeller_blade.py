"""
propeller_blade.py  ·  Fusion 360 Script
Parametric NACA 4-digit propeller blade.

Change any constant in the "Iterable constants" block and re-run the script
to rebuild the geometry.  All values are SI (metres / degrees) — the single
cm() helper handles the Fusion-internal centimetre conversion.

Linked to class_II_sizing/mtow_sizing.py sizing constants:
    D_PROP   = 1.9 m   →  BLADE_RADIUS_M = D_PROP / 2 = 0.95
    N_BLADES = 5       →  use Fusion's circular pattern on the Z-axis
                          to replicate to a full rotor after running this script

Coordinate convention
    Blade extends along +Y  (radial / spanwise direction)
    Chord runs along  +X    (tangential, plane-of-rotation direction)
    Thickness along   +Z    (parallel to thrust / shaft axis)
    Twist rotates the chord toward +Z at the root (high pitch angle there)

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
# fusion_params.py sits one level above this script (project root).
# Edit that file to change geometry without touching this script.

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
# Values come from fusion_params.py.  The defaults below are fallbacks used
# only if the file is missing or an attribute is absent.

BLADE_RADIUS_M  = _get("BLADE_RADIUS_M",  0.95)   # [m]  tip radius = D_PROP / 2
HUB_RADIUS_M    = _get("HUB_RADIUS_M",    0.114)   # [m]  root cut-out (~12 % R)
ROOT_CHORD_M    = _get("ROOT_CHORD_M",    0.175)   # [m]  chord at root
TIP_CHORD_M     = _get("TIP_CHORD_M",     0.042)   # [m]  chord at tip
ROOT_TWIST_DEG  = _get("ROOT_TWIST_DEG",  34.0)    # [°]  pitch at root
TIP_TWIST_DEG   = _get("TIP_TWIST_DEG",   11.0)    # [°]  pitch at tip
NACA_M          = _get("NACA_M",          0.04)    # [-]  max camber ratio
NACA_P          = _get("NACA_P",          0.40)    # [-]  camber peak position
NACA_T          = _get("NACA_T",          0.12)    # [-]  max thickness ratio
N_SECTIONS           = int(_get("N_SECTIONS",          6))       # [-]  radial loft sections
N_PTS                = int(_get("N_PTS",               20))      # [-]  airfoil points per side
RHO_PROPELLER_BLADE  = _get("RHO_PROPELLER_BLADE",     1600.0)   # [kg/m³]  CFRP

# Output path — resolves to fusion/geometry.json relative to this script
GEOMETRY_JSON = os.path.join(_SCRIPT_DIR, "..", "geometry.json")


# ── Unit helpers ───────────────────────────────────────────────────────────────

def cm(x_m):
    """Metres → centimetres (Fusion 360 internal unit)."""
    return x_m * 100.0


def sketch_pt(local_x_cm, local_z_cm):
    """
    2-D point in sketch-local coordinates (depth = 0).
    For a sketch on the XZ-plane-at-Y:
        local_x_cm  →  World X (chord direction)
        local_z_cm  →  World Z (thickness direction)
    """
    return adsk.core.Point3D.create(local_x_cm, local_z_cm, 0.0)


# ── NACA 4-digit airfoil ───────────────────────────────────────────────────────

def naca_profile_pts(chord_m, m_cam, p_cam, t_max, n_pts, twist_rad):
    """
    Build a closed airfoil cross-section and return a list of
    (local_x_cm, local_z_cm) sketch-coordinate pairs.

    Steps:
        1. Generate upper/lower surfaces via NACA 4-digit equations.
        2. Force a closed trailing edge (average the open-TE gap).
        3. Assemble loop:  LE → upper surface → TE → lower surface → near-LE
           (Fusion closes back to LE via spline.isClosed = True).
        4. Shift origin to quarter-chord, apply pitch twist in the XZ plane.
        5. Convert metres → centimetres.

    Arguments:
        chord_m   blade chord length at this radial station [m]
        m_cam     NACA max camber ratio      (0.04 for NACA 4-series)
        p_cam     NACA camber peak position  (0.40 = 40 % chord)
        t_max     NACA max thickness ratio   (0.12 = 12 % chord)
        n_pts     points per surface side    (20 is a good default)
        twist_rad pitch angle in radians
    """
    upper = []
    lower = []

    for i in range(n_pts + 1):
        # Cosine spacing — denser near leading and trailing edges
        beta = math.pi * i / n_pts
        xc   = 0.5 * (1.0 - math.cos(beta))   # normalised chord position 0→1
        x    = xc * chord_m

        # Symmetric thickness distribution (NACA 4-digit formula)
        yt = (t_max / 0.2) * chord_m * (
              0.2969 * math.sqrt(max(xc, 1e-10))
            - 0.1260 * xc
            - 0.3516 * xc ** 2
            + 0.2843 * xc ** 3
            - 0.1015 * xc ** 4          # note: open TE; closed explicitly below
        )

        # Camber line and slope
        if p_cam > 0.0 and xc <= p_cam:
            yc  = (m_cam / p_cam ** 2) * chord_m * (2.0 * p_cam * xc - xc ** 2)
            dyc = (2.0 * m_cam / p_cam ** 2) * (p_cam - xc)
        elif p_cam > 0.0:
            yc  = (m_cam / (1.0 - p_cam) ** 2) * chord_m * (
                      (1.0 - 2.0 * p_cam) + 2.0 * p_cam * xc - xc ** 2)
            dyc = (2.0 * m_cam / (1.0 - p_cam) ** 2) * (p_cam - xc)
        else:
            yc, dyc = 0.0, 0.0

        theta = math.atan(dyc)
        upper.append((x - yt * math.sin(theta),  yc + yt * math.cos(theta)))
        lower.append((x + yt * math.sin(theta),  yc - yt * math.cos(theta)))

    # Force a closed trailing edge by averaging the residual NACA gap
    te_z          = (upper[-1][1] + lower[-1][1]) * 0.5
    upper[-1]     = (chord_m, te_z)
    lower[-1]     = (chord_m, te_z)

    # Assemble closed loop (2 * n_pts points, isClosed handles the LE junction)
    #   upper[:-1] : LE → one-before-TE   (n_pts points, TE excluded)
    #   reversed(lower)[:-1] : TE → one-after-LE  (n_pts points, LE excluded)
    loop_raw = upper[:-1] + list(reversed(lower))[:-1]

    # Shift to quarter-chord origin and apply pitch twist in the XZ (chord-thickness) plane
    qc     = chord_m * 0.25
    result = []
    for (x, z) in loop_raw:
        x0 = x - qc                                              # QC-referenced chord
        xt =  x0 * math.cos(twist_rad) - z * math.sin(twist_rad)
        zt =  x0 * math.sin(twist_rad) + z * math.cos(twist_rad)
        result.append((cm(xt), cm(zt)))                          # metres → cm

    return result


# ── Blade solid ────────────────────────────────────────────────────────────────

def build_blade(root, comp):
    """
    Lofts N_SECTIONS NACA airfoil profiles between hub and tip to produce
    a single solid blade body.  The quarter-chord line runs along +Y.

    After running, use Fusion's circular-pattern feature to replicate to
    N_BLADES blades around the Z-axis.
    """
    sketches   = comp.sketches
    planes     = comp.constructionPlanes
    loft_feats = comp.features.loftFeatures

    # Cosine-spaced radial stations — denser near root (rapid taper/twist change)
    # and near tip (critical aerodynamic region).
    r_range = BLADE_RADIUS_M - HUB_RADIUS_M
    fracs   = [
        0.5 * (1.0 - math.cos(math.pi * i / max(N_SECTIONS - 1, 1)))
        for i in range(N_SECTIONS)
    ]

    section_profiles = []

    for frac in fracs:
        r     = HUB_RADIUS_M + frac * r_range
        chord = ROOT_CHORD_M + frac * (TIP_CHORD_M   - ROOT_CHORD_M)
        twist = ROOT_TWIST_DEG + frac * (TIP_TWIST_DEG - ROOT_TWIST_DEG)

        # ── Construction plane perpendicular to Y at radial station r ──────────
        # Offset the XZ construction plane (y = 0) by cm(r) along its +Y normal.
        pl_input = planes.createInput()
        pl_input.setByOffset(
            root.xZConstructionPlane,
            adsk.core.ValueInput.createByReal(cm(r))
        )
        section_plane = planes.add(pl_input)
        sk = sketches.add(section_plane)

        # ── Airfoil spline ──────────────────────────────────────────────────────
        profile_pts = naca_profile_pts(
            chord, NACA_M, NACA_P, NACA_T, N_PTS, math.radians(twist)
        )

        pts_col = adsk.core.ObjectCollection.create()
        for (lx, lz) in profile_pts:
            pts_col.add(sketch_pt(lx, lz))

        spline           = sk.sketchCurves.sketchFittedSplines.add(pts_col)
        spline.isClosed  = True                    # closes LE gap in the spline

        section_profiles.append(sk.profiles.item(0))

    # ── Loft all sections into a solid ─────────────────────────────────────────
    loft_input = loft_feats.createInput(
        adsk.fusion.FeatureOperations.NewBodyFeatureOperation
    )
    for prof in section_profiles:
        loft_input.loftSections.add(prof)
    loft_input.isSolid = True

    blade_feat = loft_feats.add(loft_input)
    blade_feat.bodies.item(0).name = "propeller_blade"

    return blade_feat


# ── Mass properties export ─────────────────────────────────────────────────────

def export_mass_properties(comp, out_path, density_override=None):
    """
    Merges body mass properties into geometry.json so entries from other
    Fusion scripts (hub, fuselage, etc.) are preserved.
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

        build_blade(root, root)

        results = export_mass_properties(root, GEOMETRY_JSON,
                                         density_override=RHO_PROPELLER_BLADE)

        lines = [
            "Propeller blade built.",
            f"Mass properties written to:",
            f"  {GEOMETRY_JSON}",
            "",
            "Results:",
        ]
        for name, props in results.items():
            lines.append(
                f"  {name}:  {props['mass_kg']:.4f} kg  |  "
                f"V = {props['volume_m3'] * 1e6:.1f} cm³"
            )
        lines += [
            "",
            "Next step: use Fusion's circular pattern feature",
            f"  Axis: Z-axis,  Quantity: {5}  (N_BLADES from class_II_sizing/mtow_sizing.py)",
        ]
        ui.messageBox("\n".join(lines))

    except Exception:
        if ui:
            ui.messageBox("Build failed:\n" + traceback.format_exc())
