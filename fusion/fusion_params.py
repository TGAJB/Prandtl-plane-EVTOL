# fusion_params.py
# Fusion 360 geometry input parameters.
# Edit this file, then re-run the relevant Fusion script to rebuild geometry.

# ── Propeller blade ────────────────────────────────────────────────────────────
BLADE_RADIUS_M  = 0.95    # [m]    tip radius (= D_PROP / 2 from class_II_sizing/mtow_sizing.py)
HUB_RADIUS_M    = 0.191   # [m]    root cut-out radius (~12 % of blade radius)
ROOT_CHORD_M    = 0.175   # [m]    chord at root station
TIP_CHORD_M     = 0.042   # [m]    chord at tip (linear taper to root)
ROOT_TWIST_DEG  = 14.0    # [deg]  pitch angle at root
TIP_TWIST_DEG   = 3.0    # [deg]  pitch angle at tip (washout)
#NACA_M          = 0.04    # [-]    max camber ratio       (4 %  → NACA 4xxx)
#NACA_P          = 0.40    # [-]    camber peak position   (40 % → NACA x4xx)
#NACA_T          = 0.12    # [-]    max thickness ratio    (12 % → NACA xx12)
N_SECTIONS      = 6       # [-]    radial loft cross-sections (min 3)
N_PTS           = 20      # [-]    airfoil spline points per surface side
# ── Airfoil: CLARK-Y (matches aerodynamic analysis) ─────────────────────────
# Clark-Y is NOT a NACA 4-digit section (it has a flat lower surface), so it is
# defined by explicit coordinates rather than NACA_M/P/T parameters. This needs to be updated into the fusion
AIRFOIL_NAME = "CLARK_Y"
CLARK_Y = [   # (x/c, y_upper/c, y_lower/c)
    (0.0000, 0.0000, 0.0000), (0.0125, 0.0277, -0.0140), (0.0250, 0.0381, -0.0173),
    (0.0500, 0.0530, -0.0210), (0.0750, 0.0641, -0.0229), (0.1000, 0.0729, -0.0241),
    (0.1500, 0.0865, -0.0255), (0.2000, 0.0964, -0.0258), (0.3000, 0.1077, -0.0240),
    (0.4000, 0.1112, -0.0198), (0.5000, 0.1086, -0.0150), (0.6000, 0.1003, -0.0101),
    (0.7000, 0.0866, -0.0058), (0.8000, 0.0680, -0.0025), (0.9000, 0.0442, -0.0005),
    (0.9500, 0.0298, 0.0000), (1.0000, 0.0120, 0.0000),
]

RHO_PROPELLER_BLADE = 1600.0  # [kg/m³]  blade material density (CFRP)
RHO_PROPELLER_HUB   = 2700.0  # [kg/m³]  hub material density   (aluminium)

# ── Propeller hub ──────────────────────────────────────────────────────────────
N_BLADES             = 8       # [-]    blades per rotor (shared with class_II_sizing/mtow_sizing.py)
HUB_OUTER_RADIUS_M   = 0.191   # [m]    hub outer radius — must equal HUB_RADIUS_M above
HUB_INNER_RADIUS_M   = 0.025   # [m]    motor shaft bore radius
HUB_HEIGHT_M         = 0.160   # [m]    hub axial thickness
HUB_SOCKET_RADIUS_M  = 0.030   # [m]    blade-root socket radius (circular cutout per blade)
