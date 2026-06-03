# fusion_params.py
# Fusion 360 geometry input parameters.
# Edit this file, then re-run the relevant Fusion script to rebuild geometry.

# ── Propeller blade ────────────────────────────────────────────────────────────
BLADE_RADIUS_M  = 0.95    # [m]    tip radius (= D_PROP / 2 from class_II_sizing/mtow_sizing.py)
HUB_RADIUS_M    = 0.114   # [m]    root cut-out radius (~12 % of blade radius)
ROOT_CHORD_M    = 0.175   # [m]    chord at root station
TIP_CHORD_M     = 0.042   # [m]    chord at tip (linear taper to root)
ROOT_TWIST_DEG  = 34.0    # [deg]  pitch angle at root
TIP_TWIST_DEG   = 11.0    # [deg]  pitch angle at tip (washout)
NACA_M          = 0.04    # [-]    max camber ratio       (4 %  → NACA 4xxx)
NACA_P          = 0.40    # [-]    camber peak position   (40 % → NACA x4xx)
NACA_T          = 0.12    # [-]    max thickness ratio    (12 % → NACA xx12)
N_SECTIONS      = 6       # [-]    radial loft cross-sections (min 3)
N_PTS           = 20      # [-]    airfoil spline points per surface side

RHO_PROPELLER_BLADE = 1600.0  # [kg/m³]  blade material density (CFRP)
RHO_PROPELLER_HUB   = 2700.0  # [kg/m³]  hub material density   (aluminium)

# ── Propeller hub ──────────────────────────────────────────────────────────────
N_BLADES             = 5       # [-]    blades per rotor (shared with class_II_sizing/mtow_sizing.py)
HUB_OUTER_RADIUS_M   = 0.114   # [m]    hub outer radius — must equal HUB_RADIUS_M above
HUB_INNER_RADIUS_M   = 0.025   # [m]    motor shaft bore radius
HUB_HEIGHT_M         = 0.080   # [m]    hub axial thickness
HUB_SOCKET_RADIUS_M  = 0.030   # [m]    blade-root socket radius (circular cutout per blade)
