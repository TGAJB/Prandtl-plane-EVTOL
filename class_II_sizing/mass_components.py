"""
mass_components.py
One function per structural/propulsion component.
Each function returns mass in kg.
Functions that depend on MTOW take it as their first argument.
"""

import math
import random
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from parameters import (
    G, RHO_ORIGIN,
    L_FUS, PER_FUS_MAX, N_PAX,
    N_W, TAPER_W, TIP_TO_CHORD_W,
    SIGMA_ALLOW_CFRP, RHO_CFRP, T_SKIN_MIN_CFRP,
    S_TAIL, AR_T, TAPER_TAIL, TIP_TO_CHORD, V_ANGLE,
    F_REAR_WING,
    STRUCT_SF, C_N_TAIL_MAX, V_DIVE_FACTOR, V_CRUISE,
    N_PROP, N_MOTOR, N_BLADES, D_PROP, PM,
    WING_SPAN, AREA_SPLIT, WING_LOADING_N,
    # landing-gear drop trade study
    H_L, D_EST, LIFT, N_LIMIT, ENVELOPE,
    N_SKID, SKID_RAIL_MASS, CROSSTUBE_COUNT, HINGES_PER_TUBE,
    LEAF_COUNT, HINGES_PER_LEAF, COMPOSITE_COUNT,
    CRUSH_COUNT, ELASTO_COUNT,
    BC_FACTOR, CROSS_SPAN, TUBE_HINGE_LEN, LEAF_HINGE_LEN, LEAF_DEV_FACTOR,
    COMP_CRUSH_FRAC, COMP_DELAM_FACTOR,
    CRUSH_LEAF_B, CRUSH_LEAF_L, HONEYCOMB_STRESS, HONEYCOMB_DENSITY, CRUSH_STROKE_EFF,
    ELASTO_FIXED_MASS, ELASTO_MASS_PER_N,
    QUAL, WEIGHTS, GEAR_OPT_BOUNDS, GEAR_OPT_SEED, GEAR_OPT_RESTARTS, GEAR_BOUNDS_REF_MASS,
)


# Fuselage

def fuselage_mass(mtow_kg):
    return (0.453592
            * (14.86 * ((mtow_kg * 2.20462) ** 0.144) * ((L_FUS * 3.28084) ** 0.778)
               / ((PER_FUS_MAX * 3.28084) ** 0.778))
            * ((L_FUS * 3.28084) ** 0.383) * N_PAX ** 0.455)


# Wing (dynamic sizing from design-point W/S + Class II mass)

def wing_geometry(mtow_kg):
    # -- Geometry from selected design point: S = W / (W/S) --
    weight_n = mtow_kg * G
    total_area_m2 = weight_n / WING_LOADING_N
    area_per_wing_m2 = total_area_m2 * AREA_SPLIT
    aspect_ratio = WING_SPAN ** 2 / total_area_m2
    aspect_ratio_per_wing = WING_SPAN ** 2 / area_per_wing_m2
    mean_chord_m = WING_SPAN / aspect_ratio_per_wing
    root_chord_m = 2 * area_per_wing_m2 / ((1 + TAPER_W) * WING_SPAN)
    tip_chord_m = TAPER_W * root_chord_m

    return {
        "weight_n": weight_n,
        "total_area_m2": total_area_m2,
        "area_per_wing_m2": area_per_wing_m2,
        "span_m": WING_SPAN,
        "aspect_ratio": aspect_ratio,
        "mean_chord_m": mean_chord_m,
        "root_chord_m": root_chord_m,
        "tip_chord_m": tip_chord_m,
    }


def wing_mass(mtow_kg, geometry=None):
    # -- Geometry (identical for front and rear wings) --
    if geometry is None:
        geometry = wing_geometry(mtow_kg)

    b_w    = geometry["span_m"]                         # full wing span [m]
    s      = b_w / 2                                    # semi-span [m]
    s_wing = geometry["area_per_wing_m2"]              # planform area per wing [m²]
    c_mean = geometry["mean_chord_m"]       # root chord [m]
    h_spar_mean = TIP_TO_CHORD_W * c_mean                   # spar depth at root [m]
    # no h_eff correction: wing is horizontal, bending is about the chord axis

    # -- Size front and rear wings independently (general for F_REAR_WING ≠ 0.5) --
    m_total = 0.0
    for f_lift in [(1.0 - F_REAR_WING), F_REAR_WING]:
        l_wing = f_lift * N_W * mtow_kg * G

        # Correct spar cap volume for elliptical lift distribution.
        # ∫ M(y) dy = L_wing·s²/8  (derived analytically from elliptical l(y))
        # vol_caps = STRUCT_SF · L·s² / (8·σ·h)
        vol_caps = STRUCT_SF * l_wing * s**2 / (8 * SIGMA_ALLOW_CFRP * h_spar_mean)
        m_spar   = 1.4 * vol_caps * RHO_CFRP    # caps + web, CFRP

        # Skins: min-gauge CFRP (8-ply prepreg), upper + lower surface
        m_skin   = 2 * s_wing * T_SKIN_MIN_CFRP * RHO_CFRP

        # Primary fraction 0.76 recovers ribs + secondary structure
        m_total += (m_spar + m_skin) / 0.76

    return m_total


# Landing gear
#
# Skid landing-gear sizing by a multi-architecture trade study (CS-27.725 limit +
# 27.727 reserve drops). Five energy-absorber architectures are sized for the two drops
# with scipy SLSQP (minimise whole-gear mass s.t. the drop, peak-decel and stroke-
# envelope constraints); the MTOW loop uses the weighted-score winner's mass. The
# offline study (Monte-Carlo sensitivity, weighted ranking, F-delta plot) lives in
# class_II_sizing/trade_off_landing_gear.py. All constants live in parameters.py.
#
# Architecture: 2 skid rails joined by 2 transverse cross-members; 2 knees per member
# -> 4 legs/hinges across the 2 skids. M_EFF is the TOTAL effective drop mass on the
# WHOLE gear; every concept's capacity and mass are whole-gear totals (per-member
# values scaled by the *_COUNT constants in parameters.py).


def effective_mass(mtow_kg, h, d, L, g=G):
    """CS/FAR-27.725(b) effective drop mass from MTOW.
       W_e = W (h + (1-L) d) / (h + d),  W = MTOW g for a symmetric flat skid drop.
       M_eff = W_e/g.  L=0 -> M_eff = MTOW (no lift credit; gear absorbs all)."""
    W = mtow_kg * g
    We = W * (h + (1.0 - L) * d) / (h + d)
    return We / g


# Drop energy targets -------------------------------------------------------

def v_limit():
    return math.sqrt(2 * G * H_L)


def v_reserve():
    return math.sqrt(2 * G * 1.5 * H_L)


def E_limit(delta, m_eff):
    # lift credit is already folded into m_eff, so use the full effective weight
    return 0.5 * m_eff * v_limit()**2 + m_eff * G * delta


def E_reserve(delta, m_eff):
    return 0.5 * m_eff * v_reserve()**2 + m_eff * G * delta


# Concept models (each returns whole-gear totals in one dict) ----------------
# keys: k, Ue, Up, mass, dy, dmax, Fmax, reusable

def cross_tube(x, m):
    """(A) 2 bent metal cross-tubes, 4 sacrificial plastic hinges.  x = D, t, L"""
    D, t, L = x
    d  = D - 2 * t
    A  = math.pi / 4 * (D**2 - d**2)
    I  = math.pi / 64 * (D**4 - d**4)
    Z  = I / (D / 2)
    Zp = (D**3 - d**3) / 6
    k  = BC_FACTOR * m["E"] * I / L**3                 # per cross-tube
    My, Mp = m["sy"] * Z, m["sy"] * Zp
    Fy = My / L
    dy = Fy / k
    Ue = 0.5 * k * dy**2                               # per cross-tube
    theta = (m["eu"] / (D / 2)) * (TUBE_HINGE_LEN * D)  # hinge rotation
    Up = Mp * theta * HINGES_PER_TUBE                  # per cross-tube (2 knees)
    Fmax = max(Fy, Mp / L)                             # per cross-tube
    dmax = dy + theta * L
    mass1 = m["rho"] * A * (2 * L + CROSS_SPAN)        # one cross-tube
    return whole_gear(k, Ue, Up, Fmax, mass1, dy, dmax, CROSSTUBE_COUNT, reusable=False)


def metal_leaf(x, m):
    """(B) 2 metal leaf / bow springs.  x = b, t, L"""
    b, t, L = x
    A  = b * t
    I  = b * t**3 / 12
    Z  = b * t**2 / 6
    Zp = b * t**2 / 4
    k  = BC_FACTOR * m["E"] * I / L**3
    My, Mp = m["sy"] * Z, m["sy"] * Zp
    Fy = My / L
    dy = Fy / k
    Ue = 0.5 * k * dy**2
    theta = (m["eu"] / (t / 2)) * (LEAF_HINGE_LEN * t)
    Up = Mp * theta * HINGES_PER_LEAF
    Fmax = max(Fy, Mp / L)
    dmax = dy + theta * L
    mass1 = m["rho"] * A * (LEAF_DEV_FACTOR * L)
    return whole_gear(k, Ue, Up, Fmax, mass1, dy, dmax, LEAF_COUNT, reusable=True)


def composite_leaf(x, m):
    """(B') 2 composite leaves: elastic to failure then brittle crush.  x = b, t, L"""
    b, t, L = x
    A = b * t
    I = b * t**3 / 12
    Z = b * t**2 / 6
    k = BC_FACTOR * m["E"] * I / L**3
    Ffail = (m["sy"] * Z) / L
    dfail = Ffail / k
    Ue = 0.5 * k * dfail**2
    Up = m["Gc"] * (b * COMP_DELAM_FACTOR * L)          # fracture energy * delam area
    Fmax = Ffail
    dmax = dfail + COMP_CRUSH_FRAC * L
    mass1 = m["rho"] * A * (LEAF_DEV_FACTOR * L)
    return whole_gear(k, Ue, Up, Fmax, mass1, dfail, dmax, COMPOSITE_COUNT, reusable=False)


def crushable(x, m):
    """(F) 4 units: stiff leaf (limit) + crushable honeycomb (reserve).  x = tleaf, Ac, sc"""
    tleaf, Ac, sc = x
    b, L = CRUSH_LEAF_B, CRUSH_LEAF_L
    A = b * tleaf
    I = b * tleaf**3 / 12
    Z = b * tleaf**2 / 6
    k = BC_FACTOR * m["E"] * I / L**3
    Fy = (m["sy"] * Z) / L
    dy = Fy / k
    Ue = 0.5 * k * dy**2
    Fcr = HONEYCOMB_STRESS * Ac
    Up = Fcr * sc * CRUSH_STROKE_EFF
    Fmax = max(Fy, Fcr)
    dmax = dy + sc
    mass1 = m["rho"] * A * (LEAF_DEV_FACTOR * L) + HONEYCOMB_DENSITY * Ac * sc
    return whole_gear(k, Ue, Up, Fmax, mass1, dy, dmax, CRUSH_COUNT, reusable=False)


def elastomeric(x, m):
    """(E) 4 elastomeric block mounts, hysteretic, reusable.  x = kb, dm, loss"""
    kb, dm, loss = x
    Ue = 0.5 * kb * dm**2
    Up = loss * Ue
    Fmax = kb * dm
    mass1 = ELASTO_FIXED_MASS + ELASTO_MASS_PER_N * Fmax
    return whole_gear(kb, Ue, Up, Fmax, mass1, dm, dm, ELASTO_COUNT, reusable=True)


def whole_gear(k, Ue, Up, Fmax, mass1, dy, dmax, count, reusable):
    """Scale one member's properties to the whole gear (count members in parallel) and
    add the two shared skid rails. Members deflect together, so stiffness, energy, load
    and mass add up; stroke (dy, dmax) does not."""
    return dict(
        k=count * k, Ue=count * Ue, Up=count * Up, Fmax=count * Fmax,
        mass=count * mass1 + N_SKID * SKID_RAIL_MASS,
        dy=dy, dmax=dmax, reusable=reusable,
    )


# Sizing with scipy SLSQP: minimise mass s.t. the drop constraints -----------
#   g1: Ue        - E_L        (elastic at limit)
#   g2: Ue + Up   - E_R        (survive reserve)
#   g3: N_LIMIT*W - Fmax       (peak-decel cap)
#   g4: ENVELOPE  - dmax       (fits stroke)

def constraints(x, fn, mat, m_eff):
    r = fn(x, mat)
    return np.array([
        r["Ue"]             - E_limit(r["dy"], m_eff),
        r["Ue"] + r["Up"]   - E_reserve(r["dmax"], m_eff),
        N_LIMIT * m_eff * G - r["Fmax"],
        ENVELOPE            - r["dmax"],
    ])


def feasible(x, fn, mat, m_eff):
    return bool(np.all(constraints(x, fn, mat, m_eff) >= -1e-6))


def mass_of(x, fn, mat):
    return fn(x, mat)["mass"]


def size(name, fn, mat, x0, bounds, varnames, m_eff):
    """Minimise whole-gear mass subject to all four drop constraints >= 0, trying x0 plus
    8 random restarts; keep the lightest feasible design. Returns the result dict with
    'name', 'feasible', 'params', 'SEA', 'MSe', 'MSr', 'npk' added."""
    cons = {"type": "ineq", "fun": constraints, "args": (fn, mat, m_eff)}

    starts = [np.array(x0, float)]
    for _ in range(GEAR_OPT_RESTARTS):
        starts.append(np.array([random.uniform(lo, hi) for lo, hi in bounds]))

    best_x, best_mass, ok = np.array(x0, float), np.inf, False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # hide benign SLSQP "outside bounds" notices
        for start in starts:
            out = minimize(mass_of, start, args=(fn, mat), method="SLSQP",
                           bounds=bounds, constraints=cons, options={"maxiter": 300})
            if feasible(out.x, fn, mat, m_eff) and out.fun < best_mass:
                best_x, best_mass, ok = out.x, out.fun, True

    r = fn(best_x, mat)
    r["name"]     = name
    r["feasible"] = ok
    r["params"]   = dict(zip(varnames, np.round(best_x, 5)))
    r["SEA"]      = (r["Ue"] + r["Up"]) / r["mass"]
    r["MSe"]      = r["Ue"] / E_limit(r["dy"], m_eff) - 1
    r["MSr"]      = (r["Ue"] + r["Up"]) / E_reserve(r["dmax"], m_eff) - 1
    r["npk"]      = r["Fmax"] / (m_eff * G)
    return r


# Trade-off scoring (weighted sum over feasible architectures) ---------------

def metrics(r):
    q = QUAL.get(r["name"], {})
    return {
        "SEA": r["SEA"], "mass": r["mass"], "npk": r["npk"],
        "reusable": 1.0 if r["reusable"] else 0.0,
        "tunable": q.get("tunable", 0.5),
        "cert_risk": q.get("cert_risk", 0.5),
        "cost": q.get("cost", 0.5),
    }


def normalise(values, direction):
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0 for _ in values]
    if direction == "max":
        return [(x - lo) / (hi - lo) for x in values]
    return [(hi - x) / (hi - lo) for x in values]


def score_table(data, weights):
    names = list(data)
    norm = {n: {} for n in names}
    for crit, (direction, _, _) in WEIGHTS.items():
        col = [data[n][crit] for n in names]
        for n, s in zip(names, normalise(col, direction)):
            norm[n][crit] = s
    wsum = sum(weights[c] for c in WEIGHTS)
    return {n: sum(weights[c] / wsum * norm[n][c] for c in WEIGHTS) for n in names}


def score(rows):
    feas = [r for r in rows if r["feasible"]]
    if not feas:
        return {}
    data = {r["name"]: metrics(r) for r in feas}
    base = {c: WEIGHTS[c][1] for c in WEIGHTS}
    return score_table(data, base)


# Architecture name -> concept function (search config lives in GEAR_OPT_BOUNDS).
_GEAR_CONCEPTS = {
    "A cross-tube":       cross_tube,
    "B metal leaf":       metal_leaf,
    "B' composite leaf":  composite_leaf,
    "F crushable hybrid": crushable,
    "E elastomeric":      elastomeric,
}


def _scaled_gear_bounds(m_eff):
    """Grow each architecture's UPPER bounds (and seed x0) with the landing mass so the
    trade study stays valid at any MTOW (the baseline bounds are tuned at
    GEAR_BOUNDS_REF_MASS). Per-variable exponents (cfg['scale']) scale only the
    energy-bearing, stroke-neutral dimensions ~linearly so elastic capacity tracks
    E_L ~ m_eff, while stroke / thickness / dimensionless variables stay fixed. The
    optimizer minimises mass, so a wider upper bound only enlarges the feasible search.

    Returns a per-architecture config dict shaped like GEAR_OPT_BOUNDS (scaled x0/bounds,
    same material/varnames)."""
    ratio = m_eff / GEAR_BOUNDS_REF_MASS
    out = {}
    for name, cfg in GEAR_OPT_BOUNDS.items():
        scale = cfg.get("scale", [0.0] * len(cfg["varnames"]))
        x0, bounds = [], []
        for x0_i, (lo, hi), s in zip(cfg["x0"], cfg["bounds"], scale):
            f = max(1.0, ratio ** s)                 # never shrink below the calibrated bounds
            hi_s = hi * f
            bounds.append((lo, hi_s))
            x0.append(min(max(x0_i * f, lo), hi_s))  # keep the seed inside the scaled bounds
        out[name] = dict(x0=x0, bounds=bounds,
                         varnames=cfg["varnames"], material=cfg["material"])
    return out


def size_all_architectures(m_eff):
    """Size every architecture for the effective drop mass m_eff [kg] using bounds that
    auto-scale with the mass (see _scaled_gear_bounds). Returns the list of result dicts."""
    rows = []
    for name, cfg in _scaled_gear_bounds(m_eff).items():
        rows.append(size(name, _GEAR_CONCEPTS[name], cfg["material"],
                         cfg["x0"], cfg["bounds"], cfg["varnames"], m_eff))
    return rows


def landing_gear_mass(mtow_kg, return_details=True):
    """Skid landing-gear mass [kg] from a 6-architecture drop trade study.

    The whole aircraft (gear included) decelerates in the drop, so the effective drop
    mass M_EFF is formed from the full mtow_kg; the gear self-weight feedback is closed
    by the OUTER MTOW convergence loop (consistent with wing/tail - no inner gear-mass
    iteration). Each architecture is sized with scipy SLSQP (seeded for reproducibility)
    for the CS-27 limit + reserve drops, and the MTOW loop uses the weighted-trade-off
    winner's whole-gear mass.

    Returns
    -------
    details dict   (winning row augmented with 'm_gear', 'arch', 'geom', 'score',
                   'all_architectures')   if return_details (default)
    float mass     if return_details is False
    None           if no architecture is feasible
    """
    random.seed(GEAR_OPT_SEED)                       # reproducible SLSQP restarts in the loop
    m_eff = effective_mass(mtow_kg, H_L, D_EST, LIFT)
    rows = size_all_architectures(m_eff)
    sc = score(rows)
    if not sc:
        warnings.warn(
            "landing_gear_mass: no feasible architecture - returning None (the MTOW loop "
            "falls back to a 3% class-1 gear estimate). Check the drop inputs (H_L, "
            "N_LIMIT, ENVELOPE) and the search bounds in GEAR_OPT_BOUNDS.",
            stacklevel=2,
        )
        return None
    win = max(sc, key=sc.get)
    r = next(row for row in rows if row["name"] == win)
    if not return_details:
        return r["mass"]
    return {**r, "m_gear": r["mass"], "arch": r["name"], "geom": {},
            "score": sc[win], "all_architectures": rows}


# V-tail (physics-based cantilever sizing)
#
# Design: TWO SEPARATE angled tails (no shared centerline apex). Each tail is an
# independent cantilever fixed at its own root, so the structure is sized as one
# surface and doubled. S_TAIL is the planform area of ONE tail and AR_T is that
# tail's aspect ratio, so l_panel = sqrt(AR_T * S_TAIL) -- NOT a combined-V half-span.

def tail_mass(mtow_kg):
    # -- Geometry (per individual tail) --
    l_panel = np.sqrt(AR_T * S_TAIL)                      # root-to-tip cantilever length [m]
    b_half  = l_panel * np.cos(np.radians(V_ANGLE))       # horizontal projection (vert-load arm)
    c_root  = 2 * S_TAIL / ((1 + TAPER_TAIL) * l_panel)  # trapezoid: S = 1/2 (c_r+c_t) l_panel
    h_spar  = TIP_TO_CHORD * c_root
    h_eff   = h_spar * np.cos(np.radians(V_ANGLE))

    # -- Load case 1: rear wing lift transferred through V-tail root --
    f_vert  = (F_REAR_WING * N_W * mtow_kg * G) / 2.0
    m_rear  = b_half * f_vert

    # -- Load case 2: V-tail own aero load at dive speed, max deflection --
    q_dive  = 0.5 * RHO_ORIGIN * (V_DIVE_FACTOR * V_CRUISE) ** 2
    f_aero  = q_dive * S_TAIL * C_N_TAIL_MAX              # S_TAIL is per-tail area
    m_aero  = f_aero * l_panel / 2

    # -- Ultimate root moment --
    m_root  = STRUCT_SF * (m_rear + m_aero)

    # -- Spar (caps + web): Euler-Bernoulli cantilever, CFRP --
    vol_caps = m_root * l_panel / (SIGMA_ALLOW_CFRP * h_eff)
    vol_spar = 1.4 * vol_caps          # web adds ~40% of cap volume
    m_spar   = vol_spar * RHO_CFRP

    # -- Skins: min-gauge CFRP governs --
    m_skin   = 2 * S_TAIL * T_SKIN_MIN_CFRP * RHO_CFRP

    # -- Primary fraction 0.76 accounts for ribs + fittings (secondary structure) --
    m_panel  = (m_spar + m_skin) / 0.76

    return 2.0 * m_panel               # both tails


# Motors

def motor_mass(max_power_kw):
    """Total mass of all motors [kg]."""
    m_single = 0.165 * ((max_power_kw * (1 + PM)) / N_MOTOR)
    return m_single * N_MOTOR


# Propellers

def propeller_mass(max_power_kw, use_fusion=False, m_blade_fusion=None):
    """Total propeller blade mass [kg] across all rotors."""
    if use_fusion and m_blade_fusion is not None:
        return m_blade_fusion * N_BLADES * N_PROP
    return N_PROP * 0.144 * ((D_PROP * (max_power_kw / N_PROP) * N_BLADES ** 0.5) ** 0.782)


def hub_mass(use_fusion=False, m_hub_fusion=None):
    """Total hub mass [kg] across all rotors."""
    if use_fusion and m_hub_fusion is not None:
        return m_hub_fusion * N_PROP
    return 0.0


# Miscellaneous & hinge

def misc_mass(mtow_kg):
    return 0.20 * mtow_kg


def hinge_mass(mtow_kg):
    return 0.05 * mtow_kg
