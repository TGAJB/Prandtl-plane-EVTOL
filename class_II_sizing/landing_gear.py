# =====================================================================
#  Skid landing-gear trade-off for the eVTOL   (single file, no classes)
# ---------------------------------------------------------------------
#  Architecture modelled:  2 skid rails on the ground, joined by 2
#  transverse cross-tubes; each cross-tube has 2 knees -> 4 plastic
#  hinges and 4 "legs" across the 2 skids.  M_EFF is the TOTAL effective
#  drop mass on the WHOLE gear; every concept's capacity and mass below
#  are whole-gear totals (per-member values are scaled by the *_COUNT
#  constants in section 1b).
#
#  Drop basis: EASA SC-VTOL accepts CS/FAR-27.725 (limit) + 27.727
#  (reserve) drop tests.
#    limit drop :  v_L = sqrt(2*g*h_L),  h_L >= 0.20 m floor
#    reserve    :  height = 1.5 * h_L  ->  energy = 1.5 * E_L
#
#  Requirement per concept:
#    stay ELASTIC at the limit drop      ->  Ue        >= E_L
#    survive the reserve drop (plastic)  ->  Ue + Up   >= E_R
#
#  Sizing uses scipy SLSQP: minimise mass s.t. those two drop constraints
#  plus a peak-decel cap and a stroke-envelope limit.
#
#  Every tunable constant lives in section 1 with a SOURCE tag.  See the
#  SOURCES list at the bottom of this file for where each value comes from.
#
#  Run:  python landing_gear_tradeoff.py     (needs numpy, scipy, matplotlib)
# =====================================================================

import math
import random
import warnings
import numpy as np
from scipy.optimize import minimize

warnings.filterwarnings("ignore")  # hide benign SLSQP "outside bounds" notices

# =====================================================================
# 1a.  PRIMARY INPUTS  (swap these for your eVTOL numbers)
# =====================================================================
G        = 9.81      # m/s^2

# -- whole-aircraft drop inputs --
MTOW     = 700.0     # kg   max take-off mass                            PLACEHOLDER
H_L      = 0.25      # m    limit drop height (>= 0.20)                  PLACEHOLDER
D_EST    = 0.15      # m    estimated impact deflection (gear stroke;    PLACEHOLDER
                     #      skids have no tyre) - used ONLY to size M_EFF
LIFT     = 0.0       # -    rotor/lift credit ratio L (0 = conservative, no credit)
N_LIMIT  = 20.0      # g    max allowed peak deceleration                PLACEHOLDER
ENVELOPE = 0.40      # m    available vertical stroke envelope           PLACEHOLDER

def effective_mass(MTOW, h, d, L, g=G):
    """CS/FAR-27.725(b) effective drop mass from MTOW.   [SOURCE: S8]
       W_e = W * (h + (1-L)*d) / (h + d),  with W = MTOW*g for a symmetric
       flat skid drop (the whole gear reacts the whole weight).
       M_eff = W_e/g.   L=0 -> M_eff = MTOW (no lift credit; gear absorbs all).
       For an asymmetric / single-skid-first attitude, pass the reacted
       fraction of MTOW instead of the full value."""
    W  = MTOW * g
    We = W * (h + (1.0 - L) * d) / (h + d)
    return We / g

# TOTAL effective drop mass on the whole gear (4 hinges across 2 skids)
M_EFF = effective_mass(MTOW, H_L, D_EST, LIFT)

# Materials: E [Pa], sy [Pa], rho [kg/m3], eu [-], Gc [J/m2]   [SOURCE: M1]
AL    = {"E": 71.7e9, "sy": 503e6,  "rho": 2810, "eu": 0.11,  "Gc": 0}
STEEL = {"E": 200e9,  "sy": 1200e6, "rho": 7850, "eu": 0.06,  "Gc": 0}
CFRP  = {"E": 70e9,   "sy": 600e6,  "rho": 1600, "eu": 0.015, "Gc": 1500}

# =====================================================================
# 1b.  GEAR ARCHITECTURE  (how many parallel units form the full gear)
#      2 skids, 2 legs per skid -> 4 legs, 4 hinges.
# =====================================================================
N_SKID          = 2      # longitudinal ground rails
SKID_RAIL_MASS  = 4.0    # kg per rail (ground contact tube)              [SOURCE: S5]

CROSSTUBE_COUNT = 2      # transverse cross-tubes (front + rear)
HINGES_PER_TUBE = 2      # knees per cross-tube  -> 2*2 = 4 hinges total
LEAF_COUNT      = 2      # transverse leaf springs
HINGES_PER_LEAF = 2
COMPOSITE_COUNT = 2
OLEO_COUNT      = 4      # one oleo strut per leg
CRUSH_COUNT     = 4      # one crush unit per leg
ELASTO_COUNT    = 4      # one elastomeric mount per leg

# =====================================================================
# 1c.  STRUCTURAL MODEL CONSTANTS  (all the formerly-hard-coded numbers)
# =====================================================================
BC_FACTOR         = 3.0    # k = BC*EI/L^3 (3 = tip-loaded cantilever)    [SOURCE: S1]
CROSS_SPAN        = 0.60   # m   skid track width under fuselage          PLACEHOLDER
TUBE_HINGE_LEN    = 1.0    # x D   plastic-hinge length, tube knee        [SOURCE: S2]
LEAF_HINGE_LEN    = 1.5    # x t   plastic-hinge length, leaf             [SOURCE: S2]
LEAF_DEV_FACTOR   = 2.0    # developed leaf length = factor * L
COMP_CRUSH_FRAC   = 0.5    # x L   post-failure crush travel, composite   PLACEHOLDER
COMP_DELAM_FACTOR = 2.0    # delaminated length = factor * L

OLEO_EFF          = 0.85   # shock-absorber efficiency (0.8-0.9)          [SOURCE: S3]
OLEO_LIMIT_FRAC   = 0.45   # fraction of stroke used at the limit drop    PLACEHOLDER
OLEO_MASS_PER_N   = 9.0e-4 # kg/N  strut mass vs peak load                [SOURCE: S5]
OLEO_MASS_PER_M   = 6.0    # kg/m  strut mass vs stroke                   [SOURCE: S5]

CRUSH_LEAF_B      = 0.08   # m   width of the elastic leaf in the hybrid
CRUSH_LEAF_L      = 0.45   # m   length of that leaf
HONEYCOMB_STRESS  = 8.0e6  # Pa  honeycomb crush plateau stress           [SOURCE: S4]
HONEYCOMB_DENSITY = 80.0   # kg/m3  honeycomb core density                [SOURCE: S4]
CRUSH_STROKE_EFF  = 0.75   # usable crush fraction before densification   [SOURCE: S6]

ELASTO_FIXED_MASS = 1.2    # kg   mount housing/bracket fixed mass        [SOURCE: S7]
ELASTO_MASS_PER_N = 4.0e-4 # kg/N mount mass vs peak load                 [SOURCE: S7]


# =====================================================================
# 2.  DROP ENERGY TARGETS
# =====================================================================
def v_limit():   return math.sqrt(2 * G * H_L)
def v_reserve(): return math.sqrt(2 * G * 1.5 * H_L)

def E_limit(delta):
    # lift credit is already folded into M_EFF, so use the full effective weight
    return 0.5 * M_EFF * v_limit()**2 + M_EFF * G * delta

def E_reserve(delta):
    return 0.5 * M_EFF * v_reserve()**2 + M_EFF * G * delta


# =====================================================================
# 3.  CONCEPT MODELS  (each returns whole-gear totals in one dict)
#     keys: k, Ue, Up, mass, dy, dmax, Fmax, reusable
# =====================================================================
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

def oleo(x, m):
    """(D) 4 oleo-pneumatic struts, reusable, energy = eff*F*stroke.  x = Fdes, s"""
    Fdes, s = x
    sL  = OLEO_LIMIT_FRAC * s
    Ue  = OLEO_EFF * Fdes * sL                          # per strut
    Up  = OLEO_EFF * Fdes * (s - sL)
    k   = Fdes / sL
    mass1 = OLEO_MASS_PER_N * Fdes + OLEO_MASS_PER_M * s
    return whole_gear(k, Ue, Up, Fdes, mass1, sL, s, OLEO_COUNT, reusable=True)

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
    """Scale one member's properties to the whole gear (count members in
    parallel) and add the two shared skid rails.  Members deflect together,
    so stiffness, energy, load and mass add up; stroke (dy, dmax) does not."""
    return dict(
        k=count * k, Ue=count * Ue, Up=count * Up, Fmax=count * Fmax,
        mass=count * mass1 + N_SKID * SKID_RAIL_MASS,
        dy=dy, dmax=dmax, reusable=reusable,
    )


# =====================================================================
# 4.  SIZING with scipy SLSQP:  minimise mass s.t. the drop constraints
#       g1: Ue        - E_L        (elastic at limit)
#       g2: Ue + Up   - E_R         (survive reserve)
#       g3: N_LIMIT*W - Fmax        (peak-decel cap)
#       g4: ENVELOPE  - dmax        (fits stroke)
# =====================================================================
def constraints(x, fn, mat):
    r = fn(x, mat)
    return np.array([
        r["Ue"]             - E_limit(r["dy"]),
        r["Ue"] + r["Up"]   - E_reserve(r["dmax"]),
        N_LIMIT * M_EFF * G - r["Fmax"],
        ENVELOPE            - r["dmax"],
    ])

def feasible(x, fn, mat):
    return bool(np.all(constraints(x, fn, mat) >= -1e-6))

def mass_of(x, fn, mat):
    return fn(x, mat)["mass"]

def size(name, fn, mat, x0, bounds, varnames):
    # minimise mass subject to all four constraints >= 0
    cons = {"type": "ineq", "fun": constraints, "args": (fn, mat)}

    # try x0 plus 8 random starts; keep the lightest feasible design
    starts = [np.array(x0, float)]
    for _ in range(8):
        starts.append(np.array([random.uniform(lo, hi) for lo, hi in bounds]))

    best_x, best_mass, ok = np.array(x0, float), np.inf, False
    for start in starts:
        out = minimize(mass_of, start, args=(fn, mat), method="SLSQP",
                       bounds=bounds, constraints=cons, options={"maxiter": 300})
        if feasible(out.x, fn, mat) and out.fun < best_mass:
            best_x, best_mass, ok = out.x, out.fun, True

    r = fn(best_x, mat)
    r["name"]     = name
    r["feasible"] = ok
    r["params"]   = dict(zip(varnames, np.round(best_x, 5)))
    r["SEA"]      = (r["Ue"] + r["Up"]) / r["mass"]
    r["MSe"]      = r["Ue"] / E_limit(r["dy"]) - 1
    r["MSr"]      = (r["Ue"] + r["Up"]) / E_reserve(r["dmax"]) - 1
    r["npk"]      = r["Fmax"] / (M_EFF * G)
    return r


# =====================================================================
# 5.  TRADE-OFF SCORING (weighted sum) + Monte-Carlo sensitivity
# =====================================================================
# qualitative scores 0..1 set by engineering judgement  (PLACEHOLDERS)
QUAL = {
    "A cross-tube":       dict(tunable=0.3, cert_risk=0.1, cost=0.1),
    "B metal leaf":       dict(tunable=0.5, cert_risk=0.2, cost=0.2),
    "B' composite leaf":  dict(tunable=0.5, cert_risk=0.8, cost=0.5),
    "D oleo strut":       dict(tunable=0.8, cert_risk=0.5, cost=0.9),
    "F crushable hybrid": dict(tunable=0.9, cert_risk=0.5, cost=0.5),
    "E elastomeric":      dict(tunable=0.4, cert_risk=0.4, cost=0.2),
}
# criterion -> (direction, base weight, relative uncertainty for sensitivity)
WEIGHTS = {
    "SEA":       ("max", 0.25, 0.10),
    "mass":      ("min", 0.25, 0.05),
    "npk":       ("min", 0.15, 0.10),
    "reusable":  ("max", 0.10, 0.00),
    "tunable":   ("max", 0.08, 0.20),
    "cert_risk": ("min", 0.12, 0.20),
    "cost":      ("min", 0.05, 0.20),
}

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

def sensitivity(rows, trials=4000):
    feas = [r for r in rows if r["feasible"]]
    if not feas:
        return {}
    wins = {r["name"]: 0 for r in feas}
    for _ in range(trials):
        w = {c: max(0.01, WEIGHTS[c][1] * (1 + random.gauss(0, 0.3))) for c in WEIGHTS}
        data = {}
        for r in feas:
            mv = metrics(r)
            for c, (_, _, u) in WEIGHTS.items():
                if u > 0:
                    mv[c] = mv[c] * (1 + random.gauss(0, u))
            data[r["name"]] = mv
        sc = score_table(data, w)
        wins[max(sc, key=sc.get)] += 1
    return {n: wins[n] / trials for n in wins}


# =====================================================================
# 6.  OPTIONAL F-delta PLOT
# =====================================================================
def plot(rows, filename="fd_curves.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib not available - skipping plot)")
        return
    plt.figure(figsize=(9, 6))
    for r in rows:
        Fy = r["k"] * r["dy"]
        xs = [0, r["dy"] * 1000, r["dmax"] * 1000]
        ys = [0, Fy / 1000, r["Fmax"] / 1000]
        style = "-" if r["feasible"] else "--"
        plt.plot(xs, ys, style, lw=2, label=r["name"])
    plt.xlabel("stroke  delta  [mm]")
    plt.ylabel("total reaction  F  [kN]")
    plt.title("Whole-gear load-deflection (area under curve = energy absorbed)")
    plt.grid(alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(filename, dpi=140)
    print("  F-delta plot written ->", filename)


# =====================================================================
# 7.  RUN EVERYTHING
# =====================================================================
def main():
    random.seed(0)
    print("=" * 64)
    print("DROP CONDITIONS (placeholder inputs, whole-gear basis)")
    print("=" * 64)
    print("  M_eff = %.0f kg   (4 hinges across 2 skids)" % M_EFF)
    print("  v_L   = %.2f m/s   v_R = %.2f m/s" % (v_limit(), v_reserve()))
    print("  E_L   = %.0f J     E_R = %.0f J  (= 1.5 E_L)" % (E_limit(0), E_reserve(0)))
    print()

    rows = [
        size("A cross-tube",      cross_tube,     AL,
             [0.060, 0.004, 0.45], [(0.03, 0.14), (0.002, 0.014), (0.30, 0.70)],
             ["D", "t", "L"]),
        size("B metal leaf",      metal_leaf,     STEEL,
             [0.080, 0.010, 0.45], [(0.04, 0.18), (0.004, 0.030), (0.30, 0.70)],
             ["b", "t", "L"]),
        size("B' composite leaf", composite_leaf, CFRP,
             [0.090, 0.012, 0.45], [(0.04, 0.20), (0.004, 0.035), (0.30, 0.70)],
             ["b", "t", "L"]),
        size("D oleo strut",      oleo,           AL,
             [20000.0, 0.18], [(5000, 80000), (0.06, 0.35)],
             ["Fdes", "s"]),
        size("F crushable hybrid",crushable,      STEEL,
             [0.008, 0.003, 0.10], [(0.003, 0.020), (0.0005, 0.012), (0.04, 0.25)],
             ["tleaf", "Ac", "sc"]),
        size("E elastomeric",     elastomeric,    AL,
             [6e5, 0.08, 0.4], [(1e5, 3e6), (0.03, 0.20), (0.3, 0.6)],
             ["kb", "dm", "loss"]),
    ]

    print("=" * 64)
    print("SIZING  (scipy SLSQP: lightest whole gear meeting both drops)")
    print("=" * 64)
    print("%-20s %4s %9s %9s %7s %7s %7s" %
          ("concept", "feas", "mass[kg]", "SEA", "MS_e", "MS_R", "n_pk"))
    print("-" * 64)
    for r in rows:
        print("%-20s %4s %9.1f %9.0f %7.2f %7.2f %7.1f" %
              (r["name"], "Y" if r["feasible"] else "N",
               r["mass"], r["SEA"], r["MSe"], r["MSr"], r["npk"]))
    print()

    sc = score(rows)
    pw = sensitivity(rows)
    print("=" * 64)
    print("TRADE-OFF SCORE + SENSITIVITY")
    print("=" * 64)
    if not sc:
        print("  No concept passed screening - relax inputs/bounds.")
    else:
        print("%-20s %8s %12s" % ("concept", "score", "P(rank #1)"))
        print("-" * 42)
        for n in sorted(sc, key=sc.get, reverse=True):
            print("%-20s %8.3f %10.0f%%" % (n, sc[n], pw.get(n, 0) * 100))
        win = max(sc, key=sc.get)
        rob = max(pw, key=pw.get)
        print("\n  Weighted winner : %s" % win)
        print("  Robust winner   : %s  (%.0f%% of trials)" % (rob, pw[rob] * 100))
    print()
    plot(rows)


# =====================================================================
#  SOURCES  (where to get / justify each tagged constant)
# ---------------------------------------------------------------------
#  M1  Material properties (E, sy, rho, eu): MMPDS / MIL-HDBK-5 handbook
#      (aluminium, steel) and the laminate datasheet (CFRP). Use design
#      allowables, not typical values.
#
#  S1  BC_FACTOR (stiffness coefficient): beam theory. 3 = tip-loaded
#      cantilever (k=3EI/L^3); 48 = centre-loaded simply-supported
#      (k=48EI/L^3). Pick to match how your gear is actually restrained.
#      Ref: Roark's Formulas for Stress and Strain (Young & Budynas).
#
#  S2  Plastic-hinge length (TUBE_HINGE_LEN, LEAF_HINGE_LEN): commonly
#      taken as ~ one section depth in plastic / impact analysis.
#      Ref: N. Jones, "Structural Impact", CUP. Calibrate with FEM.
#
#  S3  OLEO_EFF = 0.8-0.9: N. S. Currey, "Aircraft Landing Gear Design:
#      Principles and Practices", AIAA, 1988 (shock-absorber efficiency).
#
#  S4  HONEYCOMB_STRESS, HONEYCOMB_DENSITY: read the matching row off a
#      Hexcel HexWeb aluminium-honeycomb datasheet (crush/bare-compressive
#      strength and density are coupled per cell-size/foil/alloy).
#      Density tolerance +/-10% per Hexcel. Ref: Hexcel HexWeb CR III DS.
#
#  S5  Component masses (SKID_RAIL_MASS, OLEO_MASS_PER_*): landing-gear
#      weight-estimation methods, e.g. Currey (above) or Raymer,
#      "Aircraft Design: A Conceptual Approach" (gear group weight),
#      or a CAD mass once geometry is fixed.
#
#  S6  CRUSH_STROKE_EFF ~ 0.7-0.8: crush usable up to the densification
#      strain; honeycomb energy-absorption efficiency > 70% in the plateau.
#      Ref: honeycomb crashworthiness / lunar-lander absorber literature.
#
#  S7  ELASTO_FIXED_MASS, ELASTO_MASS_PER_N: fit mass = a + b*F to a
#      vendor catalogue. Parker-LORD / Enidine elastomeric isolators list
#      weight against static load rating per part (e.g. LORD HT2 series).
#
#  S8  effective_mass(): CS/FAR-27.725(b) effective-weight formula
#      W_e = W (h + (1-L)d)/(h + d).  W = static reaction on the gear
#      (= MTOW for a symmetric flat skid drop). L = assumed lift/weight
#      ratio (limit drop caps lift at the weight; reserve allows 1.5x).
# =====================================================================

if __name__ == "__main__":
    main()