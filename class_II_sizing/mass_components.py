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
from scipy.integrate import quad, cumulative_trapezoid
from mtow_sizing import *

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

"""
from parameters import (
    G, RHO_ORIGIN,
    L_FUS, FUSE_WIDTH, PER_FUS_MAX, N_PAX,
    N_W, TAPER_W, TIP_TO_CHORD_W,
    SIGMA_ALLOW_CFRP, RHO_CFRP, T_SKIN_MIN_CFRP,
    S_TAIL, AR_T, TAPER_TAIL, TIP_TO_CHORD, V_ANGLE,
    F_REAR_WING,
    STRUCT_SF, C_N_TAIL_MAX, V_DIVE_FACTOR, V_CRUISE,
    N_PROP, N_MOTOR, N_BLADES, D_PROP, PM,
    WING_SPAN, AREA_SPLIT, WING_LOADING_N,
    # landing-gear drop trade study
    H_L, D_EST, LIFT, N_LIMIT, ENVELOPE,
    N_SKID, D_SKID_RAIL, T_SKID_RAIL, L_SKID_RAIL, RHO_AL,
    CROSSTUBE_COUNT, HINGES_PER_TUBE,
    LEAF_COUNT, HINGES_PER_LEAF, COMPOSITE_COUNT,
    CRUSH_COUNT, ELASTO_COUNT,
    BC_FACTOR, CROSS_SPAN, FOOTPRINT_MAX_SPAN,
    TUBE_HINGE_LEN, LEAF_HINGE_LEN, LEAF_DEV_FACTOR,
    COMP_CRUSH_FRAC, COMP_DELAM_FACTOR,
    CRUSH_LEAF_B, CRUSH_LEAF_L, HONEYCOMB_STRESS, HONEYCOMB_DENSITY, CRUSH_STROKE_EFF,
    ELASTO_FIXED_MASS, ELASTO_MASS_PER_N,
    D_FRAME_TUBE, T_FRAME_TUBE, L_ARM,
    QUAL, WEIGHTS, GEAR_OPT_BOUNDS, GEAR_OPT_SEED, GEAR_OPT_RESTARTS, GEAR_BOUNDS_REF_MASS,
)
"""
from parameters import *

# Fuselage
#
# Physics-based Class-II sizing of the same shape as the wing/tail methods below:
# build the governing ULTIMATE bending moment, size the bending material (the shell
# skin smeared around a thin-walled circular tube) from the section modulus, floor it
# at min-gauge skin, then recover frames/floor/fittings with a primary fraction.
#
# The governing moment is the peak of a SUPERIMPOSED free-free beam diagram: per design
# condition every coexisting load (the two wing reactions + the tail load + the
# distributed inertia) acts at once, so their moment contributions stack along the
# fuselage. Three distinct conditions are evaluated (symmetric pull-up, dive + max tail
# load, hard landing) and the worst governs. The skin is sized for BOTH CFRP and
# aluminium and the lighter is returned. FUS_MASS_METHOD ("physics" | "regression")
# selects this method or the legacy empirical USAF/Nicolai formula; all constants live
# in parameters.py.

_FUS_MATERIALS = {
    "CFRP": {"sigma": SIGMA_ALLOW_CFRP, "rho": RHO_CFRP, "t_min": T_SKIN_MIN_CFRP, "E": CFRP["E"]},
    "AL":   {"sigma": SIGMA_ALLOW_AL,   "rho": RHO_AL,   "t_min": T_SKIN_MIN_AL,   "E": E_AL},
}

_FUS_N_STATIONS = 400   # beam discretisation along the fuselage length


def _fus_inertia_load(x, mtow_kg, n_z):
    """Downward distributed inertia line-load w(x) [N/m] over the fuselage at load
    factor n_z. The fuselage-borne mass (FUS_BORNE_MASS_FRAC * MTOW) is split into the
    payload box (M_PAYLOAD, concentrated over X_PAYLOAD +- L_PAYLOAD_BOX/2) and a
    uniform structural/systems strip over the whole length; the rest of the aircraft
    hangs on the wings and does not load the fuselage shell."""
    m_borne = FUS_BORNE_MASS_FRAC * mtow_kg
    m_uniform = max(m_borne - M_PAYLOAD, 0.0)
    w = np.full_like(x, m_uniform * n_z * G / L_FUS)

    box_lo = X_PAYLOAD - L_PAYLOAD_BOX / 2.0
    box_hi = X_PAYLOAD + L_PAYLOAD_BOX / 2.0
    w_pay = M_PAYLOAD * n_z * G / L_PAYLOAD_BOX
    in_box = (x >= box_lo) & (x <= box_hi)
    return w + np.where(in_box, w_pay, 0.0)


def _solve_two_supports(x, w_down, applied_up, xs1, xs2):
    """Two unknown upward reactions (R1 at xs1, R2 at xs2) of a statically determinate
    free-free beam, from vertical + moment equilibrium. w_down is the distributed load
    (N/m, down +); applied_up is a list of known (x_i, F_i) point loads (up +). With
    both equilibrium equations satisfied the integrated moment closes to ~0 at the free
    ends, so its interior peak is the true superimposed maximum."""
    W = float(cumulative_trapezoid(w_down, x, initial=0.0)[-1])              # total down
    Mw = float(cumulative_trapezoid(w_down * (x - xs1), x, initial=0.0)[-1])  # its moment about xs1
    sum_f = sum(f for _, f in applied_up)
    sum_fx = sum(f * (xi - xs1) for xi, f in applied_up)
    R2 = (Mw - sum_fx) / (xs2 - xs1)
    R1 = (W - sum_f) - R2
    return R1, R2


def _fus_beam_moment(x, w_down, point_loads_up):
    """Shear V(x) [N] and bending moment M(x) [N.m] of the free-free fuselage beam from
    the superimposed loading: distributed w_down (N/m, down +) plus point loads
    point_loads_up = [(x_i, F_i), ...] (up +). Two cumulative-trapezoid passes, the same
    shear->moment integration style as analyze_cantilever_distributed_load."""
    V = -cumulative_trapezoid(w_down, x, initial=0.0)
    for xi, fi in point_loads_up:
        V = V + fi * (x >= xi)
    M = cumulative_trapezoid(V, x, initial=0.0)
    return V, M


def _fus_condition(name, mtow_kg, n_z, applied_up, xs1, xs2):
    """Build one design condition: lay the distributed inertia and the known external
    point loads (applied_up) on the beam, solve the two support reactions, integrate to
    the superimposed V/M diagram and record the peak |M|."""
    x = np.linspace(0.0, L_FUS, _FUS_N_STATIONS)
    w = _fus_inertia_load(x, mtow_kg, n_z)
    R1, R2 = _solve_two_supports(x, w, applied_up, xs1, xs2)
    loads = list(applied_up) + [(xs1, R1), (xs2, R2)]
    V, M = _fus_beam_moment(x, w, loads)
    return {
        "name": name, "x": x, "w": w, "V": V, "M": M,
        "peak": float(np.max(np.abs(M))), "n_z": n_z,
        "applied": list(applied_up),
        "supports": [(xs1, R1), (xs2, R2)],
    }


def _fus_conditions(mtow_kg):
    """The three superimposed design conditions sizing the fuselage bending."""
    conds = []

    # C1 - Symmetric maneuver pull-up at the limit load factor: distributed inertia
    #      reacted by the two wings; tail trim load is small in a steady pull-up (~0).
    conds.append(_fus_condition(
        "C1 maneuver pull-up (n=N_W)", mtow_kg, N_W, [], X_WING_F, X_WING_R))

    # C2 - Dive at max tail load: the V-tail aero download (same q_dive*S_TAIL*C_N_TAIL_MAX
    #      tail_mass uses, both fins, vertical component) STACKED with limit-load inertia.
    q_dive = 0.5 * RHO_ORIGIN * (V_DIVE_FACTOR * V_CRUISE) ** 2
    f_tail = 2.0 * q_dive * S_TAIL * C_N_TAIL_MAX * np.cos(np.radians(V_ANGLE))
    conds.append(_fus_condition(
        "C2 dive + max tail load", mtow_kg, N_W,
        [(X_TAIL, -f_tail)], X_WING_F, X_WING_R))      # tail download -> down (-up)

    # C3 - Hard landing / VTOL: the whole airframe decelerates at N_LIMIT on the skids.
    #      The wing-borne mass pushes DOWN through the wing joints; the skid footprint
    #      (two contact points) reacts everything; fuselage inertia is distributed.
    m_wing_borne = max((1.0 - FUS_BORNE_MASS_FRAC) * mtow_kg, 0.0)
    f_front = (1.0 - F_REAR_WING) * m_wing_borne * N_LIMIT * G
    f_rear = F_REAR_WING * m_wing_borne * N_LIMIT * G
    skid_lo = X_GEAR - L_SKID_RAIL / 2.0
    skid_hi = X_GEAR + L_SKID_RAIL / 2.0
    conds.append(_fus_condition(
        "C3 hard landing (n=N_LIMIT)", mtow_kg, N_LIMIT,
        [(X_WING_F, -f_front), (X_WING_R, -f_rear)], skid_lo, skid_hi))

    return conds


def _fus_design_moment(mtow_kg, return_conditions=False):
    """Ultimate design bending moment [N.m] = STRUCT_SF * worst peak |M| across the
    three superimposed conditions (scalar, for reporting)."""
    conds = _fus_conditions(mtow_kg)
    governing = max(conds, key=lambda c: c["peak"])
    m_ult = STRUCT_SF * governing["peak"]
    if return_conditions:
        return m_ult, conds, governing
    return m_ult


def _fus_moment_envelope(conds):
    """Per-STATION ultimate |M| envelope [N.m] across all conditions (shared x grid):
    the worst moment seen at each fuselage station, so a variable-gauge skin can be
    thinner where every condition is lightly loaded (nose/tail) and thicker at the peak."""
    x = conds[0]["x"]
    m_env = np.zeros_like(x)
    for c in conds:
        m_env = np.maximum(m_env, np.abs(c["M"]))
    return x, STRUCT_SF * m_env


def _fus_skin_profile(m_env, material):
    """Required skin gauge [m] AT EACH STATION = max(min-gauge, yield, shell-buckling).

    Thin cylinder in bending, radius R = D_FUS/2:
      applied stress      sigma   = M / (pi R^2 t)
      yield gauge         t_yield = M / (sigma_allow * pi R^2)            (material strength)
      buckling allowable  sigma_cr= C E (t/R)  ->  t_buckle = sqrt(M / (pi R C E))
    Buckling (not yield) governs a thin monocoque shell, and t_buckle ~ sqrt(M) ~ sqrt(MTOW),
    so the loaded region grows with the aircraft while the ends stay at min gauge."""
    r = D_FUS / 2.0
    t_yield = m_env / (material["sigma"] * np.pi * r ** 2)
    t_buckle = np.sqrt(m_env / (np.pi * r * FUS_SHELL_BUCKLING_C * material["E"]))
    t_floor = np.full_like(m_env, material["t_min"])
    t_skin = np.maximum.reduce([t_floor, t_yield, t_buckle])
    return t_skin, t_yield, t_buckle


def _fus_shell_mass(mtow_kg, material, conds=None):
    """Structural mass [kg] of the variable-gauge thin-walled shell for one material set.

    Sizes the skin gauge per station against the moment ENVELOPE (yield + buckling),
    integrates the variable-thickness skin volume over the length, then recovers
    frames/floor/fittings with FUS_PRIMARY_FRACTION. Buckling makes the loaded-region
    gauge - and hence the mass - scale with MTOW."""
    if conds is None:
        _, conds, _ = _fus_design_moment(mtow_kg, return_conditions=True)
    x, m_env = _fus_moment_envelope(conds)
    t_skin, t_yield, t_buckle = _fus_skin_profile(m_env, material)

    circ = np.pi * D_FUS * FUS_AREA_CONE_FACTOR                       # shell circumference [m]
    vol_skin = circ * float(cumulative_trapezoid(t_skin, x, initial=0.0)[-1])
    m_skin = material["rho"] * vol_skin

    ipk = int(np.argmax(t_skin))                                     # mode setting the peak gauge
    if t_buckle[ipk] >= max(material["t_min"], t_yield[ipk]):
        governs = "buckling"
    elif t_yield[ipk] >= material["t_min"]:
        governs = "yield"
    else:
        governs = "min-gauge"

    return {
        "mass": m_skin / FUS_PRIMARY_FRACTION, "m_skin": m_skin,
        "t_max": float(np.max(t_skin)), "t_min_gauge": material["t_min"],
        "governs": governs, "x": x, "m_env": m_env,
        "t_skin": t_skin, "t_yield": t_yield, "t_buckle": t_buckle,
    }


def _fuselage_mass_regression(mtow_kg):
    """Legacy empirical fuselage weight (USAF/Nicolai-type GA regression, lb/ft -> kg).
    Retained as the FUS_MASS_METHOD == 'regression' branch for fallback / A-B checks."""
    return (0.453592
            * (14.86 * ((mtow_kg * 2.20462) ** 0.144) * ((L_FUS * 3.28084) ** 0.778)
               / ((PER_FUS_MAX * 3.28084) ** 0.778))
            * ((L_FUS * 3.28084) ** 0.383) * N_PAX ** 0.455)


def fuselage_mass(mtow_kg, method=None, return_details=False):
    """Fuselage structural mass [kg].

    method (default FUS_MASS_METHOD): "physics" sizes the thin-walled shell-beam for the
    worst of three superimposed load conditions, evaluates CFRP and aluminium and returns
    the lighter; "regression" returns the legacy empirical formula. return_details=True
    returns the full sizing dict (winning material, governing condition, per-condition
    V/M diagrams, per-material breakdown) instead of the float, like landing_gear_mass."""
    method = method or FUS_MASS_METHOD

    if method == "regression":
        m = _fuselage_mass_regression(mtow_kg)
        return {"mass": m, "method": "regression", "material": None} if return_details else m

    m_design, conds, governing = _fus_design_moment(mtow_kg, return_conditions=True)
    by_material = {
        name: _fus_shell_mass(mtow_kg, mat, conds)
        for name, mat in _FUS_MATERIALS.items()
    }
    best_name = min(by_material, key=lambda n: by_material[n]["mass"])
    best = by_material[best_name]

    if not return_details:
        return best["mass"]
    return {
        "mass": best["mass"], "method": "physics", "material": best_name,
        "m_design": m_design, "governing_condition": governing["name"],
        "t_max": best["t_max"], "governs": best["governs"],
        "conditions": conds, "by_material": by_material,
    }


# Wing (dynamic sizing from design-point W/S + Class II mass)
def _panel_geometry(area_per_wing_m2):
    """Trapezoidal planform descriptors for ONE box-wing of the given area.

    Both Prandtl box-wings share the span (b_aw = b_fw = WING_SPAN) and taper, so a
    panel is fully fixed by its area: a bigger panel simply has a longer chord (and
    a deeper spar). Returns the per-wing area, AR, mean/root/tip chord and MAC."""
    aspect_ratio_per_wing = WING_SPAN ** 2 / area_per_wing_m2
    mean_chord_m = WING_SPAN / aspect_ratio_per_wing
    root_chord_m = 2 * area_per_wing_m2 / ((1 + TAPER_W) * WING_SPAN)
    tip_chord_m = TAPER_W * root_chord_m
    mac_m = (2.0 / 3.0) * root_chord_m * ((1 + TAPER_W + TAPER_W ** 2) / (1 + TAPER_W))
    return {
        "area_per_wing_m2": area_per_wing_m2,
        "aspect_ratio": aspect_ratio_per_wing,
        "mean_chord_m": mean_chord_m,
        "mac_m": mac_m,
        "root_chord_m": root_chord_m,
        "tip_chord_m": tip_chord_m,
    }


def wing_geometry(mtow_kg):
    # -- Total area from selected design point: S = W / (W/S) --
    weight_n = mtow_kg * G
    total_area_m2 = weight_n / WING_LOADING_N
    aspect_ratio = WING_SPAN ** 2 / total_area_m2

    # -- Front / aft areas from the aft/total split (S_AFT_TO_S_TOTAL). The two
    #    box-wings size to DIFFERENT areas (hence chords/MACs) whenever the split
    #    is off 0.5, so the aft/total split is what makes the wing mass below
    #    respond to it. front = (1 - aft fraction). --
    aft_area_m2   = total_area_m2 * S_AFT_TO_S_TOTAL
    front_area_m2 = total_area_m2 - aft_area_m2
    front = _panel_geometry(front_area_m2)
    aft   = _panel_geometry(aft_area_m2)

    # Aggregate descriptors keep describing the SYMMETRIC (equal-area) per-wing
    # planform so the legacy printout / callers stay valid; they equal front/aft
    # exactly at the baseline 50/50 split. Structural sizing uses front/aft.
    half = _panel_geometry(total_area_m2 / 2.0)
    return {
        "weight_n": weight_n,
        "total_area_m2": total_area_m2,
        "span_m": WING_SPAN,
        "aspect_ratio": aspect_ratio,
        "front": front,
        "aft": aft,
        # ---- backward-compatible symmetric per-wing descriptors ----
        "area_per_wing_m2": half["area_per_wing_m2"],
        "mean_chord_m": half["mean_chord_m"],
        "mac_m": half["mac_m"],
        "root_chord_m": half["root_chord_m"],
        "tip_chord_m": half["tip_chord_m"],
    }
def _panel_mass(mtow_kg, panel, f_lift, semi_span_m):
    """Class-II structural mass [kg] of ONE box-wing carrying lift fraction f_lift.

    Spar caps are sized for the elliptical-lift root moment (∫M dy = L·s²/8), the
    spar depth following the panel's own mean chord; skins are min-gauge CFRP over
    the panel area. The 0.76 primary fraction recovers ribs + secondary structure."""
    h_spar_mean = TIP_TO_CHORD_W * panel["mean_chord_m"]   # spar depth [m]
    l_wing = f_lift * N_W * mtow_kg * G

    # Correct spar cap volume for elliptical lift distribution.
    # vol_caps = STRUCT_SF · L·s² / (8·σ·h)
    vol_caps = STRUCT_SF * l_wing * semi_span_m**2 / (8 * SIGMA_ALLOW_CFRP * h_spar_mean)
    m_spar   = 1.4 * vol_caps * RHO_CFRP    # caps + web, CFRP

    # Skins: min-gauge CFRP (8-ply prepreg), upper + lower surface
    m_skin   = 2 * panel["area_per_wing_m2"] * T_SKIN_MIN_CFRP * RHO_CFRP

    return (m_spar + m_skin) / 0.76


def wing_mass(mtow_kg, geometry=None):
    # -- Size front and rear box-wings independently. Each carries its lift
    #    fraction (F_REAR_WING) on its OWN area/chord, so a non-0.5 area split
    #    (S_AFT_TO_S_TOTAL) changes the per-wing spar depth + skin area and the
    #    total wing mass responds to the split. --
    if geometry is None:
        geometry = wing_geometry(mtow_kg)

    s = geometry["span_m"] / 2.0                        # semi-span [m]
    # no h_eff correction: wing is horizontal, bending is about the chord axis
    m_front = _panel_mass(mtow_kg, geometry["front"], 1.0 - F_REAR_WING, s)
    m_rear  = _panel_mass(mtow_kg, geometry["aft"],   F_REAR_WING,       s)

    ### Buckling calculations (from winglet)

    return m_front + m_rear

#Extra methods for sizing ribs n stuff
def analyze_cantilever_distributed_load(L, w_func, plot=False, x_eval=None):
    """
    Calculates reactions and plots load, shear, and moment diagrams
    for a cantilever beam (fixed at x=0) with a distributed load function.

    Parameters:
    L (float): Length of the beam (m)
    w_func (callable): Function taking position 'x' and returning load magnitude (N/m).
    plot (bool): If True, displays the shear and moment diagrams.
    x_eval (float, optional): Specific location along the beam (0 to L) to evaluate the moment.

    Returns:
    float: The bending moment at x_eval (if provided), otherwise the maximum absolute moment.
    """

    # 1. Calculate Reaction Forces at the Wall (x=0)
    W, _ = quad(w_func, 0, L)

    moment_integrand = lambda x: x * w_func(x)
    M_wall, _ = quad(moment_integrand, 0, L)

    R_y = W

    if plot:
        print("-" * 35)
        print("WALL REACTIONS (at x = 0)")
        print("-" * 35)
        print(f"Vertical Reaction (R_y): {R_y:.2f} N (Upward)")
        print(f"Reaction Moment (M):     {M_wall:.2f} N·m (Counter-Clockwise)\n")

    # 2. Discretize the beam for array operations
    x = np.linspace(0, L, 1000)
    w_arr = np.array([w_func(xi) for xi in x])

    # 3. Calculate Shear (V)
    integral_w = cumulative_trapezoid(w_arr, x, initial=0)
    V = R_y - integral_w

    # 4. Calculate Bending Moment (M)
    M = -M_wall + cumulative_trapezoid(V, x, initial=0)

    if plot:
        # 5. Plot the Diagrams
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

        # Applied Load Diagram
        ax0.plot(x, -w_arr, color='green', linewidth=2)
        ax0.fill_between(x, -w_arr, 0, color='green', alpha=0.2)
        ax0.axhline(0, color='black', linewidth=1)
        ax0.set_ylabel('Distributed Load\n$w(x)$ (N/m)')
        ax0.set_title('Cantilever Beam Analysis (Fixed at $x=0$)')
        ax0.grid(True, linestyle='--', alpha=0.6)

        # Shear Diagram
        ax1.plot(x, V, color='blue', linewidth=2)
        ax1.fill_between(x, V, 0, color='blue', alpha=0.2)
        ax1.axhline(0, color='black', linewidth=1)
        ax1.set_ylabel('Shear Force\n$V$ (N)')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Moment Diagram
        ax2.plot(x, M, color='red', linewidth=2)
        ax2.fill_between(x, M, 0, color='red', alpha=0.2)
        ax2.axhline(0, color='black', linewidth=1)
        ax2.set_ylabel('Bending Moment\n$M$ (N·m)')
        ax2.set_xlabel('Position from wall, $x$ (m)')
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        # plt.show()

    return M, x
def calculate_Ibeam_moment_of_inertia(root_chord, thick_chord_ratio, b, t_f, t_w, return_area=False):
    h = root_chord * thick_chord_ratio - 2 * t_f

    h_inner = h - 2 * t_f

    I_x = (b * h ** 3) / 12 - ((b - t_w) * h_inner ** 3) / 12

    if return_area:
        return ((h - 2 * t_f) * t_w + 2 * t_f * b)

    return I_x
def calc_individual_rib_spacing(stress, buckling_coeff, young_mod, skin_thickness, poisson_ratio):
    num = buckling_coeff * (np.pi ** 2) * young_mod * skin_thickness ** 2
    den = 12 * stress * (1 - poisson_ratio ** 2)
    return np.sqrt(num / den)
def calc_total_rib_spacing(max_stress_beam, wing_length, root_chord, thick_chord_ratio, I, number_of_beams, M, x,
                           buckling_coeff, young_mod, skin_thickness, poisson_ratio):
    # Using the rearranged critical buckling formula to find rib spacing
    curr_rib = calc_individual_rib_spacing(max_stress_beam, buckling_coeff, young_mod, skin_thickness, poisson_ratio)
    ribslst = []

    while curr_rib < wing_length:
        ribslst.append(curr_rib)
        Moment_at_point = float(np.interp(curr_rib, x, M))
        stress_beam = Moment_at_point * root_chord * thick_chord_ratio / 2 / (I * number_of_beams)
        curr_rib += calc_individual_rib_spacing(stress_beam, buckling_coeff, young_mod, skin_thickness, poisson_ratio)

    ribslst.insert(0, 0)
    ribslst.append(wing_length)

    return ribslst
# Winglet sizing
def winglet_mass(root_chord):
    def winglet_lift(x):
        return front_wing_distribution - x * (front_wing_distribution + back_wing_distribution) / winglet_length

    ###PARAMETERS

    front_wing_distribution = 500
    back_wing_distribution = 300

    # Wing & Winglet dimensions
    thick_chord_ratio = TIP_TO_CHORD_W
    taper_ratio = WINGLET_TAPER_RATIO

    wings_vertical_spacing = H_GAP_WINGS  # UPDATE THESE LATER
    wings_horizontal_spacing = WING_STAGGER  # UPDATE THESE LATER

    winglet_length = np.sqrt(wings_horizontal_spacing ** 2 + wings_vertical_spacing ** 2)  # ???
    winglet_area = WingletGeometry.S_winglet # From utku, UPDATE LATER
    winglet_skin_thickness = WINGLET_SKIN_THICKNESS

    ### Characteristics unique to the winglet, NOT USED ANYWHERE ELSE
    number_of_beams = WINGLET_BEAM_NUMBER
    rib_thickness = WINGLET_RIB_THICKNESS

    # I_BEAM
    flange_length = WINGLET_SPAR_FLANGE_LENGTH
    flange_thickness = WINGLET_SPAR_FLANGE_THICKNESS  # This is what will impact the mmoi the most
    beam_thickness = WINGLET_SPAR_BEAM_THICKNESS  # m

    # Material characteristics
    allowable_stress = 278e6 * 0.8
    poisson_ratio = 0.33
    buckling_coeff = 4
    young_mod = 70e9
    density = 2700

    I = calculate_Ibeam_moment_of_inertia(root_chord, thick_chord_ratio, flange_length, flange_thickness,
                                          beam_thickness)

    M, x = analyze_cantilever_distributed_load(winglet_length, winglet_lift, plot=False)

    M_max = float(np.max(np.abs(M)))

    max_stress_beam = M_max * root_chord * thick_chord_ratio / 2 / (I * number_of_beams)

    ribslst = calc_total_rib_spacing(max_stress_beam, winglet_length, root_chord, thick_chord_ratio, I, number_of_beams,
                                     M, x, buckling_coeff, young_mod, winglet_skin_thickness, poisson_ratio)

    surface_area = winglet_area / winglet_length

    rib_surface_area = surface_area / ((
                                                   taper_ratio - 1) * 0.5 + 1) ** 2 * 0.6  # Assuming the rib area is 0.6 times the airfoil cross section due to holes & cutouts

    # ribs_mass = rib_surface_area * density * rib_thickness * len(ribslst)
    ribs_mass = 0

    for rib_pos in ribslst:
        point_taper = ((taper_ratio - 1) / winglet_length * rib_pos + 1)
        rib_volume = surface_area * point_taper ** 2 * rib_thickness
        ribs_mass += rib_volume * density

    #print(ribs_mass)
    skin_mass = winglet_area * winglet_skin_thickness

    Ibeam_area = calculate_Ibeam_moment_of_inertia(root_chord, thick_chord_ratio, flange_length, flange_thickness,
                                                   beam_thickness, return_area=True)
    Ibeam_mass = Ibeam_area * winglet_length * density

    #print(f"The max stress is {max_stress_beam * 10 ** -6} Mpa")
    #print(f"Number of ribs: {len(ribslst) + 2}")
    #print(
    #    f"The total winglet mass is ({ribs_mass} + {skin_mass} + {Ibeam_mass} )* 2 = {(ribs_mass + skin_mass + Ibeam_mass) * 2}")

    ###once the winglet dimensions have been done sizing #AAluminum currently being used for structure sizing

winglet_mass(1.2)



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
    return whole_gear(k, Ue, Up, Fmax, mass1, dy, dmax, CROSSTUBE_COUNT, reusable=False,
                      track_m=FUSE_WIDTH + 2 * L)


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
    return whole_gear(k, Ue, Up, Fmax, mass1, dy, dmax, LEAF_COUNT, reusable=True,
                      track_m=FUSE_WIDTH + 2 * L)


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
    return whole_gear(k, Ue, Up, Fmax, mass1, dfail, dmax, COMPOSITE_COUNT, reusable=False,
                      track_m=FUSE_WIDTH + 2 * L)


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
    return whole_gear(k, Ue, Up, Fmax, mass1, dy, dmax, CRUSH_COUNT, reusable=False,
                      track_m=FUSE_WIDTH + 2 * L)


def elastomeric(x, m):
    """(E) 4 elastomeric block mounts, hysteretic, reusable.  x = kb, dm, loss"""
    kb, dm, loss = x
    Ue = 0.5 * kb * dm**2
    Up = loss * Ue
    Fmax = kb * dm
    mass1 = ELASTO_FIXED_MASS + ELASTO_MASS_PER_N * Fmax
    # The 2-cross-tube + 4-leg skeleton is added for every architecture by whole_gear; the
    # elastomer's track is set by the arm length L_ARM (the lateral splay to the skids).
    return whole_gear(kb, Ue, Up, Fmax, mass1, dm, dm, ELASTO_COUNT, reusable=True,
                      track_m=FUSE_WIDTH + 2 * L_ARM)


def skid_rail_mass():
    """Mass [kg] of ONE longitudinal skid rail, modelled as a hollow aluminium tube.
       Geometric estimate: m = rho * A * L, A the annular cross-section."""
    d = D_SKID_RAIL - 2 * T_SKID_RAIL
    A = math.pi / 4 * (D_SKID_RAIL**2 - d**2)
    return RHO_AL * A * L_SKID_RAIL


def mount_frame_mass():
    """Mass [kg] of the gear's load-path skeleton, shared by every architecture: CROSSTUBE_COUNT
    transverse cross-tubes (each spanning CROSS_SPAN) + ELASTO_COUNT legs/arms (each L_ARM long)
    carrying the energy absorbers down to the skids. Hollow aluminium tubes, geometric estimate
    m = rho * A * L.

    Every concept sits on this same skeleton (2 cross-tubes + 4 legs), so whole_gear adds it
    for all architectures. The bending members additionally span transversely in their own mass1,
    so that adds a small, deliberately conservative overlap with the cross-tubes (a few kg, far
    below the gear-selection margin)."""
    d = D_FRAME_TUBE - 2 * T_FRAME_TUBE
    A = math.pi / 4 * (D_FRAME_TUBE**2 - d**2)
    length_total = CROSSTUBE_COUNT * CROSS_SPAN + ELASTO_COUNT * L_ARM
    return RHO_AL * A * length_total


def whole_gear(k, Ue, Up, Fmax, mass1, dy, dmax, count, reusable, track_m=0.0):
    """Scale one member's properties to the whole gear (count members in parallel) and add the
    two shared skid rails plus the shared 2-cross-tube + 4-leg skeleton (mount_frame_mass), which
    every architecture carries. Members deflect together, so stiffness, energy, load and mass add
    up; stroke (dy, dmax) does not. track_m is the spanwise footprint (lateral skid track)."""
    return dict(
        k=count * k, Ue=count * Ue, Up=count * Up, Fmax=count * Fmax,
        mass=count * mass1 + N_SKID * skid_rail_mass() + mount_frame_mass(),
        dy=dy, dmax=dmax, reusable=reusable, track_m=track_m,
    )


# Sizing with scipy SLSQP: minimise mass s.t. the drop constraints -----------
#   g1: Ue        - E_L        (elastic at limit)
#   g2: Ue + Up   - E_R        (survive reserve)
#   g3: N_LIMIT*W - Fmax       (peak-decel cap)
#   g4: ENVELOPE  - dmax       (fits stroke)
#   g5: FOOTPRINT_MAX_SPAN - track_m   (lateral track fits the spanwise footprint)

def constraints(x, fn, mat, m_eff):
    r = fn(x, mat)
    return np.array([
        r["Ue"]             - E_limit(r["dy"], m_eff),
        r["Ue"] + r["Up"]   - E_reserve(r["dmax"], m_eff),
        N_LIMIT * m_eff * G - r["Fmax"],
        ENVELOPE            - r["dmax"],
        FOOTPRINT_MAX_SPAN  - r["track_m"],
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
# Four small steps, written out explicitly (plain loops, not comprehensions) for clarity:
#   metrics()     - gather the 7 trade-off criteria for one architecture
#   normalise()   - rescale one criterion column to [0, 1] across the architectures
#   score_table() - weighted sum of the normalised criteria -> one score per architecture
#   score()       - run the above on the feasible architectures with the nominal weights
#
# Each entry in WEIGHTS is a 3-tuple: (direction, base_weight, uncertainty).
#   direction    "max" if a higher raw value is better, "min" if lower is better
#   base_weight  how much this criterion counts in the weighted sum
#   uncertainty  relative scatter used only by the Monte-Carlo sensitivity study

def metrics(arch):
    """The seven trade-off criteria for one sized architecture, as a flat dict.

    SEA / mass / npk come from the sizing result; `reusable` is 1.0 or 0.0; the
    qualitative scores (tunable, cert_risk, cost) come from QUAL, defaulting to 0.5 if the
    architecture is not listed there."""
    qualitative = QUAL.get(arch["name"], {})

    criteria = {}
    criteria["SEA"] = arch["SEA"]
    criteria["mass"] = arch["mass"]
    criteria["npk"] = arch["npk"]
    if arch["reusable"]:
        criteria["reusable"] = 1.0
    else:
        criteria["reusable"] = 0.0
    criteria["tunable"] = qualitative.get("tunable", 0.5)
    criteria["cert_risk"] = qualitative.get("cert_risk", 0.5)
    criteria["cost"] = qualitative.get("cost", 0.5)
    return criteria


def normalise(values, direction):
    """Rescale one criterion's values to [0, 1] across the architectures.

    direction == "max": larger raw value is better, so the largest maps to 1.0 (min-max).
    direction == "min": smaller raw value is better, so the smallest maps to 1.0 (min-max).
    direction == "min_ratio": smaller is better AND magnitude matters: score = best/value, so
        the lightest maps to 1.0 and a design k times heavier scores 1/k (independent of the
        rest of the set). Used for mass so a much heavier gear is penalised by its true ratio,
        not merely ranked last.
    If all values are equal the criterion cannot discriminate, so everything scores 1.0."""
    if direction == "min_ratio":
        best = min(values)
        normalised_values = []
        for value in values:
            if value > 0:
                normalised_values.append(best / value)
            else:
                normalised_values.append(1.0)
        return normalised_values

    lowest = min(values)
    highest = max(values)
    spread = highest - lowest

    # Every architecture is equal on this criterion -> it cannot tell them apart.
    if spread < 1e-12:
        all_ones = []
        for _ in values:
            all_ones.append(1.0)
        return all_ones

    normalised_values = []
    for value in values:
        if direction == "max":
            # Larger is better: lowest -> 0.0, highest -> 1.0.
            normalised_value = (value - lowest) / spread
        else:
            # Smaller is better: highest -> 0.0, lowest -> 1.0.
            normalised_value = (highest - value) / spread
        normalised_values.append(normalised_value)
    return normalised_values


def score_table(metrics_by_arch, weights):
    """Weighted-sum trade-off score in [0, 1] for each architecture.

    metrics_by_arch maps architecture name -> its metrics() dict. Each criterion column is
    normalised to [0, 1] across the architectures (per WEIGHTS' max/min direction), then the
    normalised criteria are combined with `weights` re-normalised to sum to 1."""
    arch_names = list(metrics_by_arch)

    # Start each architecture with an empty dict of normalised criteria.
    normalised = {}
    for name in arch_names:
        normalised[name] = {}

    # Step 1: normalise every criterion column to [0, 1] across the architectures.
    for criterion in WEIGHTS:
        direction = WEIGHTS[criterion][0]

        # Collect this criterion's raw value from every architecture (the "column").
        column = []
        for name in arch_names:
            column.append(metrics_by_arch[name][criterion])

        # Normalise the column, then store one value back per architecture.
        normalised_column = normalise(column, direction)
        for index in range(len(arch_names)):
            name = arch_names[index]
            normalised[name][criterion] = normalised_column[index]

    # Step 2: add up the weights so we can re-normalise them to sum to 1.
    weight_total = 0.0
    for criterion in WEIGHTS:
        weight_total = weight_total + weights[criterion]

    # Step 3: weighted sum of the normalised criteria, one score per architecture.
    scores = {}
    for name in arch_names:
        running_total = 0.0
        for criterion in WEIGHTS:
            relative_weight = weights[criterion] / weight_total
            running_total = running_total + relative_weight * normalised[name][criterion]
        scores[name] = running_total
    return scores


def score(rows):
    """Weighted trade-off score for each FEASIBLE architecture, using the nominal base
    weights in WEIGHTS. Returns {} if nothing is feasible."""
    feasible_archs = []
    for arch in rows:
        if arch["feasible"]:
            feasible_archs.append(arch)
    if not feasible_archs:
        return {}

    # Metrics for each feasible architecture, keyed by name.
    metrics_by_arch = {}
    for arch in feasible_archs:
        metrics_by_arch[arch["name"]] = metrics(arch)

    # Nominal weights = the base_weight (middle element) of each WEIGHTS entry.
    base_weights = {}
    for criterion in WEIGHTS:
        base_weights[criterion] = WEIGHTS[criterion][1]

    return score_table(metrics_by_arch, base_weights)


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
    mass_ratio = m_eff / GEAR_BOUNDS_REF_MASS

    scaled = {}
    for name, cfg in GEAR_OPT_BOUNDS.items():
        scale_exponents = cfg.get("scale", [0.0] * len(cfg["varnames"]))

        scaled_x0 = []
        scaled_bounds = []
        for seed, (low, high), exponent in zip(cfg["x0"], cfg["bounds"], scale_exponents):
            # Grow the upper bound with mass. max(1.0, ...) means a mass below the
            # reference never shrinks the calibrated bound.
            growth = max(1.0, mass_ratio ** exponent)
            high_scaled = high * growth

            # Keep the seed inside the (possibly widened) bounds.
            seed_scaled = seed * growth
            if seed_scaled < low:
                seed_scaled = low
            if seed_scaled > high_scaled:
                seed_scaled = high_scaled

            scaled_x0.append(seed_scaled)
            scaled_bounds.append((low, high_scaled))

        scaled[name] = dict(x0=scaled_x0, bounds=scaled_bounds,
                            varnames=cfg["varnames"], material=cfg["material"])
    return scaled
def size_all_architectures(m_eff):
    """Size every architecture for the effective drop mass m_eff [kg] using bounds that
    auto-scale with the mass (see _scaled_gear_bounds). Returns the list of result dicts."""
    rows = []
    for name, cfg in _scaled_gear_bounds(m_eff).items():
        rows.append(size(name, _GEAR_CONCEPTS[name], cfg["material"],
                         cfg["x0"], cfg["bounds"], cfg["varnames"], m_eff))
    return rows
def landing_gear_mass(mtow_kg, return_details=True):
    """Skid landing-gear mass [kg] from a five-architecture drop trade study.

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

    scores = score(rows)                             # {architecture name: weighted score}
    if not scores:
        warnings.warn(
            "landing_gear_mass: no feasible architecture - returning None (the MTOW loop "
            "falls back to a 3% class-1 gear estimate). Check the drop inputs (H_L, "
            "N_LIMIT, ENVELOPE) and the search bounds in GEAR_OPT_BOUNDS.",
            stacklevel=2,
        )
        return None

    # Pick the architecture with the highest weighted score.
    winner_name = max(scores, key=scores.get)
    winner_score = scores[winner_name]

    # Find that architecture's full sizing-result row.
    winner = None
    for row in rows:
        if row["name"] == winner_name:
            winner = row
            break

    if not return_details:
        return winner["mass"]

    # Return a copy of the winning row plus the aliases / extras callers expect.
    details = dict(winner)
    details["m_gear"] = winner["mass"]               # alias used by mtow_sizing
    details["arch"] = winner_name                    # alias used by mtow_sizing
    details["geom"] = {}                             # these architectures have no Do/Di geometry
    details["score"] = winner_score
    details["all_architectures"] = rows
    return details
def geom_for_sketch(details):
    """Drawable landing-gear geometry derived from the sized winner (a landing_gear_mass dict).

    Architecture-agnostic: the spanwise track and skid length come from the model. L_eff is the
    moment arm from the hinge to the skid reaction (the bending archs expose 'L'; the
    discrete-mount/elastomeric frame uses the arm length L_ARM). All lengths in metres."""
    p = details.get("params", {})
    return dict(
        track_m=details["track_m"],
        L_eff=p.get("L", L_ARM),
        skid_len=L_SKID_RAIL,
        clearance=0.30,                              # [m] static belly-to-ground clearance
        fuse_width=FUSE_WIDTH,
    )


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
    #m_single = 0.165 * ((max_power_kw * (1 + PM)) / N_MOTOR)
    m_single = 21.4 + 6.8 + 4 #24 is actual kg of motor where 4 is the variable pitch controller estimate plus the 6.8 for inverter
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


# Miscellaneous
def misc_mass(mtow_kg):
    return 0.10 * mtow_kg + 32 #kg of the thermal battery management system 


def hinge_mass(mtow_kg):
    return 0.05 * mtow_kg