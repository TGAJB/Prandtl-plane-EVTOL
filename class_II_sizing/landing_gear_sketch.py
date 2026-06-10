"""
landing_gear_sketch.py
Schematic (front + side view) of the skid landing-gear design.

Configuration (per the design intent):
  * a structural cross-body TUBE bridges the fuselage, with a PLASTIC HINGE at each
    fuselage attachment;
  * the ELASTOMERIC SHOCK ABSORBERS sit at those hinges;
  * the SKIDS come off the ends of the cross-tube at an angle and run along the fuselage.

This sketch FOLLOWS THE SIZING: it reads the converged MTOW from mtow_sizing and draws the
architecture the trade study actually selects (normally "E elastomeric"). All geometry comes
from mass_components.geom_for_sketch(): the spanwise track (capped at FOOTPRINT_MAX_SPAN), the
skid rail length L_SKID_RAIL, and L_eff (moment arm from hinge to skid reaction = the winner's
sized cantilever length, or the arm length L_ARM for the discrete-mount elastomeric design).

Run:  python3 class_II_sizing/landing_gear_sketch.py   ->  landing_gear_sketch.png
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import class_II_sizing.mass_components as mc
import class_II_sizing.mtow_sizing as mtow
from parameters import FOOTPRINT_MAX_SPAN, L_FUS

# --- design point: follow the converged MTOW from the sizing loop ------------
mtow.load_final_design_state()   # solve once (cached) so MTOW_FINAL is populated
MTOW = mtow.MTOW_FINAL           # converged MTOW [kg]

# Size the gear at the converged MTOW and draw the ACTUAL weighted winner.
gear = mc.landing_gear_mass(MTOW)
arch = gear["arch"]              # winning architecture, e.g. "E elastomeric"
params = gear["params"]          # winner's sized design variables

# Geometry comes from the model + the winning architecture (geom_for_sketch).
g = mc.geom_for_sketch(gear)
track = g["track_m"]            # spanwise footprint (lateral skid track) [m]
L_eff = g["L_eff"]             # moment arm: hinge -> skid reaction [m]
SKID_LEN = g["skid_len"]       # skid length = sized rail length L_SKID_RAIL [m]
GROUND_CLEARANCE = g["clearance"]   # static fuselage belly-to-ground [m]
FUSE_WIDTH = g["fuse_width"]   # fuselage width (front view) [m]
m_struct = gear["m_gear"]      # whole-gear mass of the selected design [kg]
stroke = gear["dmax"]          # absorber stroke [m]
n_pk = gear["npk"]             # peak load factor [g]

x_skid = track / 2.0           # skid sits at half the track outboard of centreline
x_hinge = FUSE_WIDTH / 2.0     # hinge at the fuselage side
RUNNER_TOP = 0.04              # [m] skid-runner thickness
FUSE_H = 0.55                  # [m] drawn fuselage height (representative)

GROUND = "#6d4c41"
STEEL = "#37474f"
SKID = "#263238"
SPRING_C = "#c62828"
FUSE = "#90caf9"


def spring(ax, x0, y0, x1, y1, coils=5, width=0.035, **kw):
    """Zig-zag spring between (x0,y0) and (x1,y1)."""
    length = np.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    px, py = -uy, ux
    n = coils * 2
    xs, ys = [x0], [y0]
    for i in range(1, n):
        t = i / n
        side = width if i % 2 else -width
        xs.append(x0 + ux * length * t + px * side)
        ys.append(y0 + uy * length * t + py * side)
    xs.append(x1); ys.append(y1)
    ax.plot(xs, ys, **kw)


def ground(ax, x_lo, x_hi):
    ax.axhline(0, color=GROUND, lw=3)
    for xg in np.linspace(x_lo, x_hi, 26):
        ax.plot([xg, xg - 0.08], [0, -0.08], color=GROUND, lw=1)


fig, (axf, axs) = plt.subplots(1, 2, figsize=(14, 6.5))
belly = GROUND_CLEARANCE
ABS_FRAC = 0.32                # fraction of the leg (from the hinge) taken by the absorber

# ============================ FRONT VIEW ====================================
ax = axf
ax.set_title("Front view", fontsize=12, fontweight="bold")
ground(ax, -x_skid - 0.4, x_skid + 0.4)

# fuselage
ax.add_patch(FancyBboxPatch((-FUSE_WIDTH / 2, belly), FUSE_WIDTH, FUSE_H,
                            boxstyle="round,pad=0.01,rounding_size=0.06",
                            fc=FUSE, ec=STEEL, lw=1.8))
ax.text(0, belly + FUSE_H / 2, "fuselage\n(width 2.0 m)", ha="center", va="center", fontsize=9)

for sign in (-1, +1):
    xh, xs_ = sign * x_hinge, sign * x_skid
    # split the hinge->skid leg: elastomeric absorber near the hinge, rigid tube beyond
    xa = xh + (xs_ - xh) * ABS_FRAC
    ya = belly + (RUNNER_TOP - belly) * ABS_FRAC
    spring(ax, xh, belly, xa, ya, coils=5, width=0.03, color=SPRING_C, lw=2.2)
    ax.plot([xa, xs_], [ya, RUNNER_TOP], color=STEEL, lw=5, solid_capstyle="round")
    ax.plot([xh], [belly], marker="o", ms=9, mfc="white", mec=STEEL, mew=2, zorder=5)
    ax.add_patch(Rectangle((xs_ - 0.18, 0.0), 0.36, RUNNER_TOP, fc=SKID, ec="k", lw=1))

# callouts (kept on opposite sides so they don't collide)
ax.annotate("elastomeric\nshock absorber\n(at the hinge)",
            xy=(x_hinge + (x_skid - x_hinge) * ABS_FRAC / 2,
                belly + (RUNNER_TOP - belly) * ABS_FRAC / 2),
            xytext=(x_skid + 0.15, belly + 0.30), ha="left", va="center",
            fontsize=8, color=SPRING_C,
            arrowprops=dict(arrowstyle="->", color=SPRING_C))
ax.annotate("plastic hinge", xy=(-x_hinge, belly), xytext=(-x_hinge - 0.30, belly + 0.40),
            ha="center", fontsize=8, color=STEEL,
            arrowprops=dict(arrowstyle="->", color=STEEL))
ax.annotate("structural\ncross-tube",
            xy=(-(x_hinge + x_skid) / 2 - 0.05, (belly + RUNNER_TOP) / 2 - 0.02),
            xytext=(-x_hinge - 0.15, -0.30), ha="center", va="top",
            fontsize=8, color=STEEL, arrowprops=dict(arrowstyle="->", color=STEEL))

# L_eff (horizontal moment arm, hinge -> skid reaction) under the right leg
ax.annotate("", xy=(x_skid, -0.16), xytext=(x_hinge, -0.16),
            arrowprops=dict(arrowstyle="<->", color="0.2"))
ax.text((x_hinge + x_skid) / 2, -0.22,
        f"$L_{{eff}}$ = {L_eff:.2f} m\n(hinge → skid reaction)",
        ha="center", va="top", fontsize=8, color="0.2")

# ground clearance (far left)
ax.annotate("", xy=(-x_skid - 0.22, 0.0), xytext=(-x_skid - 0.22, belly),
            arrowprops=dict(arrowstyle="<->", color="0.25"))
ax.text(-x_skid - 0.27, belly / 2, f"clearance\n{GROUND_CLEARANCE*100:.0f} cm",
        ha="right", va="center", fontsize=8, color="0.25")

# track (top)
ax.annotate("", xy=(-x_skid, belly + FUSE_H + 0.12), xytext=(x_skid, belly + FUSE_H + 0.12),
            arrowprops=dict(arrowstyle="<->", color="0.3"))
ax.text(0, belly + FUSE_H + 0.16,
        f"track = {track:.2f} m  (footprint cap {FOOTPRINT_MAX_SPAN:.1f} m)",
        ha="center", fontsize=8, color="0.3")

ax.set_xlim(-x_skid - 1.0, x_skid + 1.0)
ax.set_ylim(-0.55, belly + FUSE_H + 0.35)
ax.set_aspect("equal")
ax.axis("off")

# ============================ SIDE VIEW =====================================
# Drawn to scale along the fuselage length L_FUS: the skid rail (SKID_LEN) is SHORTER than the
# fuselage and sits centred under the belly.
ax = axs
ax.set_title(f"Side view  (fuselage {L_FUS:.1f} m, skid rail {SKID_LEN:.2f} m)",
             fontsize=12, fontweight="bold")
ground(ax, -0.3, L_FUS + 0.6)

# fuselage spans the full length L_FUS
ax.add_patch(FancyBboxPatch((0.0, belly), L_FUS, FUSE_H,
                            boxstyle="round,pad=0.01,rounding_size=0.08",
                            fc=FUSE, ec=STEEL, lw=1.8))
ax.text(L_FUS / 2, belly + FUSE_H / 2, f"fuselage ({L_FUS:.1f} m)",
        ha="center", va="center", fontsize=9)

# skid runner: SKID_LEN long, centred under the fuselage, upturned nose at the front. The nose
# rise is capped well below the belly (GROUND_CLEARANCE) so it never pokes into the fuselage.
skid_x0 = (L_FUS - SKID_LEN) / 2.0
skid_x1 = skid_x0 + SKID_LEN
NOSE_LEN = 0.6                                   # [m] length of the upturned section
NOSE_RISE = 0.6 * (GROUND_CLEARANCE - RUNNER_TOP)  # tip stays at 60% of the clearance gap
xr = np.linspace(skid_x0, skid_x1, 80)
yr = np.full_like(xr, RUNNER_TOP)
nose = xr > skid_x1 - NOSE_LEN
yr[nose] = RUNNER_TOP + ((xr[nose] - (skid_x1 - NOSE_LEN)) / NOSE_LEN) ** 2 * NOSE_RISE
ax.plot(xr, yr, color=SKID, lw=5, solid_capstyle="round")

# fore + aft cross-tubes: rigid drop from the runner, elastomeric absorber at the hinge (top)
y_abs = belly - 0.14
for xc in (skid_x0 + SKID_LEN * 0.22, skid_x0 + SKID_LEN * 0.78):
    ax.plot([xc, xc], [RUNNER_TOP, y_abs], color=STEEL, lw=5, solid_capstyle="round")
    spring(ax, xc, y_abs, xc, belly, coils=4, width=0.04, color=SPRING_C, lw=2.2)
    ax.plot([xc], [belly], marker="o", ms=8, mfc="white", mec=STEEL, mew=2, zorder=5)

# skid-rail length dimension (under the runner)
ax.annotate("", xy=(skid_x0, -0.18), xytext=(skid_x1, -0.18),
            arrowprops=dict(arrowstyle="<->", color="0.3"))
ax.text((skid_x0 + skid_x1) / 2, -0.24, f"skid rail = {SKID_LEN:.2f} m",
        ha="center", va="top", fontsize=8, color="0.3")

ax.annotate("", xy=(skid_x0 - 0.25, 0.0), xytext=(skid_x0 - 0.25, belly),
            arrowprops=dict(arrowstyle="<->", color="0.25"))
ax.text(skid_x0 - 0.32, belly / 2, f"{GROUND_CLEARANCE*100:.0f} cm", ha="right", va="center",
        fontsize=8, color="0.25")
ax.text(L_FUS / 2, -0.44, "elastomeric shock absorbers at the hinges (fore + aft)",
        ha="center", va="top", fontsize=8, color=SPRING_C)
ax.annotate("skid runner", xy=(skid_x1 - 0.5, RUNNER_TOP),
            xytext=(skid_x1 + 0.2, RUNNER_TOP + 0.34), ha="left", va="bottom",
            fontsize=8, color=SKID, arrowprops=dict(arrowstyle="->", color=SKID))

ax.set_xlim(-0.6, L_FUS + 0.8)
ax.set_ylim(-0.6, belly + FUSE_H + 0.35)
ax.set_aspect("equal")
ax.axis("off")

# ============================ TITLE + DATA BOX ==============================
fig.suptitle("Skid Landing Gear — structural cross-tube + elastomeric absorbers",
             fontsize=14, fontweight="bold")
# Cross-member detail only for the bending architectures that have a leaf (b, t).
if "b" in params and "t" in params:
    member_line = (f"Cross-member: b={params['b']*1000:.0f} mm, "
                   f"t={params['t']*1000:.1f} mm     ")
else:
    member_line = ""
info = (f"Selected: {arch}   (from converged MTOW = {MTOW:.0f} kg)\n"
        f"Config: structural cross-body tube (plastic hinge at fuselage) + elastomeric "
        f"shock absorbers at the hinges; skids angled off the tube ends\n"
        f"Moment arm  L_eff : {L_eff:.2f} m     {member_line}"
        f"Gear mass: {m_struct:.0f} kg (skeleton + absorbers + rails)\n"
        f"Fuselage width: {FUSE_WIDTH:.1f} m     Skid length: {SKID_LEN:.2f} m     "
        f"Track: {track:.2f} m (cap {FOOTPRINT_MAX_SPAN:.1f} m)     "
        f"Ground clearance: {GROUND_CLEARANCE*100:.0f} cm     "
        f"Stroke δ: {stroke*100:.0f} cm     Peak: {n_pk:.1f} g")
fig.text(0.5, 0.02, info, ha="center", va="bottom", fontsize=8.5, family="monospace",
         bbox=dict(boxstyle="round,pad=0.5", fc="#fffde7", ec="0.6"))

fig.tight_layout(rect=[0, 0.12, 1, 0.95])
out = PROJECT_ROOT / "class_II_sizing" / "landing_gear_sketch.png"
fig.savefig(out, dpi=150)
print("written ->", out)
print(f"  MTOW={MTOW:.0f} kg, arch={arch}, L_eff={L_eff:.2f} m, track={track:.2f} m "
      f"(cap {FOOTPRINT_MAX_SPAN:.1f} m), skid={SKID_LEN:.2f} m, "
      f"gear mass={m_struct:.0f} kg, stroke={stroke*100:.0f} cm, n_pk={n_pk:.1f} g")
