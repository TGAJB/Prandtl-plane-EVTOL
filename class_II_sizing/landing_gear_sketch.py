"""
landing_gear_sketch.py
Schematic (front + side view) of the skid landing-gear design.

Configuration (per the design intent):
  * a structural cross-body TUBE bridges the fuselage, with a PLASTIC HINGE at each
    fuselage attachment;
  * the ELASTOMERIC SHOCK ABSORBERS sit at those hinges;
  * the SKIDS come off the ends of the cross-tube at an angle and run the fuselage length.

L_eff is the moment arm from the plastic hinge to the skid reaction; it is the sized
cantilever length of the metal cross-member spring (architecture B), pulled live from the
model. Vertical dimensions are to scale (ground clearance, stroke); the skid length is the
fuselage length L_FUS.

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
from parameters import L_FUS

# --- design point -----------------------------------------------------------
MTOW = 2198.58
FUSE_WIDTH = 2.0               # [m] fuselage width (front view)
SKID_LEN = L_FUS               # [m] skid length = fuselage length
GROUND_CLEARANCE = 0.30        # [m] static fuselage belly-to-ground

# Structural cross-member spring (architecture B) gives the plastic hinge + moment arm.
gear = mc.landing_gear_mass(MTOW)
beam = next(r for r in gear["all_architectures"] if r["name"] == "B metal leaf")
L_eff = beam["params"]["L"]    # moment arm: plastic hinge -> skid reaction [m]
b_beam = beam["params"]["b"]   # cross-member width [m]
t_beam = beam["params"]["t"]   # cross-member thickness [m]
m_struct = beam["mass"]        # structural cross-member mass [kg]
stroke = gear["dmax"]          # elastomeric absorber stroke [m]
n_pk = gear["npk"]             # peak load factor [g]

x_hinge = FUSE_WIDTH / 2.0     # hinge at the fuselage side
x_skid = x_hinge + L_eff       # skid reaction, one moment arm outboard
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
ax.text(0, belly + FUSE_H + 0.16, f"track = {2*x_skid:.2f} m", ha="center", fontsize=8,
        color="0.3")

ax.set_xlim(-x_skid - 1.0, x_skid + 1.0)
ax.set_ylim(-0.55, belly + FUSE_H + 0.35)
ax.set_aspect("equal")
ax.axis("off")

# ============================ SIDE VIEW =====================================
ax = axs
ax.set_title(f"Side view  (skid length = fuselage length = {SKID_LEN:.0f} m)",
             fontsize=12, fontweight="bold")
ground(ax, -0.3, SKID_LEN + 0.6)

# skid runner (length = fuselage length) with upturned nose
xr = np.linspace(0.0, SKID_LEN, 80)
yr = np.full_like(xr, RUNNER_TOP)
nose = xr > SKID_LEN - 0.7
yr[nose] = RUNNER_TOP + (xr[nose] - (SKID_LEN - 0.7)) ** 2 * 0.7
ax.plot(xr, yr, color=SKID, lw=5, solid_capstyle="round")

# fuselage
fx0, fx1 = 0.7, SKID_LEN - 0.7
ax.add_patch(FancyBboxPatch((fx0, belly), fx1 - fx0, FUSE_H,
                            boxstyle="round,pad=0.01,rounding_size=0.08",
                            fc=FUSE, ec=STEEL, lw=1.8))
ax.text((fx0 + fx1) / 2, belly + FUSE_H / 2, "fuselage", ha="center", va="center", fontsize=9)

# fore + aft cross-tubes: rigid drop from the runner, elastomeric absorber at the hinge (top)
y_abs = belly - 0.14
for xc in (SKID_LEN * 0.30, SKID_LEN * 0.70):
    ax.plot([xc, xc], [RUNNER_TOP, y_abs], color=STEEL, lw=5, solid_capstyle="round")
    spring(ax, xc, y_abs, xc, belly, coils=4, width=0.04, color=SPRING_C, lw=2.2)
    ax.plot([xc], [belly], marker="o", ms=8, mfc="white", mec=STEEL, mew=2, zorder=5)

ax.annotate("", xy=(0.35, 0.0), xytext=(0.35, belly),
            arrowprops=dict(arrowstyle="<->", color="0.25"))
ax.text(0.28, belly / 2, f"{GROUND_CLEARANCE*100:.0f} cm", ha="right", va="center",
        fontsize=8, color="0.25")
ax.text(SKID_LEN * 0.5, -0.20, "elastomeric shock absorbers at the hinges (fore + aft)",
        ha="center", va="top", fontsize=8, color=SPRING_C)
ax.annotate("skid runner", xy=(SKID_LEN - 0.6, RUNNER_TOP),
            xytext=(SKID_LEN - 0.1, RUNNER_TOP + 0.34), ha="right", va="bottom",
            fontsize=8, color=SKID, arrowprops=dict(arrowstyle="->", color=SKID))

ax.set_xlim(-0.6, SKID_LEN + 0.8)
ax.set_ylim(-0.55, belly + FUSE_H + 0.35)
ax.set_aspect("equal")
ax.axis("off")

# ============================ TITLE + DATA BOX ==============================
fig.suptitle("Skid Landing Gear — structural cross-tube + elastomeric absorbers",
             fontsize=14, fontweight="bold")
info = (f"Config: structural cross-body tube (plastic hinge at fuselage) + elastomeric "
        f"shock absorbers at the hinges; skids angled off the tube ends\n"
        f"Moment arm  L_eff : {L_eff:.2f} m     Cross-member: b={b_beam*1000:.0f} mm, "
        f"t={t_beam*1000:.1f} mm     Structural mass: {m_struct:.0f} kg (+ absorbers)\n"
        f"Fuselage width: {FUSE_WIDTH:.1f} m     Skid length: {SKID_LEN:.1f} m     "
        f"Track: {2*x_skid:.2f} m     Ground clearance: {GROUND_CLEARANCE*100:.0f} cm     "
        f"Stroke δ: {stroke*100:.0f} cm     Peak: {n_pk:.1f} g")
fig.text(0.5, 0.02, info, ha="center", va="bottom", fontsize=8.5, family="monospace",
         bbox=dict(boxstyle="round,pad=0.5", fc="#fffde7", ec="0.6"))

fig.tight_layout(rect=[0, 0.12, 1, 0.95])
out = PROJECT_ROOT / "class_II_sizing" / "landing_gear_sketch.png"
fig.savefig(out, dpi=150)
print("written ->", out)
print(f"  L_eff={L_eff:.2f} m, track={2*x_skid:.2f} m, skid={SKID_LEN:.1f} m, "
      f"struct mass={m_struct:.0f} kg, stroke={stroke*100:.0f} cm, n_pk={n_pk:.1f} g")
