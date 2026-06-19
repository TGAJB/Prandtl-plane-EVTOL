"""
mission_profile_plots.py

Four-panel mission-profile chart:
  (1) Altitude [m]   (2) Power [W]
  (3) Velocity [m/s] (4) Thrust [kN]
x-axis: mission time [s]

Run from the project root:  python mission_profile_plots.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from parameters import (
    G, V_CRUISE, V_I, V_AVG_TO, V_HOVER, VS_0,
    T_TAKEOFF, T_VERTICAL_CLIMB, T_CLIMB_ACC, T_CLIMB,
    T_CRUISE, T_DESCENT, T_LANDING,
    H_CRUISE, D_PROP, N_PROP, LD_CRUISE,
)
from class_II_sizing.energy import (
    takeoff_power, vertical_climb_power, climb_acceleration_power,
    climb_power, cruise_power, landing_power,
)
from class_II_sizing.mtow_sizing import load_final_design_state

# ── Converged MTOW ─────────────────────────────────────────────────────────────
mtow_kg = 1980
A_DISK = np.pi * (D_PROP / 2) ** 2   # single-rotor disk area [m^2]
VS_CLIMB = 3.0                         # vertical climb speed during climb phase [m/s]
CLIMB_ANGLE_DEG = 3.09097              # climb angle [deg]

# ── Segment names and durations ────────────────────────────────────────────────
SEG_NAMES = ["Takeoff", "V-Climb", "Transition", "Climb",
             "Cruise", "Descent", "Landing"]
SEG_DUR = [T_TAKEOFF, T_VERTICAL_CLIMB, T_CLIMB_ACC, T_CLIMB,
           T_CRUISE, T_DESCENT, T_LANDING]
N_SEG = len(SEG_NAMES)

t_bounds = np.concatenate([[0.0], np.cumsum(SEG_DUR)])  # (N_SEG+1,)

# ── Altitude at each time boundary [m] ────────────────────────────────────────
# Landing phase: VTOL hover-down from final approach altitude to ground.
# vs_avg in landing_power = VS_0/3, so altitude covered by that hover:
h_land_start = (VS_0 / 3.0) * T_LANDING   # ≈ 128 m

h_bounds = np.array([
    0.0,                                           # start
    V_AVG_TO * T_TAKEOFF,                          # end takeoff    (~10 m)
    V_AVG_TO * T_TAKEOFF + V_HOVER * T_VERTICAL_CLIMB,  # end V-climb (~85 m)
    V_AVG_TO * T_TAKEOFF + V_HOVER * T_VERTICAL_CLIMB + VS_CLIMB * T_CLIMB_ACC,  # end transition (~145 m)
    H_CRUISE,                                      # end climb      (3810 m)
    H_CRUISE,                                      # end cruise     (3810 m)
    h_land_start,                                  # end descent    (≈128 m)
    0.0,                                           # touchdown
])

# ── Power per segment [W] (constant within each segment) ──────────────────────
p_to  = takeoff_power(mtow_kg, A_DISK, vs_avg=V_AVG_TO, vs_f=V_HOVER, n_prop=N_PROP)
p_vc  = vertical_climb_power(mtow_kg, A_DISK, vs=V_HOVER, n_prop=N_PROP)
e_acc = climb_acceleration_power(mtow_kg, V_I, V_CRUISE, vs=VS_CLIMB, acc_time=T_CLIMB_ACC)
p_acc = e_acc / T_CLIMB_ACC          # energy → average power
p_cl  = climb_power(mtow_kg, CLIMB_ANGLE_DEG, V_CRUISE, VS_CLIMB)
p_c   = cruise_power(mtow_kg)
p_d   = 0.0                          # aerodynamic glide descent, no shaft power
p_l   = landing_power(mtow_kg, A_DISK, vs_0=VS_0, n_prop=N_PROP)

p_segs = [p_to, p_vc, p_acc, p_cl, p_c, p_d, p_l]

# ── Thrust per segment [kN] (constant within each segment) ────────────────────
theta = np.radians(CLIMB_ANGLE_DEG)
T_hover_N  = mtow_kg * (G + V_HOVER / T_TAKEOFF)
T_climb_N  = mtow_kg * G * (np.cos(theta) / LD_CRUISE + np.sin(theta))
T_cruise_N = mtow_kg * G / LD_CRUISE
T_land_N   = mtow_kg * (G + VS_0 / T_LANDING)
T_trans_N  = mtow_kg*G*1.4   # linear average during transition

T_segs_kN = [
    T_hover_N / 1e3,   # takeoff
    T_hover_N / 1e3,   # V-climb
    T_trans_N / 1e3,   # transition
    T_climb_N / 1e3,   # climb
    T_cruise_N / 1e3,  # cruise
    0.0,               # descent (glide)
    T_land_N / 1e3,    # landing
]

# ── Horizontal (forward) airspeed at each boundary [m/s] ──────────────────────
v_bounds = np.array([
    0.0,       # at lift-off
    0.0,       # end of takeoff   (still purely vertical)
    V_I,       # end of V-climb   (reached transition speed)
    V_I,       # start of climb-acceleration
    V_CRUISE,  # end of climb-acceleration (and maintained through climb)
    V_CRUISE,  # end of cruise
    V_CRUISE,  # end of descent   (aerodynamic)
    0.0,       # touchdown        (hover landing)
])

# ── Helper: staircase arrays for step-function plots ──────────────────────────
def to_staircase(t_bounds, seg_vals):
    """Return (t, v) arrays that draw a horizontal step per segment."""
    t = np.empty(2 * N_SEG)
    v = np.empty(2 * N_SEG)
    for i in range(N_SEG):
        t[2*i]   = t_bounds[i]
        t[2*i+1] = t_bounds[i+1]
        v[2*i]   = seg_vals[i]
        v[2*i+1] = seg_vals[i]
    return t, v

t_p, p_plot   = to_staircase(t_bounds, p_segs)
t_T, T_plot   = to_staircase(t_bounds, T_segs_kN)

# ── Segment colour palette ─────────────────────────────────────────────────────
SEG_COLORS = [
    "#fcdddd",  # Takeoff     (red-tint)
    "#fce8dd",  # V-Climb     (orange-tint)
    "#fef9cc",  # Transition  (yellow-tint)
    "#d4f0d4",  # Climb       (green-tint)
    "#d0e8fc",  # Cruise      (blue-tint)
    "#e8d8fc",  # Descent     (purple-tint)
    "#fcd8ee",  # Landing     (pink-tint)
]

def shade_segments(ax):
    """Background tint + dashed dividers on an axes object."""
    for i in range(N_SEG):
        ax.axvspan(t_bounds[i], t_bounds[i+1],
                   color=SEG_COLORS[i], alpha=1.0, linewidth=0, zorder=0)
    for tb in t_bounds[1:-1]:
        ax.axvline(tb, color="0.55", lw=0.7, ls="--", zorder=1)

# ── Figure layout ──────────────────────────────────────────────────────────────
LINE_COLOR = "#1b3d87"
FILL_ALPHA = 0.25

fig, axes = plt.subplots(4, 1, figsize=(13, 14), sharex=True)
fig.suptitle(
    f"Mission Profile  (MTOW = {mtow_kg:.0f} kg)",
    fontsize=14, fontweight="bold", y=0.995,
)

# ── Panel 1: Altitude ──────────────────────────────────────────────────────────
ax = axes[0]
shade_segments(ax)
ax.plot(t_bounds, h_bounds, color=LINE_COLOR, lw=2, zorder=3)
ax.fill_between(t_bounds, 0, h_bounds, color=LINE_COLOR, alpha=FILL_ALPHA, zorder=2)
ax.set_ylabel("Altitude [m]", fontsize=11)
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(axis="y", alpha=0.35, ls=":", zorder=1)

# Segment labels in the top panel
# Merge Takeoff / V-Climb / Transition under a single "Take-off" label
ax.text(0.5 * (t_bounds[0] + t_bounds[3]), H_CRUISE * 1.04, "Take-off",
        ha="center", va="bottom", fontsize=8.5, color="0.30", clip_on=False)
for i, name in enumerate(SEG_NAMES[3:], start=3):
    t_mid = 0.5 * (t_bounds[i] + t_bounds[i+1])
    ax.text(t_mid, H_CRUISE * 1.04, name, ha="center", va="bottom",
            fontsize=8.5, color="0.30", rotation=0 if SEG_DUR[i] > 200 else 90,
            clip_on=False)

# ── Panel 2: Power ────────────────────────────────────────────────────────────
ax = axes[1]
shade_segments(ax)
ax.plot(t_p, p_plot, color=LINE_COLOR, lw=2, zorder=3)
ax.fill_between(t_p, 0, p_plot, color=LINE_COLOR, alpha=FILL_ALPHA, zorder=2)
ax.set_ylabel("Power [W]", fontsize=11)
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(axis="y", alpha=0.35, ls=":", zorder=1)

# ── Panel 3: Velocity ─────────────────────────────────────────────────────────
ax = axes[2]
shade_segments(ax)
ax.plot(t_bounds, v_bounds, color=LINE_COLOR, lw=2, zorder=3)
ax.fill_between(t_bounds, 0, v_bounds, color=LINE_COLOR, alpha=FILL_ALPHA, zorder=2)
ax.set_ylabel("Velocity [m/s]", fontsize=11)
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.35, ls=":", zorder=1)

# ── Panel 4: Thrust ───────────────────────────────────────────────────────────
ax = axes[3]
shade_segments(ax)
ax.plot(t_T, T_plot, color=LINE_COLOR, lw=2, zorder=3)
ax.fill_between(t_T, 0, T_plot, color=LINE_COLOR, alpha=FILL_ALPHA, zorder=2)
ax.set_ylabel("Thrust [kN]", fontsize=11)
ax.set_xlabel("Mission time [s]", fontsize=11)
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.35, ls=":", zorder=1)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=SEG_COLORS[i], edgecolor="0.55",
                   linewidth=0.5, label=SEG_NAMES[i])
    for i in range(N_SEG)
]
axes[0].legend(handles=legend_patches, loc="upper left",
               fontsize=8, ncol=N_SEG, framealpha=0.8,
               handlelength=1.2, handletextpad=0.5, columnspacing=0.8)

# ── x-axis limits and tick spacing ────────────────────────────────────────────
axes[3].set_xlim(t_bounds[0], t_bounds[-1])

fig.tight_layout(rect=[0, 0, 1, 0.993])

out_path = PROJECT_ROOT / "mission_profile.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
