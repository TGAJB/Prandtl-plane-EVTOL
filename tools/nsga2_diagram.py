"""
nsga2_diagram.py
================

Standalone VISUAL explainer of how the NSGA-II optimiser
(final_characteristics/optimiser.py, pymoo) functions. Renders a single figure with
three panels and writes ``docs/architecture/nsga2_algorithm.{png,svg}``:

  * top    - the generational loop (selection -> variation -> evaluate -> sort -> survive,
             repeated each generation);
  * lower-left - the SURVIVAL-SELECTION schematic: the merged population R = P + Q is
             sorted into non-dominated fronts F1, F2, F3; the next generation is filled
             front-by-front and the splitting front is trimmed by crowding distance;
  * lower-right - OBJECTIVE space: dominated designs vs the non-dominated Pareto front
             (MTOW vs worst stability margin, both minimised).

Picture-first by design (a box-and-arrow flowchart is too text-heavy for an algorithm).
Config shown matches ``run_phase_2``: pop 24, SBX(0.9, eta=15), polynomial mutation(eta=20),
2 objectives, 18 constraints, convergence termination.

Run:  python tools/nsga2_diagram.py        (needs only matplotlib + numpy)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "architecture"

INK = "#263238"
NEUTRAL = "#90a4ae"
F1C, F2C, F3C = "#2e7d32", "#f9a825", "#ef6c00"   # front colours (best -> worst)
REJECT = "#b0bec5"
ACCENT = "#1f6feb"
GREENPILL = "#e8f5e9"


# ---------------------------------------------------------------------------
# small drawing helpers (work in 0..1 axes coordinates)
# ---------------------------------------------------------------------------
def chip(ax, cx, cy, w, h, text, fc="#ffffff", ec="#455a64", fs=10, weight="normal", tc=INK):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.006,rounding_size=0.03",
        linewidth=1.5, edgecolor=ec, facecolor=fc, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc, weight=weight, zorder=5)


def arrow(ax, p1, p2, color="#607d8b", lw=2.2, rad=0.0, ms=16):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=4))


def dot_grid(ax, x0, y0, n, cols, color, dx=0.045, dy=0.07, size=150, faded=False, crossed=False):
    xs, ys = [], []
    for i in range(n):
        xs.append(x0 + (i % cols) * dx)
        ys.append(y0 - (i // cols) * dy)
    ax.scatter(xs, ys, s=size, c=color, edgecolors="#37474f", linewidths=0.8,
               alpha=0.3 if faded else 1.0, zorder=5)
    if crossed:
        ax.scatter(xs, ys, s=size * 0.9, marker="x", c="#b71c1c", linewidths=1.4, zorder=6)
    return list(zip(xs, ys))


# ---------------------------------------------------------------------------
# Panel 1 - the generational loop
# ---------------------------------------------------------------------------
def draw_loop(ax):
    ax.text(0.5, 0.96, "Generational loop", ha="center", va="top",
            fontsize=13, weight="bold", color=INK)

    steps = ["Tournament\nselection", "Crossover\n+ mutation", "Evaluate\noffspring",
             "Combine +\nnon-dominated sort", "Crowding +\nelitist survival"]
    fills = ["#eef3fb", "#eef3fb", "#fff8e1", "#e3f2fd", "#e8f5e9"]
    xs = np.linspace(0.13, 0.87, len(steps))
    y, w, h = 0.62, 0.155, 0.30

    for x, s, fc in zip(xs, steps, fills):
        chip(ax, x, y, w, h, s, fc=fc, fs=9.5)
    for x1, x2 in zip(xs[:-1], xs[1:]):
        arrow(ax, (x1 + w / 2, y), (x2 - w / 2, y))

    # entry + exit annotations
    ax.text(0.13, 0.94, "start: initialise 24 random\ndesigns, then evaluate",
            ha="center", va="top", fontsize=8.5, style="italic", color="#546e7a")
    arrow(ax, (0.13, 0.82), (0.13, y + h / 2), color="#546e7a", lw=1.6)
    chip(ax, 0.87, 0.96, 0.21, 0.12, "converged  ->  Pareto front", fc=GREENPILL, ec=F1C, fs=9, weight="bold", tc=F1C)
    arrow(ax, (0.87, y + h / 2), (0.87, 0.90), color=F1C, lw=1.8)

    # the big "repeat" feedback arrow looping BELOW the row
    ax.add_patch(FancyArrowPatch(
        (xs[-1], y - h / 2 - 0.02), (xs[0], y - h / 2 - 0.02),
        arrowstyle="-|>", mutation_scale=18, lw=2.4, color=ACCENT,
        connectionstyle="arc3,rad=-0.34", zorder=4))
    ax.text(0.5, 0.07, "repeat every generation  until the Pareto front stops moving (or n_max_gen)",
            ha="center", va="center", fontsize=9.5, color=ACCENT, weight="bold")


# ---------------------------------------------------------------------------
# Panel 2 - survival selection (the iconic NSGA-II schematic)
# ---------------------------------------------------------------------------
def draw_survival(ax):
    ax.text(0.5, 0.99, "Survival selection — who advances to the next generation",
            ha="center", va="top", fontsize=12, weight="bold", color=INK)

    # --- merged population R = P + Q (2N) ---
    ax.add_patch(FancyBboxPatch((0.02, 0.30), 0.20, 0.45, boxstyle="round,pad=0.01,rounding_size=0.02",
                                facecolor="#eceff1", edgecolor="#90a4ae", lw=1.4, zorder=2))
    ax.text(0.12, 0.80, r"$R = P + Q$", ha="center", fontsize=11, color=INK, weight="bold")
    ax.text(0.12, 0.745, "merged (2N = 12)", ha="center", fontsize=8.5, color="#546e7a")
    dot_grid(ax, 0.07, 0.64, 12, 3, NEUTRAL, dx=0.045, dy=0.075, size=130)

    arrow(ax, (0.225, 0.52), (0.33, 0.52), lw=2.4)
    ax.text(0.277, 0.57, "non-dominated\nsort", ha="center", fontsize=8.5, color="#546e7a")

    # --- fronts F1, F2, F3 ---
    fronts = [("$F_1$  (best)", F1C, 0.66, 4), ("$F_2$  splitting front", F2C, 0.50, 4), ("$F_3$", F3C, 0.34, 4)]
    for label, color, yc, n in fronts:
        ax.add_patch(FancyBboxPatch((0.35, yc - 0.065), 0.24, 0.13, boxstyle="round,pad=0.005,rounding_size=0.02",
                                    facecolor=color, edgecolor="#37474f", lw=1.2, alpha=0.22, zorder=2))
        ax.text(0.365, yc + 0.085, label, ha="left", fontsize=9.5, color=color, weight="bold")
        dot_grid(ax, 0.40, yc, n, 4, color, dx=0.045, dy=0.06, size=130)

    # the "cut at N" line: F1 fully in, F2 is split
    ax.plot([0.34, 0.60], [0.50, 0.50], ls=(0, (5, 3)), color="#b71c1c", lw=1.6, zorder=6)
    ax.text(0.605, 0.50, "cut at N = 6", ha="left", va="center", fontsize=8.5, color="#b71c1c", weight="bold")

    arrow(ax, (0.62, 0.52), (0.72, 0.52), lw=2.4)
    ax.text(0.67, 0.585, "fill by front,\nsplit by crowding", ha="center", fontsize=8.5, color="#546e7a")

    # --- next generation P_{t+1} ---
    ax.add_patch(FancyBboxPatch((0.74, 0.42), 0.22, 0.34, boxstyle="round,pad=0.01,rounding_size=0.02",
                                facecolor=GREENPILL, edgecolor=F1C, lw=1.6, zorder=2))
    ax.text(0.85, 0.80, r"$P_{t+1}$  (N = 6)", ha="center", fontsize=11, color=F1C, weight="bold")
    dot_grid(ax, 0.79, 0.66, 4, 4, F1C, dx=0.045, dy=0.06, size=130)        # all of F1
    dot_grid(ax, 0.79, 0.50, 2, 4, F2C, dx=0.045, dy=0.06, size=130)        # 2 of F2 by crowding

    # rejected
    ax.text(0.85, 0.345, "rejected", ha="center", fontsize=8.5, color="#90a4ae", style="italic")
    dot_grid(ax, 0.79, 0.30, 2, 4, F2C, dx=0.045, dy=0.06, size=120, faded=True, crossed=True)
    dot_grid(ax, 0.79 + 0.09, 0.30, 4, 4, F3C, dx=0.045, dy=0.06, size=120, faded=True, crossed=True)

    ax.text(0.5, 0.07, "Crowding distance keeps the most spread-out members of the splitting front "
                       "(diversity), so the Pareto front stays well-sampled.",
            ha="center", fontsize=9, color="#37474f")


# ---------------------------------------------------------------------------
# Panel 3 - objective space (the Pareto front)
# ---------------------------------------------------------------------------
def draw_objective_space(ax):
    ax.text(0.5, 0.99, "Objective space — the trade-off", ha="center", va="top",
            fontsize=12, weight="bold", color=INK, zorder=8)

    # axes arrows
    arrow(ax, (0.12, 0.12), (0.95, 0.12), color=INK, lw=1.6, ms=14)
    arrow(ax, (0.12, 0.12), (0.12, 0.92), color=INK, lw=1.6, ms=14)
    ax.text(0.55, 0.045, "MTOW  " + r"$\rightarrow$" + "  (minimise)", ha="center", fontsize=9.5, color=INK)
    ax.text(0.045, 0.52, "worst stability margin  " + r"$\rightarrow$" + "  (minimise)",
            ha="center", va="center", rotation=90, fontsize=9.5, color=INK)

    rng = np.random.default_rng(3)
    # dominated cloud (worse: up and to the right)
    dx = rng.uniform(0.32, 0.9, 26)
    dy = rng.uniform(0.30, 0.85, 26)
    ax.scatter(dx, dy, s=42, c=REJECT, edgecolors="#90a4ae", linewidths=0.6, zorder=3, label="dominated designs")

    # non-dominated Pareto front (lower-left boundary)
    px = np.linspace(0.18, 0.70, 7)
    py = 0.18 + 0.62 * np.exp(-3.2 * (px - 0.18))
    ax.plot(px, py, color=ACCENT, lw=1.8, zorder=4)
    ax.scatter(px, py, s=95, c=np.linspace(0, 1, len(px)), cmap="viridis",
               edgecolors="#1a237e", linewidths=1.0, zorder=5, label="Pareto front (non-dominated)")

    ax.annotate("non-dominated:\nno design is better\nin BOTH objectives",
                xy=(px[3], py[3]), xytext=(0.52, 0.62), fontsize=8.5, color="#1a237e",
                ha="left", arrowprops=dict(arrowstyle="->", color="#1a237e", lw=1.2))
    # "better" direction
    arrow(ax, (0.42, 0.42), (0.24, 0.26), color=F1C, lw=1.8, ms=15)
    ax.text(0.43, 0.45, "better", fontsize=9, color=F1C, weight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.90), fontsize=8.0,
              frameon=True, framealpha=0.92, borderpad=0.5)


# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.35], width_ratios=[1.3, 1.0],
                          left=0.02, right=0.98, top=0.91, bottom=0.07, hspace=0.16, wspace=0.10)
    ax_top = fig.add_subplot(gs[0, :])
    ax_surv = fig.add_subplot(gs[1, 0])
    ax_obj = fig.add_subplot(gs[1, 1])
    for ax in (ax_top, ax_surv, ax_obj):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    fig.suptitle("NSGA-II — how the optimiser finds the MTOW vs stability-margin trade-off",
                 fontsize=16, weight="bold", color=INK, y=0.975)
    fig.text(0.5, 0.025,
             "final_characteristics/optimiser.py (pymoo)   ·   population 24   ·   SBX crossover "
             "(p=0.9, η=15)   ·   polynomial mutation (η=20)   ·   2 objectives   ·   "
             "18 constraints   ·   convergence-based termination",
             ha="center", fontsize=9, color="#546e7a")

    draw_loop(ax_top)
    draw_survival(ax_surv)
    draw_objective_space(ax_obj)

    for ext in ("png", "svg"):
        path = OUT_DIR / f"nsga2_algorithm.{ext}"
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"NSGA-II visual -> {OUT_DIR / 'nsga2_algorithm.png'} (+ .svg)")


if __name__ == "__main__":
    main()
