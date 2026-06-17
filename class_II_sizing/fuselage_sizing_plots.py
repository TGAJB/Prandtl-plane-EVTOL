"""
fuselage_sizing_plots.py
========================
Standalone diagnostic plots for the physics-based fuselage mass method
(``mass_components.fuselage_mass`` with ``method="physics"``).

Run directly::

    python class_II_sizing/fuselage_sizing_plots.py

Every figure is driven by the SAME sizing code the mass build-up uses
(``fuselage_mass(..., return_details=True)`` and the ``_fus_*`` helpers), never a
re-derivation, so the plots always match the converged mass. Figures are written to
``class_II_sizing/fuselage_sizing_figures/``.

How the fuselage is sized (what these plots visualise)
-----------------------------------------------------
The fuselage is modelled as a thin-walled circular shell-beam (radius ``D_FUS/2``,
length ``L_FUS``). For each design condition every coexisting load is placed on one
free-free beam at once -- the two wing reactions, the tail load, and the distributed
inertia -- and integrated to a single SUPERIMPOSED bending-moment diagram ``M(x)``.
The skin gauge is then sized PER STATION against the worst-condition moment envelope,
using the failure mode that actually governs a thin shell in bending: **buckling**
(``sigma_cr = C E t/R``), floored at a min manufacturing/handling gauge. Both CFRP and
aluminium are sized and the lighter is kept.

The four figures
----------------
1. ``fuselage_load_shear_moment.png`` -- the headline figure: per design condition, the
   load / shear / moment diagrams built from ALL loads acting together. Shows how the
   wing reactions, tail load and inertia STACK along the fuselage and which condition
   governs.
2. ``fuselage_skin_thickness.png`` -- why the gauge is what it is: the moment envelope
   and the three candidate skin gauges (min-gauge, yield, buckling) vs station, with the
   selected ``t_skin(x)`` envelope shaded. Makes the governing failure mode obvious.
3. ``fuselage_material_comparison.png`` -- the material trade: CFRP vs aluminium total
   mass split into skin and primary-fraction recovery, with the lighter winner marked.
4. ``fuselage_mass_vs_mtow.png`` -- the design-loop behaviour: fuselage mass over an MTOW
   sweep (physics vs legacy regression), showing the buckling-driven MTOW dependence.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                     # headless: save PNGs, no display needed
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from parameters import (
    D_FUS, L_FUS, FUS_SHELL_BUCKLING_C, FUS_PRIMARY_FRACTION,
    X_WING_F, X_WING_R, X_TAIL, X_GEAR, X_PAYLOAD,
)
from class_II_sizing.mass_components import fuselage_mass


def _design_mtow(fallback=2000.0):
    """Live converged MTOW [kg] of the active design (honours the chosen_design.json
    override) so the figures are drawn at the real design point, not a fixed stand-in.
    Falls back to ``fallback`` if the converged state cannot be loaded."""
    try:
        from class_II_sizing.mtow_sizing import load_final_design_state
        return float(load_final_design_state()["mtow"])
    except Exception:
        return fallback


OUT_DIR = PROJECT_ROOT / "class_II_sizing" / "fuselage_sizing_figures"
MTOW_REF = _design_mtow()   # [kg] live design MTOW for the per-condition diagrams (override-aware; 2000 kg fallback)

# Key longitudinal stations to mark on every beam axis (station, label, colour).
_STATIONS = [
    (X_WING_F, "front wing", "tab:blue"),
    (X_WING_R, "rear wing", "tab:blue"),
    (X_TAIL, "tail", "tab:red"),
    (X_GEAR, "gear", "tab:green"),
    (X_PAYLOAD, "payload", "tab:purple"),
]


def _mark_stations(ax, label=False):
    """Dotted vertical guides at the wing/tail/gear/payload stations."""
    for x_st, name, col in _STATIONS:
        ax.axvline(x_st, color=col, ls=":", lw=0.8, alpha=0.6)
        if label:
            ax.text(x_st, 1.01, name, color=col, fontsize=7, rotation=90,
                    va="bottom", ha="center", transform=ax.get_xaxis_transform())


def _caption(fig, text):
    """Italic explanatory caption along the bottom of a figure (how to read it)."""
    fig.text(0.5, 0.012, text, ha="center", va="bottom", fontsize=8.5,
             style="italic", wrap=True, color="#333333")


def plot_load_shear_moment(details, out_dir):
    """Figure 1 -- superimposed load/shear/moment diagrams for every design condition.

    Read it column by column (one condition each). Top row: the applied loading -- the
    green distributed inertia (note the payload-box step), red arrows for external point
    loads (tail), blue arrows for the solved support reactions (wings or skids). Middle
    row: the resulting shear V(x). Bottom row: the bending moment M(x) the skin must
    carry; the black dot is the peak. Because every load in a condition acts at once, the
    moment is their SUPERPOSITION. The condition with the largest peak (red title,
    '[GOVERNS]') sets the design moment. M(x) returns to ~0 at both free ends, the check
    that the free-free beam is in equilibrium."""
    conds = details["conditions"]
    governing = details["governing_condition"]
    n = len(conds)
    fig, axes = plt.subplots(3, n, figsize=(5.2 * n, 8.7), sharex=True)

    for j, c in enumerate(conds):
        x, w, V, M = c["x"], c["w"], c["V"], c["M"]
        is_gov = c["name"] == governing
        ax_w, ax_v, ax_m = axes[0, j], axes[1, j], axes[2, j]

        ax_w.plot(x, -w / 1e3, color="green", lw=1.8)
        ax_w.fill_between(x, -w / 1e3, 0, color="green", alpha=0.15)
        for xi, fi in c["applied"]:                       # external point loads (red)
            ax_w.annotate("", xy=(xi, 0.0), xytext=(xi, -np.sign(fi) * 4),
                          arrowprops=dict(arrowstyle="->", color="red", lw=1.6))
        for xi, ri in c["supports"]:                      # solved reactions (blue)
            ax_w.annotate("", xy=(xi, 0.0), xytext=(xi, -np.sign(ri) * 4),
                          arrowprops=dict(arrowstyle="->", color="blue", lw=1.6))
        tag = "  [GOVERNS]" if is_gov else ""
        ax_w.set_title(f"{c['name']}{tag}", color="firebrick" if is_gov else "black",
                       fontweight="bold" if is_gov else "normal", fontsize=10)

        ax_v.plot(x, V / 1e3, color="tab:blue", lw=1.8)
        ax_v.fill_between(x, V / 1e3, 0, color="tab:blue", alpha=0.15)

        ax_m.plot(x, M / 1e3, color="tab:red", lw=1.8)
        ax_m.fill_between(x, M / 1e3, 0, color="tab:red", alpha=0.15)
        ipk = int(np.argmax(np.abs(M)))
        ax_m.plot(x[ipk], M[ipk] / 1e3, "ko", ms=5)
        ax_m.annotate(f"peak |M| = {c['peak']/1e3:.1f} kN.m",
                      xy=(x[ipk], M[ipk] / 1e3), xytext=(0.04, 0.08),
                      textcoords="axes fraction", fontsize=8,
                      arrowprops=dict(arrowstyle="->", lw=0.8))

        for ax in (ax_w, ax_v, ax_m):
            ax.axhline(0, color="k", lw=0.8)
            ax.grid(True, ls="--", alpha=0.4)
            _mark_stations(ax)
        ax_m.set_xlabel("x from nose [m]")
        if j == 0:
            ax_w.set_ylabel("applied load\n$w(x)$ [kN/m]")
            ax_v.set_ylabel("shear\n$V(x)$ [kN]")
            ax_m.set_ylabel("bending moment\n$M(x)$ [kN.m]")

    fig.suptitle(f"Fuselage superimposed beam diagrams  (MTOW = {MTOW_REF:.0f} kg)",
                 fontsize=13, fontweight="bold")
    _caption(fig,
             "Each column is one design condition with ALL its loads applied at once "
             "(green = distributed inertia, red arrows = tail load, blue arrows = solved "
             "wing/skid reactions). Rows: load -> shear -> moment. The largest-peak "
             "column (red, [GOVERNS]) sizes the skin; M(x) -> 0 at both ends confirms "
             "equilibrium.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    path = out_dir / "fuselage_load_shear_moment.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_skin_thickness(details, out_dir):
    """Figure 2 -- where the skin gauge comes from (the governing failure mode).

    Left axis: the three CANDIDATE skin gauges for the winning material at each station --
    the flat min-gauge floor, the (tiny) material-yield gauge, and the buckling gauge
    ``t_buckle = sqrt(M/(pi R C E))``. The actually-used skin ``t_skin(x)`` is the upper
    envelope of the three (shaded). Right axis (grey): the ultimate moment envelope
    ``M_env(x)`` that drives the buckling gauge -- thickness peaks where the moment peaks.
    The takeaway: a thin monocoque shell is governed by BUCKLING, not yield, which is why
    the gauge (and the mass) grow with load/MTOW."""
    winner = details["material"]
    p = details["by_material"][winner]
    x, m_env = p["x"], p["m_env"]
    t_skin, t_yield, t_buckle = p["t_skin"], p["t_yield"], p["t_buckle"]
    t_floor = p["t_min_gauge"]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axhline(t_floor * 1e3, color="tab:blue", ls="--", lw=1.5,
               label=f"min gauge = {t_floor*1e3:.1f} mm (manufacturing/handling floor)")
    ax.plot(x, t_yield * 1e3, color="tab:green", lw=1.4, label="yield gauge $M/(\\sigma\\,\\pi R^2)$")
    ax.plot(x, t_buckle * 1e3, color="tab:red", lw=1.8, label="buckling gauge $\\sqrt{M/(\\pi R C E)}$")
    ax.plot(x, t_skin * 1e3, color="black", lw=2.4, label="selected $t_{skin}(x)$ = upper envelope")
    ax.fill_between(x, t_skin * 1e3, 0, color="black", alpha=0.06)

    axm = ax.twinx()
    axm.plot(x, m_env / 1e3, color="grey", ls="-.", lw=1.2, alpha=0.8)
    axm.set_ylabel("ultimate moment envelope $M_{env}(x)$ [kN.m]", color="grey")
    axm.tick_params(axis="y", colors="grey")
    axm.set_ylim(bottom=0)

    ax.set_xlabel("x from nose [m]")
    ax.set_ylabel(f"skin gauge [mm]  ({winner})")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Fuselage skin gauge sizing  -  governing mode: {p['governs'].upper()}  "
                 f"({winner})", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4)
    _mark_stations(ax)
    ax.legend(loc="upper right", fontsize=8.5)
    _caption(fig,
             "Skin gauge per station = upper envelope of min-gauge, yield and buckling "
             "(black). Buckling (red) tracks the moment envelope (grey, right axis) and "
             "exceeds min-gauge in the loaded mid-body -> buckling governs and the gauge "
             "rises with MTOW; the lightly-loaded nose/tail stay at min gauge.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = out_dir / "fuselage_skin_thickness.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_material_comparison(details, out_dir):
    """Figure 3 -- CFRP vs aluminium total mass, split into skin and recovery.

    Each bar is one candidate material sized to the SAME loads. The lower segment is the
    integrated skin mass; the upper segment is the frames/floor/fittings recovered via the
    primary-structure fraction (1/FUS_PRIMARY_FRACTION). The lighter feasible bar is the
    one ``fuselage_mass`` returns ('<- selected')."""
    by_mat = details["by_material"]
    winner = details["material"]
    names = list(by_mat)
    skin = [by_mat[n]["m_skin"] for n in names]
    recovery = [by_mat[n]["mass"] - by_mat[n]["m_skin"] for n in names]

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.bar(names, skin, color="tab:blue", label="integrated skin")
    ax.bar(names, recovery, bottom=skin, color="tab:cyan",
           label=f"frames/floor/fittings recovery (1/{FUS_PRIMARY_FRACTION:.2f})")
    for i, n in enumerate(names):
        total = by_mat[n]["mass"]
        mark = "  <- selected" if n == winner else ""
        ax.text(i, total + 6, f"{total:.0f} kg{mark}", ha="center",
                fontweight="bold" if n == winner else "normal", fontsize=9.5,
                color="firebrick" if n == winner else "black")
        ax.text(i, skin[i] / 2, f"{by_mat[n]['governs']}", ha="center",
                va="center", fontsize=8, color="white")

    ax.set_ylabel("fuselage mass [kg]")
    ax.set_title(f"Material trade (lighter feasible kept):  winner = {winner}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.margins(y=0.18)
    _caption(fig,
             "Both materials are sized to the same superimposed loads; bars split into "
             "integrated skin (governing mode labelled in white) plus primary-fraction "
             "recovery. The lighter bar is returned -- aluminium is penalised by its "
             "2 mm min gauge, so CFRP (buckling-sized) wins.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = out_dir / "fuselage_material_comparison.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_mass_vs_mtow(out_dir):
    """Figure 4 -- fuselage mass over an MTOW sweep: physics vs legacy regression.

    The physics curve RISES with MTOW because heavier aircraft -> larger inertia/landing
    moments -> thicker buckling-critical skin (gauge ~ sqrt(M) ~ sqrt(MTOW)). Overlaid is
    the legacy empirical regression for sanity; both share the same order of magnitude and
    the same gentle upward trend."""
    mtow = np.linspace(1500.0, 3500.0, 40)
    m_phys = np.array([fuselage_mass(m, method="physics") for m in mtow])
    m_reg = np.array([fuselage_mass(m, method="regression") for m in mtow])

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(mtow, m_phys, color="tab:green", lw=2.2,
            label="physics (buckling-sized shell-beam, lighter material)")
    ax.plot(mtow, m_reg, color="black", ls="--", lw=1.8,
            label="legacy regression (USAF/Nicolai)")
    ax.set_xlabel("MTOW [kg]")
    ax.set_ylabel("fuselage mass [kg]")
    ax.set_title("Fuselage mass vs MTOW  -  physics method vs legacy regression",
                 fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    _caption(fig,
             "Physics mass rises with MTOW because the buckling-critical skin gauge "
             "scales as ~sqrt(moment) ~ sqrt(MTOW); the lightly-loaded ends stay at min "
             "gauge. Same magnitude and trend as the empirical regression, but now "
             "traceable to loads, geometry and material.")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = out_dir / "fuselage_mass_vs_mtow.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    details = fuselage_mass(MTOW_REF, return_details=True)

    print(f"Fuselage physics sizing @ MTOW = {MTOW_REF:.0f} kg")
    print(f"  winner   : {details['material']}  ({details['mass']:.1f} kg)")
    print(f"  governs  : {details['governs']}  | condition: {details['governing_condition']}")
    print(f"  M_design : {details['m_design']/1e3:.2f} kN.m  | peak gauge: {details['t_max']*1e3:.2f} mm")

    paths = [
        plot_load_shear_moment(details, OUT_DIR),
        plot_skin_thickness(details, OUT_DIR),
        plot_material_comparison(details, OUT_DIR),
        plot_mass_vs_mtow(OUT_DIR),
    ]
    print("\nSaved figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
