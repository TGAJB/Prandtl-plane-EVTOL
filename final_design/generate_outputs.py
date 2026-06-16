"""
generate_outputs.py
===================
Runs in a FRESH interpreter (launched by run_design.py) AFTER the chosen design
has been written to final_design/chosen_design.json. Because every module is
imported fresh here, parameters.py applies the override hook at import and the
WHOLE sheet is consistent with the chosen design.

It collects, into final_design/results/:
  * report.md            - human-readable summary of all department characteristics
  * data/characteristics.json
  * data/parameters_full.{txt,json}   - the complete parameter list (dump_parameters)
  * figures/*.png        - every graph, consolidated (captured + copied)
  * run_log.txt          - full console log of this run

Department plotting mains are run BEST-EFFORT: a failure in one is logged and the
run continues. The existing Sobol/Tornado sensitivity heatmaps are NOT recomputed;
they are copied in as standalone artifacts.
"""

import importlib
import importlib.util
import json
import shutil
import sys
import traceback
from pathlib import Path

# --- path bootstrap (project root + every dir holding bare-name modules) -------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FC = PROJECT_ROOT / "final_characteristics"
for _p in (
    PROJECT_ROOT,
    PROJECT_ROOT / "final_design",
    _FC,
    _FC / "vehicle_dynamics",
    _FC / "structures",
    _FC / "aerodynamics",
    PROJECT_ROOT / "class_II_sizing",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# --- results layout ------------------------------------------------------------
RESULTS = PROJECT_ROOT / "final_design" / "results"
FIG_DIR = RESULTS / "figures"
DATA_DIR = RESULTS / "data"
for _d in (RESULTS, FIG_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- figure capture MUST be set up before anything imports pyplot --------------
import figure_capture as figcap          # noqa: E402  (sets Agg backend on import)
figcap.init(FIG_DIR)


# --- tee stdout/stderr to results/run_log.txt ---------------------------------
class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


_LOG = open(RESULTS / "run_log.txt", "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)


def _np_safe(obj):
    """Recursively make a value JSON-serialisable (numpy -> python)."""
    import numpy as np
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _np_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_np_safe(v) for v in obj]
    if isinstance(obj, bool) or isinstance(obj, (int, float, str)) or obj is None:
        return obj
    return repr(obj)


def _section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ===========================================================================
# 1. CHARACTERISTICS (structured, reliable numbers)
# ===========================================================================
def collect_characteristics():
    """Pull the headline numbers from the override-aware structured entry points."""
    chars = {}

    # --- design overrides in effect -------------------------------------------
    import parameters as params
    chars["design_overrides"] = dict(getattr(params, "_DESIGN_OVERRIDES", {}))

    # --- mass / MTOW breakdown -------------------------------------------------
    _section("MASS & MTOW (class_II_sizing.mtow_sizing)")
    try:
        import mtow_sizing
        mass = mtow_sizing.converged_mass(force_recompute=True, verbose=True)
        chars["mass"] = {k: _np_safe(v) for k, v in mass.items() if k != "wing_geom"}
        wg = mass.get("wing_geom", {})
        chars["wing_geometry"] = {
            k: _np_safe(wg.get(k))
            for k in ("total_area_m2", "aspect_ratio", "root_chord_m", "tip_chord_m", "mac_m", "span_m")
            if k in wg
        }
    except Exception as exc:
        print(f"[mass] FAILED: {exc}")
        traceback.print_exc()

    # --- inertias & c.g. (cruise + VTOL) --------------------------------------
    _section("INERTIA & C.G. (class_II_sizing.MMOI)")
    try:
        import MMOI
        cruise = MMOI.aircraft_inertia(verbose=True, configuration="cruise")
        vtol = MMOI.aircraft_inertia_vtol(verbose=False)
        chars["inertia_cruise"] = {k: _np_safe(cruise[k]) for k in ("mass", "cg", "Ixx", "Iyy", "Izz", "Ixz")}
        chars["inertia_vtol"] = {k: _np_safe(vtol[k]) for k in ("mass", "cg", "Ixx", "Iyy", "Izz", "Ixz")}
    except Exception as exc:
        print(f"[MMOI] FAILED: {exc}")
        traceback.print_exc()

    # --- stability derivatives, requirements, margins, c.g. envelopes ---------
    _section("STABILITY & C.G. ENVELOPES (final_characteristics.stability_eval)")
    try:
        import stability_eval
        res = stability_eval.evaluate_stability()
        reqs = {k: bool(v) for k, v in res.get("requirements", {}).items()}
        chars["stability"] = {
            "mtow": _np_safe(res.get("mtow")),
            "derivatives": _np_safe(res.get("derivatives", {})),
            "requirements": reqs,
            "margins": _np_safe(res.get("margins", {})),
            "worst_margin": _np_safe(res.get("worst_margin")),
            "accepted": bool(res.get("accepted", all(reqs.values()) if reqs else False)),
            "neutral_point_and_cg": _np_safe(res.get("np_cg", {})),
        }
        n_pass = sum(reqs.values())
        print(f"  requirements passing: {n_pass}/{len(reqs)}   "
              f"accepted={chars['stability']['accepted']}   "
              f"worst_margin={chars['stability']['worst_margin']}")
    except Exception as exc:
        print(f"[stability] FAILED: {exc}")
        traceback.print_exc()

    # --- matching diagram feasible W/S window ---------------------------------
    _section("MATCHING DIAGRAM (final_characteristics.matching_diagram)")
    try:
        import matching_diagram
        low, high = matching_diagram.feasible_wing_loading_range()
        chars["matching_diagram"] = {"wing_loading_min_N_m2": float(low), "wing_loading_max_N_m2": float(high)}
        print(f"  feasible W/S window: [{low:.1f}, {high:.1f}] N/m^2")
    except Exception as exc:
        print(f"[matching_diagram] FAILED: {exc}")
        traceback.print_exc()

    return chars


# ===========================================================================
# 2. DEPARTMENT GRAPHS + CONSOLE OUTPUTS (best-effort)
# ===========================================================================
# (label, module). Each department file is EXECUTED as a script (run_name
# "__main__") so it runs whether it exposes a main() or only an
# `if __name__ == "__main__":` block. Runs under figcap.section so any plt.show()
# figures are saved as <label>_NN.png. Console-only modules (ecs, aux loads) are
# included so their numbers land in run_log.txt. Best-effort: a failing or
# optional-dependency-missing department (e.g. plotly) is logged, not fatal.
PLOT_TASKS = [
    ("matching_diagram",   "matching_diagram"),
    ("aero_model",         "aero_model"),
    ("drag_polar",         "dragpolar"),
    ("propeller_ppe",      "ppe"),
    ("ecs",                "ecs_sizing"),
    ("aux_loads",          "aux_loads"),
    ("static_long_stab",   "stat_long_stab_anal"),
    ("vtol_oei",           "stat_long_stab_anal_vtol_oei_plot"),
    ("dyn_stab",           "dyn_stab_analysis"),
    ("landing_gear_trade", "trade_off_landing_gear"),
]


def run_department_plots():
    import runpy
    summary = {}
    for label, module_name in PLOT_TASKS:
        _section(f"DEPARTMENT RUN: {label}  ({module_name})")
        try:
            spec = importlib.util.find_spec(module_name)
        except Exception as exc:                  # find_spec can import a bad parent
            spec = None
            print(f"  [skip] could not locate {module_name}: {exc}")
        if spec is None or not getattr(spec, "origin", None):
            summary[label] = "module not found"
            print(f"  [skip] {module_name} not found on path")
            continue
        try:
            with figcap.section(label):
                runpy.run_path(spec.origin, run_name="__main__")
            summary[label] = "ok"
        except SystemExit as exc:                 # some scripts call sys.exit
            print(f"  [note] {label} called sys.exit({exc.code}); continuing")
            summary[label] = f"sys.exit({exc.code})"
        except ModuleNotFoundError as exc:        # optional dep (e.g. plotly) absent
            print(f"  [SKIP - missing dependency] {label}: {exc}")
            summary[label] = f"missing dependency: {exc.name}"
        except Exception as exc:
            print(f"  [FAILED] {label}: {exc}")
            traceback.print_exc()
            summary[label] = f"failed: {exc}"
    figcap.flush("misc")
    return summary


# ===========================================================================
# 3. COPY EXISTING / NEWLY-WRITTEN ARTIFACTS INTO results/figures
# ===========================================================================
# Artifacts written elsewhere in the repo: COPY leaves the originals in place
# (department plot dirs + the standalone sensitivity heatmaps, which are NOT
# re-run here). MOVE relocates cwd-relative savefigs that would otherwise litter
# the project root into results/figures.
COPY_GLOBS = [
    "final_characteristics/sensitivity_plots/**/*.png",
    "final_characteristics/sensitivity_plots/**/*.csv",
    "final_characteristics/vehicle_dynamics/plots/**/*.png",
    "final_characteristics/aerodynamics/**/*results*/**/*.png",
    "final_characteristics/aerodynamics/**/*results*/**/*.csv",
    "class_II_sizing/sensitivity_plots/*.png",
    "class_II_sizing/sensitivity_plots/*.csv",
    "class_II_sizing/fuselage_sizing_figures/*.png",
]
# cwd-relative outputs written by department scripts into the project root.
MOVE_GLOBS = [
    "polar.png",
    "propeller_*.png",
    "fd_curves.png",
    "*_curves.png",
]


def copy_artifacts():
    _section("COPYING / RELOCATING ARTIFACTS INTO results/figures")
    moved = copied = 0
    for pattern in MOVE_GLOBS:
        for src in PROJECT_ROOT.glob(pattern):
            if not src.is_file():
                continue
            dest = FIG_DIR / src.name
            try:
                shutil.move(str(src), str(dest))   # relocate so the repo root stays clean
                moved += 1
            except Exception as exc:
                print(f"  [move failed] {src} -> {dest.name}: {exc}")
    for pattern in COPY_GLOBS:
        for src in PROJECT_ROOT.glob(pattern):
            if not src.is_file():
                continue
            tag = src.parent.name                  # tag w/ parent folder so names don't collide
            dest = FIG_DIR / f"{tag}__{src.name}"
            try:
                shutil.copy2(src, dest)
                copied += 1
            except Exception as exc:
                print(f"  [copy failed] {src} -> {dest.name}: {exc}")
    print(f"  moved {moved} cwd artifact(s), copied {copied} repo artifact(s) into {FIG_DIR}")
    return moved + copied


# ===========================================================================
# 4. REPORT
# ===========================================================================
def write_report(chars, plot_summary, params_payload):
    lines = ["# Final Design — Consolidated Report", ""]

    ov = chars.get("design_overrides") or {}
    if ov:
        lines += ["## Active design (override)", "", "| variable | value |", "|---|---|"]
        lines += [f"| `{k}` | {v} |" for k, v in ov.items()]
    else:
        lines += ["## Active design", "", "_Baseline parameter sheet (no override applied)._"]
    lines.append("")

    mass = chars.get("mass", {})
    if mass:
        lines += ["## Mass & MTOW [kg]", "", "| component | mass |", "|---|---|"]
        for k in ("mtow", "fuselage", "wing", "winglet", "landing_gear", "tail",
                  "motors", "props", "hubs", "battery", "payload", "misc", "hinge"):
            if k in mass:
                lines.append(f"| {k} | {mass[k]:.2f} |" if isinstance(mass[k], (int, float)) else f"| {k} | {mass[k]} |")
        if "max_power_kw" in mass:
            lines.append(f"| max_power_kW | {mass['max_power_kw']:.1f} |")
        lines.append("")

    wg = chars.get("wing_geometry", {})
    if wg:
        lines += ["## Wing geometry", "", "| quantity | value |", "|---|---|"]
        lines += [f"| {k} | {v:.4f} |" if isinstance(v, (int, float)) else f"| {k} | {v} |" for k, v in wg.items()]
        lines.append("")

    for cfg in ("inertia_cruise", "inertia_vtol"):
        d = chars.get(cfg)
        if d:
            cg = d.get("cg")
            lines += [f"## {cfg.replace('_', ' ').title()}", "",
                      f"- total mass: {d.get('mass'):.2f} kg" if isinstance(d.get('mass'), (int, float)) else f"- total mass: {d.get('mass')}",
                      f"- c.g. (x,y,z): {cg}",
                      f"- Ixx/Iyy/Izz: {d.get('Ixx')}, {d.get('Iyy')}, {d.get('Izz')}",
                      f"- Ixz: {d.get('Ixz')}", ""]

    stab = chars.get("stability")
    if stab:
        lines += ["## Stability", "",
                  f"- MTOW used: {stab.get('mtow')}",
                  f"- accepted: **{stab.get('accepted')}**   worst normalised margin: {stab.get('worst_margin')}", ""]
        reqs = stab.get("requirements", {})
        margins = stab.get("margins", {})
        if reqs:
            lines += ["### Requirements", "", "| requirement | pass | margin |", "|---|---|---|"]
            for name in reqs:
                lines.append(f"| {name} | {'PASS' if reqs[name] else 'FAIL'} | {margins.get(name)} |")
            lines.append("")
        derivs = stab.get("derivatives", {})
        if derivs:
            lines += ["### Stability derivatives", "", "| derivative | value |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in derivs.items()]
            lines.append("")

    md = chars.get("matching_diagram")
    if md:
        lines += ["## Matching diagram", "",
                  f"- feasible W/S window: [{md['wing_loading_min_N_m2']:.1f}, {md['wing_loading_max_N_m2']:.1f}] N/m²", ""]

    lines += ["## Department runs (graphs / console)", "", "| department | status |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in plot_summary.items()]
    lines.append("")

    n_glob = len(params_payload.get("module_globals", {}))
    lines += ["## Parameter list", "",
              f"Full parameter dump: `data/parameters_full.txt` / `.json` "
              f"({n_glob} module-level constants + the AircraftParameters dataclass tree).",
              "", "## Figures", "",
              "All graphs are consolidated in `figures/`. The Sobol/Tornado sensitivity "
              "heatmaps are copied in as standalone artifacts (NOT recomputed by this run).", ""]

    (RESULTS / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print(f"final_design output generation -> {RESULTS}")
    chars = collect_characteristics()

    _section("FULL PARAMETER DUMP (dump_parameters)")
    import dump_parameters
    params_payload = dump_parameters.write_dump(DATA_DIR)
    print(f"  wrote parameters_full.txt / .json "
          f"({len(params_payload['module_globals'])} module globals)")

    plot_summary = run_department_plots()
    copy_artifacts()

    (DATA_DIR / "characteristics.json").write_text(json.dumps(_np_safe(chars), indent=2), encoding="utf-8")
    write_report(chars, plot_summary, params_payload)

    _section("DONE")
    print(f"Report : {RESULTS / 'report.md'}")
    print(f"Figures: {FIG_DIR}  ({len(list(FIG_DIR.glob('*.png')))} png)")
    print(f"Data   : {DATA_DIR}")
    _LOG.flush()


if __name__ == "__main__":
    main()
