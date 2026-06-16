"""
architecture_diagram.py
=======================

Standalone generator for model-architecture images of THIS repository.

Produces two complementary views under ``docs/architecture/``:

  1. module_dependencies  - an AUTO import-dependency graph, parsed STATICALLY with the
     ``ast`` module (the source is never imported/executed), clustered by subsystem
     (class_II_sizing, vehicle_dynamics, aerodynamics, structures, ...). parameters.py
     (the hub everything imports) and the entrypoints are highlighted.
  2. pipeline_dataflow    - a CURATED runtime pipeline / data-flow overlay (hand-maintained
     below in ``build_pipeline_model``). It captures the coupling that imports CANNOT show:
     the MTOW converger fixed-point loop and the ``setattr`` overrides the optimiser patches
     onto class_II_sizing.mass_components / MMOI.

Why static AST (not pydeps / pyreverse): the repo has no ``__init__.py`` packages and uses
bare ``sys.path`` imports, and several modules execute code on import (mass_components runs
winglet_mass(1.2), fusion_geometry loads geometry.json). Importing them to trace
dependencies is fragile; parsing them is not.

Each view is written as ``.dot`` and Mermaid ``.mmd`` (both emitted by this file, so they
work with ZERO extra binaries and render on GitHub / mermaid.live), plus ``.svg``/``.png``
when Graphviz is available. A combined ``architecture.md`` embeds both Mermaid diagrams.

Usage
-----
    python tools/architecture_diagram.py                  # svg + dot + mmd + md
    python tools/architecture_diagram.py --format png
    python tools/architecture_diagram.py --out docs/architecture --include-tests

Dependencies
------------
Raster (.svg/.png) output needs the Graphviz **system binary** ``dot`` on PATH
(Windows: ``winget install graphviz``; the ``graphviz`` Python package, ``pip install
graphviz``, is used if present but the bundled ``dot`` writer/subprocess fallback also
works). The ``.dot`` and ``.mmd`` text outputs need nothing beyond the standard library.
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Directories never walked, and individual files/prefixes skipped by default.
EXCLUDE_DIRS = {".venv", "__pycache__", ".git", ".idea", ".vscode", "node_modules", "tools"}
EXCLUDE_FILE_REL_PREFIXES = ("fusion/experimental/",)
EXCLUDE_FILE_NAMES = {"OLDmain.py"}

# Light background tint per subsystem cluster (nodes sit on top in white).
CLUSTER_COLORS = {
    "root": "#fde9d9",
    "class_II_sizing": "#dbe9fb",
    "final_characteristics": "#e2efda",
    "final_characteristics/vehicle_dynamics": "#d9eef1",
    "final_characteristics/aerodynamics": "#fce4ec",
    "final_characteristics/structures": "#ece3f6",
    "support_files": "#fff2cc",
    "fusion": "#eeeeee",
}
DEFAULT_CLUSTER_COLOR = "#f2f2f2"

# Modules given a distinct look in the dependency graph.
HUB_MODULES = {"parameters"}
ENTRYPOINT_HINTS = {
    "final_characteristics/optimiser",
    "class_II_sizing/mtow_sizing",
    "final_characteristics/stability_eval",
}


# ---------------------------------------------------------------------------
# Neutral graph model (so DOT and Mermaid share one source of truth)
# ---------------------------------------------------------------------------
@dataclass
class GNode:
    id: str
    label: str
    cluster: str = "root"
    fill: str = "#ffffff"
    border: str = "#555555"
    penwidth: float = 1.0
    fontsize: int = 11
    shape: str = "box"


@dataclass
class GEdge:
    src: str
    dst: str
    label: str = ""
    dashed: bool = False
    color: str = "#888888"
    both: bool = False


@dataclass
class GCluster:
    id: str
    label: str
    color: str = DEFAULT_CLUSTER_COLOR


@dataclass
class GModel:
    name: str
    title: str
    rankdir: str = "LR"
    nodes: list[GNode] = field(default_factory=list)
    edges: list[GEdge] = field(default_factory=list)
    clusters: dict[str, GCluster] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1-3. Discover modules, build the name index, extract import edges (no exec)
# ---------------------------------------------------------------------------
@dataclass
class ModuleInfo:
    rel: str          # 'final_characteristics/vehicle_dynamics/aircraft.py'
    mod_id: str       # 'final_characteristics/vehicle_dynamics/aircraft'
    basename: str     # 'aircraft'
    dotted: str       # 'final_characteristics.vehicle_dynamics.aircraft'
    path: Path
    has_main: bool = False
    importable: bool = True   # files with spaces in the name can't be import targets


def _subsystem_of(rel: str) -> str:
    parts = rel.split("/")
    if len(parts) == 1:
        return "root"
    if parts[0] == "final_characteristics" and len(parts) >= 3:
        return "final_characteristics/" + parts[1]
    return parts[0]


def discover_modules(root: Path, include_tests: bool) -> list[ModuleInfo]:
    modules: list[ModuleInfo] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        parts = rel.split("/")
        if any(p in EXCLUDE_DIRS for p in parts[:-1]):
            continue
        if not include_tests and parts[0] == "tests":
            continue
        if path.name in EXCLUDE_FILE_NAMES:
            continue
        if any(rel.startswith(pre) for pre in EXCLUDE_FILE_REL_PREFIXES):
            continue
        mod_id = rel[:-3]
        basename = parts[-1][:-3]
        importable = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", basename) is not None
        modules.append(ModuleInfo(
            rel=rel, mod_id=mod_id, basename=basename,
            dotted=mod_id.replace("/", "."), path=path, importable=importable,
        ))
    return modules


def build_index(modules: list[ModuleInfo]):
    by_dotted: dict[str, str] = {}
    by_basename: dict[str, list[str]] = defaultdict(list)
    for m in modules:
        if not m.importable:
            continue
        by_dotted[m.dotted] = m.mod_id
        by_basename[m.basename].append(m.mod_id)
    return by_dotted, by_basename


def _resolve(name: str, by_dotted, by_basename) -> str | None:
    """Map an imported name to an internal module id, or None if external/ambiguous.

    Handles dotted imports ('class_II_sizing.mtow_sizing') AND the repo's bare imports
    ('mtow_sizing', 'hinge_loading') that work only via sys.path hacks."""
    if name in by_dotted:
        return by_dotted[name]
    base = name.split(".")[-1]
    ids = by_basename.get(base, [])
    if len(ids) == 1:
        return ids[0]
    if len(ids) > 1:
        exact = [i for i in ids if i.replace("/", ".").endswith(name)]
        return exact[0] if len(exact) == 1 else None
    return None


def extract_edges(modules, by_dotted, by_basename):
    """Return (edges, external_top_levels, parse_errors). edges is a set of (src_id, dst_id)."""
    edges: set[tuple[str, str]] = set()
    externals: Counter = Counter()
    parse_errors: list[str] = []

    for m in modules:
        try:
            tree = ast.parse(m.path.read_text(encoding="utf-8", errors="replace"), filename=str(m.path))
        except SyntaxError as exc:
            parse_errors.append(f"{m.rel}: {exc}")
            continue

        m.has_main = _has_main_guard(tree)

        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:      # relative import (none expected)
                    continue
                if node.module:
                    candidates = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            else:
                continue

            matched_internal = False
            for cand in candidates:
                target = _resolve(cand, by_dotted, by_basename)
                if target and target != m.mod_id:
                    edges.add((m.mod_id, target))
                    matched_internal = True
            if not matched_internal and candidates:
                top = candidates[0].split(".")[0]
                externals[top] += 1

    return edges, externals, parse_errors


def _has_main_guard(tree: ast.AST) -> bool:
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.If):
            t = node.test
            if (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name)
                    and t.left.id == "__name__"):
                return True
    return False


# ---------------------------------------------------------------------------
# 4. Build the module-dependency model
# ---------------------------------------------------------------------------
def build_module_graph_model(modules, edges) -> GModel:
    indeg: Counter = Counter()
    for _src, dst in edges:
        indeg[dst] += 1

    model = GModel(name="module_dependencies",
                   title="Module dependency graph (static imports, clustered by subsystem)")

    seen_clusters: set[str] = set()
    for m in modules:
        sub = _subsystem_of(m.rel)
        if sub not in seen_clusters:
            model.clusters[sub] = GCluster(
                id=sub, label=sub, color=CLUSTER_COLORS.get(sub, DEFAULT_CLUSTER_COLOR))
            seen_clusters.add(sub)

        fill, border, penwidth = "#ffffff", "#555555", 1.0
        if m.mod_id in HUB_MODULES:
            fill, border, penwidth = "#ffd966", "#bf9000", 2.2
        elif m.mod_id in ENTRYPOINT_HINTS or m.has_main:
            fill, border, penwidth = "#eaf5ea", "#2e7d32", 2.0

        model.nodes.append(GNode(
            id=m.mod_id, label=m.basename, cluster=sub,
            fill=fill, border=border, penwidth=penwidth,
            fontsize=11 + min(indeg.get(m.mod_id, 0), 12),
        ))

    for src, dst in sorted(edges):
        model.edges.append(GEdge(src=src, dst=dst))
    return model


# ---------------------------------------------------------------------------
# 5. Curated pipeline / data-flow overlay (hand-maintained, not auto-derived)
# ---------------------------------------------------------------------------
def build_pipeline_model() -> GModel:
    """The RUNTIME architecture imports can't show: the converger fixed-point loop and the
    optimiser's setattr overrides. Edit this when the pipeline changes."""
    model = GModel(name="pipeline_dataflow",
                   title="Runtime pipeline & data flow (curated)")

    # Four left-to-right stages as columns: Inputs -> Optimiser -> MTOW loop ->
    # Stability evaluation, with one feedback edge (a failed constraint sends the
    # optimiser back for the next candidate). The hand-tuned SVG
    # (docs/architecture/pipeline_dataflow.svg) renders these as clean vertical
    # columns; this model keeps the .dot / .mmd content in sync.
    C = {
        "src": GCluster("src", "Inputs (read by every stage)", "#fff2cc"),
        "opt": GCluster("opt", "Optimiser - NSGA-II (pymoo)", "#fde9d9"),
        "mtow": GCluster("mtow", "MTOW convergence loop", "#dbe9fb"),
        "stab": GCluster("stab", "Stability evaluation", "#d9eef1"),
    }
    model.clusters = C

    def N(nid, label, cl, **kw):
        model.nodes.append(GNode(id=nid, label=label, cluster=cl, **kw))

    # Inputs
    N("in_design", "Design inputs\\ngeometry / masses /\\nmaterials / constants", "src")
    N("in_match", "Matching diagram\\nW/S range, power & CL limits", "src")

    # Optimiser
    N("opt_vars", "Tunes 7 design variables:\\nfront-wing position & height /\\nwing loading (W/S) /\\naft-total area split /\\nvertical-tail span /\\ncruise & VTOL battery position", "opt")
    N("opt_main", "NSGA-II population search\\nevolves designs toward the best\\nmass-stability trade-off", "opt",
      fill="#eaf5ea", border="#2e7d32", penwidth=2.0)
    N("opt_out", "Outputs (key values):\\nPareto front (MTOW vs stability) /\\nchosen design + 7 tuned values\\n(written back to the inputs)", "opt")

    # MTOW convergence loop
    N("mtow_loop", "Damped fixed-point loop\\nguess weight -> size all parts ->\\nre-sum -> repeat until <1% change", "mtow")
    N("mtow_parts", "Sizes each pass:\\nhover power / wing + winglet (structure) /\\nfuselage / tail / landing gear /\\nmotors / propellers + hubs / battery", "mtow")
    N("mtow_out", "Converged MTOW", "mtow", fill="#eef5ff", border="#2f5597")

    # Stability evaluation
    N("stab_aero", "Live aero (VLM)\\nlift slope, aero centre &\\ninduced drag from geometry", "stab")
    N("stab_eval", "Stability derivatives +\\ncruise / VTOL c.g. envelopes", "stab")
    N("stab_con", "Constraints checked:\\nstiffness signs (C_M_alpha<0, C_N_beta>0) /\\ncruise c.g. within limits /\\nVTOL one-engine-out c.g. /\\nwing structurally feasible /\\npower & cruise-CL feasible", "stab")

    E = model.edges.append
    E(GEdge("in_design", "opt_main", "search bounds"))
    E(GEdge("in_match", "opt_main", "W/S range + feasibility"))
    E(GEdge("opt_vars", "opt_main"))
    E(GEdge("opt_main", "opt_out", "on convergence"))
    E(GEdge("opt_main", "mtow_loop", "design vector"))
    E(GEdge("mtow_loop", "mtow_parts", "iterate", both=True))
    E(GEdge("mtow_loop", "mtow_out"))
    E(GEdge("mtow_out", "stab_eval", "converged MTOW"))
    E(GEdge("stab_aero", "stab_eval", "CL_alpha, x_ac"))
    E(GEdge("stab_eval", "stab_con", "derivatives + c.g."))
    E(GEdge("stab_con", "opt_main", "any constraint not met -> next candidate", dashed=True, color="#c0392b"))
    return model


# NSGA-II is documented by a dedicated, more VISUAL matplotlib figure:
#   tools/nsga2_diagram.py  ->  docs/architecture/nsga2_algorithm.{png,svg}
# (a box-and-arrow flowchart is too text-heavy for an algorithm explanation).


# ---------------------------------------------------------------------------
# 6. Emitters: DOT and Mermaid (both from the neutral model)
# ---------------------------------------------------------------------------
def to_dot(model: GModel) -> str:
    out = [f'digraph "{model.name}" {{',
           f'  rankdir={model.rankdir};',
           '  labelloc="t";',
           f'  label="{model.title}";',
           '  fontname="Helvetica"; fontsize=16;',
           '  node [fontname="Helvetica", style="rounded,filled"];',
           '  edge [fontname="Helvetica", fontsize=9, arrowsize=0.7];',
           '  compound=true;']

    by_cluster: dict[str, list[GNode]] = defaultdict(list)
    for n in model.nodes:
        by_cluster[n.cluster].append(n)

    for cid, cl in model.clusters.items():
        safe = re.sub(r"[^A-Za-z0-9_]", "_", cid)
        out.append(f'  subgraph "cluster_{safe}" {{')
        out.append(f'    label="{cl.label}"; style="filled,rounded"; color="#bbbbbb"; '
                   f'fillcolor="{cl.color}"; fontname="Helvetica-Bold"; fontsize=12;')
        for n in by_cluster.get(cid, []):
            label = n.label.replace('"', '\\"')
            out.append(
                f'    "{n.id}" [label="{label}", shape={n.shape}, fillcolor="{n.fill}", '
                f'color="{n.border}", penwidth={n.penwidth}, fontsize={n.fontsize}];')
        out.append("  }")

    for e in model.edges:
        attrs = [f'color="{e.color}"']
        if e.label:
            attrs.append(f'label="{e.label}"; fontcolor="{e.color}"')
        if e.dashed:
            attrs.append('style=dashed')
        if e.both:
            attrs.append('dir=both')
        out.append(f'  "{e.src}" -> "{e.dst}" [{", ".join(attrs)}];')

    out.append("}")
    return "\n".join(out)


def _mermaid_tokens(model: GModel) -> dict[str, str]:
    tokens, used = {}, set()
    for n in model.nodes:
        base = re.sub(r"[^A-Za-z0-9_]", "_", n.id) or "n"
        if base[0].isdigit():
            base = "n_" + base
        tok, i = base, 1
        while tok in used:
            tok, i = f"{base}_{i}", i + 1
        used.add(tok)
        tokens[n.id] = tok
    return tokens


def to_mermaid(model: GModel) -> str:
    tok = _mermaid_tokens(model)
    direction = "LR" if model.rankdir in ("LR", "RL") else "TB"
    out = ["```mermaid", f"flowchart {direction}"]

    by_cluster: dict[str, list[GNode]] = defaultdict(list)
    for n in model.nodes:
        by_cluster[n.cluster].append(n)

    def mlabel(s: str) -> str:
        return s.replace("\\n", "<br/>").replace('"', "'")

    def mnode(n: GNode) -> str:
        lab = mlabel(n.label)
        if n.shape == "diamond":
            return f'{tok[n.id]}{{"{lab}"}}'
        if n.shape in ("ellipse", "oval"):
            return f'{tok[n.id]}(["{lab}"])'
        return f'{tok[n.id]}["{lab}"]'

    for cid, cl in model.clusters.items():
        safe = re.sub(r"[^A-Za-z0-9_]", "_", cid)
        out.append(f'  subgraph cl_{safe}["{mlabel(cl.label)}"]')
        for n in by_cluster.get(cid, []):
            out.append(f'    {mnode(n)}')
        out.append("  end")

    for e in model.edges:
        arrow = "-.->" if e.dashed else "-->"
        if e.label:
            out.append(f'  {tok[e.src]} {arrow}|{mlabel(e.label)}| {tok[e.dst]}')
        else:
            out.append(f'  {tok[e.src]} {arrow} {tok[e.dst]}')
        if e.both:
            out.append(f'  {tok[e.dst]} {arrow} {tok[e.src]}')

    out.append("```")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Rasterise DOT -> svg/png via Graphviz (python pkg, else `dot` binary, else skip)
# ---------------------------------------------------------------------------
def render_raster(dot_text: str, out_base: Path, fmt: str) -> bool:
    try:
        import graphviz  # type: ignore
        src = graphviz.Source(dot_text, filename=out_base.name, directory=str(out_base.parent))
        src.render(format=fmt, cleanup=True)
        return True
    except Exception:
        pass
    dot_bin = shutil.which("dot")
    if dot_bin:
        try:
            subprocess.run([dot_bin, f"-T{fmt}", "-o", str(out_base.with_suffix("." + fmt))],
                           input=dot_text.encode("utf-8"), check=True)
            return True
        except Exception:
            return False
    return False


def write_view(model: GModel, out_dir: Path, fmt: str) -> dict:
    dot_text = to_dot(model)
    mmd_text = to_mermaid(model)
    base = out_dir / model.name
    (base.with_suffix(".dot")).write_text(dot_text, encoding="utf-8")
    (base.with_suffix(".mmd")).write_text(mmd_text, encoding="utf-8")
    rastered = render_raster(dot_text, base, fmt)
    return {"name": model.name, "rastered": rastered, "fmt": fmt, "mmd": mmd_text}


def write_architecture_md(out_dir: Path, results: list[dict], summary: dict):
    lines = [
        "# Repository architecture",
        "",
        "_Auto-generated by `tools/architecture_diagram.py` (static AST parse — no module is "
        "imported/executed). Re-run that script to refresh._",
        "",
        f"- Internal modules parsed: **{summary['n_modules']}**  |  import edges: "
        f"**{summary['n_edges']}**",
        f"- Most-imported modules: {summary['top_imported']}",
        f"- External libraries used: {summary['externals']}",
        "",
        "## Module dependency graph",
        "",
        next(r["mmd"] for r in results if r["name"] == "module_dependencies"),
        "",
        "## Runtime pipeline & data flow (curated)",
        "",
        "Dashed red edges are `setattr` overrides the optimiser/converger patch onto module "
        "globals at runtime — coupling the import graph cannot show.",
        "",
        next(r["mmd"] for r in results if r["name"] == "pipeline_dataflow"),
        "",
        "## NSGA-II optimiser — how it functions",
        "",
        "The optimiser (`final_characteristics/optimiser.py`, pymoo) evolves a population of "
        "design vectors toward the Pareto front of (MTOW, worst stability margin). Visual "
        "explainer generated by `tools/nsga2_diagram.py`:",
        "",
        "![NSGA-II algorithm](nsga2_algorithm.svg)",
        "",
    ]
    (out_dir / "architecture.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate repo architecture diagrams (static AST).")
    ap.add_argument("--format", default="svg", choices=["svg", "png"], help="raster format (default svg)")
    ap.add_argument("--out", default="docs/architecture", help="output directory (default docs/architecture)")
    ap.add_argument("--include-tests", action="store_true", help="include the tests/ package")
    args = ap.parse_args(argv)

    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    modules = discover_modules(REPO_ROOT, include_tests=args.include_tests)
    by_dotted, by_basename = build_index(modules)
    edges, externals, parse_errors = extract_edges(modules, by_dotted, by_basename)

    module_model = build_module_graph_model(modules, edges)
    pipeline_model = build_pipeline_model()

    results = [write_view(module_model, out_dir, args.format),
               write_view(pipeline_model, out_dir, args.format)]

    indeg = Counter(dst for _s, dst in edges)
    id_to_base = {m.mod_id: m.basename for m in modules}
    top_imported = ", ".join(f"{id_to_base[i]} ({c})" for i, c in indeg.most_common(5))
    summary = {
        "n_modules": len(modules),
        "n_edges": len(edges),
        "top_imported": top_imported or "(none)",
        "externals": ", ".join(f"{n}" for n, _ in externals.most_common(10)) or "(none)",
    }
    write_architecture_md(out_dir, results, summary)

    print(f"Architecture diagrams -> {out_dir}")
    print(f"  modules parsed : {summary['n_modules']}")
    print(f"  import edges   : {summary['n_edges']}")
    print(f"  top imported   : {summary['top_imported']}")
    print(f"  external libs  : {summary['externals']}")
    for r in results:
        status = f"{r['fmt']} OK" if r["rastered"] else f"{r['fmt']} skipped (no Graphviz 'dot')"
        print(f"  {r['name']:22s}: .dot .mmd written, {status}")
    print("  architecture.md: combined Mermaid (GitHub-renderable)")
    if parse_errors:
        print(f"  note: {len(parse_errors)} file(s) skipped (syntax errors): "
              + "; ".join(parse_errors[:3]) + (" ..." if len(parse_errors) > 3 else ""))
    if not any(r["rastered"] for r in results):
        print("  (install Graphviz for raster output: `pip install graphviz` + "
              "`winget install graphviz`; .dot/.mmd already written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
