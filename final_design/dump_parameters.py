"""
dump_parameters.py
==================
Serialise the WHOLE parameter sheet (every dataclass field + every scalar module
global) so the final_design report includes the complete, override-aware
parameter list. Reads `parameters` AFTER the override hook has run, so the values
reflect the active design (chosen_design.json) automatically.

Outputs (written by write_dump):
    results/data/parameters_full.json   machine-readable, nested
    results/data/parameters_full.txt    human-readable, grouped
"""

import dataclasses
import json
from pathlib import Path

import numpy as np

import parameters as p


def _to_plain(v):
    """Recursively convert numpy / dataclass / containers to JSON-friendly python."""
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: _to_plain(getattr(v, f.name)) for f in dataclasses.fields(v)}
    if isinstance(v, (list, tuple)):
        return [_to_plain(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_plain(x) for k, x in v.items()}
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return repr(v)               # anything exotic -> its repr (never crash the dump)


def collect_dataclasses():
    """Every field of the umbrella AircraftParameters() as a nested dict."""
    return _to_plain(p.AircraftParameters())


def collect_module_globals():
    """Scalar (and short scalar-sequence) module-level constants from parameters.py."""
    out = {}
    for name in sorted(vars(p)):
        if name.startswith("_"):
            continue
        val = getattr(p, name)
        if isinstance(val, bool) or isinstance(val, (int, float, str)):
            out[name] = val
        elif isinstance(val, np.ndarray) and val.size <= 64:
            out[name] = val.tolist()
        elif isinstance(val, (list, tuple)) and val and all(
            isinstance(x, (int, float, str, bool)) for x in val
        ):
            out[name] = list(val)
    return out


def _render_text(payload):
    lines = ["=" * 78, "FULL PARAMETER LIST (override-aware snapshot)", "=" * 78, ""]

    overrides = payload.get("design_overrides") or {}
    lines.append("DESIGN OVERRIDES ACTIVE:" if overrides else "DESIGN OVERRIDES: none (baseline sheet)")
    for k, v in overrides.items():
        lines.append(f"    {k:20s} = {v}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("PART 1 - DATACLASS SHEET (AircraftParameters)")
    lines.append("-" * 78)
    for group, fields in payload["dataclasses"].items():
        lines.append(f"\n[{group}]")
        if isinstance(fields, dict):
            for fname, fval in fields.items():
                lines.append(f"    {fname:28s} = {fval}")
        else:
            lines.append(f"    {fields}")

    lines.append("")
    lines.append("-" * 78)
    lines.append("PART 2 - MODULE-LEVEL CONSTANTS")
    lines.append("-" * 78)
    for name, val in payload["module_globals"].items():
        lines.append(f"    {name:28s} = {val}")
    lines.append("")
    return "\n".join(lines)


def write_dump(data_dir):
    """Write parameters_full.{json,txt}; return the payload dict."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "design_overrides": getattr(p, "_DESIGN_OVERRIDES", {}),
        "dataclasses": collect_dataclasses(),
        "module_globals": collect_module_globals(),
    }
    (data_dir / "parameters_full.json").write_text(json.dumps(payload, indent=2))
    (data_dir / "parameters_full.txt").write_text(_render_text(payload))
    return payload
