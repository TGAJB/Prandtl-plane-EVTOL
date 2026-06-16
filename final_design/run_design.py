"""
final_design/run_design.py
==========================
SINGLE entry point for the project. It:
  1. runs the optimiser (NSGA-II) to choose a design,
  2. writes that design to final_design/chosen_design.json  (read by
     parameters.py at import -- NO rewriting of the parameters.py source), and
  3. regenerates ALL department characteristics, graphs and the full parameter
     list into final_design/results/ by launching generate_outputs.py in a FRESH
     interpreter (so the override propagates consistently to every module).

Usage
-----
    python final_design/run_design.py                 # optimise, then regenerate
    python final_design/run_design.py --skip-optimise # reuse current design (fast, graphs only)
    python final_design/run_design.py --clear-json    # delete override, regenerate the BASELINE

Notes
-----
* The optimiser (Phase 2 NSGA-II) can take minutes; use --skip-optimise to just
  (re)generate outputs for the design already pinned in chosen_design.json (or
  the baseline sheet if none exists).
* The Sobol/Tornado SENSITIVITY heatmaps are NOT recomputed here; their existing
  outputs are copied into results/figures as standalone artifacts.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = PROJECT_ROOT / "final_design"
JSON_PATH = FINAL_DIR / "chosen_design.json"
GENERATOR = FINAL_DIR / "generate_outputs.py"


def _run_optimiser():
    """Run the optimiser fresh (from the canonical baseline) and return its chosen
    design-variable dict."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    # Import here (not at module top) so parameters is loaded only after any stale
    # override has been removed by main(), keeping the SEARCH on the baseline sheet.
    from final_characteristics import optimiser

    accepted = optimiser.run_phase_1()
    if accepted:
        print("\nDesign accepted in Phase 1 -- pinning the current/default design vector.")
        return optimiser.default_design_vector()

    best = optimiser.run_phase_2()
    if best is None:
        print("\nPhase 2 found no feasible design -- pinning the default design vector.")
        return optimiser.default_design_vector()
    return best["design_vars"]


def main():
    ap = argparse.ArgumentParser(description="Run the optimiser and regenerate all department outputs.")
    ap.add_argument("--skip-optimise", action="store_true",
                    help="reuse the design already in chosen_design.json (or baseline) instead of optimising")
    ap.add_argument("--clear-json", action="store_true",
                    help="delete chosen_design.json and regenerate the baseline design")
    args = ap.parse_args()

    if args.clear_json:
        if JSON_PATH.exists():
            JSON_PATH.unlink()
            print(f"Removed {JSON_PATH} -- regenerating the BASELINE design.")
        else:
            print("No chosen_design.json present -- already on the baseline design.")
    elif args.skip_optimise:
        if JSON_PATH.exists():
            print(f"--skip-optimise: reusing existing design in {JSON_PATH}")
        else:
            print("--skip-optimise: no chosen_design.json -- regenerating the BASELINE design.")
    else:
        # Optimise from the canonical baseline: drop any stale override FIRST so the
        # search space / bounds / default vector come from the unmodified sheet.
        if JSON_PATH.exists():
            JSON_PATH.unlink()
        design_vars = _run_optimiser()
        design_vars = {k: round(float(v), 6) for k, v in design_vars.items()}
        JSON_PATH.write_text(json.dumps(design_vars, indent=2))
        print(f"\nWrote chosen design to {JSON_PATH}:")
        for k, v in design_vars.items():
            print(f"    {k:18s} = {v}")

    # Regenerate every department output in a FRESH interpreter so parameters.py
    # reads chosen_design.json at import and the whole sheet is consistent.
    print("\nGenerating department outputs in a fresh interpreter ...\n")
    completed = subprocess.run([sys.executable, str(GENERATOR)], cwd=str(PROJECT_ROOT))
    print(f"\nResults in: {FINAL_DIR / 'results'}")
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
