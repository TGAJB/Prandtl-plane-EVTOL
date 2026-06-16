"""
figure_capture.py
=================
Headless figure capture for the final_design runner.

Importing this module forces matplotlib's non-interactive 'Agg' backend and
monkeypatches ``pyplot.show()`` so that ANY department script which calls
``plt.show()`` has its currently-open figures SAVED into the results/figures
folder instead of trying to open a (blocking) GUI window. The department scripts
themselves are NOT edited.

Usage (in the fresh subprocess, BEFORE importing any department module):

    import figure_capture as figcap
    figcap.init(FIG_DIR)
    with figcap.section("matching_diagram"):
        matching_diagram.main()      # its plt.show() -> matching_diagram_01.png
    figcap.flush("misc")             # save any figures left open without show()
"""

import contextlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # MUST run before pyplot is first used anywhere
import matplotlib.pyplot as _plt  # noqa: E402

_FIG_DIR = None
_label = "figure"
_counter = 0
_saved = []


def init(fig_dir):
    """Point the capture at an output directory and install the show() patch."""
    global _FIG_DIR
    _FIG_DIR = Path(fig_dir)
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    _plt.show = _capture_show     # monkeypatch: show() now saves+closes instead


def _save_open_figures():
    """Save every currently-open figure as <label>_<n>.png, then close it."""
    global _counter
    paths = []
    if _FIG_DIR is None:
        return paths
    for num in _plt.get_fignums():
        fig = _plt.figure(num)
        _counter += 1
        out = _FIG_DIR / f"{_label}_{_counter:02d}.png"
        try:
            fig.savefig(out, dpi=150, bbox_inches="tight")
            paths.append(out)
            _saved.append(out)
        except Exception as exc:                      # never abort a run on a bad figure
            print(f"[figcap] failed to save {out.name}: {exc}")
        _plt.close(fig)
    return paths


def _capture_show(*_args, **_kwargs):
    """Replacement for plt.show(): persist the open figures rather than blocking."""
    _save_open_figures()


@contextlib.contextmanager
def section(label):
    """Name + count figures produced inside this block as <label>_NN.png."""
    global _label, _counter
    prev_label, prev_counter = _label, _counter
    _label, _counter = label, 0
    try:
        yield
    finally:
        _save_open_figures()      # catch figures left open without a show() call
        _label, _counter = prev_label, prev_counter


def flush(label="misc"):
    """Save any still-open figures (e.g. created but never shown)."""
    global _label, _counter
    _label, _counter = label, 0
    return _save_open_figures()


def saved_files():
    """List of every figure path written so far."""
    return list(_saved)
