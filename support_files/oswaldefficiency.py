from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
B_SPAN_M = 13.0
DESIGN_GAP_M = 2.1
H_MAX_M = 5.0
N_POINTS = 500


def induced_drag_factor(height_m: np.ndarray, span_m: float) -> np.ndarray:
    h_over_b = height_m / span_m
    return (0.44 + 0.9594 * h_over_b) / (0.44 + 2.219 * h_over_b)


def oswald_efficiency(height_m: np.ndarray, span_m: float) -> np.ndarray:
    return 1.0 / induced_drag_factor(height_m, span_m)


def main() -> None:
    h_values = np.linspace(0.0, H_MAX_M, N_POINTS)
    e_values = oswald_efficiency(h_values, B_SPAN_M)
    design_e = float(oswald_efficiency(np.array([DESIGN_GAP_M]), B_SPAN_M)[0])
    design_k = float(induced_drag_factor(np.array([DESIGN_GAP_M]), B_SPAN_M)[0])

    figure, axis = plt.subplots(figsize=(9.5, 5.8))
    axis.plot(
        h_values,
        e_values,
        linewidth=1.8,
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
        label="Rizzo relation",
    )
    axis.axvline(
        DESIGN_GAP_M,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label=fr"Design gap $h = {DESIGN_GAP_M:g}$ m",
    )
    axis.scatter(
        DESIGN_GAP_M,
        design_e,
        marker="D",
        s=60,
        color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
        label=fr"$e = {design_e:.3f}$",
    )

    lower_padding = max(0.05, (design_e - 1.0) * 0.2)
    upper_padding = max(0.08, (max(e_values) - design_e) * 0.2)
    axis.set_ylim(min(e_values) - lower_padding, max(e_values) + upper_padding)

    axis.set_xlabel("Vertical gap h [m]")
    axis.set_ylabel("Oswald efficiency factor e [-]")
    axis.set_title("Vertical Gap Oswald Efficiency")
    axis.grid(True, alpha=0.35)
    axis.legend(frameon=False, loc="best", ncol=1)

    figure.tight_layout()

    png_path = SCRIPT_DIR / "oswaldvsh.png"
    pdf_path = SCRIPT_DIR / "oswaldvsh.pdf"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)

    print(f"h/b = {DESIGN_GAP_M / B_SPAN_M:.4f}")
    print(f"k = {design_k:.4f}")
    print(f"e = {design_e:.4f}")
    print(f"Outputs written to: {png_path} and {pdf_path}")


if __name__ == "__main__":
    main()
