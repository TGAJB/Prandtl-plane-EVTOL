import matplotlib.pyplot as plt
import numpy as np


B_SPAN_M = 13.0
H_LIMIT_M = 2.0
H_MAX_M = 5.0
N_POINTS = 500


def oswald_efficiency(height_m: np.ndarray, span_m: float) -> np.ndarray:
    h_over_b = height_m / span_m
    k = (0.44 + 0.9594 * h_over_b) / (0.44 + 2.219 * h_over_b)
    return 1.0 / k


def main() -> None:
    h_values = np.linspace(0.0, H_MAX_M, N_POINTS)
    e_values = oswald_efficiency(h_values, B_SPAN_M)
    e_limit = oswald_efficiency(np.array([H_LIMIT_M]), B_SPAN_M)[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(h_values, e_values, color="navy", linewidth=2, label="Oswald efficiency factor")
    ax.axvline(
        H_LIMIT_M,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Height constraint: h = {H_LIMIT_M:.1f} m",
    )
    ax.scatter(H_LIMIT_M, e_limit, color="crimson", zorder=3)

    ax.set_xlabel("h [m]")
    ax.set_ylabel("Oswald efficiency factor, e [-]")
    ax.set_title("Oswald Efficiency Factor vs Wing Gap Height")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend()
    ax.annotate(
        f"e = {e_limit:.3f} at h = {H_LIMIT_M:.1f} m",
        xy=(H_LIMIT_M, e_limit),
        xytext=(H_LIMIT_M + 0.2, e_limit + 0.03),
        arrowprops={"arrowstyle": "->", "color": "crimson"},
    )

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
