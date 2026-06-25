"""Plot the lift curve and drag polar from tabulated aerodynamic data."""

from __future__ import annotations


# Write each row from your aerodynamic table here.
# Use `None` for values you do not have yet.
TABLE_DATA = [
    {"alpha_deg": -4.0, "cl": -1.46788e-01, "cd": 1.76510e-02},
    {"alpha_deg": -2.0, "cl": -1.00540e-02, "cd": 1.70934e-02},
    {"alpha_deg": -1.83, "cl": 0.0, "cd": 1.70196e-02},  # Measured point at zero-lift AoA
    {"alpha_deg": 0.0, "cl": 1.23954e-01, "cd": 1.91767e-02},
    {"alpha_deg": 2.6, "cl": 2.95551e-01, "cd": 2.46545e-02},
    {"alpha_deg": 4.0, "cl": 3.84189e-01, "cd": 2.98738e-02},
    {"alpha_deg": 6.0, "cl": 5.07205e-01, "cd": 3.87708e-02},
    {"alpha_deg": 10.0, "cl": 7.15680e-01, "cd": 6.24211e-02},
    {"alpha_deg": 14.0, "cl": 8.82554e-01, "cd": 9.42044e-02},
    {"alpha_deg": 16.0, "cl": 9.43126e-01, "cd": 1.12920e-01},
    {"alpha_deg": 17.0, "cl": 9.70420e-01, "cd": 1.24766e-01},
    {"alpha_deg": 18.0, "cl": 9.84808e-01, "cd": 1.39266e-01},
    {"alpha_deg": 19.0, "cl": 9.91660e-01, "cd": 1.53096e-01},
    {"alpha_deg": 20.0, "cl": 9.46880e-01, "cd": 1.64734e-01},
    {"alpha_deg": 22.0, "cl": 8.11400e-01, "cd": 2.13880e-01},
    # Add more rows below when the table is extended:
    # {"alpha_deg": ..., "cl": ..., "cd": ...},
]

# Use the same alpha window as the lift-curve fit for the drag-polar fit.
DRAG_POLAR_FIT_ALPHA_MIN_DEG = -4.0
DRAG_POLAR_FIT_ALPHA_MAX_DEG = 6.0
DRAG_POLAR_EXCLUDED_ALPHA_DEG: set[float] = set()

# Use only the approximately linear pre-stall region for the lift-curve fit.
LIFT_CURVE_FIT_ALPHA_MIN_DEG = -4.0
LIFT_CURVE_FIT_ALPHA_MAX_DEG = 6.0

CRUISE_MASS_KG = 1980.0
GRAVITY_MPS2 = 9.81
CRUISE_AIR_DENSITY_KG_PER_M3 = 0.836
CRUISE_SPEED_MPS = 55.6
REFERENCE_AREA_M2 = 31.33


def split_data(
    table_data: list[dict[str, float | None]],
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Separate complete and incomplete rows."""
    complete_rows = []
    missing_rows = []

    for row in table_data:
        if row["cl"] is None or row["cd"] is None:
            missing_rows.append(row["alpha_deg"])
        else:
            complete_rows.append((row["alpha_deg"], row["cl"], row["cd"]))

    if not complete_rows:
        raise ValueError("No complete rows found. Add at least one CL/CD data point.")

    alpha_deg, cl_values, cd_values = map(list, zip(*complete_rows))
    return alpha_deg, cl_values, cd_values, missing_rows


def cd_at_zero_lift_from_data(cl_values: list[float], cd_values: list[float]) -> float:
    """Return CD at CL = 0 from a measured point or linear interpolation."""
    cl_cd_pairs = sorted(zip(cl_values, cd_values), key=lambda pair: pair[0])

    for index, (cl_value, cd_value) in enumerate(cl_cd_pairs):
        if cl_value == 0.0:
            return cd_value

        if index == 0:
            continue

        cl_prev, cd_prev = cl_cd_pairs[index - 1]
        if cl_prev <= 0.0 <= cl_value or cl_value <= 0.0 <= cl_prev:
            return cd_prev + (0.0 - cl_prev) * (cd_value - cd_prev) / (cl_value - cl_prev)

    return min(cl_cd_pairs, key=lambda pair: abs(pair[0]))[1]


def select_drag_polar_fit_data(
    alpha_deg: list[float],
    cl_values: list[float],
    cd_values: list[float],
    alpha_min_deg: float,
    alpha_max_deg: float,
    excluded_alpha_deg: set[float] | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """Keep only the selected points for the drag-polar fit."""
    excluded_alpha_deg = excluded_alpha_deg or set()
    fit_rows = [
        (alpha_value, cl_value, cd_value)
        for alpha_value, cl_value, cd_value in zip(alpha_deg, cl_values, cd_values)
        if alpha_min_deg <= alpha_value <= alpha_max_deg
        and all(abs(alpha_value - excluded_alpha) > 1e-9 for excluded_alpha in excluded_alpha_deg)
    ]

    if len(fit_rows) < 3:
        raise ValueError(
            "Need at least three complete points inside the drag-polar fit range. "
            "Add more data or widen the fit limits."
        )

    fit_alpha_deg, fit_cl_values, fit_cd_values = map(list, zip(*fit_rows))
    return fit_alpha_deg, fit_cl_values, fit_cd_values


def select_lift_curve_fit_data(
    alpha_deg: list[float],
    cl_values: list[float],
    alpha_min_deg: float,
    alpha_max_deg: float,
) -> tuple[list[float], list[float]]:
    """Keep only the approximately linear points for the lift-curve fit."""
    fit_rows = [
        (alpha_value, cl_value)
        for alpha_value, cl_value in zip(alpha_deg, cl_values)
        if alpha_min_deg <= alpha_value <= alpha_max_deg
    ]

    if len(fit_rows) < 3:
        raise ValueError(
            "Need at least three complete points inside the lift-curve fit range. "
            "Add more data or widen the fit limits."
        )

    fit_alpha_deg, fit_cl_values = map(list, zip(*fit_rows))
    return fit_alpha_deg, fit_cl_values


def fit_straight_line(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
    """Fit y = slope * x + intercept and return (slope, intercept)."""
    sample_count = len(x_values)
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    sum_xx = sum(x_value**2 for x_value in x_values)
    sum_xy = sum(x_value * y_value for x_value, y_value in zip(x_values, y_values))

    denominator = sample_count * sum_xx - sum_x**2
    if abs(denominator) < 1e-12:
        raise ValueError("Straight-line fit failed because the selected x data do not vary enough.")

    slope = (sample_count * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / sample_count
    return slope, intercept


def fit_lift_curve(alpha_deg: list[float], cl_values: list[float]) -> tuple[float, float, float]:
    """Fit the linear lift curve and derive CL0 and alpha0."""
    lift_curve_slope_per_deg, cl_at_zero_deg = fit_straight_line(alpha_deg, cl_values)
    if abs(lift_curve_slope_per_deg) < 1e-12:
        raise ValueError("Lift-curve fit failed because the slope is too close to zero.")

    alpha_zero_lift_deg = -cl_at_zero_deg / lift_curve_slope_per_deg
    return lift_curve_slope_per_deg, cl_at_zero_deg, alpha_zero_lift_deg


def calculate_cruise_cl(
    mass_kg: float,
    gravity_mps2: float,
    air_density_kg_per_m3: float,
    speed_mps: float,
    reference_area_m2: float,
) -> tuple[float, float, float]:
    """Return cruise CL together with weight and dynamic pressure."""
    if air_density_kg_per_m3 <= 0.0:
        raise ValueError("Cruise air density must be positive.")
    if speed_mps <= 0.0:
        raise ValueError("Cruise speed must be positive.")
    if reference_area_m2 <= 0.0:
        raise ValueError("Reference area must be positive.")

    weight_n = mass_kg * gravity_mps2
    dynamic_pressure_pa = 0.5 * air_density_kg_per_m3 * speed_mps**2
    cruise_cl = weight_n / (dynamic_pressure_pa * reference_area_m2)
    return cruise_cl, weight_n, dynamic_pressure_pa


def alpha_for_cl(cl_value: float, lift_curve_slope_per_deg: float, cl0: float) -> float:
    """Return alpha corresponding to a target CL from the fitted lift curve."""
    if abs(lift_curve_slope_per_deg) < 1e-12:
        raise ValueError("Cannot compute alpha because the lift-curve slope is too close to zero.")

    return (cl_value - cl0) / lift_curve_slope_per_deg


def fit_drag_polar(cl_values: list[float], cd_values: list[float]) -> tuple[float, float]:
    """Fit CD = CD0 + k * CL^2 and return (CD0, k)."""
    x_values = [cl_value**2 for cl_value in cl_values]
    sample_count = len(x_values)

    sum_x = sum(x_values)
    sum_y = sum(cd_values)
    sum_xx = sum(x_value**2 for x_value in x_values)
    sum_xy = sum(x_value * y_value for x_value, y_value in zip(x_values, cd_values))

    denominator = sample_count * sum_xx - sum_x**2
    if abs(denominator) < 1e-12:
        raise ValueError("Drag-polar fit failed because the selected CL data do not vary enough.")

    k_factor = (sample_count * sum_xy - sum_x * sum_y) / denominator
    cd0_fit = (sum_y - k_factor * sum_x) / sample_count
    return cd0_fit, k_factor


def cd_from_drag_polar(cl_value: float, cd0_fit: float, k_factor: float) -> float:
    """Return CD at a target CL from the fitted drag polar."""
    return cd0_fit + k_factor * cl_value**2


def find_cd0_data_point(
    alpha_deg: list[float],
    cl_values: list[float],
    cd_values: list[float],
) -> tuple[float, float, float]:
    """Return the measured data point closest to CL = 0."""
    zero_lift_index = min(range(len(cl_values)), key=lambda index: abs(cl_values[index]))
    return alpha_deg[zero_lift_index], cl_values[zero_lift_index], cd_values[zero_lift_index]


def build_drag_polar_curve(
    cl_min: float,
    cl_max: float,
    cd0_fit: float,
    k_factor: float,
    point_count: int = 200,
) -> tuple[list[float], list[float]]:
    """Create a smooth parabolic drag-polar curve across the fitted CL range."""
    if point_count < 2:
        raise ValueError("point_count must be at least 2.")

    step = (cl_max - cl_min) / (point_count - 1)
    cl_curve = [cl_min + index * step for index in range(point_count)]
    cd_curve = [cd0_fit + k_factor * cl_value**2 for cl_value in cl_curve]
    return cl_curve, cd_curve


def build_straight_line(
    x_min: float,
    x_max: float,
    slope: float,
    intercept: float,
    point_count: int = 200,
) -> tuple[list[float], list[float]]:
    """Create a smooth straight line between x_min and x_max."""
    if point_count < 2:
        raise ValueError("point_count must be at least 2.")

    step = (x_max - x_min) / (point_count - 1)
    x_curve = [x_min + index * step for index in range(point_count)]
    y_curve = [slope * x_value + intercept for x_value in x_curve]
    return x_curve, y_curve


def plot_cl_vs_alpha(
    plt,
    alpha_deg: list[float],
    cl_values: list[float],
    fit_alpha_deg: list[float],
    fit_cl_values: list[float],
    cl_max_idx: int,
    cl0: float,
    lift_curve_slope_per_deg: float,
    alpha_zero_lift_deg: float,
    cruise_alpha_deg: float,
    cruise_cl: float,
) -> None:
    """Plot CL as a function of alpha and show the fitted lift curve."""
    alpha_at_cl_one_deg = (1.0 - cl0) / lift_curve_slope_per_deg
    fit_line_alpha_deg, fit_line_cl_values = build_straight_line(
        min(fit_alpha_deg),
        max(max(fit_alpha_deg), alpha_at_cl_one_deg, cruise_alpha_deg),
        lift_curve_slope_per_deg,
        cl0,
    )
    alpha_zero_lift_sign = "-" if alpha_zero_lift_deg >= 0.0 else "+"
    alpha_zero_lift_magnitude = abs(alpha_zero_lift_deg)

    plt.figure(figsize=(8, 5))
    plt.plot(alpha_deg, cl_values, marker="o", linewidth=1.8, label=r"$C_L$ data")
    plt.scatter(
        fit_alpha_deg,
        fit_cl_values,
        facecolors="none",
        edgecolors="navy",
        s=90,
        linewidths=1.6,
        zorder=3,
        label="Points used for lift-curve fit",
    )
    plt.plot(
        fit_line_alpha_deg,
        fit_line_cl_values,
        color="navy",
        linewidth=2.0,
        label=(
            rf"Fit: $C_L = {lift_curve_slope_per_deg:.4f}"
            rf"(\alpha_{{deg}} {alpha_zero_lift_sign} {alpha_zero_lift_magnitude:.4f})$"
        ),
    )
    plt.scatter(
        alpha_deg[cl_max_idx],
        cl_values[cl_max_idx],
        color="crimson",
        s=80,
        zorder=3,
        label=rf"Estimated $C_{{L,\max}}$ = {cl_values[cl_max_idx]:.4f}",
    )
    plt.annotate(
        rf"$C_{{L,\max}}$ = {cl_values[cl_max_idx]:.4f}"
        + f"\nalpha = {alpha_deg[cl_max_idx]:.1f} deg",
        xy=(alpha_deg[cl_max_idx], cl_values[cl_max_idx]),
        xytext=(12, 12),
        textcoords="offset points",
        bbox={"boxstyle": "round", "fc": "white", "ec": "gray"},
        arrowprops={"arrowstyle": "->", "color": "gray"},
    )
    plt.scatter(0.0, cl0, color="darkgreen", s=70, zorder=3)
    plt.scatter(alpha_zero_lift_deg, 0.0, color="purple", s=70, zorder=3)
    plt.scatter(
        cruise_alpha_deg,
        cruise_cl,
        color="black",
        marker="D",
        s=75,
        zorder=4,
        label=rf"Cruise: $\alpha$ = {cruise_alpha_deg:.2f} deg, $C_L$ = {cruise_cl:.4f}",
    )

    axis = plt.gca()
    axis.relim()
    axis.autoscale_view()
    x_axis_min, _ = axis.get_xlim()
    y_axis_min, _ = axis.get_ylim()
    axis.vlines(
        alpha_zero_lift_deg,
        y_axis_min,
        0.0,
        colors="purple",
        linestyles="--",
        linewidth=1.2,
        alpha=0.7,
        zorder=1,
    )
    axis.hlines(
        cl0,
        x_axis_min,
        0.0,
        colors="darkgreen",
        linestyles="--",
        linewidth=1.2,
        alpha=0.7,
        zorder=1,
    )

    plt.annotate(
        rf"$C_{{L,0}}$ = {cl0:.4f}",
        xy=(0.0, cl0),
        xytext=(18, 14),
        textcoords="offset points",
        bbox={"boxstyle": "round", "fc": "white", "ec": "gray"},
        arrowprops={"arrowstyle": "->", "color": "gray"},
    )
    plt.annotate(
        rf"$\alpha_0$ = {alpha_zero_lift_deg:.2f} deg",
        xy=(alpha_zero_lift_deg, 0.0),
        xytext=(18, -22),
        textcoords="offset points",
        bbox={"boxstyle": "round", "fc": "white", "ec": "gray"},
        arrowprops={"arrowstyle": "->", "color": "gray"},
    )
    plt.annotate(
        rf"Cruise: $\alpha$ = {cruise_alpha_deg:.2f} deg"
        + f"\n$C_L$ = {cruise_cl:.4f}",
        xy=(cruise_alpha_deg, cruise_cl),
        xytext=(52, -26),
        textcoords="offset points",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.9", "fc": "white", "ec": "gray"},
        arrowprops={"arrowstyle": "->", "color": "gray"},
    )
    plt.xlabel(r"$\alpha$ [deg]")
    plt.ylabel(r"$C_L$")
    plt.title(r"$C_L$ vs $\alpha$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()


def plot_drag_polar(
    plt,
    cl_values: list[float],
    cd_values: list[float],
    fit_cl_values: list[float],
    fit_cd_values: list[float],
    cd0_data: float,
    k_factor: float,
    cd0_fit: float,
    cruise_cl: float,
    cruise_cd: float,
    cruise_alpha_deg: float,
) -> None:
    """Plot drag polar, fitted curve, and the measured CD0 point."""
    cl_curve, cd_curve = build_drag_polar_curve(
        min(min(fit_cl_values), cruise_cl),
        max(max(fit_cl_values), cruise_cl),
        cd0_fit,
        k_factor,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(cd_values, cl_values, marker="o", linewidth=1.8, label="All drag polar data")
    plt.scatter(
        fit_cd_values,
        fit_cl_values,
        facecolors="none",
        edgecolors="darkorange",
        s=90,
        linewidths=1.6,
        zorder=3,
        label="Points used for drag-polar fit",
    )
    plt.plot(
        cd_curve,
        cl_curve,
        color="darkorange",
        linewidth=2.0,
        label=rf"Fit: $C_D = {cd0_fit:.5f} + {k_factor:.5f} C_L^2$",
    )
    plt.scatter(
        cd0_data,
        0.0,
        color="darkgreen",
        s=80,
        zorder=3,
        label=rf"Measured $C_{{D,0}}$ = {cd0_data:.5f}",
    )
    plt.scatter(
        cruise_cd,
        cruise_cl,
        color="black",
        marker="D",
        s=75,
        zorder=4,
        label=rf"Cruise: $C_D$ = {cruise_cd:.5f}, $C_L$ = {cruise_cl:.4f}",
    )

    axis = plt.gca()
    axis.relim()
    axis.autoscale_view()
    y_axis_min, _ = axis.get_ylim()
    axis.vlines(
        cd0_data,
        y_axis_min,
        0.0,
        colors="darkgreen",
        linestyles="--",
        linewidth=1.2,
        alpha=0.7,
        zorder=1,
    )

    plt.annotate(
        rf"$C_{{D,0}}$ = {cd0_data:.5f}",
        xy=(cd0_data, 0.0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox={"boxstyle": "round", "fc": "white", "ec": "gray"},
        arrowprops={"arrowstyle": "->", "color": "gray"},
    )
    plt.annotate(
        rf"Cruise: $\alpha$ = {cruise_alpha_deg:.2f} deg"
        + f"\n$C_D$ = {cruise_cd:.5f}, $C_L$ = {cruise_cl:.4f}",
        xy=(cruise_cd, cruise_cl),
        xytext=(28, -58),
        textcoords="offset points",
        bbox={"boxstyle": "round", "fc": "white", "ec": "gray"},
        arrowprops={"arrowstyle": "->", "color": "gray"},
    )
    plt.xlabel(r"$C_D$")
    plt.ylabel(r"$C_L$")
    plt.title("Drag polar")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()


def main() -> None:
    alpha_deg, cl_values, cd_values, missing_alpha_deg = split_data(TABLE_DATA)

    cl_max_idx = max(range(len(cl_values)), key=lambda index: cl_values[index])
    cl_max = cl_values[cl_max_idx]
    alpha_at_cl_max = alpha_deg[cl_max_idx]

    lift_fit_alpha_deg, lift_fit_cl_values = select_lift_curve_fit_data(
        alpha_deg,
        cl_values,
        LIFT_CURVE_FIT_ALPHA_MIN_DEG,
        LIFT_CURVE_FIT_ALPHA_MAX_DEG,
    )
    lift_curve_slope_per_deg, cl0, alpha_zero_lift_deg = fit_lift_curve(
        lift_fit_alpha_deg,
        lift_fit_cl_values,
    )
    lift_curve_slope_per_rad = lift_curve_slope_per_deg * 180.0 / 3.141592653589793
    cruise_cl, cruise_weight_n, cruise_dynamic_pressure_pa = calculate_cruise_cl(
        CRUISE_MASS_KG,
        GRAVITY_MPS2,
        CRUISE_AIR_DENSITY_KG_PER_M3,
        CRUISE_SPEED_MPS,
        REFERENCE_AREA_M2,
    )
    cruise_alpha_deg = alpha_for_cl(cruise_cl, lift_curve_slope_per_deg, cl0)

    cd_at_zero_lift = cd_at_zero_lift_from_data(cl_values, cd_values)
    drag_fit_alpha_deg, drag_fit_cl_values, drag_fit_cd_values = select_drag_polar_fit_data(
        alpha_deg,
        cl_values,
        cd_values,
        DRAG_POLAR_FIT_ALPHA_MIN_DEG,
        DRAG_POLAR_FIT_ALPHA_MAX_DEG,
        DRAG_POLAR_EXCLUDED_ALPHA_DEG,
    )
    cd0_fit, k_factor = fit_drag_polar(drag_fit_cl_values, drag_fit_cd_values)
    cruise_cd = cd_from_drag_polar(cruise_cl, cd0_fit, k_factor)
    alpha_cd0_data, cl_cd0_data, cd0_data = find_cd0_data_point(alpha_deg, cl_values, cd_values)

    print(f"Estimated C_Lmax = {cl_max:.5f} at alpha = {alpha_at_cl_max:.2f} deg")
    print(
        "Fitted lift curve using alpha in "
        f"[{LIFT_CURVE_FIT_ALPHA_MIN_DEG:.1f}, {LIFT_CURVE_FIT_ALPHA_MAX_DEG:.1f}] deg:"
    )
    print("Fit points alpha [deg] = " + ", ".join(f"{alpha:.1f}" for alpha in lift_fit_alpha_deg))
    print(f"    C_L = {cl0:.5f} + {lift_curve_slope_per_deg:.5f} * alpha_deg")
    if alpha_zero_lift_deg >= 0.0:
        print(
            "    Equivalent zero-lift form: "
            f"C_L = {lift_curve_slope_per_deg:.5f} * (alpha_deg - {alpha_zero_lift_deg:.5f})"
        )
    else:
        print(
            "    Equivalent zero-lift form: "
            f"C_L = {lift_curve_slope_per_deg:.5f} * (alpha_deg + {abs(alpha_zero_lift_deg):.5f})"
        )
    print(f"Estimated C_L0 from lift-curve fit = {cl0:.5f}")
    print(
        "Estimated lift-curve slope a = "
        f"{lift_curve_slope_per_deg:.5f} per deg = {lift_curve_slope_per_rad:.5f} per rad"
    )
    print(f"Estimated alpha_0 from lift-curve fit = {alpha_zero_lift_deg:.5f} deg")
    print("Cruise condition:")
    print(
        f"    Weight = m * g = {CRUISE_MASS_KG:.1f} * {GRAVITY_MPS2:.2f} = "
        f"{cruise_weight_n:.2f} N"
    )
    print(
        f"    Dynamic pressure q = 0.5 * rho * V^2 = 0.5 * {CRUISE_AIR_DENSITY_KG_PER_M3:.3f} "
        f"* {CRUISE_SPEED_MPS:.1f}^2 = {cruise_dynamic_pressure_pa:.2f} Pa"
    )
    print(
        f"    Cruise C_L = W / (q * S) = {cruise_weight_n:.2f} / "
        f"({cruise_dynamic_pressure_pa:.2f} * {REFERENCE_AREA_M2:.2f}) = {cruise_cl:.5f}"
    )
    print(
        "    Cruise alpha from fitted lift curve = "
        f"({cruise_cl:.5f} - {cl0:.5f}) / {lift_curve_slope_per_deg:.5f} = "
        f"{cruise_alpha_deg:.5f} deg"
    )
    print(f"CD at CL = 0 from data/interpolation = {cd_at_zero_lift:.5f}")
    print(
        "Fitted parabolic drag polar using alpha in "
        f"[{DRAG_POLAR_FIT_ALPHA_MIN_DEG:.1f}, {DRAG_POLAR_FIT_ALPHA_MAX_DEG:.1f}] deg:"
    )
    if DRAG_POLAR_EXCLUDED_ALPHA_DEG:
        excluded_alpha_text = ", ".join(
            f"{alpha:.2f}" for alpha in sorted(DRAG_POLAR_EXCLUDED_ALPHA_DEG)
        )
        print(f"Excluded alpha [deg] from drag-polar fit: {excluded_alpha_text}")
    print("Fit points alpha [deg] = " + ", ".join(f"{alpha:.1f}" for alpha in drag_fit_alpha_deg))
    print(f"    C_D = {cd0_fit:.5f} + {k_factor:.5f} * C_L^2")
    print(f"    Cruise C_D from parabolic fit = {cruise_cd:.5f}")
    print(
        "Measured C_D0 data point = "
        f"{cd0_data:.5f} at alpha = {alpha_cd0_data:.2f} deg and C_L = {cl_cd0_data:.5f}"
    )
    print(f"Parabolic drag-polar fit intercept = {cd0_fit:.5f}")
    if missing_alpha_deg:
        missing_text = ", ".join(f"{alpha:.1f}" for alpha in missing_alpha_deg)
        print(f"Rows still missing CL/CD values at alpha [deg]: {missing_text}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required to show the plots. Install it and rerun.") from exc

    plot_cl_vs_alpha(
        plt,
        alpha_deg,
        cl_values,
        lift_fit_alpha_deg,
        lift_fit_cl_values,
        cl_max_idx,
        cl0,
        lift_curve_slope_per_deg,
        alpha_zero_lift_deg,
        cruise_alpha_deg,
        cruise_cl,
    )
    plot_drag_polar(
        plt,
        cl_values,
        cd_values,
        drag_fit_cl_values,
        drag_fit_cd_values,
        cd0_data,
        k_factor,
        cd0_fit,
        cruise_cl,
        cruise_cd,
        cruise_alpha_deg,
    )
    plt.show()


if __name__ == "__main__":
    main()
