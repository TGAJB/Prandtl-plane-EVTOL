"""
Validate the Flow5 polar for the NASA/Langley LS(1)-0417 airfoil against
the AirfoilTools XFOIL polar.

Outputs:
- overlay plots for CL, CD, CM, and drag polar
- difference plots for Flow5 - AirfoilTools
- transition-location comparison plots
- CSV files with point-by-point comparisons and summary metrics
- a Markdown summary table that can be copied into the report

The script assumes both polars are for Re = 1e6, Mach = 0, Ncrit = 9.
Run from the repository root with:

    python final_characteristics/aerodynamics/airfoil_characteristics.py
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "airfoil_data"
DEFAULT_AIRFOILTOOLS_PATH = DATA_DIR / "ls10417_airfoiltools_Re1e6_N9.txt"
DEFAULT_FLOW5_PATH = DATA_DIR / "ls10417_flow5_Re1e6_N9.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "airfoil_validation_results"

REFERENCE_LABEL = "AirfoilTools XFOIL 6.96"
CANDIDATE_LABEL = "Flow5 v7.56"
COMMON_COLUMNS = ("CL", "CD", "CDp", "Cm", "Top_Xtr", "Bot_Xtr")
KEY_ALPHAS_DEG = (-4.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0)


@dataclass(frozen=True)
class PolarData:
    label: str
    path: Path
    metadata: dict[str, object]
    columns: dict[str, np.ndarray]

    @property
    def alpha(self) -> np.ndarray:
        return self.columns["alpha"]

    def has(self, column: str) -> bool:
        return column in self.columns

    def values(self, column: str) -> np.ndarray:
        return self.columns[column]


def _is_separator_line(line: str) -> bool:
    stripped = line.strip()
    return len(stripped) > 8 and set(stripped) <= {"-", " "}


def _try_parse_float_tokens(line: str) -> list[float] | None:
    stripped = line.strip()
    if not stripped:
        return None

    tokens = stripped.split()
    values = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError:
            return None

    return values if len(values) >= 5 else None


def _column_names_for_row_width(row_width: int) -> list[str]:
    base_columns = ["alpha", "CL", "CD", "CDp", "Cm"]

    if row_width == 7:
        return base_columns + ["Top_Xtr", "Bot_Xtr"]

    if row_width == 10:
        return base_columns + ["Top_Xtr", "Bot_Xtr", "Cpmin", "Chinge", "XCp"]

    if row_width > len(base_columns):
        extra_columns = [f"extra_{index}" for index in range(1, row_width - len(base_columns) + 1)]
        return base_columns + extra_columns

    raise ValueError(f"Cannot parse polar table with only {row_width} columns.")


def _parse_metadata(lines: Iterable[str]) -> dict[str, object]:
    metadata: dict[str, object] = {}

    for line in lines:
        stripped = line.strip()

        if not stripped:
            continue

        if "XFOIL" in stripped.upper() or stripped.lower().startswith("flow5"):
            metadata["solver_version"] = stripped

        if "Calculated polar for:" in line:
            metadata["airfoil"] = line.split("Calculated polar for:", 1)[1].strip()

        reynolds_match = re.search(
            r"Mach\s*=\s*([+-]?\d+(?:\.\d+)?)\s+"
            r"Re\s*=\s*([+-]?\d+(?:\.\d+)?)\s*e\s*([+-]?\d+)\s+"
            r"Ncrit\s*=\s*([+-]?\d+(?:\.\d+)?)",
            line,
            flags=re.IGNORECASE,
        )
        if reynolds_match:
            metadata["mach"] = float(reynolds_match.group(1))
            metadata["reynolds"] = float(reynolds_match.group(2)) * 10 ** int(
                reynolds_match.group(3)
            )
            metadata["ncrit"] = float(reynolds_match.group(4))

    return metadata


def load_xfoil_polar(path: Path, label: str) -> PolarData:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    metadata = _parse_metadata(lines)
    rows: list[list[float]] = []
    table_started = False

    for line in lines:
        if _is_separator_line(line):
            table_started = True
            continue

        if not table_started:
            continue

        parsed_values = _try_parse_float_tokens(line)
        if parsed_values is None:
            if rows:
                break
            continue

        rows.append(parsed_values)

    if not rows:
        raise ValueError(f"No polar data rows could be parsed from {path}")

    row_widths = {len(row) for row in rows}
    if len(row_widths) != 1:
        raise ValueError(f"Rows in {path} have inconsistent widths: {sorted(row_widths)}")

    data = np.array(rows, dtype=float)
    sort_index = np.argsort(data[:, 0])
    data = data[sort_index]

    _, unique_index = np.unique(data[:, 0], return_index=True)
    data = data[np.sort(unique_index)]

    column_names = _column_names_for_row_width(data.shape[1])
    columns = {
        column_name: data[:, column_index]
        for column_index, column_name in enumerate(column_names)
    }

    return PolarData(label=label, path=path, metadata=metadata, columns=columns)


def _interp(polar: PolarData, column: str, alpha_deg: np.ndarray) -> np.ndarray:
    return np.interp(alpha_deg, polar.alpha, polar.values(column))


def _alpha_overlap(first: PolarData, second: PolarData) -> tuple[float, float]:
    lower = max(float(first.alpha.min()), float(second.alpha.min()))
    upper = min(float(first.alpha.max()), float(second.alpha.max()))

    if lower >= upper:
        raise ValueError(
            f"No overlapping angle-of-attack range between {first.label} and {second.label}."
        )

    return lower, upper


def build_point_comparison(
    reference: PolarData,
    candidate: PolarData,
    columns: Iterable[str] = COMMON_COLUMNS,
) -> tuple[list[dict[str, float]], tuple[float, float]]:
    overlap = _alpha_overlap(reference, candidate)
    alpha_mask = (reference.alpha >= overlap[0]) & (reference.alpha <= overlap[1])
    comparison_alpha = reference.alpha[alpha_mask]
    rows = []

    for index, alpha in enumerate(comparison_alpha):
        row: dict[str, float] = {"alpha_deg": float(alpha)}

        for column in columns:
            if not (reference.has(column) and candidate.has(column)):
                continue

            reference_value = float(reference.values(column)[alpha_mask][index])
            candidate_value = float(_interp(candidate, column, np.array([alpha]))[0])
            difference = candidate_value - reference_value

            row[f"{column}_airfoiltools"] = reference_value
            row[f"{column}_flow5_interp"] = candidate_value
            row[f"{column}_diff_flow5_minus_airfoiltools"] = difference

            if column in {"CD", "CDp"}:
                row[f"{column}_diff_drag_counts"] = difference * 10000.0

        rows.append(row)

    return rows, overlap


def _error_metrics(reference_values: np.ndarray, candidate_values: np.ndarray) -> dict[str, float]:
    difference = candidate_values - reference_values
    abs_difference = np.abs(difference)
    reference_range = float(reference_values.max() - reference_values.min())
    reference_scale = float(np.mean(np.abs(reference_values)))
    valid_percentage_mask = np.abs(reference_values) > 1e-8

    metrics = {
        "n_points": float(len(reference_values)),
        "bias": float(np.mean(difference)),
        "mae": float(np.mean(abs_difference)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "max_abs": float(np.max(abs_difference)),
        "max_abs_index": float(np.argmax(abs_difference)),
        "reference_range_percent_mae": math.nan,
        "mean_abs_percentage_error": math.nan,
        "r_squared": math.nan,
    }

    if reference_range > 0.0:
        metrics["reference_range_percent_mae"] = metrics["mae"] / reference_range * 100.0

    if reference_scale > 0.0 and np.any(valid_percentage_mask):
        metrics["mean_abs_percentage_error"] = float(
            np.mean(np.abs(difference[valid_percentage_mask] / reference_values[valid_percentage_mask]))
            * 100.0
        )

    residual_sum = float(np.sum(difference**2))
    total_sum = float(np.sum((reference_values - np.mean(reference_values)) ** 2))
    if total_sum > 0.0:
        metrics["r_squared"] = 1.0 - residual_sum / total_sum

    return metrics


def build_error_summary(
    reference: PolarData,
    candidate: PolarData,
    overlap: tuple[float, float],
    columns: Iterable[str] = COMMON_COLUMNS,
) -> list[dict[str, float | str]]:
    alpha_mask = (reference.alpha >= overlap[0]) & (reference.alpha <= overlap[1])
    comparison_alpha = reference.alpha[alpha_mask]
    rows: list[dict[str, float | str]] = []

    for column in columns:
        if not (reference.has(column) and candidate.has(column)):
            continue

        reference_values = reference.values(column)[alpha_mask]
        candidate_values = _interp(candidate, column, comparison_alpha)
        metrics = _error_metrics(reference_values, candidate_values)
        max_abs_index = int(metrics.pop("max_abs_index"))

        row: dict[str, float | str] = {
            "coefficient": column,
            "max_abs_alpha_deg": float(comparison_alpha[max_abs_index]),
            **metrics,
        }

        if column in {"CD", "CDp"}:
            row["bias_drag_counts"] = float(row["bias"]) * 10000.0
            row["mae_drag_counts"] = float(row["mae"]) * 10000.0
            row["rmse_drag_counts"] = float(row["rmse"]) * 10000.0
            row["max_abs_drag_counts"] = float(row["max_abs"]) * 10000.0

        rows.append(row)

    return rows


def _value_at_alpha(polar: PolarData, column: str, alpha_deg: float) -> float:
    if alpha_deg < polar.alpha.min() or alpha_deg > polar.alpha.max():
        return math.nan
    return float(_interp(polar, column, np.array([alpha_deg]))[0])


def _linear_lift_fit(
    polar: PolarData,
    linear_alpha_min: float,
    linear_alpha_max: float,
) -> tuple[float, float, float]:
    mask = (polar.alpha >= linear_alpha_min) & (polar.alpha <= linear_alpha_max)
    if np.count_nonzero(mask) < 2:
        raise ValueError(
            f"{polar.label} has fewer than two points in the requested linear region "
            f"{linear_alpha_min:g} to {linear_alpha_max:g} deg."
        )

    slope_per_deg, intercept = np.polyfit(polar.alpha[mask], polar.values("CL")[mask], 1)
    zero_lift_alpha = -intercept / slope_per_deg
    return float(slope_per_deg), float(slope_per_deg * 180.0 / math.pi), float(zero_lift_alpha)


def aerodynamic_characteristics(
    polar: PolarData,
    linear_alpha_min: float,
    linear_alpha_max: float,
) -> dict[str, float]:
    alpha = polar.alpha
    cl = polar.values("CL")
    cd = polar.values("CD")
    cm = polar.values("Cm")
    lift_to_drag = np.divide(cl, cd, out=np.full_like(cl, math.nan), where=cd > 0.0)

    cl_max_index = int(np.argmax(cl))
    cl_min_index = int(np.argmin(cl))
    cd_min_index = int(np.argmin(cd))
    ld_max_index = int(np.nanargmax(lift_to_drag))
    slope_per_deg, slope_per_rad, zero_lift_alpha = _linear_lift_fit(
        polar,
        linear_alpha_min,
        linear_alpha_max,
    )

    return {
        "n_points": float(len(alpha)),
        "alpha_min_deg": float(alpha.min()),
        "alpha_max_deg": float(alpha.max()),
        "CL_at_0deg": _value_at_alpha(polar, "CL", 0.0),
        "CD_at_0deg": _value_at_alpha(polar, "CD", 0.0),
        "Cm_at_0deg": _value_at_alpha(polar, "Cm", 0.0),
        "CL_max": float(cl[cl_max_index]),
        "alpha_at_CL_max_deg": float(alpha[cl_max_index]),
        "CL_min": float(cl[cl_min_index]),
        "alpha_at_CL_min_deg": float(alpha[cl_min_index]),
        "CD_min": float(cd[cd_min_index]),
        "alpha_at_CD_min_deg": float(alpha[cd_min_index]),
        "LD_max": float(lift_to_drag[ld_max_index]),
        "alpha_at_LD_max_deg": float(alpha[ld_max_index]),
        "CL_at_LD_max": float(cl[ld_max_index]),
        "CD_at_LD_max": float(cd[ld_max_index]),
        "lift_curve_slope_per_deg": slope_per_deg,
        "lift_curve_slope_per_rad": slope_per_rad,
        "alpha_zero_lift_deg": zero_lift_alpha,
    }


def build_characteristics_summary(
    reference: PolarData,
    candidate: PolarData,
    linear_alpha_min: float,
    linear_alpha_max: float,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, float | str]]]:
    reference_metrics = aerodynamic_characteristics(reference, linear_alpha_min, linear_alpha_max)
    candidate_metrics = aerodynamic_characteristics(candidate, linear_alpha_min, linear_alpha_max)
    rows: list[dict[str, float | str]] = []

    for metric in reference_metrics:
        reference_value = reference_metrics[metric]
        candidate_value = candidate_metrics[metric]
        rows.append(
            {
                "metric": metric,
                "airfoiltools": reference_value,
                "flow5": candidate_value,
                "diff_flow5_minus_airfoiltools": candidate_value - reference_value,
            }
        )

    return reference_metrics, candidate_metrics, rows


def build_key_alpha_rows(
    reference: PolarData,
    candidate: PolarData,
    key_alphas_deg: Iterable[float] = KEY_ALPHAS_DEG,
) -> list[dict[str, float]]:
    overlap = _alpha_overlap(reference, candidate)
    rows = []

    for alpha in key_alphas_deg:
        if alpha < overlap[0] or alpha > overlap[1]:
            continue

        row = {"alpha_deg": float(alpha)}
        for column in ("CL", "CD", "Cm"):
            reference_value = _value_at_alpha(reference, column, alpha)
            candidate_value = _value_at_alpha(candidate, column, alpha)
            row[f"{column}_airfoiltools"] = reference_value
            row[f"{column}_flow5"] = candidate_value
            row[f"{column}_diff_flow5_minus_airfoiltools"] = candidate_value - reference_value

            if column == "CD":
                row["CD_diff_drag_counts"] = (candidate_value - reference_value) * 10000.0

        rows.append(row)

    return rows


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: object, decimals: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return ""
        return f"{float(value):.{decimals}f}"

    return str(value)


def _markdown_table(headers: list[str], rows: list[list[object]], decimals: int = 4) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = [
        "| " + " | ".join(_format_float(value, decimals) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *body_lines])


def write_markdown_summary(
    path: Path,
    reference: PolarData,
    candidate: PolarData,
    overlap: tuple[float, float],
    characteristic_rows: list[dict[str, float | str]],
    error_rows: list[dict[str, float | str]],
    key_alpha_rows: list[dict[str, float]],
    linear_alpha_min: float,
    linear_alpha_max: float,
) -> None:
    selected_characteristics = [
        "CL_at_0deg",
        "CD_at_0deg",
        "Cm_at_0deg",
        "CL_max",
        "alpha_at_CL_max_deg",
        "CD_min",
        "alpha_at_CD_min_deg",
        "LD_max",
        "alpha_at_LD_max_deg",
        "lift_curve_slope_per_deg",
        "alpha_zero_lift_deg",
    ]
    characteristic_lookup = {str(row["metric"]): row for row in characteristic_rows}
    characteristic_table_rows = [
        [
            metric,
            characteristic_lookup[metric]["airfoiltools"],
            characteristic_lookup[metric]["flow5"],
            characteristic_lookup[metric]["diff_flow5_minus_airfoiltools"],
        ]
        for metric in selected_characteristics
    ]

    error_table_rows = []
    for row in error_rows:
        error_table_rows.append(
            [
                row["coefficient"],
                row["mae"],
                row["rmse"],
                row["max_abs"],
                row["max_abs_alpha_deg"],
                row["reference_range_percent_mae"],
            ]
        )

    key_table_rows = []
    for row in key_alpha_rows:
        key_table_rows.append(
            [
                row["alpha_deg"],
                row["CL_diff_flow5_minus_airfoiltools"],
                row["CD_diff_drag_counts"],
                row["Cm_diff_flow5_minus_airfoiltools"],
            ]
        )

    metadata_rows = [
        ["Reference", reference.label],
        ["Candidate", candidate.label],
        ["Reference file", reference.path.name],
        ["Candidate file", candidate.path.name],
        ["Airfoil", reference.metadata.get("airfoil", "")],
        ["Reynolds number", reference.metadata.get("reynolds", "")],
        ["Mach", reference.metadata.get("mach", "")],
        ["Ncrit", reference.metadata.get("ncrit", "")],
        ["Comparison alpha range", f"{overlap[0]:.2f} to {overlap[1]:.2f} deg"],
        ["Linear fit alpha range", f"{linear_alpha_min:.2f} to {linear_alpha_max:.2f} deg"],
    ]

    content = "\n\n".join(
        [
            "# NASA/Langley LS(1)-0417 Airfoil Validation Summary",
            (
                "AirfoilTools is treated as the reference dataset. Flow5 values are "
                "linearly interpolated onto the AirfoilTools angle-of-attack grid "
                "inside the overlapping alpha range."
            ),
            "## Dataset metadata\n\n" + _markdown_table(["Item", "Value"], metadata_rows),
            "## Aerodynamic characteristics\n\n"
            + _markdown_table(
                ["Metric", "AirfoilTools", "Flow5", "Flow5 - AirfoilTools"],
                characteristic_table_rows,
                decimals=5,
            ),
            "## Pointwise error summary\n\n"
            + _markdown_table(
                [
                    "Coeff.",
                    "MAE",
                    "RMSE",
                    "Max abs.",
                    "Max abs. alpha [deg]",
                    "MAE / ref. range [%]",
                ],
                error_table_rows,
                decimals=5,
            ),
            "## Key alpha differences\n\n"
            + _markdown_table(
                [
                    "Alpha [deg]",
                    "Delta CL",
                    "Delta CD [counts]",
                    "Delta Cm",
                ],
                key_table_rows,
            ),
            (
                "Note: this is a validation against another XFOIL-based reference "
                "calculation, not against wind-tunnel data. Agreement should be "
                "described as code-to-reference consistency."
            ),
            "",
        ]
    )

    path.write_text(content, encoding="utf-8")


def _setup_axis(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35)


def plot_overlays(reference: PolarData, candidate: PolarData, output_dir: Path) -> list[Path]:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].plot(reference.alpha, reference.values("CL"), label=reference.label)
    axes[0].plot(candidate.alpha, candidate.values("CL"), "--", label=candidate.label)
    _setup_axis(axes[0], r"Angle of attack $\alpha$ [deg]", r"$C_L$ [-]", "Lift curve")

    axes[1].plot(reference.alpha, reference.values("CD"), label=reference.label)
    axes[1].plot(candidate.alpha, candidate.values("CD"), "--", label=candidate.label)
    _setup_axis(axes[1], r"Angle of attack $\alpha$ [deg]", r"$C_D$ [-]", "Drag curve")

    axes[2].plot(reference.alpha, reference.values("Cm"), label=reference.label)
    axes[2].plot(candidate.alpha, candidate.values("Cm"), "--", label=candidate.label)
    _setup_axis(axes[2], r"Angle of attack $\alpha$ [deg]", r"$C_m$ [-]", "Moment curve")

    axes[3].plot(reference.values("CD"), reference.values("CL"), label=reference.label)
    axes[3].plot(candidate.values("CD"), candidate.values("CL"), "--", label=candidate.label)
    _setup_axis(axes[3], r"$C_D$ [-]", r"$C_L$ [-]", "Drag polar")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle("NASA/Langley LS(1)-0417 Airfoil Polar Comparison", y=0.99)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    png_path = output_dir / "airfoil_validation_overlays.png"
    pdf_path = output_dir / "airfoil_validation_overlays.pdf"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)
    return [png_path, pdf_path]


def _relative_difference_percent(
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    min_reference_abs: float = 1e-8,
) -> np.ndarray:
    difference = candidate_values - reference_values
    valid = np.abs(reference_values) >= min_reference_abs
    percent = np.full_like(reference_values, math.nan, dtype=float)
    percent[valid] = difference[valid] / reference_values[valid] * 100.0
    return percent


def plot_differences(
    reference: PolarData,
    candidate: PolarData,
    overlap: tuple[float, float],
    output_dir: Path,
) -> list[Path]:
    alpha_mask = (reference.alpha >= overlap[0]) & (reference.alpha <= overlap[1])
    alpha = reference.alpha[alpha_mask]
    reference_cl = reference.values("CL")[alpha_mask]
    delta_cl_percent = _relative_difference_percent(
        reference_cl,
        _interp(candidate, "CL", alpha),
        min_reference_abs=0.02,
    )

    figure, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(alpha, delta_cl_percent, color="tab:blue")
    ax.axhline(0.0, color="black", linewidth=0.8)
    _setup_axis(
        ax,
        r"Angle of attack $\alpha$ [deg]",
        r"Relative $\Delta C_L$ [%]",
        "",
    )
    ax.text(
        0.01,
        0.90,
        r"Near-zero $C_L$ reference points omitted",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )

    figure.tight_layout()

    png_path = output_dir / "airfoil_validation_differences.png"
    pdf_path = output_dir / "airfoil_validation_differences.pdf"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)
    return [png_path, pdf_path]


def plot_transition_locations(
    reference: PolarData,
    candidate: PolarData,
    output_dir: Path,
) -> list[Path]:
    if not all(
        polar.has(column)
        for polar in (reference, candidate)
        for column in ("Top_Xtr", "Bot_Xtr")
    ):
        return []

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(reference.alpha, reference.values("Top_Xtr"), label=reference.label)
    axes[0].plot(candidate.alpha, candidate.values("Top_Xtr"), "--", label=candidate.label)
    _setup_axis(axes[0], "", r"Top transition $x/c$ [-]", "Upper-surface transition")
    axes[0].legend(frameon=False)

    axes[1].plot(reference.alpha, reference.values("Bot_Xtr"), label=reference.label)
    axes[1].plot(candidate.alpha, candidate.values("Bot_Xtr"), "--", label=candidate.label)
    _setup_axis(
        axes[1],
        r"Angle of attack $\alpha$ [deg]",
        r"Bottom transition $x/c$ [-]",
        "Lower-surface transition",
    )

    figure.suptitle("Boundary-Layer Transition Location Comparison")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    png_path = output_dir / "airfoil_validation_transition.png"
    pdf_path = output_dir / "airfoil_validation_transition.pdf"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    plt.close(figure)
    return [png_path, pdf_path]


def print_console_summary(
    output_dir: Path,
    overlap: tuple[float, float],
    characteristic_rows: list[dict[str, float | str]],
    error_rows: list[dict[str, float | str]],
) -> None:
    characteristics = {str(row["metric"]): row for row in characteristic_rows}
    errors = {str(row["coefficient"]): row for row in error_rows}

    print("NASA/Langley LS(1)-0417 validation")
    print(f"Comparison alpha range: {overlap[0]:.2f} to {overlap[1]:.2f} deg")
    print(
        "CLmax: "
        f"AirfoilTools {characteristics['CL_max']['airfoiltools']:.4f}, "
        f"Flow5 {characteristics['CL_max']['flow5']:.4f}, "
        f"delta {characteristics['CL_max']['diff_flow5_minus_airfoiltools']:+.4f}"
    )
    print(
        "CDmin: "
        f"AirfoilTools {characteristics['CD_min']['airfoiltools']:.5f}, "
        f"Flow5 {characteristics['CD_min']['flow5']:.5f}, "
        f"delta {characteristics['CD_min']['diff_flow5_minus_airfoiltools']:+.5f}"
    )
    print(
        "Pointwise MAE: "
        f"CL {errors['CL']['mae']:.5f}, "
        f"CD {errors['CD']['mae_drag_counts']:.2f} drag counts, "
        f"Cm {errors['Cm']['mae']:.5f}"
    )
    print(f"Outputs written to: {output_dir}")


def run_validation(
    airfoiltools_path: Path,
    flow5_path: Path,
    output_dir: Path,
    linear_alpha_min: float,
    linear_alpha_max: float,
) -> None:
    if not airfoiltools_path.exists():
        raise FileNotFoundError(f"AirfoilTools polar file not found: {airfoiltools_path}")

    if not flow5_path.exists():
        raise FileNotFoundError(f"Flow5 polar file not found: {flow5_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_xfoil_polar(airfoiltools_path, REFERENCE_LABEL)
    candidate = load_xfoil_polar(flow5_path, CANDIDATE_LABEL)

    comparison_rows, overlap = build_point_comparison(reference, candidate)
    error_rows = build_error_summary(reference, candidate, overlap)
    _, _, characteristic_rows = build_characteristics_summary(
        reference,
        candidate,
        linear_alpha_min,
        linear_alpha_max,
    )
    key_alpha_rows = build_key_alpha_rows(reference, candidate)

    _write_dict_rows(output_dir / "airfoil_validation_point_comparison.csv", comparison_rows)
    _write_dict_rows(output_dir / "airfoil_validation_error_metrics.csv", error_rows)
    _write_dict_rows(output_dir / "airfoil_validation_characteristics.csv", characteristic_rows)
    _write_dict_rows(output_dir / "airfoil_validation_key_alphas.csv", key_alpha_rows)
    write_markdown_summary(
        output_dir / "airfoil_validation_summary.md",
        reference,
        candidate,
        overlap,
        characteristic_rows,
        error_rows,
        key_alpha_rows,
        linear_alpha_min,
        linear_alpha_max,
    )

    plot_overlays(reference, candidate, output_dir)
    plot_differences(reference, candidate, overlap, output_dir)
    plot_transition_locations(reference, candidate, output_dir)

    print_console_summary(output_dir, overlap, characteristic_rows, error_rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Flow5 NASA/Langley LS(1)-0417 airfoil polar against "
            "the AirfoilTools XFOIL polar."
        )
    )
    parser.add_argument(
        "--airfoiltools",
        type=Path,
        default=DEFAULT_AIRFOILTOOLS_PATH,
        help="Path to the AirfoilTools XFOIL polar text export.",
    )
    parser.add_argument(
        "--flow5",
        type=Path,
        default=DEFAULT_FLOW5_PATH,
        help="Path to the Flow5 polar text export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated plots and CSV/Markdown summaries.",
    )
    parser.add_argument(
        "--linear-alpha-min",
        type=float,
        default=-2.0,
        help="Lower alpha bound for the linear lift-curve fit, in degrees.",
    )
    parser.add_argument(
        "--linear-alpha-max",
        type=float,
        default=6.0,
        help="Upper alpha bound for the linear lift-curve fit, in degrees.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    run_validation(
        airfoiltools_path=arguments.airfoiltools,
        flow5_path=arguments.flow5,
        output_dir=arguments.output_dir,
        linear_alpha_min=arguments.linear_alpha_min,
        linear_alpha_max=arguments.linear_alpha_max,
    )
