"""
Plot the Prandtl-plane vertical-gap CL/CDi ratio trend from XFLR5 polar exports.

The script uses:
- Volde_MainWingOnly...txt as the isolated-wing reference
- Volde h = ...txt files as the two-wing cases for different vertical gaps

The main plot shows:

    (CL/CDi)_two_wings / (CL/CDi)_main_wing

The summary tables also report the isolated-wing/two-wing coefficient ratios:

    coefficient_single_wing / coefficient_two_wings

This assumes the single-wing and two-wing XFLR5 exports use consistent reference
areas. If an older export used one-wing reference area for the two-wing case,
rerun with:

    --single-wing-factor 2

Run from the repository root:

    python "final_characteristics/aerodynamics/h_b determination/h_b_lift_ratio.py"
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_DATA_DIR = SCRIPT_DIR / "Wing CL values"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "h_b_results"
DEFAULT_ALPHA_DEGS = (0.0, 4.0, 8.0)
DEFAULT_ERROR_BAND_PERCENT = 5.0

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from parameters import WING_SPAN
except ImportError:
    WING_SPAN = 13.0


@dataclass(frozen=True)
class XFLRPolar:
    path: Path
    metadata: dict[str, str]
    columns: dict[str, np.ndarray]

    @property
    def alpha(self) -> np.ndarray:
        return self.columns["alpha"]

    @property
    def label(self) -> str:
        return self.metadata.get("Plane name", self.path.stem)

    def value_at_alpha(self, column: str, alpha_deg: float) -> float:
        if column not in self.columns:
            raise KeyError(f"Column {column!r} was not found in {self.path}")

        alpha_min = float(self.alpha.min())
        alpha_max = float(self.alpha.max())
        if alpha_deg < alpha_min or alpha_deg > alpha_max:
            raise ValueError(
                f"Requested alpha={alpha_deg:g} deg is outside {self.path.name}'s "
                f"range ({alpha_min:g} to {alpha_max:g} deg)."
            )

        return float(np.interp(alpha_deg, self.alpha, self.columns[column]))


def parse_xflr5_polar(path: Path) -> XFLRPolar:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    metadata: dict[str, str] = {}
    header_index = None
    headers: list[str] | None = None

    for index, line in enumerate(lines):
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in {"Plane name", "Polar name", "Freestream speed"}:
                metadata[key] = value

        tokens = line.split()
        if "alpha" in tokens and "CL" in tokens:
            header_index = index
            headers = tokens
            break

    if header_index is None or headers is None:
        raise ValueError(f"Could not find the XFLR5 polar header in {path}")

    rows: list[list[float]] = []
    for line in lines[header_index + 1 :]:
        tokens = line.split()
        if len(tokens) < len(headers):
            continue

        try:
            row = [float(token) for token in tokens[: len(headers)]]
        except ValueError:
            continue

        rows.append(row)

    if not rows:
        raise ValueError(f"No numeric polar rows could be parsed from {path}")

    data = np.array(rows, dtype=float)
    sort_index = np.argsort(data[:, headers.index("alpha")])
    data = data[sort_index]

    columns = {
        header: data[:, column_index]
        for column_index, header in enumerate(headers)
    }

    return XFLRPolar(path=path, metadata=metadata, columns=columns)


def parse_vertical_gap_m(polar: XFLRPolar) -> float:
    search_text = " ".join(
        [
            polar.metadata.get("Plane name", ""),
            polar.path.stem,
        ]
    )
    match = re.search(r"\bh\s*=\s*([+-]?\d+(?:[._]\d+)?)", search_text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not determine vertical gap h from {polar.path.name}")

    return float(match.group(1).replace("_", "."))


def discover_polar_files(
    data_dir: Path,
    single_wing_file: Path | None,
    two_wing_glob: str,
) -> tuple[Path, list[Path]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    if single_wing_file is None:
        candidates = sorted(data_dir.glob("*MainWingOnly*.txt"))
        if len(candidates) != 1:
            raise ValueError(
                f"Expected exactly one *MainWingOnly*.txt file in {data_dir}, "
                f"found {len(candidates)}."
            )
        single_wing_file = candidates[0]

    two_wing_files = sorted(
        path for path in data_dir.glob(two_wing_glob) if "MainWingOnly" not in path.name
    )
    if not two_wing_files:
        raise ValueError(f"No two-wing files matched {two_wing_glob!r} in {data_dir}")

    return single_wing_file, two_wing_files


def build_lift_ratio_rows(
    single_wing_polar: XFLRPolar,
    two_wing_polars: list[XFLRPolar],
    alpha_deg: float,
    span_m: float,
    single_wing_factor: float,
) -> list[dict[str, float | str]]:
    single_wing_cl = single_wing_polar.value_at_alpha("CL", alpha_deg)
    single_wing_cdi = single_wing_polar.value_at_alpha("CDi", alpha_deg)
    reference_cl = single_wing_factor * single_wing_cl
    reference_cdi = single_wing_factor * single_wing_cdi
    rows: list[dict[str, float | str]] = []

    for polar in two_wing_polars:
        vertical_gap_m = parse_vertical_gap_m(polar)
        two_wing_cl = polar.value_at_alpha("CL", alpha_deg)
        two_wing_cdi = polar.value_at_alpha("CDi", alpha_deg)
        single_wing_cl_over_cdi = single_wing_cl / single_wing_cdi
        two_wing_cl_over_cdi = two_wing_cl / two_wing_cdi
        cl_over_cdi_ratio = two_wing_cl_over_cdi / single_wing_cl_over_cdi
        cl_ratio = reference_cl / two_wing_cl
        cdi_ratio = reference_cdi / two_wing_cdi
        cl_signed_distance_to_one = cl_ratio - 1.0
        cdi_signed_distance_to_one = cdi_ratio - 1.0

        rows.append(
            {
                "file": polar.path.name,
                "h_m": vertical_gap_m,
                "h_over_b": vertical_gap_m / span_m,
                "alpha_deg": alpha_deg,
                "CL_single_wing": single_wing_cl,
                "single_wing_factor": single_wing_factor,
                "CL_reference": reference_cl,
                "CL_two_wings": two_wing_cl,
                "ratio_reference_over_two_wings": cl_ratio,
                "signed_distance_to_1": cl_signed_distance_to_one,
                "absolute_distance_to_1": abs(cl_signed_distance_to_one),
                "absolute_distance_to_1_percent": abs(cl_signed_distance_to_one) * 100.0,
                "CDi_single_wing": single_wing_cdi,
                "CDi_reference": reference_cdi,
                "CDi_two_wings": two_wing_cdi,
                "CL_over_CDi_single_wing": single_wing_cl_over_cdi,
                "CL_over_CDi_two_wings": two_wing_cl_over_cdi,
                "CL_over_CDi_ratio_two_wings_over_main_wing": cl_over_cdi_ratio,
                "CDi_ratio_reference_over_two_wings": cdi_ratio,
                "CDi_signed_distance_to_1": cdi_signed_distance_to_one,
                "CDi_absolute_distance_to_1": abs(cdi_signed_distance_to_one),
                "CDi_absolute_distance_to_1_percent": abs(cdi_signed_distance_to_one) * 100.0,
            }
        )

    rows.sort(key=lambda row: float(row["h_m"]))
    return rows


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value: object, decimals: int = 5) -> str:
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _plain_ratio_definition(single_wing_factor: float) -> str:
    if abs(single_wing_factor - 1.0) < 1e-12:
        return "coefficient_single_wing / coefficient_two_wings"
    return f"({single_wing_factor:g} * coefficient_single_wing) / coefficient_two_wings"


def _math_ratio_label(single_wing_factor: float, coefficient: str) -> str:
    if abs(single_wing_factor - 1.0) < 1e-12:
        return rf"$C_{{{coefficient},1w}} / C_{{{coefficient},2w}}$ [-]"
    return rf"$({single_wing_factor:g} C_{{{coefficient},1w}}) / C_{{{coefficient},2w}}$ [-]"


def _unique_alpha_values(rows: list[dict[str, float | str]]) -> list[float]:
    return sorted({float(row["alpha_deg"]) for row in rows})


def _rows_for_alpha(
    rows: list[dict[str, float | str]],
    alpha_deg: float,
) -> list[dict[str, float | str]]:
    return [
        row
        for row in rows
        if abs(float(row["alpha_deg"]) - alpha_deg) < 1e-9
    ]


def write_markdown_summary(
    path: Path,
    rows: list[dict[str, float | str]],
    single_wing_file: Path,
    alpha_degs: list[float],
    span_m: float,
    single_wing_factor: float,
    error_band_percent: float,
) -> None:
    def build_table(
        reference_key: str,
        two_wing_key: str,
        ratio_key: str,
        error_key: str,
        coefficient_label: str,
    ) -> str:
        header = [
            "alpha [deg]",
            "h [m]",
            "h/b [-]",
            f"Reference {coefficient_label}",
            f"Two-wing {coefficient_label}",
            "Ratio",
            "|Ratio - 1| [%]",
        ]
        table = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        for row in rows:
            table_row = [
                row["alpha_deg"],
                row["h_m"],
                row["h_over_b"],
                row[reference_key],
                row[two_wing_key],
                row[ratio_key],
                row[error_key],
            ]
            table.append(
                "| " + " | ".join(_format_value(value) for value in table_row) + " |"
            )
        return "\n".join(table)

    def build_cl_over_cdi_table() -> str:
        header = [
            "alpha [deg]",
            "h [m]",
            "h/b [-]",
            "(CL/CDi) main wing",
            "(CL/CDi) two wings",
            "(CL/CDi) two wings / main wing",
        ]
        table = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        for row in rows:
            table_row = [
                row["alpha_deg"],
                row["h_m"],
                row["h_over_b"],
                row["CL_over_CDi_single_wing"],
                row["CL_over_CDi_two_wings"],
                row["CL_over_CDi_ratio_two_wings_over_main_wing"],
            ]
            table.append(
                "| " + " | ".join(_format_value(value) for value in table_row) + " |"
            )
        return "\n".join(table)

    def first_inside_band_summary(error_key: str, label: str) -> str:
        parts = []
        for alpha in alpha_degs:
            alpha_rows = _rows_for_alpha(rows, alpha)
            candidates = [
                row
                for row in alpha_rows
                if float(row[error_key]) <= error_band_percent
            ]
            if candidates:
                first_row = candidates[0]
                parts.append(
                    f"alpha = {alpha:g} deg: h = {float(first_row['h_m']):.2f} m "
                    f"(h/b = {float(first_row['h_over_b']):.4f})"
                )
            else:
                parts.append(f"alpha = {alpha:g} deg: no sampled case inside band")
        return f"First available {label} cases inside the {error_band_percent:g}% band: " + "; ".join(parts) + "."

    content = "\n\n".join(
        [
            "# h/b CL and CDi Ratio Summary",
            f"Single-wing reference file: `{single_wing_file.name}`",
            "Evaluation angles of attack: " + ", ".join(f"{alpha:g} deg" for alpha in alpha_degs),
            f"Wing span used for normalization: b = {span_m:.3f} m",
            (
                "Ratio definition: "
                f"{_plain_ratio_definition(single_wing_factor)}. "
                "A value of 1 indicates that the two-wing case has reached the "
                "same coefficient as the isolated-wing reference after the "
                "chosen reference-area correction."
            ),
            first_inside_band_summary("absolute_distance_to_1_percent", "CL"),
            first_inside_band_summary("CDi_absolute_distance_to_1_percent", "CDi"),
            "## CL/CDi ratio\n\n" + build_cl_over_cdi_table(),
            "## CL ratio\n\n"
            + build_table(
                "CL_reference",
                "CL_two_wings",
                "ratio_reference_over_two_wings",
                "absolute_distance_to_1_percent",
                "CL",
            ),
            "## CDi ratio\n\n"
            + build_table(
                "CDi_reference",
                "CDi_two_wings",
                "CDi_ratio_reference_over_two_wings",
                "CDi_absolute_distance_to_1_percent",
                "CDi",
            ),
            (
                "Note: the default assumes the XFLR5 exports use consistent "
                "reference areas. For older two-wing exports normalized by one "
                "wing area, rerun with `--single-wing-factor 2`."
            ),
            "",
        ]
    )

    path.write_text(content, encoding="utf-8")


def plot_lift_ratio(
    rows: list[dict[str, float | str]],
    output_dir: Path,
    span_m: float,
    single_wing_factor: float,
    error_band_percent: float,
) -> list[Path]:
    figure, ratio_axis = plt.subplots(figsize=(9.5, 5.8))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for alpha_index, alpha in enumerate(_unique_alpha_values(rows)):
        alpha_rows = _rows_for_alpha(rows, alpha)
        color = color_cycle[alpha_index % len(color_cycle)]
        vertical_gap_m = np.array([float(row["h_m"]) for row in alpha_rows])
        cl_over_cdi_ratio = np.array(
            [float(row["CL_over_CDi_ratio_two_wings_over_main_wing"]) for row in alpha_rows]
        )
        ratio_axis.plot(
            vertical_gap_m,
            cl_over_cdi_ratio,
            marker="o",
            linewidth=1.8,
            color=color,
            label=fr"$\alpha = {alpha:g}^\circ$",
        )

    h_values = sorted({float(row["h_m"]) for row in rows})
    for h_value in h_values:
        rows_at_h = [
            row
            for row in rows
            if abs(float(row["h_m"]) - h_value) < 1e-9
        ]
        label_y = max(
            float(row["CL_over_CDi_ratio_two_wings_over_main_wing"]) for row in rows_at_h
        )
        ratio_axis.annotate(
            f"{h_value:g} m",
            xy=(h_value, label_y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="black",
            clip_on=False,
        )

    ratio_axis.set_xlabel("Vertical gap h [m]")
    ratio_axis.set_ylabel(r"$(C_L/C_{D_i})_{2w}/(C_L/C_{D_i})_{1w}$ [-]")
    ratio_axis.set_title(r"Vertical Gap $C_L/C_{D_i}$ Ratio")
    ratio_axis.grid(True, alpha=0.35)
    ratio_axis.legend(frameon=False, loc="best", ncol=1)

    figure.tight_layout()

    png_path = output_dir / "h_b_lift_ratio.png"
    pdf_path = output_dir / "h_b_lift_ratio.pdf"
    combined_png_path = output_dir / "h_b_cl_cdi_ratio.png"
    combined_pdf_path = output_dir / "h_b_cl_cdi_ratio.pdf"
    cl_over_cdi_png_path = output_dir / "h_b_cl_over_cdi.png"
    cl_over_cdi_pdf_path = output_dir / "h_b_cl_over_cdi.pdf"
    cl_over_cdi_ratio_png_path = output_dir / "h_b_cl_over_cdi_ratio.png"
    cl_over_cdi_ratio_pdf_path = output_dir / "h_b_cl_over_cdi_ratio.pdf"
    legacy_cdi_png_path = output_dir / "h_b_cdi_ratio.png"
    legacy_cdi_pdf_path = output_dir / "h_b_cdi_ratio.pdf"
    figure.savefig(png_path, dpi=300)
    figure.savefig(pdf_path)
    figure.savefig(combined_png_path, dpi=300)
    figure.savefig(combined_pdf_path)
    figure.savefig(cl_over_cdi_png_path, dpi=300)
    figure.savefig(cl_over_cdi_pdf_path)
    figure.savefig(cl_over_cdi_ratio_png_path, dpi=300)
    figure.savefig(cl_over_cdi_ratio_pdf_path)
    figure.savefig(legacy_cdi_png_path, dpi=300)
    figure.savefig(legacy_cdi_pdf_path)
    plt.close(figure)
    return [
        png_path,
        pdf_path,
        combined_png_path,
        combined_pdf_path,
        cl_over_cdi_png_path,
        cl_over_cdi_pdf_path,
        cl_over_cdi_ratio_png_path,
        cl_over_cdi_ratio_pdf_path,
        legacy_cdi_png_path,
        legacy_cdi_pdf_path,
    ]


def print_summary(rows: list[dict[str, float | str]], output_dir: Path) -> None:
    print("h/b CL and CDi ratio analysis")
    for alpha in _unique_alpha_values(rows):
        alpha_rows = _rows_for_alpha(rows, alpha)
        best_cl_row = min(
            alpha_rows,
            key=lambda row: float(row["absolute_distance_to_1_percent"]),
        )
        best_cdi_row = min(
            alpha_rows,
            key=lambda row: float(row["CDi_absolute_distance_to_1_percent"]),
        )
        print(
            f"alpha = {alpha:g} deg | "
            f"best CL: h = {float(best_cl_row['h_m']):.2f} m, "
            f"ratio = {float(best_cl_row['ratio_reference_over_two_wings']):.4f}, "
            f"error = {float(best_cl_row['absolute_distance_to_1_percent']):.2f}% | "
            f"best CDi: h = {float(best_cdi_row['h_m']):.2f} m, "
            f"ratio = {float(best_cdi_row['CDi_ratio_reference_over_two_wings']):.4f}, "
            f"error = {float(best_cdi_row['CDi_absolute_distance_to_1_percent']):.2f}%"
        )
    print(f"Outputs written to: {output_dir}")


def run_analysis(
    data_dir: Path,
    single_wing_file: Path | None,
    two_wing_glob: str,
    output_dir: Path,
    alpha_degs: list[float],
    span_m: float,
    single_wing_factor: float,
    error_band_percent: float,
) -> None:
    single_wing_file, two_wing_files = discover_polar_files(
        data_dir=data_dir,
        single_wing_file=single_wing_file,
        two_wing_glob=two_wing_glob,
    )

    single_wing_polar = parse_xflr5_polar(single_wing_file)
    two_wing_polars = [parse_xflr5_polar(path) for path in two_wing_files]
    rows: list[dict[str, float | str]] = []
    for alpha_deg in sorted(alpha_degs):
        rows.extend(
            build_lift_ratio_rows(
                single_wing_polar=single_wing_polar,
                two_wing_polars=two_wing_polars,
                alpha_deg=alpha_deg,
                span_m=span_m,
                single_wing_factor=single_wing_factor,
            )
        )
    rows.sort(key=lambda row: (float(row["alpha_deg"]), float(row["h_m"])))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "h_b_lift_ratio.csv", rows)
    write_markdown_summary(
        path=output_dir / "h_b_lift_ratio_summary.md",
        rows=rows,
        single_wing_file=single_wing_file,
        alpha_degs=sorted(alpha_degs),
        span_m=span_m,
        single_wing_factor=single_wing_factor,
        error_band_percent=error_band_percent,
    )
    plot_lift_ratio(
        rows=rows,
        output_dir=output_dir,
        span_m=span_m,
        single_wing_factor=single_wing_factor,
        error_band_percent=error_band_percent,
    )
    print_summary(rows, output_dir)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the h/b CL and CDi ratio trends from XFLR5 VLM polar files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the XFLR5 polar text exports.",
    )
    parser.add_argument(
        "--single-wing-file",
        type=Path,
        default=None,
        help="Optional explicit path to the isolated-wing polar file.",
    )
    parser.add_argument(
        "--two-wing-glob",
        default="Volde h =*.txt",
        help="Glob pattern for the two-wing h cases inside --data-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated plots and summary tables.",
    )
    parser.add_argument(
        "--alpha-deg",
        nargs="+",
        type=float,
        default=list(DEFAULT_ALPHA_DEGS),
        help="Angles of attack at which CL and CDi are interpolated and compared.",
    )
    parser.add_argument(
        "--span-m",
        type=float,
        default=float(WING_SPAN),
        help="Wing span b used to compute h/b.",
    )
    parser.add_argument(
        "--single-wing-factor",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied to the single-wing coefficient before comparing with "
            "the two-wing coefficient. "
            "Use 1 for consistent reference-area exports; use 2 only for older "
            "two-wing exports normalized by one wing area."
        ),
    )
    parser.add_argument(
        "--error-band-percent",
        type=float,
        default=DEFAULT_ERROR_BAND_PERCENT,
        help="Acceptable |ratio - 1| band shown on the plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_analysis(
        data_dir=args.data_dir,
        single_wing_file=args.single_wing_file,
        two_wing_glob=args.two_wing_glob,
        output_dir=args.output_dir,
        alpha_degs=args.alpha_deg,
        span_m=args.span_m,
        single_wing_factor=args.single_wing_factor,
        error_band_percent=args.error_band_percent,
    )
