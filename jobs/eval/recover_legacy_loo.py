"""Recover descriptive paired LOO deltas from the surviving aggregate workbook.

The historical row-level generations are no longer present.  The workbook
``shapley_result_with_diff.xlsx`` nevertheless retains indicator-specific
paired means.  Because

    d_masked - d_full = s_full - s_masked,

its point estimates can be converted to the signed LOO estimand without
regeneration.  Historical standard errors are retained in the CSV only for
provenance; they are not used in the manuscript table because meeting IDs are
unavailable and clustering cannot be reconstructed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


SECTION_ORDER = [
    "Participants' Views on Current Conditions and the Economic Outlook",
    "Staff Review of the Economic Situation",
    "Staff Review of the Financial Situation",
]
SECTION_HEADERS = {
    SECTION_ORDER[0]: "Participants' Views",
    SECTION_ORDER[1]: "Economic Situation",
    SECTION_ORDER[2]: "Financial Situation",
}


def recover_legacy_rows(input_xlsx: Path) -> pd.DataFrame:
    frame = pd.read_excel(input_xlsx, keep_default_na=False)
    required = {
        "indicator",
        "section_name",
        "mean_base",
        "mean_indicator",
        "delta_mean(%)",
        "n",
        "std",
        "stderr",
        "t_stat",
        "p_value",
        "ci_lower",
        "ci_upper",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Legacy workbook is missing required columns: {sorted(missing)}")

    recovered = frame.loc[frame["section_name"].isin(SECTION_ORDER)].copy()
    recovered = recovered.loc[recovered["indicator"].astype(str) != "None"].copy()
    if recovered.empty:
        raise ValueError("Legacy workbook contains no rows for the three manuscript sections")

    recovered["mean_distance_full"] = recovered["mean_base"].astype(float)
    recovered["mean_distance_masked"] = recovered["mean_indicator"].astype(float)
    recovered["mean_similarity_full"] = 1.0 - recovered["mean_distance_full"]
    recovered["mean_similarity_masked"] = 1.0 - recovered["mean_distance_masked"]
    recovered["mean_delta"] = (
        recovered["mean_distance_masked"] - recovered["mean_distance_full"]
    )

    expected_percentage = (
        100.0 * recovered["mean_delta"] / recovered["mean_distance_full"]
    )
    maximum_error = float(
        (expected_percentage - recovered["delta_mean(%)"].astype(float)).abs().max()
    )
    if maximum_error > 1e-4:
        raise ValueError(
            "Legacy percentage column is inconsistent with the recovered paired "
            f"delta (maximum absolute error={maximum_error})"
        )

    output = recovered[
        [
            "indicator",
            "section_name",
            "n",
            "mean_similarity_full",
            "mean_similarity_masked",
            "mean_distance_full",
            "mean_distance_masked",
            "mean_delta",
            "std",
            "stderr",
            "t_stat",
            "p_value",
            "ci_lower",
            "ci_upper",
        ]
    ].copy()
    output = output.rename(
        columns={
            "n": "legacy_n_pairs",
            "std": "legacy_std_delta",
            "stderr": "legacy_stderr_delta_unclustered",
            "t_stat": "legacy_t_stat_unclustered",
            "p_value": "legacy_p_value_unclustered",
            "ci_lower": "legacy_ci_lower_unclustered",
            "ci_upper": "legacy_ci_upper_unclustered",
        }
    )
    output["inference_status"] = (
        "descriptive_only_missing_row_level_meeting_clusters_and_seed_metadata"
    )
    return output.sort_values(["indicator", "section_name"]).reset_index(drop=True)


def recover_legacy_internal_rows(input_xlsx: Path) -> pd.DataFrame:
    """Recover self-target deltas, for which ``delta = 1 - s_masked``."""

    frame = pd.read_excel(input_xlsx, keep_default_na=False)
    required = {
        "indicator",
        "section_name",
        "n",
        "mean",
        "std",
        "stderr",
        "t_stat",
        "p_value",
        "ci_lower",
        "ci_upper",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Legacy internal-target workbook is missing required columns: {sorted(missing)}"
        )

    recovered = frame.loc[frame["section_name"].isin(SECTION_ORDER)].copy()
    if recovered.empty:
        raise ValueError(
            "Legacy internal-target workbook contains no rows for the manuscript sections"
        )
    recovered["mean_delta"] = recovered["mean"].astype(float)
    recovered["mean_similarity_full"] = 1.0
    recovered["mean_similarity_masked"] = 1.0 - recovered["mean_delta"]
    recovered["mean_distance_full"] = 0.0
    recovered["mean_distance_masked"] = recovered["mean_delta"]

    output = recovered[
        [
            "indicator",
            "section_name",
            "n",
            "mean_similarity_full",
            "mean_similarity_masked",
            "mean_distance_full",
            "mean_distance_masked",
            "mean_delta",
            "std",
            "stderr",
            "t_stat",
            "p_value",
            "ci_lower",
            "ci_upper",
        ]
    ].copy()
    output = output.rename(
        columns={
            "n": "legacy_n_pairs",
            "std": "legacy_std_delta",
            "stderr": "legacy_stderr_delta_unclustered",
            "t_stat": "legacy_t_stat_unclustered",
            "p_value": "legacy_p_value_unclustered",
            "ci_lower": "legacy_ci_lower_unclustered",
            "ci_upper": "legacy_ci_upper_unclustered",
        }
    )
    output["inference_status"] = (
        "descriptive_only_missing_row_level_meeting_clusters_and_seed_metadata"
    )
    return output.sort_values(["indicator", "section_name"]).reset_index(drop=True)


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _format_delta(value: float) -> str:
    if abs(value) < 0.00005:
        return "0.0000"
    return f"{value:+.4f}"


def _display_indicator(value: object) -> str:
    return str(value).replace("-", " ")


def render_latex_table(recovered: pd.DataFrame) -> str:
    pivot = recovered.pivot(
        index="indicator",
        columns="section_name",
        values="mean_delta",
    ).reindex(columns=SECTION_ORDER)

    lines = [
        r"\begin{tabular}{p{4.7cm}ccc}",
        r"\toprule",
        (
            r"\textbf{Indicator} & "
            + " & ".join(
                rf"\textbf{{{_latex_escape(SECTION_HEADERS[section])}}}"
                for section in SECTION_ORDER
            )
            + r" \\"
        ),
        r"\midrule",
    ]
    for indicator, row in pivot.iterrows():
        values = [
            "" if pd.isna(row[section]) else _format_delta(float(row[section]))
            for section in SECTION_ORDER
        ]
        lines.append(
            f"{_latex_escape(_display_indicator(indicator))} & "
            + " & ".join(f"${value}$" if value else "" for value in values)
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover descriptive signed LOO deltas from the legacy aggregate workbook."
    )
    parser.add_argument("--input-xlsx", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--internal-input-xlsx", type=Path)
    parser.add_argument("--internal-output-csv", type=Path)
    parser.add_argument("--internal-output-tex", type=Path)
    args = parser.parse_args()

    recovered = recover_legacy_rows(args.input_xlsx)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    recovered.to_csv(args.output_csv, index=False)
    args.output_tex.write_text(render_latex_table(recovered), encoding="utf-8")

    if recovered["mean_delta"].map(math.isfinite).all():
        print(f"✅ Recovered {len(recovered)} descriptive paired LOO cells")
        print(f"- CSV: {args.output_csv}")
        print(f"- LaTeX: {args.output_tex}")

    internal_arguments = (
        args.internal_input_xlsx,
        args.internal_output_csv,
        args.internal_output_tex,
    )
    if any(value is not None for value in internal_arguments):
        if not all(value is not None for value in internal_arguments):
            raise ValueError(
                "--internal-input-xlsx, --internal-output-csv, and "
                "--internal-output-tex must be supplied together"
            )
        internal = recover_legacy_internal_rows(args.internal_input_xlsx)
        args.internal_output_csv.parent.mkdir(parents=True, exist_ok=True)
        args.internal_output_tex.parent.mkdir(parents=True, exist_ok=True)
        internal.to_csv(args.internal_output_csv, index=False)
        args.internal_output_tex.write_text(render_latex_table(internal), encoding="utf-8")
        print(f"✅ Recovered {len(internal)} internal-sensitivity LOO cells")
        print(f"- CSV: {args.internal_output_csv}")
        print(f"- LaTeX: {args.internal_output_tex}")


if __name__ == "__main__":
    main()
