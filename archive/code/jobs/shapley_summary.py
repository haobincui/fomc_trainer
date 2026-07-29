import argparse
import json
from pathlib import Path

import pandas as pd


SECTION_ORDER = [
    "Participants' Views on Current Conditions and the Economic Outlook",
    "Staff Review of the Economic Situation",
    "Staff Review of the Financial Situation",
]


def _slugify_indicator(value: str) -> str:
    return value.replace("-", " ")


def _format_value(value: float) -> str:
    return f"{value:.4f}"


def _load_filtered_tables(filtered_path: Path, raw_filtered_path: Path) -> dict:
    filtered = pd.read_excel(filtered_path)
    raw_filtered = pd.read_excel(raw_filtered_path)
    return {
        "synthetic_target": {
            section: filtered.loc[filtered["section_name"] == section]
            .sort_values("mean", ascending=False)
            .head(5)[["indicator", "mean", "n"]]
            .to_dict(orient="records")
            for section in SECTION_ORDER
        },
        "actual_target": {
            section: raw_filtered.loc[raw_filtered["section_name"] == section]
            .sort_values("mean", ascending=False)
            .head(5)[["indicator", "mean", "n"]]
            .to_dict(orient="records")
            for section in SECTION_ORDER
        },
    }


def _delta_summary(diff_path: Path) -> tuple[dict, str]:
    diff = pd.read_excel(diff_path)
    diff = diff[diff["section_name"].isin(SECTION_ORDER)].copy()

    rows = []
    summary = {}
    for section in SECTION_ORDER:
        section_df = diff.loc[diff["section_name"] == section].copy()
        top_positive = section_df.sort_values("delta_mean(%)", ascending=False).iloc[0]
        top_negative = section_df.sort_values("delta_mean(%)", ascending=True).iloc[0]

        summary[section] = {
            "top_positive": {
                "indicator": top_positive["indicator"],
                "delta_mean_pct": float(top_positive["delta_mean(%)"]),
                "p_value": float(top_positive["p_value"]) if pd.notna(top_positive["p_value"]) else None,
                "n": int(top_positive["n"]),
            },
            "top_negative": {
                "indicator": top_negative["indicator"],
                "delta_mean_pct": float(top_negative["delta_mean(%)"]),
                "p_value": float(top_negative["p_value"]) if pd.notna(top_negative["p_value"]) else None,
                "n": int(top_negative["n"]),
            },
        }

        rows.append(
            (
                section,
                _slugify_indicator(str(top_positive["indicator"])),
                float(top_positive["delta_mean(%)"]),
                _slugify_indicator(str(top_negative["indicator"])),
                float(top_negative["delta_mean(%)"]),
            )
        )

    latex_lines = [
        r"\begin{tabular}{p{0.31\textwidth} p{0.22\textwidth} c p{0.22\textwidth} c}",
        r"\toprule",
        r"\textbf{Section} & \textbf{Largest Positive Indicator} & \textbf{$\Delta\%$} & \textbf{Most Negative Indicator} & \textbf{$\Delta\%$} \\",
        r"\midrule",
    ]
    for section, pos_ind, pos_delta, neg_ind, neg_delta in rows:
        latex_lines.append(
            f"{section} & {pos_ind} & {_format_value(pos_delta)} & {neg_ind} & {_format_value(neg_delta)} \\\\"
        )
    latex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return summary, "\n".join(latex_lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize leave-one-out masking outputs for Chapter 2.")
    parser.add_argument(
        "--filtered-xlsx",
        type=Path,
        default=Path("reformat_shapley_result/shapley_filter.xlsx"),
    )
    parser.add_argument(
        "--raw-filtered-xlsx",
        type=Path,
        default=Path("reformat_shapley_result/shapley_by_raw_filter.xlsx"),
    )
    parser.add_argument(
        "--diff-xlsx",
        type=Path,
        default=Path("reformat_shapley_result/shapley_result_with_diff.xlsx"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("output/chapter2/shapley_summary.json"),
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=Path("output/chapter2/shapley_delta_summary.tex"),
    )
    args = parser.parse_args()

    payload = _load_filtered_tables(args.filtered_xlsx, args.raw_filtered_xlsx)
    delta_summary, latex_table = _delta_summary(args.diff_xlsx)
    payload["delta_summary"] = delta_summary

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_tex.write_text(latex_table, encoding="utf-8")

    print("\n## Leave-one-out Masking Summary")
    for section, section_summary in delta_summary.items():
        top_positive = section_summary["top_positive"]
        top_negative = section_summary["top_negative"]
        print(
            f"- {section}: "
            f"top_positive={top_positive['indicator']} ({top_positive['delta_mean_pct']:.4f}%), "
            f"top_negative={top_negative['indicator']} ({top_negative['delta_mean_pct']:.4f}%)"
        )


if __name__ == "__main__":
    main()
