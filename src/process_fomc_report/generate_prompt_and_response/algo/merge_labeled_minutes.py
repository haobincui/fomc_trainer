from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from process_fomc_report.generate_prompt_and_response.algo.common.config import load_pipeline_config, resolve_path
from process_fomc_report.generate_prompt_and_response.algo.common.labeling import (
    extract_date_from_filename,
    merge_labeled_after_2009,
    merge_labeled_before_2009,
)


def merge_labeled_minutes(
    *,
    input_dir: Path,
    output_file: Path,
    indicator_file: Path,
    scope: str,
) -> Path:
    frames: list[pd.DataFrame] = []
    for file in sorted(input_dir.glob("*.xlsx")) + sorted(input_dir.glob("*.csv")):
        frame = pd.read_excel(file) if file.suffix == ".xlsx" else pd.read_csv(file)
        frame["file_name"] = file.name
        frame["date"] = frame.get("date") if "date" in frame.columns else extract_date_from_filename(file.name)
        if scope == "after_2009":
            cleaned = merge_labeled_after_2009(frame, indicator_file)
        else:
            cleaned = merge_labeled_before_2009(frame, indicator_file, file.name)
        frames.append(cleaned)

    if not frames:
        raise FileNotFoundError(f"No labeled files found in {input_dir}")

    merged = pd.concat(frames, ignore_index=True)
    if "line_id" in merged.columns:
        merged.sort_values(by=["date", "file_name", "line_id"], inplace=True)
    else:
        merged.sort_values(by=["date", "file_name"], inplace=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(output_file, index=False)
    return output_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge and normalize labeled minutes files.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scope", choices=["after_2009", "before_2009"], default="after_2009")
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_pipeline_config(args.config)
    labeling_cfg = config["labeling"]
    default_input_dir = resolve_path(labeling_cfg["output_root"]) / args.scope
    default_output = resolve_path(labeling_cfg["output_root"]) / f"merged_labeled_{args.scope}.xlsx"
    output_path = merge_labeled_minutes(
        input_dir=args.input_dir or default_input_dir,
        output_file=args.output_file or default_output,
        indicator_file=resolve_path(labeling_cfg["indicator_file"]),
        scope=args.scope,
    )
    print(f"Saved merged labeled minutes to {output_path}")


if __name__ == "__main__":
    main()
