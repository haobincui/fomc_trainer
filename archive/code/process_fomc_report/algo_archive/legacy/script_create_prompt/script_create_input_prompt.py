from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from generate_prompt import (
    filter_last_two_years_data,
    generate_fomc_prompt,
    load_target_rate,
    multi_df_to_markdown_tables,
    process_data,
    process_unemployment_rate_data,
)


BASE_DIR = Path(__file__).resolve().parent


def get_input_data_map(input_folder: Path) -> dict[str, dict[str, pd.DataFrame]]:
    input_data_map: dict[str, dict[str, pd.DataFrame]] = {}
    for indicator_dir in sorted(input_folder.iterdir()):
        if not indicator_dir.is_dir():
            continue
        indicator_name = indicator_dir.name
        input_data_map[indicator_name] = {}
        for data_file in sorted(indicator_dir.iterdir()):
            if data_file.suffix == ".xlsx":
                frame = pd.read_excel(data_file, engine="openpyxl")
            elif data_file.suffix == ".csv":
                frame = pd.read_csv(data_file)
            else:
                continue
            if indicator_name == "Unemployment Rate":
                frame = process_unemployment_rate_data(frame)
            else:
                frame = process_data(frame)
            input_data_map[indicator_name][data_file.name] = frame
    return input_data_map


def process_indicator(indicator: str, input_data_map: dict[str, dict[str, pd.DataFrame]], meeting_date) -> tuple[str | None, str | None]:
    if indicator not in input_data_map:
        return None, None
    data_dict = {}
    for name, frame in input_data_map[indicator].items():
        data_dict[name] = filter_last_two_years_data(frame, meeting_date)
    data_label = ", ".join(data_dict)
    table = multi_df_to_markdown_tables(data_dict)
    return data_label, f"Indicators: {indicator} \n\n{table}"


def generate_section_prompt(
    input_fomc_file: Path,
    output_file: Path,
    input_data_folder: Path,
    with_reference: bool = False,
) -> Path:
    non_core = {"Other", "Non-Core", "Delete", ""}
    input_data_map = get_input_data_map(input_data_folder)
    input_fomc = pd.read_excel(input_fomc_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for idx, row in input_fomc.iterrows():
            meeting_date = row["date"]
            section_style = row["section_name"]
            topic = (row.get("relabel") or row.get("label") or "").strip()
            if topic in non_core:
                continue

            data_labels = ""
            table_strs = ""
            for indicator in [item.strip() for item in topic.split(",")]:
                if indicator in non_core:
                    continue
                data_label, table_str = process_indicator(indicator, input_data_map, meeting_date)
                if data_label and table_str:
                    data_labels += data_label
                    table_strs += table_str

            if not data_labels.strip():
                continue

            reference_excerpt = row.get("raw_text", "") if with_reference else ""
            prompt = generate_fomc_prompt(
                pd.Timestamp(meeting_date).date().isoformat(),
                topic,
                section_style,
                table_strs,
                data_labels,
                reference_excerpt,
            )
            rate_change, current_rate = load_target_rate(meeting_date)
            output_dict = {
                "index": int(idx),
                "prompt": prompt,
                "reference": reference_excerpt,
                "provided_data": table_strs,
                "rate_change": rate_change,
                "meeting_date": pd.Timestamp(meeting_date).date().isoformat(),
                "current_rate": current_rate,
                "section_name": section_style,
                "topic": topic,
                "reference_excerpt": reference_excerpt,
            }
            handle.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
    return output_file


def merge_json(file1: Path, file2: Path, output_file: Path) -> Path:
    combined = []
    for file in [file1, file2]:
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    combined.append(json.loads(line))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(combined):
            item["index"] = idx
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create FOMC QA prompts from labeled minutes data.")
    parser.add_argument("input_fomc_file", type=Path)
    parser.add_argument("output_file", type=Path)
    parser.add_argument(
        "--input-data-folder",
        type=Path,
        default=BASE_DIR.parent / "input_data" / "us_data",
    )
    parser.add_argument("--with-reference", action="store_true")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    result = generate_section_prompt(
        args.input_fomc_file,
        args.output_file,
        args.input_data_folder,
        with_reference=args.with_reference,
    )
    print(f"Saved prompts to {result}")
