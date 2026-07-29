from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from process_fomc_report.generate_prompt_and_response.algo.common.config import load_pipeline_config, resolve_path
from process_fomc_report.generate_prompt_and_response.algo.common.labeling import create_label_prompt, preprocess_label
from process_fomc_report.generate_prompt_and_response.algo.common.llm_client import get_response, load_llm_config


def label_minutes_sections(
    *,
    input_file: Path,
    output_dir: Path,
    indicator_file: Path,
    llm_payload: dict,
) -> Path:
    frame = pd.read_excel(input_file)
    has_section = "section_name" in frame.columns
    columns = ["section_name", "details"] if has_section else ["details"]
    lines = frame[columns].dropna().astype(str).values.tolist()
    if not has_section:
        lines = [["", line[0]] for line in lines]

    llm_config = load_llm_config(llm_payload)
    results = []
    for index, (section, line) in enumerate(tqdm(lines, desc=input_file.name)):
        pre_label = preprocess_label(section, line)
        if pre_label != "To-Label":
            results.append(
                {
                    "line_id": index + 1,
                    "section_name": section,
                    "raw_text": line,
                    "label": "Pre-Non-Core",
                    "label_type": "Pre-Non-Core",
                    "explanation": "Pre-Non-Core",
                    "reason": pre_label,
                    "response": "Pre-Non-Core",
                }
            )
            continue

        prompt = create_label_prompt(section, line, indicator_file)
        response, reasoning = get_response(prompt, llm_config)
        results.append(
            {
                "line_id": index + 1,
                "section_name": section,
                "raw_text": line,
                "label": "",
                "label_type": "",
                "explanation": "",
                "reason": reasoning,
                "response": response,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_file.stem}_labeled.xlsx"
    pd.DataFrame(results).to_excel(output_path, index=False)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label paragraph-level FOMC minutes sections.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_pipeline_config(args.config)
    labeling_cfg = config["labeling"]
    output_dir = args.output_dir or resolve_path(labeling_cfg["output_root"]) / "after_2009"
    indicator_file = resolve_path(labeling_cfg["indicator_file"])
    output_path = label_minutes_sections(
        input_file=args.input_file,
        output_dir=output_dir,
        indicator_file=indicator_file,
        llm_payload=labeling_cfg,
    )
    print(f"Saved labeled minutes to {output_path}")


if __name__ == "__main__":
    main()
