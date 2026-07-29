from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from label_html_files.process_label_response.process_label_response import (
    extract_date_from_filename,
    extract_labels_from_text,
    process_len,
)


BASE_DIR = Path(__file__).resolve().parent


def _preprocess_raw_text(file_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame = frame[~frame["raw_text"].astype(str).apply(process_len)]
    frame["label"] = frame["response"].apply(extract_labels_from_text)
    frame["date"] = extract_date_from_filename(file_name)
    return frame


def process_labeled_text_before_2009(
    input_dir: Path = BASE_DIR / "output" / "labeled_text" / "before_2009",
    output_file: Path = BASE_DIR / "output" / "labeled_text" / "merged_labeled_before_2009.xlsx",
) -> Path:
    frames: list[pd.DataFrame] = []
    for file in sorted(input_dir.glob("*.xlsx")):
        frame = pd.read_excel(file)
        frame = _preprocess_raw_text(file.name, frame)
        frame = frame[~frame["label"].isin(["Not-Found", "Non-Core", "Delete"])].copy()
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No labeled Excel files found in {input_dir}")

    merged = pd.concat(frames, ignore_index=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(output_file, index=False)
    return output_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge and normalize labeled before-2009 FOMC text.")
    parser.add_argument("--input-dir", type=Path, default=BASE_DIR / "output" / "labeled_text" / "before_2009")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=BASE_DIR / "output" / "labeled_text" / "merged_labeled_before_2009.xlsx",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    result = process_labeled_text_before_2009(args.input_dir, args.output_file)
    print(f"Saved normalized labeled text to {result}")
