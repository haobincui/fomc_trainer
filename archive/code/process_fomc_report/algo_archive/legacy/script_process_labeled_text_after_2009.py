from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from label_html_files.process_label_response.process_label_response import (
    LABEL_TO_SECTION,
    extract_date_from_filename,
    extract_labels_from_text,
)


BASE_DIR = Path(__file__).resolve().parent

RELABEL_MAP = {
    "Capital Adequacy Ratio": "Bank Capital",
    "Crude Oil Prices": "Commodity Prices",
    "PCE Price Index": "Personal Consumption Expenditures (PCE)",
    "PCE Price Index [LABEL]Consumer Price Index (CPI)": "Personal Consumption Expenditures (PCE), Consumer Price Index (CPI)",
    "Total Assets of Federal Reserve": "Federal Reserve Balance Sheet",
    "Total Liabilities of Federal Reserve": "Federal Reserve Balance Sheet",
    "Reserve Balances with Federal": "Federal Reserve Balance Sheet",
    "Nonfarm Payrolls": "Labour Market",
    "Job Openings (JOLTS)": "Labour Market",
    "Labor Force Participation Rate": "Labour Market",
    "Holdings of Mortgage-Backed Securities (MBS)": "Federal Reserve Balance Sheet",
    "Non-Performing Loans (NPLs)": "Bank Credit to Private Sector",
    "Mortgage Applications": "Bank Credit to Private Sector",
    "Consumer Credit": "Bank Credit to Private Sector",
}


def _clean_label(text: str) -> str:
    text = text.replace("{", "").replace("}", "").replace('"', "").replace("'", "")
    return RELABEL_MAP.get(text, text)


def _load_labeled_frames(input_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for file in sorted(input_dir.glob("*.xlsx")):
        frame = pd.read_excel(file)
        frame["date"] = extract_date_from_filename(file.name)
        frame["file_name"] = file.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No labeled Excel files found in {input_dir}")
    return pd.concat(frames, ignore_index=True)


def process_labeled_text_after_2009(
    input_dir: Path = BASE_DIR / "output" / "after_2009",
    output_file: Path = BASE_DIR / "output" / "labeled_text" / "merged_labeled_after_2009.xlsx",
) -> Path:
    df = _load_labeled_frames(input_dir)
    df = df[df["response"].notna()].copy()
    df = df[df["response"].astype(str).str.strip() != ""].copy()
    df = df[df["response"] != "Error"].copy()

    df["new_label"] = df["response"].apply(extract_labels_from_text)
    df["new_label"] = df["new_label"].apply(_clean_label)
    df = df[~df["new_label"].isin(["Not-Found", "Non-Core", "Delete"])].copy()
    df["label_type"] = df["new_label"].map(LABEL_TO_SECTION).fillna("other")
    df["relabel"] = df["new_label"].apply(lambda value: RELABEL_MAP.get(value, value))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(by=["date", "file_name", "line_id"], inplace=True)
    df.to_excel(output_file, index=False)
    return output_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge and normalize labeled after-2009 FOMC text.")
    parser.add_argument("--input-dir", type=Path, default=BASE_DIR / "output" / "after_2009")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=BASE_DIR / "output" / "labeled_text" / "merged_labeled_after_2009.xlsx",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    result = process_labeled_text_after_2009(args.input_dir, args.output_file)
    print(f"Saved normalized labeled text to {result}")
