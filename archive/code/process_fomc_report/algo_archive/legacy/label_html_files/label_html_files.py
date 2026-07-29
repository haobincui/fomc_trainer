from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import TypedDict

import pandas as pd
from tqdm import tqdm

from get_response_from_llm.get_response import get_response


BASE_DIR = Path(__file__).resolve().parent
INDICATOR_FILE = BASE_DIR.parent / "input_data" / "indicators.xlsx"
LABEL_DF = pd.read_excel(INDICATOR_FILE)
LABEL_TO_SECTION = dict(zip(LABEL_DF["indicators"], LABEL_DF["sections"]))


def create_prompt(section: str | None, text: str) -> str:
    label_list = ", ".join(f'"{label}"' for label in LABEL_TO_SECTION)
    base_instruction = "You are analyzing a segment from the FOMC minutes.\n\n"
    if section:
        base_instruction += f'The section is titled: "{section}".\n\n'

    label_instruction = f"""
Label the following paragraph with all applicable labels:

[{label_list}].

Instructions:
- If the paragraph is not part of the core FOMC analysis, label it as "Delete".
- If no suitable label exists, use "Other".
- Multiple labels must be separated with commas.
- Wrap each label using [LABEL]{{label}} and the explanation using [Explanation]{{explanation}}.

Special cases:
1. Agency debt or agency MBS purchases -> Mortgage-Backed Securities.
2. Asset purchase programs -> Government Purchases.
3. CPI -> Consumer Price Index (CPI), PCE -> Personal Consumption Expenditures (PCE).
4. Consumer sentiment/confidence -> Consumer Confidence Index.
5. Bank credit/lending/loans/deposits -> Bank Credit to Private Sector.
6. Inflation compensation -> Consumer Price Index (CPI).
7. Inventory investment/levels -> Industrial Production.
8. LIBOR-OIS spread -> Overnight Rate.

Paragraph: "{text}"
"""
    return base_instruction + label_instruction


def is_meeting_date_section(section_name: str) -> bool:
    pattern = r"^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}([–-]\d{1,2})?,\s+\d{4}$"
    return bool(re.match(pattern, section_name.strip(), flags=re.IGNORECASE))


def is_meeting_opening_line(text: str) -> bool:
    return bool(re.match(r"^a meeting of the federal open market committee was held in .*?", text.strip(), flags=re.IGNORECASE))


def is_name_line(text: str) -> bool:
    return bool(re.match(r"^\s*(Mr|Ms|Mrs|Messrs|Mses)\.?\b", text.strip(), flags=re.IGNORECASE))


def preprocess_label(section_name: str, text: str) -> str:
    section_name = section_name.strip().lower()
    text = text.strip().lower()

    non_core_sections = {
        "notation vote",
        "committee policy action",
        "voting",
        "attendance",
        "secretary",
        "summary",
        "approval of minutes",
        "meeting adjourned",
        "annual organizational matters",
        "reports",
        "reporting forms",
        "research, reports, & committees",
        "working papers and notes",
        "data, models and tools",
        "bank assets and liabilities",
    }

    if any(non_core in section_name for non_core in non_core_sections):
        return "Pre-Non-Core-Section"
    if section_name == text:
        return "Pre-Non-Core-Section"
    if not text or len(text) < 50:
        return "Limit-text"
    if is_meeting_date_section(section_name) or is_meeting_date_section(text):
        return "Meeting Date"
    if is_name_line(text):
        return "Name Line"
    if is_meeting_opening_line(text):
        return "Meeting Opening Line"
    return "To-Label"


class ResultDict(TypedDict):
    line_id: int
    section_name: str
    raw_text: str
    label: str
    label_type: str
    explanation: str
    reason: str
    response: str


def generate_label(index: int, section: str, line: str, model_name: str | None = None) -> ResultDict:
    pre_label = preprocess_label(section, line)
    if pre_label != "To-Label":
        return ResultDict(
            line_id=index + 1,
            section_name=section,
            raw_text=line,
            label="Pre-Non-Core",
            label_type="Pre-Non-Core",
            explanation="Pre-Non-Core",
            reason=pre_label,
            response="Pre-Non-Core",
        )

    prompt = create_prompt(section, line)
    response, reason = get_response(prompt, model_name=model_name)
    match = re.search(r"\[LABEL\](.+?)\s*\[Explanation\](.+)", response, re.DOTALL | re.IGNORECASE)
    if match:
        label = match.group(1).strip()
        explanation = match.group(2).strip()
    else:
        label = "Fail to match"
        explanation = "Fail to match"

    return ResultDict(
        line_id=index + 1,
        section_name=section,
        raw_text=line,
        label=label,
        label_type=LABEL_TO_SECTION.get(label, "other"),
        explanation=explanation,
        reason=reason,
        response=response,
    )


def label_html_file(
    input_file_path: Path,
    output_dir: Path,
    sleep_seconds: float = 0.0,
    model_name: str | None = None,
) -> Path:
    frame = pd.read_excel(input_file_path)
    has_section = "section_name" in frame.columns
    columns = ["section_name", "details"] if has_section else ["details"]
    lines = frame[columns].dropna().astype(str).values.tolist()
    if not has_section:
        lines = [["", line[0]] for line in lines]

    results = []
    for i, (section, line) in enumerate(tqdm(lines, desc=input_file_path.name)):
        results.append(generate_label(i, section, line, model_name=model_name))
        if sleep_seconds:
            time.sleep(sleep_seconds)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_file_path.stem}_labeled.xlsx"
    pd.DataFrame(results).to_excel(output_path, index=False)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label FOMC paragraph Excel files with indicator tags.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR.parent / "output" / "after_2009")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--model-name", type=str, default=None)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    output_path = label_html_file(
        args.input_file,
        args.output_dir,
        sleep_seconds=args.sleep_seconds,
        model_name=args.model_name,
    )
    print(f"Saved labeled file to {output_path}")
