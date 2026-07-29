from __future__ import annotations

import re
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .quality import canonicalize_section_name, canonicalize_topic


def _indicator_file(input_root: Path) -> Path:
    return input_root / "indicators.xlsx"


@lru_cache(maxsize=4)
def load_label_lookup(indicator_file: str) -> dict[str, str]:
    indicator_df = pd.read_excel(indicator_file)
    return dict(zip(indicator_df["indicators"], indicator_df["sections"]))


RELABEL_MAP = {
    "Capital Adequacy Ratio": "Bank Capital",
    "Crude Oil Prices": "Commodity Prices",
    "PCE Price Index": "Personal Consumption Expenditures (PCE)",
    "PCE Price Index [LABEL]Consumer Price Index (CPI)": "Personal Consumption Expenditures (PCE), Consumer Price Index (CPI)",
    "Total Assets of Federal Reserve": "Federal Reserve Balance Sheet",
    "Total Liabilities of Federal Reserve": "Federal Reserve Balance Sheet",
    "Reserve Balances with Federal": "Federal Reserve Balance Sheet",
    "Reserve Balances with Fed": "Federal Reserve Balance Sheet",
    "Nonfarm Payrolls": "Labour Market",
    "Job Openings (JOLTS)": "Labour Market",
    "Labor Force Participation Rate": "Labour Market",
    "Holdings of Mortgage-Backed Securities (MBS)": "Federal Reserve Balance Sheet",
    "Mortgage-Backed Securities": "Federal Reserve Balance Sheet",
    "Non-Performing Loans (NPLs)": "Bank Credit to Private Sector",
    "Mortgage Applications": "Bank Credit to Private Sector",
    "Consumer Credit": "Bank Credit to Private Sector",
}


def create_label_prompt(section: str | None, text: str, indicator_file: Path) -> str:
    label_to_section = load_label_lookup(str(indicator_file))
    label_list = ", ".join(f'"{label}"' for label in label_to_section)
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

Paragraph: "{text}"
"""
    return base_instruction + label_instruction


def extract_date_from_filename(filename: str) -> date | None:
    match = re.search(r"\d{8}", filename)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(0), "%Y%m%d").date()
    except ValueError:
        return None


def process_len(text: str) -> bool:
    return len(str(text).replace(" ", "")) <= 200


def preprocess_label(section_name: str, text: str) -> str:
    section_name = str(section_name).strip().lower()
    text = str(text).strip().lower()
    non_core_sections = {
        "notation vote",
        "committee policy action",
        "voting",
        "attendance",
        "secretary",
        "summary",
        "approval of minutes",
        "meeting adjourned",
    }
    if any(non_core in section_name for non_core in non_core_sections):
        return "Pre-Non-Core-Section"
    if section_name == text:
        return "Pre-Non-Core-Section"
    if not text or len(text) < 50:
        return "Limit-text"
    return "To-Label"


def _process_label_text(text: str) -> str:
    pattern = r"\[LABEL\](.*?)\[/LABEL\]|\[LABEL\]\{(.*?)\}|\[LABEL\]\s*(.*?)(?=\[|$)"
    matches = re.findall(pattern, str(text), flags=re.DOTALL)
    labels = [group for match in matches for group in match if group.strip()]
    labels = [RELABEL_MAP.get(label.strip().strip("{}[]\"'"), label.strip().strip("{}[]\"'")) for label in labels]
    labels = list(dict.fromkeys(filter(None, labels)))
    if not labels:
        return "Not-Found"
    return ", ".join(labels)


def merge_labeled_after_2009(frame: pd.DataFrame, indicator_file: Path) -> pd.DataFrame:
    label_to_section = load_label_lookup(str(indicator_file))
    cleaned = frame.copy()
    cleaned = cleaned[cleaned["response"].notna()].copy()
    cleaned = cleaned[cleaned["response"].astype(str).str.strip() != ""].copy()
    cleaned = cleaned[cleaned["response"] != "Error"].copy()
    cleaned["new_label"] = cleaned["response"].apply(_process_label_text)
    cleaned = cleaned[~cleaned["new_label"].isin(["Not-Found", "Non-Core", "Delete"])].copy()
    cleaned["label_type"] = cleaned["new_label"].map(label_to_section).fillna("other")
    cleaned["relabel"] = cleaned["new_label"].apply(canonicalize_topic)
    cleaned["section_name"] = cleaned["section_name"].apply(canonicalize_section_name)
    return cleaned


def merge_labeled_before_2009(frame: pd.DataFrame, indicator_file: Path, file_name: str) -> pd.DataFrame:
    label_to_section = load_label_lookup(str(indicator_file))
    cleaned = frame.copy()
    cleaned = cleaned[~cleaned["raw_text"].astype(str).apply(process_len)].copy()
    cleaned["label"] = cleaned["response"].apply(_process_label_text)
    cleaned = cleaned[~cleaned["label"].isin(["Not-Found", "Non-Core", "Delete"])].copy()
    cleaned["label_type"] = cleaned["label"].map(label_to_section).fillna("other")
    cleaned["relabel"] = cleaned["label"].apply(canonicalize_topic)
    cleaned["date"] = extract_date_from_filename(file_name)
    cleaned["section_name"] = cleaned["section_name"].apply(canonicalize_section_name)
    return cleaned
