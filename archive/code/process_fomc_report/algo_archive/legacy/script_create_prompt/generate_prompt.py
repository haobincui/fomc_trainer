from __future__ import annotations

import random
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROMPT_ROOT = BASE_DIR.parent / "prompt_template"
FFR_ROOT = BASE_DIR.parent / "input_data" / "ffr"


def multi_df_to_markdown_tables(data_dict: Dict[str, pd.DataFrame]) -> str:
    blocks = []
    for label, df in data_dict.items():
        header = "| " + " | ".join(df.columns) + " |"
        divider = "| " + " | ".join(["-" * len(column) for column in df.columns]) + " |"
        rows = ["| " + " | ".join(str(cell) for cell in row) + " |" for _, row in df.iterrows()]
        blocks.append("\n".join([f"**{label}**", header, divider, *rows]))
    return "\n\n".join(blocks)


PROMPT_WITH_REF = {
    "1": (PROMPT_ROOT / "input_prompt_with_reference" / "input_prompt_with_reference_1.md").read_text(),
    "2": (PROMPT_ROOT / "input_prompt_with_reference" / "input_prompt_with_reference_2.md").read_text(),
    "3": (PROMPT_ROOT / "input_prompt_with_reference" / "input_prompt_with_reference_3.md").read_text(),
}

PROMPT_WITHOUT_REF = {
    "1": (PROMPT_ROOT / "input_prompt_without_reference" / "input_prompt_without_reference_1.md").read_text(),
    "2": (PROMPT_ROOT / "input_prompt_without_reference" / "input_prompt_without_reference_2.md").read_text(),
    "3": (PROMPT_ROOT / "input_prompt_without_reference" / "input_prompt_without_reference_3.md").read_text(),
}


def get_random_prompt(prompt_dict: dict[str, str]) -> str:
    return prompt_dict[random.choice(list(prompt_dict.keys()))]


def generate_fomc_prompt(
    meeting_date: str,
    topic: str,
    section_style: str,
    table_str: str,
    data_label: str,
    reference_excerpt: str = "",
) -> str:
    prompt_dict = PROMPT_WITH_REF if reference_excerpt else PROMPT_WITHOUT_REF
    prompt = get_random_prompt(prompt_dict)
    return prompt.format(
        meeting_date=meeting_date,
        topic=topic,
        section_style=section_style,
        emphasized_label=data_label,
        table_str=table_str.strip(),
        reference_excerpt=reference_excerpt.strip(),
    )


def normalize_date(date_str) -> date:
    if isinstance(date_str, date):
        return date_str
    if re.match(r"^\d{4}Q[1-4]$", str(date_str)):
        year = int(date_str[:4])
        quarter = int(date_str[-1])
        month = (quarter - 1) * 3 + 1
        return date(year, month, 1)
    if re.match(r"^\d{4}M\d{2}$", str(date_str)):
        year = int(date_str[:4])
        month = int(date_str[-2:])
        return date(year, month, 1)
    if re.match(r"^\d{4}-\d{2}-\d{2}$", str(date_str)):
        year, month, day = map(int, date_str.split("-"))
        return date(year, month, day)
    if re.match(r"^[A-Za-z]{3}-\d{4}$", str(date_str)):
        return datetime.strptime(date_str, "%b-%Y").date().replace(day=1)
    raise ValueError(f"Unrecognized date format: {date_str}")


def process_unemployment_rate_data(df: pd.DataFrame) -> pd.DataFrame:
    if "observation_date" in df.columns:
        return df
    df = df.copy()
    df["combined"] = df["Year"].astype(str) + df["Period"].astype(str)
    df["observation_date"] = df["combined"].apply(normalize_date)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df[df["observation_date"].notna() & df["Value"].notna()].copy()


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "observation_date" in df.columns:
        df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce").dt.date
        return df[df["observation_date"].notna()].copy()
    for column in df.columns:
        converted = pd.to_datetime(df[column], errors="coerce")
        if converted.notna().sum() > 0:
            df["observation_date"] = converted.dt.date
            df = df[df["observation_date"].notna()].copy()
            df.drop(columns=[column], inplace=True)
            return df
    raise ValueError("No valid date column found in DataFrame.")


def filter_last_two_years_data(df: pd.DataFrame, reference_date: str | date) -> pd.DataFrame:
    ref_date = pd.to_datetime(reference_date).date()
    two_years_ago = ref_date.replace(year=ref_date.year - 2, day=1)
    df = df.copy()
    df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.date
    return df[(df["observation_date"] > two_years_ago) & (df["observation_date"] <= ref_date)]


def load_effr() -> tuple[pd.DataFrame, pd.DataFrame]:
    effr_daily = pd.read_excel(FFR_ROOT / "effr.xlsx")
    effr_monthly = pd.read_csv(FFR_ROOT / "FEDFUNDS.csv")
    effr_daily["Effective Date"] = pd.to_datetime(effr_daily["Effective Date"], format="%m/%d/%Y", errors="coerce")
    effr_monthly["Effective Date"] = pd.to_datetime(effr_monthly["Effective Date"], format="%Y/%m/%d", errors="coerce")
    return effr_daily, effr_monthly


EFFR_DAILY, EFFR_MONTHLY = load_effr()


def map_rate_change_to_label(rate_change: int) -> str:
    thresholds = [100, 75, 50, 25]
    if -12 <= rate_change <= 12:
        return "No change"
    if rate_change > 12:
        for threshold in thresholds:
            if rate_change >= threshold:
                return f"Raise by {threshold} basis points"
        return "Raise by 25 basis points"
    for threshold in thresholds:
        if rate_change <= -threshold:
            return f"Cut by {threshold} basis points"
    return "Cut by 25 basis points"


def load_target_rate(input_date) -> tuple[str, float]:
    input_date = pd.Timestamp(input_date) + pd.Timedelta(days=1)
    if input_date == pd.Timestamp("2020-03-15 00:00:00"):
        input_date = pd.Timestamp("2020-03-16 00:00:00")

    result = EFFR_DAILY[EFFR_DAILY["Effective Date"] == input_date]
    if result.empty:
        result = EFFR_MONTHLY[EFFR_MONTHLY["Effective Date"] == input_date.replace(day=1)]
    row = result.iloc[0]
    return map_rate_change_to_label(int(row["Rate Change (bps)"])), float(row["Rate (%)"])
