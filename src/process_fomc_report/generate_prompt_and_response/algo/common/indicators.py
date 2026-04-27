from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
import warnings

import pandas as pd

from .paths import INPUT_ROOT


NON_CORE_TOPICS = {"Other", "Non-Core", "Delete", ""}


def normalize_date(date_value) -> date:
    if isinstance(date_value, date):
        return date_value
    text = str(date_value).strip()
    if pd.isna(date_value) or not text:
        raise ValueError(f"Unrecognized date value: {date_value!r}")
    if len(text) == 7 and text[4] == "Q":
        year = int(text[:4])
        quarter = int(text[-1])
        month = (quarter - 1) * 3 + 1
        return date(year, month, 1)
    if len(text) == 7 and text[4] == "M":
        return date(int(text[:4]), int(text[-2:]), 1)
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        return datetime.strptime(text, "%Y-%m-%d").date()
    if len(text) == 8 and text.isdigit():
        return datetime.strptime(text, "%Y%m%d").date()
    if len(text) == 8 and text[:3].isalpha():
        return datetime.strptime(text, "%b-%Y").date().replace(day=1)
    return pd.Timestamp(date_value).date()


def process_unemployment_rate_data(frame: pd.DataFrame) -> pd.DataFrame:
    if "observation_date" in frame.columns:
        return frame
    frame = frame.copy()
    frame["combined"] = frame["Year"].astype(str) + frame["Period"].astype(str)
    frame["observation_date"] = frame["combined"].apply(normalize_date)
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    return frame[frame["observation_date"].notna() & frame["Value"].notna()].copy()


def process_data(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "observation_date" in frame.columns:
        frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce").dt.date
        return frame[frame["observation_date"].notna()].copy()
    for column in frame.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            converted = pd.to_datetime(frame[column], errors="coerce")
        if converted.notna().sum() > 0:
            frame["observation_date"] = converted.dt.date
            frame = frame[frame["observation_date"].notna()].copy()
            frame.drop(columns=[column], inplace=True)
            return frame
    raise ValueError("No valid date column found in DataFrame.")


def filter_last_two_years_data(frame: pd.DataFrame, reference_date: str | date) -> pd.DataFrame:
    ref_date = pd.Timestamp(reference_date).date()
    lower_bound = (pd.Timestamp(ref_date) - pd.DateOffset(years=2)).replace(day=1).date()
    filtered = frame.copy()
    filtered["observation_date"] = pd.to_datetime(filtered["observation_date"]).dt.date
    return filtered[(filtered["observation_date"] > lower_bound) & (filtered["observation_date"] <= ref_date)]


def multi_df_to_markdown_tables(data_dict: dict[str, pd.DataFrame]) -> str:
    blocks: list[str] = []
    for label, frame in data_dict.items():
        header = "| " + " | ".join(frame.columns) + " |"
        divider = "| " + " | ".join(["-" * len(column) for column in frame.columns]) + " |"
        rows = ["| " + " | ".join(str(cell) for cell in row) + " |" for _, row in frame.iterrows()]
        blocks.append("\n".join([f"**{label}**", header, divider, *rows]))
    return "\n\n".join(blocks)


def load_effr(ffr_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    effr_daily = pd.read_excel(ffr_root / "effr.xlsx")
    effr_monthly = pd.read_csv(ffr_root / "FEDFUNDS.csv")
    effr_daily["Effective Date"] = pd.to_datetime(effr_daily["Effective Date"], format="%m/%d/%Y", errors="coerce")
    effr_monthly["Effective Date"] = pd.to_datetime(effr_monthly["Effective Date"], format="%Y/%m/%d", errors="coerce")
    return effr_daily, effr_monthly


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


@dataclass
class IndicatorBundle:
    data_label: str
    provided_data: str
    indicator_names: list[str]
    source_files: list[str]
    missing_indicators: list[str]


class IndicatorRepository:
    def __init__(self, input_root: Path | None = None) -> None:
        self.input_root = input_root or INPUT_ROOT
        self.us_data_root = self.input_root / "us_data"
        self.ffr_root = self.input_root / "ffr"

    @lru_cache(maxsize=1)
    def get_input_data_map(self) -> dict[str, dict[str, pd.DataFrame]]:
        input_data_map: dict[str, dict[str, pd.DataFrame]] = {}
        for indicator_dir in sorted(self.us_data_root.iterdir()):
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
                processed = (
                    process_unemployment_rate_data(frame)
                    if indicator_name == "Unemployment Rate"
                    else process_data(frame)
                )
                input_data_map[indicator_name][data_file.name] = processed
        return input_data_map

    @lru_cache(maxsize=1)
    def get_effr_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return load_effr(self.ffr_root)

    def load_target_rate(self, input_date) -> tuple[str, float]:
        effr_daily, effr_monthly = self.get_effr_data()
        effective_date = pd.Timestamp(input_date) + pd.Timedelta(days=1)
        if effective_date == pd.Timestamp("2020-03-15 00:00:00"):
            effective_date = pd.Timestamp("2020-03-16 00:00:00")
        result = effr_daily[effr_daily["Effective Date"] == effective_date]
        if result.empty:
            result = effr_monthly[effr_monthly["Effective Date"] == effective_date.replace(day=1)]
        row = result.iloc[0]
        return map_rate_change_to_label(int(row["Rate Change (bps)"])), float(row["Rate (%)"])

    def build_indicator_bundle(self, topic: str, meeting_date: str) -> IndicatorBundle:
        input_data_map = self.get_input_data_map()
        data_labels: list[str] = []
        source_files: list[str] = []
        missing_indicators: list[str] = []
        table_blocks: list[str] = []

        for indicator in [item.strip() for item in str(topic).split(",")]:
            if indicator in NON_CORE_TOPICS:
                continue
            if indicator not in input_data_map:
                missing_indicators.append(indicator)
                continue
            filtered_frames = {
                file_name: filter_last_two_years_data(frame, meeting_date)
                for file_name, frame in input_data_map[indicator].items()
            }
            source_files.extend(filtered_frames)
            data_labels.extend(filtered_frames)
            table_blocks.append(
                f"Indicators: {indicator}\n\n{multi_df_to_markdown_tables(filtered_frames)}"
            )

        return IndicatorBundle(
            data_label=", ".join(dict.fromkeys(data_labels)),
            provided_data="\n\n".join(table_blocks).strip(),
            indicator_names=[item.strip() for item in str(topic).split(",") if item.strip()],
            source_files=list(dict.fromkeys(source_files)),
            missing_indicators=missing_indicators,
        )
