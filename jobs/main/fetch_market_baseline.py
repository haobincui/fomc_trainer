from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

from open_r1.utils.main_pipeline import coarse_vote, load_jsonl, write_json


DEFAULT_OUTPUT = ROOT / "dataset" / "external" / "market_baselines" / "market_implied_baseline.csv"
DEFAULT_REPORT = ROOT / "dataset" / "external" / "market_baselines" / "coverage_report.json"
DEFAULT_REFERENCE = ROOT / "dataset" / "processed" / "main" / "datasets" / "decision_grpo" / "test.jsonl"


DATE_COLUMN_CANDIDATES = ["meeting_date", "date", "meeting", "fomc_date"]
LABEL_COLUMN_CANDIDATES = ["predicted_label", "prediction", "action", "label", "vote"]
COARSE_PROB_ALIASES = {
    "cut_prob": ["cut_prob", "cut_probability", "cut"],
    "no_change_prob": ["no_change_prob", "no_change_probability", "hold_prob", "hold", "nochange_prob"],
    "raise_prob": ["raise_prob", "raise_probability", "raise"],
}
FINE_PATTERNS = {
    "Cut": re.compile(r"cut", re.IGNORECASE),
    "No change": re.compile(r"(no.?change|hold)", re.IGNORECASE),
    "Raise": re.compile(r"raise|hike", re.IGNORECASE),
}


def _resolve_input_file(url: str | None, source_file: Path | None, raw_cache_dir: Path) -> Path:
    if source_file is not None:
        return source_file
    if url is None:
        raise ValueError("Provide --url or --source-file")
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_cache_dir / Path(url).name
    urlretrieve(url, destination)
    return destination


def _load_any(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported source format: {path}")


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _normalize_label(value: str) -> str:
    value = str(value).strip()
    if not value:
        return ""
    if "basis points" in value or value == "No change":
        return value
    coarse = coarse_vote(value)
    if coarse == "Invalid":
        return value
    return coarse


def _extract_probability_columns(df: pd.DataFrame) -> pd.DataFrame:
    payload = df.copy()
    lower_map = {column.lower(): column for column in payload.columns}
    for output_name, aliases in COARSE_PROB_ALIASES.items():
        source_column = next((lower_map[alias.lower()] for alias in aliases if alias.lower() in lower_map), None)
        if source_column is not None:
            payload[output_name] = pd.to_numeric(payload[source_column], errors="coerce")

    if {"cut_prob", "no_change_prob", "raise_prob"}.issubset(payload.columns):
        return payload

    aggregate = {"cut_prob": 0.0, "no_change_prob": 0.0, "raise_prob": 0.0}
    found = False
    for column in payload.columns:
        value_series = pd.to_numeric(payload[column], errors="coerce")
        if value_series.notna().sum() == 0:
            continue
        for coarse_label, pattern in FINE_PATTERNS.items():
            if pattern.search(column):
                output_name = f"{coarse_vote(coarse_label)}_prob".replace(" ", "_").lower()
                aggregate[output_name] = aggregate[output_name] + value_series.fillna(0.0)
                found = True
                break

    if found:
        for key, value in aggregate.items():
            payload[key] = value
    return payload


def normalize_market_baseline(source_df: pd.DataFrame) -> pd.DataFrame:
    payload = source_df.copy()
    date_column = _find_column(payload.columns.tolist(), DATE_COLUMN_CANDIDATES)
    if date_column is None:
        raise ValueError(f"Could not infer meeting-date column from {payload.columns.tolist()}")
    payload["meeting_date"] = pd.to_datetime(payload[date_column], errors="coerce").dt.strftime("%Y-%m-%d")
    payload = payload.dropna(subset=["meeting_date"]).copy()

    label_column = _find_column(payload.columns.tolist(), LABEL_COLUMN_CANDIDATES)
    payload = _extract_probability_columns(payload)

    if label_column is not None:
        payload["predicted_label"] = payload[label_column].map(_normalize_label)
    elif {"cut_prob", "no_change_prob", "raise_prob"}.issubset(payload.columns):
        prob_frame = payload[["cut_prob", "no_change_prob", "raise_prob"]].fillna(0.0)
        payload["predicted_label"] = prob_frame.idxmax(axis=1).map(
            {
                "cut_prob": "Cut",
                "no_change_prob": "No change",
                "raise_prob": "Raise",
            }
        )
    else:
        raise ValueError("Source data must contain a prediction label or probability columns.")

    payload["predicted_coarse"] = payload["predicted_label"].map(coarse_vote)
    keep_columns = ["meeting_date", "predicted_label", "predicted_coarse"]
    for extra in ("cut_prob", "no_change_prob", "raise_prob"):
        if extra in payload.columns:
            keep_columns.append(extra)
    return payload[keep_columns].drop_duplicates(subset=["meeting_date"], keep="first").reset_index(drop=True)


def build_coverage_report(reference_file: Path, normalized_df: pd.DataFrame) -> dict:
    reference_meetings = sorted({str(row["meeting_date"])[:10] for row in load_jsonl(reference_file)})
    normalized_meetings = sorted(set(normalized_df["meeting_date"].tolist()))
    overlap = sorted(set(reference_meetings) & set(normalized_meetings))
    return {
        "reference_file": str(reference_file),
        "reference_meetings": reference_meetings,
        "normalized_meetings": normalized_meetings,
        "matched_meetings": overlap,
        "matched_count": len(overlap),
        "total_reference_meetings": len(reference_meetings),
        "coverage_rate": len(overlap) / len(reference_meetings) if reference_meetings else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize market-implied decision baselines for the main pipeline.")
    parser.add_argument("--url", help="Optional URL to download into the raw market-baseline cache.")
    parser.add_argument("--source-file", type=Path, help="Local CSV/XLSX/JSONL file to normalize.")
    parser.add_argument(
        "--raw-cache-dir",
        type=Path,
        default=ROOT / "dataset" / "external" / "market_baselines" / "raw",
    )
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--coverage-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reference-file", type=Path, default=DEFAULT_REFERENCE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_path = _resolve_input_file(args.url, args.source_file, args.raw_cache_dir)
    normalized = normalize_market_baseline(_load_any(source_path))

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.output_file, index=False)

    report = build_coverage_report(args.reference_file, normalized)
    report["source_file"] = str(source_path)
    report["output_file"] = str(args.output_file)
    args.coverage_report.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.coverage_report, report)

    print("✅ Market baseline normalized")
    print(f"- source: {source_path}")
    print(f"- output: {args.output_file}")
    print(f"- coverage: {report['matched_count']}/{report['total_reference_meetings']} ({report['coverage_rate']:.2%})")


if __name__ == "__main__":
    main()
