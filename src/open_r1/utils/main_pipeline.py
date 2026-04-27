from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


MAIN_SPLIT_SEED = 42
COARSE_LABELS = ["Cut", "No change", "Raise"]
RAW_ACTION_LABELS = [
    "Cut by 75 basis points",
    "Cut by 50 basis points",
    "Cut by 25 basis points",
    "No change",
    "Raise by 25 basis points",
    "Raise by 50 basis points",
    "Raise by 75 basis points",
]

MEETING_DATE_PATTERNS = [
    re.compile(r"meeting on \*\*(\d{4}-\d{2}-\d{2})(?: \d{2}:\d{2}:\d{2})?\*\*", re.IGNORECASE),
    re.compile(r"meeting scheduled for \*\*(\d{4}-\d{2}-\d{2})\*\*", re.IGNORECASE),
    re.compile(r"for the \*\*(\d{4}-\d{2}-\d{2})\*\* meeting", re.IGNORECASE),
]

SECTION_PATTERNS = [
    re.compile(r"tone and style of the \*\*(.*?)\*\* section", re.IGNORECASE | re.DOTALL),
    re.compile(r"aligned with the tone and structure of the \*\*(.*?)\*\* section", re.IGNORECASE | re.DOTALL),
    re.compile(r"style of the \*\*(.*?)\*\* section", re.IGNORECASE | re.DOTALL),
]

TOPIC_PATTERNS = [
    re.compile(r"recent trends in \*\*(.*?)\*\* and related indicators", re.IGNORECASE | re.DOTALL),
    re.compile(r"patterns and developments in \*\*(.*?)\*\*", re.IGNORECASE | re.DOTALL),
    re.compile(r"latest patterns and developments in \*\*(.*?)\*\*", re.IGNORECASE | re.DOTALL),
]

PROVIDED_DATA_PATTERN = re.compile(
    r"\n\{\n(Indicators:.*)\n\}\n\n(?:Model your tone|Write in the tone|Keep your response|Ensure your writing)",
    re.DOTALL | re.IGNORECASE,
)

REFERENCE_EXCERPT_PATTERN = re.compile(
    r"following excerpt from the FOMC minutes:\n\n\{\n(.*?)\n\}\n\nKeep your response",
    re.DOTALL | re.IGNORECASE,
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_meeting_date(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value).strip()[:10]


def extract_first(text: str, patterns: list[re.Pattern[str]]) -> str:
    if not isinstance(text, str):
        return ""
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return ""


def extract_meeting_date(text: str, fallback=None) -> str:
    fallback_value = normalize_meeting_date(fallback)
    if fallback_value:
        return fallback_value
    return extract_first(text, MEETING_DATE_PATTERNS)


def extract_section_name(text: str, fallback=None) -> str:
    if fallback:
        return str(fallback).strip()
    return extract_first(text, SECTION_PATTERNS)


def extract_topic(text: str, fallback=None) -> str:
    if fallback:
        return str(fallback).strip()
    return extract_first(text, TOPIC_PATTERNS)


def extract_provided_data(prompt: str) -> str:
    if not isinstance(prompt, str):
        return ""
    match = PROVIDED_DATA_PATTERN.search(prompt)
    if match:
        return match.group(1).strip()
    return ""


def extract_reference_excerpt(prompt: str) -> str:
    if not isinstance(prompt, str):
        return ""
    match = REFERENCE_EXCERPT_PATTERN.search(prompt)
    if match:
        return match.group(1).strip()
    return ""


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_meeting_random_split(
    meeting_dates: Iterable[str],
    *,
    seed: int = MAIN_SPLIT_SEED,
) -> dict[str, list[str]]:
    unique_sorted = sorted({normalize_meeting_date(value) for value in meeting_dates if normalize_meeting_date(value)})
    rng = random.Random(seed)
    shuffled = unique_sorted[:]
    rng.shuffle(shuffled)

    train_count = math.floor(len(shuffled) * 0.8)
    remainder = len(shuffled) - train_count
    eval_count = remainder // 2
    test_count = remainder - eval_count

    return {
        "train": sorted(shuffled[:train_count]),
        "eval": sorted(shuffled[train_count : train_count + eval_count]),
        "test": sorted(shuffled[train_count + eval_count : train_count + eval_count + test_count]),
    }


def invert_split_map(split_map: dict[str, list[str]]) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for split, meetings in split_map.items():
        for meeting_date in meetings:
            if meeting_date in assignment:
                raise ValueError(f"Duplicate meeting assignment detected for {meeting_date}")
            assignment[meeting_date] = split
    return assignment


def attach_split(records: Iterable[dict], split_map: dict[str, list[str]]) -> list[dict]:
    meeting_to_split = invert_split_map(split_map)
    output: list[dict] = []
    for row in records:
        meeting_date = normalize_meeting_date(row.get("meeting_date"))
        split = meeting_to_split.get(meeting_date)
        if not split:
            raise ValueError(f"Meeting date {meeting_date!r} is not covered by the split manifest.")
        updated = dict(row)
        updated["meeting_date"] = meeting_date
        updated["split"] = split
        output.append(updated)
    return output


def group_by_split(records: Iterable[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {"train": [], "eval": [], "test": []}
    for row in records:
        split = row["split"]
        if split not in grouped:
            raise ValueError(f"Unexpected split name {split!r}")
        grouped[split].append(row)
    return grouped


def build_row_manifest(records: Iterable[dict], extra_keys: Iterable[str] | None = None) -> list[dict]:
    keys = list(extra_keys or [])
    manifest: list[dict] = []
    for row in records:
        entry = {
            "meeting_date": normalize_meeting_date(row.get("meeting_date")),
            "split": row.get("split"),
            "index": row.get("index"),
            "source_dataset": row.get("source_dataset"),
            "prompt_hash": row.get("prompt_hash"),
        }
        for key in keys:
            entry[key] = row.get(key)
        manifest.append(entry)
    return manifest


def sort_records_for_reproducibility(records: Iterable[dict]) -> list[dict]:
    return sorted(
        records,
        key=lambda row: (
            normalize_meeting_date(row.get("meeting_date")),
            int(row.get("index", 0)),
            row.get("prompt_hash") or sha1_text(row.get("prompt", "")),
        ),
    )


def split_analysis_train_for_grpo(
    train_rows: Iterable[dict],
    *,
    seed: int = MAIN_SPLIT_SEED,
    primary_ratio: float = 0.2,
    replay_ratio: float = 0.1,
) -> tuple[list[dict], dict]:
    ordered_rows = sort_records_for_reproducibility(train_rows)
    indexed_rows = list(enumerate(ordered_rows))
    rng = random.Random(seed)
    rng.shuffle(indexed_rows)

    primary_count = max(1, math.floor(len(indexed_rows) * primary_ratio))
    remaining = indexed_rows[primary_count:]
    replay_count = max(1, math.floor(len(ordered_rows) * replay_ratio))
    replay_count = min(replay_count, len(remaining))

    selected_primary = {idx for idx, _ in indexed_rows[:primary_count]}
    selected_replay = {idx for idx, _ in remaining[:replay_count]}

    combined_rows: list[dict] = []
    for idx, row in enumerate(ordered_rows):
        if idx in selected_primary:
            updated = dict(row)
            updated["analysis_grpo_role"] = "grpo_primary"
            combined_rows.append(updated)
        elif idx in selected_replay:
            updated = dict(row)
            updated["analysis_grpo_role"] = "sft_replay"
            combined_rows.append(updated)

    selection_manifest = {
        "seed": seed,
        "primary_ratio": primary_ratio,
        "replay_ratio": replay_ratio,
        "train_row_count": len(ordered_rows),
        "primary_row_count": sum(row["analysis_grpo_role"] == "grpo_primary" for row in combined_rows),
        "replay_row_count": sum(row["analysis_grpo_role"] == "sft_replay" for row in combined_rows),
    }
    return combined_rows, selection_manifest


def deduplicate_decision_rows(rows: Iterable[dict]) -> tuple[list[dict], dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    issues: list[dict] = []

    for row in rows:
        normalized = dict(row)
        normalized["meeting_date"] = extract_meeting_date(
            normalized.get("prompt", ""),
            normalized.get("meeting_date"),
        )
        if not normalized["meeting_date"]:
            raise ValueError("Decision row is missing a parseable meeting_date.")
        normalized["prompt_hash"] = normalized.get("prompt_hash") or sha1_text(normalized.get("prompt", ""))
        grouped[normalized["meeting_date"]].append(normalized)

    deduplicated: list[dict] = []
    for meeting_date, meeting_rows in sorted(grouped.items()):
        label_pairs = {
            (
                str(row.get("rate_change")).strip(),
                float(row.get("current_rate")) if row.get("current_rate") is not None else None,
            )
            for row in meeting_rows
        }
        if len(label_pairs) > 1:
            issues.append(
                {
                    "meeting_date": meeting_date,
                    "problem": "inconsistent_target_fields",
                    "variants": sorted(label_pairs),
                }
            )
            continue

        ordered_rows = sorted(
            meeting_rows,
            key=lambda row: (
                0 if str(row.get("response", "")).strip() else 1,
                int(row.get("index", 0)),
                row["prompt_hash"],
            ),
        )
        canonical = dict(ordered_rows[0])
        canonical["source_dataset"] = canonical.get("source_dataset") or "decision_grpo_raw"
        canonical["deduplicated_from"] = len(meeting_rows)
        deduplicated.append(canonical)

    audit = {
        "input_rows": sum(len(items) for items in grouped.values()),
        "unique_meetings": len(grouped),
        "deduplicated_rows": len(deduplicated),
        "duplicate_meetings": sum(len(items) > 1 for items in grouped.values()),
        "issues": issues,
    }

    if issues:
        raise ValueError(f"Decision deduplication failed with {len(issues)} inconsistent meeting groups.")

    return deduplicated, audit


def coarse_vote(label: str | None) -> str:
    if not label:
        return "Invalid"
    label = str(label).strip()
    if label.startswith("Raise"):
        return "Raise"
    if label.startswith("Cut"):
        return "Cut"
    if label.startswith("No change"):
        return "No change"
    return "Invalid"


def compute_confusion(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, dict[str, int]]:
    confusion = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for actual, predicted in zip(y_true, y_pred):
        if actual in labels and predicted in labels:
            confusion[actual][predicted] += 1
    return confusion


def compute_classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("Cannot compute metrics on an empty label set")

    confusion = compute_confusion(y_true, y_pred, labels)
    accuracy = sum(actual == predicted for actual, predicted in zip(y_true, y_pred)) / len(y_true)

    recalls = []
    f1_scores = []
    for label in labels:
        tp = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == label and predicted == label)
        fn = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == label and predicted != label)
        fp = sum(1 for actual, predicted in zip(y_true, y_pred) if actual != label and predicted == label)

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "n": len(y_true),
        "accuracy": accuracy,
        "balanced_accuracy": sum(recalls) / len(labels),
        "macro_f1": sum(f1_scores) / len(labels),
        "confusion_matrix": confusion,
        "class_counts": dict(Counter(y_true)),
        "invalid_predictions": sum(predicted not in labels for predicted in y_pred),
    }
