import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd


DATE_PATTERNS = [
    re.compile(r"meeting on \*\*(\d{4}-\d{2}-\d{2})"),
    re.compile(r"meeting scheduled for \*\*(\d{4}-\d{2}-\d{2})"),
    re.compile(r"for the \*\*(\d{4}-\d{2}-\d{2})\*\* meeting"),
]
LABELS = ["Cut", "No change", "Raise"]


def _extract_meeting_date(row: pd.Series) -> str | None:
    if "meeting_date" in row and pd.notna(row["meeting_date"]):
        return str(row["meeting_date"])[:10]
    prompt = row.get("prompt", "")
    if not isinstance(prompt, str):
        return None
    for pattern in DATE_PATTERNS:
        match = pattern.search(prompt)
        if match:
            return match.group(1)
    return None


def _coarse_vote(value: str | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "Invalid"
    value = str(value).strip()
    if value.startswith("Raise"):
        return "Raise"
    if value.startswith("Cut"):
        return "Cut"
    if value.startswith("No change"):
        return "No change"
    return "Invalid"


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.copy()
    ordered["has_generated_vote"] = ordered["generated_vote"].notna()
    ordered["vote_rank"] = ordered["has_generated_vote"].map({True: 0, False: 1})
    ordered = ordered.sort_values(["meeting_date", "vote_rank", "index"])
    return ordered.drop_duplicates(subset=["meeting_date"], keep="first").reset_index(drop=True)


def _classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    confusion = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}

    for actual, predicted in zip(y_true, y_pred):
        if actual in LABELS and predicted in LABELS:
            confusion[actual][predicted] += 1

    accuracy = sum(actual == predicted for actual, predicted in zip(y_true, y_pred)) / len(y_true)

    recalls = []
    f1_scores = []
    for label in LABELS:
        tp = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == label and predicted == label)
        fn = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == label and predicted != label)
        fp = sum(1 for actual, predicted in zip(y_true, y_pred) if actual != label and predicted == label)

        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        recalls.append(recall)
        f1_scores.append(f1)

    return {
        "n_meetings": len(y_true),
        "accuracy": accuracy,
        "balanced_accuracy": sum(recalls) / len(LABELS),
        "macro_f1": sum(f1_scores) / len(LABELS),
        "class_counts": dict(Counter(y_true)),
        "invalid_predictions": sum(pred == "Invalid" for pred in y_pred),
        "confusion_matrix": confusion,
    }


def _valid_vote_set(df: pd.DataFrame) -> set[str]:
    return set(df["target_vote"].dropna().astype(str))


def _raw_exact_match(df: pd.DataFrame) -> dict:
    valid_votes = _valid_vote_set(df)
    invalid_mask = df["generated_vote"].isna() | (~df["generated_vote"].astype(str).isin(valid_votes))
    return {
        "n_rows": len(df),
        "unique_meetings": int(df["meeting_date"].nunique()),
        "exact_match": float(df["match_result"].mean()),
        "invalid_predictions": int(invalid_mask.sum()),
        "target_vote_counts": {str(k): int(v) for k, v in df["target_vote"].value_counts(dropna=False).items()},
        "generated_vote_counts": {str(k): int(v) for k, v in df["generated_vote"].value_counts(dropna=False).items()},
    }


def _prepare_sheet(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    if "index" not in prepared.columns:
        prepared["index"] = range(len(prepared))
    prepared["meeting_date"] = prepared.apply(_extract_meeting_date, axis=1)
    prepared = prepared[prepared["meeting_date"].notna()].copy()
    prepared["coarse_target"] = prepared["target_vote"].map(_coarse_vote)
    prepared["coarse_pred"] = prepared["generated_vote"].map(_coarse_vote)
    return prepared


def _evaluate_sheet(label: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    prepared = _prepare_sheet(df)
    raw_metrics = _raw_exact_match(prepared)
    deduplicated = _deduplicate(prepared)
    coarse_metrics = _classification_metrics(
        deduplicated["coarse_target"].tolist(),
        deduplicated["coarse_pred"].tolist(),
    )
    coarse_metrics["label"] = label
    return deduplicated, raw_metrics, coarse_metrics


def _build_baselines(reference_df: pd.DataFrame) -> dict[str, dict]:
    ordered = reference_df.sort_values("meeting_date").reset_index(drop=True)
    majority_vote = ordered["coarse_target"].mode().iloc[0]

    ordered["majority_pred"] = majority_vote
    ordered["lag1_pred"] = ordered["coarse_target"].shift(1).fillna(majority_vote)

    return {
        "Majority baseline": _classification_metrics(
            ordered["coarse_target"].tolist(),
            ordered["majority_pred"].tolist(),
        ),
        "Lag-1 action baseline": _classification_metrics(
            ordered["coarse_target"].tolist(),
            ordered["lag1_pred"].tolist(),
        ),
    }


def _print_report(
    raw_metrics_by_label: dict[str, dict],
    coarse_metrics_by_label: dict[str, dict],
    meeting_dates: list[str],
) -> None:
    print("\n## Decision Exact-Match Summary")
    for label, metrics in raw_metrics_by_label.items():
        print(
            f"- {label}: rows={metrics['n_rows']}, "
            f"unique_meetings={metrics['unique_meetings']}, "
            f"exact_match={metrics['exact_match']:.4f}, "
            f"invalid_predictions={metrics['invalid_predictions']}"
        )

    print("\n## Decision Coarse-Class Evaluation")
    print(f"- deduplicated_meetings={len(meeting_dates)} ({meeting_dates[0]} to {meeting_dates[-1]})")
    for label, metrics in coarse_metrics_by_label.items():
        print(
            f"- {label}: n={metrics['n_meetings']}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"invalid_predictions={metrics['invalid_predictions']}"
        )
        print(f"  class_counts={metrics['class_counts']}")
        print(f"  confusion_matrix={metrics['confusion_matrix']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate archived Chapter 2 decision results.")
    parser.add_argument(
        "--summary-xlsx",
        type=Path,
        default=Path("process_decsion_output/summary.xlsx"),
        help="Workbook containing the base and ft decision outputs.",
    )
    parser.add_argument(
        "--base-sheet",
        default="base",
        help="Sheet name for Backbone-0 results.",
    )
    parser.add_argument(
        "--ft-sheet",
        default="ft",
        help="Sheet name for Checkpoint-4 results.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    args = parser.parse_args()

    base_df = pd.read_excel(args.summary_xlsx, sheet_name=args.base_sheet)
    ft_df = pd.read_excel(args.summary_xlsx, sheet_name=args.ft_sheet)

    base_dedup, base_raw, base_coarse = _evaluate_sheet("Backbone-0", base_df)
    ft_dedup, ft_raw, ft_coarse = _evaluate_sheet("Checkpoint-4", ft_df)

    if base_dedup["meeting_date"].tolist() != ft_dedup["meeting_date"].tolist():
        raise ValueError("The base and fine-tuned sheets do not cover the same deduplicated meeting set.")

    baseline_metrics = _build_baselines(base_dedup[["meeting_date", "coarse_target"]].copy())

    raw_metrics_by_label = {
        "Backbone-0": base_raw,
        "Checkpoint-4": ft_raw,
    }
    coarse_metrics_by_label = {
        "Backbone-0": base_coarse,
        "Checkpoint-4": ft_coarse,
        **baseline_metrics,
    }
    meeting_dates = base_dedup["meeting_date"].tolist()

    _print_report(raw_metrics_by_label, coarse_metrics_by_label, meeting_dates)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_workbook": str(args.summary_xlsx),
            "meeting_dates": meeting_dates,
            "raw_exact_match": raw_metrics_by_label,
            "coarse_metrics": coarse_metrics_by_label,
        }
        args.output_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
