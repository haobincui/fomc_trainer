from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_ROOT = REPO_ROOT / "dataset" / "raw_data" / "generate_prompt" / "chapter2_qa"

MASTER_PATH = "master/chapter2_qa_master.jsonl"
MEETING_LEVEL_ROOT = "meeting_level"
ANALYSIS_SFT_ROOT = "analysis_sft"
ANALYSIS_GRPO_ROOT = "analysis_grpo"
AUDIT_PATH = "audit/analysis_stage1_datasets.json"

TRAINING_FIELDS = ("prompt", "response", "provided_data")

EXPECTED_MEETING_LEVEL_COUNTS = {
    "train": 3890,
    "eval": 535,
    "test": 462,
}

SFT_TRAIN_SIZE = 3112
GRPO_CORE_SIZE = 778
SFT_MIX_STEP = 10
SFT_MIX_SIZE = 311


def log(message: str) -> None:
    print(f"[build_stage1_analysis_datasets] {message}", flush=True)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def minimal_training_view(row: dict) -> dict:
    return {field: row[field] for field in TRAINING_FIELDS}


def verify_manifest_rows(rows: list[dict], master_ids: set[str], split: str) -> None:
    if len(rows) != EXPECTED_MEETING_LEVEL_COUNTS[split]:
        raise RuntimeError(
            f"Unexpected meeting-level {split} count: expected "
            f"{EXPECTED_MEETING_LEVEL_COUNTS[split]}, got {len(rows)}"
        )

    missing_fields = [field for field in ("sample_id", "meeting_date", "prompt_hash", "prompt", "response", "provided_data") if field not in rows[0]]
    if missing_fields:
        raise RuntimeError(f"Missing required manifest fields in {split}: {missing_fields}")

    unknown_sample_ids = [row["sample_id"] for row in rows if row["sample_id"] not in master_ids]
    if unknown_sample_ids:
        raise RuntimeError(f"{split} contains sample_ids not found in master: {unknown_sample_ids[:5]}")

    empty_responses = [row["sample_id"] for row in rows if not str(row["response"]).strip()]
    if empty_responses:
        raise RuntimeError(f"{split} contains empty responses: {empty_responses[:5]}")


def sorted_train_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: (row["prompt_hash"], row["sample_id"]))


def assert_meeting_disjoint(train_rows: list[dict], eval_rows: list[dict], test_rows: list[dict]) -> dict[str, int]:
    train_meetings = {row["meeting_date"] for row in train_rows}
    eval_meetings = {row["meeting_date"] for row in eval_rows}
    test_meetings = {row["meeting_date"] for row in test_rows}

    if train_meetings & eval_meetings:
        raise RuntimeError("Train/eval meeting overlap detected")
    if train_meetings & test_meetings:
        raise RuntimeError("Train/test meeting overlap detected")
    if eval_meetings & test_meetings:
        raise RuntimeError("Eval/test meeting overlap detected")

    return {
        "train": len(train_meetings),
        "eval": len(eval_meetings),
        "test": len(test_meetings),
    }


def export_dataset(root: Path, train_rows: list[dict], eval_rows: list[dict], test_rows: list[dict]) -> None:
    write_jsonl(root / "train.jsonl", [minimal_training_view(row) for row in train_rows])
    write_jsonl(root / "eval.jsonl", [minimal_training_view(row) for row in eval_rows])
    write_jsonl(root / "test.jsonl", [minimal_training_view(row) for row in test_rows])

    write_jsonl(root / "train_manifest.jsonl", train_rows)
    write_jsonl(root / "eval_manifest.jsonl", eval_rows)
    write_jsonl(root / "test_manifest.jsonl", test_rows)


def build_analysis_datasets(export_root: Path) -> dict:
    log(f"Starting Stage-1 dataset build under {export_root}")
    master_rows = load_jsonl(export_root / MASTER_PATH)
    meeting_level_root = export_root / MEETING_LEVEL_ROOT
    log(f"Loaded canonical master rows={len(master_rows)}")

    master_ids = {row["sample_id"] for row in master_rows}
    if len(master_ids) != len(master_rows):
        raise RuntimeError("Duplicate sample_id detected in canonical master")

    meeting_level = {
        split: load_jsonl(meeting_level_root / f"{split}_manifest.jsonl")
        for split in ("train", "eval", "test")
    }
    log(
        "Loaded meeting-level manifests: "
        + ", ".join(f"{split}={len(rows)}" for split, rows in meeting_level.items())
    )
    for split, rows in meeting_level.items():
        verify_manifest_rows(rows, master_ids, split)
    log("Verified meeting-level manifests and response integrity.")

    sorted_train = sorted_train_rows(meeting_level["train"])
    sft_train_rows = sorted_train[:SFT_TRAIN_SIZE]
    grpo_core_rows = sorted_train[SFT_TRAIN_SIZE:]

    if len(sft_train_rows) != SFT_TRAIN_SIZE:
        raise RuntimeError(f"Unexpected SFT train size: expected {SFT_TRAIN_SIZE}, got {len(sft_train_rows)}")
    if len(grpo_core_rows) != GRPO_CORE_SIZE:
        raise RuntimeError(f"Unexpected GRPO core size: expected {GRPO_CORE_SIZE}, got {len(grpo_core_rows)}")

    sft_mix_indices = list(range(0, len(sft_train_rows), SFT_MIX_STEP))[:SFT_MIX_SIZE]
    if len(sft_mix_indices) != SFT_MIX_SIZE:
        raise RuntimeError(f"Unexpected SFT mix size: expected {SFT_MIX_SIZE}, got {len(sft_mix_indices)}")

    sft_mix_rows = [dict(sft_train_rows[index]) for index in sft_mix_indices]
    grpo_train_rows = [dict(row, train_source="grpo_core") for row in grpo_core_rows]
    grpo_train_rows.extend(dict(row, train_source="sft_mix") for row in sft_mix_rows)
    log(
        "Derived Stage-1 splits: "
        f"analysis_sft_train={len(sft_train_rows)}, "
        f"grpo_core={len(grpo_core_rows)}, "
        f"sft_mix={len(sft_mix_rows)}, "
        f"analysis_grpo_train={len(grpo_train_rows)}"
    )

    analysis_sft_root = export_root / ANALYSIS_SFT_ROOT
    analysis_grpo_root = export_root / ANALYSIS_GRPO_ROOT

    log("Exporting analysis_sft and analysis_grpo datasets.")
    export_dataset(
        analysis_sft_root,
        sft_train_rows,
        meeting_level["eval"],
        meeting_level["test"],
    )
    export_dataset(
        analysis_grpo_root,
        grpo_train_rows,
        meeting_level["eval"],
        meeting_level["test"],
    )

    sft_meeting_counts = assert_meeting_disjoint(
        sft_train_rows,
        meeting_level["eval"],
        meeting_level["test"],
    )
    grpo_meeting_counts = assert_meeting_disjoint(
        grpo_train_rows,
        meeting_level["eval"],
        meeting_level["test"],
    )

    sft_train_ids = {row["sample_id"] for row in sft_train_rows}
    grpo_train_ids = {row["sample_id"] for row in grpo_train_rows}
    overlap_ids = sft_train_ids & grpo_train_ids

    if len(overlap_ids) != SFT_MIX_SIZE:
        raise RuntimeError(
            f"Unexpected SFT/GRPO overlap: expected {SFT_MIX_SIZE}, got {len(overlap_ids)}"
        )

    eval_prompt_match = (
        load_jsonl(analysis_sft_root / "eval.jsonl") == load_jsonl(meeting_level_root / "eval.jsonl")
        and load_jsonl(analysis_grpo_root / "eval.jsonl") == load_jsonl(meeting_level_root / "eval.jsonl")
    )
    test_prompt_match = (
        load_jsonl(analysis_sft_root / "test.jsonl") == load_jsonl(meeting_level_root / "test.jsonl")
        and load_jsonl(analysis_grpo_root / "test.jsonl") == load_jsonl(meeting_level_root / "test.jsonl")
    )
    if not eval_prompt_match or not test_prompt_match:
        raise RuntimeError("analysis_sft/analysis_grpo eval/test exports do not match meeting_level")
    log("Verified eval/test exports match the meeting-level source splits.")

    return {
        "source_root": str(meeting_level_root),
        "analysis_sft": {
            "train_rows": len(sft_train_rows),
            "eval_rows": len(meeting_level["eval"]),
            "test_rows": len(meeting_level["test"]),
            "meeting_counts": sft_meeting_counts,
        },
        "analysis_grpo": {
            "train_rows": len(grpo_train_rows),
            "eval_rows": len(meeting_level["eval"]),
            "test_rows": len(meeting_level["test"]),
            "meeting_counts": grpo_meeting_counts,
            "grpo_core_rows": len(grpo_core_rows),
            "sft_mix_rows": len(sft_mix_rows),
            "train_source_counts": {
                "grpo_core": sum(1 for row in grpo_train_rows if row["train_source"] == "grpo_core"),
                "sft_mix": sum(1 for row in grpo_train_rows if row["train_source"] == "sft_mix"),
            },
        },
        "overlap_checks": {
            "sft_grpo_train_overlap_rows": len(overlap_ids),
            "eval_matches_meeting_level": eval_prompt_match,
            "test_matches_meeting_level": test_prompt_match,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive Chapter 2 Stage-1 analysis_sft and analysis_grpo datasets from the meeting-level QA export."
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=DEFAULT_EXPORT_ROOT,
        help="Root directory containing the chapter2_qa export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_analysis_datasets(args.export_root)
    write_json(args.export_root / AUDIT_PATH, audit)
    log(f"Wrote audit summary to {args.export_root / AUDIT_PATH}")

    print("✅ Built Stage-1 analysis datasets")
    print(f"  analysis_sft:  {args.export_root / ANALYSIS_SFT_ROOT}")
    print(f"  analysis_grpo: {args.export_root / ANALYSIS_GRPO_ROOT}")
    print(
        "  counts:",
        audit["analysis_sft"]["train_rows"],
        audit["analysis_grpo"]["train_rows"],
        audit["analysis_sft"]["eval_rows"],
        audit["analysis_sft"]["test_rows"],
    )


if __name__ == "__main__":
    main()
