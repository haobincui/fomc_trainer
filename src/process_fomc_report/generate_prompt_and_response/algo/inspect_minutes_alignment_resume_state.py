from __future__ import annotations

import argparse
import json
from pathlib import Path

from process_fomc_report.generate_prompt_and_response.algo.common.config import (
    get_response_template,
    load_pipeline_config,
    resolve_path,
)
from process_fomc_report.generate_prompt_and_response.algo.common.io_utils import load_jsonl
from process_fomc_report.generate_prompt_and_response.algo.common.response_templates import parse_response_text


STAGE_ORDER = (
    "minutes_alignment_prompts",
    "minutes_alignment_teacher_responses",
    "minutes_alignment_dataset",
)
SPLITS = ("train", "eval", "test")


def _jsonl_count(path: Path) -> int | None:
    return len(load_jsonl(path)) if path.exists() else None


def _failed_jsonl_count(path: Path) -> int:
    return len(load_jsonl(path)) if path.exists() else 0


def _load_rows_if_present(path: Path) -> list[dict] | None:
    return load_jsonl(path) if path.exists() else None


def _prompt_output_state(config: dict, *, scope: str) -> dict:
    rewrite_cfg = config["rewrite"]
    source_prompt_root = resolve_path(rewrite_cfg["source_prompt_root"]) / scope
    source_teacher_response_root = resolve_path(rewrite_cfg["source_teacher_response_root"]) / scope
    prompt_root = resolve_path(rewrite_cfg["prompt_root"]) / scope
    expected_counts: dict[str, int | None] = {}
    actual_counts = {
        "output_root": str(prompt_root),
        "source_prompt_rows": {},
        "source_teacher_response_rows": {},
        "source_teacher_failed_rows": {},
        "prompt_rows": {},
    }
    reason_parts: list[str] = []

    for split in SPLITS:
        source_prompt_path = source_prompt_root / f"{split}.jsonl"
        source_teacher_response_path = source_teacher_response_root / f"{split}.jsonl"
        source_prompt_rows = _load_rows_if_present(source_prompt_path)
        expected_counts[split] = len(source_prompt_rows) if source_prompt_rows is not None else None
        actual_counts["source_prompt_rows"][split] = expected_counts[split]
        actual_counts["source_teacher_response_rows"][split] = _jsonl_count(source_teacher_response_path)
        actual_counts["source_teacher_failed_rows"][split] = _failed_jsonl_count(
            source_teacher_response_root / f"{split}_failed.jsonl"
        )
        actual_counts["prompt_rows"][split] = _jsonl_count(prompt_root / f"{split}.jsonl")

        if source_prompt_rows is None:
            reason_parts.append(f"source_prompt_{split}_missing")
            continue
        if not source_teacher_response_path.exists():
            reason_parts.append(f"source_teacher_response_{split}_missing")
        if actual_counts["prompt_rows"][split] != expected_counts[split]:
            reason_parts.append(f"{split}_prompt_count_mismatch")

    return {
        "complete": not reason_parts,
        "expected_counts": {
            "prompt_rows": expected_counts,
        },
        "actual_counts": actual_counts,
        "reason": "ok" if not reason_parts else ",".join(reason_parts),
    }


def _teacher_response_state(
    config: dict,
    *,
    scope: str,
    prompts_complete: bool,
) -> dict:
    rewrite_cfg = config["rewrite"]
    response_template = get_response_template(config)
    prompt_root = resolve_path(rewrite_cfg["prompt_root"]) / scope
    response_root = resolve_path(rewrite_cfg["teacher_response_root"]) / scope
    prompt_counts = {
        split: _jsonl_count(prompt_root / f"{split}.jsonl")
        for split in SPLITS
    }
    response_counts = {
        split: _jsonl_count(response_root / f"{split}.jsonl")
        for split in SPLITS
    }
    failed_counts = {
        split: _failed_jsonl_count(response_root / f"{split}_failed.jsonl")
        for split in SPLITS
    }
    invalid_counts: dict[str, int | None] = {}
    reason_parts: list[str] = []
    if not prompts_complete:
        reason_parts.append("minutes_alignment_prompts_incomplete")

    for split in SPLITS:
        prompt_rows = _load_rows_if_present(prompt_root / f"{split}.jsonl")
        response_rows = _load_rows_if_present(response_root / f"{split}.jsonl")
        if prompt_rows is None:
            invalid_counts[split] = None
            reason_parts.append(f"prompt_{split}_missing")
            continue

        if response_counts[split] != prompt_counts[split]:
            reason_parts.append(f"teacher_response_{split}_count_mismatch")
        if failed_counts[split] != 0:
            reason_parts.append(f"teacher_response_{split}_failed_rows_present")

        response_by_sample_id = {
            row.get("sample_id"): row
            for row in (response_rows or [])
        }
        invalid_count = 0
        for prompt_row in prompt_rows:
            response_row = response_by_sample_id.get(prompt_row.get("sample_id"))
            if response_row is None:
                invalid_count += 1
                continue
            if response_row.get("prompt_hash") != prompt_row.get("prompt_hash"):
                invalid_count += 1
                continue
            if response_row.get("response_template") != response_template:
                invalid_count += 1
                continue
            if response_row.get("status") not in {"success", "archived_master"}:
                invalid_count += 1
        invalid_counts[split] = invalid_count
        if invalid_count:
            reason_parts.append(f"teacher_response_{split}_invalid_rows_present")

    return {
        "complete": not reason_parts,
        "expected_counts": {
            "teacher_response_rows": prompt_counts,
            "failed_rows": {split: 0 for split in SPLITS},
            "invalid_rows": {split: 0 for split in SPLITS},
        },
        "actual_counts": {
            "output_root": str(response_root),
            "prompt_rows": prompt_counts,
            "teacher_response_rows": response_counts,
            "failed_rows": failed_counts,
            "invalid_rows": invalid_counts,
        },
        "reason": "ok" if not reason_parts else ",".join(reason_parts),
    }


def _expected_dataset_counts(config: dict, *, scope: str) -> tuple[dict[str, int | None], list[str]]:
    rewrite_cfg = config["rewrite"]
    prompt_root = resolve_path(rewrite_cfg["prompt_root"]) / scope
    teacher_root = resolve_path(rewrite_cfg["teacher_response_root"]) / scope
    expected_counts: dict[str, int | None] = {}
    reason_parts: list[str] = []

    for split in SPLITS:
        prompt_rows = _load_rows_if_present(prompt_root / f"{split}.jsonl")
        teacher_rows = _load_rows_if_present(teacher_root / f"{split}.jsonl")
        if prompt_rows is None:
            expected_counts[split] = None
            reason_parts.append(f"prompt_{split}_missing")
            continue
        if teacher_rows is None:
            expected_counts[split] = None
            reason_parts.append(f"teacher_response_{split}_missing")
            continue

        teacher_by_sample_id = {row.get("sample_id"): row for row in teacher_rows}
        expected_rows = 0
        for prompt_row in prompt_rows:
            if not str(prompt_row.get("reference_excerpt", "")).strip():
                continue
            if not str(prompt_row.get("raw_analysis", "")).strip():
                continue
            teacher_row = teacher_by_sample_id.get(prompt_row.get("sample_id"))
            if teacher_row is None:
                continue
            if teacher_row.get("prompt_hash") != prompt_row.get("prompt_hash"):
                continue
            if teacher_row.get("status") not in {"success", "archived_master"}:
                continue
            if not parse_response_text(teacher_row.get("response", "")).answer:
                continue
            expected_rows += 1
        expected_counts[split] = expected_rows

    return expected_counts, reason_parts


def _dataset_output_counts(output_root: Path, target_root: Path) -> dict[str, dict[str, int | None]]:
    return {
        split: {
            "rows": _jsonl_count(output_root / f"{split}.jsonl"),
            "manifest_rows": _jsonl_count(output_root / f"{split}_manifest.jsonl"),
            "target_rows": _jsonl_count(target_root / f"{split}.jsonl"),
        }
        for split in SPLITS
    }


def _dataset_state(
    config: dict,
    *,
    scope: str,
    prompts_complete: bool,
    teacher_responses_complete: bool,
) -> dict:
    rewrite_cfg = config["rewrite"]
    output_root = resolve_path(rewrite_cfg["train_root"])
    target_root = resolve_path(rewrite_cfg["normalized_target_root"])
    expected_counts, expected_reason_parts = _expected_dataset_counts(config, scope=scope)
    actual_split_counts = _dataset_output_counts(output_root, target_root)
    reason_parts = list(expected_reason_parts)
    if not prompts_complete:
        reason_parts.append("minutes_alignment_prompts_incomplete")
    if not teacher_responses_complete:
        reason_parts.append("minutes_alignment_teacher_responses_incomplete")

    for split, expected in expected_counts.items():
        if expected is None:
            continue
        split_actual = actual_split_counts[split]
        if split_actual["rows"] != expected:
            reason_parts.append(f"{split}_dataset_count_mismatch")
        if split_actual["manifest_rows"] != expected:
            reason_parts.append(f"{split}_manifest_count_mismatch")
        if split_actual["target_rows"] != expected:
            reason_parts.append(f"{split}_target_count_mismatch")

    return {
        "complete": not reason_parts,
        "expected_counts": {
            "dataset_rows": {
                split: {
                    "rows": expected,
                    "manifest_rows": expected,
                    "target_rows": expected,
                }
                for split, expected in expected_counts.items()
            },
        },
        "actual_counts": {
            "output_root": str(output_root),
            "target_root": str(target_root),
            "splits": actual_split_counts,
        },
        "reason": "ok" if not reason_parts else ",".join(dict.fromkeys(reason_parts)),
    }


def inspect_minutes_alignment_stage_status(
    config_path: str | None = None,
    *,
    scope: str = "after_2009",
) -> dict:
    if scope != "after_2009":
        raise RuntimeError("minutes_alignment resume inspection only supports scope=after_2009.")

    config = load_pipeline_config(config_path)
    prompts_state = _prompt_output_state(config, scope=scope)
    teacher_responses_state = _teacher_response_state(
        config,
        scope=scope,
        prompts_complete=prompts_state["complete"],
    )
    dataset_state = _dataset_state(
        config,
        scope=scope,
        prompts_complete=prompts_state["complete"],
        teacher_responses_complete=teacher_responses_state["complete"],
    )

    return {
        "scope": scope,
        "stages": {
            "minutes_alignment_prompts": prompts_state,
            "minutes_alignment_teacher_responses": teacher_responses_state,
            "minutes_alignment_dataset": dataset_state,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect minutes_alignment resume state for chk3.sh.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scope", choices=["after_2009"], default="after_2009")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = inspect_minutes_alignment_stage_status(
        args.config,
        scope=args.scope,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
