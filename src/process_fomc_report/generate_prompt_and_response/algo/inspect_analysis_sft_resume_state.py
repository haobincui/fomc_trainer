from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from process_fomc_report.build_qa_master import (
    EXPECTED_AFTER_2009_MASTER_ROWS,
    EXPECTED_AFTER_2009_SPLIT_ROWS,
    inspect_qa_master_export,
)
from process_fomc_report.generate_prompt_and_response.algo.assemble_analysis_training_data import (
    build_analysis_dataset_expectations,
)
from process_fomc_report.generate_prompt_and_response.algo.common.config import load_pipeline_config, resolve_path
from process_fomc_report.generate_prompt_and_response.algo.common.io_utils import load_jsonl
from process_fomc_report.generate_prompt_and_response.algo.normalize_input_sources import (
    build_normalize_input_sources_marker,
    get_normalize_input_sources_audit_path,
)


STAGE_ORDER = (
    "build_qa_master",
    "normalize_input_sources",
    "merge_labels",
    "analysis_sft_teacher_prompts",
    "analysis_sft_teacher_responses",
    "analysis_sft_prompts",
    "analysis_sft_dataset",
)


def _jsonl_count(path: Path) -> int | None:
    return len(load_jsonl(path)) if path.exists() else None


def _failed_jsonl_count(path: Path) -> int:
    return len(load_jsonl(path)) if path.exists() else 0


def _xlsx_count(path: Path) -> int | None:
    return len(pd.read_excel(path)) if path.exists() else None


def _prompt_output_state(prompt_root: Path, expected_split_counts: dict[str, int], *, label: str) -> dict:
    actual_counts = {
        split: _jsonl_count(prompt_root / f"{split}.jsonl")
        for split in ("train", "eval", "test")
    }
    mismatches = [
        f"{split}_count_mismatch"
        for split, expected in expected_split_counts.items()
        if actual_counts[split] != expected
    ]
    return {
        "complete": not mismatches,
        "expected_counts": {
            "rows": expected_split_counts,
        },
        "actual_counts": {
            "rows": actual_counts,
            "output_root": str(prompt_root),
        },
        "reason": "ok" if not mismatches else f"{label}_" + ",".join(mismatches),
    }


def _normalize_input_sources_state(config_path: str | None, config: dict) -> dict:
    audit_path = get_normalize_input_sources_audit_path(config)
    current_marker = build_normalize_input_sources_marker(config_path, config=config)
    recorded_marker = None
    if audit_path.exists():
        recorded_marker = json.loads(audit_path.read_text(encoding="utf-8"))

    reason_parts: list[str] = []
    if not audit_path.exists():
        reason_parts.append("marker_missing")
    if recorded_marker != current_marker:
        reason_parts.append("marker_mismatch")

    return {
        "complete": not reason_parts,
        "expected_counts": current_marker,
        "actual_counts": {
            "audit_path": str(audit_path),
            "audit_present": audit_path.exists(),
            "recorded_marker": recorded_marker,
        },
        "reason": "ok" if not reason_parts else ",".join(reason_parts),
    }


def _merge_labels_state(config: dict, *, scope: str, expected_labeled_rows: int) -> dict:
    labeled_paths = config["analysis"].get("labeled_paths", {})
    labeled_path = resolve_path(labeled_paths[scope])
    actual_rows = _xlsx_count(labeled_path)
    complete = actual_rows == expected_labeled_rows
    return {
        "complete": complete,
        "expected_counts": {
            "merged_labeled_rows": expected_labeled_rows,
        },
        "actual_counts": {
            "merged_labeled_rows": actual_rows,
            "path": str(labeled_path),
        },
        "reason": "ok" if complete else "merged_labeled_count_mismatch",
    }


def _teacher_responses_state(
    config: dict,
    *,
    scope: str,
    expected_split_counts: dict[str, int],
    teacher_prompt_complete: bool,
) -> dict:
    analysis_cfg = config["analysis"]
    prompt_root = resolve_path(analysis_cfg["teacher_prompt_root"]) / scope
    response_root = resolve_path(analysis_cfg["teacher_response_root"]) / scope
    prompt_counts = {
        split: _jsonl_count(prompt_root / f"{split}.jsonl")
        for split in ("train", "eval", "test")
    }
    response_counts = {
        split: _jsonl_count(response_root / f"{split}.jsonl")
        for split in ("train", "eval", "test")
    }
    failed_counts = {
        split: _failed_jsonl_count(response_root / f"{split}_failed.jsonl")
        for split in ("train", "eval", "test")
    }

    reason_parts: list[str] = []
    if not teacher_prompt_complete:
        reason_parts.append("teacher_prompts_incomplete")
    for split, expected in expected_split_counts.items():
        if prompt_counts[split] != expected:
            reason_parts.append(f"teacher_prompt_{split}_count_mismatch")
        if response_counts[split] != expected:
            reason_parts.append(f"teacher_response_{split}_count_mismatch")
        if failed_counts[split] != 0:
            reason_parts.append(f"teacher_response_{split}_failed_rows_present")

    return {
        "complete": not reason_parts,
        "expected_counts": {
            "teacher_prompt_rows": expected_split_counts,
            "teacher_response_rows": expected_split_counts,
            "failed_rows": {split: 0 for split in ("train", "eval", "test")},
        },
        "actual_counts": {
            "teacher_prompt_rows": prompt_counts,
            "teacher_response_rows": response_counts,
            "failed_rows": failed_counts,
            "output_root": str(response_root),
        },
        "reason": "ok" if not reason_parts else ",".join(reason_parts),
    }


def _dataset_output_counts(output_root: Path) -> dict[str, dict[str, int | None]]:
    return {
        split: {
            "rows": _jsonl_count(output_root / f"{split}.jsonl"),
            "manifest_rows": _jsonl_count(output_root / f"{split}_manifest.jsonl"),
        }
        for split in ("train", "eval", "test")
    }


def _analysis_sft_dataset_state(
    config_path: str | None,
    *,
    profile: str,
    student_prompt_complete: bool,
    teacher_response_complete: bool,
) -> dict:
    reason_parts: list[str] = []
    if not student_prompt_complete:
        reason_parts.append("analysis_sft_prompts_incomplete")
    if not teacher_response_complete:
        reason_parts.append("analysis_sft_teacher_responses_incomplete")

    expected_counts: dict[str, dict] = {}
    actual_counts: dict[str, dict] = {}
    try:
        expectations = build_analysis_dataset_expectations(
            config_path,
            profile=profile,
            dataset_kind="analysis_sft",
        )
    except FileNotFoundError:
        expectations = {}
        reason_parts.append("dataset_expectations_unavailable")

    for profile_name, datasets in expectations.items():
        payload = datasets.get("analysis_sft")
        if not payload:
            continue
        output_root = Path(payload["output_root"])
        expected_counts[profile_name] = payload["splits"]
        actual_counts[profile_name] = {
            "output_root": str(output_root),
            "splits": _dataset_output_counts(output_root),
        }
        for split, expected in payload["splits"].items():
            split_actual = actual_counts[profile_name]["splits"][split]
            if split_actual["rows"] != expected["rows"]:
                reason_parts.append(f"{profile_name}_{split}_dataset_count_mismatch")
            if split_actual["manifest_rows"] != expected["manifest_rows"]:
                reason_parts.append(f"{profile_name}_{split}_manifest_count_mismatch")

    return {
        "complete": not reason_parts,
        "expected_counts": expected_counts,
        "actual_counts": actual_counts,
        "reason": "ok" if not reason_parts else ",".join(dict.fromkeys(reason_parts)),
    }


def inspect_analysis_sft_stage_status(
    config_path: str | None = None,
    *,
    profile: str = "compat",
    scope: str = "after_2009",
    expected_split_counts: dict[str, int] | None = None,
    expected_qa_master_rows: int = EXPECTED_AFTER_2009_MASTER_ROWS,
    expected_labeled_rows: int = 10666,
) -> dict:
    if scope != "after_2009":
        raise RuntimeError("analysis_sft resume inspection only supports scope=after_2009.")

    config = load_pipeline_config(config_path)
    expected_split_counts = dict(expected_split_counts or EXPECTED_AFTER_2009_SPLIT_ROWS)

    split_manifest_cfg = config["analysis"]["split_manifests"]
    if scope in split_manifest_cfg:
        split_manifest_cfg = split_manifest_cfg[scope]

    build_qa_master_state = inspect_qa_master_export(
        resolve_path(config["analysis"]["pipeline_root"]),
        input_root=resolve_path(config.get("pipeline", {}).get("input_root", "dataset/processed/input_sources")),
        split_manifest_paths={
            split: resolve_path(path)
            for split, path in split_manifest_cfg.items()
        },
        expected_master_rows=expected_qa_master_rows,
        expected_split_counts=expected_split_counts,
    )
    normalize_state = _normalize_input_sources_state(config_path, config)
    merge_labels_state = _merge_labels_state(config, scope=scope, expected_labeled_rows=expected_labeled_rows)
    teacher_prompts_state = _prompt_output_state(
        resolve_path(config["analysis"]["teacher_prompt_root"]) / scope,
        expected_split_counts,
        label="teacher_prompts",
    )
    teacher_responses_state = _teacher_responses_state(
        config,
        scope=scope,
        expected_split_counts=expected_split_counts,
        teacher_prompt_complete=teacher_prompts_state["complete"],
    )
    analysis_sft_prompts_state = _prompt_output_state(
        resolve_path(config["analysis"]["analysis_sft_prompt_root"]) / scope,
        expected_split_counts,
        label="analysis_sft_prompts",
    )
    dataset_state = _analysis_sft_dataset_state(
        config_path,
        profile=profile,
        student_prompt_complete=analysis_sft_prompts_state["complete"],
        teacher_response_complete=teacher_responses_state["complete"],
    )

    return {
        "scope": scope,
        "profile": profile,
        "stages": {
            "build_qa_master": build_qa_master_state,
            "normalize_input_sources": normalize_state,
            "merge_labels": merge_labels_state,
            "analysis_sft_teacher_prompts": teacher_prompts_state,
            "analysis_sft_teacher_responses": teacher_responses_state,
            "analysis_sft_prompts": analysis_sft_prompts_state,
            "analysis_sft_dataset": dataset_state,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect analysis_sft resume state for chk1.sh.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scope", choices=["after_2009"], default="after_2009")
    parser.add_argument("--profile", choices=["strict", "compat", "both"], default="compat")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = inspect_analysis_sft_stage_status(
        args.config,
        profile=args.profile,
        scope=args.scope,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
