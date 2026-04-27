from __future__ import annotations

import argparse
import json
from pathlib import Path

from process_fomc_report.generate_prompt_and_response.algo.build_decision_datasets import (
    _normalize_decision_grpo_rows,
    _normalize_decision_sft_rows,
)
from process_fomc_report.generate_prompt_and_response.algo.common.config import (
    load_pipeline_config,
    resolve_path,
)
from process_fomc_report.generate_prompt_and_response.algo.common.io_utils import load_jsonl


STAGE_ORDER = (
    "decision_prompts",
    "decision_dataset",
)
SPLITS = ("train", "eval", "test")
DATASETS = ("decision_sft", "decision_grpo")


def _jsonl_count(path: Path) -> int | None:
    return len(load_jsonl(path)) if path.exists() else None


def _expected_decision_split_counts(config: dict, *, scope: str) -> dict[str, dict[str, int]]:
    normalized_rows = {
        "decision_sft": _normalize_decision_sft_rows(config, scope=scope),
        "decision_grpo": _normalize_decision_grpo_rows(config, scope=scope),
    }
    return {
        dataset_name: {
            split: len(rows_by_split.get(split, []))
            for split in SPLITS
        }
        for dataset_name, rows_by_split in normalized_rows.items()
    }


def _expected_decision_split_counts_from_audit(config: dict, *, scope: str) -> dict[str, dict[str, int]] | None:
    audit_root = resolve_path(config["pipeline"]["audit_root"]) / "decision"
    prompts_audit_path = audit_root / f"prompts_{scope}.json"
    if prompts_audit_path.exists():
        payload = json.loads(prompts_audit_path.read_text(encoding="utf-8"))
        return {
            dataset_name: {
                split: int(payload.get(dataset_name, {}).get(split, {}).get("rows", 0))
                for split in SPLITS
            }
            for dataset_name in DATASETS
        }

    dataset_audit_path = audit_root / f"dataset_{scope}.json"
    if dataset_audit_path.exists():
        payload = json.loads(dataset_audit_path.read_text(encoding="utf-8"))
        return {
            dataset_name: {
                split: int(payload.get(f"{dataset_name}_{split}", {}).get("rows", 0))
                for split in SPLITS
            }
            for dataset_name in DATASETS
        }

    return None


def _prompt_actual_counts(config: dict, *, scope: str) -> dict:
    decision_cfg = config["decision"]
    return {
        dataset_name: {
            "output_root": str(
                resolve_path(decision_cfg["prompt_roots"][dataset_name]) / scope
            ),
            "rows": {
                split: _jsonl_count(
                    resolve_path(decision_cfg["prompt_roots"][dataset_name])
                    / scope
                    / f"{split}.jsonl"
                )
                for split in SPLITS
            },
        }
        for dataset_name in DATASETS
    }


def _dataset_actual_counts(config: dict) -> dict:
    decision_cfg = config["decision"]
    return {
        dataset_name: {
            "output_root": str(resolve_path(decision_cfg["train_roots"][dataset_name])),
            "splits": _dataset_output_counts(resolve_path(decision_cfg["train_roots"][dataset_name])),
        }
        for dataset_name in DATASETS
    }


def _unavailable_expectation_state(config: dict, *, scope: str, source_error: str) -> dict:
    return {
        "decision_prompts": {
            "complete": False,
            "expected_counts": {
                "source_error": source_error,
                "audit_fallback": "missing",
            },
            "actual_counts": _prompt_actual_counts(config, scope=scope),
            "reason": "decision_source_counts_unavailable",
        },
        "decision_dataset": {
            "complete": False,
            "expected_counts": {
                "source_error": source_error,
                "audit_fallback": "missing",
            },
            "actual_counts": _dataset_actual_counts(config),
            "reason": "decision_source_counts_unavailable",
        },
    }


def _prompt_rows_state(config: dict, *, scope: str, expected_counts: dict[str, dict[str, int]]) -> dict:
    decision_cfg = config["decision"]
    actual_counts = {}
    reason_parts: list[str] = []

    for dataset_name in DATASETS:
        prompt_root = resolve_path(decision_cfg["prompt_roots"][dataset_name]) / scope
        rows_by_split = {
            split: _jsonl_count(prompt_root / f"{split}.jsonl")
            for split in SPLITS
        }
        actual_counts[dataset_name] = {
            "output_root": str(prompt_root),
            "rows": rows_by_split,
        }
        for split, expected in expected_counts[dataset_name].items():
            if rows_by_split[split] != expected:
                reason_parts.append(f"{dataset_name}_{split}_prompt_count_mismatch")

    return {
        "complete": not reason_parts,
        "expected_counts": {
            dataset_name: {"rows": expected_counts[dataset_name]}
            for dataset_name in DATASETS
        },
        "actual_counts": actual_counts,
        "reason": "ok" if not reason_parts else ",".join(reason_parts),
    }


def _dataset_output_counts(output_root: Path) -> dict[str, dict[str, int | None]]:
    return {
        split: {
            "rows": _jsonl_count(output_root / f"{split}.jsonl"),
            "manifest_rows": _jsonl_count(output_root / f"{split}_manifest.jsonl"),
        }
        for split in SPLITS
    }


def _dataset_rows_state(
    config: dict,
    *,
    expected_counts: dict[str, dict[str, int]],
    prompts_complete: bool,
) -> dict:
    decision_cfg = config["decision"]
    actual_counts = {}
    reason_parts: list[str] = []
    if not prompts_complete:
        reason_parts.append("decision_prompts_incomplete")

    for dataset_name in DATASETS:
        output_root = resolve_path(decision_cfg["train_roots"][dataset_name])
        split_counts = _dataset_output_counts(output_root)
        actual_counts[dataset_name] = {
            "output_root": str(output_root),
            "splits": split_counts,
        }
        for split, expected in expected_counts[dataset_name].items():
            if split_counts[split]["rows"] != expected:
                reason_parts.append(f"{dataset_name}_{split}_dataset_count_mismatch")
            if split_counts[split]["manifest_rows"] != expected:
                reason_parts.append(f"{dataset_name}_{split}_manifest_count_mismatch")

    return {
        "complete": not reason_parts,
        "expected_counts": {
            dataset_name: {
                split: {"rows": expected, "manifest_rows": expected}
                for split, expected in expected_counts[dataset_name].items()
            }
            for dataset_name in DATASETS
        },
        "actual_counts": actual_counts,
        "reason": "ok" if not reason_parts else ",".join(reason_parts),
    }


def inspect_decision_stage_status(
    config_path: str | None = None,
    *,
    scope: str = "after_2009",
) -> dict:
    if scope != "after_2009":
        raise RuntimeError("decision resume inspection only supports scope=after_2009.")

    config = load_pipeline_config(config_path)
    expected_source = "source"
    try:
        expected_counts = _expected_decision_split_counts(config, scope=scope)
    except FileNotFoundError as exc:
        expected_counts = _expected_decision_split_counts_from_audit(config, scope=scope)
        expected_source = "audit"
        if expected_counts is None:
            return {
                "scope": scope,
                "expected_source": "unavailable",
                "stages": _unavailable_expectation_state(
                    config,
                    scope=scope,
                    source_error=str(exc),
                ),
            }

    prompt_state = _prompt_rows_state(config, scope=scope, expected_counts=expected_counts)
    dataset_state = _dataset_rows_state(
        config,
        expected_counts=expected_counts,
        prompts_complete=prompt_state["complete"],
    )

    return {
        "scope": scope,
        "expected_source": expected_source,
        "stages": {
            "decision_prompts": prompt_state,
            "decision_dataset": dataset_state,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect decision resume state for chk4.sh.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scope", choices=["after_2009"], default="after_2009")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = inspect_decision_stage_status(
        args.config,
        scope=args.scope,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
