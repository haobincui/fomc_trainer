"""Apply a tolerant, fully disclosed checkpoint-level probe admission rule.

Per-sample strict results remain unchanged.  This selector relaxes only the
checkpoint-level rule: catastrophic structure/repetition failures remain
zero-tolerance, while a small number of factual or LLM-judge misses is allowed
on the eight-row screening panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.retrain_v2 import probe_paper_chk2_minutes_checkpoints as probe
from open_r1.provenance import sha256_file


SCHEMA = "paper-chk2-minutes-relaxed-checkpoint-selection-v1"
DEFAULT_OUTPUT_ROOT = probe.DEFAULT_OUTPUT_ROOT
DEFAULT_RECEIPT_NAME = "relaxed_checkpoint_selection_v1.json"

ZERO_TOLERANCE_FAILURES = {
    "non_finite_or_empty_output",
    "completion_length_cap",
    "strict_periodic_tail",
    "full_4gram_repetition_ge_0.50",
    "tail_4gram_repetition_ge_0.60",
    "missing_terminal_eos",
    "think_boundary_count_not_one",
    "empty_native_reasoning",
    "empty_final_answer",
    "final_answer_not_single_paragraph",
    "final_answer_word_count_outside_20_400",
    "final_answer_heading_list_json_or_meta",
    "final_answer_contains_citation",
    "final_answer_contains_credential_pattern",
    "final_answer_contains_control_marker",
    "sequence_exceeds_4096_tokens",
}
FACTUAL_FIELD_FAILURES = {
    "numeric_multiset_not_preserved",
    "date_set_not_preserved",
    "attribution_set_not_preserved",
    "final_answer_exact_full_source_copy",
}
THRESHOLDS = {
    "zero_tolerance_format_degeneracy_failures": 0,
    "minimum_deterministic_factual_pass_rate": 0.75,
    "minimum_validator_a_pass_rate_among_deterministic_pass": 0.75,
    "minimum_validator_b_pass_rate_among_validator_a_pass": 0.50,
    "minimum_validator_b_mean_score": 6.0,
    "minimum_full_chain_pass_rate": 0.25,
    "maximum_unresolved_judge_failures": 0,
}


class RelaxedSelectionError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RelaxedSelectionError(message)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing JSONL: {path}")
    rows = [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines()]
    _require(all(isinstance(row, dict) for row in rows), f"invalid JSONL: {path}")
    return rows


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _apply_remediation(
    output_root: Path,
    checkpoint: int,
    judge_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in judge_rows]
    remediation_root = (
        output_root / "judges" / f"checkpoint-{checkpoint}-unresolved-remediation-v1"
    )
    if not remediation_root.is_dir():
        return rows
    summary = _read_json(remediation_root / "summary.json")
    result_path = remediation_root / "results.jsonl"
    _require(
        sha256_file(result_path) == summary.get("results", {}).get("sha256"),
        "remediation result hash drift",
    )
    remediations = {row["sample_id"]: row for row in _read_jsonl(result_path)}
    for row in rows:
        remediation = remediations.get(row["sample_id"])
        if remediation is None:
            continue
        _require(
            remediation.get("candidate_text_changed") is False
            and remediation.get("generation_record_sha256")
            == row.get("generation_record_sha256"),
            "remediation changed or mismatched candidate",
        )
        row["hard_gate_status"] = remediation["amended_hard_gate_status"]
        row["all_hard_gates_pass"] = remediation["all_hard_gates_pass"]
        if remediation.get("remediated_stage") == "validator_b":
            row["validator_b"] = remediation["fresh_result"]
    return rows


def evaluate_checkpoint(
    generation_rows: Sequence[Mapping[str, Any]],
    judge_rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint: int,
) -> dict[str, Any]:
    _require(len(generation_rows) == len(judge_rows) and generation_rows, "row-count drift")
    generation_by_id = {row["sample_id"]: row for row in generation_rows}
    _require(len(generation_by_id) == len(generation_rows), "duplicate generation IDs")
    _require(
        {row["sample_id"] for row in judge_rows} == set(generation_by_id),
        "judge/generation ID mismatch",
    )
    zero_tolerance_failures: list[dict[str, Any]] = []
    factual_pass = 0
    for sample_id, generation in generation_by_id.items():
        metrics = generation["metrics"]
        failures = set(metrics.get("deterministic_hard_gate_failures") or [])
        fatal = sorted(failures & ZERO_TOLERANCE_FAILURES)
        if fatal:
            zero_tolerance_failures.append({"sample_id": sample_id, "reasons": fatal})
        if not failures.intersection(FACTUAL_FIELD_FAILURES):
            factual_pass += 1

    deterministic_eligible = [row for row in judge_rows if row.get("deterministic_pass") is True]
    a_pass_rows = [
        row for row in deterministic_eligible if row.get("validator_a", {}).get("status") == "PASS"
    ]
    b_pass_rows = [
        row for row in a_pass_rows if row.get("validator_b", {}).get("status") == "PASS"
    ]
    b_scores = [
        float(row["validator_b"]["mean_score"])
        for row in a_pass_rows
        if isinstance(row.get("validator_b", {}).get("mean_score"), (int, float))
    ]
    full_pass = sum(row.get("hard_gate_status") == "PASS_ALL_HARD_GATES" for row in judge_rows)
    unresolved = sum(
        row.get("hard_gate_status") == "UNRESOLVED_JUDGE_FAILURE" for row in judge_rows
    )
    cases = len(generation_rows)
    metrics = {
        "cases": cases,
        "zero_tolerance_format_degeneracy_failures": len(zero_tolerance_failures),
        "zero_tolerance_failure_details": zero_tolerance_failures,
        "deterministic_factual_pass_count": factual_pass,
        "deterministic_factual_pass_rate": factual_pass / cases,
        "validator_a_eligible_count": len(deterministic_eligible),
        "validator_a_pass_count": len(a_pass_rows),
        "validator_a_pass_rate_among_deterministic_pass": (
            len(a_pass_rows) / len(deterministic_eligible) if deterministic_eligible else 0.0
        ),
        "validator_b_eligible_count": len(a_pass_rows),
        "validator_b_pass_count": len(b_pass_rows),
        "validator_b_pass_rate_among_validator_a_pass": (
            len(b_pass_rows) / len(a_pass_rows) if a_pass_rows else 0.0
        ),
        "validator_b_mean_score": sum(b_scores) / len(b_scores) if b_scores else None,
        "validator_b_min_observed_mean_score": min(b_scores) if b_scores else None,
        "full_chain_pass_count": full_pass,
        "full_chain_pass_rate": full_pass / cases,
        "unresolved_judge_failures": unresolved,
    }
    checks = {
        "zero_tolerance_format_degeneracy": len(zero_tolerance_failures) == 0,
        "deterministic_factual_rate": metrics["deterministic_factual_pass_rate"]
        >= THRESHOLDS["minimum_deterministic_factual_pass_rate"],
        "validator_a_rate": metrics["validator_a_pass_rate_among_deterministic_pass"]
        >= THRESHOLDS["minimum_validator_a_pass_rate_among_deterministic_pass"],
        "validator_b_rate": metrics["validator_b_pass_rate_among_validator_a_pass"]
        >= THRESHOLDS["minimum_validator_b_pass_rate_among_validator_a_pass"],
        "validator_b_mean_score": metrics["validator_b_mean_score"] is not None
        and metrics["validator_b_mean_score"] >= THRESHOLDS["minimum_validator_b_mean_score"],
        "full_chain_rate": metrics["full_chain_pass_rate"]
        >= THRESHOLDS["minimum_full_chain_pass_rate"],
        "no_unresolved": unresolved <= THRESHOLDS["maximum_unresolved_judge_failures"],
    }
    return {
        "checkpoint": checkpoint,
        "metrics": metrics,
        "checks": checks,
        "relaxed_probe_pass": all(checks.values()),
        "strict_all_rows_pass": full_pass == cases,
    }


def select(output_root: Path, receipt_name: str = DEFAULT_RECEIPT_NAME) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    records: list[dict[str, Any]] = []
    bindings: dict[str, Any] = {}
    for checkpoint in probe.CHECKPOINTS:
        generation_root = output_root / "generations" / f"checkpoint-{checkpoint}"
        judge_root = output_root / "judges" / f"checkpoint-{checkpoint}"
        generation_summary = _read_json(generation_root / "summary.json")
        judge_summary = _read_json(judge_root / "summary.json")
        generation_path = generation_root / "results.jsonl"
        judge_path = judge_root / "results.jsonl"
        _require(
            sha256_file(generation_path) == generation_summary.get("results", {}).get("sha256"),
            f"checkpoint-{checkpoint} generation hash drift",
        )
        _require(
            sha256_file(judge_path) == judge_summary.get("results", {}).get("sha256"),
            f"checkpoint-{checkpoint} judge hash drift",
        )
        generation_rows = _read_jsonl(generation_path)
        judge_rows = _apply_remediation(output_root, checkpoint, _read_jsonl(judge_path))
        records.append(
            evaluate_checkpoint(generation_rows, judge_rows, checkpoint=checkpoint)
        )
        bindings[str(checkpoint)] = {
            "generation_results_sha256": sha256_file(generation_path),
            "judge_results_sha256": sha256_file(judge_path),
        }
    eligible = [record for record in records if record["relaxed_probe_pass"]]
    ranked = sorted(
        eligible,
        key=lambda record: (
            -record["metrics"]["full_chain_pass_rate"],
            -record["metrics"]["validator_b_mean_score"],
            -record["metrics"]["validator_a_pass_rate_among_deterministic_pass"],
            -record["metrics"]["deterministic_factual_pass_rate"],
            record["checkpoint"],
        ),
    )
    selected = ranked[0]["checkpoint"] if ranked else None
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "selected" if selected is not None else "no_checkpoint_passed",
        "policy": {
            "description": "tolerant checkpoint-level screening; per-sample strict verdicts unchanged",
            "thresholds": THRESHOLDS,
            "zero_tolerance_failure_codes": sorted(ZERO_TOLERANCE_FAILURES),
            "factual_field_failure_codes": sorted(FACTUAL_FIELD_FAILURES),
        },
        "records": records,
        "selected_checkpoint": selected,
        "selection_rank_order": [record["checkpoint"] for record in ranked],
        "bindings": bindings,
        "scope": {
            "panel_rows": 8,
            "split": "validation",
            "selection_only": True,
            "evaluation_eligible": False,
            "suitable_for_leakage_safe_evaluation": False,
            "test_generation_performed": False,
            "not_a_merge_or_promotion_authorization": True,
        },
    }
    receipt["receipt_sha256"] = _sha256_text(_canonical_json(receipt))
    _write_exclusive_json(output_root / receipt_name, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--receipt-name", default=DEFAULT_RECEIPT_NAME)
    args = parser.parse_args(argv)
    result = select(args.output_root, args.receipt_name)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
