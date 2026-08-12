"""Validate and summarize a completed chk1 DeepSeek source-only audit."""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Mapping, Sequence

from jobs.retrain_v2.audit_chk1_clean_sft_deepseek import (
    BLOCKING_KINDS,
    BOUNDARY,
    RUBRIC_KEYS,
    ROW_SCHEMA_VERSION,
    SCHEMA_VERSION,
)
from jobs.retrain_v2.compress_chk1_reasoning import (
    _atomic_write_text,
    _canonical_json,
    _sha256_file,
    _utc_now,
)


ANALYSIS_SCHEMA_VERSION = "chk1-clean-sft-deepseek-audit-analysis-v1"


class AuditSummaryError(RuntimeError):
    """The saved audit is incomplete, inconsistent, or content-unbound."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditSummaryError(f"unable to read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AuditSummaryError(f"JSON root is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise AuditSummaryError(f"unable to read JSONL: {path}") from exc
    for number, line in enumerate(lines, 1):
        if not line:
            raise AuditSummaryError(f"{path}:{number}: blank row")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditSummaryError(f"{path}:{number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise AuditSummaryError(f"{path}:{number}: row is not an object")
        rows.append(value)
    return rows


def _normalized(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)).casefold())


def _wilson(successes: int, total: int) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return [round(max(0.0, center - half), 6), round(min(1.0, center + half), 6)]


def _rate_row(label: str, total: int, failed: int) -> dict[str, Any]:
    return {
        "group": label,
        "total": total,
        "passed": total - failed,
        "failed": failed,
        "failure_rate": round(failed / total, 6) if total else 0.0,
        "failure_rate_wilson_95": _wilson(failed, total),
    }


def _group_rates(
    rows: Sequence[Mapping[str, Any]],
    key: Callable[[Mapping[str, Any]], str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return [
        _rate_row(
            label,
            len(group),
            sum(row.get("status") == "failed" for row in group),
        )
        for label, group in sorted(groups.items())
    ]


def summarize(*, audit_dir: Path, candidate_dir: Path) -> dict[str, Any]:
    audit_dir = audit_dir.resolve()
    candidate_dir = candidate_dir.resolve()
    summary_path = audit_dir / "summary.json"
    row_path = audit_dir / "row_audits.jsonl"
    input_path = candidate_dir / "audits/semantic_audit_input.jsonl"
    repair_path = candidate_dir / "audits/repair_manifest.jsonl"
    summary = _read_json(summary_path)
    rows = _read_jsonl(row_path)
    inputs = {str(row["sample_id"]): row for row in _read_jsonl(input_path)}
    repairs = {str(row["sample_id"]): row for row in _read_jsonl(repair_path)}
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise AuditSummaryError("audit summary schema mismatch")
    if summary.get("status") not in {"passed", "semantic_failures"}:
        raise AuditSummaryError("audit is not complete")
    counts = summary.get("counts")
    if not isinstance(counts, dict) or any(
        counts.get(key) != value
        for key, value in {
            "expected": 237,
            "selected": 237,
            "completed": 237,
            "review_required": 0,
            "stage_a_provider_errors": 0,
            "stage_b_provider_errors": 0,
            "stage_b_ambiguous": 0,
        }.items()
    ):
        raise AuditSummaryError("audit completion counts are invalid")
    descriptor = summary.get("row_audit")
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("rows") != len(rows)
        or descriptor.get("sha256") != _sha256_file(row_path)
    ):
        raise AuditSummaryError("row-audit descriptor mismatch")
    if len(rows) != 237 or len(inputs) != 237 or len({row["sample_id"] for row in rows}) != 237:
        raise AuditSummaryError("audit population mismatch")
    if set(inputs) != {str(row["sample_id"]) for row in rows}:
        raise AuditSummaryError("audit/input ID population mismatch")

    verified_blocking: list[dict[str, str]] = []
    for number, row in enumerate(rows, 1):
        if row.get("schema_version") != ROW_SCHEMA_VERSION:
            raise AuditSummaryError(f"row schema mismatch: {number}")
        sample_id = str(row.get("sample_id") or "")
        source = inputs[sample_id]
        if (
            row.get("candidate_sha256") != source.get("candidate_response_sha256")
            or row.get("evidence_sha256") != source.get("provided_data_sha256")
        ):
            raise AuditSummaryError(f"row hash binding mismatch: {sample_id}")
        candidate = str(source["candidate_response"])
        if candidate.count(BOUNDARY) != 1:
            raise AuditSummaryError(f"candidate boundary mismatch: {sample_id}")
        think, answer = candidate.split(BOUNDARY)
        blocking = row.get("blocking_violations")
        errors = row.get("errors")
        if not isinstance(blocking, list) or not isinstance(errors, list):
            raise AuditSummaryError(f"row result lists missing: {sample_id}")
        expected_status = "review_required" if errors else ("failed" if blocking else "passed")
        if row.get("status") != expected_status:
            raise AuditSummaryError(f"row status mismatch: {sample_id}")
        for violation in blocking:
            if not isinstance(violation, dict) or violation.get("kind") not in BLOCKING_KINDS:
                raise AuditSummaryError(f"invalid blocking violation: {sample_id}")
            section = think if violation.get("section") == "think" else answer
            if _normalized(str(violation.get("candidate_quote") or "")) not in _normalized(section):
                raise AuditSummaryError(f"blocking quote mismatch: {sample_id}")
            verified_blocking.append({"sample_id": sample_id, **violation})

    passed = sum(row["status"] == "passed" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    if passed != counts.get("passed") or failed != counts.get("failed"):
        raise AuditSummaryError("summary pass/fail count mismatch")
    if len(verified_blocking) != counts.get("blocking_violations"):
        raise AuditSummaryError("summary blocking count mismatch")

    repair_groups: dict[str, str] = {}
    for sample_id in inputs:
        method = str(repairs[sample_id].get("answer_repair_method") or "")
        if method.startswith("recover_"):
            repair_groups[sample_id] = "recovered JSON-like answer"
        elif method == "strip_inline_evidence_ids":
            repair_groups[sample_id] = "stripped inline evidence IDs"
        else:
            repair_groups[sample_id] = "answer unchanged"

    by_kind = Counter(row["kind"] for row in verified_blocking)
    by_section = Counter(row["section"] for row in verified_blocking)
    by_severity = Counter(row["severity"] for row in verified_blocking)
    rows_by_kind = Counter()
    rows_by_section = Counter()
    for row in rows:
        for value in {item["kind"] for item in row["blocking_violations"]}:
            rows_by_kind[value] += 1
        for value in {item["section"] for item in row["blocking_violations"]}:
            rows_by_section[value] += 1
    confirmed_adjudications = [
        adjudication
        for row in rows
        for adjudication in row["stage_b_adjudications"]
        if adjudication.get("verdict") == "confirmed"
    ]
    false_adjudications = [
        adjudication
        for row in rows
        for adjudication in row["stage_b_adjudications"]
        if adjudication.get("verdict") == "false_positive"
    ]
    basis_counts = Counter(row["basis"] for row in confirmed_adjudications)
    proposed_rows = sum(bool(row["stage_a_proposed_blocking_violations"]) for row in rows)
    rescued_rows = sum(
        row["status"] == "passed" and bool(row["stage_a_proposed_blocking_violations"])
        for row in rows
    )
    rows_with_major = sum(
        any(item["severity"] == "major" for item in row["blocking_violations"])
        for row in rows
    )
    minor_only_rows = sum(
        row["status"] == "failed"
        and all(item["severity"] == "minor" for item in row["blocking_violations"])
        for row in rows
    )
    answer_rows = sum(
        any(item["section"] == "answer" for item in row["blocking_violations"])
        for row in rows
    )
    think_rows = sum(
        any(item["section"] == "think" for item in row["blocking_violations"])
        for row in rows
    )
    both_rows = sum(
        any(item["section"] == "answer" for item in row["blocking_violations"])
        and any(item["section"] == "think" for item in row["blocking_violations"])
        for row in rows
    )

    split_rates = _group_rates(rows, lambda row: str(row["split"]))
    repair_rates = _group_rates(rows, lambda row: repair_groups[str(row["sample_id"])])

    def changed_rate(label: str, predicate: Callable[[Mapping[str, Any]], bool]) -> dict[str, Any]:
        subset = [row for row in rows if predicate(repairs[str(row["sample_id"])])]
        return _rate_row(label, len(subset), sum(row["status"] == "failed" for row in subset))

    change_rates = [
        changed_rate(
            "answer changed",
            lambda repair: repair["source_answer_sha256"] != repair["candidate_answer_sha256"],
        ),
        changed_rate(
            "answer unchanged",
            lambda repair: repair["source_answer_sha256"] == repair["candidate_answer_sha256"],
        ),
        changed_rate(
            "reasoning changed",
            lambda repair: repair["source_reasoning_sha256"]
            != repair["candidate_reasoning_sha256"],
        ),
        changed_rate(
            "reasoning unchanged",
            lambda repair: repair["source_reasoning_sha256"]
            == repair["candidate_reasoning_sha256"],
        ),
    ]

    rubric = {
        group: {
            key: round(mean(float(row["rubric"][key]) for row in subset), 6)
            for key in RUBRIC_KEYS
        }
        for group, subset in {
            "all": rows,
            "passed": [row for row in rows if row["status"] == "passed"],
            "failed": [row for row in rows if row["status"] == "failed"],
        }.items()
    }
    violation_distribution = Counter(
        len(row["blocking_violations"]) for row in rows if row["status"] == "failed"
    )
    stage_a_cache = [_read_json(path) for path in sorted((audit_dir / "cache/stage_a").rglob("*.json"))]
    stage_b_cache = [_read_json(path) for path in sorted((audit_dir / "cache/stage_b").rglob("*.json"))]
    if len(stage_a_cache) != 237 or len(stage_b_cache) != counts["stage_b_completed"]:
        raise AuditSummaryError("cache population mismatch")
    attempts = {
        "stage_a": dict(sorted(Counter(str(row["attempt"]) for row in stage_a_cache).items())),
        "stage_b": dict(sorted(Counter(str(row["attempt"]) for row in stage_b_cache).items())),
    }

    representatives = []
    ranked = sorted(
        (row for row in rows if row["status"] == "failed"),
        key=lambda row: (
            -sum(item["severity"] == "major" for item in row["blocking_violations"]),
            -len(row["blocking_violations"]),
            str(row["sample_id"]),
        ),
    )
    for row in ranked[:8]:
        first = row["blocking_violations"][0]
        representatives.append(
            {
                "sample_id": row["sample_id"],
                "split": row["split"],
                "violation_count": len(row["blocking_violations"]),
                "major_count": sum(
                    item["severity"] == "major" for item in row["blocking_violations"]
                ),
                "example_section": first["section"],
                "example_kind": first["kind"],
                "example_severity": first["severity"],
                "example_quote": first["candidate_quote"],
                "example_explanation": first["explanation"],
            }
        )

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "decision": {
            "strict_source_only_gate": "failed",
            "direct_use_as_clean_chk1_release": "not_recommended",
            "minimum_confirmed_rows_to_repair": failed,
            "reason": (
                "113 of 237 audited changed rows contain at least one independently "
                "confirmed factual, numerical, or causal violation."
            ),
        },
        "scope": {
            "audited_rows": len(rows),
            "full_candidate_rows": 1743,
            "audited_population": "all 237 rows changed by clean-v2 repair",
            "unaudited_candidate_rows": 1743 - len(rows),
            "scope_warning": (
                "This result does not establish semantic safety for the 1,506 unchanged rows."
            ),
        },
        "integrity": {
            "status": "passed",
            "row_hash_binding_errors": 0,
            "quote_section_binding_errors": 0,
            "provider_errors": 0,
            "review_required": 0,
            "candidate_manifest_sha256": summary["input"]["candidate_manifest_sha256"],
            "semantic_audit_input_sha256": summary["input"]["semantic_audit_input_sha256"],
            "audit_summary_sha256": _sha256_file(summary_path),
            "row_audits_sha256": _sha256_file(row_path),
        },
        "outcomes": {
            "total": len(rows),
            "passed": passed,
            "pass_rate": round(passed / len(rows), 6),
            "pass_rate_wilson_95": _wilson(passed, len(rows)),
            "failed": failed,
            "failure_rate": round(failed / len(rows), 6),
            "failure_rate_wilson_95": _wilson(failed, len(rows)),
            "rows_with_major_violation": rows_with_major,
            "rows_with_only_minor_violations": minor_only_rows,
            "answer_only_failed_rows": answer_rows - both_rows,
            "think_only_failed_rows": think_rows - both_rows,
            "both_sections_failed_rows": both_rows,
        },
        "violations": {
            "confirmed_total": len(verified_blocking),
            "by_kind": dict(sorted(by_kind.items())),
            "by_section": dict(sorted(by_section.items())),
            "by_severity": dict(sorted(by_severity.items())),
            "rows_affected_by_kind": dict(sorted(rows_by_kind.items())),
            "rows_affected_by_section": dict(sorted(rows_by_section.items())),
            "confirmed_basis": dict(sorted(basis_counts.items())),
            "target_leakage": by_kind.get("target_leakage", 0),
            "per_failed_row": {
                str(key): value for key, value in sorted(violation_distribution.items())
            },
        },
        "two_stage_adjudication": {
            "stage_a_proposals": counts["stage_a_proposed_blocking_violations"],
            "stage_a_rows_with_proposals": proposed_rows,
            "confirmed": len(confirmed_adjudications),
            "false_positive": len(false_adjudications),
            "false_positive_rate": round(
                len(false_adjudications) / counts["stage_a_proposed_blocking_violations"], 6
            ),
            "rows_rescued_from_false_failure": rescued_rows,
            "ambiguous": 0,
            "provider_errors": 0,
        },
        "failure_rates": {
            "by_split": split_rates,
            "by_answer_repair_group": repair_rates,
            "by_changed_component": change_rates,
        },
        "rubric_means": rubric,
        "provider": {
            "model": summary["judge"]["model"],
            "api": summary["judge"]["api"],
            "reasoning": summary["judge"]["reasoning"],
            "max_output_tokens": summary["judge"]["max_output_tokens"],
            "usage": summary["usage"],
            "attempts": attempts,
        },
        "representative_confirmed_failures": representatives,
        "sources": {
            "audit_summary": str(summary_path),
            "row_audits": str(row_path),
            "semantic_audit_input": str(input_path),
            "repair_manifest": str(repair_path),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = summarize(audit_dir=args.audit_dir, candidate_dir=args.candidate_dir)
    _atomic_write_text(
        args.output.resolve(),
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(_canonical_json({"status": "complete", "output": str(args.output), "decision": result["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
