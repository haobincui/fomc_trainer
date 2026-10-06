"""Run fresh Validator A then Validator B on paper-chk2 probe generations.

This stage consumes only validation generations that already passed the local
deterministic gate.  Validator A sees source analysis, student native
reasoning, and student rewritten Minutes.  Only after A passes does Validator B
see the rewritten Minutes and the sealed corresponding-meeting pre-action
official reference.  The historical teacher-target validator reports are not
reused.

The legacy field name ``teacher_response_analysis`` is used only as a schema
compatibility alias for ``student_native_reasoning`` and is disclosed in every
result.  No candidate repair is performed during a checkpoint probe; the only
permitted repair is the existing one-shot validator-report contract repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from jobs.generation import generate_paper_chk2_chk1_analysis_rewrite as validator
from jobs.generation.paper_chk2_official_reference_v2 import (
    deserialize_official_reference_bank,
)
from jobs.retrain_v2 import probe_paper_chk2_minutes_checkpoints as probe
from open_r1.provenance import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = probe.DEFAULT_OUTPUT_ROOT
REFERENCE_BANK = probe.RELEASE_ROOT / "provenance/official_pre_action_reference_bank.jsonl"
REFERENCE_BANK_SHA256 = (
    "c3a4a7d7dfb3c5df896161e1a12908ec11083345287b9159cdc3eebeff133d2b"
)
SCHEMA = "paper-chk2-minutes-checkpoint-fresh-judges-v1"
RESULT_SCHEMA = "paper-chk2-minutes-checkpoint-fresh-judge-result-v1"
SUMMARY_SCHEMA = "paper-chk2-minutes-checkpoint-fresh-judge-summary-v1"
MAX_CONCURRENCY = 8


class PaperChk2JudgeError(RuntimeError):
    """Fresh semantic/style checkpoint judging failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperChk2JudgeError(message)


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


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _require(bool(raw), f"blank row in {label}:{number}")
        value = json.loads(raw)
        _require(isinstance(value, dict), f"non-object row in {label}:{number}")
        rows.append(value)
    return rows


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    _require(not path.exists() and not path.is_symlink(), f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _implementation_sha256() -> str:
    artifacts = {
        "judge": sha256_file(Path(__file__).resolve()),
        "probe": sha256_file(Path(probe.__file__).resolve()),
        "validator": sha256_file(Path(validator.__file__).resolve()),
    }
    return _sha256_text(_canonical_json(artifacts))


def _dimensions(result: Mapping[str, Any]) -> tuple[float | None, int | None]:
    dimensions = result.get("dimensions")
    if not isinstance(dimensions, Mapping):
        return None, None
    scores = [
        item.get("score")
        for item in dimensions.values()
        if isinstance(item, Mapping) and isinstance(item.get("score"), int)
    ]
    if len(scores) != len(validator.STYLE_DIMENSIONS):
        return None, None
    return sum(scores) / len(scores), min(scores)


def classify_gate_result(
    *,
    deterministic_pass: bool,
    validator_a_pass: bool | None,
    validator_a_unresolved: bool,
    validator_b_pass: bool | None,
    validator_b_unresolved: bool,
) -> str:
    if not deterministic_pass:
        return "FAIL_DETERMINISTIC"
    if validator_a_unresolved or validator_b_unresolved:
        return "UNRESOLVED_JUDGE_FAILURE"
    if validator_a_pass is not True:
        return "FAIL_VALIDATOR_A"
    if validator_b_pass is not True:
        return "FAIL_VALIDATOR_B"
    return "PASS_ALL_HARD_GATES"


def _row_and_candidate(record: Mapping[str, Any], checkpoint: int) -> tuple[Any, Any]:
    sample_id = str(record["sample_id"])
    source = str(record["source_analysis"])
    reasoning = str(record["student_native_reasoning"])
    rewritten = str(record["rewritten_minutes"])
    row = SimpleNamespace(
        sample_id=sample_id,
        split="validation",
        meeting_date=str(record["meeting_date"]),
        source_analysis=source,
        source_analysis_sha256=_sha256_text(source),
        # Validator A/B never receive provided_data.  This digest is only a
        # cache-binding sentinel required by the inherited immutable cache.
        provided_data_sha256=_sha256_text("PROBE_PROVIDED_DATA_FORBIDDEN"),
    )
    candidate = validator.Candidate(
        teacher_response_analysis=reasoning,
        rewritten_minutes=rewritten,
        sft_response=str(record["completion"]),
        attempt=f"student_checkpoint_{checkpoint}",
        provider={
            "provider": "local_transformers_peft",
            "checkpoint": checkpoint,
            "completion_sha256": record["completion_sha256"],
        },
        deterministic_validation=dict(record["metrics"]),
    )
    return row, candidate


def _judge_one(
    record: Mapping[str, Any],
    *,
    checkpoint: int,
    stage_root: Path,
    backend: Any,
    identity: Any,
    config: Any,
    code_sha256: str,
    reference_bank: Any,
) -> dict[str, Any]:
    metrics = record.get("metrics")
    _require(isinstance(metrics, Mapping), "generation metrics missing")
    deterministic_pass = metrics.get("deterministic_hard_gate_pass") is True
    base = {
        "schema_version": RESULT_SCHEMA,
        "sample_id": record["sample_id"],
        "checkpoint": checkpoint,
        "generation_record_sha256": _sha256_text(_canonical_json(record)),
        "completion_sha256": record["completion_sha256"],
        "deterministic_pass": deterministic_pass,
        "deterministic_failures": list(metrics.get("deterministic_hard_gate_failures") or []),
        "student_reasoning_compatibility_alias": {
            "persisted_field": "student_native_reasoning",
            "validator_a_legacy_field": "teacher_response_analysis",
            "texts_identical": True,
        },
    }
    if not deterministic_pass:
        return {
            **base,
            "validator_a": {"status": "NOT_RUN_DUE_TO_DETERMINISTIC_FAIL"},
            "validator_b": {"status": "NOT_RUN_DUE_TO_DETERMINISTIC_FAIL"},
            "hard_gate_status": "FAIL_DETERMINISTIC",
            "all_hard_gates_pass": False,
        }
    row, candidate = _row_and_candidate(record, checkpoint)
    try:
        a_result, a_pass, a_reasons, a_providers = validator._validator_a_call(
            row,
            candidate,
            role=validator.ROLE_VALIDATOR_A_PRIMARY,
            output=stage_root,
            backend=backend,
            identity=identity,
            environment=None,
            config=config,
            code_sha256=code_sha256,
        )
        a_unresolved = a_result.get("contract_exhausted") is True
        a_payload = {
            "status": "UNRESOLVED" if a_unresolved else "PASS" if a_pass else "FAIL",
            "machine_pass": a_pass,
            "reasons": list(a_reasons),
            "result": a_result,
            "provider": a_providers,
        }
    except Exception as exc:
        return {
            **base,
            "validator_a": {
                "status": "UNRESOLVED",
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            },
            "validator_b": {"status": "NOT_RUN_DUE_TO_VALIDATOR_A_UNRESOLVED"},
            "hard_gate_status": "UNRESOLVED_JUDGE_FAILURE",
            "all_hard_gates_pass": False,
        }
    if a_unresolved:
        return {
            **base,
            "validator_a": a_payload,
            "validator_b": {"status": "NOT_RUN_DUE_TO_VALIDATOR_A_UNRESOLVED"},
            "hard_gate_status": "UNRESOLVED_JUDGE_FAILURE",
            "all_hard_gates_pass": False,
        }
    if not a_pass:
        return {
            **base,
            "validator_a": a_payload,
            "validator_b": {"status": "NOT_RUN_DUE_TO_VALIDATOR_A_FAIL"},
            "hard_gate_status": "FAIL_VALIDATOR_A",
            "all_hard_gates_pass": False,
        }
    try:
        b_result, b_pass, b_reasons, b_providers = validator._validator_b_call(
            row,
            candidate,
            reference_bank,
            role=validator.ROLE_VALIDATOR_B_PRIMARY,
            output=stage_root,
            backend=backend,
            identity=identity,
            environment=None,
            config=config,
            code_sha256=code_sha256,
            official_reference_bank_sha256=REFERENCE_BANK_SHA256,
        )
        b_unresolved = b_result.get("contract_exhausted") is True
        mean_score, min_score = _dimensions(b_result)
        b_payload = {
            "status": "UNRESOLVED" if b_unresolved else "PASS" if b_pass else "FAIL",
            "machine_pass": b_pass,
            "mean_score": mean_score,
            "min_score": min_score,
            "reasons": list(b_reasons),
            "result": b_result,
            "provider": b_providers,
        }
    except Exception as exc:
        return {
            **base,
            "validator_a": a_payload,
            "validator_b": {
                "status": "UNRESOLVED",
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            },
            "hard_gate_status": "UNRESOLVED_JUDGE_FAILURE",
            "all_hard_gates_pass": False,
        }
    hard_status = classify_gate_result(
        deterministic_pass=True,
        validator_a_pass=a_pass,
        validator_a_unresolved=False,
        validator_b_pass=b_pass,
        validator_b_unresolved=b_unresolved,
    )
    return {
        **base,
        "validator_a": a_payload,
        "validator_b": b_payload,
        "hard_gate_status": hard_status,
        "all_hard_gates_pass": hard_status == "PASS_ALL_HARD_GATES",
    }


def summarize(rows: Sequence[Mapping[str, Any]], checkpoint: int) -> dict[str, Any]:
    statuses = Counter(str(row["hard_gate_status"]) for row in rows)
    b_scores = [
        float(row["validator_b"]["mean_score"])
        for row in rows
        if isinstance(row.get("validator_b"), Mapping)
        and isinstance(row["validator_b"].get("mean_score"), (int, float))
    ]
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete" if not statuses.get("UNRESOLVED_JUDGE_FAILURE") else "unresolved",
        "checkpoint": checkpoint,
        "cases": len(rows),
        "hard_gate_status_counts": dict(sorted(statuses.items())),
        "all_hard_gates_pass_count": statuses.get("PASS_ALL_HARD_GATES", 0),
        "all_hard_gates_pass_rate": statuses.get("PASS_ALL_HARD_GATES", 0) / len(rows),
        "checkpoint_pass": statuses == Counter({"PASS_ALL_HARD_GATES": len(rows)}),
        "unresolved_count": statuses.get("UNRESOLVED_JUDGE_FAILURE", 0),
        "validator_b_mean_score_mean": sum(b_scores) / len(b_scores) if b_scores else None,
        "validator_b_mean_score_min": min(b_scores) if b_scores else None,
        "ordering": "deterministic_gates_then_validator_a_then_validator_b",
        "selection_only": True,
        "evaluation_eligible": False,
        "suitable_for_leakage_safe_evaluation": False,
        "test_generation_performed": False,
    }


def run_judges(
    *, output_root: Path, checkpoint: int, concurrency: int
) -> dict[str, Any]:
    _require(checkpoint in probe.CHECKPOINTS, f"unsupported checkpoint: {checkpoint}")
    _require(1 <= concurrency <= MAX_CONCURRENCY, "invalid concurrency")
    _require(bool(os.environ.get("DEEPSEEK_API_KEY")), "DEEPSEEK_API_KEY is not configured")
    output_root = output_root.expanduser().resolve()
    generation_root = output_root / "generations" / f"checkpoint-{checkpoint}"
    generation_summary = _read_json(generation_root / "summary.json", label="generation summary")
    result_path = generation_root / "results.jsonl"
    _require(
        sha256_file(result_path) == generation_summary.get("results", {}).get("sha256"),
        "generation result hash drift",
    )
    generation_rows = _read_jsonl(result_path, label="generation results")
    _require(len(generation_rows) == generation_summary.get("cases"), "generation row drift")
    _require(sha256_file(REFERENCE_BANK) == REFERENCE_BANK_SHA256, "reference bank hash drift")
    reference_bank = deserialize_official_reference_bank(REFERENCE_BANK.read_bytes())
    code_sha256 = _implementation_sha256()
    target = output_root / "judges" / f"checkpoint-{checkpoint}"
    _require(not target.exists() and not target.is_symlink(), f"judge output exists: {target}")
    stage = target.parent / f".{target.name}.staging-{os.getpid()}"
    _require(not stage.exists() and not stage.is_symlink(), f"judge staging exists: {stage}")
    stage.mkdir(parents=True, mode=0o700)
    config = validator.ProviderConfig()
    backend = validator.OpenAICompatibleBackend()
    identity = validator.ProviderIdentityRegistry()
    launch = {
        "schema_version": SCHEMA,
        "status": "running",
        "checkpoint": checkpoint,
        "cases": len(generation_rows),
        "concurrency": concurrency,
        "implementation_sha256": code_sha256,
        "generation_results_sha256": sha256_file(result_path),
        "official_reference_bank_sha256": REFERENCE_BANK_SHA256,
        "provider_contract": config.contract(),
        "credential": {
            "env_name": "DEEPSEEK_API_KEY",
            "present": True,
            "plaintext_persisted": False,
        },
        "field_isolation": {
            "validator_a_fields": [
                "source_analysis",
                "teacher_response_analysis",
                "rewritten_minutes",
            ],
            "validator_b_fields": [
                "rewritten_minutes",
                "corresponding_official_minutes_pre_action",
            ],
            "validator_a_saw_official_reference": False,
            "validator_b_saw_source_or_reasoning": False,
        },
    }
    _write_exclusive_json(stage / "launch.json", launch)
    indexed: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _judge_one,
                row,
                checkpoint=checkpoint,
                stage_root=stage,
                backend=backend,
                identity=identity,
                config=config,
                code_sha256=code_sha256,
                reference_bank=reference_bank,
            ): index
            for index, row in enumerate(generation_rows)
        }
        for future in as_completed(futures):
            index = futures[future]
            indexed[index] = future.result()
            row = indexed[index]
            print(
                f"checkpoint={checkpoint} judged={len(indexed)}/{len(generation_rows)} "
                f"sample={row['sample_id']} status={row['hard_gate_status']}",
                flush=True,
            )
    results = [indexed[index] for index in range(len(generation_rows))]
    result_payload = "".join(_canonical_json(row) + "\n" for row in results).encode("utf-8")
    result_output = stage / "results.jsonl"
    descriptor = os.open(result_output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(result_payload)
        handle.flush()
        os.fsync(handle.fileno())
    summary = summarize(results, checkpoint)
    summary = {
        **summary,
        "provider_identity": identity.as_dict(),
        "results": {
            "path": "results.jsonl",
            "rows": len(results),
            "sha256": sha256_file(result_output),
        },
    }
    _write_exclusive_json(stage / "summary.json", summary)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, target)
    return summary


def remediate_unresolved(
    *, output_root: Path, checkpoint: int
) -> dict[str, Any]:
    """Re-call only unresolved judge reports without changing model text."""

    _require(checkpoint in probe.CHECKPOINTS, f"unsupported checkpoint: {checkpoint}")
    _require(bool(os.environ.get("DEEPSEEK_API_KEY")), "DEEPSEEK_API_KEY is not configured")
    output_root = output_root.expanduser().resolve()
    original_root = output_root / "judges" / f"checkpoint-{checkpoint}"
    original_result_path = original_root / "results.jsonl"
    original_summary = _read_json(original_root / "summary.json", label="original judge summary")
    original_rows = _read_jsonl(original_result_path, label="original judge results")
    _require(
        sha256_file(original_result_path) == original_summary.get("results", {}).get("sha256"),
        "original judge result hash drift",
    )
    unresolved = [
        row for row in original_rows if row.get("hard_gate_status") == "UNRESOLVED_JUDGE_FAILURE"
    ]
    _require(bool(unresolved), "no unresolved judge rows to remediate")
    generation_rows = {
        row["sample_id"]: row
        for row in _read_jsonl(
            output_root / "generations" / f"checkpoint-{checkpoint}/results.jsonl",
            label="generation results",
        )
    }
    _require(sha256_file(REFERENCE_BANK) == REFERENCE_BANK_SHA256, "reference bank hash drift")
    reference_bank = deserialize_official_reference_bank(REFERENCE_BANK.read_bytes())
    target = output_root / "judges" / f"checkpoint-{checkpoint}-unresolved-remediation-v1"
    _require(not target.exists() and not target.is_symlink(), f"remediation output exists: {target}")
    stage = target.parent / f".{target.name}.staging-{os.getpid()}"
    _require(not stage.exists() and not stage.is_symlink(), f"remediation staging exists: {stage}")
    stage.mkdir(parents=True, mode=0o700)
    code_sha256 = _implementation_sha256()
    config = validator.ProviderConfig()
    backend = validator.OpenAICompatibleBackend()
    identity = validator.ProviderIdentityRegistry()
    remediation_rows: list[dict[str, Any]] = []
    for original in unresolved:
        sample_id = str(original["sample_id"])
        generation = generation_rows.get(sample_id)
        _require(isinstance(generation, Mapping), f"missing generation: {sample_id}")
        row, candidate = _row_and_candidate(generation, checkpoint)
        row.sample_id = f"{sample_id}::validator-contract-remediation-v1"
        a_status = original.get("validator_a", {}).get("status")
        if a_status == "PASS" and original.get("validator_b", {}).get("status") == "UNRESOLVED":
            b_result, b_pass, b_reasons, b_providers = validator._validator_b_call(
                row,
                candidate,
                reference_bank,
                role=validator.ROLE_VALIDATOR_B_PRIMARY,
                output=stage,
                backend=backend,
                identity=identity,
                environment=None,
                config=config,
                code_sha256=code_sha256,
                official_reference_bank_sha256=REFERENCE_BANK_SHA256,
            )
            b_unresolved = b_result.get("contract_exhausted") is True
            mean_score, min_score = _dimensions(b_result)
            hard_status = (
                "UNRESOLVED_JUDGE_FAILURE"
                if b_unresolved
                else "PASS_ALL_HARD_GATES"
                if b_pass
                else "FAIL_VALIDATOR_B"
            )
            fresh = {
                "status": "UNRESOLVED" if b_unresolved else "PASS" if b_pass else "FAIL",
                "machine_pass": b_pass,
                "mean_score": mean_score,
                "min_score": min_score,
                "reasons": list(b_reasons),
                "result": b_result,
                "provider": b_providers,
            }
            stage_name = "validator_b"
        else:
            raise PaperChk2JudgeError(
                f"unsupported unresolved remediation state for {sample_id}: "
                f"A={a_status} B={original.get('validator_b', {}).get('status')}"
            )
        remediation_rows.append(
            {
                "schema_version": RESULT_SCHEMA,
                "sample_id": sample_id,
                "checkpoint": checkpoint,
                "remediated_stage": stage_name,
                "candidate_text_changed": False,
                "generation_record_sha256": original["generation_record_sha256"],
                "original_judge_result_file_sha256": sha256_file(original_result_path),
                "fresh_result": fresh,
                "amended_hard_gate_status": hard_status,
                "all_hard_gates_pass": hard_status == "PASS_ALL_HARD_GATES",
            }
        )
        print(
            f"checkpoint={checkpoint} remediated={sample_id} status={hard_status}",
            flush=True,
        )
    amended_by_id = {
        row["sample_id"]: row["amended_hard_gate_status"] for row in remediation_rows
    }
    amended_statuses = Counter(
        amended_by_id.get(str(row["sample_id"]), str(row["hard_gate_status"]))
        for row in original_rows
    )
    result_output = stage / "results.jsonl"
    payload = "".join(_canonical_json(row) + "\n" for row in remediation_rows).encode("utf-8")
    descriptor = os.open(result_output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "complete" if not amended_statuses.get("UNRESOLVED_JUDGE_FAILURE") else "unresolved",
        "checkpoint": checkpoint,
        "original_judge_results_sha256": sha256_file(original_result_path),
        "remediated_rows": len(remediation_rows),
        "candidate_text_changed": False,
        "amended_hard_gate_status_counts": dict(sorted(amended_statuses.items())),
        "amended_all_hard_gates_pass_count": amended_statuses.get("PASS_ALL_HARD_GATES", 0),
        "amended_checkpoint_pass": amended_statuses == Counter({"PASS_ALL_HARD_GATES": len(original_rows)}),
        "unresolved_count": amended_statuses.get("UNRESOLVED_JUDGE_FAILURE", 0),
        "provider_identity": identity.as_dict(),
        "results": {
            "path": "results.jsonl",
            "rows": len(remediation_rows),
            "sha256": sha256_file(result_output),
        },
    }
    _write_exclusive_json(stage / "summary.json", summary)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, target)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=int, choices=probe.CHECKPOINTS, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--remediate-unresolved", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.remediate_unresolved:
        result = remediate_unresolved(
            output_root=args.output_root,
            checkpoint=args.checkpoint,
        )
    else:
        result = run_judges(
            output_root=args.output_root,
            checkpoint=args.checkpoint,
            concurrency=args.concurrency,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
