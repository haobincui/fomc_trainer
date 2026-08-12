"""Validate and summarize the sealed chk0 versus chk1 checkpoint-200 evaluation.

This command performs no model inference and no scoring.  It accepts the two
already-completed scoring bundles (strict primary analysis and length-tolerant
robustness analysis), validates their sealed audits and bound artifacts, and
writes one immutable JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from jobs.generation.generate_checkpoint_artifact import (
    MANIFEST_SCHEMA_VERSION as GENERATION_MANIFEST_SCHEMA_VERSION,
    validate_generation_progress_binding,
)
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    manifest_payload_sha256,
    seal_manifest,
    validate_manifest_integrity,
)


SCHEMA_VERSION = "chk0-chk1-cp200-generation-evaluation-summary-v1"
EVALUATOR_SCHEMA_VERSION = "checkpoint-generation-evaluation-v1"
WRAPPER_SCHEMA_VERSION = "retrain-v2-chk0-chk1-generation-evaluation-v1"
BASELINE_ARTIFACT_ID = "eval-chk0-base"
CANDIDATE_ARTIFACT_ID = "eval-chk1-clean-v2-lr1e6-cp200"
EXPECTED_ARTIFACT_IDS = (BASELINE_ARTIFACT_ID, CANDIDATE_ARTIFACT_ID)
EXPECTED_ROWS_PER_ARTIFACT = 33
EXPECTED_BOOTSTRAP_SAMPLES = 10_000
EXPECTED_BOOTSTRAP_SEED = 20260729
EXPECTED_SUBSET_ROWS = {
    "all_11_meetings": 33,
    "prospective_only_9_meetings": 27,
}
POLICY_SPECS = {
    "strict": {
        "policy_id": "strict-final-answer-v2",
        "analysis_role": "primary",
    },
    "length_tolerant": {
        "policy_id": "length-tolerant-open-tags-v1",
        "analysis_role": "diagnostic_robustness_only",
    },
}
METRIC_GROUPS = {
    "semantic": (
        "bertscore_f1",
        "bertscore_precision",
        "bertscore_recall",
        "mpnet_cosine",
    ),
    "surface": (
        "rouge_l_f1",
        "generated_token_count",
        "length_ratio",
        "repetition_rate",
        "format_compliance",
    ),
    "factual_numeric": (
        "numeric_value_accuracy",
        "evidence_value_coverage",
        "unit_accuracy",
        "time_accuracy",
        "novel_number_rate",
        "rule_covered_unsupported_rate",
    ),
    "directional": (
        "direction_consistency",
        "direction_coverage",
        "policy_stance_consistency",
        "policy_stance_coverage",
    ),
}
HEADLINE_METRICS = tuple(
    metric for metrics in METRIC_GROUPS.values() for metric in metrics
)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {resolved}")
    return payload


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {label} {resolved}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object in {label} {resolved}:{line_number}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty: {resolved}")
    return rows


def _stream_jsonl_row_count(path: Path, *, label: str) -> int:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    row_count = 0
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {label} {resolved}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object in {label} {resolved}:{line_number}"
                )
            row_count += 1
    if row_count == 0:
        raise ValueError(f"{label} is empty: {resolved}")
    return row_count


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _validate_bound_jsonl(
    *,
    audit: Mapping[str, Any],
    name: str,
    actual_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output_artifacts = _require_mapping(
        audit.get("output_artifacts"), label="audit.output_artifacts"
    )
    record = _require_mapping(
        output_artifacts.get(name), label=f"audit.output_artifacts.{name}"
    )
    recorded_path = Path(str(record.get("path") or "")).expanduser().resolve()
    resolved = actual_path.expanduser().resolve()
    if recorded_path != resolved:
        raise ValueError(
            f"{name} path is not the artifact sealed by the audit: "
            f"{resolved} != {recorded_path}"
        )
    observed_sha256 = sha256_file(resolved)
    if record.get("sha256") != observed_sha256:
        raise ValueError(f"{name} hash differs from the sealed audit")
    if _require_int(record.get("row_count"), label=f"{name}.row_count") != len(rows):
        raise ValueError(f"{name} row count differs from the sealed audit")
    audit_counts = _require_mapping(audit.get("row_counts"), label="audit.row_counts")
    if _require_int(audit_counts.get(name), label=f"audit.row_counts.{name}") != len(
        rows
    ):
        raise ValueError(f"{name} row count differs from audit.row_counts")
    return {
        "path": str(resolved),
        "sha256": observed_sha256,
        "row_count": len(rows),
    }


def _validate_bound_jsonl_count(
    *,
    audit: Mapping[str, Any],
    name: str,
    actual_path: Path,
    row_count: int,
) -> dict[str, Any]:
    output_artifacts = _require_mapping(
        audit.get("output_artifacts"), label="audit.output_artifacts"
    )
    record = _require_mapping(
        output_artifacts.get(name), label=f"audit.output_artifacts.{name}"
    )
    recorded_path = Path(str(record.get("path") or "")).expanduser().resolve()
    resolved = actual_path.expanduser().resolve()
    if recorded_path != resolved:
        raise ValueError(
            f"{name} path is not the artifact sealed by the audit: "
            f"{resolved} != {recorded_path}"
        )
    observed_sha256 = sha256_file(resolved)
    if record.get("sha256") != observed_sha256:
        raise ValueError(f"{name} hash differs from the sealed audit")
    if _require_int(record.get("row_count"), label=f"{name}.row_count") != row_count:
        raise ValueError(f"{name} row count differs from the sealed audit")
    audit_counts = _require_mapping(audit.get("row_counts"), label="audit.row_counts")
    if (
        _require_int(audit_counts.get(name), label=f"audit.row_counts.{name}")
        != row_count
    ):
        raise ValueError(f"{name} row count differs from audit.row_counts")
    return {
        "path": str(resolved),
        "sha256": observed_sha256,
        "row_count": row_count,
    }


def _resolve_receipt_path(value: object, *, receipt_path: Path, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label}.path is empty")
    raw = Path(text).expanduser()
    resolved = (raw if raw.is_absolute() else receipt_path.parent / raw).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {resolved}")
    return resolved


def _validate_receipt_file(
    record: object,
    *,
    receipt_path: Path,
    label: str,
    sealed: bool,
    expected_schema: str | None = None,
) -> dict[str, Any]:
    binding = _require_mapping(record, label=label)
    path = _resolve_receipt_path(
        binding.get("path"), receipt_path=receipt_path, label=label
    )
    observed_sha256 = sha256_file(path)
    if binding.get("sha256") != observed_sha256:
        raise ValueError(f"{label} hash differs from input_validation")
    payload: dict[str, Any] | None = None
    payload_sha256: str | None = None
    if sealed or expected_schema is not None:
        payload = _read_json(path, label=label)
    if sealed:
        assert payload is not None
        payload_sha256 = validate_manifest_integrity(payload)
        if binding.get("payload_sha256") != payload_sha256:
            raise ValueError(f"{label} payload hash differs from input_validation")
    if expected_schema is not None:
        assert payload is not None
        if payload.get("schema_version") != expected_schema:
            raise ValueError(f"{label} has an unsupported schema")
    result = {"path": str(path), "sha256": observed_sha256}
    if payload_sha256 is not None:
        result["payload_sha256"] = payload_sha256
    return result


def _validate_receipt_jsonl(
    record: object,
    *,
    receipt_path: Path,
    label: str,
    expected_rows: int,
    expected_artifact_id: str | None = None,
) -> dict[str, Any]:
    binding = _require_mapping(record, label=label)
    path = _resolve_receipt_path(
        binding.get("path"), receipt_path=receipt_path, label=label
    )
    rows = _read_jsonl(path, label=label)
    if len(rows) != expected_rows:
        raise ValueError(f"{label} must contain exactly {expected_rows} rows")
    if binding.get("sha256") != sha256_file(path):
        raise ValueError(f"{label} hash differs from input_validation")
    if expected_artifact_id is not None and any(
        row.get("artifact_id") != expected_artifact_id for row in rows
    ):
        raise ValueError(f"{label} contains the wrong artifact ID")
    return {"path": str(path), "sha256": sha256_file(path), "row_count": len(rows)}


def _validate_input_validation(
    *,
    receipt_path: Path,
    audit_path: Path,
    audit: Mapping[str, Any],
    policy_name: str,
) -> dict[str, Any]:
    receipt = _read_json(receipt_path, label=f"{policy_name} input_validation")
    receipt_payload_sha256 = validate_manifest_integrity(receipt)
    if (
        receipt.get("schema_version") != WRAPPER_SCHEMA_VERSION
        or receipt.get("status") != "validated"
        or receipt.get("complete") is not True
        or receipt.get("artifact_ids") != list(EXPECTED_ARTIFACT_IDS)
    ):
        raise ValueError(
            f"{policy_name} input_validation is not a complete validated wrapper receipt"
        )
    expected_policy = POLICY_SPECS[policy_name]["policy_id"]
    if (
        receipt.get("scoring_policy") != expected_policy
        or receipt.get("bootstrap_samples") != EXPECTED_BOOTSTRAP_SAMPLES
        or receipt.get("bootstrap_seed") != EXPECTED_BOOTSTRAP_SEED
    ):
        raise ValueError(f"{policy_name} input_validation execution contract changed")

    core = _require_mapping(
        receipt.get("core_evaluation"),
        label=f"{policy_name} input_validation.core_evaluation",
    )
    bound_audit_path = _resolve_receipt_path(
        core.get("audit_path"),
        receipt_path=receipt_path,
        label=f"{policy_name} core_evaluation.audit",
    )
    audit_integrity = _require_mapping(
        audit.get("integrity"), label=f"{policy_name} audit.integrity"
    )
    if (
        bound_audit_path != audit_path.resolve()
        or core.get("audit_sha256") != sha256_file(audit_path)
        or core.get("audit_payload_sha256") != audit_integrity.get("payload_sha256")
        or core.get("run_spec_sha256") != audit.get("run_spec_sha256")
        or core.get("status") != audit.get("status")
        or core.get("status") != "validated"
    ):
        raise ValueError(f"{policy_name} input_validation core-audit binding changed")

    frozen = _require_mapping(
        receipt.get("frozen_test"), label=f"{policy_name} frozen_test"
    )
    if (
        frozen.get("sample_count") != EXPECTED_ROWS_PER_ARTIFACT
        or frozen.get("meeting_count") != 11
        or frozen.get("prospective_meeting_count") != 9
        or frozen.get("section_count") != 3
        or frozen.get("reference_free") is not True
    ):
        raise ValueError(f"{policy_name} frozen-test contract changed")
    verified_frozen = {
        "prompts": _validate_receipt_jsonl(
            frozen.get("prompts"),
            receipt_path=receipt_path,
            label=f"{policy_name} frozen prompts",
            expected_rows=EXPECTED_ROWS_PER_ARTIFACT,
        ),
        "references": _validate_receipt_jsonl(
            frozen.get("references"),
            receipt_path=receipt_path,
            label=f"{policy_name} frozen references",
            expected_rows=EXPECTED_ROWS_PER_ARTIFACT,
        ),
        "test_manifest": _validate_receipt_file(
            frozen.get("test_manifest"),
            receipt_path=receipt_path,
            label=f"{policy_name} test manifest",
            sealed=True,
            expected_schema="checkpoint-eval-test-manifest-v1",
        ),
        "evaluation_config": _validate_receipt_file(
            frozen.get("evaluation_config"),
            receipt_path=receipt_path,
            label=f"{policy_name} evaluation config",
            sealed=False,
            expected_schema="checkpoint-generation-eval-config-v1",
        ),
    }
    verified_dependencies = {
        "checkpoint_manifest": _validate_receipt_file(
            receipt.get("checkpoint_manifest"),
            receipt_path=receipt_path,
            label=f"{policy_name} checkpoint manifest",
            sealed=True,
            expected_schema="retrain-v2-checkpoint-eval-manifest-v1",
        ),
        "lineage_manifest": _validate_receipt_file(
            receipt.get("lineage_manifest"),
            receipt_path=receipt_path,
            label=f"{policy_name} lineage manifest",
            sealed=True,
            expected_schema="retrain-v2-eval-lineage-v1",
        ),
        "semantic_manifest": _validate_receipt_file(
            receipt.get("semantic_manifest"),
            receipt_path=receipt_path,
            label=f"{policy_name} semantic manifest",
            sealed=True,
            expected_schema="checkpoint-eval-semantic-model-manifest-v1",
        ),
        "exact_merge_evidence": _validate_receipt_file(
            receipt.get("exact_merge_evidence"),
            receipt_path=receipt_path,
            label=f"{policy_name} exact-merge evidence",
            sealed=True,
            expected_schema="lora-merge-lineage-evidence-v1",
        ),
    }

    generations = _require_mapping(
        receipt.get("generations"), label=f"{policy_name} generations"
    )
    if set(generations) != set(EXPECTED_ARTIFACT_IDS):
        raise ValueError(f"{policy_name} generation receipt inventory changed")
    verified_generations: dict[str, Any] = {}
    for artifact_id in EXPECTED_ARTIFACT_IDS:
        generation = _require_mapping(
            generations.get(artifact_id),
            label=f"{policy_name} generations.{artifact_id}",
        )
        if generation.get("sample_count") != EXPECTED_ROWS_PER_ARTIFACT:
            raise ValueError(f"{policy_name}/{artifact_id} generation count changed")
        generation_file = _validate_receipt_jsonl(
            generation.get("generation"),
            receipt_path=receipt_path,
            label=f"{policy_name}/{artifact_id} generations",
            expected_rows=EXPECTED_ROWS_PER_ARTIFACT,
            expected_artifact_id=artifact_id,
        )
        manifest_file = _validate_receipt_file(
            generation.get("manifest"),
            receipt_path=receipt_path,
            label=f"{policy_name}/{artifact_id} generation manifest",
            sealed=True,
            expected_schema=GENERATION_MANIFEST_SCHEMA_VERSION,
        )
        progress_validation = validate_generation_progress_binding(
            generation_manifest_file=manifest_file["path"],
            generation_output_file=generation_file["path"],
            checkpoint_manifest_file=verified_dependencies["checkpoint_manifest"][
                "path"
            ],
            prompts_file=verified_frozen["prompts"]["path"],
            config_file=verified_frozen["evaluation_config"]["path"],
        )
        if generation.get("progress") != progress_validation:
            raise ValueError(
                f"{policy_name}/{artifact_id} progress validation changed"
            )
        verified_generations[artifact_id] = {
            "generation": generation_file,
            "manifest": manifest_file,
            "model_sha256": generation.get("model_sha256"),
            "tokenizer_sha256": generation.get("tokenizer_sha256"),
            "sample_count": generation.get("sample_count"),
            "progress": progress_validation,
        }

    core_evaluator = _require_mapping(
        receipt.get("core_evaluator"), label=f"{policy_name} core_evaluator"
    )
    if (
        core_evaluator.get("module") != "jobs.eval.eval_checkpoint_generation"
        or core_evaluator.get("expected_artifact_count") != 2
        or core_evaluator.get("paired_unit") != "sample_id"
        or core_evaluator.get("cluster_unit") != "meeting_id"
        or core_evaluator.get("multiple_testing") != "Holm"
    ):
        raise ValueError(f"{policy_name} core-evaluator contract changed")

    source_binding = {
        "artifact_ids": list(EXPECTED_ARTIFACT_IDS),
        "frozen_test_metadata": {
            key: frozen.get(key)
            for key in (
                "evaluation_id",
                "sample_count",
                "meeting_count",
                "prospective_meeting_count",
                "section_count",
                "reference_free",
            )
        },
        "frozen_test": verified_frozen,
        **verified_dependencies,
        "semantic_models": _require_mapping(
            _require_mapping(
                receipt.get("semantic_manifest"),
                label=f"{policy_name} semantic_manifest",
            ).get("models"),
            label=f"{policy_name} semantic_manifest.models",
        ),
        "exact_merge_validation": {
            key: _require_mapping(
                receipt.get("exact_merge_evidence"),
                label=f"{policy_name} exact_merge_evidence",
            ).get(key)
            for key in (
                "subject_artifact_id",
                "conclusion",
                "algorithm_version",
                "sources",
                "tensor_verification",
            )
        },
        "generations": verified_generations,
        "core_evaluator": dict(core_evaluator),
    }
    return {
        "receipt": {
            "path": str(receipt_path.resolve()),
            "sha256": sha256_file(receipt_path),
            "payload_sha256": receipt_payload_sha256,
        },
        "source_binding": source_binding,
    }


def _validate_matrix(
    *, audit: Mapping[str, Any], row_scores: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected_ids = list(EXPECTED_ARTIFACT_IDS)
    if audit.get("artifact_count") != 2 or audit.get("artifact_ids") != expected_ids:
        raise ValueError(
            "Evaluation audit must contain exactly the expected chk0/chk1 artifact IDs"
        )
    alignment = _require_mapping(audit.get("alignment"), label="audit.alignment")
    generation_counts = _require_mapping(
        alignment.get("generation_sample_count_by_artifact"),
        label="audit.alignment.generation_sample_count_by_artifact",
    )
    expected_counts = {
        artifact_id: EXPECTED_ROWS_PER_ARTIFACT for artifact_id in expected_ids
    }
    if (
        alignment.get("complete") is not True
        or alignment.get("artifact_ids") != expected_ids
        or alignment.get("sample_universe_count") != EXPECTED_ROWS_PER_ARTIFACT
        or alignment.get("reference_sample_count") != EXPECTED_ROWS_PER_ARTIFACT
        or alignment.get("evidence_sample_count") != EXPECTED_ROWS_PER_ARTIFACT
        or dict(generation_counts) != expected_counts
    ):
        raise ValueError("Audit does not attest the complete 2 x 33 evaluation matrix")
    for key in (
        "missing_evidence_sample_ids",
        "missing_reference_sample_ids",
        "reference_without_evidence_sample_ids",
        "evidence_without_reference_sample_ids",
    ):
        if alignment.get(key) != []:
            raise ValueError(f"Audit alignment is incomplete: {key}")
    missing_by_artifact = _require_mapping(
        alignment.get("missing_generation_sample_ids_by_artifact"),
        label="audit.alignment.missing_generation_sample_ids_by_artifact",
    )
    if dict(missing_by_artifact) != {artifact_id: [] for artifact_id in expected_ids}:
        raise ValueError("Audit alignment contains missing generation rows")

    if len(row_scores) != 2 * EXPECTED_ROWS_PER_ARTIFACT:
        raise ValueError("Actual row_scores is not a 2 x 33 matrix")
    sample_ids_by_artifact: dict[str, set[str]] = {
        artifact_id: set() for artifact_id in expected_ids
    }
    for row_number, row in enumerate(row_scores, start=1):
        if row.get("schema_version") != EVALUATOR_SCHEMA_VERSION:
            raise ValueError(f"row_scores row {row_number} has an unsupported schema")
        artifact_id = str(row.get("artifact_id") or "")
        sample_id = str(row.get("sample_id") or "").strip()
        if artifact_id not in sample_ids_by_artifact or not sample_id:
            raise ValueError(f"row_scores row {row_number} has an invalid matrix key")
        if sample_id in sample_ids_by_artifact[artifact_id]:
            raise ValueError(
                f"row_scores contains a duplicate key: {artifact_id}/{sample_id}"
            )
        sample_ids_by_artifact[artifact_id].add(sample_id)
    if (
        any(
            len(sample_ids) != EXPECTED_ROWS_PER_ARTIFACT
            for sample_ids in sample_ids_by_artifact.values()
        )
        or len({frozenset(value) for value in sample_ids_by_artifact.values()}) != 1
    ):
        raise ValueError(
            "Actual row_scores does not contain the same 33 samples per artifact"
        )
    return {
        "artifact_count": 2,
        "rows_per_artifact": EXPECTED_ROWS_PER_ARTIFACT,
        "total_rows": 2 * EXPECTED_ROWS_PER_ARTIFACT,
        "paired_sample_keys_identical": True,
        "sample_ids_sha256": manifest_payload_sha256(
            {"sample_ids": sorted(sample_ids_by_artifact[BASELINE_ARTIFACT_ID])}
        ),
    }


def _validate_policy_bundle(directory: Path, *, policy_name: str) -> dict[str, Any]:
    root = directory.expanduser().resolve()
    audit_path = root / "audit.json"
    receipt_path = root / "input_validation.json"
    summary_path = root / "summary.jsonl"
    contrasts_path = root / "contrasts.jsonl"
    audit = _read_json(audit_path, label=f"{policy_name} audit")
    audit_payload_sha256 = validate_manifest_integrity(audit)
    if (
        audit.get("schema_version") != EVALUATOR_SCHEMA_VERSION
        or audit.get("status") != "validated"
        or audit.get("immutable") is not True
    ):
        raise ValueError(
            f"{policy_name} audit is not an immutable validated evaluation"
        )

    expected_policy = POLICY_SPECS[policy_name]["policy_id"]
    scoring_policy = _require_mapping(
        audit.get("scoring_policy"), label=f"{policy_name} audit.scoring_policy"
    )
    if scoring_policy.get("policy_id") != expected_policy:
        raise ValueError(
            f"{policy_name} bundle uses {scoring_policy.get('policy_id')!r}, "
            f"expected {expected_policy!r}"
        )
    run_spec = _require_mapping(audit.get("run_spec"), label=f"{policy_name} run_spec")
    if audit.get("run_spec_sha256") != manifest_payload_sha256(run_spec):
        raise ValueError(f"{policy_name} audit has an invalid run_spec_sha256")
    if run_spec.get("scoring_policy") != scoring_policy:
        raise ValueError(
            f"{policy_name} top-level and run-spec scoring policies differ"
        )
    if (
        run_spec.get("artifact_ids") != list(EXPECTED_ARTIFACT_IDS)
        or run_spec.get("required_artifact_ids") != list(EXPECTED_ARTIFACT_IDS)
        or run_spec.get("expected_artifact_count") != 2
        or run_spec.get("bootstrap_samples") != EXPECTED_BOOTSTRAP_SAMPLES
        or run_spec.get("bootstrap_seed") != EXPECTED_BOOTSTRAP_SEED
    ):
        raise ValueError(
            f"{policy_name} run spec has the wrong artifact or bootstrap contract"
        )
    if run_spec.get("evaluation_subsets") != {
        "all_11_meetings": None,
        "prospective_only_9_meetings": [
            "2025-06-18",
            "2025-07-30",
            "2025-09-17",
            "2025-10-29",
            "2025-12-10",
            "2026-01-28",
            "2026-03-18",
            "2026-04-29",
            "2026-06-17",
        ],
    }:
        raise ValueError(f"{policy_name} run spec does not contain the frozen subsets")

    input_validation = _validate_input_validation(
        receipt_path=receipt_path,
        audit_path=audit_path,
        audit=audit,
        policy_name=policy_name,
    )

    output_artifacts = _require_mapping(
        audit.get("output_artifacts"), label="audit.output_artifacts"
    )
    row_scores_path = (
        Path(
            str(
                _require_mapping(
                    output_artifacts.get("row_scores"),
                    label="audit.output_artifacts.row_scores",
                ).get("path")
                or ""
            )
        )
        .expanduser()
        .resolve()
    )
    claim_checks_path = (
        Path(
            str(
                _require_mapping(
                    output_artifacts.get("claim_checks"),
                    label="audit.output_artifacts.claim_checks",
                ).get("path")
                or ""
            )
        )
        .expanduser()
        .resolve()
    )
    row_scores = _read_jsonl(row_scores_path, label=f"{policy_name} row_scores")
    claim_checks_count = _stream_jsonl_row_count(
        claim_checks_path, label=f"{policy_name} claim_checks"
    )
    summary = _read_jsonl(summary_path, label=f"{policy_name} summary")
    contrasts = _read_jsonl(contrasts_path, label=f"{policy_name} contrasts")
    source_artifacts = {
        "audit": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "payload_sha256": audit_payload_sha256,
        },
        "input_validation": input_validation["receipt"],
        "row_scores": _validate_bound_jsonl(
            audit=audit,
            name="row_scores",
            actual_path=row_scores_path,
            rows=row_scores,
        ),
        "claim_checks": _validate_bound_jsonl_count(
            audit=audit,
            name="claim_checks",
            actual_path=claim_checks_path,
            row_count=claim_checks_count,
        ),
        "summary": _validate_bound_jsonl(
            audit=audit, name="summary", actual_path=summary_path, rows=summary
        ),
        "contrasts": _validate_bound_jsonl(
            audit=audit,
            name="contrasts",
            actual_path=contrasts_path,
            rows=contrasts,
        ),
    }
    matrix = _validate_matrix(audit=audit, row_scores=row_scores)

    summary_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row_number, row in enumerate(summary, start=1):
        if row.get("schema_version") != EVALUATOR_SCHEMA_VERSION:
            raise ValueError(f"{policy_name} summary row {row_number} has wrong schema")
        artifact_id = str(row.get("artifact_id") or "")
        subset = str(row.get("evaluation_subset") or "")
        metric = str(row.get("metric") or "")
        if artifact_id not in EXPECTED_ARTIFACT_IDS:
            raise ValueError(f"{policy_name} summary contains an unknown artifact ID")
        key = (artifact_id, subset, metric)
        if key in summary_index:
            raise ValueError(f"{policy_name} summary contains duplicate row {key}")
        summary_index[key] = row

    contrast_index: dict[tuple[str, str], dict[str, Any]] = {}
    for row_number, row in enumerate(contrasts, start=1):
        if row.get("schema_version") != EVALUATOR_SCHEMA_VERSION:
            raise ValueError(
                f"{policy_name} contrast row {row_number} has wrong schema"
            )
        if (
            row.get("baseline_artifact_id") != BASELINE_ARTIFACT_ID
            or row.get("candidate_artifact_id") != CANDIDATE_ARTIFACT_ID
        ):
            raise ValueError(f"{policy_name} contrasts contain the wrong artifact edge")
        key = (
            str(row.get("evaluation_subset") or ""),
            str(row.get("metric") or ""),
        )
        if key in contrast_index:
            raise ValueError(f"{policy_name} contrasts contain duplicate row {key}")
        contrast_index[key] = row

    return {
        "audit": audit,
        "run_spec": dict(run_spec),
        "summary_index": summary_index,
        "contrast_index": contrast_index,
        "source_artifacts": source_artifacts,
        "input_source_binding": input_validation["source_binding"],
        "matrix": matrix,
    }


def _common_run_binding(run_spec: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(run_spec)
    normalized.pop("scoring_policy", None)
    normalized.pop("invalid_output_policy_version", None)
    return normalized


def _artifact_estimate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mean": row.get("mean"),
        "median": row.get("median"),
        "ci_95": {"lower": row.get("ci_lower"), "upper": row.get("ci_upper")},
        "n_rows_total": row.get("n_rows_total"),
        "n_rows_eligible": row.get("n_rows_eligible"),
        "n_rows_ineligible": row.get("n_rows_ineligible"),
        "n_meetings": row.get("n_meetings"),
        "n_meetings_total": row.get("n_meetings_total"),
    }


def _metric_summary(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    contrast: Mapping[str, Any],
) -> dict[str, Any]:
    for label, row in (
        ("baseline summary", baseline),
        ("candidate summary", candidate),
        ("contrast", contrast),
    ):
        if (
            row.get("confidence") != 0.95
            or row.get("bootstrap_unit") != "meeting_id"
            or row.get("bootstrap_samples") != EXPECTED_BOOTSTRAP_SAMPLES
            or row.get("bootstrap_seed") != EXPECTED_BOOTSTRAP_SEED
        ):
            raise ValueError(f"Unexpected inference metadata in {label}")
    for label, row in (("baseline", baseline), ("candidate", candidate)):
        n_total = _require_int(row.get("n_rows_total"), label=f"{label}.n_rows_total")
        n_eligible = _require_int(
            row.get("n_rows_eligible"), label=f"{label}.n_rows_eligible"
        )
        n_ineligible = _require_int(
            row.get("n_rows_ineligible"), label=f"{label}.n_rows_ineligible"
        )
        if n_eligible + n_ineligible != n_total:
            raise ValueError(f"Inconsistent eligible-row counts for {label}")
    n_common = _require_int(
        contrast.get("n_common_samples_total"), label="contrast.n_common_samples_total"
    )
    n_pairs_eligible = _require_int(
        contrast.get("n_pairs_eligible"), label="contrast.n_pairs_eligible"
    )
    n_pairs_ineligible = _require_int(
        contrast.get("n_pairs_ineligible"), label="contrast.n_pairs_ineligible"
    )
    if n_pairs_eligible + n_pairs_ineligible != n_common:
        raise ValueError("Inconsistent eligible-pair counts in contrast")
    directions = {
        baseline.get("metric_direction"),
        candidate.get("metric_direction"),
        contrast.get("metric_direction"),
    }
    if len(directions) != 1 or None in directions:
        raise ValueError(f"Metric direction mismatch for {baseline.get('metric')}")
    return {
        "metric_direction": directions.pop(),
        "headline": {
            "paired_baseline_mean": contrast.get("mean_baseline"),
            "paired_candidate_mean": contrast.get("mean_candidate"),
            "candidate_minus_baseline": contrast.get("paired_mean_difference"),
            "candidate_minus_baseline_ci_95": {
                "lower": contrast.get("ci_lower"),
                "upper": contrast.get("ci_upper"),
            },
        },
        "marginal_estimates": {
            BASELINE_ARTIFACT_ID: _artifact_estimate(baseline),
            CANDIDATE_ARTIFACT_ID: _artifact_estimate(candidate),
        },
        "paired_difference": {
            "definition": "candidate_minus_baseline_on_common_eligible_samples",
            "estimate": contrast.get("paired_mean_difference"),
            "ci_95": {
                "lower": contrast.get("ci_lower"),
                "upper": contrast.get("ci_upper"),
            },
            "benefit_oriented_estimate": contrast.get("benefit_difference"),
            "benefit_oriented_ci_95": {
                "lower": contrast.get("benefit_ci_lower"),
                "upper": contrast.get("benefit_ci_upper"),
            },
            "n_common_samples_total": contrast.get("n_common_samples_total"),
            "n_pairs_eligible": contrast.get("n_pairs_eligible"),
            "n_pairs_ineligible": contrast.get("n_pairs_ineligible"),
            "n_meetings": contrast.get("n_meetings"),
            "n_meetings_total": contrast.get("n_meetings_total"),
            "inference_status": contrast.get("inference_status"),
            "p_value": contrast.get("p_value"),
            "p_value_holm": contrast.get("p_value_holm"),
            "effect_size_paired_dz": contrast.get("effect_size_paired_dz"),
        },
    }


def _summarize_policy(bundle: Mapping[str, Any]) -> dict[str, Any]:
    summary_index = bundle["summary_index"]
    contrast_index = bundle["contrast_index"]
    subsets: dict[str, Any] = {}
    for subset, expected_rows in EXPECTED_SUBSET_ROWS.items():
        valid_rows: dict[str, Mapping[str, Any]] = {}
        for artifact_id in EXPECTED_ARTIFACT_IDS:
            key = (artifact_id, subset, "valid_output")
            if key not in summary_index:
                raise ValueError(f"Missing valid_output summary row: {key}")
            row = summary_index[key]
            if row.get("n_rows_total") != expected_rows:
                raise ValueError(
                    f"Unexpected {subset} row population for {artifact_id}"
                )
            n_valid = _require_int(
                row.get("n_valid_output"),
                label=f"{subset}/{artifact_id}.n_valid_output",
            )
            n_invalid = _require_int(
                row.get("n_invalid_output"),
                label=f"{subset}/{artifact_id}.n_invalid_output",
            )
            n_missing = _require_int(
                row.get("n_missing_input"),
                label=f"{subset}/{artifact_id}.n_missing_input",
            )
            if n_valid + n_invalid != expected_rows or n_missing != 0:
                raise ValueError(
                    f"Inconsistent invalid-output counts for {subset}/{artifact_id}"
                )
            valid_rows[artifact_id] = row

        groups: dict[str, Any] = {}
        for group_name, metrics in METRIC_GROUPS.items():
            group: dict[str, Any] = {}
            for metric in metrics:
                baseline_key = (BASELINE_ARTIFACT_ID, subset, metric)
                candidate_key = (CANDIDATE_ARTIFACT_ID, subset, metric)
                contrast_key = (subset, metric)
                if (
                    baseline_key not in summary_index
                    or candidate_key not in summary_index
                    or contrast_key not in contrast_index
                ):
                    raise ValueError(f"Missing required {subset}/{metric} result")
                baseline = summary_index[baseline_key]
                candidate = summary_index[candidate_key]
                contrast = contrast_index[contrast_key]
                for artifact_id, row in (
                    (BASELINE_ARTIFACT_ID, baseline),
                    (CANDIDATE_ARTIFACT_ID, candidate),
                ):
                    if row.get("n_rows_total") != expected_rows:
                        raise ValueError(
                            f"Unexpected {subset}/{metric} population for {artifact_id}"
                        )
                if contrast.get("n_common_samples_total") != expected_rows:
                    raise ValueError(
                        f"Unexpected paired population for {subset}/{metric}"
                    )
                group[metric] = _metric_summary(
                    baseline=baseline, candidate=candidate, contrast=contrast
                )
            groups[group_name] = group

        subsets[subset] = {
            "population": {
                "rows": expected_rows,
                "meetings": 11 if subset == "all_11_meetings" else 9,
                "bootstrap_unit": "meeting_id",
            },
            "invalid_counts": {
                artifact_id: {
                    "n_rows_total": row.get("n_rows_total"),
                    "n_valid_output": row.get("n_valid_output"),
                    "n_invalid_output": row.get("n_invalid_output"),
                    "n_missing_input": row.get("n_missing_input"),
                    "valid_output_rate": row.get("mean"),
                }
                for artifact_id, row in valid_rows.items()
            },
            "metric_groups": groups,
        }
    return subsets


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to replace immutable summary: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def summarize_chk0_chk1_cp200_generation_eval(
    *,
    strict_dir: Path,
    length_tolerant_dir: Path,
    output_file: Path,
) -> dict[str, Any]:
    """Validate the two evaluation policies and write the versioned summary."""

    bundles = {
        "strict": _validate_policy_bundle(strict_dir, policy_name="strict"),
        "length_tolerant": _validate_policy_bundle(
            length_tolerant_dir, policy_name="length_tolerant"
        ),
    }
    strict_binding = _common_run_binding(bundles["strict"]["run_spec"])
    tolerant_binding = _common_run_binding(bundles["length_tolerant"]["run_spec"])
    if strict_binding != tolerant_binding:
        raise ValueError(
            "Strict and length-tolerant bundles are not bound to the same run inputs"
        )
    if bundles["strict"]["matrix"] != bundles["length_tolerant"]["matrix"]:
        raise ValueError("Strict and length-tolerant matrix attestations differ")
    strict_sources = bundles["strict"]["input_source_binding"]
    tolerant_sources = bundles["length_tolerant"]["input_source_binding"]
    if strict_sources != tolerant_sources:
        raise ValueError(
            "Strict and length-tolerant input_validation receipts do not bind "
            "the same frozen inputs"
        )

    result = seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "validated",
            "comparison": {
                "baseline_artifact_id": BASELINE_ARTIFACT_ID,
                "candidate_artifact_id": CANDIDATE_ARTIFACT_ID,
                "difference_definition": "candidate_minus_baseline",
                "matrix": bundles["strict"]["matrix"],
                "common_run_binding_sha256": manifest_payload_sha256(strict_binding),
                "common_input_validation_sources_sha256": manifest_payload_sha256(
                    strict_sources
                ),
            },
            "policies": {
                policy_name: {
                    "policy_id": POLICY_SPECS[policy_name]["policy_id"],
                    "analysis_role": POLICY_SPECS[policy_name]["analysis_role"],
                    "inference": {
                        "bootstrap_unit": "meeting_id",
                        "bootstrap_samples": EXPECTED_BOOTSTRAP_SAMPLES,
                        "bootstrap_seed": EXPECTED_BOOTSTRAP_SEED,
                        "confidence": 0.95,
                    },
                    "source_artifacts": bundle["source_artifacts"],
                    "subsets": _summarize_policy(bundle),
                }
                for policy_name, bundle in bundles.items()
            },
            "conclusion_boundary": {
                "benchmark": (
                    "Frozen Chapter 2 raw D-1 point-in-time evidence to official "
                    "Minutes stress benchmark, 33 section rows across 11 meetings."
                ),
                "candidate_product_task": (
                    "chk1 is trained for point-in-time evidence to atomic FOMC analysis, "
                    "so similarity to official Minutes is an out-of-task stress signal."
                ),
                "primary_analysis": (
                    "strict-final-answer-v2; invalid outputs remain visible in the "
                    "declared 2 x 33 matrix."
                ),
                "robustness_analysis": (
                    "length-tolerant-open-tags-v1 is diagnostic only and cannot replace "
                    "the strict result."
                ),
                "inference_unit": (
                    "Paired differences are formed on common eligible sample rows, then "
                    "aggregated and bootstrapped by meeting_id; 33 rows are not 33 "
                    "independent meeting clusters."
                ),
                "metric_limits": [
                    "BERTScore, MPNet cosine, and ROUGE-L measure reference similarity, not factual correctness.",
                    "rule_covered_unsupported_rate covers only declared numeric, direction, policy-stance, and custom rules; it is not exhaustive factuality or NLI.",
                    "Direction and policy-stance consistency must be interpreted together with their coverage and eligible-pair counts.",
                    "Length ratio is descriptive; its candidate-minus-baseline sign is not inherently beneficial or harmful.",
                ],
                "not_supported": [
                    "causal attribution of metric changes to training",
                    "automatic promotion of chk1 or readiness for chk2",
                    "generalization beyond the frozen 11-meeting benchmark",
                    "claims based on length-tolerant results while omitting strict invalid counts",
                ],
                "human_or_llm_judge_used_by_this_summary": False,
                "scoring_or_generation_run_by_this_summary": False,
            },
        }
    )
    _write_immutable_json(output_file, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-dir",
        type=Path,
        required=True,
        help="Directory containing sealed strict audit.json and bound JSONL outputs.",
    )
    parser.add_argument(
        "--length-tolerant-dir",
        type=Path,
        required=True,
        help="Directory containing sealed length-tolerant audit and JSONL outputs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = summarize_chk0_chk1_cp200_generation_eval(
        strict_dir=args.strict_dir,
        length_tolerant_dir=args.length_tolerant_dir,
        output_file=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "schema_version": result["schema_version"],
                "output": str(args.output.expanduser().resolve()),
                "sha256": sha256_file(args.output),
                "payload_sha256": result["integrity"]["payload_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
