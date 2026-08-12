"""Validate and summarize sealed chk0/chk1/chk3-cp250 generation scores.

This is a read-only post-processing command.  It performs neither generation
nor scoring.  The strict score directory is the primary official-Minutes
similarity stress benchmark; the length-tolerant directory is reported only as
raw-completion diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    manifest_payload_sha256,
    seal_manifest,
    validate_manifest_integrity,
)


SCHEMA_VERSION = "chk0-chk1-chk3-cp250-generation-evaluation-summary-v1"
EVALUATOR_SCHEMA_VERSION = "checkpoint-generation-evaluation-v1"
WRAPPER_SCHEMA_VERSION = "retrain-v2-chk0-chk1-chk3-generation-evaluation-v1"

CHK0_ARTIFACT_ID = "eval-chk0-base"
CHK1_ARTIFACT_ID = "eval-chk1-clean-v2-lr1e6-cp200"
CHK3_ARTIFACT_ID = "eval-chk3-direct-chk1-cp200-sft-cp250"
EXPECTED_ARTIFACT_IDS = (CHK0_ARTIFACT_ID, CHK1_ARTIFACT_ID, CHK3_ARTIFACT_ID)
ADJACENT_CONTRASTS = (
    (CHK0_ARTIFACT_ID, CHK1_ARTIFACT_ID),
    (CHK1_ARTIFACT_ID, CHK3_ARTIFACT_ID),
)
EXPECTED_ROWS_PER_ARTIFACT = 33
EXPECTED_BOOTSTRAP_SAMPLES = 10_000
EXPECTED_BOOTSTRAP_SEED = 20_260_729
EXPECTED_SUBSET_ROWS = {
    "all_11_meetings": 33,
    "prospective_only_9_meetings": 27,
}
POLICY_SPECS = {
    "strict": {
        "policy_id": "strict-final-answer-v2",
        "analysis_role": "primary_stress_benchmark",
        "candidate_label": "strict parsed final answer",
    },
    "length_tolerant": {
        "policy_id": "length-tolerant-open-tags-v1",
        "analysis_role": "raw_completion_diagnostics_only",
        "candidate_label": "raw-completion extracted diagnostic text",
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
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {resolved}")
    return value


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


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _resolve_path(value: object, *, owner: Path, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label}.path is empty")
    raw = Path(text).expanduser()
    result = (raw if raw.is_absolute() else owner.parent / raw).resolve()
    if not result.exists():
        raise FileNotFoundError(f"{label} does not exist: {result}")
    return result


def _validate_file_binding(
    value: object,
    *,
    owner: Path,
    label: str,
    sealed: bool = False,
    expected_rows: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    record = _mapping(value, label=label)
    path = _resolve_path(record.get("path"), owner=owner, label=label)
    if not path.is_file():
        raise ValueError(f"{label} must bind a file: {path}")
    expected_sha = record.get("sha256", record.get("file_sha256"))
    observed_sha = sha256_file(path)
    if expected_sha != observed_sha:
        raise ValueError(f"{label} hash differs from input_validation")
    result: dict[str, Any] = {"path": str(path), "sha256": observed_sha}
    rows: list[dict[str, Any]] | None = None
    if expected_rows is not None:
        rows = _read_jsonl(path, label=label)
        if len(rows) != expected_rows:
            raise ValueError(f"{label} must contain exactly {expected_rows} rows")
        result["row_count"] = len(rows)
        declared_rows = record.get("row_count")
        if declared_rows is not None and declared_rows != len(rows):
            raise ValueError(f"{label} row count differs from input_validation")
    if sealed:
        payload = _read_json(path, label=label)
        payload_sha = validate_manifest_integrity(payload)
        if record.get("payload_sha256") != payload_sha:
            raise ValueError(f"{label} payload hash differs from input_validation")
        result["payload_sha256"] = payload_sha
    return result, rows


def _validate_input_validation(
    *,
    receipt_path: Path,
    audit_path: Path,
    audit: Mapping[str, Any],
    policy_name: str,
) -> dict[str, Any]:
    receipt = _read_json(receipt_path, label=f"{policy_name} input_validation")
    payload_sha = validate_manifest_integrity(receipt)
    if (
        receipt.get("schema_version") != WRAPPER_SCHEMA_VERSION
        or receipt.get("status") != "validated"
        or receipt.get("complete") is not True
        or receipt.get("artifact_ids") != list(EXPECTED_ARTIFACT_IDS)
        or receipt.get("evaluation_only") is not True
        or receipt.get("promotable_to_canonical_dag") is not False
    ):
        raise ValueError(
            f"{policy_name} input_validation is not a complete three-leg receipt"
        )
    expected_policy = POLICY_SPECS[policy_name]["policy_id"]
    if (
        receipt.get("scoring_policy") != expected_policy
        or receipt.get("bootstrap_samples") != EXPECTED_BOOTSTRAP_SAMPLES
        or receipt.get("bootstrap_seed") != EXPECTED_BOOTSTRAP_SEED
    ):
        raise ValueError(f"{policy_name} input_validation execution contract changed")

    core = _mapping(
        receipt.get("core_evaluation"),
        label=f"{policy_name} input_validation.core_evaluation",
    )
    bound_audit = _resolve_path(
        core.get("audit_path"), owner=receipt_path, label=f"{policy_name} audit"
    )
    audit_integrity = _mapping(
        audit.get("integrity"), label=f"{policy_name} audit.integrity"
    )
    if (
        bound_audit != audit_path.resolve()
        or core.get("audit_sha256") != sha256_file(audit_path)
        or core.get("audit_payload_sha256") != audit_integrity.get("payload_sha256")
        or core.get("run_spec_sha256") != audit.get("run_spec_sha256")
        or core.get("status") != "validated"
    ):
        raise ValueError(f"{policy_name} input_validation core-audit binding changed")

    core_evaluator = _mapping(
        receipt.get("core_evaluator"), label=f"{policy_name} core_evaluator"
    )
    if (
        core_evaluator.get("module") != "jobs.eval.eval_checkpoint_generation"
        or core_evaluator.get("expected_artifact_count") != 3
        or core_evaluator.get("required_artifact_ids")
        != list(EXPECTED_ARTIFACT_IDS)
        or core_evaluator.get("paired_unit") != "sample_id"
        or core_evaluator.get("cluster_unit") != "meeting_id"
        or core_evaluator.get("multiple_testing") != "Holm"
    ):
        raise ValueError(f"{policy_name} core-evaluator contract changed")

    frozen = _mapping(receipt.get("frozen_test"), label=f"{policy_name} frozen_test")
    if (
        frozen.get("sample_count") != EXPECTED_ROWS_PER_ARTIFACT
        or frozen.get("meeting_count") != 11
        or frozen.get("prospective_meeting_count") != 9
        or frozen.get("section_count") != 3
        or frozen.get("reference_free") is not True
    ):
        raise ValueError(f"{policy_name} frozen-test contract changed")
    frozen_files: dict[str, Any] = {}
    frozen_rows: dict[str, list[dict[str, Any]]] = {}
    for key in ("prompts", "references"):
        frozen_files[key], rows = _validate_file_binding(
            frozen.get(key),
            owner=receipt_path,
            label=f"{policy_name} frozen {key}",
            expected_rows=EXPECTED_ROWS_PER_ARTIFACT,
        )
        assert rows is not None
        frozen_rows[key] = rows
    frozen_files["test_manifest"], _ = _validate_file_binding(
        frozen.get("test_manifest"),
        owner=receipt_path,
        label=f"{policy_name} test_manifest",
        sealed=True,
    )
    frozen_files["evaluation_config"], _ = _validate_file_binding(
        frozen.get("evaluation_config"),
        owner=receipt_path,
        label=f"{policy_name} evaluation_config",
    )

    source_chain_files: dict[str, Any] = {}
    pair_chain = _mapping(
        receipt.get("pair_evidence_chain"), label=f"{policy_name} pair_evidence_chain"
    )
    for key in ("checkpoint_manifest", "lineage_manifest", "exact_merge_evidence"):
        source_chain_files[f"pair_evidence_chain.{key}"], _ = _validate_file_binding(
            pair_chain.get(key),
            owner=receipt_path,
            label=f"{policy_name} pair_evidence_chain.{key}",
            sealed=True,
        )
    chk3_chain = _mapping(
        receipt.get("chk3_evidence_chain"), label=f"{policy_name} chk3_evidence_chain"
    )
    for key, sealed in (
        ("manifest", True),
        ("exact_merge_evidence", True),
        ("selection_receipt", False),
        ("authorization", False),
    ):
        source_chain_files[f"chk3_evidence_chain.{key}"], _ = _validate_file_binding(
            chk3_chain.get(key),
            owner=receipt_path,
            label=f"{policy_name} chk3_evidence_chain.{key}",
            sealed=sealed,
        )
    three_leg = _mapping(
        receipt.get("three_leg_lineage"), label=f"{policy_name} three_leg_lineage"
    )
    source_chain_files["three_leg_lineage.manifest"], _ = _validate_file_binding(
        three_leg.get("manifest"),
        owner=receipt_path,
        label=f"{policy_name} three_leg_lineage.manifest",
        sealed=True,
    )
    source_chain_files["semantic_manifest"], _ = _validate_file_binding(
        receipt.get("semantic_manifest"),
        owner=receipt_path,
        label=f"{policy_name} semantic_manifest",
        sealed=True,
    )

    generations = _mapping(
        receipt.get("generations"), label=f"{policy_name} generations"
    )
    if set(generations) != set(EXPECTED_ARTIFACT_IDS):
        raise ValueError(f"{policy_name} generation inventory changed")
    verified_generations: dict[str, Any] = {}
    generation_rows: dict[str, list[dict[str, Any]]] = {}
    prompt_ids = [str(row.get("sample_id") or "").strip() for row in frozen_rows["prompts"]]
    reference_ids = [
        str(row.get("sample_id") or "").strip() for row in frozen_rows["references"]
    ]
    if (
        any(not sample_id for sample_id in prompt_ids)
        or len(set(prompt_ids)) != EXPECTED_ROWS_PER_ARTIFACT
        or reference_ids != prompt_ids
    ):
        raise ValueError(f"{policy_name} frozen prompt/reference sample matrix changed")
    for artifact_id in EXPECTED_ARTIFACT_IDS:
        generation = _mapping(
            generations[artifact_id], label=f"{policy_name}/{artifact_id} generation"
        )
        if generation.get("sample_count") != EXPECTED_ROWS_PER_ARTIFACT:
            raise ValueError(f"{policy_name}/{artifact_id} generation count changed")
        generation_file, rows = _validate_file_binding(
            generation.get("generation"),
            owner=receipt_path,
            label=f"{policy_name}/{artifact_id} generation rows",
            expected_rows=EXPECTED_ROWS_PER_ARTIFACT,
        )
        assert rows is not None
        if any(row.get("artifact_id") != artifact_id for row in rows):
            raise ValueError(f"{policy_name}/{artifact_id} generation artifact ID changed")
        generation_ids = [str(row.get("sample_id") or "").strip() for row in rows]
        if generation_ids != prompt_ids or any(
            not isinstance(row.get("valid_generation"), bool) for row in rows
        ):
            raise ValueError(
                f"{policy_name}/{artifact_id} generation sample/provenance matrix changed"
            )
        manifest_file, _ = _validate_file_binding(
            generation.get("manifest"),
            owner=receipt_path,
            label=f"{policy_name}/{artifact_id} generation manifest",
            sealed=True,
        )
        manifest_payload = _read_json(
            Path(manifest_file["path"]),
            label=f"{policy_name}/{artifact_id} generation manifest",
        )
        output_binding = _mapping(
            manifest_payload.get("output"),
            label=f"{policy_name}/{artifact_id} generation manifest.output",
        )
        output_path = _resolve_path(
            output_binding.get("path"),
            owner=Path(manifest_file["path"]),
            label=f"{policy_name}/{artifact_id} generation manifest.output",
        )
        actual_valid = sum(row.get("valid_generation") is True for row in rows)
        if (
            manifest_payload.get("schema_version")
            != "checkpoint-artifact-generation-manifest-v2"
            or manifest_payload.get("artifact_id") != artifact_id
            or manifest_payload.get("sample_count") != EXPECTED_ROWS_PER_ARTIFACT
            or manifest_payload.get("valid_count") != actual_valid
            or manifest_payload.get("invalid_count")
            != EXPECTED_ROWS_PER_ARTIFACT - actual_valid
            or output_path != Path(generation_file["path"])
            or output_binding.get("sha256") != generation_file["sha256"]
            or output_binding.get("row_count") != EXPECTED_ROWS_PER_ARTIFACT
        ):
            raise ValueError(
                f"{policy_name}/{artifact_id} generation manifest binding changed"
            )
        progress = _mapping(
            generation.get("progress"), label=f"{policy_name}/{artifact_id} progress"
        )
        if progress.get("row_count") != EXPECTED_ROWS_PER_ARTIFACT:
            raise ValueError(f"{policy_name}/{artifact_id} progress count changed")
        verified_generations[artifact_id] = {
            "generation": generation_file,
            "manifest": manifest_file,
            "model_sha256": generation.get("model_sha256"),
            "tokenizer_sha256": generation.get("tokenizer_sha256"),
            "sample_count": EXPECTED_ROWS_PER_ARTIFACT,
            "progress": dict(progress),
        }
        generation_rows[artifact_id] = rows

    normalized_source = dict(receipt)
    for key in (
        "integrity",
        "schema_version",
        "scoring_policy",
        "bootstrap_samples",
        "bootstrap_seed",
        "core_evaluation",
    ):
        normalized_source.pop(key, None)
    return {
        "receipt": {
            "path": str(receipt_path.resolve()),
            "sha256": sha256_file(receipt_path),
            "payload_sha256": payload_sha,
        },
        "normalized_source": normalized_source,
        "frozen_files": frozen_files,
        "source_chain_files": source_chain_files,
        "generations": verified_generations,
        "generation_rows": generation_rows,
    }


def _validate_bound_output(
    *,
    audit: Mapping[str, Any],
    name: str,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    outputs = _mapping(audit.get("output_artifacts"), label="audit.output_artifacts")
    record = _mapping(outputs.get(name), label=f"audit.output_artifacts.{name}")
    sealed_path = _resolve_path(record.get("path"), owner=path, label=name)
    resolved = path.resolve()
    if sealed_path != resolved:
        raise ValueError(f"{name} path is not the artifact sealed by the audit")
    digest = sha256_file(resolved)
    if record.get("sha256") != digest:
        raise ValueError(f"{name} hash differs from the sealed audit")
    declared = _integer(record.get("row_count"), label=f"{name}.row_count")
    audit_counts = _mapping(audit.get("row_counts"), label="audit.row_counts")
    if declared != len(rows) or audit_counts.get(name) != len(rows):
        raise ValueError(f"{name} row count differs from the sealed audit")
    return {"path": str(resolved), "sha256": digest, "row_count": len(rows)}


def _validate_matrix(
    *, audit: Mapping[str, Any], row_scores: Sequence[Mapping[str, Any]], policy: str
) -> dict[str, Any]:
    if (
        audit.get("artifact_count") != 3
        or audit.get("artifact_ids") != list(EXPECTED_ARTIFACT_IDS)
    ):
        raise ValueError("Evaluation audit does not contain the expected three artifacts")
    alignment = _mapping(audit.get("alignment"), label="audit.alignment")
    expected_counts = {
        artifact_id: EXPECTED_ROWS_PER_ARTIFACT
        for artifact_id in EXPECTED_ARTIFACT_IDS
    }
    if (
        alignment.get("complete") is not True
        or alignment.get("artifact_ids") != list(EXPECTED_ARTIFACT_IDS)
        or alignment.get("sample_universe_count") != EXPECTED_ROWS_PER_ARTIFACT
        or alignment.get("reference_sample_count") != EXPECTED_ROWS_PER_ARTIFACT
        or alignment.get("evidence_sample_count") != EXPECTED_ROWS_PER_ARTIFACT
        or alignment.get("generation_sample_count_by_artifact") != expected_counts
    ):
        raise ValueError("Audit does not attest the complete 3 x 33 matrix")
    for key in (
        "missing_reference_sample_ids",
        "missing_evidence_sample_ids",
        "reference_without_evidence_sample_ids",
        "evidence_without_reference_sample_ids",
    ):
        if alignment.get(key) != []:
            raise ValueError(f"Audit alignment is incomplete: {key}")
    if alignment.get("missing_generation_sample_ids_by_artifact") != {
        artifact_id: [] for artifact_id in EXPECTED_ARTIFACT_IDS
    }:
        raise ValueError("Audit alignment contains missing generation rows")
    if len(row_scores) != 3 * EXPECTED_ROWS_PER_ARTIFACT:
        raise ValueError("row_scores is not a 3 x 33 matrix")

    ids: dict[str, set[str]] = {artifact_id: set() for artifact_id in EXPECTED_ARTIFACT_IDS}
    expected_policy = POLICY_SPECS[policy]["policy_id"]
    for number, row in enumerate(row_scores, start=1):
        if row.get("schema_version") != EVALUATOR_SCHEMA_VERSION:
            raise ValueError(f"row_scores row {number} has an unsupported schema")
        artifact_id = str(row.get("artifact_id") or "")
        sample_id = str(row.get("sample_id") or "").strip()
        if artifact_id not in ids or not sample_id:
            raise ValueError(f"row_scores row {number} has an invalid matrix key")
        if sample_id in ids[artifact_id]:
            raise ValueError(f"row_scores contains duplicate key {artifact_id}/{sample_id}")
        if row.get("scoring_policy") != expected_policy:
            raise ValueError(f"row_scores row {number} uses the wrong scoring policy")
        ids[artifact_id].add(sample_id)
    if any(len(values) != EXPECTED_ROWS_PER_ARTIFACT for values in ids.values()):
        raise ValueError("row_scores does not contain 33 rows per artifact")
    if len({frozenset(values) for values in ids.values()}) != 1:
        raise ValueError("row_scores artifacts do not share the same sample matrix")
    sample_ids = sorted(ids[CHK0_ARTIFACT_ID])
    return {
        "artifact_count": 3,
        "rows_per_artifact": EXPECTED_ROWS_PER_ARTIFACT,
        "total_rows": 3 * EXPECTED_ROWS_PER_ARTIFACT,
        "paired_sample_keys_identical": True,
        "sample_ids_sha256": manifest_payload_sha256({"sample_ids": sample_ids}),
    }


def _normalized_run_spec(run_spec: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(run_spec)
    result.pop("scoring_policy", None)
    result.pop("invalid_output_policy_version", None)
    return result


def _validate_policy_bundle(directory: Path, *, policy_name: str) -> dict[str, Any]:
    root = directory.expanduser().resolve()
    paths = {
        "audit": root / "audit.json",
        "input_validation": root / "input_validation.json",
        "summary": root / "summary.jsonl",
        "contrasts": root / "contrasts.jsonl",
        "row_scores": root / "row_scores.jsonl",
        "claim_checks": root / "claim_checks.jsonl",
    }
    audit = _read_json(paths["audit"], label=f"{policy_name} audit")
    audit_payload_sha = validate_manifest_integrity(audit)
    if (
        audit.get("schema_version") != EVALUATOR_SCHEMA_VERSION
        or audit.get("status") != "validated"
        or audit.get("immutable") is not True
    ):
        raise ValueError(f"{policy_name} audit is not immutable and validated")
    scoring = _mapping(audit.get("scoring_policy"), label=f"{policy_name} scoring_policy")
    if scoring.get("policy_id") != POLICY_SPECS[policy_name]["policy_id"]:
        raise ValueError(f"{policy_name} audit uses the wrong scoring policy")
    run_spec = _mapping(audit.get("run_spec"), label=f"{policy_name} run_spec")
    if audit.get("run_spec_sha256") != manifest_payload_sha256(run_spec):
        raise ValueError(f"{policy_name} audit run_spec hash changed")
    if (
        run_spec.get("artifact_ids") != list(EXPECTED_ARTIFACT_IDS)
        or run_spec.get("required_artifact_ids") != list(EXPECTED_ARTIFACT_IDS)
        or run_spec.get("expected_artifact_count") != 3
        or run_spec.get("bootstrap_samples") != EXPECTED_BOOTSTRAP_SAMPLES
        or run_spec.get("bootstrap_seed") != EXPECTED_BOOTSTRAP_SEED
        or run_spec.get("scoring_policy") != scoring
    ):
        raise ValueError(f"{policy_name} run_spec execution contract changed")
    expected_subsets = run_spec.get("evaluation_subsets")
    if not isinstance(expected_subsets, Mapping) or set(expected_subsets) != set(
        EXPECTED_SUBSET_ROWS
    ):
        raise ValueError(f"{policy_name} run_spec subsets changed")

    input_validation = _validate_input_validation(
        receipt_path=paths["input_validation"],
        audit_path=paths["audit"],
        audit=audit,
        policy_name=policy_name,
    )
    rows_by_name = {
        name: _read_jsonl(paths[name], label=f"{policy_name} {name}")
        for name in ("summary", "contrasts", "row_scores", "claim_checks")
    }
    source_artifacts = {
        "input_validation": input_validation["receipt"],
        "audit": {
            "path": str(paths["audit"]),
            "sha256": sha256_file(paths["audit"]),
            "payload_sha256": audit_payload_sha,
        },
        **{
            name: _validate_bound_output(
                audit=audit, name=name, path=paths[name], rows=rows
            )
            for name, rows in rows_by_name.items()
        },
    }
    matrix = _validate_matrix(
        audit=audit, row_scores=rows_by_name["row_scores"], policy=policy_name
    )

    summary_index: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for number, row in enumerate(rows_by_name["summary"], start=1):
        key = (
            str(row.get("artifact_id") or ""),
            str(row.get("evaluation_subset") or ""),
            str(row.get("metric") or ""),
        )
        if row.get("schema_version") != EVALUATOR_SCHEMA_VERSION:
            raise ValueError(f"{policy_name} summary row {number} has wrong schema")
        if key[0] not in EXPECTED_ARTIFACT_IDS or key in summary_index:
            raise ValueError(f"{policy_name} summary contains an invalid key {key}")
        summary_index[key] = row

    contrast_index: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for number, row in enumerate(rows_by_name["contrasts"], start=1):
        edge = (
            str(row.get("baseline_artifact_id") or ""),
            str(row.get("candidate_artifact_id") or ""),
        )
        key = (
            *edge,
            str(row.get("evaluation_subset") or ""),
            str(row.get("metric") or ""),
        )
        if row.get("schema_version") != EVALUATOR_SCHEMA_VERSION:
            raise ValueError(f"{policy_name} contrast row {number} has wrong schema")
        if edge not in ADJACENT_CONTRASTS or key in contrast_index:
            raise ValueError(f"{policy_name} contrasts contain an invalid edge/key {key}")
        contrast_index[key] = row

    return {
        "audit": audit,
        "run_spec": dict(run_spec),
        "input_validation": input_validation,
        "source_artifacts": source_artifacts,
        "matrix": matrix,
        "summary_index": summary_index,
        "contrast_index": contrast_index,
        "row_scores": rows_by_name["row_scores"],
    }


def _artifact_estimate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mean": row.get("mean"),
        "median": row.get("median"),
        "ci_95": {"lower": row.get("ci_lower"), "upper": row.get("ci_upper")},
        "n_rows_total": row.get("n_rows_total"),
        "n_rows_eligible": row.get("n_rows_eligible"),
        "n_rows_ineligible": row.get("n_rows_ineligible"),
        "n_meetings": row.get("n_meetings"),
    }


def _metric_result(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    contrast: Mapping[str, Any],
) -> dict[str, Any]:
    for label, row in (("baseline", baseline), ("candidate", candidate), ("contrast", contrast)):
        if (
            row.get("confidence") != 0.95
            or row.get("bootstrap_unit") != "meeting_id"
            or row.get("bootstrap_samples") != EXPECTED_BOOTSTRAP_SAMPLES
            or row.get("bootstrap_seed") != EXPECTED_BOOTSTRAP_SEED
        ):
            raise ValueError(f"Unexpected inference metadata for {label}")
    directions = {
        baseline.get("metric_direction"),
        candidate.get("metric_direction"),
        contrast.get("metric_direction"),
    }
    if len(directions) != 1 or None in directions:
        raise ValueError(f"Metric direction mismatch for {baseline.get('metric')}")
    return {
        "metric_direction": directions.pop(),
        "baseline": _artifact_estimate(baseline),
        "candidate": _artifact_estimate(candidate),
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
            "inference_status": contrast.get("inference_status"),
            "p_value": contrast.get("p_value"),
            "p_value_holm": contrast.get("p_value_holm"),
            "effect_size_paired_dz": contrast.get("effect_size_paired_dz"),
        },
    }


def _population_summary(
    bundle: Mapping[str, Any], *, artifact_id: str, subset: str
) -> Mapping[str, Any]:
    key = (artifact_id, subset, "valid_output")
    row = bundle["summary_index"].get(key)
    if row is None:
        raise ValueError(f"Missing valid_output summary row {key}")
    expected = EXPECTED_SUBSET_ROWS[subset]
    n_valid = _integer(row.get("n_valid_output"), label=f"{key}.n_valid_output")
    n_invalid = _integer(row.get("n_invalid_output"), label=f"{key}.n_invalid_output")
    n_missing = _integer(row.get("n_missing_input"), label=f"{key}.n_missing_input")
    if row.get("n_rows_total") != expected or n_valid + n_invalid != expected or n_missing:
        raise ValueError(f"Inconsistent valid-output population for {key}")
    return row


def _edge_classification(
    *, baseline_valid: int, candidate_valid: int, population: int
) -> str:
    if baseline_valid == 0 and candidate_valid == 0:
        return "stress_benchmark_inconclusive_bilateral_delivery_failure"
    if baseline_valid == 0 or candidate_valid == 0:
        return "stress_benchmark_inconclusive_asymmetric_delivery_failure"
    if baseline_valid < population or candidate_valid < population:
        return "stress_benchmark_partial_delivery_comparison"
    return "stress_benchmark_comparison_available"


def _overall_classification(strict_bundle: Mapping[str, Any]) -> dict[str, Any]:
    rows = {
        artifact_id: _population_summary(
            strict_bundle, artifact_id=artifact_id, subset="all_11_meetings"
        )
        for artifact_id in EXPECTED_ARTIFACT_IDS
    }
    valid = {
        artifact_id: _integer(row.get("n_valid_output"), label=f"{artifact_id}.valid")
        for artifact_id, row in rows.items()
    }
    if all(count == 0 for count in valid.values()):
        label = "stress_benchmark_inconclusive_all_artifacts_delivery_failure"
    elif any(count == 0 for count in valid.values()):
        label = "stress_benchmark_inconclusive_due_to_delivery_failure"
    elif any(count < EXPECTED_ROWS_PER_ARTIFACT for count in valid.values()):
        label = "stress_benchmark_partial_delivery_comparison"
    else:
        label = "stress_benchmark_comparison_available"
    return {
        "label": label,
        "basis_policy": POLICY_SPECS["strict"]["policy_id"],
        "basis_subset": "all_11_meetings",
        "valid_strict_final_answers_by_artifact": valid,
        "zero_metric_guard": (
            "Zero-valued similarity/factual metrics on strict delivery failures are "
            "finite worst-case sentinels, not evidence that the delivered texts have "
            "zero semantic similarity."
        ),
    }


def _numeric_distribution(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"available_count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "available_count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def _generation_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    valid_count = sum(row.get("valid_generation") is True for row in rows)
    finish = Counter(str(row.get("generation_finish_reason") or "missing") for row in rows)
    stop = Counter(str(row.get("generation_stop_reason") or "missing") for row in rows)
    formats = Counter(str(row.get("response_format") or "missing") for row in rows)
    outputs = [
        int(value)
        for row in rows
        if isinstance((value := row.get("output_token_count")), int)
        and not isinstance(value, bool)
        and value >= 0
    ]
    native_present = 0
    native_valid = 0
    for row in rows:
        completion = str(row.get("generated") or "")
        boundary_count = completion.casefold().count("</think>")
        native_present += boundary_count > 0
        if boundary_count == 1 and completion.casefold().split("</think>", 1)[1].strip():
            native_valid += 1
    normal_stop = finish.get("stop", 0)
    return {
        "row_count": total,
        "valid_generation_count": valid_count,
        "valid_generation_rate": valid_count / total,
        "finish_reason_counts": dict(sorted(finish.items())),
        "normal_stop_count": normal_stop,
        "normal_stop_rate": normal_stop / total,
        "eos_contract": (
            "generation_finish_reason=stop is the recorded normal EOS/stop delivery "
            "signal; generation_stop_reason is retained separately and is not guessed."
        ),
        "generation_stop_reason_counts": dict(sorted(stop.items())),
        "output_token_count": _numeric_distribution(outputs),
        "response_format_counts": dict(sorted(formats.items())),
        "native_think_boundary": {
            "literal": "</think>",
            "present_count": native_present,
            "exactly_one_with_nonempty_suffix_count": native_valid,
            "rate": native_valid / total,
        },
    }


def _artifact_policy_diagnostics(
    bundle: Mapping[str, Any], *, policy_name: str, artifact_id: str
) -> dict[str, Any]:
    summary = bundle["summary_index"]
    result: dict[str, Any] = {}
    for subset in EXPECTED_SUBSET_ROWS:
        valid = _population_summary(bundle, artifact_id=artifact_id, subset=subset)
        token = summary.get((artifact_id, subset, "generated_token_count"))
        repetition = summary.get((artifact_id, subset, "repetition_rate"))
        if token is None or repetition is None:
            raise ValueError(f"Missing length/repetition summaries for {artifact_id}/{subset}")
        result[subset] = {
            "text_semantics": POLICY_SPECS[policy_name]["candidate_label"],
            "valid_output": {
                "count": valid.get("n_valid_output"),
                "invalid_count": valid.get("n_invalid_output"),
                "rate": valid.get("mean"),
            },
            "length_tokens": _artifact_estimate(token),
            "repetition_rate": _artifact_estimate(repetition),
        }
    if policy_name == "length_tolerant":
        modes = Counter(
            str(row.get("candidate_extraction_mode") or "missing")
            for row in bundle["row_scores"]
            if row.get("artifact_id") == artifact_id
        )
        result["raw_completion_extraction_mode_counts"] = dict(sorted(modes.items()))
        result["terminology_guard"] = (
            "These values score extracted raw-completion diagnostic text and must not "
            "be described as final-answer quality."
        )
    return result


def _edge_policy_summary(
    bundle: Mapping[str, Any], *, baseline_id: str, candidate_id: str
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for subset, expected_rows in EXPECTED_SUBSET_ROWS.items():
        baseline_valid = _population_summary(
            bundle, artifact_id=baseline_id, subset=subset
        )
        candidate_valid = _population_summary(
            bundle, artifact_id=candidate_id, subset=subset
        )
        groups: dict[str, Any] = {}
        for group_name, metrics in METRIC_GROUPS.items():
            group: dict[str, Any] = {}
            for metric in metrics:
                baseline_key = (baseline_id, subset, metric)
                candidate_key = (candidate_id, subset, metric)
                contrast_key = (baseline_id, candidate_id, subset, metric)
                try:
                    baseline = bundle["summary_index"][baseline_key]
                    candidate = bundle["summary_index"][candidate_key]
                    contrast = bundle["contrast_index"][contrast_key]
                except KeyError as exc:
                    raise ValueError(
                        f"Missing required adjacent contrast result {contrast_key}"
                    ) from exc
                if contrast.get("n_common_samples_total") != expected_rows:
                    raise ValueError(f"Unexpected paired population for {contrast_key}")
                group[metric] = _metric_result(baseline, candidate, contrast)
            groups[group_name] = group
        baseline_count = _integer(
            baseline_valid.get("n_valid_output"), label="baseline valid count"
        )
        candidate_count = _integer(
            candidate_valid.get("n_valid_output"), label="candidate valid count"
        )
        result[subset] = {
            "role": "primary" if subset == "all_11_meetings" else "secondary",
            "classification": _edge_classification(
                baseline_valid=baseline_count,
                candidate_valid=candidate_count,
                population=expected_rows,
            ),
            "valid_output_counts": {
                baseline_id: baseline_count,
                candidate_id: candidate_count,
            },
            "metric_groups": groups,
        }
    return result


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to replace immutable summary: {target}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def summarize_chk0_chk1_chk3_cp250_generation_eval(
    *, strict_dir: Path, length_tolerant_dir: Path, output_file: Path
) -> dict[str, Any]:
    """Validate both policies and write one immutable, sealed three-leg summary."""

    bundles = {
        "strict": _validate_policy_bundle(strict_dir, policy_name="strict"),
        "length_tolerant": _validate_policy_bundle(
            length_tolerant_dir, policy_name="length_tolerant"
        ),
    }
    if _normalized_run_spec(bundles["strict"]["run_spec"]) != _normalized_run_spec(
        bundles["length_tolerant"]["run_spec"]
    ):
        raise ValueError("Strict and length-tolerant bundles are not bound to the same run inputs")
    if bundles["strict"]["matrix"] != bundles["length_tolerant"]["matrix"]:
        raise ValueError("Strict and length-tolerant matrix attestations differ")
    if (
        bundles["strict"]["input_validation"]["normalized_source"]
        != bundles["length_tolerant"]["input_validation"]["normalized_source"]
    ):
        raise ValueError(
            "Strict and length-tolerant input_validation receipts do not bind the same sources"
        )
    strict_generation_rows = bundles["strict"]["input_validation"]["generation_rows"]
    tolerant_generation_rows = bundles["length_tolerant"]["input_validation"]["generation_rows"]
    if strict_generation_rows != tolerant_generation_rows:
        raise ValueError("Strict and length-tolerant generation matrices differ")

    result = seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "validated",
            "benchmark_class": "official_minutes_reference_similarity_stress_benchmark",
            "overall_classification": _overall_classification(bundles["strict"]),
            "matrix": {
                **bundles["strict"]["matrix"],
                "artifact_ids": list(EXPECTED_ARTIFACT_IDS),
                "adjacent_contrasts": [
                    {"baseline": baseline, "candidate": candidate}
                    for baseline, candidate in ADJACENT_CONTRASTS
                ],
                "same_sources_across_policies": True,
            },
            "artifacts": {
                artifact_id: {
                    "generation_delivery": _generation_diagnostics(
                        strict_generation_rows[artifact_id]
                    ),
                    "strict_primary": _artifact_policy_diagnostics(
                        bundles["strict"], policy_name="strict", artifact_id=artifact_id
                    ),
                    "length_tolerant_raw_completion_diagnostics": _artifact_policy_diagnostics(
                        bundles["length_tolerant"],
                        policy_name="length_tolerant",
                        artifact_id=artifact_id,
                    ),
                }
                for artifact_id in EXPECTED_ARTIFACT_IDS
            },
            "contrasts": {
                f"{baseline}_to_{candidate}": {
                    "baseline_artifact_id": baseline,
                    "candidate_artifact_id": candidate,
                    "strict_primary": _edge_policy_summary(
                        bundles["strict"], baseline_id=baseline, candidate_id=candidate
                    ),
                    "length_tolerant_raw_completion_diagnostics": {
                        "terminology_guard": (
                            "Diagnostic scores apply to extracted raw-completion text, "
                            "not final answers."
                        ),
                        "subsets": _edge_policy_summary(
                            bundles["length_tolerant"],
                            baseline_id=baseline,
                            candidate_id=candidate,
                        ),
                    },
                }
                for baseline, candidate in ADJACENT_CONTRASTS
            },
            "policies": {
                policy_name: {
                    **POLICY_SPECS[policy_name],
                    "source_artifacts": bundle["source_artifacts"],
                }
                for policy_name, bundle in bundles.items()
            },
            "interpretation_boundary": {
                "primary": "strict-final-answer-v2 on all_11_meetings",
                "secondary": "strict-final-answer-v2 on prospective_only_9_meetings",
                "diagnostic_only": (
                    "length-tolerant-open-tags-v1 uses extracted raw completions; it "
                    "cannot replace, rescue, or be described as the strict final-answer result."
                ),
                "task_scope": (
                    "This is an out-of-task official-Minutes similarity stress benchmark, "
                    "not a direct chk1 analysis-quality or chk3 promotion test."
                ),
                "not_supported": [
                    "calling finite worst-case zeros evidence of zero semantic similarity",
                    "causal attribution of differences to a training stage",
                    "automatic model promotion or canonical-DAG certification",
                    "calling length-tolerant raw-completion diagnostic text an answer",
                ],
                "generation_or_scoring_run_by_this_summary": False,
            },
        }
    )
    _write_immutable(output_file, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-dir", required=True, type=Path)
    parser.add_argument("--length-tolerant-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = summarize_chk0_chk1_chk3_cp250_generation_eval(
        strict_dir=args.strict_dir,
        length_tolerant_dir=args.length_tolerant_dir,
        output_file=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "schema_version": result["schema_version"],
                "overall_classification": result["overall_classification"]["label"],
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
