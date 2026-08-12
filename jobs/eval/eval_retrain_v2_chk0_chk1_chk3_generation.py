"""Score sealed chk0/chk1 generations with a new standalone chk3-cp250 leg.

This entry point is deliberately asymmetric.  The already-produced chk0/chk1
artifacts remain bound to their original two-artifact checkpoint manifest and
progress-v2 generation manifests.  The new chk3 artifact is bound to a separate
single-leg checkpoint manifest.  A third, combined lineage manifest is used
only after both evidence chains have been validated and declares the adjacent
chk0 -> chk1 and chk1 -> chk3 contrasts.

No generation is performed by this module.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from jobs.eval import eval_retrain_v2_chk0_chk1_generation as pair_eval
from jobs.eval.eval_checkpoint_generation import (
    BERTScoreBackend,
    MPNetCosineBackend,
    SCORING_POLICIES,
    STRICT_SCORING_POLICY,
    run_checkpoint_generation_evaluation,
)
from jobs.generation.generate_checkpoint_artifact import (
    MANIFEST_SCHEMA_VERSION as GENERATION_MANIFEST_SCHEMA_VERSION,
    validate_generation_progress_binding,
)
from open_r1.provenance import sha256_file, validate_sha256
from open_r1.validator.loo_generation_spec import (
    derive_row_seed,
    seal_manifest,
    validate_manifest_integrity,
)


SCHEMA_VERSION = "retrain-v2-chk0-chk1-chk3-generation-evaluation-v1"
VALIDATION_RECEIPT_SCHEMA_VERSION = (
    "retrain-v2-chk0-chk1-chk3-generation-input-validation-v1"
)
CHECKPOINT_MANIFEST_SCHEMA_VERSION = "retrain-v2-checkpoint-eval-manifest-v1"
LINEAGE_MANIFEST_SCHEMA_VERSION = "retrain-v2-eval-lineage-v1"
EXACT_MERGE_SCHEMA_VERSION = "lora-merge-lineage-evidence-v1"
EXACT_MERGE_ALGORITHM_VERSION = "peft-lora-fp32-exact-v1"
EXACT_MERGE_CONCLUSION = "exact_base_plus_adapter_merge_verified"
SELECTION_SCHEMA_VERSION = "chk3-standalone-checkpoint-selection-v1"
AUTHORIZATION_SCHEMA_VERSION = "chk3-cp250-generation-comparison-authorization-v1"

BASE_ARTIFACT_ID = pair_eval.BASE_ARTIFACT_ID
CHK1_ARTIFACT_ID = pair_eval.CHK1_ARTIFACT_ID
CHK3_ARTIFACT_ID = "eval-chk3-direct-chk1-cp200-sft-cp250"
PAIR_ARTIFACT_IDS = (BASE_ARTIFACT_ID, CHK1_ARTIFACT_ID)
ARTIFACT_IDS = (*PAIR_ARTIFACT_IDS, CHK3_ARTIFACT_ID)

EXPECTED_SAMPLE_COUNT = pair_eval.EXPECTED_SAMPLE_COUNT
EXPECTED_MEETING_COUNT = pair_eval.EXPECTED_MEETING_COUNT
EXPECTED_PROSPECTIVE_MEETING_COUNT = pair_eval.EXPECTED_PROSPECTIVE_MEETING_COUNT
EXPECTED_SECTION_COUNT = pair_eval.EXPECTED_SECTION_COUNT


def _read_json(path: Path, *, label: str, sealed: bool = False) -> dict[str, Any]:
    return pair_eval._read_json(path, label=label, sealed=sealed)


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    return pair_eval._read_jsonl(path, label=label)


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    return pair_eval._mapping(value, label=label)


def _integer(value: object, *, label: str) -> int:
    return pair_eval._integer(value, label=label)


def _resolve_binding_path(
    record: Mapping[str, Any],
    *,
    manifest_path: Path,
    repo_root: Path,
    label: str,
) -> Path:
    return pair_eval._resolve_bound_path(
        record.get("path"),
        repo_root=repo_root,
        manifest_path=manifest_path,
        label=label,
        require_directory=False,
    )


def _validate_document_binding(
    value: object,
    *,
    manifest_path: Path,
    repo_root: Path,
    label: str,
    actual_path: Path | None = None,
    sealed: bool,
) -> tuple[Path, dict[str, Any]]:
    record = _mapping(value, label=label)
    path = _resolve_binding_path(
        record, manifest_path=manifest_path, repo_root=repo_root, label=label
    )
    if actual_path is not None and path != actual_path.expanduser().resolve():
        raise ValueError(f"{label} binds a different file")
    expected_sha = validate_sha256(
        record.get("file_sha256", record.get("sha256")),
        label=f"{label}.file_sha256",
    )
    if expected_sha != sha256_file(path):
        raise ValueError(f"{label} file hash changed")
    payload = _read_json(path, label=label, sealed=sealed)
    if sealed:
        declared_payload = validate_sha256(
            record.get("payload_sha256"), label=f"{label}.payload_sha256"
        )
        observed_payload = validate_manifest_integrity(payload)
        if declared_payload != observed_payload:
            raise ValueError(f"{label} payload hash changed")
    return path, payload


def _validate_directory_record(
    record: Mapping[str, Any],
    *,
    path_key: str,
    hash_key: str,
    manifest_path: Path,
    repo_root: Path,
    label: str,
) -> tuple[Path, str]:
    path = pair_eval._resolve_bound_path(
        record.get(path_key),
        repo_root=repo_root,
        manifest_path=manifest_path,
        label=f"{label}.{path_key}",
        require_directory=True,
    )
    digest = validate_sha256(record.get(hash_key), label=f"{label}.{hash_key}")
    return path, digest


def _validate_selection_receipt(
    selection: Mapping[str, Any],
    *,
    adapter_path: Path,
    evidence: Mapping[str, Any],
    repo_root: Path,
    selection_path: Path,
) -> dict[str, Any]:
    if selection.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError("chk3 selection receipt has unsupported schema_version")
    if selection.get("status") != "selected" or selection.get("selected_checkpoint") != 250:
        raise ValueError("chk3 selection receipt does not select checkpoint-250")
    scope = _mapping(selection.get("scope"), label="selection receipt.scope")
    if scope.get("stage") != "chk3" or scope.get("promotable_to_canonical_dag") is not False:
        raise ValueError("chk3 selection receipt scope is not standalone/non-promotable")
    checkpoint = _mapping(
        selection.get("checkpoint"), label="selection receipt.checkpoint"
    )
    selected_path = pair_eval._resolve_bound_path(
        checkpoint.get("path"),
        repo_root=repo_root,
        manifest_path=selection_path,
        label="selection receipt checkpoint",
        require_directory=True,
    )
    if selected_path != adapter_path:
        raise ValueError("selection receipt adapter path differs from exact merge evidence")
    critical = _mapping(
        _mapping(
            _mapping(evidence.get("sources"), label="exact merge sources").get(
                "critical_files"
            ),
            label="exact merge critical_files",
        ).get("adapter_weights"),
        label="exact merge adapter_weights",
    )
    if validate_sha256(
        checkpoint.get("adapter_model_sha256"),
        label="selection receipt adapter_model_sha256",
    ) != validate_sha256(
        critical.get("sha256"), label="exact merge adapter_weights.sha256"
    ):
        raise ValueError("selection receipt adapter weights differ from exact merge evidence")
    if checkpoint.get("adapter_tensor_count") != 448:
        raise ValueError("selection receipt adapter tensor inventory changed")
    if checkpoint.get("nonfinite_tensor_count") != 0:
        raise ValueError("selection receipt contains non-finite adapter tensors")
    gate = _mapping(selection.get("generation_gate"), label="selection generation_gate")
    if gate.get("status") != "passed" or gate.get("quality_valid_cases") != gate.get("cases"):
        raise ValueError("selection receipt generation gate did not fully pass")
    return {
        "selected_checkpoint": 250,
        "adapter_path": str(adapter_path),
        "adapter_weights_sha256": critical["sha256"],
        "promotable_to_canonical_dag": False,
    }


def _validate_authorization(
    authorization: Mapping[str, Any],
    *,
    parent_model_path: Path,
    parent_model_sha256: str,
    adapter_path: Path,
    adapter_sha256: str,
    merged_path: Path,
    repo_root: Path,
    authorization_path: Path,
) -> dict[str, Any]:
    if authorization.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        raise ValueError("chk3 evaluation authorization has unsupported schema_version")
    if authorization.get("status") != "authorized":
        raise ValueError("chk3 generation comparison is not authorized")
    scope = _mapping(authorization.get("scope"), label="authorization.scope")
    if (
        scope.get("checkpoint_step") != 250
        or scope.get("artifact_status") != "evaluation_only"
        or scope.get("canonical_dag_promotable") is not False
        or scope.get("downstream_training_allowed") is not False
    ):
        raise ValueError("chk3 authorization scope changed")
    bindings = _mapping(authorization.get("bindings"), label="authorization.bindings")
    base = _mapping(bindings.get("base_model"), label="authorization base_model")
    base_path = pair_eval._resolve_bound_path(
        base.get("path"),
        repo_root=repo_root,
        manifest_path=authorization_path,
        label="authorization base_model.path",
        require_directory=True,
    )
    if base_path != parent_model_path or validate_sha256(
        base.get("sha256"), label="authorization base_model.sha256"
    ) != parent_model_sha256:
        raise ValueError("authorization base model differs from chk1")
    adapter = _mapping(
        bindings.get("adapter_checkpoint"), label="authorization adapter_checkpoint"
    )
    bound_adapter = pair_eval._resolve_bound_path(
        adapter.get("path"),
        repo_root=repo_root,
        manifest_path=authorization_path,
        label="authorization adapter_checkpoint.path",
        require_directory=True,
    )
    if bound_adapter != adapter_path or validate_sha256(
        adapter.get("directory_sha256"),
        label="authorization adapter_checkpoint.directory_sha256",
    ) != adapter_sha256:
        raise ValueError("authorization adapter differs from chk3 checkpoint")
    raw_destination = Path(str(bindings.get("merged_destination") or "")).expanduser()
    destination = (
        raw_destination
        if raw_destination.is_absolute()
        else repo_root / raw_destination
    ).resolve()
    if destination != merged_path:
        raise ValueError("authorization merged destination differs from chk3 model")
    return {
        "checkpoint_step": 250,
        "artifact_status": "evaluation_only",
        "canonical_dag_promotable": False,
        "downstream_training_allowed": False,
    }


def _validate_chk3_exact_merge(
    evidence: Mapping[str, Any],
    *,
    evidence_path: Path,
    checkpoint_record: Mapping[str, Any],
    checkpoint_path: Path,
    parent_record: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    if evidence.get("schema_version") != EXACT_MERGE_SCHEMA_VERSION:
        raise ValueError("chk3 exact-merge evidence has unsupported schema_version")
    if evidence.get("algorithm_version") != EXACT_MERGE_ALGORITHM_VERSION:
        raise ValueError("chk3 exact-merge algorithm changed")
    if evidence.get("conclusion") != EXACT_MERGE_CONCLUSION:
        raise ValueError("chk3 exact merge is not verified")
    if evidence.get("subject_artifact_id") != CHK3_ARTIFACT_ID:
        raise ValueError("chk3 exact-merge subject_artifact_id changed")

    sources = _mapping(evidence.get("sources"), label="chk3 exact merge sources")
    expected = {
        "base_model": (
            Path(str(parent_record["resolved_model_path"])),
            str(parent_record["model_sha256"]),
        ),
        "merged_model": (
            Path(str(checkpoint_record["resolved_model_path"])),
            str(checkpoint_record["model_sha256"]),
        ),
        "adapter": (
            Path(str(checkpoint_record["resolved_adapter_path"])),
            str(checkpoint_record["adapter_sha256"]),
        ),
    }
    checked_sources: dict[str, Any] = {}
    for name, (expected_path, expected_sha) in expected.items():
        source = _mapping(sources.get(name), label=f"chk3 exact merge {name}")
        observed_path = pair_eval._resolve_bound_path(
            source.get("path"),
            repo_root=repo_root,
            manifest_path=evidence_path,
            label=f"chk3 exact merge {name}.path",
            require_directory=True,
        )
        observed_sha = validate_sha256(
            source.get("sha256"), label=f"chk3 exact merge {name}.sha256"
        )
        if observed_path != expected_path or observed_sha != expected_sha:
            raise ValueError(f"chk3 exact merge {name} differs from checkpoint manifests")
        checked_sources[name] = {"path": str(observed_path), "sha256": observed_sha}

    metadata = _mapping(
        evidence.get("metadata_evidence"), label="chk3 exact merge metadata_evidence"
    )
    adapter_metadata = _mapping(metadata.get("adapter"), label="chk3 exact merge adapter metadata")
    training = _mapping(
        metadata.get("training_config"), label="chk3 exact merge training_config"
    )
    if adapter_metadata.get("base_path_matches") is not True:
        raise ValueError("chk3 adapter base path was not verified")
    for group in ("lora_metadata_checks", "path_checks"):
        checks = _mapping(training.get(group), label=f"chk3 exact merge {group}")
        if not checks or any(value is not True for value in checks.values()):
            raise ValueError(f"chk3 exact merge {group} did not fully pass")

    tensor = _mapping(
        evidence.get("tensor_verification"), label="chk3 exact merge tensor_verification"
    )
    model_count = _integer(tensor.get("model_tensor_count"), label="model_tensor_count")
    adapted = _integer(
        tensor.get("adapted_model_tensor_count"), label="adapted_model_tensor_count"
    )
    exact_adapted = _integer(
        tensor.get("exact_adapted_model_tensor_count"),
        label="exact_adapted_model_tensor_count",
    )
    unchanged = _integer(
        tensor.get("unchanged_model_tensor_count"), label="unchanged_model_tensor_count"
    )
    exact_unchanged = _integer(
        tensor.get("exact_unchanged_model_tensor_count"),
        label="exact_unchanged_model_tensor_count",
    )
    adapter_tensor_count = _integer(
        tensor.get("adapter_tensor_count"), label="adapter_tensor_count"
    )
    mismatch_count = _integer(tensor.get("mismatch_count"), label="mismatch_count")
    adapted_tensors = tensor.get("adapted_tensors")
    if (
        model_count != adapted + unchanged
        or exact_adapted != adapted
        or exact_unchanged != unchanged
        or adapter_tensor_count != 2 * adapted
        or mismatch_count != 0
        or not isinstance(adapted_tensors, list)
        or len(adapted_tensors) != adapted
        or any(
            not isinstance(item, Mapping) or item.get("exact_reconstruction") is not True
            for item in adapted_tensors
        )
    ):
        raise ValueError("chk3 exact-merge tensor verification is incomplete")
    return {
        "path": str(evidence_path),
        "subject_artifact_id": CHK3_ARTIFACT_ID,
        "algorithm_version": EXACT_MERGE_ALGORITHM_VERSION,
        "conclusion": EXACT_MERGE_CONCLUSION,
        "sources": checked_sources,
        "tensor_verification": {
            "model_tensor_count": model_count,
            "adapted_model_tensor_count": adapted,
            "unchanged_model_tensor_count": unchanged,
            "adapter_tensor_count": adapter_tensor_count,
            "mismatch_count": mismatch_count,
            "complete": True,
        },
    }


def _validate_chk3_checkpoint_manifest(
    *,
    checkpoint_path: Path,
    parent_checkpoint_path: Path,
    config_path: Path,
    test_manifest_path: Path,
    pair_records: Mapping[str, Mapping[str, Any]],
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = _read_json(
        checkpoint_path, label="chk3 checkpoint manifest", sealed=True
    )
    if checkpoint.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("chk3 checkpoint manifest has unsupported schema_version")
    if checkpoint.get("evaluation_only") is not True:
        raise ValueError("chk3 checkpoint manifest must declare evaluation_only=true")
    if checkpoint.get("promotable_to_canonical_dag") is not False:
        raise ValueError("chk3 checkpoint manifest must declare non-promotable scope")
    raw_records = checkpoint.get("artifacts")
    if not isinstance(raw_records, list) or len(raw_records) != 1:
        raise ValueError("chk3 checkpoint manifest must contain one artifact")
    record = _mapping(raw_records[0], label="chk3 checkpoint artifact")
    if record.get("artifact_id") != CHK3_ARTIFACT_ID:
        raise ValueError("chk3 checkpoint artifact_id changed")
    if (
        record.get("design_checkpoint_id") != "chk-3-direct-cp250"
        or record.get("intended_parent_id") != "chk-1"
        or record.get("verified_parent_artifact_id") != CHK1_ARTIFACT_ID
        or record.get("usable_for_evaluation") is not True
    ):
        raise ValueError("chk3 checkpoint parent/design/evaluation contract changed")
    model_path, model_sha = _validate_directory_record(
        record,
        path_key="model_path",
        hash_key="model_sha256",
        manifest_path=checkpoint_path,
        repo_root=repo_root,
        label=CHK3_ARTIFACT_ID,
    )
    tokenizer_path, tokenizer_sha = _validate_directory_record(
        record,
        path_key="tokenizer_path",
        hash_key="tokenizer_sha256",
        manifest_path=checkpoint_path,
        repo_root=repo_root,
        label=CHK3_ARTIFACT_ID,
    )
    adapter_path, adapter_sha = _validate_directory_record(
        record,
        path_key="adapter_path",
        hash_key="adapter_sha256",
        manifest_path=checkpoint_path,
        repo_root=repo_root,
        label=CHK3_ARTIFACT_ID,
    )
    checked_record = {
        **dict(record),
        "model_sha256": model_sha,
        "tokenizer_sha256": tokenizer_sha,
        "adapter_sha256": adapter_sha,
        "resolved_model_path": str(model_path),
        "resolved_tokenizer_path": str(tokenizer_path),
        "resolved_adapter_path": str(adapter_path),
    }

    bindings = _mapping(checkpoint.get("bindings"), label="chk3 checkpoint bindings")
    required_bindings = {
        "evaluation_config",
        "test_manifest",
        "parent_checkpoint_manifest",
        "selection_receipt",
        "authorization",
        "exact_merge_evidence",
    }
    missing = required_bindings - set(bindings)
    if missing:
        raise ValueError(f"chk3 checkpoint bindings are incomplete: {sorted(missing)}")
    _validate_document_binding(
        bindings["evaluation_config"],
        manifest_path=checkpoint_path,
        repo_root=repo_root,
        label="chk3 bindings.evaluation_config",
        actual_path=config_path,
        sealed=False,
    )
    _validate_document_binding(
        bindings["test_manifest"],
        manifest_path=checkpoint_path,
        repo_root=repo_root,
        label="chk3 bindings.test_manifest",
        actual_path=test_manifest_path,
        sealed=True,
    )
    _validate_document_binding(
        bindings["parent_checkpoint_manifest"],
        manifest_path=checkpoint_path,
        repo_root=repo_root,
        label="chk3 bindings.parent_checkpoint_manifest",
        actual_path=parent_checkpoint_path,
        sealed=True,
    )
    evidence_path, evidence = _validate_document_binding(
        bindings["exact_merge_evidence"],
        manifest_path=checkpoint_path,
        repo_root=repo_root,
        label="chk3 bindings.exact_merge_evidence",
        sealed=True,
    )
    exact_validation = _validate_chk3_exact_merge(
        evidence,
        evidence_path=evidence_path,
        checkpoint_record=checked_record,
        checkpoint_path=checkpoint_path,
        parent_record=pair_records[CHK1_ARTIFACT_ID],
        repo_root=repo_root,
    )
    selection_path, selection = _validate_document_binding(
        bindings["selection_receipt"],
        manifest_path=checkpoint_path,
        repo_root=repo_root,
        label="chk3 bindings.selection_receipt",
        sealed=False,
    )
    selection_validation = _validate_selection_receipt(
        selection,
        adapter_path=adapter_path,
        evidence=evidence,
        repo_root=repo_root,
        selection_path=selection_path,
    )
    authorization_path, authorization = _validate_document_binding(
        bindings["authorization"],
        manifest_path=checkpoint_path,
        repo_root=repo_root,
        label="chk3 bindings.authorization",
        sealed=False,
    )
    authorization_validation = _validate_authorization(
        authorization,
        parent_model_path=Path(
            str(pair_records[CHK1_ARTIFACT_ID]["resolved_model_path"])
        ),
        parent_model_sha256=str(pair_records[CHK1_ARTIFACT_ID]["model_sha256"]),
        adapter_path=adapter_path,
        adapter_sha256=adapter_sha,
        merged_path=model_path,
        repo_root=repo_root,
        authorization_path=authorization_path,
    )
    return checked_record, {
        "manifest": pair_eval._file_binding(checkpoint_path, checkpoint),
        "exact_merge_evidence": {
            **pair_eval._file_binding(evidence_path, evidence),
            **exact_validation,
        },
        "selection_receipt": {
            **pair_eval._file_binding(selection_path),
            **selection_validation,
        },
        "authorization": {
            **pair_eval._file_binding(authorization_path),
            **authorization_validation,
        },
    }


def _binding_matches_file(
    value: object,
    *,
    manifest_path: Path,
    repo_root: Path,
    actual_path: Path,
    label: str,
    sealed: bool,
) -> None:
    _validate_document_binding(
        value,
        manifest_path=manifest_path,
        repo_root=repo_root,
        label=label,
        actual_path=actual_path,
        sealed=sealed,
    )


def _validate_combined_lineage(
    *,
    lineage_path: Path,
    pair_lineage_path: Path,
    pair_checkpoint_path: Path,
    chk3_checkpoint_path: Path,
    pair_exact_path: Path,
    chk3_exact_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lineage = _read_json(lineage_path, label="three-leg lineage manifest", sealed=True)
    if lineage.get("schema_version") != LINEAGE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("three-leg lineage manifest has unsupported schema_version")
    records = _mapping(lineage.get("checkpoints"), label="three-leg lineage checkpoints")
    if set(records) != set(ARTIFACT_IDS):
        raise ValueError("three-leg lineage artifact inventory changed")
    expected_parents = {
        BASE_ARTIFACT_ID: None,
        CHK1_ARTIFACT_ID: BASE_ARTIFACT_ID,
        CHK3_ARTIFACT_ID: CHK1_ARTIFACT_ID,
    }
    for artifact_id, expected_parent in expected_parents.items():
        record = _mapping(records[artifact_id], label=f"lineage {artifact_id}")
        if record.get("artifact_id") != artifact_id:
            raise ValueError(f"lineage record key differs from {artifact_id}")
        observed_parent = record.get("parent_artifact_id")
        if observed_parent in (None, ""):
            observed_parent = None
        if observed_parent != expected_parent:
            raise ValueError(f"lineage parent changed for {artifact_id}")

    bindings = _mapping(lineage.get("bindings"), label="three-leg lineage bindings")
    for key, actual in (
        ("parent_checkpoint_manifest", pair_checkpoint_path),
        ("chk3_checkpoint_manifest", chk3_checkpoint_path),
    ):
        if key not in bindings:
            raise ValueError(f"three-leg lineage is missing bindings.{key}")
        _binding_matches_file(
            bindings[key],
            manifest_path=lineage_path,
            repo_root=repo_root,
            actual_path=actual,
            label=f"three-leg lineage bindings.{key}",
            sealed=True,
        )
    if "parent_lineage_manifest" not in bindings:
        raise ValueError("three-leg lineage is missing bindings.parent_lineage_manifest")
    _binding_matches_file(
        bindings["parent_lineage_manifest"],
        manifest_path=lineage_path,
        repo_root=repo_root,
        actual_path=pair_lineage_path,
        label="three-leg lineage bindings.parent_lineage_manifest",
        sealed=True,
    )

    for artifact_id, evidence_path in (
        (CHK1_ARTIFACT_ID, pair_exact_path),
        (CHK3_ARTIFACT_ID, chk3_exact_path),
    ):
        record = _mapping(records[artifact_id], label=f"lineage {artifact_id}")
        if record.get("exact_merge_subject_artifact_id") not in (
            artifact_id,
            "chk1-clean-v2-lr1e6-cp200" if artifact_id == CHK1_ARTIFACT_ID else artifact_id,
        ):
            raise ValueError(f"lineage exact-merge subject changed for {artifact_id}")
        _binding_matches_file(
            record.get("exact_merge_evidence"),
            manifest_path=lineage_path,
            repo_root=repo_root,
            actual_path=evidence_path,
            label=f"lineage {artifact_id}.exact_merge_evidence",
            sealed=True,
        )
    return lineage, {
        "manifest": pair_eval._file_binding(lineage_path, lineage),
        "parents": expected_parents,
        "contrasts": [
            {"baseline": BASE_ARTIFACT_ID, "candidate": CHK1_ARTIFACT_ID},
            {"baseline": CHK1_ARTIFACT_ID, "candidate": CHK3_ARTIFACT_ID},
        ],
    }


def _validate_chk3_generation(
    *,
    generation_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    checkpoint_record: Mapping[str, Any],
    prompts_path: Path,
    prompt_rows: Sequence[Mapping[str, Any]],
    config_path: Path,
    config: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path, label="chk3 generation manifest", sealed=True)
    if manifest.get("schema_version") != GENERATION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("chk3 generation manifest is not progress-bound v2")
    if manifest.get("artifact_id") != CHK3_ARTIFACT_ID:
        raise ValueError("chk3 generation manifest artifact_id changed")
    generation_rows = _read_jsonl(generation_path, label="chk3 generations")
    prompt_ids = pair_eval._sample_ids(prompt_rows, label="frozen prompts")
    generation_ids = pair_eval._sample_ids(generation_rows, label="chk3 generations")
    if generation_ids != prompt_ids or len(generation_rows) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("chk3 generation sample order or inventory changed")
    if any(row.get("artifact_id") != CHK3_ARTIFACT_ID for row in generation_rows):
        raise ValueError("chk3 generation row artifact IDs changed")
    output = _mapping(manifest.get("output"), label="chk3 generation output")
    if (
        validate_sha256(output.get("sha256"), label="chk3 output.sha256")
        != sha256_file(generation_path)
        or output.get("row_count") != EXPECTED_SAMPLE_COUNT
        or manifest.get("sample_count") != EXPECTED_SAMPLE_COUNT
    ):
        raise ValueError("chk3 generation output binding changed")
    pair_eval._assert_bound_path(
        output,
        key="path",
        actual_path=generation_path.resolve(),
        repo_root=repo_root,
        manifest_path=manifest_path,
        label="chk3 generation output",
    )
    valid_count = sum(row.get("valid_generation") is True for row in generation_rows)
    if (
        manifest.get("valid_count") != valid_count
        or manifest.get("invalid_count") != EXPECTED_SAMPLE_COUNT - valid_count
    ):
        raise ValueError("chk3 valid/invalid generation counts changed")

    for key, actual, expected_sha in (
        ("checkpoint_manifest", checkpoint_path, sha256_file(checkpoint_path)),
        ("prompts", prompts_path, sha256_file(prompts_path)),
        ("evaluation_config", config_path, sha256_file(config_path)),
    ):
        binding = _mapping(manifest.get(key), label=f"chk3 generation {key}")
        if validate_sha256(binding.get("sha256"), label=f"chk3 {key}.sha256") != expected_sha:
            raise ValueError(f"chk3 generation {key} binding changed")
        if key == "prompts" and binding.get("row_count") != EXPECTED_SAMPLE_COUNT:
            raise ValueError("chk3 generation prompt row count changed")
        pair_eval._assert_bound_path(
            binding,
            key="path",
            actual_path=actual.resolve(),
            repo_root=repo_root,
            manifest_path=manifest_path,
            label=f"chk3 generation {key}",
        )

    for key, hash_key, path_key in (
        ("model_artifact", "model_sha256", "resolved_model_path"),
        ("tokenizer_artifact", "tokenizer_sha256", "resolved_tokenizer_path"),
    ):
        binding = _mapping(manifest.get(key), label=f"chk3 generation {key}")
        if validate_sha256(binding.get("sha256"), label=f"chk3 {key}.sha256") != checkpoint_record[hash_key]:
            raise ValueError(f"chk3 generation {key} differs from checkpoint manifest")
        pair_eval._assert_bound_path(
            binding,
            key="path",
            actual_path=Path(str(checkpoint_record[path_key])),
            repo_root=repo_root,
            manifest_path=manifest_path,
            label=f"chk3 generation {key}",
        )

    generation_config = _mapping(config.get("generation"), label="evaluation config.generation")
    prompt_by_id = {str(row["sample_id"]): row for row in prompt_rows}
    for row in generation_rows:
        sample_id = str(row["sample_id"])
        prompt_row = prompt_by_id[sample_id]
        if (
            row.get("generation_model_sha256") != checkpoint_record["model_sha256"]
            or row.get("generation_tokenizer_sha256") != checkpoint_record["tokenizer_sha256"]
            or row.get("verified_parent_artifact_id") != CHK1_ARTIFACT_ID
            or row.get("prompt_sha256") != prompt_row.get("prompt_sha256")
            or row.get("test_set_sha256") != sha256_file(prompts_path)
            or row.get("prompt_template_sha256") != sha256_file(config_path)
            or row.get("decoding_config_sha256") != sha256_file(config_path)
            or row.get("temperature") != float(generation_config["temperature"])
            or row.get("top_p") != float(generation_config["top_p"])
            or row.get("max_new_tokens") != int(generation_config["max_new_tokens"])
            or row.get("max_model_len") != int(generation_config["max_model_len"])
            or row.get("generation_seed_policy") != str(generation_config["seed_policy"])
            or row.get("generation_seed")
            != derive_row_seed(int(generation_config["base_seed"]), sample_id)
            or not isinstance(row.get("valid_generation"), bool)
        ):
            raise ValueError(f"{CHK3_ARTIFACT_ID}/{sample_id}: generation row provenance changed")
    progress = validate_generation_progress_binding(
        generation_manifest_file=manifest_path,
        generation_output_file=generation_path,
        checkpoint_manifest_file=checkpoint_path,
        prompts_file=prompts_path,
        config_file=config_path,
    )
    return {
        "generation": pair_eval._file_binding(generation_path),
        "manifest": pair_eval._file_binding(manifest_path, manifest),
        "model_sha256": checkpoint_record["model_sha256"],
        "tokenizer_sha256": checkpoint_record["tokenizer_sha256"],
        "sample_count": EXPECTED_SAMPLE_COUNT,
        "progress": progress,
    }


def _partition_inputs(
    generations: Sequence[Path], generation_manifests: Sequence[Path]
) -> dict[str, tuple[Path, Path]]:
    if len(generations) != len(ARTIFACT_IDS):
        raise ValueError(f"Exactly {len(ARTIFACT_IDS)} generation files are required")
    if len(generation_manifests) != len(ARTIFACT_IDS):
        raise ValueError(f"Exactly {len(ARTIFACT_IDS)} generation manifests are required")
    result: dict[str, tuple[Path, Path]] = {}
    for generation, manifest_path in zip(generations, generation_manifests, strict=True):
        manifest = _read_json(manifest_path, label="generation manifest", sealed=True)
        artifact_id = str(manifest.get("artifact_id") or "").strip()
        if artifact_id not in ARTIFACT_IDS or artifact_id in result:
            raise ValueError(f"Unexpected or duplicate generation artifact: {artifact_id}")
        result[artifact_id] = (generation, manifest_path)
    if set(result) != set(ARTIFACT_IDS):
        raise ValueError(f"Generation inventory is incomplete: {sorted(result)}")
    return result


def validate_inputs(
    *,
    generations: Sequence[Path],
    generation_manifests: Sequence[Path],
    prompts: Path,
    references: Path,
    config: Path,
    test_manifest: Path,
    pair_checkpoint_manifest: Path,
    pair_lineage_manifest: Path,
    semantic_manifest: Path,
    pair_exact_merge_evidence: Path,
    chk3_checkpoint_manifest: Path,
    three_leg_lineage_manifest: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], list[Path]]:
    """Deep-validate the asymmetric three-artifact matrix before scoring."""

    root = repo_root.expanduser().resolve()
    inputs = _partition_inputs(generations, generation_manifests)
    pair_generations = [inputs[artifact_id][0] for artifact_id in PAIR_ARTIFACT_IDS]
    pair_manifests = [inputs[artifact_id][1] for artifact_id in PAIR_ARTIFACT_IDS]
    config_payload, test, semantic, pair_validation = pair_eval.validate_inputs(
        generations=pair_generations,
        generation_manifests=pair_manifests,
        prompts=prompts,
        references=references,
        config=config,
        test_manifest=test_manifest,
        checkpoint_manifest=pair_checkpoint_manifest,
        lineage_manifest=pair_lineage_manifest,
        semantic_manifest=semantic_manifest,
        exact_merge_evidence=pair_exact_merge_evidence,
        repo_root=root,
    )

    pair_checkpoint = _read_json(
        pair_checkpoint_manifest, label="pair checkpoint manifest", sealed=True
    )
    pair_records = pair_eval._checkpoint_records(
        pair_checkpoint, manifest_path=pair_checkpoint_manifest, repo_root=root
    )
    chk3_record, chk3_checkpoint_validation = _validate_chk3_checkpoint_manifest(
        checkpoint_path=chk3_checkpoint_manifest.resolve(),
        parent_checkpoint_path=pair_checkpoint_manifest.resolve(),
        config_path=config.resolve(),
        test_manifest_path=test_manifest.resolve(),
        pair_records=pair_records,
        repo_root=root,
    )
    chk3_exact_path = Path(
        str(chk3_checkpoint_validation["exact_merge_evidence"]["path"])
    )
    _, lineage_validation = _validate_combined_lineage(
        lineage_path=three_leg_lineage_manifest.resolve(),
        pair_lineage_path=pair_lineage_manifest.resolve(),
        pair_checkpoint_path=pair_checkpoint_manifest.resolve(),
        chk3_checkpoint_path=chk3_checkpoint_manifest.resolve(),
        pair_exact_path=pair_exact_merge_evidence.resolve(),
        chk3_exact_path=chk3_exact_path,
        repo_root=root,
    )
    prompt_rows = _read_jsonl(prompts, label="frozen prompts")
    chk3_generation, chk3_generation_manifest = inputs[CHK3_ARTIFACT_ID]
    chk3_generation_validation = _validate_chk3_generation(
        generation_path=chk3_generation.resolve(),
        manifest_path=chk3_generation_manifest.resolve(),
        checkpoint_path=chk3_checkpoint_manifest.resolve(),
        checkpoint_record=chk3_record,
        prompts_path=prompts.resolve(),
        prompt_rows=prompt_rows,
        config_path=config.resolve(),
        config=config_payload,
        repo_root=root,
    )

    pair_generations_validation = _mapping(
        pair_validation.get("generations"), label="pair validation generations"
    )
    generation_validation = {
        artifact_id: pair_generations_validation[artifact_id]
        for artifact_id in PAIR_ARTIFACT_IDS
    }
    generation_validation[CHK3_ARTIFACT_ID] = chk3_generation_validation
    validation = seal_manifest(
        {
            "schema_version": VALIDATION_RECEIPT_SCHEMA_VERSION,
            "status": "validated",
            "complete": True,
            "artifact_ids": list(ARTIFACT_IDS),
            "evaluation_scope": "standalone chk0 -> chk1 -> direct chk3-cp250 comparison",
            "evaluation_only": True,
            "promotable_to_canonical_dag": False,
            "frozen_test": pair_validation["frozen_test"],
            "pair_evidence_chain": {
                "input_validation_payload_sha256": pair_validation["integrity"][
                    "payload_sha256"
                ],
                "checkpoint_manifest": pair_validation["checkpoint_manifest"],
                "lineage_manifest": pair_validation["lineage_manifest"],
                "exact_merge_evidence": pair_validation["exact_merge_evidence"],
            },
            "chk3_evidence_chain": chk3_checkpoint_validation,
            "three_leg_lineage": lineage_validation,
            "semantic_manifest": pair_validation["semantic_manifest"],
            "generations": generation_validation,
            "core_evaluator": {
                "module": "jobs.eval.eval_checkpoint_generation",
                "require_formal_manifests": False,
                "expected_artifact_count": 3,
                "required_artifact_ids": list(ARTIFACT_IDS),
                "paired_unit": "sample_id",
                "cluster_unit": "meeting_id",
                "multiple_testing": "Holm",
            },
        }
    )
    ordered_generations = [inputs[artifact_id][0].resolve() for artifact_id in ARTIFACT_IDS]
    return config_payload, test, semantic, validation, ordered_generations


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _read_json(path, label="input validation receipt", sealed=True)
        if existing != payload:
            raise FileExistsError(
                f"Immutable input validation receipt differs from this run: {path}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_validation_receipt(
    *,
    output_dir: Path,
    validation: Mapping[str, Any],
    scoring_policy: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
    result: Mapping[str, Any],
) -> Path:
    audit_path = Path(str(result["paths"]["audit"])).resolve()
    audit = _read_json(audit_path, label="core evaluation audit", sealed=True)
    payload = dict(validation)
    payload.pop("integrity", None)
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "scoring_policy": scoring_policy,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "core_evaluation": {
                "audit_path": str(audit_path),
                "audit_sha256": sha256_file(audit_path),
                "audit_payload_sha256": _mapping(
                    audit.get("integrity"), label="core evaluation audit.integrity"
                ).get("payload_sha256"),
                "run_spec_sha256": audit.get("run_spec_sha256"),
                "status": audit.get("status"),
            },
        }
    )
    receipt = seal_manifest(payload)
    path = output_dir.resolve() / "input_validation.json"
    _atomic_write_json(path, receipt)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", action="append", required=True, type=Path)
    parser.add_argument("--generation-manifest", action="append", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--pair-checkpoint-manifest", required=True, type=Path)
    parser.add_argument("--pair-lineage-manifest", required=True, type=Path)
    parser.add_argument("--semantic-manifest", required=True, type=Path)
    parser.add_argument("--pair-exact-merge-evidence", required=True, type=Path)
    parser.add_argument("--chk3-checkpoint-manifest", required=True, type=Path)
    parser.add_argument("--three-leg-lineage-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--semantic-device", default="cuda:0")
    parser.add_argument("--semantic-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_729)
    parser.add_argument(
        "--scoring-policy", choices=SCORING_POLICIES, default=STRICT_SCORING_POLICY
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repo_root.expanduser().resolve()
    config, test, semantic, validation, ordered_generations = validate_inputs(
        generations=args.generations,
        generation_manifests=args.generation_manifest,
        prompts=args.prompts,
        references=args.references,
        config=args.config,
        test_manifest=args.test_manifest,
        pair_checkpoint_manifest=args.pair_checkpoint_manifest,
        pair_lineage_manifest=args.pair_lineage_manifest,
        semantic_manifest=args.semantic_manifest,
        pair_exact_merge_evidence=args.pair_exact_merge_evidence,
        chk3_checkpoint_manifest=args.chk3_checkpoint_manifest,
        three_leg_lineage_manifest=args.three_leg_lineage_manifest,
        repo_root=root,
    )
    models = _mapping(semantic.get("models"), label="semantic manifest.models")
    bert = _mapping(models.get("bertscore"), label="semantic models.bertscore")
    mpnet = _mapping(models.get("embedding_cosine"), label="semantic models.embedding_cosine")
    bert_path = pair_eval._resolve_local_model_path(
        bert.get("local_path"),
        root=root,
        manifest_path=args.semantic_manifest,
        label="BERTScore model",
    )
    mpnet_path = pair_eval._resolve_local_model_path(
        mpnet.get("local_path"),
        root=root,
        manifest_path=args.semantic_manifest,
        label="MPNet model",
    )
    scorers = [
        BERTScoreBackend(
            bert_path,
            str(bert.get("directory_sha256") or ""),
            num_layers=int(bert.get("num_layers")),
            batch_size=args.semantic_batch_size,
            device=args.semantic_device,
        ),
        MPNetCosineBackend(
            mpnet_path,
            str(mpnet.get("directory_sha256") or ""),
            batch_size=args.semantic_batch_size,
            device=args.semantic_device,
        ),
    ]
    subsets = {
        "all_11_meetings": None,
        "prospective_only_9_meetings": test.get("prospective_only_meeting_dates", []),
    }
    result = run_checkpoint_generation_evaluation(
        ordered_generations,
        args.references,
        args.prompts,
        args.output_dir,
        require_formal_manifests=False,
        semantic_scorers=scorers,
        lineage_manifest_path=args.three_leg_lineage_manifest,
        expected_artifact_count=3,
        required_artifact_ids=ARTIFACT_IDS,
        missing_policy="error",
        require_provenance=True,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        scoring_policy=args.scoring_policy,
        evaluation_subsets=subsets,
    )
    receipt_path = _write_validation_receipt(
        output_dir=args.output_dir,
        validation=validation,
        scoring_policy=args.scoring_policy,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        result=result,
    )
    print(
        json.dumps(
            {
                "status": result["audit"]["status"],
                "scoring_policy": args.scoring_policy,
                "reused_existing": result["reused_existing"],
                "paths": {**result["paths"], "input_validation": str(receipt_path)},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
