"""Score the current chk0/chk1 pair on the frozen Chapter 2 test.

The historical four-artifact formal evaluator is sealed to different model
artifacts.  This versioned entry point therefore validates the current two-model
matrix itself and then delegates row scoring, meeting-cluster aggregation,
paired contrasts, bootstrap intervals, and Holm correction to
``jobs.eval.eval_checkpoint_generation``.

No generation is performed by this module.
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
from jobs.eval.eval_checkpoint_generation import (
    BERTScoreBackend,
    MPNetCosineBackend,
    SCORING_POLICIES,
    STRICT_SCORING_POLICY,
    run_checkpoint_generation_evaluation,
)
from open_r1.provenance import sha256_file, validate_sha256
from open_r1.validator.loo_generation_spec import (
    derive_row_seed,
    seal_manifest,
    validate_manifest_integrity,
)


SCHEMA_VERSION = "retrain-v2-chk0-chk1-generation-evaluation-v1"
VALIDATION_RECEIPT_SCHEMA_VERSION = (
    "retrain-v2-chk0-chk1-generation-input-validation-v1"
)
CONFIG_SCHEMA_VERSION = "checkpoint-generation-eval-config-v1"
TEST_MANIFEST_SCHEMA_VERSION = "checkpoint-eval-test-manifest-v1"
CHECKPOINT_MANIFEST_SCHEMA_VERSION = "retrain-v2-checkpoint-eval-manifest-v1"
LINEAGE_MANIFEST_SCHEMA_VERSION = "retrain-v2-eval-lineage-v1"
SEMANTIC_MANIFEST_SCHEMA_VERSION = "checkpoint-eval-semantic-model-manifest-v1"
EXACT_MERGE_SCHEMA_VERSION = "lora-merge-lineage-evidence-v1"
EXACT_MERGE_ALGORITHM_VERSION = "peft-lora-fp32-exact-v1"
EXACT_MERGE_CONCLUSION = "exact_base_plus_adapter_merge_verified"
EXPECTED_SAMPLE_COUNT = 33
EXPECTED_MEETING_COUNT = 11
EXPECTED_PROSPECTIVE_MEETING_COUNT = 9
EXPECTED_SECTION_COUNT = 3
BASE_ARTIFACT_ID = "eval-chk0-base"
CHK1_ARTIFACT_ID = "eval-chk1-clean-v2-lr1e6-cp200"
ARTIFACT_IDS = (BASE_ARTIFACT_ID, CHK1_ARTIFACT_ID)


def _read_json(path: Path, *, label: str, sealed: bool = False) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if sealed:
        validate_manifest_integrity(value)
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{label} row {line_number} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{label} row {line_number} is not an object")
            rows.append(value)
    return rows


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if str(value).strip() != str(parsed):
        raise ValueError(f"{label} must be an integer")
    return parsed


def _resolve_bound_path(
    value: object,
    *,
    repo_root: Path,
    manifest_path: Path,
    label: str,
    require_directory: bool | None = None,
) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} path is empty")
    raw = Path(text).expanduser()
    candidates = (
        [raw] if raw.is_absolute() else [repo_root / raw, manifest_path.parent / raw]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.exists():
            continue
        if require_directory is True and not resolved.is_dir():
            continue
        if require_directory is False and not resolved.is_file():
            continue
        return resolved
    expected_kind = (
        "directory"
        if require_directory is True
        else "file" if require_directory is False else "path"
    )
    raise FileNotFoundError(
        f"{label} does not resolve to an existing {expected_kind}: {value}"
    )


def _assert_bound_path(
    record: Mapping[str, Any],
    *,
    key: str,
    actual_path: Path,
    repo_root: Path,
    manifest_path: Path,
    label: str,
) -> None:
    resolved = _resolve_bound_path(
        record.get(key),
        repo_root=repo_root,
        manifest_path=manifest_path,
        label=label,
        require_directory=actual_path.is_dir(),
    )
    if resolved != actual_path.resolve():
        raise ValueError(
            f"{label} binds a different path: expected={actual_path.resolve()}, "
            f"observed={resolved}"
        )


def _sample_ids(rows: Sequence[Mapping[str, Any]], *, label: str) -> list[str]:
    values = [str(row.get("sample_id") or "").strip() for row in rows]
    if any(not value for value in values):
        raise ValueError(f"{label} contains an empty sample_id")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate sample_id values")
    return values


def _file_binding(
    path: Path, payload: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
    }
    if payload is not None:
        integrity = _mapping(payload.get("integrity"), label=f"{path.name}.integrity")
        record["payload_sha256"] = validate_sha256(
            integrity.get("payload_sha256"),
            label=f"{path.name}.integrity.payload_sha256",
        )
    return record


def _validate_bound_file_record(
    record: Mapping[str, Any],
    *,
    actual_path: Path,
    actual_payload: Mapping[str, Any] | None,
    repo_root: Path,
    manifest_path: Path,
    label: str,
) -> None:
    _assert_bound_path(
        record,
        key="path",
        actual_path=actual_path,
        repo_root=repo_root,
        manifest_path=manifest_path,
        label=label,
    )
    expected_file_sha = validate_sha256(
        record.get("file_sha256", record.get("sha256")),
        label=f"{label}.sha256",
    )
    if expected_file_sha != sha256_file(actual_path):
        raise ValueError(f"{label} file hash changed")
    if record.get("payload_sha256") is not None:
        if actual_payload is None:
            raise ValueError(f"{label} declares a payload hash for an unsealed file")
        observed_payload = _mapping(
            actual_payload.get("integrity"), label=f"{label}.integrity"
        ).get("payload_sha256")
        if validate_sha256(
            record.get("payload_sha256"), label=f"{label}.payload_sha256"
        ) != validate_sha256(
            observed_payload, label=f"{label}.observed_payload_sha256"
        ):
            raise ValueError(f"{label} payload hash changed")


def _checkpoint_records(
    checkpoint: Mapping[str, Any],
    *,
    manifest_path: Path,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    raw_records = checkpoint.get("artifacts")
    if not isinstance(raw_records, list) or any(
        not isinstance(record, Mapping) for record in raw_records
    ):
        raise ValueError("Checkpoint manifest artifacts must be a list of objects")
    records: dict[str, dict[str, Any]] = {}
    for raw_record in raw_records:
        record = dict(raw_record)
        artifact_id = str(record.get("artifact_id") or "").strip()
        if not artifact_id or artifact_id in records:
            raise ValueError(
                "Checkpoint manifest has an empty or duplicate artifact_id"
            )
        records[artifact_id] = record
    if set(records) != set(ARTIFACT_IDS):
        raise ValueError(
            "Checkpoint manifest artifact inventory differs from the current pair: "
            f"{sorted(records)}"
        )
    expected_design_ids = {BASE_ARTIFACT_ID: "chk-0", CHK1_ARTIFACT_ID: "chk-1"}

    for artifact_id, record in records.items():
        if record.get("usable_for_evaluation") is not True:
            raise ValueError(f"{artifact_id}: usable_for_evaluation must be true")
        if record.get("design_checkpoint_id") != expected_design_ids[artifact_id]:
            raise ValueError(f"{artifact_id}: design_checkpoint_id changed")
        record["model_sha256"] = validate_sha256(
            record.get("model_sha256"), label=f"{artifact_id}.model_sha256"
        )
        record["tokenizer_sha256"] = validate_sha256(
            record.get("tokenizer_sha256"),
            label=f"{artifact_id}.tokenizer_sha256",
        )
        record["resolved_model_path"] = str(
            _resolve_bound_path(
                record.get("model_path"),
                repo_root=repo_root,
                manifest_path=manifest_path,
                label=f"{artifact_id}.model_path",
                require_directory=True,
            )
        )
        record["resolved_tokenizer_path"] = str(
            _resolve_bound_path(
                record.get("tokenizer_path"),
                repo_root=repo_root,
                manifest_path=manifest_path,
                label=f"{artifact_id}.tokenizer_path",
                require_directory=True,
            )
        )

    base_parent = records[BASE_ARTIFACT_ID].get("verified_parent_artifact_id")
    if base_parent not in (None, ""):
        raise ValueError(f"{BASE_ARTIFACT_ID} must be the lineage root")
    if (
        str(records[CHK1_ARTIFACT_ID].get("verified_parent_artifact_id") or "")
        != BASE_ARTIFACT_ID
    ):
        raise ValueError(
            f"{CHK1_ARTIFACT_ID} must declare {BASE_ARTIFACT_ID} as parent"
        )
    if records[BASE_ARTIFACT_ID].get("intended_parent_id") not in (None, ""):
        raise ValueError(f"{BASE_ARTIFACT_ID} intended parent must be empty")
    if records[CHK1_ARTIFACT_ID].get("intended_parent_id") != "chk-0":
        raise ValueError(f"{CHK1_ARTIFACT_ID} intended parent must be chk-0")
    return records


def _lineage_records(lineage: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw_records = lineage.get("checkpoints")
    if not isinstance(raw_records, Mapping) or any(
        not isinstance(record, Mapping) for record in raw_records.values()
    ):
        raise ValueError("Lineage manifest checkpoints must be an object of objects")
    records = {str(key): value for key, value in raw_records.items()}
    if set(records) != set(ARTIFACT_IDS):
        raise ValueError(
            "Lineage manifest artifact inventory differs from the current pair: "
            f"{sorted(records)}"
        )
    for artifact_id, record in records.items():
        if record.get("artifact_id") != artifact_id:
            raise ValueError(
                f"Lineage record key and artifact_id differ: {artifact_id}"
            )
    if records[BASE_ARTIFACT_ID].get("parent_artifact_id") not in (None, ""):
        raise ValueError(f"{BASE_ARTIFACT_ID} must be the lineage root")
    if (
        str(records[CHK1_ARTIFACT_ID].get("parent_artifact_id") or "")
        != BASE_ARTIFACT_ID
    ):
        raise ValueError(
            f"{CHK1_ARTIFACT_ID} lineage parent must be {BASE_ARTIFACT_ID}"
        )
    return records


def _validate_exact_merge_evidence(
    *,
    evidence: Mapping[str, Any],
    evidence_path: Path,
    lineage_record: Mapping[str, Any],
    lineage_path: Path,
    checkpoint_records: Mapping[str, Mapping[str, Any]],
    checkpoint_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    if evidence.get("schema_version") != EXACT_MERGE_SCHEMA_VERSION:
        raise ValueError("Exact-merge evidence has unsupported schema_version")
    if evidence.get("algorithm_version") != EXACT_MERGE_ALGORITHM_VERSION:
        raise ValueError("Exact-merge evidence algorithm_version changed")
    if evidence.get("conclusion") != EXACT_MERGE_CONCLUSION:
        raise ValueError("Exact-merge evidence conclusion is not verified")

    evidence_binding = lineage_record.get("exact_merge_evidence")
    binding_record = (
        evidence_binding
        if isinstance(evidence_binding, Mapping)
        else {"path": evidence_binding}
    )
    _assert_bound_path(
        binding_record,
        key="path",
        actual_path=evidence_path,
        repo_root=repo_root,
        manifest_path=lineage_path,
        label="lineage exact_merge_evidence",
    )
    observed_evidence_sha = sha256_file(evidence_path)
    if (
        binding_record.get("file_sha256") is not None
        or binding_record.get("sha256") is not None
    ):
        expected = validate_sha256(
            binding_record.get("file_sha256", binding_record.get("sha256")),
            label="lineage exact_merge_evidence file_sha256",
        )
        if expected != observed_evidence_sha:
            raise ValueError("Lineage exact-merge evidence file hash changed")
    if binding_record.get("payload_sha256") is not None:
        observed_payload = _mapping(
            evidence.get("integrity"), label="exact-merge evidence.integrity"
        ).get("payload_sha256")
        if validate_sha256(
            binding_record.get("payload_sha256"),
            label="lineage exact_merge_evidence payload_sha256",
        ) != validate_sha256(
            observed_payload, label="exact-merge evidence integrity.payload_sha256"
        ):
            raise ValueError("Lineage exact-merge evidence payload hash changed")

    declared_subject = str(
        lineage_record.get("exact_merge_subject_artifact_id") or ""
    ).strip()
    observed_subject = str(evidence.get("subject_artifact_id") or "").strip()
    if not observed_subject:
        raise ValueError("Exact-merge evidence subject_artifact_id is empty")
    if declared_subject and declared_subject != observed_subject:
        raise ValueError("Lineage and exact-merge evidence subject IDs differ")

    sources = _mapping(evidence.get("sources"), label="exact-merge evidence.sources")
    source_plan = {
        "base_model": (BASE_ARTIFACT_ID, "model"),
        "merged_model": (CHK1_ARTIFACT_ID, "model"),
    }
    checked_sources: dict[str, Any] = {}
    for source_name, (artifact_id, _) in source_plan.items():
        source = _mapping(
            sources.get(source_name),
            label=f"exact-merge evidence.sources.{source_name}",
        )
        checkpoint_record = checkpoint_records[artifact_id]
        source_hash = validate_sha256(
            source.get("sha256"), label=f"exact-merge {source_name}.sha256"
        )
        if source_hash != checkpoint_record["model_sha256"]:
            raise ValueError(
                f"Exact-merge {source_name} hash differs from checkpoint manifest"
            )
        source_path = _resolve_bound_path(
            source.get("path"),
            repo_root=repo_root,
            manifest_path=evidence_path,
            label=f"exact-merge evidence.sources.{source_name}.path",
            require_directory=True,
        )
        if source_path != Path(str(checkpoint_record["resolved_model_path"])):
            raise ValueError(
                f"Exact-merge {source_name} path differs from checkpoint manifest"
            )
        checked_sources[source_name] = {
            "path": str(source_path),
            "sha256": source_hash,
        }

    adapter_source = _mapping(
        sources.get("adapter"), label="exact-merge evidence.sources.adapter"
    )
    adapter_hash = validate_sha256(
        adapter_source.get("sha256"), label="exact-merge adapter.sha256"
    )
    adapter_path = _resolve_bound_path(
        adapter_source.get("path"),
        repo_root=repo_root,
        manifest_path=evidence_path,
        label="exact-merge evidence.sources.adapter.path",
        require_directory=True,
    )
    chk1_record = checkpoint_records[CHK1_ARTIFACT_ID]
    if chk1_record.get("adapter_sha256") is not None:
        if adapter_hash != validate_sha256(
            chk1_record.get("adapter_sha256"),
            label=f"{CHK1_ARTIFACT_ID}.adapter_sha256",
        ):
            raise ValueError(
                "Exact-merge adapter hash differs from checkpoint manifest"
            )
    if chk1_record.get("adapter_path") is not None:
        expected_adapter = _resolve_bound_path(
            chk1_record.get("adapter_path"),
            repo_root=repo_root,
            manifest_path=checkpoint_path,
            label=f"{CHK1_ARTIFACT_ID}.adapter_path",
            require_directory=True,
        )
        if adapter_path != expected_adapter:
            raise ValueError(
                "Exact-merge adapter path differs from checkpoint manifest"
            )
    checked_sources["adapter"] = {"path": str(adapter_path), "sha256": adapter_hash}

    tensor = _mapping(
        evidence.get("tensor_verification"),
        label="exact-merge evidence.tensor_verification",
    )
    model_count = _integer(tensor.get("model_tensor_count"), label="model_tensor_count")
    adapted_count = _integer(
        tensor.get("adapted_model_tensor_count"), label="adapted_model_tensor_count"
    )
    exact_adapted = _integer(
        tensor.get("exact_adapted_model_tensor_count"),
        label="exact_adapted_model_tensor_count",
    )
    unchanged_count = _integer(
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
    if model_count != adapted_count + unchanged_count:
        raise ValueError("Exact-merge tensor counts do not cover every model tensor")
    if exact_adapted != adapted_count or exact_unchanged != unchanged_count:
        raise ValueError(
            "Exact-merge evidence did not verify every model tensor exactly"
        )
    if adapter_tensor_count != 2 * adapted_count:
        raise ValueError("Exact-merge adapter tensor inventory is incomplete")
    if mismatch_count != 0:
        raise ValueError("Exact-merge evidence contains tensor mismatches")

    return {
        "subject_artifact_id": observed_subject,
        "conclusion": EXACT_MERGE_CONCLUSION,
        "algorithm_version": EXACT_MERGE_ALGORITHM_VERSION,
        "sources": checked_sources,
        "tensor_verification": {
            "model_tensor_count": model_count,
            "adapted_model_tensor_count": adapted_count,
            "unchanged_model_tensor_count": unchanged_count,
            "adapter_tensor_count": adapter_tensor_count,
            "mismatch_count": mismatch_count,
            "complete": True,
        },
    }


def _validate_semantic_manifest(
    semantic: Mapping[str, Any],
    *,
    semantic_path: Path,
    config: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    if semantic.get("schema_version") != SEMANTIC_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Semantic model manifest has unsupported schema_version")
    if semantic.get("created_for_evaluation_id") != config.get("evaluation_id"):
        raise ValueError("Semantic manifest and evaluation config IDs differ")
    if semantic.get("network_at_scoring_time") is not False:
        raise ValueError("Semantic scoring must be pinned to local, offline models")
    models = _mapping(semantic.get("models"), label="semantic manifest.models")
    checked: dict[str, Any] = {}
    for key in ("bertscore", "embedding_cosine"):
        record = _mapping(models.get(key), label=f"semantic manifest.models.{key}")
        model_path = _resolve_bound_path(
            record.get("local_path"),
            repo_root=repo_root,
            manifest_path=semantic_path,
            label=f"semantic manifest.models.{key}.local_path",
            require_directory=True,
        )
        directory_hash = validate_sha256(
            record.get("directory_sha256"),
            label=f"semantic manifest.models.{key}.directory_sha256",
        )
        revision = str(record.get("resolved_revision") or "").strip()
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
            raise ValueError(f"semantic manifest.models.{key} revision is not pinned")
        checked[key] = {"path": str(model_path), "sha256": directory_hash}
    if models["embedding_cosine"].get("independent_from_training_reward") is not True:
        raise ValueError(
            "Embedding cosine model must be independent from training reward"
        )
    if (
        _integer(models["bertscore"].get("num_layers"), label="BERTScore num_layers")
        < 1
    ):
        raise ValueError("BERTScore num_layers must be positive")
    return checked


def validate_inputs(
    *,
    generations: Sequence[Path],
    generation_manifests: Sequence[Path],
    prompts: Path,
    references: Path,
    config: Path,
    test_manifest: Path,
    checkpoint_manifest: Path,
    lineage_manifest: Path,
    semantic_manifest: Path,
    exact_merge_evidence: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate the complete two-artifact input matrix before any scoring."""

    if len(generations) != len(ARTIFACT_IDS):
        raise ValueError(f"Exactly {len(ARTIFACT_IDS)} generation files are required")
    if len(generation_manifests) != len(ARTIFACT_IDS):
        raise ValueError(
            f"Exactly {len(ARTIFACT_IDS)} generation manifests are required"
        )

    root = repo_root.expanduser().resolve()
    config_payload = _read_json(config, label="evaluation config")
    test = _read_json(test_manifest, label="test manifest", sealed=True)
    checkpoint = _read_json(
        checkpoint_manifest, label="checkpoint manifest", sealed=True
    )
    lineage = _read_json(lineage_manifest, label="lineage manifest", sealed=True)
    semantic = _read_json(
        semantic_manifest, label="semantic model manifest", sealed=True
    )
    evidence = _read_json(
        exact_merge_evidence, label="exact-merge evidence", sealed=True
    )

    if config_payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Evaluation config has unsupported schema_version")
    if test.get("schema_version") != TEST_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Frozen test manifest has unsupported schema_version")
    if checkpoint.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Checkpoint manifest has unsupported schema_version")
    if lineage.get("schema_version") != LINEAGE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Lineage manifest has unsupported schema_version")

    if config_payload.get("evaluation_id") != test.get("evaluation_id"):
        raise ValueError("Evaluation config and frozen test manifest IDs differ")
    if test.get("sample_count") != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"Frozen test must contain {EXPECTED_SAMPLE_COUNT} samples")
    if test.get("section_count") != EXPECTED_SECTION_COUNT:
        raise ValueError(f"Frozen test must contain {EXPECTED_SECTION_COUNT} sections")
    meeting_dates = test.get("meeting_dates")
    prospective_dates = test.get("prospective_only_meeting_dates")
    if (
        not isinstance(meeting_dates, list)
        or len(meeting_dates) != EXPECTED_MEETING_COUNT
    ):
        raise ValueError(f"Frozen test must contain {EXPECTED_MEETING_COUNT} meetings")
    if (
        not isinstance(prospective_dates, list)
        or len(prospective_dates) != EXPECTED_PROSPECTIVE_MEETING_COUNT
        or not set(map(str, prospective_dates)) < set(map(str, meeting_dates))
    ):
        raise ValueError("Frozen prospective-only meeting subset changed")
    if test.get("reference_in_prompt") is not False:
        raise ValueError("Frozen test must be reference-free")
    if test.get("secondary_llm_summary") is not False:
        raise ValueError("Frozen test cannot contain a secondary LLM summary")

    config_input = _mapping(
        _mapping(test.get("inputs"), label="test manifest.inputs").get("config"),
        label="test manifest.inputs.config",
    )
    if validate_sha256(
        config_input.get("sha256"), label="test manifest config sha256"
    ) != sha256_file(config):
        raise ValueError("Evaluation config no longer matches the frozen test manifest")

    checkpoint_bindings = _mapping(
        checkpoint.get("bindings"), label="checkpoint manifest.bindings"
    )
    _validate_bound_file_record(
        _mapping(
            checkpoint_bindings.get("evaluation_config"),
            label="checkpoint bindings.evaluation_config",
        ),
        actual_path=config.resolve(),
        actual_payload=None,
        repo_root=root,
        manifest_path=checkpoint_manifest,
        label="checkpoint bindings.evaluation_config",
    )
    _validate_bound_file_record(
        _mapping(
            checkpoint_bindings.get("test_manifest"),
            label="checkpoint bindings.test_manifest",
        ),
        actual_path=test_manifest.resolve(),
        actual_payload=test,
        repo_root=root,
        manifest_path=checkpoint_manifest,
        label="checkpoint bindings.test_manifest",
    )
    _validate_bound_file_record(
        _mapping(
            checkpoint_bindings.get("exact_merge_evidence"),
            label="checkpoint bindings.exact_merge_evidence",
        ),
        actual_path=exact_merge_evidence.resolve(),
        actual_payload=evidence,
        repo_root=root,
        manifest_path=checkpoint_manifest,
        label="checkpoint bindings.exact_merge_evidence",
    )

    prompt_rows = _read_jsonl(prompts, label="frozen prompts")
    reference_rows = _read_jsonl(references, label="frozen references")
    prompt_ids = _sample_ids(prompt_rows, label="frozen prompts")
    reference_ids = _sample_ids(reference_rows, label="frozen references")
    if (
        len(prompt_rows) != EXPECTED_SAMPLE_COUNT
        or len(reference_rows) != EXPECTED_SAMPLE_COUNT
    ):
        raise ValueError(
            "Frozen prompts and references must each contain exactly 33 rows"
        )
    if prompt_ids != reference_ids:
        raise ValueError("Frozen prompt/reference sample order or inventory changed")
    outputs = _mapping(test.get("outputs"), label="test manifest.outputs")
    for name, path, rows in (
        ("prompts", prompts, prompt_rows),
        ("references", references, reference_rows),
    ):
        record = _mapping(outputs.get(name), label=f"test manifest.outputs.{name}")
        if validate_sha256(
            record.get("sha256"), label=f"test manifest {name} sha256"
        ) != sha256_file(path) or record.get("row_count") != len(rows):
            raise ValueError(f"Frozen {name} no longer matches the test manifest")

    checkpoint_records = _checkpoint_records(
        checkpoint, manifest_path=checkpoint_manifest, repo_root=root
    )
    lineage_records = _lineage_records(lineage)
    exact_validation = _validate_exact_merge_evidence(
        evidence=evidence,
        evidence_path=exact_merge_evidence.resolve(),
        lineage_record=lineage_records[CHK1_ARTIFACT_ID],
        lineage_path=lineage_manifest,
        checkpoint_records=checkpoint_records,
        checkpoint_path=checkpoint_manifest,
        repo_root=root,
    )
    semantic_validation = _validate_semantic_manifest(
        semantic,
        semantic_path=semantic_manifest,
        config=config_payload,
        repo_root=root,
    )

    checkpoint_sha = sha256_file(checkpoint_manifest)
    config_sha = sha256_file(config)
    prompt_sha = sha256_file(prompts)
    generation_bindings: dict[str, Any] = {}
    seen_ids: set[str] = set()
    for generation_path, manifest_path in zip(
        generations, generation_manifests, strict=True
    ):
        manifest = _read_json(manifest_path, label="generation manifest", sealed=True)
        if manifest.get("schema_version") != GENERATION_MANIFEST_SCHEMA_VERSION:
            raise ValueError("Generation manifest has unsupported schema_version")
        artifact_id = str(manifest.get("artifact_id") or "").strip()
        if artifact_id not in ARTIFACT_IDS or artifact_id in seen_ids:
            raise ValueError(
                f"Unexpected or duplicate generation artifact: {artifact_id}"
            )
        seen_ids.add(artifact_id)
        checkpoint_record = checkpoint_records[artifact_id]
        output = _mapping(manifest.get("output"), label=f"{artifact_id}.output")
        generation_rows = _read_jsonl(
            generation_path, label=f"{artifact_id} generations"
        )
        generation_ids = _sample_ids(
            generation_rows, label=f"{artifact_id} generations"
        )
        if generation_ids != prompt_ids:
            raise ValueError(
                f"{artifact_id}: generation sample order or inventory changed"
            )
        if any(
            str(row.get("artifact_id") or "").strip() != artifact_id
            for row in generation_rows
        ):
            raise ValueError(f"{artifact_id}: generation row artifact IDs differ")
        if (
            validate_sha256(output.get("sha256"), label=f"{artifact_id}.output.sha256")
            != sha256_file(generation_path)
            or output.get("row_count") != len(generation_rows)
            or len(generation_rows) != EXPECTED_SAMPLE_COUNT
            or manifest.get("sample_count") != EXPECTED_SAMPLE_COUNT
        ):
            raise ValueError(f"{artifact_id}: generation output binding changed")
        _assert_bound_path(
            output,
            key="path",
            actual_path=generation_path.resolve(),
            repo_root=root,
            manifest_path=manifest_path,
            label=f"{artifact_id}.output",
        )
        actual_valid_count = sum(
            row.get("valid_generation") is True for row in generation_rows
        )
        if (
            manifest.get("valid_count") != actual_valid_count
            or manifest.get("invalid_count")
            != EXPECTED_SAMPLE_COUNT - actual_valid_count
        ):
            raise ValueError(f"{artifact_id}: valid/invalid generation counts changed")

        checkpoint_binding = _mapping(
            manifest.get("checkpoint_manifest"),
            label=f"{artifact_id}.checkpoint_manifest",
        )
        if (
            validate_sha256(
                checkpoint_binding.get("sha256"),
                label=f"{artifact_id}.checkpoint_manifest.sha256",
            )
            != checkpoint_sha
        ):
            raise ValueError(f"{artifact_id}: checkpoint-manifest binding changed")
        _assert_bound_path(
            checkpoint_binding,
            key="path",
            actual_path=checkpoint_manifest.resolve(),
            repo_root=root,
            manifest_path=manifest_path,
            label=f"{artifact_id}.checkpoint_manifest",
        )
        prompt_binding = _mapping(
            manifest.get("prompts"), label=f"{artifact_id}.prompts"
        )
        if (
            validate_sha256(
                prompt_binding.get("sha256"), label=f"{artifact_id}.prompts.sha256"
            )
            != prompt_sha
            or prompt_binding.get("row_count") != EXPECTED_SAMPLE_COUNT
        ):
            raise ValueError(f"{artifact_id}: prompt binding changed")
        _assert_bound_path(
            prompt_binding,
            key="path",
            actual_path=prompts.resolve(),
            repo_root=root,
            manifest_path=manifest_path,
            label=f"{artifact_id}.prompts",
        )
        config_binding = _mapping(
            manifest.get("evaluation_config"),
            label=f"{artifact_id}.evaluation_config",
        )
        if (
            validate_sha256(
                config_binding.get("sha256"),
                label=f"{artifact_id}.evaluation_config.sha256",
            )
            != config_sha
        ):
            raise ValueError(f"{artifact_id}: evaluation config binding changed")
        _assert_bound_path(
            config_binding,
            key="path",
            actual_path=config.resolve(),
            repo_root=root,
            manifest_path=manifest_path,
            label=f"{artifact_id}.evaluation_config",
        )
        for artifact_key, checkpoint_hash_key, checkpoint_path_key in (
            ("model_artifact", "model_sha256", "resolved_model_path"),
            ("tokenizer_artifact", "tokenizer_sha256", "resolved_tokenizer_path"),
        ):
            artifact_record = _mapping(
                manifest.get(artifact_key), label=f"{artifact_id}.{artifact_key}"
            )
            if (
                validate_sha256(
                    artifact_record.get("sha256"),
                    label=f"{artifact_id}.{artifact_key}.sha256",
                )
                != checkpoint_record[checkpoint_hash_key]
            ):
                raise ValueError(
                    f"{artifact_id}: {artifact_key} differs from checkpoint manifest"
                )
            _assert_bound_path(
                artifact_record,
                key="path",
                actual_path=Path(str(checkpoint_record[checkpoint_path_key])),
                repo_root=root,
                manifest_path=manifest_path,
                label=f"{artifact_id}.{artifact_key}",
            )
        generation_config = _mapping(
            config_payload.get("generation"), label="evaluation config.generation"
        )
        expected_parent = None if artifact_id == BASE_ARTIFACT_ID else BASE_ARTIFACT_ID
        prompt_by_id = {str(row["sample_id"]): row for row in prompt_rows}
        for row in generation_rows:
            sample_id = str(row["sample_id"])
            prompt_row = prompt_by_id[sample_id]
            if (
                row.get("generation_model_sha256") != checkpoint_record["model_sha256"]
                or row.get("generation_tokenizer_sha256")
                != checkpoint_record["tokenizer_sha256"]
                or row.get("verified_parent_artifact_id") != expected_parent
                or row.get("prompt_sha256") != prompt_row.get("prompt_sha256")
                or row.get("test_set_sha256") != prompt_sha
                or row.get("prompt_template_sha256") != config_sha
                or row.get("decoding_config_sha256") != config_sha
                or row.get("temperature") != float(generation_config["temperature"])
                or row.get("top_p") != float(generation_config["top_p"])
                or row.get("max_new_tokens") != int(generation_config["max_new_tokens"])
                or row.get("max_model_len") != int(generation_config["max_model_len"])
                or row.get("generation_seed_policy")
                != str(generation_config["seed_policy"])
                or row.get("generation_seed")
                != derive_row_seed(int(generation_config["base_seed"]), sample_id)
                or not isinstance(row.get("valid_generation"), bool)
            ):
                raise ValueError(
                    f"{artifact_id}/{sample_id}: generation row provenance changed"
                )
        progress_validation = validate_generation_progress_binding(
            generation_manifest_file=manifest_path,
            generation_output_file=generation_path,
            checkpoint_manifest_file=checkpoint_manifest,
            prompts_file=prompts,
            config_file=config,
        )
        generation_bindings[artifact_id] = {
            "generation": _file_binding(generation_path),
            "manifest": _file_binding(manifest_path, manifest),
            "model_sha256": checkpoint_record["model_sha256"],
            "tokenizer_sha256": checkpoint_record["tokenizer_sha256"],
            "sample_count": len(generation_rows),
            "progress": progress_validation,
        }
    if seen_ids != set(ARTIFACT_IDS):
        raise ValueError(f"Generation inventory is incomplete: {sorted(seen_ids)}")

    validation = seal_manifest(
        {
            "schema_version": VALIDATION_RECEIPT_SCHEMA_VERSION,
            "status": "validated",
            "complete": True,
            "artifact_ids": list(ARTIFACT_IDS),
            "frozen_test": {
                "evaluation_id": test.get("evaluation_id"),
                "sample_count": EXPECTED_SAMPLE_COUNT,
                "meeting_count": EXPECTED_MEETING_COUNT,
                "prospective_meeting_count": EXPECTED_PROSPECTIVE_MEETING_COUNT,
                "section_count": EXPECTED_SECTION_COUNT,
                "reference_free": True,
                "prompts": _file_binding(prompts),
                "references": _file_binding(references),
                "test_manifest": _file_binding(test_manifest, test),
                "evaluation_config": _file_binding(config),
            },
            "checkpoint_manifest": _file_binding(checkpoint_manifest, checkpoint),
            "lineage_manifest": _file_binding(lineage_manifest, lineage),
            "semantic_manifest": {
                **_file_binding(semantic_manifest, semantic),
                "models": semantic_validation,
            },
            "exact_merge_evidence": {
                **_file_binding(exact_merge_evidence, evidence),
                **exact_validation,
            },
            "generations": dict(sorted(generation_bindings.items())),
            "core_evaluator": {
                "module": "jobs.eval.eval_checkpoint_generation",
                "expected_artifact_count": 2,
                "paired_unit": "sample_id",
                "cluster_unit": "meeting_id",
                "multiple_testing": "Holm",
            },
        }
    )
    return config_payload, test, semantic, validation


def _resolve_local_model_path(
    value: object, *, root: Path, manifest_path: Path, label: str
) -> Path:
    return _resolve_bound_path(
        value,
        repo_root=root,
        manifest_path=manifest_path,
        label=label,
        require_directory=True,
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _read_json(path, label="input validation receipt", sealed=True)
        if existing != payload:
            raise FileExistsError(
                f"Immutable input validation receipt differs from this run: {path}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
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
    receipt_payload = dict(validation)
    receipt_payload.pop("integrity", None)
    receipt_payload.update(
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
    receipt = seal_manifest(receipt_payload)
    path = output_dir.resolve() / "input_validation.json"
    _atomic_write_json(path, receipt)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", action="append", required=True, type=Path)
    parser.add_argument(
        "--generation-manifest", action="append", required=True, type=Path
    )
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-manifest", required=True, type=Path)
    parser.add_argument("--lineage-manifest", required=True, type=Path)
    parser.add_argument("--semantic-manifest", required=True, type=Path)
    parser.add_argument("--exact-merge-evidence", required=True, type=Path)
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
    config, test, semantic, validation = validate_inputs(
        generations=args.generations,
        generation_manifests=args.generation_manifest,
        prompts=args.prompts,
        references=args.references,
        config=args.config,
        test_manifest=args.test_manifest,
        checkpoint_manifest=args.checkpoint_manifest,
        lineage_manifest=args.lineage_manifest,
        semantic_manifest=args.semantic_manifest,
        exact_merge_evidence=args.exact_merge_evidence,
        repo_root=root,
    )

    models = _mapping(semantic.get("models"), label="semantic manifest.models")
    bert = _mapping(models.get("bertscore"), label="semantic models.bertscore")
    mpnet = _mapping(
        models.get("embedding_cosine"), label="semantic models.embedding_cosine"
    )
    bert_path = _resolve_local_model_path(
        bert.get("local_path"),
        root=root,
        manifest_path=args.semantic_manifest,
        label="BERTScore model",
    )
    mpnet_path = _resolve_local_model_path(
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
        args.generations,
        args.references,
        args.prompts,
        args.output_dir,
        require_formal_manifests=False,
        semantic_scorers=scorers,
        lineage_manifest_path=args.lineage_manifest,
        expected_artifact_count=2,
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
