"""Audit and recover the historical Chapter 2 checkpoint artifacts.

This module deliberately separates three operations:

* compare an allegedly merged Minutes model with a known decision-GRPO model;
* recover the Minutes branch into a brand-new version directory; and
* register an immutable invalidation record for outputs generated from a bad
  checkpoint.

Every command is read-only by default.  ``--execute`` is required before a
model directory or invalidation record is written.  No operation trains a
model or modifies a source checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from open_r1.provenance import (
    fingerprint_artifact_path,
    sha256_file,
    validate_sha256,
)
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


CHECKPOINT_MANIFEST_SCHEMA_VERSION = "checkpoint-provenance-manifest-v1"
INVALIDATION_SCHEMA_VERSION = "experiment-invalidation-record-v1"

EVAL_BASE_ID = "eval-base"
EVAL_ANALYSIS_SFT_ID = "eval-analysis-sft"
EVAL_LEGACY_GRPO_ID = "eval-legacy-grpo-from-chk0"
EVAL_MINUTES_SFT_ID = "eval-minutes-sft-from-chk1"
CORRUPTED_MINUTES_ID = "legacy-minutes-merged-corrupted"

TOKENIZER_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)
TOKENIZER_CORE_FILES = frozenset(
    {"tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json"}
)
TRAINING_METADATA_FILES = (
    "README.md",
    "all_results.json",
    "eval_results.json",
    "loss_history.jsonl",
    "reward_history.jsonl",
    "train_results.json",
    "trainer_state.json",
    "training_args.bin",
    "training_curve.png",
)
REQUIRED_TRAINING_METADATA_FILES = frozenset(
    {"trainer_state.json", "training_args.bin"}
)
ADAPTER_WEIGHT_FILES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
)

MergeExecutor = Callable[[str, Path, Path], None]


@dataclass(frozen=True)
class RecoveryRequest:
    """Paths required to audit and recover the five manifest artifacts."""

    foundation_model: Path
    analysis_sft_model: Path
    legacy_grpo_model: Path
    legacy_grpo_adapter: Path
    minutes_adapter: Path
    historical_minutes_model: Path
    overwrite_reference_model: Path
    destination: Path
    analysis_sft_adapter: Path | None = None
    legacy_grpo_tokenizer: Path | None = None
    minutes_metadata_source: Path | None = None


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _normalise_directory(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {resolved}")
    return resolved


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _serialise_json(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Create an immutable JSON sidecar, allowing only exact idempotent writes."""

    output = Path(path).expanduser().resolve()
    serialised = _serialise_json(payload)
    if output.is_file():
        if output.read_text(encoding="utf-8") != serialised:
            raise ValueError(f"Refusing to overwrite incompatible record: {output}")
        return sha256_file(output)
    if output.exists():
        raise ValueError(f"Record destination is not a regular file: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialised)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            if not output.is_file() or output.read_text(encoding="utf-8") != serialised:
                raise ValueError(
                    f"Refusing to overwrite incompatible record: {output}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(output)


def _inventory_fingerprint(
    root: Path,
    files: Sequence[Path],
    *,
    kind: str,
) -> dict[str, Any]:
    resolved_root = root.expanduser().resolve()
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    total_bytes = 0
    relative_paths: set[str] = set()

    for candidate in sorted({path.resolve() for path in files}):
        try:
            relative = candidate.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Inventory file is outside its root: {candidate} vs {resolved_root}"
            ) from exc
        if relative in relative_paths:
            continue
        if not candidate.is_file():
            raise FileNotFoundError(f"Inventory file does not exist: {candidate}")
        size = candidate.stat().st_size
        if size <= 0:
            raise ValueError(f"Inventory file is empty: {candidate}")
        file_sha256 = sha256_file(candidate)
        relative_paths.add(relative)
        total_bytes += size
        digest.update(f"{relative}\0{size}\0{file_sha256}\n".encode("utf-8"))
        records.append(
            {
                "relative_path": relative,
                "size": size,
                "sha256": file_sha256,
            }
        )

    if not records:
        raise ValueError(f"{kind} inventory is empty: {resolved_root}")
    return {
        "path": str(resolved_root),
        "kind": kind,
        "sha256": digest.hexdigest(),
        "file_count": len(records),
        "total_bytes": total_bytes,
        "algorithm": (
            "sha256(sorted UTF-8 records '<relative_path>\\0<size>\\0<file_sha256>\\n')"
        ),
        "files": records,
    }


def _model_weight_files(model_path: Path) -> list[Path]:
    model = _normalise_directory(model_path, label="model artifact")
    index_path = model / "model.safetensors.index.json"
    indexed_names: set[str] = set()
    if index_path.is_file():
        index = _read_json_object(index_path, label="model weight index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Model weight index has no weight_map: {index_path}")
        for raw_name in weight_map.values():
            name = str(raw_name or "").strip()
            relative = Path(name)
            if (
                not name
                or relative.is_absolute()
                or ".." in relative.parts
                or len(relative.parts) != 1
            ):
                raise ValueError(
                    f"Unsafe shard name in model weight index: {raw_name!r}"
                )
            indexed_names.add(name)

    discovered = {
        path.name
        for pattern in ("model*.safetensors", "pytorch_model*.bin")
        for path in model.glob(pattern)
        if path.is_file() and not path.name.startswith("adapter_model")
    }
    names = indexed_names | discovered
    if not names:
        raise FileNotFoundError(f"No standalone model weights found in {model}")
    missing = sorted(name for name in indexed_names if not (model / name).is_file())
    if missing:
        raise FileNotFoundError(
            f"Model weight index references missing shards in {model}: {missing}"
        )
    return [model / name for name in sorted(names)]


def fingerprint_weight_payload(model_path: str | Path) -> dict[str, Any]:
    model = _normalise_directory(model_path, label="model artifact")
    return _inventory_fingerprint(
        model,
        _model_weight_files(model),
        kind="model-weight-payload",
    )


def fingerprint_model_payload(model_path: str | Path) -> dict[str, Any]:
    """Fingerprint model weights plus the configuration that interprets them."""

    model = _normalise_directory(model_path, label="model artifact")
    config = model / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Model config does not exist: {config}")
    files = [*_model_weight_files(model), config]
    for name in ("generation_config.json", "model.safetensors.index.json"):
        candidate = model / name
        if candidate.is_file():
            files.append(candidate)
    return _inventory_fingerprint(model, files, kind="model-load-payload")


def _tokenizer_paths(tokenizer_path: Path) -> list[Path]:
    tokenizer = _normalise_directory(tokenizer_path, label="tokenizer artifact")
    files = [
        tokenizer / name for name in TOKENIZER_FILES if (tokenizer / name).is_file()
    ]
    names = {path.name for path in files}
    if "tokenizer_config.json" not in names:
        raise FileNotFoundError(f"Tokenizer config does not exist in {tokenizer}")
    if not (names & TOKENIZER_CORE_FILES):
        raise FileNotFoundError(
            f"Tokenizer has no supported core vocabulary file in {tokenizer}"
        )
    return files


def fingerprint_tokenizer_payload(
    tokenizer_path: str | Path,
) -> dict[str, Any]:
    tokenizer = _normalise_directory(tokenizer_path, label="tokenizer artifact")
    return _inventory_fingerprint(
        tokenizer,
        _tokenizer_paths(tokenizer),
        kind="tokenizer-payload",
    )


def _metadata_paths(metadata_path: Path) -> list[Path]:
    source = _normalise_directory(metadata_path, label="training metadata source")
    files = [
        source / name for name in TRAINING_METADATA_FILES if (source / name).is_file()
    ]
    names = {path.name for path in files}
    missing = sorted(REQUIRED_TRAINING_METADATA_FILES - names)
    if missing:
        raise FileNotFoundError(
            f"Training metadata source is missing required files {missing}: {source}"
        )
    return files


def fingerprint_training_metadata(
    metadata_path: str | Path,
) -> dict[str, Any]:
    source = _normalise_directory(metadata_path, label="training metadata source")
    return _inventory_fingerprint(
        source,
        _metadata_paths(source),
        kind="training-metadata",
    )


def _adapter_weight_paths(adapter_path: Path) -> list[Path]:
    adapter = _normalise_directory(adapter_path, label="adapter artifact")
    files = [
        adapter / name for name in ADAPTER_WEIGHT_FILES if (adapter / name).is_file()
    ]
    if len(files) != 1:
        raise ValueError(
            f"Adapter must contain exactly one supported adapter weight file: {adapter}"
        )
    return files


def fingerprint_adapter_payload(adapter_path: str | Path) -> dict[str, Any]:
    adapter = _normalise_directory(adapter_path, label="adapter artifact")
    config = adapter / "adapter_config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Adapter config does not exist: {config}")
    return _inventory_fingerprint(
        adapter,
        [config, *_adapter_weight_paths(adapter)],
        kind="peft-adapter-payload",
    )


def verify_adapter_parent(
    adapter_path: str | Path,
    expected_parent: str | Path,
) -> dict[str, Any]:
    """Verify a PEFT adapter's declared base against a local parent artifact."""

    adapter = _normalise_directory(adapter_path, label="adapter artifact")
    expected = _normalise_directory(expected_parent, label="expected adapter parent")
    config_path = adapter / "adapter_config.json"
    config = _read_json_object(config_path, label="adapter config")
    declared_text = str(config.get("base_model_name_or_path") or "").strip()
    if not declared_text:
        raise ValueError(
            f"Adapter config has no base_model_name_or_path: {config_path}"
        )
    declared = Path(declared_text).expanduser()
    if declared.is_absolute():
        candidates = [declared.resolve()]
    else:
        candidates = [(Path.cwd() / declared).resolve()]
        candidates.extend(
            (ancestor / declared).resolve() for ancestor in adapter.parents
        )

    matching = [candidate for candidate in candidates if candidate == expected]
    if not matching:
        raise ValueError(
            f"Adapter parent mismatch: declared={declared_text!r}, expected={expected}"
        )
    return {
        "adapter_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "declared_base_model_name_or_path": declared_text,
        "verified_parent_path": str(expected),
        "verification": "resolved_declared_path_matches_expected_parent",
    }


def audit_overwritten_model(
    historical_model: str | Path,
    overwrite_reference_model: str | Path,
) -> dict[str, Any]:
    """Detect a historical directory whose model payload was overwritten."""

    historical = _normalise_directory(historical_model, label="historical merged model")
    reference = _normalise_directory(
        overwrite_reference_model, label="overwrite reference model"
    )
    historical_weights = fingerprint_weight_payload(historical)
    reference_weights = fingerprint_weight_payload(reference)
    historical_payload = fingerprint_model_payload(historical)
    reference_payload = fingerprint_model_payload(reference)
    weight_match = historical_weights["sha256"] == reference_weights["sha256"]
    payload_match = historical_payload["sha256"] == reference_payload["sha256"]
    status = (
        "payload_overwritten_by_reference_artifact"
        if weight_match and payload_match
        else "no_exact_overwrite_match"
    )
    return {
        "status": status,
        "historical_model_path": str(historical),
        "overwrite_reference_model_path": str(reference),
        "weight_payload_exact_match": weight_match,
        "model_load_payload_exact_match": payload_match,
        "historical_weight_payload": historical_weights,
        "reference_weight_payload": reference_weights,
        "historical_model_payload": historical_payload,
        "reference_model_payload": reference_payload,
        "usable_for_evaluation": False if weight_match and payload_match else None,
    }


def _fingerprint_cached(
    path: Path,
    cache: dict[Path, dict[str, Any]],
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if resolved not in cache:
        cache[resolved] = fingerprint_artifact_path(resolved)
    return cache[resolved]


def _artifact_record(
    *,
    artifact_id: str,
    design_checkpoint_id: str,
    intended_parent_id: str | None,
    verified_parent_artifact_id: str | None,
    lineage_status: str,
    model_path: Path,
    tokenizer_path: Path,
    adapter_path: Path | None,
    usable_for_evaluation: bool,
    cache: dict[Path, dict[str, Any]],
    model_sha256: str | None = None,
    tokenizer_sha256: str | None = None,
    adapter_sha256: str | None = None,
) -> dict[str, Any]:
    model = model_path.expanduser().resolve()
    tokenizer = tokenizer_path.expanduser().resolve()
    adapter = adapter_path.expanduser().resolve() if adapter_path else None
    if model_sha256 is None and model.exists():
        model_sha256 = _fingerprint_cached(model, cache)["sha256"]
    if tokenizer_sha256 is None and tokenizer.exists():
        tokenizer_sha256 = _fingerprint_cached(tokenizer, cache)["sha256"]
    if adapter is not None and adapter_sha256 is None:
        adapter_sha256 = _fingerprint_cached(adapter, cache)["sha256"]
    return {
        "artifact_id": artifact_id,
        "design_checkpoint_id": design_checkpoint_id,
        "intended_parent_id": intended_parent_id,
        "verified_parent_artifact_id": verified_parent_artifact_id,
        "lineage_status": lineage_status,
        "model_path": str(model),
        "tokenizer_path": str(tokenizer),
        "model_sha256": model_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "adapter_path": str(adapter) if adapter is not None else None,
        "adapter_sha256": adapter_sha256,
        "usable_for_evaluation": usable_for_evaluation,
    }


def _metadata_source(request: RecoveryRequest) -> Path:
    if request.minutes_metadata_source is not None:
        return request.minutes_metadata_source.expanduser().resolve()
    adapter = request.minutes_adapter.expanduser().resolve()
    if adapter.name.startswith("checkpoint-"):
        return adapter.parent
    return adapter


def _legacy_tokenizer_source(request: RecoveryRequest) -> Path:
    return (
        (
            request.legacy_grpo_tokenizer
            if request.legacy_grpo_tokenizer is not None
            else request.legacy_grpo_adapter
        )
        .expanduser()
        .resolve()
    )


def _validate_request(
    request: RecoveryRequest,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    foundation = _normalise_directory(
        request.foundation_model, label="foundation model"
    )
    analysis = _normalise_directory(
        request.analysis_sft_model, label="analysis SFT model"
    )
    _normalise_directory(request.legacy_grpo_model, label="legacy GRPO model")
    legacy_adapter = _normalise_directory(
        request.legacy_grpo_adapter, label="legacy GRPO adapter"
    )
    minutes_adapter = _normalise_directory(
        request.minutes_adapter, label="Minutes adapter"
    )
    _normalise_directory(
        request.historical_minutes_model, label="historical Minutes model"
    )
    _normalise_directory(
        request.overwrite_reference_model, label="overwrite reference model"
    )
    _normalise_directory(
        _legacy_tokenizer_source(request), label="legacy GRPO tokenizer"
    )
    _normalise_directory(_metadata_source(request), label="Minutes metadata")

    if request.destination.expanduser().resolve().exists():
        raise FileExistsError(
            "Recovery destination must be a brand-new path: "
            f"{request.destination.expanduser().resolve()}"
        )
    minutes_parent = verify_adapter_parent(minutes_adapter, analysis)
    legacy_parent = verify_adapter_parent(legacy_adapter, foundation)
    analysis_parent = None
    if request.analysis_sft_adapter is not None:
        analysis_adapter = _normalise_directory(
            request.analysis_sft_adapter, label="analysis SFT adapter"
        )
        analysis_parent = verify_adapter_parent(analysis_adapter, foundation)
    return minutes_parent, legacy_parent, analysis_parent


def _manifest_artifacts(
    request: RecoveryRequest,
    *,
    cache: dict[Path, dict[str, Any]],
    recovered_model_path: Path,
    recovered_model_sha256: str | None,
    recovered_tokenizer_sha256: str | None,
    recovery_complete: bool,
) -> list[dict[str, Any]]:
    analysis_adapter = (
        request.analysis_sft_adapter.expanduser().resolve()
        if request.analysis_sft_adapter is not None
        else None
    )
    records = [
        _artifact_record(
            artifact_id=EVAL_BASE_ID,
            design_checkpoint_id="chk-0",
            intended_parent_id=None,
            verified_parent_artifact_id=None,
            lineage_status="verified_root_artifact",
            model_path=request.foundation_model,
            tokenizer_path=request.foundation_model,
            adapter_path=None,
            usable_for_evaluation=True,
            cache=cache,
        ),
        _artifact_record(
            artifact_id=EVAL_ANALYSIS_SFT_ID,
            design_checkpoint_id="chk-1",
            intended_parent_id="chk-0",
            verified_parent_artifact_id=EVAL_BASE_ID,
            lineage_status="verified_parent_matches_intended_parent",
            model_path=request.analysis_sft_model,
            tokenizer_path=request.analysis_sft_model,
            adapter_path=analysis_adapter,
            usable_for_evaluation=True,
            cache=cache,
        ),
        _artifact_record(
            artifact_id=EVAL_LEGACY_GRPO_ID,
            design_checkpoint_id="chk-2",
            intended_parent_id="chk-1",
            verified_parent_artifact_id=EVAL_BASE_ID,
            lineage_status="verified_parent_differs_from_intended_parent",
            model_path=request.legacy_grpo_model,
            tokenizer_path=_legacy_tokenizer_source(request),
            adapter_path=request.legacy_grpo_adapter,
            usable_for_evaluation=True,
            cache=cache,
        ),
        _artifact_record(
            artifact_id=EVAL_MINUTES_SFT_ID,
            design_checkpoint_id="chk-3",
            intended_parent_id="chk-2",
            verified_parent_artifact_id=EVAL_ANALYSIS_SFT_ID,
            lineage_status=(
                "verified_parent_differs_from_intended_parent"
                if recovery_complete
                else "planned_recovery_not_executed"
            ),
            model_path=recovered_model_path,
            tokenizer_path=recovered_model_path,
            adapter_path=request.minutes_adapter,
            usable_for_evaluation=recovery_complete,
            cache=cache,
            model_sha256=recovered_model_sha256,
            tokenizer_sha256=recovered_tokenizer_sha256,
        ),
        _artifact_record(
            artifact_id=CORRUPTED_MINUTES_ID,
            design_checkpoint_id="chk-3",
            intended_parent_id="chk-2",
            verified_parent_artifact_id=None,
            lineage_status="payload_overwritten_by_reference_artifact",
            model_path=request.historical_minutes_model,
            tokenizer_path=request.historical_minutes_model,
            adapter_path=request.minutes_adapter,
            usable_for_evaluation=False,
            cache=cache,
        ),
    ]
    return records


def _prepare_merge_adapter(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / "adapter_config.json", destination / "adapter_config.json")
    for weight_path in _adapter_weight_paths(source):
        shutil.copy2(weight_path, destination / weight_path.name)

    # The historical merge utility knows which PEFT compatibility keys to
    # remove.  It is safe here because this is a private temporary copy.
    from jobs.merge_model import clean_adapter_config

    clean_adapter_config(destination)


def _copy_tokenizer_bundle(source: Path, destination: Path) -> dict[str, Any]:
    source_paths = _tokenizer_paths(source)
    for name in TOKENIZER_FILES:
        (destination / name).unlink(missing_ok=True)
    for path in source_paths:
        shutil.copy2(path, destination / path.name)
    source_fingerprint = fingerprint_tokenizer_payload(source)
    destination_fingerprint = fingerprint_tokenizer_payload(destination)
    if source_fingerprint["sha256"] != destination_fingerprint["sha256"]:
        raise ValueError("Recovered tokenizer files do not match their source")
    return destination_fingerprint


def _copy_training_metadata(source: Path, destination: Path) -> dict[str, Any]:
    source_paths = _metadata_paths(source)
    for path in source_paths:
        shutil.copy2(path, destination / path.name)
    source_fingerprint = fingerprint_training_metadata(source)
    destination_fingerprint = _inventory_fingerprint(
        destination,
        [destination / path.name for path in source_paths],
        kind="training-metadata",
    )
    if source_fingerprint["sha256"] != destination_fingerprint["sha256"]:
        raise ValueError("Recovered training metadata does not match its source")
    return destination_fingerprint


def _default_merge_executor(
    base_model_path: str,
    adapter_path: Path,
    merged_path: Path,
) -> None:
    # Reuse the established project merge implementation.  The caller supplies
    # a fresh destination and a sanitized temporary adapter copy.
    from jobs.merge_model import merge_model

    merge_model(base_model_path, adapter_path, merged_path)


def _assert_fingerprint_unchanged(
    *,
    label: str,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    compared = ("sha256", "file_count", "total_bytes", "algorithm")
    differences = {
        field: {"before": before.get(field), "after": after.get(field)}
        for field in compared
        if before.get(field) != after.get(field)
    }
    if differences:
        raise ValueError(f"{label} changed during recovery: {differences}")


def recover_minutes_checkpoint(
    request: RecoveryRequest,
    *,
    execute: bool = False,
    merge_executor: MergeExecutor | None = None,
    generated_at_utc: str | None = None,
) -> tuple[dict[str, Any], Path | None]:
    """Audit sources and optionally recover the Minutes model without training."""

    minutes_parent, legacy_parent, analysis_parent = _validate_request(request)
    overwrite_audit = audit_overwritten_model(
        request.historical_minutes_model,
        request.overwrite_reference_model,
    )
    if overwrite_audit["status"] != "payload_overwritten_by_reference_artifact":
        raise ValueError(
            "Historical Minutes model does not exactly match the declared "
            "overwrite reference; refusing to label or recover it"
        )

    destination = request.destination.expanduser().resolve()
    final_model_path = destination / "model"
    metadata_source = _metadata_source(request)
    cache: dict[Path, dict[str, Any]] = {}

    if not execute:
        artifacts = _manifest_artifacts(
            request,
            cache=cache,
            recovered_model_path=final_model_path,
            recovered_model_sha256=None,
            recovered_tokenizer_sha256=None,
            recovery_complete=False,
        )
        return (
            {
                "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
                "status": "dry_run",
                "generated_at_utc": generated_at_utc or _utc_now(),
                "generation_only": False,
                "training_performed": False,
                "write_performed": False,
                "artifacts": artifacts,
                "lineage_evidence": {
                    EVAL_ANALYSIS_SFT_ID: analysis_parent,
                    EVAL_LEGACY_GRPO_ID: legacy_parent,
                    EVAL_MINUTES_SFT_ID: minutes_parent,
                },
                "historical_overwrite_audit": overwrite_audit,
                "recovery": {
                    "destination": str(destination),
                    "model_path": str(final_model_path),
                    "method": "peft-merge-and-unload-via-fresh-staging-copy",
                    "tokenizer_source": str(
                        request.minutes_adapter.expanduser().resolve()
                    ),
                    "metadata_source": str(metadata_source),
                    "requires_execute_flag": True,
                },
            },
            None,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Recovery destination already exists: {destination}")

    base_before = fingerprint_artifact_path(request.analysis_sft_model)
    adapter_before = fingerprint_artifact_path(request.minutes_adapter)
    metadata_before = fingerprint_artifact_path(metadata_source)
    reference_weights = overwrite_audit["reference_weight_payload"]

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.recovery-",
            dir=destination.parent,
        )
    )
    committed = False
    try:
        staged_adapter = staging / ".merge_adapter"
        staged_model = staging / "model"
        _prepare_merge_adapter(
            request.minutes_adapter.expanduser().resolve(),
            staged_adapter,
        )
        (merge_executor or _default_merge_executor)(
            str(request.analysis_sft_model.expanduser().resolve()),
            staged_adapter,
            staged_model,
        )
        if not staged_model.is_dir():
            raise FileNotFoundError(
                f"Merge executor did not create a model directory: {staged_model}"
            )

        tokenizer_payload = _copy_tokenizer_bundle(
            request.minutes_adapter.expanduser().resolve(),
            staged_model,
        )
        metadata_payload = _copy_training_metadata(
            metadata_source,
            staged_model,
        )
        shutil.rmtree(staged_adapter)

        recovered_weights = fingerprint_weight_payload(staged_model)
        if recovered_weights["sha256"] == reference_weights["sha256"]:
            raise ValueError(
                "Recovered weights still equal the overwrite reference payload"
            )
        recovered_payload = fingerprint_model_payload(staged_model)
        recovered_model_artifact = fingerprint_artifact_path(staged_model)

        base_after = fingerprint_artifact_path(request.analysis_sft_model)
        adapter_after = fingerprint_artifact_path(request.minutes_adapter)
        metadata_after = fingerprint_artifact_path(metadata_source)
        _assert_fingerprint_unchanged(
            label="Analysis SFT base", before=base_before, after=base_after
        )
        _assert_fingerprint_unchanged(
            label="Minutes adapter", before=adapter_before, after=adapter_after
        )
        _assert_fingerprint_unchanged(
            label="Minutes metadata", before=metadata_before, after=metadata_after
        )

        # Source fingerprints already computed above are also the exact values
        # consumed by the artifact records; avoid re-hashing large checkpoints.
        cache[request.analysis_sft_model.expanduser().resolve()] = base_after
        cache[request.minutes_adapter.expanduser().resolve()] = adapter_after
        artifacts = _manifest_artifacts(
            request,
            cache=cache,
            recovered_model_path=final_model_path,
            recovered_model_sha256=recovered_model_artifact["sha256"],
            recovered_tokenizer_sha256=recovered_model_artifact["sha256"],
            recovery_complete=True,
        )
        manifest = seal_manifest(
            {
                "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
                "status": "complete",
                "generated_at_utc": generated_at_utc or _utc_now(),
                "generation_only": False,
                "training_performed": False,
                "write_performed": True,
                "artifacts": artifacts,
                "lineage_evidence": {
                    EVAL_ANALYSIS_SFT_ID: analysis_parent,
                    EVAL_LEGACY_GRPO_ID: legacy_parent,
                    EVAL_MINUTES_SFT_ID: minutes_parent,
                },
                "historical_overwrite_audit": overwrite_audit,
                "recovery": {
                    "destination": str(destination),
                    "model_path": str(final_model_path),
                    "method": "peft-merge-and-unload-via-fresh-staging-copy",
                    "tokenizer_source": str(
                        request.minutes_adapter.expanduser().resolve()
                    ),
                    "metadata_source": str(metadata_source),
                    "source_immutability": {
                        "analysis_sft_model": base_after,
                        "minutes_adapter": adapter_after,
                        "minutes_metadata": metadata_after,
                    },
                    "output_validation": {
                        "status": "passed",
                        "model_artifact": {
                            **recovered_model_artifact,
                            "path": str(final_model_path),
                        },
                        "model_load_payload": {
                            **recovered_payload,
                            "path": str(final_model_path),
                        },
                        "weight_payload": {
                            **recovered_weights,
                            "path": str(final_model_path),
                        },
                        "tokenizer_payload": {
                            **tokenizer_payload,
                            "path": str(final_model_path),
                        },
                        "training_metadata": {
                            **metadata_payload,
                            "path": str(final_model_path),
                        },
                        "differs_from_overwrite_reference": True,
                        "tokenizer_matches_adapter_checkpoint": True,
                        "metadata_matches_training_output": True,
                    },
                },
                "claim_boundary": (
                    "The recovered Minutes artifact is verified as analysis-SFT "
                    "plus the Minutes adapter. It is not evidence that the "
                    "historical chk-1→chk-2→chk-3 chain was executed."
                ),
            }
        )
        manifest_path = staging / "checkpoint_manifest.json"
        write_immutable_json(manifest_path, manifest)

        if destination.exists():
            raise FileExistsError(
                f"Recovery destination appeared during execution: {destination}"
            )
        staging.rename(destination)
        committed = True

        final_fingerprint = fingerprint_artifact_path(final_model_path)
        if final_fingerprint["sha256"] != recovered_model_artifact["sha256"]:
            raise ValueError(
                "Recovered model fingerprint changed after final directory commit"
            )
        final_manifest = _read_json_object(
            destination / "checkpoint_manifest.json",
            label="recovered checkpoint manifest",
        )
        validate_manifest_integrity(final_manifest)
        return final_manifest, destination / "checkpoint_manifest.json"
    finally:
        if not committed and staging.exists():
            shutil.rmtree(staging)


def verify_checkpoint_artifact(
    *,
    checkpoint_manifest: str | Path,
    artifact_id: str,
    model_path: str | Path,
    tokenizer_path: str | Path,
    expected_manifest_sha256: str,
    expected_manifest_payload_sha256: str,
    expected_model_sha256: str,
    expected_tokenizer_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless a runtime artifact matches a sealed provenance record.

    The checkpoint manifest is not treated as a self-authenticating source:
    callers must pin both its file digest and sealed payload digest.  Runtime
    model and tokenizer directory trees are then hashed and compared with both
    the caller's frozen digests and the selected usable artifact record.
    """

    manifest_path = Path(checkpoint_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint provenance manifest does not exist: {manifest_path}"
        )

    expected_manifest_digest = validate_sha256(
        expected_manifest_sha256,
        label="expected checkpoint manifest SHA-256",
    )
    actual_manifest_digest = sha256_file(manifest_path)
    if actual_manifest_digest != expected_manifest_digest:
        raise ValueError(
            "Checkpoint provenance manifest file digest mismatch: "
            f"expected={expected_manifest_digest}, actual={actual_manifest_digest}"
        )

    manifest = _read_json_object(
        manifest_path,
        label="checkpoint provenance manifest",
    )
    validate_manifest_integrity(manifest)
    if manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported checkpoint provenance manifest schema: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("status") != "complete":
        raise ValueError(
            "Checkpoint provenance manifest is not complete: "
            f"{manifest.get('status')!r}"
        )

    expected_payload_digest = validate_sha256(
        expected_manifest_payload_sha256,
        label="expected checkpoint manifest payload SHA-256",
    )
    actual_payload_digest = validate_sha256(
        manifest.get("integrity", {}).get("payload_sha256"),
        label="checkpoint manifest payload SHA-256",
    )
    if actual_payload_digest != expected_payload_digest:
        raise ValueError(
            "Checkpoint provenance manifest payload digest mismatch: "
            f"expected={expected_payload_digest}, actual={actual_payload_digest}"
        )

    selected_id = str(artifact_id or "").strip()
    if not selected_id:
        raise ValueError("Checkpoint artifact ID must be non-empty")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Checkpoint provenance manifest has no artifact inventory")
    matches = [
        row
        for row in artifacts
        if isinstance(row, dict) and row.get("artifact_id") == selected_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Checkpoint artifact ID must resolve exactly once: {selected_id!r}"
        )
    artifact = matches[0]
    if artifact.get("usable_for_evaluation") is not True:
        raise ValueError(
            f"Checkpoint artifact is not approved for evaluation: {selected_id!r}"
        )

    runtime_model = _normalise_directory(model_path, label="runtime model")
    runtime_tokenizer = _normalise_directory(
        tokenizer_path,
        label="runtime tokenizer",
    )
    registered_model = _normalise_directory(
        str(artifact.get("model_path") or ""),
        label="registered model",
    )
    registered_tokenizer = _normalise_directory(
        str(artifact.get("tokenizer_path") or ""),
        label="registered tokenizer",
    )
    if runtime_model != registered_model:
        raise ValueError(
            "Runtime model path differs from the registered checkpoint artifact: "
            f"{runtime_model} != {registered_model}"
        )
    if runtime_tokenizer != registered_tokenizer:
        raise ValueError(
            "Runtime tokenizer path differs from the registered checkpoint artifact: "
            f"{runtime_tokenizer} != {registered_tokenizer}"
        )

    manifest_model_digest = validate_sha256(
        artifact.get("model_sha256"),
        label="registered model SHA-256",
    )
    manifest_tokenizer_digest = validate_sha256(
        artifact.get("tokenizer_sha256"),
        label="registered tokenizer SHA-256",
    )
    frozen_model_digest = validate_sha256(
        expected_model_sha256,
        label="expected runtime model SHA-256",
    )
    frozen_tokenizer_digest = validate_sha256(
        expected_tokenizer_sha256,
        label="expected runtime tokenizer SHA-256",
    )
    if manifest_model_digest != frozen_model_digest:
        raise ValueError(
            "Frozen model digest differs from the checkpoint manifest: "
            f"{frozen_model_digest} != {manifest_model_digest}"
        )
    if manifest_tokenizer_digest != frozen_tokenizer_digest:
        raise ValueError(
            "Frozen tokenizer digest differs from the checkpoint manifest: "
            f"{frozen_tokenizer_digest} != {manifest_tokenizer_digest}"
        )

    fingerprint_cache: dict[Path, dict[str, Any]] = {}
    actual_model = _fingerprint_cached(runtime_model, fingerprint_cache)
    actual_tokenizer = _fingerprint_cached(runtime_tokenizer, fingerprint_cache)
    if actual_model["sha256"] != manifest_model_digest:
        raise ValueError(
            "Runtime model content differs from the registered checkpoint: "
            f"expected={manifest_model_digest}, actual={actual_model['sha256']}"
        )
    if actual_tokenizer["sha256"] != manifest_tokenizer_digest:
        raise ValueError(
            "Runtime tokenizer content differs from the registered checkpoint: "
            f"expected={manifest_tokenizer_digest}, "
            f"actual={actual_tokenizer['sha256']}"
        )

    if selected_id == EVAL_MINUTES_SFT_ID:
        output_validation = manifest.get("recovery", {}).get("output_validation", {})
        if (
            not isinstance(output_validation, dict)
            or output_validation.get("status") != "passed"
            or output_validation.get("differs_from_overwrite_reference") is not True
        ):
            raise ValueError(
                "Recovered Minutes checkpoint lacks a passed output validation "
                "that differs from the overwrite reference"
            )
        recovered_model = output_validation.get("model_artifact")
        if (
            not isinstance(recovered_model, dict)
            or recovered_model.get("sha256") != manifest_model_digest
        ):
            raise ValueError(
                "Recovered Minutes model digest disagrees with output validation"
            )

    return {
        "status": "passed",
        "schema_version": "checkpoint-runtime-verification-v1",
        "artifact_id": selected_id,
        "checkpoint_manifest": {
            "path": str(manifest_path),
            "sha256": actual_manifest_digest,
            "payload_sha256": actual_payload_digest,
        },
        "model": actual_model,
        "tokenizer": actual_tokenizer,
        "usable_for_evaluation": True,
    }


def build_invalidation_record(
    *,
    run_path: str | Path,
    invalid_model: str | Path,
    matched_reference_model: str | Path,
    reason: str,
    reason_code: str = "checkpoint_payload_identity_mismatch",
    checkpoint_manifest: str | Path | None = None,
    invalidated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a sealed record without changing the invalid run directory."""

    run = _normalise_directory(run_path, label="invalid run")
    invalid = _normalise_directory(invalid_model, label="invalid generation model")
    reference = _normalise_directory(
        matched_reference_model, label="matched reference model"
    )
    if not str(reason).strip():
        raise ValueError("Invalidation reason must be non-empty")
    invalid_weights = fingerprint_weight_payload(invalid)
    reference_weights = fingerprint_weight_payload(reference)
    if invalid_weights["sha256"] != reference_weights["sha256"]:
        raise ValueError(
            "Invalid model weights do not match the declared reference payload"
        )

    evidence: dict[str, Any] = {
        "invalid_model_artifact": fingerprint_artifact_path(invalid),
        "invalid_model_weight_payload": invalid_weights,
        "matched_reference_artifact": fingerprint_artifact_path(reference),
        "matched_reference_weight_payload": reference_weights,
    }
    if checkpoint_manifest is not None:
        manifest_path = Path(checkpoint_manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint manifest does not exist: {manifest_path}"
            )
        evidence["checkpoint_manifest"] = fingerprint_artifact_path(manifest_path)

    return seal_manifest(
        {
            "schema_version": INVALIDATION_SCHEMA_VERSION,
            "status": "invalidated",
            "invalidated_at_utc": invalidated_at_utc or _utc_now(),
            "reason_code": str(reason_code).strip(),
            "reason": str(reason).strip(),
            "scope": "entire_generation_run",
            "run_artifact_before_invalidation": fingerprint_artifact_path(run),
            "evidence": evidence,
            "downstream_use": {
                "generation": "prohibited",
                "scoring": "prohibited",
                "publication": "prohibited",
            },
            "source_run_mutated": False,
        }
    )


def invalidate_run(
    *,
    run_path: str | Path,
    invalid_model: str | Path,
    matched_reference_model: str | Path,
    reason: str,
    reason_code: str = "checkpoint_payload_identity_mismatch",
    checkpoint_manifest: str | Path | None = None,
    output: str | Path | None = None,
    execute: bool = False,
    invalidated_at_utc: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Plan or immutably register an invalid run in a sibling registry."""

    run = _normalise_directory(run_path, label="invalid run")
    output_path = (
        Path(output).expanduser().resolve()
        if output is not None
        else run.parent / "_invalidations" / f"{run.name}.json"
    )
    if output_path == run or run in output_path.parents:
        raise ValueError(
            "Invalidation records must be outside the source run directory"
        )
    record = build_invalidation_record(
        run_path=run,
        invalid_model=invalid_model,
        matched_reference_model=matched_reference_model,
        reason=reason,
        reason_code=reason_code,
        checkpoint_manifest=checkpoint_manifest,
        invalidated_at_utc=invalidated_at_utc,
    )
    if execute:
        before = fingerprint_artifact_path(run)
        write_immutable_json(output_path, record)
        after = fingerprint_artifact_path(run)
        _assert_fingerprint_unchanged(
            label="Invalidated run", before=before, after=after
        )
    return record, output_path


def _recovery_request_from_args(args: argparse.Namespace) -> RecoveryRequest:
    return RecoveryRequest(
        foundation_model=args.foundation_model,
        analysis_sft_model=args.analysis_sft_model,
        analysis_sft_adapter=args.analysis_sft_adapter,
        legacy_grpo_model=args.legacy_grpo_model,
        legacy_grpo_adapter=args.legacy_grpo_adapter,
        legacy_grpo_tokenizer=args.legacy_grpo_tokenizer,
        minutes_adapter=args.minutes_adapter,
        minutes_metadata_source=args.minutes_metadata_source,
        historical_minutes_model=args.historical_minutes_model,
        overwrite_reference_model=args.overwrite_reference_model,
        destination=args.destination,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and recover checkpoint provenance without training or "
            "overwriting historical artifacts."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit-overwrite",
        help="Compare a historical merged model with a suspected overwrite source.",
    )
    audit.add_argument("--historical-model", type=Path, required=True)
    audit.add_argument("--overwrite-reference-model", type=Path, required=True)

    recover = subparsers.add_parser(
        "recover-minutes",
        help=(
            "Recover analysis-SFT + Minutes adapter into a new version "
            "directory. Defaults to dry-run."
        ),
    )
    recover.add_argument("--foundation-model", type=Path, required=True)
    recover.add_argument("--analysis-sft-model", type=Path, required=True)
    recover.add_argument("--analysis-sft-adapter", type=Path)
    recover.add_argument("--legacy-grpo-model", type=Path, required=True)
    recover.add_argument("--legacy-grpo-adapter", type=Path, required=True)
    recover.add_argument("--legacy-grpo-tokenizer", type=Path)
    recover.add_argument("--minutes-adapter", type=Path, required=True)
    recover.add_argument("--minutes-metadata-source", type=Path)
    recover.add_argument("--historical-minutes-model", type=Path, required=True)
    recover.add_argument("--overwrite-reference-model", type=Path, required=True)
    recover.add_argument("--destination", type=Path, required=True)
    recover.add_argument(
        "--execute",
        action="store_true",
        help="Perform the merge and write the new version directory.",
    )

    invalidate = subparsers.add_parser(
        "invalidate-run",
        help=(
            "Create an immutable sibling invalidation record. Defaults to "
            "dry-run and never edits the source run."
        ),
    )
    invalidate.add_argument("--run", type=Path, required=True)
    invalidate.add_argument("--invalid-model", type=Path, required=True)
    invalidate.add_argument("--matched-reference-model", type=Path, required=True)
    invalidate.add_argument("--reason", required=True)
    invalidate.add_argument(
        "--reason-code",
        default="checkpoint_payload_identity_mismatch",
    )
    invalidate.add_argument("--checkpoint-manifest", type=Path)
    invalidate.add_argument("--output", type=Path)
    invalidate.add_argument("--invalidated-at-utc")
    invalidate.add_argument(
        "--execute",
        action="store_true",
        help="Write the immutable invalidation record.",
    )

    verify = subparsers.add_parser(
        "verify-artifact",
        help=(
            "Fail closed unless runtime model/tokenizer content matches one "
            "usable artifact in a sealed and externally pinned manifest."
        ),
    )
    verify.add_argument("--checkpoint-manifest", type=Path, required=True)
    verify.add_argument("--artifact-id", required=True)
    verify.add_argument("--model", type=Path, required=True)
    verify.add_argument("--tokenizer", type=Path, required=True)
    verify.add_argument("--expected-manifest-sha256", required=True)
    verify.add_argument("--expected-manifest-payload-sha256", required=True)
    verify.add_argument("--expected-model-sha256", required=True)
    verify.add_argument("--expected-tokenizer-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "audit-overwrite":
            payload = audit_overwritten_model(
                args.historical_model,
                args.overwrite_reference_model,
            )
            print(_serialise_json(payload), end="")
            return 0

        if args.command == "recover-minutes":
            manifest, path = recover_minutes_checkpoint(
                _recovery_request_from_args(args),
                execute=args.execute,
            )
            if path is None:
                print(_serialise_json(manifest), end="")
            else:
                print(f"checkpoint_manifest={path}")
                print(f"manifest_sha256={sha256_file(path)}")
                print(f"payload_sha256={manifest['integrity']['payload_sha256']}")
            return 0

        if args.command == "verify-artifact":
            verification = verify_checkpoint_artifact(
                checkpoint_manifest=args.checkpoint_manifest,
                artifact_id=args.artifact_id,
                model_path=args.model,
                tokenizer_path=args.tokenizer,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_manifest_payload_sha256=(
                    args.expected_manifest_payload_sha256
                ),
                expected_model_sha256=args.expected_model_sha256,
                expected_tokenizer_sha256=args.expected_tokenizer_sha256,
            )
            print(_serialise_json(verification), end="")
            return 0

        record, output = invalidate_run(
            run_path=args.run,
            invalid_model=args.invalid_model,
            matched_reference_model=args.matched_reference_model,
            reason=args.reason,
            reason_code=args.reason_code,
            checkpoint_manifest=args.checkpoint_manifest,
            output=args.output,
            execute=args.execute,
            invalidated_at_utc=args.invalidated_at_utc,
        )
        if args.execute:
            print(f"invalidation_record={output}")
            print(f"record_sha256={sha256_file(output)}")
        else:
            print(_serialise_json(record), end="")
            print(f"planned_output={output}")
        return 0
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
