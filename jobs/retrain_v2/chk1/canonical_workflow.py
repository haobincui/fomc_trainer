"""Validate and publish a production chk1 canonical release without model calls.

This module is deliberately separate from the generation workflow.  Its
commands can only replay and package a completed ``mode=full`` generation;
they never call a teacher, critic, or Minutes resolver.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from open_r1.provenance import sha256_file, validate_sha256

from . import generation_pipeline as generation
from .contracts import canonical_json, sha256_text
from .release import QualityThresholds, publish_release


SPLITS = ("train", "eval", "test")
FULL_GENERATION_HANDOFF = Path(
    "output/data/retrain_v2/chk1/generation_full_v7/generation_handoff.json"
)
PREPARE_HANDOFF = Path(
    "output/data/retrain_v2/chk1/prepared_sparse_v3/prepare_handoff.json"
)
RELEASE_ROOT = Path("output/data/retrain_v2/chk1/canonical_releases")

GENERATION_BUNDLE_SCHEMA_VERSION = "chk1-generation-source-bundle-v1"
PREPARATION_BUNDLE_SCHEMA_VERSION = "chk1-preparation-source-bundle-v1"
SOURCE_CONTRACT_SCHEMA_VERSION = "chk1-canonical-source-contract-v1"
CANONICAL_ADMISSION_SCHEMA_VERSION = "chk1-canonical-admission-v2"

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class CanonicalWorkflowError(RuntimeError):
    """Raised when validation or publication cannot be proven safe."""


@dataclass(frozen=True)
class BoundFile:
    source_path: Path
    source_relative: str
    release_path: str
    sha256: str
    bytes: int
    rows: int | None = None


@dataclass(frozen=True)
class VerifiedGeneration:
    repo_root: Path
    generation_root: Path
    generation_handoff_path: Path
    generation_handoff: dict[str, Any]
    generation_handoff_sha256: str
    preparation_root: Path
    prepare_handoff_path: Path
    prepare_handoff: dict[str, Any]
    prepare_handoff_sha256: str
    preparation_binding: dict[str, Any]
    selection: dict[str, Any]
    cache_manifest: dict[str, Any]
    population: tuple[dict[str, Any], ...]
    sft_rows: dict[str, tuple[dict[str, Any], ...]]
    manifest_rows: dict[str, tuple[dict[str, Any], ...]]
    exclusions: tuple[dict[str, Any], ...]
    generation_files: tuple[BoundFile, ...]
    preparation_files: tuple[BoundFile, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanonicalWorkflowError(message)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _payload_with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = json.loads(canonical_json(dict(payload)))
    return {**body, "payload_sha256": sha256_text(canonical_json(body))}


def _validate_payload_hash(
    payload: Mapping[str, Any], *, schema_version: str, label: str
) -> dict[str, Any]:
    _require(payload.get("schema_version") == schema_version, f"{label} schema mismatch")
    digest = payload.get("payload_sha256")
    try:
        validate_sha256(digest, label=f"{label}.payload_sha256")
    except ValueError as exc:
        raise CanonicalWorkflowError(str(exc)) from exc
    body = dict(payload)
    body.pop("payload_sha256", None)
    _require(
        sha256_text(canonical_json(body)) == digest,
        f"{label} payload hash mismatch",
    )
    return dict(payload)


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or a symlink: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalWorkflowError(f"unable to parse {label}: {path}") from exc
    _require(isinstance(payload, dict), f"{label} must contain one JSON object")
    return payload


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or a symlink: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                _require(bool(line.strip()), f"{label} has a blank row at {line_number}")
                value = json.loads(line)
                _require(isinstance(value, dict), f"{label} row {line_number} is not an object")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalWorkflowError(f"unable to parse {label}: {path}") from exc
    return rows


def _count_jsonl(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except (OSError, UnicodeDecodeError) as exc:
        raise CanonicalWorkflowError(f"unable to count JSONL rows: {path}") from exc


def _require_inside_no_symlink(root: Path, path: Path, *, label: str) -> Path:
    root = root.resolve()
    lexical = path if path.is_absolute() else root / path
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise CanonicalWorkflowError(f"{label} escapes repository root") from exc
    cursor = root
    for part in relative.parts:
        _require(part not in {"", ".", ".."}, f"{label} has an unsafe path")
        cursor = cursor / part
        if cursor.exists() or cursor.is_symlink():
            _require(not cursor.is_symlink(), f"{label} contains a symlink: {cursor}")
    resolved = lexical.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CanonicalWorkflowError(f"{label} escapes repository root") from exc
    return resolved


def _safe_record_path(
    base: Path,
    record: Any,
    *,
    repo_root: Path,
    label: str,
    require_rows: bool | None = None,
) -> tuple[Path, str, str, int | None]:
    _require(isinstance(record, Mapping), f"{label} record must be an object")
    raw = record.get("path")
    _require(isinstance(raw, str) and raw.strip(), f"{label}.path is missing")
    relative = Path(raw)
    _require(not relative.is_absolute() and ".." not in relative.parts, f"{label}.path is unsafe")
    path = _require_inside_no_symlink(repo_root, base / relative, label=label)
    _require(path.is_file() and not path.is_symlink(), f"{label} file is missing")
    try:
        expected_sha = str(record.get("sha256") or "")
        validate_sha256(expected_sha, label=f"{label}.sha256")
    except ValueError as exc:
        raise CanonicalWorkflowError(str(exc)) from exc
    _require(sha256_file(path) == expected_sha, f"{label} SHA-256 mismatch")
    rows: int | None = None
    if require_rows is not False:
        raw_rows = record.get("rows", record.get("row_count"))
        if raw_rows is not None:
            _require(
                isinstance(raw_rows, int) and not isinstance(raw_rows, bool) and raw_rows >= 0,
                f"{label} row count is invalid",
            )
            rows = raw_rows
            if path.suffix == ".jsonl":
                _require(_count_jsonl(path) == rows, f"{label} row count mismatch")
        elif require_rows is True:
            raise CanonicalWorkflowError(f"{label} row count is missing")
    expected_bytes = record.get("bytes")
    if expected_bytes is not None:
        _require(
            isinstance(expected_bytes, int)
            and not isinstance(expected_bytes, bool)
            and expected_bytes >= 0
            and path.stat().st_size == expected_bytes,
            f"{label} byte count mismatch",
        )
    return path, relative.as_posix(), expected_sha, rows


def _file_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _bound_file(
    *,
    source_path: Path,
    source_relative: str,
    release_prefix: str,
    digest: str,
    rows: int | None,
) -> BoundFile:
    return BoundFile(
        source_path=source_path,
        source_relative=source_relative,
        release_path=f"{release_prefix}/{source_relative}",
        sha256=digest,
        bytes=source_path.stat().st_size,
        rows=rows,
    )


def _preparation_file_records(
    *, repo_root: Path, handoff_path: Path, handoff: Mapping[str, Any]
) -> tuple[BoundFile, ...]:
    root = handoff_path.parent
    artifacts = handoff.get("artifacts")
    _require(isinstance(artifacts, Mapping), "preparation artifact roster is missing")
    records: list[tuple[str, Any, bool | None]] = []
    for key in ("inventory", "style_guide", "canonical_population"):
        records.append((key, artifacts.get(key), False))
    prepared = artifacts.get("prepared")
    _require(isinstance(prepared, Mapping) and set(prepared) == set(SPLITS), "preparation split records are incomplete")
    for split in SPLITS:
        records.append((f"prepared.{split}", prepared[split], True))
    audit = artifacts.get("audit")
    expected_audit = {
        "evidence_exclusions",
        "sample_exclusions",
        "precanonical_exclusions",
    }
    _require(isinstance(audit, Mapping) and set(audit) == expected_audit, "preparation audit records are incomplete")
    for key in sorted(expected_audit):
        records.append((f"audit.{key}", audit[key], True))

    files: list[BoundFile] = [
        _bound_file(
            source_path=handoff_path,
            source_relative="prepare_handoff.json",
            release_prefix="source/preparation",
            digest=sha256_file(handoff_path),
            rows=None,
        )
    ]
    seen = {"prepare_handoff.json"}
    for label, record, require_rows in records:
        path, relative, digest, rows = _safe_record_path(
            root,
            record,
            repo_root=repo_root,
            label=f"preparation {label}",
            require_rows=require_rows,
        )
        _require(relative not in seen, f"duplicate preparation artifact path: {relative}")
        seen.add(relative)
        files.append(
            _bound_file(
                source_path=path,
                source_relative=relative,
                release_prefix="source/preparation",
                digest=digest,
                rows=rows,
            )
        )
    return tuple(sorted(files, key=lambda item: item.source_relative))


def _generation_file_records(
    *,
    repo_root: Path,
    handoff_path: Path,
    handoff: Mapping[str, Any],
) -> tuple[tuple[BoundFile, ...], dict[str, Any], dict[str, Any]]:
    root = handoff_path.parent
    files_payload = handoff.get("files")
    _require(
        isinstance(files_payload, Mapping)
        and set(files_payload)
        == {"sft", "manifests", "exclusions", "selection", "cache_manifest"},
        "generation handoff file roster is not canonical",
    )
    sft = files_payload["sft"]
    manifests = files_payload["manifests"]
    _require(isinstance(sft, Mapping) and set(sft) == set(SPLITS), "generation SFT records are incomplete")
    _require(isinstance(manifests, Mapping) and set(manifests) == set(SPLITS), "generation manifest records are incomplete")
    records: list[tuple[str, Any]] = []
    for split in SPLITS:
        records.append((f"sft.{split}", sft[split]))
        records.append((f"manifests.{split}", manifests[split]))
    records.extend(
        (
            ("exclusions", files_payload["exclusions"]),
            ("selection", files_payload["selection"]),
            ("cache_manifest", files_payload["cache_manifest"]),
        )
    )

    bound: list[BoundFile] = [
        _bound_file(
            source_path=handoff_path,
            source_relative="generation_handoff.json",
            release_prefix="source/generation",
            digest=sha256_file(handoff_path),
            rows=None,
        )
    ]
    seen = {"generation_handoff.json"}
    record_paths: dict[str, Path] = {}
    for label, record in records:
        path, relative, digest, rows = _safe_record_path(
            root,
            record,
            repo_root=repo_root,
            label=f"generation {label}",
        )
        _require(relative not in seen, f"duplicate generation artifact path: {relative}")
        seen.add(relative)
        record_paths[label] = path
        bound.append(
            _bound_file(
                source_path=path,
                source_relative=relative,
                release_prefix="source/generation",
                digest=digest,
                rows=rows,
            )
        )

    selection = _validate_payload_hash(
        _load_json(record_paths["selection"], label="selection manifest"),
        schema_version=generation.SELECTION_SCHEMA_VERSION,
        label="selection manifest",
    )
    cache_manifest = _validate_payload_hash(
        _load_json(record_paths["cache_manifest"], label="sample cache manifest"),
        schema_version=generation.CACHE_MANIFEST_SCHEMA_VERSION,
        label="sample cache manifest",
    )
    entries = cache_manifest.get("entries")
    _require(isinstance(entries, list), "sample cache manifest entries are invalid")
    entry_ids: set[str] = set()
    for index, entry in enumerate(entries):
        _require(isinstance(entry, Mapping), f"sample cache entry {index} is invalid")
        sample_id = str(entry.get("sample_id") or "")
        _require(sample_id and sample_id not in entry_ids, "sample cache IDs are empty or duplicated")
        entry_ids.add(sample_id)
        path, relative, digest, rows = _safe_record_path(
            root,
            entry,
            repo_root=repo_root,
            label=f"sample cache entry {index}",
        )
        _require(relative not in seen, f"duplicate generation artifact path: {relative}")
        _require(rows == 1, f"sample cache entry {index} must bind one JSON object")
        seen.add(relative)
        bound.append(
            _bound_file(
                source_path=path,
                source_relative=relative,
                release_prefix="source/generation",
                digest=digest,
                rows=rows,
            )
        )
    selected_ids = selection.get("selected_sample_ids")
    _require(
        isinstance(selected_ids, list)
        and len(selected_ids) == len(set(selected_ids))
        and set(str(item) for item in selected_ids) == entry_ids,
        "sample cache entries do not cover the exact selection",
    )
    return tuple(sorted(bound, key=lambda item: item.source_relative)), selection, cache_manifest


def _generation_code_files(
    *, repo_root: Path, generation_provenance: Mapping[str, Any]
) -> tuple[BoundFile, ...]:
    code = generation_provenance.get("generation_code")
    _require(isinstance(code, Mapping), "generation code provenance is missing")
    payload = dict(code)
    digest = payload.pop("payload_sha256", None)
    try:
        validate_sha256(digest, label="generation_code.payload_sha256")
    except ValueError as exc:
        raise CanonicalWorkflowError(str(exc)) from exc
    _require(
        sha256_text(canonical_json(payload)) == digest,
        "generation code provenance payload hash mismatch",
    )
    records = code.get("files")
    _require(isinstance(records, list) and records, "generation code file roster is empty")
    files: list[BoundFile] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        path, relative, file_sha, rows = _safe_record_path(
            repo_root,
            record,
            repo_root=repo_root,
            label=f"generation code file {index}",
            require_rows=False,
        )
        _require(relative not in seen, f"duplicate generation code path: {relative}")
        seen.add(relative)
        files.append(
            _bound_file(
                source_path=path,
                source_relative=f"generation_code/{relative}",
                release_prefix="source/generation",
                digest=file_sha,
                rows=rows,
            )
        )
    return tuple(sorted(files, key=lambda item: item.source_relative))


def _read_generation_rows(
    generation_root: Path, handoff: Mapping[str, Any]
) -> tuple[
    dict[str, tuple[dict[str, Any], ...]],
    dict[str, tuple[dict[str, Any], ...]],
    tuple[dict[str, Any], ...],
]:
    files = handoff["files"]
    sft: dict[str, tuple[dict[str, Any], ...]] = {}
    manifests: dict[str, tuple[dict[str, Any], ...]] = {}
    for split in SPLITS:
        sft[split] = tuple(
            _load_jsonl(
                generation_root / files["sft"][split]["path"],
                label=f"generation SFT {split}",
            )
        )
        manifests[split] = tuple(
            _load_jsonl(
                generation_root / files["manifests"][split]["path"],
                label=f"generation manifests {split}",
            )
        )
        _require(len(sft[split]) == len(manifests[split]), f"{split} SFT/manifest count mismatch")
    exclusions = tuple(
        _load_jsonl(
            generation_root / files["exclusions"]["path"],
            label="generation exclusions",
        )
    )
    return sft, manifests, exclusions


def _replay_projected_terminals(
    *,
    population: Sequence[Mapping[str, Any]],
    sft_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    exclusions: Sequence[Mapping[str, Any]],
    model_input_projector: Any,
    preparation_binding_sha256: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any] | None],
]:
    """Replay safe projections and return exact per-sample terminal artifacts."""

    prepared_by_id = {str(row["sample_id"]): dict(row) for row in population}
    _require(
        len(prepared_by_id) == len(population),
        "prepared population contains duplicate sample IDs",
    )
    terminal_artifacts: dict[str, dict[str, Any]] = {}
    projected_inputs: dict[str, dict[str, Any] | None] = {}
    expected_budget_keys = {
        "schema_version",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "max_prompt_tokens",
        "max_completion_tokens",
        "max_total_tokens",
        "overflow_policy",
        "truncated",
        "passed",
    }
    for split in SPLITS:
        paired_sft = list(sft_rows[split])
        paired_manifests = list(manifest_rows[split])
        _require(
            len(paired_sft) == len(paired_manifests),
            f"{split} terminal SFT/manifest count mismatch",
        )
        for index, (sft, manifest) in enumerate(
            zip(paired_sft, paired_manifests, strict=True)
        ):
            label = f"generation manifests/{split}[{index}]"
            sample_id = str(manifest.get("sample_id") or "")
            prepared = prepared_by_id.get(sample_id)
            _require(prepared is not None, f"{label} is not in the prepared population")
            _require(sample_id not in terminal_artifacts, f"{label} is duplicated")
            _require(
                all(
                    manifest.get(field) == prepared.get(field)
                    for field in ("split", "meeting_date", "atomic_topic")
                ),
                f"{label} prepared identity mismatch",
            )
            inputs = generation._execution_inputs(
                prepared, model_input_projector=model_input_projector
            )
            projected_inputs[sample_id] = inputs
            binding = manifest.get("generation")
            _require(isinstance(binding, Mapping), f"{label} generation binding is invalid")
            expected_prepared_sha = sha256_text(canonical_json(prepared))
            _require(
                binding.get("prepared_row_sha256") == expected_prepared_sha,
                f"{label} prepared row SHA mismatch",
            )
            _require(
                sft.get("prompt") == inputs["student_prompt"]
                and sft.get("provided_data") == inputs["provided_data"],
                f"{label} safe SFT prompt/provided_data projection mismatch",
            )
            _require(
                manifest.get("evidence_lineage") == inputs["evidence_lineage"]
                and binding.get("model_input_projection")
                == inputs["model_input_projection"],
                f"{label} evidence/projection attestation mismatch",
            )
            _require(
                binding.get("generator_tokenizer_sha256")
                == inputs["generator_tokenizer_sha256"]
                and binding.get("student_tokenizer_sha256")
                == inputs["student_tokenizer_sha256"]
                and manifest.get("tokenizer_sha256")
                == inputs["student_tokenizer_sha256"]
                and binding.get("preparation_binding_sha256")
                == preparation_binding_sha256
                and manifest.get("style_guide_sha256")
                == inputs["style_guide_sha256"],
                f"{label} tokenizer/style/preparation projection binding mismatch",
            )
            budget = binding.get("sft_token_budget")
            budget_is_self_consistent = (
                isinstance(budget, Mapping)
                and isinstance(budget.get("prompt_tokens"), int)
                and isinstance(budget.get("completion_tokens"), int)
                and isinstance(budget.get("total_tokens"), int)
                and budget.get("total_tokens")
                == budget.get("prompt_tokens") + budget.get("completion_tokens")
                and budget.get("passed")
                is (
                    budget.get("prompt_tokens") <= 3072
                    and budget.get("completion_tokens") <= 1024
                    and budget.get("total_tokens") <= 4096
                )
            )
            _require(
                isinstance(budget, Mapping)
                and set(budget) == expected_budget_keys
                and budget.get("schema_version") == "chk1-sft-token-budget-v1"
                and budget.get("max_prompt_tokens") == 3072
                and budget.get("max_completion_tokens") == 1024
                and budget.get("max_total_tokens") == 4096
                and budget.get("overflow_policy") == "error"
                and budget.get("truncated") is False
                and budget_is_self_consistent,
                f"{label} acquisition token-budget observation is invalid",
            )
            terminal_artifacts[sample_id] = {
                "sft_row": dict(sft),
                "manifest_row": dict(manifest),
                "exclusion": None,
            }

    for index, exclusion in enumerate(exclusions):
        label = f"generation exclusions[{index}]"
        sample_id = str(exclusion.get("sample_id") or "")
        prepared = prepared_by_id.get(sample_id)
        _require(prepared is not None, f"{label} is not in the prepared population")
        _require(sample_id not in terminal_artifacts, f"{label} is duplicated")
        expected_prepared_sha = sha256_text(canonical_json(prepared))
        _require(
            all(
                exclusion.get(field) == prepared.get(field)
                for field in ("split", "meeting_date", "atomic_topic")
            )
            and exclusion.get("prepared_row_sha256") == expected_prepared_sha,
            f"{label} prepared row binding mismatch",
        )
        stage = str(exclusion.get("stage") or "")
        try:
            inputs = generation._execution_inputs(
                prepared, model_input_projector=model_input_projector
            )
        except (generation.PreparedDataError, TypeError, ValueError):
            _require(
                stage == "prepared_validation",
                f"{label} no longer has a valid prepared projection",
            )
            inputs = None
        else:
            _require(
                stage != "prepared_validation",
                f"{label} claims prepared-validation failure but projection passes",
            )
            _require(
                inputs["prepared_row_sha256"] == expected_prepared_sha
                and inputs["student_tokenizer_sha256"]
                == prepared["prompt_budget"]["student"]["tokenizer_sha256"]
                and inputs["generator_tokenizer_sha256"]
                == prepared["prompt_budget"]["generator"]["tokenizer_sha256"],
                f"{label} projected prepared binding mismatch",
            )
        projected_inputs[sample_id] = inputs
        terminal_artifacts[sample_id] = {
            "sft_row": None,
            "manifest_row": None,
            "exclusion": dict(exclusion),
        }
    return prepared_by_id, terminal_artifacts, projected_inputs


def _replay_sample_caches(
    *,
    generation_root: Path,
    cache_manifest: Mapping[str, Any],
    prepared_by_id: Mapping[str, Mapping[str, Any]],
    terminal_artifacts: Mapping[str, Mapping[str, Any]],
    projected_inputs: Mapping[str, Mapping[str, Any] | None],
    generation_provenance_sha256: str,
    preparation_binding_sha256: str,
) -> None:
    """Prove every self-hashed cache payload equals its published terminal."""

    _require(
        cache_manifest.get("generation_provenance_sha256")
        == generation_provenance_sha256
        and cache_manifest.get("preparation_binding_sha256")
        == preparation_binding_sha256,
        "sample cache manifest provenance mismatch",
    )
    entries = cache_manifest.get("entries")
    _require(isinstance(entries, list), "sample cache manifest entries are invalid")
    observed: set[str] = set()
    expected_payload_keys = {
        "schema_version",
        "sample_id",
        "mode",
        "prepared_row_sha256",
        "minutes_reference_sha256",
        "generation_provenance_sha256",
        "preparation_binding_sha256",
        "artifact",
        "payload_sha256",
    }
    for index, entry in enumerate(entries):
        _require(isinstance(entry, Mapping), f"sample cache entry {index} is invalid")
        sample_id = str(entry.get("sample_id") or "")
        _require(
            sample_id in terminal_artifacts and sample_id not in observed,
            f"sample cache entry {index} sample identity is invalid",
        )
        observed.add(sample_id)
        relative = Path(str(entry.get("path") or ""))
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            f"sample cache entry {index} path is unsafe",
        )
        payload = _validate_payload_hash(
            _load_json(generation_root / relative, label=f"sample cache {sample_id}"),
            schema_version=generation.SAMPLE_CACHE_SCHEMA_VERSION,
            label=f"sample cache {sample_id}",
        )
        _require(
            set(payload) == expected_payload_keys,
            f"sample cache {sample_id} payload keys are not canonical",
        )
        prepared = prepared_by_id[sample_id]
        _require(
            payload.get("sample_id") == sample_id
            and payload.get("mode") == "full"
            and payload.get("prepared_row_sha256")
            == sha256_text(canonical_json(dict(prepared)))
            and payload.get("generation_provenance_sha256")
            == generation_provenance_sha256
            and payload.get("preparation_binding_sha256")
            == preparation_binding_sha256,
            f"sample cache {sample_id} identity/provenance mismatch",
        )
        terminal = terminal_artifacts[sample_id]
        _require(
            payload.get("artifact") == terminal,
            f"sample cache {sample_id} artifact differs from the published terminal",
        )
        projection = projected_inputs[sample_id]
        exclusion = terminal.get("exclusion")
        if projection is None:
            expected_minutes: str | None = None
        elif isinstance(exclusion, Mapping) and exclusion.get("stage") == "runtime_resolver":
            if exclusion.get("error_type") == "MinutesReferenceMismatch":
                observed_minutes = payload.get("minutes_reference_sha256")
                try:
                    validate_sha256(
                        observed_minutes,
                        label=f"sample cache {sample_id}.minutes_reference_sha256",
                    )
                except ValueError as exc:
                    raise CanonicalWorkflowError(str(exc)) from exc
                _require(
                    observed_minutes != projection["minutes_reference_sha256"],
                    f"sample cache {sample_id} resolver mismatch is self-contradictory",
                )
                expected_minutes = str(observed_minutes)
            else:
                expected_minutes = None
        else:
            expected_minutes = str(projection["minutes_reference_sha256"])
        _require(
            payload.get("minutes_reference_sha256") == expected_minutes,
            f"sample cache {sample_id} Minutes-reference binding mismatch",
        )
    _require(
        observed == set(terminal_artifacts),
        "sample caches do not cover every terminal exactly once",
    )


def verify_full_generation(
    *,
    repo_root: str | Path,
    generation_handoff: str | Path,
    prepare_handoff: str | Path,
) -> VerifiedGeneration:
    """Replay every preparation/generation binding without executing a model."""

    root = Path(repo_root).resolve()
    imported_code_root = Path(generation.__file__).resolve().parents[3]
    _require(
        imported_code_root == root,
        "imported generation code does not come from the declared repository root",
    )
    generation_path = _require_inside_no_symlink(
        root, Path(generation_handoff), label="generation handoff"
    )
    prepare_path = _require_inside_no_symlink(
        root, Path(prepare_handoff), label="preparation handoff"
    )
    _require(generation_path.name == "generation_handoff.json", "generation input must be generation_handoff.json")
    _require(prepare_path.name == "prepare_handoff.json", "preparation input must be prepare_handoff.json")
    raw_handoff = _validate_payload_hash(
        _load_json(generation_path, label="generation handoff"),
        schema_version=generation.GENERATION_HANDOFF_SCHEMA_VERSION,
        label="generation handoff",
    )
    _require(raw_handoff.get("status") == "complete", "generation handoff is incomplete")
    _require(raw_handoff.get("mode") == "full", "canonical publication accepts only mode=full generation")

    raw_prepare = _validate_payload_hash(
        _load_json(prepare_path, label="preparation handoff"),
        schema_version=generation.PREPARE_HANDOFF_SCHEMA_VERSION,
        label="preparation handoff",
    )
    preparation_files = _preparation_file_records(
        repo_root=root, handoff_path=prepare_path, handoff=raw_prepare
    )
    population, preparation_binding = generation._load_prepare_bundle(prepare_path)
    current_provenance = generation.build_generation_provenance(repo_root=root)
    # This loads tokenizer metadata on CPU only; it does not instantiate either
    # model or issue a generation call.
    model_input_projector, sft_token_auditor = (
        generation._build_model_input_contracts(repo_root=root)
    )
    selected = generation.select_generation_rows(population, mode="full")
    expected_selection = generation._selection_manifest(
        population,
        selected,
        mode="full",
        generation_provenance_sha256=str(current_provenance["payload_sha256"]),
        preparation_binding_sha256=str(preparation_binding["binding_sha256"]),
    )
    _require(
        len(selected) == len(population)
        and raw_handoff.get("selected_count") == len(population),
        "full generation does not cover every prepared row",
    )
    generation._validate_complete_handoff(
        generation_path,
        output_dir=generation_path.parent,
        expected_selection=expected_selection,
        expected_generation_provenance=current_provenance,
        expected_preparation_binding=preparation_binding,
        expected_sft_token_auditor=sft_token_auditor,
    )
    generation_files, selection, cache_manifest = _generation_file_records(
        repo_root=root,
        handoff_path=generation_path,
        handoff=raw_handoff,
    )
    generation_files = tuple(
        sorted(
            (
                *generation_files,
                *_generation_code_files(
                    repo_root=root, generation_provenance=current_provenance
                ),
            ),
            key=lambda item: item.source_relative,
        )
    )
    _require(selection == expected_selection, "selection manifest replay mismatch")
    _require(raw_handoff.get("preparation_binding") == preparation_binding, "generation/preparation binding mismatch")
    _require(raw_handoff.get("generation_provenance") == current_provenance, "generation provenance replay mismatch")
    sft_rows, manifest_rows, exclusions = _read_generation_rows(
        generation_path.parent, raw_handoff
    )
    prepared_by_id, terminal_artifacts, projected_inputs = (
        _replay_projected_terminals(
            population=population,
            sft_rows=sft_rows,
            manifest_rows=manifest_rows,
            exclusions=exclusions,
            model_input_projector=model_input_projector,
            preparation_binding_sha256=str(preparation_binding["binding_sha256"]),
        )
    )
    terminal_ids = [
        str(row.get("sample_id") or "")
        for split in SPLITS
        for row in manifest_rows[split]
    ] + [str(row.get("sample_id") or "") for row in exclusions]
    selected_ids = [str(item) for item in selection["selected_sample_ids"]]
    _require(
        len(terminal_ids) == len(selected_ids)
        and len(set(terminal_ids)) == len(terminal_ids)
        and set(terminal_ids) == set(selected_ids),
        "generation terminals do not cover the exact selection once",
    )
    _require(
        set(terminal_artifacts) == set(selected_ids),
        "projected terminal replay does not cover the exact selection",
    )
    _replay_sample_caches(
        generation_root=generation_path.parent,
        cache_manifest=cache_manifest,
        prepared_by_id=prepared_by_id,
        terminal_artifacts=terminal_artifacts,
        projected_inputs=projected_inputs,
        generation_provenance_sha256=str(current_provenance["payload_sha256"]),
        preparation_binding_sha256=str(preparation_binding["binding_sha256"]),
    )
    return VerifiedGeneration(
        repo_root=root,
        generation_root=generation_path.parent,
        generation_handoff_path=generation_path,
        generation_handoff=raw_handoff,
        generation_handoff_sha256=sha256_file(generation_path),
        preparation_root=prepare_path.parent,
        prepare_handoff_path=prepare_path,
        prepare_handoff=raw_prepare,
        prepare_handoff_sha256=sha256_file(prepare_path),
        preparation_binding=dict(preparation_binding),
        selection=selection,
        cache_manifest=cache_manifest,
        population=tuple(dict(row) for row in population),
        sft_rows=sft_rows,
        manifest_rows=manifest_rows,
        exclusions=exclusions,
        generation_files=generation_files,
        preparation_files=preparation_files,
    )


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _release_terminals(
    verified: VerifiedGeneration,
) -> tuple[
    dict[str, tuple[dict[str, Any], ...]],
    dict[str, tuple[dict[str, Any], ...]],
    tuple[dict[str, Any], ...],
]:
    """Apply deterministic training-admission checks without mutating generation."""

    admitted_sft: dict[str, list[dict[str, Any]]] = {
        split: [] for split in SPLITS
    }
    admitted_manifests: dict[str, list[dict[str, Any]]] = {
        split: [] for split in SPLITS
    }
    exclusions = [dict(row) for row in verified.exclusions]
    for split in SPLITS:
        for sft, manifest in zip(
            verified.sft_rows[split],
            verified.manifest_rows[split],
            strict=True,
        ):
            parts = str(sft.get("response") or "").split("\n</think>\n")
            if len(parts) == 2 and all(part.strip() for part in parts):
                admitted_sft[split].append(dict(sft))
                admitted_manifests[split].append(dict(manifest))
                continue
            empty_parts = [
                name
                for name, part in zip(("reasoning", "final_analysis"), parts)
                if not part.strip()
            ]
            exclusions.append(
                {
                    "schema_version": generation.EXCLUSION_SCHEMA_VERSION,
                    "sample_id": str(manifest["sample_id"]),
                    "split": split,
                    "meeting_date": str(manifest["meeting_date"]),
                    "atomic_topic": str(manifest["atomic_topic"]),
                    "stage": "canonical_admission",
                    "reason_code": "empty_response_component",
                    "error_type": "EmptyResponseComponent",
                    "error_codes": [
                        f"empty_{name}" for name in empty_parts
                    ] or ["malformed_reasoning_boundary"],
                    "prepared_row_sha256": str(
                        manifest["generation"]["prepared_row_sha256"]
                    ),
                }
            )
    return (
        {split: tuple(admitted_sft[split]) for split in SPLITS},
        {split: tuple(admitted_manifests[split]) for split in SPLITS},
        tuple(exclusions),
    )


def _derive_quality_thresholds(verified: VerifiedGeneration) -> QualityThresholds:
    sample_ids: dict[str, list[str]] = {split: [] for split in SPLITS}
    meetings: dict[str, set[str]] = {split: set() for split in SPLITS}
    topics: set[str] = set()
    for index, row in enumerate(verified.population):
        label = f"prepared population row {index}"
        split = str(row.get("split") or "")
        sample_id = str(row.get("sample_id") or "")
        meeting = str(row.get("meeting_date") or "")
        topic = str(row.get("atomic_topic") or "")
        _require(split in SPLITS and sample_id and meeting and topic, f"{label} identity is invalid")
        sample_ids[split].append(sample_id)
        meetings[split].add(meeting)
        topics.add(topic)
    selected = {str(item) for item in verified.selection["selected_sample_ids"]}
    population_ids = {sample_id for values in sample_ids.values() for sample_id in values}
    _require(selected == population_ids and len(selected) == len(verified.population), "selection/population identity mismatch")
    return QualityThresholds(
        expected_meetings={split: tuple(sorted(meetings[split])) for split in SPLITS},
        expected_sample_ids={split: tuple(sorted(sample_ids[split])) for split in SPLITS},
        expected_topics=tuple(sorted(topics)),
        train_acceptance_rate=0.70,
        per_topic_acceptance_rate=0.50,
    )


def _copy_bound_file(source: BoundFile, release_dir: Path) -> None:
    _require(source.source_path.is_file() and not source.source_path.is_symlink(), f"source artifact changed: {source.source_path}")
    _require(sha256_file(source.source_path) == source.sha256, f"source artifact SHA changed: {source.source_relative}")
    destination = release_dir / source.release_path
    _require(not destination.exists() and not destination.is_symlink(), f"duplicate staged artifact: {source.release_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.source_path.open("rb") as input_handle, destination.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
        output_handle.flush()
        os.fsync(output_handle.fileno())
    _require(destination.stat().st_size == source.bytes, f"staged artifact byte mismatch: {source.release_path}")
    _require(sha256_file(destination) == source.sha256, f"staged artifact SHA mismatch: {source.release_path}")
    if source.rows is not None and destination.suffix == ".jsonl":
        _require(_count_jsonl(destination) == source.rows, f"staged artifact row mismatch: {source.release_path}")


def _bundle_entry(bound: BoundFile) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "source_relative_path": bound.source_relative,
        "release_path": bound.release_path,
        "sha256": bound.sha256,
        "bytes": bound.bytes,
    }
    if bound.rows is not None:
        entry["rows"] = bound.rows
    return entry


def _build_bundle_manifests(
    verified: VerifiedGeneration,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generation_bundle = _payload_with_hash(
        {
            "schema_version": GENERATION_BUNDLE_SCHEMA_VERSION,
            "generation_handoff_sha256": verified.generation_handoff_sha256,
            "generation_handoff_payload_sha256": verified.generation_handoff["payload_sha256"],
            "generation_provenance_sha256": verified.generation_handoff[
                "generation_provenance"
            ]["payload_sha256"],
            "generation_code_payload_sha256": verified.generation_handoff[
                "generation_provenance"
            ]["generation_code"]["payload_sha256"],
            "preparation_binding_sha256": verified.preparation_binding["binding_sha256"],
            "selected_sample_ids_sha256": verified.selection["selected_sample_ids_sha256"],
            "files": [_bundle_entry(item) for item in verified.generation_files],
        }
    )
    preparation_bundle = _payload_with_hash(
        {
            "schema_version": PREPARATION_BUNDLE_SCHEMA_VERSION,
            "prepare_handoff_sha256": verified.prepare_handoff_sha256,
            "prepare_handoff_payload_sha256": verified.prepare_handoff["payload_sha256"],
            "preparation_binding_sha256": verified.preparation_binding["binding_sha256"],
            "files": [_bundle_entry(item) for item in verified.preparation_files],
        }
    )
    return generation_bundle, preparation_bundle


def _record_for_release_file(
    release_dir: Path, relative: str, *, rows: int | None = None
) -> dict[str, Any]:
    path = release_dir / relative
    record: dict[str, Any] = {
        "path": relative,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _production_identity(
    verified: VerifiedGeneration, *, release_id: str
) -> str:
    return sha256_text(
        canonical_json(
            {
                "schema_version": SOURCE_CONTRACT_SCHEMA_VERSION,
                "admission_schema_version": CANONICAL_ADMISSION_SCHEMA_VERSION,
                "release_id": release_id,
                "generation_handoff_sha256": verified.generation_handoff_sha256,
                "prepare_handoff_sha256": verified.prepare_handoff_sha256,
                "preparation_binding_sha256": verified.preparation_binding[
                    "binding_sha256"
                ],
            }
        )
    )


def _write_source_contracts(
    release_dir: Path,
    *,
    verified: VerifiedGeneration,
    production_identity_sha256: str,
) -> None:
    for bound in (*verified.generation_files, *verified.preparation_files):
        _copy_bound_file(bound, release_dir)
    generation_bundle, preparation_bundle = _build_bundle_manifests(verified)
    generation_bundle_path = release_dir / "source/generation_bundle_manifest.json"
    preparation_bundle_path = release_dir / "source/preparation_bundle_manifest.json"
    _atomic_write(generation_bundle_path, _json_text(generation_bundle).encode("utf-8"))
    _atomic_write(preparation_bundle_path, _json_text(preparation_bundle).encode("utf-8"))

    generation_by_relative = {
        item.source_relative: item for item in verified.generation_files
    }
    preparation_by_relative = {
        item.source_relative: item for item in verified.preparation_files
    }
    handoff_path = release_dir / "handoff.json"
    handoff = _load_json(handoff_path, label="staged canonical handoff")
    handoff["production_identity_sha256"] = production_identity_sha256
    handoff["source_generation"] = {
        "schema_version": SOURCE_CONTRACT_SCHEMA_VERSION,
        "generation_handoff": _record_for_release_file(
            release_dir, generation_by_relative["generation_handoff.json"].release_path
        ),
        "prepare_handoff": _record_for_release_file(
            release_dir, preparation_by_relative["prepare_handoff.json"].release_path
        ),
        "selection_manifest": _record_for_release_file(
            release_dir, generation_by_relative["selection_manifest.json"].release_path
        ),
        "cache_manifest": _record_for_release_file(
            release_dir,
            generation_by_relative["cache/cache_manifest.json"].release_path,
        ),
        "generation_bundle_manifest": _record_for_release_file(
            release_dir, "source/generation_bundle_manifest.json"
        ),
        "preparation_bundle_manifest": _record_for_release_file(
            release_dir, "source/preparation_bundle_manifest.json"
        ),
        "generation_provenance_sha256": verified.generation_handoff[
            "generation_provenance"
        ]["payload_sha256"],
        "generation_code_payload_sha256": verified.generation_handoff[
            "generation_provenance"
        ]["generation_code"]["payload_sha256"],
        "preparation_binding_sha256": verified.preparation_binding["binding_sha256"],
        "selected_sample_ids_sha256": verified.selection[
            "selected_sample_ids_sha256"
        ],
    }
    _atomic_write(handoff_path, _json_text(handoff).encode("utf-8"))


def _bundle_manifest_record(
    destination: Path, record: Any, *, schema_version: str, label: str
) -> dict[str, Any]:
    _require(isinstance(record, Mapping), f"{label} record is missing")
    relative = Path(str(record.get("path") or ""))
    _require(not relative.is_absolute() and ".." not in relative.parts, f"{label} record path is unsafe")
    path = destination / relative
    _require(path.is_file() and not path.is_symlink(), f"{label} file is missing")
    _require(sha256_file(path) == record.get("sha256"), f"{label} file SHA mismatch")
    return _validate_payload_hash(
        _load_json(path, label=label), schema_version=schema_version, label=label
    )


def _verify_bundle_entries(destination: Path, manifest: Mapping[str, Any], *, label: str) -> None:
    entries = manifest.get("files")
    _require(isinstance(entries, list) and entries, f"{label} entries are empty")
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        _require(isinstance(entry, Mapping), f"{label} entry {index} is invalid")
        relative = str(entry.get("release_path") or "")
        path_value = Path(relative)
        _require(relative and not path_value.is_absolute() and ".." not in path_value.parts, f"{label} entry {index} path is unsafe")
        _require(relative not in seen, f"{label} has duplicate release paths")
        seen.add(relative)
        path = destination / path_value
        _require(path.is_file() and not path.is_symlink(), f"{label} artifact is missing: {relative}")
        _require(path.stat().st_size == entry.get("bytes"), f"{label} byte mismatch: {relative}")
        _require(sha256_file(path) == entry.get("sha256"), f"{label} SHA mismatch: {relative}")
        if "rows" in entry and path.suffix == ".jsonl":
            _require(_count_jsonl(path) == entry["rows"], f"{label} row mismatch: {relative}")


def _tree_has_no_symlinks(root: Path) -> None:
    _require(not root.is_symlink(), f"published release is a symlink: {root}")
    for path in root.rglob("*"):
        _require(not path.is_symlink(), f"published release contains a symlink: {path}")


def _tree_is_sealed(root: Path) -> bool:
    writable = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    return not (root.stat().st_mode & writable) and all(
        not (path.stat().st_mode & writable)
        for path in root.rglob("*")
        if not path.is_symlink()
    )


def _verify_existing_release(
    destination: Path, *, production_identity_sha256: str
) -> Path:
    _require(destination.is_dir() and not destination.is_symlink(), "canonical destination is not a real directory")
    _tree_has_no_symlinks(destination)
    handoff = _load_json(destination / "handoff.json", label="existing canonical handoff")
    _require(
        handoff.get("production_identity_sha256") == production_identity_sha256,
        "immutable canonical release ID already exists with different inputs",
    )
    source = handoff.get("source_generation")
    _require(isinstance(source, Mapping) and source.get("schema_version") == SOURCE_CONTRACT_SCHEMA_VERSION, "existing source_generation contract is invalid")
    generation_bundle = _bundle_manifest_record(
        destination,
        source.get("generation_bundle_manifest"),
        schema_version=GENERATION_BUNDLE_SCHEMA_VERSION,
        label="generation bundle manifest",
    )
    preparation_bundle = _bundle_manifest_record(
        destination,
        source.get("preparation_bundle_manifest"),
        schema_version=PREPARATION_BUNDLE_SCHEMA_VERSION,
        label="preparation bundle manifest",
    )
    _verify_bundle_entries(destination, generation_bundle, label="generation bundle")
    _verify_bundle_entries(destination, preparation_bundle, label="preparation bundle")
    for label, record in (
        ("quality report", handoff.get("quality_report")),
        ("legacy inventory", handoff.get("legacy_inventory")),
    ):
        _require(isinstance(record, Mapping), f"existing {label} record is invalid")
        relative = Path(str(record.get("path") or ""))
        _require(not relative.is_absolute() and ".." not in relative.parts, f"existing {label} path is unsafe")
        path = destination / relative
        _require(path.is_file() and not path.is_symlink(), f"existing {label} is missing")
        _require(sha256_file(path) == record.get("sha256"), f"existing {label} SHA mismatch")
        if "rows" in record and path.suffix == ".jsonl":
            _require(_count_jsonl(path) == record["rows"], f"existing {label} row mismatch")
    for group in ("split_files", "manifest_files"):
        records = handoff.get(group)
        _require(isinstance(records, Mapping) and set(records) == set(SPLITS), f"existing {group} is invalid")
        for split in SPLITS:
            record = records[split]
            path = destination / str(record.get("path") or "")
            _require(path.is_file() and not path.is_symlink(), f"existing {group}.{split} is missing")
            _require(sha256_file(path) == record.get("sha256"), f"existing {group}.{split} SHA mismatch")
            _require(_count_jsonl(path) == record.get("rows"), f"existing {group}.{split} row mismatch")
    return destination / "handoff.json"


def _seal_tree(root: Path, *, seal_root: bool = True) -> None:
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
        path.chmod(
            stat.S_IRUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH
        )
    if seal_root:
        root.chmod(
            stat.S_IRUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(path)
    _fsync_directory(root)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish without ever clobbering a raced destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(
        renameat2 is not None,
        "atomic renameat2(RENAME_NOREPLACE) is unavailable",
    )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise CanonicalWorkflowError(
            f"immutable canonical destination already exists: {destination}"
        )
    raise CanonicalWorkflowError(
        "atomic no-overwrite canonical publication failed: "
        f"{destination}: {os.strerror(error)}"
    )


def _remove_owned_staging(path: Path) -> None:
    for item in path.rglob("*"):
        if not item.is_symlink():
            try:
                item.chmod(stat.S_IRWXU)
            except OSError:
                pass
    path.chmod(stat.S_IRWXU)
    shutil.rmtree(path)


def _recover_stale_builds(
    release_root: Path, *, release_id: str, production_identity_sha256: str
) -> None:
    for stale in sorted(release_root.glob(f".{release_id}.canonical-build-*")):
        _require(stale.is_dir() and not stale.is_symlink(), f"unsafe stale canonical staging path: {stale}")
        _tree_has_no_symlinks(stale)
        marker = _load_json(stale / "build_identity.json", label="stale build identity")
        _require(
            marker
            == {
                "release_id": release_id,
                "production_identity_sha256": production_identity_sha256,
            },
            f"stale canonical staging belongs to different immutable inputs: {stale}",
        )
        _remove_owned_staging(stale)


def publish_canonical_release(
    *,
    repo_root: str | Path,
    generation_handoff: str | Path,
    prepare_handoff: str | Path,
    release_root: str | Path,
    release_id: str,
) -> Path:
    """Atomically publish an immutable release from already completed inputs."""

    verified = verify_full_generation(
        repo_root=repo_root,
        generation_handoff=generation_handoff,
        prepare_handoff=prepare_handoff,
    )
    admitted_sft, admitted_manifests, admitted_exclusions = _release_terminals(
        verified
    )
    identity = _production_identity(verified, release_id=release_id)
    root = _require_inside_no_symlink(
        verified.repo_root, Path(release_root), label="canonical release root"
    )
    root.mkdir(parents=True, exist_ok=True)
    _require_inside_no_symlink(
        verified.repo_root, root, label="canonical release root"
    )
    destination = root / release_id
    _recover_stale_builds(
        root,
        release_id=release_id,
        production_identity_sha256=identity,
    )
    if destination.exists() or destination.is_symlink():
        existing = _verify_existing_release(
            destination, production_identity_sha256=identity
        )
        if not _tree_is_sealed(destination):
            _seal_tree(destination)
            _fsync_tree(destination)
            _fsync_directory(root)
        return existing

    build = Path(
        tempfile.mkdtemp(prefix=f".{release_id}.canonical-build-", dir=root)
    )
    marker = {
        "release_id": release_id,
        "production_identity_sha256": identity,
    }
    _atomic_write(build / "build_identity.json", _json_text(marker).encode("utf-8"))
    try:
        inventory_record = verified.prepare_handoff["artifacts"]["inventory"]
        inventory_path, _, _, _ = _safe_record_path(
            verified.preparation_root,
            inventory_record,
            repo_root=verified.repo_root,
            label="preparation inventory",
            require_rows=False,
        )
        legacy_inventory = _load_json(inventory_path, label="preparation inventory")
        provenance = verified.generation_handoff["generation_provenance"]
        core_root = build / "core"
        core_handoff = publish_release(
            release_root=core_root,
            release_id=release_id,
            sft_rows=admitted_sft,
            manifest_rows=admitted_manifests,
            exclusions=admitted_exclusions,
            thresholds=_derive_quality_thresholds(verified),
            prompt_template_sha256=str(provenance["prompt_template_sha256"]),
            style_guide_sha256=str(verified.prepare_handoff["style_guide_sha256"]),
            teacher_model_sha256=str(provenance["teacher_model"]["sha256"]),
            tokenizer_sha256=str(provenance["student_tokenizer"]["sha256"]),
            legacy_inventory=legacy_inventory,
            seal_permissions=False,
        )
        staged_release = core_handoff.parent
        _write_source_contracts(
            staged_release,
            verified=verified,
            production_identity_sha256=identity,
        )
        _verify_existing_release(
            staged_release, production_identity_sha256=identity
        )
        _fsync_tree(staged_release)
        # Keep only the staging root writable until rename; every file and
        # descendant directory is already sealed at this point.
        _seal_tree(staged_release, seal_root=False)
        _fsync_directory(staged_release.parent)
        _require(not destination.exists() and not destination.is_symlink(), "canonical destination appeared during publication")
        _rename_noreplace(staged_release, destination)
        _seal_tree(destination)
        _fsync_directory(root)
        result = _verify_existing_release(
            destination, production_identity_sha256=identity
        )
        _remove_owned_staging(build)
        return result
    except BaseException:
        if build.exists() and not build.is_symlink():
            _remove_owned_staging(build)
        raise


def _resolved(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else root / value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    publish = commands.add_parser("publish")
    publish.add_argument(
        "--generation-handoff", type=Path, default=FULL_GENERATION_HANDOFF
    )
    publish.add_argument("--prepare-handoff", type=Path, default=PREPARE_HANDOFF)
    publish.add_argument("--release-id", required=True)
    publish.add_argument("--release-root", type=Path, default=RELEASE_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repo_root.resolve()
    try:
        handoff = publish_canonical_release(
            repo_root=root,
            generation_handoff=_resolved(root, args.generation_handoff),
            prepare_handoff=_resolved(root, args.prepare_handoff),
            release_root=_resolved(root, args.release_root),
            release_id=args.release_id,
        )
        result = {
            "status": "published",
            "release_id": args.release_id,
            "handoff": str(handoff),
            "handoff_sha256": sha256_file(handoff),
        }
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
