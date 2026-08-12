"""Generate one leg of the provenance-aware Chapter 2 checkpoint matrix."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from generate_new_response import ROW_SEED_POLICY_SAMPLE, generate_new_response
from open_r1.provenance import (
    fingerprint_artifact_path,
    sha256_file,
    sha256_text,
    validate_sha256,
)
from open_r1.structured_response import PLAIN_TEXT_FORMAT, parse_structured_response
from open_r1.validator.loo_generation_spec import (
    derive_row_seed,
    seal_manifest,
    validate_manifest_integrity,
)


SCHEMA_VERSION = "checkpoint-artifact-generations-v1"
LEGACY_MANIFEST_SCHEMA_VERSION = "checkpoint-artifact-generation-manifest-v1"
MANIFEST_SCHEMA_VERSION = "checkpoint-artifact-generation-manifest-v2"
PROGRESS_BINDING_SCHEMA_VERSION = (
    "checkpoint-artifact-generation-progress-binding-v1"
)
PROGRESS_SCHEMA_VERSION = "checkpoint-artifact-generation-progress-v1"
PROGRESS_ROW_SCHEMA_VERSION = "checkpoint-artifact-generation-progress-row-v1"
PROGRESS_CONTRACT_SCHEMA_VERSION = (
    "checkpoint-artifact-generation-progress-contract-v1"
)
SAFE_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _capture_inference_environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in (
        "torch",
        "transformers",
        "vllm",
        "accelerate",
        "tokenizers",
        "safetensors",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    cuda: dict[str, Any] = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "available": False,
        "device_count": 0,
        "torch_cuda_version": None,
        "cudnn_version": None,
        "devices": [],
    }
    try:
        import torch

        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        cuda.update(
            {
                "available": available,
                "device_count": count,
                "torch_cuda_version": torch.version.cuda,
                "cudnn_version": (
                    int(torch.backends.cudnn.version())
                    if torch.backends.cudnn.version() is not None
                    else None
                ),
                "devices": [
                    {
                        "logical_index": index,
                        "name": torch.cuda.get_device_name(index),
                        "capability": list(torch.cuda.get_device_capability(index)),
                        "total_memory_bytes": int(
                            torch.cuda.get_device_properties(index).total_memory
                        ),
                    }
                    for index in range(count)
                ],
            }
        )
    except (ImportError, RuntimeError, AttributeError):
        pass

    return {
        "schema_version": "checkpoint-inference-environment-v1",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
        "cuda": cuda,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _validate_artifact_id(artifact_id: object) -> str:
    value = str(artifact_id or "")
    if not SAFE_ARTIFACT_ID_RE.fullmatch(value):
        raise ValueError(
            "artifact_id must be a 1-128 character safe slug containing only "
            "ASCII letters, digits, dot, underscore, or hyphen, and must start "
            "with a letter or digit"
        )
    return value


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{label} row {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def _read_jsonl_bytes(content: bytes, *, label: str) -> list[dict[str, Any]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} row {line_number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} row {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def _atomic_write(path: Path, content: str) -> None:
    _durable_mkdir(path.parent)
    if path.exists():
        raise FileExistsError(f"Immutable generation artifact exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_mkdir(path: Path) -> None:
    """Create a directory chain and durably commit every new directory entry."""

    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    if not cursor.is_dir():
        raise NotADirectoryError(f"Directory ancestor is not a directory: {cursor}")
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise
        _fsync_directory(directory)
        _fsync_directory(directory.parent)


def _atomic_replace(path: Path, content: str) -> None:
    """Durably replace a mutable receipt without exposing a partial file."""

    _durable_mkdir(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


@contextmanager
def _exclusive_progress_lock(path: Path) -> Iterator[None]:
    """Serialize writers while keeping a stable lock inode across runs."""

    _durable_mkdir(path.parent)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another generation process holds the progress lock: {path}"
            ) from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(f"{_canonical_json(row)}\n" for row in rows),
    )


def _artifact_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = manifest.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError("Checkpoint manifest requires a non-empty artifacts list")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Checkpoint manifest artifacts must be objects")
    return records


def select_artifact(
    manifest: dict[str, Any],
    artifact_id: str,
) -> dict[str, Any]:
    artifact_id = _validate_artifact_id(artifact_id)
    matches = [
        record
        for record in _artifact_records(manifest)
        if str(record.get("artifact_id") or "") == artifact_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one checkpoint artifact {artifact_id!r}, "
            f"found {len(matches)}"
        )
    record = matches[0]
    if record.get("usable_for_evaluation") is not True:
        raise ValueError(f"Artifact {artifact_id!r} is not usable for evaluation")
    return record


def _resolve_artifact_path(
    value: object,
    *,
    manifest_path: Path,
    label: str,
) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is empty")
    raw = Path(text).expanduser()
    candidates = (
        [raw]
        if raw.is_absolute()
        else [manifest_path.parent / raw, Path.cwd() / raw]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"{label} does not exist: {value}")


def _validate_fingerprint(
    path: Path,
    expected: object,
    *,
    label: str,
) -> dict[str, Any]:
    expected_hash = validate_sha256(expected, label=f"{label} SHA-256")
    observed = fingerprint_artifact_path(path)
    if observed["sha256"] != expected_hash:
        raise ValueError(
            f"{label} fingerprint mismatch: expected={expected_hash}, "
            f"observed={observed['sha256']}"
        )
    return observed


def _final_answer(generated: str) -> tuple[str, str, bool]:
    parsed = parse_structured_response(generated)
    if parsed.is_well_formed:
        return parsed.answer.strip(), parsed.format_name, True
    if parsed.format_name == PLAIN_TEXT_FORMAT and parsed.answer.strip():
        return parsed.answer.strip(), parsed.format_name, True
    return "", parsed.format_name, False


def finalise_generation_rows(
    expected_rows: list[dict[str, Any]],
    generated_rows: list[dict[str, Any]],
    *,
    artifact: dict[str, Any],
    model_fingerprint: dict[str, Any],
    tokenizer_fingerprint: dict[str, Any],
    shared_provenance: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    expected_ids = [str(row.get("sample_id") or "") for row in expected_rows]
    if any(not sample_id for sample_id in expected_ids):
        raise ValueError("Every prompt requires a non-empty sample_id")
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Prompt sample IDs are not unique")

    by_sample: dict[str, dict[str, Any]] = {}
    for row in generated_rows:
        sample_id = str(row.get("sample_id") or "")
        if sample_id not in set(expected_ids):
            raise ValueError(f"Generation contains unknown sample_id {sample_id!r}")
        if sample_id in by_sample:
            raise ValueError(f"Generation contains duplicate sample_id {sample_id!r}")
        by_sample[sample_id] = row

    output: list[dict[str, Any]] = []
    for prompt_row in expected_rows:
        sample_id = str(prompt_row["sample_id"])
        source = by_sample.get(sample_id)
        expected_seed: int | None = None
        if shared_provenance and (
            shared_provenance.get("generation_seed_policy")
            == ROW_SEED_POLICY_SAMPLE
        ):
            expected_seed = derive_row_seed(
                int(shared_provenance["generation_base_seed"]),
                sample_id,
            )
        if source is None:
            source = dict(prompt_row)
            source.update(
                {
                    "generated": "",
                    "generated_sha256": sha256_text(""),
                    "generation_finish_reason": "missing_generation",
                    "generation_stop_reason": None,
                    "prompt_token_count": None,
                    "prompt_preflight_token_count": None,
                    "output_token_count": None,
                    "input_was_truncated": None,
                    "generation_seed": expected_seed,
                    "generation_seed_policy": (
                        ROW_SEED_POLICY_SAMPLE if expected_seed is not None else None
                    ),
                }
            )
        elif expected_seed is not None and (
            source.get("generation_seed_policy") != ROW_SEED_POLICY_SAMPLE
            or source.get("generation_seed") != expected_seed
        ):
            raise ValueError(
                f"{sample_id}: generation seed or seed policy is inconsistent"
            )
        generated = str(source.get("generated") or "")
        answer, response_format, parseable = _final_answer(generated)
        finish_reason = str(source.get("generation_finish_reason") or "")
        input_was_truncated = source.get("input_was_truncated")
        valid = bool(
            parseable
            and answer
            and finish_reason == "stop"
            and input_was_truncated is False
        )
        invalid_reasons: list[str] = []
        if not generated.strip():
            invalid_reasons.append("empty_or_missing_generation")
        if not parseable:
            invalid_reasons.append("unparseable_final_answer")
        if finish_reason != "stop":
            invalid_reasons.append(
                f"non_normal_finish:{finish_reason or 'missing'}"
            )
        if input_was_truncated is not False:
            invalid_reasons.append("input_truncation_not_proven_false")

        public = dict(source)
        public.update(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_id": artifact["artifact_id"],
                "design_checkpoint_id": artifact.get("design_checkpoint_id"),
                "intended_parent_id": artifact.get("intended_parent_id"),
                "verified_parent_artifact_id": artifact.get(
                    "verified_parent_artifact_id"
                ),
                "lineage_status": artifact.get("lineage_status"),
                "generation_model_sha256": model_fingerprint["sha256"],
                "generation_tokenizer_sha256": tokenizer_fingerprint["sha256"],
                "response_format": response_format,
                "final_answer": answer,
                "final_answer_sha256": sha256_text(answer),
                "valid_generation": valid,
                "invalid_reasons": invalid_reasons,
                **(shared_provenance or {}),
            }
        )
        output.append(public)
    return output


def validate_generation_artifact(
    *,
    checkpoint_manifest_file: str | Path,
    prompts_file: str | Path,
    config_file: str | Path,
    artifact_id: str,
    output_dir: str | Path,
    manifest_file: str | Path | None = None,
    allow_legacy_without_progress: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Validate and reuse one already committed immutable generation leg."""

    artifact_id = _validate_artifact_id(artifact_id)
    checkpoint_path = Path(checkpoint_manifest_file).expanduser().resolve()
    prompts_path = Path(prompts_file).expanduser().resolve()
    config_path = Path(config_file).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    output_path = destination / f"{artifact_id}.jsonl"
    manifest_path = (
        Path(manifest_file).expanduser().resolve()
        if manifest_file is not None
        else destination / f"{artifact_id}.manifest.json"
    )
    manifest = _read_json_object(manifest_path, label="generation manifest")
    validate_manifest_integrity(manifest)
    manifest_schema = manifest.get("schema_version")
    legacy_manifest = manifest_schema == LEGACY_MANIFEST_SCHEMA_VERSION
    if manifest.get("artifact_id") != artifact_id or manifest_schema not in {
        MANIFEST_SCHEMA_VERSION,
        LEGACY_MANIFEST_SCHEMA_VERSION,
    }:
        raise ValueError(
            f"Generation manifest does not describe artifact {artifact_id!r}"
        )
    if legacy_manifest and not allow_legacy_without_progress:
        raise ValueError(
            "Legacy generation manifest lacks a frozen progress binding; "
            "explicit versioned resealing is required"
        )

    checkpoint_manifest = _read_json_object(
        checkpoint_path, label="checkpoint manifest"
    )
    validate_manifest_integrity(checkpoint_manifest)
    artifact = select_artifact(checkpoint_manifest, artifact_id)
    config = _read_json_object(config_path, label="evaluation config")
    generation_config = config.get("generation")
    if not isinstance(generation_config, dict):
        raise ValueError("Evaluation config lacks generation settings")
    prompts = _read_jsonl(prompts_path, label="evaluation prompts")
    output_bytes = output_path.read_bytes()
    rows = _read_jsonl_bytes(output_bytes, label="generation output")
    expected_ids = [str(row.get("sample_id") or "") for row in prompts]
    observed_ids = [str(row.get("sample_id") or "") for row in rows]
    if (
        any(not sample_id for sample_id in expected_ids)
        or len(expected_ids) != len(set(expected_ids))
        or observed_ids != expected_ids
    ):
        raise ValueError("Generation output does not preserve the prompt matrix")

    bindings = (
        ("checkpoint_manifest", checkpoint_path),
        ("evaluation_config", config_path),
        ("prompts", prompts_path),
    )
    for label, path in bindings:
        record = manifest.get(label)
        if (
            not isinstance(record, dict)
            or record.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"Generation manifest {label} binding changed")
    output_record = manifest.get("output")
    if (
        not isinstance(output_record, dict)
        or output_record.get("sha256") != _sha256_bytes(output_bytes)
        or output_record.get("row_count") != len(rows)
    ):
        raise ValueError("Generation output hash or row count changed")

    model_path = _resolve_artifact_path(
        artifact.get("model_path"),
        manifest_path=checkpoint_path,
        label=f"{artifact_id} model",
    )
    tokenizer_path = _resolve_artifact_path(
        artifact.get("tokenizer_path"),
        manifest_path=checkpoint_path,
        label=f"{artifact_id} tokenizer",
    )
    model_fingerprint = _validate_fingerprint(
        model_path,
        artifact.get("model_sha256"),
        label=f"{artifact_id} model",
    )
    tokenizer_fingerprint = _validate_fingerprint(
        tokenizer_path,
        artifact.get("tokenizer_sha256"),
        label=f"{artifact_id} tokenizer",
    )
    if (
        manifest.get("model_artifact", {}).get("sha256")
        != model_fingerprint["sha256"]
        or manifest.get("tokenizer_artifact", {}).get("sha256")
        != tokenizer_fingerprint["sha256"]
    ):
        raise ValueError("Generation manifest model/tokenizer binding changed")

    prompt_by_id = {str(row["sample_id"]): row for row in prompts}
    prompts_sha256 = sha256_file(prompts_path)
    config_sha256 = sha256_file(config_path)
    base_seed = int(generation_config["base_seed"])
    seed_policy = str(generation_config.get("seed_policy") or "")
    for row in rows:
        sample_id = str(row["sample_id"])
        expected_prompt = prompt_by_id[sample_id]
        if (
            row.get("artifact_id") != artifact_id
            or row.get("prompt_sha256") != expected_prompt.get("prompt_sha256")
            or row.get("generation_model_sha256") != model_fingerprint["sha256"]
            or row.get("generation_tokenizer_sha256")
            != tokenizer_fingerprint["sha256"]
            or row.get("test_set_sha256") != prompts_sha256
            or row.get("prompt_template_sha256") != config_sha256
            or row.get("decoding_config_sha256") != config_sha256
            or row.get("temperature")
            != float(generation_config["temperature"])
            or row.get("top_p") != float(generation_config["top_p"])
            or row.get("max_new_tokens")
            != int(generation_config["max_new_tokens"])
            or row.get("max_model_len")
            != int(generation_config["max_model_len"])
            or row.get("generation_seed_policy") != seed_policy
            or row.get("generation_seed") != derive_row_seed(base_seed, sample_id)
            or not isinstance(row.get("valid_generation"), bool)
        ):
            raise ValueError(f"{sample_id}: generation provenance is inconsistent")
    manifest_environment = manifest.get("inference_environment")
    manifest_environment_sha256 = manifest.get("inference_environment_sha256")
    if manifest_environment is not None or manifest_environment_sha256 is not None:
        if (
            not isinstance(manifest_environment, dict)
            or manifest_environment_sha256
            != sha256_text(_canonical_json(manifest_environment))
        ):
            raise ValueError("Generation manifest inference environment is inconsistent")
        for row in rows:
            if (
                row.get("inference_environment") != manifest_environment
                or row.get("inference_environment_sha256")
                != manifest_environment_sha256
            ):
                raise ValueError(
                    f"{row['sample_id']}: inference environment binding changed"
                )
    valid_count = sum(bool(row["valid_generation"]) for row in rows)
    if (
        manifest.get("sample_count") != len(prompts)
        or manifest.get("valid_count") != valid_count
        or manifest.get("invalid_count") != len(rows) - valid_count
    ):
        raise ValueError("Generation manifest validity counts changed")
    if not legacy_manifest:
        _validate_progress_binding(
            manifest=manifest,
            manifest_path=manifest_path,
            output_path=output_path,
            prompts_path=prompts_path,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            prompts=prompts,
            output_bytes=output_bytes,
        )
    return manifest_path, manifest


def _progress_contract(
    *,
    checkpoint_path: Path,
    prompts_path: Path,
    config_path: Path,
    artifact: dict[str, Any],
    prompts: list[dict[str, Any]],
    generation: dict[str, Any],
    model_path: Path,
    tokenizer_path: Path,
    model_fingerprint: dict[str, Any],
    tokenizer_fingerprint: dict[str, Any],
    batch_size: int,
    inference_environment: dict[str, Any],
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if generation.get("seed_policy") != ROW_SEED_POLICY_SAMPLE:
        raise ValueError(
            "Checkpoint generation requires sample-id-sha256-v1 seed policy"
        )
    ordered_samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, row in enumerate(prompts):
        sample_id = str(row.get("sample_id") or "")
        prompt = str(row.get("prompt") or "")
        if not sample_id or sample_id in seen:
            raise ValueError("Prompt sample IDs must be non-empty and unique")
        seen.add(sample_id)
        prompt_sha256 = validate_sha256(
            row.get("prompt_sha256"),
            label=f"{sample_id} prompt SHA-256",
        )
        if prompt_sha256 != sha256_text(prompt):
            raise ValueError(f"{sample_id}: prompt SHA-256 does not match prompt text")
        ordered_samples.append(
            {
                "position": position,
                "sample_id": sample_id,
                "prompt_sha256": prompt_sha256,
                "input_row_sha256": sha256_text(_canonical_json(row)),
                "generation_seed": derive_row_seed(
                    int(generation["base_seed"]), sample_id
                ),
            }
        )

    return {
        "schema_version": PROGRESS_CONTRACT_SCHEMA_VERSION,
        "artifact_id": str(artifact["artifact_id"]),
        "checkpoint_manifest": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "evaluation_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "prompts": {
            "path": str(prompts_path),
            "sha256": sha256_file(prompts_path),
            "row_count": len(prompts),
            "ordered_samples": ordered_samples,
        },
        "selected_artifact_sha256": sha256_text(_canonical_json(artifact)),
        "model_artifact": model_fingerprint,
        "tokenizer_artifact": tokenizer_fingerprint,
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "generation": {
            "batch_size": batch_size,
            "base_seed": int(generation["base_seed"]),
            "seed_policy": ROW_SEED_POLICY_SAMPLE,
            "temperature": float(generation["temperature"]),
            "top_p": float(generation["top_p"]),
            "max_new_tokens": int(generation["max_new_tokens"]),
            "max_model_len": int(generation["max_model_len"]),
            "system_prompt_sha256": sha256_text(str(generation["system_prompt"])),
        },
        "inference_environment": inference_environment,
        "inference_environment_sha256": sha256_text(
            _canonical_json(inference_environment)
        ),
    }


def _progress_paths(destination: Path, artifact_id: str) -> tuple[Path, Path, Path]:
    artifact_id = _validate_artifact_id(artifact_id)
    progress_dir = destination / ".partial" / artifact_id
    return (
        progress_dir / "generations.progress.v1.jsonl",
        progress_dir / "state.progress.v1.json",
        progress_dir / ".lock",
    )


def _write_progress_state(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    payload = dict(state)
    payload.pop("integrity", None)
    payload["updated_at"] = _utc_now()
    sealed = seal_manifest(payload)
    _atomic_replace(
        path,
        json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return sealed


def _initial_progress_state(
    *,
    artifact_id: str,
    contract: dict[str, Any],
    contract_sha256: str,
    partial_path: Path,
    sample_count: int,
) -> dict[str, Any]:
    now = _utc_now()
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "sample_count": sample_count,
        "row_count": 0,
        "next_prompt_index": 0,
        "commit_count": 0,
        "resume_count": 0,
        "generation_complete": False,
        "status": "in_progress",
        "partial": {
            "path": str(partial_path),
            "committed_bytes": 0,
            "sha256": sha256_text(""),
            "row_count": 0,
        },
        "final_candidate_sha256": None,
        "created_at": now,
        "updated_at": now,
    }


def _validate_progress_row(
    envelope: dict[str, Any],
    *,
    position: int,
    prompt: dict[str, Any],
    contract: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    sample_id = str(prompt["sample_id"])
    expected_sample = contract["prompts"]["ordered_samples"][position]
    if (
        envelope.get("schema_version") != PROGRESS_ROW_SCHEMA_VERSION
        or envelope.get("artifact_id") != contract["artifact_id"]
        or envelope.get("contract_sha256") != contract_sha256
        or envelope.get("sequence_index") != position
        or envelope.get("sample_id") != sample_id
        or envelope.get("input_row_sha256")
        != expected_sample["input_row_sha256"]
        or envelope.get("generation_seed")
        != expected_sample["generation_seed"]
    ):
        raise ValueError(f"{sample_id}: progress envelope binding is inconsistent")
    row = envelope.get("row")
    if not isinstance(row, dict):
        raise ValueError(f"{sample_id}: progress row payload is not an object")
    for key, value in prompt.items():
        if row.get(key) != value:
            raise ValueError(f"{sample_id}: progress row changed input field {key!r}")

    generation = contract["generation"]
    expected_index = prompt.get("index", position)
    expected_source_index = prompt.get("source_index", expected_index)
    checks = {
        "sample_id": sample_id,
        "index": expected_index,
        "source_index": expected_source_index,
        "generation_position": position,
        "source_prompt_sha256": sha256_text(str(prompt.get("prompt") or "")),
        "target": prompt.get("response", ""),
        "replicate_id": "0",
        "generation_seed": expected_sample["generation_seed"],
        "generation_seed_policy": ROW_SEED_POLICY_SAMPLE,
        "generation_model": contract["model_path"],
        "generation_tokenizer": contract["tokenizer_path"],
        "generation_system_prompt_sha256": generation["system_prompt_sha256"],
        "generation_batch_size": generation["batch_size"],
        "decoding_temperature": generation["temperature"],
        "decoding_top_p": generation["top_p"],
        "max_new_tokens": generation["max_new_tokens"],
        "max_model_len": generation["max_model_len"],
        "artifact_id": contract["artifact_id"],
        "checkpoint_manifest_sha256": contract["checkpoint_manifest"]["sha256"],
        "evaluation_config_sha256": contract["evaluation_config"]["sha256"],
    }
    for key, expected in checks.items():
        if row.get(key) != expected:
            raise ValueError(f"{sample_id}: progress row field {key!r} changed")
    generated = row.get("generated")
    if (
        not isinstance(generated, str)
        or not generated.strip()
        or row.get("generated_sha256") != sha256_text(generated)
    ):
        raise ValueError(f"{sample_id}: progress generation is empty or corrupted")
    return row


def _parse_progress_prefix(
    content: bytes,
    *,
    prompts: list[dict[str, Any]],
    contract: dict[str, Any],
    contract_sha256: str,
) -> list[dict[str, Any]]:
    if content and not content.endswith(b"\n"):
        raise ValueError("Committed progress JSONL has a truncated final line")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Committed progress JSONL is not valid UTF-8") from exc
    rows: list[dict[str, Any]] = []
    for position, line in enumerate(text.splitlines()):
        if not line:
            raise ValueError("Committed progress JSONL contains a blank line")
        try:
            envelope = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Committed progress JSONL row {position + 1} is invalid"
            ) from exc
        if not isinstance(envelope, dict):
            raise ValueError(
                f"Committed progress JSONL row {position + 1} is not an object"
            )
        if line != _canonical_json(envelope):
            raise ValueError(
                f"Committed progress JSONL row {position + 1} is not canonical"
            )
        if position >= len(prompts):
            raise ValueError("Progress contains more rows than the prompt matrix")
        rows.append(
            _validate_progress_row(
                envelope,
                position=position,
                prompt=prompts[position],
                contract=contract,
                contract_sha256=contract_sha256,
            )
        )
    return rows


def _load_or_create_progress(
    *,
    partial_path: Path,
    state_path: Path,
    artifact_id: str,
    prompts: list[dict[str, Any]],
    contract: dict[str, Any],
    contract_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _durable_mkdir(partial_path.parent)
    state_existed = state_path.exists()
    if not state_existed:
        if partial_path.exists() and partial_path.stat().st_size:
            raise ValueError("Progress JSONL exists without its state receipt")
        if not partial_path.exists():
            _atomic_write(partial_path, "")
            _fsync_directory(partial_path.parent)
        state = _initial_progress_state(
            artifact_id=artifact_id,
            contract=contract,
            contract_sha256=contract_sha256,
            partial_path=partial_path,
            sample_count=len(prompts),
        )
        state = _write_progress_state(state_path, state)
    else:
        state = _read_json_object(state_path, label="generation progress state")
        validate_manifest_integrity(state)
        if (
            state.get("schema_version") != PROGRESS_SCHEMA_VERSION
            or state.get("artifact_id") != artifact_id
            or state.get("contract_sha256") != contract_sha256
            or state.get("contract") != contract
            or state.get("sample_count") != len(prompts)
        ):
            raise ValueError("Generation progress contract changed")
        partial_binding = state.get("partial")
        if (
            not isinstance(partial_binding, dict)
            or partial_binding.get("path") != str(partial_path)
        ):
            raise ValueError("Generation progress partial-file binding changed")
        if not partial_path.is_file():
            raise FileNotFoundError(
                f"Generation progress JSONL is missing: {partial_path}"
            )

    partial_binding = state["partial"]
    committed_bytes = partial_binding.get("committed_bytes")
    committed_rows = partial_binding.get("row_count")
    if (
        not isinstance(committed_bytes, int)
        or isinstance(committed_bytes, bool)
        or committed_bytes < 0
        or not isinstance(committed_rows, int)
        or isinstance(committed_rows, bool)
        or committed_rows < 0
    ):
        raise ValueError("Generation progress state has invalid committed bounds")
    for field in ("row_count", "next_prompt_index", "commit_count", "resume_count"):
        value = state.get(field)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(f"Generation progress state has invalid {field}")
    if (
        not isinstance(state.get("generation_complete"), bool)
        or state.get("status")
        not in {"in_progress", "generation_complete", "sealed"}
    ):
        raise ValueError("Generation progress completion status is invalid")
    validate_sha256(
        partial_binding.get("sha256"), label="generation progress SHA-256"
    )
    observed_size = partial_path.stat().st_size
    if observed_size < committed_bytes:
        raise ValueError("Generation progress JSONL is shorter than committed state")
    with partial_path.open("rb") as handle:
        committed_content = handle.read(committed_bytes)
    if sha256_text(committed_content.decode("utf-8")) != partial_binding.get(
        "sha256"
    ):
        raise ValueError("Generation progress committed prefix hash changed")
    if observed_size > committed_bytes:
        with partial_path.open("r+b") as handle:
            handle.truncate(committed_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(partial_path.parent)

    rows = _parse_progress_prefix(
        committed_content,
        prompts=prompts,
        contract=contract,
        contract_sha256=contract_sha256,
    )
    if (
        len(rows) != committed_rows
        or state.get("row_count") != len(rows)
        or state.get("next_prompt_index") != len(rows)
        or len(rows) > len(prompts)
    ):
        raise ValueError("Generation progress row counts are inconsistent")
    if state_existed and state.get("generation_complete") is not True:
        state = _write_progress_state(
            state_path,
            {
                **state,
                "resume_count": int(state.get("resume_count", 0)) + 1,
            },
        )
    return state, rows


def _build_progress_binding(
    *,
    state_path: Path,
    partial_path: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    validate_manifest_integrity(state)
    if (
        state.get("schema_version") != PROGRESS_SCHEMA_VERSION
        or state.get("generation_complete") is not True
        or state.get("status") != "generation_complete"
        or "final_manifest" in state
    ):
        raise ValueError(
            "Final manifest requires a frozen, non-circular generation_complete state"
        )
    state_bytes = state_path.read_bytes()
    if json.loads(state_bytes) != state:
        raise ValueError("In-memory progress state differs from the frozen state file")
    partial_bytes = partial_path.read_bytes()
    partial = state.get("partial")
    if not isinstance(partial, dict) or partial.get("path") != str(partial_path):
        raise ValueError("Frozen progress state has an invalid partial binding")
    if (
        len(partial_bytes) != partial.get("committed_bytes")
        or _sha256_bytes(partial_bytes) != partial.get("sha256")
        or partial.get("row_count") != state.get("row_count")
    ):
        raise ValueError("Frozen progress partial JSONL differs from its state")
    return {
        "schema_version": PROGRESS_BINDING_SCHEMA_VERSION,
        "contract_sha256": state["contract_sha256"],
        "state": {
            "path": str(state_path),
            "sha256": _sha256_bytes(state_bytes),
            "payload_sha256": state["integrity"]["payload_sha256"],
            "schema_version": state["schema_version"],
            "status": state["status"],
            "generation_complete": True,
        },
        "partial": {
            "path": str(partial_path),
            "sha256": partial["sha256"],
            "committed_bytes": partial["committed_bytes"],
            "row_count": partial["row_count"],
        },
    }


def _validate_progress_binding(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    output_path: Path,
    prompts_path: Path,
    config_path: Path,
    checkpoint_path: Path,
    prompts: list[dict[str, Any]],
    output_bytes: bytes,
) -> dict[str, Any]:
    artifact_id = str(manifest.get("artifact_id") or "")
    binding = manifest.get("progress")
    if (
        not isinstance(binding, dict)
        or binding.get("schema_version") != PROGRESS_BINDING_SCHEMA_VERSION
    ):
        raise ValueError(
            "Generation manifest lacks the required frozen progress binding"
        )
    expected_partial, expected_state, _ = _progress_paths(
        output_path.parent, artifact_id
    )
    state_binding = binding.get("state")
    partial_binding = binding.get("partial")
    if not isinstance(state_binding, dict) or not isinstance(partial_binding, dict):
        raise ValueError("Generation manifest progress binding is malformed")
    state_path = _resolve_artifact_path(
        state_binding.get("path"),
        manifest_path=manifest_path,
        label=f"{artifact_id} frozen progress state",
    )
    partial_path = _resolve_artifact_path(
        partial_binding.get("path"),
        manifest_path=manifest_path,
        label=f"{artifact_id} frozen progress JSONL",
    )
    if (
        state_path.parent != expected_state.parent
        or state_path.name
        not in {expected_state.name, "state.progress.frozen.v1.json"}
        or partial_path != expected_partial
    ):
        raise ValueError("Generation manifest progress paths are not canonical")
    state_bytes = state_path.read_bytes()
    try:
        state = json.loads(state_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Frozen generation progress state is invalid JSON") from exc
    if not isinstance(state, dict):
        raise ValueError("Frozen generation progress state must be an object")
    validate_manifest_integrity(state)
    if (
        state_binding.get("sha256") != _sha256_bytes(state_bytes)
        or state_binding.get("payload_sha256")
        != state.get("integrity", {}).get("payload_sha256")
        or state_binding.get("schema_version") != PROGRESS_SCHEMA_VERSION
        or state_binding.get("status") != "generation_complete"
        or state_binding.get("generation_complete") is not True
        or state.get("schema_version") != PROGRESS_SCHEMA_VERSION
        or state.get("status") != "generation_complete"
        or state.get("generation_complete") is not True
        or "final_manifest" in state
    ):
        raise ValueError("Frozen generation progress state binding changed")
    contract = state.get("contract")
    contract_sha256 = state.get("contract_sha256")
    if (
        not isinstance(contract, dict)
        or contract_sha256 != sha256_text(_canonical_json(contract))
        or binding.get("contract_sha256") != contract_sha256
    ):
        raise ValueError("Frozen generation progress contract changed")
    state_partial = state.get("partial")
    partial_bytes = partial_path.read_bytes()
    if (
        not isinstance(state_partial, dict)
        or state_partial.get("path") != str(partial_path)
        or partial_binding.get("path") != str(partial_path)
        or partial_binding.get("sha256") != state_partial.get("sha256")
        or partial_binding.get("committed_bytes")
        != state_partial.get("committed_bytes")
        or partial_binding.get("row_count") != state_partial.get("row_count")
        or len(partial_bytes) != state_partial.get("committed_bytes")
        or _sha256_bytes(partial_bytes) != state_partial.get("sha256")
    ):
        raise ValueError("Frozen generation progress JSONL binding changed")
    progress_rows = _parse_progress_prefix(
        partial_bytes,
        prompts=prompts,
        contract=contract,
        contract_sha256=contract_sha256,
    )
    output_sha256 = _sha256_bytes(output_bytes)
    if (
        len(progress_rows) != len(prompts)
        or state.get("sample_count") != len(prompts)
        or state.get("row_count") != len(prompts)
        or state.get("next_prompt_index") != len(prompts)
        or state.get("final_candidate_sha256") != output_sha256
        or state.get("final_candidate_bytes") != len(output_bytes)
    ):
        raise ValueError("Frozen generation progress is incomplete or mismatched")
    contract_checks = (
        ("checkpoint_manifest", checkpoint_path, manifest["checkpoint_manifest"]),
        ("evaluation_config", config_path, manifest["evaluation_config"]),
        ("prompts", prompts_path, manifest["prompts"]),
    )
    for key, actual_path, manifest_record in contract_checks:
        record = contract.get(key)
        if (
            not isinstance(record, dict)
            or record.get("path") != str(actual_path)
            or record.get("sha256") != sha256_file(actual_path)
            or record.get("sha256") != manifest_record.get("sha256")
        ):
            raise ValueError(f"Frozen progress {key} contract changed")
    if (
        contract.get("artifact_id") != artifact_id
        or contract.get("model_artifact", {}).get("sha256")
        != manifest.get("model_artifact", {}).get("sha256")
        or contract.get("tokenizer_artifact", {}).get("sha256")
        != manifest.get("tokenizer_artifact", {}).get("sha256")
        or contract.get("inference_environment_sha256")
        != manifest.get("inference_environment_sha256")
    ):
        raise ValueError("Frozen progress artifact/environment contract changed")
    return {
        "state_path": str(state_path),
        "state_sha256": state_binding["sha256"],
        "partial_path": str(partial_path),
        "partial_sha256": partial_binding["sha256"],
        "contract_sha256": contract_sha256,
        "row_count": len(progress_rows),
    }


def _generation_manifest(
    *,
    artifact_id: str,
    prompts: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    checkpoint_path: Path,
    prompts_path: Path,
    config_path: Path,
    model_fingerprint: dict[str, Any],
    tokenizer_fingerprint: dict[str, Any],
    inference_environment: dict[str, Any],
    inference_environment_sha256: str,
    output_path: Path,
    progress_binding: dict[str, Any],
) -> dict[str, Any]:
    return seal_manifest({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "sample_count": len(prompts),
        "valid_count": sum(row["valid_generation"] for row in rows),
        "invalid_count": sum(not row["valid_generation"] for row in rows),
        "checkpoint_manifest": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "evaluation_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "prompts": {
            "path": str(prompts_path),
            "sha256": sha256_file(prompts_path),
            "row_count": len(prompts),
        },
        "model_artifact": model_fingerprint,
        "tokenizer_artifact": tokenizer_fingerprint,
        "inference_environment": inference_environment,
        "inference_environment_sha256": inference_environment_sha256,
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "row_count": len(rows),
        },
        "progress": progress_binding,
        "created_at": _utc_now(),
    })


def validate_generation_progress_binding(
    *,
    generation_manifest_file: str | Path,
    generation_output_file: str | Path,
    checkpoint_manifest_file: str | Path,
    prompts_file: str | Path,
    config_file: str | Path,
) -> dict[str, Any]:
    """Validate the immutable progress chain bound by a v2 final manifest."""

    manifest_path = Path(generation_manifest_file).expanduser().resolve()
    output_path = Path(generation_output_file).expanduser().resolve()
    checkpoint_path = Path(checkpoint_manifest_file).expanduser().resolve()
    prompts_path = Path(prompts_file).expanduser().resolve()
    config_path = Path(config_file).expanduser().resolve()
    manifest = _read_json_object(manifest_path, label="generation manifest")
    validate_manifest_integrity(manifest)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Generation manifest must use the progress-bound v2 schema"
        )
    output_bytes = output_path.read_bytes()
    output_record = manifest.get("output")
    if (
        not isinstance(output_record, dict)
        or output_record.get("path") != str(output_path)
        or output_record.get("sha256") != _sha256_bytes(output_bytes)
    ):
        raise ValueError("Generation manifest output binding changed")
    prompts = _read_jsonl(prompts_path, label="evaluation prompts")
    return _validate_progress_binding(
        manifest=manifest,
        manifest_path=manifest_path,
        output_path=output_path,
        prompts_path=prompts_path,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        prompts=prompts,
        output_bytes=output_bytes,
    )


def reseal_legacy_generation_manifest_with_progress(
    *,
    checkpoint_manifest_file: str | Path,
    prompts_file: str | Path,
    config_file: str | Path,
    artifact_id: str,
    output_dir: str | Path,
    resealed_manifest_file: str | Path | None = None,
    frozen_state_file: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create versioned v2 receipts without overwriting a legacy v1 artifact."""

    artifact_id = _validate_artifact_id(artifact_id)
    checkpoint_path = Path(checkpoint_manifest_file).expanduser().resolve()
    prompts_path = Path(prompts_file).expanduser().resolve()
    config_path = Path(config_file).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    output_path = destination / f"{artifact_id}.jsonl"
    legacy_manifest_path = destination / f"{artifact_id}.manifest.json"
    manifest_path = (
        Path(resealed_manifest_file).expanduser().resolve()
        if resealed_manifest_file is not None
        else destination / f"{artifact_id}.manifest.progress-v2.json"
    )
    partial_path, state_path, lock_path = _progress_paths(destination, artifact_id)
    frozen_path = (
        Path(frozen_state_file).expanduser().resolve()
        if frozen_state_file is not None
        else state_path.with_name("state.progress.frozen.v1.json")
    )
    if manifest_path == legacy_manifest_path or frozen_path == state_path:
        raise ValueError("Legacy resealing requires new versioned receipt paths")
    if manifest_path.parent != destination:
        raise ValueError("Resealed manifest must remain directly inside output_dir")
    if frozen_path.parent != state_path.parent:
        raise ValueError(
            "Frozen state receipt must remain inside the artifact progress directory"
        )

    with _exclusive_progress_lock(lock_path):
        _, legacy_manifest = validate_generation_artifact(
            checkpoint_manifest_file=checkpoint_path,
            prompts_file=prompts_path,
            config_file=config_path,
            artifact_id=artifact_id,
            output_dir=destination,
            allow_legacy_without_progress=True,
        )
        if legacy_manifest.get("schema_version") != LEGACY_MANIFEST_SCHEMA_VERSION:
            raise ValueError("Resealing requires an existing legacy v1 manifest")
        if manifest_path.exists() and not frozen_path.exists():
            raise ValueError(
                "Resealed manifest exists without its frozen state receipt"
            )
        state_bytes = state_path.read_bytes()
        try:
            state = json.loads(state_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Legacy generation progress state is invalid") from exc
        if not isinstance(state, dict):
            raise ValueError("Legacy generation progress state must be an object")
        validate_manifest_integrity(state)
        prompts = _read_jsonl(prompts_path, label="evaluation prompts")
        if (
            state.get("schema_version") != PROGRESS_SCHEMA_VERSION
            or state.get("generation_complete") is not True
            or state.get("row_count") != len(prompts)
        ):
            raise ValueError("Legacy progress state is not complete")
        state_partial = state.get("partial")
        partial_bytes = partial_path.read_bytes()
        output_bytes = output_path.read_bytes()
        if (
            not isinstance(state_partial, dict)
            or state_partial.get("path") != str(partial_path)
            or state_partial.get("committed_bytes") != len(partial_bytes)
            or state_partial.get("sha256") != _sha256_bytes(partial_bytes)
        ):
            raise ValueError("Legacy progress JSONL differs from its state")
        contract = state.get("contract")
        contract_sha256 = state.get("contract_sha256")
        if (
            not isinstance(contract, dict)
            or contract_sha256 != sha256_text(_canonical_json(contract))
        ):
            raise ValueError("Legacy progress contract is invalid")
        progress_rows = _parse_progress_prefix(
            partial_bytes,
            prompts=prompts,
            contract=contract,
            contract_sha256=contract_sha256,
        )
        if (
            len(progress_rows) != len(prompts)
            or state.get("final_candidate_sha256") != _sha256_bytes(output_bytes)
            or state.get("final_candidate_bytes") != len(output_bytes)
        ):
            raise ValueError("Legacy progress does not reproduce the final output")

        frozen_payload = dict(state)
        frozen_payload.pop("integrity", None)
        frozen_payload.pop("final_manifest", None)
        frozen_payload["status"] = "generation_complete"
        frozen_payload["legacy_state_source"] = {
            "path": str(state_path),
            "sha256": _sha256_bytes(state_bytes),
            "payload_sha256": state["integrity"]["payload_sha256"],
        }
        expected_frozen_state = seal_manifest(frozen_payload)
        if frozen_path.exists():
            frozen_bytes = frozen_path.read_bytes()
            try:
                frozen_state = json.loads(frozen_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("Existing frozen state receipt is invalid") from exc
            if frozen_state != expected_frozen_state:
                raise ValueError(
                    "Existing frozen state receipt does not match legacy inputs"
                )
            validate_manifest_integrity(frozen_state)
        else:
            frozen_state = expected_frozen_state
            _atomic_write(
                frozen_path,
                json.dumps(frozen_state, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
            )
            _fsync_directory(frozen_path.parent)
        progress_binding = _build_progress_binding(
            state_path=frozen_path,
            partial_path=partial_path,
            state=frozen_state,
        )
        if manifest_path.exists():
            _, existing_manifest = validate_generation_artifact(
                checkpoint_manifest_file=checkpoint_path,
                prompts_file=prompts_path,
                config_file=config_path,
                artifact_id=artifact_id,
                output_dir=destination,
                manifest_file=manifest_path,
            )
            if existing_manifest.get("progress") != progress_binding:
                raise ValueError(
                    "Existing resealed manifest has a different progress binding"
                )
            return manifest_path, existing_manifest
        manifest_payload = dict(legacy_manifest)
        manifest_payload.pop("integrity", None)
        manifest_payload["schema_version"] = MANIFEST_SCHEMA_VERSION
        manifest_payload["progress"] = progress_binding
        manifest_payload["created_at"] = _utc_now()
        manifest = seal_manifest(manifest_payload)
        _atomic_write(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )
        _fsync_directory(manifest_path.parent)
        validate_generation_artifact(
            checkpoint_manifest_file=checkpoint_path,
            prompts_file=prompts_path,
            config_file=config_path,
            artifact_id=artifact_id,
            output_dir=destination,
            manifest_file=manifest_path,
        )
        return manifest_path, manifest


def generate_artifact(
    *,
    checkpoint_manifest_file: str | Path,
    prompts_file: str | Path,
    config_file: str | Path,
    artifact_id: str,
    output_dir: str | Path,
    batch_size: int = 20,
) -> tuple[Path, dict[str, Any]]:
    artifact_id = _validate_artifact_id(artifact_id)
    checkpoint_path = Path(checkpoint_manifest_file).expanduser().resolve()
    prompts_path = Path(prompts_file).expanduser().resolve()
    config_path = Path(config_file).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    output_path = destination / f"{artifact_id}.jsonl"
    manifest_path = destination / f"{artifact_id}.manifest.json"
    if manifest_path.exists():
        return validate_generation_artifact(
            checkpoint_manifest_file=checkpoint_path,
            prompts_file=prompts_path,
            config_file=config_path,
            artifact_id=artifact_id,
            output_dir=destination,
        )

    checkpoint_manifest = _read_json_object(
        checkpoint_path, label="checkpoint manifest"
    )
    validate_manifest_integrity(checkpoint_manifest)
    artifact = select_artifact(checkpoint_manifest, artifact_id)
    config = _read_json_object(config_path, label="evaluation config")
    prompts = _read_jsonl(prompts_path, label="evaluation prompts")
    generation = config.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("Evaluation config lacks generation settings")

    model_path = _resolve_artifact_path(
        artifact.get("model_path"),
        manifest_path=checkpoint_path,
        label=f"{artifact_id} model",
    )
    tokenizer_path = _resolve_artifact_path(
        artifact.get("tokenizer_path"),
        manifest_path=checkpoint_path,
        label=f"{artifact_id} tokenizer",
    )
    model_fingerprint = _validate_fingerprint(
        model_path,
        artifact.get("model_sha256"),
        label=f"{artifact_id} model",
    )
    tokenizer_fingerprint = _validate_fingerprint(
        tokenizer_path,
        artifact.get("tokenizer_sha256"),
        label=f"{artifact_id} tokenizer",
    )
    inference_environment = _capture_inference_environment()
    inference_environment_sha256 = sha256_text(
        _canonical_json(inference_environment)
    )
    contract = _progress_contract(
        checkpoint_path=checkpoint_path,
        prompts_path=prompts_path,
        config_path=config_path,
        artifact=artifact,
        prompts=prompts,
        generation=generation,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        model_fingerprint=model_fingerprint,
        tokenizer_fingerprint=tokenizer_fingerprint,
        batch_size=batch_size,
        inference_environment=inference_environment,
    )
    contract_sha256 = sha256_text(_canonical_json(contract))
    partial_path, state_path, lock_path = _progress_paths(destination, artifact_id)

    with _exclusive_progress_lock(lock_path):
        if manifest_path.exists():
            return validate_generation_artifact(
                checkpoint_manifest_file=checkpoint_path,
                prompts_file=prompts_path,
                config_file=config_path,
                artifact_id=artifact_id,
                output_dir=destination,
            )
        progress_state, generated_rows = _load_or_create_progress(
            partial_path=partial_path,
            state_path=state_path,
            artifact_id=artifact_id,
            prompts=prompts,
            contract=contract,
            contract_sha256=contract_sha256,
        )
        print(f"progress_jsonl={partial_path}")
        print(f"progress_state={state_path}")
        print(f"committed_count={len(generated_rows)}/{len(prompts)}")

        if output_path.exists() and len(generated_rows) != len(prompts):
            raise FileExistsError(
                "Generation output exists without a manifest or complete progress: "
                f"{output_path}"
            )

        if len(generated_rows) < len(prompts):
            pending: list[dict[str, Any]] = []
            for position, prompt in enumerate(
                prompts[len(generated_rows) :], start=len(generated_rows)
            ):
                row = dict(prompt)
                row.setdefault("index", position)
                row.setdefault("source_index", row.get("index", position))
                row["generation_position"] = position
                pending.append(row)

            def persist_batch(
                batch_rows: list[dict[str, Any]], expected_indices: list[int]
            ) -> None:
                nonlocal progress_state, generated_rows
                start = len(generated_rows)
                if len(batch_rows) != len(expected_indices):
                    raise ValueError(
                        "Generation batch did not return exactly one record per prompt"
                    )
                expected_positions = list(range(start, start + len(batch_rows)))
                observed_positions = [
                    row.get("generation_position") for row in batch_rows
                ]
                if observed_positions != expected_positions:
                    raise ValueError(
                        "Generation batch does not continue the committed prompt prefix"
                    )

                envelopes: list[dict[str, Any]] = []
                validated_rows: list[dict[str, Any]] = []
                for position, row in zip(
                    expected_positions, batch_rows, strict=True
                ):
                    prompt = prompts[position]
                    sample_contract = contract["prompts"]["ordered_samples"][
                        position
                    ]
                    envelope = {
                        "schema_version": PROGRESS_ROW_SCHEMA_VERSION,
                        "artifact_id": artifact_id,
                        "contract_sha256": contract_sha256,
                        "sequence_index": position,
                        "sample_id": prompt["sample_id"],
                        "input_row_sha256": sample_contract["input_row_sha256"],
                        "generation_seed": sample_contract["generation_seed"],
                        "row": row,
                    }
                    validated_rows.append(
                        _validate_progress_row(
                            envelope,
                            position=position,
                            prompt=prompt,
                            contract=contract,
                            contract_sha256=contract_sha256,
                        )
                    )
                    envelopes.append(envelope)
                encoded = "".join(
                    f"{_canonical_json(envelope)}\n" for envelope in envelopes
                ).encode("utf-8")
                with partial_path.open("ab") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())

                new_count = start + len(validated_rows)
                new_size = partial_path.stat().st_size
                progress_state = _write_progress_state(
                    state_path,
                    {
                        **progress_state,
                        "row_count": new_count,
                        "next_prompt_index": new_count,
                        "commit_count": int(progress_state["commit_count"]) + 1,
                        "partial": {
                            "path": str(partial_path),
                            "committed_bytes": new_size,
                            "sha256": sha256_file(partial_path),
                            "row_count": new_count,
                        },
                    },
                )
                generated_rows.extend(validated_rows)
                print(
                    f"progress_committed={new_count}/{len(prompts)} "
                    f"partial={partial_path} state={state_path}"
                )

            generated_now = generate_new_response(
                pending,
                str(model_path),
                output_file=None,
                batch_size=batch_size,
                seed=int(generation["base_seed"]),
                replicate_id="0",
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                max_new_tokens=int(generation["max_new_tokens"]),
                max_model_len=int(generation["max_model_len"]),
                tokenizer_path=str(tokenizer_path),
                system_prompt=str(generation["system_prompt"]),
                seed_policy=ROW_SEED_POLICY_SAMPLE,
                generation_metadata={
                    "artifact_id": artifact_id,
                    "checkpoint_manifest_sha256": sha256_file(checkpoint_path),
                    "evaluation_config_sha256": sha256_file(config_path),
                },
                fail_closed=True,
                progress_callback=persist_batch,
            )
            if len(generated_now) != len(pending):
                raise RuntimeError(
                    "Generation ended before every pending prompt was committed"
                )

        if len(generated_rows) != len(prompts):
            raise RuntimeError("Generation progress is incomplete; refusing to seal")
        rows = finalise_generation_rows(
            prompts,
            generated_rows,
            artifact=artifact,
            model_fingerprint=model_fingerprint,
            tokenizer_fingerprint=tokenizer_fingerprint,
            shared_provenance={
                "test_set_sha256": sha256_file(prompts_path),
                "prompt_template_sha256": sha256_file(config_path),
                "decoding_config_sha256": sha256_file(config_path),
                "temperature": float(generation["temperature"]),
                "top_p": float(generation["top_p"]),
                "max_new_tokens": int(generation["max_new_tokens"]),
                "max_model_len": int(generation["max_model_len"]),
                "generation_base_seed": int(generation["base_seed"]),
                "generation_seed_policy": ROW_SEED_POLICY_SAMPLE,
                "inference_environment": inference_environment,
                "inference_environment_sha256": inference_environment_sha256,
            },
        )
        output_content = "".join(
            f"{_canonical_json(row)}\n" for row in rows
        )
        output_sha256 = sha256_text(output_content)
        progress_state = _write_progress_state(
            state_path,
            {
                **progress_state,
                "generation_complete": True,
                "status": "generation_complete",
                "final_candidate_sha256": output_sha256,
                "final_candidate_bytes": len(output_content.encode("utf-8")),
            },
        )

        if output_path.exists():
            if sha256_file(output_path) != output_sha256:
                raise ValueError(
                    "Existing generation output differs from complete progress"
                )
        else:
            _atomic_write(output_path, output_content)
            _fsync_directory(output_path.parent)
        progress_binding = _build_progress_binding(
            state_path=state_path,
            partial_path=partial_path,
            state=progress_state,
        )
        manifest = _generation_manifest(
            artifact_id=artifact_id,
            prompts=prompts,
            rows=rows,
            checkpoint_path=checkpoint_path,
            prompts_path=prompts_path,
            config_path=config_path,
            model_fingerprint=model_fingerprint,
            tokenizer_fingerprint=tokenizer_fingerprint,
            inference_environment=inference_environment,
            inference_environment_sha256=inference_environment_sha256,
            output_path=output_path,
            progress_binding=progress_binding,
        )
        _atomic_write(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )
        _fsync_directory(manifest_path.parent)
        return manifest_path, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one frozen checkpoint artifact on every common-test "
            "sample. This command performs inference only, never training."
        )
    )
    parser.add_argument("--checkpoint-manifest", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--reseal-legacy-progress",
        action="store_true",
        help=(
            "Create versioned v2 manifest/frozen-state receipts for an existing "
            "legacy v1 artifact; never overwrite the legacy files."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.reseal_legacy_progress:
        path, manifest = reseal_legacy_generation_manifest_with_progress(
            checkpoint_manifest_file=args.checkpoint_manifest,
            prompts_file=args.prompts,
            config_file=args.config,
            artifact_id=args.artifact_id,
            output_dir=args.output_dir,
        )
    else:
        path, manifest = generate_artifact(
            checkpoint_manifest_file=args.checkpoint_manifest,
            prompts_file=args.prompts,
            config_file=args.config,
            artifact_id=args.artifact_id,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    print(f"generation_manifest={path}")
    print(f"valid_count={manifest['valid_count']}")
    print(f"invalid_count={manifest['invalid_count']}")


if __name__ == "__main__":
    main()
