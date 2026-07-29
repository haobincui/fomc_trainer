"""Frozen, generation-only experiment specifications for canonical LOO runs.

The helpers in this module deliberately do not load, train, merge, or mutate a
model.  They bind an experiment specification to local model, tokenizer, and
source artifacts; derive row-level sampling seeds without using batching or
indicator identity; and reject generation records that could have been
truncated.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from open_r1.provenance import (
    fingerprint_artifact_path,
    sha256_file,
    validate_sha256,
)


SPEC_SCHEMA_VERSION = "loo-generation-spec-v1"
ARTIFACT_FINGERPRINT_SCHEMA_VERSION = "loo-frozen-artifacts-v1"
ROW_SEED_SCHEMA_VERSION = "loo-row-seed-v1"
MANIFEST_INTEGRITY_ALGORITHM = "sha256(canonical-json-without-integrity)"

# A conservative range understood by Python, NumPy, PyTorch, and vLLM seed
# interfaces.  Zero is a valid deterministic seed.
MAX_ROW_SEED = (2**31) - 1

DEFAULT_SUCCESS_FINISH_REASONS = frozenset(
    {
        "completed",
        "end_of_sequence",
        "eos",
        "eos_token",
        "stop",
    }
)
TOKEN_LIMIT_FINISH_REASONS = frozenset(
    {
        "length",
        "max_length",
        "max_new_tokens",
        "max_tokens",
        "token_limit",
    }
)


class GenerationSpecError(ValueError):
    """Base class for invalid generation-only experiment specifications."""


class ManifestIntegrityError(GenerationSpecError):
    """Raised when a sealed manifest was changed after it was created."""


class FrozenArtifactMismatchError(GenerationSpecError):
    """Raised when a frozen model, tokenizer, or source artifact has changed."""


class GenerationSafetyError(GenerationSpecError):
    """Raised when token accounting or a finish reason is not canonical-safe."""


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        serialised = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise GenerationSpecError(
            f"Manifest payload must contain finite JSON values only: {exc}"
        ) from exc
    return serialised.encode("utf-8")


def manifest_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a manifest payload using the canonical JSON representation."""

    if not isinstance(payload, Mapping):
        raise GenerationSpecError("Manifest payload must be a mapping")
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def seal_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-copied payload with an embedded tamper-evident digest."""

    if not isinstance(payload, Mapping):
        raise GenerationSpecError("Manifest payload must be a mapping")
    if "integrity" in payload:
        raise GenerationSpecError(
            "Refusing to seal a payload that already contains an integrity field"
        )
    sealed = copy.deepcopy(dict(payload))
    sealed["integrity"] = {
        "algorithm": MANIFEST_INTEGRITY_ALGORITHM,
        "payload_sha256": manifest_payload_sha256(sealed),
    }
    return sealed


def validate_manifest_integrity(
    manifest: Mapping[str, Any],
    *,
    expected_payload_sha256: str | None = None,
) -> str:
    """Validate the embedded digest and, optionally, an externally frozen digest.

    The optional external digest prevents an actor from changing a manifest and
    simply recomputing its embedded digest.  The embedded digest alone is still
    useful for detecting accidental edits.
    """

    if not isinstance(manifest, Mapping):
        raise ManifestIntegrityError("Manifest must be a JSON object")
    integrity = manifest.get("integrity")
    if not isinstance(integrity, Mapping):
        raise ManifestIntegrityError("Manifest is missing its integrity record")
    if integrity.get("algorithm") != MANIFEST_INTEGRITY_ALGORITHM:
        raise ManifestIntegrityError(
            f"Unsupported manifest integrity algorithm: "
            f"{integrity.get('algorithm')!r}"
        )
    try:
        recorded_digest = validate_sha256(
            integrity.get("payload_sha256"),
            label="integrity.payload_sha256",
        )
    except ValueError as exc:
        raise ManifestIntegrityError(str(exc)) from exc

    payload = copy.deepcopy(dict(manifest))
    payload.pop("integrity", None)
    observed_digest = manifest_payload_sha256(payload)
    if not hmac.compare_digest(recorded_digest, observed_digest):
        raise ManifestIntegrityError(
            "Manifest payload digest mismatch; the manifest was modified after "
            "it was sealed"
        )

    if expected_payload_sha256 is not None:
        try:
            expected_digest = validate_sha256(
                expected_payload_sha256,
                label="expected_payload_sha256",
            )
        except ValueError as exc:
            raise ManifestIntegrityError(str(exc)) from exc
        if not hmac.compare_digest(expected_digest, observed_digest):
            raise ManifestIntegrityError(
                "Manifest does not match the externally frozen payload digest"
            )
    return observed_digest


def _normalise_named_paths(
    paths: Mapping[str, str | Path],
    *,
    category: str,
) -> dict[str, Path]:
    if not isinstance(paths, Mapping) or not paths:
        raise GenerationSpecError(
            f"At least one named {category} artifact is required"
        )
    normalised: dict[str, Path] = {}
    for raw_name, raw_path in paths.items():
        name = str(raw_name).strip()
        if not name:
            raise GenerationSpecError(f"{category} artifact names cannot be empty")
        if name in normalised:
            raise GenerationSpecError(
                f"Duplicate {category} artifact name after normalisation: {name!r}"
            )
        normalised[name] = Path(raw_path)
    return normalised


def fingerprint_frozen_artifacts(
    *,
    models: Mapping[str, str | Path],
    tokenizers: Mapping[str, str | Path],
    sources: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Fingerprint every byte in named model, tokenizer, and source artifacts."""

    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for category, raw_paths in (
        ("models", models),
        ("tokenizers", tokenizers),
        ("sources", sources),
    ):
        paths = _normalise_named_paths(raw_paths, category=category)
        groups[category] = {
            name: fingerprint_artifact_path(path)
            for name, path in sorted(paths.items())
        }
    return {
        "schema_version": ARTIFACT_FINGERPRINT_SCHEMA_VERSION,
        **groups,
    }


def _validate_fingerprint_record(
    record: Mapping[str, Any],
    *,
    category: str,
    name: str,
    verify_path: bool,
) -> None:
    if not isinstance(record, Mapping):
        raise GenerationSpecError(
            f"Frozen {category} artifact {name!r} must be a mapping"
        )
    path_text = str(record.get("path") or "").strip()
    if not path_text:
        raise GenerationSpecError(
            f"Frozen {category} artifact {name!r} has no local path"
        )
    try:
        validate_sha256(
            record.get("sha256"),
            label=f"frozen_artifacts.{category}.{name}.sha256",
        )
    except ValueError as exc:
        raise GenerationSpecError(str(exc)) from exc
    if record.get("kind") not in {"file", "directory"}:
        raise GenerationSpecError(
            f"Frozen {category} artifact {name!r} has invalid kind "
            f"{record.get('kind')!r}"
        )
    for field in ("file_count", "total_bytes"):
        value = record.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GenerationSpecError(
                f"Frozen {category} artifact {name!r} has invalid {field}"
            )

    if not verify_path:
        return
    try:
        observed = fingerprint_artifact_path(path_text)
    except (FileNotFoundError, ValueError) as exc:
        raise FrozenArtifactMismatchError(
            f"Frozen {category} artifact {name!r} is unavailable: {exc}"
        ) from exc
    compared_fields = (
        "path",
        "kind",
        "sha256",
        "file_count",
        "total_bytes",
        "algorithm",
    )
    mismatches = {
        field: {
            "expected": record.get(field),
            "observed": observed.get(field),
        }
        for field in compared_fields
        if record.get(field) != observed.get(field)
    }
    if mismatches:
        raise FrozenArtifactMismatchError(
            f"Frozen {category} artifact {name!r} no longer matches its "
            f"fingerprint: {mismatches}"
        )


def validate_frozen_artifacts(
    frozen_artifacts: Mapping[str, Any],
    *,
    verify_paths: bool = True,
) -> None:
    """Validate the frozen inventory and optionally re-hash its local paths."""

    if not isinstance(frozen_artifacts, Mapping):
        raise GenerationSpecError("frozen_artifacts must be a mapping")
    if (
        frozen_artifacts.get("schema_version")
        != ARTIFACT_FINGERPRINT_SCHEMA_VERSION
    ):
        raise GenerationSpecError(
            "Unsupported frozen-artifact fingerprint schema: "
            f"{frozen_artifacts.get('schema_version')!r}"
        )
    for category in ("models", "tokenizers", "sources"):
        records = frozen_artifacts.get(category)
        if not isinstance(records, Mapping) or not records:
            raise GenerationSpecError(
                f"frozen_artifacts.{category} must contain at least one artifact"
            )
        for raw_name, record in records.items():
            name = str(raw_name).strip()
            if not name:
                raise GenerationSpecError(
                    f"frozen_artifacts.{category} contains an empty artifact name"
                )
            _validate_fingerprint_record(
                record,
                category=category,
                name=name,
                verify_path=verify_paths,
            )


def _validate_non_negative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GenerationSpecError(f"{label} must be a non-negative integer")
    return value


def _validate_positive_int(value: Any, *, label: str) -> int:
    integer = _validate_non_negative_int(value, label=label)
    if integer == 0:
        raise GenerationSpecError(f"{label} must be positive")
    return integer


def derive_row_seed(replicate_seed: int, sample_id: str) -> int:
    """Derive a stable engine-safe row seed from only replicate and sample.

    Indicator identity, treatment arm, input ordering, generation position, and
    batch size are intentionally absent from both the interface and hash input.
    """

    replicate_seed = _validate_non_negative_int(
        replicate_seed,
        label="replicate_seed",
    )
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise GenerationSpecError("sample_id must be a non-empty string")
    seed_material = _canonical_json_bytes(
        {
            "schema_version": ROW_SEED_SCHEMA_VERSION,
            "replicate_seed": replicate_seed,
            "sample_id": sample_id,
        }
    )
    digest = hashlib.sha256(seed_material).digest()
    return int.from_bytes(digest[:8], byteorder="big") % (MAX_ROW_SEED + 1)


def validate_context_budget(
    *,
    input_token_count: int,
    max_new_tokens: int,
    context_limit: int,
    input_was_truncated: bool = False,
    consumed_input_token_count: int | None = None,
) -> dict[str, int]:
    """Reject prompt truncation and generation requests exceeding context."""

    input_token_count = _validate_non_negative_int(
        input_token_count,
        label="input_token_count",
    )
    max_new_tokens = _validate_positive_int(
        max_new_tokens,
        label="max_new_tokens",
    )
    context_limit = _validate_positive_int(
        context_limit,
        label="context_limit",
    )
    if not isinstance(input_was_truncated, bool):
        raise GenerationSafetyError("input_was_truncated must be a boolean")
    if input_was_truncated:
        raise GenerationSafetyError(
            "Input truncation is forbidden for a canonical LOO generation"
        )
    if consumed_input_token_count is not None:
        consumed_input_token_count = _validate_non_negative_int(
            consumed_input_token_count,
            label="consumed_input_token_count",
        )
        if consumed_input_token_count != input_token_count:
            raise GenerationSafetyError(
                "Consumed input token count differs from the pre-tokenized input; "
                "the prompt may have been truncated or rewritten"
            )
    requested_total = input_token_count + max_new_tokens
    if requested_total > context_limit:
        raise GenerationSafetyError(
            "Generation request exceeds the model context window: "
            f"input={input_token_count}, max_new_tokens={max_new_tokens}, "
            f"context_limit={context_limit}, overflow={requested_total - context_limit}"
        )
    return {
        "input_token_count": input_token_count,
        "max_new_tokens": max_new_tokens,
        "context_limit": context_limit,
        "unused_context_tokens": context_limit - requested_total,
    }


def _normalise_finish_reason(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return text


def validate_generation_completion(
    *,
    input_token_count: int,
    output_token_count: int,
    max_new_tokens: int,
    context_limit: int,
    finish_reason: str,
    input_was_truncated: bool = False,
    consumed_input_token_count: int | None = None,
    allowed_finish_reasons: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate context accounting and reject token-limit/unknown completions."""

    context_audit = validate_context_budget(
        input_token_count=input_token_count,
        max_new_tokens=max_new_tokens,
        context_limit=context_limit,
        input_was_truncated=input_was_truncated,
        consumed_input_token_count=consumed_input_token_count,
    )
    output_token_count = _validate_positive_int(
        output_token_count,
        label="output_token_count",
    )
    if output_token_count > max_new_tokens:
        raise GenerationSafetyError(
            "output_token_count exceeds the requested max_new_tokens"
        )
    if input_token_count + output_token_count > context_limit:
        raise GenerationSafetyError(
            "Observed input and output token counts exceed the context window"
        )

    normalised_reason = _normalise_finish_reason(finish_reason)
    if normalised_reason in TOKEN_LIMIT_FINISH_REASONS:
        raise GenerationSafetyError(
            f"Generation ended at a token limit ({finish_reason!r}); the output "
            "is not canonical-safe"
        )
    allowed = (
        DEFAULT_SUCCESS_FINISH_REASONS
        if allowed_finish_reasons is None
        else frozenset(
            _normalise_finish_reason(reason)
            for reason in allowed_finish_reasons
            if _normalise_finish_reason(reason)
        )
    )
    if not allowed:
        raise GenerationSafetyError(
            "allowed_finish_reasons must contain at least one non-empty value"
        )
    if normalised_reason not in allowed:
        raise GenerationSafetyError(
            f"Unknown or unsuccessful finish reason {finish_reason!r}; allowed "
            f"reasons are {sorted(allowed)}"
        )
    return {
        **context_audit,
        "output_token_count": output_token_count,
        "finish_reason": normalised_reason,
        "consumed_full_output_budget": output_token_count == max_new_tokens,
    }


def build_generation_spec(
    *,
    run_id: str,
    phase: str,
    population_id: str,
    models: Mapping[str, str | Path],
    tokenizers: Mapping[str, str | Path],
    sources: Mapping[str, str | Path],
    generation_config: Mapping[str, Any],
    replicate_seeds: Sequence[int],
) -> dict[str, Any]:
    """Build and seal a ``loo-generation-spec-v1`` experiment specification."""

    identifiers = {
        "run_id": run_id,
        "phase": phase,
        "population_id": population_id,
    }
    for label, value in identifiers.items():
        if not isinstance(value, str) or not value.strip():
            raise GenerationSpecError(f"{label} must be a non-empty string")
        identifiers[label] = value.strip()
    if not isinstance(generation_config, Mapping) or not generation_config:
        raise GenerationSpecError("generation_config must be a non-empty mapping")

    validated_seeds = [
        _validate_non_negative_int(seed, label="replicate_seeds item")
        for seed in replicate_seeds
    ]
    if not validated_seeds:
        raise GenerationSpecError("replicate_seeds must not be empty")
    if len(validated_seeds) != len(set(validated_seeds)):
        raise GenerationSpecError("replicate_seeds must be unique")

    frozen_artifacts = fingerprint_frozen_artifacts(
        models=models,
        tokenizers=tokenizers,
        sources=sources,
    )
    payload = {
        "schema_version": SPEC_SCHEMA_VERSION,
        **identifiers,
        "frozen_artifacts": frozen_artifacts,
        "generation_config": copy.deepcopy(dict(generation_config)),
        "seed_policy": {
            "schema_version": ROW_SEED_SCHEMA_VERSION,
            "derivation_inputs": ["replicate_seed", "sample_id"],
            "algorithm": (
                "sha256(canonical-json({schema_version,replicate_seed,sample_id}))"
                f" mod {MAX_ROW_SEED + 1}"
            ),
            "output_min": 0,
            "output_max": MAX_ROW_SEED,
            "replicate_seeds": validated_seeds,
        },
        "safety_policy": {
            "input_truncation": "forbidden",
            "token_limit_finish": "forbidden",
            "context_preflight": "input_tokens + max_new_tokens <= context_limit",
        },
    }
    # Validate JSON compatibility before sealing so a spec cannot be built but
    # later fail canonical hashing.
    manifest_payload_sha256(payload)
    return seal_manifest(payload)


def validate_generation_spec(
    spec: Mapping[str, Any],
    *,
    verify_artifact_paths: bool = True,
    expected_payload_sha256: str | None = None,
) -> str:
    """Validate schema, integrity, seed policy, and frozen artifact contents."""

    payload_digest = validate_manifest_integrity(
        spec,
        expected_payload_sha256=expected_payload_sha256,
    )
    if spec.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise GenerationSpecError(
            f"Unsupported generation spec schema: {spec.get('schema_version')!r}"
        )
    for label in ("run_id", "phase", "population_id"):
        if not isinstance(spec.get(label), str) or not str(spec[label]).strip():
            raise GenerationSpecError(f"{label} must be a non-empty string")
    if not isinstance(spec.get("generation_config"), Mapping) or not spec[
        "generation_config"
    ]:
        raise GenerationSpecError("generation_config must be a non-empty mapping")

    seed_policy = spec.get("seed_policy")
    if not isinstance(seed_policy, Mapping):
        raise GenerationSpecError("seed_policy must be a mapping")
    expected_seed_policy = {
        "schema_version": ROW_SEED_SCHEMA_VERSION,
        "derivation_inputs": ["replicate_seed", "sample_id"],
        "output_min": 0,
        "output_max": MAX_ROW_SEED,
    }
    mismatches = {
        key: {"expected": value, "observed": seed_policy.get(key)}
        for key, value in expected_seed_policy.items()
        if seed_policy.get(key) != value
    }
    if mismatches:
        raise GenerationSpecError(
            f"Generation spec has an incompatible row-seed policy: {mismatches}"
        )
    seeds = seed_policy.get("replicate_seeds")
    if not isinstance(seeds, list) or not seeds:
        raise GenerationSpecError("seed_policy.replicate_seeds must be a non-empty list")
    validated_seeds = [
        _validate_non_negative_int(seed, label="seed_policy.replicate_seeds item")
        for seed in seeds
    ]
    if len(validated_seeds) != len(set(validated_seeds)):
        raise GenerationSpecError(
            "seed_policy.replicate_seeds must contain unique values"
        )

    safety_policy = spec.get("safety_policy")
    if not isinstance(safety_policy, Mapping):
        raise GenerationSpecError("safety_policy must be a mapping")
    expected_safety = {
        "input_truncation": "forbidden",
        "token_limit_finish": "forbidden",
        "context_preflight": (
            "input_tokens + max_new_tokens <= context_limit"
        ),
    }
    safety_mismatches = {
        key: {"expected": value, "observed": safety_policy.get(key)}
        for key, value in expected_safety.items()
        if safety_policy.get(key) != value
    }
    if safety_mismatches:
        raise GenerationSpecError(
            f"Generation spec has an incompatible safety policy: "
            f"{safety_mismatches}"
        )

    validate_frozen_artifacts(
        spec.get("frozen_artifacts"),
        verify_paths=verify_artifact_paths,
    )
    return payload_digest


def write_frozen_generation_spec(
    path: str | Path,
    spec: Mapping[str, Any],
    *,
    verify_artifact_paths: bool = True,
) -> str:
    """Atomically write a spec, refusing to overwrite different frozen bytes."""

    validate_generation_spec(
        spec,
        verify_artifact_paths=verify_artifact_paths,
    )
    destination = Path(path)
    serialised = (
        json.dumps(
            spec,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if destination.is_file():
        if destination.read_text(encoding="utf-8") != serialised:
            raise ManifestIntegrityError(
                f"Refusing to overwrite incompatible frozen generation spec "
                f"{destination}; use a new run directory"
            )
        return sha256_file(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(serialised, encoding="utf-8")
    temporary.replace(destination)
    return sha256_file(destination)


def load_and_validate_generation_spec(
    path: str | Path,
    *,
    verify_artifact_paths: bool = True,
    expected_file_sha256: str | None = None,
    expected_payload_sha256: str | None = None,
) -> dict[str, Any]:
    """Load a frozen spec and validate manifest and artifact tampering."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Generation spec does not exist: {source}")
    if expected_file_sha256 is not None:
        try:
            expected_file_digest = validate_sha256(
                expected_file_sha256,
                label="expected_file_sha256",
            )
        except ValueError as exc:
            raise ManifestIntegrityError(str(exc)) from exc
        observed_file_digest = sha256_file(source)
        if not hmac.compare_digest(expected_file_digest, observed_file_digest):
            raise ManifestIntegrityError(
                "Generation spec file does not match the externally frozen "
                "file digest"
            )
    try:
        spec = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GenerationSpecError(f"Invalid generation spec JSON {source}: {exc}") from exc
    if not isinstance(spec, dict):
        raise GenerationSpecError("Generation spec must contain a JSON object")
    validate_generation_spec(
        spec,
        verify_artifact_paths=verify_artifact_paths,
        expected_payload_sha256=expected_payload_sha256,
    )
    return spec
