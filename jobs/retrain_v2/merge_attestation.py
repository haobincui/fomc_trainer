"""Immutable in-directory attestations for retrain-v2 adapter merges."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


ATTESTATION_FILENAME = "merge_attestation.json"
SCHEMA_VERSION = 1
SEMANTIC_METHOD = "peft_merge_and_unload_in_process_v1"
SEMANTIC_BOUNDARY = (
    "Structural in-process PEFT merge evidence only; no end-to-end logits "
    "equivalence claim is made for the 8B CPU merge."
)


class MergeAttestationError(ValueError):
    """Raised when merge inputs, process evidence, or published bytes drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MergeAttestationError(message)


def _canonical_bytes(payload: Any) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MergeAttestationError("Merge attestation is not canonical JSON") from exc
    return rendered.encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MergeAttestationError(f"Unable to hash merged output: {path}") from exc
    return digest.hexdigest()


def _payload_tree(root: Path) -> dict[str, Any]:
    _require(not root.is_symlink(), "Merged output root must not be a symlink")
    _require(root.is_dir(), f"Merged output root is missing: {root}")
    files: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise MergeAttestationError(f"Unable to scan merged output: {directory}") from exc
        for entry in entries:
            candidate = Path(entry.path)
            if entry.is_symlink():
                raise MergeAttestationError(
                    f"Merged output must not contain symlinks: {candidate}"
                )
            if entry.is_dir(follow_symlinks=False):
                visit(candidate)
                continue
            if not entry.is_file(follow_symlinks=False):
                raise MergeAttestationError(
                    f"Merged output contains a non-regular entry: {candidate}"
                )
            relative = candidate.relative_to(root).as_posix()
            if relative == ATTESTATION_FILENAME:
                continue
            files.append(
                {
                    "path": relative,
                    "size": candidate.stat().st_size,
                    "sha256": _sha256_file(candidate),
                }
            )

    visit(root)
    files.sort(key=lambda item: item["path"])
    _require(files, "Merged output payload contains no files")
    payload = {"algorithm": "sha256-canonical-file-inventory-v1", "files": files}
    return {**payload, "sha256": _canonical_sha256(payload)}


def _validate_semantic_evidence(value: Any) -> dict[str, Any]:
    expected_keys = {
        "method",
        "merge_completed",
        "lora_modules_before",
        "residual_lora_modules_after",
        "residual_lora_state_keys_after",
        "merged_parameter_tensors",
        "merged_parameter_count",
        "functional_equivalence_boundary",
    }
    _require(isinstance(value, dict), "Merge semantic evidence must be an object")
    _require(set(value) == expected_keys, "Merge semantic evidence schema mismatch")
    _require(value["method"] == SEMANTIC_METHOD, "Wrong merge semantic evidence method")
    _require(value["merge_completed"] is True, "PEFT merge did not complete")
    for key in ("lora_modules_before", "merged_parameter_tensors", "merged_parameter_count"):
        _require(
            isinstance(value[key], int)
            and not isinstance(value[key], bool)
            and value[key] > 0,
            f"Invalid merge semantic evidence field: {key}",
        )
    for key in ("residual_lora_modules_after", "residual_lora_state_keys_after"):
        _require(value[key] == 0, f"Merge left residual LoRA state: {key}")
    _require(
        value["functional_equivalence_boundary"] == SEMANTIC_BOUNDARY,
        "Merge semantic evidence boundary mismatch",
    )
    return dict(value)


def create_merge_attestation(
    merged_root: str | Path,
    *,
    binding: Mapping[str, Any],
    semantic_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Write the attestation before an unpublished merged directory is renamed."""

    root = Path(merged_root).resolve()
    path = root / ATTESTATION_FILENAME
    _require(not path.exists() and not path.is_symlink(), "Merge attestation already exists")
    _require(isinstance(binding, Mapping) and binding, "Merge binding is missing")
    payload_without_sha = {
        "schema_version": SCHEMA_VERSION,
        "attestation_type": "retrain_v2_merge",
        "binding": dict(binding),
        "semantic_evidence": _validate_semantic_evidence(dict(semantic_evidence)),
        "output_payload_fingerprint": _payload_tree(root),
    }
    payload = {
        **payload_without_sha,
        "canonical_payload_sha256": _canonical_sha256(payload_without_sha),
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return payload


def verify_merge_attestation(
    merged_root: str | Path, *, expected_binding: Mapping[str, Any]
) -> dict[str, Any]:
    """Revalidate an attested published merge, including crash recovery."""

    root = Path(merged_root).resolve()
    path = root / ATTESTATION_FILENAME
    _require(not path.is_symlink(), "Merge attestation must not be a symlink")
    _require(path.is_file(), f"Merge attestation is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MergeAttestationError("Unable to parse merge attestation") from exc
    expected_keys = {
        "schema_version",
        "attestation_type",
        "binding",
        "semantic_evidence",
        "output_payload_fingerprint",
        "canonical_payload_sha256",
    }
    _require(isinstance(payload, dict), "Merge attestation must contain an object")
    _require(set(payload) == expected_keys, "Merge attestation schema mismatch")
    _require(payload["schema_version"] == SCHEMA_VERSION, "Unsupported merge attestation schema")
    _require(payload["attestation_type"] == "retrain_v2_merge", "Wrong merge attestation type")
    without_sha = dict(payload)
    recorded_sha = without_sha.pop("canonical_payload_sha256")
    _require(
        isinstance(recorded_sha, str)
        and recorded_sha == _canonical_sha256(without_sha),
        "Merge attestation payload hash mismatch",
    )
    _require(payload["binding"] == dict(expected_binding), "Merge attestation binding mismatch")
    _validate_semantic_evidence(payload["semantic_evidence"])
    _require(
        payload["output_payload_fingerprint"] == _payload_tree(root),
        "Merge attestation output payload drift",
    )
    return payload


__all__ = [
    "ATTESTATION_FILENAME",
    "MergeAttestationError",
    "SEMANTIC_BOUNDARY",
    "SEMANTIC_METHOD",
    "create_merge_attestation",
    "verify_merge_attestation",
]
