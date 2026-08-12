"""Hash-bound single-GPU generation probe for chk4 Decision-SFT.

The probe is deliberately separate from the training DAG and has two phases:

``prepare``
    Verify an immutable chk4 release and bind every row in its sealed
    Decision-SFT validation split.  It also marks one deterministic short,
    medium, and long quick subset.  The resulting manifest contains only row
    locations, token counts, seeds, and hashes; it never copies prompt or
    teacher-response text.

``run``
    Re-open every hash-bound input, fingerprint the complete local model (and
    optional PEFT adapter), then generate one greedy and four sampled cases per
    validation row with Transformers on exactly one visible GPU.  It writes one
    JSONL row per completion plus a summary.  No dataset, reward, training
    configuration, model, or adapter is modified.

The output contract checked here is intentionally narrower than a semantic
evaluation: exactly one native ``</think>`` boundary must be followed by one
plain JSON object whose only keys are ``direction`` and ``magnitude_bp``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import statistics
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path
from typing import Any

from open_r1.provenance import fingerprint_artifact_path


PROBE_SCHEMA_VERSION = "chk4-decision-sft-generation-probe-v1"
SAMPLE_SCHEMA_VERSION = "chk4-decision-sft-generation-samples-v1"
SUMMARY_SCHEMA_VERSION = "chk4-decision-sft-generation-summary-v1"
RELEASE_SCHEMA_VERSION = "chk4-decision-training-release-v1"
BOUNDARY = "</think>"
VALIDATION_RELATIVE = "decision_sft/validation.jsonl"
UNIQUE_VALIDATION_RELATIVE = "manifests/unique/validation.jsonl"
BUCKETS = ("short", "medium", "long")
DECISION_KEYS = frozenset({"direction", "magnitude_bp"})
ALLOWED_DIRECTIONS = frozenset({"cut", "hold", "hike"})
ALLOWED_ACTION_MAGNITUDES = frozenset({25, 50, 75, 100})
DEFAULT_SEED = 20260810
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TAIL_TOKENS = 256
SAMPLED_GENERATIONS_PER_GROUP = 4
DEFAULT_SAMPLE_TEMPERATURE = 0.7
DEFAULT_SAMPLE_TOP_P = 0.9
FULL_REPETITION_LIMIT = 0.50
TAIL_REPETITION_LIMIT = 0.60
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ProbeError(RuntimeError):
    """A probe input or result is not trustworthy."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _validate_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProbeError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ProbeError(f"{label} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ProbeError(f"{label} root must be an object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[Mapping[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ProbeError(f"{label} is not a regular file: {path}")
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ProbeError(f"blank line in {label}: {path}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ProbeError(
                        f"non-object row in {label}: {path}:{line_number}"
                    )
                rows.append(value)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"invalid JSON in {label} {path}: {exc}") from exc
    except OSError as exc:
        raise ProbeError(f"cannot read {label} {path}: {exc}") from exc
    return rows


def _require_file_hash(path: Path, expected: object, *, label: str) -> str:
    expected_sha = _validate_sha256(expected, label=f"{label} SHA-256")
    if not path.is_file() or path.is_symlink():
        raise ProbeError(f"{label} is not a regular file: {path}")
    observed = sha256_file(path)
    if observed != expected_sha:
        raise ProbeError(
            f"{label} SHA-256 mismatch: expected={expected_sha}, observed={observed}"
        )
    return observed


def _release_file(
    release_root: Path,
    manifest: Mapping[str, Any],
    relative: str,
) -> tuple[Path, Mapping[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ProbeError("release manifest has no files object")
    record = files.get(relative)
    if not isinstance(record, Mapping) or record.get("path") != relative:
        raise ProbeError(f"release manifest has no exact file record for {relative}")
    path = (release_root / relative).resolve()
    if release_root != path and release_root not in path.parents:
        raise ProbeError(f"release child escapes release root: {relative}")
    _require_file_hash(path, record.get("sha256"), label=relative)
    return path, record


def _default_release_verifier(
    release_root: Path, *, expected_manifest_sha256: str
) -> Mapping[str, Any]:
    # Import lazily so pure metric/unit-test use does not load the publisher's
    # tokenizer and release-validation dependencies.
    from jobs.retrain_v2.publish_chk4_training_data import verify_release

    try:
        return verify_release(
            release_root, expected_manifest_sha256=expected_manifest_sha256
        )
    except Exception as exc:
        raise ProbeError(f"sealed release verification failed: {exc}") from exc


def _verify_release_and_manifest(
    release_root: Path,
    expected_manifest_sha256: str,
    *,
    release_verifier: Callable[..., Mapping[str, Any]] | None = None,
) -> tuple[Path, Path, Mapping[str, Any], str]:
    expected_sha = _validate_sha256(
        expected_manifest_sha256, label="release manifest SHA-256"
    )
    raw_root = release_root.expanduser()
    if raw_root.is_symlink():
        raise ProbeError(f"release root must not be a symlink: {raw_root}")
    root = raw_root.resolve()
    if not root.is_dir():
        raise ProbeError(f"release root is not a directory: {root}")
    manifest_path = root / "release_manifest.json"
    observed_sha = _require_file_hash(
        manifest_path, expected_sha, label="release manifest"
    )
    verifier = release_verifier or _default_release_verifier
    try:
        verified = verifier(root, expected_manifest_sha256=expected_sha)
    except ProbeError:
        raise
    except Exception as exc:
        raise ProbeError(f"sealed release verification failed: {exc}") from exc
    if not isinstance(verified, Mapping):
        raise ProbeError("sealed release verifier returned a non-object")
    disk_manifest = _read_json(manifest_path, label="release manifest")
    if canonical_json(verified) != canonical_json(disk_manifest):
        raise ProbeError("release verifier result differs from the bound manifest")
    if disk_manifest.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise ProbeError("unsupported chk4 release schema")
    if not (
        disk_manifest.get("immutable") is True
        and disk_manifest.get("quality_status") == "passed"
        and disk_manifest.get("training_ready") is True
    ):
        raise ProbeError("release is not immutable, passed, and training-ready")
    return root, manifest_path, disk_manifest, observed_sha


def _load_training_contract(
    config_path: Path,
    *,
    release_root: Path,
    release_manifest_path: Path,
    release_manifest_sha256: str,
) -> tuple[str, str, Mapping[str, Any]]:
    path = config_path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ProbeError(f"training config is not a regular file: {path}")
    try:
        import yaml

        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProbeError(f"cannot load training config {path}: {exc}") from exc
    if not isinstance(config, Mapping):
        raise ProbeError("training config root must be an object")
    if config.get("dataset_chk4_role") != "decision_sft":
        raise ProbeError("training config is not bound to chk4 decision_sft")
    if config.get("dataset_test_split") != "validation":
        raise ProbeError("training config test split must be validation")
    if config.get("dataset_prompt_column") != "prompt":
        raise ProbeError("training config prompt column must be prompt")
    if config.get("user_prompt_suffix") not in (None, ""):
        raise ProbeError("chk4 Decision-SFT forbids a user_prompt_suffix")
    if config.get("dataset_chk4_release_manifest_sha256") != release_manifest_sha256:
        raise ProbeError("training config/release manifest SHA-256 binding mismatch")

    def resolve_declared(value: object, *, label: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ProbeError(f"training config {label} must be a path")
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        # Training configs in this repository use repository/CWD-relative
        # paths.  Also accept a config-relative path for portable fixtures.
        cwd_candidate = (Path.cwd() / candidate).resolve()
        config_candidate = (path.parent / candidate).resolve()
        if cwd_candidate.exists() or not config_candidate.exists():
            return cwd_candidate
        return config_candidate

    declared_manifest = resolve_declared(
        config.get("dataset_chk4_release_manifest"),
        label="dataset_chk4_release_manifest",
    )
    if declared_manifest != release_manifest_path:
        raise ProbeError("training config points to a different release manifest")
    declared_dataset = resolve_declared(
        config.get("dataset_name"), label="dataset_name"
    )
    if declared_dataset != (release_root / "decision_sft").resolve():
        raise ProbeError("training config points to a different Decision-SFT dataset")
    system_prompt = config.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ProbeError("training config system_prompt must be non-empty")
    return system_prompt, sha256_file(path), config


def _tokenizer_binding(
    manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Mapping[str, Any]], str]:
    sources = manifest.get("sources")
    tokenizer_record = sources.get("tokenizer") if isinstance(sources, Mapping) else None
    if not isinstance(tokenizer_record, Mapping):
        raise ProbeError("release has no tokenizer source binding")
    tokenizer_path_value = tokenizer_record.get("path")
    files = tokenizer_record.get("files")
    if not isinstance(tokenizer_path_value, str) or not tokenizer_path_value:
        raise ProbeError("release tokenizer path is invalid")
    if not isinstance(files, Mapping) or not files:
        raise ProbeError("release tokenizer file binding is empty")
    tokenizer_path = Path(tokenizer_path_value).expanduser().resolve()
    if not tokenizer_path.is_dir() or tokenizer_path.is_symlink():
        raise ProbeError(f"release tokenizer directory is invalid: {tokenizer_path}")
    normalized: dict[str, Mapping[str, Any]] = {}
    for relative, raw_record in sorted(files.items()):
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ProbeError("release tokenizer has an invalid relative path")
        if not isinstance(raw_record, Mapping):
            raise ProbeError(f"invalid tokenizer file record: {relative}")
        item = (tokenizer_path / relative).resolve()
        if tokenizer_path not in item.parents:
            raise ProbeError(f"tokenizer file escapes its directory: {relative}")
        observed = _require_file_hash(
            item, raw_record.get("sha256"), label=f"tokenizer {relative}"
        )
        size = item.stat().st_size
        if raw_record.get("bytes") != size:
            raise ProbeError(f"tokenizer file size mismatch: {relative}")
        normalized[relative] = {"bytes": size, "sha256": observed}
    bundle_sha = sha256_text(canonical_json(normalized))
    return tokenizer_path, normalized, bundle_sha


def _messages(system_prompt: str, prompt: object) -> list[dict[str, str]]:
    if not isinstance(prompt, str) or not prompt:
        raise ProbeError("validation row prompt must be a non-empty string")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _prompt_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    try:
        value = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            truncation=False,
            return_dict=False,
        )
    except Exception as exc:
        raise ProbeError(f"chat-template tokenization failed: {exc}") from exc
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], Sequence) and not isinstance(
        value[0], (str, bytes)
    ):
        if len(value) != 1:
            raise ProbeError("chat template returned an unexpected batch")
        value = value[0]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProbeError("chat template returned invalid token IDs")
    try:
        result = [int(token_id) for token_id in value]
    except (TypeError, ValueError) as exc:
        raise ProbeError("chat template returned non-integer token IDs") from exc
    if not result or any(token_id < 0 for token_id in result):
        raise ProbeError("chat template returned empty or negative token IDs")
    return result


def _load_validation_rows(
    release_root: Path, manifest: Mapping[str, Any]
) -> tuple[
    Path,
    Mapping[str, Any],
    list[Mapping[str, Any]],
    Path,
    Mapping[str, Any],
    list[Mapping[str, Any]],
]:
    validation_path, validation_record = _release_file(
        release_root, manifest, VALIDATION_RELATIVE
    )
    unique_path, unique_record = _release_file(
        release_root, manifest, UNIQUE_VALIDATION_RELATIVE
    )
    rows = _read_jsonl(validation_path, label="Decision-SFT validation")
    unique_rows = _read_jsonl(unique_path, label="unique validation manifest")
    if len(rows) != len(unique_rows) or len(rows) < 3:
        raise ProbeError(
            "Decision-SFT and unique validation manifests must align and contain "
            "at least three rows"
        )
    for record, observed, label in (
        (validation_record, len(rows), VALIDATION_RELATIVE),
        (unique_record, len(unique_rows), UNIQUE_VALIDATION_RELATIVE),
    ):
        if record.get("rows") != observed:
            raise ProbeError(f"release row-count binding mismatch: {label}")
    return (
        validation_path,
        validation_record,
        rows,
        unique_path,
        unique_record,
        unique_rows,
    )


def build_sample_manifest(
    *,
    release_root: Path,
    release_manifest_sha256: str,
    tokenizer: Any,
    training_config: Path,
    seed: int = DEFAULT_SEED,
    release_verifier: Callable[..., Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Build a reproducible, text-free short/medium/long sample manifest."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ProbeError("seed must be a non-negative integer")
    root, manifest_path, manifest, observed_manifest_sha = (
        _verify_release_and_manifest(
            release_root,
            release_manifest_sha256,
            release_verifier=release_verifier,
        )
    )
    system_prompt, config_sha, _ = _load_training_contract(
        training_config,
        release_root=root,
        release_manifest_path=manifest_path,
        release_manifest_sha256=observed_manifest_sha,
    )
    tokenizer_path, tokenizer_files, tokenizer_bundle_sha = _tokenizer_binding(
        manifest
    )
    (
        validation_path,
        validation_record,
        rows,
        unique_path,
        unique_record,
        unique_rows,
    ) = _load_validation_rows(root, manifest)

    candidates: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for line_number, (row, unique) in enumerate(
        zip(rows, unique_rows, strict=True), start=1
    ):
        if set(row) != {"prompt", "response"}:
            raise ProbeError(
                f"unexpected Decision-SFT validation schema at line {line_number}"
            )
        sample_id = unique.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ProbeError(f"missing sample_id at validation line {line_number}")
        if sample_id in seen_sample_ids:
            raise ProbeError(f"duplicate validation sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)
        if unique.get("split") != "validation":
            raise ProbeError(f"split drift for validation sample {sample_id}")
        prompt = row.get("prompt")
        response = row.get("response")
        if not isinstance(response, str) or not response:
            raise ProbeError(f"empty response for validation sample {sample_id}")
        prompt_sha = sha256_text(str(prompt)) if isinstance(prompt, str) else None
        response_sha = sha256_text(response)
        if prompt_sha != unique.get("prompt_sha256"):
            raise ProbeError(f"prompt hash drift for validation sample {sample_id}")
        if response_sha != unique.get("response_sha256"):
            raise ProbeError(f"response hash drift for validation sample {sample_id}")
        prompt_token_count = len(_prompt_ids(tokenizer, _messages(system_prompt, prompt)))
        candidates.append(
            {
                "sample_id": sample_id,
                "split": "validation",
                "line_number": line_number,
                "prompt_sha256": prompt_sha,
                "response_sha256": response_sha,
                "prompt_token_count": prompt_token_count,
            }
        )

    ordered = sorted(
        candidates,
        key=lambda item: (int(item["prompt_token_count"]), str(item["sample_id"])),
    )
    indexes = (0, (len(ordered) - 1) // 2, len(ordered) - 1)
    quick_buckets = {
        str(ordered[index]["sample_id"]): bucket
        for bucket, index in zip(BUCKETS, indexes, strict=True)
    }
    if len(quick_buckets) != 3:
        raise ProbeError("short/medium/long selection did not produce unique samples")
    # Keep every sealed validation row for the formal gate.  ``quick_bucket``
    # marks the deterministic min/lower-median/max subset without changing the
    # original validation order.  Seeds are fixed in the manifest so neither
    # runtime ordering nor worker state can change a generation group.
    selected: list[dict[str, Any]] = []
    for index, item in enumerate(candidates):
        group_seed = seed + index * (SAMPLED_GENERATIONS_PER_GROUP + 1)
        selected.append(
            {
                **item,
                "quick_bucket": quick_buckets.get(str(item["sample_id"])),
                "greedy_seed": group_seed,
                "sample_seeds": [
                    group_seed + offset
                    for offset in range(1, SAMPLED_GENERATIONS_PER_GROUP + 1)
                ],
            }
        )

    config_path = training_config.expanduser().resolve()
    return {
        "schema_version": SAMPLE_SCHEMA_VERSION,
        "release": {
            "path": str(root),
            "release_id": manifest.get("release_id"),
            "manifest_path": str(manifest_path),
            "manifest_sha256": observed_manifest_sha,
            "schema_version": manifest.get("schema_version"),
        },
        "dataset": {
            "role": "decision_sft",
            "split": "validation",
            "path": str(validation_path),
            "sha256": validation_record["sha256"],
            "rows": len(rows),
            "unique_manifest_path": str(unique_path),
            "unique_manifest_sha256": unique_record["sha256"],
        },
        "prompt_contract": {
            "training_config": str(config_path),
            "training_config_sha256": config_sha,
            "system_prompt_sha256": sha256_text(system_prompt),
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "files": tokenizer_files,
            "bundle_sha256": tokenizer_bundle_sha,
        },
        "selection": {
            "algorithm": "validation-token-length-min-lower-median-max-v1",
            "source_rows": len(rows),
            "formal_rows": len(rows),
            "quick_rows": 3,
            "quick_sample_ids": {
                bucket: str(ordered[index]["sample_id"])
                for bucket, index in zip(BUCKETS, indexes, strict=True)
            },
            "base_seed": seed,
            "formal_generation_contract": {
                "greedy_per_row": 1,
                "sampled_per_row": SAMPLED_GENERATIONS_PER_GROUP,
                "sample_temperature": DEFAULT_SAMPLE_TEMPERATURE,
                "sample_top_p": DEFAULT_SAMPLE_TOP_P,
            },
        },
        "samples": selected,
    }


def _normalize_eos_ids(eos_token_ids: int | Iterable[int] | None) -> set[int]:
    if eos_token_ids is None:
        return set()
    if isinstance(eos_token_ids, Integral) and not isinstance(eos_token_ids, bool):
        result = {int(eos_token_ids)}
    elif isinstance(eos_token_ids, (str, bytes, Mapping)) or not isinstance(
        eos_token_ids, Iterable
    ):
        raise ProbeError("EOS token IDs must be an integer or iterable of integers")
    else:
        values = list(eos_token_ids)
        if any(
            not isinstance(value, Integral) or isinstance(value, bool)
            for value in values
        ):
            raise ProbeError("EOS token IDs contain a non-integer")
        result = {int(value) for value in values}
    if any(value < 0 for value in result):
        raise ProbeError("EOS token IDs contain a negative integer")
    return result


def decode_completion_preserving_boundary(
    tokenizer: Any,
    generated_token_ids: Sequence[int],
    eos_token_ids: int | Iterable[int] | None,
) -> str:
    try:
        raw_ids = [int(value) for value in generated_token_ids]
    except (TypeError, ValueError) as exc:
        raise ProbeError("generated token IDs contain a non-integer") from exc
    eos_ids = _normalize_eos_ids(eos_token_ids)
    content_ids = raw_ids[:-1] if raw_ids and raw_ids[-1] in eos_ids else raw_ids
    try:
        text = tokenizer.decode(
            content_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception as exc:
        raise ProbeError(f"completion decoding failed: {exc}") from exc
    if not isinstance(text, str):
        raise ProbeError("tokenizer decode did not return text")
    return text


def _ngram_repetition(token_ids: Sequence[int], *, n: int = 4) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if len(token_ids) < n:
        return 0.0
    grams = [
        tuple(token_ids[index : index + n])
        for index in range(len(token_ids) - n + 1)
    ]
    return 1.0 - len(set(grams)) / len(grams)


def has_strict_periodic_tail(token_ids: Sequence[int]) -> bool:
    """Detect a literal repeated suffix of at least 48 tokens and 3 periods."""

    size = len(token_ids)
    for period in range(1, min(128, size // 3) + 1):
        if period * 3 < 48:
            continue
        unit = list(token_ids[size - period :])
        repeats = 1
        cursor = size - 2 * period
        while cursor >= 0 and list(token_ids[cursor : cursor + period]) == unit:
            repeats += 1
            cursor -= period
        if repeats >= 3 and repeats * period >= 48:
            return True
    return False


def _parse_decision_json(suffix: str) -> Mapping[str, Any]:
    result: dict[str, Any] = {
        "plain_json": False,
        "json_object": False,
        "exact_keys": False,
        "decision_domain_valid": False,
        "parsed_decision": None,
        "json_error": None,
    }
    stripped = suffix.lstrip()
    if not stripped.startswith("{"):
        result["json_error"] = "answer_does_not_start_with_json_object"
        return result
    try:
        value, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError as exc:
        result["json_error"] = f"invalid_json:{exc.msg}"
        return result
    if stripped[end:].strip():
        result["json_error"] = "trailing_non_whitespace_after_json"
        return result
    result["plain_json"] = True
    if not isinstance(value, Mapping):
        result["json_error"] = "json_root_is_not_object"
        return result
    result["json_object"] = True
    exact_keys = set(value) == DECISION_KEYS
    result["exact_keys"] = exact_keys
    if not exact_keys:
        result["json_error"] = "decision_keys_are_not_exact"
        return result
    direction = value.get("direction")
    magnitude = value.get("magnitude_bp")
    magnitude_is_int = isinstance(magnitude, int) and not isinstance(magnitude, bool)
    domain_valid = (
        isinstance(direction, str)
        and direction in ALLOWED_DIRECTIONS
        and magnitude_is_int
        and (
            (direction == "hold" and magnitude == 0)
            or (direction in {"cut", "hike"} and magnitude in ALLOWED_ACTION_MAGNITUDES)
        )
    )
    result["decision_domain_valid"] = domain_valid
    result["parsed_decision"] = dict(value)
    if not domain_valid:
        result["json_error"] = "decision_values_outside_contract"
    return result


def analyze_completion(
    *,
    text: str,
    generated_token_ids: Sequence[int],
    eos_token_ids: int | Iterable[int] | None,
    max_new_tokens: int,
    tail_tokens: int = DEFAULT_TAIL_TOKENS,
) -> Mapping[str, Any]:
    if max_new_tokens <= 0 or tail_tokens <= 0:
        raise ProbeError("token limits must be positive")
    if not isinstance(text, str):
        raise ProbeError("completion text must be a string")
    try:
        raw_ids = [int(value) for value in generated_token_ids]
    except (TypeError, ValueError) as exc:
        raise ProbeError("generated token IDs contain a non-integer") from exc
    eos_ids = _normalize_eos_ids(eos_token_ids)
    hit_eos = bool(raw_ids and raw_ids[-1] in eos_ids)
    content_ids = raw_ids[:-1] if hit_eos else raw_ids
    cap_reached = len(raw_ids) >= max_new_tokens and not hit_eos
    boundary_count = text.count(BOUNDARY)
    reasoning = text.split(BOUNDARY, 1)[0] if boundary_count == 1 else ""
    suffix = text.split(BOUNDARY, 1)[1] if boundary_count == 1 else ""
    parsed = _parse_decision_json(suffix) if boundary_count == 1 else {
        "plain_json": False,
        "json_object": False,
        "exact_keys": False,
        "decision_domain_valid": False,
        "parsed_decision": None,
        "json_error": "think_boundary_count_is_not_one",
    }
    full_repetition = _ngram_repetition(content_ids, n=4)
    tail_ids = content_ids[-tail_tokens:]
    tail_repetition = _ngram_repetition(tail_ids, n=4)
    periodic = has_strict_periodic_tail(content_ids)
    finite = bool(raw_ids) and bool(text) and all(value >= 0 for value in raw_ids)
    contract_valid = bool(
        finite
        and boundary_count == 1
        and reasoning.strip()
        and parsed["plain_json"]
        and parsed["exact_keys"]
        and parsed["decision_domain_valid"]
    )
    repetition_valid = (
        not periodic
        and full_repetition < FULL_REPETITION_LIMIT
        and tail_repetition < TAIL_REPETITION_LIMIT
    )
    delivery_valid = contract_valid and hit_eos and not cap_reached and repetition_valid
    failures: list[str] = []
    if not finite:
        failures.append("empty_or_invalid_completion")
    if boundary_count != 1:
        failures.append("think_boundary_count_not_one")
    elif not reasoning.strip():
        failures.append("empty_reasoning")
    if boundary_count == 1 and not parsed["plain_json"]:
        failures.append("answer_not_plain_single_json")
    if boundary_count == 1 and parsed["plain_json"] and not parsed["exact_keys"]:
        failures.append("decision_keys_not_exact")
    if boundary_count == 1 and parsed["exact_keys"] and not parsed["decision_domain_valid"]:
        failures.append("decision_domain_invalid")
    if not hit_eos:
        failures.append("missing_terminal_eos")
    if cap_reached:
        failures.append("completion_length_cap")
    if periodic:
        failures.append("strict_periodic_tail")
    if full_repetition >= FULL_REPETITION_LIMIT:
        failures.append("full_4gram_repetition_ge_0.50")
    if tail_repetition >= TAIL_REPETITION_LIMIT:
        failures.append("tail_4gram_repetition_ge_0.60")
    return {
        "status": "valid" if delivery_valid else "invalid",
        "raw_generated_token_count": len(raw_ids),
        "completion_token_count": len(content_ids),
        "hit_eos": hit_eos,
        "cap_reached": cap_reached,
        "think_boundary_count": boundary_count,
        "has_nonempty_reasoning": bool(reasoning.strip()),
        **parsed,
        "contract_valid": contract_valid,
        "full_token_4gram_repetition": round(full_repetition, 8),
        "tail_token_count": len(tail_ids),
        "tail_token_4gram_repetition": round(tail_repetition, 8),
        "strict_periodic_tail": periodic,
        "repetition_valid": repetition_valid,
        "delivery_valid": delivery_valid,
        "failure_reasons": failures,
    }


def replay_decision_dense_v2(
    completion: str, target: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Replay the deterministic chk4 reward without writing its reward log.

    This uses the training reward's own parser, then mirrors its coefficients.
    EOS, cap, repetition, and this probe's stricter plain-JSON gate are not
    reward coefficients, matching the training reward implementation.
    """

    from open_r1.trainer.rewards.reward_funcs.decision_reward_v2 import (
        parse_decision_json,
    )

    prediction = parse_decision_json(completion)
    if prediction is None:
        return {
            "reward_name": "decision_dense_v2",
            "reward": 0.0,
            "direction_correct": False,
            "magnitude_score": 0.0,
            "exact": False,
        }
    predicted_direction, predicted_magnitude = prediction
    target_direction = target.get("direction")
    target_magnitude = target.get("magnitude_bp")
    direction_correct = predicted_direction == target_direction
    magnitude_score = 0.0
    if direction_correct:
        magnitude_score = max(
            0.0,
            1.0 - abs(int(predicted_magnitude) - int(target_magnitude)) / 100.0,
        )
    exact = (
        predicted_direction == target_direction
        and predicted_magnitude == target_magnitude
    )
    reward = (
        0.05
        + 0.45 * float(direction_correct)
        + 0.30 * magnitude_score
        + 0.20 * float(exact)
    )
    return {
        "reward_name": "decision_dense_v2",
        "reward": round(max(0.0, min(1.0, reward)), 8),
        "direction_correct": direction_correct,
        "magnitude_score": round(magnitude_score, 8),
        "exact": exact,
    }


def _generation_cases(
    sample_manifest: Mapping[str, Any], *, probe_mode: str
) -> list[Mapping[str, Any]]:
    if probe_mode not in {"formal", "quick"}:
        raise ProbeError("probe_mode must be formal or quick")
    samples = [
        sample
        for sample in sample_manifest["samples"]
        if probe_mode == "formal" or sample.get("quick_bucket") is not None
    ]
    cases: list[Mapping[str, Any]] = []
    for sample in samples:
        cases.append(
            {
                "sample": sample,
                "generation_mode": "greedy",
                "generation_index": 0,
                "seed": int(sample["greedy_seed"]),
            }
        )
        for generation_index, seed in enumerate(sample["sample_seeds"], start=1):
            cases.append(
                {
                    "sample": sample,
                    "generation_mode": "sampled",
                    "generation_index": generation_index,
                    "seed": int(seed),
                }
            )
    return cases


def _validate_sample_manifest(
    path: Path, expected_sha256: str
) -> tuple[Mapping[str, Any], str]:
    observed = _require_file_hash(path, expected_sha256, label="sample manifest")
    manifest = _read_json(path, label="sample manifest")
    if manifest.get("schema_version") != SAMPLE_SCHEMA_VERSION:
        raise ProbeError("unsupported sample manifest schema")
    samples = manifest.get("samples")
    dataset = manifest.get("dataset")
    expected_rows = dataset.get("rows") if isinstance(dataset, Mapping) else None
    if (
        not isinstance(samples, list)
        or not isinstance(expected_rows, int)
        or isinstance(expected_rows, bool)
        or expected_rows < 3
        or len(samples) != expected_rows
    ):
        raise ProbeError("sample manifest must contain every bound validation row")
    quick_buckets = [
        item.get("quick_bucket")
        for item in samples
        if isinstance(item, Mapping) and item.get("quick_bucket") is not None
    ]
    if sorted(quick_buckets) != sorted(BUCKETS):
        raise ProbeError("sample manifest quick buckets must be short, medium, long")
    sample_ids = [item.get("sample_id") for item in samples if isinstance(item, Mapping)]
    if len(sample_ids) != expected_rows or len(set(sample_ids)) != expected_rows:
        raise ProbeError("sample manifest IDs are missing or duplicated")
    all_seeds: list[int] = []
    for item in samples:
        if not isinstance(item, Mapping):
            raise ProbeError("sample manifest contains a non-object sample")
        greedy_seed = item.get("greedy_seed")
        sample_seeds = item.get("sample_seeds")
        if (
            not isinstance(greedy_seed, int)
            or isinstance(greedy_seed, bool)
            or not isinstance(sample_seeds, list)
            or len(sample_seeds) != SAMPLED_GENERATIONS_PER_GROUP
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in sample_seeds
            )
        ):
            raise ProbeError("sample manifest generation seeds are invalid")
        if item.get("split") != "validation":
            raise ProbeError("sample manifest contains a non-validation row")
        prompt_tokens = item.get("prompt_token_count")
        if (
            not isinstance(prompt_tokens, int)
            or isinstance(prompt_tokens, bool)
            or prompt_tokens <= 0
        ):
            raise ProbeError("sample manifest prompt token count is invalid")
        all_seeds.extend([greedy_seed, *sample_seeds])
    if len(set(all_seeds)) != len(all_seeds):
        raise ProbeError("sample manifest generation seeds are duplicated")
    ordered = sorted(
        samples,
        key=lambda item: (int(item["prompt_token_count"]), str(item["sample_id"])),
    )
    expected_quick = {
        bucket: ordered[index]["sample_id"]
        for bucket, index in zip(
            BUCKETS, (0, (len(ordered) - 1) // 2, len(ordered) - 1), strict=True
        )
    }
    observed_quick = {
        str(item["quick_bucket"]): item["sample_id"]
        for item in samples
        if item.get("quick_bucket") is not None
    }
    if observed_quick != expected_quick:
        raise ProbeError("sample manifest quick length selection drift")
    selection = manifest.get("selection")
    formal_contract = (
        selection.get("formal_generation_contract")
        if isinstance(selection, Mapping)
        else None
    )
    if formal_contract != {
        "greedy_per_row": 1,
        "sampled_per_row": SAMPLED_GENERATIONS_PER_GROUP,
        "sample_temperature": DEFAULT_SAMPLE_TEMPERATURE,
        "sample_top_p": DEFAULT_SAMPLE_TOP_P,
    }:
        raise ProbeError("sample manifest formal generation contract drift")
    return manifest, observed


def _load_bound_rows(
    sample_manifest: Mapping[str, Any],
    *,
    release_verifier: Callable[..., Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Any], str]:
    release = sample_manifest.get("release")
    if not isinstance(release, Mapping):
        raise ProbeError("sample manifest has no release binding")
    root, manifest_path, manifest, manifest_sha = _verify_release_and_manifest(
        Path(str(release.get("path"))),
        str(release.get("manifest_sha256")),
        release_verifier=release_verifier,
    )
    if str(manifest_path) != release.get("manifest_path"):
        raise ProbeError("sample manifest release path drift")
    dataset = sample_manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ProbeError("sample manifest has no dataset binding")
    (
        validation_path,
        validation_record,
        rows,
        unique_path,
        unique_record,
        unique_rows,
    ) = _load_validation_rows(root, manifest)
    expected_dataset = {
        "role": "decision_sft",
        "split": "validation",
        "path": str(validation_path),
        "sha256": validation_record["sha256"],
        "rows": len(rows),
        "unique_manifest_path": str(unique_path),
        "unique_manifest_sha256": unique_record["sha256"],
    }
    if canonical_json(dataset) != canonical_json(expected_dataset):
        raise ProbeError("sample manifest dataset binding drift")
    bound: dict[str, Mapping[str, Any]] = {}
    for sample in sample_manifest["samples"]:
        if not isinstance(sample, Mapping):
            raise ProbeError("sample manifest contains a non-object sample")
        line_number = sample.get("line_number")
        if not isinstance(line_number, int) or isinstance(line_number, bool) or line_number <= 0:
            raise ProbeError("sample manifest contains an invalid line number")
        try:
            row = rows[line_number - 1]
            unique = unique_rows[line_number - 1]
        except IndexError as exc:
            raise ProbeError("sample line number is outside validation") from exc
        sample_id = sample.get("sample_id")
        if unique.get("sample_id") != sample_id:
            raise ProbeError(f"sample ID drift at validation:{line_number}")
        for field, hash_key in (
            ("prompt", "prompt_sha256"),
            ("response", "response_sha256"),
        ):
            value = row.get(field)
            if not isinstance(value, str) or sha256_text(value) != sample.get(hash_key):
                raise ProbeError(
                    f"sample field drift at validation:{line_number}:{field}"
                )
        direction = unique.get("direction")
        magnitude = unique.get("magnitude_bp")
        if (
            direction not in ALLOWED_DIRECTIONS
            or not isinstance(magnitude, int)
            or isinstance(magnitude, bool)
            or (
                (direction == "hold" and magnitude != 0)
                or (
                    direction in {"cut", "hike"}
                    and magnitude not in ALLOWED_ACTION_MAGNITUDES
                )
            )
        ):
            raise ProbeError(f"invalid target at validation:{line_number}")
        bound[str(sample_id)] = {
            "row": row,
            "target": {"direction": direction, "magnitude_bp": magnitude},
            "meeting_date": unique.get("meeting_date"),
        }
    return bound, manifest, manifest_sha


def _require_single_visible_gpu(environ: Mapping[str, str] | None = None) -> str:
    environment = os.environ if environ is None else environ
    raw = environment.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        raise ProbeError(
            "CUDA_VISIBLE_DEVICES must explicitly expose exactly one GPU"
        )
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    if len(devices) != 1 or devices[0] == "-1":
        raise ProbeError(
            "CUDA_VISIBLE_DEVICES must explicitly expose exactly one GPU"
        )
    return devices[0]


def _write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        handle.write("\n")


def summarize_results(
    results: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, Any],
    probe_mode: str,
) -> Mapping[str, Any]:
    if probe_mode not in {"formal", "quick"}:
        raise ProbeError("probe_mode must be formal or quick")
    if not results:
        raise ProbeError("cannot summarize an empty probe")
    indexed: dict[str, list[Mapping[str, Any]]] = {}
    seen_cases: set[tuple[str, str, int]] = set()
    for row in results:
        sample_id = str(row.get("sample_id"))
        generation_mode = str(row.get("generation_mode"))
        generation_index = int(row.get("generation_index", -1))
        key = (sample_id, generation_mode, generation_index)
        if key in seen_cases:
            raise ProbeError(f"duplicate generation case: {key}")
        seen_cases.add(key)
        indexed.setdefault(sample_id, []).append(row)
    expected_group_size = 1 + SAMPLED_GENERATIONS_PER_GROUP
    for sample_id, rows in indexed.items():
        greedy = [row for row in rows if row.get("generation_mode") == "greedy"]
        sampled = [row for row in rows if row.get("generation_mode") == "sampled"]
        if len(rows) != expected_group_size or len(greedy) != 1 or len(sampled) != SAMPLED_GENERATIONS_PER_GROUP:
            raise ProbeError(f"incomplete generation group for {sample_id}")
        if sorted(int(row.get("generation_index", -1)) for row in sampled) != list(
            range(1, SAMPLED_GENERATIONS_PER_GROUP + 1)
        ):
            raise ProbeError(f"sampled generation indexes drift for {sample_id}")

    total = len(results)
    prompt_counts = [int(row["prompt_token_count"]) for row in results]
    completion_counts = [int(row["completion_token_count"]) for row in results]
    quality_passed = all(bool(row.get("delivery_valid")) for row in results)
    sampled_group_rows: list[Mapping[str, Any]] = []
    sampled_rewards: list[float] = []
    for sample_id, rows in indexed.items():
        sampled = sorted(
            (row for row in rows if row.get("generation_mode") == "sampled"),
            key=lambda row: int(row["generation_index"]),
        )
        rewards = [float(row["decision_dense_v2_reward"]) for row in sampled]
        sampled_rewards.extend(rewards)
        reward_std = statistics.pstdev(rewards)
        sampled_group_rows.append(
            {
                "sample_id": sample_id,
                "quick_bucket": sampled[0].get("quick_bucket"),
                "rewards": rewards,
                "mean_reward": round(statistics.fmean(rewards), 8),
                "reward_std": round(reward_std, 8),
                "zero_reward_group": all(abs(value) <= 1e-12 for value in rewards),
                "zero_std_group": reward_std <= 1e-12,
            }
        )
    zero_reward_groups = sum(
        bool(group["zero_reward_group"]) for group in sampled_group_rows
    )
    zero_std_groups = sum(bool(group["zero_std_group"]) for group in sampled_group_rows)
    group_count = len(sampled_group_rows)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "complete",
        "quality_status": "passed" if quality_passed else "failed",
        "probe_mode": probe_mode,
        "cases": total,
        "validation_groups": group_count,
        "greedy_cases": sum(
            row.get("generation_mode") == "greedy" for row in results
        ),
        "sampled_cases": sum(
            row.get("generation_mode") == "sampled" for row in results
        ),
        "sampled_generations_per_group": SAMPLED_GENERATIONS_PER_GROUP,
        "quick_buckets": list(BUCKETS),
        "seeds": [int(row["seed"]) for row in results],
        "provenance": dict(provenance),
        "rates": {
            "exactly_one_think_boundary": sum(
                row.get("think_boundary_count") == 1 for row in results
            )
            / total,
            "plain_json": sum(bool(row.get("plain_json")) for row in results) / total,
            "exact_keys": sum(bool(row.get("exact_keys")) for row in results) / total,
            "decision_domain_valid": sum(
                bool(row.get("decision_domain_valid")) for row in results
            )
            / total,
            "contract_valid": sum(bool(row.get("contract_valid")) for row in results)
            / total,
            "eos": sum(bool(row.get("hit_eos")) for row in results) / total,
            "cap": sum(bool(row.get("cap_reached")) for row in results) / total,
            "strict_periodic_tail": sum(
                bool(row.get("strict_periodic_tail")) for row in results
            )
            / total,
            "delivery_valid": sum(bool(row.get("delivery_valid")) for row in results)
            / total,
        },
        "token_counts": {
            "prompt": {
                "min": min(prompt_counts),
                "mean": round(sum(prompt_counts) / total, 3),
                "max": max(prompt_counts),
            },
            "completion": {
                "min": min(completion_counts),
                "mean": round(sum(completion_counts) / total, 3),
                "max": max(completion_counts),
            },
        },
        "repetition": {
            "thresholds": {
                "full_4gram_lt": FULL_REPETITION_LIMIT,
                "tail_4gram_lt": TAIL_REPETITION_LIMIT,
            },
            "mean_full_token_4gram": round(
                sum(float(row["full_token_4gram_repetition"]) for row in results)
                / total,
                8,
            ),
            "max_full_token_4gram": max(
                float(row["full_token_4gram_repetition"]) for row in results
            ),
            "mean_tail_token_4gram": round(
                sum(float(row["tail_token_4gram_repetition"]) for row in results)
                / total,
                8,
            ),
            "max_tail_token_4gram": max(
                float(row["tail_token_4gram_repetition"]) for row in results
            ),
        },
        "decision_dense_v2_sampled_groups": {
            "reward_replay": (
                "exact deterministic decision_dense_v2 formula; no reward log write"
            ),
            "groups": group_count,
            "sampled_completions": len(sampled_rewards),
            "mean_reward": round(statistics.fmean(sampled_rewards), 8),
            "zero_reward_groups": zero_reward_groups,
            "zero_reward_group_rate": zero_reward_groups / group_count,
            "zero_std_groups": zero_std_groups,
            "zero_std_group_rate": zero_std_groups / group_count,
            "per_group": sampled_group_rows,
        },
        "failed_cases": [
            {
                "sample_id": row.get("sample_id"),
                "quick_bucket": row.get("quick_bucket"),
                "generation_mode": row.get("generation_mode"),
                "generation_index": row.get("generation_index"),
                "failure_reasons": list(row.get("failure_reasons") or []),
            }
            for row in results
            if not row.get("delivery_valid")
        ],
    }


def run_generation_probe(
    *,
    base_model: Path,
    adapter: Path | None,
    sample_manifest_path: Path,
    sample_manifest_sha256: str,
    output_dir: Path,
    model_label: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    tail_tokens: int = DEFAULT_TAIL_TOKENS,
    probe_mode: str = "formal",
    sample_temperature: float = DEFAULT_SAMPLE_TEMPERATURE,
    sample_top_p: float = DEFAULT_SAMPLE_TOP_P,
    load_in_4bit: bool = True,
    attn_implementation: str = "sdpa",
) -> Mapping[str, Any]:
    """Generate fixed validation groups without mutating any bound input."""

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", model_label):
        raise ProbeError("model_label contains unsupported characters")
    if max_new_tokens <= 0 or tail_tokens <= 0:
        raise ProbeError("token limits must be positive")
    if probe_mode not in {"formal", "quick"}:
        raise ProbeError("probe_mode must be formal or quick")
    if not math.isfinite(sample_temperature) or sample_temperature <= 0:
        raise ProbeError("sample_temperature must be finite and positive")
    if not math.isfinite(sample_top_p) or not 0 < sample_top_p <= 1:
        raise ProbeError("sample_top_p must be finite and in (0, 1]")
    if probe_mode == "formal" and (
        sample_temperature != DEFAULT_SAMPLE_TEMPERATURE
        or sample_top_p != DEFAULT_SAMPLE_TOP_P
    ):
        raise ProbeError(
            "formal mode is pinned to sample_temperature=0.7 and sample_top_p=0.9"
        )
    if output_dir.exists():
        raise ProbeError(f"probe output already exists: {output_dir}")
    visible_gpu = _require_single_visible_gpu()
    sample_manifest, observed_sample_sha = _validate_sample_manifest(
        sample_manifest_path.expanduser().resolve(), sample_manifest_sha256
    )
    bound_rows, release_manifest, release_manifest_sha = _load_bound_rows(
        sample_manifest
    )
    release_root = Path(str(sample_manifest["release"]["path"]))
    release_manifest_path = Path(str(sample_manifest["release"]["manifest_path"]))
    prompt_contract = sample_manifest.get("prompt_contract")
    if not isinstance(prompt_contract, Mapping):
        raise ProbeError("sample manifest has no prompt contract")
    training_config_path = Path(str(prompt_contract.get("training_config")))
    system_prompt, config_sha, _ = _load_training_contract(
        training_config_path,
        release_root=release_root,
        release_manifest_path=release_manifest_path,
        release_manifest_sha256=release_manifest_sha,
    )
    if config_sha != prompt_contract.get("training_config_sha256"):
        raise ProbeError("training config changed after sample preparation")
    if sha256_text(system_prompt) != prompt_contract.get("system_prompt_sha256"):
        raise ProbeError("training system prompt changed after sample preparation")
    tokenizer_path, tokenizer_files, tokenizer_bundle_sha = _tokenizer_binding(
        release_manifest
    )
    if canonical_json(tokenizer_files) != canonical_json(
        sample_manifest.get("tokenizer", {}).get("files")
    ):
        raise ProbeError("tokenizer file binding changed after sample preparation")
    if tokenizer_bundle_sha != sample_manifest.get("tokenizer", {}).get(
        "bundle_sha256"
    ):
        raise ProbeError("tokenizer bundle hash changed after sample preparation")

    base_model = base_model.expanduser().resolve()
    try:
        base_fingerprint = fingerprint_artifact_path(base_model)
        adapter_fingerprint = (
            fingerprint_artifact_path(adapter.expanduser().resolve())
            if adapter is not None
            else None
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise ProbeError(f"cannot fingerprint model artifact: {exc}") from exc
    provenance = {
        "sample_manifest_sha256": observed_sample_sha,
        "release_manifest_sha256": release_manifest_sha,
        "validation_dataset_sha256": sample_manifest["dataset"]["sha256"],
        "unique_validation_manifest_sha256": sample_manifest["dataset"][
            "unique_manifest_sha256"
        ],
        "training_config_sha256": config_sha,
        "tokenizer_bundle_sha256": tokenizer_bundle_sha,
        "base_model": base_fingerprint,
        "adapter": adapter_fingerprint,
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    launch = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "initializing",
        "created_at_utc": utc_now(),
        "model_label": model_label,
        "provenance": provenance,
        "generation": {
            "probe_mode": probe_mode,
            "greedy_per_row": 1,
            "sampled_per_row": SAMPLED_GENERATIONS_PER_GROUP,
            "max_new_tokens": max_new_tokens,
            "tail_tokens": tail_tokens,
            "sample_temperature": sample_temperature,
            "sample_top_p": sample_top_p,
            "load_in_4bit": load_in_4bit,
            "attn_implementation": attn_implementation,
        },
        "runtime": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "cuda_visible_devices": visible_gpu,
        },
    }
    _write_json(output_dir / "launch.json", launch, exclusive=True)

    result_path = output_dir / "results.jsonl"
    results: list[Mapping[str, Any]] = []
    started = datetime.now(timezone.utc)
    model: Any = None
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise ProbeError("CUDA is required for the Decision-SFT probe")
        if torch.cuda.device_count() != 1:
            raise ProbeError("the probe process must see exactly one CUDA device")
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, trust_remote_code=False
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ProbeError("tokenizer has neither pad nor EOS token")
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(
            str(base_model),
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            quantization_config=quantization_config,
            device_map={"": 0},
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                str(adapter.expanduser().resolve()),
                is_trainable=False,
                local_files_only=True,
            )
        model.eval()
        model.config.use_cache = True
        eos_value = getattr(model.generation_config, "eos_token_id", None)
        if eos_value is None:
            eos_value = tokenizer.eos_token_id
        eos_ids = _normalize_eos_ids(eos_value)
        if not eos_ids:
            raise ProbeError("model/tokenizer has no EOS token ID")

        torch.cuda.reset_peak_memory_stats()
        with result_path.open("x", encoding="utf-8") as result_handle:
            for case in _generation_cases(sample_manifest, probe_mode=probe_mode):
                sample = case["sample"]
                sample_id = str(sample["sample_id"])
                bound = bound_rows[sample_id]
                row = bound["row"]
                prompt_ids = _prompt_ids(
                    tokenizer, _messages(system_prompt, row.get("prompt"))
                )
                if len(prompt_ids) != sample.get("prompt_token_count"):
                    raise ProbeError(f"prompt token-count drift for {sample_id}")
                context_limit = int(
                    getattr(model.config, "max_position_embeddings", 0) or 0
                )
                if context_limit and len(prompt_ids) + max_new_tokens > context_limit:
                    raise ProbeError(
                        f"context overflow for {sample_id}: prompt={len(prompt_ids)}, "
                        f"completion={max_new_tokens}, limit={context_limit}"
                    )
                seed = int(case["seed"])
                transformers.set_seed(seed)
                input_ids = torch.tensor(
                    [prompt_ids], dtype=torch.long, device="cuda:0"
                )
                attention_mask = torch.ones_like(input_ids)
                generation_kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": sorted(eos_ids),
                    "use_cache": True,
                    "do_sample": case["generation_mode"] == "sampled",
                }
                if case["generation_mode"] == "sampled":
                    generation_kwargs.update(
                        temperature=sample_temperature, top_p=sample_top_p
                    )
                with torch.inference_mode():
                    sequences = model.generate(**generation_kwargs)
                generated_ids = sequences[0, input_ids.shape[1] :].tolist()
                completion = decode_completion_preserving_boundary(
                    tokenizer, generated_ids, eos_ids
                )
                metrics = analyze_completion(
                    text=completion,
                    generated_token_ids=generated_ids,
                    eos_token_ids=eos_ids,
                    max_new_tokens=max_new_tokens,
                    tail_tokens=tail_tokens,
                )
                reward = replay_decision_dense_v2(completion, bound["target"])
                result = {
                    "schema_version": PROBE_SCHEMA_VERSION,
                    "model_label": model_label,
                    "sample_id": sample_id,
                    "split": "validation",
                    "line_number": sample["line_number"],
                    "quick_bucket": sample["quick_bucket"],
                    "generation_mode": case["generation_mode"],
                    "generation_index": case["generation_index"],
                    "seed": seed,
                    "generation_parameters": {
                        "do_sample": case["generation_mode"] == "sampled",
                        "temperature": (
                            sample_temperature
                            if case["generation_mode"] == "sampled"
                            else 0.0
                        ),
                        "top_p": (
                            sample_top_p
                            if case["generation_mode"] == "sampled"
                            else 1.0
                        ),
                        "max_new_tokens": max_new_tokens,
                    },
                    "prompt_token_count": len(prompt_ids),
                    "completion_sha256": sha256_text(completion),
                    "completion": completion,
                    "provenance": {
                        "sample_manifest_sha256": observed_sample_sha,
                        "release_manifest_sha256": release_manifest_sha,
                        "validation_dataset_sha256": sample_manifest["dataset"][
                            "sha256"
                        ],
                        "training_config_sha256": config_sha,
                        "tokenizer_bundle_sha256": tokenizer_bundle_sha,
                        "base_model_sha256": base_fingerprint["sha256"],
                        "adapter_sha256": (
                            adapter_fingerprint["sha256"]
                            if adapter_fingerprint is not None
                            else None
                        ),
                    },
                    **metrics,
                    "decision_dense_v2_reward": reward["reward"],
                    "decision_direction_correct": reward["direction_correct"],
                    "decision_magnitude_score": reward["magnitude_score"],
                    "decision_exact": reward["exact"],
                }
                result_handle.write(canonical_json(result) + "\n")
                result_handle.flush()
                os.fsync(result_handle.fileno())
                results.append(result)
                del input_ids, attention_mask, sequences

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        summary = dict(
            summarize_results(
                results, provenance=provenance, probe_mode=probe_mode
            )
        )
        summary.update(
            {
                "created_at_utc": utc_now(),
                "elapsed_seconds": round(elapsed, 3),
                "model_label": model_label,
                "generation": launch["generation"],
                "runtime": {
                    **launch["runtime"],
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "peak_allocated_gib": round(
                        torch.cuda.max_memory_allocated() / 1024**3, 3
                    ),
                    "peak_reserved_gib": round(
                        torch.cuda.max_memory_reserved() / 1024**3, 3
                    ),
                },
                "results": {
                    "path": str(result_path.resolve()),
                    "rows": len(results),
                    "sha256": sha256_file(result_path),
                },
            }
        )
        _write_json(output_dir / "summary.json", summary, exclusive=True)
        return summary
    except Exception as exc:
        failure = {
            **launch,
            "status": "failed",
            "failed_at_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_cases": len(results),
        }
        _write_json(output_dir / "failure.json", failure, exclusive=True)
        raise
    finally:
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _write_new_json(path: Path, value: Any) -> None:
    if path.exists():
        raise ProbeError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, value, exclusive=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="verify the release and create the fixed sample manifest"
    )
    prepare.add_argument("--release", required=True, type=Path)
    prepare.add_argument("--release-manifest-sha256", required=True)
    prepare.add_argument("--training-config", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)

    run = subparsers.add_parser(
        "run", help="run the fixed probe with Transformers and optional PEFT"
    )
    run.add_argument("--base-model", required=True, type=Path)
    run.add_argument("--adapter", type=Path)
    run.add_argument("--sample-manifest", required=True, type=Path)
    run.add_argument("--sample-manifest-sha256", required=True)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--model-label", required=True)
    run.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    run.add_argument("--tail-tokens", type=int, default=DEFAULT_TAIL_TOKENS)
    run.add_argument(
        "--probe-mode", choices=("formal", "quick"), default="formal"
    )
    run.add_argument(
        "--sample-temperature", type=float, default=DEFAULT_SAMPLE_TEMPERATURE
    )
    run.add_argument("--sample-top-p", type=float, default=DEFAULT_SAMPLE_TOP_P)
    run.add_argument(
        "--load-in-4bit", action=argparse.BooleanOptionalAction, default=True
    )
    run.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
        default="sdpa",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        if args.command == "prepare":
            # The tokenizer path itself is release-bound; no caller-provided
            # tokenizer override can silently change length selection.
            root, _, manifest, observed_release_sha = _verify_release_and_manifest(
                args.release, args.release_manifest_sha256
            )
            tokenizer_path, _, _ = _tokenizer_binding(manifest)
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path),
                local_files_only=True,
                trust_remote_code=False,
            )

            def reuse_verified_release(
                candidate_root: Path, *, expected_manifest_sha256: str
            ) -> Mapping[str, Any]:
                if (
                    candidate_root.resolve() != root
                    or expected_manifest_sha256 != observed_release_sha
                ):
                    raise ProbeError("internal verified-release binding drift")
                return manifest

            artifact = build_sample_manifest(
                release_root=args.release,
                release_manifest_sha256=args.release_manifest_sha256,
                tokenizer=tokenizer,
                training_config=args.training_config,
                seed=args.seed,
                release_verifier=reuse_verified_release,
            )
            _write_new_json(args.output, artifact)
            print(
                canonical_json(
                    {
                        "status": "prepared",
                        "path": str(args.output.resolve()),
                        "sha256": sha256_file(args.output),
                        "samples": len(artifact["samples"]),
                    }
                )
            )
            return 0
        summary = run_generation_probe(
            base_model=args.base_model,
            adapter=args.adapter,
            sample_manifest_path=args.sample_manifest,
            sample_manifest_sha256=args.sample_manifest_sha256,
            output_dir=args.output_dir,
            model_label=args.model_label,
            max_new_tokens=args.max_new_tokens,
            tail_tokens=args.tail_tokens,
            probe_mode=args.probe_mode,
            sample_temperature=args.sample_temperature,
            sample_top_p=args.sample_top_p,
            load_in_4bit=args.load_in_4bit,
            attn_implementation=args.attn_implementation,
        )
        print(canonical_json(summary))
        return 0 if summary["quality_status"] == "passed" else 2
    except (ProbeError, OSError) as exc:
        print(canonical_json({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
