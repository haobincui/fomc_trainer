"""Hash-bound, target-stratified chk4 Decision-GRPO pilot.

``prepare`` selects exactly one deterministic lower-median-length training
sample for each of ``hold``, ``hike``, and ``cut`` from the sealed
``decision_grpo/train.jsonl`` population.  The immutable manifest binds the
release, GRPO configuration, tokenizer, source files, targets, token counts,
and generation seeds without copying prompt text.

``run`` re-opens and verifies every bound artifact, then loads either one
merged local model or one local base model plus a PEFT adapter.  It generates
one greedy plus four sampled (temperature 0.7, top-p 0.9) completions per
target direction and replays ``decision_dense_v3``.  A completion that did
not terminate with EOS, or that reached the 1024-token cap, is fail-closed to
zero even when its visible prefix is parseable.  Results and summaries are
create-only and made read-only after they are finalized.

The full-release verifier may audit every sealed file, but selection, manifest,
generation, replay, and summary use only the train split: validation/test
targets are never selected or emitted.  This utility never starts training.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs.retrain_v2 import probe_chk4_decision_sft_generation as base_probe
from open_r1.provenance import fingerprint_artifact_path
from open_r1.trainer.dataset_release import (
    CHK4_PRE2009_AUGMENTED_RELEASE_ID,
    CHK4_PRE2009_AUGMENTED_RELEASE_SCHEMA,
    CHK4_PRE2009_GRPO_ROLE,
    verify_chk4_pre2009_augmented_release,
)
from open_r1.trainer.rewards.reward_funcs.decision_reward_v3 import (
    FENCED_JSON,
    STRICT_JSON,
    decision_dense_reward_v3,
    parse_decision_response_v3,
)


SAMPLE_SCHEMA_VERSION = "chk4-decision-grpo-stratified-samples-v1"
PROBE_SCHEMA_VERSION = "chk4-decision-grpo-stratified-probe-v1"
SUMMARY_SCHEMA_VERSION = "chk4-decision-grpo-stratified-summary-v1"
CORE_GRPO_ROLE = "decision_grpo"
CORE_RELEASE_SCHEMA = base_probe.RELEASE_SCHEMA_VERSION
SUPPORTED_GRPO_ROLES = frozenset({CORE_GRPO_ROLE, CHK4_PRE2009_GRPO_ROLE})
ROLE_BY_RELEASE_SCHEMA = {
    CORE_RELEASE_SCHEMA: CORE_GRPO_ROLE,
    CHK4_PRE2009_AUGMENTED_RELEASE_SCHEMA: CHK4_PRE2009_GRPO_ROLE,
}
TRAIN_RELATIVE = "decision_grpo/train.jsonl"
UNIQUE_TRAIN_RELATIVE = "manifests/unique/train.jsonl"
DIRECTIONS = ("hold", "hike", "cut")
ALLOWED_DIRECTIONS = frozenset(DIRECTIONS)
ALLOWED_ACTION_MAGNITUDES = frozenset({25, 50, 75, 100})
DEFAULT_SEED = 20260810
MAX_NEW_TOKENS = 1024
TAIL_TOKENS = 256
SAMPLED_PER_DIRECTION = 4
SAMPLE_TEMPERATURE = 0.7
SAMPLE_TOP_P = 0.9
_LABEL_RE = re.compile(r"[A-Za-z0-9_.-]+")
MERGED_MODEL_MODE = "merged_model"
PEFT_ADAPTER_MODE = "base_model_plus_adapter"
PEFT_COMPOSITION_ALGORITHM = (
    "sha256(canonical JSON with base_model_directory_sha256 and "
    "adapter_directory_sha256)"
)
_PEFT_COMPATIBILITY_DEFAULTS = {
    "corda_config": None,
    "eva_config": None,
    "exclude_modules": None,
    "lora_bias": False,
    "trainable_token_indices": None,
}


# Reuse the base probe's error type because its hash/token/boundary helpers can
# fail directly.  A single public exception keeps all fail-closed paths equally
# catchable by the CLI and tests.
PilotError = base_probe.ProbeError


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _valid_target(direction: object, magnitude: object) -> bool:
    return bool(
        direction in ALLOWED_DIRECTIONS
        and _is_int(magnitude)
        and (
            (direction == "hold" and magnitude == 0)
            or (direction in {"cut", "hike"} and magnitude in ALLOWED_ACTION_MAGNITUDES)
        )
    )


def _resolve_declared(config_path: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PilotError(f"training config {label} must be a path")
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    config_candidate = (config_path.parent / candidate).resolve()
    if cwd_candidate.exists() or not config_candidate.exists():
        return cwd_candidate
    return config_candidate


def _read_training_config(config_path: Path) -> tuple[Path, Mapping[str, Any]]:
    path = config_path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise PilotError(f"training config is not a regular file: {path}")
    try:
        import yaml

        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PilotError(f"cannot load training config {path}: {exc}") from exc
    if not isinstance(config, Mapping):
        raise PilotError("training config root must be an object")
    return path, config


def _configured_grpo_role(config: Mapping[str, Any]) -> str:
    role = config.get("dataset_chk4_role")
    if role not in SUPPORTED_GRPO_ROLES:
        raise PilotError(
            "training config dataset_chk4_role must be decision_grpo or "
            f"{CHK4_PRE2009_GRPO_ROLE}"
        )
    return str(role)


def _load_grpo_contract(
    config_path: Path,
    *,
    release_root: Path,
    release_manifest_path: Path,
    release_manifest_sha256: str,
    release_schema: str,
) -> tuple[str, str, Mapping[str, Any]]:
    path, config = _read_training_config(config_path)
    expected_role = ROLE_BY_RELEASE_SCHEMA.get(release_schema)
    if expected_role is None:
        raise PilotError(f"unsupported chk4 release schema: {release_schema!r}")

    exact = {
        "dataset_chk4_role": expected_role,
        "dataset_prompt_column": "prompt",
        "dataset_train_split": "train",
        "reward_funcs": ["decision_dense_v3"],
        "max_completion_length": MAX_NEW_TOKENS,
        "num_generations": SAMPLED_PER_DIRECTION,
        "temperature": SAMPLE_TEMPERATURE,
        "top_p": SAMPLE_TOP_P,
        "use_vllm": False,
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise PilotError(
                f"training config {key} must be {expected!r}, got {config.get(key)!r}"
            )
    if config.get("dataset_chk4_release_manifest_sha256") != release_manifest_sha256:
        raise PilotError("training config/release manifest SHA-256 binding mismatch")
    declared_manifest = _resolve_declared(
        path,
        config.get("dataset_chk4_release_manifest"),
        label="dataset_chk4_release_manifest",
    )
    if declared_manifest != release_manifest_path:
        raise PilotError("training config points to a different release manifest")
    declared_dataset = _resolve_declared(
        path, config.get("dataset_name"), label="dataset_name"
    )
    if declared_dataset != (release_root / CORE_GRPO_ROLE).resolve():
        raise PilotError("training config points to a different Decision-GRPO dataset")
    max_prompt_length = config.get("max_prompt_length")
    if not _is_int(max_prompt_length) or int(max_prompt_length) <= 0:
        raise PilotError("training config max_prompt_length must be positive")
    system_prompt = config.get("system_prompt")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise PilotError("training config system_prompt must be non-empty")
    return system_prompt, base_probe.sha256_file(path), config


def _pre2009_fallback_tokenizer_model(manifest: Mapping[str, Any]) -> Path:
    """Resolve the core parent's audited tokenizer source for prepare-time replay."""

    parents = manifest.get("parent_releases")
    core = parents.get("core_v3") if isinstance(parents, Mapping) else None
    if not isinstance(core, Mapping):
        raise PilotError("pre-2009 release has no core-v3 parent binding")
    core_value = core.get("path")
    if not isinstance(core_value, str) or not core_value:
        raise PilotError("pre-2009 core-v3 parent path is invalid")
    core_root = Path(core_value).expanduser()
    if not core_root.is_absolute():
        core_root = Path(__file__).resolve().parents[2] / core_root
    core_root = core_root.resolve()
    core_manifest_path = core_root / "release_manifest.json"
    base_probe._require_file_hash(
        core_manifest_path,
        core.get("manifest_sha256"),
        label="pre-2009 core-v3 parent manifest",
    )
    core_manifest = base_probe._read_json(
        core_manifest_path, label="pre-2009 core-v3 parent manifest"
    )
    sources = core_manifest.get("sources")
    tokenizer = sources.get("tokenizer") if isinstance(sources, Mapping) else None
    tokenizer_path = tokenizer.get("path") if isinstance(tokenizer, Mapping) else None
    if not isinstance(tokenizer_path, str) or not tokenizer_path:
        raise PilotError("pre-2009 core-v3 parent has no tokenizer source path")
    result = Path(tokenizer_path).expanduser().resolve()
    if not result.is_dir() or result.is_symlink():
        raise PilotError(f"pre-2009 tokenizer model is invalid: {result}")
    return result


def _validate_pre2009_runtime_binding(
    binding: object,
    *,
    release_root: Path,
    release_manifest_path: Path,
    release_manifest_sha256: str,
) -> Mapping[str, Any]:
    if not isinstance(binding, Mapping):
        raise PilotError("pre-2009 release verifier returned a non-object")
    expected = {
        "schema_version": "chk4-pre2009-augmented-runtime-binding-v1",
        "release_id": CHK4_PRE2009_AUGMENTED_RELEASE_ID,
        "dataset_role": CHK4_PRE2009_GRPO_ROLE,
        "physical_dataset_role": CORE_GRPO_ROLE,
        "release_manifest_path": str(release_manifest_path),
        "release_manifest_sha256": release_manifest_sha256,
        "test_verified_but_not_loaded": True,
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise PilotError(
                f"pre-2009 runtime binding {key} must be {value!r}, "
                f"got {binding.get(key)!r}"
            )
    split_files = binding.get("split_files")
    if not isinstance(split_files, Mapping):
        raise PilotError("pre-2009 runtime binding has no split files")
    expected_splits = {
        "train": (release_root / TRAIN_RELATIVE).resolve(),
        "validation": (release_root / "decision_grpo/validation.jsonl").resolve(),
    }
    for split, expected_path in expected_splits.items():
        value = split_files.get(split)
        if not isinstance(value, (str, Path)) or Path(value).resolve() != expected_path:
            raise PilotError(f"pre-2009 runtime binding {split} split path drift")
    if not isinstance(binding.get("tokenizer_binding"), Mapping):
        raise PilotError("pre-2009 runtime binding has no tokenizer binding")
    return binding


def _verify_pre2009_release_and_manifest(
    release_root: Path,
    expected_manifest_sha256: str,
    *,
    training_config: Path,
    verification_model_path: Path | None,
    release_verifier: Callable[..., Mapping[str, Any]] | None,
) -> tuple[Path, Path, Mapping[str, Any], str, Mapping[str, Any]]:
    expected_sha = base_probe._validate_sha256(
        expected_manifest_sha256, label="release manifest SHA-256"
    )
    raw_root = release_root.expanduser()
    if raw_root.is_symlink():
        raise PilotError(f"release root must not be a symlink: {raw_root}")
    root = raw_root.resolve()
    if not root.is_dir():
        raise PilotError(f"release root is not a directory: {root}")
    manifest_path = root / "release_manifest.json"
    observed_sha = base_probe._require_file_hash(
        manifest_path, expected_sha, label="release manifest"
    )
    manifest = base_probe._read_json(manifest_path, label="release manifest")
    if manifest.get("schema_version") != CHK4_PRE2009_AUGMENTED_RELEASE_SCHEMA:
        raise PilotError("release/config role mismatch: expected pre-2009 schema")
    if (
        manifest.get("release_id") != CHK4_PRE2009_AUGMENTED_RELEASE_ID
        or root.name != CHK4_PRE2009_AUGMENTED_RELEASE_ID
        or manifest.get("release_type") != "train_only_augmentation"
        or manifest.get("immutable") is not True
        or manifest.get("quality_status") != "passed"
        or manifest.get("training_ready") is not True
    ):
        raise PilotError("pre-2009 release identity or readiness drift")
    system_prompt, _, config = _load_grpo_contract(
        training_config,
        release_root=root,
        release_manifest_path=manifest_path,
        release_manifest_sha256=observed_sha,
        release_schema=CHK4_PRE2009_AUGMENTED_RELEASE_SCHEMA,
    )
    dataset_dir = _resolve_declared(
        training_config.expanduser().resolve(),
        config.get("dataset_name"),
        label="dataset_name",
    )
    model_for_verification = verification_model_path
    if model_for_verification is None:
        configured_model = config.get("model_name_or_path")
        if isinstance(configured_model, str) and configured_model.strip():
            candidate = _resolve_declared(
                training_config.expanduser().resolve(),
                configured_model,
                label="model_name_or_path",
            )
            if candidate.is_dir() and not candidate.is_symlink():
                model_for_verification = candidate
    if model_for_verification is None:
        model_for_verification = _pre2009_fallback_tokenizer_model(manifest)
    verifier = release_verifier or verify_chk4_pre2009_augmented_release
    try:
        runtime_binding = verifier(
            dataset_dir=dataset_dir,
            manifest_path=manifest_path,
            expected_manifest_sha256=observed_sha,
            dataset_role=CHK4_PRE2009_GRPO_ROLE,
            system_prompt=system_prompt,
            model_path=model_for_verification.expanduser().resolve(),
        )
    except PilotError:
        raise
    except Exception as exc:
        raise PilotError(f"sealed pre-2009 release verification failed: {exc}") from exc
    normalized_binding = _validate_pre2009_runtime_binding(
        runtime_binding,
        release_root=root,
        release_manifest_path=manifest_path,
        release_manifest_sha256=observed_sha,
    )
    return root, manifest_path, manifest, observed_sha, normalized_binding


def _verify_probe_release(
    release_root: Path,
    expected_manifest_sha256: str,
    *,
    training_config: Path,
    verification_model_path: Path | None = None,
    release_verifier: Callable[..., Mapping[str, Any]] | None = None,
) -> tuple[Path, Path, Mapping[str, Any], str, Mapping[str, Any] | None]:
    _, config = _read_training_config(training_config)
    role = _configured_grpo_role(config)
    if role == CORE_GRPO_ROLE:
        root, manifest_path, manifest, manifest_sha = (
            base_probe._verify_release_and_manifest(
                release_root,
                expected_manifest_sha256,
                release_verifier=release_verifier,
            )
        )
        return root, manifest_path, manifest, manifest_sha, None
    return _verify_pre2009_release_and_manifest(
        release_root,
        expected_manifest_sha256,
        training_config=training_config,
        verification_model_path=verification_model_path,
        release_verifier=release_verifier,
    )


def _tokenizer_binding_for_release(
    manifest: Mapping[str, Any],
    *,
    runtime_binding: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Mapping[str, Any]], str]:
    if manifest.get("schema_version") == CORE_RELEASE_SCHEMA:
        candidate = base_probe._tokenizer_binding(manifest)
    elif manifest.get("schema_version") == CHK4_PRE2009_AUGMENTED_RELEASE_SCHEMA:
        if runtime_binding is None:
            raise PilotError(
                "pre-2009 release is missing its runtime tokenizer binding"
            )
        tokenizer = runtime_binding.get("tokenizer_binding")
        if not isinstance(tokenizer, Mapping):
            raise PilotError("pre-2009 runtime tokenizer binding is invalid")
        synthetic = {
            "sources": {
                "tokenizer": {
                    "path": tokenizer.get("model_path"),
                    "files": tokenizer.get("files"),
                }
            }
        }
        candidate = base_probe._tokenizer_binding(synthetic)
        if tokenizer.get("bundle_sha256") != candidate[2]:
            raise PilotError("pre-2009 runtime tokenizer bundle SHA-256 drift")
    else:
        raise PilotError("unsupported chk4 release schema")

    if override is None:
        return candidate
    if set(override) != {"path", "files", "bundle_sha256"}:
        raise PilotError("sample tokenizer binding schema drift")
    synthetic_override = {
        "sources": {
            "tokenizer": {
                "path": override.get("path"),
                "files": override.get("files"),
            }
        }
    }
    rebound = base_probe._tokenizer_binding(synthetic_override)
    if override.get("bundle_sha256") != rebound[2]:
        raise PilotError("sample tokenizer bundle SHA-256 drift")
    if (
        base_probe.canonical_json(rebound[1]) != base_probe.canonical_json(candidate[1])
        or rebound[2] != candidate[2]
    ):
        raise PilotError("sample/release tokenizer bytes disagree")
    return rebound


def _sample_tokenizer_binding(
    sample_manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Mapping[str, Any]], str]:
    tokenizer = sample_manifest.get("tokenizer")
    if not isinstance(tokenizer, Mapping) or set(tokenizer) != {
        "path",
        "files",
        "bundle_sha256",
    }:
        raise PilotError("sample tokenizer binding schema drift")
    synthetic = {
        "sources": {
            "tokenizer": {
                "path": tokenizer.get("path"),
                "files": tokenizer.get("files"),
            }
        }
    }
    binding = base_probe._tokenizer_binding(synthetic)
    if tokenizer.get("bundle_sha256") != binding[2]:
        raise PilotError("sample tokenizer bundle SHA-256 drift")
    return binding


def _reuse_verified_release(
    *,
    root: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    runtime_binding: Mapping[str, Any] | None,
) -> Callable[..., Mapping[str, Any]]:
    def verifier(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        if runtime_binding is None:
            candidate_root = args[0] if args else None
            if (
                not isinstance(candidate_root, Path)
                or candidate_root.resolve() != root
                or kwargs.get("expected_manifest_sha256") != manifest_sha256
            ):
                raise PilotError("internal verified core-release binding drift")
            return manifest
        expected = {
            "dataset_dir": (root / CORE_GRPO_ROLE).resolve(),
            "manifest_path": manifest_path,
            "expected_manifest_sha256": manifest_sha256,
            "dataset_role": CHK4_PRE2009_GRPO_ROLE,
        }
        for key, value in expected.items():
            candidate = kwargs.get(key)
            if key in {"dataset_dir", "manifest_path"}:
                if (
                    not isinstance(candidate, (str, Path))
                    or Path(candidate).resolve() != value
                ):
                    raise PilotError("internal verified pre-2009 release binding drift")
            elif candidate != value:
                raise PilotError("internal verified pre-2009 release binding drift")
        return runtime_binding

    return verifier


def _load_train_rows(
    release_root: Path, manifest: Mapping[str, Any]
) -> tuple[
    Path,
    Mapping[str, Any],
    list[Mapping[str, Any]],
    Path,
    Mapping[str, Any],
    list[Mapping[str, Any]],
]:
    train_path, train_record = base_probe._release_file(
        release_root, manifest, TRAIN_RELATIVE
    )
    unique_path, unique_record = base_probe._release_file(
        release_root, manifest, UNIQUE_TRAIN_RELATIVE
    )
    rows = base_probe._read_jsonl(train_path, label="Decision-GRPO train")
    unique_rows = base_probe._read_jsonl(
        unique_path, label="unique Decision-GRPO train manifest"
    )
    if len(rows) < 3 or len(unique_rows) < 3:
        raise PilotError("Decision-GRPO train populations are unexpectedly small")
    for record, observed, label in (
        (train_record, len(rows), TRAIN_RELATIVE),
        (unique_record, len(unique_rows), UNIQUE_TRAIN_RELATIVE),
    ):
        if record.get("rows") != observed:
            raise PilotError(f"release row-count binding mismatch: {label}")
    return train_path, train_record, rows, unique_path, unique_record, unique_rows


def _validated_candidates(
    *,
    rows: Sequence[Mapping[str, Any]],
    unique_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    system_prompt: str,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    unique_by_id: dict[str, Mapping[str, Any]] = {}
    for line_number, unique in enumerate(unique_rows, start=1):
        sample_id = unique.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise PilotError(f"missing unique train sample_id at line {line_number}")
        if sample_id in unique_by_id:
            raise PilotError(f"duplicate unique train sample_id: {sample_id}")
        if unique.get("split") != "train":
            raise PilotError(f"non-train target in unique train manifest: {sample_id}")
        if not _valid_target(unique.get("direction"), unique.get("magnitude_bp")):
            raise PilotError(f"invalid unique train target: {sample_id}")
        prompt_sha = unique.get("prompt_sha256")
        base_probe._validate_sha256(
            prompt_sha, label=f"unique train prompt {sample_id} SHA-256"
        )
        repeat_factor = unique.get("repeat_factor")
        if not _is_int(repeat_factor) or int(repeat_factor) <= 0:
            raise PilotError(f"invalid repeat_factor for {sample_id}")
        unique_by_id[sample_id] = unique

    physical: dict[str, dict[str, Any]] = {}
    expected_keys = {"direction", "magnitude_bp", "prompt", "sample_id"}
    for line_number, row in enumerate(rows, start=1):
        if set(row) != expected_keys:
            raise PilotError(
                f"unexpected Decision-GRPO train schema at line {line_number}"
            )
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in unique_by_id:
            raise PilotError(
                f"physical train sample is absent from unique manifest: {sample_id}"
            )
        unique = unique_by_id[sample_id]
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise PilotError(f"empty physical train prompt: {sample_id}")
        if base_probe.sha256_text(prompt) != unique.get("prompt_sha256"):
            raise PilotError(f"physical/unique prompt hash drift: {sample_id}")
        if row.get("direction") != unique.get("direction") or row.get(
            "magnitude_bp"
        ) != unique.get("magnitude_bp"):
            raise PilotError(f"physical/unique target drift: {sample_id}")
        current = physical.setdefault(sample_id, {"prompt": prompt, "line_numbers": []})
        if current["prompt"] != prompt:
            raise PilotError(f"physical prompt variants found for {sample_id}")
        current["line_numbers"].append(line_number)

    if set(physical) != set(unique_by_id):
        missing = sorted(set(unique_by_id) - set(physical))
        raise PilotError(f"unique train samples missing physically: {missing[:3]}")
    candidates: list[dict[str, Any]] = []
    for sample_id, unique in unique_by_id.items():
        bound = physical[sample_id]
        line_numbers = list(bound["line_numbers"])
        if len(line_numbers) != unique.get("repeat_factor"):
            raise PilotError(f"repeat_factor drift for {sample_id}")
        prompt_ids = base_probe._prompt_ids(
            tokenizer, base_probe._messages(system_prompt, bound["prompt"])
        )
        candidates.append(
            {
                "sample_id": sample_id,
                "split": "train",
                "direction": unique["direction"],
                "magnitude_bp": unique["magnitude_bp"],
                "physical_line_numbers": line_numbers,
                "prompt_sha256": unique["prompt_sha256"],
                "prompt_token_count": len(prompt_ids),
            }
        )
    counts = Counter(str(item["direction"]) for item in candidates)
    if set(counts) != set(DIRECTIONS) or any(counts[item] < 1 for item in DIRECTIONS):
        raise PilotError(f"train target population is incomplete: {dict(counts)}")
    return candidates, physical


def build_stratified_manifest(
    *,
    release_root: Path,
    release_manifest_sha256: str,
    tokenizer: Any,
    training_config: Path,
    seed: int = DEFAULT_SEED,
    release_verifier: Callable[..., Mapping[str, Any]] | None = None,
    verification_model_path: Path | None = None,
    tokenizer_binding_override: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Build the text-free, train-only three-direction sample manifest."""

    if not _is_int(seed) or seed < 0:
        raise PilotError("seed must be a non-negative integer")
    root, manifest_path, manifest, manifest_sha, runtime_binding = (
        _verify_probe_release(
            release_root,
            release_manifest_sha256,
            training_config=training_config,
            verification_model_path=verification_model_path,
            release_verifier=release_verifier,
        )
    )
    system_prompt, config_sha, config = _load_grpo_contract(
        training_config,
        release_root=root,
        release_manifest_path=manifest_path,
        release_manifest_sha256=manifest_sha,
        release_schema=str(manifest.get("schema_version")),
    )
    tokenizer_path, tokenizer_files, tokenizer_bundle_sha = (
        _tokenizer_binding_for_release(
            manifest,
            runtime_binding=runtime_binding,
            override=tokenizer_binding_override,
        )
    )
    train_path, train_record, rows, unique_path, unique_record, unique_rows = (
        _load_train_rows(root, manifest)
    )
    candidates, _ = _validated_candidates(
        rows=rows,
        unique_rows=unique_rows,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )
    over_limit = [
        (str(item["sample_id"]), int(item["prompt_token_count"]))
        for item in candidates
        if int(item["prompt_token_count"]) > int(config["max_prompt_length"])
    ]
    if over_limit:
        raise PilotError(
            f"sealed train prompt exceeds GRPO max_prompt_length: {over_limit[:3]}"
        )

    selected: list[dict[str, Any]] = []
    for direction_index, direction in enumerate(DIRECTIONS):
        ordered = sorted(
            (item for item in candidates if item["direction"] == direction),
            key=lambda item: (int(item["prompt_token_count"]), str(item["sample_id"])),
        )
        chosen = ordered[(len(ordered) - 1) // 2]
        group_seed = seed + direction_index * (SAMPLED_PER_DIRECTION + 1)
        selected.append(
            {
                **chosen,
                "direction_population_unique_rows": len(ordered),
                "direction_median_rank_zero_based": (len(ordered) - 1) // 2,
                "greedy_seed": group_seed,
                "sample_seeds": [
                    group_seed + offset
                    for offset in range(1, SAMPLED_PER_DIRECTION + 1)
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
            "manifest_sha256": manifest_sha,
            "schema_version": manifest.get("schema_version"),
        },
        "dataset": {
            "role": config["dataset_chk4_role"],
            "split": "train",
            "path": str(train_path),
            "sha256": train_record["sha256"],
            "rows": len(rows),
            "unique_manifest_path": str(unique_path),
            "unique_manifest_sha256": unique_record["sha256"],
            "unique_rows": len(unique_rows),
        },
        "prompt_contract": {
            "training_config": str(config_path),
            "training_config_sha256": config_sha,
            "system_prompt_sha256": base_probe.sha256_text(system_prompt),
            "max_prompt_length": int(config["max_prompt_length"]),
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "files": tokenizer_files,
            "bundle_sha256": tokenizer_bundle_sha,
        },
        "selection": {
            "algorithm": "per-direction-rendered-prompt-token-lower-median-v1",
            "direction_order": list(DIRECTIONS),
            "base_seed": seed,
            "source_split": "train",
            "selected_rows": len(selected),
            "generation_contract": {
                "greedy_per_direction": 1,
                "sampled_per_direction": SAMPLED_PER_DIRECTION,
                "sample_temperature": SAMPLE_TEMPERATURE,
                "sample_top_p": SAMPLE_TOP_P,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
        },
        "samples": selected,
    }


def _write_exclusive_readonly_json(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise PilotError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
    except FileExistsError as exc:
        raise PilotError(f"refusing to overwrite artifact: {path}") from exc


def _validate_sample_manifest(
    path: Path, expected_sha256: str
) -> tuple[Mapping[str, Any], str]:
    observed = base_probe._require_file_hash(
        path, expected_sha256, label="stratified sample manifest"
    )
    manifest = base_probe._read_json(path, label="stratified sample manifest")
    if manifest.get("schema_version") != SAMPLE_SCHEMA_VERSION:
        raise PilotError("unsupported stratified sample manifest schema")
    dataset = manifest.get("dataset")
    release = manifest.get("release")
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("split") != "train"
        or dataset.get("role") not in SUPPORTED_GRPO_ROLES
    ):
        raise PilotError("sample manifest is not exclusively train-bound")
    if not isinstance(release, Mapping):
        raise PilotError("sample manifest has no release binding")
    expected_role = ROLE_BY_RELEASE_SCHEMA.get(release.get("schema_version"))
    if expected_role is None or dataset.get("role") != expected_role:
        raise PilotError("sample manifest release schema/logical role drift")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != len(DIRECTIONS):
        raise PilotError("sample manifest must contain exactly three directions")
    if [item.get("direction") for item in samples if isinstance(item, Mapping)] != list(
        DIRECTIONS
    ):
        raise PilotError("sample manifest direction ordering or membership drift")
    all_seeds: list[int] = []
    for sample in samples:
        if not isinstance(sample, Mapping) or sample.get("split") != "train":
            raise PilotError("sample manifest contains a non-train row")
        if not _valid_target(sample.get("direction"), sample.get("magnitude_bp")):
            raise PilotError("sample manifest contains an invalid target")
        sample_seeds = sample.get("sample_seeds")
        greedy_seed = sample.get("greedy_seed")
        if (
            not _is_int(greedy_seed)
            or greedy_seed < 0
            or not isinstance(sample_seeds, list)
            or len(sample_seeds) != SAMPLED_PER_DIRECTION
            or any(not _is_int(value) or value < 0 for value in sample_seeds)
        ):
            raise PilotError("sample manifest contains invalid generation seeds")
        all_seeds.extend([int(greedy_seed), *[int(value) for value in sample_seeds]])
    if len(all_seeds) != 15 or len(set(all_seeds)) != 15:
        raise PilotError("sample manifest must bind 15 unique generation seeds")
    expected_contract = {
        "greedy_per_direction": 1,
        "sampled_per_direction": SAMPLED_PER_DIRECTION,
        "sample_temperature": SAMPLE_TEMPERATURE,
        "sample_top_p": SAMPLE_TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    selection = manifest.get("selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("generation_contract") != expected_contract
    ):
        raise PilotError("sample manifest generation contract drift")
    return manifest, observed


def _generation_cases(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cases: list[Mapping[str, Any]] = []
    for sample in manifest["samples"]:
        cases.append(
            {
                "sample": sample,
                "generation_mode": "greedy",
                "generation_index": 0,
                "seed": int(sample["greedy_seed"]),
            }
        )
        for index, seed in enumerate(sample["sample_seeds"], start=1):
            cases.append(
                {
                    "sample": sample,
                    "generation_mode": "sampled",
                    "generation_index": index,
                    "seed": int(seed),
                }
            )
    return cases


def replay_decision_dense_v3(
    completion: str,
    target: Mapping[str, Any],
    *,
    hit_eos: bool,
    cap_reached: bool,
) -> Mapping[str, Any]:
    """Replay v3 and fail closed on an unterminated/truncated completion."""

    parsed = parse_decision_response_v3(completion)
    prediction = parsed.prediction
    if not hit_eos or cap_reached:
        reward = 0.0
        forced_zero_reason = "truncated_or_unterminated"
    else:
        values = decision_dense_reward_v3(
            [[{"role": "assistant", "content": completion}]],
            direction=[str(target.get("direction"))],
            magnitude_bp=[int(target.get("magnitude_bp"))],
        )
        reward = float(values[0])
        forced_zero_reason = None
    direction_correct = bool(
        prediction is not None and prediction[0] == target.get("direction")
    )
    exact = bool(
        prediction is not None
        and prediction[0] == target.get("direction")
        and prediction[1] == target.get("magnitude_bp")
    )
    return {
        "reward_name": "decision_dense_v3",
        "reward": round(reward, 8),
        "nonzero": reward > 1e-12,
        "response_format": parsed.response_format,
        "strict_json": parsed.response_format == STRICT_JSON,
        "fenced_json": parsed.response_format == FENCED_JSON,
        "prediction": (
            {"direction": prediction[0], "magnitude_bp": prediction[1]}
            if prediction is not None
            else None
        ),
        "direction_correct": direction_correct,
        "exact": exact,
        "rejection_reason": parsed.rejection_reason,
        "forced_zero_reason": forced_zero_reason,
    }


def _rate(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return round(sum(bool(row.get(field)) for row in rows) / len(rows), 8)


def _metric_block(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise PilotError("cannot summarize an empty metric block")
    rewards = [float(row["decision_dense_v3_reward"]) for row in rows]
    return {
        "cases": len(rows),
        "rewards": rewards,
        "reward_mean": round(statistics.fmean(rewards), 8),
        "reward_std": round(statistics.pstdev(rewards), 8),
        "nonzero_count": sum(value > 1e-12 for value in rewards),
        "nonzero_rate": round(sum(value > 1e-12 for value in rewards) / len(rows), 8),
        "zero_std": statistics.pstdev(rewards) <= 1e-12,
        "direction_correct_count": sum(
            bool(row.get("decision_direction_correct")) for row in rows
        ),
        "direction_correct_rate": _rate(rows, "decision_direction_correct"),
        "exact_count": sum(bool(row.get("decision_exact")) for row in rows),
        "exact_rate": _rate(rows, "decision_exact"),
        "cap_count": sum(bool(row.get("cap_reached")) for row in rows),
        "cap_rate": _rate(rows, "cap_reached"),
        "boundary_count": sum(row.get("think_boundary_count") == 1 for row in rows),
        "boundary_rate": round(
            sum(row.get("think_boundary_count") == 1 for row in rows) / len(rows), 8
        ),
        "fence_count": sum(bool(row.get("fenced_json")) for row in rows),
        "fence_rate": _rate(rows, "fenced_json"),
        "bare_count": sum(bool(row.get("strict_json")) for row in rows),
        "bare_rate": _rate(rows, "strict_json"),
        "eos_count": sum(bool(row.get("hit_eos")) for row in rows),
        "eos_rate": _rate(rows, "hit_eos"),
    }


def summarize_stratified_results(
    results: Sequence[Mapping[str, Any]], *, provenance: Mapping[str, Any]
) -> Mapping[str, Any]:
    if len(results) != len(DIRECTIONS) * (SAMPLED_PER_DIRECTION + 1):
        raise PilotError("pilot must contain exactly 15 generation cases")
    seen: set[tuple[str, str, int]] = set()
    by_direction: dict[str, list[Mapping[str, Any]]] = {item: [] for item in DIRECTIONS}
    for row in results:
        if row.get("split") != "train":
            raise PilotError("pilot results contain a non-train row")
        direction = str(row.get("target_direction"))
        if direction not in by_direction:
            raise PilotError(f"pilot result has invalid target direction: {direction}")
        key = (
            str(row.get("sample_id")),
            str(row.get("generation_mode")),
            int(row.get("generation_index", -1)),
        )
        if key in seen:
            raise PilotError(f"duplicate generation case: {key}")
        seen.add(key)
        if (not row.get("hit_eos") or row.get("cap_reached")) and float(
            row.get("decision_dense_v3_reward", -1)
        ) != 0.0:
            raise PilotError(
                "unterminated or capped completion received nonzero reward"
            )
        by_direction[direction].append(row)

    direction_summary: dict[str, Any] = {}
    gate_reasons: list[str] = []
    for direction in DIRECTIONS:
        rows = by_direction[direction]
        greedy = [row for row in rows if row.get("generation_mode") == "greedy"]
        sampled = sorted(
            (row for row in rows if row.get("generation_mode") == "sampled"),
            key=lambda row: int(row.get("generation_index", -1)),
        )
        if (
            len(rows) != 5
            or len(greedy) != 1
            or len(sampled) != 4
            or [int(row.get("generation_index", -1)) for row in sampled] != [1, 2, 3, 4]
        ):
            raise PilotError(f"incomplete generation group for {direction}")
        all_metrics = _metric_block(rows)
        sampled_metrics = _metric_block(sampled)
        direction_summary[direction] = {
            "sample_id": rows[0]["sample_id"],
            "target_magnitude_bp": rows[0]["target_magnitude_bp"],
            "all": all_metrics,
            "sampled": sampled_metrics,
        }
        if sampled_metrics["nonzero_count"] < 1:
            gate_reasons.append(f"{direction}:sampled_nonzero_count_lt_1")
        if sampled_metrics["direction_correct_count"] < 1:
            gate_reasons.append(f"{direction}:sampled_direction_correct_count_lt_1")
        if sampled_metrics["zero_std"]:
            gate_reasons.append(f"{direction}:sampled_reward_zero_std")

    overall = _metric_block(list(results))
    if overall["cap_rate"] > 0.25:
        gate_reasons.append("overall_cap_rate_gt_0.25")
    if overall["boundary_rate"] < 0.75:
        gate_reasons.append("overall_boundary_rate_lt_0.75")
    periodic_count = sum(bool(row.get("strict_periodic_tail")) for row in results)
    if periodic_count:
        gate_reasons.append("strict_periodic_tail_present")
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "complete",
        "quality_status": "passed" if not gate_reasons else "failed",
        "quality_gate": {
            "contract": {
                "each_direction_sampled_nonzero_min": 1,
                "each_direction_sampled_direction_correct_min": 1,
                "each_direction_sampled_reward_std_gt": 0.0,
                "overall_cap_rate_max": 0.25,
                "overall_boundary_rate_min": 0.75,
                "strict_periodic_tail_max": 0,
            },
            "reasons": gate_reasons,
        },
        "cases": len(results),
        "train_groups": len(DIRECTIONS),
        "greedy_cases": 3,
        "sampled_cases": 12,
        "direction_order": list(DIRECTIONS),
        "directions": direction_summary,
        "overall": {**overall, "strict_periodic_tail_count": periodic_count},
        "provenance": dict(provenance),
    }


def _load_bound_samples(
    manifest: Mapping[str, Any],
    *,
    tokenizer: Any,
    verification_model_path: Path | None = None,
    release_verifier: Callable[..., Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Any], str, str, str]:
    release = manifest.get("release")
    prompt_contract = manifest.get("prompt_contract")
    if not isinstance(release, Mapping) or not isinstance(prompt_contract, Mapping):
        raise PilotError("sample manifest release/prompt binding is missing")
    training_config = Path(str(prompt_contract.get("training_config")))
    root, manifest_path, release_manifest, release_sha, runtime_binding = (
        _verify_probe_release(
            Path(str(release.get("path"))),
            str(release.get("manifest_sha256")),
            training_config=training_config,
            verification_model_path=verification_model_path,
            release_verifier=release_verifier,
        )
    )
    system_prompt, config_sha, _ = _load_grpo_contract(
        training_config,
        release_root=root,
        release_manifest_path=manifest_path,
        release_manifest_sha256=release_sha,
        release_schema=str(release_manifest.get("schema_version")),
    )
    reuse_verifier = _reuse_verified_release(
        root=root,
        manifest_path=manifest_path,
        manifest=release_manifest,
        manifest_sha256=release_sha,
        runtime_binding=runtime_binding,
    )
    rebuilt = build_stratified_manifest(
        release_root=root,
        release_manifest_sha256=release_sha,
        tokenizer=tokenizer,
        training_config=training_config,
        seed=int(manifest["selection"]["base_seed"]),
        release_verifier=reuse_verifier,
        verification_model_path=verification_model_path,
        tokenizer_binding_override=manifest.get("tokenizer"),
    )
    if base_probe.canonical_json(rebuilt) != base_probe.canonical_json(manifest):
        raise PilotError("sample manifest no longer matches its sealed inputs")
    _, _, rows, _, _, unique_rows = _load_train_rows(root, release_manifest)
    _, physical = _validated_candidates(
        rows=rows,
        unique_rows=unique_rows,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )
    bound: dict[str, Mapping[str, Any]] = {}
    for sample in manifest["samples"]:
        sample_id = str(sample["sample_id"])
        source = physical[sample_id]
        bound[sample_id] = {
            "prompt": source["prompt"],
            "target": {
                "direction": sample["direction"],
                "magnitude_bp": sample["magnitude_bp"],
            },
        }
    return bound, release_manifest, release_sha, system_prompt, config_sha


def _resolve_model_loading_mode(
    *,
    model_path: Path | None,
    base_model_path: Path | None,
    adapter_path: Path | None,
) -> tuple[str, Path, Path | None]:
    """Resolve exactly one complete model-loading mode."""

    if model_path is not None:
        if base_model_path is not None or adapter_path is not None:
            raise PilotError(
                "--model is mutually exclusive with --base-model/--adapter"
            )
        return MERGED_MODEL_MODE, model_path.expanduser().resolve(), None
    if base_model_path is None and adapter_path is None:
        raise PilotError("run requires --model or --base-model plus --adapter")
    if base_model_path is None or adapter_path is None:
        raise PilotError("--base-model and --adapter must be provided together")
    expanded_base = base_model_path.expanduser()
    expanded_adapter = adapter_path.expanduser()
    if expanded_base.is_symlink():
        raise PilotError("base model directory must not be a symlink")
    if expanded_adapter.is_symlink():
        raise PilotError("adapter directory must not be a symlink")
    base = expanded_base.resolve()
    adapter = expanded_adapter.resolve()
    if base == adapter:
        raise PilotError("base model and adapter paths must be distinct")
    return PEFT_ADAPTER_MODE, base, adapter


def _require_local_directory(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise PilotError(f"{label} directory must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_dir():
        raise PilotError(f"{label} directory is missing: {resolved}")
    for candidate in resolved.rglob("*"):
        if candidate.is_symlink():
            raise PilotError(f"{label} contains a symlink: {candidate}")
    return resolved


def _read_local_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PilotError(f"{label} is not a regular local file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotError(f"cannot read {label}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PilotError(f"{label} root must be an object")
    return payload


def _file_hash_record(path: Path, *, root: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PilotError(f"{label} is not a regular local file: {path}")
    before = path.stat()
    digest = base_probe.sha256_file(path)
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise PilotError(f"{label} changed while it was being hashed")
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": after.st_size,
        "sha256": digest,
    }


def _directory_fingerprint(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        fingerprint = fingerprint_artifact_path(path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise PilotError(f"cannot fingerprint {label}: {exc}") from exc
    if fingerprint.get("kind") != "directory":
        raise PilotError(f"{label} fingerprint is not a directory")
    return fingerprint


def _fingerprint_base_model(path: Path) -> Mapping[str, Any]:
    root = _require_local_directory(path, label="base model")
    config_path = root / "config.json"
    _read_local_json_object(config_path, label="base model config")

    weight_paths = sorted(
        {
            *(
                candidate
                for candidate in root.glob("model*.safetensors")
                if candidate.is_file()
            ),
            *(
                candidate
                for candidate in root.glob("pytorch_model*.bin")
                if candidate.is_file()
            ),
        }
    )
    if not weight_paths:
        raise PilotError(f"base model has no supported local weight files: {root}")
    index_paths = [
        candidate
        for candidate in (
            root / "model.safetensors.index.json",
            root / "pytorch_model.bin.index.json",
        )
        if candidate.is_file()
    ]
    if len(index_paths) > 1:
        raise PilotError("base model contains ambiguous weight indexes")
    if index_paths:
        index = _read_local_json_object(index_paths[0], label="base weight index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise PilotError("base weight index has no weight_map")
        referenced = {str(value) for value in weight_map.values()}
        if any(
            not name or Path(name).is_absolute() or len(Path(name).parts) != 1
            for name in referenced
        ):
            raise PilotError("base weight index contains an unsafe shard path")
        observed = {candidate.name for candidate in weight_paths}
        if referenced != observed:
            raise PilotError("base weight files do not exactly match the weight index")
    elif len(weight_paths) != 1 or weight_paths[0].name not in {
        "model.safetensors",
        "pytorch_model.bin",
    }:
        raise PilotError("unindexed base model weights are ambiguous")

    critical_paths = [config_path]
    generation_config = root / "generation_config.json"
    if generation_config.is_file():
        _read_local_json_object(
            generation_config, label="base generation configuration"
        )
        critical_paths.append(generation_config)
    critical_paths.extend(index_paths)
    critical_paths.extend(weight_paths)
    file_records = {
        candidate.relative_to(root).as_posix(): _file_hash_record(
            candidate,
            root=root,
            label=f"base model file {candidate.name}",
        )
        for candidate in critical_paths
    }
    return {
        "directory": _directory_fingerprint(root, label="base model directory"),
        "files": file_records,
    }


def _declared_base_candidates(value: str, *, adapter_root: Path) -> set[Path]:
    declared = Path(value).expanduser()
    if declared.is_absolute():
        return {declared.resolve()}
    repo_root = Path(__file__).resolve().parents[2]
    return {
        (Path.cwd() / declared).resolve(),
        (repo_root / declared).resolve(),
        (adapter_root / declared).resolve(),
    }


def _validated_adapter_config(
    path: Path, *, adapter_root: Path, base_model: Path
) -> dict[str, Any]:
    payload = dict(_read_local_json_object(path, label="adapter config"))
    if payload.get("peft_type") != "LORA":
        raise PilotError("adapter config peft_type must be LORA")
    if payload.get("task_type") != "CAUSAL_LM":
        raise PilotError("adapter config task_type must be CAUSAL_LM")
    declared_base = payload.get("base_model_name_or_path")
    if not isinstance(declared_base, str) or not declared_base.strip():
        raise PilotError("adapter config has no base_model_name_or_path")
    if base_model not in _declared_base_candidates(
        declared_base, adapter_root=adapter_root
    ):
        raise PilotError("adapter config is bound to a different base model")
    rank = payload.get("r")
    if not _is_int(rank) or int(rank) <= 0:
        raise PilotError("adapter config has an invalid LoRA rank")
    target_modules = payload.get("target_modules")
    if (
        not isinstance(target_modules, list)
        or not target_modules
        or any(not isinstance(value, str) or not value for value in target_modules)
    ):
        raise PilotError("adapter config has invalid target_modules")
    for key, expected in _PEFT_COMPATIBILITY_DEFAULTS.items():
        if payload.get(key, expected) != expected:
            raise PilotError(
                f"adapter {key} uses unsupported non-default inference semantics"
            )
    return payload


def _fingerprint_adapter(
    path: Path, *, base_model: Path
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    root = _require_local_directory(path, label="adapter")
    config_path = root / "adapter_config.json"
    payload = _validated_adapter_config(
        config_path, adapter_root=root, base_model=base_model
    )
    weight_paths = [
        candidate
        for candidate in (
            root / "adapter_model.safetensors",
            root / "adapter_model.bin",
        )
        if candidate.is_file()
    ]
    if len(weight_paths) != 1:
        raise PilotError("adapter must contain exactly one supported weight file")
    files = {
        candidate.name: _file_hash_record(
            candidate,
            root=root,
            label=f"adapter file {candidate.name}",
        )
        for candidate in (config_path, *weight_paths)
    }
    binding = {
        "directory": _directory_fingerprint(root, label="adapter directory"),
        "files": files,
        "declared_base_model_name_or_path": payload["base_model_name_or_path"],
    }
    return binding, payload


def _prepare_model_source(
    *,
    model_path: Path | None,
    base_model_path: Path | None,
    adapter_path: Path | None,
) -> Mapping[str, Any]:
    mode, load_path, resolved_adapter = _resolve_model_loading_mode(
        model_path=model_path,
        base_model_path=base_model_path,
        adapter_path=adapter_path,
    )
    if mode == MERGED_MODEL_MODE:
        try:
            model_fingerprint = fingerprint_artifact_path(load_path)
        except (FileNotFoundError, ValueError, OSError) as exc:
            raise PilotError(f"cannot fingerprint model artifact: {exc}") from exc
        return {
            "mode": mode,
            "load_path": load_path,
            "adapter_path": None,
            "adapter_config": None,
            "model_fingerprint": model_fingerprint,
            "adapter_fingerprint": None,
            "effective_model_fingerprint": model_fingerprint,
            "provenance": {"mode": mode},
            "result_provenance": {},
        }

    if resolved_adapter is None:  # Defensive: mode resolution must bind both paths.
        raise PilotError("internal PEFT adapter path binding is missing")
    base_binding = _fingerprint_base_model(load_path)
    adapter_binding, adapter_config = _fingerprint_adapter(
        resolved_adapter, base_model=load_path
    )
    composition_payload = {
        "base_model_directory_sha256": base_binding["directory"]["sha256"],
        "adapter_directory_sha256": adapter_binding["directory"]["sha256"],
    }
    composition_sha = base_probe.sha256_text(
        base_probe.canonical_json(composition_payload)
    )
    effective_model_fingerprint = {
        "path": f"{load_path}::peft::{resolved_adapter}",
        "kind": "peft_composition",
        "sha256": composition_sha,
        "file_count": (
            int(base_binding["directory"]["file_count"])
            + int(adapter_binding["directory"]["file_count"])
        ),
        "total_bytes": (
            int(base_binding["directory"]["total_bytes"])
            + int(adapter_binding["directory"]["total_bytes"])
        ),
        "algorithm": PEFT_COMPOSITION_ALGORITHM,
    }
    return {
        "mode": mode,
        "load_path": load_path,
        "adapter_path": resolved_adapter,
        "adapter_config": adapter_config,
        "model_fingerprint": base_binding["directory"],
        "adapter_fingerprint": adapter_binding["directory"],
        "effective_model_fingerprint": effective_model_fingerprint,
        "provenance": {
            "mode": mode,
            "base_model": base_binding,
            "adapter": adapter_binding,
            "composition": {
                "algorithm": PEFT_COMPOSITION_ALGORITHM,
                "payload": composition_payload,
                "sha256": composition_sha,
            },
        },
        "result_provenance": {
            "model_loading_mode": mode,
            "base_model_sha256": base_binding["directory"]["sha256"],
            "adapter_sha256": adapter_binding["directory"]["sha256"],
            "effective_model_sha256": composition_sha,
            "adapter_config_sha256": adapter_binding["files"]["adapter_config.json"][
                "sha256"
            ],
            "adapter_weights_sha256": next(
                record["sha256"]
                for name, record in adapter_binding["files"].items()
                if name.startswith("adapter_model.")
            ),
        },
    }


def _attach_local_peft_adapter(
    model: Any,
    *,
    adapter_path: Path,
    adapter_config: Mapping[str, Any],
) -> Any:
    try:
        from peft import LoraConfig, PeftModel
        from peft.tuners.lora import LoraLayer

        compatible = dict(adapter_config)
        for key in _PEFT_COMPATIBILITY_DEFAULTS:
            compatible.pop(key, None)
        config = LoraConfig(**compatible)
        wrapped = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            config=config,
            device_map={"": 0},
            is_trainable=False,
            local_files_only=True,
        )
    except Exception as exc:
        raise PilotError(f"cannot load local PEFT adapter: {exc}") from exc
    if not any(isinstance(module, LoraLayer) for module in wrapped.modules()):
        raise PilotError("loaded PEFT adapter exposes no LoRA modules")
    meta_tensors = [
        *(name for name, tensor in wrapped.named_parameters() if tensor.is_meta),
        *(name for name, tensor in wrapped.named_buffers() if tensor.is_meta),
    ]
    if meta_tensors:
        raise PilotError(
            "loaded PEFT model contains unmaterialized meta tensors: "
            + ", ".join(meta_tensors[:8])
        )
    return wrapped


def run_stratified_pilot(
    *,
    model_path: Path | None = None,
    base_model_path: Path | None = None,
    adapter_path: Path | None = None,
    sample_manifest_path: Path,
    sample_manifest_sha256: str,
    output_dir: Path,
    model_label: str,
    load_in_4bit: bool = True,
    attn_implementation: str = "sdpa",
) -> Mapping[str, Any]:
    """Run the fixed 15-case pilot on exactly one visible GPU."""

    if _LABEL_RE.fullmatch(model_label) is None:
        raise PilotError("model_label contains unsupported characters")
    if output_dir.exists() or output_dir.is_symlink():
        raise PilotError(f"pilot output already exists: {output_dir}")
    _, verification_model_path, _ = _resolve_model_loading_mode(
        model_path=model_path,
        base_model_path=base_model_path,
        adapter_path=adapter_path,
    )
    visible_gpu = base_probe._require_single_visible_gpu()
    sample_manifest, sample_manifest_sha = _validate_sample_manifest(
        sample_manifest_path.expanduser().resolve(), sample_manifest_sha256
    )
    tokenizer_path, tokenizer_files, tokenizer_bundle_sha = _sample_tokenizer_binding(
        sample_manifest
    )

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, trust_remote_code=False
        )
    except Exception as exc:
        raise PilotError(f"cannot load release-bound tokenizer: {exc}") from exc

    bound, _, release_sha, system_prompt, config_sha = _load_bound_samples(
        sample_manifest,
        tokenizer=tokenizer,
        verification_model_path=verification_model_path,
    )
    rebound_path, rebound_files, rebound_bundle_sha = _sample_tokenizer_binding(
        sample_manifest
    )
    if (
        rebound_path != tokenizer_path
        or base_probe.canonical_json(rebound_files)
        != base_probe.canonical_json(tokenizer_files)
        or rebound_bundle_sha != tokenizer_bundle_sha
    ):
        raise PilotError("tokenizer binding changed during verification")
    model_source = _prepare_model_source(
        model_path=model_path,
        base_model_path=base_model_path,
        adapter_path=adapter_path,
    )
    model_fingerprint = model_source["model_fingerprint"]
    adapter_fingerprint = model_source["adapter_fingerprint"]
    provenance = {
        "sample_manifest_sha256": sample_manifest_sha,
        "release_manifest_sha256": release_sha,
        "train_dataset_sha256": sample_manifest["dataset"]["sha256"],
        "unique_train_manifest_sha256": sample_manifest["dataset"][
            "unique_manifest_sha256"
        ],
        "training_config_sha256": config_sha,
        "tokenizer_bundle_sha256": tokenizer_bundle_sha,
        "model": model_fingerprint,
        "adapter": adapter_fingerprint,
        "effective_model": model_source["effective_model_fingerprint"],
        "model_source": model_source["provenance"],
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    launch = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "initializing",
        "created_at_utc": base_probe.utc_now(),
        "model_label": model_label,
        "provenance": provenance,
        "generation": sample_manifest["selection"]["generation_contract"],
        "runtime": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "cuda_visible_devices": visible_gpu,
        },
    }
    _write_exclusive_readonly_json(output_dir / "launch.json", launch)

    result_path = output_dir / "results.jsonl"
    results: list[Mapping[str, Any]] = []
    model: Any = None
    started = datetime.now(timezone.utc)
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise PilotError("pilot requires exactly one visible CUDA device")
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise PilotError("tokenizer has neither pad nor EOS token")
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
            str(model_source["load_path"]),
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            quantization_config=quantization_config,
            device_map={"": 0},
            local_files_only=True,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        if model_source["mode"] == PEFT_ADAPTER_MODE:
            model = _attach_local_peft_adapter(
                model,
                adapter_path=model_source["adapter_path"],
                adapter_config=model_source["adapter_config"],
            )
        model.eval()
        model.config.use_cache = True
        eos_value = getattr(model.generation_config, "eos_token_id", None)
        if eos_value is None:
            eos_value = tokenizer.eos_token_id
        eos_ids = base_probe._normalize_eos_ids(eos_value)
        if not eos_ids:
            raise PilotError("model/tokenizer has no EOS token ID")

        torch.cuda.reset_peak_memory_stats()
        with result_path.open("x", encoding="utf-8") as handle:
            for case in _generation_cases(sample_manifest):
                sample = case["sample"]
                sample_id = str(sample["sample_id"])
                source = bound[sample_id]
                prompt_ids = base_probe._prompt_ids(
                    tokenizer,
                    base_probe._messages(system_prompt, source["prompt"]),
                )
                if len(prompt_ids) != sample["prompt_token_count"]:
                    raise PilotError(f"prompt token-count drift for {sample_id}")
                context_limit = int(
                    getattr(model.config, "max_position_embeddings", 0) or 0
                )
                if context_limit and len(prompt_ids) + MAX_NEW_TOKENS > context_limit:
                    raise PilotError(
                        f"context overflow for {sample_id}: prompt={len(prompt_ids)}, "
                        f"completion={MAX_NEW_TOKENS}, limit={context_limit}"
                    )
                seed = int(case["seed"])
                transformers.set_seed(seed)
                input_ids = torch.tensor(
                    [prompt_ids], dtype=torch.long, device="cuda:0"
                )
                attention_mask = torch.ones_like(input_ids)
                sampled = case["generation_mode"] == "sampled"
                generation_kwargs: dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": sorted(eos_ids),
                    "use_cache": True,
                    "do_sample": sampled,
                }
                if sampled:
                    generation_kwargs.update(
                        temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P
                    )
                with torch.inference_mode():
                    sequences = model.generate(**generation_kwargs)
                generated_ids = sequences[0, input_ids.shape[1] :].tolist()
                completion = base_probe.decode_completion_preserving_boundary(
                    tokenizer, generated_ids, eos_ids
                )
                metrics = base_probe.analyze_completion(
                    text=completion,
                    generated_token_ids=generated_ids,
                    eos_token_ids=eos_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    tail_tokens=TAIL_TOKENS,
                )
                replay = replay_decision_dense_v3(
                    completion,
                    source["target"],
                    hit_eos=bool(metrics["hit_eos"]),
                    cap_reached=bool(metrics["cap_reached"]),
                )
                result = {
                    "schema_version": PROBE_SCHEMA_VERSION,
                    "model_label": model_label,
                    "sample_id": sample_id,
                    "split": "train",
                    "target_direction": sample["direction"],
                    "target_magnitude_bp": sample["magnitude_bp"],
                    "generation_mode": case["generation_mode"],
                    "generation_index": case["generation_index"],
                    "seed": seed,
                    "generation_parameters": {
                        "do_sample": sampled,
                        "temperature": SAMPLE_TEMPERATURE if sampled else 0.0,
                        "top_p": SAMPLE_TOP_P if sampled else 1.0,
                        "max_new_tokens": MAX_NEW_TOKENS,
                    },
                    "prompt_token_count": len(prompt_ids),
                    "completion_sha256": base_probe.sha256_text(completion),
                    "completion": completion,
                    "provenance": {
                        "sample_manifest_sha256": sample_manifest_sha,
                        "release_manifest_sha256": release_sha,
                        "training_config_sha256": config_sha,
                        "tokenizer_bundle_sha256": tokenizer_bundle_sha,
                        "model_sha256": model_fingerprint["sha256"],
                        **model_source["result_provenance"],
                    },
                    **metrics,
                    "decision_dense_v3_reward": replay["reward"],
                    "decision_dense_v3_nonzero": replay["nonzero"],
                    "response_format": replay["response_format"],
                    "strict_json": replay["strict_json"],
                    "fenced_json": replay["fenced_json"],
                    "decision_prediction": replay["prediction"],
                    "decision_direction_correct": replay["direction_correct"],
                    "decision_exact": replay["exact"],
                    "decision_rejection_reason": replay["rejection_reason"],
                    "decision_forced_zero_reason": replay["forced_zero_reason"],
                }
                handle.write(base_probe.canonical_json(result) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                results.append(result)
                del input_ids, attention_mask, sequences
        result_path.chmod(0o444)

        summary = dict(summarize_stratified_results(results, provenance=provenance))
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        summary.update(
            {
                "created_at_utc": base_probe.utc_now(),
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
                    "sha256": base_probe.sha256_file(result_path),
                },
            }
        )
        _write_exclusive_readonly_json(output_dir / "summary.json", summary)
        output_dir.chmod(0o555)
        return summary
    except Exception as exc:
        if result_path.exists():
            result_path.chmod(0o444)
        failure = {
            **launch,
            "status": "failed",
            "failed_at_utc": base_probe.utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_cases": len(results),
        }
        _write_exclusive_readonly_json(output_dir / "failure.json", failure)
        output_dir.chmod(0o555)
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="create immutable train pilot manifest"
    )
    prepare.add_argument("--release", required=True, type=Path)
    prepare.add_argument("--release-manifest-sha256", required=True)
    prepare.add_argument("--training-config", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)

    run = subparsers.add_parser("run", help="run the immutable 15-case pilot")
    model_source = run.add_mutually_exclusive_group(required=True)
    model_source.add_argument(
        "--model",
        type=Path,
        help="Local merged model directory.",
    )
    model_source.add_argument(
        "--base-model",
        type=Path,
        help="Local base model directory; requires --adapter.",
    )
    run.add_argument(
        "--adapter",
        type=Path,
        help="Local PEFT checkpoint adapter; requires --base-model.",
    )
    run.add_argument("--sample-manifest", required=True, type=Path)
    run.add_argument("--sample-manifest-sha256", required=True)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--model-label", required=True)
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
            root, manifest_path, manifest, manifest_sha, runtime_binding = (
                _verify_probe_release(
                    args.release,
                    args.release_manifest_sha256,
                    training_config=args.training_config,
                )
            )
            tokenizer_path, _, _ = _tokenizer_binding_for_release(
                manifest, runtime_binding=runtime_binding
            )
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(tokenizer_path), local_files_only=True, trust_remote_code=False
            )

            reuse_verified_release = _reuse_verified_release(
                root=root,
                manifest_path=manifest_path,
                manifest=manifest,
                manifest_sha256=manifest_sha,
                runtime_binding=runtime_binding,
            )

            artifact = build_stratified_manifest(
                release_root=args.release,
                release_manifest_sha256=args.release_manifest_sha256,
                tokenizer=tokenizer,
                training_config=args.training_config,
                seed=args.seed,
                release_verifier=reuse_verified_release,
            )
            _write_exclusive_readonly_json(args.output, artifact)
            print(
                base_probe.canonical_json(
                    {
                        "status": "prepared",
                        "path": str(args.output.resolve()),
                        "sha256": base_probe.sha256_file(args.output),
                        "samples": 3,
                        "cases": 15,
                    }
                )
            )
            return 0
        summary = run_stratified_pilot(
            model_path=args.model,
            base_model_path=args.base_model,
            adapter_path=args.adapter,
            sample_manifest_path=args.sample_manifest,
            sample_manifest_sha256=args.sample_manifest_sha256,
            output_dir=args.output_dir,
            model_label=args.model_label,
            load_in_4bit=args.load_in_4bit,
            attn_implementation=args.attn_implementation,
        )
        print(base_probe.canonical_json(summary))
        return 0 if summary["quality_status"] == "passed" else 2
    except (PilotError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
