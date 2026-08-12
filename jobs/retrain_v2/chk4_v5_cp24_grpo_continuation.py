"""Hash-bound chk4 v5 checkpoint-24 GRPO smoke and continuation.

This workflow is intentionally independent from the historical v5 canonical
GRPO stage.  Its only model parent is the create-only, exact-verified merge of
the explicitly selected SFT ``checkpoint-24``.  A two-step GPU1 smoke must pass
before a separate full-run authorization can be created.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from jobs.retrain_v2 import merge_chk4_selected_sft_checkpoint as selected
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


SOURCE_PROFILE = selected.V5_PROFILE_NAME
SOURCE_BRANCH_ID = (
    "chk4_from_chk1_cp200_sft_grpo_hier_balanced_lr1e5_steps24_v5_20260811"
)
CHECKPOINT_STEP = 24
CHECKPOINT_SHA256 = "91310635af35dc97ad440bba748795134572838fbe855dca0cd4bb6781ee95bf"
PILOT_MANIFEST_SHA256 = (
    "bd5aac9ac26909b6ce50de35506544c9050cc9c3a31c39a318216daf9408ec4c"
)
PILOT_SUMMARY_SHA256 = (
    "0a2064f11b9c88a0598a62da311e61176415277aa222b048c4daf0e1d0013d1e"
)
PILOT_RESULTS_SHA256 = (
    "a45809955fbfd0557b4cb134b4cc4fc5043dace28889e843741c7b38b48d980a"
)
GRPO_RELEASE_SHA256 = "8b05dce09bcb0a0b27ee240da1e5d730be93921ce19e650a3b3e8b2689adb893"
GRPO_TRAIN_SHA256 = "54ca475e08a3da2995424bf5d2c9b13626bc15b88a17d8e8793e670ee8afac24"
SMOKE_SOURCE_INDICES = (131, 55, 125, 138)
SMOKE_SOURCE_ROWS = (
    ("dec-ed9ce1ed2c210798357e0bb5", "hold", 0),
    ("dec-5848e27930ae03a1fa6bd5cb", "hike", 25),
    ("dec-de9d3a26759a82709f068f65", "hold", 0),
    ("dec-f66acacb4f43fea1e06bbfe8", "cut", 25),
)
SMOKE_SCHEMA = "chk4-v5-cp24-grpo-smoke-authorization-v1"
FULL_SCHEMA = "chk4-v5-cp24-grpo-full-authorization-v1"
GATE_SCHEMA = "chk4-v5-cp24-grpo-smoke-gate-v2"
STEP_GATE_SCHEMA = "chk4-decision-grpo-smoke-step-gate-v2"
SMOKE_STEP_TARGET_DIRECTIONS = {
    1: ("hold", "hike"),
    2: ("hold", "cut"),
}
SMOKE_GENERATIONS_PER_PROMPT = 4
SMOKE_MIN_GRAD_NORM = 1e-12
SMOKE_SEED = 5
SMOKE_DATA_SEED = 5
SMOKE_SHUFFLE_DATASET = True
SMOKE_TRAIN_ROWS = 141
SMOKE_CALLBACK_NAME = "chk4_decision_smoke_gate"
GRPO_DATASET_ROLE = "decision_grpo"
RUNTIME_PROFILE_PREFIX = "v5_cp24_selected"
CONTINUATION_SCHEMA_PREFIX = "chk4-v5-cp24-grpo"
FORBIDDEN_CANONICAL_SCOPE = "historical_v5_canonical_grpo"
ADDITIONAL_FORBIDDEN_SCOPES: tuple[str, ...] = ()
CONTINUATION_IMPLEMENTATION_FILES: tuple[str, ...] = ()
MAX_COMPLETION_LENGTH = 1024
AUTHORIZATION_CONTEXT_PROVIDER: Callable[[Path, "Phase"], Mapping[str, Any]] | None = (
    None
)
_TRAINING_LOCK_HANDLE = None

SOURCE_RUN_ROOT = Path(f"output/training/retrain_v2/{SOURCE_BRANCH_ID}")
PILOT_MANIFEST = (
    SOURCE_RUN_ROOT / "selected_sft_checkpoints/train_stratified_manifest_v1.json"
)
PILOT_SUMMARY = (
    SOURCE_RUN_ROOT / "selected_sft_checkpoints/checkpoint-24/pilot_v1/summary.json"
)
SELECTED_ROOT = SOURCE_RUN_ROOT / "selected_sft_checkpoints/checkpoint-24"
SELECTED_MODEL = SELECTED_ROOT / "merged/chk4_sft"
SELECTION_RECEIPT = SELECTED_ROOT / "receipts/selection_receipt.json"


class ContinuationError(RuntimeError):
    """A continuation input, output, or gate is not trustworthy."""


@dataclass(frozen=True)
class Phase:
    name: str
    branch_id: str
    config: Path
    config_sha256: str
    authorization_schema: str
    max_steps: int

    @property
    def run_root(self) -> Path:
        return Path(f"output/training/retrain_v2/{self.branch_id}")


SMOKE = Phase(
    name="smoke",
    branch_id="chk4_from_hier_balanced_v5_cp24_selected_grpo_smoke_v1_20260811",
    config=Path(
        "configs/retrain_v2/"
        "chk4_decision_grpo_from_v5_cp24_selected_smoke_v1_20260811.yaml"
    ),
    config_sha256=("70a848df0a0ab1f1841406543091631affb76808dfae988e44a58aaf22860261"),
    authorization_schema=SMOKE_SCHEMA,
    max_steps=2,
)
FULL = Phase(
    name="full",
    branch_id="chk4_from_hier_balanced_v5_cp24_selected_grpo_full_v1_20260811",
    config=Path(
        "configs/retrain_v2/"
        "chk4_decision_grpo_from_v5_cp24_selected_full_v1_20260811.yaml"
    ),
    config_sha256=("48540132da9dd64d4dfbc4450c10a3d8e0117e274a8096f71460872edf678c83"),
    authorization_schema=FULL_SCHEMA,
    max_steps=-1,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _resolve(repo_root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContinuationError(f"{label} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContinuationError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContinuationError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContinuationError(f"{label} must be an object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise ContinuationError(f"{label} is missing or unsafe: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    raise ContinuationError(f"{label} contains blank row {line_number}")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ContinuationError(
                        f"{label} row {line_number} is not an object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContinuationError(f"{label} is invalid JSONL: {path}") from exc
    return rows


def _read_yaml(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ContinuationError(f"{label} is missing or unsafe: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ContinuationError(f"{label} is invalid YAML: {path}") from exc
    if not isinstance(value, dict):
        raise ContinuationError(f"{label} must be a mapping: {path}")
    return value


def _require_equal(observed: Any, expected: Any, *, label: str) -> None:
    if observed != expected:
        raise ContinuationError(
            f"{label} drift: expected={expected!r}, observed={observed!r}"
        )


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise ContinuationError(f"refusing to overwrite create-only record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return sha256_file(path)


def _config_path(repo_root: Path, phase: Phase) -> Path:
    path = (repo_root / phase.config).resolve()
    if sha256_file(path) != phase.config_sha256:
        raise ContinuationError(f"{phase.name} GRPO config hash drift")
    return path


def _implementation_descriptor(repo_root: Path, relative_path: str) -> dict[str, str]:
    path = (repo_root / relative_path).resolve()
    if not path.is_file() or path.is_symlink():
        raise ContinuationError(f"runtime implementation is missing or unsafe: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def _runtime_implementation_contract(repo_root: Path) -> dict[str, dict[str, str]]:
    contract = {
        "reward": _implementation_descriptor(
            repo_root,
            "src/open_r1/trainer/rewards/reward_funcs/decision_reward_v3.py",
        ),
        "reward_registry": _implementation_descriptor(
            repo_root, "src/open_r1/trainer/rewards/reward_register.py"
        ),
        "callbacks": _implementation_descriptor(
            repo_root, "src/open_r1/utils/callbacks.py"
        ),
        "trainer_runtime": _implementation_descriptor(
            repo_root, "src/open_r1/trainer/trainer.py"
        ),
        "grpo_runtime": _implementation_descriptor(
            repo_root, "src/open_r1/trainer/grpo_trainer.py"
        ),
        "entrypoint": _implementation_descriptor(repo_root, "jobs/train/train_grpo.py"),
    }
    for index, relative_path in enumerate(CONTINUATION_IMPLEMENTATION_FILES, start=1):
        contract[f"continuation_{index}"] = _implementation_descriptor(
            repo_root, relative_path
        )
    return contract


def _smoke_sampling_contract(
    repo_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    import inspect
    from itertools import islice

    from trl.trainer import grpo_trainer as trl_grpo_trainer
    from trl.trainer.utils import RepeatSampler

    dataset_dir = _resolve(repo_root, config.get("dataset_name"), label="smoke dataset")
    if not dataset_dir.is_dir():
        raise ContinuationError("smoke dataset path is not a directory")
    train_path = dataset_dir / "train.jsonl"
    if (
        not train_path.is_file()
        or train_path.is_symlink()
        or sha256_file(train_path) != GRPO_TRAIN_SHA256
    ):
        raise ContinuationError("smoke source train hash drift")
    rows = _read_jsonl(train_path, label="smoke source train")
    if len(rows) != SMOKE_TRAIN_ROWS:
        raise ContinuationError("smoke source train row count drift")
    sampler = RepeatSampler(
        rows,
        mini_repeat_count=4,
        batch_size=2,
        repeat_count=8,
        shuffle=SMOKE_SHUFFLE_DATASET,
        seed=SMOKE_SEED,
    )
    observed_prefix = tuple(int(value) for value in islice(sampler, 72))
    expected_first_chunk = (SMOKE_SOURCE_INDICES[0],) * 4 + (
        SMOKE_SOURCE_INDICES[1],
    ) * 4
    expected_second_chunk = (SMOKE_SOURCE_INDICES[2],) * 4 + (
        SMOKE_SOURCE_INDICES[3],
    ) * 4
    expected_prefix = expected_first_chunk * 8 + expected_second_chunk
    if observed_prefix != expected_prefix:
        raise ContinuationError(
            "TRL RepeatSampler no longer produces the sealed two-step prefix"
        )
    observed_indices = tuple(observed_prefix[offset] for offset in (0, 4, 64, 68))
    observed_rows = tuple(
        (
            str(rows[index].get("sample_id")),
            str(rows[index].get("direction")),
            int(rows[index].get("magnitude_bp")),
        )
        for index in observed_indices
    )
    if observed_rows != SMOKE_SOURCE_ROWS:
        raise ContinuationError("smoke seeded direction schedule drift")
    sampler_source = Path(inspect.getfile(RepeatSampler)).resolve()
    integration_source = Path(inspect.getfile(trl_grpo_trainer)).resolve()
    return {
        "type": "trl_repeat_sampler_seeded_direction_coverage_v1",
        "seed": SMOKE_SEED,
        "data_seed": SMOKE_DATA_SEED,
        "shuffle_dataset": SMOKE_SHUFFLE_DATASET,
        "source_train": {
            "path": str(train_path.resolve()),
            "sha256": GRPO_TRAIN_SHA256,
            "rows": SMOKE_TRAIN_ROWS,
        },
        "source_indices": list(SMOKE_SOURCE_INDICES),
        "source_rows": [
            {
                "sample_id": sample_id,
                "direction": direction,
                "magnitude_bp": magnitude,
            }
            for sample_id, direction, magnitude in SMOKE_SOURCE_ROWS
        ],
        "optimizer_step_target_groups": {
            "1": ["hold", "hike"],
            "2": ["hold", "cut"],
        },
        "num_generations": 4,
        "unique_prompts_per_step": 2,
        "expected_reward_rows_per_step": 8,
        "verified_sampler_prefix": {
            "length": len(observed_prefix),
            "sha256": _sha256_text(_canonical_json(list(observed_prefix))),
            "first_chunk": list(expected_first_chunk),
            "first_chunk_repeat_count": 8,
            "second_chunk_first_repeat": list(expected_second_chunk),
        },
        "sampler_source": {
            "path": str(sampler_source),
            "sha256": sha256_file(sampler_source),
        },
        "trainer_sampler_integration_source": {
            "path": str(integration_source),
            "sha256": sha256_file(integration_source),
        },
    }


def _validate_config(repo_root: Path, phase: Phase) -> dict[str, Any]:
    path = _config_path(repo_root, phase)
    config = _read_yaml(path, label=f"{phase.name} GRPO config")
    expected_model = (repo_root / SELECTED_MODEL).resolve()
    expected_output = (repo_root / phase.run_root / "adapters/chk4_grpo").resolve()
    expected_merged = (repo_root / phase.run_root / "merged/chk4_grpo").resolve()
    _require_equal(
        _resolve(repo_root, config.get("model_name_or_path"), label="model parent"),
        expected_model,
        label=f"{phase.name} selected model parent",
    )
    _require_equal(
        _resolve(repo_root, config.get("output_dir"), label="output"),
        expected_output,
        label=f"{phase.name} output",
    )
    _require_equal(
        _resolve(repo_root, config.get("peft_merged_model_path"), label="merge"),
        expected_merged,
        label=f"{phase.name} merge output",
    )
    expected_common = {
        "dataset_chk4_role": GRPO_DATASET_ROLE,
        "dataset_chk4_release_manifest_sha256": GRPO_RELEASE_SHA256,
        "reward_funcs": ["decision_dense_v3"],
        "reward_weights": [1.0],
        "save_reward": True,
        "max_prompt_length": 2560,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "num_generations": 4,
        "generation_batch_size": 8,
        "temperature": 0.7,
        "top_p": 0.9,
        "gradient_accumulation_steps": 8,
        "per_device_train_batch_size": 1,
        "mask_truncated_completions": True,
        "scale_rewards": "group",
        "log_completions": True,
        "overwrite_output_dir": False,
        "resume_from_checkpoint": None,
        "logging_steps": 1,
        "save_steps": 1,
    }
    for key, expected in expected_common.items():
        _require_equal(config.get(key), expected, label=f"{phase.name} config {key}")
    if phase is SMOKE:
        _require_equal(config.get("max_steps"), 2, label="smoke max_steps")
        _require_equal(config.get("seed"), SMOKE_SEED, label="smoke sampler seed")
        _require_equal(
            config.get("data_seed"), SMOKE_DATA_SEED, label="smoke data seed"
        )
        _require_equal(
            config.get("shuffle_dataset"),
            SMOKE_SHUFFLE_DATASET,
            label="smoke shuffle contract",
        )
        _smoke_sampling_contract(repo_root, config)
        callbacks = config.get("callbacks")
        if not isinstance(callbacks, list) or SMOKE_CALLBACK_NAME not in callbacks:
            raise ContinuationError("smoke config is missing its immediate step gate")
        _require_equal(
            config.get("checkpoint_keep_every_n_steps"),
            0,
            label="smoke periodic retention",
        )
    else:
        if "max_steps" in config:
            raise ContinuationError("full config must use its sealed epoch contract")
        if SMOKE_CALLBACK_NAME in (config.get("callbacks") or []):
            raise ContinuationError("full config must not reuse the two-step callback")
        _require_equal(
            config.get("checkpoint_keep_every_n_steps"),
            10,
            label="full periodic retention",
        )
    return config


def _selection_kwargs(repo_root: Path) -> dict[str, Any]:
    return {
        "repo_root": repo_root,
        "checkpoint_step": CHECKPOINT_STEP,
        "pilot_manifest": repo_root / PILOT_MANIFEST,
        "pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
        "pilot_summary": repo_root / PILOT_SUMMARY,
        "pilot_summary_sha256": PILOT_SUMMARY_SHA256,
    }


def _published_selection(repo_root: Path):
    selected._activate_selection_profile(SOURCE_PROFILE)
    try:
        plan, receipt = selected.verify_published_selection(
            **_selection_kwargs(repo_root)
        )
    except selected.SelectionMergeError as exc:
        raise ContinuationError(str(exc)) from exc
    if plan.checkpoint_step != CHECKPOINT_STEP:
        raise ContinuationError(f"selection is not checkpoint-{CHECKPOINT_STEP}")
    if plan.checkpoint_fingerprint.get("sha256") != CHECKPOINT_SHA256:
        raise ContinuationError(
            f"selected checkpoint-{CHECKPOINT_STEP} fingerprint drift"
        )
    if plan.pilot_results.get("sha256") != PILOT_RESULTS_SHA256:
        raise ContinuationError("selected pilot results hash drift")
    if receipt.get("root_adapter_is_source") is not False:
        raise ContinuationError("root adapter cannot be the selected source")
    return plan, receipt


def _authorization_path(repo_root: Path, phase: Phase) -> Path:
    lexical_root = repo_root / phase.run_root
    resolved_root = lexical_root.resolve()
    if resolved_root != lexical_root or lexical_root.is_symlink():
        raise ContinuationError(f"{phase.name} run root contains a symlink")
    return resolved_root / "receipts/authorization.json"


def _smoke_gate_path(repo_root: Path) -> Path:
    return (repo_root / SMOKE.run_root / "receipts/smoke_gate.json").resolve()


def _authorization_payload(repo_root: Path, phase: Phase) -> dict[str, Any]:
    config = _validate_config(repo_root, phase)
    # Full authorization is intentionally impossible until the independent
    # smoke gate has been published and revalidated.
    smoke_gate = _verify_smoke_gate(repo_root) if phase is FULL else None
    plan, selection_receipt = _published_selection(repo_root)
    selection_integrity = selection_receipt.get("integrity")
    if not isinstance(selection_integrity, Mapping):
        raise ContinuationError("selection receipt integrity is missing")
    exact = _read_json(plan.exact_evidence_path, label="selection exact evidence")
    validate_manifest_integrity(exact)
    output = _resolve(repo_root, config["output_dir"], label="GRPO output")
    merged_output = _resolve(
        repo_root, config["peft_merged_model_path"], label="GRPO merge output"
    )
    for path, label in ((output, "adapter output"), (merged_output, "merge output")):
        if path.exists() or path.is_symlink():
            raise ContinuationError(
                f"fresh {phase.name} {label} already exists: {path}"
            )
    payload = {
        "schema_version": phase.authorization_schema,
        "status": "authorized",
        "created_at_utc": _utc_now(),
        "authorization_basis": "explicit_user_instruction",
        "phase": phase.name,
        "branch_id": phase.branch_id,
        "scope": {
            "allowed": [f"chk4_decision_grpo_{phase.name}"],
            "canonical_dag_bindable": False,
            "not_authorized": [
                FORBIDDEN_CANONICAL_SCOPE,
                "root_sft_adapter",
                "different_sft_checkpoint",
                "gpu0",
                *ADDITIONAL_FORBIDDEN_SCOPES,
            ],
        },
        "source": {
            "profile": SOURCE_PROFILE,
            "branch_id": SOURCE_BRANCH_ID,
            "checkpoint_step": CHECKPOINT_STEP,
            "source_checkpoint": dict(plan.checkpoint_fingerprint),
            "root_adapter_used": False,
            "source_branch_authorization": dict(plan.parent_lineage["authorization"]),
            "selection_receipt": {
                "path": str(plan.receipt_path),
                "file_sha256": sha256_file(plan.receipt_path),
                "payload_sha256": selection_integrity["payload_sha256"],
            },
            "exact_merge_evidence": {
                "path": str(plan.exact_evidence_path),
                "file_sha256": sha256_file(plan.exact_evidence_path),
                "payload_sha256": exact["integrity"]["payload_sha256"],
            },
            "selected_merged_model": fingerprint_artifact_path(plan.destination),
            "pilot": {
                "manifest_sha256": PILOT_MANIFEST_SHA256,
                "summary_sha256": PILOT_SUMMARY_SHA256,
                "results_sha256": PILOT_RESULTS_SHA256,
            },
        },
        "config": {
            "path": str(_config_path(repo_root, phase)),
            "sha256": phase.config_sha256,
        },
        "runtime": {
            "gpu": 1,
            "visible_device_count": 1,
            "num_processes": 1,
            "reward_funcs": ["decision_dense_v3"],
            "max_completion_length": MAX_COMPLETION_LENGTH,
            "max_steps": phase.max_steps,
            "fresh_output": True,
            "sampling_contract": (
                _smoke_sampling_contract(repo_root, config) if phase is SMOKE else None
            ),
            "implementation": _runtime_implementation_contract(repo_root),
        },
        "grpo_release_manifest_sha256": GRPO_RELEASE_SHA256,
        "output": {
            "run_root": str((repo_root / phase.run_root).resolve()),
            "adapter": str(output),
            "merged": str(merged_output),
        },
        "smoke_gate": smoke_gate,
    }
    if AUTHORIZATION_CONTEXT_PROVIDER is not None:
        payload["continuation_context"] = dict(
            AUTHORIZATION_CONTEXT_PROVIDER(repo_root, phase)
        )
    return payload


def authorize(repo_root: Path, phase: Phase, *, execute: bool) -> dict[str, Any]:
    payload = _authorization_payload(repo_root, phase)
    sealed = seal_manifest(payload)
    path = _authorization_path(repo_root, phase)
    if not execute:
        return {**sealed, "authorization_path": str(path), "create_only": True}
    _write_create_only(path, sealed)
    return sealed


def _verify_authorization(repo_root: Path, phase: Phase) -> dict[str, Any]:
    path = _authorization_path(repo_root, phase)
    authorization = _read_json(path, label=f"{phase.name} authorization")
    validate_manifest_integrity(authorization)
    if (
        authorization.get("schema_version") != phase.authorization_schema
        or authorization.get("status") != "authorized"
        or authorization.get("phase") != phase.name
        or authorization.get("branch_id") != phase.branch_id
    ):
        raise ContinuationError(f"{phase.name} authorization scope drift")
    config = authorization.get("config")
    source = authorization.get("source")
    runtime = authorization.get("runtime")
    output_binding = authorization.get("output")
    config_payload = _validate_config(repo_root, phase)
    expected_output = {
        "run_root": str((repo_root / phase.run_root).resolve()),
        "adapter": str(
            _resolve(repo_root, config_payload["output_dir"], label="GRPO output")
        ),
        "merged": str(
            _resolve(
                repo_root,
                config_payload["peft_merged_model_path"],
                label="GRPO merge output",
            )
        ),
    }
    expected_implementation = _runtime_implementation_contract(repo_root)
    expected_sampling_contract = (
        _smoke_sampling_contract(repo_root, config_payload) if phase is SMOKE else None
    )
    expected_continuation_context = (
        dict(AUTHORIZATION_CONTEXT_PROVIDER(repo_root, phase))
        if AUTHORIZATION_CONTEXT_PROVIDER is not None
        else None
    )
    if (
        authorization.get("authorization_basis") != "explicit_user_instruction"
        or authorization.get("grpo_release_manifest_sha256") != GRPO_RELEASE_SHA256
        or config
        != {
            "path": str(_config_path(repo_root, phase)),
            "sha256": phase.config_sha256,
        }
        or not isinstance(source, Mapping)
        or source.get("profile") != SOURCE_PROFILE
        or source.get("branch_id") != SOURCE_BRANCH_ID
        or source.get("checkpoint_step") != CHECKPOINT_STEP
        or source.get("root_adapter_used") is not False
        or source.get("source_checkpoint", {}).get("sha256") != CHECKPOINT_SHA256
        or source.get("pilot")
        != {
            "manifest_sha256": PILOT_MANIFEST_SHA256,
            "summary_sha256": PILOT_SUMMARY_SHA256,
            "results_sha256": PILOT_RESULTS_SHA256,
        }
        or not isinstance(runtime, Mapping)
        or runtime.get("gpu") != 1
        or runtime.get("visible_device_count") != 1
        or runtime.get("num_processes") != 1
        or runtime.get("reward_funcs") != ["decision_dense_v3"]
        or runtime.get("max_completion_length") != MAX_COMPLETION_LENGTH
        or runtime.get("max_steps") != phase.max_steps
        or runtime.get("sampling_contract") != expected_sampling_contract
        or runtime.get("implementation") != expected_implementation
        or output_binding != expected_output
        or authorization.get("continuation_context") != expected_continuation_context
    ):
        raise ContinuationError(f"{phase.name} authorization binding drift")
    plan, current_receipt = _published_selection(repo_root)
    descriptor = source.get("selection_receipt")
    exact_descriptor = source.get("exact_merge_evidence")
    exact = _read_json(plan.exact_evidence_path, label="selection exact evidence")
    validate_manifest_integrity(exact)
    if (
        not isinstance(descriptor, Mapping)
        or Path(str(descriptor.get("path"))).resolve() != plan.receipt_path
        or descriptor.get("file_sha256") != sha256_file(plan.receipt_path)
        or descriptor.get("payload_sha256")
        != current_receipt["integrity"]["payload_sha256"]
        or source.get("source_branch_authorization")
        != dict(plan.parent_lineage["authorization"])
        or not isinstance(exact_descriptor, Mapping)
        or Path(str(exact_descriptor.get("path"))).resolve() != plan.exact_evidence_path
        or exact_descriptor.get("file_sha256") != sha256_file(plan.exact_evidence_path)
        or exact_descriptor.get("payload_sha256")
        != exact["integrity"]["payload_sha256"]
        or source.get("selected_merged_model")
        != fingerprint_artifact_path(plan.destination)
    ):
        raise ContinuationError(f"{phase.name} selected-model binding drift")
    if phase is FULL:
        gate = _verify_smoke_gate(repo_root)
        if authorization.get("smoke_gate") != gate:
            raise ContinuationError("full authorization smoke gate binding drift")
    elif authorization.get("smoke_gate") is not None:
        raise ContinuationError("smoke authorization cannot bind a future gate")
    return authorization


def training_command(repo_root: Path, phase: Phase) -> dict[str, Any]:
    authorization = _verify_authorization(repo_root, phase)
    config = _validate_config(repo_root, phase)
    output = _resolve(repo_root, config["output_dir"], label="GRPO output")
    if output.exists() or output.is_symlink():
        raise ContinuationError(
            f"{phase.name} launch is fresh-only and output already exists: {output}"
        )
    argv = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        "1",
        "--mixed_precision",
        "bf16",
        "--dynamo_backend",
        "no",
        "-m",
        "jobs.train.train_grpo",
        "--config",
        str(_config_path(repo_root, phase)),
    ]
    return {
        "schema_version": f"{CONTINUATION_SCHEMA_PREFIX}-launch-v1",
        "status": "ready",
        "phase": phase.name,
        "branch_id": phase.branch_id,
        "gpus": [1],
        "num_processes": 1,
        "fresh_output": True,
        "authorization": {
            "path": str(_authorization_path(repo_root, phase)),
            "sha256": sha256_file(_authorization_path(repo_root, phase)),
            "payload_sha256": authorization["integrity"]["payload_sha256"],
        },
        "config": {
            "path": str(_config_path(repo_root, phase)),
            "sha256": phase.config_sha256,
        },
        "argv": argv,
    }


def _acquire_training_lock(repo_root: Path, phase: Phase) -> None:
    global _TRAINING_LOCK_HANDLE
    if _TRAINING_LOCK_HANDLE is not None:
        raise ContinuationError("continuation process already owns a training lock")
    lock_path = _authorization_path(repo_root, phase).parent.parent / ".training.lock"
    if lock_path.is_symlink():
        raise ContinuationError(f"{phase.name} training lock is unsafe")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise ContinuationError(
            f"another {phase.name} continuation process owns the training lock"
        ) from exc
    os.set_inheritable(handle.fileno(), True)
    _TRAINING_LOCK_HANDLE = handle


def execute_training(repo_root: Path, phase: Phase) -> None:
    command = training_command(repo_root, phase)
    _acquire_training_lock(repo_root, phase)
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "1",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "FOMC_CHK4_BRANCH_ID": phase.branch_id,
            "FOMC_CHK4_BRANCH_PROFILE": (f"{RUNTIME_PROFILE_PREFIX}_{phase.name}_v1"),
            "FOMC_CHK4_BRANCH_STAGE": "decision_grpo",
            "FOMC_CHK4_BRANCH_CONFIG_PATH": command["config"]["path"],
            "FOMC_CHK4_BRANCH_CONFIG_SHA256": command["config"]["sha256"],
        }
    )
    os.chdir(repo_root)
    os.execvpe(command["argv"][0], command["argv"], environment)


def _finite_positive(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContinuationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ContinuationError(f"{label} must be finite and positive")
    return result


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContinuationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContinuationError(f"{label} must be finite")
    return result


def _smoke_prediction_direction(row: Mapping[str, Any], *, label: str) -> str | None:
    prediction = row.get("prediction")
    if prediction is None:
        return None
    if not isinstance(prediction, Mapping) or set(prediction) != {
        "direction",
        "magnitude_bp",
    }:
        raise ContinuationError(f"{label} prediction schema is invalid")
    direction = prediction.get("direction")
    magnitude = prediction.get("magnitude_bp")
    if direction not in {"hold", "hike", "cut"}:
        raise ContinuationError(f"{label} prediction direction is invalid")
    if isinstance(magnitude, bool) or not isinstance(magnitude, int):
        raise ContinuationError(f"{label} prediction magnitude is invalid")
    allowed = {0} if direction == "hold" else {25, 50, 75, 100}
    if magnitude not in allowed:
        raise ContinuationError(f"{label} prediction magnitude is invalid")
    return str(direction)


def _replay_smoke_reward_batch(
    step: int, batch: list[dict[str, Any]]
) -> tuple[list[float], dict[str, dict[str, float | int]]]:
    expected_directions = SMOKE_STEP_TARGET_DIRECTIONS[step]
    expected_layout = [
        direction
        for direction in expected_directions
        for _ in range(SMOKE_GENERATIONS_PER_PROMPT)
    ]
    observed_layout: list[str] = []
    values: list[float] = []
    correct_flags: list[bool] = []
    semantic_scores: list[float] = []
    for offset, row in enumerate(batch, start=1):
        label = f"smoke step {step} reward row {offset}"
        if row.get("type") != "decision_dense_v3":
            raise ContinuationError(f"{label} schema drift")
        target = row.get("target")
        if not isinstance(target, Mapping) or set(target) != {
            "direction",
            "magnitude_bp",
        }:
            raise ContinuationError(f"{label} target schema is invalid")
        direction = target.get("direction")
        magnitude = target.get("magnitude_bp")
        if direction not in {"hold", "hike", "cut"}:
            raise ContinuationError(f"{label} target direction is invalid")
        if isinstance(magnitude, bool) or not isinstance(magnitude, int):
            raise ContinuationError(f"{label} target magnitude is invalid")
        allowed = {0} if direction == "hold" else {25, 50, 75, 100}
        if magnitude not in allowed:
            raise ContinuationError(f"{label} target magnitude is invalid")
        observed_layout.append(str(direction))

        reward = _finite(row.get("reward"), label=f"{label} reward")
        if not 0.0 <= reward <= 1.0:
            raise ContinuationError(f"{label} reward is outside [0, 1]")
        semantic = _finite(
            row.get("semantic_score_before_discount"),
            label=f"{label} semantic score",
        )
        if not 0.0 <= semantic <= 0.95:
            raise ContinuationError(f"{label} semantic score is outside [0, 0.95]")
        direction_correct = row.get("direction_correct")
        if not isinstance(direction_correct, bool):
            raise ContinuationError(f"{label} direction_correct must be boolean")
        predicted_direction = _smoke_prediction_direction(row, label=label)
        if direction_correct is not (predicted_direction == direction):
            raise ContinuationError(f"{label} direction_correct is inconsistent")
        values.append(reward)
        semantic_scores.append(semantic)
        correct_flags.append(direction_correct)
    if observed_layout != expected_layout:
        raise ContinuationError(
            f"smoke step {step} target layout drift: "
            f"expected={expected_layout}, observed={observed_layout}"
        )

    target_groups: dict[str, dict[str, float | int]] = {}
    for group_index, direction in enumerate(expected_directions):
        start = group_index * SMOKE_GENERATIONS_PER_PROMPT
        stop = start + SMOKE_GENERATIONS_PER_PROMPT
        group_values = values[start:stop]
        correct_nonzero = sum(
            correct and reward > 0.0 and semantic > 0.0
            for correct, reward, semantic in zip(
                correct_flags[start:stop],
                group_values,
                semantic_scores[start:stop],
                strict=True,
            )
        )
        target_groups[direction] = {
            "reward_rows": SMOKE_GENERATIONS_PER_PROMPT,
            "direction_correct_nonzero_count": correct_nonzero,
            "reward_mean": statistics.fmean(group_values),
            "reward_std": statistics.pstdev(group_values),
        }
    return values, target_groups


def _smoke_gate_payload(repo_root: Path) -> dict[str, Any]:
    authorization = _verify_authorization(repo_root, SMOKE)
    config = _validate_config(repo_root, SMOKE)
    output = _resolve(repo_root, config["output_dir"], label="smoke output")
    runtime_path = output / "resolved_runtime_config.json"
    state_path = output / "trainer_state.json"
    results_path = output / "train_results.json"
    reward_path = output / "reward.jsonl"
    safety_path = output / "runtime_safety.jsonl"
    step_gate_path = output / "chk4_smoke_step_gate.jsonl"
    runtime = _read_json(runtime_path, label="smoke runtime")
    branch = runtime.get("chk4_standalone_branch")
    rewards_binding = runtime.get("rewards")
    training = runtime.get("training")
    generation = runtime.get("generation")
    environment = runtime.get("environment")
    model = runtime.get("model")
    dataset = runtime.get("dataset")
    chk4_dataset = (
        dataset.get("chk4_decision") if isinstance(dataset, Mapping) else None
    )
    release = (
        chk4_dataset.get("release_manifest")
        if isinstance(chk4_dataset, Mapping)
        else None
    )
    if (
        not isinstance(branch, Mapping)
        or branch.get("branch_id") != SMOKE.branch_id
        or branch.get("profile") != f"{RUNTIME_PROFILE_PREFIX}_smoke_v1"
        or branch.get("stage") != "decision_grpo"
        or branch.get("config")
        != {"path": str(_config_path(repo_root, SMOKE)), "sha256": SMOKE.config_sha256}
        or rewards_binding
        != {"reward_funcs": ["decision_dense_v3"], "reward_weights": [1.0]}
        or not isinstance(training, Mapping)
        or training.get("max_steps") != 2
        or training.get("gradient_accumulation_steps") != 8
        or training.get("per_device_train_batch_size") != 1
        or training.get("seed") != SMOKE_SEED
        or _resolve(repo_root, training.get("output_dir"), label="runtime output")
        != output
        or not isinstance(generation, Mapping)
        or generation.get("max_prompt_length") != 2560
        or generation.get("max_completion_length") != MAX_COMPLETION_LENGTH
        or generation.get("num_generations") != 4
        or generation.get("temperature") != 0.7
        or generation.get("top_p") != 0.9
        or not isinstance(environment, Mapping)
        or environment.get("world_size") != 1
        or environment.get("process_index") != 0
        or environment.get("local_process_index") != 0
        or environment.get("cuda_visible_devices") != "1"
        or environment.get("cuda_available") is not True
        or environment.get("cuda_device_count") != 1
        or not isinstance(model, Mapping)
        or _resolve(repo_root, model.get("model_name_or_path"), label="runtime model")
        != (repo_root / SELECTED_MODEL).resolve()
        or not isinstance(release, Mapping)
        or release.get("sha256") != GRPO_RELEASE_SHA256
        or _resolve(
            repo_root,
            release.get("path"),
            label="runtime release manifest",
        )
        != _resolve(
            repo_root,
            config.get("dataset_chk4_release_manifest"),
            label="configured release manifest",
        )
    ):
        raise ContinuationError("smoke resolved-runtime binding drift")
    state = _read_json(state_path, label="smoke trainer state")
    _require_equal(state.get("global_step"), 2, label="smoke completed step")
    _require_equal(state.get("max_steps"), 2, label="smoke configured max_steps")
    results = _read_json(results_path, label="smoke train results")
    _finite(results.get("train_loss"), label="smoke train loss")
    reward_rows = _read_jsonl(reward_path, label="smoke reward audit")
    safety_rows = [
        row
        for row in _read_jsonl(safety_path, label="smoke runtime safety")
        if isinstance(row.get("metrics"), Mapping)
        and row["metrics"].get("loss") is not None
    ]
    step_gate_rows = _read_jsonl(step_gate_path, label="smoke immediate gate")
    if len(reward_rows) != 16 or len(safety_rows) != 2 or len(step_gate_rows) != 2:
        raise ContinuationError(
            "smoke does not contain exactly two complete step groups"
        )
    step_summaries: list[dict[str, Any]] = []
    for step in (1, 2):
        batch = reward_rows[(step - 1) * 8 : step * 8]
        values, target_groups = _replay_smoke_reward_batch(step, batch)
        correct_nonzero = sum(
            int(group["direction_correct_nonzero_count"])
            for group in target_groups.values()
        )
        reward_std = statistics.pstdev(values)
        safety = safety_rows[step - 1]
        if safety.get("step") != step:
            raise ContinuationError(f"smoke runtime safety step {step} is misaligned")
        metrics = safety["metrics"]
        loss = _finite(metrics.get("loss"), label=f"smoke step {step} loss")
        grad_norm = _finite(
            metrics.get("grad_norm"), label=f"smoke step {step} grad norm"
        )
        if grad_norm <= SMOKE_MIN_GRAD_NORM:
            raise ContinuationError(
                f"smoke step {step} grad norm did not exceed {SMOKE_MIN_GRAD_NORM}"
            )
        clipped = safety.get("completion_clipped_ratio")
        if (
            isinstance(clipped, bool)
            or not isinstance(clipped, (int, float))
            or not math.isfinite(float(clipped))
            or not 0.0 <= float(clipped) <= 0.25
        ):
            raise ContinuationError(f"smoke step {step} clipped ratio exceeded 25%")
        if any(
            int(group["direction_correct_nonzero_count"]) < 1
            or float(group["reward_std"]) <= 0.0
            for group in target_groups.values()
        ):
            raise ContinuationError(
                f"smoke step {step} has an unsafe per-target reward signal"
            )
        gate_row = step_gate_rows[step - 1]
        expected_checks = {
            "each_target_direction_correct_nonzero": True,
            "each_target_reward_std_gt_zero": True,
            "loss_finite": True,
            "grad_norm_gt_1e_12": True,
            "clipped_ratio_le_0_25": True,
        }
        gate_reward_mean = _finite(
            gate_row.get("reward_mean"), label=f"smoke step {step} gate reward mean"
        )
        gate_reward_std = _finite(
            gate_row.get("reward_std"), label=f"smoke step {step} gate reward std"
        )
        gate_loss = _finite(gate_row.get("loss"), label=f"smoke step {step} gate loss")
        gate_grad_norm = _finite(
            gate_row.get("grad_norm"), label=f"smoke step {step} gate grad norm"
        )
        gate_clipped = _finite(
            gate_row.get("completion_clipped_ratio"),
            label=f"smoke step {step} gate clipped ratio",
        )
        if (
            gate_row.get("schema_version") != STEP_GATE_SCHEMA
            or type(gate_row.get("step")) is not int
            or gate_row.get("step") != step
            or gate_row.get("status") != "passed"
            or _canonical_json(gate_row.get("checks"))
            != _canonical_json(expected_checks)
            or type(gate_row.get("reward_rows")) is not int
            or gate_row.get("reward_rows") != 8
            or gate_row.get("target_directions")
            != list(SMOKE_STEP_TARGET_DIRECTIONS[step])
            or _canonical_json(gate_row.get("target_groups"))
            != _canonical_json(target_groups)
            or type(gate_row.get("direction_correct_nonzero_count")) is not int
            or gate_row.get("direction_correct_nonzero_count") != correct_nonzero
            or not math.isclose(
                gate_reward_mean,
                statistics.fmean(values),
                abs_tol=1e-12,
            )
            or not math.isclose(gate_reward_std, reward_std, abs_tol=1e-12)
            or not math.isclose(gate_loss, loss, abs_tol=1e-12)
            or not math.isclose(gate_grad_norm, grad_norm, abs_tol=1e-12)
            or not math.isclose(gate_clipped, float(clipped), abs_tol=1e-12)
        ):
            raise ContinuationError(f"smoke step {step} immediate gate drift")
        step_summaries.append(
            {
                "step": step,
                "target_directions": list(SMOKE_STEP_TARGET_DIRECTIONS[step]),
                "target_groups": target_groups,
                "direction_correct_nonzero_count": correct_nonzero,
                "reward_mean": statistics.fmean(values),
                "reward_std": reward_std,
                "loss": loss,
                "grad_norm": grad_norm,
                "completion_clipped_ratio": float(clipped),
            }
        )
    return {
        "schema_version": GATE_SCHEMA,
        "status": "passed",
        "created_at_utc": _utc_now(),
        "branch_id": SMOKE.branch_id,
        "authorization": {
            "path": str(_authorization_path(repo_root, SMOKE)),
            "file_sha256": sha256_file(_authorization_path(repo_root, SMOKE)),
            "payload_sha256": authorization["integrity"]["payload_sha256"],
        },
        "config": {
            "path": str(_config_path(repo_root, SMOKE)),
            "sha256": SMOKE.config_sha256,
        },
        "source_checkpoint_sha256": CHECKPOINT_SHA256,
        "pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
        "pilot_summary_sha256": PILOT_SUMMARY_SHA256,
        "steps": step_summaries,
        "artifacts": {
            "runtime": {"path": str(runtime_path), "sha256": sha256_file(runtime_path)},
            "trainer_state": {
                "path": str(state_path),
                "sha256": sha256_file(state_path),
            },
            "train_results": {
                "path": str(results_path),
                "sha256": sha256_file(results_path),
            },
            "reward": {
                "path": str(reward_path),
                "sha256": sha256_file(reward_path),
                "rows": 16,
            },
            "runtime_safety": {
                "path": str(safety_path),
                "sha256": sha256_file(safety_path),
            },
            "immediate_gate": {
                "path": str(step_gate_path),
                "sha256": sha256_file(step_gate_path),
            },
        },
        "gate_contract": {
            "steps": 2,
            "rewards_per_step": 8,
            "sampling_contract": _smoke_sampling_contract(repo_root, config),
            "step_target_directions": {
                str(step): list(directions)
                for step, directions in SMOKE_STEP_TARGET_DIRECTIONS.items()
            },
            "rewards_per_target_group": SMOKE_GENERATIONS_PER_PROMPT,
            "aggregate_target_directions": ["hold", "hike", "cut"],
            "direction_correct_nonzero_min_per_target_group": 1,
            "reward_std_gt_per_target_group": 0.0,
            "loss_finite_per_step": True,
            "grad_norm_gt_per_step": SMOKE_MIN_GRAD_NORM,
            "completion_clipped_ratio_max_per_step": 0.25,
        },
    }


def gate_smoke(repo_root: Path, *, execute: bool) -> dict[str, Any]:
    payload = seal_manifest(_smoke_gate_payload(repo_root))
    path = _smoke_gate_path(repo_root)
    if not execute:
        return {**payload, "gate_path": str(path), "create_only": True}
    _write_create_only(path, payload)
    return payload


def _verify_smoke_gate(repo_root: Path) -> dict[str, Any]:
    path = _smoke_gate_path(repo_root)
    gate = _read_json(path, label="smoke gate receipt")
    validate_manifest_integrity(gate)
    if (
        gate.get("schema_version") != GATE_SCHEMA
        or gate.get("status") != "passed"
        or gate.get("branch_id") != SMOKE.branch_id
        or gate.get("source_checkpoint_sha256") != CHECKPOINT_SHA256
        or gate.get("pilot_manifest_sha256") != PILOT_MANIFEST_SHA256
        or gate.get("pilot_summary_sha256") != PILOT_SUMMARY_SHA256
        or len(gate.get("steps") or []) != 2
    ):
        raise ContinuationError("smoke gate receipt binding drift")
    expected = seal_manifest(_smoke_gate_payload(repo_root))
    for key, value in expected.items():
        if key in {"created_at_utc", "integrity"}:
            continue
        _require_equal(gate.get(key), value, label=f"smoke gate {key}")
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "payload_sha256": gate["integrity"]["payload_sha256"],
    }


def preflight(repo_root: Path) -> dict[str, Any]:
    smoke = _validate_config(repo_root, SMOKE)
    full = _validate_config(repo_root, FULL)
    selected._activate_selection_profile(SOURCE_PROFILE)
    receipt = (repo_root / SELECTION_RECEIPT).resolve()
    if receipt.is_file() and not receipt.is_symlink():
        plan, _ = _published_selection(repo_root)
        selection_status = "published_and_verified"
    else:
        try:
            plan = selected.build_selection_plan(**_selection_kwargs(repo_root))
        except selected.SelectionMergeError as exc:
            raise ContinuationError(str(exc)) from exc
        selection_status = "ready_for_cpu_merge"
    return {
        "schema_version": f"{CONTINUATION_SCHEMA_PREFIX}-continuation-preflight-v1",
        "status": "passed",
        "selection": {
            "status": selection_status,
            "checkpoint_step": plan.checkpoint_step,
            "checkpoint_sha256": plan.checkpoint_fingerprint["sha256"],
            "destination": str(plan.destination),
            "receipt": str(plan.receipt_path),
            "pilot_manifest_sha256": plan.pilot_manifest_sha256,
            "pilot_summary_sha256": plan.pilot_summary_sha256,
        },
        "smoke": {
            "branch_id": SMOKE.branch_id,
            "config": str(_config_path(repo_root, SMOKE)),
            "config_sha256": SMOKE.config_sha256,
            "output": smoke["output_dir"],
            "authorization": str(_authorization_path(repo_root, SMOKE)),
        },
        "full": {
            "branch_id": FULL.branch_id,
            "config": str(_config_path(repo_root, FULL)),
            "config_sha256": FULL.config_sha256,
            "output": full["output_dir"],
            "authorization": str(_authorization_path(repo_root, FULL)),
            "authorization_requires_passed_smoke_gate": True,
        },
        "gpu_contract": {"device": 1, "visible_device_count": 1, "gpu0": False},
    }


def status(repo_root: Path) -> dict[str, Any]:
    result = {
        "schema_version": f"{CONTINUATION_SCHEMA_PREFIX}-continuation-status-v1",
        "selection": "not_published",
        "smoke_authorization": "missing",
        "smoke": "not_started",
        "smoke_gate": "missing",
        "full_authorization": "missing",
        "full": "not_started",
    }
    if (repo_root / SELECTION_RECEIPT).is_file():
        try:
            _published_selection(repo_root)
        except Exception:
            result["selection"] = "invalid"
        else:
            result["selection"] = "verified"
    for phase in (SMOKE, FULL):
        auth_key = f"{phase.name}_authorization"
        auth_path = _authorization_path(repo_root, phase)
        if auth_path.is_file():
            try:
                _verify_authorization(repo_root, phase)
            except Exception:
                result[auth_key] = "invalid"
            else:
                result[auth_key] = "verified"
        config = _validate_config(repo_root, phase)
        output = _resolve(repo_root, config["output_dir"], label="output")
        if output.exists():
            state = output / "trainer_state.json"
            result[phase.name] = "complete" if state.is_file() else "incomplete"
    gate_path = _smoke_gate_path(repo_root)
    if gate_path.is_file():
        try:
            _verify_smoke_gate(repo_root)
        except Exception:
            result["smoke_gate"] = "invalid"
        else:
            result["smoke_gate"] = "verified"
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "status"):
        subparsers.add_parser(command)
    for command in (
        "authorize-smoke",
        "launch-smoke",
        "gate-smoke",
        "authorize-full",
        "launch-full",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    try:
        if args.command == "preflight":
            result = preflight(repo_root)
        elif args.command == "status":
            result = status(repo_root)
        elif args.command == "authorize-smoke":
            result = authorize(repo_root, SMOKE, execute=args.execute)
        elif args.command == "authorize-full":
            result = authorize(repo_root, FULL, execute=args.execute)
        elif args.command == "gate-smoke":
            result = gate_smoke(repo_root, execute=args.execute)
        elif args.command in {"launch-smoke", "launch-full"}:
            phase = SMOKE if args.command == "launch-smoke" else FULL
            if args.execute:
                execute_training(repo_root, phase)
                raise AssertionError("os.execvpe unexpectedly returned")
            result = training_command(repo_root, phase)
        else:  # pragma: no cover
            raise ContinuationError(f"unsupported command: {args.command}")
    except (
        ContinuationError,
        selected.SelectionMergeError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
