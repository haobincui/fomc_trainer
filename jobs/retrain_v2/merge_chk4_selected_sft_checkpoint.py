"""Select and CPU-merge one evaluated chk4 SFT checkpoint, without overwrite.

The historical ``warm_fix_lr1e5_steps24_v3`` interface remains the default.
An explicit, versioned v5 profile additionally permits only the pilot-approved
hier-balanced checkpoint 24.  Both profiles derive the adapter path from the
explicit integer; callers cannot substitute the root adapter or another path.

Selection is allowed only after the complete 24-step SFT run has been sealed
and a hash-bound train-stratified pilot has passed while evaluating the exact
parent plus exact checkpoint adapter.  ``--execute`` performs a CPU-only PEFT
merge, creates an in-model attestation, verifies every merged tensor exactly,
and publishes a create-only selection receipt.  It never starts training or a
GPU process and never mutates the source checkpoint.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


LEGACY_PROFILE_NAME = "warm_fix_lr1e5_steps24_v3"
V5_PROFILE_NAME = "hier_balanced_lr1e5_steps24_v5"
PRE2009_CP38_PROFILE_NAME = "pre2009_balanced_lr1e5_steps39_v1"


@dataclass(frozen=True)
class SelectionProfile:
    """Immutable branch-specific inputs for selected-checkpoint publication."""

    name: str
    branch_id: str
    parent_config: Path
    sft_config: Path
    grpo_config: Path
    sft_release_sha256: str
    grpo_release_sha256: str
    sft_dataset_role: str
    grpo_dataset_role: str
    completed_step: int
    eval_steps: int
    allowed_checkpoint_steps: frozenset[int]
    checkpoint_keep_every_n_steps: int
    checkpoint_keep_steps: tuple[int, ...]
    fixed_schedule_runtime: bool = False
    fixed_schedule_schema: str | None = None
    fixed_schedule_type: str | None = None
    fixed_schedule_sha256: str | None = None
    fixed_schedule_rows: int | None = None
    fixed_train_sha256: str | None = None
    fixed_train_rows: int | None = None
    required_pilot_manifest_sha256: str | None = None
    required_pilot_summary_sha256: str | None = None
    required_pilot_results_sha256: str | None = None
    required_pilot_launch_sha256: str | None = None
    required_checkpoint_sha256: str | None = None
    required_adapter_weights_sha256: str | None = None
    base_artifact_id: str = "chk4-steps24-parent"

    @property
    def run_root(self) -> Path:
        return Path(f"output/training/retrain_v2/{self.branch_id}")


SELECTION_PROFILES = {
    LEGACY_PROFILE_NAME: SelectionProfile(
        name=LEGACY_PROFILE_NAME,
        branch_id=("chk4_from_chk1_cp200_sft_grpo_warm_fix_lr1e5_steps24_v3_20260811"),
        parent_config=Path(
            "configs/retrain_v2/"
            "chk1_cp200_parent_reuse_for_chk4_warm_fix_lr1e5_steps24_v3_20260811.yaml"
        ),
        sft_config=Path(
            "configs/retrain_v2/"
            "chk4_decision_sft_from_chk1_cp200_warm_fix_lr1e5_steps24_v3_20260811.yaml"
        ),
        grpo_config=Path(
            "configs/retrain_v2/"
            "chk4_decision_grpo_from_sft_warm_fix_lr1e5_steps24_v3_20260811.yaml"
        ),
        sft_release_sha256=(
            "8b05dce09bcb0a0b27ee240da1e5d730be93921ce19e650a3b3e8b2689adb893"
        ),
        grpo_release_sha256=(
            "8b05dce09bcb0a0b27ee240da1e5d730be93921ce19e650a3b3e8b2689adb893"
        ),
        sft_dataset_role="decision_sft",
        grpo_dataset_role="decision_grpo",
        completed_step=24,
        eval_steps=4,
        allowed_checkpoint_steps=frozenset({10, 20, 24}),
        checkpoint_keep_every_n_steps=10,
        checkpoint_keep_steps=(),
    ),
    V5_PROFILE_NAME: SelectionProfile(
        name=V5_PROFILE_NAME,
        branch_id=(
            "chk4_from_chk1_cp200_sft_grpo_hier_balanced_lr1e5_steps24_v5_20260811"
        ),
        parent_config=Path(
            "configs/retrain_v2/"
            "chk1_cp200_parent_reuse_for_chk4_hier_balanced_lr1e5_steps24_v5_20260811.yaml"
        ),
        sft_config=Path(
            "configs/retrain_v2/"
            "chk4_decision_sft_hier_balanced_lr1e5_steps24_v5_20260811.yaml"
        ),
        grpo_config=Path(
            "configs/retrain_v2/"
            "chk4_decision_grpo_from_hier_balanced_sft_lr1e5_steps24_v5_20260811.yaml"
        ),
        sft_release_sha256=(
            "18c6f47ed0a085f7aef39d599e4baa30b34e5b560f3a594e7d5e3cf78bfde255"
        ),
        grpo_release_sha256=(
            "8b05dce09bcb0a0b27ee240da1e5d730be93921ce19e650a3b3e8b2689adb893"
        ),
        sft_dataset_role="decision_sft_hier_balanced",
        grpo_dataset_role="decision_grpo",
        completed_step=24,
        eval_steps=4,
        allowed_checkpoint_steps=frozenset({24}),
        checkpoint_keep_every_n_steps=0,
        checkpoint_keep_steps=(12, 16, 20, 24),
        fixed_schedule_runtime=True,
        fixed_schedule_schema="manifest-fixed-schedule-runtime-v1",
        fixed_schedule_type="manifest_fixed_schedule_v1",
        fixed_schedule_sha256=(
            "d467503e8890ab20010026675f41b6b45bbb9d16502562be9038210d9d0e6504"
        ),
        fixed_schedule_rows=192,
        fixed_train_sha256=(
            "8ded0ec903414e1efdc357b153e399a4fa985490b36224b101d6b6417309016e"
        ),
        fixed_train_rows=192,
        required_pilot_manifest_sha256=(
            "bd5aac9ac26909b6ce50de35506544c9050cc9c3a31c39a318216daf9408ec4c"
        ),
        required_pilot_summary_sha256=(
            "0a2064f11b9c88a0598a62da311e61176415277aa222b048c4daf0e1d0013d1e"
        ),
    ),
    PRE2009_CP38_PROFILE_NAME: SelectionProfile(
        name=PRE2009_CP38_PROFILE_NAME,
        branch_id=(
            "chk4_from_chk1_cp200_sft_grpo_pre2009_balanced_lr1e5_steps39_v1_20260811"
        ),
        parent_config=Path(
            "configs/retrain_v2/"
            "chk1_cp200_parent_reuse_for_chk4_pre2009_balanced_"
            "lr1e5_steps39_v1_20260811.yaml"
        ),
        sft_config=Path(
            "configs/retrain_v2/"
            "chk4_decision_sft_pre2009_balanced_lr1e5_steps39_v1_20260811.yaml"
        ),
        grpo_config=Path(
            "configs/retrain_v2/"
            "chk4_decision_grpo_from_pre2009_balanced_sft_"
            "lr1e5_steps39_v1_20260811.yaml"
        ),
        sft_release_sha256=(
            "5141e00569b94e4af96f030f46176e9cd409199a6f26ac1c63a2ce335fe1e2d6"
        ),
        grpo_release_sha256=(
            "5141e00569b94e4af96f030f46176e9cd409199a6f26ac1c63a2ce335fe1e2d6"
        ),
        sft_dataset_role="decision_sft_pre2009_balanced",
        grpo_dataset_role="decision_grpo_pre2009_balanced",
        completed_step=39,
        eval_steps=13,
        allowed_checkpoint_steps=frozenset({38}),
        checkpoint_keep_every_n_steps=0,
        checkpoint_keep_steps=(13, 26, 39),
        fixed_schedule_runtime=True,
        fixed_schedule_schema="manifest-fixed-schedule-runtime-v2",
        fixed_schedule_type="manifest_fixed_schedule_v2",
        fixed_schedule_sha256=(
            "9020fa26b58ca1539179b99611d2d01a01fa79bd6db07614df44d22e74fbd14b"
        ),
        fixed_schedule_rows=312,
        fixed_train_sha256=(
            "d034bb0c40cf3945f5c5ab984d4e3b1bc7f2b0b4bbe4b4d32bd0b955ed1b89e9"
        ),
        fixed_train_rows=312,
        required_pilot_manifest_sha256=(
            "99853a20f246684b0e03a90087af5dcb3ca23732bf5db06fd5433b09516611a7"
        ),
        required_pilot_summary_sha256=(
            "2b33278c9b96b27954e710c0e705b3dc929ade2ecf906d5296e572cc0ba75a0b"
        ),
        required_pilot_results_sha256=(
            "86b0085e27f21d3fb1ba3b88893309b89f8e2363148c5a576d20fe0178b590a9"
        ),
        required_pilot_launch_sha256=(
            "6f193bd91a50c0f4eeaf57c3b1492d1e6ee108085eae33a9195e69107f14ed3f"
        ),
        required_checkpoint_sha256=(
            "339671e06053f0dc80d4dfc6d006500e9a9fccba1945aa5ff770f05f755a2a4b"
        ),
        required_adapter_weights_sha256=(
            "513cb7bc91136f9a9b5c7d83c93d8808cdd583dc0fb293ad3af452428354b09b"
        ),
        base_artifact_id="chk4-pre2009-steps39-parent",
    ),
}


def _activate_selection_profile(name: str) -> SelectionProfile:
    """Select a process-local profile while retaining the legacy default."""

    try:
        profile = SELECTION_PROFILES[name]
    except KeyError as exc:
        raise SelectionMergeError(f"unsupported selection profile: {name}") from exc
    global ACTIVE_SELECTION_PROFILE
    global PROFILE_NAME, BRANCH_ID, RUN_ROOT_RELATIVE
    global PARENT_CONFIG_RELATIVE, SFT_CONFIG_RELATIVE, GRPO_CONFIG_RELATIVE
    global SFT_RELEASE_MANIFEST_SHA256, RELEASE_MANIFEST_SHA256
    global ALLOWED_CHECKPOINT_STEPS
    ACTIVE_SELECTION_PROFILE = profile
    PROFILE_NAME = profile.name
    BRANCH_ID = profile.branch_id
    RUN_ROOT_RELATIVE = profile.run_root
    PARENT_CONFIG_RELATIVE = profile.parent_config
    SFT_CONFIG_RELATIVE = profile.sft_config
    GRPO_CONFIG_RELATIVE = profile.grpo_config
    SFT_RELEASE_MANIFEST_SHA256 = profile.sft_release_sha256
    # Kept as the GRPO release alias for backwards compatibility and pilot code.
    RELEASE_MANIFEST_SHA256 = profile.grpo_release_sha256
    ALLOWED_CHECKPOINT_STEPS = profile.allowed_checkpoint_steps
    return profile


ACTIVE_SELECTION_PROFILE: SelectionProfile
PROFILE_NAME: str
BRANCH_ID: str
RUN_ROOT_RELATIVE: Path
PARENT_CONFIG_RELATIVE: Path
SFT_CONFIG_RELATIVE: Path
GRPO_CONFIG_RELATIVE: Path
SFT_RELEASE_MANIFEST_SHA256: str
RELEASE_MANIFEST_SHA256: str
ALLOWED_CHECKPOINT_STEPS: frozenset[int]
_activate_selection_profile(LEGACY_PROFILE_NAME)
COMPLETED_STEP = 24
SELECTION_SCHEMA = "chk4-selected-sft-checkpoint-merge-v1"
MERGE_BINDING_SCHEMA = "chk4-selected-sft-checkpoint-merge-binding-v1"
PILOT_MANIFEST_SCHEMA = "chk4-decision-grpo-stratified-samples-v1"
PILOT_SUMMARY_SCHEMA = "chk4-decision-grpo-stratified-summary-v1"
EXACT_CONCLUSION = "exact_base_plus_adapter_merge_verified"
_SHA_RE = __import__("re").compile(r"[0-9a-f]{64}")


class SelectionMergeError(RuntimeError):
    """A selected-checkpoint merge input or output is untrustworthy."""


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


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SelectionMergeError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectionMergeError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SelectionMergeError(f"{label} must contain a JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise SelectionMergeError(f"{label} is missing or unsafe: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    raise SelectionMergeError(
                        f"{label} contains a blank row at line {line_number}"
                    )
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise SelectionMergeError(
                        f"{label} row {line_number} is not an object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectionMergeError(f"{label} is invalid JSONL: {path}") from exc
    return rows


def _read_yaml(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SelectionMergeError(f"{label} is missing or unsafe: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise SelectionMergeError(f"{label} is invalid YAML: {path}") from exc
    if not isinstance(value, dict):
        raise SelectionMergeError(f"{label} must contain a YAML mapping: {path}")
    return value


def _validate_sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise SelectionMergeError(f"{label} must be a lowercase SHA-256")
    return value


def _require_hash(path: Path, expected: object, *, label: str) -> str:
    expected_sha = _validate_sha(expected, label=f"{label} SHA-256")
    if not path.is_file() or path.is_symlink():
        raise SelectionMergeError(f"{label} is missing or unsafe: {path}")
    observed = sha256_file(path)
    if observed != expected_sha:
        raise SelectionMergeError(
            f"{label} hash drift: expected={expected_sha}, observed={observed}"
        )
    return observed


def _resolve(repo_root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SelectionMergeError(f"{label} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _require_equal(observed: Any, expected: Any, *, label: str) -> None:
    if observed != expected:
        raise SelectionMergeError(
            f"{label} drift: expected {expected!r}, observed {observed!r}"
        )


def _require_finite(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise SelectionMergeError(f"{label} must be finite")
    return float(value)


@dataclass(frozen=True)
class SelectionPlan:
    repo_root: Path
    checkpoint_step: int
    run_root: Path
    sft_config_path: Path
    grpo_config_path: Path
    training_output: Path
    checkpoint: Path
    base_model: Path
    destination: Path
    exact_evidence_path: Path
    receipt_path: Path
    sft_config_sha256: str
    grpo_config_sha256: str
    runtime_path: Path
    runtime_sha256: str
    trainer_state_path: Path
    trainer_state_sha256: str
    train_results_path: Path
    train_results_sha256: str
    base_fingerprint: Mapping[str, Any]
    checkpoint_fingerprint: Mapping[str, Any]
    parent_lineage: Mapping[str, Any]
    pilot_manifest_path: Path
    pilot_manifest_sha256: str
    pilot_summary_path: Path
    pilot_summary_sha256: str
    pilot_results: Mapping[str, Any]
    pilot_launch: Mapping[str, Any] | None

    def public(self) -> dict[str, Any]:
        return {
            "schema_version": SELECTION_SCHEMA,
            "status": "ready",
            "profile": PROFILE_NAME,
            "branch_id": BRANCH_ID,
            "checkpoint_step": self.checkpoint_step,
            "source_checkpoint": dict(self.checkpoint_fingerprint),
            "root_adapter_is_source": False,
            "base_model": dict(self.base_fingerprint),
            "parent_lineage": dict(self.parent_lineage),
            "destination": str(self.destination),
            "receipt": str(self.receipt_path),
            "exact_evidence": str(self.exact_evidence_path),
            "training": {
                "config": {
                    "path": str(self.sft_config_path),
                    "sha256": self.sft_config_sha256,
                },
                "runtime": {
                    "path": str(self.runtime_path),
                    "sha256": self.runtime_sha256,
                },
                "trainer_state": {
                    "path": str(self.trainer_state_path),
                    "sha256": self.trainer_state_sha256,
                    "global_step": ACTIVE_SELECTION_PROFILE.completed_step,
                },
                "train_results": {
                    "path": str(self.train_results_path),
                    "sha256": self.train_results_sha256,
                },
            },
            "pilot": {
                "manifest": {
                    "path": str(self.pilot_manifest_path),
                    "sha256": self.pilot_manifest_sha256,
                },
                "summary": {
                    "path": str(self.pilot_summary_path),
                    "sha256": self.pilot_summary_sha256,
                },
                "results": dict(self.pilot_results),
                **(
                    {"launch": dict(self.pilot_launch)}
                    if self.pilot_launch is not None
                    else {}
                ),
            },
            "execution": {
                "device": "cpu",
                "merge": "peft_merge_and_unload",
                "exact_tensor_verification": True,
                "create_only": True,
            },
        }


def _validate_configs(
    repo_root: Path,
) -> tuple[Path, dict[str, Any], str, Path, dict[str, Any], str]:
    sft_path = (repo_root / SFT_CONFIG_RELATIVE).resolve()
    grpo_path = (repo_root / GRPO_CONFIG_RELATIVE).resolve()
    sft = _read_yaml(sft_path, label="steps24 SFT config")
    grpo = _read_yaml(grpo_path, label="steps24 GRPO config")
    run_root = (repo_root / RUN_ROOT_RELATIVE).resolve()
    expected_sft_output = run_root / "adapters/chk4_sft"
    expected_sft_merge = run_root / "merged/chk4_sft"
    for key, expected in {
        "learning_rate": 1.0e-5,
        "max_steps": ACTIVE_SELECTION_PROFILE.completed_step,
        "warmup_steps": 2,
        "eval_steps": ACTIVE_SELECTION_PROFILE.eval_steps,
        "gradient_accumulation_steps": 8,
        "save_steps": 1,
        "save_total_limit": None,
        "checkpoint_keep_last": 3,
        "checkpoint_keep_every_n_steps": (
            ACTIVE_SELECTION_PROFILE.checkpoint_keep_every_n_steps
        ),
        "overwrite_output_dir": False,
    }.items():
        _require_equal(sft.get(key), expected, label=f"SFT config {key}")
    _require_equal(
        tuple(sft.get("checkpoint_keep_steps") or ()),
        ACTIVE_SELECTION_PROFILE.checkpoint_keep_steps,
        label="SFT config checkpoint_keep_steps",
    )
    _require_equal(
        sft.get("dataset_chk4_role"),
        ACTIVE_SELECTION_PROFILE.sft_dataset_role,
        label="SFT config dataset role",
    )
    _require_equal(
        _resolve(repo_root, sft.get("output_dir"), label="SFT output"),
        expected_sft_output,
        label="SFT output path",
    )
    _require_equal(
        _resolve(repo_root, sft.get("peft_merged_model_path"), label="SFT merge"),
        expected_sft_merge,
        label="SFT root merge path",
    )
    for key, expected in {
        "dataset_chk4_role": ACTIVE_SELECTION_PROFILE.grpo_dataset_role,
        "reward_funcs": ["decision_dense_v3"],
        "max_completion_length": 1024,
        "num_generations": 4,
        "temperature": 0.7,
        "top_p": 0.9,
    }.items():
        _require_equal(grpo.get(key), expected, label=f"GRPO config {key}")
    _require_equal(
        _resolve(repo_root, grpo.get("model_name_or_path"), label="GRPO parent"),
        expected_sft_merge,
        label="GRPO/SFT parent chain",
    )
    _require_equal(
        sft.get("dataset_chk4_release_manifest_sha256"),
        SFT_RELEASE_MANIFEST_SHA256,
        label="SFT release SHA",
    )
    _require_equal(
        grpo.get("dataset_chk4_release_manifest_sha256"),
        RELEASE_MANIFEST_SHA256,
        label="GRPO release SHA",
    )
    return (
        sft_path,
        sft,
        sha256_file(sft_path),
        grpo_path,
        grpo,
        sha256_file(grpo_path),
    )


def _validate_complete_training(
    *,
    repo_root: Path,
    run_root: Path,
    config_path: Path,
    config: Mapping[str, Any],
    config_sha: str,
) -> tuple[Path, Path, str, Path, str, Path, str]:
    output = _resolve(repo_root, config.get("output_dir"), label="SFT output")
    if output != run_root / "adapters/chk4_sft":
        raise SelectionMergeError("SFT output is outside the selected branch")
    if not output.is_dir() or output.is_symlink():
        raise SelectionMergeError("completed SFT output is missing or unsafe")
    runtime_path = output / "resolved_runtime_config.json"
    state_path = output / "trainer_state.json"
    results_path = output / "train_results.json"
    for path, label in (
        (runtime_path, "SFT runtime"),
        (state_path, "root trainer state"),
        (results_path, "SFT train results"),
        (output / "adapter_config.json", "root adapter config"),
        (output / "adapter_model.safetensors", "root adapter weights"),
    ):
        if not path.is_file() or path.is_symlink():
            raise SelectionMergeError(f"completed training is missing {label}: {path}")

    runtime = _read_json(runtime_path, label="SFT runtime")
    branch = runtime.get("chk4_standalone_branch")
    if not isinstance(branch, Mapping):
        raise SelectionMergeError("SFT runtime has no branch binding")
    _require_equal(branch.get("branch_id"), BRANCH_ID, label="runtime branch")
    _require_equal(branch.get("profile"), PROFILE_NAME, label="runtime profile")
    _require_equal(branch.get("stage"), "decision_sft", label="runtime stage")
    runtime_config = branch.get("config")
    if not isinstance(runtime_config, Mapping):
        raise SelectionMergeError("SFT runtime config binding is missing")
    _require_equal(
        Path(str(runtime_config.get("path"))).resolve(),
        config_path,
        label="runtime config path",
    )
    _require_equal(runtime_config.get("sha256"), config_sha, label="runtime config SHA")
    training = runtime.get("training")
    if not isinstance(training, Mapping):
        raise SelectionMergeError("SFT runtime training binding is missing")
    for key, expected in {
        "learning_rate": 1.0e-5,
        "max_steps": ACTIVE_SELECTION_PROFILE.completed_step,
        "warmup_steps": 2,
        "gradient_accumulation_steps": 8,
        "per_device_train_batch_size": 1,
    }.items():
        _require_equal(training.get(key), expected, label=f"runtime training {key}")
    _require_equal(
        _resolve(repo_root, training.get("output_dir"), label="runtime output"),
        output,
        label="runtime output path",
    )
    model = runtime.get("model")
    if not isinstance(model, Mapping):
        raise SelectionMergeError("SFT runtime model binding is missing")
    base_model = _resolve(
        repo_root, model.get("model_name_or_path"), label="runtime parent model"
    )
    _require_equal(
        base_model,
        _resolve(repo_root, config.get("model_name_or_path"), label="SFT parent"),
        label="runtime parent model",
    )
    dataset = runtime.get("dataset")
    chk4 = dataset.get("chk4_decision") if isinstance(dataset, Mapping) else None
    scope = chk4.get("scope") if isinstance(chk4, Mapping) else None
    release = chk4.get("release_manifest") if isinstance(chk4, Mapping) else None
    if (
        not isinstance(scope, Mapping)
        or scope.get("role") != ACTIVE_SELECTION_PROFILE.sft_dataset_role
        or not isinstance(release, Mapping)
        or release.get("sha256") != SFT_RELEASE_MANIFEST_SHA256
    ):
        raise SelectionMergeError("SFT runtime release binding drift")
    if ACTIVE_SELECTION_PROFILE.fixed_schedule_runtime:
        sampler = runtime.get("training_sampler")
        expected_sampler = {
            "schema_version": ACTIVE_SELECTION_PROFILE.fixed_schedule_schema,
            "type": ACTIVE_SELECTION_PROFILE.fixed_schedule_type,
        }
        if not isinstance(sampler, Mapping) or any(
            sampler.get(key) != value for key, value in expected_sampler.items()
        ):
            raise SelectionMergeError("SFT fixed-schedule runtime binding is missing")
        sampler_release = sampler.get("release_manifest")
        schedule = sampler.get("schedule")
        train_file = sampler.get("train_file")
        source = sampler.get("sampler_source")
        sampler_runtime = sampler.get("runtime")
        if (
            not isinstance(sampler_release, Mapping)
            or sampler_release.get("sha256") != SFT_RELEASE_MANIFEST_SHA256
            or not isinstance(schedule, Mapping)
            or schedule.get("sha256") != ACTIVE_SELECTION_PROFILE.fixed_schedule_sha256
            or schedule.get("rows") != ACTIVE_SELECTION_PROFILE.fixed_schedule_rows
            or schedule.get("order_is_authoritative") is not True
            or not isinstance(train_file, Mapping)
            or train_file.get("sha256") != ACTIVE_SELECTION_PROFILE.fixed_train_sha256
            or train_file.get("rows") != ACTIVE_SELECTION_PROFILE.fixed_train_rows
            or not isinstance(source, Mapping)
            or source.get("sha256")
            != sha256_file(repo_root / "src/open_r1/trainer/fixed_schedule_sampler.py")
            or not isinstance(sampler_runtime, Mapping)
            or sampler_runtime
            != {
                "world_size": 1,
                "visible_gpus": 1,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "effective_batch_size": 8,
                "max_steps": ACTIVE_SELECTION_PROFILE.completed_step,
                "shuffle_dataset": False,
                "trl_shuffle_dataset": False,
                "secondary_shuffle": False,
                "use_liger_kernel": False,
            }
        ):
            raise SelectionMergeError("SFT fixed-schedule runtime binding drift")

    state = _read_json(state_path, label="root trainer state")
    _require_equal(
        state.get("global_step"),
        ACTIVE_SELECTION_PROFILE.completed_step,
        label="completed step",
    )
    _require_equal(
        state.get("max_steps"),
        ACTIVE_SELECTION_PROFILE.completed_step,
        label="configured end step",
    )
    _require_finite(state.get("epoch"), label="completed epoch")
    results = _read_json(results_path, label="SFT train results")
    _require_finite(results.get("train_loss"), label="SFT train loss")
    return (
        output,
        runtime_path,
        sha256_file(runtime_path),
        state_path,
        sha256_file(state_path),
        results_path,
        sha256_file(results_path),
    )


def _validate_parent_lineage(
    *,
    repo_root: Path,
    run_root: Path,
    base_fingerprint: Mapping[str, Any],
    sft_config_path: Path,
    sft_config_sha: str,
    grpo_config_path: Path,
    grpo_config_sha: str,
) -> Mapping[str, Any]:
    parent_config_path = (repo_root / PARENT_CONFIG_RELATIVE).resolve()
    if not parent_config_path.is_file() or parent_config_path.is_symlink():
        raise SelectionMergeError("steps24 parent config is missing or unsafe")
    parent_config_sha = sha256_file(parent_config_path)
    authorization_path = run_root / "receipts/chk1_cp200_to_chk4_authorization.json"
    parent_stage_path = run_root / "receipts/parent_stage.json"
    authorization = _read_json(authorization_path, label="steps24 authorization")
    authorization_payload = dict(authorization)
    authorization_sha = authorization_payload.pop("authorization_sha256", None)
    if (
        authorization.get("schema_version") != "chk1-cp200-to-chk4-authorization-v1"
        or authorization.get("status") != "authorized"
        or authorization.get("authorization_basis") != "explicit_user_instruction"
        or authorization.get("profile") != PROFILE_NAME
        or authorization.get("branch_id") != BRANCH_ID
        or authorization_sha != _sha256_text(_canonical_json(authorization_payload))
    ):
        raise SelectionMergeError("steps24 authorization is invalid")
    bindings = authorization.get("bindings")
    configs = bindings.get("configs") if isinstance(bindings, Mapping) else None
    expected_configs = {
        "parent": {"path": str(parent_config_path), "sha256": parent_config_sha},
        "sft": {"path": str(sft_config_path), "sha256": sft_config_sha},
        "grpo": {"path": str(grpo_config_path), "sha256": grpo_config_sha},
    }
    reused = bindings.get("reused_parent") if isinstance(bindings, Mapping) else None
    release = (
        bindings.get("release_manifest") if isinstance(bindings, Mapping) else None
    )
    grpo_release = (
        bindings.get("grpo_release_manifest") if isinstance(bindings, Mapping) else None
    )
    if (
        configs != expected_configs
        or Path(str(bindings.get("run_root"))).resolve() != run_root
        or not isinstance(reused, Mapping)
        or reused.get("merged_artifact") != dict(base_fingerprint)
        or not isinstance(release, Mapping)
        or release.get("sha256") != SFT_RELEASE_MANIFEST_SHA256
        or (
            SFT_RELEASE_MANIFEST_SHA256 != RELEASE_MANIFEST_SHA256
            and (
                not isinstance(grpo_release, Mapping)
                or grpo_release.get("sha256") != RELEASE_MANIFEST_SHA256
            )
        )
    ):
        raise SelectionMergeError("steps24 authorization lineage binding drift")

    parent_stage = _read_json(parent_stage_path, label="steps24 parent stage")
    unsigned_parent_stage = dict(parent_stage)
    parent_stage_payload_sha = unsigned_parent_stage.pop("receipt_sha256", None)
    authorization_descriptor = parent_stage.get("authorization")
    if (
        parent_stage.get("schema_version") != "chk4-reused-parent-stage-receipt-v1"
        or parent_stage.get("status") != "passed"
        or parent_stage.get("stage") != "parent_reuse"
        or parent_stage.get("profile") != PROFILE_NAME
        or parent_stage.get("branch_id") != BRANCH_ID
        or parent_stage_payload_sha
        != _sha256_text(_canonical_json(unsigned_parent_stage))
        or parent_stage.get("config") != expected_configs["parent"]
        or parent_stage.get("merged_artifact") != dict(base_fingerprint)
        or parent_stage.get("reused_parent") != reused
        or not isinstance(authorization_descriptor, Mapping)
        or Path(str(authorization_descriptor.get("path"))).resolve()
        != authorization_path
        or authorization_descriptor.get("file_sha256")
        != sha256_file(authorization_path)
        or authorization_descriptor.get("payload_sha256") != authorization_sha
    ):
        raise SelectionMergeError("steps24 parent-stage receipt binding drift")

    source_exact_descriptor = reused.get("source_exact_merge_evidence")
    source_attestation_descriptor = reused.get("source_merge_attestation")
    source_parent_descriptor = reused.get("source_parent_stage")
    source_authorization_descriptor = reused.get("source_authorization")
    for descriptor, label in (
        (source_parent_descriptor, "source parent stage"),
        (source_authorization_descriptor, "source authorization"),
        (source_exact_descriptor, "source exact merge evidence"),
        (source_attestation_descriptor, "source merge attestation"),
    ):
        if not isinstance(descriptor, Mapping):
            raise SelectionMergeError(f"reused parent {label} binding is missing")
        _require_hash(
            Path(str(descriptor.get("path"))).resolve(),
            descriptor.get("file_sha256"),
            label=f"reused parent {label}",
        )

    source_exact_path = Path(str(source_exact_descriptor["path"])).resolve()
    source_exact = _read_json(source_exact_path, label="source exact merge evidence")
    validate_manifest_integrity(
        source_exact,
        expected_payload_sha256=source_exact_descriptor.get("payload_sha256"),
    )
    source_exact_sources = source_exact.get("sources")
    if (
        source_exact.get("conclusion") != EXACT_CONCLUSION
        or not isinstance(source_exact_sources, Mapping)
        or source_exact_sources.get("merged_model") != dict(base_fingerprint)
    ):
        raise SelectionMergeError("source parent exact-merge proof drift")
    source_attestation_path = Path(str(source_attestation_descriptor["path"])).resolve()
    source_attestation = _read_json(
        source_attestation_path, label="source merge attestation"
    )
    source_attestation_binding = source_attestation.get("binding")
    if not isinstance(source_attestation_binding, Mapping) or _sha256_text(
        _canonical_json(source_attestation_binding)
    ) != source_attestation_descriptor.get("binding_sha256"):
        raise SelectionMergeError("source parent merge-attestation binding drift")
    try:
        from jobs.retrain_v2.merge_attestation import verify_merge_attestation

        verify_merge_attestation(
            source_attestation_path.parent,
            expected_binding=source_attestation_binding,
        )
    except Exception as exc:
        raise SelectionMergeError("source parent merge attestation is invalid") from exc
    return {
        "authorization": {
            "path": str(authorization_path),
            "file_sha256": sha256_file(authorization_path),
            "payload_sha256": authorization_sha,
        },
        "parent_stage": {
            "path": str(parent_stage_path),
            "file_sha256": sha256_file(parent_stage_path),
            "receipt_sha256": parent_stage_payload_sha,
        },
        "source_exact_merge_evidence": dict(source_exact_descriptor),
        "source_merge_attestation": dict(source_attestation_descriptor),
        "verified_parent": dict(base_fingerprint),
    }


def _validate_checkpoint(
    *,
    repo_root: Path,
    output: Path,
    step: int,
    base_model: Path,
    config: Mapping[str, Any],
) -> tuple[Path, Mapping[str, Any]]:
    if isinstance(step, bool) or step not in ALLOWED_CHECKPOINT_STEPS:
        allowed = ", ".join(str(value) for value in sorted(ALLOWED_CHECKPOINT_STEPS))
        raise SelectionMergeError(f"checkpoint step must be one of: {allowed}")
    checkpoint = output / f"checkpoint-{step}"
    if checkpoint.parent != output or checkpoint.name != f"checkpoint-{step}":
        raise SelectionMergeError("checkpoint path derivation drift")
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise SelectionMergeError(
            f"selected checkpoint is missing or unsafe: {checkpoint}"
        )
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    }
    missing = sorted(
        name
        for name in required
        if not (checkpoint / name).is_file() or (checkpoint / name).is_symlink()
    )
    if missing:
        raise SelectionMergeError(
            f"selected checkpoint-{step} is incomplete; missing {missing}"
        )
    if not any(
        path.is_file() and not path.is_symlink()
        for path in (checkpoint / "rng_state.pth", checkpoint / "rng_state_0.pth")
    ):
        raise SelectionMergeError(f"selected checkpoint-{step} has no RNG state")
    state = _read_json(checkpoint / "trainer_state.json", label="checkpoint state")
    _require_equal(state.get("global_step"), step, label="checkpoint global_step")
    _require_equal(
        state.get("max_steps"),
        ACTIVE_SELECTION_PROFILE.completed_step,
        label="checkpoint max_steps",
    )
    _require_finite(state.get("epoch"), label="checkpoint epoch")
    adapter_config = _read_json(
        checkpoint / "adapter_config.json", label="checkpoint adapter config"
    )
    adapter_parent = _resolve(
        repo_root,
        adapter_config.get("base_model_name_or_path"),
        label="checkpoint adapter parent",
    )
    _require_equal(adapter_parent, base_model, label="checkpoint parent")
    for key, expected in {
        "r": config.get("peft_r"),
        "lora_alpha": config.get("peft_lora_alpha"),
        "lora_dropout": config.get("peft_lora_dropout"),
        "bias": config.get("peft_bias"),
    }.items():
        _require_equal(adapter_config.get(key), expected, label=f"adapter {key}")
    checkpoint_fingerprint = fingerprint_artifact_path(checkpoint)
    required_checkpoint_sha = ACTIVE_SELECTION_PROFILE.required_checkpoint_sha256
    if (
        required_checkpoint_sha is not None
        and checkpoint_fingerprint.get("sha256") != required_checkpoint_sha
    ):
        raise SelectionMergeError("selected checkpoint fingerprint drift")
    required_weights_sha = ACTIVE_SELECTION_PROFILE.required_adapter_weights_sha256
    if required_weights_sha is not None:
        _require_hash(
            checkpoint / "adapter_model.safetensors",
            required_weights_sha,
            label="selected checkpoint adapter weights",
        )
    return checkpoint, checkpoint_fingerprint


def _validate_pilot(
    *,
    repo_root: Path,
    manifest_path: Path,
    manifest_sha: str,
    summary_path: Path,
    summary_sha: str,
    grpo_config_path: Path,
    grpo_config_sha: str,
    base_fingerprint: Mapping[str, Any],
    checkpoint_fingerprint: Mapping[str, Any],
) -> Mapping[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    summary_path = summary_path.expanduser().resolve()
    observed_manifest_sha = _require_hash(
        manifest_path, manifest_sha, label="pilot sample manifest"
    )
    observed_summary_sha = _require_hash(
        summary_path, summary_sha, label="pilot summary"
    )
    manifest = _read_json(manifest_path, label="pilot sample manifest")
    if manifest.get("schema_version") != PILOT_MANIFEST_SCHEMA:
        raise SelectionMergeError("unsupported pilot sample manifest schema")
    dataset = manifest.get("dataset")
    selection = manifest.get("selection")
    samples = manifest.get("samples")
    if (
        not isinstance(dataset, Mapping)
        or dataset.get("role") != ACTIVE_SELECTION_PROFILE.grpo_dataset_role
        or dataset.get("split") != "train"
        or not isinstance(selection, Mapping)
        or selection.get("source_split") != "train"
        or not isinstance(samples, list)
        or len(samples) != 3
        or [item.get("direction") for item in samples if isinstance(item, Mapping)]
        != ["hold", "hike", "cut"]
        or any(
            not isinstance(item, Mapping) or item.get("split") != "train"
            for item in samples
        )
    ):
        raise SelectionMergeError(
            "pilot manifest is not the three-direction train pilot"
        )
    prompt_contract = manifest.get("prompt_contract")
    if not isinstance(prompt_contract, Mapping):
        raise SelectionMergeError("pilot prompt contract is missing")
    _require_equal(
        Path(str(prompt_contract.get("training_config"))).resolve(),
        grpo_config_path,
        label="pilot GRPO config path",
    )
    _require_equal(
        prompt_contract.get("training_config_sha256"),
        grpo_config_sha,
        label="pilot GRPO config SHA",
    )
    release = manifest.get("release")
    tokenizer = manifest.get("tokenizer")
    if (
        not isinstance(release, Mapping)
        or release.get("manifest_sha256") != RELEASE_MANIFEST_SHA256
        or not isinstance(tokenizer, Mapping)
        or _SHA_RE.fullmatch(str(tokenizer.get("bundle_sha256") or "")) is None
    ):
        raise SelectionMergeError("pilot release/tokenizer binding drift")

    summary = _read_json(summary_path, label="pilot summary")
    if (
        summary.get("schema_version") != PILOT_SUMMARY_SCHEMA
        or summary.get("status") != "complete"
        or summary.get("quality_status") != "passed"
        or summary.get("cases") != 15
        or summary.get("train_groups") != 3
    ):
        raise SelectionMergeError("stratified pilot did not pass completely")
    quality = summary.get("quality_gate")
    expected_contract = {
        "each_direction_sampled_nonzero_min": 1,
        "each_direction_sampled_direction_correct_min": 1,
        "each_direction_sampled_reward_std_gt": 0.0,
        "overall_cap_rate_max": 0.25,
        "overall_boundary_rate_min": 0.75,
        "strict_periodic_tail_max": 0,
    }
    if (
        not isinstance(quality, Mapping)
        or quality.get("contract") != expected_contract
        or quality.get("reasons") != []
    ):
        raise SelectionMergeError("pilot quality gate contract did not pass")
    directions = summary.get("directions")
    if not isinstance(directions, Mapping) or set(directions) != {
        "hold",
        "hike",
        "cut",
    }:
        raise SelectionMergeError("pilot direction summary is incomplete")
    for direction in ("hold", "hike", "cut"):
        sampled = directions[direction].get("sampled")
        if (
            not isinstance(sampled, Mapping)
            or sampled.get("cases") != 4
            or int(sampled.get("nonzero_count", 0)) < 1
            or int(sampled.get("direction_correct_count", 0)) < 1
            or bool(sampled.get("zero_std"))
        ):
            raise SelectionMergeError(f"pilot {direction} sampled gate did not pass")
    overall = summary.get("overall")
    if (
        not isinstance(overall, Mapping)
        or float(overall.get("cap_rate", 1.0)) > 0.25
        or float(overall.get("boundary_rate", 0.0)) < 0.75
        or int(overall.get("strict_periodic_tail_count", 1)) != 0
    ):
        raise SelectionMergeError("pilot overall delivery gate did not pass")
    provenance = summary.get("provenance")
    if not isinstance(provenance, Mapping):
        raise SelectionMergeError("pilot provenance is missing")
    for key, expected in {
        "sample_manifest_sha256": observed_manifest_sha,
        "release_manifest_sha256": RELEASE_MANIFEST_SHA256,
        "training_config_sha256": grpo_config_sha,
        "tokenizer_bundle_sha256": tokenizer["bundle_sha256"],
    }.items():
        _require_equal(provenance.get(key), expected, label=f"pilot provenance {key}")
    _require_equal(
        provenance.get("model"), dict(base_fingerprint), label="pilot base model"
    )
    _require_equal(
        provenance.get("adapter"),
        dict(checkpoint_fingerprint),
        label="pilot checkpoint adapter",
    )
    source = provenance.get("model_source")
    effective = provenance.get("effective_model")
    expected_composition_payload = {
        "base_model_directory_sha256": base_fingerprint["sha256"],
        "adapter_directory_sha256": checkpoint_fingerprint["sha256"],
    }
    expected_composition_sha = _sha256_text(
        _canonical_json(expected_composition_payload)
    )
    if (
        not isinstance(source, Mapping)
        or source.get("mode") != "base_model_plus_adapter"
    ):
        raise SelectionMergeError("pilot did not evaluate a base-plus-adapter model")
    source_base = source.get("base_model")
    source_adapter = source.get("adapter")
    composition = source.get("composition")
    if (
        not isinstance(source_base, Mapping)
        or source_base.get("directory") != dict(base_fingerprint)
        or not isinstance(source_adapter, Mapping)
        or source_adapter.get("directory") != dict(checkpoint_fingerprint)
        or not isinstance(composition, Mapping)
        or composition.get("payload") != expected_composition_payload
        or composition.get("sha256") != expected_composition_sha
        or not isinstance(effective, Mapping)
        or effective.get("kind") != "peft_composition"
        or effective.get("sha256") != expected_composition_sha
    ):
        raise SelectionMergeError("pilot PEFT composition provenance drift")
    results = summary.get("results")
    if not isinstance(results, Mapping):
        raise SelectionMergeError("pilot results binding is missing")
    results_path = Path(str(results.get("path"))).resolve()
    results_sha = _require_hash(
        results_path, results.get("sha256"), label="pilot results"
    )
    required_results_sha = ACTIVE_SELECTION_PROFILE.required_pilot_results_sha256
    if required_results_sha is not None and results_sha != required_results_sha:
        raise SelectionMergeError("pilot results are not the profile-pinned artifact")
    _require_equal(results.get("rows"), 15, label="pilot results rows")
    result_rows = _read_jsonl(results_path, label="pilot results")
    if len(result_rows) != 15:
        raise SelectionMergeError("pilot results must contain exactly 15 rows")

    sample_by_direction = {
        str(item["direction"]): item for item in samples if isinstance(item, Mapping)
    }
    expected_cases: set[tuple[str, str, int, int]] = set()
    for direction in ("hold", "hike", "cut"):
        sample = sample_by_direction[direction]
        expected_cases.add(
            (str(sample["sample_id"]), "greedy", 0, int(sample["greedy_seed"]))
        )
        for index, seed in enumerate(sample["sample_seeds"], start=1):
            expected_cases.add((str(sample["sample_id"]), "sampled", index, int(seed)))
    observed_cases: set[tuple[str, str, int, int]] = set()
    expected_row_provenance = {
        "sample_manifest_sha256": observed_manifest_sha,
        "release_manifest_sha256": RELEASE_MANIFEST_SHA256,
        "training_config_sha256": grpo_config_sha,
        "tokenizer_bundle_sha256": tokenizer["bundle_sha256"],
        "model_sha256": base_fingerprint["sha256"],
        "model_loading_mode": "base_model_plus_adapter",
        "base_model_sha256": base_fingerprint["sha256"],
        "adapter_sha256": checkpoint_fingerprint["sha256"],
        "effective_model_sha256": expected_composition_sha,
    }
    for index, row in enumerate(result_rows, start=1):
        direction = str(row.get("target_direction"))
        sample = sample_by_direction.get(direction)
        if sample is None:
            raise SelectionMergeError(f"pilot row {index} has an invalid direction")
        case = (
            str(row.get("sample_id")),
            str(row.get("generation_mode")),
            int(row.get("generation_index", -1)),
            int(row.get("seed", -1)),
        )
        observed_cases.add(case)
        if (
            row.get("schema_version") != "chk4-decision-grpo-stratified-probe-v1"
            or row.get("split") != "train"
            or str(row.get("sample_id")) != str(sample.get("sample_id"))
            or row.get("target_magnitude_bp") != sample.get("magnitude_bp")
            or not isinstance(row.get("completion"), str)
            or row.get("completion_sha256") != _sha256_text(str(row.get("completion")))
        ):
            raise SelectionMergeError(f"pilot row {index} binding drift")
        if row.get("think_boundary_count") != str(row["completion"]).count(
            "</think>"
        ) or (bool(row.get("hit_eos")) and bool(row.get("cap_reached"))):
            raise SelectionMergeError(f"pilot row {index} delivery metrics drift")
        row_provenance = row.get("provenance")
        if not isinstance(row_provenance, Mapping):
            raise SelectionMergeError(f"pilot row {index} provenance is missing")
        for key, expected in expected_row_provenance.items():
            _require_equal(
                row_provenance.get(key), expected, label=f"pilot row {index} {key}"
            )
    if observed_cases != expected_cases:
        raise SelectionMergeError(
            "pilot result case matrix does not match its manifest"
        )

    # Recompute the complete quality summary from the immutable JSONL instead of
    # trusting copied aggregate values in summary.json.
    try:
        from jobs.retrain_v2.probe_chk4_decision_grpo_stratified import (
            replay_decision_dense_v3,
            summarize_stratified_results,
        )

        for index, row in enumerate(result_rows, start=1):
            replayed = replay_decision_dense_v3(
                str(row["completion"]),
                {
                    "direction": row["target_direction"],
                    "magnitude_bp": row["target_magnitude_bp"],
                },
                hit_eos=bool(row.get("hit_eos")),
                cap_reached=bool(row.get("cap_reached")),
            )
            replay_fields = {
                "decision_dense_v3_reward": replayed["reward"],
                "decision_dense_v3_nonzero": replayed["nonzero"],
                "response_format": replayed["response_format"],
                "strict_json": replayed["strict_json"],
                "fenced_json": replayed["fenced_json"],
                "decision_prediction": replayed["prediction"],
                "decision_direction_correct": replayed["direction_correct"],
                "decision_exact": replayed["exact"],
                "decision_rejection_reason": replayed["rejection_reason"],
                "decision_forced_zero_reason": replayed["forced_zero_reason"],
            }
            for key, expected in replay_fields.items():
                _require_equal(
                    row.get(key), expected, label=f"pilot row {index} replay {key}"
                )
        recomputed = summarize_stratified_results(result_rows, provenance=provenance)
    except Exception as exc:
        raise SelectionMergeError("pilot results cannot be re-summarized") from exc
    for key in (
        "schema_version",
        "status",
        "quality_status",
        "quality_gate",
        "cases",
        "train_groups",
        "greedy_cases",
        "sampled_cases",
        "direction_order",
        "directions",
        "overall",
        "provenance",
    ):
        _require_equal(
            summary.get(key), recomputed.get(key), label=f"pilot summary {key}"
        )
    launch_descriptor: dict[str, Any] | None = None
    required_launch_sha = ACTIVE_SELECTION_PROFILE.required_pilot_launch_sha256
    if required_launch_sha is not None:
        launch_path = summary_path.parent / "launch.json"
        observed_launch_sha = _require_hash(
            launch_path, required_launch_sha, label="pilot launch"
        )
        launch = _read_json(launch_path, label="pilot launch")
        launch_provenance = launch.get("provenance")
        if (
            launch.get("schema_version") != "chk4-decision-grpo-stratified-probe-v1"
            or launch.get("status") != "initializing"
            or not isinstance(launch_provenance, Mapping)
            or launch_provenance.get("model") != dict(base_fingerprint)
            or launch_provenance.get("adapter") != dict(checkpoint_fingerprint)
            or launch_provenance.get("sample_manifest_sha256") != observed_manifest_sha
            or launch_provenance.get("release_manifest_sha256")
            != RELEASE_MANIFEST_SHA256
            or launch_provenance.get("training_config_sha256") != grpo_config_sha
        ):
            raise SelectionMergeError("pilot launch provenance drift")
        launch_descriptor = {
            "path": str(launch_path),
            "sha256": observed_launch_sha,
        }
    return {
        "manifest": {
            "path": str(manifest_path),
            "sha256": observed_manifest_sha,
        },
        "summary": {"path": str(summary_path), "sha256": observed_summary_sha},
        "results": {
            "path": str(results_path),
            "sha256": results_sha,
            "rows": 15,
        },
        "launch": launch_descriptor,
    }


def build_selection_plan(
    *,
    repo_root: Path,
    checkpoint_step: int,
    pilot_manifest: Path,
    pilot_manifest_sha256: str,
    pilot_summary: Path,
    pilot_summary_sha256: str,
    require_published: bool = False,
) -> SelectionPlan:
    root = repo_root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise SelectionMergeError(f"repository root is missing or unsafe: {root}")
    if (
        isinstance(checkpoint_step, bool)
        or checkpoint_step not in ALLOWED_CHECKPOINT_STEPS
    ):
        if ACTIVE_SELECTION_PROFILE.name == LEGACY_PROFILE_NAME:
            raise SelectionMergeError("checkpoint step must be exactly 10, 20, or 24")
        allowed = ", ".join(str(step) for step in sorted(ALLOWED_CHECKPOINT_STEPS))
        raise SelectionMergeError(f"checkpoint step must be one of: {allowed}")
    if (
        ACTIVE_SELECTION_PROFILE.required_pilot_manifest_sha256 is not None
        and pilot_manifest_sha256
        != ACTIVE_SELECTION_PROFILE.required_pilot_manifest_sha256
    ):
        raise SelectionMergeError("pilot manifest is not the profile-pinned artifact")
    if (
        ACTIVE_SELECTION_PROFILE.required_pilot_summary_sha256 is not None
        and pilot_summary_sha256
        != ACTIVE_SELECTION_PROFILE.required_pilot_summary_sha256
    ):
        raise SelectionMergeError("pilot summary is not the profile-pinned artifact")
    run_root = (root / RUN_ROOT_RELATIVE).resolve()
    if run_root != root / RUN_ROOT_RELATIVE or run_root.is_symlink():
        raise SelectionMergeError("steps24 run root is unsafe")
    (
        sft_config_path,
        sft_config,
        sft_config_sha,
        grpo_config_path,
        _,
        grpo_config_sha,
    ) = _validate_configs(root)
    (
        training_output,
        runtime_path,
        runtime_sha,
        state_path,
        state_sha,
        results_path,
        results_sha,
    ) = _validate_complete_training(
        repo_root=root,
        run_root=run_root,
        config_path=sft_config_path,
        config=sft_config,
        config_sha=sft_config_sha,
    )
    base_model = _resolve(
        root, sft_config.get("model_name_or_path"), label="SFT parent model"
    )
    if not base_model.is_dir() or base_model.is_symlink():
        raise SelectionMergeError("SFT parent model is missing or unsafe")
    checkpoint, checkpoint_fingerprint = _validate_checkpoint(
        repo_root=root,
        output=training_output,
        step=checkpoint_step,
        base_model=base_model,
        config=sft_config,
    )
    base_fingerprint = fingerprint_artifact_path(base_model)
    parent_lineage = _validate_parent_lineage(
        repo_root=root,
        run_root=run_root,
        base_fingerprint=base_fingerprint,
        sft_config_path=sft_config_path,
        sft_config_sha=sft_config_sha,
        grpo_config_path=grpo_config_path,
        grpo_config_sha=grpo_config_sha,
    )
    pilot = _validate_pilot(
        repo_root=root,
        manifest_path=pilot_manifest,
        manifest_sha=pilot_manifest_sha256,
        summary_path=pilot_summary,
        summary_sha=pilot_summary_sha256,
        grpo_config_path=grpo_config_path,
        grpo_config_sha=grpo_config_sha,
        base_fingerprint=base_fingerprint,
        checkpoint_fingerprint=checkpoint_fingerprint,
    )
    selection_root = (
        run_root / "selected_sft_checkpoints" / f"checkpoint-{checkpoint_step}"
    )
    destination = selection_root / "merged/chk4_sft"
    evidence_path = selection_root / "receipts/exact_merge.json"
    receipt_path = selection_root / "receipts/selection_receipt.json"
    for directory in (
        run_root / "selected_sft_checkpoints",
        selection_root,
        selection_root / "merged",
        selection_root / "receipts",
    ):
        if directory.is_symlink():
            raise SelectionMergeError(f"selection path contains a symlink: {directory}")
        if directory.exists() and not directory.is_dir():
            raise SelectionMergeError(
                f"selection directory path is occupied by a file: {directory}"
            )
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise SelectionMergeError(
            f"selected merge destination is unsafe: {destination}"
        )
    if evidence_path.is_symlink():
        raise SelectionMergeError(f"exact evidence path is unsafe: {evidence_path}")
    if receipt_path.is_symlink():
        raise SelectionMergeError(f"selection receipt is unsafe: {receipt_path}")
    if require_published and not receipt_path.is_file():
        raise SelectionMergeError(
            f"published selection receipt is missing: {receipt_path}"
        )
    if not require_published and receipt_path.exists():
        raise SelectionMergeError(
            f"selection receipt already exists; create-only contract: {receipt_path}"
        )
    return SelectionPlan(
        repo_root=root,
        checkpoint_step=checkpoint_step,
        run_root=run_root,
        sft_config_path=sft_config_path,
        grpo_config_path=grpo_config_path,
        training_output=training_output,
        checkpoint=checkpoint,
        base_model=base_model,
        destination=destination,
        exact_evidence_path=evidence_path,
        receipt_path=receipt_path,
        sft_config_sha256=sft_config_sha,
        grpo_config_sha256=grpo_config_sha,
        runtime_path=runtime_path,
        runtime_sha256=runtime_sha,
        trainer_state_path=state_path,
        trainer_state_sha256=state_sha,
        train_results_path=results_path,
        train_results_sha256=results_sha,
        base_fingerprint=base_fingerprint,
        checkpoint_fingerprint=checkpoint_fingerprint,
        parent_lineage=parent_lineage,
        pilot_manifest_path=Path(pilot["manifest"]["path"]),
        pilot_manifest_sha256=pilot["manifest"]["sha256"],
        pilot_summary_path=Path(pilot["summary"]["path"]),
        pilot_summary_sha256=pilot["summary"]["sha256"],
        pilot_results=pilot["results"],
        pilot_launch=pilot["launch"],
    )


def verify_published_selection(
    *,
    repo_root: Path,
    checkpoint_step: int,
    pilot_manifest: Path,
    pilot_manifest_sha256: str,
    pilot_summary: Path,
    pilot_summary_sha256: str,
) -> tuple[SelectionPlan, Mapping[str, Any]]:
    """Revalidate every input plus the create-only published merge receipt."""

    plan = build_selection_plan(
        repo_root=repo_root,
        checkpoint_step=checkpoint_step,
        pilot_manifest=pilot_manifest,
        pilot_manifest_sha256=pilot_manifest_sha256,
        pilot_summary=pilot_summary,
        pilot_summary_sha256=pilot_summary_sha256,
        require_published=True,
    )
    receipt = _read_json(plan.receipt_path, label="selection receipt")
    validate_manifest_integrity(receipt)
    public = plan.public()
    for key, expected in public.items():
        if key == "status":
            continue
        _require_equal(receipt.get(key), expected, label=f"selection receipt {key}")
    if receipt.get("status") != "passed":
        raise SelectionMergeError("selection receipt did not pass")
    binding = _binding(plan)
    _require_equal(
        receipt.get("merge_binding_sha256"),
        _sha256_text(_canonical_json(binding)),
        label="selection merge binding",
    )
    evidence = _verify_existing_exact(plan)
    exact_descriptor = receipt.get("exact_merge_evidence")
    if (
        not isinstance(exact_descriptor, Mapping)
        or Path(str(exact_descriptor.get("path"))).resolve() != plan.exact_evidence_path
        or exact_descriptor.get("sha256") != sha256_file(plan.exact_evidence_path)
        or exact_descriptor.get("payload_sha256")
        != evidence["integrity"]["payload_sha256"]
    ):
        raise SelectionMergeError("selection exact-evidence descriptor drift")
    merged_fingerprint = fingerprint_artifact_path(plan.destination)
    _require_equal(
        receipt.get("merged_artifact"),
        merged_fingerprint,
        label="selection merged artifact",
    )
    attestation_descriptor = receipt.get("merge_attestation")
    attestation_path = plan.destination / "merge_attestation.json"
    if (
        not isinstance(attestation_descriptor, Mapping)
        or Path(str(attestation_descriptor.get("path"))).resolve() != attestation_path
        or attestation_descriptor.get("sha256") != sha256_file(attestation_path)
    ):
        raise SelectionMergeError("selection merge-attestation descriptor drift")
    try:
        from jobs.retrain_v2.merge_attestation import verify_merge_attestation

        verify_merge_attestation(plan.destination, expected_binding=binding)
    except Exception as exc:
        raise SelectionMergeError("selection merge attestation is invalid") from exc
    return plan, receipt


def _binding(plan: SelectionPlan) -> dict[str, Any]:
    return {
        "schema_version": MERGE_BINDING_SCHEMA,
        "profile": PROFILE_NAME,
        "branch_id": BRANCH_ID,
        "scope": {
            "operation": "selected_sft_checkpoint_cpu_merge",
            "canonical_dag_bindable": False,
            "not_authorized": ["root_adapter", "different_checkpoint", "gpu_merge"],
        },
        "checkpoint_step": plan.checkpoint_step,
        "source_checkpoint": dict(plan.checkpoint_fingerprint),
        "root_training_output": str(plan.training_output),
        "root_adapter_used": False,
        "base_model": dict(plan.base_fingerprint),
        "destination": str(plan.destination),
        "training": plan.public()["training"],
        "pilot": plan.public()["pilot"],
    }


def _write_exclusive_readonly_json(path: Path, payload: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise SelectionMergeError(f"refusing to overwrite create-only record: {path}")
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


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise SelectionMergeError(f"merged destination contains symlink: {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


@contextmanager
def _selection_lock(plan: SelectionPlan) -> Iterator[None]:
    lock_path = plan.run_root / ".branch.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_symlink():
        raise SelectionMergeError("branch lock is unsafe")
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SelectionMergeError(
                "steps24 training/branch operation still owns the lock"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _verify_existing_exact(plan: SelectionPlan) -> Mapping[str, Any]:
    evidence = _read_json(plan.exact_evidence_path, label="exact merge evidence")
    validate_manifest_integrity(evidence)
    sources = evidence.get("sources")
    if (
        evidence.get("conclusion") != EXACT_CONCLUSION
        or evidence.get("subject_artifact_id")
        != f"chk4-decision-sft-selected-checkpoint-{plan.checkpoint_step}"
        or not isinstance(sources, Mapping)
        or sources.get("base_model") != dict(plan.base_fingerprint)
        or sources.get("adapter") != dict(plan.checkpoint_fingerprint)
        or sources.get("merged_model") != fingerprint_artifact_path(plan.destination)
    ):
        raise SelectionMergeError("existing exact evidence binding drift")
    return evidence


def execute_selection(
    *,
    repo_root: Path,
    checkpoint_step: int,
    pilot_manifest: Path,
    pilot_manifest_sha256: str,
    pilot_summary: Path,
    pilot_summary_sha256: str,
    merge_runner: Callable[..., Mapping[str, Any]] | None = None,
    exact_verifier: Callable[..., Mapping[str, Any]] | None = None,
    attestation_verifier: Callable[..., Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", "-1"):
        raise SelectionMergeError(
            "CPU-only merge requires CUDA_VISIBLE_DEVICES to be unset, empty, or -1"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    plan = build_selection_plan(
        repo_root=repo_root,
        checkpoint_step=checkpoint_step,
        pilot_manifest=pilot_manifest,
        pilot_manifest_sha256=pilot_manifest_sha256,
        pilot_summary=pilot_summary,
        pilot_summary_sha256=pilot_summary_sha256,
    )
    binding = _binding(plan)
    with _selection_lock(plan):
        # Rebuild under the training lock so no checkpoint, config, runtime, or
        # pilot input can change between authorization and publication.
        locked = build_selection_plan(
            repo_root=repo_root,
            checkpoint_step=checkpoint_step,
            pilot_manifest=pilot_manifest,
            pilot_manifest_sha256=pilot_manifest_sha256,
            pilot_summary=pilot_summary,
            pilot_summary_sha256=pilot_summary_sha256,
        )
        if locked.public() != plan.public():
            raise SelectionMergeError("selection inputs changed before merge")
        binding = _binding(locked)
        if (
            merge_runner is None
            or exact_verifier is None
            or attestation_verifier is None
        ):
            from jobs.main.verify_lora_merge_lineage import verify_exact_lora_merge
            from jobs.retrain_v2.merge_adapter import merge_from_config
            from jobs.retrain_v2.merge_attestation import verify_merge_attestation

            merge_runner = merge_runner or merge_from_config
            exact_verifier = exact_verifier or verify_exact_lora_merge
            attestation_verifier = attestation_verifier or verify_merge_attestation

        if locked.destination.exists():
            if locked.destination.is_symlink() or not locked.destination.is_dir():
                raise SelectionMergeError("selected merge destination is unsafe")
            attestation_verifier(locked.destination, expected_binding=binding)
        else:
            locked.destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix=f".checkpoint-{checkpoint_step}-selection-config-",
                dir=locked.run_root,
            ) as temporary:
                merge_config = Path(temporary) / "merge.yaml"
                merge_config.write_text(
                    yaml.safe_dump(
                        {
                            "model_name_or_path": str(locked.base_model),
                            "output_dir": str(locked.checkpoint),
                            "peft_merged_model_path": str(locked.destination),
                        },
                        sort_keys=False,
                    ),
                    encoding="utf-8",
                )
                merge_runner(
                    merge_config,
                    repo_root=locked.repo_root,
                    merge_attestation_binding=binding,
                )
            attestation_verifier(locked.destination, expected_binding=binding)

        evidence: Mapping[str, Any]
        if locked.exact_evidence_path.exists():
            evidence = _verify_existing_exact(locked)
        else:
            evidence = exact_verifier(
                base_model=locked.base_model,
                adapter=locked.checkpoint,
                merged_model=locked.destination,
                base_artifact_id=ACTIVE_SELECTION_PROFILE.base_artifact_id,
                merged_artifact_id=(
                    f"chk4-decision-sft-selected-checkpoint-{checkpoint_step}"
                ),
            )
            if evidence.get("conclusion") != EXACT_CONCLUSION:
                raise SelectionMergeError(
                    "exact tensor merge verification did not pass"
                )
            validate_manifest_integrity(evidence)
            sources = evidence.get("sources")
            if (
                not isinstance(sources, Mapping)
                or sources.get("base_model") != dict(locked.base_fingerprint)
                or sources.get("adapter") != dict(locked.checkpoint_fingerprint)
                or sources.get("merged_model")
                != fingerprint_artifact_path(locked.destination)
            ):
                raise SelectionMergeError("exact evidence source binding drift")
            _write_exclusive_readonly_json(locked.exact_evidence_path, evidence)

        _seal_tree(locked.destination)
        attestation_path = locked.destination / "merge_attestation.json"
        exact_descriptor = {
            "path": str(locked.exact_evidence_path),
            "sha256": sha256_file(locked.exact_evidence_path),
            "payload_sha256": evidence["integrity"]["payload_sha256"],
        }
        receipt_payload = {
            **locked.public(),
            "status": "passed",
            "merge_binding_sha256": _sha256_text(_canonical_json(binding)),
            "merge_attestation": {
                "path": str(attestation_path),
                "sha256": sha256_file(attestation_path),
            },
            "exact_merge_evidence": exact_descriptor,
            "merged_artifact": fingerprint_artifact_path(locked.destination),
        }
        receipt = seal_manifest(receipt_payload)
        _write_exclusive_readonly_json(locked.receipt_path, receipt)
        return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--profile",
        choices=sorted(SELECTION_PROFILES),
        default=LEGACY_PROFILE_NAME,
        help="Versioned source branch; the historical profile remains the default.",
    )
    parser.add_argument(
        "--checkpoint-step",
        required=True,
        type=int,
    )
    parser.add_argument("--pilot-manifest", required=True, type=Path)
    parser.add_argument("--pilot-manifest-sha256", required=True)
    parser.add_argument("--pilot-summary", required=True, type=Path)
    parser.add_argument("--pilot-summary-sha256", required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the CPU merge. Without this flag all checks are read-only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        _activate_selection_profile(args.profile)
        kwargs = {
            "repo_root": args.repo_root,
            "checkpoint_step": args.checkpoint_step,
            "pilot_manifest": args.pilot_manifest,
            "pilot_manifest_sha256": args.pilot_manifest_sha256,
            "pilot_summary": args.pilot_summary,
            "pilot_summary_sha256": args.pilot_summary_sha256,
        }
        result = (
            execute_selection(**kwargs)
            if args.execute
            else build_selection_plan(**kwargs).public()
        )
    except (SelectionMergeError, FileExistsError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
