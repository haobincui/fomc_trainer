"""Standalone chk1-cp200 -> Decision-SFT -> Decision-GRPO workflow.

This workflow intentionally does not modify the canonical retrain-v2 DAG.  All
mutating operations require ``--execute``; training is launched separately so
that a failed or interrupted stage can resume from its numerically latest
checkpoint through the normal Trainer recovery path.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Iterator, Mapping, Sequence

import yaml

from jobs.main.checkpoint_provenance import write_immutable_json
from jobs.main.verify_lora_merge_lineage import verify_exact_lora_merge
from jobs.retrain_v2.merge_adapter import merge_from_config
from jobs.retrain_v2.merge_attestation import verify_merge_attestation
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.trainer.dataset_release import (
    CHK4_PRE2009_GRPO_ROLE,
    CHK4_PRE2009_ROLES,
    CHK4_PRE2009_SFT_ROLE,
    verify_chk4_decision_release,
    verify_chk4_hier_balanced_release,
    verify_chk4_pre2009_augmented_release,
)
from open_r1.trainer.fixed_schedule_sampler import (
    SUPPORTED_SAMPLER_TYPES,
    sampler_source_binding,
)
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


PROFILE_ENV = "FOMC_CHK4_BRANCH_PROFILE"
DEFAULT_PROFILE_NAME = "core_v3"
WARM_FIX_PROFILE_NAME = "warm_fix_v1"
WARM_FIX_LR1E5_PROFILE_NAME = "warm_fix_lr1e5_v2"
WARM_FIX_LR1E5_STEPS24_PROFILE_NAME = "warm_fix_lr1e5_steps24_v3"
HIER_BALANCED_PROFILE_NAME = "hier_balanced_lr1e5_steps24_v4"
HIER_BALANCED_V5_PROFILE_NAME = "hier_balanced_lr1e5_steps24_v5"
PRE2009_BALANCED_PROFILE_NAME = "pre2009_balanced_lr1e5_steps39_v1"
DEFAULT_BRANCH_ID = "chk4_from_chk1_cp200_sft_grpo_core_v3_20260810"
WARM_FIX_BRANCH_ID = "chk4_from_chk1_cp200_sft_grpo_warm_fix_v1_20260810"
WARM_FIX_LR1E5_BRANCH_ID = "chk4_from_chk1_cp200_sft_grpo_warm_fix_lr1e5_v2_20260810"
WARM_FIX_LR1E5_STEPS24_BRANCH_ID = (
    "chk4_from_chk1_cp200_sft_grpo_warm_fix_lr1e5_steps24_v3_20260811"
)
HIER_BALANCED_BRANCH_ID = (
    "chk4_from_chk1_cp200_sft_grpo_hier_balanced_lr1e5_steps24_v4_20260811"
)
HIER_BALANCED_V5_BRANCH_ID = (
    "chk4_from_chk1_cp200_sft_grpo_hier_balanced_lr1e5_steps24_v5_20260811"
)
PRE2009_BALANCED_BRANCH_ID = (
    "chk4_from_chk1_cp200_sft_grpo_pre2009_balanced_lr1e5_steps39_v1_20260811"
)
BRANCH_SCHEMA = "chk4-standalone-sft-grpo-branch-v1"
AUTHORIZATION_SCHEMA = "chk1-cp200-to-chk4-authorization-v1"
MERGE_BINDING_SCHEMA = "chk4-standalone-merge-binding-v1"
STAGE_RECEIPT_SCHEMA = "chk4-standalone-stage-receipt-v1"
PARENT_REUSE_RECEIPT_SCHEMA = "chk4-reused-parent-stage-receipt-v1"


@dataclass(frozen=True)
class BranchProfile:
    """Versioned, process-local workflow selection."""

    name: str
    branch_id: str
    parent_config: Path
    sft_config: Path
    grpo_config: Path
    sft_learning_rate: float
    sft_max_steps: int
    sft_warmup_steps: int
    sft_eval_steps: int
    grpo_max_completion_length: int
    grpo_reward_func: str
    sft_num_train_epochs: int = 3
    reused_parent_branch_id: str | None = None
    sft_dataset_role: str = "decision_sft"
    sft_release_manifest: Path | None = None
    sft_release_sha256: str | None = None
    sft_train_sampler: str = "default"
    sft_checkpoint_keep_steps: tuple[int, ...] = ()
    grpo_dataset_role: str = "decision_grpo"
    grpo_release_manifest: Path | None = None
    grpo_release_sha256: str | None = None

    @property
    def run_root(self) -> Path:
        return Path(f"output/training/retrain_v2/{self.branch_id}")


BRANCH_PROFILES = {
    DEFAULT_PROFILE_NAME: BranchProfile(
        name=DEFAULT_PROFILE_NAME,
        branch_id=DEFAULT_BRANCH_ID,
        parent_config=Path(
            "configs/retrain_v2/chk1_cp200_merge_for_chk4_core_v3_20260810.yaml"
        ),
        sft_config=Path(
            "configs/retrain_v2/chk4_decision_sft_from_chk1_cp200_core_v3_20260810.yaml"
        ),
        grpo_config=Path(
            "configs/retrain_v2/chk4_decision_grpo_from_sft_core_v3_20260810.yaml"
        ),
        sft_learning_rate=1.0e-6,
        sft_max_steps=-1,
        sft_warmup_steps=0,
        sft_eval_steps=10,
        grpo_max_completion_length=512,
        grpo_reward_func="decision_dense_v2",
    ),
    WARM_FIX_PROFILE_NAME: BranchProfile(
        name=WARM_FIX_PROFILE_NAME,
        branch_id=WARM_FIX_BRANCH_ID,
        parent_config=Path(
            "configs/retrain_v2/"
            "chk1_cp200_parent_reuse_for_chk4_warm_fix_v1_20260810.yaml"
        ),
        sft_config=Path(
            "configs/retrain_v2/"
            "chk4_decision_sft_from_chk1_cp200_warm_fix_v1_20260810.yaml"
        ),
        grpo_config=Path(
            "configs/retrain_v2/chk4_decision_grpo_from_sft_warm_fix_v1_20260810.yaml"
        ),
        sft_learning_rate=5.0e-6,
        sft_max_steps=12,
        sft_warmup_steps=2,
        sft_eval_steps=6,
        grpo_max_completion_length=1024,
        grpo_reward_func="decision_dense_v2",
        reused_parent_branch_id=DEFAULT_BRANCH_ID,
    ),
    WARM_FIX_LR1E5_PROFILE_NAME: BranchProfile(
        name=WARM_FIX_LR1E5_PROFILE_NAME,
        branch_id=WARM_FIX_LR1E5_BRANCH_ID,
        parent_config=Path(
            "configs/retrain_v2/"
            "chk1_cp200_parent_reuse_for_chk4_warm_fix_lr1e5_v2_20260810.yaml"
        ),
        sft_config=Path(
            "configs/retrain_v2/"
            "chk4_decision_sft_from_chk1_cp200_warm_fix_lr1e5_v2_20260810.yaml"
        ),
        grpo_config=Path(
            "configs/retrain_v2/"
            "chk4_decision_grpo_from_sft_warm_fix_lr1e5_v2_20260810.yaml"
        ),
        sft_learning_rate=1.0e-5,
        sft_max_steps=8,
        sft_warmup_steps=2,
        sft_eval_steps=4,
        grpo_max_completion_length=1024,
        grpo_reward_func="decision_dense_v2",
        reused_parent_branch_id=DEFAULT_BRANCH_ID,
    ),
    WARM_FIX_LR1E5_STEPS24_PROFILE_NAME: BranchProfile(
        name=WARM_FIX_LR1E5_STEPS24_PROFILE_NAME,
        branch_id=WARM_FIX_LR1E5_STEPS24_BRANCH_ID,
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
        sft_learning_rate=1.0e-5,
        sft_max_steps=24,
        sft_warmup_steps=2,
        sft_eval_steps=4,
        grpo_max_completion_length=1024,
        grpo_reward_func="decision_dense_v3",
        reused_parent_branch_id=DEFAULT_BRANCH_ID,
    ),
    HIER_BALANCED_PROFILE_NAME: BranchProfile(
        name=HIER_BALANCED_PROFILE_NAME,
        branch_id=HIER_BALANCED_BRANCH_ID,
        parent_config=Path(
            "configs/retrain_v2/"
            "chk1_cp200_merge_for_chk4_hier_balanced_lr1e5_steps24_v4_20260811.yaml"
        ),
        sft_config=Path(
            "configs/retrain_v2/"
            "chk4_decision_sft_hier_balanced_lr1e5_steps24_v4_20260811.yaml"
        ),
        grpo_config=Path(
            "configs/retrain_v2/"
            "chk4_decision_grpo_from_hier_balanced_sft_lr1e5_steps24_v4_20260811.yaml"
        ),
        sft_learning_rate=1.0e-5,
        sft_max_steps=24,
        sft_warmup_steps=2,
        sft_eval_steps=4,
        grpo_max_completion_length=1024,
        grpo_reward_func="decision_dense_v3",
        reused_parent_branch_id=DEFAULT_BRANCH_ID,
        sft_dataset_role="decision_sft_hier_balanced",
        sft_release_manifest=Path(
            "dataset/processed/retrain_v2/"
            "chk4_decision_sft_hier_balanced_v1_20260811/release_manifest.json"
        ),
        sft_release_sha256=(
            "18c6f47ed0a085f7aef39d599e4baa30b34e5b560f3a594e7d5e3cf78bfde255"
        ),
        sft_train_sampler="manifest_fixed_schedule_v1",
        sft_checkpoint_keep_steps=(12, 16, 20, 24),
    ),
    HIER_BALANCED_V5_PROFILE_NAME: BranchProfile(
        name=HIER_BALANCED_V5_PROFILE_NAME,
        branch_id=HIER_BALANCED_V5_BRANCH_ID,
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
        sft_learning_rate=1.0e-5,
        sft_max_steps=24,
        sft_warmup_steps=2,
        sft_eval_steps=4,
        grpo_max_completion_length=1024,
        grpo_reward_func="decision_dense_v3",
        reused_parent_branch_id=DEFAULT_BRANCH_ID,
        sft_dataset_role="decision_sft_hier_balanced",
        sft_release_manifest=Path(
            "dataset/processed/retrain_v2/"
            "chk4_decision_sft_hier_balanced_v1_20260811/release_manifest.json"
        ),
        sft_release_sha256=(
            "18c6f47ed0a085f7aef39d599e4baa30b34e5b560f3a594e7d5e3cf78bfde255"
        ),
        sft_train_sampler="manifest_fixed_schedule_v1",
        sft_checkpoint_keep_steps=(12, 16, 20, 24),
    ),
    PRE2009_BALANCED_PROFILE_NAME: BranchProfile(
        name=PRE2009_BALANCED_PROFILE_NAME,
        branch_id=PRE2009_BALANCED_BRANCH_ID,
        parent_config=Path(
            "configs/retrain_v2/"
            "chk1_cp200_parent_reuse_for_chk4_pre2009_balanced_"
            "lr1e5_steps39_v1_20260811.yaml"
        ),
        sft_config=Path(
            "configs/retrain_v2/"
            "chk4_decision_sft_pre2009_balanced_"
            "lr1e5_steps39_v1_20260811.yaml"
        ),
        grpo_config=Path(
            "configs/retrain_v2/"
            "chk4_decision_grpo_from_pre2009_balanced_sft_"
            "lr1e5_steps39_v1_20260811.yaml"
        ),
        sft_learning_rate=1.0e-5,
        sft_max_steps=39,
        sft_warmup_steps=2,
        sft_eval_steps=13,
        grpo_max_completion_length=1024,
        grpo_reward_func="decision_dense_v3",
        sft_num_train_epochs=1,
        reused_parent_branch_id=DEFAULT_BRANCH_ID,
        sft_dataset_role=CHK4_PRE2009_SFT_ROLE,
        sft_release_manifest=Path(
            "dataset/processed/retrain_v2/"
            "chk4_decision_pre2009_train_balanced_v1_20260811/"
            "release_manifest.json"
        ),
        sft_release_sha256=(
            "5141e00569b94e4af96f030f46176e9cd409199a6f26ac1c63a2ce335fe1e2d6"
        ),
        sft_train_sampler="manifest_fixed_schedule_v2",
        sft_checkpoint_keep_steps=(13, 26, 39),
        grpo_dataset_role=CHK4_PRE2009_GRPO_ROLE,
        grpo_release_manifest=Path(
            "dataset/processed/retrain_v2/"
            "chk4_decision_pre2009_train_balanced_v1_20260811/"
            "release_manifest.json"
        ),
        grpo_release_sha256=(
            "5141e00569b94e4af96f030f46176e9cd409199a6f26ac1c63a2ce335fe1e2d6"
        ),
    ),
}


def _select_profile() -> BranchProfile:
    name = os.environ.get(PROFILE_ENV, DEFAULT_PROFILE_NAME).strip()
    profile = BRANCH_PROFILES.get(name)
    if profile is None:
        supported = ", ".join(sorted(BRANCH_PROFILES))
        raise ValueError(
            f"unsupported {PROFILE_ENV}={name!r}; expected one of: {supported}"
        )
    return profile


ACTIVE_PROFILE = _select_profile()
PROFILE_NAME = ACTIVE_PROFILE.name
BRANCH_ID = ACTIVE_PROFILE.branch_id
PARENT_CONFIG = ACTIVE_PROFILE.parent_config
SFT_CONFIG = ACTIVE_PROFILE.sft_config
GRPO_CONFIG = ACTIVE_PROFILE.grpo_config
RELEASE_MANIFEST = Path(
    "dataset/processed/retrain_v2/"
    "chk4_decision_warmstart_grpo_core_v3_20260810/release_manifest.json"
)
RUN_ROOT = ACTIVE_PROFILE.run_root
DEFAULT_RUN_ROOT = BRANCH_PROFILES[DEFAULT_PROFILE_NAME].run_root

EXPECTED_RELEASE_SHA256 = (
    "8b05dce09bcb0a0b27ee240da1e5d730be93921ce19e650a3b3e8b2689adb893"
)
EXPECTED_BASE_SHA256 = (
    "bfb086cb87e9616e60805fc6f6f95826294b1de70b158cfea2d3e213df38ca11"
)
EXPECTED_CP200_ADAPTER_SHA256 = (
    "fa454f73071b3989bf7c9ea3aaabb5a599196396a3ecb369d8956b849bacdc2f"
)
EXPECTED_SYSTEM_PROMPT_SHA256 = (
    "426b64532dfb955a3941db2cc2bcfc9a785c0540ffba9ee94afabc189fe4ce18"
)


def _profile_sft_release_manifest(profile: BranchProfile) -> Path:
    return profile.sft_release_manifest or RELEASE_MANIFEST


def _profile_sft_release_sha256(profile: BranchProfile) -> str:
    return profile.sft_release_sha256 or EXPECTED_RELEASE_SHA256


def _profile_grpo_release_manifest(profile: BranchProfile) -> Path:
    return profile.grpo_release_manifest or RELEASE_MANIFEST


def _profile_grpo_release_sha256(profile: BranchProfile) -> str:
    return profile.grpo_release_sha256 or EXPECTED_RELEASE_SHA256


def _profile_release_manifest(profile: BranchProfile, role: str) -> Path:
    if role == "decision_sft":
        return _profile_sft_release_manifest(profile)
    if role == "decision_grpo":
        return _profile_grpo_release_manifest(profile)
    raise Chk4BranchError(f"unsupported chk4 training role: {role}")


def _profile_release_sha256(profile: BranchProfile, role: str) -> str:
    if role == "decision_sft":
        return _profile_sft_release_sha256(profile)
    if role == "decision_grpo":
        return _profile_grpo_release_sha256(profile)
    raise Chk4BranchError(f"unsupported chk4 training role: {role}")


def _profile_for_branch_id(branch_id: str) -> BranchProfile:
    matches = [
        profile
        for profile in BRANCH_PROFILES.values()
        if profile.branch_id == branch_id
    ]
    if len(matches) != 1:
        raise Chk4BranchError(f"unknown or ambiguous chk4 branch_id: {branch_id}")
    return matches[0]


def _runtime_dataset_role(role: str, profile: BranchProfile | None = None) -> str:
    selected = profile or ACTIVE_PROFILE
    if role == "decision_sft":
        return selected.sft_dataset_role
    if role == "decision_grpo":
        return selected.grpo_dataset_role
    raise Chk4BranchError(f"unsupported chk4 training role: {role}")


def _runtime_release_sha256(role: str, profile: BranchProfile | None = None) -> str:
    selected = profile or ACTIVE_PROFILE
    return _profile_release_sha256(selected, role)


def _runtime_dataset_directory(dataset_role: str) -> str:
    if dataset_role in {
        "decision_sft",
        "decision_sft_hier_balanced",
        CHK4_PRE2009_SFT_ROLE,
    }:
        return "decision_sft"
    if dataset_role in {"decision_grpo", CHK4_PRE2009_GRPO_ROLE}:
        return "decision_grpo"
    raise Chk4BranchError(f"unsupported chk4 dataset role: {dataset_role}")


def _verify_runtime_release(
    *,
    dataset_dir: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
    dataset_role: str,
    system_prompt: str | None,
    model_path: Path,
) -> dict[str, Any]:
    if dataset_role in CHK4_PRE2009_ROLES:
        verifier = verify_chk4_pre2009_augmented_release
    elif dataset_role == "decision_sft_hier_balanced":
        verifier = verify_chk4_hier_balanced_release
    else:
        verifier = verify_chk4_decision_release
    return verifier(
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        dataset_role=dataset_role,
        system_prompt=system_prompt,
        model_path=model_path,
    )


# Kept alive across ``exec`` so the accelerate parent process owns the same
# lock used by the merge operations for the entire training lifetime.
_EXEC_LOCK_HANDLE: IO[str] | None = None


class Chk4BranchError(RuntimeError):
    """Raised when a standalone chk4 branch invariant is not satisfied."""


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


def _write_readonly_json(path: Path, payload: Mapping[str, Any]) -> str:
    digest = write_immutable_json(path, payload)
    path.chmod(0o400)
    return digest


def _verify_training_environment() -> dict[str, str]:
    expected = {
        "accelerate": "1.4.0",
        "bitsandbytes": "0.48.2",
        "peft": "0.15.2",
        "torch": "2.10.0+cu128",
        "transformers": "4.57.6",
        "trl": "1.2.0",
    }
    observed: dict[str, str] = {}
    for package, version in expected.items():
        try:
            observed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise Chk4BranchError(
                f"fomc_trainer dependency is missing: {package}"
            ) from exc
        if observed[package] != version:
            raise Chk4BranchError(
                f"fomc_trainer {package} version drift: "
                f"expected {version}, observed {observed[package]}"
            )
    if Path(sys.prefix).name != "fomc_trainer":
        raise Chk4BranchError(
            f"chk4 branch must run in fomc_trainer; sys.prefix={sys.prefix}"
        )
    return observed


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise Chk4BranchError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Chk4BranchError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Chk4BranchError(f"{label} must contain a JSON object: {path}")
    return value


def _read_yaml(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise Chk4BranchError(f"{label} is missing or unsafe: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise Chk4BranchError(f"{label} is invalid YAML: {path}") from exc
    if not isinstance(value, dict):
        raise Chk4BranchError(f"{label} must contain a YAML mapping: {path}")
    return value


def _repo_path(repo_root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def _config_path(repo_root: Path, path: Path) -> Path:
    resolved = _repo_path(repo_root, path)
    if not resolved.is_file() or resolved.is_symlink():
        raise Chk4BranchError(f"branch config is missing or unsafe: {resolved}")
    return resolved


def _path_value(repo_root: Path, config: Mapping[str, Any], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise Chk4BranchError(f"training config is missing {key}")
    return _repo_path(repo_root, value)


def _require_equal(observed: Any, expected: Any, *, label: str) -> None:
    if observed != expected:
        raise Chk4BranchError(
            f"{label} drift: expected {expected!r}, observed {observed!r}"
        )


def _config_bundle(repo_root: Path) -> dict[str, Any]:
    paths = {
        "parent": _config_path(repo_root, PARENT_CONFIG),
        "sft": _config_path(repo_root, SFT_CONFIG),
        "grpo": _config_path(repo_root, GRPO_CONFIG),
    }
    payloads = {
        name: _read_yaml(path, label=f"{name} config") for name, path in paths.items()
    }
    return {
        "paths": paths,
        "payloads": payloads,
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
    }


def _validate_config_contract(repo_root: Path, bundle: Mapping[str, Any]) -> None:
    configs = bundle["payloads"]
    parent = configs["parent"]
    sft = configs["sft"]
    grpo = configs["grpo"]

    parent_base = _path_value(repo_root, parent, "model_name_or_path")
    parent_adapter = _path_value(repo_root, parent, "output_dir")
    parent_merged = _path_value(repo_root, parent, "peft_merged_model_path")
    sft_parent = _path_value(repo_root, sft, "model_name_or_path")
    sft_adapter = _path_value(repo_root, sft, "output_dir")
    sft_merged = _path_value(repo_root, sft, "peft_merged_model_path")
    grpo_parent = _path_value(repo_root, grpo, "model_name_or_path")
    grpo_adapter = _path_value(repo_root, grpo, "output_dir")
    grpo_merged = _path_value(repo_root, grpo, "peft_merged_model_path")
    branch_root = _repo_path(repo_root, RUN_ROOT)
    raw_branch_root = repo_root / RUN_ROOT
    if raw_branch_root.is_symlink():
        raise Chk4BranchError("standalone branch root must not be a symlink")

    _require_equal(sft_parent, parent_merged, label="SFT parent chain")
    _require_equal(grpo_parent, sft_merged, label="GRPO parent chain")
    if (
        len(
            {
                parent_base,
                parent_adapter,
                parent_merged,
                sft_adapter,
                sft_merged,
                grpo_adapter,
                grpo_merged,
            }
        )
        != 7
    ):
        raise Chk4BranchError("base, adapters, and merged outputs must be distinct")
    if "selected_cp200_for_chk2" in str(sft_parent):
        raise Chk4BranchError("the chk2-scoped merged cp200 artifact is prohibited")
    expected_parent = (
        _repo_path(repo_root, DEFAULT_RUN_ROOT / "parent/merged/chk1")
        if ACTIVE_PROFILE.reused_parent_branch_id is not None
        else branch_root / "parent/merged/chk1"
    )
    expected_outputs = {
        "parent merge": expected_parent,
        "SFT adapter": branch_root / "adapters/chk4_sft",
        "SFT merge": branch_root / "merged/chk4_sft",
        "GRPO adapter": branch_root / "adapters/chk4_grpo",
        "GRPO merge": branch_root / "merged/chk4_grpo",
    }
    observed_outputs = {
        "parent merge": parent_merged,
        "SFT adapter": sft_adapter,
        "SFT merge": sft_merged,
        "GRPO adapter": grpo_adapter,
        "GRPO merge": grpo_merged,
    }
    for label, expected in expected_outputs.items():
        _require_equal(observed_outputs[label], expected, label=f"{label} path")

    core_release_path = _repo_path(repo_root, RELEASE_MANIFEST)
    if sha256_file(core_release_path) != EXPECTED_RELEASE_SHA256:
        raise Chk4BranchError("chk4 release manifest hash drift")
    release = _read_json(core_release_path, label="chk4 release manifest")
    if (
        release.get("schema_version") != "chk4-decision-training-release-v1"
        or release.get("quality_status") != "passed"
        or release.get("immutable") is not True
        or release.get("training_ready") is not True
        or release.get("canonical_dag_bindable") is not False
    ):
        raise Chk4BranchError("chk4 release is not the approved standalone release")
    if release.get("semantic_assurance") != {
        "structural_lineage_replay": "passed",
        "known_pattern_target_outcome_hits": 0,
        "independent_source_only_semantic_judge": "not_run",
    }:
        raise Chk4BranchError("chk4 release semantic-assurance contract drift")

    sft_release_path = _repo_path(
        repo_root, _profile_sft_release_manifest(ACTIVE_PROFILE)
    )
    sft_release_sha256 = _profile_sft_release_sha256(ACTIVE_PROFILE)
    grpo_release_path = _repo_path(
        repo_root, _profile_grpo_release_manifest(ACTIVE_PROFILE)
    )
    grpo_release_sha256 = _profile_grpo_release_sha256(ACTIVE_PROFILE)
    if sha256_file(sft_release_path) != sft_release_sha256:
        raise Chk4BranchError("chk4 Decision-SFT release manifest hash drift")
    if sha256_file(grpo_release_path) != grpo_release_sha256:
        raise Chk4BranchError("chk4 Decision-GRPO release manifest hash drift")
    checked_release_paths: set[Path] = set()
    for role, release_path in (
        ("decision_sft", sft_release_path),
        ("decision_grpo", grpo_release_path),
    ):
        if release_path == core_release_path or release_path in checked_release_paths:
            continue
        checked_release_paths.add(release_path)
        role_release = _read_json(
            release_path, label=f"{role} standalone release manifest"
        )
        dataset_role = _runtime_dataset_role(role)
        expected_schema = (
            "chk4-decision-pre2009-train-balanced-release-v1"
            if dataset_role in CHK4_PRE2009_ROLES
            else "chk4-decision-sft-hier-balanced-release-v1"
        )
        if (
            role_release.get("schema_version") != expected_schema
            or role_release.get("quality_status") != "passed"
            or role_release.get("immutable") is not True
            or role_release.get("training_ready") is not True
            or role_release.get("canonical_dag_bindable") is not False
            or (
                dataset_role == "decision_sft_hier_balanced"
                and role_release.get("dataset_role") != dataset_role
            )
        ):
            raise Chk4BranchError(f"{role} standalone release readiness drift")

    for role, config in (("decision_sft", sft), ("decision_grpo", grpo)):
        expected_role = _runtime_dataset_role(role)
        expected_release_path = _repo_path(
            repo_root, _profile_release_manifest(ACTIVE_PROFILE, role)
        )
        expected_release_sha = _runtime_release_sha256(role)
        _require_equal(
            config.get("dataset_chk4_role"),
            expected_role,
            label=f"{role} role",
        )
        _require_equal(
            _repo_path(repo_root, str(config.get("dataset_chk4_release_manifest"))),
            expected_release_path,
            label=f"{role} release path",
        )
        _require_equal(
            config.get("dataset_chk4_release_manifest_sha256"),
            expected_release_sha,
            label=f"{role} release SHA",
        )
        expected_dataset_dir = expected_release_path.parent / (
            _runtime_dataset_directory(expected_role)
        )
        _require_equal(
            _path_value(repo_root, config, "dataset_name"),
            expected_dataset_dir,
            label=f"{role} dataset path",
        )
        _require_equal(config.get("dataset_train_split"), "train", label="train split")
        _require_equal(
            config.get("dataset_test_split"), "validation", label="validation split"
        )
        system_prompt = config.get("system_prompt")
        if not isinstance(system_prompt, str):
            raise Chk4BranchError(f"{role} system prompt is missing")
        _require_equal(
            _sha256_text(system_prompt),
            EXPECTED_SYSTEM_PROMPT_SHA256,
            label=f"{role} system prompt SHA",
        )
        for key, expected in (
            ("load_in_4bit", True),
            ("dtype", "bfloat16"),
            ("gradient_accumulation_steps", 8),
            ("per_device_train_batch_size", 1),
            (
                "num_train_epochs",
                ACTIVE_PROFILE.sft_num_train_epochs if role == "decision_sft" else 3,
            ),
            ("save_steps", 1),
            ("save_total_limit", None),
            ("checkpoint_keep_last", 3),
            ("overwrite_output_dir", False),
            ("peft_bias", "none"),
        ):
            _require_equal(config.get(key), expected, label=f"{role}.{key}")
        expected_keep_every = (
            0
            if role == "decision_sft" and ACTIVE_PROFILE.sft_checkpoint_keep_steps
            else 10
        )
        _require_equal(
            config.get("checkpoint_keep_every_n_steps"),
            expected_keep_every,
            label=f"{role}.checkpoint_keep_every_n_steps",
        )
        _require_equal(
            config.get("checkpoint_keep_steps", []),
            (
                list(ACTIVE_PROFILE.sft_checkpoint_keep_steps)
                if role == "decision_sft"
                else []
            ),
            label=f"{role}.checkpoint_keep_steps",
        )

    _require_equal(sft.get("max_length"), 3072, label="SFT max_length")
    _require_equal(
        float(sft.get("learning_rate")),
        ACTIVE_PROFILE.sft_learning_rate,
        label="SFT learning_rate",
    )
    _require_equal(
        sft.get("max_steps"), ACTIVE_PROFILE.sft_max_steps, label="SFT max_steps"
    )
    _require_equal(
        sft.get("warmup_steps", 0),
        ACTIVE_PROFILE.sft_warmup_steps,
        label="SFT warmup_steps",
    )
    _require_equal(
        sft.get("eval_steps"),
        ACTIVE_PROFILE.sft_eval_steps,
        label="SFT eval_steps",
    )
    _require_equal(
        sft.get("completion_only_loss"), True, label="SFT completion-only loss"
    )
    _require_equal(
        sft.get("train_sampler", "default"),
        ACTIVE_PROFILE.sft_train_sampler,
        label="SFT train sampler",
    )
    if ACTIVE_PROFILE.sft_train_sampler in SUPPORTED_SAMPLER_TYPES:
        for key, expected in (
            ("shuffle_dataset", False),
            ("group_by_length", False),
            ("dataloader_drop_last", False),
            ("dataloader_num_workers", 0),
            ("use_liger_kernel", False),
        ):
            _require_equal(sft.get(key), expected, label=f"SFT {key}")
    _require_equal(
        grpo.get("reward_funcs"),
        [ACTIVE_PROFILE.grpo_reward_func],
        label="GRPO reward",
    )
    _require_equal(grpo.get("max_prompt_length"), 2560, label="GRPO prompt budget")
    _require_equal(
        grpo.get("max_completion_length"),
        ACTIVE_PROFILE.grpo_max_completion_length,
        label="GRPO completion budget",
    )
    _require_equal(grpo.get("num_generations"), 4, label="GRPO generations")
    _require_equal(grpo.get("generation_batch_size"), 8, label="GRPO batch")
    _require_equal(grpo.get("use_vllm"), False, label="GRPO vLLM isolation")

    if not parent_base.is_dir() or not parent_adapter.is_dir():
        raise Chk4BranchError("chk0 base or chk1 checkpoint-200 adapter is missing")


def _validate_existing_authorization(
    repo_root: Path, preflight_record: Mapping[str, Any]
) -> dict[str, Any] | None:
    authorization_path = _receipt_paths(repo_root)["authorization"]
    if not authorization_path.exists():
        return None
    authorization = _read_json(authorization_path, label="authorization")
    _validate_authorization(authorization)
    bindings = authorization.get("bindings")
    if not isinstance(bindings, Mapping):
        raise Chk4BranchError("chk4 branch authorization bindings are missing")
    if (
        bindings.get("configs") != preflight_record.get("configs")
        or bindings.get("release_manifest") != preflight_record.get("release_manifest")
        or bindings.get("grpo_release_manifest")
        != preflight_record.get("grpo_release_manifest")
        or bindings.get("reused_parent") != preflight_record.get("reused_parent")
        or Path(str(bindings.get("run_root"))).resolve()
        != _receipt_paths(repo_root)["root"]
    ):
        raise Chk4BranchError("chk4 branch authorization binding drift")
    return authorization


def _reused_parent_binding(repo_root: Path) -> dict[str, Any] | None:
    """Revalidate and describe the default profile's chk4-scoped parent."""

    source_branch_id = ACTIVE_PROFILE.reused_parent_branch_id
    if source_branch_id is None:
        return None
    if source_branch_id != DEFAULT_BRANCH_ID:
        raise Chk4BranchError("warm-fix parent source profile is unsupported")
    source_root = _repo_path(repo_root, DEFAULT_RUN_ROOT)
    source_stage_path = source_root / "receipts/parent_stage.json"
    receipt = _load_stage_receipt(
        source_stage_path,
        expected_stage="parent_merge",
        expected_branch_id=source_branch_id,
    )
    attestation_path = Path(str(receipt["merge_attestation"]["path"])).resolve()
    attestation = _read_json(attestation_path, label="reused parent attestation")
    binding = attestation.get("binding")
    authorization_descriptor = (
        binding.get("authorization") if isinstance(binding, Mapping) else None
    )
    if not isinstance(authorization_descriptor, Mapping):
        raise Chk4BranchError("reused parent authorization binding is missing")
    authorization_path = Path(str(authorization_descriptor.get("path"))).resolve()
    if sha256_file(authorization_path) != authorization_descriptor.get("file_sha256"):
        raise Chk4BranchError("reused parent authorization file drift")
    authorization = _read_json(authorization_path, label="reused parent authorization")
    _validate_authorization(
        authorization, profile=BRANCH_PROFILES[DEFAULT_PROFILE_NAME]
    )
    if authorization.get("branch_id") != source_branch_id or authorization.get(
        "authorization_sha256"
    ) != authorization_descriptor.get("payload_sha256"):
        raise Chk4BranchError("reused parent authorization scope drift")

    exact_path = Path(str(receipt["exact_merge_evidence"]["path"])).resolve()
    exact = _read_json(exact_path, label="reused parent exact evidence")
    sources = exact.get("sources")
    if not isinstance(sources, Mapping):
        raise Chk4BranchError("reused parent exact sources are missing")
    configured_parent = _path_value(
        repo_root,
        _config_bundle(repo_root)["payloads"]["parent"],
        "peft_merged_model_path",
    )
    if Path(str(receipt["merged_artifact"]["path"])).resolve() != configured_parent:
        raise Chk4BranchError("reused parent path disagrees with warm-fix config")
    return {
        "source_branch_id": source_branch_id,
        "source_parent_stage": {
            "path": str(source_stage_path),
            "file_sha256": sha256_file(source_stage_path),
            "receipt_sha256": receipt["receipt_sha256"],
        },
        "source_authorization": {
            "path": str(authorization_path),
            "file_sha256": sha256_file(authorization_path),
            "payload_sha256": authorization["authorization_sha256"],
        },
        "source_merge_attestation": dict(receipt["merge_attestation"]),
        "source_exact_merge_evidence": dict(receipt["exact_merge_evidence"]),
        "merged_artifact": dict(receipt["merged_artifact"]),
        "sources": {
            "base_model": dict(sources["base_model"]),
            "adapter": dict(sources["adapter"]),
        },
    }


def preflight(repo_root: Path) -> dict[str, Any]:
    bundle = _config_bundle(repo_root)
    _validate_config_contract(repo_root, bundle)
    configs = bundle["payloads"]
    core_release_path = _repo_path(repo_root, RELEASE_MANIFEST)
    core_release = _read_json(core_release_path, label="chk4 release manifest")
    grpo_release_path = _repo_path(
        repo_root, _profile_grpo_release_manifest(ACTIVE_PROFILE)
    )
    grpo_release_sha256 = _profile_grpo_release_sha256(ACTIVE_PROFILE)
    sft_release_path = _repo_path(
        repo_root, _profile_sft_release_manifest(ACTIVE_PROFILE)
    )
    sft_release_sha256 = _profile_sft_release_sha256(ACTIVE_PROFILE)
    parent_merged = _path_value(repo_root, configs["parent"], "peft_merged_model_path")
    sources = core_release.get("sources")
    tokenizer = sources.get("tokenizer") if isinstance(sources, Mapping) else None
    if not isinstance(tokenizer, Mapping):
        raise Chk4BranchError("chk4 release tokenizer binding is missing")
    # Keep the authorization input stable across the crash window in which the
    # parent model has been atomically published but its stage receipt has not.
    tokenizer_probe = Path(str(tokenizer.get("path") or "")).resolve()
    verified_release = _verify_runtime_release(
        dataset_dir=sft_release_path.parent
        / _runtime_dataset_directory(ACTIVE_PROFILE.sft_dataset_role),
        manifest_path=sft_release_path,
        expected_manifest_sha256=sft_release_sha256,
        dataset_role=ACTIVE_PROFILE.sft_dataset_role,
        system_prompt=configs["sft"]["system_prompt"],
        model_path=tokenizer_probe,
    )
    verified_grpo_release = None
    if ACTIVE_PROFILE.grpo_dataset_role in CHK4_PRE2009_ROLES:
        verified_grpo_release = _verify_runtime_release(
            dataset_dir=grpo_release_path.parent
            / _runtime_dataset_directory(ACTIVE_PROFILE.grpo_dataset_role),
            manifest_path=grpo_release_path,
            expected_manifest_sha256=grpo_release_sha256,
            dataset_role=ACTIVE_PROFILE.grpo_dataset_role,
            system_prompt=configs["grpo"]["system_prompt"],
            model_path=tokenizer_probe,
        )
    parent_tokenizer_binding = None
    if parent_merged.is_dir():
        parent_verified = _verify_runtime_release(
            dataset_dir=sft_release_path.parent
            / _runtime_dataset_directory(ACTIVE_PROFILE.sft_dataset_role),
            manifest_path=sft_release_path,
            expected_manifest_sha256=sft_release_sha256,
            dataset_role=ACTIVE_PROFILE.sft_dataset_role,
            system_prompt=configs["sft"]["system_prompt"],
            model_path=parent_merged,
        )
        parent_tokenizer_binding = parent_verified["tokenizer_binding"]
        reference_bundle = dict(verified_release["tokenizer_binding"])
        observed_bundle = dict(parent_tokenizer_binding)
        reference_bundle.pop("model_path", None)
        observed_bundle.pop("model_path", None)
        _require_equal(
            observed_bundle,
            reference_bundle,
            label="chk4 parent tokenizer bundle",
        )
    reused_parent = _reused_parent_binding(repo_root)
    release_record = {
        "path": str(sft_release_path),
        "sha256": sft_release_sha256,
        "runtime_binding_schema": verified_release["schema_version"],
        "test_verified_but_not_loaded": verified_release[
            "test_verified_but_not_loaded"
        ],
        "tokenizer_reference": verified_release["tokenizer_binding"],
    }
    if ACTIVE_PROFILE.sft_dataset_role == CHK4_PRE2009_SFT_ROLE:
        sampler = verified_release.get("sampler_contract")
        if not isinstance(sampler, Mapping):
            raise Chk4BranchError("pre-2009 fixed-schedule contract is missing")
        release_record.update(
            {
                "dataset_role": ACTIVE_PROFILE.sft_dataset_role,
                "sampler_contract": {
                    key: sampler[key]
                    for key in (
                        "type",
                        "schedule_rows",
                        "train_rows",
                        "optimizer_steps",
                        "world_size",
                        "per_device_train_batch_size",
                        "gradient_accumulation_steps",
                        "effective_batch_size",
                        "order_is_authoritative",
                        "secondary_shuffle_forbidden",
                    )
                },
                "checkpoint_milestones": list(ACTIVE_PROFILE.sft_checkpoint_keep_steps),
                "required_runtime_topology": {
                    "world_size": 1,
                    "visible_gpus": 1,
                    "gpu_ids": [1],
                },
            }
        )
    result = {
        "schema_version": BRANCH_SCHEMA,
        "status": "passed",
        "profile": PROFILE_NAME,
        "branch_id": BRANCH_ID,
        "canonical_dag_bindable": False,
        "release_manifest": release_record,
        "parent_model_tokenizer": parent_tokenizer_binding,
        "reused_parent": reused_parent,
        "configs": {
            name: {"path": str(bundle["paths"][name]), "sha256": digest}
            for name, digest in bundle["sha256"].items()
        },
        "paths": {
            "base": str(
                _path_value(repo_root, configs["parent"], "model_name_or_path")
            ),
            "source_adapter": str(
                _path_value(repo_root, configs["parent"], "output_dir")
            ),
            "parent_merged": str(
                _path_value(repo_root, configs["parent"], "peft_merged_model_path")
            ),
            "sft_adapter": str(_path_value(repo_root, configs["sft"], "output_dir")),
            "sft_merged": str(
                _path_value(repo_root, configs["sft"], "peft_merged_model_path")
            ),
            "grpo_adapter": str(_path_value(repo_root, configs["grpo"], "output_dir")),
        },
    }
    if sft_release_path != grpo_release_path or verified_grpo_release is not None:
        grpo_record = {
            "path": str(grpo_release_path),
            "sha256": grpo_release_sha256,
        }
        if verified_grpo_release is not None:
            grpo_record.update(
                {
                    "dataset_role": ACTIVE_PROFILE.grpo_dataset_role,
                    "runtime_binding_schema": verified_grpo_release["schema_version"],
                    "test_verified_but_not_loaded": verified_grpo_release[
                        "test_verified_but_not_loaded"
                    ],
                }
            )
        result["grpo_release_manifest"] = grpo_record
    authorization = _validate_existing_authorization(repo_root, result)
    result["authorization"] = (
        {
            "status": "verified",
            "path": str(_receipt_paths(repo_root)["authorization"]),
            "file_sha256": sha256_file(_receipt_paths(repo_root)["authorization"]),
            "payload_sha256": authorization["authorization_sha256"],
        }
        if authorization is not None
        else None
    )
    return result


def _receipt_paths(repo_root: Path) -> dict[str, Path]:
    root = _repo_path(repo_root, RUN_ROOT)
    receipts = root / "receipts"
    return {
        "root": root,
        "authorization": receipts / "chk1_cp200_to_chk4_authorization.json",
        "parent_exact": receipts / "parent_exact_merge.json",
        "parent_stage": receipts / "parent_stage.json",
        "sft_exact": receipts / "sft_exact_merge.json",
        "sft_stage": receipts / "sft_merge_stage.json",
        "lock": root / ".branch.lock",
    }


@contextmanager
def _branch_lock(repo_root: Path) -> Iterator[None]:
    lock_path = _receipt_paths(repo_root)["lock"]
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_symlink():
        raise Chk4BranchError("branch lock must not be a symlink")
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Chk4BranchError(
                "another chk4 branch operation owns the lock"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _acquire_training_lock(repo_root: Path) -> None:
    """Hold the branch lock across the accelerate process lifetime."""

    global _EXEC_LOCK_HANDLE
    if _EXEC_LOCK_HANDLE is not None:
        raise Chk4BranchError("the current process already owns the chk4 branch lock")
    lock_path = _receipt_paths(repo_root)["lock"]
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.is_symlink():
        raise Chk4BranchError("branch lock must not be a symlink")
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise Chk4BranchError("another chk4 branch operation owns the lock") from exc
    # Python descriptors are non-inheritable by default.  The accelerate
    # launcher replaces this process, so explicitly preserve this descriptor.
    os.set_inheritable(handle.fileno(), True)
    _EXEC_LOCK_HANDLE = handle


def _authorization(
    repo_root: Path,
    preflight_record: Mapping[str, Any],
    base: Mapping[str, Any],
    adapter: Mapping[str, Any],
) -> dict[str, Any]:
    bindings = {
        "base_model": dict(base),
        "source_checkpoint_200_adapter": dict(adapter),
        "release_manifest": dict(preflight_record["release_manifest"]),
        "configs": dict(preflight_record["configs"]),
        "run_root": str(_receipt_paths(repo_root)["root"]),
    }
    grpo_release_manifest = preflight_record.get("grpo_release_manifest")
    if grpo_release_manifest is not None:
        bindings["grpo_release_manifest"] = dict(grpo_release_manifest)
    reused_parent = preflight_record.get("reused_parent")
    if reused_parent is not None:
        bindings["reused_parent"] = dict(reused_parent)
    risk_acknowledgements = [
        "the prior cp200 merged artifact is chk2-scoped and is not reused",
        "the chk1 clean-v2 source-only semantic audit previously failed",
        "the chk4 release is core-only with seven out-of-support evaluation actions",
        "an independent source-only semantic judge was not run for the chk4 release",
    ]
    if ACTIVE_PROFILE.sft_dataset_role == CHK4_PRE2009_SFT_ROLE:
        risk_acknowledgements = [
            "the prior cp200 merged artifact is chk2-scoped and is not reused",
            "the chk1 clean-v2 source-only semantic audit previously failed",
            "the pre-2009 augmentation changes train only and inherits validation and test unchanged",
            "the fixed 4/2/2 schedule repeats minority-direction sources up to the manifest cap",
        ]
    payload = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": "authorized",
        "authorization_basis": "explicit_user_instruction",
        "branch_id": BRANCH_ID,
        "scope": {
            "allowed": [
                "chk4_decision_sft_parent",
                "chk4_decision_sft_training",
                "chk4_decision_sft_merge",
                "chk4_decision_grpo_training",
            ],
            "not_authorized": [
                "canonical_retrain_v2_dag",
                "chk2_parent_model",
                "chk3_parent_model",
            ],
        },
        "risk_acknowledgements": risk_acknowledgements,
        "bindings": bindings,
    }
    runtime_overrides = _profile_runtime_overrides(ACTIVE_PROFILE)
    if runtime_overrides is not None:
        payload["profile"] = PROFILE_NAME
        payload["runtime_overrides"] = runtime_overrides
    return {**payload, "authorization_sha256": _sha256_text(_canonical_json(payload))}


def _profile_runtime_overrides(profile: BranchProfile) -> dict[str, Any] | None:
    if profile.name == DEFAULT_PROFILE_NAME:
        return None
    overrides = {
        "decision_sft": {
            "learning_rate": profile.sft_learning_rate,
            "max_steps": profile.sft_max_steps,
            "warmup_steps": profile.sft_warmup_steps,
            "eval_steps": profile.sft_eval_steps,
        },
        "decision_grpo": {
            "max_completion_length": profile.grpo_max_completion_length,
            "reward_funcs": [profile.grpo_reward_func],
        },
    }
    if profile.sft_train_sampler != "default":
        overrides["decision_sft"].update(
            {
                "dataset_role": profile.sft_dataset_role,
                "release_manifest_sha256": _profile_sft_release_sha256(profile),
                "train_sampler": profile.sft_train_sampler,
                "checkpoint_keep_steps": list(profile.sft_checkpoint_keep_steps),
            }
        )
    if profile.sft_num_train_epochs != 3:
        overrides["decision_sft"]["num_train_epochs"] = profile.sft_num_train_epochs
    if (
        profile.grpo_dataset_role != "decision_grpo"
        or profile.grpo_release_manifest is not None
        or profile.grpo_release_sha256 is not None
    ):
        overrides["decision_grpo"].update(
            {
                "dataset_role": profile.grpo_dataset_role,
                "release_manifest_sha256": _profile_grpo_release_sha256(profile),
            }
        )
    return overrides


def _validate_authorization(
    payload: Mapping[str, Any], *, profile: BranchProfile | None = None
) -> None:
    expected_profile = profile or ACTIVE_PROFILE
    value = dict(payload)
    digest = value.pop("authorization_sha256", None)
    if (
        payload.get("schema_version") != AUTHORIZATION_SCHEMA
        or payload.get("status") != "authorized"
        or payload.get("authorization_basis") != "explicit_user_instruction"
        or payload.get("branch_id") != expected_profile.branch_id
        or digest != _sha256_text(_canonical_json(value))
    ):
        raise Chk4BranchError("chk4 branch authorization receipt is invalid")
    expected_overrides = _profile_runtime_overrides(expected_profile)
    if expected_overrides is None:
        if "profile" in payload or "runtime_overrides" in payload:
            raise Chk4BranchError("core chk4 authorization profile scope drift")
    elif (
        payload.get("profile") != expected_profile.name
        or payload.get("runtime_overrides") != expected_overrides
    ):
        raise Chk4BranchError("chk4 authorization profile runtime drift")


def _merge_binding(
    *,
    kind: str,
    authorization_path: Path,
    authorization: Mapping[str, Any],
    config_path: Path,
    base: Mapping[str, Any],
    adapter: Mapping[str, Any],
    upstream: Mapping[str, Any] | None = None,
    training: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "schema_version": MERGE_BINDING_SCHEMA,
        "branch_id": BRANCH_ID,
        "merge_kind": kind,
        "scope_boundary": {
            "allowed": [
                "chk4_decision_sft_parent"
                if kind == "parent"
                else "chk4_decision_grpo_parent"
            ],
            "not_authorized": ["canonical_retrain_v2_dag", "chk2", "chk3"],
        },
        "authorization": {
            "path": str(authorization_path),
            "file_sha256": sha256_file(authorization_path),
            "payload_sha256": authorization["authorization_sha256"],
        },
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "base_model": dict(base),
        "source_adapter": dict(adapter),
        "release_manifest": {
            "path": str(_profile_sft_release_manifest(ACTIVE_PROFILE)),
            "sha256": _profile_sft_release_sha256(ACTIVE_PROFILE),
        },
    }
    if upstream is not None:
        binding["upstream_parent_stage"] = dict(upstream)
    if training is not None:
        binding["training_completion"] = dict(training)
    return binding


def _exact_evidence(
    *,
    base: Path,
    adapter: Path,
    merged: Path,
    config_path: Path,
    artifact_id: str,
) -> dict[str, Any]:
    evidence = verify_exact_lora_merge(
        base_model=base,
        adapter=adapter,
        merged_model=merged,
        training_config=config_path,
        base_artifact_id=f"{artifact_id}-base",
        merged_artifact_id=artifact_id,
    )
    if evidence.get("conclusion") != "exact_base_plus_adapter_merge_verified":
        raise Chk4BranchError("exact LoRA merge verification did not pass")
    validate_manifest_integrity(evidence)
    return evidence


def _receipt_descriptor(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    integrity = payload.get("integrity")
    payload_sha = None
    if isinstance(integrity, Mapping):
        payload_sha = integrity.get("payload_sha256")
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "payload_sha256": payload_sha,
    }


def _write_stage_receipt(
    path: Path,
    *,
    stage: str,
    config_path: Path,
    merged: Path,
    attestation_binding: Mapping[str, Any],
    exact_path: Path,
    exact_payload: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": STAGE_RECEIPT_SCHEMA,
        "status": "passed",
        "branch_id": BRANCH_ID,
        "stage": stage,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "merged_artifact": fingerprint_artifact_path(merged),
        "merge_attestation": {
            "path": str(merged / "merge_attestation.json"),
            "file_sha256": sha256_file(merged / "merge_attestation.json"),
            "binding_sha256": _sha256_text(_canonical_json(attestation_binding)),
        },
        "exact_merge_evidence": _receipt_descriptor(exact_path, exact_payload),
    }
    receipt = {**payload, "receipt_sha256": _sha256_text(_canonical_json(payload))}
    _write_readonly_json(path, receipt)
    return receipt


def _parent_reuse_receipt(
    *,
    repo_root: Path,
    config_path: Path,
    authorization_path: Path,
    authorization: Mapping[str, Any],
    reused_parent: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": PARENT_REUSE_RECEIPT_SCHEMA,
        "status": "passed",
        "profile": PROFILE_NAME,
        "branch_id": BRANCH_ID,
        "stage": "parent_reuse",
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "authorization": {
            "path": str(authorization_path),
            "file_sha256": sha256_file(authorization_path),
            "payload_sha256": authorization["authorization_sha256"],
        },
        "reused_parent": dict(reused_parent),
        "merged_artifact": dict(reused_parent["merged_artifact"]),
    }
    return {**payload, "receipt_sha256": _sha256_text(_canonical_json(payload))}


def _load_parent_reuse_receipt(repo_root: Path) -> dict[str, Any]:
    path = _receipt_paths(repo_root)["parent_stage"]
    receipt = _read_json(path, label="parent reuse receipt")
    unsigned = dict(receipt)
    digest = unsigned.pop("receipt_sha256", None)
    if (
        receipt.get("schema_version") != PARENT_REUSE_RECEIPT_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("profile") != PROFILE_NAME
        or receipt.get("branch_id") != BRANCH_ID
        or receipt.get("stage") != "parent_reuse"
        or digest != _sha256_text(_canonical_json(unsigned))
    ):
        raise Chk4BranchError("parent reuse receipt is invalid")
    config = receipt.get("config")
    config_path = _config_path(repo_root, PARENT_CONFIG)
    if (
        not isinstance(config, Mapping)
        or Path(str(config.get("path"))).resolve() != config_path
        or config.get("sha256") != sha256_file(config_path)
    ):
        raise Chk4BranchError("parent reuse config binding drift")
    authorization_descriptor = receipt.get("authorization")
    authorization_path = _receipt_paths(repo_root)["authorization"]
    authorization = _read_json(authorization_path, label="authorization")
    _validate_authorization(authorization)
    if (
        not isinstance(authorization_descriptor, Mapping)
        or Path(str(authorization_descriptor.get("path"))).resolve()
        != authorization_path
        or authorization_descriptor.get("file_sha256")
        != sha256_file(authorization_path)
        or authorization_descriptor.get("payload_sha256")
        != authorization.get("authorization_sha256")
    ):
        raise Chk4BranchError("parent reuse authorization binding drift")
    reused_parent = _reused_parent_binding(repo_root)
    if reused_parent is None or receipt.get("reused_parent") != reused_parent:
        raise Chk4BranchError("reused parent source lineage drift")
    merged = reused_parent["merged_artifact"]
    if receipt.get("merged_artifact") != merged:
        raise Chk4BranchError("parent reuse merged artifact binding drift")
    return receipt


def prepare_parent(repo_root: Path, *, execute: bool) -> dict[str, Any]:
    checked = preflight(repo_root)
    bundle = _config_bundle(repo_root)
    parent_config = bundle["paths"]["parent"]
    parent = bundle["payloads"]["parent"]
    base_path = _path_value(repo_root, parent, "model_name_or_path")
    adapter_path = _path_value(repo_root, parent, "output_dir")
    merged_path = _path_value(repo_root, parent, "peft_merged_model_path")
    plan = {
        "status": "planned" if not execute else "executing",
        "operation": "prepare-parent",
        "base": str(base_path),
        "adapter": str(adapter_path),
        "destination": str(merged_path),
        "execute": execute,
    }
    if not execute:
        return plan

    _verify_training_environment()
    with _branch_lock(repo_root):
        if ACTIVE_PROFILE.reused_parent_branch_id is not None:
            reused_parent = _reused_parent_binding(repo_root)
            if reused_parent is None or reused_parent != checked.get("reused_parent"):
                raise Chk4BranchError("reused parent changed after preflight")
            base = dict(reused_parent["sources"]["base_model"])
            adapter = dict(reused_parent["sources"]["adapter"])
            _require_equal(
                base.get("sha256"), EXPECTED_BASE_SHA256, label="chk0 fingerprint"
            )
            _require_equal(
                adapter.get("sha256"),
                EXPECTED_CP200_ADAPTER_SHA256,
                label="checkpoint-200 fingerprint",
            )
            _require_equal(
                Path(str(base.get("path"))).resolve(),
                base_path,
                label="reused parent base path",
            )
            _require_equal(
                Path(str(adapter.get("path"))).resolve(),
                adapter_path,
                label="reused parent adapter path",
            )
            _require_equal(
                Path(str(reused_parent["merged_artifact"]["path"])).resolve(),
                merged_path,
                label="reused parent merged path",
            )
            paths = _receipt_paths(repo_root)
            authorization = _authorization(repo_root, checked, base, adapter)
            _write_readonly_json(paths["authorization"], authorization)
            stored_auth = _read_json(paths["authorization"], label="authorization")
            _validate_authorization(stored_auth)
            receipt = _parent_reuse_receipt(
                repo_root=repo_root,
                config_path=parent_config,
                authorization_path=paths["authorization"],
                authorization=stored_auth,
                reused_parent=reused_parent,
            )
            _write_readonly_json(paths["parent_stage"], receipt)
            receipt = _load_parent_reuse_receipt(repo_root)
            return {**plan, "status": "passed", "receipt": receipt, "reused": True}

        base = fingerprint_artifact_path(base_path)
        adapter = fingerprint_artifact_path(adapter_path)
        _require_equal(base["sha256"], EXPECTED_BASE_SHA256, label="chk0 fingerprint")
        _require_equal(
            adapter["sha256"],
            EXPECTED_CP200_ADAPTER_SHA256,
            label="checkpoint-200 fingerprint",
        )
        paths = _receipt_paths(repo_root)
        authorization = _authorization(repo_root, checked, base, adapter)
        _write_readonly_json(paths["authorization"], authorization)
        stored_auth = _read_json(paths["authorization"], label="authorization")
        _validate_authorization(stored_auth)
        binding = _merge_binding(
            kind="parent",
            authorization_path=paths["authorization"],
            authorization=stored_auth,
            config_path=parent_config,
            base=base,
            adapter=adapter,
        )
        if merged_path.exists():
            verify_merge_attestation(merged_path, expected_binding=binding)
        else:
            merge_from_config(
                parent_config,
                repo_root=repo_root,
                merge_attestation_binding=binding,
            )
        verify_merge_attestation(merged_path, expected_binding=binding)
        evidence = _exact_evidence(
            base=base_path,
            adapter=adapter_path,
            merged=merged_path,
            config_path=parent_config,
            artifact_id="chk1-cp200-parent-for-chk4",
        )
        _write_readonly_json(paths["parent_exact"], evidence)
        receipt = _write_stage_receipt(
            paths["parent_stage"],
            stage="parent_merge",
            config_path=parent_config,
            merged=merged_path,
            attestation_binding=binding,
            exact_path=paths["parent_exact"],
            exact_payload=evidence,
        )
    return {**plan, "status": "passed", "receipt": receipt}


def _load_stage_receipt(
    path: Path, *, expected_stage: str, expected_branch_id: str | None = None
) -> dict[str, Any]:
    scoped_branch_id = expected_branch_id or BRANCH_ID
    scoped_profile = _profile_for_branch_id(scoped_branch_id)
    receipt = _read_json(path, label=f"{expected_stage} receipt")
    unsigned = dict(receipt)
    recorded_receipt_sha = unsigned.pop("receipt_sha256", None)
    if (
        receipt.get("schema_version") != STAGE_RECEIPT_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("branch_id") != scoped_branch_id
        or receipt.get("stage") != expected_stage
        or recorded_receipt_sha != _sha256_text(_canonical_json(unsigned))
    ):
        raise Chk4BranchError(f"{expected_stage} receipt is invalid")
    config = receipt.get("config")
    if not isinstance(config, Mapping):
        raise Chk4BranchError(f"{expected_stage} receipt has no config binding")
    config_path = Path(str(config.get("path"))).resolve()
    if sha256_file(config_path) != config.get("sha256"):
        raise Chk4BranchError(f"{expected_stage} config drift")
    exact = receipt.get("exact_merge_evidence")
    if not isinstance(exact, Mapping):
        raise Chk4BranchError(f"{expected_stage} exact evidence is missing")
    exact_path = Path(str(exact.get("path"))).resolve()
    if sha256_file(exact_path) != exact.get("file_sha256"):
        raise Chk4BranchError(f"{expected_stage} exact evidence drift")
    exact_payload = _read_json(exact_path, label=f"{expected_stage} exact evidence")
    validate_manifest_integrity(exact_payload)
    if exact_payload.get("conclusion") != "exact_base_plus_adapter_merge_verified":
        raise Chk4BranchError(f"{expected_stage} exact evidence did not pass")
    exact_integrity = exact_payload.get("integrity")
    if not isinstance(exact_integrity, Mapping) or exact.get(
        "payload_sha256"
    ) != exact_integrity.get("payload_sha256"):
        raise Chk4BranchError(f"{expected_stage} exact evidence payload drift")
    merged = receipt.get("merged_artifact")
    if not isinstance(merged, Mapping):
        raise Chk4BranchError(f"{expected_stage} merged artifact binding is missing")
    merged_path = Path(str(merged.get("path"))).resolve()
    observed_merged = fingerprint_artifact_path(merged_path)
    if observed_merged != dict(merged):
        raise Chk4BranchError(f"{expected_stage} merged artifact drift")
    exact_sources = exact_payload.get("sources")
    if not isinstance(exact_sources, Mapping) or exact_sources.get(
        "merged_model"
    ) != dict(merged):
        raise Chk4BranchError(
            f"{expected_stage} exact evidence does not bind the merged artifact"
        )
    expected_subject = {
        "parent_merge": "chk1-cp200-parent-for-chk4",
        "sft_merge": "chk4-decision-sft-warm-start",
    }.get(expected_stage)
    if exact_payload.get("subject_artifact_id") != expected_subject:
        raise Chk4BranchError(f"{expected_stage} exact evidence subject drift")
    metadata = exact_payload.get("metadata_evidence")
    training_config = (
        metadata.get("training_config") if isinstance(metadata, Mapping) else None
    )
    if (
        not isinstance(training_config, Mapping)
        or Path(str(training_config.get("path"))).resolve() != config_path
        or training_config.get("sha256") != config.get("sha256")
        or not all(
            value is True
            for value in dict(training_config.get("path_checks") or {}).values()
        )
        or not all(
            value is True
            for value in dict(
                training_config.get("lora_metadata_checks") or {}
            ).values()
        )
    ):
        raise Chk4BranchError(f"{expected_stage} exact evidence config drift")
    attestation = receipt.get("merge_attestation")
    if not isinstance(attestation, Mapping):
        raise Chk4BranchError(f"{expected_stage} merge attestation binding is missing")
    attestation_path = merged_path / "merge_attestation.json"
    if Path(str(attestation.get("path"))).resolve() != attestation_path or sha256_file(
        attestation_path
    ) != attestation.get("file_sha256"):
        raise Chk4BranchError(f"{expected_stage} merge attestation drift")
    attestation_payload = _read_json(
        attestation_path, label=f"{expected_stage} merge attestation"
    )
    attestation_binding = attestation_payload.get("binding")
    if not isinstance(attestation_binding, Mapping) or _sha256_text(
        _canonical_json(attestation_binding)
    ) != attestation.get("binding_sha256"):
        raise Chk4BranchError(f"{expected_stage} merge attestation binding drift")
    verify_merge_attestation(
        merged_path,
        expected_binding=attestation_binding,
    )
    expected_merge_kind = {
        "parent_merge": "parent",
        "sft_merge": "sft_warm_start",
    }.get(expected_stage)
    if (
        attestation_binding.get("schema_version") != MERGE_BINDING_SCHEMA
        or attestation_binding.get("branch_id") != scoped_branch_id
        or attestation_binding.get("merge_kind") != expected_merge_kind
        or not isinstance(attestation_binding.get("authorization"), Mapping)
        or not isinstance(attestation_binding.get("release_manifest"), Mapping)
        or attestation_binding["release_manifest"].get("sha256")
        != _profile_sft_release_sha256(scoped_profile)
    ):
        raise Chk4BranchError(f"{expected_stage} merge scope binding drift")
    return receipt


def _training_completion(
    repo_root: Path, config_path: Path, role: str
) -> dict[str, Any]:
    config = _read_yaml(config_path, label=f"{role} config")
    output = _path_value(repo_root, config, "output_dir")
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "resolved_runtime_config.json",
        "train_results.json",
        "trainer_state.json",
    }
    missing = sorted(name for name in required if not (output / name).is_file())
    if missing:
        raise Chk4BranchError(f"{role} training is incomplete; missing {missing}")
    state = _read_json(output / "trainer_state.json", label=f"{role} trainer state")
    epoch = state.get("epoch")
    global_step = state.get("global_step")
    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, (int, float))
        or not math.isfinite(float(epoch))
        or isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step <= 0
        or not _training_reached_end(
            config, epoch=float(epoch), global_step=global_step
        )
    ):
        raise Chk4BranchError(
            f"{role} trainer state has not reached the configured end"
        )
    results = _read_json(output / "train_results.json", label=f"{role} train results")
    train_loss = results.get("train_loss")
    if (
        isinstance(train_loss, bool)
        or not isinstance(train_loss, (int, float))
        or not math.isfinite(float(train_loss))
    ):
        raise Chk4BranchError(f"{role} train loss is not finite")
    runtime = _read_json(
        output / "resolved_runtime_config.json", label=f"{role} runtime config"
    )
    _validate_training_runtime(
        repo_root,
        config_path=config_path,
        config=config,
        output=output,
        role=role,
        runtime=runtime,
    )
    dataset = runtime.get("dataset")
    if not isinstance(dataset, Mapping):
        raise Chk4BranchError(f"{role} runtime dataset binding is missing")
    binding = dataset.get("chk4_decision")
    if not isinstance(binding, Mapping):
        raise Chk4BranchError(f"{role} runtime chk4 binding is missing")
    scope = binding.get("scope")
    manifest = binding.get("release_manifest")
    if (
        not isinstance(scope, Mapping)
        or scope.get("role") != _runtime_dataset_role(role)
        or not isinstance(manifest, Mapping)
        or manifest.get("sha256") != _runtime_release_sha256(role)
    ):
        raise Chk4BranchError(f"{role} runtime release binding drift")
    return {
        "role": role,
        "output_dir": str(output),
        "config_sha256": sha256_file(config_path),
        "adapter": fingerprint_artifact_path(output),
        "trainer_state_sha256": sha256_file(output / "trainer_state.json"),
        "runtime_config_sha256": sha256_file(output / "resolved_runtime_config.json"),
        "train_results_sha256": sha256_file(output / "train_results.json"),
        "global_step": global_step,
        "epoch": float(epoch),
        "train_loss": float(train_loss),
    }


def _training_reached_end(
    config: Mapping[str, Any], *, epoch: float, global_step: int
) -> bool:
    max_steps = int(config.get("max_steps", -1))
    if max_steps > 0:
        return global_step >= max_steps
    return epoch + 1e-6 >= float(config.get("num_train_epochs", 0))


def _validate_training_runtime(
    repo_root: Path,
    *,
    config_path: Path,
    config: Mapping[str, Any],
    output: Path,
    role: str,
    runtime: Mapping[str, Any],
) -> None:
    branch = runtime.get("chk4_standalone_branch")
    if (
        not isinstance(branch, Mapping)
        or branch.get("branch_id") != BRANCH_ID
        or branch.get("stage") != role
    ):
        raise Chk4BranchError(f"{role} runtime branch binding drift")
    if PROFILE_NAME != DEFAULT_PROFILE_NAME and branch.get("profile") != PROFILE_NAME:
        raise Chk4BranchError(f"{role} runtime branch profile drift")
    runtime_config = branch.get("config")
    if (
        not isinstance(runtime_config, Mapping)
        or Path(str(runtime_config.get("path"))).resolve() != config_path
        or runtime_config.get("sha256") != sha256_file(config_path)
    ):
        raise Chk4BranchError(f"{role} runtime training config drift")

    model = runtime.get("model")
    training = runtime.get("training")
    peft = runtime.get("peft")
    if not all(isinstance(item, Mapping) for item in (model, training, peft)):
        raise Chk4BranchError(f"{role} runtime model/training/peft binding is missing")
    if _repo_path(repo_root, str(model.get("model_name_or_path"))) != _path_value(
        repo_root, config, "model_name_or_path"
    ):
        raise Chk4BranchError(f"{role} runtime parent model drift")
    if _repo_path(repo_root, str(training.get("output_dir"))) != output:
        raise Chk4BranchError(f"{role} runtime output directory drift")
    dataset = runtime.get("dataset")
    chk4_binding = (
        dataset.get("chk4_decision") if isinstance(dataset, Mapping) else None
    )
    scope = chk4_binding.get("scope") if isinstance(chk4_binding, Mapping) else None
    release_binding = (
        chk4_binding.get("release_manifest")
        if isinstance(chk4_binding, Mapping)
        else None
    )
    if dataset is not None or (
        role == "decision_sft" and ACTIVE_PROFILE.sft_train_sampler != "default"
    ):
        if (
            not isinstance(scope, Mapping)
            or scope.get("role") != _runtime_dataset_role(role)
            or not isinstance(release_binding, Mapping)
            or release_binding.get("sha256") != _runtime_release_sha256(role)
        ):
            raise Chk4BranchError(f"{role} runtime release binding drift")

    expected_training = {
        "learning_rate": float(config["learning_rate"]),
        "num_train_epochs": float(config["num_train_epochs"]),
        "max_steps": int(config.get("max_steps", -1)),
        "optimizer": config["optim"],
        "lr_scheduler_type": config["lr_scheduler_type"],
        "warmup_ratio": float(config["warmup_ratio"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "gradient_checkpointing": bool(config["gradient_checkpointing"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(config["per_device_eval_batch_size"]),
        "seed": int(config["seed"]),
        "bf16": bool(config["bf16"]),
    }
    if "warmup_steps" in config:
        expected_training["warmup_steps"] = int(config["warmup_steps"])
    for key, expected in expected_training.items():
        observed = training.get(key)
        if isinstance(expected, float) and isinstance(observed, (int, float)):
            if isinstance(observed, bool) or not math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1e-15
            ):
                raise Chk4BranchError(f"{role} runtime training.{key} drift")
        elif observed != expected:
            raise Chk4BranchError(f"{role} runtime training.{key} drift")

    expected_peft = {
        "merged_model_path": str(config["peft_merged_model_path"]),
        "r": int(config["peft_r"]),
        "lora_alpha": int(config["peft_lora_alpha"]),
        "lora_dropout": float(config["peft_lora_dropout"]),
        "bias": config["peft_bias"],
        "target_modules": list(config["peft_target_modules"]),
    }
    for key, expected in expected_peft.items():
        observed = peft.get(key)
        if key == "merged_model_path":
            if _repo_path(repo_root, str(observed)) != _repo_path(repo_root, expected):
                raise Chk4BranchError(f"{role} runtime peft.{key} drift")
        elif isinstance(expected, float) and isinstance(observed, (int, float)):
            if isinstance(observed, bool) or not math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1e-15
            ):
                raise Chk4BranchError(f"{role} runtime peft.{key} drift")
        elif observed != expected:
            raise Chk4BranchError(f"{role} runtime peft.{key} drift")

    if role == "decision_sft" and ACTIVE_PROFILE.sft_train_sampler != "default":
        sampler_receipt = runtime.get("training_sampler")
        if not isinstance(sampler_receipt, Mapping):
            raise Chk4BranchError("decision_sft fixed-schedule receipt is missing")
        release_manifest = _repo_path(
            repo_root, _profile_sft_release_manifest(ACTIVE_PROFILE)
        )
        release = _read_json(
            release_manifest, label="hier-balanced runtime release manifest"
        )
        contract = release.get("sampler_contract")
        if not isinstance(contract, Mapping):
            raise Chk4BranchError("hier-balanced sampler contract is missing")
        receipt_release = sampler_receipt.get("release_manifest")
        receipt_schedule = sampler_receipt.get("schedule")
        receipt_train = sampler_receipt.get("train_file")
        receipt_source = sampler_receipt.get("sampler_source")
        receipt_runtime = sampler_receipt.get("runtime")
        if not all(
            isinstance(value, Mapping)
            for value in (
                receipt_release,
                receipt_schedule,
                receipt_train,
                receipt_source,
                receipt_runtime,
            )
        ):
            raise Chk4BranchError("fixed-schedule runtime receipt schema drift")
        expected_schedule_path = release_manifest.parent / str(
            contract["schedule_path"]
        )
        expected_train_path = release_manifest.parent / str(contract["train_path"])
        expected_rows = contract.get("schedule_rows")
        expected_steps = contract.get("optimizer_steps")
        expected_receipt_schema = (
            "manifest-fixed-schedule-runtime-v1"
            if ACTIVE_PROFILE.sft_train_sampler == "manifest_fixed_schedule_v1"
            else "manifest-fixed-schedule-runtime-v2"
        )
        if (
            isinstance(expected_rows, bool)
            or not isinstance(expected_rows, int)
            or expected_rows <= 0
            or isinstance(expected_steps, bool)
            or not isinstance(expected_steps, int)
            or expected_steps <= 0
            or contract.get("train_rows") != expected_rows
        ):
            raise Chk4BranchError("fixed-schedule manifest dimensions are invalid")
        if (
            sampler_receipt.get("schema_version") != expected_receipt_schema
            or sampler_receipt.get("type") != ACTIVE_PROFILE.sft_train_sampler
            or Path(str(receipt_release.get("path"))).resolve() != release_manifest
            or receipt_release.get("sha256")
            != _profile_sft_release_sha256(ACTIVE_PROFILE)
            or receipt_release.get("release_id") != release.get("release_id")
            or Path(str(receipt_schedule.get("path"))).resolve()
            != expected_schedule_path
            or receipt_schedule.get("sha256") != contract.get("schedule_sha256")
            or receipt_schedule.get("rows") != expected_rows
            or receipt_schedule.get("order_is_authoritative") is not True
            or Path(str(receipt_train.get("path"))).resolve() != expected_train_path
            or receipt_train.get("sha256") != contract.get("train_sha256")
            or receipt_train.get("rows") != expected_rows
            or dict(receipt_source) != sampler_source_binding()
            or receipt_runtime
            != {
                "world_size": 1,
                "visible_gpus": 1,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "effective_batch_size": 8,
                "max_steps": expected_steps,
                "shuffle_dataset": False,
                "trl_shuffle_dataset": False,
                "secondary_shuffle": False,
                "use_liger_kernel": False,
            }
        ):
            raise Chk4BranchError("decision_sft fixed-schedule runtime binding drift")
    elif runtime.get("training_sampler") is not None:
        raise Chk4BranchError(f"{role} has an unexpected fixed-schedule receipt")

    if role == "decision_grpo":
        generation = runtime.get("generation")
        rewards = runtime.get("rewards")
        if not isinstance(generation, Mapping) or not isinstance(rewards, Mapping):
            raise Chk4BranchError(
                "decision_grpo runtime generation/reward binding missing"
            )
        for key in (
            "max_prompt_length",
            "max_completion_length",
            "num_generations",
        ):
            if generation.get(key) != config.get(key):
                raise Chk4BranchError(f"decision_grpo runtime generation.{key} drift")
        if rewards.get("reward_funcs") != [ACTIVE_PROFILE.grpo_reward_func]:
            raise Chk4BranchError("decision_grpo runtime reward function drift")


def _training_resume_state(
    repo_root: Path, *, config_path: Path, role: str
) -> dict[str, Any]:
    config = _read_yaml(config_path, label=f"{role} config")
    output = _path_value(repo_root, config, "output_dir")
    if not output.exists():
        return {"mode": "fresh", "output_dir": str(output)}
    if output.is_symlink() or not output.is_dir():
        raise Chk4BranchError(f"{role} output is not a safe training directory")
    runtime_path = output / "resolved_runtime_config.json"
    runtime = _read_json(runtime_path, label=f"{role} runtime config")
    _validate_training_runtime(
        repo_root,
        config_path=config_path,
        config=config,
        output=output,
        role=role,
        runtime=runtime,
    )

    root_state_path = output / "trainer_state.json"
    if root_state_path.is_file():
        root_state = _read_json(root_state_path, label=f"{role} root trainer state")
        epoch = root_state.get("epoch")
        global_step = root_state.get("global_step")
        if (
            not isinstance(epoch, bool)
            and isinstance(epoch, (int, float))
            and math.isfinite(float(epoch))
            and not isinstance(global_step, bool)
            and isinstance(global_step, int)
            and _training_reached_end(
                config, epoch=float(epoch), global_step=global_step
            )
        ):
            raise Chk4BranchError(
                f"{role} training is already complete; refusing to relaunch"
            )

    checkpoints: dict[int, Path] = {}
    for candidate in output.iterdir():
        if (
            candidate.is_dir()
            and not candidate.is_symlink()
            and candidate.name.startswith("checkpoint-")
            and candidate.name.removeprefix("checkpoint-").isdigit()
        ):
            checkpoints[int(candidate.name.removeprefix("checkpoint-"))] = candidate
    if not checkpoints:
        raise Chk4BranchError(
            f"{role} output exists but has no resumable numeric checkpoint"
        )
    latest_step = max(checkpoints)
    latest = checkpoints[latest_step]
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
    }
    missing = sorted(name for name in required if not (latest / name).is_file())
    if missing:
        raise Chk4BranchError(
            f"{role} latest checkpoint-{latest_step} is incomplete; missing {missing}"
        )
    rng_files = (latest / "rng_state.pth", latest / "rng_state_0.pth")
    if not any(path.is_file() and not path.is_symlink() for path in rng_files):
        raise Chk4BranchError(
            f"{role} latest checkpoint-{latest_step} is incomplete; missing RNG state"
        )
    latest_state = _read_json(
        latest / "trainer_state.json", label=f"{role} latest checkpoint state"
    )
    if latest_state.get("global_step") != latest_step:
        raise Chk4BranchError(f"{role} latest checkpoint global_step drift")
    return {
        "mode": "resume",
        "output_dir": str(output),
        "checkpoint": str(latest),
        "global_step": latest_step,
    }


def _parent_ready(repo_root: Path) -> dict[str, Any]:
    if ACTIVE_PROFILE.reused_parent_branch_id is not None:
        return _load_parent_reuse_receipt(repo_root)
    paths = _receipt_paths(repo_root)
    return _load_stage_receipt(paths["parent_stage"], expected_stage="parent_merge")


def merge_sft(repo_root: Path, *, execute: bool) -> dict[str, Any]:
    preflight(repo_root)
    parent_receipt = _parent_ready(repo_root)
    bundle = _config_bundle(repo_root)
    config_path = bundle["paths"]["sft"]
    config = bundle["payloads"]["sft"]
    base_path = _path_value(repo_root, config, "model_name_or_path")
    adapter_path = _path_value(repo_root, config, "output_dir")
    merged_path = _path_value(repo_root, config, "peft_merged_model_path")
    training = _training_completion(repo_root, config_path, "decision_sft")
    plan = {
        "status": "planned" if not execute else "executing",
        "operation": "merge-sft",
        "base": str(base_path),
        "adapter": str(adapter_path),
        "destination": str(merged_path),
        "execute": execute,
    }
    if not execute:
        return plan

    _verify_training_environment()
    with _branch_lock(repo_root):
        paths = _receipt_paths(repo_root)
        authorization = _read_json(paths["authorization"], label="authorization")
        _validate_authorization(authorization)
        base = fingerprint_artifact_path(base_path)
        adapter = fingerprint_artifact_path(adapter_path)
        if adapter != training["adapter"]:
            raise Chk4BranchError("Decision-SFT adapter changed after completion check")
        parent_descriptor = {
            "path": str(paths["parent_stage"]),
            "file_sha256": sha256_file(paths["parent_stage"]),
            "merged_artifact_sha256": parent_receipt["merged_artifact"]["sha256"],
        }
        binding = _merge_binding(
            kind="sft_warm_start",
            authorization_path=paths["authorization"],
            authorization=authorization,
            config_path=config_path,
            base=base,
            adapter=adapter,
            upstream=parent_descriptor,
            training=training,
        )
        if merged_path.exists():
            verify_merge_attestation(merged_path, expected_binding=binding)
        else:
            merge_from_config(
                config_path,
                repo_root=repo_root,
                merge_attestation_binding=binding,
            )
        verify_merge_attestation(merged_path, expected_binding=binding)
        evidence = _exact_evidence(
            base=base_path,
            adapter=adapter_path,
            merged=merged_path,
            config_path=config_path,
            artifact_id="chk4-decision-sft-warm-start",
        )
        _write_readonly_json(paths["sft_exact"], evidence)
        receipt = _write_stage_receipt(
            paths["sft_stage"],
            stage="sft_merge",
            config_path=config_path,
            merged=merged_path,
            attestation_binding=binding,
            exact_path=paths["sft_exact"],
            exact_payload=evidence,
        )
    return {**plan, "status": "passed", "receipt": receipt}


def _sft_merge_ready(repo_root: Path) -> dict[str, Any]:
    return _load_stage_receipt(
        _receipt_paths(repo_root)["sft_stage"], expected_stage="sft_merge"
    )


def training_command(repo_root: Path, *, stage: str) -> dict[str, Any]:
    preflight(repo_root)
    versions = _verify_training_environment()
    if stage == "sft":
        _parent_ready(repo_root)
        module = "jobs.train.train_sft"
        config_path = _config_path(repo_root, SFT_CONFIG)
        role = "decision_sft"
    elif stage == "grpo":
        _sft_merge_ready(repo_root)
        module = "jobs.train.train_grpo"
        config_path = _config_path(repo_root, GRPO_CONFIG)
        role = "decision_grpo"
    else:
        raise Chk4BranchError("training stage must be sft or grpo")
    resume = _training_resume_state(repo_root, config_path=config_path, role=role)
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
        module,
        "--config",
        str(config_path),
    ]
    return {
        "schema_version": BRANCH_SCHEMA,
        "status": "ready",
        "stage": stage,
        "gpus": [1],
        "environment": "fomc_trainer",
        "environment_versions": versions,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "resume": resume,
        "argv": argv,
        "automatic_resume": "numerically_latest_checkpoint",
    }


def execute_training(repo_root: Path, *, stage: str) -> None:
    command = training_command(repo_root, stage=stage)
    _acquire_training_lock(repo_root)
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "1",
            "NCCL_P2P_DISABLE": "1",
            "NCCL_IB_DISABLE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "FOMC_CHK4_BRANCH_ID": BRANCH_ID,
            "FOMC_CHK4_BRANCH_PROFILE": PROFILE_NAME,
            "FOMC_CHK4_BRANCH_STAGE": (
                "decision_sft" if stage == "sft" else "decision_grpo"
            ),
            "FOMC_CHK4_BRANCH_CONFIG_PATH": command["config"]["path"],
            "FOMC_CHK4_BRANCH_CONFIG_SHA256": command["config"]["sha256"],
        }
    )
    os.chdir(repo_root)
    os.execvpe(command["argv"][0], command["argv"], environment)


def _training_state(repo_root: Path, config_path: Path, role: str) -> str:
    config = _read_yaml(config_path, label=f"{role} config")
    output = _path_value(repo_root, config, "output_dir")
    if not output.exists():
        return "not_started"
    try:
        _training_completion(repo_root, config_path, role)
    except Chk4BranchError:
        checkpoints = [
            int(path.name.split("-", 1)[1])
            for path in output.glob("checkpoint-[0-9]*")
            if path.is_dir() and path.name.split("-", 1)[1].isdigit()
        ]
        return (
            f"in_progress_latest_checkpoint_{max(checkpoints)}"
            if checkpoints
            else "incomplete"
        )
    return "complete"


def status(repo_root: Path) -> dict[str, Any]:
    bundle = _config_bundle(repo_root)
    configs = bundle["payloads"]
    paths = _receipt_paths(repo_root)
    parent = "not_prepared"
    if paths["parent_stage"].is_file():
        try:
            _parent_ready(repo_root)
        except (Chk4BranchError, FileNotFoundError, RuntimeError, ValueError):
            parent = "invalid"
        else:
            parent = "verified"
    sft_merge_state = "not_merged"
    if paths["sft_stage"].is_file():
        try:
            _sft_merge_ready(repo_root)
        except (Chk4BranchError, FileNotFoundError, RuntimeError, ValueError):
            sft_merge_state = "invalid"
        else:
            sft_merge_state = "verified"
    return {
        "schema_version": BRANCH_SCHEMA,
        "branch_id": BRANCH_ID,
        "parent": parent,
        "decision_sft": _training_state(
            repo_root, bundle["paths"]["sft"], "decision_sft"
        ),
        "decision_sft_merge": sft_merge_state,
        "decision_grpo": _training_state(
            repo_root, bundle["paths"]["grpo"], "decision_grpo"
        ),
        "paths": {
            "parent": str(
                _path_value(repo_root, configs["parent"], "peft_merged_model_path")
            ),
            "sft_adapter": str(_path_value(repo_root, configs["sft"], "output_dir")),
            "sft_merged": str(
                _path_value(repo_root, configs["sft"], "peft_merged_model_path")
            ),
            "grpo_adapter": str(_path_value(repo_root, configs["grpo"], "output_dir")),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("status")
    for name in ("prepare-parent", "merge-sft"):
        child = subparsers.add_parser(name)
        child.add_argument("--execute", action="store_true")
    launch = subparsers.add_parser("launch")
    launch.add_argument("--stage", choices=("sft", "grpo"), required=True)
    launch.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    try:
        if args.command == "preflight":
            result = preflight(repo_root)
        elif args.command == "status":
            result = status(repo_root)
        elif args.command == "prepare-parent":
            result = prepare_parent(repo_root, execute=args.execute)
        elif args.command == "merge-sft":
            result = merge_sft(repo_root, execute=args.execute)
        elif args.command == "launch" and args.execute:
            execute_training(repo_root, stage=args.stage)
            raise AssertionError("os.execvpe unexpectedly returned")
        elif args.command == "launch":
            result = training_command(repo_root, stage=args.stage)
        else:  # pragma: no cover - argparse makes this unreachable
            raise Chk4BranchError(f"unsupported command: {args.command}")
    except (
        Chk4BranchError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema_version": BRANCH_SCHEMA,
                    "status": "blocked",
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
