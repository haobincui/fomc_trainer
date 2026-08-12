from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import yaml
from trl import TrlParser

from jobs.retrain_v2 import chk4_sft_grpo_branch as branch
from open_r1.configs import (
    GRPOConfig,
    GRPOScriptArguments,
    LoraArguments,
    ModelConfig,
    SFTConfig,
    SFTScriptArguments,
)
from open_r1.trainer.dataset_release import verify_chk4_hier_balanced_release


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = branch.BRANCH_PROFILES[branch.HIER_BALANCED_V5_PROFILE_NAME]
MANIFEST = REPO_ROOT / PROFILE.sft_release_manifest


def _yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_hier_balanced_configs_parse_and_bind_fixed_schedule() -> None:
    sft_path = REPO_ROOT / PROFILE.sft_config
    grpo_path = REPO_ROOT / PROFILE.grpo_config
    parent_path = REPO_ROOT / PROFILE.parent_config
    sft_script, sft_training, sft_model, _sft_peft = TrlParser(
        (SFTScriptArguments, SFTConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(args=["--config", str(sft_path)])
    grpo_script, grpo_training, grpo_model, _grpo_peft = TrlParser(
        (GRPOScriptArguments, GRPOConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(args=["--config", str(grpo_path)])

    assert sft_script.dataset_chk4_role == "decision_sft_hier_balanced"
    assert sft_script.dataset_chk4_release_manifest_sha256 == (
        "18c6f47ed0a085f7aef39d599e4baa30b34e5b560f3a594e7d5e3cf78bfde255"
    )
    assert sft_training.train_sampler == "manifest_fixed_schedule_v1"
    assert sft_training.learning_rate == 1.0e-5
    assert sft_training.max_steps == 24
    assert sft_training.warmup_steps == 2
    assert sft_training.eval_steps == 4
    assert sft_training.gradient_accumulation_steps == 8
    assert sft_training.checkpoint_keep_every_n_steps == 0
    assert sft_training.checkpoint_keep_steps == [12, 16, 20, 24]
    assert sft_training.group_by_length is False
    assert sft_training.dataloader_drop_last is False
    assert sft_training.shuffle_dataset is False
    assert sft_training.use_liger_kernel is False
    assert sft_model.load_in_4bit is True

    assert grpo_script.dataset_chk4_role == "decision_grpo"
    assert grpo_script.reward_funcs == ["decision_dense_v3"]
    assert grpo_training.max_completion_length == 1024
    assert grpo_model.load_in_4bit is True

    parent = _yaml(parent_path)
    sft = _yaml(sft_path)
    grpo = _yaml(grpo_path)
    assert PROFILE.reused_parent_branch_id == branch.DEFAULT_BRANCH_ID
    assert parent["peft_merged_model_path"] == sft["model_name_or_path"]
    assert branch.DEFAULT_BRANCH_ID in str(sft["model_name_or_path"])
    assert PROFILE.branch_id in str(sft["output_dir"])
    assert PROFILE.branch_id in str(grpo["output_dir"])
    assert grpo["dataset_chk4_release_manifest_sha256"] == (
        branch.EXPECTED_RELEASE_SHA256
    )


def test_training_verifier_replays_real_hier_balanced_release() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    sft = _yaml(REPO_ROOT / PROFILE.sft_config)
    observed_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    assert observed_sha == PROFILE.sft_release_sha256
    verified = verify_chk4_hier_balanced_release(
        dataset_dir=MANIFEST.parent / "decision_sft",
        manifest_path=MANIFEST,
        expected_manifest_sha256=observed_sha,
        dataset_role="decision_sft_hier_balanced",
        system_prompt=str(sft["system_prompt"]),
        model_path=Path(str(manifest["token_contract"]["tokenizer_path"])),
    )

    sampler = verified["sampler_contract"]
    assert sampler["schedule_sha256"] == (
        "d467503e8890ab20010026675f41b6b45bbb9d16502562be9038210d9d0e6504"
    )
    assert sampler["train_sha256"] == (
        "8ded0ec903414e1efdc357b153e399a4fa985490b36224b101d6b6417309016e"
    )
    assert sampler["world_size"] == 1
    assert sampler["effective_batch_size"] == 8
    assert sampler["optimizer_steps"] == 24
    assert sampler["order_is_authoritative"] is True
    assert sampler["secondary_shuffle_forbidden"] is True


def test_hier_balanced_profile_contract_and_launcher_are_registered(
    monkeypatch,
) -> None:
    monkeypatch.setattr(branch, "ACTIVE_PROFILE", PROFILE)
    monkeypatch.setattr(branch, "PROFILE_NAME", PROFILE.name)
    monkeypatch.setattr(branch, "BRANCH_ID", PROFILE.branch_id)
    monkeypatch.setattr(branch, "PARENT_CONFIG", PROFILE.parent_config)
    monkeypatch.setattr(branch, "SFT_CONFIG", PROFILE.sft_config)
    monkeypatch.setattr(branch, "GRPO_CONFIG", PROFILE.grpo_config)
    monkeypatch.setattr(branch, "RUN_ROOT", PROFILE.run_root)
    bundle = branch._config_bundle(REPO_ROOT)
    branch._validate_config_contract(REPO_ROOT, bundle)

    launcher = (REPO_ROOT / "run/retrain_v2/chk4_sft_grpo_branch.sh").read_text(
        encoding="utf-8"
    )
    assert branch.HIER_BALANCED_V5_PROFILE_NAME in launcher
    assert branch.HIER_BALANCED_V5_BRANCH_ID in launcher
    overrides = branch._profile_runtime_overrides(PROFILE)
    assert overrides is not None
    assert overrides["decision_sft"]["train_sampler"] == ("manifest_fixed_schedule_v1")
    assert overrides["decision_sft"]["checkpoint_keep_steps"] == [12, 16, 20, 24]


def test_pre2009_profile_role_routing_is_ready_without_registering_final_yaml() -> None:
    manifest = Path("dataset/processed/retrain_v2/pre2009/release_manifest.json")
    digest = "a" * 64
    profile = replace(
        PROFILE,
        sft_dataset_role="decision_sft_pre2009_balanced",
        sft_release_manifest=manifest,
        sft_release_sha256=digest,
        sft_train_sampler="manifest_fixed_schedule_v2",
        grpo_dataset_role="decision_grpo_pre2009_balanced",
        grpo_release_manifest=manifest,
        grpo_release_sha256=digest,
    )

    assert branch._runtime_dataset_role("decision_sft", profile) == (
        "decision_sft_pre2009_balanced"
    )
    assert branch._runtime_dataset_role("decision_grpo", profile) == (
        "decision_grpo_pre2009_balanced"
    )
    assert branch._runtime_dataset_directory(profile.sft_dataset_role) == "decision_sft"
    assert (
        branch._runtime_dataset_directory(profile.grpo_dataset_role) == "decision_grpo"
    )
    assert branch._profile_release_manifest(profile, "decision_sft") == manifest
    assert branch._profile_release_manifest(profile, "decision_grpo") == manifest
    assert branch._runtime_release_sha256("decision_sft", profile) == digest
    assert branch._runtime_release_sha256("decision_grpo", profile) == digest
    overrides = branch._profile_runtime_overrides(profile)
    assert overrides is not None
    assert overrides["decision_sft"]["train_sampler"] == ("manifest_fixed_schedule_v2")
    assert overrides["decision_grpo"] == {
        "max_completion_length": profile.grpo_max_completion_length,
        "reward_funcs": [profile.grpo_reward_func],
        "dataset_role": "decision_grpo_pre2009_balanced",
        "release_manifest_sha256": digest,
    }
