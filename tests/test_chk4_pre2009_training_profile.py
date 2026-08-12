from __future__ import annotations

import hashlib
import json
import os
import subprocess
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
from open_r1.trainer.dataset_release import verify_chk4_release_for_role


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE = branch.BRANCH_PROFILES[branch.PRE2009_BALANCED_PROFILE_NAME]
MANIFEST = REPO_ROOT / PROFILE.sft_release_manifest
EXPECTED_MANIFEST_SHA256 = (
    "5141e00569b94e4af96f030f46176e9cd409199a6f26ac1c63a2ce335fe1e2d6"
)


def _yaml(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _activate(monkeypatch) -> None:
    monkeypatch.setattr(branch, "ACTIVE_PROFILE", PROFILE)
    monkeypatch.setattr(branch, "PROFILE_NAME", PROFILE.name)
    monkeypatch.setattr(branch, "BRANCH_ID", PROFILE.branch_id)
    monkeypatch.setattr(branch, "PARENT_CONFIG", PROFILE.parent_config)
    monkeypatch.setattr(branch, "SFT_CONFIG", PROFILE.sft_config)
    monkeypatch.setattr(branch, "GRPO_CONFIG", PROFILE.grpo_config)
    monkeypatch.setattr(branch, "RUN_ROOT", PROFILE.run_root)


def test_pre2009_configs_parse_and_bind_release_schedule_and_isolated_outputs() -> None:
    sft_path = REPO_ROOT / PROFILE.sft_config
    grpo_path = REPO_ROOT / PROFILE.grpo_config
    parent_path = REPO_ROOT / PROFILE.parent_config
    sft_script, sft_training, sft_model, _sft_peft = TrlParser(
        (SFTScriptArguments, SFTConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(args=["--config", str(sft_path)])
    grpo_script, grpo_training, grpo_model, _grpo_peft = TrlParser(
        (GRPOScriptArguments, GRPOConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(args=["--config", str(grpo_path)])

    assert sft_script.dataset_chk4_role == "decision_sft_pre2009_balanced"
    assert sft_script.dataset_chk4_release_manifest_sha256 == EXPECTED_MANIFEST_SHA256
    assert sft_training.train_sampler == "manifest_fixed_schedule_v2"
    assert sft_training.shuffle_dataset is False
    assert sft_training.group_by_length is False
    assert sft_training.dataloader_drop_last is False
    assert sft_training.dataloader_num_workers == 0
    assert sft_training.per_device_train_batch_size == 1
    assert sft_training.gradient_accumulation_steps == 8
    assert sft_training.learning_rate == 1.0e-5
    assert sft_training.lr_scheduler_type.value == "cosine"
    assert sft_training.warmup_steps == 2
    assert sft_training.num_train_epochs == 1
    assert sft_training.max_steps == 39
    assert sft_training.eval_steps == 13
    assert sft_training.checkpoint_keep_every_n_steps == 0
    assert sft_training.checkpoint_keep_steps == [13, 26, 39]
    assert sft_training.max_length == 3072
    assert sft_model.load_in_4bit is True

    assert grpo_script.dataset_chk4_role == "decision_grpo_pre2009_balanced"
    assert grpo_script.dataset_chk4_release_manifest_sha256 == EXPECTED_MANIFEST_SHA256
    assert grpo_script.reward_funcs == ["decision_dense_v3"]
    assert grpo_training.max_prompt_length == 2560
    assert grpo_training.max_completion_length == 1024
    assert grpo_training.num_train_epochs == 3
    assert grpo_model.load_in_4bit is True

    parent = _yaml(parent_path)
    sft = _yaml(sft_path)
    grpo = _yaml(grpo_path)
    assert PROFILE.reused_parent_branch_id == branch.DEFAULT_BRANCH_ID
    assert parent["peft_merged_model_path"] == sft["model_name_or_path"]
    assert branch.DEFAULT_BRANCH_ID in str(sft["model_name_or_path"])
    assert PROFILE.branch_id in str(sft["output_dir"])
    assert PROFILE.branch_id in str(sft["peft_merged_model_path"])
    assert PROFILE.branch_id in str(grpo["output_dir"])
    assert PROFILE.branch_id in str(grpo["peft_merged_model_path"])
    assert sft["dataset_chk4_release_manifest"] == grpo["dataset_chk4_release_manifest"]


def test_real_pre2009_release_verifies_for_both_training_roles() -> None:
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == (
        EXPECTED_MANIFEST_SHA256
    )
    release = json.loads(MANIFEST.read_text(encoding="utf-8"))
    core_path = Path(release["parent_releases"]["core_v3"]["path"])
    if not core_path.is_absolute():
        core_path = REPO_ROOT / core_path
    core = json.loads((core_path / "release_manifest.json").read_text())
    model_path = Path(core["sources"]["tokenizer"]["path"])
    for role, physical, config_path in (
        (
            "decision_sft_pre2009_balanced",
            "decision_sft",
            REPO_ROOT / PROFILE.sft_config,
        ),
        (
            "decision_grpo_pre2009_balanced",
            "decision_grpo",
            REPO_ROOT / PROFILE.grpo_config,
        ),
    ):
        config = _yaml(config_path)
        verified = verify_chk4_release_for_role(
            dataset_dir=MANIFEST.parent / physical,
            manifest_path=MANIFEST,
            expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
            dataset_role=role,
            system_prompt=str(config["system_prompt"]),
            model_path=model_path,
        )
        assert verified["dataset_role"] == role
        assert verified["physical_dataset_role"] == physical
        assert set(verified["split_files"]) == {"train", "validation"}
        assert verified["test_verified_but_not_loaded"] is True
        assert verified["sampler_contract"]["type"] == ("manifest_fixed_schedule_v2")
        assert verified["sampler_contract"]["schedule_rows"] == 312
        assert verified["sampler_contract"]["optimizer_steps"] == 39


def test_pre2009_profile_real_cpu_preflight_and_unsigned_authorization_preview(
    monkeypatch,
) -> None:
    _activate(monkeypatch)
    run_root = REPO_ROOT / PROFILE.run_root
    before_exists = os.path.lexists(run_root)
    receipt_paths = branch._receipt_paths(REPO_ROOT)
    before_receipts = {
        name: path.read_bytes()
        for name, path in receipt_paths.items()
        if name != "root" and path.is_file()
    }
    bundle = branch._config_bundle(REPO_ROOT)
    branch._validate_config_contract(REPO_ROOT, bundle)
    checked = branch.preflight(REPO_ROOT)
    assert checked["status"] == "passed"
    assert checked["profile"] == PROFILE.name
    assert checked["branch_id"] == PROFILE.branch_id
    assert checked["authorization"] is None or checked["authorization"]["status"] == (
        "verified"
    )
    assert checked["release_manifest"]["sha256"] == EXPECTED_MANIFEST_SHA256
    assert checked["release_manifest"]["dataset_role"] == (
        "decision_sft_pre2009_balanced"
    )
    assert checked["release_manifest"]["sampler_contract"]["type"] == (
        "manifest_fixed_schedule_v2"
    )
    assert checked["release_manifest"]["sampler_contract"]["optimizer_steps"] == 39
    assert checked["release_manifest"]["required_runtime_topology"] == {
        "world_size": 1,
        "visible_gpus": 1,
        "gpu_ids": [1],
    }
    assert checked["release_manifest"]["checkpoint_milestones"] == [13, 26, 39]
    assert checked["grpo_release_manifest"]["dataset_role"] == (
        "decision_grpo_pre2009_balanced"
    )

    reused = checked["reused_parent"]
    authorization = branch._authorization(
        REPO_ROOT,
        checked,
        reused["sources"]["base_model"],
        reused["sources"]["adapter"],
    )
    branch._validate_authorization(authorization)
    assert authorization["runtime_overrides"]["decision_sft"] == {
        "learning_rate": 1.0e-5,
        "max_steps": 39,
        "warmup_steps": 2,
        "eval_steps": 13,
        "dataset_role": "decision_sft_pre2009_balanced",
        "release_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "train_sampler": "manifest_fixed_schedule_v2",
        "checkpoint_keep_steps": [13, 26, 39],
        "num_train_epochs": 1,
    }
    assert authorization["runtime_overrides"]["decision_grpo"] == {
        "max_completion_length": 1024,
        "reward_funcs": ["decision_dense_v3"],
        "dataset_role": "decision_grpo_pre2009_balanced",
        "release_manifest_sha256": EXPECTED_MANIFEST_SHA256,
    }
    assert os.path.lexists(run_root) is before_exists
    after_receipts = {
        name: path.read_bytes()
        for name, path in receipt_paths.items()
        if name != "root" and path.is_file()
    }
    assert after_receipts == before_receipts


def test_pre2009_launcher_profile_is_registered_and_gpu1_execute_gated() -> None:
    launcher = REPO_ROOT / "run/retrain_v2/chk4_sft_grpo_branch.sh"
    text = launcher.read_text(encoding="utf-8")
    assert branch.PRE2009_BALANCED_PROFILE_NAME in text
    assert branch.PRE2009_BALANCED_BRANCH_ID in text
    assert 'gpu_gate.sh" 1' in text
    assert "--execute" in text
    subprocess.run(["bash", "-n", str(launcher)], check=True)
