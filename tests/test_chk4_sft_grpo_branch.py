from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from jobs.retrain_v2 import chk4_sft_grpo_branch as branch
from open_r1.provenance import fingerprint_artifact_path
from open_r1.validator.loo_generation_spec import seal_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _activate_profile(
    monkeypatch: pytest.MonkeyPatch, profile_name: str
) -> branch.BranchProfile:
    profile = branch.BRANCH_PROFILES[profile_name]
    monkeypatch.setattr(branch, "ACTIVE_PROFILE", profile)
    monkeypatch.setattr(branch, "PROFILE_NAME", profile.name)
    monkeypatch.setattr(branch, "BRANCH_ID", profile.branch_id)
    monkeypatch.setattr(branch, "PARENT_CONFIG", profile.parent_config)
    monkeypatch.setattr(branch, "SFT_CONFIG", profile.sft_config)
    monkeypatch.setattr(branch, "GRPO_CONFIG", profile.grpo_config)
    monkeypatch.setattr(branch, "RUN_ROOT", profile.run_root)
    return profile


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    for relative in (
        branch.PARENT_CONFIG,
        branch.SFT_CONFIG,
        branch.GRPO_CONFIG,
        branch.RELEASE_MANIFEST,
    ):
        source = REPO_ROOT / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    bundle = branch._config_bundle(root)
    parent = bundle["payloads"]["parent"]
    base = branch._path_value(root, parent, "model_name_or_path")
    adapter = branch._path_value(root, parent, "output_dir")
    base.mkdir(parents=True)
    adapter.mkdir(parents=True)
    (base / "base.bin").write_bytes(b"base")
    (adapter / "adapter.bin").write_bytes(b"adapter")
    monkeypatch.setattr(
        branch,
        "EXPECTED_BASE_SHA256",
        fingerprint_artifact_path(base)["sha256"],
    )
    monkeypatch.setattr(
        branch,
        "EXPECTED_CP200_ADAPTER_SHA256",
        fingerprint_artifact_path(adapter)["sha256"],
    )
    monkeypatch.setattr(
        branch,
        "verify_chk4_decision_release",
        lambda **_: {
            "schema_version": "chk4-decision-runtime-binding-v1",
            "test_verified_but_not_loaded": True,
            "tokenizer_binding": {"bundle_sha256": "a" * 64},
        },
    )
    return root


def _fake_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    def merge_from_config(config_path, *, repo_root, merge_attestation_binding):
        config = branch._read_yaml(Path(config_path), label="fake config")
        destination = branch._path_value(
            Path(repo_root), config, "peft_merged_model_path"
        )
        if destination.exists():
            raise FileExistsError(destination)
        destination.mkdir(parents=True)
        (destination / "model.safetensors").write_bytes(b"merged")
        _write_json(
            destination / "merge_attestation.json",
            {"binding": merge_attestation_binding},
        )
        return {"status": "merged"}

    def verify_merge_attestation(merged_root, *, expected_binding):
        payload = json.loads(
            (Path(merged_root) / "merge_attestation.json").read_text(encoding="utf-8")
        )
        if payload["binding"] != expected_binding:
            raise ValueError("binding mismatch")
        return payload

    def exact_evidence(**kwargs):
        config_path = Path(kwargs["config_path"]).resolve()
        return seal_manifest(
            {
                "schema_version": "lora-merge-lineage-evidence-v1",
                "conclusion": "exact_base_plus_adapter_merge_verified",
                "subject_artifact_id": kwargs["artifact_id"],
                "sources": {
                    "base_model": fingerprint_artifact_path(kwargs["base"]),
                    "adapter": fingerprint_artifact_path(kwargs["adapter"]),
                    "merged_model": fingerprint_artifact_path(kwargs["merged"]),
                },
                "metadata_evidence": {
                    "training_config": {
                        "path": str(config_path),
                        "sha256": branch.sha256_file(config_path),
                        "path_checks": {
                            "model_name_or_path_matches_base": True,
                            "output_dir_matches_adapter": True,
                            "peft_merged_model_path_matches_merged": True,
                        },
                        "lora_metadata_checks": {
                            "bias_matches": True,
                            "lora_alpha_matches": True,
                            "r_matches": True,
                            "target_modules_match": True,
                        },
                    }
                },
            }
        )

    monkeypatch.setattr(branch, "merge_from_config", merge_from_config)
    monkeypatch.setattr(branch, "verify_merge_attestation", verify_merge_attestation)
    monkeypatch.setattr(branch, "_exact_evidence", exact_evidence)


def _write_complete_sft(root: Path) -> Path:
    bundle = branch._config_bundle(root)
    config = bundle["payloads"]["sft"]
    config_path = bundle["paths"]["sft"]
    output = branch._path_value(root, config, "output_dir")
    output.mkdir(parents=True)
    _write_json(output / "adapter_config.json", {"peft_type": "LORA"})
    (output / "adapter_model.safetensors").write_bytes(b"adapter")
    max_steps = int(config.get("max_steps", -1))
    global_step = max_steps if max_steps > 0 else 54
    epoch = 0.75 if max_steps > 0 else 3.0
    _write_json(
        output / "trainer_state.json",
        {"epoch": epoch, "global_step": global_step},
    )
    _write_json(output / "train_results.json", {"train_loss": 0.75})
    _write_json(
        output / "resolved_runtime_config.json",
        {
            "chk4_standalone_branch": {
                "branch_id": branch.BRANCH_ID,
                "stage": "decision_sft",
                "config": {
                    "path": str(config_path),
                    "sha256": branch.sha256_file(config_path),
                },
            },
            "model": {"model_name_or_path": config["model_name_or_path"]},
            "dataset": {
                "chk4_decision": {
                    "scope": {"role": "decision_sft"},
                    "release_manifest": {"sha256": branch.EXPECTED_RELEASE_SHA256},
                }
            },
            "training": {
                "output_dir": config["output_dir"],
                "learning_rate": config["learning_rate"],
                "num_train_epochs": config["num_train_epochs"],
                "max_steps": config["max_steps"],
                "optimizer": config["optim"],
                "lr_scheduler_type": config["lr_scheduler_type"],
                "warmup_ratio": config["warmup_ratio"],
                "gradient_accumulation_steps": config["gradient_accumulation_steps"],
                "gradient_checkpointing": config["gradient_checkpointing"],
                "per_device_train_batch_size": config["per_device_train_batch_size"],
                "per_device_eval_batch_size": config["per_device_eval_batch_size"],
                "seed": config["seed"],
                "bf16": config["bf16"],
            },
            "peft": {
                "merged_model_path": config["peft_merged_model_path"],
                "r": config["peft_r"],
                "lora_alpha": config["peft_lora_alpha"],
                "lora_dropout": config["peft_lora_dropout"],
                "bias": config["peft_bias"],
                "target_modules": config["peft_target_modules"],
            },
        },
    )
    if branch.PROFILE_NAME != branch.DEFAULT_PROFILE_NAME:
        runtime_path = output / "resolved_runtime_config.json"
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime["chk4_standalone_branch"]["profile"] = branch.PROFILE_NAME
        runtime["training"]["warmup_steps"] = config["warmup_steps"]
        _write_json(runtime_path, runtime)
    return output


def test_preflight_and_dry_runs_are_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_repo(tmp_path, monkeypatch)

    result = branch.preflight(root)
    plan = branch.prepare_parent(root, execute=False)

    assert result["status"] == "passed"
    assert plan["status"] == "planned"
    assert not (root / branch.RUN_ROOT).exists()
    with pytest.raises(branch.Chk4BranchError, match="receipt"):
        branch.training_command(root, stage="sft")


def test_parent_then_sft_merge_unlocks_stages_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_repo(tmp_path, monkeypatch)
    _fake_merge(monkeypatch)

    parent = branch.prepare_parent(root, execute=True)
    sft_command = branch.training_command(root, stage="sft")
    with pytest.raises(branch.Chk4BranchError, match="receipt"):
        branch.training_command(root, stage="grpo")

    _write_complete_sft(root)
    sft_merge = branch.merge_sft(root, execute=True)
    grpo_command = branch.training_command(root, stage="grpo")
    status = branch.status(root)

    assert parent["status"] == "passed"
    assert sft_command["argv"][-3:] == [
        "jobs.train.train_sft",
        "--config",
        str((root / branch.SFT_CONFIG).resolve()),
    ]
    assert sft_command["gpus"] == [1]
    assert sft_command["argv"][sft_command["argv"].index("--num_processes") + 1] == "1"
    assert sft_merge["status"] == "passed"
    assert grpo_command["argv"][-3:] == [
        "jobs.train.train_grpo",
        "--config",
        str((root / branch.GRPO_CONFIG).resolve()),
    ]
    assert grpo_command["gpus"] == [1]
    assert status["parent"] == "verified"
    assert status["decision_sft"] == "complete"
    assert status["decision_sft_merge"] == "verified"


def test_preflight_authorization_binding_is_stable_after_parent_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_repo(tmp_path, monkeypatch)
    before = branch.preflight(root)
    parent = branch._path_value(
        root,
        branch._config_bundle(root)["payloads"]["parent"],
        "peft_merged_model_path",
    )
    parent.mkdir(parents=True)
    (parent / "model.safetensors").write_bytes(b"parent")
    after = branch.preflight(root)

    assert before["release_manifest"] == after["release_manifest"]
    assert before["parent_model_tokenizer"] is None
    assert after["parent_model_tokenizer"] is not None


def test_stage_receipt_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_repo(tmp_path, monkeypatch)
    _fake_merge(monkeypatch)
    branch.prepare_parent(root, execute=True)
    receipt_path = branch._receipt_paths(root)["parent_stage"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "forged"
    receipt_path.chmod(0o600)
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    with pytest.raises(branch.Chk4BranchError, match="receipt is invalid"):
        branch.training_command(root, stage="sft")


def test_authorization_rejects_downstream_config_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_repo(tmp_path, monkeypatch)
    _fake_merge(monkeypatch)
    branch.prepare_parent(root, execute=True)
    sft_path = root / branch.SFT_CONFIG
    sft = branch._read_yaml(sft_path, label="SFT config")
    sft["learning_rate"] = 2.0e-6
    sft_path.write_text(yaml.safe_dump(sft, sort_keys=False), encoding="utf-8")

    with pytest.raises(branch.Chk4BranchError, match="drift"):
        branch.training_command(root, stage="sft")


def test_existing_foreign_training_directory_is_not_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_repo(tmp_path, monkeypatch)
    _fake_merge(monkeypatch)
    branch.prepare_parent(root, execute=True)
    config = branch._config_bundle(root)["payloads"]["sft"]
    output = branch._path_value(root, config, "output_dir")
    output.mkdir(parents=True)
    _write_json(output / "resolved_runtime_config.json", {"foreign": True})

    with pytest.raises(branch.Chk4BranchError, match="runtime branch binding drift"):
        branch.training_command(root, stage="sft")


def test_training_completion_rejects_wrong_runtime_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_repo(tmp_path, monkeypatch)
    output = _write_complete_sft(root)
    runtime_path = output / "resolved_runtime_config.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["dataset"]["chk4_decision"]["scope"]["role"] = "decision_grpo"
    _write_json(runtime_path, runtime)

    with pytest.raises(branch.Chk4BranchError, match="runtime release binding drift"):
        branch._training_completion(
            root, (root / branch.SFT_CONFIG).resolve(), "decision_sft"
        )


def test_shell_launcher_is_gpu1_only_and_execute_gated() -> None:
    launcher = REPO_ROOT / "run/retrain_v2/chk4_sft_grpo_branch.sh"
    text = launcher.read_text(encoding="utf-8")

    assert 'gpu_gate.sh" 1' in text
    assert "--execute" in text
    assert "fomc_trainer" in text
    assert "stage.sh" not in text
    assert 'tee -a "${LOG_PATH}"' in text
    assert "FOMC_CHK4_BRANCH_PROFILE:-core_v3" in text
    assert "warm_fix_v1" in text
    assert "warm_fix_lr1e5_v2" in text
    assert "warm_fix_lr1e5_steps24_v3" in text
    assert os.access(launcher, os.X_OK)


def test_profile_environment_defaults_old_and_selects_warm_fix() -> None:
    script = (
        "import json; from jobs.retrain_v2 import chk4_sft_grpo_branch as b; "
        "print(json.dumps({'profile': b.PROFILE_NAME, 'branch': b.BRANCH_ID, "
        "'sft': str(b.SFT_CONFIG), 'root': str(b.RUN_ROOT)}))"
    )
    base_env = {**os.environ, "PYTHONPATH": "src:."}
    base_env.pop(branch.PROFILE_ENV, None)
    default = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=base_env,
        capture_output=True,
        text=True,
        check=True,
    )
    warm = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env={**base_env, branch.PROFILE_ENV: branch.WARM_FIX_PROFILE_NAME},
        capture_output=True,
        text=True,
        check=True,
    )
    default_payload = json.loads(default.stdout)
    warm_payload = json.loads(warm.stdout)
    assert default_payload["profile"] == branch.DEFAULT_PROFILE_NAME
    assert default_payload["branch"] == branch.DEFAULT_BRANCH_ID
    assert warm_payload["profile"] == branch.WARM_FIX_PROFILE_NAME
    assert warm_payload["branch"] == branch.WARM_FIX_BRANCH_ID
    assert "warm_fix_v1" in warm_payload["sft"]
    assert "warm_fix_v1" in warm_payload["root"]

    lr1e5 = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env={
            **base_env,
            branch.PROFILE_ENV: branch.WARM_FIX_LR1E5_PROFILE_NAME,
        },
        capture_output=True,
        text=True,
        check=True,
    )
    lr1e5_payload = json.loads(lr1e5.stdout)
    assert lr1e5_payload["profile"] == branch.WARM_FIX_LR1E5_PROFILE_NAME
    assert lr1e5_payload["branch"] == branch.WARM_FIX_LR1E5_BRANCH_ID
    assert "warm_fix_lr1e5_v2" in lr1e5_payload["sft"]
    assert "warm_fix_lr1e5_v2" in lr1e5_payload["root"]

    steps24 = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env={
            **base_env,
            branch.PROFILE_ENV: branch.WARM_FIX_LR1E5_STEPS24_PROFILE_NAME,
        },
        capture_output=True,
        text=True,
        check=True,
    )
    steps24_payload = json.loads(steps24.stdout)
    assert steps24_payload["profile"] == branch.WARM_FIX_LR1E5_STEPS24_PROFILE_NAME
    assert steps24_payload["branch"] == branch.WARM_FIX_LR1E5_STEPS24_BRANCH_ID
    assert "warm_fix_lr1e5_steps24_v3" in steps24_payload["sft"]
    assert "warm_fix_lr1e5_steps24_v3" in steps24_payload["root"]

    invalid = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env={**base_env, branch.PROFILE_ENV: "unknown"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode != 0
    assert "unsupported FOMC_CHK4_BRANCH_PROFILE" in invalid.stderr


def test_steps24_authorization_binds_v3_reward_and_rejects_signed_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _activate_profile(monkeypatch, branch.WARM_FIX_LR1E5_STEPS24_PROFILE_NAME)
    preflight_record = {
        "release_manifest": {"sha256": branch.EXPECTED_RELEASE_SHA256},
        "configs": {
            "parent": {"sha256": "a" * 64},
            "sft": {"sha256": "b" * 64},
            "grpo": {"sha256": "c" * 64},
        },
        "reused_parent": {"source_branch_id": branch.DEFAULT_BRANCH_ID},
    }
    authorization = branch._authorization(
        tmp_path,
        preflight_record,
        {"path": "base", "sha256": "d" * 64},
        {"path": "adapter", "sha256": "e" * 64},
    )
    branch._validate_authorization(authorization)
    assert authorization["profile"] == profile.name
    assert authorization["branch_id"] == profile.branch_id
    assert authorization["runtime_overrides"] == branch._profile_runtime_overrides(
        profile
    )
    assert authorization["runtime_overrides"]["decision_grpo"] == {
        "max_completion_length": 1024,
        "reward_funcs": ["decision_dense_v3"],
    }

    drifted = dict(authorization)
    drifted["runtime_overrides"] = json.loads(
        json.dumps(authorization["runtime_overrides"])
    )
    drifted["runtime_overrides"]["decision_grpo"]["reward_funcs"] = [
        "decision_dense_v2"
    ]
    unsigned = dict(drifted)
    unsigned.pop("authorization_sha256")
    drifted["authorization_sha256"] = branch._sha256_text(
        branch._canonical_json(unsigned)
    )
    with pytest.raises(branch.Chk4BranchError, match="profile runtime drift"):
        branch._validate_authorization(drifted)


def test_steps24_runtime_validation_requires_profile_v3_reward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = _activate_profile(monkeypatch, branch.WARM_FIX_LR1E5_STEPS24_PROFILE_NAME)
    root = tmp_path / "repo"
    config_path = root / profile.grpo_config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / profile.grpo_config, config_path)
    config = branch._read_yaml(config_path, label="GRPO config")
    output = branch._path_value(root, config, "output_dir")
    runtime = {
        "chk4_standalone_branch": {
            "branch_id": profile.branch_id,
            "profile": profile.name,
            "stage": "decision_grpo",
            "config": {
                "path": str(config_path.resolve()),
                "sha256": branch.sha256_file(config_path),
            },
        },
        "model": {"model_name_or_path": config["model_name_or_path"]},
        "training": {
            "output_dir": config["output_dir"],
            "learning_rate": config["learning_rate"],
            "num_train_epochs": config["num_train_epochs"],
            "max_steps": config.get("max_steps", -1),
            "optimizer": config["optim"],
            "lr_scheduler_type": config["lr_scheduler_type"],
            "warmup_ratio": config["warmup_ratio"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "gradient_checkpointing": config["gradient_checkpointing"],
            "per_device_train_batch_size": config["per_device_train_batch_size"],
            "per_device_eval_batch_size": config["per_device_eval_batch_size"],
            "seed": config["seed"],
            "bf16": config["bf16"],
        },
        "peft": {
            "merged_model_path": config["peft_merged_model_path"],
            "r": config["peft_r"],
            "lora_alpha": config["peft_lora_alpha"],
            "lora_dropout": config["peft_lora_dropout"],
            "bias": config["peft_bias"],
            "target_modules": config["peft_target_modules"],
        },
        "generation": {
            "max_prompt_length": config["max_prompt_length"],
            "max_completion_length": config["max_completion_length"],
            "num_generations": config["num_generations"],
        },
        "rewards": {"reward_funcs": ["decision_dense_v3"]},
    }
    branch._validate_training_runtime(
        root,
        config_path=config_path.resolve(),
        config=config,
        output=output,
        role="decision_grpo",
        runtime=runtime,
    )
    runtime["rewards"]["reward_funcs"] = ["decision_dense_v2"]
    with pytest.raises(branch.Chk4BranchError, match="reward function drift"):
        branch._validate_training_runtime(
            root,
            config_path=config_path.resolve(),
            config=config,
            output=output,
            role="decision_grpo",
            runtime=runtime,
        )


@pytest.mark.parametrize(
    "profile_name",
    [
        branch.WARM_FIX_PROFILE_NAME,
        branch.WARM_FIX_LR1E5_PROFILE_NAME,
        branch.WARM_FIX_LR1E5_STEPS24_PROFILE_NAME,
    ],
)
def test_warm_fix_reuses_only_exact_verified_parent_with_new_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile_name: str
) -> None:
    root = _fixture_repo(tmp_path, monkeypatch)
    _fake_merge(monkeypatch)
    old_parent = branch.prepare_parent(root, execute=True)
    old_parent_path = branch._receipt_paths(root)["parent_stage"]
    old_parent_sha = branch.sha256_file(old_parent_path)

    warm = _activate_profile(monkeypatch, profile_name)
    for relative in (warm.parent_config, warm.sft_config, warm.grpo_config):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)

    def prohibited_merge(*_args, **_kwargs):
        raise AssertionError("warm-fix must not merge or copy the old parent")

    monkeypatch.setattr(branch, "merge_from_config", prohibited_merge)
    checked = branch.preflight(root)
    prepared = branch.prepare_parent(root, execute=True)
    ready = branch.training_command(root, stage="sft")
    paths = branch._receipt_paths(root)
    receipt = json.loads(paths["parent_stage"].read_text(encoding="utf-8"))
    authorization = json.loads(paths["authorization"].read_text(encoding="utf-8"))

    assert old_parent["status"] == "passed"
    assert checked["reused_parent"]["source_parent_stage"]["file_sha256"] == (
        old_parent_sha
    )
    assert prepared["reused"] is True
    assert receipt["schema_version"] == branch.PARENT_REUSE_RECEIPT_SCHEMA
    assert receipt["reused_parent"] == checked["reused_parent"]
    assert authorization["profile"] == profile_name
    assert authorization["runtime_overrides"]["decision_grpo"]["reward_funcs"] == [
        warm.grpo_reward_func
    ]
    assert authorization["bindings"]["reused_parent"] == checked["reused_parent"]
    assert paths["root"] == (root / warm.run_root).resolve()
    assert ready["resume"]["mode"] == "fresh"
    assert ready["gpus"] == [1]
    assert not branch._path_value(
        root, branch._config_bundle(root)["payloads"]["sft"], "output_dir"
    ).exists()
    assert not branch._path_value(
        root,
        branch._config_bundle(root)["payloads"]["sft"],
        "peft_merged_model_path",
    ).exists()


@pytest.mark.parametrize(
    ("profile_name", "expected_step"),
    [
        (branch.WARM_FIX_PROFILE_NAME, 12),
        (branch.WARM_FIX_LR1E5_PROFILE_NAME, 8),
        (branch.WARM_FIX_LR1E5_STEPS24_PROFILE_NAME, 24),
    ],
)
def test_warm_fix_max_steps_counts_as_complete_before_three_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile_name: str,
    expected_step: int,
) -> None:
    warm = _activate_profile(monkeypatch, profile_name)
    root = tmp_path / "repo"
    for relative in (warm.parent_config, warm.sft_config, warm.grpo_config):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination)
    output = _write_complete_sft(root)

    completed = branch._training_completion(
        root, (root / warm.sft_config).resolve(), "decision_sft"
    )
    assert completed["global_step"] == expected_step
    assert completed["epoch"] < 3.0
    assert output.is_dir()


def test_training_lock_blocks_a_second_branch_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    monkeypatch.setattr(branch, "_EXEC_LOCK_HANDLE", None)
    branch._acquire_training_lock(root)
    try:
        script = (
            "from pathlib import Path; "
            "from jobs.retrain_v2.chk4_sft_grpo_branch import _branch_lock; "
            f"root=Path({str(root)!r}); "
            "\nwith _branch_lock(root):\n    pass\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": "src:."},
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode != 0
        assert "another chk4 branch operation owns the lock" in completed.stderr
    finally:
        assert branch._EXEC_LOCK_HANDLE is not None
        branch._EXEC_LOCK_HANDLE.close()
        monkeypatch.setattr(branch, "_EXEC_LOCK_HANDLE", None)
