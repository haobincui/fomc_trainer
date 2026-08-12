from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import yaml
from trl import TrlParser

from jobs.retrain_v2 import chk4_grpo_relaxed_branch as branch
from open_r1.configs import GRPOConfig, GRPOScriptArguments, LoraArguments, ModelConfig
from open_r1.provenance import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _config(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _fake_bindings() -> dict[str, object]:
    digest = "a" * 64
    return {
        "source_sft": {
            "branch_id": branch.SOURCE_BRANCH_ID,
            "training": {
                "adapter": {"path": "/source/adapter", "sha256": digest},
                "runtime_config": {"path": "/source/runtime", "sha256": digest},
                "trainer_state": {"path": "/source/state", "sha256": digest},
            },
            "merge_stage": {"path": "/source/stage", "sha256": digest},
            "merge_attestation": {"path": "/source/attestation", "sha256": digest},
            "exact_merge_evidence": {"path": "/source/exact", "sha256": digest},
            "merged_artifact": {"path": "/source/merged", "sha256": digest},
        },
        "grpo_config": {"path": "/config", "sha256": digest},
        "source_sft_config": {"path": "/source/config", "sha256": digest},
        "release_manifest": {"path": "/release", "sha256": digest},
        "reward": {
            "source": {"path": "/reward", "sha256": digest},
            "config": {"path": "/config", "sha256": digest},
        },
        "paths": {
            "source_merged_sft": "/source/merged",
            "fresh_grpo_adapter": "/new/adapter",
            "fresh_grpo_merged": "/new/merged",
        },
    }


def _isolated_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, object]]:
    root = tmp_path / "repo"
    relative_config = Path("configs/grpo.yaml")
    config_path = root / relative_config
    config_path.parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / branch.GRPO_CONFIG, config_path)
    monkeypatch.setattr(branch, "GRPO_CONFIG", relative_config)
    monkeypatch.setattr(branch, "RUN_ROOT", Path("output/isolated-relaxed-v3"))
    monkeypatch.setattr(branch, "_binding_snapshot", lambda _root: _fake_bindings())
    monkeypatch.setattr(branch, "_verify_environment", lambda: {"trl": "1.2.0"})
    return root, config_path, _config(config_path)


def _runtime(
    root: Path, config_path: Path, config: dict[str, object]
) -> dict[str, object]:
    return {
        "chk4_standalone_branch": {
            "branch_id": branch.BRANCH_ID,
            "profile": branch.PROFILE_NAME,
            "stage": "decision_grpo",
            "config": {
                "path": str(config_path.resolve()),
                "sha256": sha256_file(config_path),
            },
        },
        "model": {"model_name_or_path": config["model_name_or_path"]},
        "dataset": {
            "chk4_decision": {
                "scope": {"role": "decision_grpo"},
                "release_manifest": {"sha256": branch.EXPECTED_RELEASE_SHA256},
            }
        },
        "generation": {
            key: config[key]
            for key in (
                "max_prompt_length",
                "max_completion_length",
                "num_generations",
                "temperature",
                "top_p",
            )
        },
        "rewards": {
            "reward_funcs": ["decision_dense_v3"],
            "reward_weights": [1.0],
        },
        "training": {
            "output_dir": config["output_dir"],
            "learning_rate": config["learning_rate"],
            "num_train_epochs": config["num_train_epochs"],
            "max_steps": config.get("max_steps", -1),
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "gradient_checkpointing": config["gradient_checkpointing"],
            "per_device_train_batch_size": config["per_device_train_batch_size"],
            "per_device_eval_batch_size": config["per_device_eval_batch_size"],
            "seed": config["seed"],
            "bf16": config["bf16"],
        },
        "peft": {"merged_model_path": config["peft_merged_model_path"]},
        "environment": {
            "cuda_visible_devices": "1",
            "n_gpu": 1,
            "world_size": 1,
        },
    }


def _checkpoint(output: Path, step: int, *, complete: bool = True) -> Path:
    checkpoint = output / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    names = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "training_args.bin",
        "rng_state.pth",
    )
    if not complete:
        names = names[:2]
    for name in names:
        if name == "trainer_state.json":
            _write_json(checkpoint / name, {"global_step": step, "epoch": 0.25})
        else:
            (checkpoint / name).write_bytes(b"x")
    return checkpoint


def test_repository_config_is_v3_single_gpu_effective_batch_eight() -> None:
    config_path = REPO_ROOT / branch.GRPO_CONFIG
    script, training, model, peft = TrlParser(
        (GRPOScriptArguments, GRPOConfig, ModelConfig, LoraArguments)
    ).parse_args_and_config(args=["--config", str(config_path)])
    bindings = branch._config_and_reward_bindings(REPO_ROOT)

    assert script.reward_funcs == ["decision_dense_v3"]
    assert training.max_completion_length == 1024
    assert training.max_prompt_length == 2560
    assert training.generation_batch_size == 8
    assert training.num_generations == 4
    assert training.per_device_train_batch_size == 1
    assert training.gradient_accumulation_steps == 8
    assert model.load_in_4bit is True
    assert peft.peft_r == 32
    assert bindings["reward"]["source"]["sha256"] == sha256_file(
        REPO_ROOT / branch.REWARD_SOURCE
    )
    assert bindings["reward"]["config"]["sha256"] == sha256_file(config_path)
    assert bindings["paths"]["source_merged_sft"].endswith(
        f"{branch.SOURCE_BRANCH_ID}/merged/chk4_sft"
    )
    assert branch.BRANCH_ID in bindings["paths"]["fresh_grpo_adapter"]


def test_real_preflight_fails_closed_until_exact_source_merge_exists() -> None:
    source_stage = REPO_ROOT / branch.SOURCE_ROOT / "receipts/sft_merge_stage.json"
    if source_stage.is_file():
        pytest.skip("source exact SFT merge has now been published")
    with pytest.raises(branch.RelaxedBranchError, match="source merge stage"):
        branch.preflight(REPO_ROOT)


def test_authorization_is_immutable_idempotent_and_binds_relaxation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, _ = _isolated_repo(tmp_path, monkeypatch)

    dry = branch.authorize(root, execute=False)
    first = branch.authorize(root, execute=True)
    second = branch.authorize(root, execute=True)
    paths = branch._paths(root)
    authorization = json.loads(paths["authorization"].read_text(encoding="utf-8"))
    receipt = json.loads(paths["source_receipt"].read_text(encoding="utf-8"))

    assert dry["status"] == "planned"
    assert first["status"] == second["status"] == "passed"
    assert authorization["authorization_basis"] == "explicit_user_instruction"
    assert authorization["explicit_relaxation_instruction"]["verbatim"] == (
        branch.USER_RELAXATION_INSTRUCTION
    )
    assert authorization["bindings"] == _fake_bindings()
    assert receipt["training_performed"] is False
    assert receipt["source_copied"] is False
    assert paths["authorization"].stat().st_mode & 0o777 == 0o400
    assert paths["source_receipt"].stat().st_mode & 0o777 == 0o400


def test_changed_source_or_reward_binding_invalidates_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, _ = _isolated_repo(tmp_path, monkeypatch)
    branch.authorize(root, execute=True)
    changed = _fake_bindings()
    changed["reward"]["source"]["sha256"] = "b" * 64
    monkeypatch.setattr(branch, "_binding_snapshot", lambda _root: changed)

    with pytest.raises(branch.RelaxedBranchError, match="authorization bindings drift"):
        branch.training_command(root)


def test_fresh_launch_is_gpu1_only_and_does_not_retrain_sft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, _ = _isolated_repo(tmp_path, monkeypatch)
    branch.authorize(root, execute=True)

    command = branch.training_command(root)

    assert command["resume"]["mode"] == "fresh"
    assert command["gpus"] == [1]
    assert command["cuda_visible_devices"] == "1"
    assert command["accelerate_num_processes"] == 1
    assert command["effective_train_batch"] == 8
    assert command["argv"][command["argv"].index("--num_processes") + 1] == "1"
    assert "jobs.train.train_grpo" in command["argv"]
    assert "jobs.train.train_sft" not in command["argv"]
    assert "--resume_from_checkpoint" not in command["argv"]


def test_resume_selects_explicit_numerically_latest_complete_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config_path, config = _isolated_repo(tmp_path, monkeypatch)
    branch.authorize(root, execute=True)
    output = branch._repo_path(root, str(config["output_dir"]))
    output.mkdir(parents=True)
    _write_json(output / "resolved_runtime_config.json", _runtime(root, config_path, config))
    _checkpoint(output, 2)
    latest = _checkpoint(output, 10)

    command = branch.training_command(root)

    assert command["resume"] == {
        "mode": "resume",
        "output_dir": str(output),
        "checkpoint": str(latest),
        "global_step": 10,
    }
    position = command["argv"].index("--resume_from_checkpoint")
    assert command["argv"][position + 1] == str(latest)


def test_incomplete_latest_or_foreign_runtime_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config_path, config = _isolated_repo(tmp_path, monkeypatch)
    branch.authorize(root, execute=True)
    output = branch._repo_path(root, str(config["output_dir"]))
    output.mkdir(parents=True)
    runtime = _runtime(root, config_path, config)
    _write_json(output / "resolved_runtime_config.json", runtime)
    _checkpoint(output, 3)
    _checkpoint(output, 4, complete=False)

    with pytest.raises(branch.RelaxedBranchError, match="checkpoint-4 is incomplete"):
        branch.training_command(root)

    shutil.rmtree(output / "checkpoint-4")
    runtime["rewards"]["reward_funcs"] = ["decision_dense_v2"]
    _write_json(output / "resolved_runtime_config.json", runtime)
    with pytest.raises(branch.RelaxedBranchError, match="source/reward/GPU"):
        branch.training_command(root)


def test_shell_launcher_is_execute_gated_and_gpu1_only() -> None:
    launcher = REPO_ROOT / "run/retrain_v2/chk4_grpo_relaxed_branch.sh"
    text = launcher.read_text(encoding="utf-8")

    assert 'gpu_gate.sh" 1' in text
    assert "CUDA_VISIBLE_DEVICES=0" not in text
    assert "--execute" in text
    assert "fomc_trainer" in text
    assert "jobs.train.train_sft" not in text
    assert os.access(launcher, os.X_OK)
