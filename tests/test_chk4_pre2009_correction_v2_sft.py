from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from trl import TrlParser

from jobs.retrain_v2 import chk4_pre2009_correction_v2_sft as workflow
from jobs.retrain_v2.train_chk4_pre2009_correction_v2 import (
    CorrectionV2SFTConfig,
)
from open_r1.configs import LoraArguments, ModelConfig, SFTScriptArguments
from open_r1.trainer.fixed_schedule_sampler_correction_v2 import SAMPLER_TYPE


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    REPO_ROOT / "configs/retrain_v2/"
    "chk4_decision_sft_from_pre2009_cp38_correction_v2_lr2e6_steps6_20260811.yaml"
)


def test_custom_trl_parser_accepts_only_the_additive_v2_sampler() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    parser = TrlParser(
        (SFTScriptArguments, CorrectionV2SFTConfig, ModelConfig, LoraArguments)
    )
    # TrainingArguments checks GPU bf16 capability during construction. This
    # parser unit runs on CPU; the exact production mapping is frozen below.
    cpu_parse = config | {"bf16": False, "tf32": False, "use_cpu": True}
    _script, training, _model, _peft = parser.parse_dict(cpu_parse)
    assert training.train_sampler == SAMPLER_TYPE
    assert training.max_steps == 6
    assert training.gradient_accumulation_steps == 8

    with pytest.raises(ValueError, match="correction-v2 requires train_sampler"):
        CorrectionV2SFTConfig(output_dir="/tmp/unused", train_sampler="default")


def test_config_freezes_fresh_six_step_cp38_correction_contract() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config == workflow._expected_config(
        "54bc364e1ad99fe7f3a7d981c526bdbba2c6785c5221ff643cd4b7bbf892e427"
    )
    assert config["model_name_or_path"].endswith(
        "selected_sft_checkpoints/checkpoint-38/merged/chk4_sft"
    )
    assert config["dataset_chk4_role"] == "decision_sft_pre2009_correction_v2"
    assert config["train_sampler"] == SAMPLER_TYPE
    assert config["learning_rate"] == 2.0e-6
    assert config["warmup_steps"] == 1
    assert config["lr_scheduler_type"] == "cosine"
    assert config["per_device_train_batch_size"] == 1
    assert config["gradient_accumulation_steps"] == 8
    assert config["max_steps"] == 6
    assert config["checkpoint_keep_steps"] == [2, 4, 6]
    assert config["seed"] == config["data_seed"] == 1710190225
    assert config["resume_from_checkpoint"] is None
    assert config["overwrite_output_dir"] is False
    assert config["peft_r"] == 32
    assert config["dataset_chk4_release_manifest_sha256"] == (
        "54bc364e1ad99fe7f3a7d981c526bdbba2c6785c5221ff643cd4b7bbf892e427"
    )


def test_manifest_placeholder_blocks_preflight_before_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = workflow._read_yaml

    def draft(path: Path) -> dict[str, object]:
        config = original(path)
        config["dataset_chk4_release_manifest_sha256"] = workflow.MANIFEST_PLACEHOLDER
        return config

    monkeypatch.setattr(workflow, "_read_yaml", draft)
    with pytest.raises(
        workflow.CorrectionV2SFTWorkflowError,
        match="formal correction-v2 release SHA-256 has not been pinned",
    ):
        workflow._config_contract(REPO_ROOT)


def test_formal_config_contract_is_cpu_safe_and_schema_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    path, config, manifest_sha = workflow._config_contract(REPO_ROOT)
    assert path == CONFIG
    assert config == workflow._expected_config(manifest_sha)
    with pytest.raises(
        workflow.CorrectionV2SFTWorkflowError,
        match="schema lost config keys",
    ):
        workflow._validate_parser_schema(config | {"unrecognized_training_knob": True})


def test_authorization_inventory_and_environment_are_exact_and_dynamic() -> None:
    implementation = workflow._implementation(REPO_ROOT)
    assert tuple(implementation) == workflow.IMPLEMENTATION_RELATIVES
    assert all(
        record["sha256"] == workflow.sha256_file(record["path"])
        for record in implementation.values()
    )
    environment = workflow._environment_contract()
    assert environment["python"] == {
        "executable": "/home/haobin_cui/.conda/envs/fomc_trainer/bin/python",
        "version": "3.10.9",
    }
    assert (
        environment["package_versions"]
        == workflow.materializer.EXPECTED_PACKAGE_VERSIONS
    )
    assert set(environment["runtime_sources"]) == {
        "accelerate_launch",
        "trl_sft_trainer",
        "trl_parser",
        "transformers_trainer",
        "torch_sequential_sampler",
        "peft_lora_config",
        "python_executable",
    }


def test_training_command_uses_custom_entry_and_exact_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(workflow, "_verify_authorization", lambda _root: {})
    monkeypatch.setattr(
        workflow,
        "_config_contract",
        lambda _root: (CONFIG, {}, "a" * 64),
    )
    command = workflow.training_command(REPO_ROOT)
    assert command["argv"][0] == "/home/haobin_cui/.conda/envs/fomc_trainer/bin/python"
    assert "jobs.retrain_v2.train_chk4_pre2009_correction_v2" in command["argv"]
    assert "jobs.train.train_sft" not in command["argv"]
    assert command["resume"] == "forbidden_fresh_only_v2"


def test_launcher_is_exact_env_gpu1_only_and_never_resumes() -> None:
    launcher = (
        REPO_ROOT / "run/retrain_v2/chk4_pre2009_correction_v2_sft.sh"
    ).read_text(encoding="utf-8")
    assert (
        'TRAIN_PYTHON="/home/haobin_cui/.conda/envs/fomc_trainer/bin/python"'
        in launcher
    )
    assert 'export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}"' in launcher
    assert '"${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" 1' in launcher
    assert "export CUDA_VISIBLE_DEVICES=1" in launcher
    assert 'gpu_gate.sh" 0' not in launcher
    assert "resume" not in launcher.lower()


def test_authorization_mode_must_remain_0400(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(workflow, "RUN_ROOT", Path("run"))
    payload = {
        "schema_version": workflow.AUTHORIZATION_SCHEMA,
        "status": "authorized",
    }
    sealed = workflow.seal_manifest(payload)
    path = workflow._authorization_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(sealed) + "\n", encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setattr(workflow, "_authorization_payload", lambda _root: payload)
    with pytest.raises(
        workflow.CorrectionV2SFTWorkflowError, match="mode must be 0400"
    ):
        workflow._verify_authorization(tmp_path)


def test_execute_lock_is_inherited_across_accelerate_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExecIntercept(RuntimeError):
        pass

    captured: dict[str, object] = {}
    command = {
        "argv": [str(workflow.EXPECTED_PYTHON), "-m", "accelerate.commands.launch"],
        "config": {"path": "/tmp/config.yaml", "sha256": "a" * 64},
    }

    def fake_exec(
        executable: str, argv: list[str], environment: dict[str, str]
    ) -> None:
        captured.update(
            {"executable": executable, "argv": argv, "environment": environment}
        )
        raise ExecIntercept

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(workflow, "training_command", lambda _root: command)
    monkeypatch.setattr(workflow.os, "execvpe", fake_exec)
    previous = Path.cwd()
    try:
        with pytest.raises(ExecIntercept):
            workflow.execute_training(tmp_path)
        assert workflow._EXEC_LOCK is not None
        assert os.get_inheritable(workflow._EXEC_LOCK.fileno()) is True
        assert captured["executable"] == str(workflow.EXPECTED_PYTHON)
        assert captured["environment"]["PYTHONPATH"] == f"{tmp_path / 'src'}:{tmp_path}"
    finally:
        os.chdir(previous)
        if workflow._EXEC_LOCK is not None:
            workflow._EXEC_LOCK.close()
            workflow._EXEC_LOCK = None
