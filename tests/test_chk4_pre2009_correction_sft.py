from __future__ import annotations

from pathlib import Path

from jobs.retrain_v2 import chk4_pre2009_correction_sft as workflow
from jobs.retrain_v2 import chk4_pre2009_cp38_grpo_continuation as parent_workflow
from open_r1.provenance import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_correction_sft_config_binds_final_release_and_selected_cp38_parent():
    config_path, config = workflow._config_contract(REPO_ROOT)

    assert sha256_file(config_path) == workflow.CONFIG_SHA256
    assert config["dataset_chk4_release_manifest_sha256"] == (
        "1930991a24ce4cd615b0e96ef233991e2885b3ee9a45423686e1005181be0713"
    )
    assert config["model_name_or_path"] == str(parent_workflow.SELECTED_MODEL)
    assert config["train_sampler"] == "manifest_fixed_schedule_v3"
    assert config["learning_rate"] == 2.0e-6
    assert config["max_steps"] == 6
    assert config["eval_steps"] == config["save_steps"] == 1
    assert config["checkpoint_keep_steps"] == [2, 4, 6]
    assert config["seed"] == config["data_seed"] == 42
    assert config["resume_from_checkpoint"] is None
    assert config["overwrite_output_dir"] is False


def test_authorization_implementation_covers_real_runtime_and_launcher():
    implementation = workflow._implementation(REPO_ROOT)
    assert set(implementation) == {
        "jobs/retrain_v2/chk4_pre2009_correction_sft.py",
        "jobs/retrain_v2/materialize_chk4_pre2009_correction_release.py",
        "jobs/train/train_sft.py",
        "run/retrain_v2/chk4_pre2009_correction_sft.sh",
        "src/open_r1/configs.py",
        "src/open_r1/trainer/dataset_release.py",
        "src/open_r1/trainer/fixed_schedule_sampler_v3.py",
        "src/open_r1/trainer/sft_trainer.py",
        "src/open_r1/trainer/trainer.py",
    }
    assert all(record["sha256"] == sha256_file(record["path"]) for record in implementation.values())


def test_training_command_uses_standard_additive_dispatch(monkeypatch, tmp_path):
    monkeypatch.setattr(workflow, "_verify_authorization", lambda _root: {})
    monkeypatch.setattr(
        workflow,
        "_config_contract",
        lambda root: ((root / workflow.CONFIG).resolve(), {}),
    )

    command = workflow.training_command(tmp_path)

    assert command["gpu"] == 1
    assert command["resume"] == "forbidden_fresh_only_v1"
    assert "jobs.train.train_sft" in command["argv"]
    assert "jobs.train.train_chk4_pre2009_correction_sft" not in command["argv"]


def test_launcher_is_gpu1_only_and_create_only():
    launcher = (REPO_ROOT / "run/retrain_v2/chk4_pre2009_correction_sft.sh").read_text(
        encoding="utf-8"
    )

    assert '"${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" 1' in launcher
    assert "export CUDA_VISIBLE_DEVICES=1" in launcher
    assert "gpu_gate.sh\" 0" not in launcher
    assert "resume" not in launcher.lower()
    assert "authorize ${EXECUTE:+--execute}" in launcher
