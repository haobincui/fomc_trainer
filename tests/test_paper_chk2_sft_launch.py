from __future__ import annotations

from pathlib import Path

import pytest

from jobs.retrain_v2 import paper_chk2_sft_launch as launch


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_formal_config_is_exact_and_has_release_manifest_pin() -> None:
    config = launch._read_yaml(REPO_ROOT / launch.CONFIG)
    manifest_sha = config["dataset_paper_chk2_release_manifest_sha256"]

    assert manifest_sha == (
        "cb022dcd379e069e3d5ad54d7c7fdbc9e2ee29f958c85d5ff45d066f7728c097"
    )
    assert config == launch._expected_config(manifest_sha)
    assert config["dataset_paper_chk2_scope"] == launch.TRAINING_SCOPE
    assert config["dataset_train_split"] == "train"
    assert config["dataset_test_split"] == "validation"
    assert config["resume_from_checkpoint"] is None
    assert config["overwrite_output_dir"] is False
    assert config["max_steps"] == -1
    assert config["num_train_epochs"] == 3
    assert config["completion_only_loss"] is True
    assert config["packing"] is False
    assert config["max_length"] == 4096


def test_student_prompts_have_the_pinned_generator_hashes() -> None:
    assert launch._sha256_text(launch.STUDENT_SYSTEM_PROMPT) == (
        "4730a4ed585238547447ab850836db5a9fc67e5c3b1328b485c88d701ae78c4e"
    )
    assert launch._sha256_text(launch.STUDENT_USER_PROMPT_TEMPLATE) == (
        "423e79849cb66d6361c986f705ea4d4b16a03e28fec5826113eb9c8030a976d0"
    )
    assert "prompt, JSON transport" in launch.STUDENT_SYSTEM_PROMPT
    assert (
        "Do not discuss\ninstructions, prompts, JSON"
        not in launch.STUDENT_SYSTEM_PROMPT
    )


def test_schedule_is_exactly_three_epochs_and_sixty_optimizer_steps() -> None:
    schedule = launch._optimizer_steps(305)

    assert schedule == {
        "train_rows": 305,
        "world_size": 2,
        "per_rank_batches_per_epoch": 153,
        "gradient_accumulation_steps": 8,
        "updates_per_epoch": 20,
        "epochs": 3,
        "optimizer_steps": 60,
        "effective_global_batch": 16,
    }


def test_config_contract_accepts_only_the_formal_pinned_release() -> None:
    _path, _config, manifest_sha = launch._config_contract(REPO_ROOT)

    assert manifest_sha == (
        "cb022dcd379e069e3d5ad54d7c7fdbc9e2ee29f958c85d5ff45d066f7728c097"
    )


def test_training_command_is_dual_gpu_standard_sft_and_fresh_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder: true\n", encoding="utf-8")
    monkeypatch.setattr(
        launch,
        "_config_contract",
        lambda _root: (config_path, {}, "a" * 64),
    )

    command = launch.training_command(tmp_path)

    assert command["gpus"] == [0, 1]
    assert command["num_processes"] == 2
    assert command["resume"] == "forbidden_fresh_only_v1"
    assert command["argv"][:3] == [
        str(launch.EXPECTED_PYTHON),
        "-m",
        "accelerate.commands.launch",
    ]
    assert command["argv"][command["argv"].index("--num_processes") + 1] == "2"
    assert "jobs.train.train_sft" in command["argv"]
    assert "stage.sh" not in " ".join(command["argv"])


def test_gpu_gate_allows_recorded_co_tenants_and_high_utilization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = "\n".join(
        (
            "0, NVIDIA A30, 24576, 3500, 20540, 99",
            "1, NVIDIA A30, 24576, 3500, 20540, 98",
        )
    )

    def fake_smi(arguments):
        if arguments[0].startswith("--query-gpu"):
            return telemetry
        if arguments[0] == "--id=0":
            return "600451, /envs/wgan/python, 344\n600453, /envs/wgan/python, 344\n"
        return "600452, /envs/wgan/python, 344\n"

    monkeypatch.setattr(launch, "_run_nvidia_smi", fake_smi)

    rows = launch._verify_gpus()

    assert [row["index"] for row in rows] == [0, 1]
    assert rows[0]["utilization_percent"] == 99
    assert rows[0]["active_compute_pids"] == [600451, 600453]
    assert rows[0]["active_compute_process_memory_mib"] == 688
    assert rows[0]["co_tenancy_observed"] is True
    assert rows[0]["shared_memory_headroom_mib"] == 20540 - 17408
    assert rows[1]["active_compute_processes"] == [
        {
            "pid": 600452,
            "process_name": "/envs/wgan/python",
            "used_memory_mib": 344,
        }
    ]


def test_gpu_gate_fails_closed_below_shared_memory_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = "\n".join(
        (
            "0, NVIDIA A30, 24576, 3500, 20540, 99",
            "1, NVIDIA A30, 24576, 8000, 16500, 99",
        )
    )

    def fake_smi(arguments):
        if arguments[0].startswith("--query-gpu"):
            return telemetry
        return "600451, /envs/wgan/python, 344\n"

    monkeypatch.setattr(launch, "_run_nvidia_smi", fake_smi)

    with pytest.raises(
        launch.PaperChk2LaunchError,
        match="required 17408 MiB, observed 16500 MiB",
    ):
        launch._verify_gpus()


def test_shared_gpu_policy_is_pinned_to_measured_qlora_headroom() -> None:
    policy = launch._gpu_allocation_policy()

    assert policy == {
        "mode": "shared-remaining-memory-v1",
        "selected_gpu_ids": [0, 1],
        "existing_compute_processes_allowed": True,
        "high_utilization_allowed": True,
        "minimum_free_memory_per_gpu_mib": 17408,
        "reference_qlora_peak_reserved_mib": 14970,
        "safety_headroom_mib": 2438,
        "reference": "docs/summary/20260803T150258Z/fomc_retraining_v2_plan.md",
        "external_processes_must_not_be_signaled_or_stopped": True,
    }


def test_execute_records_preflight_and_runtime_cotenancy_without_process_actions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(launch, "RUN_ROOT", Path("run"))
    monkeypatch.setattr(launch, "OUTPUT_DIR", Path("run/adapters/chk2"))
    monkeypatch.setattr(launch, "PREFLIGHT_RECEIPT", Path("run/preflight.json"))
    monkeypatch.setattr(launch, "RUNTIME_LAUNCH_RECEIPT", Path("run/runtime.json"))
    gpu_snapshot = [
        {
            "index": gpu,
            "memory_free_mib": 20540,
            "utilization_percent": 99,
            "co_tenancy_observed": True,
            "active_compute_pids": [600451 + gpu],
            "active_compute_processes": [
                {
                    "pid": 600451 + gpu,
                    "process_name": "/envs/wgan/python",
                    "used_memory_mib": 344,
                }
            ],
        }
        for gpu in (0, 1)
    ]
    preflight_receipt = {
        "schema_version": "paper-chk2-minutes-sft-preflight-v2",
        "status": "ready",
        "resources": {"gpu_snapshot": {"gpus": gpu_snapshot}},
    }
    monkeypatch.setattr(launch, "preflight", lambda _root: preflight_receipt)
    monkeypatch.setattr(launch, "_verify_gpus", lambda: gpu_snapshot)
    monkeypatch.setattr(
        launch,
        "training_command",
        lambda _root: {
            "status": "ready",
            "branch_id": launch.BRANCH_ID,
            "gpus": [0, 1],
            "num_processes": 2,
            "config": {"path": "/config.yaml", "sha256": "a" * 64},
            "argv": ["/bin/true"],
            "resume": "forbidden_fresh_only_v1",
        },
    )

    def fake_execvpe(*_args):
        raise RuntimeError("exec intercepted")

    # Production never returns after chdir because exec replaces the process.
    # Keep the pytest worker's cwd stable when the intercepted exec does return.
    monkeypatch.setattr(launch.os, "chdir", lambda _path: None)
    monkeypatch.setattr(launch.os, "execvpe", fake_execvpe)

    with pytest.raises(RuntimeError, match="exec intercepted"):
        launch.execute_training(tmp_path)

    stored_preflight = launch._read_json(
        tmp_path / "run/preflight.json", label="preflight receipt"
    )
    stored_runtime = launch._read_json(
        tmp_path / "run/runtime.json", label="runtime receipt"
    )
    assert stored_preflight == preflight_receipt
    assert stored_runtime["gpu_snapshot_immediately_before_exec"]["gpus"] == (
        gpu_snapshot
    )
    assert stored_runtime["external_process_actions"] == {
        "signals_sent": [],
        "processes_stopped": [],
        "policy": "observe-only; never signal or stop co-tenant processes",
    }
    if launch._EXEC_LOCK is not None:
        launch._EXEC_LOCK.close()
        launch._EXEC_LOCK = None


def test_current_fomc_environment_allows_only_unrelated_extras() -> None:
    environment = launch._verify_environment()

    assert environment["core_versions"] == launch.EXPECTED_CORE_VERSIONS
    assert environment["required_freeze"]["distribution_count"] == 115
    assert environment["unrelated_extra_distribution_count"] == len(
        environment["unrelated_extra_distributions"]
    )
    assert environment["unrelated_extra_distribution_count"] >= 36
    assert environment["forbidden_distributions_present"] == []


def test_launcher_is_background_dual_gpu_create_only_and_not_resumable() -> None:
    path = REPO_ROOT / ("run/retrain_v2/start_paper_chk2_minutes_sft_v6_recovery.sh")
    source = path.read_text(encoding="utf-8")

    assert "nohup setsid" in source
    assert "export CUDA_VISIBLE_DEVICES=0,1" in source
    assert "export NCCL_P2P_DISABLE=1" in source
    assert "export NCCL_IB_DISABLE=1" in source
    assert "export PYTORCH_ALLOC_CONF=expandable_segments:True" in source
    assert "export TOKENIZERS_PARALLELISM=false" in source
    assert "training.pid" in source
    assert "SharedMinFreeMiB=17408" in source
    assert "preflight_receipt.json" in source
    assert "runtime_launch_receipt.json" in source
    assert "ExpectedSteps=60" in source
    assert "stage.sh" not in source
    assert "resume" not in source.lower()
    assert "merge_adapter" not in source


def test_status_is_read_only_and_reports_expected_horizon(tmp_path: Path) -> None:
    result = launch.status(tmp_path)

    assert result["status"] == "not_running"
    assert result["checkpoints"] == []
    assert result["expected_final_step"] == 60
    assert not (tmp_path / launch.RUN_ROOT).exists()
