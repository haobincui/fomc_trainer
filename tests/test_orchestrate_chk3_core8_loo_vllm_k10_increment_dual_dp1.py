from __future__ import annotations

from pathlib import Path

import pytest

from jobs.eval import eval_chk3_core8_loo_vllm_k10_increment_dual_dp1 as runner
from jobs.eval import orchestrate_chk3_core8_loo_vllm_k10_increment_dual_dp1 as subject


def test_increment_profile_uses_canonical_cross_pipeline_physical_gpu_locks() -> None:
    previous_evaluation_id = subject.shared.EVALUATION_ID
    with subject.configured_shared_orchestrator():
        assert subject.shared.EVALUATION_ID == runner.EVALUATION_ID
        assert subject.shared.MODEL_ORDER == ("chk3",)
        assert subject.shared.MAX_NUM_SEQS == (16,)
        assert subject.shared.GPU_LOCKS == subject.GPU_LOCKS
    with runner.configured_shared_implementation():
        assert (
            runner.frozen.shared.GPU_LOCK_PATH_TEMPLATE
            == runner.GPU_LOCK_PATH_TEMPLATE
        )
    assert subject.shared.EVALUATION_ID == previous_evaluation_id

    assert runner.GPU_LOCK_PATH_TEMPLATE == (
        "/tmp/fomc_trainer_chk3_beta_core8_vllm_k5_gpu{index}.lock"
    )
    assert len(set(subject.GPU_LOCKS)) == 2
    assert subject.GPU_LOCKS == tuple(
        Path(runner.GPU_LOCK_PATH_TEMPLATE.format(index=index))
        for index in range(2)
    )


def test_supervisor_lock_fd_passes_actual_increment_inherited_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real supervisor/worker inode and canonical-path contract."""

    monkeypatch.setattr(runner.frozen.shared, "_external_gpu_processes", lambda index: [])
    lock_path = subject.GPU_LOCKS[0]
    handle = subject._open_lock(lock_path)
    try:
        with runner.inherited_gpu_lease(
            physical_gpu_index=0,
            inherited_lock_fd=handle.fileno(),
        ) as evidence:
            assert Path(evidence["lock_path"]) == lock_path.resolve()
            assert evidence["lock_fd"] == handle.fileno()
            assert evidence["lease_mode"] == "supervisor_preacquired_inherited_fd"
    finally:
        handle.close()


def test_increment_worker_is_single_gpu_and_uses_increment_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("VLLM_DP_RANK", "7")
    monkeypatch.setenv("NCCL_COMM_ID", "unsafe")
    environment = subject._worker_environment(
        physical_gpu_index=1,
        lock_fd=17,
        run_token="a" * 64,
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert not any(key.startswith("VLLM_DP_") for key in environment)
    assert not any(key.startswith("NCCL_") for key in environment)

    command = subject._runner_command(
        python_bin=Path("/python"),
        model_id="chk3",
        shard_id=1,
        cohort=Path("/cohort.json"),
        cohort_sha256="b" * 64,
        output_dir=Path("/output/shard1"),
        scope="infrastructure_smoke",
        max_num_seqs=16,
        control_dir=Path("/control"),
        run_token="a" * 64,
        lock_fd=17,
        resume=False,
        gpu_wait_timeout_seconds=10,
        gpu_poll_seconds=1,
    )
    assert subject.RUNNER_MODULE in command
    assert "--smoke" in command


def test_increment_policy_adapter_translates_runner_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_path: Path) -> dict:
        raise runner.Core8LooVllmK10IncrementError("policy drift")

    monkeypatch.setattr(runner, "load_execution_policy", fail)
    with pytest.raises(subject.ExecutionPolicyError, match="policy drift"):
        subject._load_policy(Path("/policy.json"))
