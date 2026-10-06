from __future__ import annotations

import copy
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs.eval import orchestrate_chk3_beta_core8_merged_vllm_k5_dual_dp1 as subject
from open_r1.validator.loo_generation_spec import seal_manifest


FORMAL_AUTHORIZATION = {
    "official_smoke_suite": {"path": "/sealed/smoke.json", "sha256": "s"},
    "benchmark_selection": {"path": "/sealed/selection.json", "sha256": "b"},
    "generation_gates": {
        "benchmark_candidate_exact_token_parity_8_12_16": True,
        "selected_chk1_smoke_exact_replay": True,
        "formal_generation_unblocked": True,
    },
}


def _write_sealed(path: Path, payload: dict) -> dict:
    value = seal_manifest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(subject._canonical(value) + "\n", encoding="utf-8")
    return subject._binding(path, payload_sha256=value["integrity"]["payload_sha256"])


def _ready_payload(
    *,
    token: str,
    model_id: str,
    shard_id: int,
    scope: str,
    max_num_seqs: int,
    formal_authorization: dict | None = None,
) -> dict:
    return {
        "schema_version": "chk3-beta-core8-vllm-k5-dual-dp1-ready-v1",
        "status": "ready",
        "evaluation_id": subject.EVALUATION_ID,
        "run_token": token,
        "model_id": model_id,
        "shard_id": shard_id,
        "physical_gpu_index": shard_id,
        "evaluation_scope": scope,
        "max_num_seqs": max_num_seqs,
        "formal_authorization": formal_authorization,
        "worker_pid": 1000 + shard_id,
        "engine_pid": 2000 + shard_id,
        "gpu_uuid": subject.GPU_IDENTITIES[shard_id]["uuid"],
        "gpu_pci_bus_id": subject.GPU_IDENTITIES[shard_id]["pci_bus_id"],
        "cuda_visible_devices": str(shard_id),
    }


def _go_payload(
    *,
    token: str,
    model_id: str,
    scope: str,
    max_num_seqs: int,
    epoch_ns: int,
    monotonic_ns: int,
    ready_bindings: dict[int, dict],
    formal_authorization: dict | None = None,
) -> dict:
    return {
        "schema_version": subject.GO_SCHEMA,
        "status": "released",
        "evaluation_id": subject.EVALUATION_ID,
        "run_token": token,
        "model_id": model_id,
        "evaluation_scope": scope,
        "max_num_seqs": max_num_seqs,
        "physical_gpu_indexes": [0, 1],
        "formal_authorization": formal_authorization,
        "go_epoch_ns": epoch_ns,
        "go_monotonic_ns": monotonic_ns,
        "ready_manifests": {
            f"shard{shard_id}": ready_bindings[shard_id] for shard_id in (0, 1)
        },
    }


def _done_payload(
    *,
    token: str,
    model_id: str,
    shard_id: int,
    scope: str,
    max_num_seqs: int,
    go_epoch_ns: int,
    go_monotonic_ns: int,
    resume_count: int,
    observed_go_monotonic_ns: int | None = None,
    formal_authorization: dict | None = None,
) -> dict:
    return {
        "schema_version": "chk3-beta-core8-vllm-k5-dual-dp1-done-v1",
        "status": "done",
        "evaluation_id": subject.EVALUATION_ID,
        "run_token": token,
        "model_id": model_id,
        "shard_id": shard_id,
        "physical_gpu_index": shard_id,
        "evaluation_scope": scope,
        "max_num_seqs": max_num_seqs,
        "formal_authorization": formal_authorization,
        "generation_started_epoch_ns": go_epoch_ns + 10,
        "generation_finished_epoch_ns": go_epoch_ns + 100,
        "observed_go_epoch_ns": go_epoch_ns,
        "observed_go_monotonic_ns": (
            go_monotonic_ns
            if observed_go_monotonic_ns is None
            else observed_go_monotonic_ns
        ),
        "generation_started_monotonic_ns": go_monotonic_ns + 10,
        "generation_finished_monotonic_ns": go_monotonic_ns + 1_000_000_000,
        "generated_output_tokens": 100 + shard_id,
        "completed_requests": 20,
        "resume_count": resume_count,
    }


def _shard_manifest(
    *,
    path: Path,
    ready_binding: dict,
    go_binding: dict,
    done_binding: dict,
    shard_id: int,
    resume_count: int,
) -> Path:
    _write_sealed(
        path,
        {
            "runtime": {
                "resume_count": resume_count,
                "engine_runtime": {
                    "ready_manifest": ready_binding,
                    "go_manifest": go_binding,
                    "done_manifest": done_binding,
                    "canonical_generated_output_tokens": 100 + shard_id,
                    "generated_output_tokens": 100 + shard_id,
                    "completed_requests_in_timed_session": 20,
                    "engine_session_resume_count": resume_count,
                },
            }
        },
    )
    return path


def _idle_sample_records() -> list[dict]:
    rows = [
        {
            "physical_gpu_index": shard_id,
            "uuid": subject.GPU_IDENTITIES[shard_id]["uuid"],
            "pci_bus_id": subject.GPU_IDENTITIES[shard_id]["pci_bus_id"],
            "memory_used_mib": 14,
            "memory_total_mib": 49140,
            "utilization_gpu_percent": 0,
        }
        for shard_id in (0, 1)
    ]
    return [
        {
            "sampled_at_utc": f"2026-08-16T00:00:0{sample_id}Z",
            "compute_processes": [],
            "gpus": copy.deepcopy(rows),
        }
        for sample_id in (1, 2)
    ]


def test_worker_environment_is_single_gpu_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_DP_RANK", "7")
    monkeypatch.setenv("VLLM_DP_CUSTOM", "unsafe")
    monkeypatch.setenv("NCCL_COMM_ID", "unsafe")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    env = subject._worker_environment(
        physical_gpu_index=1, lock_fd=17, run_token="a" * 64
    )
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["FOMC_PREACQUIRED_GPU_LOCK_FD"] == "17"
    assert not any(key.startswith("VLLM_DP_") for key in env)
    assert not any(key.startswith("NCCL_") for key in env)
    assert "WORLD_SIZE" not in env


def test_gpu_lock_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    target = tmp_path / "target.lock"
    target.touch()
    symlink = tmp_path / "symlink.lock"
    symlink.symlink_to(target)
    with pytest.raises(subject.DualDp1OrchestrationError, match="symlink"):
        subject._open_lock(symlink)

    hardlink = tmp_path / "hardlink.lock"
    os.link(target, hardlink)
    with pytest.raises(subject.DualDp1OrchestrationError, match="regular"):
        subject._open_lock(target)


def test_atomic_gpu_pair_releases_partial_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    locks = (tmp_path / "gpu0.lock", tmp_path / "gpu1.lock")
    monkeypatch.setattr(subject, "GPU_LOCKS", locks)
    held = subject._open_lock(locks[1])
    try:
        with pytest.raises(subject.DualDp1OrchestrationError, match="timed out"):
            subject._acquire_gpu_lock_pair(timeout_seconds=1, poll_seconds=1)
        probe = subject._open_lock(locks[0])
        probe.close()
    finally:
        held.close()


def test_ready_validation_requires_schema_pid_uuid_pci_and_cvd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = tmp_path / "control"
    token = "b" * 64
    ready = {
        "schema_version": "chk3-beta-core8-vllm-k5-dual-dp1-ready-v1",
        "status": "ready",
        "evaluation_id": subject.EVALUATION_ID,
        "run_token": token,
        "model_id": "chk1",
        "shard_id": 0,
        "physical_gpu_index": 0,
        "evaluation_scope": "infrastructure_smoke",
        "max_num_seqs": 8,
        "worker_pid": 123,
        "engine_pid": 456,
        "gpu_uuid": subject.GPU_IDENTITIES[0]["uuid"],
        "gpu_pci_bus_id": subject.GPU_IDENTITIES[0]["pci_bus_id"],
        "cuda_visible_devices": "0",
    }
    _write_sealed(control / "shard0.ready.json", ready)
    process = SimpleNamespace(pid=123, poll=lambda: None)
    values, _ = subject._wait_for_ready(
        processes={0: process},
        control_dir=control,
        run_token=token,
        model_id="chk1",
        scope="infrastructure_smoke",
        max_num_seqs=8,
        timeout_seconds=1,
    )
    assert values[0]["engine_pid"] == 456

    ready["gpu_pci_bus_id"] = "bad"
    (control / "shard0.ready.json").unlink()
    _write_sealed(control / "shard0.ready.json", ready)
    with pytest.raises(subject.DualDp1OrchestrationError, match="PID/GPU"):
        subject._wait_for_ready(
            processes={0: process},
            control_dir=control,
            run_token=token,
            model_id="chk1",
            scope="infrastructure_smoke",
            max_num_seqs=8,
            timeout_seconds=1,
        )


def test_pre_go_zero_exit_is_failure(tmp_path: Path) -> None:
    process = SimpleNamespace(pid=9, poll=lambda: 0)
    with pytest.raises(subject.DualDp1OrchestrationError, match="before GO"):
        subject._wait_for_ready(
            processes={0: process},
            control_dir=tmp_path,
            run_token="c" * 64,
            model_id="chk1",
            scope="infrastructure_smoke",
            max_num_seqs=8,
            timeout_seconds=1,
        )


def test_cleanup_signals_recorded_group_even_if_worker_leader_already_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[tuple[int, int]] = []

    def fake_killpg(pgid: int, sent_signal: int) -> None:
        signals.append((pgid, sent_signal))
        if sent_signal == 0:
            raise ProcessLookupError

    class ExitedProcess:
        pid = 4242

        @staticmethod
        def wait(timeout: float | None = None) -> int:
            return 7

    monkeypatch.setattr(subject.os, "killpg", fake_killpg)
    monkeypatch.setattr(subject, "_run_checked", lambda command: "")
    subject._terminate_process_groups({0: ExitedProcess()})  # type: ignore[arg-type]
    assert (4242, signal.SIGINT) in signals


def test_post_exit_gpu_evidence_requires_two_exact_idle_samples() -> None:
    evidence = subject._post_exit_gpu_evidence(_idle_sample_records())
    assert evidence["timeout_seconds"] == 60
    assert evidence["poll_seconds"] == 5
    subject._validate_post_exit_gpu_evidence(evidence, reconstructed=False)
    evidence["samples"][1]["gpus"][1]["uuid"] = "GPU-tampered"
    with pytest.raises(subject.DualDp1OrchestrationError, match="identity/idleness"):
        subject._validate_post_exit_gpu_evidence(evidence, reconstructed=False)
    with pytest.raises(subject.DualDp1OrchestrationError, match="falsely claims"):
        subject._validate_post_exit_gpu_evidence(evidence, reconstructed=True)


def test_go_and_timing_use_shared_monotonic_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(subject.time, "time_ns", lambda: 1_000_000_000)
    monkeypatch.setattr(subject.time, "monotonic_ns", lambda: 5_000_000_000)
    ready_bindings = {
        shard: _write_sealed(tmp_path / f"ready{shard}.json", {"shard": shard})
        for shard in (0, 1)
    }
    go, go_binding = subject._release_go(
        control_dir=tmp_path / "control",
        run_token="d" * 64,
        model_id="chk1",
        scope="infrastructure_smoke",
        max_num_seqs=8,
        ready_bindings=ready_bindings,
    )
    assert go["go_monotonic_ns"] == 5_000_000_000
    assert go["physical_gpu_indexes"] == [0, 1]
    logs = {}
    for shard in (0, 1):
        path = tmp_path / f"shard{shard}.log"
        path.write_text("ok\n", encoding="utf-8")
        logs[shard] = path
    done_values = {
        0: {
            "generation_finished_epoch_ns": 2_000_000_000,
            "generation_finished_monotonic_ns": 7_000_000_000,
            "generation_started_monotonic_ns": 5_100_000_000,
            "observed_go_epoch_ns": 1_000_000_000,
            "observed_go_monotonic_ns": 5_000_000_000,
            "generated_output_tokens": 100,
            "resume_count": 0,
        },
        1: {
            "generation_finished_epoch_ns": 2_100_000_000,
            "generation_finished_monotonic_ns": 7_500_000_000,
            "generation_started_monotonic_ns": 5_200_000_000,
            "observed_go_epoch_ns": 1_000_000_000,
            "observed_go_monotonic_ns": 5_000_000_000,
            "generated_output_tokens": 150,
            "resume_count": 0,
        },
    }
    timing_path = tmp_path / "timing.json"
    value = subject._seal_timing(
        path=timing_path,
        model_id="chk1",
        scope="infrastructure_smoke",
        max_num_seqs=8,
        run_token="d" * 64,
        go=go,
        go_binding=go_binding,
        ready_bindings=ready_bindings,
        done_values=done_values,
        done_bindings=ready_bindings,
        logs=logs,
        gpu_idle_evidence=[],
        processes={0: SimpleNamespace(pid=10), 1: SimpleNamespace(pid=11)},
        remediation_receipt={"sha256": "r"},
        gpu_idle_after_worker_exit=subject._post_exit_gpu_evidence(
            _idle_sample_records()
        ),
    )
    assert value["shared_timing"]["parallel_generation_wall_seconds"] == 2.5
    assert value["shared_timing"]["aggregate_generated_output_tokens"] == 250
    assert value["shared_timing"]["aggregate_output_tokens_per_second"] == 100.0
    assert value["gpu_idle_after_worker_exit_status"] == (
        subject.POST_EXIT_GPU_PASSED_STATUS
    )
    assert value["gpu_idle_after_worker_exit_claimed"] is True
    assert len(value["gpu_idle_after_worker_exit"]["samples"]) == 2


def test_smoke_partial_pair_fails_before_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "chk1"
    (model_root / "shard0").mkdir(parents=True)
    monkeypatch.setattr(
        subject.remediation,
        "load_and_validate_receipt",
        lambda path: {"receipt_binding": {"path": str(path), "sha256": "x"}},
    )
    with pytest.raises(subject.DualDp1OrchestrationError, match="fresh shard pair"):
        subject.run_model(
            python_bin=Path("/python"),
            model_id="chk1",
            cohort=tmp_path / "cohort.json",
            cohort_sha256="a" * 64,
            model_root=model_root,
            scope="infrastructure_smoke",
            max_num_seqs=8,
            allow_resume=False,
            gpu_wait_timeout_seconds=1,
            gpu_poll_seconds=1,
            ready_timeout_seconds=1,
        )


def test_formal_without_official_smoke_authorization_fails_before_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("GPU/receipt path must not be reached")

    monkeypatch.setattr(subject, "_acquire_gpu_lock_pair", forbidden)
    monkeypatch.setattr(subject.remediation, "load_and_validate_receipt", forbidden)
    model_root = tmp_path / "formal" / "chk1"
    with pytest.raises(
        subject.DualDp1OrchestrationError, match="requires the official"
    ):
        subject.run_model(
            python_bin=Path("/python"),
            model_id="chk1",
            cohort=tmp_path / "cohort.json",
            cohort_sha256="a" * 64,
            model_root=model_root,
            scope="formal_merged_panel",
            max_num_seqs=12,
            allow_resume=True,
            formal_authorization=None,
            gpu_wait_timeout_seconds=1,
            gpu_poll_seconds=1,
            ready_timeout_seconds=1,
        )
    assert not model_root.exists()


def test_worker_command_passes_scope_and_resume() -> None:
    command = subject._runner_command(
        python_bin=Path("/python"),
        model_id="chk3",
        shard_id=1,
        cohort=Path("/cohort"),
        cohort_sha256="a" * 64,
        output_dir=Path("/out"),
        scope="formal_merged_panel",
        max_num_seqs=12,
        control_dir=Path("/control"),
        run_token="f" * 64,
        lock_fd=19,
        resume=True,
        gpu_wait_timeout_seconds=10,
        gpu_poll_seconds=1,
        formal_authorization=Path("/smoke/manifest.json"),
    )
    assert command[command.index("--scope") + 1] == "formal_merged_panel"
    assert "--resume" in command
    assert "--smoke" not in command
    assert command[command.index("--formal-authorization") + 1] == (
        "/smoke/manifest.json"
    )


def test_reconstructs_fresh_completed_pair_without_claiming_new_idle_audit(
    tmp_path: Path,
) -> None:
    scope = "infrastructure_smoke"
    token = "1" * 64
    control = tmp_path / "control_fresh"
    ready_bindings = {
        shard_id: _write_sealed(
            control / f"shard{shard_id}.ready.json",
            _ready_payload(
                token=token,
                model_id="chk1",
                shard_id=shard_id,
                scope=scope,
                max_num_seqs=8,
            ),
        )
        for shard_id in (0, 1)
    }
    go_binding = _write_sealed(
        control / "go.json",
        _go_payload(
            token=token,
            model_id="chk1",
            scope=scope,
            max_num_seqs=8,
            epoch_ns=1_000_000_000,
            monotonic_ns=5_000_000_000,
            ready_bindings=ready_bindings,
        ),
    )
    manifests = []
    for shard_id in (0, 1):
        done_binding = _write_sealed(
            control / f"shard{shard_id}.done.json",
            _done_payload(
                token=token,
                model_id="chk1",
                shard_id=shard_id,
                scope=scope,
                max_num_seqs=8,
                go_epoch_ns=1_000_000_000,
                go_monotonic_ns=5_000_000_000,
                resume_count=0,
            ),
        )
        manifests.append(
            _shard_manifest(
                path=tmp_path / f"shard{shard_id}" / "manifest.json",
                ready_binding=ready_bindings[shard_id],
                go_binding=go_binding,
                done_binding=done_binding,
                shard_id=shard_id,
                resume_count=0,
            )
        )
        (control / f"shard{shard_id}.console.log").write_text(
            "sealed worker output\n", encoding="utf-8"
        )

    timing = subject._reconstruct_completed_pair_timing(
        timing_path=tmp_path / "timing.json",
        model_id="chk1",
        scope=scope,
        max_num_seqs=8,
        shard_manifests=manifests,
        remediation_receipt={"sha256": "receipt"},
    )
    assert timing["shared_timing"]["speed_measurement_valid_for_candidate_selection"]
    assert timing["shared_timing"]["aggregate_generated_output_tokens"] == 201
    assert timing["gpu_idle_before_spawn"] is None
    assert timing["gpu_idle_after_worker_exit"] is None
    assert timing["gpu_idle_after_worker_exit_claimed"] is False
    assert timing["gpu_idle_after_worker_exit_status"] == (
        subject.POST_EXIT_GPU_UNAVAILABLE_STATUS
    )
    assert timing["timing_reconstruction"] == {
        "no_new_gpu_work": True,
        "reconstructed_from_two_deeply_validated_sealed_shards": True,
    }


def test_reconstructs_paired_formal_resume_as_speed_ineligible(tmp_path: Path) -> None:
    scope = "formal_merged_panel"
    token = "2" * 64
    control = tmp_path / "control_paired_resume"
    ready_bindings = {
        shard_id: _write_sealed(
            control / f"shard{shard_id}.ready.json",
            _ready_payload(
                token=token,
                model_id="chk3",
                shard_id=shard_id,
                scope=scope,
                max_num_seqs=12,
                formal_authorization=FORMAL_AUTHORIZATION,
            ),
        )
        for shard_id in (0, 1)
    }
    go_binding = _write_sealed(
        control / "go.json",
        _go_payload(
            token=token,
            model_id="chk3",
            scope=scope,
            max_num_seqs=12,
            epoch_ns=2_000_000_000,
            monotonic_ns=6_000_000_000,
            ready_bindings=ready_bindings,
            formal_authorization=FORMAL_AUTHORIZATION,
        ),
    )
    manifests = []
    for shard_id in (0, 1):
        done_binding = _write_sealed(
            control / f"shard{shard_id}.done.json",
            _done_payload(
                token=token,
                model_id="chk3",
                shard_id=shard_id,
                scope=scope,
                max_num_seqs=12,
                go_epoch_ns=2_000_000_000,
                go_monotonic_ns=6_000_000_000,
                resume_count=1,
                formal_authorization=FORMAL_AUTHORIZATION,
            ),
        )
        manifests.append(
            _shard_manifest(
                path=tmp_path / f"shard{shard_id}" / "manifest.json",
                ready_binding=ready_bindings[shard_id],
                go_binding=go_binding,
                done_binding=done_binding,
                shard_id=shard_id,
                resume_count=1,
            )
        )
        (control / f"shard{shard_id}.console.log").write_text(
            "sealed worker output\n", encoding="utf-8"
        )

    timing = subject._reconstruct_completed_pair_timing(
        timing_path=tmp_path / "timing.json",
        model_id="chk3",
        scope=scope,
        max_num_seqs=12,
        shard_manifests=manifests,
        remediation_receipt={"sha256": "receipt"},
        formal_authorization=FORMAL_AUTHORIZATION,
    )
    assert timing["recovery"]["mode"] == "paired_resume"
    assert timing["recovery"]["launched_shards"] == [0, 1]
    assert not timing["shared_timing"][
        "speed_measurement_valid_for_candidate_selection"
    ]
    assert timing["shared_timing"]["aggregate_output_tokens_per_second"] is None
    assert timing["gpu_idle_after_worker_exit"] is None
    assert timing["gpu_idle_after_worker_exit_claimed"] is False
    assert timing["gpu_idle_after_worker_exit_status"] == (
        subject.POST_EXIT_GPU_UNAVAILABLE_STATUS
    )


def _asymmetric_reconstruction_fixture(
    tmp_path: Path, *, tamper_latest_done_go: bool
) -> list[Path]:
    scope = "formal_merged_panel"
    tokens = {0: "3" * 64, 1: "4" * 64}
    controls = {0: tmp_path / "attempt_old", 1: tmp_path / "attempt_new"}
    ready_bindings = {
        shard_id: _write_sealed(
            controls[shard_id] / f"shard{shard_id}.ready.json",
            _ready_payload(
                token=tokens[shard_id],
                model_id="chk0",
                shard_id=shard_id,
                scope=scope,
                max_num_seqs=16,
                formal_authorization=FORMAL_AUTHORIZATION,
            ),
        )
        for shard_id in (0, 1)
    }
    # Each historical GO binds both engine-ready slots. Only the shard's own
    # binding is carried by that shard manifest and semantically required here.
    other_ready_bindings = {}
    for shard_id in (0, 1):
        other = 1 - shard_id
        other_ready_bindings[shard_id] = _write_sealed(
            controls[shard_id] / f"shard{other}.ready.json",
            _ready_payload(
                token=tokens[shard_id],
                model_id="chk0",
                shard_id=other,
                scope=scope,
                max_num_seqs=16,
                formal_authorization=FORMAL_AUTHORIZATION,
            ),
        )
    go_bindings = {}
    manifests = []
    for shard_id in (0, 1):
        session_ready = {
            shard_id: ready_bindings[shard_id],
            1 - shard_id: other_ready_bindings[shard_id],
        }
        epoch_ns = 3_000_000_000 + shard_id * 1_000_000_000
        monotonic_ns = 7_000_000_000 + shard_id * 1_000_000_000
        go_bindings[shard_id] = _write_sealed(
            controls[shard_id] / "go.json",
            _go_payload(
                token=tokens[shard_id],
                model_id="chk0",
                scope=scope,
                max_num_seqs=16,
                epoch_ns=epoch_ns,
                monotonic_ns=monotonic_ns,
                ready_bindings=session_ready,
                formal_authorization=FORMAL_AUTHORIZATION,
            ),
        )
        done_binding = _write_sealed(
            controls[shard_id] / f"shard{shard_id}.done.json",
            _done_payload(
                token=tokens[shard_id],
                model_id="chk0",
                shard_id=shard_id,
                scope=scope,
                max_num_seqs=16,
                go_epoch_ns=epoch_ns,
                go_monotonic_ns=monotonic_ns,
                resume_count=shard_id,
                formal_authorization=FORMAL_AUTHORIZATION,
                observed_go_monotonic_ns=(
                    monotonic_ns + 1
                    if tamper_latest_done_go and shard_id == 1
                    else None
                ),
            ),
        )
        manifests.append(
            _shard_manifest(
                path=tmp_path / f"shard{shard_id}" / "manifest.json",
                ready_binding=ready_bindings[shard_id],
                go_binding=go_bindings[shard_id],
                done_binding=done_binding,
                shard_id=shard_id,
                resume_count=shard_id,
            )
        )
    (controls[1] / "shard1.console.log").write_text(
        "sealed recovery output\n", encoding="utf-8"
    )
    return manifests


def test_reconstructs_asymmetric_formal_resume_from_latest_attempt(
    tmp_path: Path,
) -> None:
    manifests = _asymmetric_reconstruction_fixture(
        tmp_path, tamper_latest_done_go=False
    )
    timing = subject._reconstruct_completed_pair_timing(
        timing_path=tmp_path / "timing.json",
        model_id="chk0",
        scope="formal_merged_panel",
        max_num_seqs=16,
        shard_manifests=manifests,
        remediation_receipt={"sha256": "receipt"},
        formal_authorization=FORMAL_AUTHORIZATION,
    )
    assert timing["recovery"] == {
        "launched_shards": [1],
        "mode": "asymmetric_resume",
        "preexisting_complete_shards": [0],
        "selection_eligible": False,
    }
    assert set(timing["control_artifacts"]["worker_console_logs"]) == {"shard1"}


def test_reconstruction_rejects_done_go_tamper(tmp_path: Path) -> None:
    manifests = _asymmetric_reconstruction_fixture(tmp_path, tamper_latest_done_go=True)
    with pytest.raises(subject.DualDp1OrchestrationError, match="accounting drift"):
        subject._reconstruct_completed_pair_timing(
            timing_path=tmp_path / "timing.json",
            model_id="chk0",
            scope="formal_merged_panel",
            max_num_seqs=16,
            shard_manifests=manifests,
            remediation_receipt={"sha256": "receipt"},
            formal_authorization=FORMAL_AUTHORIZATION,
        )
