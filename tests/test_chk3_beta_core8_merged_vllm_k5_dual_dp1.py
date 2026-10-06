from __future__ import annotations

import copy
import fcntl
import hashlib
import os
import threading
import time
from pathlib import Path

import pytest

from jobs.eval import eval_chk3_beta_core8_merged_vllm_k5_dual_dp1 as runner
from open_r1.validator.loo_generation_spec import seal_manifest


MAX_SEQS = 12


def _case(index: int) -> dict:
    return {
        "absolute_case_index": index,
        "absolute_chunk_id": index // runner.ABSOLUTE_CHUNK_SIZE,
    }


def test_contract_is_two_external_dp1_graph_workers() -> None:
    contract = runner.sampling_contract(max_num_seqs=MAX_SEQS)
    assert contract["workers"] == 2
    assert contract["data_parallel_size_per_worker"] == 1
    assert contract["tensor_parallel_size_per_worker"] == 1
    assert contract["pipeline_parallel_size_per_worker"] == 1
    assert contract["shard_function"] == "absolute_case_index_mod_2"
    assert contract["cases_per_absolute_chunk_per_worker"] == 20
    assert contract["gpu_memory_utilization_per_worker"] == 0.95
    assert contract["enforce_eager"] is False
    assert contract["cuda_graphs"] is True
    assert contract["async_output_processing"] is True
    assert contract["quantization"] is None


def test_parity_sharding_is_exact_20_20_for_every_meeting_window() -> None:
    cases = [_case(index) for index in range(80)]
    shard0 = runner.shard_cases(cases, 0)
    shard1 = runner.shard_cases(cases, 1)
    assert [case["absolute_case_index"] for case in shard0] == list(range(0, 80, 2))
    assert [case["absolute_case_index"] for case in shard1] == list(range(1, 80, 2))
    assert not (
        {case["absolute_case_index"] for case in shard0}
        & {case["absolute_case_index"] for case in shard1}
    )
    assert {case["absolute_case_index"] for case in shard0 + shard1} == set(range(80))
    for chunk_id in range(2):
        assert sum(case["absolute_chunk_id"] == chunk_id for case in shard0) == 20
        assert sum(case["absolute_chunk_id"] == chunk_id for case in shard1) == 20


def test_shard_completion_allows_completion_order_only_in_active_chunk() -> None:
    first = set(range(0, 40, 2))
    partial = {40, 46, 58}
    assert runner._completion_status(first | partial, shard_id=0, total_cases=80) == (
        1,
        partial,
    )
    with pytest.raises(runner.DualDp1GenerationError, match="skips"):
        runner._completion_status(first | {42, 80}, shard_id=0, total_cases=120)
    with pytest.raises(runner.DualDp1GenerationError, match="wrong-parity"):
        runner._completion_status({1}, shard_id=0, total_cases=40)


def test_environment_requires_one_exact_gpu_and_rejects_all_dp_ambient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in runner.REQUIRED_VLLM_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    runner._assert_required_environment(physical_gpu_index=1, require_cuda=True)
    monkeypatch.setenv("VLLM_DP_SIZE", "1")
    with pytest.raises(runner.DualDp1GenerationError, match="must be absent"):
        runner._assert_required_environment(physical_gpu_index=1, require_cuda=True)


def test_inherited_lock_proves_canonical_supervisor_flock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = tmp_path / "gpu0.lock"
    lock.touch(mode=0o600)
    monkeypatch.setattr(runner, "gpu_lock_path", lambda index: lock)
    monkeypatch.setattr(runner, "_external_gpu_processes", lambda index: [])
    fd = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        with runner.inherited_gpu_lease(
            physical_gpu_index=0, inherited_lock_fd=fd
        ) as evidence:
            assert evidence["lock_inode"] == os.fstat(fd).st_ino
            assert evidence["lease_mode"] == "supervisor_preacquired_inherited_fd"
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_ready_go_done_barrier_binds_worker_and_shared_go(
    tmp_path: Path,
) -> None:
    token = "a" * 64
    formal_authorization = {
        "official_smoke_suite": {"path": "/sealed/smoke.json"},
        "benchmark_selection": {"path": "/sealed/selection.json"},
        "generation_gates": {"formal_generation_unblocked": True},
    }
    identity = {
        "physical_gpu_index": 0,
        "uuid": "GPU-test-0",
        "pci_bus_id": "00000000:01:00.0",
    }
    runtime = {
        "physical_gpu_identity": identity,
        "engine_core_gpu_binding": {
            "worker_pid": 101,
            "engine_core_pid": 202,
            "physical_gpu_index": 0,
        },
    }

    def release() -> None:
        ready_path = tmp_path / "shard0.ready.json"
        while not ready_path.exists():
            time.sleep(0.005)
        ready = runner._binding_with_payload(ready_path)
        other_path = tmp_path / "shard1.proxy.json"
        runner.core._write_new_json(
            other_path,
            seal_manifest({"schema_version": "proxy-v1", "status": "complete"}),
        )
        other = runner._binding_with_payload(other_path)
        runner.core._write_new_json(
            tmp_path / "go.json",
            seal_manifest(
                {
                    "schema_version": runner.GO_SCHEMA,
                    "status": "released",
                    "evaluation_id": runner.EVALUATION_ID,
                    "run_token": token,
                    "model_id": "chk1",
                    "evaluation_scope": "formal_merged_panel",
                    "max_num_seqs": MAX_SEQS,
                    "formal_authorization": formal_authorization,
                    "go_epoch_ns": time.time_ns(),
                    "go_monotonic_ns": time.monotonic_ns(),
                    "go_at_utc": runner.core._utc_now(),
                    "physical_gpu_indexes": [0, 1],
                    "ready_manifests": {"shard0": ready, "shard1": other},
                }
            ),
        )

    thread = threading.Thread(target=release)
    thread.start()
    ready, go = runner._publish_ready_and_wait_go(
        control_dir=tmp_path,
        run_token=token,
        model_id="chk1",
        shard_id=0,
        physical_gpu_index=0,
        evaluation_scope="formal_merged_panel",
        max_num_seqs=MAX_SEQS,
        formal_authorization=formal_authorization,
        engine_runtime=runtime,
        timeout_seconds=2,
    )
    thread.join()
    assert ready["payload_sha256"]
    assert go["manifest"]["status"] == "released"
    done = runner._publish_done(
        control_dir=tmp_path,
        run_token=token,
        model_id="chk1",
        shard_id=0,
        physical_gpu_index=0,
        evaluation_scope="formal_merged_panel",
        max_num_seqs=MAX_SEQS,
        formal_authorization=formal_authorization,
        generation_started_epoch_ns=10,
        generation_finished_epoch_ns=20,
        observed_go_epoch_ns=9,
        observed_go_monotonic_ns=90,
        generation_started_monotonic_ns=100,
        generation_finished_monotonic_ns=200,
        generated_output_tokens=300,
        completed_requests=20,
        resume_count=1,
    )
    assert runner.core._read_json(Path(done["path"]))["resume_count"] == 1
    closure_runtime = {
        "ready_manifest": ready,
        "go_manifest": go["binding"],
        "control_attempt": {
            "control_dir": str(tmp_path.resolve()),
            "run_token_sha256": runner._sha256_text(token),
            "ready_path": str((tmp_path / "shard0.ready.json").resolve()),
            "go_path": str((tmp_path / "go.json").resolve()),
            "done_path": str((tmp_path / "shard0.done.json").resolve()),
        },
        "generated_output_tokens": 300,
        "completed_requests_in_timed_session": 20,
        "engine_session_resume_count": 1,
    }
    first_recovery = runner._recover_one_record_ahead_done(
        engine_runtime=closure_runtime,
        model_id="chk1",
        shard_id=0,
        physical_gpu_index=0,
        evaluation_scope="formal_merged_panel",
        max_num_seqs=MAX_SEQS,
        formal_authorization=formal_authorization,
    )
    second_recovery = runner._recover_one_record_ahead_done(
        engine_runtime=closure_runtime,
        model_id="chk1",
        shard_id=0,
        physical_gpu_index=0,
        evaluation_scope="formal_merged_panel",
        max_num_seqs=MAX_SEQS,
        formal_authorization=formal_authorization,
    )
    assert first_recovery == second_recovery
    assert first_recovery["done_manifest"] == done


def test_formal_run_shard_requires_authorization_before_gpu_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def gpu_must_not_be_touched(**_: object) -> None:
        raise AssertionError("GPU/environment inspection occurred before authorization")

    monkeypatch.setattr(runner, "_assert_required_environment", gpu_must_not_be_touched)
    output_dir = tmp_path / "formal-shard"
    with pytest.raises(
        runner.DualDp1GenerationError,
        match="formal generation requires a deeply validated official smoke authorization",
    ):
        runner.run_shard(
            model_id="chk1",
            shard_id=0,
            physical_gpu_index=0,
            cohort_path=tmp_path / "cohort.json",
            cohort_sha256="0" * 64,
            output_dir=output_dir,
            resume=False,
            smoke=False,
            gpu_wait_timeout_seconds=1,
            gpu_poll_seconds=1,
            max_num_seqs=MAX_SEQS,
            control_dir=tmp_path / "control",
            run_token="a" * 64,
            inherited_lock_fd=99,
            control_timeout_seconds=1,
            expected_scope="formal_merged_panel",
        )
    assert not output_dir.exists()


def test_cli_freezes_worker_mapping_barrier_and_merge_timing() -> None:
    parser = runner._parser()
    parsed = parser.parse_args(
        [
            "run-shard",
            "--model-id",
            "chk1",
            "--shard-id",
            "0",
            "--physical-gpu-index",
            "0",
            "--cohort",
            "cohort.json",
            "--cohort-sha256",
            "a" * 64,
            "--output-dir",
            "out/shard0",
            "--max-num-seqs",
            "12",
            "--control-dir",
            "control",
            "--run-token",
            "b" * 64,
            "--inherited-lock-fd",
            "7",
            "--scope",
            "formal_merged_panel",
        ]
    )
    assert parsed.shard_id == parsed.physical_gpu_index == 0
    assert parsed.max_num_seqs == 12
    merged = parser.parse_args(
        [
            "merge-run",
            "--model-id",
            "chk1",
            "--cohort",
            "cohort.json",
            "--cohort-sha256",
            "a" * 64,
            "--output-dir",
            "out/chk1",
            "--scope",
            "infrastructure_smoke",
            "--max-num-seqs",
            "12",
            "--orchestrator-timing",
            "out/chk1/orchestrator.json",
        ]
    )
    assert merged.orchestrator_timing.name == "orchestrator.json"


def test_wal_one_row_ahead_and_receipt_tamper_are_detected(tmp_path: Path) -> None:
    wal = tmp_path / "rows.jsonl"
    first = {"absolute_case_index": 2, "value": "first"}
    second = {"absolute_case_index": 0, "value": "second"}
    first_encoded = runner._encoded_row(first)
    wal.write_bytes(first_encoded + runner._encoded_row(second))
    rows, tracker, prefix = runner._read_canonical_jsonl(wal, recorded_prefix_rows=1)
    assert rows == [first, second]
    assert tracker.rows == 2
    assert prefix["sha256"] == hashlib.sha256(first_encoded).hexdigest()
    wal.write_bytes(wal.read_bytes()[:-1])
    with pytest.raises(runner.dp2.VllmK5GenerationError, match="incomplete tail"):
        runner._read_canonical_jsonl(wal)

    fake_rows = {
        index: {
            "sample_id": f"sample-{index}",
            "replicate_id": index % 5,
            "row_seed": 1000 + index,
            "completion_sha256": f"{index:064x}",
            "generated_token_ids_sha256": f"{index + 100:064x}",
        }
        for index in range(0, 40, 2)
    }
    receipt = runner._chunk_receipt(
        model_id="chk1",
        shard_id=0,
        chunk_id=0,
        rows_by_index=fake_rows,
        total_cases=40,
    )
    runner._validate_receipts(
        [receipt],
        model_id="chk1",
        shard_id=0,
        rows_by_index=fake_rows,
        total_cases=40,
    )
    receipt["canonical_row_bindings_sha256"] = "0" * 64
    with pytest.raises(runner.DualDp1GenerationError, match="receipt drift"):
        runner._validate_receipts(
            [receipt],
            model_id="chk1",
            shard_id=0,
            rows_by_index=fake_rows,
            total_cases=40,
        )


def test_wrong_shard_and_duplicate_wal_rows_fail_before_payload_rebuild() -> None:
    cases = [_case(index) for index in range(40)]
    kwargs = {
        "cases": cases,
        "model_id": "chk1",
        "model_label": "CHK1",
        "bound_rows": {},
        "sample_manifest_sha256": "a" * 64,
        "source_artifact_sha256s": {},
        "max_num_seqs": MAX_SEQS,
        "shard_id": 0,
        "physical_gpu_index": 0,
    }
    with pytest.raises(runner.DualDp1GenerationError, match="wrong-parity"):
        runner.validate_wal_rows([{"absolute_case_index": 1}], **kwargs)
    with pytest.raises(runner.DualDp1GenerationError, match="duplicate"):
        runner.validate_wal_rows(
            [{"absolute_case_index": 0}, {"absolute_case_index": 0}], **kwargs
        )


def test_canonical_crash_recovery_exact_validates_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    path = tmp_path / "canonical.jsonl"
    rows = [{"absolute_case_index": 0}, {"absolute_case_index": 1}]
    runner._write_or_validate_canonical_jsonl(path, rows)
    runner._write_or_validate_canonical_jsonl(path, rows)
    with pytest.raises(runner.DualDp1GenerationError, match="drift"):
        runner._write_or_validate_canonical_jsonl(
            path, [{"absolute_case_index": 0}, {"absolute_case_index": 2}]
        )


def test_exact_union_rejects_gap_overlap_and_wrong_parity() -> None:
    valid = {
        0: [{"absolute_case_index": 0}, {"absolute_case_index": 2}],
        1: [{"absolute_case_index": 1}, {"absolute_case_index": 3}],
    }
    assert set(runner._exact_shard_union(valid, total_cases=4)) == set(range(4))
    with pytest.raises(runner.DualDp1GenerationError, match="exact case union"):
        runner._exact_shard_union(
            {0: valid[0], 1: [{"absolute_case_index": 1}]}, total_cases=4
        )
    with pytest.raises(runner.DualDp1GenerationError, match="parity"):
        runner._exact_shard_union(
            {0: valid[0], 1: [{"absolute_case_index": 1}, {"absolute_case_index": 2}]},
            total_cases=4,
        )


def _fresh_timing(tokens: int, wall: float) -> dict:
    gpu_rows = [
        {
            "physical_gpu_index": gpu_id,
            "uuid": runner.EXPECTED_GPU_IDENTITIES[gpu_id]["uuid"],
            "pci_bus_id": runner.EXPECTED_GPU_IDENTITIES[gpu_id]["pci_bus_id"],
            "memory_used_mib": 14,
            "memory_total_mib": 49140,
            "utilization_gpu_percent": 0,
        }
        for gpu_id in (0, 1)
    ]
    return {
        "schema_version": runner.ORCHESTRATOR_TIMING_SCHEMA,
        "status": "complete",
        "evaluation_id": runner.EVALUATION_ID,
        "model_id": "chk1",
        "evaluation_scope": "infrastructure_smoke",
        "max_num_seqs_per_engine": MAX_SEQS,
        "run_token": "c" * 64,
        "gpu_idle_before_spawn": gpu_rows,
        "gpu_idle_after_worker_exit_claimed": True,
        "gpu_idle_after_worker_exit_status": ("passed_while_canonical_gpu_locks_held"),
        "gpu_idle_after_worker_exit": {
            "status": "passed",
            "claimed": True,
            "checked_while_canonical_gpu_locks_held": True,
            "two_consecutive_idle_samples": True,
            "no_compute_processes": True,
            "timeout_seconds": 60,
            "poll_seconds": 5,
            "samples": [
                {
                    "sampled_at_utc": f"2026-08-16T00:00:0{sample_id}Z",
                    "compute_processes": [],
                    "gpus": copy.deepcopy(gpu_rows),
                }
                for sample_id in (1, 2)
            ],
        },
        "timing_reconstruction": None,
        "parallel_topology": {
            "independent_engine_count": 2,
            "data_parallel_size_per_engine": 1,
            "tensor_parallel_size_per_engine": 1,
            "physical_gpu_indices": [0, 1],
            "cross_gpu_collectives": False,
            "shared_go_barrier": True,
        },
        "control_artifacts": {
            "go": {},
            "ready": {},
            "done": {},
            "worker_console_logs": {},
        },
        "shared_timing": {
            "parallel_generation_wall_seconds": wall,
            "aggregate_generated_output_tokens": tokens,
            "aggregate_output_tokens_per_second": tokens / wall,
            "speed_measurement_valid_for_candidate_selection": True,
        },
    }


def test_orchestrator_timing_methodology_tamper_fails(tmp_path: Path) -> None:
    path = tmp_path / "timing.json"
    runner.core._write_new_json(path, seal_manifest(_fresh_timing(100, 2.0)))
    runner._load_orchestrator_timing(
        path,
        model_id="chk1",
        scope="infrastructure_smoke",
        max_num_seqs=MAX_SEQS,
        expected_tokens=100,
    )
    bad = _fresh_timing(100, 2.0)
    bad["shared_timing"]["aggregate_output_tokens_per_second"] = 99.0
    bad_path = tmp_path / "bad-timing.json"
    runner.core._write_new_json(bad_path, seal_manifest(bad))
    with pytest.raises(runner.DualDp1GenerationError, match="throughput drift"):
        runner._load_orchestrator_timing(
            bad_path,
            model_id="chk1",
            scope="infrastructure_smoke",
            max_num_seqs=MAX_SEQS,
            expected_tokens=100,
        )


def test_initialization_crash_layouts_are_narrowly_recoverable(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert runner._classify_resume_layout(empty) == "empty_root_before_launch"
    assert runner._initialization_recovery_resume_count("empty_root_before_launch") == 1

    prestage = tmp_path / "prestate"
    partial = prestage / ".partial"
    partial.mkdir(parents=True)
    runner.core._write_new_json(
        prestage / "launch.json",
        seal_manifest(
            {"schema_version": "launch-fixture-v1", "status": "initializing"}
        ),
    )
    (partial / "generations.completion_order.progress.v1.jsonl").touch()
    (partial / "absolute_chunks.progress.v1.jsonl").touch()
    assert (
        runner._classify_resume_layout(prestage) == "launch_and_empty_wals_before_state"
    )
    assert (
        runner._initialization_recovery_resume_count(
            "launch_and_empty_wals_before_state"
        )
        == 1
    )
    (partial / "generations.completion_order.progress.v1.jsonl").write_text(
        "not-empty", encoding="utf-8"
    )
    with pytest.raises(runner.DualDp1GenerationError, match="exact empty"):
        runner._classify_resume_layout(prestage)


def test_merge_rejects_symlink_root_before_reading_any_sources(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(runner.DualDp1GenerationError, match="symlink"):
        runner.merge_run(
            cohort_path=tmp_path / "missing-cohort.json",
            cohort_sha256="a" * 64,
            output_dir=linked,
            model_id="chk1",
            scope="infrastructure_smoke",
            max_num_seqs=MAX_SEQS,
            orchestrator_timing_path=tmp_path / "missing-timing.json",
        )
