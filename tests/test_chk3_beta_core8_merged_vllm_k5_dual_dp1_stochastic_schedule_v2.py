from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs.eval import (
    eval_chk3_beta_core8_merged_vllm_k5_dual_dp1_stochastic_schedule_v2 as runner,
)
from jobs.eval import (
    orchestrate_chk3_beta_core8_merged_vllm_k5_dual_dp1_stochastic_schedule_v2 as orchestrator,
)
from open_r1.validator.loo_generation_spec import seal_manifest


MAX_SEQS = 16


def _cases(total: int = 80) -> list[dict]:
    return [
        {
            "absolute_case_index": index,
            "absolute_chunk_id": index // runner.ABSOLUTE_CHUNK_SIZE,
            "sample": {"sample_id": f"sample-{index // 5:04d}"},
            "replicate_id": index % 5,
            "row_seed": 10_000 + index,
        }
        for index in range(total)
    ]


def _receipt_row(index: int) -> dict:
    return {
        "absolute_case_index": index,
        "sample_id": f"sample-{index // 5:04d}",
        "replicate_id": index % 5,
        "row_seed": 10_000 + index,
        "completion_sha256": f"completion-{index}",
        "generated_token_ids_sha256": f"tokens-{index}",
    }


def _prefix_binding(rows: int) -> dict:
    return {
        "path": "/sealed/progress.jsonl",
        "sha256": f"prefix-{rows}",
        "bytes": rows * 100,
        "rows": rows,
    }


def test_v2_contract_seals_schedule_sensitive_non_replay_semantics() -> None:
    contract = runner.sampling_contract(max_num_seqs=MAX_SEQS)
    assert runner.EVALUATION_ID.endswith("dual-independent-dp1-stochastic-schedule-v2")
    assert contract["max_num_seqs_selection"] == (
        "speed_only_fixed_16_after_schedule_sensitivity_audit"
    )
    assert contract["token_identity_replay_gate"] is False
    assert contract["resume_dispatch"] == (
        "only_missing_tuple_keys_never_fsynced_wal_keys"
    )
    assert contract["durable_wal_tuple_policy"] == (
        "immutable_validate_only_never_redispatch"
    )
    with pytest.raises(runner.DualDp1GenerationError, match="requires max_num_seqs=16"):
        runner.sampling_contract(max_num_seqs=12)


def test_partial_chunk_resume_dispatches_only_missing_durable_union() -> None:
    cases = _cases()
    durable = {0, 2, 4, 6, 8}
    durable_snapshot = copy.deepcopy({index: _receipt_row(index) for index in durable})
    receipt_only, pending = runner._resume_dispatch_plan(
        cases=cases,
        durable_indexes=set(durable_snapshot),
        committed_receipt_count=0,
        shard_id=0,
    )
    dispatched = {
        int(case["absolute_case_index"])
        for _, chunk_cases in pending
        for case in chunk_cases
    }
    expected = set(range(0, 80, 2))
    assert receipt_only == []
    assert durable.isdisjoint(dispatched)
    assert durable | dispatched == expected
    assert len(dispatched) == 35
    assert [chunk_id for chunk_id, _ in pending] == [0, 1]
    # Planning is read-only: previously fsynced row objects remain byte-for-byte
    # unchanged and are not represented in the dispatch list.
    assert durable_snapshot == {index: _receipt_row(index) for index in durable}


def test_complete_unreceipted_chunk_is_receipt_only_and_never_dispatched() -> None:
    cases = _cases()
    durable = set(range(0, 40, 2))
    receipt_only, pending = runner._resume_dispatch_plan(
        cases=cases,
        durable_indexes=durable,
        committed_receipt_count=0,
        shard_id=0,
    )
    dispatched = {
        int(case["absolute_case_index"])
        for _, chunk_cases in pending
        for case in chunk_cases
    }
    assert receipt_only == [0]
    assert dispatched == set(range(40, 80, 2))
    assert durable.isdisjoint(dispatched)
    rows = {index: _receipt_row(index) for index in durable}
    receipt = runner._chunk_receipt(
        model_id="chk3",
        shard_id=0,
        chunk_id=0,
        rows_by_index=rows,
        total_cases=len(cases),
    )
    assert receipt["absolute_case_indexes"] == sorted(durable)
    assert receipt["cases"] == 20


def test_receipt_and_completion_guards_reject_missing_or_redispatched_keys() -> None:
    cases = _cases(40)
    incomplete_rows = {index: _receipt_row(index) for index in range(0, 38, 2)}
    with pytest.raises(runner.DualDp1GenerationError, match="exact full chunk"):
        runner._chunk_receipt(
            model_id="chk3",
            shard_id=0,
            chunk_id=0,
            rows_by_index=incomplete_rows,
            total_cases=len(cases),
        )
    with pytest.raises(runner.DualDp1GenerationError, match="was redispatched"):
        runner._assert_planned_missing_completion(
            chunk_id=0,
            absolute_case_index=0,
            planned_indexes_by_chunk={0: {0}},
            outstanding_dispatch_indexes={0},
            durable_indexes_at_session_start={0},
            current_indexes={0},
        )
    with pytest.raises(runner.DualDp1GenerationError, match="duplicate completion"):
        runner._assert_planned_missing_completion(
            chunk_id=0,
            absolute_case_index=2,
            planned_indexes_by_chunk={0: {2}},
            outstanding_dispatch_indexes=set(),
            durable_indexes_at_session_start=set(),
            current_indexes={2},
        )


def test_dispatch_evidence_and_final_coverage_are_exact_and_tamper_evident() -> None:
    cases = _cases(40)
    durable = {0, 2, 4, 6}
    _, pending = runner._resume_dispatch_plan(
        cases=cases,
        durable_indexes=durable,
        committed_receipt_count=0,
        shard_id=0,
    )
    evidence = runner._resume_dispatch_evidence(
        cases=cases,
        durable_indexes=durable,
        pending_chunks=pending,
        shard_id=0,
        durable_wal_prefix_binding=_prefix_binding(len(durable)),
    )
    runner._validate_resume_dispatch_evidence(
        evidence,
        cases=cases,
        shard_id=0,
        observed_durable_wal_prefix_binding=_prefix_binding(len(durable)),
    )
    expected = set(range(0, 40, 2))
    runner._validate_exact_shard_coverage(
        expected,
        shard_id=0,
        total_cases=40,
    )
    tampered = copy.deepcopy(evidence)
    tampered["dispatched_absolute_case_indexes"].pop()
    with pytest.raises(runner.DualDp1GenerationError):
        runner._validate_resume_dispatch_evidence(
            tampered,
            cases=cases,
            shard_id=0,
            observed_durable_wal_prefix_binding=_prefix_binding(len(durable)),
        )
    with pytest.raises(runner.DualDp1GenerationError, match="not exact"):
        runner._validate_exact_shard_coverage(
            expected - {38},
            shard_id=0,
            total_cases=40,
        )
    with pytest.raises(runner.DualDp1GenerationError, match="not exact"):
        runner._validate_exact_shard_coverage(
            expected | {1},
            shard_id=0,
            total_cases=40,
        )


def test_appending_missing_rows_preserves_the_fsynced_wal_prefix(
    tmp_path: Path,
) -> None:
    wal = tmp_path / "progress.jsonl"
    durable_rows = [{"absolute_case_index": index} for index in (0, 2, 4)]
    with wal.open("xb") as handle:
        for row in durable_rows:
            handle.write(runner._encoded_row(row))
        handle.flush()
        os.fsync(handle.fileno())
    before = runner._wal_prefix_binding(wal, rows=len(durable_rows))
    with wal.open("ab") as handle:
        for index in (6, 8):
            handle.write(runner._encoded_row({"absolute_case_index": index}))
        handle.flush()
        os.fsync(handle.fileno())
    after = runner._wal_prefix_binding(wal, rows=len(durable_rows))
    assert after == before
    assert runner._wal_prefix_binding(wal, rows=5)["rows"] == 5


def test_one_fsynced_row_ahead_is_accounted_without_regeneration() -> None:
    runtime = {
        "generated_output_tokens": 10,
        "completed_requests_in_timed_session": 3,
        "generation_started_epoch_ns": 1,
        "generation_started_monotonic_ns": 1,
    }
    recovered = runner._recover_one_wal_row_ahead_runtime(
        runtime,
        ahead_row={
            "absolute_case_index": 6,
            "completion_sha256": "completion-6",
            "generated_token_ids_sha256": "tokens-6",
            "generated_token_ids": [101, 102, 103],
        },
    )
    assert recovered["generated_output_tokens"] == 13
    assert recovered["completed_requests_in_timed_session"] == 4
    assert (
        recovered["wal_one_record_ahead_recovery"]["generation_rows_redispatched"] == 0
    )
    assert runtime["generated_output_tokens"] == 10


def test_formal_authorization_failure_precedes_receipt_gpu_lock_and_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def reject_authorization(**_: object) -> None:
        events.append("authorization")
        raise orchestrator.DualDp1OrchestrationError("authorization rejected")

    monkeypatch.setattr(
        orchestrator,
        "_formal_authorization_evidence",
        reject_authorization,
    )
    monkeypatch.setattr(
        orchestrator.remediation,
        "load_and_validate_receipt",
        lambda *_args, **_kwargs: events.append("receipt"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_acquire_gpu_lock_pair",
        lambda **_kwargs: events.append("gpu_lock"),
    )
    monkeypatch.setattr(
        orchestrator.subprocess,
        "Popen",
        lambda *_args, **_kwargs: events.append("spawn"),
    )
    with pytest.raises(
        orchestrator.DualDp1OrchestrationError,
        match="authorization rejected",
    ):
        orchestrator.run_model(
            python_bin=Path("/usr/bin/python3"),
            model_id="chk3",
            cohort=tmp_path / "cohort.json",
            cohort_sha256="0" * 64,
            model_root=tmp_path / "formal",
            scope="formal_merged_panel",
            max_num_seqs=16,
            allow_resume=True,
            gpu_wait_timeout_seconds=1,
            gpu_poll_seconds=1,
            ready_timeout_seconds=1,
            formal_authorization=tmp_path / "smoke.json",
        )
    assert events == ["authorization"]


def test_all_shards_timing_recovery_allows_receipt_only_resume_counts(
    tmp_path: Path,
) -> None:
    token = "a" * 64
    go = {
        "go_epoch_ns": 1_000,
        "go_monotonic_ns": 1_000,
    }
    done_values = {
        shard_id: {
            "run_token": token,
            "observed_go_epoch_ns": 1_000,
            "observed_go_monotonic_ns": 1_000,
            "generation_started_monotonic_ns": 1_100,
            "generation_finished_monotonic_ns": 2_000 + shard_id,
            "generated_output_tokens": 100 + shard_id,
            # The last engine session was resume=0; a later CPU-only closure
            # recovery incremented the shard manifest's resume_count to 1.
            "resume_count": 0,
        }
        for shard_id in (0, 1)
    }
    final_evidence = {
        shard_id: {
            "resume_count": 1,
            "engine_session_resume_count": 0,
            "canonical_generated_output_tokens": 100 + shard_id,
            "ready": {"path": f"/sealed/ready-{shard_id}.json"},
            "go": {"path": "/sealed/go.json"},
            "done": {"path": f"/sealed/done-{shard_id}.json"},
        }
        for shard_id in (0, 1)
    }
    logs = {shard_id: tmp_path / f"shard{shard_id}.log" for shard_id in (0, 1)}
    for path in logs.values():
        path.write_text("complete\n", encoding="utf-8")
    value = orchestrator._seal_recovery_timing(
        path=tmp_path / "timing.json",
        model_id="chk3",
        max_num_seqs=16,
        run_token=token,
        go=go,
        go_binding={"path": "/sealed/go.json"},
        launched_shards=[0, 1],
        preexisting_complete_shards=[],
        done_values=done_values,
        logs=logs,
        gpu_idle_evidence=[],
        processes={0: SimpleNamespace(pid=100), 1: SimpleNamespace(pid=101)},
        final_evidence=final_evidence,
        shard_manifest_bindings={
            0: {"path": "/sealed/shard0.json"},
            1: {"path": "/sealed/shard1.json"},
        },
        remediation_receipt={"path": "/sealed/amendment.json"},
        formal_authorization={"official_smoke_suite": {"path": "/sealed/smoke"}},
        reconstruction_evidence={
            "reconstructed_from_two_deeply_validated_sealed_shards": True,
            "no_new_gpu_work": True,
        },
    )
    assert value["shared_timing"]["both_workers_resume_count"] == [1, 1]
    assert (
        value["shared_timing"]["speed_measurement_valid_for_candidate_selection"]
        is False
    )
    assert value["gpu_idle_after_worker_exit_claimed"] is False


def test_final_fsynced_chunk_closes_with_zero_requests_and_recoverable_done(
    tmp_path: Path,
) -> None:
    token = "b" * 64
    ready_path = tmp_path / "shard0.ready.json"
    go_path = tmp_path / "go.json"
    done_path = tmp_path / "shard0.done.json"
    runner.core._write_new_json(
        ready_path,
        seal_manifest(
            {
                "schema_version": runner.READY_SCHEMA,
                "status": "ready",
                "evaluation_id": runner.EVALUATION_ID,
                "run_token": token,
                "model_id": "chk3",
                "shard_id": 0,
                "physical_gpu_index": 0,
                "evaluation_scope": "infrastructure_smoke",
                "max_num_seqs": 16,
                "formal_authorization": None,
            }
        ),
    )
    runner.core._write_new_json(
        go_path,
        seal_manifest(
            {
                "schema_version": runner.GO_SCHEMA,
                "status": "released",
                "evaluation_id": runner.EVALUATION_ID,
                "run_token": token,
                "model_id": "chk3",
                "evaluation_scope": "infrastructure_smoke",
                "max_num_seqs": 16,
                "formal_authorization": None,
                "go_epoch_ns": 1_000,
                "go_monotonic_ns": 10_000,
            }
        ),
    )
    engine_runtime = {
        "ready_manifest": runner._binding_with_payload(ready_path),
        "go_manifest": runner._binding_with_payload(go_path),
        "control_attempt": {
            "control_dir": str(tmp_path.resolve()),
            "run_token_sha256": runner._sha256_text(token),
            "ready_path": str(ready_path.resolve()),
            "go_path": str(go_path.resolve()),
            "done_path": str(done_path.resolve()),
        },
        "generation_started_epoch_ns": 2_000,
        "generation_started_monotonic_ns": 11_000,
        "last_completion_epoch_ns": 3_000,
        "last_completion_monotonic_ns": 12_000,
        "generation_wall_seconds": 0.001,
        "generated_output_tokens": 777,
        "completed_requests_in_timed_session": 20,
        "engine_session_resume_count": 0,
    }
    closed = runner._recover_receipt_only_done(
        engine_runtime=engine_runtime,
        model_id="chk3",
        shard_id=0,
        physical_gpu_index=0,
        evaluation_scope="infrastructure_smoke",
        max_num_seqs=16,
        formal_authorization=None,
        closure_gpu_lease={
            "lease_mode": "supervisor_preacquired_inherited_fd",
            "physical_gpu_index": 0,
        },
        recovered_receipt_chunks=[0],
        durable_wal_prefix_binding={
            "path": "/sealed/progress.jsonl",
            "sha256": "d" * 64,
            "bytes": 1234,
            "rows": 20,
        },
    )
    assert done_path.is_file()
    assert closed["completion_recovery"]["generation_rows_redispatched"] == 0
    assert closed["completion_recovery"]["model_engine_spawned"] is False
    assert closed["generation_finished_monotonic_ns"] == 12_000
    replayed_closure = runner._recover_one_record_ahead_done(
        engine_runtime=engine_runtime,
        model_id="chk3",
        shard_id=0,
        physical_gpu_index=0,
        evaluation_scope="infrastructure_smoke",
        max_num_seqs=16,
        formal_authorization=None,
    )
    assert replayed_closure["completion_recovery"] == closed["completion_recovery"]
    assert replayed_closure["done_manifest"] == closed["done_manifest"]


def test_supervisor_recognizes_zero_request_worker_as_early_complete(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control"
    control.mkdir()
    manifest = tmp_path / "shard0" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("{}\n", encoding="utf-8")
    values, bindings, early = orchestrator._wait_for_ready(
        processes={0: SimpleNamespace(poll=lambda: 0)},
        control_dir=control,
        run_token="c" * 64,
        model_id="chk3",
        scope="formal_merged_panel",
        max_num_seqs=16,
        timeout_seconds=1,
        formal_authorization={"official_smoke_suite": {}},
        shard_manifests={0: manifest},
    )
    assert values == {}
    assert bindings == {}
    assert early == {0}


def test_cleanup_excludes_exited_and_pid_reused_early_complete_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = {
        0: {
            "pid": 100,
            "process_group_id": 100,
            "session_id": 100,
            "start_time_ticks": 1_000,
        },
        1: {
            "pid": 101,
            "process_group_id": 101,
            "session_id": 101,
            "start_time_ticks": 1_001,
        },
        2: {
            "pid": 102,
            "process_group_id": 102,
            "session_id": 102,
            "start_time_ticks": 1_002,
        },
    }
    processes = {
        0: SimpleNamespace(pid=100, poll=lambda: 0),  # early-complete/reaped
        1: SimpleNamespace(pid=101, poll=lambda: None),  # same live task
        2: SimpleNamespace(pid=102, poll=lambda: None),  # PID was reused
    }
    observed = {
        101: identities[1],
        102: {**identities[2], "start_time_ticks": 9_999},
    }
    monkeypatch.setattr(
        orchestrator,
        "_process_identity",
        lambda pid: copy.deepcopy(observed.get(pid)),
    )
    selected = orchestrator._live_cleanup_processes(processes, identities)
    assert selected == {1: processes[1]}

    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(
        orchestrator.os,
        "killpg",
        lambda pgid, sig: signalled.append((pgid, sig)),
    )
    orchestrator._terminate_process_groups(
        {0: processes[0], 2: processes[2]},
        identities={0: identities[0], 2: identities[2]},
    )
    assert signalled == []
