from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path

import pytest

from jobs.eval import migrate_chk3_beta_k10_to_vllm_k5 as subject
from open_r1.validator.loo_generation_spec import derive_row_seed, seal_manifest


def _write_canonical(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(subject._canonical(value) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, *, state_rows: int = 2, wal_rows: int = 2):
    sample_path = tmp_path / "samples.json"
    samples = [{"sample_id": f"sample-{index}"} for index in range(2)]
    _write_canonical(sample_path, seal_manifest({"samples": samples}))
    sample_sha = subject._sha256_file(sample_path)
    rows = []
    for index in range(wal_rows):
        sample_index, replicate_id = divmod(index, len(subject.REPLICATE_SEEDS))
        sample_id = samples[sample_index]["sample_id"]
        replicate_seed = subject.REPLICATE_SEEDS[replicate_id]
        rows.append(
            {
                "evaluation_id": subject.OLD_EVALUATION_ID,
                "model_id": "chk1",
                "sample_id": sample_id,
                "replicate_id": replicate_id,
                "replicate_seed": replicate_seed,
                "row_seed": derive_row_seed(replicate_seed, sample_id),
                "sample_manifest_sha256": sample_sha,
                "generated_text": f"row {index}",
            }
        )
    wal_path = tmp_path / "generations.progress.v1.jsonl"
    wal_path.write_text(
        "".join(subject._canonical(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    prefix = "".join(
        subject._canonical(row) + "\n" for row in rows[:state_rows]
    ).encode("utf-8")
    last = rows[state_rows - 1] if state_rows else None
    last_tuple = (
        None
        if last is None
        else {
            "model_id": last["model_id"],
            "sample_id": last["sample_id"],
            "replicate_id": last["replicate_id"],
        }
    )
    state = seal_manifest(
        {
            "schema_version": "chk3-beta-core8-merged-generation-state-v1",
            "status": "generating",
            "evaluation_id": subject.OLD_EVALUATION_ID,
            "model_id": "chk1",
            "completed_cases": state_rows,
            "expected_cases": subject.EXPECTED_OLD_CASES,
            "last_tuple": last_tuple,
            "completion_matrix": {
                "completed_cases": state_rows,
                "last_tuple": last_tuple,
            },
            "partial_results": {
                "path": str(wal_path.resolve()),
                "bytes": len(prefix),
                "sha256": subject.hashlib.sha256(prefix).hexdigest(),
            },
        }
    )
    state_path = tmp_path / "state.progress.v1.json"
    _write_canonical(state_path, state)
    return state_path, wal_path, sample_path, rows


def test_stopped_audit_accepts_exact_prefix(tmp_path: Path) -> None:
    state, wal, samples, _ = _fixture(tmp_path)
    audit = subject.audit_stopped_progress(
        state_path=state,
        wal_path=wal,
        sample_manifest_path=samples,
    )
    assert audit["completed_cases"] == 2
    assert audit["actual_wal"]["rows"] == 2
    assert audit["fsynced_one_row_ahead_of_state"] is False
    assert audit["resume_contract_valid"] is True


def test_stopped_audit_accepts_exactly_one_fsynced_row_ahead(tmp_path: Path) -> None:
    state, wal, samples, rows = _fixture(tmp_path, state_rows=2, wal_rows=3)
    audit = subject.audit_stopped_progress(
        state_path=state,
        wal_path=wal,
        sample_manifest_path=samples,
    )
    assert audit["actual_wal"]["rows"] == 3
    assert audit["actual_wal"]["last_tuple"]["replicate_id"] == rows[2]["replicate_id"]
    assert audit["fsynced_one_row_ahead_of_state"] is True


def test_stopped_audit_rejects_more_than_one_row_ahead(tmp_path: Path) -> None:
    state, wal, samples, _ = _fixture(tmp_path, state_rows=1, wal_rows=3)
    with pytest.raises(subject.MigrationError, match="more than one row"):
        subject.audit_stopped_progress(
            state_path=state,
            wal_path=wal,
            sample_manifest_path=samples,
        )


def test_audit_rejects_incomplete_or_noncanonical_wal(tmp_path: Path) -> None:
    state, wal, samples, rows = _fixture(tmp_path, state_rows=0, wal_rows=1)
    wal.write_text(json.dumps(rows[0]), encoding="utf-8")
    with pytest.raises(subject.MigrationError, match="incomplete"):
        subject.audit_stopped_progress(
            state_path=state,
            wal_path=wal,
            sample_manifest_path=samples,
        )


def test_audit_rejects_tuple_tampering_even_with_resealed_state(tmp_path: Path) -> None:
    state, wal, samples, rows = _fixture(tmp_path, state_rows=1, wal_rows=1)
    rows[0]["replicate_id"] = 9
    payload = (subject._canonical(rows[0]) + "\n").encode("utf-8")
    wal.write_bytes(payload)
    state_value = json.loads(state.read_text(encoding="utf-8"))
    state_value.pop("integrity")
    state_value["partial_results"].update(
        {"bytes": len(payload), "sha256": subject.hashlib.sha256(payload).hexdigest()}
    )
    _write_canonical(state, seal_manifest(state_value))
    with pytest.raises(subject.MigrationError, match="noncanonical WAL"):
        subject.audit_stopped_progress(
            state_path=state,
            wal_path=wal,
            sample_manifest_path=samples,
        )


def test_readonly_receipt_is_create_only_and_integrity_checked(tmp_path: Path) -> None:
    path = tmp_path / "migration" / "receipt.json"
    receipt = seal_manifest(
        {
            "schema_version": subject.RECEIPT_SCHEMA,
            "status": "passed",
            "migration": {
                "old_prefix_preserved": True,
                "old_rows_reused_in_new_suite": False,
            },
            "new_suite": {
                **subject.TARGET_VLLM_CONTRACT,
                "replicates": 5,
                "expected_total_rows": subject.EXPECTED_NEW_TOTAL_CASES,
            },
        }
    )
    subject.write_readonly_receipt(path, receipt)
    result = subject.validate_receipt(path)
    assert result["status"] == "passed"
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    with pytest.raises(subject.MigrationError, match="already exists"):
        subject.write_readonly_receipt(path, receipt)


def test_live_process_gate_requires_exact_pane_group_and_exclusive_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_production_pids", lambda: [123])
    monkeypatch.setattr(subject, "_tmux_pane_pid", lambda: 456)
    monkeypatch.setattr(subject.os, "getpgid", lambda _pid: 456)
    monkeypatch.setattr(
        subject,
        "_gpu0",
        lambda: {"identity": "0, GPU-test, A30, 24576", "compute_pids": [123]},
    )
    value = subject.validate_live_process()
    assert value["python_pid"] == 123
    monkeypatch.setattr(
        subject,
        "_gpu0",
        lambda: {"identity": "0, GPU-test, A30, 24576", "compute_pids": [123, 999]},
    )
    with pytest.raises(subject.MigrationError, match="exclusively"):
        subject.validate_live_process()


def test_controlled_stop_sends_only_tmux_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(subject, "validate_live_process", lambda: {"python_pid": 123})
    monkeypatch.setattr(
        subject,
        "audit_state_bound_prefix",
        lambda **_kwargs: {"completed_cases": 10},
    )
    monkeypatch.setattr(subject, "_wait_for_next_commit", lambda *_args: 11)
    monkeypatch.setattr(subject.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(subject, "_production_pids", lambda: [])
    monkeypatch.setattr(
        subject,
        "_gpu0",
        lambda: {"identity": "0, GPU-test, A30, 24576", "compute_pids": []},
    )
    monkeypatch.setattr(subject, "_lock_is_free", lambda _path: True)
    monkeypatch.setattr(
        subject,
        "_gpu",
        lambda index: {
            "identity": f"{index}, GPU-test-{index}, A30, 24576",
            "compute_pids": [],
        },
    )
    monkeypatch.setattr(
        subject,
        "audit_stopped_progress",
        lambda **_kwargs: {"resume_contract_valid": True},
    )

    def fake_command(args, *, check=True):
        commands.append(list(args))
        return subprocess.CompletedProcess(
            args=list(args),
            returncode=(1 if list(args)[:2] == ["tmux", "has-session"] else 0),
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(subject, "_command", fake_command)
    evidence = subject.controlled_stop(boundary_wait_seconds=1, exit_wait_seconds=1)
    assert evidence["signal"] == "SIGINT_via_tmux_C-c"
    assert ["tmux", "send-keys", "-t", f"{subject.OLD_SESSION}:0.0", "C-c"] in commands
    assert not any("kill" in command for args in commands for command in args)
