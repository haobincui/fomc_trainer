from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from jobs.eval import seal_chk3_beta_core8_vllm_k5_fresh_root_addendum as addendum
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


def test_historical_receipts_bind_zero_row_failure() -> None:
    observed = addendum.audit_historical_receipts()
    assert observed["original_migration_receipt"]["sha256"] == (
        addendum.ORIGINAL_MIGRATION_SHA256
    )
    assert observed["async_output_remediation_receipt"]["sha256"] == (
        addendum.ASYNC_OUTPUT_REMEDIATION_SHA256
    )
    assert observed["historical_remediation_launcher"]["sha256"] == (
        addendum.PRE_ADDENDUM_LAUNCHER_SHA256
    )
    assert observed["zero_row_failure_bound"] is True
    assert observed["failed_generated_rows"] == 0
    assert [
        row["physical_gpu_index"]
        for row in observed["historical_physical_gpu_identities"]
    ] == [0, 1]


def test_idle_audit_rejects_gpu_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = [
        {"physical_gpu_index": 0, "uuid": "GPU-0", "pci_bus_id": "00:00.0"},
        {"physical_gpu_index": 1, "uuid": "GPU-1", "pci_bus_id": "01:00.0"},
    ]
    observed = [dict(row) for row in expected]
    observed[1]["uuid"] = "GPU-drift"
    monkeypatch.setattr(addendum, "_gpu_identity_rows", lambda: observed)
    with pytest.raises(addendum.FreshRootAddendumError, match="identities drifted"):
        addendum.audit_idle_gpus(expected)


def test_fresh_root_absence_rejects_directory_and_dangling_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate = tmp_path / "v2"
    monkeypatch.setattr(addendum, "PATCHED_BENCHMARK_ROOT", candidate)
    evidence = addendum.audit_fresh_root_absence()
    assert evidence["new_benchmark_root_absent_at_seal"] is True
    assert evidence["lexists"] is False

    candidate.mkdir()
    with pytest.raises(addendum.FreshRootAddendumError, match="must not exist"):
        addendum.audit_fresh_root_absence()
    candidate.rmdir()

    candidate.symlink_to(tmp_path / "missing-target")
    assert os.path.lexists(candidate)
    with pytest.raises(addendum.FreshRootAddendumError, match="must not exist"):
        addendum.audit_fresh_root_absence()


def test_readonly_addendum_is_create_only(tmp_path: Path) -> None:
    destination = tmp_path / "addendum.json"
    receipt = seal_manifest(
        {
            "schema_version": "unit-test",
            "status": "passed",
        }
    )
    addendum.write_readonly_receipt(destination, receipt)
    observed = json.loads(destination.read_text(encoding="utf-8"))
    assert (
        validate_manifest_integrity(observed) == receipt["integrity"]["payload_sha256"]
    )
    assert stat.S_IMODE(destination.stat().st_mode) == 0o444
    with pytest.raises(addendum.FreshRootAddendumError, match="create-only"):
        addendum.write_readonly_receipt(destination, receipt)


def test_wrong_confirmation_does_not_write_addendum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(addendum, "DEFAULT_RECEIPT", destination)
    assert addendum.main(["create", "--confirm", "wrong"]) == 1
    assert not destination.exists()


def test_launcher_requires_addendum_after_historical_migration() -> None:
    source = addendum.PATCHED_LAUNCHER.read_text(encoding="utf-8")
    assert "vllm_v1_async_output_fresh_root_addendum_receipt.json" in source
    assert "jobs.eval.seal_chk3_beta_core8_vllm_k5_fresh_root_addendum" in source
    assert "benchmark_max_num_seqs_chk1_core8_k5_v2" in source
    invocation = source.rsplit("prepare_or_validate", maxsplit=1)[1]
    migration_index = invocation.index("require_migration_receipt")
    historical_index = invocation.index("require_async_output_remediation_receipt")
    addendum_index = invocation.index("require_fresh_root_addendum_receipt")
    runtime_index = invocation.index("require_vllm_bf16_contract")
    assert migration_index < historical_index < addendum_index < runtime_index
    assert "jobs.eval.remediate_chk3_beta_core8_vllm_k5_v1_async_output" not in source
