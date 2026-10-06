from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from jobs.eval import (
    remediate_chk3_beta_core8_vllm_k5_v1_async_output as remediation,
)
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


def test_production_failure_evidence_is_sealed_and_has_zero_rows() -> None:
    original = remediation.audit_original_migration()
    failed = remediation.audit_failed_attempt()
    assert original["sha256"] == remediation.ORIGINAL_MIGRATION_SHA256
    assert original["payload_sha256"] == (remediation.ORIGINAL_MIGRATION_PAYLOAD_SHA256)
    assert failed["launch"]["sha256"] == remediation.FAILED_LAUNCH_SHA256
    assert failed["state"]["sha256"] == remediation.FAILED_STATE_SHA256
    assert failed["generated_rows"] == 0
    assert failed["completed_chunks"] == 0
    assert failed["model_load_started"] is False
    assert failed["completion_wal"]["bytes"] == 0
    assert failed["completion_wal"]["sha256"] == remediation.EMPTY_SHA256
    assert failed["absolute_chunk_receipts"]["bytes"] == 0
    assert failed["canonical_generations_absent"] is True
    assert failed["sealed_run_manifest_absent"] is True


def test_patched_runner_omits_unsupported_v1_argument() -> None:
    binding = remediation._validate_patched_runner_source(remediation.PATCHED_RUNNER)
    assert binding["sha256"] != remediation.OLD_RUNNER_SHA256


def test_readonly_receipt_is_create_only(tmp_path: Path) -> None:
    destination = tmp_path / "receipt.json"
    receipt = seal_manifest(
        {
            "schema_version": "unit-test",
            "status": "passed",
        }
    )
    remediation.write_readonly_receipt(destination, receipt)
    observed = json.loads(destination.read_text(encoding="utf-8"))
    assert (
        validate_manifest_integrity(observed) == receipt["integrity"]["payload_sha256"]
    )
    assert stat.S_IMODE(destination.stat().st_mode) == 0o444
    with pytest.raises(remediation.RemediationError, match="create-only"):
        remediation.write_readonly_receipt(destination, receipt)


def test_wrong_confirmation_is_rejected_without_writing_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    destination = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(remediation, "DEFAULT_RECEIPT", destination)
    assert remediation.main(["create", "--confirm", "wrong"]) == 1
    assert not destination.exists()
