import copy
import hashlib
import json

import pytest

from jobs.retrain_v2.import_sealed_stage import (
    SealedStageImportError,
    _canonical_sha256,
    _load_receipt,
    _validate_stored_execution_contract,
)


def _stored_contract():
    source_payload = {"files": [{"path": "trainer.py", "sha256": "a" * 64}]}
    environment_payload = {"python": {"version": "3.10.9"}}
    source_sha = _canonical_sha256(source_payload)
    environment_sha = _canonical_sha256(environment_payload)
    topology = {"chk1": {"launcher": "ddp", "policy_gpus": [0, 1], "world_size": 2}}
    deterministic = {
        "schema_version": 1,
        "source_bundle_sha256": source_sha,
        "environment_sha256": environment_sha,
        "stage_topology": topology,
    }
    return {
        "schema_version": 1,
        "created_at_utc": "2026-08-04T00:00:00Z",
        "source_bundle": {
            "algorithm": "sha256",
            "payload": source_payload,
            "sha256": source_sha,
        },
        "environment": {
            "algorithm": "sha256",
            "payload": environment_payload,
            "sha256": environment_sha,
        },
        "stage_topology": topology,
        "contract_sha256": _canonical_sha256(deterministic),
    }


def test_stored_execution_contract_is_validated_without_live_source_replay():
    contract = _stored_contract()
    assert _validate_stored_execution_contract(contract) == contract["contract_sha256"]


def test_stored_execution_contract_rejects_internal_tampering():
    contract = copy.deepcopy(_stored_contract())
    contract["source_bundle"]["payload"]["files"][0]["path"] = "tampered.py"
    with pytest.raises(SealedStageImportError, match="layer hash mismatch"):
        _validate_stored_execution_contract(contract)


def test_import_receipt_rejects_lineage_tampering(tmp_path):
    lineage = {"run_id": "source", "stage_id": "chk1"}
    receipt = {
        "schema_version": 1,
        "receipt_type": "training",
        "recorded_at_utc": "2026-08-04T00:00:00Z",
        "lineage": lineage,
        "lineage_sha256": hashlib.sha256(
            json.dumps(
                lineage,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    receipt["lineage"]["stage_id"] = "chk2"
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(SealedStageImportError, match="lineage hash mismatch"):
        _load_receipt(path, receipt_type="training")
