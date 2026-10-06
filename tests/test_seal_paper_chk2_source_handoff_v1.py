from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from jobs.generation import seal_paper_chk2_source_handoff_v1 as sealer


@pytest.fixture(autouse=True)
def _stub_reference_deserialize(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sealer, "deserialize_official_reference_bank", lambda _: object()
    )
    monkeypatch.setattr(sealer.v5, "verify_official_reference_bank", lambda *_: None)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _empty_handoff(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    handoff = tmp_path / "handoff"
    code_sha = sealer.v5._implementation_contract()["composite_sha256"]
    prompt_contract = sealer.v5._prompt_contract(
        code_sha256=code_sha, config=sealer.v5.ProviderConfig()
    )
    empty_digest = sealer.v5.sha256_text(sealer.v5.canonical_json([]))
    empty_prepare = {
        "split_counts": {split: 0 for split in sealer.SPLITS},
        "meeting_counts": {split: 0 for split in sealer.SPLITS},
        "sample_id_digest": empty_digest,
        "source_rows_digest": empty_digest,
    }
    empty_prepare["prepare_manifest_sha256"] = sealer.v5.sha256_text(
        sealer.v5.canonical_json(empty_prepare)
    )
    source_values = {
        "prompt_contract.json": json.dumps(prompt_contract) + "\n",
        "official_pre_action_reference_bank.jsonl": "",
        "prepare_manifest.json": json.dumps(empty_prepare) + "\n",
        "preparation_summary.json": "{}\n",
        "source_admission_receipt.json": json.dumps({"provider_identities": {}}) + "\n",
    }
    for filename, value in source_values.items():
        _write(source / filename, value)
        _write(handoff / "sealed_source" / filename, value)
    for split in sealer.SPLITS:
        _write(source / "prepared" / f"{split}.jsonl", "")
        _write(handoff / "records" / f"{split}.jsonl", "")
        _write(source / "source_audit" / f"{split}.jsonl", "")
        _write(handoff / "sealed_source/source_audit" / f"{split}.jsonl", "")
    receipt = {
        "schema_version": sealer.v5.SOURCE_AUDIT_SCHEMA_VERSION,
        "status": "source_admission_complete",
        "quality_status": "passed",
        "source_rows": 0,
        "admitted_rows": 0,
        "rejected_rows": 0,
        "unresolved_rows": 0,
        "status_counts": {split: {} for split in sealer.SPLITS},
        "artifacts": {
            split: {
                "rows": 0,
                "bytes": 0,
                "sha256": sealer.v5.sha256_file(
                    source / "source_audit" / f"{split}.jsonl"
                ),
            }
            for split in sealer.SPLITS
        },
        "provider_identities": {},
        "lineage": dict(sealer.v5.LINEAGE),
    }
    source_values["source_admission_receipt.json"] = json.dumps(receipt) + "\n"
    _write(
        source / "source_admission_receipt.json",
        source_values["source_admission_receipt.json"],
    )
    _write(
        handoff / "sealed_source/source_admission_receipt.json",
        source_values["source_admission_receipt.json"],
    )
    bindings = {
        "implementation_composite_sha256": code_sha,
        "prompt_contract_sha256": sealer.v5.sha256_file(
            source / "prompt_contract.json"
        ),
        "official_reference_bank_sha256": sealer.v5.sha256_file(
            source / "official_pre_action_reference_bank.jsonl"
        ),
        "prepare_manifest_sha256": sealer.v5.sha256_file(
            source / "prepare_manifest.json"
        ),
        "preparation_summary_sha256": sealer.v5.sha256_file(
            source / "preparation_summary.json"
        ),
        "source_admission_receipt_sha256": sealer.v5.sha256_file(
            source / "source_admission_receipt.json"
        ),
    }
    artifacts = {}
    for split in sealer.SPLITS:
        path = handoff / "records" / f"{split}.jsonl"
        artifacts[split] = {
            "path": f"records/{split}.jsonl",
            "bytes": 0,
            "rows": 0,
            "sha256": sealer.v5.sha256_file(path),
        }
    sealed = {}
    for filename in source_values:
        path = handoff / "sealed_source" / filename
        sealed[filename] = {
            "path": f"sealed_source/{filename}",
            "bytes": path.stat().st_size,
            "sha256": sealer.v5.sha256_file(path),
        }
    for split in sealer.SPLITS:
        filename = f"source_audit/{split}.jsonl"
        path = handoff / "sealed_source" / filename
        sealed[filename] = {
            "path": f"sealed_source/{filename}",
            "bytes": 0,
            "rows": 0,
            "sha256": sealer.v5.sha256_file(path),
        }
    manifest = {
        "schema_version": sealer.SCHEMA_VERSION,
        "status": "sealed",
        "source_acquisition_root": str(source.resolve()),
        "total_rows": 0,
        "split_counts": {split: 0 for split in sealer.SPLITS},
        "status_counts": {},
        "source_provider_identities": {},
        "sample_id_digest": sealer.v5.sha256_text(sealer.v5.canonical_json([])),
        "bindings": bindings,
        "artifacts": artifacts,
        "sealed_source_artifacts": sealed,
    }
    manifest["manifest_sha256"] = sealer.v5.sha256_text(
        sealer.v5.canonical_json(manifest)
    )
    _write(
        handoff / "handoff_manifest.json",
        json.dumps(manifest, sort_keys=True) + "\n",
    )
    return handoff


def _one_row_acquisition(tmp_path: Path) -> tuple[Path, sealer.v5.PreparedRow]:
    source = tmp_path / "one-source"
    analysis = "Credit conditions eased."
    provided = '{"evidence":"Credit conditions eased."}'
    prompt_text = "Analyze supplied evidence."
    row = sealer.v5.PreparedRow(
        sample_id="sample-1",
        split="train",
        source_split="train",
        split_index=0,
        source_line_number=1,
        generation_manifest_line_number=1,
        meeting_date="2020-01-01",
        atomic_topic="Credit",
        section_style_id="style-1",
        prompt=prompt_text,
        provided_data=provided,
        source_analysis=analysis,
        prompt_sha256=sealer.v5.sha256_text(prompt_text),
        provided_data_sha256=sealer.v5.sha256_text(provided),
        source_analysis_sha256=sealer.v5.sha256_text(analysis),
        candidate_response_sha256="1" * 64,
        source_answer_sha256=sealer.v5.sha256_text(analysis),
        source_response_sha256="2" * 64,
        generation_manifest_row_sha256="3" * 64,
        source_row_sha256="4" * 64,
    )
    code_sha = sealer.v5._implementation_contract()["composite_sha256"]
    prompt_contract = sealer.v5._prompt_contract(
        code_sha256=code_sha, config=sealer.v5.ProviderConfig()
    )
    _write(source / "prompt_contract.json", json.dumps(prompt_contract) + "\n")
    _write(source / "official_pre_action_reference_bank.jsonl", "reference\n")
    for split in sealer.SPLITS:
        rows = [row.to_dict()] if split == "train" else []
        _write(
            source / "prepared" / f"{split}.jsonl",
            "".join(sealer.v5.canonical_json(value) + "\n" for value in rows),
        )
    sample_digest = sealer.v5.sha256_text(sealer.v5.canonical_json([row.sample_id]))
    rows_digest = sealer.v5.sha256_text(
        sealer.v5.canonical_json([row.source_row_sha256])
    )
    split_counts = {"train": 1, "validation": 0, "test": 0}
    meeting_counts = {"train": 1, "validation": 0, "test": 0}
    prepare = {
        "split_counts": split_counts,
        "meeting_counts": meeting_counts,
        "sample_id_digest": sample_digest,
        "source_rows_digest": rows_digest,
    }
    prepare["prepare_manifest_sha256"] = sealer.v5.sha256_text(
        sealer.v5.canonical_json(prepare)
    )
    _write(source / "prepare_manifest.json", json.dumps(prepare) + "\n")
    _write(
        source / "preparation_summary.json",
        json.dumps(
            {
                "source": {
                    "split_counts": split_counts,
                    "meeting_counts": meeting_counts,
                    "sample_id_sha256": sample_digest,
                    "rows_sha256": rows_digest,
                }
            }
        )
        + "\n",
    )
    raw_content_sha = sealer.v5.sha256_text("{}")
    raw_reasoning_sha = sealer.v5.sha256_text("reasoning")
    provider = {
        "returned_model": sealer.v5.MODEL,
        "system_fingerprint": "fingerprint",
        "response_id": "response-1",
        "raw_content_sha256": raw_content_sha,
        "raw_reasoning_sha256": raw_reasoning_sha,
    }
    result = {
        "complete": True,
        "machine_pass": True,
        "reasons": [],
        "contract_repair_used": False,
        "result": {
            "primary": {"machine_pass": True},
            "adjudication": None,
            "contract_repair": None,
        },
        "provider": {sealer.v5.ROLE_SOURCE_AUDIT_PRIMARY: provider},
    }
    sample_hash = sealer.v5.sha256_text(row.sample_id)
    terminal = {
        "binding": {
            "schema_version": sealer.v5.SOURCE_AUDIT_SCHEMA_VERSION,
            "sample_id": row.sample_id,
            "source_analysis_sha256": row.source_analysis_sha256,
            "provided_data_sha256": row.provided_data_sha256,
            "code_sha256": code_sha,
        },
        "result": result,
        "result_sha256": sealer.v5.sha256_text(sealer.v5.canonical_json(result)),
    }
    _write(
        source / "cache/source_terminal" / f"{sample_hash}.json",
        json.dumps(terminal) + "\n",
    )
    provider_cache = {
        "raw_content_sha256": raw_content_sha,
        "raw_reasoning_sha256": raw_reasoning_sha,
    }
    _write(
        source / "cache" / sealer.v5.ROLE_SOURCE_AUDIT_PRIMARY / f"{sample_hash}.json",
        json.dumps(provider_cache) + "\n",
    )
    audit_rows = {
        "train": [{"sample_id": row.sample_id, "split": "train", **result}],
        "validation": [],
        "test": [],
    }
    artifacts = {}
    for split in sealer.SPLITS:
        path = source / "source_audit" / f"{split}.jsonl"
        _write(
            path,
            "".join(
                sealer.v5.canonical_json(value) + "\n" for value in audit_rows[split]
            ),
        )
        artifacts[split] = {
            "rows": len(audit_rows[split]),
            "bytes": path.stat().st_size,
            "sha256": sealer.v5.sha256_file(path),
        }
    receipt = {
        "schema_version": sealer.v5.SOURCE_AUDIT_SCHEMA_VERSION,
        "status": "source_admission_complete",
        "quality_status": "passed",
        "source_rows": 1,
        "admitted_rows": 1,
        "rejected_rows": 0,
        "unresolved_rows": 0,
        "status_counts": {
            "train": {"PASS": 1},
            "validation": {},
            "test": {},
        },
        "artifacts": artifacts,
        "provider_identities": {},
        "lineage": dict(sealer.v5.LINEAGE),
    }
    _write(source / "source_admission_receipt.json", json.dumps(receipt) + "\n")
    return source, row


def test_load_empty_handoff_replays_all_bound_artifacts(tmp_path: Path) -> None:
    handoff = sealer.load_and_verify_source_handoff(
        _empty_handoff(tmp_path), expected_total=0
    )
    assert handoff.rows == ()
    assert handoff.source_results == {}
    assert set(handoff.prepared) == set(sealer.SPLITS)


def test_seal_happy_path_is_self_contained(tmp_path: Path) -> None:
    old_handoff = _empty_handoff(tmp_path)
    source = Path(
        json.loads((old_handoff / "handoff_manifest.json").read_text())[
            "source_acquisition_root"
        ]
    )
    shutil.rmtree(old_handoff)
    prepare = {
        "split_counts": {split: 0 for split in sealer.SPLITS},
        "meeting_counts": {split: 0 for split in sealer.SPLITS},
        "sample_id_digest": sealer.v5.sha256_text(sealer.v5.canonical_json([])),
        "source_rows_digest": sealer.v5.sha256_text(sealer.v5.canonical_json([])),
    }
    prepare["prepare_manifest_sha256"] = sealer.v5.sha256_text(
        sealer.v5.canonical_json(prepare)
    )
    _write(source / "prepare_manifest.json", json.dumps(prepare) + "\n")
    _write(
        source / "preparation_summary.json",
        json.dumps(
            {
                "source": {
                    "split_counts": {split: 0 for split in sealer.SPLITS},
                    "meeting_counts": {split: 0 for split in sealer.SPLITS},
                    "sample_id_sha256": sealer.v5.sha256_text(
                        sealer.v5.canonical_json([])
                    ),
                    "rows_sha256": sealer.v5.sha256_text(sealer.v5.canonical_json([])),
                }
            }
        )
        + "\n",
    )
    result = sealer.seal_source_handoff(source, old_handoff, expected_total=0)
    assert result.rows == ()
    assert (old_handoff / "sealed_source/source_admission_receipt.json").is_file()


def test_nonempty_seal_survives_old_acquisition_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, row = _one_row_acquisition(tmp_path)
    handoff_root = tmp_path / "one-handoff"
    monkeypatch.setattr(sealer.v5, "_validate_source_terminal_result", lambda *_: None)
    monkeypatch.setattr(
        sealer.v5, "_resume_source_terminal_provider_caches", lambda **_: None
    )
    sealed = sealer.seal_source_handoff(source, handoff_root, expected_total=1)
    assert sealed.prepared["train"][0] == row
    assert sealed.records["train"][0]["status"] == "PASS"
    shutil.rmtree(source)
    loaded = sealer.load_and_verify_source_handoff(handoff_root, expected_total=1)
    assert loaded.prepared["train"][0].source_analysis == row.source_analysis
    assert loaded.source_results[row.sample_id]["machine_pass"] is True


def test_provider_index_path_tamper_is_rejected_after_resigning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _row = _one_row_acquisition(tmp_path)
    handoff_root = tmp_path / "tamper-handoff"
    monkeypatch.setattr(sealer.v5, "_validate_source_terminal_result", lambda *_: None)
    monkeypatch.setattr(
        sealer.v5, "_resume_source_terminal_provider_caches", lambda **_: None
    )
    sealer.seal_source_handoff(source, handoff_root, expected_total=1)
    record_path = handoff_root / "records/train.jsonl"
    record = json.loads(record_path.read_text())
    record["provider_cache_index"][sealer.v5.ROLE_SOURCE_AUDIT_PRIMARY]["path"] = (
        "sealed_source/cache/source_audit_primary/wrong.json"
    )
    _write(record_path, sealer.v5.canonical_json(record) + "\n")
    manifest_path = handoff_root / "handoff_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["train"]["bytes"] = record_path.stat().st_size
    manifest["artifacts"]["train"]["sha256"] = sealer.v5.sha256_file(record_path)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = sealer.v5.sha256_text(
        sealer.v5.canonical_json(manifest)
    )
    _write(manifest_path, json.dumps(manifest) + "\n")
    with pytest.raises(sealer.SourceHandoffError, match="provider path drift"):
        sealer.load_and_verify_source_handoff(handoff_root, expected_total=1)


def test_manifest_tamper_is_rejected(tmp_path: Path) -> None:
    root = _empty_handoff(tmp_path)
    path = root / "handoff_manifest.json"
    value = json.loads(path.read_text())
    value["status"] = "changed"
    _write(path, json.dumps(value))
    with pytest.raises(sealer.SourceHandoffError, match="manifest drift"):
        sealer.load_and_verify_source_handoff(root, expected_total=0)


def test_missing_record_artifact_is_rejected(tmp_path: Path) -> None:
    root = _empty_handoff(tmp_path)
    (root / "records" / "validation.jsonl").unlink()
    with pytest.raises(sealer.SourceHandoffError, match="missing validation artifact"):
        sealer.load_and_verify_source_handoff(root, expected_total=0)


def test_prompt_code_binding_tamper_is_rejected(tmp_path: Path) -> None:
    root = _empty_handoff(tmp_path)
    _write(
        root / "sealed_source" / "prompt_contract.json",
        json.dumps({"code_sha256": "b" * 64}),
    )
    with pytest.raises(sealer.SourceHandoffError, match="artifact drift"):
        sealer.load_and_verify_source_handoff(root, expected_total=0)


def test_sealed_reference_tamper_is_rejected(tmp_path: Path) -> None:
    root = _empty_handoff(tmp_path)
    _write(root / "sealed_source" / "official_pre_action_reference_bank.jsonl", "x")
    with pytest.raises(sealer.SourceHandoffError, match="artifact drift"):
        sealer.load_and_verify_source_handoff(root, expected_total=0)
