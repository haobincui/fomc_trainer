from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs.generation import generate_paper_chk2_downstream_v6 as v6
from jobs.generation import publish_paper_chk2_downstream_v6 as publisher


def _receipt(**updates):
    payload = {
        "schema_version": "paper-chk2-downstream-execution-v1",
        "status": "complete",
        "phase": "all",
        "configured_concurrency": 128,
        "maximum_concurrency": 128,
        "runner_sha256": "1" * 64,
        "implementation_composite_sha256": "2" * 64,
        "source_handoff_manifest_sha256": "3" * 64,
        "source_handoff_manifest_file_sha256": "4" * 64,
        "run_binding_sha256": "5" * 64,
        "prompt_contract_sha256": "6" * 64,
        "official_reference_bank_sha256": "7" * 64,
        "tokenizer_contract_sha256": "8" * 64,
        "provider_identities": {"returned_model": "deepseek-v4-flash"},
        "source_audit_provider_calls": 0,
        "cache_counts": {"rewrite_primary": 1, "terminal": 1},
        "terminal_cache_count": 1,
        "artifacts": {"terminal_manifest": {"sha256": "9" * 64}},
    }
    payload.update(updates)
    payload["receipt_sha256"] = publisher.sha256_text(publisher.canonical_json(payload))
    return payload


def test_execution_receipt_requires_exact_complete_concurrency_128_contract() -> None:
    publisher._validate_execution_receipt_shape(_receipt())
    publisher._validate_execution_receipt_shape(_receipt(phase="verify"))

    for update, match in (
        ({"configured_concurrency": 8}, "concurrency"),
        ({"status": "running"}, "incomplete"),
        ({"phase": "preflight"}, "phase"),
        ({"schema_version": "wrong"}, "schema"),
    ):
        with pytest.raises(publisher.PublicationError, match=match):
            publisher._validate_execution_receipt_shape(_receipt(**update))

    with pytest.raises(publisher.PublicationError, match="filename/phase"):
        publisher._validate_execution_receipt_shape(
            _receipt(phase="verify"), expected_phase="all"
        )


def test_execution_receipt_rejects_extra_field_and_self_sha_drift() -> None:
    extra = _receipt()
    extra["unexpected"] = True
    with pytest.raises(publisher.PublicationError, match="field drift"):
        publisher._validate_execution_receipt_shape(extra)

    drifted = _receipt()
    drifted["implementation_composite_sha256"] = "0" * 64
    with pytest.raises(publisher.PublicationError, match="self-SHA drift"):
        publisher._validate_execution_receipt_shape(drifted)


def test_release_binds_exact_publisher_and_token_replay_implementations() -> None:
    contract = publisher._publisher_implementation_contract()
    assert set(contract["artifacts"]) == {
        "downstream_v6_publisher",
        "token_replay_publisher",
    }
    manifest = {"publisher_implementation": contract}
    handoff = {"publisher_implementation": contract}
    publisher._validate_publisher_implementation_binding(manifest, handoff)

    drifted = json.loads(json.dumps(contract))
    drifted["artifacts"]["downstream_v6_publisher"]["sha256"] = "0" * 64
    with pytest.raises(publisher.PublicationError, match="publisher implementation"):
        publisher._validate_publisher_implementation_binding(
            {"publisher_implementation": drifted}, handoff
        )


def test_safe_artifact_rejects_traversal_and_hash_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "record.json"
    artifact.write_text(json.dumps({"ok": True}), encoding="utf-8")
    descriptor = {
        "path": artifact.name,
        "bytes": artifact.stat().st_size,
        "sha256": publisher.sha256_file(artifact),
    }
    assert (
        publisher._safe_artifact_path(tmp_path, descriptor, label="record") == artifact
    )

    with pytest.raises(publisher.PublicationError, match="unsafe"):
        publisher._safe_artifact_path(
            tmp_path,
            {**descriptor, "path": "../record.json"},
            label="record",
        )

    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    target = real / "x.json"
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(publisher.PublicationError, match="symlink"):
        publisher._safe_artifact_path(
            tmp_path,
            {
                "path": "linked/x.json",
                "bytes": target.stat().st_size,
                "sha256": publisher.sha256_file(target),
            },
            label="linked record",
        )
    with pytest.raises(publisher.PublicationError, match="SHA drift"):
        publisher._safe_artifact_path(
            tmp_path,
            {**descriptor, "sha256": "0" * 64},
            label="record",
        )


def test_cache_inventory_rejects_nested_json(tmp_path: Path) -> None:
    nested = tmp_path / "cache" / "rewrite_primary" / "nested"
    nested.mkdir(parents=True)
    (nested / "x.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(publisher.PublicationError, match="nested"):
        publisher._strict_cache_inventory(tmp_path, allowed_roles={"rewrite_primary"})


def test_existing_release_is_verified_without_old_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    sentinel = {"status": "complete"}
    monkeypatch.setattr(publisher, "verify_release", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(
        publisher,
        "_verify_source_handoff",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("old inputs touched")),
    )
    assert publisher.publish_release(release_root=release) is sentinel


def test_verify_downstream_calls_real_terminal_replay_with_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard the publisher/producer replay signature at their integration seam."""

    monkeypatch.setattr(publisher, "SPLITS", ("train",))
    monkeypatch.setattr(publisher.v5, "EXPECTED_TOTAL", 1)
    implementation = {
        "artifacts": {"downstream_v6": {"sha256": "d" * 64}},
        "composite_sha256": "e" * 64,
    }
    monkeypatch.setattr(v6, "_implementation_contract", lambda: implementation)
    monkeypatch.setattr(v6, "_prompt_contract", lambda **kwargs: {})
    monkeypatch.setattr(v6, "load_and_verify_downstream", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        v6.official_v2, "deserialize_official_reference_bank", lambda payload: {}
    )

    digest = publisher.sha256_text("publisher-real-replay")
    row = publisher.v5.PreparedRow(
        sample_id="sample-train-0",
        split="train",
        source_split="train",
        split_index=0,
        source_line_number=1,
        generation_manifest_line_number=1,
        meeting_date="2021-12-15",
        atomic_topic="inflation",
        section_style_id="economic_conditions",
        prompt="prompt",
        provided_data="Inflation rose.",
        source_analysis="Inflation rose.",
        prompt_sha256=digest,
        provided_data_sha256=digest,
        source_analysis_sha256=digest,
        candidate_response_sha256=digest,
        source_answer_sha256=digest,
        source_response_sha256=digest,
        generation_manifest_row_sha256=digest,
        source_row_sha256=digest,
    )
    source_audit = {
        "complete": True,
        "machine_pass": False,
        "reasons": ["source_claim_not_supported"],
        "result": {"primary": {}},
        "provider": {},
        "contract_repair_used": False,
    }
    handoff_root = tmp_path / "handoff"
    sealed = handoff_root / "sealed_source"
    sealed.mkdir(parents=True)
    sealed_reference = sealed / "official_pre_action_reference_bank.jsonl"
    sealed_reference.write_text("{}\n", encoding="utf-8")
    signed_handoff_sha = "a" * 64
    source = publisher.VerifiedSourceHandoff(
        handoff=SimpleNamespace(
            root=handoff_root,
            prepared={"train": (row,)},
            source_results={row.sample_id: source_audit},
        ),
        manifest_file_sha256="f" * 64,
        signed_manifest_sha256=signed_handoff_sha,
        source_acquisition_root=tmp_path / "source-acquisition",
    )

    downstream_root = tmp_path / "downstream"
    downstream_root.mkdir()
    reference_path = downstream_root / "official_pre_action_reference_bank.jsonl"
    reference_path.write_bytes(sealed_reference.read_bytes())
    prompt_path = downstream_root / "prompt_contract.json"
    prompt_path.write_text("{}\n", encoding="utf-8")
    run_binding_sha = v6._run_binding_sha(
        implementation["composite_sha256"], signed_handoff_sha
    )
    terminal = v6._process_downstream_terminal(
        row,
        output=downstream_root,
        tokenizer=object(),
        reference_bank={},
        official_reference_bank_sha256=publisher.sha256_file(reference_path),
        backend=object(),
        identity=publisher.v5.ProviderIdentityRegistry(),
        environment={},
        config=publisher.v5.ProviderConfig(),
        code_sha256=run_binding_sha,
        source_audit=source_audit,
    )

    artifacts = downstream_root / "artifacts"
    artifacts.mkdir()
    terminal_path = artifacts / "train_terminal.jsonl"
    sft_path = artifacts / "train_sft.jsonl"
    manifest_path = artifacts / "train_manifest.jsonl"
    terminal_path.write_text(publisher.canonical_json(terminal) + "\n", encoding="utf-8")
    sft_path.write_text("", encoding="utf-8")
    manifest_path.write_text("", encoding="utf-8")

    def descriptor(path: Path, *, rows: int | None = None) -> dict[str, object]:
        value: dict[str, object] = {
            "path": str(path.relative_to(downstream_root)),
            "bytes": path.stat().st_size,
            "sha256": publisher.sha256_file(path),
        }
        if rows is not None:
            value["rows"] = rows
        return value

    final = {
        "schema_version": v6.SCHEMA_VERSION,
        "status": "complete",
        "quality_status": "passed",
        "total_source_rows": 1,
        "terminal_classified": 1,
        "unresolved_failure_count": 0,
        "source_handoff_manifest_sha256": signed_handoff_sha,
        "implementation_composite_sha256": implementation["composite_sha256"],
        "run_binding_sha256": run_binding_sha,
        "prompt_contract_sha256": publisher.sha256_file(prompt_path),
        "official_reference_bank_sha256": publisher.sha256_file(reference_path),
        "artifacts": {
            "train": {
                "terminal": descriptor(terminal_path, rows=1),
                "sft_candidate": descriptor(sft_path, rows=0),
                "manifest": descriptor(manifest_path, rows=0),
            }
        },
    }
    final_path = downstream_root / "final_summary.json"
    final_path.write_text(publisher.canonical_json(final) + "\n", encoding="utf-8")
    receipt = _receipt(
        runner_sha256=implementation["artifacts"]["downstream_v6"]["sha256"],
        implementation_composite_sha256=implementation["composite_sha256"],
        source_handoff_manifest_sha256=signed_handoff_sha,
        source_handoff_manifest_file_sha256=source.manifest_file_sha256,
        run_binding_sha256=run_binding_sha,
        prompt_contract_sha256=publisher.sha256_file(prompt_path),
        official_reference_bank_sha256=publisher.sha256_file(reference_path),
        tokenizer_contract_sha256=publisher.sha256_text(
            publisher.canonical_json(
                publisher.v5._expected_tokenizer_runtime_contract()
            )
        ),
        provider_identities={},
        cache_counts={"terminal": 1},
        terminal_cache_count=1,
        artifacts={"final_summary": descriptor(final_path)},
    )
    receipt_root = downstream_root / "execution_receipts"
    receipt_root.mkdir()
    (receipt_root / "all.json").write_text(
        publisher.canonical_json(receipt) + "\n", encoding="utf-8"
    )

    # A source-rejected fixture has no PASS row, so verification stops immediately
    # after the real producer replay. Omitting tokenizer= at that call site instead
    # produces a wrapped missing-argument error and fails this regression assertion.
    with pytest.raises(publisher.PublicationError, match="train has no PASS training rows"):
        publisher._verify_downstream(root=downstream_root, source=source, tokenizer=object())


def test_publish_happy_path_is_pass_only_and_binds_both_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(publisher.v5, "EXPECTED_TOTAL", 3)
    source_handoff_root = tmp_path / "source_handoff"
    source_handoff_root.mkdir()
    signed_source_sha = "a" * 64
    source_manifest = {"manifest_sha256": signed_source_sha}
    source_manifest_path = source_handoff_root / "handoff_manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    sealed_source = source_handoff_root / "sealed_source"
    sealed_source.mkdir()
    (sealed_source / "source_admission_receipt.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (sealed_source / "prompt_contract.json").write_text("{}\n", encoding="utf-8")
    (sealed_source / "official_pre_action_reference_bank.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    source = publisher.VerifiedSourceHandoff(
        handoff=SimpleNamespace(root=source_handoff_root),
        manifest_file_sha256=publisher.sha256_file(source_manifest_path),
        signed_manifest_sha256=signed_source_sha,
        source_acquisition_root=tmp_path / "source_acquisition",
    )

    downstream_root = tmp_path / "downstream"
    (downstream_root / "execution_receipts").mkdir(parents=True)
    receipt = _receipt(
        source_handoff_manifest_sha256=signed_source_sha,
        source_handoff_manifest_file_sha256=source.manifest_file_sha256,
        cache_counts={"rewrite_primary": 1, "terminal": 3},
        terminal_cache_count=3,
    )
    execution_path = downstream_root / "execution_receipts" / "all.json"
    execution_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    (downstream_root / "prompt_contract.json").write_text("{}\n", encoding="utf-8")
    (downstream_root / "final_summary.json").write_text("{}\n", encoding="utf-8")

    terminals = {}
    pass_rows = {}
    pass_manifests = {}
    tokenizer_rows = []
    for split in publisher.SPLITS:
        sample_id = f"sample-{split}"
        prompt = f"prompt for {split}"
        response = f"reasoning for {split}\n</think>\nanswer for {split}."
        terminal = {
            "sample_id": sample_id,
            "split": split,
            "source_index": 0,
            "terminal_status": publisher.v5.TERMINAL_PASS,
            "rejection_stage": None,
            "rejection_reasons": [],
            "source_analysis_sha256": "b" * 64,
            "provided_data_sha256": "c" * 64,
            "source_audit": {"complete": True, "machine_pass": True},
            "validator_a": {},
            "validator_b": {},
            "repair_history": [],
        }
        terminals[split] = (terminal,)
        pass_rows[split] = ({"prompt": prompt, "response": response},)
        pass_manifests[split] = (
            {
                "sample_id": sample_id,
                "split": split,
                "source_index": 0,
                "meeting_date": "2024-01-01",
                "atomic_topic": "topic",
                "section_style_id": "style",
                "terminal_status": publisher.v5.TERMINAL_PASS,
                "source_analysis_sha256": "b" * 64,
                "prompt_sha256": publisher.sha256_text(prompt),
                "response_sha256": publisher.sha256_text(response),
                "lineage": {"training_only": True},
            },
        )
        tokenizer_rows.append(
            {
                "sample_id": sample_id,
                "split": split,
                "prompt_tokens": 3,
                "reasoning_tokens": 4,
                "answer_tokens": 5,
                "completion_tokens": 6,
                "total_tokens": 9,
                "masked_prompt_tokens": 3,
                "unmasked_completion_tokens": 6,
            }
        )
    downstream = publisher.VerifiedDownstream(
        root=downstream_root,
        receipt=receipt,
        receipt_path=execution_path,
        final_summary={},
        terminals=terminals,
        pass_rows=pass_rows,
        pass_manifests=pass_manifests,
        tokenizer_replay=tuple(tokenizer_rows),
        provider_identities={"returned_model": "deepseek-v4-flash"},
    )
    monkeypatch.setattr(publisher, "_verify_source_handoff", lambda **kwargs: source)
    monkeypatch.setattr(publisher, "_verify_downstream", lambda **kwargs: downstream)
    monkeypatch.setattr(
        publisher.v5, "_verify_tokenizer_runtime_contract", lambda tokenizer: {}
    )

    def token_replay(**kwargs):
        del kwargs
        return {
            "prompt_tokens": 3,
            "reasoning_tokens": 4,
            "answer_tokens": 5,
            "completion_tokens": 6,
            "total_tokens": 9,
            "masked_prompt_tokens": 3,
            "unmasked_completion_tokens": 6,
        }

    monkeypatch.setattr(publisher.v5_publisher, "_token_replay", token_replay)
    release_root = tmp_path / "release"
    manifest = publisher.publish_release(
        source_acquisition_root=tmp_path / "source_acquisition",
        source_handoff_root=source_handoff_root,
        downstream_root=downstream_root,
        release_root=release_root,
        tokenizer=object(),
    )

    assert manifest["split_pass_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert manifest["source_handoff"]["manifest_sha256"] == signed_source_sha
    assert (release_root / "audits/source_admission.jsonl").is_file()
    assert (release_root / "audits/validator_a.jsonl").is_file()
    assert (release_root / "audits/validator_b.jsonl").is_file()
    assert (release_root / "audits/repair_history.jsonl").is_file()
    assert (
        release_root / "provenance/official_pre_action_reference_bank.jsonl"
    ).is_file()
    for split in publisher.SPLITS:
        rows = publisher._read_jsonl(
            release_root / "minutes_alignment" / f"{split}.jsonl",
            label=split,
        )
        assert len(rows) == 1
        assert set(rows[0]) == {"prompt", "response"}

    data_path = release_root / "minutes_alignment" / "train.jsonl"
    data_path.write_text('{"prompt":"tampered","response":"tampered"}\n')
    with pytest.raises(publisher.PublicationError, match="byte drift"):
        publisher.verify_release(release_root, tokenizer=object())
