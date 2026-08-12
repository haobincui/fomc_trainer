from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.main.checkpoint_provenance import (
    CORRUPTED_MINUTES_ID,
    EVAL_ANALYSIS_SFT_ID,
    EVAL_BASE_ID,
    EVAL_LEGACY_GRPO_ID,
    EVAL_MINUTES_SFT_ID,
    RecoveryRequest,
    audit_overwritten_model,
    build_parser,
    fingerprint_tokenizer_payload,
    invalidate_run,
    recover_minutes_checkpoint,
    verify_checkpoint_artifact,
)
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_tokenizer(path: Path, marker: str) -> None:
    _write_json(
        path / "tokenizer.json",
        {"model": {"type": "BPE", "marker": marker}},
    )
    _write_json(
        path / "tokenizer_config.json",
        {"tokenizer_class": "LlamaTokenizerFast", "marker": marker},
    )
    _write_json(path / "special_tokens_map.json", {"eos_token": "</s>"})
    (path / "chat_template.jinja").write_text(f"template:{marker}\n", encoding="utf-8")


def _write_model(
    path: Path,
    *,
    weight: bytes,
    tokenizer_marker: str | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _write_json(path / "config.json", {"model_type": "llama", "hidden_size": 2})
    _write_json(path / "generation_config.json", {"max_length": 64})
    (path / "model.safetensors").write_bytes(weight)
    if tokenizer_marker is not None:
        _write_tokenizer(path, tokenizer_marker)


def _write_adapter(
    path: Path,
    *,
    parent: Path,
    tokenizer_marker: str,
    include_compatibility_keys: bool = False,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "base_model_name_or_path": str(parent.resolve()),
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 8,
        "lora_alpha": 16,
        "target_modules": ["q_proj", "v_proj"],
    }
    if include_compatibility_keys:
        config.update(
            {
                "corda_config": None,
                "eva_config": None,
                "exclude_modules": None,
                "lora_bias": False,
                "trainable_token_indices": None,
            }
        )
    _write_json(path / "adapter_config.json", config)
    (path / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    _write_tokenizer(path, tokenizer_marker)


def _request(root: Path) -> tuple[RecoveryRequest, Path]:
    foundation = root / "foundation"
    analysis = root / "analysis-sft"
    legacy_grpo = root / "legacy-grpo"
    historical = root / "historical-minutes-corrupted"
    overwrite_reference = root / "decision-grpo-reference"
    _write_model(
        foundation,
        weight=b"foundation-weights",
        tokenizer_marker="foundation-tokenizer",
    )
    _write_model(
        analysis,
        weight=b"analysis-weights",
        tokenizer_marker="analysis-tokenizer",
    )
    _write_model(legacy_grpo, weight=b"legacy-grpo-weights")
    _write_model(
        historical,
        weight=b"decision-weights",
        tokenizer_marker="stale-minutes-tokenizer",
    )
    _write_model(
        overwrite_reference,
        weight=b"decision-weights",
        tokenizer_marker="decision-tokenizer",
    )

    analysis_adapter = root / "analysis-adapter"
    legacy_adapter = root / "legacy-grpo-adapter"
    minutes_root = root / "minutes-adapter"
    minutes_checkpoint = minutes_root / "checkpoint-1668"
    _write_adapter(
        analysis_adapter,
        parent=foundation,
        tokenizer_marker="analysis-training-tokenizer",
    )
    _write_adapter(
        legacy_adapter,
        parent=foundation,
        tokenizer_marker="legacy-grpo-tokenizer",
    )
    _write_adapter(
        minutes_checkpoint,
        parent=analysis,
        tokenizer_marker="minutes-training-tokenizer",
        include_compatibility_keys=True,
    )
    _write_json(
        minutes_root / "trainer_state.json",
        {"global_step": 1668, "max_steps": 1668},
    )
    (minutes_root / "training_args.bin").write_bytes(b"training-arguments")
    _write_json(minutes_root / "train_results.json", {"train_loss": 0.1})
    (minutes_root / "loss_history.jsonl").write_text('{"loss":0.1}\n', encoding="utf-8")

    return (
        RecoveryRequest(
            foundation_model=foundation,
            analysis_sft_model=analysis,
            analysis_sft_adapter=analysis_adapter,
            legacy_grpo_model=legacy_grpo,
            legacy_grpo_adapter=legacy_adapter,
            minutes_adapter=minutes_checkpoint,
            historical_minutes_model=historical,
            overwrite_reference_model=overwrite_reference,
            destination=root / "recovered-minutes-v1",
        ),
        minutes_root,
    )


def _fake_merge(
    base_model_path: str,
    adapter_path: Path,
    merged_path: Path,
) -> None:
    config = json.loads((adapter_path / "adapter_config.json").read_text())
    assert "corda_config" not in config
    assert "trainable_token_indices" not in config
    base = Path(base_model_path)
    _write_model(
        merged_path,
        weight=b"recovered-minutes-weights",
        tokenizer_marker="merge-executor-base-tokenizer",
    )
    # Model config is produced from the supplied base, while tokenizer files
    # will subsequently be replaced byte-for-byte from the adapter checkpoint.
    (merged_path / "config.json").write_bytes((base / "config.json").read_bytes())


def test_audit_identifies_exact_decision_payload_overwrite(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)

    audit = audit_overwritten_model(
        request.historical_minutes_model,
        request.overwrite_reference_model,
    )

    assert audit["status"] == "payload_overwritten_by_reference_artifact"
    assert audit["weight_payload_exact_match"] is True
    assert audit["model_load_payload_exact_match"] is True
    assert audit["usable_for_evaluation"] is False
    assert (
        fingerprint_artifact_path(request.historical_minutes_model)["sha256"]
        != fingerprint_artifact_path(request.overwrite_reference_model)["sha256"]
    )


def test_recovery_defaults_to_read_only_dry_run(tmp_path: Path) -> None:
    request, _ = _request(tmp_path)
    called = False

    def unexpected_merge(*_args) -> None:
        nonlocal called
        called = True

    manifest, manifest_path = recover_minutes_checkpoint(
        request,
        execute=False,
        merge_executor=unexpected_merge,
        generated_at_utc="2026-07-29T00:00:00Z",
    )

    assert called is False
    assert manifest_path is None
    assert manifest["status"] == "dry_run"
    assert manifest["training_performed"] is False
    assert not request.destination.exists()
    records = {row["artifact_id"]: row for row in manifest["artifacts"]}
    assert set(records) == {
        EVAL_BASE_ID,
        EVAL_ANALYSIS_SFT_ID,
        EVAL_LEGACY_GRPO_ID,
        EVAL_MINUTES_SFT_ID,
        CORRUPTED_MINUTES_ID,
    }
    required_fields = {
        "artifact_id",
        "design_checkpoint_id",
        "intended_parent_id",
        "verified_parent_artifact_id",
        "lineage_status",
        "model_path",
        "tokenizer_path",
        "model_sha256",
        "tokenizer_sha256",
        "adapter_path",
        "adapter_sha256",
        "usable_for_evaluation",
    }
    assert all(required_fields <= set(record) for record in records.values())
    assert records[EVAL_MINUTES_SFT_ID]["usable_for_evaluation"] is False
    assert records[EVAL_MINUTES_SFT_ID]["model_sha256"] is None
    assert records[CORRUPTED_MINUTES_ID]["usable_for_evaluation"] is False

    parsed = build_parser().parse_args(
        [
            "recover-minutes",
            "--foundation-model",
            str(request.foundation_model),
            "--analysis-sft-model",
            str(request.analysis_sft_model),
            "--legacy-grpo-model",
            str(request.legacy_grpo_model),
            "--legacy-grpo-adapter",
            str(request.legacy_grpo_adapter),
            "--minutes-adapter",
            str(request.minutes_adapter),
            "--historical-minutes-model",
            str(request.historical_minutes_model),
            "--overwrite-reference-model",
            str(request.overwrite_reference_model),
            "--destination",
            str(request.destination),
        ]
    )
    assert parsed.execute is False


def test_execute_recovers_to_new_version_and_validates_sources(
    tmp_path: Path,
) -> None:
    request, metadata_source = _request(tmp_path)
    source_config_hash = sha256_file(request.minutes_adapter / "adapter_config.json")
    source_tokenizer = fingerprint_tokenizer_payload(request.minutes_adapter)

    manifest, manifest_path = recover_minutes_checkpoint(
        request,
        execute=True,
        merge_executor=_fake_merge,
        generated_at_utc="2026-07-29T00:00:00Z",
    )

    assert manifest_path == request.destination / "checkpoint_manifest.json"
    assert manifest_path.is_file()
    validate_manifest_integrity(manifest)
    assert manifest["status"] == "complete"
    assert manifest["training_performed"] is False
    assert (
        sha256_file(request.minutes_adapter / "adapter_config.json")
        == source_config_hash
    )

    model = request.destination / "model"
    assert (model / "model.safetensors").read_bytes() == (b"recovered-minutes-weights")
    assert (model / "tokenizer.json").read_bytes() == (
        request.minutes_adapter / "tokenizer.json"
    ).read_bytes()
    assert (model / "trainer_state.json").read_bytes() == (
        metadata_source / "trainer_state.json"
    ).read_bytes()
    assert fingerprint_tokenizer_payload(model)["sha256"] == source_tokenizer["sha256"]

    records = {row["artifact_id"]: row for row in manifest["artifacts"]}
    minutes = records[EVAL_MINUTES_SFT_ID]
    assert minutes["intended_parent_id"] == "chk-2"
    assert minutes["verified_parent_artifact_id"] == EVAL_ANALYSIS_SFT_ID
    assert minutes["lineage_status"] == "verified_parent_differs_from_intended_parent"
    assert minutes["usable_for_evaluation"] is True
    assert minutes["model_sha256"] == fingerprint_artifact_path(model)["sha256"]
    assert minutes["tokenizer_sha256"] == fingerprint_artifact_path(model)["sha256"]
    assert records[EVAL_LEGACY_GRPO_ID]["verified_parent_artifact_id"] == EVAL_BASE_ID
    assert records[CORRUPTED_MINUTES_ID]["usable_for_evaluation"] is False
    assert manifest["recovery"]["output_validation"]["status"] == "passed"
    assert not list(tmp_path.glob(".recovered-minutes-v1.recovery-*"))


def test_runtime_verification_binds_model_to_pinned_manifest(
    tmp_path: Path,
) -> None:
    request, _ = _request(tmp_path)
    manifest, manifest_path = recover_minutes_checkpoint(
        request,
        execute=True,
        merge_executor=_fake_merge,
        generated_at_utc="2026-07-29T00:00:00Z",
    )
    assert manifest_path is not None
    model = request.destination / "model"
    artifact = next(
        row
        for row in manifest["artifacts"]
        if row["artifact_id"] == EVAL_MINUTES_SFT_ID
    )

    verification = verify_checkpoint_artifact(
        checkpoint_manifest=manifest_path,
        artifact_id=EVAL_MINUTES_SFT_ID,
        model_path=model,
        tokenizer_path=model,
        expected_manifest_sha256=sha256_file(manifest_path),
        expected_manifest_payload_sha256=manifest["integrity"]["payload_sha256"],
        expected_model_sha256=artifact["model_sha256"],
        expected_tokenizer_sha256=artifact["tokenizer_sha256"],
    )

    assert verification["status"] == "passed"
    assert verification["artifact_id"] == EVAL_MINUTES_SFT_ID
    assert verification["model"]["sha256"] == artifact["model_sha256"]
    assert verification["usable_for_evaluation"] is True

    (model / "unexpected-runtime-mutation.txt").write_text(
        "changed after provenance sealing\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Runtime model content differs"):
        verify_checkpoint_artifact(
            checkpoint_manifest=manifest_path,
            artifact_id=EVAL_MINUTES_SFT_ID,
            model_path=model,
            tokenizer_path=model,
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_manifest_payload_sha256=(
                manifest["integrity"]["payload_sha256"]
            ),
            expected_model_sha256=artifact["model_sha256"],
            expected_tokenizer_sha256=artifact["tokenizer_sha256"],
        )


def test_runtime_verification_rejects_corrupted_or_unpinned_artifact(
    tmp_path: Path,
) -> None:
    request, _ = _request(tmp_path)
    manifest, manifest_path = recover_minutes_checkpoint(
        request,
        execute=True,
        merge_executor=_fake_merge,
        generated_at_utc="2026-07-29T00:00:00Z",
    )
    assert manifest_path is not None
    records = {row["artifact_id"]: row for row in manifest["artifacts"]}
    corrupted = records[CORRUPTED_MINUTES_ID]

    with pytest.raises(ValueError, match="not approved for evaluation"):
        verify_checkpoint_artifact(
            checkpoint_manifest=manifest_path,
            artifact_id=CORRUPTED_MINUTES_ID,
            model_path=request.historical_minutes_model,
            tokenizer_path=request.historical_minutes_model,
            expected_manifest_sha256=sha256_file(manifest_path),
            expected_manifest_payload_sha256=(
                manifest["integrity"]["payload_sha256"]
            ),
            expected_model_sha256=corrupted["model_sha256"],
            expected_tokenizer_sha256=corrupted["tokenizer_sha256"],
        )

    minutes = records[EVAL_MINUTES_SFT_ID]
    with pytest.raises(ValueError, match="manifest file digest mismatch"):
        verify_checkpoint_artifact(
            checkpoint_manifest=manifest_path,
            artifact_id=EVAL_MINUTES_SFT_ID,
            model_path=request.destination / "model",
            tokenizer_path=request.destination / "model",
            expected_manifest_sha256="0" * 64,
            expected_manifest_payload_sha256=(
                manifest["integrity"]["payload_sha256"]
            ),
            expected_model_sha256=minutes["model_sha256"],
            expected_tokenizer_sha256=minutes["tokenizer_sha256"],
        )

    parsed = build_parser().parse_args(
        [
            "verify-artifact",
            "--checkpoint-manifest",
            str(manifest_path),
            "--artifact-id",
            EVAL_MINUTES_SFT_ID,
            "--model",
            str(request.destination / "model"),
            "--tokenizer",
            str(request.destination / "model"),
            "--expected-manifest-sha256",
            sha256_file(manifest_path),
            "--expected-manifest-payload-sha256",
            manifest["integrity"]["payload_sha256"],
            "--expected-model-sha256",
            minutes["model_sha256"],
            "--expected-tokenizer-sha256",
            minutes["tokenizer_sha256"],
        ]
    )
    assert parsed.command == "verify-artifact"


def test_recovery_rejects_existing_destination_and_parent_mismatch(
    tmp_path: Path,
) -> None:
    request, _ = _request(tmp_path)
    request.destination.mkdir()
    with pytest.raises(FileExistsError, match="brand-new"):
        recover_minutes_checkpoint(request, execute=False)

    request.destination.rmdir()
    config_path = request.minutes_adapter / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = str(request.foundation_model.resolve())
    _write_json(config_path, config)
    with pytest.raises(ValueError, match="Adapter parent mismatch"):
        recover_minutes_checkpoint(request, execute=False)


def test_recovery_rejects_output_that_still_matches_overwrite_reference(
    tmp_path: Path,
) -> None:
    request, _ = _request(tmp_path)

    def bad_merge(
        _base_model_path: str,
        _adapter_path: Path,
        merged_path: Path,
    ) -> None:
        _write_model(
            merged_path,
            weight=b"decision-weights",
            tokenizer_marker="temporary-tokenizer",
        )

    with pytest.raises(ValueError, match="still equal"):
        recover_minutes_checkpoint(
            request,
            execute=True,
            merge_executor=bad_merge,
        )
    assert not request.destination.exists()
    assert not list(tmp_path.glob(".recovered-minutes-v1.recovery-*"))


def test_invalidation_is_dry_run_by_default_and_never_mutates_run(
    tmp_path: Path,
) -> None:
    run = tmp_path / "runs" / "invalid-run"
    run.mkdir(parents=True)
    (run / "partial.jsonl").write_text('{"generated":"bad"}\n', encoding="utf-8")
    invalid_model = tmp_path / "invalid-model"
    reference_model = tmp_path / "reference-model"
    _write_model(invalid_model, weight=b"same-overwritten-weights")
    _write_model(reference_model, weight=b"same-overwritten-weights")
    before = fingerprint_artifact_path(run)

    record, output = invalidate_run(
        run_path=run,
        invalid_model=invalid_model,
        matched_reference_model=reference_model,
        reason="Run used the overwritten Minutes model payload.",
        execute=False,
        invalidated_at_utc="2026-07-29T01:02:03Z",
    )
    assert record["status"] == "invalidated"
    assert output == run.parent / "_invalidations" / "invalid-run.json"
    assert not output.exists()
    assert fingerprint_artifact_path(run)["sha256"] == before["sha256"]

    written, same_output = invalidate_run(
        run_path=run,
        invalid_model=invalid_model,
        matched_reference_model=reference_model,
        reason="Run used the overwritten Minutes model payload.",
        execute=True,
        invalidated_at_utc="2026-07-29T01:02:03Z",
    )
    assert same_output == output
    assert output.is_file()
    validate_manifest_integrity(written)
    assert fingerprint_artifact_path(run)["sha256"] == before["sha256"]
    assert written["source_run_mutated"] is False
    assert written["downstream_use"]["scoring"] == "prohibited"

    # An exact repeat is idempotent, while a contradictory record cannot
    # replace the first immutable audit decision.
    invalidate_run(
        run_path=run,
        invalid_model=invalid_model,
        matched_reference_model=reference_model,
        reason="Run used the overwritten Minutes model payload.",
        execute=True,
        invalidated_at_utc="2026-07-29T01:02:03Z",
    )
    with pytest.raises(ValueError, match="incompatible"):
        invalidate_run(
            run_path=run,
            invalid_model=invalid_model,
            matched_reference_model=reference_model,
            reason="A contradictory reason.",
            execute=True,
            invalidated_at_utc="2026-07-29T01:02:03Z",
        )


def test_invalidation_rejects_unmatched_reference_and_in_run_sidecar(
    tmp_path: Path,
) -> None:
    run = tmp_path / "invalid-run"
    run.mkdir()
    (run / "partial.jsonl").write_text("{}\n", encoding="utf-8")
    invalid_model = tmp_path / "invalid-model"
    reference_model = tmp_path / "reference-model"
    _write_model(invalid_model, weight=b"bad")
    _write_model(reference_model, weight=b"different")

    with pytest.raises(ValueError, match="do not match"):
        invalidate_run(
            run_path=run,
            invalid_model=invalid_model,
            matched_reference_model=reference_model,
            reason="Mismatch",
        )

    (reference_model / "model.safetensors").write_bytes(b"bad")
    with pytest.raises(ValueError, match="outside"):
        invalidate_run(
            run_path=run,
            invalid_model=invalid_model,
            matched_reference_model=reference_model,
            reason="Mismatch",
            output=run / "invalidation.json",
        )
