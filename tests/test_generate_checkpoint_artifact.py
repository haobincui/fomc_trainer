from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from jobs.generation.generate_checkpoint_artifact import (
    LEGACY_MANIFEST_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    PROGRESS_ROW_SCHEMA_VERSION,
    PROGRESS_SCHEMA_VERSION,
    _capture_inference_environment,
    _durable_mkdir,
    _exclusive_progress_lock,
    finalise_generation_rows,
    generate_artifact,
    reseal_legacy_generation_manifest_with_progress,
    select_artifact,
    validate_generation_artifact,
    validate_generation_progress_binding,
)
from open_r1.provenance import fingerprint_artifact_path, sha256_text
from open_r1.validator.loo_generation_spec import (
    derive_row_seed,
    seal_manifest,
    validate_manifest_integrity,
)


ARTIFACT = {
    "artifact_id": "eval-base",
    "design_checkpoint_id": "chk-0",
    "intended_parent_id": None,
    "verified_parent_artifact_id": None,
    "lineage_status": "verified",
}
FINGERPRINT = {"sha256": "a" * 64}


def _prompt(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "prompt": "prompt",
        "prompt_sha256": "b" * 64,
    }


def test_preserves_missing_rows_as_invalid() -> None:
    rows = finalise_generation_rows(
        [_prompt("a"), _prompt("b")],
        [
            {
                **_prompt("a"),
                "generated": "answer",
                "generation_finish_reason": "stop",
                "input_was_truncated": False,
            }
        ],
        artifact=ARTIFACT,
        model_fingerprint=FINGERPRINT,
        tokenizer_fingerprint=FINGERPRINT,
    )
    assert len(rows) == 2
    assert rows[0]["valid_generation"] is True
    assert rows[0]["final_answer"] == "answer"
    assert rows[1]["valid_generation"] is False
    assert "empty_or_missing_generation" in rows[1]["invalid_reasons"]


def test_missing_row_retains_deterministic_sample_seed_metadata() -> None:
    rows = finalise_generation_rows(
        [_prompt("sample-a")],
        [],
        artifact=ARTIFACT,
        model_fingerprint=FINGERPRINT,
        tokenizer_fingerprint=FINGERPRINT,
        shared_provenance={
            "generation_base_seed": 20260729,
            "generation_seed_policy": "sample-id-sha256-v1",
        },
    )

    assert rows[0]["generation_seed_policy"] == "sample-id-sha256-v1"
    assert rows[0]["generation_seed"] == derive_row_seed(20260729, "sample-a")


def test_inference_environment_records_software_and_hardware_shape() -> None:
    environment = _capture_inference_environment()

    assert environment["schema_version"] == "checkpoint-inference-environment-v1"
    assert environment["python"]["version"]
    assert set(environment["packages"]) >= {"torch", "transformers", "vllm"}
    assert isinstance(environment["cuda"]["devices"], list)


def test_token_limit_completion_is_invalid_but_retained() -> None:
    rows = finalise_generation_rows(
        [_prompt("a")],
        [
            {
                **_prompt("a"),
                "generated": "reasoning</think>answer",
                "generation_finish_reason": "length",
                "input_was_truncated": False,
            }
        ],
        artifact=ARTIFACT,
        model_fingerprint=FINGERPRINT,
        tokenizer_fingerprint=FINGERPRINT,
    )
    assert rows[0]["final_answer"] == "answer"
    assert rows[0]["valid_generation"] is False
    assert "non_normal_finish:length" in rows[0]["invalid_reasons"]


def test_attaches_common_test_and_decoding_provenance_to_every_row() -> None:
    provenance = {
        "test_set_sha256": "c" * 64,
        "prompt_template_sha256": "d" * 64,
        "decoding_config_sha256": "e" * 64,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    rows = finalise_generation_rows(
        [_prompt("a"), _prompt("b")],
        [],
        artifact=ARTIFACT,
        model_fingerprint=FINGERPRINT,
        tokenizer_fingerprint=FINGERPRINT,
        shared_provenance=provenance,
    )

    assert len(rows) == 2
    assert all(
        all(row[key] == value for key, value in provenance.items())
        for row in rows
    )


def test_rejects_duplicate_generation_rows() -> None:
    generated = [
        {
            **_prompt("a"),
            "generated": "answer",
            "generation_finish_reason": "stop",
            "input_was_truncated": False,
        }
    ]
    with pytest.raises(ValueError, match="duplicate"):
        finalise_generation_rows(
            [_prompt("a")],
            generated + generated,
            artifact=ARTIFACT,
            model_fingerprint=FINGERPRINT,
            tokenizer_fingerprint=FINGERPRINT,
        )


def test_select_artifact_rejects_unusable_checkpoint() -> None:
    with pytest.raises(ValueError, match="not usable"):
        select_artifact(
            {
                "artifacts": [
                    {
                        "artifact_id": "broken",
                        "usable_for_evaluation": False,
                    }
                ]
            },
            "broken",
        )


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _generation_fixture(tmp_path: Path, *, sample_count: int = 3) -> dict:
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"weights")
    fingerprint = fingerprint_artifact_path(model)
    checkpoint = tmp_path / "checkpoints.json"
    _write_json(
        checkpoint,
        seal_manifest(
            {
                "schema_version": "test-checkpoints-v1",
                "artifacts": [
                    {
                        "artifact_id": "eval-base",
                        "design_checkpoint_id": "chk-0",
                        "intended_parent_id": None,
                        "verified_parent_artifact_id": None,
                        "lineage_status": "verified",
                        "usable_for_evaluation": True,
                        "model_path": str(model),
                        "model_sha256": fingerprint["sha256"],
                        "tokenizer_path": str(model),
                        "tokenizer_sha256": fingerprint["sha256"],
                    }
                ],
            }
        ),
    )
    prompts = tmp_path / "prompts.jsonl"
    prompt_rows = [
        {
            "sample_id": f"sample-{index}",
            "prompt": f"prompt-{index}",
            "prompt_sha256": sha256_text(f"prompt-{index}"),
            "response": f"target-{index}",
        }
        for index in range(sample_count)
    ]
    prompts.write_text(
        "".join(json.dumps(row) + "\n" for row in prompt_rows),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "generation": {
                "base_seed": 20260729,
                "seed_policy": "sample-id-sha256-v1",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_new_tokens": 128,
                "max_model_len": 1024,
                "system_prompt": "Return an answer.",
            }
        },
    )
    return {
        "checkpoint": checkpoint,
        "prompts": prompts,
        "config": config,
        "output": tmp_path / "output",
        "prompt_rows": prompt_rows,
    }


def _successful_results(prompts: list[str], *_args, **_kwargs) -> list[dict]:
    return [
        {
            "text": f"answer for {prompt}",
            "finish_reason": "stop",
            "stop_reason": None,
            "prompt_token_count": 10,
            "prompt_preflight_token_count": 10,
            "output_token_count": 4,
            "input_was_truncated": False,
        }
        for prompt in prompts
    ]


def test_generation_commits_each_batch_and_resumes_only_missing_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _generation_fixture(tmp_path)
    monkeypatch.setattr(
        "jobs.generation.generate_checkpoint_artifact._capture_inference_environment",
        lambda: {"schema_version": "test-environment-v1"},
    )
    first_call = True

    def interrupt_second_batch(prompts: list[str], *_args, **_kwargs) -> list[dict]:
        nonlocal first_call
        if first_call:
            first_call = False
            return _successful_results(prompts)
        raise RuntimeError("simulated interruption")

    with patch(
        "generate_new_response.generate_responses",
        side_effect=interrupt_second_batch,
    ):
        with pytest.raises(RuntimeError, match="generation batch starting"):
            generate_artifact(
                checkpoint_manifest_file=fixture["checkpoint"],
                prompts_file=fixture["prompts"],
                config_file=fixture["config"],
                artifact_id="eval-base",
                output_dir=fixture["output"],
                batch_size=2,
            )

    partial_dir = fixture["output"] / ".partial" / "eval-base"
    partial = partial_dir / "generations.progress.v1.jsonl"
    state_path = partial_dir / "state.progress.v1.json"
    progress_rows = [json.loads(line) for line in partial.read_text().splitlines()]
    state = json.loads(state_path.read_text())
    validate_manifest_integrity(state)
    assert state["schema_version"] == PROGRESS_SCHEMA_VERSION
    assert state["row_count"] == 2
    assert state["commit_count"] == 1
    assert [row["schema_version"] for row in progress_rows] == [
        PROGRESS_ROW_SCHEMA_VERSION,
        PROGRESS_ROW_SCHEMA_VERSION,
    ]
    assert not (fixture["output"] / "eval-base.jsonl").exists()
    with partial.open("ab") as handle:
        handle.write(b'{"uncommitted_torn_tail":')

    observed_prompts: list[list[str]] = []

    def finish(prompts: list[str], *_args, **_kwargs) -> list[dict]:
        observed_prompts.append(prompts)
        return _successful_results(prompts)

    with patch("generate_new_response.generate_responses", side_effect=finish):
        manifest_path, manifest = generate_artifact(
            checkpoint_manifest_file=fixture["checkpoint"],
            prompts_file=fixture["prompts"],
            config_file=fixture["config"],
            artifact_id="eval-base",
            output_dir=fixture["output"],
            batch_size=2,
        )

    assert observed_prompts == [["prompt-2"]]
    assert manifest_path.is_file()
    assert manifest["sample_count"] == 3
    final_rows = [
        json.loads(line)
        for line in (fixture["output"] / "eval-base.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [row["sample_id"] for row in final_rows] == [
        "sample-0",
        "sample-1",
        "sample-2",
    ]
    assert [row["generation_position"] for row in final_rows] == [0, 1, 2]
    assert all("contract_sha256" not in row for row in final_rows)
    state = json.loads(state_path.read_text())
    validate_manifest_integrity(state)
    assert state["status"] == "generation_complete"
    assert state["row_count"] == 3
    assert state["resume_count"] == 1
    frozen_state_bytes = state_path.read_bytes()
    assert manifest["schema_version"] == "checkpoint-artifact-generation-manifest-v2"
    assert manifest["progress"]["state"]["sha256"] == sha256_text(
        frozen_state_bytes.decode("utf-8")
    )
    assert manifest["progress"]["partial"]["sha256"] == state["partial"][
        "sha256"
    ]
    assert "final_manifest" not in state

    generate_artifact(
        checkpoint_manifest_file=fixture["checkpoint"],
        prompts_file=fixture["prompts"],
        config_file=fixture["config"],
        artifact_id="eval-base",
        output_dir=fixture["output"],
        batch_size=2,
    )
    assert state_path.read_bytes() == frozen_state_bytes


def test_progress_contract_and_committed_bytes_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _generation_fixture(tmp_path)
    monkeypatch.setattr(
        "jobs.generation.generate_checkpoint_artifact._capture_inference_environment",
        lambda: {"schema_version": "test-environment-v1"},
    )
    with patch(
        "generate_new_response.generate_responses",
        side_effect=[_successful_results(["prompt-0", "prompt-1"]), RuntimeError()],
    ):
        with pytest.raises(RuntimeError):
            generate_artifact(
                checkpoint_manifest_file=fixture["checkpoint"],
                prompts_file=fixture["prompts"],
                config_file=fixture["config"],
                artifact_id="eval-base",
                output_dir=fixture["output"],
                batch_size=2,
            )

    with pytest.raises(ValueError, match="progress contract changed"):
        generate_artifact(
            checkpoint_manifest_file=fixture["checkpoint"],
            prompts_file=fixture["prompts"],
            config_file=fixture["config"],
            artifact_id="eval-base",
            output_dir=fixture["output"],
            batch_size=1,
        )

    config = json.loads(fixture["config"].read_text())
    config["generation"]["temperature"] = 0.1
    _write_json(fixture["config"], config)
    with pytest.raises(ValueError, match="progress contract changed"):
        generate_artifact(
            checkpoint_manifest_file=fixture["checkpoint"],
            prompts_file=fixture["prompts"],
            config_file=fixture["config"],
            artifact_id="eval-base",
            output_dir=fixture["output"],
            batch_size=2,
        )

    config["generation"]["temperature"] = 0.0
    _write_json(fixture["config"], config)
    partial = (
        fixture["output"]
        / ".partial"
        / "eval-base"
        / "generations.progress.v1.jsonl"
    )
    content = partial.read_bytes()
    partial.write_bytes(b"X" + content[1:])
    with pytest.raises(ValueError, match="prefix hash changed"):
        generate_artifact(
            checkpoint_manifest_file=fixture["checkpoint"],
            prompts_file=fixture["prompts"],
            config_file=fixture["config"],
            artifact_id="eval-base",
            output_dir=fixture["output"],
            batch_size=2,
        )


def test_progress_lock_rejects_a_second_writer(tmp_path: Path) -> None:
    lock_path = tmp_path / ".partial" / "eval-base" / ".lock"
    with _exclusive_progress_lock(lock_path):
        with pytest.raises(RuntimeError, match="holds the progress lock"):
            with _exclusive_progress_lock(lock_path):
                pass


def test_output_without_manifest_is_recovered_only_from_complete_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _generation_fixture(tmp_path, sample_count=2)
    monkeypatch.setattr(
        "jobs.generation.generate_checkpoint_artifact._capture_inference_environment",
        lambda: {"schema_version": "test-environment-v1"},
    )
    from jobs.generation import generate_checkpoint_artifact as module

    original_atomic_write = module._atomic_write

    def fail_manifest(path: Path, content: str) -> None:
        if path.name == "eval-base.manifest.json":
            raise RuntimeError("simulated manifest commit crash")
        original_atomic_write(path, content)

    monkeypatch.setattr(module, "_atomic_write", fail_manifest)
    with patch(
        "generate_new_response.generate_responses",
        side_effect=_successful_results,
    ):
        with pytest.raises(RuntimeError, match="manifest commit crash"):
            generate_artifact(
                checkpoint_manifest_file=fixture["checkpoint"],
                prompts_file=fixture["prompts"],
                config_file=fixture["config"],
                artifact_id="eval-base",
                output_dir=fixture["output"],
                batch_size=2,
            )
    output_path = fixture["output"] / "eval-base.jsonl"
    manifest_path = fixture["output"] / "eval-base.manifest.json"
    original_output = output_path.read_bytes()
    assert not manifest_path.exists()

    monkeypatch.setattr(module, "_atomic_write", original_atomic_write)
    with patch(
        "generate_new_response.generate_responses",
        side_effect=AssertionError("completed progress must not regenerate"),
    ) as mocked_generation:
        recovered_path, _ = generate_artifact(
            checkpoint_manifest_file=fixture["checkpoint"],
            prompts_file=fixture["prompts"],
            config_file=fixture["config"],
            artifact_id="eval-base",
            output_dir=fixture["output"],
            batch_size=2,
        )
    assert mocked_generation.call_count == 0
    assert recovered_path == manifest_path
    assert output_path.read_bytes() == original_output


def _complete_generation(
    fixture: dict, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict]:
    monkeypatch.setattr(
        "jobs.generation.generate_checkpoint_artifact._capture_inference_environment",
        lambda: {"schema_version": "test-environment-v1"},
    )
    with patch(
        "generate_new_response.generate_responses",
        side_effect=_successful_results,
    ):
        return generate_artifact(
            checkpoint_manifest_file=fixture["checkpoint"],
            prompts_file=fixture["prompts"],
            config_file=fixture["config"],
            artifact_id="eval-base",
            output_dir=fixture["output"],
            batch_size=2,
        )


def test_final_validator_rejects_missing_or_tampered_progress_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _generation_fixture(tmp_path, sample_count=2)
    manifest_path, manifest = _complete_generation(fixture, monkeypatch)
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION

    missing = dict(manifest)
    missing.pop("integrity")
    missing.pop("progress")
    _write_json(manifest_path, seal_manifest(missing))
    with pytest.raises(ValueError, match="frozen progress binding"):
        validate_generation_artifact(
            checkpoint_manifest_file=fixture["checkpoint"],
            prompts_file=fixture["prompts"],
            config_file=fixture["config"],
            artifact_id="eval-base",
            output_dir=fixture["output"],
        )

    _write_json(manifest_path, manifest)
    state_path = (
        fixture["output"]
        / ".partial"
        / "eval-base"
        / "state.progress.v1.json"
    )
    state = json.loads(state_path.read_text())
    state.pop("integrity")
    state["commit_count"] += 1
    _write_json(state_path, seal_manifest(state))
    with pytest.raises(ValueError, match="state binding changed"):
        validate_generation_artifact(
            checkpoint_manifest_file=fixture["checkpoint"],
            prompts_file=fixture["prompts"],
            config_file=fixture["config"],
            artifact_id="eval-base",
            output_dir=fixture["output"],
        )


def test_legacy_manifest_can_be_resealed_to_versioned_v2_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _generation_fixture(tmp_path, sample_count=2)
    manifest_path, manifest = _complete_generation(fixture, monkeypatch)
    legacy = dict(manifest)
    legacy.pop("integrity")
    legacy.pop("progress")
    legacy["schema_version"] = LEGACY_MANIFEST_SCHEMA_VERSION
    _write_json(manifest_path, seal_manifest(legacy))
    legacy_bytes = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="Legacy generation manifest"):
        validate_generation_artifact(
            checkpoint_manifest_file=fixture["checkpoint"],
            prompts_file=fixture["prompts"],
            config_file=fixture["config"],
            artifact_id="eval-base",
            output_dir=fixture["output"],
        )
    resealed_path, resealed = reseal_legacy_generation_manifest_with_progress(
        checkpoint_manifest_file=fixture["checkpoint"],
        prompts_file=fixture["prompts"],
        config_file=fixture["config"],
        artifact_id="eval-base",
        output_dir=fixture["output"],
    )
    assert manifest_path.read_bytes() == legacy_bytes
    assert resealed_path.name == "eval-base.manifest.progress-v2.json"
    assert resealed["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert resealed["progress"]["state"]["path"].endswith(
        "state.progress.frozen.v1.json"
    )
    validate_generation_artifact(
        checkpoint_manifest_file=fixture["checkpoint"],
        prompts_file=fixture["prompts"],
        config_file=fixture["config"],
        artifact_id="eval-base",
        output_dir=fixture["output"],
        manifest_file=resealed_path,
    )


def _convert_primary_manifest_to_legacy(manifest_path: Path) -> bytes:
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("integrity")
    manifest.pop("progress")
    manifest["schema_version"] = LEGACY_MANIFEST_SCHEMA_VERSION
    _write_json(manifest_path, seal_manifest(manifest))
    return manifest_path.read_bytes()


def test_legacy_reseal_recovers_frozen_state_without_manifest_and_reuses_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _generation_fixture(tmp_path, sample_count=2)
    primary, _ = _complete_generation(fixture, monkeypatch)
    legacy_bytes = _convert_primary_manifest_to_legacy(primary)
    from jobs.generation import generate_checkpoint_artifact as module

    original_atomic_write = module._atomic_write

    def crash_before_manifest(path: Path, content: str) -> None:
        if path.name == "eval-base.manifest.progress-v2.json":
            raise RuntimeError("simulated reseal manifest crash")
        original_atomic_write(path, content)

    monkeypatch.setattr(module, "_atomic_write", crash_before_manifest)
    with pytest.raises(RuntimeError, match="reseal manifest crash"):
        reseal_legacy_generation_manifest_with_progress(
            checkpoint_manifest_file=fixture["checkpoint"],
            prompts_file=fixture["prompts"],
            config_file=fixture["config"],
            artifact_id="eval-base",
            output_dir=fixture["output"],
        )
    frozen = (
        fixture["output"]
        / ".partial"
        / "eval-base"
        / "state.progress.frozen.v1.json"
    )
    frozen_bytes = frozen.read_bytes()
    resealed = fixture["output"] / "eval-base.manifest.progress-v2.json"
    assert not resealed.exists()

    monkeypatch.setattr(module, "_atomic_write", original_atomic_write)
    path, manifest = reseal_legacy_generation_manifest_with_progress(
        checkpoint_manifest_file=fixture["checkpoint"],
        prompts_file=fixture["prompts"],
        config_file=fixture["config"],
        artifact_id="eval-base",
        output_dir=fixture["output"],
    )
    manifest_bytes = path.read_bytes()
    assert frozen.read_bytes() == frozen_bytes
    assert primary.read_bytes() == legacy_bytes
    reused_path, reused = reseal_legacy_generation_manifest_with_progress(
        checkpoint_manifest_file=fixture["checkpoint"],
        prompts_file=fixture["prompts"],
        config_file=fixture["config"],
        artifact_id="eval-base",
        output_dir=fixture["output"],
    )
    assert reused_path == path
    assert reused == manifest
    assert path.read_bytes() == manifest_bytes
    assert frozen.read_bytes() == frozen_bytes


def test_legacy_reseal_manifest_without_frozen_state_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _generation_fixture(tmp_path, sample_count=2)
    primary, _ = _complete_generation(fixture, monkeypatch)
    _convert_primary_manifest_to_legacy(primary)
    reseal_legacy_generation_manifest_with_progress(
        checkpoint_manifest_file=fixture["checkpoint"],
        prompts_file=fixture["prompts"],
        config_file=fixture["config"],
        artifact_id="eval-base",
        output_dir=fixture["output"],
    )
    frozen = (
        fixture["output"]
        / ".partial"
        / "eval-base"
        / "state.progress.frozen.v1.json"
    )
    frozen.unlink()
    with pytest.raises(ValueError, match="without its frozen state"):
        reseal_legacy_generation_manifest_with_progress(
            checkpoint_manifest_file=fixture["checkpoint"],
            prompts_file=fixture["prompts"],
            config_file=fixture["config"],
            artifact_id="eval-base",
            output_dir=fixture["output"],
        )


def test_artifact_slug_and_custom_reseal_paths_are_confined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _generation_fixture(tmp_path, sample_count=2)
    with pytest.raises(ValueError, match="safe slug"):
        generate_artifact(
            checkpoint_manifest_file=fixture["checkpoint"],
            prompts_file=fixture["prompts"],
            config_file=fixture["config"],
            artifact_id="../escape",
            output_dir=fixture["output"],
        )
    primary, _ = _complete_generation(fixture, monkeypatch)
    _convert_primary_manifest_to_legacy(primary)
    with pytest.raises(ValueError, match="directly inside output_dir"):
        reseal_legacy_generation_manifest_with_progress(
            checkpoint_manifest_file=fixture["checkpoint"],
            prompts_file=fixture["prompts"],
            config_file=fixture["config"],
            artifact_id="eval-base",
            output_dir=fixture["output"],
            resealed_manifest_file=tmp_path / "outside.json",
        )
    with pytest.raises(ValueError, match="artifact progress directory"):
        reseal_legacy_generation_manifest_with_progress(
            checkpoint_manifest_file=fixture["checkpoint"],
            prompts_file=fixture["prompts"],
            config_file=fixture["config"],
            artifact_id="eval-base",
            output_dir=fixture["output"],
            frozen_state_file=fixture["output"] / "outside-state.json",
        )


def test_progress_validator_reads_state_partial_and_output_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _generation_fixture(tmp_path, sample_count=2)
    manifest_path, _ = _complete_generation(fixture, monkeypatch)
    output = fixture["output"] / "eval-base.jsonl"
    state = (
        fixture["output"]
        / ".partial"
        / "eval-base"
        / "state.progress.v1.json"
    )
    partial = state.with_name("generations.progress.v1.jsonl")
    targets = {path.resolve(): 0 for path in (output, state, partial)}
    original_read_bytes = Path.read_bytes

    def counted_read_bytes(path: Path) -> bytes:
        resolved = path.resolve()
        if resolved in targets:
            targets[resolved] += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    validate_generation_progress_binding(
        generation_manifest_file=manifest_path,
        generation_output_file=output,
        checkpoint_manifest_file=fixture["checkpoint"],
        prompts_file=fixture["prompts"],
        config_file=fixture["config"],
    )
    assert targets == {path.resolve(): 1 for path in (output, state, partial)}


def test_durable_mkdir_fsyncs_each_new_directory_and_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(
        "jobs.generation.generate_checkpoint_artifact._fsync_directory",
        lambda path: calls.append(path),
    )
    target = tmp_path / "one" / "two" / "three"
    _durable_mkdir(target)
    assert target.is_dir()
    for directory in (tmp_path / "one", tmp_path / "one" / "two", target):
        assert directory in calls
        assert directory.parent in calls
