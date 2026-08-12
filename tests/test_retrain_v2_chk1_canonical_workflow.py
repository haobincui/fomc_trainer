from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from jobs.retrain_v2.chk1 import canonical_workflow as workflow
from jobs.retrain_v2.chk1.contracts import MANIFEST_SCHEMA_VERSION, canonical_json, sha256_text


DIGEST = "a" * 64
TOPICS = ("GDP Growth", "Unemployment Rate", "Treasury Yields", "Bank Capital")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _bound(
    path: Path, *, source_root: Path, source_relative: str, release_prefix: str, rows: int | None = None
) -> workflow.BoundFile:
    return workflow.BoundFile(
        source_path=path,
        source_relative=source_relative,
        release_path=f"{release_prefix}/{source_relative}",
        sha256=workflow.sha256_file(path),
        bytes=path.stat().st_size,
        rows=rows,
    )


def _verified_generation(root: Path, *, count: int = 210) -> workflow.VerifiedGeneration:
    generation_root = root / "generation"
    preparation_root = root / "preparation"
    generation_root.mkdir(parents=True)
    preparation_root.mkdir(parents=True)
    split_sizes = {"train": 150, "eval": 30, "test": 30}
    split_sequence = [split for split in workflow.SPLITS for _ in range(split_sizes[split])]
    assert len(split_sequence) == count
    sft_rows: dict[str, list[dict]] = {split: [] for split in workflow.SPLITS}
    manifest_rows: dict[str, list[dict]] = {split: [] for split in workflow.SPLITS}
    population: list[dict] = []
    start = date(2010, 1, 1)
    for index, split in enumerate(split_sequence):
        meeting = (start + timedelta(days=index * 31)).isoformat()
        topic = TOPICS[index % len(TOPICS)]
        sample_id = f"sample-{index:04d}"
        prompt = f"Safe point-in-time prompt {sample_id} for {topic}."
        reasoning = f"Reasoning from evidence E-{index:04d}."
        final = f"Final grounded analysis for {sample_id}."
        response = f"{reasoning}\n</think>\n{final}"
        provided_data = canonical_json(
            {
                "schema_version": "safe-v1",
                "atomic_topic": topic,
                "evidence": [
                    {
                        "evidence_id": f"E-{index:04d}",
                        "source_sha256": DIGEST,
                        "value": str(index),
                    }
                ],
            }
        )
        sft = {"prompt": prompt, "response": response, "provided_data": provided_data}
        prepared = {
            "sample_id": sample_id,
            "split": split,
            "meeting_date": meeting,
            "atomic_topic": topic,
            "row_sha256": DIGEST,
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "sample_id": sample_id,
            "meeting_date": meeting,
            "atomic_topic": topic,
            "section_style_id": "participants_views",
            "split": split,
            "cutoff_ts": f"{meeting}T23:59:59Z",
            "evidence_lineage": [
                {"evidence_id": f"E-{index:04d}", "source_sha256": DIGEST}
            ],
            "style_guide_sha256": DIGEST,
            "teacher_model_sha256": DIGEST,
            "tokenizer_sha256": DIGEST,
            "generation": {
                "cache_key": DIGEST,
                "generation_provenance_sha256": DIGEST,
                "prompt_template_sha256": DIGEST,
                "generator_tokenizer_sha256": DIGEST,
                "student_tokenizer_sha256": DIGEST,
                "prepared_row_sha256": sha256_text(canonical_json(prepared)),
                "preparation_binding_sha256": DIGEST,
                "model_input_projection": {"attestation_sha256": DIGEST},
                "sft_token_budget": {
                    "schema_version": "chk1-sft-token-budget-v1",
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                    "max_prompt_tokens": 3072,
                    "max_completion_tokens": 1024,
                    "max_total_tokens": 4096,
                    "overflow_policy": "error",
                    "truncated": False,
                    "passed": True,
                },
            },
            "prompt_sha256": sha256_text(prompt),
            "reasoning_sha256": sha256_text(reasoning),
            "final_analysis_sha256": sha256_text(final),
            "response_sha256": sha256_text(response),
            "provided_data_sha256": sha256_text(provided_data),
            "input_truncated": False,
        }
        sft_rows[split].append(sft)
        manifest_rows[split].append(manifest)
        population.append(prepared)

    selected_ids = [row["sample_id"] for row in population]
    generation_handoff = {
        "payload_sha256": DIGEST,
        "generation_provenance": {
            "payload_sha256": DIGEST,
            "generation_code": {"payload_sha256": DIGEST},
            "prompt_template_sha256": DIGEST,
            "teacher_model": {"sha256": DIGEST},
            "student_tokenizer": {"sha256": DIGEST},
        },
    }
    generation_handoff_path = generation_root / "generation_handoff.json"
    selection_path = generation_root / "selection_manifest.json"
    cache_manifest_path = generation_root / "cache/cache_manifest.json"
    sample_cache_path = generation_root / "cache/samples/sample.json"
    _write_json(generation_handoff_path, generation_handoff)
    _write_json(
        selection_path,
        {
            "selected_sample_ids": selected_ids,
            "selected_sample_ids_sha256": sha256_text(canonical_json(selected_ids)),
        },
    )
    _write_json(cache_manifest_path, {"entries": [{"sample_id": item} for item in selected_ids]})
    _write_json(sample_cache_path, {"status": "bound"})

    inventory_path = preparation_root / "inventory.json"
    prepared_path = preparation_root / "prepared/train.jsonl"
    _write_json(inventory_path, {"schema_version": "test-inventory-v1", "files": []})
    _write_jsonl(prepared_path, population)
    inventory_record = {
        "path": "inventory.json",
        "sha256": workflow.sha256_file(inventory_path),
        "bytes": inventory_path.stat().st_size,
    }
    prepare_handoff = {
        "payload_sha256": DIGEST,
        "style_guide_sha256": DIGEST,
        "artifacts": {"inventory": inventory_record},
    }
    prepare_handoff_path = preparation_root / "prepare_handoff.json"
    _write_json(prepare_handoff_path, prepare_handoff)

    generation_files = (
        _bound(
            generation_handoff_path,
            source_root=generation_root,
            source_relative="generation_handoff.json",
            release_prefix="source/generation",
        ),
        _bound(
            selection_path,
            source_root=generation_root,
            source_relative="selection_manifest.json",
            release_prefix="source/generation",
        ),
        _bound(
            cache_manifest_path,
            source_root=generation_root,
            source_relative="cache/cache_manifest.json",
            release_prefix="source/generation",
        ),
        _bound(
            sample_cache_path,
            source_root=generation_root,
            source_relative="cache/samples/sample.json",
            release_prefix="source/generation",
            rows=1,
        ),
    )
    preparation_files = (
        _bound(
            prepare_handoff_path,
            source_root=preparation_root,
            source_relative="prepare_handoff.json",
            release_prefix="source/preparation",
        ),
        _bound(
            inventory_path,
            source_root=preparation_root,
            source_relative="inventory.json",
            release_prefix="source/preparation",
        ),
        _bound(
            prepared_path,
            source_root=preparation_root,
            source_relative="prepared/train.jsonl",
            release_prefix="source/preparation",
            rows=len(population),
        ),
    )
    return workflow.VerifiedGeneration(
        repo_root=root,
        generation_root=generation_root,
        generation_handoff_path=generation_handoff_path,
        generation_handoff=generation_handoff,
        generation_handoff_sha256=workflow.sha256_file(generation_handoff_path),
        preparation_root=preparation_root,
        prepare_handoff_path=prepare_handoff_path,
        prepare_handoff=prepare_handoff,
        prepare_handoff_sha256=workflow.sha256_file(prepare_handoff_path),
        preparation_binding={"binding_sha256": DIGEST},
        selection={
            "selected_sample_ids": selected_ids,
            "selected_sample_ids_sha256": sha256_text(canonical_json(selected_ids)),
        },
        cache_manifest={"entries": []},
        population=tuple(population),
        sft_rows={split: tuple(rows) for split, rows in sft_rows.items()},
        manifest_rows={split: tuple(rows) for split, rows in manifest_rows.items()},
        exclusions=(),
        generation_files=generation_files,
        preparation_files=preparation_files,
    )


def test_release_admission_excludes_empty_final_without_mutating_generation(
    tmp_path: Path,
) -> None:
    verified = _verified_generation(tmp_path)
    sft_rows = {
        split: tuple(dict(row) for row in verified.sft_rows[split])
        for split in workflow.SPLITS
    }
    manifest_rows = {
        split: tuple(json.loads(canonical_json(row)) for row in verified.manifest_rows[split])
        for split in workflow.SPLITS
    }
    reasoning_only = "Reasoning remains valid.\n</think>\n"
    sft_rows["train"][0]["response"] = reasoning_only
    manifest_rows["train"][0]["response_sha256"] = sha256_text(reasoning_only)
    manifest_rows["train"][0]["final_analysis_sha256"] = sha256_text("")
    changed = replace(
        verified,
        sft_rows=sft_rows,
        manifest_rows=manifest_rows,
    )

    admitted_sft, admitted_manifests, exclusions = workflow._release_terminals(changed)

    assert len(admitted_sft["train"]) == len(verified.sft_rows["train"]) - 1
    assert len(admitted_manifests["train"]) == len(verified.manifest_rows["train"]) - 1
    assert sum(len(rows) for rows in admitted_sft.values()) + len(exclusions) == len(
        verified.population
    )
    assert exclusions == (
        {
            "schema_version": workflow.generation.EXCLUSION_SCHEMA_VERSION,
            "sample_id": verified.manifest_rows["train"][0]["sample_id"],
            "split": "train",
            "meeting_date": verified.manifest_rows["train"][0]["meeting_date"],
            "atomic_topic": verified.manifest_rows["train"][0]["atomic_topic"],
            "stage": "canonical_admission",
            "reason_code": "empty_response_component",
            "error_type": "EmptyResponseComponent",
            "error_codes": ["empty_final_analysis"],
            "prepared_row_sha256": verified.manifest_rows["train"][0]["generation"][
                "prepared_row_sha256"
            ],
        },
    )
    assert verified.sft_rows["train"][0]["response"].endswith(
        "Final grounded analysis for sample-0000."
    )


def test_publish_is_idempotent_conflicts_on_new_inputs_and_binds_full_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified_generation(tmp_path)
    monkeypatch.setattr(workflow, "verify_full_generation", lambda **_kwargs: verified)
    kwargs = {
        "repo_root": tmp_path,
        "generation_handoff": verified.generation_handoff_path,
        "prepare_handoff": verified.prepare_handoff_path,
        "release_root": tmp_path / "canonical",
        "release_id": "release-a",
    }
    first = workflow.publish_canonical_release(**kwargs)
    second = workflow.publish_canonical_release(**kwargs)
    assert first == second
    handoff = json.loads(first.read_text(encoding="utf-8"))
    assert handoff["source_generation"]["generation_bundle_manifest"]["sha256"]
    assert handoff["source_generation"]["preparation_bundle_manifest"]["sha256"]
    assert "audit_approval" not in handoff
    assert not (first.parent / "audit/human_audit.jsonl").exists()
    generation_bundle = json.loads(
        (first.parent / handoff["source_generation"]["generation_bundle_manifest"]["path"]).read_text()
    )
    assert any(
        entry["release_path"].endswith("cache/samples/sample.json")
        for entry in generation_bundle["files"]
    )
    preparation_bundle = json.loads(
        (first.parent / handoff["source_generation"]["preparation_bundle_manifest"]["path"]).read_text()
    )
    assert any(
        entry["release_path"].endswith("prepared/train.jsonl")
        for entry in preparation_bundle["files"]
    )

    changed = replace(verified, generation_handoff_sha256="b" * 64)
    monkeypatch.setattr(workflow, "verify_full_generation", lambda **_kwargs: changed)
    with pytest.raises(workflow.CanonicalWorkflowError, match="different inputs"):
        workflow.publish_canonical_release(**kwargs)


def test_publish_rejects_generation_mutation_before_new_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verified = _verified_generation(tmp_path)
    monkeypatch.setattr(workflow, "verify_full_generation", lambda **_kwargs: verified)
    sample_cache = next(
        item.source_path
        for item in verified.generation_files
        if item.source_relative.startswith("cache/samples/")
    )
    sample_cache.write_text('{"mutated":true}\n', encoding="utf-8")
    with pytest.raises(workflow.CanonicalWorkflowError, match="source artifact SHA changed"):
        workflow.publish_canonical_release(
            repo_root=tmp_path,
            generation_handoff=verified.generation_handoff_path,
            prepare_handoff=verified.prepare_handoff_path,
            release_root=tmp_path / "canonical",
            release_id="release-mutated",
        )
    assert not (tmp_path / "canonical/release-mutated").exists()


def test_atomic_noreplace_never_overwrites_an_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "source.txt").write_text("source", encoding="utf-8")
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("raced-owner", encoding="utf-8")
    with pytest.raises(
        workflow.CanonicalWorkflowError, match="destination already exists"
    ):
        workflow._rename_noreplace(source, destination)
    assert source.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "raced-owner"


def test_publish_race_preserves_the_competing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified = _verified_generation(tmp_path)
    monkeypatch.setattr(workflow, "verify_full_generation", lambda **_kwargs: verified)
    real_noreplace = workflow._rename_noreplace

    def inject_race(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "sentinel.txt").write_text(
            "competing-publisher", encoding="utf-8"
        )
        real_noreplace(source, destination)

    monkeypatch.setattr(workflow, "_rename_noreplace", inject_race)
    with pytest.raises(
        workflow.CanonicalWorkflowError, match="destination already exists"
    ):
        workflow.publish_canonical_release(
            repo_root=tmp_path,
            generation_handoff=verified.generation_handoff_path,
            prepare_handoff=verified.prepare_handoff_path,
            release_root=tmp_path / "canonical",
            release_id="release-race",
        )
    sentinel = tmp_path / "canonical/release-race/sentinel.txt"
    assert sentinel.read_text(encoding="utf-8") == "competing-publisher"
    assert not list((tmp_path / "canonical").glob(".release-race.canonical-build-*"))


def test_projection_replay_rejects_forged_safe_prompt_with_rehashed_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = {
        "sample_id": "sample-safe",
        "split": "train",
        "meeting_date": "2020-01-01",
        "atomic_topic": "GDP Growth",
        "prompt_budget": {
            "generator": {"tokenizer_sha256": DIGEST},
            "student": {"tokenizer_sha256": DIGEST},
        },
    }
    expected_prepared_sha = sha256_text(canonical_json(prepared))
    projected = {
        "student_prompt": "the recomputed safe prompt",
        "provided_data": '{"safe":true}',
        "evidence_lineage": [{"evidence_id": "E1"}],
        "model_input_projection": {"attestation_sha256": DIGEST},
        "generator_tokenizer_sha256": DIGEST,
        "student_tokenizer_sha256": DIGEST,
        "prepared_row_sha256": expected_prepared_sha,
        "style_guide_sha256": DIGEST,
        "minutes_reference_sha256": DIGEST,
    }
    monkeypatch.setattr(
        workflow.generation,
        "_execution_inputs",
        lambda _row, *, model_input_projector: dict(projected),
    )
    forged_prompt = "attacker-replaced but self-hashed prompt"
    response = "reasoning\n</think>\nanalysis"
    sft = {
        "prompt": forged_prompt,
        "response": response,
        "provided_data": projected["provided_data"],
    }
    manifest = {
        "sample_id": prepared["sample_id"],
        "split": prepared["split"],
        "meeting_date": prepared["meeting_date"],
        "atomic_topic": prepared["atomic_topic"],
        "evidence_lineage": projected["evidence_lineage"],
        "style_guide_sha256": DIGEST,
        "tokenizer_sha256": DIGEST,
        "prompt_sha256": sha256_text(forged_prompt),
        "response_sha256": sha256_text(response),
        "provided_data_sha256": sha256_text(projected["provided_data"]),
        "generation": {
            "prepared_row_sha256": expected_prepared_sha,
            "model_input_projection": projected["model_input_projection"],
            "generator_tokenizer_sha256": DIGEST,
            "student_tokenizer_sha256": DIGEST,
            "preparation_binding_sha256": DIGEST,
            "sft_token_budget": {
                "schema_version": "chk1-sft-token-budget-v1",
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
                "max_prompt_tokens": 3072,
                "max_completion_tokens": 1024,
                "max_total_tokens": 4096,
                "overflow_policy": "error",
                "truncated": False,
                "passed": True,
            },
        },
    }
    with pytest.raises(workflow.CanonicalWorkflowError, match="safe SFT prompt"):
        workflow._replay_projected_terminals(
            population=[prepared],
            sft_rows={"train": [sft], "eval": [], "test": []},
            manifest_rows={"train": [manifest], "eval": [], "test": []},
            exclusions=[],
            model_input_projector=object(),
            preparation_binding_sha256=DIGEST,
        )


def test_cache_replay_rejects_self_hashed_forged_terminal(tmp_path: Path) -> None:
    prepared = {
        "sample_id": "sample-cache",
        "minutes_reference_sha256": DIGEST,
    }
    expected_terminal = {
        "sft_row": {"prompt": "safe", "response": "target", "provided_data": "{}"},
        "manifest_row": {"sample_id": "sample-cache"},
        "exclusion": None,
    }
    forged_terminal = json.loads(canonical_json(expected_terminal))
    forged_terminal["sft_row"]["prompt"] = "forged-but-rehashed"
    payload = workflow._payload_with_hash(
        {
            "schema_version": workflow.generation.SAMPLE_CACHE_SCHEMA_VERSION,
            "sample_id": "sample-cache",
            "mode": "full",
            "prepared_row_sha256": sha256_text(canonical_json(prepared)),
            "minutes_reference_sha256": DIGEST,
            "generation_provenance_sha256": DIGEST,
            "preparation_binding_sha256": DIGEST,
            "artifact": forged_terminal,
        }
    )
    cache_path = tmp_path / "cache/samples/sample-cache.json"
    _write_json(cache_path, payload)
    cache_manifest = {
        "generation_provenance_sha256": DIGEST,
        "preparation_binding_sha256": DIGEST,
        "entries": [
            {
                "sample_id": "sample-cache",
                "path": "cache/samples/sample-cache.json",
                "sha256": workflow.sha256_file(cache_path),
                "rows": 1,
            }
        ],
    }
    with pytest.raises(workflow.CanonicalWorkflowError, match="published terminal"):
        workflow._replay_sample_caches(
            generation_root=tmp_path,
            cache_manifest=cache_manifest,
            prepared_by_id={"sample-cache": prepared},
            terminal_artifacts={"sample-cache": expected_terminal},
            projected_inputs={
                "sample-cache": {"minutes_reference_sha256": DIGEST}
            },
            generation_provenance_sha256=DIGEST,
            preparation_binding_sha256=DIGEST,
        )
