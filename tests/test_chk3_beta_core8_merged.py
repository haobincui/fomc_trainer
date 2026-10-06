from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

import pytest

from jobs.eval import assemble_chk3_beta_core8_meeting_documents as assembly
from jobs.eval import chk3_beta_core8_merged_contract as contract
from jobs.eval import eval_chk3_beta_core8_merged_stochastic_k10 as profile
from jobs.eval import prepare_chk3_beta_core8_merged_k10 as preparer
from open_r1.provenance import sha256_file, sha256_text
from open_r1.validator.loo_generation_spec import seal_manifest


class TinyTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(text.split())))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _release(
    root: Path,
    *,
    era: str,
    first_start: date,
    decision_cutoff_for_first: bool = False,
    duplicate_first_topic: bool = False,
    sensitivity_index: int = 5,
) -> Path:
    roster: list[dict] = []
    rows: list[dict] = []
    fixture_dates: list[tuple[date, date]] = []
    if era == "post":
        for meeting_index in range(119):
            start = first_start + timedelta(days=meeting_index * 3)
            end = start + timedelta(days=1 if meeting_index % 2 == 0 else 0)
            fixture_dates.append((start, end))
        fixture_dates[sensitivity_index] = (date(2020, 3, 15), date(2020, 3, 15))
        fixture_dates.extend(
            (date.fromisoformat(value), date.fromisoformat(value))
            for value in contract.CP318_SELECTION_EXPOSED_MEETINGS
        )
    for meeting_index in range(contract.EXPECTED_MEETINGS_PER_ERA):
        if era == "post":
            start, end = fixture_dates[meeting_index]
        else:
            start = first_start + timedelta(days=meeting_index * 3)
            end = start + timedelta(days=1 if meeting_index % 2 == 0 else 0)
        cutoff = start - timedelta(days=1)
        if decision_cutoff_for_first and meeting_index == 0:
            cutoff = end - timedelta(days=1)
        meeting_id = f"fixture-{era}-{meeting_index:03d}"
        split = (
            "train"
            if meeting_index < 102
            else "validation"
            if meeting_index < 115
            else "test"
        )
        meeting_type = (
            "emergency_unscheduled"
            if era == "post" and meeting_index == sensitivity_index
            else "regular"
        )
        sensitivity = meeting_type != "regular"
        if sensitivity:
            start = end = date(2020, 3, 15)
        roster_row = {
            "meeting_id": meeting_id,
            "meeting_start_date": start.isoformat(),
            "meeting_end_date": end.isoformat(),
            "evidence_cutoff": cutoff.isoformat(),
            "meeting_type": meeting_type,
            "scheduled": not sensitivity,
        }
        if era == "post":
            roster_row["original_post_split_role"] = split
            roster_row["original_split_role"] = split
            roster_row["source_split"] = split
            roster_row["original_qa_split"] = "eval" if split == "validation" else split
            roster_row["sensitivity_flag"] = sensitivity
            roster_row["sensitivity_reason"] = (
                "emergency_unscheduled_sunday_meeting" if sensitivity else None
            )
        roster.append(roster_row)
        for topic_index, original_topic in enumerate(contract.CORE_TOPICS):
            topic = (
                contract.CORE_TOPICS[0]
                if duplicate_first_topic and meeting_index == 0 and topic_index == 1
                else original_topic
            )
            analysis = f"For {topic}, fixture value was {meeting_index}.{topic_index}."
            reference = (
                f"The latest information for {topic} was {meeting_index}.{topic_index}."
            )
            prompt = contract.PROMPT_PREFIX + json.dumps(
                {"analysis": analysis}, separators=(",", ":")
            )
            row = {
                "sample_id": f"fixture-{era}-{meeting_index:03d}-{topic_index:02d}",
                "meeting_id": meeting_id,
                "meeting_start_date": start.isoformat(),
                "meeting_end_date": end.isoformat(),
                "evidence_cutoff": cutoff.isoformat(),
                "topic": topic,
                "prompt": prompt,
                "source_analysis": analysis,
                "reference_minutes": reference,
                "source_id": f"fixture-source:{era}:{meeting_index}:{topic_index}",
                "prompt_sha256": sha256_text(prompt),
                "source_analysis_sha256": sha256_text(analysis),
                "reference_minutes_sha256": sha256_text(reference),
                "topic_evidence_sha256": sha256_text(
                    f"fixture:{era}:{meeting_index}:{topic_index}"
                ),
            }
            if era == "post":
                row["original_post_split_role"] = split
                row["original_split_role"] = split
                row["source_split"] = split
                row["original_qa_split"] = "eval" if split == "validation" else split
                row["meeting_type"] = meeting_type
                row["sensitivity_flag"] = sensitivity
                row["sensitivity_reason"] = (
                    "emergency_unscheduled_sunday_meeting" if sensitivity else None
                )
                row["scheduled"] = not sensitivity
            rows.append(row)
    roster_path = root / "official_meeting_roster.jsonl"
    core_path = root / "panels/core8.jsonl"
    _write_jsonl(roster_path, roster)
    _write_jsonl(core_path, rows)
    manifest = seal_manifest(
        {
            "schema_version": "chk3-external-evaluation-release-v1",
            "release_id": f"fixture-{era}",
            "status": "passed",
            "immutable": True,
            "evaluation_only": True,
            "trainable": False,
            "checkpoint_selection_allowed": False,
            "promotable": False,
            "task_contract": contract.TASK_CONTRACT,
            "grain": "one_meeting_topic_per_row",
            "meeting_count": contract.EXPECTED_MEETINGS_PER_ERA,
            "core_topic_count": len(contract.CORE_TOPICS),
            "core8_rows": contract.EXPECTED_ROWS_PER_ERA,
            "evidence_cutoff_policy": contract.EVIDENCE_CUTOFF_POLICY,
            "files": {
                "official_meeting_roster.jsonl": {
                    "path": "official_meeting_roster.jsonl",
                    "sha256": sha256_file(roster_path),
                    "bytes": roster_path.stat().st_size,
                    "rows": len(roster),
                },
                "panels/core8.jsonl": {
                    "path": "panels/core8.jsonl",
                    "sha256": sha256_file(core_path),
                    "bytes": core_path.stat().st_size,
                    "rows": len(rows),
                },
            },
        }
    )
    manifest_path = root / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def test_harmonized_sources_are_exactly_256_by_core8_and_start_d1(
    tmp_path: Path,
) -> None:
    pre = _release(tmp_path / "pre", era="pre", first_start=date(1993, 1, 1))
    post = _release(tmp_path / "post", era="post", first_start=date(2009, 1, 1))
    sources = contract.load_harmonized_sources(
        pre_release_manifest=pre,
        post_release_manifest=post,
        pre_release_sha256=sha256_file(pre),
        post_release_sha256=sha256_file(post),
    )
    assert len(sources.rows) == 2048
    assert len(sources.meeting_ids) == 256
    assert set(sources.topic_counts.values()) == {256}
    assert sources.era_counts == {
        "post2008_chk3_release": 1024,
        "pre2009_external": 1024,
    }
    assert len(sources.sensitivity_meetings) == 1
    assert sources.source_role_counts == {
        "pre2009_external": {"external_holdout": {"meetings": 128, "prompts": 1024}},
        "post2008_chk3_release": {
            "train": {"meetings": 102, "prompts": 816},
            "validation": {"meetings": 13, "prompts": 104},
            "test": {"meetings": 13, "prompts": 104},
        },
    }
    assert sources.cp318_selection_exposure["meeting_count"] == 9
    assert sources.cp318_selection_exposure["prompt_count"] == 72
    assert sum(row["cp318_selection_exposed"] for row in sources.rows) == 72
    first_topics = tuple(row["topic"] for row in sources.rows[:8])
    assert first_topics == contract.CORE_TOPICS
    assert sources.rows[0]["meeting_end_date"] < sources.rows[-1]["meeting_end_date"]


def test_decision_date_d1_post_release_is_rejected(tmp_path: Path) -> None:
    pre = _release(tmp_path / "pre", era="pre", first_start=date(1993, 1, 1))
    post = _release(
        tmp_path / "post",
        era="post",
        first_start=date(2009, 1, 1),
        decision_cutoff_for_first=True,
    )
    with pytest.raises(contract.MergedCore8ContractError, match="start-date D-1"):
        contract.load_harmonized_sources(
            pre_release_manifest=pre, post_release_manifest=post
        )


def test_2020_03_15_emergency_sensitivity_metadata_is_preserved(
    tmp_path: Path,
) -> None:
    pre = _release(tmp_path / "pre", era="pre", first_start=date(1993, 1, 1))
    post = _release(
        tmp_path / "post",
        era="post",
        first_start=date(2020, 3, 15),
        sensitivity_index=0,
    )
    sources = contract.load_harmonized_sources(
        pre_release_manifest=pre, post_release_manifest=post
    )
    emergency = [row for row in sources.rows if row["meeting_end_date"] == "2020-03-15"]
    assert len(emergency) == 8
    assert {row["sensitivity_flag"] for row in emergency} == {True}
    assert {row["sensitivity_reason"] for row in emergency} == {
        "emergency_unscheduled_sunday_meeting"
    }
    assert {row["scheduled"] for row in emergency} == {False}


def test_missing_or_duplicate_core_topic_is_rejected(tmp_path: Path) -> None:
    pre = _release(tmp_path / "pre", era="pre", first_start=date(1993, 1, 1))
    post = _release(
        tmp_path / "post",
        era="post",
        first_start=date(2009, 1, 1),
        duplicate_first_topic=True,
    )
    with pytest.raises(contract.MergedCore8ContractError, match="identity/key"):
        contract.load_harmonized_sources(
            pre_release_manifest=pre, post_release_manifest=post
        )


def test_source_release_symlinks_are_rejected(tmp_path: Path) -> None:
    pre = _release(tmp_path / "pre", era="pre", first_start=date(1993, 1, 1))
    post = _release(tmp_path / "post", era="post", first_start=date(2009, 1, 1))
    linked_manifest = tmp_path / "linked-post-release.json"
    linked_manifest.symlink_to(post)
    with pytest.raises(contract.MergedCore8ContractError, match="symlink"):
        contract.load_harmonized_sources(
            pre_release_manifest=pre,
            post_release_manifest=linked_manifest,
        )

    panels = post.parent / "panels"
    actual_panels = post.parent / "actual-panels"
    panels.rename(actual_panels)
    panels.symlink_to(actual_panels, target_is_directory=True)
    with pytest.raises(contract.MergedCore8ContractError, match="contains a symlink"):
        contract.load_harmonized_sources(
            pre_release_manifest=pre,
            post_release_manifest=post,
        )


def test_profile_rejects_gpu_lock_override() -> None:
    with pytest.raises(
        profile.core.StochasticBootstrapGenerationError,
        match="cannot be overridden",
    ):
        profile._reject_gpu_lock_override(
            ["run-suite", "--gpu-lock-path", "/tmp/unsafe.lock"]
        )


def test_compatibility_rows_preserve_lineage_and_canonical_order(
    tmp_path: Path,
) -> None:
    pre = _release(tmp_path / "pre", era="pre", first_start=date(1993, 1, 1))
    post = _release(tmp_path / "post", era="post", first_start=date(2009, 1, 1))
    sources = contract.load_harmonized_sources(
        pre_release_manifest=pre, post_release_manifest=post
    )
    data, rows, metadata = preparer._build_compatibility_rows(sources, TinyTokenizer())
    assert len(data) == len(rows) == len(metadata) == 2048
    assert [row["topic"] for row in data[:8]] == list(contract.CORE_TOPICS)
    assert all(row["split"] == row["source_split"] == "test" for row in rows)
    assert data[0]["source_split"] == "external_holdout"
    post_row = next(row for row in data if row["era"] == "post2008_chk3_release")
    assert post_row["source_split"] in {"train", "validation", "test"}
    assert post_row["original_post_split_role"] == post_row["source_split"]
    assert sum(row["cp318_selection_exposed"] for row in data) == 72
    assert all(row["not_all_held_out"] is True for row in data)
    assert {row["research_scope"] for row in data} == {
        "formal_merged_panel_descriptive"
    }
    assert re.fullmatch(
        r"chk3-beta-core8-\d{4}-\d{2}-\d{2}-[0-9a-f]{24}",
        data[0]["sample_id"],
    )
    assert len(profile.REPLICATE_SEEDS) == 10
    assert 2048 * 10 * 3 == 61_440


def test_profile_exposes_transport_roles_and_cp318_selection_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pre = _release(tmp_path / "pre", era="pre", first_start=date(1993, 1, 1))
    post = _release(tmp_path / "post", era="post", first_start=date(2009, 1, 1))
    sources = contract.load_harmonized_sources(
        pre_release_manifest=pre, post_release_manifest=post
    )
    monkeypatch.setattr(profile, "_ACTIVE_SOURCES", sources)
    sampling = profile._profile_sampling_contract()
    assert sampling["not_all_held_out"] is True
    assert sampling["transport_split_role"].startswith("test_compatibility_shim")
    assert sampling["source_role_counts"] == sources.source_role_counts
    assert (
        sampling["cp318_selection_exposure"]["selection_manifest"]["sha256"]
        == contract.CP318_SELECTION_N12_MANIFEST_SHA256
    )
    payload = profile.profile_manifest_payload(sources)
    assert payload["research_scope"] == "formal_merged_panel_descriptive"
    assert payload["cp318_selection_exposure"]["generation_rows_per_model"] == 720


def _assembly_fixture() -> tuple[list[dict], dict[str, list[dict]]]:
    meeting_id = "2025-01-29"
    samples: list[dict] = []
    rows_by_model: dict[str, list[dict]] = {
        model: [] for model in ("chk1", "chk3", "chk0")
    }
    for topic_order, topic in enumerate(contract.CORE_TOPICS):
        sample_id = f"sample-{topic_order}"
        samples.append(
            {
                "sample_id": sample_id,
                "meeting_id": meeting_id,
                "topic": topic,
                "topic_order": topic_order,
            }
        )
        for model_id in rows_by_model:
            for replicate_id, replicate_seed in enumerate((11, 22)):
                answer = f"{model_id} {replicate_id} {topic}"
                rows_by_model[model_id].append(
                    {
                        "model_id": model_id,
                        "model_label": model_id + "-label",
                        "sample_id": sample_id,
                        "source_sample_id": "source-" + sample_id,
                        "meeting_id": meeting_id,
                        "era": "post2008_chk3_release",
                        "source_split": "test",
                        "original_post_split_role": "test",
                        "original_qa_split": "test",
                        "meeting_type": "regular",
                        "sensitivity_flag": False,
                        "sensitivity_reason": None,
                        "scheduled": True,
                        "meeting_start_date": "2025-01-28",
                        "meeting_end_date": meeting_id,
                        "cp318_selection_exposed": True,
                        "research_scope": profile.RESEARCH_SCOPE,
                        "transport_split_role": profile.TRANSPORT_SPLIT_ROLE,
                        "not_all_held_out": True,
                        "topic": topic,
                        "topic_order": topic_order,
                        "replicate_id": replicate_id,
                        "replicate_seed": replicate_seed,
                        "row_seed": 1000 + topic_order,
                        "finish_reason": "eos",
                        "answer": answer,
                        "answer_sha256": sha256_text(answer),
                        "completion_sha256": sha256_text("full-" + answer),
                        "generated_token_ids_sha256": sha256_text("ids-" + answer),
                        "input_truncated": False,
                    }
                )
    return samples, rows_by_model


def test_meeting_assembly_preserves_all_eight_topics_without_gate_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples, rows = _assembly_fixture()
    monkeypatch.setattr(contract, "EXPECTED_ROWS", 8)
    monkeypatch.setattr(contract, "EXPECTED_MEETINGS", 1)
    monkeypatch.setattr(profile, "REPLICATE_SEEDS", (11, 22))
    monkeypatch.setattr(assembly.core, "MODEL_ORDER", ("chk1", "chk3", "chk0"))
    monkeypatch.setattr(assembly, "EXPECTED_DOCUMENTS", 6)
    documents = assembly.assemble_documents(samples=samples, rows_by_model=rows)
    assert len(documents) == 6
    assert documents[0]["topic_order"] == list(contract.CORE_TOPICS)
    assert documents[0]["section_count"] == 8
    assert [section["topic"] for section in documents[0]["sections"]] == list(
        contract.CORE_TOPICS
    )
    assert documents[0]["input_truncation_sections"] == 0
    assert documents[0]["document_text_not_direct_512_token_model_input"] is True
    assert (
        documents[0]["recommended_downstream_scoring"]
        == "score_each_topic_section_then_equal-weight"
    )


def test_meeting_assembly_rejects_missing_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples, rows = _assembly_fixture()
    rows["chk1"].pop()
    monkeypatch.setattr(contract, "EXPECTED_ROWS", 8)
    monkeypatch.setattr(contract, "EXPECTED_MEETINGS", 1)
    monkeypatch.setattr(profile, "REPLICATE_SEEDS", (11, 22))
    monkeypatch.setattr(assembly.core, "MODEL_ORDER", ("chk1", "chk3", "chk0"))
    monkeypatch.setattr(assembly, "EXPECTED_DOCUMENTS", 6)
    with pytest.raises(assembly.MeetingAssemblyError, match="exactly Core8"):
        assembly.assemble_documents(samples=samples, rows_by_model=rows)


def test_assembly_validator_enforces_physical_rows_and_512_token_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents_path = tmp_path / "meeting_documents.jsonl"
    _write_jsonl(documents_path, [{"document_id": "a"}, {"document_id": "b"}])
    monkeypatch.setattr(assembly, "EXPECTED_DOCUMENTS", 1)
    manifest = seal_manifest(
        {
            "schema_version": assembly.MANIFEST_SCHEMA,
            "status": "complete",
            "coverage": {"documents": 1, "input_truncation_sections": 0},
            "assembly": {
                "document_text_not_direct_512_token_model_input": True,
                "recommended_downstream_scoring": (
                    assembly.DOWNSTREAM_SCORING_RECOMMENDATION
                ),
                "direct_document_scoring_requires_versioned_chunk_policy": True,
            },
            "meeting_documents": {
                "path": str(documents_path),
                "sha256": sha256_file(documents_path),
                "bytes": documents_path.stat().st_size,
                "rows": 1,
            },
        }
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(assembly.MeetingAssemblyError, match="binding drift"):
        assembly.validate_assembly(manifest_path)


def test_large_incremental_wal_state_never_rescans_growing_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wal = tmp_path / "generations.progress.v1.jsonl"
    wal.touch()
    profile._reset_wal_cache()
    scan_calls = 0
    original_scan = profile._scan_wal

    def counted_scan(path: Path) -> dict:
        nonlocal scan_calls
        scan_calls += 1
        return original_scan(path)

    monkeypatch.setattr(profile, "_scan_wal", counted_scan)
    monkeypatch.setattr(profile.core, "STATE_SCHEMA_VERSION", "fixture-state-v1")
    monkeypatch.setattr(profile.core, "EVALUATION_ID", "fixture-eval")
    results: list[dict] = []
    state_sizes: list[int] = []
    profile._profile_state_payload(
        status="initializing",
        model_id="chk1",
        results=results,
        expected_cases=20_480,
        resume_count=0,
        progress_path=wal,
    )
    with wal.open("a", encoding="utf-8") as handle:
        for index in range(5_000):
            row = {
                "model_id": "chk1",
                "sample_id": f"sample-{index // 10:04d}",
                "replicate_id": index % 10,
                "payload": "x" * 32,
            }
            handle.write(profile.core._canonical_json(row) + "\n")
            handle.flush()
            results.append(row)
            state = profile._profile_state_payload(
                status="generating",
                model_id="chk1",
                results=results,
                expected_cases=20_480,
                resume_count=0,
                progress_path=wal,
            )
            assert (
                profile._profile_sha256_file(wal) == state["partial_results"]["sha256"]
            )
            state_sizes.append(len(json.dumps(state, sort_keys=True)))
    assert scan_calls == 0
    assert profile._profile_sha256_file(wal) == sha256_file(wal)
    assert max(state_sizes) < 1_500
    assert max(state_sizes) - min(state_sizes) < 128
    assert state["completion_matrix"]["completed_cases"] == 5_000
    assert state["completion_matrix"]["completed_full_samples"] == 500
    assert not isinstance(state["completion_matrix"], list)


def test_resume_wal_scans_once_then_extends_only_the_new_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wal = tmp_path / "generations.progress.v1.jsonl"
    results = [
        {
            "model_id": "chk1",
            "sample_id": f"sample-{index // 10:04d}",
            "replicate_id": index % 10,
        }
        for index in range(10_000)
    ]
    _write_jsonl(wal, results)
    profile._reset_wal_cache()
    scan_calls = 0
    original_scan = profile._scan_wal

    def counted_scan(path: Path) -> dict:
        nonlocal scan_calls
        scan_calls += 1
        return original_scan(path)

    monkeypatch.setattr(profile, "_scan_wal", counted_scan)
    monkeypatch.setattr(profile.core, "STATE_SCHEMA_VERSION", "fixture-state-v1")
    monkeypatch.setattr(profile.core, "EVALUATION_ID", "fixture-eval")
    assert profile._profile_sha256_file(wal) == sha256_file(wal)
    profile._profile_state_payload(
        status="generating",
        model_id="chk1",
        results=results,
        expected_cases=20_480,
        resume_count=1,
        progress_path=wal,
    )
    new_row = {
        "model_id": "chk1",
        "sample_id": "sample-1000",
        "replicate_id": 0,
    }
    with wal.open("a", encoding="utf-8") as handle:
        handle.write(profile.core._canonical_json(new_row) + "\n")
        handle.flush()
    results.append(new_row)
    state = profile._profile_state_payload(
        status="generating",
        model_id="chk1",
        results=results,
        expected_cases=20_480,
        resume_count=1,
        progress_path=wal,
    )
    assert scan_calls == 1
    assert state["partial_results"]["sha256"] == sha256_file(wal)


def test_incremental_wal_rejects_noncanonical_external_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wal = tmp_path / "generations.progress.v1.jsonl"
    wal.touch()
    profile._reset_wal_cache()
    monkeypatch.setattr(profile.core, "STATE_SCHEMA_VERSION", "fixture-state-v1")
    monkeypatch.setattr(profile.core, "EVALUATION_ID", "fixture-eval")
    row = {"model_id": "chk1", "sample_id": "sample-0", "replicate_id": 0}
    with wal.open("a", encoding="utf-8") as handle:
        handle.write(profile.core._canonical_json(row) + "\n")
    profile._profile_state_payload(
        status="generating",
        model_id="chk1",
        results=[row],
        expected_cases=20_480,
        resume_count=0,
        progress_path=wal,
    )
    with wal.open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(
        profile.core.StochasticBootstrapGenerationError,
        match="truncated or changed",
    ):
        profile._profile_state_payload(
            status="failed",
            model_id="chk1",
            results=[row],
            expected_cases=20_480,
            resume_count=0,
            progress_path=wal,
        )
