from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from jobs.eval import (
    assemble_chk3_beta_core8_meeting_documents_vllm_k5_dual_dp1_stochastic_schedule_v2 as assembly,
)
from jobs.eval import chk3_beta_core8_merged_contract as data_contract
from jobs.eval import prepare_chk3_beta_core8_merged_vllm_k5 as preparation


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _complete_matrix() -> tuple[list[dict], list[dict]]:
    samples = [
        {
            "meeting_id": f"meeting-{meeting_index:03d}",
            "topic": topic,
        }
        for meeting_index in range(data_contract.EXPECTED_MEETINGS)
        for topic in data_contract.CORE_TOPICS
    ]
    documents: list[dict] = []
    for model_id in preparation.MODEL_ORDER:
        source_line = 0
        for meeting_index in range(data_contract.EXPECTED_MEETINGS):
            meeting_id = f"meeting-{meeting_index:03d}"
            for replicate_id in range(len(preparation.REPLICATE_SEEDS)):
                sections: list[dict] = []
                for topic in data_contract.CORE_TOPICS:
                    source_line += 1
                    sections.append(
                        {
                            "model_id": model_id,
                            "meeting_id": meeting_id,
                            "replicate_id": replicate_id,
                            "topic": topic,
                            "input_truncated": False,
                            "status": "ok",
                            "finish_reason": "eos",
                            "source_generation_line_number": source_line,
                            "answer": f"{model_id}/{meeting_id}/{replicate_id}/{topic}",
                        }
                    )
                document_text = "\n\n".join(section["answer"] for section in sections)
                documents.append(
                    {
                        "schema_version": assembly.DOCUMENT_SCHEMA,
                        "document_id": (
                            f"{model_id}::{meeting_id}::replicate-{replicate_id:02d}"
                        ),
                        "model_id": model_id,
                        "meeting_id": meeting_id,
                        "replicate_id": replicate_id,
                        "section_count": len(data_contract.CORE_TOPICS),
                        "sections": sections,
                        "document_text": document_text,
                        "document_text_sha256": assembly.legacy.sha256_text(
                            document_text
                        ),
                        "cp318_selection_exposed": False,
                    }
                )
    return samples, documents


def test_v4_assembly_contract_has_exact_full_matrix() -> None:
    assert assembly.EXPECTED_DOCUMENTS == 3_840
    assert assembly.DOCUMENTS_PER_MODEL == 1_280
    assert assembly.EXPECTED_SECTIONS == 30_720
    assert assembly.FIXED_MAX_NUM_SEQS == 16
    assert assembly.DOCUMENT_FILENAME == "meeting_documents.v2.jsonl"


def test_v4_assembly_matrix_audit_closes_coverage_and_ordering() -> None:
    samples, documents = _complete_matrix()
    result = assembly._audit_document_matrix(documents, samples=samples)
    assert result == {
        "models": 3,
        "meetings": 256,
        "replicates": 5,
        "topics_per_document": 8,
        "documents": 3_840,
        "documents_per_model": {"chk1": 1_280, "chk3": 1_280, "chk0": 1_280},
        "sections": 30_720,
        "normal_finish_sections": 30_720,
        "input_truncation_sections": 0,
        "unique_document_ids": 3_840,
        "cp318_selection_exposed_documents": 0,
        "source_generation_lines_exact_per_model": True,
        "model_meeting_replicate_order_exact": True,
        "core8_topic_order_exact": True,
    }

    documents[0], documents[1] = documents[1], documents[0]
    with pytest.raises(
        assembly.StochasticScheduleMeetingAssemblyError,
        match="ordering drift",
    ):
        assembly._audit_document_matrix(documents, samples=samples)


def test_v4_assembly_matrix_audit_rejects_duplicate_source_line() -> None:
    samples, documents = _complete_matrix()
    documents[0]["sections"][1]["source_generation_line_number"] = 1
    with pytest.raises(
        assembly.StochasticScheduleMeetingAssemblyError,
        match="invalid or duplicated",
    ):
        assembly._audit_document_matrix(documents, samples=samples)


def test_v4_assembly_publication_is_readonly_atomic_and_no_replace(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging-one"
    staging.mkdir()
    rows = [{"z": 2, "a": 1}, {"text": "FOMC"}]
    document_path = staging / assembly.DOCUMENT_FILENAME
    assembly._write_new_jsonl_readonly(document_path, rows)
    assert document_path.read_text(encoding="utf-8") == "".join(
        _canonical(row) + "\n" for row in rows
    )
    assert stat.S_IMODE(document_path.stat().st_mode) == 0o444

    published = tmp_path / "published"
    assembly._rename_noreplace(staging, published)
    assert not staging.exists()
    assert published.is_dir()

    second_staging = tmp_path / "staging-two"
    second_staging.mkdir()
    with pytest.raises(
        assembly.StochasticScheduleMeetingAssemblyError,
        match="already exists",
    ):
        assembly._rename_noreplace(second_staging, published)
    assert second_staging.is_dir()
    assert published.is_dir()


def test_v4_assembly_lock_owns_and_closes_its_only_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "assembly.lock"
    monkeypatch.setattr(assembly, "ASSEMBLY_LOCK", lock_path)
    with assembly._assembly_lock() as handle:
        descriptor = handle.fileno()
        assert stat.S_ISREG(os.fstat(descriptor).st_mode)
        assert lock_path.is_file()
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_v4_assembly_accepts_only_fixed_sealed_paths_and_publication_evidence() -> None:
    resolved = assembly._require_exact_target_contract(
        cohort_manifest=assembly.DEFAULT_COHORT,
        cohort_manifest_sha256=assembly.COHORT_SHA256,
        suite_manifest=assembly.DEFAULT_SUITE,
        output_dir=assembly.DEFAULT_OUTPUT,
        max_num_seqs=16,
    )
    assert resolved == tuple(
        path.resolve()
        for path in (
            assembly.DEFAULT_COHORT,
            assembly.DEFAULT_SUITE,
            assembly.DEFAULT_OUTPUT,
        )
    )
    with pytest.raises(
        assembly.StochasticScheduleMeetingAssemblyError,
        match="fixed-16 v4 target contract",
    ):
        assembly._require_exact_target_contract(
            cohort_manifest=assembly.DEFAULT_COHORT,
            cohort_manifest_sha256=assembly.COHORT_SHA256,
            suite_manifest=assembly.DEFAULT_SUITE,
            output_dir=assembly.DEFAULT_OUTPUT,
            max_num_seqs=12,
        )

    attempt = "a" * 32
    staging = f"{assembly.DEFAULT_OUTPUT.name}.staging.12345.{attempt}"
    assert assembly._validated_publication(
        {
            "publish_attempt_id": attempt,
            "staging_basename": staging,
            "preflight_free_bytes": assembly.MIN_FREE_BYTES,
            "minimum_required_free_bytes": assembly.MIN_FREE_BYTES,
            "final_root": str(assembly.DEFAULT_OUTPUT.resolve()),
        },
        final_root=assembly.DEFAULT_OUTPUT.resolve(),
    ) == (attempt, staging, assembly.MIN_FREE_BYTES)
    with pytest.raises(
        assembly.StochasticScheduleMeetingAssemblyError,
        match="publication evidence drift",
    ):
        assembly._validated_publication(
            {
                "publish_attempt_id": attempt,
                "staging_basename": staging,
                "preflight_free_bytes": True,
                "minimum_required_free_bytes": assembly.MIN_FREE_BYTES,
                "final_root": str(assembly.DEFAULT_OUTPUT.resolve()),
            },
            final_root=assembly.DEFAULT_OUTPUT.resolve(),
        )


def test_v4_retag_is_deep_and_does_not_mutate_legacy_document() -> None:
    original = [{"schema_version": "legacy", "sections": [{"answer": "x"}]}]
    tagged = assembly._retag_documents(original)
    assert tagged[0]["schema_version"] == assembly.DOCUMENT_SCHEMA
    tagged[0]["sections"][0]["answer"] = "changed"
    assert original == [{"schema_version": "legacy", "sections": [{"answer": "x"}]}]
