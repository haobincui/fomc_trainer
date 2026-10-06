from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from jobs.eval import assemble_chk3_beta_core8_meeting_documents_vllm_k5 as assembly
from jobs.eval import chk3_beta_core8_merged_contract as data_contract
from jobs.eval import prepare_chk3_beta_core8_merged_vllm_k5 as preparation
from open_r1.provenance import sha256_file, sha256_text
from open_r1.validator.loo_generation_spec import derive_row_seed, seal_manifest


ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _token_sha(token_ids: list[int]) -> str:
    return sha256_text(_canonical(token_ids))


def _fixture() -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    samples: list[dict] = []
    ledger: list[dict] = []
    rows_by_model: dict[str, list[dict]] = {
        model_id: [] for model_id in preparation.MODEL_ORDER
    }
    meeting_id = "2025-01-29"
    for topic_order, topic in enumerate(data_contract.CORE_TOPICS):
        sample_id = f"sample-{topic_order}"
        source_prompt = f"prompt {topic}"
        source_analysis = f"analysis {topic}"
        reference_minutes = f"reference {topic}"
        sample = {
            "sample_id": sample_id,
            "source_sample_id": f"source-{sample_id}",
            "meeting_id": meeting_id,
            "topic": topic,
            "topic_order": topic_order,
            "cp318_selection_exposed": True,
            "source_prompt_sha256": sha256_text(source_prompt),
            "source_analysis_sha256": sha256_text(source_analysis),
            "reference_minutes_sha256": sha256_text(reference_minutes),
        }
        samples.append(sample)
        prompt_token_ids = [11, 100 + topic_order, 12]
        ledger_row = {
            "schema_version": preparation.LEDGER_ROW_SCHEMA,
            "absolute_prompt_index": topic_order,
            "sample_id": sample_id,
            "prompt_token_ids": prompt_token_ids,
            "prompt_token_ids_sha256": _token_sha(prompt_token_ids),
        }
        ledger.append(ledger_row)
        for model_id in preparation.MODEL_ORDER:
            for replicate_id, replicate_seed in enumerate(preparation.REPLICATE_SEEDS):
                answer = f"{model_id} answer r{replicate_id} {topic}"
                generated_text = f"reasoning <answer>{answer}</answer>"
                generated_token_ids = [200 + topic_order, 300 + replicate_id]
                rows_by_model[model_id].append(
                    {
                        "schema_version": "fixture-vllm-generation-row-v1",
                        "model_id": model_id,
                        "model_label": f"{model_id}-exact-merged-bf16",
                        "sample_id": sample_id,
                        "source_sample_id": f"source-{sample_id}",
                        "meeting_id": meeting_id,
                        "meeting_start_date": "2025-01-28",
                        "meeting_end_date": meeting_id,
                        "evidence_cutoff": "2025-01-27",
                        "era": "post2008_chk3_release",
                        "source_split": "test",
                        "original_post_split_role": "test",
                        "original_qa_split": "test",
                        "meeting_type": "regular",
                        "sensitivity_flag": False,
                        "sensitivity_reason": None,
                        "scheduled": True,
                        "research_scope": "formal_merged_panel_descriptive",
                        "transport_split_role": (
                            "test_compatibility_shim_not_a_held_out_claim"
                        ),
                        "not_all_held_out": True,
                        "cp318_selection_exposed": True,
                        "topic": topic,
                        "topic_order": topic_order,
                        "replicate_id": replicate_id,
                        "replicate_seed": replicate_seed,
                        "row_seed": derive_row_seed(replicate_seed, sample_id),
                        "prompt_token_ids_sha256": _token_sha(prompt_token_ids),
                        "source_prompt": source_prompt,
                        "source_prompt_sha256": sha256_text(source_prompt),
                        "source_analysis": source_analysis,
                        "source_analysis_sha256": sha256_text(source_analysis),
                        "reference_minutes": reference_minutes,
                        "reference_minutes_sha256": sha256_text(reference_minutes),
                        "source_release": {"sha256": "a" * 64},
                        "source_artifact_sha256s": {"panel": "b" * 64},
                        "generated_text": generated_text,
                        "generated_text_sha256": sha256_text(generated_text),
                        "answer": answer,
                        "answer_sha256": sha256_text(answer),
                        "generated_token_ids": generated_token_ids,
                        "generated_token_ids_sha256": _token_sha(generated_token_ids),
                        "input_truncated": False,
                        "finish_reason": "stop",
                        "stop_reason": 128001,
                    }
                )
    return samples, ledger, rows_by_model


@pytest.fixture
def tiny_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_contract, "EXPECTED_ROWS", 8)
    monkeypatch.setattr(data_contract, "EXPECTED_MEETINGS", 1)
    monkeypatch.setattr(preparation, "REPLICATE_SEEDS", (20260811, 21260811))
    monkeypatch.setattr(preparation, "EXPECTED_CASES_PER_MODEL", 16)
    monkeypatch.setattr(preparation, "EXPECTED_TOTAL_CASES", 48)
    monkeypatch.setattr(assembly, "DOCUMENTS_PER_MODEL", 2)
    monkeypatch.setattr(assembly, "EXPECTED_DOCUMENTS", 6)
    monkeypatch.setattr(assembly, "EXPECTED_SELECTION_EXPOSED_MEETINGS", 1)


def test_k5_assembly_preserves_complete_generation_records(
    tiny_contract: None,
) -> None:
    samples, ledger, rows_by_model = _fixture()
    documents = assembly.assemble_documents(
        samples=samples,
        token_ledger_rows=ledger,
        rows_by_model=rows_by_model,
    )
    assert len(documents) == 6
    assert [document["model_id"] for document in documents] == [
        "chk1",
        "chk1",
        "chk3",
        "chk3",
        "chk0",
        "chk0",
    ]
    first = documents[0]
    assert first["section_count"] == 8
    assert first["full_generation_records_preserved"] is True
    assert first["hard_gate_filtering"] is False
    section = first["sections"][0]
    source_row = rows_by_model["chk1"][0]
    assert all(section[key] == value for key, value in source_row.items())
    assert section["generated_token_ids"] == source_row["generated_token_ids"]
    assert section["input_prompt_token_ids"] == ledger[0]["prompt_token_ids"]
    assert section["source_release"] == source_row["source_release"]
    assert section["cp318_selection_exposed"] is True
    assert section["generation_record_sha256"] == assembly._record_sha256(source_row)
    assert first["document_text"] == "\n\n".join(
        row["answer"] for row in first["sections"]
    )


def test_k5_assembly_rejects_noncanonical_row_order(tiny_contract: None) -> None:
    samples, ledger, rows_by_model = _fixture()
    rows_by_model["chk3"][0], rows_by_model["chk3"][1] = (
        rows_by_model["chk3"][1],
        rows_by_model["chk3"][0],
    )
    with pytest.raises(
        assembly.VllmK5MeetingAssemblyError, match="generation value drift"
    ):
        assembly.assemble_documents(
            samples=samples,
            token_ledger_rows=ledger,
            rows_by_model=rows_by_model,
        )


def test_k5_assembly_rejects_token_or_source_hash_drift(
    tiny_contract: None,
) -> None:
    samples, ledger, rows_by_model = _fixture()
    rows_by_model["chk0"][0]["generated_token_ids"][0] += 1
    with pytest.raises(
        assembly.VllmK5MeetingAssemblyError, match="generated-token ID binding drift"
    ):
        assembly.assemble_documents(
            samples=samples,
            token_ledger_rows=ledger,
            rows_by_model=rows_by_model,
        )


def test_deep_validator_detects_nested_generation_tampering(
    tmp_path: Path, tiny_contract: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    samples, ledger, rows_by_model = _fixture()
    documents = assembly.assemble_documents(
        samples=samples,
        token_ledger_rows=ledger,
        rows_by_model=rows_by_model,
    )
    monkeypatch.setattr(
        assembly,
        "_load_bound_assembly_sources",
        lambda _manifest: (
            rows_by_model,
            {str(row["sample_id"]): row for row in samples},
            {str(row["sample_id"]): row for row in ledger},
        ),
    )
    documents_path = tmp_path / "meeting_documents.v1.jsonl"
    documents_path.write_text(
        "".join(_canonical(row) + "\n" for row in documents), encoding="utf-8"
    )
    manifest = seal_manifest(
        {
            "schema_version": assembly.MANIFEST_SCHEMA,
            "status": "complete",
            "immutable": True,
            "evaluation_id": preparation.EVALUATION_ID,
            "model_order": list(preparation.MODEL_ORDER),
            "backend_contract": {
                "backend": "vllm-async-engine-v1-continuous-batching",
                "weight_precision": "bfloat16",
                "quantization": None,
                "mixed_with_nf4_k10_rows": False,
                "data_parallel_size": 2,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "physical_gpu_indexes": [0, 1],
                "gpu_memory_utilization_per_replica": 0.95,
                "max_num_seqs": 16,
                "enforce_eager": True,
                "full_bf16_replica_per_gpu": True,
            },
            "runtime_limitations": {"per_row_dp_replica_assignment_unavailable": True},
            "coverage": {
                "documents": 6,
                "sections": 48,
                "input_truncation_sections": 0,
            },
            "assembly": {"full_generation_records_preserved": True},
            "meeting_documents": {
                "path": str(documents_path),
                "sha256": sha256_file(documents_path),
                "bytes": documents_path.stat().st_size,
                "rows": 6,
            },
        }
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    result = assembly.validate_assembly(manifest_path)
    assert result["documents"] == 6
    assert result["sections"] == 48

    tampered = copy.deepcopy(documents)
    tampered[0]["sections"][0]["answer"] += " tampered"
    documents_path.write_text(
        "".join(_canonical(row) + "\n" for row in tampered), encoding="utf-8"
    )
    tampered_manifest = seal_manifest(
        {
            **{
                key: value
                for key, value in manifest.items()
                if key not in {"integrity", "meeting_documents"}
            },
            "meeting_documents": {
                "path": str(documents_path),
                "sha256": sha256_file(documents_path),
                "bytes": documents_path.stat().st_size,
                "rows": 6,
            },
        }
    )
    manifest_path.write_text(json.dumps(tampered_manifest) + "\n", encoding="utf-8")
    with pytest.raises(
        assembly.VllmK5MeetingAssemblyError,
        match="section generation-record SHA drift",
    ):
        assembly.validate_assembly(manifest_path)


def test_launcher_freezes_dual_gpu_dp2_and_core8_k5_smoke() -> None:
    launcher = (ROOT / "run/eval_chk3_beta_core8_merged_vllm_k5.sh").read_text(
        encoding="utf-8"
    )
    assert "export CUDA_VISIBLE_DEVICES=0,1\n" in launcher
    assert "export PYTHONNOUSERSITE=1\n" in launcher
    assert "export VLLM_WORKER_MULTIPROC_METHOD=spawn\n" in launcher
    assert "export VLLM_DP_MASTER_IP=127.0.0.1\n" in launcher
    assert "export VLLM_DP_MASTER_PORT=0\n" in launcher
    assert 'REQUESTED_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"' in launcher
    assert 'VLLM_MAX_NUM_SEQS=""' in launcher
    assert "MODEL_ORDER=(chk1 chk3 chk0)" in launcher
    assert (
        'FAILED_BENCHMARK_ROOT_V1="${RUN_ROOT}/benchmark_max_num_seqs_chk1_core8_k5"'
        in launcher
    )
    assert (
        'BENCHMARK_ROOT="${RUN_ROOT}/benchmark_max_num_seqs_chk1_core8_k5_v2"'
        in launcher
    )
    assert "vllm_v1_async_output_remediation_receipt.json" in launcher
    assert "vllm_v1_async_output_fresh_root_addendum_receipt.json" in launcher
    assert "jobs.eval.seal_chk3_beta_core8_vllm_k5_fresh_root_addendum" in launcher
    assert "max_num_seqs_8/chk1/manifest.json" in launcher
    assert "max_num_seqs_12/chk1/manifest.json" in launcher
    assert "max_num_seqs_16/chk1/manifest.json" in launcher
    assert "select_chk3_beta_core8_vllm_k5_benchmark" in launcher
    assert ".selection.selected_max_num_seqs" in launcher
    assert "generation_smoke_core8_k5_three_models" in launcher
    assert ".coverage.rows_per_model == 40" in launcher
    assert ".coverage.total_rows == 120" in launcher
    assert launcher.count('--max-num-seqs "${max_num_seqs}"') == 2
    assert launcher.count('--max-num-seqs "${VLLM_MAX_NUM_SEQS}"') == 3
    assert "data_parallel=2 tensor_parallel=1" in launcher
    assert (
        launcher.index("require_migration_receipt\n")
        < launcher.rindex("require_async_output_remediation_receipt\n")
        < launcher.rindex("require_vllm_bf16_contract\n")
    )
