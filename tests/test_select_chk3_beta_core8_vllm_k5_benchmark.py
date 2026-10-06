from __future__ import annotations

import copy
import json
import os
import stat
from pathlib import Path

import pytest

from jobs.eval import select_chk3_beta_core8_vllm_k5_benchmark as subject
from open_r1.validator.loo_generation_spec import seal_manifest


def _rows() -> list[dict]:
    rows = []
    for index in range(subject.EXPECTED_CASES):
        text = f"completion-{index}"
        answer = f"answer-{index}"
        token_ids = [1000 + index, 128001]
        rows.append(
            {
                "model_id": "chk1",
                "sample_id": f"sample-{index // 5}",
                "replicate_id": index % 5,
                "row_seed": 20260811 + index,
                "absolute_case_index": index,
                "input_truncated": False,
                "generated_token_ids": token_ids,
                "generated_text": text,
                "answer": answer,
                "finish_reason": "eos",
                "completion_sha256": subject.core._sha256_text(text),
                "answer_sha256": subject.core._sha256_text(answer),
                "generated_token_ids_sha256": subject._sha256_value(token_ids),
            }
        )
    return rows


def _write_candidate(path: Path, candidate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"candidate": candidate}) + "\n", encoding="utf-8")


def _loaded(
    path: Path,
    candidate: int,
    rate: float,
    *,
    rows: list[dict] | None = None,
    resume_count: int = 0,
    speed_valid: bool = True,
) -> dict:
    rows = copy.deepcopy(_rows() if rows is None else rows)
    output_tokens = sum(len(row["generated_token_ids"]) for row in rows)
    seconds = output_tokens / rate
    source_hashes = {
        "source_sample_manifest_sha256": "a" * 64,
        "vllm_k5_cohort_sha256": "b" * 64,
        "vllm_k5_token_ledger_sha256": "c" * 64,
        "vllm_k5_sampling_contract_sha256": f"{candidate:064x}",
        "implementation_sources": {
            "runner": {"path": "/runner.py", "sha256": "d" * 64, "bytes": 10}
        },
    }
    engine = {
        "model_load_wall_seconds": 10.0,
        "generation_wall_seconds": seconds,
        "generated_output_tokens": output_tokens,
        "timed_session_generated_output_tokens": output_tokens,
        "completed_requests_in_timed_session": subject.EXPECTED_CASES,
        "output_tokens_per_second": rate,
        "timing_scope": "current_engine_session_after_dp_pid_validation",
        "preemption_count": None,
        "preemption_count_status": "unavailable_vllm_async_public_api",
        "speed_measurement_valid_for_candidate_selection": speed_valid,
        "visible_cuda_devices": [
            {
                "logical_cuda_index": index,
                "physical_gpu_index": index,
                "uuid": f"GPU-{index}",
                "pci_bus_id": f"00000000:0{index}:00.0",
            }
            for index in (0, 1)
        ],
        "engine_core_gpu_bindings": [
            {
                "physical_gpu_index": index,
                "data_parallel_rank": index,
                "engine_core_pid": candidate * 100 + index + 1,
            }
            for index in (0, 1)
        ],
        "gpu_lease": {"physical_gpu_indexes": [0, 1]},
        "engine": {
            "parallel_topology": ("two_full_bf16_model_replicas_one_per_physical_gpu"),
            "data_parallel_size": 2,
            "tensor_parallel_size": 1,
        },
    }
    manifest_binding = subject._file_binding(path, payload_sha256="e" * 64)
    return {
        "manifest": {
            "status": "complete",
            "model_id": "chk1",
            "evaluation_scope": "infrastructure_smoke",
            "runtime": {
                "resume_count": resume_count,
                "engine_runtime": engine,
            },
            "summary": {
                "status": "complete",
                "evaluation_scope": "infrastructure_smoke",
                "cases": subject.EXPECTED_CASES,
                "input_truncation_cases": 0,
            },
            "source_artifact_sha256s": source_hashes,
            "artifacts": {
                "canonical_generations": {
                    "path": str((path.parent / "generations.jsonl").resolve()),
                    "sha256": f"{candidate + 100:064x}",
                    "bytes": 1234,
                    "rows": subject.EXPECTED_CASES,
                }
            },
        },
        "manifest_binding": manifest_binding,
        "results": rows,
    }


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, str, dict[int, Path]]:
    cohort = tmp_path / "cohort.json"
    cohort.write_text('{"cohort":true}\n', encoding="utf-8")
    manifests = {}
    for candidate in subject.CANDIDATES:
        path = tmp_path / f"maxseq-{candidate}" / "chk1" / "manifest.json"
        _write_candidate(path, candidate)
        manifests[candidate] = path
    return cohort, subject.core._sha256_file(cohort), manifests


def _install_loader(
    monkeypatch: pytest.MonkeyPatch,
    manifests: dict[int, Path],
    rates: dict[int, float],
    *,
    changed_rows: dict[int, list[dict]] | None = None,
    resume_counts: dict[int, int] | None = None,
    speed_valid: dict[int, bool] | None = None,
) -> list[int]:
    calls = []

    def fake_load(
        manifest_path: Path,
        *,
        cohort_path: Path,
        cohort_sha256: str,
        expected_model_id: str,
        expected_scope: str,
        max_num_seqs: int,
    ) -> dict:
        del cohort_path, cohort_sha256
        assert expected_model_id == "chk1"
        assert expected_scope == "infrastructure_smoke"
        assert manifest_path == manifests[max_num_seqs].resolve()
        calls.append(max_num_seqs)
        return _loaded(
            manifest_path,
            max_num_seqs,
            rates[max_num_seqs],
            rows=(changed_rows or {}).get(max_num_seqs),
            resume_count=(resume_counts or {}).get(max_num_seqs, 0),
            speed_valid=(speed_valid or {}).get(max_num_seqs, True),
        )

    monkeypatch.setattr(subject.runner, "load_and_validate_run", fake_load)
    return calls


def test_selects_12_inside_five_percent_band_and_validates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: tuple[Path, str, dict[int, Path]],
) -> None:
    cohort, cohort_sha, manifests = files
    calls = _install_loader(monkeypatch, manifests, {8: 100.0, 12: 96.0, 16: 101.0})
    output = tmp_path / "benchmark" / "selection.v1.json"
    selected = subject.select_and_seal(
        candidate_manifest_paths=manifests,
        cohort_path=cohort,
        cohort_sha256=cohort_sha,
        output_path=output,
    )
    assert calls == [8, 12, 16]
    assert selected["selection"]["selected_max_num_seqs"] == 12
    assert selected["selection"]["fastest_max_num_seqs"] == 16
    assert selected["selection"]["candidates_within_five_percent_of_fastest"] == [
        8,
        12,
        16,
    ]
    assert selected["cross_candidate_equivalence"]["tuple_identities"] == 40
    assert all(record["stable"] for record in selected["candidate_runs"])
    assert stat.S_IMODE(output.stat().st_mode) == 0o444

    validated = subject.load_and_validate_selection(
        output, cohort_path=cohort, cohort_sha256=cohort_sha
    )
    assert validated == selected
    assert calls == [8, 12, 16, 8, 12, 16]
    with pytest.raises(subject.BenchmarkSelectionError, match="overwrite"):
        subject.select_and_seal(
            candidate_manifest_paths=manifests,
            cohort_path=cohort,
            cohort_sha256=cohort_sha,
            output_path=output,
        )


def test_selects_exact_fastest_when_12_is_outside_band() -> None:
    records = [
        {"max_num_seqs": 8, "timing": {"output_tokens_per_second": 100.0}},
        {"max_num_seqs": 12, "timing": {"output_tokens_per_second": 90.0}},
        {"max_num_seqs": 16, "timing": {"output_tokens_per_second": 110.0}},
    ]
    result = subject._select_candidate(records)
    assert result["selected_max_num_seqs"] == 16
    assert result["candidates_within_five_percent_of_fastest"] == [16]


@pytest.mark.parametrize("changed_field", ["identity", "tokens", "text", "hash"])
def test_rejects_cross_candidate_output_or_identity_drift(
    changed_field: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: tuple[Path, str, dict[int, Path]],
) -> None:
    cohort, cohort_sha, manifests = files
    changed = _rows()
    if changed_field == "identity":
        changed[0]["sample_id"] = "different-sample"
    elif changed_field == "tokens":
        changed[0]["generated_token_ids"] = [999, 128001]
        changed[0]["generated_token_ids_sha256"] = subject._sha256_value(
            changed[0]["generated_token_ids"]
        )
    elif changed_field == "text":
        changed[0]["generated_text"] = "different text"
        changed[0]["completion_sha256"] = subject.core._sha256_text(
            changed[0]["generated_text"]
        )
    else:
        changed[0]["answer_sha256"] = "f" * 64
    _install_loader(
        monkeypatch,
        manifests,
        {8: 100.0, 12: 100.0, 16: 100.0},
        changed_rows={16: changed},
    )
    with pytest.raises(
        subject.BenchmarkSelectionError,
        match="tuple identities|changed generated token IDs",
    ):
        subject.select_and_seal(
            candidate_manifest_paths=manifests,
            cohort_path=cohort,
            cohort_sha256=cohort_sha,
            output_path=tmp_path / "selection.json",
        )


@pytest.mark.parametrize("invalid", ["resume", "speed", "truncation", "gpu"])
def test_rejects_unstable_timing_truncation_or_dual_gpu_evidence(
    invalid: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: tuple[Path, str, dict[int, Path]],
) -> None:
    cohort, cohort_sha, manifests = files
    changed_rows: dict[int, list[dict]] = {}
    resume_counts: dict[int, int] = {}
    speed_valid: dict[int, bool] = {}
    if invalid == "resume":
        resume_counts[12] = 1
    elif invalid == "speed":
        speed_valid[12] = False
    elif invalid == "truncation":
        rows = _rows()
        rows[0]["input_truncated"] = True
        changed_rows[12] = rows

    def mutate_loaded(value: dict, candidate: int) -> dict:
        if invalid == "gpu" and candidate == 12:
            value["manifest"]["runtime"]["engine_runtime"]["engine_core_gpu_bindings"][
                1
            ]["engine_core_pid"] = value["manifest"]["runtime"]["engine_runtime"][
                "engine_core_gpu_bindings"
            ][0]["engine_core_pid"]
        return value

    def fake_load(manifest_path: Path, *, max_num_seqs: int, **kwargs: object) -> dict:
        del kwargs
        return mutate_loaded(
            _loaded(
                manifest_path,
                max_num_seqs,
                100.0,
                rows=changed_rows.get(max_num_seqs),
                resume_count=resume_counts.get(max_num_seqs, 0),
                speed_valid=speed_valid.get(max_num_seqs, True),
            ),
            max_num_seqs,
        )

    monkeypatch.setattr(subject.runner, "load_and_validate_run", fake_load)
    with pytest.raises(subject.BenchmarkSelectionError):
        subject.select_and_seal(
            candidate_manifest_paths=manifests,
            cohort_path=cohort,
            cohort_sha256=cohort_sha,
            output_path=tmp_path / "selection.json",
        )


def test_validation_rejects_resealed_selection_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: tuple[Path, str, dict[int, Path]],
) -> None:
    cohort, cohort_sha, manifests = files
    _install_loader(monkeypatch, manifests, {8: 100.0, 12: 100.0, 16: 100.0})
    output = tmp_path / "selection.json"
    subject.select_and_seal(
        candidate_manifest_paths=manifests,
        cohort_path=cohort,
        cohort_sha256=cohort_sha,
        output_path=output,
    )
    os.chmod(output, 0o644)
    value = json.loads(output.read_text(encoding="utf-8"))
    value.pop("integrity")
    value["selection"]["selected_max_num_seqs"] = 16
    output.write_text(
        json.dumps(seal_manifest(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(output, 0o444)
    with pytest.raises(subject.BenchmarkSelectionError, match="selection drift"):
        subject.load_and_validate_selection(
            output, cohort_path=cohort, cohort_sha256=cohort_sha
        )
