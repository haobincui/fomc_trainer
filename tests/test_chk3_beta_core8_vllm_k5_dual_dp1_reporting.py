from __future__ import annotations

import copy
from pathlib import Path

import pytest

from jobs.eval import seal_chk3_beta_core8_vllm_k5_dual_dp1_suite as suite
from jobs.eval import select_chk3_beta_core8_vllm_k5_dual_dp1_benchmark as selector


def _rows() -> list[dict]:
    rows = []
    for index in range(40):
        token_ids = [1000 + index, 128001]
        rows.append(
            {
                "model_id": "chk1",
                "sample_id": f"sample-{index // 5}",
                "replicate_id": index % 5,
                "row_seed": 123 + index,
                "absolute_case_index": index,
                "generated_token_ids": token_ids,
                "generated_text": f"text-{index}",
                "answer": f"answer-{index}",
                "finish_reason": "eos",
                "completion_sha256": f"{index + 1:064x}",
                "answer_sha256": f"{index + 2:064x}",
                "generated_token_ids_sha256": f"{index + 3:064x}",
                "input_truncated": False,
            }
        )
    return rows


def _record(candidate: int, rate: float) -> dict:
    return {
        "max_num_seqs": candidate,
        "stable": True,
        "run_manifest": {
            "path": f"/candidate-{candidate}/manifest.json",
            "sha256": f"{candidate:064x}",
            "bytes": 1,
            "payload_sha256": f"{candidate + 1:064x}",
        },
        "canonical_generations": {},
        "orchestrator_timing": {},
        "generation_contract": {"max_num_seqs_per_worker": candidate},
        "common_generation_contract": {"backend": "dual"},
        "model": {"path": "/model"},
        "sealed_anchor_manifest": {"sha256": "a" * 64},
        "cohort": {"path": "/cohort", "sha256": "b" * 64},
        "timing": {
            "parallel_generation_wall_seconds": 2.0,
            "aggregate_generated_output_tokens": 100,
            "aggregate_output_tokens_per_second": rate,
        },
        "identity_sha256": "c" * 64,
        "exact_output_sha256": "d" * 64,
    }


def test_selector_prefers_12_within_five_percent() -> None:
    result = selector._select(
        [_record(8, 100.0), _record(12, 96.0), _record(16, 101.0)]
    )
    assert result["selected_max_num_seqs"] == 12
    assert result["fastest_max_num_seqs"] == 16


def test_build_requires_exact_8_12_16_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = tmp_path / "cohort.json"
    cohort.write_text("{}\n", encoding="utf-8")
    paths = {}
    for candidate in selector.CANDIDATES:
        path = tmp_path / str(candidate) / "manifest.json"
        path.parent.mkdir()
        path.write_text("{}\n", encoding="utf-8")
        paths[candidate] = path
    rows = _rows()

    def fake_require(**kwargs: object) -> tuple[dict, list[dict]]:
        candidate = int(kwargs["candidate"])
        return _record(candidate, float(candidate)), copy.deepcopy(rows)

    monkeypatch.setattr(selector, "_require_candidate", fake_require)
    built = selector._build(
        paths=paths,
        cohort_path=cohort,
        cohort_sha256=selector.runner.core._sha256_file(cohort),
        created_at_utc="2026-08-16T00:00:00Z",
    )
    assert built["cross_candidate_equivalence"]["exact_outputs"] is True

    def changed_require(**kwargs: object) -> tuple[dict, list[dict]]:
        candidate = int(kwargs["candidate"])
        changed = copy.deepcopy(rows)
        if candidate == 16:
            changed[0]["generated_token_ids"] = [999]
        return _record(candidate, float(candidate)), changed

    monkeypatch.setattr(selector, "_require_candidate", changed_require)
    with pytest.raises(selector.DualDp1BenchmarkError, match="changed exact"):
        selector._build(
            paths=paths,
            cohort_path=cohort,
            cohort_sha256=selector.runner.core._sha256_file(cohort),
            created_at_utc="2026-08-16T00:00:00Z",
        )


def test_selected_chk1_smoke_must_exactly_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows()
    binding = {
        "path": "/selected/manifest.json",
        "sha256": "a" * 64,
        "bytes": 1,
        "payload_sha256": "b" * 64,
    }
    selection = {
        "selection": {"selected_max_num_seqs": 12},
        "candidate_runs": [{"max_num_seqs": 12, "run_manifest": binding}],
        "cohort_manifest": {"path": "/cohort", "sha256": "c" * 64},
    }
    monkeypatch.setattr(
        selector.runner,
        "load_and_validate_run",
        lambda *args, **kwargs: {"results": copy.deepcopy(rows)},
    )
    smoke = {
        "manifest": {
            "model_id": "chk1",
            "evaluation_scope": "infrastructure_smoke",
            "generation_contract": {"max_num_seqs_per_worker": 12},
            "cohort": {"path": "/cohort", "sha256": "c" * 64},
            "runtime": {
                "resume_counts": {"shard0": 0, "shard1": 0},
                "speed_measurement_valid_for_candidate_selection": True,
            },
        },
        "manifest_binding": {"path": "/smoke/manifest.json"},
        "results": copy.deepcopy(rows),
    }
    evidence = selector.validate_selected_smoke_replay(
        selection=selection, smoke_loaded=smoke
    )
    assert evidence["exact_generated_token_ids"] is True
    smoke["results"][0]["generated_text"] = "drift"
    with pytest.raises(selector.DualDp1BenchmarkError, match="does not exactly replay"):
        selector.validate_selected_smoke_replay(selection=selection, smoke_loaded=smoke)


def test_suite_smoke_gate_requires_benchmark_parity_and_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = {
        "cross_candidate_equivalence": {"exact_outputs": True, "tuple_identities": 40}
    }
    monkeypatch.setattr(
        suite.selector,
        "validate_selected_smoke_replay",
        lambda **kwargs: {"status": "passed", "rows": 40},
    )
    gate = suite._smoke_gate(selection=selection, runs={"chk1": {}})
    assert gate["formal_generation_unblocked"] is True
    selection["cross_candidate_equivalence"]["exact_outputs"] = False
    with pytest.raises(suite.DualDp1SuiteError, match="parity"):
        suite._smoke_gate(selection=selection, runs={"chk1": {}})


def test_formal_suite_requires_all_runs_to_bind_exact_smoke_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cohort = tmp_path / "cohort.json"
    cohort.write_text("{}\n", encoding="utf-8")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}\n", encoding="utf-8")
    smoke_path = tmp_path / "smoke.json"
    smoke_path.write_text("{}\n", encoding="utf-8")
    gates = {
        "benchmark_candidate_exact_token_parity_8_12_16": True,
        "selected_chk1_smoke_exact_replay": True,
        "formal_generation_unblocked": True,
    }
    selection_binding = {"path": str(selection_path.resolve()), "sha256": "b" * 64}
    smoke = {
        "manifest": {
            "generation_gates": gates,
            "benchmark_selection": selection_binding,
        },
        "manifest_binding": {"path": str(smoke_path.resolve()), "sha256": "s" * 64},
    }
    authorization = suite._formal_authorization_from_smoke(smoke)
    runs = {
        model_id: {
            "manifest": {
                "remediation_receipt": {"sha256": "r" * 64},
                "formal_authorization": copy.deepcopy(authorization),
            },
            "manifest_binding": {"path": f"/{model_id}/manifest.json"},
            "results": [{"input_truncated": False}],
        }
        for model_id in suite.MODEL_ORDER
    }
    selection = {
        "selection": {"selected_max_num_seqs": 12},
        "integrity": {"payload_sha256": "p" * 64},
    }
    monkeypatch.setattr(suite, "_load_selection", lambda *args, **kwargs: selection)
    monkeypatch.setattr(suite, "_load_runs", lambda **kwargs: runs)
    monkeypatch.setattr(
        suite,
        "load_and_validate_suite",
        lambda *args, **kwargs: smoke,
    )
    runs["chk3"]["manifest"]["formal_authorization"] = {"drift": True}
    with pytest.raises(suite.DualDp1SuiteError, match="formal runs"):
        suite.seal_suite(
            cohort_path=cohort,
            cohort_sha256="c" * 64,
            output_dir=tmp_path / "formal_drift",
            scope="formal_merged_panel",
            max_num_seqs=12,
            benchmark_selection=selection_path,
            smoke_suite_manifest=smoke_path,
        )

    runs["chk3"]["manifest"]["formal_authorization"] = copy.deepcopy(authorization)
    sealed = suite.seal_suite(
        cohort_path=cohort,
        cohort_sha256="c" * 64,
        output_dir=tmp_path / "formal_exact",
        scope="formal_merged_panel",
        max_num_seqs=12,
        benchmark_selection=selection_path,
        smoke_suite_manifest=smoke_path,
    )
    assert sealed["formal_authorization"] == authorization
