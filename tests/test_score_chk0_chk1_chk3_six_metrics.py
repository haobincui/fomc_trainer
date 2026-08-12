from __future__ import annotations

import pytest

from jobs.eval import score_chk0_chk1_chk3_six_metrics as subject
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


class FakeBERT:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str]]] = []

    def score(self, candidates, references):
        self.calls.append((list(candidates), list(references)))
        size = len(candidates)
        return {
            "bertscore_precision": [0.7] * size,
            "bertscore_recall": [0.9] * size,
            "bertscore_f1": [0.8] * size,
        }

    def semantic_metadata(self):
        return {"chunk_audit": {"silent_truncation": False}}


class FakeMPNet:
    def score(self, candidates, references):
        return [0.6] * len(candidates)

    def semantic_metadata(self):
        return {"chunk_audit": {"silent_truncation": False}}


def _row(index: int) -> dict:
    source = f"In January 2024, the reported rate was {index + 1}.0 percent."
    answer = source
    text = f"Reasoning for case {index}.\n</think>\n{answer}"
    return {
        "sample_id": f"sample-{index:02d}",
        "length_bucket": ("short", "medium", "long")[index // 4],
        "source_prompt_sha256": f"{index + 1:064x}",
        "source_analysis_sha256": f"{index + 21:064x}",
        "reference_minutes_sha256": f"{index + 41:064x}",
        "answer_sha256": f"{index + 61:064x}",
        "completion_sha256": f"{index + 81:064x}",
        "normalized_identity": index == 0,
        "source_analysis": source,
        "answer": answer,
        "reference_minutes": source,
        "generated_text": text,
        "hit_eos": True,
        "cap_reached": False,
        "think_boundary_count": 1,
        "exact_boundary_delimiter_count": 1,
        "final_answer_single_paragraph": True,
        "full_token_4gram_repetition": 0.0,
        "tail_token_4gram_repetition": 0.0,
        "strict_periodic_tail": False,
        "quality_valid": True,
        "quality_failures": [],
    }


def _run(run_id: str) -> dict:
    model_label = run_id
    if run_id == "chk3":
        model_label = "chk3-cp318-exact-merged"
    return {
        "run_id": run_id,
        "stage_id": run_id,
        "model_label": model_label,
        "manifest_binding": {
            "path": f"/{run_id}/manifest.json",
            "sha256": "a" * 64,
            "payload_sha256": "b" * 64,
        },
        "manifest": {
            "model": {
                "path": f"/{run_id}/model",
                "files": {"model.safetensors": f"{len(run_id):064x}"},
            },
            "adapter": None,
            "artifacts": {
                "generations": {
                    "path": f"/{run_id}/generations.jsonl",
                    "sha256": "c" * 64,
                    "rows": 12,
                }
            },
        },
        "results": [_row(index) for index in range(12)],
        "generation_contract": {"generation_mode": "greedy"},
    }


def _runs() -> dict[str, dict]:
    return {run_id: _run(run_id) for run_id in subject.RUN_ORDER}


def _build(runs=None, bert=None):
    return subject.build_scorecard(
        runs=_runs() if runs is None else runs,
        bert=FakeBERT() if bert is None else bert,
        mpnet=FakeMPNet(),
        semantic_provenance={"binding": {"sha256": "d" * 64}},
        sample_manifest_binding={"sha256": "e" * 64},
        source_bindings={"six_metric_scorer": {"sha256": "f" * 64}},
        execution={"semantic_device": "cpu", "cuda_visible_devices": ""},
    )


def test_scorecard_is_sealed_and_has_exactly_six_unweighted_metrics() -> None:
    result = _build()
    assert validate_manifest_integrity(result) == result["integrity"][
        "payload_sha256"
    ]
    assert len(result["row_scores"]) == 36
    assert result["scoring_contract"]["metric_order"] == list(subject.SIX_METRICS)
    assert result["scoring_contract"]["weighted_composite_score_authorized"] is False
    assert "weighted_composite_score" not in result["summaries"]
    assert result["execution"] == {
        "semantic_device": "cpu",
        "cuda_visible_devices": "",
    }
    for run_id in subject.RUN_ORDER:
        cohorts = result["summaries"][run_id]["cohorts"]
        assert cohorts["all_n12"]["cases"] == 12
        assert cohorts["non_identity_n11"]["cases"] == 11
        assert set(cohorts["all_n12"]["six_metrics"]) == set(
            subject.SIX_METRICS
        )


def test_invalid_core_row_skips_encoders_and_gets_zero_semantic_penalty() -> None:
    runs = _runs()
    row = runs["chk3"]["results"][1]
    row["answer"] = row["answer"].replace("2.0", "9.0")
    row["generated_text"] = f"Reasoning.\n</think>\n{row['answer']}"
    bert = FakeBERT()
    result = _build(runs=runs, bert=bert)

    assert len(bert.calls) == 1
    assert len(bert.calls[0][0]) == 35
    scored = next(
        item
        for item in result["row_scores"]
        if item["run_id"] == "chk3" and item["sample_id"] == "sample-01"
    )
    assert scored["generation_metrics"] == {
        "structure_delivery": True,
        "numeric_fidelity": False,
        "date_fidelity": True,
        "degeneration_free": True,
    }
    assert scored["raw_semantic_metrics"] is None
    assert scored["semantic_zero_penalty_applied"] is True
    assert scored["six_metrics"]["bertscore_f1"] == 0.0
    assert scored["six_metrics"]["mpnet_cosine"] == 0.0
    all_n12 = result["summaries"]["chk3"]["cohorts"]["all_n12"]
    assert all_n12["six_metrics"]["numeric_fidelity"]["score"] == pytest.approx(
        11 / 12
    )
    assert all_n12["six_metrics"]["bertscore_f1"]["score"] == pytest.approx(
        11 * 0.8 / 12
    )
    assert all_n12["semantic_zero_penalty_cases"] == 1


@pytest.mark.parametrize(
    ("mutation", "failed_metric"),
    (
        ("structure", "structure_delivery"),
        ("date", "date_fidelity"),
        ("degeneration", "degeneration_free"),
    ),
)
def test_each_generation_metric_is_independently_recomputed(
    mutation: str, failed_metric: str
) -> None:
    row = _row(1)
    if mutation == "structure":
        row["answer"] = f"- {row['answer']}"
        row["generated_text"] = f"Reasoning.\n</think>\n{row['answer']}"
    elif mutation == "date":
        row["answer"] = row["answer"].replace("January", "February")
        row["generated_text"] = f"Reasoning.\n</think>\n{row['answer']}"
    else:
        row["full_token_4gram_repetition"] = 0.75
    gate = subject._gate(row)
    assert gate["metrics"][failed_metric] is False
    assert gate["preregistered_core_valid"] is False


def test_run_inventory_order_fails_closed() -> None:
    runs = _runs()
    reordered = {"chk1": runs["chk1"], "chk0": runs["chk0"], "chk3": runs["chk3"]}
    with pytest.raises(subject.SixMetricError, match="inventory/order"):
        _build(runs=reordered)
