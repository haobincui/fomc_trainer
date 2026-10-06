from __future__ import annotations

import json
import math
from pathlib import Path

from jobs.eval import paper_chk2_text_similarity_scoring as subject


class _FakeBertScore:
    def score(self, candidates: list[str], references: list[str]) -> dict[str, list[float]]:
        assert len(candidates) == len(references)
        return {
            "bertscore_precision": [0.8] * len(candidates),
            "bertscore_recall": [0.6] * len(candidates),
            "bertscore_f1": [0.7] * len(candidates),
        }

    def semantic_metadata(self) -> dict[str, object]:
        return {"chunk_audit": {"silent_truncation": False}}


class _FakeMpnet:
    def score(self, candidates: list[str], references: list[str]) -> list[float]:
        assert len(candidates) == len(references)
        return [0.6] * len(candidates)

    def semantic_metadata(self) -> dict[str, object]:
        return {"chunk_audit": {"silent_truncation": False}}


def _sample(sample_id: str, split: str = "train", meeting: str = "m0") -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "split": split,
        "meeting_id": meeting,
        "prompt_sha256": "a" * 64,
        "reference_sha256": "b" * 64,
        "reference": f"reference {sample_id}",
    }


def _generation(
    model: str,
    sample_id: str,
    replicate: int,
    *,
    recovered: str = "candidate",
    delivery_valid: bool = True,
) -> dict[str, object]:
    return {
        "model_id": model,
        "sample_id": sample_id,
        "replicate_id": replicate,
        "replicate_seed": 100 + replicate,
        "row_seed": 200 + replicate,
        "completion": "native reasoning</think>candidate",
        "completion_token_ids": [1, 2, 3],
        "finish_reason": "stop",
        "recovered_text": recovered,
        "delivery_valid": delivery_valid,
        "delivery_failures": [] if delivery_valid else ["missing_boundary"],
    }


def test_raw_and_delivery_penalized_are_distinct(monkeypatch) -> None:
    monkeypatch.setattr(subject, "REPLICATE_IDS", (0,))
    monkeypatch.setattr(subject, "EXPECTED_TOTAL_GENERATIONS", 6)
    samples = [_sample("s0"), _sample("s1")]
    generations = [
        _generation(model, sample["sample_id"], 0)
        for model in subject.MODEL_ORDER
        for sample in samples
    ]
    generations[0] = _generation("chk0", "s0", 0, recovered="   ")
    generations[3] = _generation(
        "chk1", "s1", 0, recovered="recoverable text", delivery_valid=False
    )
    rows, audit = subject.build_row_scores(
        samples=samples,
        generations=generations,
        bert=_FakeBertScore(),
        mpnet=_FakeMpnet(),
        score_contract_sha256="c" * 64,
    )
    subject.validate_row_scores(
        rows,
        samples=samples,
        generations=generations,
        score_contract_sha256="c" * 64,
    )
    empty = rows[0]
    invalid = rows[3]
    assert empty["raw_best_effort"] == {
        "mpnet_cosine": 0.0,
        "bertscore_f1": 0.0,
    }
    assert invalid["raw_best_effort"] == {
        "mpnet_cosine": 0.6,
        "bertscore_f1": 0.7,
    }
    assert invalid["delivery_penalized"] == {
        "mpnet_cosine": 0.0,
        "bertscore_f1": 0.0,
    }
    assert audit["scored_nonempty_pairs"] == 5


def test_generation_worker_aliases_are_normalized(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(subject, "REPLICATE_IDS", (0,))
    monkeypatch.setattr(subject, "EXPECTED_TOTAL_GENERATIONS", 3)
    monkeypatch.setattr(subject, "EXPECTED_ROWS_PER_MODEL", 1)
    samples = [_sample("s0")]
    rows = []
    for model in subject.MODEL_ORDER:
        row = _generation(model, "s0", 0)
        row["stage_id"] = row.pop("model_id")
        row["replicate_index"] = row.pop("replicate_id")
        row["output_token_ids"] = row.pop("completion_token_ids")
        rows.append(row)
    path = tmp_path / "generation_rows.jsonl"
    path.write_text(
        "".join(subject._canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    normalized, binding = subject.load_generations(path, samples=samples)
    assert len(normalized) == 3
    assert binding["rows"] == 3
    assert normalized[0]["model_id"] == "chk0"
    assert normalized[0]["replicate_id"] == 0
    assert normalized[0]["completion_token_ids"] == [1, 2, 3]


def test_paired_bootstrap_and_raw_holm_are_deterministic(monkeypatch) -> None:
    monkeypatch.setattr(subject, "REPLICATE_IDS", (0, 1))
    monkeypatch.setattr(
        subject, "EXPECTED_SPLIT_ROWS", {"train": 2, "validation": 1, "test": 1}
    )
    monkeypatch.setattr(
        subject,
        "EXPECTED_SPLIT_MEETINGS",
        {"train": 1, "validation": 1, "test": 1},
    )
    monkeypatch.setattr(subject, "EXPECTED_SAMPLES", 4)
    monkeypatch.setattr(subject, "EXPECTED_MEETINGS", 3)
    monkeypatch.setattr(subject, "EXPECTED_ROWS_PER_MODEL", 8)
    monkeypatch.setattr(subject, "EXPECTED_TOTAL_GENERATIONS", 24)
    monkeypatch.setattr(subject, "BOOTSTRAP_DRAWS", 32)
    monkeypatch.setattr(subject, "SIGN_FLIP_DRAWS", 128)
    samples = [
        _sample("s0", "train", "m0"),
        _sample("s1", "train", "m0"),
        _sample("s2", "validation", "m1"),
        _sample("s3", "test", "m2"),
    ]
    scores = []
    for model_index, model in enumerate(subject.MODEL_ORDER):
        for sample_index, sample in enumerate(samples):
            for replicate in subject.REPLICATE_IDS:
                value = 0.1 + 0.1 * model_index + 0.01 * sample_index + 0.001 * replicate
                scores.append(
                    {
                        "model_id": model,
                        "sample_id": sample["sample_id"],
                        "replicate_id": replicate,
                        "raw_best_effort": {
                            "mpnet_cosine": value,
                            "bertscore_f1": value / 2,
                        },
                        "delivery_penalized": {
                            "mpnet_cosine": value,
                            "bertscore_f1": value / 2,
                        },
                        "recovered_text_empty": False,
                        "delivery_valid": True,
                        "delivery_failures": [],
                    }
                )
    first = subject.build_statistics(row_scores=scores, samples=samples)
    second = subject.build_statistics(row_scores=scores, samples=samples)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    draw_rows = first[1]
    assert len({row["replicate_plan_sha256"] for row in draw_rows}) == 1
    for draw in draw_rows:
        indices = draw["replicate_resample_indices"]
        assert len(indices) == len(subject.REPLICATE_IDS)
        for scheme in ("meeting_cluster_primary", "row_stratified_sensitivity"):
            for policy in subject.POLICY_ORDER:
                for model in subject.MODEL_ORDER:
                    for metric in subject.METRIC_ORDER:
                        view = draw["views"][scheme][policy][model][metric]
                        expected = sum(view["per_k"][index] for index in indices) / len(
                            indices
                        )
                        assert math.isclose(
                            view["pooled_k10"], expected, rel_tol=0.0, abs_tol=1e-15
                        )
    contrasts = first[-1]
    inferential = [
        row
        for row in contrasts
        if row["bootstrap_scheme"] == "meeting_cluster_primary"
        and row["scoring_policy"] == "raw_best_effort"
        and row["aggregate_scope"] == "pooled_k10"
    ]
    assert len(inferential) == 6
    assert all(row["p_value"] is not None for row in inferential)
    assert all(row["holm_adjusted_p"] is not None for row in inferential)
    assert all(
        row["p_value"] is None
        for row in contrasts
        if row["scoring_policy"] == "delivery_penalized"
    )
    assert subject._holm([0.01, 0.03, 0.2]) == [0.03, 0.06, 0.2]
