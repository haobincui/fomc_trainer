from __future__ import annotations

import json

import pytest

from jobs.eval import paper_chk2_text_similarity_bootstrap_parallel as parallel
from jobs.eval import paper_chk2_text_similarity_scoring as canonical


def _sample(sample_id: str, split: str, meeting_id: str) -> dict[str, object]:
    return {"sample_id": sample_id, "split": split, "meeting_id": meeting_id}


def _fixture_scores() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    samples = [
        _sample("s0", "train", "m0"),
        _sample("s1", "train", "m0"),
        _sample("s2", "validation", "m1"),
        _sample("s3", "test", "m2"),
    ]
    rows: list[dict[str, object]] = []
    for model_index, model in enumerate(canonical.MODEL_ORDER):
        for sample_index, sample in enumerate(samples):
            for replicate_id in canonical.REPLICATE_IDS:
                value = (
                    0.1
                    + 0.1 * model_index
                    + 0.01 * sample_index
                    + 0.001 * replicate_id
                )
                metrics = {"mpnet_cosine": value, "bertscore_f1": value / 2}
                rows.append(
                    {
                        "model_id": model,
                        "sample_id": sample["sample_id"],
                        "replicate_id": replicate_id,
                        "raw_best_effort": dict(metrics),
                        "delivery_penalized": dict(metrics),
                        "recovered_text_empty": False,
                        "delivery_valid": True,
                        "delivery_failures": [],
                    }
                )
    return samples, rows


@pytest.fixture(autouse=True)
def _small_contract(monkeypatch):
    monkeypatch.setattr(canonical, "REPLICATE_IDS", (0, 1))
    monkeypatch.setattr(
        canonical, "EXPECTED_SPLIT_ROWS", {"train": 2, "validation": 1, "test": 1}
    )
    monkeypatch.setattr(
        canonical,
        "EXPECTED_SPLIT_MEETINGS",
        {"train": 1, "validation": 1, "test": 1},
    )
    monkeypatch.setattr(canonical, "EXPECTED_SAMPLES", 4)
    monkeypatch.setattr(canonical, "EXPECTED_MEETINGS", 3)
    monkeypatch.setattr(canonical, "EXPECTED_ROWS_PER_MODEL", 8)
    monkeypatch.setattr(canonical, "EXPECTED_TOTAL_GENERATIONS", 24)
    monkeypatch.setattr(canonical, "BOOTSTRAP_DRAWS", 32)
    monkeypatch.setattr(canonical, "SIGN_FLIP_DRAWS", 128)


def test_parallel_statistics_are_byte_equivalent_to_canonical() -> None:
    samples, rows = _fixture_scores()
    expected = canonical.build_statistics(row_scores=rows, samples=samples)
    observed = parallel.build_statistics_parallel(
        row_scores=rows, samples=samples, workers=4
    )
    assert json.dumps(observed, sort_keys=True) == json.dumps(expected, sort_keys=True)


def test_multiple_worker_counts_are_byte_equivalent() -> None:
    samples, rows = _fixture_scores()
    artifacts = []
    for workers in (1, 2, 6):
        statistics = parallel.build_statistics_parallel(
            row_scores=rows, samples=samples, workers=workers
        )
        artifacts.append(canonical._artifact_texts(statistics))
    assert artifacts[0] == artifacts[1] == artifacts[2]


def test_parallel_patch_is_restored_after_execution() -> None:
    samples, rows = _fixture_scores()
    original_draw = canonical._draw_matrix
    original_rows = canonical._draw_rows
    original_flip = canonical._sign_flip_p_value
    parallel.build_statistics_parallel(row_scores=rows, samples=samples, workers=3)
    assert canonical._draw_matrix is original_draw
    assert canonical._draw_rows is original_rows
    assert canonical._sign_flip_p_value is original_flip


@pytest.mark.parametrize("workers", [0, -1, True, 13])
def test_workers_must_be_bounded_positive_integer(workers: int) -> None:
    samples, rows = _fixture_scores()
    with pytest.raises(canonical.PaperChk2SimilarityError, match="positive integer|must not exceed"):
        parallel.build_statistics_parallel(
            row_scores=rows, samples=samples, workers=workers
        )
