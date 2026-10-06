from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from jobs.eval import score_paper_chk2_core8_loo_vllm_k10 as target


class _Backend:
    def __init__(self, values, *, bert: bool = False):
        self.values = list(values)
        self.bert = bert
        self.calls = []

    def score(self, candidates, references):
        self.calls.append((list(candidates), list(references)))
        if self.bert:
            return {
                "bertscore_precision": self.values,
                "bertscore_recall": self.values,
                "bertscore_f1": self.values,
            }
        return self.values

    def semantic_metadata(self):
        return {
            "chunk_audit": {
                "silent_truncation": False,
                "document_count": len(self.values),
            }
        }


def _generation(index: int, answer: str, *, valid: bool) -> dict:
    return {
        "absolute_case_index": index,
        "generation_key": f"key-{index}",
        "sample_id": f"sample-{index}",
        "answer": answer,
        "answer_sha256": target._sha_text(answer),
        "reference_minutes": f"reference {index}",
        "reference_minutes_sha256": target._sha_text(f"reference {index}"),
        "generation_diagnostics": {
            "structure_delivery": valid,
            "numeric_fidelity": valid,
            "date_fidelity": valid,
            "degeneration_free": valid,
            "preregistered_core_valid": valid,
            "failure_codes": [] if valid else ["numeric_fidelity_failed"],
            "used_to_filter_primary": False,
        },
    }


def test_semantic_metric_scores_all_nonempty_and_only_zeroes_empty():
    rows = [
        _generation(0, "first answer", valid=False),
        _generation(1, "", valid=False),
        _generation(2, "third answer", valid=True),
    ]
    backend = _Backend([0.25, 0.75])
    scored, audit = target._score_metric_rows(rows, metric="mpnet_cosine", backend=backend)
    assert [row["value"] for row in scored] == [0.25, 0.0, 0.75]
    assert scored[0]["value"] != 0.0  # a gate failure is not a score penalty
    assert scored[1]["empty_answer_fixed_to_zero"] is True
    assert backend.calls == [
        (["first answer", "third answer"], ["reference 0", "reference 2"])
    ]
    assert audit["chunk_audit"]["silent_truncation"] is False


def test_bertscore_inventory_projects_f1_only():
    rows = [_generation(0, "answer", valid=True)]
    scored, _audit = target._score_metric_rows(
        rows, metric="bertscore_f1", backend=_Backend([0.61], bert=True)
    )
    assert scored[0]["value"] == pytest.approx(0.61)


def _score_panel(meetings: int = 2, replicates: int = 3) -> list[dict]:
    rows = []
    for meeting in range(meetings):
        for variant in range(target.EXPECTED_VARIANTS):
            arm, topic = target._expected_variant(variant)
            for replicate in range(replicates):
                full = 0.8 + meeting * 0.01 + replicate * 0.001
                value = full if variant == 0 else full - variant * 0.01
                rows.append(
                    {
                        "meeting_rank": meeting,
                        "meeting_id": f"meeting-{meeting}",
                        "variant_rank": variant,
                        "replicate_id": replicate,
                        "replicate_seed": 100 + replicate,
                        "row_seed": 10_000 + meeting * 100 + replicate,
                        "reference_minutes_sha256": f"ref-{meeting}",
                        "arm": arm,
                        "intervention_topic": topic,
                        "raw_semantic_scores": {
                            "mpnet_cosine": value,
                            "bertscore_f1": value - 0.1,
                        },
                        "generation_diagnostics": {
                            "preregistered_core_valid": variant % 2 == 0,
                        },
                    }
                )
    return sorted(
        rows,
        key=lambda row: (row["meeting_rank"], row["variant_rank"], row["replicate_id"]),
    )


def test_meeting_cells_preserve_paired_full_minus_intervention_mapping():
    cells, matrices, meeting_ids = target.build_meeting_cells(
        _score_panel(), expected_meetings=2, expected_replicates=3
    )
    assert len(cells) == 2 * 8 * 2
    assert meeting_ids == ["meeting-0", "meeting-1"]
    first = matrices[(target.TOPICS[0], "exact_deletion", "mpnet_cosine")]
    neutral = matrices[(target.TOPICS[0], "token_matched_neutral", "bertscore_f1")]
    np.testing.assert_allclose(first, 0.01)
    np.testing.assert_allclose(neutral, 0.02)
    assert cells[0]["generation_diagnostics"]["used_to_filter_primary"] is False


def test_shared_bootstrap_plan_and_threaded_cell_results_are_deterministic():
    meeting_ids = [f"m{index}" for index in range(6)]
    plan_a = target.make_bootstrap_plan(
        meeting_ids=meeting_ids, draws=200, seed=77, replicates=10
    )
    plan_b = target.make_bootstrap_plan(
        meeting_ids=meeting_ids, draws=200, seed=77, replicates=10
    )
    assert plan_a.sha256 == plan_b.sha256
    np.testing.assert_array_equal(plan_a.meeting_indices, plan_b.meeting_indices)
    rng = np.random.default_rng(9)
    items = [
        ((target.TOPICS[0], target.ARMS[0], metric), rng.normal(0.02, 0.03, size=(6, 10)))
        for metric in target.METRICS
    ]
    sequential = [target._compute_cell(item, plan=plan_a) for item in items]
    for workers in (1, 2, 6):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            threaded = list(
                executor.map(lambda item: target._compute_cell(item, plan=plan_a), items)
            )
        assert [result.key for result in sequential] == [result.key for result in threaded]
        assert [result.row for result in sequential] == [result.row for result in threaded]
        for left, right in zip(sequential, threaded, strict=True):
            np.testing.assert_array_equal(left.nested_draws, right.nested_draws)


def test_k_adequacy_flags_large_decoding_variance_share():
    plan = target.make_bootstrap_plan(
        meeting_ids=[f"m{index}" for index in range(8)], draws=300, seed=4, replicates=10
    )
    alternating = np.tile(np.array([-1.0, 1.0] * 5), (8, 1))
    alternating += np.arange(8, dtype=float)[:, None] * 0.001
    result = target._compute_cell(
        ((target.TOPICS[0], target.ARMS[0], target.METRICS[0]), alternating),
        plan=plan,
    )
    assert "decoding_variance_share_above_threshold" in result.row["k_adequacy"]["increase_k_reasons"]


def test_holm_adjustment_is_joint_monotone_and_restores_input_order():
    raw = [0.01, 0.04, 0.03, 0.002]
    adjusted = target.holm_adjust(raw)
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.008])
    ordered = sorted(zip(raw, adjusted), key=lambda pair: pair[0])
    assert [value for _raw, value in ordered] == sorted(value for _raw, value in ordered)
