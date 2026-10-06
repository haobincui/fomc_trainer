from __future__ import annotations

import copy

import pytest

from jobs.eval import (
    eval_chk3_beta_core8_sentiment_neural_stochastic_schedule_v2 as neural_v2,
)
from jobs.eval import seal_chk3_beta_sentiment_regression_suite_v2 as suite_v2


class _Transformers55Tokenizer:
    """The removed build_inputs_with_special_tokens API is intentionally absent."""

    model_max_length = 512
    cls_token_id = 101
    sep_token_id = 102
    padding_side = "right"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(int(text))) if text else []

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        assert pair is False
        return 2


def test_explicit_cls_content_sep_contract_survives_removed_tokenizer_api() -> None:
    tokenizer = _Transformers55Tokenizer()
    specs = neural_v2._window_specs(
        tokenizer, "523", backend_id=neural_v2.sentiment_v1.DISTIL_BACKEND
    )
    assert [(spec.token_start, spec.token_end) for spec in specs] == [
        (0, 510),
        (382, 523),
    ]
    assert all(spec.input_ids[0] == 101 for spec in specs)
    assert all(spec.input_ids[-1] == 102 for spec in specs)
    assert all(spec.input_ids[1:-1] == spec.content_ids for spec in specs)
    assert sum(spec.aggregation_weight for spec in specs) == pytest.approx(523.0)


def test_special_token_contract_rejects_identity_or_padding_drift() -> None:
    tokenizer = _Transformers55Tokenizer()
    tokenizer.sep_token_id = 103
    with pytest.raises(neural_v2.NeuralSentimentV2Error, match="special-token"):
        neural_v2._special_token_contract(
            tokenizer, neural_v2.sentiment_v1.FINBERT_BACKEND
        )


def test_meeting_v2_preserves_exposure_and_rejects_core8_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(neural_v2.sentiment_v1, "TOPIC_COUNT", 2)
    monkeypatch.setattr(neural_v2.sentiment_v1.data_contract, "CORE_TOPICS", ("a", "b"))
    monkeypatch.setattr(neural_v2.sentiment_v1, "EXPECTED_MEETING_ROWS", 4)
    rows = []
    for arm in neural_v2.sentiment_v1.ARM_ORDER:
        replicate = None if arm == "reference" else 0
        for order, topic in enumerate(("a", "b")):
            rows.append(
                {
                    "schema_version": neural_v2.TOPIC_SCHEMA,
                    "backend": neural_v2.sentiment_v1.DISTIL_BACKEND,
                    "construct": "monetary_policy_stance_hawkish_minus_dovish",
                    "text_id": f"{arm}-{order}",
                    "text_sha256": "a" * 64,
                    "arm": arm,
                    "sample_id": f"sample-{order}",
                    "meeting_id": "2000-01-01",
                    "meeting_date": "2000-01-01",
                    "meeting_start_date": "1999-12-31",
                    "era": "test",
                    "role": "external_holdout",
                    "topic": topic,
                    "topic_order": order,
                    "replicate_id": replicate,
                    "replicate_seed": None if replicate is None else 1,
                    "cp318_selection_exposed": False,
                    "score": 0.25,
                }
            )
    meetings = neural_v2._aggregate_meeting_rows(
        rows, backend_id=neural_v2.sentiment_v1.DISTIL_BACKEND
    )
    assert len(meetings) == 4
    assert all(row["cp318_selection_exposed"] is False for row in meetings)
    altered = copy.deepcopy(rows)
    altered[1]["cp318_selection_exposed"] = True
    with pytest.raises(neural_v2.NeuralSentimentV2Error, match="metadata drift"):
        neural_v2._aggregate_meeting_rows(
            altered, backend_id=neural_v2.sentiment_v1.DISTIL_BACKEND
        )


def test_regression_exclusion_contract_is_exact_zero_variance() -> None:
    topics = [
        {
            "arm": "reference",
            "hawkish_count": 0,
            "dovish_count": 0,
        }
        for _ in range(2048)
    ]
    meetings = [
        {
            "arm": "reference",
            "role": "external_holdout" if index < 128 else "post_cutoff",
            "score": 0.0,
        }
        for index in range(256)
    ]
    loaded = {
        "topic_scores": topics,
        "meeting_scores": meetings,
        "manifest_binding": {"path": "/evidence", "sha256": "a" * 64},
    }
    exclusion = suite_v2._lexicon_exclusion(loaded)
    assert exclusion["used_for_regression"] is False
    assert exclusion["reason"] == "zero_reference_variance"
    assert exclusion["reference_topic_rows"] == 2048
    assert exclusion["reference_meeting_rows"] == 256
    assert exclusion["pre_external_reference_meeting_rows"] == 128
    assert exclusion["reference_matched_topic_rows"] == 0
    assert exclusion["reference_total_matches"] == 0
    assert exclusion["pre_external_reference_score_standard_deviation_ddof1"] == 0.0


def test_regression_exclusion_rejects_any_reference_match() -> None:
    topics = [
        {"arm": "reference", "hawkish_count": 0, "dovish_count": 0} for _ in range(2048)
    ]
    topics[0]["hawkish_count"] = 1
    meetings = [
        {
            "arm": "reference",
            "role": "external_holdout" if index < 128 else "post_cutoff",
            "score": 0.0,
        }
        for index in range(256)
    ]
    with pytest.raises(
        suite_v2.SentimentRegressionSuiteV2Error, match="zero-reference-variance"
    ):
        suite_v2._lexicon_exclusion(
            {
                "topic_scores": topics,
                "meeting_scores": meetings,
                "manifest_binding": {"path": "/evidence", "sha256": "a" * 64},
            }
        )
