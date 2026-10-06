from __future__ import annotations

import pytest

from jobs.eval.score_chk3_external_holdout_semantic import _summary


def _row(*, valid: bool, bert: float, cosine: float) -> dict[str, object]:
    return {
        "core_valid": valid,
        "bertscore_f1_raw": bert,
        "mpnet_cosine_raw": cosine,
        "bertscore_f1_zero_penalized": bert if valid else 0.0,
        "mpnet_cosine_zero_penalized": cosine if valid else 0.0,
    }


def test_summary_separates_raw_zero_penalized_and_valid_only() -> None:
    result = _summary(
        [_row(valid=True, bert=0.8, cosine=0.6), _row(valid=False, bert=1.0, cosine=0.9)]
    )
    assert result["cases"] == 2
    assert result["core_valid_cases"] == 1
    assert result["zero_penalty_cases"] == 1
    assert result["bertscore_f1"]["raw_all_mean"] == pytest.approx(0.9)
    assert result["bertscore_f1"]["hard_gate_zero_mean"] == pytest.approx(0.4)
    assert result["bertscore_f1"]["valid_only_mean"] == pytest.approx(0.8)
    assert result["mpnet_cosine"]["raw_all_mean"] == pytest.approx(0.75)
    assert result["mpnet_cosine"]["hard_gate_zero_mean"] == pytest.approx(0.3)


def test_summary_handles_no_valid_rows_without_fabricating_score() -> None:
    result = _summary([_row(valid=False, bert=0.7, cosine=0.5)])
    assert result["core_valid_cases"] == 0
    assert result["bertscore_f1"]["valid_only_mean"] is None
    assert result["mpnet_cosine"]["valid_only_mean"] is None
    assert result["bertscore_f1"]["hard_gate_zero_mean"] == 0.0
