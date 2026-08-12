from __future__ import annotations

import copy
from pathlib import Path

import pytest

from jobs.retrain_v2 import audit_chk4_pre2009_correction_cp4_fixed as audit


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_real_cp4_artifacts_replay_to_advance_only() -> None:
    result = audit.audit(REPO_ROOT)
    assert result["outcome"] == "failed_not_selected_advance_to_final_cp6"
    assert result["selected"] is False
    assert result["advance_allowed"] is True
    assert result["next_and_final_candidate"] == 6
    assert result["screen_artifacts"]["results"]["rows"] == 16


def _row() -> dict[str, object]:
    return {
        "generated_token_ids": [10, 11, 128001],
        "batched_padding_normalization": {
            "schema_version": "batched-first-eos-normalization-v1",
            "method": "truncate_at_first_eos_inclusive",
            "generation_not_changed": True,
            "original_token_count": 6,
            "trimmed_token_count": 3,
            "first_eos_index": 2,
            "discarded_after_first_eos_count": 3,
            "discarded_unique_token_ids": [128001],
            "all_discarded_tokens_are_bound_eos": True,
        },
    }


def test_padding_audit_accepts_first_eos_inclusive_with_only_eos_discarded() -> None:
    audit._validate_padding_row(_row(), {128001})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("first_eos_index", 1),
        ("trimmed_token_count", 4),
        ("discarded_unique_token_ids", [7]),
        ("all_discarded_tokens_are_bound_eos", False),
    ],
)
def test_padding_audit_fails_closed_on_inconsistent_evidence(
    field: str, value: object
) -> None:
    row = _row()
    evidence = copy.deepcopy(row["batched_padding_normalization"])
    assert isinstance(evidence, dict)
    evidence[field] = value
    row["batched_padding_normalization"] = evidence
    with pytest.raises(audit.Cp4DecisionError):
        audit._validate_padding_row(row, {128001})


def test_padding_audit_accepts_no_eos_cap_case() -> None:
    row = _row()
    row["generated_token_ids"] = [10, 11, 12]
    row["batched_padding_normalization"] = {
        "schema_version": "batched-first-eos-normalization-v1",
        "method": "truncate_at_first_eos_inclusive",
        "generation_not_changed": True,
        "original_token_count": 3,
        "trimmed_token_count": 3,
        "first_eos_index": None,
        "discarded_after_first_eos_count": 0,
        "discarded_unique_token_ids": [],
        "all_discarded_tokens_are_bound_eos": False,
    }
    audit._validate_padding_row(row, {128001})
