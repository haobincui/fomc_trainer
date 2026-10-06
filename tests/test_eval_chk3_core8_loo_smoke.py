from __future__ import annotations

import pytest

from jobs.eval import eval_chk3_core8_loo_smoke as smoke


def test_select_meetings_uses_one_deterministic_meeting_per_bin() -> None:
    bins = [[year, year + 1] for year in range(1993, 2009, 2)]
    rows = [
        {"meeting_id": f"m-{year}-a", "meeting_start_date": f"{year}-01-01"}
        for year in range(1993, 2009)
    ]
    selected = smoke._select_meetings(rows, bins)
    assert len(selected) == 8
    assert selected == smoke._select_meetings(list(reversed(rows)), bins)
    for meeting_id, (start, end) in zip(selected, bins, strict=True):
        assert start <= int(meeting_id.split("-")[1]) <= end


def test_rendered_blocks_are_explicitly_labeled_and_parseable() -> None:
    rendered = smoke._render_analysis([("CPI", "Prices rose."), ("GDP", "Growth slowed.")])
    assert rendered.count("--- LOO-INDICATOR-BLOCK:") == 2
    assert "--- LOO-INDICATOR-BLOCK: CPI ---\nPrices rose.\n--- END-LOO-INDICATOR-BLOCK: CPI ---" in rendered
    prompt = smoke._user_prompt(rendered)
    assert prompt.startswith("Rewrite the following analysis as formal FOMC Minutes prose:\n\n")
    assert '{"analysis":' in prompt


def test_run_summary_keeps_all_four_hard_gates_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    gates = iter(
        [
            {
                "delivery_valid": True,
                "native_structure_valid": True,
                "numeric_multiset_preserved": True,
                "date_set_preserved": False,
                "degeneration_free": True,
                "preregistered_core_valid": False,
                "preregistered_core_failures": ["source_date_set_not_preserved"],
            },
            {
                "delivery_valid": True,
                "native_structure_valid": True,
                "numeric_multiset_preserved": False,
                "date_set_preserved": True,
                "degeneration_free": False,
                "preregistered_core_valid": False,
                "preregistered_core_failures": [
                    "source_numeric_multiset_not_preserved",
                    "degeneration_detected",
                ],
            },
        ]
    )
    monkeypatch.setattr(smoke.semantic_shared, "_recompute_core_hard_gate", lambda _row: next(gates))
    rows = [
        {
            "arm": "full",
            "hit_eos": True,
            "cap_reached": False,
            "strict_periodic_tail": False,
            "content_tokens": 10,
        },
        {
            "arm": "full",
            "hit_eos": True,
            "cap_reached": False,
            "strict_periodic_tail": True,
            "content_tokens": 20,
        },
    ]
    result = smoke._run_summary(rows)
    summary = result["overall"]
    assert summary["structure_delivery_rate"] == 1.0
    assert summary["numeric_fidelity_rate"] == 0.5
    assert summary["date_fidelity_rate"] == 0.5
    assert summary["degeneration_free_rate"] == 0.5
    assert result["failure_counts"] == {
        "degeneration_detected": 1,
        "source_date_set_not_preserved": 1,
        "source_numeric_multiset_not_preserved": 1,
    }
