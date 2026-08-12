from __future__ import annotations

import pytest

from jobs.eval.build_checkpoint_eval_dataset import (
    _series_evidence,
    extract_last_update,
    extract_minutes_section,
    render_meeting_evidence,
)
from open_r1.validator.text_leakage import (
    validate_no_prompt_reference_token_overlap,
)


HTML = """
<html><body>
<p><strong>Staff Review of the Economic Situation</strong><br/>
Economic opening.</p>
<p>Economic continuation.</p>
<p><strong>Staff Review of the Financial Situation</strong><br/>
Financial opening.</p>
<p>Financial continuation.</p>
<p><strong>Participants' Views on Current Conditions and the Economic Outlook</strong><br/>
Participants opening.</p>
<p>Participants continuation.</p>
<p><strong>Committee Policy Actions</strong><br/>Do not include.</p>
<div id="lastUpdate">Last Update: April 09, 2025</div>
</body></html>
"""


def _row(meeting_date: str = "2025-03-19") -> dict:
    return {
        "meeting_date": meeting_date,
        "information_as_of_date": "2025-03-18",
        "indicator": "Inflation",
        "source_payload": {
            "series": [
                {
                    "series_id": "CPI",
                    "title": "Consumer Price Index",
                    "units": "Percent",
                    "observations": [
                        {"date": "2024-02-01", "value": "2.0"},
                        {"date": "2025-01-01", "value": "2.4"},
                        {"date": "2025-02-01", "value": "2.5"},
                    ],
                }
            ]
        },
    }


def test_extracts_exact_sections_without_next_heading() -> None:
    economic = extract_minutes_section(
        HTML, "Staff Review of the Economic Situation"
    )
    participants = extract_minutes_section(
        HTML,
        "Participants' Views on Current Conditions and the Economic Outlook",
    )
    assert economic == "Economic opening.\n\nEconomic continuation."
    assert participants == "Participants opening.\n\nParticipants continuation."
    assert "Committee Policy Actions" not in participants


def test_extracts_official_last_update() -> None:
    assert extract_last_update(HTML) == "2025-04-09"


def test_mechanical_renderer_preserves_values_and_units() -> None:
    rendered, facts = _series_evidence(
        _row()["source_payload"]["series"][0],
        cutoff="2025-03-18",
        indicator="Inflation",
    )
    assert "latest 2025-02-01 = 2.5" in rendered
    assert "latest-minus-previous = 0.1 percentage points" in rendered
    assert "year-comparable 2024-02-01 = 2" in rendered
    assert any(
        fact["kind"] == "derived_year_percent_change" for fact in facts
    )


def test_meeting_renderer_enforces_d_minus_one() -> None:
    evidence, facts, cutoff = render_meeting_evidence(
        [_row()],
        meeting_date="2025-03-19",
    )
    assert cutoff == "2025-03-18"
    assert evidence.startswith("### Inflation")
    assert facts


def test_renderer_rejects_meeting_day_observation() -> None:
    row = _row()
    row["source_payload"]["series"][0]["observations"].append(
        {"date": "2025-03-19", "value": "2.6"}
    )
    with pytest.raises(ValueError, match="exceeds"):
        render_meeting_evidence([row], meeting_date="2025-03-19")


def test_exact_short_reference_check_remains_separate_from_ngram_guard() -> None:
    prompt = "prefix exact short reference suffix"
    reference = "exact short reference"

    assert reference in prompt
    validate_no_prompt_reference_token_overlap(
        prompt,
        reference,
        sample_id="fixture",
    )
