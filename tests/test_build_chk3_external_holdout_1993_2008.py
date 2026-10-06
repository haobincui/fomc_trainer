from collections import Counter
from datetime import date, timedelta

import pytest

from jobs.eval.build_chk3_external_holdout_1993_2008 import (
    CORE_TOPICS,
    EXPECTED_END_DATES,
    EXPECTED_MEETINGS,
    NUMBER_RE,
    _analysis_text,
    _minutes_url,
    _parse_heading,
    _reference_text,
)


def test_official_regular_roster_contract_is_128_meetings() -> None:
    assert len(EXPECTED_END_DATES) == 16
    assert all(len(values) == 8 for values in EXPECTED_END_DATES.values())
    assert sum(map(len, EXPECTED_END_DATES.values())) == EXPECTED_MEETINGS == 128
    assert len(CORE_TOPICS) == 8


@pytest.mark.parametrize(
    ("heading", "start", "end"),
    [
        ("March 23 Meeting - 1993", "1993-03-23", "1993-03-23"),
        ("July 6-7 Meeting - 1993", "1993-07-06", "1993-07-07"),
        (
            "January 31-February 1 Meeting - 1995",
            "1995-01-31",
            "1995-02-01",
        ),
        ("June 30-July 1 Meeting - 1998", "1998-06-30", "1998-07-01"),
    ],
)
def test_parse_official_heading_uses_meeting_start_date(
    heading: str, start: str, end: str
) -> None:
    observed_start, observed_end = _parse_heading(heading)
    assert (observed_start, observed_end) == (start, end)
    assert (date.fromisoformat(observed_start) - timedelta(days=1)).isoformat() < start


def test_special_meeting_is_not_in_regular_roster() -> None:
    assert "2003-09-15" not in EXPECTED_END_DATES[2003]
    assert "2003-09-16" in EXPECTED_END_DATES[2003]


def test_minutes_url_selects_html_and_rejects_ambiguity() -> None:
    url = _minutes_url(
        archive_url="https://www.federalreserve.gov/monetarypolicy/fomchistorical2008.htm",
        end_date="2008-01-30",
        links=[
            ("HTML", "/monetarypolicy/fomcminutes20080130.htm"),
            ("364 KB PDF", "/monetarypolicy/files/fomcminutes20080130.pdf"),
        ],
    )
    assert url.endswith("/monetarypolicy/fomcminutes20080130.htm")
    with pytest.raises(Exception, match="ambiguous"):
        _minutes_url(
            archive_url="https://www.federalreserve.gov/",
            end_date="2008-01-30",
            links=[],
        )


def test_deterministic_analysis_reference_preserve_numeric_multiset() -> None:
    evidence = {
        "series": [
            {
                "series": "Unemployment Rate",
                "latest_value": "7.0",
                "units": "Percent",
                "relative_changes": [
                    {"relative_horizon": "previous", "value_change": "-0.1"},
                    {"relative_horizon": "long", "value_change": "0.3"},
                ],
            }
        ]
    }
    analysis = _analysis_text("Unemployment-Rate", evidence)
    reference = _reference_text("Unemployment-Rate", evidence)
    assert Counter(NUMBER_RE.findall(analysis)) == Counter(
        NUMBER_RE.findall(reference)
    )
    assert "-0.1" in analysis and "-0.1" in reference
