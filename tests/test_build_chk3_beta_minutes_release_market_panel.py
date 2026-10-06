from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from jobs.eval import build_chk3_beta_minutes_release_market_panel as panel


def _meeting() -> dict[str, object]:
    return {
        "meeting_id": "fomc-20250128-20250129",
        "era": "post",
        "source_generation_era": "post2008_chk3_release",
        "generation_meeting_id": "2025-01-29",
        "original_split_role": "test",
        "original_qa_split": "test",
        "source_split": "test",
        "cp318_selection_exposed": True,
        "transport_split_role": "test_compatibility_shim_not_a_held_out_claim",
        "not_all_held_out": True,
        "meeting_start_date": "2025-01-28",
        "meeting_end_date": "2025-01-29",
        "meeting_type": "regular",
        "scheduled": True,
        "sensitivity_flag": False,
        "sensitivity_reason": None,
        "official_minutes_url": (
            "https://www.federalreserve.gov/monetarypolicy/fomcminutes20250129.htm"
        ),
    }


def test_parse_release_evidence_uses_exact_minutes_link_and_explicit_release(
    tmp_path: Path,
) -> None:
    html = b"""
    <html><body>
      <a href="/monetarypolicy/fomcminutes20240131.htm">HTML</a>
      <div>Minutes: PDF | HTML (Released February 21, 2024)</div>
      <div class="fomc-meeting__minutes">
        Minutes: <a href="/monetarypolicy/files/fomcminutes20250129.pdf">PDF</a> |
        <a href="/monetarypolicy/fomcminutes20250129.htm">HTML</a>
        (Released February 19, 2025)
      </div>
      <div>Last Update: March 1, 2025</div>
    </body></html>
    """
    path = tmp_path / "fomccalendars.htm"
    path.write_bytes(html)
    row = panel.parse_release_evidence(
        html_bytes=html,
        meeting=_meeting(),
        source_url=panel.CURRENT_CALENDAR_URL,
        source_path=path,
    )
    assert row["release_date"] == "2025-02-19"
    assert row["release_lag_calendar_days"] == 21
    assert row["matched_anchor_text"] == "HTML"
    assert row["last_update_footer_accepted"] is False
    assert row["cp318_selection_exposed"] is True
    assert row["original_split_role"] == "test"


def test_parse_release_evidence_rejects_last_update_only(tmp_path: Path) -> None:
    html = b"""
    <html><body>
      <a href="/monetarypolicy/fomcminutes20250129.htm">Minutes</a>
      <div>Last Update: February 19, 2025</div>
    </body></html>
    """
    with pytest.raises(panel.MinutesReleaseMarketPanelError, match="explicit"):
        panel.parse_release_evidence(
            html_bytes=html,
            meeting=_meeting(),
            source_url=panel.CURRENT_CALENDAR_URL,
            source_path=tmp_path / "page.htm",
        )


def _series(values: list[float | None]) -> pd.Series:
    return pd.Series(
        values,
        index=pd.to_datetime(
            [
                "2025-02-14",
                "2025-02-18",
                "2025-02-19",
                "2025-02-20",
                "2025-02-21",
            ]
        ),
        dtype=float,
    )


def _release(meeting_id: str, release_date: str, era: str = "post") -> dict[str, object]:
    return {
        **_meeting(),
        "meeting_id": meeting_id,
        "era": era,
        "release_date": release_date,
        "release_lag_calendar_days": 21,
        "evidence_text_sha256": "a" * 64,
    }


def test_market_rows_use_exact_release_day_and_common_complete_case() -> None:
    series = {
        "DGS2": _series([4.10, 4.12, 4.15, 4.13, 4.11]),
        "DGS5": _series([4.20, 4.22, 4.24, 4.21, 4.19]),
        "DGS10": _series([4.30, 4.31, 4.32, 4.30, 4.29]),
        "VIXCLS": _series([15.0, 16.0, 18.0, 17.0, 16.5]),
    }
    releases = [
        _release("kept", "2025-02-19"),
        _release("excluded", "2025-02-20", era="pre_external"),
    ]
    # Exact release-day DGS2 is missing for the second meeting; it must not be
    # silently replaced by the next available observation.
    series["DGS2"].loc[pd.Timestamp("2025-02-20")] = float("nan")
    rows, exclusions = panel.build_market_rows(releases, series)
    assert [row["meeting_id"] for row in rows] == ["kept"]
    assert rows[0]["dgs2_previous_date"] == "2025-02-18"
    assert rows[0]["dgs2_release_date"] == "2025-02-19"
    assert rows[0]["dgs2_change_bps"] == pytest.approx(3.0)
    assert rows[0]["dgs2_next_minus_release_bp"] == pytest.approx(-4.0)
    assert rows[0]["dgs2_placebo_previous_minus_previous_2_bp"] == pytest.approx(2.0)
    assert rows[0]["vix_log_change"] == pytest.approx(100.0 * __import__("math").log(18 / 16))
    assert rows[0]["exact_release_day_all_series"] is True
    assert rows[0]["missing_reasons"] == []
    assert exclusions == [
        {
            "schema_version": panel.EXCLUSION_SCHEMA,
            "meeting_id": "excluded",
            "era": "pre_external",
            "original_split_role": "test",
            "cp318_selection_exposed": True,
            "meeting_end_date": "2025-01-29",
            "release_date": "2025-02-20",
            "reason": "common_window_missing",
            "missing_series": ["DGS2"],
            "required_window": "previous_2,previous,release,next_valid_observations",
        }
    ]


def test_market_window_uses_previous_valid_observation_across_holiday_gap() -> None:
    values = pd.Series(
        [4.0, 4.1, 4.2, 4.3],
        index=pd.to_datetime(["2006-12-28", "2006-12-29", "2007-01-03", "2007-01-04"]),
    )
    window = panel._market_window(values, "2007-01-03")
    assert window is not None
    assert window["previous_date"] == "2006-12-29"
    assert window["previous_gap_calendar_days"] == 5

