from collections import Counter
from datetime import date, timedelta

import pytest

from jobs.eval.build_chk3_external_holdout_1993_2008 import (
    NUMBER_RE,
    _analysis_text,
    _reference_text,
)
from jobs.eval.build_chk3_post2008_fixed_core8_d1 import (
    CSV_INVENTORY_SHA256,
    DEFAULT_BASE_REGISTRY,
    DEFAULT_BASE_ROSTER,
    DEFAULT_RAW_MINUTES_ROOT,
    DEFAULT_SPLIT_MANIFEST,
    DERIVED_ROSTER_PAYLOAD_SHA256,
    EMERGENCY_END_DATE,
    EXCLUDED_SENSITIVITY_END_DATE,
    EXPECTED_CORE_ROWS,
    EXPECTED_MEETINGS,
    EXPECTED_ONE_DAY_MEETINGS,
    EXPECTED_QA_SPLIT_COUNTS,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_TWO_DAY_MEETINGS,
    XLSX_INVENTORY_SHA256,
    BuildError,
    CORE_TOPICS,
    _derive_meeting_dates,
    _inventory_digest,
    _require_core8_closure,
    derive_meeting_roster,
    prepare,
)


@pytest.mark.parametrize(
    ("text", "end_date", "start_date", "mode"),
    [
        (
            "January 27-28, 2009",
            "2009-01-28",
            "2009-01-27",
            "heading_same_month",
        ),
        (
            "April 30-May 1, 2019",
            "2019-05-01",
            "2019-04-30",
            "heading_cross_month",
        ),
        (
            "March 15, 2020",
            "2020-03-15",
            "2020-03-15",
            "heading_single_day",
        ),
        (
            "January 28â\x80\x9329, 2025A joint meeting followed.",
            "2025-01-29",
            "2025-01-28",
            "heading_same_month",
        ),
    ],
)
def test_heading_parser_supports_frozen_formats(
    text: str, end_date: str, start_date: str, mode: str
) -> None:
    observed = _derive_meeting_dates([text], expected_end_date=end_date)
    assert observed.start_date == start_date
    assert observed.end_date == end_date
    assert observed.mode == mode


def test_heading_parser_uses_bounded_meeting_prose_fallback() -> None:
    text = (
        "A joint meeting of the Federal Open Market Committee and the Board of "
        "Governors was held on Tuesday, April 30, 2019, at 10:00 a.m. and "
        "continued on Wednesday, May 1, 2019, at 9:00 a.m."
    )
    observed = _derive_meeting_dates([text], expected_end_date="2019-05-01")
    assert observed.start_date == "2019-04-30"
    assert observed.end_date == "2019-05-01"
    assert observed.mode == "meeting_prose_fallback"


def test_heading_parser_fails_closed_on_wrong_end_date() -> None:
    with pytest.raises(BuildError, match="did not close uniquely"):
        _derive_meeting_dates(
            ["January 27-28, 2009"], expected_end_date="2009-01-29"
        )


def test_raw_inventory_regression_hashes_cover_all_129_pairs() -> None:
    csv_paths = sorted(DEFAULT_RAW_MINUTES_ROOT.glob("fomcminutes*_labeled.csv"))
    xlsx_paths = sorted(DEFAULT_RAW_MINUTES_ROOT.glob("fomcminutes*_labeled.xlsx"))
    assert len(csv_paths) == len(xlsx_paths) == 129
    assert _inventory_digest(csv_paths) == CSV_INVENTORY_SHA256
    assert _inventory_digest(xlsx_paths) == XLSX_INVENTORY_SHA256


def test_derived_roster_closes_128_meetings_and_cross_checks_xlsx() -> None:
    roster, inventory, audit = derive_meeting_roster(
        split_manifest=DEFAULT_SPLIT_MANIFEST,
        raw_minutes_root=DEFAULT_RAW_MINUTES_ROOT,
    )
    assert len(roster) == EXPECTED_MEETINGS == 128
    assert len(inventory) == 129
    assert Counter(row["source_split"] for row in roster) == EXPECTED_SPLIT_COUNTS
    assert Counter(row["meeting_duration_days"] for row in roster) == {
        2: EXPECTED_TWO_DAY_MEETINGS,
        1: EXPECTED_ONE_DAY_MEETINGS,
    }
    assert audit["derived_roster_payload_sha256"] == DERIVED_ROSTER_PAYLOAD_SHA256
    assert audit["csv_xlsx_date_mismatches"] == 0
    assert all(
        row["evidence_cutoff"]
        == (
            date.fromisoformat(row["meeting_start_date"]) - timedelta(days=1)
        ).isoformat()
        for row in roster
    )
    emergency = [
        row for row in roster if row["meeting_end_date"] == EMERGENCY_END_DATE
    ]
    assert len(emergency) == 1
    assert emergency[0]["meeting_type"] == "emergency"
    assert emergency[0]["scheduled"] is False
    assert emergency[0]["sensitivity_flag"] is True
    assert emergency[0]["sensitivity_reason"] == (
        "emergency_unscheduled_sunday_meeting"
    )
    assert emergency[0]["original_qa_split"] == "train"
    assert all(
        row["scheduled"] is True
        for row in roster
        if row["meeting_end_date"] != EMERGENCY_END_DATE
    )
    excluded = [
        row
        for row in inventory
        if row["meeting_end_date"] == EXCLUDED_SENSITIVITY_END_DATE
    ]
    assert len(excluded) == 1
    assert excluded[0]["admitted_by_frozen_split_manifest"] is False
    assert excluded[0]["sensitivity_flag"] is True
    assert excluded[0]["sensitivity_reason"] == (
        "excluded_from_frozen_128_meeting_roster"
    )


def test_prepare_creates_fresh_start_date_source_plan(tmp_path) -> None:
    output_root = tmp_path / "candidate"
    summary = prepare(
        output_root=output_root,
        split_manifest=DEFAULT_SPLIT_MANIFEST,
        raw_minutes_root=DEFAULT_RAW_MINUTES_ROOT,
        base_roster=DEFAULT_BASE_ROSTER,
        base_registry=DEFAULT_BASE_REGISTRY,
    )
    assert summary["meeting_count"] == EXPECTED_MEETINGS
    assert summary["core8_target_rows"] == EXPECTED_CORE_ROWS
    assert summary["two_day_meetings_requiring_fresh_source_vintage"] == 119
    assert summary["legacy_end_d1_rows_reused"] == 0
    plan = __import__("json").loads(
        (output_root / "sources/plan/source_plan.json").read_text(encoding="utf-8")
    )
    planned = {
        value
        for population in plan["populations"]
        for value in population["meeting_dates"]
    }
    assert len(planned) == EXPECTED_MEETINGS
    assert "2009-01-27" in planned
    assert "2009-01-28" not in planned
    assert plan["meeting_counts"] == EXPECTED_QA_SPLIT_COUNTS


def test_core8_gate_refuses_one_missing_tuple() -> None:
    topics = {"2009-01-27": {topic: object() for topic in CORE_TOPICS}}
    _require_core8_closure(topics, ["2009-01-27"])
    del topics["2009-01-27"][CORE_TOPICS[-1]]
    with pytest.raises(BuildError, match="publication is blocked"):
        _require_core8_closure(topics, ["2009-01-27"])


def test_deterministic_reference_preserves_numeric_multiset() -> None:
    evidence = {
        "series": [
            {
                "series": "Unemployment Rate",
                "latest_value": "4.7",
                "units": "Percent",
                "relative_changes": [
                    {"relative_horizon": "previous", "value_change": "-0.1"},
                    {"relative_horizon": "long", "value_change": "-0.2"},
                ],
            }
        ]
    }
    analysis = _analysis_text("Unemployment-Rate", evidence)
    reference = _reference_text("Unemployment-Rate", evidence)
    assert Counter(NUMBER_RE.findall(analysis)) == Counter(
        NUMBER_RE.findall(reference)
    )
