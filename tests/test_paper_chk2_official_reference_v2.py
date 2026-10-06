from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from jobs.generation.paper_chk2_chk1_source import load_chk1_source_rows
from jobs.generation.paper_chk2_official_reference_v2 import (
    EXACT_SECTION_BOUNDARY_METHOD,
    EXPECTED_BOUNDARY_METHOD_COUNTS,
    EXPECTED_OFFICIAL_ROSTER_SHA256,
    OFFICIAL_REFERENCE_SCHEMA_VERSION,
    PINNED_2009_ACTION_BOUNDARIES,
    PINNED_2009_BOUNDARY_METHOD,
    OfficialReferenceError,
    build_official_reference_bank,
    deserialize_official_reference_bank,
    serialize_official_reference_bank,
    verify_official_reference_bank,
)


CSV_FIELDS = (
    "line_id",
    "section_name",
    "raw_text",
    "label",
    "label_type",
    "explanation",
    "reason",
    "response",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_sha(text: str) -> str:
    import re
    import unicodedata

    value = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _row(
    line_id: int,
    section_name: str,
    raw_text: str,
    *,
    label: str = "{Other}",
    label_type: str = "other",
) -> dict[str, str]:
    return {
        "line_id": str(line_id),
        "section_name": section_name,
        "raw_text": raw_text,
        "label": label,
        "label_type": label_type,
        "explanation": "fixture",
        "reason": "fixture",
        "response": "fixture",
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _fixture_sources(tmp_path: Path) -> dict[str, Any]:
    meetings = (
        ("2009-12-16", "train", True),
        ("2010-01-27", "validation", False),
        ("2010-03-16", "test", False),
    )
    source_rows: list[dict[str, str]] = []
    roster_rows: list[dict[str, Any]] = []
    exception_boundaries: dict[str, dict[str, Any]] = {}
    for meeting_date, split, is_exception in meetings:
        source_rows.append({"meeting_date": meeting_date, "split": split})
        csv_path = tmp_path / f"minutes-{meeting_date}.csv"
        if is_exception:
            boundary_text = (
                "In their discussion of monetary policy for the period ahead, "
                "Committee members agreed to take an action."
            )
            rows = [
                _row(
                    1,
                    "Minutes of the Federal Open Market Committee",
                    "Minutes of the Federal Open Market Committee",
                    label="Pre-Non-Core",
                    label_type="Pre-Non-core",
                ),
                _row(
                    2,
                    "Participants' Views",
                    "Participants observed that activity expanded while inflation eased.",
                ),
                _row(3, "Participants' Views", boundary_text),
                _row(
                    4,
                    "Participants' Views",
                    "This post-boundary paragraph must never be retained.",
                ),
            ]
            exception_boundaries[meeting_date] = {
                "line_id": 3,
                "section_name": "Participants' Views",
                "text_sha256": _normalized_sha(boundary_text),
            }
        else:
            rows = [
                _row(
                    1,
                    "Attendance",
                    "Attended the meeting as an observer.",
                ),
                _row(
                    2,
                    "Staff Review of the Economic Situation",
                    "Economic activity expanded moderately and employment increased.",
                ),
                _row(
                    3,
                    "Staff Review of the Economic Situation",
                    "A. This list-form directive is not analytical prose.",
                ),
                _row(
                    4,
                    "Committee Policy Action",
                    "Committee members agreed on the policy action.",
                ),
                _row(
                    5,
                    "Committee Policy Action",
                    "Post-boundary policy text must not be retained.",
                ),
            ]
        _write_csv(csv_path, rows)
        roster_rows.append(
            {
                "schema_version": "chk3-post2008-start-d1-meeting-roster-v1",
                "meeting_end_date": meeting_date,
                "meeting_id": f"fomc-{meeting_date}",
                "source_split": split,
                "raw_minutes_csv": str(csv_path),
                "raw_minutes_csv_sha256": _sha256_file(csv_path),
            }
        )
    roster_path = tmp_path / "official_meeting_roster.jsonl"
    roster_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in roster_rows
        ),
        encoding="utf-8",
    )
    return {
        "source_rows": source_rows,
        "roster_path": roster_path,
        "roster_sha256": _sha256_file(roster_path),
        "expected_meeting_counts": {"train": 1, "validation": 1, "test": 1},
        "exception_boundaries": exception_boundaries,
        "expected_boundary_method_counts": {
            EXACT_SECTION_BOUNDARY_METHOD: 2,
            PINNED_2009_BOUNDARY_METHOD: 1,
        },
    }


def _build_fixture(fixture: dict[str, Any]):
    return build_official_reference_bank(
        fixture["source_rows"],
        fixture["roster_path"],
        expected_roster_sha256=fixture["roster_sha256"],
        expected_meeting_counts=fixture["expected_meeting_counts"],
        exception_boundaries=fixture["exception_boundaries"],
        expected_boundary_method_counts=fixture["expected_boundary_method_counts"],
    )


def test_build_filters_and_cuts_before_action(tmp_path: Path) -> None:
    fixture = _fixture_sources(tmp_path)
    bank = _build_fixture(fixture)

    assert bank.schema_version == OFFICIAL_REFERENCE_SCHEMA_VERSION
    assert len(bank.meetings) == 3
    assert bank.meeting_counts == {"train": 1, "validation": 1, "test": 1}
    exception = bank.reference_for_meeting("2009-12-16")
    assert exception.action_boundary.method == PINNED_2009_BOUNDARY_METHOD
    assert [paragraph.line_id for paragraph in exception.paragraphs] == [2]
    standard = bank.reference_for_meeting("2010-01-27")
    assert standard.action_boundary.method == EXACT_SECTION_BOUNDARY_METHOD
    assert standard.action_boundary.line_id == 4
    assert [paragraph.line_id for paragraph in standard.paragraphs] == [2]
    assert standard.exclusion_counts == {
        "administrative_section": 1,
        "list_item": 1,
    }
    assert all(
        paragraph.line_id < meeting.action_boundary.line_id
        for meeting in bank.meetings
        for paragraph in meeting.paragraphs
    )


def test_serialization_is_canonical_and_round_trips(tmp_path: Path) -> None:
    fixture = _fixture_sources(tmp_path)
    bank = _build_fixture(fixture)
    serialized = serialize_official_reference_bank(bank)

    assert len(serialized.splitlines()) == 4
    assert deserialize_official_reference_bank(serialized) == bank
    assert deserialize_official_reference_bank(serialized.decode("utf-8")) == bank
    verify_official_reference_bank(
        bank,
        fixture["source_rows"],
        fixture["roster_path"],
        expected_roster_sha256=fixture["roster_sha256"],
        expected_meeting_counts=fixture["expected_meeting_counts"],
        exception_boundaries=fixture["exception_boundaries"],
        expected_boundary_method_counts=fixture["expected_boundary_method_counts"],
    )


def test_deserialize_rejects_tampered_reference_text(tmp_path: Path) -> None:
    serialized = serialize_official_reference_bank(
        _build_fixture(_fixture_sources(tmp_path))
    )
    tampered = serialized.replace(b"activity expanded", b"activity declined", 1)
    with pytest.raises(OfficialReferenceError, match="text hash mismatch"):
        deserialize_official_reference_bank(tampered)


def test_verifier_rejects_in_memory_tamper(tmp_path: Path) -> None:
    fixture = _fixture_sources(tmp_path)
    bank = _build_fixture(fixture)
    first = bank.meetings[0]
    tampered_paragraph = replace(first.paragraphs[0], text="tampered")
    tampered_meeting = replace(first, paragraphs=(tampered_paragraph,))
    tampered_bank = replace(bank, meetings=(tampered_meeting, *bank.meetings[1:]))
    with pytest.raises(OfficialReferenceError, match="does not match"):
        verify_official_reference_bank(
            tampered_bank,
            fixture["source_rows"],
            fixture["roster_path"],
            expected_roster_sha256=fixture["roster_sha256"],
            expected_meeting_counts=fixture["expected_meeting_counts"],
            exception_boundaries=fixture["exception_boundaries"],
            expected_boundary_method_counts=fixture["expected_boundary_method_counts"],
        )


def test_raw_csv_sha_drift_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture_sources(tmp_path)
    csv_path = tmp_path / "minutes-2010-01-27.csv"
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(OfficialReferenceError, match="raw Minutes CSV hash mismatch"):
        _build_fixture(fixture)


def test_source_roster_meeting_join_is_exact(tmp_path: Path) -> None:
    fixture = _fixture_sources(tmp_path)
    fixture["source_rows"][0]["meeting_date"] = "2009-12-15"
    with pytest.raises(OfficialReferenceError, match="absent from chk-1 source"):
        _build_fixture(fixture)


def test_exception_boundary_requires_unique_controlled_match(tmp_path: Path) -> None:
    fixture = _fixture_sources(tmp_path)
    csv_path = tmp_path / "minutes-2009-12-16.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    rows[0]["raw_text"] = (
        "In the discussion of monetary policy for the period ahead, members spoke."
    )
    _write_csv(csv_path, rows)
    roster_rows = [
        json.loads(line) for line in fixture["roster_path"].read_text().splitlines()
    ]
    roster_rows[0]["raw_minutes_csv_sha256"] = _sha256_file(csv_path)
    fixture["roster_path"].write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in roster_rows
        ),
        encoding="utf-8",
    )
    fixture["roster_sha256"] = _sha256_file(fixture["roster_path"])
    with pytest.raises(OfficialReferenceError, match="not unique"):
        _build_fixture(fixture)


def test_real_sealed_sources_cover_all_128_meetings() -> None:
    source = load_chk1_source_rows()
    bank = build_official_reference_bank(source)

    assert bank.roster.sha256 == EXPECTED_OFFICIAL_ROSTER_SHA256
    assert len(bank.meetings) == 128
    assert bank.meeting_counts == {"train": 102, "validation": 13, "test": 13}
    assert bank.boundary_method_counts == EXPECTED_BOUNDARY_METHOD_COUNTS
    assert {
        meeting.meeting_date
        for meeting in bank.meetings
        if meeting.action_boundary.method == PINNED_2009_BOUNDARY_METHOD
    } == set(PINNED_2009_ACTION_BOUNDARIES)
    assert all(meeting.paragraph_count > 0 for meeting in bank.meetings)
    assert all(
        paragraph.line_id < meeting.action_boundary.line_id
        for meeting in bank.meetings
        for paragraph in meeting.paragraphs
    )
    assert all(
        "Committee Policy Action" not in paragraph.section_name
        for meeting in bank.meetings
        if meeting.action_boundary.method == EXACT_SECTION_BOUNDARY_METHOD
        for paragraph in meeting.paragraphs
    )
    assert sum(bank.meeting_counts.values()) == 128
    verify_official_reference_bank(bank, source)


def test_real_references_exclude_action_vote_directive_and_admin_text() -> None:
    bank = build_official_reference_bank(load_chk1_source_rows())
    retained = {
        (meeting.meeting_date, paragraph.line_id): paragraph.text
        for meeting in bank.meetings
        for paragraph in meeting.paragraphs
    }

    # These adversarially discovered rows mix otherwise plausible market prose
    # with an enacted vote or formal directive and must be removed as a whole;
    # the source paragraph is never silently truncated or rewritten.
    for identity in {
        ("2009-01-28", 105),
        ("2010-04-28", 30),
        ("2012-04-25", 66),
        ("2014-01-29", 160),
        ("2014-09-17", 76),
        ("2017-06-14", 79),
        ("2017-06-14", 81),
        ("2020-03-15", 109),
        ("2020-03-15", 110),
        ("2020-03-15", 111),
        ("2020-03-15", 112),
        ("2020-03-15", 113),
        ("2020-07-29", 81),
        ("2014-10-29", 81),
        ("2015-03-18", 91),
    }:
        assert identity not in retained

    clear_vote = re.compile(
        r"\b(?:by (?:a )?unanimous vote|"
        r"(?:the )?(?:committee|board)(?: members)? "
        r"(?:unanimously )?(?:voted|ratified)|"
        r"(?:the )?(?:committee|board|members?) (?:also )?"
        r"(?:unanimously )?approved the following resolution|"
        r"all but one member approved the following resolution|"
        r"voting (?:for|against) this action)\b",
        re.IGNORECASE,
    )
    formal_directive = re.compile(
        r"\b(?:federal open market committee(?: \(fomc\))? authorizes "
        r"the federal reserve bank of new york to conduct|"
        r"federal reserve bank of new york is directed to|"
        r"domestic policy directive|foreign currency directive)\b",
        re.IGNORECASE,
    )
    administrative = re.compile(
        r"\b(?:the meeting (?:convened|adjourned)|"
        r"voting (?:for|against) this action|return to text)\b",
        re.IGNORECASE,
    )
    enacted_action = re.compile(
        r"(?:\bwith .{0,180}? dissenting, (?:the )?committee "
        r"(?:agreed|approved|voted|decided) to\b|"
        r"\bthe vote to .{0,160}? was taken at this meeting\b|"
        r"\b(?:participants|members) unanimously "
        r"(?:supported|approved|endorsed) (?:the|this) proposal\b|"
        r"\ball participants agreed to "
        r"(?:augment|adopt|approve|publish|release|implement|establish|"
        r"extend|renew|modify)\b|"
        r"\ball but one participant could support the publication of the "
        r"following statement\b|"
        r"\bfollowing (?:this |the )?discussion, (?:the )?chair indicated "
        r"that .{0,400}? would be implemented\b)",
        re.IGNORECASE,
    )
    assert not [text for text in retained.values() if clear_vote.search(text)]
    assert not [text for text in retained.values() if formal_directive.search(text)]
    assert not [text for text in retained.values() if administrative.search(text)]
    assert not [text for text in retained.values() if enacted_action.search(text)]

    embedded_action_heading = re.compile(
        r"\bCommittee Policy Actions?(?=[A-Z])", re.IGNORECASE
    )
    embedded = {
        identity
        for identity, text in retained.items()
        if embedded_action_heading.search(text)
    }
    assert not embedded
    # The two exceptional combined headings are removed without discarding the
    # genuine analysis that follows them.
    assert retained[("2009-03-18", 52)].startswith(
        "In the discussion of the economic situation and outlook"
    )
    assert retained[("2009-04-29", 51)].startswith(
        "In conjunction with this FOMC meeting"
    )


def test_real_reference_text_normalization_is_audited_and_clean() -> None:
    bank = build_official_reference_bank(load_chk1_source_rows())
    assert bank.normalized_paragraph_count == 1059
    assert bank.text_normalization_counts == {
        "concatenated_section_heading_removed": 535,
        "cp1252_c1_character_repaired": 6,
        "nfkc_whitespace_normalized": 165,
        "soft_hyphen_removed": 198,
        "unicode_format_control_removed": 3,
        "utf8_latin1_mojibake_sequence_repaired": 819,
    }
    assert bank.reference_bank_sha256 == (
        "03ac60884c9f0297094064ae62b9aea301bc9ff10c4511aa214d7fa28e51c9bf"
    )

    encoding_artifact = re.compile(
        r"(?:â[\x80-\xbf]|Â[\x80-\xbf]|Å[\x80-\xbf]|[\x80-\x9f]|\u00ad)"
    )
    roman_list_item = re.compile(r"^\(?[ivxlcdm]+[.)]\s+", re.IGNORECASE)
    personnel = re.compile(
        r"\b(?:executive vice presidents?|deputy directors?|"
        r"associate directors?)\b.{0,180}"
        r"\b(?:federal reserve banks?|board of governors)\b",
        re.IGNORECASE,
    )
    control_heading = re.compile(
        r"^(?:balance sheet .*principles|principles for .*balance sheet|"
        r"authorization for .* operations|foreign currency directive)\b",
        re.IGNORECASE,
    )
    for meeting in bank.meetings:
        raw_path = Path(meeting.raw_minutes_csv.path)
        if not raw_path.is_absolute():
            raw_path = Path(__file__).resolve().parents[1] / raw_path
        raw_rows = list(csv.DictReader(raw_path.open(encoding="utf-8-sig", newline="")))
        section_names = {
            re.sub(
                r"\s+",
                " ",
                unicodedata.normalize("NFKC", row["section_name"]),
            ).strip()
            for row in raw_rows
        }
        for paragraph in meeting.paragraphs:
            assert paragraph.source_text_sha256
            assert not encoding_artifact.search(paragraph.text)
            assert not roman_list_item.search(paragraph.text)
            assert not personnel.search(paragraph.text)
            assert not control_heading.search(paragraph.text)
            assert not any(
                len(heading) >= 8
                and paragraph.text.startswith(heading)
                and len(paragraph.text) > len(heading)
                and paragraph.text[len(heading)].isalnum()
                for heading in section_names
            )

    retained_ids = {
        (meeting.meeting_date, paragraph.line_id)
        for meeting in bank.meetings
        for paragraph in meeting.paragraphs
    }
    for identity in {
        ("2021-09-22", 66),
        ("2017-11-01", 52),
        ("2014-10-29", 78),
        ("2019-03-20", 84),
        ("2016-09-21", 76),
        ("2022-01-26", 218),
        ("2019-03-20", 85),
        ("2022-01-26", 219),
    }:
        assert identity not in retained_ids
