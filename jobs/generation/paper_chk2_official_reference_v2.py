"""Frozen corresponding-meeting official-Minutes references for paper chk-2 v2.

The reference bank is deliberately derived from the sealed 128-meeting roster
and the roster-bound raw CSV files.  It retains only substantive, pre-action
Minutes paragraphs.  In particular, it never infers an action boundary from a
substring appearing somewhere in a policy paragraph: 123 meetings use the
first row whose section name is exactly ``Committee Policy Action`` or
``Committee Policy Actions`` and the five structurally older 2009 documents use
date-specific, line-id- and paragraph-hash-pinned action starts.

The public verifier rebuilds the bank from the sealed sources and compares the
entire value.  This is intentional: downstream acquisition and publication can
independently prove that a serialized reference was not edited or re-cut.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jobs.generation.paper_chk2_chk1_source import (
    EXPECTED_MEETING_COUNTS,
    PreparedSourceDataset,
    PreparedSourceRow,
    sha256_file,
    sha256_json,
    sha256_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_ROSTER_PATH = (
    REPO_ROOT / "dataset/processed/retrain_v2/"
    "chk3_minutes_post2008_2009_2025_fixed_core8_d1_v1/"
    "official_meeting_roster.jsonl"
)
EXPECTED_OFFICIAL_ROSTER_SHA256 = (
    "dfcb594abe8c17d895c525925bac30d6b99763b8a9d2d73bba1a426ae752c150"
)
EXPECTED_ROSTER_SCHEMA_VERSION = "chk3-post2008-start-d1-meeting-roster-v1"
OFFICIAL_REFERENCE_SCHEMA_VERSION = (
    "paper-chk2-corresponding-official-pre-action-reference-v2"
)
OFFICIAL_REFERENCE_SERIALIZATION_VERSION = (
    "paper-chk2-corresponding-official-pre-action-reference-jsonl-v2"
)
TEXT_NORMALIZATION_SCHEMA_VERSION = "paper-chk2-official-text-normalization-v1"
TEXT_NORMALIZATION_RULES = (
    "utf8_latin1_mojibake_sequence_repaired",
    "cp1252_c1_character_repaired",
    "soft_hyphen_removed",
    "unicode_format_control_removed",
    "nfkc_whitespace_normalized",
    "concatenated_section_heading_removed",
)

EXACT_ACTION_SECTION_NAMES = frozenset(
    {"Committee Policy Action", "Committee Policy Actions"}
)
EXACT_SECTION_BOUNDARY_METHOD = "exact_committee_policy_action_section"
PINNED_2009_BOUNDARY_METHOD = "pinned_2009_action_start"
EXPECTED_BOUNDARY_METHOD_COUNTS = {
    EXACT_SECTION_BOUNDARY_METHOD: 123,
    PINNED_2009_BOUNDARY_METHOD: 5,
}

# SHA-256 is calculated from NFKC-normalized, whitespace-collapsed raw_text.
# The section is also pinned so that a future relabeling cannot silently change
# the semantics of the exceptional cut while retaining coincidentally equal
# text.
PINNED_2009_ACTION_BOUNDARIES: Mapping[str, Mapping[str, Any]] = {
    "2009-01-28": {
        "line_id": 138,
        "section_name": "Meeting Participants' Views and Committee Policy Action",
        "text_sha256": (
            "e7d27c5c92f5ea171462bcec01740c6999c165a751a89d429358dc3d41a2f793"
        ),
    },
    "2009-03-18": {
        "line_id": 60,
        "section_name": "Meeting Participants' Views and Committee Policy Action",
        "text_sha256": (
            "49f62259227efe20093f5ff02c6cec84c38de735812951ea7b49e24eb63ca2cf"
        ),
    },
    "2009-04-29": {
        "line_id": 62,
        "section_name": "Participants' Views and Committee Policy Action",
        "text_sha256": (
            "5d0caa0134ff2204b0d55b4d564b88abffabc5b59c9e6b2cf1297f290e186f15"
        ),
    },
    "2009-06-24": {
        "line_id": 71,
        "section_name": "Participants' Views and Committee Policy Action",
        "text_sha256": (
            "a3fdd13897d3a5c524fcd6b3f26313e33c331b738f6619ceef7db9e05b7f2265"
        ),
    },
    "2009-12-16": {
        "line_id": 61,
        "section_name": (
            "Participants' Views on Current Conditions and the Economic Outlook"
        ),
        "text_sha256": (
            "29b7a8cb7fc28c1ef44d21061b8be00ed8963df4a5390f58c8fb26525683aac6"
        ),
    },
}

_EXPECTED_CSV_FIELDS = (
    "line_id",
    "section_name",
    "raw_text",
    "label",
    "label_type",
    "explanation",
    "reason",
    "response",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'’][A-Za-z0-9]+)*")
_ACTION_START_RE = re.compile(
    r"^In (?:(?:their|the) discussion|the Committee['’]s discussion) "
    r"of monetary policy\b",
    re.IGNORECASE,
)
_DISALLOWED_LABEL_VALUES = frozenset(
    {
        "pre non core",
        "error",
        "delete",
        "non core",
        "not found",
        "fail to match",
    }
)
_LIST_LINE_RE = re.compile(
    r"^(?:[-*•]|\(?\d+[.)]|[A-Z][.)]|\(?[ivxlcdm]+[.)])\s+", re.IGNORECASE
)
_ADMIN_SECTION_RE = re.compile(
    r"(?:^|\b)(?:attendance|secretary|annual organizational matters|"
    r"organizational matters|notation votes?|conference call|"
    r"selection of committee officer|committee ethics discussion)(?:\b|$)",
    re.IGNORECASE,
)
_VOTE_SECTION_RE = re.compile(r"^Voting (?:for|against) this action:?", re.I)
_VOTE_TEXT_RE = re.compile(
    r"\b(?:voting (?:for|against) this action|"
    r"by (?:a )?unanimous vote|"
    r"(?:the )?(?:committee|board)(?: members)? "
    r"(?:unanimously )?(?:voted|ratified)|"
    r"(?:the )?(?:committee|board|members?) "
    r"(?:also )?(?:unanimously )?approved the following resolution|"
    r"all but one member approved the following resolution|"
    r"with [^.]{0,100}? dissenting, the committee agreed to|"
    r"the vote (?:also )?encompassed approval)\b",
    re.IGNORECASE,
)
_DIRECTIVE_TEXT_RE = re.compile(
    r"\b(?:domestic policy directive|foreign currency directive|"
    r"committee directs? the desk|execute transactions in the soma|"
    r"authorizes? and directs? (?:the )?open market desk|"
    r"authorization and direction of the federal open market committee)\b",
    re.IGNORECASE,
)
_FORMAL_ACTION_TEXT_RE = re.compile(
    r"^[\"“”']?(?:(?:the )?federal open market committee|(?:the )?committee) "
    r"(?:extends?\b.{0,80}\bauthorizations?|authorizes?|directs?|adopts?)\b|"
    r"\bfederal reserve bank of new york is directed to\b|"
    r"\bfederal open market committee(?: \(fomc\))? authorizes "
    r"the federal reserve bank of new york to conduct\b|"
    r"\bcommittee (?:unanimously )?approved the following (?:resolution|directive)\b",
    re.IGNORECASE,
)
_EMBEDDED_ACTION_HEADING_RE = re.compile(
    r"\bCommittee Policy Actions?(?=[A-Z])", re.IGNORECASE
)
_ADMIN_TEXT_RE = re.compile(
    r"\b(?:the meeting (?:convened|adjourned)|minutes of the previous meeting|"
    r"attended (?:tuesday|wednesday|the |a portion)|return to text)\b",
    re.IGNORECASE,
)
_PERSONNEL_TEXT_RE = re.compile(
    r"\b(?:executive vice presidents?|senior vice presidents?|"
    r"deputy directors?|associate directors?|assistant directors?)\b"
    r".{0,180}\b(?:federal reserve banks?|board of governors)\b",
    re.IGNORECASE,
)
_CONTROL_DOCUMENT_START_RE = re.compile(
    r"(?:\b(?:approved|adopted) the following resolution\b|"
    r"\bvoted unanimously to adopt the principles\b|"
    r"^[\"“”']?(?:balance sheet .*principles|principles for .*balance sheet|"
    r"authorization for .* operations|foreign currency directive)\b.*"
    r"\b(?:adopted|amended|effective)\b|"
    r"^[\"“”']?the federal open market committee(?: \(fomc\))? "
    r"(?:modifies|authorizes|directs|adopts)\b)",
    re.IGNORECASE,
)
_ENACTED_ACTION_RE = re.compile(
    r"(?:\bwith .{0,180}? dissenting, (?:the )?committee "
    r"(?:agreed|approved|voted|decided) to\b|"
    r"\bthe vote to .{0,160}? was taken at this meeting\b|"
    r"\b(?:participants|members) unanimously "
    r"(?:supported|approved|endorsed) (?:the|this) proposal\b|"
    r"\ball participants agreed to "
    r"(?:augment|adopt|approve|publish|release|implement|establish|extend|"
    r"renew|modify)\b|"
    r"\ball but one participant could support the publication of the "
    r"following statement\b|"
    r"\bfollowing (?:this |the )?discussion, (?:the )?chair indicated that "
    r".{0,400}? would be implemented\b|"
    r"\bparticipants (?:also )?indicated strong support for related actions "
    r"taken by the board of governors\b)",
    re.IGNORECASE,
)
_ENACTED_ACTION_BLOCK_START_RE = re.compile(
    r"(?:\ball but one participant could support the publication of the "
    r"following statement\b|"
    r"\bparticipants unanimously supported the proposal\b|"
    r"\bparticipants generally agreed that,? .{0,180}? it would be "
    r"appropriate to increase the federal reserve['’]s holdings\b)",
    re.IGNORECASE,
)
_DATE_HEADING_RE = re.compile(
    r"^(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2}(?:\s*[-–—]\s*|\s+to\s+)?"
    r"(?:[A-Z][a-z]+\s+)?\d{0,2},?\s+\d{4}$",
    re.IGNORECASE,
)
_UTF8_LATIN1_MOJIBAKE_RE = re.compile(r"(?:â[\x80-\xbf]{2}|Â[\x80-\xbf]|Å[\x80-\xbf])")
_C1_CONTROL_RE = re.compile(r"[\x80-\x9f]")
_ENCODING_ARTIFACT_RE = re.compile(
    r"(?:â[\x80-\xbf]|Â[\x80-\xbf]|Å[\x80-\xbf]|[\x80-\x9f]|\u00ad)"
)
_UNICODE_FORMAT_CONTROLS = frozenset({"\u200b", "\u200c", "\u200d", "\u200e", "\u200f"})


class OfficialReferenceError(ValueError):
    """Raised when official-reference provenance fails closed."""


@dataclass(frozen=True, slots=True)
class OfficialReferenceArtifact:
    path: str
    bytes: int
    rows: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OfficialReferenceParagraph:
    paragraph_id: str
    paragraph_index: int
    line_id: int
    section_name: str
    text: str
    word_count: int
    source_text_sha256: str
    text_sha256: str
    normalization_counts: Mapping[str, int]
    source_row_sha256: str
    paragraph_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OfficialActionBoundary:
    method: str
    line_id: int
    section_name: str
    text_sha256: str
    source_row_sha256: str
    controlled_match_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OfficialMeetingReference:
    schema_version: str
    meeting_date: str
    meeting_id: str
    split: str
    roster_line_number: int
    roster_row_sha256: str
    raw_minutes_csv: OfficialReferenceArtifact
    action_boundary: OfficialActionBoundary
    source_row_count: int
    pre_action_row_count: int
    excluded_pre_action_row_count: int
    post_boundary_row_count: int
    exclusion_counts: Mapping[str, int]
    paragraph_count: int
    word_count: int
    normalized_paragraph_count: int
    text_normalization_counts: Mapping[str, int]
    normalized_text_sha256: str
    paragraphs: tuple[OfficialReferenceParagraph, ...]
    meeting_reference_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "meeting_date": self.meeting_date,
            "meeting_id": self.meeting_id,
            "split": self.split,
            "roster_line_number": self.roster_line_number,
            "roster_row_sha256": self.roster_row_sha256,
            "raw_minutes_csv": self.raw_minutes_csv.to_dict(),
            "action_boundary": self.action_boundary.to_dict(),
            "source_row_count": self.source_row_count,
            "pre_action_row_count": self.pre_action_row_count,
            "excluded_pre_action_row_count": self.excluded_pre_action_row_count,
            "post_boundary_row_count": self.post_boundary_row_count,
            "exclusion_counts": dict(sorted(self.exclusion_counts.items())),
            "paragraph_count": self.paragraph_count,
            "word_count": self.word_count,
            "normalized_paragraph_count": self.normalized_paragraph_count,
            "text_normalization_counts": dict(
                sorted(self.text_normalization_counts.items())
            ),
            "normalized_text_sha256": self.normalized_text_sha256,
            "paragraphs": [paragraph.to_dict() for paragraph in self.paragraphs],
            "meeting_reference_sha256": self.meeting_reference_sha256,
        }


@dataclass(frozen=True, slots=True)
class OfficialReferenceBank:
    schema_version: str
    text_normalization_schema_version: str
    text_normalization_rules: tuple[str, ...]
    roster: OfficialReferenceArtifact
    meeting_counts: Mapping[str, int]
    boundary_method_counts: Mapping[str, int]
    meeting_date_digest: str
    meetings_digest: str
    normalized_paragraph_count: int
    text_normalization_counts: Mapping[str, int]
    meetings: tuple[OfficialMeetingReference, ...]
    reference_bank_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "text_normalization_schema_version": (
                self.text_normalization_schema_version
            ),
            "text_normalization_rules": list(self.text_normalization_rules),
            "roster": self.roster.to_dict(),
            "meeting_counts": dict(self.meeting_counts),
            "boundary_method_counts": dict(self.boundary_method_counts),
            "meeting_date_digest": self.meeting_date_digest,
            "meetings_digest": self.meetings_digest,
            "normalized_paragraph_count": self.normalized_paragraph_count,
            "text_normalization_counts": dict(
                sorted(self.text_normalization_counts.items())
            ),
            "meetings": [meeting.to_dict() for meeting in self.meetings],
            "reference_bank_sha256": self.reference_bank_sha256,
        }

    def reference_for_meeting(self, meeting_date: str) -> OfficialMeetingReference:
        matches = [row for row in self.meetings if row.meeting_date == meeting_date]
        if len(matches) != 1:
            raise OfficialReferenceError(
                f"meeting reference lookup is not unique: {meeting_date!r}"
            )
        return matches[0]


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OfficialReferenceError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_roster_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def _normalize_reference_text(value: str) -> tuple[str, Counter[str]]:
    """Repair only enumerated encoding artifacts and report every mutation."""

    counts: Counter[str] = Counter()

    def repair_utf8_latin1(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            repaired = token.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError) as exc:
            raise OfficialReferenceError(
                f"unrecognized UTF-8/Latin-1 mojibake sequence: {token!r}"
            ) from exc
        counts["utf8_latin1_mojibake_sequence_repaired"] += 1
        return repaired

    repaired = _UTF8_LATIN1_MOJIBAKE_RE.sub(repair_utf8_latin1, value)

    def repair_c1(match: re.Match[str]) -> str:
        token = match.group(0)
        try:
            replacement = bytes([ord(token)]).decode("cp1252")
        except UnicodeDecodeError as exc:
            raise OfficialReferenceError(
                f"unrecognized standalone C1 character: U+{ord(token):04X}"
            ) from exc
        counts["cp1252_c1_character_repaired"] += 1
        return replacement

    repaired = _C1_CONTROL_RE.sub(repair_c1, repaired)
    soft_hyphens = repaired.count("\u00ad")
    if soft_hyphens:
        counts["soft_hyphen_removed"] += soft_hyphens
        repaired = repaired.replace("\u00ad", "")
    format_controls = sum(char in _UNICODE_FORMAT_CONTROLS for char in repaired)
    if format_controls:
        counts["unicode_format_control_removed"] += format_controls
        repaired = "".join(
            char for char in repaired if char not in _UNICODE_FORMAT_CONTROLS
        )
    normalized = _normalize_text(repaired)
    if normalized != repaired:
        counts["nfkc_whitespace_normalized"] += 1
    return normalized, counts


def _strip_concatenated_section_heading(
    text: str, section_names: Sequence[str]
) -> tuple[str, bool]:
    candidates = [
        heading
        for heading in section_names
        if len(heading) >= 8
        and text.startswith(heading)
        and len(text) > len(heading)
        and text[len(heading)].isalnum()
    ]
    if not candidates:
        return text, False
    heading = max(candidates, key=len)
    body = text[len(heading) :].strip()
    if not body:
        raise OfficialReferenceError(
            "concatenated section-heading removal produced empty text"
        )
    return body, True


def _matching_concatenated_heading(
    text: str, section_names: Sequence[str]
) -> str | None:
    candidates = [
        heading
        for heading in section_names
        if len(heading) >= 8
        and text.startswith(heading)
        and len(text) > len(heading)
        and text[len(heading)].isalnum()
    ]
    return max(candidates, key=len) if candidates else None


def _control_document_block_line_ids(
    rows: Sequence[Mapping[str, str]], section_names: Sequence[str]
) -> set[int]:
    """Identify adopted resolution/directive blocks without mutating their text."""

    excluded: set[int] = set()
    active_section: str | None = None
    for row in rows:
        text, _ = _normalize_reference_text(row["raw_text"])
        section_name = _normalize_text(row["section_name"])
        embedded_heading = _matching_concatenated_heading(text, section_names)
        if active_section is not None and (
            section_name != active_section or embedded_heading is not None
        ):
            active_section = None
        if _CONTROL_DOCUMENT_START_RE.search(
            text
        ) or _ENACTED_ACTION_BLOCK_START_RE.search(text):
            active_section = section_name
        if active_section is not None:
            excluded.add(int(row["line_id"]))
    return excluded


def _normalize_enum(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"\[/?label\]", " ", value)
    value = value.replace("{", " ").replace("}", " ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _required_text(row: Mapping[str, Any], field: str, *, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise OfficialReferenceError(f"{label}.{field} must be non-empty text")
    return value


def _required_sha(row: Mapping[str, Any], field: str, *, label: str) -> str:
    value = _required_text(row, field, label=label)
    if not _SHA256_RE.fullmatch(value):
        raise OfficialReferenceError(f"{label}.{field} is not a lowercase SHA-256")
    return value


def _read_roster(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise OfficialReferenceError(f"missing sealed official roster: {path}")
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise OfficialReferenceError(
                    f"blank sealed-roster line at {line_number}"
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OfficialReferenceError(
                    f"invalid sealed-roster JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise OfficialReferenceError(
                    f"sealed-roster line {line_number} must be an object"
                )
            rows.append((line_number, row))
    return rows


def _read_raw_csv(path: Path, *, meeting_date: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise OfficialReferenceError(
            f"missing raw Minutes CSV for {meeting_date}: {path}"
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _EXPECTED_CSV_FIELDS:
            raise OfficialReferenceError(
                f"raw Minutes CSV schema drift for {meeting_date}: {reader.fieldnames!r}"
            )
        rows = list(reader)
    if not rows:
        raise OfficialReferenceError(f"raw Minutes CSV is empty for {meeting_date}")
    expected_line_ids = list(range(1, len(rows) + 1))
    observed: list[int] = []
    for record_number, row in enumerate(rows, 1):
        if None in row or any(not isinstance(value, str) for value in row.values()):
            raise OfficialReferenceError(
                f"malformed raw CSV record {record_number} for {meeting_date}"
            )
        try:
            line_id = int(row["line_id"])
        except ValueError as exc:
            raise OfficialReferenceError(
                f"non-integer line_id at record {record_number} for {meeting_date}"
            ) from exc
        observed.append(line_id)
    if observed != expected_line_ids:
        raise OfficialReferenceError(
            f"raw CSV line_id sequence drift for {meeting_date}"
        )
    return rows


def _source_meeting_split_map(
    source_rows: PreparedSourceDataset
    | Sequence[PreparedSourceRow]
    | Sequence[Mapping[str, Any]],
    *,
    expected_meeting_counts: Mapping[str, int],
) -> dict[str, str]:
    values: Sequence[Any]
    values = (
        source_rows.rows
        if isinstance(source_rows, PreparedSourceDataset)
        else source_rows
    )
    meeting_splits: dict[str, str] = {}
    for position, row in enumerate(values, 1):
        if isinstance(row, Mapping):
            meeting_date = row.get("meeting_date")
            split = row.get("split")
        else:
            meeting_date = getattr(row, "meeting_date", None)
            split = getattr(row, "split", None)
        if not isinstance(meeting_date, str) or not meeting_date:
            raise OfficialReferenceError(
                f"source row {position} has no valid meeting_date"
            )
        if split not in expected_meeting_counts:
            raise OfficialReferenceError(
                f"source row {position} has invalid split: {split!r}"
            )
        prior = meeting_splits.setdefault(meeting_date, split)
        if prior != split:
            raise OfficialReferenceError(
                f"source meeting appears in multiple splits: {meeting_date}"
            )
    observed_counts = Counter(meeting_splits.values())
    if dict(observed_counts) != dict(expected_meeting_counts):
        raise OfficialReferenceError(
            "source meeting split counts mismatch: "
            f"expected {dict(expected_meeting_counts)}, observed {dict(observed_counts)}"
        )
    return meeting_splits


def _header_only(section_name: str, text: str) -> bool:
    if not text:
        return True
    normalized_section = _normalize_text(section_name).casefold()
    normalized_text = text.casefold().strip(" .:;,-–—")
    if normalized_text == normalized_section.strip(" .:;,-–—"):
        return True
    if _DATE_HEADING_RE.fullmatch(text):
        return True
    if text.casefold() in {
        "minutes of the federal open market committee",
        "federal open market committee",
        "board of governors of the federal reserve system",
    }:
        return True
    if len(_WORD_RE.findall(text)) <= 20 and any(char.isalpha() for char in text):
        letters = "".join(char for char in text if char.isalpha())
        if letters and letters.isupper():
            return True
    return not _WORD_RE.search(text)


def _exclusion_reason(
    row: Mapping[str, str], normalized_text: str, *, meeting_date: str
) -> str | None:
    section_name = _normalize_text(row["section_name"])
    label_type = _normalize_enum(row["label_type"])
    label = _normalize_enum(row["label"])
    if label_type in _DISALLOWED_LABEL_VALUES:
        return f"label_type_{label_type.replace(' ', '_')}"
    if label in _DISALLOWED_LABEL_VALUES:
        return f"label_{label.replace(' ', '_')}"
    if _ADMIN_SECTION_RE.search(section_name):
        return "administrative_section"
    # In the 123 regular-format documents the parser assigns the section-heading
    # paragraph to the preceding section and switches section_name on the next
    # row.  The fused heading therefore identifies action text that appears one
    # row before the exact structural boundary.  Two pinned 2009 mixed-section
    # documents use the same words as a combined section heading followed by
    # genuine economic analysis; their later, paragraph-pinned boundary remains
    # authoritative.
    if (
        meeting_date not in PINNED_2009_ACTION_BOUNDARIES
        and _EMBEDDED_ACTION_HEADING_RE.search(normalized_text)
    ):
        return "embedded_action_heading"
    if _VOTE_SECTION_RE.search(section_name) or _VOTE_TEXT_RE.search(normalized_text):
        return "vote_content"
    if _ENACTED_ACTION_RE.search(normalized_text):
        return "enacted_action_content"
    if _DIRECTIVE_TEXT_RE.search(normalized_text) or _FORMAL_ACTION_TEXT_RE.search(
        normalized_text
    ):
        return "directive_content"
    if _ADMIN_TEXT_RE.search(normalized_text):
        return "administrative_content"
    if _PERSONNEL_TEXT_RE.search(normalized_text):
        return "personnel_content"
    if _LIST_LINE_RE.match(normalized_text):
        return "list_item"
    if _header_only(section_name, normalized_text):
        return "header_only"
    return None


def _boundary_for_meeting(
    meeting_date: str,
    rows: Sequence[Mapping[str, str]],
    *,
    exception_boundaries: Mapping[str, Mapping[str, Any]],
) -> tuple[int, OfficialActionBoundary]:
    if meeting_date in exception_boundaries:
        pin = exception_boundaries[meeting_date]
        line_id = pin.get("line_id")
        section_name = pin.get("section_name")
        expected_text_sha = pin.get("text_sha256")
        if (
            isinstance(line_id, bool)
            or not isinstance(line_id, int)
            or line_id < 1
            or not isinstance(section_name, str)
            or not section_name
            or not isinstance(expected_text_sha, str)
            or not _SHA256_RE.fullmatch(expected_text_sha)
        ):
            raise OfficialReferenceError(
                f"invalid exceptional boundary pin for {meeting_date}"
            )
        controlled_matches = [
            index
            for index, row in enumerate(rows)
            if _ACTION_START_RE.search(_normalize_text(row["raw_text"]))
        ]
        if len(controlled_matches) != 1:
            raise OfficialReferenceError(
                f"exceptional action-start match is not unique for {meeting_date}: "
                f"{len(controlled_matches)}"
            )
        index = controlled_matches[0]
        row = rows[index]
        observed_line_id = int(row["line_id"])
        text = _normalize_text(row["raw_text"])
        if observed_line_id != line_id:
            raise OfficialReferenceError(
                f"exceptional boundary line_id drift for {meeting_date}: "
                f"expected {line_id}, observed {observed_line_id}"
            )
        if row["section_name"] != section_name:
            raise OfficialReferenceError(
                f"exceptional boundary section drift for {meeting_date}"
            )
        if sha256_text(text) != expected_text_sha:
            raise OfficialReferenceError(
                f"exceptional boundary paragraph hash drift for {meeting_date}"
            )
        return index, OfficialActionBoundary(
            method=PINNED_2009_BOUNDARY_METHOD,
            line_id=line_id,
            section_name=section_name,
            text_sha256=expected_text_sha,
            source_row_sha256=sha256_json(row),
            controlled_match_count=1,
        )

    indices = [
        index
        for index, row in enumerate(rows)
        if _normalize_text(row["section_name"]) in EXACT_ACTION_SECTION_NAMES
    ]
    if not indices:
        raise OfficialReferenceError(
            f"missing exact Committee Policy Action(s) section for {meeting_date}"
        )
    first = indices[0]
    if indices != list(range(first, first + len(indices))):
        raise OfficialReferenceError(
            f"non-contiguous exact action section for {meeting_date}"
        )
    row = rows[first]
    text = _normalize_text(row["raw_text"])
    return first, OfficialActionBoundary(
        method=EXACT_SECTION_BOUNDARY_METHOD,
        line_id=int(row["line_id"]),
        section_name=_normalize_text(row["section_name"]),
        text_sha256=sha256_text(text),
        source_row_sha256=sha256_json(row),
        controlled_match_count=len(indices),
    )


def _meeting_digest_payload(meeting: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in meeting.items()
        if key != "meeting_reference_sha256"
    }


def _bank_digest_payload(bank: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in bank.items() if key != "reference_bank_sha256"}


def build_official_reference_bank(
    source_rows: PreparedSourceDataset
    | Sequence[PreparedSourceRow]
    | Sequence[Mapping[str, Any]],
    roster_path: str | Path = DEFAULT_OFFICIAL_ROSTER_PATH,
    *,
    expected_roster_sha256: str = EXPECTED_OFFICIAL_ROSTER_SHA256,
    expected_meeting_counts: Mapping[str, int] = EXPECTED_MEETING_COUNTS,
    exception_boundaries: Mapping[
        str, Mapping[str, Any]
    ] = PINNED_2009_ACTION_BOUNDARIES,
    expected_boundary_method_counts: Mapping[
        str, int
    ] = EXPECTED_BOUNDARY_METHOD_COUNTS,
) -> OfficialReferenceBank:
    """Build the frozen corresponding-meeting pre-action reference bank."""

    if not _SHA256_RE.fullmatch(expected_roster_sha256):
        raise OfficialReferenceError("expected roster SHA-256 is invalid")
    if set(expected_meeting_counts) != {"train", "validation", "test"}:
        raise OfficialReferenceError(
            "expected_meeting_counts must define train/validation/test"
        )
    meeting_splits = _source_meeting_split_map(
        source_rows, expected_meeting_counts=expected_meeting_counts
    )
    roster_path = Path(roster_path).resolve()
    roster_rows = _read_roster(roster_path)
    observed_roster_sha = sha256_file(roster_path)
    if observed_roster_sha != expected_roster_sha256:
        raise OfficialReferenceError(
            "sealed official roster hash mismatch: "
            f"expected {expected_roster_sha256}, observed {observed_roster_sha}"
        )
    roster_binding = OfficialReferenceArtifact(
        path=_portable_path(roster_path),
        bytes=roster_path.stat().st_size,
        rows=len(roster_rows),
        sha256=observed_roster_sha,
    )
    if len(roster_rows) != len(meeting_splits):
        raise OfficialReferenceError(
            f"sealed roster/source meeting count mismatch: "
            f"{len(roster_rows)} != {len(meeting_splits)}"
        )

    roster_dates: set[str] = set()
    roster_ids: set[str] = set()
    raw_paths: set[Path] = set()
    meetings: list[OfficialMeetingReference] = []
    for roster_line, roster_row in roster_rows:
        label = f"sealed roster line {roster_line}"
        if roster_row.get("schema_version") != EXPECTED_ROSTER_SCHEMA_VERSION:
            raise OfficialReferenceError(f"{label} schema version drift")
        meeting_date = _required_text(roster_row, "meeting_end_date", label=label)
        meeting_id = _required_text(roster_row, "meeting_id", label=label)
        split = _required_text(roster_row, "source_split", label=label)
        if meeting_date in roster_dates or meeting_id in roster_ids:
            raise OfficialReferenceError(f"duplicate meeting identity in {label}")
        roster_dates.add(meeting_date)
        roster_ids.add(meeting_id)
        expected_split = meeting_splits.get(meeting_date)
        if expected_split is None:
            raise OfficialReferenceError(
                f"{label} meeting_end_date is absent from chk-1 source: {meeting_date}"
            )
        if split != expected_split:
            raise OfficialReferenceError(
                f"{label} source split disagrees with chk-1 source"
            )
        raw_path_text = _required_text(roster_row, "raw_minutes_csv", label=label)
        raw_path = _resolve_roster_path(raw_path_text)
        if raw_path in raw_paths:
            raise OfficialReferenceError(f"duplicate raw CSV binding in {label}")
        raw_paths.add(raw_path)
        expected_raw_sha = _required_sha(
            roster_row, "raw_minutes_csv_sha256", label=label
        )
        observed_raw_sha = sha256_file(raw_path) if raw_path.is_file() else ""
        if observed_raw_sha != expected_raw_sha:
            raise OfficialReferenceError(
                f"raw Minutes CSV hash mismatch for {meeting_date}: "
                f"expected {expected_raw_sha}, observed {observed_raw_sha or 'MISSING'}"
            )
        raw_rows = _read_raw_csv(raw_path, meeting_date=meeting_date)
        source_section_names = tuple(
            sorted(
                {_normalize_text(row["section_name"]) for row in raw_rows},
                key=lambda value: (-len(value), value),
            )
        )
        raw_binding = OfficialReferenceArtifact(
            path=_portable_path(raw_path),
            bytes=raw_path.stat().st_size,
            rows=len(raw_rows),
            sha256=observed_raw_sha,
        )
        boundary_index, boundary = _boundary_for_meeting(
            meeting_date,
            raw_rows,
            exception_boundaries=exception_boundaries,
        )
        if boundary_index < 1:
            raise OfficialReferenceError(
                f"action boundary leaves no pre-action rows for {meeting_date}"
            )

        paragraphs: list[OfficialReferenceParagraph] = []
        exclusions: Counter[str] = Counter()
        control_document_lines = _control_document_block_line_ids(
            raw_rows[:boundary_index], source_section_names
        )
        for row in raw_rows[:boundary_index]:
            normalized_text, normalization_counts = _normalize_reference_text(
                row["raw_text"]
            )
            reason = _exclusion_reason(row, normalized_text, meeting_date=meeting_date)
            if reason is not None:
                exclusions[reason] += 1
                continue
            if int(row["line_id"]) in control_document_lines:
                exclusions["control_document_block"] += 1
                continue
            normalized_text, removed_heading = _strip_concatenated_section_heading(
                normalized_text, source_section_names
            )
            if removed_heading:
                normalization_counts["concatenated_section_heading_removed"] += 1
            if _ENCODING_ARTIFACT_RE.search(normalized_text):
                raise OfficialReferenceError(
                    f"unresolved text-encoding artifact for {meeting_date} "
                    f"line {row['line_id']}"
                )
            if any(
                normalized_text.startswith(heading)
                and len(normalized_text) > len(heading)
                and normalized_text[len(heading)].isalnum()
                for heading in source_section_names
                if len(heading) >= 8
            ):
                raise OfficialReferenceError(
                    f"unresolved concatenated section heading for {meeting_date} "
                    f"line {row['line_id']}"
                )
            line_id = int(row["line_id"])
            section_name = _normalize_text(row["section_name"])
            text_sha = sha256_text(normalized_text)
            paragraph_payload: dict[str, Any] = {
                "paragraph_id": (
                    f"official-{meeting_date.replace('-', '')}-line-{line_id}"
                ),
                "paragraph_index": len(paragraphs) + 1,
                "line_id": line_id,
                "section_name": section_name,
                "text": normalized_text,
                "word_count": len(_WORD_RE.findall(normalized_text)),
                "source_text_sha256": sha256_text(row["raw_text"]),
                "text_sha256": text_sha,
                "normalization_counts": dict(sorted(normalization_counts.items())),
                "source_row_sha256": sha256_json(row),
            }
            paragraph_payload["paragraph_sha256"] = sha256_json(paragraph_payload)
            paragraphs.append(OfficialReferenceParagraph(**paragraph_payload))
        if not paragraphs:
            raise OfficialReferenceError(
                f"no substantive pre-action paragraphs remain for {meeting_date}"
            )

        meeting_payload: dict[str, Any] = {
            "schema_version": OFFICIAL_REFERENCE_SCHEMA_VERSION,
            "meeting_date": meeting_date,
            "meeting_id": meeting_id,
            "split": split,
            "roster_line_number": roster_line,
            "roster_row_sha256": sha256_json(roster_row),
            "raw_minutes_csv": raw_binding.to_dict(),
            "action_boundary": boundary.to_dict(),
            "source_row_count": len(raw_rows),
            "pre_action_row_count": boundary_index,
            "excluded_pre_action_row_count": sum(exclusions.values()),
            "post_boundary_row_count": len(raw_rows) - boundary_index,
            "exclusion_counts": dict(sorted(exclusions.items())),
            "paragraph_count": len(paragraphs),
            "word_count": sum(row.word_count for row in paragraphs),
            "normalized_paragraph_count": sum(
                bool(row.normalization_counts) for row in paragraphs
            ),
            "text_normalization_counts": dict(
                sorted(
                    sum(
                        (Counter(row.normalization_counts) for row in paragraphs),
                        Counter(),
                    ).items()
                )
            ),
            "normalized_text_sha256": sha256_text(
                "\n\n".join(row.text for row in paragraphs)
            ),
            "paragraphs": [row.to_dict() for row in paragraphs],
        }
        meeting_payload["meeting_reference_sha256"] = sha256_json(meeting_payload)
        meetings.append(
            OfficialMeetingReference(
                schema_version=meeting_payload["schema_version"],
                meeting_date=meeting_payload["meeting_date"],
                meeting_id=meeting_payload["meeting_id"],
                split=meeting_payload["split"],
                roster_line_number=meeting_payload["roster_line_number"],
                roster_row_sha256=meeting_payload["roster_row_sha256"],
                raw_minutes_csv=raw_binding,
                action_boundary=boundary,
                source_row_count=meeting_payload["source_row_count"],
                pre_action_row_count=meeting_payload["pre_action_row_count"],
                excluded_pre_action_row_count=meeting_payload[
                    "excluded_pre_action_row_count"
                ],
                post_boundary_row_count=meeting_payload["post_boundary_row_count"],
                exclusion_counts=meeting_payload["exclusion_counts"],
                paragraph_count=meeting_payload["paragraph_count"],
                word_count=meeting_payload["word_count"],
                normalized_paragraph_count=meeting_payload[
                    "normalized_paragraph_count"
                ],
                text_normalization_counts=meeting_payload["text_normalization_counts"],
                normalized_text_sha256=meeting_payload["normalized_text_sha256"],
                paragraphs=tuple(paragraphs),
                meeting_reference_sha256=meeting_payload["meeting_reference_sha256"],
            )
        )

    if roster_dates != set(meeting_splits):
        raise OfficialReferenceError(
            "sealed roster does not exactly cover source meetings"
        )
    if [meeting.meeting_date for meeting in meetings] != sorted(roster_dates):
        raise OfficialReferenceError(
            "sealed roster/reference meetings must be ordered by meeting_end_date"
        )
    meeting_counts = Counter(meeting.split for meeting in meetings)
    if dict(meeting_counts) != dict(expected_meeting_counts):
        raise OfficialReferenceError("reference meeting split counts drift")
    method_counts = Counter(meeting.action_boundary.method for meeting in meetings)
    if dict(method_counts) != dict(expected_boundary_method_counts):
        raise OfficialReferenceError(
            "action boundary method counts mismatch: "
            f"expected {dict(expected_boundary_method_counts)}, "
            f"observed {dict(method_counts)}"
        )

    bank_payload: dict[str, Any] = {
        "schema_version": OFFICIAL_REFERENCE_SCHEMA_VERSION,
        "text_normalization_schema_version": TEXT_NORMALIZATION_SCHEMA_VERSION,
        "text_normalization_rules": list(TEXT_NORMALIZATION_RULES),
        "roster": roster_binding.to_dict(),
        "meeting_counts": dict(expected_meeting_counts),
        "boundary_method_counts": dict(expected_boundary_method_counts),
        "meeting_date_digest": sha256_json(
            [meeting.meeting_date for meeting in meetings]
        ),
        "meetings_digest": sha256_json(
            [meeting.meeting_reference_sha256 for meeting in meetings]
        ),
        "normalized_paragraph_count": sum(
            meeting.normalized_paragraph_count for meeting in meetings
        ),
        "text_normalization_counts": dict(
            sorted(
                sum(
                    (
                        Counter(meeting.text_normalization_counts)
                        for meeting in meetings
                    ),
                    Counter(),
                ).items()
            )
        ),
        "meetings": [meeting.to_dict() for meeting in meetings],
    }
    bank_payload["reference_bank_sha256"] = sha256_json(bank_payload)
    bank = OfficialReferenceBank(
        schema_version=bank_payload["schema_version"],
        text_normalization_schema_version=bank_payload[
            "text_normalization_schema_version"
        ],
        text_normalization_rules=tuple(bank_payload["text_normalization_rules"]),
        roster=roster_binding,
        meeting_counts=bank_payload["meeting_counts"],
        boundary_method_counts=bank_payload["boundary_method_counts"],
        meeting_date_digest=bank_payload["meeting_date_digest"],
        meetings_digest=bank_payload["meetings_digest"],
        normalized_paragraph_count=bank_payload["normalized_paragraph_count"],
        text_normalization_counts=bank_payload["text_normalization_counts"],
        meetings=tuple(meetings),
        reference_bank_sha256=bank_payload["reference_bank_sha256"],
    )
    _verify_embedded_hashes(bank)
    return bank


def verify_official_reference_bank(
    bank: OfficialReferenceBank,
    source_rows: PreparedSourceDataset
    | Sequence[PreparedSourceRow]
    | Sequence[Mapping[str, Any]],
    roster_path: str | Path = DEFAULT_OFFICIAL_ROSTER_PATH,
    *,
    expected_roster_sha256: str = EXPECTED_OFFICIAL_ROSTER_SHA256,
    expected_meeting_counts: Mapping[str, int] = EXPECTED_MEETING_COUNTS,
    exception_boundaries: Mapping[
        str, Mapping[str, Any]
    ] = PINNED_2009_ACTION_BOUNDARIES,
    expected_boundary_method_counts: Mapping[
        str, int
    ] = EXPECTED_BOUNDARY_METHOD_COUNTS,
) -> None:
    """Independently rebuild and exactly compare a reference bank."""

    if not isinstance(bank, OfficialReferenceBank):
        raise OfficialReferenceError("reference bank must be OfficialReferenceBank")
    rebuilt = build_official_reference_bank(
        source_rows,
        roster_path,
        expected_roster_sha256=expected_roster_sha256,
        expected_meeting_counts=expected_meeting_counts,
        exception_boundaries=exception_boundaries,
        expected_boundary_method_counts=expected_boundary_method_counts,
    )
    if bank.to_dict() != rebuilt.to_dict():
        raise OfficialReferenceError(
            "official reference bank does not match independently rebuilt sources"
        )


def serialize_official_reference_bank(bank: OfficialReferenceBank) -> bytes:
    """Return deterministic JSONL: one metadata record plus one row per meeting."""

    payload = bank.to_dict()
    metadata = {
        "record_type": "metadata",
        "serialization_version": OFFICIAL_REFERENCE_SERIALIZATION_VERSION,
        **{key: value for key, value in payload.items() if key != "meetings"},
    }
    records = [metadata]
    records.extend(
        {"record_type": "meeting_reference", **meeting.to_dict()}
        for meeting in bank.meetings
    )
    return b"".join(_canonical_json_bytes(record) + b"\n" for record in records)


def _artifact_from_dict(value: Mapping[str, Any]) -> OfficialReferenceArtifact:
    return OfficialReferenceArtifact(
        path=value["path"],
        bytes=value["bytes"],
        rows=value["rows"],
        sha256=value["sha256"],
    )


def _paragraph_from_dict(value: Mapping[str, Any]) -> OfficialReferenceParagraph:
    return OfficialReferenceParagraph(**value)


def _meeting_from_dict(value: Mapping[str, Any]) -> OfficialMeetingReference:
    return OfficialMeetingReference(
        schema_version=value["schema_version"],
        meeting_date=value["meeting_date"],
        meeting_id=value["meeting_id"],
        split=value["split"],
        roster_line_number=value["roster_line_number"],
        roster_row_sha256=value["roster_row_sha256"],
        raw_minutes_csv=_artifact_from_dict(value["raw_minutes_csv"]),
        action_boundary=OfficialActionBoundary(**value["action_boundary"]),
        source_row_count=value["source_row_count"],
        pre_action_row_count=value["pre_action_row_count"],
        excluded_pre_action_row_count=value["excluded_pre_action_row_count"],
        post_boundary_row_count=value["post_boundary_row_count"],
        exclusion_counts=value["exclusion_counts"],
        paragraph_count=value["paragraph_count"],
        word_count=value["word_count"],
        normalized_paragraph_count=value["normalized_paragraph_count"],
        text_normalization_counts=value["text_normalization_counts"],
        normalized_text_sha256=value["normalized_text_sha256"],
        paragraphs=tuple(_paragraph_from_dict(row) for row in value["paragraphs"]),
        meeting_reference_sha256=value["meeting_reference_sha256"],
    )


def deserialize_official_reference_bank(data: bytes | str) -> OfficialReferenceBank:
    """Parse deterministic JSONL and verify every embedded structural hash."""

    if isinstance(data, bytes):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OfficialReferenceError("reference JSONL is not UTF-8") from exc
    elif isinstance(data, str):
        text = data
    else:
        raise OfficialReferenceError("reference JSONL must be bytes or text")
    if not text.endswith("\n"):
        raise OfficialReferenceError("reference JSONL must end with one newline")
    lines = text.splitlines()
    if not lines:
        raise OfficialReferenceError("reference JSONL is empty")
    try:
        records = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise OfficialReferenceError(f"invalid reference JSONL: {exc}") from exc
    if not all(isinstance(record, dict) for record in records):
        raise OfficialReferenceError("every reference JSONL row must be an object")
    metadata = records[0]
    if (
        metadata.get("record_type") != "metadata"
        or metadata.get("serialization_version")
        != OFFICIAL_REFERENCE_SERIALIZATION_VERSION
    ):
        raise OfficialReferenceError("reference JSONL metadata contract mismatch")
    meeting_records = records[1:]
    if not meeting_records or any(
        record.get("record_type") != "meeting_reference" for record in meeting_records
    ):
        raise OfficialReferenceError("reference JSONL meeting records are invalid")
    meetings = tuple(
        _meeting_from_dict(
            {key: value for key, value in record.items() if key != "record_type"}
        )
        for record in meeting_records
    )
    bank = OfficialReferenceBank(
        schema_version=metadata["schema_version"],
        text_normalization_schema_version=metadata["text_normalization_schema_version"],
        text_normalization_rules=tuple(metadata["text_normalization_rules"]),
        roster=_artifact_from_dict(metadata["roster"]),
        meeting_counts=metadata["meeting_counts"],
        boundary_method_counts=metadata["boundary_method_counts"],
        meeting_date_digest=metadata["meeting_date_digest"],
        meetings_digest=metadata["meetings_digest"],
        normalized_paragraph_count=metadata["normalized_paragraph_count"],
        text_normalization_counts=metadata["text_normalization_counts"],
        meetings=meetings,
        reference_bank_sha256=metadata["reference_bank_sha256"],
    )
    _verify_embedded_hashes(bank)
    canonical_input = data.encode("utf-8") if isinstance(data, str) else data
    if serialize_official_reference_bank(bank) != canonical_input:
        raise OfficialReferenceError("reference JSONL is not canonical")
    return bank


def _verify_embedded_hashes(bank: OfficialReferenceBank) -> None:
    if bank.schema_version != OFFICIAL_REFERENCE_SCHEMA_VERSION:
        raise OfficialReferenceError("reference bank schema version mismatch")
    if bank.text_normalization_schema_version != TEXT_NORMALIZATION_SCHEMA_VERSION:
        raise OfficialReferenceError("text normalization schema version mismatch")
    if bank.text_normalization_rules != TEXT_NORMALIZATION_RULES:
        raise OfficialReferenceError("text normalization rule contract mismatch")
    if len(bank.meetings) != sum(bank.meeting_counts.values()):
        raise OfficialReferenceError("embedded reference meeting count mismatch")
    if Counter(row.split for row in bank.meetings) != Counter(bank.meeting_counts):
        raise OfficialReferenceError("embedded reference split counts mismatch")
    if Counter(row.action_boundary.method for row in bank.meetings) != Counter(
        bank.boundary_method_counts
    ):
        raise OfficialReferenceError("embedded boundary method counts mismatch")
    if [row.meeting_date for row in bank.meetings] != sorted(
        row.meeting_date for row in bank.meetings
    ):
        raise OfficialReferenceError("embedded reference meeting order mismatch")
    for meeting in bank.meetings:
        if meeting.schema_version != OFFICIAL_REFERENCE_SCHEMA_VERSION:
            raise OfficialReferenceError("embedded meeting schema version mismatch")
        if meeting.paragraph_count != len(meeting.paragraphs):
            raise OfficialReferenceError("embedded paragraph count mismatch")
        if meeting.word_count != sum(row.word_count for row in meeting.paragraphs):
            raise OfficialReferenceError("embedded meeting word count mismatch")
        expected_normalized_count = sum(
            bool(row.normalization_counts) for row in meeting.paragraphs
        )
        if meeting.normalized_paragraph_count != expected_normalized_count:
            raise OfficialReferenceError("embedded normalized paragraph count mismatch")
        expected_normalization_counts = sum(
            (Counter(row.normalization_counts) for row in meeting.paragraphs),
            Counter(),
        )
        if Counter(meeting.text_normalization_counts) != expected_normalization_counts:
            raise OfficialReferenceError(
                "embedded meeting text normalization counts mismatch"
            )
        if meeting.pre_action_row_count != (
            meeting.paragraph_count + meeting.excluded_pre_action_row_count
        ):
            raise OfficialReferenceError("embedded pre-action partition mismatch")
        if meeting.source_row_count != (
            meeting.pre_action_row_count + meeting.post_boundary_row_count
        ):
            raise OfficialReferenceError("embedded source-row partition mismatch")
        if any(
            paragraph.line_id >= meeting.action_boundary.line_id
            for paragraph in meeting.paragraphs
        ):
            raise OfficialReferenceError("embedded paragraph crosses action boundary")
        for expected_index, paragraph in enumerate(meeting.paragraphs, 1):
            payload = paragraph.to_dict()
            observed_sha = payload.pop("paragraph_sha256")
            if expected_index != paragraph.paragraph_index:
                raise OfficialReferenceError("embedded paragraph order mismatch")
            if sha256_text(paragraph.text) != paragraph.text_sha256:
                raise OfficialReferenceError("embedded paragraph text hash mismatch")
            if not _SHA256_RE.fullmatch(paragraph.source_text_sha256):
                raise OfficialReferenceError("embedded source text hash is invalid")
            if set(paragraph.normalization_counts) - set(TEXT_NORMALIZATION_RULES):
                raise OfficialReferenceError("unknown embedded normalization rule")
            if any(
                isinstance(count, bool) or not isinstance(count, int) or count < 1
                for count in paragraph.normalization_counts.values()
            ):
                raise OfficialReferenceError("invalid embedded normalization count")
            if _ENCODING_ARTIFACT_RE.search(paragraph.text):
                raise OfficialReferenceError("embedded text encoding artifact remains")
            if len(_WORD_RE.findall(paragraph.text)) != paragraph.word_count:
                raise OfficialReferenceError("embedded paragraph word count mismatch")
            if sha256_json(payload) != observed_sha:
                raise OfficialReferenceError("embedded paragraph hash mismatch")
        meeting_payload = meeting.to_dict()
        observed_meeting_sha = meeting_payload.pop("meeting_reference_sha256")
        if sha256_json(meeting_payload) != observed_meeting_sha:
            raise OfficialReferenceError("embedded meeting reference hash mismatch")
    if bank.meeting_date_digest != sha256_json(
        [meeting.meeting_date for meeting in bank.meetings]
    ):
        raise OfficialReferenceError("embedded meeting-date digest mismatch")
    if bank.meetings_digest != sha256_json(
        [meeting.meeting_reference_sha256 for meeting in bank.meetings]
    ):
        raise OfficialReferenceError("embedded meetings digest mismatch")
    if bank.normalized_paragraph_count != sum(
        meeting.normalized_paragraph_count for meeting in bank.meetings
    ):
        raise OfficialReferenceError("embedded bank normalized count mismatch")
    expected_bank_normalizations = sum(
        (Counter(meeting.text_normalization_counts) for meeting in bank.meetings),
        Counter(),
    )
    if Counter(bank.text_normalization_counts) != expected_bank_normalizations:
        raise OfficialReferenceError("embedded bank text normalization counts mismatch")
    payload = bank.to_dict()
    observed_bank_sha = payload.pop("reference_bank_sha256")
    if sha256_json(payload) != observed_bank_sha:
        raise OfficialReferenceError("embedded reference-bank hash mismatch")


__all__ = [
    "DEFAULT_OFFICIAL_ROSTER_PATH",
    "EXACT_ACTION_SECTION_NAMES",
    "EXACT_SECTION_BOUNDARY_METHOD",
    "EXPECTED_BOUNDARY_METHOD_COUNTS",
    "EXPECTED_OFFICIAL_ROSTER_SHA256",
    "EXPECTED_ROSTER_SCHEMA_VERSION",
    "OFFICIAL_REFERENCE_SCHEMA_VERSION",
    "OFFICIAL_REFERENCE_SERIALIZATION_VERSION",
    "OfficialActionBoundary",
    "OfficialMeetingReference",
    "OfficialReferenceArtifact",
    "OfficialReferenceBank",
    "OfficialReferenceError",
    "OfficialReferenceParagraph",
    "PINNED_2009_ACTION_BOUNDARIES",
    "PINNED_2009_BOUNDARY_METHOD",
    "TEXT_NORMALIZATION_RULES",
    "TEXT_NORMALIZATION_SCHEMA_VERSION",
    "build_official_reference_bank",
    "deserialize_official_reference_bank",
    "serialize_official_reference_bank",
    "verify_official_reference_bank",
]
