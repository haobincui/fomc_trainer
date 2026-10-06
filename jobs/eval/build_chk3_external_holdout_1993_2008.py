"""Build the 1993--2008 all-regular-meeting CHK3 external holdout.

This builder intentionally creates a new evaluation-only lineage.  It does not
modify or extend either the historical CHK4 decision supplement or the CHK3
training release.  The official FOMC calendar supplies meeting identity and
Minutes inventory; model-facing evidence is acquired at the start of each
meeting minus one calendar day.

The source pipeline uses its historical ``train`` transport partition only to
enable sparse ALFRED acquisition.  The published release is fail-closed as
evaluation-only and is forbidden for training or checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from jobs.generation.prepare_chk4_supplement import (
    _supplement_alfred_http_get,
    compress_indicator_row,
)
from jobs.main.checkpoint_provenance import fingerprint_tokenizer_payload
from jobs.main.fetch_loo_source_snapshots import load_source_registry
from jobs.retrain_v2.chk1.source_pipeline import (
    create_source_plan,
    materialize_source_plan,
    validate_source_handoff,
)
from open_r1.provenance import sha256_file, sha256_text
from open_r1.validator.loo_generation_spec import (
    ManifestIntegrityError,
    seal_manifest,
    validate_manifest_integrity,
)
from open_r1.validator.loo_ledger import _load_registry, _load_roster


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk3/"
    "chk3_minutes_external_holdout_1993_2008_all_regular_v1_candidate"
)
DEFAULT_RELEASE_ROOT = REPO_ROOT / (
    "dataset/processed/retrain_v2/"
    "chk3_minutes_external_holdout_1993_2008_all_regular_v1"
)
DEFAULT_BASE_ROSTER = REPO_ROOT / "configs/main/leave_one_out_roster.json"
DEFAULT_BASE_REGISTRY = REPO_ROOT / "configs/main/loo_indicator_sources.json"
DEFAULT_TOKENIZER = REPO_ROOT / (
    "output/training/retrain_v2/"
    "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
)
CURRENT_CHK3_RELEASE = REPO_ROOT / (
    "dataset/processed/retrain_v2/chk3_minutes_clean_v3_20260805"
)
PRIOR_SUPPLEMENT_ADMITTED = REPO_ROOT / (
    "output/data/retrain_v2/chk4/decision_supplement_1993_2008_v1/"
    "manifests/admitted.jsonl"
)

RELEASE_ID = "chk3_minutes_external_holdout_1993_2008_all_regular_v1"
ROSTER_SCHEMA = "chk3-external-official-meeting-roster-v1"
SAMPLE_SCHEMA = "chk3-external-analysis-to-minutes-eval-row-v1"
EXPECTED_MEETINGS = 128
EXPECTED_PER_YEAR = 8
YEARS = tuple(range(1993, 2009))
FED_ARCHIVE_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"

CORE_TOPICS = (
    "Consumer-Price-Index-(CPI)",
    "GDP-Growth",
    "Government-Purchases",
    "Housing-Starts",
    "Industrial-Production",
    "Labour-Market",
    "Money-Supply",
    "Unemployment-Rate",
)

# The scheduled regular-meeting end dates from the official yearly archive.
# The 2003-09-15 special meeting is deliberately excluded from this panel.
EXPECTED_END_DATES = {
    1993: "02-03 03-23 05-18 07-07 08-17 09-21 11-16 12-21",
    1994: "02-04 03-22 05-17 07-06 08-16 09-27 11-15 12-20",
    1995: "02-01 03-28 05-23 07-06 08-22 09-26 11-15 12-19",
    1996: "01-31 03-26 05-21 07-03 08-20 09-24 11-13 12-17",
    1997: "02-05 03-25 05-20 07-02 08-19 09-30 11-12 12-16",
    1998: "02-04 03-31 05-19 07-01 08-18 09-29 11-17 12-22",
    1999: "02-03 03-30 05-18 06-30 08-24 10-05 11-16 12-21",
    2000: "02-02 03-21 05-16 06-28 08-22 10-03 11-15 12-19",
    2001: "01-31 03-20 05-15 06-27 08-21 10-02 11-06 12-11",
    2002: "01-30 03-19 05-07 06-26 08-13 09-24 11-06 12-10",
    2003: "01-29 03-18 05-06 06-25 08-12 09-16 10-28 12-09",
    2004: "01-28 03-16 05-04 06-30 08-10 09-21 11-10 12-14",
    2005: "02-02 03-22 05-03 06-30 08-09 09-20 11-01 12-13",
    2006: "01-31 03-28 05-10 06-29 08-08 09-20 10-25 12-12",
    2007: "01-31 03-21 05-09 06-28 08-07 09-18 10-31 12-11",
    2008: "01-30 03-18 04-30 06-25 08-05 09-16 10-29 12-16",
}
EXPECTED_END_DATES = {
    year: tuple(f"{year}-{value}" for value in values.split())
    for year, values in EXPECTED_END_DATES.items()
}

MONTHS = {
    name: number
    for number, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        1,
    )
}
HEADING_RE = re.compile(r"^(.*?) Meeting - (\d{4})$")
RANGE_SAME_MONTH_RE = re.compile(r"^([A-Za-z]+) (\d{1,2})-(\d{1,2})$")
RANGE_CROSS_MONTH_RE = re.compile(
    r"^([A-Za-z]+) (\d{1,2})-([A-Za-z]+) (\d{1,2})$"
)
SINGLE_RE = re.compile(r"^([A-Za-z]+) (\d{1,2})$")
NUMBER_RE = re.compile(r"(?<![\w.])[+-]?(?:\d[\d,]*)(?:\.\d+)?(?![\w.])")


class BuildError(RuntimeError):
    """The release cannot be built without weakening a frozen contract."""


@dataclass(frozen=True)
class Meeting:
    meeting_id: str
    year: int
    heading: str
    start_date: str
    end_date: str
    evidence_cutoff: str
    archive_url: str
    minutes_url: str


class _ArchiveParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, Any]] = []
        self._row: dict[str, Any] | None = None
        self._in_heading = False
        self._in_link = False
        self._link_href = ""
        self._link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "h5":
            self._row = {"heading": [], "links": []}
            self.rows.append(self._row)
            self._in_heading = True
        elif tag == "a" and self._row is not None:
            self._in_link = True
            self._link_href = dict(attrs).get("href") or ""
            self._link_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "h5":
            self._in_heading = False
        elif tag == "a" and self._in_link and self._row is not None:
            text = " ".join("".join(self._link_text).split())
            self._row["links"].append((text, self._link_href))
            self._in_link = False

    def handle_data(self, data: str) -> None:
        if self._in_heading and self._row is not None:
            self._row["heading"].append(data)
        if self._in_link:
            self._link_text.append(data)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable_bytes(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise BuildError(f"immutable artifact drift: {path}")
        return
    _atomic_write_bytes(path, content)


def _write_json(path: Path, value: Any, *, immutable: bool = True) -> None:
    content = _pretty_json(value).encode("utf-8")
    if immutable:
        _write_immutable_bytes(path, content)
    else:
        _atomic_write_bytes(path, content)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(_canonical_json(dict(row)) + "\n" for row in rows).encode()
    _write_immutable_bytes(path, content)


def _parse_heading(heading: str) -> tuple[str, str]:
    match = HEADING_RE.fullmatch(heading)
    if match is None:
        raise BuildError(f"unexpected official meeting heading: {heading!r}")
    body, raw_year = match.groups()
    year = int(raw_year)
    same = RANGE_SAME_MONTH_RE.fullmatch(body)
    cross = RANGE_CROSS_MONTH_RE.fullmatch(body)
    single = SINGLE_RE.fullmatch(body)
    if same:
        month, start_day, end_day = same.groups()
        start = date(year, MONTHS[month], int(start_day))
        end = date(year, MONTHS[month], int(end_day))
    elif cross:
        start_month, start_day, end_month, end_day = cross.groups()
        start = date(year, MONTHS[start_month], int(start_day))
        end = date(year, MONTHS[end_month], int(end_day))
    elif single:
        month, day = single.groups()
        start = end = date(year, MONTHS[month], int(day))
    else:
        raise BuildError(f"unsupported official meeting heading: {heading!r}")
    return start.isoformat(), end.isoformat()


def _minutes_url(
    *, archive_url: str, end_date: str, links: Sequence[tuple[str, str]]
) -> str:
    token = end_date.replace("-", "")
    candidates: list[str] = []
    for text, href in links:
        absolute = urljoin(archive_url, href)
        lower = absolute.lower()
        if token in lower and (
            "minutes" in lower
            or re.search(rf"/fomc{token}\.htm$", lower) is not None
        ):
            if lower.endswith((".htm", ".html")):
                candidates.append(absolute.replace("http://", "https://", 1))
        elif text.strip().lower() == "minutes" and lower.endswith((".htm", ".html")):
            candidates.append(absolute.replace("http://", "https://", 1))
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise BuildError(
            f"official Minutes link is ambiguous for {end_date}: {unique}"
        )
    return unique[0]


def _http_get(url: str, *, timeout: float = 60.0) -> tuple[bytes, str]:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "fomc-trainer-external-holdout/1.0"},
    )
    response.raise_for_status()
    if not response.content:
        raise BuildError(f"empty official response: {url}")
    return response.content, str(response.url)


def fetch_official_roster(output_root: Path) -> list[Meeting]:
    meetings: list[Meeting] = []
    page_records: list[dict[str, Any]] = []
    for year in YEARS:
        archive_url = FED_ARCHIVE_URL.format(year=year)
        body, resolved_url = _http_get(archive_url)
        page_path = output_root / f"sources/official_calendar/pages/{year}.html"
        _write_immutable_bytes(page_path, body)
        parser = _ArchiveParser()
        parser.feed(body.decode("utf-8", errors="replace"))
        expected = set(EXPECTED_END_DATES[year])
        observed: dict[str, Meeting] = {}
        for raw in parser.rows:
            heading = " ".join("".join(raw["heading"]).split())
            if " Meeting - " not in heading or "Conference Call" in heading:
                continue
            start_date, end_date = _parse_heading(heading)
            if end_date not in expected:
                continue
            if end_date in observed:
                raise BuildError(f"duplicate official meeting end date: {end_date}")
            minutes_url = _minutes_url(
                archive_url=archive_url,
                end_date=end_date,
                links=raw["links"],
            )
            start = date.fromisoformat(start_date)
            observed[end_date] = Meeting(
                meeting_id=f"fomc-{start_date.replace('-', '')}-{end_date.replace('-', '')}",
                year=year,
                heading=heading,
                start_date=start_date,
                end_date=end_date,
                evidence_cutoff=(start - timedelta(days=1)).isoformat(),
                archive_url=resolved_url,
                minutes_url=minutes_url,
            )
        if set(observed) != expected:
            raise BuildError(
                f"official roster mismatch for {year}: "
                f"missing={sorted(expected - set(observed))}, "
                f"extra={sorted(set(observed) - expected)}"
            )
        meetings.extend(observed[item] for item in sorted(observed))
        page_records.append(
            {
                "year": year,
                "url": resolved_url,
                "path": str(page_path.resolve()),
                "sha256": sha256_file(page_path),
                "bytes": page_path.stat().st_size,
                "regular_meeting_count": len(observed),
            }
        )
    if len(meetings) != EXPECTED_MEETINGS:
        raise BuildError(f"official roster has {len(meetings)} meetings, expected 128")
    _write_json(
        output_root / "sources/official_calendar/pages_manifest.json",
        {
            "schema_version": "chk3-external-official-calendar-pages-v1",
            "status": "complete",
            "pages": page_records,
        },
    )
    return meetings


def download_minutes_inventory(
    *, output_root: Path, meetings: Sequence[Meeting]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, meeting in enumerate(meetings, 1):
        body, resolved_url = _http_get(meeting.minutes_url)
        path = output_root / (
            "sources/official_minutes/raw/" + meeting.meeting_id + ".html"
        )
        _write_immutable_bytes(path, body)
        rows.append(
            {
                "schema_version": "chk3-external-official-minutes-inventory-v1",
                "meeting_id": meeting.meeting_id,
                "meeting_end_date": meeting.end_date,
                "url": resolved_url,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "reference_role": "secondary_official_minutes_context_only",
            }
        )
        if index % 16 == 0:
            print(f"official_minutes_downloaded={index}/{len(meetings)}", flush=True)
    _write_jsonl(output_root / "sources/official_minutes/inventory.jsonl", rows)
    return rows


def _derive_source_contracts(
    *, output_root: Path, base_roster: Path, base_registry: Path
) -> tuple[Path, Path]:
    roster = json.loads(base_roster.read_text(encoding="utf-8"))
    registry = json.loads(base_registry.read_text(encoding="utf-8"))
    roster = deepcopy(roster)
    registry = deepcopy(registry)
    indicators = roster.get("indicators")
    if not isinstance(indicators, list) or len(indicators) != 26:
        raise BuildError("base roster is not the frozen 26-topic roster")
    if set(registry.get("indicators") or {}) != set(indicators):
        raise BuildError("base registry and roster topics differ")
    policy = registry.get("policy")
    required_policy = {
        "vintage_lag_calendar_days": 1,
        "same_meeting_day_data": "excluded",
        "unknown_availability": "fail_closed",
        "runtime_fallback": "forbidden",
    }
    if not isinstance(policy, Mapping) or any(
        policy.get(key) != value for key, value in required_policy.items()
    ):
        raise BuildError("base registry is not the exact D-1 contract")
    roster["roster_id"] = "chk3-external-historical-26-indicators-1993-2008-v1"
    roster["contexts"] = ["1993_2008_regular_meeting_start_d1"]
    roster["derived_from"] = {
        "path": str(base_roster.resolve()),
        "sha256": sha256_file(base_roster),
    }
    registry["registry_id"] = "chk3-external-keyless-start-d1-1993-2008-v1"
    registry["roster_id"] = roster["roster_id"]
    registry["derived_from"] = {
        "path": str(base_registry.resolve()),
        "sha256": sha256_file(base_registry),
    }
    roster_path = output_root / "sources/contracts/leave_one_out_roster.json"
    registry_path = output_root / "sources/contracts/loo_indicator_sources.json"
    _write_json(roster_path, roster)
    _write_json(registry_path, registry)
    load_source_registry(registry_path)
    roster_values = _load_roster(roster_path)
    _load_registry(registry_path, roster=roster_values)
    return roster_path, registry_path


def prepare(
    *,
    output_root: Path,
    base_roster: Path,
    base_registry: Path,
    download_minutes: bool,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"candidate output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    meetings = fetch_official_roster(output_root)
    roster_rows = [
        {
            "schema_version": ROSTER_SCHEMA,
            "meeting_id": item.meeting_id,
            "year": item.year,
            "meeting_type": "regular",
            "scheduled": True,
            "official_heading": item.heading,
            "meeting_start_date": item.start_date,
            "meeting_end_date": item.end_date,
            "evidence_cutoff": item.evidence_cutoff,
            "archive_url": item.archive_url,
            "official_minutes_url": item.minutes_url,
        }
        for item in meetings
    ]
    roster_path = output_root / "sources/official_meeting_roster.jsonl"
    _write_jsonl(roster_path, roster_rows)
    inventory = (
        download_minutes_inventory(output_root=output_root, meetings=meetings)
        if download_minutes
        else []
    )
    source_roster, source_registry = _derive_source_contracts(
        output_root=output_root,
        base_roster=base_roster,
        base_registry=base_registry,
    )
    # Sparse acquisition is currently scoped to the source pipeline's train
    # transport partition.  This label never propagates to the release role.
    plan_path = create_source_plan(
        meetings_by_split={
            "train": [item.start_date for item in meetings],
            "eval": [],
            "test": [],
        },
        output_dir=output_root / "sources/plan",
    )
    summary = {
        "schema_version": "chk3-external-holdout-preparation-v1",
        "status": "source_acquisition_pending",
        "release_id": RELEASE_ID,
        "intended_use": "evaluation_only_external_historical_holdout",
        "meeting_count": len(meetings),
        "meetings_per_year": {
            str(year): sum(item.year == year for item in meetings) for year in YEARS
        },
        "date_range": [meetings[0].start_date, meetings[-1].end_date],
        "evidence_cutoff_policy": "meeting_start_date_minus_1_calendar_day",
        "source_transport_split": "train_for_sparse_acquisition_only",
        "official_roster": {
            "path": str(roster_path.resolve()),
            "sha256": sha256_file(roster_path),
            "rows": len(roster_rows),
        },
        "official_minutes_inventory": {
            "downloaded": download_minutes,
            "rows": len(inventory),
            "path": str(
                (output_root / "sources/official_minutes/inventory.jsonl").resolve()
            )
            if download_minutes
            else None,
        },
        "source_plan": {
            "path": str(plan_path.resolve()),
            "sha256": sha256_file(plan_path),
        },
        "source_roster_sha256": sha256_file(source_roster),
        "source_registry_sha256": sha256_file(source_registry),
        "network_requests": len(YEARS) + (len(meetings) if download_minutes else 0),
    }
    _write_json(output_root / "reports/preparation.json", summary)
    return summary


def acquire(
    *,
    output_root: Path,
    resume: bool,
    requests_per_second: float,
    max_workers: int,
) -> dict[str, Any]:
    plan = output_root / "sources/plan/source_plan.json"
    roster = output_root / "sources/contracts/leave_one_out_roster.json"
    registry = output_root / "sources/contracts/loo_indicator_sources.json"
    for path in (plan, roster, registry):
        if not path.is_file():
            raise BuildError(f"preparation artifact is missing: {path}")
    handoff = materialize_source_plan(
        plan_path=plan,
        registry_path=registry,
        roster_path=roster,
        output_dir=output_root / "sources/materialized",
        sparse_train=True,
        resume=resume,
        max_workers=max_workers,
        requests_per_second=requests_per_second,
        sparse_http_get=_supplement_alfred_http_get,
    )
    validated = validate_source_handoff(
        handoff_path=handoff,
        plan_path=plan,
        registry_path=registry,
        roster_path=roster,
    )
    summary = {
        "schema_version": "chk3-external-holdout-source-acquisition-v1",
        "status": "complete",
        "handoff": {
            "path": str(handoff.resolve()),
            "sha256": sha256_file(handoff),
            "payload_sha256": validated["payload_sha256"],
        },
        "population_count": len(validated["ledgers"]),
        "meeting_count": sum(int(row["meeting_count"]) for row in validated["ledgers"]),
        "ready_topic_rows": sum(int(row["row_count"]) for row in validated["ledgers"]),
        "excluded_topic_rows": sum(
            int(row.get("excluded_sample_count") or 0) for row in validated["ledgers"]
        ),
    }
    if summary["meeting_count"] != EXPECTED_MEETINGS:
        raise BuildError(f"source handoff meeting closure failed: {summary}")
    _write_json(output_root / "reports/source_acquisition.json", summary)
    return summary


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BuildError(f"invalid JSONL: {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise BuildError(f"JSONL row is not an object: {path}:{line_number}")
            rows.append(row)
    return rows


def _load_tokenizer(path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def _topic_label(topic: str) -> str:
    return topic.replace("-(", " (").replace("-", " ")


def _change_clause(changes: Sequence[Mapping[str, Any]]) -> str:
    values = [
        f"{item['value_change']} over the {item['relative_horizon']} comparison"
        for item in changes
    ]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + ", and " + values[-1]


def _analysis_text(topic: str, evidence: Mapping[str, Any]) -> str:
    sentences: list[str] = []
    for series in evidence["series"]:
        clause = _change_clause(series.get("relative_changes") or [])
        sentence = (
            f"For {_topic_label(topic)}, the latest available reading for "
            f"{series['series']} was {series['latest_value']} {series['units']}."
        )
        if clause:
            sentence += f" Its recorded changes were {clause}."
        sentences.append(sentence)
    return " ".join(sentences)


def _reference_text(topic: str, evidence: Mapping[str, Any]) -> str:
    sentences: list[str] = []
    for index, series in enumerate(evidence["series"]):
        lead = "The latest information indicated" if index == 0 else "The data also indicated"
        sentence = (
            f"{lead} that {series['series']} stood at "
            f"{series['latest_value']} {series['units']}."
        )
        clause = _change_clause(series.get("relative_changes") or [])
        if clause:
            sentence += f" The corresponding changes were {clause}."
        sentences.append(sentence)
    return " ".join(sentences)


def _existing_chk3_meetings() -> set[str]:
    pattern = re.compile(r"chk1-analysis-(\d{4}-\d{2}-\d{2})-")
    meetings: set[str] = set()
    for split in ("train", "validation", "test"):
        path = CURRENT_CHK3_RELEASE / f"minutes_alignment/manifests/{split}.jsonl"
        for row in _load_jsonl(path):
            match = pattern.search(str(row.get("sample_id") or ""))
            if match is None:
                raise BuildError(f"cannot recover meeting from current CHK3 row: {row}")
            meetings.add(match.group(1))
    return meetings


def _prior_supplement_coverage(
    official_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prior_rows = _load_jsonl(PRIOR_SUPPLEMENT_ADMITTED)
    prior_dates = [str(row.get("meeting_date") or "") for row in prior_rows]
    if not prior_dates or any(not value for value in prior_dates):
        raise BuildError("prior 1993--2008 supplement meeting roster is empty/invalid")
    if len(set(prior_dates)) != len(prior_dates):
        raise BuildError("prior 1993--2008 supplement has duplicate meetings")
    by_start = {str(row["meeting_start_date"]): row for row in official_rows}
    by_end = {str(row["meeting_end_date"]): row for row in official_rows}
    start_matches = sorted(value for value in prior_dates if value in by_start)
    end_only_matches = sorted(
        value for value in prior_dates if value not in by_start and value in by_end
    )
    unmatched_prior = sorted(
        value for value in prior_dates if value not in by_start and value not in by_end
    )
    covered_ids = {
        str((by_start.get(value) or by_end.get(value))["meeting_id"])
        for value in prior_dates
        if value in by_start or value in by_end
    }
    missing = [
        {
            "meeting_id": str(row["meeting_id"]),
            "meeting_start_date": str(row["meeting_start_date"]),
            "meeting_end_date": str(row["meeting_end_date"]),
        }
        for row in official_rows
        if str(row["meeting_id"]) not in covered_ids
    ]
    if unmatched_prior:
        raise BuildError(f"prior supplement has non-official meetings: {unmatched_prior}")
    return {
        "schema_version": "chk3-external-prior-supplement-coverage-audit-v1",
        "status": "passed",
        "prior_supplement_path": str(PRIOR_SUPPLEMENT_ADMITTED.resolve()),
        "prior_supplement_sha256": sha256_file(PRIOR_SUPPLEMENT_ADMITTED),
        "prior_meeting_count": len(prior_dates),
        "official_regular_meeting_count": len(official_rows),
        "matched_by_official_start_date": len(start_matches),
        "matched_by_official_end_date_only": len(end_only_matches),
        "meetings_requiring_start_date_d1_cutoff_correction": len(end_only_matches),
        "previously_absent_official_regular_meetings": len(missing),
        "previously_absent_meetings": missing,
        "unmatched_prior_meetings": [],
    }


def _release_files(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "release_manifest.json":
            continue
        record: dict[str, Any] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".jsonl":
            record["rows"] = sum(
                1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            )
        result[relative] = record
    return result


def publish(
    *, output_root: Path, release_root: Path, tokenizer_path: Path
) -> dict[str, Any]:
    if release_root.exists():
        raise FileExistsError(f"release destination already exists: {release_root}")
    official_rows = _load_jsonl(output_root / "sources/official_meeting_roster.jsonl")
    if len(official_rows) != EXPECTED_MEETINGS:
        raise BuildError("official roster is incomplete")
    by_start = {str(row["meeting_start_date"]): row for row in official_rows}
    if len(by_start) != EXPECTED_MEETINGS:
        raise BuildError("official roster start dates are not unique")
    plan = output_root / "sources/plan/source_plan.json"
    source_roster = output_root / "sources/contracts/leave_one_out_roster.json"
    source_registry = output_root / "sources/contracts/loo_indicator_sources.json"
    handoff = output_root / "sources/materialized/source_handoff.json"
    validated = validate_source_handoff(
        handoff_path=handoff,
        plan_path=plan,
        registry_path=source_registry,
        roster_path=source_roster,
    )
    indicator_rows: list[dict[str, Any]] = []
    for record in validated["ledgers"]:
        if record.get("coverage_mode") != "sparse":
            raise BuildError("external holdout source coverage is not sparse")
        indicator_rows.extend(
            _load_jsonl(Path(str(record["ledger_dir"])) / "indicator_inputs.jsonl")
        )
    topic_map: dict[str, dict[str, tuple[dict[str, Any], str]]] = defaultdict(dict)
    pit_violations: list[str] = []
    for row in indicator_rows:
        start = str(row.get("meeting_date") or "")
        if start not in by_start:
            raise BuildError(f"source meeting is outside official roster: {start}")
        topic = str(row.get("indicator") or "")
        expected_cutoff = by_start[start]["evidence_cutoff"]
        for field in (
            "requested_vintage_date",
            "availability_as_of_date",
            "information_as_of_date",
        ):
            if str(row.get(field) or "") != expected_cutoff:
                pit_violations.append(f"{start}::{topic}::{field}")
        compressed = compress_indicator_row(row)
        if compressed is None:
            continue
        for series in compressed["provenance"]:
            if series["latest_observation_date"] > expected_cutoff:
                pit_violations.append(f"{start}::{topic}::future_observation")
        if topic in topic_map[start]:
            raise BuildError(f"duplicate source meeting/topic: {start}::{topic}")
        topic_map[start][topic] = (
            {key: value for key, value in compressed.items() if key != "provenance"},
            str(row.get("source_id") or ""),
        )
    if pit_violations:
        raise BuildError(f"point-in-time violations: {pit_violations[:5]}")
    missing_core = {
        start: sorted(set(CORE_TOPICS) - set(topic_map.get(start, {})))
        for start in sorted(by_start)
        if set(CORE_TOPICS) - set(topic_map.get(start, {}))
    }
    if missing_core:
        raise BuildError(
            "Core-8 is unavailable for every meeting; publish is blocked: "
            + json.dumps(dict(list(missing_core.items())[:5]), sort_keys=True)
        )
    tokenizer = _load_tokenizer(tokenizer_path)
    core_rows: list[dict[str, Any]] = []
    expanded_rows: list[dict[str, Any]] = []
    prompt_tokens: list[int] = []
    for start, meeting in sorted(by_start.items()):
        for topic in sorted(topic_map[start]):
            evidence, source_id = topic_map[start][topic]
            analysis = _analysis_text(topic, evidence)
            reference = _reference_text(topic, evidence)
            if Counter(NUMBER_RE.findall(analysis)) != Counter(NUMBER_RE.findall(reference)):
                raise BuildError(f"deterministic reference changed numbers: {start}::{topic}")
            prompt = (
                "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
                + _canonical_json({"analysis": analysis})
            )
            token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
            prompt_tokens.append(token_count)
            sample_id = (
                "chk3-external-"
                + sha256_text(f"{RELEASE_ID}\0{meeting['meeting_id']}\0{topic}")[:24]
            )
            row = {
                "schema_version": SAMPLE_SCHEMA,
                "sample_id": sample_id,
                "meeting_id": meeting["meeting_id"],
                "meeting_start_date": start,
                "meeting_end_date": meeting["meeting_end_date"],
                "evidence_cutoff": meeting["evidence_cutoff"],
                "topic": topic,
                "panel": "core8" if topic in CORE_TOPICS else "expanded_only",
                "prompt": prompt,
                "source_analysis": analysis,
                "reference_minutes": reference,
                "reference_type": "deterministic_source_grounded_minutes_style_v1",
                "official_minutes_url": meeting["official_minutes_url"],
                "official_minutes_role": "secondary_context_not_model_input",
                "source_id": source_id,
                "topic_evidence": evidence,
                "prompt_tokens": token_count,
                "prompt_sha256": sha256_text(prompt),
                "source_analysis_sha256": sha256_text(analysis),
                "reference_minutes_sha256": sha256_text(reference),
                "topic_evidence_sha256": sha256_text(_canonical_json(evidence)),
            }
            expanded_rows.append(row)
            if topic in CORE_TOPICS:
                core_rows.append(row)
    if len(core_rows) != EXPECTED_MEETINGS * len(CORE_TOPICS):
        raise BuildError(f"Core-8 closure failed: {len(core_rows)}")
    current_meetings = _existing_chk3_meetings()
    overlap = sorted(set(by_start) & current_meetings)
    if overlap:
        raise BuildError(f"external holdout overlaps current CHK3 meetings: {overlap}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{release_root.name}.", dir=release_root.parent)
    )
    try:
        _write_jsonl(staging / "official_meeting_roster.jsonl", official_rows)
        _write_jsonl(staging / "panels/core8.jsonl", core_rows)
        _write_jsonl(staging / "panels/expanded_all_available.jsonl", expanded_rows)
        source_inventory = _load_jsonl(
            output_root / "sources/official_minutes/inventory.jsonl"
        )
        minutes_raw = staging / "references/official_minutes_raw"
        minutes_raw.mkdir(parents=True, exist_ok=False)
        release_inventory: list[dict[str, Any]] = []
        for source_record in source_inventory:
            source_path = Path(str(source_record["path"]))
            if not source_path.is_file():
                raise BuildError(f"official Minutes source is missing: {source_path}")
            if sha256_file(source_path) != source_record.get("sha256"):
                raise BuildError(f"official Minutes source hash drift: {source_path}")
            destination = minutes_raw / source_path.name
            shutil.copyfile(source_path, destination)
            release_record = dict(source_record)
            release_record["path"] = destination.relative_to(staging).as_posix()
            release_inventory.append(release_record)
        _write_jsonl(
            staging / "references/official_minutes_inventory.jsonl",
            release_inventory,
        )
        roster_audit = {
            "schema_version": "chk3-external-roster-audit-v1",
            "status": "passed",
            "regular_meetings": len(official_rows),
            "meetings_per_year": dict(
                sorted(Counter(str(row["year"]) for row in official_rows).items())
            ),
            "expected_per_year": EXPECTED_PER_YEAR,
            "unique_meeting_ids": len({row["meeting_id"] for row in official_rows}),
            "unique_start_dates": len(by_start),
            "current_chk3_meeting_overlap": 0,
        }
        source_audit = {
            "schema_version": "chk3-external-source-audit-v1",
            "status": "passed",
            "meeting_count": len(topic_map),
            "core8_rows": len(core_rows),
            "expanded_rows": len(expanded_rows),
            "topic_coverage": dict(
                sorted(Counter(row["topic"] for row in expanded_rows).items())
            ),
            "point_in_time_violations": 0,
            "input_truncation": 0,
            "prompt_tokens": {
                "min": min(prompt_tokens),
                "max": max(prompt_tokens),
            },
            "source_handoff": {
                "path": str(handoff.resolve()),
                "sha256": sha256_file(handoff),
                "payload_sha256": validated["payload_sha256"],
            },
        }
        reference_audit = {
            "schema_version": "chk3-external-reference-audit-v1",
            "status": "passed",
            "reference_type": "deterministic_source_grounded_minutes_style_v1",
            "numeric_multiset_mismatch_rows": 0,
            "official_minutes_reference_role": "secondary_context_only",
            "official_minutes_rows": len(source_inventory),
            "official_minutes_in_model_prompt": False,
            "decision_gold_in_model_prompt": False,
        }
        prior_coverage_audit = _prior_supplement_coverage(official_rows)
        _write_json(staging / "audits/roster_completeness.json", roster_audit)
        _write_json(staging / "audits/source_quality.json", source_audit)
        _write_json(staging / "audits/reference_quality.json", reference_audit)
        _write_json(
            staging / "audits/prior_supplement_coverage.json",
            prior_coverage_audit,
        )
        tokenizer_fingerprint = fingerprint_tokenizer_payload(tokenizer_path)
        files = _release_files(staging)
        manifest = {
            "schema_version": "chk3-external-evaluation-release-v1",
            "release_id": RELEASE_ID,
            "created_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "status": "passed",
            "immutable": True,
            "evaluation_only": True,
            "trainable": False,
            "checkpoint_selection_allowed": False,
            "promotable": False,
            "task_contract": "analysis_to_formal_fomc_minutes_prose",
            "grain": "one_regular_meeting_topic_per_row",
            "meeting_count": EXPECTED_MEETINGS,
            "core_topic_count": len(CORE_TOPICS),
            "core8_rows": len(core_rows),
            "expanded_rows": len(expanded_rows),
            "evidence_cutoff_policy": "meeting_start_date_minus_1_calendar_day",
            "source_transport_split_is_not_release_split": True,
            "historical_external_holdout_not_prospective": True,
            "base_model_pretraining_independence_not_claimed": True,
            "tokenizer": {
                "path": str(tokenizer_path.resolve()),
                "payload": tokenizer_fingerprint,
            },
            "source_lineage": {
                "candidate_root": str(output_root.resolve()),
                "builder_sha256": sha256_file(Path(__file__)),
                "official_roster_sha256": sha256_file(
                    output_root / "sources/official_meeting_roster.jsonl"
                ),
                "source_handoff_sha256": sha256_file(handoff),
                "source_handoff_payload_sha256": validated["payload_sha256"],
            },
            "files": files,
        }
        _write_json(staging / "release_manifest.json", seal_manifest(manifest))
        os.rename(staging, release_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    result = {
        "status": "complete",
        "release_root": str(release_root.resolve()),
        "release_manifest": str((release_root / "release_manifest.json").resolve()),
        "release_manifest_sha256": sha256_file(release_root / "release_manifest.json"),
        "meeting_count": EXPECTED_MEETINGS,
        "core8_rows": len(core_rows),
        "expanded_rows": len(expanded_rows),
    }
    _write_json(output_root / "reports/publish.json", result)
    return result


def validate_release(release_root: Path) -> dict[str, Any]:
    manifest_path = release_root / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        payload_sha256 = validate_manifest_integrity(manifest)
    except ManifestIntegrityError as exc:
        raise BuildError(f"release manifest integrity failed: {exc}") from exc
    if manifest.get("status") != "passed" or manifest.get("evaluation_only") is not True:
        raise BuildError("release status/scope is invalid")
    if (
        manifest.get("trainable") is not False
        or manifest.get("checkpoint_selection_allowed") is not False
        or manifest.get("promotable") is not False
        or manifest.get("meeting_count") != EXPECTED_MEETINGS
        or manifest.get("core8_rows") != EXPECTED_MEETINGS * len(CORE_TOPICS)
    ):
        raise BuildError("release scope/count contract is invalid")
    for relative, record in manifest.get("files", {}).items():
        path = release_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise BuildError(f"release file binding changed: {relative}")
    core = _load_jsonl(release_root / "panels/core8.jsonl")
    if len(core) != EXPECTED_MEETINGS * len(CORE_TOPICS):
        raise BuildError("validated release Core-8 closure failed")
    keys = {(row["meeting_id"], row["topic"]) for row in core}
    if len(keys) != len(core):
        raise BuildError("validated release has duplicate meeting/topic keys")
    for relative in (
        "audits/prior_supplement_coverage.json",
        "audits/reference_quality.json",
        "audits/roster_completeness.json",
        "audits/source_quality.json",
    ):
        audit = json.loads((release_root / relative).read_text(encoding="utf-8"))
        if audit.get("status") != "passed":
            raise BuildError(f"release audit is not passed: {relative}")
    inventory = _load_jsonl(
        release_root / "references/official_minutes_inventory.jsonl"
    )
    if len(inventory) != EXPECTED_MEETINGS:
        raise BuildError("official Minutes inventory closure failed")
    for record in inventory:
        relative = str(record.get("path") or "")
        source = release_root / relative
        if not source.is_file() or sha256_file(source) != record.get("sha256"):
            raise BuildError(f"official Minutes binding changed: {relative}")
    return {
        "status": "valid",
        "release_manifest_sha256": sha256_file(manifest_path),
        "release_manifest_payload_sha256": payload_sha256,
        "meeting_count": len({row["meeting_id"] for row in core}),
        "core8_rows": len(core),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--base-roster", type=Path, default=DEFAULT_BASE_ROSTER)
    prepare_parser.add_argument("--base-registry", type=Path, default=DEFAULT_BASE_REGISTRY)
    prepare_parser.add_argument("--skip-minutes-download", action="store_true")
    acquire_parser = subparsers.add_parser("acquire")
    acquire_parser.add_argument("--resume", action="store_true")
    acquire_parser.add_argument("--requests-per-second", type=float, default=2.0)
    acquire_parser.add_argument("--max-workers", type=int, default=2)
    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    subparsers.add_parser("validate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    release_root = args.release_root.expanduser().resolve()
    try:
        if args.command == "prepare":
            result = prepare(
                output_root=output_root,
                base_roster=args.base_roster.expanduser().resolve(),
                base_registry=args.base_registry.expanduser().resolve(),
                download_minutes=not args.skip_minutes_download,
            )
        elif args.command == "acquire":
            result = acquire(
                output_root=output_root,
                resume=args.resume,
                requests_per_second=args.requests_per_second,
                max_workers=args.max_workers,
            )
        elif args.command == "publish":
            result = publish(
                output_root=output_root,
                release_root=release_root,
                tokenizer_path=args.tokenizer.expanduser().resolve(),
            )
        else:
            result = validate_release(release_root)
        print(_pretty_json(result), end="")
        return 0
    except (BuildError, FileExistsError, OSError, ValueError, requests.RequestException) as exc:
        print(_pretty_json({"status": "blocked", "error": str(exc)}), end="", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CORE_TOPICS",
    "EXPECTED_END_DATES",
    "EXPECTED_MEETINGS",
    "Meeting",
    "_parse_heading",
    "fetch_official_roster",
    "prepare",
    "publish",
    "validate_release",
]
