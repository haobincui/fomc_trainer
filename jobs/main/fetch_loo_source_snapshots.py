"""Fetch immutable, keyless ALFRED vintage snapshots for canonical LOO.

The ALFRED graph CSV exporter is an official public download surface that does
not require a FRED API key.  It accepts vector-valued graph parameters, but the
exporter silently truncates graphs after twelve lines.  This module therefore
freezes the interface as ``alfred-graph-csv-v1`` and deterministically chunks
the 26 canonical meeting vintages as 12 + 12 + 2.

Every response is validated before it enters the cache.  In particular, the
column names must echo every requested vintage date exactly.  This is critical
because the public endpoint can return HTTP 200 and silently fall back to the
current vintage when given an invalid or future vintage date.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import threading
import time
import zipfile
from math import ceil
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


SNAPSHOT_SCHEMA_VERSION = "loo-source-snapshot-manifest-v1"
SOURCE_INTERFACE = "alfred-graph-csv-v1"
ALFRED_GRAPH_CSV_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
MAX_VINTAGES_PER_REQUEST = 12
CANONICAL_VINTAGE_COUNT = 26
CHECKPOINT_EVAL_VINTAGE_COUNT = 11
MAX_CONCURRENCY = 2
MAX_REQUESTS_PER_SECOND = 2.0
DEFAULT_LOOKBACK_YEARS = 5
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 4
DEFAULT_USER_AGENT = "fomc-trainer-canonical-loo/1"
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")
MAX_ZIP_MEMBER_COUNT = 16
MAX_ZIP_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
ZIP_NORMALIZATION = "alfred-partitioned-csv-zip-to-csv-v1"


class SnapshotFetchError(RuntimeError):
    """Raised when a source snapshot cannot be fetched or validated safely."""


@dataclass(frozen=True)
class SourceSeries:
    """One ALFRED source selected by the frozen source registry."""

    source_key: str
    series_id: str
    lookback_years: int


@dataclass(frozen=True)
class SnapshotRequest:
    """One graph-export request containing at most twelve vintage columns."""

    request_id: str
    source_key: str
    series_id: str
    vintage_dates: tuple[str, ...]
    cosd: tuple[str, ...]
    coed: tuple[str, ...]
    canonical_url: str
    raw_relative_path: str


@dataclass(frozen=True)
class HttpResult:
    """Transport-neutral HTTP response used by the fetcher and its tests."""

    status_code: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class CsvValidation:
    """Validated response facts retained in the snapshot manifest."""

    row_count: int
    column_nonempty_counts: tuple[int, ...]
    first_observation_date: str
    last_observation_date: str


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _canonical_date(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"{label} must be an ISO YYYY-MM-DD date, got {value!r}"
        ) from exc
    if parsed.isoformat() != text:
        raise ValueError(f"{label} must use canonical YYYY-MM-DD form, got {value!r}")
    return text


def _safe_component(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not SAFE_COMPONENT.fullmatch(text) or text in {".", ".."}:
        raise ValueError(
            f"{label} must contain only letters, digits, dot, underscore, "
            f"or hyphen: {value!r}"
        )
    return text


def _lookback_years(
    raw_source: Mapping[str, Any],
    *,
    default_years: int,
    source_key: str,
) -> int:
    """Return a conservative calendar window for the graph CSV request.

    The downstream ledger applies the registry's exact observation/calendar
    lookback after parsing.  Acquisition may safely request a wider window, so
    this helper keeps at least ``default_years`` and only expands it when the
    configured lookback is longer.
    """

    explicit = raw_source.get(
        "snapshot_lookback_years",
        raw_source.get("fetch_lookback_years", raw_source.get("lookback_years")),
    )
    if explicit is not None:
        if (
            isinstance(explicit, bool)
            or not isinstance(explicit, int)
            or not 1 <= explicit <= 25
        ):
            raise ValueError(
                f"Source {source_key!r} snapshot lookback years must be 1..25"
            )
        return max(default_years, explicit)

    lookback = raw_source.get("lookback")
    amount: int | None = None
    unit = ""
    if isinstance(lookback, Mapping):
        raw_amount = lookback.get("value")
        raw_unit = lookback.get("unit")
        if isinstance(raw_amount, int) and not isinstance(raw_amount, bool):
            amount = raw_amount
            unit = str(raw_unit or "").strip().lower()
        else:
            for candidate in ("days", "weeks", "months", "quarters", "years"):
                raw_amount = lookback.get(candidate)
                if isinstance(raw_amount, int) and not isinstance(raw_amount, bool):
                    amount = raw_amount
                    unit = candidate
                    break
    elif isinstance(lookback, str):
        match = re.fullmatch(
            r"\s*([1-9][0-9]*)\s*"
            r"(d|days?|w|weeks?|m|months?|q|quarters?|y|years?)\s*",
            lookback,
            flags=re.IGNORECASE,
        )
        if match:
            amount = int(match.group(1))
            unit = match.group(2).lower()

    if amount is None:
        # Observation-count policies need a frequency-aware downstream
        # selection.  The frozen five-year acquisition window is deliberately
        # wider than the current registry's 25-point maximum.
        return default_years
    if amount <= 0:
        raise ValueError(f"Source {source_key!r} lookback must be positive")

    if unit in {"d", "day", "days"}:
        configured_years = ceil(amount / 365)
    elif unit in {"w", "week", "weeks"}:
        configured_years = ceil(amount / 52)
    elif unit in {"m", "month", "months"}:
        configured_years = ceil(amount / 12)
    elif unit in {"q", "quarter", "quarters"}:
        configured_years = ceil(amount / 4)
    elif unit in {"y", "year", "years"}:
        configured_years = amount
    else:
        # An observation-count policy (or an unknown policy rejected later by
        # the ledger) does not narrow the conservative acquisition window.
        return default_years
    if configured_years > 25:
        raise ValueError(
            f"Source {source_key!r} requires a snapshot lookback over 25 years"
        )
    return max(default_years, configured_years)


def load_source_registry(path: str | Path) -> tuple[dict[str, Any], list[SourceSeries]]:
    """Load selected sources from the canonical ``sources``/``indicators`` maps."""

    registry_path = Path(path).expanduser().resolve()
    payload = _load_json_object(registry_path, label="source registry")
    raw_sources = payload.get("sources", payload.get("source"))
    raw_indicators = payload.get("indicators")
    raw_defaults = payload.get("series_defaults", {})
    if not isinstance(raw_sources, Mapping) or not raw_sources:
        raise ValueError("Source registry requires a non-empty 'sources' map")
    if not isinstance(raw_indicators, Mapping) or not raw_indicators:
        raise ValueError("Source registry requires a non-empty 'indicators' map")
    if not isinstance(raw_defaults, Mapping):
        raise ValueError("Source registry series_defaults must be an object")

    ordered_source_keys: list[str] = []
    for indicator, raw_indicator in raw_indicators.items():
        if not str(indicator).strip():
            raise ValueError("Source registry contains an empty indicator name")
        if isinstance(raw_indicator, list):
            source_keys = raw_indicator
        elif isinstance(raw_indicator, Mapping):
            source_keys = raw_indicator.get("source_keys")
        else:
            raise ValueError(
                f"Indicator {indicator!r} must be an object with source_keys"
            )
        if not isinstance(source_keys, list) or not source_keys:
            raise ValueError(
                f"Indicator {indicator!r} requires a non-empty source_keys list"
            )
        enabled_count = 0
        for raw_source_key in source_keys:
            source_key = _safe_component(
                raw_source_key,
                label=f"{indicator!r} source_key",
            )
            if source_key not in raw_sources:
                raise ValueError(
                    f"Indicator {indicator!r} references unknown source_key "
                    f"{source_key!r}"
                )
            source_override = raw_sources[source_key]
            if not isinstance(source_override, Mapping):
                raise ValueError(f"Source {source_key!r} must be a JSON object")
            merged_source = {**dict(raw_defaults), **dict(source_override)}
            enabled = merged_source.get("enabled", True)
            if not isinstance(enabled, bool):
                raise ValueError(f"Source {source_key!r} enabled must be boolean")
            if enabled:
                enabled_count += 1
            if enabled and source_key not in ordered_source_keys:
                ordered_source_keys.append(source_key)
        if enabled_count == 0:
            raise ValueError(f"Indicator {indicator!r} selects no enabled source")

    default_lookback = payload.get(
        "default_lookback_years",
        DEFAULT_LOOKBACK_YEARS,
    )
    if (
        isinstance(default_lookback, bool)
        or not isinstance(default_lookback, int)
        or not 1 <= default_lookback <= 25
    ):
        raise ValueError("default_lookback_years must be an integer from 1 to 25")

    series: list[SourceSeries] = []
    seen_series_ids: set[str] = set()
    for source_key in ordered_source_keys:
        source_override = raw_sources[source_key]
        if not isinstance(source_override, Mapping):
            raise ValueError(f"Source {source_key!r} must be a JSON object")
        raw_source = {**dict(raw_defaults), **dict(source_override)}
        source_interface = str(
            raw_source.get(
                "access_interface",
                raw_source.get("source_interface", SOURCE_INTERFACE),
            )
        ).strip()
        if source_interface != SOURCE_INTERFACE:
            raise ValueError(
                f"Source {source_key!r} uses unsupported source_interface "
                f"{source_interface!r}"
            )
        series_id = _safe_component(
            raw_source.get("series_id"),
            label=f"{source_key!r} series_id",
        )
        if series_id in seen_series_ids:
            raise ValueError(
                f"Source registry assigns ALFRED series {series_id!r} more than once"
            )
        lookback_years = _lookback_years(
            raw_source,
            default_years=default_lookback,
            source_key=source_key,
        )
        series.append(
            SourceSeries(
                source_key=source_key,
                series_id=series_id,
                lookback_years=lookback_years,
            )
        )
        seen_series_ids.add(series_id)
    if not series:
        raise ValueError("Source registry selects no enabled source series")
    return payload, series


def load_meetings(
    population_paths: Sequence[str | Path],
    *,
    expected_vintage_count: int | None = CANONICAL_VINTAGE_COUNT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load population files and derive the conservative previous-day vintages."""

    if not population_paths:
        raise ValueError("At least one population JSON is required")
    populations: list[dict[str, Any]] = []
    populations_by_meeting: dict[str, list[str]] = {}
    seen_population_ids: set[str] = set()
    for raw_path in population_paths:
        path = Path(raw_path).expanduser().resolve()
        payload = _load_json_object(path, label="population")
        population_id = str(payload.get("population_id") or "").strip()
        raw_dates = payload.get("meeting_dates")
        if not population_id or population_id in seen_population_ids:
            raise ValueError(
                "Population IDs must be non-empty and unique across inputs"
            )
        if not isinstance(raw_dates, list) or not raw_dates:
            raise ValueError(f"Population {population_id!r} requires meeting_dates")
        meeting_dates = [
            _canonical_date(value, label=f"{population_id} meeting date")
            for value in raw_dates
        ]
        if meeting_dates != sorted(set(meeting_dates)):
            raise ValueError(
                f"Population {population_id!r} meeting_dates must be unique "
                "and ascending"
            )
        for meeting_date in meeting_dates:
            populations_by_meeting.setdefault(meeting_date, []).append(population_id)
        populations.append(
            {
                "population_id": population_id,
                "path": str(path),
                "sha256": sha256_file(path),
                "meeting_dates": meeting_dates,
            }
        )
        seen_population_ids.add(population_id)

    if expected_vintage_count is not None and (
        len(populations_by_meeting) != expected_vintage_count
    ):
        raise ValueError(
            "Canonical snapshot acquisition requires exactly "
            f"{expected_vintage_count} unique meetings, found "
            f"{len(populations_by_meeting)}"
        )

    meetings = []
    for meeting_date in sorted(populations_by_meeting):
        information_as_of = (
            date.fromisoformat(meeting_date) - timedelta(days=1)
        ).isoformat()
        meetings.append(
            {
                "meeting_date": meeting_date,
                "information_as_of_date": information_as_of,
                "population_ids": sorted(populations_by_meeting[meeting_date]),
            }
        )
    return meetings, populations


def _subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def _chunks(values: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _request_identity_payload(
    *,
    source_key: str,
    series_id: str,
    vintage_dates: Sequence[str],
    cosd: Sequence[str],
    coed: Sequence[str],
    canonical_url: str,
) -> dict[str, Any]:
    return {
        "source_key": source_key,
        "series_id": series_id,
        "vintage_dates": list(vintage_dates),
        "cosd": list(cosd),
        "coed": list(coed),
        "canonical_url": canonical_url,
    }


def build_snapshot_requests(
    sources: Sequence[SourceSeries],
    meetings: Sequence[Mapping[str, Any]],
) -> list[SnapshotRequest]:
    """Build deterministic graph requests using the verified twelve-line cap."""

    if not sources:
        raise ValueError("At least one source series is required")
    vintage_dates = [
        _canonical_date(
            meeting.get("information_as_of_date"),
            label="information_as_of_date",
        )
        for meeting in meetings
    ]
    if vintage_dates != sorted(set(vintage_dates)):
        raise ValueError(
            "Meeting information_as_of_date values must be unique and ascending"
        )

    requests: list[SnapshotRequest] = []
    for source in sources:
        source_vintages = tuple(vintage_dates)
        starts = tuple(
            _subtract_years(
                date.fromisoformat(vintage_date),
                source.lookback_years,
            ).isoformat()
            for vintage_date in source_vintages
        )
        ends = source_vintages
        indexes = list(range(len(source_vintages)))
        for chunk_indexes in _chunks(indexes, MAX_VINTAGES_PER_REQUEST):
            chunk_vintages = tuple(source_vintages[index] for index in chunk_indexes)
            chunk_starts = tuple(starts[index] for index in chunk_indexes)
            chunk_ends = tuple(ends[index] for index in chunk_indexes)
            params = {
                "id": ",".join([source.series_id] * len(chunk_vintages)),
                "cosd": ",".join(chunk_starts),
                "coed": ",".join(chunk_ends),
                "vintage_date": ",".join(chunk_vintages),
            }
            canonical_url = f"{ALFRED_GRAPH_CSV_URL}?{urlencode(params)}"
            identity = _request_identity_payload(
                source_key=source.source_key,
                series_id=source.series_id,
                vintage_dates=chunk_vintages,
                cosd=chunk_starts,
                coed=chunk_ends,
                canonical_url=canonical_url,
            )
            request_id = _sha256_bytes(_canonical_json(identity).encode("utf-8"))
            relative_path = (
                Path("raw") / "alfred" / source.source_key / request_id / "response.csv"
            ).as_posix()
            requests.append(
                SnapshotRequest(
                    request_id=request_id,
                    source_key=source.source_key,
                    series_id=source.series_id,
                    vintage_dates=chunk_vintages,
                    cosd=chunk_starts,
                    coed=chunk_ends,
                    canonical_url=canonical_url,
                    raw_relative_path=relative_path,
                )
            )
    return requests


def validate_alfred_csv(
    body: bytes,
    request: SnapshotRequest,
) -> CsvValidation:
    """Validate exact vintage echoing, bounds, ordering, and decimal values."""

    column_count = len(request.vintage_dates)
    if (
        not 1 <= column_count <= MAX_VINTAGES_PER_REQUEST
        or len(request.cosd) != column_count
        or len(request.coed) != column_count
    ):
        raise SnapshotFetchError(
            f"{request.request_id}: request arrays must have equal lengths "
            f"from 1 to {MAX_VINTAGES_PER_REQUEST}"
        )
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SnapshotFetchError(
            f"{request.request_id}: ALFRED CSV is not valid UTF-8"
        ) from exc
    if "\x00" in text:
        raise SnapshotFetchError(
            f"{request.request_id}: ALFRED CSV contains a NUL byte"
        )
    try:
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except csv.Error as exc:
        raise SnapshotFetchError(
            f"{request.request_id}: malformed ALFRED CSV: {exc}"
        ) from exc
    if not rows:
        raise SnapshotFetchError(f"{request.request_id}: empty ALFRED CSV")

    expected_header = ["observation_date"] + [
        f"{request.series_id}_{vintage_date.replace('-', '')}"
        for vintage_date in request.vintage_dates
    ]
    if rows[0] != expected_header:
        raise SnapshotFetchError(
            f"{request.request_id}: ALFRED header mismatch; "
            f"expected={expected_header!r}, observed={rows[0]!r}"
        )
    ends = [date.fromisoformat(value) for value in request.coed]
    nonempty_counts = [0] * len(request.vintage_dates)
    previous_date: date | None = None
    first_date: date | None = None
    last_date: date | None = None
    data_row_count = 0
    for row_number, row in enumerate(rows[1:], 2):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) != len(expected_header):
            raise SnapshotFetchError(
                f"{request.request_id}: row {row_number} has {len(row)} "
                f"columns, expected {len(expected_header)}"
            )
        raw_date = row[0].strip()
        try:
            observation_date = date.fromisoformat(raw_date)
        except ValueError as exc:
            raise SnapshotFetchError(
                f"{request.request_id}: row {row_number} has invalid "
                f"observation_date {raw_date!r}"
            ) from exc
        if observation_date.isoformat() != raw_date:
            raise SnapshotFetchError(
                f"{request.request_id}: row {row_number} observation_date "
                "is not canonical YYYY-MM-DD"
            )
        if previous_date is not None and observation_date <= previous_date:
            raise SnapshotFetchError(
                f"{request.request_id}: observation dates are not strictly "
                "ascending and unique"
            )
        previous_date = observation_date
        first_date = first_date or observation_date
        last_date = observation_date
        data_row_count += 1

        for column_index, raw_value in enumerate(row[1:]):
            value = raw_value.strip()
            if not value:
                continue
            # The graph exporter includes the full containing period for some
            # monthly/quarterly series, so its first observation_date can be
            # earlier than cosd.  That is safe and the ledger applies the exact
            # lower lookback bound.  Values after coed would cross the frozen
            # information cutoff and must fail closed here.
            if observation_date > ends[column_index]:
                raise SnapshotFetchError(
                    f"{request.request_id}: observation {raw_date} in column "
                    f"{expected_header[column_index + 1]!r} is after coed "
                    f"{request.coed[column_index]}"
                )
            try:
                numeric = Decimal(value)
            except InvalidOperation as exc:
                raise SnapshotFetchError(
                    f"{request.request_id}: non-decimal value {value!r} at "
                    f"row {row_number}, column {column_index + 2}"
                ) from exc
            if not numeric.is_finite():
                raise SnapshotFetchError(
                    f"{request.request_id}: non-finite value {value!r} at "
                    f"row {row_number}, column {column_index + 2}"
                )
            nonempty_counts[column_index] += 1

    if first_date is None or last_date is None:
        raise SnapshotFetchError(
            f"{request.request_id}: ALFRED CSV has no observation rows"
        )
    empty_vintages = [
        request.vintage_dates[index]
        for index, count in enumerate(nonempty_counts)
        if count == 0
    ]
    if empty_vintages:
        raise SnapshotFetchError(
            f"{request.request_id}: no numeric observations for vintages "
            f"{empty_vintages}"
        )
    return CsvValidation(
        row_count=data_row_count,
        column_nonempty_counts=tuple(nonempty_counts),
        first_observation_date=first_date.isoformat(),
        last_observation_date=last_date.isoformat(),
    )


def _normalise_alfred_zip(
    body: bytes,
    request: SnapshotRequest,
) -> tuple[bytes, dict[str, Any]]:
    """Merge ALFRED's definition-split ZIP response into one canonical CSV.

    ALFRED occasionally partitions requested vintages into separate CSV
    members when a series definition changes.  This function accepts only a
    safe, exact partition of the requested vintage columns and reconstructs
    the ordinary graph-export CSV expected by the downstream validator.
    """

    expected_columns = [
        f"{request.series_id}_{vintage_date.replace('-', '')}"
        for vintage_date in request.vintage_dates
    ]
    expected_set = set(expected_columns)
    values_by_column: dict[str, dict[str, str]] = {}
    all_dates: set[str] = set()
    member_records: list[dict[str, Any]] = []

    try:
        archive = zipfile.ZipFile(io.BytesIO(body))
    except (OSError, zipfile.BadZipFile) as exc:
        raise SnapshotFetchError(
            f"{request.request_id}: malformed ALFRED ZIP response"
        ) from exc

    with archive:
        members = archive.infolist()
        if not 1 <= len(members) <= MAX_ZIP_MEMBER_COUNT:
            raise SnapshotFetchError(
                f"{request.request_id}: ALFRED ZIP contains {len(members)} "
                f"members; allowed range is 1..{MAX_ZIP_MEMBER_COUNT}"
            )
        total_uncompressed = sum(member.file_size for member in members)
        if total_uncompressed > MAX_ZIP_UNCOMPRESSED_BYTES:
            raise SnapshotFetchError(
                f"{request.request_id}: ALFRED ZIP expands to "
                f"{total_uncompressed} bytes, exceeding the safety limit"
            )

        csv_member_count = 0
        for member in members:
            name = member.filename
            if (
                not name
                or "\x00" in name
                or "/" in name
                or "\\" in name
                or name in {".", ".."}
                or member.is_dir()
            ):
                raise SnapshotFetchError(
                    f"{request.request_id}: unsafe ALFRED ZIP member {name!r}"
                )
            if member.flag_bits & 0x1:
                raise SnapshotFetchError(
                    f"{request.request_id}: encrypted ALFRED ZIP members "
                    "are not permitted"
                )
            if name.casefold() == "readme.txt":
                continue
            if not name.casefold().endswith(".csv"):
                raise SnapshotFetchError(
                    f"{request.request_id}: unexpected ALFRED ZIP member "
                    f"{name!r}"
                )

            csv_member_count += 1
            try:
                member_body = archive.read(member)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise SnapshotFetchError(
                    f"{request.request_id}: could not read ALFRED ZIP "
                    f"member {name!r}"
                ) from exc
            if len(member_body) != member.file_size:
                raise SnapshotFetchError(
                    f"{request.request_id}: ALFRED ZIP member {name!r} "
                    "size does not match its archive directory entry"
                )
            try:
                text = member_body.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                raise SnapshotFetchError(
                    f"{request.request_id}: ALFRED ZIP member {name!r} "
                    "is not valid UTF-8"
                ) from exc
            if "\x00" in text:
                raise SnapshotFetchError(
                    f"{request.request_id}: ALFRED ZIP member {name!r} "
                    "contains a NUL byte"
                )
            try:
                rows = list(csv.reader(io.StringIO(text, newline="")))
            except csv.Error as exc:
                raise SnapshotFetchError(
                    f"{request.request_id}: malformed CSV member {name!r}: "
                    f"{exc}"
                ) from exc
            if not rows or len(rows[0]) < 2 or rows[0][0] != "observation_date":
                raise SnapshotFetchError(
                    f"{request.request_id}: invalid header in ALFRED ZIP "
                    f"member {name!r}"
                )
            columns = rows[0][1:]
            if len(columns) != len(set(columns)):
                raise SnapshotFetchError(
                    f"{request.request_id}: duplicate vintage column within "
                    f"ALFRED ZIP member {name!r}"
                )
            unexpected = sorted(set(columns) - expected_set)
            duplicate = sorted(set(columns) & set(values_by_column))
            if unexpected or duplicate:
                raise SnapshotFetchError(
                    f"{request.request_id}: ALFRED ZIP column partition is "
                    f"invalid; unexpected={unexpected}, duplicate={duplicate}"
                )

            member_values = {column: {} for column in columns}
            previous_date: date | None = None
            for row_number, row in enumerate(rows[1:], 2):
                if not row or not any(cell.strip() for cell in row):
                    continue
                if len(row) != len(rows[0]):
                    raise SnapshotFetchError(
                        f"{request.request_id}: ZIP member {name!r} row "
                        f"{row_number} has {len(row)} columns, expected "
                        f"{len(rows[0])}"
                    )
                raw_date = row[0].strip()
                try:
                    observation_date = date.fromisoformat(raw_date)
                except ValueError as exc:
                    raise SnapshotFetchError(
                        f"{request.request_id}: ZIP member {name!r} row "
                        f"{row_number} has invalid observation_date "
                        f"{raw_date!r}"
                    ) from exc
                if observation_date.isoformat() != raw_date:
                    raise SnapshotFetchError(
                        f"{request.request_id}: ZIP member {name!r} row "
                        f"{row_number} observation_date is not canonical"
                    )
                if previous_date is not None and observation_date <= previous_date:
                    raise SnapshotFetchError(
                        f"{request.request_id}: ZIP member {name!r} dates "
                        "are not strictly ascending and unique"
                    )
                previous_date = observation_date
                all_dates.add(raw_date)
                for column, raw_value in zip(columns, row[1:], strict=True):
                    member_values[column][raw_date] = raw_value.strip()

            values_by_column.update(member_values)
            member_records.append(
                {
                    "name": name,
                    "sha256": _sha256_bytes(member_body),
                    "byte_count": len(member_body),
                    "columns": columns,
                }
            )

    if csv_member_count == 0:
        raise SnapshotFetchError(
            f"{request.request_id}: ALFRED ZIP contains no CSV members"
        )
    missing = [column for column in expected_columns if column not in values_by_column]
    if missing:
        raise SnapshotFetchError(
            f"{request.request_id}: ALFRED ZIP is missing requested vintage "
            f"columns {missing}"
        )

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["observation_date", *expected_columns])
    for observation_date in sorted(all_dates):
        writer.writerow(
            [
                observation_date,
                *[
                    values_by_column[column].get(observation_date, "")
                    for column in expected_columns
                ],
            ]
        )
    normalized_body = output.getvalue().encode("utf-8")
    # Apply the ordinary strict validator before returning normalized bytes.
    validate_alfred_csv(normalized_body, request)
    return normalized_body, {
        "normalization": ZIP_NORMALIZATION,
        "archive_format": "zip",
        "members": member_records,
    }


def _normalise_alfred_response(
    response: HttpResult,
    request: SnapshotRequest,
) -> tuple[HttpResult, bytes | None, dict[str, Any] | None]:
    """Return a canonical CSV response and optional preserved transport facts."""

    content_type = (
        str(response.headers.get("content-type") or "")
        .split(";", 1)[0]
        .strip()
        .lower()
    )
    if content_type == "application/csv":
        return response, None, None
    if content_type not in {"application/zip", "application/x-zip-compressed"}:
        raise SnapshotFetchError(
            f"{request.request_id}: expected application/csv or a supported "
            f"ALFRED ZIP response, received {content_type!r}"
        )

    normalized_body, details = _normalise_alfred_zip(response.body, request)
    normalized_headers = dict(response.headers)
    normalized_headers["content-type"] = "application/csv"
    transport = {
        **details,
        "content_type": content_type,
        "sha256": _sha256_bytes(response.body),
        "byte_count": len(response.body),
        "relative_path": "response.transport.zip",
    }
    return (
        HttpResult(
            status_code=response.status_code,
            headers=normalized_headers,
            body=normalized_body,
        ),
        response.body,
        transport,
    )


class RateLimiter:
    """Thread-safe limiter for request-start times."""

    def __init__(
        self,
        requests_per_second: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 0 < requests_per_second <= MAX_REQUESTS_PER_SECOND:
            raise ValueError(
                "requests_per_second must be greater than zero and no more "
                f"than {MAX_REQUESTS_PER_SECOND}"
            )
        self._interval = 1.0 / requests_per_second
        self._clock = clock
        self._sleep = sleep
        self._next_start = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = self._clock()
            delay = self._next_start - now
            if delay > 0:
                self._sleep(delay)
                now = self._clock()
            self._next_start = max(now, self._next_start) + self._interval


def _default_http_get(url: str, timeout_seconds: float) -> HttpResult:
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return HttpResult(
                status_code=int(response.status),
                headers={key.lower(): value for key, value in response.headers.items()},
                body=response.read(),
            )
    except HTTPError as exc:
        return HttpResult(
            status_code=int(exc.code),
            headers={key.lower(): value for key, value in exc.headers.items()},
            body=exc.read(),
        )


def _retry_delay(headers: Mapping[str, str], attempt: int) -> float:
    raw_retry_after = str(headers.get("retry-after") or "").strip()
    if raw_retry_after:
        try:
            return min(max(float(raw_retry_after), 0.0), 60.0)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(raw_retry_after)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                delay = (
                    parsed.astimezone(timezone.utc) - datetime.now(timezone.utc)
                ).total_seconds()
                return min(max(delay, 0.0), 60.0)
            except (TypeError, ValueError):
                pass
    return min(float(2**attempt), 30.0)


def _download_with_retry(
    request: SnapshotRequest,
    *,
    http_get: Callable[[str, float], HttpResult],
    limiter: RateLimiter,
    timeout_seconds: float,
    max_retries: int,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[HttpResult, str]:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        limiter.acquire()
        retrieved_at = _utc_now()
        try:
            response = http_get(request.canonical_url, timeout_seconds)
        except (OSError, TimeoutError, URLError) as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            sleep(min(float(2**attempt), 30.0))
            continue
        response = HttpResult(
            status_code=response.status_code,
            headers={
                str(key).lower(): str(value) for key, value in response.headers.items()
            },
            body=response.body,
        )
        if response.status_code == 200:
            return response, retrieved_at
        if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
            sleep(_retry_delay(response.headers, attempt))
            continue
        raise SnapshotFetchError(
            f"{request.request_id}: ALFRED returned HTTP "
            f"{response.status_code}; no current-value fallback is permitted"
        )
    raise SnapshotFetchError(
        f"{request.request_id}: ALFRED request failed after "
        f"{max_retries + 1} attempts: {last_error}"
    ) from last_error


def _resolve_relative_path(base: Path, relative_path: str) -> Path:
    candidate = (base / relative_path).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError as exc:
        raise SnapshotFetchError(
            f"Raw artifact path escapes snapshot directory: {relative_path!r}"
        ) from exc
    return candidate


def _write_atomic_request_cache(
    *,
    output_dir: Path,
    request: SnapshotRequest,
    response: HttpResult,
    retrieved_at_utc: str,
    validation: CsvValidation,
    transport_body: bytes | None = None,
    transport_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw_path = _resolve_relative_path(output_dir, request.raw_relative_path)
    request_dir = raw_path.parent
    request_dir.parent.mkdir(parents=True, exist_ok=True)
    if request_dir.exists():
        raise FileExistsError(f"Immutable request cache already exists: {request_dir}")

    temporary_dir = Path(tempfile.mkdtemp(prefix=".request-", dir=request_dir.parent))
    try:
        temporary_raw = temporary_dir / "response.csv"
        with temporary_raw.open("wb") as handle:
            handle.write(response.body)
            handle.flush()
            os.fsync(handle.fileno())
        content_type = (
            str(response.headers.get("content-type") or "")
            .split(
                ";",
                1,
            )[0]
            .strip()
            .lower()
        )
        metadata = {
            "request_id": request.request_id,
            "canonical_url": request.canonical_url,
            "raw_sha256": _sha256_bytes(response.body),
            "byte_count": len(response.body),
            "content_type": content_type,
            "retrieved_at_utc": retrieved_at_utc,
            "status_code": response.status_code,
            "validation": {
                "row_count": validation.row_count,
                "column_nonempty_counts": list(validation.column_nonempty_counts),
                "first_observation_date": validation.first_observation_date,
                "last_observation_date": validation.last_observation_date,
            },
        }
        if transport_body is not None or transport_metadata is not None:
            if transport_body is None or transport_metadata is None:
                raise SnapshotFetchError(
                    f"{request.request_id}: transport body and metadata "
                    "must be supplied together"
                )
            expected_transport_sha = _sha256_bytes(transport_body)
            if (
                transport_metadata.get("sha256") != expected_transport_sha
                or transport_metadata.get("byte_count") != len(transport_body)
                or transport_metadata.get("relative_path")
                != "response.transport.zip"
            ):
                raise SnapshotFetchError(
                    f"{request.request_id}: transport metadata does not "
                    "match the preserved response"
                )
            temporary_transport = temporary_dir / "response.transport.zip"
            with temporary_transport.open("wb") as handle:
                handle.write(transport_body)
                handle.flush()
                os.fsync(handle.fileno())
            metadata["transport"] = dict(transport_metadata)
        temporary_metadata = temporary_dir / "response.json"
        metadata_bytes = (
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        with temporary_metadata.open("wb") as handle:
            handle.write(metadata_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary_dir, request_dir)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    return metadata


def _load_cached_request(
    *,
    output_dir: Path,
    request: SnapshotRequest,
) -> dict[str, Any]:
    raw_path = _resolve_relative_path(output_dir, request.raw_relative_path)
    metadata_path = raw_path.with_name("response.json")
    if not raw_path.is_file() or not metadata_path.is_file():
        raise SnapshotFetchError(
            f"{request.request_id}: incomplete request cache; choose a new "
            "output directory rather than repairing provenance in place"
        )
    metadata = _load_json_object(metadata_path, label="request cache metadata")
    if (
        metadata.get("request_id") != request.request_id
        or metadata.get("canonical_url") != request.canonical_url
        or metadata.get("status_code") != 200
        or metadata.get("content_type") != "application/csv"
    ):
        raise SnapshotFetchError(
            f"{request.request_id}: cached request metadata does not match"
        )
    body = raw_path.read_bytes()
    if metadata.get("raw_sha256") != _sha256_bytes(body) or metadata.get(
        "byte_count"
    ) != len(body):
        raise SnapshotFetchError(
            f"{request.request_id}: cached raw response hash/size mismatch"
        )
    transport = metadata.get("transport")
    if transport is not None:
        if not isinstance(transport, Mapping):
            raise SnapshotFetchError(
                f"{request.request_id}: cached transport metadata is invalid"
            )
        transport_relative_path = transport.get("relative_path")
        if transport_relative_path != "response.transport.zip":
            raise SnapshotFetchError(
                f"{request.request_id}: cached transport path is invalid"
            )
        transport_path = raw_path.with_name(transport_relative_path)
        if not transport_path.is_file():
            raise SnapshotFetchError(
                f"{request.request_id}: preserved transport response is missing"
            )
        transport_body = transport_path.read_bytes()
        if (
            transport.get("sha256") != _sha256_bytes(transport_body)
            or transport.get("byte_count") != len(transport_body)
        ):
            raise SnapshotFetchError(
                f"{request.request_id}: cached transport hash/size mismatch"
            )
        reconstructed, _, reconstructed_transport = _normalise_alfred_response(
            HttpResult(
                status_code=200,
                headers={"content-type": str(transport.get("content_type") or "")},
                body=transport_body,
            ),
            request,
        )
        if (
            reconstructed.body != body
            or reconstructed_transport != dict(transport)
        ):
            raise SnapshotFetchError(
                f"{request.request_id}: cached transport no longer "
                "reconstructs the normalized CSV"
            )
    observed = validate_alfred_csv(body, request)
    expected_validation = metadata.get("validation")
    current_validation = {
        "row_count": observed.row_count,
        "column_nonempty_counts": list(observed.column_nonempty_counts),
        "first_observation_date": observed.first_observation_date,
        "last_observation_date": observed.last_observation_date,
    }
    if expected_validation != current_validation:
        raise SnapshotFetchError(
            f"{request.request_id}: cached CSV validation facts changed"
        )
    return metadata


def _request_manifest_record(
    request: SnapshotRequest,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    record = {
        "request_id": request.request_id,
        "source_key": request.source_key,
        "series_id": request.series_id,
        "vintage_dates": list(request.vintage_dates),
        "cosd": list(request.cosd),
        "coed": list(request.coed),
        "canonical_url": request.canonical_url,
        "raw_relative_path": request.raw_relative_path,
        "raw_sha256": metadata["raw_sha256"],
        "byte_count": metadata["byte_count"],
        "content_type": metadata["content_type"],
        "retrieved_at_utc": metadata["retrieved_at_utc"],
        "status_code": metadata["status_code"],
        "validation": metadata["validation"],
    }
    if "transport" in metadata:
        record["transport"] = metadata["transport"]
    return record


def _write_new_file_atomically(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_complete_manifest(
    *,
    manifest_path: Path,
    output_dir: Path,
    registry_path: Path,
    populations: Sequence[Mapping[str, Any]],
    meetings: Sequence[Mapping[str, Any]],
    requests: Sequence[SnapshotRequest],
) -> dict[str, Any]:
    manifest = _load_json_object(manifest_path, label="snapshot manifest")
    validate_manifest_integrity(manifest)
    if (
        manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
        or manifest.get("source_interface") != SOURCE_INTERFACE
        or manifest.get("status") != "complete"
    ):
        raise SnapshotFetchError("Existing snapshot manifest is not complete")
    registry = manifest.get("registry")
    if not isinstance(registry, Mapping) or (
        registry.get("path") != str(registry_path)
        or registry.get("sha256") != sha256_file(registry_path)
    ):
        raise SnapshotFetchError(
            "Existing snapshot manifest uses a different source registry"
        )
    if manifest.get("meetings") != list(meetings):
        raise SnapshotFetchError(
            "Existing snapshot manifest uses a different meeting inventory"
        )
    if manifest.get("populations") != list(populations):
        raise SnapshotFetchError(
            "Existing snapshot manifest uses different population artifacts"
        )
    raw_records = manifest.get("requests")
    if not isinstance(raw_records, list):
        raise SnapshotFetchError("Existing snapshot manifest has no requests")
    records = {
        str(record.get("request_id")): record
        for record in raw_records
        if isinstance(record, Mapping)
    }
    expected_ids = [request.request_id for request in requests]
    if list(records) != expected_ids:
        raise SnapshotFetchError("Existing snapshot manifest request inventory differs")
    for request in requests:
        metadata = _load_cached_request(
            output_dir=output_dir,
            request=request,
        )
        if records[request.request_id] != _request_manifest_record(
            request,
            metadata,
        ):
            raise SnapshotFetchError(
                f"{request.request_id}: manifest record differs from cache"
            )
    return manifest


def fetch_source_snapshots(
    *,
    registry_path: str | Path,
    population_paths: Sequence[str | Path],
    output_dir: str | Path,
    resume: bool = False,
    max_workers: int = MAX_CONCURRENCY,
    requests_per_second: float = MAX_REQUESTS_PER_SECOND,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    expected_vintage_count: int | None = None,
    http_get: Callable[[str, float], HttpResult] = _default_http_get,
    retry_sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Fetch, validate, cache, and seal the complete raw snapshot inventory."""

    if isinstance(max_workers, bool) or not 1 <= max_workers <= MAX_CONCURRENCY:
        raise ValueError(f"max_workers must be 1..{MAX_CONCURRENCY}")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if isinstance(max_retries, bool) or max_retries < 0:
        raise ValueError("max_retries must be a non-negative integer")
    limiter = RateLimiter(requests_per_second)

    resolved_registry = Path(registry_path).expanduser().resolve()
    registry_payload, sources = load_source_registry(resolved_registry)
    if expected_vintage_count is None:
        population_count = len(population_paths)
        if population_count == 1:
            expected_vintage_count = 13
        elif population_count == 2:
            expected_vintage_count = CANONICAL_VINTAGE_COUNT
        else:
            raise ValueError(
                "Cannot infer the expected vintage count: pass one population "
                "for 13 vintages, two populations for 26, or set "
                "--expected-vintage-count explicitly"
            )
    elif expected_vintage_count not in {
        CHECKPOINT_EVAL_VINTAGE_COUNT,
        13,
        CANONICAL_VINTAGE_COUNT,
    }:
        raise ValueError("expected_vintage_count must be 11, 13, or 26")
    meetings, populations = load_meetings(
        population_paths,
        expected_vintage_count=expected_vintage_count,
    )
    requests = build_snapshot_requests(sources, meetings)
    destination = Path(output_dir).expanduser().resolve()
    manifest_path = destination / "snapshot_manifest.json"

    if manifest_path.exists():
        if not resume:
            raise FileExistsError(
                f"Snapshot manifest already exists; pass --resume only to "
                f"verify and reuse it: {manifest_path}"
            )
        return _validate_complete_manifest(
            manifest_path=manifest_path,
            output_dir=destination,
            registry_path=resolved_registry,
            populations=populations,
            meetings=meetings,
            requests=requests,
        )
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise FileExistsError(f"Snapshot output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()

    def fetch_one(snapshot_request: SnapshotRequest) -> dict[str, Any]:
        request_path = _resolve_relative_path(
            destination,
            snapshot_request.raw_relative_path,
        )
        if request_path.parent.exists():
            if not resume:
                raise FileExistsError(
                    f"Immutable request cache already exists: {request_path.parent}"
                )
            metadata = _load_cached_request(
                output_dir=destination,
                request=snapshot_request,
            )
        else:
            response, retrieved_at = _download_with_retry(
                snapshot_request,
                http_get=http_get,
                limiter=limiter,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                sleep=retry_sleep,
            )
            response, transport_body, transport_metadata = (
                _normalise_alfred_response(response, snapshot_request)
            )
            validation = validate_alfred_csv(
                response.body,
                snapshot_request,
            )
            metadata = _write_atomic_request_cache(
                output_dir=destination,
                request=snapshot_request,
                response=response,
                retrieved_at_utc=retrieved_at,
                validation=validation,
                transport_body=transport_body,
                transport_metadata=transport_metadata,
            )
        return _request_manifest_record(snapshot_request, metadata)

    records_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_one, snapshot_request): snapshot_request
            for snapshot_request in requests
        }
        try:
            for future in as_completed(futures):
                record = future.result()
                records_by_id[str(record["request_id"])] = record
        except Exception:
            for future in futures:
                future.cancel()
            raise
    ordered_records = [
        records_by_id[snapshot_request.request_id] for snapshot_request in requests
    ]

    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source_interface": SOURCE_INTERFACE,
        "status": "complete",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "registry": {
            "path": str(resolved_registry),
            "sha256": sha256_file(resolved_registry),
            "schema_version": registry_payload.get("schema_version"),
            "registry_id": registry_payload.get("registry_id"),
        },
        "populations": populations,
        "meetings": meetings,
        "series_count": len(sources),
        "vintage_count": len(meetings),
        "request_count": len(ordered_records),
        "max_vintages_per_request": MAX_VINTAGES_PER_REQUEST,
        "requests": ordered_records,
    }
    sealed = seal_manifest(payload)
    manifest_bytes = (
        json.dumps(sealed, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_new_file_atomically(manifest_path, manifest_bytes)
    return sealed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch immutable, keyless ALFRED graph CSV snapshots for a frozen "
            "LOO or common-checkpoint evaluation population."
        )
    )
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument(
        "--population",
        action="append",
        required=True,
        type=Path,
        help="Frozen population JSON; repeat for pilot and formal populations.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--expected-vintage-count",
        type=int,
        choices=(
            CHECKPOINT_EVAL_VINTAGE_COUNT,
            13,
            CANONICAL_VINTAGE_COUNT,
        ),
        default=None,
        help=(
            "Expected unique meetings. Pass 11 for the clean checkpoint "
            "comparison; by default this is inferred as 13 for one "
            "--population and 26 for two."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_CONCURRENCY,
        choices=range(1, MAX_CONCURRENCY + 1),
    )
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=MAX_REQUESTS_PER_SECOND,
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = fetch_source_snapshots(
            registry_path=args.registry,
            population_paths=args.population,
            output_dir=args.output_dir,
            resume=args.resume,
            max_workers=args.max_workers,
            requests_per_second=args.requests_per_second,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
            expected_vintage_count=args.expected_vintage_count,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        SnapshotFetchError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    print(f"status={manifest['status']}")
    print(f"series_count={manifest['series_count']}")
    print(f"vintage_count={manifest['vintage_count']}")
    print(f"request_count={manifest['request_count']}")
    print(
        "manifest="
        f"{Path(args.output_dir).expanduser().resolve() / 'snapshot_manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
