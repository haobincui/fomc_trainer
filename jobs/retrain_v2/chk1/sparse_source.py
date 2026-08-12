"""Chk1-local sparse ALFRED acquisition and replayable D-1 ledgers.

The shared LOO fetcher and ledger deliberately require a dense inventory.  A
historical ALFRED request can instead return either HTTP 404 or HTTP 200 with
the current-vintage column silently substituted for the requested vintage.
This module contains the narrower chk1 policy for those cases:

* requests start in the same at-most-twelve-vintage batches as the canonical
  fetcher;
* a 404 or a well-formed response which does not echo the exact requested
  vintage is bisected until singleton requests identify the affected cells;
* only an exact, strictly validated HTTP-200 response is usable evidence; and
* a meeting/topic is excluded only when every configured source has a sealed
  unusable singleton response.

All artifacts live in a caller-selected chk1 directory.  Nothing in this file
changes the shared fetcher, registry, roster, or canonical dense validator.
The intentionally private imports below are thin adapters around the frozen
canonical transport and sampling implementation; replay tests pin that
coupling so a shared refactor fails closed rather than drifting silently.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode

from jobs.main.fetch_loo_source_snapshots import (
    ALFRED_GRAPH_CSV_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_REQUESTS_PER_SECOND,
    RETRYABLE_STATUS_CODES,
    HttpResult,
    RateLimiter,
    SnapshotFetchError,
    SnapshotRequest,
    _default_http_get,
    _load_cached_request,
    _normalise_alfred_response,
    _request_manifest_record,
    _retry_delay,
    _write_atomic_request_cache,
    build_snapshot_requests,
    load_source_registry,
    validate_alfred_csv,
)
from open_r1.provenance import sha256_file, sha256_text
from open_r1.validator.loo_generation_spec import (
    ManifestIntegrityError,
    seal_manifest,
    validate_manifest_integrity,
)
from open_r1.validator.loo_ledger import (
    AVAILABILITY_EVIDENCE_TYPE,
    LEDGER_ROW_SCHEMA_VERSION,
    LooSamplingWindowEmptyError,
    SAMPLING_POLICY_VERSION,
    SOURCE_EVIDENCE_SCHEMA_VERSION,
    SOURCE_INTERFACE,
    _format_timestamp,
    _load_registry,
    _load_roster,
    _parse_csv_observations,
    _reject_forbidden_keys,
    _sample_observations,
    compute_request_id,
    decision_identity_timestamp,
    information_as_of_date,
)

from .source_data import stable_sample_id
from .topic_styles import topic_for_ledger_indicator


SNAPSHOT_MANIFEST_SCHEMA_VERSION = "chk1-sparse-source-snapshot-manifest-v1"
REQUEST_CACHE_SCHEMA_VERSION = "chk1-sparse-request-cache-v1"
LEDGER_MANIFEST_SCHEMA_VERSION = "chk1-sparse-loo-ledger-manifest-v1"
COVERAGE_SCHEMA_VERSION = "chk1-sparse-loo-coverage-v1"
SAMPLE_EXCLUSION_SCHEMA_VERSION = "chk1-source-exclusion-v1"

OUTCOME_AVAILABLE = "available"
OUTCOME_UNUSABLE = "unusable"
OUTCOME_SPLIT = "split"
REASON_HTTP_404 = "alfred_vintage_http_404"
REASON_NOT_ECHOED = "exact_vintage_not_echoed"
REASON_TRANSPORT_RETRY_EXHAUSTED = "transport_retry_exhausted"
REASON_NO_SAFE_NUMERIC_OBSERVATION = "no_safe_numeric_observation"
REASON_NO_OBSERVATION_IN_SAMPLING_WINDOW = (
    "no_observation_in_frozen_sampling_window"
)
SAMPLE_EXCLUSION_REASON = "no_proven_d1_vintage_evidence"

_ALLOWED_SPLITS = frozenset({"train", "eval", "test"})


class SparseSourceError(ValueError):
    """Raised when chk1 sparse evidence cannot be proved safely."""


class SparseSourceIntegrityError(SparseSourceError):
    """Raised when a sealed sparse artifact cannot be replayed exactly."""


class SparseSourceTransportError(SparseSourceError):
    """Raised when an ALFRED transport failure survives every retry."""

    def __init__(
        self,
        *,
        request_id: str,
        attempt_count: int,
        retrieved_at_utc: str,
        error: BaseException,
    ) -> None:
        self.request_id = request_id
        self.attempt_count = attempt_count
        self.retrieved_at_utc = retrieved_at_utc
        self.error_type = type(error).__name__
        self.error_message = str(error)
        super().__init__(
            f"{request_id}: ALFRED transport failed after {attempt_count} "
            f"attempts: {error}"
        )


@dataclass(frozen=True)
class _Attempt:
    request: SnapshotRequest
    record: dict[str, Any]
    outcome: str
    reason_code: str | None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_text(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(_canonical_json(dict(row)) + "\n" for row in rows)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _parse_date(value: object, *, label: str) -> date:
    text = str(value or "").strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise SparseSourceError(f"{label} is not a canonical date: {value!r}") from exc
    if parsed.isoformat() != text:
        raise SparseSourceError(f"{label} is not canonical YYYY-MM-DD: {value!r}")
    return parsed


def _parse_timestamp(value: object, *, label: str) -> datetime:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise SparseSourceIntegrityError(f"{label} is not an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SparseSourceIntegrityError(f"{label} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _normalise_meetings(
    meeting_dates: Sequence[str],
    *,
    split: str,
) -> list[dict[str, str]]:
    if split not in _ALLOWED_SPLITS:
        raise SparseSourceError(f"Unsupported split {split!r}")
    parsed = [_parse_date(item, label="meeting_date") for item in meeting_dates]
    if not parsed or parsed != sorted(set(parsed)):
        raise SparseSourceError(
            "meeting_dates must be non-empty, unique, and ascending"
        )
    return [
        {
            "meeting_date": meeting.isoformat(),
            "information_as_of_date": information_as_of_date(meeting).isoformat(),
            "split": split,
        }
        for meeting in parsed
    ]


def _safe_relative_file(base: Path, relative: object, *, label: str) -> Path:
    text = str(relative or "").strip()
    pure = PurePosixPath(text)
    if not text or pure.is_absolute() or ".." in pure.parts:
        raise SparseSourceIntegrityError(f"{label} is not a safe relative path")
    candidate = base.joinpath(*pure.parts).resolve()
    resolved_base = base.resolve()
    if not candidate.is_relative_to(resolved_base):
        raise SparseSourceIntegrityError(f"{label} escapes its artifact directory")
    current = resolved_base
    for part in candidate.relative_to(resolved_base).parts:
        current = current / part
        if current.is_symlink():
            raise SparseSourceIntegrityError(f"{label} traverses a symlink")
    if not candidate.is_file():
        raise SparseSourceIntegrityError(f"{label} is not a regular file")
    return candidate


def _write_new_bytes(path: Path, body: bytes) -> None:
    """Atomically create one immutable file without an overwrite branch."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Immutable artifact already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_text(path: Path, text: str) -> None:
    _write_new_bytes(path, text.encode("utf-8"))


def _request_directory(output_dir: Path, request: SnapshotRequest) -> Path:
    relative = PurePosixPath(request.raw_relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise SparseSourceError("Snapshot request has an unsafe raw path")
    path = output_dir.joinpath(*relative.parts).parent.resolve()
    if not path.is_relative_to(output_dir.resolve()):
        raise SparseSourceError("Snapshot request raw path escapes output directory")
    return path


def _subrequest(request: SnapshotRequest, indexes: Sequence[int]) -> SnapshotRequest:
    if not indexes or list(indexes) != sorted(set(indexes)):
        raise SparseSourceError("Subrequest indexes must be non-empty and ordered")
    vintages = tuple(request.vintage_dates[index] for index in indexes)
    starts = tuple(request.cosd[index] for index in indexes)
    ends = tuple(request.coed[index] for index in indexes)
    params = {
        "id": ",".join([request.series_id] * len(vintages)),
        "cosd": ",".join(starts),
        "coed": ",".join(ends),
        "vintage_date": ",".join(vintages),
    }
    canonical_url = f"{ALFRED_GRAPH_CSV_URL}?{urlencode(params)}"
    descriptor = {
        "source_key": request.source_key,
        "series_id": request.series_id,
        "vintage_dates": list(vintages),
        "cosd": list(starts),
        "coed": list(ends),
        "canonical_url": canonical_url,
    }
    request_id = compute_request_id(descriptor)
    relative_path = (
        Path("raw") / "alfred" / request.source_key / request_id / "response.csv"
    ).as_posix()
    return SnapshotRequest(
        request_id=request_id,
        source_key=request.source_key,
        series_id=request.series_id,
        vintage_dates=vintages,
        cosd=starts,
        coed=ends,
        canonical_url=canonical_url,
        raw_relative_path=relative_path,
    )


def _expected_header(request: SnapshotRequest) -> list[str]:
    return ["observation_date"] + [
        f"{request.series_id}_{value.replace('-', '')}"
        for value in request.vintage_dates
    ]


def _not_echoed_header(body: bytes, request: SnapshotRequest) -> list[str] | None:
    """Recognise only a well-formed same-series vintage substitution.

    Values from this response are validated only enough to distinguish a
    structured ALFRED response from malformed transport.  They are never
    returned to the ledger or parsed as evidence.
    """

    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SparseSourceError(
            f"{request.request_id}: non-UTF-8 HTTP-200 response"
        ) from exc
    if "\x00" in text:
        raise SparseSourceError(f"{request.request_id}: response contains NUL")
    try:
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error as exc:
        raise SparseSourceError(
            f"{request.request_id}: malformed HTTP-200 CSV"
        ) from exc
    if not rows or len(rows[0]) < 2 or rows[0][0] != "observation_date":
        raise SparseSourceError(f"{request.request_id}: malformed HTTP-200 CSV header")
    header = rows[0]
    if header == _expected_header(request):
        return None
    if len(header) != len(set(header)):
        raise SparseSourceError(f"{request.request_id}: duplicate HTTP-200 CSV columns")
    column_pattern = re.compile(
        rf"^{re.escape(request.series_id)}_(?P<vintage>[0-9]{{8}})$"
    )
    for column in header[1:]:
        match = column_pattern.fullmatch(column)
        if match is None:
            raise SparseSourceError(
                f"{request.request_id}: HTTP-200 response uses the wrong series"
            )
        raw_vintage = match.group("vintage")
        try:
            parsed_vintage = datetime.strptime(raw_vintage, "%Y%m%d").date()
        except ValueError as exc:
            raise SparseSourceError(
                f"{request.request_id}: invalid echoed vintage column"
            ) from exc
        if parsed_vintage.strftime("%Y%m%d") != raw_vintage:
            raise SparseSourceError(
                f"{request.request_id}: non-canonical echoed vintage column"
            )

    previous: date | None = None
    nonempty_value_count = 0
    latest_allowed = max(date.fromisoformat(value) for value in request.coed)
    for row_number, row in enumerate(rows[1:], 2):
        if not row or not any(cell.strip() for cell in row):
            continue
        if len(row) != len(header):
            raise SparseSourceError(
                f"{request.request_id}: malformed CSV row {row_number}"
            )
        observation = _parse_date(
            row[0].strip(), label=f"{request.request_id} row {row_number} date"
        )
        if previous is not None and observation <= previous:
            raise SparseSourceError(
                f"{request.request_id}: observation dates are not ascending"
            )
        if observation > latest_allowed:
            raise SparseSourceError(
                f"{request.request_id}: substituted response contains future values"
            )
        previous = observation
        for raw_value in row[1:]:
            # The substituted values are deliberately opaque: this branch
            # exists only to prove that ALFRED did not echo the requested
            # vintage.  No value parsing or evidence projection is allowed.
            nonempty_value_count += int(bool(raw_value.strip()))
    if previous is None or nonempty_value_count == 0:
        raise SparseSourceError(
            f"{request.request_id}: substituted response has no numeric rows"
        )
    return header


def _normalised_content_type(headers: Mapping[str, str]) -> str:
    return str(headers.get("content-type") or "").split(";", 1)[0].strip().lower()


def _download(
    request: SnapshotRequest,
    *,
    http_get: Callable[[str, float], HttpResult],
    limiter: Any,
    timeout_seconds: float,
    max_retries: int,
    retry_sleep: Callable[[float], None],
    utc_now: Callable[[], str],
) -> tuple[HttpResult, str]:
    last_error: BaseException | None = None
    last_retrieved_at: str | None = None
    for attempt in range(max_retries + 1):
        limiter.acquire()
        retrieved_at = utc_now()
        last_retrieved_at = retrieved_at
        try:
            raw = http_get(request.canonical_url, timeout_seconds)
        except (OSError, TimeoutError, URLError) as exc:
            last_error = exc
            if attempt < max_retries:
                retry_sleep(min(float(2**attempt), 30.0))
                continue
            break
        response = HttpResult(
            status_code=int(raw.status_code),
            headers={
                str(key).lower(): str(value) for key, value in raw.headers.items()
            },
            body=bytes(raw.body),
        )
        if response.status_code in {200, 404}:
            return response, retrieved_at
        if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
            retry_sleep(_retry_delay(response.headers, attempt))
            continue
        raise SparseSourceError(
            f"{request.request_id}: ALFRED returned HTTP {response.status_code}; "
            "only an exact singleton 404 is an excludable outcome"
        )
    if last_error is None or last_retrieved_at is None:
        raise SparseSourceError(
            f"{request.request_id}: ALFRED request failed without a transport result"
        )
    raise SparseSourceTransportError(
        request_id=request.request_id,
        attempt_count=max_retries + 1,
        retrieved_at_utc=last_retrieved_at,
        error=last_error,
    ) from last_error


def _negative_raw_relative_path(request: SnapshotRequest) -> str:
    return str(PurePosixPath(request.raw_relative_path).with_name("response.unusable"))


def _transport_raw_relative_path(request: SnapshotRequest) -> str:
    return str(
        PurePosixPath(request.raw_relative_path).with_name("response.transport.json")
    )


def _write_transport_split_cache(
    *,
    output_dir: Path,
    request: SnapshotRequest,
    failure: SparseSourceTransportError,
) -> dict[str, Any]:
    if len(request.vintage_dates) <= 1:
        raise SparseSourceError(
            f"{request.request_id}: singleton transport failures must abort"
        )
    request_dir = _request_directory(output_dir, request)
    request_dir.parent.mkdir(parents=True, exist_ok=True)
    if request_dir.exists():
        raise FileExistsError(f"Immutable request cache already exists: {request_dir}")
    transport_failure = {
        "attempt_count": failure.attempt_count,
        "error_message": failure.error_message,
        "error_type": failure.error_type,
    }
    body = _json_text(transport_failure).encode("utf-8")
    relative_raw = _transport_raw_relative_path(request)
    metadata = {
        "schema_version": REQUEST_CACHE_SCHEMA_VERSION,
        "request_id": request.request_id,
        "source_key": request.source_key,
        "series_id": request.series_id,
        "vintage_dates": list(request.vintage_dates),
        "cosd": list(request.cosd),
        "coed": list(request.coed),
        "canonical_url": request.canonical_url,
        "outcome": OUTCOME_SPLIT,
        "reason_code": REASON_TRANSPORT_RETRY_EXHAUSTED,
        "status_code": None,
        "content_type": "application/json",
        "retrieved_at_utc": failure.retrieved_at_utc,
        "raw_relative_path": relative_raw,
        "raw_sha256": hashlib.sha256(body).hexdigest(),
        "byte_count": len(body),
        "expected_header": _expected_header(request),
        "observed_header": [],
        "transport_failure": transport_failure,
    }
    temporary_dir = Path(tempfile.mkdtemp(prefix=".request-", dir=request_dir.parent))
    try:
        raw_path = temporary_dir / "response.transport.json"
        with raw_path.open("wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        with (temporary_dir / "response.json").open("w", encoding="utf-8") as handle:
            handle.write(_json_text(metadata))
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary_dir, request_dir)
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    return metadata


def _load_transport_split_cache(
    *,
    output_dir: Path,
    request: SnapshotRequest,
) -> dict[str, Any]:
    request_dir = _request_directory(output_dir, request)
    metadata_path = request_dir / "response.json"
    raw_path = request_dir / "response.transport.json"
    if not metadata_path.is_file() or not raw_path.is_file():
        raise SparseSourceIntegrityError(
            f"{request.request_id}: incomplete transport-split cache"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SparseSourceIntegrityError(
            f"{request.request_id}: invalid transport-split metadata"
        ) from exc
    expected_descriptor = {
        "request_id": request.request_id,
        "source_key": request.source_key,
        "series_id": request.series_id,
        "vintage_dates": list(request.vintage_dates),
        "cosd": list(request.cosd),
        "coed": list(request.coed),
        "canonical_url": request.canonical_url,
        "outcome": OUTCOME_SPLIT,
        "reason_code": REASON_TRANSPORT_RETRY_EXHAUSTED,
        "status_code": None,
        "content_type": "application/json",
        "expected_header": _expected_header(request),
        "observed_header": [],
        "raw_relative_path": _transport_raw_relative_path(request),
    }
    mismatched = {
        key
        for key, expected in expected_descriptor.items()
        if metadata.get(key) != expected
    }
    if (
        metadata.get("schema_version") != REQUEST_CACHE_SCHEMA_VERSION
        or mismatched
        or len(request.vintage_dates) <= 1
    ):
        raise SparseSourceIntegrityError(
            f"{request.request_id}: transport-split metadata mismatch "
            f"{sorted(mismatched)}"
        )
    body = raw_path.read_bytes()
    if (
        metadata.get("byte_count") != len(body)
        or metadata.get("raw_sha256") != hashlib.sha256(body).hexdigest()
    ):
        raise SparseSourceIntegrityError(
            f"{request.request_id}: transport-split body hash/size mismatch"
        )
    try:
        transport_failure = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SparseSourceIntegrityError(
            f"{request.request_id}: invalid transport-split body"
        ) from exc
    if (
        not isinstance(transport_failure, dict)
        or metadata.get("transport_failure") != transport_failure
        or set(transport_failure) != {"attempt_count", "error_message", "error_type"}
        or isinstance(transport_failure.get("attempt_count"), bool)
        or not isinstance(transport_failure.get("attempt_count"), int)
        or transport_failure["attempt_count"] < 1
        or not str(transport_failure.get("error_type") or "").strip()
    ):
        raise SparseSourceIntegrityError(
            f"{request.request_id}: invalid cached transport failure"
        )
    _parse_timestamp(metadata.get("retrieved_at_utc"), label="retrieved_at_utc")
    return metadata


def _write_negative_cache(
    *,
    output_dir: Path,
    request: SnapshotRequest,
    response: HttpResult,
    retrieved_at_utc: str,
    reason_code: str,
    observed_header: Sequence[str],
) -> dict[str, Any]:
    request_dir = _request_directory(output_dir, request)
    request_dir.parent.mkdir(parents=True, exist_ok=True)
    if request_dir.exists():
        raise FileExistsError(f"Immutable request cache already exists: {request_dir}")
    relative_raw = _negative_raw_relative_path(request)
    metadata = {
        "schema_version": REQUEST_CACHE_SCHEMA_VERSION,
        "request_id": request.request_id,
        "source_key": request.source_key,
        "series_id": request.series_id,
        "vintage_dates": list(request.vintage_dates),
        "cosd": list(request.cosd),
        "coed": list(request.coed),
        "canonical_url": request.canonical_url,
        "outcome": OUTCOME_UNUSABLE,
        "reason_code": reason_code,
        "status_code": response.status_code,
        "content_type": _normalised_content_type(response.headers),
        "retrieved_at_utc": retrieved_at_utc,
        "raw_relative_path": relative_raw,
        "raw_sha256": hashlib.sha256(response.body).hexdigest(),
        "byte_count": len(response.body),
        "expected_header": _expected_header(request),
        "observed_header": list(observed_header),
    }
    temporary_dir = Path(tempfile.mkdtemp(prefix=".request-", dir=request_dir.parent))
    try:
        raw_path = temporary_dir / "response.unusable"
        with raw_path.open("wb") as handle:
            handle.write(response.body)
            handle.flush()
            os.fsync(handle.fileno())
        with (temporary_dir / "response.json").open("w", encoding="utf-8") as handle:
            handle.write(_json_text(metadata))
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary_dir, request_dir)
    except BaseException:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    return metadata


def _load_negative_cache(
    *,
    output_dir: Path,
    request: SnapshotRequest,
) -> dict[str, Any]:
    request_dir = _request_directory(output_dir, request)
    metadata_path = request_dir / "response.json"
    raw_path = request_dir / "response.unusable"
    if not metadata_path.is_file() or not raw_path.is_file():
        raise SparseSourceIntegrityError(
            f"{request.request_id}: incomplete unusable-response cache"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SparseSourceIntegrityError(
            f"{request.request_id}: invalid unusable-response metadata"
        ) from exc
    expected_descriptor = {
        "request_id": request.request_id,
        "source_key": request.source_key,
        "series_id": request.series_id,
        "vintage_dates": list(request.vintage_dates),
        "cosd": list(request.cosd),
        "coed": list(request.coed),
        "canonical_url": request.canonical_url,
        "outcome": OUTCOME_UNUSABLE,
        "expected_header": _expected_header(request),
        "raw_relative_path": _negative_raw_relative_path(request),
    }
    mismatched = {
        key
        for key, expected in expected_descriptor.items()
        if metadata.get(key) != expected
    }
    if metadata.get("schema_version") != REQUEST_CACHE_SCHEMA_VERSION or mismatched:
        raise SparseSourceIntegrityError(
            f"{request.request_id}: unusable-response metadata mismatch {sorted(mismatched)}"
        )
    body = raw_path.read_bytes()
    if (
        metadata.get("byte_count") != len(body)
        or metadata.get("raw_sha256") != hashlib.sha256(body).hexdigest()
    ):
        raise SparseSourceIntegrityError(
            f"{request.request_id}: unusable-response body hash/size mismatch"
        )
    reason = metadata.get("reason_code")
    status = metadata.get("status_code")
    if reason == REASON_HTTP_404:
        if status != 404 or metadata.get("observed_header") != []:
            raise SparseSourceIntegrityError(
                f"{request.request_id}: invalid cached 404 classification"
            )
    elif reason == REASON_NOT_ECHOED:
        if status != 200:
            raise SparseSourceIntegrityError(
                f"{request.request_id}: invalid not-echoed status"
            )
        observed = _not_echoed_header(body, request)
        if observed is None or metadata.get("observed_header") != observed:
            raise SparseSourceIntegrityError(
                f"{request.request_id}: not-echoed classification changed"
            )
    else:
        raise SparseSourceIntegrityError(
            f"{request.request_id}: unknown unusable reason {reason!r}"
        )
    _parse_timestamp(metadata.get("retrieved_at_utc"), label="retrieved_at_utc")
    return metadata


def _attempt_record(
    *,
    request: SnapshotRequest,
    metadata: Mapping[str, Any],
    outcome: str,
    reason_code: str | None,
    parent_request_id: str | None,
    depth: int,
    terminal: bool,
) -> dict[str, Any]:
    if outcome == OUTCOME_AVAILABLE:
        base = _request_manifest_record(request, metadata)
    else:
        base = {
            key: metadata[key]
            for key in (
                "request_id",
                "source_key",
                "series_id",
                "vintage_dates",
                "cosd",
                "coed",
                "canonical_url",
                "raw_relative_path",
                "raw_sha256",
                "byte_count",
                "content_type",
                "retrieved_at_utc",
                "status_code",
                "expected_header",
                "observed_header",
            )
        }
        if outcome == OUTCOME_SPLIT:
            base["transport_failure"] = metadata["transport_failure"]
    return {
        **base,
        "outcome": outcome,
        "reason_code": reason_code,
        "parent_request_id": parent_request_id,
        "depth": depth,
        "terminal": terminal,
        "cache_metadata_relative_path": (
            PurePosixPath(request.raw_relative_path)
            .with_name("response.json")
            .as_posix()
        ),
    }


def _load_or_fetch_attempt(
    *,
    output_dir: Path,
    request: SnapshotRequest,
    resume: bool,
    allow_network: bool,
    http_get: Callable[[str, float], HttpResult],
    limiter: Any,
    timeout_seconds: float,
    max_retries: int,
    retry_sleep: Callable[[float], None],
    utc_now: Callable[[], str],
) -> _Attempt:
    request_dir = _request_directory(output_dir, request)
    if request_dir.exists():
        if not resume:
            raise FileExistsError(
                f"Immutable request cache already exists: {request_dir}"
            )
        if (request_dir / "response.csv").is_file():
            try:
                metadata = _load_cached_request(output_dir=output_dir, request=request)
            except SnapshotFetchError as exc:
                raise SparseSourceIntegrityError(str(exc)) from exc
            return _Attempt(request, dict(metadata), OUTCOME_AVAILABLE, None)
        if (request_dir / "response.transport.json").is_file():
            metadata = _load_transport_split_cache(
                output_dir=output_dir,
                request=request,
            )
            return _Attempt(
                request,
                metadata,
                OUTCOME_SPLIT,
                REASON_TRANSPORT_RETRY_EXHAUSTED,
            )
        metadata = _load_negative_cache(output_dir=output_dir, request=request)
        return _Attempt(
            request,
            metadata,
            OUTCOME_UNUSABLE,
            str(metadata["reason_code"]),
        )
    if not allow_network:
        raise SparseSourceIntegrityError(
            f"{request.request_id}: required request cache is missing"
        )

    try:
        response, retrieved_at = _download(
            request,
            http_get=http_get,
            limiter=limiter,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
            utc_now=utc_now,
        )
    except SparseSourceTransportError as exc:
        if len(request.vintage_dates) == 1:
            raise
        metadata = _write_transport_split_cache(
            output_dir=output_dir,
            request=request,
            failure=exc,
        )
        return _Attempt(
            request,
            metadata,
            OUTCOME_SPLIT,
            REASON_TRANSPORT_RETRY_EXHAUSTED,
        )
    if response.status_code == 404:
        metadata = _write_negative_cache(
            output_dir=output_dir,
            request=request,
            response=response,
            retrieved_at_utc=retrieved_at,
            reason_code=REASON_HTTP_404,
            observed_header=(),
        )
        return _Attempt(request, metadata, OUTCOME_UNUSABLE, REASON_HTTP_404)

    try:
        normalised, transport_body, transport_metadata = _normalise_alfred_response(
            response, request
        )
        validation = validate_alfred_csv(normalised.body, request)
    except SnapshotFetchError as exc:
        if _normalised_content_type(response.headers) != "application/csv":
            raise SparseSourceError(str(exc)) from exc
        observed = _not_echoed_header(response.body, request)
        if observed is None:
            raise SparseSourceError(str(exc)) from exc
        metadata = _write_negative_cache(
            output_dir=output_dir,
            request=request,
            response=response,
            retrieved_at_utc=retrieved_at,
            reason_code=REASON_NOT_ECHOED,
            observed_header=observed,
        )
        return _Attempt(request, metadata, OUTCOME_UNUSABLE, REASON_NOT_ECHOED)

    metadata = _write_atomic_request_cache(
        output_dir=output_dir,
        request=request,
        response=normalised,
        retrieved_at_utc=retrieved_at,
        validation=validation,
        transport_body=transport_body,
        transport_metadata=transport_metadata,
    )
    return _Attempt(request, dict(metadata), OUTCOME_AVAILABLE, None)


def _walk_request(
    *,
    output_dir: Path,
    request: SnapshotRequest,
    parent_request_id: str | None,
    depth: int,
    resume: bool,
    allow_network: bool,
    http_get: Callable[[str, float], HttpResult],
    limiter: Any,
    timeout_seconds: float,
    max_retries: int,
    retry_sleep: Callable[[float], None],
    utc_now: Callable[[], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    attempt = _load_or_fetch_attempt(
        output_dir=output_dir,
        request=request,
        resume=resume,
        allow_network=allow_network,
        http_get=http_get,
        limiter=limiter,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_sleep=retry_sleep,
        utc_now=utc_now,
    )
    terminal = attempt.outcome == OUTCOME_AVAILABLE or len(request.vintage_dates) == 1
    record = _attempt_record(
        request=request,
        metadata=attempt.record,
        outcome=attempt.outcome,
        reason_code=attempt.reason_code,
        parent_request_id=parent_request_id,
        depth=depth,
        terminal=terminal,
    )
    if terminal:
        outcomes = []
        for index, vintage in enumerate(request.vintage_dates):
            meeting = date.fromordinal(
                date.fromisoformat(vintage).toordinal() + 1
            ).isoformat()
            outcomes.append(
                {
                    "source_key": request.source_key,
                    "series_id": request.series_id,
                    "meeting_date": meeting,
                    "information_as_of_date": vintage,
                    "outcome": attempt.outcome,
                    "reason_code": attempt.reason_code,
                    "request_id": request.request_id,
                    "selected_batch_index": index,
                    "status_code": attempt.record["status_code"],
                    "raw_relative_path": (
                        request.raw_relative_path
                        if attempt.outcome == OUTCOME_AVAILABLE
                        else attempt.record["raw_relative_path"]
                    ),
                    "raw_sha256": attempt.record["raw_sha256"],
                    "byte_count": attempt.record["byte_count"],
                    "retrieved_at_utc": attempt.record["retrieved_at_utc"],
                }
            )
        return [record], outcomes

    midpoint = len(request.vintage_dates) // 2
    left = _subrequest(request, tuple(range(0, midpoint)))
    right = _subrequest(request, tuple(range(midpoint, len(request.vintage_dates))))
    attempts = [record]
    outcomes: list[dict[str, Any]] = []
    for child in (left, right):
        child_attempts, child_outcomes = _walk_request(
            output_dir=output_dir,
            request=child,
            parent_request_id=request.request_id,
            depth=depth + 1,
            resume=resume,
            allow_network=allow_network,
            http_get=http_get,
            limiter=limiter,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
            utc_now=utc_now,
        )
        attempts.extend(child_attempts)
        outcomes.extend(child_outcomes)
    return attempts, outcomes


def _source_context(
    registry_path: Path,
    roster_path: Path,
) -> tuple[
    dict[str, Any],
    list[Any],
    list[str],
    dict[str, dict[str, Any]],
    dict[str, list[str]],
]:
    registry_payload, source_series = load_source_registry(registry_path)
    roster = _load_roster(roster_path)
    sources, indicator_sources, _ = _load_registry(registry_path, roster=roster)
    enabled = list(
        dict.fromkeys(
            source_key
            for indicator in roster
            for source_key in indicator_sources[indicator]
        )
    )
    fetched = [item.source_key for item in source_series]
    if fetched != enabled:
        raise SparseSourceError(
            "Fetcher and ledger disagree on the enabled source inventory"
        )
    return registry_payload, source_series, roster, sources, indicator_sources


def _expected_outcome_keys(
    *, sources: Sequence[Any], meetings: Sequence[Mapping[str, str]]
) -> list[tuple[str, str]]:
    return [
        (source.source_key, meeting["meeting_date"])
        for source in sources
        for meeting in meetings
    ]


def _validate_outcome_coverage(
    outcomes: Sequence[Mapping[str, Any]],
    *,
    sources: Sequence[Any],
    meetings: Sequence[Mapping[str, str]],
) -> None:
    expected = _expected_outcome_keys(sources=sources, meetings=meetings)
    observed = [
        (str(item.get("source_key")), str(item.get("meeting_date")))
        for item in outcomes
    ]
    if len(observed) != len(set(observed)):
        raise SparseSourceIntegrityError("Terminal source-vintage outcomes overlap")
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise SparseSourceIntegrityError(
            f"Terminal source-vintage coverage differs: missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )
    for item in outcomes:
        outcome = item.get("outcome")
        if outcome == OUTCOME_AVAILABLE:
            if item.get("reason_code") is not None or item.get("status_code") != 200:
                raise SparseSourceIntegrityError("Invalid available terminal outcome")
        elif outcome == OUTCOME_UNUSABLE:
            if item.get("reason_code") not in {REASON_HTTP_404, REASON_NOT_ECHOED}:
                raise SparseSourceIntegrityError("Invalid unusable terminal outcome")
        else:
            raise SparseSourceIntegrityError(f"Unknown terminal outcome {outcome!r}")


def _snapshot_payload(
    *,
    population_id: str,
    registry_path: Path,
    registry_payload: Mapping[str, Any],
    roster_path: Path,
    meetings: Sequence[Mapping[str, str]],
    sources: Sequence[Any],
    attempts: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    started_at_utc: str,
    completed_at_utc: str,
) -> dict[str, Any]:
    available_count = sum(item.get("outcome") == OUTCOME_AVAILABLE for item in outcomes)
    return {
        "schema_version": SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "population_id": population_id,
        "source_interface": SOURCE_INTERFACE,
        "policy": _snapshot_policy(),
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
            "schema_version": registry_payload.get("schema_version"),
            "registry_id": registry_payload.get("registry_id"),
        },
        "roster": {
            "path": str(roster_path),
            "sha256": sha256_file(roster_path),
        },
        "meetings": list(meetings),
        "series_count": len(sources),
        "meeting_count": len(meetings),
        "source_vintage_count": len(outcomes),
        "available_source_vintage_count": available_count,
        "unusable_source_vintage_count": len(outcomes) - available_count,
        "request_attempt_count": len(attempts),
        "attempts": list(attempts),
        "terminal_outcomes": list(outcomes),
    }


def _snapshot_policy() -> dict[str, Any]:
    return {
        "version": "chk1-sparse-source-policy-v2",
        "batch_limit": 12,
        "split_on": [
            REASON_HTTP_404,
            REASON_NOT_ECHOED,
            REASON_TRANSPORT_RETRY_EXHAUSTED,
        ],
        "split_strategy": "recursive_bisection_to_singleton",
        "usable_response": "http_200_exact_vintage_echo_and_strict_csv",
        "current_value_fallback": "forbidden",
        "singleton_transport_failure": "abort",
        "unknown_format": "abort",
    }


def _legacy_snapshot_policy() -> dict[str, Any]:
    """The immutable policy emitted before transport-aware bisection."""

    return {
        "batch_limit": 12,
        "split_on": [REASON_HTTP_404, REASON_NOT_ECHOED],
        "split_strategy": "recursive_bisection_to_singleton",
        "usable_response": "http_200_exact_vintage_echo_and_strict_csv",
        "current_value_fallback": "forbidden",
        "unknown_transport_or_format": "abort",
    }


def acquire_sparse_source_snapshots(
    *,
    registry_file: str | Path,
    roster_file: str | Path,
    meeting_dates: Sequence[str],
    output_dir: str | Path,
    population_id: str,
    split: str = "train",
    resume: bool = False,
    http_get: Callable[[str, float], HttpResult] = _default_http_get,
    rate_limiter: Any | None = None,
    requests_per_second: float = MAX_REQUESTS_PER_SECOND,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_sleep: Callable[[float], None] = time.sleep,
    utc_now: Callable[[], str] = _utc_now,
) -> Path:
    """Acquire and seal exact D-1 outcomes for every enabled source/meeting."""

    if not str(population_id).strip():
        raise SparseSourceError("population_id must be non-empty")
    if timeout_seconds <= 0:
        raise SparseSourceError("timeout_seconds must be positive")
    if isinstance(max_retries, bool) or max_retries < 0:
        raise SparseSourceError("max_retries must be a non-negative integer")
    registry_path = Path(registry_file).expanduser().resolve()
    roster_path = Path(roster_file).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    manifest_path = destination / "snapshot_manifest.json"
    meetings = _normalise_meetings(meeting_dates, split=split)
    registry_payload, source_series, _, _, _ = _source_context(
        registry_path, roster_path
    )

    if manifest_path.exists():
        if not resume:
            raise FileExistsError(
                f"Sparse snapshot manifest already exists: {manifest_path}"
            )
        validate_sparse_snapshot_manifest(
            manifest_path,
            registry_file=registry_path,
            roster_file=roster_path,
            expected_meeting_dates=meeting_dates,
            expected_population_id=population_id,
        )
        return manifest_path
    if destination.exists() and any(destination.iterdir()) and not resume:
        raise FileExistsError(f"Sparse snapshot directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    limiter = rate_limiter or RateLimiter(requests_per_second)
    started_at = utc_now()
    roots = build_snapshot_requests(source_series, meetings)
    attempts: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for request in roots:
        request_attempts, request_outcomes = _walk_request(
            output_dir=destination,
            request=request,
            parent_request_id=None,
            depth=0,
            resume=resume,
            allow_network=True,
            http_get=http_get,
            limiter=limiter,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
            utc_now=utc_now,
        )
        attempts.extend(request_attempts)
        outcomes.extend(request_outcomes)
    _validate_outcome_coverage(outcomes, sources=source_series, meetings=meetings)
    payload = _snapshot_payload(
        population_id=str(population_id).strip(),
        registry_path=registry_path,
        registry_payload=registry_payload,
        roster_path=roster_path,
        meetings=meetings,
        sources=source_series,
        attempts=attempts,
        outcomes=outcomes,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
    )
    _write_new_text(manifest_path, _json_text(seal_manifest(payload)))
    validate_sparse_snapshot_manifest(
        manifest_path,
        registry_file=registry_path,
        roster_file=roster_path,
        expected_meeting_dates=meeting_dates,
        expected_population_id=population_id,
    )
    return manifest_path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SparseSourceIntegrityError(f"Unable to read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise SparseSourceIntegrityError(f"{label} must be a JSON object")
    return value


def _replay_snapshot(
    *,
    manifest_path: Path,
    registry_path: Path,
    roster_path: Path,
    expected_meeting_dates: Sequence[str] | None,
    expected_population_id: str | None,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path, label="sparse snapshot manifest")
    if (
        manifest.get("schema_version") != SNAPSHOT_MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("source_interface") != SOURCE_INTERFACE
    ):
        raise SparseSourceIntegrityError(
            "Sparse snapshot manifest schema/status mismatch"
        )
    if manifest.get("policy") not in (_legacy_snapshot_policy(), _snapshot_policy()):
        raise SparseSourceIntegrityError("Sparse snapshot policy changed")
    if not str(manifest.get("population_id") or "").strip():
        raise SparseSourceIntegrityError("Sparse snapshot population_id is empty")
    started_at = _parse_timestamp(
        manifest.get("started_at_utc"), label="started_at_utc"
    )
    completed_at = _parse_timestamp(
        manifest.get("completed_at_utc"), label="completed_at_utc"
    )
    if completed_at < started_at:
        raise SparseSourceIntegrityError(
            "Sparse snapshot completed before acquisition started"
        )
    try:
        payload_sha256 = validate_manifest_integrity(manifest)
    except ManifestIntegrityError as exc:
        raise SparseSourceIntegrityError(str(exc)) from exc
    if (
        expected_population_id is not None
        and manifest.get("population_id") != str(expected_population_id).strip()
    ):
        raise SparseSourceIntegrityError("Sparse snapshot population_id mismatch")
    for label, path in (("registry", registry_path), ("roster", roster_path)):
        binding = manifest.get(label)
        if not isinstance(binding, Mapping) or binding.get("path") != str(path):
            raise SparseSourceIntegrityError(f"Snapshot {label} binding mismatch")
        if binding.get("sha256") != sha256_file(path):
            raise SparseSourceIntegrityError(f"Snapshot {label} hash changed")
    raw_meetings = manifest.get("meetings")
    if not isinstance(raw_meetings, list):
        raise SparseSourceIntegrityError("Sparse snapshot meetings are missing")
    try:
        meetings = _normalise_meetings(
            [str(item["meeting_date"]) for item in raw_meetings],
            split=str(raw_meetings[0]["split"]),
        )
    except (KeyError, IndexError, TypeError) as exc:
        raise SparseSourceIntegrityError("Malformed sparse snapshot meetings") from exc
    if meetings != raw_meetings:
        raise SparseSourceIntegrityError("Sparse snapshot meeting records changed")
    if expected_meeting_dates is not None and [
        item["meeting_date"] for item in meetings
    ] != list(expected_meeting_dates):
        raise SparseSourceIntegrityError("Sparse snapshot meeting membership mismatch")
    registry_payload, source_series, _, sources, indicator_sources = _source_context(
        registry_path, roster_path
    )
    roots = build_snapshot_requests(source_series, meetings)
    attempts: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []

    class _NoNetworkLimiter:
        def acquire(self) -> None:
            return None

    def no_network(_url: str, _timeout: float) -> HttpResult:
        raise AssertionError("offline sparse replay attempted network access")

    for request in roots:
        replayed_attempts, replayed_outcomes = _walk_request(
            output_dir=manifest_path.parent,
            request=request,
            parent_request_id=None,
            depth=0,
            resume=True,
            allow_network=False,
            http_get=no_network,
            limiter=_NoNetworkLimiter(),
            timeout_seconds=1.0,
            max_retries=0,
            retry_sleep=lambda _delay: None,
            utc_now=_utc_now,
        )
        attempts.extend(replayed_attempts)
        outcomes.extend(replayed_outcomes)
    _validate_outcome_coverage(outcomes, sources=source_series, meetings=meetings)
    if manifest.get("attempts") != attempts:
        raise SparseSourceIntegrityError("Sparse snapshot attempts do not replay")
    if manifest.get("terminal_outcomes") != outcomes:
        raise SparseSourceIntegrityError(
            "Sparse snapshot terminal outcomes do not replay"
        )

    expected_cache_files: set[str] = set()
    for item in attempts:
        expected_cache_files.add(str(item["cache_metadata_relative_path"]))
        expected_cache_files.add(str(item["raw_relative_path"]))
        transport = item.get("transport")
        if transport is not None:
            if not isinstance(transport, Mapping):
                raise SparseSourceIntegrityError("Invalid cached transport record")
            expected_cache_files.add(
                PurePosixPath(str(item["raw_relative_path"]))
                .with_name(str(transport.get("relative_path") or ""))
                .as_posix()
            )
    observed_cache_files = {
        path.relative_to(manifest_path.parent).as_posix()
        for path in manifest_path.parent.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if observed_cache_files != expected_cache_files:
        raise SparseSourceIntegrityError(
            "Sparse snapshot request cache inventory differs"
        )
    if (
        manifest.get("series_count") != len(source_series)
        or manifest.get("meeting_count") != len(meetings)
        or manifest.get("source_vintage_count") != len(outcomes)
        or manifest.get("request_attempt_count") != len(attempts)
    ):
        raise SparseSourceIntegrityError("Sparse snapshot summary counts differ")
    available_count = sum(item["outcome"] == OUTCOME_AVAILABLE for item in outcomes)
    if (
        manifest.get("available_source_vintage_count") != available_count
        or manifest.get("unusable_source_vintage_count")
        != len(outcomes) - available_count
    ):
        raise SparseSourceIntegrityError("Sparse snapshot outcome counts differ")
    return {
        "manifest": manifest,
        "payload_sha256": payload_sha256,
        "meetings": meetings,
        "sources": sources,
        "indicator_sources": indicator_sources,
        "source_series": source_series,
        "attempts": attempts,
        "outcomes": outcomes,
        "registry_payload": registry_payload,
    }


def validate_sparse_snapshot_manifest(
    manifest_file: str | Path,
    *,
    registry_file: str | Path,
    roster_file: str | Path,
    expected_meeting_dates: Sequence[str] | None = None,
    expected_population_id: str | None = None,
) -> dict[str, Any]:
    """Replay every request cache and verify exact source-vintage closure."""

    manifest_path = Path(manifest_file).expanduser().resolve()
    state = _replay_snapshot(
        manifest_path=manifest_path,
        registry_path=Path(registry_file).expanduser().resolve(),
        roster_path=Path(roster_file).expanduser().resolve(),
        expected_meeting_dates=expected_meeting_dates,
        expected_population_id=expected_population_id,
    )
    return {
        "status": "valid",
        "schema_version": SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "population_id": state["manifest"]["population_id"],
        "manifest_payload_sha256": state["payload_sha256"],
        "source_vintage_count": len(state["outcomes"]),
        "available_source_vintage_count": sum(
            item["outcome"] == OUTCOME_AVAILABLE for item in state["outcomes"]
        ),
        "unusable_source_vintage_count": sum(
            item["outcome"] == OUTCOME_UNUSABLE for item in state["outcomes"]
        ),
    }


def _request_by_id(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for item in state["attempts"]:
        request_id = str(item["request_id"])
        if request_id in result:
            raise SparseSourceIntegrityError(f"Duplicate request attempt {request_id}")
        result[request_id] = item
    return result


def _expanded_request(
    *,
    outcome: Mapping[str, Any],
    attempt: Mapping[str, Any],
    snapshot_dir: Path,
) -> dict[str, Any]:
    batch_index = int(outcome["selected_batch_index"])
    vintage_values = list(attempt["vintage_dates"])
    cosd_values = list(attempt["cosd"])
    coed_values = list(attempt["coed"])
    if not 0 <= batch_index < len(vintage_values):
        raise SparseSourceIntegrityError("Invalid selected_batch_index")
    raw_path = _safe_relative_file(
        snapshot_dir,
        attempt["raw_relative_path"],
        label="available raw_relative_path",
    )
    expanded = deepcopy(dict(attempt))
    for field in (
        "outcome",
        "reason_code",
        "parent_request_id",
        "depth",
        "terminal",
        "cache_metadata_relative_path",
    ):
        expanded.pop(field, None)
    expanded.update(
        {
            "_raw_path": raw_path,
            "_retrieved": _parse_timestamp(
                attempt["retrieved_at_utc"], label="retrieved_at_utc"
            ),
            "_vintage": date.fromisoformat(vintage_values[batch_index]),
            "_cosd": date.fromisoformat(cosd_values[batch_index]),
            "_coed": date.fromisoformat(coed_values[batch_index]),
            "_batch_index": batch_index,
        }
    )
    return expanded


def _sampling_policy() -> dict[str, str]:
    return {
        "version": SAMPLING_POLICY_VERSION,
        "daily_weekly": (
            "trailing 24 calendar months; last available observation per month "
            "plus latest; maximum 25"
        ),
        "monthly": "latest 24 safe observations",
        "quarterly": "latest 8 safe observations",
        "semiannual": "latest 6 safe observations",
        "annual": "latest 5 safe observations",
    }


def _output_record(
    path: str, content: str, *, row_count: int | None = None
) -> dict[str, Any]:
    record: dict[str, Any] = {"path": path, "sha256": sha256_text(content)}
    if row_count is not None:
        record["row_count"] = row_count
    return record


def _build_ledger_artifacts(
    *,
    snapshot_manifest_path: Path,
    registry_path: Path,
    roster_path: Path,
) -> dict[str, Any]:
    state = _replay_snapshot(
        manifest_path=snapshot_manifest_path,
        registry_path=registry_path,
        roster_path=roster_path,
        expected_meeting_dates=None,
        expected_population_id=None,
    )
    manifest = state["manifest"]
    population_id = str(manifest["population_id"])
    roster = _load_roster(roster_path)
    sources: dict[str, dict[str, Any]] = state["sources"]
    indicator_sources: dict[str, list[str]] = state["indicator_sources"]
    registry_sha256 = sha256_file(registry_path)
    snapshot_payload_sha256 = state["payload_sha256"]
    request_index = _request_by_id(state)
    outcomes = {
        (str(item["meeting_date"]), str(item["source_key"])): item
        for item in state["outcomes"]
    }
    meetings = [date.fromisoformat(item["meeting_date"]) for item in state["meetings"]]
    split_by_meeting = {
        str(item["meeting_date"]): str(item["split"]) for item in state["meetings"]
    }
    expected_sample_count = len(meetings) * len(roster)
    ledger_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    record_exclusions: list[dict[str, Any]] = []
    sample_exclusions: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    used_source_keys: set[str] = set()

    for meeting in meetings:
        meeting_text = meeting.isoformat()
        as_of = information_as_of_date(meeting)
        as_of_text = as_of.isoformat()
        for indicator in roster:
            configured_keys = indicator_sources[indicator]
            sealed_unusable_proofs: list[dict[str, Any]] = []
            available_keys = [
                key
                for key in configured_keys
                if outcomes[(meeting_text, key)]["outcome"] == OUTCOME_AVAILABLE
            ]
            for source_key in configured_keys:
                outcome = outcomes[(meeting_text, source_key)]
                if outcome["outcome"] == OUTCOME_AVAILABLE:
                    continue
                if outcome["outcome"] != OUTCOME_UNUSABLE:
                    raise SparseSourceIntegrityError(
                        f"{meeting_text}/{indicator}: terminal source outcome is "
                        "neither available nor sealed unusable"
                    )
                sealed_unusable_proofs.append(
                    {
                        "source_key": source_key,
                        "series_id": outcome["series_id"],
                        "reason_code": outcome["reason_code"],
                        "status_code": outcome["status_code"],
                        "request_id": outcome["request_id"],
                        "raw_relative_path": outcome["raw_relative_path"],
                        "raw_sha256": outcome["raw_sha256"],
                        "byte_count": outcome["byte_count"],
                        "retrieved_at_utc": outcome["retrieved_at_utc"],
                    }
                )
            if not available_keys:
                display_topic = topic_for_ledger_indicator(indicator)
                sample_exclusions.append(
                    {
                        "schema_version": SAMPLE_EXCLUSION_SCHEMA_VERSION,
                        "sample_id": stable_sample_id(meeting_text, display_topic),
                        "split": split_by_meeting[meeting_text],
                        "meeting_date": meeting_text,
                        "atomic_topic": display_topic,
                        "ledger_indicator": indicator,
                        "reason_code": SAMPLE_EXCLUSION_REASON,
                        "information_as_of_date": as_of_text,
                        "configured_source_keys": list(configured_keys),
                        "unusable_sources": sealed_unusable_proofs,
                        "registry_sha256": registry_sha256,
                        "roster_sha256": sha256_file(roster_path),
                        "snapshot_manifest_payload_sha256": snapshot_payload_sha256,
                    }
                )
                coverage_rows.append(
                    {
                        "meeting_date": meeting_text,
                        "information_as_of_date": as_of_text,
                        "indicator": indicator,
                        "status": "excluded",
                        "source_count": 0,
                        "source_keys": [],
                        "configured_source_keys": list(configured_keys),
                        "reason_code": SAMPLE_EXCLUSION_REASON,
                    }
                )
                continue

            canonical_sample_id = f"{meeting_text}::{indicator}"
            series_payloads: list[dict[str, Any]] = []
            request_evidence: list[dict[str, Any]] = []
            retrieval_timestamps: list[datetime] = []
            all_observation_dates: list[str] = []
            sampling_unusable_proofs: list[dict[str, Any]] = []
            for source_key in available_keys:
                outcome = outcomes[(meeting_text, source_key)]
                attempt = request_index[str(outcome["request_id"])]
                request = _expanded_request(
                    outcome=outcome,
                    attempt=attempt,
                    snapshot_dir=snapshot_manifest_path.parent,
                )
                source = deepcopy(sources[source_key])
                source["_information_as_of_date"] = as_of
                raw_bytes = request["_raw_path"].read_bytes()
                observations, parse_exclusions = _parse_csv_observations(
                    raw_bytes,
                    source=source,
                    request=request,
                    meeting_date=meeting,
                    indicator=indicator,
                )
                record_exclusions.extend(parse_exclusions)
                if not observations:
                    sampling_unusable_proofs.append(
                        {
                            "source_key": source_key,
                            "series_id": outcome["series_id"],
                            "reason_code": REASON_NO_SAFE_NUMERIC_OBSERVATION,
                            "status_code": outcome["status_code"],
                            "request_id": outcome["request_id"],
                            "raw_relative_path": outcome["raw_relative_path"],
                            "raw_sha256": outcome["raw_sha256"],
                            "byte_count": outcome["byte_count"],
                            "retrieved_at_utc": outcome["retrieved_at_utc"],
                            "candidate_observation_count": 0,
                            "selected_observation_count": 0,
                            "information_as_of_date": as_of_text,
                        }
                    )
                    continue
                try:
                    sampled, sample_records = _sample_observations(
                        observations,
                        source=source,
                        sample_id=canonical_sample_id,
                    )
                except LooSamplingWindowEmptyError as exc:
                    record_exclusions.extend(exc.exclusions)
                    sampling_unusable_proofs.append(
                        {
                            "source_key": source_key,
                            "series_id": outcome["series_id"],
                            "reason_code": (
                                REASON_NO_OBSERVATION_IN_SAMPLING_WINDOW
                            ),
                            "status_code": outcome["status_code"],
                            "request_id": outcome["request_id"],
                            "raw_relative_path": outcome["raw_relative_path"],
                            "raw_sha256": outcome["raw_sha256"],
                            "byte_count": outcome["byte_count"],
                            "retrieved_at_utc": outcome["retrieved_at_utc"],
                            "candidate_observation_count": len(observations),
                            "selected_observation_count": 0,
                            "latest_candidate_observation_date": observations[-1][
                                "date"
                            ],
                            "information_as_of_date": as_of_text,
                            "sampling_policy_version": SAMPLING_POLICY_VERSION,
                        }
                    )
                    continue
                record_exclusions.extend(sample_records)
                all_observation_dates.extend(item["date"] for item in sampled)
                retrieval_timestamps.append(request["_retrieved"])
                title = str(
                    source.get("title") or source.get("label") or source["series_id"]
                ).strip()
                series_payloads.append(
                    {
                        "source_key": source_key,
                        "series_id": source["series_id"],
                        "title": title,
                        "frequency": source["frequency"],
                        "units": source["units"],
                        "seasonal_adjustment": source["seasonal_adjustment"],
                        "transformation": source["transformation"],
                        "availability_as_of_date": as_of_text,
                        "requested_vintage_date": as_of_text,
                        "observations": sampled,
                    }
                )
                request_evidence.append(
                    {
                        "source_key": source_key,
                        "series_id": source["series_id"],
                        "license": source["license"],
                        "redistribution_allowed": source["redistribution_allowed"],
                        "request_id": request["request_id"],
                        "raw_relative_path": request["raw_relative_path"],
                        "raw_sha256": request["raw_sha256"],
                        "byte_count": request["byte_count"],
                        "retrieved_at_utc": _format_timestamp(request["_retrieved"]),
                        "request_batch": {
                            "vintage_dates": request["vintage_dates"],
                            "cosd": request["cosd"],
                            "coed": request["coed"],
                        },
                        "selected_batch_index": request["_batch_index"],
                        "requested_vintage_date": request["_vintage"].isoformat(),
                        "observation_start": request["_cosd"].isoformat(),
                        "observation_end": request["_coed"].isoformat(),
                        "candidate_observation_count": len(observations),
                        "selected_observation_count": len(sampled),
                        "selected_observation_sha256": sha256_text(
                            _canonical_json(sampled)
                        ),
                    }
                )
                used_source_keys.add(source_key)

            if not series_payloads:
                display_topic = topic_for_ledger_indicator(indicator)
                proofs_by_key = {
                    str(item["source_key"]): item
                    for item in (
                        sealed_unusable_proofs + sampling_unusable_proofs
                    )
                }
                if set(proofs_by_key) != set(configured_keys):
                    raise SparseSourceIntegrityError(
                        f"{meeting_text}/{indicator}: source exclusion proof "
                        "does not cover every configured source"
                    )
                proofs = [proofs_by_key[key] for key in configured_keys]
                sample_exclusions.append(
                    {
                        "schema_version": SAMPLE_EXCLUSION_SCHEMA_VERSION,
                        "sample_id": stable_sample_id(meeting_text, display_topic),
                        "split": split_by_meeting[meeting_text],
                        "meeting_date": meeting_text,
                        "atomic_topic": display_topic,
                        "ledger_indicator": indicator,
                        "reason_code": SAMPLE_EXCLUSION_REASON,
                        "information_as_of_date": as_of_text,
                        "configured_source_keys": list(configured_keys),
                        "unusable_sources": proofs,
                        "registry_sha256": registry_sha256,
                        "roster_sha256": sha256_file(roster_path),
                        "snapshot_manifest_payload_sha256": (
                            snapshot_payload_sha256
                        ),
                    }
                )
                coverage_rows.append(
                    {
                        "meeting_date": meeting_text,
                        "information_as_of_date": as_of_text,
                        "indicator": indicator,
                        "status": "excluded",
                        "source_count": 0,
                        "source_keys": [],
                        "configured_source_keys": list(configured_keys),
                        "reason_code": SAMPLE_EXCLUSION_REASON,
                        "source_exclusions": proofs,
                    }
                )
                continue

            source_payload = {
                "sampling_policy": _sampling_policy(),
                "series": series_payloads,
            }
            _reject_forbidden_keys(source_payload)
            source_payload_sha256 = sha256_text(_canonical_json(source_payload))
            source_id = f"canonical-loo-d1:{population_id}:{meeting_text}:{indicator}"
            evidence_binding = {
                "schema_version": SOURCE_EVIDENCE_SCHEMA_VERSION,
                "population_id": population_id,
                "sample_id": canonical_sample_id,
                "source_id": source_id,
                "information_as_of_date": as_of_text,
                "registry_sha256": registry_sha256,
                "snapshot_manifest_payload_sha256": snapshot_payload_sha256,
                "source_payload_sha256": source_payload_sha256,
                "requests": request_evidence,
            }
            source_sha256 = sha256_text(_canonical_json(evidence_binding))
            evidence_rows.append({**evidence_binding, "source_sha256": source_sha256})
            observation_date = max(all_observation_dates)
            ledger_rows.append(
                {
                    "schema_version": LEDGER_ROW_SCHEMA_VERSION,
                    "meeting_id": meeting_text,
                    "sample_id": canonical_sample_id,
                    "meeting_timestamp": decision_identity_timestamp(meeting),
                    "meeting_date": meeting_text,
                    "indicator": indicator,
                    "source_id": source_id,
                    "source_sha256": source_sha256,
                    "source_timestamp": _format_timestamp(max(retrieval_timestamps)),
                    "information_as_of_date": as_of_text,
                    "requested_vintage_date": as_of_text,
                    "availability_as_of_date": as_of_text,
                    "availability_evidence_type": AVAILABILITY_EVIDENCE_TYPE,
                    "source_interface": SOURCE_INTERFACE,
                    "observation_date": observation_date,
                    "source_payload": source_payload,
                }
            )
            coverage_rows.append(
                {
                    "sample_id": canonical_sample_id,
                    "meeting_date": meeting_text,
                    "information_as_of_date": as_of_text,
                    "indicator": indicator,
                    "status": "ready",
                    "source_count": len(series_payloads),
                    "observation_count": len(all_observation_dates),
                    "source_keys": [item["source_key"] for item in series_payloads],
                    "configured_source_keys": list(configured_keys),
                    **(
                        {"source_exclusions": sampling_unusable_proofs}
                        if sampling_unusable_proofs
                        else {}
                    ),
                    "latest_observation_date": observation_date,
                }
            )

    if len(ledger_rows) + len(sample_exclusions) != expected_sample_count:
        raise SparseSourceIntegrityError("Sparse ledger sample closure failed")
    ready_ids = {row["sample_id"] for row in ledger_rows}
    if len(ready_ids) != len(ledger_rows):
        raise SparseSourceIntegrityError("Sparse ledger has duplicate ready rows")
    excluded_keys = {
        (row["meeting_date"], row["ledger_indicator"]) for row in sample_exclusions
    }
    if len(excluded_keys) != len(sample_exclusions):
        raise SparseSourceIntegrityError("Sparse ledger has duplicate exclusions")
    record_exclusions.sort(
        key=lambda row: (
            str(row.get("sample_id") or ""),
            str(row.get("source_key") or ""),
            str(row.get("observation_date") or ""),
            str(row.get("row_number") or ""),
            str(row.get("reason") or ""),
        )
    )
    sample_exclusions.sort(key=lambda row: (row["meeting_date"], row["atomic_topic"]))
    coverage = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "population_id": population_id,
        "meeting_count": len(meetings),
        "indicator_count": len(roster),
        "expected_sample_count": expected_sample_count,
        "ready_sample_count": len(ledger_rows),
        "excluded_sample_count": len(sample_exclusions),
        "sample_coverage_complete": True,
        "source_series_count": sum(
            int(row.get("source_count", 0)) for row in coverage_rows
        ),
        "observation_count": sum(
            int(row.get("observation_count", 0)) for row in coverage_rows
        ),
        "excluded_record_count": len(record_exclusions),
        "information_as_of_dates": {
            meeting.isoformat(): information_as_of_date(meeting).isoformat()
            for meeting in meetings
        },
        "rows": coverage_rows,
    }
    contents = {
        "indicator_inputs.jsonl": _jsonl_text(ledger_rows),
        "source_evidence.jsonl": _jsonl_text(evidence_rows),
        "excluded_records.jsonl": _jsonl_text(record_exclusions),
        "sample_exclusions.jsonl": _jsonl_text(sample_exclusions),
        "coverage.json": _json_text(coverage),
    }
    used_sources = [key for key in sources if key in used_source_keys]
    license_records = [
        {
            "source_key": key,
            "license": sources[key]["license"],
            "redistribution_allowed": sources[key]["redistribution_allowed"],
        }
        for key in used_sources
    ]
    manifest_payload = {
        "schema_version": LEDGER_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "population_id": population_id,
        "information_cutoff_policy": (
            "exact ALFRED historical vintage at meeting_date minus one calendar "
            "day; unusable source vintages are sealed and never substituted"
        ),
        "license_summary": {
            "local_research_use_only": any(
                not item["redistribution_allowed"] for item in license_records
            ),
            "redistribution_allowed_for_all_sources": all(
                item["redistribution_allowed"] for item in license_records
            ),
            "sources": license_records,
        },
        "inputs": {
            "registry": {
                "path": str(registry_path),
                "sha256": registry_sha256,
            },
            "roster": {
                "path": str(roster_path),
                "sha256": sha256_file(roster_path),
            },
            "snapshot_manifest": {
                "path": str(snapshot_manifest_path),
                "sha256": sha256_file(snapshot_manifest_path),
                "payload_sha256": snapshot_payload_sha256,
            },
        },
        "outputs": {
            "indicator_inputs": _output_record(
                "indicator_inputs.jsonl",
                contents["indicator_inputs.jsonl"],
                row_count=len(ledger_rows),
            ),
            "source_evidence": _output_record(
                "source_evidence.jsonl",
                contents["source_evidence.jsonl"],
                row_count=len(evidence_rows),
            ),
            "excluded_records": _output_record(
                "excluded_records.jsonl",
                contents["excluded_records.jsonl"],
                row_count=len(record_exclusions),
            ),
            "sample_exclusions": _output_record(
                "sample_exclusions.jsonl",
                contents["sample_exclusions.jsonl"],
                row_count=len(sample_exclusions),
            ),
            "coverage": _output_record("coverage.json", contents["coverage.json"]),
        },
    }
    ledger_manifest = seal_manifest(manifest_payload)
    contents["ledger_manifest.json"] = _json_text(ledger_manifest)
    return {
        "manifest": ledger_manifest,
        "contents": contents,
        "ledger_rows": ledger_rows,
        "evidence_rows": evidence_rows,
        "record_exclusions": record_exclusions,
        "sample_exclusions": sample_exclusions,
        "coverage": coverage,
    }


def build_sparse_loo_ledger(
    *,
    snapshot_manifest_file: str | Path,
    registry_file: str | Path,
    roster_file: str | Path,
    output_dir: str | Path,
) -> Path:
    """Build an immutable topic-sparse ledger and release-ready exclusions."""

    snapshot_path = Path(snapshot_manifest_file).expanduser().resolve()
    registry_path = Path(registry_file).expanduser().resolve()
    roster_path = Path(roster_file).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Sparse ledger directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = _build_ledger_artifacts(
        snapshot_manifest_path=snapshot_path,
        registry_path=registry_path,
        roster_path=roster_path,
    )
    for name in (
        "indicator_inputs.jsonl",
        "source_evidence.jsonl",
        "excluded_records.jsonl",
        "sample_exclusions.jsonl",
        "coverage.json",
    ):
        _write_new_text(destination / name, artifacts["contents"][name])
    manifest_path = destination / "ledger_manifest.json"
    _write_new_text(manifest_path, artifacts["contents"]["ledger_manifest.json"])
    validate_sparse_loo_ledger(
        manifest_path,
        snapshot_manifest_file=snapshot_path,
        registry_file=registry_path,
        roster_file=roster_path,
    )
    return manifest_path


def validate_sparse_loo_ledger(
    ledger_manifest_file: str | Path,
    *,
    snapshot_manifest_file: str | Path,
    registry_file: str | Path,
    roster_file: str | Path,
) -> dict[str, Any]:
    """Offline-replay raw bytes and compare every sparse ledger output."""

    manifest_path = Path(ledger_manifest_file).expanduser().resolve()
    snapshot_path = Path(snapshot_manifest_file).expanduser().resolve()
    registry_path = Path(registry_file).expanduser().resolve()
    roster_path = Path(roster_file).expanduser().resolve()
    manifest = _read_json(manifest_path, label="sparse ledger manifest")
    if (
        manifest.get("schema_version") != LEDGER_MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        raise SparseSourceIntegrityError("Sparse ledger schema/status mismatch")
    try:
        payload_sha256 = validate_manifest_integrity(manifest)
    except ManifestIntegrityError as exc:
        raise SparseSourceIntegrityError(str(exc)) from exc
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise SparseSourceIntegrityError("Sparse ledger input bindings are missing")
    for label, path in (
        ("registry", registry_path),
        ("roster", roster_path),
        ("snapshot_manifest", snapshot_path),
    ):
        binding = inputs.get(label)
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != str(path)
            or binding.get("sha256") != sha256_file(path)
        ):
            raise SparseSourceIntegrityError(f"Sparse ledger {label} input changed")
    expected = _build_ledger_artifacts(
        snapshot_manifest_path=snapshot_path,
        registry_path=registry_path,
        roster_path=roster_path,
    )
    if manifest != expected["manifest"]:
        raise SparseSourceIntegrityError("Sparse ledger manifest does not replay")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise SparseSourceIntegrityError("Sparse ledger outputs are missing")
    for label, filename in (
        ("indicator_inputs", "indicator_inputs.jsonl"),
        ("source_evidence", "source_evidence.jsonl"),
        ("excluded_records", "excluded_records.jsonl"),
        ("sample_exclusions", "sample_exclusions.jsonl"),
        ("coverage", "coverage.json"),
    ):
        record = outputs.get(label)
        if not isinstance(record, Mapping) or record.get("path") != filename:
            raise SparseSourceIntegrityError(f"Sparse ledger output {label} differs")
        path = _safe_relative_file(manifest_path.parent, filename, label=label)
        content = path.read_text(encoding="utf-8")
        if content != expected["contents"][filename] or record.get(
            "sha256"
        ) != sha256_text(content):
            raise SparseSourceIntegrityError(
                f"Sparse ledger output {label} cannot be replayed"
            )
    return {
        "status": "valid",
        "schema_version": LEDGER_MANIFEST_SCHEMA_VERSION,
        "population_id": manifest["population_id"],
        "manifest_payload_sha256": payload_sha256,
        "ready_sample_count": len(expected["ledger_rows"]),
        "excluded_sample_count": len(expected["sample_exclusions"]),
        "expected_sample_count": expected["coverage"]["expected_sample_count"],
    }


__all__ = [
    "COVERAGE_SCHEMA_VERSION",
    "LEDGER_MANIFEST_SCHEMA_VERSION",
    "OUTCOME_AVAILABLE",
    "OUTCOME_SPLIT",
    "OUTCOME_UNUSABLE",
    "REASON_HTTP_404",
    "REASON_NO_OBSERVATION_IN_SAMPLING_WINDOW",
    "REASON_NO_SAFE_NUMERIC_OBSERVATION",
    "REASON_NOT_ECHOED",
    "REASON_TRANSPORT_RETRY_EXHAUSTED",
    "SAMPLE_EXCLUSION_REASON",
    "SNAPSHOT_MANIFEST_SCHEMA_VERSION",
    "SparseSourceError",
    "SparseSourceIntegrityError",
    "SparseSourceTransportError",
    "acquire_sparse_source_snapshots",
    "build_sparse_loo_ledger",
    "validate_sparse_loo_ledger",
    "validate_sparse_snapshot_manifest",
]
