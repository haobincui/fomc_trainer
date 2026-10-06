"""Build a sealed daily Treasury panel around official FOMC Minutes releases.

The builder is intentionally independent of the downstream sentiment scorer and
regression.  It freezes the Federal Reserve calendar pages that explicitly say
``Minutes (Released ...)``, binds the exact 256-meeting CHK3 beta roster, and
constructs one common complete-case market panel for DGS2, DGS5, DGS10, and
VIX.  A page footer such as ``Last update`` is never accepted as release-date
evidence.

The daily-data estimands are exploratory, not high-frequency event windows:

* primary: DGS2 release-date close minus the previous observed business-day
  close, in basis points;
* maturity sensitivities: the analogous DGS5 and DGS10 changes;
* delayed response: release close to the next observed business-day close;
* placebo: the two preceding observed business-day closes; and
* VIX sensitivity: log changes over the same three windows.

Publication is create-only.  Raw official pages, HTTP receipts, roster and
market snapshots, parsed evidence, exclusions, the panel, and the manifest are
all retained and hashed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import bs4
from bs4 import BeautifulSoup, Tag

from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    ROOT / "output/evaluation/main/chk3_beta_core8_merged_n2048_vllm_k5_20260816_v1"
)
DEFAULT_OUTPUT = RUN_ROOT / "market_panel_fed_minutes_release_daily_v1"
DEFAULT_SUITE = (
    RUN_ROOT
    / "generation_formal_n2048_k5_three_models_v4_stochastic_schedule_v2"
    / "manifest.json"
)
DEFAULT_PRE_ROSTER = (
    ROOT
    / "dataset/processed/retrain_v2/"
    "chk3_minutes_external_holdout_1993_2008_all_regular_v1/"
    "official_meeting_roster.jsonl"
)
DEFAULT_POST_ROSTER = (
    ROOT
    / "dataset/processed/retrain_v2/"
    "chk3_minutes_post2008_2009_2025_fixed_core8_d1_v1/"
    "official_meeting_roster.jsonl"
)
MARKET_SOURCES = {
    "DGS2": ROOT
    / "dataset/raw_data/input_data/us_data/Treasury Yields/"
    "Market Yield on U.S. Treasury Securities at 2-Year Constant Maturity"
    "(Percent, Not Seasonally Adjusted).csv",
    "DGS5": ROOT
    / "dataset/raw_data/input_data/us_data/Treasury Yields/"
    "Market Yield on U.S. Treasury Securities at 5-Year Constant Maturity"
    "(Percent, Not Seasonally Adjusted).csv",
    "DGS10": ROOT
    / "dataset/raw_data/input_data/us_data/Treasury Yields/"
    "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity"
    "(Percent, Not Seasonally Adjusted).csv",
    "VIXCLS": ROOT
    / "dataset/raw_data/input_data/us_data/Market Volatility (VIX)/"
    "CBOE Volatility Index(Index, Not Seasonally Adjusted).csv",
}

SCHEMA_VERSION = "chk3-beta-fomc-minutes-release-market-panel-manifest-v1"
RELEASE_SCHEMA = "chk3-beta-official-minutes-release-date-v1"
PANEL_SCHEMA = "chk3-beta-minutes-release-daily-market-panel-row-v1"
EXCLUSION_SCHEMA = "chk3-beta-minutes-release-market-exclusion-v1"
RECEIPT_SCHEMA = "chk3-beta-fed-calendar-http-receipt-v1"
EXPECTED_MEETINGS = 256
EXPECTED_PRE = 128
EXPECTED_POST = 128
FED_ORIGIN = "https://www.federalreserve.gov"
CURRENT_CALENDAR_URL = f"{FED_ORIGIN}/monetarypolicy/fomccalendars.htm"
RELEASE_RE = re.compile(
    r"\bMinutes\b.{0,180}?\(\s*Released\s+"
    r"([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})\s*\)",
    re.IGNORECASE,
)
DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y")


class MinutesReleaseMarketPanelError(RuntimeError):
    """The official-release or market-panel contract failed closed."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_bytes(path: Path, value: bytes) -> None:
    if os.path.lexists(path):
        raise MinutesReleaseMarketPanelError(f"refusing overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)
    _fsync_directory(path.parent)


def _write_new_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_new_bytes(
        path,
        "".join(_canonical(dict(row)) + "\n" for row in rows).encode("utf-8"),
    )


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_new_bytes(path, (_canonical(dict(value)) + "\n").encode("utf-8"))


def _binding(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise MinutesReleaseMarketPanelError(f"bound path is a symlink: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise MinutesReleaseMarketPanelError(f"bound file missing: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise MinutesReleaseMarketPanelError(f"JSONL source missing/symlink: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MinutesReleaseMarketPanelError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise MinutesReleaseMarketPanelError(
                    f"non-object JSONL row at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _calendar_url(year: int) -> str:
    if 1993 <= year <= 2020:
        return f"{FED_ORIGIN}/monetarypolicy/fomchistorical{year}.htm"
    if 2021 <= year <= 2025:
        return CURRENT_CALENDAR_URL
    raise MinutesReleaseMarketPanelError(f"year outside frozen beta scope: {year}")


def _calendar_filename(url: str) -> str:
    name = Path(urlparse(url).path).name
    if not name or not name.lower().endswith((".htm", ".html")):
        raise MinutesReleaseMarketPanelError(f"unsafe calendar URL filename: {url}")
    return name


def _load_rosters(pre_path: Path, post_path: Path) -> list[dict[str, Any]]:
    pre = _read_jsonl(pre_path)
    post = _read_jsonl(post_path)
    if len(pre) != EXPECTED_PRE or len(post) != EXPECTED_POST:
        raise MinutesReleaseMarketPanelError(
            f"roster closure is not {EXPECTED_PRE}+{EXPECTED_POST}: "
            f"{len(pre)}+{len(post)}"
        )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for era, rows in (("pre_external", pre), ("post", post)):
        for source_row in rows:
            row = copy.deepcopy(source_row)
            meeting_id = str(row.get("meeting_id") or "")
            if not meeting_id or meeting_id in seen:
                raise MinutesReleaseMarketPanelError(
                    f"duplicate/empty meeting_id: {meeting_id!r}"
                )
            seen.add(meeting_id)
            try:
                start = date.fromisoformat(str(row["meeting_start_date"]))
                end = date.fromisoformat(str(row["meeting_end_date"]))
            except (KeyError, ValueError) as exc:
                raise MinutesReleaseMarketPanelError(
                    f"invalid meeting dates for {meeting_id}"
                ) from exc
            if start > end:
                raise MinutesReleaseMarketPanelError(
                    f"meeting start after end for {meeting_id}"
                )
            minutes_url = str(row.get("official_minutes_url") or "")
            parsed = urlparse(minutes_url)
            if parsed.scheme != "https" or parsed.netloc.lower() != "www.federalreserve.gov":
                raise MinutesReleaseMarketPanelError(
                    f"non-official minutes URL for {meeting_id}: {minutes_url}"
                )
            row["era"] = era
            result.append(row)
    if len(result) != EXPECTED_MEETINGS:
        raise MinutesReleaseMarketPanelError("combined roster is not N=256")
    if any(
        str(result[index]["meeting_end_date"])
        >= str(result[index + 1]["meeting_end_date"])
        for index in range(len(result) - 1)
    ):
        raise MinutesReleaseMarketPanelError("combined roster is not chronological")
    return result


def _load_generation_meeting_metadata(
    suite_value: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, dict[str, Any]]]:
    cohort_binding = suite_value.get("cohort")
    if not isinstance(cohort_binding, Mapping):
        raise MinutesReleaseMarketPanelError("v4 suite cohort binding missing")
    cohort_path = _validate_bound_file(cohort_binding)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if not isinstance(cohort, dict):
        raise MinutesReleaseMarketPanelError("cohort manifest is not an object")
    validate_manifest_integrity(cohort)
    sample_binding = cohort.get("source_sample_manifest")
    if not isinstance(sample_binding, Mapping):
        raise MinutesReleaseMarketPanelError("cohort source-sample binding missing")
    sample_path = _validate_bound_file(sample_binding)
    sample_manifest = json.loads(sample_path.read_text(encoding="utf-8"))
    if not isinstance(sample_manifest, dict):
        raise MinutesReleaseMarketPanelError("sample manifest is not an object")
    validate_manifest_integrity(sample_manifest)
    samples = sample_manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != 2048:
        raise MinutesReleaseMarketPanelError("source sample manifest is not N=2,048")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise MinutesReleaseMarketPanelError("source sample is not an object")
        grouped.setdefault(str(sample.get("meeting_end_date") or ""), []).append(sample)
    if len(grouped) != EXPECTED_MEETINGS or any(len(rows) != 8 for rows in grouped.values()):
        raise MinutesReleaseMarketPanelError("source samples do not form 256 Core8 meetings")
    metadata: dict[str, dict[str, Any]] = {}
    fields = (
        "era",
        "original_post_split_role",
        "original_qa_split",
        "source_split",
        "cp318_selection_exposed",
        "transport_split_role",
        "not_all_held_out",
    )
    for meeting_end_date, rows in grouped.items():
        first = rows[0]
        if any(any(row.get(key) != first.get(key) for key in fields) for row in rows[1:]):
            raise MinutesReleaseMarketPanelError(
                f"generation meeting metadata drift: {meeting_end_date}"
            )
        source_era = str(first.get("era") or "")
        if source_era == "pre2009_external":
            era = "pre_external"
            original_split_role = "external_holdout"
        elif source_era == "post2008_chk3_release":
            era = "post"
            original_split_role = str(first.get("original_post_split_role") or "")
        else:
            raise MinutesReleaseMarketPanelError(
                f"unknown generation era for {meeting_end_date}: {source_era}"
            )
        if not original_split_role:
            raise MinutesReleaseMarketPanelError(
                f"missing original split role for {meeting_end_date}"
            )
        metadata[meeting_end_date] = {
            "era": era,
            "source_generation_era": source_era,
            "generation_meeting_id": first.get("meeting_id"),
            "original_split_role": original_split_role,
            "original_qa_split": first.get("original_qa_split"),
            "source_split": first.get("source_split"),
            "cp318_selection_exposed": bool(first.get("cp318_selection_exposed")),
            "transport_split_role": first.get("transport_split_role"),
            "not_all_held_out": bool(first.get("not_all_held_out")),
        }
    return cohort_path, sample_path, metadata


def _enrich_rosters(
    rosters: Sequence[Mapping[str, Any]], metadata: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in rosters:
        row = copy.deepcopy(dict(source))
        meeting_end = str(row["meeting_end_date"])
        values = metadata.get(meeting_end)
        if values is None or values.get("era") != row.get("era"):
            raise MinutesReleaseMarketPanelError(
                f"roster/generation meeting metadata mismatch: {meeting_end}"
            )
        row.update(copy.deepcopy(dict(values)))
        result.append(row)
    if len(result) != EXPECTED_MEETINGS:
        raise MinutesReleaseMarketPanelError("enriched roster is not N=256")
    return result


def _fetch_page(url: str, *, timeout_seconds: float, attempts: int = 3) -> tuple[bytes, dict[str, str]]:
    last_error: BaseException | None = None
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; FOMC academic release-date audit; "
            "+https://www.federalreserve.gov/)"
        )
    }
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, headers=headers, timeout=timeout_seconds)
            response.raise_for_status()
            content = bytes(response.content)
            if len(content) < 10_000 or b"<html" not in content[:10_000].lower():
                raise MinutesReleaseMarketPanelError(
                    f"official calendar response is not plausible HTML: {url}"
                )
            final = urlparse(response.url)
            if final.scheme != "https" or final.netloc.lower() != "www.federalreserve.gov":
                raise MinutesReleaseMarketPanelError(
                    f"official calendar redirected off origin: {response.url}"
                )
            receipt_headers = {
                key.lower(): value
                for key, value in response.headers.items()
                if key.lower()
                in {"content-type", "date", "etag", "last-modified", "content-length"}
            }
            receipt_headers["final_url"] = response.url
            receipt_headers["status_code"] = str(response.status_code)
            return content, receipt_headers
        except BaseException as exc:  # retain the final network/contract error
            last_error = exc
            if attempt < attempts:
                time.sleep(0.4 * attempt)
    raise MinutesReleaseMarketPanelError(f"failed to fetch {url}: {last_error}")


def _parse_date(text: str) -> date:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    raise MinutesReleaseMarketPanelError(f"unparseable official release date: {text}")


def parse_release_evidence(
    *,
    html_bytes: bytes,
    meeting: Mapping[str, Any],
    source_url: str,
    source_path: Path,
) -> dict[str, Any]:
    """Parse one explicit Fed calendar release annotation for one meeting."""

    soup = BeautifulSoup(html_bytes, "html.parser")
    official_path = urlparse(str(meeting["official_minutes_url"])).path.casefold()
    candidates: list[Tag] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "")
        parsed = urlparse(href)
        if not parsed.fragment and parsed.path.casefold() == official_path:
            candidates.append(anchor)
    if len(candidates) != 1:
        raise MinutesReleaseMarketPanelError(
            f"official minutes link closure for {meeting['meeting_id']}: "
            f"expected=1 observed={len(candidates)}"
        )
    anchor = candidates[0]
    evidence_node: Tag | None = None
    match: re.Match[str] | None = None
    node: Tag | None = anchor
    for _depth in range(6):
        parent = node.parent if node is not None else None
        if not isinstance(parent, Tag):
            break
        node = parent
        evidence_text = " ".join(node.stripped_strings)
        candidate_match = RELEASE_RE.search(evidence_text)
        if candidate_match is not None:
            evidence_node = node
            match = candidate_match
            break
    if evidence_node is None or match is None:
        raise MinutesReleaseMarketPanelError(
            f"explicit Minutes (Released ...) evidence missing for {meeting['meeting_id']}"
        )
    release_date = _parse_date(match.group(1))
    meeting_end = date.fromisoformat(str(meeting["meeting_end_date"]))
    lag_days = (release_date - meeting_end).days
    if lag_days < 1 or lag_days > 100:
        raise MinutesReleaseMarketPanelError(
            f"implausible release lag for {meeting['meeting_id']}: {lag_days}"
        )
    evidence_text = " ".join(evidence_node.stripped_strings)
    evidence_html = str(evidence_node)
    matched_href = str(anchor["href"])
    return {
        "schema_version": RELEASE_SCHEMA,
        "meeting_id": meeting["meeting_id"],
        "era": meeting["era"],
        "source_generation_era": meeting.get("source_generation_era"),
        "generation_meeting_id": meeting.get("generation_meeting_id"),
        "original_split_role": meeting.get("original_split_role"),
        "original_qa_split": meeting.get("original_qa_split"),
        "source_split": meeting.get("source_split"),
        "cp318_selection_exposed": bool(meeting.get("cp318_selection_exposed")),
        "transport_split_role": meeting.get("transport_split_role"),
        "not_all_held_out": bool(meeting.get("not_all_held_out")),
        "meeting_start_date": meeting["meeting_start_date"],
        "meeting_end_date": meeting["meeting_end_date"],
        "meeting_type": meeting.get("meeting_type"),
        "scheduled": meeting.get("scheduled"),
        "sensitivity_flag": bool(meeting.get("sensitivity_flag", False)),
        "sensitivity_reason": meeting.get("sensitivity_reason"),
        "official_minutes_url": meeting["official_minutes_url"],
        "release_date": release_date.isoformat(),
        "release_lag_calendar_days": lag_days,
        "release_time": None,
        "release_timezone": None,
        "release_time_status": "not_stated_on_bound_calendar_page",
        "release_evidence_type": "fed_calendar_explicit_minutes_released_annotation",
        "source_calendar_url": source_url,
        "source_calendar_path": str(source_path.resolve()),
        "source_calendar_sha256": _sha256_bytes(html_bytes),
        "matched_minutes_href": matched_href,
        "matched_minutes_url": urljoin(source_url, matched_href),
        "matched_anchor_text": " ".join(anchor.stripped_strings),
        "evidence_text": evidence_text,
        "evidence_text_sha256": _sha256_text(evidence_text),
        "evidence_html": evidence_html,
        "evidence_html_sha256": _sha256_text(evidence_html),
        "parser_contract": "exact_official_minutes_path_then_nearest_explicit_released_annotation_v1",
        "last_update_footer_accepted": False,
    }


def _copy_source(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise MinutesReleaseMarketPanelError(f"source missing/symlink: {source}")
    if os.path.lexists(target):
        raise MinutesReleaseMarketPanelError(f"refusing overwrite: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle:
        _write_new_bytes(target, input_handle.read())


def _load_market_series(path: Path, series_id: str) -> tuple[pd.Series, dict[str, Any]]:
    frame = pd.read_csv(path)
    if list(frame.columns) != ["observation_date", series_id]:
        raise MinutesReleaseMarketPanelError(
            f"unexpected {series_id} columns: {frame.columns.tolist()}"
        )
    dates = pd.to_datetime(frame["observation_date"], errors="raise")
    values = pd.to_numeric(frame[series_id], errors="coerce")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise MinutesReleaseMarketPanelError(f"{series_id} dates not unique/sorted")
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name=series_id)
    valid = series.dropna()
    if valid.empty or not (valid > 0).all():
        raise MinutesReleaseMarketPanelError(f"{series_id} has no valid positive data")
    profile = {
        "series_id": series_id,
        "rows": len(series),
        "missing_values": int(series.isna().sum()),
        "first_date": series.index.min().date().isoformat(),
        "last_date": series.index.max().date().isoformat(),
        "first_valid_date": valid.index.min().date().isoformat(),
        "last_valid_date": valid.index.max().date().isoformat(),
    }
    return series, profile


def _market_window(series: pd.Series, release_date: str) -> dict[str, Any] | None:
    release = pd.Timestamp(release_date)
    valid = series.dropna()
    location = valid.index.get_indexer([release])[0]
    if location < 2 or location + 1 >= len(valid) or location < 0:
        return None
    dates = valid.index[location - 2 : location + 2]
    values = valid.iloc[location - 2 : location + 2]
    if len(dates) != 4 or dates[2] != release:
        return None
    return {
        "previous_2_date": dates[0].date().isoformat(),
        "previous_date": dates[1].date().isoformat(),
        "release_date": dates[2].date().isoformat(),
        "next_date": dates[3].date().isoformat(),
        "previous_2_value": float(values.iloc[0]),
        "previous_value": float(values.iloc[1]),
        "release_value": float(values.iloc[2]),
        "next_value": float(values.iloc[3]),
        "previous_gap_calendar_days": int((dates[2] - dates[1]).days),
        "next_gap_calendar_days": int((dates[3] - dates[2]).days),
    }


def build_market_rows(
    release_rows: Sequence[Mapping[str, Any]],
    series_by_id: Mapping[str, pd.Series],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct a four-series common complete-case panel and exclusions."""

    expected_ids = {"DGS2", "DGS5", "DGS10", "VIXCLS"}
    if set(series_by_id) != expected_ids:
        raise MinutesReleaseMarketPanelError("market series set drift")
    panel: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for release_row in release_rows:
        meeting_id = str(release_row["meeting_id"])
        if meeting_id in seen:
            raise MinutesReleaseMarketPanelError(f"duplicate release meeting: {meeting_id}")
        seen.add(meeting_id)
        windows = {
            series_id: _market_window(series, str(release_row["release_date"]))
            for series_id, series in series_by_id.items()
        }
        missing = sorted(series_id for series_id, window in windows.items() if window is None)
        if missing:
            exclusions.append(
                {
                    "schema_version": EXCLUSION_SCHEMA,
                    "meeting_id": meeting_id,
                    "era": release_row["era"],
                    "original_split_role": release_row.get("original_split_role"),
                    "cp318_selection_exposed": release_row.get(
                        "cp318_selection_exposed"
                    ),
                    "meeting_end_date": release_row["meeting_end_date"],
                    "release_date": release_row["release_date"],
                    "reason": "common_window_missing",
                    "missing_series": missing,
                    "required_window": "previous_2,previous,release,next_valid_observations",
                }
            )
            continue
        exact_windows = {key: value for key, value in windows.items() if value is not None}
        row: dict[str, Any] = {
            "schema_version": PANEL_SCHEMA,
            "meeting_id": meeting_id,
            "era": release_row["era"],
            "sample_roles": [
                "full",
                "pre" if release_row["era"] == "pre_external" else "post",
            ],
            "in_full_sample": True,
            "in_pre_sample": release_row["era"] == "pre_external",
            "in_post_sample": release_row["era"] == "post",
            "source_generation_era": release_row.get("source_generation_era"),
            "generation_meeting_id": release_row.get("generation_meeting_id"),
            "original_split_role": release_row.get("original_split_role"),
            "original_qa_split": release_row.get("original_qa_split"),
            "source_split": release_row.get("source_split"),
            "cp318_selection_exposed": release_row.get("cp318_selection_exposed"),
            "transport_split_role": release_row.get("transport_split_role"),
            "not_all_held_out": release_row.get("not_all_held_out"),
            "meeting_start_date": release_row["meeting_start_date"],
            "meeting_end_date": release_row["meeting_end_date"],
            "meeting_type": release_row.get("meeting_type"),
            "scheduled": release_row.get("scheduled"),
            "sensitivity_flag": release_row.get("sensitivity_flag"),
            "sensitivity_reason": release_row.get("sensitivity_reason"),
            "release_date": release_row["release_date"],
            "release_lag_calendar_days": release_row["release_lag_calendar_days"],
            "release_evidence_text_sha256": release_row["evidence_text_sha256"],
            "official_minutes_url": release_row["official_minutes_url"],
            "market_windows": exact_windows,
            "exact_release_day_all_series": True,
            "missing_reasons": [],
        }
        for series_id in ("DGS2", "DGS5", "DGS10"):
            window = exact_windows[series_id]
            prefix = series_id.lower()
            row[f"{prefix}_exact_release_day"] = True
            row[f"{prefix}_previous_date"] = window["previous_date"]
            row[f"{prefix}_previous_value_pct"] = window["previous_value"]
            row[f"{prefix}_release_date"] = window["release_date"]
            row[f"{prefix}_release_value_pct"] = window["release_value"]
            row[f"{prefix}_next_date"] = window["next_date"]
            row[f"{prefix}_next_value_pct"] = window["next_value"]
            row[f"{prefix}_release_minus_previous_bp"] = 100.0 * (
                window["release_value"] - window["previous_value"]
            )
            row[f"{prefix}_change_bps"] = row[
                f"{prefix}_release_minus_previous_bp"
            ]
            row[f"{prefix}_next_minus_release_bp"] = 100.0 * (
                window["next_value"] - window["release_value"]
            )
            row[f"{prefix}_placebo_previous_minus_previous_2_bp"] = 100.0 * (
                window["previous_value"] - window["previous_2_value"]
            )
        vix = exact_windows["VIXCLS"]
        row["vix_exact_release_day"] = True
        row["vix_previous_date"] = vix["previous_date"]
        row["vix_previous_value"] = vix["previous_value"]
        row["vix_release_date"] = vix["release_date"]
        row["vix_release_value"] = vix["release_value"]
        row["vix_next_date"] = vix["next_date"]
        row["vix_next_value"] = vix["next_value"]
        row["vix_release_minus_previous_points"] = (
            vix["release_value"] - vix["previous_value"]
        )
        row["vix_next_minus_release_points"] = vix["next_value"] - vix["release_value"]
        row["vix_placebo_previous_minus_previous_2_points"] = (
            vix["previous_value"] - vix["previous_2_value"]
        )
        row["vix_release_minus_previous_log_pct"] = 100.0 * math.log(
            vix["release_value"] / vix["previous_value"]
        )
        row["vix_log_change"] = row["vix_release_minus_previous_log_pct"]
        row["vix_next_minus_release_log_pct"] = 100.0 * math.log(
            vix["next_value"] / vix["release_value"]
        )
        row["vix_placebo_previous_minus_previous_2_log_pct"] = 100.0 * math.log(
            vix["previous_value"] / vix["previous_2_value"]
        )
        panel.append(row)
    if len(panel) + len(exclusions) != len(release_rows):
        raise MinutesReleaseMarketPanelError("panel/exclusion partition failed")
    return panel, exclusions


def _validate_bound_file(binding: Mapping[str, Any], *, rows: int | None = None) -> Path:
    path = Path(str(binding.get("path") or ""))
    if path.is_symlink() or not path.is_file():
        raise MinutesReleaseMarketPanelError(f"bound artifact missing/symlink: {path}")
    if (
        ("bytes" in binding and binding.get("bytes") != path.stat().st_size)
        or binding.get("sha256") != sha256_file(path)
    ):
        raise MinutesReleaseMarketPanelError(f"bound artifact drift: {path}")
    if rows is not None and binding.get("rows") != rows:
        raise MinutesReleaseMarketPanelError(f"bound row count drift: {path}")
    return path


def _freeze_output_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        if path.is_symlink():
            raise MinutesReleaseMarketPanelError(f"output contains symlink: {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)
    _fsync_directory(root.parent)


def _require_frozen_tree(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            raise MinutesReleaseMarketPanelError(f"sealed tree contains symlink: {path}")
        if path.stat().st_mode & 0o222:
            raise MinutesReleaseMarketPanelError(f"sealed tree remains writable: {path}")


def validate_panel(manifest_path: Path) -> dict[str, Any]:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise MinutesReleaseMarketPanelError("panel manifest missing/symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise MinutesReleaseMarketPanelError("panel manifest is not an object")
    validate_manifest_integrity(manifest)
    _require_frozen_tree(manifest_path.resolve().parent)
    coverage = manifest.get("coverage")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or not isinstance(coverage, Mapping)
        or coverage.get("release_dates") != EXPECTED_MEETINGS
        or coverage.get("candidate_meetings") != EXPECTED_MEETINGS
        or coverage.get("common_panel_rows") + coverage.get("excluded_meetings")
        != EXPECTED_MEETINGS
    ):
        raise MinutesReleaseMarketPanelError("panel manifest contract drift")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise MinutesReleaseMarketPanelError("artifact bindings missing")
    release_path = _validate_bound_file(
        artifacts["release_dates"], rows=EXPECTED_MEETINGS
    )
    panel_path = _validate_bound_file(
        artifacts["market_panel"], rows=int(coverage["common_panel_rows"])
    )
    exclusion_path = _validate_bound_file(
        artifacts["exclusions"], rows=int(coverage["excluded_meetings"])
    )
    receipt_path = _validate_bound_file(
        artifacts["http_receipts"], rows=int(coverage["official_calendar_pages"])
    )
    release_rows = _read_jsonl(release_path)
    panel_rows = _read_jsonl(panel_path)
    exclusion_rows = _read_jsonl(exclusion_path)
    receipts = _read_jsonl(receipt_path)
    if len({row.get("meeting_id") for row in release_rows}) != EXPECTED_MEETINGS:
        raise MinutesReleaseMarketPanelError("release-date identities are not unique")
    if {row.get("meeting_id") for row in panel_rows}.intersection(
        row.get("meeting_id") for row in exclusion_rows
    ):
        raise MinutesReleaseMarketPanelError("panel/exclusion identities overlap")
    page_hashes = {
        str(receipt["local_path"]): str(receipt["sha256"]) for receipt in receipts
    }
    for path_text, expected_sha in page_hashes.items():
        page_path = Path(path_text)
        if page_path.is_symlink() or not page_path.is_file() or sha256_file(page_path) != expected_sha:
            raise MinutesReleaseMarketPanelError(f"official page drift: {page_path}")
    for row in release_rows:
        if page_hashes.get(str(row["source_calendar_path"])) != row["source_calendar_sha256"]:
            raise MinutesReleaseMarketPanelError(
                f"release evidence/page mismatch: {row['meeting_id']}"
            )
        if row.get("last_update_footer_accepted") is not False:
            raise MinutesReleaseMarketPanelError("Last update evidence is forbidden")
        rebuilt = parse_release_evidence(
            html_bytes=Path(str(row["source_calendar_path"])).read_bytes(),
            meeting=row,
            source_url=str(row["source_calendar_url"]),
            source_path=Path(str(row["source_calendar_path"])),
        )
        if _canonical(rebuilt) != _canonical(row):
            raise MinutesReleaseMarketPanelError(
                f"release evidence recomputation drift: {row['meeting_id']}"
            )
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or not isinstance(
        sources.get("market_series"), Mapping
    ):
        raise MinutesReleaseMarketPanelError("market source bindings missing")
    series_by_id: dict[str, pd.Series] = {}
    for series_id in ("DGS2", "DGS5", "DGS10", "VIXCLS"):
        source_value = sources["market_series"].get(series_id)
        if not isinstance(source_value, Mapping) or not isinstance(
            source_value.get("snapshot"), Mapping
        ):
            raise MinutesReleaseMarketPanelError(
                f"market snapshot binding missing: {series_id}"
            )
        snapshot_path = _validate_bound_file(source_value["snapshot"])
        series_by_id[series_id], profile = _load_market_series(
            snapshot_path, series_id
        )
        if profile != source_value.get("profile"):
            raise MinutesReleaseMarketPanelError(
                f"market profile recomputation drift: {series_id}"
            )
    rebuilt_panel, rebuilt_exclusions = build_market_rows(release_rows, series_by_id)
    if _canonical(rebuilt_panel) != _canonical(panel_rows):
        raise MinutesReleaseMarketPanelError("market panel recomputation drift")
    if _canonical(rebuilt_exclusions) != _canonical(exclusion_rows):
        raise MinutesReleaseMarketPanelError("market exclusions recomputation drift")
    return {
        "manifest": manifest,
        "release_rows": release_rows,
        "panel_rows": panel_rows,
        "exclusion_rows": exclusion_rows,
    }


def build(
    *,
    output_dir: Path,
    pre_roster: Path,
    post_roster: Path,
    suite_manifest: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    unresolved_output = output_dir.expanduser()
    if unresolved_output.is_symlink() or os.path.lexists(unresolved_output):
        raise MinutesReleaseMarketPanelError("output root must be fresh")
    output = unresolved_output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    _fsync_directory(output.parent)

    pre_roster = pre_roster.expanduser().resolve()
    post_roster = post_roster.expanduser().resolve()
    suite_manifest = suite_manifest.expanduser().resolve()
    if suite_manifest.is_symlink() or not suite_manifest.is_file():
        raise MinutesReleaseMarketPanelError("sealed v4 suite manifest missing/symlink")
    suite_value = json.loads(suite_manifest.read_text(encoding="utf-8"))
    validate_manifest_integrity(suite_value)
    if (
        suite_value.get("status") != "complete"
        or suite_value.get("schema_version")
        != "chk3-beta-core8-vllm-k5-dual-dp1-stochastic-schedule-suite-v2"
        or not isinstance(suite_value.get("integrity"), Mapping)
    ):
        raise MinutesReleaseMarketPanelError("v4 suite is not complete and sealed")

    cohort_path, sample_manifest_path, meeting_metadata = (
        _load_generation_meeting_metadata(suite_value)
    )
    rosters = _enrich_rosters(
        _load_rosters(pre_roster, post_roster), meeting_metadata
    )
    roster_dir = output / "sources/rosters"
    pre_snapshot = roster_dir / "pre2009.official_meeting_roster.jsonl"
    post_snapshot = roster_dir / "post2008.official_meeting_roster.jsonl"
    _copy_source(pre_roster, pre_snapshot)
    _copy_source(post_roster, post_snapshot)
    generation_source_dir = output / "sources/generation_contract"
    cohort_snapshot = generation_source_dir / "cohort_n2048_k5.v1.json"
    sample_manifest_snapshot = generation_source_dir / "samples_n2048_k10.json"
    _copy_source(cohort_path, cohort_snapshot)
    _copy_source(sample_manifest_path, sample_manifest_snapshot)

    retrieved_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    page_dir = output / "sources/federal_reserve_calendars"
    pages: dict[str, bytes] = {}
    page_paths: dict[str, Path] = {}
    receipts: list[dict[str, Any]] = []
    for url in sorted({_calendar_url(int(row["meeting_end_date"][:4])) for row in rosters}):
        content, headers = _fetch_page(url, timeout_seconds=timeout_seconds)
        local_path = page_dir / _calendar_filename(url)
        _write_new_bytes(local_path, content)
        pages[url] = content
        page_paths[url] = local_path
        receipts.append(
            {
                "schema_version": RECEIPT_SCHEMA,
                "requested_url": url,
                "final_url": headers.pop("final_url"),
                "status_code": int(headers.pop("status_code")),
                "retrieved_at_utc": retrieved_at,
                "response_headers": headers,
                "local_path": str(local_path.resolve()),
                "bytes": len(content),
                "sha256": _sha256_bytes(content),
            }
        )
    receipt_path = output / "official_calendar_http_receipts.v1.jsonl"
    _write_new_jsonl(receipt_path, receipts)

    release_rows: list[dict[str, Any]] = []
    for meeting in rosters:
        year = int(str(meeting["meeting_end_date"])[:4])
        url = _calendar_url(year)
        release_rows.append(
            parse_release_evidence(
                html_bytes=pages[url],
                meeting=meeting,
                source_url=url,
                source_path=page_paths[url],
            )
        )
    if len(release_rows) != EXPECTED_MEETINGS:
        raise MinutesReleaseMarketPanelError("release-date parse closure is not N=256")
    if len({row["meeting_id"] for row in release_rows}) != EXPECTED_MEETINGS:
        raise MinutesReleaseMarketPanelError("release-date identities are not unique")
    release_path = output / "official_minutes_release_dates.v1.jsonl"
    _write_new_jsonl(release_path, release_rows)

    market_dir = output / "sources/market"
    series_by_id: dict[str, pd.Series] = {}
    market_profiles: dict[str, Any] = {}
    market_bindings: dict[str, Any] = {}
    for series_id, source in MARKET_SOURCES.items():
        snapshot = market_dir / f"{series_id}.csv"
        _copy_source(source.resolve(), snapshot)
        series, profile = _load_market_series(snapshot, series_id)
        series_by_id[series_id] = series
        market_profiles[series_id] = profile
        market_bindings[series_id] = {
            "upstream": _binding(source.resolve()),
            "snapshot": _binding(snapshot, rows=len(series)),
            "profile": profile,
        }

    panel_rows, exclusions = build_market_rows(release_rows, series_by_id)
    panel_path = output / "market_panel.jsonl"
    exclusion_path = output / "market_panel.exclusions.v1.jsonl"
    _write_new_jsonl(panel_path, panel_rows)
    _write_new_jsonl(exclusion_path, exclusions)
    role_counts = {
        "full": len(panel_rows),
        "pre": sum(bool(row["in_pre_sample"]) for row in panel_rows),
        "post": sum(bool(row["in_post_sample"]) for row in panel_rows),
    }
    exclusion_counts = dict(sorted(Counter(row["era"] for row in exclusions).items()))
    if role_counts["pre"] + role_counts["post"] != role_counts["full"]:
        raise MinutesReleaseMarketPanelError("pre/post roles do not partition full")

    coverage_value = {
        "schema_version": "chk3-beta-minutes-release-market-coverage-v1",
        "candidate_meetings": EXPECTED_MEETINGS,
        "release_dates": len(release_rows),
        "official_calendar_pages": len(receipts),
        "common_panel_rows": len(panel_rows),
        "excluded_meetings": len(exclusions),
        "sample_role_counts": role_counts,
        "exclusions_by_era": exclusion_counts,
        "release_date_range": [
            min(row["release_date"] for row in release_rows),
            max(row["release_date"] for row in release_rows),
        ],
        "meeting_end_date_range": [
            min(row["meeting_end_date"] for row in release_rows),
            max(row["meeting_end_date"] for row in release_rows),
        ],
        "release_lag_calendar_days": {
            "minimum": min(row["release_lag_calendar_days"] for row in release_rows),
            "maximum": max(row["release_lag_calendar_days"] for row in release_rows),
        },
        "market_profiles": market_profiles,
    }
    coverage_path = output / "market_panel.coverage.v1.json"
    _write_new_json(coverage_path, coverage_value)

    manifest = seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "immutable": True,
            "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "research_scope": "exploratory_daily_minutes_release_association",
            "causal_claim_permitted": False,
            "sources": {
                "pre_roster_upstream": _binding(pre_roster, rows=EXPECTED_PRE),
                "post_roster_upstream": _binding(post_roster, rows=EXPECTED_POST),
                "pre_roster_snapshot": _binding(pre_snapshot, rows=EXPECTED_PRE),
                "post_roster_snapshot": _binding(post_snapshot, rows=EXPECTED_POST),
                "sealed_v4_generation_suite": _binding(suite_manifest),
                "generation_cohort_upstream": _binding(cohort_path),
                "generation_cohort_snapshot": _binding(cohort_snapshot),
                "source_sample_manifest_upstream": _binding(sample_manifest_path),
                "source_sample_manifest_snapshot": _binding(sample_manifest_snapshot),
                "market_series": market_bindings,
            },
            "release_date_contract": {
                "authority": "Board of Governors of the Federal Reserve System",
                "accepted_evidence": "explicit Minutes (Released <date>) on bound Fed calendar page",
                "last_update_footer_accepted": False,
                "official_pages_archived": True,
                "parse_key": "exact normalized official_minutes_url path",
                "release_time_available": False,
            },
            "market_contract": {
                "same_sample_across_outcomes_models_and_replicates": True,
                "release_date_must_be_observed": True,
                "trading_day_mapping": "previous_two_and_next_valid_observations_around_exact_release_date",
                "primary_outcome": "dgs2_release_minus_previous_bp",
                "maturity_sensitivities": [
                    "dgs5_release_minus_previous_bp",
                    "dgs10_release_minus_previous_bp",
                ],
                "delayed_response_sensitivities": [
                    "dgs2_next_minus_release_bp",
                    "dgs5_next_minus_release_bp",
                    "dgs10_next_minus_release_bp",
                    "vix_next_minus_release_log_pct",
                ],
                "placebo_outcomes": [
                    "dgs2_placebo_previous_minus_previous_2_bp",
                    "dgs5_placebo_previous_minus_previous_2_bp",
                    "dgs10_placebo_previous_minus_previous_2_bp",
                    "vix_placebo_previous_minus_previous_2_log_pct",
                ],
                "vix_event_sensitivity": "vix_release_minus_previous_log_pct",
                "daily_window_warning": (
                    "Daily closes include non-announcement variation and do not identify "
                    "a narrow causal market reaction."
                ),
            },
            "coverage": coverage_value,
            "artifacts": {
                "release_dates": _binding(release_path, rows=len(release_rows)),
                "market_panel": _binding(panel_path, rows=len(panel_rows)),
                "exclusions": _binding(exclusion_path, rows=len(exclusions)),
                "coverage": _binding(coverage_path),
                "http_receipts": _binding(receipt_path, rows=len(receipts)),
            },
            "implementation": _binding(Path(__file__).resolve()),
            "runtime": {
                "python_executable": sys.executable,
                "python_version": sys.version.split()[0],
                "requests_version": requests.__version__,
                "beautifulsoup4_version": bs4.__version__,
                "pandas_version": pd.__version__,
            },
        }
    )
    manifest_path = output / "manifest.json"
    _write_new_json(manifest_path, manifest)
    _freeze_output_tree(output)
    validated = validate_panel(manifest_path)
    return validated["manifest"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze official FOMC Minutes release dates and a common daily market panel."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pre-roster", type=Path, default=DEFAULT_PRE_ROSTER)
    parser.add_argument("--post-roster", type=Path, default=DEFAULT_POST_ROSTER)
    parser.add_argument("--suite-manifest", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--validate-only", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.validate_only is not None:
            manifest = validate_panel(args.validate_only)["manifest"]
        else:
            manifest = build(
                output_dir=args.output_dir,
                pre_roster=args.pre_roster,
                post_roster=args.post_roster,
                suite_manifest=args.suite_manifest,
                timeout_seconds=args.timeout_seconds,
            )
    except MinutesReleaseMarketPanelError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(
        _canonical(
            {
                "status": manifest["status"],
                "output": str(args.validate_only or (args.output_dir / "manifest.json")),
                "release_dates": manifest["coverage"]["release_dates"],
                "common_panel_rows": manifest["coverage"]["common_panel_rows"],
                "sample_role_counts": manifest["coverage"]["sample_role_counts"],
                "payload_sha256": manifest["integrity"]["payload_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
