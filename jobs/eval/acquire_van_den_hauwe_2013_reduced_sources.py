#!/usr/bin/env python3
"""Acquire the frozen monthly inputs for the reduced van den Hauwe model.

The output deliberately separates two source policies:

* ``IP`` and ``INF`` are reconstructed from end-of-prior-month ALFRED
  vintages.  Each monthly feature therefore uses only observations visible in
  that historical vintage.
* ``6TFF`` is reconstructed from the historical monthly TB6MS and FEDFUNDS
  observations.  These market-rate histories are treated as non-revised
  observations, not as a generic current-vintage substitute.

The target and decision-month closure reproduces the paper's January 1990 to
June 2008 calendar: 222 months, 157 decision months, and 40/86/31
cut/hold/hike outcomes.  The source root is sealed only after every raw file
and every derived row validates.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import tempfile
import time
import urllib.parse
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/main/van_den_hauwe_2013_reduced_v1.json"
DEFAULT_ROSTER = (
    ROOT
    / "output/evaluation/retrain_v2/decision_comparison_direction_only_v4_audit_20260829/action_roster.jsonl"
)
SCHEMA = "van-den-hauwe-2013-reduced-source-v1"
MONTHLY_SCHEMA = "van-den-hauwe-2013-reduced-month-v1"
MONTHS = 222
DECISION_COUNTS = {"cut": 40, "hold": 86, "hike": 31}
PREDICTORS = ("6TFF", "IP", "INF")
ALFRED_SERIES = {"IP": "INDPRO", "INF": "CPIAUCSL"}
MARKET_SERIES = {"TB6MS", "FEDFUNDS"}
FED_HISTORY_URLS = {
    year: f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
    for year in (1990, 1991, 1992)
}
PAPER_URL = "https://papers.tinbergen.nl/11093.pdf"
THESIS_URL = "https://repub.eur.nl/pub/80126/vanderHauwe.pdf"
UNSCHEDULED_DECISION_MONTHS = {
    "1991-01",
    "1991-04",
    "1991-09",
    "1992-04",
    "1992-09",
    "1994-04",
    "1998-10",
    "2001-04",
    "2001-09",
}
PRE1993_REGULAR_MONTHS = {
    f"{year}-{month:02d}"
    for year in (1990, 1991, 1992)
    for month in (2, 3, 5, 7, 8, 10, 11, 12)
}


class SourceAcquisitionError(RuntimeError):
    """A frozen data, timing, or provenance contract failed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceAcquisitionError(f"cannot read JSON {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SourceAcquisitionError(
                        f"JSONL row is not an object: {path}:{lineno}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceAcquisitionError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def atomic_write(path: Path, data: bytes, *, exclusive: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if exclusive else "wb"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, mode.replace("x", "w")) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and path.exists():
            raise FileExistsError(path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def month_range(start: str, end: str) -> list[str]:
    start_year, start_month = (int(part) for part in start.split("-"))
    end_year, end_month = (int(part) for part in end.split("-"))
    values: list[str] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        values.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year += 1
            month = 1
    return values


def previous_month_end(month: str) -> str:
    year, number = (int(part) for part in month.split("-"))
    if number == 1:
        year -= 1
        number = 12
    else:
        number -= 1
    return date(year, number, calendar.monthrange(year, number)[1]).isoformat()


def subtract_years(value: str, years: int) -> str:
    original = date.fromisoformat(value)
    try:
        return original.replace(year=original.year - years).isoformat()
    except ValueError:
        return original.replace(year=original.year - years, day=28).isoformat()


def http_get(
    url: str,
    *,
    timeout_seconds: float,
    retries: int,
    backoff_initial_seconds: float,
    backoff_max_seconds: float,
) -> tuple[bytes, Mapping[str, str]]:
    """Download with curl so FRED/ALFRED can negotiate HTTP/2 reliably.

    In this execution environment Python's ``http.client`` connection to the
    FRED graph endpoint can stall before the response status while curl
    completes the same HTTPS request promptly.  The argument-vector form
    avoids shell interpretation, and the surrounding retry loop remains fully
    deterministic.
    """

    delay = backoff_initial_seconds
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            completed = subprocess.run(
                [
                    "curl",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--http1.1",
                    "--connect-timeout",
                    "10",
                    "--max-time",
                    str(max(1, min(30, math.ceil(timeout_seconds)))),
                    url,
                ],
                check=False,
                capture_output=True,
                timeout=min(timeout_seconds, 30.0) + 5.0,
            )
            if completed.returncode != 0:
                stderr = completed.stderr.decode("utf-8", errors="replace").strip()
                raise SourceAcquisitionError(
                    f"curl exit {completed.returncode}: {stderr or 'no stderr'}"
                )
            body = completed.stdout
            if not body:
                raise SourceAcquisitionError(f"empty HTTP response: {url}")
            return body, {}
        except (OSError, subprocess.SubprocessError, SourceAcquisitionError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(delay)
            delay = min(max(delay * 2.0, 0.1), backoff_max_seconds)
    raise SourceAcquisitionError(
        f"HTTP request failed after {retries} attempts: {url}: {last_error}"
    )


def acquire_file(
    root: Path,
    *,
    provider: str,
    series_id: str,
    url: str,
    suffix: str,
    resume: bool,
    http_options: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    request_id = sha256_bytes(url.encode("utf-8"))
    path = root / "raw" / provider / series_id / f"{request_id}.{suffix}"
    if path.exists():
        if not resume or not path.is_file() or path.is_symlink():
            raise SourceAcquisitionError(f"existing source without valid resume: {path}")
        body = path.read_bytes()
    else:
        body, _headers = http_get(url, **http_options)
        atomic_write(path, body, exclusive=True)
    record = {
        "provider": provider,
        "series_id": series_id,
        "request_id": request_id,
        "canonical_url": url,
        "path": str(path.relative_to(root)),
        "bytes": len(body),
        "sha256": sha256_bytes(body),
    }
    return path, record


def parse_fred_csv(path: Path, series_id: str) -> list[tuple[str, float]]:
    try:
        reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))
        if reader.fieldnames != ["observation_date", series_id]:
            raise SourceAcquisitionError(
                f"unexpected FRED columns for {series_id}: {reader.fieldnames}"
            )
        values: list[tuple[str, float]] = []
        for row in reader:
            raw = row.get(series_id, "")
            if raw in (None, "", "."):
                continue
            value = float(raw)
            if not math.isfinite(value):
                raise SourceAcquisitionError(f"non-finite {series_id} value")
            observation = date.fromisoformat(str(row["observation_date"])).isoformat()
            values.append((observation, value))
    except (OSError, csv.Error, TypeError, ValueError) as exc:
        if isinstance(exc, SourceAcquisitionError):
            raise
        raise SourceAcquisitionError(f"cannot parse {series_id} CSV: {exc}") from exc
    if not values:
        raise SourceAcquisitionError(f"no usable values for {series_id}")
    return values


def parse_alfred_chunk(
    path: Path, *, series_id: str, vintages: Sequence[str]
) -> dict[str, list[tuple[str, float]]]:
    expected = [f"{series_id}_{value.replace('-', '')}" for value in vintages]
    try:
        reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))
        if reader.fieldnames != ["observation_date", *expected]:
            raise SourceAcquisitionError(
                f"unexpected ALFRED columns for {series_id}: {reader.fieldnames}"
            )
        output: dict[str, list[tuple[str, float]]] = {value: [] for value in vintages}
        for row in reader:
            observation = date.fromisoformat(str(row["observation_date"])).isoformat()
            for vintage, column in zip(vintages, expected, strict=True):
                raw = row.get(column, "")
                if raw in (None, "", "."):
                    continue
                value = float(raw)
                if not math.isfinite(value):
                    raise SourceAcquisitionError(
                        f"non-finite {series_id} value for {vintage}"
                    )
                if observation > vintage:
                    raise SourceAcquisitionError(
                        f"future {series_id} observation {observation} in {vintage} vintage"
                    )
                output[vintage].append((observation, value))
    except (OSError, csv.Error, TypeError, ValueError) as exc:
        if isinstance(exc, SourceAcquisitionError):
            raise
        raise SourceAcquisitionError(f"cannot parse ALFRED CSV: {exc}") from exc
    for vintage, values in output.items():
        if len(values) < 2:
            raise SourceAcquisitionError(
                f"ALFRED {series_id} vintage {vintage} has fewer than two values"
            )
        values.sort()
    return output


def latest_for_month(
    values: Sequence[tuple[str, float]], month: str
) -> tuple[str, float]:
    eligible = [(day, value) for day, value in values if day[:7] <= month]
    if not eligible:
        raise SourceAcquisitionError(f"no observation available for {month}")
    return max(eligible, key=lambda item: item[0])


def derive_decision_months(roster: Sequence[Mapping[str, Any]]) -> set[str]:
    regular = {
        str(row["meeting_end_date"])[:7]
        for row in roster
        if "1993-01" <= str(row.get("meeting_end_date", ""))[:7] <= "2008-06"
        and row.get("meeting_type") == "regular"
        and row.get("scheduled") is True
    }
    if len(regular) != 124:
        raise SourceAcquisitionError(
            f"expected 124 post-1992 regular meeting months, found {len(regular)}"
        )
    scheduled = regular | PRE1993_REGULAR_MONTHS
    if len(scheduled) != 148:
        raise SourceAcquisitionError("scheduled meeting-month closure failed")
    decision = scheduled | UNSCHEDULED_DECISION_MONTHS
    if len(decision) != 157:
        raise SourceAcquisitionError("decision meeting-month closure failed")
    return decision


def validate_fed_history_pages(paths: Mapping[int, Path]) -> None:
    expected = {
        year: [
            f"{calendar.month_name[month]}" for month in (2, 3, 5, 7, 8, 10, 11, 12)
        ]
        for year in (1990, 1991, 1992)
    }
    for year, path in paths.items():
        text = path.read_text(encoding="utf-8", errors="replace")
        for month_name in expected[year]:
            if f"{month_name} " not in text or f"Meeting - {year}" not in text:
                raise SourceAcquisitionError(
                    f"Federal Reserve history-page validation failed for {year}-{month_name}"
                )


def build_monthly_rows(
    *,
    months: Sequence[str],
    decision_months: set[str],
    target_values: Sequence[tuple[str, float]],
    market_values: Mapping[str, Sequence[tuple[str, float]]],
    vintage_values: Mapping[str, Mapping[str, Sequence[tuple[str, float]]]],
    raw_hashes: Mapping[str, str],
) -> list[dict[str, Any]]:
    target_by_month: dict[str, tuple[str, float]] = {}
    for observation, value in target_values:
        if "1989-12" <= observation[:7] <= "2008-06":
            current = target_by_month.get(observation[:7])
            if current is None or observation > current[0]:
                target_by_month[observation[:7]] = (observation, value)

    rows: list[dict[str, Any]] = []
    for month in months:
        vintage = previous_month_end(month)
        prior_month = vintage[:7]
        if month not in target_by_month or prior_month not in target_by_month:
            raise SourceAcquisitionError(f"missing target-rate month for {month}")
        target_observation, target_rate = target_by_month[month]
        prior_target_observation, prior_target_rate = target_by_month[prior_month]
        change = target_rate - prior_target_rate
        direction = (
            "cut" if change < -1e-10 else "hike" if change > 1e-10 else "hold"
        )

        tbill_observation, tbill_value = latest_for_month(
            market_values["TB6MS"], prior_month
        )
        funds_observation, funds_value = latest_for_month(
            market_values["FEDFUNDS"], prior_month
        )
        six_tff = tbill_value - funds_value

        predictor_payload: dict[str, Any] = {
            "6TFF": {
                "value": six_tff,
                "transform": "av",
                "averaging_months": 1,
                "observation_period": prior_month,
                "observation_dates": [tbill_observation, funds_observation],
                "availability_date": vintage,
                "vintage_date": None,
                "real_time_vintage": False,
                "source_policy": "historical_market_nonrevised",
                "series_ids": ["TB6MS", "FEDFUNDS"],
                "source_sha256": sha256_bytes(
                    canonical_json(
                        {
                            "TB6MS": raw_hashes["TB6MS"],
                            "FEDFUNDS": raw_hashes["FEDFUNDS"],
                        }
                    ).encode("utf-8")
                ),
            }
        }

        for paper_id in ("IP", "INF"):
            series_id = ALFRED_SERIES[paper_id]
            observations = list(vintage_values[paper_id][vintage])
            latest_two = observations[-2:]
            previous_value = latest_two[0][1]
            latest_value = latest_two[1][1]
            if previous_value == 0:
                raise SourceAcquisitionError(
                    f"zero denominator for {paper_id} in {month}"
                )
            growth = 1200.0 * (latest_value - previous_value) / previous_value
            predictor_payload[paper_id] = {
                "value": growth,
                "transform": "gr",
                "averaging_months": 1,
                "observation_period": latest_two[-1][0][:7],
                "observation_dates": [item[0] for item in latest_two],
                "raw_values": [item[1] for item in latest_two],
                "availability_date": vintage,
                "vintage_date": vintage,
                "real_time_vintage": True,
                "source_policy": "alfred_end_of_prior_month_vintage",
                "series_ids": [series_id],
                "source_sha256": raw_hashes[f"{series_id}:{vintage}"],
            }

        row: dict[str, Any] = {
            "schema_version": MONTHLY_SCHEMA,
            "month": month,
            "information_cutoff": vintage,
            "target_rate_end_month_pct": target_rate,
            "target_rate_observation_date": target_observation,
            "previous_target_rate_pct": prior_target_rate,
            "previous_target_rate_observation_date": prior_target_observation,
            "decision_month": month in decision_months,
            "scheduled_meeting_month": month
            in (decision_months - UNSCHEDULED_DECISION_MONTHS),
            "unscheduled_only_decision_month": month in UNSCHEDULED_DECISION_MONTHS,
            "direction": direction if month in decision_months else None,
            "predictors": predictor_payload,
            "target_source_sha256": raw_hashes["DFEDTAR"],
        }
        row["row_sha256"] = sha256_bytes(canonical_json(row).encode("utf-8"))
        rows.append(row)

    validate_monthly_rows(rows)
    return rows


def validate_monthly_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    expected_months = month_range("1990-01", "2008-06")
    if len(rows) != MONTHS or [row.get("month") for row in rows] != expected_months:
        raise SourceAcquisitionError("monthly population closure failed")
    decision_rows = [row for row in rows if row.get("decision_month") is True]
    if len(decision_rows) != 157:
        raise SourceAcquisitionError("decision-month count closure failed")
    if Counter(str(row.get("direction")) for row in decision_rows) != Counter(
        DECISION_COUNTS
    ):
        raise SourceAcquisitionError("decision direction closure failed")
    if sum(row.get("scheduled_meeting_month") is True for row in rows) != 148:
        raise SourceAcquisitionError("scheduled meeting-month count closure failed")
    if sum(row.get("unscheduled_only_decision_month") is True for row in rows) != 9:
        raise SourceAcquisitionError("unscheduled meeting-month count closure failed")
    for row in rows:
        cutoff = str(row.get("information_cutoff"))
        if cutoff != previous_month_end(str(row.get("month"))):
            raise SourceAcquisitionError("information-cutoff drift")
        predictors = row.get("predictors")
        if not isinstance(predictors, Mapping) or set(predictors) != set(PREDICTORS):
            raise SourceAcquisitionError("predictor population drift")
        for paper_id, payload in predictors.items():
            if not isinstance(payload, Mapping):
                raise SourceAcquisitionError(f"invalid {paper_id} payload")
            value = float(payload.get("value"))
            if not math.isfinite(value):
                raise SourceAcquisitionError(f"non-finite {paper_id} feature")
            availability = str(payload.get("availability_date"))
            if availability > cutoff:
                raise SourceAcquisitionError(f"future {paper_id} availability")
            if paper_id in {"IP", "INF"}:
                if (
                    payload.get("real_time_vintage") is not True
                    or payload.get("vintage_date") != cutoff
                    or payload.get("source_policy")
                    != "alfred_end_of_prior_month_vintage"
                ):
                    raise SourceAcquisitionError(
                        f"{paper_id} real-time vintage contract failed"
                    )
            elif (
                payload.get("source_policy") != "historical_market_nonrevised"
                or payload.get("real_time_vintage") is not False
            ):
                raise SourceAcquisitionError("6TFF source policy failed")
        expected_hash = sha256_bytes(
            canonical_json({key: value for key, value in row.items() if key != "row_sha256"}).encode(
                "utf-8"
            )
        )
        if row.get("row_sha256") != expected_hash:
            raise SourceAcquisitionError("monthly row hash mismatch")


def validate_sealed_root(
    root: Path,
    *,
    config_path: Path | None = None,
    roster_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    monthly_path = root / "monthly_source.jsonl"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA:
        raise SourceAcquisitionError("source manifest schema mismatch")
    if config_path is not None:
        expected_config = str(config_path.relative_to(ROOT))
        if (
            manifest.get("config") != expected_config
            or manifest.get("config_sha256") != sha256_file(config_path)
        ):
            raise SourceAcquisitionError("sealed source config binding drift")
    if roster_path is not None:
        expected_roster = str(roster_path.relative_to(ROOT))
        if (
            manifest.get("roster") != expected_roster
            or manifest.get("roster_sha256") != sha256_file(roster_path)
        ):
            raise SourceAcquisitionError("sealed source roster binding drift")
    validate_monthly_rows(read_jsonl(monthly_path))
    files = manifest.get("files")
    if not isinstance(files, list):
        raise SourceAcquisitionError("source manifest files missing")
    for record in files:
        path = root / str(record.get("path"))
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record.get("bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise SourceAcquisitionError(f"sealed source file drift: {path}")
    if manifest.get("monthly_source_sha256") != sha256_file(monthly_path):
        raise SourceAcquisitionError("monthly source manifest hash mismatch")
    return manifest


def acquire(
    *, config_path: Path, roster_path: Path, output_root: Path, resume: bool
) -> dict[str, Any]:
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"source root already sealed: {output_root}")
        return validate_sealed_root(
            output_root, config_path=config_path, roster_path=roster_path
        )
    if output_root.exists() and not resume:
        raise FileExistsError(f"source root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    config = read_json(config_path)
    sample = config.get("sample", {})
    months = month_range(str(sample.get("start_month")), str(sample.get("end_month")))
    if len(months) != MONTHS:
        raise SourceAcquisitionError("frozen configuration sample drift")
    roster = read_jsonl(roster_path)
    decision_months = derive_decision_months(roster)
    http_options = {
        "timeout_seconds": 180.0,
        "retries": 6,
        "backoff_initial_seconds": 1.0,
        "backoff_max_seconds": 30.0,
    }
    files: list[dict[str, Any]] = []

    fed_paths: dict[int, Path] = {}
    for year, url in FED_HISTORY_URLS.items():
        path, record = acquire_file(
            output_root,
            provider="federal_reserve",
            series_id=f"fomc_history_{year}",
            url=url,
            suffix="html",
            resume=resume,
            http_options=http_options,
        )
        fed_paths[year] = path
        files.append(record)
    validate_fed_history_pages(fed_paths)

    paper_path, paper_record = acquire_file(
        output_root,
        provider="tinbergen",
        series_id="working_paper_11093",
        url=PAPER_URL,
        suffix="pdf",
        resume=resume,
        http_options=http_options,
    )
    if not paper_path.read_bytes().startswith(b"%PDF"):
        raise SourceAcquisitionError("Tinbergen working-paper response is not a PDF")
    files.append(paper_record)

    thesis_path, thesis_record = acquire_file(
        output_root,
        provider="erasmus_research_repository",
        series_id="van_den_hauwe_thesis_2015",
        url=THESIS_URL,
        suffix="pdf",
        resume=resume,
        http_options=http_options,
    )
    if not thesis_path.read_bytes().startswith(b"%PDF"):
        raise SourceAcquisitionError("van den Hauwe thesis response is not a PDF")
    files.append(thesis_record)

    fred_base = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    fred_paths: dict[str, Path] = {}
    for series_id, start, end in (
        ("DFEDTAR", "1989-12-01", "2008-06-30"),
        ("TB6MS", "1988-01-01", "2008-05-31"),
        ("FEDFUNDS", "1988-01-01", "2008-05-31"),
    ):
        url = f"{fred_base}?{urllib.parse.urlencode({'id': series_id, 'cosd': start, 'coed': end})}"
        path, record = acquire_file(
            output_root,
            provider="fred",
            series_id=series_id,
            url=url,
            suffix="csv",
            resume=resume,
            http_options=http_options,
        )
        parse_fred_csv(path, series_id)
        fred_paths[series_id] = path
        files.append(record)

    vintage_dates = [previous_month_end(month) for month in months]
    all_vintages: dict[str, dict[str, list[tuple[str, float]]]] = {
        "IP": {},
        "INF": {},
    }
    raw_hashes: dict[str, str] = {
        series_id: sha256_file(path) for series_id, path in fred_paths.items()
    }
    alfred_base = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
    chunk_size = 12
    for paper_id, series_id in ALFRED_SERIES.items():
        for offset in range(0, len(vintage_dates), chunk_size):
            vintages = vintage_dates[offset : offset + chunk_size]
            starts = [subtract_years(value, 3) for value in vintages]
            params = {
                "id": ",".join([series_id] * len(vintages)),
                "cosd": ",".join(starts),
                "coed": ",".join(vintages),
                "vintage_date": ",".join(vintages),
            }
            url = f"{alfred_base}?{urllib.parse.urlencode(params)}"
            path, record = acquire_file(
                output_root,
                provider="alfred",
                series_id=series_id,
                url=url,
                suffix="csv",
                resume=resume,
                http_options=http_options,
            )
            parsed = parse_alfred_chunk(path, series_id=series_id, vintages=vintages)
            if set(parsed) & set(all_vintages[paper_id]):
                raise SourceAcquisitionError("duplicate ALFRED vintage")
            all_vintages[paper_id].update(parsed)
            for vintage in vintages:
                raw_hashes[f"{series_id}:{vintage}"] = record["sha256"]
            record["vintage_dates"] = vintages
            files.append(record)
            print(
                canonical_json(
                    {
                        "event": "source_request_complete",
                        "paper_id": paper_id,
                        "vintage_first": vintages[0],
                        "vintage_last": vintages[-1],
                    }
                ),
                flush=True,
            )

    rows = build_monthly_rows(
        months=months,
        decision_months=decision_months,
        target_values=parse_fred_csv(fred_paths["DFEDTAR"], "DFEDTAR"),
        market_values={
            series_id: parse_fred_csv(fred_paths[series_id], series_id)
            for series_id in MARKET_SERIES
        },
        vintage_values=all_vintages,
        raw_hashes=raw_hashes,
    )
    monthly_path = output_root / "monthly_source.jsonl"
    payload = b"".join(
        canonical_json(row).encode("utf-8") + b"\n" for row in rows
    )
    atomic_write(monthly_path, payload, exclusive=True)
    monthly_record = {
        "provider": "derived",
        "series_id": "monthly_source",
        "path": "monthly_source.jsonl",
        "bytes": monthly_path.stat().st_size,
        "sha256": sha256_file(monthly_path),
    }
    files.append(monthly_record)

    manifest = {
        "schema_version": SCHEMA,
        "created_at_utc": utc_now(),
        "replication_claim": config.get("replication_claim"),
        "sample": {
            "start_month": months[0],
            "end_month": months[-1],
            "months": len(months),
            "decision_months": 157,
            "scheduled_meeting_months": 148,
            "unscheduled_only_decision_months": 9,
            "direction_counts": DECISION_COUNTS,
        },
        "predictors": [
            {
                "paper_id": "6TFF",
                "series_ids": ["TB6MS", "FEDFUNDS"],
                "transform": "av",
                "m": 1,
                "source_policy": "historical_market_nonrevised",
            },
            {
                "paper_id": "IP",
                "series_ids": ["INDPRO"],
                "transform": "gr",
                "m": 1,
                "source_policy": "alfred_end_of_prior_month_vintage",
            },
            {
                "paper_id": "INF",
                "series_ids": ["CPIAUCSL"],
                "transform": "gr",
                "m": 1,
                "source_policy": "alfred_end_of_prior_month_vintage",
            },
        ],
        "timing_policy": "forecast month t uses information visible at end of month t-1",
        "current_vintage_fallback_allowed": False,
        "market_history_caveat": (
            "TB6MS and FEDFUNDS use downloaded historical observations because early "
            "ALFRED vintage columns are unavailable; these market-rate series are treated "
            "as non-revised histories, with later source corrections still possible."
        ),
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256_file(config_path),
        "roster": str(roster_path.relative_to(ROOT)),
        "roster_sha256": sha256_file(roster_path),
        "monthly_source_sha256": sha256_file(monthly_path),
        "files": sorted(files, key=lambda record: str(record["path"])),
    }
    atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        exclusive=True,
    )
    return validate_sealed_root(
        output_root, config_path=config_path, roster_path=roster_path
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--roster", default=str(DEFAULT_ROSTER))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    roster_path = Path(args.roster).resolve()
    output_root = Path(args.output_root).resolve()
    manifest = acquire(
        config_path=config_path,
        roster_path=roster_path,
        output_root=output_root,
        resume=bool(args.resume),
    )
    print(
        canonical_json(
            {
                "event": "source_root_sealed",
                "output_root": str(output_root),
                "monthly_source_sha256": manifest["monthly_source_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
