from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from jobs.main.fetch_loo_source_snapshots import HttpResult
from jobs.retrain_v2.chk1.source_data import (
    evidence_from_loo_ledger,
    stable_sample_id,
)
from jobs.retrain_v2.chk1.sparse_source import (
    OUTCOME_AVAILABLE,
    OUTCOME_SPLIT,
    OUTCOME_UNUSABLE,
    REASON_HTTP_404,
    REASON_NO_OBSERVATION_IN_SAMPLING_WINDOW,
    REASON_NOT_ECHOED,
    REASON_TRANSPORT_RETRY_EXHAUSTED,
    SAMPLE_EXCLUSION_REASON,
    SparseSourceError,
    SparseSourceIntegrityError,
    acquire_sparse_source_snapshots,
    build_sparse_loo_ledger,
    validate_sparse_loo_ledger,
    validate_sparse_snapshot_manifest,
)
from jobs.retrain_v2.chk1.topic_styles import (
    ATOMIC_TOPICS,
    ledger_indicator_for_topic,
)


ROSTER = [ledger_indicator_for_topic(topic) for topic in ATOMIC_TOPICS]
BANK_CAPITAL = ledger_indicator_for_topic("Bank Capital")
SOURCES = {
    "primary": ("alfred__PRIMARY", "PRIMARY"),
    "alternate": ("alfred__ALT", "ALT"),
    "common": ("alfred__COMMON", "COMMON"),
}
FIXED_TIME = "2026-08-03T12:00:00Z"


class _NoOpLimiter:
    def acquire(self) -> None:
        return None


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _source(series_id: str) -> dict[str, object]:
    return {
        "series_id": series_id,
        "provider": "ALFRED",
        "access_interface": "alfred-graph-csv-v1",
        "frequency": "daily",
        "units": "Index",
        "seasonal_adjustment": "Not Seasonally Adjusted",
        "lookback": {"unit": "months", "value": 24},
        "selection": {
            "method": "period_end_plus_latest",
            "period": "month",
            "max_points": 25,
        },
        "transformation": {"method": "identity"},
        "license": "public-domain",
        "redistribution_allowed": True,
        "enabled": True,
    }


def _configuration(root: Path) -> tuple[Path, Path]:
    roster_path = root / "roster.json"
    registry_path = root / "registry.json"
    _write_json(roster_path, {"indicators": ROSTER})
    indicators = {
        indicator: {
            "source_keys": (
                [SOURCES["primary"][0], SOURCES["alternate"][0]]
                if indicator == BANK_CAPITAL
                else [SOURCES["common"][0]]
            )
        }
        for indicator in ROSTER
    }
    _write_json(
        registry_path,
        {
            "schema_version": "loo-indicator-source-registry-v1",
            "registry_id": "chk1-sparse-test-registry",
            "series_defaults": {
                "provider": "ALFRED",
                "access_interface": "alfred-graph-csv-v1",
                "date_column": "observation_date",
                "value_column_template": "{series_id}_{vintage_date_yyyymmdd}",
                "lookback": {"unit": "months", "value": 24},
                "selection": {
                    "method": "period_end_plus_latest",
                    "period": "month",
                    "max_points": 25,
                },
                "transformation": {"method": "identity"},
                "enabled": True,
            },
            "policy": {
                "network_mode": "keyless",
                "meeting_timezone": "America/New_York",
                "vintage_lag_calendar_days": 1,
                "same_meeting_day_data": "excluded",
                "unknown_availability": "fail_closed",
                "runtime_fallback": "forbidden",
                "local_legacy_values": "mapping_only",
                "synthetic_text_values": "mapping_only",
                "access_interface": "alfred-graph-csv-v1",
                "request_url_template": (
                    "https://alfred.stlouisfed.org/graph/alfredgraph.csv?"
                    "id={series_id}&vintage_date={vintage_date}"
                ),
                "availability_evidence_type": "alfred_vintage_snapshot",
                "expected_http_status": 200,
                "raw_response_required": True,
                "raw_response_sha256_required": True,
                "restricted_artifacts_git_policy": "gitignored",
                "ledger_git_policy": "gitignored",
            },
            "sources": {
                source_key: _source(series_id)
                for source_key, series_id in SOURCES.values()
            },
            "indicators": indicators,
        },
    )
    return registry_path, roster_path


def _meeting_dates(count: int) -> list[str]:
    start = date(2020, 1, 2)
    return [(start + timedelta(days=28 * index)).isoformat() for index in range(count)]


def _exact_csv(
    series_id: str, vintages: tuple[str, ...], *, future: bool = False
) -> bytes:
    first_vintage = min(date.fromisoformat(value) for value in vintages)
    observation = (
        max(date.fromisoformat(value) for value in vintages) + timedelta(days=1)
        if future
        else first_vintage - timedelta(days=10)
    )
    header = ["observation_date"] + [
        f"{series_id}_{value.replace('-', '')}" for value in vintages
    ]
    values = [str(index + 1) for index in range(len(vintages))]
    return (
        ",".join(header) + "\n" + ",".join([observation.isoformat(), *values]) + "\n"
    ).encode()


def _not_echoed_csv(series_id: str, vintages: tuple[str, ...]) -> bytes:
    observation = min(date.fromisoformat(value) for value in vintages) - timedelta(
        days=10
    )
    return (
        f"observation_date,{series_id}_20990101\n{observation.isoformat()},999999\n"
    ).encode()


def _stale_csv(series_id: str, vintages: tuple[str, ...]) -> bytes:
    observation = min(date.fromisoformat(value) for value in vintages) - timedelta(
        days=3 * 365
    )
    header = ["observation_date"] + [
        f"{series_id}_{value.replace('-', '')}" for value in vintages
    ]
    values = [str(index + 1) for index in range(len(vintages))]
    return (
        ",".join(header)
        + "\n"
        + ",".join([observation.isoformat(), *values])
        + "\n"
    ).encode()


def _transport(
    behavior: dict[tuple[str, str], str] | None = None,
) -> tuple[Callable[[str, float], HttpResult], list[tuple[str, tuple[str, ...]]]]:
    rules = behavior or {}
    calls: list[tuple[str, tuple[str, ...]]] = []

    def http_get(url: str, _timeout: float) -> HttpResult:
        query = parse_qs(urlparse(url).query)
        ids = tuple(query["id"][0].split(","))
        series_id = ids[0]
        assert set(ids) == {series_id}
        vintages = tuple(query["vintage_date"][0].split(","))
        calls.append((series_id, vintages))
        modes = {rules.get((series_id, vintage), "exact") for vintage in vintages}
        if "timeout" in modes:
            raise TimeoutError("synthetic timeout")
        if "503" in modes:
            return HttpResult(503, {"content-type": "text/plain"}, b"busy")
        if "404" in modes:
            return HttpResult(404, {"content-type": "text/html"}, b"")
        if "not_echoed" in modes:
            return HttpResult(
                200,
                {"content-type": "application/csv"},
                _not_echoed_csv(series_id, vintages),
            )
        if "wrong_series" in modes:
            return HttpResult(
                200,
                {"content-type": "application/csv"},
                _not_echoed_csv("WRONG", vintages),
            )
        if "malformed" in modes:
            return HttpResult(
                200,
                {"content-type": "application/csv"},
                b"not,a,valid,alfred,header\n",
            )
        if "stale" in modes:
            return HttpResult(
                200,
                {"content-type": "application/csv"},
                _stale_csv(series_id, vintages),
            )
        return HttpResult(
            200,
            {"content-type": "application/csv"},
            _exact_csv(series_id, vintages, future="future" in modes),
        )

    return http_get, calls


def _acquire(
    root: Path,
    *,
    dates: list[str],
    behavior: dict[tuple[str, str], str] | None = None,
    name: str = "snapshots",
) -> tuple[Path, Path, Path, list[tuple[str, tuple[str, ...]]]]:
    registry, roster = _configuration(root)
    http_get, calls = _transport(behavior)
    manifest = acquire_sparse_source_snapshots(
        registry_file=registry,
        roster_file=roster,
        meeting_dates=dates,
        output_dir=root / name,
        population_id="population-test",
        http_get=http_get,
        rate_limiter=_NoOpLimiter(),
        max_retries=0,
        utc_now=lambda: FIXED_TIME,
    )
    return manifest, registry, roster, calls


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_batch_success_manifest_closure_and_offline_resume(tmp_path: Path) -> None:
    dates = _meeting_dates(13)
    manifest_path, registry, roster, calls = _acquire(tmp_path, dates=dates)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(calls) == 6  # three series, each chunked as 12 + 1
    assert [len(item["vintage_dates"]) for item in manifest["attempts"]] == [
        12,
        1,
        12,
        1,
        12,
        1,
    ]
    assert manifest["source_vintage_count"] == 39
    assert manifest["available_source_vintage_count"] == 39
    assert manifest["unusable_source_vintage_count"] == 0
    assert (
        len(
            {
                (item["source_key"], item["meeting_date"])
                for item in manifest["terminal_outcomes"]
            }
        )
        == 39
    )

    validation = validate_sparse_snapshot_manifest(
        manifest_path,
        registry_file=registry,
        roster_file=roster,
        expected_meeting_dates=dates,
        expected_population_id="population-test",
    )
    assert validation["status"] == "valid"

    resumed = acquire_sparse_source_snapshots(
        registry_file=registry,
        roster_file=roster,
        meeting_dates=dates,
        output_dir=manifest_path.parent,
        population_id="population-test",
        resume=True,
        http_get=lambda _url, _timeout: pytest.fail("resume accessed the network"),
        rate_limiter=_NoOpLimiter(),
    )
    assert resumed == manifest_path
    with pytest.raises(FileExistsError):
        acquire_sparse_source_snapshots(
            registry_file=registry,
            roster_file=roster,
            meeting_dates=dates,
            output_dir=manifest_path.parent,
            population_id="population-test",
            rate_limiter=_NoOpLimiter(),
        )


def test_recursive_bisection_seals_404_and_not_echoed_raw_bytes(tmp_path: Path) -> None:
    dates = _meeting_dates(4)
    primary_vintage = (date.fromisoformat(dates[1]) - timedelta(days=1)).isoformat()
    alternate_vintage = (date.fromisoformat(dates[2]) - timedelta(days=1)).isoformat()
    behavior = {
        (SOURCES["primary"][1], primary_vintage): "not_echoed",
        (SOURCES["alternate"][1], alternate_vintage): "404",
    }
    manifest_path, registry, roster, calls = _acquire(
        tmp_path, dates=dates, behavior=behavior
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    outcomes = {
        (item["series_id"], item["information_as_of_date"]): item
        for item in manifest["terminal_outcomes"]
    }
    primary = outcomes[(SOURCES["primary"][1], primary_vintage)]
    alternate = outcomes[(SOURCES["alternate"][1], alternate_vintage)]
    assert primary["outcome"] == OUTCOME_UNUSABLE
    assert primary["reason_code"] == REASON_NOT_ECHOED
    assert primary["status_code"] == 200
    assert alternate["outcome"] == OUTCOME_UNUSABLE
    assert alternate["reason_code"] == REASON_HTTP_404
    assert alternate["status_code"] == 404
    assert any(len(vintages) > 1 for _, vintages in calls)
    assert any(vintages == (primary_vintage,) for _, vintages in calls)
    assert any(vintages == (alternate_vintage,) for _, vintages in calls)

    primary_attempt = next(
        item
        for item in manifest["attempts"]
        if item["request_id"] == primary["request_id"]
    )
    assert primary_attempt["terminal"] is True
    assert primary_attempt["expected_header"] == [
        "observation_date",
        f"PRIMARY_{primary_vintage.replace('-', '')}",
    ]
    assert primary_attempt["observed_header"] == [
        "observation_date",
        "PRIMARY_20990101",
    ]
    raw_path = manifest_path.parent / primary["raw_relative_path"]
    raw = raw_path.read_bytes()
    assert b"999999" in raw
    assert hashlib.sha256(raw).hexdigest() == primary["raw_sha256"]
    validate_sparse_snapshot_manifest(
        manifest_path,
        registry_file=registry,
        roster_file=roster,
    )


def test_batch_transport_timeout_is_cached_and_bisected_for_offline_replay(
    tmp_path: Path,
) -> None:
    dates = _meeting_dates(4)
    registry, roster = _configuration(tmp_path)
    calls: list[tuple[str, tuple[str, ...]]] = []

    def http_get(url: str, _timeout: float) -> HttpResult:
        query = parse_qs(urlparse(url).query)
        series_id = query["id"][0].split(",")[0]
        vintages = tuple(query["vintage_date"][0].split(","))
        calls.append((series_id, vintages))
        if len(vintages) > 2:
            raise TimeoutError("synthetic batch-only timeout")
        return HttpResult(
            200,
            {"content-type": "application/csv"},
            _exact_csv(series_id, vintages),
        )

    manifest_path = acquire_sparse_source_snapshots(
        registry_file=registry,
        roster_file=roster,
        meeting_dates=dates,
        output_dir=tmp_path / "snapshots-transport-split",
        population_id="population-test",
        http_get=http_get,
        rate_limiter=_NoOpLimiter(),
        max_retries=0,
        utc_now=lambda: FIXED_TIME,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_attempts = [
        item for item in manifest["attempts"] if item["outcome"] == OUTCOME_SPLIT
    ]

    assert len(split_attempts) == 3
    assert all(item["terminal"] is False for item in split_attempts)
    assert all(item["status_code"] is None for item in split_attempts)
    assert all(
        item["reason_code"] == REASON_TRANSPORT_RETRY_EXHAUSTED
        for item in split_attempts
    )
    assert all(
        item["transport_failure"]
        == {
            "attempt_count": 1,
            "error_message": "synthetic batch-only timeout",
            "error_type": "TimeoutError",
        }
        for item in split_attempts
    )
    assert all(
        item["outcome"] == OUTCOME_AVAILABLE
        for item in manifest["terminal_outcomes"]
    )
    assert any(len(vintages) == 4 for _, vintages in calls)
    assert any(len(vintages) == 2 for _, vintages in calls)
    validate_sparse_snapshot_manifest(
        manifest_path,
        registry_file=registry,
        roster_file=roster,
    )

    split_raw = manifest_path.parent / split_attempts[0]["raw_relative_path"]
    split_raw.write_bytes(split_raw.read_bytes() + b"tampered")
    with pytest.raises(SparseSourceIntegrityError, match="hash/size mismatch"):
        validate_sparse_snapshot_manifest(
            manifest_path,
            registry_file=registry,
            roster_file=roster,
        )


def test_transport_bisection_only_marks_singleton_404_unusable(tmp_path: Path) -> None:
    dates = _meeting_dates(2)
    target_vintage = (date.fromisoformat(dates[0]) - timedelta(days=1)).isoformat()
    registry, roster = _configuration(tmp_path)

    def http_get(url: str, _timeout: float) -> HttpResult:
        query = parse_qs(urlparse(url).query)
        series_id = query["id"][0].split(",")[0]
        vintages = tuple(query["vintage_date"][0].split(","))
        if len(vintages) > 1:
            raise TimeoutError("synthetic multi-vintage timeout")
        if series_id == SOURCES["primary"][1] and vintages == (target_vintage,):
            return HttpResult(404, {"content-type": "text/html"}, b"")
        return HttpResult(
            200,
            {"content-type": "application/csv"},
            _exact_csv(series_id, vintages),
        )

    manifest_path = acquire_sparse_source_snapshots(
        registry_file=registry,
        roster_file=roster,
        meeting_dates=dates,
        output_dir=tmp_path / "snapshots-transport-singleton-404",
        population_id="population-test",
        http_get=http_get,
        rate_limiter=_NoOpLimiter(),
        max_retries=0,
        utc_now=lambda: FIXED_TIME,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = next(
        item
        for item in manifest["terminal_outcomes"]
        if item["series_id"] == SOURCES["primary"][1]
        and item["information_as_of_date"] == target_vintage
    )

    assert target["outcome"] == OUTCOME_UNUSABLE
    assert target["reason_code"] == REASON_HTTP_404
    assert target["status_code"] == 404
    assert any(item["outcome"] == OUTCOME_SPLIT for item in manifest["attempts"])
    assert not any(
        item["outcome"] == OUTCOME_UNUSABLE
        and len(item["vintage_dates"]) > 1
        and item["reason_code"] == REASON_TRANSPORT_RETRY_EXHAUSTED
        for item in manifest["attempts"]
    )
    validate_sparse_snapshot_manifest(
        manifest_path,
        registry_file=registry,
        roster_file=roster,
    )


@pytest.mark.parametrize(
    "mode", ["503", "timeout", "wrong_series", "malformed", "future"]
)
def test_unknown_transport_format_and_future_values_abort(
    tmp_path: Path, mode: str
) -> None:
    dates = _meeting_dates(1)
    vintage = (date.fromisoformat(dates[0]) - timedelta(days=1)).isoformat()
    registry, roster = _configuration(tmp_path)
    http_get, _ = _transport({(SOURCES["primary"][1], vintage): mode})
    with pytest.raises(SparseSourceError):
        acquire_sparse_source_snapshots(
            registry_file=registry,
            roster_file=roster,
            meeting_dates=dates,
            output_dir=tmp_path / f"snapshots-{mode}",
            population_id="population-test",
            http_get=http_get,
            rate_limiter=_NoOpLimiter(),
            max_retries=0,
            utc_now=lambda: FIXED_TIME,
        )
    assert not (tmp_path / f"snapshots-{mode}" / "snapshot_manifest.json").exists()


def test_resume_and_offline_validation_reject_cache_tampering(tmp_path: Path) -> None:
    dates = _meeting_dates(1)
    manifest_path, registry, roster, _ = _acquire(tmp_path, dates=dates)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = next(
        item for item in manifest["attempts"] if item["outcome"] == OUTCOME_AVAILABLE
    )
    raw_path = manifest_path.parent / first["raw_relative_path"]
    raw_path.write_bytes(raw_path.read_bytes() + b"tampered")

    with pytest.raises(SparseSourceIntegrityError, match="hash/size mismatch"):
        validate_sparse_snapshot_manifest(
            manifest_path,
            registry_file=registry,
            roster_file=roster,
        )
    with pytest.raises(SparseSourceIntegrityError):
        acquire_sparse_source_snapshots(
            registry_file=registry,
            roster_file=roster,
            meeting_dates=dates,
            output_dir=manifest_path.parent,
            population_id="population-test",
            resume=True,
            http_get=lambda _url, _timeout: pytest.fail("resume accessed network"),
            rate_limiter=_NoOpLimiter(),
        )


def test_sparse_ledger_keeps_all_successful_sources_and_partial_topic(
    tmp_path: Path,
) -> None:
    dates = _meeting_dates(2)
    first_vintage = (date.fromisoformat(dates[0]) - timedelta(days=1)).isoformat()
    manifest_path, registry, roster, _ = _acquire(
        tmp_path,
        dates=dates,
        behavior={(SOURCES["primary"][1], first_vintage): "not_echoed"},
    )
    ledger_manifest = build_sparse_loo_ledger(
        snapshot_manifest_file=manifest_path,
        registry_file=registry,
        roster_file=roster,
        output_dir=tmp_path / "ledger",
    )
    ledger_dir = ledger_manifest.parent
    rows = _read_jsonl(ledger_dir / "indicator_inputs.jsonl")
    evidence = _read_jsonl(ledger_dir / "source_evidence.jsonl")
    exclusions = _read_jsonl(ledger_dir / "sample_exclusions.jsonl")
    coverage = json.loads((ledger_dir / "coverage.json").read_text(encoding="utf-8"))

    assert len(rows) == 2 * 26
    assert len(evidence) == len(rows)
    assert exclusions == []
    assert coverage["sample_coverage_complete"] is True
    assert coverage["ready_sample_count"] == 52
    first_bank = next(
        row
        for row in rows
        if row["meeting_date"] == dates[0] and row["indicator"] == BANK_CAPITAL
    )
    assert [item["source_key"] for item in first_bank["source_payload"]["series"]] == [
        SOURCES["alternate"][0]
    ]
    first_evidence = next(
        item for item in evidence if item["sample_id"] == first_bank["sample_id"]
    )
    projected = evidence_from_loo_ledger(
        first_bank,
        first_evidence,
        atomic_topic="Bank Capital",
    )
    assert {item["series_id"] for item in projected} == {SOURCES["alternate"][1]}
    second_bank = next(
        row
        for row in rows
        if row["meeting_date"] == dates[1] and row["indicator"] == BANK_CAPITAL
    )
    assert [item["source_key"] for item in second_bank["source_payload"]["series"]] == [
        SOURCES["primary"][0],
        SOURCES["alternate"][0],
    ]
    assert "999999" not in (ledger_dir / "indicator_inputs.jsonl").read_text(
        encoding="utf-8"
    )
    validation = validate_sparse_loo_ledger(
        ledger_manifest,
        snapshot_manifest_file=manifest_path,
        registry_file=registry,
        roster_file=roster,
    )
    assert validation["ready_sample_count"] == 52
    with pytest.raises(FileExistsError):
        build_sparse_loo_ledger(
            snapshot_manifest_file=manifest_path,
            registry_file=registry,
            roster_file=roster,
            output_dir=ledger_dir,
        )


def test_zero_source_becomes_release_exclusion_and_replay_detects_tamper(
    tmp_path: Path,
) -> None:
    dates = _meeting_dates(1)
    vintage = (date.fromisoformat(dates[0]) - timedelta(days=1)).isoformat()
    manifest_path, registry, roster, _ = _acquire(
        tmp_path,
        dates=dates,
        behavior={
            (SOURCES["primary"][1], vintage): "404",
            (SOURCES["alternate"][1], vintage): "not_echoed",
        },
    )
    ledger_manifest = build_sparse_loo_ledger(
        snapshot_manifest_file=manifest_path,
        registry_file=registry,
        roster_file=roster,
        output_dir=tmp_path / "ledger",
    )
    ledger_dir = ledger_manifest.parent
    rows = _read_jsonl(ledger_dir / "indicator_inputs.jsonl")
    exclusions = _read_jsonl(ledger_dir / "sample_exclusions.jsonl")
    coverage = json.loads((ledger_dir / "coverage.json").read_text(encoding="utf-8"))

    assert len(rows) == 25
    assert len(exclusions) == 1
    exclusion = exclusions[0]
    assert exclusion["sample_id"] == stable_sample_id(dates[0], "Bank Capital")
    assert exclusion["split"] == "train"
    assert exclusion["atomic_topic"] == "Bank Capital"
    assert exclusion["reason_code"] == SAMPLE_EXCLUSION_REASON
    assert {item["reason_code"] for item in exclusion["unusable_sources"]} == {
        REASON_HTTP_404,
        REASON_NOT_ECHOED,
    }
    assert coverage["expected_sample_count"] == 26
    assert coverage["ready_sample_count"] == 25
    assert coverage["excluded_sample_count"] == 1
    assert coverage["sample_coverage_complete"] is True

    validation = validate_sparse_loo_ledger(
        ledger_manifest,
        snapshot_manifest_file=manifest_path,
        registry_file=registry,
        roster_file=roster,
    )
    assert validation == {
        "status": "valid",
        "schema_version": "chk1-sparse-loo-ledger-manifest-v1",
        "population_id": "population-test",
        "manifest_payload_sha256": validation["manifest_payload_sha256"],
        "ready_sample_count": 25,
        "excluded_sample_count": 1,
        "expected_sample_count": 26,
    }

    ledger_path = ledger_dir / "indicator_inputs.jsonl"
    ledger_path.write_text(
        ledger_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(SparseSourceIntegrityError, match="cannot be replayed"):
        validate_sparse_loo_ledger(
            ledger_manifest,
            snapshot_manifest_file=manifest_path,
            registry_file=registry,
            roster_file=roster,
        )


def test_stale_high_frequency_source_is_skipped_when_alternate_survives(
    tmp_path: Path,
) -> None:
    dates = _meeting_dates(1)
    vintage = (date.fromisoformat(dates[0]) - timedelta(days=1)).isoformat()
    manifest_path, registry, roster, _ = _acquire(
        tmp_path,
        dates=dates,
        behavior={(SOURCES["primary"][1], vintage): "stale"},
    )
    ledger_manifest = build_sparse_loo_ledger(
        snapshot_manifest_file=manifest_path,
        registry_file=registry,
        roster_file=roster,
        output_dir=tmp_path / "ledger-stale-partial",
    )
    ledger_dir = ledger_manifest.parent
    rows = _read_jsonl(ledger_dir / "indicator_inputs.jsonl")
    bank = next(row for row in rows if row["indicator"] == BANK_CAPITAL)
    assert [
        item["source_key"] for item in bank["source_payload"]["series"]
    ] == [SOURCES["alternate"][0]]
    coverage = json.loads((ledger_dir / "coverage.json").read_text(encoding="utf-8"))
    bank_coverage = next(
        row for row in coverage["rows"] if row["indicator"] == BANK_CAPITAL
    )
    assert bank_coverage["status"] == "ready"
    assert bank_coverage["source_exclusions"][0]["reason_code"] == (
        REASON_NO_OBSERVATION_IN_SAMPLING_WINDOW
    )
    assert _read_jsonl(ledger_dir / "sample_exclusions.jsonl") == []
    assert validate_sparse_loo_ledger(
        ledger_manifest,
        snapshot_manifest_file=manifest_path,
        registry_file=registry,
        roster_file=roster,
    )["status"] == "valid"


def test_all_stale_high_frequency_sources_exclude_topic_with_replayable_proof(
    tmp_path: Path,
) -> None:
    dates = _meeting_dates(1)
    vintage = (date.fromisoformat(dates[0]) - timedelta(days=1)).isoformat()
    manifest_path, registry, roster, _ = _acquire(
        tmp_path,
        dates=dates,
        behavior={
            (SOURCES["primary"][1], vintage): "stale",
            (SOURCES["alternate"][1], vintage): "stale",
        },
    )
    ledger_manifest = build_sparse_loo_ledger(
        snapshot_manifest_file=manifest_path,
        registry_file=registry,
        roster_file=roster,
        output_dir=tmp_path / "ledger-stale-all",
    )
    ledger_dir = ledger_manifest.parent
    assert len(_read_jsonl(ledger_dir / "indicator_inputs.jsonl")) == 25
    exclusions = _read_jsonl(ledger_dir / "sample_exclusions.jsonl")
    assert len(exclusions) == 1
    assert {
        item["reason_code"] for item in exclusions[0]["unusable_sources"]
    } == {REASON_NO_OBSERVATION_IN_SAMPLING_WINDOW}
    validation = validate_sparse_loo_ledger(
        ledger_manifest,
        snapshot_manifest_file=manifest_path,
        registry_file=registry,
        roster_file=roster,
    )
    assert validation["ready_sample_count"] == 25
    assert validation["excluded_sample_count"] == 1
