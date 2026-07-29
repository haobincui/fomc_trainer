import hashlib
import json
import tempfile
import threading
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from jobs.main.fetch_loo_source_snapshots import (
    HttpResult,
    SnapshotFetchError,
    SnapshotRequest,
    SourceSeries,
    _download_with_retry,
    build_snapshot_requests,
    fetch_source_snapshots,
    load_source_registry,
    validate_alfred_csv,
)
from open_r1.validator.loo_generation_spec import (
    validate_manifest_integrity,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _request(
    *,
    vintages: tuple[str, ...] = ("2024-01-30", "2024-03-19"),
) -> SnapshotRequest:
    starts = tuple("2019-01-01" for _ in vintages)
    return SnapshotRequest(
        request_id="a" * 64,
        source_key="alfred__TEST",
        series_id="TEST",
        vintage_dates=vintages,
        cosd=starts,
        coed=vintages,
        canonical_url="https://example.invalid/snapshot.csv",
        raw_relative_path="raw/alfred/alfred__TEST/a/response.csv",
    )


def _csv_for_request(request: SnapshotRequest) -> bytes:
    header = ["observation_date"] + [
        f"{request.series_id}_{value.replace('-', '')}"
        for value in request.vintage_dates
    ]
    observation_date = request.vintage_dates[0]
    values = [str(index + 1) for index in range(len(request.vintage_dates))]
    return (
        ",".join(header) + "\n" + ",".join([observation_date, *values]) + "\n"
    ).encode("utf-8")


def _registry_payload() -> dict:
    return {
        "schema_version": "loo-indicator-source-registry-v1",
        "registry_id": "test-registry",
        "default_lookback_years": 5,
        "series_defaults": {
            "access_interface": "alfred-graph-csv-v1",
            "lookback": {"unit": "months", "value": 24},
            "enabled": True,
        },
        "sources": {
            "alfred__TEST": {
                "series_id": "TEST",
            },
            "alfred__DISABLED": {
                "series_id": "DO_NOT_FETCH",
                "enabled": False,
            },
        },
        "indicators": {
            "Example": {
                "source_keys": [
                    "alfred__DISABLED",
                    "alfred__TEST",
                ]
            }
        },
    }


class TestSourceRegistry(unittest.TestCase):
    def test_deep_merges_defaults_and_skips_disabled_sources(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry_path = Path(temporary_directory) / "registry.json"
            _write_json(registry_path, _registry_payload())

            _, sources = load_source_registry(registry_path)

        self.assertEqual(
            sources,
            [
                SourceSeries(
                    source_key="alfred__TEST",
                    series_id="TEST",
                    lookback_years=5,
                )
            ],
        )

    def test_rejects_indicator_without_enabled_source(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            registry_path = Path(temporary_directory) / "registry.json"
            payload = _registry_payload()
            payload["sources"]["alfred__TEST"]["enabled"] = False
            _write_json(registry_path, payload)

            with self.assertRaisesRegex(ValueError, "no enabled source"):
                load_source_registry(registry_path)


class TestRequestConstruction(unittest.TestCase):
    def test_canonical_26_vintages_are_chunked_12_12_2(self):
        first_vintage = date(2021, 1, 1)
        meetings = [
            {
                "information_as_of_date": (
                    first_vintage + timedelta(days=index * 30)
                ).isoformat()
            }
            for index in range(26)
        ]

        requests = build_snapshot_requests(
            [SourceSeries("alfred__TEST", "TEST", 5)],
            meetings,
        )

        self.assertEqual(
            [len(request.vintage_dates) for request in requests],
            [12, 12, 2],
        )
        for request in requests:
            query = parse_qs(urlparse(request.canonical_url).query)
            vector_lengths = {
                key: len(query[key][0].split(","))
                for key in ("id", "cosd", "coed", "vintage_date")
            }
            self.assertEqual(
                set(vector_lengths.values()),
                {len(request.vintage_dates)},
            )
            descriptor = {
                "source_key": request.source_key,
                "series_id": request.series_id,
                "vintage_dates": list(request.vintage_dates),
                "cosd": list(request.cosd),
                "coed": list(request.coed),
                "canonical_url": request.canonical_url,
            }
            canonical = json.dumps(
                descriptor,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            self.assertEqual(
                request.request_id,
                hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            )


class TestCsvValidation(unittest.TestCase):
    def test_accepts_exact_headers_dates_and_finite_decimals(self):
        request = _request()
        body = (
            "observation_date,TEST_20240130,TEST_20240319\n"
            "2023-01-01,1.25,\n"
            "2024-01-01,2.50,3.75\n"
        ).encode()

        validation = validate_alfred_csv(body, request)

        self.assertEqual(validation.row_count, 2)
        self.assertEqual(validation.column_nonempty_counts, (2, 1))
        self.assertEqual(validation.first_observation_date, "2023-01-01")
        self.assertEqual(validation.last_observation_date, "2024-01-01")

    def test_rejects_silent_current_vintage_fallback_header(self):
        request = _request()
        body = ("observation_date,TEST_20260728\n2024-01-01,1.0\n").encode()

        with self.assertRaisesRegex(SnapshotFetchError, "header mismatch"):
            validate_alfred_csv(body, request)

    def test_rejects_non_decimal_and_out_of_window_values(self):
        request = _request()
        non_decimal = (
            "observation_date,TEST_20240130,TEST_20240319\n2024-01-01,not-a-number,1\n"
        ).encode()
        outside_window = (
            "observation_date,TEST_20240130,TEST_20240319\n2024-02-01,1,2\n"
        ).encode()

        with self.assertRaisesRegex(SnapshotFetchError, "non-decimal"):
            validate_alfred_csv(non_decimal, request)
        with self.assertRaisesRegex(SnapshotFetchError, "after coed"):
            validate_alfred_csv(outside_window, request)

    def test_rejects_unequal_request_vectors(self):
        valid = _request()
        request = SnapshotRequest(
            **{
                **valid.__dict__,
                "cosd": ("2019-01-01",),
            }
        )

        with self.assertRaisesRegex(SnapshotFetchError, "equal lengths"):
            validate_alfred_csv(_csv_for_request(valid), request)


class _NoOpLimiter:
    def acquire(self) -> None:
        return None


class TestRetry(unittest.TestCase):
    def test_retries_retryable_status_and_normalises_headers(self):
        request = _request(vintages=("2024-01-30",))
        responses = [
            HttpResult(503, {"Retry-After": "0"}, b"busy"),
            HttpResult(
                200,
                {"Content-Type": "Application/CSV; charset=utf-8"},
                _csv_for_request(request),
            ),
        ]
        sleeps: list[float] = []

        response, retrieved_at = _download_with_retry(
            request,
            http_get=lambda _url, _timeout: responses.pop(0),
            limiter=_NoOpLimiter(),
            timeout_seconds=1,
            max_retries=1,
            sleep=sleeps.append,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["content-type"], "Application/CSV; charset=utf-8"
        )
        self.assertEqual(sleeps, [0.0])
        self.assertTrue(retrieved_at.endswith("Z"))


class TestSnapshotAcquisition(unittest.TestCase):
    def test_fetches_seals_resumes_and_detects_raw_tampering(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry_path = root / "registry.json"
            first_population = root / "pilot.json"
            second_population = root / "formal.json"
            output_dir = root / "snapshots"
            _write_json(registry_path, _registry_payload())
            start = date(2021, 1, 2)
            dates = [
                (start + timedelta(days=index * 30)).isoformat() for index in range(26)
            ]
            _write_json(
                first_population,
                {
                    "population_id": "pilot",
                    "meeting_dates": dates[:13],
                },
            )
            _write_json(
                second_population,
                {
                    "population_id": "formal",
                    "meeting_dates": dates[13:],
                },
            )
            calls: list[str] = []
            lock = threading.Lock()

            def fake_http_get(url: str, _timeout: float) -> HttpResult:
                query = parse_qs(urlparse(url).query)
                series_ids = query["id"][0].split(",")
                vintages = tuple(query["vintage_date"][0].split(","))
                starts = tuple(query["cosd"][0].split(","))
                ends = tuple(query["coed"][0].split(","))
                with lock:
                    calls.append(url)
                request = SnapshotRequest(
                    request_id="fake",
                    source_key="alfred__TEST",
                    series_id=series_ids[0],
                    vintage_dates=vintages,
                    cosd=starts,
                    coed=ends,
                    canonical_url=url,
                    raw_relative_path="unused",
                )
                return HttpResult(
                    status_code=200,
                    headers={"Content-Type": "application/csv; charset=utf-8"},
                    body=_csv_for_request(request),
                )

            with patch(
                "jobs.main.fetch_loo_source_snapshots.RateLimiter.acquire",
                return_value=None,
            ):
                manifest = fetch_source_snapshots(
                    registry_path=registry_path,
                    population_paths=[first_population, second_population],
                    output_dir=output_dir,
                    http_get=fake_http_get,
                )

            self.assertEqual(len(calls), 3)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["vintage_count"], 26)
            self.assertEqual(manifest["request_count"], 3)
            self.assertEqual(
                [len(record["vintage_dates"]) for record in manifest["requests"]],
                [12, 12, 2],
            )
            validate_manifest_integrity(manifest)
            self.assertTrue((output_dir / "snapshot_manifest.json").is_file())
            for record in manifest["requests"]:
                raw_path = output_dir / record["raw_relative_path"]
                self.assertTrue(raw_path.is_file())
                self.assertEqual(
                    hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                    record["raw_sha256"],
                )

            resumed = fetch_source_snapshots(
                registry_path=registry_path,
                population_paths=[first_population, second_population],
                output_dir=output_dir,
                resume=True,
                http_get=lambda _url, _timeout: self.fail(
                    "resume must not access the network"
                ),
            )
            self.assertEqual(resumed, manifest)

            first_raw = output_dir / manifest["requests"][0]["raw_relative_path"]
            first_raw.write_bytes(first_raw.read_bytes() + b"tampered")
            with self.assertRaisesRegex(
                SnapshotFetchError,
                "hash/size mismatch",
            ):
                fetch_source_snapshots(
                    registry_path=registry_path,
                    population_paths=[first_population, second_population],
                    output_dir=output_dir,
                    resume=True,
                    http_get=lambda _url, _timeout: self.fail(
                        "tampered resume must not access the network"
                    ),
                )

    def test_one_population_infers_thirteen_vintages(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            registry_path = root / "registry.json"
            population_path = root / "pilot.json"
            _write_json(registry_path, _registry_payload())
            start = date(2023, 1, 2)
            meeting_dates = [
                (start + timedelta(days=index * 30)).isoformat() for index in range(13)
            ]
            _write_json(
                population_path,
                {
                    "population_id": "pilot",
                    "meeting_dates": meeting_dates,
                },
            )

            def fake_http_get(url: str, _timeout: float) -> HttpResult:
                query = parse_qs(urlparse(url).query)
                vintages = tuple(query["vintage_date"][0].split(","))
                request = SnapshotRequest(
                    request_id="fake",
                    source_key="alfred__TEST",
                    series_id=query["id"][0].split(",")[0],
                    vintage_dates=vintages,
                    cosd=tuple(query["cosd"][0].split(",")),
                    coed=tuple(query["coed"][0].split(",")),
                    canonical_url=url,
                    raw_relative_path="unused",
                )
                return HttpResult(
                    200,
                    {"content-type": "application/csv"},
                    _csv_for_request(request),
                )

            with patch(
                "jobs.main.fetch_loo_source_snapshots.RateLimiter.acquire",
                return_value=None,
            ):
                manifest = fetch_source_snapshots(
                    registry_path=registry_path,
                    population_paths=[population_path],
                    output_dir=root / "snapshots",
                    http_get=fake_http_get,
                )

            self.assertEqual(manifest["vintage_count"], 13)
            self.assertEqual(
                [len(record["vintage_dates"]) for record in manifest["requests"]],
                [12, 1],
            )


if __name__ == "__main__":
    unittest.main()
