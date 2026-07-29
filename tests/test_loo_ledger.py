import csv
import io
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import seal_manifest
from open_r1.validator.loo_ledger import (
    EXPECTED_LEDGER_ROW_COUNT,
    LEDGER_ROW_SCHEMA_VERSION,
    LooLedgerError,
    LooLedgerIntegrityError,
    build_loo_indicator_ledger,
    compute_request_id,
    decision_identity_timestamp,
    information_as_of_date,
    validate_loo_indicator_ledger,
)
from open_r1.validator.loo_ledger import _sample_observations


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _subtract_months(value: date, months: int) -> date:
    absolute = value.year * 12 + value.month - 1 - months
    year, month_zero = divmod(absolute, 12)
    return date(year, month_zero + 1, 1)


def _month_starts(first: date, last: date) -> list[date]:
    values: list[date] = []
    current = date(first.year, first.month, 1)
    end = date(last.year, last.month, 1)
    while current <= end:
        values.append(current)
        absolute = current.year * 12 + current.month
        year, month_zero = divmod(absolute, 12)
        current = date(year, month_zero + 1, 1)
    return values


class LedgerFixture:
    def __init__(self, root: Path):
        self.root = root
        self.roster_path = root / "roster.json"
        self.population_path = root / "population.json"
        self.registry_path = root / "registry.json"
        self.snapshot_path = root / "snapshots" / "snapshot_manifest.json"
        self.output_dir = root / "ledger"
        self.source_key = "alfred:TESTD1"
        self.series_id = "TESTD1"

        roster = json.loads(
            (
                REPO_ROOT / "configs/main/leave_one_out_roster.json"
            ).read_text(encoding="utf-8")
        )
        population = json.loads(
            (
                REPO_ROOT
                / "configs/main/loo_population_pilot_eval_13.json"
            ).read_text(encoding="utf-8")
        )
        _write_json(self.roster_path, roster)
        _write_json(self.population_path, population)
        self.indicators = roster["indicators"]
        self.meeting_dates = [
            date.fromisoformat(value) for value in population["meeting_dates"]
        ]
        self.population_id = population["population_id"]

        registry = {
            "schema_version": "loo-indicator-source-registry-v1",
            "policy": {
                "vintage_lag_calendar_days": 1,
                "same_meeting_day_data": "excluded",
                "unknown_availability": "fail_closed",
                "runtime_fallback": "forbidden",
                "local_legacy_values": "mapping_only",
                "synthetic_text_values": "mapping_only",
                "access_interface": "alfred-graph-csv-v1",
                "raw_response_required": True,
                "raw_response_sha256_required": True,
            },
            "sources": {
                self.source_key: {
                    "source_id": self.source_key,
                    "artifact_id": "alfred__TESTD1",
                    "provider": "alfred_graph_csv",
                    "access_interface": "alfred-graph-csv-v1",
                    "series_id": self.series_id,
                    "title": "Synthetic D-1 validation series",
                    "date_column": "observation_date",
                    "value_column_template": (
                        "{series_id}_{vintage_date_yyyymmdd}"
                    ),
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
                    "license": "Public domain test fixture",
                    "redistribution_allowed": True,
                    "enabled": True,
                }
            },
            "indicators": {
                indicator: {"source_keys": [self.source_key]}
                for indicator in self.indicators
            },
            "legacy_crosswalk": {"files": []},
        }
        _write_json(self.registry_path, registry)
        self._write_snapshot_manifest()

    def _write_snapshot_manifest(self) -> None:
        requests: list[dict] = []
        as_of_dates = [
            information_as_of_date(meeting) for meeting in self.meeting_dates
        ]
        for chunk_index in range(0, len(as_of_dates), 12):
            chunk = as_of_dates[chunk_index : chunk_index + 12]
            vintage_dates = [value.isoformat() for value in chunk]
            cosd = [
                _subtract_months(value, 24).isoformat() for value in chunk
            ]
            coed = list(vintage_dates)
            request_stub = {
                "source_key": self.source_key,
                "series_id": self.series_id,
                "vintage_dates": vintage_dates,
                "cosd": cosd,
                "coed": coed,
                "canonical_url": (
                    "https://api.stlouisfed.org/graph/fredgraph.csv?"
                    f"series_id={self.series_id}&batch={chunk_index // 12}"
                ),
            }
            request_id = compute_request_id(request_stub)
            relative_path = (
                Path("raw")
                / "alfred"
                / "test_source"
                / f"{request_id}.csv"
            )
            raw_path = self.snapshot_path.parent / relative_path
            raw_path.parent.mkdir(parents=True, exist_ok=True)

            buffer = io.StringIO()
            columns = [
                f"{self.series_id}_{value.strftime('%Y%m%d')}"
                for value in chunk
            ]
            writer = csv.DictWriter(
                buffer,
                fieldnames=["observation_date", *columns],
                lineterminator="\n",
            )
            writer.writeheader()
            first = min(_subtract_months(value, 24) for value in chunk)
            last = max(chunk)
            for row_index, observation_date in enumerate(
                _month_starts(first, _subtract_months(last, -2))
            ):
                row = {"observation_date": observation_date.isoformat()}
                for column_index, column in enumerate(columns):
                    row[column] = str(100 + row_index + column_index / 10)
                writer.writerow(row)
            raw_path.write_text(buffer.getvalue(), encoding="utf-8")

            requests.append(
                {
                    **request_stub,
                    "request_id": request_id,
                    "raw_relative_path": relative_path.as_posix(),
                    "raw_sha256": sha256_file(raw_path),
                    "byte_count": raw_path.stat().st_size,
                    "content_type": "text/csv; charset=utf-8",
                    "retrieved_at_utc": "2026-07-28T12:00:00Z",
                    "status_code": 200,
                }
            )

        snapshot_payload = {
            "schema_version": "loo-source-snapshot-manifest-v1",
            "status": "complete",
            "registry": {
                "path": str(self.registry_path),
                "sha256": sha256_file(self.registry_path),
            },
            "meetings": [
                {
                    "meeting_date": meeting.isoformat(),
                    "information_as_of_date": information_as_of_date(
                        meeting
                    ).isoformat(),
                    "population_ids": [self.population_id],
                }
                for meeting in self.meeting_dates
            ],
            "requests": requests,
        }
        _write_json(self.snapshot_path, seal_manifest(snapshot_payload))

    def build(self) -> dict:
        return build_loo_indicator_ledger(
            registry_file=self.registry_path,
            snapshot_manifest_file=self.snapshot_path,
            population_file=self.population_path,
            roster_file=self.roster_path,
            output_dir=self.output_dir,
        )

    def validate(self) -> dict:
        return validate_loo_indicator_ledger(
            ledger_manifest_file=self.output_dir / "ledger_manifest.json",
            registry_file=self.registry_path,
            snapshot_manifest_file=self.snapshot_path,
            population_file=self.population_path,
            roster_file=self.roster_path,
        )


class TestD1LedgerConstruction(unittest.TestCase):
    def test_builds_exact_338_row_replayable_ledger(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LedgerFixture(Path(temporary_directory))
            manifest = fixture.build()
            rows = _read_jsonl(fixture.output_dir / "indicator_inputs.jsonl")
            evidence = _read_jsonl(
                fixture.output_dir / "source_evidence.jsonl"
            )
            exclusions = _read_jsonl(
                fixture.output_dir / "excluded_records.jsonl"
            )
            coverage = json.loads(
                (fixture.output_dir / "coverage.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(len(rows), EXPECTED_LEDGER_ROW_COUNT)
            self.assertEqual(len(evidence), EXPECTED_LEDGER_ROW_COUNT)
            self.assertEqual(
                manifest["outputs"]["indicator_inputs"]["row_count"],
                EXPECTED_LEDGER_ROW_COUNT,
            )
            self.assertEqual(
                manifest["inputs"]["roster"]["sha256"],
                sha256_file(fixture.roster_path),
            )
            self.assertEqual(coverage["row_count"], EXPECTED_LEDGER_ROW_COUNT)
            self.assertTrue(exclusions)
            self.assertTrue(
                any(
                    row["reason"]
                    == "observation_after_information_as_of_date"
                    for row in exclusions
                )
            )
            self.assertTrue(
                any(
                    row["sample_id"].startswith("2023-02-01::")
                    and row.get("observation_date") == "2023-02-01"
                    and row["reason"]
                    == "observation_after_information_as_of_date"
                    for row in exclusions
                )
            )

            first = rows[0]
            self.assertEqual(
                first["schema_version"], LEDGER_ROW_SCHEMA_VERSION
            )
            self.assertEqual(first["meeting_id"], "2021-12-15")
            self.assertEqual(
                first["information_as_of_date"], "2021-12-14"
            )
            self.assertEqual(
                first["requested_vintage_date"], "2021-12-14"
            )
            self.assertEqual(
                first["availability_as_of_date"], "2021-12-14"
            )
            self.assertNotIn("release_timestamp", first)
            self.assertEqual(
                first["meeting_timestamp"], "2021-12-15T18:59:59Z"
            )
            observations = first["source_payload"]["series"][0][
                "observations"
            ]
            self.assertLessEqual(len(observations), 25)
            self.assertEqual(
                observations, sorted(observations, key=lambda row: row["date"])
            )
            self.assertTrue(
                all(
                    row["date"] <= first["information_as_of_date"]
                    for row in observations
                )
            )

            validation = fixture.validate()
            self.assertEqual(validation["status"], "valid")
            self.assertEqual(
                validation["row_count"], EXPECTED_LEDGER_ROW_COUNT
            )

    def test_rerun_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LedgerFixture(Path(temporary_directory))
            first = fixture.build()
            first_bytes = {
                name: (fixture.output_dir / name).read_bytes()
                for name in (
                    "indicator_inputs.jsonl",
                    "source_evidence.jsonl",
                    "excluded_records.jsonl",
                    "coverage.json",
                    "ledger_manifest.json",
                )
            }
            second = fixture.build()
            self.assertEqual(first, second)
            for name, content in first_bytes.items():
                self.assertEqual(
                    (fixture.output_dir / name).read_bytes(), content
                )

    def test_daily_sampling_keeps_one_value_per_month(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LedgerFixture(Path(temporary_directory))
            fixture.build()
            first = _read_jsonl(
                fixture.output_dir / "indicator_inputs.jsonl"
            )[0]
            observations = first["source_payload"]["series"][0][
                "observations"
            ]
            months = {row["date"][:7] for row in observations}
            self.assertEqual(len(months), len(observations))
            self.assertLessEqual(len(observations), 25)

    def test_decision_identity_timestamp_is_dst_aware(self):
        self.assertEqual(
            decision_identity_timestamp(date(2023, 11, 1)),
            "2023-11-01T17:59:59Z",
        )
        self.assertEqual(
            decision_identity_timestamp(date(2024, 11, 7)),
            "2024-11-07T18:59:59Z",
        )

    def test_restricted_sources_remain_valid_for_local_canonical_use(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LedgerFixture(Path(temporary_directory))
            registry = json.loads(
                fixture.registry_path.read_text(encoding="utf-8")
            )
            registry["sources"][fixture.source_key][
                "redistribution_allowed"
            ] = False
            registry["sources"][fixture.source_key][
                "license"
            ] = "restricted local research use"
            _write_json(fixture.registry_path, registry)
            fixture._write_snapshot_manifest()

            manifest = fixture.build()
            self.assertTrue(
                manifest["license_summary"]["local_research_use_only"]
            )
            self.assertFalse(
                manifest["license_summary"][
                    "redistribution_allowed_for_all_sources"
                ]
            )
            self.assertFalse(
                manifest["license_summary"]["sources"][0][
                    "redistribution_allowed"
                ]
            )


class TestFrequencyAwareSampling(unittest.TestCase):
    def _source(self, frequency: str) -> dict:
        return {
            "source_key": f"source-{frequency}",
            "frequency": frequency,
            "lookback_policy": {"mode": "observations", "value": 100},
            "selection": {
                "method": "period_end_plus_latest",
                "period": "month",
                "max_points": 100,
            },
            "_information_as_of_date": date(2025, 1, 28),
        }

    def test_low_frequency_caps_are_frozen(self):
        observations = [
            {"date": value.isoformat(), "value": str(index)}
            for index, value in enumerate(
                _month_starts(date(2020, 1, 1), date(2025, 1, 1))
            )
        ]
        expected = {
            "monthly": 24,
            "quarterly": 8,
            "semiannual": 6,
            "annual": 5,
        }
        for frequency, count in expected.items():
            with self.subTest(frequency=frequency):
                sampled, _ = _sample_observations(
                    observations,
                    source=self._source(frequency),
                    sample_id=f"meeting::{frequency}",
                )
                self.assertEqual(len(sampled), count)

    def test_daily_sampling_keeps_month_end_plus_latest_for_24_months(self):
        observations: list[dict[str, str]] = []
        for index, month_start in enumerate(
            _month_starts(date(2022, 1, 1), date(2025, 1, 1))
        ):
            observations.extend(
                [
                    {
                        "date": month_start.isoformat(),
                        "value": str(index),
                    },
                    {
                        "date": month_start.replace(day=15).isoformat(),
                        "value": str(index + 0.5),
                    },
                ]
            )
        sampled, _ = _sample_observations(
            observations,
            source=self._source("daily"),
            sample_id="meeting::daily",
        )
        self.assertLessEqual(len(sampled), 25)
        self.assertEqual(len({row["date"][:7] for row in sampled}), len(sampled))
        self.assertTrue(all(row["date"].endswith("-15") for row in sampled))

    def test_annual_tail_is_not_pretrimmed_by_generic_calendar_lookback(self):
        source = self._source("annual")
        source["lookback_policy"] = {
            "mode": "calendar",
            "unit": "months",
            "value": 24,
        }
        observations = [
            {"date": f"{year}-01-01", "value": str(year)}
            for year in range(2017, 2022)
        ]
        sampled, _ = _sample_observations(
            observations,
            source=source,
            sample_id="2022-01-26::Government-Purchases",
        )
        self.assertEqual(
            [row["date"] for row in sampled],
            [
                "2017-01-01",
                "2018-01-01",
                "2019-01-01",
                "2020-01-01",
                "2021-01-01",
            ],
        )


class TestD1LedgerRejections(unittest.TestCase):
    def test_rejects_request_whose_observation_end_is_meeting_day(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LedgerFixture(Path(temporary_directory))
            manifest = json.loads(
                fixture.snapshot_path.read_text(encoding="utf-8")
            )
            manifest.pop("integrity")
            request = manifest["requests"][0]
            request["coed"][0] = fixture.meeting_dates[0].isoformat()
            request["request_id"] = compute_request_id(request)
            _write_json(fixture.snapshot_path, seal_manifest(manifest))

            with self.assertRaisesRegex(
                LooLedgerError, "must use D-1"
            ):
                fixture.build()

    def test_rejects_tampered_raw_csv(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LedgerFixture(Path(temporary_directory))
            manifest = json.loads(
                fixture.snapshot_path.read_text(encoding="utf-8")
            )
            relative = manifest["requests"][0]["raw_relative_path"]
            raw_path = fixture.snapshot_path.parent / relative
            raw_path.write_text(
                raw_path.read_text(encoding="utf-8") + "tamper\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                LooLedgerIntegrityError, "byte count changed"
            ):
                fixture.build()

    def test_rejects_unsealed_snapshot_edit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LedgerFixture(Path(temporary_directory))
            manifest = json.loads(
                fixture.snapshot_path.read_text(encoding="utf-8")
            )
            manifest["meetings"][0][
                "information_as_of_date"
            ] = fixture.meeting_dates[0].isoformat()
            _write_json(fixture.snapshot_path, manifest)

            with self.assertRaisesRegex(
                LooLedgerIntegrityError, "digest mismatch"
            ):
                fixture.build()

    def test_replay_validator_rejects_output_tampering(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LedgerFixture(Path(temporary_directory))
            fixture.build()
            ledger_path = fixture.output_dir / "indicator_inputs.jsonl"
            ledger_path.write_text(
                ledger_path.read_text(encoding="utf-8").replace(
                    "2021-12-14", "2021-12-15", 1
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                LooLedgerIntegrityError, "Output hash mismatch"
            ):
                fixture.validate()

    def test_replay_validator_rejects_roster_reordering(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LedgerFixture(Path(temporary_directory))
            fixture.build()
            roster = json.loads(
                fixture.roster_path.read_text(encoding="utf-8")
            )
            roster["indicators"][0], roster["indicators"][1] = (
                roster["indicators"][1],
                roster["indicators"][0],
            )
            _write_json(fixture.roster_path, roster)

            with self.assertRaisesRegex(
                LooLedgerIntegrityError, "Ledger input changed: roster"
            ):
                fixture.validate()

    def test_rejects_current_vintage_even_for_old_observations(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LedgerFixture(Path(temporary_directory))
            manifest = json.loads(
                fixture.snapshot_path.read_text(encoding="utf-8")
            )
            manifest.pop("integrity")
            request = manifest["requests"][0]
            request["vintage_dates"][0] = fixture.meeting_dates[0].isoformat()
            request["coed"][0] = fixture.meeting_dates[0].isoformat()
            request["request_id"] = compute_request_id(request)
            _write_json(fixture.snapshot_path, seal_manifest(manifest))

            with self.assertRaisesRegex(
                LooLedgerError, "Missing D-1 snapshot request"
            ):
                fixture.build()


if __name__ == "__main__":
    unittest.main()
