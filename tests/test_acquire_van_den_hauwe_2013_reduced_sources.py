from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jobs.eval import acquire_van_den_hauwe_2013_reduced_sources as acquire


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _monthly_rows() -> list[dict[str, Any]]:
    months = acquire.month_range("1990-01", "2008-06")
    directions = ["cut"] * 40 + ["hold"] * 86 + ["hike"] * 31
    rows: list[dict[str, Any]] = []
    for index, month in enumerate(months):
        cutoff = acquire.previous_month_end(month)
        decision_month = index < len(directions)
        predictors = {
            "6TFF": {
                "value": 0.1,
                "availability_date": cutoff,
                "vintage_date": None,
                "real_time_vintage": False,
                "source_policy": "historical_market_nonrevised",
            },
            "IP": {
                "value": 1.0,
                "availability_date": cutoff,
                "vintage_date": cutoff,
                "real_time_vintage": True,
                "source_policy": "alfred_end_of_prior_month_vintage",
            },
            "INF": {
                "value": 2.0,
                "availability_date": cutoff,
                "vintage_date": cutoff,
                "real_time_vintage": True,
                "source_policy": "alfred_end_of_prior_month_vintage",
            },
        }
        row: dict[str, Any] = {
            "schema_version": acquire.MONTHLY_SCHEMA,
            "month": month,
            "information_cutoff": cutoff,
            "decision_month": decision_month,
            "scheduled_meeting_month": index < 148,
            "unscheduled_only_decision_month": 148 <= index < 157,
            "direction": directions[index] if decision_month else None,
            "predictors": predictors,
        }
        row["row_sha256"] = acquire.sha256_bytes(
            acquire.canonical_json(row).encode("utf-8")
        )
        rows.append(row)
    return rows


def _seal_fixture(
    root: Path, config_path: Path, roster_path: Path
) -> list[dict[str, Any]]:
    root.mkdir()
    rows = _monthly_rows()
    monthly = root / "monthly_source.jsonl"
    monthly.write_text(
        "".join(acquire.canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": acquire.SCHEMA,
        "config": str(config_path.relative_to(acquire.ROOT)),
        "config_sha256": acquire.sha256_file(config_path),
        "roster": str(roster_path.relative_to(acquire.ROOT)),
        "roster_sha256": acquire.sha256_file(roster_path),
        "monthly_source_sha256": acquire.sha256_file(monthly),
        "files": [
            {
                "path": "monthly_source.jsonl",
                "bytes": monthly.stat().st_size,
                "sha256": acquire.sha256_file(monthly),
            }
        ],
    }
    _write_json(root / "manifest.json", manifest)
    return rows


def test_resume_rejects_config_and_roster_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(acquire, "ROOT", tmp_path)
    config_path = tmp_path / "config.json"
    roster_path = tmp_path / "roster.jsonl"
    _write_json(config_path, {"version": 1})
    roster_path.write_text('{"meeting": "a"}\n', encoding="utf-8")
    source_root = tmp_path / "source"
    _seal_fixture(source_root, config_path, roster_path)

    acquire.acquire(
        config_path=config_path,
        roster_path=roster_path,
        output_root=source_root,
        resume=True,
    )
    _write_json(config_path, {"version": 2})
    with pytest.raises(acquire.SourceAcquisitionError, match="config binding drift"):
        acquire.acquire(
            config_path=config_path,
            roster_path=roster_path,
            output_root=source_root,
            resume=True,
        )

    _write_json(config_path, {"version": 1})
    roster_path.write_text('{"meeting": "b"}\n', encoding="utf-8")
    with pytest.raises(acquire.SourceAcquisitionError, match="roster binding drift"):
        acquire.acquire(
            config_path=config_path,
            roster_path=roster_path,
            output_root=source_root,
            resume=True,
        )


def test_monthly_population_and_timing_contracts() -> None:
    rows = _monthly_rows()
    acquire.validate_monthly_rows(rows)

    rows[0]["predictors"]["IP"]["availability_date"] = "1990-01-01"
    rows[0]["row_sha256"] = acquire.sha256_bytes(
        acquire.canonical_json(
            {key: value for key, value in rows[0].items() if key != "row_sha256"}
        ).encode("utf-8")
    )
    with pytest.raises(acquire.SourceAcquisitionError, match="future IP availability"):
        acquire.validate_monthly_rows(rows)


def test_month_and_transform_helpers() -> None:
    assert acquire.month_range("1999-12", "2000-02") == [
        "1999-12",
        "2000-01",
        "2000-02",
    ]
    assert acquire.previous_month_end("2000-03") == "2000-02-29"
    assert acquire.subtract_years("2000-02-29", 1) == "1999-02-28"
    assert acquire.latest_for_month(
        [("1999-12-01", 1.0), ("2000-01-01", 2.0), ("2000-02-01", 3.0)],
        "2000-01",
    ) == ("2000-01-01", 2.0)


def test_fred_and_alfred_parsers_enforce_exact_vintage_columns(
    tmp_path: Path,
) -> None:
    fred = tmp_path / "fred.csv"
    fred.write_text(
        "observation_date,TB6MS\n2000-01-01,5.0\n2000-02-01,.\n",
        encoding="utf-8",
    )
    assert acquire.parse_fred_csv(fred, "TB6MS") == [("2000-01-01", 5.0)]

    alfred = tmp_path / "alfred.csv"
    alfred.write_text(
        "observation_date,INDPRO_20000229\n"
        "1999-12-01,100.0\n2000-01-01,101.0\n",
        encoding="utf-8",
    )
    assert acquire.parse_alfred_chunk(
        alfred, series_id="INDPRO", vintages=["2000-02-29"]
    ) == {"2000-02-29": [("1999-12-01", 100.0), ("2000-01-01", 101.0)]}

    alfred.write_text(
        "observation_date,INDPRO_wrong\n"
        "1999-12-01,100.0\n2000-01-01,101.0\n",
        encoding="utf-8",
    )
    with pytest.raises(acquire.SourceAcquisitionError, match="unexpected ALFRED"):
        acquire.parse_alfred_chunk(
            alfred, series_id="INDPRO", vintages=["2000-02-29"]
        )
