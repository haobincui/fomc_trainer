from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from jobs.eval import download_wrds_fed_funds_futures as downloader
from jobs.eval import evaluate_decision_comparison_baselines as comparison


def _metadata_frame() -> pd.DataFrame:
    row = {column: None for column in downloader.METADATA_COLUMNS}
    row.update(
        {
            "futcode": 101,
            "contrcode": downloader.PRODUCT_CONTRCODE,
            "clscode": downloader.PRODUCT_CLSCODE,
            "dsmnem": "CFF1230",
            "contrname": downloader.PRODUCT_NAME,
            "unitcode": 1,
            "ticksizeunitcode": 1,
            "startdate": "1987-01-01",
            "lasttrddate": "2030-12-31",
            # A contract can settle on the first day of the following month.
            # Its delivery month is bound by the mnemonic and last trade date.
            "sttlmntdate": "2031-01-01",
            "exchtickersymb": downloader.PRODUCT_TICKER,
        }
    )
    return pd.DataFrame([row], columns=downloader.METADATA_COLUMNS)


def _price_frame(day: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "futcode": 101,
                "date_": day,
                "open_": 95.0,
                "high": 95.2,
                "low": 94.9,
                "volume": 10,
                "settlement": 95.1,
                "openinterest": 20,
            }
        ],
        columns=downloader.PRICE_COLUMNS,
    )


def _profile_frame() -> pd.DataFrame:
    row = {field: 0 for field in downloader.PROFILE_FIELDS}
    row.update({"unique_keys": 1, "source_rows": 5, "duplicated_source_keys": 1})
    return pd.DataFrame([row], columns=downloader.PROFILE_FIELDS)


class FakeConnection:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def raw_sql(self, sql: str) -> pd.DataFrame:
        self.calls.append(sql)
        if "source_multiplicity" in sql:
            return _profile_frame()
        if "select i.futcode" in sql:
            return _metadata_frame()
        match = __import__("re").search(
            r"between date '(\d{4}-\d{2}-\d{2})' and date '(\d{4}-\d{2}-\d{2})'",
            sql,
        )
        assert match is not None
        return _price_frame(match.group(1))


def _prepare_and_acquire(tmp_path, start: date, end: date):
    root = tmp_path / "wrds"
    spec = downloader.build_run_spec(start, end, 1)
    downloader.initialize_run(root, spec, resume=False)
    connection = FakeConnection()
    for chunk in spec["chunking"]["chunks"]:
        downloader.acquire_chunk(connection, root, chunk)
    return root, spec, connection


def test_full_range_defaults_and_calendar_year_chunks() -> None:
    args = downloader.parse_args(["--output-root", "unused"])
    assert args.start_date == date(1988, 1, 1)
    assert args.end_date == date(2026, 7, 27)
    assert args.phase == "all"
    chunks = downloader.iter_date_chunks(args.start_date, args.end_date)
    assert len(chunks) == 39
    assert chunks[0] == {
        "chunk_id": "1988-01-01_1988-12-31",
        "start": "1988-01-01",
        "end": "1988-12-31",
    }
    assert chunks[-1] == {
        "chunk_id": "2026-01-01_2026-07-27",
        "start": "2026-01-01",
        "end": "2026-07-27",
    }


def test_explicit_legacy_dates_remain_supported() -> None:
    args = downloader.parse_args(
        [
            "--output-root",
            "legacy",
            "--start-date",
            "2004-11-29",
            "--end-date",
            "2015-04-30",
        ]
    )
    assert args.start_date == date(2004, 11, 29)
    assert args.end_date == date(2015, 4, 30)
    assert args.start_date_explicit is True
    assert args.end_date_explicit is True
    with pytest.raises(ValueError, match="chunk-years"):
        downloader.expected_spec_from_args(
            downloader.parse_args(["--output-root", "invalid", "--chunk-years", "0"])
        )


def test_chunk_is_validated_hash_bound_and_safe_to_skip_on_resume(tmp_path) -> None:
    root, spec, connection = _prepare_and_acquire(
        tmp_path, date(2020, 1, 1), date(2020, 12, 31)
    )
    chunk = spec["chunking"]["chunks"][0]
    manifest = downloader.validate_chunk_artifact(root, chunk)
    assert manifest["quality_report"]["price_rows"] == 1
    assert manifest["redistribution"]["raw_data_redistribution_permitted"] is False
    assert len(connection.calls) == 3

    pending = downloader.pending_chunks(root, spec)
    assert pending == []
    # No pgpass read or WRDS connection occurs when every chunk has already
    # passed its manifest and file-hash checks.
    result = downloader.acquire(root, spec, tmp_path / "does-not-exist.pgpass")
    assert result == {"status": "acquired", "downloaded_chunks": 0, "skipped_chunks": 1}


def test_empty_year_chunk_is_a_valid_audited_result(tmp_path) -> None:
    class EmptyConnection:
        def raw_sql(self, sql: str) -> pd.DataFrame:
            if "source_multiplicity" in sql:
                return pd.DataFrame(
                    [{field: 0 for field in downloader.PROFILE_FIELDS}],
                    columns=downloader.PROFILE_FIELDS,
                )
            if "select i.futcode" in sql:
                return pd.DataFrame(columns=downloader.METADATA_COLUMNS)
            return pd.DataFrame(columns=downloader.PRICE_COLUMNS)

    root = tmp_path / "wrds"
    spec = downloader.build_run_spec(date(1988, 1, 1), date(1988, 12, 31), 1)
    downloader.initialize_run(root, spec, resume=False)
    manifest = downloader.acquire_chunk(
        EmptyConnection(), root, spec["chunking"]["chunks"][0]
    )
    assert manifest["quality_report"]["price_rows"] == 0
    assert manifest["quality_report"]["price_min_date"] is None
    assert downloader.finalize(root, spec)["quality_report"]["price_rows"] == 0


def test_corrupt_completed_chunk_fails_closed_instead_of_redownloading(
    tmp_path,
) -> None:
    root, spec, _ = _prepare_and_acquire(tmp_path, date(2020, 1, 1), date(2020, 12, 31))
    chunk = spec["chunking"]["chunks"][0]
    price_path = (
        downloader.chunk_directory(root, chunk) / "contract_prices_daily.csv.gz"
    )
    price_path.write_bytes(price_path.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="size mismatch"):
        downloader.pending_chunks(root, spec)
    report = downloader.status_report(root)
    assert report["status"] == "invalid"
    assert report["invalid_chunks"] == [chunk["chunk_id"]]


def test_finalize_aggregates_chunks_and_seals_create_only(tmp_path) -> None:
    root, spec, _ = _prepare_and_acquire(tmp_path, date(2020, 1, 1), date(2021, 12, 31))
    manifest = downloader.finalize(root, spec)
    assert manifest["status"] == "complete"
    assert manifest["chunking"]["chunk_count"] == 2
    assert manifest["quality_report"]["metadata_rows"] == 1
    assert manifest["quality_report"]["price_rows"] == 2
    assert manifest["credentials"] == {
        "embedded": False,
        "pgpass_path_recorded": False,
        "username_recorded": False,
        "password_recorded": False,
    }
    assert manifest["redistribution"]["public_release_permitted"] is False
    assert set(manifest["files"]) == set(downloader.FINAL_FILE_NAMES)
    assert not (root / ".seal_staging").exists()
    before = downloader.sha256_file(root / "manifest.json")

    # Finalize is idempotent validation after sealing and never rewrites bytes.
    assert downloader.finalize(root, spec) == manifest

    loaded = comparison.load_futures_source(root)
    assert loaded["available"] is True
    assert loaded["contract_by_month"] == {"2030-12": "101"}
    assert loaded["restriction"] == "restricted_licensed_source_data"
    assert downloader.sha256_file(root / "manifest.json") == before
    assert downloader.status_report(root)["status"] == "complete"


def test_finalize_refuses_incomplete_acquisition(tmp_path) -> None:
    root = tmp_path / "wrds"
    spec = downloader.build_run_spec(date(2020, 1, 1), date(2021, 12, 31), 1)
    downloader.initialize_run(root, spec, resume=False)
    downloader.acquire_chunk(FakeConnection(), root, spec["chunking"]["chunks"][0])
    with pytest.raises(RuntimeError, match="1 acquisition chunks are incomplete"):
        downloader.finalize(root, spec)


def test_status_phase_never_reads_credentials_or_contacts_wrds(
    tmp_path, monkeypatch, capsys
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("status must be offline")

    monkeypatch.setattr(downloader, "read_wrds_credentials", forbidden)
    monkeypatch.setattr(downloader, "open_wrds_connection", forbidden)
    root = tmp_path / "missing"
    assert downloader.main(["--output-root", str(root), "--status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_initialized"


def test_pgpass_requires_exact_mode_0400(tmp_path) -> None:
    pgpass = tmp_path / ".pgpass"
    pgpass.write_text(
        "wrds-pgdata.wharton.upenn.edu:9737:wrds:alice:secret\n",
        encoding="utf-8",
    )
    pgpass.chmod(0o600)
    with pytest.raises(PermissionError, match="exact mode 0400"):
        downloader.read_wrds_credentials(pgpass)


def test_provider_exception_is_redacted(tmp_path, monkeypatch) -> None:
    root = tmp_path / "wrds"
    spec = downloader.build_run_spec(date(2020, 1, 1), date(2020, 12, 31), 1)
    downloader.initialize_run(root, spec, resume=False)
    pgpass = tmp_path / ".pgpass"
    pgpass.write_text(
        "wrds-pgdata.wharton.upenn.edu:9737:wrds:alice:hunter2\n",
        encoding="utf-8",
    )
    pgpass.chmod(0o400)

    def explode(**kwargs):
        raise RuntimeError(
            f"connection failed for {kwargs['wrds_username']}:{kwargs['wrds_password']}"
        )

    monkeypatch.setattr(downloader, "open_wrds_connection", explode)
    with pytest.raises(RuntimeError) as caught:
        downloader.acquire(root, spec, pgpass)
    message = str(caught.value)
    assert "hunter2" not in message
    assert "alice" not in message
    assert "redacted" in message


def test_resume_rejects_changed_date_contract(tmp_path) -> None:
    root = tmp_path / "wrds"
    original = downloader.build_run_spec(date(2020, 1, 1), date(2020, 12, 31), 1)
    changed = downloader.build_run_spec(date(2020, 1, 1), date(2021, 12, 31), 1)
    downloader.initialize_run(root, original, resume=False)
    with pytest.raises(RuntimeError, match="immutable run spec"):
        downloader.initialize_run(root, changed, resume=True)


def test_resume_without_repeating_nondefault_dates_uses_bound_run_spec(
    tmp_path, capsys
) -> None:
    root, _, _ = _prepare_and_acquire(tmp_path, date(2004, 11, 29), date(2004, 12, 31))
    assert (
        downloader.main(["--output-root", str(root), "--phase", "acquire", "--resume"])
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["downloaded_chunks"] == 0
    assert payload["chunks"] == 1
