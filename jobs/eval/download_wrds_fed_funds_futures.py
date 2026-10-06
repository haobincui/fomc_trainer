#!/usr/bin/env python3
"""Download and seal historical 30-Day Federal Funds futures from WRDS.

The source is the WRDS Datastream futures view. Acquisition is split into
calendar-year chunks so a long run can be resumed without re-downloading
validated chunks. The final contract-level panel remains create-only and does
not infer constant-maturity FF1/FF3/FF6/FF12 series; that mapping requires a
separately frozen ranking/roll convention.

Raw WRDS/LSEG data are licensed, access-controlled material. The manifests
written by this job explicitly prohibit redistribution of the raw extracts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


PRODUCT_CONTRCODE = 331
PRODUCT_CLSCODE = 245
PRODUCT_NAME = "30 DAY US FEDERAL FUNDS"
PRODUCT_TICKER = "FF"
SOURCE_SCHEMA = "trdstrm"

DEFAULT_START_DATE = date(1988, 1, 1)
DEFAULT_END_DATE = date(2026, 7, 27)
DEFAULT_CHUNK_YEARS = 1
RUN_SPEC_SCHEMA = "wrds-ff-futures-run-spec-v2"
CHUNK_SCHEMA = "wrds-ff-futures-chunk-v2"
FINAL_SCHEMA = "wrds-ff-futures-contract-panel-v2"

METADATA_COLUMNS = (
    "futcode",
    "contrcode",
    "clscode",
    "dsmnem",
    "contrname",
    "ldb",
    "contrdate",
    "contrdatefmt",
    "isocurrcode",
    "isocurrdesc",
    "currunitcode",
    "unitcode",
    "unitdesc",
    "trdstatcode",
    "startdate",
    "lasttrddate",
    "sttlmntdate",
    "expirationdate",
    "firstnoticedate",
    "lastnoticedate",
    "firstdelvrydate",
    "ticksizeunitcode",
    "exchtickersymb",
    "trdmonths",
)
METADATA_DATE_COLUMNS = (
    "startdate",
    "lasttrddate",
    "sttlmntdate",
    "expirationdate",
    "firstnoticedate",
    "lastnoticedate",
    "firstdelvrydate",
)
PRICE_COLUMNS = (
    "futcode",
    "date_",
    "open_",
    "high",
    "low",
    "volume",
    "settlement",
    "openinterest",
)
NUMERIC_PRICE_FIELDS = (
    "open_",
    "high",
    "low",
    "volume",
    "settlement",
    "openinterest",
)
PROFILE_FIELDS = (
    "unique_keys",
    "duplicated_source_keys",
    "source_rows",
    "open_conflicts",
    "high_conflicts",
    "low_conflicts",
    "volume_conflicts",
    "settlement_conflicts",
    "openinterest_conflicts",
    "settlement_all_null_keys",
)
CONFLICT_FIELDS = (
    "open_conflicts",
    "high_conflicts",
    "low_conflicts",
    "volume_conflicts",
    "settlement_conflicts",
    "openinterest_conflicts",
)
CHUNK_FILE_NAMES = (
    "contract_info.csv.gz",
    "contract_prices_daily.csv.gz",
    "source_queries.sql",
    "quality_report.json",
)
FINAL_FILE_NAMES = CHUNK_FILE_NAMES

RESTRICTED_REDISTRIBUTION = {
    "classification": "restricted_licensed_source_data",
    "raw_data_redistribution_permitted": False,
    "public_release_permitted": False,
    "authorized_audience": "Authorized WRDS subscribers covered by applicable WRDS/LSEG terms",
    "license_notice": (
        "Raw extracts remain subject to WRDS and LSEG/Refinitiv Datastream "
        "license terms and must not be committed to a public repository or "
        "redistributed outside the authorized research environment."
    ),
    "derived_outputs": (
        "Only appropriately aggregated or derived results may be shared, "
        "subject to the applicable institutional license."
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--chunk-years", type=int, default=None)
    parser.add_argument(
        "--phase",
        choices=("acquire", "finalize", "all", "status"),
        default=None,
        help="all is the default; status never reads credentials or contacts WRDS",
    )
    parser.add_argument(
        "--status", action="store_true", help="Alias for --phase status"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an unsealed run after verifying its immutable run spec and chunks",
    )
    parser.add_argument("--pgpass", type=Path, default=Path.home() / ".pgpass")
    args = parser.parse_args(argv)
    args.start_date_explicit = args.start_date is not None
    args.end_date_explicit = args.end_date is not None
    args.chunk_years_explicit = args.chunk_years is not None
    args.start_date = args.start_date or DEFAULT_START_DATE
    args.end_date = args.end_date or DEFAULT_END_DATE
    if args.chunk_years is None:
        args.chunk_years = DEFAULT_CHUNK_YEARS
    if args.status and args.phase not in (None, "status"):
        parser.error("--status cannot be combined with a non-status --phase")
    args.phase = "status" if args.status else (args.phase or "all")
    return args


def split_pgpass_line(line: str) -> list[str]:
    parts = re.split(r"(?<!\\):", line.rstrip("\n"), maxsplit=4)
    return [part.replace(r"\:", ":").replace(r"\\", "\\") for part in parts]


def read_wrds_credentials(pgpass: Path) -> tuple[str, int, str, str, str]:
    """Read one WRDS pgpass entry without emitting credential material."""
    if not pgpass.is_file() or pgpass.is_symlink():
        raise FileNotFoundError("WRDS pgpass file was not found")
    mode = pgpass.stat().st_mode & 0o777
    if mode != 0o400:
        raise PermissionError("WRDS pgpass must have exact mode 0400")
    for line in pgpass.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = split_pgpass_line(line)
        if len(parts) != 5:
            continue
        host, port, database, username, password = parts
        if (
            host in {"*", "wrds-pgdata.wharton.upenn.edu"}
            and port in {"*", "9737"}
            and database in {"*", "wrds"}
        ):
            return (
                "wrds-pgdata.wharton.upenn.edu" if host == "*" else host,
                9737 if port == "*" else int(port),
                "wrds" if database == "*" else database,
                username,
                password,
            )
    raise RuntimeError("No usable WRDS PostgreSQL entry was found in pgpass")


def open_wrds_connection(**kwargs: Any) -> Any:
    """Import the optional WRDS client only for an actual acquisition phase."""
    wrds_module = importlib.import_module("wrds")
    return wrds_module.Connection(**kwargs)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def secure_file(path: Path) -> None:
    path.chmod(0o600)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    secure_file(temporary)
    os.replace(temporary, path)
    secure_file(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, canonical_json(payload))


def write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(canonical_json(payload), encoding="utf-8")
    secure_file(temporary)
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"Create-only file already exists: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)
    secure_file(path)


def csv_gzip_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        date_format="%Y-%m-%d",
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    secure_file(temporary)
    os.replace(temporary, path)
    secure_file(path)


def normalise_ids(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("Int64")
    return frame


def iter_date_chunks(
    start_date: date,
    end_date: date,
    chunk_years: int = DEFAULT_CHUNK_YEARS,
) -> list[dict[str, str]]:
    if start_date > end_date:
        raise ValueError("start-date must not be after end-date")
    if chunk_years < 1:
        raise ValueError("chunk-years must be at least 1")
    chunks: list[dict[str, str]] = []
    cursor = start_date
    while cursor <= end_date:
        boundary = date(cursor.year + chunk_years - 1, 12, 31)
        chunk_end = min(boundary, end_date)
        start = cursor.isoformat()
        end = chunk_end.isoformat()
        chunks.append({"chunk_id": f"{start}_{end}", "start": start, "end": end})
        cursor = date(chunk_end.year + 1, 1, 1)
    return chunks


def build_run_spec(
    start_date: date, end_date: date, chunk_years: int
) -> dict[str, Any]:
    chunks = iter_date_chunks(start_date, end_date, chunk_years)
    return {
        "schema_version": RUN_SPEC_SCHEMA,
        "artifact": "wrds_datastream_30day_federal_funds_contract_panel",
        "requested_date_range": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        },
        "chunking": {
            "method": "calendar_year_boundaries",
            "chunk_years": chunk_years,
            "chunks": chunks,
        },
        "source": {
            "provider": "WRDS",
            "database_family": "LSEG/Refinitiv Datastream Futures",
            "schema": SOURCE_SCHEMA,
            "metadata_view": "wrds_contract_info",
            "price_view": "wrds_fut_contract",
            "credentials_embedded": False,
        },
        "product_filter": {
            "contrcode": PRODUCT_CONTRCODE,
            "clscode": PRODUCT_CLSCODE,
            "contrname": PRODUCT_NAME,
            "exchtickersymb": PRODUCT_TICKER,
        },
        "redistribution": RESTRICTED_REDISTRIBUTION,
    }


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"JSON input must be a regular non-symlink file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def load_run_spec(output_root: Path) -> dict[str, Any]:
    path = output_root / "run_spec.json"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Run spec is missing: {path}")
    payload = load_json(path)
    if payload.get("schema_version") != RUN_SPEC_SCHEMA:
        raise RuntimeError("Run spec schema mismatch")
    try:
        observed_range = payload["requested_date_range"]
        observed_chunking = payload["chunking"]
        expected = build_run_spec(
            date.fromisoformat(observed_range["start"]),
            date.fromisoformat(observed_range["end"]),
            int(observed_chunking["chunk_years"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Run spec structure is invalid") from error
    if payload != expected:
        raise RuntimeError("Run spec does not match the canonical acquisition contract")
    return payload


def assert_run_spec_matches(observed: dict[str, Any], expected: dict[str, Any]) -> None:
    if observed != expected:
        raise RuntimeError(
            "Resume/finalize arguments do not match the immutable run spec; "
            "use the original date range and chunk-years"
        )


def assert_explicit_args_match_spec(
    args: argparse.Namespace,
    observed: dict[str, Any],
) -> None:
    observed_range = observed["requested_date_range"]
    if (
        args.start_date_explicit
        and args.start_date.isoformat() != observed_range["start"]
    ):
        raise RuntimeError("Explicit start-date does not match the immutable run spec")
    if args.end_date_explicit and args.end_date.isoformat() != observed_range["end"]:
        raise RuntimeError("Explicit end-date does not match the immutable run spec")
    if args.chunk_years_explicit and args.chunk_years != int(
        observed["chunking"]["chunk_years"]
    ):
        raise RuntimeError("Explicit chunk-years does not match the immutable run spec")


def initialize_run(output_root: Path, spec: dict[str, Any], resume: bool) -> None:
    if output_root.exists():
        if not resume:
            raise FileExistsError(
                f"Output root already exists; pass --resume after inspection: {output_root}"
            )
        assert_run_spec_matches(load_run_spec(output_root), spec)
        return
    output_root.mkdir(parents=True, mode=0o700)
    write_json_create_only(output_root / "run_spec.json", spec)


def target_codes_sql() -> str:
    return f"""
        select distinct futcode
        from {SOURCE_SCHEMA}.wrds_contract_info
        where contrcode = {PRODUCT_CONTRCODE}
          and clscode = {PRODUCT_CLSCODE}
          and contrname = '{PRODUCT_NAME}'
          and exchtickersymb = '{PRODUCT_TICKER}'
    """.strip()


def build_chunk_queries(chunk_start: str, chunk_end: str) -> dict[str, str]:
    codes = target_codes_sql()
    metadata_sql = f"""
        with target_codes as ({codes})
        select i.futcode, i.contrcode, i.clscode, i.dsmnem, i.contrname,
               i.ldb, i.contrdate, i.contrdatefmt, i.isocurrcode,
               i.isocurrdesc, i.currunitcode, i.unitcode, i.unitdesc,
               i.trdstatcode, i.startdate, i.lasttrddate, i.sttlmntdate,
               i.expirationdate, i.firstnoticedate, i.lastnoticedate,
               i.firstdelvrydate, i.ticksizeunitcode, i.exchtickersymb,
               i.trdmonths
        from {SOURCE_SCHEMA}.wrds_contract_info i
        join target_codes t using (futcode)
        where exists (
            select 1
            from {SOURCE_SCHEMA}.wrds_fut_contract p
            where p.futcode = i.futcode
              and p.date_ between date '{chunk_start}' and date '{chunk_end}'
        )
        order by i.lasttrddate, i.futcode
    """.strip()
    profile_sql = f"""
        with target_codes as ({codes}), grouped as (
            select p.futcode, p.date_, count(*) as source_multiplicity,
                   min(p.open_) as min_open, max(p.open_) as max_open,
                   min(p.high) as min_high, max(p.high) as max_high,
                   min(p.low) as min_low, max(p.low) as max_low,
                   min(p.volume) as min_volume, max(p.volume) as max_volume,
                   min(p.settlement) as min_settlement,
                   max(p.settlement) as max_settlement,
                   min(p.openinterest) as min_openinterest,
                   max(p.openinterest) as max_openinterest
            from {SOURCE_SCHEMA}.wrds_fut_contract p
            join target_codes t using (futcode)
            where p.date_ between date '{chunk_start}' and date '{chunk_end}'
            group by p.futcode, p.date_
        )
        select count(*)::bigint as unique_keys,
               count(*) filter (where source_multiplicity > 1)::bigint
                   as duplicated_source_keys,
               coalesce(sum(source_multiplicity), 0)::bigint as source_rows,
               count(*) filter (where min_open is distinct from max_open)::bigint
                   as open_conflicts,
               count(*) filter (where min_high is distinct from max_high)::bigint
                   as high_conflicts,
               count(*) filter (where min_low is distinct from max_low)::bigint
                   as low_conflicts,
               count(*) filter (where min_volume is distinct from max_volume)::bigint
                   as volume_conflicts,
               count(*) filter (
                   where min_settlement is distinct from max_settlement
               )::bigint as settlement_conflicts,
               count(*) filter (
                   where min_openinterest is distinct from max_openinterest
               )::bigint as openinterest_conflicts,
               count(*) filter (where min_settlement is null)::bigint
                   as settlement_all_null_keys
        from grouped
    """.strip()
    price_sql = f"""
        with target_codes as ({codes})
        select p.futcode, p.date_, max(p.open_) as open_, max(p.high) as high,
               max(p.low) as low, max(p.volume) as volume,
               max(p.settlement) as settlement,
               max(p.openinterest) as openinterest
        from {SOURCE_SCHEMA}.wrds_fut_contract p
        join target_codes t using (futcode)
        where p.date_ between date '{chunk_start}' and date '{chunk_end}'
        group by p.futcode, p.date_
        order by p.date_, p.futcode
    """.strip()
    return {"metadata": metadata_sql, "profile": profile_sql, "prices": price_sql}


def render_queries(queries: dict[str, str]) -> str:
    return (
        "-- Contract metadata\n" + queries["metadata"] + ";\n\n"
        "-- Duplicate/conflict profile\n" + queries["profile"] + ";\n\n"
        "-- Canonical daily contract panel\n" + queries["prices"] + ";\n"
    )


def profile_record(frame: pd.DataFrame) -> dict[str, int]:
    if len(frame) != 1:
        raise RuntimeError("WRDS profile query must return exactly one row")
    missing = sorted(set(PROFILE_FIELDS) - set(frame.columns))
    if missing:
        raise RuntimeError(f"WRDS profile is missing fields: {missing}")
    record: dict[str, int] = {}
    for field in PROFILE_FIELDS:
        value = frame.iloc[0][field]
        record[field] = 0 if pd.isna(value) else int(value)
    return record


def normalise_download_frames(
    metadata: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing_metadata = sorted(set(METADATA_COLUMNS) - set(metadata.columns))
    missing_prices = sorted(set(PRICE_COLUMNS) - set(prices.columns))
    if missing_metadata:
        raise RuntimeError(f"Metadata query is missing columns: {missing_metadata}")
    if missing_prices:
        raise RuntimeError(f"Price query is missing columns: {missing_prices}")
    metadata = metadata.loc[:, METADATA_COLUMNS].copy()
    prices = prices.loc[:, PRICE_COLUMNS].copy()
    metadata = normalise_ids(
        metadata,
        ("futcode", "contrcode", "clscode", "unitcode", "ticksizeunitcode"),
    )
    prices = normalise_ids(prices, ("futcode",))
    for column in METADATA_DATE_COLUMNS:
        metadata[column] = pd.to_datetime(metadata[column], errors="coerce").dt.date
    prices["date_"] = pd.to_datetime(prices["date_"], errors="raise").dt.date
    return metadata, prices


def validate_chunk_frames(
    metadata: pd.DataFrame,
    prices: pd.DataFrame,
    profile: dict[str, int],
    chunk_start: date,
    chunk_end: date,
) -> dict[str, Any]:
    if any(profile[field] != 0 for field in CONFLICT_FIELDS):
        raise RuntimeError(
            "Conflicting primary values were found in duplicated WRDS rows"
        )
    metadata_duplicate_keys = int(metadata.duplicated(["futcode"]).sum())
    price_duplicate_keys = int(prices.duplicated(["futcode", "date_"]).sum())
    if metadata_duplicate_keys or price_duplicate_keys:
        raise RuntimeError("Chunk output key uniqueness check failed")
    if prices["futcode"].isna().any() or metadata["futcode"].isna().any():
        raise RuntimeError("Null futcode found in WRDS output")
    price_contracts = set(prices["futcode"].astype(int))
    metadata_contracts = set(metadata["futcode"].astype(int))
    if price_contracts != metadata_contracts:
        raise RuntimeError("Chunk price-to-metadata contract coverage check failed")
    if profile["unique_keys"] != len(prices):
        raise RuntimeError("Chunk database profile and downloaded row count disagree")
    if profile["source_rows"] < profile["unique_keys"]:
        raise RuntimeError(
            "Chunk database profile has fewer source rows than unique keys"
        )
    if len(prices):
        if min(prices["date_"]) < chunk_start or max(prices["date_"]) > chunk_end:
            raise RuntimeError(
                "Chunk contains a price date outside its requested bounds"
            )
    if len(metadata):
        if not (metadata["contrcode"].dropna().astype(int) == PRODUCT_CONTRCODE).all():
            raise RuntimeError("Metadata contains a non-target contrcode")
        if not (metadata["clscode"].dropna().astype(int) == PRODUCT_CLSCODE).all():
            raise RuntimeError("Metadata contains a non-target clscode")
        if not (metadata["contrname"].dropna() == PRODUCT_NAME).all():
            raise RuntimeError("Metadata contains a non-target contract name")
        if not (metadata["exchtickersymb"].dropna() == PRODUCT_TICKER).all():
            raise RuntimeError("Metadata contains a non-target exchange ticker")

    null_counts = {
        field: int(prices[field].isna().sum()) for field in NUMERIC_PRICE_FIELDS
    }
    negative_counts = {
        field: int((prices[field].dropna() < 0).sum())
        for field in ("volume", "openinterest")
    }
    high_below_low = int(
        (
            prices["high"].notna()
            & prices["low"].notna()
            & (prices["high"] < prices["low"])
        ).sum()
    )
    return {
        "status": "passed",
        "requested_date_range": {
            "start": chunk_start.isoformat(),
            "end": chunk_end.isoformat(),
        },
        "intended_grain": ["futcode", "date_"],
        "metadata_rows": int(len(metadata)),
        "metadata_unique_futcodes": int(metadata["futcode"].nunique()),
        "metadata_duplicate_futcodes": metadata_duplicate_keys,
        "price_rows": int(len(prices)),
        "price_unique_keys": int(
            prices[["futcode", "date_"]].drop_duplicates().shape[0]
        ),
        "price_duplicate_keys": price_duplicate_keys,
        "price_contract_count": int(prices["futcode"].nunique()),
        "price_min_date": min(prices["date_"]).isoformat() if len(prices) else None,
        "price_max_date": max(prices["date_"]).isoformat() if len(prices) else None,
        "source_view_profile": profile,
        "primary_field_conflicts_after_source_grouping": {
            field: profile[field] for field in CONFLICT_FIELDS
        },
        "null_counts": null_counts,
        "negative_counts": negative_counts,
        "high_below_low_count": high_below_low,
        "contract_coverage_matches_metadata": True,
        "source_view_duplicate_treatment": (
            "The WRDS friendly price view may repeat (futcode,date_) keys when "
            "auxiliary item values occupy repeated rows. Primary OHLC, "
            "settlement, volume, and open-interest fields must be invariant "
            "within every repeated key before rows are collapsed with MAX."
        ),
    }


def file_records(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    }


def chunk_directory(output_root: Path, chunk: dict[str, str]) -> Path:
    return output_root / "chunks" / chunk["chunk_id"]


def validate_file_records(root: Path, records: dict[str, Any]) -> None:
    for name, expected in records.items():
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"Manifest-bound file is missing: {path}")
        if path.stat().st_size != int(expected["bytes"]):
            raise RuntimeError(f"Manifest-bound file size mismatch: {path}")
        if sha256_file(path) != expected["sha256"]:
            raise RuntimeError(f"Manifest-bound file hash mismatch: {path}")


def validate_chunk_artifact(
    output_root: Path,
    chunk: dict[str, str],
) -> dict[str, Any]:
    directory = chunk_directory(output_root, chunk)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError(f"Completed chunk manifest is missing: {manifest_path}")
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != CHUNK_SCHEMA
        or manifest.get("status") != "complete"
    ):
        raise RuntimeError(f"Invalid chunk manifest: {manifest_path}")
    if manifest.get("chunk") != chunk:
        raise RuntimeError(f"Chunk manifest range mismatch: {manifest_path}")
    if manifest.get("redistribution") != RESTRICTED_REDISTRIBUTION:
        raise RuntimeError(f"Chunk redistribution metadata mismatch: {manifest_path}")
    records = manifest.get("files")
    if not isinstance(records, dict) or set(records) != set(CHUNK_FILE_NAMES):
        raise RuntimeError(f"Chunk file inventory mismatch: {manifest_path}")
    validate_file_records(directory, records)
    quality = load_json(directory / "quality_report.json")
    if quality.get("status") != "passed" or quality != manifest.get("quality_report"):
        raise RuntimeError(f"Chunk quality report mismatch: {manifest_path}")
    return manifest


def acquire_chunk(
    connection: Any,
    output_root: Path,
    chunk: dict[str, str],
) -> dict[str, Any]:
    queries = build_chunk_queries(chunk["start"], chunk["end"])
    metadata = connection.raw_sql(queries["metadata"])
    profile_frame = connection.raw_sql(queries["profile"])
    prices = connection.raw_sql(queries["prices"])
    metadata, prices = normalise_download_frames(metadata, prices)
    profile = profile_record(profile_frame)
    quality = validate_chunk_frames(
        metadata,
        prices,
        profile,
        date.fromisoformat(chunk["start"]),
        date.fromisoformat(chunk["end"]),
    )

    directory = chunk_directory(output_root, chunk)
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / "contract_info.csv.gz"
    prices_path = directory / "contract_prices_daily.csv.gz"
    sql_path = directory / "source_queries.sql"
    quality_path = directory / "quality_report.json"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite sealed chunk: {manifest_path}")
    csv_gzip_atomic(metadata, metadata_path)
    csv_gzip_atomic(prices, prices_path)
    write_text_atomic(sql_path, render_queries(queries))
    write_json_atomic(quality_path, quality)
    files = file_records((metadata_path, prices_path, sql_path, quality_path))
    manifest = {
        "schema_version": CHUNK_SCHEMA,
        "status": "complete",
        "created_at_utc": utc_now(),
        "chunk": chunk,
        "source": {
            "provider": "WRDS",
            "schema": SOURCE_SCHEMA,
            "credentials_embedded": False,
        },
        "product_filter": {
            "contrcode": PRODUCT_CONTRCODE,
            "clscode": PRODUCT_CLSCODE,
            "contrname": PRODUCT_NAME,
            "exchtickersymb": PRODUCT_TICKER,
        },
        "redistribution": RESTRICTED_REDISTRIBUTION,
        "files": files,
        "quality_report": quality,
    }
    write_json_create_only(manifest_path, manifest)
    return manifest


def pending_chunks(output_root: Path, spec: dict[str, Any]) -> list[dict[str, str]]:
    pending: list[dict[str, str]] = []
    for chunk in spec["chunking"]["chunks"]:
        manifest_path = chunk_directory(output_root, chunk) / "manifest.json"
        if manifest_path.is_file():
            validate_chunk_artifact(output_root, chunk)
        else:
            pending.append(chunk)
    return pending


def acquire(output_root: Path, spec: dict[str, Any], pgpass: Path) -> dict[str, Any]:
    pending = pending_chunks(output_root, spec)
    if not pending:
        return {
            "status": "acquired",
            "downloaded_chunks": 0,
            "skipped_chunks": len(spec["chunking"]["chunks"]),
        }

    host, port, database, username, password = read_wrds_credentials(pgpass)
    connection = None
    downloaded = 0
    try:
        connection = open_wrds_connection(
            wrds_hostname=host,
            wrds_port=port,
            wrds_dbname=database,
            wrds_username=username,
            wrds_password=password,
        )
        for chunk in pending:
            acquire_chunk(connection, output_root, chunk)
            downloaded += 1
    except Exception as error:
        # Provider/driver errors can contain connection strings and secrets.
        raise RuntimeError(
            f"WRDS acquisition failed ({type(error).__name__}); credential and connection details were redacted"
        ) from None
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        password = "<redacted>"
        username = "<redacted>"
    return {
        "status": "acquired",
        "downloaded_chunks": downloaded,
        "skipped_chunks": len(spec["chunking"]["chunks"]) - downloaded,
    }


def read_chunk_frames(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(directory / "contract_info.csv.gz", low_memory=False)
    prices = pd.read_csv(directory / "contract_prices_daily.csv.gz", low_memory=False)
    return normalise_download_frames(metadata, prices)


def aggregate_chunks(
    output_root: Path,
    spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, Any], list[dict[str, Any]]]:
    metadata_frames: list[pd.DataFrame] = []
    price_frames: list[pd.DataFrame] = []
    query_blocks: list[str] = []
    chunk_bindings: list[dict[str, Any]] = []
    profiles: list[dict[str, int]] = []
    for chunk in spec["chunking"]["chunks"]:
        manifest = validate_chunk_artifact(output_root, chunk)
        directory = chunk_directory(output_root, chunk)
        metadata, prices = read_chunk_frames(directory)
        metadata_frames.append(metadata)
        price_frames.append(prices)
        query_blocks.append(
            f"-- Chunk {chunk['chunk_id']}\n"
            + (directory / "source_queries.sql").read_text(encoding="utf-8").rstrip()
            + "\n"
        )
        profiles.append(manifest["quality_report"]["source_view_profile"])
        chunk_bindings.append(
            {
                **chunk,
                "manifest_sha256": sha256_file(directory / "manifest.json"),
            }
        )

    metadata = (
        pd.concat(metadata_frames, ignore_index=True)
        if metadata_frames
        else pd.DataFrame(columns=METADATA_COLUMNS)
    )
    prices = (
        pd.concat(price_frames, ignore_index=True)
        if price_frames
        else pd.DataFrame(columns=PRICE_COLUMNS)
    )
    metadata = metadata.drop_duplicates().copy()
    if metadata.duplicated(["futcode"]).any():
        raise RuntimeError("Contract metadata changed across acquisition chunks")
    if prices.duplicated(["futcode", "date_"]).any():
        raise RuntimeError("Price keys overlap across acquisition chunks")
    metadata = metadata.sort_values(
        ["lasttrddate", "futcode"], na_position="last"
    ).reset_index(drop=True)
    prices = prices.sort_values(["date_", "futcode"]).reset_index(drop=True)
    if set(metadata["futcode"].dropna().astype(int)) != set(
        prices["futcode"].dropna().astype(int)
    ):
        raise RuntimeError("Final price-to-metadata contract coverage check failed")

    start = spec["requested_date_range"]["start"]
    end = spec["requested_date_range"]["end"]
    null_counts = {
        field: int(prices[field].isna().sum()) for field in NUMERIC_PRICE_FIELDS
    }
    negative_counts = {
        field: int((prices[field].dropna() < 0).sum())
        for field in ("volume", "openinterest")
    }
    aggregate_profile = {
        field: sum(int(profile[field]) for profile in profiles)
        for field in PROFILE_FIELDS
    }
    quality = {
        "status": "passed",
        "requested_date_range": {"start": start, "end": end},
        "intended_grain": ["futcode", "date_"],
        "chunk_count": len(chunk_bindings),
        "all_chunks_validated": True,
        "metadata_rows": int(len(metadata)),
        "metadata_unique_futcodes": int(metadata["futcode"].nunique()),
        "metadata_duplicate_futcodes": int(metadata.duplicated(["futcode"]).sum()),
        "price_rows": int(len(prices)),
        "price_unique_keys": int(
            prices[["futcode", "date_"]].drop_duplicates().shape[0]
        ),
        "price_duplicate_keys": int(prices.duplicated(["futcode", "date_"]).sum()),
        "price_contract_count": int(prices["futcode"].nunique()),
        "price_min_date": min(prices["date_"]).isoformat() if len(prices) else None,
        "price_max_date": max(prices["date_"]).isoformat() if len(prices) else None,
        "source_view_profile": aggregate_profile,
        "primary_field_conflicts_after_source_grouping": {
            field: aggregate_profile[field] for field in CONFLICT_FIELDS
        },
        "null_counts": null_counts,
        "negative_counts": negative_counts,
        "high_below_low_count": int(
            (
                prices["high"].notna()
                & prices["low"].notna()
                & (prices["high"] < prices["low"])
            ).sum()
        ),
        "contract_coverage_matches_metadata": True,
    }
    return metadata, prices, "\n".join(query_blocks), quality, chunk_bindings


def publish_create_only(staged: Path, target: Path) -> None:
    if target.exists():
        if (
            target.is_file()
            and not target.is_symlink()
            and sha256_file(target) == sha256_file(staged)
        ):
            return
        raise FileExistsError(
            f"Create-only final file already exists with different content: {target}"
        )
    try:
        os.link(staged, target)
    except FileExistsError:
        if (
            target.is_symlink()
            or not target.is_file()
            or sha256_file(target) != sha256_file(staged)
        ):
            raise FileExistsError(
                f"Concurrent create-only publish conflict: {target}"
            ) from None
    secure_file(target)


def validate_sealed_artifact(output_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError("Final manifest is missing")
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != FINAL_SCHEMA
        or manifest.get("status") != "complete"
    ):
        raise RuntimeError("Final manifest is not a complete v2 artifact")
    if manifest.get("run_spec_sha256") != sha256_file(output_root / "run_spec.json"):
        raise RuntimeError("Final manifest run-spec binding failed")
    if manifest.get("requested_date_range") != spec["requested_date_range"]:
        raise RuntimeError("Final manifest date range mismatch")
    if manifest.get("redistribution") != RESTRICTED_REDISTRIBUTION:
        raise RuntimeError("Final redistribution metadata mismatch")
    records = manifest.get("files")
    if not isinstance(records, dict) or set(records) != set(FINAL_FILE_NAMES):
        raise RuntimeError("Final file inventory mismatch")
    validate_file_records(output_root, records)
    expected_bindings = {
        binding["chunk_id"]: binding["manifest_sha256"]
        for binding in manifest.get("chunks", [])
    }
    if set(expected_bindings) != {
        chunk["chunk_id"] for chunk in spec["chunking"]["chunks"]
    }:
        raise RuntimeError("Final chunk inventory mismatch")
    for chunk in spec["chunking"]["chunks"]:
        chunk_manifest = chunk_directory(output_root, chunk) / "manifest.json"
        if sha256_file(chunk_manifest) != expected_bindings[chunk["chunk_id"]]:
            raise RuntimeError(f"Final chunk binding failed: {chunk['chunk_id']}")
    return manifest


def finalize(output_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    if (output_root / "manifest.json").is_file():
        return validate_sealed_artifact(output_root, spec)
    missing = pending_chunks(output_root, spec)
    if missing:
        raise RuntimeError(
            f"Cannot finalize: {len(missing)} acquisition chunks are incomplete"
        )
    metadata, prices, query_text, quality, chunk_bindings = aggregate_chunks(
        output_root, spec
    )

    staging = output_root / ".seal_staging"
    staging.mkdir(mode=0o700, exist_ok=True)
    metadata_path = staging / "contract_info.csv.gz"
    prices_path = staging / "contract_prices_daily.csv.gz"
    sql_path = staging / "source_queries.sql"
    quality_path = staging / "quality_report.json"
    csv_gzip_atomic(metadata, metadata_path)
    csv_gzip_atomic(prices, prices_path)
    write_text_atomic(sql_path, query_text)
    write_json_atomic(quality_path, quality)
    files = file_records((metadata_path, prices_path, sql_path, quality_path))
    for name in FINAL_FILE_NAMES:
        publish_create_only(staging / name, output_root / name)

    try:
        wrds_version = importlib.metadata.version("wrds")
    except importlib.metadata.PackageNotFoundError:
        wrds_version = "unknown"
    manifest = {
        "schema_version": FINAL_SCHEMA,
        "status": "complete",
        "created_at_utc": utc_now(),
        "artifact": "wrds_datastream_30day_federal_funds_contract_panel_v2",
        "run_spec_sha256": sha256_file(output_root / "run_spec.json"),
        "source": spec["source"],
        "product_filter": spec["product_filter"],
        "requested_date_range": spec["requested_date_range"],
        "chunking": {
            "method": spec["chunking"]["method"],
            "chunk_years": spec["chunking"]["chunk_years"],
            "chunk_count": len(chunk_bindings),
        },
        "chunks": chunk_bindings,
        "purpose": (
            "Contract-level source panel for Federal Funds futures decision "
            "baselines and related research."
        ),
        "constant_maturity_series_constructed": False,
        "interpretation_caveat": (
            "FF1/FF3/FF6/FF12 require a separately frozen contract-ranking "
            "and roll convention; this release does not infer that convention."
        ),
        "credentials": {
            "embedded": False,
            "pgpass_path_recorded": False,
            "username_recorded": False,
            "password_recorded": False,
        },
        "redistribution": RESTRICTED_REDISTRIBUTION,
        "environment": {
            "python": platform.python_version(),
            "wrds": wrds_version,
            "pandas": pd.__version__,
        },
        "files": files,
        "quality_report": quality,
    }
    staged_manifest = staging / "manifest.json"
    write_json_atomic(staged_manifest, manifest)
    publish_create_only(staged_manifest, output_root / "manifest.json")
    shutil.rmtree(staging)
    return validate_sealed_artifact(output_root, spec)


def status_report(output_root: Path) -> dict[str, Any]:
    if not output_root.exists():
        return {"status": "not_initialized", "output_root": str(output_root.resolve())}
    try:
        spec = load_run_spec(output_root)
    except Exception as error:
        return {
            "status": "invalid",
            "output_root": str(output_root.resolve()),
            "reason": type(error).__name__,
        }
    complete: list[str] = []
    missing: list[str] = []
    partial: list[str] = []
    invalid: list[str] = []
    for chunk in spec["chunking"]["chunks"]:
        directory = chunk_directory(output_root, chunk)
        manifest_path = directory / "manifest.json"
        if manifest_path.is_file():
            try:
                validate_chunk_artifact(output_root, chunk)
                complete.append(chunk["chunk_id"])
            except Exception:
                invalid.append(chunk["chunk_id"])
        elif directory.exists():
            partial.append(chunk["chunk_id"])
        else:
            missing.append(chunk["chunk_id"])
    sealed = False
    seal_valid = False
    if (output_root / "manifest.json").is_file():
        sealed = True
        try:
            validate_sealed_artifact(output_root, spec)
            seal_valid = True
        except Exception:
            seal_valid = False
    if sealed and seal_valid:
        status = "complete"
    elif sealed or invalid:
        status = "invalid"
    elif len(complete) == len(spec["chunking"]["chunks"]):
        status = "ready_to_finalize"
    else:
        status = "acquiring"
    return {
        "status": status,
        "output_root": str(output_root.resolve()),
        "requested_date_range": spec["requested_date_range"],
        "expected_chunks": len(spec["chunking"]["chunks"]),
        "complete_chunks": len(complete),
        "missing_chunks": missing,
        "partial_chunks": partial,
        "invalid_chunks": invalid,
        "sealed": sealed,
        "seal_valid": seal_valid,
    }


def expected_spec_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return build_run_spec(args.start_date, args.end_date, args.chunk_years)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase == "status":
        print(json.dumps(status_report(args.output_root), sort_keys=True))
        return 0

    spec = expected_spec_from_args(args)
    manifest_path = args.output_root / "manifest.json"
    if manifest_path.is_file():
        observed_spec = load_run_spec(args.output_root)
        assert_explicit_args_match_spec(args, observed_spec)
        manifest = validate_sealed_artifact(args.output_root, observed_spec)
        if not args.resume and args.phase != "finalize":
            raise FileExistsError(
                f"Create-only output is already sealed; use --status: {args.output_root}"
            )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output_root": str(args.output_root.resolve()),
                    "price_rows": manifest["quality_report"]["price_rows"],
                    "chunks": manifest["chunking"]["chunk_count"],
                },
                sort_keys=True,
            )
        )
        return 0

    if args.phase in ("acquire", "all"):
        if args.output_root.exists() and args.resume:
            observed_spec = load_run_spec(args.output_root)
            assert_explicit_args_match_spec(args, observed_spec)
            spec = observed_spec
        initialize_run(args.output_root, spec, resume=args.resume)
        acquisition = acquire(args.output_root, spec, args.pgpass)
    else:
        if not args.output_root.exists():
            raise FileNotFoundError(
                f"Cannot finalize a missing output root: {args.output_root}"
            )
        observed_spec = load_run_spec(args.output_root)
        assert_explicit_args_match_spec(args, observed_spec)
        spec = observed_spec
        acquisition = None

    if args.phase == "acquire":
        print(
            json.dumps(
                {
                    **(acquisition or {}),
                    "output_root": str(args.output_root.resolve()),
                    "chunks": len(spec["chunking"]["chunks"]),
                },
                sort_keys=True,
            )
        )
        return 0

    manifest = finalize(args.output_root, spec)
    quality = manifest["quality_report"]
    print(
        json.dumps(
            {
                "status": "complete",
                "output_root": str(args.output_root.resolve()),
                "metadata_rows": quality["metadata_rows"],
                "price_rows": quality["price_rows"],
                "contracts": quality["price_contract_count"],
                "date_min": quality["price_min_date"],
                "date_max": quality["price_max_date"],
                "chunks": manifest["chunking"]["chunk_count"],
                "settlement_nulls": quality["null_counts"]["settlement"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
