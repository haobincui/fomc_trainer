"""Build the harmonized 2009--2025 CHK3 fixed-Core8 start-D1 panel.

The legacy CHK1 source handoff keys its 128 meetings by the Minutes filename
(the meeting *end* date).  That is not interchangeable with the 1993--2008
external panel, whose point-in-time cutoff is the day before the meeting
*start* date.  This builder therefore derives and seals the official start/end
roster from the paired raw Minutes CSV/XLSX files, creates a new source plan
using the start dates, and materializes every source snapshot again.

No teacher answer, teacher final analysis, current-vintage substitute, or
placeholder is admitted.  Publication is fail-closed unless all 128 meetings
have all eight Core8 topics under the start-date-minus-one-day contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from jobs.eval.build_chk3_external_holdout_1993_2008 import (
    NUMBER_RE,
    BuildError,
    CORE_TOPICS,
    _analysis_text,
    _canonical_json,
    _load_jsonl,
    _load_tokenizer,
    _reference_text,
    _release_files,
    _write_json,
    _write_jsonl,
)
from jobs.generation.prepare_chk4_supplement import (
    _supplement_alfred_http_get,
    compress_indicator_row,
)
from jobs.main.checkpoint_provenance import fingerprint_tokenizer_payload
from jobs.main.fetch_loo_source_snapshots import load_source_registry
from jobs.retrain_v2.chk1.source_pipeline import (
    create_source_plan,
    materialize_source_plan,
    validate_source_handoff,
)
from open_r1.provenance import sha256_file, sha256_text
from open_r1.validator.loo_generation_spec import (
    ManifestIntegrityError,
    seal_manifest,
    validate_manifest_integrity,
)
from open_r1.validator.loo_ledger import _load_registry, _load_roster


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk3/"
    "chk3_minutes_post2008_2009_2025_fixed_core8_d1_v1_candidate"
)
DEFAULT_RELEASE_ROOT = REPO_ROOT / (
    "dataset/processed/retrain_v2/"
    "chk3_minutes_post2008_2009_2025_fixed_core8_d1_v1"
)
DEFAULT_SPLIT_MANIFEST = (
    REPO_ROOT / "dataset/processed/manifests/qa_meeting_split_manifest.json"
)
DEFAULT_RAW_MINUTES_ROOT = (
    REPO_ROOT / "dataset/raw_data/labeled_text/after_2009"
)
DEFAULT_BASE_ROSTER = REPO_ROOT / "configs/main/leave_one_out_roster.json"
DEFAULT_BASE_REGISTRY = REPO_ROOT / "configs/main/loo_indicator_sources.json"
DEFAULT_TOKENIZER = REPO_ROOT / (
    "output/training/retrain_v2/"
    "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
)
LEGACY_END_D1_HANDOFF = (
    REPO_ROOT / "output/data/retrain_v2/chk1/sources_sparse_v1/source_handoff.json"
)

RELEASE_ID = "chk3_minutes_post2008_2009_2025_fixed_core8_d1_v1"
SAMPLE_SCHEMA = "chk3-external-analysis-to-minutes-eval-row-v1"
ROSTER_SCHEMA = "chk3-post2008-start-d1-meeting-roster-v1"
EXPECTED_MEETINGS = 128
EXPECTED_CORE_ROWS = EXPECTED_MEETINGS * len(CORE_TOPICS)
EXPECTED_QA_SPLIT_COUNTS = {"train": 102, "eval": 13, "test": 13}
EXPECTED_SPLIT_COUNTS = {"train": 102, "validation": 13, "test": 13}
EXPECTED_TWO_DAY_MEETINGS = 119
EXPECTED_ONE_DAY_MEETINGS = 9
EXCLUDED_SENSITIVITY_END_DATE = "2009-11-04"
EMERGENCY_END_DATE = "2020-03-15"

# Independent audit anchors.  The first two cover all 129 paired raw files;
# the third covers the 128 meetings admitted by the frozen split manifest.
SPLIT_MANIFEST_SHA256 = (
    "7dfe6bed994e900dc79b41f99241de2cc87928745c9be9d032323cbcbc9f2974"
)
CSV_INVENTORY_SHA256 = (
    "01540870180430452c0f86529978c99acd5be684c07ff1291f06e74f9dfc4d90"
)
XLSX_INVENTORY_SHA256 = (
    "58f3c38ff9b49553f37c9b2a25935b4f840a672adf63c6e6e43f5918d19b7b58"
)
DERIVED_ROSTER_PAYLOAD_SHA256 = (
    "303a2b479466e68fc85180a3ed3d4f3f292ed4fa44e37416f72cd35b7bdfe589"
)

MONTHS = {
    name: number
    for number, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        1,
    )
}
_MONTH_PATTERN = "|".join(MONTHS)
_CROSS_MONTH_HEADING_RE = re.compile(
    rf"^({_MONTH_PATTERN})\s+(\d{{1,2}})\s*-\s*"
    rf"({_MONTH_PATTERN})\s+(\d{{1,2}}),\s*(\d{{4}})"
)
_SAME_MONTH_HEADING_RE = re.compile(
    rf"^({_MONTH_PATTERN})\s+(\d{{1,2}})\s*-\s*(\d{{1,2}}),\s*(\d{{4}})"
)
_SINGLE_DAY_HEADING_RE = re.compile(
    rf"^({_MONTH_PATTERN})\s+(\d{{1,2}}),\s*(\d{{4}})"
)
_FULL_DATE_RE = re.compile(
    rf"({_MONTH_PATTERN})\s+(\d{{1,2}}),\s*(\d{{4}})"
)
_MINUTES_NAME_RE = re.compile(r"^fomcminutes(\d{8})_labeled\.(csv|xlsx)$")


@dataclass(frozen=True)
class HeadingEvidence:
    start_date: str
    end_date: str
    mode: str
    row_number: int
    normalized_text: str


def _normalize_heading_text(value: object) -> str:
    text = str(value or "")
    # Some 2025 CSV rows contain a UTF-8 en dash decoded as Latin-1 controls.
    for token in ("â\x80\x93", "â\x80\x94", "\u2013", "\u2014", "\u2212"):
        text = text.replace(token, "-")
    return " ".join(text.replace("\xa0", " ").split())


def _heading_candidate(text: str) -> tuple[date, date, str] | None:
    match = _CROSS_MONTH_HEADING_RE.match(text)
    if match is not None:
        start_month, start_day, end_month, end_day, year = match.groups()
        return (
            date(int(year), MONTHS[start_month], int(start_day)),
            date(int(year), MONTHS[end_month], int(end_day)),
            "heading_cross_month",
        )
    match = _SAME_MONTH_HEADING_RE.match(text)
    if match is not None:
        month, start_day, end_day, year = match.groups()
        return (
            date(int(year), MONTHS[month], int(start_day)),
            date(int(year), MONTHS[month], int(end_day)),
            "heading_same_month",
        )
    match = _SINGLE_DAY_HEADING_RE.match(text)
    if match is not None:
        month, day, year = match.groups()
        observed = date(int(year), MONTHS[month], int(day))
        return observed, observed, "heading_single_day"
    return None


def _derive_meeting_dates(
    raw_texts: Sequence[object], *, expected_end_date: str
) -> HeadingEvidence:
    expected_end = date.fromisoformat(expected_end_date)
    prose_candidates: list[HeadingEvidence] = []
    for row_number, value in enumerate(raw_texts[:80], 1):
        text = _normalize_heading_text(value)
        heading = _heading_candidate(text)
        if heading is not None and heading[1] == expected_end:
            return HeadingEvidence(
                start_date=heading[0].isoformat(),
                end_date=heading[1].isoformat(),
                mode=heading[2],
                row_number=row_number,
                normalized_text=text,
            )
        lower = text.lower()
        if "meeting" not in lower or " held " not in lower:
            continue
        mentioned = [
            date(int(year), MONTHS[month], int(day))
            for month, day, year in _FULL_DATE_RE.findall(text)
        ]
        if expected_end not in mentioned:
            continue
        safe_dates = [item for item in mentioned if item <= expected_end]
        if not safe_dates:
            continue
        prose_candidates.append(
            HeadingEvidence(
                start_date=min(safe_dates).isoformat(),
                end_date=expected_end.isoformat(),
                mode="meeting_prose_fallback",
                row_number=row_number,
                normalized_text=text,
            )
        )
    if len(prose_candidates) == 1:
        return prose_candidates[0]
    raise BuildError(
        "meeting heading did not close uniquely within first 80 rows for "
        f"{expected_end_date}: fallback_candidates={len(prose_candidates)}"
    )


def _csv_raw_texts(path: Path) -> list[object]:
    values: list[object] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "raw_text" not in (reader.fieldnames or []):
            raise BuildError(f"CSV lacks raw_text column: {path}")
        for index, row in enumerate(reader):
            if index >= 80:
                break
            values.append(row.get("raw_text"))
    return values


def _xlsx_raw_texts(path: Path) -> list[object]:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise BuildError("openpyxl is required for the XLSX roster cross-check") from exc
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook[workbook.sheetnames[0]]
        iterator = worksheet.iter_rows(values_only=True)
        header = next(iterator, None)
        if header is None or "raw_text" not in header:
            raise BuildError(f"XLSX lacks raw_text column: {path}")
        raw_index = header.index("raw_text")
        values: list[object] = []
        for index, row in enumerate(iterator):
            if index >= 80:
                break
            values.append(row[raw_index] if raw_index < len(row) else None)
        return values
    finally:
        workbook.close()


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise BuildError(f"source path is outside the repository: {path}") from exc


def _inventory_digest(paths: Sequence[Path]) -> str:
    lines = [f"{sha256_file(path)}  {_repo_relative(path)}\n" for path in paths]
    return sha256_text("".join(lines))


def _raw_inventory(raw_minutes_root: Path) -> tuple[list[Path], list[Path]]:
    csv_paths = sorted(raw_minutes_root.glob("fomcminutes*_labeled.csv"))
    xlsx_paths = sorted(raw_minutes_root.glob("fomcminutes*_labeled.xlsx"))
    if len(csv_paths) != 129 or len(xlsx_paths) != 129:
        raise BuildError(
            "raw Minutes inventory is not the audited 129 CSV + 129 XLSX files: "
            f"csv={len(csv_paths)}, xlsx={len(xlsx_paths)}"
        )
    csv_keys = {path.stem for path in csv_paths}
    xlsx_keys = {path.stem for path in xlsx_paths}
    if csv_keys != xlsx_keys:
        raise BuildError("raw Minutes CSV/XLSX filename inventories differ")
    if _inventory_digest(csv_paths) != CSV_INVENTORY_SHA256:
        raise BuildError("raw Minutes CSV inventory digest changed")
    if _inventory_digest(xlsx_paths) != XLSX_INVENTORY_SHA256:
        raise BuildError("raw Minutes XLSX inventory digest changed")
    return csv_paths, xlsx_paths


def derive_meeting_roster(
    *, split_manifest: Path, raw_minutes_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if sha256_file(split_manifest) != SPLIT_MANIFEST_SHA256:
        raise BuildError("frozen 128-meeting split manifest hash changed")
    raw_manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    if not isinstance(raw_manifest, list) or len(raw_manifest) != EXPECTED_MEETINGS:
        raise BuildError("frozen split manifest is not a 128-row array")
    csv_paths, xlsx_paths = _raw_inventory(raw_minutes_root)
    csv_by_stem = {path.stem: path for path in csv_paths}
    xlsx_by_stem = {path.stem: path for path in xlsx_paths}

    manifest_dates: set[str] = set()
    roster: list[dict[str, Any]] = []
    derived_payload: list[dict[str, str]] = []
    for item in raw_manifest:
        if not isinstance(item, Mapping):
            raise BuildError("split manifest row is not an object")
        end_date = date.fromisoformat(str(item.get("meeting_date") or "")).isoformat()
        split = str(item.get("split") or "")
        if split not in EXPECTED_QA_SPLIT_COUNTS:
            raise BuildError(f"invalid original split role: {split!r}")
        if end_date in manifest_dates:
            raise BuildError(f"duplicate frozen meeting end date: {end_date}")
        manifest_dates.add(end_date)
        stem = f"fomcminutes{end_date.replace('-', '')}_labeled"
        csv_path = csv_by_stem.get(stem)
        xlsx_path = xlsx_by_stem.get(stem)
        if csv_path is None or xlsx_path is None:
            raise BuildError(f"paired raw Minutes files are missing for {end_date}")
        csv_heading = _derive_meeting_dates(
            _csv_raw_texts(csv_path), expected_end_date=end_date
        )
        xlsx_heading = _derive_meeting_dates(
            _xlsx_raw_texts(xlsx_path), expected_end_date=end_date
        )
        if (
            csv_heading.start_date,
            csv_heading.end_date,
        ) != (
            xlsx_heading.start_date,
            xlsx_heading.end_date,
        ):
            raise BuildError(f"CSV/XLSX meeting-date cross-check failed: {end_date}")
        start_date = csv_heading.start_date
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        if start > end or (end - start).days not in {0, 1}:
            raise BuildError(f"unsupported meeting duration: {start_date}::{end_date}")
        meeting_type = "emergency" if end_date == EMERGENCY_END_DATE else "regular"
        formal_split = "validation" if split == "eval" else split
        sensitivity_flag = end_date == EMERGENCY_END_DATE
        sensitivity_reason = (
            "emergency_unscheduled_sunday_meeting" if sensitivity_flag else None
        )
        csv_relative = _repo_relative(csv_path)
        xlsx_relative = _repo_relative(xlsx_path)
        csv_sha = sha256_file(csv_path)
        xlsx_sha = sha256_file(xlsx_path)
        derived_payload.append(
            {
                "meeting_start_date": start_date,
                "meeting_end_date": end_date,
                "split": split,
                "raw_csv": csv_relative,
                "raw_csv_sha256": csv_sha,
            }
        )
        roster.append(
            {
                "schema_version": ROSTER_SCHEMA,
                "meeting_id": (
                    f"fomc-{start_date.replace('-', '')}-{end_date.replace('-', '')}"
                ),
                "meeting_start_date": start_date,
                "meeting_end_date": end_date,
                "evidence_cutoff": (start - timedelta(days=1)).isoformat(),
                "meeting_duration_days": (end - start).days + 1,
                "meeting_type": meeting_type,
                "scheduled": meeting_type == "regular",
                "sensitivity_flag": sensitivity_flag,
                "sensitivity_reason": sensitivity_reason,
                "source_split": formal_split,
                "original_split_role": formal_split,
                "original_qa_split": split,
                "raw_minutes_csv": csv_relative,
                "raw_minutes_csv_sha256": csv_sha,
                "raw_minutes_xlsx": xlsx_relative,
                "raw_minutes_xlsx_sha256": xlsx_sha,
                "csv_heading_mode": csv_heading.mode,
                "csv_heading_row_number": csv_heading.row_number,
                "xlsx_heading_mode": xlsx_heading.mode,
                "xlsx_heading_row_number": xlsx_heading.row_number,
                "official_minutes_url": (
                    "https://www.federalreserve.gov/monetarypolicy/"
                    f"fomcminutes{end_date.replace('-', '')}.htm"
                ),
            }
        )

    observed_split_counts = dict(Counter(row["source_split"] for row in roster))
    if observed_split_counts != EXPECTED_SPLIT_COUNTS:
        raise BuildError(f"frozen split counts changed: {observed_split_counts}")
    if len({row["meeting_start_date"] for row in roster}) != EXPECTED_MEETINGS:
        raise BuildError("derived meeting start dates are not unique")
    duration_counts = Counter(row["meeting_duration_days"] for row in roster)
    if duration_counts != {
        2: EXPECTED_TWO_DAY_MEETINGS,
        1: EXPECTED_ONE_DAY_MEETINGS,
    }:
        raise BuildError(f"meeting-duration closure changed: {dict(duration_counts)}")
    derived_sha = sha256_text(_canonical_json(derived_payload))
    if derived_sha != DERIVED_ROSTER_PAYLOAD_SHA256:
        raise BuildError(f"derived roster payload changed: {derived_sha}")

    selected_stems = {
        f"fomcminutes{row['meeting_end_date'].replace('-', '')}_labeled"
        for row in roster
    }
    unselected = sorted(set(csv_by_stem) - selected_stems)
    expected_excluded_stem = (
        f"fomcminutes{EXCLUDED_SENSITIVITY_END_DATE.replace('-', '')}_labeled"
    )
    if unselected != [expected_excluded_stem]:
        raise BuildError(f"unexpected raw Minutes exclusions: {unselected}")
    inventory_rows: list[dict[str, Any]] = []
    for csv_path in csv_paths:
        match = _MINUTES_NAME_RE.fullmatch(csv_path.name)
        if match is None:
            raise BuildError(f"unexpected raw Minutes filename: {csv_path.name}")
        end_date = date(
            int(match.group(1)[:4]),
            int(match.group(1)[4:6]),
            int(match.group(1)[6:]),
        ).isoformat()
        xlsx_path = xlsx_by_stem[csv_path.stem]
        inventory_rows.append(
            {
                "schema_version": "chk3-post2008-raw-minutes-binding-v1",
                "meeting_end_date": end_date,
                "admitted_by_frozen_split_manifest": end_date in manifest_dates,
                "sensitivity_flag": end_date == EXCLUDED_SENSITIVITY_END_DATE,
                "sensitivity_reason": (
                    "excluded_from_frozen_128_meeting_roster"
                    if end_date == EXCLUDED_SENSITIVITY_END_DATE
                    else None
                ),
                "csv_path": _repo_relative(csv_path),
                "csv_sha256": sha256_file(csv_path),
                "csv_bytes": csv_path.stat().st_size,
                "xlsx_path": _repo_relative(xlsx_path),
                "xlsx_sha256": sha256_file(xlsx_path),
                "xlsx_bytes": xlsx_path.stat().st_size,
            }
        )
    audit = {
        "schema_version": "chk3-post2008-start-d1-roster-audit-v1",
        "status": "passed",
        "split_manifest_path": _repo_relative(split_manifest),
        "split_manifest_sha256": SPLIT_MANIFEST_SHA256,
        "csv_inventory_count": len(csv_paths),
        "csv_inventory_sha256": CSV_INVENTORY_SHA256,
        "xlsx_inventory_count": len(xlsx_paths),
        "xlsx_inventory_sha256": XLSX_INVENTORY_SHA256,
        "derived_roster_payload_sha256": derived_sha,
        "meeting_count": len(roster),
        "formal_split_counts": observed_split_counts,
        "original_qa_split_counts": dict(
            Counter(row["original_qa_split"] for row in roster)
        ),
        "two_day_meetings": duration_counts[2],
        "one_day_meetings": duration_counts[1],
        "csv_xlsx_date_mismatches": 0,
        "emergency_meeting": EMERGENCY_END_DATE,
        "excluded_sensitivity_meeting": EXCLUDED_SENSITIVITY_END_DATE,
    }
    return roster, inventory_rows, audit


def _derive_source_contracts(
    *, output_root: Path, base_roster: Path, base_registry: Path
) -> tuple[Path, Path]:
    roster = deepcopy(json.loads(base_roster.read_text(encoding="utf-8")))
    registry = deepcopy(json.loads(base_registry.read_text(encoding="utf-8")))
    indicators = roster.get("indicators")
    if not isinstance(indicators, list) or len(indicators) != 26:
        raise BuildError("base roster is not the frozen 26-topic roster")
    if set(registry.get("indicators") or {}) != set(indicators):
        raise BuildError("base registry and roster topics differ")
    required_policy = {
        "vintage_lag_calendar_days": 1,
        "same_meeting_day_data": "excluded",
        "unknown_availability": "fail_closed",
        "runtime_fallback": "forbidden",
    }
    policy = registry.get("policy")
    if not isinstance(policy, Mapping) or any(
        policy.get(key) != value for key, value in required_policy.items()
    ):
        raise BuildError("base registry is not the exact D-1 contract")
    roster["roster_id"] = "chk3-post2008-start-d1-26-indicators-v1"
    roster["contexts"] = ["2009_2025_frozen_128_meeting_start_d1"]
    roster["derived_from"] = {
        "path": _repo_relative(base_roster),
        "sha256": sha256_file(base_roster),
    }
    registry["registry_id"] = "chk3-post2008-keyless-start-d1-v1"
    registry["roster_id"] = roster["roster_id"]
    registry["derived_from"] = {
        "path": _repo_relative(base_registry),
        "sha256": sha256_file(base_registry),
    }
    roster_path = output_root / "sources/contracts/leave_one_out_roster.json"
    registry_path = output_root / "sources/contracts/loo_indicator_sources.json"
    _write_json(roster_path, roster)
    _write_json(registry_path, registry)
    load_source_registry(registry_path)
    roster_values = _load_roster(roster_path)
    _load_registry(registry_path, roster=roster_values)
    return roster_path, registry_path


def prepare(
    *,
    output_root: Path,
    split_manifest: Path,
    raw_minutes_root: Path,
    base_roster: Path,
    base_registry: Path,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"candidate output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    roster, inventory, roster_audit = derive_meeting_roster(
        split_manifest=split_manifest,
        raw_minutes_root=raw_minutes_root,
    )
    roster_path = output_root / "sources/meeting_roster.jsonl"
    inventory_path = output_root / "sources/raw_minutes_inventory.jsonl"
    _write_jsonl(roster_path, roster)
    _write_jsonl(inventory_path, inventory)
    _write_json(output_root / "reports/roster_audit.json", roster_audit)
    source_roster, source_registry = _derive_source_contracts(
        output_root=output_root,
        base_roster=base_roster,
        base_registry=base_registry,
    )
    meetings_by_split = {
        split: sorted(
            str(row["meeting_start_date"])
            for row in roster
            if row["original_qa_split"] == split
        )
        for split in EXPECTED_QA_SPLIT_COUNTS
    }
    plan_path = create_source_plan(
        meetings_by_split=meetings_by_split,
        output_dir=output_root / "sources/plan",
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("meeting_counts") != EXPECTED_QA_SPLIT_COUNTS:
        raise BuildError(f"start-D1 source plan split closure failed: {plan}")
    summary = {
        "schema_version": "chk3-post2008-start-d1-preparation-v1",
        "status": "source_acquisition_pending",
        "release_id": RELEASE_ID,
        "meeting_count": len(roster),
        "core8_target_rows": EXPECTED_CORE_ROWS,
        "formal_split_counts": EXPECTED_SPLIT_COUNTS,
        "source_plan_split_counts": EXPECTED_QA_SPLIT_COUNTS,
        "evidence_cutoff_policy": "meeting_start_date_minus_1_calendar_day",
        "two_day_meetings_requiring_fresh_source_vintage": (
            EXPECTED_TWO_DAY_MEETINGS
        ),
        "legacy_end_d1_rows_reused": 0,
        "teacher_final_analysis_used": False,
        "placeholder_rows_allowed": False,
        "meeting_roster": {
            "path": str(roster_path.resolve()),
            "sha256": sha256_file(roster_path),
        },
        "raw_minutes_inventory": {
            "path": str(inventory_path.resolve()),
            "sha256": sha256_file(inventory_path),
            "rows": len(inventory),
        },
        "source_plan": {
            "path": str(plan_path.resolve()),
            "sha256": sha256_file(plan_path),
            "payload_sha256": plan["payload_sha256"],
        },
        "source_roster_sha256": sha256_file(source_roster),
        "source_registry_sha256": sha256_file(source_registry),
    }
    _write_json(output_root / "reports/preparation.json", summary)
    return summary


def acquire(
    *,
    output_root: Path,
    resume: bool,
    requests_per_second: float,
    max_workers: int,
) -> dict[str, Any]:
    plan = output_root / "sources/plan/source_plan.json"
    roster = output_root / "sources/contracts/leave_one_out_roster.json"
    registry = output_root / "sources/contracts/loo_indicator_sources.json"
    for path in (plan, roster, registry):
        if not path.is_file():
            raise BuildError(f"preparation artifact is missing: {path}")
    handoff = materialize_source_plan(
        plan_path=plan,
        registry_path=registry,
        roster_path=roster,
        output_dir=output_root / "sources/materialized",
        sparse_train=True,
        resume=resume,
        max_workers=max_workers,
        requests_per_second=requests_per_second,
        sparse_http_get=_supplement_alfred_http_get,
    )
    validated = validate_source_handoff(
        handoff_path=handoff,
        plan_path=plan,
        registry_path=registry,
        roster_path=roster,
    )
    summary = {
        "schema_version": "chk3-post2008-start-d1-source-acquisition-v1",
        "status": "complete_pending_core8_publish_gate",
        "handoff": {
            "path": str(handoff.resolve()),
            "sha256": sha256_file(handoff),
            "payload_sha256": validated["payload_sha256"],
        },
        "population_count": len(validated["ledgers"]),
        "meeting_count": sum(
            int(record["meeting_count"]) for record in validated["ledgers"]
        ),
        "ready_topic_rows": sum(
            int(record["row_count"]) for record in validated["ledgers"]
        ),
        "excluded_topic_rows": sum(
            int(record.get("excluded_sample_count") or 0)
            for record in validated["ledgers"]
        ),
    }
    if summary["meeting_count"] != EXPECTED_MEETINGS:
        raise BuildError(f"source handoff meeting closure failed: {summary}")
    _write_json(output_root / "reports/source_acquisition.json", summary)
    return summary


def _core8_missing(
    topic_map: Mapping[str, Mapping[str, Any]], meetings: Sequence[str]
) -> dict[str, list[str]]:
    return {
        meeting: sorted(set(CORE_TOPICS) - set(topic_map.get(meeting, {})))
        for meeting in meetings
        if set(CORE_TOPICS) - set(topic_map.get(meeting, {}))
    }


def _require_core8_closure(
    topic_map: Mapping[str, Mapping[str, Any]], meetings: Sequence[str]
) -> None:
    missing = _core8_missing(topic_map, meetings)
    if missing:
        raise BuildError(
            "start-D1 Core8 closure failed; publication is blocked: "
            + json.dumps(missing, sort_keys=True)
        )


def _validated_source_rows(
    *, output_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = output_root / "sources/plan/source_plan.json"
    source_roster = output_root / "sources/contracts/leave_one_out_roster.json"
    source_registry = output_root / "sources/contracts/loo_indicator_sources.json"
    handoff = output_root / "sources/materialized/source_handoff.json"
    validated = validate_source_handoff(
        handoff_path=handoff,
        plan_path=plan,
        registry_path=source_registry,
        roster_path=source_roster,
    )
    rows: list[dict[str, Any]] = []
    for record in validated["ledgers"]:
        ledger_dir = Path(str(record["ledger_dir"]))
        rows.extend(_load_jsonl(ledger_dir / "indicator_inputs.jsonl"))
    return rows, validated


def publish(
    *, output_root: Path, release_root: Path, tokenizer_path: Path
) -> dict[str, Any]:
    if release_root.exists():
        raise FileExistsError(f"release destination already exists: {release_root}")
    roster_rows = _load_jsonl(output_root / "sources/meeting_roster.jsonl")
    if len(roster_rows) != EXPECTED_MEETINGS:
        raise BuildError("sealed meeting roster is incomplete")
    by_start = {str(row["meeting_start_date"]): row for row in roster_rows}
    if len(by_start) != EXPECTED_MEETINGS:
        raise BuildError("sealed meeting roster start dates are not unique")
    source_rows, validated = _validated_source_rows(output_root=output_root)
    topic_map: dict[str, dict[str, tuple[dict[str, Any], str]]] = defaultdict(dict)
    provenance_map: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    pit_violations: list[str] = []
    for row in source_rows:
        meeting = str(row.get("meeting_date") or "")
        if meeting not in by_start:
            raise BuildError(f"source meeting is outside sealed start roster: {meeting}")
        topic = str(row.get("indicator") or "")
        cutoff = str(by_start[meeting]["evidence_cutoff"])
        for field in (
            "requested_vintage_date",
            "availability_as_of_date",
            "information_as_of_date",
        ):
            if str(row.get(field) or "") != cutoff:
                pit_violations.append(f"{meeting}::{topic}::{field}")
        compressed = compress_indicator_row(row)
        if compressed is None:
            continue
        for series in compressed["provenance"]:
            if str(series["latest_observation_date"]) > cutoff:
                pit_violations.append(f"{meeting}::{topic}::future_observation")
        if topic in topic_map[meeting]:
            raise BuildError(f"duplicate source meeting/topic: {meeting}::{topic}")
        source_id = str(row.get("source_id") or "")
        if not source_id:
            raise BuildError(f"source_id is missing: {meeting}::{topic}")
        provenance_map[meeting][topic] = list(compressed["provenance"])
        topic_map[meeting][topic] = (
            {key: value for key, value in compressed.items() if key != "provenance"},
            source_id,
        )
    if pit_violations:
        raise BuildError(f"point-in-time violations: {pit_violations[:5]}")
    meetings = sorted(by_start)
    _require_core8_closure(topic_map, meetings)

    tokenizer = _load_tokenizer(tokenizer_path)
    rows: list[dict[str, Any]] = []
    prompt_tokens: list[int] = []
    for start in meetings:
        meeting = by_start[start]
        for topic in sorted(CORE_TOPICS):
            evidence, source_id = topic_map[start][topic]
            analysis = _analysis_text(topic, evidence)
            reference = _reference_text(topic, evidence)
            if Counter(NUMBER_RE.findall(analysis)) != Counter(
                NUMBER_RE.findall(reference)
            ):
                raise BuildError(f"deterministic reference changed numbers: {start}::{topic}")
            prompt = (
                "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
                + _canonical_json({"analysis": analysis})
            )
            token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
            prompt_tokens.append(token_count)
            sample_id = (
                "chk3-post2008-"
                + sha256_text(
                    f"{RELEASE_ID}\0{meeting['meeting_id']}\0{topic}"
                )[:24]
            )
            rows.append(
                {
                    "schema_version": SAMPLE_SCHEMA,
                    "sample_id": sample_id,
                    "meeting_id": meeting["meeting_id"],
                    "meeting_start_date": start,
                    "meeting_end_date": meeting["meeting_end_date"],
                    "evidence_cutoff": meeting["evidence_cutoff"],
                    "meeting_type": meeting["meeting_type"],
                    "scheduled": meeting["scheduled"],
                    "sensitivity_flag": meeting["sensitivity_flag"],
                    "sensitivity_reason": meeting["sensitivity_reason"],
                    "source_split": meeting["source_split"],
                    "original_split_role": meeting["original_split_role"],
                    "original_qa_split": meeting["original_qa_split"],
                    "topic": topic,
                    "panel": "core8",
                    "prompt": prompt,
                    "source_analysis": analysis,
                    "reference_minutes": reference,
                    "reference_type": (
                        "deterministic_source_grounded_minutes_style_v1"
                    ),
                    "official_minutes_url": meeting["official_minutes_url"],
                    "official_minutes_role": "secondary_context_not_model_input",
                    "raw_minutes_csv": meeting["raw_minutes_csv"],
                    "raw_minutes_csv_sha256": meeting["raw_minutes_csv_sha256"],
                    "source_id": source_id,
                    "source_vintage_basis": "meeting_start_date_minus_1_calendar_day",
                    "topic_evidence": evidence,
                    "topic_evidence_provenance": provenance_map[start][topic],
                    "prompt_tokens": token_count,
                    "prompt_sha256": sha256_text(prompt),
                    "source_analysis_sha256": sha256_text(analysis),
                    "reference_minutes_sha256": sha256_text(reference),
                    "topic_evidence_sha256": sha256_text(_canonical_json(evidence)),
                }
            )
    if len(rows) != EXPECTED_CORE_ROWS:
        raise BuildError(f"start-D1 Core8 row closure failed: {len(rows)}")

    release_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{release_root.name}.", dir=release_root.parent)
    )
    try:
        _write_jsonl(staging / "official_meeting_roster.jsonl", roster_rows)
        _write_jsonl(staging / "panels/core8.jsonl", rows)
        inventory_rows = _load_jsonl(
            output_root / "sources/raw_minutes_inventory.jsonl"
        )
        _write_jsonl(
            staging / "references/raw_minutes_inventory.jsonl", inventory_rows
        )
        split_counts = dict(Counter(row["source_split"] for row in roster_rows))
        roster_audit = json.loads(
            (output_root / "reports/roster_audit.json").read_text(encoding="utf-8")
        )
        source_handoff = output_root / "sources/materialized/source_handoff.json"
        source_audit = {
            "schema_version": "chk3-post2008-start-d1-source-audit-v1",
            "status": "passed",
            "meeting_count": EXPECTED_MEETINGS,
            "core8_rows": len(rows),
            "topic_coverage": dict(sorted(Counter(row["topic"] for row in rows).items())),
            "split_counts": split_counts,
            "point_in_time_violations": 0,
            "input_truncation": 0,
            "prompt_tokens": {"min": min(prompt_tokens), "max": max(prompt_tokens)},
            "evidence_cutoff_policy": "meeting_start_date_minus_1_calendar_day",
            "two_day_meetings_reacquired": EXPECTED_TWO_DAY_MEETINGS,
            "legacy_end_d1_rows_reused": 0,
            "legacy_end_d1_handoff": {
                "path": str(LEGACY_END_D1_HANDOFF.resolve()),
                "sha256": sha256_file(LEGACY_END_D1_HANDOFF),
            },
            "source_handoff": {
                "path": str(source_handoff.resolve()),
                "sha256": sha256_file(source_handoff),
                "payload_sha256": validated["payload_sha256"],
            },
        }
        reference_audit = {
            "schema_version": "chk3-post2008-reference-audit-v1",
            "status": "passed",
            "reference_type": "deterministic_source_grounded_minutes_style_v1",
            "numeric_multiset_mismatch_rows": 0,
            "teacher_final_analysis_used": False,
            "teacher_answer_used": False,
            "placeholder_rows": 0,
            "official_minutes_in_model_prompt": False,
            "decision_gold_in_model_prompt": False,
        }
        sensitivity_audit = {
            "schema_version": "chk3-post2008-sensitivity-audit-v1",
            "status": "passed",
            "included_emergency_meeting": {
                "meeting_end_date": EMERGENCY_END_DATE,
                "meeting_type": "emergency",
                "scheduled": False,
                "sensitivity_flag": True,
                "sensitivity_reason": "emergency_unscheduled_sunday_meeting",
            },
            "excluded_regular_meeting": {
                "meeting_end_date": EXCLUDED_SENSITIVITY_END_DATE,
                "reason": "absent_from_frozen_128_meeting_split_manifest",
                "sensitivity_flag": True,
                "sensitivity_reason": "excluded_from_frozen_128_meeting_roster",
                "raw_source_files_are_bound_in_inventory": True,
            },
            "primary_panel_includes_excluded_regular_meeting": False,
        }
        _write_json(staging / "audits/roster_completeness.json", roster_audit)
        _write_json(staging / "audits/source_quality.json", source_audit)
        _write_json(staging / "audits/reference_quality.json", reference_audit)
        _write_json(staging / "audits/sensitivity_scope.json", sensitivity_audit)
        tokenizer_fingerprint = fingerprint_tokenizer_payload(tokenizer_path)
        files = _release_files(staging)
        manifest = {
            "schema_version": "chk3-external-evaluation-release-v1",
            "release_id": RELEASE_ID,
            "created_at_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "status": "passed",
            "immutable": True,
            "evaluation_only": True,
            "trainable": False,
            "checkpoint_selection_allowed": False,
            "promotable": False,
            "external_holdout": False,
            "post2008_training_domain_overlap": True,
            "task_contract": "analysis_to_formal_fomc_minutes_prose",
            "grain": "one_frozen_meeting_topic_per_row",
            "meeting_count": EXPECTED_MEETINGS,
            "core_topic_count": len(CORE_TOPICS),
            "core8_rows": len(rows),
            "source_split_counts": split_counts,
            "evidence_cutoff_policy": "meeting_start_date_minus_1_calendar_day",
            "legacy_end_d1_rows_reused": 0,
            "tokenizer": {
                "path": str(tokenizer_path.resolve()),
                "payload": tokenizer_fingerprint,
            },
            "source_lineage": {
                "candidate_root": str(output_root.resolve()),
                "builder_sha256": sha256_file(Path(__file__)),
                "meeting_roster_path": str(
                    (output_root / "sources/meeting_roster.jsonl").resolve()
                ),
                "meeting_roster_sha256": sha256_file(
                    output_root / "sources/meeting_roster.jsonl"
                ),
                "source_plan_path": str(
                    (output_root / "sources/plan/source_plan.json").resolve()
                ),
                "source_plan_sha256": sha256_file(
                    output_root / "sources/plan/source_plan.json"
                ),
                "source_roster_path": str(
                    (
                        output_root
                        / "sources/contracts/leave_one_out_roster.json"
                    ).resolve()
                ),
                "source_roster_sha256": sha256_file(
                    output_root / "sources/contracts/leave_one_out_roster.json"
                ),
                "source_registry_path": str(
                    (
                        output_root
                        / "sources/contracts/loo_indicator_sources.json"
                    ).resolve()
                ),
                "source_registry_sha256": sha256_file(
                    output_root / "sources/contracts/loo_indicator_sources.json"
                ),
                "source_handoff_path": str(source_handoff.resolve()),
                "source_handoff_sha256": sha256_file(source_handoff),
                "source_handoff_payload_sha256": validated["payload_sha256"],
                "derived_roster_payload_sha256": DERIVED_ROSTER_PAYLOAD_SHA256,
            },
            "files": files,
        }
        _write_json(staging / "release_manifest.json", seal_manifest(manifest))
        os.rename(staging, release_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    result = {
        "status": "complete",
        "release_root": str(release_root.resolve()),
        "release_manifest": str((release_root / "release_manifest.json").resolve()),
        "release_manifest_sha256": sha256_file(release_root / "release_manifest.json"),
        "meeting_count": EXPECTED_MEETINGS,
        "core8_rows": len(rows),
    }
    _write_json(output_root / "reports/publish.json", result)
    return result


def _validate_external_binding(record: Mapping[str, Any], *, suffix: str) -> None:
    relative = str(record.get(f"{suffix}_path") or "")
    path = REPO_ROOT / relative
    if (
        not path.is_file()
        or path.stat().st_size != record.get(f"{suffix}_bytes")
        or sha256_file(path) != record.get(f"{suffix}_sha256")
    ):
        raise BuildError(f"raw Minutes {suffix} binding changed: {relative}")


def validate_release(release_root: Path) -> dict[str, Any]:
    manifest_path = release_root / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        payload_sha256 = validate_manifest_integrity(manifest)
    except ManifestIntegrityError as exc:
        raise BuildError(f"release manifest integrity failed: {exc}") from exc
    if (
        manifest.get("schema_version") != "chk3-external-evaluation-release-v1"
        or manifest.get("status") != "passed"
        or manifest.get("evaluation_only") is not True
        or manifest.get("trainable") is not False
        or manifest.get("checkpoint_selection_allowed") is not False
        or manifest.get("meeting_count") != EXPECTED_MEETINGS
        or manifest.get("core8_rows") != EXPECTED_CORE_ROWS
        or manifest.get("legacy_end_d1_rows_reused") != 0
    ):
        raise BuildError("release scope/count contract is invalid")
    for relative, record in manifest.get("files", {}).items():
        path = release_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or sha256_file(path) != record.get("sha256")
        ):
            raise BuildError(f"release file binding changed: {relative}")
    roster = _load_jsonl(release_root / "official_meeting_roster.jsonl")
    core = _load_jsonl(release_root / "panels/core8.jsonl")
    if len(roster) != EXPECTED_MEETINGS or len(core) != EXPECTED_CORE_ROWS:
        raise BuildError("validated release row closure failed")
    roster_by_id = {str(row["meeting_id"]): row for row in roster}
    if len(roster_by_id) != EXPECTED_MEETINGS:
        raise BuildError("validated release has duplicate meeting IDs")
    keys = {(row["meeting_id"], row["topic"]) for row in core}
    if len(keys) != EXPECTED_CORE_ROWS:
        raise BuildError("validated release has duplicate meeting/topic keys")
    if set(Counter(row["topic"] for row in core).values()) != {EXPECTED_MEETINGS}:
        raise BuildError("validated release topic coverage changed")
    for row in core:
        meeting = roster_by_id.get(str(row["meeting_id"]))
        if meeting is None:
            raise BuildError(f"Core8 row has unknown meeting: {row['meeting_id']}")
        cutoff = (date.fromisoformat(row["meeting_start_date"]) - timedelta(days=1)).isoformat()
        for field in (
            "meeting_start_date",
            "meeting_end_date",
            "evidence_cutoff",
            "meeting_type",
            "scheduled",
            "sensitivity_flag",
            "sensitivity_reason",
            "source_split",
            "original_split_role",
            "original_qa_split",
        ):
            if row.get(field) != meeting.get(field):
                raise BuildError(f"Core8 roster binding changed: {row['sample_id']}::{field}")
        if row["evidence_cutoff"] != cutoff:
            raise BuildError(f"Core8 D-1 cutoff changed: {row['sample_id']}")
        evidence = row.get("topic_evidence")
        if not isinstance(evidence, Mapping):
            raise BuildError(f"Core8 evidence missing: {row['sample_id']}")
        analysis = _analysis_text(str(row["topic"]), evidence)
        reference = _reference_text(str(row["topic"]), evidence)
        prompt = (
            "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
            + _canonical_json({"analysis": analysis})
        )
        if (
            row.get("source_analysis") != analysis
            or row.get("reference_minutes") != reference
            or row.get("prompt") != prompt
            or row.get("source_analysis_sha256") != sha256_text(analysis)
            or row.get("reference_minutes_sha256") != sha256_text(reference)
            or row.get("prompt_sha256") != sha256_text(prompt)
            or row.get("topic_evidence_sha256")
            != sha256_text(_canonical_json(evidence))
        ):
            raise BuildError(f"deterministic row binding changed: {row['sample_id']}")
        if Counter(NUMBER_RE.findall(analysis)) != Counter(
            NUMBER_RE.findall(reference)
        ):
            raise BuildError(f"numeric multiset changed: {row['sample_id']}")
    inventory = _load_jsonl(
        release_root / "references/raw_minutes_inventory.jsonl"
    )
    if len(inventory) != 129:
        raise BuildError("raw Minutes inventory row closure failed")
    for record in inventory:
        _validate_external_binding(record, suffix="csv")
        _validate_external_binding(record, suffix="xlsx")
    csv_paths = sorted(REPO_ROOT / record["csv_path"] for record in inventory)
    xlsx_paths = sorted(REPO_ROOT / record["xlsx_path"] for record in inventory)
    if (
        _inventory_digest(csv_paths) != CSV_INVENTORY_SHA256
        or _inventory_digest(xlsx_paths) != XLSX_INVENTORY_SHA256
    ):
        raise BuildError("raw Minutes inventory aggregate binding changed")
    lineage = manifest.get("source_lineage") or {}
    for label in ("source_plan", "source_roster", "source_registry", "source_handoff"):
        path = Path(str(lineage.get(f"{label}_path") or ""))
        if not path.is_file() or sha256_file(path) != lineage.get(f"{label}_sha256"):
            raise BuildError(f"source lineage binding changed: {label}")
    validated = validate_source_handoff(
        handoff_path=Path(lineage["source_handoff_path"]),
        plan_path=Path(lineage["source_plan_path"]),
        registry_path=Path(lineage["source_registry_path"]),
        roster_path=Path(lineage["source_roster_path"]),
    )
    if validated["payload_sha256"] != lineage.get("source_handoff_payload_sha256"):
        raise BuildError("source handoff payload binding changed")
    for relative in (
        "audits/reference_quality.json",
        "audits/roster_completeness.json",
        "audits/sensitivity_scope.json",
        "audits/source_quality.json",
    ):
        audit = json.loads((release_root / relative).read_text(encoding="utf-8"))
        if audit.get("status") != "passed":
            raise BuildError(f"release audit is not passed: {relative}")
    return {
        "status": "valid",
        "release_manifest_sha256": sha256_file(manifest_path),
        "release_manifest_payload_sha256": payload_sha256,
        "meeting_count": len(roster),
        "core8_rows": len(core),
        "source_split_counts": dict(Counter(row["source_split"] for row in roster)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    prepare_parser.add_argument("--raw-minutes-root", type=Path, default=DEFAULT_RAW_MINUTES_ROOT)
    prepare_parser.add_argument("--base-roster", type=Path, default=DEFAULT_BASE_ROSTER)
    prepare_parser.add_argument("--base-registry", type=Path, default=DEFAULT_BASE_REGISTRY)
    acquire_parser = subparsers.add_parser("acquire")
    acquire_parser.add_argument("--resume", action="store_true")
    acquire_parser.add_argument("--requests-per-second", type=float, default=2.0)
    acquire_parser.add_argument("--max-workers", type=int, default=2)
    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    subparsers.add_parser("validate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    release_root = args.release_root.expanduser().resolve()
    try:
        if args.command == "prepare":
            result = prepare(
                output_root=output_root,
                split_manifest=args.split_manifest.expanduser().resolve(),
                raw_minutes_root=args.raw_minutes_root.expanduser().resolve(),
                base_roster=args.base_roster.expanduser().resolve(),
                base_registry=args.base_registry.expanduser().resolve(),
            )
        elif args.command == "acquire":
            result = acquire(
                output_root=output_root,
                resume=args.resume,
                requests_per_second=args.requests_per_second,
                max_workers=args.max_workers,
            )
        elif args.command == "publish":
            result = publish(
                output_root=output_root,
                release_root=release_root,
                tokenizer_path=args.tokenizer.expanduser().resolve(),
            )
        else:
            result = validate_release(release_root)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", end="")
        return 0
    except (
        BuildError,
        FileExistsError,
        OSError,
        ValueError,
        requests.RequestException,
    ) as exc:
        print(
            json.dumps(
                {"status": "blocked", "error": str(exc)},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            end="",
            file=os.sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CSV_INVENTORY_SHA256",
    "DERIVED_ROSTER_PAYLOAD_SHA256",
    "EXPECTED_CORE_ROWS",
    "EXPECTED_MEETINGS",
    "EXPECTED_ONE_DAY_MEETINGS",
    "EXPECTED_QA_SPLIT_COUNTS",
    "EXPECTED_SPLIT_COUNTS",
    "EXPECTED_TWO_DAY_MEETINGS",
    "RELEASE_ID",
    "XLSX_INVENTORY_SHA256",
    "_derive_meeting_dates",
    "_inventory_digest",
    "_require_core8_closure",
    "derive_meeting_roster",
    "prepare",
    "publish",
    "validate_release",
]
