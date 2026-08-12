"""Build the immutable v1 TOTALSL unit-only overlay for retrain-v2 chk2.

The source and destination are deliberately fixed.  This builder changes only
``evidence[*].units`` for ``series_id == "TOTALSL"`` from millions to billions,
including the one embedded copy of ``provided_data`` in each affected prompt.
It fails closed on any source, count, schema, or byte-level invariant drift and
publishes with one fsynced ``renameat2(RENAME_NOREPLACE)`` operation.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SOURCE_RELEASE_ID = "analysis_base_full_v7_automated_v3_20260804"
SOURCE_RELEASE_RELATIVE = Path("dataset/processed/retrain_v2") / SOURCE_RELEASE_ID
SOURCE_DATASET_RELATIVE = SOURCE_RELEASE_RELATIVE / "analysis_grpo"
SOURCE_MANIFEST_RELATIVE = SOURCE_RELEASE_RELATIVE / "base_release_manifest.json"

OVERLAY_ID = "analysis_grpo_full_v7_totalsl_billions_v1_20260810"
DESTINATION_RELATIVE = Path("dataset/processed/retrain_v2") / OVERLAY_ID

MANIFEST_FILENAME = "overlay_manifest.json"
ATTESTATION_FILENAME = "overlay_attestation.json"
MANIFEST_SCHEMA_VERSION = "retrain-v2-totalsl-unit-overlay-manifest-v1"
ATTESTATION_SCHEMA_VERSION = "retrain-v2-totalsl-unit-overlay-attestation-v1"

SERIES_ID = "TOTALSL"
SOURCE_UNITS = "Millions of Dollars"
TARGET_UNITS = "Billions of Dollars"
SPLITS = ("train", "eval", "test")
ROW_KEYS = {"meeting_date", "prompt", "provided_data", "sample_id"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "e0dd631097b0a79a7592599161c801b493866e4c48ae59935dd4a1017fbf2c52"
)
EXPECTED_SOURCE_DATASET_ARTIFACT_SHA256 = (
    "21695c76be83658dce57d453d8d470504c1482765c1c6f3bc5e93fc3ac4ffec2"
)
SOURCE_SPLIT_CONTRACT: dict[str, dict[str, Any]] = {
    "train": {
        "manifest_key": "train",
        "rows": 493,
        "affected_rows": 30,
        "evidence_entries": 46,
        "sha256": "da32684c3001d8ac908a99c20dcc5d2397a7351e57f3b9ea15ede63990f2b4bd",
    },
    "eval": {
        "manifest_key": "validation",
        "rows": 199,
        "affected_rows": 12,
        "evidence_entries": 12,
        "sha256": "fdbf39b642714b549da18fd360510b9b050b20e7512fa795c44bd955db51be09",
    },
    "test": {
        "manifest_key": "test",
        "rows": 190,
        "affected_rows": 13,
        "evidence_entries": 13,
        "sha256": "8e0e89fe89d271b6d6862f5059431e5ad12d03646a4d8f1ab5e304c84b939a20",
    },
}
EXPECTED_TOTAL_ROWS = 882
EXPECTED_AFFECTED_ROWS = 55
EXPECTED_EVIDENCE_ENTRIES = 71
EXPECTED_UNCHANGED_ROWS = 827

INVARIANTS = {
    "source_parent_binding_verified": True,
    "exact_split_row_counts": True,
    "exact_affected_row_counts": True,
    "exact_totalsl_evidence_entry_count": True,
    "only_totalsl_units_changed": True,
    "prompt_embedded_provided_data_unique_and_synchronized": True,
    "sample_ids_order_dates_values_evidence_ids_source_sha256_unchanged": True,
    "unaffected_rows_byte_identical": True,
    "source_immutable_and_unchanged_before_publication": True,
}


class TotalslOverlayError(ValueError):
    """Raised before publication when any overlay invariant fails."""


@dataclass(frozen=True)
class SourceSplit:
    name: str
    path: Path
    payload: bytes
    rows: tuple[dict[str, Any], ...]
    lines: tuple[bytes, ...]
    sha256: str


@dataclass(frozen=True)
class SourceSnapshot:
    manifest_payload: bytes
    manifest_sha256: str
    dataset_artifact_sha256: str
    splits: Mapping[str, SourceSplit]


@dataclass(frozen=True)
class TransformedSplit:
    name: str
    payload: bytes
    rows: tuple[dict[str, Any], ...]
    lines: tuple[bytes, ...]
    sha256: str
    affected_rows: int
    evidence_entries: int
    changed_rows: tuple[dict[str, Any], ...]
    row_order_binding_sha256: str
    immutable_fields_binding_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TotalslOverlayError(message)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TotalslOverlayError("payload is not canonical finite JSON") from exc
    return rendered.encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TotalslOverlayError("payload is not finite JSON") from exc
    return (rendered + "\n").encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return _sha_bytes(_canonical_bytes(value))


def _validate_sha256(value: Any, *, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} is not a lowercase SHA-256 digest",
    )
    return value


def _parse_utc(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and value, f"{label} must be a UTC timestamp")
    candidate = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise TotalslOverlayError(f"{label} must be an ISO-8601 timestamp") from exc
    _require(
        parsed.tzinfo is not None and parsed.utcoffset() is not None,
        f"{label} must include a timezone",
    )
    _require(parsed.utcoffset().total_seconds() == 0, f"{label} must be UTC")
    _require(parsed.microsecond == 0, f"{label} must not contain sub-seconds")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _repo_root(value: str | Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(value)))
    _require(lexical.is_dir(), f"repository root is missing: {lexical}")
    _require(not lexical.is_symlink(), f"repository root must not be a symlink: {lexical}")
    return lexical.resolve()


def _fixed_path(repo_root: Path, relative: Path, *, label: str) -> Path:
    _require(not relative.is_absolute() and ".." not in relative.parts, f"unsafe {label}")
    current = repo_root
    for part in relative.parts:
        current /= part
        if current.exists() or current.is_symlink():
            _require(not current.is_symlink(), f"{label} contains symlink: {current}")
    resolved = current.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise TotalslOverlayError(f"{label} escapes repository: {resolved}") from exc
    return resolved


def _require_readonly(path: Path, *, label: str) -> None:
    _require(path.exists() and not path.is_symlink(), f"{label} is missing: {path}")
    writable = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    _require(path.stat().st_mode & writable == 0, f"{label} is writable: {path}")


def _read_file(path: Path, *, label: str) -> bytes:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise TotalslOverlayError(f"unable to read {label}: {path}") from exc


def _load_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TotalslOverlayError(f"{label} is not strict UTF-8 JSON") from exc
    _require(isinstance(value, dict), f"{label} must contain an object")
    return value


def _directory_artifact_sha256(root: Path) -> str:
    _require(root.is_dir() and not root.is_symlink(), f"dataset root is invalid: {root}")
    records: list[bytes] = []
    observed_files: set[str] = set()
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), f"dataset contains symlink: {path}")
        if path.is_dir():
            continue
        _require(path.is_file(), f"dataset contains non-regular entry: {path}")
        relative = path.relative_to(root).as_posix()
        observed_files.add(relative)
        payload = _read_file(path, label=f"dataset artifact {relative}")
        records.append(
            f"{relative}\0{len(payload)}\0{_sha_bytes(payload)}\n".encode("utf-8")
        )
    _require(
        observed_files == {f"{split}.jsonl" for split in SPLITS},
        "source dataset file roster is not exactly the three bound splits",
    )
    return _sha_bytes(b"".join(records))


def _parse_row(line: bytes, *, label: str) -> dict[str, Any]:
    _require(line.endswith(b"\n"), f"{label} lacks its terminal newline")
    _require(b"\r" not in line and b"\x00" not in line, f"{label} has invalid bytes")
    try:
        text = line[:-1].decode("utf-8", errors="strict")
        row = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TotalslOverlayError(f"{label} is not strict UTF-8 JSON") from exc
    _require(isinstance(row, dict) and set(row) == ROW_KEYS, f"{label} schema mismatch")
    _require(_canonical_bytes(row) + b"\n" == line, f"{label} is not canonical JSONL")
    for key in ROW_KEYS:
        _require(
            isinstance(row[key], str) and row[key],
            f"{label}.{key} must be non-empty text",
        )
    try:
        parsed_date = date.fromisoformat(row["meeting_date"])
    except ValueError as exc:
        raise TotalslOverlayError(f"{label}.meeting_date is invalid") from exc
    _require(
        parsed_date.isoformat() == row["meeting_date"],
        f"{label}.meeting_date is not canonical",
    )
    return row


def _parse_fact_card(text: str, *, label: str) -> dict[str, Any]:
    try:
        card = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TotalslOverlayError(f"{label} is not JSON") from exc
    _require(isinstance(card, dict), f"{label} must contain an object")
    _require(
        _canonical_bytes(card).decode("utf-8") == text,
        f"{label} is not canonical JSON",
    )
    evidence = card.get("evidence")
    _require(isinstance(evidence, list), f"{label}.evidence must be a list")
    for index, item in enumerate(evidence):
        item_label = f"{label}.evidence[{index}]"
        _require(isinstance(item, dict), f"{item_label} must be an object")
        for key in (
            "series_id",
            "units",
            "observation_date",
            "value",
            "evidence_id",
            "source_sha256",
        ):
            _require(isinstance(item.get(key), str), f"{item_label}.{key} must be text")
        _validate_sha256(item["source_sha256"], label=f"{item_label}.source_sha256")
    return card


def _identity_projection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    projection: list[dict[str, Any]] = []
    for row in rows:
        card = json.loads(str(row["provided_data"]))
        projection.append(
            {
                "sample_id": row["sample_id"],
                "meeting_date": row["meeting_date"],
                "evidence": [
                    {
                        key: evidence[key]
                        for key in (
                            "series_id",
                            "observation_date",
                            "value",
                            "evidence_id",
                            "source_sha256",
                        )
                    }
                    for evidence in card["evidence"]
                ],
            }
        )
    return {"rows": projection}


def _row_order_binding(rows: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_sha256([row["sample_id"] for row in rows])


def _immutable_fields_binding(rows: Sequence[Mapping[str, Any]]) -> str:
    return _canonical_sha256(_identity_projection(rows))


def _assert_allowed_delta(
    source: Mapping[str, Any],
    output: Mapping[str, Any],
    *,
    target_indexes: Sequence[int],
    label: str,
) -> None:
    _require(set(source) == set(output) == ROW_KEYS, f"{label} outer keys changed")
    for key in ("sample_id", "meeting_date"):
        _require(source[key] == output[key], f"{label}.{key} changed")

    source_data = str(source["provided_data"])
    output_data = str(output["provided_data"])
    source_card = _parse_fact_card(source_data, label=f"{label}.source.provided_data")
    output_card = _parse_fact_card(output_data, label=f"{label}.output.provided_data")
    _require(set(source_card) == set(output_card), f"{label} fact-card keys changed")
    _require(
        len(source_card["evidence"]) == len(output_card["evidence"]),
        f"{label} evidence length changed",
    )
    target_set = set(target_indexes)
    for index, (before, after) in enumerate(
        zip(source_card["evidence"], output_card["evidence"], strict=True)
    ):
        if index in target_set:
            expected = dict(before)
            expected["units"] = TARGET_UNITS
            _require(after == expected, f"{label} changed non-unit TOTALSL fields")
        else:
            _require(after == before, f"{label} changed non-TOTALSL evidence")

    source_prompt = str(source["prompt"])
    output_prompt = str(output["prompt"])
    _require(
        source_prompt.count(source_data) == 1,
        f"{label}.prompt must contain provided_data exactly once",
    )
    start = source_prompt.index(source_data)
    expected_prompt = source_prompt[:start] + output_data + source_prompt[start + len(source_data) :]
    _require(output_prompt == expected_prompt, f"{label}.prompt changed outside provided_data")
    _require(
        output_prompt.count(output_data) == 1,
        f"{label}.output prompt must contain provided_data exactly once",
    )


def _transform_split(source: SourceSplit) -> TransformedSplit:
    output_rows: list[dict[str, Any]] = []
    output_lines: list[bytes] = []
    changed_rows: list[dict[str, Any]] = []
    evidence_entries = 0
    for row_index, (source_row, source_line) in enumerate(
        zip(source.rows, source.lines, strict=True)
    ):
        label = f"{source.name}[{row_index}]"
        source_data = source_row["provided_data"]
        _require(
            source_row["prompt"].count(source_data) == 1,
            f"{label}.prompt must contain provided_data exactly once",
        )
        card = _parse_fact_card(source_data, label=f"{label}.provided_data")
        target_indexes = [
            index
            for index, evidence in enumerate(card["evidence"])
            if evidence["series_id"] == SERIES_ID
        ]
        for index in target_indexes:
            _require(
                card["evidence"][index]["units"] == SOURCE_UNITS,
                f"{label}.evidence[{index}] TOTALSL source units are not bound millions",
            )

        if not target_indexes:
            output_row = deepcopy(source_row)
            output_line = source_line
        else:
            output_card = deepcopy(card)
            for index in target_indexes:
                output_card["evidence"][index]["units"] = TARGET_UNITS
            output_data = _canonical_bytes(output_card).decode("utf-8")
            output_row = deepcopy(source_row)
            output_row["provided_data"] = output_data
            output_row["prompt"] = source_row["prompt"].replace(
                source_data,
                output_data,
                1,
            )
            output_line = _canonical_bytes(output_row) + b"\n"

        _assert_allowed_delta(
            source_row,
            output_row,
            target_indexes=target_indexes,
            label=label,
        )
        _require(len(output_line) == len(source_line), f"{label} byte length changed")
        if target_indexes:
            _require(output_line != source_line, f"{label} affected row did not change")
            field_changes = []
            for index in target_indexes:
                evidence = card["evidence"][index]
                field_changes.append(
                    {
                        "evidence_id": evidence["evidence_id"],
                        "evidence_index": index,
                        "from": SOURCE_UNITS,
                        "prompt_mirror_synchronized": True,
                        "provided_data_path": f"evidence[{index}].units",
                        "to": TARGET_UNITS,
                    }
                )
            changed_rows.append(
                {
                    "field_changes": field_changes,
                    "output_bytes": len(output_line),
                    "output_row_sha256": _sha_bytes(output_line),
                    "row_index": row_index,
                    "sample_id": source_row["sample_id"],
                    "source_bytes": len(source_line),
                    "source_row_sha256": _sha_bytes(source_line),
                    "split": source.name,
                    "totalsl_evidence_entries": len(target_indexes),
                }
            )
        else:
            _require(output_line == source_line, f"{label} unaffected bytes changed")
        evidence_entries += len(target_indexes)
        output_rows.append(output_row)
        output_lines.append(output_line)

    _require(
        _row_order_binding(source.rows) == _row_order_binding(output_rows),
        f"{source.name} sample order changed",
    )
    _require(
        _immutable_fields_binding(source.rows)
        == _immutable_fields_binding(output_rows),
        f"{source.name} immutable evidence fields changed",
    )
    payload = b"".join(output_lines)
    return TransformedSplit(
        name=source.name,
        payload=payload,
        rows=tuple(output_rows),
        lines=tuple(output_lines),
        sha256=_sha_bytes(payload),
        affected_rows=len(changed_rows),
        evidence_entries=evidence_entries,
        changed_rows=tuple(changed_rows),
        row_order_binding_sha256=_row_order_binding(output_rows),
        immutable_fields_binding_sha256=_immutable_fields_binding(output_rows),
    )


def _load_source_split(path: Path, *, split: str) -> SourceSplit:
    contract = SOURCE_SPLIT_CONTRACT[split]
    _require_readonly(path, label=f"source {split}")
    payload = _read_file(path, label=f"source {split}")
    observed_sha = _sha_bytes(payload)
    _require(observed_sha == contract["sha256"], f"source {split} SHA-256 mismatch")
    lines = tuple(payload.splitlines(keepends=True))
    _require(b"".join(lines) == payload, f"source {split} line framing changed")
    _require(len(lines) == contract["rows"], f"source {split} row count mismatch")
    rows = tuple(
        _parse_row(line, label=f"source {split}[{index}]")
        for index, line in enumerate(lines)
    )
    return SourceSplit(
        name=split,
        path=path,
        payload=payload,
        rows=rows,
        lines=lines,
        sha256=observed_sha,
    )


def _load_source_snapshot(repo_root: Path) -> SourceSnapshot:
    source_release = _fixed_path(
        repo_root,
        SOURCE_RELEASE_RELATIVE,
        label="source release",
    )
    source_dataset = _fixed_path(
        repo_root,
        SOURCE_DATASET_RELATIVE,
        label="source dataset",
    )
    manifest_path = _fixed_path(
        repo_root,
        SOURCE_MANIFEST_RELATIVE,
        label="source manifest",
    )
    for path, label in (
        (source_release, "source release"),
        (source_dataset, "source dataset"),
        (manifest_path, "source manifest"),
    ):
        _require_readonly(path, label=label)

    manifest_payload = _read_file(manifest_path, label="source manifest")
    manifest_sha = _sha_bytes(manifest_payload)
    _require(
        manifest_sha == EXPECTED_SOURCE_MANIFEST_SHA256,
        "source base_release_manifest SHA-256 mismatch",
    )
    manifest = _load_json_object(manifest_payload, label="source manifest")
    _require(manifest.get("schema_version") == 1, "source manifest schema mismatch")
    _require(manifest.get("release_type") == "analysis_base", "wrong source type")
    _require(manifest.get("release_id") == SOURCE_RELEASE_ID, "wrong source release")
    datasets = manifest.get("datasets")
    _require(isinstance(datasets, dict), "source datasets record is missing")
    dataset_record = datasets.get("analysis_grpo")
    _require(isinstance(dataset_record, dict), "source analysis_grpo record is missing")
    _require(dataset_record.get("path") == "analysis_grpo", "source dataset path drift")
    _require(
        dataset_record.get("artifact_sha256")
        == EXPECTED_SOURCE_DATASET_ARTIFACT_SHA256,
        "source dataset artifact binding mismatch",
    )
    split_records = dataset_record.get("split_files")
    _require(isinstance(split_records, dict), "source split records are missing")

    splits: dict[str, SourceSplit] = {}
    all_sample_ids: set[str] = set()
    for split in SPLITS:
        contract = SOURCE_SPLIT_CONTRACT[split]
        manifest_key = contract["manifest_key"]
        record = split_records.get(manifest_key)
        _require(isinstance(record, dict), f"source manifest {split} record is missing")
        _require(
            record.get("path") == f"analysis_grpo/{split}.jsonl",
            f"source manifest {split} path mismatch",
        )
        _require(
            record.get("sha256") == contract["sha256"],
            f"source manifest {split} hash mismatch",
        )
        source_split = _load_source_split(source_dataset / f"{split}.jsonl", split=split)
        for row in source_split.rows:
            sample_id = row["sample_id"]
            _require(sample_id not in all_sample_ids, f"duplicate source sample_id: {sample_id}")
            all_sample_ids.add(sample_id)
        splits[split] = source_split

    artifact_sha = _directory_artifact_sha256(source_dataset)
    _require(
        artifact_sha == EXPECTED_SOURCE_DATASET_ARTIFACT_SHA256,
        "source analysis_grpo directory fingerprint mismatch",
    )
    return SourceSnapshot(
        manifest_payload=manifest_payload,
        manifest_sha256=manifest_sha,
        dataset_artifact_sha256=artifact_sha,
        splits=splits,
    )


def _transform_snapshot(
    source: SourceSnapshot,
) -> dict[str, TransformedSplit]:
    _require(set(SOURCE_SPLIT_CONTRACT) == set(SPLITS), "split contract roster mismatch")
    _require(
        sum(int(SOURCE_SPLIT_CONTRACT[split]["rows"]) for split in SPLITS)
        == EXPECTED_TOTAL_ROWS,
        "bound total row contract is internally inconsistent",
    )
    _require(
        sum(int(SOURCE_SPLIT_CONTRACT[split]["affected_rows"]) for split in SPLITS)
        == EXPECTED_AFFECTED_ROWS,
        "bound affected-row contract is internally inconsistent",
    )
    _require(
        sum(int(SOURCE_SPLIT_CONTRACT[split]["evidence_entries"]) for split in SPLITS)
        == EXPECTED_EVIDENCE_ENTRIES,
        "bound evidence-entry contract is internally inconsistent",
    )
    _require(
        EXPECTED_TOTAL_ROWS - EXPECTED_AFFECTED_ROWS == EXPECTED_UNCHANGED_ROWS,
        "bound unchanged-row contract is internally inconsistent",
    )

    transformed = {
        split: _transform_split(source.splits[split]) for split in SPLITS
    }
    for split in SPLITS:
        contract = SOURCE_SPLIT_CONTRACT[split]
        output = transformed[split]
        _require(
            len(output.rows) == contract["rows"],
            f"{split} output row count mismatch",
        )
        _require(
            output.affected_rows == contract["affected_rows"],
            f"{split} affected row count mismatch",
        )
        _require(
            output.evidence_entries == contract["evidence_entries"],
            f"{split} TOTALSL evidence entry count mismatch",
        )
    _require(
        sum(item.affected_rows for item in transformed.values())
        == EXPECTED_AFFECTED_ROWS,
        "global affected row count mismatch",
    )
    _require(
        sum(item.evidence_entries for item in transformed.values())
        == EXPECTED_EVIDENCE_ENTRIES,
        "global TOTALSL evidence entry count mismatch",
    )
    return transformed


def _split_descriptor(
    source: SourceSnapshot,
    transformed: Mapping[str, TransformedSplit],
    split: str,
) -> dict[str, Any]:
    original = source.splits[split]
    output = transformed[split]
    return {
        "affected_rows": output.affected_rows,
        "evidence_entries_changed": output.evidence_entries,
        "output": {
            "bytes": len(output.payload),
            "path": f"{split}.jsonl",
            "sha256": output.sha256,
        },
        "rows": len(output.rows),
        "source": {
            "bytes": len(original.payload),
            "path": (SOURCE_DATASET_RELATIVE / f"{split}.jsonl").as_posix(),
            "sha256": original.sha256,
        },
        "unchanged_rows": len(output.rows) - output.affected_rows,
    }


def _output_payload_sha256(
    transformed: Mapping[str, TransformedSplit],
) -> str:
    payload = {
        split: {
            "bytes": len(transformed[split].payload),
            "rows": len(transformed[split].rows),
            "sha256": transformed[split].sha256,
        }
        for split in SPLITS
    }
    return _canonical_sha256(payload)


def _build_manifest(
    source: SourceSnapshot,
    transformed: Mapping[str, TransformedSplit],
    *,
    generated_at_utc: str,
    builder_sha256: str,
    builder_bytes: int,
) -> dict[str, Any]:
    return {
        "attestation": {
            "path": ATTESTATION_FILENAME,
            "schema_version": ATTESTATION_SCHEMA_VERSION,
        },
        "builder": {
            "bytes": builder_bytes,
            "path": "jobs/retrain_v2/build_totalsl_unit_overlay.py",
            "sha256": builder_sha256,
        },
        "counts": {
            "affected_rows": EXPECTED_AFFECTED_ROWS,
            "evidence_entries_changed": EXPECTED_EVIDENCE_ENTRIES,
            "rows": EXPECTED_TOTAL_ROWS,
            "unchanged_rows": EXPECTED_UNCHANGED_ROWS,
        },
        "generated_at_utc": generated_at_utc,
        "invariants": dict(INVARIANTS),
        "output": {
            "path": DESTINATION_RELATIVE.as_posix(),
            "payload_sha256": _output_payload_sha256(transformed),
            "splits": {
                split: _split_descriptor(source, transformed, split)
                for split in SPLITS
            },
        },
        "overlay_id": OVERLAY_ID,
        "overlay_type": "totalsl_unit_only",
        "publication_policy": {
            "atomic": "renameat2(RENAME_NOREPLACE)",
            "destination_must_not_exist": True,
            "immutable_after_publish": True,
            "stale_staging_policy": "fail_closed",
        },
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": {
            "base_release_manifest": {
                "path": SOURCE_MANIFEST_RELATIVE.as_posix(),
                "sha256": source.manifest_sha256,
            },
            "dataset_artifact_sha256": source.dataset_artifact_sha256,
            "path": SOURCE_DATASET_RELATIVE.as_posix(),
            "release_id": SOURCE_RELEASE_ID,
        },
        "transformation": {
            "field": "evidence[*].units",
            "from": SOURCE_UNITS,
            "prompt_projection": "replace the unique embedded provided_data copy",
            "series_id": SERIES_ID,
            "to": TARGET_UNITS,
        },
    }


def _build_attestation(
    source: SourceSnapshot,
    transformed: Mapping[str, TransformedSplit],
    *,
    generated_at_utc: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    split_attestations = {}
    changed_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        original = source.splits[split]
        output = transformed[split]
        source_order = _row_order_binding(original.rows)
        source_immutable = _immutable_fields_binding(original.rows)
        _require(source_order == output.row_order_binding_sha256, "row-order binding drift")
        _require(
            source_immutable == output.immutable_fields_binding_sha256,
            "immutable-field binding drift",
        )
        split_attestations[split] = {
            "affected_rows": output.affected_rows,
            "evidence_entries_changed": output.evidence_entries,
            "immutable_fields_binding_sha256": source_immutable,
            "output_sha256": output.sha256,
            "row_order_binding_sha256": source_order,
            "rows": len(output.rows),
            "source_sha256": original.sha256,
            "unchanged_rows": len(output.rows) - output.affected_rows,
        }
        changed_rows.extend(output.changed_rows)

    payload: dict[str, Any] = {
        "attestation_type": "retrain_v2_totalsl_unit_overlay",
        "changed_rows": changed_rows,
        "changed_sample_ids_sha256": _canonical_sha256(
            [item["sample_id"] for item in changed_rows]
        ),
        "generated_at_utc": generated_at_utc,
        "invariants": dict(INVARIANTS),
        "manifest": {"path": MANIFEST_FILENAME, "sha256": manifest_sha256},
        "output_payload_sha256": _output_payload_sha256(transformed),
        "overlay_id": OVERLAY_ID,
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "source_dataset_artifact_sha256": source.dataset_artifact_sha256,
        "splits": split_attestations,
    }
    payload["attestation_payload_sha256"] = _canonical_sha256(payload)
    return payload


def _write_fsynced(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _tree_has_no_symlinks(root: Path, *, label: str) -> None:
    _require(root.is_dir() and not root.is_symlink(), f"{label} is not a real directory")
    for path in root.rglob("*"):
        _require(not path.is_symlink(), f"{label} contains symlink: {path}")


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)


def _fsync_tree(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directories = sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for path in (*directories, root):
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(
        renameat2 is not None,
        "atomic renameat2(RENAME_NOREPLACE) is unavailable",
    )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise TotalslOverlayError(f"immutable destination already exists: {destination}")
    raise TotalslOverlayError(
        f"atomic no-overwrite publication failed: {destination}: {os.strerror(error)}"
    )


def _cleanup_staging(staging: Path) -> None:
    if not staging.exists() and not staging.is_symlink():
        return
    _require(staging.is_dir() and not staging.is_symlink(), "unsafe staging cleanup")
    _tree_has_no_symlinks(staging, label="owned staging")
    for path in staging.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    staging.chmod(0o700)
    shutil.rmtree(staging)


def _verify_source_unchanged(repo_root: Path, source: SourceSnapshot) -> None:
    manifest_path = _fixed_path(
        repo_root,
        SOURCE_MANIFEST_RELATIVE,
        label="source manifest",
    )
    _require(
        _read_file(manifest_path, label="source manifest") == source.manifest_payload,
        "source manifest changed during overlay build",
    )
    dataset_root = _fixed_path(
        repo_root,
        SOURCE_DATASET_RELATIVE,
        label="source dataset",
    )
    for split in SPLITS:
        path = dataset_root / f"{split}.jsonl"
        _require_readonly(path, label=f"source {split}")
        _require(
            _sha_bytes(_read_file(path, label=f"source {split}"))
            == source.splits[split].sha256,
            f"source {split} changed during overlay build",
        )
    _require(
        _directory_artifact_sha256(dataset_root) == source.dataset_artifact_sha256,
        "source dataset changed during overlay build",
    )


def _validate_overlay_tree(
    overlay_root: Path,
    source: SourceSnapshot,
    transformed: Mapping[str, TransformedSplit],
    *,
    require_immutable: bool,
) -> dict[str, Any]:
    _tree_has_no_symlinks(overlay_root, label="overlay")
    files = {path.relative_to(overlay_root).as_posix() for path in overlay_root.rglob("*") if path.is_file()}
    _require(
        files
        == {
            *(f"{split}.jsonl" for split in SPLITS),
            MANIFEST_FILENAME,
            ATTESTATION_FILENAME,
        },
        "overlay file roster mismatch",
    )
    if require_immutable:
        writable = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
        for path in (overlay_root, *overlay_root.rglob("*")):
            _require(path.stat().st_mode & writable == 0, f"overlay is writable: {path}")

    for split in SPLITS:
        observed = _read_file(overlay_root / f"{split}.jsonl", label=f"output {split}")
        _require(observed == transformed[split].payload, f"output {split} bytes mismatch")

    manifest_payload = _read_file(
        overlay_root / MANIFEST_FILENAME,
        label="overlay manifest",
    )
    manifest = _load_json_object(manifest_payload, label="overlay manifest")
    generated = _parse_utc(manifest.get("generated_at_utc"), label="generated_at_utc")
    builder = manifest.get("builder")
    _require(isinstance(builder, dict), "overlay builder record is missing")
    _require(
        builder.get("path") == "jobs/retrain_v2/build_totalsl_unit_overlay.py",
        "overlay builder path mismatch",
    )
    builder_sha = _validate_sha256(builder.get("sha256"), label="builder.sha256")
    builder_bytes = builder.get("bytes")
    _require(
        isinstance(builder_bytes, int) and not isinstance(builder_bytes, bool) and builder_bytes > 0,
        "builder.bytes is invalid",
    )
    expected_manifest = _build_manifest(
        source,
        transformed,
        generated_at_utc=generated,
        builder_sha256=builder_sha,
        builder_bytes=builder_bytes,
    )
    _require(manifest == expected_manifest, "overlay manifest contract mismatch")
    _require(
        manifest_payload == _json_bytes(expected_manifest),
        "overlay manifest is not canonical formatted JSON",
    )

    attestation_payload = _read_file(
        overlay_root / ATTESTATION_FILENAME,
        label="overlay attestation",
    )
    attestation = _load_json_object(attestation_payload, label="overlay attestation")
    expected_attestation = _build_attestation(
        source,
        transformed,
        generated_at_utc=generated,
        manifest_sha256=_sha_bytes(manifest_payload),
    )
    _require(attestation == expected_attestation, "overlay attestation contract mismatch")
    _require(
        attestation_payload == _json_bytes(expected_attestation),
        "overlay attestation is not canonical formatted JSON",
    )
    claimed_payload_sha = attestation["attestation_payload_sha256"]
    unsigned_attestation = dict(attestation)
    del unsigned_attestation["attestation_payload_sha256"]
    _require(
        claimed_payload_sha == _canonical_sha256(unsigned_attestation),
        "overlay attestation payload hash mismatch",
    )
    return {
        "attestation_sha256": _sha_bytes(attestation_payload),
        "manifest_sha256": _sha_bytes(manifest_payload),
        "output_payload_sha256": _output_payload_sha256(transformed),
        "status": "valid",
    }


def validate_totalsl_unit_overlay(repo_root: str | Path = ".") -> dict[str, Any]:
    """Revalidate the fixed published overlay against its immutable parent."""

    root = _repo_root(repo_root)
    source = _load_source_snapshot(root)
    transformed = _transform_snapshot(source)
    destination = _fixed_path(root, DESTINATION_RELATIVE, label="overlay destination")
    _require(destination.is_dir(), f"overlay destination is missing: {destination}")
    return _validate_overlay_tree(
        destination,
        source,
        transformed,
        require_immutable=True,
    )


def build_totalsl_unit_overlay(
    repo_root: str | Path = ".",
    *,
    generated_at_utc: str | None = None,
) -> Path:
    """Validate, stage, seal, and atomically publish the fixed v1 overlay."""

    root = _repo_root(repo_root)
    destination_parent = _fixed_path(
        root,
        DESTINATION_RELATIVE.parent,
        label="overlay parent",
    )
    _require(
        destination_parent.is_dir(),
        f"overlay parent is missing: {destination_parent}",
    )
    destination = _fixed_path(root, DESTINATION_RELATIVE, label="overlay destination")
    _require(
        not destination.exists() and not destination.is_symlink(),
        f"immutable destination already exists: {destination}",
    )
    stale = sorted(destination_parent.glob(f".{OVERLAY_ID}.build-*"))
    _require(not stale, "stale TOTALSL overlay staging exists; inspect it manually")

    source = _load_source_snapshot(root)
    transformed = _transform_snapshot(source)
    generated = _parse_utc(
        generated_at_utc or _utc_now(),
        label="generated_at_utc",
    )
    builder_path = Path(__file__).resolve()
    builder_payload = builder_path.read_bytes()
    builder_sha = _sha_bytes(builder_payload)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{OVERLAY_ID}.build-",
            dir=destination_parent,
        )
    )
    try:
        for split in SPLITS:
            _write_fsynced(staging / f"{split}.jsonl", transformed[split].payload)
        manifest = _build_manifest(
            source,
            transformed,
            generated_at_utc=generated,
            builder_sha256=builder_sha,
            builder_bytes=len(builder_payload),
        )
        manifest_payload = _json_bytes(manifest)
        _write_fsynced(staging / MANIFEST_FILENAME, manifest_payload)
        attestation = _build_attestation(
            source,
            transformed,
            generated_at_utc=generated,
            manifest_sha256=_sha_bytes(manifest_payload),
        )
        _write_fsynced(staging / ATTESTATION_FILENAME, _json_bytes(attestation))

        _validate_overlay_tree(
            staging,
            source,
            transformed,
            require_immutable=False,
        )
        _verify_source_unchanged(root, source)
        _require(
            builder_path.read_bytes() == builder_payload,
            "overlay builder changed during build",
        )
        _require(
            not destination.exists() and not destination.is_symlink(),
            f"immutable destination already exists: {destination}",
        )
        _seal_tree(staging)
        _fsync_tree(staging)
        _rename_noreplace(staging, destination)
        parent_descriptor = os.open(
            destination_parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return destination
    except BaseException:
        _cleanup_staging(staging)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--generated-at-utc")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="revalidate the already-published fixed overlay without writing",
    )
    args = parser.parse_args(argv)
    if args.verify_only:
        result = validate_totalsl_unit_overlay(args.repo_root)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    destination = build_totalsl_unit_overlay(
        args.repo_root,
        generated_at_utc=args.generated_at_utc,
    )
    print(
        json.dumps(
            {"path": str(destination), "status": "published"},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
