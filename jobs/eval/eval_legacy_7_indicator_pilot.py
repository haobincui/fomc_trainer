"""Re-score the historical ft_20250330 seven-indicator pilot.

This command consumes the surviving Excel workbooks directly and computes

    delta = cos(full_output, target) - cos(masked_output, target).

It is intentionally separate from the canonical Chapter 2 evaluator because
the historical prompt intervention is not an exact one-block deletion.  The
result is therefore a legacy prompt-perturbation reanalysis, not a canonical
LOO run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import math
import os
import platform
import re
import sys
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from open_r1.provenance import fingerprint_artifact_path, sha256_file, sha256_text
from open_r1.validator.leave_one_out import leave_one_out_metrics_from_similarities


SCHEMA_VERSION = "legacy-7-indicator-pilot-rescore-v1"
EMBEDDING_CACHE_SCHEMA_VERSION = "legacy-mean-pooling-cache-v1"
EMBEDDING_IMPLEMENTATION_VERSION = (
    "automodel-last-hidden-state-attention-mask-mean-float32-l2-v1"
)
RUN_ID = "ft_20250330"
EXCEL_CELL_CHARACTER_LIMIT = 32_767
INDICATORS = (
    "fed_rate",
    "gdp",
    "inflation_rate",
    "sp500",
    "unemployment_rate",
    "us_t10",
    "yield_curve",
)
BATCHES = tuple(range(1, 20))
EXPECTED_ROWS_PER_WORKBOOK = 20

KNOWN_TRAINING_QA_SHA256 = (
    "3c2b1a94c31c4328328725764e8471a49b062af7ea67d67756ebe5bee7249c3a"
)
KNOWN_TAGGED_QA_SHA256 = (
    "e204d7dad881364545d8b6cd44a86d4429cb44404cff23d3bc8823931c2c0c61"
)
KNOWN_METADATA_BINDING_SHA256 = (
    "8a21b3751434198581c92f056e80dc8e14647afd6ef84378a7f72a64c1380856"
)
KNOWN_FULL_WORKBOOK_INVENTORY_SHA256 = (
    "0bf74f6c76ff9d29eb48ebf98a1884fa7387e65d21916ac46e6536c614b8bd14"
)
KNOWN_MASK_WORKBOOK_INVENTORY_SHA256 = (
    "4dd5dff569381e356b8a769ad694d919234aa35c1991c70f6d707c56c1bb4a5b"
)
KNOWN_EXCLUSIONS = {
    ("fed_rate", 6, 128),
    ("gdp", 6, 21),
    ("gdp", 11, 5),
    ("unemployment_rate", 7, 81),
    ("us_t10", 6, 303),
}

_MASK_FILENAME = re.compile(
    r"^mask_(?P<indicator>fed_rate|gdp|inflation_rate|sp500|"
    r"unemployment_rate|us_t10|yield_curve)_generated_(?P<batch>\d+)\.xlsx$"
)
_MEETING_FOLDER = re.compile(r"^fomcminutes(?P<date>\d{8})_cleaned_sections$")


class TruncatedLegacyResponseError(ValueError):
    """Raised when a response is truncated at Excel's cell limit."""


def _require_columns(
    frame: pd.DataFrame, columns: Iterable[str], *, path: Path
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def _coerce_index(value: Any, *, path: Path, row_number: int) -> int:
    if isinstance(value, bool) or pd.isna(value):
        raise ValueError(f"Invalid QA index in {path}, row {row_number}: {value!r}")
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid QA index in {path}, row {row_number}: {value!r}"
        ) from exc
    if float(value) != float(integer):
        raise ValueError(
            f"Non-integral QA index in {path}, row {row_number}: {value!r}"
        )
    return integer


def parse_legacy_assistant_content(value: Any, *, source: str) -> str:
    """Parse a historical Python-repr assistant message without altering content."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} is not a non-empty string")
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        if len(value) == EXCEL_CELL_CHARACTER_LIMIT:
            raise TruncatedLegacyResponseError(
                f"{source} is truncated at {EXCEL_CELL_CHARACTER_LIMIT} characters"
            ) from exc
        raise ValueError(f"{source} is not a valid Python literal: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{source} must decode to a dictionary")
    if parsed.get("role") != "assistant":
        raise ValueError(f"{source} has unexpected role={parsed.get('role')!r}")
    content = parsed.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"{source} has no non-blank assistant content")
    return content


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read JSON list from {path}: {exc}") from exc
    if not isinstance(payload, list) or not all(
        isinstance(row, dict) for row in payload
    ):
        raise ValueError(f"{path} must contain a JSON list of objects")
    return payload


def _inventory_fingerprint(paths: Sequence[Path]) -> dict[str, Any]:
    """Hash sorted ``filename<TAB>file_sha256`` records for historical inputs."""

    if not paths:
        raise ValueError("Cannot fingerprint an empty file inventory")
    digest = hashlib.sha256()
    total_bytes = 0
    rows: list[dict[str, Any]] = []
    for path in sorted(paths, key=lambda candidate: candidate.name):
        file_digest = sha256_file(path)
        size = path.stat().st_size
        total_bytes += size
        digest.update(f"{path.name}\t{file_digest}\n".encode("utf-8"))
        rows.append({"filename": path.name, "sha256": file_digest, "bytes": size})
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(paths),
        "total_bytes": total_bytes,
        "algorithm": "sha256(sorted UTF-8 lines '<filename>\\t<file_sha256>\\n')",
        "files": rows,
    }


def _validate_known_hash(
    observed: str,
    expected: str,
    *,
    label: str,
    strict: bool,
) -> None:
    if observed == expected:
        return
    message = f"{label} SHA-256 mismatch: expected {expected}, observed {observed}"
    if strict:
        raise ValueError(message)
    logging.warning("%s; continuing because source hash checks were relaxed", message)


def _meeting_metadata(
    tagged_rows: list[dict[str, Any]],
    sections_root: Path,
) -> dict[int, dict[str, str]]:
    section_files = sorted(sections_root.glob("*/*.txt"))
    if not section_files:
        raise FileNotFoundError(f"No section text files found under {sections_root}")

    paths_by_text: dict[str, list[Path]] = {}
    for path in section_files:
        paths_by_text.setdefault(path.read_text(encoding="utf-8"), []).append(path)

    metadata: dict[int, dict[str, str]] = {}
    for qa_index, row in enumerate(tagged_rows):
        tagged_target = row.get("output")
        section_name = str(row.get("section_tag") or "").strip()
        if not isinstance(tagged_target, str) or not tagged_target:
            raise ValueError(f"Tagged QA row {qa_index} has no output text")
        if not section_name:
            raise ValueError(f"Tagged QA row {qa_index} has no section_tag")

        matches = paths_by_text.get(tagged_target, [])
        if len(matches) != 1:
            raise ValueError(
                f"Tagged QA row {qa_index} maps to {len(matches)} section files; "
                "expected exactly one"
            )
        section_path = matches[0]
        if section_path.stem != section_name:
            raise ValueError(
                f"Section mismatch for QA row {qa_index}: tag={section_name!r}, "
                f"file={section_path.stem!r}"
            )
        folder_match = _MEETING_FOLDER.fullmatch(section_path.parent.name)
        if folder_match is None:
            raise ValueError(
                f"Cannot parse meeting date from {section_path.parent.name!r}"
            )
        compact_date = folder_match.group("date")
        meeting_date = f"{compact_date[:4]}-{compact_date[4:6]}-{compact_date[6:8]}"
        metadata[qa_index] = {
            "meeting_date": meeting_date,
            "section_name": section_name,
            "section_source_file": str(section_path.resolve()),
        }
    return metadata


def _metadata_binding_fingerprint(
    metadata_by_index: dict[int, dict[str, str]],
    sections_root: Path,
) -> dict[str, Any]:
    """Bind each QA index to its dated section path and source-file content."""

    sections_root = sections_root.resolve()
    digest = hashlib.sha256()
    total_bytes = 0
    unique_files: set[str] = set()
    for qa_index in sorted(metadata_by_index):
        metadata = metadata_by_index[qa_index]
        section_path = Path(metadata["section_source_file"]).resolve()
        try:
            relative_path = section_path.relative_to(sections_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Section source is outside metadata root: {section_path}"
            ) from exc
        payload = {
            "qa_index": qa_index,
            "meeting_date": metadata["meeting_date"],
            "section_name": metadata["section_name"],
            "section_relative_path": relative_path,
            "section_file_sha256": sha256_file(section_path),
        }
        digest.update(
            (
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
        )
        if relative_path not in unique_files:
            total_bytes += section_path.stat().st_size
            unique_files.add(relative_path)

    return {
        "sha256": digest.hexdigest(),
        "row_count": len(metadata_by_index),
        "unique_section_files": len(unique_files),
        "total_unique_file_bytes": total_bytes,
        "sections_root": str(sections_root),
        "algorithm": (
            "sha256(sorted QA-index JSONL containing meeting_date, section_name, "
            "section_relative_path, and section_file_sha256)"
        ),
    }


def load_legacy_pilot_rows(
    source_root: Path,
    *,
    indicators: Sequence[str] = INDICATORS,
    batches: Sequence[int] = BATCHES,
    expected_rows_per_workbook: int = EXPECTED_ROWS_PER_WORKBOOK,
    strict_known_hashes: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Recover validated triplets and explicit exclusions from legacy workbooks."""

    source_root = source_root.expanduser().resolve()
    training_qa_path = source_root / "data/training/input_qa.json"
    tagged_qa_path = source_root / "output/archive/qa_json/input_qa_tagged.json"
    sections_root = source_root / "data/processed/sections"
    full_root = (
        source_root / "output/archive/generated/section_detail_responses/ft_20250330"
    )
    mask_root = source_root / "output/archive/masked/ft_20250330"

    for required_path in (
        training_qa_path,
        tagged_qa_path,
        sections_root,
        full_root,
        mask_root,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required legacy source is missing: {required_path}"
            )

    training_qa_sha256 = sha256_file(training_qa_path)
    tagged_qa_sha256 = sha256_file(tagged_qa_path)
    if tuple(indicators) == INDICATORS and tuple(batches) == BATCHES:
        _validate_known_hash(
            training_qa_sha256,
            KNOWN_TRAINING_QA_SHA256,
            label="training QA target",
            strict=strict_known_hashes,
        )
        _validate_known_hash(
            tagged_qa_sha256,
            KNOWN_TAGGED_QA_SHA256,
            label="tagged QA metadata",
            strict=strict_known_hashes,
        )

    training_rows = _read_json_list(training_qa_path)
    tagged_rows = _read_json_list(tagged_qa_path)
    if len(training_rows) != len(tagged_rows):
        raise ValueError(
            f"QA source length mismatch: training={len(training_rows)}, "
            f"tagged={len(tagged_rows)}"
        )
    instruction_mismatches = [
        index
        for index, (training_row, tagged_row) in enumerate(
            zip(training_rows, tagged_rows, strict=True)
        )
        if training_row.get("instruction") != tagged_row.get("instruction")
    ]
    if instruction_mismatches:
        raise ValueError(
            "Training and tagged QA rows are not index-aligned; instruction "
            f"mismatches include {instruction_mismatches[:5]}"
        )
    metadata_by_index = _meeting_metadata(tagged_rows, sections_root)
    metadata_binding = _metadata_binding_fingerprint(
        metadata_by_index,
        sections_root,
    )
    if tuple(indicators) == INDICATORS and tuple(batches) == BATCHES:
        _validate_known_hash(
            metadata_binding["sha256"],
            KNOWN_METADATA_BINDING_SHA256,
            label="derived QA-index metadata mapping",
            strict=strict_known_hashes,
        )

    full_paths = [full_root / f"generated_{batch}.xlsx" for batch in batches]
    missing_full_paths = [path for path in full_paths if not path.is_file()]
    if missing_full_paths:
        raise FileNotFoundError(
            "Missing full-generation workbooks: "
            + ", ".join(str(path) for path in missing_full_paths)
        )
    full_inventory = _inventory_fingerprint(full_paths)
    if tuple(indicators) == INDICATORS and tuple(batches) == BATCHES:
        _validate_known_hash(
            full_inventory["sha256"],
            KNOWN_FULL_WORKBOOK_INVENTORY_SHA256,
            label="full workbook inventory",
            strict=strict_known_hashes,
        )

    full_by_occurrence: dict[tuple[int, int], dict[str, Any]] = {}
    target_cell_exact = 0
    target_cell_truncated_prefix = 0
    for batch, path in zip(batches, full_paths, strict=True):
        frame = pd.read_excel(path, engine="openpyxl")
        _require_columns(
            frame,
            ("index", "prompts", "targets", "generateds"),
            path=path,
        )
        if len(frame) != expected_rows_per_workbook:
            raise ValueError(
                f"{path} has {len(frame)} rows; expected {expected_rows_per_workbook}"
            )
        seen_indices: set[int] = set()
        for position, row in frame.iterrows():
            qa_index = _coerce_index(
                row["index"],
                path=path,
                row_number=int(position) + 2,
            )
            if qa_index in seen_indices:
                raise ValueError(f"Duplicate QA index {qa_index} in {path}")
            seen_indices.add(qa_index)
            if not 0 <= qa_index < len(training_rows):
                raise ValueError(f"QA index {qa_index} in {path} is out of range")

            target = training_rows[qa_index].get("output")
            if not isinstance(target, str) or not target.strip():
                raise ValueError(f"Training QA row {qa_index} has no target output")
            workbook_target = row["targets"]
            if not isinstance(workbook_target, str):
                raise ValueError(
                    f"Target cell for QA index {qa_index} in {path} is invalid"
                )
            if workbook_target == target:
                target_cell_exact += 1
            elif len(
                workbook_target
            ) == EXCEL_CELL_CHARACTER_LIMIT and target.startswith(workbook_target):
                target_cell_truncated_prefix += 1
            else:
                raise ValueError(
                    f"Target cell for batch {batch}, QA index {qa_index} does not "
                    "match data/training/input_qa.json"
                )

            raw_full_response = row["generateds"]
            full_output = parse_legacy_assistant_content(
                raw_full_response,
                source=f"{path}:row={int(position) + 2}:generateds",
            )
            prompt = row["prompts"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"Full prompt for QA index {qa_index} in {path} is blank"
                )

            occurrence_key = (int(batch), qa_index)
            if occurrence_key in full_by_occurrence:
                raise ValueError(f"Duplicate full occurrence key: {occurrence_key}")
            full_by_occurrence[occurrence_key] = {
                "run_id": RUN_ID,
                "legacy_batch": int(batch),
                "qa_index": qa_index,
                "occurrence_id": (
                    f"{RUN_ID}::batch={int(batch):02d}::qa={qa_index:03d}"
                ),
                "target": target,
                "target_sha256": sha256_text(target),
                "target_characters": len(target),
                "full_output": full_output,
                "full_output_sha256": sha256_text(full_output),
                "full_output_characters": len(full_output),
                "full_prompt": prompt,
                "raw_full_response": raw_full_response,
                "full_source_file": str(path.resolve()),
                "full_source_row": int(position) + 2,
                **metadata_by_index[qa_index],
            }

    mask_paths_by_identity: dict[tuple[str, int], Path] = {}
    unparsable_names: list[str] = []
    for path in sorted(mask_root.glob("mask_*_generated_*.xlsx")):
        match = _MASK_FILENAME.fullmatch(path.name)
        if match is None:
            unparsable_names.append(path.name)
            continue
        identity = (match.group("indicator"), int(match.group("batch")))
        if identity in mask_paths_by_identity:
            raise ValueError(f"Duplicate masked workbook identity {identity}")
        mask_paths_by_identity[identity] = path
    if unparsable_names:
        raise ValueError(
            "Unexpected masked workbook names: " + ", ".join(unparsable_names)
        )

    expected_mask_identities = {
        (str(indicator), int(batch)) for indicator in indicators for batch in batches
    }
    actual_mask_identities = set(mask_paths_by_identity)
    if actual_mask_identities != expected_mask_identities:
        missing = sorted(expected_mask_identities - actual_mask_identities)
        extra = sorted(actual_mask_identities - expected_mask_identities)
        raise ValueError(
            f"Masked workbook universe mismatch; missing={missing}, extra={extra}"
        )

    mask_paths = list(mask_paths_by_identity.values())
    mask_inventory = _inventory_fingerprint(mask_paths)
    if tuple(indicators) == INDICATORS and tuple(batches) == BATCHES:
        _validate_known_hash(
            mask_inventory["sha256"],
            KNOWN_MASK_WORKBOOK_INVENTORY_SHA256,
            label="masked workbook inventory",
            strict=strict_known_hashes,
        )

    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    intervention_ids: set[str] = set()
    for indicator in indicators:
        for batch in batches:
            path = mask_paths_by_identity[(str(indicator), int(batch))]
            frame = pd.read_excel(path, engine="openpyxl")
            _require_columns(
                frame,
                (
                    "index",
                    "unmask_prompt",
                    "unmask_response",
                    "mask_prompt",
                    "mask_response",
                ),
                path=path,
            )
            if len(frame) != expected_rows_per_workbook:
                raise ValueError(
                    f"{path} has {len(frame)} rows; expected {expected_rows_per_workbook}"
                )
            indices: list[int] = []
            for position, row in frame.iterrows():
                qa_index = _coerce_index(
                    row["index"],
                    path=path,
                    row_number=int(position) + 2,
                )
                indices.append(qa_index)
                occurrence = full_by_occurrence.get((int(batch), qa_index))
                if occurrence is None:
                    raise ValueError(
                        f"Masked row {(indicator, batch, qa_index)} has no full occurrence"
                    )
                if row["unmask_prompt"] != occurrence["full_prompt"]:
                    raise ValueError(
                        f"Full prompt mismatch for {(indicator, batch, qa_index)}"
                    )
                expected_fragment = str(occurrence["raw_full_response"]).split(
                    "'content':"
                )[-1]
                if row["unmask_response"] != expected_fragment:
                    raise ValueError(
                        f"Copied full response mismatch for {(indicator, batch, qa_index)}"
                    )
                mask_prompt = row["mask_prompt"]
                if not isinstance(mask_prompt, str) or not mask_prompt.strip():
                    raise ValueError(
                        f"Masked prompt is blank for {(indicator, batch, qa_index)}"
                    )

                intervention_id = (
                    f"{occurrence['occurrence_id']}::indicator={indicator}"
                )
                if intervention_id in intervention_ids:
                    raise ValueError(f"Duplicate intervention ID: {intervention_id}")
                intervention_ids.add(intervention_id)

                raw_masked_response = row["mask_response"]
                try:
                    masked_output = parse_legacy_assistant_content(
                        raw_masked_response,
                        source=f"{path}:row={int(position) + 2}:mask_response",
                    )
                except TruncatedLegacyResponseError:
                    exclusions.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "run_id": RUN_ID,
                            "intervention_id": intervention_id,
                            "indicator": str(indicator),
                            "legacy_batch": int(batch),
                            "qa_index": qa_index,
                            "meeting_date": occurrence["meeting_date"],
                            "section_name": occurrence["section_name"],
                            "mask_source_file": str(path.resolve()),
                            "mask_source_row": int(position) + 2,
                            "masked_response_characters": len(raw_masked_response),
                            "masked_response_sha256": sha256_text(raw_masked_response),
                            "exclusion_reason": "excel_cell_truncated_masked_output",
                        }
                    )
                    continue

                records.append(
                    {
                        **{
                            key: value
                            for key, value in occurrence.items()
                            if key not in {"raw_full_response", "full_prompt"}
                        },
                        "schema_version": SCHEMA_VERSION,
                        "intervention_id": intervention_id,
                        "indicator": str(indicator),
                        "masked_output": masked_output,
                        "masked_output_sha256": sha256_text(masked_output),
                        "masked_output_characters": len(masked_output),
                        "full_prompt_sha256": sha256_text(occurrence["full_prompt"]),
                        "masked_prompt_sha256": sha256_text(mask_prompt),
                        "mask_source_file": str(path.resolve()),
                        "mask_source_row": int(position) + 2,
                        "intervention_status": "legacy_noncanonical_prompt_rewrite",
                    }
                )
            if len(indices) != len(set(indices)):
                raise ValueError(f"Duplicate QA indices in {path}")
            expected_indices = {
                qa_index
                for (candidate_batch, qa_index) in full_by_occurrence
                if candidate_batch == int(batch)
            }
            if set(indices) != expected_indices:
                raise ValueError(
                    f"Masked/full QA index mismatch for {(indicator, batch)}"
                )

    if (
        len(intervention_ids)
        != len(indicators) * len(batches) * expected_rows_per_workbook
    ):
        raise ValueError(
            f"Recovered {len(intervention_ids)} interventions; expected "
            f"{len(indicators) * len(batches) * expected_rows_per_workbook}"
        )

    if tuple(indicators) == INDICATORS and tuple(batches) == BATCHES:
        observed_exclusions = {
            (row["indicator"], row["legacy_batch"], row["qa_index"])
            for row in exclusions
        }
        if observed_exclusions != KNOWN_EXCLUSIONS:
            raise ValueError(
                "Historical exclusion set mismatch; "
                f"expected={sorted(KNOWN_EXCLUSIONS)}, "
                f"observed={sorted(observed_exclusions)}"
            )
        if len(full_by_occurrence) != 380 or len(records) != 2_655:
            raise ValueError(
                f"Historical population mismatch: full={len(full_by_occurrence)}, "
                f"retained={len(records)}"
            )

    source_audit = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "source_root": str(source_root),
        "analysis_classification": "legacy_prompt_perturbation_reanalysis",
        "canonical_leave_one_out": False,
        "formula": ("delta = cos(full_output, target) - cos(masked_output, target)"),
        "target_source": {
            "path": str(training_qa_path.resolve()),
            "sha256": training_qa_sha256,
            "row_count": len(training_rows),
            "field": "output",
            "role": "scoring_target",
        },
        "metadata_source": {
            "path": str(tagged_qa_path.resolve()),
            "sha256": tagged_qa_sha256,
            "row_count": len(tagged_rows),
            "role": "meeting_and_section_metadata_only",
            "derived_qa_index_mapping": metadata_binding,
        },
        "full_workbook_inventory": full_inventory,
        "mask_workbook_inventory": mask_inventory,
        "counts": {
            "indicators": len(indicators),
            "batches": len(batches),
            "rows_per_workbook": expected_rows_per_workbook,
            "full_occurrences": len(full_by_occurrence),
            "unique_qa_indices": len(
                {row["qa_index"] for row in full_by_occurrence.values()}
            ),
            "unique_meetings": len(
                {row["meeting_date"] for row in full_by_occurrence.values()}
            ),
            "candidate_interventions": len(intervention_ids),
            "retained_interventions": len(records),
            "excluded_interventions": len(exclusions),
            "workbook_targets_exact": target_cell_exact,
            "workbook_targets_truncated_exact_prefix": target_cell_truncated_prefix,
        },
        "strict_known_source_hashes": strict_known_hashes,
        "response_parsing": "ast.literal_eval(python_repr)['content']",
        "known_limitations": [
            "historical prompts are template rewrites, not exact single-block deletions",
            "historical generation used unseeded stochastic decoding",
            "historical generation and embedding checkpoints are not reproducible at recorded paths",
            "five masked outputs are truncated at the Excel cell limit",
        ],
    }
    return records, exclusions, source_audit


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_write_text(path, text)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


class CachedMeanPoolingEmbedder:
    """Mean-pool a local transformer and persist one vector per unique text."""

    def __init__(
        self,
        model_path: Path,
        *,
        model_fingerprint: dict[str, Any],
        cache_root: Path,
        batch_size: int,
        max_tokens: int | None,
    ):
        if batch_size <= 0:
            raise ValueError("embedding batch size must be positive")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive when supplied")

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_path = model_path
        self.model_fingerprint = model_fingerprint
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                raise ValueError("Tokenizer has no pad, EOS, or unknown token")

        logging.info("Loading embedding model from %s", model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        self.input_device = self.model.get_input_embeddings().weight.device

        policy = {
            "cache_schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
            "implementation_version": EMBEDDING_IMPLEMENTATION_VERSION,
            "pooling": "attention_mask_mean_pooling_float32",
            "normalization": "l2",
            "max_tokens": max_tokens,
            "truncation": max_tokens is not None,
        }
        policy_hash = sha256_text(json.dumps(policy, sort_keys=True))[:16]
        self.cache_dir = cache_root / f"{model_fingerprint['sha256']}_{policy_hash}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.policy = policy

    def _cache_path(self, text_hash: str) -> Path:
        return self.cache_dir / f"{text_hash}.npy"

    @staticmethod
    def _load_cached(path: Path) -> np.ndarray | None:
        if not path.is_file():
            return None
        try:
            value = np.load(path, allow_pickle=False)
        except (OSError, ValueError):
            return None
        if value.ndim != 1 or not np.isfinite(value).all():
            return None
        norm = float(np.linalg.norm(value))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
            return None
        return value.astype(np.float32, copy=False)

    @staticmethod
    def _save_cached(path: Path, value: np.ndarray) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
        temporary.replace(path)

    def encode(
        self,
        texts_by_hash: dict[str, str],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        embeddings: dict[str, np.ndarray] = {}
        missing: list[tuple[str, str, int]] = []
        token_counts: dict[str, int] = {}

        for text_hash, text in texts_by_hash.items():
            cached = self._load_cached(self._cache_path(text_hash))
            if cached is not None:
                embeddings[text_hash] = cached
                continue
            token_ids = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
            token_count = len(token_ids)
            token_counts[text_hash] = token_count
            missing.append((text_hash, text, token_count))

        config_limit = getattr(self.model.config, "max_position_embeddings", None)
        effective_limit = self.max_tokens or config_limit
        if effective_limit is not None:
            too_long = [
                (text_hash, token_count)
                for text_hash, _, token_count in missing
                if token_count > int(effective_limit) and self.max_tokens is None
            ]
            if too_long:
                raise ValueError(
                    f"{len(too_long)} texts exceed model max_position_embeddings="
                    f"{effective_limit}; examples={too_long[:3]}"
                )

        missing.sort(key=lambda item: item[2])
        logging.info(
            "Embedding cache: %d hits, %d missing unique texts",
            len(embeddings),
            len(missing),
        )
        for offset in range(0, len(missing), self.batch_size):
            batch = missing[offset : offset + self.batch_size]
            batch_texts = [item[1] for item in batch]
            tokenizer_kwargs: dict[str, Any] = {
                "padding": True,
                "return_tensors": "pt",
                "truncation": self.max_tokens is not None,
            }
            if self.max_tokens is not None:
                tokenizer_kwargs["max_length"] = self.max_tokens
            encoded = self.tokenizer(batch_texts, **tokenizer_kwargs)
            encoded = {
                key: value.to(self.input_device) for key, value in encoded.items()
            }
            model_kwargs = dict(encoded)
            if hasattr(self.model.config, "use_cache"):
                model_kwargs["use_cache"] = False
            with self.torch.inference_mode():
                model_output = self.model(**model_kwargs)
                token_embeddings = model_output[0]
                attention_mask = encoded["attention_mask"].to(token_embeddings.device)
                expanded_mask = attention_mask.unsqueeze(-1).expand(
                    token_embeddings.size()
                )
                pooled = (token_embeddings.float() * expanded_mask.float()).sum(
                    dim=1
                ) / expanded_mask.float().sum(dim=1).clamp(min=1e-9)
                pooled = self.torch.nn.functional.normalize(
                    pooled,
                    p=2,
                    dim=1,
                )
                batch_embeddings = pooled.cpu().numpy().astype(np.float32)

            for (text_hash, _, _), vector in zip(
                batch,
                batch_embeddings,
                strict=True,
            ):
                self._save_cached(self._cache_path(text_hash), vector)
                embeddings[text_hash] = vector
            completed = min(offset + len(batch), len(missing))
            if completed == len(missing) or completed % 25 == 0:
                logging.info(
                    "Embedded %d/%d missing texts (%d/%d total available)",
                    completed,
                    len(missing),
                    len(embeddings),
                    len(texts_by_hash),
                )

        observed_token_counts = list(token_counts.values())
        runtime = {
            "cache_directory": str(self.cache_dir.resolve()),
            "cache_hits": len(texts_by_hash) - len(missing),
            "cache_misses": len(missing),
            "unique_texts": len(texts_by_hash),
            "embedding_batch_size": self.batch_size,
            "policy": self.policy,
            "token_count_missing_min": (
                min(observed_token_counts) if observed_token_counts else None
            ),
            "token_count_missing_max": (
                max(observed_token_counts) if observed_token_counts else None
            ),
            "tokenizer_class": self.tokenizer.__class__.__name__,
            "tokenizer_model_max_length": self.tokenizer.model_max_length,
            "model_class": self.model.__class__.__name__,
            "model_max_position_embeddings": config_limit,
            "model_device_map": getattr(self.model, "hf_device_map", None),
            "input_device": str(self.input_device),
            "cuda_available": self.torch.cuda.is_available(),
            "cuda_device_count_visible": self.torch.cuda.device_count(),
        }
        return embeddings, runtime


def score_records_from_embeddings(
    records: Sequence[dict[str, Any]],
    embeddings: dict[str, np.ndarray],
    *,
    embedding_model_path: str,
    embedding_model_sha256: str,
    include_text: bool = False,
) -> list[dict[str, Any]]:
    """Attach both cosine components and their signed difference."""

    scored_rows: list[dict[str, Any]] = []
    for record in records:
        target_vector = embeddings[record["target_sha256"]]
        full_vector = embeddings[record["full_output_sha256"]]
        masked_vector = embeddings[record["masked_output_sha256"]]
        similarity_full = float(np.dot(target_vector, full_vector))
        similarity_masked = float(np.dot(target_vector, masked_vector))
        self_similarity = float(np.dot(full_vector, masked_vector))
        metrics = leave_one_out_metrics_from_similarities(
            similarity_full,
            similarity_masked,
            self_similarity=self_similarity,
        )
        excluded_fields = {"target", "full_output", "masked_output"}
        public_record = {
            key: value
            for key, value in record.items()
            if include_text or key not in excluded_fields
        }
        public_record.update(metrics)
        public_record["embedding_model_path"] = embedding_model_path
        public_record["embedding_model_sha256"] = embedding_model_sha256
        scored_rows.append(public_record)
    return scored_rows


def _collect_unique_texts(records: Sequence[dict[str, Any]]) -> dict[str, str]:
    texts_by_hash: dict[str, str] = {}
    for record in records:
        for text_field, hash_field in (
            ("target", "target_sha256"),
            ("full_output", "full_output_sha256"),
            ("masked_output", "masked_output_sha256"),
        ):
            text = record[text_field]
            text_hash = record[hash_field]
            previous = texts_by_hash.setdefault(text_hash, text)
            if previous != text:
                raise ValueError(f"SHA-256 collision detected for {text_hash}")
    return texts_by_hash


def _summarise(
    scored_rows: Sequence[dict[str, Any]],
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(scored_rows)
    summaries: list[dict[str, Any]] = []
    grouper: str | list[str]
    grouper = group_fields[0] if len(group_fields) == 1 else list(group_fields)
    for group_key, group in frame.groupby(grouper, sort=True, dropna=False):
        key_values = (group_key,) if len(group_fields) == 1 else tuple(group_key)
        meeting_level = (
            group.groupby("meeting_date", sort=True)[
                ["similarity_full", "similarity_masked", "delta", "self_distance"]
            ]
            .mean()
            .reset_index()
        )
        delta = group["delta"].astype(float)
        summary = {
            field: value for field, value in zip(group_fields, key_values, strict=True)
        }
        summary.update(
            {
                "n_pairs": int(len(group)),
                "n_meetings": int(group["meeting_date"].nunique()),
                "n_qa_indices": int(group["qa_index"].nunique()),
                "n_batches": int(group["legacy_batch"].nunique()),
                "mean_similarity_full_row_weighted": float(
                    group["similarity_full"].mean()
                ),
                "mean_similarity_masked_row_weighted": float(
                    group["similarity_masked"].mean()
                ),
                "mean_delta_row_weighted": float(delta.mean()),
                "median_delta_row_weighted": float(delta.median()),
                "sd_delta_row_weighted": (
                    float(delta.std(ddof=1)) if len(delta) > 1 else None
                ),
                "q1_delta_row_weighted": float(delta.quantile(0.25)),
                "q3_delta_row_weighted": float(delta.quantile(0.75)),
                "positive_delta_share": float((delta > 0).mean()),
                "negative_delta_share": float((delta < 0).mean()),
                "mean_delta_meeting_balanced": float(meeting_level["delta"].mean()),
                "mean_similarity_full_meeting_balanced": float(
                    meeting_level["similarity_full"].mean()
                ),
                "mean_similarity_masked_meeting_balanced": float(
                    meeting_level["similarity_masked"].mean()
                ),
                "mean_self_distance_meeting_balanced": float(
                    meeting_level["self_distance"].mean()
                ),
            }
        )
        summaries.append(summary)
    return summaries


def _meeting_level_rows(scored_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(scored_rows)
    grouped = (
        frame.groupby(
            ["indicator", "section_name", "meeting_date"],
            sort=True,
            dropna=False,
        )
        .agg(
            n_pairs=("delta", "size"),
            n_qa_indices=("qa_index", "nunique"),
            mean_similarity_full=("similarity_full", "mean"),
            mean_similarity_masked=("similarity_masked", "mean"),
            mean_delta=("delta", "mean"),
            mean_self_distance=("self_distance", "mean"),
        )
        .reset_index()
    )
    return grouped.to_dict(orient="records")


def _markdown_table(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[tuple[str, str]],
) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, separator]
    for row in rows:
        values: list[str] = []
        for field, _ in columns:
            value = row.get(field)
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            elif value is None:
                values.append("N/A")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _results_markdown(
    *,
    indicator_summary: Sequence[dict[str, Any]],
    section_summary: Sequence[dict[str, Any]],
    audit: dict[str, Any],
) -> str:
    count = audit["source_audit"]["counts"]
    model = audit["embedding_model_artifact"]
    return (
        "# Legacy Seven-Indicator Pilot: Target-Relative Re-score\n\n"
        "The signed metric is\n\n"
        "\\[\n"
        "\\Delta_i = \\cos(o_{\\mathrm{full}}, y)"
        " - \\cos(o_{\\mathrm{masked},i}, y).\n"
        "\\]\n\n"
        f"- Retained pairs: {count['retained_interventions']}\n"
        f"- Excluded truncated masked outputs: {count['excluded_interventions']}\n"
        f"- Meetings: {count['unique_meetings']}\n"
        f"- Embedding model: `{model['path']}`\n"
        f"- Embedding fingerprint: `{model['sha256']}`\n"
        "- Classification: legacy prompt-perturbation reanalysis; non-canonical\n\n"
        "## Indicator summary\n\n"
        + _markdown_table(
            indicator_summary,
            (
                ("indicator", "Indicator"),
                ("n_pairs", "N"),
                ("n_meetings", "Meetings"),
                ("mean_similarity_full_row_weighted", "Mean s_full"),
                ("mean_similarity_masked_row_weighted", "Mean s_masked"),
                ("mean_delta_row_weighted", "Mean delta"),
                ("median_delta_row_weighted", "Median delta"),
                ("mean_delta_meeting_balanced", "Meeting-balanced delta"),
            ),
        )
        + "\n\n## Indicator-by-section summary\n\n"
        + _markdown_table(
            section_summary,
            (
                ("indicator", "Indicator"),
                ("section_name", "Section"),
                ("n_pairs", "N"),
                ("n_meetings", "Meetings"),
                ("mean_delta_row_weighted", "Mean delta"),
                ("median_delta_row_weighted", "Median delta"),
                ("mean_delta_meeting_balanced", "Meeting-balanced delta"),
            ),
        )
        + "\n"
    )


def run_legacy_rescore(
    *,
    source_root: Path,
    output_dir: Path,
    embedding_model_path: Path | None,
    embedding_model_sha256: str | None = None,
    embedding_batch_size: int = 1,
    max_tokens: int | None = None,
    strict_known_hashes: bool = True,
    include_text: bool = False,
    validate_only: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-pid{os.getpid()}"
    )
    run_status_path = output_dir / (
        "input_validation_status.json" if validate_only else "run_status.json"
    )
    _write_json(
        run_status_path,
        {
            "schema_version": SCHEMA_VERSION,
            "execution_id": execution_id,
            "mode": "validate_only" if validate_only else "full_rescore",
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "completion_rule": (
                "This run validates inputs only and does not modify the status "
                "of any prior full re-score."
                if validate_only
                else "Treat score artifacts as complete only when this file has "
                "status='complete' for mode='full_rescore'."
            ),
        },
    )
    if not validate_only:
        for artifact_name in (
            "audit.json",
            "results.md",
            "scores.csv",
            "scores.jsonl",
            "indicator_summary.csv",
            "indicator_summary.jsonl",
            "indicator_section_summary.csv",
            "indicator_section_summary.jsonl",
            "meeting_level_scores.csv",
        ):
            (output_dir / artifact_name).unlink(missing_ok=True)

    records, exclusions, source_audit = load_legacy_pilot_rows(
        source_root,
        strict_known_hashes=strict_known_hashes,
    )
    artifact_prefix = "input_validation_" if validate_only else ""
    exclusions_jsonl_path = output_dir / f"{artifact_prefix}exclusions.jsonl"
    exclusions_csv_path = output_dir / f"{artifact_prefix}exclusions.csv"
    prepared_rows_path = output_dir / f"{artifact_prefix}prepared_rows.jsonl"
    _write_jsonl(exclusions_jsonl_path, exclusions)
    _write_csv(exclusions_csv_path, exclusions)
    prepared_rows = [
        {
            key: value
            for key, value in record.items()
            if key not in {"target", "full_output", "masked_output"}
        }
        for record in records
    ]
    _write_jsonl(prepared_rows_path, prepared_rows)
    validation_audit = {
        "schema_version": SCHEMA_VERSION,
        "execution_id": execution_id,
        "status": "validated_inputs",
        "source_audit": source_audit,
        "outputs": {
            "prepared_rows": str(prepared_rows_path.resolve()),
            "exclusions_jsonl": str(exclusions_jsonl_path.resolve()),
            "exclusions_csv": str(exclusions_csv_path.resolve()),
        },
    }
    _write_json(output_dir / "input_validation_audit.json", validation_audit)
    if validate_only:
        _write_json(
            run_status_path,
            {
                "schema_version": SCHEMA_VERSION,
                "execution_id": execution_id,
                "mode": "validate_only",
                "status": "validated_inputs",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "completion_rule": (
                    "This run validated inputs only and did not produce scores."
                ),
            },
        )
        logging.info("Input validation complete; --validate-only requested")
        return validation_audit

    if embedding_model_path is None:
        raise ValueError(
            "--embedding-model-path is required unless --validate-only is used"
        )
    embedding_model_path = embedding_model_path.expanduser().resolve()
    embedding_artifact = fingerprint_artifact_path(embedding_model_path)
    if embedding_model_sha256 is not None:
        expected_hash = embedding_model_sha256.strip().lower()
        if embedding_artifact["sha256"] != expected_hash:
            raise ValueError(
                "Embedding model fingerprint mismatch: "
                f"expected {expected_hash}, observed {embedding_artifact['sha256']}"
            )

    texts_by_hash = _collect_unique_texts(records)
    embedder = CachedMeanPoolingEmbedder(
        embedding_model_path,
        model_fingerprint=embedding_artifact,
        cache_root=output_dir / "embedding_cache",
        batch_size=embedding_batch_size,
        max_tokens=max_tokens,
    )
    embeddings, embedding_runtime = embedder.encode(texts_by_hash)
    scored_rows = score_records_from_embeddings(
        records,
        embeddings,
        embedding_model_path=str(embedding_model_path),
        embedding_model_sha256=embedding_artifact["sha256"],
        include_text=include_text,
    )
    indicator_summary = _summarise(scored_rows, ("indicator",))
    section_summary = _summarise(scored_rows, ("indicator", "section_name"))
    meeting_rows = _meeting_level_rows(scored_rows)

    _write_jsonl(output_dir / "scores.jsonl", scored_rows)
    _write_csv(output_dir / "scores.csv", scored_rows)
    _write_jsonl(output_dir / "indicator_summary.jsonl", indicator_summary)
    _write_csv(output_dir / "indicator_summary.csv", indicator_summary)
    _write_jsonl(
        output_dir / "indicator_section_summary.jsonl",
        section_summary,
    )
    _write_csv(
        output_dir / "indicator_section_summary.csv",
        section_summary,
    )
    _write_csv(output_dir / "meeting_level_scores.csv", meeting_rows)

    audit = {
        "schema_version": SCHEMA_VERSION,
        "execution_id": execution_id,
        "status": "complete",
        "source_audit": source_audit,
        "embedding_model_artifact": embedding_artifact,
        "embedding_runtime": embedding_runtime,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "score_counts": {
            "rows": len(scored_rows),
            "indicator_summaries": len(indicator_summary),
            "indicator_section_summaries": len(section_summary),
            "meeting_level_rows": len(meeting_rows),
            "unique_texts": len(texts_by_hash),
        },
        "outputs": {
            "scores_jsonl": str((output_dir / "scores.jsonl").resolve()),
            "scores_csv": str((output_dir / "scores.csv").resolve()),
            "indicator_summary_csv": str(
                (output_dir / "indicator_summary.csv").resolve()
            ),
            "indicator_section_summary_csv": str(
                (output_dir / "indicator_section_summary.csv").resolve()
            ),
            "meeting_level_scores_csv": str(
                (output_dir / "meeting_level_scores.csv").resolve()
            ),
            "exclusions_jsonl": str((output_dir / "exclusions.jsonl").resolve()),
        },
    }
    _write_json(output_dir / "audit.json", audit)
    _atomic_write_text(
        output_dir / "results.md",
        _results_markdown(
            indicator_summary=indicator_summary,
            section_summary=section_summary,
            audit=audit,
        ),
    )
    _write_json(
        run_status_path,
        {
            "schema_version": SCHEMA_VERSION,
            "execution_id": execution_id,
            "mode": "full_rescore",
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "audit": str((output_dir / "audit.json").resolve()),
            "results": str((output_dir / "results.md").resolve()),
        },
    )
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score the ft_20250330 seven-indicator pilot with "
            "delta = cos(full,target) - cos(masked,target)."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model-path", type=Path)
    parser.add_argument("--embedding-model-sha256")
    parser.add_argument("--embedding-batch-size", type=int, default=1)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="Explicit tokenizer truncation limit; 0 means no truncation.",
    )
    parser.add_argument(
        "--allow-source-hash-mismatch",
        action="store_true",
        help="Permit inputs that do not match the audited historical file hashes.",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include full target/output text in scores.csv and scores.jsonl.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and inventory Excel inputs without loading an embedding model.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    audit = run_legacy_rescore(
        source_root=args.source_root,
        output_dir=args.output_dir,
        embedding_model_path=args.embedding_model_path,
        embedding_model_sha256=args.embedding_model_sha256,
        embedding_batch_size=args.embedding_batch_size,
        max_tokens=args.max_tokens or None,
        strict_known_hashes=not args.allow_source_hash_mismatch,
        include_text=args.include_text,
        validate_only=args.validate_only,
    )
    print("Legacy seven-indicator pilot task finished")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
