"""Validate and atomically publish an immutable chk1 dataset release."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from open_r1.provenance import sha256_file, validate_sha256

from .contracts import (
    HANDOFF_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    RELEASE_SCHEMA_VERSION,
    canonical_json,
    sha256_text,
)

QUALITY_SCHEMA_VERSION = "chk1-automated-data-quality-report-v2"
_SAFE_RELEASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SPLITS = ("train", "eval", "test")
_FORBIDDEN_PROMPT_MARKERS = (
    "reference_excerpt",
    "archived_response",
    "teacher_response",
    "decision_label",
    "rate_change",
    "current_rate",
    "<<<begin-reference",
    "same-meeting minutes",
)


class ReleaseValidationError(ValueError):
    """Raised when a draft cannot be certified as canonical chk1 data."""


@dataclass(frozen=True)
class QualityThresholds:
    expected_meetings: Mapping[str, int | Collection[str]]
    expected_sample_ids: Mapping[str, Collection[str]]
    expected_topics: Collection[str]
    train_acceptance_rate: float = 0.70
    per_topic_acceptance_rate: float = 0.50


def _normalise_string_sequence(
    value: Collection[str],
    *,
    label: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Collection):
        raise ReleaseValidationError(f"{label} must be a collection of strings")
    normalised: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ReleaseValidationError(f"{label}[{index}] must be a non-empty string")
        cleaned = item.strip()
        if "\x00" in cleaned or "\ufffd" in cleaned:
            raise ReleaseValidationError(f"{label}[{index}] contains invalid encoding")
        normalised.append(cleaned)
    if not normalised and not allow_empty:
        raise ReleaseValidationError(f"{label} must not be empty")
    if len(normalised) != len(set(normalised)):
        raise ReleaseValidationError(f"{label} contains duplicate values")
    return tuple(sorted(normalised))


def _require_split_mapping(value: Mapping[str, Any], *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ReleaseValidationError(f"{label} must be a split mapping")
    observed = set(value)
    expected = set(_SPLITS)
    if observed != expected:
        raise ReleaseValidationError(
            f"{label} split keys mismatch: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _normalise_threshold_bindings(
    thresholds: QualityThresholds,
) -> tuple[
    dict[str, int | tuple[str, ...]],
    dict[str, tuple[str, ...]],
    tuple[str, ...],
]:
    _require_split_mapping(thresholds.expected_meetings, label="expected_meetings")
    _require_split_mapping(
        thresholds.expected_sample_ids,
        label="expected_sample_ids",
    )

    meetings: dict[str, int | tuple[str, ...]] = {}
    for split in _SPLITS:
        value = thresholds.expected_meetings[split]
        if isinstance(value, bool):
            raise ReleaseValidationError(
                f"expected_meetings[{split!r}] must be a count or exact membership"
            )
        if isinstance(value, int):
            if value < 0:
                raise ReleaseValidationError(
                    f"expected_meetings[{split!r}] must be non-negative"
                )
            meetings[split] = value
        else:
            meetings[split] = _normalise_string_sequence(
                value,
                label=f"expected_meetings[{split!r}]",
            )

    sample_ids = {
        split: _normalise_string_sequence(
            thresholds.expected_sample_ids[split],
            label=f"expected_sample_ids[{split!r}]",
        )
        for split in _SPLITS
    }
    owners: dict[str, str] = {}
    for split, identifiers in sample_ids.items():
        for sample_id in identifiers:
            prior = owners.setdefault(sample_id, split)
            if prior != split:
                raise ReleaseValidationError(
                    f"expected sample_id {sample_id!r} appears in {prior!r} and {split!r}"
                )

    topics = _normalise_string_sequence(
        thresholds.expected_topics,
        label="expected_topics",
    )
    for label, value in (
        ("train_acceptance_rate", thresholds.train_acceptance_rate),
        ("per_topic_acceptance_rate", thresholds.per_topic_acceptance_rate),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ReleaseValidationError(f"{label} must be numeric")
        if not 0 <= float(value) <= 1:
            raise ReleaseValidationError(f"{label} must be within 0..1")
    return meetings, sample_ids, topics


def _binding_hashes(
    *,
    expected_meetings: Mapping[str, int | tuple[str, ...]],
    expected_sample_ids: Mapping[str, tuple[str, ...]],
    expected_topics: tuple[str, ...],
) -> dict[str, str]:
    population = {
        "sample_ids": sorted(
            sample_id for split in _SPLITS for sample_id in expected_sample_ids[split]
        )
    }
    split_binding = {
        split: {
            "expected_meetings": (
                {"count": expected_meetings[split]}
                if isinstance(expected_meetings[split], int)
                else {"members": list(expected_meetings[split])}
            ),
            "sample_ids": list(expected_sample_ids[split]),
        }
        for split in _SPLITS
    }
    return {
        "population_binding_sha256": sha256_text(canonical_json(population)),
        "split_binding_sha256": sha256_text(canonical_json(split_binding)),
        "topic_binding_sha256": sha256_text(
            canonical_json({"topics": list(expected_topics)})
        ),
    }


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _required_text(row: Mapping[str, Any], field: str, *, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ReleaseValidationError(f"{label} requires non-empty string {field!r}")
    if "\x00" in value or "\ufffd" in value:
        raise ReleaseValidationError(f"{label}.{field} contains invalid encoding")
    return value


def _validate_sft_row(row: Mapping[str, Any], *, label: str) -> dict[str, str]:
    if set(row) != {"prompt", "response", "provided_data"}:
        raise ReleaseValidationError(
            f"{label} SFT keys must be exactly prompt/response/provided_data"
        )
    prompt = _required_text(row, "prompt", label=label)
    response = _required_text(row, "response", label=label)
    provided_data = _required_text(row, "provided_data", label=label)
    lowered = prompt.casefold()
    found = [marker for marker in _FORBIDDEN_PROMPT_MARKERS if marker in lowered]
    if found:
        raise ReleaseValidationError(
            f"{label} prompt contains prohibited markers: {found}"
        )
    if response.count("</think>") != 1 or "<think>" in response.casefold():
        raise ReleaseValidationError(
            f"{label} response must contain exactly one </think> boundary and no <think>"
        )
    return {"prompt": prompt, "response": response, "provided_data": provided_data}


def _validate_manifest_row(
    row: Mapping[str, Any],
    *,
    split: str,
    sft_row: Mapping[str, str],
    label: str,
) -> dict[str, Any]:
    required = {
        "schema_version",
        "sample_id",
        "meeting_date",
        "atomic_topic",
        "section_style_id",
        "split",
        "cutoff_ts",
        "evidence_lineage",
        "style_guide_sha256",
        "teacher_model_sha256",
        "tokenizer_sha256",
        "generation",
        "prompt_sha256",
        "reasoning_sha256",
        "final_analysis_sha256",
        "response_sha256",
        "provided_data_sha256",
        "input_truncated",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ReleaseValidationError(f"{label} manifest missing keys: {missing}")
    if row.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ReleaseValidationError(f"{label} has wrong manifest schema")
    if row.get("split") != split:
        raise ReleaseValidationError(f"{label} split mismatch")
    for field in ("sample_id", "meeting_date", "atomic_topic", "section_style_id"):
        _required_text(row, field, label=label)
    for field in (
        "style_guide_sha256",
        "teacher_model_sha256",
        "tokenizer_sha256",
        "prompt_sha256",
        "reasoning_sha256",
        "final_analysis_sha256",
        "response_sha256",
        "provided_data_sha256",
    ):
        validate_sha256(row.get(field), label=f"{label}.{field}")
    if row["prompt_sha256"] != sha256_text(sft_row["prompt"]):
        raise ReleaseValidationError(f"{label} prompt hash mismatch")
    if row["response_sha256"] != sha256_text(sft_row["response"]):
        raise ReleaseValidationError(f"{label} response hash mismatch")
    if row["provided_data_sha256"] != sha256_text(sft_row["provided_data"]):
        raise ReleaseValidationError(f"{label} provided_data hash mismatch")
    response_parts = sft_row["response"].split("\n</think>\n")
    if len(response_parts) != 2:
        raise ReleaseValidationError(f"{label} response boundary is malformed")
    if row["reasoning_sha256"] != sha256_text(response_parts[0].strip()):
        raise ReleaseValidationError(f"{label} reasoning hash mismatch")
    if row["final_analysis_sha256"] != sha256_text(response_parts[1].strip()):
        raise ReleaseValidationError(f"{label} final hash mismatch")
    if row.get("input_truncated") is not False:
        raise ReleaseValidationError(f"{label} input was truncated")
    generation = row.get("generation")
    if not isinstance(generation, Mapping):
        raise ReleaseValidationError(f"{label} generation provenance is invalid")
    for field in (
        "cache_key",
        "generation_provenance_sha256",
        "prompt_template_sha256",
        "generator_tokenizer_sha256",
        "student_tokenizer_sha256",
    ):
        try:
            validate_sha256(
                generation.get(field),
                label=f"{label}.generation.{field}",
            )
        except ValueError as exc:
            raise ReleaseValidationError(str(exc)) from exc
    if generation["student_tokenizer_sha256"] != row["tokenizer_sha256"]:
        raise ReleaseValidationError(
            f"{label} student/manifest tokenizer provenance mismatch"
        )
    lineage = row.get("evidence_lineage")
    if not isinstance(lineage, list) or not lineage:
        raise ReleaseValidationError(f"{label} evidence_lineage is empty")
    for evidence in lineage:
        if not isinstance(evidence, Mapping):
            raise ReleaseValidationError(f"{label} has invalid evidence lineage")
        validate_sha256(evidence.get("source_sha256"), label=f"{label}.source_sha256")
        if not evidence.get("evidence_id"):
            raise ReleaseValidationError(f"{label} has empty evidence ID")
    return dict(row)


def assess_release_quality(
    *,
    sft_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    exclusions: Sequence[Mapping[str, Any]],
    thresholds: QualityThresholds,
) -> tuple[
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, str]]],
]:
    expected_meetings, expected_sample_ids, expected_topics = (
        _normalise_threshold_bindings(thresholds)
    )
    binding_hashes = _binding_hashes(
        expected_meetings=expected_meetings,
        expected_sample_ids=expected_sample_ids,
        expected_topics=expected_topics,
    )
    validated_manifests: dict[str, list[dict[str, Any]]] = {}
    validated_sft: dict[str, list[dict[str, str]]] = {}
    all_sample_ids: set[str] = set()
    accepted_sample_ids_by_split = {split: set() for split in _SPLITS}
    all_keys: set[tuple[str, str]] = set()
    all_prompt_hashes: set[str] = set()
    all_response_hashes: set[str] = set()
    accepted_meetings_by_split: dict[str, set[str]] = {}
    accepted_by_topic: Counter[str] = Counter()
    candidates_by_topic: Counter[str] = Counter()
    manifest_provenance: dict[str, set[str]] = {
        "generation_provenance_sha256": set(),
        "prompt_template_sha256": set(),
        "generator_tokenizer_sha256": set(),
        "style_guide_sha256": set(),
        "teacher_model_sha256": set(),
        "tokenizer_sha256": set(),
    }

    for split in _SPLITS:
        raw_sft = list(sft_rows.get(split, ()))
        raw_manifest = list(manifest_rows.get(split, ()))
        if len(raw_sft) != len(raw_manifest):
            raise ReleaseValidationError(
                f"{split}: SFT/manifest row mismatch {len(raw_sft)} != {len(raw_manifest)}"
            )
        split_sft: list[dict[str, str]] = []
        split_manifests: list[dict[str, Any]] = []
        meetings: set[str] = set()
        for index, (sft, manifest) in enumerate(
            zip(raw_sft, raw_manifest, strict=True)
        ):
            label = f"{split}[{index}]"
            clean_sft = _validate_sft_row(sft, label=label)
            clean_manifest = _validate_manifest_row(
                manifest, split=split, sft_row=clean_sft, label=label
            )
            sample_id = str(clean_manifest["sample_id"])
            key = (
                str(clean_manifest["meeting_date"]),
                str(clean_manifest["atomic_topic"]),
            )
            prompt_hash = str(clean_manifest["prompt_sha256"])
            response_hash = str(clean_manifest["response_sha256"])
            if sample_id in all_sample_ids:
                raise ReleaseValidationError(f"duplicate sample_id: {sample_id}")
            if key in all_keys:
                raise ReleaseValidationError(f"duplicate canonical key: {key}")
            if prompt_hash in all_prompt_hashes:
                raise ReleaseValidationError(f"duplicate prompt hash: {prompt_hash}")
            if response_hash in all_response_hashes:
                raise ReleaseValidationError(
                    f"duplicate response hash: {response_hash}"
                )
            all_sample_ids.add(sample_id)
            accepted_sample_ids_by_split[split].add(sample_id)
            all_keys.add(key)
            all_prompt_hashes.add(prompt_hash)
            all_response_hashes.add(response_hash)
            meetings.add(key[0])
            accepted_by_topic[key[1]] += 1
            candidates_by_topic[key[1]] += 1
            manifest_provenance["prompt_template_sha256"].add(
                str(clean_manifest["generation"]["prompt_template_sha256"])
            )
            manifest_provenance["generation_provenance_sha256"].add(
                str(clean_manifest["generation"]["generation_provenance_sha256"])
            )
            manifest_provenance["generator_tokenizer_sha256"].add(
                str(clean_manifest["generation"]["generator_tokenizer_sha256"])
            )
            for field in (
                "style_guide_sha256",
                "teacher_model_sha256",
                "tokenizer_sha256",
            ):
                manifest_provenance[field].add(str(clean_manifest[field]))
            split_sft.append(clean_sft)
            split_manifests.append(clean_manifest)
        accepted_meetings_by_split[split] = meetings
        validated_sft[split] = split_sft
        validated_manifests[split] = split_manifests

    nonuniform_provenance = {
        field: sorted(values)
        for field, values in manifest_provenance.items()
        if len(values) > 1
    }
    if nonuniform_provenance:
        raise ReleaseValidationError(
            f"accepted manifests mix generation provenance: {nonuniform_provenance}"
        )

    exclusion_reasons: Counter[str] = Counter()
    excluded_by_split: Counter[str] = Counter()
    excluded_sample_ids: set[str] = set()
    excluded_sample_ids_by_split = {split: set() for split in _SPLITS}
    candidate_meetings_by_split = {
        split: set(accepted_meetings_by_split[split]) for split in _SPLITS
    }
    for index, exclusion in enumerate(exclusions):
        if not isinstance(exclusion, Mapping):
            raise ReleaseValidationError(f"exclusion row {index} is not an object")
        label = f"exclusion row {index}"
        sample_id = _required_text(exclusion, "sample_id", label=label)
        split = _required_text(exclusion, "split", label=label)
        topic = _required_text(exclusion, "atomic_topic", label=label)
        reason = _required_text(exclusion, "reason_code", label=label)
        meeting = _required_text(exclusion, "meeting_date", label=label)
        try:
            parsed_meeting = datetime.fromisoformat(meeting).date().isoformat()
        except ValueError as exc:
            raise ReleaseValidationError(
                f"{label}.meeting_date must be YYYY-MM-DD"
            ) from exc
        if parsed_meeting != meeting:
            raise ReleaseValidationError(
                f"{label}.meeting_date must be canonical YYYY-MM-DD"
            )
        if split not in _SPLITS:
            raise ReleaseValidationError(f"{label} has invalid split: {split!r}")
        if sample_id in all_sample_ids:
            raise ReleaseValidationError(
                f"sample_id appears in accepted and excluded rows: {sample_id}"
            )
        if sample_id in excluded_sample_ids:
            raise ReleaseValidationError(f"duplicate excluded sample_id: {sample_id}")
        excluded_sample_ids.add(sample_id)
        excluded_sample_ids_by_split[split].add(sample_id)
        candidate_meetings_by_split[split].add(meeting)
        excluded_by_split[split] += 1
        candidates_by_topic[topic] += 1
        exclusion_reasons[reason] += 1

    observed_sample_ids_by_split = {
        split: accepted_sample_ids_by_split[split] | excluded_sample_ids_by_split[split]
        for split in _SPLITS
    }
    for split in _SPLITS:
        expected_ids = set(expected_sample_ids[split])
        observed_ids = observed_sample_ids_by_split[split]
        if observed_ids != expected_ids:
            raise ReleaseValidationError(
                f"{split}: expected population mismatch: "
                f"missing={sorted(expected_ids - observed_ids)}, "
                f"extra={sorted(observed_ids - expected_ids)}"
            )
        expected = expected_meetings[split]
        observed_meetings = candidate_meetings_by_split[split]
        if isinstance(expected, int):
            if len(observed_meetings) != expected:
                raise ReleaseValidationError(
                    f"{split}: expected {expected} meetings, "
                    f"found {len(observed_meetings)}"
                )
        elif observed_meetings != set(expected):
            raise ReleaseValidationError(
                f"{split}: exact meeting membership mismatch: "
                f"missing={sorted(set(expected) - observed_meetings)}, "
                f"extra={sorted(observed_meetings - set(expected))}"
            )

    for left_index, left in enumerate(_SPLITS):
        for right in _SPLITS[left_index + 1 :]:
            overlap = sorted(
                candidate_meetings_by_split[left] & candidate_meetings_by_split[right]
            )
            if overlap:
                raise ReleaseValidationError(
                    f"meeting split overlap {left}/{right}: {overlap[:5]}"
                )

    observed_topics = set(candidates_by_topic)
    if observed_topics != set(expected_topics):
        raise ReleaseValidationError(
            "expected topic roster mismatch: "
            f"missing={sorted(set(expected_topics) - observed_topics)}, "
            f"extra={sorted(observed_topics - set(expected_topics))}"
        )

    split_counts = {split: len(validated_sft[split]) for split in _SPLITS}
    train_denominator = split_counts["train"] + excluded_by_split["train"]
    train_acceptance = (
        split_counts["train"] / train_denominator if train_denominator else 0.0
    )
    topic_rates = {
        topic: accepted_by_topic[topic] / count
        for topic, count in sorted(candidates_by_topic.items())
        if count
    }
    low_topics = {
        topic: rate
        for topic, rate in topic_rates.items()
        if rate < thresholds.per_topic_acceptance_rate
    }
    zero_tolerance_reasons = {
        reason: count
        for reason, count in exclusion_reasons.items()
        if reason
        in {
            "point_in_time_violation",
            "input_truncated",
            "teacher_prompt_leakage",
            "split_leakage",
            "conflicting_canonical_key",
        }
        and count
    }
    gates = {
        "train_acceptance": train_acceptance >= thresholds.train_acceptance_rate,
        "per_topic_acceptance": not low_topics,
        "zero_tolerance_exclusions": not zero_tolerance_reasons,
    }
    quality = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "status": "passed" if all(gates.values()) else "failed",
        "gates": gates,
        **binding_hashes,
        "population_count": sum(len(ids) for ids in expected_sample_ids.values()),
        "split_counts": split_counts,
        "candidate_counts": {
            split: len(observed_sample_ids_by_split[split]) for split in _SPLITS
        },
        "meeting_counts": {
            split: len(candidate_meetings_by_split[split]) for split in _SPLITS
        },
        "accepted_meeting_counts": {
            split: len(accepted_meetings_by_split[split]) for split in _SPLITS
        },
        "meeting_binding_modes": {
            split: (
                "count"
                if isinstance(expected_meetings[split], int)
                else "exact_membership"
            )
            for split in _SPLITS
        },
        "excluded_counts": dict(sorted(excluded_by_split.items())),
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "train_acceptance_rate": train_acceptance,
        "required_train_acceptance_rate": thresholds.train_acceptance_rate,
        "per_topic_acceptance_rates": topic_rates,
        "topics_below_threshold": low_topics,
        "zero_tolerance_exclusions": zero_tolerance_reasons,
        "manifest_provenance": {
            field: next(iter(values)) if values else None
            for field, values in manifest_provenance.items()
        },
        "invariants": {
            "canonical_key_unique": True,
            "sample_id_unique": True,
            "prompt_hash_unique": True,
            "response_hash_unique": True,
            "meeting_splits_disjoint": True,
            "expected_population_covered_once": True,
            "expected_topic_roster_exact": True,
            "input_truncation_count": 0,
            "teacher_prompt_leakage_count": 0,
        },
    }
    return quality, validated_manifests, validated_sft


def publish_release(
    *,
    release_root: str | Path,
    release_id: str,
    sft_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    exclusions: Sequence[Mapping[str, Any]],
    thresholds: QualityThresholds,
    prompt_template_sha256: str,
    style_guide_sha256: str,
    teacher_model_sha256: str,
    tokenizer_sha256: str,
    legacy_inventory: Mapping[str, Any],
    generated_at_utc: str | None = None,
    seal_permissions: bool = True,
) -> Path:
    """Publish only a fully passing release; partial drafts never get handoff."""

    if not _SAFE_RELEASE_ID.fullmatch(release_id) or release_id in {".", ".."}:
        raise ReleaseValidationError(f"unsafe release_id: {release_id!r}")
    for label, digest in (
        ("prompt_template_sha256", prompt_template_sha256),
        ("style_guide_sha256", style_guide_sha256),
        ("teacher_model_sha256", teacher_model_sha256),
        ("tokenizer_sha256", tokenizer_sha256),
    ):
        validate_sha256(digest, label=label)
    quality, manifests, sft = assess_release_quality(
        sft_rows=sft_rows,
        manifest_rows=manifest_rows,
        exclusions=exclusions,
        thresholds=thresholds,
    )
    if quality["status"] != "passed":
        failed = [name for name, passed in quality["gates"].items() if not passed]
        raise ReleaseValidationError(
            f"chk1 release failed quality gates and was not published: {failed}"
        )
    expected_provenance = {
        "prompt_template_sha256": prompt_template_sha256,
        "style_guide_sha256": style_guide_sha256,
        "teacher_model_sha256": teacher_model_sha256,
        "tokenizer_sha256": tokenizer_sha256,
    }
    observed_provenance = quality["manifest_provenance"]
    mismatched_provenance = {
        field: {"expected": expected, "observed": observed_provenance.get(field)}
        for field, expected in expected_provenance.items()
        if observed_provenance.get(field) != expected
    }
    if mismatched_provenance:
        raise ReleaseValidationError(
            "release provenance does not match accepted manifests: "
            f"{mismatched_provenance}"
        )

    root = Path(release_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / release_id
    if destination.exists():
        raise FileExistsError(
            f"Immutable release already exists; choose a new ID: {destination}"
        )
    staging = root / f".{release_id}.staging"
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")
    staging.mkdir()
    try:
        for split in _SPLITS:
            _atomic_write(staging / "sft" / f"{split}.jsonl", _jsonl_text(sft[split]))
            _atomic_write(
                staging / "manifests" / f"{split}.jsonl",
                _jsonl_text(manifests[split]),
            )
        _atomic_write(staging / "audit" / "exclusions.jsonl", _jsonl_text(exclusions))
        _atomic_write(staging / "audit" / "quality_report.json", _json_text(quality))
        _atomic_write(
            staging / "audit" / "legacy_inventory.json", _json_text(legacy_inventory)
        )

        split_files: dict[str, dict[str, Any]] = {}
        manifest_files: dict[str, dict[str, Any]] = {}
        for split in _SPLITS:
            sft_path = staging / "sft" / f"{split}.jsonl"
            manifest_path = staging / "manifests" / f"{split}.jsonl"
            split_files[split] = {
                "path": f"sft/{split}.jsonl",
                "rows": len(sft[split]),
                "sha256": sha256_file(sft_path),
            }
            manifest_files[split] = {
                "path": f"manifests/{split}.jsonl",
                "rows": len(manifests[split]),
                "sha256": sha256_file(manifest_path),
            }
        generated = generated_at_utc or _utc_now()
        handoff = {
            "schema_version": HANDOFF_SCHEMA_VERSION,
            "release_schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": release_id,
            "generated_at_utc": generated,
            "immutable": True,
            "quality_status": "passed",
            "population_binding_sha256": quality["population_binding_sha256"],
            "split_binding_sha256": quality["split_binding_sha256"],
            "topic_binding_sha256": quality["topic_binding_sha256"],
            "population_count": quality["population_count"],
            "split_counts": quality["split_counts"],
            "split_files": split_files,
            "manifest_files": manifest_files,
            "prompt_template_sha256": prompt_template_sha256,
            "generation_provenance_sha256": observed_provenance[
                "generation_provenance_sha256"
            ],
            "generator_tokenizer_sha256": observed_provenance[
                "generator_tokenizer_sha256"
            ],
            "style_guide_sha256": style_guide_sha256,
            "teacher_model_sha256": teacher_model_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "quality_report": {
                "path": "audit/quality_report.json",
                "sha256": sha256_file(staging / "audit" / "quality_report.json"),
            },
            "legacy_inventory": {
                "path": "audit/legacy_inventory.json",
                "sha256": sha256_file(staging / "audit" / "legacy_inventory.json"),
            },
        }
        _atomic_write(staging / "handoff.json", _json_text(handoff))
        os.replace(staging, destination)
        if seal_permissions:
            for path in sorted(destination.rglob("*"), reverse=True):
                if path.is_file():
                    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
                elif path.is_dir():
                    path.chmod(
                        stat.S_IRUSR
                        | stat.S_IXUSR
                        | stat.S_IRGRP
                        | stat.S_IXGRP
                        | stat.S_IROTH
                        | stat.S_IXOTH
                    )
            destination.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
        return destination / "handoff.json"
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
