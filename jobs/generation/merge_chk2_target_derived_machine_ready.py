"""Merge immutable CHK2 machine-ready releases without weakening provenance.

The base release defines the canonical 3,198-row screened population.  Every
row contributed by the recovery release must be a previously rejected member
of that same population.  Candidate/manifest pairs are kept together, sorted
by sample ID, fully re-hashed, and published to a new directory atomically.
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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_machine_ready_v1_20260829"
)
DEFAULT_ADDITION_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_recovery_regenerate_"
    "machine_ready_v1_20260829"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_machine_ready_v2_20260829"
)

SPLITS = ("train", "validation", "test")
READY_STATUS = "machine_screen_complete_training_ready"
READY_SCHEMA_VERSION = "chk2-target-derived-machine-ready-v1"
MERGE_SCHEMA_VERSION = "chk2-target-derived-machine-ready-union-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_STUDENT_PROMPT_PREFIX = (
    "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
)
_CORE_SOURCE_FIELDS = (
    "sample_id",
    "split",
    "meeting_date",
    "line_id",
    "official_minutes_paragraph",
    "official_minutes_raw",
    "official_minutes_sha256",
    "official_minutes_raw_sha256",
    "legacy_sample_ids",
    "official_source_files",
    "official_source_match",
    "section_category",
    "section_name",
    "topic",
)


class MergeError(RuntimeError):
    """A source release is not safe to merge."""


@dataclass(frozen=True)
class CandidatePair:
    candidate: dict[str, Any]
    manifest: dict[str, Any]
    source_label: str


@dataclass(frozen=True)
class ReadyRelease:
    root: Path
    summary_path: Path
    summary: dict[str, Any]
    handoff_path: Path
    handoff: dict[str, Any]
    prompt_contract_path: Path
    prompt_contract_sha256: str
    pairs: dict[str, tuple[CandidatePair, ...]]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MergeError(f"cannot hash file: {path}: {exc}") from exc
    return digest.hexdigest()


def _id_digest(values: Sequence[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values))
    return _sha256_text(payload)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MergeError(f"cannot read JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MergeError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MergeError(f"cannot read JSONL: {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise MergeError(f"blank JSONL row: {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MergeError(f"invalid JSONL row: {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise MergeError(f"non-object JSONL row: {path}:{line_number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(dict(row)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Publish atomically without replacing a concurrently created release."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise MergeError(
            "atomic renameat2(RENAME_NOREPLACE) is unavailable; refusing to publish"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise MergeError(f"output was created concurrently: {destination}")
    raise MergeError(
        "atomic release publication failed: "
        f"{source} -> {destination}: {os.strerror(error_number)}"
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _declared_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MergeError(f"{label} path must be non-empty text")
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def _relative_artifact(root: Path, record: object, *, label: str) -> Path:
    if not isinstance(record, dict):
        raise MergeError(f"{label} record is missing")
    relative = record.get("path")
    expected_sha = record.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise MergeError(f"{label}.path must be non-empty text")
    if not isinstance(expected_sha, str) or not _SHA256_RE.fullmatch(expected_sha):
        raise MergeError(f"{label}.sha256 is invalid")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise MergeError(f"{label} escapes release root: {relative}") from exc
    if not path.is_file():
        raise MergeError(f"{label} is missing: {path}")
    observed = _sha256_file(path)
    if observed != expected_sha:
        raise MergeError(
            f"{label} SHA256 mismatch: expected={expected_sha} observed={observed}"
        )
    return path


def _require_sha(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise MergeError(f"{label} is not a SHA256 digest")
    return value


def _validate_pair(
    candidate: dict[str, Any],
    manifest: dict[str, Any],
    *,
    split: str,
    source_label: str,
) -> CandidatePair:
    sample_id = manifest.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise MergeError(f"{source_label}/{split} manifest sample_id is invalid")
    if manifest.get("split") != split:
        raise MergeError(f"wrong split for {source_label} row: {sample_id}")
    if manifest.get("machine_pass") is not True:
        raise MergeError(f"non-passing row in {source_label}: {sample_id}")
    if manifest.get("training_ready") is not True:
        raise MergeError(f"non-ready row in {source_label}: {sample_id}")
    if "human_review_status" in manifest:
        raise MergeError(f"review field remains in {source_label}: {sample_id}")
    if manifest.get("training_only") is not True:
        raise MergeError(f"row is not training-only in {source_label}: {sample_id}")
    if manifest.get("evaluation_eligible") is not False:
        raise MergeError(f"row is unexpectedly evaluation-eligible: {sample_id}")
    expected_split_role = (
        "target_derived_training"
        if split == "train"
        else "target_derived_monitoring_quarantine"
    )
    if manifest.get("split_role") != expected_split_role:
        raise MergeError(f"row has invalid split role: {sample_id}")
    expected_lineage_flags = {
        "analysis_is_reference_free": False,
        "analysis_source_lineage": "target_derived_from_official_minutes",
        "analysis_teacher_saw_official_target": True,
        "reasoning_teacher_saw_official_target": True,
        "student_prompt_has_direct_target_field": False,
        "target_is_teacher_synthetic_rewrite": False,
        "reported_checkpoint_eligible": False,
        "suitable_for_leakage_safe_evaluation": False,
    }
    for field, expected_value in expected_lineage_flags.items():
        if manifest.get(field) != expected_value:
            raise MergeError(
                f"row has invalid target-derived lineage flag {field}: {sample_id}"
            )
    meeting_date = manifest.get("meeting_date")
    if not isinstance(meeting_date, str) or not meeting_date:
        raise MergeError(f"row has invalid meeting_date: {sample_id}")
    legacy_ids = manifest.get("legacy_sample_ids")
    if not isinstance(legacy_ids, list) or not all(
        isinstance(value, str) and value for value in legacy_ids
    ):
        raise MergeError(f"row has invalid legacy_sample_ids: {sample_id}")
    if len(legacy_ids) != len(set(legacy_ids)):
        raise MergeError(f"row has duplicate legacy_sample_ids: {sample_id}")

    if set(candidate) != {"prompt", "response"}:
        raise MergeError(f"unexpected SFT candidate keys for {sample_id}")
    prompt = candidate.get("prompt")
    response = candidate.get("response")
    if not isinstance(prompt, str) or not prompt.strip():
        raise MergeError(f"empty prompt for {sample_id}")
    if not isinstance(response, str) or not response.strip():
        raise MergeError(f"empty response for {sample_id}")
    if _sha256_text(prompt) != _require_sha(
        manifest.get("prompt_sha256"), label=f"{sample_id}.prompt_sha256"
    ):
        raise MergeError(f"prompt binding mismatch for {sample_id}")
    if _sha256_text(response) != _require_sha(
        manifest.get("response_sha256"), label=f"{sample_id}.response_sha256"
    ):
        raise MergeError(f"response binding mismatch for {sample_id}")

    analysis = manifest.get("analysis")
    reasoning = manifest.get("reasoning")
    official = manifest.get("official_minutes_paragraph")
    if not all(isinstance(item, str) and item.strip() for item in (analysis, reasoning, official)):
        raise MergeError(f"missing analysis/reasoning/official text for {sample_id}")
    expected_prompt = _STUDENT_PROMPT_PREFIX + _canonical_json(
        {"analysis": analysis.strip()}
    )
    if prompt != expected_prompt:
        raise MergeError(f"student prompt contract mismatch for {sample_id}")
    if _sha256_text(reasoning) != _require_sha(
        manifest.get("reasoning_sha256"), label=f"{sample_id}.reasoning_sha256"
    ):
        raise MergeError(f"reasoning binding mismatch for {sample_id}")
    if response != f"{reasoning}\n</think>\n{official}":
        raise MergeError(f"completion assembly mismatch for {sample_id}")

    official_sha = _require_sha(
        manifest.get("official_minutes_sha256"),
        label=f"{sample_id}.official_minutes_sha256",
    )
    if _sha256_text(official) != official_sha:
        raise MergeError(f"official paragraph binding mismatch for {sample_id}")
    raw = manifest.get("official_minutes_raw")
    if not isinstance(raw, str) or _sha256_text(raw) != _require_sha(
        manifest.get("official_minutes_raw_sha256"),
        label=f"{sample_id}.official_minutes_raw_sha256",
    ):
        raise MergeError(f"raw official paragraph binding mismatch for {sample_id}")
    if not sample_id.endswith(f"-{official_sha[:8]}"):
        raise MergeError(f"sample_id target hash suffix mismatch for {sample_id}")
    return CandidatePair(
        candidate=dict(candidate),
        manifest=dict(manifest),
        source_label=source_label,
    )


def _source_root_from_summary(summary: Mapping[str, Any], *, label: str) -> Path:
    record = summary.get("source_release")
    if not isinstance(record, dict):
        raise MergeError(f"{label} source_release is missing")
    source_root = _declared_path(record.get("path"), label=f"{label}.source_release")
    summary_path = source_root / "summary.json"
    if not summary_path.is_file():
        raise MergeError(f"{label} source summary is missing: {summary_path}")
    expected = _require_sha(
        record.get("summary_sha256"), label=f"{label}.source_release.summary_sha256"
    )
    observed = _sha256_file(summary_path)
    if observed != expected:
        raise MergeError(
            f"{label} source summary SHA256 mismatch: expected={expected} observed={observed}"
        )
    return source_root


def _load_ready_release(root: Path, *, label: str) -> ReadyRelease:
    resolved = root.resolve()
    summary_path = resolved / "summary.json"
    handoff_path = resolved / "handoff.json"
    summary = _read_json(summary_path)
    handoff = _read_json(handoff_path)
    if summary.get("schema_version") != READY_SCHEMA_VERSION:
        raise MergeError(f"{label} is not a v1 machine-ready release")
    if summary.get("status") != READY_STATUS or summary.get("quality_status") != READY_STATUS:
        raise MergeError(f"{label} status is not machine-ready")
    if summary.get("training_ready") is not True:
        raise MergeError(f"{label} training_ready is not true")
    if "human_review_status" in summary:
        raise MergeError(f"{label} summary still contains human review state")
    if summary.get("failure_count") != 0:
        raise MergeError(f"{label} contains unresolved provider failures")
    lineage = summary.get("lineage")
    if not isinstance(lineage, dict) or lineage.get("training_ready") is not True:
        raise MergeError(f"{label} lineage is not training-ready")
    if lineage.get("training_only") is not True or lineage.get("evaluation_eligible") is not False:
        raise MergeError(f"{label} lineage scope is invalid")

    expected_total = summary.get("total_training_rows")
    if not isinstance(expected_total, int) or expected_total <= 0:
        raise MergeError(f"{label} has no training rows")
    if summary.get("total_machine_pass") != expected_total:
        raise MergeError(f"{label} machine-pass/training total mismatch")

    summary_record = handoff.get("summary")
    if not isinstance(summary_record, dict):
        raise MergeError(f"{label} handoff summary record is missing")
    if summary_record.get("path") != "summary.json":
        raise MergeError(f"{label} handoff summary path is invalid")
    expected_summary_sha = _require_sha(
        summary_record.get("sha256"), label=f"{label}.handoff.summary.sha256"
    )
    if _sha256_file(summary_path) != expected_summary_sha:
        raise MergeError(f"{label} handoff does not bind summary.json")
    if handoff.get("schema_version") != READY_SCHEMA_VERSION:
        raise MergeError(f"{label} handoff schema is invalid")
    if handoff.get("status") != READY_STATUS or handoff.get("training_ready") is not True:
        raise MergeError(f"{label} handoff is not machine-ready")
    if handoff.get("evaluation_eligible") is not False:
        raise MergeError(f"{label} handoff evaluation scope is invalid")
    if handoff.get("teacher_model") != summary.get("teacher_model"):
        raise MergeError(f"{label} handoff teacher model mismatch")
    if handoff.get("lineage") != lineage:
        raise MergeError(f"{label} handoff lineage mismatch")
    if handoff.get("source_release") != summary.get("source_release"):
        raise MergeError(f"{label} handoff source binding mismatch")
    if handoff.get("dataset_path") != _display_path(resolved / "sft_candidate"):
        raise MergeError(f"{label} handoff dataset path mismatch")
    if handoff.get("manifest_path") != _display_path(resolved / "manifests"):
        raise MergeError(f"{label} handoff manifest path mismatch")
    if handoff.get("total_training_rows") != expected_total:
        raise MergeError(f"{label} handoff total mismatch")

    artifacts = summary.get("artifacts")
    split_counts = summary.get("split_counts")
    handoff_counts = handoff.get("split_counts")
    if not isinstance(artifacts, dict) or not isinstance(split_counts, dict):
        raise MergeError(f"{label} artifacts or split counts are missing")
    if not isinstance(handoff_counts, dict):
        raise MergeError(f"{label} handoff split counts are missing")

    pairs_by_split: dict[str, tuple[CandidatePair, ...]] = {}
    seen_ids: set[str] = set()
    seen_official: set[str] = set()
    for split in SPLITS:
        record = artifacts.get(split)
        if not isinstance(record, dict):
            raise MergeError(f"{label}/{split} artifact record is missing")
        expected_rows = record.get("rows")
        split_record = split_counts.get(split)
        split_machine_pass = (
            split_record.get("machine_pass") if isinstance(split_record, dict) else None
        )
        if not isinstance(expected_rows, int) or expected_rows < 0:
            raise MergeError(f"{label}/{split} row count is invalid")
        if expected_rows != split_machine_pass or expected_rows != handoff_counts.get(split):
            raise MergeError(f"{label}/{split} summary/handoff count mismatch")
        candidate_path = _relative_artifact(
            resolved, record.get("sft_candidate"), label=f"{label}/{split}.sft_candidate"
        )
        manifest_path = _relative_artifact(
            resolved, record.get("manifest"), label=f"{label}/{split}.manifest"
        )
        candidate_rows = _read_jsonl(candidate_path)
        manifest_rows = _read_jsonl(manifest_path)
        if len(candidate_rows) != expected_rows or len(manifest_rows) != expected_rows:
            raise MergeError(f"{label}/{split} physical row count mismatch")
        pairs: list[CandidatePair] = []
        for candidate, manifest in zip(candidate_rows, manifest_rows):
            pair = _validate_pair(
                candidate, manifest, split=split, source_label=label
            )
            sample_id = pair.manifest["sample_id"]
            official_sha = pair.manifest["official_minutes_sha256"]
            if sample_id in seen_ids:
                raise MergeError(f"duplicate sample_id inside {label}: {sample_id}")
            if official_sha in seen_official:
                raise MergeError(
                    f"duplicate official target inside {label}: {official_sha}"
                )
            seen_ids.add(sample_id)
            seen_official.add(official_sha)
            teacher = pair.manifest.get("teacher")
            provider_identity = summary.get("provider_identity")
            if not isinstance(teacher, dict) or not isinstance(
                provider_identity, dict
            ):
                raise MergeError(f"{label} teacher/provider identity is missing")
            if teacher.get("model") != summary.get("teacher_model"):
                raise MergeError(f"{label} row teacher model mismatch: {sample_id}")
            if teacher.get("system_fingerprint") != provider_identity.get(
                "system_fingerprint"
            ):
                raise MergeError(
                    f"{label} row provider fingerprint mismatch: {sample_id}"
                )
            if not isinstance(teacher.get("provider"), str) or not teacher.get(
                "provider"
            ):
                raise MergeError(f"{label} row provider is missing: {sample_id}")
            pairs.append(pair)
        pairs_by_split[split] = tuple(pairs)
    if sum(len(rows) for rows in pairs_by_split.values()) != expected_total:
        raise MergeError(f"{label} physical total mismatch")

    source_root = _source_root_from_summary(summary, label=label)
    prompt_contract_path = source_root / "prompt_contract.json"
    if not prompt_contract_path.is_file():
        raise MergeError(f"{label} prompt contract is missing: {prompt_contract_path}")
    return ReadyRelease(
        root=resolved,
        summary_path=summary_path,
        summary=summary,
        handoff_path=handoff_path,
        handoff=handoff,
        prompt_contract_path=prompt_contract_path,
        prompt_contract_sha256=_sha256_file(prompt_contract_path),
        pairs=pairs_by_split,
    )


def _read_split_rows(root: Path, directory: str) -> dict[str, list[dict[str, Any]]]:
    return {
        split: _read_jsonl(root / directory / f"{split}.jsonl") for split in SPLITS
    }


def _validate_promoted_source_copy(
    release: ReadyRelease, *, label: str
) -> dict[str, Any]:
    """Prove that a ready parent is the complete normalized copy of its source."""

    source_root = _source_root_from_summary(release.summary, label=label)
    source_summary_path = source_root / "summary.json"
    source_summary = _read_json(source_summary_path)
    if source_summary.get("status") != "machine_screen_complete_human_review_pending":
        raise MergeError(f"{label} source machine screen is not complete")
    if source_summary.get("phase") != "verify" or source_summary.get("failure_count") != 0:
        raise MergeError(f"{label} source verification is incomplete")
    if source_summary.get("training_ready") is not False:
        raise MergeError(f"{label} source has an invalid pre-promotion readiness state")
    expected_total = release.summary.get("total_training_rows")
    if source_summary.get("total_machine_pass") != expected_total:
        raise MergeError(f"{label} promoted/source machine-pass total mismatch")
    source_split_counts = source_summary.get("split_counts")
    if not isinstance(source_split_counts, dict):
        raise MergeError(f"{label} source split counts are missing")

    artifact_records: dict[str, Any] = {}
    for split in SPLITS:
        candidate_path = source_root / "sft_candidate" / f"{split}.jsonl"
        manifest_path = source_root / "manifests" / f"{split}.jsonl"
        source_candidates = _read_jsonl(candidate_path)
        source_manifests = _read_jsonl(manifest_path)
        promoted_pairs = release.pairs[split]
        split_record = source_split_counts.get(split)
        expected_split = (
            split_record.get("machine_pass")
            if isinstance(split_record, dict)
            else None
        )
        if (
            len(source_candidates) != len(source_manifests)
            or len(source_candidates) != len(promoted_pairs)
            or len(source_candidates) != expected_split
        ):
            raise MergeError(f"{label}/{split} promoted/source row count mismatch")
        for line_number, (source_candidate, source_manifest, promoted) in enumerate(
            zip(source_candidates, source_manifests, promoted_pairs), start=1
        ):
            expected_manifest = dict(source_manifest)
            expected_manifest.pop("human_review_status", None)
            expected_manifest["training_ready"] = True
            if source_candidate != promoted.candidate:
                raise MergeError(
                    f"{label}/{split} candidate differs from source line {line_number}"
                )
            if expected_manifest != promoted.manifest:
                raise MergeError(
                    f"{label}/{split} manifest differs from normalized source "
                    f"line {line_number}"
                )
        artifact_records[split] = {
            "rows": len(source_candidates),
            "sft_candidate": {
                "path": _display_path(candidate_path),
                "sha256": _sha256_file(candidate_path),
            },
            "manifest": {
                "path": _display_path(manifest_path),
                "sha256": _sha256_file(manifest_path),
            },
        }
    return {
        "path": _display_path(source_root),
        "summary_sha256": _sha256_file(source_summary_path),
        "total_machine_pass": expected_total,
        "artifacts": artifact_records,
    }


def _canonical_artifact_records(source_root: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for split in SPLITS:
        records[split] = {}
        for directory in ("prepared", "machine_rejections"):
            path = source_root / directory / f"{split}.jsonl"
            rows = _read_jsonl(path)
            records[split][directory] = {
                "path": _display_path(path),
                "rows": len(rows),
                "sha256": _sha256_file(path),
            }
    return records


def _validate_canonical_population(
    *, base: ReadyRelease, addition: ReadyRelease
) -> tuple[dict[str, dict[str, Any]], set[str], dict[str, int], dict[str, Any]]:
    source_root = _source_root_from_summary(base.summary, label="base")
    source_summary = _read_json(source_root / "summary.json")
    prepared_by_split = _read_split_rows(source_root, "prepared")
    rejected_by_split = _read_split_rows(source_root, "machine_rejections")
    prepared_index: dict[str, dict[str, Any]] = {}
    rejection_ids: set[str] = set()
    prepared_counts: dict[str, int] = {}
    for split in SPLITS:
        prepared_counts[split] = len(prepared_by_split[split])
        for row in prepared_by_split[split]:
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise MergeError(f"canonical prepared row has invalid sample_id in {split}")
            if sample_id in prepared_index:
                raise MergeError(f"duplicate canonical prepared sample_id: {sample_id}")
            if row.get("split") != split:
                raise MergeError(f"canonical prepared split mismatch: {sample_id}")
            prepared_index[sample_id] = row
        for row in rejected_by_split[split]:
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or row.get("split") != split:
                raise MergeError(f"canonical rejection row is invalid in {split}")
            if sample_id in rejection_ids:
                raise MergeError(f"duplicate canonical rejection sample_id: {sample_id}")
            rejection_ids.add(sample_id)

    expected_total = source_summary.get("total_prepared")
    expected_rejected = source_summary.get("total_machine_rejected_or_pending")
    if sum(prepared_counts.values()) != expected_total:
        raise MergeError("canonical prepared rows do not match source summary")
    if len(rejection_ids) != expected_rejected:
        raise MergeError("canonical rejection rows do not match source summary")
    source_split_counts = source_summary.get("split_counts")
    if not isinstance(source_split_counts, dict):
        raise MergeError("canonical source split counts are missing")
    for split in SPLITS:
        record = source_split_counts.get(split)
        if not isinstance(record, dict) or record.get("prepared") != prepared_counts[split]:
            raise MergeError(f"canonical prepared split count mismatch: {split}")

    base_pairs = [pair for split in SPLITS for pair in base.pairs[split]]
    addition_pairs = [pair for split in SPLITS for pair in addition.pairs[split]]
    base_ids = {pair.manifest["sample_id"] for pair in base_pairs}
    addition_ids = {pair.manifest["sample_id"] for pair in addition_pairs}
    if base_ids & rejection_ids:
        raise MergeError("base ready rows overlap canonical rejection ledger")
    if base_ids | rejection_ids != set(prepared_index):
        raise MergeError("base ready rows and rejection ledger do not partition population")
    unexpected_additions = addition_ids - rejection_ids
    if unexpected_additions:
        raise MergeError(
            "addition contains rows outside canonical rejection ledger: "
            f"{sorted(unexpected_additions)[:3]}"
        )

    for pair in base_pairs + addition_pairs:
        sample_id = pair.manifest["sample_id"]
        prepared = prepared_index.get(sample_id)
        if prepared is None:
            raise MergeError(f"merged row is absent from canonical population: {sample_id}")
        for field in _CORE_SOURCE_FIELDS:
            if pair.manifest.get(field) != prepared.get(field):
                raise MergeError(f"canonical source field mismatch {field}: {sample_id}")
        if pair.manifest.get("official_minutes_normalization_repairs") != prepared.get(
            "official_normalization_repairs"
        ):
            raise MergeError(
                "canonical source field mismatch official normalization repairs: "
                f"{sample_id}"
            )
    return prepared_index, rejection_ids, prepared_counts, source_summary


def _sum_usage(parents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usage_records = [item.get("usage") for item in parents]
    if not all(isinstance(item, dict) for item in usage_records):
        raise MergeError("parent usage records are missing")
    usages = [dict(item) for item in usage_records if isinstance(item, dict)]
    scalar_fields = (
        "actual_request_attempts",
        "failed_request_attempts",
        "successful_requests",
    )
    result: dict[str, Any] = {}
    for field in scalar_fields:
        values = [item.get(field) for item in usages]
        if not all(isinstance(value, int) and value >= 0 for value in values):
            raise MergeError(f"parent usage field is invalid: {field}")
        result[field] = sum(values)
    token_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    token_usage: dict[str, int] = {}
    for field in token_fields:
        values = [
            item.get("token_usage", {}).get(field)
            if isinstance(item.get("token_usage"), dict)
            else None
            for item in usages
        ]
        if not all(isinstance(value, int) and value >= 0 for value in values):
            raise MergeError(f"parent token usage field is invalid: {field}")
        token_usage[field] = sum(values)
    if token_usage["prompt_tokens"] + token_usage["completion_tokens"] != token_usage["total_tokens"]:
        raise MergeError("summed token usage does not reconcile")
    result.update(
        {
            "token_usage": token_usage,
            "cost_usd": None,
            "cost_note": "provider pricing not assumed; cumulative parent token usage only",
            "scope": "sum_of_parent_generation_and_verification_requests",
        }
    )
    return result


def _parent_record(
    release: ReadyRelease, *, role: str, promoted_source: Mapping[str, Any]
) -> dict[str, Any]:
    split_counts = {
        split: len(release.pairs[split]) for split in SPLITS
    }
    return {
        "role": role,
        "path": _display_path(release.root),
        "summary_sha256": _sha256_file(release.summary_path),
        "handoff_sha256": _sha256_file(release.handoff_path),
        "prompt_contract": {
            "path": _display_path(release.prompt_contract_path),
            "sha256": release.prompt_contract_sha256,
        },
        "promoted_source": dict(promoted_source),
        "split_contribution": split_counts,
        "total_contribution": sum(split_counts.values()),
    }


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_key(item, key) for item in value)
    return False


def verify_merged_release(
    root: Path,
    *,
    expected_summary: Mapping[str, Any] | None = None,
    published_root: Path | None = None,
) -> dict[str, Any]:
    """Deeply verify a staged or published merged release."""

    resolved = root.resolve()
    declared_root = (published_root or resolved).resolve()
    summary_path = resolved / "summary.json"
    receipt_path = resolved / "merge_receipt.json"
    handoff_path = resolved / "handoff.json"
    summary = _read_json(summary_path)
    receipt = _read_json(receipt_path)
    handoff = _read_json(handoff_path)
    if expected_summary is not None and summary != dict(expected_summary):
        raise MergeError("staged summary differs from the constructed summary")
    for label, record in (
        ("summary", summary),
        ("receipt", receipt),
        ("handoff", handoff),
    ):
        if record.get("schema_version") != MERGE_SCHEMA_VERSION:
            raise MergeError(f"merged {label} schema is invalid")
        if record.get("status") != READY_STATUS:
            raise MergeError(f"merged {label} status is invalid")
    if summary.get("quality_status") != READY_STATUS:
        raise MergeError("merged quality status is invalid")
    if summary.get("training_ready") is not True:
        raise MergeError("merged summary is not training-ready")
    if handoff.get("training_ready") is not True:
        raise MergeError("merged handoff is not training-ready")
    if summary.get("evaluation_eligible") is not False or handoff.get(
        "evaluation_eligible"
    ) is not False:
        raise MergeError("merged release has invalid evaluation scope")
    if _contains_key((summary, receipt, handoff), "human_review_status"):
        raise MergeError("merged metadata contains human review state")

    artifacts = summary.get("artifacts")
    split_counts = summary.get("split_counts")
    if not isinstance(artifacts, dict) or not isinstance(split_counts, dict):
        raise MergeError("merged artifacts or split counts are missing")
    all_ids: list[str] = []
    all_official: list[str] = []
    all_legacy: list[str] = []
    ids_by_split: dict[str, list[str]] = {}
    meeting_splits: dict[str, str] = {}
    physical_counts: dict[str, int] = {}
    for split in SPLITS:
        record = artifacts.get(split)
        if not isinstance(record, dict):
            raise MergeError(f"merged/{split} artifact record is missing")
        candidate_path = _relative_artifact(
            resolved,
            record.get("sft_candidate"),
            label=f"merged/{split}.sft_candidate",
        )
        manifest_path = _relative_artifact(
            resolved,
            record.get("manifest"),
            label=f"merged/{split}.manifest",
        )
        candidates = _read_jsonl(candidate_path)
        manifests = _read_jsonl(manifest_path)
        expected_rows = record.get("rows")
        split_record = split_counts.get(split)
        if (
            not isinstance(expected_rows, int)
            or len(candidates) != len(manifests)
            or len(candidates) != expected_rows
            or not isinstance(split_record, dict)
            or split_record.get("machine_pass") != expected_rows
            or split_record.get("training_rows") != expected_rows
        ):
            raise MergeError(f"merged/{split} row count mismatch")
        split_ids: list[str] = []
        for candidate, manifest in zip(candidates, manifests):
            pair = _validate_pair(
                candidate, manifest, split=split, source_label="merged"
            )
            sample_id = pair.manifest["sample_id"]
            split_ids.append(sample_id)
            all_ids.append(sample_id)
            all_official.append(pair.manifest["official_minutes_sha256"])
            all_legacy.extend(pair.manifest["legacy_sample_ids"])
            meeting_date = pair.manifest["meeting_date"]
            previous = meeting_splits.setdefault(meeting_date, split)
            if previous != split:
                raise MergeError(
                    f"merged meeting appears in multiple splits: {meeting_date}"
                )
        if split_ids != sorted(split_ids):
            raise MergeError(f"merged/{split} rows are not sorted by sample_id")
        ids_by_split[split] = split_ids
        physical_counts[split] = expected_rows

    if len(all_ids) != len(set(all_ids)):
        raise MergeError("merged release contains duplicate sample IDs")
    if len(all_official) != len(set(all_official)):
        raise MergeError("merged release contains duplicate official targets")
    if len(all_legacy) != len(set(all_legacy)):
        raise MergeError("merged release contains duplicate legacy sample IDs")
    total = len(all_ids)
    if summary.get("total_machine_pass") != total or summary.get(
        "total_training_rows"
    ) != total:
        raise MergeError("merged summary total does not match physical rows")
    if handoff.get("total_training_rows") != total or receipt.get(
        "total_training_rows"
    ) != total:
        raise MergeError("merged handoff/receipt total mismatch")
    if handoff.get("split_counts") != physical_counts or receipt.get(
        "split_counts"
    ) != physical_counts:
        raise MergeError("merged handoff/receipt split counts mismatch")

    integrity = summary.get("integrity")
    if not isinstance(integrity, dict):
        raise MergeError("merged integrity record is missing")
    if integrity.get("sample_id_sha256") != _id_digest(all_ids):
        raise MergeError("merged sample ID digest mismatch")
    if integrity.get("official_minutes_sha256_digest") != _id_digest(
        all_official
    ):
        raise MergeError("merged official target digest mismatch")
    split_id_digests = integrity.get("split_sample_id_sha256")
    if not isinstance(split_id_digests, dict):
        raise MergeError("merged split ID digests are missing")
    for split in SPLITS:
        if split_id_digests.get(split) != _id_digest(ids_by_split[split]):
            raise MergeError(f"merged/{split} sample ID digest mismatch")
    if receipt.get("integrity") != integrity or receipt.get("artifacts") != artifacts:
        raise MergeError("merged receipt does not bind integrity/artifacts")
    if receipt.get("parent_releases") != summary.get("parent_releases"):
        raise MergeError("merged receipt parent bindings mismatch")
    if handoff.get("parent_releases") != summary.get("parent_releases"):
        raise MergeError("merged handoff parent bindings mismatch")
    if handoff.get("lineage") != summary.get("lineage"):
        raise MergeError("merged handoff lineage mismatch")
    if handoff.get("teacher_model") != summary.get("teacher_model"):
        raise MergeError("merged handoff teacher mismatch")
    if handoff.get("dataset_path") != _display_path(
        declared_root / "sft_candidate"
    ):
        raise MergeError("merged handoff dataset path mismatch")
    if handoff.get("manifest_path") != _display_path(declared_root / "manifests"):
        raise MergeError("merged handoff manifest path mismatch")
    if receipt.get("output_release") != _display_path(declared_root):
        raise MergeError("merged receipt output path mismatch")
    handoff_summary = handoff.get("summary")
    handoff_receipt = handoff.get("merge_receipt")
    if not isinstance(handoff_summary, dict) or handoff_summary.get(
        "sha256"
    ) != _sha256_file(summary_path):
        raise MergeError("merged handoff does not bind summary")
    if not isinstance(handoff_receipt, dict) or handoff_receipt.get(
        "sha256"
    ) != _sha256_file(receipt_path):
        raise MergeError("merged handoff does not bind receipt")
    return summary


def merge_releases(
    *, base_root: Path, addition_root: Path, output_root: Path
) -> dict[str, Any]:
    base_path = base_root.resolve()
    addition_path = addition_root.resolve()
    output = output_root.resolve()
    if len({base_path, addition_path, output}) != 3:
        raise MergeError("base, addition, and output roots must differ")
    if _paths_overlap(output, base_path) or _paths_overlap(output, addition_path):
        raise MergeError("output root must not contain or be nested in a parent root")
    if output.exists():
        raise MergeError(f"output already exists: {output}")

    base = _load_ready_release(base_path, label="base")
    addition = _load_ready_release(addition_path, label="addition")
    base_promoted_source = _validate_promoted_source_copy(base, label="base")
    addition_promoted_source = _validate_promoted_source_copy(
        addition, label="addition"
    )
    canonical_source_root = _source_root_from_summary(base.summary, label="base")
    addition_source_root = _source_root_from_summary(
        addition.summary, label="addition"
    )
    if _paths_overlap(output, canonical_source_root) or _paths_overlap(
        output, addition_source_root
    ):
        raise MergeError("output root must not contain or be nested in a source root")
    if base.summary.get("teacher_model") != addition.summary.get("teacher_model"):
        raise MergeError("teacher model mismatch between parents")
    if base.summary.get("provider_identity") != addition.summary.get("provider_identity"):
        raise MergeError("provider identity mismatch between parents")
    if base.summary.get("lineage") != addition.summary.get("lineage"):
        raise MergeError("lineage mismatch between parents")
    if base.prompt_contract_sha256 != addition.prompt_contract_sha256:
        raise MergeError("prompt contract mismatch between parents")
    base_split_authority = (
        base.summary.get("preparation", {})
        .get("meeting_split_manifest", {})
        .get("sha256")
    )
    addition_split_authority = (
        addition.summary.get("preparation", {})
        .get("meeting_split_manifest", {})
        .get("sha256")
    )
    if base_split_authority != addition_split_authority or not isinstance(
        base_split_authority, str
    ):
        raise MergeError("meeting split authority mismatch between parents")

    prepared_index, rejection_ids, prepared_counts, canonical_summary = (
        _validate_canonical_population(base=base, addition=addition)
    )
    base_ids = {
        pair.manifest["sample_id"]
        for split in SPLITS
        for pair in base.pairs[split]
    }
    addition_ids = {
        pair.manifest["sample_id"]
        for split in SPLITS
        for pair in addition.pairs[split]
    }
    overlap = base_ids & addition_ids
    if overlap:
        raise MergeError(f"parent sample_id overlap: {sorted(overlap)[:3]}")

    all_pairs = [
        pair
        for split in SPLITS
        for release in (base, addition)
        for pair in release.pairs[split]
    ]
    official_values = [pair.manifest["official_minutes_sha256"] for pair in all_pairs]
    if len(set(official_values)) != len(official_values):
        raise MergeError("official target overlap between parents")
    legacy_values: set[str] = set()
    meeting_splits: dict[str, str] = {}
    for pair in all_pairs:
        for legacy_id in pair.manifest["legacy_sample_ids"]:
            if legacy_id in legacy_values:
                raise MergeError(f"legacy sample ID overlap: {legacy_id}")
            legacy_values.add(legacy_id)
        meeting_date = pair.manifest["meeting_date"]
        split = pair.manifest["split"]
        previous_split = meeting_splits.setdefault(meeting_date, split)
        if previous_split != split:
            raise MergeError(
                f"meeting appears in multiple splits: {meeting_date}"
            )
    merged_ids = base_ids | addition_ids
    if not merged_ids <= set(prepared_index):
        raise MergeError("merged IDs escape canonical population")
    remaining_rejections = rejection_ids - addition_ids

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        artifacts: dict[str, Any] = {}
        merged_counts: dict[str, int] = {}
        remaining_counts: dict[str, int] = {}
        for split in SPLITS:
            pairs = list(base.pairs[split]) + list(addition.pairs[split])
            pairs.sort(key=lambda pair: pair.manifest["sample_id"])
            candidate_rows = [pair.candidate for pair in pairs]
            manifest_rows = [pair.manifest for pair in pairs]
            candidate_output = staging / "sft_candidate" / f"{split}.jsonl"
            manifest_output = staging / "manifests" / f"{split}.jsonl"
            _write_jsonl(candidate_output, candidate_rows)
            _write_jsonl(manifest_output, manifest_rows)
            merged_counts[split] = len(pairs)
            remaining_counts[split] = prepared_counts[split] - len(pairs)
            artifacts[split] = {
                "rows": len(pairs),
                "sft_candidate": {
                    "path": f"sft_candidate/{split}.jsonl",
                    "sha256": _sha256_file(candidate_output),
                },
                "manifest": {
                    "path": f"manifests/{split}.jsonl",
                    "sha256": _sha256_file(manifest_output),
                },
            }

        merged_total = sum(merged_counts.values())
        canonical_total = len(prepared_index)
        remaining_total = len(remaining_rejections)
        if merged_total != len(merged_ids):
            raise MergeError("merged physical total does not match unique IDs")
        if merged_total + remaining_total != canonical_total:
            raise MergeError("merged and remaining rows do not reconcile")
        if sum(remaining_counts.values()) != remaining_total:
            raise MergeError("remaining split counts do not reconcile")

        parent_records = [
            _parent_record(
                base,
                role="base_machine_ready",
                promoted_source=base_promoted_source,
            ),
            _parent_record(
                addition,
                role="recovery_machine_ready",
                promoted_source=addition_promoted_source,
            ),
        ]
        canonical_artifacts = _canonical_artifact_records(canonical_source_root)
        split_counts = {
            split: {
                "prepared": prepared_counts[split],
                "machine_pass": merged_counts[split],
                "training_rows": merged_counts[split],
                "machine_rejected_or_pending": remaining_counts[split],
            }
            for split in SPLITS
        }
        integrity = {
            "sample_id_sha256": _id_digest(sorted(merged_ids)),
            "base_sample_id_sha256": _id_digest(sorted(base_ids)),
            "addition_sample_id_sha256": _id_digest(sorted(addition_ids)),
            "canonical_sample_id_sha256": _id_digest(sorted(prepared_index)),
            "original_rejection_sample_id_sha256": _id_digest(
                sorted(rejection_ids)
            ),
            "remaining_rejection_sample_id_sha256": _id_digest(
                sorted(remaining_rejections)
            ),
            "split_sample_id_sha256": {
                split: _id_digest(
                    sorted(
                        pair.manifest["sample_id"]
                        for pair in all_pairs
                        if pair.manifest["split"] == split
                    )
                )
                for split in SPLITS
            },
            "official_minutes_sha256_digest": _id_digest(official_values),
            "sample_ids_unique": True,
            "legacy_sample_ids_unique": True,
            "official_targets_unique": True,
            "meeting_split_conflicts": 0,
            "candidate_manifest_bindings_verified": merged_total,
            "canonical_population_partition_verified": True,
            "prompt_contract_sha256": base.prompt_contract_sha256,
            "meeting_split_manifest_sha256": base_split_authority,
        }
        summary = {
            "schema_version": MERGE_SCHEMA_VERSION,
            "status": READY_STATUS,
            "quality_status": READY_STATUS,
            "phase": "merge",
            "acquisition_mode": "quality_gated_recovery_union",
            "teacher_model": base.summary.get("teacher_model"),
            "provider_identity": base.summary.get("provider_identity"),
            "training_ready": True,
            "evaluation_eligible": False,
            "lineage": base.summary.get("lineage"),
            "failure_count": 0,
            "total_prepared": canonical_total,
            "total_machine_pass": merged_total,
            "total_training_rows": merged_total,
            "total_machine_rejected_or_pending": remaining_total,
            "recovered_rows": len(addition_ids),
            "remaining_original_rejections": remaining_total,
            "split_counts": split_counts,
            "artifacts": artifacts,
            "merge_policy": {
                "key": "sample_id",
                "sort": "sample_id_ascending_with_candidate_manifest_pairs_bound",
                "conflict_policy": "fail_closed",
                "addition_must_be_in_original_rejection_ledger": True,
                "machine_pass_required": True,
                "human_review_required": False,
                "evaluation_eligible": False,
                "recovery_manifest_lineage": (
                    "retained_from_recovery_staging; canonical target identity "
                    "verified separately"
                ),
            },
            "parent_releases": parent_records,
            "canonical_source_release": {
                "path": _display_path(canonical_source_root),
                "summary_sha256": _sha256_file(canonical_source_root / "summary.json"),
                "total_prepared": canonical_total,
                "original_machine_rejections": len(rejection_ids),
                "artifacts": canonical_artifacts,
            },
            "usage": _sum_usage((base.summary, addition.summary)),
            "integrity": integrity,
        }
        _write_json(staging / "summary.json", summary)

        receipt = {
            "schema_version": MERGE_SCHEMA_VERSION,
            "status": READY_STATUS,
            "output_release": _display_path(output),
            "parent_releases": parent_records,
            "split_counts": merged_counts,
            "total_training_rows": merged_total,
            "recovered_rows": len(addition_ids),
            "remaining_original_rejections": remaining_total,
            "integrity": integrity,
            "artifacts": artifacts,
        }
        _write_json(staging / "merge_receipt.json", receipt)
        handoff = {
            "schema_version": MERGE_SCHEMA_VERSION,
            "status": READY_STATUS,
            "dataset_path": _display_path(output / "sft_candidate"),
            "manifest_path": _display_path(output / "manifests"),
            "teacher_model": base.summary.get("teacher_model"),
            "split_counts": merged_counts,
            "total_training_rows": merged_total,
            "training_ready": True,
            "evaluation_eligible": False,
            "lineage": base.summary.get("lineage"),
            "parent_releases": parent_records,
            "summary": {
                "path": "summary.json",
                "sha256": _sha256_file(staging / "summary.json"),
            },
            "merge_receipt": {
                "path": "merge_receipt.json",
                "sha256": _sha256_file(staging / "merge_receipt.json"),
            },
        }
        _write_json(staging / "handoff.json", handoff)
        verify_merged_release(
            staging,
            expected_summary=summary,
            published_root=output,
        )

        fresh_base = _load_ready_release(base_path, label="base")
        fresh_addition = _load_ready_release(addition_path, label="addition")
        fresh_parent_records = [
            _parent_record(
                fresh_base,
                role="base_machine_ready",
                promoted_source=_validate_promoted_source_copy(
                    fresh_base, label="base"
                ),
            ),
            _parent_record(
                fresh_addition,
                role="recovery_machine_ready",
                promoted_source=_validate_promoted_source_copy(
                    fresh_addition, label="addition"
                ),
            ),
        ]
        if fresh_parent_records != parent_records:
            raise MergeError("a parent release drifted during merge construction")
        if _canonical_artifact_records(canonical_source_root) != canonical_artifacts:
            raise MergeError("canonical source artifacts drifted during merge construction")

        _fsync_directory(staging / "sft_candidate")
        _fsync_directory(staging / "manifests")
        _fsync_directory(staging)
        _rename_noreplace(staging, output)
        _fsync_directory(output.parent)
        verify_merged_release(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge base and recovery CHK2 machine-ready releases."
    )
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--addition-root", type=Path, default=DEFAULT_ADDITION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="deeply verify an existing --output-root without writing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only:
        summary = verify_merged_release(args.output_root)
    else:
        summary = merge_releases(
            base_root=args.base_root,
            addition_root=args.addition_root,
            output_root=args.output_root,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
