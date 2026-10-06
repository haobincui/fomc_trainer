"""Frozen data contract for the merged pre/post-2008 CHK3 beta panel.

The module does not acquire or impute data.  It accepts only two sealed,
harmonized Core-8 evaluation releases whose evidence cutoff is the calendar
day before the *official meeting start*.  This avoids pooling the historical
start-date D-1 panel with the older post-2008 decision-date D-1 artifacts.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from jobs.eval.build_chk3_external_holdout_1993_2008 import CORE_TOPICS
from jobs.retrain_v2 import probe_chk3_sft_degeneration as native_probe
from open_r1.provenance import sha256_file, sha256_text
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


ROOT = Path(__file__).resolve().parents[2]
PRE_RELEASE_MANIFEST = ROOT / (
    "dataset/processed/retrain_v2/"
    "chk3_minutes_external_holdout_1993_2008_all_regular_v1/"
    "release_manifest.json"
)
POST_RELEASE_MANIFEST = ROOT / (
    "dataset/processed/retrain_v2/"
    "chk3_minutes_post2008_2009_2025_fixed_core8_d1_v1/"
    "release_manifest.json"
)
PRE_RELEASE_MANIFEST_SHA256 = (
    "82b045866d9ed6ccbc0d4f00014bffc97694859c2827215309dc8eabe5406937"
)

EXPECTED_MEETINGS_PER_ERA = 128
EXPECTED_ROWS_PER_ERA = EXPECTED_MEETINGS_PER_ERA * len(CORE_TOPICS)
EXPECTED_MEETINGS = 2 * EXPECTED_MEETINGS_PER_ERA
EXPECTED_ROWS = 2 * EXPECTED_ROWS_PER_ERA
EVIDENCE_CUTOFF_POLICY = "meeting_start_date_minus_1_calendar_day"
TASK_CONTRACT = "analysis_to_formal_fomc_minutes_prose"
MERGED_RELEASE_ID = "chk3-beta-core8-merged-1993-2025-n2048-v1"
CP318_SELECTION_N12_MANIFEST = ROOT / (
    "docs/summary/20260811T003000Z/"
    "chk3_native_analysis_to_minutes_cp250_eval/samples_n12.json"
)
CP318_SELECTION_N12_MANIFEST_SHA256 = (
    "371d29601e343acf98cac6173663d842ead9acaa4a454991d512b3f00bf5ee77"
)
CP318_SELECTION_N12_PAYLOAD_SHA256 = (
    "691edd595422cb0c384728d88c86d361628da7914c8e22c32a9f6967b1e88ee6"
)
CP318_SELECTION_EXPOSED_MEETINGS = (
    "2023-07-26",
    "2023-09-20",
    "2024-01-31",
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-07",
    "2024-12-18",
    "2025-01-29",
)
PROMPT_PREFIX = "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
ALLOWED_POST_SPLITS = frozenset({"train", "validation", "test"})
ALLOWED_QA_SPLITS = frozenset({"train", "eval", "test"})
ERA_ORDER = ("pre2009_external", "post2008_chk3_release")


class MergedCore8ContractError(RuntimeError):
    """A source release cannot enter the harmonized merged panel."""


@dataclass(frozen=True)
class HarmonizedSources:
    """Deep-validated source rows and immutable source bindings."""

    rows: tuple[dict[str, Any], ...]
    meeting_ids: tuple[str, ...]
    source_bindings: Mapping[str, Mapping[str, Any]]
    topic_counts: Mapping[str, int]
    era_counts: Mapping[str, int]
    sensitivity_meetings: tuple[str, ...]
    source_role_counts: Mapping[str, Mapping[str, Mapping[str, int]]]
    cp318_selection_exposure: Mapping[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MergedCore8ContractError(f"invalid JSON artifact: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MergedCore8ContractError(f"JSON artifact is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MergedCore8ContractError(f"cannot read JSONL: {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MergedCore8ContractError(
                f"invalid JSONL row: {path}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise MergedCore8ContractError(
                f"JSONL row is not an object: {path}:{line_number}"
            )
        rows.append(row)
    return rows


def _load_cp318_selection_exposure() -> dict[str, Any]:
    unresolved_path = CP318_SELECTION_N12_MANIFEST.expanduser()
    if unresolved_path.is_symlink():
        raise MergedCore8ContractError("cp318 N12 selection manifest is a symlink")
    path = unresolved_path.resolve()
    if not path.is_file() or sha256_file(path) != CP318_SELECTION_N12_MANIFEST_SHA256:
        raise MergedCore8ContractError("cp318 N12 selection manifest binding drift")
    manifest = _read_json(path)
    try:
        payload_sha = validate_manifest_integrity(
            manifest,
            expected_payload_sha256=CP318_SELECTION_N12_PAYLOAD_SHA256,
        )
    except Exception as exc:
        raise MergedCore8ContractError(
            f"cp318 N12 selection manifest integrity failed: {exc}"
        ) from exc
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != 12:
        raise MergedCore8ContractError("cp318 selection manifest is not N12")
    pattern = re.compile(r"^chk1-analysis-(\d{4}-\d{2}-\d{2})-[0-9a-f]+$")
    meetings: set[str] = set()
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise MergedCore8ContractError("cp318 N12 sample is not an object")
        match = pattern.fullmatch(str(sample.get("sample_id") or ""))
        if match is None:
            raise MergedCore8ContractError("cp318 N12 sample ID is not parseable")
        meetings.add(match.group(1))
    if tuple(sorted(meetings)) != CP318_SELECTION_EXPOSED_MEETINGS:
        raise MergedCore8ContractError("cp318 N12 exposed meeting inventory drift")
    return {
        "selection_manifest": {
            "path": str(path),
            "sha256": CP318_SELECTION_N12_MANIFEST_SHA256,
            "bytes": path.stat().st_size,
            "payload_sha256": payload_sha,
            "rows": 12,
        },
        "exposed_meetings": list(CP318_SELECTION_EXPOSED_MEETINGS),
        "meeting_count": 9,
        "prompt_count": 72,
        "generation_rows_per_model": 720,
        "interpretation": "post_selection_exposure_requires_separate_and_excluded_sensitivity_views",
    }


def _bound_release_file(
    *, manifest_path: Path, manifest: Mapping[str, Any], relative: str, rows: int
) -> tuple[Path, Mapping[str, Any]]:
    record = (manifest.get("files") or {}).get(relative)
    if not isinstance(record, Mapping):
        raise MergedCore8ContractError(f"release does not bind {relative}")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise MergedCore8ContractError(f"release has invalid path for {relative}")
    release_root = manifest_path.parent.resolve()
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise MergedCore8ContractError(f"release has unsafe path for {relative}")
    unresolved_path = release_root / relative_path
    component = release_root
    for part in relative_path.parts:
        component /= part
        if component.is_symlink():
            raise MergedCore8ContractError(
                f"release file path contains a symlink: {relative}"
            )
    path = unresolved_path.resolve()
    try:
        path.relative_to(release_root)
    except ValueError as exc:
        raise MergedCore8ContractError(
            f"release file escapes release root: {relative}"
        ) from exc
    if (
        not path.is_file()
        or record.get("rows") != rows
        or record.get("sha256") != sha256_file(path)
        or record.get("bytes") != path.stat().st_size
    ):
        raise MergedCore8ContractError(f"release file binding drift: {relative}")
    return path, record


def _iso_date(value: Any, *, field: str, context: str) -> str:
    if not isinstance(value, str):
        raise MergedCore8ContractError(f"{context} has invalid {field}")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise MergedCore8ContractError(f"{context} has invalid {field}") from exc


def _post_split_role(row: Mapping[str, Any], roster: Mapping[str, Any]) -> str:
    for key in (
        "original_post_split_role",
        "original_split_role",
        "source_split_role",
        "source_split",
    ):
        value = row.get(key)
        if value is None:
            value = roster.get(key)
        if isinstance(value, str) and value:
            if value not in ALLOWED_POST_SPLITS:
                raise MergedCore8ContractError(
                    f"post meeting has unsupported original split role: {value!r}"
                )
            return value
    raise MergedCore8ContractError("post meeting is missing its original split role")


def _sensitivity_metadata(
    row: Mapping[str, Any], roster: Mapping[str, Any], *, era: str
) -> tuple[bool, str | None]:
    for key in ("sensitivity_flag", "sensitivity_meeting"):
        value = row.get(key)
        if value is None:
            value = roster.get(key)
        if value is not None:
            if not isinstance(value, bool):
                raise MergedCore8ContractError(f"invalid {key}: expected boolean")
            reason = row.get("sensitivity_reason")
            if reason is None:
                reason = roster.get("sensitivity_reason")
            if reason is not None and (not isinstance(reason, str) or not reason):
                raise MergedCore8ContractError(
                    "sensitivity_reason must be a nonempty string or null"
                )
            if era == "post2008_chk3_release":
                if value and reason is None:
                    raise MergedCore8ContractError(
                        "post sensitivity meeting has no sensitivity reason"
                    )
                if not value and reason is not None:
                    raise MergedCore8ContractError(
                        "post nonsensitivity meeting has a sensitivity reason"
                    )
            return value, reason
    meeting_type = str(row.get("meeting_type") or roster.get("meeting_type") or "")
    scheduled = row.get("scheduled", roster.get("scheduled"))
    # The frozen fallback is explicit and reproducible for the older pre release,
    # which predates the sensitivity flag but records type/scheduled status.
    flag = meeting_type != "regular" or scheduled is not True
    return flag, "nonregular_or_unscheduled_meeting" if flag else None


def _validate_release_identity(
    *, manifest_path: Path, manifest: Mapping[str, Any], era: str
) -> str:
    try:
        payload_sha = validate_manifest_integrity(manifest)
    except Exception as exc:
        raise MergedCore8ContractError(
            f"{era} release integrity failed: {exc}"
        ) from exc
    required = {
        "schema_version": "chk3-external-evaluation-release-v1",
        "status": "passed",
        "immutable": True,
        "evaluation_only": True,
        "trainable": False,
        "checkpoint_selection_allowed": False,
        "promotable": False,
        "task_contract": TASK_CONTRACT,
        "meeting_count": EXPECTED_MEETINGS_PER_ERA,
        "core_topic_count": len(CORE_TOPICS),
        "core8_rows": EXPECTED_ROWS_PER_ERA,
        "evidence_cutoff_policy": EVIDENCE_CUTOFF_POLICY,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise MergedCore8ContractError(
                f"{era} release contract drift at {key}: "
                f"expected={expected!r}, observed={manifest.get(key)!r}"
            )
    if manifest_path.is_symlink():
        raise MergedCore8ContractError(f"{era} release manifest is a symlink")
    return payload_sha


def _validate_one_release(
    *, manifest_path: Path, era: str, pinned_sha256: str | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unresolved_manifest = manifest_path.expanduser()
    if unresolved_manifest.is_symlink():
        raise MergedCore8ContractError(f"{era} release manifest is a symlink")
    manifest_path = unresolved_manifest.resolve()
    if not manifest_path.is_file():
        raise MergedCore8ContractError(f"{era} release is missing: {manifest_path}")
    observed_manifest_sha = sha256_file(manifest_path)
    if pinned_sha256 is not None and observed_manifest_sha != pinned_sha256:
        raise MergedCore8ContractError(f"{era} release manifest SHA-256 drift")
    manifest = _read_json(manifest_path)
    payload_sha = _validate_release_identity(
        manifest_path=manifest_path, manifest=manifest, era=era
    )
    roster_path, roster_binding = _bound_release_file(
        manifest_path=manifest_path,
        manifest=manifest,
        relative="official_meeting_roster.jsonl",
        rows=EXPECTED_MEETINGS_PER_ERA,
    )
    core_path, core_binding = _bound_release_file(
        manifest_path=manifest_path,
        manifest=manifest,
        relative="panels/core8.jsonl",
        rows=EXPECTED_ROWS_PER_ERA,
    )
    roster_rows = _read_jsonl(roster_path)
    core_rows = _read_jsonl(core_path)
    roster_by_id: dict[str, dict[str, Any]] = {}
    for index, roster in enumerate(roster_rows, 1):
        meeting_id = roster.get("meeting_id")
        if (
            not isinstance(meeting_id, str)
            or not meeting_id
            or meeting_id in roster_by_id
        ):
            raise MergedCore8ContractError(
                f"{era} roster has invalid/duplicate meeting at line {index}"
            )
        roster_by_id[meeting_id] = roster

    seen_samples: set[str] = set()
    seen_keys: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for line_number, row in enumerate(core_rows, 1):
        context = f"{era} core8 line {line_number}"
        sample_id = row.get("sample_id")
        meeting_id = row.get("meeting_id")
        topic = row.get("topic")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or sample_id in seen_samples
            or not isinstance(meeting_id, str)
            or meeting_id not in roster_by_id
            or topic not in CORE_TOPICS
            or (meeting_id, str(topic)) in seen_keys
        ):
            raise MergedCore8ContractError(f"{context} has invalid identity/key")
        seen_samples.add(sample_id)
        seen_keys.add((meeting_id, str(topic)))
        roster = roster_by_id[meeting_id]
        start = _iso_date(row.get("meeting_start_date"), field="start", context=context)
        end = _iso_date(row.get("meeting_end_date"), field="end", context=context)
        cutoff = _iso_date(row.get("evidence_cutoff"), field="cutoff", context=context)
        if any(
            value != roster.get(key)
            for value, key in (
                (start, "meeting_start_date"),
                (end, "meeting_end_date"),
                (cutoff, "evidence_cutoff"),
            )
        ):
            raise MergedCore8ContractError(f"{context} disagrees with official roster")
        if date.fromisoformat(cutoff) != date.fromisoformat(start) - timedelta(days=1):
            raise MergedCore8ContractError(f"{context} is not official-start-date D-1")
        if date.fromisoformat(end) < date.fromisoformat(start):
            raise MergedCore8ContractError(f"{context} has end before start")
        prompt = row.get("prompt")
        analysis = row.get("source_analysis")
        reference = row.get("reference_minutes")
        source_id = row.get("source_id")
        if not all(
            isinstance(value, str) and value
            for value in (prompt, analysis, reference, source_id)
        ):
            raise MergedCore8ContractError(
                f"{context} has incomplete native text/source"
            )
        if (
            not str(prompt).startswith(PROMPT_PREFIX)
            or native_probe.extract_source_analysis(str(prompt)) != analysis
            or sha256_text(str(prompt)) != row.get("prompt_sha256")
            or sha256_text(str(analysis)) != row.get("source_analysis_sha256")
            or sha256_text(str(reference)) != row.get("reference_minutes_sha256")
        ):
            raise MergedCore8ContractError(f"{context} native prompt/hash drift")
        meeting_type = row.get("meeting_type") or roster.get("meeting_type")
        if not isinstance(meeting_type, str) or not meeting_type:
            raise MergedCore8ContractError(f"{context} has no meeting type")
        scheduled = row.get("scheduled")
        if scheduled is None:
            scheduled = roster.get("scheduled")
        if not isinstance(scheduled, bool):
            raise MergedCore8ContractError(f"{context} has no boolean scheduled flag")
        original_post_split = (
            _post_split_role(row, roster) if era == "post2008_chk3_release" else None
        )
        original_qa_split: str | None = None
        if era == "post2008_chk3_release":
            original_qa_split = row.get("original_qa_split")
            if original_qa_split is None:
                original_qa_split = roster.get("original_qa_split")
            if original_qa_split not in ALLOWED_QA_SPLITS:
                raise MergedCore8ContractError(
                    f"post meeting has unsupported original QA split: {original_qa_split!r}"
                )
            expected_qa = (
                "eval" if original_post_split == "validation" else original_post_split
            )
            if original_qa_split != expected_qa:
                raise MergedCore8ContractError(
                    "post formal/original QA split roles disagree"
                )
        origin_split_role = original_post_split or "external_holdout"
        sensitivity, sensitivity_reason = _sensitivity_metadata(row, roster, era=era)
        normalized.append(
            {
                "era": era,
                "source_split": origin_split_role,
                "original_post_split_role": original_post_split,
                "original_qa_split": original_qa_split,
                "meeting_type": meeting_type,
                "sensitivity_flag": sensitivity,
                "sensitivity_reason": sensitivity_reason,
                "scheduled": scheduled,
                "meeting_id": meeting_id,
                "meeting_start_date": start,
                "meeting_end_date": end,
                "evidence_cutoff": cutoff,
                "topic": str(topic),
                "topic_order": CORE_TOPICS.index(str(topic)),
                "cp318_selection_exposed": end in CP318_SELECTION_EXPOSED_MEETINGS,
                "source_sample_id": sample_id,
                "source_id": source_id,
                "source_core8_line_number": line_number,
                "prompt": prompt,
                "source_analysis": analysis,
                "reference_minutes": reference,
                "prompt_sha256": row["prompt_sha256"],
                "source_analysis_sha256": row["source_analysis_sha256"],
                "reference_minutes_sha256": row["reference_minutes_sha256"],
                "topic_evidence_sha256": row.get("topic_evidence_sha256"),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        grouped[str(row["meeting_id"])].append(row)
    if len(grouped) != EXPECTED_MEETINGS_PER_ERA:
        raise MergedCore8ContractError(f"{era} meeting closure is not N128")
    for meeting_id, rows in grouped.items():
        if (
            len(rows) != len(CORE_TOPICS)
            or {str(row["topic"]) for row in rows} != set(CORE_TOPICS)
            or len({str(row["meeting_end_date"]) for row in rows}) != 1
        ):
            raise MergedCore8ContractError(
                f"{era} meeting is not exactly Core8: {meeting_id}"
            )
        if len({row["source_split"] for row in rows}) != 1:
            raise MergedCore8ContractError(
                f"{era} meeting crosses original split roles: {meeting_id}"
            )
    binding = {
        "path": str(manifest_path),
        "sha256": observed_manifest_sha,
        "payload_sha256": payload_sha,
        "release_id": manifest.get("release_id"),
        "schema_version": manifest.get("schema_version"),
        "core8": {
            "path": str(core_path),
            "sha256": core_binding["sha256"],
            "bytes": core_binding["bytes"],
            "rows": core_binding["rows"],
        },
        "official_roster": {
            "path": str(roster_path),
            "sha256": roster_binding["sha256"],
            "bytes": roster_binding["bytes"],
            "rows": roster_binding["rows"],
        },
        "evidence_cutoff_policy": manifest.get("evidence_cutoff_policy"),
    }
    return normalized, binding


def load_harmonized_sources(
    *,
    pre_release_manifest: Path = PRE_RELEASE_MANIFEST,
    post_release_manifest: Path = POST_RELEASE_MANIFEST,
    pre_release_sha256: str | None = None,
    post_release_sha256: str | None = None,
) -> HarmonizedSources:
    """Load and deep-validate exactly 256 meetings x frozen Core8."""

    pre_rows, pre_binding = _validate_one_release(
        manifest_path=pre_release_manifest,
        era="pre2009_external",
        pinned_sha256=pre_release_sha256
        or (
            PRE_RELEASE_MANIFEST_SHA256
            if pre_release_manifest.expanduser().resolve()
            == PRE_RELEASE_MANIFEST.expanduser().resolve()
            else None
        ),
    )
    post_rows, post_binding = _validate_one_release(
        manifest_path=post_release_manifest,
        era="post2008_chk3_release",
        pinned_sha256=post_release_sha256,
    )
    pre_end_dates = {str(row["meeting_end_date"]) for row in pre_rows}
    post_end_dates = {str(row["meeting_end_date"]) for row in post_rows}
    if pre_end_dates & post_end_dates:
        raise MergedCore8ContractError("pre/post meeting-end dates overlap")
    if max(pre_end_dates) >= min(str(row["meeting_start_date"]) for row in post_rows):
        raise MergedCore8ContractError("pre/post eras are not chronologically disjoint")
    rows = sorted(
        [*pre_rows, *post_rows],
        key=lambda row: (
            str(row["meeting_end_date"]),
            int(row["topic_order"]),
            str(row["source_sample_id"]),
        ),
    )
    if len(rows) != EXPECTED_ROWS:
        raise MergedCore8ContractError("merged row closure is not N2048")
    meeting_ids = tuple(sorted({str(row["meeting_end_date"]) for row in rows}))
    if len(meeting_ids) != EXPECTED_MEETINGS:
        raise MergedCore8ContractError("merged meeting closure is not N256")
    by_end: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_end[str(row["meeting_end_date"])].append(row)
    for meeting_id in meeting_ids:
        observed_topics = tuple(str(row["topic"]) for row in by_end[meeting_id])
        if observed_topics != CORE_TOPICS:
            raise MergedCore8ContractError(
                f"merged canonical topic order drift for {meeting_id}"
            )
    topic_counts = Counter(str(row["topic"]) for row in rows)
    if set(topic_counts.values()) != {EXPECTED_MEETINGS}:
        raise MergedCore8ContractError("merged topic allocation is not 256 each")
    era_counts = Counter(str(row["era"]) for row in rows)
    if dict(era_counts) != {
        "pre2009_external": EXPECTED_ROWS_PER_ERA,
        "post2008_chk3_release": EXPECTED_ROWS_PER_ERA,
    }:
        raise MergedCore8ContractError("merged era allocation drift")
    sensitivity_meetings = tuple(
        sorted(
            {
                str(row["meeting_end_date"])
                for row in rows
                if bool(row["sensitivity_flag"])
            }
        )
    )
    post_role_meetings: dict[str, set[str]] = defaultdict(set)
    post_role_prompts: Counter[str] = Counter()
    for row in rows:
        if row["era"] == "post2008_chk3_release":
            role = str(row["source_split"])
            post_role_meetings[role].add(str(row["meeting_end_date"]))
            post_role_prompts[role] += 1
    observed_post_meetings = {
        role: len(post_role_meetings.get(role, set()))
        for role in ("train", "validation", "test")
    }
    observed_post_prompts = {
        role: post_role_prompts.get(role, 0) for role in ("train", "validation", "test")
    }
    if observed_post_meetings != {"train": 102, "validation": 13, "test": 13}:
        raise MergedCore8ContractError(
            f"post original meeting-role counts drift: {observed_post_meetings}"
        )
    if observed_post_prompts != {"train": 816, "validation": 104, "test": 104}:
        raise MergedCore8ContractError(
            f"post original prompt-role counts drift: {observed_post_prompts}"
        )
    source_role_counts = {
        "pre2009_external": {
            "external_holdout": {
                "meetings": EXPECTED_MEETINGS_PER_ERA,
                "prompts": EXPECTED_ROWS_PER_ERA,
            }
        },
        "post2008_chk3_release": {
            role: {
                "meetings": observed_post_meetings[role],
                "prompts": observed_post_prompts[role],
            }
            for role in ("train", "validation", "test")
        },
    }
    exposure = _load_cp318_selection_exposure()
    exposed_rows = [row for row in rows if row["cp318_selection_exposed"]]
    if (
        len(exposed_rows) != 72
        or len({str(row["meeting_end_date"]) for row in exposed_rows}) != 9
        or any(row["era"] != "post2008_chk3_release" for row in exposed_rows)
    ):
        raise MergedCore8ContractError("cp318 merged-panel exposure closure drift")
    return HarmonizedSources(
        rows=tuple(rows),
        meeting_ids=meeting_ids,
        source_bindings={
            "pre2009_external": pre_binding,
            "post2008_chk3_release": post_binding,
        },
        topic_counts=dict(sorted(topic_counts.items())),
        era_counts=dict(sorted(era_counts.items())),
        sensitivity_meetings=sensitivity_meetings,
        source_role_counts=source_role_counts,
        cp318_selection_exposure=exposure,
    )


def row_source_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the source metadata persisted in every sample/generation row."""

    return {
        key: row[key]
        for key in (
            "era",
            "source_split",
            "original_post_split_role",
            "original_qa_split",
            "meeting_type",
            "sensitivity_flag",
            "sensitivity_reason",
            "scheduled",
            "meeting_start_date",
            "meeting_end_date",
            "evidence_cutoff",
            "topic",
            "topic_order",
            "cp318_selection_exposed",
            "source_sample_id",
            "source_id",
            "source_core8_line_number",
            "prompt_sha256",
            "source_analysis_sha256",
            "reference_minutes_sha256",
            "topic_evidence_sha256",
        )
    }


def merged_sample_id(row: Mapping[str, Any], source_binding: Mapping[str, Any]) -> str:
    """Derive the stable, meeting-parseable compatibility sample ID."""

    payload = "\0".join(
        (
            MERGED_RELEASE_ID,
            str(row["era"]),
            str(source_binding["sha256"]),
            str(row["source_sample_id"]),
        )
    )
    return f"chk3-beta-core8-{row['meeting_end_date']}-{sha256_text(payload)[:24]}"


__all__ = [
    "ALLOWED_POST_SPLITS",
    "ALLOWED_QA_SPLITS",
    "CP318_SELECTION_EXPOSED_MEETINGS",
    "CP318_SELECTION_N12_MANIFEST",
    "CP318_SELECTION_N12_MANIFEST_SHA256",
    "CP318_SELECTION_N12_PAYLOAD_SHA256",
    "CORE_TOPICS",
    "EVIDENCE_CUTOFF_POLICY",
    "ERA_ORDER",
    "EXPECTED_MEETINGS",
    "EXPECTED_MEETINGS_PER_ERA",
    "EXPECTED_ROWS",
    "EXPECTED_ROWS_PER_ERA",
    "HarmonizedSources",
    "MERGED_RELEASE_ID",
    "MergedCore8ContractError",
    "POST_RELEASE_MANIFEST",
    "PRE_RELEASE_MANIFEST",
    "PRE_RELEASE_MANIFEST_SHA256",
    "PROMPT_PREFIX",
    "TASK_CONTRACT",
    "load_harmonized_sources",
    "merged_sample_id",
    "row_source_metadata",
]
