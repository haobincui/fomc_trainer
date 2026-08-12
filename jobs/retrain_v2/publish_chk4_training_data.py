"""Validate and immutably publish chk4 Decision-SFT and Decision-GRPO data.

The source materialization is kept unchanged.  This publisher performs a
second, independent row-level audit, applies four deterministic reasoning-only
transport repairs, validates the real DeepSeek tokenizer contract, and
publishes with ``renameat2(RENAME_NOREPLACE)``.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs.generation.generate_chk4_meeting_briefs import (
    DATE_RE as BRIEF_DATE_RE,
    NUMBER_RE as BRIEF_NUMBER_RE,
    USER_PREFIX as BRIEF_USER_PREFIX,
)
from jobs.generation.generate_chk4_sft_targets import (
    STUDENT_SYSTEM_PROMPT,
    USER_PROMPT_PREFIX,
)
from open_r1.trainer.sft_prompt_renderer import (
    render_sft_prompt,
    tokenize_sft_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "output/data/retrain_v2/chk4/training_data_v2"
DEFAULT_TEACHER = REPO_ROOT / "output/data/retrain_v2/chk4/deepseek_v4_pro_v2"
DEFAULT_BRIEFS = REPO_ROOT / "output/data/retrain_v2/chk4/meeting_decision_briefs_v1"
DEFAULT_TOKENIZER = (
    REPO_ROOT
    / "output/training/retrain_v2/"
    "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "dataset/processed/retrain_v2/"
    "chk4_decision_warmstart_grpo_core_v3_20260810"
)
SFT_RENDERER_SOURCE = REPO_ROOT / "src/open_r1/trainer/sft_prompt_renderer.py"

SPLITS = ("train", "validation", "test")
EXPECTED_UNIQUE = {"train": 102, "validation": 13, "test": 13}
EXPECTED_PHYSICAL = {"train": 141, "validation": 13, "test": 13}
EXPECTED_REPEAT_FACTORS = {"cut": 4, "hold": 1, "hike": 4}
EXPECTED_UNIQUE_TRAIN_DIRECTIONS = {"cut": 4, "hold": 89, "hike": 9}
EXPECTED_PHYSICAL_TRAIN_DIRECTIONS = {"cut": 16, "hold": 89, "hike": 36}

SOURCE_SCHEMA = "chk4-decision-training-data-v1"
REPEAT_SCHEMA = "chk4-decision-repeat-manifest-v1"
RELEASE_SCHEMA = "chk4-decision-training-release-v1"
AUDIT_SCHEMA = "chk4-decision-data-quality-v1"
HANDOFF_SCHEMA = "chk4-decision-training-handoff-v1"
REPAIR_SCHEMA = "chk4-decision-reasoning-repair-v1"
INPUT_CONTRACT_SCHEMA = "chk4-target-decision-blind-input-contract-v1"
BOUNDARY = "\n</think>\n"
MAX_SFT_TOKENS = 3072
MAX_GRPO_PROMPT_TOKENS = 2560
MAX_GRPO_COMPLETION_TOKENS = 512

_SHA_RE = re.compile(r"[0-9a-f]{64}")
_ROW_RE = re.compile(r"row-[0-9a-f]{24}")
_DATE_RE = re.compile(r"\b(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b")
_DIGIT_RE = re.compile(r"\d")
_EXPLICIT_REASONING_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December|first quarter|second quarter|third quarter|fourth quarter|"
    r"first half|second half)\b"
)
_SPELLED_NUMBER_RE = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.I,
)
_CONTROL_RE = re.compile(r"</?(?:think|answer)>|<\|[^>]+\|>", re.I)
_PROMPT_FORBIDDEN = (
    "current_rate",
    "rate_change",
    "gold label",
    "teacher target",
    "teacher-only canonical decision",
    "actual committee action",
    "answer key",
)
_TARGET_OUTCOME_PATTERNS = (
    re.compile(
        r"\b(?:at|for)\s+(?:the\s+)?(?:target|current|this)\s+meeting\b",
        re.I,
    ),
    re.compile(r"(?:[\"'])?\b(?:direction|magnitude_bp)\b(?:[\"'])?\s*[:=]", re.I),
    re.compile(r"\b(?:outcome|meeting action|answer key|gold label)\s*:", re.I),
    re.compile(
        r"\btoday\b.{0,80}\b(?:cut|lowered|reduced|raised|increased|hiked|held|left)\b"
        r".{0,40}\b(?:federal funds rate|policy rate|rates?|target range)\b",
        re.I,
    ),
)
_PRIOR_MEETING_CONTEXT_RE = re.compile(
    r"\b(?:at|during|after)\s+(?:the\s+)?(?:prior|previous|earlier)\s+meeting\b",
    re.I,
)
_CALENDAR_CONTEXT_RE = re.compile(
    r"\b(?:in\s+(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December|19\d{2}|20\d{2})|by\s+(?:late|early|mid)[- ]?[A-Za-z]+)\b",
    re.I,
)
_CONTEXTUAL_TARGET_ACTION_PATTERNS = (
    re.compile(
        r"\b(?:Committee|FOMC|policymakers|officials)\b.{0,80}\b"
        r"(?:voted|decided|elected|opted|agreed|chose|should|would)\b.{0,40}\b"
        r"(?:cut|lower|reduce|raise|increase|hike|hold|keep|leave)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:Committee|FOMC|policymakers|officials)\b.{0,80}\b"
        r"(?:cut|lowered|reduced|raised|increased|hiked|held|left|maintained|kept)\b"
        r".{0,50}\b(?:federal funds rate|policy rate|rates?|target range)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:Federal Reserve|Fed|central bank)\b.{0,60}\b"
        r"(?:voted|decided|cut|lowered|reduced|raised|increased|hiked|left)\b"
        r".{0,60}\b(?:federal funds rate|policy rate|rates?|target range)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:target range|federal funds rate|policy rate|rates?)\b.{0,80}\b"
        r"(?:was|were|has been|had been)\s+"
        r"(?:cut|lowered|reduced|raised|increased|hiked|left unchanged)\b",
        re.I,
    ),
    re.compile(
        r"\bno change was made to (?:the )?(?:target range|federal funds rate|rates?)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:the )?(?:vote|decision)\s+(?:resulted in|was|is)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:\d+|[a-z-]+)\s+basis[- ]point\s+(?:rate\s+)?"
        r"(?:cut|reduction|increase|hike)\s+(?:was|is|has been)\s+"
        r"(?:announced|approved|implemented)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:recommend|recommended|should|would)\s+"
        r"(?:cut|lower|reduce|raise|increase|hike|hold)\s+"
        r"(?:the\s+)?(?:federal funds rate|policy rate|rates?|target range)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:a\s+)?(?:rate\s+)?(?:cut|hike|hold)(?:ing)?\s+"
        r"(?:is|was|would be)\s+"
        r"(?:appropriate|warranted)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:federal funds rate|policy rate|rates?|target range)\s+"
        r"would\s+(?:remain|be left)\s+unchanged\b",
        re.I,
    ),
    re.compile(
        r"\b(?:a\s+)?(?:quarter|half|three-quarter)[ -]point\s+"
        r"(?:rate\s+)?(?:cut|reduction|increase|hike)\s+"
        r"(?:was|is|has been)\s+(?:announced|approved|implemented)\b",
        re.I,
    ),
    re.compile(
        r"^\s*(?:rates?|federal funds rate|policy rate|target range)\s+"
        r"(?:rose|fell|increased|decreased|declined)\s+by\s+"
        r"(?:\d+|quarter|half|three-quarter)\s*(?:bp|basis points?)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:cut|hike|hold)\s+(?:by\s+)?(?:\d+|quarter|half|three-quarter)\s*"
        r"(?:bp|basis points?)\b",
        re.I,
    ),
)

# These accepted teacher rationales violated the declared qualitative
# reasoning contract. The replacements remove explicit dates and quantities;
# prompt, decision, and all other target bytes remain unchanged.
REASONING_REPAIRS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "dec-074801857171e915b5940f10": (
        ("sits near zero", "sits at the effective lower bound"),
        ("at the zero lower bound", "at the effective lower bound"),
    ),
    "dec-182af3b697a19fb23d910281": (
        ("in the first quarter", "earlier in the period"),
        ("in May", "in the latest period"),
    ),
    "dec-26b22d74159882e6864770f4": (
        ("symmetric two percent goal", "symmetric price-stability goal"),
    ),
    "dec-3fdc181032d6858e47e138ca": (
        ("in early 2015", "early in the period"),
    ),
    "dec-4cd08026270edd2701526a33": (
        (
            "federal funds rate near zero",
            "federal funds rate at the effective lower bound",
        ),
    ),
    "dec-4dfab939b0a910c949931475": (
        ("in August", "in the latest reading"),
    ),
    "dec-5a353e765cc4c37c7d7b8582": (
        ("near-zero rates", "rates at the effective lower bound"),
    ),
    "dec-6ca7c562a9728904984426ef": (
        (
            "federal funds rate near zero",
            "federal funds rate at the effective lower bound",
        ),
    ),
    "dec-7a68fc4079fd2e1168d403e9": (
        (
            "federal funds rate already near zero",
            "federal funds rate already at the effective lower bound",
        ),
    ),
    "dec-7e6740d911f1d22db36f3213": (
        ("rates near zero", "rates at the effective lower bound"),
    ),
    "dec-82c2ba926e589181546adb53": (
        (
            "federal funds rate is already near zero",
            "federal funds rate is already at the effective lower bound",
        ),
    ),
    "dec-83e938b302112d0bbab107a3": (
        ("near-zero policy rates", "policy rates at the effective lower bound"),
    ),
    "dec-94f83b18ef25c77e86a74962": (
        (
            "an unemployment rate near 4.4 percent",
            "an unemployment rate that remained low",
        ),
        ("in the second quarter", "earlier in the period"),
    ),
    "dec-a49bfafd7a8626caa1cde624": (
        ("rates near zero", "rates at the effective lower bound"),
    ),
    "dec-bb4c79d05fb7873b2f1a880a": (
        ("in the third quarter", "in the recent period"),
    ),
    "dec-ca38445121f8ee4ffd5c6ea7": (
        (
            "federal funds rate is already near zero",
            "federal funds rate is already at the effective lower bound",
        ),
    ),
    "dec-d275636bbb2d0c3718247dda": (
        ("in the first half", "earlier in the period"),
        (
            "federal funds rate near zero",
            "federal funds rate at the effective lower bound",
        ),
    ),
    "dec-d2968fccbeedab8569d986a9": (
        ("after a strong January", "after an earlier strong reading"),
    ),
    "dec-d6168de339874bc09330c8cb": (
        ("in early 2016", "early in the period"),
    ),
    "dec-d91ee644a53020e41376ca65": (
        ("rates near zero", "rates at the effective lower bound"),
    ),
    "dec-de9d3a26759a82709f068f65": (
        (
            "federal funds rate was already near zero",
            "federal funds rate was already at the effective lower bound",
        ),
    ),
    "dec-ad16b82a66959c20aa4e5283": (
        ("in early 2022", "early in the period"),
    ),
    "dec-f66acacb4f43fea1e06bbfe8": (
        (
            "symmetric two percent inflation",
            "inflation at the symmetric price-stability objective",
        ),
    ),
}


class Chk4ReleaseError(RuntimeError):
    """The source data or immutable publication contract is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Chk4ReleaseError(message)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    _require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    _require(root.is_dir() and not root.is_symlink(), f"not a regular directory: {root}")
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), f"source tree contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Chk4ReleaseError(f"invalid {label}: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise Chk4ReleaseError(f"cannot read {label}: {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        _require(line != "", f"{label}:{line_number}: blank row")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Chk4ReleaseError(
                f"{label}:{line_number}: invalid JSON: {exc}"
            ) from exc
        _require(isinstance(value, dict), f"{label}:{line_number}: row is not an object")
        rows.append(value)
    return rows


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_text(path, "".join(_canonical_json(row) + "\n" for row in rows))


def _gold(direction: Any, magnitude: Any) -> dict[str, Any]:
    _require(direction in {"cut", "hold", "hike"}, f"invalid direction: {direction!r}")
    _require(
        isinstance(magnitude, int) and not isinstance(magnitude, bool),
        f"invalid magnitude type: {magnitude!r}",
    )
    allowed = {0} if direction == "hold" else {25, 50, 75, 100}
    _require(magnitude in allowed, f"invalid decision: {direction}/{magnitude}")
    return {"direction": direction, "magnitude_bp": magnitude}


def _gold_text(direction: Any, magnitude: Any) -> str:
    return _canonical_json(_gold(direction, magnitude))


def _calendar_context_is_proven_prior(
    clause: str,
    *,
    action_start: int,
    action_end: int,
    meeting_date: str,
    point_in_time_attested: bool,
) -> bool:
    target = datetime.strptime(meeting_date, "%Y-%m-%d")
    contexts = list(_CALENDAR_CONTEXT_RE.finditer(clause))
    if not contexts:
        return False
    nearest = min(
        contexts,
        key=lambda item: min(
            abs(item.start() - action_end), abs(item.end() - action_start)
        ),
    ).group(0)
    years = [
        int(value)
        for value in re.findall(r"\b(?:in\s+)?((?:19|20)\d{2})\b", nearest)
    ]
    if years:
        return all(year < target.year for year in years)
    month_names = (
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
    )
    matched_months = [
        index
        for index, month in enumerate(month_names, start=1)
        if re.search(rf"\b{month}\b", nearest, re.I)
    ]
    return bool(
        point_in_time_attested
        and matched_months
        and all(month != target.month for month in matched_months)
    )


def _contains_target_outcome_language(
    analysis: str, *, meeting_date: str, point_in_time_attested: bool
) -> bool:
    if any(pattern.search(analysis) for pattern in _TARGET_OUTCOME_PATTERNS):
        return True
    clauses = re.split(
        r"(?<=[.!?;])\s+|[\r\n]+|,\s+(?=(?:and|but|while|whereas)\b)|"
        r"\s+(?:and|but)\s+(?=(?:today\b|(?:the\s+)?(?:Committee|FOMC|Fed|"
        r"Federal Reserve|central bank|policymakers|officials)\b))",
        analysis,
        flags=re.I,
    )
    for clause in clauses:
        if not clause.strip():
            continue
        for pattern in _CONTEXTUAL_TARGET_ACTION_PATTERNS:
            for match in pattern.finditer(clause):
                if _PRIOR_MEETING_CONTEXT_RE.search(clause):
                    continue
                if _calendar_context_is_proven_prior(
                    clause,
                    action_start=match.start(),
                    action_end=match.end(),
                    meeting_date=meeting_date,
                    point_in_time_attested=point_in_time_attested,
                ):
                    continue
                return True
    return False


def _parse_prompt(
    prompt: Any,
    *,
    meeting_date: str,
    label: str,
    point_in_time_attested: bool = False,
) -> str:
    _require(isinstance(prompt, str) and prompt.strip(), f"{label}: empty prompt")
    lowered = prompt.casefold()
    for fragment in _PROMPT_FORBIDDEN:
        _require(fragment not in lowered, f"{label}: prompt leaks {fragment!r}")
    target = datetime.strptime(meeting_date, "%Y-%m-%d")
    month_name = target.strftime("%B")
    month_abbr = target.strftime("%b")
    date_variants = {
        meeting_date,
        f"{month_name} {target.day}, {target.year}",
        f"{month_name} {target.day} {target.year}",
        f"{month_abbr} {target.day}, {target.year}",
        f"{target.month}/{target.day}/{target.year}",
        f"{target.month:02d}/{target.day:02d}/{target.year}",
        f"{target.month}-{target.day}-{target.year}",
        f"{target.month:02d}-{target.day:02d}-{target.year}",
    }
    _require(
        not any(value.casefold() in lowered for value in date_variants),
        f"{label}: prompt leaks meeting_date",
    )
    _require(_CONTROL_RE.search(prompt) is None, f"{label}: prompt contains control tags")
    prefix = USER_PROMPT_PREFIX
    _require(prompt.startswith(prefix), f"{label}: unexpected prompt prefix")
    try:
        payload = json.loads(prompt[len(prefix) :])
    except json.JSONDecodeError as exc:
        raise Chk4ReleaseError(f"{label}: prompt payload is invalid JSON") from exc
    _require(
        isinstance(payload, dict) and set(payload) == {"analysis"},
        f"{label}: prompt payload must contain only analysis",
    )
    analysis = payload["analysis"]
    _require(isinstance(analysis, str) and analysis.strip(), f"{label}: empty analysis")
    _require(
        not _contains_target_outcome_language(
            analysis,
            meeting_date=meeting_date,
            point_in_time_attested=point_in_time_attested,
        ),
        f"{label}: analysis contains target-meeting outcome language",
    )
    return analysis


def _parse_response(response: Any, *, label: str) -> tuple[str, dict[str, Any]]:
    _require(isinstance(response, str), f"{label}: response must be text")
    _require(response.count(BOUNDARY) == 1, f"{label}: invalid reasoning boundary")
    reasoning, answer = response.split(BOUNDARY)
    reasoning = reasoning.strip()
    _require(reasoning, f"{label}: empty reasoning")
    _require(_CONTROL_RE.search(reasoning) is None, f"{label}: reasoning has control tags")
    _require(_DIGIT_RE.search(reasoning) is None, f"{label}: reasoning contains digits")
    _require(_DATE_RE.search(reasoning) is None, f"{label}: reasoning contains a date")
    _require(
        _EXPLICIT_REASONING_DATE_RE.search(reasoning) is None,
        f"{label}: reasoning contains an explicit month or quarter",
    )
    _require(
        _SPELLED_NUMBER_RE.search(reasoning) is None,
        f"{label}: reasoning contains a spelled-out number",
    )
    try:
        decision = json.loads(answer)
    except json.JSONDecodeError as exc:
        raise Chk4ReleaseError(f"{label}: final decision is invalid JSON") from exc
    _require(isinstance(decision, dict), f"{label}: final decision must be an object")
    canonical = _gold(decision.get("direction"), decision.get("magnitude_bp"))
    _require(set(decision) == set(canonical), f"{label}: final decision has extra keys")
    _require(answer == _canonical_json(canonical), f"{label}: final JSON is not canonical")
    return reasoning, canonical


def _expected_row_id(stage: str, sample_id: str, repeat_index: int) -> str:
    digest = _sha256_text(
        f"{SOURCE_SCHEMA}\0{stage}\0{sample_id}\0{repeat_index}"
    )[:24]
    return f"row-{digest}"


def _apply_repairs(
    *, source_id: str, response: str, split: str
) -> tuple[str, dict[str, Any] | None]:
    replacements = REASONING_REPAIRS.get(source_id)
    if replacements is None:
        return response, None
    updated = response
    changes: list[dict[str, str]] = []
    for old_phrase, new_phrase in replacements:
        _require(
            updated.count(old_phrase) == 1,
            f"{split}/{source_id}: expected repair phrase is missing or ambiguous",
        )
        updated = updated.replace(old_phrase, new_phrase, 1)
        changes.append(
            {
                "old_phrase_sha256": _sha256_text(old_phrase),
                "new_phrase_sha256": _sha256_text(new_phrase),
            }
        )
    record = {
        "schema_version": REPAIR_SCHEMA,
        "sample_id": source_id,
        "split": split,
        "section": "reasoning",
        "method": "replace_explicit_number_or_date_with_qualitative_source_preserving_phrase",
        "changes": changes,
        "old_response_sha256": _sha256_text(response),
        "new_response_sha256": _sha256_text(updated),
        "prompt_and_final_decision_unchanged": True,
    }
    return updated, record


def _validate_source_summaries(
    *, source: Path, teacher: Path, briefs: Path
) -> dict[str, Any]:
    expected_source_files = {"summary.json"}
    expected_source_files.update(
        f"{stage}/{split}.jsonl"
        for stage in ("decision_sft", "decision_grpo")
        for split in SPLITS
    )
    expected_source_files.update(
        f"manifests/{kind}/{split}.jsonl"
        for kind in ("unique", "repeats")
        for split in SPLITS
    )
    actual_source_files: set[str] = set()
    for path in source.rglob("*"):
        _require(not path.is_symlink(), f"source contains symlink: {path}")
        if path.is_file():
            actual_source_files.add(path.relative_to(source).as_posix())
    _require(
        actual_source_files == expected_source_files,
        "source candidate contains missing or unexpected files",
    )
    source_summary = _read_json(source / "summary.json", label="source summary")
    _require(source_summary.get("schema_version") == SOURCE_SCHEMA, "source schema drift")
    _require(source_summary.get("status") == "complete", "source materialization incomplete")
    _require(source_summary.get("unique_counts") == EXPECTED_UNIQUE, "unique counts drift")
    expected_physical = {
        "decision_sft": EXPECTED_PHYSICAL,
        "decision_grpo": EXPECTED_PHYSICAL,
    }
    _require(
        source_summary.get("physical_counts") == expected_physical,
        "physical counts drift",
    )
    _require(
        source_summary.get("repeat_factors") == EXPECTED_REPEAT_FACTORS,
        "repeat factors drift",
    )
    _require(
        source_summary.get("system_prompt_sha256") == _sha256_text(STUDENT_SYSTEM_PROMPT),
        "student system prompt drift",
    )
    _require(
        source_summary.get("train_unique_direction_counts")
        == EXPECTED_UNIQUE_TRAIN_DIRECTIONS,
        "unique direction counts drift",
    )
    _require(
        source_summary.get("train_physical_direction_counts")
        == EXPECTED_PHYSICAL_TRAIN_DIRECTIONS,
        "physical direction counts drift",
    )
    _require(source_summary.get("test_is_sealed_evaluation_only") is True, "test is not sealed")

    teacher_summary = _read_json(teacher / "summary.json", label="teacher summary")
    _require(
        teacher_summary.get("schema_version") == "chk4-decision-generation-summary-v2",
        "teacher schema drift",
    )
    _require(teacher_summary.get("status") == "complete", "teacher acquisition incomplete")
    _require(teacher_summary.get("accepted_count") == 128, "teacher accepted count drift")
    _require(teacher_summary.get("failure_count") == 0, "teacher acquisition has failures")
    _require(
        teacher_summary.get("preparation_audit", {}).get("split_counts") == EXPECTED_UNIQUE,
        "teacher split counts drift",
    )
    _require(
        teacher_summary.get("preparation_audit", {}).get("supplement_admitted") == 0,
        "unexpected supplement population",
    )
    teacher_tree = _tree_sha256(teacher)
    _require(
        source_summary.get("teacher_root_sha256") == teacher_tree,
        "source/teacher tree binding drift",
    )

    brief_summary = _read_json(briefs / "summary.json", label="brief summary")
    _require(
        brief_summary.get("schema_version") == "chk4-deepseek-meeting-brief-v1",
        "brief schema drift",
    )
    _require(brief_summary.get("status") == "complete", "meeting briefs incomplete")
    _require(brief_summary.get("gold_blind") is True, "meeting briefs are not gold blind")
    _require(brief_summary.get("meeting_counts") == EXPECTED_UNIQUE, "brief counts drift")
    _require(brief_summary.get("accepted_count") == 128, "brief accepted count drift")
    _require(brief_summary.get("failure_count") == 0, "meeting briefs have failures")
    _require(brief_summary.get("api_requests") == 0, "brief materialization used API requests")

    return {
        "source_summary": source_summary,
        "teacher_summary": teacher_summary,
        "brief_summary": brief_summary,
        "source_tree_sha256": _tree_sha256(source),
        "teacher_tree_sha256": teacher_tree,
        "brief_tree_sha256": _tree_sha256(briefs),
    }


def _build_input_contract(source_info: Mapping[str, Any]) -> dict[str, Any]:
    """Define target blindness without discarding legitimate pre-meeting state."""

    source_summary = source_info["source_summary"]
    brief_summary = source_info["brief_summary"]
    payload: dict[str, Any] = {
        "schema_version": INPUT_CONTRACT_SCHEMA,
        "status": "active",
        "population_scope": "core-only; supplement_admitted=0",
        "definition": (
            "target-decision-blind is established structurally by excluding target fields and "
            "binding inputs to cutoff-safe chk1 evidence. A deterministic known-pattern scan "
            "adds defense in depth but is not an exhaustive semantic proof; "
            "direct sample identifiers and the exact target-meeting date field are absent; "
            "point-in-time policy state known before the meeting remains legitimate evidence. "
            "This is a direct-field contract, not a claim that historical clues make the "
            "meeting impossible to infer"
        ),
        "allowed_point_in_time_fields": [
            "effective_federal_funds_rate",
            "pre_meeting_target_range",
            "historical_policy_actions_before_the_target_meeting",
            "federal_reserve_balance_sheet_state",
            "other_pre_meeting_macro_and_financial_evidence",
        ],
        "forbidden_target_fields": [
            "target_meeting_direction",
            "target_meeting_magnitude_bp",
            "target_meeting_vote",
            "target_meeting_minutes",
            "post_meeting_information",
            "gold_or_teacher_fields",
            "direct_sample_id_or_exact_target_meeting_date_field",
        ],
        "identity_inference_risk": (
            "historical dates, policy state, and macroeconomic context can indirectly identify "
            "a meeting; downstream evaluation must not describe this release as inference-proof"
        ),
        "semantic_assurance": {
            "structural_lineage": "full replay required",
            "known_pattern_scan": "required with zero hits",
            "independent_source_only_semantic_judge": "not_run",
            "claim_limit": (
                "do not describe the deterministic language scanner as an exhaustive proof "
                "that every possible paraphrase of an outcome is absent"
            ),
        },
        "required_checks": [
            "prompt_meeting_date_absent",
            "prompt_sample_and_source_ids_absent",
            "known_pattern_target_outcome_hits_equal_zero",
            "brief_numbers_are_subset_of_point_in_time_atomic_source_numbers",
            "brief_dates_are_subset_of_point_in_time_atomic_source_dates",
            "sft_and_grpo_prompts_are_byte_identical",
        ],
        "superseded_brief_contract": {
            "schema_version": "chk4-deepseek-brief-contract-v1",
            "sha256": brief_summary.get("contract_sha256"),
            "reason": (
                "the old wording prohibited all target-range references even when they were "
                "pre-meeting state; the original contract is preserved but is not claimed passed"
            ),
        },
        "teacher_contract_sha256": source_summary.get("teacher_contract_sha256"),
        "student_system_prompt": STUDENT_SYSTEM_PROMPT,
        "student_system_prompt_sha256": _sha256_text(STUDENT_SYSTEM_PROMPT),
    }
    payload["contract_sha256"] = _sha256_text(_canonical_json(payload))
    return payload


def _atoms(pattern: re.Pattern[str], text: str) -> set[str]:
    return {match.group(0).replace(",", "").casefold() for match in pattern.finditer(text)}


def _brief_atomic_source(prepared_prompt: Any, *, label: str) -> list[dict[str, str]]:
    _require(
        isinstance(prepared_prompt, str) and prepared_prompt.startswith(BRIEF_USER_PREFIX),
        f"{label}: unexpected brief-preparation prompt",
    )
    try:
        payload = json.loads(prepared_prompt[len(BRIEF_USER_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise Chk4ReleaseError(f"{label}: invalid brief-preparation payload") from exc
    _require(
        isinstance(payload, dict) and set(payload) == {"atomic_analyses"},
        f"{label}: invalid brief-preparation payload schema",
    )
    atomic = payload["atomic_analyses"]
    _require(isinstance(atomic, list) and atomic, f"{label}: atomic analyses missing")
    output: list[dict[str, str]] = []
    for index, item in enumerate(atomic):
        _require(
            isinstance(item, dict) and set(item) == {"atomic_topic", "analysis"},
            f"{label}: invalid atomic analysis {index}",
        )
        topic = item["atomic_topic"]
        analysis = item["analysis"]
        _require(
            isinstance(topic, str) and topic and isinstance(analysis, str) and analysis,
            f"{label}: empty atomic analysis {index}",
        )
        output.append({"atomic_topic": topic, "analysis": analysis})
    return output


def _parse_utc_timestamp(value: Any, *, label: str) -> datetime:
    _require(isinstance(value, str) and value.endswith("Z"), f"{label}: invalid UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise Chk4ReleaseError(f"{label}: invalid UTC timestamp") from exc
    return parsed


def _selected_evidence_is_uniquely_bound(
    selected_ids: Sequence[Any], evidence_ids: Sequence[str]
) -> bool:
    """Accept canonical short IDs only when every prefix resolves uniquely."""

    if not selected_ids:
        return True
    if (
        any(not isinstance(item, str) or not item for item in selected_ids)
        or len(selected_ids) != len(set(selected_ids))
    ):
        return False
    resolved: list[str] = []
    for selected in selected_ids:
        matches = [
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id == selected or evidence_id.startswith(selected)
        ]
        if len(matches) != 1:
            return False
        resolved.append(matches[0])
    return len(resolved) == len(set(resolved))


def _load_canonical_chk1_sources(
    briefs: Path,
) -> tuple[
    Path,
    str,
    dict[str, tuple[str, dict[str, Any], dict[str, Any]]],
    dict[str, int],
]:
    brief_summary = _read_json(briefs / "summary.json", label="brief summary lineage")
    source_root_value = brief_summary.get("source_chk1_root")
    expected_tree_sha = brief_summary.get("source_chk1_sha256")
    _require(isinstance(source_root_value, str) and source_root_value, "canonical chk1 path missing")
    _require(
        isinstance(expected_tree_sha, str) and _SHA_RE.fullmatch(expected_tree_sha),
        "canonical chk1 tree SHA missing",
    )
    source_root = Path(source_root_value).expanduser().resolve()
    _require(source_root.is_dir() and not source_root.is_symlink(), "canonical chk1 root missing")
    _require(
        _tree_sha256(source_root) == expected_tree_sha,
        "canonical chk1 tree hash drift",
    )
    handoff = _read_json(source_root / "handoff.json", label="canonical chk1 handoff")
    _require(handoff.get("quality_status") == "passed", "canonical chk1 quality did not pass")
    _require(handoff.get("immutable") is True, "canonical chk1 is not immutable")
    split_names = {"train": "train", "validation": "eval", "test": "test"}
    source_rows: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {}
    citation_binding: Counter[str] = Counter()
    for decision_split, canonical_split in split_names.items():
        manifest_record = handoff.get("manifest_files", {}).get(canonical_split)
        sft_record = handoff.get("split_files", {}).get(canonical_split)
        _require(
            isinstance(manifest_record, dict) and isinstance(sft_record, dict),
            f"canonical chk1 {canonical_split}: split records missing",
        )
        manifest_path = source_root / str(manifest_record.get("path"))
        sft_path = source_root / str(sft_record.get("path"))
        _require(
            _sha256_file(manifest_path) == manifest_record.get("sha256"),
            f"canonical chk1 {canonical_split}: manifest hash drift",
        )
        _require(
            _sha256_file(sft_path) == sft_record.get("sha256"),
            f"canonical chk1 {canonical_split}: SFT hash drift",
        )
        manifests = _read_jsonl(manifest_path, label=f"canonical chk1 {canonical_split} manifest")
        sft_rows = _read_jsonl(sft_path, label=f"canonical chk1 {canonical_split} SFT")
        _require(
            len(manifests)
            == len(sft_rows)
            == manifest_record.get("rows")
            == sft_record.get("rows"),
            f"canonical chk1 {canonical_split}: row count drift",
        )
        for manifest, sft in zip(manifests, sft_rows, strict=True):
            source_id = manifest.get("sample_id")
            _require(isinstance(source_id, str) and source_id, "canonical chk1 sample ID missing")
            _require(source_id not in source_rows, "canonical chk1 duplicate sample ID")
            _require(manifest.get("split") == canonical_split, "canonical chk1 split drift")
            _require(manifest.get("input_truncated") is False, "canonical chk1 input truncated")
            _require(set(sft) == {"prompt", "provided_data", "response"}, "canonical chk1 SFT schema drift")
            _require(
                _sha256_text(str(sft["provided_data"])) == manifest.get("provided_data_sha256"),
                "canonical chk1 provided-data hash drift",
            )
            try:
                provided_data = json.loads(str(sft["provided_data"]))
            except json.JSONDecodeError as exc:
                raise Chk4ReleaseError("canonical chk1 provided-data JSON drift") from exc
            evidence_lineage = manifest.get("evidence_lineage")
            selected_ids = manifest.get("selected_evidence_ids")
            _require(
                isinstance(provided_data, dict)
                and provided_data.get("atomic_topic") == manifest.get("atomic_topic")
                and isinstance(provided_data.get("evidence"), list)
                and isinstance(evidence_lineage, list)
                and isinstance(selected_ids, list),
                "canonical chk1 provided-data/evidence schema drift",
            )
            provided_evidence_ids = [
                item.get("evidence_id") if isinstance(item, dict) else None
                for item in provided_data["evidence"]
            ]
            lineage_evidence_ids = [
                item.get("evidence_id") if isinstance(item, dict) else None
                for item in evidence_lineage
            ]
            _require(
                provided_evidence_ids == lineage_evidence_ids
                and all(isinstance(item, str) and item for item in lineage_evidence_ids)
                and len(lineage_evidence_ids) == len(set(lineage_evidence_ids))
                and all(isinstance(item, str) and item for item in selected_ids)
                and len(selected_ids) == len(set(selected_ids)),
                "canonical chk1 selected/provided evidence binding drift",
            )
            if not selected_ids:
                citation_binding["empty"] += 1
            elif all(item in lineage_evidence_ids for item in selected_ids):
                citation_binding["exact"] += 1
            elif _selected_evidence_is_uniquely_bound(
                selected_ids, lineage_evidence_ids
            ):
                citation_binding["unique_prefix"] += 1
            else:
                citation_binding["unresolved"] += 1
            response = sft["response"]
            _require(
                isinstance(response, str) and response.count(BOUNDARY) == 1,
                "canonical chk1 response boundary drift",
            )
            final_analysis = response.split(BOUNDARY, 1)[1].strip()
            _require(
                _sha256_text(final_analysis) == manifest.get("final_analysis_sha256"),
                "canonical chk1 final-analysis hash drift",
            )
            source_rows[source_id] = (decision_split, manifest, sft)
    _require(len(source_rows) == 2072, "canonical chk1 source population drift")
    _require(sum(citation_binding.values()) == len(source_rows), "citation audit drift")
    return source_root, expected_tree_sha, source_rows, dict(citation_binding)


def _validate_external_lineage(
    *, clean_root: Path, teacher: Path, briefs: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Bind every clean prompt to its teacher and point-in-time brief sources."""

    lineage_rows: list[dict[str, Any]] = []
    policy_state_counts: Counter[str] = Counter()
    (
        canonical_root,
        canonical_tree_sha,
        canonical_sources,
        citation_binding,
    ) = _load_canonical_chk1_sources(briefs)
    canonical_source_ids_used: set[str] = set()
    evidence_rows_checked = 0
    target_teacher_rows_bound = 0
    brief_teacher_rows_bound = 0
    for split in SPLITS:
        unique_rows = _read_jsonl(
            clean_root / "manifests/unique" / f"{split}.jsonl",
            label=f"{split} clean unique lineage",
        )
        sft_rows = _read_jsonl(
            clean_root / "decision_sft" / f"{split}.jsonl",
            label=f"{split} clean SFT lineage",
        )
        target_prepared_rows = _read_jsonl(
            teacher / "prepared" / f"{split}.jsonl",
            label=f"{split} target prepared",
        )
        target_manifest_rows = _read_jsonl(
            teacher / "manifests" / f"{split}.jsonl",
            label=f"{split} target manifest",
        )
        target_sft_rows = _read_jsonl(
            teacher / "sft" / f"{split}.jsonl",
            label=f"{split} teacher SFT",
        )
        target_response_rows = _read_jsonl(
            teacher / "teacher_responses" / f"{split}.jsonl",
            label=f"{split} teacher raw responses",
        )
        _require(
            len(target_prepared_rows)
            == len(target_manifest_rows)
            == len(target_sft_rows)
            == len(target_response_rows)
            == EXPECTED_UNIQUE[split],
            f"{split}: teacher target population drift",
        )
        target_prepared: dict[str, dict[str, Any]] = {}
        target_manifest: dict[str, dict[str, Any]] = {}
        target_sft: dict[str, dict[str, Any]] = {}
        for prepared_row, manifest_row, teacher_sft, raw_response in zip(
            target_prepared_rows,
            target_manifest_rows,
            target_sft_rows,
            target_response_rows,
            strict=True,
        ):
            source_id = prepared_row.get("sample_id")
            _require(
                isinstance(source_id, str)
                and source_id
                and source_id == manifest_row.get("sample_id")
                == raw_response.get("sample_id"),
                f"{split}: teacher target sample binding drift",
            )
            _require(source_id not in target_prepared, f"{split}: duplicate teacher target")
            _require(
                set(teacher_sft) == {"prompt", "response"}
                and teacher_sft.get("prompt") == prepared_row.get("prompt"),
                f"{split}/{source_id}: teacher SFT prompt drift",
            )
            provider = raw_response.get("provider")
            _require(
                isinstance(provider, dict) and provider.get("finish_reason") == "stop",
                f"{split}/{source_id}: teacher response did not stop",
            )
            try:
                content = json.loads(str(raw_response.get("content")))
            except json.JSONDecodeError as exc:
                raise Chk4ReleaseError(
                    f"{split}/{source_id}: malformed teacher response"
                ) from exc
            _require(
                isinstance(content, dict)
                and set(content) == {"reasoning", "direction", "magnitude_bp"}
                and isinstance(content.get("reasoning"), str)
                and content["reasoning"].strip(),
                f"{split}/{source_id}: teacher response schema drift",
            )
            expected_response = (
                content["reasoning"].strip()
                + BOUNDARY
                + _gold_text(content["direction"], content["magnitude_bp"])
            )
            _require(
                teacher_sft.get("response") == expected_response,
                f"{split}/{source_id}: teacher raw/SFT response drift",
            )
            target_prepared[source_id] = prepared_row
            target_manifest[source_id] = manifest_row
            target_sft[source_id] = teacher_sft
            target_teacher_rows_bound += 1
        brief_outputs = {
            row.get("meeting_date"): row
            for row in _read_jsonl(briefs / f"{split}.jsonl", label=f"{split} briefs")
        }
        brief_prepared_rows = _read_jsonl(
            briefs / "prepared" / f"{split}.jsonl",
            label=f"{split} brief prepared",
        )
        brief_prepared = {row.get("input_sha256"): row for row in brief_prepared_rows}
        brief_manifest_rows = _read_jsonl(
            briefs / "manifests" / f"{split}.jsonl",
            label=f"{split} brief manifest",
        )
        brief_response_rows = _read_jsonl(
            briefs / "teacher_responses" / f"{split}.jsonl",
            label=f"{split} brief raw responses",
        )
        brief_manifest = {row.get("meeting_date"): row for row in brief_manifest_rows}
        brief_responses = {row.get("sample_id"): row for row in brief_response_rows}
        _require(
            len(target_prepared)
            == len(target_manifest)
            == len(brief_outputs)
            == len(brief_prepared)
            == len(brief_manifest)
            == len(brief_responses)
            == EXPECTED_UNIQUE[split],
            f"{split}: external lineage population drift",
        )
        sft_index = 0
        for unique in unique_rows:
            sample_id = str(unique["sample_id"])
            meeting_date = str(unique["meeting_date"])
            label = f"{split}/{sample_id}"
            sft = sft_rows[sft_index]
            sft_index += int(unique["repeat_factor"])
            prepared = target_prepared.get(sample_id)
            manifest = target_manifest.get(sample_id)
            teacher_sft = target_sft.get(sample_id)
            brief = brief_outputs.get(meeting_date)
            brief_manifest_row = brief_manifest.get(meeting_date)
            _require(
                all(
                    isinstance(value, dict)
                    for value in (
                        prepared,
                        manifest,
                        teacher_sft,
                        brief,
                        brief_manifest_row,
                    )
                ),
                f"{label}: missing external lineage row",
            )
            assert isinstance(prepared, dict)
            assert isinstance(manifest, dict)
            assert isinstance(teacher_sft, dict)
            assert isinstance(brief, dict)
            assert isinstance(brief_manifest_row, dict)
            _require(prepared.get("prompt") == sft["prompt"], f"{label}: target prompt drift")
            expected_clean_response, _repair = _apply_repairs(
                source_id=sample_id,
                response=str(teacher_sft.get("response")),
                split=split,
            )
            _require(
                sft.get("response") == expected_clean_response,
                f"{label}: clean SFT/teacher target response drift",
            )
            _require(
                prepared.get("prompt_sha256") == unique["prompt_sha256"],
                f"{label}: target prepared prompt SHA drift",
            )
            _require(manifest.get("prompt_sha256") == unique["prompt_sha256"], f"{label}: target manifest prompt SHA drift")
            _require(manifest.get("meeting_date") == meeting_date, f"{label}: target meeting drift")
            _require(manifest.get("source_ids") == unique["source_ids"], f"{label}: target source IDs drift")
            _require(manifest.get("gold_sha256") == unique["gold_sha256"], f"{label}: target gold drift")
            analysis = _parse_prompt(
                sft["prompt"],
                meeting_date=meeting_date,
                label=label,
                point_in_time_attested=True,
            )
            _require(
                brief.get("meeting_decision_brief") == analysis,
                f"{label}: brief/training prompt drift",
            )
            _require(brief.get("brief_sha256") == _sha256_text(analysis), f"{label}: brief SHA drift")
            _require(brief.get("source_ids") == unique["source_ids"], f"{label}: brief source IDs drift")
            _require(
                brief_manifest_row.get("source_ids") == unique["source_ids"],
                f"{label}: brief manifest source IDs drift",
            )
            brief_sample_id = brief_manifest_row.get("sample_id")
            brief_response = brief_responses.get(brief_sample_id)
            _require(
                isinstance(brief_sample_id, str)
                and isinstance(brief_response, dict)
                and brief_response.get("sample_id") == brief_sample_id,
                f"{label}: brief teacher response binding drift",
            )
            brief_provider = brief_response.get("provider")
            _require(
                isinstance(brief_provider, dict)
                and brief_provider.get("finish_reason") == "stop",
                f"{label}: brief teacher response did not stop",
            )
            try:
                brief_content = json.loads(str(brief_response.get("content")))
            except json.JSONDecodeError as exc:
                raise Chk4ReleaseError(
                    f"{label}: malformed brief teacher response"
                ) from exc
            _require(
                isinstance(brief_content, dict)
                and set(brief_content) == {"meeting_decision_brief"}
                and brief_content.get("meeting_decision_brief") == analysis,
                f"{label}: brief raw/output response drift",
            )
            brief_teacher_rows_bound += 1
            input_sha = str(brief.get("source_input_sha256") or "")
            _require(
                brief_manifest_row.get("input_sha256") == input_sha,
                f"{label}: brief input SHA drift",
            )
            brief_source = brief_prepared.get(input_sha)
            _require(isinstance(brief_source, dict), f"{label}: brief source payload missing")
            atomic = _brief_atomic_source(brief_source.get("prompt"), label=label)
            _require(
                len(atomic) == len(unique["source_ids"]),
                f"{label}: atomic/source ID count drift",
            )
            remaining_source_ids = set(unique["source_ids"])
            for atomic_row in atomic:
                matching_source_ids: list[str] = []
                for candidate_id in remaining_source_ids:
                    candidate = canonical_sources.get(candidate_id)
                    if candidate is None:
                        continue
                    candidate_split, candidate_manifest, candidate_sft = candidate
                    candidate_analysis = str(candidate_sft["response"]).split(
                        BOUNDARY, 1
                    )[1].strip()
                    if (
                        candidate_split == split
                        and candidate_manifest.get("meeting_date") == meeting_date
                        and candidate_manifest.get("atomic_topic")
                        == atomic_row["atomic_topic"]
                        and candidate_analysis == atomic_row["analysis"]
                    ):
                        matching_source_ids.append(candidate_id)
                _require(
                    len(matching_source_ids) == 1,
                    f"{label}: canonical atomic analysis mapping is not unique",
                )
                source_id = matching_source_ids[0]
                remaining_source_ids.remove(source_id)
                canonical = canonical_sources[source_id]
                canonical_split, canonical_manifest, canonical_sft = canonical
                _require(canonical_split == split, f"{label}: canonical source split drift")
                _require(
                    canonical_manifest.get("meeting_date") == meeting_date,
                    f"{label}: canonical source meeting drift",
                )
                _require(
                    canonical_manifest.get("atomic_topic") == atomic_row["atomic_topic"],
                    f"{label}: canonical atomic topic drift",
                )
                canonical_analysis = str(canonical_sft["response"]).split(
                    BOUNDARY, 1
                )[1].strip()
                _require(
                    canonical_analysis == atomic_row["analysis"],
                    f"{label}: canonical final analysis text drift",
                )
                cutoff = _parse_utc_timestamp(
                    canonical_manifest.get("cutoff_ts"),
                    label=f"{label}/{source_id}/cutoff",
                )
                _require(
                    cutoff.date().isoformat() == meeting_date,
                    f"{label}: canonical cutoff/meeting drift",
                )
                evidence_lineage = canonical_manifest.get("evidence_lineage")
                selected_ids = canonical_manifest.get("selected_evidence_ids")
                _require(
                    isinstance(evidence_lineage, list)
                    and evidence_lineage
                    and isinstance(selected_ids, list),
                    f"{label}: canonical evidence lineage missing",
                )
                evidence_ids: list[str] = []
                for evidence_index, evidence in enumerate(evidence_lineage):
                    _require(
                        isinstance(evidence, dict),
                        f"{label}: canonical evidence row invalid",
                    )
                    evidence_id = evidence.get("evidence_id")
                    _require(
                        isinstance(evidence_id, str) and evidence_id,
                        f"{label}: canonical evidence ID missing",
                    )
                    available = _parse_utc_timestamp(
                        evidence.get("availability_upper_bound_ts"),
                        label=f"{label}/{source_id}/evidence-{evidence_index}",
                    )
                    _require(
                        available <= cutoff,
                        f"{label}: future evidence crosses cutoff",
                    )
                    _require(
                        evidence.get("cutoff_ts") == canonical_manifest.get("cutoff_ts"),
                        f"{label}: evidence cutoff binding drift",
                    )
                    evidence_ids.append(evidence_id)
                    evidence_rows_checked += 1
                _require(
                    len(evidence_ids) == len(set(evidence_ids))
                    and all(isinstance(item, str) and item for item in selected_ids),
                    f"{label}: evidence lineage membership drift",
                )
                canonical_source_ids_used.add(source_id)
            _require(
                not remaining_source_ids,
                f"{label}: canonical source IDs were not consumed",
            )
            source_text = "\n".join(row["analysis"] for row in atomic)
            unsupported_numbers = _atoms(BRIEF_NUMBER_RE, analysis) - _atoms(
                BRIEF_NUMBER_RE, source_text
            )
            unsupported_dates = _atoms(BRIEF_DATE_RE, analysis) - _atoms(
                BRIEF_DATE_RE, source_text
            )
            _require(not unsupported_numbers, f"{label}: unsupported brief numbers")
            _require(not unsupported_dates, f"{label}: unsupported brief dates")
            state_flags = {
                "contains_pre_meeting_target_range": bool(
                    re.search(r"\btarget\s+range\b", analysis, re.I)
                ),
                "contains_pre_meeting_federal_funds_rate": bool(
                    re.search(r"\bfederal funds(?: effective)? rate\b", analysis, re.I)
                    or re.search(r"\beffective federal funds rate\b", analysis, re.I)
                ),
                "contains_pre_meeting_fed_balance_sheet": bool(
                    re.search(r"\bFederal Reserve(?:'s|’s)? balance sheet\b", analysis, re.I)
                ),
            }
            for name, present in state_flags.items():
                policy_state_counts[name] += int(present)
            lineage_rows.append(
                {
                    "schema_version": "chk4-target-decision-lineage-row-v1",
                    "sample_id": sample_id,
                    "split": split,
                    "meeting_date": meeting_date,
                    "prompt_sha256": unique["prompt_sha256"],
                    "brief_sha256": brief["brief_sha256"],
                    "brief_input_sha256": input_sha,
                    "source_ids_sha256": _sha256_text(_canonical_json(unique["source_ids"])),
                    "source_atomic_count": len(atomic),
                    "unsupported_number_count": 0,
                    "unsupported_date_count": 0,
                    "known_pattern_target_outcome_hit": False,
                    **state_flags,
                }
            )
        _require(sft_index == len(sft_rows), f"{split}: SFT lineage row count drift")
    _require(len(lineage_rows) == 128, "external lineage count drift")
    _require(
        canonical_source_ids_used == set(canonical_sources),
        "canonical chk1 source population is not used exactly once across meetings",
    )
    _require(
        target_teacher_rows_bound == brief_teacher_rows_bound == 128,
        "teacher response lineage population drift",
    )
    return (
        {
            "schema_version": "chk4-target-decision-lineage-audit-v1",
            "status": "passed",
            "rows": len(lineage_rows),
            "split_counts": EXPECTED_UNIQUE,
            "known_pattern_target_outcome_hits": 0,
            "unsupported_number_violations": 0,
            "unsupported_date_violations": 0,
            "teacher_target_rows_bound": target_teacher_rows_bound,
            "teacher_brief_rows_bound": brief_teacher_rows_bound,
            "canonical_chk1": {
                "path": str(canonical_root),
                "tree_sha256": canonical_tree_sha,
                "source_rows": len(canonical_source_ids_used),
                "evidence_rows_checked": evidence_rows_checked,
                "future_evidence_violations": 0,
                "meeting_or_cutoff_mismatches": 0,
                "atomic_analysis_text_mismatches": 0,
                "selected_evidence_id_binding": citation_binding,
            },
            "point_in_time_policy_state_rows": dict(policy_state_counts),
            "policy_state_interpretation": (
                "allowed pre-meeting evidence under the versioned target-decision-blind contract; "
                "not the target meeting outcome"
            ),
        },
        lineage_rows,
    )


def _materialize_clean_copy(source: Path, staging: Path) -> list[dict[str, Any]]:
    repairs_by_id: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        unique_rows = _read_jsonl(
            source / "manifests/unique" / f"{split}.jsonl",
            label=f"{split} unique manifest",
        )
        sft_rows = _read_jsonl(
            source / "decision_sft" / f"{split}.jsonl",
            label=f"{split} SFT",
        )
        grpo_rows = _read_jsonl(
            source / "decision_grpo" / f"{split}.jsonl",
            label=f"{split} GRPO",
        )
        repeat_rows = _read_jsonl(
            source / "manifests/repeats" / f"{split}.jsonl",
            label=f"{split} repeat manifest",
        )
        expected: list[tuple[dict[str, Any], int]] = []
        output_unique: list[dict[str, Any]] = []
        for unique in unique_rows:
            _require(unique.get("split") == split, f"{split}: unique split drift")
            factor = unique.get("repeat_factor")
            _require(
                isinstance(factor, int) and not isinstance(factor, bool) and factor > 0,
                f"{split}: invalid repeat factor",
            )
            if split != "train":
                _require(factor == 1, f"{split}: validation/test must not repeat")
            for repeat_index in range(factor):
                expected.append((unique, repeat_index))
            output_unique.append(dict(unique))
        _require(len(expected) == len(sft_rows) == len(grpo_rows), f"{split}: row count drift")
        _require(len(repeat_rows) == 2 * len(expected), f"{split}: repeat manifest drift")

        output_sft: list[dict[str, Any]] = []
        for index, ((unique, repeat_index), sft, grpo) in enumerate(
            zip(expected, sft_rows, grpo_rows, strict=True)
        ):
            sample_id = str(unique.get("sample_id") or "")
            _require(sample_id != "", f"{split}:{index}: empty sample_id")
            _require(set(sft) == {"prompt", "response"}, f"{split}:{index}: SFT schema")
            _require(
                set(grpo) == {"sample_id", "prompt", "direction", "magnitude_bp"},
                f"{split}:{index}: GRPO schema",
            )
            _require(grpo.get("sample_id") == sample_id, f"{split}:{index}: sample drift")
            _require(sft.get("prompt") == grpo.get("prompt"), f"{split}:{index}: prompt drift")
            _require(
                _sha256_text(str(sft.get("prompt"))) == unique.get("prompt_sha256"),
                f"{split}:{index}: prompt hash drift",
            )
            _require(
                _sha256_text(str(sft.get("response"))) == unique.get("response_sha256"),
                f"{split}:{index}: response hash drift",
            )
            _require(
                _sha256_text(_gold_text(grpo.get("direction"), grpo.get("magnitude_bp")))
                == unique.get("gold_sha256"),
                f"{split}:{index}: gold hash drift",
            )
            updated, repair = _apply_repairs(
                source_id=sample_id,
                response=str(sft["response"]),
                split=split,
            )
            if repair is not None:
                previous = repairs_by_id.setdefault(sample_id, repair)
                _require(previous == repair, f"{split}:{sample_id}: inconsistent repeated repair")
            output_sft.append({"prompt": sft["prompt"], "response": updated})

            for offset, stage in enumerate(("decision_sft", "decision_grpo")):
                repeat = repeat_rows[2 * index + offset]
                _require(repeat.get("schema_version") == REPEAT_SCHEMA, "repeat schema drift")
                _require(repeat.get("stage") == stage, f"{split}:{index}: repeat stage drift")
                _require(repeat.get("split") == split, f"{split}:{index}: repeat split drift")
                _require(
                    repeat.get("source_sample_id") == sample_id,
                    f"{split}:{index}: repeat sample drift",
                )
                _require(
                    repeat.get("repeat_index") == repeat_index
                    and repeat.get("repeat_factor") == unique.get("repeat_factor"),
                    f"{split}:{index}: repeat index/factor drift",
                )
                expected_id = _expected_row_id(stage, sample_id, repeat_index)
                _require(repeat.get("training_row_id") == expected_id, "training row ID drift")

        for unique in output_unique:
            repair = repairs_by_id.get(str(unique["sample_id"]))
            if repair is not None:
                unique["source_response_sha256"] = unique["response_sha256"]
                unique["response_sha256"] = repair["new_response_sha256"]
                unique["reasoning_repair_method"] = repair["method"]

        _write_jsonl(staging / "decision_sft" / f"{split}.jsonl", output_sft)
        _write_jsonl(staging / "decision_grpo" / f"{split}.jsonl", grpo_rows)
        _write_jsonl(staging / "manifests/unique" / f"{split}.jsonl", output_unique)
        _write_jsonl(staging / "manifests/repeats" / f"{split}.jsonl", repeat_rows)

    _require(set(repairs_by_id) == set(REASONING_REPAIRS), "repair population drift")
    repairs = [repairs_by_id[sample_id] for sample_id in sorted(repairs_by_id)]
    _write_jsonl(staging / "manifests/reasoning_repairs.jsonl", repairs)
    return repairs


def _percentiles(values: Sequence[int]) -> dict[str, int | float]:
    _require(bool(values), "cannot summarize empty token lengths")
    ordered = sorted(values)

    def percentile(value: float) -> float:
        position = (len(ordered) - 1) * value
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    def clean(value: float) -> int | float:
        rounded = round(value, 3)
        return int(rounded) if rounded.is_integer() else rounded

    return {
        "min": ordered[0],
        "p50": clean(percentile(0.50)),
        "p95": clean(percentile(0.95)),
        "max": ordered[-1],
    }


def _tokenizer_files(tokenizer_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name in (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.jinja",
        "config.json",
    ):
        path = tokenizer_root / name
        if path.is_file() and not path.is_symlink():
            records[name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
    _require("tokenizer_config.json" in records, "tokenizer config missing")
    _require("tokenizer.json" in records or "tokenizer.model" in records, "tokenizer missing")
    return records


def validate_clean_dataset(root: Path, tokenizer_root: Path) -> dict[str, Any]:
    """Deeply validate one clean materialized directory and real token contract."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    sample_ids: set[str] = set()
    meeting_dates: set[str] = set()
    prompt_hashes: dict[str, set[str]] = {}
    source_ids_by_split: dict[str, set[str]] = {}
    training_row_ids: set[str] = set()
    split_stats: dict[str, Any] = {}
    direction_unique: Counter[str] = Counter()
    direction_physical: Counter[str] = Counter()
    action_unique: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}

    for split in SPLITS:
        unique_rows = _read_jsonl(
            root / "manifests/unique" / f"{split}.jsonl", label=f"{split} unique"
        )
        sft_rows = _read_jsonl(
            root / "decision_sft" / f"{split}.jsonl", label=f"{split} SFT"
        )
        grpo_rows = _read_jsonl(
            root / "decision_grpo" / f"{split}.jsonl", label=f"{split} GRPO"
        )
        repeats = _read_jsonl(
            root / "manifests/repeats" / f"{split}.jsonl", label=f"{split} repeats"
        )
        _require(len(unique_rows) == EXPECTED_UNIQUE[split], f"{split}: unique count")
        _require(len(sft_rows) == len(grpo_rows) == EXPECTED_PHYSICAL[split], f"{split}: physical count")
        _require(len(repeats) == 2 * len(sft_rows), f"{split}: repeat row count")

        by_id: dict[str, dict[str, Any]] = {}
        expected: list[tuple[dict[str, Any], int]] = []
        split_sources: set[str] = set()
        split_prompt_hashes: set[str] = set()
        for row_index, unique in enumerate(unique_rows):
            sample_id = str(unique.get("sample_id") or "")
            meeting_date = str(unique.get("meeting_date") or "")
            _require(sample_id and sample_id not in sample_ids, f"{split}: duplicate sample")
            _require(meeting_date and meeting_date not in meeting_dates, f"{split}: duplicate meeting")
            _require(unique.get("split") == split, f"{split}: manifest split drift")
            _require(unique.get("population_role") == "core", f"{split}: unexpected supplement")
            _gold(unique.get("direction"), unique.get("magnitude_bp"))
            action_unique[split][
                f"{unique.get('direction')}:{unique.get('magnitude_bp')}"
            ] += 1
            factor = unique.get("repeat_factor")
            expected_factor = (
                EXPECTED_REPEAT_FACTORS[str(unique.get("direction"))]
                if split == "train"
                else 1
            )
            _require(factor == expected_factor, f"{split}:{sample_id}: repeat factor")
            prompt_sha = str(unique.get("prompt_sha256") or "")
            response_sha = str(unique.get("response_sha256") or "")
            gold_sha = str(unique.get("gold_sha256") or "")
            _require(_SHA_RE.fullmatch(prompt_sha) is not None, "invalid prompt SHA")
            _require(_SHA_RE.fullmatch(response_sha) is not None, "invalid response SHA")
            _require(_SHA_RE.fullmatch(gold_sha) is not None, "invalid gold SHA")
            _require(prompt_sha not in split_prompt_hashes, f"{split}: duplicate unique prompt")
            source_ids = unique.get("source_ids")
            _require(isinstance(source_ids, list) and source_ids, f"{split}: source IDs missing")
            _require(
                all(isinstance(item, str) and item for item in source_ids),
                f"{split}: invalid source ID",
            )
            _require(len(source_ids) == len(set(source_ids)), f"{split}: duplicate source ID")
            _require(not (split_sources & set(source_ids)), f"{split}: source ID reused")
            split_sources.update(source_ids)
            split_prompt_hashes.add(prompt_sha)
            sample_ids.add(sample_id)
            meeting_dates.add(meeting_date)
            by_id[sample_id] = unique
            direction_unique[str(unique["direction"])] += int(split == "train")
            for repeat_index in range(expected_factor):
                expected.append((unique, repeat_index))
        prompt_hashes[split] = split_prompt_hashes
        source_ids_by_split[split] = split_sources

        prompt_lengths: list[int] = []
        completion_lengths: list[int] = []
        total_lengths: list[int] = []
        for index, ((unique, repeat_index), sft, grpo) in enumerate(
            zip(expected, sft_rows, grpo_rows, strict=True)
        ):
            sample_id = str(unique["sample_id"])
            label = f"{split}:{index}:{sample_id}"
            _require(set(sft) == {"prompt", "response"}, f"{label}: SFT schema")
            _require(
                set(grpo) == {"sample_id", "prompt", "direction", "magnitude_bp"},
                f"{label}: GRPO schema",
            )
            _require(grpo["sample_id"] == sample_id, f"{label}: sample mapping")
            _require(sft["prompt"] == grpo["prompt"], f"{label}: prompt parity")
            analysis = _parse_prompt(
                sft["prompt"],
                meeting_date=str(unique["meeting_date"]),
                label=label,
                point_in_time_attested=True,
            )
            _require(sample_id not in analysis, f"{label}: prompt leaks sample ID")
            _require(
                not any(source_id in sft["prompt"] for source_id in unique["source_ids"]),
                f"{label}: prompt leaks source IDs",
            )
            _require(_sha256_text(sft["prompt"]) == unique["prompt_sha256"], f"{label}: prompt SHA")
            _require(_sha256_text(sft["response"]) == unique["response_sha256"], f"{label}: response SHA")
            _reasoning, decision = _parse_response(sft["response"], label=label)
            expected_gold = _gold(unique["direction"], unique["magnitude_bp"])
            _require(decision == expected_gold, f"{label}: SFT gold mismatch")
            _require(
                _gold(grpo["direction"], grpo["magnitude_bp"]) == expected_gold,
                f"{label}: GRPO gold mismatch",
            )
            _require(
                _sha256_text(_canonical_json(expected_gold)) == unique["gold_sha256"],
                f"{label}: gold SHA",
            )

            for offset, stage in enumerate(("decision_sft", "decision_grpo")):
                repeat = repeats[2 * index + offset]
                _require(repeat.get("schema_version") == REPEAT_SCHEMA, f"{label}: repeat schema")
                _require(repeat.get("stage") == stage, f"{label}: repeat stage")
                _require(repeat.get("source_sample_id") == sample_id, f"{label}: repeat sample")
                _require(repeat.get("split") == split, f"{label}: repeat split")
                _require(repeat.get("repeat_index") == repeat_index, f"{label}: repeat index")
                _require(repeat.get("repeat_factor") == unique["repeat_factor"], f"{label}: repeat factor")
                row_id = str(repeat.get("training_row_id") or "")
                _require(_ROW_RE.fullmatch(row_id) is not None, f"{label}: row ID syntax")
                _require(row_id == _expected_row_id(stage, sample_id, repeat_index), f"{label}: row ID")
                _require(row_id not in training_row_ids, f"{label}: duplicate training row ID")
                training_row_ids.add(row_id)

            messages = [
                {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
                {"role": "user", "content": sft["prompt"]},
            ]
            rendered = render_sft_prompt(tokenizer, messages)
            prompt_ids = tokenize_sft_text(tokenizer, rendered)
            completion = sft["response"]
            if not completion.endswith(tokenizer.eos_token):
                completion += tokenizer.eos_token
            full_ids = tokenize_sft_text(tokenizer, rendered + completion)
            _require(full_ids[: len(prompt_ids)] == prompt_ids, f"{label}: prefix drift")
            _require(full_ids.count(tokenizer.bos_token_id) == 1, f"{label}: BOS count")
            _require(full_ids[-1] == tokenizer.eos_token_id, f"{label}: final EOS")
            completion_ids = full_ids[len(prompt_ids) :]
            opening_ids = tokenizer("<think>", add_special_tokens=False)["input_ids"]
            boundary_ids = tokenizer("</think>", add_special_tokens=False)["input_ids"]
            _require(len(opening_ids) == 1, f"{label}: opening tokenization drift")
            _require(len(boundary_ids) == 1, f"{label}: boundary tokenization drift")
            _require(
                prompt_ids.count(opening_ids[0]) == 1,
                f"{label}: opening think token count",
            )
            _require(
                completion_ids.count(boundary_ids[0]) == 1,
                f"{label}: closing think token not supervised exactly once",
            )
            _require(len(prompt_ids) <= MAX_GRPO_PROMPT_TOKENS, f"{label}: prompt overflow")
            _require(len(full_ids) <= MAX_SFT_TOKENS, f"{label}: SFT overflow")
            prompt_lengths.append(len(prompt_ids))
            completion_lengths.append(len(completion_ids))
            total_lengths.append(len(full_ids))
            if split == "train":
                direction_physical[str(unique["direction"])] += 1

        split_stats[split] = {
            "unique_rows": len(unique_rows),
            "physical_rows": len(sft_rows),
            "prompt_tokens": _percentiles(prompt_lengths),
            "completion_tokens": _percentiles(completion_lengths),
            "total_tokens": _percentiles(total_lengths),
            "sft_overflow_rows": sum(value > MAX_SFT_TOKENS for value in total_lengths),
            "grpo_prompt_overflow_rows": sum(
                value > MAX_GRPO_PROMPT_TOKENS for value in prompt_lengths
            ),
        }

    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            _require(not (prompt_hashes[left] & prompt_hashes[right]), "cross-split prompt overlap")
            _require(
                not (source_ids_by_split[left] & source_ids_by_split[right]),
                "cross-split source overlap",
            )
    _require(len(sample_ids) == 128, "global sample count drift")
    _require(len(meeting_dates) == 128, "global meeting count drift")
    _require(dict(direction_unique) == EXPECTED_UNIQUE_TRAIN_DIRECTIONS, "train direction drift")
    _require(dict(direction_physical) == EXPECTED_PHYSICAL_TRAIN_DIRECTIONS, "physical direction drift")

    train_actions = set(action_unique["train"])
    out_of_support = {
        split: {
            action: count
            for action, count in sorted(action_unique[split].items())
            if action not in train_actions
        }
        for split in ("validation", "test")
    }

    repairs = _read_jsonl(
        root / "manifests/reasoning_repairs.jsonl", label="reasoning repairs"
    )
    _require(len(repairs) == len(REASONING_REPAIRS), "repair count drift")
    _require({row.get("sample_id") for row in repairs} == set(REASONING_REPAIRS), "repair IDs drift")
    _require(
        all(row.get("prompt_and_final_decision_unchanged") is True for row in repairs),
        "repair scope drift",
    )

    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed",
        "grain": "one unique pre-meeting decision brief per meeting; physical train repeats are manifest-bound",
        "unique_split_counts": EXPECTED_UNIQUE,
        "physical_split_counts": {
            "decision_sft": EXPECTED_PHYSICAL,
            "decision_grpo": EXPECTED_PHYSICAL,
        },
        "unique_meetings": len(meeting_dates),
        "unique_sample_ids": len(sample_ids),
        "training_row_ids": len(training_row_ids),
        "train_unique_direction_counts": dict(direction_unique),
        "train_physical_direction_counts": dict(direction_physical),
        "unique_action_counts": {
            split: dict(sorted(counts.items())) for split, counts in action_unique.items()
        },
        "evaluation_actions_absent_from_train": out_of_support,
        "evaluation_meetings_with_action_absent_from_train": sum(
            sum(counts.values()) for counts in out_of_support.values()
        ),
        "repeat_factors": EXPECTED_REPEAT_FACTORS,
        "reasoning_repairs": len(repairs),
        "prompt_leakage_violations": 0,
        "cross_split_overlaps": 0,
        "sft_grpo_prompt_mismatches": 0,
        "label_mismatches": 0,
        "invalid_reasoning_boundaries": 0,
        "reasoning_digit_or_date_violations": 0,
        "reasoning_explicit_quantity_or_date_violations": 0,
        "split_token_stats": split_stats,
        "token_contract": {
            "tokenizer_path": str(tokenizer_root),
            "tokenizer_files": _tokenizer_files(tokenizer_root),
            "tokenizer_class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}",
            "loader_parameters": {
                "local_files_only": True,
                "fix_mistral_regex": "omitted_to_match_current_training_runtime",
            },
            "runtime_versions": {
                "python": platform.python_version(),
                "transformers": importlib.metadata.version("transformers"),
                "tokenizers": importlib.metadata.version("tokenizers"),
                "trl": importlib.metadata.version("trl"),
            },
            "single_bos": True,
            "final_eos": True,
            "completion_mask_covers_reasoning_boundary_answer_eos": True,
            "sft_max_length": MAX_SFT_TOKENS,
            "grpo_max_prompt_length": MAX_GRPO_PROMPT_TOKENS,
            "grpo_max_completion_length": MAX_GRPO_COMPLETION_TOKENS,
            "truncation": False,
        },
        "test_is_sealed_evaluation_only": True,
        "supplement_admitted": 0,
        "population_scope": "core-only",
    }


def _file_records(root: Path, *, excluded: set[str] | None = None) -> dict[str, Any]:
    excluded = excluded or set()
    records: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        records[relative] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        if path.suffix == ".jsonl":
            records[relative]["rows"] = len(path.read_text(encoding="utf-8").splitlines())
    return records


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)


def _unseal_staging(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    root.chmod(0o700)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(renameat2 is not None, "renameat2(RENAME_NOREPLACE) unavailable")
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
        raise Chk4ReleaseError(f"immutable destination already exists: {destination}")
    raise Chk4ReleaseError(f"atomic publish failed: {os.strerror(error)}")


def verify_release(
    root: Path, *, expected_manifest_sha256: str | None = None
) -> dict[str, Any]:
    _require(not root.is_symlink(), f"release root is a symlink: {root}")
    root = root.resolve()
    manifest_path = root / "release_manifest.json"
    if expected_manifest_sha256 is not None:
        _require(
            _SHA_RE.fullmatch(expected_manifest_sha256) is not None,
            "expected release manifest SHA is invalid",
        )
        _require(
            _sha256_file(manifest_path) == expected_manifest_sha256,
            "release manifest does not match the externally pinned SHA",
        )
    manifest = _read_json(manifest_path, label="release manifest")
    _require(manifest.get("schema_version") == RELEASE_SCHEMA, "release schema drift")
    _require(manifest.get("quality_status") == "passed", "release is not passed")
    _require(manifest.get("immutable") is True, "release is not immutable")
    _require(manifest.get("training_ready") is True, "release is not training ready")
    files = manifest.get("files")
    _require(isinstance(files, dict) and files, "release file records missing")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    _require(
        actual_files == set(files) | {"release_manifest.json", "handoff.json"},
        "release contains missing or unlisted files",
    )
    for relative, record in files.items():
        _require(isinstance(relative, str) and isinstance(record, dict), "invalid file record")
        path = root / relative
        _require(path.is_file() and not path.is_symlink(), f"release file missing: {relative}")
        _require(path.stat().st_size == record.get("bytes"), f"size drift: {relative}")
        _require(_sha256_file(path) == record.get("sha256"), f"hash drift: {relative}")
        if relative.endswith(".jsonl"):
            rows = len(path.read_text(encoding="utf-8").splitlines())
            _require(rows == record.get("rows"), f"row count drift: {relative}")
    handoff = _read_json(root / "handoff.json", label="handoff")
    _require(handoff.get("schema_version") == HANDOFF_SCHEMA, "handoff schema drift")
    _require(handoff.get("quality_status") == "passed", "handoff is not passed")
    _require(handoff.get("immutable") is True, "handoff is not immutable")
    _require(
        handoff.get("release_manifest_sha256") == _sha256_file(root / "release_manifest.json"),
        "handoff manifest binding drift",
    )
    handoff_record = manifest.get("handoff")
    _require(isinstance(handoff_record, dict), "manifest handoff binding missing")
    unsigned_handoff = dict(handoff)
    unsigned_handoff.pop("release_manifest_sha256", None)
    _require(
        handoff_record
        == {
            "path": "handoff.json",
            "schema_version": HANDOFF_SCHEMA,
            "unsigned_payload_sha256": _sha256_text(
                _canonical_json(unsigned_handoff)
            ),
        },
        "handoff payload binding drift",
    )
    contract = _read_json(
        root / "contracts/decision_input_contract.json", label="decision input contract"
    )
    _require(contract.get("schema_version") == INPUT_CONTRACT_SCHEMA, "input contract schema drift")
    contract_digest = contract.get("contract_sha256")
    unsigned_contract = dict(contract)
    unsigned_contract.pop("contract_sha256", None)
    _require(
        contract_digest == _sha256_text(_canonical_json(unsigned_contract)),
        "input contract self-digest drift",
    )
    _require(
        manifest.get("input_contract", {}).get("contract_sha256") == contract_digest,
        "manifest/input contract binding drift",
    )
    data_quality = _read_json(root / "audits/data_quality.json", label="data-quality audit")
    point_in_time = _read_json(
        root / "audits/point_in_time_lineage.json", label="point-in-time audit"
    )
    _require(data_quality.get("status") == "passed", "data-quality audit did not pass")
    _require(point_in_time.get("status") == "passed", "point-in-time audit did not pass")
    _require(
        data_quality.get("input_contract", {}).get("contract_sha256") == contract_digest,
        "data-quality/input contract binding drift",
    )

    sources = manifest.get("sources")
    _require(isinstance(sources, dict), "release sources missing")
    source_candidate = sources.get("materialized_candidate")
    teacher_source = sources.get("teacher_acquisition")
    brief_source = sources.get("gold_blind_meeting_briefs")
    canonical_source = sources.get("canonical_chk1_analysis_source")
    tokenizer_source = sources.get("tokenizer")
    _require(
        all(
            isinstance(value, dict)
            for value in (
                source_candidate,
                teacher_source,
                brief_source,
                canonical_source,
                tokenizer_source,
            )
        ),
        "release source records are incomplete",
    )
    assert isinstance(source_candidate, dict)
    assert isinstance(teacher_source, dict)
    assert isinstance(brief_source, dict)
    assert isinstance(canonical_source, dict)
    assert isinstance(tokenizer_source, dict)
    source_path = Path(str(source_candidate.get("path"))).expanduser().resolve()
    teacher_path = Path(str(teacher_source.get("path"))).expanduser().resolve()
    brief_path = Path(str(brief_source.get("path"))).expanduser().resolve()
    canonical_path = Path(str(canonical_source.get("path"))).expanduser().resolve()
    tokenizer_path = Path(str(tokenizer_source.get("path"))).expanduser().resolve()
    _require(
        _tree_sha256(source_path) == source_candidate.get("tree_sha256"),
        "materialized source tree drift",
    )
    _require(
        _tree_sha256(teacher_path) == teacher_source.get("tree_sha256"),
        "teacher source tree drift",
    )
    _require(
        _tree_sha256(brief_path) == brief_source.get("tree_sha256"),
        "brief source tree drift",
    )
    _require(
        _tree_sha256(canonical_path) == canonical_source.get("tree_sha256"),
        "canonical chk1 source tree drift",
    )
    _require(
        _tokenizer_files(tokenizer_path) == tokenizer_source.get("files"),
        "tokenizer bundle drift",
    )
    publisher_source = sources.get("publisher")
    _require(isinstance(publisher_source, dict), "publisher source record missing")
    publisher_snapshot = root / str(publisher_source.get("snapshot"))
    renderer_snapshot = root / str(
        publisher_source.get("sft_prompt_renderer_snapshot")
    )
    _require(
        _sha256_file(publisher_snapshot)
        == publisher_source.get("snapshot_sha256")
        == publisher_source.get("sha256"),
        "publisher snapshot binding drift",
    )
    _require(
        _sha256_file(renderer_snapshot)
        == publisher_source.get("sft_prompt_renderer_snapshot_sha256"),
        "SFT renderer snapshot binding drift",
    )

    replay_lineage, replay_rows = _validate_external_lineage(
        clean_root=root,
        teacher=teacher_path,
        briefs=brief_path,
    )
    _require(
        replay_lineage.get("canonical_chk1", {}).get("path") == str(canonical_path)
        and replay_lineage.get("canonical_chk1", {}).get("tree_sha256")
        == canonical_source.get("tree_sha256")
        and replay_lineage.get("canonical_chk1", {}).get("source_rows")
        == canonical_source.get("source_rows")
        and replay_lineage.get("canonical_chk1", {}).get("evidence_rows_checked")
        == canonical_source.get("evidence_rows_checked"),
        "manifest/canonical chk1 replay binding drift",
    )
    for key, value in replay_lineage.items():
        _require(point_in_time.get(key) == value, f"lineage semantic replay drift: {key}")
    persisted_rows = _read_jsonl(
        root / "audits/point_in_time_rows.jsonl", label="point-in-time lineage rows"
    )
    _require(persisted_rows == replay_rows, "point-in-time row replay drift")
    replay_data_quality = validate_clean_dataset(root, tokenizer_path)
    for key, value in replay_data_quality.items():
        _require(data_quality.get(key) == value, f"data-quality semantic replay drift: {key}")
    for path in root.rglob("*"):
        _require(not path.is_symlink(), f"release contains symlink: {path}")
        mode = path.stat().st_mode & 0o777
        _require(mode == (0o444 if path.is_file() else 0o555), f"mutable mode: {path}")
    _require((root.stat().st_mode & 0o777) == 0o555, "release root is mutable")
    return manifest


def publish(
    *,
    source: Path,
    teacher: Path,
    briefs: Path,
    tokenizer: Path,
    destination: Path,
) -> dict[str, Any]:
    source = source.resolve()
    teacher = teacher.resolve()
    briefs = briefs.resolve()
    tokenizer = tokenizer.resolve()
    publisher_bytes = Path(__file__).resolve().read_bytes()
    renderer_bytes = SFT_RENDERER_SOURCE.read_bytes()
    publisher_sha256 = _sha256_bytes(publisher_bytes)
    renderer_sha256 = _sha256_bytes(renderer_bytes)
    raw_destination = destination.absolute()
    _require(
        not os.path.lexists(raw_destination),
        f"destination exists: {raw_destination}",
    )
    destination = raw_destination.parent.resolve() / raw_destination.name
    _require(
        not os.path.lexists(destination),
        f"destination exists after canonicalization: {destination}",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_info = _validate_source_summaries(source=source, teacher=teacher, briefs=briefs)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent))
    renamed = False
    try:
        repairs = _materialize_clean_copy(source, staging)
        provenance = staging / "provenance"
        _write_json(provenance / "source_materialization_summary.json", source_info["source_summary"])
        _write_json(provenance / "teacher_summary.json", source_info["teacher_summary"])
        _write_json(provenance / "meeting_brief_summary.json", source_info["brief_summary"])
        _write_bytes(provenance / "publisher_snapshot.py", publisher_bytes)
        _write_bytes(
            provenance / "sft_prompt_renderer_snapshot.py",
            renderer_bytes,
        )
        input_contract = _build_input_contract(source_info)
        _write_json(staging / "contracts/decision_input_contract.json", input_contract)
        lineage_audit, lineage_rows = _validate_external_lineage(
            clean_root=staging,
            teacher=teacher,
            briefs=briefs,
        )
        lineage_audit = {**lineage_audit, "created_at_utc": _utc_now()}
        _write_json(staging / "audits/point_in_time_lineage.json", lineage_audit)
        _write_jsonl(staging / "audits/point_in_time_rows.jsonl", lineage_rows)
        audit = validate_clean_dataset(staging, tokenizer)
        audit = {
            **audit,
            "created_at_utc": _utc_now(),
            "input_contract": {
                "path": "contracts/decision_input_contract.json",
                "schema_version": INPUT_CONTRACT_SCHEMA,
                "contract_sha256": input_contract["contract_sha256"],
            },
            "point_in_time_lineage_audit": {
                "path": "audits/point_in_time_lineage.json",
                "status": lineage_audit["status"],
                "rows": lineage_audit["rows"],
            },
        }
        _write_json(staging / "audits/data_quality.json", audit)

        files = _file_records(staging)
        release_id = destination.name
        handoff_unsigned = {
            "schema_version": HANDOFF_SCHEMA,
            "release_id": release_id,
            "created_at_utc": _utc_now(),
            "quality_status": "passed",
            "immutable": True,
            "training_ready": True,
            "release_manifest": "release_manifest.json",
            "decision_sft_path": "decision_sft",
            "decision_grpo_path": "decision_grpo",
            "unique_split_counts": EXPECTED_UNIQUE,
            "physical_split_counts": EXPECTED_PHYSICAL,
            "test_is_sealed_evaluation_only": True,
            "population_scope": "core-only",
            "target_blindness": "target-decision-blind",
            "semantic_assurance": {
                "structural_lineage_replay": "passed",
                "known_pattern_target_outcome_hits": 0,
                "independent_source_only_semantic_judge": "not_run",
            },
            "known_limitations": [
                "train repeats reduce but do not eliminate direction imbalance",
                "seven validation/test meetings use magnitudes absent from train",
                "the 1993-2008 supplement is not admitted",
                "eight upstream chk1 selected-evidence citation lists contain unresolved IDs; "
                "the full provided evidence lineage remains hash-bound and point-in-time valid",
                "teacher SFT rationales are deterministic local targets, not an independent "
                "source-only semantic judgment",
            ],
            "required_model_path": (
                "selected chk1 -> decision_sft -> merged chk4_sft -> decision_grpo"
            ),
        }
        handoff_payload_sha256 = _sha256_text(_canonical_json(handoff_unsigned))
        manifest = {
            "schema_version": RELEASE_SCHEMA,
            "release_id": release_id,
            "created_at_utc": _utc_now(),
            "quality_status": "passed",
            "immutable": True,
            "training_ready": True,
            "canonical_dag_bindable": False,
            "canonical_dag_blocker": (
                "the current canonical DAG expects chk2 as parent; this release is for an explicit "
                "chk1 -> chk4_sft -> chk4 standalone branch"
            ),
            "grain": "one unique target-decision-blind pre-meeting brief per meeting",
            "population_scope": "core-only",
            "target_blindness": "target-decision-blind",
            "semantic_assurance": {
                "structural_lineage_replay": "passed",
                "known_pattern_target_outcome_hits": 0,
                "independent_source_only_semantic_judge": "not_run",
            },
            "unique_split_counts": EXPECTED_UNIQUE,
            "physical_split_counts": {
                "decision_sft": EXPECTED_PHYSICAL,
                "decision_grpo": EXPECTED_PHYSICAL,
            },
            "training_roles": {
                "decision_sft": {
                    "dataset_path": "decision_sft",
                    "parent_role": "selected_chk1_merged",
                    "max_length": MAX_SFT_TOKENS,
                    "completion_only_loss": True,
                },
                "decision_grpo": {
                    "dataset_path": "decision_grpo",
                    "parent_role": "merged_decision_sft_warm_start",
                    "reward": "decision_dense_v2",
                    "max_prompt_length": MAX_GRPO_PROMPT_TOKENS,
                    "max_completion_length": MAX_GRPO_COMPLETION_TOKENS,
                },
            },
            "sources": {
                "materialized_candidate": {
                    "path": str(source),
                    "tree_sha256": source_info["source_tree_sha256"],
                },
                "teacher_acquisition": {
                    "path": str(teacher),
                    "tree_sha256": source_info["teacher_tree_sha256"],
                    "model": source_info["teacher_summary"].get("teacher_model"),
                    "provider_identity": source_info["teacher_summary"].get("provider_identity"),
                },
                "gold_blind_meeting_briefs": {
                    "path": str(briefs),
                    "tree_sha256": source_info["brief_tree_sha256"],
                    "source_chk1_sha256": source_info["brief_summary"].get("source_chk1_sha256"),
                },
                "canonical_chk1_analysis_source": {
                    "path": lineage_audit["canonical_chk1"]["path"],
                    "tree_sha256": lineage_audit["canonical_chk1"]["tree_sha256"],
                    "source_rows": lineage_audit["canonical_chk1"]["source_rows"],
                    "evidence_rows_checked": lineage_audit["canonical_chk1"][
                        "evidence_rows_checked"
                    ],
                },
                "tokenizer": {
                    "path": str(tokenizer),
                    "files": audit["token_contract"]["tokenizer_files"],
                },
                "publisher": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": publisher_sha256,
                    "snapshot": "provenance/publisher_snapshot.py",
                    "snapshot_sha256": publisher_sha256,
                    "sft_prompt_renderer_snapshot": (
                        "provenance/sft_prompt_renderer_snapshot.py"
                    ),
                    "sft_prompt_renderer_snapshot_sha256": renderer_sha256,
                },
            },
            "repairs": {
                "count": len(repairs),
                "manifest": "manifests/reasoning_repairs.jsonl",
                "scope": "reasoning-only; prompts and final decision JSON unchanged",
            },
            "input_contract": {
                "path": "contracts/decision_input_contract.json",
                "schema_version": INPUT_CONTRACT_SCHEMA,
                "contract_sha256": input_contract["contract_sha256"],
                "supersedes_without_overwriting": source_info["brief_summary"].get(
                    "contract_sha256"
                ),
            },
            "known_coverage_limits": {
                "train_unique_direction_counts": EXPECTED_UNIQUE_TRAIN_DIRECTIONS,
                "train_physical_direction_counts": EXPECTED_PHYSICAL_TRAIN_DIRECTIONS,
                "evaluation_meetings_with_action_absent_from_train": audit[
                    "evaluation_meetings_with_action_absent_from_train"
                ],
                "supplement_admitted": 0,
            },
            "handoff": {
                "path": "handoff.json",
                "schema_version": HANDOFF_SCHEMA,
                "unsigned_payload_sha256": handoff_payload_sha256,
            },
            "test_is_sealed_evaluation_only": True,
            "supplement_admitted": 0,
            "files": files,
        }
        _write_json(staging / "release_manifest.json", manifest)
        manifest_sha = _sha256_file(staging / "release_manifest.json")
        handoff = {**handoff_unsigned, "release_manifest_sha256": manifest_sha}
        _write_json(staging / "handoff.json", handoff)
        _seal_tree(staging)
        verify_release(staging, expected_manifest_sha256=manifest_sha)
        _rename_noreplace(staging, destination)
        renamed = True
        verified = verify_release(
            destination, expected_manifest_sha256=manifest_sha
        )
        return {
            "status": "published",
            "release_id": verified["release_id"],
            "destination": str(destination),
            "release_manifest_sha256": _sha256_file(destination / "release_manifest.json"),
            "handoff_sha256": _sha256_file(destination / "handoff.json"),
            "audit_sha256": _sha256_file(destination / "audits/data_quality.json"),
        }
    except Exception:
        if staging.exists():
            _unseal_staging(staging)
            shutil.rmtree(staging)
        if renamed and destination.exists():
            _unseal_staging(destination)
            shutil.rmtree(destination)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    publish_parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    publish_parser.add_argument("--briefs", type=Path, default=DEFAULT_BRIEFS)
    publish_parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    publish_parser.add_argument("--destination", type=Path, default=DEFAULT_OUTPUT)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--release", type=Path, default=DEFAULT_OUTPUT)
    verify_parser.add_argument("--expected-manifest-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "publish":
            result = publish(
                source=args.source,
                teacher=args.teacher,
                briefs=args.briefs,
                tokenizer=args.tokenizer,
                destination=args.destination,
            )
        else:
            verified = verify_release(
                args.release,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
            result = {
                "status": "verified",
                "release_id": verified["release_id"],
                "release_manifest_sha256": _sha256_file(
                    args.release.resolve() / "release_manifest.json"
                ),
            }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (Chk4ReleaseError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
