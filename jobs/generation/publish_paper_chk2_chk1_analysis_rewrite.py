"""Publish the paper chk-2 dataset built from exact chk-1 final analyses.

This module is intentionally API-free.  It independently reconstructs the
1,743-row chk-1 population, verifies the terminal acquisition partition,
recomputes the style gate, replays the exact cp200 tokenizer contract, and
atomically publishes only rows that pass every gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from jobs.generation import generate_paper_chk2_chk1_analysis_rewrite as generator
from jobs.generation.paper_chk2_chk1_source import (
    DEFAULT_GENERATION_ROOT,
    EXPECTED_SOURCE_FILE_SHA256,
)
from jobs.generation.paper_chk2_official_reference_v2 import (
    DEFAULT_OFFICIAL_ROSTER_PATH,
    EXPECTED_BOUNDARY_METHOD_COUNTS,
    EXPECTED_OFFICIAL_ROSTER_SHA256,
    PINNED_2009_ACTION_BOUNDARIES,
    OfficialReferenceBank,
    OfficialReferenceError,
    build_official_reference_bank,
    deserialize_official_reference_bank,
    serialize_official_reference_bank,
    verify_official_reference_bank,
)
from open_r1.trainer.sft_prompt_renderer import render_sft_prompt, tokenize_sft_text


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT / "output/data/retrain_v2/chk1/"
    "chk1_reasoning_compressed_flash_max_v2_clean_20260809_candidate"
)
DEFAULT_GENERATION_MANIFEST_ROOT = DEFAULT_GENERATION_ROOT
DEFAULT_ACQUISITION_ROOT = (
    REPO_ROOT / "output/data/retrain_v2/chk2/"
    "chk1_final_analysis_to_minutes_flash_official_reference_v2_20260831"
)
DEFAULT_RELEASE_ROOT = (
    REPO_ROOT / "dataset/processed/retrain_v2/"
    "chk2_chk1_final_analysis_synthetic_minutes_flash_official_reference_v2_20260831"
)
DEFAULT_TOKENIZER_PATH = (
    REPO_ROOT / "output/training/retrain_v2/"
    "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
)

SPLITS = ("train", "validation", "test")
SOURCE_SPLITS = {"train": "train", "validation": "eval", "test": "test"}
SOURCE_SPLIT_COUNTS = {"train": 1354, "validation": 199, "test": 190}
SOURCE_MEETING_COUNTS = {"train": 102, "validation": 13, "test": 13}
SOURCE_SAMPLE_ID_SHA256 = (
    "e3ec34a76fa720488e9aee50ad39b6bd53114fa6ddcc5220078ffc7139d4b8db"
)
SOURCE_FILE_SHA256 = dict(EXPECTED_SOURCE_FILE_SHA256)

RELEASE_SCHEMA_VERSION = "paper-chk2-chk1-analysis-rewrite-release-v5"
TERMINAL_SCHEMA_VERSION = generator.TERMINAL_SCHEMA_VERSION
DATASET_ROLE = "paper_chk2_chk1_final_analysis_to_synthetic_minutes_sft"
TRAINING_SCOPE = "paper-chk2-chk1-cp200-minutes-sft-v5"
BOUNDARY = "\n</think>\n"
MAX_TOTAL_TOKENS = 4096

STYLE_DIMENSIONS = tuple(generator.STYLE_DIMENSIONS)
STYLE_MEAN_THRESHOLD = 7.0
STYLE_MIN_THRESHOLD = 6

TERMINAL_FIELDS = {
    "schema_version",
    "sample_id",
    "split",
    "source_split",
    "source_index",
    "meeting_date",
    "atomic_topic",
    "section_style_id",
    "terminal_status",
    "training_pass",
    "rejection_stage",
    "rejection_reasons",
    "provided_data_sha256",
    "source_analysis",
    "source_analysis_sha256",
    "source_audit",
    "teacher_response_analysis",
    "rewritten_minutes",
    "student_prompt",
    "sft_response",
    "teacher_response_analysis_sha256",
    "rewritten_minutes_sha256",
    "prompt_sha256",
    "response_sha256",
    "generation",
    "validator_a",
    "validator_b",
    "repair_history",
    "lineage",
}
AUDIT_FIELDS = {"complete", "machine_pass", "reasons", "result", "provider"}
SOURCE_AUDIT_FIELDS = {
    "complete",
    "machine_pass",
    "reasons",
    "contract_repair_used",
    "result",
    "provider",
}
GENERATION_FIELDS = {
    "selected_attempt",
    "fidelity_repair_used",
    "style_repair_used",
    "deterministic_validation",
    "provider",
}
DETERMINISTIC_FIELDS = {"machine_pass", "reasons", "diagnostics"}
VALIDATOR_B_FIELDS = {
    "complete",
    "machine_pass",
    "mean_score",
    "min_score",
    "reasons",
    "result",
    "provider",
}
TERMINAL_STATUSES = {
    "PASS",
    "SOURCE_QUALITY_REJECT",
    "SOURCE_AUDIT_CONTRACT_REJECT",
    "GENERATION_QUALITY_REJECT",
    "INPUT_FIDELITY_REJECT",
    "VALIDATOR_A_CONTRACT_REJECT",
    "STYLE_QUALITY_REJECT",
    "STYLE_REPAIR_FIDELITY_REJECT",
    "VALIDATOR_B_CONTRACT_REJECT",
    "REFERENCE_COMPARISON_UNAVAILABLE_REJECT",
}
REQUIRED_LINEAGE = dict(generator.LINEAGE)

_CONTROL_RE = re.compile(r"<think>|</think>|<answer>|</answer>", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'’][A-Za-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_FINAL_BAD_RE = re.compile(
    r"(?:^|\n)\s*(?:[-*#]|•|\d+[.)])\s|```|\{\s*[\"']|"
    r"\b(?:here is|the answer is|as an ai|rewrite:)\b",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(
    r"https?://|\bwww\.|\bdoi\s*:|\[\s*\d+(?:\s*,\s*\d+)*\s*\]|"
    r"\([A-Z][A-Za-z'-]+(?:\s+et\s+al\.)?\s*,\s*(?:19|20)\d{2}[a-z]?\)|"
    r"(?:^|\s)(?:sources?|references?)\s*:",
    re.IGNORECASE,
)
_CONTROL_MARKERS = (
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
    "<|channel>",
    "<channel|>",
    "<｜Assistant｜>",
    "<｜User｜>",
    "<｜begin▁of▁sentence｜>",
    "<｜end▁of▁sentence｜>",
)
_NUMBER_ATOM = (
    r"(?:\d{1,3}(?:,\d{3})+|\d+)-\d+/\d+"
    r"|\d+/\d+"
    r"|(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
)
_NUMBER_EXPRESSION_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?P<number>{_NUMBER_ATOM})"
    r"(?:\s*(?P<scale>thousand|million|billion|trillion))?"
    r"(?:\s*(?P<rate>percentage\s+points?|percent|basis\s+points?|bps?|%))?"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_MONTH_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:\.)?(?![A-Za-z])"
)
_MONTH_DAY_RE = re.compile(
    r"\b(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:\.)?\s+"
    r"(?P<day>(?:[12]?\d|3[01]))(?:st|nd|rd|th)?\b"
)
_YEAR_RE = re.compile(r"(?<![\d,])(?:19|20)\d{2}(?!\d)(?!\.\d)")
_EVIDENCE_ID_RE = re.compile(r"\bev-[0-9a-f]{6,}\b", re.IGNORECASE)
_SCALE_FACTORS = {
    "thousand": Decimal("1000"),
    "million": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "trillion": Decimal("1000000000000"),
}
_ATTRIBUTION_PATTERNS = {
    "staff": (re.compile(r"\bstaff\b", re.IGNORECASE),),
    "participants": (
        re.compile(
            r"\b(?:participants?|members?|officials?|policymakers?)\b",
            re.IGNORECASE,
        ),
    ),
    "federal_reserve_body": (
        re.compile(
            r"\b(?:the\s+)?(?:Committee|FOMC|Board|Federal\s+Reserve|Fed)\b",
            re.IGNORECASE,
        ),
    ),
    "meeting_discussion": (
        re.compile(r"\b(?:meetings?|discussions?|deliberations?)\b", re.IGNORECASE),
    ),
    "vote_decision": (
        re.compile(
            r"\b(?:votes?|voted|voting|decisions?|decided|policy\s+actions?)\b",
            re.IGNORECASE,
        ),
    ),
}
_SECRET_RE = re.compile(r"(?:Bearer\s+[A-Za-z0-9._-]{12,}|\bsk-[A-Za-z0-9_-]{12,})")
_SECRET_KEYS = {"api_key", "apikey", "authorization", "access_token", "secret"}
_CREDENTIAL_METADATA_KEYS = {"api_key_env", "credential_env", "api_key_source"}


class PublicationError(ValueError):
    """Raised when acquisition or release bytes fail closed."""


@dataclass(frozen=True)
class SourceRow:
    sample_id: str
    split: str
    source_split: str
    source_index: int
    meeting_date: str
    atomic_topic: str
    section_style_id: str
    provided_data: str
    source_analysis: str
    original_prompt_sha256: str
    candidate_response_sha256: str

    @property
    def provided_data_sha256(self) -> str:
        return sha256_text(self.provided_data)

    @property
    def source_analysis_sha256(self) -> str:
        return sha256_text(self.source_analysis)

    @property
    def student_prompt(self) -> str:
        return render_user_prompt(self.source_analysis)


def _prepared_row_for_terminal_replay(source: SourceRow) -> generator.PreparedRow:
    """Project publisher source lineage into the generator's resume contract."""

    placeholder_sha = "0" * 64
    return generator.PreparedRow(
        sample_id=source.sample_id,
        split=source.split,
        source_split=source.source_split,
        split_index=source.source_index,
        source_line_number=source.source_index,
        generation_manifest_line_number=source.source_index,
        meeting_date=source.meeting_date,
        atomic_topic=source.atomic_topic,
        section_style_id=source.section_style_id,
        prompt="",
        provided_data=source.provided_data,
        source_analysis=source.source_analysis,
        prompt_sha256=source.original_prompt_sha256,
        provided_data_sha256=source.provided_data_sha256,
        source_analysis_sha256=source.source_analysis_sha256,
        candidate_response_sha256=source.candidate_response_sha256,
        source_answer_sha256=source.source_analysis_sha256,
        source_response_sha256=placeholder_sha,
        generation_manifest_row_sha256=placeholder_sha,
        source_row_sha256=placeholder_sha,
    )


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PublicationError(f"cannot hash file: {path}: {exc}") from exc
    return digest.hexdigest()


def render_user_prompt(source_analysis: str) -> str:
    return (
        "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
        + json.dumps(
            {"analysis": source_analysis},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in _SENTENCE_RE.split(text.strip()) if item.strip()]


def _normalized_prose(text: str) -> str:
    return " ".join(_EVIDENCE_ID_RE.sub("", text).casefold().split())


def _decimal_number_atom(raw: str) -> Decimal:
    normalized = raw.replace(",", "")
    mixed = re.fullmatch(r"(?P<whole>\d+)-(?P<num>\d+)/(?P<den>\d+)", normalized)
    if mixed is not None:
        denominator = Decimal(mixed.group("den"))
        if denominator == 0:
            raise InvalidOperation("fraction denominator is zero")
        return Decimal(mixed.group("whole")) + Decimal(mixed.group("num")) / denominator
    fraction = re.fullmatch(r"(?P<num>\d+)/(?P<den>\d+)", normalized)
    if fraction is not None:
        denominator = Decimal(fraction.group("den"))
        if denominator == 0:
            raise InvalidOperation("fraction denominator is zero")
        return Decimal(fraction.group("num")) / denominator
    return Decimal(normalized)


def _canonical_decimal(value: Decimal) -> str:
    return format((Decimal(0) if value == 0 else value).normalize(), "f")


def _numeric_values(text: str) -> Counter[str]:
    values: Counter[str] = Counter()
    without_citations = _EVIDENCE_ID_RE.sub("", text)
    without_days = _MONTH_DAY_RE.sub(
        lambda match: str(match.group("month")), without_citations
    )
    for match in _NUMBER_EXPRESSION_RE.finditer(without_days):
        raw = match.group("number")
        if (
            not match.group("scale")
            and not match.group("rate")
            and re.fullmatch(r"(?:19|20)\d{2}", raw)
        ):
            continue
        try:
            value = _decimal_number_atom(raw)
        except (InvalidOperation, ZeroDivisionError):
            continue
        scale = str(match.group("scale") or "").casefold()
        if scale:
            value *= _SCALE_FACTORS[scale]
        rate = re.sub(r"\s+", " ", str(match.group("rate") or "").casefold())
        if rate in {"basis point", "basis points", "bp", "bps"}:
            value /= Decimal("100")
        values[_canonical_decimal(value)] += 1
    return values


def _date_values(text: str) -> set[str]:
    aliases = {
        "jan": "january",
        "feb": "february",
        "mar": "march",
        "apr": "april",
        "jun": "june",
        "jul": "july",
        "aug": "august",
        "sep": "september",
        "sept": "september",
        "oct": "october",
        "nov": "november",
        "dec": "december",
    }
    text = _EVIDENCE_ID_RE.sub("", text)
    result = {
        aliases.get(
            match.group(0).casefold().rstrip("."), match.group(0).casefold().rstrip(".")
        )
        for match in _MONTH_RE.finditer(text)
    }
    for match in _MONTH_DAY_RE.finditer(text):
        month = match.group("month").casefold().rstrip(".")
        result.add(f"{aliases.get(month, month)}-{int(match.group('day'))}")
    result.update(match.group(0) for match in _YEAR_RE.finditer(text))
    return result


def _attribution_categories(text: str) -> set[str]:
    return {
        category
        for category, patterns in _ATTRIBUTION_PATTERNS.items()
        if any(pattern.search(text) for pattern in patterns)
    }


def _projection_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _id_digest(values: Sequence[str]) -> str:
    return sha256_text("".join(f"{value}\n" for value in sorted(values)))


def _ordered_source_id_digest(values: Sequence[str]) -> str:
    return sha256_text(canonical_json(list(values)))


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationError(message)


def _string(value: Any, *, label: str) -> str:
    _require(
        isinstance(value, str) and bool(value.strip()),
        f"{label} must be a non-empty string",
    )
    return str(value)


def _sha(value: Any, *, label: str) -> str:
    text = _string(value, label=label)
    _require(
        bool(re.fullmatch(r"[0-9a-f]{64}", text)), f"{label} must be lowercase SHA-256"
    )
    return text


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _string_list(value: Any, *, label: str, allow_empty: bool = True) -> list[str]:
    _require(
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, str) and bool(item) for item in value),
        f"{label} must be a list of non-empty strings",
    )
    return list(value)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationError(f"invalid {label}: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PublicationError(f"missing {label}: {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        _require(bool(line.strip()), f"blank row in {label}: {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PublicationError(
                f"invalid JSON in {label}:{line_number}: {exc}"
            ) from exc
        _require(isinstance(value, dict), f"non-object row in {label}:{line_number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _contains_secret(value: Any, *, location: str = "root") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in _CREDENTIAL_METADATA_KEYS:
                if not isinstance(item, str) or not re.fullmatch(
                    r"[A-Z][A-Z0-9_]{2,127}", item
                ):
                    return f"{location}.{key}"
                continue
            if normalized in _SECRET_KEYS:
                return f"{location}.{key}"
            found = _contains_secret(item, location=f"{location}.{key}")
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            found = _contains_secret(item, location=f"{location}[{index}]")
            if found:
                return found
    elif isinstance(value, str) and _SECRET_RE.search(value):
        return location
    return None


def _descriptor(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.relative_to(relative_to))
        if relative_to
        else str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".jsonl":
        record["rows"] = len(path.read_text(encoding="utf-8").splitlines())
    return record


def _resolve_descriptor(root: Path, value: Any, *, label: str) -> Path:
    record = _mapping(value, label=label)
    relative = Path(_string(record.get("path"), label=f"{label}.path"))
    _require(not relative.is_absolute(), f"{label}.path must be relative")
    candidates = [
        path
        for path in dict.fromkeys(
            ((root / relative).resolve(), (REPO_ROOT / relative).resolve())
        )
        if path.is_file()
    ]
    _require(len(candidates) == 1, f"{label}.path must resolve uniquely")
    path = candidates[0]
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise PublicationError(f"{label}.path escapes acquisition root") from exc
    _require(
        path.is_file() and not path.is_symlink(), f"missing or unsafe {label}: {path}"
    )
    _require(record.get("sha256") == sha256_file(path), f"{label} SHA-256 mismatch")
    _require(record.get("bytes") == path.stat().st_size, f"{label} byte count mismatch")
    if "rows" in record:
        rows = len(path.read_text(encoding="utf-8").splitlines())
        _require(record.get("rows") == rows, f"{label} row count mismatch")
    return path


def _extract_final_answer(response: str, *, label: str) -> str:
    _require(
        response.count(BOUNDARY) == 1, f"{label} must contain exactly one boundary"
    )
    answer = response.split(BOUNDARY, 1)[1]
    _require(
        bool(answer) and answer == answer.strip(),
        f"{label} final answer is empty or padded",
    )
    return answer


def _source_paths(source_root: Path, generation_manifest_root: Path) -> dict[str, Path]:
    manifest_root = generation_manifest_root
    if not (manifest_root / "train.jsonl").is_file():
        manifest_root = manifest_root / "manifests"
    result = {
        "candidate/audits/repair_manifest.jsonl": source_root
        / "audits/repair_manifest.jsonl",
    }
    for source_split in SOURCE_SPLITS.values():
        result[f"candidate/analysis_sft/{source_split}.jsonl"] = (
            source_root / f"analysis_sft/{source_split}.jsonl"
        )
        result[f"generation/manifests/{source_split}.jsonl"] = (
            manifest_root / f"{source_split}.jsonl"
        )
    return result


def _load_source_rows(
    source_root: Path,
    generation_manifest_root: Path,
    *,
    expected_split_counts: Mapping[str, int],
    expected_sample_id_sha256: str,
    expected_source_file_sha256: Mapping[str, str] | None,
) -> tuple[
    dict[str, list[SourceRow]],
    dict[str, dict[str, Any]],
    dict[str, int],
]:
    source_root = source_root.resolve()
    generation_manifest_root = generation_manifest_root.resolve()
    _require(
        source_root.is_dir() and not source_root.is_symlink(),
        f"invalid source root: {source_root}",
    )
    _require(
        generation_manifest_root.is_dir() and not generation_manifest_root.is_symlink(),
        f"invalid generation manifest root: {generation_manifest_root}",
    )
    paths = _source_paths(source_root, generation_manifest_root)
    for key, path in paths.items():
        _require(
            path.is_file() and not path.is_symlink(), f"missing source artifact: {key}"
        )
    if expected_source_file_sha256 is not None:
        _require(
            set(expected_source_file_sha256) == set(paths),
            "expected source file pin set mismatch",
        )
        for key, expected in expected_source_file_sha256.items():
            _require(
                sha256_file(paths[key]) == expected, f"source SHA-256 mismatch: {key}"
            )

    repairs = _read_jsonl(
        paths["candidate/audits/repair_manifest.jsonl"],
        label="chk1 repair manifest",
    )
    repairs_by_split: dict[str, dict[int, Mapping[str, Any]]] = {
        source_split: {} for source_split in SOURCE_SPLITS.values()
    }
    response_changed_rows = 0
    answer_changed_rows = 0
    for repair in repairs:
        source_split = repair.get("split")
        _require(
            source_split in repairs_by_split,
            f"repair manifest split drift: {source_split}",
        )
        line_number = repair.get("source_line_number")
        _require(
            isinstance(line_number, int) and line_number > 0,
            "invalid repair source_line_number",
        )
        _require(
            line_number not in repairs_by_split[source_split],
            "duplicate repair source line",
        )
        candidate_response_sha = _sha(
            repair.get("candidate_response_sha256"),
            label="repair candidate_response_sha256",
        )
        source_response_sha = _sha(
            repair.get("source_response_sha256"),
            label="repair source_response_sha256",
        )
        candidate_answer_sha = _sha(
            repair.get("candidate_answer_sha256"),
            label="repair candidate_answer_sha256",
        )
        source_answer_sha = _sha(
            repair.get("source_answer_sha256"),
            label="repair source_answer_sha256",
        )
        response_changed_rows += candidate_response_sha != source_response_sha
        answer_changed_rows += candidate_answer_sha != source_answer_sha
        repairs_by_split[source_split][line_number] = repair

    result: dict[str, list[SourceRow]] = {}
    all_ids: list[str] = []
    meeting_splits: dict[str, str] = {}
    for split in SPLITS:
        source_split = SOURCE_SPLITS[split]
        candidates = _read_jsonl(
            paths[f"candidate/analysis_sft/{source_split}.jsonl"],
            label=f"chk1 {source_split} candidate",
        )
        generations = _read_jsonl(
            paths[f"generation/manifests/{source_split}.jsonl"],
            label=f"chk1 {source_split} generation manifest",
        )
        expected = expected_split_counts.get(split)
        _require(
            isinstance(expected, int) and expected > 0 and len(candidates) == expected,
            f"source split count mismatch: {split}",
        )
        split_repairs = repairs_by_split[source_split]
        _require(
            len(split_repairs) == expected,
            f"repair population mismatch: {source_split}",
        )
        rows: list[SourceRow] = []
        # The acquisition generator carries the candidate JSONL line number as
        # ``source_index``.  JSONL line numbers (and the repair sidecar's
        # ``source_line_number``) are one-based, so reconstruct the same
        # identity here instead of silently shifting every row by one.
        for source_index, candidate in enumerate(candidates, start=1):
            label = f"source {split}:{source_index}"
            _require(
                set(candidate) == {"prompt", "provided_data", "response"},
                f"{label} schema drift",
            )
            line_number = source_index
            repair = _mapping(split_repairs.get(line_number), label=f"{label} repair")
            prompt = _string(candidate.get("prompt"), label=f"{label}.prompt")
            provided = _string(
                candidate.get("provided_data"), label=f"{label}.provided_data"
            )
            response = _string(candidate.get("response"), label=f"{label}.response")
            answer = _extract_final_answer(response, label=f"{label}.response")
            _require(
                repair.get("split") == source_split, f"{label} repair split mismatch"
            )
            _require(
                repair.get("prompt_sha256") == sha256_text(prompt),
                f"{label} prompt hash mismatch",
            )
            _require(
                repair.get("provided_data_sha256") == sha256_text(provided),
                f"{label} provided_data hash mismatch",
            )
            _require(
                repair.get("candidate_response_sha256") == sha256_text(response),
                f"{label} response hash mismatch",
            )
            _require(
                repair.get("candidate_answer_sha256") == sha256_text(answer),
                f"{label} final-answer hash mismatch",
            )
            generation_line = repair.get("generation_manifest_line_number")
            _require(
                isinstance(generation_line, int)
                and 1 <= generation_line <= len(generations),
                f"{label} generation line invalid",
            )
            generation = generations[generation_line - 1]
            sample_id = _string(repair.get("sample_id"), label=f"{label}.sample_id")
            _require(
                generation.get("sample_id") == sample_id,
                f"{label} generation identity mismatch",
            )
            _require(
                generation.get("split") == source_split,
                f"{label} generation split mismatch",
            )
            _require(
                generation.get("prompt_sha256") == sha256_text(prompt),
                f"{label} generation prompt mismatch",
            )
            _require(
                generation.get("provided_data_sha256") == sha256_text(provided),
                f"{label} generation provided_data mismatch",
            )
            _require(
                generation.get("final_analysis_sha256")
                == repair.get("source_answer_sha256"),
                f"{label} generation/source answer hash mismatch",
            )
            meeting_date = _string(
                generation.get("meeting_date"), label=f"{label}.meeting_date"
            )
            previous = meeting_splits.setdefault(meeting_date, split)
            _require(
                previous == split, f"meeting crosses source splits: {meeting_date}"
            )
            row = SourceRow(
                sample_id=sample_id,
                split=split,
                source_split=source_split,
                source_index=source_index,
                meeting_date=meeting_date,
                atomic_topic=_string(
                    generation.get("atomic_topic"), label=f"{label}.atomic_topic"
                ),
                section_style_id=_string(
                    generation.get("section_style_id"),
                    label=f"{label}.section_style_id",
                ),
                provided_data=provided,
                source_analysis=answer,
                original_prompt_sha256=sha256_text(prompt),
                candidate_response_sha256=sha256_text(response),
            )
            rows.append(row)
            all_ids.append(sample_id)
        result[split] = rows
    _require(len(all_ids) == len(set(all_ids)), "source sample IDs are not unique")
    _require(
        _ordered_source_id_digest(all_ids) == expected_sample_id_sha256,
        "source sample-ID digest mismatch",
    )
    bindings = {key: _descriptor(path) for key, path in paths.items()}
    repair_provenance = {
        "candidate_response_changed_rows": response_changed_rows,
        "candidate_final_answer_changed_rows": answer_changed_rows,
        "source_rows": len(all_ids),
    }
    return result, bindings, repair_provenance


def _collect_descriptor_shas(value: Any) -> set[str]:
    shas: set[str] = set()
    if isinstance(value, Mapping):
        if isinstance(value.get("sha256"), str):
            shas.add(str(value["sha256"]))
        for item in value.values():
            shas.update(_collect_descriptor_shas(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            shas.update(_collect_descriptor_shas(item))
    return shas


def _validate_required_lineage(value: Any, *, label: str) -> dict[str, Any]:
    lineage = dict(_mapping(value, label=label))
    for key, expected in REQUIRED_LINEAGE.items():
        _require(lineage.get(key) == expected, f"{label}.{key} drift")
    return lineage


def _validate_prompt_contract(
    contract: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    _require(
        contract.get("schema_version") == generator.PROMPT_CONTRACT_SCHEMA_VERSION,
        "prompt contract is not the official-reference v2 schema",
    )
    implementation = generator._implementation_contract()
    _require(
        contract.get("implementation") == implementation
        and contract.get("code_sha256") == implementation["composite_sha256"],
        "prompt contract implementation digest drift",
    )
    expected_contract = generator._prompt_contract(
        code_sha256=str(implementation["composite_sha256"]),
        config=generator.ProviderConfig(),
    )
    student = _mapping(contract.get("student"), label="prompt_contract.student")
    system_prompt = _string(
        student.get("system_prompt"), label="prompt_contract.student.system_prompt"
    )
    _require(
        student.get("system_prompt_sha256") == sha256_text(system_prompt),
        "student system prompt hash mismatch",
    )
    _require(
        student.get("user_prompt_template") == generator.STUDENT_USER_PROMPT_TEMPLATE,
        "student user-prompt template drift",
    )
    _require(
        student.get("response_contract")
        == "native_reasoning + '\\n</think>\\n' + rewritten_minutes"
        and student.get("opening_think_supplied_by_chat_template") is True
        and student.get("answer_tags_forbidden") is True
        and student.get("max_total_tokens") == MAX_TOTAL_TOKENS,
        "student response/token contract drift",
    )
    _require(
        student.get("tokenizer_path")
        == generator._display_path(generator.DEFAULT_TOKENIZER_PATH)
        and student.get("tokenizer_file_sha256")
        == dict(generator.EXPECTED_TOKENIZER_FILE_SHA256)
        and student.get("tokenizer_runtime_contract")
        == generator._expected_tokenizer_runtime_contract(),
        "student tokenizer provenance drift",
    )
    expected_roles = {
        generator.ROLE_SOURCE_AUDIT_PRIMARY: generator.SOURCE_AUDIT_SYSTEM_PROMPT,
        generator.ROLE_SOURCE_AUDIT_ADJUDICATION: (
            generator.SOURCE_ADJUDICATION_SYSTEM_PROMPT
        ),
        generator.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR: (
            generator.SOURCE_CONTRACT_REPAIR_SYSTEM_PROMPT
        ),
        generator.ROLE_REWRITE_PRIMARY: generator.REWRITE_SYSTEM_PROMPT,
        generator.ROLE_REWRITE_FIDELITY_REPAIR: (
            generator.FIDELITY_REPAIR_SYSTEM_PROMPT
        ),
        generator.ROLE_REWRITE_STYLE_REPAIR: generator.STYLE_REPAIR_SYSTEM_PROMPT,
        generator.ROLE_VALIDATOR_A_PRIMARY: generator.VALIDATOR_A_SYSTEM_PROMPT,
        generator.ROLE_VALIDATOR_A_FIDELITY_REPAIR: (
            generator.VALIDATOR_A_SYSTEM_PROMPT
        ),
        generator.ROLE_VALIDATOR_A_STYLE_REPAIR: generator.VALIDATOR_A_SYSTEM_PROMPT,
        generator.ROLE_VALIDATOR_B_PRIMARY: generator.VALIDATOR_B_SYSTEM_PROMPT,
        generator.ROLE_VALIDATOR_B_STYLE_REPAIR: generator.VALIDATOR_B_SYSTEM_PROMPT,
        generator.ROLE_VALIDATOR_A_PRIMARY_CONTRACT_REPAIR: (
            generator.VALIDATOR_A_CONTRACT_REPAIR_SYSTEM_PROMPT
        ),
        generator.ROLE_VALIDATOR_A_FIDELITY_REPAIR_CONTRACT_REPAIR: (
            generator.VALIDATOR_A_CONTRACT_REPAIR_SYSTEM_PROMPT
        ),
        generator.ROLE_VALIDATOR_A_STYLE_REPAIR_CONTRACT_REPAIR: (
            generator.VALIDATOR_A_CONTRACT_REPAIR_SYSTEM_PROMPT
        ),
        generator.ROLE_VALIDATOR_B_PRIMARY_CONTRACT_REPAIR: (
            generator.VALIDATOR_B_CONTRACT_REPAIR_SYSTEM_PROMPT
        ),
        generator.ROLE_VALIDATOR_B_STYLE_REPAIR_CONTRACT_REPAIR: (
            generator.VALIDATOR_B_CONTRACT_REPAIR_SYSTEM_PROMPT
        ),
    }
    _require(
        contract.get("roles") == expected_roles,
        "prompt contract role prompt map drift",
    )
    gate = _mapping(
        contract.get("validator_b_gate"), label="prompt_contract.validator_b_gate"
    )
    _require(
        gate.get("reference_scope") == "corresponding_meeting_pre_action_analysis_body"
        and gate.get("comparison_statuses") == ["comparable", "no_comparable_passage"]
        and gate.get("passage_match_types") == sorted(generator.PASSAGE_MATCH_TYPES)
        and gate.get("candidate_sentence_coverage_required") is True
        and gate.get("dimension_evidence_must_use_passage_match_paragraphs") is True
        and gate.get("dimensions") == list(STYLE_DIMENSIONS)
        and gate.get("mean_minimum") == STYLE_MEAN_THRESHOLD
        and gate.get("dimension_minimum") == STYLE_MIN_THRESHOLD
        and gate.get("critical_errors_allowed") == 0
        and gate.get("verbatim_reference_copy_is_critical") is True
        and gate.get("no_comparable_passage_is_reject_without_repair") is True
        and gate.get("style_repair_eligibility") == "comparable_low_score_only"
        and gate.get("critical_errors_reject_without_repair") is True,
        "validator B gate contract drift",
    )
    _require(
        contract.get("source_audit_contract_repair")
        == expected_contract["source_audit_contract_repair"],
        "source-audit contract-repair policy drift",
    )
    _require(
        contract.get("validator_contract_repair")
        == expected_contract["validator_contract_repair"],
        "validator contract-repair policy drift",
    )
    _require(
        contract.get("bulk_preflight_gate")
        == {
            "default_rows": generator.DEFAULT_PREFLIGHT_ROWS,
            "applies_before_bulk_phases": [
                "source-audit",
                "generate",
                "verify",
                "all",
            ],
            "selected_rows_must_reach_terminal_pass": True,
            "quality_rejects_are_terminal": True,
            "quality_rejects_trigger_same_split_top_up": True,
            "candidate_order": "fixed_source_order_with_fixed_split_quotas",
            "required_chain": [
                "source_audit",
                "rewrite_teacher",
                "validator_a",
                "validator_b",
                "tokenizer_replay",
            ],
            "tokenizer_invariants": [
                "single_bos",
                "single_eos",
                "completion_only_prompt_masked",
                "completion_mask_covers_reasoning_boundary_answer_eos",
                "no_truncation",
            ],
            "bulk_blocked_on_unfilled_terminal_pass_quota": True,
            "bulk_blocked_on_unresolved_failure": True,
        },
        "bulk preflight gate contract drift",
    )
    provider = _mapping(contract.get("provider"), label="prompt_contract.provider")
    _require(
        provider.get("model") == "deepseek-v4-flash"
        and provider.get("api_key_env") == "DEEPSEEK_API_KEY"
        and provider.get("fallback") == "forbidden"
        and provider.get("temperature") is None
        and provider.get("top_p") is None,
        "provider contract drift",
    )
    lineage = _validate_required_lineage(
        contract.get("lineage"), label="prompt_contract.lineage"
    )
    return system_prompt, lineage


def _validate_exact_tokenizer_files(
    tokenizer_path: Path, *, expected_tokenizer_path: Path
) -> None:
    if expected_tokenizer_path.resolve() != DEFAULT_TOKENIZER_PATH.resolve():
        return
    for name, expected_sha in generator.EXPECTED_TOKENIZER_FILE_SHA256.items():
        artifact = tokenizer_path.resolve() / name
        _require(
            artifact.is_file() and not artifact.is_symlink(),
            f"exact chk1 cp200 tokenizer artifact is missing or unsafe: {name}",
        )
        _require(
            sha256_file(artifact) == expected_sha,
            f"exact chk1 cp200 tokenizer artifact SHA-256 drift: {name}",
        )


def _load_official_reference_bank(
    path: Path,
    source_rows: Sequence[SourceRow],
    *,
    roster_path: Path,
    expected_roster_sha256: str,
    expected_meeting_counts: Mapping[str, int],
    expected_boundary_method_counts: Mapping[str, int],
    exception_boundaries: Mapping[str, Mapping[str, Any]],
) -> OfficialReferenceBank:
    """Load the acquisition bank, then rebuild it from sealed raw sources."""

    _require(
        path.is_file() and not path.is_symlink(), "official reference bank missing"
    )
    try:
        acquisition_bytes = path.read_bytes()
        bank = deserialize_official_reference_bank(acquisition_bytes)
    except (OSError, OfficialReferenceError) as exc:
        raise PublicationError(f"invalid official reference bank: {exc}") from exc
    _require(
        _contains_secret(bank.to_dict()) is None,
        "official reference bank contains credential material",
    )
    source_projection = [
        {"meeting_date": row.meeting_date, "split": row.split} for row in source_rows
    ]
    try:
        verify_official_reference_bank(
            bank,
            source_projection,
            roster_path=roster_path,
            expected_roster_sha256=expected_roster_sha256,
            expected_meeting_counts=expected_meeting_counts,
            exception_boundaries=exception_boundaries,
            expected_boundary_method_counts=expected_boundary_method_counts,
        )
        rebuilt = build_official_reference_bank(
            source_projection,
            roster_path=roster_path,
            expected_roster_sha256=expected_roster_sha256,
            expected_meeting_counts=expected_meeting_counts,
            exception_boundaries=exception_boundaries,
            expected_boundary_method_counts=expected_boundary_method_counts,
        )
    except OfficialReferenceError as exc:
        raise PublicationError(
            f"official reference provenance does not rebuild exactly: {exc}"
        ) from exc
    _require(
        serialize_official_reference_bank(rebuilt) == acquisition_bytes,
        "official reference bank bytes differ from sealed roster/raw CSV rebuild",
    )
    _require(
        bank.reference_bank_sha256 == rebuilt.reference_bank_sha256,
        "official reference bank embedded digest differs from rebuild",
    )
    return bank


def _new_provider_audit() -> dict[str, Any]:
    return {
        "identity": None,
        "roles": set(),
        "source_roles": set(),
        "receipt_keys": set(),
        "cache_raw_content": {},
        "cache_raw_reasoning": {},
        "cache_records": {},
        "current_sample_binding": None,
        "official_reference_bank_sha256": None,
    }


def _provider_complete(
    value: Any,
    *,
    label: str,
    provider_audit: dict[str, Any],
    expected_role: str | None = None,
    expected_field_hashes: Mapping[str, Mapping[str, str]] | None = None,
) -> None:
    """Validate one provider receipt or a named collection of receipts."""

    provider = _mapping(value, label=label)
    if "returned_model" not in provider:
        _require(bool(provider), f"{label} provider receipt collection is empty")
        for role, nested in provider.items():
            _require(
                role in generator.PROVIDER_ROLES,
                f"{label} contains an unknown provider role: {role}",
            )
            _provider_complete(
                nested,
                label=f"{label}.{role}",
                provider_audit=provider_audit,
                expected_role=str(role),
                expected_field_hashes=expected_field_hashes,
            )
        return
    _require(expected_role is not None, f"{label} provider role is not explicit")
    _require(
        provider.get("returned_model") == "deepseek-v4-flash",
        f"{label} model drift",
    )
    fingerprint = _string(
        provider.get("system_fingerprint"),
        label=f"{label}.system_fingerprint",
    )
    response_id = _string(provider.get("response_id"), label=f"{label}.response_id")
    _require(provider.get("finish_reason") == "stop", f"{label} did not stop normally")
    projection = _mapping(
        provider.get("request_projection"), label=f"{label}.request_projection"
    )
    expected_fields = list(generator._role_fields(expected_role))
    _require(
        set(projection)
        == {
            "role",
            "allowed_fields",
            "field_value_sha256",
            "canonical_payload_sha256",
            "official_text_policy",
            "api_key_env",
            "plaintext_credential_persisted",
        },
        f"{label} request projection schema drift",
    )
    _require(
        projection.get("role") == expected_role
        and projection.get("allowed_fields") == expected_fields,
        f"{label} request projection role/field drift",
    )
    field_hashes = _mapping(
        projection.get("field_value_sha256"),
        label=f"{label}.request_projection.field_value_sha256",
    )
    _require(
        set(field_hashes) == set(expected_fields)
        and all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in field_hashes.values()
        ),
        f"{label} request projection field hashes drift",
    )
    expected_subset = (expected_field_hashes or {}).get(expected_role, {})
    _require(
        all(
            field_hashes.get(field) == digest
            for field, digest in expected_subset.items()
        ),
        f"{label} request projection value binding drift",
    )
    _sha(
        projection.get("canonical_payload_sha256"),
        label=f"{label}.request_projection.canonical_payload_sha256",
    )
    official_policy = (
        "corresponding_meeting_pre_action_analysis_body"
        if expected_role.startswith("validator_b")
        else "forbidden"
    )
    _require(
        projection.get("official_text_policy") == official_policy
        and projection.get("api_key_env") == "DEEPSEEK_API_KEY"
        and projection.get("plaintext_credential_persisted") is False,
        f"{label} request projection policy drift",
    )
    if not isinstance(provider_audit.get("current_sample_binding"), Mapping):
        identity = (str(provider["returned_model"]), fingerprint)
        if provider_audit["identity"] is None:
            provider_audit["identity"] = identity
        _require(
            provider_audit["identity"] == identity,
            f"{label} provider model/fingerprint identity drift",
        )
        provider_audit["roles"].add(expected_role)
        if expected_role.startswith("source_audit"):
            provider_audit["source_roles"].add(expected_role)
        provider_audit["receipt_keys"].add((expected_role, response_id))
        return
    cache_records = _mapping(
        provider_audit.get("cache_records"), label=f"{label}.cache_records"
    )
    cache_record = _mapping(
        cache_records.get((expected_role, response_id)),
        label=f"{label}.provider_cache",
    )
    cache_projection = _mapping(
        cache_record.get("request_projection"),
        label=f"{label}.provider_cache.request_projection",
    )
    cache_binding = _mapping(
        cache_record.get("binding"), label=f"{label}.provider_cache.binding"
    )
    cached_response = _mapping(
        cache_record.get("provider_response"),
        label=f"{label}.provider_cache.provider_response",
    )
    _require(
        cache_record.get("binding_sha256") == sha256_text(canonical_json(cache_binding))
        and cache_projection == projection
        and cache_binding.get("request_projection_sha256")
        == sha256_text(canonical_json(projection)),
        f"{label} provider cache binding/projection drift",
    )
    sample_binding = _mapping(
        provider_audit.get("current_sample_binding"),
        label=f"{label}.current_sample_binding",
    )
    expected_system_prompts = {
        generator.ROLE_SOURCE_AUDIT_PRIMARY: generator.SOURCE_AUDIT_SYSTEM_PROMPT,
        generator.ROLE_SOURCE_AUDIT_ADJUDICATION: (
            generator.SOURCE_ADJUDICATION_SYSTEM_PROMPT
        ),
        generator.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR: (
            generator.SOURCE_CONTRACT_REPAIR_SYSTEM_PROMPT
        ),
        generator.ROLE_REWRITE_PRIMARY: generator.REWRITE_SYSTEM_PROMPT,
        generator.ROLE_REWRITE_FIDELITY_REPAIR: (
            generator.FIDELITY_REPAIR_SYSTEM_PROMPT
        ),
        generator.ROLE_REWRITE_STYLE_REPAIR: generator.STYLE_REPAIR_SYSTEM_PROMPT,
        generator.ROLE_VALIDATOR_A_PRIMARY: generator.VALIDATOR_A_SYSTEM_PROMPT,
        generator.ROLE_VALIDATOR_A_FIDELITY_REPAIR: (
            generator.VALIDATOR_A_SYSTEM_PROMPT
        ),
        generator.ROLE_VALIDATOR_A_STYLE_REPAIR: generator.VALIDATOR_A_SYSTEM_PROMPT,
        generator.ROLE_VALIDATOR_B_PRIMARY: generator.VALIDATOR_B_SYSTEM_PROMPT,
        generator.ROLE_VALIDATOR_B_STYLE_REPAIR: generator.VALIDATOR_B_SYSTEM_PROMPT,
        generator.ROLE_VALIDATOR_A_PRIMARY_CONTRACT_REPAIR: (
            generator.VALIDATOR_A_CONTRACT_REPAIR_SYSTEM_PROMPT
        ),
        generator.ROLE_VALIDATOR_A_FIDELITY_REPAIR_CONTRACT_REPAIR: (
            generator.VALIDATOR_A_CONTRACT_REPAIR_SYSTEM_PROMPT
        ),
        generator.ROLE_VALIDATOR_A_STYLE_REPAIR_CONTRACT_REPAIR: (
            generator.VALIDATOR_A_CONTRACT_REPAIR_SYSTEM_PROMPT
        ),
        generator.ROLE_VALIDATOR_B_PRIMARY_CONTRACT_REPAIR: (
            generator.VALIDATOR_B_CONTRACT_REPAIR_SYSTEM_PROMPT
        ),
        generator.ROLE_VALIDATOR_B_STYLE_REPAIR_CONTRACT_REPAIR: (
            generator.VALIDATOR_B_CONTRACT_REPAIR_SYSTEM_PROMPT
        ),
    }
    _require(
        cache_binding.get("schema_version") == generator.CACHE_SCHEMA_VERSION
        and cache_binding.get("role") == expected_role
        and cache_binding.get("sample_id") == sample_binding.get("sample_id")
        and cache_binding.get("split") == sample_binding.get("split")
        and cache_binding.get("source_analysis_sha256")
        == sample_binding.get("source_analysis_sha256")
        and cache_binding.get("provided_data_sha256")
        == sample_binding.get("provided_data_sha256")
        and cache_binding.get("official_reference_bank_sha256")
        == (
            provider_audit.get("official_reference_bank_sha256")
            if expected_role.startswith("validator_b")
            else None
        )
        and cache_binding.get("system_prompt_sha256")
        == sha256_text(expected_system_prompts[expected_role])
        and bool(
            re.fullmatch(
                r"[0-9a-f]{64}", str(cache_binding.get("user_prompt_sha256") or "")
            )
        )
        and cache_binding.get("provider_contract_sha256")
        == sample_binding.get("provider_contract_sha256")
        and cache_binding.get("code_sha256") == sample_binding.get("code_sha256"),
        f"{label} provider cache sample/request binding drift",
    )
    replayed_provider = generator._provider_record(
        generator.ProviderResponse.from_dict(dict(cached_response)), cache_record
    )
    _require(
        dict(provider) == replayed_provider,
        f"{label} terminal provider receipt differs from provider cache",
    )
    identity = (str(provider["returned_model"]), fingerprint)
    if provider_audit["identity"] is None:
        provider_audit["identity"] = identity
    _require(
        provider_audit["identity"] == identity,
        f"{label} provider model/fingerprint identity drift",
    )
    provider_audit["roles"].add(expected_role)
    if expected_role.startswith("source_audit"):
        provider_audit["source_roles"].add(expected_role)
    provider_audit["receipt_keys"].add((expected_role, response_id))


def _provider_identity_projection(
    provider_audit: Mapping[str, Any], *, roles: set[str]
) -> dict[str, Any]:
    identity = provider_audit.get("identity")
    _require(
        isinstance(identity, tuple) and len(identity) == 2,
        "no provider identity was observed",
    )
    return {
        "returned_model": identity[0],
        "system_fingerprint": identity[1],
        "roles_observed": sorted(roles),
    }


def _cached_raw_content(
    provider_audit: Mapping[str, Any],
    *,
    role: str,
    provider: Mapping[str, Any],
    label: str,
) -> str:
    cache = _mapping(
        provider_audit.get("cache_raw_content"), label=f"{label}.cache_raw_content"
    )
    response_id = _string(provider.get("response_id"), label=f"{label}.response_id")
    raw = cache.get((role, response_id))
    _require(isinstance(raw, str) and bool(raw), f"{label} raw provider cache missing")
    _require(
        provider.get("raw_content_sha256") == sha256_text(raw),
        f"{label} raw provider cache SHA drift",
    )
    return raw


def _cached_raw_reasoning(
    provider_audit: Mapping[str, Any],
    *,
    role: str,
    provider: Mapping[str, Any],
    label: str,
) -> str:
    cache = _mapping(
        provider_audit.get("cache_raw_reasoning"),
        label=f"{label}.cache_raw_reasoning",
    )
    response_id = _string(provider.get("response_id"), label=f"{label}.response_id")
    raw = cache.get((role, response_id))
    _require(isinstance(raw, str), f"{label} raw provider reasoning cache missing")
    _require(
        provider.get("raw_reasoning_sha256") == sha256_text(raw),
        f"{label} raw provider reasoning cache SHA drift",
    )
    return raw


def _validate_source_claim_report(
    value: Any,
    *,
    label: str,
    source_analysis: str,
    provided_data: str,
    claims_key: str,
    issues_key: str,
) -> bool:
    report = _mapping(value, label=label)
    _require(
        set(report) == {claims_key, issues_key, "overall_pass", "machine_pass"},
        f"{label} field set mismatch",
    )
    claims = report.get(claims_key)
    issues = report.get(issues_key)
    _require(isinstance(claims, list) and bool(claims), f"{label} claims are empty")
    _require(isinstance(issues, list), f"{label} issues must be a list")
    _require(
        isinstance(report.get("overall_pass"), bool), f"{label} overall_pass invalid"
    )
    source_spans: list[str] = []
    failed_claims: Counter[tuple[str, str | None, str]] = Counter()
    semantic_pass = True
    for index, raw in enumerate(claims):
        claim = _mapping(raw, label=f"{label}.{claims_key}[{index}]")
        _require(
            set(claim) == {"analysis_span", "evidence_span", "verdict", "issue_code"},
            f"{label} claim schema drift",
        )
        analysis_span = _string(
            claim.get("analysis_span"), label=f"{label}.analysis_span[{index}]"
        )
        _require(
            analysis_span in source_analysis,
            f"{label} analysis span is not exact: {index}",
        )
        source_spans.append(analysis_span)
        verdict = claim.get("verdict")
        _require(
            verdict in {"supported", "unsupported", "contradicted"},
            f"{label} claim verdict drift: {index}",
        )
        evidence = claim.get("evidence_span")
        issue_code = claim.get("issue_code")
        if verdict == "supported":
            evidence_text = _string(evidence, label=f"{label}.evidence_span[{index}]")
            _require(
                evidence_text in provided_data,
                f"{label} evidence span is not exact: {index}",
            )
            _require(issue_code is None, f"{label} supported claim has issue code")
        else:
            semantic_pass = False
            _require(
                issue_code in generator._SOURCE_ISSUES,
                f"{label} issue code drift: {index}",
            )
            if verdict == "unsupported":
                _require(evidence is None, f"{label} unsupported claim has evidence")
            else:
                evidence_text = _string(
                    evidence, label=f"{label}.evidence_span[{index}]"
                )
                _require(
                    evidence_text in provided_data,
                    f"{label} contradicted evidence is not exact: {index}",
                )
            failed_claims[(analysis_span, evidence, str(issue_code))] += 1
    for index, sentence in enumerate(_sentences(source_analysis)):
        _require(
            any(sentence in span for span in source_spans),
            f"{label} source sentence is uncovered: {index}",
        )
    reported_issues: Counter[tuple[str, str | None, str]] = Counter()
    for index, raw in enumerate(issues):
        issue = _mapping(raw, label=f"{label}.{issues_key}[{index}]")
        _require(
            set(issue) == {"analysis_span", "evidence_span", "issue_code"},
            f"{label} blocking issue schema drift",
        )
        span = _string(
            issue.get("analysis_span"), label=f"{label}.issue.analysis_span[{index}]"
        )
        _require(
            span in source_analysis, f"{label} blocking span is not exact: {index}"
        )
        evidence = issue.get("evidence_span")
        if evidence is not None:
            evidence_text = _string(
                evidence, label=f"{label}.issue.evidence_span[{index}]"
            )
            _require(
                evidence_text in provided_data,
                f"{label} blocking evidence is not exact: {index}",
            )
        code = issue.get("issue_code")
        _require(code in generator._SOURCE_ISSUES, f"{label} blocking code drift")
        reported_issues[(span, evidence, str(code))] += 1
    _require(
        failed_claims == reported_issues,
        f"{label} claim/issue partition mismatch",
    )
    local_pass = semantic_pass and not issues
    _require(
        report.get("machine_pass") is local_pass,
        f"{label} machine_pass differs from local source gate",
    )
    return local_pass


def _validate_source_audit_record(
    value: Any,
    *,
    label: str,
    source_analysis: str,
    provided_data: str,
    provider_audit: dict[str, Any],
) -> bool:
    record = _mapping(value, label=label)
    _require(set(record) == SOURCE_AUDIT_FIELDS, f"{label} field set mismatch")
    _require(record.get("complete") is True, f"{label} is unresolved")
    result = _mapping(record.get("result"), label=f"{label}.result")
    _require(
        set(result) == {"primary", "adjudication", "contract_repair"},
        f"{label} result drift",
    )
    contract_repair = result.get("contract_repair")
    if (
        isinstance(contract_repair, Mapping)
        and contract_repair.get("contract_exhausted") is True
    ):
        provider = _mapping(record.get("provider"), label=f"{label}.provider")
        repair = _mapping(contract_repair, label=f"{label}.contract_repair")
        exhaustion = _mapping(
            repair.get("contract_exhaustion"),
            label=f"{label}.contract_exhaustion",
        )
        try:
            generator._validate_contract_exhaustion_receipt(
                exhaustion,
                stage="source_audit",
                providers=provider,
                controlled_codes=generator._controlled_source_contract_codes,
                sample_id=label,
            )
        except (generator.SyntheticRewriteError, generator.ContractError) as exc:
            raise PublicationError(str(exc)) from exc
        target = repair.get("target_report_type")
        _require(target in {"primary", "adjudication"}, f"{label} repair target drift")
        trigger_codes = _string_list(
            repair.get("trigger_contract_error_codes"),
            label=f"{label}.trigger_contract_error_codes",
            allow_empty=False,
        )
        invalid_role = (
            generator.ROLE_SOURCE_AUDIT_PRIMARY
            if target == "primary"
            else generator.ROLE_SOURCE_AUDIT_ADJUDICATION
        )
        cache_raw = _mapping(
            provider_audit.get("cache_raw_content"),
            label=f"{label}.provider_cache_raw_content",
        )
        invalid_provider = _mapping(
            provider.get(invalid_role), label=f"{label}.{invalid_role}.provider"
        )
        repair_provider = _mapping(
            provider.get(generator.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR),
            label=f"{label}.contract_repair.provider",
        )
        invalid_raw = cache_raw.get(
            (invalid_role, str(invalid_provider.get("response_id")))
        )
        replacement_raw = cache_raw.get(
            (
                generator.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR,
                str(repair_provider.get("response_id")),
            )
        )
        exhausted_role = str(exhaustion.get("exhausted_role"))
        exhausted_provider = _mapping(
            provider.get(exhausted_role),
            label=f"{label}.{exhausted_role}.provider",
        )
        exhausted_raw = cache_raw.get(
            (exhausted_role, str(exhausted_provider.get("response_id")))
        )
        _require(
            isinstance(invalid_raw, str)
            and isinstance(replacement_raw, str)
            and isinstance(exhausted_raw, str)
            and sha256_text(invalid_raw) == repair.get("invalid_report_sha256")
            and sha256_text(replacement_raw) == repair.get("replacement_report_sha256")
            and sha256_text(exhausted_raw) == exhaustion.get("exhausted_report_sha256"),
            f"{label} exhausted source raw SHA drift",
        )
        expected_hashes = {
            generator.ROLE_SOURCE_AUDIT_PRIMARY: {
                "source_analysis": _projection_hash(source_analysis),
                "provided_data": _projection_hash(provided_data),
            }
        }
        if generator.ROLE_SOURCE_AUDIT_ADJUDICATION in provider:
            expected_hashes[generator.ROLE_SOURCE_AUDIT_ADJUDICATION] = {
                "source_analysis": _projection_hash(source_analysis),
                "provided_data": _projection_hash(provided_data),
                "primary_findings": _projection_hash(result.get("primary")),
            }
        expected_hashes[generator.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR] = {
            "source_analysis": _projection_hash(source_analysis),
            "provided_data": _projection_hash(provided_data),
            "target_report_type": _projection_hash(target),
            "invalid_report": _projection_hash(invalid_raw),
            "contract_error_codes": _projection_hash(trigger_codes),
        }
        _provider_complete(
            provider,
            label=f"{label}.provider",
            provider_audit=provider_audit,
            expected_field_hashes=expected_hashes,
        )
        controlled = _string_list(
            exhaustion.get("controlled_error_codes"),
            label=f"{label}.controlled_error_codes",
            allow_empty=False,
        )
        _require(
            record.get("contract_repair_used") is True
            and record.get("machine_pass") is False
            and record.get("reasons") == controlled,
            f"{label} exhausted source verdict drift",
        )
        return False
    primary_pass = _validate_source_claim_report(
        result.get("primary"),
        label=f"{label}.primary",
        source_analysis=source_analysis,
        provided_data=provided_data,
        claims_key="claims",
        issues_key="blocking_issues",
    )
    adjudication = result.get("adjudication")
    if primary_pass:
        _require(adjudication is None, f"{label} has unnecessary adjudication")
        local_pass = True
        roles = {generator.ROLE_SOURCE_AUDIT_PRIMARY}
    else:
        _require(adjudication is not None, f"{label} lacks required adjudication")
        local_pass = _validate_source_claim_report(
            adjudication,
            label=f"{label}.adjudication",
            source_analysis=source_analysis,
            provided_data=provided_data,
            claims_key="reviewed_claims",
            issues_key="confirmed_blocking_issues",
        )
        roles = {
            generator.ROLE_SOURCE_AUDIT_PRIMARY,
            generator.ROLE_SOURCE_AUDIT_ADJUDICATION,
        }
    contract_repair_used = record.get("contract_repair_used")
    contract_repair = result.get("contract_repair")
    _require(
        isinstance(contract_repair_used, bool)
        and contract_repair_used is (contract_repair is not None),
        f"{label} contract-repair flag drift",
    )
    provider = _mapping(record.get("provider"), label=f"{label}.provider")
    if contract_repair_used:
        repair = _mapping(contract_repair, label=f"{label}.contract_repair")
        _require(
            set(repair)
            == {
                "target_report_type",
                "trigger_contract_error_codes",
                "invalid_report_sha256",
                "replacement_report_sha256",
                "contract_exhausted",
                "contract_exhaustion",
            },
            f"{label} contract-repair receipt schema drift",
        )
        target = repair.get("target_report_type")
        _require(
            repair.get("contract_exhausted") is False
            and repair.get("contract_exhaustion") is None,
            f"{label} unexpected source contract exhaustion",
        )
        _require(target in {"primary", "adjudication"}, f"{label} repair target drift")
        trigger_codes = _string_list(
            repair.get("trigger_contract_error_codes"),
            label=f"{label}.trigger_contract_error_codes",
            allow_empty=False,
        )
        _require(
            all(
                code in generator._SOURCE_CONTRACT_CODE_PREFIXES
                for code in trigger_codes
            )
            and len(trigger_codes) == len(set(trigger_codes)),
            f"{label} contains uncontrolled source contract-repair codes",
        )
        invalid_sha = _sha(
            repair.get("invalid_report_sha256"),
            label=f"{label}.invalid_report_sha256",
        )
        replacement_sha = _sha(
            repair.get("replacement_report_sha256"),
            label=f"{label}.replacement_report_sha256",
        )
        roles.add(generator.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR)
    else:
        _require(contract_repair is None, f"{label} unexpected contract repair")
    _require(set(provider) == roles, f"{label} provider role sequence drift")
    expected_hashes = {
        generator.ROLE_SOURCE_AUDIT_PRIMARY: {
            "source_analysis": _projection_hash(source_analysis),
            "provided_data": _projection_hash(provided_data),
        }
    }
    if adjudication is not None:
        expected_hashes[generator.ROLE_SOURCE_AUDIT_ADJUDICATION] = {
            "source_analysis": _projection_hash(source_analysis),
            "provided_data": _projection_hash(provided_data),
            "primary_findings": _projection_hash(result["primary"]),
        }
    if contract_repair_used:
        invalid_role = (
            generator.ROLE_SOURCE_AUDIT_PRIMARY
            if target == "primary"
            else generator.ROLE_SOURCE_AUDIT_ADJUDICATION
        )
        invalid_provider = _mapping(
            provider.get(invalid_role), label=f"{label}.{invalid_role}.provider"
        )
        replacement_provider = _mapping(
            provider.get(generator.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR),
            label=f"{label}.contract_repair.provider",
        )
        cache_raw = _mapping(
            provider_audit.get("cache_raw_content"),
            label=f"{label}.provider_cache_raw_content",
        )
        invalid_raw = cache_raw.get(
            (invalid_role, str(invalid_provider.get("response_id")))
        )
        replacement_raw = cache_raw.get(
            (
                generator.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR,
                str(replacement_provider.get("response_id")),
            )
        )
        _require(
            isinstance(invalid_raw, str)
            and isinstance(replacement_raw, str)
            and sha256_text(invalid_raw) == invalid_sha
            and sha256_text(replacement_raw) == replacement_sha
            and invalid_provider.get("raw_content_sha256") == invalid_sha
            and replacement_provider.get("raw_content_sha256") == replacement_sha,
            f"{label} raw contract-repair receipt SHA drift",
        )
        expected_hashes[generator.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR] = {
            "source_analysis": _projection_hash(source_analysis),
            "provided_data": _projection_hash(provided_data),
            "target_report_type": _projection_hash(target),
            "invalid_report": _projection_hash(invalid_raw),
            "contract_error_codes": _projection_hash(trigger_codes),
        }
    _provider_complete(
        provider,
        label=f"{label}.provider",
        provider_audit=provider_audit,
        expected_field_hashes=expected_hashes,
    )
    expected_reasons = [] if local_pass else ["source_claim_not_supported"]
    _require(
        record.get("machine_pass") is local_pass
        and record.get("reasons") == expected_reasons,
        f"{label} top-level local verdict/reasons drift",
    )
    return local_pass


def _exact_optional_span(
    value: Any, *, text: str, required: bool, label: str
) -> str | None:
    if value is None:
        _require(not required, f"{label} is required")
        return None
    span = _string(value, label=label)
    _require(span in text, f"{label} is not an exact span")
    return span


def _provider_response_for_replay(
    provider: Mapping[str, Any], *, raw_content: str, raw_reasoning: str
) -> generator.ProviderResponse:
    return generator.ProviderResponse(
        raw_reasoning=raw_reasoning,
        raw_content=raw_content,
        response_id=_string(provider.get("response_id"), label="provider.response_id"),
        returned_model=_string(
            provider.get("returned_model"), label="provider.returned_model"
        ),
        system_fingerprint=_string(
            provider.get("system_fingerprint"),
            label="provider.system_fingerprint",
        ),
        finish_reason=_string(
            provider.get("finish_reason"), label="provider.finish_reason"
        ),
        created=provider.get("created"),
        usage=dict(provider.get("usage") or {}),
    )


def _recompute_validator_a_raw(
    *,
    raw_content: str,
    provider: Mapping[str, Any],
    source_analysis: str,
    reasoning: str,
    answer: str,
) -> tuple[dict[str, Any] | None, bool, list[str], list[str]]:
    response = _provider_response_for_replay(
        provider, raw_content=raw_content, raw_reasoning=""
    )
    try:
        normalized, passed, reasons = generator._validate_validator_a(
            SimpleNamespace(source_analysis=source_analysis),
            SimpleNamespace(
                teacher_response_analysis=reasoning,
                rewritten_minutes=answer,
            ),
            response,
        )
    except generator.ContractError as exc:
        return None, False, list(exc.reasons), list(exc.reasons)
    contract_reasons = normalized.get("contract_reasons")
    _require(
        isinstance(contract_reasons, list),
        "Validator-A replay lacks contract reasons",
    )
    return normalized, passed, reasons, list(contract_reasons)


def _recompute_validator_b_raw(
    *,
    raw_content: str,
    provider: Mapping[str, Any],
    answer: str,
    reference: Any,
) -> tuple[dict[str, Any] | None, bool, list[str], list[str]]:
    meeting_date = _string(
        getattr(reference, "meeting_date", None), label="reference.meeting_date"
    )
    bank = {
        "meetings": {
            meeting_date: {
                "meeting_date": meeting_date,
                "paragraphs": [
                    {
                        "paragraph_id": paragraph.paragraph_id,
                        "section_name": paragraph.section_name,
                        "text": paragraph.text,
                    }
                    for paragraph in reference.paragraphs
                ],
            }
        }
    }
    response = _provider_response_for_replay(
        provider, raw_content=raw_content, raw_reasoning=""
    )
    try:
        normalized, passed, reasons = generator._validate_validator_b(
            SimpleNamespace(meeting_date=meeting_date),
            SimpleNamespace(rewritten_minutes=answer),
            response,
            bank,
        )
    except generator.ContractError as exc:
        return None, False, list(exc.reasons), list(exc.reasons)
    contract_reasons = normalized.get("contract_reasons")
    _require(
        isinstance(contract_reasons, list),
        "Validator-B replay lacks contract reasons",
    )
    return normalized, passed, reasons, list(contract_reasons)


def _validator_result_without_contract_receipt(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"contract_repair_used", "contract_repair"}
    }


def _validate_validator_contract_repair_receipt(
    *,
    result: Mapping[str, Any],
    provider: Mapping[str, Any],
    validator: str,
    expected_target_role: str | None,
    base_field_hashes: Mapping[str, str],
    provider_audit: dict[str, Any],
    recompute_report: Callable[
        [str, Mapping[str, Any]],
        tuple[dict[str, Any] | None, bool, list[str], list[str]],
    ],
    label: str,
    normalized_result_matches: Callable[[Mapping[str, Any], Mapping[str, Any]], bool]
    | None = None,
) -> dict[str, dict[str, str]]:
    role_map = {
        "validator_a": generator.VALIDATOR_A_CONTRACT_REPAIR_ROLES,
        "validator_b": generator.VALIDATOR_B_CONTRACT_REPAIR_ROLES,
    }.get(validator)
    prefixes = {
        "validator_a": generator._VALIDATOR_A_CONTRACT_CODE_PREFIXES,
        "validator_b": generator._VALIDATOR_B_CONTRACT_CODE_PREFIXES,
    }.get(validator)
    _require(role_map is not None and prefixes is not None, f"{label} type drift")
    matches = normalized_result_matches or (
        lambda replayed, recorded: replayed
        == _validator_result_without_contract_receipt(recorded)
    )
    used = result.get("contract_repair_used")
    repair = result.get("contract_repair")
    _require(
        isinstance(used, bool) and used is (repair is not None),
        f"{label} contract-repair flag drift",
    )
    if not used:
        if expected_target_role is not None:
            unexpected_role = role_map.get(expected_target_role)
            _require(
                unexpected_role not in provider,
                f"{label} has an unreceipted contract-repair provider",
            )
            selected_provider = _mapping(
                provider.get(expected_target_role),
                label=f"{label}.{expected_target_role}.provider",
            )
            selected_raw = _cached_raw_content(
                provider_audit,
                role=expected_target_role,
                provider=selected_provider,
                label=f"{label}.{expected_target_role}",
            )
            replayed, _, _, contract_reasons = recompute_report(
                selected_raw, selected_provider
            )
            _require(
                replayed is not None
                and contract_reasons == []
                and matches(replayed, result),
                f"{label} selected raw report differs from normalized result",
            )
        return {}
    repair_record = _mapping(repair, label=f"{label}.contract_repair")
    _require(
        set(repair_record)
        == {
            "target_validator_role",
            "contract_repair_role",
            "trigger_contract_error_codes",
            "invalid_report_sha256",
            "replacement_report_sha256",
        },
        f"{label} contract-repair receipt schema drift",
    )
    target_role = _string(
        repair_record.get("target_validator_role"),
        label=f"{label}.target_validator_role",
    )
    repair_role = _string(
        repair_record.get("contract_repair_role"),
        label=f"{label}.contract_repair_role",
    )
    _require(
        target_role in role_map
        and role_map[target_role] == repair_role
        and (expected_target_role is None or target_role == expected_target_role),
        f"{label} contract-repair role drift",
    )
    trigger_codes = _string_list(
        repair_record.get("trigger_contract_error_codes"),
        label=f"{label}.trigger_contract_error_codes",
        allow_empty=False,
    )
    _require(
        trigger_codes
        == generator._controlled_validator_contract_codes(validator, trigger_codes)
        and len(trigger_codes) == len(set(trigger_codes)),
        f"{label} uncontrolled contract-repair codes",
    )
    invalid_sha = _sha(
        repair_record.get("invalid_report_sha256"),
        label=f"{label}.invalid_report_sha256",
    )
    replacement_sha = _sha(
        repair_record.get("replacement_report_sha256"),
        label=f"{label}.replacement_report_sha256",
    )
    target_provider = _mapping(
        provider.get(target_role), label=f"{label}.{target_role}.provider"
    )
    repair_provider = _mapping(
        provider.get(repair_role), label=f"{label}.{repair_role}.provider"
    )
    invalid_raw = _cached_raw_content(
        provider_audit,
        role=target_role,
        provider=target_provider,
        label=f"{label}.{target_role}",
    )
    replacement_raw = _cached_raw_content(
        provider_audit,
        role=repair_role,
        provider=repair_provider,
        label=f"{label}.{repair_role}",
    )
    _require(
        sha256_text(invalid_raw) == invalid_sha
        and sha256_text(replacement_raw) == replacement_sha,
        f"{label} raw contract-repair SHA drift",
    )
    _, _, _, invalid_contract_reasons = recompute_report(invalid_raw, target_provider)
    _require(
        bool(invalid_contract_reasons),
        f"{label} contract repair was not triggered by a contract defect",
    )
    try:
        recomputed_trigger_codes = generator._controlled_validator_contract_codes(
            validator, invalid_contract_reasons
        )
    except generator.ContractError as exc:
        raise PublicationError(
            f"{label} invalid raw report has uncontrolled contract errors: "
            + ",".join(exc.reasons)
        ) from exc
    _require(
        trigger_codes == recomputed_trigger_codes,
        f"{label} trigger contract codes differ from invalid raw report",
    )
    replayed, _, _, replacement_contract_reasons = recompute_report(
        replacement_raw, repair_provider
    )
    exhausted = result.get("contract_exhausted")
    exhaustion = result.get("contract_exhaustion")
    if exhausted is True:
        exhaustion_record = _mapping(exhaustion, label=f"{label}.contract_exhaustion")
        try:
            generator._validate_contract_exhaustion_receipt(
                exhaustion_record,
                stage=validator,
                providers=provider,
                controlled_codes=lambda values: (
                    generator._controlled_validator_contract_codes(validator, values)
                ),
                sample_id=label,
            )
            residual_codes = generator._controlled_validator_contract_codes(
                validator, replacement_contract_reasons
            )
        except generator.SyntheticRewriteError as exc:
            raise PublicationError(str(exc)) from exc
        except generator.ContractError as exc:
            raise PublicationError(
                f"{label} replacement raw has uncontrolled contract errors: "
                + ",".join(exc.reasons)
            ) from exc
        _require(
            replayed is None or bool(replacement_contract_reasons),
            f"{label} exhausted replacement unexpectedly became valid",
        )
        _require(
            exhaustion_record.get("exhausted_role") == repair_role
            and exhaustion_record.get("controlled_error_codes") == residual_codes
            and result.get("contract_reasons") == residual_codes,
            f"{label} exhausted replacement code drift",
        )
        return {
            repair_role: {
                **dict(base_field_hashes),
                "contract_error_codes": _projection_hash(trigger_codes),
            }
        }
    _require(
        exhausted in {None, False} and exhaustion is None,
        f"{label} contract exhaustion flag drift",
    )
    _require(
        replayed is not None
        and replacement_contract_reasons == []
        and matches(replayed, result),
        f"{label} replacement raw report differs from normalized result",
    )
    return {
        repair_role: {
            **dict(base_field_hashes),
            "contract_error_codes": _projection_hash(trigger_codes),
        }
    }


def _expected_validator_invocation_roles(
    *,
    result: Mapping[str, Any],
    validator: str,
    base_role: str,
    label: str,
) -> set[str]:
    """Derive one invocation's exact roles without trusting its provider map."""

    role_map = {
        "validator_a": generator.VALIDATOR_A_CONTRACT_REPAIR_ROLES,
        "validator_b": generator.VALIDATOR_B_CONTRACT_REPAIR_ROLES,
    }.get(validator)
    _require(
        role_map is not None and base_role in role_map,
        f"{label} validator/base role drift",
    )
    used = result.get("contract_repair_used")
    repair = result.get("contract_repair")
    _require(
        isinstance(used, bool) and used is (repair is not None),
        f"{label} contract-repair flag drift",
    )
    roles = {base_role}
    if used:
        receipt = _mapping(repair, label=f"{label}.contract_repair")
        _require(
            receipt.get("target_validator_role") == base_role
            and receipt.get("contract_repair_role") == role_map[base_role],
            f"{label} contract-repair role drift",
        )
        roles.add(role_map[base_role])
    return roles


def _validate_validator_a_record(
    value: Any,
    *,
    label: str,
    source_analysis: str,
    reasoning: str,
    answer: str,
    provider_audit: dict[str, Any],
    expected_selected_role: str | None,
    expected_provider_roles: set[str] | None = None,
) -> tuple[bool, Mapping[str, Any]]:
    record = _mapping(value, label=label)
    _require(set(record) == AUDIT_FIELDS, f"{label} field set mismatch")
    _require(record.get("complete") is True, f"{label} is unresolved")
    result = _mapping(record.get("result"), label=f"{label}.result")
    if result.get("contract_exhausted") is True:
        _require(
            expected_selected_role is not None,
            f"{label} exhausted Validator-A role is missing",
        )
        provider = _mapping(record.get("provider"), label=f"{label}.provider")
        base_hashes = {
            "source_analysis": _projection_hash(source_analysis),
            "teacher_response_analysis": _projection_hash(reasoning),
            "rewritten_minutes": _projection_hash(answer),
        }
        expected_hashes = {expected_selected_role: base_hashes}
        expected_hashes.update(
            _validate_validator_contract_repair_receipt(
                result=result,
                provider=provider,
                validator="validator_a",
                expected_target_role=expected_selected_role,
                base_field_hashes=base_hashes,
                provider_audit=provider_audit,
                recompute_report=lambda raw, receipt: _recompute_validator_a_raw(
                    raw_content=raw,
                    provider=receipt,
                    source_analysis=source_analysis,
                    reasoning=reasoning,
                    answer=answer,
                ),
                label=label,
            )
        )
        if expected_provider_roles is not None:
            _require(
                set(provider) == expected_provider_roles,
                f"{label} provider invocation sequence drift",
            )
        _provider_complete(
            provider,
            label=f"{label}.provider",
            provider_audit=provider_audit,
            expected_field_hashes=expected_hashes,
        )
        controlled = _string_list(
            result.get("contract_reasons"),
            label=f"{label}.contract_reasons",
            allow_empty=False,
        )
        _require(
            record.get("machine_pass") is False and record.get("reasons") == controlled,
            f"{label} exhausted local verdict/reasons drift",
        )
        return False, result
    required = {
        "source_claims",
        "rewrite_claims",
        "reasoning_issues",
        "issues",
        "reported_issues",
        "ignored_operational_issues",
        "ignored_lexical_overlap_issues",
        "ignored_operational_reasoning_issues",
        "contract_reasons",
        "machine_pass",
        "local_verdict_recomputed",
        "contract_repair_used",
        "contract_repair",
    }
    _require(required <= set(result), f"{label} normalized result fields are missing")
    source_claims = result.get("source_claims")
    rewrite_claims = result.get("rewrite_claims")
    _require(
        isinstance(source_claims, list) and bool(source_claims),
        f"{label} source claims are empty",
    )
    _require(
        isinstance(rewrite_claims, list) and bool(rewrite_claims),
        f"{label} rewrite claims are empty",
    )
    source_spans: list[str] = []
    source_pass = True
    reason_codes: list[str] = []
    for index, raw in enumerate(source_claims):
        claim = _mapping(raw, label=f"{label}.source_claims[{index}]")
        _require(
            set(claim)
            == {
                "source_span",
                "reasoning_evidence",
                "rewrite_evidence",
                "rewrite_verdict",
                "reasoning_verdict",
            },
            f"{label} source claim schema drift",
        )
        span = _exact_optional_span(
            claim.get("source_span"),
            text=source_analysis,
            required=True,
            label=f"{label}.source_span[{index}]",
        )
        assert span is not None
        source_spans.append(span)
        rewrite_verdict = claim.get("rewrite_verdict")
        reasoning_verdict = claim.get("reasoning_verdict")
        _require(
            rewrite_verdict in {"entailed", "omitted", "contradicted"}
            and reasoning_verdict in {"covered", "not_covered", "conflicted"},
            f"{label} source claim verdict drift",
        )
        _exact_optional_span(
            claim.get("rewrite_evidence"),
            text=answer,
            required=rewrite_verdict in {"entailed", "contradicted"},
            label=f"{label}.rewrite_evidence[{index}]",
        )
        _exact_optional_span(
            claim.get("reasoning_evidence"),
            text=reasoning,
            required=reasoning_verdict in {"covered", "conflicted"},
            label=f"{label}.reasoning_evidence[{index}]",
        )
        if rewrite_verdict != "entailed":
            source_pass = False
            reason_codes.append("validator_a_source_claim_not_entailed")
        if reasoning_verdict != "covered":
            source_pass = False
            reason_codes.append("validator_a_reasoning_claim_not_covered")
    for index, sentence in enumerate(_sentences(source_analysis)):
        _require(
            any(sentence in span for span in source_spans),
            f"{label} source sentence is uncovered: {index}",
        )
    rewrite_spans: list[str] = []
    rewrite_pass = True
    for index, raw in enumerate(rewrite_claims):
        claim = _mapping(raw, label=f"{label}.rewrite_claims[{index}]")
        _require(
            set(claim) == {"rewrite_span", "source_evidence", "verdict"},
            f"{label} rewrite claim schema drift",
        )
        span = _exact_optional_span(
            claim.get("rewrite_span"),
            text=answer,
            required=True,
            label=f"{label}.rewrite_span[{index}]",
        )
        assert span is not None
        rewrite_spans.append(span)
        verdict = claim.get("verdict")
        _require(
            verdict in {"supported", "unsupported", "contradicted"},
            f"{label} rewrite verdict drift",
        )
        _exact_optional_span(
            claim.get("source_evidence"),
            text=source_analysis,
            required=verdict in {"supported", "contradicted"},
            label=f"{label}.source_evidence[{index}]",
        )
        if verdict != "supported":
            rewrite_pass = False
            reason_codes.append("validator_a_rewrite_claim_not_supported")
    for index, sentence in enumerate(_sentences(answer)):
        _require(
            any(sentence in span for span in rewrite_spans),
            f"{label} rewrite sentence is uncovered: {index}",
        )
    reasoning_issues = result.get("reasoning_issues")
    ignored_reasoning = result.get("ignored_operational_reasoning_issues")
    _require(
        isinstance(reasoning_issues, list) and isinstance(ignored_reasoning, list),
        f"{label} reasoning issue partitions must be lists",
    )
    factual_types = {
        "unsupported_claim",
        "source_conflict",
        "rewrite_conflict",
        "missing_claim_coverage",
    }
    for name, collection, allowed in (
        ("reasoning_issues", reasoning_issues, factual_types),
        (
            "ignored_operational_reasoning_issues",
            ignored_reasoning,
            {"meta_discussion", "duplicated_draft"},
        ),
    ):
        for index, raw in enumerate(collection):
            issue = _mapping(raw, label=f"{label}.{name}[{index}]")
            _require(
                set(issue) == {"reasoning_span", "issue_type"},
                f"{label} reasoning issue schema drift",
            )
            _exact_optional_span(
                issue.get("reasoning_span"),
                text=reasoning,
                required=True,
                label=f"{label}.{name}.reasoning_span[{index}]",
            )
            _require(
                issue.get("issue_type") in allowed,
                f"{label} reasoning issue type drift",
            )
    issues = _string_list(result.get("issues"), label=f"{label}.issues")
    reported = _string_list(
        result.get("reported_issues"), label=f"{label}.reported_issues"
    )
    ignored_operational = _string_list(
        result.get("ignored_operational_issues"),
        label=f"{label}.ignored_operational_issues",
    )
    ignored_overlap = _string_list(
        result.get("ignored_lexical_overlap_issues"),
        label=f"{label}.ignored_lexical_overlap_issues",
    )
    _require(
        Counter(reported) == Counter([*issues, *ignored_operational, *ignored_overlap]),
        f"{label} issue partition mismatch",
    )
    misclassified_factual = [
        issue
        for issue in ignored_operational
        if re.search(
            r"\b(?:wrong|incorrect|inaccurate|numeric(?:al)?|number|date|"
            r"attribution|direction|hallucinat|fact(?:ual)?)\b",
            issue,
            re.IGNORECASE,
        )
    ]
    if reasoning_issues:
        reason_codes.append("validator_a_reasoning_issues")
    if issues or misclassified_factual:
        reason_codes.append("validator_a_reported_issues")
    contract_reasons = result.get("contract_reasons")
    _require(
        contract_reasons == [] and result.get("local_verdict_recomputed") is True,
        f"{label} contains unresolved contract reasons",
    )
    local_pass = (
        source_pass
        and rewrite_pass
        and not reasoning_issues
        and not issues
        and not misclassified_factual
    )
    expected_reasons = list(dict.fromkeys(reason_codes))
    _require(
        result.get("machine_pass") is local_pass
        and record.get("machine_pass") is local_pass
        and record.get("reasons") == expected_reasons,
        f"{label} local verdict/reasons drift",
    )
    provider = _mapping(record.get("provider"), label=f"{label}.provider")
    _require(
        set(provider)
        <= {
            generator.ROLE_VALIDATOR_A_PRIMARY,
            generator.ROLE_VALIDATOR_A_FIDELITY_REPAIR,
            generator.ROLE_VALIDATOR_A_STYLE_REPAIR,
            generator.ROLE_VALIDATOR_A_PRIMARY_CONTRACT_REPAIR,
            generator.ROLE_VALIDATOR_A_FIDELITY_REPAIR_CONTRACT_REPAIR,
            generator.ROLE_VALIDATOR_A_STYLE_REPAIR_CONTRACT_REPAIR,
        }
        and bool(provider),
        f"{label} provider role drift",
    )
    if expected_provider_roles is not None:
        _require(
            set(provider) == expected_provider_roles,
            f"{label} provider invocation sequence drift",
        )
    expected_hashes: dict[str, dict[str, str]] = {}
    if expected_selected_role is not None:
        _require(
            expected_selected_role in provider, f"{label} selected role is missing"
        )
        expected_hashes[expected_selected_role] = {
            "source_analysis": _projection_hash(source_analysis),
            "teacher_response_analysis": _projection_hash(reasoning),
            "rewritten_minutes": _projection_hash(answer),
        }
        expected_hashes.update(
            _validate_validator_contract_repair_receipt(
                result=result,
                provider=provider,
                validator="validator_a",
                expected_target_role=expected_selected_role,
                base_field_hashes=expected_hashes[expected_selected_role],
                provider_audit=provider_audit,
                recompute_report=lambda raw, receipt: _recompute_validator_a_raw(
                    raw_content=raw,
                    provider=receipt,
                    source_analysis=source_analysis,
                    reasoning=reasoning,
                    answer=answer,
                ),
                label=label,
            )
        )
    _provider_complete(
        provider,
        label=f"{label}.provider",
        provider_audit=provider_audit,
        expected_field_hashes=expected_hashes,
    )
    return local_pass, result


def _deterministic_reasons(
    *, source_analysis: str, reasoning: str, answer: str
) -> list[str]:
    reasons: list[str] = []
    if not reasoning.strip():
        reasons.append("empty_teacher_response_analysis")
    if answer != answer.strip() or not answer:
        reasons.append("rewritten_minutes_outer_whitespace")
    word_count = len(_WORD_RE.findall(answer))
    if not 20 <= word_count <= 400:
        reasons.append(f"rewritten_minutes_words:{word_count}")
    if "\n" in answer or "\r" in answer:
        reasons.append("rewritten_minutes_not_single_paragraph")
    if _FINAL_BAD_RE.search(answer):
        reasons.append("rewritten_minutes_heading_list_json_or_meta")
    if _CITATION_RE.search(answer):
        reasons.append("rewritten_minutes_contains_citation")
    for name, text in (
        ("teacher_response_analysis", reasoning),
        ("rewritten_minutes", answer),
    ):
        markers = [marker for marker in _CONTROL_MARKERS if marker in text]
        if markers:
            reasons.append(f"{name}_control_markers:{','.join(markers)}")
    response = f"{reasoning}{BOUNDARY}{answer}"
    if (
        response.count("</think>") != 1
        or "<think>" in response
        or "<answer>" in response
    ):
        reasons.append("response_control_boundary_invalid")
    if _normalized_prose(answer) == _normalized_prose(source_analysis):
        reasons.append("rewritten_minutes_exactly_copies_source_analysis")
    if _numeric_values(source_analysis) != _numeric_values(answer):
        reasons.append("numeric_multiset_mismatch")
    if _date_values(source_analysis) != _date_values(answer):
        reasons.append("date_set_mismatch")
    if _attribution_categories(source_analysis) != _attribution_categories(answer):
        reasons.append("attribution_set_mismatch")
    return list(dict.fromkeys(reasons))


def _validate_deterministic_record(
    value: Any,
    *,
    label: str,
    source_analysis: str,
    reasoning: str | None,
    answer: str | None,
) -> bool:
    record = _mapping(value, label=label)
    _require(set(record) == DETERMINISTIC_FIELDS, f"{label} field set mismatch")
    cached_pass = record.get("machine_pass")
    cached_reasons = _string_list(record.get("reasons"), label=f"{label}.reasons")
    _mapping(record.get("diagnostics"), label=f"{label}.diagnostics")
    _require(isinstance(cached_pass, bool), f"{label}.machine_pass invalid")
    if reasoning is None or answer is None:
        _require(
            cached_pass is False and bool(cached_reasons),
            f"{label} textless deterministic rejection lacks reasons",
        )
        return False
    local_reasons = _deterministic_reasons(
        source_analysis=source_analysis, reasoning=reasoning, answer=answer
    )
    _require(
        not local_reasons,
        f"{label} independently failed: {','.join(local_reasons)}",
    )
    _require(
        cached_pass is True and cached_reasons == [],
        f"{label} cached verdict differs from independent checks",
    )
    return True


def _candidate_from_generation_cache(
    *,
    attempt: str,
    generation_provider: Mapping[str, Any],
    provider_audit: Mapping[str, Any],
    label: str,
) -> tuple[str, str]:
    role = {
        "primary": generator.ROLE_REWRITE_PRIMARY,
        "fidelity_repair": generator.ROLE_REWRITE_FIDELITY_REPAIR,
        "style_repair": generator.ROLE_REWRITE_STYLE_REPAIR,
    }.get(attempt)
    _require(role is not None, f"{label} candidate attempt drift")
    provider = _mapping(generation_provider.get(role), label=f"{label}.{role}.provider")
    raw_content = _cached_raw_content(
        provider_audit, role=role, provider=provider, label=f"{label}.{role}"
    )
    raw_reasoning = _cached_raw_reasoning(
        provider_audit, role=role, provider=provider, label=f"{label}.{role}"
    )
    try:
        transport = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise PublicationError(f"{label} rewrite transport is invalid JSON") from exc
    _require(
        isinstance(transport, dict)
        and set(transport) == {"answer"}
        and isinstance(transport.get("answer"), str)
        and bool(transport["answer"]),
        f"{label} rewrite transport schema drift",
    )
    return raw_reasoning, transport["answer"]


def _validate_repair_history(
    value: Any,
    *,
    label: str,
    fidelity_repair_used: bool,
    style_repair_used: bool,
    forbidden_reference_texts: Sequence[str],
    forbidden_reference_ids: Sequence[str],
    provider_audit: dict[str, Any],
    source_analysis: str,
    generation_provider: Mapping[str, Any],
    official_reference: Any,
) -> list[Mapping[str, Any]]:
    _require(isinstance(value, list), f"{label} must be a list")
    serialized = canonical_json(value)
    _require(
        "official_evidence" not in serialized
        and "official_span" not in serialized
        and "official_paragraph_id" not in serialized
        and "paragraph_id" not in serialized
        and all(text not in serialized for text in forbidden_reference_texts)
        and all(
            reference_id not in serialized for reference_id in forbidden_reference_ids
        ),
        f"{label} leaks corresponding official Minutes text or identifiers",
    )
    events: list[Mapping[str, Any]] = []
    for index, raw in enumerate(value):
        event = _mapping(raw, label=f"{label}[{index}]")
        repair_type = event.get("repair_type")
        trigger = event.get("trigger_stage")
        _require(
            repair_type in {"fidelity", "style"}
            and isinstance(trigger, str)
            and event.get("attempt") in {"primary", "fidelity_repair"},
            f"{label}[{index}] repair identity drift",
        )
        reason_codes = _string_list(
            event.get("reason_codes"),
            label=f"{label}[{index}].reason_codes",
            allow_empty=False,
        )
        if trigger == "rewrite_primary_deterministic_gate":
            _require(
                repair_type == "fidelity"
                and set(event)
                == {
                    "repair_type",
                    "trigger_stage",
                    "attempt",
                    "reason_codes",
                    "deterministic_validation",
                },
                f"{label}[{index}] deterministic repair event schema drift",
            )
            deterministic = _mapping(
                event.get("deterministic_validation"),
                label=f"{label}[{index}].deterministic_validation",
            )
            _require(
                set(deterministic) == DETERMINISTIC_FIELDS
                and deterministic.get("machine_pass") is False
                and deterministic.get("reasons") == reason_codes,
                f"{label}[{index}] deterministic snapshot drift",
            )
        elif trigger == "validator_a_primary":
            candidate_reasoning, candidate_answer = _candidate_from_generation_cache(
                attempt=str(event.get("attempt")),
                generation_provider=generation_provider,
                provider_audit=provider_audit,
                label=f"{label}[{index}].validator_a_candidate",
            )
            _require(
                repair_type == "fidelity"
                and set(event)
                == {
                    "repair_type",
                    "trigger_stage",
                    "attempt",
                    "reason_codes",
                    "validator_a",
                    "provider",
                },
                f"{label}[{index}] Validator-A repair event schema drift",
            )
            report = _mapping(
                event.get("validator_a"), label=f"{label}[{index}].validator_a"
            )
            _require(
                report.get("machine_pass") is False
                and report.get("contract_reasons") == [],
                f"{label}[{index}] Validator-A snapshot is not a quality failure",
            )
            provider = _mapping(
                event.get("provider"), label=f"{label}[{index}].provider"
            )
            a_repair_role = generator.ROLE_VALIDATOR_A_PRIMARY_CONTRACT_REPAIR
            _require(
                set(provider)
                in (
                    {generator.ROLE_VALIDATOR_A_PRIMARY},
                    {generator.ROLE_VALIDATOR_A_PRIMARY, a_repair_role},
                ),
                f"{label}[{index}] Validator-A provider role drift",
            )
            primary_provider = _mapping(
                provider.get(generator.ROLE_VALIDATOR_A_PRIMARY),
                label=f"{label}[{index}].validator_a_primary",
            )
            primary_hashes = _mapping(
                _mapping(
                    primary_provider.get("request_projection"),
                    label=f"{label}[{index}].validator_a_primary.projection",
                ).get("field_value_sha256"),
                label=f"{label}[{index}].validator_a_primary.field_hashes",
            )
            expected_hashes = {generator.ROLE_VALIDATOR_A_PRIMARY: dict(primary_hashes)}
            expected_hashes.update(
                _validate_validator_contract_repair_receipt(
                    result=report,
                    provider=provider,
                    validator="validator_a",
                    expected_target_role=generator.ROLE_VALIDATOR_A_PRIMARY,
                    base_field_hashes=primary_hashes,
                    provider_audit=provider_audit,
                    recompute_report=lambda raw, receipt: _recompute_validator_a_raw(
                        raw_content=raw,
                        provider=receipt,
                        source_analysis=source_analysis,
                        reasoning=candidate_reasoning,
                        answer=candidate_answer,
                    ),
                    label=f"{label}[{index}].validator_a",
                )
            )
            _provider_complete(
                provider,
                label=f"{label}[{index}].provider",
                provider_audit=provider_audit,
                expected_field_hashes=expected_hashes,
            )
        elif trigger == "validator_b_primary":
            candidate_reasoning, candidate_answer = _candidate_from_generation_cache(
                attempt=str(event.get("attempt")),
                generation_provider=generation_provider,
                provider_audit=provider_audit,
                label=f"{label}[{index}].validator_b_candidate",
            )
            _require(
                repair_type == "style"
                and set(event)
                == {
                    "repair_type",
                    "trigger_stage",
                    "attempt",
                    "reason_codes",
                    "validator_a",
                    "validator_b",
                    "provider",
                },
                f"{label}[{index}] Validator-B repair event schema drift",
            )
            validator_a_snapshot = _mapping(
                event.get("validator_a"),
                label=f"{label}[{index}].validator_a",
            )
            validator_a_result = _mapping(
                validator_a_snapshot.get("result"),
                label=f"{label}[{index}].validator_a.result",
            )
            validator_a_role = (
                generator.ROLE_VALIDATOR_A_FIDELITY_REPAIR
                if event.get("attempt") == "fidelity_repair"
                else generator.ROLE_VALIDATOR_A_PRIMARY
            )
            validator_a_roles = _expected_validator_invocation_roles(
                result=validator_a_result,
                validator="validator_a",
                base_role=validator_a_role,
                label=f"{label}[{index}].validator_a",
            )
            validator_a_pass, _ = _validate_validator_a_record(
                validator_a_snapshot,
                label=f"{label}[{index}].validator_a",
                source_analysis=source_analysis,
                reasoning=candidate_reasoning,
                answer=candidate_answer,
                provider_audit=provider_audit,
                expected_selected_role=validator_a_role,
                expected_provider_roles=validator_a_roles,
            )
            _require(
                validator_a_pass,
                f"{label}[{index}] style repair was not admitted by Validator A",
            )
            report = _mapping(
                event.get("validator_b"), label=f"{label}[{index}].validator_b"
            )
            _require(
                set(report)
                == {
                    "machine_pass",
                    "mean_score",
                    "min_score",
                    "style_feedback",
                    "contract_repair_used",
                    "contract_repair",
                }
                and report.get("machine_pass") is False,
                f"{label}[{index}] Validator-B snapshot is not a style failure",
            )
            feedback = _mapping(
                report.get("style_feedback"),
                label=f"{label}[{index}].validator_b.style_feedback",
            )
            _require(
                set(feedback) == {"dimensions", "critical_style_errors"}
                and "official_evidence" not in canonical_json(feedback),
                f"{label}[{index}] Validator-B history is not safely projected",
            )
            dimensions = _mapping(
                feedback.get("dimensions"),
                label=f"{label}[{index}].validator_b.style_feedback.dimensions",
            )
            _require(
                set(dimensions) == set(STYLE_DIMENSIONS),
                f"{label}[{index}] Validator-B history dimension drift",
            )
            scores: list[int] = []
            for dimension_name, raw_dimension in dimensions.items():
                dimension = _mapping(
                    raw_dimension,
                    label=f"{label}[{index}].validator_b.{dimension_name}",
                )
                _require(
                    set(dimension)
                    == {"score", "candidate_evidence", "issue_codes", "action_codes"},
                    f"{label}[{index}] Validator-B safe dimension schema drift",
                )
                score = dimension.get("score")
                _require(
                    isinstance(score, int)
                    and not isinstance(score, bool)
                    and 1 <= score <= 10,
                    f"{label}[{index}] Validator-B safe score drift",
                )
                scores.append(score)
                _string_list(
                    dimension.get("candidate_evidence"),
                    label=f"{label}[{index}].candidate_evidence",
                    allow_empty=False,
                )
                issue_codes = _string_list(
                    dimension.get("issue_codes"),
                    label=f"{label}[{index}].issue_codes",
                )
                action_codes = _string_list(
                    dimension.get("action_codes"),
                    label=f"{label}[{index}].action_codes",
                )
                _require(
                    all(code in generator.STYLE_ISSUE_CODES for code in issue_codes)
                    and all(
                        code in generator.STYLE_ACTION_CODES for code in action_codes
                    ),
                    f"{label}[{index}] Validator-B safe codes drift",
                )
            critical = feedback.get("critical_style_errors")
            _require(
                isinstance(critical, list), f"{label}[{index}] critical list drift"
            )
            for critical_index, raw_critical in enumerate(critical):
                critical_item = _mapping(
                    raw_critical,
                    label=f"{label}[{index}].critical[{critical_index}]",
                )
                _require(
                    set(critical_item) == {"error_code", "candidate_evidence"}
                    and critical_item.get("error_code")
                    in generator.CRITICAL_STYLE_CODES
                    and isinstance(critical_item.get("candidate_evidence"), str)
                    and bool(critical_item.get("candidate_evidence")),
                    f"{label}[{index}] critical safe projection drift",
                )
            _require(
                report.get("mean_score") == round(sum(scores) / len(scores), 6)
                and report.get("min_score") == min(scores),
                f"{label}[{index}] Validator-B safe score aggregate drift",
            )
            provider = _mapping(
                event.get("provider"), label=f"{label}[{index}].provider"
            )
            b_repair_role = generator.ROLE_VALIDATOR_B_PRIMARY_CONTRACT_REPAIR
            _require(
                set(provider)
                in (
                    {generator.ROLE_VALIDATOR_B_PRIMARY},
                    {generator.ROLE_VALIDATOR_B_PRIMARY, b_repair_role},
                ),
                f"{label}[{index}] Validator-B provider role drift",
            )
            primary_provider = _mapping(
                provider.get(generator.ROLE_VALIDATOR_B_PRIMARY),
                label=f"{label}[{index}].validator_b_primary",
            )
            primary_hashes = _mapping(
                _mapping(
                    primary_provider.get("request_projection"),
                    label=f"{label}[{index}].validator_b_primary.projection",
                ).get("field_value_sha256"),
                label=f"{label}[{index}].validator_b_primary.field_hashes",
            )
            expected_hashes = {generator.ROLE_VALIDATOR_B_PRIMARY: dict(primary_hashes)}
            expected_hashes.update(
                _validate_validator_contract_repair_receipt(
                    result=report,
                    provider=provider,
                    validator="validator_b",
                    expected_target_role=generator.ROLE_VALIDATOR_B_PRIMARY,
                    base_field_hashes=primary_hashes,
                    provider_audit=provider_audit,
                    recompute_report=lambda raw, receipt: _recompute_validator_b_raw(
                        raw_content=raw,
                        provider=receipt,
                        answer=candidate_answer,
                        reference=official_reference,
                    ),
                    label=f"{label}[{index}].validator_b",
                    normalized_result_matches=lambda replayed, recorded: (
                        replayed.get("machine_pass") is False
                        and recorded.get("machine_pass") is False
                        and recorded.get("mean_score") == replayed.get("mean_score")
                        and recorded.get("min_score") == replayed.get("min_score")
                        and recorded.get("style_feedback")
                        == _safe_style_feedback_projection(replayed)
                    ),
                )
            )
            _provider_complete(
                provider,
                label=f"{label}[{index}].provider",
                provider_audit=provider_audit,
                expected_field_hashes=expected_hashes,
            )
        else:
            raise PublicationError(f"{label}[{index}] trigger stage drift")
        events.append(event)
    fidelity_events = [event for event in events if event["repair_type"] == "fidelity"]
    style_events = [event for event in events if event["repair_type"] == "style"]
    _require(
        len(fidelity_events) == int(fidelity_repair_used)
        and len(style_events) == int(style_repair_used),
        f"{label} does not match repair flags",
    )
    _require(
        not style_events
        or events.index(style_events[0])
        > (events.index(fidelity_events[0]) if fidelity_events else -1),
        f"{label} repair event order drift",
    )
    return events


def _validator_roles_from_repair_history(
    events: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[str]]:
    """Derive exact history-only validator roles from sealed result receipts."""

    validator_a_roles: set[str] = set()
    validator_b_roles: set[str] = set()
    for index, event in enumerate(events):
        trigger = event.get("trigger_stage")
        if trigger == "validator_a_primary":
            result = _mapping(
                event.get("validator_a"),
                label=f"repair_history[{index}].validator_a",
            )
            validator_a_roles.update(
                _expected_validator_invocation_roles(
                    result=result,
                    validator="validator_a",
                    base_role=generator.ROLE_VALIDATOR_A_PRIMARY,
                    label=f"repair_history[{index}].validator_a",
                )
            )
        elif trigger == "validator_b_primary":
            validator_a_snapshot = _mapping(
                event.get("validator_a"),
                label=f"repair_history[{index}].validator_a",
            )
            validator_a_result = _mapping(
                validator_a_snapshot.get("result"),
                label=f"repair_history[{index}].validator_a.result",
            )
            validator_a_base_role = (
                generator.ROLE_VALIDATOR_A_FIDELITY_REPAIR
                if event.get("attempt") == "fidelity_repair"
                else generator.ROLE_VALIDATOR_A_PRIMARY
            )
            validator_a_roles.update(
                _expected_validator_invocation_roles(
                    result=validator_a_result,
                    validator="validator_a",
                    base_role=validator_a_base_role,
                    label=f"repair_history[{index}].validator_a",
                )
            )
            validator_b_result = _mapping(
                event.get("validator_b"),
                label=f"repair_history[{index}].validator_b",
            )
            validator_b_roles.update(
                _expected_validator_invocation_roles(
                    result=validator_b_result,
                    validator="validator_b",
                    base_role=generator.ROLE_VALIDATOR_B_PRIMARY,
                    label=f"repair_history[{index}].validator_b",
                )
            )
    return validator_a_roles, validator_b_roles


def _machine_record(
    value: Any, *, label: str, fields: set[str]
) -> tuple[bool, bool, Mapping[str, Any]]:
    record = _mapping(value, label=label)
    _require(set(record) == fields, f"{label} field set mismatch")
    complete = record.get("complete")
    machine_pass = record.get("machine_pass")
    _require(
        isinstance(complete, bool) and isinstance(machine_pass, bool),
        f"{label} complete/pass must be boolean",
    )
    _string_list(record.get("reasons"), label=f"{label}.reasons")
    if complete:
        result = _mapping(record.get("result"), label=f"{label}.result")
    else:
        _require(
            machine_pass is False
            and record.get("result") is None
            and record.get("provider") == {},
            f"{label} unrun state drift",
        )
        result = {}
    _require(not machine_pass or complete, f"{label} incomplete PASS")
    return complete, machine_pass, result


def _validate_validator_b(
    value: Any,
    *,
    label: str,
    answer: str | None,
    reference: Any,
    provider_audit: dict[str, Any],
    expected_field_hashes: Mapping[str, Mapping[str, str]] | None = None,
    expected_selected_role: str | None = None,
    expected_provider_roles: set[str] | None = None,
) -> tuple[bool, bool, float | None, int | None, str | None]:
    record = _mapping(value, label=label)
    _require(set(record) == VALIDATOR_B_FIELDS, f"{label} field set mismatch")
    complete = record.get("complete")
    machine_pass = record.get("machine_pass")
    _require(
        isinstance(complete, bool) and isinstance(machine_pass, bool),
        f"{label} complete/pass must be boolean",
    )
    validator_reasons = _string_list(record.get("reasons"), label=f"{label}.reasons")
    if not complete:
        _require(
            machine_pass is False
            and record.get("mean_score") is None
            and record.get("min_score") is None
            and record.get("result") is None
            and record.get("provider") == {},
            f"{label} unrun state drift",
        )
        return False, False, None, None, None
    _require(answer is not None, f"{label} complete without rewritten Minutes")
    result = _mapping(record.get("result"), label=f"{label}.result")
    if result.get("contract_exhausted") is True:
        _require(
            expected_selected_role is not None,
            f"{label} exhausted Validator-B role is missing",
        )
        provider = _mapping(record.get("provider"), label=f"{label}.provider")
        completed_expected_hashes = {
            role: dict(hashes) for role, hashes in (expected_field_hashes or {}).items()
        }
        _require(
            expected_selected_role in completed_expected_hashes,
            f"{label} selected Validator-B hash binding is missing",
        )
        completed_expected_hashes.update(
            _validate_validator_contract_repair_receipt(
                result=result,
                provider=provider,
                validator="validator_b",
                expected_target_role=expected_selected_role,
                base_field_hashes=completed_expected_hashes[expected_selected_role],
                provider_audit=provider_audit,
                recompute_report=lambda raw, receipt: _recompute_validator_b_raw(
                    raw_content=raw,
                    provider=receipt,
                    answer=answer,
                    reference=reference,
                ),
                label=label,
            )
        )
        if expected_provider_roles is not None:
            _require(
                set(provider) == expected_provider_roles,
                f"{label} provider invocation sequence drift",
            )
        _provider_complete(
            provider,
            label=f"{label}.provider",
            provider_audit=provider_audit,
            expected_field_hashes=completed_expected_hashes,
        )
        controlled = _string_list(
            result.get("contract_reasons"),
            label=f"{label}.contract_reasons",
            allow_empty=False,
        )
        _require(
            machine_pass is False
            and validator_reasons == controlled
            and record.get("mean_score") is None
            and record.get("min_score") is None,
            f"{label} exhausted local verdict/reasons drift",
        )
        return True, False, None, None, None
    _require(
        set(result)
        == {
            "comparison_status",
            "passage_matches",
            "dimensions",
            "critical_style_errors",
            "overall_pass",
            "mean_score",
            "min_score",
            "machine_pass",
            "contract_reasons",
            "contract_repair_used",
            "contract_repair",
        },
        f"{label} normalized result field drift",
    )
    comparison_status = result.get("comparison_status")
    _require(
        comparison_status in {"comparable", "no_comparable_passage"},
        f"{label} comparison status drift",
    )
    _require(
        getattr(reference, "meeting_date", None),
        f"{label} missing corresponding meeting reference",
    )
    paragraph_index = {
        paragraph.paragraph_id: paragraph.text for paragraph in reference.paragraphs
    }
    _require(
        len(paragraph_index) == len(reference.paragraphs) and bool(paragraph_index),
        f"{label} corresponding official paragraph IDs are not unique",
    )
    passage_matches = result.get("passage_matches")
    _require(isinstance(passage_matches, list), f"{label} passage matches drift")
    candidate_match_spans: list[str] = []
    matched_paragraph_ids: set[str] = set()
    for index, raw_match in enumerate(passage_matches):
        match = _mapping(raw_match, label=f"{label}.passage_matches[{index}]")
        _require(
            set(match)
            == {
                "candidate_span",
                "official_paragraph_id",
                "official_span",
                "match_type",
            },
            f"{label} passage-match schema drift",
        )
        candidate_span = _string(
            match.get("candidate_span"), label=f"{label}.candidate_span[{index}]"
        )
        paragraph_id = _string(
            match.get("official_paragraph_id"),
            label=f"{label}.official_paragraph_id[{index}]",
        )
        official_span = _string(
            match.get("official_span"), label=f"{label}.official_span[{index}]"
        )
        _require(
            candidate_span in answer, f"{label} candidate passage span is not exact"
        )
        _require(
            paragraph_id in paragraph_index
            and official_span in paragraph_index[paragraph_id],
            f"{label} official passage span/paragraph is not from the corresponding meeting",
        )
        _require(
            match.get("match_type") in generator.PASSAGE_MATCH_TYPES,
            f"{label} passage match type drift",
        )
        candidate_match_spans.append(candidate_span)
        matched_paragraph_ids.add(paragraph_id)
    if comparison_status == "comparable":
        _require(bool(passage_matches), f"{label} comparable result has no matches")
        for sentence_index, sentence in enumerate(_sentences(answer)):
            _require(
                any(sentence in span for span in candidate_match_spans),
                f"{label} candidate sentence is not passage-matched: {sentence_index}",
            )
    else:
        _require(
            passage_matches == [], f"{label} no-comparable result contains matches"
        )

    dimensions = _mapping(result.get("dimensions"), label=f"{label}.result.dimensions")
    _require(set(dimensions) == set(STYLE_DIMENSIONS), f"{label} dimension set drift")
    scores: list[int] = []
    for name in STYLE_DIMENSIONS:
        dimension = _mapping(dimensions[name], label=f"{label}.{name}")
        _require(
            set(dimension)
            == {
                "score",
                "candidate_evidence",
                "official_evidence",
                "issue_codes",
                "action_codes",
            },
            f"{label}.{name} field set mismatch",
        )
        score = dimension.get("score")
        _require(
            isinstance(score, int) and not isinstance(score, bool) and 1 <= score <= 10,
            f"{label}.{name}.score invalid",
        )
        scores.append(score)
        candidate_evidence = _string_list(
            dimension.get("candidate_evidence"),
            label=f"{label}.{name}.candidate_evidence",
            allow_empty=False,
        )
        _require(
            all(span in answer for span in candidate_evidence),
            f"{label}.{name} candidate evidence is not exact",
        )
        issue_codes = _string_list(
            dimension.get("issue_codes"), label=f"{label}.{name}.issue_codes"
        )
        action_codes = _string_list(
            dimension.get("action_codes"), label=f"{label}.{name}.action_codes"
        )
        _require(
            all(code in generator.STYLE_ISSUE_CODES for code in issue_codes)
            and all(code in generator.STYLE_ACTION_CODES for code in action_codes),
            f"{label}.{name} contains uncontrolled issue/action codes",
        )
        evidence = dimension.get("official_evidence")
        _require(
            isinstance(evidence, list)
            and (comparison_status != "comparable" or bool(evidence)),
            f"{label}.{name}.official_evidence contract drift",
        )
        for evidence_index, evidence_item in enumerate(evidence):
            item = _mapping(
                evidence_item,
                label=f"{label}.{name}.official_evidence[{evidence_index}]",
            )
            _require(
                set(item) == {"paragraph_id", "exact_span", "style_feature_code"},
                f"{label}.{name} official evidence schema drift",
            )
            paragraph_id = _string(
                item.get("paragraph_id"), label=f"{label}.{name}.paragraph_id"
            )
            _require(
                paragraph_id in paragraph_index,
                f"{label} references an official paragraph outside the corresponding meeting",
            )
            _require(
                comparison_status != "comparable"
                or paragraph_id in matched_paragraph_ids,
                f"{label}.{name} official evidence is not from a passage-matched paragraph",
            )
            exact_span = _string(
                item.get("exact_span"), label=f"{label}.{name}.exact_span"
            )
            feature_code = _string(
                item.get("style_feature_code"),
                label=f"{label}.{name}.style_feature_code",
            )
            _require(
                exact_span in paragraph_index[paragraph_id],
                f"{label}.{name} official evidence is not exact",
            )
            _require(
                feature_code in generator.STYLE_FEATURE_CODES,
                f"{label}.{name} contains an uncontrolled style feature code",
            )
    critical = result.get("critical_style_errors")
    _require(
        isinstance(critical, list), f"{label}.critical_style_errors must be a list"
    )
    for index, error in enumerate(critical):
        error_record = _mapping(error, label=f"{label}.critical_style_errors[{index}]")
        _require(
            set(error_record) == {"error_code", "candidate_evidence"},
            f"{label} critical error schema drift",
        )
        error_code = _string(
            error_record.get("error_code"), label=f"{label}.critical error code"
        )
        candidate_span = _string(
            error_record.get("candidate_evidence"),
            label=f"{label}.critical candidate evidence",
        )
        _require(
            error_code in generator.CRITICAL_STYLE_CODES and candidate_span in answer,
            f"{label} critical style error evidence/code drift",
        )
    copied_reference = any(answer == text for text in paragraph_index.values())
    reported_copy = any(
        _mapping(item, label=f"{label}.critical_copy").get("error_code")
        == "VERBATIM_REFERENCE_COPY"
        for item in critical
    )
    _require(
        copied_reference == reported_copy,
        f"{label} verbatim official-paragraph copy verdict drift",
    )
    contract_reasons = _string_list(
        result.get("contract_reasons"), label=f"{label}.contract_reasons"
    )
    _require(not contract_reasons, f"{label} contains unresolved contract errors")
    mean_score = round(sum(scores) / len(scores), 6)
    min_score = min(scores)
    computed = (
        comparison_status == "comparable"
        and mean_score >= STYLE_MEAN_THRESHOLD
        and min_score >= STYLE_MIN_THRESHOLD
        and not critical
    )
    expected_reasons: list[str] = []
    if comparison_status == "no_comparable_passage":
        expected_reasons.append("validator_b_no_comparable_passage")
    if mean_score < STYLE_MEAN_THRESHOLD:
        expected_reasons.append("validator_b_mean_below_7")
    if min_score < STYLE_MIN_THRESHOLD:
        expected_reasons.append("validator_b_dimension_below_6")
    if critical:
        expected_reasons.append("validator_b_critical_style_error")
    _require(
        validator_reasons == expected_reasons,
        f"{label} top-level reason codes differ from recomputed gate",
    )
    _require(
        isinstance(record.get("mean_score"), (int, float))
        and not isinstance(record.get("mean_score"), bool),
        f"{label}.mean_score invalid",
    )
    _require(
        abs(float(record["mean_score"]) - mean_score) < 1e-9,
        f"{label}.mean_score drift",
    )
    _require(record.get("min_score") == min_score, f"{label}.min_score drift")
    _require(
        result.get("mean_score") == record.get("mean_score")
        and result.get("min_score") == min_score,
        f"{label} result/top-level score drift",
    )
    _require(
        result.get("machine_pass") is computed
        and isinstance(result.get("overall_pass"), bool),
        f"{label} result verdict drift",
    )
    _require(
        machine_pass is computed,
        f"{label} machine_pass does not match recomputed style gate",
    )
    provider = _mapping(record.get("provider"), label=f"{label}.provider")
    _require(
        bool(provider)
        and set(provider)
        <= {
            generator.ROLE_VALIDATOR_B_PRIMARY,
            generator.ROLE_VALIDATOR_B_STYLE_REPAIR,
            generator.ROLE_VALIDATOR_B_PRIMARY_CONTRACT_REPAIR,
            generator.ROLE_VALIDATOR_B_STYLE_REPAIR_CONTRACT_REPAIR,
        },
        f"{label} provider role drift",
    )
    if expected_provider_roles is not None:
        _require(
            set(provider) == expected_provider_roles,
            f"{label} provider invocation sequence drift",
        )
    completed_expected_hashes = {
        role: dict(hashes) for role, hashes in (expected_field_hashes or {}).items()
    }
    if expected_selected_role is not None:
        _require(
            expected_selected_role in completed_expected_hashes,
            f"{label} selected Validator-B role hash binding is missing",
        )
        completed_expected_hashes.update(
            _validate_validator_contract_repair_receipt(
                result=result,
                provider=provider,
                validator="validator_b",
                expected_target_role=expected_selected_role,
                base_field_hashes=completed_expected_hashes[expected_selected_role],
                provider_audit=provider_audit,
                recompute_report=lambda raw, receipt: _recompute_validator_b_raw(
                    raw_content=raw,
                    provider=receipt,
                    answer=answer,
                    reference=reference,
                ),
                label=label,
            )
        )
    _provider_complete(
        provider,
        label=f"{label}.provider",
        provider_audit=provider_audit,
        expected_field_hashes=completed_expected_hashes,
    )
    return True, computed, mean_score, min_score, str(comparison_status)


def _safe_style_feedback_projection(result: Mapping[str, Any]) -> dict[str, Any]:
    dimensions = _mapping(result.get("dimensions"), label="validator_b.dimensions")
    return {
        "dimensions": {
            name: {
                "score": dimensions[name]["score"],
                "candidate_evidence": list(dimensions[name]["candidate_evidence"]),
                "issue_codes": list(dimensions[name]["issue_codes"]),
                "action_codes": list(dimensions[name]["action_codes"]),
            }
            for name in STYLE_DIMENSIONS
        },
        "critical_style_errors": [
            {
                "error_code": item["error_code"],
                "candidate_evidence": item["candidate_evidence"],
            }
            for item in result["critical_style_errors"]
        ],
    }


def _validate_style_repair_trigger_snapshot(
    *,
    event: Mapping[str, Any],
    generation_provider: Mapping[str, Any],
    reference: Any,
    official_payload: Mapping[str, Any],
    provider_audit: dict[str, Any],
    label: str,
) -> str:
    attempt = event.get("attempt")
    rewrite_role = (
        generator.ROLE_REWRITE_FIDELITY_REPAIR
        if attempt == "fidelity_repair"
        else generator.ROLE_REWRITE_PRIMARY
    )
    rewrite_provider = _mapping(
        generation_provider.get(rewrite_role), label=f"{label}.{rewrite_role}"
    )
    raw_rewrite = _cached_raw_content(
        provider_audit,
        role=rewrite_role,
        provider=rewrite_provider,
        label=f"{label}.{rewrite_role}",
    )
    try:
        rewrite_transport = json.loads(raw_rewrite)
    except json.JSONDecodeError as exc:
        raise PublicationError(
            f"{label} pre-repair rewrite cache is invalid JSON"
        ) from exc
    _require(
        isinstance(rewrite_transport, dict)
        and set(rewrite_transport) == {"answer"}
        and isinstance(rewrite_transport.get("answer"), str)
        and bool(rewrite_transport["answer"]),
        f"{label} pre-repair rewrite transport drift",
    )
    pre_repair_answer = rewrite_transport["answer"]
    event_provider = _mapping(event.get("provider"), label=f"{label}.provider")
    report = _mapping(event.get("validator_b"), label=f"{label}.validator_b")
    b_role = (
        generator.ROLE_VALIDATOR_B_PRIMARY_CONTRACT_REPAIR
        if report.get("contract_repair_used") is True
        else generator.ROLE_VALIDATOR_B_PRIMARY
    )
    b_provider = _mapping(
        event_provider.get(b_role),
        label=f"{label}.{b_role}",
    )
    raw_b = _cached_raw_content(
        provider_audit,
        role=b_role,
        provider=b_provider,
        label=f"{label}.{b_role}",
    )
    try:
        raw_result = json.loads(raw_b)
    except json.JSONDecodeError as exc:
        raise PublicationError(
            f"{label} primary Validator-B cache is invalid JSON"
        ) from exc
    _require(
        isinstance(raw_result, dict)
        and set(raw_result)
        == {
            "comparison_status",
            "passage_matches",
            "dimensions",
            "critical_style_errors",
            "overall_pass",
        },
        f"{label} primary Validator-B raw schema drift",
    )
    dimensions = _mapping(raw_result.get("dimensions"), label=f"{label}.dimensions")
    _require(set(dimensions) == set(STYLE_DIMENSIONS), f"{label} dimension drift")
    scores = [dimensions[name].get("score") for name in STYLE_DIMENSIONS]
    _require(
        all(isinstance(score, int) and not isinstance(score, bool) for score in scores),
        f"{label} score drift",
    )
    critical = list(raw_result.get("critical_style_errors") or [])
    if any(pre_repair_answer == paragraph.text for paragraph in reference.paragraphs):
        if not any(
            isinstance(item, Mapping)
            and item.get("error_code") == "VERBATIM_REFERENCE_COPY"
            for item in critical
        ):
            critical.append(
                {
                    "error_code": "VERBATIM_REFERENCE_COPY",
                    "candidate_evidence": pre_repair_answer,
                }
            )
    mean_score = round(sum(scores) / len(scores), 6)
    min_score = min(scores)
    reasons: list[str] = []
    if raw_result.get("comparison_status") == "no_comparable_passage":
        reasons.append("validator_b_no_comparable_passage")
    if mean_score < STYLE_MEAN_THRESHOLD:
        reasons.append("validator_b_mean_below_7")
    if min_score < STYLE_MIN_THRESHOLD:
        reasons.append("validator_b_dimension_below_6")
    if critical:
        reasons.append("validator_b_critical_style_error")
    normalized = {
        **raw_result,
        "critical_style_errors": critical,
        "mean_score": mean_score,
        "min_score": min_score,
        "machine_pass": False,
        "contract_reasons": [],
        "contract_repair_used": report.get("contract_repair_used"),
        "contract_repair": report.get("contract_repair"),
    }
    synthetic_record = {
        "complete": True,
        "machine_pass": False,
        "mean_score": mean_score,
        "min_score": min_score,
        "reasons": reasons,
        "result": normalized,
        "provider": dict(event_provider),
    }
    _, passed, _, _, comparison_status = _validate_validator_b(
        synthetic_record,
        label=f"{label}.recomputed_validator_b",
        answer=pre_repair_answer,
        reference=reference,
        provider_audit=provider_audit,
        expected_field_hashes={
            generator.ROLE_VALIDATOR_B_PRIMARY: {
                "rewritten_minutes": _projection_hash(pre_repair_answer),
                "corresponding_official_minutes_pre_action": _projection_hash(
                    official_payload
                ),
            }
        },
        expected_selected_role=generator.ROLE_VALIDATOR_B_PRIMARY,
    )
    _require(
        not passed
        and comparison_status == "comparable"
        and normalized["critical_style_errors"] == []
        and bool(reasons)
        and set(reasons)
        <= {
            "validator_b_mean_below_7",
            "validator_b_dimension_below_6",
        },
        f"{label} is not a comparable low-score-only repair-eligible failure",
    )
    _require(
        report.get("mean_score") == mean_score
        and report.get("min_score") == min_score
        and report.get("machine_pass") is False
        and report.get("style_feedback") == _safe_style_feedback_projection(normalized),
        f"{label} safe style feedback differs from recomputed primary Validator B",
    )
    return pre_repair_answer


def _token_replay(
    *, tokenizer: Any, system_prompt: str, prompt: str, response: str, sample_id: str
) -> dict[str, int]:
    _require(response.count(BOUNDARY) == 1, f"invalid response boundary: {sample_id}")
    _require(
        "<think>" not in response.lower(),
        f"response contains opening think tag: {sample_id}",
    )
    reasoning, answer = response.split(BOUNDARY, 1)
    _require(bool(reasoning.strip()), f"empty teacher reasoning: {sample_id}")
    _require(
        answer == answer.strip() and bool(answer),
        f"invalid answer whitespace: {sample_id}",
    )
    _require(
        "\n" not in answer and not _CONTROL_RE.search(answer),
        f"answer is not one clean paragraph: {sample_id}",
    )
    rendered = render_sft_prompt(
        tokenizer,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    _require(
        rendered.count("<think>") == 1 and "</think>" not in rendered,
        f"chat-template think prefix drift: {sample_id}",
    )
    eos = _string(getattr(tokenizer, "eos_token", None), label="tokenizer.eos_token")
    completion = response if response.endswith(eos) else response + eos
    prompt_ids = tokenize_sft_text(tokenizer, rendered)
    full_ids = tokenize_sft_text(tokenizer, rendered + completion)
    _require(
        full_ids[: len(prompt_ids)] == prompt_ids,
        f"completion-only prompt prefix drift: {sample_id}",
    )
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    _require(
        bos_id is not None and full_ids.count(int(bos_id)) == 1,
        f"single-BOS gate failed: {sample_id}",
    )
    _require(
        eos_id is not None
        and full_ids.count(int(eos_id)) == 1
        and full_ids[-1] == int(eos_id),
        f"single-EOS gate failed: {sample_id}",
    )
    _require(
        len(full_ids) <= MAX_TOTAL_TOKENS, f"sample exceeds 4,096 tokens: {sample_id}"
    )
    completion_tokens = len(full_ids) - len(prompt_ids)
    _require(completion_tokens > 0, f"empty completion mask: {sample_id}")
    return {
        "prompt_tokens": len(prompt_ids),
        "reasoning_tokens": len(tokenize_sft_text(tokenizer, reasoning)),
        "answer_tokens": len(tokenize_sft_text(tokenizer, answer)),
        "completion_tokens": completion_tokens,
        "total_tokens": len(full_ids),
        "masked_prompt_tokens": len(prompt_ids),
        "unmasked_completion_tokens": completion_tokens,
    }


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _stats(values: Sequence[int | float]) -> dict[str, int | float]:
    if not values:
        return {"min": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": ordered[round((len(ordered) - 1) * 0.50)],
        "p95": ordered[round((len(ordered) - 1) * 0.95)],
        "p99": ordered[round((len(ordered) - 1) * 0.99)],
        "max": ordered[-1],
    }


def _file_records(root: Path, paths: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {
        relative: _descriptor(root / relative, relative_to=root)
        for relative in sorted(paths)
    }


def _load_acquisition(
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    dict[str, Path],
    dict[tuple[str, str], str],
    dict[tuple[str, str], str],
    dict[tuple[str, str], dict[str, Any]],
]:
    root = root.resolve()
    _require(
        root.is_dir() and not root.is_symlink(), f"invalid acquisition root: {root}"
    )
    preparation = _read_json(
        root / "preparation_summary.json", label="preparation summary"
    )
    prepare_manifest = _read_json(
        root / "prepare_manifest.json", label="prepare manifest"
    )
    receipt = _read_json(
        root / "source_admission_receipt.json", label="source admission receipt"
    )
    contract = _read_json(root / "prompt_contract.json", label="prompt contract")
    final = _read_json(root / "final_summary.json", label="final summary")
    _require(
        _contains_secret((preparation, prepare_manifest, receipt, contract, final))
        is None,
        "acquisition metadata contains credential material",
    )
    _require(
        not (root / "style_bank.json").exists(),
        "legacy v1 style-bank artifact is forbidden in a v2 acquisition",
    )
    provider_cache_raw: dict[tuple[str, str], str] = {}
    provider_cache_reasoning: dict[tuple[str, str], str] = {}
    provider_cache_records: dict[tuple[str, str], dict[str, Any]] = {}
    for cache_path in sorted((root / "cache").glob("**/*.json")):
        cache_record = _read_json(cache_path, label=f"provider cache {cache_path.name}")
        binding = cache_record.get("binding")
        cache_schema = (
            binding.get("schema_version") if isinstance(binding, Mapping) else None
        )
        _require(
            cache_schema
            not in {
                "paper-chk2-chk1-provider-cache-v1",
                "paper-chk2-chk1-terminal-v1",
            },
            f"legacy v1 cache is forbidden: {cache_path}",
        )
        if cache_schema == generator.CACHE_SCHEMA_VERSION:
            binding_map = _mapping(binding, label=f"cache binding {cache_path}")
            _require(
                cache_record.get("binding_sha256")
                == sha256_text(canonical_json(binding_map)),
                f"cache binding SHA drift: {cache_path}",
            )
            role = _string(binding_map.get("role"), label=f"cache role {cache_path}")
            _require(
                role in generator.PROVIDER_ROLES, f"cache role drift: {cache_path}"
            )
            cache_sample_id = _string(
                binding_map.get("sample_id"), label=f"cache sample ID {cache_path}"
            )
            canonical_cache_path = (
                root / "cache" / role / f"{sha256_text(cache_sample_id)}.json"
            )
            _require(
                not cache_path.is_symlink()
                and cache_path.resolve() == canonical_cache_path.resolve(),
                f"provider cache path/binding drift: {cache_path}",
            )
            response = _mapping(
                cache_record.get("provider_response"),
                label=f"cache provider response {cache_path}",
            )
            projection = _mapping(
                cache_record.get("request_projection"),
                label=f"cache request projection {cache_path}",
            )
            _require(
                projection.get("role") == role
                and binding_map.get("request_projection_sha256")
                == sha256_text(canonical_json(projection)),
                f"cache request projection binding drift: {cache_path}",
            )
            response_id = _string(
                response.get("response_id"), label=f"cache response id {cache_path}"
            )
            raw_content = _string(
                response.get("raw_content"), label=f"cache raw content {cache_path}"
            )
            raw_reasoning = response.get("raw_reasoning")
            _require(
                isinstance(raw_reasoning, str),
                f"cache raw reasoning must be text: {cache_path}",
            )
            _require(
                cache_record.get("raw_content_sha256") == sha256_text(raw_content),
                f"cache raw-content SHA drift: {cache_path}",
            )
            _require(
                cache_record.get("raw_reasoning_sha256") == sha256_text(raw_reasoning),
                f"cache raw-reasoning SHA drift: {cache_path}",
            )
            cache_key = (role, response_id)
            _require(
                cache_key not in provider_cache_raw,
                f"duplicate provider cache receipt: {cache_key}",
            )
            provider_cache_raw[cache_key] = raw_content
            provider_cache_reasoning[cache_key] = raw_reasoning
            provider_cache_records[cache_key] = dict(cache_record)
    _require(
        preparation.get("schema_version") == generator.PREPARED_SCHEMA_VERSION
        and prepare_manifest.get("schema_version") == generator.PREPARED_SCHEMA_VERSION
        and receipt.get("schema_version") == generator.SOURCE_AUDIT_SCHEMA_VERSION
        and final.get("schema_version") == generator.SUMMARY_SCHEMA_VERSION
        and final.get("quality_status") == "passed"
        and final.get("unresolved_failure_count") == 0,
        "acquisition is not complete and passed",
    )
    reference_path = root / "official_pre_action_reference_bank.jsonl"
    bindings = {
        "preparation_summary_sha256": root / "preparation_summary.json",
        "source_admission_receipt_sha256": root / "source_admission_receipt.json",
        "prompt_contract_sha256": root / "prompt_contract.json",
        "official_reference_bank_sha256": reference_path,
    }
    for key, path in bindings.items():
        _require(
            final.get(key) == sha256_file(path), f"final summary binding drift: {key}"
        )
    artifacts = _mapping(final.get("artifacts"), label="final_summary.artifacts")
    described_reference = _resolve_descriptor(
        root,
        artifacts.get("official_reference_bank"),
        label="acquisition official_reference_bank",
    )
    _require(
        described_reference == reference_path.resolve(),
        "official reference descriptor path drift",
    )
    terminal: dict[str, list[dict[str, Any]]] = {}
    paths: dict[str, Path] = {}
    for split in SPLITS:
        split_artifacts = _mapping(
            artifacts.get(split), label=f"final_summary.artifacts.{split}"
        )
        for name in ("terminal", "sft_candidate", "manifest"):
            path = _resolve_descriptor(
                root, split_artifacts.get(name), label=f"acquisition {split}.{name}"
            )
            paths[f"{split}.{name}"] = path
        terminal[split] = _read_jsonl(
            paths[f"{split}.terminal"], label=f"acquisition {split} terminal"
        )
    audit_artifacts = _mapping(
        artifacts.get("audits"), label="final_summary.artifacts.audits"
    )
    for name in ("rejections", "validator_b", "repair_history"):
        paths[f"audits.{name}"] = _resolve_descriptor(
            root, audit_artifacts.get(name), label=f"acquisition audit {name}"
        )
    paths["official_reference_bank"] = described_reference
    return (
        preparation,
        prepare_manifest,
        receipt,
        contract,
        final,
        terminal,
        paths,
        provider_cache_raw,
        provider_cache_reasoning,
        provider_cache_records,
    )


def publish_release(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    generation_manifest_root: Path = DEFAULT_GENERATION_MANIFEST_ROOT,
    acquisition_root: Path = DEFAULT_ACQUISITION_ROOT,
    release_root: Path = DEFAULT_RELEASE_ROOT,
    tokenizer: Any,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    expected_tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    expected_split_counts: Mapping[str, int] = SOURCE_SPLIT_COUNTS,
    expected_meeting_counts: Mapping[str, int] = SOURCE_MEETING_COUNTS,
    expected_sample_id_sha256: str = SOURCE_SAMPLE_ID_SHA256,
    expected_source_file_sha256: Mapping[str, str] | None = SOURCE_FILE_SHA256,
    official_roster_path: Path = DEFAULT_OFFICIAL_ROSTER_PATH,
    expected_official_roster_sha256: str = EXPECTED_OFFICIAL_ROSTER_SHA256,
    expected_boundary_method_counts: Mapping[str, int] = (
        EXPECTED_BOUNDARY_METHOD_COUNTS
    ),
    official_reference_exception_boundaries: Mapping[str, Mapping[str, Any]] = (
        PINNED_2009_ACTION_BOUNDARIES
    ),
) -> dict[str, Any]:
    """Validate and atomically publish a PASS-only chk-2 SFT release."""

    publisher_implementation = {
        "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "sha256": sha256_file(Path(__file__).resolve()),
    }
    _require(
        tokenizer_path.resolve() == expected_tokenizer_path.resolve(),
        "tokenizer path is not the exact authorized chk1 cp200 tokenizer",
    )
    _require(
        tokenizer_path.resolve().is_dir() and not tokenizer_path.resolve().is_symlink(),
        "exact chk1 cp200 tokenizer directory is missing or unsafe",
    )
    _validate_exact_tokenizer_files(
        tokenizer_path, expected_tokenizer_path=expected_tokenizer_path
    )
    tokenizer_runtime_contract = generator._expected_tokenizer_runtime_contract()
    if expected_tokenizer_path.resolve() == DEFAULT_TOKENIZER_PATH.resolve():
        generator._verify_tokenizer_runtime_contract(tokenizer)
    source_by_split, source_bindings, upstream_repair_provenance = _load_source_rows(
        source_root,
        generation_manifest_root,
        expected_split_counts=expected_split_counts,
        expected_sample_id_sha256=expected_sample_id_sha256,
        expected_source_file_sha256=expected_source_file_sha256,
    )
    (
        preparation,
        prepare_manifest,
        source_receipt,
        contract,
        final_summary,
        terminal_by_split,
        acquisition_paths,
        provider_cache_raw,
        provider_cache_reasoning,
        provider_cache_records,
    ) = _load_acquisition(acquisition_root)
    system_prompt, contract_lineage = _validate_prompt_contract(contract)
    prep_source = _mapping(
        preparation.get("source"), label="preparation_summary.source"
    )
    _require(
        prep_source.get("split_counts") == dict(expected_split_counts),
        "preparation source split-count drift",
    )
    _require(
        prep_source.get("meeting_counts") == dict(expected_meeting_counts),
        "preparation source meeting-count drift",
    )
    _require(
        prep_source.get("total_rows") == sum(expected_split_counts.values()),
        "preparation source total drift",
    )
    _require(
        prep_source.get("sample_id_sha256") == expected_sample_id_sha256,
        "preparation source ID digest drift",
    )
    _require(
        prep_source.get("upstream_chk1_candidate_repair_provenance")
        == upstream_repair_provenance,
        "preparation upstream chk1 repair provenance drift",
    )
    observed_source_shas = _collect_descriptor_shas(prep_source.get("artifacts"))
    required_source_shas = {record["sha256"] for record in source_bindings.values()}
    _require(
        required_source_shas <= observed_source_shas,
        "preparation does not bind every exact chk1 source artifact",
    )
    _validate_required_lineage(
        preparation.get("lineage"), label="preparation_summary.lineage"
    )

    source_total = sum(expected_split_counts.values())
    _require(
        source_receipt.get("status") == "source_admission_complete"
        and source_receipt.get("authorized_scope")
        == "paper_chk2_dataset_construction_only",
        "source admission receipt is not complete or correctly scoped",
    )
    _validate_required_lineage(
        source_receipt.get("lineage"), label="source_admission_receipt.lineage"
    )
    receipt_source = _mapping(
        source_receipt.get("source"), label="source_admission_receipt.source"
    )
    _require(
        receipt_source == prep_source,
        "source admission receipt source binding drift",
    )
    receipt_counts = _mapping(
        source_receipt.get("status_counts"),
        label="source_admission_receipt.status_counts",
    )
    admitted = 0
    rejected = 0
    _require(set(receipt_counts) == set(SPLITS), "source receipt split set drift")
    for split in SPLITS:
        split_receipt = _mapping(
            receipt_counts.get(split), label=f"source receipt {split}"
        )
        _require(
            set(split_receipt)
            <= {
                "PASS",
                "SOURCE_QUALITY_REJECT",
                "SOURCE_AUDIT_CONTRACT_REJECT",
            }
            and all(
                isinstance(value, int) and value >= 0
                for value in split_receipt.values()
            ),
            f"source receipt status drift: {split}",
        )
        split_admitted = int(split_receipt.get("PASS", 0))
        split_rejected = int(split_receipt.get("SOURCE_QUALITY_REJECT", 0))
        split_rejected += int(split_receipt.get("SOURCE_AUDIT_CONTRACT_REJECT", 0))
        _require(
            split_admitted + split_rejected == expected_split_counts[split],
            f"source receipt population drift: {split}",
        )
        admitted += split_admitted
        rejected += split_rejected
    _require(
        admitted + rejected == source_total,
        "source admission receipt does not partition source",
    )

    train_meetings = {row.meeting_date for row in source_by_split["train"]}
    heldout_meetings = {
        row.meeting_date
        for split in ("validation", "test")
        for row in source_by_split[split]
    }
    _require(
        train_meetings.isdisjoint(heldout_meetings), "source meeting isolation drift"
    )
    for split in SPLITS:
        _require(
            len({row.meeting_date for row in source_by_split[split]})
            == expected_meeting_counts[split],
            f"source meeting-count mismatch: {split}",
        )
    all_source_rows = [row for split in SPLITS for row in source_by_split[split]]
    reference_bank = _load_official_reference_bank(
        acquisition_paths["official_reference_bank"],
        all_source_rows,
        roster_path=official_roster_path,
        expected_roster_sha256=expected_official_roster_sha256,
        expected_meeting_counts=expected_meeting_counts,
        expected_boundary_method_counts=expected_boundary_method_counts,
        exception_boundaries=official_reference_exception_boundaries,
    )
    reference_file_sha256 = sha256_file(acquisition_paths["official_reference_bank"])
    _require(
        prepare_manifest.get("official_reference_bank_sha256") == reference_file_sha256
        and prepare_manifest.get("official_roster_sha256")
        == expected_official_roster_sha256
        and final_summary.get("official_reference_bank_sha256")
        == reference_file_sha256,
        "official reference preparation/final binding drift",
    )

    release = release_root.resolve()
    if release.exists():
        return verify_release(release, expected_manifest_sha256=None)
    _require(not release.is_symlink(), f"unsafe release path: {release}")
    release.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{release.name}.", dir=release.parent))
    try:
        pass_ids: list[str] = []
        reject_ids: list[str] = []
        pass_by_split: dict[str, list[str]] = {}
        reject_by_split: dict[str, list[str]] = {}
        split_counts: dict[str, int] = {}
        rejection_counts: dict[str, int] = {}
        rejection_reasons: Counter[str] = Counter()
        rejection_statuses: Counter[str] = Counter()
        repair_counts: Counter[str] = Counter()
        token_values: dict[str, list[int]] = {
            name: []
            for name in ("prompt", "reasoning", "answer", "completion", "total")
        }
        style_means: list[float] = []
        style_mins: list[int] = []
        all_style_means: list[float] = []
        all_style_mins: list[int] = []
        release_rejections: list[dict[str, Any]] = []
        source_audits: list[dict[str, Any]] = []
        validator_b_audits: list[dict[str, Any]] = []
        acquisition_rejection_projection: list[dict[str, Any]] = []
        acquisition_validator_b_projection: list[dict[str, Any]] = []
        acquisition_repair_projection: list[dict[str, Any]] = []
        terminal_status_counts: dict[str, dict[str, int]] = {}
        provider_audit = _new_provider_audit()
        provider_audit["cache_raw_content"] = provider_cache_raw
        provider_audit["cache_raw_reasoning"] = provider_cache_reasoning
        provider_audit["cache_records"] = provider_cache_records
        provider_audit["official_reference_bank_sha256"] = reference_file_sha256
        terminal_replay_identity = generator.ProviderIdentityRegistry()
        terminal_replay_config = generator.ProviderConfig()
        terminal_replay_code_sha256 = generator._implementation_contract()[
            "composite_sha256"
        ]
        validator_b_by_split: dict[str, Counter[str]] = {
            split: Counter() for split in SPLITS
        }

        for split in SPLITS:
            sources = source_by_split[split]
            terminals = terminal_by_split[split]
            _require(
                len(terminals) == len(sources),
                f"terminal does not cover source population: {split}",
            )
            acquisition_candidates = _read_jsonl(
                acquisition_paths[f"{split}.sft_candidate"],
                label=f"acquisition {split} candidates",
            )
            acquisition_manifests = _read_jsonl(
                acquisition_paths[f"{split}.manifest"],
                label=f"acquisition {split} manifests",
            )
            output_rows: list[dict[str, str]] = []
            sidecars: list[dict[str, Any]] = []
            split_pass: list[str] = []
            split_reject: list[str] = []
            candidate_index = 0
            for source, terminal in zip(sources, terminals):
                label = f"terminal {split}:{source.source_index}"
                provider_audit["current_sample_binding"] = {
                    "sample_id": source.sample_id,
                    "split": split,
                    "source_analysis_sha256": source.source_analysis_sha256,
                    "provided_data_sha256": source.provided_data_sha256,
                    "code_sha256": generator._implementation_contract()[
                        "composite_sha256"
                    ],
                    "provider_contract_sha256": (
                        generator.ProviderConfig().contract_sha256
                    ),
                }
                try:
                    official_reference = reference_bank.reference_for_meeting(
                        source.meeting_date
                    )
                except OfficialReferenceError as exc:
                    raise PublicationError(
                        f"{label} reference lookup failed: {exc}"
                    ) from exc
                _require(
                    official_reference.split == split,
                    f"{label} official reference split drift",
                )
                official_payload = {
                    "meeting_date": official_reference.meeting_date,
                    "paragraphs": [
                        {
                            "paragraph_id": paragraph.paragraph_id,
                            "section_name": paragraph.section_name,
                            "text": paragraph.text,
                        }
                        for paragraph in official_reference.paragraphs
                    ],
                }
                _require(
                    set(terminal) == TERMINAL_FIELDS, f"{label} field set mismatch"
                )
                secret_path = _contains_secret(terminal)
                _require(
                    secret_path is None,
                    f"terminal contains credential material at {secret_path}",
                )
                _require(
                    terminal.get("schema_version") == TERMINAL_SCHEMA_VERSION,
                    f"{label} schema version drift",
                )
                _require(
                    terminal.get("sample_id") == source.sample_id
                    and terminal.get("split") == split
                    and terminal.get("source_split") == source.source_split
                    and terminal.get("source_index") == source.source_index
                    and terminal.get("meeting_date") == source.meeting_date
                    and terminal.get("atomic_topic") == source.atomic_topic
                    and terminal.get("section_style_id") == source.section_style_id,
                    f"{label} source identity/order drift",
                )
                analysis = _string(
                    terminal.get("source_analysis"), label=f"{label}.source_analysis"
                )
                _require(
                    analysis == source.source_analysis,
                    f"{label} source analysis is not exact chk1 final answer",
                )
                _require(
                    terminal.get("source_analysis_sha256")
                    == source.source_analysis_sha256,
                    f"{label} source analysis hash drift",
                )
                _require(
                    terminal.get("provided_data_sha256") == source.provided_data_sha256,
                    f"{label} provided_data binding drift",
                )
                terminal_lineage = _validate_required_lineage(
                    terminal.get("lineage"), label=f"{label}.lineage"
                )
                for key, value in contract_lineage.items():
                    _require(
                        terminal_lineage.get(key) == value,
                        f"{label} lineage differs from prompt contract: {key}",
                    )

                source_pass = _validate_source_audit_record(
                    terminal.get("source_audit"),
                    label=f"{label}.source_audit",
                    source_analysis=source.source_analysis,
                    provided_data=source.provided_data,
                    provider_audit=provider_audit,
                )
                source_audits.append(
                    {
                        "schema_version": RELEASE_SCHEMA_VERSION,
                        "sample_id": source.sample_id,
                        "split": split,
                        "source_index": source.source_index,
                        "source_analysis_sha256": source.source_analysis_sha256,
                        "provided_data_sha256": source.provided_data_sha256,
                        "admitted": source_pass,
                        "audit": dict(
                            _mapping(
                                terminal.get("source_audit"),
                                label=f"{label}.source_audit",
                            )
                        ),
                    }
                )

                generation = _mapping(
                    terminal.get("generation"), label=f"{label}.generation"
                )
                if not source_pass:
                    _require(
                        generation == {}
                        and terminal.get("validator_a") == {}
                        and terminal.get("validator_b") == {},
                        f"{label} rejected source must not run downstream stages",
                    )
                    _require(
                        terminal.get("repair_history") == [],
                        f"{label} rejected source has repair history",
                    )
                    fidelity_repair = False
                    style_repair = False
                    deterministic: Mapping[str, Any] = {}
                    det_pass = False
                    a_complete = a_pass = False
                    b_complete = b_pass = False
                    b_mean = None
                    b_min = None
                else:
                    _require(
                        set(generation) == GENERATION_FIELDS,
                        f"{label}.generation field set mismatch",
                    )
                    fidelity_repair = generation.get("fidelity_repair_used")
                    style_repair = generation.get("style_repair_used")
                    _require(
                        isinstance(fidelity_repair, bool)
                        and isinstance(style_repair, bool),
                        f"{label} repair flags must be boolean",
                    )
                    deterministic = _mapping(
                        generation.get("deterministic_validation"),
                        label=f"{label}.deterministic",
                    )
                    _require(
                        set(deterministic) == DETERMINISTIC_FIELDS,
                        f"{label}.deterministic field set mismatch",
                    )
                    det_pass = deterministic.get("machine_pass")
                    _require(
                        isinstance(det_pass, bool),
                        f"{label}.deterministic.machine_pass must be boolean",
                    )
                    _string_list(
                        deterministic.get("reasons"),
                        label=f"{label}.deterministic.reasons",
                    )
                    _mapping(
                        deterministic.get("diagnostics"),
                        label=f"{label}.deterministic.diagnostics",
                    )
                    selected_attempt = generation.get("selected_attempt")
                    _require(
                        selected_attempt
                        in {"primary", "fidelity_repair", "style_repair"},
                        f"{label} selected attempt drift",
                    )
                    expected_generation_roles = {generator.ROLE_REWRITE_PRIMARY}
                    if fidelity_repair:
                        expected_generation_roles.add(
                            generator.ROLE_REWRITE_FIDELITY_REPAIR
                        )
                    if style_repair:
                        expected_generation_roles.add(
                            generator.ROLE_REWRITE_STYLE_REPAIR
                        )
                    generation_provider = _mapping(
                        generation.get("provider"),
                        label=f"{label}.generation.provider",
                    )
                    _require(
                        set(generation_provider) == expected_generation_roles,
                        f"{label} generation provider role sequence drift",
                    )
                    generation_hashes = {
                        generator.ROLE_REWRITE_PRIMARY: {
                            "analysis": _projection_hash(source.source_analysis)
                        }
                    }
                    if fidelity_repair:
                        generation_hashes[generator.ROLE_REWRITE_FIDELITY_REPAIR] = {
                            "analysis": _projection_hash(source.source_analysis)
                        }
                    if style_repair:
                        generation_hashes[generator.ROLE_REWRITE_STYLE_REPAIR] = {
                            "source_analysis": _projection_hash(source.source_analysis)
                        }
                    _provider_complete(
                        generation.get("provider"),
                        label=f"{label}.generation.provider",
                        provider_audit=provider_audit,
                        expected_field_hashes=generation_hashes,
                    )
                    _require(
                        not fidelity_repair
                        or selected_attempt in {"fidelity_repair", "style_repair"},
                        f"{label} fidelity repair flag drift",
                    )
                    _require(
                        not style_repair or selected_attempt == "style_repair",
                        f"{label} style repair flag drift",
                    )
                    repair_events = _validate_repair_history(
                        terminal.get("repair_history"),
                        label=f"{label}.repair_history",
                        fidelity_repair_used=fidelity_repair,
                        style_repair_used=style_repair,
                        forbidden_reference_texts=[
                            paragraph.text
                            for paragraph in official_reference.paragraphs
                        ],
                        forbidden_reference_ids=[
                            paragraph.paragraph_id
                            for paragraph in official_reference.paragraphs
                        ],
                        provider_audit=provider_audit,
                        source_analysis=source.source_analysis,
                        generation_provider=generation_provider,
                        official_reference=official_reference,
                    )
                    (
                        history_validator_a_roles,
                        history_validator_b_roles,
                    ) = _validator_roles_from_repair_history(repair_events)
                    if style_repair:
                        style_event = next(
                            event
                            for event in repair_events
                            if event.get("repair_type") == "style"
                        )
                        safe_feedback = _mapping(
                            _mapping(
                                style_event.get("validator_b"),
                                label=f"{label}.style_event.validator_b",
                            ).get("style_feedback"),
                            label=f"{label}.style_event.style_feedback",
                        )
                        style_provider = _mapping(
                            _mapping(
                                generation.get("provider"),
                                label=f"{label}.generation.provider",
                            ).get(generator.ROLE_REWRITE_STYLE_REPAIR),
                            label=f"{label}.rewrite_style_repair.provider",
                        )
                        style_hashes = _mapping(
                            _mapping(
                                style_provider.get("request_projection"),
                                label=f"{label}.rewrite_style_repair.projection",
                            ).get("field_value_sha256"),
                            label=f"{label}.rewrite_style_repair.field_hashes",
                        )
                        primary_b_provider = _mapping(
                            _mapping(
                                style_event.get("provider"),
                                label=f"{label}.style_event.provider",
                            ).get(generator.ROLE_VALIDATOR_B_PRIMARY),
                            label=f"{label}.style_event.validator_b_primary",
                        )
                        primary_b_hashes = _mapping(
                            _mapping(
                                primary_b_provider.get("request_projection"),
                                label=f"{label}.style_event.b_projection",
                            ).get("field_value_sha256"),
                            label=f"{label}.style_event.b_field_hashes",
                        )
                        pre_repair_answer = _validate_style_repair_trigger_snapshot(
                            event=style_event,
                            generation_provider=_mapping(
                                generation.get("provider"),
                                label=f"{label}.generation.provider",
                            ),
                            reference=official_reference,
                            official_payload=official_payload,
                            provider_audit=provider_audit,
                            label=f"{label}.style_repair_trigger",
                        )
                        _require(
                            style_hashes.get("source_analysis")
                            == _projection_hash(source.source_analysis)
                            and style_hashes.get("style_feedback")
                            == _projection_hash(safe_feedback)
                            and style_hashes.get("current_rewritten_minutes")
                            == primary_b_hashes.get("rewritten_minutes")
                            == _projection_hash(pre_repair_answer),
                            f"{label} style repair request is not the safe Validator-B projection",
                        )
                    if repair_events:
                        acquisition_repair_projection.append(
                            {
                                "sample_id": source.sample_id,
                                "split": split,
                                "terminal_status": terminal.get("terminal_status"),
                                "events": [dict(event) for event in repair_events],
                            }
                        )
                    if fidelity_repair:
                        repair_counts["fidelity_repair"] += 1
                    if style_repair:
                        repair_counts["style_repair"] += 1
                    if not fidelity_repair and not style_repair:
                        repair_counts["primary_no_repair"] += 1

                    reasoning_value = terminal.get("teacher_response_analysis")
                    answer_value = terminal.get("rewritten_minutes")
                    candidate_reasoning = (
                        reasoning_value
                        if isinstance(reasoning_value, str) and reasoning_value
                        else None
                    )
                    candidate_answer = (
                        answer_value
                        if isinstance(answer_value, str) and answer_value
                        else None
                    )
                    if det_pass:
                        _require(
                            candidate_reasoning is not None
                            and candidate_answer is not None,
                            f"{label} deterministic PASS lacks candidate text",
                        )
                    det_pass = _validate_deterministic_record(
                        deterministic,
                        label=f"{label}.deterministic",
                        source_analysis=source.source_analysis,
                        reasoning=candidate_reasoning if det_pass else None,
                        answer=candidate_answer if det_pass else None,
                    )
                    if det_pass:
                        (
                            cached_candidate_reasoning,
                            cached_candidate_answer,
                        ) = _candidate_from_generation_cache(
                            attempt=str(selected_attempt),
                            generation_provider=generation_provider,
                            provider_audit=provider_audit,
                            label=f"{label}.selected_generation_candidate",
                        )
                        _require(
                            candidate_reasoning == cached_candidate_reasoning
                            and candidate_answer == cached_candidate_answer,
                            f"{label} terminal candidate differs from selected rewrite cache",
                        )

                    if terminal.get("validator_a") == {}:
                        a_complete = a_pass = False
                    else:
                        if candidate_reasoning is None or candidate_answer is None:
                            a_complete, a_pass, _ = _machine_record(
                                terminal.get("validator_a"),
                                label=f"{label}.validator_a",
                                fields=AUDIT_FIELDS,
                            )
                            a_provider = _mapping(
                                _mapping(
                                    terminal.get("validator_a"),
                                    label=f"{label}.validator_a",
                                ).get("provider"),
                                label=f"{label}.validator_a.provider",
                            )
                            _provider_complete(
                                a_provider,
                                label=f"{label}.validator_a.provider",
                                provider_audit=provider_audit,
                            )
                        else:
                            if selected_attempt == "style_repair" and det_pass:
                                selected_a_role = (
                                    generator.ROLE_VALIDATOR_A_STYLE_REPAIR
                                )
                            elif selected_attempt == "fidelity_repair" and det_pass:
                                selected_a_role = (
                                    generator.ROLE_VALIDATOR_A_FIDELITY_REPAIR
                                )
                            elif selected_attempt == "style_repair" and fidelity_repair:
                                selected_a_role = (
                                    generator.ROLE_VALIDATOR_A_FIDELITY_REPAIR
                                )
                            else:
                                selected_a_role = generator.ROLE_VALIDATOR_A_PRIMARY
                            a_pass, _ = _validate_validator_a_record(
                                terminal.get("validator_a"),
                                label=f"{label}.validator_a",
                                source_analysis=source.source_analysis,
                                reasoning=candidate_reasoning,
                                answer=candidate_answer,
                                provider_audit=provider_audit,
                                expected_selected_role=selected_a_role,
                                expected_provider_roles=(
                                    history_validator_a_roles
                                    | _expected_validator_invocation_roles(
                                        result=_mapping(
                                            _mapping(
                                                terminal.get("validator_a"),
                                                label=f"{label}.validator_a",
                                            ).get("result"),
                                            label=f"{label}.validator_a.result",
                                        ),
                                        validator="validator_a",
                                        base_role=selected_a_role,
                                        label=f"{label}.validator_a.selected",
                                    )
                                ),
                            )
                            a_complete = True
                    answer_for_b = (
                        answer_value if isinstance(answer_value, str) else None
                    )
                    if terminal.get("validator_b") == {}:
                        b_complete = b_pass = False
                        b_mean = None
                        b_min = None
                        b_comparison_status = None
                        b_has_critical = False
                    else:
                        if style_repair and det_pass and a_pass:
                            selected_b_role = generator.ROLE_VALIDATOR_B_STYLE_REPAIR
                        else:
                            selected_b_role = generator.ROLE_VALIDATOR_B_PRIMARY
                        b_hashes = {
                            selected_b_role: {
                                "corresponding_official_minutes_pre_action": (
                                    _projection_hash(official_payload)
                                )
                            }
                        }
                        if not (style_repair and det_pass and not a_pass):
                            b_hashes[selected_b_role]["rewritten_minutes"] = (
                                _projection_hash(answer_for_b)
                            )
                        (
                            b_complete,
                            b_pass,
                            b_mean,
                            b_min,
                            b_comparison_status,
                        ) = _validate_validator_b(
                            terminal.get("validator_b"),
                            label=f"{label}.validator_b",
                            answer=answer_for_b,
                            reference=official_reference,
                            provider_audit=provider_audit,
                            expected_field_hashes=b_hashes,
                            expected_selected_role=selected_b_role,
                            expected_provider_roles=(
                                history_validator_b_roles
                                | _expected_validator_invocation_roles(
                                    result=_mapping(
                                        _mapping(
                                            terminal.get("validator_b"),
                                            label=f"{label}.validator_b",
                                        ).get("result"),
                                        label=f"{label}.validator_b.result",
                                    ),
                                    validator="validator_b",
                                    base_role=selected_b_role,
                                    label=f"{label}.validator_b.selected",
                                )
                            ),
                        )
                        b_has_critical = bool(
                            _mapping(
                                _mapping(
                                    terminal.get("validator_b"),
                                    label=f"{label}.validator_b",
                                ).get("result"),
                                label=f"{label}.validator_b.result",
                            ).get("critical_style_errors")
                        )

                source_contract_exhausted = bool(
                    isinstance(
                        terminal.get("source_audit", {})
                        .get("result", {})
                        .get("contract_repair"),
                        Mapping,
                    )
                    and terminal["source_audit"]["result"]["contract_repair"].get(
                        "contract_exhausted"
                    )
                    is True
                )
                a_contract_exhausted = bool(
                    terminal.get("validator_a", {})
                    .get("result", {})
                    .get("contract_exhausted")
                    is True
                )
                b_contract_exhausted = bool(
                    terminal.get("validator_b", {})
                    .get("result", {})
                    .get("contract_exhausted")
                    is True
                )
                if not source_pass:
                    expected_status = (
                        "SOURCE_AUDIT_CONTRACT_REJECT"
                        if source_contract_exhausted
                        else "SOURCE_QUALITY_REJECT"
                    )
                    _require(
                        not a_complete and not b_complete,
                        f"{label} downstream validators ran after source rejection",
                    )
                elif not det_pass:
                    if style_repair:
                        det_reasons = _string_list(
                            deterministic.get("reasons"),
                            label=f"{label}.deterministic.reasons",
                            allow_empty=False,
                        )
                        fidelity_failure = any(
                            reason.startswith(
                                (
                                    "numeric_multiset_mismatch",
                                    "date_set_mismatch",
                                    "attribution_set_mismatch",
                                )
                            )
                            for reason in det_reasons
                        )
                        expected_status = (
                            "STYLE_REPAIR_FIDELITY_REJECT"
                            if fidelity_failure
                            else "STYLE_QUALITY_REJECT"
                        )
                        expected_stage = (
                            "style_repair_deterministic_fidelity"
                            if fidelity_failure
                            else "style_repair_deterministic_style"
                        )
                        _require(
                            terminal.get("rejection_stage") == expected_stage
                            and a_complete
                            and a_pass
                            and b_complete
                            and not b_pass,
                            f"{label} style-repair deterministic failure lost prior gates",
                        )
                    else:
                        expected_status = "GENERATION_QUALITY_REJECT"
                        _require(
                            not b_complete,
                            f"{label} validator B ran after deterministic failure",
                        )
                        if (
                            terminal.get("rejection_stage")
                            == "generation_deterministic_gate"
                        ):
                            _require(
                                not a_complete,
                                f"{label} validator A ran before initial generation passed",
                            )
                        elif (
                            terminal.get("rejection_stage")
                            == "fidelity_repair_deterministic_gate"
                        ):
                            _require(
                                a_complete and not a_pass and fidelity_repair,
                                f"{label} lost the initial validator-A failure before fidelity repair",
                            )
                        else:
                            raise PublicationError(
                                f"{label} deterministic rejection stage drift"
                            )
                elif a_contract_exhausted:
                    expected_status = "VALIDATOR_A_CONTRACT_REJECT"
                    _require(
                        a_complete and not b_complete,
                        f"{label} Validator-A contract reject sequencing drift",
                    )
                elif not a_pass:
                    expected_status = (
                        "STYLE_REPAIR_FIDELITY_REJECT"
                        if style_repair
                        else "INPUT_FIDELITY_REJECT"
                    )
                    _require(
                        a_complete and not b_complete,
                        f"{label} validator sequencing drift",
                    )
                elif b_contract_exhausted:
                    expected_status = "VALIDATOR_B_CONTRACT_REJECT"
                    _require(
                        a_complete and b_complete and not b_pass,
                        f"{label} Validator-B contract reject sequencing drift",
                    )
                elif b_comparison_status == "no_comparable_passage":
                    expected_status = "REFERENCE_COMPARISON_UNAVAILABLE_REJECT"
                    _require(
                        a_complete
                        and b_complete
                        and not b_pass
                        and not style_repair
                        and terminal.get("rejection_stage")
                        == "validator_b_reference_comparison",
                        f"{label} no-comparable result must reject without repair",
                    )
                elif b_has_critical:
                    expected_status = "STYLE_QUALITY_REJECT"
                    _require(
                        a_complete
                        and b_complete
                        and not b_pass
                        and not style_repair
                        and terminal.get("rejection_stage")
                        == "validator_b_critical_style_error",
                        f"{label} critical Validator-B error must reject without repair",
                    )
                elif not b_pass:
                    expected_status = "STYLE_QUALITY_REJECT"
                    _require(
                        a_complete and b_complete,
                        f"{label} style rejection is incomplete",
                    )
                else:
                    expected_status = "PASS"
                    _require(
                        a_complete and b_complete, f"{label} PASS validators incomplete"
                    )
                status = terminal.get("terminal_status")
                _require(
                    status in TERMINAL_STATUSES and status == expected_status,
                    f"{label} terminal/status consistency failure",
                )
                training_pass = terminal.get("training_pass")
                eligible = source_pass and det_pass and a_pass and b_pass
                _require(
                    isinstance(training_pass, bool)
                    and training_pass == eligible == (status == "PASS"),
                    f"{label} PASS eligibility drift",
                )
                reasons = _string_list(
                    terminal.get("rejection_reasons"),
                    label=f"{label}.rejection_reasons",
                    allow_empty=status == "PASS",
                )
                if status == "PASS":
                    _require(
                        terminal.get("rejection_stage") is None and not reasons,
                        f"{label} PASS has rejection metadata",
                    )
                else:
                    _string(
                        terminal.get("rejection_stage"),
                        label=f"{label}.rejection_stage",
                    )
                    _require(
                        terminal.get("student_prompt") == ""
                        and terminal.get("sft_response") == ""
                        and terminal.get("prompt_sha256") is None
                        and terminal.get("response_sha256") is None,
                        f"{label} rejected row contains a student training record",
                    )
                    if status in {
                        "SOURCE_QUALITY_REJECT",
                        "SOURCE_AUDIT_CONTRACT_REJECT",
                        "GENERATION_QUALITY_REJECT",
                    }:
                        _require(
                            terminal.get("teacher_response_analysis") == ""
                            and terminal.get("rewritten_minutes") == ""
                            and terminal.get("teacher_response_analysis_sha256") is None
                            and terminal.get("rewritten_minutes_sha256") is None,
                            f"{label} pre-candidate rejection contains candidate text",
                        )
                    else:
                        rejected_reasoning = _string(
                            terminal.get("teacher_response_analysis"),
                            label=f"{label}.teacher_response_analysis",
                        )
                        rejected_answer = _string(
                            terminal.get("rewritten_minutes"),
                            label=f"{label}.rewritten_minutes",
                        )
                        _require(
                            terminal.get("teacher_response_analysis_sha256")
                            == sha256_text(rejected_reasoning)
                            and terminal.get("rewritten_minutes_sha256")
                            == sha256_text(rejected_answer),
                            f"{label} rejected candidate hash drift",
                        )

                # Finally replay the generator's immutable resume protocol.  The
                # publisher's independent semantic checks above retain precise
                # failure diagnostics; this second seal proves every referenced
                # cache lives at its canonical role/sample path and that its
                # complete system prompt, user prompt, projection, binding, raw
                # response, and terminal receipt all reconstruct exactly.
                try:
                    replayed_terminal = generator._load_terminal(
                        acquisition_root.resolve(),
                        _prepared_row_for_terminal_replay(source),
                        code_sha256=terminal_replay_code_sha256,
                        official_reference_bank_sha256=reference_file_sha256,
                        identity=terminal_replay_identity,
                        reference_bank=reference_bank,
                        config=terminal_replay_config,
                        environment={},
                    )
                except (
                    generator.SyntheticRewriteError,
                    OSError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as exc:
                    raise PublicationError(
                        f"{label} generator terminal/cache replay failed: {exc}"
                    ) from exc
                _require(
                    replayed_terminal is not None
                    and canonical_json(replayed_terminal) == canonical_json(terminal),
                    f"{label} terminal JSONL differs from immutable terminal cache",
                )

                if b_complete:
                    validator_b_by_split[split]["final_complete"] += 1
                    if b_contract_exhausted:
                        validator_b_by_split[split]["final_contract_reject"] += 1
                    else:
                        assert b_mean is not None and b_min is not None
                        all_style_means.append(b_mean)
                        all_style_mins.append(b_min)
                        validator_b_by_split[split][
                            "final_pass" if b_pass else "final_low"
                        ] += 1
                    if style_repair:
                        validator_b_by_split[split]["style_repair_attempted"] += 1
                    if status == "STYLE_QUALITY_REJECT":
                        validator_b_by_split[split]["final_style_reject"] += 1
                    terminal_b_provider = _mapping(
                        _mapping(
                            terminal.get("validator_b"),
                            label=f"{label}.validator_b",
                        ).get("provider"),
                        label=f"{label}.validator_b.provider",
                    )
                    final_b_is_style_repair = (
                        generator.ROLE_VALIDATOR_B_STYLE_REPAIR in terminal_b_provider
                    )
                    for event in repair_events:
                        if event.get("repair_type") != "style":
                            continue
                        initial = _mapping(
                            event.get("validator_b"),
                            label=f"{label}.repair_history.initial_validator_b",
                        )
                        initial_mean = float(initial["mean_score"])
                        initial_min = int(initial["min_score"])
                        if final_b_is_style_repair:
                            all_style_means.append(initial_mean)
                            all_style_mins.append(initial_min)
                        validator_b_by_split[split]["primary_low"] += 1
                    validator_b_audits.append(
                        {
                            "schema_version": RELEASE_SCHEMA_VERSION,
                            "sample_id": source.sample_id,
                            "split": split,
                            "source_index": source.source_index,
                            "mean_score": b_mean,
                            "min_score": b_min,
                            "machine_pass": b_pass,
                            "used_for_selection": True,
                            "triggered_style_repair": style_repair,
                            "validator": dict(
                                _mapping(
                                    terminal.get("validator_b"),
                                    label=f"{label}.validator_b",
                                )
                            ),
                        }
                    )
                    acquisition_validator_b_projection.append(
                        {
                            "sample_id": source.sample_id,
                            "split": split,
                            "terminal_status": status,
                            "validator_b": dict(
                                _mapping(
                                    terminal.get("validator_b"),
                                    label=f"{label}.validator_b",
                                )
                            ),
                        }
                    )

                if status != "PASS":
                    acquisition_rejection_projection.append(dict(terminal))
                    split_reject.append(source.sample_id)
                    reject_ids.append(source.sample_id)
                    rejection_statuses[status] += 1
                    for reason in reasons:
                        rejection_reasons[reason] += 1
                    release_rejections.append(
                        {
                            "schema_version": RELEASE_SCHEMA_VERSION,
                            "sample_id": source.sample_id,
                            "split": split,
                            "source_index": source.source_index,
                            "meeting_date": source.meeting_date,
                            "terminal_status": status,
                            "rejection_stage": terminal.get("rejection_stage"),
                            "rejection_reasons": reasons,
                            "fidelity_repair_used": fidelity_repair,
                            "style_repair_used": style_repair,
                            "terminal_record_sha256": sha256_text(
                                canonical_json(terminal)
                            ),
                        }
                    )
                    continue

                reasoning = _string(
                    terminal.get("teacher_response_analysis"),
                    label=f"{label}.teacher_response_analysis",
                )
                answer = _string(
                    terminal.get("rewritten_minutes"),
                    label=f"{label}.rewritten_minutes",
                )
                prompt = _string(
                    terminal.get("student_prompt"), label=f"{label}.student_prompt"
                )
                response = _string(
                    terminal.get("sft_response"), label=f"{label}.sft_response"
                )
                _require(
                    prompt == source.student_prompt,
                    f"{label} student prompt does not contain exact source analysis",
                )
                _require(
                    response == reasoning + BOUNDARY + answer,
                    f"{label} response assembly drift",
                )
                hashes = {
                    "teacher_response_analysis_sha256": sha256_text(reasoning),
                    "rewritten_minutes_sha256": sha256_text(answer),
                    "prompt_sha256": sha256_text(prompt),
                    "response_sha256": sha256_text(response),
                }
                for key, expected in hashes.items():
                    _require(terminal.get(key) == expected, f"{label}.{key} drift")
                tokens = _token_replay(
                    tokenizer=tokenizer,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    response=response,
                    sample_id=source.sample_id,
                )
                diagnostics = _mapping(
                    deterministic.get("diagnostics"),
                    label=f"{label}.deterministic.diagnostics",
                )
                cached_replay = _mapping(
                    diagnostics.get("tokenizer_replay"),
                    label=f"{label}.deterministic.tokenizer_replay",
                )
                _require(
                    cached_replay
                    == {
                        "prompt_tokens": tokens["prompt_tokens"],
                        "completion_tokens": tokens["completion_tokens"],
                        "total_tokens": tokens["total_tokens"],
                        "single_bos": True,
                        "single_eos": True,
                        "completion_only_prompt_masked": True,
                        "completion_mask_covers_reasoning_boundary_answer_eos": True,
                        "no_truncation": True,
                    }
                    and diagnostics.get("reasoning_words")
                    == len(_WORD_RE.findall(reasoning))
                    and diagnostics.get("rewritten_minutes_words")
                    == len(_WORD_RE.findall(answer))
                    and diagnostics.get("prompt_tokens") == tokens["prompt_tokens"]
                    and diagnostics.get("completion_tokens")
                    == tokens["completion_tokens"]
                    and diagnostics.get("total_tokens") == tokens["total_tokens"]
                    and diagnostics.get("total_token_limit") == MAX_TOTAL_TOKENS
                    and diagnostics.get("truncation") is False
                    and diagnostics.get("source_numbers")
                    == dict(sorted(_numeric_values(source.source_analysis).items()))
                    and diagnostics.get("rewrite_numbers")
                    == dict(sorted(_numeric_values(answer).items()))
                    and diagnostics.get("source_dates")
                    == sorted(_date_values(source.source_analysis))
                    and diagnostics.get("rewrite_dates") == sorted(_date_values(answer))
                    and diagnostics.get("source_attributions")
                    == sorted(_attribution_categories(source.source_analysis))
                    and diagnostics.get("rewrite_attributions")
                    == sorted(_attribution_categories(answer)),
                    f"{label} cached tokenizer/deterministic diagnostics drift",
                )
                for key in token_values:
                    token_values[key].append(tokens[f"{key}_tokens"])
                assert b_mean is not None and b_min is not None
                style_means.append(b_mean)
                style_mins.append(b_min)

                output = {"prompt": prompt, "response": response}
                _require(
                    candidate_index < len(acquisition_candidates),
                    f"missing acquisition PASS candidate: {source.sample_id}",
                )
                _require(
                    acquisition_candidates[candidate_index] == output,
                    f"acquisition PASS candidate drift: {source.sample_id}",
                )
                acquisition_manifest = acquisition_manifests[candidate_index]
                _require(
                    acquisition_manifest.get("sample_id") == source.sample_id,
                    f"acquisition PASS manifest order drift: {source.sample_id}",
                )
                for key, expected in {
                    "source_analysis_sha256": source.source_analysis_sha256,
                    "prompt_sha256": sha256_text(prompt),
                    "response_sha256": sha256_text(response),
                }.items():
                    if key in acquisition_manifest:
                        _require(
                            acquisition_manifest.get(key) == expected,
                            f"acquisition manifest {key} drift: {source.sample_id}",
                        )
                candidate_index += 1

                release_index = len(output_rows)
                output_rows.append(output)
                split_pass.append(source.sample_id)
                pass_ids.append(source.sample_id)
                sidecars.append(
                    {
                        "schema_version": RELEASE_SCHEMA_VERSION,
                        "sample_id": source.sample_id,
                        "split": split,
                        "source_split": source.source_split,
                        "release_index": release_index,
                        "source_index": source.source_index,
                        "meeting_date": source.meeting_date,
                        "atomic_topic": source.atomic_topic,
                        "section_style_id": source.section_style_id,
                        "provided_data_sha256": source.provided_data_sha256,
                        "source_analysis_sha256": source.source_analysis_sha256,
                        "prompt_sha256": sha256_text(prompt),
                        "teacher_response_analysis_sha256": sha256_text(reasoning),
                        "rewritten_minutes_sha256": sha256_text(answer),
                        "response_sha256": sha256_text(response),
                        "reasoning_transformation": "none",
                        "source_audit_record_sha256": sha256_text(
                            canonical_json(terminal.get("source_audit"))
                        ),
                        "generation_record_sha256": sha256_text(
                            canonical_json(generation)
                        ),
                        "validator_a_record_sha256": sha256_text(
                            canonical_json(terminal.get("validator_a"))
                        ),
                        "validator_b": {
                            "comparison_status": b_comparison_status,
                            "mean_score": b_mean,
                            "min_score": b_min,
                            "machine_pass": True,
                            "used_for_selection": True,
                            "style_repair_used": style_repair,
                            "record_sha256": sha256_text(
                                canonical_json(terminal.get("validator_b"))
                            ),
                        },
                        "official_reference": {
                            "meeting_date": official_reference.meeting_date,
                            "meeting_reference_sha256": (
                                official_reference.meeting_reference_sha256
                            ),
                            "reference_bank_sha256": (
                                reference_bank.reference_bank_sha256
                            ),
                        },
                        "tokens": tokens,
                        "lineage": terminal_lineage,
                    }
                )
            _require(
                candidate_index
                == len(acquisition_candidates)
                == len(acquisition_manifests),
                f"acquisition PASS artifacts mismatch: {split}",
            )
            _require(bool(output_rows), f"PASS-only release split is empty: {split}")
            _write_jsonl(staging / f"minutes_alignment/{split}.jsonl", output_rows)
            _write_jsonl(
                staging / f"minutes_alignment/manifests/{split}.jsonl", sidecars
            )
            pass_by_split[split] = split_pass
            reject_by_split[split] = split_reject
            split_counts[split] = len(split_pass)
            rejection_counts[split] = len(split_reject)
            terminal_status_counts[split] = dict(
                Counter(str(row["terminal_status"]) for row in terminals)
            )

        source_ids = [
            row.sample_id for split in SPLITS for row in source_by_split[split]
        ]
        _require(set(pass_ids).isdisjoint(reject_ids), "PASS and REJECT overlap")
        _require(
            set(pass_ids) | set(reject_ids) == set(source_ids),
            "PASS/REJECT do not partition all 1,743 source IDs",
        )
        _require(
            len(pass_ids) + len(reject_ids) == source_total,
            "terminal partition count mismatch",
        )
        _require(
            sum(1 for row in source_audits if row["admitted"]) == admitted,
            "source receipt admitted count drift",
        )
        _require(
            sum(1 for row in source_audits if not row["admitted"]) == rejected,
            "source receipt rejected count drift",
        )
        _require(
            set(provider_cache_records) == set(provider_audit["receipt_keys"]),
            "provider cache/terminal receipt partition drift",
        )
        all_provider_identity = _provider_identity_projection(
            provider_audit, roles=set(provider_audit["roles"])
        )
        source_receipt_identity = _mapping(
            source_receipt.get("provider_identities"),
            label="source receipt provider identities",
        )
        receipt_roles = source_receipt_identity.get("roles_observed")
        _require(
            source_receipt_identity.get("returned_model")
            == all_provider_identity["returned_model"]
            and source_receipt_identity.get("system_fingerprint")
            == all_provider_identity["system_fingerprint"]
            and isinstance(receipt_roles, list)
            and receipt_roles == sorted(set(receipt_roles))
            and set(provider_audit["source_roles"]) <= set(receipt_roles)
            and set(receipt_roles) <= set(provider_audit["roles"]),
            "source receipt provider identity/role projection drift",
        )
        _require(
            final_summary.get("provider_identities") == all_provider_identity,
            "final summary provider identity/role projection drift",
        )
        _require(
            final_summary.get("total_source_rows") == source_total
            and final_summary.get("terminal_classified") == source_total
            and final_summary.get("status_counts") == terminal_status_counts
            and final_summary.get("training_ready_candidate") is True
            and final_summary.get("evaluation_eligible") is False,
            "final summary population/status contract drift",
        )
        _validate_required_lineage(
            final_summary.get("lineage"), label="final_summary.lineage"
        )
        _require(
            _read_jsonl(
                acquisition_paths["audits.rejections"],
                label="acquisition rejection audit",
            )
            == acquisition_rejection_projection,
            "acquisition rejection audit does not exactly match terminal rejects",
        )
        _require(
            _read_jsonl(
                acquisition_paths["audits.validator_b"],
                label="acquisition validator B audit",
            )
            == acquisition_validator_b_projection,
            "acquisition validator B audit does not exactly match terminal validators",
        )
        _require(
            _read_jsonl(
                acquisition_paths["audits.repair_history"],
                label="acquisition repair-history audit",
            )
            == acquisition_repair_projection,
            "acquisition repair-history audit does not exactly match terminals",
        )

        _write_jsonl(staging / "audits/rejections.jsonl", release_rejections)
        _write_jsonl(staging / "audits/source_audits.jsonl", source_audits)
        _write_jsonl(staging / "audits/validator_b.jsonl", validator_b_audits)
        _write_jsonl(
            staging / "audits/repair_history.jsonl", acquisition_repair_projection
        )
        shutil.copyfile(
            acquisition_root / "prompt_contract.json", staging / "prompt_contract.json"
        )
        shutil.copyfile(
            acquisition_paths["official_reference_bank"],
            staging / "official_pre_action_reference_bank.jsonl",
        )
        shutil.copyfile(
            acquisition_root / "prepare_manifest.json",
            staging / "prepare_manifest.json",
        )
        shutil.copyfile(
            acquisition_root / "source_admission_receipt.json",
            staging / "source_admission_receipt.json",
        )

        tokenizer_replay = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "status": "passed",
            "tokenizer": {
                "path": str(tokenizer_path.resolve()),
                "class": type(tokenizer).__name__,
                "bos_token": getattr(tokenizer, "bos_token", None),
                "bos_token_id": getattr(tokenizer, "bos_token_id", None),
                "eos_token": getattr(tokenizer, "eos_token", None),
                "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                "runtime_contract": tokenizer_runtime_contract,
            },
            "row_count": len(pass_ids),
            "contract": {
                "single_bos": True,
                "single_eos": True,
                "single_literal_close_think_boundary": True,
                "chat_template_supplies_opening_think": True,
                "completion_only_prompt_masked": True,
                "reasoning_boundary_answer_eos_unmasked": True,
                "truncation": False,
                "total_length_gate": True,
                "total_max": MAX_TOTAL_TOKENS,
            },
            "token_stats": {
                key: _stats(values) for key, values in token_values.items()
            },
        }
        _write_json(staging / "audits/tokenizer_replay.json", tokenizer_replay)

        coverage = {
            split: {
                "source": expected_split_counts[split],
                "pass": split_counts[split],
                "reject": rejection_counts[split],
                "pass_rate": round(
                    split_counts[split] / expected_split_counts[split], 8
                ),
            }
            for split in SPLITS
        }
        audit = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "status": "passed",
            "source_rows": source_total,
            "pass_rows": len(pass_ids),
            "reject_rows": len(reject_ids),
            "split_counts": split_counts,
            "rejection_counts": rejection_counts,
            "coverage": coverage,
            "coverage_threshold": None,
            "all_splits_nonempty": True,
            "unresolved_rows": 0,
            "selection_rule": "source audit AND deterministic gate AND validator A AND recomputed validator B style gate",
            "validator_b_used_for_selection": True,
            "validator_b_threshold": {
                "mean_at_least": STYLE_MEAN_THRESHOLD,
                "each_dimension_at_least": STYLE_MIN_THRESHOLD,
                "critical_style_errors": 0,
            },
            "validator_b_mean_stats": _stats(style_means),
            "validator_b_min_stats": _stats(style_mins),
            "validator_b_all_attempt_mean_stats": _stats(all_style_means),
            "validator_b_all_attempt_min_stats": _stats(all_style_mins),
            "validator_b_by_split": {
                split: {
                    **dict(sorted(validator_b_by_split[split].items())),
                    "primary_low_rate_of_source": round(
                        validator_b_by_split[split]["primary_low"]
                        / expected_split_counts[split],
                        8,
                    ),
                    "style_repair_rate_of_source": round(
                        validator_b_by_split[split]["style_repair_attempted"]
                        / expected_split_counts[split],
                        8,
                    ),
                    "final_style_reject_rate_of_source": round(
                        validator_b_by_split[split]["final_style_reject"]
                        / expected_split_counts[split],
                        8,
                    ),
                }
                for split in SPLITS
            },
            "repair_counts": dict(sorted(repair_counts.items())),
            "rejection_status_counts": dict(sorted(rejection_statuses.items())),
            "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
            "integrity": {
                "source_sample_id_sha256": _id_digest(source_ids),
                "pass_sample_id_sha256": _id_digest(pass_ids),
                "reject_sample_id_sha256": _id_digest(reject_ids),
                "split_pass_sample_id_sha256": {
                    split: _id_digest(pass_by_split[split]) for split in SPLITS
                },
                "split_reject_sample_id_sha256": {
                    split: _id_digest(reject_by_split[split]) for split in SPLITS
                },
                "pass_reject_disjoint": True,
                "pass_reject_exact_source_partition": True,
                "meeting_split_conflicts": 0,
            },
            "checks": {
                "exact_chk1_source_binding": "passed",
                "source_admission_partition": "passed",
                "source_audit_exact_evidence_recomputed": "passed",
                "deterministic_text_gates_recomputed": "passed",
                "validator_a_exact_evidence_recomputed": "passed",
                "terminal_status_consistency": "passed",
                "validator_order": "passed",
                "validator_b_style_gate_recomputed": "passed",
                "official_reference_rebuilt_from_sealed_sources": "passed",
                "corresponding_meeting_passage_evidence_recomputed": "passed",
                "style_repair_official_text_isolation": "passed",
                "legacy_v1_cache_absence": "passed",
                "provider_request_projections": "passed",
                "provider_model_fingerprint_fixed": "passed",
                "pass_only_schema": "passed",
                "credential_absence": "passed",
                "tokenizer_replay": "passed",
            },
            "provider_request_projection_audit": {
                "identity": all_provider_identity,
                "roles": sorted(provider_audit["roles"]),
                "receipt_count": len(provider_audit["receipt_keys"]),
                "all_calls_have_projection": True,
                "official_text_policy_enforced": True,
                "c8_fields_forbidden": True,
                "plaintext_credentials_persisted": False,
            },
            "tokenizer_runtime_contract": tokenizer_runtime_contract,
            "lineage": contract_lineage,
        }
        _write_json(staging / "audits/data_quality.json", audit)

        payload_paths = [
            "prompt_contract.json",
            "prepare_manifest.json",
            "official_pre_action_reference_bank.jsonl",
            "source_admission_receipt.json",
            "audits/data_quality.json",
            "audits/rejections.jsonl",
            "audits/repair_history.jsonl",
            "audits/source_audits.jsonl",
            "audits/tokenizer_replay.json",
            "audits/validator_b.jsonl",
            *(f"minutes_alignment/{split}.jsonl" for split in SPLITS),
            *(f"minutes_alignment/manifests/{split}.jsonl" for split in SPLITS),
        ]
        files = _file_records(staging, payload_paths)
        manifest = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": release.name,
            "created_at_utc": _utc_now(),
            "immutable": True,
            "quality_status": "passed",
            "dataset_role": DATASET_ROLE,
            "training_scope": TRAINING_SCOPE,
            "paper_model_label": "Model chk-2",
            "canonical_stage_id": None,
            "dag_bindable": False,
            "promotable_as_canonical_chk2": False,
            "training_mapping": "exact chk-1 final analysis -> teacher native reasoning -> teacher-synthetic Minutes rewrite",
            "publisher_implementation": publisher_implementation,
            "tokenizer_runtime_contract": tokenizer_runtime_contract,
            "source": {
                "candidate_root": str(source_root.resolve()),
                "generation_manifest_root": str(generation_manifest_root.resolve()),
                "split_counts": dict(expected_split_counts),
                "sample_id_sha256": expected_sample_id_sha256,
                "files": source_bindings,
            },
            "acquisition": {
                "path": str(acquisition_root.resolve()),
                "final_summary_sha256": sha256_file(
                    acquisition_root / "final_summary.json"
                ),
                "preparation_summary_sha256": sha256_file(
                    acquisition_root / "preparation_summary.json"
                ),
                "prompt_contract_sha256": sha256_file(
                    acquisition_root / "prompt_contract.json"
                ),
                "prepare_manifest_sha256": sha256_file(
                    acquisition_root / "prepare_manifest.json"
                ),
                "official_reference_bank_sha256": reference_file_sha256,
                "official_reference_bank_embedded_sha256": (
                    reference_bank.reference_bank_sha256
                ),
                "official_roster_path": str(official_roster_path.resolve()),
                "official_roster_sha256": expected_official_roster_sha256,
                "source_admission_receipt_sha256": sha256_file(
                    acquisition_root / "source_admission_receipt.json"
                ),
            },
            "lineage": contract_lineage,
            "split_counts": split_counts,
            "rejection_counts": rejection_counts,
            "source_rows": source_total,
            "total_rows": len(pass_ids),
            "files": files,
        }
        _write_json(staging / "release_manifest.json", manifest)
        handoff = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": release.name,
            "created_at_utc": manifest["created_at_utc"],
            "immutable": True,
            "quality_status": "passed",
            "dataset_role": DATASET_ROLE,
            "training_scope": TRAINING_SCOPE,
            "dataset_path": str(release / "minutes_alignment"),
            "release_manifest": {
                "path": "release_manifest.json",
                "sha256": sha256_file(staging / "release_manifest.json"),
            },
            "data_quality_audit": {
                "path": "audits/data_quality.json",
                "sha256": sha256_file(staging / "audits/data_quality.json"),
            },
            "tokenizer_replay": {
                "path": "audits/tokenizer_replay.json",
                "sha256": sha256_file(staging / "audits/tokenizer_replay.json"),
            },
            "split_counts": split_counts,
            "rejection_counts": rejection_counts,
            "source_rows": source_total,
            "total_rows": len(pass_ids),
            "publisher_implementation": publisher_implementation,
            "lineage": contract_lineage,
        }
        _write_json(staging / "handoff.json", handoff)
        os.replace(staging, release)
        return handoff
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_release(
    release_root: Path, *, expected_manifest_sha256: str | None
) -> dict[str, Any]:
    """Verify every sealed byte and the core PASS-only release invariants."""

    root = release_root.resolve()
    _require(root.is_dir() and not root.is_symlink(), f"invalid release root: {root}")
    manifest_path = root / "release_manifest.json"
    handoff_path = root / "handoff.json"
    if expected_manifest_sha256 is not None:
        _require(
            sha256_file(manifest_path) == expected_manifest_sha256,
            "release manifest SHA-256 mismatch",
        )
    manifest = _read_json(manifest_path, label="release manifest")
    handoff = _read_json(handoff_path, label="release handoff")
    _require(
        manifest.get("schema_version") == RELEASE_SCHEMA_VERSION, "release schema drift"
    )
    _require(
        manifest.get("dataset_role") == DATASET_ROLE
        and manifest.get("training_scope") == TRAINING_SCOPE,
        "release identity drift",
    )
    _require(
        manifest.get("quality_status") == "passed"
        and manifest.get("immutable") is True,
        "release is not immutable/pass",
    )
    expected_tokenizer_runtime = generator._expected_tokenizer_runtime_contract()
    _require(
        manifest.get("tokenizer_runtime_contract") == expected_tokenizer_runtime,
        "release tokenizer runtime contract drift",
    )
    _require(
        manifest.get("dag_bindable") is False
        and manifest.get("promotable_as_canonical_chk2") is False,
        "unauthorized canonical chk2 claim",
    )
    _validate_required_lineage(manifest.get("lineage"), label="release lineage")
    _require(
        handoff.get("release_manifest", {}).get("sha256") == sha256_file(manifest_path),
        "handoff manifest binding drift",
    )
    publisher_implementation = _mapping(
        manifest.get("publisher_implementation"),
        label="release publisher implementation",
    )
    _string(
        publisher_implementation.get("path"),
        label="release publisher implementation path",
    )
    _sha(
        publisher_implementation.get("sha256"),
        label="release publisher implementation sha256",
    )
    _require(
        handoff.get("publisher_implementation") == publisher_implementation,
        "handoff publisher implementation binding drift",
    )
    expected_files = {
        "prompt_contract.json",
        "prepare_manifest.json",
        "official_pre_action_reference_bank.jsonl",
        "source_admission_receipt.json",
        "audits/data_quality.json",
        "audits/rejections.jsonl",
        "audits/repair_history.jsonl",
        "audits/source_audits.jsonl",
        "audits/tokenizer_replay.json",
        "audits/validator_b.jsonl",
        *(f"minutes_alignment/{split}.jsonl" for split in SPLITS),
        *(f"minutes_alignment/manifests/{split}.jsonl" for split in SPLITS),
    }
    files = _mapping(manifest.get("files"), label="release files")
    _require(set(files) == expected_files, "release sealed file set drift")
    physical = {
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    }
    _require(
        physical == expected_files | {"release_manifest.json", "handoff.json"},
        "release contains unbound or missing files",
    )
    for relative, descriptor in files.items():
        path = _resolve_descriptor(root, descriptor, label=f"release file {relative}")
        _require(
            str(path.relative_to(root)) == relative,
            f"release file key/path drift: {relative}",
        )

    sealed_prompt_contract = _read_json(
        root / "prompt_contract.json", label="sealed prompt contract"
    )
    sealed_student = _mapping(
        sealed_prompt_contract.get("student"), label="sealed prompt contract student"
    )
    _require(
        sealed_student.get("tokenizer_runtime_contract") == expected_tokenizer_runtime,
        "sealed prompt tokenizer runtime contract drift",
    )

    counts = _mapping(manifest.get("split_counts"), label="release split_counts")
    _require(
        set(counts) == set(SPLITS)
        and all(
            isinstance(counts[split], int) and counts[split] > 0 for split in SPLITS
        ),
        "release requires nonempty train/validation/test",
    )
    _require(
        sum(counts.values()) == manifest.get("total_rows"), "release total_rows drift"
    )
    try:
        sealed_reference_bank = deserialize_official_reference_bank(
            (root / "official_pre_action_reference_bank.jsonl").read_bytes()
        )
    except (OSError, OfficialReferenceError) as exc:
        raise PublicationError(
            f"sealed official reference bank is invalid: {exc}"
        ) from exc
    sample_ids: list[str] = []
    for split in SPLITS:
        data = _read_jsonl(
            root / f"minutes_alignment/{split}.jsonl", label=f"release {split} data"
        )
        sidecars = _read_jsonl(
            root / f"minutes_alignment/manifests/{split}.jsonl",
            label=f"release {split} manifests",
        )
        _require(
            len(data) == len(sidecars) == counts[split],
            f"release {split} population drift",
        )
        previous_source_index = -1
        for release_index, (row, sidecar) in enumerate(zip(data, sidecars)):
            _require(
                set(row) == {"prompt", "response"},
                f"training row schema drift: {split}:{release_index}",
            )
            response = _string(
                row.get("response"), label=f"training response {split}:{release_index}"
            )
            _require(
                response.count(BOUNDARY) == 1 and "<think>" not in response.lower(),
                f"training response boundary drift: {split}:{release_index}",
            )
            sample_id = _string(
                sidecar.get("sample_id"),
                label=f"sidecar sample_id {split}:{release_index}",
            )
            source_index = sidecar.get("source_index")
            _require(
                sidecar.get("schema_version") == RELEASE_SCHEMA_VERSION
                and sidecar.get("split") == split
                and sidecar.get("release_index") == release_index
                and isinstance(source_index, int)
                and source_index > previous_source_index,
                f"release sidecar identity/order drift: {sample_id}",
            )
            previous_source_index = source_index
            reasoning, answer = response.split(BOUNDARY, 1)
            for key, expected in {
                "prompt_sha256": sha256_text(row["prompt"]),
                "response_sha256": sha256_text(response),
                "teacher_response_analysis_sha256": sha256_text(reasoning),
                "rewritten_minutes_sha256": sha256_text(answer),
            }.items():
                _require(
                    sidecar.get(key) == expected, f"sidecar {key} drift: {sample_id}"
                )
            validator_b = _mapping(
                sidecar.get("validator_b"), label=f"sidecar validator B: {sample_id}"
            )
            _require(
                validator_b.get("comparison_status") == "comparable"
                and validator_b.get("machine_pass") is True
                and validator_b.get("used_for_selection") is True
                and float(validator_b.get("mean_score", 0)) >= STYLE_MEAN_THRESHOLD
                and int(validator_b.get("min_score", 0)) >= STYLE_MIN_THRESHOLD,
                f"published row did not pass style gate: {sample_id}",
            )
            sidecar_reference = _mapping(
                sidecar.get("official_reference"),
                label=f"sidecar official reference: {sample_id}",
            )
            try:
                meeting_reference = sealed_reference_bank.reference_for_meeting(
                    str(sidecar.get("meeting_date"))
                )
            except OfficialReferenceError as exc:
                raise PublicationError(
                    f"sidecar official reference lookup failed: {sample_id}: {exc}"
                ) from exc
            _require(
                sidecar_reference
                == {
                    "meeting_date": meeting_reference.meeting_date,
                    "meeting_reference_sha256": (
                        meeting_reference.meeting_reference_sha256
                    ),
                    "reference_bank_sha256": (
                        sealed_reference_bank.reference_bank_sha256
                    ),
                },
                f"sidecar official reference binding drift: {sample_id}",
            )
            tokens = _mapping(
                sidecar.get("tokens"), label=f"sidecar tokens: {sample_id}"
            )
            _require(
                isinstance(tokens.get("total_tokens"), int)
                and 0 < tokens["total_tokens"] <= MAX_TOTAL_TOKENS
                and tokens.get("masked_prompt_tokens") == tokens.get("prompt_tokens")
                and tokens.get("unmasked_completion_tokens")
                == tokens.get("completion_tokens"),
                f"sidecar token/mask drift: {sample_id}",
            )
            _validate_required_lineage(
                sidecar.get("lineage"), label=f"sidecar lineage: {sample_id}"
            )
            sample_ids.append(sample_id)
    _require(
        len(sample_ids) == len(set(sample_ids)), "release sample IDs are duplicated"
    )

    audit = _read_json(root / "audits/data_quality.json", label="data quality audit")
    rejections = _read_jsonl(root / "audits/rejections.jsonl", label="rejection audit")
    source_audits = _read_jsonl(
        root / "audits/source_audits.jsonl", label="source audits"
    )
    validator_b_rows = _read_jsonl(
        root / "audits/validator_b.jsonl", label="validator B audits"
    )
    repair_history_rows = _read_jsonl(
        root / "audits/repair_history.jsonl", label="repair-history audits"
    )
    _require(
        "official_evidence" not in canonical_json(repair_history_rows)
        and "official_paragraph_id" not in canonical_json(repair_history_rows)
        and "official_span" not in canonical_json(repair_history_rows),
        "sealed repair history leaks official reference evidence",
    )
    rejection_ids = [
        _string(row.get("sample_id"), label="rejection.sample_id") for row in rejections
    ]
    _require(
        audit.get("status") == "passed"
        and audit.get("split_counts") == counts
        and audit.get("unresolved_rows") == 0
        and audit.get("validator_b_used_for_selection") is True
        and audit.get("tokenizer_runtime_contract") == expected_tokenizer_runtime
        and len(rejection_ids) == audit.get("reject_rows")
        and len(source_audits) == manifest.get("source_rows")
        and set(rejection_ids).isdisjoint(sample_ids),
        "release audit population/selection drift",
    )
    integrity = _mapping(audit.get("integrity"), label="audit.integrity")
    _require(
        integrity.get("pass_sample_id_sha256") == _id_digest(sample_ids)
        and integrity.get("reject_sample_id_sha256") == _id_digest(rejection_ids)
        and integrity.get("pass_reject_disjoint") is True
        and integrity.get("pass_reject_exact_source_partition") is True,
        "release population digest drift",
    )
    _require(
        all(row.get("used_for_selection") is True for row in validator_b_rows)
        and {row["sample_id"] for row in validator_b_rows}.issuperset(sample_ids),
        "validator B audit does not cover PASS rows",
    )
    replay = _read_json(root / "audits/tokenizer_replay.json", label="tokenizer replay")
    _require(
        replay.get("status") == "passed"
        and replay.get("row_count") == len(sample_ids)
        and replay.get("tokenizer", {}).get("runtime_contract")
        == expected_tokenizer_runtime
        and replay.get("contract", {}).get("truncation") is False
        and replay.get("contract", {}).get("total_length_gate") is True
        and replay.get("contract", {}).get("total_max") == MAX_TOTAL_TOKENS
        and replay.get("token_stats", {})
        .get("total", {})
        .get("max", MAX_TOTAL_TOKENS + 1)
        <= MAX_TOTAL_TOKENS,
        "tokenizer replay receipt drift",
    )
    _require(
        _contains_secret(
            (manifest, handoff, audit, rejections, source_audits, validator_b_rows)
        )
        is None,
        "published release contains credential material",
    )
    return handoff


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise PublicationError(
            "transformers is required to load the tokenizer"
        ) from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            **generator.TOKENIZER_LOADER_KWARGS,
        )
        generator._verify_tokenizer_runtime_contract(tokenizer)
        return tokenizer
    except Exception as exc:  # pragma: no cover
        raise PublicationError(
            f"cannot load exact cp200 tokenizer: {path}: {exc}"
        ) from exc


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--generation-manifest-root",
        type=Path,
        default=DEFAULT_GENERATION_MANIFEST_ROOT,
    )
    parser.add_argument(
        "--acquisition-root", type=Path, default=DEFAULT_ACQUISITION_ROOT
    )
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument(
        "--official-roster", type=Path, default=DEFAULT_OFFICIAL_ROSTER_PATH
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify_only:
        result = verify_release(
            args.release_root, expected_manifest_sha256=args.expected_manifest_sha256
        )
    else:
        tokenizer = _load_tokenizer(args.tokenizer_path)
        result = publish_release(
            source_root=args.source_root,
            generation_manifest_root=args.generation_manifest_root,
            acquisition_root=args.acquisition_root,
            release_root=args.release_root,
            tokenizer=tokenizer,
            tokenizer_path=args.tokenizer_path,
            official_roster_path=args.official_roster,
        )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BOUNDARY",
    "DATASET_ROLE",
    "MAX_TOTAL_TOKENS",
    "PublicationError",
    "RELEASE_SCHEMA_VERSION",
    "STYLE_DIMENSIONS",
    "TERMINAL_FIELDS",
    "TERMINAL_SCHEMA_VERSION",
    "TRAINING_SCOPE",
    "publish_release",
    "render_user_prompt",
    "sha256_file",
    "sha256_text",
    "verify_release",
]
