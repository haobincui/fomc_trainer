"""Materialize an auditable official-Minutes supervision release.

This is the paper-level ``Model chk-2`` analysis-to-Minutes dataset.  The
repository historically called the same task ``chk3``.  Unlike the current
2,072-row clean release, the final answer in this release is an official FOMC
Minutes excerpt, not the rewrite teacher's synthetic answer.

The source manifests were produced by the legacy, row-aligned pipeline and
contain all three required pieces:

* ``raw_analysis``: the accepted first-stage teacher analysis;
* ``teacher_rewrite_reasoning``: the rewrite teacher's reasoning content; and
* ``reference_excerpt``: the corresponding official Minutes source text.

The trainable completion is exactly::

    sanitized_teacher_reasoning\n</think>\nofficial_minutes_paragraph

The synthetic ``teacher_rewrite_response`` is never used as a target.  It is
retained only by hash in the side manifest so that target provenance cannot be
silently confused later.

The legacy analysis teacher saw the row-associated official excerpt as a style
and framing reference.  The source analysis is therefore reference-conditioned
even though the student prompt contains no direct target field.  This release
must not be represented as a reference-free or leakage-safe evaluation set.

The deterministic screen is intentionally conservative but is not a semantic
entailment audit. The release manifest therefore records that sentence-level
semantic and human review are required before confirmatory training claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from jobs.generation.generate_chk3_sft_targets import (
    CONTROL_MARKERS,
    _attribution_categories,
    _date_values,
    _numeric_values,
    _reasoning_meta_categories,
    _sanitize_reasoning,
    build_sft_completion,
    canonical_json,
    render_user_prompt,
)
from jobs.retrain_v2.token_budget_gate import _count_sft_row


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "dataset/processed/train/minutes_alignment"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "dataset/processed/retrain_v2/"
    "chk2_official_minutes_v1_20260826"
)
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "models/DeepSeek-R1-Distill-Llama-8B"

SOURCE_SPLITS = ("train", "eval", "test")
OUTPUT_SPLIT = {"train": "train", "eval": "validation", "test": "test"}
SCHEMA_VERSION = "chk2-official-minutes-release-v1"
ROW_SCHEMA_VERSION = "chk2-official-minutes-row-v1"
PROMPT_TOKEN_LIMIT = 3072
TOTAL_TOKEN_LIMIT = 4096
MIN_REASONING_TOKENS = 64
MAX_REASONING_TOKENS = 2400
MIN_TARGET_WORDS = 20
MAX_TARGET_WORDS = 400
MIN_LEXICAL_JACCARD = 0.10

STUDENT_SYSTEM_PROMPT = """\
You are a Federal Reserve Minutes editor. The user supplies an economic or
financial analysis. In the native reasoning section, identify the subset of
source claims, quantities, dates, directions, comparisons, and uncertainty
that can be expressed faithfully in one formal FOMC Minutes paragraph, and
plan neutral institutional wording. The paragraph may omit source details, but
it must not introduce a fact, quantity, date, attribution, cause, policy
action, decision, or vote that is unsupported by the supplied analysis.

Reason only about the supplied analysis and the rewrite. Do not reproduce the
complete analysis, expose internal transport details, or emit model-control
tags. After closing the reasoning section, output exactly one formal FOMC
Minutes paragraph. Do not output headings, lists, JSON, commentary, citations,
or answer tags.
"""

_ADMIN_OR_POLICY_RE = re.compile(
    r"\b(?:voted|vote|voting|ratified|directive|policy\s+actions?|"
    r"policy\s+decisions?|policy\s+stance|FOMC\s+actions?|decided\s+to|"
    r"authorized\s+and\s+directed)\b|"
    r"\b(?:Committee|FOMC)(?:'s)?\s+(?:decision|agreement)\b|"
    r"\b(?:maintain|raise|lower|reduce|leave)\b.{0,80}"
    r"\btarget\s+range\b|"
    r"\btarget\s+range\s+for\s+the\s+federal\s+funds\s+rate\b|"
    r"\b(?:easing|tightening)\b.{0,80}\b(?:appropriate|warranted)\b|"
    r"\b(?:appropriate|warranted)\b.{0,80}"
    r"\b(?:at\s+this\s+meeting|policy\s+action|target\s+range|"
    r"federal\s+funds\s+rate)\b|"
    r"PRESENT:|Secretary\s+and\s+Economist|Board\s+of\s+Governors",
    flags=re.IGNORECASE,
)
_SECTION_HEADING_PREFIX_RE = re.compile(
    r"^(?:Staff\s+Review\s+of\s+the\s+(?:Economic(?:\s+and\s+Financial)?|"
    r"Financial)\s+Situation|Staff\s+Economic\s+Outlook|"
    r"Participants['’]\s+Views\s+on\s+Current\s+Conditions\s+and\s+the\s+"
    r"Economic\s+Outlook|Developments\s+in\s+Financial\s+Markets\s+and\s+"
    r"(?:the\s+Federal\s+Reserve['’]s\s+Balance\s+Sheet|Open\s+Market\s+"
    r"Operations)|Committee\s+Policy\s+Actions?|Discussion\s+of\s+Policy\s+"
    r"Normalization)",
    flags=re.IGNORECASE,
)
_DRAFT_TAIL_RE = re.compile(
    r"(?:^|(?<=[.!?])\s+|\n+)"
    r"(?:let\s+me\s+(?:draft|write)|let'?s\s+(?:draft|write|rewrite)|"
    r"i(?:'ll|\s+will)\s+(?:draft|write|produce)|"
    r"(?:draft|proposed\s+(?:answer|paragraph)|final\s+(?:answer|paragraph))\s*:|"
    r"(?:write|produce)\s+something\s+like\s*:)",
    flags=re.IGNORECASE,
)
_LEGACY_TRANSPORT_META_PATTERNS = {
    "user_request": re.compile(
        r"\b(?:the|this)\s+user\b|\buser\s+(?:wants?|asked|requests?|requires?)\b",
        re.IGNORECASE,
    ),
    "task_description": re.compile(
        r"\b(?:the|this|my)\s+task\b|\btask\s+(?:is|requires?|asks?)\b",
        re.IGNORECASE,
    ),
    "source_envelope": re.compile(
        r"\b(?:raw|input|source|provided)\s+analysis\b|"
        r"\b(?:the|this)\s+input\s+(?:is|contains?|covers?|includes?|discusses?)\b|"
        r"^(?:it|this)\s+(?:also\s+)?(?:contains?|covers?|includes?|discusses?)\b|"
        r"\b(?:mentioning|refer(?:ring)?\s+to)\s+the\s+source\b",
        re.IGNORECASE,
    ),
    "target_metadata": re.compile(
        r"\btarget\s+(?:section|meeting|date)\b|\bmeeting\s+date\b|"
        r"\b(?:the\s+)?meeting\s+(?:is|was|would\s+be)\b|"
        r"\bminutes\s+for\s+(?:this|that|the)\s+meeting\b",
        re.IGNORECASE,
    ),
    "format_directive": re.compile(
        r"\b(?:one|single)[,\s]+"
        r"(?:coherent[,\s]+|formal[,\s]+|concise[,\s]+|flowing[,\s]+)?"
        r"paragraph\b|"
        r"\b(?:output|answer)\s+(?:should|must|needs?|is|required)\b|"
        r"\b(?:no|avoid|do\s+not|don't)(?:\s+\w+){0,2}\s+(?:lists?|bullets?|"
        r"titles?|headings?|verbatim|new\s+(?:facts?|data)|unsupported\s+judgments?|"
        r"mentioning\s+the\s+rewrite)\b",
        re.IGNORECASE,
    ),
    "instruction_reference": re.compile(
        r"\b(?:requirements?|instructions?)\b|\b(?:as|per)\s+requested\b",
        re.IGNORECASE,
    ),
    "rewrite_chatter": re.compile(
        r"\b(?:let'?s|let\s+me)\s+(?:rewrite|transform)\b|"
        r"\b(?:transform|rewrite)\s+(?:this|the|it|what(?:'s|\s+is)?|whatever)\b|"
        r"\b(?:all\s+preserved|no\s+title|good|this\s+is\s+a\s+problem)\s*[.!]?$",
        re.IGNORECASE,
    ),
}
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
_STOPWORDS = frozenset(
    """
    the a an and or of to in on for with by from as at that this these those
    was were is are be been being it its their his her they them he she we our
    may might could would should can will not no into over under between while
    during than then also more most less such
    """.split()
)
_BLOCKING_SOURCE_FLAGS = frozenset({"abnormal_section", "header_only_table"})
_ENUMERATED_TARGET_REPAIRS = {
    "\u00ad": "",
    "â\x80\x8d": "\u200d",
    "â\x80\x8e": "\u200e",
    "â\x80\x91": "‑",
    "â\x80\x92": "‒",
    "â\x80\x93": "–",
    "â\x80\x94": "—",
    "â\x80¯": "\u202f",
    "thedownside": "the downside",
}


class OfficialMinutesDataError(RuntimeError):
    """The official-Minutes release cannot be materialized safely."""


@dataclass(frozen=True)
class BuildPolicy:
    min_reasoning_tokens: int = MIN_REASONING_TOKENS
    max_reasoning_tokens: int = MAX_REASONING_TOKENS
    min_target_words: int = MIN_TARGET_WORDS
    max_target_words: int = MAX_TARGET_WORDS
    min_lexical_jaccard: float = MIN_LEXICAL_JACCARD
    prompt_token_limit: int = PROMPT_TOKEN_LIMIT
    total_token_limit: int = TOTAL_TOKEN_LIMIT
    require_official_source_match: bool = True


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise OfficialMinutesDataError(f"missing source manifest: {path}") from exc
    with handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                raise OfficialMinutesDataError(
                    f"blank source row: {path}:{line_number}"
                )
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise OfficialMinutesDataError(
                    f"invalid source JSON: {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise OfficialMinutesDataError(
                    f"source row is not an object: {path}:{line_number}"
                )
            row["_source_path"] = str(path)
            row["_source_line"] = line_number
            rows.append(row)
    return rows


def _normalized_official_paragraph(value: str) -> tuple[str, list[str]]:
    # The archived CSV extraction contains a small, enumerable set of UTF-8
    # punctuation sequences decoded as Latin-1, discretionary soft hyphens,
    # and one audited missing-space artifact. Repair only those exact patterns,
    # then normalize whitespace. Raw and projected hashes are both retained so
    # the projection cannot hide text edits.
    projected = str(value)
    repairs: list[str] = []
    for malformed, replacement in _ENUMERATED_TARGET_REPAIRS.items():
        count = projected.count(malformed)
        if count:
            projected = projected.replace(malformed, replacement)
            repairs.extend([f"{malformed.encode('unicode_escape').decode()}->{replacement}"] * count)
    return " ".join(projected.split()), repairs


def _lexical_tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in _WORD_RE.findall(str(value))
        if len(token) > 2 and token.casefold() not in _STOPWORDS
    }


def lexical_jaccard(left: str, right: str) -> float:
    left_tokens = _lexical_tokens(left)
    right_tokens = _lexical_tokens(right)
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _token_stats(values: Sequence[int]) -> dict[str, int]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"min": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}

    def percentile(fraction: float) -> int:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


@lru_cache(maxsize=None)
def _cached_source_file_records(meeting_date: str) -> tuple[dict[str, Any], ...]:
    compact = meeting_date.replace("-", "")
    root = REPO_ROOT / "dataset/raw_data/labeled_text/after_2009"
    records = []
    for suffix in ("csv", "xlsx"):
        path = root / f"fomcminutes{compact}_labeled.{suffix}"
        if path.is_file():
            records.append(
                {
                    "path": str(path.relative_to(REPO_ROOT)),
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    return tuple(records)


def _source_file_records(meeting_date: str) -> list[dict[str, Any]]:
    return [dict(record) for record in _cached_source_file_records(meeting_date)]


@lru_cache(maxsize=None)
def _official_csv_rows(meeting_date: str) -> tuple[dict[str, str], ...]:
    compact = meeting_date.replace("-", "")
    path = (
        REPO_ROOT
        / "dataset/raw_data/labeled_text/after_2009"
        / f"fomcminutes{compact}_labeled.csv"
    )
    if not path.is_file():
        return ()
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"line_id", "section_name", "raw_text"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise OfficialMinutesDataError(
                    f"official labeled CSV schema is invalid: {path}"
                )
            return tuple(
                {
                    "line_id": str(row.get("line_id") or ""),
                    "section_name": str(row.get("section_name") or ""),
                    "raw_text": str(row.get("raw_text") or ""),
                }
                for row in reader
            )
    except (OSError, csv.Error) as exc:
        raise OfficialMinutesDataError(
            f"cannot read official labeled CSV: {path}: {exc}"
        ) from exc


def _official_source_matches(
    meeting_date: str, official: str
) -> list[dict[str, Any]]:
    compact = meeting_date.replace("-", "")
    relative = (
        Path("dataset/raw_data/labeled_text/after_2009")
        / f"fomcminutes{compact}_labeled.csv"
    )
    matches: list[dict[str, Any]] = []
    for row in _official_csv_rows(meeting_date):
        projected, repairs = _normalized_official_paragraph(row["raw_text"])
        if projected != official:
            continue
        matches.append(
            {
                "path": str(relative),
                "line_id": row["line_id"],
                "section_name": row["section_name"],
                "raw_text_sha256": _sha256_text(row["raw_text"]),
                "projected_text_sha256": _sha256_text(projected),
                "normalization_repairs": repairs,
            }
        )
    return matches


def _legacy_transport_meta_categories(text: str) -> set[str]:
    return {
        category
        for category, pattern in _LEGACY_TRANSPORT_META_PATTERNS.items()
        if pattern.search(text)
    }


def _sanitize_legacy_reasoning(
    raw_reasoning: str, *, analysis: str, official: str
) -> tuple[str, int, bool, list[str]]:
    """Project legacy teacher traces into substantive editorial reasoning.

    The historical rewrite teacher was given an explicit target section and
    often narrated the user request before doing useful fidelity planning. It
    also sometimes drafted the synthetic answer inside its native reasoning.
    Neither behavior belongs in student supervision for the official-target
    task, so both are removed deterministically and recorded in the manifest.
    """

    reasoning, removed_segments = _sanitize_reasoning(
        raw_reasoning,
        analysis=analysis,
        minutes=official,
    )
    detected_categories = sorted(_legacy_transport_meta_categories(reasoning))
    draft_match = _DRAFT_TAIL_RE.search(reasoning)
    draft_tail_removed = draft_match is not None
    if draft_match is not None:
        reasoning = reasoning[: draft_match.start()].rstrip()
        removed_segments += 1

    kept_paragraphs: list[str] = []
    for raw_paragraph in re.split(r"\n\s*\n", reasoning):
        kept_sentences: list[str] = []
        for raw_sentence in re.split(r"(?<=[.!?])\s+", raw_paragraph.strip()):
            sentence = raw_sentence.strip()
            if not sentence:
                continue
            if _legacy_transport_meta_categories(sentence):
                removed_segments += 1
                continue
            kept_sentences.append(sentence)
        if kept_sentences:
            kept_paragraphs.append(" ".join(kept_sentences))
    return (
        "\n\n".join(kept_paragraphs).strip(),
        removed_segments,
        draft_tail_removed,
        detected_categories,
    )


def _candidate(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    token_config: Mapping[str, Any],
    policy: BuildPolicy,
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    required = (
        "sample_id",
        "split",
        "meeting_date",
        "source_row_index",
        "raw_analysis",
        "reference_excerpt",
        "teacher_rewrite_reasoning",
        "teacher_rewrite_response",
        "teacher_status",
    )
    missing = [key for key in required if key not in row]
    if missing:
        raise OfficialMinutesDataError(
            f"source row lacks required keys {missing}: {row.get('_source_path')}:"
            f"{row.get('_source_line')}"
        )

    sample_id = str(row["sample_id"]).strip()
    split = str(row["split"]).strip()
    meeting_date = str(row["meeting_date"]).strip()
    analysis = str(row["raw_analysis"]).strip()
    official_raw = str(row["reference_excerpt"]).strip()
    official, official_normalization_repairs = _normalized_official_paragraph(
        official_raw
    )
    official_source_matches = _official_source_matches(meeting_date, official)
    raw_reasoning = str(row["teacher_rewrite_reasoning"]).strip()
    synthetic_rewrite = str(row["teacher_rewrite_response"]).strip()
    (
        reasoning,
        removed_segments,
        draft_tail_removed,
        removed_transport_categories,
    ) = _sanitize_legacy_reasoning(
        raw_reasoning,
        analysis=analysis,
        official=official,
    )

    reasons: list[str] = []
    source_flags = sorted({str(flag) for flag in row.get("quality_flags", [])})
    reasons.extend(sorted(_BLOCKING_SOURCE_FLAGS & set(source_flags)))
    if str(row["teacher_status"]) not in {"success", "archived_master"}:
        reasons.append("teacher_status_not_accepted")
    if not sample_id:
        reasons.append("empty_sample_id")
    if split not in SOURCE_SPLITS:
        reasons.append("invalid_split")
    if not meeting_date:
        reasons.append("empty_meeting_date")
    if not analysis:
        reasons.append("empty_analysis")
    if not official:
        reasons.append("empty_official_target")
    if not raw_reasoning:
        reasons.append("empty_teacher_reasoning")
    if not synthetic_rewrite:
        reasons.append("empty_teacher_synthetic_rewrite")

    target_words = len(official.split())
    if target_words < policy.min_target_words:
        reasons.append("official_target_too_short")
    if target_words > policy.max_target_words:
        reasons.append("official_target_too_long")
    source_paragraphs = [
        paragraph
        for paragraph in re.split(r"\n\s*\n", official_raw)
        if paragraph.strip()
    ]
    if len(source_paragraphs) != 1:
        reasons.append("official_target_not_one_source_paragraph")
    if _ADMIN_OR_POLICY_RE.search(official):
        reasons.append("administrative_or_policy_target")
    if _SECTION_HEADING_PREFIX_RE.search(official):
        reasons.append("official_target_contains_section_heading")
    if policy.require_official_source_match and len(official_source_matches) != 1:
        reasons.append("official_source_exact_match_count_not_one")

    for field_name, value in (
        ("analysis", analysis),
        ("reasoning", reasoning),
        ("official_target", official),
    ):
        if any(marker in value for marker in CONTROL_MARKERS):
            reasons.append(f"{field_name}_contains_control_marker")

    analysis_numbers = _numeric_values(analysis)
    target_numbers = _numeric_values(official)
    missing_numbers = analysis_numbers - target_numbers
    unsupported_numbers = target_numbers - analysis_numbers
    analysis_dates = _date_values(analysis)
    target_dates = _date_values(official)
    missing_dates = sorted(analysis_dates - target_dates)
    unsupported_dates = sorted(target_dates - analysis_dates)
    analysis_attributions = _attribution_categories(analysis)
    target_attributions = _attribution_categories(official)
    unsupported_attributions = sorted(target_attributions - analysis_attributions)
    reasoning_numbers = _numeric_values(reasoning)
    unsupported_reasoning_numbers = reasoning_numbers - analysis_numbers
    reasoning_dates = _date_values(reasoning)
    unsupported_reasoning_dates = sorted(reasoning_dates - analysis_dates)
    reasoning_attributions = _attribution_categories(reasoning)
    unsupported_reasoning_attributions = sorted(
        reasoning_attributions - analysis_attributions
    )
    overlap = lexical_jaccard(analysis, official)

    if unsupported_numbers:
        reasons.append("unsupported_target_numbers")
    if unsupported_dates:
        reasons.append("unsupported_target_dates")
    if unsupported_attributions:
        reasons.append("unsupported_target_attributions")
    if unsupported_reasoning_numbers:
        reasons.append("unsupported_reasoning_numbers")
    if unsupported_reasoning_dates:
        reasons.append("unsupported_reasoning_dates")
    if unsupported_reasoning_attributions:
        reasons.append("unsupported_reasoning_attributions")
    if overlap < policy.min_lexical_jaccard:
        reasons.append("low_lexical_overlap")

    reasoning_meta = sorted(
        _reasoning_meta_categories(reasoning)
        | _legacy_transport_meta_categories(reasoning)
    )
    if reasoning_meta:
        reasons.append("reasoning_contains_transport_meta")

    prompt = ""
    response = ""
    prompt_tokens = completion_tokens = total_tokens = reasoning_tokens = 0
    try:
        prompt = render_user_prompt(analysis)
        response = build_sft_completion(reasoning, official)
        reasoning_tokens = len(tokenizer.encode(reasoning, add_special_tokens=False))
        prompt_tokens, completion_tokens, total_tokens = _count_sft_row(
            {"prompt": prompt, "response": response},
            tokenizer=tokenizer,
            config=token_config,
        )
    except Exception as exc:  # converted into a row-level rejection ledger
        reasons.append(f"serialization_or_tokenization_error:{type(exc).__name__}")
    if reasoning_tokens < policy.min_reasoning_tokens:
        reasons.append("reasoning_too_short")
    if reasoning_tokens > policy.max_reasoning_tokens:
        reasons.append("reasoning_too_long")
    if prompt_tokens > policy.prompt_token_limit:
        reasons.append("prompt_token_overflow")
    if total_tokens > policy.total_token_limit:
        reasons.append("total_token_overflow")

    reasons = list(dict.fromkeys(reasons))
    diagnostics: dict[str, Any] = {
        "lexical_jaccard": round(overlap, 8),
        "analysis_numbers": _counter_dict(analysis_numbers),
        "official_target_numbers": _counter_dict(target_numbers),
        "omitted_source_numbers": _counter_dict(missing_numbers),
        "unsupported_target_numbers": _counter_dict(unsupported_numbers),
        "analysis_dates": sorted(analysis_dates),
        "official_target_dates": sorted(target_dates),
        "omitted_source_dates": missing_dates,
        "unsupported_target_dates": unsupported_dates,
        "analysis_attributions": sorted(analysis_attributions),
        "official_target_attributions": sorted(target_attributions),
        "unsupported_target_attributions": unsupported_attributions,
        "reasoning_numbers": _counter_dict(reasoning_numbers),
        "unsupported_reasoning_numbers": _counter_dict(
            unsupported_reasoning_numbers
        ),
        "reasoning_dates": sorted(reasoning_dates),
        "unsupported_reasoning_dates": unsupported_reasoning_dates,
        "reasoning_attributions": sorted(reasoning_attributions),
        "unsupported_reasoning_attributions": unsupported_reasoning_attributions,
        "reasoning_meta_categories": reasoning_meta,
        "reasoning_removed_segments": removed_segments,
        "reasoning_removed_transport_categories": removed_transport_categories,
        "reasoning_draft_tail_removed": draft_tail_removed,
        "target_words": target_words,
        "official_source_exact_match_count": len(official_source_matches),
        "prompt_tokens": prompt_tokens,
        "reasoning_tokens": reasoning_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
    manifest = {
        "schema_version": ROW_SCHEMA_VERSION,
        "sample_id": sample_id,
        "split": OUTPUT_SPLIT.get(split, split),
        "source_split": split,
        "meeting_date": meeting_date,
        "section_name": str(row.get("section_name") or ""),
        "topic": str(row.get("topic") or ""),
        "source_row_index": row["source_row_index"],
        "target_type": (
            "whitespace_and_enumerated_text_normalized_official_"
            "fomc_minutes_paragraph"
        ),
        "target_source_field": "reference_excerpt",
        "target_is_teacher_synthetic_rewrite": False,
        "target_alignment_status": "row_associated_deterministic_screen_only",
        "analysis_source_lineage": "legacy_reference_conditioned_teacher_answer",
        "analysis_teacher_saw_row_official_excerpt": True,
        "analysis_is_reference_free": False,
        "analysis_point_in_time_status": "not_established_for_legacy_lineage",
        "analysis": analysis,
        "reasoning": reasoning,
        "official_minutes_paragraph": official,
        "analysis_sha256": _sha256_text(analysis),
        "raw_reasoning_sha256": _sha256_text(raw_reasoning),
        "reasoning_sha256": _sha256_text(reasoning),
        "official_minutes_raw_sha256": _sha256_text(official_raw),
        "official_minutes_sha256": _sha256_text(official),
        "official_minutes_normalization_repairs": official_normalization_repairs,
        "official_source_match": (
            official_source_matches[0] if len(official_source_matches) == 1 else None
        ),
        "teacher_synthetic_rewrite_sha256": _sha256_text(synthetic_rewrite),
        "prompt_sha256": _sha256_text(prompt) if prompt else "",
        "response_sha256": _sha256_text(response) if response else "",
        "teacher_model": str(row.get("teacher_model") or ""),
        "teacher_status": str(row["teacher_status"]),
        "source_quality_flags": source_flags,
        "source_manifest": {
            "path": _display_path(Path(str(row["_source_path"]))),
            "line": int(row["_source_line"]),
        },
        "official_source_files": _source_file_records(meeting_date),
        "diagnostics": diagnostics,
    }
    training = {"prompt": prompt, "response": response}
    return manifest, training, reasons


def _meeting_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    meetings = sorted({str(row["meeting_date"]) for row in rows})
    return {
        "meetings": len(meetings),
        "first_meeting": meetings[0] if meetings else None,
        "last_meeting": meetings[-1] if meetings else None,
    }


def _file_record(root: Path, relative: str, *, rows: int | None = None) -> dict[str, Any]:
    path = root / relative
    record: dict[str, Any] = {
        "path": relative,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _external_file_record(path: Path, *, rows: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _display_path(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows:
        with path.open(encoding="utf-8") as handle:
            record["rows"] = sum(1 for line in handle if line.strip())
    return record


def _tokenizer_provenance(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "path": None,
            "files": {},
            "warning": "tokenizer path was not supplied to the library call",
        }
    resolved = path.resolve()
    files = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        candidate = resolved / name
        if candidate.is_file():
            files[name] = _external_file_record(candidate)
    return {"path": _display_path(resolved), "files": files}


def materialize_release(
    *,
    source_root: Path,
    output_root: Path,
    tokenizer: Any,
    tokenizer_path: Path | None = None,
    policy: BuildPolicy = BuildPolicy(),
) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise OfficialMinutesDataError(f"output release already exists: {output_root}")

    source_by_split: dict[str, list[dict[str, Any]]] = {}
    source_ids: set[str] = set()
    source_meetings: dict[str, set[str]] = {}
    for split in SOURCE_SPLITS:
        path = source_root / f"{split}_manifest.jsonl"
        rows = _load_jsonl(path)
        source_by_split[split] = rows
        source_meetings[split] = {str(row.get("meeting_date") or "") for row in rows}
        for row in rows:
            sample_id = str(row.get("sample_id") or "")
            if sample_id in source_ids:
                raise OfficialMinutesDataError(f"duplicate source sample ID: {sample_id}")
            source_ids.add(sample_id)
    for left_index, left in enumerate(SOURCE_SPLITS):
        for right in SOURCE_SPLITS[left_index + 1 :]:
            overlap = sorted(source_meetings[left] & source_meetings[right])
            if overlap:
                raise OfficialMinutesDataError(
                    f"meeting split overlap: {left}/{right}: {overlap[:5]}"
                )

    token_config = {
        "_stage_kind": "minutes_sft",
        "dataset_prompt_column": "prompt",
        "system_prompt": STUDENT_SYSTEM_PROMPT,
    }
    staging_parent = output_root.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=staging_parent))
    try:
        accepted_by_split: dict[str, list[dict[str, Any]]] = {}
        training_by_split: dict[str, list[dict[str, str]]] = {}
        rejected_by_split: dict[str, list[dict[str, Any]]] = {}
        rejection_counts: Counter[str] = Counter()
        seen_target_hashes: set[str] = set()
        seen_prompt_hashes: set[str] = set()

        for source_split in SOURCE_SPLITS:
            output_split = OUTPUT_SPLIT[source_split]
            accepted: list[dict[str, Any]] = []
            training_rows: list[dict[str, str]] = []
            rejected: list[dict[str, Any]] = []
            for row in source_by_split[source_split]:
                manifest, training, reasons = _candidate(
                    row,
                    tokenizer=tokenizer,
                    token_config=token_config,
                    policy=policy,
                )
                target_hash = manifest["official_minutes_sha256"]
                prompt_hash = manifest["prompt_sha256"]
                if not reasons and target_hash in seen_target_hashes:
                    reasons.append("duplicate_official_target")
                if not reasons and prompt_hash in seen_prompt_hashes:
                    reasons.append("duplicate_student_prompt")
                if reasons:
                    rejection_counts.update(set(reasons))
                    rejected.append(
                        {
                            "schema_version": ROW_SCHEMA_VERSION,
                            "sample_id": manifest["sample_id"],
                            "split": output_split,
                            "meeting_date": manifest["meeting_date"],
                            "topic": manifest["topic"],
                            "source_row_index": manifest["source_row_index"],
                            "analysis_sha256": manifest["analysis_sha256"],
                            "official_minutes_sha256": target_hash,
                            "rejection_reasons": sorted(reasons),
                            "diagnostics": manifest["diagnostics"],
                            "source_manifest": manifest["source_manifest"],
                        }
                    )
                    continue
                seen_target_hashes.add(target_hash)
                seen_prompt_hashes.add(prompt_hash)
                accepted.append(manifest)
                training_rows.append(training)

            accepted_by_split[output_split] = accepted
            training_by_split[output_split] = training_rows
            rejected_by_split[output_split] = rejected
            _write_jsonl(staging / f"minutes_alignment/{output_split}.jsonl", training_rows)
            _write_jsonl(
                staging / f"minutes_alignment/manifests/{output_split}.jsonl",
                accepted,
            )
            _write_jsonl(staging / f"rejections/{output_split}.jsonl", rejected)

        accepted_all = [
            row for split in OUTPUT_SPLIT.values() for row in accepted_by_split[split]
        ]
        if not accepted_all:
            raise OfficialMinutesDataError("no row passed the official-Minutes gates")
        accepted_ids = [str(row["sample_id"]) for row in accepted_all]
        if len(accepted_ids) != len(set(accepted_ids)):
            raise OfficialMinutesDataError("accepted sample IDs are not unique")
        accepted_meetings = {
            split: {str(row["meeting_date"]) for row in accepted_by_split[split]}
            for split in OUTPUT_SPLIT.values()
        }
        split_names = tuple(OUTPUT_SPLIT.values())
        for left_index, left in enumerate(split_names):
            for right in split_names[left_index + 1 :]:
                if accepted_meetings[left] & accepted_meetings[right]:
                    raise OfficialMinutesDataError(
                        f"accepted meeting split overlap: {left}/{right}"
                    )

        prompt_tokens = [int(row["diagnostics"]["prompt_tokens"]) for row in accepted_all]
        reasoning_tokens = [
            int(row["diagnostics"]["reasoning_tokens"]) for row in accepted_all
        ]
        completion_tokens = [
            int(row["diagnostics"]["completion_tokens"]) for row in accepted_all
        ]
        total_tokens = [int(row["diagnostics"]["total_tokens"]) for row in accepted_all]
        target_words = [int(row["diagnostics"]["target_words"]) for row in accepted_all]
        split_counts = {
            split: len(accepted_by_split[split]) for split in OUTPUT_SPLIT.values()
        }
        source_counts = {
            OUTPUT_SPLIT[split]: len(source_by_split[split]) for split in SOURCE_SPLITS
        }
        rejection_split_counts = {
            split: len(rejected_by_split[split]) for split in OUTPUT_SPLIT.values()
        }
        audit = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed_deterministic_gates",
            "training_ready": False,
            "paper_model": "chk-2",
            "repository_task_alias": "chk3/minutes_alignment",
            "intended_use": (
                "reference-conditioned analysis-to-official-FOMC-Minutes "
                "paragraph SFT candidate"
            ),
            "target_provenance": (
                "row-associated official reference_excerpt with enumerated "
                "encoding and whitespace normalization; teacher synthetic "
                "answer excluded"
            ),
            "analysis_provenance": {
                "lineage": "legacy_reference_conditioned_teacher_answer",
                "analysis_teacher_saw_row_official_excerpt": True,
                "analysis_is_reference_free": False,
                "student_prompt_has_direct_target_field": False,
                "point_in_time_status": (
                    "not_established; legacy lineage is not bound to the canonical "
                    "D-1 evidence replay"
                ),
                "warning": (
                    "The same row's official excerpt was visible to the legacy "
                    "analysis teacher as a style and framing reference. This "
                    "artifact is circular supervision and is not a leakage-safe "
                    "evaluation set."
                ),
            },
            "source_counts": source_counts,
            "accepted_counts": split_counts,
            "rejected_counts": rejection_split_counts,
            "total_source_rows": sum(source_counts.values()),
            "total_accepted_rows": sum(split_counts.values()),
            "total_rejected_rows": sum(rejection_split_counts.values()),
            "accepted_fraction": round(
                sum(split_counts.values()) / sum(source_counts.values()), 8
            ),
            "unique_accepted_sample_ids": len(accepted_ids),
            "unique_official_targets": len(seen_target_hashes),
            "unique_student_prompts": len(seen_prompt_hashes),
            "meetings": {
                split: _meeting_summary(accepted_by_split[split])
                for split in OUTPUT_SPLIT.values()
            },
            "rejection_reason_row_counts": dict(sorted(rejection_counts.items())),
            "topic_counts": dict(
                sorted(Counter(str(row["topic"]) for row in accepted_all).items())
            ),
            "teacher_model_counts": dict(
                sorted(
                    Counter(str(row["teacher_model"]) for row in accepted_all).items()
                )
            ),
            "reasoning_sanitization": {
                "rows_with_draft_tail_removed": sum(
                    bool(row["diagnostics"]["reasoning_draft_tail_removed"])
                    for row in accepted_all
                ),
                "removed_transport_category_row_counts": dict(
                    sorted(
                        Counter(
                            category
                            for row in accepted_all
                            for category in row["diagnostics"][
                                "reasoning_removed_transport_categories"
                            ]
                        ).items()
                    )
                ),
            },
            "token_contract": {
                "prompt_max": policy.prompt_token_limit,
                "reasoning_min": policy.min_reasoning_tokens,
                "reasoning_max": policy.max_reasoning_tokens,
                "total_max": policy.total_token_limit,
                "overflow_policy": "reject_without_truncation",
                "boundary": "exactly_one_literal_</think>",
            },
            "token_stats": {
                "prompt": _token_stats(prompt_tokens),
                "reasoning": _token_stats(reasoning_tokens),
                "completion": _token_stats(completion_tokens),
                "total": _token_stats(total_tokens),
            },
            "official_target_word_stats": _token_stats(target_words),
            "alignment_contract": {
                "official_target_may_omit_source_facts": True,
                "official_target_may_add_source_unsupported_numbers": False,
                "official_target_may_add_source_unsupported_dates": False,
                "official_target_may_add_source_unsupported_attributions": False,
                "reasoning_may_add_source_unsupported_numbers": False,
                "reasoning_may_add_source_unsupported_dates": False,
                "reasoning_may_add_source_unsupported_attributions": False,
                "minimum_lexical_jaccard": policy.min_lexical_jaccard,
                "administrative_and_policy_action_targets": (
                    "screened_by_conservative_deterministic_regex"
                ),
                "section_heading_prefixes": "rejected_by_known_heading_regex",
                "official_source_exact_match_count": (
                    "required_to_equal_one_in_meeting_labeled_csv"
                ),
                "single_paragraph_contract": (
                    "one source text block serialized with normalized whitespace"
                ),
                "source_blocking_quality_flags": sorted(_BLOCKING_SOURCE_FLAGS),
            },
            "semantic_review": {
                "status": "not_run",
                "training_readiness": "conditional_on_sentence_level_semantic_and_human_audit",
                "warning": (
                    "Deterministic number/date/attribution and lexical gates do not prove "
                    "sentence-level entailment or full claim alignment. The legacy "
                    "teacher reasoning was produced for a rewrite task rather than "
                    "conditioned on the selected official target. The source "
                    "analysis was generated with the same official excerpt visible."
                ),
            },
            "checks": {
                "required_fields": "passed_for_accepted_rows",
                "meeting_disjoint_splits": "passed",
                "sample_id_uniqueness": "passed",
                "prompt_uniqueness": "passed",
                "official_target_uniqueness": "passed",
                "official_target_provenance": (
                    "passed_exact_one_labeled_csv_row"
                    if policy.require_official_source_match
                    else "not_enforced_by_test_policy"
                ),
                "teacher_synthetic_answer_excluded": "passed",
                "reasoning_transport_meta_removed": "passed",
                "reasoning_synthetic_draft_tail_removed": "passed",
                "reasoning_unsupported_structured_facts": "passed",
                "response_boundary": "passed",
                "single_paragraph_serialization": "passed",
                "unsupported_structured_facts": "passed",
                "token_budget": "passed",
            },
        }
        _write_json(staging / "audits/data_quality.json", audit)
        _write_json(
            staging / "prompt_contract.json",
            {
                "schema_version": SCHEMA_VERSION,
                "student_system_prompt": STUDENT_SYSTEM_PROMPT,
                "student_system_prompt_sha256": _sha256_text(STUDENT_SYSTEM_PROMPT),
                "student_user_prompt": (
                    "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
                    '{"analysis":"<accepted first-stage teacher analysis>"}'
                ),
                "completion_mapping": {
                    "reasoning": (
                        "deterministically_sanitized_legacy_teacher_rewrite_reasoning"
                    ),
                    "boundary": "\\n</think>\\n",
                    "final_answer": (
                        "enumerated-text-and-whitespace-normalized official "
                        "reference_excerpt"
                    ),
                    "teacher_synthetic_rewrite": "audit_hash_only_not_supervision",
                },
                "reasoning_provenance_warning": (
                    "Legacy teacher reasoning was not generated with the official "
                    "reference_excerpt as its visible target; semantic review is required."
                ),
                "analysis_provenance_warning": (
                    "The legacy analysis teacher saw the row-associated official "
                    "reference_excerpt; this is not a reference-free source lineage."
                ),
            },
        )

        files: dict[str, dict[str, Any]] = {}
        for split in OUTPUT_SPLIT.values():
            files[f"minutes_alignment/{split}.jsonl"] = _file_record(
                staging,
                f"minutes_alignment/{split}.jsonl",
                rows=len(training_by_split[split]),
            )
            files[f"minutes_alignment/manifests/{split}.jsonl"] = _file_record(
                staging,
                f"minutes_alignment/manifests/{split}.jsonl",
                rows=len(accepted_by_split[split]),
            )
            files[f"rejections/{split}.jsonl"] = _file_record(
                staging,
                f"rejections/{split}.jsonl",
                rows=len(rejected_by_split[split]),
            )
        files["audits/data_quality.json"] = _file_record(
            staging, "audits/data_quality.json"
        )
        files["prompt_contract.json"] = _file_record(staging, "prompt_contract.json")
        source_files = {
            f"{split}_manifest.jsonl": {
                "path": _display_path(source_root / f"{split}_manifest.jsonl"),
                "sha256": _sha256_file(source_root / f"{split}_manifest.jsonl"),
                "rows": len(source_by_split[split]),
            }
            for split in SOURCE_SPLITS
        }
        analysis_lineage_files: dict[str, dict[str, Any]] = {}
        for phase in ("teacher_prompts", "teacher_responses"):
            phase_root = (
                REPO_ROOT
                / "dataset/processed/pipeline/analysis_sft"
                / phase
                / "after_2009"
            )
            for split in SOURCE_SPLITS:
                path = phase_root / f"{split}.jsonl"
                if path.is_file():
                    analysis_lineage_files[f"{phase}/{split}.jsonl"] = (
                        _external_file_record(path, rows=True)
                    )
        code_dependencies = {}
        for path in (
            REPO_ROOT / "jobs/generation/generate_chk3_sft_targets.py",
            REPO_ROOT / "jobs/retrain_v2/token_budget_gate.py",
        ):
            code_dependencies[_display_path(path)] = _external_file_record(path)
        release_manifest = {
            "schema_version": SCHEMA_VERSION,
            "release_id": output_root.name,
            "created_at_utc": _utc_now(),
            "paper_model": "chk-2",
            "repository_task_alias": "chk3/minutes_alignment",
            "dataset_role": "official_minutes_final_answer_supervision_candidate",
            "dataset_path": str(
                _display_path(output_root / "minutes_alignment")
            ),
            "quality_status": (
                "deterministic_screen_passed_reference_conditioned_"
                "semantic_review_pending"
            ),
            "source_bytes_hash_bound": True,
            "full_rebuild_environment_sealed": False,
            "target_type": (
                "whitespace_and_enumerated_text_normalized_official_"
                "fomc_minutes_paragraph"
            ),
            "target_is_teacher_generated": False,
            "teacher_synthetic_rewrite_used_for_supervision": False,
            "analysis_is_reference_free": False,
            "analysis_teacher_saw_row_official_excerpt": True,
            "analysis_point_in_time_status": "not_established_for_legacy_lineage",
            "training_ready": False,
            "reported_checkpoint_eligible": False,
            "split_counts": split_counts,
            "total_rows": sum(split_counts.values()),
            "source": {
                "root": _display_path(source_root),
                "files": source_files,
                "legacy_analysis_lineage_files": analysis_lineage_files,
            },
            "builder": {
                "path": _display_path(Path(__file__)),
                "sha256": _sha256_file(Path(__file__)),
                "code_dependencies": code_dependencies,
                "python": sys.version,
            },
            "tokenizer": _tokenizer_provenance(tokenizer_path),
            "files": files,
        }
        _write_json(staging / "release_manifest.json", release_manifest)
        release_manifest_sha = _sha256_file(staging / "release_manifest.json")
        handoff = {
            "schema_version": SCHEMA_VERSION,
            "release_id": output_root.name,
            "dataset_path": release_manifest["dataset_path"],
            "paper_model": "chk-2",
            "quality_status": release_manifest["quality_status"],
            "training_ready": False,
            "semantic_review_status": "not_run",
            "split_counts": split_counts,
            "total_rows": sum(split_counts.values()),
            "release_manifest": {
                "path": "release_manifest.json",
                "sha256": release_manifest_sha,
            },
            "data_quality_audit": {
                "path": "audits/data_quality.json",
                "sha256": _sha256_file(staging / "audits/data_quality.json"),
            },
        }
        _write_json(staging / "handoff.json", handoff)
        os.replace(staging, output_root)
        return handoff
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise OfficialMinutesDataError("transformers is required for token audit") from exc
    return AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer = _load_tokenizer(args.tokenizer_path.resolve())
    handoff = materialize_release(
        source_root=args.source_root,
        output_root=args.output_root,
        tokenizer=tokenizer,
        tokenizer_path=args.tokenizer_path,
    )
    print(json.dumps(handoff, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
