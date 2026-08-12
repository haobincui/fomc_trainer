"""Build an immutable chk3 release by deleting reasoning contamination locally.

This is deliberately a deletion-only transform.  It never calls a model and
never changes the user prompt or the final Minutes paragraph.  The source v3
release is fully hash-verified before use.  Rows that become shorter than the
64-token reasoning floor are written to ``needs_regeneration.jsonl`` and block
publication of the complete release.

An optional recovery ledger can supply a previously generated, independently
audited reasoning string for those rows.  Recovery is bound by sample ID,
prompt hash and final-answer hash and is revalidated under the same v4 gates.
"""

from __future__ import annotations

import argparse
import ctypes
import difflib
import errno
import json
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from jobs.generation.generate_chk3_sft_targets import (
    DEFAULT_TOKENIZER_PATH,
    MAX_REASONING_TOKENS,
    MIN_REASONING_TOKENS,
    STUDENT_SYSTEM_PROMPT,
    USER_PROMPT_PREFIX,
    Chk3DataError,
    _load_tokenizer,
    canonical_json,
    sha256_file,
    sha256_text,
)
from src.open_r1.trainer.sft_prompt_renderer import (
    render_sft_prompt,
    tokenize_sft_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "chk3-reasoning-clean-local-v1"
RELEASE_SCHEMA_VERSION = "chk3-minutes-reasoning-clean-release-v1"
RECOVERY_SCHEMA_VERSION = "chk3-reasoning-local-recovery-v1"
DEFAULT_SOURCE_RELEASE = (
    REPO_ROOT
    / "dataset/processed/retrain_v2/chk3_minutes_clean_v3_20260805"
)
DEFAULT_RELEASE_PARENT = REPO_ROOT / "dataset/processed/retrain_v2"
DEFAULT_RELEASE_ID = "chk3_minutes_clean_v4_local_reasoning_20260810"
DEFAULT_WORK_ROOT = (
    REPO_ROOT / "output/data/retrain_v2/chk3/reasoning_clean_v4_20260810"
)
EXPECTED_SPLIT_COUNTS = {"train": 1683, "validation": 199, "test": 190}
OUTPUT_TO_SOURCE_SPLIT = {"train": "train", "validation": "eval", "test": "test"}
BOUNDARY = "\n</think>\n"
TOTAL_TOKEN_LIMIT = 4096
PROMPT_TOKEN_LIMIT = 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_EVIDENCE_ID_RE = re.compile(r"\bev-[0-9a-f]+\b", flags=re.IGNORECASE)
_EV_CITATION_RE = re.compile(
    r"\b(?:ev[- ]citations?|evidence[- ]citations?)\b", flags=re.IGNORECASE
)
_CONTROL_RE = re.compile(r"</?think>|</?answer>|<\|(?:begin|end)_of_sentence\|>")
_BULLET_RE = re.compile(r"^\s*(?:[-*+\u2022]|\d+[.)]|[A-Za-z][.)])\s+")
_HEADING_ONLY_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:plan|outline|reasoning|analysis|checking|check|"
    r"draft|final(?:\s+answer)?|rewrite|fidelity(?:\s+check)?|steps?|"
    r"claims?|quantities|dates?|directions?|comparisons?|uncertainty)\s*:?\s*$",
    flags=re.IGNORECASE,
)
_HEADING_PREFIX_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:plan|outline|reasoning|analysis|checking|check|"
    r"draft|final(?:\s+answer)?|rewrite|fidelity(?:\s+check)?|steps?|"
    r"claims?|quantities|dates?|directions?|comparisons?|uncertainty)\s*:\s*",
    flags=re.IGNORECASE,
)
_SOURCE_LEAD_RE = re.compile(
    r"^\s*(?:the|this)\s+(?:(?:provided|source|original|supplied)\s+)?"
    r"(?:analysis|text|passage)\s+(?:states|reports|notes|indicates|shows|"
    r"describes|contains|says|presents|explains|focuses\s+on|covers)"
    r"(?:\s+that)?\s*[:,;-]?\s*",
    flags=re.IGNORECASE,
)
_META_PATTERNS: dict[str, re.Pattern[str]] = {
    "transport_or_role": re.compile(
        r"\b(?:prompt|instruction|task|user|teacher|student|assistant|json|api|"
        r"schema|field|key|response|final answer|answer tag|control tag|"
        r"output (?:format|contract|text|answer|response)|the output)\b",
        flags=re.IGNORECASE,
    ),
    "fidelity_inventory": re.compile(
        r"\b(?:silent fidelity ledger|fidelity ledger|quantity occurrences?|"
        r"date markers?|required dates?|numeric multiset|all quantities|"
        r"every occurrence|correction notes?|source surface forms?)\b",
        flags=re.IGNORECASE,
    ),
    "checking": re.compile(
        r"(?:^|\b)(?:check(?:ing|ed)?|verify|verification|ensure|double[- ]check|"
        r"confirm(?:ing|ed)?|make sure|"
        r"all (?:quantities|dates|claims|directions|comparisons) "
        r"(?:are |remain )?(?:present|preserved)|good\.?$)",
        flags=re.IGNORECASE,
    ),
    "drafting": re.compile(
        r"\b(?:draft(?:ing|ed)?|rewrite|rephrase|reword|formulation|"
        r"one formal paragraph|one paragraph|single paragraph|word limit|"
        r"token limit|minutes[- ]style|formal (?:fomc )?minutes|formal tone|"
        r"passive voice|past tense|prose should|sentence should|fomc minutes)\b",
        flags=re.IGNORECASE,
    ),
    "instruction_recap": re.compile(
        r"\b(?:must|need(?:s)? to|should|do not|don['\u2019]?t|avoid|remove|"
        r"preserve|retain|include|mention|use exactly|must appear|no extra|"
        r"without adding|not add|never add|not introduce|no new facts?|"
        r"no extra information|"
        r"no rounding|no conversion|no derived quantities|use exact numbers?)\b",
        flags=re.IGNORECASE,
    ),
    "self_narration": re.compile(
        r"(?:^|\b)(?:let['\u2019]?s|we\s+(?:need|should|must|will|can)|"
        r"we['\u2019]?ll|so\s+we['\u2019]?ll|i['\u2019]?ll|"
        r"i\s+(?:need|should|must|will|can|have|haven['\u2019]?t)|"
        r"now\s+(?:craft|write|produce|check|rewrite))\b",
        flags=re.IGNORECASE,
    ),
    "citation_process": re.compile(
        r"\b(?:citations?|ev[- ]citations?|according to (?:the )?(?:data|source))\b",
        flags=re.IGNORECASE,
    ),
    "source_narration": re.compile(
        r"\b(?:(?:provided|source|original|supplied)\s+(?:analysis|text|passage)|"
        r"the analysis|this analysis)\b",
        flags=re.IGNORECASE,
    ),
    "evaluation_chatter": re.compile(
        r"\b(?:seems? faithful|that is faithful|this is faithful|this seems fine|"
        r"construct (?:a )?(?:cohesive )?paragraph|perhaps\s*:|"
        r"modal verb may|may is (?:not )?a month)",
        flags=re.IGNORECASE,
    ),
}
_DRAFT_PREFIX_RE = re.compile(
    r"^\s*(?:possible|potential|suggested|candidate|concise)\s+"
    r"(?:draft|formulation|wording|version|sentence|paragraph)\s*:\s*",
    flags=re.IGNORECASE,
)
_META_INVENTORY_PREFIX_RE = re.compile(
    r"^\s*(?:(?:the\s+)?(?:required\s+)?"
    r"(?:quantity occurrences?|quantities|date markers?|dates|directions?|"
    r"comparisons?|claims?)\s*(?:are|include|were)?|"
    r"(?:ensure|check|verify|preserve|retain)\b[^:]{0,180})\s*:\s*",
    flags=re.IGNORECASE,
)
_QUOTE_RE = re.compile(r"[\"\u201c](.{40,}?)[\"\u201d]", flags=re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])(?:[\"'\u201d\u2019)\]]*)\s+|\n+")


class LocalReasoningCleanError(Chk3DataError):
    """The local chk3 reasoning cleanup could not complete safely."""


@dataclass(frozen=True)
class PriorRow:
    sample_id: str
    output_split: str
    source_split: str
    source_index: int
    prompt: str
    analysis: str
    reasoning: str
    minutes: str
    response: str
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class CleanResult:
    reasoning: str
    matched_rules: tuple[str, ...]
    removed_units: int
    original_units: int


@dataclass(frozen=True)
class CleanRow:
    prior: PriorRow
    reasoning: str
    response: str
    reasoning_tokens: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    matched_rules: tuple[str, ...]
    recovery: Mapping[str, Any] | None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalReasoningCleanError(message)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalReasoningCleanError(f"invalid {label}: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object: {path}")
    return value


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing or unsafe {label}: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                _require(raw.strip() != "", f"blank {label} row: {path}:{line_number}")
                value = json.loads(raw)
                _require(
                    isinstance(value, dict),
                    f"non-object {label} row: {path}:{line_number}",
                )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise LocalReasoningCleanError(f"invalid {label}: {path}: {exc}") from exc
    return rows


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
    _atomic_write(path, "".join(canonical_json(dict(row)) + "\n" for row in rows))


def _safe_release_file(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    _require(
        relative != "" and not candidate.is_absolute() and ".." not in candidate.parts,
        f"unsafe release file path: {relative!r}",
    )
    path = root / candidate
    _require(path.is_file() and not path.is_symlink(), f"missing or unsafe release file: {relative}")
    return path


def verify_source_release(source_root: Path) -> dict[str, Any]:
    """Verify every file bound by the source release before reading rows."""

    _require(source_root.is_dir() and not source_root.is_symlink(), f"unsafe source release: {source_root}")
    handoff = _load_json(source_root / "handoff.json", label="source handoff")
    _require(handoff.get("immutable") is True, "source handoff is not immutable")
    _require(handoff.get("quality_status") == "passed", "source handoff did not pass")
    manifest_ref = handoff.get("release_manifest")
    _require(isinstance(manifest_ref, Mapping), "source handoff lacks manifest binding")
    _require(manifest_ref.get("path") == "release_manifest.json", "unexpected source manifest path")
    manifest_path = source_root / "release_manifest.json"
    _require(
        sha256_file(manifest_path) == manifest_ref.get("sha256"),
        "source release manifest hash mismatch",
    )
    manifest = _load_json(manifest_path, label="source release manifest")
    _require(manifest.get("immutable") is True, "source manifest is not immutable")
    _require(manifest.get("quality_status") == "passed", "source manifest did not pass")
    _require(
        handoff.get("release_id") == manifest.get("release_id"),
        "source release ID mismatch",
    )
    files = manifest.get("files")
    _require(isinstance(files, Mapping) and files, "source manifest has no file inventory")
    for relative, raw_record in files.items():
        _require(isinstance(relative, str), "source manifest has non-string file path")
        _require(isinstance(raw_record, Mapping), f"invalid source file record: {relative}")
        path = _safe_release_file(source_root, relative)
        _require(raw_record.get("path") == relative, f"source file path binding mismatch: {relative}")
        _require(sha256_file(path) == raw_record.get("sha256"), f"source file hash mismatch: {relative}")
        _require(path.stat().st_size == raw_record.get("bytes"), f"source file byte count mismatch: {relative}")
        if "rows" in raw_record:
            rows = sum(1 for _ in path.open(encoding="utf-8"))
            _require(rows == raw_record.get("rows"), f"source file row count mismatch: {relative}")
    audit_ref = handoff.get("data_quality_audit")
    _require(isinstance(audit_ref, Mapping), "source handoff lacks audit binding")
    audit_path = _safe_release_file(source_root, str(audit_ref.get("path") or ""))
    _require(sha256_file(audit_path) == audit_ref.get("sha256"), "source audit hash mismatch")
    return {"handoff": handoff, "manifest": manifest}


def _parse_analysis(prompt: str, *, sample_id: str) -> str:
    _require(prompt.startswith(USER_PROMPT_PREFIX), f"prompt contract mismatch: {sample_id}")
    try:
        payload = json.loads(prompt[len(USER_PROMPT_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise LocalReasoningCleanError(f"prompt JSON mismatch: {sample_id}: {exc}") from exc
    _require(
        isinstance(payload, dict) and set(payload) == {"analysis"},
        f"prompt schema mismatch: {sample_id}",
    )
    analysis = payload.get("analysis")
    _require(isinstance(analysis, str) and analysis.strip(), f"empty prompt analysis: {sample_id}")
    return analysis


def load_prior_rows(
    source_root: Path,
    *,
    expected_split_counts: Mapping[str, int],
) -> dict[str, list[PriorRow]]:
    result: dict[str, list[PriorRow]] = {}
    observed_ids: set[str] = set()
    for output_split, expected in expected_split_counts.items():
        _require(output_split in OUTPUT_TO_SOURCE_SPLIT, f"unsupported output split: {output_split}")
        source_split = OUTPUT_TO_SOURCE_SPLIT[output_split]
        data_path = source_root / "minutes_alignment" / f"{output_split}.jsonl"
        manifest_path = source_root / "minutes_alignment/manifests" / f"{output_split}.jsonl"
        training_rows = _load_jsonl(data_path, label=f"source {output_split} data")
        manifests = _load_jsonl(manifest_path, label=f"source {output_split} manifests")
        _require(
            len(training_rows) == expected and len(manifests) == expected,
            f"source split count mismatch: {output_split}",
        )
        output: list[PriorRow] = []
        for index, (training, manifest) in enumerate(zip(training_rows, manifests, strict=True)):
            _require(set(training) == {"prompt", "response"}, f"source training schema mismatch: {output_split}[{index}]")
            prompt = training.get("prompt")
            response = training.get("response")
            _require(isinstance(prompt, str) and prompt, f"empty source prompt: {output_split}[{index}]")
            _require(isinstance(response, str) and response, f"empty source response: {output_split}[{index}]")
            sample_id = str(manifest.get("sample_id") or "")
            _require(sample_id and sample_id not in observed_ids, f"invalid or duplicate sample ID: {sample_id!r}")
            observed_ids.add(sample_id)
            _require(manifest.get("split") == output_split, f"output split mismatch: {sample_id}")
            _require(manifest.get("source_split") == source_split, f"source split mismatch: {sample_id}")
            _require(manifest.get("source_index") == index, f"source order mismatch: {sample_id}")
            _require(response.count(BOUNDARY) == 1, f"response boundary mismatch: {sample_id}")
            reasoning, minutes = response.split(BOUNDARY, 1)
            _require(reasoning.strip() == reasoning and minutes.strip() == minutes, f"response edge whitespace mismatch: {sample_id}")
            _require(reasoning and minutes, f"empty reasoning or Minutes: {sample_id}")
            _require(not _CONTROL_RE.search(reasoning), f"control marker in source reasoning: {sample_id}")
            _require(not _CONTROL_RE.search(minutes), f"control marker in source Minutes: {sample_id}")
            analysis = _parse_analysis(prompt, sample_id=sample_id)
            expected_hashes = {
                "prompt_sha256": sha256_text(prompt),
                "response_sha256": sha256_text(response),
                "reasoning_sha256": sha256_text(reasoning),
                "minutes_sha256": sha256_text(minutes),
                "analysis_sha256": sha256_text(analysis),
            }
            for field, observed in expected_hashes.items():
                _require(manifest.get(field) == observed, f"source {field} mismatch: {sample_id}")
            source_response_sha = str(manifest.get("source_response_sha256") or "")
            _require(_SHA256_RE.fullmatch(source_response_sha) is not None, f"invalid source response hash: {sample_id}")
            output.append(
                PriorRow(
                    sample_id=sample_id,
                    output_split=output_split,
                    source_split=source_split,
                    source_index=index,
                    prompt=prompt,
                    analysis=analysis,
                    reasoning=reasoning,
                    minutes=minutes,
                    response=response,
                    manifest=manifest,
                )
            )
        result[output_split] = output
    _require(
        len(observed_ids) == sum(expected_split_counts.values()),
        "source population mismatch",
    )
    return result


def _normalized_segment(text: str) -> str:
    return " ".join(re.sub(r"[^\w%$.-]+", " ", text.casefold()).split())


def _quoted_draft(sentence: str, *, analysis: str, minutes: str) -> bool:
    references: list[str] = []
    for value in (analysis, minutes):
        if len(value) >= 40:
            references.append(_normalized_segment(value))
        references.extend(
            normalized
            for normalized in (
                _normalized_segment(unit)
                for unit in _SENTENCE_SPLIT_RE.split(value)
            )
            if len(normalized) >= 40
        )
    for match in _QUOTE_RE.finditer(sentence):
        quote = _normalized_segment(match.group(1))
        if len(quote) < 32:
            continue
        if any(
            quote == reference
            or quote in reference
            or reference in quote
            or difflib.SequenceMatcher(None, quote, reference, autojunk=False).ratio() >= 0.78
            for reference in references
        ):
            return True
    return False


def _meta_categories(text: str) -> set[str]:
    return {name for name, pattern in _META_PATTERNS.items() if pattern.search(text)}


def contamination_hits(
    reasoning: str,
    *,
    analysis: str,
    minutes: str,
) -> list[str]:
    hits: set[str] = set()
    if _EVIDENCE_ID_RE.search(reasoning):
        hits.add("evidence_id")
    if _EV_CITATION_RE.search(reasoning):
        hits.add("ev_citation_phrase")
    if _CONTROL_RE.search(reasoning):
        hits.add("control_marker")
    if reasoning.count('"') % 2 or reasoning.count("\u201c") != reasoning.count("\u201d"):
        hits.add("unmatched_quote")
    if any(_HEADING_ONLY_RE.match(line) or _HEADING_PREFIX_RE.match(line) for line in reasoning.splitlines()):
        hits.add("heading")
    if any(_BULLET_RE.match(line) for line in reasoning.splitlines()):
        hits.add("bullet_marker")
    hits.update(f"meta:{name}" for name in _meta_categories(reasoning))
    normalized_reasoning = _normalized_segment(reasoning)
    for label, value in (("source_analysis", analysis), ("final_minutes", minutes)):
        normalized_value = _normalized_segment(value)
        if len(normalized_value) >= 80 and normalized_value in normalized_reasoning:
            hits.add(f"quoted_or_repeated:{label}")
    seen: set[str] = set()
    for unit in _SENTENCE_SPLIT_RE.split(reasoning):
        normalized = _normalized_segment(unit)
        if len(normalized.split()) < 5:
            continue
        if normalized in seen:
            hits.add("duplicate_sentence")
        seen.add(normalized)
    return sorted(hits)


def _clean_reasoning_once(reasoning: str, *, analysis: str, minutes: str) -> CleanResult:
    """Delete local contamination without introducing any new factual prose."""

    text = str(reasoning).replace("\r\n", "\n").replace("\r", "\n").strip()
    rules: list[str] = []
    removed_units = 0
    if analysis and analysis in text:
        text = text.replace(analysis, " ")
        rules.append("quoted_source_analysis")
    if minutes and minutes in text:
        text = text.replace(minutes, " ")
        rules.append("quoted_final_minutes")

    citation_container = re.compile(
        r"\([^()\n]*(?:ev-[0-9a-f]+|ev[- ]citations?|evidence[- ]citations?)[^()\n]*\)"
        r"|\[[^\[\]\n]*(?:ev-[0-9a-f]+|ev[- ]citations?|evidence[- ]citations?)[^\[\]\n]*\]",
        flags=re.IGNORECASE,
    )
    text, containers = citation_container.subn(" ", text)
    if containers:
        rules.append("citation_clause")

    # Make compact numbered/bulleted lists visible to the unit-level cleaner.
    text = re.sub(r"(^|\s)(?=\d+[.)]\s+)", r"\1\n", text)
    text = re.sub(r"(?<=[.!?])\s+(?=[-*+\u2022]\s+)", "\n", text)
    raw_units = [unit.strip() for unit in _SENTENCE_SPLIT_RE.split(text) if unit.strip()]
    reference_fragments: set[str] = set()
    for reference in (analysis, minutes):
        for fragment in _SENTENCE_SPLIT_RE.split(reference):
            normalized_fragment = _normalized_segment(fragment)
            if len(normalized_fragment) >= 40:
                reference_fragments.add(normalized_fragment)
    kept: list[str] = []
    seen: set[str] = set()
    for raw_unit in raw_units:
        unit = raw_unit.strip()
        unquoted = re.sub(r'^["\u201c]+\s*', "", unit)
        unquoted = re.sub(r'\s*["\u201d]+$', "", unquoted)
        if unquoted != unit:
            rules.append("unmatched_quote")
            unit = unquoted
        if re.fullmatch(r"\d+[.)]", unit):
            rules.append("orphan_list_marker")
            removed_units += 1
            continue
        if _HEADING_ONLY_RE.fullmatch(unit):
            rules.append("heading")
            removed_units += 1
            continue
        prefixed = _HEADING_PREFIX_RE.sub("", unit)
        if prefixed != unit:
            rules.append("heading_marker")
            unit = prefixed.strip()
            if not unit:
                removed_units += 1
                continue
        unbulleted = _BULLET_RE.sub("", unit)
        if unbulleted != unit:
            rules.append("bullet_marker")
            unit = unbulleted.strip()

        if _quoted_draft(unit, analysis=analysis, minutes=minutes) or _DRAFT_PREFIX_RE.match(unit):
            rules.append("quoted_draft")
            removed_units += 1
            continue

        without_lead = _SOURCE_LEAD_RE.sub("", unit)
        if without_lead != unit:
            rules.append("source_narration_lead")
            unit = without_lead.strip()
        if not unit:
            removed_units += 1
            continue

        without_inventory_prefix = _META_INVENTORY_PREFIX_RE.sub("", unit)
        if without_inventory_prefix != unit and without_inventory_prefix.strip():
            rules.append("meta_inventory_prefix")
            unit = without_inventory_prefix.strip()

        normalized_unit = _normalized_segment(unit)
        normalized_analysis = _normalized_segment(analysis)
        normalized_minutes = _normalized_segment(minutes)
        if (
            len(normalized_analysis) >= 80
            and normalized_analysis in normalized_unit
        ):
            rules.append("quoted_source_analysis")
            removed_units += 1
            continue
        if (
            len(normalized_minutes) >= 80
            and normalized_minutes in normalized_unit
        ):
            rules.append("quoted_final_minutes")
            removed_units += 1
            continue
        if any(
            normalized_unit == fragment
            or (
                len(normalized_unit) >= 60
                and len(fragment) >= 60
                and difflib.SequenceMatcher(
                    None, normalized_unit, fragment, autojunk=False
                ).ratio()
                >= 0.96
            )
            for fragment in reference_fragments
        ):
            rules.append("quoted_source_or_final_fragment")
            removed_units += 1
            continue

        if _EV_CITATION_RE.search(unit):
            rules.append("citation_process_clause")
            removed_units += 1
            continue
        unit, evidence_ids = _EVIDENCE_ID_RE.subn("", unit)
        if evidence_ids:
            rules.append("evidence_id")
        unit = re.sub(r"\(\s*(?:[,;]\s*)*\)|\[\s*(?:[,;]\s*)*\]", "", unit)
        unit = re.sub(r"\s+([,.;:!?])", r"\1", unit)
        unit = " ".join(unit.split()).strip(" ,;:-")
        if not unit:
            removed_units += 1
            continue

        categories = _meta_categories(unit)
        if categories:
            rules.extend(f"meta_{category}" for category in sorted(categories))
            removed_units += 1
            continue

        normalized = _normalized_segment(unit)
        if not normalized:
            removed_units += 1
            continue
        if normalized in seen:
            rules.append("duplicate_sentence")
            removed_units += 1
            continue
        seen.add(normalized)
        kept.append(unit)

    # A deletion can expose a boundary that was not a sentence boundary in the
    # raw provider text.  Run one final exact-deduplication pass on the joined
    # result so the transform is idempotent.
    final_kept: list[str] = []
    final_seen: set[str] = set()
    for unit in _SENTENCE_SPLIT_RE.split(" ".join(kept).strip()):
        unit = _BULLET_RE.sub("", unit).strip()
        if not unit:
            continue
        normalized = _normalized_segment(unit)
        if normalized in final_seen:
            rules.append("duplicate_sentence")
            removed_units += 1
            continue
        final_seen.add(normalized)
        final_kept.append(unit)
    cleaned = " ".join(final_kept).strip()
    if cleaned.count('"') % 2:
        cleaned = cleaned.replace('"', "")
        rules.append("unmatched_quote")
    if cleaned.count("\u201c") != cleaned.count("\u201d"):
        cleaned = cleaned.replace("\u201c", "").replace("\u201d", "")
        rules.append("unmatched_quote")
    return CleanResult(
        reasoning=cleaned,
        matched_rules=tuple(dict.fromkeys(rules)),
        removed_units=removed_units,
        original_units=len(raw_units),
    )


def clean_reasoning(reasoning: str, *, analysis: str, minutes: str) -> CleanResult:
    """Apply deletion-only cleanup to a fixed point (normally one or two passes)."""

    current = str(reasoning)
    all_rules: list[str] = []
    removed_units = 0
    original_units = 0
    for pass_index in range(1, 7):
        result = _clean_reasoning_once(
            current,
            analysis=analysis,
            minutes=minutes,
        )
        if pass_index == 1:
            original_units = result.original_units
        all_rules.extend(result.matched_rules)
        removed_units += result.removed_units
        if result.reasoning == current.strip():
            return CleanResult(
                reasoning=result.reasoning,
                matched_rules=tuple(dict.fromkeys(all_rules)),
                removed_units=removed_units,
                original_units=original_units,
            )
        current = result.reasoning
    raise LocalReasoningCleanError("reasoning cleanup did not reach a fixed point")


def _token_count(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def _count_row(
    *,
    tokenizer: Any,
    system_prompt: str,
    prompt: str,
    response: str,
) -> tuple[int, int, int]:
    prompt_text = render_sft_prompt(
        tokenizer,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    eos = getattr(tokenizer, "eos_token", None)
    _require(isinstance(eos, str) and eos, "tokenizer has no EOS token")
    _require(eos not in response, "response already contains a literal EOS token")
    prompt_ids = tokenize_sft_text(tokenizer, prompt_text)
    full_ids = tokenize_sft_text(tokenizer, prompt_text + response + eos)
    _require(full_ids[: len(prompt_ids)] == prompt_ids, "prompt is not a token prefix of the SFT row")
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if bos_id is not None:
        _require(full_ids.count(int(bos_id)) == 1, "SFT row does not contain exactly one BOS")
    if eos_id is not None:
        _require(full_ids[-1] == int(eos_id), "SFT row does not end with EOS")
    return len(prompt_ids), len(full_ids) - len(prompt_ids), len(full_ids)


def _load_recovery_candidates(
    path: Path | None,
    *,
    tokenizer: Any,
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = _load_jsonl(path, label="reasoning recovery candidates")
    candidates: dict[str, dict[str, Any]] = {}
    required = {
        "schema_version",
        "sample_id",
        "split",
        "source_index",
        "prompt_sha256",
        "final_answer_sha256",
        "recovered_reasoning",
        "recovered_reasoning_sha256",
        "recovered_reasoning_tokens",
        "source_kind",
        "source_version",
        "source_path",
        "source_file_sha256",
        "source_row_locator",
        "source_response_sha256",
        "response_sha256",
        "meta_hits",
        "citation_hits",
        "selection_reason",
    }
    for row in rows:
        _require(set(row) == required, f"recovery candidate schema mismatch: {sorted(set(row) ^ required)}")
        _require(row.get("schema_version") == RECOVERY_SCHEMA_VERSION, "recovery schema version mismatch")
        sample_id = str(row.get("sample_id") or "")
        _require(sample_id and sample_id not in candidates, f"invalid or duplicate recovery candidate: {sample_id!r}")
        reasoning = row.get("recovered_reasoning")
        _require(isinstance(reasoning, str) and reasoning.strip() == reasoning and reasoning, f"invalid recovered reasoning: {sample_id}")
        _require(sha256_text(reasoning) == row.get("recovered_reasoning_sha256"), f"recovered reasoning hash mismatch: {sample_id}")
        tokens = _token_count(tokenizer, reasoning)
        _require(tokens == row.get("recovered_reasoning_tokens"), f"recovered reasoning token count mismatch: {sample_id}")
        _require(MIN_REASONING_TOKENS <= tokens <= MAX_REASONING_TOKENS, f"recovered reasoning token range mismatch: {sample_id}")
        _require(row.get("meta_hits") == [] and row.get("citation_hits") == [], f"recovery candidate declares contamination: {sample_id}")
        for field in ("prompt_sha256", "final_answer_sha256", "recovered_reasoning_sha256", "source_file_sha256", "source_response_sha256", "response_sha256"):
            _require(_SHA256_RE.fullmatch(str(row.get(field) or "")) is not None, f"invalid recovery hash {field}: {sample_id}")
        source_path = Path(str(row.get("source_path") or ""))
        if not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
        _require(source_path.is_file() and not source_path.is_symlink(), f"missing recovery source file: {sample_id}")
        _require(sha256_file(source_path) == row.get("source_file_sha256"), f"recovery source file hash mismatch: {sample_id}")
        candidates[sample_id] = row
    return candidates


def _validate_recovery_binding(candidate: Mapping[str, Any], prior: PriorRow) -> str:
    sample_id = prior.sample_id
    _require(candidate.get("split") == prior.output_split, f"recovery split mismatch: {sample_id}")
    _require(candidate.get("source_index") == prior.source_index, f"recovery index mismatch: {sample_id}")
    _require(candidate.get("prompt_sha256") == sha256_text(prior.prompt), f"recovery prompt binding mismatch: {sample_id}")
    _require(candidate.get("final_answer_sha256") == sha256_text(prior.minutes), f"recovery final-answer binding mismatch: {sample_id}")
    return str(candidate["recovered_reasoning"])


def _token_stats(values: Sequence[int]) -> dict[str, int]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"min": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}

    def at(fraction: float) -> int:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "min": ordered[0],
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def _row_manifest(row: CleanRow) -> dict[str, Any]:
    prior = row.prior
    manifest = dict(prior.manifest)
    manifest.update(
        {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "target_mode": (
                "local_reasoning_recovery_v4"
                if row.recovery is not None
                else "local_reasoning_clean_v4"
            ),
            "prompt_sha256": sha256_text(prior.prompt),
            "response_sha256": sha256_text(row.response),
            "reasoning_sha256": sha256_text(row.reasoning),
            "minutes_sha256": sha256_text(prior.minutes),
            "prompt_tokens": row.prompt_tokens,
            "completion_tokens": row.completion_tokens,
            "total_tokens": row.total_tokens,
            "reasoning_tokens": row.reasoning_tokens,
            "local_cleanup": {
                "schema_version": SCHEMA_VERSION,
                "source_response_sha256": sha256_text(prior.response),
                "source_reasoning_sha256": sha256_text(prior.reasoning),
                "matched_rules": list(row.matched_rules),
                "prompt_bytes_unchanged": True,
                "final_minutes_bytes_unchanged": True,
                "recovery_source": (
                    {
                        key: row.recovery[key]
                        for key in (
                            "source_kind",
                            "source_version",
                            "source_path",
                            "source_file_sha256",
                            "source_row_locator",
                            "source_response_sha256",
                            "response_sha256",
                            "selection_reason",
                        )
                    }
                    if row.recovery is not None
                    else None
                ),
            },
        }
    )
    return manifest


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _seal_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise LocalReasoningCleanError(f"staging contains a symlink: {path}")
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), key=lambda value: len(value.parts), reverse=True):
        path.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    root.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(renameat2 is not None, "atomic renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise LocalReasoningCleanError(f"release destination already exists: {destination}")
        raise LocalReasoningCleanError(f"renameat2 failed for {destination}: {os.strerror(error)}")


def _publish_release(
    *,
    source_root: Path,
    source_binding: Mapping[str, Any],
    release_parent: Path,
    release_id: str,
    clean_by_split: Mapping[str, Sequence[CleanRow]],
    expected_split_counts: Mapping[str, int],
    source_config: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    destination = release_parent / release_id
    _require(not destination.exists() and not destination.is_symlink(), f"release destination already exists: {destination}")
    release_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{release_id}.", dir=release_parent))
    sealed = False
    try:
        for split, expected in expected_split_counts.items():
            rows = list(clean_by_split[split])
            _require(len(rows) == expected, f"publish split count mismatch: {split}")
            _write_jsonl(
                staging / "minutes_alignment" / f"{split}.jsonl",
                [{"prompt": row.prior.prompt, "response": row.response} for row in rows],
            )
            _write_jsonl(
                staging / "minutes_alignment/manifests" / f"{split}.jsonl",
                [_row_manifest(row) for row in rows],
            )
        _write_json(staging / "audits/data_quality.json", audit)
        config = dict(source_config)
        config["dataset_name"] = f"dataset/processed/retrain_v2/{release_id}/minutes_alignment"
        _atomic_write(
            staging / "chk3_minutes_sft.template.yaml",
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        )
        file_records: dict[str, dict[str, Any]] = {}
        for path in sorted(staging.rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(staging))
            record: dict[str, Any] = {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            if path.suffix == ".jsonl":
                record["rows"] = sum(1 for _ in path.open(encoding="utf-8"))
            file_records[relative] = record
        created = _utc_now()
        release_manifest = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": release_id,
            "created_at_utc": created,
            "immutable": True,
            "quality_status": "passed",
            "dataset_role": "standalone_chk3_minutes_alignment_reasoning_clean",
            "dag_bindable": False,
            "dag_binding_blocker": "sealed chk2 parent and full chk2-derived release are not available",
            "training_mapping": "chk1 final analysis -> locally cleaned reasoning -> unchanged formal Minutes paragraph",
            "source": {
                "release_path": str(source_root),
                "handoff_sha256": sha256_file(source_root / "handoff.json"),
                "release_manifest_sha256": sha256_file(source_root / "release_manifest.json"),
                "release_id": source_binding["manifest"].get("release_id"),
                "transform_schema_version": SCHEMA_VERSION,
                "network_requests": 0,
            },
            "split_counts": dict(expected_split_counts),
            "total_rows": sum(expected_split_counts.values()),
            "files": file_records,
            "config_template": "chk3_minutes_sft.template.yaml",
        }
        _write_json(staging / "release_manifest.json", release_manifest)
        handoff = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": release_id,
            "created_at_utc": created,
            "immutable": True,
            "quality_status": "passed",
            "dataset_path": f"dataset/processed/retrain_v2/{release_id}/minutes_alignment",
            "split_counts": dict(expected_split_counts),
            "total_rows": sum(expected_split_counts.values()),
            "release_manifest": {
                "path": "release_manifest.json",
                "sha256": sha256_file(staging / "release_manifest.json"),
            },
            "data_quality_audit": {
                "path": "audits/data_quality.json",
                "sha256": sha256_file(staging / "audits/data_quality.json"),
            },
            "dag_bindable": False,
            "requires_sealed_parent": "chk2",
        }
        _write_json(staging / "handoff.json", handoff)
        _fsync_tree(staging)
        _seal_tree(staging)
        sealed = True
        _rename_noreplace(staging, destination)
        descriptor = os.open(release_parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return handoff
    finally:
        if staging.exists():
            if sealed:
                for path in staging.rglob("*"):
                    if path.is_dir():
                        path.chmod(0o700)
                    elif path.is_file():
                        path.chmod(0o600)
                staging.chmod(0o700)
            shutil.rmtree(staging)


def run(
    *,
    source_release: Path,
    tokenizer_path: Path,
    work_root: Path,
    release_parent: Path,
    release_id: str,
    recovery_candidates: Path | None = None,
    dry_run: bool = False,
    tokenizer: Any | None = None,
    expected_split_counts: Mapping[str, int] = EXPECTED_SPLIT_COUNTS,
) -> dict[str, Any]:
    _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", release_id) is not None, "invalid release ID")
    _require(sum(expected_split_counts.values()) > 0, "expected split population is empty")
    source_binding = verify_source_release(source_release)
    prior_by_split = load_prior_rows(source_release, expected_split_counts=expected_split_counts)
    source_config_path = source_release / "chk3_minutes_sft.template.yaml"
    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    _require(isinstance(source_config, dict), "source chk3 config is invalid")
    _require(source_config.get("system_prompt") == STUDENT_SYSTEM_PROMPT, "source system prompt contract mismatch")
    active_tokenizer = tokenizer or _load_tokenizer(tokenizer_path)
    recovery_by_id = _load_recovery_candidates(recovery_candidates, tokenizer=active_tokenizer)

    clean_by_split: dict[str, list[CleanRow]] = {split: [] for split in expected_split_counts}
    needs: list[dict[str, Any]] = []
    cleanup_manifests: dict[str, list[dict[str, Any]]] = {split: [] for split in expected_split_counts}
    rule_counts: Counter[str] = Counter()
    original_reasoning_tokens: list[int] = []
    cleaned_reasoning_tokens: list[int] = []
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    total_tokens: list[int] = []
    changed_rows = 0
    recovered_rows = 0

    for split in expected_split_counts:
        for prior in prior_by_split[split]:
            local = clean_reasoning(prior.reasoning, analysis=prior.analysis, minutes=prior.minutes)
            original_tokens = _token_count(active_tokenizer, prior.reasoning)
            local_tokens = _token_count(active_tokenizer, local.reasoning)
            original_reasoning_tokens.append(original_tokens)
            selected = local.reasoning
            selected_rules = local.matched_rules
            recovery = recovery_by_id.get(prior.sample_id)
            if local_tokens < MIN_REASONING_TOKENS and recovery is not None:
                selected = _validate_recovery_binding(recovery, prior)
                recovered_clean = clean_reasoning(selected, analysis=prior.analysis, minutes=prior.minutes)
                _require(recovered_clean.reasoning == selected, f"recovered reasoning is not cleanup-idempotent: {prior.sample_id}")
                selected_rules = tuple(dict.fromkeys((*local.matched_rules, "bound_local_recovery")))
                recovered_rows += 1
            elif recovery is not None:
                raise LocalReasoningCleanError(f"unexpected recovery candidate for non-undersized row: {prior.sample_id}")

            selected_tokens = _token_count(active_tokenizer, selected)
            response = f"{selected}{BOUNDARY}{prior.minutes}"
            hits = contamination_hits(selected, analysis=prior.analysis, minutes=prior.minutes)
            failure_reasons: list[str] = []
            if selected_tokens < MIN_REASONING_TOKENS:
                failure_reasons.append(f"reasoning_tokens:{selected_tokens}<{MIN_REASONING_TOKENS}")
            if selected_tokens > MAX_REASONING_TOKENS:
                failure_reasons.append(f"reasoning_tokens:{selected_tokens}>{MAX_REASONING_TOKENS}")
            if hits:
                failure_reasons.append("residual_contamination:" + ",".join(hits))
            if failure_reasons:
                needs.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "status": "needs_regeneration",
                        "sample_id": prior.sample_id,
                        "split": prior.output_split,
                        "source_index": prior.source_index,
                        "prompt_sha256": sha256_text(prior.prompt),
                        "final_answer_sha256": sha256_text(prior.minutes),
                        "minutes_sha256": sha256_text(prior.minutes),
                        "source_response_sha256": str(prior.manifest["source_response_sha256"]),
                        "original_response_sha256": sha256_text(prior.response),
                        "original_reasoning_sha256": sha256_text(prior.reasoning),
                        "cleaned_reasoning_sha256": sha256_text(selected),
                        "original_reasoning_tokens": original_tokens,
                        "cleaned_reasoning_tokens": selected_tokens,
                        "failure_reason": ";".join(failure_reasons),
                        "matched_cleanup_rules": list(selected_rules),
                    }
                )
                continue

            _require(response.count(BOUNDARY) == 1, f"clean response boundary mismatch: {prior.sample_id}")
            _require(response.split(BOUNDARY, 1)[1] == prior.minutes, f"final Minutes changed: {prior.sample_id}")
            p_tokens, c_tokens, t_tokens = _count_row(
                tokenizer=active_tokenizer,
                system_prompt=str(source_config["system_prompt"]),
                prompt=prior.prompt,
                response=response,
            )
            _require(p_tokens <= PROMPT_TOKEN_LIMIT, f"prompt token overflow: {prior.sample_id}")
            _require(t_tokens <= TOTAL_TOKEN_LIMIT, f"total token overflow: {prior.sample_id}")
            clean_row = CleanRow(
                prior=prior,
                reasoning=selected,
                response=response,
                reasoning_tokens=selected_tokens,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=t_tokens,
                matched_rules=selected_rules,
                recovery=recovery,
            )
            clean_by_split[split].append(clean_row)
            cleanup_manifests[split].append(_row_manifest(clean_row))
            cleaned_reasoning_tokens.append(selected_tokens)
            prompt_tokens.append(p_tokens)
            completion_tokens.append(c_tokens)
            total_tokens.append(t_tokens)
            rule_counts.update(selected_rules)
            if selected != prior.reasoning:
                changed_rows += 1

    unused_recoveries = sorted(set(recovery_by_id) - {row.prior.sample_id for rows in clean_by_split.values() for row in rows})
    _require(not unused_recoveries, f"unused recovery candidates: {unused_recoveries[:5]}")

    work_root.mkdir(parents=True, exist_ok=True)
    for split in expected_split_counts:
        _write_jsonl(
            work_root / "candidate_rows" / f"{split}.jsonl",
            [
                {"prompt": row.prior.prompt, "response": row.response}
                for row in clean_by_split[split]
            ],
        )
        _write_jsonl(
            work_root / "candidate_manifests" / f"{split}.jsonl",
            cleanup_manifests[split],
        )
    _write_jsonl(work_root / "needs_regeneration.jsonl", needs)

    total_expected = sum(expected_split_counts.values())
    status = "needs_regeneration" if needs else ("ready" if dry_run else "publishing")
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "release_id": release_id,
        "source_release": str(source_release),
        "source_release_manifest_sha256": sha256_file(source_release / "release_manifest.json"),
        "total_rows": total_expected,
        "accepted_rows": total_expected - len(needs),
        "needs_regeneration_rows": len(needs),
        "changed_rows": changed_rows,
        "recovered_rows": recovered_rows,
        "network_requests": 0,
        "prompt_bytes_changed": 0,
        "final_minutes_bytes_changed": 0,
        "rule_counts": dict(sorted(rule_counts.items())),
        "token_stats": {
            "original_reasoning": _token_stats(original_reasoning_tokens),
            "cleaned_reasoning": _token_stats(cleaned_reasoning_tokens),
            "prompt": _token_stats(prompt_tokens),
            "completion": _token_stats(completion_tokens),
            "total": _token_stats(total_tokens),
        },
        "checks": {
            "source_manifest_integrity": "passed",
            "population_and_order": "passed",
            "prompt_byte_identity": "passed",
            "final_minutes_byte_identity": "passed",
            "reasoning_contamination": "blocked" if needs else "passed",
            "reasoning_token_floor": "blocked" if needs else "passed",
            "single_bos_final_eos": "passed_for_accepted_rows",
            "total_token_budget": "passed_for_accepted_rows",
            "publication": "blocked" if needs else ("not_requested" if dry_run else "pending"),
        },
    }
    _write_json(work_root / "summary.json", summary)
    if needs or dry_run:
        return summary

    audit = {
        **summary,
        "status": "passed",
        "quality_status": "passed",
        "accepted_rows": total_expected,
        "needs_regeneration_rows": 0,
        "checks": {**summary["checks"], "publication": "passed"},
    }
    _publish_release(
        source_root=source_release,
        source_binding=source_binding,
        release_parent=release_parent,
        release_id=release_id,
        clean_by_split=clean_by_split,
        expected_split_counts=expected_split_counts,
        source_config=source_config,
        audit=audit,
    )
    summary.update(
        {
            "status": "complete",
            "release_path": str(release_parent / release_id),
            "handoff": str(release_parent / release_id / "handoff.json"),
            "checks": {**summary["checks"], "publication": "passed"},
        }
    )
    _write_json(work_root / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, default=DEFAULT_SOURCE_RELEASE)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--release-parent", type=Path, default=DEFAULT_RELEASE_PARENT)
    parser.add_argument("--release-id", default=DEFAULT_RELEASE_ID)
    parser.add_argument("--recovery-candidates", type=Path)
    # Publication is intentionally unavailable from the CLI until every
    # needs-regeneration row has a separately reviewed recovery ledger.
    parser.add_argument("--dry-run", action="store_true", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(
            source_release=args.source_release.resolve(),
            tokenizer_path=args.tokenizer_path.resolve(),
            work_root=args.work_root.resolve(),
            release_parent=args.release_parent.resolve(),
            release_id=args.release_id,
            recovery_candidates=(
                args.recovery_candidates.resolve()
                if args.recovery_candidates is not None
                else None
            ),
            dry_run=args.dry_run,
        )
    except (Chk3DataError, OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 3 if summary.get("status") == "needs_regeneration" else 0


if __name__ == "__main__":
    raise SystemExit(main())
