"""Repair and immutably publish the locally recoverable chk1 SFT release.

This program is deliberately offline.  It joins the sealed compressed rows to
the canonical generation manifests by ``sample_id`` and content hashes, repairs
only the known transport contamination, and fails closed on any drift in the
expected population.  Publication requires a separate, completed source-only
semantic audit; building a deterministic candidate can never create a passed
release by itself.
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
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPLITS = ("train", "eval", "test")
BOUNDARY = "\n</think>\n"

CANDIDATE_SCHEMA_VERSION = "chk1-clean-sft-candidate-v2"
RELEASE_SCHEMA_VERSION = "chk1-clean-sft-release-v2"
REPAIR_MANIFEST_SCHEMA_VERSION = "chk1-sft-contamination-repair-v2"
SEMANTIC_AUDIT_INPUT_SCHEMA_VERSION = "chk1-clean-sft-semantic-audit-input-v1"
SEMANTIC_AUDIT_SCHEMA_VERSION = "chk1-clean-sft-source-audit-v1"
DETERMINISTIC_VALIDATION_SCHEMA_VERSION = "chk1-clean-sft-validator-v1"
DETERMINISTIC_DATA_CONTRACT_VERSION = "chk1-clean-sft-data-contract-v2"
DETERMINISTIC_TOKEN_CONTRACT_VERSION = "chk1-sft-single-bos-completion-mask-v1"

DEFAULT_SOURCE_RELEASE = Path(
    "dataset/processed/retrain_v2/"
    "chk1_reasoning_compressed_flash_max_v1_20260805"
)
DEFAULT_BASE_RELEASE = Path(
    "dataset/processed/retrain_v2/"
    "analysis_base_full_v7_automated_v3_20260804"
)
DEFAULT_GENERATION_DIR = Path("output/data/retrain_v2/chk1/generation_full_v7")
DEFAULT_TEACHER_CACHE = Path("output/data/retrain_v2/chk1/deepseek_teacher_cache_v5")
DEFAULT_TOKENIZER = DEFAULT_BASE_RELEASE / "provenance/tokenizer_bundle"
DEFAULT_RELEASE = Path(
    "dataset/processed/retrain_v2/"
    "chk1_reasoning_compressed_flash_max_v2_clean_20260809"
)

EXPECTED_SPLIT_COUNTS = {"train": 1354, "eval": 199, "test": 190}
EXPECTED_ANSWER_METHOD_COUNTS = {
    "unchanged": 1542,
    "strip_inline_evidence_ids": 159,
    "recover_nested_content_answer": 4,
    "recover_content_answer_string": 24,
    "recover_reasoning_final_answer_json": 11,
    "recover_marked_final_answer_quote": 3,
}
EXPECTED_REASONING_REPAIR_COUNTS = {
    "replace_fact_card_meta": 37,
    "replace_provided_data_meta": 3,
    "rewrite_fixed_final_answer_meta": 1,
    "drop_repeated_answer_tail": 1,
}
EXPECTED_CHANGED_ROWS = 237
SEMANTIC_BLOCKING_KINDS = {"factual", "numerical", "causal", "target_leakage"}
SEMANTIC_JUDGE_MODEL = "Qwen3.5-9B"
SEMANTIC_RUBRIC_KEYS = {
    "data_fidelity",
    "trend_reasoning",
    "policy_relevance",
    "uncertainty_calibration",
    "fomc_style",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_EVIDENCE_ID_RE = re.compile(r"\bev-[0-9a-z]+\b", flags=re.IGNORECASE)
_EVIDENCE_META_RE = re.compile(r"\bevidence[_ ]ids?\b", flags=re.IGNORECASE)
_JSON_KEY_RE = re.compile(r'"(?:analysis|answer|content|reasoning_content)"\s*:', re.I)
_CONTROL_TAG_RE = re.compile(r"</?(?:think|answer)>|<\|[^>]+\|>", re.I)
_MARKDOWN_RE = re.compile(
    r"(?:```|~~~|^\s{0,3}#{1,6}\s|^\s*(?:[-*+] |\d+[.)] )|\[[^\]]+\]\([^)]+\))",
    flags=re.MULTILINE,
)
_ANSWER_META_RE = re.compile(
    r"\b(?:fact card|provided data|fixed final answer|source prompt|json schema|"
    r"schema field|output field)\b",
    flags=re.IGNORECASE,
)


class RepairError(RuntimeError):
    """The clean-release repair or publication contract was violated."""


@dataclass(frozen=True)
class RepairExpectations:
    split_counts: Mapping[str, int]
    answer_method_counts: Mapping[str, int]
    reasoning_repair_counts: Mapping[str, int]
    changed_rows: int

    @classmethod
    def production(cls) -> "RepairExpectations":
        return cls(
            split_counts=EXPECTED_SPLIT_COUNTS,
            answer_method_counts=EXPECTED_ANSWER_METHOD_COUNTS,
            reasoning_repair_counts=EXPECTED_REASONING_REPAIR_COUNTS,
            changed_rows=EXPECTED_CHANGED_ROWS,
        )


@dataclass(frozen=True)
class RepairInputs:
    source_release: Path
    base_release: Path
    generation_dir: Path
    teacher_cache: Path
    tokenizer_path: Path

    def resolved(self) -> "RepairInputs":
        return RepairInputs(
            source_release=self.source_release.resolve(),
            base_release=self.base_release.resolve(),
            generation_dir=self.generation_dir.resolve(),
            teacher_cache=self.teacher_cache.resolve(),
            tokenizer_path=self.tokenizer_path.resolve(),
        )


@dataclass(frozen=True)
class GenerationRecord:
    split: str
    sample_id: str
    prompt_sha256: str
    provided_data_sha256: str
    response_sha256: str
    final_analysis_sha256: str
    cache_key: str
    generator_prompt_sha256: str
    generation_provenance_sha256: str
    manifest_path: Path
    manifest_line_number: int


@dataclass(frozen=True)
class CandidatePayload:
    split_rows: Mapping[str, tuple[Mapping[str, str], ...]]
    repair_records: tuple[Mapping[str, Any], ...]
    semantic_audit_rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RepairError(message)


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "")
    _require(_SHA256_RE.fullmatch(text) is not None, f"{label} is not SHA-256")
    return text


def _read_json(path: Path) -> Mapping[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing regular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RepairError(f"invalid JSON file {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing regular JSONL file: {path}")
    rows: list[Mapping[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RepairError(f"cannot read JSONL file {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        _require(bool(line), f"blank JSONL row: {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RepairError(f"invalid JSONL row {path}:{line_number}: {exc}") from exc
        _require(isinstance(value, dict), f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment contract
        raise RepairError("transformers is required for the chk0 tokenizer gate") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - tokenizer loaders vary
        raise RepairError(f"cannot load local chk0 tokenizer {path}: {exc}") from exc
    _require(getattr(tokenizer, "eos_token_id", None) is not None, "tokenizer lacks EOS")
    return tokenizer


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        values = tokenizer.encode(text, add_special_tokens=False)
    except Exception as exc:  # noqa: BLE001 - tokenizer implementations vary
        raise RepairError(f"tokenization failed: {exc}") from exc
    _require(
        isinstance(values, list) and all(isinstance(item, int) for item in values),
        "tokenizer returned invalid token IDs",
    )
    return values


def is_json_like_answer(answer: str) -> bool:
    text = answer.lstrip()
    return text.startswith(("{", "[")) or _JSON_KEY_RE.search(text[:300]) is not None


def _json_answer_strings(text: str) -> list[str]:
    values: list[str] = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r'"answer"\s*:\s*', text):
        try:
            value, _end = decoder.raw_decode(text, match.end())
        except json.JSONDecodeError:
            continue
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return values


def _nested_content_answer(raw_content: str) -> str | None:
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    content = payload.get("content")
    if isinstance(content, dict):
        answer = content.get("answer")
        return answer.strip() if isinstance(answer, str) and answer.strip() else None
    if isinstance(content, str):
        try:
            nested = json.loads(content)
        except json.JSONDecodeError:
            return None
        if isinstance(nested, dict):
            answer = nested.get("answer")
            return answer.strip() if isinstance(answer, str) and answer.strip() else None
    return None


def _marked_final_answer_quote(reasoning: str) -> str | None:
    markers = re.compile(
        r"(?im)^(?:i(?:'|’)ll (?:craft the answer|write)|"
        r"final answer paragraph|final answer)\s*:\s*$"
    )
    decoder = json.JSONDecoder()
    candidates: list[str] = []
    for match in markers.finditer(reasoning):
        tail = reasoning[match.end() :].lstrip()
        if not tail.startswith('"'):
            continue
        try:
            value, _end = decoder.raw_decode(tail)
        except json.JSONDecodeError:
            continue
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    meaningful = [value for value in candidates if len(value) >= 16]
    _require(len(meaningful) <= 1, "ambiguous marked final-answer quotations in cache")
    return meaningful[0] if meaningful else None


def recover_json_like_answer(raw_content: str, raw_reasoning: str) -> tuple[str, str]:
    """Recover one explicit final answer using the fixed local-source priority."""

    nested = _nested_content_answer(raw_content)
    if nested is not None:
        return "recover_nested_content_answer", nested

    content_values = [value for value in _json_answer_strings(raw_content) if len(value) >= 16]
    _require(len(content_values) <= 1, "ambiguous answer strings in provider content")
    if content_values:
        return "recover_content_answer_string", content_values[0]

    reasoning_values = [
        value for value in _json_answer_strings(raw_reasoning) if len(value) >= 16
    ]
    _require(len(reasoning_values) <= 1, "ambiguous final-answer JSON in provider reasoning")
    if reasoning_values:
        return "recover_reasoning_final_answer_json", reasoning_values[0]

    quoted = _marked_final_answer_quote(raw_reasoning)
    if quoted is not None:
        return "recover_marked_final_answer_quote", quoted
    raise RepairError("JSON-like answer has no unambiguous local recovery source")


def strip_inline_evidence_ids(text: str) -> tuple[str, int]:
    """Remove internal evidence citations without rewriting the surrounding prose."""

    matches = list(_EVIDENCE_ID_RE.finditer(text))

    def clean_container(match: re.Match[str]) -> str:
        opening = match.group(0)[0]
        closing = match.group(0)[-1]
        inner = _EVIDENCE_ID_RE.sub("", match.group(0)[1:-1])
        # An ID can be the first, last, or middle member of a mixed factual
        # parenthetical.  Remove only separators orphaned by that ID; keep all
        # remaining words and numbers byte-for-byte apart from whitespace.
        inner = re.sub(r"^(?:\s*[,;]\s*)+", "", inner)
        inner = re.sub(r"(?:\s*[,;]\s*)+$", "", inner)
        inner = re.sub(r"([,;])(?:\s*[,;])+", r"\1", inner)
        inner = inner.strip()
        return f"{opening}{inner}{closing}" if inner else ""

    container_pattern = re.compile(
        r"\([^()]*\bev-[0-9a-z]+\b[^()]*\)|"
        r"\[[^\[\]]*\bev-[0-9a-z]+\b[^\[\]]*\]",
        flags=re.IGNORECASE,
    )
    cleaned = container_pattern.sub(clean_container, text)
    cleaned = _EVIDENCE_ID_RE.sub("", cleaned)
    cleaned = re.sub(r"\(\s*[,;/|&-]*\s*\)", "", cleaned)
    cleaned = re.sub(r"\[\s*[,;/|&-]*\s*\]", "", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", cleaned)
    cleaned = re.sub(r"[,;:]\s*([.!?])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    cleaned = " ".join(cleaned.split()).strip()
    return cleaned, len(matches)


def _replace_meta_phrase(
    text: str, pattern: re.Pattern[str], *, with_article: bool
) -> tuple[str, int]:
    def replacement(match: re.Match[str]) -> str:
        value = "the available evidence" if with_article else "available evidence"
        if match.group(0)[0].isupper():
            value = value[0].upper() + value[1:]
        return value

    return pattern.subn(replacement, text)


def clean_reasoning(reasoning: str, source_answer: str) -> tuple[str, tuple[str, ...]]:
    """Remove the four known reasoning transport artifacts, fail-closed."""

    cleaned = reasoning.strip()
    repairs: list[str] = []

    # A single known row repeats its final answer inside the last reasoning
    # paragraph.  Drop that whole summarizing paragraph, rather than slicing a
    # sentence out of it and leaving a semantically broken tail.
    paragraphs = re.split(r"\n\s*\n", cleaned)
    containing = (
        [index for index, paragraph in enumerate(paragraphs) if source_answer in paragraph]
        if len(source_answer) >= 16 and not is_json_like_answer(source_answer)
        else []
    )
    if containing:
        _require(
            containing == [len(paragraphs) - 1],
            "answer reproduction is not confined to one final reasoning paragraph",
        )
        paragraphs.pop()
        cleaned = "\n\n".join(paragraphs).strip()
        repairs.append("drop_repeated_answer_tail")

    fact_patterns = (
        (re.compile(r"\bthe\s+(?:available\s+)?fact card\b", re.I), True),
        (re.compile(r"\b(?:available\s+)?fact card\b", re.I), False),
    )
    fact_replacements = 0
    for pattern, with_article in fact_patterns:
        cleaned, count = _replace_meta_phrase(cleaned, pattern, with_article=with_article)
        fact_replacements += count
    if fact_replacements:
        repairs.append("replace_fact_card_meta")

    provided_patterns = (
        (re.compile(r"\bthe\s+provided data\b", re.I), True),
        (re.compile(r"\bprovided data\b", re.I), False),
    )
    provided_replacements = 0
    for pattern, with_article in provided_patterns:
        cleaned, count = _replace_meta_phrase(cleaned, pattern, with_article=with_article)
        provided_replacements += count
    if provided_replacements:
        repairs.append("replace_provided_data_meta")

    fixed_pattern = re.compile(
        r"Based on these observations, the fixed final answer is supported:\s*",
        re.I,
    )
    cleaned, fixed_count = fixed_pattern.subn(
        "Taken together, the available evidence indicates that ", cleaned
    )
    _require(fixed_count <= 1, "multiple fixed-final-answer meta phrases")
    if fixed_count:
        repairs.append("rewrite_fixed_final_answer_meta")

    _require(
        re.search(r"\b(?:fact card|provided data|fixed final answer)\b", cleaned, re.I)
        is None,
        "reasoning meta cleanup left a banned phrase",
    )
    _require(BOUNDARY not in cleaned, "reasoning contains the native boundary")
    _require("<think>" not in cleaned.lower(), "reasoning contains an opening think tag")
    return cleaned, tuple(repairs)


def validate_answer(answer: str, tokenizer: Any) -> int:
    _require(bool(answer.strip()), "answer is empty")
    _require(answer == answer.strip(), "answer has surrounding whitespace")
    _require("\n" not in answer and "\r" not in answer, "answer is not one plain-text paragraph")
    _require(not is_json_like_answer(answer), "answer is JSON-like")
    _require(_EVIDENCE_ID_RE.search(answer) is None, "answer contains an evidence ID")
    _require(_EVIDENCE_META_RE.search(answer) is None, "answer contains evidence-ID metadata")
    _require(_CONTROL_TAG_RE.search(answer) is None, "answer contains a control tag")
    _require(_MARKDOWN_RE.search(answer) is None, "answer contains Markdown")
    _require(_ANSWER_META_RE.search(answer) is None, "answer contains schema/task metadata")
    count = len(_token_ids(tokenizer, answer))
    _require(16 <= count <= 512, f"answer token count is outside 16..512: {count}")
    return count


def _has_strict_periodic_tail(token_ids: Sequence[int]) -> bool:
    """Detect a literal repeated suffix of at least 48 tokens and 3 periods."""

    size = len(token_ids)
    for period in range(1, min(128, size // 3) + 1):
        if period * 3 < 48:
            continue
        unit = list(token_ids[size - period :])
        repeats = 1
        cursor = size - 2 * period
        while cursor >= 0 and list(token_ids[cursor : cursor + period]) == unit:
            repeats += 1
            cursor -= period
        if repeats >= 3 and repeats * period >= 48:
            return True
    return False


def _load_generation_index(
    generation_dir: Path,
) -> tuple[dict[tuple[str, str, str], GenerationRecord], Mapping[str, Any]]:
    handoff_path = generation_dir / "generation_handoff.json"
    handoff = _read_json(handoff_path)
    _require(handoff.get("status") == "complete", "generation handoff is not complete")
    index: dict[tuple[str, str, str], GenerationRecord] = {}
    seen_sample_ids: set[str] = set()
    manifest_files: dict[str, Mapping[str, Any]] = {}
    for split in SPLITS:
        path = generation_dir / "manifests" / f"{split}.jsonl"
        rows = _read_jsonl(path)
        manifest_files[split] = {
            "path": str(path),
            "rows": len(rows),
            "sha256": sha256_file(path),
        }
        for line_number, row in enumerate(rows, 1):
            sample_id = str(row.get("sample_id") or "")
            _require(sample_id and sample_id not in seen_sample_ids, f"duplicate sample_id: {sample_id}")
            seen_sample_ids.add(sample_id)
            _require(row.get("split") == split, f"generation split mismatch: {sample_id}")
            generation = row.get("generation")
            _require(isinstance(generation, dict), f"generation metadata missing: {sample_id}")
            cache_key = _require_sha256(generation.get("cache_key"), f"{sample_id}.cache_key")
            projection = generation.get("model_input_projection")
            _require(isinstance(projection, dict), f"model projection missing: {sample_id}")
            record = GenerationRecord(
                split=split,
                sample_id=sample_id,
                prompt_sha256=_require_sha256(row.get("prompt_sha256"), f"{sample_id}.prompt"),
                provided_data_sha256=_require_sha256(
                    row.get("provided_data_sha256"), f"{sample_id}.provided_data"
                ),
                response_sha256=_require_sha256(
                    row.get("response_sha256"), f"{sample_id}.response"
                ),
                final_analysis_sha256=_require_sha256(
                    row.get("final_analysis_sha256"), f"{sample_id}.final_analysis"
                ),
                cache_key=cache_key,
                generator_prompt_sha256=_require_sha256(
                    projection.get("generator_prompt_sha256"),
                    f"{sample_id}.generator_prompt",
                ),
                generation_provenance_sha256=_require_sha256(
                    generation.get("generation_provenance_sha256"),
                    f"{sample_id}.generation_provenance",
                ),
                manifest_path=path,
                manifest_line_number=line_number,
            )
            key = (split, record.prompt_sha256, record.provided_data_sha256)
            _require(key not in index, f"ambiguous generation hash join: {key}")
            index[key] = record
    sources = {
        "generation_handoff": {
            "path": str(handoff_path),
            "sha256": sha256_file(handoff_path),
        },
        "generation_manifests": manifest_files,
    }
    return index, sources


def _load_dataset_index(
    dataset_dir: Path, *, label: str
) -> tuple[dict[tuple[str, str, str], tuple[Mapping[str, str], int]], Mapping[str, Any]]:
    index: dict[tuple[str, str, str], tuple[Mapping[str, str], int]] = {}
    files: dict[str, Mapping[str, Any]] = {}
    for split in SPLITS:
        path = dataset_dir / f"{split}.jsonl"
        rows = _read_jsonl(path)
        files[split] = {"path": str(path), "rows": len(rows), "sha256": sha256_file(path)}
        for line_number, raw in enumerate(rows, 1):
            _require(
                set(raw) == {"prompt", "response", "provided_data"},
                f"{label} row schema mismatch: {split}:{line_number}",
            )
            row = {key: str(raw[key]) for key in ("prompt", "response", "provided_data")}
            key = (split, sha256_text(row["prompt"]), sha256_text(row["provided_data"]))
            _require(key not in index, f"duplicate {label} hash identity: {split}:{line_number}")
            index[key] = (row, line_number)
    return index, files


def _load_teacher_cache(inputs: RepairInputs, record: GenerationRecord) -> tuple[Mapping[str, Any], Path, str]:
    path = inputs.teacher_cache / record.cache_key[:2] / f"{record.cache_key}.json"
    payload = _read_json(path)
    _require(payload.get("cache_key") == record.cache_key, f"teacher cache key mismatch: {record.sample_id}")
    _require(
        payload.get("prompt_sha256") == record.generator_prompt_sha256,
        f"teacher cache prompt mismatch: {record.sample_id}",
    )
    _require(
        payload.get("generation_provenance_sha256")
        == record.generation_provenance_sha256,
        f"teacher cache provenance mismatch: {record.sample_id}",
    )
    provider_raw = payload.get("provider_raw")
    _require(isinstance(provider_raw, dict), f"provider_raw missing: {record.sample_id}")
    _require(
        isinstance(provider_raw.get("content"), str)
        and isinstance(provider_raw.get("reasoning_content"), str),
        f"provider_raw is incomplete: {record.sample_id}",
    )
    return payload, path, sha256_file(path)


def _split_native_response(response: str, *, label: str) -> tuple[str, str]:
    _require(response.count(BOUNDARY) == 1, f"{label}: expected exactly one native boundary")
    reasoning, answer = response.split(BOUNDARY, 1)
    _require(bool(reasoning.strip()), f"{label}: empty reasoning")
    return reasoning.strip(), answer.strip()


def _assert_expected_counter(
    actual: Mapping[str, int], expected: Mapping[str, int], *, label: str
) -> None:
    normalized = {key: int(value) for key, value in actual.items() if int(value)}
    wanted = {key: int(value) for key, value in expected.items() if int(value)}
    _require(normalized == wanted, f"{label} drift: expected {wanted}, got {normalized}")


def _candidate_sources(
    inputs: RepairInputs,
    *,
    source_files: Mapping[str, Any],
    base_files: Mapping[str, Any],
    generation_sources: Mapping[str, Any],
    recovery_cache_files: Sequence[Mapping[str, str]],
) -> Mapping[str, Any]:
    source_manifest = inputs.source_release / "release_manifest.json"
    base_manifest = inputs.base_release / "base_release_manifest.json"
    cache_binding = [
        {"sample_id": row["sample_id"], "sha256": row["sha256"]}
        for row in sorted(recovery_cache_files, key=lambda item: item["sample_id"])
    ]
    return {
        "source_release_manifest": {
            "path": str(source_manifest),
            "sha256": sha256_file(source_manifest),
        },
        "source_split_files": source_files,
        "base_release_manifest": {
            "path": str(base_manifest),
            "sha256": sha256_file(base_manifest),
        },
        "base_split_files": base_files,
        **generation_sources,
        "teacher_cache": {
            "root": str(inputs.teacher_cache),
            "recovery_files": len(cache_binding),
            "recovery_binding_sha256": sha256_text(canonical_json(cache_binding)),
        },
        "tokenizer": {
            "path": str(inputs.tokenizer_path),
            "tokenizer_json_sha256": sha256_file(inputs.tokenizer_path / "tokenizer.json"),
        },
    }


def build_candidate_payload(
    inputs: RepairInputs,
    tokenizer: Any,
    *,
    expectations: RepairExpectations | None = None,
) -> CandidatePayload:
    """Construct the complete candidate in memory after all deterministic gates."""

    inputs = inputs.resolved()
    source_manifest = _read_json(inputs.source_release / "release_manifest.json")
    _require(source_manifest.get("status") == "passed", "source compressed release is not passed")
    _read_json(inputs.base_release / "base_release_manifest.json")

    source_index, source_files = _load_dataset_index(
        inputs.source_release / "analysis_sft", label="compressed source"
    )
    base_index, base_files = _load_dataset_index(
        inputs.base_release / "analysis_sft", label="base source"
    )
    generation_index, generation_sources = _load_generation_index(inputs.generation_dir)

    split_rows: dict[str, list[Mapping[str, str]]] = {split: [] for split in SPLITS}
    repair_records: list[Mapping[str, Any]] = []
    audit_rows: list[Mapping[str, Any]] = []
    answer_methods: Counter[str] = Counter()
    reasoning_repairs: Counter[str] = Counter()
    changed_sample_ids: list[str] = []
    recovery_cache_files: list[Mapping[str, str]] = []
    sample_ids: set[str] = set()
    cross_split_identities: dict[str, set[str]] = defaultdict(set)
    cross_split_responses: dict[str, set[str]] = defaultdict(set)

    for split in SPLITS:
        source_path = inputs.source_release / "analysis_sft" / f"{split}.jsonl"
        source_rows = _read_jsonl(source_path)
        for line_number, raw_row in enumerate(source_rows, 1):
            row = {key: str(raw_row[key]) for key in ("prompt", "response", "provided_data")}
            prompt_sha = sha256_text(row["prompt"])
            provided_sha = sha256_text(row["provided_data"])
            join_key = (split, prompt_sha, provided_sha)
            generation = generation_index.get(join_key)
            _require(generation is not None, f"no canonical sample_id hash join: {split}:{line_number}")
            _require(generation.sample_id not in sample_ids, f"duplicate admitted sample_id: {generation.sample_id}")
            sample_ids.add(generation.sample_id)

            base_entry = base_index.get(join_key)
            _require(base_entry is not None, f"base source hash join missing: {generation.sample_id}")
            base_row, base_line_number = base_entry
            base_reasoning, base_answer = _split_native_response(
                base_row["response"], label=f"base:{generation.sample_id}"
            )
            del base_reasoning
            source_reasoning, source_answer = _split_native_response(
                row["response"], label=f"compressed:{generation.sample_id}"
            )
            _require(source_answer == base_answer, f"compressed answer drift: {generation.sample_id}")
            _require(
                sha256_text(base_row["response"]) == generation.response_sha256,
                f"generation response hash mismatch: {generation.sample_id}",
            )
            _require(
                sha256_text(base_answer) == generation.final_analysis_sha256,
                f"generation answer hash mismatch: {generation.sample_id}",
            )

            json_like = is_json_like_answer(source_answer)
            has_evidence_id = _EVIDENCE_ID_RE.search(source_answer) is not None
            cache_path: Path | None = None
            cache_sha: str | None = None
            if json_like:
                cache, cache_path, cache_sha = _load_teacher_cache(inputs, generation)
                raw_provider = cache["provider_raw"]
                method, recovered = recover_json_like_answer(
                    str(raw_provider["content"]), str(raw_provider["reasoning_content"])
                )
                repaired_answer, removed_ids = strip_inline_evidence_ids(recovered)
                recovery_cache_files.append(
                    {
                        "sample_id": generation.sample_id,
                        "path": str(cache_path),
                        "sha256": cache_sha,
                    }
                )
            elif has_evidence_id:
                method = "strip_inline_evidence_ids"
                repaired_answer, removed_ids = strip_inline_evidence_ids(source_answer)
            else:
                method = "unchanged"
                repaired_answer = source_answer
                removed_ids = 0
            answer_methods[method] += 1

            repaired_reasoning, applied_reasoning_repairs = clean_reasoning(
                source_reasoning, source_answer
            )
            reasoning_repairs.update(applied_reasoning_repairs)
            answer_tokens = validate_answer(repaired_answer, tokenizer)
            reasoning_tokens = len(_token_ids(tokenizer, repaired_reasoning))
            _require(
                512 <= reasoning_tokens <= 2400,
                f"reasoning token count outside 512..2400 for {generation.sample_id}: {reasoning_tokens}",
            )
            _require(
                repaired_answer not in repaired_reasoning,
                f"answer remains copied in reasoning: {generation.sample_id}",
            )
            repaired_response = repaired_reasoning + BOUNDARY + repaired_answer
            _require(repaired_response.count(BOUNDARY) == 1, "repaired boundary count drift")
            _require("<think>" not in repaired_response.lower(), "raw target contains opening think tag")
            response_ids = _token_ids(tokenizer, repaired_response)
            _require(
                not _has_strict_periodic_tail(response_ids),
                f"strict periodic target tail: {generation.sample_id}",
            )

            output_row = {
                "prompt": row["prompt"],
                "response": repaired_response,
                "provided_data": row["provided_data"],
            }
            split_rows[split].append(output_row)
            changed = repaired_response != row["response"]
            if changed:
                changed_sample_ids.append(generation.sample_id)
                audit_rows.append(
                    {
                        "schema_version": SEMANTIC_AUDIT_INPUT_SCHEMA_VERSION,
                        "sample_id": generation.sample_id,
                        "split": split,
                        "prompt": row["prompt"],
                        "provided_data": row["provided_data"],
                        "candidate_response": repaired_response,
                        "prompt_sha256": prompt_sha,
                        "provided_data_sha256": provided_sha,
                        "candidate_response_sha256": sha256_text(repaired_response),
                    }
                )

            repair_records.append(
                {
                    "schema_version": REPAIR_MANIFEST_SCHEMA_VERSION,
                    "sample_id": generation.sample_id,
                    "split": split,
                    "source_line_number": line_number,
                    "base_line_number": base_line_number,
                    "generation_manifest_line_number": generation.manifest_line_number,
                    "prompt_sha256": prompt_sha,
                    "provided_data_sha256": provided_sha,
                    "source_response_sha256": sha256_text(row["response"]),
                    "candidate_response_sha256": sha256_text(repaired_response),
                    "old_response_sha256": sha256_text(row["response"]),
                    "new_response_sha256": sha256_text(repaired_response),
                    "source_reasoning_sha256": sha256_text(source_reasoning),
                    "candidate_reasoning_sha256": sha256_text(repaired_reasoning),
                    "source_answer_sha256": sha256_text(source_answer),
                    "candidate_answer_sha256": sha256_text(repaired_answer),
                    "answer_repair_method": method,
                    "inline_evidence_ids_removed": removed_ids,
                    "reasoning_repairs": list(applied_reasoning_repairs),
                    "reasoning_tokens": reasoning_tokens,
                    "answer_tokens": answer_tokens,
                    "changed": changed,
                    "teacher_cache": (
                        None
                        if cache_path is None
                        else {
                            "path": str(cache_path),
                            "sha256": cache_sha,
                            "cache_key": generation.cache_key,
                        }
                    ),
                }
            )
            identity = sha256_text(row["prompt"] + "\0" + row["provided_data"])
            cross_split_identities[identity].add(split)
            cross_split_responses[sha256_text(repaired_response)].add(split)

    split_counts = {split: len(split_rows[split]) for split in SPLITS}
    _require(
        not any(len(values) > 1 for values in cross_split_identities.values()),
        "cross-split prompt/provided_data duplicate",
    )
    _require(
        not any(len(values) > 1 for values in cross_split_responses.values()),
        "cross-split response duplicate",
    )
    _require(len(repair_records) == len(sample_ids), "repair manifest/sample identity drift")
    _require(len(audit_rows) == len(changed_sample_ids), "semantic audit population drift")

    if expectations is not None:
        _assert_expected_counter(split_counts, expectations.split_counts, label="split counts")
        _assert_expected_counter(
            answer_methods, expectations.answer_method_counts, label="answer repair methods"
        )
        _assert_expected_counter(
            reasoning_repairs,
            expectations.reasoning_repair_counts,
            label="reasoning repair methods",
        )
        _require(
            len(changed_sample_ids) == expectations.changed_rows,
            f"changed row count drift: expected {expectations.changed_rows}, got {len(changed_sample_ids)}",
        )

    sources = _candidate_sources(
        inputs,
        source_files=source_files,
        base_files=base_files,
        generation_sources=generation_sources,
        recovery_cache_files=recovery_cache_files,
    )
    changed_ids_sha = sha256_text(canonical_json(sorted(changed_sample_ids)))
    summary = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "quality_status": "pending_semantic_audit",
        "immutable_candidate": True,
        "sources": sources,
        "split_counts": split_counts,
        "answer_repair_method_counts": dict(sorted(answer_methods.items())),
        "reasoning_repair_counts": dict(sorted(reasoning_repairs.items())),
        "repair_manifest_rows": len(repair_records),
        "changed_rows": len(changed_sample_ids),
        "changed_sample_ids_sha256": changed_ids_sha,
        "semantic_audit_contract": {
            "schema_version": SEMANTIC_AUDIT_SCHEMA_VERSION,
            "required_rows": len(changed_sample_ids),
            "required_validated_violation_counts": {
                "factual": 0,
                "numerical": 0,
                "causal": 0,
                "target_leakage": 0,
            },
            "required_judge_errors": 0,
        },
        "quality_gates": {
            "plain_text_answers": "passed",
            "reasoning_contract": "passed",
            "stable_sample_id_hash_join": "passed",
            "source_prompt_and_data_hashes": "passed",
            "cross_split_duplicates": 0,
            "strict_periodic_tails": 0,
        },
    }
    return CandidatePayload(
        split_rows={split: tuple(split_rows[split]) for split in SPLITS},
        repair_records=tuple(repair_records),
        semantic_audit_rows=tuple(audit_rows),
        summary=summary,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    directories = sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for path in directories:
        _fsync_directory(path)
    _fsync_directory(root)


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        _require(not path.is_symlink(), f"refusing to seal symlink: {path}")
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)


def _remove_tree(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts)):
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o755)
    root.chmod(0o755)
    shutil.rmtree(root)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Use Linux renameat2(RENAME_NOREPLACE), matching the release builders."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(renameat2 is not None, "atomic renameat2(RENAME_NOREPLACE) is unavailable")
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
        raise RepairError(f"immutable destination already exists: {destination}")
    raise RepairError(
        f"atomic no-overwrite publication failed for {destination}: {os.strerror(error)}"
    )


def materialize_candidate(
    *,
    inputs: RepairInputs,
    work_dir: Path,
    tokenizer: Any,
    expectations: RepairExpectations | None = None,
) -> Mapping[str, Any]:
    """Build one immutable candidate directory, never a passed release."""

    work_dir = work_dir.resolve()
    _require(not work_dir.exists() and not work_dir.is_symlink(), f"candidate exists: {work_dir}")
    work_dir.parent.mkdir(parents=True, exist_ok=True)
    payload = build_candidate_payload(inputs, tokenizer, expectations=expectations)
    staging = Path(tempfile.mkdtemp(prefix=f".{work_dir.name}.", dir=work_dir.parent))
    try:
        dataset_dir = staging / "analysis_sft"
        audit_dir = staging / "audits"
        dataset_dir.mkdir()
        audit_dir.mkdir()
        split_files: dict[str, Mapping[str, Any]] = {}
        for split in SPLITS:
            path = dataset_dir / f"{split}.jsonl"
            _write_jsonl(path, payload.split_rows[split])
            split_files[split] = {
                "path": f"analysis_sft/{split}.jsonl",
                "rows": len(payload.split_rows[split]),
                "sha256": sha256_file(path),
            }
        repair_path = audit_dir / "repair_manifest.jsonl"
        _write_jsonl(repair_path, payload.repair_records)
        audit_input_path = audit_dir / "semantic_audit_input.jsonl"
        _write_jsonl(audit_input_path, payload.semantic_audit_rows)
        manifest = {
            **payload.summary,
            "candidate_id": work_dir.name,
            "created_at_utc": _utc_now(),
            "split_files": split_files,
            "repair_manifest": {
                "path": "audits/repair_manifest.jsonl",
                "rows": len(payload.repair_records),
                "sha256": sha256_file(repair_path),
            },
            "semantic_audit_input": {
                "path": "audits/semantic_audit_input.jsonl",
                "rows": len(payload.semantic_audit_rows),
                "sha256": sha256_file(audit_input_path),
            },
        }
        _write_json(staging / "candidate_manifest.json", manifest)
        _seal_tree(staging)
        _fsync_tree(staging)
        _seal_tree(staging)
        _fsync_directory(staging.parent)
        _rename_noreplace(staging, work_dir)
        _fsync_directory(work_dir.parent)
        return manifest
    except BaseException:
        _remove_tree(staging)
        raise


def _inputs_from_candidate_manifest(manifest: Mapping[str, Any]) -> RepairInputs:
    sources = manifest.get("sources")
    _require(isinstance(sources, dict), "candidate sources are missing")

    def source_path(key: str) -> Path:
        value = sources.get(key)
        _require(isinstance(value, dict) and isinstance(value.get("path"), str), f"missing source {key}")
        return Path(str(value["path"]))

    teacher = sources.get("teacher_cache")
    tokenizer = sources.get("tokenizer")
    _require(isinstance(teacher, dict) and isinstance(teacher.get("root"), str), "teacher cache source missing")
    _require(isinstance(tokenizer, dict) and isinstance(tokenizer.get("path"), str), "tokenizer source missing")
    return RepairInputs(
        source_release=source_path("source_release_manifest").parent,
        base_release=source_path("base_release_manifest").parent,
        generation_dir=source_path("generation_handoff").parent,
        teacher_cache=Path(str(teacher["root"])),
        tokenizer_path=Path(str(tokenizer["path"])),
    ).resolved()


def verify_candidate(
    work_dir: Path,
    *,
    tokenizer: Any | None = None,
    expectations: RepairExpectations | None = None,
) -> Mapping[str, Any]:
    """Replay repair from bound sources and byte-compare every candidate artifact."""

    work_dir = work_dir.resolve()
    manifest_path = work_dir / "candidate_manifest.json"
    manifest = _read_json(manifest_path)
    _require(manifest.get("schema_version") == CANDIDATE_SCHEMA_VERSION, "candidate schema mismatch")
    _require(manifest.get("quality_status") == "pending_semantic_audit", "candidate status mismatch")
    _require(manifest.get("immutable_candidate") is True, "candidate immutability flag missing")
    inputs = _inputs_from_candidate_manifest(manifest)
    active_tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(inputs.tokenizer_path)
    payload = build_candidate_payload(inputs, active_tokenizer, expectations=expectations)

    for split in SPLITS:
        path = work_dir / "analysis_sft" / f"{split}.jsonl"
        expected_text = "".join(canonical_json(row) + "\n" for row in payload.split_rows[split])
        _require(path.read_text(encoding="utf-8") == expected_text, f"candidate split drift: {split}")
        descriptor = manifest.get("split_files", {}).get(split, {})
        _require(descriptor.get("path") == f"analysis_sft/{split}.jsonl", f"split path drift: {split}")
        _require(descriptor.get("rows") == len(payload.split_rows[split]), f"split rows drift: {split}")
        _require(descriptor.get("sha256") == sha256_file(path), f"split hash drift: {split}")

    comparisons = (
        (
            work_dir / "audits/repair_manifest.jsonl",
            payload.repair_records,
            manifest.get("repair_manifest"),
            "repair manifest",
        ),
        (
            work_dir / "audits/semantic_audit_input.jsonl",
            payload.semantic_audit_rows,
            manifest.get("semantic_audit_input"),
            "semantic audit input",
        ),
    )
    for path, rows, descriptor, label in comparisons:
        _require(isinstance(descriptor, dict), f"{label} descriptor missing")
        expected_text = "".join(canonical_json(row) + "\n" for row in rows)
        _require(path.read_text(encoding="utf-8") == expected_text, f"{label} content drift")
        _require(descriptor.get("rows") == len(rows), f"{label} row drift")
        _require(descriptor.get("sha256") == sha256_file(path), f"{label} hash drift")

    for key in (
        "split_counts",
        "answer_repair_method_counts",
        "reasoning_repair_counts",
        "repair_manifest_rows",
        "changed_rows",
        "changed_sample_ids_sha256",
        "sources",
        "quality_gates",
    ):
        _require(manifest.get(key) == payload.summary.get(key), f"candidate summary drift: {key}")
    return manifest


def _validate_semantic_audit(
    *,
    summary_path: Path,
    candidate_dir: Path,
    candidate_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    summary = _read_json(summary_path)
    _require(summary.get("schema_version") == SEMANTIC_AUDIT_SCHEMA_VERSION, "semantic audit schema mismatch")
    _require(summary.get("status") == "passed", "semantic audit did not pass")
    repair_path = candidate_dir / str(candidate_manifest["repair_manifest"]["path"])
    _require(
        summary.get("repair_manifest_sha256") == sha256_file(repair_path),
        "semantic audit repair-manifest binding mismatch",
    )
    counts = summary.get("counts")
    _require(isinstance(counts, dict), "semantic audit counts missing")
    required = int(candidate_manifest["changed_rows"])
    _require(counts.get("expected") == required, "semantic audit expected count mismatch")
    _require(counts.get("completed") == required, "semantic audit is incomplete")
    _require(counts.get("passed") == required, "semantic audit passed count mismatch")
    for key in ("failed", "judge_errors", "blocking_violations"):
        _require(counts.get(key) == 0, f"semantic audit {key} is nonzero")
    _require(summary.get("errors") == [], "semantic audit contains judge errors")
    judge = summary.get("judge")
    health = judge.get("health") if isinstance(judge, dict) else None
    _require(
        isinstance(judge, dict)
        and judge.get("model") == SEMANTIC_JUDGE_MODEL
        and isinstance(health, dict)
        and health.get("model") == SEMANTIC_JUDGE_MODEL
        and Path(str(health.get("loaded_model_root") or "")).name
        == SEMANTIC_JUDGE_MODEL
        and health.get("status") == "ready"
        and health.get("tokenizer_parity") is True
        and health.get("weight_attested") is True,
        "semantic audit judge health is not attested",
    )

    row_descriptor = summary.get("row_audit")
    _require(isinstance(row_descriptor, dict), "semantic row-audit descriptor missing")
    row_path = summary_path.parent / str(row_descriptor.get("path") or "")
    _require(row_descriptor.get("rows") == required, "semantic row-audit row count mismatch")
    _require(row_descriptor.get("sha256") == sha256_file(row_path), "semantic row-audit hash mismatch")
    row_audits = _read_jsonl(row_path)
    repair_records = _read_jsonl(repair_path)
    expected = {
        str(row["sample_id"]): (
            str(row["split"]),
            str(row["new_response_sha256"]),
            str(row["provided_data_sha256"]),
        )
        for row in repair_records
        if row.get("old_response_sha256") != row.get("new_response_sha256")
    }
    observed: dict[str, tuple[str, str, str]] = {}
    for number, row in enumerate(row_audits, 1):
        _require(
            row.get("schema_version") == SEMANTIC_AUDIT_SCHEMA_VERSION,
            f"semantic row schema mismatch: {number}",
        )
        _require(row.get("status") == "passed", f"semantic row failed: {number}")
        _require(
            row.get("blocking_violations") == [],
            f"semantic row has blocking violations: {number}",
        )
        validated = row.get("validated_violations")
        _require(isinstance(validated, list), f"semantic row violations missing: {number}")
        _require(
            not any(
                isinstance(violation, dict)
                and violation.get("kind") in SEMANTIC_BLOCKING_KINDS
                for violation in validated
            ),
            f"semantic row contains a validated blocking violation: {number}",
        )
        rubric = row.get("rubric")
        _require(
            isinstance(rubric, dict)
            and set(rubric) == SEMANTIC_RUBRIC_KEYS
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value <= 4
                for value in rubric.values()
            ),
            f"semantic row rubric is invalid: {number}",
        )
        sample_id = str(row.get("sample_id") or "")
        _require(sample_id and sample_id not in observed, f"duplicate semantic sample: {number}")
        split = str(row.get("split") or "")
        _require(split in SPLITS, f"semantic row split mismatch: {number}")
        candidate_sha = _require_sha256(row.get("candidate_sha256"), f"semantic row {number}")
        evidence_sha = _require_sha256(
            row.get("evidence_sha256"), f"semantic evidence row {number}"
        )
        attempts = row.get("judge_attempts")
        _require(
            isinstance(attempts, int) and not isinstance(attempts, bool) and attempts >= 1,
            f"semantic row judge attempts invalid: {number}",
        )
        _require_sha256(row.get("judge_raw_sha256"), f"semantic judge raw row {number}")
        observed[sample_id] = (split, candidate_sha, evidence_sha)
    _require(observed == expected, "semantic row-audit candidate binding mismatch")
    return summary


def _validate_deterministic_receipt(
    *,
    report_path: Path,
    candidate_dir: Path,
    candidate_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind publication to the completed full-data and real-tokenizer audit."""

    report = _read_json(report_path)
    _require(
        report.get("schema_version") == DETERMINISTIC_VALIDATION_SCHEMA_VERSION,
        "deterministic validation schema mismatch",
    )
    _require(report.get("status") == "passed", "deterministic validation did not pass")
    claimed_validation_sha = _require_sha256(
        report.get("validation_sha256"), "deterministic validation digest"
    )
    digest_payload = dict(report)
    digest_payload.pop("validation_sha256", None)
    _require(
        claimed_validation_sha == sha256_text(canonical_json(digest_payload)),
        "deterministic validation self-digest mismatch",
    )

    contracts = report.get("contracts")
    _require(isinstance(contracts, dict), "deterministic validation contracts missing")
    _require(
        contracts.get("data") == DETERMINISTIC_DATA_CONTRACT_VERSION
        and contracts.get("tokenization") == DETERMINISTIC_TOKEN_CONTRACT_VERSION,
        "deterministic validation contract mismatch",
    )
    validator = report.get("validator")
    validator_source = Path(__file__).with_name("validate_chk1_clean_sft.py").resolve()
    _require(isinstance(validator, dict), "deterministic validator provenance missing")
    _require(
        validator.get("version") == DETERMINISTIC_VALIDATION_SCHEMA_VERSION
        and validator.get("source_sha256") == sha256_file(validator_source),
        "deterministic validator provenance mismatch",
    )

    configuration = report.get("configuration")
    _require(isinstance(configuration, dict), "deterministic validation configuration missing")
    _require(
        configuration.get("expected_split_counts")
        == dict(candidate_manifest["split_counts"]),
        "deterministic validation split expectation mismatch",
    )
    _require(
        configuration.get("max_length") == 4608
        and configuration.get("reasoning_token_range") == [512, 2400]
        and configuration.get("answer_token_range") == [16, 512],
        "deterministic validation token contract mismatch",
    )

    inputs = report.get("inputs")
    _require(isinstance(inputs, dict), "deterministic validation inputs missing")
    clean_input = inputs.get("clean_release")
    _require(isinstance(clean_input, dict), "deterministic clean input missing")
    _require(
        clean_input.get("release_id") == candidate_dir.name,
        "deterministic validation candidate identity mismatch",
    )
    clean_splits = clean_input.get("splits")
    _require(isinstance(clean_splits, dict), "deterministic clean split binding missing")
    for split in SPLITS:
        split_input = clean_splits.get(split)
        split_descriptor = candidate_manifest["split_files"][split]
        _require(
            isinstance(split_input, dict)
            and split_input.get("sha256") == split_descriptor["sha256"],
            f"deterministic validation clean split hash mismatch: {split}",
        )
    manifest_inputs = clean_input.get("manifests")
    _require(isinstance(manifest_inputs, list), "deterministic candidate manifest binding missing")
    candidate_manifest_sha = sha256_file(candidate_dir / "candidate_manifest.json")
    _require(
        sum(
            isinstance(item, dict)
            and item.get("name") == "candidate_manifest.json"
            and item.get("sha256") == candidate_manifest_sha
            for item in manifest_inputs
        )
        == 1,
        "deterministic candidate manifest hash mismatch",
    )

    source_input = inputs.get("source_release")
    source_files = candidate_manifest["sources"]["source_split_files"]
    _require(isinstance(source_input, dict), "deterministic source input missing")
    source_splits = source_input.get("splits")
    _require(isinstance(source_splits, dict), "deterministic source split binding missing")
    for split in SPLITS:
        _require(
            isinstance(source_splits.get(split), dict)
            and source_splits[split].get("sha256") == source_files[split]["sha256"],
            f"deterministic validation source split hash mismatch: {split}",
        )

    repair_input = inputs.get("repair_manifest")
    _require(isinstance(repair_input, dict), "deterministic repair binding missing")
    _require(
        repair_input.get("sha256") == candidate_manifest["repair_manifest"]["sha256"]
        and repair_input.get("rows") == candidate_manifest["repair_manifest"]["rows"],
        "deterministic repair-manifest binding mismatch",
    )
    tokenizer_input = inputs.get("tokenizer")
    _require(isinstance(tokenizer_input, dict), "deterministic tokenizer binding missing")
    tokenizer_files = tokenizer_input.get("files")
    _require(isinstance(tokenizer_files, list), "deterministic tokenizer files missing")
    expected_tokenizer_json = candidate_manifest["sources"]["tokenizer"][
        "tokenizer_json_sha256"
    ]
    _require(
        sum(
            isinstance(item, dict)
            and item.get("name") == "tokenizer.json"
            and item.get("sha256") == expected_tokenizer_json
            for item in tokenizer_files
        )
        == 1,
        "deterministic tokenizer.json binding mismatch",
    )
    tokenizer_bundle_sha = _require_sha256(
        tokenizer_input.get("tokenizer_bundle_sha256"),
        "deterministic tokenizer bundle",
    )
    _require(
        tokenizer_bundle_sha == sha256_text(canonical_json(tokenizer_files)),
        "deterministic tokenizer bundle digest mismatch",
    )

    total_rows = sum(int(candidate_manifest["split_counts"][split]) for split in SPLITS)
    counts = report.get("counts")
    _require(isinstance(counts, dict), "deterministic validation counts missing")
    for key, expected in (
        ("expected_rows", total_rows),
        ("observed_rows", total_rows),
        ("valid_rows", total_rows),
        ("prompt_hashes_unchanged", total_rows),
        ("provided_data_hashes_unchanged", total_rows),
        ("order_unchanged", total_rows),
        ("truncated_rows", 0),
        ("issue_rows_or_groups", 0),
    ):
        _require(counts.get(key) == expected, f"deterministic validation count mismatch: {key}")
    _require(
        counts.get("split_counts") == dict(candidate_manifest["split_counts"]),
        "deterministic validation observed split mismatch",
    )
    gates = report.get("quality_gates")
    _require(
        isinstance(gates, dict)
        and bool(gates)
        and all(value is True for value in gates.values()),
        "deterministic validation quality gate failed",
    )
    _require(report.get("issues") == [], "deterministic validation contains issues")

    repair_records = _read_jsonl(
        candidate_dir / str(candidate_manifest["repair_manifest"]["path"])
    )
    expected_binding = [
        {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "line_number": row["source_line_number"],
            "prompt_sha256": row["prompt_sha256"],
            "provided_data_sha256": row["provided_data_sha256"],
            "response_sha256": row["new_response_sha256"],
        }
        for row in repair_records
    ]
    _require(
        report.get("content_binding_sha256")
        == sha256_text(canonical_json(expected_binding)),
        "deterministic validation row-content binding mismatch",
    )
    statistics = report.get("token_statistics")
    _require(isinstance(statistics, dict), "deterministic token statistics missing")
    max_total_tokens = statistics.get("max_total_tokens")
    _require(
        isinstance(max_total_tokens, int) and 0 < max_total_tokens <= 4608,
        "deterministic validation max token count is invalid",
    )
    return report


def publish_release(
    *,
    work_dir: Path,
    semantic_audit_summary: Path,
    deterministic_validation_report: Path,
    release_dir: Path,
    tokenizer: Any | None = None,
    expectations: RepairExpectations | None = None,
) -> Mapping[str, Any]:
    """Seal a verified candidate only after a hash-bound passed Qwen audit."""

    work_dir = work_dir.resolve()
    release_dir = release_dir.resolve()
    semantic_audit_summary = semantic_audit_summary.resolve()
    deterministic_validation_report = deterministic_validation_report.resolve()
    _require(not release_dir.exists() and not release_dir.is_symlink(), f"release exists: {release_dir}")
    candidate_manifest = verify_candidate(
        work_dir, tokenizer=tokenizer, expectations=expectations
    )
    audit = _validate_semantic_audit(
        summary_path=semantic_audit_summary,
        candidate_dir=work_dir,
        candidate_manifest=candidate_manifest,
    )
    deterministic_validation = _validate_deterministic_receipt(
        report_path=deterministic_validation_report,
        candidate_dir=work_dir,
        candidate_manifest=candidate_manifest,
    )
    release_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{release_dir.name}.", dir=release_dir.parent))
    try:
        shutil.copytree(work_dir / "analysis_sft", staging / "analysis_sft")
        (staging / "audits").mkdir()
        repair_source = work_dir / str(candidate_manifest["repair_manifest"]["path"])
        shutil.copy2(repair_source, staging / "audits/repair_manifest.jsonl")
        shutil.copy2(semantic_audit_summary, staging / "audits/semantic_audit_summary.json")
        shutil.copy2(
            deterministic_validation_report,
            staging / "audits/deterministic_validation.json",
        )
        row_descriptor = audit["row_audit"]
        row_source = semantic_audit_summary.parent / str(row_descriptor["path"])
        shutil.copy2(row_source, staging / "audits/semantic_row_audits.jsonl")

        split_files: dict[str, Mapping[str, Any]] = {}
        for split in SPLITS:
            path = staging / "analysis_sft" / f"{split}.jsonl"
            split_files[split] = {
                "path": f"analysis_sft/{split}.jsonl",
                "rows": int(candidate_manifest["split_counts"][split]),
                "sha256": sha256_file(path),
            }
        repair_path = staging / "audits/repair_manifest.jsonl"
        audit_summary_path = staging / "audits/semantic_audit_summary.json"
        audit_rows_path = staging / "audits/semantic_row_audits.jsonl"
        validation_path = staging / "audits/deterministic_validation.json"
        manifest = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": release_dir.name,
            "created_at_utc": _utc_now(),
            "quality_status": "passed",
            "immutable": True,
            "split_counts": dict(candidate_manifest["split_counts"]),
            "split_files": split_files,
            "repair_manifest": {
                "path": "audits/repair_manifest.jsonl",
                "rows": int(candidate_manifest["repair_manifest"]["rows"]),
                "sha256": sha256_file(repair_path),
            },
            "semantic_audit": {
                "schema_version": SEMANTIC_AUDIT_SCHEMA_VERSION,
                "summary": {
                    "path": "audits/semantic_audit_summary.json",
                    "sha256": sha256_file(audit_summary_path),
                },
                "rows": {
                    "path": "audits/semantic_row_audits.jsonl",
                    "rows": int(row_descriptor["rows"]),
                    "sha256": sha256_file(audit_rows_path),
                },
            },
            "deterministic_validation": {
                "schema_version": DETERMINISTIC_VALIDATION_SCHEMA_VERSION,
                "report": {
                    "path": "audits/deterministic_validation.json",
                    "sha256": sha256_file(validation_path),
                },
                "validation_sha256": deterministic_validation["validation_sha256"],
                "max_length": deterministic_validation["configuration"]["max_length"],
                "tokenizer_bundle_sha256": deterministic_validation["inputs"]["tokenizer"][
                    "tokenizer_bundle_sha256"
                ],
            },
            "candidate_manifest": {
                "path": str(work_dir / "candidate_manifest.json"),
                "sha256": sha256_file(work_dir / "candidate_manifest.json"),
            },
            "source_hashes": candidate_manifest["sources"],
            "answer_repair_method_counts": candidate_manifest[
                "answer_repair_method_counts"
            ],
            "reasoning_repair_counts": candidate_manifest["reasoning_repair_counts"],
            "changed_rows": candidate_manifest["changed_rows"],
            "changed_sample_ids_sha256": candidate_manifest[
                "changed_sample_ids_sha256"
            ],
        }
        _write_json(staging / "release_manifest.json", manifest)
        _fsync_tree(staging)
        _seal_tree(staging)
        _fsync_directory(staging.parent)
        _rename_noreplace(staging, release_dir)
        _fsync_directory(release_dir.parent)
        return manifest
    except BaseException:
        _remove_tree(staging)
        raise


def _default_inputs(args: argparse.Namespace) -> RepairInputs:
    return RepairInputs(
        source_release=args.source_release,
        base_release=args.base_release,
        generation_dir=args.generation_dir,
        teacher_cache=args.teacher_cache,
        tokenizer_path=args.tokenizer,
    )


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-release", type=Path, default=DEFAULT_SOURCE_RELEASE)
    parser.add_argument("--base-release", type=Path, default=DEFAULT_BASE_RELEASE)
    parser.add_argument("--generation-dir", type=Path, default=DEFAULT_GENERATION_DIR)
    parser.add_argument("--teacher-cache", type=Path, default=DEFAULT_TEACHER_CACHE)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="read-only deterministic source audit")
    _add_source_arguments(audit_parser)

    build_parser = subparsers.add_parser("build", help="materialize an immutable candidate")
    _add_source_arguments(build_parser)
    build_parser.add_argument("--work-dir", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify", help="replay and verify a candidate")
    verify_parser.add_argument("--work-dir", type=Path, required=True)

    publish_parser = subparsers.add_parser("publish", help="publish after the Qwen audit")
    publish_parser.add_argument("--work-dir", type=Path, required=True)
    publish_parser.add_argument("--semantic-audit-summary", type=Path, required=True)
    publish_parser.add_argument(
        "--deterministic-validation-report", type=Path, required=True
    )
    publish_parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)

    args = parser.parse_args(argv)
    expectations = RepairExpectations.production()
    if args.command == "audit":
        inputs = _default_inputs(args).resolved()
        payload = build_candidate_payload(
            inputs, _load_tokenizer(inputs.tokenizer_path), expectations=expectations
        )
        result = payload.summary
    elif args.command == "build":
        inputs = _default_inputs(args).resolved()
        result = materialize_candidate(
            inputs=inputs,
            work_dir=args.work_dir,
            tokenizer=_load_tokenizer(inputs.tokenizer_path),
            expectations=expectations,
        )
    elif args.command == "verify":
        result = verify_candidate(args.work_dir, expectations=expectations)
    else:
        result = publish_release(
            work_dir=args.work_dir,
            semantic_audit_summary=args.semantic_audit_summary,
            deterministic_validation_report=args.deterministic_validation_report,
            release_dir=args.release_dir,
            expectations=expectations,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
