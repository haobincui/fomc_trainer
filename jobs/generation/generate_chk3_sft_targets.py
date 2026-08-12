"""Build chk3 Minutes-rewrite SFT targets with DeepSeek V4 Pro.

The only model-facing source text is the final answer from the immutable chk1
canonical release.  This module deliberately has no chk2 dependency: chk1
sample IDs and splits define the chk3 population.

The DeepSeek chat template opens ``<think>`` before the completion.  Therefore
the stored SFT completion is exactly::

    reasoning_content\n</think>\nMinutes paragraph

The provider's ``reasoning_content`` supplies the first part and the JSON
``content.answer`` supplies the second part.  No ``<think>`` opening tag or
``<answer>`` wrapper is serialized in the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from jobs.retrain_v2.chk1.deepseek_teacher import (
    DEEPSEEK_BASE_URL,
    DeepSeekTeacherConfig,
    DeepSeekTeacherResponse,
    OpenAIDeepSeekBackend,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHK1_HANDOFF = REPO_ROOT / (
    "output/data/retrain_v2/chk1/canonical_releases/"
    "chk1_full_v7_automated_v2_20260804/handoff.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output/data/retrain_v2/chk3/deepseek_v4_pro_v2"
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "models/DeepSeek-R1-Distill-Llama-8B"

SPLITS = ("train", "eval", "test")
EXPECTED_SPLIT_COUNTS = {"train": 1683, "eval": 199, "test": 190}
EXPECTED_TOTAL = sum(EXPECTED_SPLIT_COUNTS.values())
TEACHER_MODEL = "deepseek-v4-pro"
MAX_TOKENS = 4096
MIN_REASONING_TOKENS = 64
MAX_REASONING_TOKENS = 2400
DEFAULT_CONCURRENCY = 8
API_KEY_ENV = "DEEPSEEK_API_KEY"

PREPARED_SCHEMA_VERSION = "chk3-v4-pro-prepared-v2"
TEACHER_CACHE_SCHEMA_VERSION = "chk3-v4-pro-teacher-cache-v2"
MANIFEST_SCHEMA_VERSION = "chk3-v4-pro-sft-manifest-v2"
PROMPT_CONTRACT_SCHEMA_VERSION = "chk3-v4-pro-prompt-contract-v2"

STUDENT_SYSTEM_PROMPT = """\
You are a Federal Reserve Minutes editor. The user supplies an analysis to be
rewritten. In the native reasoning section, plan a faithful rewrite by
identifying the claims, quantities, dates, directions, comparisons, and
expressions of uncertainty that must be preserved, and choose formal FOMC
Minutes wording. Preserve the numeric-quantity multiset: every source quantity
must appear the same number of times, although an exactly equivalent unit or
number form may be used.

Provide a substantive fidelity plan even for a short input; do not emit only
a transition or closing sentence in the reasoning section.

Reason only about the supplied analysis and the rewrite. Do not discuss
instructions, prompts, JSON, APIs, schemas, keys, fields, response formats,
output contracts, validation errors, tools, teacher or student roles, or the
act of answering. Do not introduce facts, causes, people, attributions,
policy actions, decisions, or votes that are not in the analysis. Treat
tokens beginning with ev- as internal evidence citations and do not reproduce
them in the final paragraph.

After closing the reasoning section, output exactly one formal FOMC
Minutes-style paragraph. Do not output headings, lists, JSON, commentary,
citations, or answer tags.
"""

TEACHER_SYSTEM_PROMPT = """\
You are the DeepSeek V4 Pro reasoning teacher for a Minutes-rewrite SFT
dataset. In reasoning_content, plan a faithful rewrite by identifying the
claims, quantities, dates, directions, comparisons, and expressions of
uncertainty that must be preserved, and choose formal FOMC Minutes wording.
Preserve the numeric-quantity multiset: every source quantity must appear the
same number of times. Exact unit and number-form conversions are allowed, but
rounding, approximation, and newly derived quantities are not. Preserve every
source month, but do not treat the lowercase modal verb may as a month.

Reason only about the supplied analysis and the rewrite. Do not mention or
discuss instructions, prompts, JSON, APIs, schemas, keys, fields, response
formats, output contracts, validation feedback, tools, teacher or student
roles, or the act of answering. Keep the reasoning concise: do not reproduce
the complete analysis or draft the final paragraph more than once. Even for a
short input, provide a substantive fidelity plan that covers its claims,
quantities or dates, direction, comparisons, and uncertainty as applicable.

Do not introduce facts, causes, people, attributions, policy actions,
decisions, or votes that are not in the analysis. Tokens beginning with ev-
are internal evidence citations and must not appear in the final paragraph.
Return exactly one JSON object in content with the single key answer. The
answer must be exactly one formal FOMC Minutes-style paragraph. Do not put
reasoning, headings, lists, Markdown, citations, commentary, or model-control
tags in the answer. Never discuss this transport requirement in
reasoning_content.
"""

TEACHER_REPAIR_SYSTEM_PROMPT = """\
Regenerate the Minutes-rewrite target using only the supplied analysis. Apply
the supplied validation feedback silently: do not quote, name, summarize, or
discuss the feedback in reasoning_content. In reasoning_content, concisely
identify the source claims, quantities, dates, directions, comparisons, and
uncertainty that the rewrite must preserve. Do not reproduce the complete
analysis or draft the final paragraph more than once. Preserve the
numeric-quantity multiset: every source quantity must appear the same number
of times, using exact conversions if desired. Preserve every source month, but
do not treat the lowercase modal verb may as a month. Even for a short input,
provide a substantive fidelity plan rather than a transition or closing
sentence.

Do not mention or discuss instructions, prompts, JSON, APIs, schemas, keys,
fields, response formats, output contracts, validation errors, tools, teacher
or student roles, or the act of answering. Exact unit and number-form
conversions are allowed, but rounding, approximation, newly derived
quantities, unsupported attributions, and omitted source quantities or dates
are not.

Return exactly one JSON object in content with the single key answer. The
answer must be exactly one formal FOMC Minutes-style paragraph and must not
add any fact, cause, person, attribution, policy action, decision, or vote.
Remove internal ev- evidence citations from the final paragraph. Never
discuss this transport requirement in reasoning_content. This is the only
repair attempt.
"""

USER_PROMPT_PREFIX = "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"

CONTROL_MARKERS = (
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
    flags=re.IGNORECASE,
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
_EVIDENCE_ID_RE = re.compile(r"\bev-[0-9a-f]{6,}\b", flags=re.IGNORECASE)

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

_REASONING_META_PATTERNS = {
    "json": re.compile(r"\bJSON\b", re.IGNORECASE),
    "api": re.compile(r"\bAPIs?\b|\bAPI\s+contracts?\b", re.IGNORECASE),
    "schema": re.compile(r"\bschemas?\b", re.IGNORECASE),
    "answer_key": re.compile(
        r"\banswer\s+(?:keys?|fields?)\b|\b(?:keys?|fields?)\s+[\"']?answer\b",
        re.IGNORECASE,
    ),
    "response_contract": re.compile(
        r"\b(?:response\s+formats?|output\s+contracts?|transport\s+requirements?)\b",
        re.IGNORECASE,
    ),
    "validation": re.compile(r"\bvalidation\s+(?:errors?|feedback)\b", re.IGNORECASE),
    "prompt_instruction": re.compile(
        r"\b(?:system|user)\s+prompts?\b|\bprompts?\s+(?:asks?|requires?|says?)\b|"
        r"\b(?:the|this|that)\s+prompts?\b|"
        r"\binstructions?\s+(?:asks?|requires?|says?)\b|"
        r"\b(?:the|these|this|that|strict|supplied)\s+instructions?\b|"
        r"\b(?:system|user|assistant)\s+(?:messages?|roles?)\b|"
        r"\b(?:the|this)\s+user\s+(?:asks?|says?|requests?|requires?)\b",
        re.IGNORECASE,
    ),
    "roles": re.compile(
        r"\bteacher\b|\bstudent\s+(?:models?|roles?)\b|"
        r"\bacting\s+as\s+(?:an?\s+)?assistant\b",
        re.IGNORECASE,
    ),
    "tools": re.compile(
        r"\btool\s+(?:calls?|usage)\b|\busing\s+(?:a\s+)?tool\b", re.IGNORECASE
    ),
    "answering": re.compile(
        r"\bwe\s+(?:are|were)\s+(?:asked|given)\b|"
        r"\bwe\b|"
        r"\b(?:need|must|should|will)\s+to\s+"
        r"(?:rewrite|produce|output|return|answer)\b|"
        r"\blet['’]?s\s+(?:parse|extract|check|craft|write|produce|output)\b|"
        r"\b(?:the\s+)?(?:final\s+)?answer\b|"
        r"\b(?:the\s+)?final\s+paragraph\b|"
        r"\b(?:one\s+)?possible\s+rewrite\b|"
        r"\b(?:one|single|exactly\s+one)\s+paragraph\b|"
        r"\bnow\s+(?:output|return)\b",
        re.IGNORECASE,
    ),
    "reasoning_content": re.compile(r"\breasoning_content\b", re.IGNORECASE),
    "inventory_contract": re.compile(
        r"\brequired_(?:dates|quantity_occurrences)\b|"
        r"\b(?:targeted_attempt|prior_diagnostics|execution_rules)\b|"
        r"\boccurrence-level\s+(?:quantity\s+)?inventory\b|"
        r"\b(?:quantity|date|fidelity)\s+inventory\b",
        re.IGNORECASE,
    ),
    "length_contract": re.compile(
        r"\bword\s+counts?\b|\bword\s+limits?\b|\btoken\s+limits?\b|"
        r"\b(?:under|within|below|fewer\s+than|no\s+more\s+than)\s+"
        r"\d[\d,]*\s+words?\b|\b\d[\d,]*\s*-\s*\d[\d,]*\s+words?\b",
        re.IGNORECASE,
    ),
    "answer_output_meta": re.compile(
        r"\b(?:the\s+)?final\s+output\b|\boutput\s+(?:only|exactly|must|should|will)\b|"
        r"\b(?:return|produce)\s+(?:only|exactly)\b|"
        r"\b(?:the\s+)?answer\s+(?:must|should|will|field|key)\b|"
        r"\b(?:inside|within)\s+(?:the\s+)?answer\b|"
        r"\bcontent\s+(?:field|key)\b|\bthe\s+only\s+key\b",
        re.IGNORECASE,
    ),
    "reasoning_protocol": re.compile(
        r"\b(?:the\s+)?reasoning\s+(?:should|must|will|is\s+(?:internal|separate|done))\b|"
        r"\b(?:reasoning|answer|content)\s+(?:section|field)s?\b|"
        r"\b(?:this|that|it)\s+is\s+confusing\b",
        re.IGNORECASE,
    ),
    "transport_residue": re.compile(
        r"\bno\s+reasoning\s+in\s+(?:the\s+)?content\b|"
        r"\b(?:internal|ev-)\s+(?:evidence\s+)?(?:citations?|tokens?)\b|"
        r"\bmodal\s+verb\s+may\b|\bmay\s+is\s+not\s+(?:a\s+)?month\b|"
        r"\bused\s+as\s+(?:a\s+)?month\b|"
        r"\bI['’]ll\s+(?:output|return|provide)\b|"
        r"\bthe\s+reasoning\s+is\s+done\b|"
        r"\bthe\s+paragraph\s+(?:must|should|will)\b|"
        r"\bno\s+mention\s+of\s+instructions?\b|"
        r"\bensure\b[^.?!]*\b(?:instructions?|answer|output|paragraph|citations?)\b",
        re.IGNORECASE,
    ),
}


class Chk3DataError(RuntimeError):
    """Base error for deterministic chk3 data-contract failures."""


class OutputContractError(Chk3DataError):
    """A DeepSeek response cannot become a chk3 SFT target."""

    def __init__(self, codes: Sequence[str]) -> None:
        normalized = tuple(dict.fromkeys(str(code) for code in codes if code))
        super().__init__(";".join(normalized) or "unknown_output_contract_error")
        self.codes = normalized or ("unknown_output_contract_error",)


class ModelDriftError(Chk3DataError):
    """The provider identity changed within one immutable generation run."""


class TeacherBackend(Protocol):
    def generate(
        self,
        *,
        config: DeepSeekTeacherConfig,
        system_prompt: str,
        user_prompt: str,
        environment: Mapping[str, str] | None,
    ) -> DeepSeekTeacherResponse: ...


def _log(message: str) -> None:
    print(f"[chk3-v4-pro] {message}", flush=True)


@dataclass(frozen=True)
class PreparedRow:
    sample_id: str
    split: str
    source_index: int
    analysis: str
    user_prompt: str
    analysis_sha256: str
    prompt_sha256: str
    source_response_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PREPARED_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "split": self.split,
            "source_index": self.source_index,
            "analysis": self.analysis,
            "prompt": self.user_prompt,
            "analysis_sha256": self.analysis_sha256,
            "prompt_sha256": self.prompt_sha256,
            "source_response_sha256": self.source_response_sha256,
        }


@dataclass(frozen=True)
class ValidatedTarget:
    reasoning: str
    minutes: str
    completion: str
    raw_reasoning_sha256: str
    reasoning_was_sanitized: bool
    reasoning_removed_segments: int
    reasoning_token_count: int
    rendered_token_count: int


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Chk3DataError(f"missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Chk3DataError(f"invalid JSON in {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Chk3DataError(f"{label} must be a JSON object: {path}")
    return value


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise Chk3DataError(
                        f"invalid JSON in {label} at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise Chk3DataError(
                        f"{label} row must be an object at {path}:{line_number}"
                    )
                rows.append(row)
    except FileNotFoundError as exc:
        raise Chk3DataError(f"missing {label}: {path}") from exc
    return rows


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    serialized = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    _atomic_write(path, serialized)


def _store_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = canonical_json(dict(payload)) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != serialized:
            raise Chk3DataError(f"immutable cache collision: {path}")
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _resolve_release_file(
    release_root: Path, record: Mapping[str, Any], *, label: str
) -> Path:
    relative = record.get("path")
    expected_sha = record.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise Chk3DataError(f"{label}.path must be non-empty text")
    if not isinstance(expected_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha
    ):
        raise Chk3DataError(f"{label}.sha256 is invalid")
    path = (release_root / relative).resolve()
    try:
        path.relative_to(release_root.resolve())
    except ValueError as exc:
        raise Chk3DataError(f"{label} escapes chk1 release: {relative}") from exc
    if not path.is_file():
        raise Chk3DataError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_sha:
        raise Chk3DataError(
            f"{label} SHA256 mismatch: expected={expected_sha} observed={observed}"
        )
    return path


def extract_chk1_final_answer(response: str) -> str:
    if not isinstance(response, str):
        raise Chk3DataError("chk1 response must be text")
    if response.count("</think>") != 1:
        raise Chk3DataError("chk1 response must contain exactly one </think>")
    reasoning, answer = response.split("</think>", 1)
    if not reasoning.strip():
        raise Chk3DataError("chk1 response reasoning is empty")
    answer = answer.strip()
    if not answer:
        raise Chk3DataError("chk1 final answer is empty")
    found = [marker for marker in CONTROL_MARKERS if marker in answer]
    if found:
        raise Chk3DataError(
            f"chk1 final answer contains model-control markers: {found}"
        )
    return answer


def render_user_prompt(analysis: str) -> str:
    normalized = str(analysis).strip()
    if not normalized:
        raise Chk3DataError("analysis must be non-empty")
    payload = canonical_json({"analysis": normalized})
    rendered = USER_PROMPT_PREFIX + payload
    parsed = json.loads(rendered[len(USER_PROMPT_PREFIX) :])
    if set(parsed) != {"analysis"} or parsed["analysis"] != normalized:
        raise Chk3DataError("user prompt did not round-trip through its JSON boundary")
    return rendered


def prompt_contract(*, code_sha256: str) -> dict[str, Any]:
    teacher_config = DeepSeekTeacherConfig(
        model=TEACHER_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        max_tokens=MAX_TOKENS,
        reasoning_effort="high",
    )
    return {
        "schema_version": PROMPT_CONTRACT_SCHEMA_VERSION,
        "student_system_prompt": STUDENT_SYSTEM_PROMPT,
        "student_system_prompt_sha256": sha256_text(STUDENT_SYSTEM_PROMPT),
        "teacher_system_prompt": TEACHER_SYSTEM_PROMPT,
        "teacher_system_prompt_sha256": sha256_text(TEACHER_SYSTEM_PROMPT),
        "teacher_repair_system_prompt": TEACHER_REPAIR_SYSTEM_PROMPT,
        "teacher_repair_system_prompt_sha256": sha256_text(
            TEACHER_REPAIR_SYSTEM_PROMPT
        ),
        "user_prompt_prefix": USER_PROMPT_PREFIX,
        "user_prompt_prefix_sha256": sha256_text(USER_PROMPT_PREFIX),
        "teacher_contract": teacher_config.contract(),
        "teacher_contract_sha256": teacher_config.contract_sha256,
        "mapping": {
            "reasoning": "deterministically_sanitized(message.reasoning_content)",
            "raw_reasoning_provenance": "message.reasoning_content",
            "minutes": "message.content.answer",
            "completion": "reasoning + '\\n</think>\\n' + minutes",
            "opening_think_source": "student_tokenizer_chat_template",
            "answer_tags": False,
        },
        "validation": {
            "number_comparison": "exact_decimal_counter_with_unit_conversion",
            "allowed_number_conversions": [
                "thousand_million_billion_trillion",
                "percent_percentage_point_basis_point",
                "decimal_fraction",
            ],
            "month_comparison": "case_sensitive_canonical_set",
            "lowercase_may_is_modal": True,
            "unsupported_attributions": sorted(_ATTRIBUTION_PATTERNS),
            "reasoning_meta_categories": sorted(_REASONING_META_PATTERNS),
            "reasoning_transport_sanitization": "paragraph_then_sentence_filter",
            "min_reasoning_tokens": MIN_REASONING_TOKENS,
            "max_reasoning_tokens": MAX_REASONING_TOKENS,
            "max_rendered_tokens": MAX_TOKENS,
            "truncation": False,
        },
        "source_policy": {
            "sample_universe": "immutable_chk1_canonical_release",
            "user_content": "chk1_final_answer_only",
            "prohibited": [
                "chk1_reasoning",
                "provided_data",
                "fact_card",
                "raw_minutes",
                "chk2_artifacts",
                "decision",
                "vote",
            ],
        },
        "code_sha256": code_sha256,
    }


def prepare_chk1_release(
    chk1_handoff_path: str | Path,
    *,
    output_root: str | Path,
    expected_counts: Mapping[str, int] = EXPECTED_SPLIT_COUNTS,
) -> tuple[dict[str, list[PreparedRow]], dict[str, Any]]:
    handoff_path = Path(chk1_handoff_path).resolve()
    handoff = _load_json(handoff_path, label="chk1 handoff")
    release_root = handoff_path.parent
    if handoff.get("schema_version") != "chk1-local-data-handoff-v1":
        raise Chk3DataError("unsupported chk1 handoff schema")
    if (
        handoff.get("quality_status") != "passed"
        or handoff.get("immutable") is not True
    ):
        raise Chk3DataError("chk1 handoff must be immutable and quality_status=passed")
    if handoff.get("split_counts") != dict(expected_counts):
        raise Chk3DataError(
            f"chk1 split counts differ from required chk3 population: "
            f"{handoff.get('split_counts')}"
        )

    split_files = handoff.get("split_files")
    manifest_files = handoff.get("manifest_files")
    if not isinstance(split_files, dict) or not isinstance(manifest_files, dict):
        raise Chk3DataError("chk1 handoff split file declarations are missing")

    prepared: dict[str, list[PreparedRow]] = {}
    observed_ids: set[str] = set()
    for split in SPLITS:
        source_path = _resolve_release_file(
            release_root, split_files.get(split, {}), label=f"split_files.{split}"
        )
        manifest_path = _resolve_release_file(
            release_root,
            manifest_files.get(split, {}),
            label=f"manifest_files.{split}",
        )
        source_rows = _load_jsonl(source_path, label=f"chk1 {split} SFT")
        source_manifests = _load_jsonl(manifest_path, label=f"chk1 {split} manifest")
        expected = int(expected_counts[split])
        if len(source_rows) != expected or len(source_manifests) != expected:
            raise Chk3DataError(
                f"chk1 {split} row count mismatch: "
                f"sft={len(source_rows)} manifest={len(source_manifests)} expected={expected}"
            )

        output_rows: list[PreparedRow] = []
        for index, (source, manifest) in enumerate(
            zip(source_rows, source_manifests, strict=True)
        ):
            if set(source) != {"prompt", "provided_data", "response"}:
                raise Chk3DataError(
                    f"chk1 {split}[{index}] SFT schema changed: {sorted(source)}"
                )
            sample_id = manifest.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise Chk3DataError(f"chk1 {split}[{index}] sample_id is invalid")
            if sample_id in observed_ids:
                raise Chk3DataError(f"duplicate chk1 sample_id: {sample_id}")
            observed_ids.add(sample_id)
            if manifest.get("split") != split:
                raise Chk3DataError(f"chk1 manifest split mismatch: {sample_id}")
            response = source.get("response")
            if not isinstance(response, str):
                raise Chk3DataError(f"chk1 response is not text: {sample_id}")
            response_sha = sha256_text(response)
            if manifest.get("response_sha256") != response_sha:
                raise Chk3DataError(f"chk1 response hash mismatch: {sample_id}")
            analysis = extract_chk1_final_answer(response)
            user_prompt = render_user_prompt(analysis)
            output_rows.append(
                PreparedRow(
                    sample_id=sample_id,
                    split=split,
                    source_index=index,
                    analysis=analysis,
                    user_prompt=user_prompt,
                    analysis_sha256=sha256_text(analysis),
                    prompt_sha256=sha256_text(user_prompt),
                    source_response_sha256=response_sha,
                )
            )
        prepared[split] = output_rows

    if len(observed_ids) != sum(expected_counts.values()):
        raise Chk3DataError(
            f"chk3 population mismatch: observed={len(observed_ids)} "
            f"expected={sum(expected_counts.values())}"
        )

    output = Path(output_root).resolve()
    for parent, child in (
        (release_root.resolve(), output),
        (output, release_root.resolve()),
    ):
        try:
            child.relative_to(parent)
        except ValueError:
            continue
        raise Chk3DataError(
            "chk3 output root must not overlap the immutable chk1 release: "
            f"output={output} source={release_root.resolve()}"
        )
    code_sha = sha256_file(Path(__file__).resolve())
    contract = prompt_contract(code_sha256=code_sha)
    _write_json(output / "prompt_contract.json", contract)
    for split in SPLITS:
        _write_jsonl(
            output / "prepared" / f"{split}.jsonl",
            [row.as_dict() for row in prepared[split]],
        )
    preparation = {
        "schema_version": "chk3-v4-pro-preparation-summary-v2",
        "status": "prepared",
        "source_release_id": handoff.get("release_id"),
        "source_handoff_path": str(handoff_path),
        "source_handoff_sha256": sha256_file(handoff_path),
        "split_counts": {split: len(prepared[split]) for split in SPLITS},
        "total_rows": len(observed_ids),
        "prompt_contract_sha256": sha256_text(canonical_json(contract)),
        "code_sha256": code_sha,
        "chk2_dependency": False,
        "model_facing_source": "chk1_final_answer_only",
    }
    return prepared, preparation


def _decimal_number_atom(raw: str) -> Decimal:
    normalized = raw.replace(",", "")
    mixed = re.fullmatch(r"(?P<whole>\d+)-(?P<num>\d+)/(?P<den>\d+)", normalized)
    if mixed is not None:
        denominator = Decimal(mixed.group("den"))
        if denominator == 0:
            raise InvalidOperation("fraction denominator is zero")
        return Decimal(mixed.group("whole")) + (
            Decimal(mixed.group("num")) / denominator
        )
    fraction = re.fullmatch(r"(?P<num>\d+)/(?P<den>\d+)", normalized)
    if fraction is not None:
        denominator = Decimal(fraction.group("den"))
        if denominator == 0:
            raise InvalidOperation("fraction denominator is zero")
        return Decimal(fraction.group("num")) / denominator
    return Decimal(normalized)


def _canonical_decimal(value: Decimal) -> str:
    if value == 0:
        value = Decimal(0)
    return format(value.normalize(), "f")


def _numeric_values(text: str) -> Counter[str]:
    values: Counter[str] = Counter()
    without_internal_citations = _EVIDENCE_ID_RE.sub("", text)
    without_calendar_days = _MONTH_DAY_RE.sub(
        lambda match: match.group("month"), without_internal_citations
    )
    for match in _NUMBER_EXPRESSION_RE.finditer(without_calendar_days):
        raw_number = match.group("number")
        # Calendar years are dates rather than economic quantities.  Compare
        # them as a set in _date_values so repeating a year after several
        # month names does not create a false unsupported-number error.
        if (
            not match.group("scale")
            and not match.group("rate")
            and re.fullmatch(r"(?:19|20)\d{2}", raw_number)
        ):
            continue
        try:
            value = _decimal_number_atom(raw_number)
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


def _counter_values(counter: Counter[str]) -> str:
    rendered: list[str] = []
    for value in sorted(counter, key=lambda item: Decimal(item)):
        count = counter[value]
        rendered.append(value if count == 1 else f"{value}*{count}")
    return ",".join(rendered)


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
    without_internal_citations = _EVIDENCE_ID_RE.sub("", text)
    result: set[str] = set()
    for match in _MONTH_RE.finditer(without_internal_citations):
        value = match.group(0).casefold().rstrip(".")
        result.add(aliases.get(value, value))
    for match in _MONTH_DAY_RE.finditer(without_internal_citations):
        month = match.group("month").casefold().rstrip(".")
        month = aliases.get(month, month)
        result.add(f"{month}-{int(match.group('day'))}")
    result.update(
        match.group(0) for match in _YEAR_RE.finditer(without_internal_citations)
    )
    return result


def _attribution_categories(text: str) -> set[str]:
    return {
        category
        for category, patterns in _ATTRIBUTION_PATTERNS.items()
        if any(pattern.search(text) for pattern in patterns)
    }


def _reasoning_meta_categories(text: str) -> set[str]:
    return {
        category
        for category, pattern in _REASONING_META_PATTERNS.items()
        if pattern.search(text)
    }


def _normalized_prose(text: str) -> str:
    return " ".join(_EVIDENCE_ID_RE.sub("", text).casefold().split())


def _sanitize_reasoning(
    raw_reasoning: str, *, analysis: str, minutes: str
) -> tuple[str, int]:
    """Remove provider transport chatter while retaining substantive planning.

    DeepSeek's native reasoning can acknowledge the provider's required JSON
    envelope even when explicitly instructed not to.  Those transport-only
    sentences are not useful student targets, so the projection removes them
    deterministically while preserving the untouched provider response in the
    cache provenance.
    """

    normalized_minutes = _normalized_prose(minutes)
    kept_paragraphs: list[str] = []
    removed_segments = 0
    for raw_paragraph in re.split(r"\n\s*\n", str(raw_reasoning).strip()):
        paragraph = raw_paragraph.strip()
        if not paragraph:
            continue
        # Providers often quote the complete source and then continue with a
        # useful fidelity plan in the same paragraph.  Remove only the exact
        # quoted source instead of discarding the surrounding plan.
        if analysis and analysis in paragraph:
            paragraph = paragraph.replace(analysis, " ").strip()
            removed_segments += 1
            if not paragraph:
                continue
        if minutes and minutes in paragraph:
            paragraph = paragraph.replace(minutes, " ").strip()
            removed_segments += 1
            if not paragraph:
                continue
        normalized_paragraph = _normalized_prose(paragraph)
        if len(normalized_minutes) >= 80 and normalized_minutes in normalized_paragraph:
            removed_segments += 1
            continue
        kept_sentences: list[str] = []
        for raw_sentence in re.split(r"(?<=[.!?])\s+", paragraph):
            sentence = raw_sentence.strip()
            if not sentence:
                continue
            if _reasoning_meta_categories(sentence):
                removed_segments += 1
                continue
            kept_sentences.append(sentence)
        if kept_sentences:
            kept_paragraphs.append(" ".join(kept_sentences))
    return "\n\n".join(kept_paragraphs).strip(), removed_segments


def _strict_content_answer(response: DeepSeekTeacherResponse) -> str:
    raw = str(response.raw_content or "").strip()
    if not raw:
        raise OutputContractError(("empty_content",))
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OutputContractError((f"content_not_json:{exc.msg}",)) from exc
    if not isinstance(payload, dict) or set(payload) != {"answer"}:
        raise OutputContractError(("content_schema_must_be_answer_only",))
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise OutputContractError(("empty_answer",))
    if response.answer.strip() != answer.strip():
        raise OutputContractError(("parsed_answer_mismatch",))
    return answer.strip()


def build_sft_completion(reasoning: str, minutes: str) -> str:
    clean_reasoning = str(reasoning).strip()
    clean_minutes = str(minutes).strip()
    errors: list[str] = []
    if not clean_reasoning:
        errors.append("empty_reasoning_content")
    if not clean_minutes:
        errors.append("empty_minutes_answer")
    for field_name, value in (
        ("reasoning", clean_reasoning),
        ("minutes", clean_minutes),
    ):
        found = [marker for marker in CONTROL_MARKERS if marker in value]
        if found:
            errors.append(f"{field_name}_contains_control_markers:{','.join(found)}")
    if errors:
        raise OutputContractError(errors)
    completion = f"{clean_reasoning}\n</think>\n{clean_minutes}"
    if completion.count("</think>") != 1:
        raise OutputContractError(("completion_boundary_count",))
    return completion


def _render_student_prompt(tokenizer: Any, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if not isinstance(rendered, str) or not rendered.endswith("<think>\n"):
        raise Chk3DataError(
            "student tokenizer chat template must end with '<think>\\n'"
        )
    return rendered


def _token_count(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def _token_stats(values: Sequence[int]) -> dict[str, int]:
    if not values:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0}
    ordered = sorted(int(value) for value in values)

    def percentile(fraction: float) -> int:
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def validate_teacher_target(
    *,
    response: DeepSeekTeacherResponse,
    analysis: str,
    user_prompt: str,
    tokenizer: Any,
    max_length: int = MAX_TOKENS,
    min_reasoning_tokens: int = MIN_REASONING_TOKENS,
    max_reasoning_tokens: int = MAX_REASONING_TOKENS,
) -> ValidatedTarget:
    errors: list[str] = []
    if response.finish_reason not in {"stop", "end_turn"}:
        errors.append(f"finish_reason:{response.finish_reason}")
    try:
        minutes = _strict_content_answer(response)
    except OutputContractError as exc:
        errors.extend(exc.codes)
        minutes = ""
    raw_reasoning = response.analysis.strip()
    reasoning, reasoning_removed_segments = _sanitize_reasoning(
        raw_reasoning,
        analysis=analysis,
        minutes=minutes,
    )
    try:
        completion = build_sft_completion(reasoning, minutes)
    except OutputContractError as exc:
        errors.extend(exc.codes)
        completion = ""

    analysis_numbers = _numeric_values(analysis)
    minutes_numbers = _numeric_values(minutes)
    missing_numbers = analysis_numbers - minutes_numbers
    unsupported_numbers = minutes_numbers - analysis_numbers
    if missing_numbers:
        errors.append("missing_numbers:" + _counter_values(missing_numbers))
    if unsupported_numbers:
        errors.append("unsupported_numbers:" + _counter_values(unsupported_numbers))
    analysis_dates = _date_values(analysis)
    minutes_dates = _date_values(minutes)
    missing_dates = sorted(analysis_dates - minutes_dates)
    unsupported_dates = sorted(minutes_dates - analysis_dates)
    if missing_dates:
        errors.append("missing_dates:" + ",".join(missing_dates))
    if unsupported_dates:
        errors.append("unsupported_dates:" + ",".join(unsupported_dates))
    unsupported_attributions = sorted(
        _attribution_categories(minutes) - _attribution_categories(analysis)
    )
    if unsupported_attributions:
        errors.append("unsupported_attributions:" + ",".join(unsupported_attributions))
    reasoning_meta = sorted(_reasoning_meta_categories(reasoning))
    if reasoning_meta:
        errors.append("reasoning_contains_meta:" + ",".join(reasoning_meta))
    normalized_analysis = _normalized_prose(analysis)
    normalized_reasoning = _normalized_prose(reasoning)
    if len(normalized_analysis) >= 80 and normalized_analysis in normalized_reasoning:
        errors.append("reasoning_repeats_complete_analysis")
    normalized_minutes = _normalized_prose(minutes)
    if (
        len(normalized_minutes) >= 80
        and normalized_reasoning.count(normalized_minutes) > 1
    ):
        errors.append("reasoning_repeats_final_draft")
    if _EVIDENCE_ID_RE.search(minutes):
        errors.append("minutes_contains_internal_evidence_id")
    if "\n" in minutes:
        errors.append("minutes_not_one_paragraph")

    rendered = _render_student_prompt(tokenizer, user_prompt)
    reasoning_token_count = _token_count(tokenizer, reasoning) if reasoning else 0
    if reasoning and reasoning_token_count < min_reasoning_tokens:
        errors.append(
            f"reasoning_tokens:{reasoning_token_count}<{min_reasoning_tokens}"
        )
    if reasoning and reasoning_token_count > max_reasoning_tokens:
        errors.append(
            f"reasoning_tokens:{reasoning_token_count}>{max_reasoning_tokens}"
        )
    rendered_token_count = (
        _token_count(tokenizer, rendered + completion) if completion else 0
    )
    if completion and rendered_token_count > max_length:
        errors.append(f"student_total_tokens:{rendered_token_count}>{max_length}")
    if errors:
        raise OutputContractError(errors)
    return ValidatedTarget(
        reasoning=reasoning,
        minutes=minutes,
        completion=completion,
        raw_reasoning_sha256=sha256_text(raw_reasoning),
        reasoning_was_sanitized=reasoning != raw_reasoning,
        reasoning_removed_segments=reasoning_removed_segments,
        reasoning_token_count=reasoning_token_count,
        rendered_token_count=rendered_token_count,
    )


class ProviderIdentityGuard:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._identity: tuple[str, str] | None = None

    @property
    def identity(self) -> tuple[str, str] | None:
        with self._lock:
            return self._identity

    def bind(self, *, returned_model: str, system_fingerprint: str) -> None:
        model = str(returned_model).strip()
        fingerprint = str(system_fingerprint).strip()
        if model != TEACHER_MODEL:
            raise ModelDriftError(
                f"provider returned {model!r}, expected exactly {TEACHER_MODEL!r}"
            )
        if not fingerprint or fingerprint == "unavailable":
            raise ModelDriftError("provider system_fingerprint is unavailable")
        identity = (model, fingerprint)
        with self._lock:
            if self._identity is None:
                self._identity = identity
            elif self._identity != identity:
                raise ModelDriftError(
                    f"provider identity drift: expected={self._identity} observed={identity}"
                )


def _teacher_config() -> DeepSeekTeacherConfig:
    return DeepSeekTeacherConfig(
        model=TEACHER_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        api_key_env=API_KEY_ENV,
        max_tokens=MAX_TOKENS,
        reasoning_effort="high",
    )


def _cache_key(row: PreparedRow, *, code_sha256: str) -> str:
    contract = prompt_contract(code_sha256=code_sha256)
    payload = {
        "schema_version": TEACHER_CACHE_SCHEMA_VERSION,
        "sample_id": row.sample_id,
        "split": row.split,
        "analysis_sha256": row.analysis_sha256,
        "prompt_sha256": row.prompt_sha256,
        "prompt_contract_sha256": sha256_text(canonical_json(contract)),
        "teacher_contract_sha256": contract["teacher_contract_sha256"],
        "code_sha256": code_sha256,
    }
    return sha256_text(canonical_json(payload))


def _accepted_cache_path(output_root: Path, cache_key: str) -> Path:
    return output_root / "cache" / "accepted" / cache_key[:2] / f"{cache_key}.json"


def _rejected_cache_path(
    output_root: Path,
    *,
    cache_key: str,
    response: DeepSeekTeacherResponse | None,
    error: str,
) -> Path:
    response_id = str(response.response_id).strip() if response is not None else ""
    identity = response_id or sha256_text(error)[:24]
    safe_identity = re.sub(r"[^A-Za-z0-9_.-]", "_", identity)
    return (
        output_root
        / "cache"
        / "rejected"
        / cache_key[:2]
        / cache_key
        / f"{safe_identity}.json"
    )


def _response_provenance(response: DeepSeekTeacherResponse) -> dict[str, Any]:
    return {
        "response_id": response.response_id,
        "returned_model": response.returned_model,
        "system_fingerprint": response.system_fingerprint,
        "finish_reason": response.finish_reason,
        "created": response.created,
        "usage": dict(response.usage),
    }


def _accepted_payload(
    *,
    row: PreparedRow,
    response: DeepSeekTeacherResponse,
    target: ValidatedTarget,
    cache_key: str,
    code_sha256: str,
    attempt: str,
) -> dict[str, Any]:
    return {
        "schema_version": TEACHER_CACHE_SCHEMA_VERSION,
        "status": "accepted",
        "cache_key": cache_key,
        "sample_id": row.sample_id,
        "split": row.split,
        "source_index": row.source_index,
        "analysis_sha256": row.analysis_sha256,
        "prompt_sha256": row.prompt_sha256,
        "source_response_sha256": row.source_response_sha256,
        "code_sha256": code_sha256,
        "attempt": attempt,
        "teacher": _response_provenance(response),
        "provider_raw": {
            "reasoning_content": response.analysis,
            "content": response.raw_content,
        },
        "target": {
            "reasoning": target.reasoning,
            "minutes": target.minutes,
            "completion": target.completion,
            "raw_reasoning_sha256": target.raw_reasoning_sha256,
            "reasoning_was_sanitized": target.reasoning_was_sanitized,
            "reasoning_removed_segments": target.reasoning_removed_segments,
            "reasoning_token_count": target.reasoning_token_count,
            "rendered_token_count": target.rendered_token_count,
            "reasoning_sha256": sha256_text(target.reasoning),
            "minutes_sha256": sha256_text(target.minutes),
            "completion_sha256": sha256_text(target.completion),
        },
    }


def _validate_cached_payload(
    payload: Mapping[str, Any],
    *,
    row: PreparedRow,
    cache_key: str,
    code_sha256: str,
) -> dict[str, Any]:
    if (
        payload.get("schema_version") != TEACHER_CACHE_SCHEMA_VERSION
        or payload.get("status") != "accepted"
        or payload.get("cache_key") != cache_key
        or payload.get("sample_id") != row.sample_id
        or payload.get("split") != row.split
        or payload.get("analysis_sha256") != row.analysis_sha256
        or payload.get("prompt_sha256") != row.prompt_sha256
        or payload.get("source_response_sha256") != row.source_response_sha256
        or payload.get("code_sha256") != code_sha256
    ):
        raise Chk3DataError(f"accepted cache binding mismatch: {row.sample_id}")
    target = payload.get("target")
    teacher = payload.get("teacher")
    provider_raw = payload.get("provider_raw")
    if (
        not isinstance(target, dict)
        or not isinstance(teacher, dict)
        or not isinstance(provider_raw, dict)
    ):
        raise Chk3DataError(f"accepted cache schema is incomplete: {row.sample_id}")
    completion = target.get("completion")
    reasoning = target.get("reasoning")
    minutes = target.get("minutes")
    if (
        not isinstance(completion, str)
        or not isinstance(reasoning, str)
        or not isinstance(minutes, str)
        or build_sft_completion(reasoning, minutes) != completion
        or target.get("reasoning_sha256") != sha256_text(reasoning)
        or target.get("minutes_sha256") != sha256_text(minutes)
        or target.get("completion_sha256") != sha256_text(completion)
        or not isinstance(target.get("raw_reasoning_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", target["raw_reasoning_sha256"])
        or target.get("raw_reasoning_sha256")
        != sha256_text(str(provider_raw.get("reasoning_content", "")).strip())
        or not isinstance(target.get("reasoning_was_sanitized"), bool)
        or not isinstance(target.get("reasoning_removed_segments"), int)
        or target.get("reasoning_removed_segments", -1) < 0
        or not isinstance(target.get("reasoning_token_count"), int)
        or target.get("reasoning_token_count", 0) < MIN_REASONING_TOKENS
        or target.get("reasoning_token_count", 0) > MAX_REASONING_TOKENS
        or not isinstance(target.get("rendered_token_count"), int)
        or target.get("rendered_token_count", 0) > MAX_TOKENS
    ):
        raise Chk3DataError(f"accepted cache target is invalid: {row.sample_id}")
    return dict(payload)


def _rejection_payload(
    *,
    row: PreparedRow,
    cache_key: str,
    attempt: str,
    error: str,
    response: DeepSeekTeacherResponse | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": TEACHER_CACHE_SCHEMA_VERSION,
        "status": "rejected",
        "cache_key": cache_key,
        "sample_id": row.sample_id,
        "split": row.split,
        "source_index": row.source_index,
        "attempt": attempt,
        "error": error,
    }
    if response is not None:
        payload["teacher"] = _response_provenance(response)
        payload["provider_raw"] = {
            "reasoning_content": response.analysis,
            "content": response.raw_content,
        }
    return payload


def _repair_user_prompt(row: PreparedRow, errors: Sequence[str]) -> str:
    return (
        row.user_prompt
        + "\n\n"
        + canonical_json({"output_contract_errors": list(errors)})
    )


def generate_one_target(
    row: PreparedRow,
    *,
    output_root: str | Path,
    tokenizer: Any,
    backend: TeacherBackend,
    identity_guard: ProviderIdentityGuard,
    environment: Mapping[str, str] | None,
    code_sha256: str,
    max_length: int = MAX_TOKENS,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    config = _teacher_config()
    cache_key = _cache_key(row, code_sha256=code_sha256)
    attempts = (
        ("primary", TEACHER_SYSTEM_PROMPT, row.user_prompt),
        ("repair", TEACHER_REPAIR_SYSTEM_PROMPT, None),
    )
    prior_errors: tuple[str, ...] = ()
    for attempt, system_prompt, user_prompt in attempts:
        effective_user_prompt = (
            user_prompt
            if user_prompt is not None
            else _repair_user_prompt(row, prior_errors)
        )
        response: DeepSeekTeacherResponse | None = None
        try:
            response = backend.generate(
                config=config,
                system_prompt=system_prompt,
                user_prompt=effective_user_prompt,
                environment=environment,
            )
            identity_guard.bind(
                returned_model=response.returned_model,
                system_fingerprint=response.system_fingerprint,
            )
            target = validate_teacher_target(
                response=response,
                analysis=row.analysis,
                user_prompt=row.user_prompt,
                tokenizer=tokenizer,
                max_length=max_length,
            )
            payload = _accepted_payload(
                row=row,
                response=response,
                target=target,
                cache_key=cache_key,
                code_sha256=code_sha256,
                attempt=attempt,
            )
            _store_immutable_json(_accepted_cache_path(output, cache_key), payload)
            return payload
        except ModelDriftError:
            error = "model_drift"
            rejected = _rejection_payload(
                row=row,
                cache_key=cache_key,
                attempt=attempt,
                error=error,
                response=response,
            )
            _store_immutable_json(
                _rejected_cache_path(
                    output,
                    cache_key=cache_key,
                    response=response,
                    error=error,
                ),
                rejected,
            )
            raise
        except OutputContractError as exc:
            prior_errors = exc.codes
            error = str(exc)
            rejected = _rejection_payload(
                row=row,
                cache_key=cache_key,
                attempt=attempt,
                error=error,
                response=response,
            )
            _store_immutable_json(
                _rejected_cache_path(
                    output,
                    cache_key=cache_key,
                    response=response,
                    error=f"{attempt}:{error}",
                ),
                rejected,
            )
            continue
        except Exception as exc:  # noqa: BLE001 - provider exceptions vary
            error = f"{type(exc).__name__}:{exc}"
            rejected = _rejection_payload(
                row=row,
                cache_key=cache_key,
                attempt=attempt,
                error=error,
                response=response,
            )
            _store_immutable_json(
                _rejected_cache_path(
                    output,
                    cache_key=cache_key,
                    response=response,
                    error=f"{attempt}:{error}",
                ),
                rejected,
            )
            return {
                "status": "failed",
                "sample_id": row.sample_id,
                "split": row.split,
                "source_index": row.source_index,
                "error": error,
            }
    return {
        "status": "failed",
        "sample_id": row.sample_id,
        "split": row.split,
        "source_index": row.source_index,
        "error": ";".join(prior_errors),
    }


def _load_tokenizer(path: str | Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment contract
        raise Chk3DataError("transformers is unavailable") from exc
    tokenizer_path = Path(path).resolve()
    if not tokenizer_path.is_dir():
        raise Chk3DataError(f"tokenizer path is missing: {tokenizer_path}")
    return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


def _flatten_prepared(
    prepared: Mapping[str, Sequence[PreparedRow]],
) -> list[PreparedRow]:
    return [row for split in SPLITS for row in prepared.get(split, ())]


def _load_resume_cache(
    rows: Sequence[PreparedRow],
    *,
    output_root: Path,
    code_sha256: str,
    resume: bool,
    identity_guard: ProviderIdentityGuard,
) -> dict[str, dict[str, Any]]:
    accepted: dict[str, dict[str, Any]] = {}
    existing_paths: list[Path] = []
    for row in rows:
        key = _cache_key(row, code_sha256=code_sha256)
        path = _accepted_cache_path(output_root, key)
        if not path.exists():
            continue
        existing_paths.append(path)
        if not resume:
            continue
        payload = _validate_cached_payload(
            _load_json(path, label="accepted teacher cache"),
            row=row,
            cache_key=key,
            code_sha256=code_sha256,
        )
        teacher = payload["teacher"]
        identity_guard.bind(
            returned_model=str(teacher.get("returned_model", "")),
            system_fingerprint=str(teacher.get("system_fingerprint", "")),
        )
        accepted[row.sample_id] = payload
    if existing_paths and not resume:
        raise Chk3DataError(
            "accepted chk3 cache already exists; use --resume after verifying the "
            "same immutable inputs"
        )
    return accepted


def _materialize_outputs(
    *,
    prepared: Mapping[str, Sequence[PreparedRow]],
    accepted: Mapping[str, Mapping[str, Any]],
    failures: Mapping[str, Mapping[str, Any]],
    output_root: Path,
    source_handoff_sha256: str,
    identity: tuple[str, str] | None,
) -> dict[str, Any]:
    split_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        teacher_rows: list[dict[str, Any]] = []
        sft_rows: list[dict[str, str]] = []
        manifest_rows: list[dict[str, Any]] = []
        for row in prepared[split]:
            payload = accepted.get(row.sample_id)
            if payload is None:
                continue
            target = payload["target"]
            teacher = payload["teacher"]
            teacher_rows.append(
                {
                    "sample_id": row.sample_id,
                    "split": split,
                    "source_index": row.source_index,
                    "prompt_sha256": row.prompt_sha256,
                    "reasoning_content": target["reasoning"],
                    "content": payload["provider_raw"]["content"],
                    "answer": target["minutes"],
                    "cache_key": payload["cache_key"],
                    "teacher": teacher,
                    "status": "accepted",
                }
            )
            sft_rows.append(
                {"prompt": row.user_prompt, "response": target["completion"]}
            )
            manifest_rows.append(
                {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "sample_id": row.sample_id,
                    "split": split,
                    "source_index": row.source_index,
                    "source_handoff_sha256": source_handoff_sha256,
                    "source_response_sha256": row.source_response_sha256,
                    "analysis_sha256": row.analysis_sha256,
                    "prompt_sha256": row.prompt_sha256,
                    "reasoning_sha256": target["reasoning_sha256"],
                    "raw_reasoning_sha256": target["raw_reasoning_sha256"],
                    "reasoning_was_sanitized": target["reasoning_was_sanitized"],
                    "reasoning_removed_segments": target["reasoning_removed_segments"],
                    "minutes_sha256": target["minutes_sha256"],
                    "response_sha256": target["completion_sha256"],
                    "reasoning_token_count": target["reasoning_token_count"],
                    "rendered_token_count": target["rendered_token_count"],
                    "cache_key": payload["cache_key"],
                    "teacher": teacher,
                }
            )
        _write_jsonl(output_root / "teacher_responses" / f"{split}.jsonl", teacher_rows)
        _write_jsonl(output_root / "sft" / f"{split}.jsonl", sft_rows)
        _write_jsonl(output_root / "manifests" / f"{split}.jsonl", manifest_rows)
        split_counts[split] = {
            "prepared": len(prepared[split]),
            "accepted": len(sft_rows),
            "pending_or_failed": len(prepared[split]) - len(sft_rows),
        }

    ordered_failures = sorted(
        (dict(value) for value in failures.values()),
        key=lambda item: (
            SPLITS.index(str(item.get("split"))),
            int(item.get("source_index", -1)),
        ),
    )
    _write_jsonl(output_root / "failures.jsonl", ordered_failures)
    total_accepted = len(accepted)
    total_prepared = sum(len(prepared[split]) for split in SPLITS)
    summary = {
        "schema_version": "chk3-v4-pro-generation-summary-v2",
        "status": "complete" if total_accepted == total_prepared else "incomplete",
        "teacher_model": TEACHER_MODEL,
        "provider_identity": (
            {"returned_model": identity[0], "system_fingerprint": identity[1]}
            if identity is not None
            else None
        ),
        "split_counts": split_counts,
        "total_prepared": total_prepared,
        "total_accepted": total_accepted,
        "total_failed_or_pending": total_prepared - total_accepted,
        "source_handoff_sha256": source_handoff_sha256,
        "chk2_dependency": False,
        "min_reasoning_tokens": MIN_REASONING_TOKENS,
        "max_reasoning_tokens": MAX_REASONING_TOKENS,
        "max_rendered_tokens": MAX_TOKENS,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def run_generation(
    prepared: Mapping[str, Sequence[PreparedRow]],
    *,
    output_root: str | Path,
    source_handoff_sha256: str,
    tokenizer: Any,
    backend: TeacherBackend,
    environment: Mapping[str, str] | None,
    concurrency: int = DEFAULT_CONCURRENCY,
    resume: bool = False,
    max_length: int = MAX_TOKENS,
) -> dict[str, Any]:
    if concurrency <= 0:
        raise Chk3DataError("concurrency must be positive")
    output = Path(output_root).resolve()
    rows = _flatten_prepared(prepared)
    code_sha = sha256_file(Path(__file__).resolve())
    identity_guard = ProviderIdentityGuard()
    accepted = _load_resume_cache(
        rows,
        output_root=output,
        code_sha256=code_sha,
        resume=resume,
        identity_guard=identity_guard,
    )
    pending = [row for row in rows if row.sample_id not in accepted]
    failures: dict[str, dict[str, Any]] = {}
    stop_event = threading.Event()
    _log(
        f"population={len(rows)} cache_hits={len(accepted)} pending={len(pending)} "
        f"concurrency={concurrency} resume={resume}"
    )

    def worker(row: PreparedRow) -> dict[str, Any]:
        if stop_event.is_set():
            return {
                "status": "failed",
                "sample_id": row.sample_id,
                "split": row.split,
                "source_index": row.source_index,
                "error": "not_attempted_after_model_drift",
            }
        try:
            return generate_one_target(
                row,
                output_root=output,
                tokenizer=tokenizer,
                backend=backend,
                identity_guard=identity_guard,
                environment=environment,
                code_sha256=code_sha,
                max_length=max_length,
            )
        except ModelDriftError as exc:
            stop_event.set()
            return {
                "status": "failed",
                "sample_id": row.sample_id,
                "split": row.split,
                "source_index": row.source_index,
                "error": f"ModelDriftError:{exc}",
            }

    if pending:
        with ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="chk3-deepseek"
        ) as executor:
            futures: dict[Future[dict[str, Any]], PreparedRow] = {
                executor.submit(worker, row): row for row in pending
            }
            for future in as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "status": "failed",
                        "sample_id": row.sample_id,
                        "split": row.split,
                        "source_index": row.source_index,
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                if result.get("status") == "accepted":
                    accepted[row.sample_id] = result
                else:
                    failures[row.sample_id] = result
                completed = len(accepted) + len(failures)
                _log(
                    f"progress={completed}/{len(rows)} split={row.split} "
                    f"sample_id={row.sample_id} status={result.get('status')} "
                    f"accepted={len(accepted)} failed={len(failures)}"
                )

    return _materialize_outputs(
        prepared=prepared,
        accepted=accepted,
        failures=failures,
        output_root=output,
        source_handoff_sha256=source_handoff_sha256,
        identity=identity_guard.identity,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare chk1-final-answer inputs and optionally generate chk3 "
            "reasoning+Minutes SFT targets with DeepSeek V4 Pro."
        )
    )
    parser.add_argument("--chk1-handoff", default=str(DEFAULT_CHK1_HANDOFF))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--tokenizer-path", default=str(DEFAULT_TOKENIZER_PATH))
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("DEEPSEEK_CONCURRENCY", DEFAULT_CONCURRENCY)),
    )
    parser.add_argument("--max-length", type=int, default=MAX_TOKENS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare and validate all prompts without reading credentials or calling API.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only immutable accepted caches with exact current bindings.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root).resolve()
    prepared, preparation = prepare_chk1_release(
        args.chk1_handoff, output_root=output_root
    )
    _log(
        f"prepared={preparation['total_rows']} splits={preparation['split_counts']} "
        f"source_release={preparation['source_release_id']}"
    )
    tokenizer = _load_tokenizer(args.tokenizer_path)
    prompt_token_counts: list[int] = []
    for row in _flatten_prepared(prepared):
        rendered = _render_student_prompt(tokenizer, row.user_prompt)
        prompt_tokens = _token_count(tokenizer, rendered)
        prompt_token_counts.append(prompt_tokens)
        if prompt_tokens >= args.max_length:
            raise Chk3DataError(
                f"student prompt alone exceeds max_length for {row.sample_id}: "
                f"{prompt_tokens}>={args.max_length}"
            )
    preparation["prompt_token_stats"] = _token_stats(prompt_token_counts)
    preparation["student_system_prompt_sha256"] = sha256_text(STUDENT_SYSTEM_PROMPT)
    if args.dry_run:
        summary = dict(preparation)
        summary["status"] = "dry_run_ready"
        summary["max_length"] = args.max_length
        _write_json(output_root / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    environment = os.environ
    if not str(environment.get(API_KEY_ENV, "")).strip():
        raise Chk3DataError(
            f"missing {API_KEY_ENV}; dry-run preparation completed without API calls"
        )
    summary = run_generation(
        prepared,
        output_root=output_root,
        source_handoff_sha256=preparation["source_handoff_sha256"],
        tokenizer=tokenizer,
        backend=OpenAIDeepSeekBackend(),
        environment=environment,
        concurrency=args.concurrency,
        resume=args.resume,
        max_length=args.max_length,
    )
    summary["preparation"] = preparation
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Chk3DataError as exc:
        print(f"chk3 data generation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
