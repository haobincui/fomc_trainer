"""Build paper-CHK2 synthetic Minutes rewrites with independent validation.

The immutable 2,083-row target-derived release is the sole source universe.
For every row, the rewrite teacher sees only the de-styled ``source_analysis``.
Its native ``reasoning_content`` and JSON ``content.answer`` become the SFT
completion::

    teacher_response_analysis\n</think>\nrewritten_minutes

Validator A is the training gate and sees only the source analysis, native
teacher reasoning, and synthetic rewrite.  Validator B sees the normalized
official Minutes paragraph and synthetic rewrite, but is diagnostic-only: its
score can emit a warning and can never change training eligibility or trigger
repair.  A single rewrite repair is allowed, using analysis-based diagnostics
only.  Official text is never present in a rewrite or Validator-A request.

Provider responses and terminal decisions are stored in immutable, hash-bound
per-role caches.  The materialized ``terminal`` ledgers contain one row for
every source row; the strict SFT files contain only ``{"prompt","response"}``
for rows passing deterministic validation and Validator A.
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
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from jobs.generation.generate_chk3_sft_targets import (
    CONTROL_MARKERS,
    _attribution_categories,
    _date_values,
    _numeric_values,
    _reasoning_meta_categories,
    canonical_json,
    render_user_prompt,
)
from jobs.retrain_v2.token_budget_gate import _count_sft_row


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_machine_ready_v2_20260829"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_synthetic_rewrite_v1_20260830"
)
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "models/DeepSeek-R1-Distill-Llama-8B"

SPLITS = ("train", "validation", "test")
EXPECTED_SPLIT_COUNTS = {"train": 1686, "validation": 221, "test": 176}
EXPECTED_TOTAL = 2083
EXPECTED_SOURCE_SUMMARY_SHA256 = (
    "7c8e9626ac6aaea435f2ca0a3e40e52dc3f9fc1b46e70f5f2f86acd53bb76c71"
)
EXPECTED_SOURCE_HANDOFF_SHA256 = (
    "dd0ed476709a4c2e74c80f84ec175ce606e868297d84ac1d46520951fdb6d554"
)
EXPECTED_SOURCE_MERGE_RECEIPT_SHA256 = (
    "00e2b5f40227d28615b5a0d2fd35446923698ec858c5f1100a53085aebbbc1ab"
)
EXPECTED_SOURCE_SAMPLE_ID_SHA256 = (
    "9703dc5964bdb21ca6609bcf1cda19df64227c61a3e7c78df3e0a4df0227186a"
)

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_CONCURRENCY = 8
MAX_CONCURRENCY = 32
DEFAULT_PREFLIGHT_ROWS = 8
MIN_REWRITE_WORDS = 20
MAX_REWRITE_WORDS = 400
MIN_REASONING_WORDS = 20

PREPARED_SCHEMA_VERSION = "paper-chk2-synthetic-rewrite-prepared-v1"
CACHE_SCHEMA_VERSION = "paper-chk2-synthetic-rewrite-provider-cache-v1"
TERMINAL_SCHEMA_VERSION = "paper-chk2-synthetic-rewrite-terminal-v1"
MANIFEST_SCHEMA_VERSION = "paper-chk2-synthetic-rewrite-manifest-v1"
SUMMARY_SCHEMA_VERSION = "paper-chk2-synthetic-rewrite-summary-v1"
PROMPT_CONTRACT_SCHEMA_VERSION = "paper-chk2-synthetic-rewrite-prompts-v1"

TERMINAL_PASS = "PASS"
TERMINAL_GENERATION_REJECT = "GENERATION_QUALITY_REJECT"
TERMINAL_FIDELITY_REJECT = "INPUT_FIDELITY_REJECT"
TERMINAL_STATUSES = {
    TERMINAL_PASS,
    TERMINAL_GENERATION_REJECT,
    TERMINAL_FIDELITY_REJECT,
}

ROLE_REWRITE_PRIMARY = "rewrite_primary"
ROLE_REWRITE_REPAIR = "rewrite_repair"
ROLE_VALIDATOR_A_PRIMARY = "validator_a_primary"
ROLE_VALIDATOR_A_REPAIR = "validator_a_repair"
ROLE_VALIDATOR_B = "validator_b"
PROVIDER_ROLES = (
    ROLE_REWRITE_PRIMARY,
    ROLE_REWRITE_REPAIR,
    ROLE_VALIDATOR_A_PRIMARY,
    ROLE_VALIDATOR_A_REPAIR,
    ROLE_VALIDATOR_B,
)

STUDENT_SYSTEM_PROMPT = """\
You are a Federal Reserve Minutes editor. The user supplies a complete
economic or financial analysis. Use the native reasoning section for the
complete reasoning process, including any useful deliberation about the task,
prompt, JSON transport, answer contract, length, or drafting. Within that full
reasoning trace, identify every substantive claim, quantity, date, direction,
comparison, attribution, causal relation, and expression of uncertainty that
the formal rewrite must preserve. Then express the same information as exactly
one formal FOMC Minutes paragraph.

Do not add, remove, broaden, narrow, or contradict any substantive claim.
Preserve the numeric-quantity multiset and every explicit calendar reference.
The final paragraph may reuse phrases, sentences, or extensive wording from
the analysis when that wording is already suitable; lexical overlap is not an
error. The final paragraph as a whole must not be a verbatim copy of the whole
analysis. Do not emit headings, lists, JSON, citations, answer tags, or
model-control tags in the final paragraph.
"""

STUDENT_USER_PROMPT_TEMPLATE = (
    "Rewrite the following analysis as formal FOMC Minutes prose:\n\n"
    '{"analysis":"[SOURCE_ANALYSIS]"}'
)
STUDENT_RESPONSE_CONTRACT = (
    "teacher_response_analysis + '\\n</think>\\n' + rewritten_minutes; "
    "the tokenizer chat template supplies the opening <think>; no answer tags"
)

REWRITE_SYSTEM_PROMPT = """\
You are the reasoning teacher for an analysis-to-FOMC-Minutes rewriting
dataset. The user supplies exactly one complete economic or financial
analysis. Use that analysis as the sole factual source.

Use the native reasoning channel for your complete, unabridged reasoning
process. It may include deliberation about the prompt, JSON, the answer field,
transport or formatting requirements, length, and drafting. This operational
reasoning is permitted and will be retained verbatim. It must also substantively
track the claims, quantities, dates, directions, comparisons, attributions,
causal relations, scope, and uncertainty that the final paragraph preserves.
Do not place model-control tags in either channel.

Return exactly one JSON object in content with the single key answer. The
answer must be a single 20--400 word formal FOMC Minutes-style paragraph that
is bidirectionally equivalent to the supplied analysis. Add no fact, entity,
quantity, date, attribution, cause, policy action, decision, or vote. Preserve
all quantities and explicit calendar references. You may reuse any source
phrase, sentence, or extensive wording when it is already appropriate FOMC
Minutes prose; near-copying is not an error. The complete answer must not be
verbatim identical to the complete source analysis. Do not emit headings,
lists, Markdown, citations, commentary, control tags, or text outside the JSON.
"""

REWRITE_REPAIR_SYSTEM_PROMPT = """\
You are a Federal Reserve fidelity editor. Use the supplied analysis as the
sole factual source and address the supplied repair feedback. Use the native
reasoning channel for the complete, unabridged reasoning process. Deliberation
about the prompt, JSON, answer field, transport or formatting requirements,
length, and drafting is permitted and will be retained verbatim. The reasoning
must also track the source facts relevant to the repaired paragraph. Do not
place model-control tags in either channel.

Content must be exactly one JSON object with the single key answer, whose
value is one 20--400 word formal FOMC Minutes-style paragraph. Preserve every
source claim, quantity, date, direction, comparison, attribution, causal
relation, scope, and expression of uncertainty, and add nothing. Source
phrases, sentences, and extensive wording may be reused; near-copying is not
an error. The complete answer must not be verbatim identical to the complete
source analysis. This is the only repair.
"""

VALIDATOR_A_SYSTEM_PROMPT = """\
Act as an adversarial fidelity verifier for analysis-to-FOMC-Minutes training
data. Judge only the supplied source_analysis, teacher_response_analysis, and
rewritten_minutes. Do not use outside knowledge. You will not receive an
official Minutes target. Judge semantic support rather than surface novelty;
lexical divergence is not required.

Decompose the source and rewrite into substantive claims. Check both
directions, and check whether the teacher reasoning covers every source claim
without endorsing a conflicting or unsupported source-domain fact. Operational
reasoning about prompts, instructions, JSON, answer fields, schemas, APIs,
transport, formatting, validation, length, or drafting is permitted. A draft
or rehearsal inside the reasoning is also permitted. Do not report either as
an issue, and do not mark reasoning incompatible merely because it contains
such material. Considering and rejecting a possible fact is not the same as
endorsing it. Reuse of source phrases or complete source sentences in the
rewrite is permitted and is not a fidelity issue.
Evidence strings must be exact substrings of the corresponding supplied text.
For a failed verdict, the unavailable evidence field must be null.

Return exactly one JSON object with exactly these keys:
- source_claims: nonempty array of objects with exactly source_span,
  reasoning_evidence, rewrite_evidence, rewrite_verdict, reasoning_verdict.
  rewrite_verdict is entailed, omitted, or contradicted; reasoning_verdict is
  covered, not_covered, or conflicted.
- rewrite_claims: nonempty array of objects with exactly rewrite_span,
  source_evidence, verdict, where verdict is supported, unsupported, or
  contradicted.
- reasoning_issues: array of objects with exactly reasoning_span and
  issue_type. issue_type is unsupported_claim, source_conflict,
  rewrite_conflict, or missing_claim_coverage. Use this array only for factual
  or coverage problems, never for operational deliberation or draft wording.
- bidirectional_entailment: boolean.
- reasoning_compatible: boolean.
- issues: array of strings.
- overall_pass: boolean.

overall_pass may be true only when every source claim is entailed and covered,
every rewrite claim is supported, both booleans are true, and both issue
arrays are empty. Do not place commentary outside the JSON object.
"""

VALIDATOR_B_SYSTEM_PROMPT = """\
Act as an independent reference-quality diagnostician. Judge only the supplied
normalized official_minutes paragraph and rewritten_minutes paragraph. Do not
use outside knowledge, source analysis, teacher reasoning, prior claim cards,
or another validator's result. This diagnostic never edits or repairs text.

Decompose both paragraphs into claims with exact substring evidence. Score:
factual_consistency 0--30, claim_coverage 0--25,
absence_of_unsupported_content 0--20, fomc_minutes_style 0--15,
clarity_and_coherence 0--5, and independent_rewriting 0--5.

Lexical overlap is allowed: the rewrite may share phrases, complete sentences,
or extensive wording with the official paragraph because its source analysis
may already use Minutes-style prose. Do not lower a score, add a critical
error, or fail the diagnostic merely for such overlap. The
independent_rewriting dimension instead asks whether the rewrite is a coherent,
self-contained paragraph without broken quotation or extraction artifacts;
surface novelty is not required.

Return exactly one JSON object with exactly these keys:
- official_claims: nonempty array of objects with exactly official_span,
  rewrite_evidence, verdict; verdict is exactly preserved, omitted, or
  contradicted. Use preserved, never supported, when the rewrite contains the
  official claim. rewrite_evidence is null or a nonempty array of exact,
  individually contiguous substrings from rewritten_minutes.
- rewrite_claims: nonempty array of objects with exactly rewrite_span,
  official_evidence, verdict; verdict is supported, unsupported, or
  contradicted. official_evidence is null or a nonempty array of exact,
  individually contiguous substrings from official_minutes. Never combine
  noncontiguous quotations into one string.
- scores: object containing exactly the six score keys above.
- critical_errors: array of strings.
- overall_score: integer equal to the six-score sum.
- overall_pass: boolean.

overall_pass should be true only when all claims align, critical_errors is
empty, and overall_score is at least 90. Do not place commentary outside the
JSON object.
"""

_CONTROL_OR_HEADING_RE = re.compile(
    r"```|~~~|(?:^|\n)\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s)",
    re.MULTILINE,
)
_JSONISH_RE = re.compile(r"^\s*[\[{]|[}\]]\s*$")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class SyntheticRewriteError(RuntimeError):
    """The synthetic rewrite release cannot be constructed safely."""


class ModelDriftError(SyntheticRewriteError):
    """A provider identity changed within one role-bound release."""


class ProviderRequestError(SyntheticRewriteError):
    """A provider request exhausted transport retries."""

    def __init__(self, message: str, *, attempts: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = tuple(dict(item) for item in attempts)


class ContractError(SyntheticRewriteError):
    """A provider response or local row violates a deterministic contract."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(dict.fromkeys(str(item) for item in reasons if str(item)))
        super().__init__(";".join(self.reasons))


@dataclass(frozen=True)
class ProviderConfig:
    model: str = MODEL
    base_url: str = BASE_URL
    api_key_env: str = API_KEY_ENV
    reasoning_effort: str = "high"
    timeout_seconds: float = 180.0
    max_transport_retries: int = 3

    def __post_init__(self) -> None:
        if self.model != MODEL:
            raise SyntheticRewriteError(f"provider model must be exactly {MODEL}")
        if self.base_url != BASE_URL:
            raise SyntheticRewriteError(f"provider base URL must be exactly {BASE_URL}")
        if self.api_key_env != API_KEY_ENV:
            raise SyntheticRewriteError(
                f"provider credential must be read only from {API_KEY_ENV}"
            )
        if self.reasoning_effort != "high":
            raise SyntheticRewriteError("reasoning_effort must be high")
        if self.max_transport_retries != 3:
            raise SyntheticRewriteError("transport retry count must remain 3")

    def contract(self) -> dict[str, Any]:
        return {
            "provider": "deepseek",
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "reasoning_effort": self.reasoning_effort,
            "thinking": {"type": "enabled"},
            "response_format": {"type": "json_object"},
            "temperature": None,
            "top_p": None,
            "fallback": "forbidden",
            "max_tokens": None,
            "output_limit_policy": "provider_default_no_client_side_token_cap",
            "timeout_seconds": self.timeout_seconds,
            "max_transport_retries": self.max_transport_retries,
        }

    @property
    def contract_sha256(self) -> str:
        return sha256_text(canonical_json(self.contract()))


@dataclass(frozen=True)
class ProviderResponse:
    raw_reasoning: str
    raw_content: str
    response_id: str
    returned_model: str
    system_fingerprint: str
    finish_reason: str
    created: int | None
    usage: Mapping[str, int | None]
    attempts: tuple[Mapping[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_reasoning": self.raw_reasoning,
            "raw_content": self.raw_content,
            "response_id": self.response_id,
            "returned_model": self.returned_model,
            "system_fingerprint": self.system_fingerprint,
            "finish_reason": self.finish_reason,
            "created": self.created,
            "usage": dict(self.usage),
            "attempts": [dict(item) for item in self.attempts],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProviderResponse":
        usage = value.get("usage")
        attempts = value.get("attempts", [])
        if not isinstance(usage, dict):
            raise SyntheticRewriteError("cached provider usage is invalid")
        if not isinstance(attempts, list) or not all(
            isinstance(item, dict) for item in attempts
        ):
            raise SyntheticRewriteError("cached provider attempts are invalid")
        created = value.get("created")
        return cls(
            raw_reasoning=str(value.get("raw_reasoning") or ""),
            raw_content=str(value.get("raw_content") or ""),
            response_id=str(value.get("response_id") or ""),
            returned_model=str(value.get("returned_model") or ""),
            system_fingerprint=str(value.get("system_fingerprint") or ""),
            finish_reason=str(value.get("finish_reason") or ""),
            created=None if created is None else int(created),
            usage={
                str(key): None if item is None else int(item)
                for key, item in usage.items()
            },
            attempts=tuple(dict(item) for item in attempts),
        )


class ProviderBackend(Protocol):
    def generate(
        self,
        *,
        role: str,
        config: ProviderConfig,
        system_prompt: str,
        user_prompt: str,
        environment: Mapping[str, str] | None,
    ) -> ProviderResponse: ...


@dataclass(frozen=True)
class PreparedRow:
    sample_id: str
    split: str
    source_index: int
    meeting_date: str
    source_analysis: str
    official_minutes: str
    student_prompt: str
    source_analysis_sha256: str
    official_minutes_sha256: str
    prompt_sha256: str
    source_manifest_sha256: str
    source_response_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PREPARED_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "split": self.split,
            "source_index": self.source_index,
            "meeting_date": self.meeting_date,
            "source_analysis": self.source_analysis,
            "official_minutes": self.official_minutes,
            "student_prompt": self.student_prompt,
            "source_analysis_sha256": self.source_analysis_sha256,
            "official_minutes_sha256": self.official_minutes_sha256,
            "prompt_sha256": self.prompt_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_response_sha256": self.source_response_sha256,
        }


@dataclass(frozen=True)
class Candidate:
    teacher_response_analysis: str
    rewritten_minutes: str
    sft_response: str
    attempt: str
    provider: Mapping[str, Any]
    deterministic_validation: Mapping[str, Any]


@dataclass(frozen=True)
class GenerationOutcome:
    candidate: Candidate | None
    repair_used: bool
    rejection_reasons: tuple[str, ...]
    provider: Mapping[str, Any]
    deterministic_validation: Mapping[str, Any]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_digest(values: Sequence[str]) -> str:
    return sha256_text("".join(f"{value}\n" for value in sorted(values)))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyntheticRewriteError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SyntheticRewriteError(f"{label} must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SyntheticRewriteError(f"cannot read {label}: {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise SyntheticRewriteError(f"blank {label} row: {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SyntheticRewriteError(
                f"invalid {label} row: {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise SyntheticRewriteError(
                f"non-object {label} row: {path}:{line_number}"
            )
        rows.append(value)
    return rows


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


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


def _store_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        dict(value), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != serialized:
            raise SyntheticRewriteError(f"immutable cache conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = path.read_text(encoding="utf-8")
            if existing != serialized:
                raise SyntheticRewriteError(f"immutable cache race: {path}")
        os.unlink(temporary)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _strict_json_object(raw: str) -> dict[str, Any]:
    duplicates: list[str] = []

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            str(raw), object_pairs_hook=pairs_hook, parse_constant=reject_constant
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ContractError((f"content_not_strict_json:{exc}",)) from exc
    if duplicates:
        raise ContractError(
            ("content_duplicate_keys:" + ",".join(sorted(set(duplicates))),)
        )
    if not isinstance(value, dict):
        raise ContractError(("content_json_must_be_object",))
    return value


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in _SENTENCE_RE.split(text.strip()) if item.strip()]


def _counter_dict(value: Counter[str]) -> dict[str, int]:
    return {key: int(value[key]) for key in sorted(value)}


def _normalized_prose(text: str) -> str:
    return " ".join(str(text).casefold().split())


def _is_permitted_operational_issue(value: str) -> bool:
    """Recognize legacy Validator-A complaints that the new CoT policy allows."""

    lowered = " ".join(str(value).casefold().split())
    factual_problem_terms = (
        "unsupported fact",
        "unsupported claim",
        "contradict",
        "conflict",
        "omitted claim",
        "missing claim",
        "not covered",
    )
    if any(term in lowered for term in factual_problem_terms):
        return False
    operational_terms = (
        "meta-discussion",
        "meta discussion",
        "operational",
        "prompt",
        "json",
        "answer field",
        "transport",
        "formatting",
        "word count",
        "length requirement",
        "draft",
        "rehearsal",
    )
    return any(term in lowered for term in operational_terms)


def _contains_operational_reasoning_signal(value: str) -> bool:
    lowered = " ".join(str(value).casefold().split())
    return any(
        term in lowered
        for term in (
            "prompt",
            "instruction",
            "json",
            "answer",
            "schema",
            "api",
            "transport",
            "format",
            "word count",
            "length",
            "draft",
            "rehears",
            "output",
        )
    )


def _is_lexical_overlap_only_issue(value: str) -> bool:
    lowered = " ".join(str(value).casefold().split())
    factual_problem_terms = (
        "unsupported",
        "contradict",
        "conflict",
        "omitted",
        "missing",
        "incorrect",
        "inaccurate",
    )
    if any(term in lowered for term in factual_problem_terms):
        return False
    return any(
        term in lowered
        for term in (
            "copy",
            "copied",
            "verbatim",
            "lexical overlap",
            "wording overlap",
            "too similar",
            "not independent",
        )
    )


def _source_lineage() -> dict[str, Any]:
    return {
        "source_population_target_derived": True,
        "analysis_source_lineage": "target_derived_from_official_minutes",
        "analysis_teacher_saw_official_target": True,
        "rewrite_teacher_input_fields": ["source_analysis"],
        "rewrite_teacher_saw_official_target": False,
        "rewrite_teacher_received_official_validator_feedback": False,
        "input_fidelity_validator_saw_official_target": False,
        "reference_validator_saw_official_target": True,
        "reference_validator_used_for_training_selection": False,
        "reference_audit_only": True,
        "official_reference_validator_saw_official_target": True,
        "official_reference_feedback_returned_to_rewrite_teacher": False,
        "official_reference_score_used_for_selection": False,
        "rewrite_stage_target_assisted_selection": False,
        "student_prompt_has_direct_target_field": False,
        "official_minutes_used_as_student_target": False,
        "target_is_teacher_synthetic_rewrite": True,
        "teacher_response_analysis_was_sanitized": False,
        "teacher_response_analysis_is_complete_native_cot": True,
        "teacher_response_analysis_operational_meta_allowed": True,
        "teacher_response_analysis_draft_deliberation_allowed": True,
        "source_wording_reuse_allowed": True,
        "near_copy_used_for_training_rejection": False,
        "exact_full_source_copy_used_for_training_rejection": True,
        "reference_lexical_overlap_used_for_warning": False,
        "token_length_used_for_training_rejection": False,
        "maximum_total_tokens": None,
        "training_only": True,
        "evaluation_eligible": False,
        "suitable_for_leakage_safe_evaluation": False,
        "human_review_required": False,
    }


def _prompt_contract(*, code_sha256: str, config: ProviderConfig) -> dict[str, Any]:
    return {
        "schema_version": PROMPT_CONTRACT_SCHEMA_VERSION,
        "code_sha256": code_sha256,
        "student_system_prompt": STUDENT_SYSTEM_PROMPT,
        "student_system_prompt_sha256": sha256_text(STUDENT_SYSTEM_PROMPT),
        "student_user_prompt_template": STUDENT_USER_PROMPT_TEMPLATE,
        "student_user_prompt_contract": {
            "prefix": "Rewrite the following analysis as formal FOMC Minutes prose:\n\n",
            "json_keys": ["analysis"],
            "source_field": "source_analysis",
        },
        "student_response_contract": STUDENT_RESPONSE_CONTRACT,
        "rewrite_system_prompt": REWRITE_SYSTEM_PROMPT,
        "rewrite_system_prompt_sha256": sha256_text(REWRITE_SYSTEM_PROMPT),
        "rewrite_repair_system_prompt": REWRITE_REPAIR_SYSTEM_PROMPT,
        "rewrite_repair_system_prompt_sha256": sha256_text(
            REWRITE_REPAIR_SYSTEM_PROMPT
        ),
        "validator_a_system_prompt": VALIDATOR_A_SYSTEM_PROMPT,
        "validator_a_system_prompt_sha256": sha256_text(VALIDATOR_A_SYSTEM_PROMPT),
        "validator_b_system_prompt": VALIDATOR_B_SYSTEM_PROMPT,
        "validator_b_system_prompt_sha256": sha256_text(VALIDATOR_B_SYSTEM_PROMPT),
        "provider_contract": config.contract(),
        "provider_contract_sha256": config.contract_sha256,
        "request_field_allowlists": {
            ROLE_REWRITE_PRIMARY: ["analysis"],
            ROLE_REWRITE_REPAIR: ["analysis", "repair_feedback"],
            ROLE_VALIDATOR_A_PRIMARY: [
                "source_analysis",
                "teacher_response_analysis",
                "rewritten_minutes",
            ],
            ROLE_VALIDATOR_A_REPAIR: [
                "source_analysis",
                "teacher_response_analysis",
                "rewritten_minutes",
            ],
            ROLE_VALIDATOR_B: ["official_minutes", "rewritten_minutes"],
        },
        "selection_policy": {
            "deterministic_gate_required": True,
            "validator_a_required": True,
            "validator_b_diagnostic_only": True,
            "validator_b_never_repairs": True,
            "validator_b_never_changes_training_pass": True,
            "maximum_rewrite_repairs": 1,
            "complete_native_cot_preserved": True,
            "reasoning_operational_meta_allowed": True,
            "reasoning_meta_used_for_rejection": False,
            "reasoning_sanitization": "none",
            "source_phrase_sentence_reuse_allowed": True,
            "near_copy_used_for_rejection": False,
            "exact_full_source_copy_used_for_rejection": True,
            "validator_b_lexical_overlap_warning": False,
            "total_token_limit": None,
            "total_token_used_for_rejection": False,
            "tokenizer_length_recorded": True,
            "truncation": False,
            "human_review": False,
        },
        "lineage": _source_lineage(),
    }


def prepare_source_release(
    *,
    source_root: str | Path,
    output_root: str | Path,
    enforce_pins: bool = True,
) -> tuple[dict[str, list[PreparedRow]], dict[str, Any]]:
    """Load and bind the immutable 2,083-row source release."""

    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    summary_path = source / "summary.json"
    handoff_path = source / "handoff.json"
    receipt_path = source / "merge_receipt.json"
    required = (summary_path, handoff_path, receipt_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SyntheticRewriteError(f"source release files missing: {missing}")
    observed_pins = {
        "summary_sha256": sha256_file(summary_path),
        "handoff_sha256": sha256_file(handoff_path),
        "merge_receipt_sha256": sha256_file(receipt_path),
    }
    expected_pins = {
        "summary_sha256": EXPECTED_SOURCE_SUMMARY_SHA256,
        "handoff_sha256": EXPECTED_SOURCE_HANDOFF_SHA256,
        "merge_receipt_sha256": EXPECTED_SOURCE_MERGE_RECEIPT_SHA256,
    }
    if enforce_pins and observed_pins != expected_pins:
        raise SyntheticRewriteError(
            f"source release pin mismatch: expected={expected_pins} observed={observed_pins}"
        )

    summary = _read_json(summary_path, label="source summary")
    handoff = _read_json(handoff_path, label="source handoff")
    receipt = _read_json(receipt_path, label="source merge receipt")
    if summary.get("training_ready") is not True or handoff.get("training_ready") is not True:
        raise SyntheticRewriteError("source release is not training-ready")
    if handoff.get("evaluation_eligible") is not False:
        raise SyntheticRewriteError("source release must remain evaluation-ineligible")

    prepared: dict[str, list[PreparedRow]] = {split: [] for split in SPLITS}
    all_ids: list[str] = []
    seen_ids: set[str] = set()
    meeting_splits: dict[str, str] = {}
    source_artifacts: dict[str, Any] = {}
    for split in SPLITS:
        manifest_path = source / "manifests" / f"{split}.jsonl"
        candidate_path = source / "sft_candidate" / f"{split}.jsonl"
        manifests = _read_jsonl(manifest_path, label=f"{split} source manifest")
        candidates = _read_jsonl(candidate_path, label=f"{split} source candidate")
        if len(manifests) != len(candidates):
            raise SyntheticRewriteError(
                f"source candidate/manifest count mismatch for {split}"
            )
        if enforce_pins and len(manifests) != EXPECTED_SPLIT_COUNTS[split]:
            raise SyntheticRewriteError(
                f"source split count mismatch for {split}: {len(manifests)}"
            )
        manifest_file_sha = sha256_file(manifest_path)
        candidate_file_sha = sha256_file(candidate_path)
        source_artifacts[split] = {
            "rows": len(manifests),
            "manifest": {
                "path": _display_path(manifest_path),
                "sha256": manifest_file_sha,
            },
            "sft_candidate": {
                "path": _display_path(candidate_path),
                "sha256": candidate_file_sha,
            },
        }
        for source_index, (manifest, candidate) in enumerate(
            zip(manifests, candidates, strict=True)
        ):
            if set(candidate) != {"prompt", "response"}:
                raise SyntheticRewriteError(
                    f"source candidate must have prompt/response only: {split}:{source_index}"
                )
            sample_id = str(manifest.get("sample_id") or "").strip()
            meeting_date = str(manifest.get("meeting_date") or "").strip()
            analysis = str(manifest.get("analysis") or "")
            official = str(manifest.get("official_minutes_paragraph") or "")
            if not sample_id or sample_id in seen_ids:
                raise SyntheticRewriteError(f"invalid/duplicate sample_id: {sample_id!r}")
            if manifest.get("split") != split:
                raise SyntheticRewriteError(f"source split mismatch: {sample_id}")
            if not meeting_date or not analysis or not official:
                raise SyntheticRewriteError(f"source row has empty required text: {sample_id}")
            previous_split = meeting_splits.setdefault(meeting_date, split)
            if previous_split != split:
                raise SyntheticRewriteError(
                    f"meeting split leakage: {meeting_date} in {previous_split}/{split}"
                )
            prompt = render_user_prompt(analysis)
            if candidate["prompt"] != prompt:
                raise SyntheticRewriteError(f"source prompt/analysis mismatch: {sample_id}")
            source_response = str(candidate["response"])
            hashes = {
                "analysis": sha256_text(analysis),
                "official": sha256_text(official),
                "prompt": sha256_text(prompt),
                "response": sha256_text(source_response),
            }
            declared_analysis = manifest.get("analysis_sha256")
            if declared_analysis is not None and declared_analysis != hashes["analysis"]:
                raise SyntheticRewriteError(
                    f"source analysis hash binding mismatch: {sample_id}"
                )
            declared = {
                "official": manifest.get("official_minutes_sha256"),
                "prompt": manifest.get("prompt_sha256"),
                "response": manifest.get("response_sha256"),
            }
            expected_declared = {
                "official": hashes["official"],
                "prompt": hashes["prompt"],
                "response": hashes["response"],
            }
            if declared != expected_declared:
                raise SyntheticRewriteError(
                    "source row hash binding mismatch: "
                    f"{sample_id}: {declared} != {expected_declared}"
                )
            row = PreparedRow(
                sample_id=sample_id,
                split=split,
                source_index=source_index,
                meeting_date=meeting_date,
                source_analysis=analysis,
                official_minutes=official,
                student_prompt=prompt,
                source_analysis_sha256=hashes["analysis"],
                official_minutes_sha256=hashes["official"],
                prompt_sha256=hashes["prompt"],
                source_manifest_sha256=manifest_file_sha,
                source_response_sha256=hashes["response"],
            )
            prepared[split].append(row)
            seen_ids.add(sample_id)
            all_ids.append(sample_id)

    observed_counts = {split: len(rows) for split, rows in prepared.items()}
    observed_id_digest = _id_digest(all_ids)
    if enforce_pins:
        if observed_counts != EXPECTED_SPLIT_COUNTS or len(all_ids) != EXPECTED_TOTAL:
            raise SyntheticRewriteError(
                f"source population mismatch: counts={observed_counts} total={len(all_ids)}"
            )
        if observed_id_digest != EXPECTED_SOURCE_SAMPLE_ID_SHA256:
            raise SyntheticRewriteError(
                "source sample-ID digest mismatch: "
                f"expected={EXPECTED_SOURCE_SAMPLE_ID_SHA256} observed={observed_id_digest}"
            )
        receipt_digest = receipt.get("integrity", {}).get("sample_id_sha256")
        if receipt_digest != EXPECTED_SOURCE_SAMPLE_ID_SHA256:
            raise SyntheticRewriteError("source merge receipt sample-ID pin mismatch")

    declared_artifacts = summary.get("artifacts")
    if not isinstance(declared_artifacts, dict):
        raise SyntheticRewriteError("source summary artifacts are missing")
    for split in SPLITS:
        declared_split = declared_artifacts.get(split)
        if not isinstance(declared_split, dict):
            raise SyntheticRewriteError(f"source summary artifact missing: {split}")
        observed = source_artifacts[split]
        if declared_split.get("rows") != observed["rows"]:
            raise SyntheticRewriteError(f"source artifact row count mismatch: {split}")
        for artifact_key in ("manifest", "sft_candidate"):
            descriptor = declared_split.get(artifact_key)
            if not isinstance(descriptor, dict):
                raise SyntheticRewriteError(
                    f"source artifact descriptor missing: {split}/{artifact_key}"
                )
            if descriptor.get("sha256") != observed[artifact_key]["sha256"]:
                raise SyntheticRewriteError(
                    f"source artifact SHA mismatch: {split}/{artifact_key}"
                )

    for split, rows in prepared.items():
        _write_jsonl(output / "prepared" / f"{split}.jsonl", [row.as_dict() for row in rows])
    result = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "status": "prepared",
        "root": _display_path(source),
        "source_pins_enforced": enforce_pins,
        "summary_sha256": observed_pins["summary_sha256"],
        "handoff_sha256": observed_pins["handoff_sha256"],
        "merge_receipt_sha256": observed_pins["merge_receipt_sha256"],
        "sample_id_sha256": observed_id_digest,
        "split_counts": observed_counts,
        "total_rows": len(all_ids),
        "artifacts": declared_artifacts,
        "lineage": _source_lineage(),
    }
    _write_json(output / "preparation_summary.json", result)
    return prepared, result


class OpenAICompatibleBackend:
    """Fixed DeepSeek backend; only the API key is read from the environment."""

    def __init__(self) -> None:
        self._local = threading.local()

    def _client(self, *, config: ProviderConfig, api_key: str) -> Any:
        cached = getattr(self._local, "client_record", None)
        binding = (config.base_url, config.timeout_seconds, api_key)
        if cached is not None and cached[0] == binding:
            return cached[1]
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise SyntheticRewriteError("openai package is unavailable") from exc
        client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,
        )
        self._local.client_record = (binding, client)
        return client

    def generate(
        self,
        *,
        role: str,
        config: ProviderConfig,
        system_prompt: str,
        user_prompt: str,
        environment: Mapping[str, str] | None,
    ) -> ProviderResponse:
        if role not in PROVIDER_ROLES:
            raise SyntheticRewriteError(f"invalid provider role: {role}")
        env = os.environ if environment is None else environment
        api_key = str(env.get(API_KEY_ENV) or "").strip()
        if not api_key:
            raise SyntheticRewriteError(f"missing provider credential: {API_KEY_ENV}")
        client = self._client(config=config, api_key=api_key)
        attempts: list[dict[str, Any]] = []
        last_error: Exception | None = None
        for attempt_index in range(config.max_transport_retries + 1):
            started = time.monotonic()
            started_at = _utc_now()
            try:
                request = {
                    "model": config.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "reasoning_effort": config.reasoning_effort,
                    "response_format": {"type": "json_object"},
                    "extra_body": {"thinking": {"type": "enabled"}},
                    "stream": False,
                }
                completion = client.chat.completions.create(**request)
                if not completion.choices:
                    raise SyntheticRewriteError("provider returned no choices")
                choice = completion.choices[0]
                message = choice.message
                usage = getattr(completion, "usage", None)
                usage_payload = {
                    key: (
                        None
                        if getattr(usage, key, None) is None
                        else int(getattr(usage, key))
                    )
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                }
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "started_at_utc": started_at,
                        "latency_ms": round((time.monotonic() - started) * 1000, 3),
                        "status": "success",
                    }
                )
                return ProviderResponse(
                    raw_reasoning=str(
                        getattr(message, "reasoning_content", "") or ""
                    ),
                    raw_content=str(getattr(message, "content", "") or ""),
                    response_id=str(getattr(completion, "id", "") or ""),
                    returned_model=str(getattr(completion, "model", "") or ""),
                    system_fingerprint=str(
                        getattr(completion, "system_fingerprint", "") or ""
                    ),
                    finish_reason=str(getattr(choice, "finish_reason", "") or ""),
                    created=(
                        None
                        if getattr(completion, "created", None) is None
                        else int(completion.created)
                    ),
                    usage=usage_payload,
                    attempts=tuple(attempts),
                )
            except Exception as exc:  # provider exception classes vary
                last_error = exc
                status_code = getattr(exc, "status_code", None)
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "started_at_utc": started_at,
                        "latency_ms": round((time.monotonic() - started) * 1000, 3),
                        "status": "error",
                        "status_code": (
                            None if status_code is None else int(status_code)
                        ),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc)[:500],
                    }
                )
                if status_code in {400, 401, 403, 404, 422}:
                    break
                if attempt_index >= config.max_transport_retries:
                    break
                time.sleep(min(2**attempt_index, 8))
        assert last_error is not None
        raise ProviderRequestError(
            f"provider request failed for {role}: {type(last_error).__name__}",
            attempts=attempts,
        ) from last_error


class ProviderIdentityRegistry:
    """Pin one model/fingerprint pair across every role in the release."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._identity: tuple[str, str] | None = None
        self._roles: set[str] = set()

    def bind(self, role: str, response: ProviderResponse) -> None:
        if response.returned_model != MODEL:
            raise ModelDriftError(
                f"{role} returned {response.returned_model!r}, expected {MODEL!r}"
            )
        fingerprint = response.system_fingerprint.strip()
        if not fingerprint or fingerprint == "unavailable":
            raise ModelDriftError(f"{role} provider fingerprint is unavailable")
        if not response.response_id.strip():
            raise ModelDriftError(f"{role} provider response ID is unavailable")
        identity = (response.returned_model, fingerprint)
        with self._lock:
            if self._identity is None:
                self._identity = identity
            elif self._identity != identity:
                raise ModelDriftError(
                    f"provider identity drift for {role}: {self._identity} != {identity}"
                )
            self._roles.add(role)

    def as_dict(self) -> dict[str, Any]:
        with self._lock:
            if self._identity is None:
                return {}
            model, fingerprint = self._identity
            return {
                "returned_model": model,
                "system_fingerprint": fingerprint,
                "roles_observed": sorted(self._roles),
            }

    def bind_record(self, role: str, provider: Mapping[str, Any]) -> None:
        """Restore the global identity from a materialized terminal record."""

        if not provider:
            return
        response = ProviderResponse(
            raw_reasoning="",
            raw_content="",
            response_id=str(provider.get("response_id") or "terminal-replay"),
            returned_model=str(provider.get("returned_model") or provider.get("model") or ""),
            system_fingerprint=str(provider.get("system_fingerprint") or ""),
            finish_reason=str(provider.get("finish_reason") or "stop"),
            created=None,
            usage={},
        )
        self.bind(role, response)


def _provider_record(
    response: ProviderResponse, cache_payload: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    record = {
        "response_id": response.response_id,
        "returned_model": response.returned_model,
        "system_fingerprint": response.system_fingerprint,
        "finish_reason": response.finish_reason,
        "created": response.created,
        "usage": dict(response.usage),
        "raw_reasoning_sha256": sha256_text(response.raw_reasoning),
        "raw_content_sha256": sha256_text(response.raw_content),
    }
    if cache_payload is not None:
        projection = cache_payload.get("request_projection")
        if not isinstance(projection, dict):
            raise SyntheticRewriteError("provider cache lacks request projection")
        record["request_projection"] = dict(projection)
    return record


def _request_projection(
    *, role: str, user_prompt: str, repair_trigger_source: str | None
) -> dict[str, Any]:
    try:
        payload = json.loads(user_prompt.split("\n\n", 1)[1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise SyntheticRewriteError(f"cannot project {role} request payload") from exc
    if not isinstance(payload, dict):
        raise SyntheticRewriteError(f"{role} request payload must be an object")
    semantic_by_role = {
        ROLE_REWRITE_PRIMARY: ("source_analysis",),
        ROLE_REWRITE_REPAIR: ("source_analysis", "repair_feedback"),
        ROLE_VALIDATOR_A_PRIMARY: (
            "source_analysis",
            "teacher_response_analysis",
            "rewritten_minutes",
        ),
        ROLE_VALIDATOR_A_REPAIR: (
            "source_analysis",
            "teacher_response_analysis",
            "rewritten_minutes",
        ),
        ROLE_VALIDATOR_B: ("official_minutes", "rewritten_minutes"),
    }
    wire_by_role = {
        ROLE_REWRITE_PRIMARY: ("analysis",),
        ROLE_REWRITE_REPAIR: ("analysis", "repair_feedback"),
        ROLE_VALIDATOR_A_PRIMARY: semantic_by_role[ROLE_VALIDATOR_A_PRIMARY],
        ROLE_VALIDATOR_A_REPAIR: semantic_by_role[ROLE_VALIDATOR_A_REPAIR],
        ROLE_VALIDATOR_B: semantic_by_role[ROLE_VALIDATOR_B],
    }
    wire_fields = wire_by_role[role]
    if set(payload) != set(wire_fields):
        raise SyntheticRewriteError(
            f"{role} request fields drift: {sorted(payload)} != {sorted(wire_fields)}"
        )
    semantic_values = dict(payload)
    if "analysis" in semantic_values:
        semantic_values["source_analysis"] = semantic_values.pop("analysis")
    semantic_fields = semantic_by_role[role]
    if set(semantic_values) != set(semantic_fields):
        raise SyntheticRewriteError(f"{role} semantic request projection drift")
    if role == ROLE_REWRITE_REPAIR:
        if repair_trigger_source not in {"deterministic_validation", "validator_a"}:
            raise SyntheticRewriteError("repair trigger must be deterministic or Validator A")
    elif repair_trigger_source is not None:
        raise SyntheticRewriteError(f"non-repair role has repair trigger: {role}")
    return {
        "role": role,
        "allowed_fields": list(semantic_fields),
        "wire_fields": list(wire_fields),
        "field_value_sha256": {
            key: sha256_text(canonical_json(semantic_values[key]))
            for key in semantic_fields
        },
        "canonical_payload_sha256": sha256_text(canonical_json(payload)),
        "official_target_policy": (
            "required_exact_normalized_official_minutes"
            if role == ROLE_VALIDATOR_B
            else "forbidden"
        ),
        "repair_trigger_source": repair_trigger_source,
        "reference_diagnostic_can_trigger_repair": False,
        "api_key_env": API_KEY_ENV,
        "plaintext_credential_persisted": False,
    }


def _cache_path(output_root: Path, role: str, row: PreparedRow) -> Path:
    return output_root / "cache" / role / f"{sha256_text(row.sample_id)}.json"


def _cache_binding(
    *,
    role: str,
    row: PreparedRow,
    system_prompt: str,
    user_prompt: str,
    config: ProviderConfig,
    code_sha256: str,
    request_projection: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "role": role,
        "sample_id": row.sample_id,
        "split": row.split,
        "source_analysis_sha256": row.source_analysis_sha256,
        "official_minutes_sha256": (
            row.official_minutes_sha256 if role == ROLE_VALIDATOR_B else None
        ),
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(user_prompt),
        "provider_contract_sha256": config.contract_sha256,
        "code_sha256": code_sha256,
        "request_projection_sha256": sha256_text(
            canonical_json(dict(request_projection))
        ),
    }


def _load_or_call(
    *,
    role: str,
    row: PreparedRow,
    output_root: Path,
    system_prompt: str,
    user_prompt: str,
    config: ProviderConfig,
    backend: ProviderBackend,
    identity_registry: ProviderIdentityRegistry,
    environment: Mapping[str, str] | None,
    code_sha256: str,
    repair_trigger_source: str | None = None,
) -> tuple[dict[str, Any], ProviderResponse]:
    request_projection = _request_projection(
        role=role,
        user_prompt=user_prompt,
        repair_trigger_source=repair_trigger_source,
    )
    binding = _cache_binding(
        role=role,
        row=row,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=config,
        code_sha256=code_sha256,
        request_projection=request_projection,
    )
    binding_sha = sha256_text(canonical_json(binding))
    path = _cache_path(output_root, role, row)
    if path.is_file():
        payload = _read_json(path, label=f"{role} cache")
        if payload.get("binding") != binding or payload.get("binding_sha256") != binding_sha:
            raise SyntheticRewriteError(f"cache binding mismatch: {path}")
        if payload.get("request_projection") != request_projection:
            raise SyntheticRewriteError(f"cache request projection mismatch: {path}")
        provider = payload.get("provider_response")
        if not isinstance(provider, dict):
            raise SyntheticRewriteError(f"cache provider response invalid: {path}")
        response = ProviderResponse.from_dict(provider)
        if payload.get("raw_reasoning_sha256") != sha256_text(response.raw_reasoning):
            raise SyntheticRewriteError(f"cache reasoning SHA mismatch: {path}")
        if payload.get("raw_content_sha256") != sha256_text(response.raw_content):
            raise SyntheticRewriteError(f"cache content SHA mismatch: {path}")
        identity_registry.bind(role, response)
        return payload, response

    response = backend.generate(
        role=role,
        config=config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        environment=environment,
    )
    identity_registry.bind(role, response)
    payload = {
        "binding": binding,
        "binding_sha256": binding_sha,
        "provider_response": response.as_dict(),
        "request_projection": request_projection,
        "raw_reasoning_sha256": sha256_text(response.raw_reasoning),
        "raw_content_sha256": sha256_text(response.raw_content),
    }
    _store_immutable_json(path, payload)
    return payload, response


def _teacher_user_prompt(row: PreparedRow) -> str:
    rendered = render_user_prompt(row.source_analysis)
    payload = json.loads(rendered.split("\n\n", 1)[1])
    if set(payload) != {"analysis"} or payload["analysis"] != row.source_analysis:
        raise SyntheticRewriteError("rewrite prompt analysis-only boundary failed")
    if row.official_minutes in rendered:
        raise SyntheticRewriteError("official Minutes leaked into rewrite prompt")
    return rendered


def _repair_user_prompt(
    row: PreparedRow,
    reasons: Sequence[str],
    *,
    validator_a_result: Mapping[str, Any] | None = None,
) -> str:
    evidence: dict[str, Any] | None = None
    if validator_a_result is not None:
        # Only A's analysis/rewrite/reasoning evidence is propagated.  The A
        # schema has no official-target field, and B is called only after the
        # repair decision is terminal.
        evidence = {
            "source_claims": validator_a_result.get("source_claims", []),
            "rewrite_claims": validator_a_result.get("rewrite_claims", []),
            "reasoning_issues": validator_a_result.get("reasoning_issues", []),
            "issues": validator_a_result.get("issues", []),
        }
    payload = {
        "analysis": row.source_analysis,
        "repair_feedback": {
            # Complete operational deliberation is part of the native CoT, so
            # exact deterministic and Validator-A reasons may be supplied.
            "reason_codes": [str(item) for item in reasons],
            "validator_a_exact_evidence": evidence,
        },
    }
    rendered = "Source record:\n\n" + canonical_json(payload)
    if row.official_minutes in rendered:
        raise SyntheticRewriteError("official Minutes leaked into repair prompt")
    return rendered


def _validator_a_user_prompt(row: PreparedRow, candidate: Candidate) -> str:
    payload = {
        "source_analysis": row.source_analysis,
        "teacher_response_analysis": candidate.teacher_response_analysis,
        "rewritten_minutes": candidate.rewritten_minutes,
    }
    rendered = "Verify this analysis-grounded rewrite:\n\n" + canonical_json(payload)
    if row.official_minutes in rendered and row.official_minutes not in {
        row.source_analysis,
        candidate.teacher_response_analysis,
        candidate.rewritten_minutes,
    }:
        raise SyntheticRewriteError("official Minutes leaked into Validator A prompt")
    return rendered


def _validator_b_user_prompt(row: PreparedRow, candidate: Candidate) -> str:
    payload = {
        "official_minutes": row.official_minutes,
        "rewritten_minutes": candidate.rewritten_minutes,
    }
    return "Score this official-reference rewrite:\n\n" + canonical_json(payload)


def _strict_answer(response: ProviderResponse) -> str:
    payload = _strict_json_object(response.raw_content)
    if set(payload) != {"answer"}:
        raise ContractError(("rewrite_content_keys_must_equal_answer",))
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer:
        raise ContractError(("rewritten_minutes_must_be_nonempty_text",))
    return answer


def _candidate_from_response(
    row: PreparedRow,
    response: ProviderResponse,
    *,
    attempt: str,
    tokenizer: Any,
    provider_record: Mapping[str, Any] | None = None,
) -> Candidate:
    reasons: list[str] = []
    if response.finish_reason != "stop":
        reasons.append(f"finish_reason_not_stop:{response.finish_reason}")
    reasoning = response.raw_reasoning
    if not reasoning.strip():
        reasons.append("empty_teacher_response_analysis")
    try:
        rewritten = _strict_answer(response)
    except ContractError as exc:
        reasons.extend(exc.reasons)
        rewritten = ""
    if rewritten and rewritten != rewritten.strip():
        reasons.append("rewritten_minutes_outer_whitespace")

    for field_name, text in (
        ("teacher_response_analysis", reasoning),
        ("rewritten_minutes", rewritten),
    ):
        markers = [marker for marker in CONTROL_MARKERS if marker in text]
        if markers:
            reasons.append(f"{field_name}_control_markers:{','.join(markers)}")
    meta = sorted(_reasoning_meta_categories(reasoning)) if reasoning else []
    reasoning_words = len(reasoning.split())
    if reasoning and reasoning_words < MIN_REASONING_WORDS:
        reasons.append(f"teacher_response_analysis_words:{reasoning_words}")
    rewrite_words = len(rewritten.split())
    if rewritten and not MIN_REWRITE_WORDS <= rewrite_words <= MAX_REWRITE_WORDS:
        reasons.append(f"rewritten_minutes_words:{rewrite_words}")
    if rewritten and ("\n" in rewritten or "\r" in rewritten):
        reasons.append("rewritten_minutes_not_single_paragraph")
    if rewritten and _CONTROL_OR_HEADING_RE.search(rewritten):
        reasons.append("rewritten_minutes_heading_list_or_markdown")
    if rewritten and _JSONISH_RE.search(rewritten):
        reasons.append("rewritten_minutes_jsonish")
    if rewritten and _normalized_prose(rewritten) == _normalized_prose(
        row.source_analysis
    ):
        reasons.append("rewritten_minutes_exactly_copies_source_analysis")

    source_numbers = _numeric_values(row.source_analysis)
    rewrite_numbers = _numeric_values(rewritten)
    source_dates = _date_values(row.source_analysis)
    rewrite_dates = _date_values(rewritten)
    source_attributions = _attribution_categories(row.source_analysis)
    rewrite_attributions = _attribution_categories(rewritten)
    if rewrite_numbers != source_numbers:
        reasons.append("numeric_multiset_mismatch")
    if rewrite_dates != source_dates:
        reasons.append("date_set_mismatch")
    if rewrite_attributions != source_attributions:
        reasons.append("attribution_set_mismatch")

    sft_response = (
        f"{reasoning}\n</think>\n{rewritten}" if reasoning and rewritten else ""
    )
    if sft_response:
        if sft_response.count("</think>") != 1:
            reasons.append("response_boundary_count_not_one")
        if "<think>" in sft_response or "<answer>" in sft_response:
            reasons.append("response_contains_forbidden_open_or_answer_tag")
    prompt_tokens = completion_tokens = total_tokens = 0
    if sft_response:
        try:
            prompt_tokens, completion_tokens, total_tokens = _count_sft_row(
                {"prompt": row.student_prompt, "response": sft_response},
                tokenizer=tokenizer,
                config={
                    "system_prompt": STUDENT_SYSTEM_PROMPT,
                    "dataset_prompt_column": "prompt",
                    "chat_template_kwargs": {},
                },
            )
        except Exception as exc:
            reasons.append(f"tokenizer_replay:{type(exc).__name__}:{exc}")
    diagnostics = {
        "machine_pass": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "diagnostics": {
            "reasoning_words": reasoning_words,
            "rewritten_minutes_words": rewrite_words,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "source_numbers": _counter_dict(source_numbers),
            "rewrite_numbers": _counter_dict(rewrite_numbers),
            "source_dates": sorted(source_dates),
            "rewrite_dates": sorted(rewrite_dates),
            "source_attributions": sorted(source_attributions),
            "rewrite_attributions": sorted(rewrite_attributions),
            "raw_reasoning_preserved": True,
            "complete_native_cot_preserved": True,
            "reasoning_meta_categories": meta,
            "reasoning_meta_used_for_rejection": False,
            "reasoning_sanitization": "none",
            "source_sentence_reuse_count": sum(
                1
                for sentence in _sentences(row.source_analysis)
                if _normalized_prose(sentence)
                and _normalized_prose(sentence) in _normalized_prose(rewritten)
            ),
            "near_copy_used_for_rejection": False,
            "exact_full_source_copy_used_for_rejection": True,
            "total_token_limit": None,
            "total_token_used_for_rejection": False,
            "tokenizer_length_recorded": True,
            "truncation": False,
        },
    }
    if reasons:
        raise ContractError(reasons)
    return Candidate(
        teacher_response_analysis=reasoning,
        rewritten_minutes=rewritten,
        sft_response=sft_response,
        attempt=attempt,
        provider=(
            dict(provider_record)
            if provider_record is not None
            else _provider_record(response)
        ),
        deterministic_validation=diagnostics,
    )


def _candidate_failure_diagnostics(
    row: PreparedRow, response: ProviderResponse, *, tokenizer: Any
) -> dict[str, Any]:
    try:
        _candidate_from_response(row, response, attempt="diagnostic", tokenizer=tokenizer)
    except ContractError as exc:
        return {"machine_pass": False, "reasons": list(exc.reasons), "diagnostics": {}}
    return {"machine_pass": True, "reasons": [], "diagnostics": {}}


def _claim_evidence(
    value: Any,
    *,
    field: str,
    text: str,
    required: bool,
    reasons: list[str],
) -> str | None:
    if value is None:
        if required:
            reasons.append(f"{field}_required")
        return None
    if not isinstance(value, str) or not value:
        reasons.append(f"{field}_must_be_nonempty_or_null")
        return None
    if value not in text:
        reasons.append(f"{field}_not_exact")
    return value


def _claim_evidence_parts(
    value: Any,
    *,
    field: str,
    text: str,
    required: bool,
    reasons: list[str],
) -> list[str] | None:
    """Validate one or more individually exact evidence spans.

    Validator B occasionally joins two exact, noncontiguous quotations with a
    semicolon. That legacy representation is split deterministically; no fuzzy
    matching or text repair is performed.
    """

    if value is None:
        if required:
            reasons.append(f"{field}_required")
        return None
    if isinstance(value, str):
        values = [value]
        if value not in text and ";" in value:
            values = [part.strip() for part in value.split(";") if part.strip()]
    elif isinstance(value, list) and value:
        values = value
    else:
        reasons.append(f"{field}_must_be_nonempty_array_string_or_null")
        return None
    if not values or not all(isinstance(item, str) and item for item in values):
        reasons.append(f"{field}_must_be_nonempty_array_string_or_null")
        return None
    for index, item in enumerate(values, start=1):
        if item not in text:
            reasons.append(f"{field}:{index}_not_exact")
    return list(values)


def _validate_validator_a(
    row: PreparedRow, candidate: Candidate, response: ProviderResponse
) -> tuple[dict[str, Any], bool, list[str]]:
    reasons: list[str] = []
    if response.finish_reason != "stop":
        reasons.append(f"validator_a_finish_reason:{response.finish_reason}")
    payload = _strict_json_object(response.raw_content)
    expected = {
        "source_claims",
        "rewrite_claims",
        "reasoning_issues",
        "bidirectional_entailment",
        "reasoning_compatible",
        "issues",
        "overall_pass",
    }
    if set(payload) != expected:
        reasons.append("validator_a_content_keys")
    source_claims = payload.get("source_claims")
    rewrite_claims = payload.get("rewrite_claims")
    reasoning_issues = payload.get("reasoning_issues")
    issues = payload.get("issues")
    if not isinstance(source_claims, list) or not source_claims:
        reasons.append("validator_a_source_claims_nonempty_array")
        source_claims = []
    if not isinstance(rewrite_claims, list) or not rewrite_claims:
        reasons.append("validator_a_rewrite_claims_nonempty_array")
        rewrite_claims = []
    if not isinstance(reasoning_issues, list):
        reasons.append("validator_a_reasoning_issues_array")
        reasoning_issues = []
    if not isinstance(issues, list) or not all(isinstance(item, str) for item in issues):
        reasons.append("validator_a_issues_string_array")
        issues = ["invalid issues"]
    reported_issues = list(issues)
    ignored_operational_issues = [
        item for item in reported_issues if _is_permitted_operational_issue(item)
    ]
    ignored_lexical_overlap_issues = [
        item for item in reported_issues if _is_lexical_overlap_only_issue(item)
    ]
    issues = [
        item
        for item in reported_issues
        if not _is_permitted_operational_issue(item)
        and not _is_lexical_overlap_only_issue(item)
    ]

    booleans: dict[str, bool] = {}
    for key in ("bidirectional_entailment", "reasoning_compatible", "overall_pass"):
        value = payload.get(key)
        if not isinstance(value, bool):
            reasons.append(f"validator_a_{key}_boolean")
            value = False
        booleans[key] = value

    normalized_source: list[dict[str, Any]] = []
    source_spans: list[str] = []
    source_verdicts: list[str] = []
    reasoning_verdicts: list[str] = []
    for index, item in enumerate(source_claims, start=1):
        keys = {
            "source_span",
            "reasoning_evidence",
            "rewrite_evidence",
            "rewrite_verdict",
            "reasoning_verdict",
        }
        if not isinstance(item, dict) or set(item) != keys:
            reasons.append(f"validator_a_source_claim_schema:{index}")
            continue
        source_span = _claim_evidence(
            item.get("source_span"),
            field=f"validator_a_source_span:{index}",
            text=row.source_analysis,
            required=True,
            reasons=reasons,
        )
        rewrite_verdict = str(item.get("rewrite_verdict") or "")
        reasoning_verdict = str(item.get("reasoning_verdict") or "")
        if rewrite_verdict not in {"entailed", "omitted", "contradicted"}:
            reasons.append(f"validator_a_rewrite_verdict:{index}")
        if reasoning_verdict not in {"covered", "not_covered", "conflicted"}:
            reasons.append(f"validator_a_reasoning_verdict:{index}")
        rewrite_evidence = _claim_evidence(
            item.get("rewrite_evidence"),
            field=f"validator_a_rewrite_evidence:{index}",
            text=candidate.rewritten_minutes,
            required=rewrite_verdict in {"entailed", "contradicted"},
            reasons=reasons,
        )
        reasoning_evidence = _claim_evidence(
            item.get("reasoning_evidence"),
            field=f"validator_a_reasoning_evidence:{index}",
            text=candidate.teacher_response_analysis,
            required=reasoning_verdict in {"covered", "conflicted"},
            reasons=reasons,
        )
        if source_span is not None:
            source_spans.append(source_span)
        source_verdicts.append(rewrite_verdict)
        reasoning_verdicts.append(reasoning_verdict)
        normalized_source.append(
            {
                "source_span": source_span,
                "reasoning_evidence": reasoning_evidence,
                "rewrite_evidence": rewrite_evidence,
                "rewrite_verdict": rewrite_verdict,
                "reasoning_verdict": reasoning_verdict,
            }
        )
    for index, sentence in enumerate(_sentences(row.source_analysis), start=1):
        if not any(span in sentence or sentence in span for span in source_spans):
            reasons.append(f"validator_a_source_sentence_uncovered:{index}")

    normalized_rewrite: list[dict[str, Any]] = []
    rewrite_spans: list[str] = []
    rewrite_verdicts: list[str] = []
    for index, item in enumerate(rewrite_claims, start=1):
        if not isinstance(item, dict) or set(item) != {
            "rewrite_span",
            "source_evidence",
            "verdict",
        }:
            reasons.append(f"validator_a_rewrite_claim_schema:{index}")
            continue
        rewrite_span = _claim_evidence(
            item.get("rewrite_span"),
            field=f"validator_a_rewrite_span:{index}",
            text=candidate.rewritten_minutes,
            required=True,
            reasons=reasons,
        )
        verdict = str(item.get("verdict") or "")
        if verdict not in {"supported", "unsupported", "contradicted"}:
            reasons.append(f"validator_a_claim_verdict:{index}")
        source_evidence = _claim_evidence(
            item.get("source_evidence"),
            field=f"validator_a_source_evidence:{index}",
            text=row.source_analysis,
            required=verdict in {"supported", "contradicted"},
            reasons=reasons,
        )
        if rewrite_span is not None:
            rewrite_spans.append(rewrite_span)
        rewrite_verdicts.append(verdict)
        normalized_rewrite.append(
            {
                "rewrite_span": rewrite_span,
                "source_evidence": source_evidence,
                "verdict": verdict,
            }
        )
    for index, sentence in enumerate(_sentences(candidate.rewritten_minutes), start=1):
        if not any(span in sentence or sentence in span for span in rewrite_spans):
            reasons.append(f"validator_a_rewrite_sentence_uncovered:{index}")

    normalized_reasoning_issues: list[dict[str, str]] = []
    factual_issue_types = {
        "unsupported_claim",
        "source_conflict",
        "rewrite_conflict",
        "missing_claim_coverage",
    }
    permitted_legacy_issue_types = {"meta_discussion", "duplicated_draft"}
    ignored_operational_reasoning_issues: list[dict[str, str]] = []
    for index, item in enumerate(reasoning_issues, start=1):
        if not isinstance(item, dict) or set(item) != {"reasoning_span", "issue_type"}:
            reasons.append(f"validator_a_reasoning_issue_schema:{index}")
            continue
        span = _claim_evidence(
            item.get("reasoning_span"),
            field=f"validator_a_reasoning_issue_span:{index}",
            text=candidate.teacher_response_analysis,
            required=True,
            reasons=reasons,
        )
        issue_type = str(item.get("issue_type") or "")
        normalized_issue = {"reasoning_span": span or "", "issue_type": issue_type}
        if issue_type in permitted_legacy_issue_types or (
            span is not None and _contains_operational_reasoning_signal(span)
        ):
            ignored_operational_reasoning_issues.append(normalized_issue)
            continue
        if issue_type not in factual_issue_types:
            reasons.append(f"validator_a_reasoning_issue_type:{index}")
        normalized_reasoning_issues.append(normalized_issue)

    nonblocking_policy_override = bool(
        ignored_operational_issues
        or ignored_lexical_overlap_issues
        or ignored_operational_reasoning_issues
    )
    reasoning_compatible_for_gate = booleans["reasoning_compatible"] or (
        nonblocking_policy_override
        and not normalized_reasoning_issues
        and all(value == "covered" for value in reasoning_verdicts)
    )
    overall_pass_for_gate = booleans["overall_pass"] or (
        nonblocking_policy_override
        and booleans["bidirectional_entailment"]
        and reasoning_compatible_for_gate
        and not normalized_reasoning_issues
        and not issues
    )

    machine_pass = (
        not reasons
        and bool(source_verdicts)
        and all(value == "entailed" for value in source_verdicts)
        and all(value == "covered" for value in reasoning_verdicts)
        and bool(rewrite_verdicts)
        and all(value == "supported" for value in rewrite_verdicts)
        and not normalized_reasoning_issues
        and not issues
        and booleans["bidirectional_entailment"]
        and reasoning_compatible_for_gate
        and overall_pass_for_gate
    )
    if not machine_pass and not reasons:
        if any(value != "entailed" for value in source_verdicts):
            reasons.append("validator_a_source_claim_not_entailed")
        if any(value != "covered" for value in reasoning_verdicts):
            reasons.append("validator_a_reasoning_claim_not_covered")
        if any(value != "supported" for value in rewrite_verdicts):
            reasons.append("validator_a_rewrite_claim_not_supported")
        if normalized_reasoning_issues:
            reasons.append("validator_a_reasoning_issues")
        if issues:
            reasons.append("validator_a_reported_issues")
        if not booleans["bidirectional_entailment"]:
            reasons.append("validator_a_not_bidirectional")
        if not reasoning_compatible_for_gate:
            reasons.append("validator_a_reasoning_incompatible")
        if not overall_pass_for_gate:
            reasons.append("validator_a_overall_fail")
    normalized = {
        "source_claims": normalized_source,
        "rewrite_claims": normalized_rewrite,
        "reasoning_issues": normalized_reasoning_issues,
        "bidirectional_entailment": booleans["bidirectional_entailment"],
        "reasoning_compatible": booleans["reasoning_compatible"],
        "issues": list(issues),
        "overall_pass": booleans["overall_pass"],
        "reasoning_compatible_for_gate": reasoning_compatible_for_gate,
        "overall_pass_for_gate": overall_pass_for_gate,
        "reported_issues": reported_issues,
        "ignored_operational_issues": ignored_operational_issues,
        "ignored_lexical_overlap_issues": ignored_lexical_overlap_issues,
        "ignored_operational_reasoning_issues": (
            ignored_operational_reasoning_issues
        ),
        "complete_native_cot_preserved": True,
        "reasoning_meta_used_for_rejection": False,
        "machine_pass": machine_pass,
    }
    return normalized, machine_pass, list(dict.fromkeys(reasons))


_REFERENCE_SCORE_MAXIMA = {
    "factual_consistency": 30,
    "claim_coverage": 25,
    "absence_of_unsupported_content": 20,
    "fomc_minutes_style": 15,
    "clarity_and_coherence": 5,
    "independent_rewriting": 5,
}


def _validate_validator_b(
    row: PreparedRow, candidate: Candidate, response: ProviderResponse
) -> tuple[dict[str, Any], bool, bool, list[str]]:
    reasons: list[str] = []
    if response.finish_reason != "stop":
        reasons.append(f"validator_b_finish_reason:{response.finish_reason}")
    payload = _strict_json_object(response.raw_content)
    expected = {
        "official_claims",
        "rewrite_claims",
        "scores",
        "critical_errors",
        "overall_score",
        "overall_pass",
    }
    if set(payload) != expected:
        reasons.append("validator_b_content_keys")
    official_claims = payload.get("official_claims")
    rewrite_claims = payload.get("rewrite_claims")
    critical_errors = payload.get("critical_errors")
    if not isinstance(official_claims, list) or not official_claims:
        reasons.append("validator_b_official_claims_nonempty_array")
        official_claims = []
    if not isinstance(rewrite_claims, list) or not rewrite_claims:
        reasons.append("validator_b_rewrite_claims_nonempty_array")
        rewrite_claims = []
    if not isinstance(critical_errors, list) or not all(
        isinstance(item, str) for item in critical_errors
    ):
        reasons.append("validator_b_critical_errors_string_array")
        critical_errors = ["invalid critical_errors"]
    reported_critical_errors = list(critical_errors)
    ignored_lexical_overlap_errors = [
        item for item in reported_critical_errors if _is_lexical_overlap_only_issue(item)
    ]
    critical_errors = [
        item
        for item in reported_critical_errors
        if not _is_lexical_overlap_only_issue(item)
    ]
    scores = payload.get("scores")
    normalized_scores: dict[str, int] = {}
    if not isinstance(scores, dict) or set(scores) != set(_REFERENCE_SCORE_MAXIMA):
        reasons.append("validator_b_score_keys")
        scores = {}
    for key, maximum in _REFERENCE_SCORE_MAXIMA.items():
        value = scores.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= maximum:
            reasons.append(f"validator_b_score_range:{key}")
            value = 0
        normalized_scores[key] = int(value)
    overall_score = payload.get("overall_score")
    if not isinstance(overall_score, int) or isinstance(overall_score, bool):
        reasons.append("validator_b_overall_score_integer")
        overall_score = 0
    if overall_score != sum(normalized_scores.values()):
        reasons.append("validator_b_score_sum_mismatch")
    overall_pass = payload.get("overall_pass")
    if not isinstance(overall_pass, bool):
        reasons.append("validator_b_overall_pass_boolean")
        overall_pass = False

    normalized_official: list[dict[str, Any]] = []
    official_spans: list[str] = []
    official_verdicts: list[str] = []
    for index, item in enumerate(official_claims, start=1):
        if not isinstance(item, dict) or set(item) != {
            "official_span",
            "rewrite_evidence",
            "verdict",
        }:
            reasons.append(f"validator_b_official_claim_schema:{index}")
            continue
        official_span = _claim_evidence(
            item.get("official_span"),
            field=f"validator_b_official_span:{index}",
            text=row.official_minutes,
            required=True,
            reasons=reasons,
        )
        reported_verdict = str(item.get("verdict") or "")
        # Some otherwise schema-correct DeepSeek reports use the semantically
        # equivalent word "supported" for this direction. Normalize that
        # narrow alias deterministically; every other enum drift remains fatal.
        verdict = "preserved" if reported_verdict == "supported" else reported_verdict
        if verdict not in {"preserved", "omitted", "contradicted"}:
            reasons.append(f"validator_b_official_verdict:{index}")
        evidence = _claim_evidence_parts(
            item.get("rewrite_evidence"),
            field=f"validator_b_rewrite_evidence:{index}",
            text=candidate.rewritten_minutes,
            required=verdict in {"preserved", "contradicted"},
            reasons=reasons,
        )
        if official_span is not None:
            official_spans.append(official_span)
        official_verdicts.append(verdict)
        normalized_official.append(
            {
                "official_span": official_span,
                "rewrite_evidence": evidence,
                "verdict": verdict,
                "reported_verdict": reported_verdict,
            }
        )
    for index, sentence in enumerate(_sentences(row.official_minutes), start=1):
        if not any(span in sentence or sentence in span for span in official_spans):
            reasons.append(f"validator_b_official_sentence_uncovered:{index}")

    normalized_rewrite: list[dict[str, Any]] = []
    rewrite_spans: list[str] = []
    rewrite_verdicts: list[str] = []
    for index, item in enumerate(rewrite_claims, start=1):
        if not isinstance(item, dict) or set(item) != {
            "rewrite_span",
            "official_evidence",
            "verdict",
        }:
            reasons.append(f"validator_b_rewrite_claim_schema:{index}")
            continue
        rewrite_span = _claim_evidence(
            item.get("rewrite_span"),
            field=f"validator_b_rewrite_span:{index}",
            text=candidate.rewritten_minutes,
            required=True,
            reasons=reasons,
        )
        verdict = str(item.get("verdict") or "")
        if verdict not in {"supported", "unsupported", "contradicted"}:
            reasons.append(f"validator_b_rewrite_verdict:{index}")
        evidence = _claim_evidence_parts(
            item.get("official_evidence"),
            field=f"validator_b_official_evidence:{index}",
            text=row.official_minutes,
            required=verdict in {"supported", "contradicted"},
            reasons=reasons,
        )
        if rewrite_span is not None:
            rewrite_spans.append(rewrite_span)
        rewrite_verdicts.append(verdict)
        normalized_rewrite.append(
            {
                "rewrite_span": rewrite_span,
                "official_evidence": evidence,
                "verdict": verdict,
            }
        )
    for index, sentence in enumerate(_sentences(candidate.rewritten_minutes), start=1):
        if not any(span in sentence or sentence in span for span in rewrite_spans):
            reasons.append(f"validator_b_rewrite_sentence_uncovered:{index}")

    contract_reasons = list(dict.fromkeys(reasons))
    report_complete = not contract_reasons
    diagnostic_pass = (
        report_complete
        and bool(official_verdicts)
        and all(value == "preserved" for value in official_verdicts)
        and bool(rewrite_verdicts)
        and all(value == "supported" for value in rewrite_verdicts)
        and not critical_errors
        and overall_score >= 90
    )
    diagnostic_reasons: list[str] = []
    if report_complete and not diagnostic_pass:
        if any(value != "preserved" for value in official_verdicts):
            diagnostic_reasons.append("reference_official_claim_not_preserved")
        if any(value != "supported" for value in rewrite_verdicts):
            diagnostic_reasons.append("reference_rewrite_claim_not_supported")
        if critical_errors:
            diagnostic_reasons.append("reference_critical_errors")
        if overall_score < 90:
            diagnostic_reasons.append("reference_score_below_90")
    normalized_exact_copy = (
        _normalized_prose(row.official_minutes)
        == _normalized_prose(candidate.rewritten_minutes)
    )
    normalized = {
        "official_claims": normalized_official,
        "rewrite_claims": normalized_rewrite,
        "scores": normalized_scores,
        "critical_errors": list(critical_errors),
        "reported_critical_errors": reported_critical_errors,
        "ignored_lexical_overlap_errors": ignored_lexical_overlap_errors,
        "overall_score": overall_score,
        "overall_pass": overall_pass,
        "diagnostic_pass": diagnostic_pass,
        "normalized_exact_copy": normalized_exact_copy,
        "lexical_overlap_used_for_warning": False,
    }
    return (
        normalized,
        report_complete,
        diagnostic_pass,
        list(dict.fromkeys(contract_reasons + diagnostic_reasons)),
    )


def _acquire_generation(
    row: PreparedRow,
    *,
    output_root: Path,
    tokenizer: Any,
    backend: ProviderBackend,
    identity_registry: ProviderIdentityRegistry,
    environment: Mapping[str, str] | None,
    config: ProviderConfig,
    code_sha256: str,
) -> GenerationOutcome:
    primary_prompt = _teacher_user_prompt(row)
    primary_cache, primary_response = _load_or_call(
        role=ROLE_REWRITE_PRIMARY,
        row=row,
        output_root=output_root,
        system_prompt=REWRITE_SYSTEM_PROMPT,
        user_prompt=primary_prompt,
        config=config,
        backend=backend,
        identity_registry=identity_registry,
        environment=environment,
        code_sha256=code_sha256,
    )
    try:
        candidate = _candidate_from_response(
            row,
            primary_response,
            attempt="primary",
            tokenizer=tokenizer,
            provider_record=_provider_record(primary_response, primary_cache),
        )
        return GenerationOutcome(
            candidate=candidate,
            repair_used=False,
            rejection_reasons=(),
            provider=candidate.provider,
            deterministic_validation=candidate.deterministic_validation,
        )
    except ContractError as primary_error:
        repair_prompt = _repair_user_prompt(row, primary_error.reasons)
        repair_cache, repair_response = _load_or_call(
            role=ROLE_REWRITE_REPAIR,
            row=row,
            output_root=output_root,
            system_prompt=REWRITE_REPAIR_SYSTEM_PROMPT,
            user_prompt=repair_prompt,
            config=config,
            backend=backend,
            identity_registry=identity_registry,
            environment=environment,
            code_sha256=code_sha256,
            repair_trigger_source="deterministic_validation",
        )
        try:
            candidate = _candidate_from_response(
                row,
                repair_response,
                attempt="repair",
                tokenizer=tokenizer,
                provider_record=_provider_record(repair_response, repair_cache),
            )
            return GenerationOutcome(
                candidate=candidate,
                repair_used=True,
                rejection_reasons=(),
                provider=candidate.provider,
                deterministic_validation=candidate.deterministic_validation,
            )
        except ContractError as repair_error:
            return GenerationOutcome(
                candidate=None,
                repair_used=True,
                rejection_reasons=repair_error.reasons,
                provider=_provider_record(repair_response, repair_cache),
                deterministic_validation={
                    "machine_pass": False,
                    "reasons": list(repair_error.reasons),
                    "diagnostics": {},
                },
            )


def _acquire_repair_after_validator_a(
    row: PreparedRow,
    *,
    reasons: Sequence[str],
    validator_a_result: Mapping[str, Any],
    output_root: Path,
    tokenizer: Any,
    backend: ProviderBackend,
    identity_registry: ProviderIdentityRegistry,
    environment: Mapping[str, str] | None,
    config: ProviderConfig,
    code_sha256: str,
) -> GenerationOutcome:
    prompt = _repair_user_prompt(
        row, reasons, validator_a_result=validator_a_result
    )
    cache, response = _load_or_call(
        role=ROLE_REWRITE_REPAIR,
        row=row,
        output_root=output_root,
        system_prompt=REWRITE_REPAIR_SYSTEM_PROMPT,
        user_prompt=prompt,
        config=config,
        backend=backend,
        identity_registry=identity_registry,
        environment=environment,
        code_sha256=code_sha256,
        repair_trigger_source="validator_a",
    )
    try:
        candidate = _candidate_from_response(
            row,
            response,
            attempt="repair",
            tokenizer=tokenizer,
            provider_record=_provider_record(response, cache),
        )
        return GenerationOutcome(
            candidate=candidate,
            repair_used=True,
            rejection_reasons=(),
            provider=candidate.provider,
            deterministic_validation=candidate.deterministic_validation,
        )
    except ContractError as exc:
        return GenerationOutcome(
            candidate=None,
            repair_used=True,
            rejection_reasons=exc.reasons,
            provider=_provider_record(response, cache),
            deterministic_validation={
                "machine_pass": False,
                "reasons": list(exc.reasons),
                "diagnostics": {},
            },
        )


def _validator_a_call(
    row: PreparedRow,
    candidate: Candidate,
    *,
    output_root: Path,
    backend: ProviderBackend,
    identity_registry: ProviderIdentityRegistry,
    environment: Mapping[str, str] | None,
    config: ProviderConfig,
    code_sha256: str,
) -> tuple[dict[str, Any], bool, list[str], dict[str, Any]]:
    role = (
        ROLE_VALIDATOR_A_REPAIR
        if candidate.attempt == "repair"
        else ROLE_VALIDATOR_A_PRIMARY
    )
    prompt = _validator_a_user_prompt(row, candidate)
    cache, response = _load_or_call(
        role=role,
        row=row,
        output_root=output_root,
        system_prompt=VALIDATOR_A_SYSTEM_PROMPT,
        user_prompt=prompt,
        config=config,
        backend=backend,
        identity_registry=identity_registry,
        environment=environment,
        code_sha256=code_sha256,
    )
    try:
        result, machine_pass, reasons = _validate_validator_a(row, candidate, response)
    except ContractError as exc:
        result, machine_pass, reasons = {}, False, list(exc.reasons)
    return result, machine_pass, reasons, _provider_record(response, cache)


def _validator_b_call(
    row: PreparedRow,
    candidate: Candidate,
    *,
    output_root: Path,
    backend: ProviderBackend,
    identity_registry: ProviderIdentityRegistry,
    environment: Mapping[str, str] | None,
    config: ProviderConfig,
    code_sha256: str,
) -> tuple[dict[str, Any], bool, bool, list[str], dict[str, Any]]:
    prompt = _validator_b_user_prompt(row, candidate)
    cache, response = _load_or_call(
        role=ROLE_VALIDATOR_B,
        row=row,
        output_root=output_root,
        system_prompt=VALIDATOR_B_SYSTEM_PROMPT,
        user_prompt=prompt,
        config=config,
        backend=backend,
        identity_registry=identity_registry,
        environment=environment,
        code_sha256=code_sha256,
    )
    result, report_complete, diagnostic_pass, reasons = _validate_validator_b(
        row, candidate, response
    )
    return (
        result,
        report_complete,
        diagnostic_pass,
        reasons,
        _provider_record(response, cache),
    )


def _terminal_binding(row: PreparedRow, *, code_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "sample_id": row.sample_id,
        "split": row.split,
        "source_analysis_sha256": row.source_analysis_sha256,
        "official_minutes_sha256": row.official_minutes_sha256,
        "student_system_prompt_sha256": sha256_text(STUDENT_SYSTEM_PROMPT),
        "code_sha256": code_sha256,
    }


def _terminal_cache_path(output_root: Path, row: PreparedRow) -> Path:
    return output_root / "cache" / "terminal" / f"{sha256_text(row.sample_id)}.json"


def _validate_terminal_record(row: PreparedRow, record: Mapping[str, Any]) -> None:
    status = record.get("terminal_status")
    training_pass = record.get("training_pass")
    if status not in TERMINAL_STATUSES:
        raise SyntheticRewriteError(f"invalid terminal status: {row.sample_id}")
    if not isinstance(training_pass, bool) or training_pass != (status == TERMINAL_PASS):
        raise SyntheticRewriteError(f"terminal training_pass mismatch: {row.sample_id}")
    for field, text in (
        ("source_analysis_sha256", row.source_analysis),
        ("official_minutes_sha256", row.official_minutes),
    ):
        if record.get(field) != sha256_text(text):
            raise SyntheticRewriteError(f"terminal {field} mismatch: {row.sample_id}")
    if training_pass:
        prompt = record.get("student_prompt")
        response = record.get("sft_response")
        if prompt != row.student_prompt or not isinstance(response, str) or not response:
            raise SyntheticRewriteError(f"terminal PASS text invalid: {row.sample_id}")
        reference = record.get("reference_diagnostic")
        if not isinstance(reference, dict) or reference.get("complete") is not True:
            raise SyntheticRewriteError(
                f"terminal PASS reference diagnostic incomplete: {row.sample_id}"
            )


def _load_terminal(
    output_root: Path,
    row: PreparedRow,
    *,
    code_sha256: str,
    identity_registry: ProviderIdentityRegistry,
) -> dict[str, Any] | None:
    path = _terminal_cache_path(output_root, row)
    if not path.is_file():
        return None
    payload = _read_json(path, label="terminal cache")
    binding = _terminal_binding(row, code_sha256=code_sha256)
    if payload.get("binding") != binding:
        raise SyntheticRewriteError(f"terminal cache binding mismatch: {path}")
    record = payload.get("record")
    if not isinstance(record, dict):
        raise SyntheticRewriteError(f"terminal cache record invalid: {path}")
    _validate_terminal_record(row, record)
    generation = record.get("generation")
    validator_a = record.get("validator_a")
    reference = record.get("reference_diagnostic")
    if isinstance(generation, dict):
        attempt = str(generation.get("selected_attempt") or "primary")
        role = ROLE_REWRITE_REPAIR if attempt == "repair" else ROLE_REWRITE_PRIMARY
        provider = generation.get("provider")
        if isinstance(provider, dict):
            identity_registry.bind_record(role, provider)
    if isinstance(validator_a, dict) and validator_a.get("complete") is True:
        attempt = str(generation.get("selected_attempt") if isinstance(generation, dict) else "primary")
        role = ROLE_VALIDATOR_A_REPAIR if attempt == "repair" else ROLE_VALIDATOR_A_PRIMARY
        provider = validator_a.get("provider")
        if isinstance(provider, dict):
            identity_registry.bind_record(role, provider)
    if isinstance(reference, dict) and reference.get("complete") is True:
        provider = reference.get("provider")
        if isinstance(provider, dict):
            identity_registry.bind_record(ROLE_VALIDATOR_B, provider)
    return record


def _store_terminal(
    output_root: Path,
    row: PreparedRow,
    record: Mapping[str, Any],
    *,
    code_sha256: str,
) -> dict[str, Any]:
    _validate_terminal_record(row, record)
    payload = {
        "binding": _terminal_binding(row, code_sha256=code_sha256),
        "record": dict(record),
    }
    _store_immutable_json(_terminal_cache_path(output_root, row), payload)
    return dict(record)


def _base_terminal_record(row: PreparedRow) -> dict[str, Any]:
    return {
        "schema_version": TERMINAL_SCHEMA_VERSION,
        "sample_id": row.sample_id,
        "split": row.split,
        "source_index": row.source_index,
        "meeting_date": row.meeting_date,
        "source_analysis": row.source_analysis,
        "source_analysis_sha256": row.source_analysis_sha256,
        "official_minutes_sha256": row.official_minutes_sha256,
        "teacher_response_analysis": None,
        "rewritten_minutes": None,
        "student_prompt": None,
        "sft_response": None,
        "teacher_response_analysis_sha256": None,
        "rewritten_minutes_sha256": None,
        "prompt_sha256": None,
        "response_sha256": None,
        "terminal_status": None,
        "training_pass": False,
        "rejection_stage": None,
        "rejection_reasons": [],
        "generation": {
            "selected_attempt": None,
            "repair_used": False,
            "deterministic_validation": {
                "machine_pass": False,
                "reasons": [],
                "diagnostics": {},
            },
            "provider": {},
        },
        "validator_a": {
            "complete": False,
            "machine_pass": False,
            "reasons": [],
            "result": None,
            "provider": {},
        },
        "reference_diagnostic": {
            "complete": False,
            "diagnostic_pass": None,
            "overall_score": None,
            "warning": None,
            "reasons": [],
            "result": None,
            "provider": {},
        },
        "lineage": _source_lineage(),
    }


def _record_generation_reject(
    row: PreparedRow, outcome: GenerationOutcome
) -> dict[str, Any]:
    record = _base_terminal_record(row)
    record.update(
        {
            "terminal_status": TERMINAL_GENERATION_REJECT,
            "training_pass": False,
            "rejection_stage": "generation_deterministic_gate",
            "rejection_reasons": list(outcome.rejection_reasons),
        }
    )
    record["generation"].update(
        {
            "selected_attempt": "repair" if outcome.repair_used else "primary",
            "repair_used": outcome.repair_used,
            "deterministic_validation": dict(outcome.deterministic_validation),
            "provider": dict(outcome.provider),
        }
    )
    return record


def _process_verify_row(
    row: PreparedRow,
    *,
    output_root: Path,
    tokenizer: Any,
    backend: ProviderBackend,
    identity_registry: ProviderIdentityRegistry,
    environment: Mapping[str, str] | None,
    config: ProviderConfig,
    code_sha256: str,
) -> dict[str, Any]:
    existing = _load_terminal(
        output_root,
        row,
        code_sha256=code_sha256,
        identity_registry=identity_registry,
    )
    if existing is not None:
        return existing
    generation = _acquire_generation(
        row,
        output_root=output_root,
        tokenizer=tokenizer,
        backend=backend,
        identity_registry=identity_registry,
        environment=environment,
        config=config,
        code_sha256=code_sha256,
    )
    if generation.candidate is None:
        return _store_terminal(
            output_root,
            row,
            _record_generation_reject(row, generation),
            code_sha256=code_sha256,
        )

    candidate = generation.candidate
    validator_result, validator_pass, validator_reasons, validator_provider = (
        _validator_a_call(
            row,
            candidate,
            output_root=output_root,
            backend=backend,
            identity_registry=identity_registry,
            environment=environment,
            config=config,
            code_sha256=code_sha256,
        )
    )
    if not validator_pass and not generation.repair_used:
        generation = _acquire_repair_after_validator_a(
            row,
            reasons=validator_reasons,
            validator_a_result=validator_result,
            output_root=output_root,
            tokenizer=tokenizer,
            backend=backend,
            identity_registry=identity_registry,
            environment=environment,
            config=config,
            code_sha256=code_sha256,
        )
        if generation.candidate is None:
            return _store_terminal(
                output_root,
                row,
                _record_generation_reject(row, generation),
                code_sha256=code_sha256,
            )
        candidate = generation.candidate
        validator_result, validator_pass, validator_reasons, validator_provider = (
            _validator_a_call(
                row,
                candidate,
                output_root=output_root,
                backend=backend,
                identity_registry=identity_registry,
                environment=environment,
                config=config,
                code_sha256=code_sha256,
            )
        )

    record = _base_terminal_record(row)
    record.update(
        {
            "teacher_response_analysis": candidate.teacher_response_analysis,
            "rewritten_minutes": candidate.rewritten_minutes,
            "teacher_response_analysis_sha256": sha256_text(
                candidate.teacher_response_analysis
            ),
            "rewritten_minutes_sha256": sha256_text(candidate.rewritten_minutes),
        }
    )
    record["generation"] = {
        "selected_attempt": candidate.attempt,
        "repair_used": generation.repair_used,
        "deterministic_validation": dict(candidate.deterministic_validation),
        "provider": dict(candidate.provider),
    }
    record["validator_a"] = {
        "complete": True,
        "machine_pass": validator_pass,
        "reasons": list(validator_reasons),
        "result": validator_result,
        "provider": validator_provider,
    }
    if not validator_pass:
        record.update(
            {
                "terminal_status": TERMINAL_FIDELITY_REJECT,
                "training_pass": False,
                "rejection_stage": "validator_a_input_fidelity",
                "rejection_reasons": list(validator_reasons),
            }
        )
        return _store_terminal(
            output_root, row, record, code_sha256=code_sha256
        )

    (
        reference_result,
        reference_complete,
        reference_pass,
        reference_reasons,
        reference_provider,
    ) = _validator_b_call(
            row,
            candidate,
            output_root=output_root,
            backend=backend,
            identity_registry=identity_registry,
            environment=environment,
            config=config,
            code_sha256=code_sha256,
        )
    if not reference_complete:
        raise ContractError(
            ["reference_diagnostic_report_incomplete", *reference_reasons]
        )
    record.update(
        {
            "terminal_status": TERMINAL_PASS,
            "training_pass": True,
            "rejection_stage": None,
            "rejection_reasons": [],
            "student_prompt": row.student_prompt,
            "sft_response": candidate.sft_response,
            "prompt_sha256": sha256_text(row.student_prompt),
            "response_sha256": sha256_text(candidate.sft_response),
        }
    )
    record["reference_diagnostic"] = {
        "complete": reference_complete,
        "diagnostic_pass": reference_pass,
        "overall_score": reference_result["overall_score"],
        "warning": None if reference_pass else "REFERENCE_DIAGNOSTIC_WARNING",
        "reasons": list(reference_reasons),
        "result": reference_result,
        "provider": reference_provider,
    }
    return _store_terminal(output_root, row, record, code_sha256=code_sha256)


def _flatten(prepared: Mapping[str, Sequence[PreparedRow]]) -> list[PreparedRow]:
    return [row for split in SPLITS for row in prepared.get(split, ())]


def _select_preflight_rows(
    prepared: Mapping[str, Sequence[PreparedRow]], count: int
) -> list[PreparedRow]:
    selected: list[PreparedRow] = []
    positions = {split: 0 for split in SPLITS}
    while len(selected) < count:
        progressed = False
        for split in SPLITS:
            rows = prepared.get(split, ())
            index = positions[split]
            if index < len(rows) and len(selected) < count:
                selected.append(rows[index])
                positions[split] += 1
                progressed = True
        if not progressed:
            break
    return selected


def _run_wave(
    rows: Sequence[PreparedRow],
    *,
    worker: Any,
    concurrency: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    results: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=concurrency, thread_name_prefix="paper-chk2-rewrite"
    ) as executor:
        futures: dict[Future[dict[str, Any]], PreparedRow] = {
            executor.submit(worker, row): row for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                results[row.sample_id] = future.result()
            except Exception as exc:  # fail closed; no secret-bearing request dump
                failures.append(
                    {
                        "sample_id": row.sample_id,
                        "split": row.split,
                        "source_index": row.source_index,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    }
                )
    return results, failures


def _artifact_record(path: Path, *, rows: int) -> dict[str, Any]:
    return {
        "path": _display_path(path),
        "rows": rows,
        "sha256": sha256_file(path),
    }


def _summary_source_binding(preparation: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "root",
        "summary_sha256",
        "handoff_sha256",
        "merge_receipt_sha256",
        "sample_id_sha256",
        "split_counts",
        "total_rows",
        "artifacts",
    }
    missing = sorted(keys - set(preparation))
    if missing:
        raise SyntheticRewriteError(f"preparation source binding missing: {missing}")
    return {key: preparation[key] for key in sorted(keys)}


def _materialize_final(
    prepared: Mapping[str, Sequence[PreparedRow]],
    *,
    output_root: Path,
    terminals: Mapping[str, Mapping[str, Any]],
    preparation: Mapping[str, Any],
    prompt_contract_sha256: str,
    identities: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
    phase: str,
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    split_counts: dict[str, Any] = {}
    for split in SPLITS:
        terminal_rows = [dict(terminals[row.sample_id]) for row in prepared.get(split, ())]
        pass_rows = [row for row in terminal_rows if row["training_pass"] is True]
        candidate_rows = [
            {"prompt": row["student_prompt"], "response": row["sft_response"]}
            for row in pass_rows
        ]
        manifest_rows = [
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "sample_id": row["sample_id"],
                "split": row["split"],
                "source_index": row["source_index"],
                "meeting_date": row["meeting_date"],
                "source_analysis": row["source_analysis"],
                "teacher_response_analysis": row["teacher_response_analysis"],
                "rewritten_minutes": row["rewritten_minutes"],
                "source_analysis_sha256": row["source_analysis_sha256"],
                "official_minutes_sha256": row["official_minutes_sha256"],
                "teacher_response_analysis_sha256": row[
                    "teacher_response_analysis_sha256"
                ],
                "rewritten_minutes_sha256": row["rewritten_minutes_sha256"],
                "prompt_sha256": row["prompt_sha256"],
                "response_sha256": row["response_sha256"],
                "generation": row["generation"],
                "validator_a": row["validator_a"],
                "reference_diagnostic": row["reference_diagnostic"],
                "terminal_status": row["terminal_status"],
                "training_pass": True,
                **row["lineage"],
            }
            for row in pass_rows
        ]
        terminal_path = output_root / "terminal" / f"{split}.jsonl"
        candidate_path = output_root / "sft_candidate" / f"{split}.jsonl"
        manifest_path = output_root / "manifests" / f"{split}.jsonl"
        _write_jsonl(terminal_path, terminal_rows)
        _write_jsonl(candidate_path, candidate_rows)
        _write_jsonl(manifest_path, manifest_rows)
        artifacts[split] = {
            "terminal": _artifact_record(terminal_path, rows=len(terminal_rows)),
            "sft_candidate": _artifact_record(candidate_path, rows=len(candidate_rows)),
            "manifest": _artifact_record(manifest_path, rows=len(manifest_rows)),
        }
        status_counter = Counter(row["terminal_status"] for row in terminal_rows)
        warnings = sum(
            row["reference_diagnostic"]["warning"] is not None for row in pass_rows
        )
        split_counts[split] = {
            "source_rows": len(terminal_rows),
            "training_pass": len(pass_rows),
            "generation_quality_reject": status_counter[TERMINAL_GENERATION_REJECT],
            "input_fidelity_reject": status_counter[TERMINAL_FIDELITY_REJECT],
            "reference_diagnostic_warnings": warnings,
        }
    total_source = sum(value["source_rows"] for value in split_counts.values())
    total_pass = sum(value["training_pass"] for value in split_counts.values())
    unresolved = len(failures)
    all_splits_nonempty = all(
        split_counts[split]["training_pass"] > 0 for split in SPLITS
    )
    status = (
        "complete"
        if unresolved == 0 and total_source == len(_flatten(prepared))
        else "incomplete"
    )
    if status == "complete" and not all_splits_nonempty:
        status = "complete_no_publish_empty_split"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": status,
        "quality_status": (
            "passed" if status == "complete" and all_splits_nonempty else "failed"
        ),
        "phase": phase,
        "teacher_model": MODEL,
        "split_counts": split_counts,
        "total_source_rows": total_source,
        "total_training_pass": total_pass,
        "total_quality_reject": total_source - total_pass,
        "reference_diagnostic_affects_selection": False,
        "unresolved_failure_count": unresolved,
        "all_splits_nonempty": all_splits_nonempty,
        "training_ready": status == "complete" and all_splits_nonempty,
        "evaluation_eligible": False,
        "source": _summary_source_binding(preparation),
        "prompt_contract_sha256": prompt_contract_sha256,
        "provider_identities": dict(identities),
        "artifacts": artifacts,
        "lineage": _source_lineage(),
    }
    _write_json(output_root / "summary.json", summary)
    _write_jsonl(output_root / "failures.jsonl", [dict(item) for item in failures])
    _write_json(
        output_root / "handoff.json",
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": status,
            "dataset_path": _display_path(output_root / "sft_candidate"),
            "manifest_path": _display_path(output_root / "manifests"),
            "terminal_path": _display_path(output_root / "terminal"),
            "summary": {
                "path": "summary.json",
                "sha256": sha256_file(output_root / "summary.json"),
            },
            "split_counts": {
                split: split_counts[split]["training_pass"] for split in SPLITS
            },
            "training_ready": summary["training_ready"],
            "evaluation_eligible": False,
            "lineage": _source_lineage(),
        },
    )
    return summary


def run_pipeline(
    prepared: Mapping[str, Sequence[PreparedRow]],
    *,
    preparation: Mapping[str, Any],
    output_root: str | Path,
    tokenizer: Any,
    backend: ProviderBackend,
    environment: Mapping[str, str] | None,
    phase: str = "all",
    concurrency: int = DEFAULT_CONCURRENCY,
    preflight_rows: int = DEFAULT_PREFLIGHT_ROWS,
    resume: bool = False,
) -> dict[str, Any]:
    """Run generation and/or verification without a quality coverage floor."""

    if phase not in {"prepare", "generate", "verify", "all"}:
        raise SyntheticRewriteError(f"invalid phase: {phase}")
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise SyntheticRewriteError(f"concurrency must be in [1,{MAX_CONCURRENCY}]")
    if not 1 <= preflight_rows <= 64:
        raise SyntheticRewriteError("preflight_rows must be in [1,64]")
    output = Path(output_root).resolve()
    code_sha256 = sha256_file(Path(__file__).resolve())
    config = ProviderConfig()
    contract = _prompt_contract(code_sha256=code_sha256, config=config)
    _write_json(output / "prompt_contract.json", contract)
    if phase == "prepare":
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "prepared",
            "phase": phase,
            "source": _summary_source_binding(preparation),
            "prompt_contract_sha256": sha256_file(output / "prompt_contract.json"),
            "training_ready": False,
        }
        _write_json(output / "summary.json", summary)
        return summary

    rows = _flatten(prepared)
    if not rows:
        raise SyntheticRewriteError("prepared population is empty")
    relevant_roles = (
        (ROLE_REWRITE_PRIMARY, ROLE_REWRITE_REPAIR)
        if phase == "generate"
        else PROVIDER_ROLES + ("terminal",)
    )
    if not resume:
        for role in relevant_roles:
            root = output / "cache" / role
            existing = next(root.glob("*.json"), None) if root.is_dir() else None
            if existing is not None:
                raise SyntheticRewriteError(
                    f"{role} cache exists; use --resume: {existing}"
                )

    identities = ProviderIdentityRegistry()
    failures: list[dict[str, Any]] = []
    preflight = _select_preflight_rows(prepared, min(preflight_rows, len(rows)))

    if phase == "generate":
        def generation_worker(row: PreparedRow) -> dict[str, Any]:
            outcome = _acquire_generation(
                row,
                output_root=output,
                tokenizer=tokenizer,
                backend=backend,
                identity_registry=identities,
                environment=environment,
                config=config,
                code_sha256=code_sha256,
            )
            return {
                "sample_id": row.sample_id,
                "split": row.split,
                "deterministic_pass": outcome.candidate is not None,
                "repair_used": outcome.repair_used,
                "rejection_reasons": list(outcome.rejection_reasons),
            }

        preflight_results, preflight_failures = _run_wave(
            preflight, worker=generation_worker, concurrency=concurrency
        )
        failures.extend(preflight_failures)
        _write_json(
            output / "preflight.json",
            {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "phase": phase,
                "rows": len(preflight),
                "classified": len(preflight_results),
                "quality_pass_floor": None,
                "failures": preflight_failures,
                "passed": not preflight_failures,
            },
        )
        if preflight_failures:
            raise SyntheticRewriteError("generation preflight has unresolved failures")
        remaining = [row for row in rows if row.sample_id not in preflight_results]
        remaining_results, remaining_failures = _run_wave(
            remaining, worker=generation_worker, concurrency=concurrency
        )
        failures.extend(remaining_failures)
        all_results = {**preflight_results, **remaining_results}
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": (
                "generation_complete_verification_pending"
                if not failures and len(all_results) == len(rows)
                else "incomplete"
            ),
            "phase": phase,
            "source": _summary_source_binding(preparation),
            "total_source_rows": len(rows),
            "generation_classified": len(all_results),
            "generation_deterministic_pass": sum(
                bool(value["deterministic_pass"]) for value in all_results.values()
            ),
            "unresolved_failure_count": len(failures),
            "prompt_contract_sha256": sha256_file(output / "prompt_contract.json"),
            "provider_identities": identities.as_dict(),
            "training_ready": False,
        }
        _write_json(output / "summary.json", summary)
        _write_jsonl(output / "failures.jsonl", failures)
        return summary

    # verify/all: the same state machine loads exact generation caches on
    # resume, or acquires them when absent.  Validator-B semantic scores never
    # enter the terminal status calculation.
    def verify_worker(row: PreparedRow) -> dict[str, Any]:
        return _process_verify_row(
            row,
            output_root=output,
            tokenizer=tokenizer,
            backend=backend,
            identity_registry=identities,
            environment=environment,
            config=config,
            code_sha256=code_sha256,
        )

    preflight_results, preflight_failures = _run_wave(
        preflight, worker=verify_worker, concurrency=concurrency
    )
    failures.extend(preflight_failures)
    _write_json(
        output / "preflight.json",
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "phase": phase,
            "rows": len(preflight),
            "classified": len(preflight_results),
            "terminal_status_counts": dict(
                Counter(
                    row["terminal_status"] for row in preflight_results.values()
                )
            ),
            "quality_pass_floor": None,
            "failures": preflight_failures,
            "passed": not preflight_failures,
        },
    )
    if preflight_failures:
        raise SyntheticRewriteError("verification preflight has unresolved failures")
    remaining = [row for row in rows if row.sample_id not in preflight_results]
    remaining_results, remaining_failures = _run_wave(
        remaining, worker=verify_worker, concurrency=concurrency
    )
    failures.extend(remaining_failures)
    terminals = {**preflight_results, **remaining_results}
    if failures:
        # Do not materialize authoritative ledgers from a partial population.
        # Successful immutable row caches remain resumable, while the failure
        # ledger and summary make the unresolved state explicit and prevent a
        # publisher from mistaking a short terminal file for a complete run.
        incomplete_summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "incomplete",
            "quality_status": "failed",
            "phase": phase,
            "source": _summary_source_binding(preparation),
            "total_source_rows": len(rows),
            "terminal_classified": len(terminals),
            "unresolved_failure_count": len(failures),
            "prompt_contract_sha256": sha256_file(
                output / "prompt_contract.json"
            ),
            "provider_identities": identities.as_dict(),
            "training_ready": False,
            "evaluation_eligible": False,
            "lineage": _source_lineage(),
        }
        _write_json(output / "summary.json", incomplete_summary)
        _write_jsonl(output / "failures.jsonl", failures)
        raise SyntheticRewriteError(
            "verification has unresolved failures; resume after correcting "
            "the provider or contract error"
        )
    return _materialize_final(
        prepared,
        output_root=output,
        terminals=terminals,
        preparation=preparation,
        prompt_contract_sha256=sha256_file(output / "prompt_contract.json"),
        identities=identities.as_dict(),
        failures=failures,
        phase=phase,
    )


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SyntheticRewriteError("transformers is unavailable") from exc
    if not path.is_dir():
        raise SyntheticRewriteError(f"tokenizer path is missing: {path}")
    return AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument(
        "--phase", choices=("prepare", "generate", "verify", "all"), default="all"
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--preflight-rows", type=int, default=DEFAULT_PREFLIGHT_ROWS)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepared, preparation = prepare_source_release(
        source_root=args.source_root,
        output_root=args.output_root,
        enforce_pins=True,
    )
    tokenizer = _load_tokenizer(args.tokenizer_path)
    if args.phase != "prepare" and not os.environ.get(API_KEY_ENV, "").strip():
        raise SyntheticRewriteError(f"missing provider credential: {API_KEY_ENV}")
    summary = run_pipeline(
        prepared,
        preparation=preparation,
        output_root=args.output_root,
        tokenizer=tokenizer,
        backend=OpenAICompatibleBackend(),
        environment=os.environ,
        phase=args.phase,
        concurrency=args.concurrency,
        preflight_rows=args.preflight_rows,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
