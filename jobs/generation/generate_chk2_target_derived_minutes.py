"""Build target-derived analysis-to-official-Minutes SFT candidates.

This job deliberately ignores every legacy ``raw_analysis``, rewrite trace,
and synthetic rewrite.  It uses only the row-associated official Minutes
paragraph and the immutable meeting-level split lineage from the legacy
manifests.  DeepSeek V4 Pro performs two separate operations:

1. official paragraph -> atomic claim card, complete paraphrased analysis,
   and target-conditioned fidelity reasoning;
2. an independent, adversarial bidirectional-entailment verification call.

The two waves are sent concurrently with bounded thread pools.  Successful
provider responses are stored in immutable, hash-bound per-row caches so a
run can be resumed without paying for completed calls again.  The student
sees only the generated analysis.  A machine-passing completion is exactly::

    fidelity_reasoning\n</think>\nliteral_official_minutes_paragraph

This construction is intentionally target-derived.  It is training-only,
not reference-free, and not suitable for leakage-safe evaluation.  Machine
PASS remains ``training_ready=false`` until human review is completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from jobs.generation.generate_chk3_sft_targets import (
    CONTROL_MARKERS,
    _attribution_categories,
    _date_values,
    _numeric_values,
    _reasoning_meta_categories,
    build_sft_completion,
    canonical_json,
    render_user_prompt,
)
from jobs.generation.materialize_chk2_official_minutes import (
    _ADMIN_OR_POLICY_RE,
    _BLOCKING_SOURCE_FLAGS,
    _SECTION_HEADING_PREFIX_RE,
    _normalized_official_paragraph,
    _official_source_matches,
    _source_file_records,
)
from jobs.retrain_v2.token_budget_gate import _count_sft_row


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "dataset/processed/train/minutes_alignment"
DEFAULT_SPLIT_MANIFEST = REPO_ROOT / (
    "dataset/processed/pipeline/analysis_sft/audit/meeting_split_manifest.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk2/target_derived_minutes_deepseek_v4_flash_v1_20260828"
)
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "models/DeepSeek-R1-Distill-Llama-8B"

SOURCE_SPLITS = ("train", "eval", "test")
OUTPUT_SPLITS = ("train", "validation", "test")
SOURCE_TO_OUTPUT = {"train": "train", "eval": "validation", "test": "test"}
OUTPUT_TO_SOURCE = {value: key for key, value in SOURCE_TO_OUTPUT.items()}

TEACHER_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
API_KEY_ENV = "DEEPSEEK_API_KEY"
BASE_URL_ENV = "DEEPSEEK_BASE_URL"
REVISION_ENV = "DEEPSEEK_TEACHER_REVISION"
DEFAULT_CONCURRENCY = 8
MAX_CONCURRENCY = 32
DEFAULT_PREFLIGHT_ROWS = 8
PREFLIGHT_MIN_PASS_FRACTION = 0.75

MIN_TARGET_WORDS = 20
MAX_TARGET_WORDS = 400
MIN_ANALYSIS_WORDS = 20
MAX_ANALYSIS_WORDS = 700
MIN_REASONING_WORDS = 35
MAX_REASONING_WORDS = 500
PROMPT_TOKEN_LIMIT = 3072
TOTAL_TOKEN_LIMIT = 4096

PREPARED_SCHEMA_VERSION = "chk2-target-derived-prepared-v1"
GENERATION_CACHE_SCHEMA_VERSION = "chk2-target-derived-generation-cache-v1"
VERIFICATION_CACHE_SCHEMA_VERSION = "chk2-target-derived-verification-cache-v1"
MANIFEST_SCHEMA_VERSION = "chk2-target-derived-manifest-v1"
SUMMARY_SCHEMA_VERSION = "chk2-target-derived-summary-v1"
PROMPT_CONTRACT_SCHEMA_VERSION = "chk2-target-derived-prompt-contract-v1"


STUDENT_SYSTEM_PROMPT = """\
You are a Federal Reserve Minutes editor. The user supplies a complete
economic or financial analysis. In the native reasoning section, identify
every substantive claim, quantity, date, direction, comparison, attribution,
causal relation, and expression of uncertainty that the formal rewrite must
preserve. Then express the same information as exactly one formal FOMC
Minutes paragraph.

Do not add, remove, broaden, narrow, or contradict any substantive claim.
Preserve the numeric-quantity multiset and every explicit calendar reference.
Reason only about the supplied analysis and rewrite. Do not discuss prompts,
instructions, schemas, APIs, validation, teacher or student roles, or output
protocols. Do not emit headings, lists, JSON, citations, answer tags, or
model-control tags.
"""

GENERATION_SYSTEM_PROMPT = """\
You are DeepSeek V4 Pro constructing training-only supervision for an
analysis-to-FOMC-Minutes rewriting model. The visible official Minutes
paragraph is the sole factual source.

First de-style it into a complete, neutral economic or financial analysis.
The analysis must be bidirectionally equivalent to the paragraph: preserve
every fact, entity, quantity, calendar reference, direction, comparison,
scope, attribution, causal relation, and expression of uncertainty, while
adding nothing. Reorganize and paraphrase substantially; do not copy the
Minutes paragraph or imitate Minutes prose. Preserve numeric strings and
explicit month/year expressions so deterministic fidelity checks can compare
them. Do not use headings, Markdown, bullets, citations, or outside knowledge.

Also write concise substantive fidelity reasoning suitable for supervising
the student's hidden reasoning. It must account for all claims and structured
facts without discussing the task, prompts, JSON, schemas, validation, roles,
or output protocol. Use declarative editorial reasoning rather than narrating
what you were asked to do.

Return exactly one JSON object with exactly these keys:
- atomic_claim_card: nonempty array of objects with exactly claim_id,
  target_span, proposition. claim_id values are c1, c2, ...; target_span is an
  exact nonempty substring of the official paragraph; proposition is a
  faithful atomic paraphrase.
- analysis: the complete de-styled analysis string.
- fidelity_reasoning: the target-conditioned substantive reasoning string.
- claim_alignment: one object per claim with exactly claim_id and
  analysis_evidence, where analysis_evidence is an exact nonempty substring
  of analysis. Every claim_id must appear exactly once.

Do not place commentary outside the JSON object.
"""

VERIFICATION_SYSTEM_PROMPT = """\
Act as an adversarial verifier for target-derived training data. Judge only
the supplied official paragraph, generated analysis, atomic claim card, and
fidelity reasoning. Topical similarity is insufficient. Decompose both texts
into claims and demand explicit textual evidence for every verdict. Use no
outside knowledge and fail closed on ambiguity.

Return exactly one JSON object with exactly these keys:
- target_claims: a nonempty array of objects with exactly target_span,
  analysis_evidence, verdict. Both spans must be exact substrings of their
  respective texts. verdict is entailed, unsupported, or contradicted.
- analysis_claims: a nonempty array of objects with exactly analysis_span,
  target_evidence, verdict. Both spans must be exact substrings of their
  respective texts. verdict is supported or unsupported.
- reasoning_compatible: boolean, true only if the reasoning covers and does
  not conflict with the official paragraph and generated analysis.
- bidirectional_entailment: boolean, true only if the two prose texts express
  the same substantive information in both directions.
- issues: array of concise strings; empty only when no issue exists.
- overall_pass: boolean. It may be true only when every target claim is
  entailed, every analysis claim is supported, both booleans are true, and
  issues is empty.

Do not place commentary outside the JSON object.
"""

GENERATION_USER_PREFIX = "Official Minutes paragraph (sole factual source):\n\n"
VERIFICATION_USER_PREFIX = "Verify this target-derived candidate:\n\n"


_SECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "core_economic_financial",
        re.compile(
            r"^(?:Staff Review of (?:the )?(?:Economic|Financial|Economic and "
            r"Financial) Situation|Staff Economic Outlook|"
            r"Participants['’] View(?:s)? (?:on|of) Current Conditions and "
            r"(?:the )?Economic Outlook)$",
            re.IGNORECASE,
        ),
    ),
    (
        "descriptive_market_operations",
        re.compile(
            r"^(?:Developments in Financial Markets and Open Market Operations|"
            r"Developments in Financial Markets and the Federal Reserve['’]s "
            r"Balance Sheet|Market Developments and Open Market Operations|"
            r"Financial Developments and Open Market Operations|"
            r"Discussion of Financial Markets and Open Market Operations|"
            r"Participants['’] Discussion of Recent Money Market Developments|"
            r"Recent Developments in the Banking Sector)$",
            re.IGNORECASE,
        ),
    ),
    (
        "descriptive_research_topic",
        re.compile(
            r"^(?:Inflation Analysis and Forecasting|Structural Unemployment|"
            r"Equilibrium Real Interest Rates|Dynamic Stochastic General "
            r"Equilibrium Models|Role of Financial Conditions in Economic "
            r"Recovery: Lending and Leverage)$",
            re.IGNORECASE,
        ),
    ),
)

_POLICY_OR_ADMIN_PARAGRAPH_RE = re.compile(
    r"\b(?:policy\s+(?:decision|action|stance|choice|option|implementation)|"
    r"forward\s+guidance|target\s+range\s+for\s+the\s+federal\s+funds\s+rate|"
    r"decided\s+to\s+(?:maintain|raise|lower|reduce|increase|continue)|"
    r"agreed\s+(?:that|to)\s+(?:maintain|raise|lower|reduce|increase|continue)|"
    r"supported\s+(?:maintaining|raising|lowering|reducing|increasing)|"
    r"favou?red\s+(?:maintaining|raising|lowering|reducing|increasing)|"
    r"following\s+the\s+discussion,?\s+the\s+committee|"
    r"next\s+(?:scheduled\s+)?meeting|meeting\s+adjourned|"
    r"by\s+unanimous\s+vote|ratified|reaffirmed|authorizes?\s+and\s+directs?)\b",
    re.IGNORECASE,
)

_POLICY_DELIBERATION_RE = re.compile(
    r"(?:"
    r"\b(?:in|with\s+regard\s+to)\s+(?:their\s+)?discussion\s+of\s+"
    r"(?:monetary\s+)?policy\b|"
    r"\b(?:meeting\s+participants|(?<!market\s)participants|the\s+committee|"
    r"committee\s+members|policymakers)\b.{0,240}"
    r"\b(?:discussed|considered|supported|favou?red|preferred|agreed|decided|"
    r"endorsed|recommended|expressed\s+support|viewed\s+as\s+appropriate|"
    r"judged\s+appropriate|should|needed?|would\s+need|could\s+be\s+useful)\b"
    r".{0,300}\b(?:monetary\s+policy|policy\s+(?:accommodation|firming|rate|"
    r"stance|tools?|outlook)|federal\s+funds\s+rate|target\s+range|"
    r"asset\s+(?:purchases?|sales?)|purchases?\s+of\s+(?:assets?|securities)|"
    r"securities\s+(?:purchases?|holdings?)|balance\s+sheet|reinvest(?:ment|ing)?|"
    r"redemptions?|the\s+desk|open\s+market\s+(?:purchases?|operations)|"
    r"reserve\s+balances?|forward\s+guidance)\b|"
    r"\b(?:meeting\s+participants|(?<!market\s)participants|the\s+committee|"
    r"committee\s+members|policymakers)\b.{0,180}"
    r"\b(?:monetary\s+policy|policy\s+(?:accommodation|firming|rate|stance|"
    r"tools?|outlook)|federal\s+funds\s+rate|target\s+range|"
    r"asset\s+(?:purchases?|sales?)|securities\s+(?:purchases?|holdings?)|"
    r"balance\s+sheet|reinvest(?:ment|ing)?|redemptions?|the\s+desk|"
    r"forward\s+guidance)\b.{0,300}"
    r"\b(?:discussed|considered|supported|favou?red|preferred|agreed|decided|"
    r"endorsed|recommended|expressed\s+support|appropriate|warranted|should|"
    r"needed?|would\s+need|could\s+be\s+useful)\b|"
    r"\bstaff\b.{0,100}\b(?:briefed|presented|reviewed|provided)\b.{0,180}"
    r"\b(?:possible|alternative|feasible|options?|strateg(?:y|ies)|plans?)\b"
    r".{0,180}\b(?:policy|asset\s+purchases?|asset\s+sales?|balance\s+sheet|"
    r"federal\s+funds\s+rate|reserve\s+balances?|open\s+market)\b|"
    r"\b(?:options?|plans?|strateg(?:y|ies)|approaches?)\s+for\s+"
    r"(?:the\s+)?(?:continuation|implementation|reduction|normalization|"
    r"purchase|sale).{0,120}\b(?:assets?|securities|balance\s+sheet|policy)\b|"
    r"\b(?:current\s+)?authori[sz](?:ation|e|ed)\b"
    r")",
    re.IGNORECASE,
)

_POLICY_ACTOR_ACTION_RE = re.compile(
    r"\b(?:meeting\s+participants|(?<!market\s)participants|the\s+committee)\b"
    r".{0,220}\b(?:supported|favou?red|preferred|agreed|decided|discussed|"
    r"appropriate|warranted|should)\b.{0,260}"
    r"\b(?:target\s+range|federal\s+funds\s+rate|asset\s+purchases?|"
    r"asset\s+sales?|balance\s+sheet|monetary\s+policy|policy\s+rate|"
    r"securities\s+holdings?)\b|"
    r"\b(?:appropriate|preferred|projected)\s+(?:path|timing|stance|pace)\b"
    r".{0,120}\b(?:monetary\s+policy|policy\s+rate|federal\s+funds\s+rate|"
    r"rate\s+increases?)\b|"
    r"\b(?:monetary\s+policy|policy\s+rate|federal\s+funds\s+rate)\b"
    r".{0,120}\b(?:appropriate|preferred|warranted)\b|"
    r"\bparticipants['’]\s+discussion\s+of\s+policy\s+planning\b",
    re.IGNORECASE,
)

_POLICY_FRAME_RE = re.compile(
    r"\b(?:discussion|consideration)\s+of\s+"
    r"(?:(?:the\s+)?(?:path|outlook)\s+for\s+|"
    r"economic\s+conditions\s+and\s+)?monetary\s+policy\b|"
    r"\bin\s+discussing\b.{0,160}\boutlook\s+for\s+monetary\s+policy\b|"
    r"\b(?:committee(?:['’]s)?(?:\s+current)?|existing|outcome-based)\s+"
    r"guidance(?:\s+for\s+(?:the\s+federal\s+funds\s+rate|asset\s+purchases?))?\b|"
    r"\b(?:case\s+for\s+a\s+rate\s+cut|"
    r"proposed\s+quarter-point\s+policy\s+easing)\b|"
    r"\b(?:meeting\s+participants|(?<!market\s)participants|committee\s+members)"
    r"\b.{0,140}"
    r"\b(?:approved|supported|agreed|decided|endorsed)\b.{0,220}"
    r"\b(?:(?:liquidity\s+)?facilities|TALF|swap\s+arrangements|net\s+asset\s+"
    r"purchases?|policy\s+(?:approach|restraint|firming))\b|"
    r"\b(?:meeting\s+participants|(?<!market\s)participants|committee\s+members)"
    r"\b.{0,140}"
    r"\bjudged\b.{0,120}\bcurrent\s+stance\s+of\s+monetary\s+policy\b|"
    r"\bshould\s+the\s+committee\s+decide\b.{0,180}"
    r"\b(?:net\s+)?asset\s+purchases?\b|"
    r"\bcommittee(?:['’]s)?\s+future\s+decisions\s+regarding\s+policy\b|"
    r"\bpolicy\s+restraint\b.{0,140}\bcommittee\b.{0,100}"
    r"\b(?:putting|implementing)\b",
    re.IGNORECASE,
)

_FORMAL_ACTION_RE = re.compile(
    r"\b(?:the\s+committee|FOMC|federal\s+open\s+market\s+committee)\b"
    r".{0,140}\b(?:authoriz(?:e|es|ed)|unanimously\s+approved|"
    r"made\s+no\s+(?:decision|change))\b|"
    r"\ball\s+but\s+one\s+(?:member|participant)\b.{0,80}"
    r"\b(?:approved|agreed|supported)\b|:\s*$",
    re.IGNORECASE,
)

_GLUED_SECTION_HEADING_RE = re.compile(
    r"^(?:Staff Review of (?:the )?(?:Economic|Financial|Economic and Financial) "
    r"Situation|Staff Economic Outlook|Participants['’] View(?:s)? (?:on|of) "
    r"Current Conditions and (?:the )?Economic Outlook|Developments in Financial "
    r"Markets(?:, Open Market Operations, and Policy Normalization| and Open "
    r"Market Operations| and the Federal Reserve['’]s Balance Sheet)|Options for "
    r"the Continuation of Asset Purchases|Participants['’] Discussion of Policy "
    r"Planning)(?=[A-Z])",
    re.IGNORECASE,
)

_GLUED_TITLE_RE = re.compile(
    r"^[A-Z][A-Za-z0-9&,’'(): -]{4,100}?[a-z]"
    r"(?=(?:Staff|The|Participants|Meeting|Domestic|In)\b)"
)

_MARKDOWN_OR_HEADING_RE = re.compile(
    r"```|~~~|(?:^|\n)\s*(?:#{1,6}\s|[-*+]\s)|\*\*|__",
    re.MULTILINE,
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_SAFE_SAMPLE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SECRET_RE = re.compile(r"\b(?:sk-|bearer\s+)[A-Za-z0-9._-]{8,}\b", re.I)


class TargetDerivedDataError(RuntimeError):
    """The target-derived release cannot be built safely."""


class ModelDriftError(TargetDerivedDataError):
    """The provider identity differs within the same acquisition release."""


class ProviderRequestError(TargetDerivedDataError):
    """All transport attempts failed."""

    def __init__(self, message: str, *, attempts: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = tuple(dict(item) for item in attempts)


@dataclass(frozen=True)
class TeacherConfig:
    model: str = TEACHER_MODEL
    base_url: str = DEEPSEEK_BASE_URL
    revision: str = "provider-current"
    api_key_env: str = API_KEY_ENV
    timeout_seconds: float = 180.0
    max_retries: int = 3
    reasoning_effort: str = "high"

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> "TeacherConfig":
        env = os.environ if environment is None else environment
        model_override = str(env.get("DEEPSEEK_TEACHER_MODEL") or "").strip()
        if model_override and model_override != TEACHER_MODEL:
            raise TargetDerivedDataError(
                f"teacher model override is forbidden; expected {TEACHER_MODEL}"
            )
        return cls(
            base_url=str(env.get(BASE_URL_ENV) or DEEPSEEK_BASE_URL).strip(),
            revision=str(env.get(REVISION_ENV) or "provider-current").strip(),
        )

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if self.model != TEACHER_MODEL:
            raise TargetDerivedDataError(
                f"teacher must be exactly {TEACHER_MODEL}"
            )
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise TargetDerivedDataError("DeepSeek base URL must be absolute HTTP(S)")
        if parsed.scheme != "https" and parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise TargetDerivedDataError("remote DeepSeek endpoint must use HTTPS")
        if self.max_retries != 3:
            raise TargetDerivedDataError("teacher max_retries must remain 3")
        if self.reasoning_effort != "high":
            raise TargetDerivedDataError("reasoning_effort must remain high")

    def contract(self) -> dict[str, Any]:
        parsed = urlparse(self.base_url)
        return {
            "provider": "deepseek",
            "model": self.model,
            "revision": self.revision,
            "base_url_origin": f"{parsed.scheme}://{parsed.netloc}",
            "api_key_env": self.api_key_env,
            "max_tokens": None,
            "output_limit_policy": "provider_default_no_client_side_token_cap",
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "reasoning_effort": self.reasoning_effort,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled"},
            "unsupported_sampling_parameters": ["temperature", "top_p"],
            "fallback": "forbidden",
        }

    @property
    def contract_sha256(self) -> str:
        return sha256_text(canonical_json(self.contract()))


@dataclass(frozen=True)
class TeacherResponse:
    raw_reasoning: str
    raw_content: str
    response_id: str
    returned_model: str
    system_fingerprint: str
    finish_reason: str
    created: int | None
    usage: Mapping[str, int | None]
    attempts: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, payload: Mapping[str, Any]) -> "TeacherResponse":
        attempts = payload.get("attempts")
        usage = payload.get("usage")
        if not isinstance(attempts, list) or not all(
            isinstance(item, dict) for item in attempts
        ):
            raise TargetDerivedDataError("cached provider attempts are invalid")
        if not isinstance(usage, dict):
            raise TargetDerivedDataError("cached provider usage is invalid")
        created = payload.get("created")
        return cls(
            raw_reasoning=str(payload.get("raw_reasoning") or ""),
            raw_content=str(payload.get("raw_content") or ""),
            response_id=str(payload.get("response_id") or ""),
            returned_model=str(payload.get("returned_model") or ""),
            system_fingerprint=str(payload.get("system_fingerprint") or ""),
            finish_reason=str(payload.get("finish_reason") or ""),
            created=None if created is None else int(created),
            usage={str(key): _optional_int(value) for key, value in usage.items()},
            attempts=tuple(dict(item) for item in attempts),
        )


class TeacherBackend(Protocol):
    def generate(
        self,
        *,
        config: TeacherConfig,
        system_prompt: str,
        user_prompt: str,
        environment: Mapping[str, str] | None,
    ) -> TeacherResponse: ...


@dataclass(frozen=True)
class PreparedRow:
    sample_id: str
    split: str
    source_split: str
    source_index: int
    meeting_date: str
    section_name: str
    section_category: str
    topic: str
    line_id: str
    official_minutes_raw: str
    official_minutes_paragraph: str
    official_minutes_raw_sha256: str
    official_minutes_sha256: str
    official_normalization_repairs: tuple[str, ...]
    official_source_match: Mapping[str, Any]
    official_source_files: tuple[Mapping[str, Any], ...]
    legacy_sample_ids: tuple[str, ...]
    legacy_source_rows: tuple[Mapping[str, Any], ...]
    duplicate_cluster_size: int
    generation_user_prompt: str
    generation_prompt_sha256: str

    def prepared_record(self) -> dict[str, Any]:
        return {
            "schema_version": PREPARED_SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "split": self.split,
            "source_split": self.source_split,
            "source_index": self.source_index,
            "meeting_date": self.meeting_date,
            "section_name": self.section_name,
            "section_category": self.section_category,
            "topic": self.topic,
            "line_id": self.line_id,
            "official_minutes_raw": self.official_minutes_raw,
            "official_minutes_paragraph": self.official_minutes_paragraph,
            "official_minutes_raw_sha256": self.official_minutes_raw_sha256,
            "official_minutes_sha256": self.official_minutes_sha256,
            "official_normalization_repairs": list(self.official_normalization_repairs),
            "official_source_match": dict(self.official_source_match),
            "official_source_files": [
                dict(item) for item in self.official_source_files
            ],
            "legacy_sample_ids": list(self.legacy_sample_ids),
            "legacy_source_rows": [dict(item) for item in self.legacy_source_rows],
            "duplicate_cluster_size": self.duplicate_cluster_size,
            "generation_user_prompt": self.generation_user_prompt,
            "generation_prompt_sha256": self.generation_prompt_sha256,
            "lineage": _lineage_flags(),
        }


class ProviderIdentityGuard:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._identity: tuple[str, str] | None = None

    @property
    def identity(self) -> tuple[str, str] | None:
        with self._lock:
            return self._identity

    def bind(self, response: TeacherResponse) -> None:
        model = response.returned_model.strip()
        fingerprint = response.system_fingerprint.strip()
        if model != TEACHER_MODEL:
            raise ModelDriftError(
                f"provider returned {model!r}, expected exactly {TEACHER_MODEL!r}"
            )
        if not fingerprint or fingerprint == "unavailable":
            raise ModelDriftError("provider system_fingerprint is unavailable")
        if not response.response_id.strip():
            raise ModelDriftError("provider response_id is unavailable")
        identity = (model, fingerprint)
        with self._lock:
            if self._identity is None:
                self._identity = identity
            elif self._identity != identity:
                raise ModelDriftError(
                    f"provider identity drift: expected={self._identity} "
                    f"observed={identity}"
                )


class OpenAICompatibleDeepSeekBackend:
    """OpenAI-compatible backend with per-thread clients and explicit retries."""

    def __init__(self) -> None:
        self._local = threading.local()

    def _client(self, *, config: TeacherConfig, api_key: str) -> Any:
        cached = getattr(self._local, "client_record", None)
        binding = (config.base_url, config.timeout_seconds, api_key)
        if cached is not None and cached[0] == binding:
            return cached[1]
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise TargetDerivedDataError("openai package is unavailable") from exc
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
        config: TeacherConfig,
        system_prompt: str,
        user_prompt: str,
        environment: Mapping[str, str] | None,
    ) -> TeacherResponse:
        env = os.environ if environment is None else environment
        api_key = str(env.get(config.api_key_env) or "").strip()
        if not api_key:
            raise TargetDerivedDataError(
                f"missing DeepSeek teacher credential: {config.api_key_env}"
            )
        client = self._client(config=config, api_key=api_key)
        attempts: list[dict[str, Any]] = []
        last_error: Exception | None = None
        for attempt_index in range(config.max_retries + 1):
            started = time.monotonic()
            started_at = _utc_now()
            try:
                request: dict[str, Any] = {
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
                    raise TargetDerivedDataError(
                        "DeepSeek teacher returned no completion choices"
                    )
                choice = completion.choices[0]
                message = choice.message
                usage = getattr(completion, "usage", None)
                usage_payload = {
                    key: _optional_int(getattr(usage, key, None))
                    for key in (
                        "prompt_tokens",
                        "completion_tokens",
                        "total_tokens",
                    )
                }
                response_id = str(getattr(completion, "id", "") or "").strip()
                returned_model = str(getattr(completion, "model", "") or "").strip()
                system_fingerprint = str(
                    getattr(completion, "system_fingerprint", "") or ""
                ).strip()
                finish_reason = str(getattr(choice, "finish_reason", "") or "").strip()
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "started_at_utc": started_at,
                        "latency_ms": round((time.monotonic() - started) * 1000, 3),
                        "status": "success",
                        "response_id": response_id,
                        "returned_model": returned_model,
                        "system_fingerprint": system_fingerprint,
                        "finish_reason": finish_reason,
                        "usage": usage_payload,
                    }
                )
                return TeacherResponse(
                    raw_reasoning=str(getattr(message, "reasoning_content", "") or ""),
                    raw_content=str(getattr(message, "content", "") or ""),
                    response_id=response_id,
                    returned_model=returned_model,
                    system_fingerprint=system_fingerprint,
                    finish_reason=finish_reason,
                    created=_optional_int(getattr(completion, "created", None)),
                    usage=usage_payload,
                    attempts=tuple(attempts),
                )
            except TargetDerivedDataError as exc:
                last_error = exc
                status_code = getattr(exc, "status_code", None)
            except Exception as exc:  # provider exception classes vary
                last_error = exc
                status_code = getattr(exc, "status_code", None)
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "started_at_utc": started_at,
                    "latency_ms": round((time.monotonic() - started) * 1000, 3),
                    "status": "error",
                    "status_code": _optional_int(status_code),
                    "error_type": type(last_error).__name__,
                    "error_message": _safe_error(last_error),
                }
            )
            permanent = status_code in {400, 401, 403, 404, 422}
            if permanent or attempt_index >= config.max_retries:
                break
            time.sleep(min(2**attempt_index, 8))
        assert last_error is not None
        raise ProviderRequestError(
            f"DeepSeek request failed: {type(last_error).__name__}: "
            f"{_safe_error(last_error)}",
            attempts=attempts,
        ) from last_error


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
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


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _safe_error(error: Any) -> str:
    rendered = _SECRET_RE.sub("[REDACTED]", str(error)).replace("\n", " ").strip()
    return rendered[:1000]


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
    _atomic_write(
        path,
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
    )


def _store_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = canonical_json(dict(payload)) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != serialized:
            raise TargetDerivedDataError(f"immutable cache collision: {path}")
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TargetDerivedDataError(f"missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TargetDerivedDataError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TargetDerivedDataError(f"{label} must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise TargetDerivedDataError(f"missing {label}: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise TargetDerivedDataError(
                    f"blank row in {label}: {path}:{line_number}"
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TargetDerivedDataError(
                    f"invalid JSON in {label}: {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise TargetDerivedDataError(
                    f"non-object row in {label}: {path}:{line_number}"
                )
            row["_source_path"] = str(path.resolve())
            row["_source_line"] = line_number
            rows.append(row)
    return rows


def _lineage_flags() -> dict[str, Any]:
    return {
        "analysis_source_lineage": "target_derived_from_official_minutes",
        "analysis_teacher_saw_official_target": True,
        "reasoning_teacher_saw_official_target": True,
        "student_prompt_has_direct_target_field": False,
        "analysis_is_reference_free": False,
        "suitable_for_leakage_safe_evaluation": False,
        "training_only": True,
        "target_is_teacher_synthetic_rewrite": False,
        "evaluation_eligible": False,
        "reported_checkpoint_eligible": False,
        "training_ready": False,
    }


def _section_category(section_name: str) -> str | None:
    normalized = " ".join(str(section_name).split())
    for category, pattern in _SECTION_RULES:
        if pattern.fullmatch(normalized):
            return category
    return None


def _source_file_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": _display_path(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _generation_user_prompt(official: str) -> str:
    payload = canonical_json({"official_minutes_paragraph": official})
    rendered = GENERATION_USER_PREFIX + payload
    round_trip = json.loads(rendered[len(GENERATION_USER_PREFIX) :])
    if round_trip != {"official_minutes_paragraph": official}:
        raise TargetDerivedDataError("generation prompt JSON boundary failed")
    return rendered


def _verification_user_prompt(row: PreparedRow, generation: Mapping[str, Any]) -> str:
    payload = {
        "official_minutes_paragraph": row.official_minutes_paragraph,
        "atomic_claim_card": generation["atomic_claim_card"],
        "analysis": generation["analysis"],
        "fidelity_reasoning": generation["fidelity_reasoning"],
    }
    return VERIFICATION_USER_PREFIX + canonical_json(payload)


def _make_sample_id(meeting_date: str, line_id: str, target_sha: str) -> str:
    safe_line_id = _SAFE_SAMPLE_COMPONENT_RE.sub("-", line_id).strip("-") or "line"
    return f"official-{meeting_date}-{safe_line_id}-{target_sha[:8]}"


def prepare_official_targets(
    *,
    source_root: str | Path,
    split_manifest_path: str | Path,
    output_root: str | Path,
    selected_splits: Sequence[str] = OUTPUT_SPLITS,
    limit: int | None = None,
) -> tuple[dict[str, list[PreparedRow]], dict[str, Any]]:
    """Prepare only official targets; never project legacy generated text."""

    source = Path(source_root).resolve()
    split_manifest = Path(split_manifest_path).resolve()
    output = Path(output_root).resolve()
    selected = tuple(dict.fromkeys(str(item) for item in selected_splits))
    invalid = sorted(set(selected) - set(OUTPUT_SPLITS))
    if invalid or not selected:
        raise TargetDerivedDataError(f"invalid selected splits: {invalid}")
    if limit is not None and limit <= 0:
        raise TargetDerivedDataError("limit must be positive")

    try:
        split_rows = json.loads(split_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetDerivedDataError(
            f"cannot read meeting split manifest: {split_manifest}: {exc}"
        ) from exc
    if not isinstance(split_rows, list) or not all(
        isinstance(item, dict) for item in split_rows
    ):
        raise TargetDerivedDataError("meeting split manifest must be an object array")
    meeting_authority: dict[str, tuple[str, int]] = {}
    for item in split_rows:
        date = str(item.get("meeting_date") or "")
        split = str(item.get("split") or "")
        count = _optional_int(item.get("sample_count"))
        if not date or split not in SOURCE_SPLITS or count is None:
            raise TargetDerivedDataError("invalid meeting split authority row")
        if date in meeting_authority:
            raise TargetDerivedDataError(f"duplicate meeting split row: {date}")
        meeting_authority[date] = (split, count)

    source_files: dict[str, dict[str, Any]] = {}
    all_source_rows: list[dict[str, Any]] = []
    observed_meeting_counts: Counter[str] = Counter()
    for output_split in selected:
        source_split = OUTPUT_TO_SOURCE[output_split]
        path = source / f"{source_split}_manifest.jsonl"
        rows = _read_jsonl(path, label=f"{source_split} legacy manifest")
        source_files[source_split] = _source_file_record(path, rows=len(rows))
        for row in rows:
            if str(row.get("split") or "") != source_split:
                raise TargetDerivedDataError(
                    f"source split mismatch: {path}:{row['_source_line']}"
                )
            meeting_date = str(row.get("meeting_date") or "")
            authority = meeting_authority.get(meeting_date)
            if authority is None or authority[0] != source_split:
                raise TargetDerivedDataError(
                    f"meeting split authority mismatch: {meeting_date}/{source_split}"
                )
            observed_meeting_counts[meeting_date] += 1
            row["_output_split"] = output_split
            all_source_rows.append(row)
    # The authority file was produced one stage earlier than the legacy
    # Minutes-alignment manifests.  Its meeting assignment is authoritative,
    # while a small number of rows were removed before the alignment manifests
    # were sealed.  Retain count differences as provenance diagnostics instead
    # of inventing or silently recovering a missing paragraph.
    meeting_count_mismatches = {
        meeting_date: {
            "alignment_manifest_rows": int(count),
            "split_authority_rows": int(meeting_authority[meeting_date][1]),
        }
        for meeting_date, count in sorted(observed_meeting_counts.items())
        if count != meeting_authority[meeting_date][1]
    }

    seen_legacy_ids: set[str] = set()
    prelim: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for source_index, row in enumerate(all_source_rows):
        required = {
            "sample_id",
            "split",
            "meeting_date",
            "source_row_index",
            "reference_excerpt",
        }
        missing = sorted(required - set(row))
        if missing:
            raise TargetDerivedDataError(
                f"source row missing {missing}: {row['_source_path']}:"
                f"{row['_source_line']}"
            )
        legacy_id = str(row["sample_id"]).strip()
        if not legacy_id or legacy_id in seen_legacy_ids:
            raise TargetDerivedDataError(
                f"duplicate/empty legacy sample_id: {legacy_id}"
            )
        seen_legacy_ids.add(legacy_id)
        meeting_date = str(row["meeting_date"]).strip()
        raw = str(row["reference_excerpt"]).strip()
        official, repairs = _normalized_official_paragraph(raw)
        target_sha = sha256_text(official)
        reasons: list[str] = []
        source_flags = sorted({str(item) for item in row.get("quality_flags", [])})
        reasons.extend(sorted(_BLOCKING_SOURCE_FLAGS & set(source_flags)))
        if not official:
            reasons.append("empty_official_target")
        words = len(official.split())
        if words < MIN_TARGET_WORDS:
            reasons.append("official_target_too_short")
        if words > MAX_TARGET_WORDS:
            reasons.append("official_target_too_long")
        if len([p for p in re.split(r"\n\s*\n", raw) if p.strip()]) != 1:
            reasons.append("official_target_not_one_source_paragraph")
        category = _section_category(str(row.get("section_name") or ""))
        if category is None:
            reasons.append("section_not_in_descriptive_allowlist")
        if (
            _ADMIN_OR_POLICY_RE.search(official)
            or _POLICY_OR_ADMIN_PARAGRAPH_RE.search(official)
            or _POLICY_DELIBERATION_RE.search(official)
            or _POLICY_ACTOR_ACTION_RE.search(official)
            or _POLICY_FRAME_RE.search(official)
            or _FORMAL_ACTION_RE.search(official)
        ):
            reasons.append("administrative_or_policy_target")
        if (
            _SECTION_HEADING_PREFIX_RE.search(official)
            or _GLUED_SECTION_HEADING_RE.search(official)
            or _GLUED_TITLE_RE.search(official)
        ):
            reasons.append("official_target_contains_section_heading")
        if any(marker in official for marker in CONTROL_MARKERS):
            reasons.append("official_target_contains_control_marker")
        source_matches = _official_source_matches(meeting_date, official)
        if len(source_matches) != 1:
            reasons.append("official_source_exact_match_count_not_one")
        source_files_for_meeting = _source_file_records(meeting_date)
        suffixes = {Path(str(item["path"])).suffix for item in source_files_for_meeting}
        if suffixes != {".csv", ".xlsx"}:
            reasons.append("official_source_pair_incomplete")

        base = {
            "legacy_sample_id": legacy_id,
            "split": str(row["_output_split"]),
            "source_split": str(row["split"]),
            "source_index": source_index,
            "meeting_date": meeting_date,
            "section_name": str(row.get("section_name") or ""),
            "section_category": category,
            "topic": str(row.get("topic") or ""),
            "source_row_index": row["source_row_index"],
            "raw": raw,
            "official": official,
            "repairs": repairs,
            "target_sha": target_sha,
            "source_match": source_matches[0] if len(source_matches) == 1 else None,
            "source_files": source_files_for_meeting,
            "source_path": _display_path(Path(str(row["_source_path"]))),
            "source_line": int(row["_source_line"]),
            "source_flags": source_flags,
            "target_words": words,
        }
        if reasons:
            unique_reasons = list(dict.fromkeys(reasons))
            rejection_counts.update(unique_reasons)
            rejections.append(
                {
                    "schema_version": PREPARED_SCHEMA_VERSION,
                    "legacy_sample_id": legacy_id,
                    "split": base["split"],
                    "meeting_date": meeting_date,
                    "official_minutes_sha256": target_sha,
                    "rejection_stage": "target_only_eligibility",
                    "rejection_reasons": unique_reasons,
                    "source_manifest": {
                        "path": base["source_path"],
                        "line": base["source_line"],
                    },
                }
            )
        else:
            prelim.append(base)

    by_target: dict[str, list[dict[str, Any]]] = {}
    for item in prelim:
        by_target.setdefault(str(item["target_sha"]), []).append(item)
    deduplicated: list[dict[str, Any]] = []
    for target_sha, cluster in sorted(by_target.items()):
        cluster.sort(
            key=lambda item: (
                str(item["meeting_date"]),
                str(item["source_match"]["line_id"]),
                str(item["legacy_sample_id"]),
            )
        )
        cluster_splits = {str(item["split"]) for item in cluster}
        if len(cluster_splits) > 1:
            for item in cluster:
                rejection_counts["cross_split_duplicate_target"] += 1
                rejections.append(
                    {
                        "schema_version": PREPARED_SCHEMA_VERSION,
                        "legacy_sample_id": item["legacy_sample_id"],
                        "split": item["split"],
                        "meeting_date": item["meeting_date"],
                        "official_minutes_sha256": target_sha,
                        "rejection_stage": "target_deduplication",
                        "rejection_reasons": ["cross_split_duplicate_target"],
                        "duplicate_legacy_sample_ids": [
                            candidate["legacy_sample_id"] for candidate in cluster
                        ],
                    }
                )
            continue
        canonical = cluster[0]
        canonical["duplicate_cluster"] = cluster
        deduplicated.append(canonical)
        for duplicate in cluster[1:]:
            rejection_counts["duplicate_official_target_same_split"] += 1
            rejections.append(
                {
                    "schema_version": PREPARED_SCHEMA_VERSION,
                    "legacy_sample_id": duplicate["legacy_sample_id"],
                    "split": duplicate["split"],
                    "meeting_date": duplicate["meeting_date"],
                    "official_minutes_sha256": target_sha,
                    "rejection_stage": "target_deduplication",
                    "rejection_reasons": ["duplicate_official_target_same_split"],
                    "canonical_legacy_sample_id": canonical["legacy_sample_id"],
                }
            )

    deduplicated.sort(
        key=lambda item: (
            OUTPUT_SPLITS.index(str(item["split"])),
            str(item["meeting_date"]),
            int(item["source_row_index"]),
            str(item["legacy_sample_id"]),
        )
    )
    population_before_limit = len(deduplicated)
    if limit is not None:
        deduplicated = deduplicated[:limit]

    prepared: dict[str, list[PreparedRow]] = {split: [] for split in selected}
    for item in deduplicated:
        source_match = dict(item["source_match"])
        line_id = str(source_match["line_id"])
        official = str(item["official"])
        prompt = _generation_user_prompt(official)
        cluster = list(item["duplicate_cluster"])
        row = PreparedRow(
            sample_id=_make_sample_id(
                str(item["meeting_date"]), line_id, str(item["target_sha"])
            ),
            split=str(item["split"]),
            source_split=str(item["source_split"]),
            source_index=int(item["source_index"]),
            meeting_date=str(item["meeting_date"]),
            section_name=str(item["section_name"]),
            section_category=str(item["section_category"]),
            topic=str(item["topic"]),
            line_id=line_id,
            official_minutes_raw=str(item["raw"]),
            official_minutes_paragraph=official,
            official_minutes_raw_sha256=sha256_text(str(item["raw"])),
            official_minutes_sha256=str(item["target_sha"]),
            official_normalization_repairs=tuple(str(x) for x in item["repairs"]),
            official_source_match=source_match,
            official_source_files=tuple(dict(x) for x in item["source_files"]),
            legacy_sample_ids=tuple(str(x["legacy_sample_id"]) for x in cluster),
            legacy_source_rows=tuple(
                {
                    "sample_id": str(x["legacy_sample_id"]),
                    "source_row_index": x["source_row_index"],
                    "manifest_path": str(x["source_path"]),
                    "manifest_line": int(x["source_line"]),
                }
                for x in cluster
            ),
            duplicate_cluster_size=len(cluster),
            generation_user_prompt=prompt,
            generation_prompt_sha256=sha256_text(prompt),
        )
        prepared[row.split].append(row)

    ids = [row.sample_id for rows in prepared.values() for row in rows]
    targets = [
        row.official_minutes_sha256 for rows in prepared.values() for row in rows
    ]
    if len(ids) != len(set(ids)):
        raise TargetDerivedDataError("prepared sample IDs are not unique")
    if len(targets) != len(set(targets)):
        raise TargetDerivedDataError("prepared target hashes are not unique")
    target_by_split = {
        split: {row.official_minutes_sha256 for row in prepared[split]}
        for split in selected
    }
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            if target_by_split[left] & target_by_split[right]:
                raise TargetDerivedDataError(
                    f"cross-split target overlap: {left}/{right}"
                )
    meeting_by_split = {
        split: {row.meeting_date for row in prepared[split]} for split in selected
    }
    for left_index, left in enumerate(selected):
        for right in selected[left_index + 1 :]:
            if meeting_by_split[left] & meeting_by_split[right]:
                raise TargetDerivedDataError(f"meeting split overlap: {left}/{right}")

    for split in selected:
        _write_jsonl(
            output / "prepared" / f"{split}.jsonl",
            [row.prepared_record() for row in prepared[split]],
        )
        split_rejections = [row for row in rejections if row["split"] == split]
        _write_jsonl(output / "rejections" / f"{split}.jsonl", split_rejections)

    summary = {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "status": "prepared",
        "source_policy": "official_target_only_legacy_generated_fields_ignored",
        "source_files": source_files,
        "meeting_split_manifest": _source_file_record(split_manifest),
        "meeting_count_mismatches": meeting_count_mismatches,
        "selected_splits": list(selected),
        "source_rows": len(all_source_rows),
        "source_unique_legacy_sample_ids": len(seen_legacy_ids),
        "eligible_unique_targets_before_limit": population_before_limit,
        "limit": limit,
        "prepared_counts": {split: len(prepared[split]) for split in selected},
        "total_prepared": len(ids),
        "rejection_rows": len(rejections),
        "rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "meeting_counts": {split: len(meeting_by_split[split]) for split in selected},
        "lineage": _lineage_flags(),
    }
    _write_json(output / "preparation_summary.json", summary)
    return prepared, summary


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
        payload = json.loads(
            str(raw),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise TargetDerivedDataError(f"content_not_strict_json:{exc}") from exc
    if duplicates:
        raise TargetDerivedDataError(
            f"content_duplicate_json_keys:{','.join(sorted(set(duplicates)))}"
        )
    if not isinstance(payload, dict):
        raise TargetDerivedDataError("content_json_must_be_object")
    return payload


def _nonempty_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TargetDerivedDataError(f"{field}_must_be_nonempty_text")
    return value.strip()


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in _SENTENCE_RE.split(text.strip()) if item.strip()]


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def _token_copy_diagnostics(analysis: str, target: str) -> dict[str, Any]:
    analysis_tokens = re.findall(r"\b\w+(?:['’.-]\w+)*\b", analysis.casefold())
    target_tokens = re.findall(r"\b\w+(?:['’.-]\w+)*\b", target.casefold())
    matcher = SequenceMatcher(a=analysis_tokens, b=target_tokens, autojunk=False)
    longest = matcher.find_longest_match().size
    return {
        "token_sequence_similarity": round(matcher.ratio(), 8),
        "longest_common_token_run": int(longest),
        "target_token_count": len(target_tokens),
        "analysis_token_count": len(analysis_tokens),
    }


def validate_generation(
    row: PreparedRow, response: TeacherResponse, *, tokenizer: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    if response.finish_reason != "stop":
        errors.append(f"finish_reason_not_stop:{response.finish_reason}")
    if not response.raw_reasoning.strip():
        errors.append("empty_provider_native_reasoning")
    try:
        payload = _strict_json_object(response.raw_content)
    except TargetDerivedDataError as exc:
        raise TargetDerivedDataError(str(exc)) from exc
    expected = {
        "atomic_claim_card",
        "analysis",
        "fidelity_reasoning",
        "claim_alignment",
    }
    if set(payload) != expected:
        errors.append(
            "generation_content_keys:" + ",".join(sorted(str(key) for key in payload))
        )
    try:
        analysis = _nonempty_text(payload.get("analysis"), field="analysis")
        reasoning = _nonempty_text(
            payload.get("fidelity_reasoning"), field="fidelity_reasoning"
        )
    except TargetDerivedDataError as exc:
        errors.append(str(exc))
        analysis = str(payload.get("analysis") or "").strip()
        reasoning = str(payload.get("fidelity_reasoning") or "").strip()
    card = payload.get("atomic_claim_card")
    alignments = payload.get("claim_alignment")
    if not isinstance(card, list) or not card:
        errors.append("atomic_claim_card_must_be_nonempty_array")
        card = []
    if not isinstance(alignments, list) or not alignments:
        errors.append("claim_alignment_must_be_nonempty_array")
        alignments = []

    claim_ids: list[str] = []
    claim_spans: list[str] = []
    normalized_card: list[dict[str, str]] = []
    for index, item in enumerate(card, start=1):
        if not isinstance(item, dict) or set(item) != {
            "claim_id",
            "target_span",
            "proposition",
        }:
            errors.append(f"atomic_claim_schema:{index}")
            continue
        try:
            claim_id = _nonempty_text(item["claim_id"], field="claim_id")
            target_span = _nonempty_text(item["target_span"], field="target_span")
            proposition = _nonempty_text(item["proposition"], field="proposition")
        except TargetDerivedDataError as exc:
            errors.append(f"atomic_claim_{index}:{exc}")
            continue
        if claim_id != f"c{index}":
            errors.append(f"nonconsecutive_claim_id:{claim_id}")
        if target_span not in row.official_minutes_paragraph:
            errors.append(f"target_span_not_exact:{claim_id}")
        claim_ids.append(claim_id)
        claim_spans.append(target_span)
        normalized_card.append(
            {
                "claim_id": claim_id,
                "target_span": target_span,
                "proposition": proposition,
            }
        )
    for sentence_index, sentence in enumerate(
        _sentences(row.official_minutes_paragraph), start=1
    ):
        if not any(span in sentence or sentence in span for span in claim_spans):
            errors.append(f"target_sentence_uncovered:{sentence_index}")

    alignment_ids: list[str] = []
    normalized_alignments: list[dict[str, str]] = []
    for index, item in enumerate(alignments, start=1):
        if not isinstance(item, dict) or set(item) != {
            "claim_id",
            "analysis_evidence",
        }:
            errors.append(f"claim_alignment_schema:{index}")
            continue
        try:
            claim_id = _nonempty_text(item["claim_id"], field="claim_id")
            evidence = _nonempty_text(
                item["analysis_evidence"], field="analysis_evidence"
            )
        except TargetDerivedDataError as exc:
            errors.append(f"claim_alignment_{index}:{exc}")
            continue
        if evidence not in analysis:
            errors.append(f"analysis_evidence_not_exact:{claim_id}")
        alignment_ids.append(claim_id)
        normalized_alignments.append(
            {"claim_id": claim_id, "analysis_evidence": evidence}
        )
    if claim_ids != alignment_ids or len(set(claim_ids)) != len(claim_ids):
        errors.append("claim_alignment_ids_do_not_match_claim_card")

    for field_name, value in (("analysis", analysis), ("reasoning", reasoning)):
        markers = [marker for marker in CONTROL_MARKERS if marker in value]
        if markers:
            errors.append(f"{field_name}_control_markers:{','.join(markers)}")
        if _MARKDOWN_OR_HEADING_RE.search(value):
            errors.append(f"{field_name}_contains_markdown_or_heading")
    analysis_words = len(analysis.split())
    reasoning_words = len(reasoning.split())
    if not MIN_ANALYSIS_WORDS <= analysis_words <= MAX_ANALYSIS_WORDS:
        errors.append(f"analysis_words:{analysis_words}")
    if not MIN_REASONING_WORDS <= reasoning_words <= MAX_REASONING_WORDS:
        errors.append(f"reasoning_words:{reasoning_words}")
    target = row.official_minutes_paragraph
    if " ".join(analysis.casefold().split()) == " ".join(target.casefold().split()):
        errors.append("analysis_exactly_copies_official_target")
    copy_diagnostics = _token_copy_diagnostics(analysis, target)
    if copy_diagnostics["token_sequence_similarity"] >= 0.90 or (
        copy_diagnostics["token_sequence_similarity"] >= 0.80
        and copy_diagnostics["longest_common_token_run"] >= 24
    ):
        errors.append("analysis_insufficiently_destylized")

    target_numbers = _numeric_values(target)
    analysis_numbers = _numeric_values(analysis)
    reasoning_numbers = _numeric_values(reasoning)
    target_dates = _date_values(target)
    analysis_dates = _date_values(analysis)
    reasoning_dates = _date_values(reasoning)
    target_attributions = _attribution_categories(target)
    analysis_attributions = _attribution_categories(analysis)
    reasoning_attributions = _attribution_categories(reasoning)
    if analysis_numbers != target_numbers:
        errors.append("analysis_numeric_multiset_mismatch")
    if reasoning_numbers != target_numbers:
        errors.append("reasoning_numeric_multiset_mismatch")
    if analysis_dates != target_dates:
        errors.append("analysis_date_set_mismatch")
    if reasoning_dates != target_dates:
        errors.append("reasoning_date_set_mismatch")
    if analysis_attributions != target_attributions:
        errors.append("analysis_attribution_set_mismatch")
    if reasoning_attributions != target_attributions:
        errors.append("reasoning_attribution_set_mismatch")
    reasoning_meta = sorted(_reasoning_meta_categories(reasoning))
    if reasoning_meta:
        errors.append("reasoning_transport_meta:" + ",".join(reasoning_meta))

    prompt = ""
    completion = ""
    prompt_tokens = completion_tokens = total_tokens = 0
    try:
        prompt = render_user_prompt(analysis)
        completion = build_sft_completion(reasoning, target)
        prompt_tokens, completion_tokens, total_tokens = _count_sft_row(
            {"prompt": prompt, "response": completion},
            tokenizer=tokenizer,
            config={
                "system_prompt": STUDENT_SYSTEM_PROMPT,
                "dataset_prompt_column": "prompt",
                "chat_template_kwargs": {},
            },
        )
    except Exception as exc:  # row-level deterministic rejection
        errors.append(f"serialization_or_tokenization:{type(exc).__name__}:{exc}")
    if prompt_tokens > PROMPT_TOKEN_LIMIT:
        errors.append(f"prompt_token_overflow:{prompt_tokens}")
    if total_tokens > TOTAL_TOKEN_LIMIT:
        errors.append(f"total_token_overflow:{total_tokens}")
    if completion and completion.split("\n</think>\n", 1)[1] != target:
        errors.append("official_target_not_literal_completion_suffix")

    diagnostics = {
        "analysis_words": analysis_words,
        "reasoning_words": reasoning_words,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "target_numbers": _counter_dict(target_numbers),
        "analysis_numbers": _counter_dict(analysis_numbers),
        "reasoning_numbers": _counter_dict(reasoning_numbers),
        "target_dates": sorted(target_dates),
        "analysis_dates": sorted(analysis_dates),
        "reasoning_dates": sorted(reasoning_dates),
        "target_attributions": sorted(target_attributions),
        "analysis_attributions": sorted(analysis_attributions),
        "reasoning_attributions": sorted(reasoning_attributions),
        "reasoning_meta_categories": reasoning_meta,
        **copy_diagnostics,
    }
    if errors:
        raise TargetDerivedDataError(";".join(dict.fromkeys(errors)))
    return (
        {
            "atomic_claim_card": normalized_card,
            "analysis": analysis,
            "fidelity_reasoning": reasoning,
            "claim_alignment": normalized_alignments,
            "student_prompt": prompt,
            "sft_completion": completion,
        },
        diagnostics,
    )


def validate_verification(
    row: PreparedRow,
    generation: Mapping[str, Any],
    response: TeacherResponse,
) -> tuple[dict[str, Any], bool, list[str]]:
    errors: list[str] = []
    if response.finish_reason != "stop":
        errors.append(f"finish_reason_not_stop:{response.finish_reason}")
    if not response.raw_reasoning.strip():
        errors.append("empty_provider_native_reasoning")
    payload = _strict_json_object(response.raw_content)
    expected = {
        "target_claims",
        "analysis_claims",
        "reasoning_compatible",
        "bidirectional_entailment",
        "issues",
        "overall_pass",
    }
    if set(payload) != expected:
        errors.append(
            "verification_content_keys:" + ",".join(sorted(str(key) for key in payload))
        )
    target_claims = payload.get("target_claims")
    analysis_claims = payload.get("analysis_claims")
    issues = payload.get("issues")
    if not isinstance(target_claims, list) or not target_claims:
        errors.append("target_claims_must_be_nonempty_array")
        target_claims = []
    if not isinstance(analysis_claims, list) or not analysis_claims:
        errors.append("analysis_claims_must_be_nonempty_array")
        analysis_claims = []
    if not isinstance(issues, list) or not all(
        isinstance(item, str) for item in issues
    ):
        errors.append("issues_must_be_string_array")
        issues = ["invalid issues field"]
    booleans: dict[str, bool] = {}
    for key in ("reasoning_compatible", "bidirectional_entailment", "overall_pass"):
        value = payload.get(key)
        if not isinstance(value, bool):
            errors.append(f"{key}_must_be_boolean")
            value = False
        booleans[key] = value

    normalized_target_claims: list[dict[str, str]] = []
    target_spans: list[str] = []
    target_verdicts: list[str] = []
    for index, item in enumerate(target_claims, start=1):
        if not isinstance(item, dict) or set(item) != {
            "target_span",
            "analysis_evidence",
            "verdict",
        }:
            errors.append(f"target_claim_schema:{index}")
            continue
        try:
            target_span = _nonempty_text(item["target_span"], field="target_span")
            evidence = _nonempty_text(
                item["analysis_evidence"], field="analysis_evidence"
            )
            verdict = _nonempty_text(item["verdict"], field="verdict")
        except TargetDerivedDataError as exc:
            errors.append(f"target_claim_{index}:{exc}")
            continue
        if target_span not in row.official_minutes_paragraph:
            errors.append(f"verification_target_span_not_exact:{index}")
        if evidence not in str(generation["analysis"]):
            errors.append(f"verification_analysis_evidence_not_exact:{index}")
        if verdict not in {"entailed", "unsupported", "contradicted"}:
            errors.append(f"target_claim_verdict_invalid:{index}")
        target_spans.append(target_span)
        target_verdicts.append(verdict)
        normalized_target_claims.append(
            {
                "target_span": target_span,
                "analysis_evidence": evidence,
                "verdict": verdict,
            }
        )
    for sentence_index, sentence in enumerate(
        _sentences(row.official_minutes_paragraph), start=1
    ):
        if not any(span in sentence or sentence in span for span in target_spans):
            errors.append(f"verification_target_sentence_uncovered:{sentence_index}")

    normalized_analysis_claims: list[dict[str, str]] = []
    analysis_spans: list[str] = []
    analysis_verdicts: list[str] = []
    for index, item in enumerate(analysis_claims, start=1):
        if not isinstance(item, dict) or set(item) != {
            "analysis_span",
            "target_evidence",
            "verdict",
        }:
            errors.append(f"analysis_claim_schema:{index}")
            continue
        try:
            analysis_span = _nonempty_text(item["analysis_span"], field="analysis_span")
            evidence = _nonempty_text(item["target_evidence"], field="target_evidence")
            verdict = _nonempty_text(item["verdict"], field="verdict")
        except TargetDerivedDataError as exc:
            errors.append(f"analysis_claim_{index}:{exc}")
            continue
        if analysis_span not in str(generation["analysis"]):
            errors.append(f"verification_analysis_span_not_exact:{index}")
        if evidence not in row.official_minutes_paragraph:
            errors.append(f"verification_target_evidence_not_exact:{index}")
        if verdict not in {"supported", "unsupported"}:
            errors.append(f"analysis_claim_verdict_invalid:{index}")
        analysis_spans.append(analysis_span)
        analysis_verdicts.append(verdict)
        normalized_analysis_claims.append(
            {
                "analysis_span": analysis_span,
                "target_evidence": evidence,
                "verdict": verdict,
            }
        )
    for sentence_index, sentence in enumerate(
        _sentences(str(generation["analysis"])), start=1
    ):
        if not any(span in sentence or sentence in span for span in analysis_spans):
            errors.append(f"verification_analysis_sentence_uncovered:{sentence_index}")

    machine_pass = (
        not errors
        and bool(target_verdicts)
        and all(item == "entailed" for item in target_verdicts)
        and bool(analysis_verdicts)
        and all(item == "supported" for item in analysis_verdicts)
        and booleans["reasoning_compatible"]
        and booleans["bidirectional_entailment"]
        and not issues
        and booleans["overall_pass"]
    )
    normalized = {
        "target_claims": normalized_target_claims,
        "analysis_claims": normalized_analysis_claims,
        "reasoning_compatible": booleans["reasoning_compatible"],
        "bidirectional_entailment": booleans["bidirectional_entailment"],
        "issues": [str(item) for item in issues],
        "overall_pass": booleans["overall_pass"],
        "machine_pass": machine_pass,
    }
    rejection_reasons = list(dict.fromkeys(errors))
    if not machine_pass and not rejection_reasons:
        if any(item != "entailed" for item in target_verdicts):
            rejection_reasons.append("verifier_target_claim_not_entailed")
        if any(item != "supported" for item in analysis_verdicts):
            rejection_reasons.append("verifier_analysis_claim_not_supported")
        if not booleans["reasoning_compatible"]:
            rejection_reasons.append("verifier_reasoning_incompatible")
        if not booleans["bidirectional_entailment"]:
            rejection_reasons.append("verifier_not_bidirectionally_entailed")
        if issues:
            rejection_reasons.append("verifier_reported_issues")
        if not booleans["overall_pass"]:
            rejection_reasons.append("verifier_overall_fail")
    return normalized, machine_pass, rejection_reasons


def prompt_contract(*, config: TeacherConfig, code_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": PROMPT_CONTRACT_SCHEMA_VERSION,
        "code_sha256": code_sha256,
        "teacher_contract": config.contract(),
        "teacher_contract_sha256": config.contract_sha256,
        "generation_system_prompt": GENERATION_SYSTEM_PROMPT,
        "generation_system_prompt_sha256": sha256_text(GENERATION_SYSTEM_PROMPT),
        "verification_system_prompt": VERIFICATION_SYSTEM_PROMPT,
        "verification_system_prompt_sha256": sha256_text(VERIFICATION_SYSTEM_PROMPT),
        "student_system_prompt": STUDENT_SYSTEM_PROMPT,
        "student_system_prompt_sha256": sha256_text(STUDENT_SYSTEM_PROMPT),
        "mapping": {
            "source": "literal official Minutes paragraph only",
            "analysis": "generation content.analysis",
            "reasoning": "generation content.fidelity_reasoning",
            "provider_native_reasoning": "audit_only_not_supervision",
            "student_prompt": "render_user_prompt(generated analysis)",
            "completion": (
                "generated fidelity_reasoning + '\\n</think>\\n' + "
                "literal official Minutes paragraph"
            ),
        },
        "quality_policy": {
            "two_separate_provider_calls": True,
            "bidirectional_entailment_required": True,
            "numeric_multisets_equal": True,
            "calendar_sets_equal": True,
            "attribution_sets_equal": True,
            "exact_evidence_spans_required": True,
            "provider_identity_fixed_within_release": True,
            "human_review_required": True,
            "training_ready": False,
        },
        "lineage": _lineage_flags(),
    }


def _cache_path(output_root: Path, phase: str, row: PreparedRow) -> Path:
    return output_root / "cache" / phase / f"{sha256_text(row.sample_id)}.json"


def _quarantine_preflight_caches(
    rows: Sequence[PreparedRow],
    *,
    output_root: Path,
    phase: str,
) -> list[dict[str, Any]]:
    """Preserve rejected canary responses without poisoning resume caches."""
    sources = [
        (row, _cache_path(output_root, phase, row))
        for row in rows
        if _cache_path(output_root, phase, row).is_file()
    ]
    if not sources:
        return []
    quarantine_root = output_root / "diagnostics" / "preflight_rejected"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    bundle = Path(tempfile.mkdtemp(prefix=f"{timestamp}_{phase}_", dir=quarantine_root))
    records: list[dict[str, Any]] = []
    for row, source in sources:
        destination = bundle / source.name
        cache_sha256 = sha256_file(source)
        os.replace(source, destination)
        records.append(
            {
                "sample_id": row.sample_id,
                "phase": phase,
                "path": _display_path(destination),
                "sha256": cache_sha256,
            }
        )
    return records


def _generation_binding(
    row: PreparedRow, *, config: TeacherConfig, code_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": GENERATION_CACHE_SCHEMA_VERSION,
        "phase": "generation",
        "sample_id": row.sample_id,
        "split": row.split,
        "official_minutes_sha256": row.official_minutes_sha256,
        "prompt_sha256": row.generation_prompt_sha256,
        "system_prompt_sha256": sha256_text(GENERATION_SYSTEM_PROMPT),
        "teacher_contract_sha256": config.contract_sha256,
        "code_sha256": code_sha256,
    }


def _verification_binding(
    row: PreparedRow,
    *,
    generation: Mapping[str, Any],
    verification_prompt: str,
    config: TeacherConfig,
    code_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": VERIFICATION_CACHE_SCHEMA_VERSION,
        "phase": "verification",
        "sample_id": row.sample_id,
        "split": row.split,
        "official_minutes_sha256": row.official_minutes_sha256,
        "generation_sha256": sha256_text(canonical_json(dict(generation))),
        "prompt_sha256": sha256_text(verification_prompt),
        "system_prompt_sha256": sha256_text(VERIFICATION_SYSTEM_PROMPT),
        "teacher_contract_sha256": config.contract_sha256,
        "code_sha256": code_sha256,
    }


def _binding_sha(binding: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(dict(binding)))


def _cache_payload(
    *, binding: Mapping[str, Any], response: TeacherResponse
) -> dict[str, Any]:
    return {
        "binding": dict(binding),
        "binding_sha256": _binding_sha(binding),
        "provider_response": response.to_dict(),
        "raw_content_sha256": sha256_text(response.raw_content),
        "raw_reasoning_sha256": sha256_text(response.raw_reasoning),
    }


def _load_cache(
    path: Path,
    *,
    binding: Mapping[str, Any],
    identity_guard: ProviderIdentityGuard,
) -> tuple[dict[str, Any], TeacherResponse]:
    payload = _read_json(path, label="teacher cache")
    if payload.get("binding") != dict(binding):
        raise TargetDerivedDataError(f"cache binding mismatch: {path}")
    if payload.get("binding_sha256") != _binding_sha(binding):
        raise TargetDerivedDataError(f"cache binding SHA mismatch: {path}")
    provider = payload.get("provider_response")
    if not isinstance(provider, dict):
        raise TargetDerivedDataError(f"cache provider response invalid: {path}")
    response = TeacherResponse.from_dict(provider)
    if payload.get("raw_content_sha256") != sha256_text(response.raw_content):
        raise TargetDerivedDataError(f"cache content SHA mismatch: {path}")
    if payload.get("raw_reasoning_sha256") != sha256_text(response.raw_reasoning):
        raise TargetDerivedDataError(f"cache reasoning SHA mismatch: {path}")
    identity_guard.bind(response)
    return payload, response


def _flatten(prepared: Mapping[str, Sequence[PreparedRow]]) -> list[PreparedRow]:
    rows: list[PreparedRow] = []
    for split in OUTPUT_SPLITS:
        rows.extend(prepared.get(split, ()))
    return rows


def _spread_indices(size: int) -> list[int]:
    """Return deterministic low/high/midpoint indices for broad length coverage."""
    if size <= 0:
        return []
    fractions = [0.0, 1.0]
    level = 1
    while len(fractions) < size + 1:
        denominator = 2**level
        fractions.extend(
            numerator / denominator for numerator in range(1, denominator, 2)
        )
        level += 1
    indices: list[int] = []
    for fraction in fractions:
        index = round(fraction * (size - 1))
        if index not in indices:
            indices.append(index)
        if len(indices) == size:
            break
    indices.extend(index for index in range(size) if index not in indices)
    return indices


def _select_preflight_rows(
    rows: Sequence[PreparedRow], count: int
) -> list[PreparedRow]:
    """Cover splits and paragraph-length quantiles in a deterministic canary."""
    grouped: dict[str, list[PreparedRow]] = {split: [] for split in OUTPUT_SPLITS}
    for row in rows:
        grouped.setdefault(row.split, []).append(row)
    queues: dict[str, list[PreparedRow]] = {}
    for split in OUTPUT_SPLITS:
        ordered = sorted(
            grouped.get(split, []),
            key=lambda row: (
                len(row.official_minutes_paragraph.split()),
                row.meeting_date,
                row.sample_id,
            ),
        )
        queues[split] = [ordered[index] for index in _spread_indices(len(ordered))]
    selected: list[PreparedRow] = []
    cursor = Counter()
    while len(selected) < min(count, len(rows)):
        progressed = False
        for split in OUTPUT_SPLITS:
            index = cursor[split]
            if index >= len(queues[split]):
                continue
            selected.append(queues[split][index])
            cursor[split] += 1
            progressed = True
            if len(selected) >= min(count, len(rows)):
                break
        if not progressed:
            break
    return selected


def _run_concurrent_wave(
    tasks: Sequence[tuple[PreparedRow, str, str, Mapping[str, Any]]],
    *,
    phase: str,
    output_root: Path,
    config: TeacherConfig,
    backend: TeacherBackend,
    identity_guard: ProviderIdentityGuard,
    environment: Mapping[str, str] | None,
    concurrency: int,
) -> tuple[dict[str, tuple[dict[str, Any], TeacherResponse]], list[dict[str, Any]]]:
    results: dict[str, tuple[dict[str, Any], TeacherResponse]] = {}
    failures: list[dict[str, Any]] = []
    stop_event = threading.Event()

    def worker(
        task: tuple[PreparedRow, str, str, Mapping[str, Any]],
    ) -> tuple[dict[str, Any], TeacherResponse]:
        row, system_prompt, user_prompt, binding = task
        if stop_event.is_set():
            raise ModelDriftError("not attempted after provider identity drift")
        response = backend.generate(
            config=config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            environment=environment,
        )
        try:
            identity_guard.bind(response)
        except ModelDriftError as exc:
            # The provider call has already been billed. Carry its audited
            # attempt/usage evidence to the failure ledger even though the
            # response must never enter the canonical release cache.
            setattr(exc, "provider_response", response)
            raise
        payload = _cache_payload(binding=binding, response=response)
        _store_immutable_json(_cache_path(output_root, phase, row), payload)
        return payload, response

    with ThreadPoolExecutor(
        max_workers=concurrency, thread_name_prefix=f"chk2-{phase}-v4pro"
    ) as executor:
        future_rows: dict[
            Future[tuple[dict[str, Any], TeacherResponse]], PreparedRow
        ] = {executor.submit(worker, task): task[0] for task in tasks}
        for future in as_completed(future_rows):
            row = future_rows[future]
            try:
                results[row.sample_id] = future.result()
                status = "cached"
            except ModelDriftError as exc:
                stop_event.set()
                drift_response = getattr(exc, "provider_response", None)
                attempts = (
                    [dict(item) for item in drift_response.attempts]
                    if isinstance(drift_response, TeacherResponse)
                    else []
                )
                failures.append(
                    {
                        "sample_id": row.sample_id,
                        "split": row.split,
                        "phase": phase,
                        "error_type": type(exc).__name__,
                        "error": _safe_error(exc),
                        "attempts": attempts,
                    }
                )
                status = "model_drift"
            except ProviderRequestError as exc:
                failures.append(
                    {
                        "sample_id": row.sample_id,
                        "split": row.split,
                        "phase": phase,
                        "error_type": type(exc).__name__,
                        "error": _safe_error(exc),
                        "attempts": [dict(item) for item in exc.attempts],
                    }
                )
                status = "request_failed"
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "sample_id": row.sample_id,
                        "split": row.split,
                        "phase": phase,
                        "error_type": type(exc).__name__,
                        "error": _safe_error(exc),
                        "attempts": [],
                    }
                )
                status = "failed"
            print(
                f"[{phase}] {len(results) + len(failures)}/{len(tasks)} "
                f"{row.sample_id} {status}",
                file=sys.stderr,
                flush=True,
            )
    return results, failures


def _load_or_generate_wave(
    rows: Sequence[PreparedRow],
    *,
    output_root: Path,
    config: TeacherConfig,
    backend: TeacherBackend,
    identity_guard: ProviderIdentityGuard,
    environment: Mapping[str, str] | None,
    concurrency: int,
    resume: bool,
    code_sha256: str,
) -> tuple[dict[str, tuple[dict[str, Any], TeacherResponse]], list[dict[str, Any]]]:
    loaded: dict[str, tuple[dict[str, Any], TeacherResponse]] = {}
    tasks: list[tuple[PreparedRow, str, str, Mapping[str, Any]]] = []
    for row in rows:
        binding = _generation_binding(row, config=config, code_sha256=code_sha256)
        path = _cache_path(output_root, "generation", row)
        if path.exists():
            if not resume:
                raise TargetDerivedDataError(
                    f"generation cache exists; pass --resume: {path}"
                )
            loaded[row.sample_id] = _load_cache(
                path, binding=binding, identity_guard=identity_guard
            )
        else:
            tasks.append(
                (row, GENERATION_SYSTEM_PROMPT, row.generation_user_prompt, binding)
            )
    generated, failures = (
        _run_concurrent_wave(
            tasks,
            phase="generation",
            output_root=output_root,
            config=config,
            backend=backend,
            identity_guard=identity_guard,
            environment=environment,
            concurrency=concurrency,
        )
        if tasks
        else ({}, [])
    )
    loaded.update(generated)
    return loaded, failures


def _load_generation_caches(
    rows: Sequence[PreparedRow],
    *,
    output_root: Path,
    config: TeacherConfig,
    identity_guard: ProviderIdentityGuard,
    code_sha256: str,
) -> dict[str, tuple[dict[str, Any], TeacherResponse]]:
    loaded: dict[str, tuple[dict[str, Any], TeacherResponse]] = {}
    for row in rows:
        binding = _generation_binding(row, config=config, code_sha256=code_sha256)
        path = _cache_path(output_root, "generation", row)
        if not path.is_file():
            continue
        loaded[row.sample_id] = _load_cache(
            path, binding=binding, identity_guard=identity_guard
        )
    return loaded


def _load_or_verify_wave(
    rows_and_generation: Sequence[tuple[PreparedRow, Mapping[str, Any]]],
    *,
    output_root: Path,
    config: TeacherConfig,
    backend: TeacherBackend,
    identity_guard: ProviderIdentityGuard,
    environment: Mapping[str, str] | None,
    concurrency: int,
    resume: bool,
    code_sha256: str,
) -> tuple[dict[str, tuple[dict[str, Any], TeacherResponse]], list[dict[str, Any]]]:
    loaded: dict[str, tuple[dict[str, Any], TeacherResponse]] = {}
    tasks: list[tuple[PreparedRow, str, str, Mapping[str, Any]]] = []
    for row, generation in rows_and_generation:
        user_prompt = _verification_user_prompt(row, generation)
        binding = _verification_binding(
            row,
            generation=generation,
            verification_prompt=user_prompt,
            config=config,
            code_sha256=code_sha256,
        )
        path = _cache_path(output_root, "verification", row)
        if path.exists():
            if not resume:
                raise TargetDerivedDataError(
                    f"verification cache exists; pass --resume: {path}"
                )
            loaded[row.sample_id] = _load_cache(
                path, binding=binding, identity_guard=identity_guard
            )
        else:
            tasks.append((row, VERIFICATION_SYSTEM_PROMPT, user_prompt, binding))
    generated, failures = (
        _run_concurrent_wave(
            tasks,
            phase="verification",
            output_root=output_root,
            config=config,
            backend=backend,
            identity_guard=identity_guard,
            environment=environment,
            concurrency=concurrency,
        )
        if tasks
        else ({}, [])
    )
    loaded.update(generated)
    return loaded, failures


def _attempt_usage(
    caches: Sequence[tuple[dict[str, Any], TeacherResponse]],
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attempts: list[Mapping[str, Any]] = []
    for _, response in caches:
        attempts.extend(response.attempts)
    for failure in failures:
        value = failure.get("attempts")
        if isinstance(value, list):
            attempts.extend(item for item in value if isinstance(item, dict))
    usage = Counter()
    successful = 0
    errored = 0
    for attempt in attempts:
        if attempt.get("status") == "success":
            successful += 1
            attempt_usage = attempt.get("usage")
            if isinstance(attempt_usage, dict):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = _optional_int(attempt_usage.get(key))
                    if value is not None:
                        usage[key] += value
        else:
            errored += 1
    return {
        "actual_request_attempts": len(attempts),
        "successful_requests": successful,
        "failed_request_attempts": errored,
        "token_usage": {key: int(usage[key]) for key in sorted(usage)},
        "cost_usd": None,
        "cost_note": "provider pricing not assumed; use recorded token usage",
    }


def _release_fatal_failures(
    failures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    permanent_statuses = {400, 401, 403, 404, 422}
    fatal: list[dict[str, Any]] = []
    for failure in failures:
        is_fatal = failure.get("error_type") == "ModelDriftError"
        attempts = failure.get("attempts")
        if isinstance(attempts, list):
            is_fatal = is_fatal or any(
                _optional_int(attempt.get("status_code")) in permanent_statuses
                for attempt in attempts
                if isinstance(attempt, dict)
            )
        if is_fatal:
            fatal.append(dict(failure))
    return fatal


def _persist_acquisition_abort(
    *,
    output_root: Path,
    reason: str,
    failures: Sequence[Mapping[str, Any]],
    generation_caches: Mapping[str, tuple[dict[str, Any], TeacherResponse]],
    verification_caches: Mapping[str, tuple[dict[str, Any], TeacherResponse]],
    identity: tuple[str, str] | None,
) -> None:
    ordered_failures = sorted(
        (dict(item) for item in failures),
        key=lambda item: (
            str(item.get("phase")),
            str(item.get("split")),
            str(item.get("sample_id")),
        ),
    )
    _write_jsonl(output_root / "failures.jsonl", ordered_failures)
    caches = list(generation_caches.values()) + list(verification_caches.values())
    _write_json(
        output_root / "acquisition_abort.json",
        {
            "schema_version": "chk2-target-derived-acquisition-abort-v1",
            "status": "aborted",
            "reason": reason,
            "aborted_at_utc": _utc_now(),
            "failure_count": len(ordered_failures),
            "failures_path": "failures.jsonl",
            "usage": _attempt_usage(caches, ordered_failures),
            "provider_identity": (
                {
                    "returned_model": identity[0],
                    "system_fingerprint": identity[1],
                }
                if identity
                else None
            ),
            "training_ready": False,
        },
    )


def _materialize(
    prepared: Mapping[str, Sequence[PreparedRow]],
    *,
    output_root: Path,
    generation_caches: Mapping[str, tuple[dict[str, Any], TeacherResponse]],
    verification_caches: Mapping[str, tuple[dict[str, Any], TeacherResponse]],
    tokenizer: Any,
    phase: str,
    failures: Sequence[Mapping[str, Any]],
    identity: tuple[str, str] | None,
    preparation: Mapping[str, Any],
    acquisition_mode: str,
) -> dict[str, Any]:
    rows_by_id = {row.sample_id: row for row in _flatten(prepared)}
    parsed_generations: dict[str, dict[str, Any]] = {}
    generation_diagnostics: dict[str, dict[str, Any]] = {}
    generation_rejections: dict[str, list[str]] = {}
    generation_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in prepared}
    for sample_id, (cache, response) in generation_caches.items():
        row = rows_by_id[sample_id]
        rejection_reasons: list[str] = []
        normalized: dict[str, Any] | None = None
        diagnostics: dict[str, Any] | None = None
        try:
            normalized, diagnostics = validate_generation(
                row, response, tokenizer=tokenizer
            )
            parsed_generations[sample_id] = normalized
            generation_diagnostics[sample_id] = diagnostics
        except Exception as exc:  # row-level fail-closed gate
            rejection_reasons = [item for item in str(exc).split(";") if item]
            generation_rejections[sample_id] = rejection_reasons
        generation_rows[row.split].append(
            {
                "schema_version": GENERATION_CACHE_SCHEMA_VERSION,
                "sample_id": sample_id,
                "split": row.split,
                "official_minutes_sha256": row.official_minutes_sha256,
                "binding_sha256": cache["binding_sha256"],
                "response_id": response.response_id,
                "returned_model": response.returned_model,
                "system_fingerprint": response.system_fingerprint,
                "finish_reason": response.finish_reason,
                "created": response.created,
                "usage": dict(response.usage),
                "raw_content": response.raw_content,
                "raw_reasoning": response.raw_reasoning,
                "raw_content_sha256": cache["raw_content_sha256"],
                "raw_reasoning_sha256": cache["raw_reasoning_sha256"],
                "parsed_generation": normalized,
                "deterministic_gate_pass": normalized is not None,
                "deterministic_gate_rejection_reasons": rejection_reasons,
                "diagnostics": diagnostics,
            }
        )

    verifier_results: dict[str, tuple[dict[str, Any], bool, list[str]]] = {}
    verification_rows: dict[str, list[dict[str, Any]]] = {
        split: [] for split in prepared
    }
    for sample_id, (cache, response) in verification_caches.items():
        row = rows_by_id[sample_id]
        generation = parsed_generations.get(sample_id)
        if generation is None:
            continue
        try:
            normalized, machine_pass, reasons = validate_verification(
                row, generation, response
            )
        except Exception as exc:
            normalized = None
            machine_pass = False
            reasons = [item for item in str(exc).split(";") if item]
        if normalized is not None:
            verifier_results[sample_id] = (normalized, machine_pass, reasons)
        verification_rows[row.split].append(
            {
                "schema_version": VERIFICATION_CACHE_SCHEMA_VERSION,
                "sample_id": sample_id,
                "split": row.split,
                "official_minutes_sha256": row.official_minutes_sha256,
                "binding_sha256": cache["binding_sha256"],
                "response_id": response.response_id,
                "returned_model": response.returned_model,
                "system_fingerprint": response.system_fingerprint,
                "finish_reason": response.finish_reason,
                "created": response.created,
                "usage": dict(response.usage),
                "raw_content": response.raw_content,
                "raw_reasoning": response.raw_reasoning,
                "raw_content_sha256": cache["raw_content_sha256"],
                "raw_reasoning_sha256": cache["raw_reasoning_sha256"],
                "parsed_verification": normalized,
                "machine_pass": machine_pass,
                "rejection_reasons": reasons,
            }
        )

    split_counts: dict[str, dict[str, int]] = {}
    for split in prepared:
        candidate_rows: list[dict[str, str]] = []
        manifest_rows: list[dict[str, Any]] = []
        rejection_rows: list[dict[str, Any]] = []
        for row in prepared[split]:
            generation = parsed_generations.get(row.sample_id)
            verifier = verifier_results.get(row.sample_id)
            machine_pass = bool(verifier and verifier[1])
            if generation is None:
                reasons = generation_rejections.get(
                    row.sample_id, ["generation_missing_or_failed"]
                )
            elif phase == "generate":
                reasons = ["verification_not_run"]
            elif verifier is None:
                reasons = ["verification_missing_or_failed"]
            else:
                reasons = verifier[2]
            if machine_pass and generation is not None:
                candidate_rows.append(
                    {
                        "prompt": str(generation["student_prompt"]),
                        "response": str(generation["sft_completion"]),
                    }
                )
                manifest_rows.append(
                    {
                        "schema_version": MANIFEST_SCHEMA_VERSION,
                        "sample_id": row.sample_id,
                        "split": row.split,
                        "split_role": (
                            "target_derived_training"
                            if row.split == "train"
                            else "target_derived_monitoring_quarantine"
                        ),
                        "meeting_date": row.meeting_date,
                        "line_id": row.line_id,
                        "section_name": row.section_name,
                        "section_category": row.section_category,
                        "topic": row.topic,
                        "analysis": generation["analysis"],
                        "reasoning": generation["fidelity_reasoning"],
                        "atomic_claim_card": generation["atomic_claim_card"],
                        "claim_alignment": generation["claim_alignment"],
                        "official_minutes_raw": row.official_minutes_raw,
                        "official_minutes_paragraph": row.official_minutes_paragraph,
                        "official_minutes_raw_sha256": row.official_minutes_raw_sha256,
                        "official_minutes_sha256": row.official_minutes_sha256,
                        "official_minutes_normalization_repairs": list(
                            row.official_normalization_repairs
                        ),
                        "official_source_match": dict(row.official_source_match),
                        "official_source_files": [
                            dict(item) for item in row.official_source_files
                        ],
                        "legacy_sample_ids": list(row.legacy_sample_ids),
                        "legacy_source_rows": [
                            dict(item) for item in row.legacy_source_rows
                        ],
                        "generation_cache_sha256": generation_caches[row.sample_id][0][
                            "binding_sha256"
                        ],
                        "verification_cache_sha256": verification_caches[row.sample_id][
                            0
                        ]["binding_sha256"],
                        "teacher": {
                            "provider": "deepseek",
                            "model": TEACHER_MODEL,
                            "system_fingerprint": identity[1] if identity else None,
                            "generation_response_id": generation_caches[row.sample_id][
                                1
                            ].response_id,
                            "verification_response_id": verification_caches[
                                row.sample_id
                            ][1].response_id,
                            "temperature": None,
                            "seed": None,
                        },
                        "generation_diagnostics": generation_diagnostics[row.sample_id],
                        "verification": verifier[0],
                        "machine_pass": True,
                        "human_review_status": "pending",
                        "prompt_sha256": sha256_text(str(generation["student_prompt"])),
                        "reasoning_sha256": sha256_text(
                            str(generation["fidelity_reasoning"])
                        ),
                        "response_sha256": sha256_text(
                            str(generation["sft_completion"])
                        ),
                        **_lineage_flags(),
                    }
                )
            else:
                rejection_rows.append(
                    {
                        "schema_version": MANIFEST_SCHEMA_VERSION,
                        "sample_id": row.sample_id,
                        "split": row.split,
                        "meeting_date": row.meeting_date,
                        "official_minutes_sha256": row.official_minutes_sha256,
                        "rejection_stage": (
                            "generation_gate"
                            if generation is None
                            else "verification_gate"
                        ),
                        "rejection_reasons": reasons,
                        "training_ready": False,
                    }
                )
        _write_jsonl(
            output_root / "teacher_generations" / f"{split}.jsonl",
            sorted(generation_rows[split], key=lambda item: item["sample_id"]),
        )
        _write_jsonl(
            output_root / "teacher_verifications" / f"{split}.jsonl",
            sorted(verification_rows[split], key=lambda item: item["sample_id"]),
        )
        _write_jsonl(output_root / "sft_candidate" / f"{split}.jsonl", candidate_rows)
        _write_jsonl(output_root / "manifests" / f"{split}.jsonl", manifest_rows)
        _write_jsonl(
            output_root / "machine_rejections" / f"{split}.jsonl", rejection_rows
        )
        split_counts[split] = {
            "prepared": len(prepared[split]),
            "generation_responses": len(generation_rows[split]),
            "generation_gate_pass": sum(
                row.sample_id in parsed_generations for row in prepared[split]
            ),
            "verification_responses": len(verification_rows[split]),
            "machine_pass": len(candidate_rows),
            "machine_rejected_or_pending": len(prepared[split]) - len(candidate_rows),
        }

    ordered_failures = sorted(
        (dict(item) for item in failures),
        key=lambda item: (
            OUTPUT_SPLITS.index(str(item.get("split")))
            if item.get("split") in OUTPUT_SPLITS
            else 99,
            str(item.get("sample_id")),
            str(item.get("phase")),
        ),
    )
    _write_jsonl(output_root / "failures.jsonl", ordered_failures)
    all_cache_values = list(generation_caches.values()) + list(
        verification_caches.values()
    )
    usage = _attempt_usage(all_cache_values, ordered_failures)
    total_prepared = sum(len(rows) for rows in prepared.values())
    total_machine_pass = sum(value["machine_pass"] for value in split_counts.values())
    acquisition_complete = (
        not ordered_failures
        and len(generation_caches) == total_prepared
        and (phase == "generate" or len(verification_caches) == len(parsed_generations))
    )
    if phase == "generate" and acquisition_complete:
        status = "generation_complete_verification_pending"
    elif acquisition_complete:
        status = "machine_screen_complete_human_review_pending"
    else:
        status = "incomplete"
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": status,
        "phase": phase,
        "acquisition_mode": acquisition_mode,
        "content_quality_gates_blocked_acquisition": False,
        "teacher_model": TEACHER_MODEL,
        "provider_identity": (
            {"returned_model": identity[0], "system_fingerprint": identity[1]}
            if identity
            else None
        ),
        "split_counts": split_counts,
        "total_prepared": total_prepared,
        "total_machine_pass": total_machine_pass,
        "total_machine_rejected_or_pending": total_prepared - total_machine_pass,
        "failure_count": len(ordered_failures),
        "usage": usage,
        "quality_status": (
            "deferred_post_acquisition"
            if acquisition_mode == "acquire_first"
            else status
        ),
        "human_review_status": "pending",
        "training_ready": False,
        "preparation": dict(preparation),
        "lineage": _lineage_flags(),
    }
    _write_json(output_root / "summary.json", summary)
    _write_json(
        output_root / "handoff.json",
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "dataset_path": _display_path(output_root / "sft_candidate"),
            "manifest_path": _display_path(output_root / "manifests"),
            "status": status,
            "acquisition_mode": acquisition_mode,
            "teacher_model": TEACHER_MODEL,
            "split_counts": split_counts,
            "total_machine_pass": total_machine_pass,
            "training_ready": False,
            "evaluation_eligible": False,
            "human_review_status": "pending",
            "summary": {
                "path": "summary.json",
                "sha256": sha256_file(output_root / "summary.json"),
            },
            "lineage": _lineage_flags(),
        },
    )
    return summary


def _validate_runtime_options(*, concurrency: int, preflight_rows: int) -> None:
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise TargetDerivedDataError(
            f"concurrency must be between 1 and {MAX_CONCURRENCY}"
        )
    if not 1 <= preflight_rows <= 64:
        raise TargetDerivedDataError("preflight_rows must be between 1 and 64")


def run_pipeline(
    prepared: Mapping[str, Sequence[PreparedRow]],
    *,
    preparation: Mapping[str, Any],
    output_root: str | Path,
    tokenizer: Any,
    backend: TeacherBackend,
    environment: Mapping[str, str] | None,
    concurrency: int = DEFAULT_CONCURRENCY,
    preflight_rows: int = DEFAULT_PREFLIGHT_ROWS,
    resume: bool = False,
    phase: str = "all",
    acquire_first: bool = False,
) -> dict[str, Any]:
    _validate_runtime_options(concurrency=concurrency, preflight_rows=preflight_rows)
    if phase not in {"generate", "verify", "all"}:
        raise TargetDerivedDataError(f"invalid phase: {phase}")
    if acquire_first and phase != "generate":
        raise TargetDerivedDataError(
            "acquire_first requires generation-only phase before later filtering"
        )
    output = Path(output_root).resolve()
    rows = _flatten(prepared)
    if not rows:
        raise TargetDerivedDataError("no prepared rows are available for acquisition")
    code_sha256 = sha256_file(Path(__file__).resolve())
    config = TeacherConfig.from_environment(environment)
    identity_guard = ProviderIdentityGuard()
    if not resume:
        cache_phases = []
        if phase in {"generate", "all"}:
            cache_phases.append("generation")
        if phase in {"verify", "all"}:
            cache_phases.append("verification")
        for cache_phase in cache_phases:
            cache_root = output / "cache" / cache_phase
            existing = sorted(cache_root.glob("*.json")) if cache_root.is_dir() else []
            if existing:
                raise TargetDerivedDataError(
                    f"{cache_phase} cache exists; pass --resume: {existing[0]}"
                )
    _write_json(
        output / "prompt_contract.json",
        prompt_contract(config=config, code_sha256=code_sha256),
    )

    failures: list[dict[str, Any]] = []
    if acquire_first:
        _write_json(
            output / "preflight.json",
            {
                "schema_version": "chk2-target-derived-preflight-v1",
                "status": "skipped_for_acquire_first",
                "purpose": "content_quality_gates_deferred_until_full_acquisition",
                "request_or_spending_budget": None,
                "total_generation_rows": len(rows),
                "generation": None,
                "verification": None,
            },
        )
        generation_caches, generation_failures = _load_or_generate_wave(
            rows,
            output_root=output,
            config=config,
            backend=backend,
            identity_guard=identity_guard,
            environment=environment,
            concurrency=concurrency,
            resume=resume,
            code_sha256=code_sha256,
        )
        failures.extend(generation_failures)
        fatal_generation_failures = _release_fatal_failures(generation_failures)
        if fatal_generation_failures:
            _persist_acquisition_abort(
                output_root=output,
                reason=(
                    "release-fatal provider integrity failure during acquire-first "
                    "generation"
                ),
                failures=failures,
                generation_caches=generation_caches,
                verification_caches={},
                identity=identity_guard.identity,
            )
            raise TargetDerivedDataError(
                "acquire-first generation encountered a release-fatal provider "
                f"integrity failure; see {output / 'acquisition_abort.json'}"
            )
        return _materialize(
            prepared,
            output_root=output,
            generation_caches=generation_caches,
            verification_caches={},
            tokenizer=tokenizer,
            phase="generate",
            failures=failures,
            identity=identity_guard.identity,
            preparation=preparation,
            acquisition_mode="acquire_first",
        )

    preflight_report: dict[str, Any] = {
        "schema_version": "chk2-target-derived-preflight-v1",
        "configured_rows": preflight_rows,
        "minimum_pass_fraction": PREFLIGHT_MIN_PASS_FRACTION,
        "purpose": "end_to_end_contract_quality_gate",
        "selection_strategy": "split_round_robin_with_length_quantile_coverage",
        "request_or_spending_budget": None,
        "full_acquisition_after_pass": True,
        "generation": None,
        "verification": None,
    }
    generation_caches: dict[str, tuple[dict[str, Any], TeacherResponse]] = {}
    verification_caches: dict[str, tuple[dict[str, Any], TeacherResponse]] = {}
    generation_preflight_rows: list[PreparedRow] = []
    generation_preflight_candidates: list[tuple[PreparedRow, Mapping[str, Any]]] = []

    # Stage 1 of the end-to-end quality gate: acquire and fully parse a small
    # generation wave. Passing this gate does not cap the subsequent run.
    if phase in {"generate", "all"}:
        generation_preflight_rows = _select_preflight_rows(rows, preflight_rows)
        generation_caches, generation_failures = _load_or_generate_wave(
            generation_preflight_rows,
            output_root=output,
            config=config,
            backend=backend,
            identity_guard=identity_guard,
            environment=environment,
            concurrency=concurrency,
            resume=resume,
            code_sha256=code_sha256,
        )
        failures.extend(generation_failures)
        generation_checks: list[dict[str, Any]] = []
        generation_passes = 0
        for row in generation_preflight_rows:
            cached = generation_caches.get(row.sample_id)
            if cached is None:
                generation_checks.append(
                    {
                        "sample_id": row.sample_id,
                        "passed": False,
                        "error": "provider_response_missing",
                    }
                )
                continue
            try:
                generation, _ = validate_generation(row, cached[1], tokenizer=tokenizer)
                generation_passes += 1
                generation_preflight_candidates.append((row, generation))
                generation_checks.append(
                    {"sample_id": row.sample_id, "passed": True, "error": None}
                )
            except Exception as exc:  # fail-closed smoke contract
                generation_checks.append(
                    {
                        "sample_id": row.sample_id,
                        "passed": False,
                        "error": _safe_error(exc),
                    }
                )
        required_generation_passes = max(
            1,
            math.ceil(len(generation_preflight_rows) * PREFLIGHT_MIN_PASS_FRACTION),
        )
        generation_preflight_passed = (
            not generation_failures and generation_passes >= required_generation_passes
        )
        quarantined_generation_caches = (
            []
            if generation_preflight_passed
            else _quarantine_preflight_caches(
                generation_preflight_rows,
                output_root=output,
                phase="generation",
            )
        )
        preflight_report["generation"] = {
            "rows": len(generation_preflight_rows),
            "required_passes": required_generation_passes,
            "observed_passes": generation_passes,
            "passed": generation_preflight_passed,
            "checks": generation_checks,
            "transport_failures": [dict(item) for item in generation_failures],
            "usage": _attempt_usage(
                list(generation_caches.values()), generation_failures
            ),
            "quarantined_caches": quarantined_generation_caches,
        }
        _write_json(output / "preflight.json", preflight_report)
        print(
            "[preflight] generation "
            f"passed={generation_passes}/{len(generation_preflight_rows)} "
            f"required={required_generation_passes}",
            file=sys.stderr,
            flush=True,
        )
        if not generation_preflight_passed:
            raise TargetDerivedDataError(
                "generation preflight failed; full acquisition was not started. "
                f"See {output / 'preflight.json'}"
            )
    else:
        generation_caches = _load_generation_caches(
            rows,
            output_root=output,
            config=config,
            identity_guard=identity_guard,
            code_sha256=code_sha256,
        )

    # Stage 2 of the end-to-end quality gate happens before the remaining
    # generation calls when phase=all. For verify-only runs, choose the first
    # valid cached generations instead.
    verification_preflight: list[tuple[PreparedRow, Mapping[str, Any]]] = []
    if phase in {"verify", "all"}:
        if phase == "all":
            verification_preflight = generation_preflight_candidates
        else:
            for row in rows:
                cached = generation_caches.get(row.sample_id)
                if cached is None:
                    continue
                try:
                    generation, _ = validate_generation(
                        row, cached[1], tokenizer=tokenizer
                    )
                except Exception:
                    continue
                verification_preflight.append((row, generation))
                if len(verification_preflight) >= preflight_rows:
                    break
        if not verification_preflight:
            preflight_report["verification"] = {
                "rows": 0,
                "required_passes": 1,
                "observed_passes": 0,
                "passed": False,
                "checks": [],
                "transport_failures": [],
                "usage": _attempt_usage([], []),
                "quarantined_caches": [],
            }
            _write_json(output / "preflight.json", preflight_report)
            raise TargetDerivedDataError(
                "verification preflight has no valid generation candidates"
            )
        verification_caches, verification_failures = _load_or_verify_wave(
            verification_preflight,
            output_root=output,
            config=config,
            backend=backend,
            identity_guard=identity_guard,
            environment=environment,
            concurrency=concurrency,
            resume=resume,
            code_sha256=code_sha256,
        )
        failures.extend(verification_failures)
        verification_checks: list[dict[str, Any]] = []
        verification_passes = 0
        for row, generation in verification_preflight:
            cached = verification_caches.get(row.sample_id)
            if cached is None:
                verification_checks.append(
                    {
                        "sample_id": row.sample_id,
                        "passed": False,
                        "error": "provider_response_missing",
                    }
                )
                continue
            try:
                _, machine_pass, reasons = validate_verification(
                    row, generation, cached[1]
                )
                if machine_pass:
                    verification_passes += 1
                verification_checks.append(
                    {
                        "sample_id": row.sample_id,
                        "passed": machine_pass,
                        "error": None if machine_pass else ";".join(reasons),
                    }
                )
            except Exception as exc:  # fail-closed smoke contract
                verification_checks.append(
                    {
                        "sample_id": row.sample_id,
                        "passed": False,
                        "error": _safe_error(exc),
                    }
                )
        verification_denominator = (
            len(generation_preflight_rows)
            if phase == "all"
            else len(verification_preflight)
        )
        required_verification_passes = max(
            1,
            math.ceil(verification_denominator * PREFLIGHT_MIN_PASS_FRACTION),
        )
        verification_preflight_passed = (
            not verification_failures
            and verification_passes >= required_verification_passes
        )
        quarantined_verification_caches = (
            []
            if verification_preflight_passed
            else _quarantine_preflight_caches(
                [row for row, _ in verification_preflight],
                output_root=output,
                phase="verification",
            )
        )
        preflight_report["verification"] = {
            "rows": len(verification_preflight),
            "end_to_end_denominator": verification_denominator,
            "required_passes": required_verification_passes,
            "observed_passes": verification_passes,
            "passed": verification_preflight_passed,
            "checks": verification_checks,
            "transport_failures": [dict(item) for item in verification_failures],
            "usage": _attempt_usage(
                list(verification_caches.values()), verification_failures
            ),
            "quarantined_caches": quarantined_verification_caches,
        }
        _write_json(output / "preflight.json", preflight_report)
        print(
            "[preflight] verification "
            f"passed={verification_passes}/{len(verification_preflight)} "
            f"required={required_verification_passes}",
            file=sys.stderr,
            flush=True,
        )
        if not verification_preflight_passed:
            raise TargetDerivedDataError(
                "verification preflight failed; full verification was not started. "
                f"See {output / 'preflight.json'}"
            )

    # Both requested preflight stages have passed. There is deliberately no
    # request-count or spending budget: acquire every remaining eligible row.
    if phase in {"generate", "all"}:
        generation_preflight_ids = {row.sample_id for row in generation_preflight_rows}
        remaining_generation_rows = [
            row for row in rows if row.sample_id not in generation_preflight_ids
        ]
        if remaining_generation_rows:
            remaining_caches, remaining_failures = _load_or_generate_wave(
                remaining_generation_rows,
                output_root=output,
                config=config,
                backend=backend,
                identity_guard=identity_guard,
                environment=environment,
                concurrency=concurrency,
                resume=resume,
                code_sha256=code_sha256,
            )
            generation_caches.update(remaining_caches)
            failures.extend(remaining_failures)
            fatal_generation_failures = _release_fatal_failures(remaining_failures)
            if fatal_generation_failures:
                _persist_acquisition_abort(
                    output_root=output,
                    reason=(
                        "release-fatal provider failure during full generation; "
                        "verification was not continued"
                    ),
                    failures=failures,
                    generation_caches=generation_caches,
                    verification_caches=verification_caches,
                    identity=identity_guard.identity,
                )
                raise TargetDerivedDataError(
                    "full generation encountered a release-fatal provider failure; "
                    f"see {output / 'acquisition_abort.json'}"
                )

    if phase in {"verify", "all"}:
        parsed_for_verification: list[tuple[PreparedRow, Mapping[str, Any]]] = []
        for row in rows:
            cached = generation_caches.get(row.sample_id)
            if cached is None:
                continue
            try:
                generation, _ = validate_generation(row, cached[1], tokenizer=tokenizer)
            except Exception:
                continue
            parsed_for_verification.append((row, generation))
        preflight_ids = {row.sample_id for row, _ in verification_preflight}
        remaining_verification = [
            pair
            for pair in parsed_for_verification
            if pair[0].sample_id not in preflight_ids
        ]
        if remaining_verification:
            remaining_caches, remaining_failures = _load_or_verify_wave(
                remaining_verification,
                output_root=output,
                config=config,
                backend=backend,
                identity_guard=identity_guard,
                environment=environment,
                concurrency=concurrency,
                resume=resume,
                code_sha256=code_sha256,
            )
            verification_caches.update(remaining_caches)
            failures.extend(remaining_failures)

    return _materialize(
        prepared,
        output_root=output,
        generation_caches=generation_caches,
        verification_caches=verification_caches,
        tokenizer=tokenizer,
        phase=phase,
        failures=failures,
        identity=identity_guard.identity,
        preparation=preparation,
        acquisition_mode="quality_gated",
    )


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise TargetDerivedDataError(
            "transformers is required for token audit"
        ) from exc
    return AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)


def _parse_splits(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(result) - set(OUTPUT_SPLITS))
    if not result or invalid:
        raise argparse.ArgumentTypeError(
            f"splits must be comma-separated from {OUTPUT_SPLITS}; invalid={invalid}"
        )
    return tuple(dict.fromkeys(result))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--splits", type=_parse_splits, default=OUTPUT_SPLITS)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Deterministic total row limit after target-only eligibility/deduplication.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("DEEPSEEK_CONCURRENCY", DEFAULT_CONCURRENCY)),
    )
    parser.add_argument(
        "--preflight-rows",
        type=int,
        default=DEFAULT_PREFLIGHT_ROWS,
        help=(
            "End-to-end contract checks before full acquisition; this is a "
            "quality gate, not a request or spending limit."
        ),
    )
    parser.add_argument("--phase", choices=("generate", "verify", "all"), default="all")
    parser.add_argument(
        "--acquire-first",
        action="store_true",
        help=(
            "Acquire every raw generation response before applying content gates; "
            "verification is deferred to a later filtering pass."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only immutable caches with exact source/prompt/config/code bindings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare target-only inputs and contracts without reading credentials/API calls.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_runtime_options(
        concurrency=args.concurrency, preflight_rows=args.preflight_rows
    )
    effective_phase = "generate" if args.acquire_first else args.phase
    output = args.output_root.resolve()
    prepared, preparation = prepare_official_targets(
        source_root=args.source_root,
        split_manifest_path=args.split_manifest,
        output_root=output,
        selected_splits=args.splits,
        limit=args.limit,
    )
    config = TeacherConfig.from_environment(os.environ)
    _write_json(
        output / "prompt_contract.json",
        prompt_contract(
            config=config, code_sha256=sha256_file(Path(__file__).resolve())
        ),
    )
    if args.dry_run:
        total_prepared = preparation["total_prepared"]
        planned_generation = (
            total_prepared if effective_phase in {"generate", "all"} else 0
        )
        planned_verification = (
            total_prepared if effective_phase in {"verify", "all"} else 0
        )
        planned_total = planned_generation + planned_verification
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "dry_run_ready",
            "phase": effective_phase,
            "acquisition_mode": (
                "acquire_first" if args.acquire_first else "quality_gated"
            ),
            "resume": args.resume,
            "teacher_model": TEACHER_MODEL,
            "concurrency": args.concurrency,
            "preflight_rows": 0 if args.acquire_first else args.preflight_rows,
            "provider_max_tokens": None,
            "request_or_spending_budget": None,
            "planned_generation_requests": planned_generation,
            "planned_verification_requests": planned_verification,
            "planned_fresh_requests": planned_total,
            "planned_request_note": (
                "Fresh-cache count; --resume can reduce transport calls by reusing "
                "exact bound caches. This is not a budget."
            ),
            "planned_maximum_transport_attempts": (
                planned_total * (config.max_retries + 1)
            ),
            "preparation": preparation,
            "training_ready": False,
            "lineage": _lineage_flags(),
        }
        _write_json(output / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if not str(os.environ.get(API_KEY_ENV) or "").strip():
        raise TargetDerivedDataError(
            f"missing {API_KEY_ENV}; preparation completed without API calls. "
            "Set the key in the environment (do not paste it into logs) and rerun."
        )
    tokenizer = _load_tokenizer(args.tokenizer_path.resolve())
    summary = run_pipeline(
        prepared,
        preparation=preparation,
        output_root=output,
        tokenizer=tokenizer,
        backend=OpenAICompatibleDeepSeekBackend(),
        environment=os.environ,
        concurrency=args.concurrency,
        preflight_rows=args.preflight_rows,
        resume=args.resume,
        phase=effective_phase,
        acquire_first=args.acquire_first,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] != "incomplete" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TargetDerivedDataError as exc:
        print(f"target-derived chk2 generation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
