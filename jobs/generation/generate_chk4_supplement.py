"""Run the DeepSeek and release stages for the chk4 1993--2008 supplement.

The three provider stages are deliberately isolated:

* ``summarize`` is gold blind and sees only compact exact-D-1 evidence;
* ``blind-predict`` sees only the saved meeting brief and is audit-only; and
* ``teacher-targets`` joins hidden local gold only to acquire grounded SFT
  reasoning, while serializing the final decision locally.

No command in this module starts model training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from jobs.generation.generate_chk4_sft_targets import (
    API_KEY_ENV,
    DEFAULT_TOKENIZER,
    MANIFEST_SCHEMA,
    STUDENT_SYSTEM_PROMPT,
    TEACHER_BASE_URL,
    TEACHER_MODEL,
    OpenAICompatibleDeepSeekBackend,
    TeacherResponse,
    canonical_json,
    render_student_prompt,
    sha256_file,
    sha256_text,
)
from jobs.generation.materialize_chk4_training_data import (
    MaterializationError,
    load_unique_rows,
    materialize,
)
from jobs.generation.prepare_chk4_supplement import (
    ADMISSION_PROFILE,
    EXPECTED_ACTION_COUNTS,
    MIN_ADMITTED,
    MIN_VALID_ATOMIC_TOPICS,
    POPULATION,
    REQUIRED_CATEGORIES,
    SUMMARY_SYSTEM_PROMPT,
    SupplementPreparationError,
    _load_json,
    _load_jsonl,
    _update_root_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk4/decision_supplement_1993_2008_v1"
)
DEFAULT_CORE_TEACHER_ROOT = REPO_ROOT / "output/data/retrain_v2/chk4/deepseek_v4_pro_v2"

DEFAULT_CONCURRENCY = 8
SUMMARY_MAX_TOKENS = 4096
DECISION_MAX_TOKENS = 2048
MAX_BRIEF_TOKENS = 1200
MAX_STUDENT_PROMPT_TOKENS = 2560
MAX_COMPLETION_TOKENS = 512
MAX_TOTAL_TOKENS = 3072
SUMMARY_PROMPT_CONTRACT_FILE = "prompt_contract.grounding-v6.json"

SUMMARY_SCHEMA = "chk4-decision-supplement-summary-v1"
BLIND_SCHEMA = "chk4-decision-supplement-blind-prediction-v1"
TARGET_SCHEMA = "chk4-decision-supplement-teacher-target-v1"
CACHE_SCHEMA = "chk4-decision-supplement-provider-cache-v1"
CONTRACT_SCHEMA = "chk4-decision-supplement-provider-contract-v1"
COMBINED_SCHEMA = "chk4-decision-combined-teacher-release-v1"

SUMMARY_REPAIR_SYSTEM_PROMPT = """\
Regenerate the target-neutral pre-meeting summary using only the supplied
point-in-time topic evidence and silently correct the format or grounding
failure. Do not discuss validation. Do not infer meeting identity; do not
state, recommend, predict, or imply a policy decision, vote, rate change,
target range, or action actually taken. Add no fact, cause, number, or date
absent from the evidence.

Return exactly one JSON object with the single key meeting_decision_brief. Its
value must be one coherent paragraph under 180 words covering inflation,
employment or real activity, financial conditions, and risks. For this repair,
use fully qualitative language with no digits, dates, percentages, basis
points, base years, or spelled-out numeric quantities. If a quantitative fact
cannot be expressed qualitatively, omit it. This is the only repair attempt.
"""

BLIND_SYSTEM_PROMPT = """\
You are auditing whether a target-neutral pre-meeting analysis is sufficient
for one FOMC policy decision. Use only the supplied analysis. Weigh the
evidence under the maximum-employment and price-stability objectives.

Do not use outside or remembered historical information, infer the meeting
identity, or claim that the Committee actually took an action. Return content
as exactly one JSON object with keys reasoning, direction, and magnitude_bp.
Reasoning must be one grounded paragraph under 180 words. Direction must be
cut, hold, or hike. Hold requires magnitude 0; cut and hike require 25, 50,
75, or 100. Output no other fields or content.
"""

BLIND_REPAIR_SYSTEM_PROMPT = """\
Regenerate the decision JSON using only the supplied pre-meeting analysis and
silently correct the format or grounding failure. Return exactly the keys
reasoning, direction, and magnitude_bp. Keep reasoning to one grounded
paragraph under 180 words. Direction must be cut, hold, or hike; hold requires
zero and cut or hike requires 25, 50, 75, or 100. This is the only repair
attempt.
"""

TARGET_SYSTEM_PROMPT = """\
You are preparing one grounded rationale for an FOMC policy Decision-SFT
example. Use only the supplied target-neutral pre-meeting analysis. In the
reasoning field, weigh its evidence under the maximum-employment and
price-stability objectives and support the canonical decision supplied at the
end of the request.

Do not use outside or remembered historical information, infer or mention the
meeting identity, or claim that the Committee actually took an action. Do not
mention gold labels, teacher targets, answer keys, hidden fields, known
decisions, historical actions, prompts, schemas, validation, or teacher and
student roles. Write qualitative reasoning without digits, dates,
percentages, basis-point amounts, or explicit numeric quantities. Add no fact
absent from the analysis.

Return exactly one JSON object with keys reasoning, direction, and
magnitude_bp. Reasoning must be one paragraph under 180 words. Direction and
magnitude_bp must exactly equal the supplied canonical decision.
"""

TARGET_REPAIR_SYSTEM_PROMPT = """\
Regenerate the three-key JSON using only the supplied analysis and canonical
decision, silently correcting the format or grounding failure. Do not mention
the hidden decision or historical outcome. The reasoning must be one
qualitative paragraph under 180 words with no digits, dates, percentages, or
basis-point amounts. Direction and magnitude_bp must exactly equal the
supplied decision. This is the only repair attempt.
"""

BLIND_USER_PREFIX = (
    "Make one policy decision using only the following pre-meeting analysis:\n\n"
)
TARGET_GOLD_PREFIX = (
    "\n\nCanonical decision for rationale alignment only; never mention its hidden "
    "status or any historical outcome:\n"
)

CONTROL_MARKERS = (
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
    "\\boxed",
    "<|channel>",
    "<channel|>",
)
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![A-Za-z0-9])"
)
DATE_RE = re.compile(
    r"\b(?:(?i:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)|May)\b|\b(?:19|20)\d{2}\b|\b\d{4}-\d{2}-\d{2}\b"
)
FULL_YEAR_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})\s*[-–]\s*((?:19|20)\d{2})\b")
ABBREVIATED_YEAR_RANGE_RE = re.compile(r"\b((?:19|20)\d{2})\s*[-–]\s*(\d{2})\b")
BASIS_POINT_RE = re.compile(
    r"(?<![A-Za-z0-9])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*"
    r"(?:basis\s+points?|bps?)\b",
    re.I,
)
QUALITATIVE_SPELLED_NUMBER_RE = re.compile(
    r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.I,
)
POLICY_LEAK_PATTERNS = (
    re.compile(r"\b(?:Committee|FOMC)\s+(?:voted|decided|raised|cut|held)\b", re.I),
    re.compile(r"\b(?:recommend|recommended|should)\s+(?:cut|raise|hike|hold)\b", re.I),
    re.compile(r"\b(?:actual|historical|known)\s+(?:decision|action|vote)\b", re.I),
)
TARGET_META_PATTERNS = (
    re.compile(r"\bgold(?:en)?\s+(?:label|decision|target)\b", re.I),
    re.compile(r"\bteacher(?:-only)?\s+(?:label|target|decision)\b", re.I),
    re.compile(r"\b(?:known|actual|historical)\s+(?:decision|action|vote)\b", re.I),
)


class SupplementGenerationError(SupplementPreparationError):
    """Provider output or release materialization violates the contract."""


class StageOutputError(SupplementGenerationError):
    def __init__(self, codes: Sequence[str]):
        self.codes = tuple(str(code) for code in codes)
        super().__init__(";".join(self.codes))


class ProviderDriftError(SupplementGenerationError):
    """DeepSeek model identity or fingerprint changed during the release."""


@dataclass(frozen=True)
class ProviderConfig:
    model: str = TEACHER_MODEL
    base_url: str = TEACHER_BASE_URL
    max_tokens: int = DECISION_MAX_TOKENS
    timeout_seconds: float = 180.0
    max_retries: int = 3
    reasoning_effort: str = "high"

    def contract(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "thinking": {"type": "enabled"},
            "reasoning_effort": self.reasoning_effort,
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "fallback": "forbidden",
            "repair_attempts": 1,
            "api_key_env": API_KEY_ENV,
        }


class ProviderBackend(Protocol):
    def generate(
        self,
        *,
        config: ProviderConfig,
        system_prompt: str,
        user_prompt: str,
        environment: Mapping[str, str] | None,
    ) -> TeacherResponse: ...


@dataclass(frozen=True)
class StageRow:
    sample_id: str
    prompt: str
    input_text: str
    input_sha256: str
    prompt_sha256: str
    meeting_date: str = ""
    direction: str = ""
    magnitude_bp: int = -1
    source_ids: tuple[str, ...] = ()

    @property
    def gold(self) -> dict[str, Any]:
        return {"direction": self.direction, "magnitude_bp": self.magnitude_bp}


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(
    path: Path, payload: Mapping[str, Any], *, immutable: bool = False
) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if immutable and path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SupplementGenerationError(f"immutable artifact drift: {path}")
        return
    _atomic_write(path, text)


def _write_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]], *, immutable: bool = False
) -> None:
    text = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    if immutable and path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SupplementGenerationError(f"immutable artifact drift: {path}")
        return
    _atomic_write(path, text)


def _store_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    _write_json(path, payload, immutable=True)


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SupplementGenerationError("transformers is unavailable") from exc
    if not path.is_dir():
        raise SupplementGenerationError(f"tokenizer path is missing: {path}")
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _render_student(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def _canonical_gold(direction: str, magnitude: int) -> str:
    return json.dumps(
        {"direction": direction, "magnitude_bp": magnitude},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _number_atoms(text: str) -> set[str]:
    atoms: set[str] = set()
    for match in NUMBER_RE.finditer(text):
        raw = match.group(0).replace(",", "")
        try:
            value = Decimal(raw)
        except InvalidOperation:
            atoms.add(raw)
            continue
        rendered = format(value, "f")
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
        atoms.add("0" if rendered in {"", "-0"} else rendered)
    return atoms


def qualitative_reasoning_has_number_or_date(text: str) -> bool:
    """Use one producer/runtime predicate for forbidden quantitative language."""

    return bool(
        _number_atoms(text)
        or DATE_RE.search(text)
        or QUALITATIVE_SPELLED_NUMBER_RE.search(text)
    )


def _grounded_summary_number_atoms(text: str) -> set[str]:
    """Include unit-explicit absolute equivalents of supplied source values."""

    allowed = _number_atoms(text)
    json_start = text.find("{")
    if json_start < 0:
        return allowed
    try:
        payload = json.loads(text[json_start:])
    except (json.JSONDecodeError, TypeError):
        return allowed
    topics = payload.get("topic_evidence") if isinstance(payload, dict) else None
    if not isinstance(topics, list):
        return allowed
    unit_scales = {
        "thousand": Decimal(1_000),
        "million": Decimal(1_000_000),
        "billion": Decimal(1_000_000_000),
    }
    for topic in topics:
        if not isinstance(topic, dict) or not isinstance(topic.get("series"), list):
            continue
        for series in topic["series"]:
            if not isinstance(series, dict):
                continue
            units = str(series.get("units") or "").lower()
            scale = next(
                (value for marker, value in unit_scales.items() if marker in units),
                None,
            )
            if scale is None:
                continue
            raw_values = [series.get("latest_value")]
            changes = series.get("relative_changes")
            if isinstance(changes, list):
                raw_values.extend(
                    change.get("value_change")
                    for change in changes
                    if isinstance(change, dict)
                )
            for raw in raw_values:
                try:
                    scaled = abs(Decimal(str(raw)) * scale)
                except (InvalidOperation, TypeError, ValueError):
                    continue
                if not scaled.is_finite():
                    continue
                rendered = format(scaled, "f")
                if "." in rendered:
                    rendered = rendered.rstrip("0").rstrip(".")
                allowed.add("0" if rendered in {"", "-0"} else rendered)
    return allowed


def _expand_grounded_abbreviated_year_ranges(input_text: str, brief: str) -> str:
    """Expand only occurrences that exactly match a supplied full year range."""

    supplied = {
        (match.group(1), match.group(2))
        for match in FULL_YEAR_RANGE_RE.finditer(input_text)
    }

    def expand(match: re.Match[str]) -> str:
        start, short_end = match.groups()
        full_end = start[:2] + short_end
        if (start, full_end) in supplied:
            return f"{start}-{full_end}"
        return match.group(0)

    return ABBREVIATED_YEAR_RANGE_RE.sub(expand, brief)


def _expand_grounded_basis_points(input_text: str, brief: str) -> str:
    """Normalize only basis-point occurrences backed by percent-series changes."""

    json_start = input_text.find("{")
    if json_start < 0:
        return brief
    try:
        payload = json.loads(input_text[json_start:])
    except (json.JSONDecodeError, TypeError):
        return brief
    topics = payload.get("topic_evidence") if isinstance(payload, dict) else None
    if not isinstance(topics, list):
        return brief

    supplied_changes: set[Decimal] = set()
    for topic in topics:
        if not isinstance(topic, dict) or not isinstance(topic.get("series"), list):
            continue
        for series in topic["series"]:
            if not isinstance(series, dict):
                continue
            units = str(series.get("units") or "").lower()
            if "percent" not in units:
                continue
            changes = series.get("relative_changes")
            if not isinstance(changes, list):
                continue
            for change in changes:
                if not isinstance(change, dict):
                    continue
                try:
                    value = abs(Decimal(str(change.get("value_change"))))
                except (InvalidOperation, TypeError, ValueError):
                    continue
                if value.is_finite():
                    supplied_changes.add(value)

    def expand(match: re.Match[str]) -> str:
        try:
            percentage_points = Decimal(match.group(1).replace(",", "")) / 100
        except InvalidOperation:
            return match.group(0)
        if percentage_points not in supplied_changes:
            return match.group(0)
        rendered = format(percentage_points, "f")
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
        return f"{rendered} percentage points"

    return BASIS_POINT_RE.sub(expand, brief)


def _strict_decision(content: str) -> tuple[str, str, int]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise StageOutputError(("invalid_json",)) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "reasoning",
        "direction",
        "magnitude_bp",
    }:
        raise StageOutputError(("decision_json_keys",))
    reasoning = payload.get("reasoning")
    direction = payload.get("direction")
    magnitude = payload.get("magnitude_bp")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise StageOutputError(("empty_reasoning",))
    if direction not in {"cut", "hold", "hike"}:
        raise StageOutputError(("unsupported_direction",))
    if isinstance(magnitude, bool) or not isinstance(magnitude, int):
        raise StageOutputError(("magnitude_not_integer",))
    allowed = {0} if direction == "hold" else {25, 50, 75, 100}
    if magnitude not in allowed:
        raise StageOutputError(("unsupported_magnitude",))
    return reasoning.strip(), direction, magnitude


def _validate_summary(
    row: StageRow, response: TeacherResponse, tokenizer: Any
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        payload = json.loads(response.content)
    except json.JSONDecodeError:
        payload = None
        errors.append("invalid_json")
    if not isinstance(payload, dict) or set(payload) != {"meeting_decision_brief"}:
        brief = ""
        errors.append("summary_json_keys")
    else:
        brief = str(payload.get("meeting_decision_brief") or "").strip()
    if not brief:
        errors.append("empty_brief")
    if "\n" in brief:
        errors.append("brief_not_one_paragraph")
    if any(marker.lower() in brief.lower() for marker in CONTROL_MARKERS):
        errors.append("control_marker")
    if any(pattern.search(brief) for pattern in POLICY_LEAK_PATTERNS):
        errors.append("policy_action_or_recommendation")
    grounded_numbers = _grounded_summary_number_atoms(row.input_text)
    normalized_brief = _expand_grounded_abbreviated_year_ranges(row.input_text, brief)
    normalized_brief = _expand_grounded_basis_points(row.input_text, normalized_brief)
    unsupported_numbers = sorted(_number_atoms(normalized_brief) - grounded_numbers)
    if unsupported_numbers:
        errors.append("unsupported_numbers:" + ",".join(unsupported_numbers))
    input_dates = {match.group(0).lower() for match in DATE_RE.finditer(row.input_text)}
    unsupported_dates = sorted(
        {match.group(0).lower() for match in DATE_RE.finditer(brief)} - input_dates
    )
    if unsupported_dates:
        errors.append("unsupported_dates:" + ",".join(unsupported_dates))
    lowered = brief.lower()
    if not any(word in lowered for word in ("inflation", "price", "commodity")):
        errors.append("missing_prices_synthesis")
    if not any(
        word in lowered
        for word in ("employment", "labor", "labour", "activity", "output", "growth")
    ):
        errors.append("missing_activity_synthesis")
    if not any(
        word in lowered
        for word in (
            "financial",
            "credit",
            "yield",
            "market",
            "mortgage",
            "exchange",
            "money",
            "monetary",
            "liquidity",
        )
    ):
        errors.append("missing_financial_synthesis")
    brief_tokens = _token_count(tokenizer, brief) if brief else 0
    student_prompt = render_student_prompt(brief) if brief else ""
    student_prompt_tokens = (
        _token_count(tokenizer, _render_student(tokenizer, student_prompt))
        if brief
        else 0
    )
    if brief_tokens > MAX_BRIEF_TOKENS:
        errors.append(f"brief_tokens:{brief_tokens}>{MAX_BRIEF_TOKENS}")
    if student_prompt_tokens > MAX_STUDENT_PROMPT_TOKENS:
        errors.append(
            f"student_prompt_tokens:{student_prompt_tokens}>{MAX_STUDENT_PROMPT_TOKENS}"
        )
    if response.finish_reason != "stop":
        errors.append(f"finish_reason:{response.finish_reason}")
    if errors:
        raise StageOutputError(errors)
    return {
        "meeting_decision_brief": brief,
        "brief_sha256": sha256_text(brief),
        "brief_tokens": brief_tokens,
        "student_prompt_tokens": student_prompt_tokens,
    }


def _validate_blind(
    row: StageRow, response: TeacherResponse, tokenizer: Any
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        reasoning, direction, magnitude = _strict_decision(response.content)
    except StageOutputError as exc:
        errors.extend(exc.codes)
        reasoning, direction, magnitude = "", "", -1
    if any(marker.lower() in reasoning.lower() for marker in CONTROL_MARKERS):
        errors.append("control_marker")
    if any(pattern.search(reasoning) for pattern in TARGET_META_PATTERNS):
        errors.append("forbidden_target_reference")
    unsupported_numbers = sorted(
        _number_atoms(reasoning) - _number_atoms(row.input_text)
    )
    if unsupported_numbers:
        errors.append("unsupported_reasoning_numbers:" + ",".join(unsupported_numbers))
    if len(reasoning.split()) > 180:
        errors.append("reasoning_words_gt_180")
    completion = reasoning + "\n" + _canonical_gold(direction, magnitude)
    completion_tokens = _token_count(tokenizer, completion)
    prompt_tokens = _token_count(tokenizer, _render_student(tokenizer, row.prompt))
    total_tokens = _token_count(
        tokenizer, _render_student(tokenizer, row.prompt) + completion
    )
    if prompt_tokens > MAX_STUDENT_PROMPT_TOKENS:
        errors.append(f"prompt_tokens:{prompt_tokens}>{MAX_STUDENT_PROMPT_TOKENS}")
    if completion_tokens > MAX_COMPLETION_TOKENS:
        errors.append(f"completion_tokens:{completion_tokens}>{MAX_COMPLETION_TOKENS}")
    if total_tokens > MAX_TOTAL_TOKENS:
        errors.append(f"total_tokens:{total_tokens}>{MAX_TOTAL_TOKENS}")
    if response.finish_reason != "stop":
        errors.append(f"finish_reason:{response.finish_reason}")
    if errors:
        raise StageOutputError(errors)
    return {
        "reasoning": reasoning,
        "direction": direction,
        "magnitude_bp": magnitude,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _validate_target(
    row: StageRow, response: TeacherResponse, tokenizer: Any
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        reasoning, direction, magnitude = _strict_decision(response.content)
    except StageOutputError as exc:
        errors.extend(exc.codes)
        reasoning, direction, magnitude = "", "", -1
    if (direction, magnitude) != (row.direction, row.magnitude_bp):
        errors.append("teacher_content_disagrees_with_gold")
    if any(marker.lower() in reasoning.lower() for marker in CONTROL_MARKERS):
        errors.append("control_marker")
    if any(pattern.search(reasoning) for pattern in TARGET_META_PATTERNS):
        errors.append("forbidden_target_reference")
    if qualitative_reasoning_has_number_or_date(reasoning):
        errors.append("target_reasoning_contains_number_or_date")
    if len(reasoning.split()) > 180:
        errors.append("reasoning_words_gt_180")
    completion = (
        reasoning + "\n</think>\n" + _canonical_gold(row.direction, row.magnitude_bp)
    )
    if completion.count("</think>") != 1:
        errors.append("think_boundary_count")
    completion_tokens = _token_count(tokenizer, completion)
    rendered = _render_student(tokenizer, row.prompt)
    prompt_tokens = _token_count(tokenizer, rendered)
    total_tokens = _token_count(tokenizer, rendered + completion)
    if prompt_tokens > MAX_STUDENT_PROMPT_TOKENS:
        errors.append(f"prompt_tokens:{prompt_tokens}>{MAX_STUDENT_PROMPT_TOKENS}")
    if completion_tokens > MAX_COMPLETION_TOKENS:
        errors.append(f"completion_tokens:{completion_tokens}>{MAX_COMPLETION_TOKENS}")
    if total_tokens > MAX_TOTAL_TOKENS:
        errors.append(f"total_tokens:{total_tokens}>{MAX_TOTAL_TOKENS}")
    if response.finish_reason != "stop":
        errors.append(f"finish_reason:{response.finish_reason}")
    if errors:
        raise StageOutputError(errors)
    return {
        "reasoning": reasoning,
        "completion": completion,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


class ReleaseIdentityGuard:
    def __init__(self, output_root: Path):
        self._lock = threading.Lock()
        self._identity: tuple[str, str] | None = None
        self._response_ids: set[str] = set()
        self._identity_path = output_root / "provider_identity.json"
        if self._identity_path.is_file():
            payload = _load_json(self._identity_path)
            identity = (
                str(payload.get("returned_model") or ""),
                str(payload.get("system_fingerprint") or ""),
            )
            if not all(identity):
                raise ProviderDriftError("stored provider identity is incomplete")
            self._identity = identity

    @property
    def identity(self) -> tuple[str, str] | None:
        with self._lock:
            return self._identity

    def bind(self, response: TeacherResponse) -> None:
        if response.returned_model != TEACHER_MODEL:
            raise ProviderDriftError(
                f"returned model {response.returned_model!r} != {TEACHER_MODEL!r}"
            )
        if not response.system_fingerprint or not response.response_id:
            raise ProviderDriftError("provider fingerprint or response_id is missing")
        observed = (response.returned_model, response.system_fingerprint)
        with self._lock:
            if self._identity is None:
                self._identity = observed
            elif self._identity != observed:
                raise ProviderDriftError(
                    f"provider identity drift: {self._identity} != {observed}"
                )
            if response.response_id in self._response_ids:
                raise ProviderDriftError(
                    f"duplicate response_id: {response.response_id}"
                )
            self._response_ids.add(response.response_id)

    def seal(self) -> None:
        identity = self.identity
        if identity is None:
            return
        _write_json(
            self._identity_path,
            {"returned_model": identity[0], "system_fingerprint": identity[1]},
            immutable=True,
        )


def _contract(
    *,
    stage: str,
    system_prompt: str,
    repair_prompt: str,
    max_tokens: int,
    repair_attempts: int = 1,
) -> dict[str, Any]:
    if repair_attempts < 0:
        raise SupplementGenerationError("repair attempts cannot be negative")
    payload = {
        "schema_version": CONTRACT_SCHEMA,
        "stage": stage,
        "system_prompt": system_prompt,
        "repair_system_prompt": repair_prompt,
        "provider": ProviderConfig(max_tokens=max_tokens).contract(),
        "model_fallback": "forbidden",
        "repair_attempts": repair_attempts,
        "code_sha256": sha256_file(Path(__file__).resolve()),
    }
    payload["contract_sha256"] = sha256_text(canonical_json(payload))
    return payload


def _cache_key(stage: str, row: StageRow, contract: Mapping[str, Any]) -> str:
    return sha256_text(
        canonical_json(
            {
                "schema_version": CACHE_SCHEMA,
                "stage": stage,
                "sample_id": row.sample_id,
                "input_sha256": row.input_sha256,
                "prompt_sha256": row.prompt_sha256,
                "gold": row.gold if row.direction else None,
                "contract_sha256": contract["contract_sha256"],
            }
        )
    )


def _response_from_cache(payload: Mapping[str, Any]) -> TeacherResponse:
    provider = payload.get("provider") or {}
    raw = payload.get("provider_raw") or {}
    return TeacherResponse(
        reasoning=str(raw.get("reasoning_content") or ""),
        content=str(raw.get("content") or ""),
        response_id=str(provider.get("response_id") or ""),
        returned_model=str(provider.get("returned_model") or ""),
        system_fingerprint=str(provider.get("system_fingerprint") or ""),
        finish_reason=str(provider.get("finish_reason") or ""),
        created=provider.get("created"),
        usage=provider.get("usage") or {},
    )


def _run_one(
    row: StageRow,
    *,
    stage: str,
    output_root: Path,
    system_prompt: str,
    repair_prompt: str,
    max_tokens: int,
    contract: Mapping[str, Any],
    tokenizer: Any,
    validator: Callable[[StageRow, TeacherResponse, Any], dict[str, Any]],
    backend: ProviderBackend,
    guard: ReleaseIdentityGuard,
    environment: Mapping[str, str] | None,
) -> dict[str, Any]:
    cache_key = _cache_key(stage, row, contract)
    errors: tuple[str, ...] = ()
    repair_attempts = int(contract.get("repair_attempts") or 0)
    attempts = ("primary",) + tuple(
        "repair" if index == 1 else f"repair_{index}"
        for index in range(1, repair_attempts + 1)
    )
    for attempt in attempts:
        response: TeacherResponse | None = None
        try:
            response = backend.generate(
                config=ProviderConfig(max_tokens=max_tokens),
                system_prompt=system_prompt if attempt == "primary" else repair_prompt,
                user_prompt=(
                    row.prompt
                    if attempt == "primary"
                    else row.prompt
                    + "\n\nRegenerate the required JSON under the same source-only contract."
                ),
                environment=environment,
            )
            guard.bind(response)
            target = validator(row, response, tokenizer)
            payload = {
                "schema_version": CACHE_SCHEMA,
                "status": "accepted",
                "stage": stage,
                "cache_key": cache_key,
                "sample_id": row.sample_id,
                "input_sha256": row.input_sha256,
                "prompt_sha256": row.prompt_sha256,
                "gold_sha256": (
                    sha256_text(canonical_json(row.gold)) if row.direction else None
                ),
                "contract_sha256": contract["contract_sha256"],
                "attempt": attempt,
                "provider": {
                    "response_id": response.response_id,
                    "returned_model": response.returned_model,
                    "system_fingerprint": response.system_fingerprint,
                    "finish_reason": response.finish_reason,
                    "created": response.created,
                    "usage": dict(response.usage),
                },
                "provider_raw": {
                    "reasoning_content": response.reasoning,
                    "content": response.content,
                },
                "target": target,
            }
            _store_immutable(
                output_root
                / "cache"
                / stage
                / "accepted"
                / cache_key[:2]
                / f"{cache_key}.json",
                payload,
            )
            return payload
        except ProviderDriftError:
            raise
        except StageOutputError as exc:
            errors = exc.codes
            rejected = {
                "schema_version": CACHE_SCHEMA,
                "status": "rejected",
                "stage": stage,
                "sample_id": row.sample_id,
                "attempt": attempt,
                "errors": list(errors),
                "provider": None
                if response is None
                else {
                    "response_id": response.response_id,
                    "returned_model": response.returned_model,
                    "system_fingerprint": response.system_fingerprint,
                    "finish_reason": response.finish_reason,
                    "created": response.created,
                    "usage": dict(response.usage),
                },
                "provider_raw": None
                if response is None
                else {
                    "reasoning_content": response.reasoning,
                    "content": response.content,
                },
            }
            rejection_key = sha256_text(canonical_json(rejected))
            _store_immutable(
                output_root
                / "cache"
                / stage
                / "rejected"
                / cache_key[:2]
                / cache_key
                / f"{rejection_key}.json",
                rejected,
            )
        except Exception as exc:
            return {
                "status": "failed",
                "stage": stage,
                "sample_id": row.sample_id,
                "error": f"{type(exc).__name__}:{exc}",
            }
    return {
        "status": "failed",
        "stage": stage,
        "sample_id": row.sample_id,
        "error": ";".join(errors),
    }


def _compatible_prior_contract(
    prior: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    """Allow revalidation only when provider prompts/config are unchanged."""

    prior_payload = dict(prior)
    prior_digest = prior_payload.pop("contract_sha256", None)
    if prior_digest != sha256_text(canonical_json(prior_payload)):
        return False
    current_payload = dict(current)
    current_payload.pop("contract_sha256", None)
    prior_repairs = prior_payload.pop("repair_attempts", None)
    current_repairs = current_payload.pop("repair_attempts", None)
    prior_payload.pop("code_sha256", None)
    current_payload.pop("code_sha256", None)
    return (
        isinstance(prior_repairs, int)
        and isinstance(current_repairs, int)
        and 0 <= prior_repairs <= current_repairs
        and prior_payload == current_payload
    )


def _run_stage(
    *,
    stage: str,
    rows: Sequence[StageRow],
    output_root: Path,
    stage_root: Path,
    system_prompt: str,
    repair_prompt: str,
    max_tokens: int,
    tokenizer: Any,
    validator: Callable[[StageRow, TeacherResponse, Any], dict[str, Any]],
    concurrency: int,
    resume: bool,
    backend: ProviderBackend | None,
    environment: Mapping[str, str] | None,
    prompt_contract_file: str = "prompt_contract.json",
    repair_attempts: int = 1,
    revalidate_contract_files: Sequence[str] = (),
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if concurrency < 1:
        raise SupplementGenerationError("concurrency must be positive")
    contract = _contract(
        stage=stage,
        system_prompt=system_prompt,
        repair_prompt=repair_prompt,
        max_tokens=max_tokens,
        repair_attempts=repair_attempts,
    )
    if Path(prompt_contract_file).name != prompt_contract_file:
        raise SupplementGenerationError("prompt contract filename is unsafe")
    _write_json(stage_root / prompt_contract_file, contract, immutable=True)
    guard = ReleaseIdentityGuard(output_root)
    accepted: dict[str, dict[str, Any]] = {}
    accepted_root = output_root / "cache" / stage / "accepted"
    existing = list(accepted_root.glob("*/*.json"))
    if existing and not resume:
        raise SupplementGenerationError(f"{stage} cache exists; use --resume")
    for row in rows:
        cache_key = _cache_key(stage, row, contract)
        path = accepted_root / cache_key[:2] / f"{cache_key}.json"
        if not path.is_file():
            continue
        payload = _load_json(path)
        if any(
            (
                payload.get("schema_version") != CACHE_SCHEMA,
                payload.get("status") != "accepted",
                payload.get("stage") != stage,
                payload.get("sample_id") != row.sample_id,
                payload.get("input_sha256") != row.input_sha256,
                payload.get("prompt_sha256") != row.prompt_sha256,
                payload.get("contract_sha256") != contract["contract_sha256"],
            )
        ):
            raise SupplementGenerationError(
                f"resume cache binding mismatch: {row.sample_id}"
            )
        response = _response_from_cache(payload)
        guard.bind(response)
        if payload.get("target") != validator(row, response, tokenizer):
            raise SupplementGenerationError(f"resume target drift: {row.sample_id}")
        accepted[row.sample_id] = payload

    revalidated_count = 0
    if resume:
        for legacy_name in revalidate_contract_files:
            if len(accepted) == len(rows):
                break
            if (
                Path(legacy_name).name != legacy_name
                or legacy_name == prompt_contract_file
            ):
                raise SupplementGenerationError(
                    "legacy prompt contract filename is unsafe"
                )
            legacy_contract_path = stage_root / legacy_name
            legacy_contract = _load_json(legacy_contract_path)
            if not _compatible_prior_contract(legacy_contract, contract):
                raise SupplementGenerationError(
                    f"legacy prompt contract is incompatible: {legacy_name}"
                )
            for row in rows:
                if row.sample_id in accepted:
                    continue
                legacy_key = _cache_key(stage, row, legacy_contract)
                legacy_path = accepted_root / legacy_key[:2] / f"{legacy_key}.json"
                if not legacy_path.is_file():
                    continue
                legacy = _load_json(legacy_path)
                expected_gold_sha = (
                    sha256_text(canonical_json(row.gold)) if row.direction else None
                )
                if any(
                    (
                        legacy.get("schema_version") != CACHE_SCHEMA,
                        legacy.get("status") != "accepted",
                        legacy.get("stage") != stage,
                        legacy.get("cache_key") != legacy_key,
                        legacy.get("sample_id") != row.sample_id,
                        legacy.get("input_sha256") != row.input_sha256,
                        legacy.get("prompt_sha256") != row.prompt_sha256,
                        legacy.get("gold_sha256") != expected_gold_sha,
                        legacy.get("contract_sha256")
                        != legacy_contract["contract_sha256"],
                    )
                ):
                    raise SupplementGenerationError(
                        f"legacy cache binding mismatch: {row.sample_id}"
                    )
                response = _response_from_cache(legacy)
                guard.bind(response)
                target = validator(row, response, tokenizer)
                current_key = _cache_key(stage, row, contract)
                promoted = {
                    "schema_version": CACHE_SCHEMA,
                    "status": "accepted",
                    "stage": stage,
                    "cache_key": current_key,
                    "sample_id": row.sample_id,
                    "input_sha256": row.input_sha256,
                    "prompt_sha256": row.prompt_sha256,
                    "gold_sha256": expected_gold_sha,
                    "contract_sha256": contract["contract_sha256"],
                    "attempt": legacy.get("attempt"),
                    "provider": legacy.get("provider"),
                    "provider_raw": legacy.get("provider_raw"),
                    "target": target,
                    "revalidated_from": {
                        "path": str(legacy_path.relative_to(output_root)),
                        "sha256": sha256_file(legacy_path),
                        "contract_path": str(
                            legacy_contract_path.relative_to(output_root)
                        ),
                        "contract_sha256": legacy_contract["contract_sha256"],
                    },
                }
                current_path = accepted_root / current_key[:2] / f"{current_key}.json"
                _store_immutable(current_path, promoted)
                accepted[row.sample_id] = promoted
                revalidated_count += 1

    pending = [row for row in rows if row.sample_id not in accepted]
    provider = backend or OpenAICompatibleDeepSeekBackend()
    failures: list[dict[str, Any]] = []
    drift: BaseException | None = None
    completed = len(accepted)
    print(
        f"[chk4-supplement:{stage}] prepared={len(rows)} resumed={completed} pending={len(pending)}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _run_one,
                row,
                stage=stage,
                output_root=output_root,
                system_prompt=system_prompt,
                repair_prompt=repair_prompt,
                max_tokens=max_tokens,
                contract=contract,
                tokenizer=tokenizer,
                validator=validator,
                backend=provider,
                guard=guard,
                environment=environment,
            ): row
            for row in pending
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
                if result.get("status") == "accepted":
                    accepted[row.sample_id] = result
                    completed += 1
                    print(
                        f"[chk4-supplement:{stage}] accepted={completed}/{len(rows)}",
                        flush=True,
                    )
                else:
                    failures.append(dict(result))
            except ProviderDriftError as exc:
                drift = exc
                for other in futures:
                    other.cancel()
                break
            except Exception as exc:
                failures.append(
                    {
                        "status": "failed",
                        "stage": stage,
                        "sample_id": row.sample_id,
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
    if drift is not None:
        raise ProviderDriftError(str(drift))
    guard.seal()
    summary = {
        "schema_version": f"chk4-decision-supplement-{stage}-summary-v1",
        "status": "complete"
        if len(accepted) == len(rows) and not failures
        else "incomplete",
        "stage": stage,
        "prepared_count": len(rows),
        "accepted_count": len(accepted),
        "failure_count": len(failures),
        "resumed_count": len(rows) - len(pending),
        "revalidated_count": revalidated_count,
        "api_requests": len(pending),
        "contract_sha256": contract["contract_sha256"],
        "prompt_contract_file": prompt_contract_file,
        "provider_identity": guard.identity,
    }
    _write_jsonl(stage_root / "failures.jsonl", failures)
    _write_json(stage_root / "summary.json", summary)
    _refresh_root_failures(output_root)
    _update_root_summary(output_root, stage, summary)
    if summary["status"] != "complete":
        raise SupplementGenerationError(
            f"{stage} incomplete: {len(accepted)}/{len(rows)} accepted"
        )
    return accepted, summary


def _refresh_root_failures(output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("failures.jsonl")):
        if path == output_root / "failures.jsonl":
            continue
        rows.extend(_load_jsonl(path))
    _write_jsonl(output_root / "failures.jsonl", rows)


def _evidence_profile_binding(output_root: Path) -> dict[str, Any]:
    audit_path = output_root / "reports/evidence_audit.json"
    audit = _load_json(audit_path)
    admitted_path = output_root / "manifests/admitted.jsonl"
    admitted_count = len(_load_jsonl(admitted_path))
    if any(
        (
            audit.get("schema_version") != "chk4-decision-supplement-evidence-audit-v2",
            audit.get("status") != "complete",
            audit.get("admission_profile") != ADMISSION_PROFILE,
            admitted_count < 1,
            audit.get("admitted_count") != admitted_count,
            audit.get("candidate_count") != admitted_count,
            audit.get("minimum_admitted") != admitted_count,
            audit.get("rejected_count") != 0,
            audit.get("minimum_valid_atomic_topics") != MIN_VALID_ATOMIC_TOPICS,
        )
    ):
        raise SupplementGenerationError("historical sparse evidence profile drift")
    handoff_path = output_root / "sources/materialized/source_handoff.json"
    handoff = _load_json(handoff_path)
    if handoff.get("schema_version") != "chk1-source-handoff-v1" or handoff.get(
        "payload_sha256"
    ) != audit.get("source_handoff_payload_sha256"):
        raise SupplementGenerationError("source handoff/evidence profile drift")
    return {
        "admission_profile": ADMISSION_PROFILE,
        "evidence_audit": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "schema_version": audit["schema_version"],
        },
        "source_handoff": {
            "path": str(handoff_path),
            "sha256": sha256_file(handoff_path),
            "payload_sha256": handoff["payload_sha256"],
        },
        "admitted_manifest": {
            "path": str(admitted_path),
            "sha256": sha256_file(admitted_path),
            "rows": admitted_count,
        },
    }


def _summary_rows(output_root: Path) -> list[StageRow]:
    _evidence_profile_binding(output_root)
    rows = []
    for item in _load_jsonl(output_root / "prepared/summary_requests.jsonl"):
        rows.append(
            StageRow(
                sample_id=str(item["sample_id"]),
                prompt=str(item["prompt"]),
                input_text=str(item["prompt"]),
                input_sha256=str(item["input_sha256"]),
                prompt_sha256=str(item["prompt_sha256"]),
            )
        )
    admitted = _load_jsonl(output_root / "manifests/admitted.jsonl")
    if (
        len(rows) != len(admitted)
        or len({row.sample_id for row in rows}) != len(rows)
        or any(row.get("admission_profile") != ADMISSION_PROFILE for row in admitted)
    ):
        raise SupplementGenerationError("summary request/admission closure failed")
    return rows


def run_summaries(
    *,
    output_root: Path,
    tokenizer_path: Path,
    concurrency: int,
    resume: bool,
    backend: ProviderBackend | None = None,
    environment: Mapping[str, str] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    rows = _summary_rows(output_root)
    tokenizer = tokenizer or _load_tokenizer(tokenizer_path)
    stage_root = output_root / "summaries"
    accepted, summary = _run_stage(
        stage="summaries",
        rows=rows,
        output_root=output_root,
        stage_root=stage_root,
        system_prompt=SUMMARY_SYSTEM_PROMPT,
        repair_prompt=SUMMARY_REPAIR_SYSTEM_PROMPT,
        max_tokens=SUMMARY_MAX_TOKENS,
        tokenizer=tokenizer,
        validator=_validate_summary,
        concurrency=concurrency,
        resume=resume,
        backend=backend,
        environment=environment,
        prompt_contract_file=SUMMARY_PROMPT_CONTRACT_FILE,
        repair_attempts=2,
        revalidate_contract_files=("prompt_contract.grounding-v5.json",),
    )
    summary = {**summary, **_evidence_profile_binding(output_root)}
    _write_json(stage_root / "summary.json", summary)
    output_rows = []
    teacher_rows = []
    for row in rows:
        payload = accepted[row.sample_id]
        target = payload["target"]
        output_rows.append(
            {
                "schema_version": SUMMARY_SCHEMA,
                "sample_id": row.sample_id,
                **target,
                "input_sha256": row.input_sha256,
                "generation_contract_sha256": summary["contract_sha256"],
            }
        )
        teacher_rows.append(
            {
                "sample_id": row.sample_id,
                "attempt": payload["attempt"],
                "provider": payload["provider"],
                "reasoning_content": payload["provider_raw"]["reasoning_content"],
                "content": payload["provider_raw"]["content"],
                "revalidated_from": payload.get("revalidated_from"),
            }
        )
    _write_jsonl(stage_root / "meeting_decision_briefs.jsonl", output_rows)
    _write_jsonl(stage_root / "teacher_responses.jsonl", teacher_rows)
    return summary


def _brief_map(output_root: Path) -> dict[str, dict[str, Any]]:
    rows = _load_jsonl(output_root / "summaries/meeting_decision_briefs.jsonl")
    result = {str(row.get("sample_id") or ""): row for row in rows}
    if len(result) != len(rows) or not result:
        raise SupplementGenerationError("saved summaries are empty or duplicated")
    return result


def _decision_rows(output_root: Path, *, include_gold: bool) -> list[StageRow]:
    briefs = _brief_map(output_root)
    if not include_gold:
        return [
            StageRow(
                sample_id=sample_id,
                prompt=BLIND_USER_PREFIX
                + canonical_json(
                    {"analysis": str(briefs[sample_id]["meeting_decision_brief"])}
                ),
                input_text=str(briefs[sample_id]["meeting_decision_brief"]),
                input_sha256=sha256_text(
                    str(briefs[sample_id]["meeting_decision_brief"])
                ),
                prompt_sha256=sha256_text(
                    BLIND_USER_PREFIX
                    + canonical_json(
                        {"analysis": str(briefs[sample_id]["meeting_decision_brief"])}
                    )
                ),
            )
            for sample_id in sorted(briefs)
        ]
    manifests = {
        str(row["sample_id"]): row
        for row in _load_jsonl(output_root / "manifests/admitted.jsonl")
    }
    if set(briefs) != set(manifests):
        raise SupplementGenerationError("summary/admitted manifest closure failed")
    rows: list[StageRow] = []
    for sample_id in sorted(briefs):
        analysis = str(briefs[sample_id]["meeting_decision_brief"])
        student_prompt = render_student_prompt(analysis)
        manifest = manifests[sample_id]
        gold = manifest.get("gold") or {}
        teacher_prompt = student_prompt
        direction = ""
        magnitude = -1
        direction = str(gold.get("direction") or "")
        magnitude = int(gold.get("magnitude_bp"))
        teacher_prompt += TARGET_GOLD_PREFIX + _canonical_gold(direction, magnitude)
        rows.append(
            StageRow(
                sample_id=sample_id,
                prompt=teacher_prompt,
                input_text=analysis,
                input_sha256=sha256_text(analysis),
                prompt_sha256=sha256_text(student_prompt),
                meeting_date=str(manifest["meeting_date"]),
                direction=direction,
                magnitude_bp=magnitude,
                source_ids=tuple(str(value) for value in manifest["source_ids"]),
            )
        )
    return rows


def run_blind_predictions(
    *,
    output_root: Path,
    tokenizer_path: Path,
    concurrency: int,
    resume: bool,
    backend: ProviderBackend | None = None,
    environment: Mapping[str, str] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    # Only saved summaries are read to construct the provider request. Gold and
    # meeting manifests are joined later by the separate audit command.
    rows = _decision_rows(output_root, include_gold=False)
    tokenizer = tokenizer or _load_tokenizer(tokenizer_path)
    stage_root = output_root / "blind_predictions"
    accepted, summary = _run_stage(
        stage="blind_predictions",
        rows=rows,
        output_root=output_root,
        stage_root=stage_root,
        system_prompt=BLIND_SYSTEM_PROMPT,
        repair_prompt=BLIND_REPAIR_SYSTEM_PROMPT,
        max_tokens=DECISION_MAX_TOKENS,
        tokenizer=tokenizer,
        validator=_validate_blind,
        concurrency=concurrency,
        resume=resume,
        backend=backend,
        environment=environment,
    )
    summary = {**summary, **_evidence_profile_binding(output_root)}
    _write_json(stage_root / "summary.json", summary)
    predictions = []
    teacher_rows = []
    for row in rows:
        payload = accepted[row.sample_id]
        predictions.append(
            {
                "schema_version": BLIND_SCHEMA,
                "sample_id": row.sample_id,
                **payload["target"],
                "analysis_sha256": row.input_sha256,
                "contract_sha256": summary["contract_sha256"],
            }
        )
        teacher_rows.append(
            {
                "sample_id": row.sample_id,
                "attempt": payload["attempt"],
                "provider": payload["provider"],
                "reasoning_content": payload["provider_raw"]["reasoning_content"],
                "content": payload["provider_raw"]["content"],
            }
        )
    _write_jsonl(stage_root / "predictions.jsonl", predictions)
    _write_jsonl(stage_root / "teacher_responses.jsonl", teacher_rows)
    return summary


def _signed(direction: str, magnitude: int) -> int:
    return -magnitude if direction == "cut" else magnitude if direction == "hike" else 0


def _direction_metrics(gold: Sequence[str], predicted: Sequence[str]) -> dict[str, Any]:
    labels = ("cut", "hold", "hike")
    matrix = {actual: {guess: 0 for guess in labels} for actual in labels}
    for actual, guess in zip(gold, predicted, strict=True):
        matrix[actual][guess] += 1
    f1_values = []
    recalls = []
    for label in labels:
        true_positive = matrix[label][label]
        false_positive = sum(
            matrix[actual][label] for actual in labels if actual != label
        )
        false_negative = sum(matrix[label][guess] for guess in labels if guess != label)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1_values.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        recalls.append(recall)
    return {
        "macro_f1": sum(f1_values) / len(f1_values),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "confusion_matrix": matrix,
    }


def audit_blind_predictions(output_root: Path) -> dict[str, Any]:
    manifests = sorted(
        _load_jsonl(output_root / "manifests/admitted.jsonl"),
        key=lambda row: str(row["meeting_date"]),
    )
    predictions = {
        str(row["sample_id"]): row
        for row in _load_jsonl(output_root / "blind_predictions/predictions.jsonl")
    }
    if set(predictions) != {str(row["sample_id"]) for row in manifests}:
        raise SupplementGenerationError("blind prediction/admission closure failed")
    gold_direction = [str(row["gold"]["direction"]) for row in manifests]
    predicted_direction = [
        str(predictions[str(row["sample_id"])]["direction"]) for row in manifests
    ]
    exact = []
    absolute_errors = []
    action_matrix: dict[str, Counter[str]] = {}
    gold_actions = []
    for row in manifests:
        gold = row["gold"]
        prediction = predictions[str(row["sample_id"])]
        gold_action = f"{gold['direction']}:{gold['magnitude_bp']}"
        predicted_action = f"{prediction['direction']}:{prediction['magnitude_bp']}"
        gold_actions.append(gold_action)
        exact.append(gold_action == predicted_action)
        absolute_errors.append(
            abs(
                _signed(str(gold["direction"]), int(gold["magnitude_bp"]))
                - _signed(str(prediction["direction"]), int(prediction["magnitude_bp"]))
            )
        )
        action_matrix.setdefault(gold_action, Counter())[predicted_action] += 1
    majority_action = Counter(gold_actions).most_common(1)[0][0]
    lag_correct = [
        gold_actions[index - 1] == gold_actions[index]
        for index in range(1, len(gold_actions))
    ]
    metrics = {
        "schema_version": "chk4-decision-supplement-blind-audit-v1",
        "status": "complete",
        "sample_count": len(manifests),
        "exact_action_accuracy": sum(exact) / len(exact),
        **_direction_metrics(gold_direction, predicted_direction),
        "bp_mae_signed_action": sum(absolute_errors) / len(absolute_errors),
        "exact_action_confusion_matrix": {
            actual: dict(sorted(counter.items()))
            for actual, counter in sorted(action_matrix.items())
        },
        "majority_baseline": {
            "action": majority_action,
            "exact_action_accuracy": gold_actions.count(majority_action)
            / len(gold_actions),
        },
        "lag1_baseline": {
            "evaluated_count": len(lag_correct),
            "exact_action_accuracy": (
                sum(lag_correct) / len(lag_correct) if lag_correct else None
            ),
        },
        "selection_policy": "audit_only_never_filter_training_rows",
    }
    _write_json(output_root / "reports/blind_prediction_metrics.json", metrics)
    _update_root_summary(output_root, "blind_audit", metrics)
    return metrics


def run_teacher_targets(
    *,
    output_root: Path,
    tokenizer_path: Path,
    concurrency: int,
    resume: bool,
    backend: ProviderBackend | None = None,
    environment: Mapping[str, str] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    evidence_binding = _evidence_profile_binding(output_root)
    rows = _decision_rows(output_root, include_gold=True)
    tokenizer = tokenizer or _load_tokenizer(tokenizer_path)
    stage_root = output_root / "teacher_targets/supplement"
    accepted, summary = _run_stage(
        stage="teacher_targets",
        rows=rows,
        output_root=output_root,
        stage_root=stage_root,
        system_prompt=TARGET_SYSTEM_PROMPT,
        repair_prompt=TARGET_REPAIR_SYSTEM_PROMPT,
        max_tokens=DECISION_MAX_TOKENS,
        tokenizer=tokenizer,
        validator=_validate_target,
        concurrency=concurrency,
        resume=resume,
        backend=backend,
        environment=environment,
    )
    prepared_rows = []
    manifest_rows = []
    sft_rows = []
    teacher_rows = []
    for row in rows:
        payload = accepted[row.sample_id]
        student_prompt = render_student_prompt(row.input_text)
        gold_text = _canonical_gold(row.direction, row.magnitude_bp)
        prepared_rows.append(
            {
                "schema_version": TARGET_SCHEMA,
                "sample_id": row.sample_id,
                "prompt": student_prompt,
                "input_sha256": row.input_sha256,
                "prompt_sha256": sha256_text(student_prompt),
                "gold_sha256": sha256_text(gold_text),
            }
        )
        manifest_rows.append(
            {
                "schema_version": MANIFEST_SCHEMA,
                "sample_id": row.sample_id,
                "meeting_date": row.meeting_date,
                "split": "train",
                "population": POPULATION,
                "population_role": "supplement",
                "admission_profile": evidence_binding["admission_profile"],
                "evidence_audit_sha256": evidence_binding["evidence_audit"]["sha256"],
                "source_handoff_payload_sha256": evidence_binding["source_handoff"][
                    "payload_sha256"
                ],
                "source_ids": list(row.source_ids),
                "gold": row.gold,
                "input_sha256": row.input_sha256,
                "prompt_sha256": sha256_text(student_prompt),
                "gold_sha256": sha256_text(gold_text),
                "contract_sha256": summary["contract_sha256"],
            }
        )
        sft_rows.append(
            {"prompt": student_prompt, "response": payload["target"]["completion"]}
        )
        teacher_rows.append(
            {
                "sample_id": row.sample_id,
                "attempt": payload["attempt"],
                "provider": payload["provider"],
                "reasoning_content": payload["provider_raw"]["reasoning_content"],
                "content": payload["provider_raw"]["content"],
            }
        )
    for split in ("train", "validation", "test"):
        is_train = split == "train"
        _write_jsonl(
            stage_root / "prepared" / f"{split}.jsonl",
            prepared_rows if is_train else [],
        )
        _write_jsonl(
            stage_root / "manifests" / f"{split}.jsonl",
            manifest_rows if is_train else [],
        )
        _write_jsonl(
            stage_root / "sft" / f"{split}.jsonl", sft_rows if is_train else []
        )
        _write_jsonl(
            stage_root / "teacher_responses" / f"{split}.jsonl",
            teacher_rows if is_train else [],
        )
    target_summary = {
        **summary,
        **evidence_binding,
        "teacher_output_mapping": {
            "reasoning": "json.loads(message.content)['reasoning']",
            "decision": "locally_serialized_canonical_gold",
            "native_reasoning_content": "provenance_only",
        },
    }
    _write_json(stage_root / "summary.json", target_summary)
    return target_summary


def _paired_rows(root: Path, split: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    manifests = _load_jsonl(root / "manifests" / f"{split}.jsonl")
    sft = _load_jsonl(root / "sft" / f"{split}.jsonl")
    if len(manifests) != len(sft):
        raise SupplementGenerationError(f"manifest/SFT mismatch in {root}:{split}")
    return list(zip(manifests, sft, strict=True))


def combine_and_materialize(
    *, output_root: Path, core_teacher_root: Path, resume: bool
) -> dict[str, Any]:
    evidence_binding = _evidence_profile_binding(output_root)
    supplement_root = output_root / "teacher_targets/supplement"
    core_rows = load_unique_rows(core_teacher_root)
    supplement_rows = load_unique_rows(supplement_root)
    if {split: len(rows) for split, rows in core_rows.items()} != {
        "train": 102,
        "validation": 13,
        "test": 13,
    }:
        raise SupplementGenerationError("core teacher release counts changed")
    if supplement_rows["validation"] or supplement_rows["test"]:
        raise SupplementGenerationError("supplement leaked outside train")
    if len(supplement_rows["train"]) < MIN_ADMITTED:
        raise SupplementGenerationError(
            "supplement teacher release is below admission minimum"
        )
    supplement_summary = _load_json(supplement_root / "summary.json")
    if any(
        (
            supplement_summary.get("admission_profile") != ADMISSION_PROFILE,
            supplement_summary.get("evidence_audit")
            != evidence_binding["evidence_audit"],
            supplement_summary.get("source_handoff")
            != evidence_binding["source_handoff"],
            supplement_summary.get("admitted_manifest")
            != evidence_binding["admitted_manifest"],
        )
    ):
        raise SupplementGenerationError("supplement teacher admission binding drift")

    combined_root = output_root / "teacher_targets/combined_v1"
    for split in ("train", "validation", "test"):
        pairs = _paired_rows(core_teacher_root, split)
        if split == "train":
            pairs += _paired_rows(supplement_root, split)
        pairs.sort(key=lambda pair: str(pair[0]["sample_id"]))
        _write_jsonl(
            combined_root / "manifests" / f"{split}.jsonl",
            [pair[0] for pair in pairs],
            immutable=True,
        )
        _write_jsonl(
            combined_root / "sft" / f"{split}.jsonl",
            [pair[1] for pair in pairs],
            immutable=True,
        )
    composite_contract = sha256_text(
        canonical_json(
            {
                "core": _load_json(core_teacher_root / "summary.json").get(
                    "contract_sha256"
                ),
                "supplement": _load_json(supplement_root / "summary.json").get(
                    "contract_sha256"
                ),
            }
        )
    )
    combined_summary = {
        "schema_version": COMBINED_SCHEMA,
        "status": "complete",
        "contract_sha256": composite_contract,
        **evidence_binding,
        "core_teacher_root": str(core_teacher_root),
        "core_teacher_summary_sha256": sha256_file(core_teacher_root / "summary.json"),
        "supplement_teacher_root": str(supplement_root),
        "supplement_teacher_summary_sha256": sha256_file(
            supplement_root / "summary.json"
        ),
        "unique_counts": {
            "train": len(core_rows["train"]) + len(supplement_rows["train"]),
            "validation": len(core_rows["validation"]),
            "test": len(core_rows["test"]),
        },
    }
    _write_json(combined_root / "summary.json", combined_summary, immutable=True)
    training_root = output_root / "training/combined_v1"
    training_summary = materialize(
        teacher_root=combined_root,
        output_root=training_root,
        allow_existing=resume and training_root.exists(),
    )
    result = {
        "schema_version": "chk4-decision-supplement-combined-materialization-v1",
        "status": "complete",
        "combined_teacher_summary": combined_summary,
        "training_summary": training_summary,
    }
    _write_json(output_root / "reports/training_materialization.json", result)
    _update_root_summary(output_root, "materialization", result)
    return result


def release_qa(output_root: Path) -> dict[str, Any]:
    evidence_binding = _evidence_profile_binding(output_root)
    evidence = _load_json(output_root / "reports/evidence_audit.json")
    blind = _load_json(output_root / "reports/blind_prediction_metrics.json")
    target = _load_json(output_root / "teacher_targets/supplement/summary.json")
    training = _load_json(output_root / "training/combined_v1/summary.json")
    admitted = _load_jsonl(output_root / "manifests/admitted.jsonl")
    failures = _load_jsonl(output_root / "failures.jsonl")
    errors: list[str] = []
    if (
        evidence.get("status") != "complete"
        or evidence.get("admission_profile") != ADMISSION_PROFILE
        or len(admitted) != MIN_ADMITTED
        or any(row.get("admission_profile") != ADMISSION_PROFILE for row in admitted)
    ):
        errors.append("evidence_admission")
    actions = {
        f"{row['gold']['direction']}:{row['gold']['magnitude_bp']}" for row in admitted
    }
    if actions != set(EXPECTED_ACTION_COUNTS):
        errors.append("action_class_retention")
    if any(
        set(row.get("category_coverage") or ()) < REQUIRED_CATEGORIES
        for row in admitted
    ):
        errors.append("category_coverage")
    if blind.get("status") != "complete" or blind.get("sample_count") != len(admitted):
        errors.append("blind_audit_closure")
    if (
        target.get("status") != "complete"
        or target.get("accepted_count") != len(admitted)
        or target.get("admission_profile") != ADMISSION_PROFILE
        or target.get("evidence_audit") != evidence_binding["evidence_audit"]
        or target.get("source_handoff") != evidence_binding["source_handoff"]
        or target.get("admitted_manifest") != evidence_binding["admitted_manifest"]
    ):
        errors.append("target_closure")
    unique = training.get("unique_counts") or {}
    if unique != {"train": 102 + len(admitted), "validation": 13, "test": 13}:
        errors.append("combined_split_counts")
    if failures:
        errors.append("unresolved_failures")
    response_ids: set[str] = set()
    identity = None
    for path in (
        output_root / "summaries/teacher_responses.jsonl",
        output_root / "blind_predictions/teacher_responses.jsonl",
        output_root / "teacher_targets/supplement/teacher_responses/train.jsonl",
    ):
        for row in _load_jsonl(path):
            provider = row.get("provider") or {}
            response_id = str(provider.get("response_id") or "")
            observed = (
                str(provider.get("returned_model") or ""),
                str(provider.get("system_fingerprint") or ""),
            )
            if not response_id or response_id in response_ids:
                errors.append("duplicate_or_empty_response_id")
            response_ids.add(response_id)
            if identity is None:
                identity = observed
            elif identity != observed:
                errors.append("provider_identity_drift")
    report = {
        "schema_version": "chk4-decision-supplement-release-qa-v2",
        "status": "complete" if not errors else "blocked",
        "admission_profile": ADMISSION_PROFILE,
        "evidence_audit_sha256": evidence_binding["evidence_audit"]["sha256"],
        "source_handoff_payload_sha256": evidence_binding["source_handoff"][
            "payload_sha256"
        ],
        "errors": sorted(set(errors)),
        "candidate_count": 109,
        "admitted_count": len(admitted),
        "rejected_count": 109 - len(admitted),
        "action_classes": sorted(actions),
        "blind_metrics_are_audit_only": True,
        "combined_unique_counts": unique,
        "combined_physical_counts": training.get("physical_counts"),
        "provider_response_count": len(response_ids),
        "provider_identity": identity,
    }
    _write_json(output_root / "reports/release_qa.json", report)
    _update_root_summary(output_root, "qa", report)
    if errors:
        raise SupplementGenerationError(
            "release QA blocked: " + ",".join(sorted(set(errors)))
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--resume", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("summarize")
    subparsers.add_parser("blind-predict")
    subparsers.add_parser("audit")
    subparsers.add_parser("teacher-targets")
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument(
        "--core-teacher-root", type=Path, default=DEFAULT_CORE_TEACHER_ROOT
    )
    subparsers.add_parser("qa")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output_root = args.output_root.expanduser().resolve()
        tokenizer_path = args.tokenizer_path.expanduser().resolve()
        if args.command == "summarize":
            result = run_summaries(
                output_root=output_root,
                tokenizer_path=tokenizer_path,
                concurrency=args.concurrency,
                resume=args.resume,
            )
        elif args.command == "blind-predict":
            result = run_blind_predictions(
                output_root=output_root,
                tokenizer_path=tokenizer_path,
                concurrency=args.concurrency,
                resume=args.resume,
            )
        elif args.command == "audit":
            result = audit_blind_predictions(output_root)
        elif args.command == "teacher-targets":
            result = run_teacher_targets(
                output_root=output_root,
                tokenizer_path=tokenizer_path,
                concurrency=args.concurrency,
                resume=args.resume,
            )
        elif args.command == "materialize":
            result = combine_and_materialize(
                output_root=output_root,
                core_teacher_root=args.core_teacher_root.expanduser().resolve(),
                resume=args.resume,
            )
        else:
            result = release_qa(output_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (
        SupplementGenerationError,
        MaterializationError,
        OSError,
        ValueError,
    ) as exc:
        print(
            json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False),
            file=os.sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ProviderConfig",
    "QUALITATIVE_SPELLED_NUMBER_RE",
    "StageOutputError",
    "SupplementGenerationError",
    "audit_blind_predictions",
    "combine_and_materialize",
    "release_qa",
    "qualitative_reasoning_has_number_or_date",
    "run_blind_predictions",
    "run_summaries",
    "run_teacher_targets",
]
