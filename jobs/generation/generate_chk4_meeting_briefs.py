"""Generate gold-blind meeting Decision briefs with DeepSeek V4 Pro.

The source is the immutable chk1 canonical release.  Atomic final answers are
grouped by meeting, but meeting identity and every policy label are removed
before the model call.  DeepSeek receives only ``atomic_topic`` and
``analysis`` and returns one target-neutral ``meeting_decision_brief``.

This stage is deliberately separate from chk4 target acquisition: it never
loads the Decision workbook, FFR history, Minutes, current rates, or gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from jobs.generation.generate_chk4_sft_targets import (
    API_KEY_ENV,
    STUDENT_SYSTEM_PROMPT,
    TEACHER_BASE_URL,
    TEACHER_MODEL,
    Chk4TargetError,
    ModelDriftError,
    OpenAICompatibleDeepSeekBackend,
    ProviderIdentityGuard,
    TeacherResponse,
    canonical_json,
    sha256_file,
    sha256_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHK1_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk1/canonical_releases/"
    "chk1_full_v7_automated_v2_20260804"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "output/data/retrain_v2/chk4/meeting_decision_briefs_v1"
)
DEFAULT_TOKENIZER = REPO_ROOT / "models/DeepSeek-R1-Distill-Llama-8B"

SOURCE_SPLITS = {"train": "train", "eval": "validation", "test": "test"}
EXPECTED_ATOMIC_COUNTS = {"train": 1683, "validation": 199, "test": 190}
EXPECTED_MEETING_COUNTS = {"train": 102, "validation": 13, "test": 13}
MAX_TEACHER_TOKENS = 4096
MAX_STUDENT_PROMPT_TOKENS = 2560
DEFAULT_CONCURRENCY = 8
SCHEMA = "chk4-deepseek-meeting-brief-v1"
CACHE_SCHEMA = "chk4-deepseek-meeting-brief-cache-v1"
CONTRACT_SCHEMA = "chk4-deepseek-brief-contract-v1"

SYSTEM_PROMPT = """\
You are a macroeconomic synthesis analyst preparing a target-neutral,
pre-meeting FOMC decision brief. Use only the supplied collection of atomic
topic analyses. In reasoning_content, reconcile overlap and identify the
balance of evidence across inflation, employment and real activity, financial
conditions, and risks.

Do not infer or mention the meeting identity. Do not use outside or remembered
historical information. Do not state, recommend, predict, or imply a policy
decision, vote, rate change, target range, or action actually taken. Do not
mention gold labels, Minutes, teacher targets, hidden fields, prompts, or
schemas. Preserve material directions, comparisons, uncertainty, dates, and
quantities from the supplied analyses, but do not introduce new facts,
causes, numbers, or dates. Do not reproduce internal evidence IDs.

Return content as exactly one JSON object with the single key
meeting_decision_brief. Its value must be coherent formal prose, without
headings, lists, recommendations, policy actions, citations, or JSON embedded
inside the prose. Aim for a concise brief that leaves enough room for a
separate reasoning-and-decision completion in a 3,072-token student context.
"""

REPAIR_SYSTEM_PROMPT = """\
Regenerate the target-neutral pre-meeting synthesis using only the supplied
atomic topic analyses and silently correct the supplied contract errors. Do
not discuss validation. Do not infer or mention meeting identity; do not state,
recommend, predict, or imply a policy decision, vote, rate change, target
range, or action actually taken. Add no fact, cause, number, or date absent
from the supplied analyses. Remove headings, lists, citations, internal IDs,
and model-control text.

Return content as exactly one JSON object with the single key
meeting_decision_brief. This is the only repair attempt.
"""

USER_PREFIX = (
    "Synthesize the following point-in-time atomic analyses into one "
    "target-neutral pre-meeting brief:\n\n"
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
POLICY_LEAK_PATTERNS = (
    re.compile(r"\b(?:Committee|FOMC)\s+(?:voted|decided|raised|cut|held)\b", re.I),
    re.compile(r"\b(?:recommend|recommended|should|would)\s+(?:cut|raise|hike|hold)\b", re.I),
    re.compile(r"\b(?:actual|historical|known)\s+(?:decision|action|vote)\b", re.I),
    re.compile(r"\b(?:the\s+)?(?:decision|vote)\s+was\b", re.I),
    re.compile(r"\b(?:cut|raise|hike|hold)\s+(?:the\s+)?(?:policy|federal funds|target)\s+rate\b", re.I),
)
EVIDENCE_ID_RE = re.compile(r"\bev-[0-9a-f]{6,}\b", re.I)
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![A-Za-z0-9])")
DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\b|\b(?:19|20)\d{2}\b|\b\d{4}-\d{2}-\d{2}\b",
    re.I,
)

PRICE_WORDS = ("price", "inflation", "cpi", "pce", "commodity")
ACTIVITY_WORDS = (
    "employment",
    "unemployment",
    "labor",
    "job",
    "payroll",
    "gdp",
    "output",
    "production",
    "spending",
    "sales",
    "housing",
    "income",
)
FINANCIAL_WORDS = (
    "interest",
    "federal funds",
    "treasury",
    "credit",
    "bank",
    "financial",
    "dollar",
    "exchange",
    "equity",
    "mortgage",
    "money",
)


class BriefGenerationError(Chk4TargetError):
    """Gold-blind brief preparation or provider output is invalid."""


class BriefOutputError(BriefGenerationError):
    def __init__(self, codes: Sequence[str]):
        self.codes = tuple(str(code) for code in codes)
        super().__init__(";".join(self.codes))


@dataclass(frozen=True)
class BriefTeacherConfig:
    model: str = TEACHER_MODEL
    base_url: str = TEACHER_BASE_URL
    max_tokens: int = MAX_TEACHER_TOKENS
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
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "fallback": "forbidden",
            "repair_attempts": 1,
            "api_key_env": API_KEY_ENV,
        }


class BriefBackend(Protocol):
    def generate(
        self,
        *,
        config: BriefTeacherConfig,
        system_prompt: str,
        user_prompt: str,
        environment: Mapping[str, str] | None,
    ) -> TeacherResponse: ...


@dataclass(frozen=True)
class MeetingInput:
    sample_id: str
    split: str
    meeting_date: str
    atomic: tuple[dict[str, str], ...]
    source_ids: tuple[str, ...]
    user_prompt: str
    input_sha256: str
    prompt_sha256: str
    category_coverage: tuple[str, ...]


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write(path, "".join(canonical_json(row) + "\n" for row in rows))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BriefGenerationError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise BriefGenerationError(f"JSON root is not an object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BriefGenerationError(f"source file is missing: {path}")
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BriefGenerationError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise BriefGenerationError(
                    f"JSONL row is not an object: {path}:{line_number}"
                )
            result.append(value)
    return result


def _final_answer(response: str) -> str:
    if not isinstance(response, str) or response.count("</think>") != 1:
        raise BriefGenerationError("chk1 response must contain one </think>")
    reasoning, answer = response.split("</think>", 1)
    if not reasoning.strip() or not answer.strip():
        raise BriefGenerationError("chk1 response has an empty semantic section")
    if any(marker in answer for marker in ("<think>", "<answer>", "</answer>")):
        raise BriefGenerationError("chk1 final answer contains a control marker")
    return answer.strip()


def _categories(topics: Sequence[str]) -> tuple[str, ...]:
    text = " ".join(topics).lower()
    result: list[str] = []
    if any(word in text for word in PRICE_WORDS):
        result.append("prices")
    if any(word in text for word in ACTIVITY_WORDS):
        result.append("employment_activity")
    if any(word in text for word in FINANCIAL_WORDS):
        result.append("financial_conditions")
    return tuple(result)


def load_meeting_inputs(chk1_root: Path) -> dict[str, list[MeetingInput]]:
    grouped: dict[str, list[MeetingInput]] = {
        split: [] for split in EXPECTED_MEETING_COUNTS
    }
    all_meetings: set[str] = set()
    for source_split, output_split in SOURCE_SPLITS.items():
        manifests = _load_jsonl(chk1_root / "manifests" / f"{source_split}.jsonl")
        sft = _load_jsonl(chk1_root / "sft" / f"{source_split}.jsonl")
        if len(manifests) != EXPECTED_ATOMIC_COUNTS[output_split]:
            raise BriefGenerationError(
                f"{output_split} atomic count changed: {len(manifests)}"
            )
        if len(manifests) != len(sft):
            raise BriefGenerationError(f"{output_split} manifest/SFT mismatch")
        meetings: dict[str, list[dict[str, str]]] = {}
        source_ids: dict[str, list[str]] = {}
        for manifest, training in zip(manifests, sft, strict=True):
            meeting = str(manifest.get("meeting_date") or "").strip()
            topic = str(manifest.get("atomic_topic") or "").strip()
            sample_id = str(manifest.get("sample_id") or "").strip()
            response = str(training.get("response") or "")
            answer = _final_answer(response)
            if not meeting or not topic or not sample_id:
                raise BriefGenerationError("chk1 manifest identity is incomplete")
            if manifest.get("response_sha256") != sha256_text(response):
                raise BriefGenerationError(f"chk1 response SHA drift: {sample_id}")
            if manifest.get("final_analysis_sha256") != sha256_text(answer):
                raise BriefGenerationError(f"chk1 final-answer SHA drift: {sample_id}")
            meetings.setdefault(meeting, []).append(
                {"atomic_topic": topic, "analysis": answer}
            )
            source_ids.setdefault(meeting, []).append(sample_id)
        if len(meetings) != EXPECTED_MEETING_COUNTS[output_split]:
            raise BriefGenerationError(
                f"{output_split} meeting count changed: {len(meetings)}"
            )
        for meeting, atomic_values in meetings.items():
            if meeting in all_meetings:
                raise BriefGenerationError(f"meeting split overlap: {meeting}")
            atomic_values.sort(key=lambda value: value["atomic_topic"])
            if len(atomic_values) < 11:
                raise BriefGenerationError(
                    f"meeting {meeting} has only {len(atomic_values)} atomic topics"
                )
            payload = {"atomic_analyses": atomic_values}
            prompt = USER_PREFIX + canonical_json(payload)
            if meeting in prompt:
                raise BriefGenerationError(
                    f"atomic analyses leak their meeting identity: {meeting}"
                )
            opaque_id = "brief-" + sha256_text(
                f"{SCHEMA}\0{output_split}\0{meeting}"
            )[:24]
            coverage = _categories(
                [value["atomic_topic"] for value in atomic_values]
            )
            grouped[output_split].append(
                MeetingInput(
                    sample_id=opaque_id,
                    split=output_split,
                    meeting_date=meeting,
                    atomic=tuple(atomic_values),
                    source_ids=tuple(sorted(source_ids[meeting])),
                    user_prompt=prompt,
                    input_sha256=sha256_text(canonical_json(payload)),
                    prompt_sha256=sha256_text(prompt),
                    category_coverage=coverage,
                )
            )
            all_meetings.add(meeting)
        grouped[output_split].sort(key=lambda value: value.sample_id)
    return grouped


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise BriefGenerationError("transformers is unavailable") from exc
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _student_prompt_tokens(tokenizer: Any, brief: str) -> int:
    user = (
        "Make one policy decision using only the following pre-meeting analysis:\n\n"
        + canonical_json({"analysis": brief})
    )
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return _token_count(tokenizer, rendered)


def _number_atoms(text: str) -> set[str]:
    return {match.group(0).replace(",", "") for match in NUMBER_RE.finditer(text)}


def validate_brief(
    row: MeetingInput, response: TeacherResponse, tokenizer: Any
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        content = json.loads(response.content)
    except json.JSONDecodeError:
        content = None
        errors.append("invalid_json")
    if not isinstance(content, dict) or set(content) != {"meeting_decision_brief"}:
        errors.append("content_keys")
        brief = ""
    else:
        brief = str(content.get("meeting_decision_brief") or "").strip()
    if not brief:
        errors.append("empty_brief")
    if "\n-" in brief or "\n*" in brief or re.search(r"(?m)^\s*#{1,6}\s", brief):
        errors.append("brief_has_heading_or_list")
    if any(marker.lower() in brief.lower() for marker in CONTROL_MARKERS):
        errors.append("brief_has_control_marker")
    if EVIDENCE_ID_RE.search(brief):
        errors.append("brief_has_internal_evidence_id")
    if row.meeting_date in brief:
        errors.append("brief_leaks_meeting_date")
    if any(pattern.search(brief) for pattern in POLICY_LEAK_PATTERNS):
        errors.append("brief_contains_policy_action_or_recommendation")
    source_text = "\n".join(value["analysis"] for value in row.atomic)
    unsupported_numbers = sorted(_number_atoms(brief) - _number_atoms(source_text))
    if unsupported_numbers:
        errors.append("unsupported_numbers:" + ",".join(unsupported_numbers))
    source_dates = {m.group(0).lower() for m in DATE_RE.finditer(source_text)}
    unsupported_dates = sorted(
        {m.group(0).lower() for m in DATE_RE.finditer(brief)} - source_dates
    )
    if unsupported_dates:
        errors.append("unsupported_dates:" + ",".join(unsupported_dates))
    prompt_tokens = _student_prompt_tokens(tokenizer, brief) if brief else 0
    if prompt_tokens > MAX_STUDENT_PROMPT_TOKENS:
        errors.append(
            f"student_prompt_tokens:{prompt_tokens}>{MAX_STUDENT_PROMPT_TOKENS}"
        )
    if errors:
        raise BriefOutputError(errors)
    return {
        "meeting_decision_brief": brief,
        "brief_sha256": sha256_text(brief),
        "student_prompt_tokens": prompt_tokens,
    }


def build_contract(code_sha256: str) -> dict[str, Any]:
    payload = {
        "schema_version": CONTRACT_SCHEMA,
        "gold_blind": True,
        "source_fields_sent": ["atomic_topic", "analysis"],
        "identity_fields_removed": ["meeting_date", "sample_id", "split"],
        "forbidden_sources": [
            "decision_gold",
            "rate_change",
            "current_rate",
            "minutes",
            "chk2",
        ],
        "system_prompt": SYSTEM_PROMPT,
        "repair_system_prompt": REPAIR_SYSTEM_PROMPT,
        "teacher": BriefTeacherConfig().contract(),
        "student_prompt_limit": MAX_STUDENT_PROMPT_TOKENS,
        "code_sha256": code_sha256,
    }
    payload["contract_sha256"] = sha256_text(canonical_json(payload))
    return payload


def _cache_key(row: MeetingInput, contract: Mapping[str, Any]) -> str:
    return sha256_text(
        canonical_json(
            {
                "schema_version": CACHE_SCHEMA,
                "sample_id": row.sample_id,
                "input_sha256": row.input_sha256,
                "prompt_sha256": row.prompt_sha256,
                "contract_sha256": contract["contract_sha256"],
            }
        )
    )


def _cache_path(root: Path, key: str) -> Path:
    return root / "cache" / "accepted" / key[:2] / f"{key}.json"


def _store_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise BriefGenerationError(f"immutable cache collision: {path}")
        return
    _atomic_write(path, rendered)


def _repair_prompt(row: MeetingInput, errors: Sequence[str]) -> str:
    return row.user_prompt + "\n\n" + canonical_json(
        {"contract_errors_to_correct_silently": list(errors)}
    )


def generate_one(
    row: MeetingInput,
    *,
    output_root: Path,
    tokenizer: Any,
    backend: BriefBackend,
    guard: ProviderIdentityGuard,
    contract: Mapping[str, Any],
    environment: Mapping[str, str] | None,
) -> dict[str, Any]:
    errors: tuple[str, ...] = ()
    for attempt in ("primary", "repair"):
        response: TeacherResponse | None = None
        try:
            response = backend.generate(
                config=BriefTeacherConfig(),
                system_prompt=(
                    SYSTEM_PROMPT if attempt == "primary" else REPAIR_SYSTEM_PROMPT
                ),
                user_prompt=(
                    row.user_prompt
                    if attempt == "primary"
                    else _repair_prompt(row, errors)
                ),
                environment=environment,
            )
            guard.bind(response)
            target = validate_brief(row, response, tokenizer)
            payload = {
                "schema_version": CACHE_SCHEMA,
                "status": "accepted",
                "cache_key": _cache_key(row, contract),
                "sample_id": row.sample_id,
                "input_sha256": row.input_sha256,
                "prompt_sha256": row.prompt_sha256,
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
                _cache_path(output_root, payload["cache_key"]), payload
            )
            return payload
        except ModelDriftError:
            raise
        except BriefOutputError as exc:
            errors = exc.codes
            rejected = {
                "schema_version": CACHE_SCHEMA,
                "status": "rejected",
                "sample_id": row.sample_id,
                "attempt": attempt,
                "errors": list(errors),
                "response_id": None if response is None else response.response_id,
            }
            key = sha256_text(canonical_json(rejected))
            _store_immutable(
                output_root
                / "cache"
                / "rejected"
                / row.sample_id
                / f"{key}.json",
                rejected,
            )
        except Exception as exc:  # provider/runtime errors are retained
            return {
                "status": "failed",
                "sample_id": row.sample_id,
                "error": f"{type(exc).__name__}:{exc}",
            }
    return {
        "status": "failed",
        "sample_id": row.sample_id,
        "error": ";".join(errors),
    }


def _flatten(grouped: Mapping[str, Sequence[MeetingInput]]) -> list[MeetingInput]:
    return [row for split in EXPECTED_MEETING_COUNTS for row in grouped[split]]


def _prepare_outputs(
    output_root: Path,
    grouped: Mapping[str, Sequence[MeetingInput]],
    contract: Mapping[str, Any],
) -> None:
    contract_path = output_root / "prompt_contract.json"
    if contract_path.exists() and _load_json(contract_path) != contract:
        raise BriefGenerationError("existing prompt contract drift")
    _write_json(contract_path, contract)
    for split, rows in grouped.items():
        _write_jsonl(
            output_root / "prepared" / f"{split}.jsonl",
            [
                {
                    "schema_version": SCHEMA,
                    "sample_id": row.sample_id,
                    "prompt": row.user_prompt,
                    "input_sha256": row.input_sha256,
                    "prompt_sha256": row.prompt_sha256,
                }
                for row in rows
            ],
        )
        _write_jsonl(
            output_root / "manifests" / f"{split}.jsonl",
            [
                {
                    "schema_version": SCHEMA,
                    "sample_id": row.sample_id,
                    "meeting_date": row.meeting_date,
                    "split": row.split,
                    "source_ids": list(row.source_ids),
                    "valid_atomic_topic_count": len(row.atomic),
                    "category_coverage": list(row.category_coverage),
                    "input_sha256": row.input_sha256,
                    "prompt_sha256": row.prompt_sha256,
                    "contract_sha256": contract["contract_sha256"],
                }
                for row in rows
            ],
        )


def _load_resume(
    rows: Sequence[MeetingInput],
    *,
    output_root: Path,
    contract: Mapping[str, Any],
    tokenizer: Any,
    resume: bool,
    guard: ProviderIdentityGuard,
) -> dict[str, dict[str, Any]]:
    existing = list((output_root / "cache" / "accepted").glob("*/*.json"))
    if existing and not resume:
        raise BriefGenerationError("accepted cache exists; use --resume")
    accepted: dict[str, dict[str, Any]] = {}
    for row in rows:
        path = _cache_path(output_root, _cache_key(row, contract))
        if not path.exists():
            continue
        payload = _load_json(path)
        if (
            payload.get("schema_version") != CACHE_SCHEMA
            or payload.get("status") != "accepted"
            or payload.get("sample_id") != row.sample_id
            or payload.get("input_sha256") != row.input_sha256
            or payload.get("prompt_sha256") != row.prompt_sha256
            or payload.get("contract_sha256") != contract["contract_sha256"]
        ):
            raise BriefGenerationError(f"resume cache mismatch: {row.sample_id}")
        provider = payload.get("provider") or {}
        raw = payload.get("provider_raw") or {}
        response = TeacherResponse(
            reasoning=str(raw.get("reasoning_content") or ""),
            content=str(raw.get("content") or ""),
            response_id=str(provider.get("response_id") or ""),
            returned_model=str(provider.get("returned_model") or ""),
            system_fingerprint=str(provider.get("system_fingerprint") or ""),
            finish_reason=str(provider.get("finish_reason") or ""),
            created=provider.get("created"),
            usage=provider.get("usage") or {},
        )
        guard.bind(response)
        if payload.get("target") != validate_brief(row, response, tokenizer):
            raise BriefGenerationError(f"resume target mismatch: {row.sample_id}")
        accepted[row.sample_id] = payload
    return accepted


def _materialize(
    output_root: Path,
    grouped: Mapping[str, Sequence[MeetingInput]],
    accepted: Mapping[str, Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    for split, rows in grouped.items():
        output_rows: list[dict[str, Any]] = []
        teacher_rows: list[dict[str, Any]] = []
        for row in rows:
            payload = accepted.get(row.sample_id)
            if payload is None:
                continue
            target = payload["target"]
            output_rows.append(
                {
                    "meeting_date": row.meeting_date,
                    "meeting_decision_brief": target["meeting_decision_brief"],
                    "source_ids": list(row.source_ids),
                    "valid_atomic_topic_count": len(row.atomic),
                    "category_coverage": list(row.category_coverage),
                    "brief_sha256": target["brief_sha256"],
                    "source_input_sha256": row.input_sha256,
                    "generation_contract_sha256": summary["contract_sha256"],
                }
            )
            teacher_rows.append(
                {
                    "sample_id": row.sample_id,
                    "attempt": payload["attempt"],
                    "provider": payload["provider"],
                    "reasoning_content": payload["provider_raw"][
                        "reasoning_content"
                    ],
                    "content": payload["provider_raw"]["content"],
                }
            )
        _write_jsonl(output_root / f"{split}.jsonl", output_rows)
        _write_jsonl(
            output_root / "teacher_responses" / f"{split}.jsonl", teacher_rows
        )
    _write_jsonl(output_root / "failures.jsonl", list(failures))
    _write_json(output_root / "summary.json", summary)


def run(
    *,
    chk1_root: Path,
    output_root: Path,
    tokenizer_path: Path,
    dry_run: bool,
    resume: bool,
    concurrency: int,
    backend: BriefBackend | None = None,
    environment: Mapping[str, str] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    if concurrency < 1:
        raise BriefGenerationError("concurrency must be positive")
    if dry_run and resume:
        raise BriefGenerationError("--dry-run and --resume are mutually exclusive")
    grouped = load_meeting_inputs(chk1_root)
    rows = _flatten(grouped)
    code_sha = sha256_file(Path(__file__).resolve())
    contract = build_contract(code_sha)
    output_root.mkdir(parents=True, exist_ok=True)
    _prepare_outputs(output_root, grouped, contract)
    base_summary = {
        "schema_version": SCHEMA,
        "mode": "dry_run" if dry_run else "generation",
        "status": "prepared",
        "gold_blind": True,
        "source_chk1_root": str(chk1_root),
        "source_chk1_sha256": _tree_sha(chk1_root),
        "contract_sha256": contract["contract_sha256"],
        "meeting_counts": {split: len(grouped[split]) for split in grouped},
        "atomic_counts": EXPECTED_ATOMIC_COUNTS,
        "api_requests": 0,
    }
    if dry_run:
        _write_json(output_root / "summary.json", base_summary)
        return base_summary
    tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(tokenizer_path)
    guard = ProviderIdentityGuard()
    accepted = _load_resume(
        rows,
        output_root=output_root,
        contract=contract,
        tokenizer=tokenizer,
        resume=resume,
        guard=guard,
    )
    pending = [row for row in rows if row.sample_id not in accepted]
    provider = backend or OpenAICompatibleDeepSeekBackend()
    failures: list[dict[str, Any]] = []
    print(
        f"[chk4-brief] prepared={len(rows)} resumed={len(accepted)} pending={len(pending)}",
        flush=True,
    )
    completed = len(accepted)
    drift: BaseException | None = None
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                generate_one,
                row,
                output_root=output_root,
                tokenizer=tokenizer,
                backend=provider,
                guard=guard,
                contract=contract,
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
                        f"[chk4-brief] accepted={completed}/{len(rows)} split={row.split}",
                        flush=True,
                    )
                else:
                    failures.append(dict(result))
                    print(
                        f"[chk4-brief] failed sample={row.sample_id}", flush=True
                    )
            except ModelDriftError as exc:
                drift = exc
                for other in futures:
                    other.cancel()
                break
            except Exception as exc:
                failures.append(
                    {
                        "status": "failed",
                        "sample_id": row.sample_id,
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
    if drift is not None:
        raise ModelDriftError(str(drift))
    summary = {
        **base_summary,
        "status": (
            "complete" if len(accepted) == len(rows) and not failures else "incomplete"
        ),
        "accepted_count": len(accepted),
        "failure_count": len(failures),
        "resumed_count": len(rows) - len(pending),
        "api_requests": len(pending),
        "provider_identity": guard.identity,
    }
    _materialize(output_root, grouped, accepted, failures, summary)
    if summary["status"] != "complete":
        raise BriefGenerationError(
            f"brief generation incomplete: {len(accepted)}/{len(rows)}"
        )
    return summary


def _tree_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chk1-root", type=Path, default=DEFAULT_CHK1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(
            chk1_root=args.chk1_root.resolve(),
            output_root=args.output_root.resolve(),
            tokenizer_path=args.tokenizer_path.resolve(),
            dry_run=args.dry_run,
            resume=args.resume,
            concurrency=args.concurrency,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (BriefGenerationError, OSError, ValueError) as exc:
        print(
            json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False),
            file=os.sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
