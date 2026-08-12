"""Prepare and acquire chk4 Decision-SFT targets from DeepSeek V4 Pro.

This entrypoint intentionally does not run chk2.  It consumes one already
prepared, target-neutral ``meeting_decision_brief`` per meeting, joins the
brief to a locally derived policy-action label, and asks DeepSeek V4 Pro for a
grounded rationale in its validated JSON content. The provider's native
``reasoning_content`` is retained only for provenance because it can contain
instruction-processing text and teacher-only labels. The final decision JSON
is always serialized locally from gold and is never trusted to the teacher.

The stored completion is::

    message.content["reasoning"]\n</think>\n{"direction":"...","magnitude_bp":N}

The model's tokenizer supplies the opening ``<think>`` token at render time.
Meeting identity, population, split, source IDs, and gold remain in manifests;
the SFT dataset contains only ``prompt`` and ``response``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BRIEFS_ROOT = (
    REPO_ROOT / "output/data/retrain_v2/chk4/meeting_decision_briefs_v1"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output/data/retrain_v2/chk4/deepseek_v4_pro_v2"
DEFAULT_WORKBOOK = (
    REPO_ROOT
    / "archive/code/process_decsion_output/summary_base_with_meeting_date.xlsx"
)
DEFAULT_FFR_HISTORY = REPO_ROOT / "dataset/raw_data/input_data/ffr/ffr_hist.xlsx"
DEFAULT_CORE_MANIFEST_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk1/canonical_releases/"
    "chk1_full_v7_automated_v2_20260804/manifests"
)
DEFAULT_TOKENIZER = REPO_ROOT / "models/DeepSeek-R1-Distill-Llama-8B"

SPLITS = ("train", "validation", "test")
CORE_SPLIT_COUNTS = {"train": 102, "validation": 13, "test": 13}
EXPECTED_WORKBOOK_ROWS = 241
EXPECTED_UNIQUE_LABELS = 237
EXPECTED_SUPPLEMENT_CANDIDATES = 109
MIN_SUPPLEMENT_ADMITTED = 98
TEACHER_MODEL = "deepseek-v4-pro"
TEACHER_BASE_URL = "https://api.deepseek.com"
API_KEY_ENV = "DEEPSEEK_API_KEY"
MAX_TEACHER_TOKENS = 2048
MAX_PROMPT_TOKENS = 2560
MAX_COMPLETION_TOKENS = 512
MAX_TOTAL_TOKENS = 3072
DEFAULT_CONCURRENCY = 8

PREPARED_SCHEMA = "chk4-decision-prepared-v2"
MANIFEST_SCHEMA = "chk4-decision-manifest-v2"
CACHE_SCHEMA = "chk4-decision-teacher-cache-v2"
CONTRACT_SCHEMA = "chk4-decision-v4-pro-contract-v2"

STUDENT_SYSTEM_PROMPT = """\
You are an FOMC policy decision analyst. Use only the supplied target-neutral
pre-meeting analysis. In the native reasoning section, weigh the evidence under
the maximum-employment and price-stability objectives.

Do not use outside or remembered historical information, infer the meeting
identity, or claim that the Committee actually took an action. After closing
the reasoning section, output exactly one JSON object with keys direction and
magnitude_bp. Direction must be cut, hold, or hike. Hold requires magnitude 0;
cut and hike require 25, 50, 75, or 100. Output no headings, commentary, answer
tags, or additional keys.
"""

TEACHER_SYSTEM_PROMPT = """\
You are preparing one grounded rationale for an FOMC policy Decision-SFT
example. Use only the supplied target-neutral pre-meeting analysis. In the
JSON reasoning field, weigh its evidence under the maximum-employment and
price-stability objectives and support the canonical decision supplied at the
end of the request.

Do not use outside or remembered historical information, infer or mention the
meeting identity, or claim that the Committee actually took an action. Do not
mention gold labels, teacher targets, answer keys, hidden fields, known
decisions, actual historical actions, prompts, schemas, validation, or the
teacher/student roles. Write qualitative reasoning without digits, dates,
percentages, basis-point amounts, or explicit numeric quantities. Do not add
substantive facts absent from the analysis.

Return content as exactly one JSON object with keys reasoning, direction, and
magnitude_bp. The reasoning value must be one concise paragraph under 180
words. Direction and magnitude_bp must exactly equal the supplied canonical
decision. Output no other fields or content.
"""

TEACHER_REPAIR_SYSTEM_PROMPT = """\
Regenerate the JSON reasoning field using only the target-neutral pre-meeting
analysis and silently correct the supplied contract errors. Do not quote or
discuss the errors. Do not infer or mention meeting identity, actual historical
action, gold labels, teacher targets, hidden fields, prompts, schemas,
validation, or teacher/student roles. The reasoning must be one qualitative
paragraph under 180 words and contain no digits, dates, percentages,
basis-point amounts, or explicit numeric quantities. Add no substantive fact
absent from the analysis.

Return content as exactly one JSON object with keys reasoning, direction, and
magnitude_bp. Direction and magnitude_bp must exactly equal the supplied
canonical decision. This is the only repair attempt.
"""

USER_PROMPT_PREFIX = (
    "Make one policy decision using only the following pre-meeting analysis:\n\n"
)

CONTROL_MARKERS = (
    "<think>",
    "<answer>",
    "</answer>",
    "\\boxed",
    "<|channel>",
    "<channel|>",
    "<｜Assistant｜>",
    "<｜User｜>",
)
FORBIDDEN_REASONING_PATTERNS = (
    re.compile(r"\bgold(?:en)?\s+(?:label|decision|target)\b", re.I),
    re.compile(r"\bteacher(?:-only)?\s+(?:label|target|decision)\b", re.I),
    re.compile(r"\b(?:known|actual|historical)\s+(?:decision|action|vote)\b", re.I),
    re.compile(r"\bthe\s+Committee\s+(?:voted|decided|raised|cut|held)\b", re.I),
    re.compile(r"\bFOMC\s+(?:voted|decided|raised|cut|held)\b", re.I),
)
NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:%|\s*(?:percent|percentage points?|basis points?|bp|bps))?(?![A-Za-z0-9])",
    re.I,
)
DATE_RE = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\b|\b(?:19|20)\d{2}\b|\b\d{4}-\d{2}-\d{2}\b",
    re.I,
)
ACTION_RE = re.compile(
    r"^(?P<verb>No change|Raise|Cut)(?: by (?P<bp>\d+) basis points)?$",
    re.I,
)
SAFE_SAMPLE_ID_RE = re.compile(r"^dec-[0-9a-f]{24}$")


class Chk4TargetError(RuntimeError):
    """Raised when preparation, provider output, or provenance is invalid."""


class OutputContractError(Chk4TargetError):
    def __init__(self, codes: Sequence[str]):
        self.codes = tuple(str(code) for code in codes)
        super().__init__(";".join(self.codes))


class ModelDriftError(Chk4TargetError):
    """Returned provider identity differs from the fixed acquisition run."""


@dataclass(frozen=True)
class TeacherConfig:
    model: str = TEACHER_MODEL
    base_url: str = TEACHER_BASE_URL
    max_tokens: int = MAX_TEACHER_TOKENS
    timeout_seconds: float = 180.0
    max_retries: int = 3
    reasoning_effort: str = "high"

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if self.model != TEACHER_MODEL:
            raise Chk4TargetError("teacher fallback or model override is forbidden")
        if parsed.scheme != "https" or not parsed.hostname:
            raise Chk4TargetError("DeepSeek endpoint must be absolute HTTPS")
        if self.max_tokens != MAX_TEACHER_TOKENS:
            raise Chk4TargetError("teacher max_tokens must remain 2048")
        if self.reasoning_effort != "high":
            raise Chk4TargetError("teacher reasoning_effort must remain high")

    def contract(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "base_url_origin": f"https://{urlparse(self.base_url).netloc}",
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


@dataclass(frozen=True)
class TeacherResponse:
    reasoning: str
    content: str
    response_id: str
    returned_model: str
    system_fingerprint: str
    finish_reason: str
    created: int | None
    usage: Mapping[str, int | None]


class TeacherBackend(Protocol):
    def generate(
        self,
        *,
        config: TeacherConfig,
        system_prompt: str,
        user_prompt: str,
        environment: Mapping[str, str] | None,
    ) -> TeacherResponse: ...


class OpenAICompatibleDeepSeekBackend:
    def generate(
        self,
        *,
        config: TeacherConfig,
        system_prompt: str,
        user_prompt: str,
        environment: Mapping[str, str] | None,
    ) -> TeacherResponse:
        env = os.environ if environment is None else environment
        api_key = str(env.get(API_KEY_ENV) or "").strip()
        if not api_key:
            raise Chk4TargetError(f"missing API key: {API_KEY_ENV}")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise Chk4TargetError("openai package is unavailable") from exc
        client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,
        )
        last_error: Exception | None = None
        for attempt in range(config.max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model=config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=config.max_tokens,
                    reasoning_effort=config.reasoning_effort,
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "enabled"}},
                    stream=False,
                )
                if not completion.choices:
                    raise Chk4TargetError("teacher returned no completion choices")
                choice = completion.choices[0]
                message = choice.message
                usage = getattr(completion, "usage", None)
                usage_payload = {
                    key: _optional_int(getattr(usage, key, None))
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                }
                return TeacherResponse(
                    reasoning=str(
                        getattr(message, "reasoning_content", "") or ""
                    ).strip(),
                    content=str(getattr(message, "content", "") or "").strip(),
                    response_id=str(getattr(completion, "id", "") or "").strip(),
                    returned_model=str(
                        getattr(completion, "model", "") or ""
                    ).strip(),
                    system_fingerprint=str(
                        getattr(completion, "system_fingerprint", "") or ""
                    ).strip(),
                    finish_reason=str(
                        getattr(choice, "finish_reason", "") or ""
                    ).strip(),
                    created=_optional_int(getattr(completion, "created", None)),
                    usage=usage_payload,
                )
            except Chk4TargetError:
                raise
            except Exception as exc:  # provider exception classes vary
                last_error = exc
                if getattr(exc, "status_code", None) in {400, 401, 403, 404, 422}:
                    break
                if attempt >= config.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        assert last_error is not None
        raise Chk4TargetError(
            f"DeepSeek request failed: {type(last_error).__name__}: {last_error}"
        ) from last_error


@dataclass(frozen=True)
class PreparedRow:
    sample_id: str
    meeting_date: str
    split: str
    population: str
    analysis: str
    prompt: str
    teacher_prompt: str
    direction: str
    magnitude_bp: int
    source_ids: tuple[str, ...]
    input_sha256: str
    prompt_sha256: str
    gold_sha256: str

    @property
    def gold(self) -> dict[str, Any]:
        return {"direction": self.direction, "magnitude_bp": self.magnitude_bp}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Chk4TargetError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Chk4TargetError(f"JSON must contain one object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise Chk4TargetError(f"brief split is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Chk4TargetError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise Chk4TargetError(f"row is not an object: {path}:{line_number}")
            rows.append(row)
    return rows


def _find_split_file(root: Path, split: str) -> Path:
    aliases = ("eval", "validation") if split == "validation" else (split,)
    candidates = [root / f"{name}.jsonl" for name in aliases]
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) != 1:
        raise Chk4TargetError(
            f"expected exactly one {split} brief file; tried {candidates}"
        )
    return existing[0]


def map_rate_change(value: str) -> tuple[str, int]:
    match = ACTION_RE.fullmatch(str(value).strip())
    if match is None:
        raise Chk4TargetError(f"unsupported rate_change label: {value!r}")
    verb = match.group("verb").lower()
    raw_bp = match.group("bp")
    if verb == "no change":
        if raw_bp is not None:
            raise Chk4TargetError(f"hold label unexpectedly has magnitude: {value}")
        return "hold", 0
    if raw_bp is None:
        raise Chk4TargetError(f"directional label lacks magnitude: {value}")
    magnitude = int(raw_bp)
    if magnitude not in {25, 50, 75, 100}:
        raise Chk4TargetError(f"unsupported policy magnitude: {magnitude}")
    return ("hike" if verb == "raise" else "cut"), magnitude


def load_and_validate_labels(
    workbook: Path, ffr_history: Path
) -> tuple[dict[str, tuple[str, int]], dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise Chk4TargetError("pandas/openpyxl is required for local labels") from exc
    frame = pd.read_excel(workbook, usecols=["meeting_date", "rate_change"])[
        ["meeting_date", "rate_change"]
    ]
    if len(frame) != EXPECTED_WORKBOOK_ROWS:
        raise Chk4TargetError(
            f"Decision workbook has {len(frame)} rows, expected {EXPECTED_WORKBOOK_ROWS}"
        )
    labels: dict[str, tuple[str, int]] = {}
    duplicate_rows = 0
    for raw_date, raw_action in frame.itertuples(index=False, name=None):
        meeting = pd.Timestamp(raw_date).date().isoformat()
        action = map_rate_change(str(raw_action))
        if meeting in labels:
            duplicate_rows += 1
            if labels[meeting] != action:
                raise Chk4TargetError(f"conflicting labels for meeting {meeting}")
        labels[meeting] = action
    if len(labels) != EXPECTED_UNIQUE_LABELS or duplicate_rows != 4:
        raise Chk4TargetError(
            f"label dedup mismatch: unique={len(labels)}, duplicates={duplicate_rows}"
        )
    ffr = pd.read_excel(ffr_history, usecols=["date", "rate_change"])
    changes = {
        pd.Timestamp(raw_date).date(): int(raw_change)
        for raw_date, raw_change in ffr.itertuples(index=False, name=None)
    }
    mismatches: list[str] = []
    for meeting_text, (direction, magnitude) in labels.items():
        meeting = date.fromisoformat(meeting_text)
        observed = changes.get(meeting)
        if observed is None:
            observed = changes.get(meeting + timedelta(days=1), 0)
        expected = 0 if direction == "hold" else magnitude * (1 if direction == "hike" else -1)
        if observed != expected:
            mismatches.append(f"{meeting_text}:{expected}!={observed}")
    if mismatches:
        raise Chk4TargetError(
            "same/next-day FFR reconciliation failed: " + ",".join(mismatches[:10])
        )
    counts: dict[str, int] = {}
    for direction, magnitude in labels.values():
        key = f"{direction}:{magnitude}"
        counts[key] = counts.get(key, 0) + 1
    return labels, {
        "workbook_rows": len(frame),
        "unique_meetings": len(labels),
        "duplicate_rows": duplicate_rows,
        "ffr_mismatches": 0,
        "action_counts": counts,
        "workbook_sha256": sha256_file(workbook),
        "ffr_history_sha256": sha256_file(ffr_history),
    }


def load_core_split_map(manifest_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in SPLITS:
        source_name = "eval" if split == "validation" else split
        path = manifest_root / f"{source_name}.jsonl"
        meetings: set[str] = set()
        for row in _load_jsonl(path):
            meeting = str(row.get("meeting_date") or "").strip()
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", meeting):
                raise Chk4TargetError(f"invalid core meeting_date in {path}")
            meetings.add(meeting)
        counts[split] = len(meetings)
        for meeting in meetings:
            if meeting in result:
                raise Chk4TargetError(f"core split overlap for {meeting}")
            result[meeting] = split
    if counts != CORE_SPLIT_COUNTS:
        raise Chk4TargetError(
            f"canonical core meeting counts changed: {counts} != {CORE_SPLIT_COUNTS}"
        )
    return result


def render_student_prompt(analysis: str) -> str:
    return USER_PROMPT_PREFIX + canonical_json({"analysis": analysis})


def _opaque_sample_id(meeting_date: str, population: str) -> str:
    digest = sha256_text(f"chk4-decision-v1\0{population}\0{meeting_date}")[:24]
    return f"dec-{digest}"


def _canonical_gold(direction: str, magnitude: int) -> str:
    # Required output key order is direction, then magnitude_bp.
    return json.dumps(
        {"direction": direction, "magnitude_bp": magnitude},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _teacher_user_prompt(prompt: str, direction: str, magnitude: int) -> str:
    return (
        prompt
        + "\n\nTeacher-only canonical decision (do not mention it in reasoning):\n"
        + _canonical_gold(direction, magnitude)
    )


def prepare_rows(
    *,
    briefs_root: Path,
    labels: Mapping[str, tuple[str, int]],
    core_splits: Mapping[str, str],
) -> tuple[dict[str, list[PreparedRow]], dict[str, Any]]:
    prepared = {split: [] for split in SPLITS}
    seen_dates: set[str] = set()
    supplement_actions: set[tuple[str, int]] = set()
    admitted_supplement_actions: set[tuple[str, int]] = set()
    supplement_candidates = {
        meeting
        for meeting in labels
        if meeting < "2009-01-01" and meeting not in core_splits
    }
    if len(supplement_candidates) != EXPECTED_SUPPLEMENT_CANDIDATES:
        raise Chk4TargetError(
            f"supplement candidate count changed: {len(supplement_candidates)}"
        )
    supplement_actions = {labels[meeting] for meeting in supplement_candidates}
    split_paths: dict[str, str] = {}
    for split in SPLITS:
        path = _find_split_file(briefs_root, split)
        split_paths[split] = str(path)
        for row in _load_jsonl(path):
            meeting = str(row.get("meeting_date") or "").strip()
            analysis = str(row.get("meeting_decision_brief") or "").strip()
            if not meeting or meeting not in labels:
                raise Chk4TargetError(f"brief has unknown meeting_date: {meeting!r}")
            if meeting in seen_dates:
                raise Chk4TargetError(f"one-meeting-one-row violation: {meeting}")
            if not analysis:
                raise Chk4TargetError(f"empty meeting_decision_brief: {meeting}")
            if meeting in core_splits:
                population = "decision_core_post2009_v1"
                expected_split = core_splits[meeting]
            elif meeting in supplement_candidates:
                population = "decision_supplement_1993_2008_v1"
                expected_split = "train"
                admitted_supplement_actions.add(labels[meeting])
                topic_count = int(row.get("valid_atomic_topic_count") or 0)
                categories = set(row.get("category_coverage") or ())
                required_categories = {
                    "prices",
                    "employment_activity",
                    "financial_conditions",
                }
                if topic_count < 11 or not required_categories.issubset(categories):
                    raise Chk4TargetError(
                        f"supplement admission contract failed: {meeting}"
                    )
            else:
                raise Chk4TargetError(f"meeting is outside both populations: {meeting}")
            if split != expected_split:
                raise Chk4TargetError(
                    f"split mismatch for {meeting}: {split} != {expected_split}"
                )
            direction, magnitude = labels[meeting]
            prompt = render_student_prompt(analysis)
            sample_id = _opaque_sample_id(meeting, population)
            source_ids_value = row.get("source_ids") or []
            if not isinstance(source_ids_value, list) or not all(
                isinstance(value, str) and value for value in source_ids_value
            ):
                raise Chk4TargetError(f"invalid source_ids for {meeting}")
            prepared[split].append(
                PreparedRow(
                    sample_id=sample_id,
                    meeting_date=meeting,
                    split=split,
                    population=population,
                    analysis=analysis,
                    prompt=prompt,
                    teacher_prompt=_teacher_user_prompt(prompt, direction, magnitude),
                    direction=direction,
                    magnitude_bp=magnitude,
                    source_ids=tuple(source_ids_value),
                    input_sha256=sha256_text(analysis),
                    prompt_sha256=sha256_text(prompt),
                    gold_sha256=sha256_text(_canonical_gold(direction, magnitude)),
                )
            )
            seen_dates.add(meeting)
    missing_core = sorted(set(core_splits) - seen_dates)
    if missing_core:
        raise Chk4TargetError(
            f"core population is incomplete ({len(missing_core)} missing): "
            + ",".join(missing_core[:10])
        )
    admitted_supplement = len(seen_dates & supplement_candidates)
    if admitted_supplement:
        if admitted_supplement < MIN_SUPPLEMENT_ADMITTED:
            raise Chk4TargetError(
                f"supplement has {admitted_supplement} meetings; minimum is {MIN_SUPPLEMENT_ADMITTED}"
            )
        if admitted_supplement_actions != supplement_actions:
            raise Chk4TargetError("supplement admission dropped an original action class")
    for split in SPLITS:
        prepared[split].sort(key=lambda row: row.sample_id)
    return prepared, {
        "brief_split_files": split_paths,
        "core_meetings": len(core_splits),
        "supplement_candidates": len(supplement_candidates),
        "supplement_admitted": admitted_supplement,
        "total_prepared": sum(len(rows) for rows in prepared.values()),
        "split_counts": {split: len(prepared[split]) for split in SPLITS},
    }


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise Chk4TargetError("transformers is unavailable") from exc
    if not path.is_dir():
        raise Chk4TargetError(f"tokenizer path is missing: {path}")
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _render_student_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as exc:
        raise Chk4TargetError("student tokenizer cannot render the chat contract") from exc
    if not isinstance(rendered, str) or not rendered:
        raise Chk4TargetError("student chat template returned empty text")
    return rendered


def _numeric_atoms(text: str) -> set[str]:
    return {re.sub(r"\s+", " ", match.group(0).lower()) for match in NUMBER_RE.finditer(text)}


def validate_prepared_token_budgets(
    prepared: Mapping[str, Sequence[PreparedRow]], tokenizer: Any
) -> dict[str, Any]:
    prompt_counts: list[int] = []
    for split in SPLITS:
        for row in prepared[split]:
            rendered = _render_student_prompt(tokenizer, row.prompt)
            count = _token_count(tokenizer, rendered)
            if count > MAX_PROMPT_TOKENS:
                raise Chk4TargetError(
                    f"prompt token overflow for {row.sample_id}: {count}>{MAX_PROMPT_TOKENS}"
                )
            prompt_counts.append(count)
    return {
        "count": len(prompt_counts),
        "min": min(prompt_counts),
        "max": max(prompt_counts),
        "mean": sum(prompt_counts) / len(prompt_counts),
        "truncated": False,
    }


def _strict_decision_content(content: str) -> tuple[str, str, int]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise OutputContractError(("invalid_teacher_json",)) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "reasoning",
        "direction",
        "magnitude_bp",
    }:
        raise OutputContractError(("teacher_json_keys",))
    reasoning = payload.get("reasoning")
    direction = payload.get("direction")
    magnitude = payload.get("magnitude_bp")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise OutputContractError(("empty_reasoning_content",))
    if direction not in {"cut", "hold", "hike"}:
        raise OutputContractError(("unsupported_direction",))
    if isinstance(magnitude, bool) or not isinstance(magnitude, int):
        raise OutputContractError(("magnitude_not_integer",))
    allowed = {0} if direction == "hold" else {25, 50, 75, 100}
    if magnitude not in allowed:
        raise OutputContractError(("unsupported_magnitude",))
    return reasoning.strip(), direction, magnitude


def validate_response(
    *,
    row: PreparedRow,
    response: TeacherResponse,
    tokenizer: Any,
) -> dict[str, Any]:
    errors: list[str] = []
    reasoning = ""
    try:
        reasoning, observed_direction, observed_magnitude = (
            _strict_decision_content(response.content)
        )
        if (observed_direction, observed_magnitude) != (
            row.direction,
            row.magnitude_bp,
        ):
            errors.append("teacher_content_disagrees_with_gold")
    except OutputContractError as exc:
        errors.extend(exc.codes)
    if "</think>" in reasoning:
        errors.append("reasoning_contains_closing_think")
    for marker in CONTROL_MARKERS:
        if marker.lower() in reasoning.lower():
            errors.append(f"control_marker:{marker}")
    for pattern in FORBIDDEN_REASONING_PATTERNS:
        if pattern.search(reasoning):
            errors.append("reasoning_mentions_forbidden_target_or_action")
            break
    allowed_numbers = _numeric_atoms(row.analysis)
    unsupported_numbers = sorted(_numeric_atoms(reasoning) - allowed_numbers)
    if unsupported_numbers:
        errors.append("unsupported_reasoning_numbers:" + ",".join(unsupported_numbers))
    analysis_dates = {match.group(0).lower() for match in DATE_RE.finditer(row.analysis)}
    unsupported_dates = sorted(
        {match.group(0).lower() for match in DATE_RE.finditer(reasoning)} - analysis_dates
    )
    if unsupported_dates:
        errors.append("unsupported_reasoning_dates:" + ",".join(unsupported_dates))
    completion = reasoning + "\n</think>\n" + _canonical_gold(
        row.direction, row.magnitude_bp
    )
    if completion.count("</think>") != 1:
        errors.append("completion_think_boundary_count")
    completion_tokens = _token_count(tokenizer, completion)
    rendered = _render_student_prompt(tokenizer, row.prompt)
    total_tokens = _token_count(tokenizer, rendered + completion)
    if completion_tokens > MAX_COMPLETION_TOKENS:
        errors.append(
            f"completion_tokens:{completion_tokens}>{MAX_COMPLETION_TOKENS}"
        )
    if total_tokens > MAX_TOTAL_TOKENS:
        errors.append(f"total_tokens:{total_tokens}>{MAX_TOTAL_TOKENS}")
    if errors:
        raise OutputContractError(errors)
    return {
        "reasoning": reasoning,
        "completion": completion,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


class ProviderIdentityGuard:
    def __init__(self) -> None:
        self._identity: tuple[str, str] | None = None
        self._response_ids: set[str] = set()
        self._lock = threading.Lock()

    @property
    def identity(self) -> tuple[str, str] | None:
        with self._lock:
            return self._identity

    def bind(self, response: TeacherResponse) -> None:
        if response.returned_model != TEACHER_MODEL:
            raise ModelDriftError(
                f"returned model {response.returned_model!r} != {TEACHER_MODEL!r}"
            )
        if not response.response_id:
            raise ModelDriftError("provider response_id is missing")
        if not response.system_fingerprint:
            raise ModelDriftError("provider system_fingerprint is missing")
        identity = (response.returned_model, response.system_fingerprint)
        with self._lock:
            if self._identity is None:
                self._identity = identity
            elif self._identity != identity:
                raise ModelDriftError(
                    f"provider identity drift: {self._identity} != {identity}"
                )
            if response.response_id in self._response_ids:
                raise ModelDriftError(f"duplicate response_id: {response.response_id}")
            self._response_ids.add(response.response_id)


def build_prompt_contract(code_sha256: str) -> dict[str, Any]:
    teacher = TeacherConfig()
    payload = {
        "schema_version": CONTRACT_SCHEMA,
        "student_system_prompt": STUDENT_SYSTEM_PROMPT,
        "student_system_prompt_sha256": sha256_text(STUDENT_SYSTEM_PROMPT),
        "student_user_prefix": USER_PROMPT_PREFIX,
        "teacher_system_prompt": TEACHER_SYSTEM_PROMPT,
        "teacher_repair_system_prompt": TEACHER_REPAIR_SYSTEM_PROMPT,
        "teacher_contract": teacher.contract(),
        "teacher_output_mapping": {
            "reasoning": "json.loads(message.content)['reasoning'].strip()",
            "decision": "locally_canonicalized_gold_json",
            "response": "reasoning + '\\n</think>\\n' + canonical_gold_json",
            "native_reasoning_content": "provenance_only_not_training_data",
        },
        "student_token_budgets": {
            "prompt": MAX_PROMPT_TOKENS,
            "completion": MAX_COMPLETION_TOKENS,
            "total": MAX_TOTAL_TOKENS,
            "truncation": False,
        },
        "code_sha256": code_sha256,
    }
    payload["contract_sha256"] = sha256_text(canonical_json(payload))
    return payload


def _cache_key(row: PreparedRow, contract: Mapping[str, Any]) -> str:
    return sha256_text(
        canonical_json(
            {
                "schema_version": CACHE_SCHEMA,
                "sample_id": row.sample_id,
                "input_sha256": row.input_sha256,
                "prompt_sha256": row.prompt_sha256,
                "gold_sha256": row.gold_sha256,
                "contract_sha256": contract["contract_sha256"],
            }
        )
    )


def _cache_path(output_root: Path, cache_key: str) -> Path:
    return output_root / "cache" / "accepted" / cache_key[:2] / f"{cache_key}.json"


def _store_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise Chk4TargetError(f"immutable cache collision: {path}")
        return
    _atomic_write(path, rendered)


def _repair_prompt(row: PreparedRow, _errors: Sequence[str]) -> str:
    return (
        row.teacher_prompt
        + "\n\n"
        + canonical_json(
            {
                "repair_directive": (
                    "Regenerate the required three-key JSON. Keep the reasoning "
                    "qualitative, grounded, and under the stated limit."
                )
            }
        )
    )


def generate_one(
    row: PreparedRow,
    *,
    output_root: Path,
    tokenizer: Any,
    backend: TeacherBackend,
    guard: ProviderIdentityGuard,
    contract: Mapping[str, Any],
    environment: Mapping[str, str] | None,
) -> dict[str, Any]:
    config = TeacherConfig()
    cache_key = _cache_key(row, contract)
    errors: tuple[str, ...] = ()
    for attempt in ("primary", "repair"):
        response: TeacherResponse | None = None
        try:
            response = backend.generate(
                config=config,
                system_prompt=(
                    TEACHER_SYSTEM_PROMPT
                    if attempt == "primary"
                    else TEACHER_REPAIR_SYSTEM_PROMPT
                ),
                user_prompt=(
                    row.teacher_prompt
                    if attempt == "primary"
                    else _repair_prompt(row, errors)
                ),
                environment=environment,
            )
            guard.bind(response)
            target = validate_response(row=row, response=response, tokenizer=tokenizer)
            payload = {
                "schema_version": CACHE_SCHEMA,
                "status": "accepted",
                "cache_key": cache_key,
                "sample_id": row.sample_id,
                "input_sha256": row.input_sha256,
                "prompt_sha256": row.prompt_sha256,
                "gold_sha256": row.gold_sha256,
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
            _store_immutable(_cache_path(output_root, cache_key), payload)
            return payload
        except ModelDriftError:
            raise
        except OutputContractError as exc:
            errors = exc.codes
            rejected = {
                "schema_version": CACHE_SCHEMA,
                "status": "rejected",
                "sample_id": row.sample_id,
                "attempt": attempt,
                "errors": list(errors),
                "provider": (
                    None
                    if response is None
                    else {
                        "response_id": response.response_id,
                        "returned_model": response.returned_model,
                        "system_fingerprint": response.system_fingerprint,
                    }
                ),
                "provider_raw": (
                    None
                    if response is None
                    else {
                        "reasoning_content": response.reasoning,
                        "content": response.content,
                        "finish_reason": response.finish_reason,
                        "usage": dict(response.usage),
                    }
                ),
            }
            rejection_key = sha256_text(canonical_json(rejected))
            _store_immutable(
                output_root
                / "cache"
                / "rejected"
                / cache_key[:2]
                / cache_key
                / f"{rejection_key}.json",
                rejected,
            )
        except Exception as exc:  # provider/runtime error is not repaired as content
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


def _materialize_prepared(
    output_root: Path,
    prepared: Mapping[str, Sequence[PreparedRow]],
    contract: Mapping[str, Any],
) -> None:
    contract_path = output_root / "prompt_contract.json"
    if contract_path.exists() and _load_json(contract_path) != contract:
        raise Chk4TargetError("prompt contract drift in existing output root")
    _write_json(contract_path, contract)
    for split in SPLITS:
        prepared_rows: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        for row in prepared[split]:
            prepared_rows.append(
                {
                    "schema_version": PREPARED_SCHEMA,
                    "sample_id": row.sample_id,
                    "prompt": row.prompt,
                    "input_sha256": row.input_sha256,
                    "prompt_sha256": row.prompt_sha256,
                    "gold_sha256": row.gold_sha256,
                }
            )
            manifest_rows.append(
                {
                    "schema_version": MANIFEST_SCHEMA,
                    "sample_id": row.sample_id,
                    "meeting_date": row.meeting_date,
                    "split": row.split,
                    "population": row.population,
                    "population_role": (
                        "core" if row.population == "decision_core_post2009_v1" else "supplement"
                    ),
                    "source_ids": list(row.source_ids),
                    "gold": row.gold,
                    "input_sha256": row.input_sha256,
                    "prompt_sha256": row.prompt_sha256,
                    "gold_sha256": row.gold_sha256,
                    "contract_sha256": contract["contract_sha256"],
                }
            )
        _write_jsonl(output_root / "prepared" / f"{split}.jsonl", prepared_rows)
        _write_jsonl(output_root / "manifests" / f"{split}.jsonl", manifest_rows)


def _load_resume_cache(
    rows: Sequence[PreparedRow],
    *,
    output_root: Path,
    contract: Mapping[str, Any],
    resume: bool,
    guard: ProviderIdentityGuard,
    tokenizer: Any,
) -> dict[str, dict[str, Any]]:
    accepted: dict[str, dict[str, Any]] = {}
    existing_cache = list((output_root / "cache" / "accepted").glob("*/*.json"))
    if existing_cache and not resume:
        raise Chk4TargetError("accepted cache exists; use --resume")
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
            or payload.get("gold_sha256") != row.gold_sha256
            or payload.get("contract_sha256") != contract["contract_sha256"]
        ):
            raise Chk4TargetError(f"resume cache binding mismatch: {row.sample_id}")
        provider = payload.get("provider") or {}
        raw = payload.get("provider_raw") or {}
        cached_response = TeacherResponse(
            reasoning=str(raw.get("reasoning_content") or ""),
            content=str(raw.get("content") or ""),
            response_id=str(provider.get("response_id") or ""),
            returned_model=str(provider.get("returned_model") or ""),
            system_fingerprint=str(provider.get("system_fingerprint") or ""),
            finish_reason=str(provider.get("finish_reason") or ""),
            created=_optional_int(provider.get("created")),
            usage=provider.get("usage") or {},
        )
        guard.bind(cached_response)
        target = validate_response(row=row, response=cached_response, tokenizer=tokenizer)
        if payload.get("target") != target:
            raise Chk4TargetError(f"resume target validation failed: {row.sample_id}")
        accepted[row.sample_id] = payload
    return accepted


def _materialize_final(
    output_root: Path,
    prepared: Mapping[str, Sequence[PreparedRow]],
    accepted: Mapping[str, Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    for split in SPLITS:
        teacher_rows: list[dict[str, Any]] = []
        sft_rows: list[dict[str, Any]] = []
        for row in prepared[split]:
            payload = accepted.get(row.sample_id)
            if payload is None:
                continue
            teacher_rows.append(
                {
                    "sample_id": row.sample_id,
                    "attempt": payload["attempt"],
                    "provider": payload["provider"],
                    "reasoning_content": payload["provider_raw"]["reasoning_content"],
                    "content": payload["provider_raw"]["content"],
                }
            )
            sft_rows.append(
                {"prompt": row.prompt, "response": payload["target"]["completion"]}
            )
        _write_jsonl(output_root / "teacher_responses" / f"{split}.jsonl", teacher_rows)
        _write_jsonl(output_root / "sft" / f"{split}.jsonl", sft_rows)
    _write_jsonl(output_root / "failures.jsonl", list(failures))
    _write_json(output_root / "summary.json", summary)


def run(
    *,
    briefs_root: Path,
    output_root: Path,
    workbook: Path,
    ffr_history: Path,
    core_manifest_root: Path,
    tokenizer_path: Path,
    dry_run: bool,
    resume: bool,
    concurrency: int,
    backend: TeacherBackend | None = None,
    environment: Mapping[str, str] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    if concurrency < 1:
        raise Chk4TargetError("concurrency must be positive")
    if dry_run and resume:
        raise Chk4TargetError("--dry-run and --resume are mutually exclusive")
    code_sha256 = sha256_file(Path(__file__).resolve())
    contract = build_prompt_contract(code_sha256)
    labels, label_audit = load_and_validate_labels(workbook, ffr_history)
    core_splits = load_core_split_map(core_manifest_root)
    prepared, preparation_audit = prepare_rows(
        briefs_root=briefs_root,
        labels=labels,
        core_splits=core_splits,
    )
    tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(tokenizer_path)
    token_audit = validate_prepared_token_budgets(prepared, tokenizer)
    output_root.mkdir(parents=True, exist_ok=True)
    _materialize_prepared(output_root, prepared, contract)
    base_summary = {
        "schema_version": "chk4-decision-generation-summary-v2",
        "mode": "dry_run" if dry_run else "generation",
        "teacher_model": TEACHER_MODEL,
        "contract_sha256": contract["contract_sha256"],
        "label_audit": label_audit,
        "preparation_audit": preparation_audit,
        "token_audit": token_audit,
    }
    if dry_run:
        summary = {**base_summary, "status": "prepared", "api_requests": 0}
        _write_json(output_root / "summary.json", summary)
        return summary
    rows = [row for split in SPLITS for row in prepared[split]]
    guard = ProviderIdentityGuard()
    accepted = _load_resume_cache(
        rows,
        output_root=output_root,
        contract=contract,
        resume=resume,
        guard=guard,
        tokenizer=tokenizer,
    )
    pending = [row for row in rows if row.sample_id not in accepted]
    provider = backend or OpenAICompatibleDeepSeekBackend()
    failures: list[dict[str, Any]] = []
    drift_error: BaseException | None = None
    completed = len(accepted)
    print(
        f"[chk4-target] prepared={len(rows)} resumed={completed} pending={len(pending)}",
        flush=True,
    )
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
                        f"[chk4-target] accepted={completed}/{len(rows)} split={row.split}",
                        flush=True,
                    )
                else:
                    failures.append(dict(result))
                    print(
                        f"[chk4-target] failed sample={row.sample_id}", flush=True
                    )
            except ModelDriftError as exc:
                drift_error = exc
                for other in futures:
                    other.cancel()
                break
            except Exception as exc:  # fail is explicit and retained
                failures.append(
                    {
                        "status": "failed",
                        "sample_id": row.sample_id,
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
    if drift_error is not None:
        raise ModelDriftError(str(drift_error))
    summary = {
        **base_summary,
        "status": "complete" if len(accepted) == len(rows) and not failures else "incomplete",
        "prepared_count": len(rows),
        "accepted_count": len(accepted),
        "failure_count": len(failures),
        "resumed_count": len(rows) - len(pending),
        "provider_identity": guard.identity,
    }
    _materialize_final(output_root, prepared, accepted, failures, summary)
    if summary["status"] != "complete":
        raise Chk4TargetError(
            f"generation incomplete: {len(accepted)}/{len(rows)} accepted"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--briefs-root", type=Path, default=DEFAULT_BRIEFS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--decision-workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--ffr-history", type=Path, default=DEFAULT_FFR_HISTORY)
    parser.add_argument(
        "--core-manifest-root", type=Path, default=DEFAULT_CORE_MANIFEST_ROOT
    )
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
            briefs_root=args.briefs_root.resolve(),
            output_root=args.output_root.resolve(),
            workbook=args.decision_workbook.resolve(),
            ffr_history=args.ffr_history.resolve(),
            core_manifest_root=args.core_manifest_root.resolve(),
            tokenizer_path=args.tokenizer_path.resolve(),
            dry_run=args.dry_run,
            resume=args.resume,
            concurrency=args.concurrency,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except (Chk4TargetError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "blocked", "error": str(exc)},
                ensure_ascii=False,
            ),
            file=os.sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
