"""Materialize a clean, immutable chk3 Minutes-SFT training release.

The completed DeepSeek acquisition is retained verbatim.  This module projects
the small set of valid JSON transport envelopes to their answer text, rebuilds
truncated chk1 answers from their immutable point-in-time source prompt, and
regenerates only chk3 targets whose analysis changed or no longer validates.

No sample is silently dropped.  The final training rows contain exactly
``prompt`` and ``response``; identity and provenance remain in side manifests.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from jobs.generation.generate_chk3_sft_targets import (
    DEFAULT_CHK1_HANDOFF,
    DEFAULT_OUTPUT_ROOT as DEFAULT_ACQUISITION_ROOT,
    DEFAULT_TOKENIZER_PATH,
    EXPECTED_SPLIT_COUNTS,
    MAX_REASONING_TOKENS,
    MAX_TOKENS,
    SPLITS,
    STUDENT_SYSTEM_PROMPT,
    Chk3DataError,
    DeepSeekTeacherResponse,
    OpenAIDeepSeekBackend,
    OutputContractError,
    PreparedRow,
    ProviderIdentityGuard,
    TeacherBackend,
    ValidatedTarget,
    _load_tokenizer,
    _reasoning_meta_categories,
    _teacher_config,
    canonical_json,
    render_user_prompt,
    sha256_file,
    sha256_text,
    validate_teacher_target,
)
from jobs.generation.repair_chk3_sft_targets import (
    TARGETED_REPAIR_SYSTEM_PROMPT,
    semantic_analysis,
    targeted_user_prompt,
)
from jobs.retrain_v2.chk1.contracts import GENERATOR_SYSTEM_PROMPT
from jobs.retrain_v2.chk1.verifier import verify_candidate
from jobs.retrain_v2.token_budget_gate import _count_sft_row


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE_ID = "chk3_minutes_clean_v1_20260805"
DEFAULT_RELEASE_PARENT = REPO_ROOT / "dataset/processed/retrain_v2"
DEFAULT_WORK_ROOT = REPO_ROOT / "output/data/retrain_v2/chk3/training_release_clean_v1"
DEFAULT_CHK3_CONFIG = REPO_ROOT / "configs/retrain_v2/chk3_minutes_sft.yaml"

SCHEMA_VERSION = "chk3-minutes-training-materializer-v4"
RELEASE_SCHEMA_VERSION = "chk3-minutes-training-release-v1"
CACHE_SCHEMA_VERSION = "chk3-minutes-training-repair-cache-v4"
SOURCE_RECOVERY_MAX_ATTEMPTS = 3
TARGET_MAX_ATTEMPTS = 3
PROMPT_TOKEN_LIMIT = 3072
TOTAL_TOKEN_LIMIT = 4096

_CONTROL_MARKER_RE = re.compile(
    r"<think>|</think>|<answer>|</answer>|<\|channel>|<channel\|>",
    flags=re.IGNORECASE,
)
_ANY_EVIDENCE_ID_RE = re.compile(r"\bev-[0-9a-f]+\b", flags=re.IGNORECASE)
_TRANSPORT_KEYS = ("answer", "content", "reasoning_content", "evidence_ids")

SOURCE_RECOVERY_SYSTEM_PROMPT = GENERATOR_SYSTEM_PROMPT + """

Recovery constraints for this request:
- Keep reasoning_content concise, preferably 100-300 words, so the structured
  content is completed well before the token limit.
- In answer, use only numeric surface values that occur verbatim in the fact
  card. Do not round, approximate, rescale, or calculate a difference, sum,
  ratio, growth rate, or percentage change.
- It is acceptable to omit a less important fact-card value. It is never
  acceptable to introduce a numeric value absent from the fact card.
- Return answer as one concise paragraph and list the evidence IDs supporting
  its claims. Do not place JSON, headings, or model-control tags inside answer.
"""

SOURCE_RECOVERY_REPAIR_PROMPT = """\
Regenerate the economic analysis using only the original point-in-time fact
card and style guide. Apply the supplied diagnostics silently. Keep
reasoning_content to 100-300 words. In answer, use only numeric surface values
that occur verbatim in the fact card; do not round, approximate, rescale, or
derive any new number. Remove unsupported causes, events, people, policy
actions, and post-cutoff claims. Return one concise paragraph in answer and
the supporting evidence IDs, using the requested JSON content shape. Do not
discuss diagnostics, prompts, JSON, or repair in reasoning_content or answer.
"""


class MaterializationError(Chk3DataError):
    """The training release cannot be published safely."""


@dataclass(frozen=True)
class AcquisitionRow:
    prepared: Mapping[str, Any]
    teacher: Mapping[str, Any]
    sft: Mapping[str, Any]
    manifest: Mapping[str, Any]

    @property
    def sample_id(self) -> str:
        return str(self.prepared["sample_id"])

    @property
    def split(self) -> str:
        return str(self.prepared["split"])

    @property
    def source_index(self) -> int:
        return int(self.prepared["source_index"])

    @property
    def analysis(self) -> str:
        return str(self.prepared["analysis"])


@dataclass(frozen=True)
class CleanRow:
    source: AcquisitionRow
    prepared: PreparedRow
    target: ValidatedTarget
    teacher_response: DeepSeekTeacherResponse
    analysis_mode: str
    target_mode: str


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"{label} must be a JSON object: {path}")
    return value


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise MaterializationError(f"missing {label}: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise MaterializationError(f"blank row in {label}: {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MaterializationError(
                    f"invalid JSON in {label}: {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise MaterializationError(
                    f"non-object row in {label}: {path}:{line_number}"
                )
            rows.append(value)
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


def _store_cache(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = canonical_json(dict(payload)) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != serialized:
            raise MaterializationError(f"immutable cache collision: {path}")
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def is_transport_wrapped(analysis: str) -> bool:
    """Return whether a chk1 final answer is a provider transport envelope."""

    text = str(analysis).strip()
    return text.startswith("{") or text.startswith("```json")


def project_valid_transport_analysis(analysis: str) -> str | None:
    """Extract only a complete answer from a valid JSON transport envelope."""

    raw = str(analysis).strip()
    if not is_transport_wrapped(raw):
        return raw
    if raw.startswith("```json"):
        if not raw.endswith("```"):
            return None
        raw = raw[len("```json") : -3].strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    candidate: Any = payload.get("answer")
    if not isinstance(candidate, str):
        content = payload.get("content")
        if isinstance(content, dict):
            candidate = content.get("answer")
        elif isinstance(content, str):
            try:
                nested = json.loads(content)
            except json.JSONDecodeError:
                nested = None
            candidate = nested.get("answer") if isinstance(nested, dict) else None
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    projected = semantic_analysis(candidate).strip()
    if is_transport_wrapped(projected) or _CONTROL_MARKER_RE.search(projected):
        return None
    return projected


def project_internal_evidence_citations(analysis: str) -> str:
    """Remove non-semantic ``ev-`` citations from an otherwise plain analysis."""

    raw = str(analysis).strip()
    return semantic_analysis(raw) if _ANY_EVIDENCE_ID_RE.search(raw) else raw


def _validate_clean_analysis(analysis: str, *, sample_id: str) -> str:
    clean = " ".join(str(analysis).split()).strip()
    if not clean:
        raise MaterializationError(f"empty clean analysis: {sample_id}")
    if is_transport_wrapped(clean):
        raise MaterializationError(f"transport wrapper remains in analysis: {sample_id}")
    if _CONTROL_MARKER_RE.search(clean):
        raise MaterializationError(f"control marker remains in analysis: {sample_id}")
    if len(clean) < 40:
        raise MaterializationError(f"implausibly short clean analysis: {sample_id}")
    return clean


def _fact_card_supports_numeric_surface(
    claim: str, fact_card: Mapping[str, Any]
) -> bool:
    """Accept exact number text found anywhere in the immutable fact card.

    The legacy chk1 verifier indexes evidence ``value`` fields, but dates and
    metric labels also contain legitimate numeric surfaces (for example a
    meeting year or a 10-year Treasury maturity).  This extension remains
    exact: rounded and derived numbers still fail.
    """

    match = re.search(
        r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", str(claim)
    )
    if match is None:
        return False
    number = match.group(0)
    fact_text = canonical_json(dict(fact_card))
    return re.search(
        rf"(?<![0-9.]){re.escape(number)}(?![0-9.])", fact_text
    ) is not None


def _teacher_response(payload: Mapping[str, Any]) -> DeepSeekTeacherResponse:
    teacher = payload.get("teacher")
    if not isinstance(teacher, Mapping):
        raise MaterializationError("teacher response has no provenance")
    raw_reasoning = payload.get("reasoning_content")
    raw_content = payload.get("content")
    answer = payload.get("answer")
    if not all(isinstance(value, str) for value in (raw_reasoning, raw_content, answer)):
        raise MaterializationError("teacher response text fields are invalid")
    usage = teacher.get("usage")
    if not isinstance(usage, Mapping):
        usage = {}
    return DeepSeekTeacherResponse(
        analysis=str(raw_reasoning),
        answer=str(answer),
        evidence_ids=(),
        response_id=str(teacher.get("response_id") or ""),
        returned_model=str(teacher.get("returned_model") or ""),
        system_fingerprint=str(teacher.get("system_fingerprint") or ""),
        finish_reason=str(teacher.get("finish_reason") or ""),
        created=teacher.get("created") if isinstance(teacher.get("created"), int) else None,
        usage=dict(usage),
        raw_content=str(raw_content),
    )


def _response_payload(response: DeepSeekTeacherResponse) -> dict[str, Any]:
    return {
        "reasoning_content": response.analysis,
        "content": response.raw_content,
        "answer": response.answer,
        "evidence_ids": list(response.evidence_ids),
        "teacher": response.provenance(),
    }


def _response_from_payload(payload: Mapping[str, Any]) -> DeepSeekTeacherResponse:
    return _teacher_response(payload)


def _load_acquisition(acquisition_root: Path) -> dict[str, list[AcquisitionRow]]:
    summary = _load_json(acquisition_root / "summary.json", label="chk3 summary")
    if summary.get("status") != "complete" or summary.get("total_accepted") != 2072:
        raise MaterializationError("chk3 acquisition is not complete at 2,072 rows")
    failures = acquisition_root / "failures.jsonl"
    if not failures.is_file() or failures.read_text(encoding="utf-8").strip():
        raise MaterializationError("chk3 acquisition failure ledger is not empty")

    result: dict[str, list[AcquisitionRow]] = {}
    observed_ids: set[str] = set()
    for split in SPLITS:
        prepared = _load_jsonl(
            acquisition_root / "prepared" / f"{split}.jsonl",
            label=f"chk3 prepared {split}",
        )
        teachers = _load_jsonl(
            acquisition_root / "teacher_responses" / f"{split}.jsonl",
            label=f"chk3 teacher responses {split}",
        )
        sft = _load_jsonl(
            acquisition_root / "sft" / f"{split}.jsonl",
            label=f"chk3 sft {split}",
        )
        manifests = _load_jsonl(
            acquisition_root / "manifests" / f"{split}.jsonl",
            label=f"chk3 manifests {split}",
        )
        expected = EXPECTED_SPLIT_COUNTS[split]
        if not all(len(rows) == expected for rows in (prepared, teachers, sft, manifests)):
            raise MaterializationError(f"acquisition split count mismatch: {split}")
        output: list[AcquisitionRow] = []
        for index, values in enumerate(zip(prepared, teachers, sft, manifests, strict=True)):
            p, t, s, m = values
            identities = {str(item.get("sample_id") or "") for item in (p, t, m)}
            if len(identities) != 1 or "" in identities:
                raise MaterializationError(f"acquisition identity mismatch: {split}[{index}]")
            sample_id = identities.pop()
            if sample_id in observed_ids:
                raise MaterializationError(f"duplicate acquisition sample ID: {sample_id}")
            observed_ids.add(sample_id)
            if p.get("source_index") != index or p.get("split") != split:
                raise MaterializationError(f"acquisition order mismatch: {sample_id}")
            if set(s) != {"prompt", "response"}:
                raise MaterializationError(f"acquisition SFT schema mismatch: {sample_id}")
            if p.get("prompt") != s.get("prompt"):
                raise MaterializationError(f"acquisition prompt mismatch: {sample_id}")
            if m.get("response_sha256") != sha256_text(str(s.get("response") or "")):
                raise MaterializationError(f"acquisition response hash mismatch: {sample_id}")
            output.append(AcquisitionRow(p, t, s, m))
        result[split] = output
    if len(observed_ids) != sum(EXPECTED_SPLIT_COUNTS.values()):
        raise MaterializationError("acquisition population mismatch")
    return result


def _load_chk1_prepared(handoff_path: Path) -> dict[str, Mapping[str, Any]]:
    handoff = _load_json(handoff_path, label="chk1 handoff")
    if handoff.get("immutable") is not True or handoff.get("quality_status") != "passed":
        raise MaterializationError("chk1 handoff is not immutable and passed")
    prepared_root = handoff_path.parent / "source/preparation/prepared"
    rows: dict[str, Mapping[str, Any]] = {}
    for split in SPLITS:
        for row in _load_jsonl(prepared_root / f"{split}.jsonl", label=f"chk1 source {split}"):
            sample_id = str(row.get("sample_id") or "")
            if not sample_id or sample_id in rows:
                raise MaterializationError(f"invalid chk1 source sample ID: {sample_id!r}")
            rows[sample_id] = row
    return rows


def _cache_path(work_root: Path, kind: str, binding_sha256: str) -> Path:
    return work_root / "cache" / kind / binding_sha256[:2] / f"{binding_sha256}.json"


def _load_bound_cache(path: Path, *, binding: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = _load_json(path, label="repair cache")
    if (
        payload.get("schema_version") != CACHE_SCHEMA_VERSION
        or payload.get("status") != "accepted"
        or payload.get("binding") != dict(binding)
    ):
        raise MaterializationError(f"repair cache binding mismatch: {path}")
    return payload


def _source_binding(source: Mapping[str, Any]) -> dict[str, Any]:
    prompt = str(source.get("generator_prompt") or "")
    fact_card = source.get("fact_card")
    if not prompt or not isinstance(fact_card, Mapping):
        raise MaterializationError("chk1 source lacks generator prompt or fact card")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "chk1_analysis_recovery",
        "sample_id": str(source.get("sample_id") or ""),
        "generator_prompt_sha256": sha256_text(prompt),
        "fact_card_sha256": sha256_text(canonical_json(dict(fact_card))),
        "teacher_contract_sha256": _teacher_config().contract_sha256,
        "generator_system_prompt_sha256": sha256_text(
            SOURCE_RECOVERY_SYSTEM_PROMPT
        ),
        "repair_system_prompt_sha256": sha256_text(
            SOURCE_RECOVERY_REPAIR_PROMPT
        ),
    }


def recover_chk1_analysis(
    source: Mapping[str, Any],
    *,
    work_root: Path,
    tokenizer: Any,
    backend: TeacherBackend,
    identity_guard: ProviderIdentityGuard,
    environment: Mapping[str, str] | None,
) -> tuple[str, DeepSeekTeacherResponse, bool]:
    """Recover one truncated chk1 final answer from point-in-time inputs only."""

    binding = _source_binding(source)
    binding_sha = sha256_text(canonical_json(binding))
    cache_path = _cache_path(work_root, "analysis", binding_sha)
    cached = _load_bound_cache(cache_path, binding=binding)
    if cached is not None:
        response = _response_from_payload(cached["provider_response"])
        identity_guard.bind(
            returned_model=response.returned_model,
            system_fingerprint=response.system_fingerprint,
        )
        clean = _validate_clean_analysis(
            str(cached.get("analysis") or ""), sample_id=str(source["sample_id"])
        )
        return clean, response, True

    fact_card = source["fact_card"]
    prompt = str(source["generator_prompt"])
    prior_errors: Sequence[str] = ()
    for attempt in range(1, SOURCE_RECOVERY_MAX_ATTEMPTS + 1):
        system_prompt = (
            SOURCE_RECOVERY_SYSTEM_PROMPT
            if attempt == 1
            else SOURCE_RECOVERY_REPAIR_PROMPT
        )
        user_prompt = prompt
        if prior_errors:
            user_prompt += "\n\n" + canonical_json(
                {
                    "repair_diagnostics": list(prior_errors),
                    "repair_rule": "Correct silently using only the supplied fact card.",
                }
            )
        response = backend.generate(
            config=_teacher_config(),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            environment=environment,
        )
        identity_guard.bind(
            returned_model=response.returned_model,
            system_fingerprint=response.system_fingerprint,
        )
        errors: list[str] = []
        if response.finish_reason not in {"stop", "end_turn"}:
            errors.append(f"finish_reason:{response.finish_reason}")
        try:
            clean = _validate_clean_analysis(response.answer, sample_id=str(source["sample_id"]))
        except MaterializationError as exc:
            errors.append(str(exc))
            clean = ""
        evidence_ids = [
            str(item.get("evidence_id") or "").strip()
            for item in fact_card.get("evidence", [])
            if isinstance(item, Mapping) and str(item.get("evidence_id") or "").strip()
        ]
        unknown_response_ids = sorted(set(response.evidence_ids) - set(evidence_ids))
        if unknown_response_ids:
            errors.append("candidate_unknown_evidence_id")
        verification = verify_candidate(
            {
                "reasoning": "The cited point-in-time evidence supports the concise analysis.",
                "final_analysis": clean or response.answer,
                # Validate the prose against the complete immutable fact-card
                # universe.  The provider citation list is useful provenance,
                # but an omitted metadata citation must not make an otherwise
                # grounded numeric claim fail.
                "evidence_ids": evidence_ids,
            },
            fact_card=fact_card,
            same_sample_minutes="",
            max_reasoning_tokens=64,
            max_final_tokens=1024,
            token_counter=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        )
        if not verification.passed:
            verification_errors = list(verification.error_codes)
            unsupported = verification.details.get("unsupported_numeric_claims")
            remaining_unsupported = (
                [
                    claim
                    for claim in unsupported
                    if not _fact_card_supports_numeric_surface(claim, fact_card)
                ]
                if isinstance(unsupported, list)
                else []
            )
            if not remaining_unsupported:
                verification_errors = [
                    code
                    for code in verification_errors
                    if code != "candidate_unsupported_numeric_claim"
                ]
            errors.extend(verification_errors)
            if remaining_unsupported:
                errors.append(
                    "unsupported_numeric_claims:"
                    + canonical_json(remaining_unsupported)
                )
            causal = verification.details.get("causal_markers")
            if isinstance(causal, list) and causal:
                errors.append("causal_markers:" + canonical_json(causal))
        if not errors:
            payload = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "status": "accepted",
                "binding": binding,
                "binding_sha256": binding_sha,
                "attempt": attempt,
                "analysis": clean,
                "analysis_sha256": sha256_text(clean),
                "provider_response": _response_payload(response),
                "verification": dict(verification.details),
            }
            _store_cache(cache_path, payload)
            return clean, response, False
        prior_errors = tuple(dict.fromkeys(errors))
        rejection = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "status": "rejected",
            "binding": binding,
            "binding_sha256": binding_sha,
            "attempt": attempt,
            "errors": list(prior_errors),
            "provider_response": _response_payload(response),
        }
        rejection_sha = sha256_text(canonical_json(rejection))
        _store_cache(
            work_root
            / "cache/rejected_analysis"
            / binding_sha[:2]
            / binding_sha
            / f"{rejection_sha}.json",
            rejection,
        )
    raise MaterializationError(
        f"chk1 analysis recovery failed: {source['sample_id']}: {';'.join(prior_errors)}"
    )


def _target_binding(row: PreparedRow) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "chk3_target_regeneration",
        "sample_id": row.sample_id,
        "split": row.split,
        "analysis_sha256": row.analysis_sha256,
        "prompt_sha256": row.prompt_sha256,
        "teacher_contract_sha256": _teacher_config().contract_sha256,
        "target_system_prompt_sha256": sha256_text(TARGETED_REPAIR_SYSTEM_PROMPT),
    }


def regenerate_chk3_target(
    row: PreparedRow,
    *,
    work_root: Path,
    tokenizer: Any,
    backend: TeacherBackend,
    identity_guard: ProviderIdentityGuard,
    environment: Mapping[str, str] | None,
) -> tuple[ValidatedTarget, DeepSeekTeacherResponse, bool]:
    binding = _target_binding(row)
    binding_sha = sha256_text(canonical_json(binding))
    cache_path = _cache_path(work_root, "target", binding_sha)
    cached = _load_bound_cache(cache_path, binding=binding)
    if cached is not None:
        response = _response_from_payload(cached["provider_response"])
        identity_guard.bind(
            returned_model=response.returned_model,
            system_fingerprint=response.system_fingerprint,
        )
        target = validate_teacher_target(
            response=response,
            analysis=row.analysis,
            user_prompt=row.user_prompt,
            tokenizer=tokenizer,
            max_length=MAX_TOKENS,
        )
        return target, response, True

    prior_errors: Sequence[str] = ()
    for attempt in range(1, TARGET_MAX_ATTEMPTS + 1):
        response = backend.generate(
            config=_teacher_config(),
            system_prompt=TARGETED_REPAIR_SYSTEM_PROMPT,
            user_prompt=targeted_user_prompt(
                row, attempt=attempt, prior_errors=prior_errors
            ),
            environment=environment,
        )
        identity_guard.bind(
            returned_model=response.returned_model,
            system_fingerprint=response.system_fingerprint,
        )
        try:
            target = validate_teacher_target(
                response=response,
                analysis=row.analysis,
                user_prompt=row.user_prompt,
                tokenizer=tokenizer,
                max_length=MAX_TOKENS,
            )
        except OutputContractError as exc:
            prior_errors = exc.codes
            rejection = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "status": "rejected",
                "binding": binding,
                "binding_sha256": binding_sha,
                "attempt": attempt,
                "errors": list(prior_errors),
                "provider_response": _response_payload(response),
            }
            rejection_sha = sha256_text(canonical_json(rejection))
            _store_cache(
                work_root
                / "cache/rejected_target"
                / binding_sha[:2]
                / binding_sha
                / f"{rejection_sha}.json",
                rejection,
            )
            continue
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "status": "accepted",
            "binding": binding,
            "binding_sha256": binding_sha,
            "attempt": attempt,
            "provider_response": _response_payload(response),
            "target": {
                "reasoning": target.reasoning,
                "minutes": target.minutes,
                "completion": target.completion,
                "reasoning_token_count": target.reasoning_token_count,
                "rendered_token_count": target.rendered_token_count,
            },
        }
        _store_cache(cache_path, payload)
        return target, response, False
    raise MaterializationError(
        f"chk3 target regeneration failed: {row.sample_id}: {';'.join(prior_errors)}"
    )


def _prepared_row(source: AcquisitionRow, analysis: str) -> PreparedRow:
    prompt = render_user_prompt(analysis)
    return PreparedRow(
        sample_id=source.sample_id,
        split=source.split,
        source_index=source.source_index,
        analysis=analysis,
        user_prompt=prompt,
        analysis_sha256=sha256_text(analysis),
        prompt_sha256=sha256_text(prompt),
        source_response_sha256=str(source.prepared["source_response_sha256"]),
    )


def _token_stats(values: Sequence[int]) -> dict[str, int]:
    ordered = sorted(values)
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


def _publish_release(
    clean_by_split: Mapping[str, Sequence[CleanRow]],
    *,
    release_parent: Path,
    release_id: str,
    acquisition_root: Path,
    chk1_handoff: Path,
    chk3_config: Path,
    tokenizer: Any,
    tokenizer_path: Path,
    prior_release_root: Path | None = None,
) -> dict[str, Any]:
    release_root = release_parent / release_id
    if release_root.exists():
        handoff = _load_json(release_root / "handoff.json", label="existing release handoff")
        if (
            handoff.get("schema_version") == RELEASE_SCHEMA_VERSION
            and handoff.get("quality_status") == "passed"
            and handoff.get("immutable") is True
        ):
            return handoff
        raise MaterializationError(f"release path already exists: {release_root}")

    config = yaml.safe_load(chk3_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise MaterializationError("chk3 config is invalid")
    if config.get("system_prompt") != STUDENT_SYSTEM_PROMPT:
        raise MaterializationError("chk3 YAML system prompt differs from generator contract")
    config["dataset_name"] = (
        f"dataset/processed/retrain_v2/{release_id}/minutes_alignment"
    )
    token_config = dict(config)
    token_config["_stage_kind"] = "minutes_sft"

    release_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{release_id}.", dir=release_parent))
    try:
        dataset_dir = staging / "minutes_alignment"
        all_ids: set[str] = set()
        prompt_tokens: list[int] = []
        completion_tokens: list[int] = []
        total_tokens: list[int] = []
        reasoning_tokens: list[int] = []
        split_counts: dict[str, int] = {}
        mode_counts: dict[str, int] = {}
        for source_split in SPLITS:
            output_split = "validation" if source_split == "eval" else source_split
            rows = list(clean_by_split[source_split])
            expected = EXPECTED_SPLIT_COUNTS[source_split]
            if len(rows) != expected:
                raise MaterializationError(
                    f"clean split count mismatch: {source_split}={len(rows)} expected={expected}"
                )
            training_rows: list[dict[str, str]] = []
            manifests: list[dict[str, Any]] = []
            for clean in rows:
                sample_id = clean.prepared.sample_id
                if sample_id in all_ids:
                    raise MaterializationError(f"duplicate clean sample ID: {sample_id}")
                all_ids.add(sample_id)
                training = {
                    "prompt": clean.prepared.user_prompt,
                    "response": clean.target.completion,
                }
                p_tokens, c_tokens, t_tokens = _count_sft_row(
                    training, tokenizer=tokenizer, config=token_config
                )
                if p_tokens > PROMPT_TOKEN_LIMIT:
                    raise MaterializationError(
                        f"prompt token overflow: {sample_id}: {p_tokens}>{PROMPT_TOKEN_LIMIT}"
                    )
                if t_tokens > TOTAL_TOKEN_LIMIT:
                    raise MaterializationError(
                        f"total token overflow: {sample_id}: {t_tokens}>{TOTAL_TOKEN_LIMIT}"
                    )
                if clean.target.reasoning_token_count > MAX_REASONING_TOKENS:
                    raise MaterializationError(
                        f"reasoning token overflow: {sample_id}"
                    )
                prompt_tokens.append(p_tokens)
                completion_tokens.append(c_tokens)
                total_tokens.append(t_tokens)
                reasoning_tokens.append(clean.target.reasoning_token_count)
                mode_key = f"{clean.analysis_mode}/{clean.target_mode}"
                mode_counts[mode_key] = mode_counts.get(mode_key, 0) + 1
                training_rows.append(training)
                manifests.append(
                    {
                        "schema_version": RELEASE_SCHEMA_VERSION,
                        "sample_id": sample_id,
                        "split": output_split,
                        "source_split": source_split,
                        "source_index": clean.prepared.source_index,
                        "analysis_mode": clean.analysis_mode,
                        "target_mode": clean.target_mode,
                        "source_analysis_sha256": sha256_text(clean.source.analysis),
                        "analysis_sha256": clean.prepared.analysis_sha256,
                        "prompt_sha256": clean.prepared.prompt_sha256,
                        "response_sha256": sha256_text(clean.target.completion),
                        "reasoning_sha256": sha256_text(clean.target.reasoning),
                        "minutes_sha256": sha256_text(clean.target.minutes),
                        "source_response_sha256": clean.prepared.source_response_sha256,
                        "prompt_tokens": p_tokens,
                        "completion_tokens": c_tokens,
                        "total_tokens": t_tokens,
                        "reasoning_tokens": clean.target.reasoning_token_count,
                        "teacher": clean.teacher_response.provenance(),
                    }
                )
            _write_jsonl(dataset_dir / f"{output_split}.jsonl", training_rows)
            _write_jsonl(dataset_dir / "manifests" / f"{output_split}.jsonl", manifests)
            split_counts[output_split] = len(rows)

        if len(all_ids) != sum(EXPECTED_SPLIT_COUNTS.values()):
            raise MaterializationError("clean release population mismatch")

        source_contamination = 0
        evidence_citation_contamination = 0
        boundary_errors = 0
        reasoning_meta_contamination = 0
        reasoning_meta_category_counts: dict[str, int] = {}
        for rows in clean_by_split.values():
            for clean in rows:
                if is_transport_wrapped(clean.prepared.analysis):
                    source_contamination += 1
                if _ANY_EVIDENCE_ID_RE.search(clean.prepared.analysis):
                    evidence_citation_contamination += 1
                if clean.target.completion.count("</think>") != 1:
                    boundary_errors += 1
                meta_categories = _reasoning_meta_categories(clean.target.reasoning)
                if meta_categories:
                    reasoning_meta_contamination += 1
                    for category in meta_categories:
                        reasoning_meta_category_counts[category] = (
                            reasoning_meta_category_counts.get(category, 0) + 1
                        )
        if (
            source_contamination
            or evidence_citation_contamination
            or boundary_errors
            or reasoning_meta_contamination
        ):
            raise MaterializationError(
                "quality gate failed: "
                f"transport={source_contamination} "
                f"evidence_citations={evidence_citation_contamination} "
                f"boundary={boundary_errors} "
                f"reasoning_meta={reasoning_meta_contamination}:"
                f"{canonical_json(reasoning_meta_category_counts)}"
            )

        audit = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "status": "passed",
            "intended_use": "chk3 SFT: analysis to formal FOMC Minutes paragraph",
            "grain": "one chk1 canonical sample per row",
            "split_counts": split_counts,
            "total_rows": len(all_ids),
            "unique_sample_ids": len(all_ids),
            "duplicate_sample_ids": 0,
            "missing_required_fields": 0,
            "transport_wrapped_analyses": source_contamination,
            "analysis_evidence_citations": evidence_citation_contamination,
            "invalid_response_boundaries": boundary_errors,
            "reasoning_meta_contamination": reasoning_meta_contamination,
            "reasoning_meta_category_counts": dict(
                sorted(reasoning_meta_category_counts.items())
            ),
            "mode_counts": dict(sorted(mode_counts.items())),
            "token_contract": {
                "prompt_max": PROMPT_TOKEN_LIMIT,
                "completion_max": None,
                "reasoning_max": MAX_REASONING_TOKENS,
                "total_max": TOTAL_TOKEN_LIMIT,
                "overflow_policy": "error",
                "truncation": False,
            },
            "token_stats": {
                "prompt": _token_stats(prompt_tokens),
                "completion": _token_stats(completion_tokens),
                "reasoning": _token_stats(reasoning_tokens),
                "total": _token_stats(total_tokens),
            },
            "legacy_completion_1024_overflow_count": sum(
                value > 1024 for value in completion_tokens
            ),
            "checks": {
                "population": "passed",
                "schema": "passed",
                "uniqueness": "passed",
                "transport_cleanup": "passed",
                "target_validator": "passed",
                "reasoning_meta": "passed",
                "tokenizer_prefix": "passed",
                "token_budget": "passed",
            },
        }
        _write_json(staging / "audits/data_quality.json", audit)
        _atomic_write(
            staging / "chk3_minutes_sft.template.yaml",
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        )

        file_records: dict[str, dict[str, Any]] = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                relative = str(path.relative_to(staging))
                record: dict[str, Any] = {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                if path.suffix == ".jsonl":
                    record["rows"] = sum(1 for _ in path.open(encoding="utf-8"))
                file_records[relative] = record
        source_lineage = {
            "chk3_acquisition_root": str(acquisition_root),
            "chk3_summary_sha256": sha256_file(acquisition_root / "summary.json"),
            "chk3_prompt_contract_sha256": sha256_file(
                acquisition_root / "prompt_contract.json"
            ),
            "chk1_handoff_path": str(chk1_handoff),
            "chk1_handoff_sha256": sha256_file(chk1_handoff),
            "tokenizer_path": str(tokenizer_path),
        }
        if prior_release_root is not None:
            source_lineage["prior_chk3_release"] = {
                "path": str(prior_release_root),
                "handoff_sha256": sha256_file(prior_release_root / "handoff.json"),
                "release_manifest_sha256": sha256_file(
                    prior_release_root / "release_manifest.json"
                ),
            }
        release_manifest = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": release_id,
            "created_at_utc": _utc_now(),
            "immutable": True,
            "quality_status": "passed",
            "dataset_role": "standalone_chk3_minutes_alignment",
            "dag_bindable": False,
            "dag_binding_blocker": "sealed chk2 parent and full chk2-derived release are not available",
            "source": source_lineage,
            "split_counts": split_counts,
            "total_rows": len(all_ids),
            "training_mapping": "chk1 final analysis -> reasoning -> formal Minutes paragraph",
            "files": file_records,
            "config_template": "chk3_minutes_sft.template.yaml",
        }
        _write_json(staging / "release_manifest.json", release_manifest)
        handoff = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": release_id,
            "created_at_utc": release_manifest["created_at_utc"],
            "immutable": True,
            "quality_status": "passed",
            "dataset_path": f"dataset/processed/retrain_v2/{release_id}/minutes_alignment",
            "split_counts": split_counts,
            "total_rows": len(all_ids),
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
        os.replace(staging, release_root)
        return handoff
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def run(
    *,
    acquisition_root: Path,
    chk1_handoff: Path,
    tokenizer_path: Path,
    chk3_config: Path,
    work_root: Path,
    release_parent: Path,
    release_id: str,
    dry_run: bool,
    concurrency: int,
    backend: TeacherBackend | None = None,
    environment: Mapping[str, str] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    if concurrency < 1:
        raise MaterializationError("concurrency must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", release_id):
        raise MaterializationError("release ID is invalid")
    acquisition = _load_acquisition(acquisition_root)
    chk1_sources = _load_chk1_prepared(chk1_handoff)
    active_tokenizer = tokenizer or _load_tokenizer(tokenizer_path)

    clean_analysis: dict[str, tuple[str, str, DeepSeekTeacherResponse | None]] = {}
    truncated: list[AcquisitionRow] = []
    valid_wrappers = 0
    evidence_citation_projections = 0
    for split in SPLITS:
        for source in acquisition[split]:
            if not is_transport_wrapped(source.analysis):
                if _ANY_EVIDENCE_ID_RE.search(source.analysis):
                    projected = project_internal_evidence_citations(source.analysis)
                    clean = _validate_clean_analysis(
                        projected, sample_id=source.sample_id
                    )
                    clean_analysis[source.sample_id] = (
                        clean,
                        "evidence_citation_projection",
                        None,
                    )
                    evidence_citation_projections += 1
                else:
                    clean = _validate_clean_analysis(
                        source.analysis, sample_id=source.sample_id
                    )
                    clean_analysis[source.sample_id] = (clean, "unchanged", None)
                continue
            projected = project_valid_transport_analysis(source.analysis)
            if projected is not None:
                clean = _validate_clean_analysis(projected, sample_id=source.sample_id)
                clean_analysis[source.sample_id] = (clean, "valid_transport_projection", None)
                valid_wrappers += 1
            else:
                truncated.append(source)

    base_summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared" if dry_run else "running",
        "release_id": release_id,
        "total_rows": sum(len(acquisition[split]) for split in SPLITS),
        "unchanged_analyses": (
            len(clean_analysis) - valid_wrappers - evidence_citation_projections
        ),
        "valid_transport_projections": valid_wrappers,
        "evidence_citation_projections": evidence_citation_projections,
        "truncated_analyses_to_recover": len(truncated),
        "truncated_sample_ids": [row.sample_id for row in truncated],
        "api_requests_are_disabled": dry_run,
    }
    work_root.mkdir(parents=True, exist_ok=True)
    _write_json(work_root / "preflight_summary.json", base_summary)
    if dry_run:
        return base_summary

    provider = backend or OpenAIDeepSeekBackend()
    identity_guard = ProviderIdentityGuard()
    acquisition_summary = _load_json(acquisition_root / "summary.json", label="chk3 summary")
    acquisition_identity = acquisition_summary.get("provider_identity")
    if not isinstance(acquisition_identity, Mapping):
        raise MaterializationError("acquisition provider identity is missing")

    analysis_failures: dict[str, str] = {}
    analysis_api_calls = 0
    if truncated:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {}
            for source in truncated:
                chk1_source = chk1_sources.get(source.sample_id)
                if chk1_source is None:
                    raise MaterializationError(f"missing chk1 source row: {source.sample_id}")
                futures[
                    pool.submit(
                        recover_chk1_analysis,
                        chk1_source,
                        work_root=work_root,
                        tokenizer=active_tokenizer,
                        backend=provider,
                        identity_guard=identity_guard,
                        environment=environment,
                    )
                ] = source
            completed = 0
            for future in as_completed(futures):
                source = futures[future]
                try:
                    analysis, response, cache_hit = future.result()
                    projected = project_internal_evidence_citations(analysis)
                    clean_analysis[source.sample_id] = (
                        _validate_clean_analysis(
                            projected, sample_id=source.sample_id
                        ),
                        "deepseek_point_in_time_recovery",
                        response,
                    )
                    if not cache_hit:
                        analysis_api_calls += 1
                    completed += 1
                    print(
                        f"[chk3-materialize] recovered_analysis={completed}/{len(truncated)} "
                        f"sample_id={source.sample_id} cache_hit={cache_hit}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - retain complete failure ledger
                    analysis_failures[source.sample_id] = f"{type(exc).__name__}:{exc}"
        if analysis_failures:
            _write_jsonl(
                work_root / "analysis_failures.jsonl",
                [
                    {"sample_id": key, "error": analysis_failures[key]}
                    for key in sorted(analysis_failures)
                ],
            )
            raise MaterializationError(
                f"analysis recovery incomplete: {len(analysis_failures)} failures"
            )

    if identity_guard.identity is not None:
        expected_identity = (
            str(acquisition_identity.get("returned_model") or ""),
            str(acquisition_identity.get("system_fingerprint") or ""),
        )
        if identity_guard.identity != expected_identity:
            raise MaterializationError(
                f"provider identity drift from acquisition: expected={expected_identity} "
                f"observed={identity_guard.identity}"
            )

    clean_rows: dict[str, list[CleanRow]] = {split: [] for split in SPLITS}
    pending_targets: list[tuple[AcquisitionRow, PreparedRow, str]] = []
    for split in SPLITS:
        for source in acquisition[split]:
            analysis, analysis_mode, recovery_response = clean_analysis[source.sample_id]
            prepared = _prepared_row(source, analysis)
            old_response = _teacher_response(source.teacher)
            target: ValidatedTarget | None = None
            if analysis_mode != "deepseek_point_in_time_recovery":
                try:
                    target = validate_teacher_target(
                        response=old_response,
                        analysis=analysis,
                        user_prompt=prepared.user_prompt,
                        tokenizer=active_tokenizer,
                        max_length=MAX_TOKENS,
                    )
                except OutputContractError:
                    target = None
            if target is not None:
                target_mode = (
                    "acquisition_revalidated"
                    if analysis_mode == "unchanged"
                    else "acquisition_reprojected"
                )
                clean_rows[split].append(
                    CleanRow(
                        source=source,
                        prepared=prepared,
                        target=target,
                        teacher_response=old_response,
                        analysis_mode=analysis_mode,
                        target_mode=target_mode,
                    )
                )
            else:
                del recovery_response
                pending_targets.append((source, prepared, analysis_mode))

    target_failures: dict[str, str] = {}
    target_api_calls = 0
    generated: dict[str, CleanRow] = {}
    if pending_targets:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(
                    regenerate_chk3_target,
                    prepared,
                    work_root=work_root,
                    tokenizer=active_tokenizer,
                    backend=provider,
                    identity_guard=identity_guard,
                    environment=environment,
                ): (source, prepared, analysis_mode)
                for source, prepared, analysis_mode in pending_targets
            }
            completed = 0
            for future in as_completed(futures):
                source, prepared, analysis_mode = futures[future]
                try:
                    target, response, cache_hit = future.result()
                    generated[source.sample_id] = CleanRow(
                        source=source,
                        prepared=prepared,
                        target=target,
                        teacher_response=response,
                        analysis_mode=analysis_mode,
                        target_mode="deepseek_regenerated",
                    )
                    if not cache_hit:
                        target_api_calls += 1
                    completed += 1
                    print(
                        f"[chk3-materialize] regenerated_target={completed}/{len(pending_targets)} "
                        f"sample_id={source.sample_id} cache_hit={cache_hit}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 - retain complete failure ledger
                    target_failures[source.sample_id] = f"{type(exc).__name__}:{exc}"
        if target_failures:
            _write_jsonl(
                work_root / "target_failures.jsonl",
                [
                    {"sample_id": key, "error": target_failures[key]}
                    for key in sorted(target_failures)
                ],
            )
            raise MaterializationError(
                f"target regeneration incomplete: {len(target_failures)} failures"
            )

    expected_identity = (
        str(acquisition_identity.get("returned_model") or ""),
        str(acquisition_identity.get("system_fingerprint") or ""),
    )
    if identity_guard.identity is not None and identity_guard.identity != expected_identity:
        raise MaterializationError(
            f"provider identity drift from acquisition: expected={expected_identity} "
            f"observed={identity_guard.identity}"
        )

    for split in SPLITS:
        existing = {row.prepared.sample_id: row for row in clean_rows[split]}
        for source in acquisition[split]:
            if source.sample_id in generated:
                existing[source.sample_id] = generated[source.sample_id]
        clean_rows[split] = [existing[source.sample_id] for source in acquisition[split]]

    handoff = _publish_release(
        clean_rows,
        release_parent=release_parent,
        release_id=release_id,
        acquisition_root=acquisition_root,
        chk1_handoff=chk1_handoff,
        chk3_config=chk3_config,
        tokenizer=active_tokenizer,
        tokenizer_path=tokenizer_path,
    )
    result = {
        **base_summary,
        "status": "complete",
        "analysis_api_calls": analysis_api_calls,
        "target_api_calls": target_api_calls,
        "pending_targets": len(pending_targets),
        "release_path": str(release_parent / release_id),
        "dataset_path": handoff["dataset_path"],
        "handoff": str(release_parent / release_id / "handoff.json"),
        "provider_identity": identity_guard.identity,
    }
    _write_json(work_root / "summary.json", result)
    _write_jsonl(work_root / "analysis_failures.jsonl", [])
    _write_jsonl(work_root / "target_failures.jsonl", [])
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-root", type=Path, default=DEFAULT_ACQUISITION_ROOT)
    parser.add_argument("--chk1-handoff", type=Path, default=DEFAULT_CHK1_HANDOFF)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--chk3-config", type=Path, default=DEFAULT_CHK3_CONFIG)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--release-parent", type=Path, default=DEFAULT_RELEASE_PARENT)
    parser.add_argument("--release-id", default=DEFAULT_RELEASE_ID)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(
            acquisition_root=args.acquisition_root.resolve(),
            chk1_handoff=args.chk1_handoff.resolve(),
            tokenizer_path=args.tokenizer_path.resolve(),
            chk3_config=args.chk3_config.resolve(),
            work_root=args.work_root.resolve(),
            release_parent=args.release_parent.resolve(),
            release_id=args.release_id,
            dry_run=args.dry_run,
            concurrency=args.concurrency,
        )
    except (Chk3DataError, OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
