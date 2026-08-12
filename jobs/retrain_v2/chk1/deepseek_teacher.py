"""DeepSeek Reasoner teacher for chk1 analysis-SFT target distillation.

The remote teacher owns both semantic parts of the target: DeepSeek's
``reasoning_content`` becomes the distilled analysis and the JSON ``answer``
in ``content`` becomes the final answer.  Qwen is intentionally absent from
this module; it is reserved for the chk2 GRPO LLM-reward service.

No local verifier, critic, reward model, or semantic repair pass participates
in this acquisition step.  A strict structural contract prevents malformed or
format-polluted provider output from entering SFT data, while the accepted
provider text is still cached byte-for-byte alongside its normalized fields.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlparse

from open_r1.provenance import validate_sha256

from .contracts import (
    CANDIDATE_SCHEMA_VERSION,
    GENERATOR_SYSTEM_PROMPT,
    canonical_json,
    sha256_text,
)
from .local_models import (
    CacheIntegrityError,
    ModelOutputContractError,
)
DEEPSEEK_TEACHER_MODEL = "deepseek-v4-pro"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_MODEL_ENV = "DEEPSEEK_TEACHER_MODEL"
DEEPSEEK_BASE_URL_ENV = "DEEPSEEK_BASE_URL"
DEEPSEEK_REVISION_ENV = "DEEPSEEK_TEACHER_REVISION"
DEEPSEEK_CACHE_SCHEMA_VERSION = "chk1-deepseek-teacher-cache-v4"
DEEPSEEK_TEACHER_CONTRACT_VERSION = "chk1-deepseek-v4-thinking-teacher-v3"
DEEPSEEK_REJECTED_ATTEMPT_SCHEMA_VERSION = (
    "chk1-deepseek-teacher-rejected-attempt-v1"
)

_EVIDENCE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_INLINE_EV_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])ev-[A-Za-z0-9][A-Za-z0-9_-]*(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)
_GENERIC_EVIDENCE_CITATION_RE = re.compile(
    r"(?:\(|\[)\s*[a-z]+\d+(?:\s*[,;]\s*[a-z]+\d+)*\s*(?:\)|\])|"
    r"\bevidence(?:\s+id)?\s*[:#]?\s*[a-z]+\d+\b",
    re.IGNORECASE,
)
_SCHEMA_META_RE = re.compile(
    r"\b(?:json|schema|evidence[_ -]?ids?|reasoning_content|"
    r"output[_ -]?contract|final[_ -]?answer)\b",
    re.IGNORECASE,
)
_CONTROL_TOKEN_RE = re.compile(
    r"<\s*/?\s*(?:think|answer|analysis|reasoning)\s*>|"
    r"<\|[^>\r\n]+\|>|\[(?:/?INST|SYSTEM|USER|ASSISTANT)\]",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>\r\n]{0,80}>")
_MARKDOWN_RE = re.compile(
    r"```|~~~|`[^`\r\n]+`|\*\*|__|!?\[[^\]\r\n]+\]\([^\)\r\n]+\)|"
    r"(?:^|\n)\s*(?:#{1,6}\s|>\s|[-*+]\s|\d+[.)]\s)",
    re.MULTILINE,
)


class TeacherResponseRejectedError(RuntimeError):
    """Raised after all configured DeepSeek content-contract attempts fail."""

    def __init__(self, message: str, *, payload: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.payload = dict(payload)


@dataclass(frozen=True)
class DeepSeekTeacherConfig:
    """Non-secret request contract for the advanced teacher."""

    model: str = DEEPSEEK_TEACHER_MODEL
    base_url: str = DEEPSEEK_BASE_URL
    api_key_env: str = DEEPSEEK_API_KEY_ENV
    revision: str = "provider-current"
    max_tokens: int = 4096
    timeout_seconds: float = 180.0
    max_retries: int = 3
    reasoning_effort: str = "high"

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> "DeepSeekTeacherConfig":
        env = os.environ if environment is None else environment
        return cls(
            model=str(env.get(DEEPSEEK_MODEL_ENV) or DEEPSEEK_TEACHER_MODEL).strip(),
            base_url=str(env.get(DEEPSEEK_BASE_URL_ENV) or DEEPSEEK_BASE_URL).strip(),
            revision=str(env.get(DEEPSEEK_REVISION_ENV) or "provider-current").strip(),
        )

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("DeepSeek teacher model must be non-empty")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("DeepSeek base URL must be an absolute HTTP(S) URL")
        if parsed.scheme != "https" and parsed.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("remote DeepSeek endpoints must use HTTPS")
        if not self.api_key_env or not self.api_key_env.isidentifier():
            raise ValueError("DeepSeek API key environment name is invalid")
        if self.max_tokens != 4096:
            raise ValueError("DeepSeek teacher max_tokens must be 4096")
        if self.max_retries != 3:
            raise ValueError("DeepSeek teacher max_retries must be 3")
        if self.reasoning_effort != "high":
            raise ValueError("DeepSeek teacher reasoning_effort must be high")

    def contract(self) -> dict[str, Any]:
        return {
            "schema_version": DEEPSEEK_TEACHER_CONTRACT_VERSION,
            "provider": "deepseek",
            "model": self.model,
            "revision": self.revision,
            "base_url_origin": _endpoint_origin(self.base_url),
            "api_key_env": self.api_key_env,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "response_format": {"type": "json_object"},
            "reasoning_effort": self.reasoning_effort,
            "thinking": {"type": "enabled"},
            "analysis_source": "message.reasoning_content",
            "answer_source": "message.content.answer",
            "acceptance_contract": {
                "finish_reason": "stop",
                "content_required_keys": ["answer", "evidence_ids"],
                "content_additional_properties": False,
                "reasoning_non_empty": True,
                "answer_format": "plain_text_without_inline_evidence_ids",
                "evidence_ids_separate_non_empty_unique_list": True,
                "evidence_ids_must_reference_fact_card": True,
            },
            "unsupported_sampling_parameters": ["temperature", "top_p"],
        }

    @property
    def contract_sha256(self) -> str:
        return sha256_text(canonical_json(self.contract()))


@dataclass(frozen=True)
class DeepSeekTeacherResponse:
    analysis: str
    answer: str
    evidence_ids: tuple[str, ...]
    response_id: str
    returned_model: str
    system_fingerprint: str
    finish_reason: str
    created: int | None
    usage: Mapping[str, int | None]
    raw_content: str = ""
    rejected_attempts: tuple[Mapping[str, Any], ...] = ()
    accepted_attempt: int = 1

    def candidate(self) -> dict[str, Any]:
        # The SFT/release layer retains its historical internal names; the
        # explicit teacher_output below remains analysis/answer.
        return {
            "reasoning": self.analysis.strip(),
            "final_analysis": self.answer.strip(),
            "evidence_ids": [item.strip() for item in self.evidence_ids],
        }

    def teacher_output(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis.strip(),
            "answer": self.answer.strip(),
            "evidence_ids": [item.strip() for item in self.evidence_ids],
        }

    def provider_raw(self) -> dict[str, str]:
        content = self.raw_content
        if not content:
            content = canonical_json(
                {
                    "answer": self.answer.strip(),
                    "evidence_ids": [item.strip() for item in self.evidence_ids],
                }
            )
        return {
            "reasoning_content": self.analysis,
            "content": content,
        }

    def provenance(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "returned_model": self.returned_model,
            "system_fingerprint": self.system_fingerprint,
            "finish_reason": self.finish_reason,
            "created": self.created,
            "usage": dict(self.usage),
        }


class DeepSeekTeacherBackend(Protocol):
    def generate(
        self,
        *,
        config: DeepSeekTeacherConfig,
        system_prompt: str,
        user_prompt: str,
        environment: Mapping[str, str] | None,
        allowed_evidence_ids: frozenset[str] | None = None,
    ) -> DeepSeekTeacherResponse: ...


class OpenAIDeepSeekBackend:
    """Lazy OpenAI-compatible DeepSeek client with explicit retry control."""

    def generate(
        self,
        *,
        config: DeepSeekTeacherConfig,
        system_prompt: str,
        user_prompt: str,
        environment: Mapping[str, str] | None,
        allowed_evidence_ids: frozenset[str] | None = None,
    ) -> DeepSeekTeacherResponse:
        env = os.environ if environment is None else environment
        api_key = str(env.get(config.api_key_env) or "").strip()
        if not api_key:
            raise ModelOutputContractError(
                f"missing DeepSeek teacher credential: {config.api_key_env}"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment contract
            raise ModelOutputContractError("openai package is unavailable") from exc

        client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=0,
        )
        last_error: Exception | None = None
        rejected_attempts: list[dict[str, Any]] = []
        for attempt in range(config.max_retries + 1):
            raw_attempt_context: tuple[str, str, Mapping[str, Any]] | None = None
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
                    raise ModelOutputContractError(
                        "DeepSeek teacher returned no completion choices"
                    )
                choice = completion.choices[0]
                message = choice.message
                analysis = str(getattr(message, "reasoning_content", "") or "")
                content = str(getattr(message, "content", "") or "")
                usage = _usage_payload(getattr(completion, "usage", None))
                provenance = {
                    "response_id": str(getattr(completion, "id", "") or ""),
                    "returned_model": str(
                        getattr(completion, "model", "") or config.model
                    ),
                    "system_fingerprint": str(
                        getattr(completion, "system_fingerprint", "") or "unavailable"
                    ),
                    "finish_reason": str(
                        getattr(choice, "finish_reason", "") or "unknown"
                    ),
                    "created": _optional_int(getattr(completion, "created", None)),
                    "usage": usage,
                }
                raw_attempt_context = (analysis, content, provenance)
                answer, evidence_ids = _parse_content(
                    content, allowed_evidence_ids=allowed_evidence_ids
                )
                validated = _validated_response(
                    DeepSeekTeacherResponse(
                        analysis=analysis,
                        answer=answer,
                        evidence_ids=evidence_ids,
                        response_id=str(provenance["response_id"]),
                        returned_model=str(provenance["returned_model"]),
                        system_fingerprint=str(provenance["system_fingerprint"]),
                        finish_reason=str(provenance["finish_reason"]),
                        created=provenance["created"],
                        usage=usage,
                        raw_content=content,
                    ),
                    allowed_evidence_ids=allowed_evidence_ids,
                )
                return replace(
                    validated,
                    rejected_attempts=tuple(rejected_attempts),
                    accepted_attempt=attempt + 1,
                )
            except ModelOutputContractError as exc:
                # A malformed/truncated completion is retryable.  The caller
                # must never receive a best-effort answer synthesized from raw
                # provider content.
                last_error = exc
                if raw_attempt_context is not None:
                    analysis, content, provenance = raw_attempt_context
                    rejected_attempts.append(
                        _rejected_attempt_payload(
                            attempt=attempt + 1,
                            reasoning_content=analysis,
                            content=content,
                            provenance=provenance,
                            error=exc,
                        )
                    )
                if attempt >= config.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
            except Exception as exc:  # noqa: BLE001 - provider exceptions vary
                last_error = exc
                if raw_attempt_context is not None:
                    analysis, content, provenance = raw_attempt_context
                    rejected_attempts.append(
                        _rejected_attempt_payload(
                            attempt=attempt + 1,
                            reasoning_content=analysis,
                            content=content,
                            provenance=provenance,
                            error=exc,
                        )
                    )
                status_code = getattr(exc, "status_code", None)
                if status_code in {400, 401, 403, 404, 422}:
                    break
                if attempt >= config.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        assert last_error is not None
        if rejected_attempts:
            raise TeacherResponseRejectedError(
                "DeepSeek teacher exhausted response-contract retries: "
                f"{last_error}",
                payload=_rejection_payload(rejected_attempts),
            ) from last_error
        raise ModelOutputContractError(
            "DeepSeek teacher request failed closed after "
            f"{config.max_retries + 1} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error


class MockDeepSeekTeacherBackend:
    """Deterministic structured backend for unit tests."""

    def __init__(self, responses: Sequence[DeepSeekTeacherResponse]):
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        config: DeepSeekTeacherConfig,
        system_prompt: str,
        user_prompt: str,
        environment: Mapping[str, str] | None,
        allowed_evidence_ids: frozenset[str] | None = None,
    ) -> DeepSeekTeacherResponse:
        del environment
        last_error: ModelOutputContractError | None = None
        rejected_attempts: list[dict[str, Any]] = []
        for attempt in range(config.max_retries + 1):
            if not self._responses:
                break
            self.requests.append(
                {
                    "config": config.contract(),
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                }
            )
            response = self._responses.pop(0)
            try:
                validated = _validated_response(
                    response, allowed_evidence_ids=allowed_evidence_ids
                )
                return replace(
                    validated,
                    rejected_attempts=tuple(rejected_attempts),
                    accepted_attempt=attempt + 1,
                )
            except ModelOutputContractError as exc:
                last_error = exc
                raw = response.provider_raw()
                rejected_attempts.append(
                    _rejected_attempt_payload(
                        attempt=attempt + 1,
                        reasoning_content=raw["reasoning_content"],
                        content=raw["content"],
                        provenance=response.provenance(),
                        error=exc,
                    )
                )
        if last_error is not None:
            raise TeacherResponseRejectedError(
                "mock DeepSeek teacher exhausted contract retries: " f"{last_error}",
                payload=_rejection_payload(rejected_attempts),
            ) from last_error
        raise ModelOutputContractError("mock DeepSeek response queue is empty")


def _endpoint_origin(value: str) -> str:
    parsed = urlparse(value)
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_payload(value: Any) -> dict[str, int | None]:
    def get(name: str) -> int | None:
        raw = getattr(value, name, None) if value is not None else None
        return _optional_int(raw)

    return {
        "prompt_tokens": get("prompt_tokens"),
        "completion_tokens": get("completion_tokens"),
        "total_tokens": get("total_tokens"),
    }


def _contract_error_code(error: Exception) -> str:
    message = str(error).casefold()
    patterns = (
        ("finish_reason", "finish_reason_not_stop"),
        ("reasoning_content", "reasoning_empty"),
        ("duplicate json key", "content_duplicate_key"),
        ("non-finite json", "content_non_finite_json"),
        ("strict json", "content_malformed_json"),
        ("exactly answer and evidence_ids", "content_schema_invalid"),
        ("content must be non-empty", "content_empty"),
        ("inline evidence id", "answer_inline_evidence_id"),
        ("schema/meta", "answer_schema_meta"),
        ("control/markup", "answer_control_markup"),
        ("markdown", "answer_markdown"),
        ("structured data", "answer_structured_data"),
        ("content.answer must be non-empty", "answer_empty"),
        ("evidence_ids must be a non-empty list", "evidence_ids_empty"),
        ("evidence id must be trimmed", "evidence_id_not_trimmed"),
        ("evidence id has an invalid format", "evidence_id_invalid"),
        ("evidence id is absent from the fact card", "evidence_id_not_in_fact_card"),
        ("evidence ids must be unique", "evidence_ids_duplicate"),
        ("normalized fields", "provider_normalization_mismatch"),
    )
    for fragment, code in patterns:
        if fragment in message:
            return code
    return "response_contract_invalid"


def _rejected_attempt_payload(
    *,
    attempt: int,
    reasoning_content: str,
    content: str,
    provenance: Mapping[str, Any],
    error: Exception,
) -> dict[str, Any]:
    raw = {
        "reasoning_content": reasoning_content,
        "content": content,
    }
    finish_reason = str(provenance.get("finish_reason") or "unknown")
    return {
        "schema_version": DEEPSEEK_REJECTED_ATTEMPT_SCHEMA_VERSION,
        "attempt": attempt,
        "status": "rejected",
        "error_code": _contract_error_code(error),
        "finish_reason": finish_reason,
        "provider_raw": raw,
        "provider_raw_sha256": sha256_text(canonical_json(raw)),
        "reasoning_content_sha256": sha256_text(reasoning_content),
        "content_sha256": sha256_text(content),
        "teacher_provenance": dict(provenance),
    }


def _rejection_payload(
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frozen_attempts = [dict(item) for item in attempts]
    return {
        "attempts": frozen_attempts,
        "rejection_error_codes": list(
            dict.fromkeys(str(item["error_code"]) for item in frozen_attempts)
        ),
    }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ModelOutputContractError(
                f"DeepSeek teacher content has duplicate JSON key: {key}"
            )
        payload[key] = value
    return payload


def _reject_non_finite_json(value: str) -> None:
    raise ModelOutputContractError(
        f"DeepSeek teacher content contains non-finite JSON value: {value}"
    )


def _validate_analysis(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelOutputContractError(
            "DeepSeek teacher reasoning_content must be non-empty text"
        )
    return value.strip()


def _contains_bound_identifier(text: str, identifier: str) -> bool:
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_-]){re.escape(identifier)}(?![A-Za-z0-9_-])",
        re.IGNORECASE,
    )
    return pattern.search(text) is not None


def _validate_answer(
    value: Any, *, allowed_evidence_ids: frozenset[str] | None = None
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelOutputContractError(
            "DeepSeek teacher content.answer must be non-empty text"
        )
    answer = value.strip()
    if (
        _INLINE_EV_ID_RE.search(answer)
        or _GENERIC_EVIDENCE_CITATION_RE.search(answer)
        or (
            allowed_evidence_ids is not None
            and any(
                _contains_bound_identifier(answer, evidence_id)
                for evidence_id in allowed_evidence_ids
            )
        )
    ):
        raise ModelOutputContractError(
            "DeepSeek teacher content.answer contains an inline evidence ID"
        )
    if _SCHEMA_META_RE.search(answer):
        raise ModelOutputContractError(
            "DeepSeek teacher content.answer contains schema/meta text"
        )
    if _CONTROL_TOKEN_RE.search(answer) or _HTML_TAG_RE.search(answer):
        raise ModelOutputContractError(
            "DeepSeek teacher content.answer contains a control/markup tag"
        )
    if _MARKDOWN_RE.search(answer):
        raise ModelOutputContractError(
            "DeepSeek teacher content.answer contains Markdown"
        )
    if answer[0] in "[{" or answer[-1] in "]}":
        raise ModelOutputContractError(
            "DeepSeek teacher content.answer contains structured data"
        )
    return answer


def _validate_evidence_ids(
    value: Any, *, allowed_evidence_ids: frozenset[str] | None = None
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ModelOutputContractError(
            "DeepSeek teacher content.evidence_ids must be a non-empty list"
        )
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str) or item != item.strip():
            raise ModelOutputContractError(
                "DeepSeek teacher evidence ID must be trimmed text"
            )
        if not _EVIDENCE_ID_RE.fullmatch(item):
            raise ModelOutputContractError(
                "DeepSeek teacher evidence ID has an invalid format"
            )
        cleaned.append(item)
    if len(set(cleaned)) != len(cleaned):
        raise ModelOutputContractError(
            "DeepSeek teacher evidence IDs must be unique"
        )
    if allowed_evidence_ids is not None:
        missing = [item for item in cleaned if item not in allowed_evidence_ids]
        if missing:
            raise ModelOutputContractError(
                "DeepSeek teacher evidence ID is absent from the fact card"
            )
    return tuple(cleaned)


def _parse_content(
    content: str, *, allowed_evidence_ids: frozenset[str] | None = None
) -> tuple[str, tuple[str, ...]]:
    """Decode exactly ``{answer, evidence_ids}`` or fail closed."""

    text = str(content or "").strip()
    if not text:
        raise ModelOutputContractError("DeepSeek teacher content must be non-empty")
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_non_finite_json,
        )
    except ModelOutputContractError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ModelOutputContractError(
            "DeepSeek teacher content must be strict JSON"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"answer", "evidence_ids"}:
        raise ModelOutputContractError(
            "DeepSeek teacher content must contain exactly answer and evidence_ids"
        )
    return (
        _validate_answer(
            payload["answer"], allowed_evidence_ids=allowed_evidence_ids
        ),
        _validate_evidence_ids(
            payload["evidence_ids"], allowed_evidence_ids=allowed_evidence_ids
        ),
    )


def _validated_response(
    response: DeepSeekTeacherResponse,
    *,
    allowed_evidence_ids: frozenset[str] | None = None,
) -> DeepSeekTeacherResponse:
    if response.finish_reason != "stop":
        raise ModelOutputContractError(
            "DeepSeek teacher finish_reason must be stop"
        )
    _validate_analysis(response.analysis)
    answer = _validate_answer(
        response.answer, allowed_evidence_ids=allowed_evidence_ids
    )
    evidence_ids = _validate_evidence_ids(
        response.evidence_ids, allowed_evidence_ids=allowed_evidence_ids
    )
    if response.raw_content:
        raw_answer, raw_ids = _parse_content(
            response.raw_content, allowed_evidence_ids=allowed_evidence_ids
        )
        if raw_answer != answer or raw_ids != evidence_ids:
            raise ModelOutputContractError(
                "DeepSeek teacher normalized fields do not match provider content"
            )
    return response


def candidate_to_sft_response(candidate: Mapping[str, Any]) -> str:
    expected = {"reasoning", "final_analysis", "evidence_ids"}
    if set(candidate) != expected:
        raise ModelOutputContractError("candidate schema is invalid")
    analysis = candidate["reasoning"]
    answer = candidate["final_analysis"]
    evidence_ids = candidate["evidence_ids"]
    normalized_analysis = _validate_analysis(analysis)
    normalized_answer = _validate_answer(answer)
    _validate_evidence_ids(evidence_ids)
    response = f"{normalized_analysis}\n</think>\n{normalized_answer}"
    return response


def build_teacher_config(
    *, environment: Mapping[str, str] | None = None
) -> DeepSeekTeacherConfig:
    return DeepSeekTeacherConfig.from_environment(environment)


def teacher_model_provenance(config: DeepSeekTeacherConfig) -> dict[str, Any]:
    return {
        "provider": "deepseek",
        "model": config.model,
        "revision": config.revision,
        "sha256": config.contract_sha256,
    }


def _fact_card_evidence_ids(fact_card: Mapping[str, Any]) -> frozenset[str]:
    evidence = fact_card.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ModelOutputContractError(
            "DeepSeek teacher fact card must contain non-empty evidence"
        )
    identifiers: list[str] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, Mapping):
            raise ModelOutputContractError(
                f"DeepSeek teacher fact card evidence {index} must be an object"
            )
        identifier = item.get("evidence_id")
        if not isinstance(identifier, str) or identifier != identifier.strip():
            raise ModelOutputContractError(
                f"DeepSeek teacher fact card evidence {index} has an invalid ID"
            )
        identifiers.append(identifier)
    validated = _validate_evidence_ids(identifiers)
    return frozenset(validated)


def _cache_bindings(
    *,
    prompt: str,
    fact_card: Mapping[str, Any],
    same_sample_minutes: str,
    teacher: DeepSeekTeacherConfig,
) -> dict[str, Any]:
    contract = teacher.contract()
    return {
        "prompt_sha256": sha256_text(prompt),
        "fact_card_sha256": sha256_text(canonical_json(dict(fact_card))),
        "same_sample_minutes_sha256": sha256_text(same_sample_minutes),
        "teacher_contract": contract,
        "teacher_contract_sha256": sha256_text(canonical_json(contract)),
        "system_prompt_sha256": sha256_text(GENERATOR_SYSTEM_PROMPT),
    }


def _cache_key(
    *,
    prompt: str,
    fact_card: Mapping[str, Any],
    same_sample_minutes: str,
    teacher: DeepSeekTeacherConfig,
    generation_provenance_sha256: str,
) -> str:
    bindings = _cache_bindings(
        prompt=prompt,
        fact_card=fact_card,
        same_sample_minutes=same_sample_minutes,
        teacher=teacher,
    )
    payload = {
        "schema_version": DEEPSEEK_CACHE_SCHEMA_VERSION,
        "candidate_schema_version": CANDIDATE_SCHEMA_VERSION,
        "generation_provenance_sha256": generation_provenance_sha256,
        **bindings,
    }
    return sha256_text(canonical_json(payload))


def _cache_path(cache_dir: str | Path, cache_key: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", cache_key):
        raise CacheIntegrityError("DeepSeek cache key must be a SHA-256 digest")
    return Path(cache_dir) / cache_key[:2] / f"{cache_key}.json"


def _load_cache(
    cache_dir: str | Path,
    cache_key: str,
    *,
    generation_provenance_sha256: str,
    expected_bindings: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = _cache_path(cache_dir, cache_key)
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise CacheIntegrityError(f"DeepSeek cache entry is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CacheIntegrityError(f"invalid DeepSeek cache entry: {path}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != DEEPSEEK_CACHE_SCHEMA_VERSION
        or payload.get("cache_key") != cache_key
        or payload.get("generation_provenance_sha256")
        != generation_provenance_sha256
        or any(payload.get(key) != value for key, value in expected_bindings.items())
    ):
        raise CacheIntegrityError(f"DeepSeek cache binding mismatch: {path}")
    _validate_cache_payload_sha256(payload)
    return payload


def _recover_prior_accepted_cache(
    cache_root: str | Path,
    *,
    cache_key: str,
    generation_provenance_sha256: str,
    expected_bindings: Mapping[str, Any],
    allowed_evidence_ids: frozenset[str],
) -> dict[str, Any] | None:
    """Promote one unambiguous prior accepted response into the current cache.

    Generation-policy revisions intentionally change provenance and therefore
    the normal cache key.  The provider response itself depends on the teacher
    prompt and teacher contract, so a previously fetched response with those
    exact bindings can be reused instead of paying for the same API call again.
    """

    root = Path(cache_root)
    if not root.is_dir():
        return None
    matches: list[dict[str, Any]] = []
    for path in root.rglob("*.json"):
        if path.is_symlink():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("schema_version") == DEEPSEEK_CACHE_SCHEMA_VERSION
            and payload.get("status") == "accepted"
            and all(
                payload.get(key) == value for key, value in expected_bindings.items()
            )
        ):
            try:
                _validate_cache_payload_sha256(payload)
                _validate_cached_result(
                    payload,
                    expected_bindings=expected_bindings,
                    allowed_evidence_ids=allowed_evidence_ids,
                )
            except CacheIntegrityError:
                continue
            matches.append(payload)
    if len(matches) != 1:
        return None
    recovered = dict(matches[0])
    recovered["reused_from_cache_key"] = recovered.get("cache_key")
    recovered["cache_key"] = cache_key
    recovered["generation_provenance_sha256"] = generation_provenance_sha256
    recovered.pop("cache_payload_sha256", None)
    _store_cache(root, recovered)
    return _validate_cached_result(
        recovered,
        expected_bindings=expected_bindings,
        allowed_evidence_ids=allowed_evidence_ids,
    )


def _validate_cache_payload_sha256(payload: Mapping[str, Any]) -> None:
    declared = payload.get("cache_payload_sha256")
    if not isinstance(declared, str) or not re.fullmatch(r"[0-9a-f]{64}", declared):
        raise CacheIntegrityError("DeepSeek cache payload digest is missing")
    unsigned = dict(payload)
    unsigned.pop("cache_payload_sha256", None)
    if declared != sha256_text(canonical_json(unsigned)):
        raise CacheIntegrityError("DeepSeek cache payload digest mismatch")


def _signed_cache_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    signed = dict(payload)
    signed.pop("cache_payload_sha256", None)
    signed["cache_payload_sha256"] = sha256_text(canonical_json(signed))
    return signed


def _store_cache(cache_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    path = _cache_path(cache_dir, str(payload.get("cache_key") or ""))
    serialized = canonical_json(_signed_cache_payload(payload)) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != serialized:
            raise CacheIntegrityError(f"immutable DeepSeek cache collision: {path}")
        return path
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o400)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _validate_rejected_attempt_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CacheIntegrityError("rejected DeepSeek attempt must be an object")
    attempt = dict(value)
    expected = {
        "schema_version",
        "attempt",
        "status",
        "error_code",
        "finish_reason",
        "provider_raw",
        "provider_raw_sha256",
        "reasoning_content_sha256",
        "content_sha256",
        "teacher_provenance",
    }
    if set(attempt) != expected:
        raise CacheIntegrityError("rejected DeepSeek attempt schema is invalid")
    number = attempt["attempt"]
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise CacheIntegrityError("rejected DeepSeek attempt number is invalid")
    if (
        attempt["schema_version"] != DEEPSEEK_REJECTED_ATTEMPT_SCHEMA_VERSION
        or attempt["status"] != "rejected"
        or not isinstance(attempt["error_code"], str)
        or not re.fullmatch(r"[a-z][a-z0-9_]*", attempt["error_code"])
        or not isinstance(attempt["finish_reason"], str)
        or not attempt["finish_reason"]
    ):
        raise CacheIntegrityError("rejected DeepSeek attempt metadata is invalid")
    raw = attempt["provider_raw"]
    if (
        not isinstance(raw, Mapping)
        or set(raw) != {"reasoning_content", "content"}
        or not isinstance(raw.get("reasoning_content"), str)
        or not isinstance(raw.get("content"), str)
    ):
        raise CacheIntegrityError("rejected DeepSeek provider raw is invalid")
    provenance = attempt["teacher_provenance"]
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("finish_reason") != attempt["finish_reason"]
    ):
        raise CacheIntegrityError("rejected DeepSeek provenance is invalid")
    expected_hashes = {
        "provider_raw_sha256": sha256_text(canonical_json(dict(raw))),
        "reasoning_content_sha256": sha256_text(raw["reasoning_content"]),
        "content_sha256": sha256_text(raw["content"]),
    }
    for field, expected_hash in expected_hashes.items():
        if attempt[field] != expected_hash:
            raise CacheIntegrityError(f"rejected DeepSeek {field} mismatch")
    return attempt


def _validate_rejected_attempts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise CacheIntegrityError("rejected DeepSeek attempts must be non-empty")
    attempts = [_validate_rejected_attempt_payload(item) for item in value]
    numbers = [int(item["attempt"]) for item in attempts]
    if numbers != sorted(set(numbers)):
        raise CacheIntegrityError("rejected DeepSeek attempt order is invalid")
    return attempts


def _validate_rejected_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if result.get("status") != "rejected":
        raise CacheIntegrityError("cached DeepSeek rejection status is invalid")
    prohibited = {"selected_candidate", "selected_response", "teacher_output"}
    if prohibited.intersection(result):
        raise CacheIntegrityError("rejected DeepSeek cache contains a candidate")
    attempts = _validate_rejected_attempts(result.get("attempts"))
    expected_codes = list(
        dict.fromkeys(str(item["error_code"]) for item in attempts)
    )
    if result.get("rejection_error_codes") != expected_codes:
        raise CacheIntegrityError("rejected DeepSeek error codes mismatch")
    result["cache_hit"] = True
    return result


def _validate_cached_result(
    payload: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, Any],
    allowed_evidence_ids: frozenset[str],
) -> dict[str, Any]:
    result = dict(payload)
    candidate = result.get("selected_candidate")
    response = result.get("selected_response")
    try:
        if result.get("status") != "accepted":
            raise ModelOutputContractError("cached result is not accepted")
        if any(result.get(key) != value for key, value in expected_bindings.items()):
            raise ModelOutputContractError("cached result binding mismatch")
        if not isinstance(candidate, Mapping) or not isinstance(response, str):
            raise ModelOutputContractError("cached candidate/response is invalid")
        if candidate_to_sft_response(candidate) != response:
            raise ModelOutputContractError("cached candidate response mismatch")
        teacher_output = result.get("teacher_output")
        expected_teacher_output = {
            "analysis": str(candidate["reasoning"]).strip(),
            "answer": str(candidate["final_analysis"]).strip(),
            "evidence_ids": list(
                _validate_evidence_ids(
                    candidate["evidence_ids"],
                    allowed_evidence_ids=allowed_evidence_ids,
                )
            ),
        }
        if (
            not isinstance(teacher_output, Mapping)
            or set(teacher_output) != set(expected_teacher_output)
            or dict(teacher_output) != expected_teacher_output
        ):
            raise ModelOutputContractError("cached teacher output mismatch")
        provenance = result.get("teacher_provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("finish_reason") != "stop"
        ):
            raise ModelOutputContractError("cached finish_reason is not stop")
        provider_raw = result.get("provider_raw")
        if (
            not isinstance(provider_raw, Mapping)
            or set(provider_raw) != {"reasoning_content", "content"}
            or not isinstance(provider_raw.get("reasoning_content"), str)
            or not isinstance(provider_raw.get("content"), str)
        ):
            raise ModelOutputContractError("cached provider raw payload is invalid")
        if (
            _validate_analysis(provider_raw["reasoning_content"])
            != expected_teacher_output["analysis"]
        ):
            raise ModelOutputContractError("cached raw reasoning mismatch")
        raw_answer, raw_ids = _parse_content(
            provider_raw["content"],
            allowed_evidence_ids=allowed_evidence_ids,
        )
        if (
            raw_answer != expected_teacher_output["answer"]
            or list(raw_ids) != expected_teacher_output["evidence_ids"]
        ):
            raise ModelOutputContractError("cached raw content mismatch")
        attempts = result.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise ModelOutputContractError("cached attempts are invalid")
        accepted_attempt = attempts[-1]
        accepted_number = (
            accepted_attempt.get("attempt")
            if isinstance(accepted_attempt, Mapping)
            else None
        )
        if (
            not isinstance(accepted_attempt, Mapping)
            or accepted_attempt.get("status") != "accepted"
            or isinstance(accepted_number, bool)
            or not isinstance(accepted_number, int)
            or accepted_number < 1
            or accepted_attempt.get("teacher_provenance") != dict(provenance)
            or accepted_attempt.get("teacher_output_sha256")
            != sha256_text(canonical_json(expected_teacher_output))
            or accepted_attempt.get("provider_raw_sha256")
            != sha256_text(canonical_json(dict(provider_raw)))
        ):
            raise ModelOutputContractError("cached accepted attempt is invalid")
        rejected_attempts = attempts[:-1]
        if rejected_attempts:
            validated_rejections = _validate_rejected_attempts(rejected_attempts)
            if int(validated_rejections[-1]["attempt"]) >= accepted_number:
                raise ModelOutputContractError(
                    "cached rejected attempt sequence is invalid"
                )
    except (KeyError, ModelOutputContractError, TypeError, ValueError) as exc:
        raise CacheIntegrityError(
            "cached DeepSeek result no longer passes strict gates"
        ) from exc
    result["cache_hit"] = True
    return result


def run_deepseek_chk1_generation(
    *,
    prompt: str,
    repo_root: str | Path,
    fact_card: Mapping[str, Any] | None = None,
    same_sample_minutes: str = "",
    cache_dir: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
    teacher_backend: DeepSeekTeacherBackend | None = None,
    generation_provenance_sha256: str | None = None,
    mock: bool = False,
) -> dict[str, Any]:
    """Request and retain one chk1 analysis/answer target from DeepSeek."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be non-empty text")
    if not isinstance(fact_card, Mapping):
        raise ValueError("fact_card must be provided")
    if not isinstance(same_sample_minutes, str):
        raise ValueError("same_sample_minutes must be text")
    Path(repo_root).resolve()
    frozen_card = json.loads(canonical_json(dict(fact_card)))
    allowed_evidence_ids = _fact_card_evidence_ids(frozen_card)
    teacher = build_teacher_config(environment=environment)
    cache_bindings = _cache_bindings(
        prompt=prompt,
        fact_card=frozen_card,
        same_sample_minutes=same_sample_minutes,
        teacher=teacher,
    )
    if generation_provenance_sha256 is None:
        provenance_payload = {
            "teacher": teacher.contract(),
        }
        provenance_sha256 = sha256_text(canonical_json(provenance_payload))
    else:
        provenance_sha256 = validate_sha256(
            generation_provenance_sha256,
            label="generation_provenance_sha256",
        )
    cache_key = _cache_key(
        prompt=prompt,
        fact_card=frozen_card,
        same_sample_minutes=same_sample_minutes,
        teacher=teacher,
        generation_provenance_sha256=provenance_sha256,
    )
    if cache_dir is not None:
        cached = _load_cache(
            cache_dir,
            cache_key,
            generation_provenance_sha256=provenance_sha256,
            expected_bindings=cache_bindings,
        )
        if cached is not None:
            if cached.get("status") == "rejected":
                rejected = _validate_rejected_result(cached)
                raise TeacherResponseRejectedError(
                    "cached DeepSeek teacher target was rejected",
                    payload=rejected,
                )
            return _validate_cached_result(
                cached,
                expected_bindings=cache_bindings,
                allowed_evidence_ids=allowed_evidence_ids,
            )
        recovered = _recover_prior_accepted_cache(
            cache_dir,
            cache_key=cache_key,
            generation_provenance_sha256=provenance_sha256,
            expected_bindings=cache_bindings,
            allowed_evidence_ids=allowed_evidence_ids,
        )
        if recovered is not None:
            return recovered

    if teacher_backend is None:
        teacher_backend = OpenAIDeepSeekBackend()
    if not mock:
        env = os.environ if environment is None else environment
        if not str(env.get(teacher.api_key_env) or "").strip():
            raise ModelOutputContractError(
                f"missing DeepSeek teacher credential: {teacher.api_key_env}"
            )

    try:
        teacher_response = teacher_backend.generate(
            config=teacher,
            system_prompt=GENERATOR_SYSTEM_PROMPT,
            user_prompt=prompt,
            environment=environment,
            allowed_evidence_ids=allowed_evidence_ids,
        )
    except TeacherResponseRejectedError as exc:
        rejected_attempts = _validate_rejected_attempts(
            exc.payload.get("attempts")
        )
        expected_codes = list(
            dict.fromkeys(str(item["error_code"]) for item in rejected_attempts)
        )
        if exc.payload.get("rejection_error_codes") != expected_codes:
            raise CacheIntegrityError(
                "DeepSeek rejection payload error codes mismatch"
            ) from exc
        rejected_result = {
            "schema_version": DEEPSEEK_CACHE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "generation_provenance_sha256": provenance_sha256,
            **cache_bindings,
            "attempts": rejected_attempts,
            "rejection_error_codes": expected_codes,
            "status": "rejected",
            "cache_hit": False,
        }
        cache_payload = dict(rejected_result)
        cache_payload.pop("cache_hit")
        if cache_dir is not None:
            _store_cache(cache_dir, cache_payload)
        raise TeacherResponseRejectedError(
            "DeepSeek teacher target failed the response contract",
            payload=rejected_result,
        ) from exc
    teacher_response = _validated_response(
        teacher_response, allowed_evidence_ids=allowed_evidence_ids
    )
    candidate = teacher_response.candidate()
    rejected_attempts = [dict(item) for item in teacher_response.rejected_attempts]
    if rejected_attempts:
        _validate_rejected_attempts(rejected_attempts)
    accepted_attempt = teacher_response.accepted_attempt
    if (
        isinstance(accepted_attempt, bool)
        or not isinstance(accepted_attempt, int)
        or accepted_attempt < 1
        or (
            rejected_attempts
            and accepted_attempt <= int(rejected_attempts[-1]["attempt"])
        )
    ):
        raise ModelOutputContractError(
            "DeepSeek teacher accepted attempt metadata is invalid"
        )
    provider_raw = teacher_response.provider_raw()
    attempts = rejected_attempts + [
        {
            "attempt": accepted_attempt,
            "status": "accepted",
            "teacher_provenance": teacher_response.provenance(),
            "teacher_output_sha256": sha256_text(
                canonical_json(teacher_response.teacher_output())
            ),
            "provider_raw_sha256": sha256_text(canonical_json(provider_raw)),
        }
    ]
    result = {
        "schema_version": DEEPSEEK_CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "generation_provenance_sha256": provenance_sha256,
        **cache_bindings,
        "teacher_output": teacher_response.teacher_output(),
        "provider_raw": provider_raw,
        "teacher_provenance": teacher_response.provenance(),
        "attempts": attempts,
        "selected_from": "initial",
        "selected_candidate": candidate,
        "selected_response": candidate_to_sft_response(candidate),
        "rejection_error_codes": [],
        "status": "accepted",
        "cache_hit": False,
    }
    cache_payload = dict(result)
    cache_payload.pop("cache_hit")
    if cache_dir is not None:
        _store_cache(cache_dir, cache_payload)
    return result


__all__ = [
    "DEEPSEEK_API_KEY_ENV",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_CACHE_SCHEMA_VERSION",
    "DEEPSEEK_TEACHER_MODEL",
    "DeepSeekTeacherBackend",
    "DeepSeekTeacherConfig",
    "DeepSeekTeacherResponse",
    "MockDeepSeekTeacherBackend",
    "OpenAIDeepSeekBackend",
    "TeacherResponseRejectedError",
    "build_teacher_config",
    "candidate_to_sft_response",
    "run_deepseek_chk1_generation",
    "teacher_model_provenance",
]
