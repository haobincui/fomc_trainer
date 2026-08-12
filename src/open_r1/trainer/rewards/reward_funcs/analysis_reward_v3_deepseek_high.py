"""DeepSeek Responses transport for the grounded-analysis v3 reward.

This module is intentionally additive.  The local deterministic components and
reward formula are shared with :mod:`analysis_reward_v3`; only the provider
transport, four-candidate wire contract, and privacy-safe cache are versioned.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from open_r1.structured_response import parse_structured_response
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v2 import (
    DEFAULT_JUDGE_TOKENIZER_PATH,
    JudgeInfrastructureError,
    _canonical_tokenizer_directory,
    _completion_text,
    _complete_candidate_response,
    _concision_score,
    _meeting_date_metadata,
    _opaque_evidence_literals,
    _save_records,
    get_judge_tokenizer,
    validate_judge_evidence,
)
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v3 import (
    _JUDGE_SYSTEM_PROMPT_V3,
    _RUBRIC_KEYS,
    _candidate_numbers,
    _judge_schema_v3,
    _judge_score,
    _penalty_breakdown,
    _validate_evaluation_v3,
    _validated_violations,
    _violation_audit,
    numeric_grounding_v3,
    reasoning_efficiency,
)


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_REASONING_EFFORT = "high"
DEEPSEEK_MAX_OUTPUT_TOKENS = 16_384
DEEPSEEK_GROUP_SIZE = 4
DEEPSEEK_TIMEOUT_SECONDS = 420.0
DEEPSEEK_MAX_ATTEMPTS = 2
DEEPSEEK_REQUEST_MAX_BYTES = 512 * 1024

_WIRE_SCHEMA_VERSION = "grounded-analysis-v3-deepseek-high-request-v1"
_CACHE_SCHEMA_VERSION = "grounded-analysis-v3-deepseek-high-cache-v1"
_CONTRACT_SCHEMA_VERSION = "grounded-analysis-v3-deepseek-high-contract-v1"
_CANDIDATE_IDS = tuple(f"candidate_{index}" for index in range(DEEPSEEK_GROUP_SIZE))
_CACHE_DIRECTORY = "deepseek_reward_cache"
_CONTRACT_FILENAME = "deepseek_reward_contract.json"

_CLIENT_LOCK = threading.Lock()
_CLIENTS: dict[tuple[str, float, str], Any] = {}
_FILE_LOCK = threading.Lock()

_MONETARY_SCALE_FACTORS = {
    "thousand": 1_000.0,
    "million": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "trillion": 1_000_000_000_000.0,
}
_NUMBER_LITERAL = r"[-+]?\d[\d,]*(?:\.\d+)?"
_MONETARY_SCALE_RE = re.compile(
    r"\b(thousands?|millions?|billions?|trillions?)\b",
    flags=re.IGNORECASE,
)
_DOLLAR_UNIT_RE = re.compile(
    r"(?:\bUSD\b|\bdollars?\b|\bU\.?\s*S\.?\s+dollars?\b|\$)",
    flags=re.IGNORECASE,
)
_NUMERIC_SCALAR_RE = re.compile(
    rf"\s*(?:(?:USD|U\.?S\.?\$|\$)\s*)?(?P<number>{_NUMBER_LITERAL})\s*",
    flags=re.IGNORECASE,
)
_COMBINED_MONETARY_VALUE_RE = re.compile(
    rf"\s*(?P<prefix>(?:USD|U\.?S\.?\$|\$)\s*)?"
    rf"(?P<number>{_NUMBER_LITERAL})\s*(?P<units>.+?)\s*",
    flags=re.IGNORECASE,
)
_CANDIDATE_MONETARY_SCALE_RE = re.compile(
    rf"(?<![A-Za-z0-9_-])(?P<number>{_NUMBER_LITERAL})\s*"
    r"(?P<scale>thousands?|millions?|billions?|trillions?)\b"
    r"(?!\s*(?:%|percent(?:age)?(?:\s+points?)?|basis\s+points?|bps?\b))",
    flags=re.IGNORECASE,
)


_DEEPSEEK_BATCH_SYSTEM_PROMPT = f"""{_JUDGE_SYSTEM_PROMPT_V3}

The input contains one shared evidence object and exactly four candidates named
candidate_0 through candidate_3. Evaluate every candidate independently against
the shared evidence. Never compare candidates with each other, transfer a claim
from one candidate to another, or use one candidate as evidence for another.
Return exactly the four named evaluation objects required by the JSON schema."""


class _RetryableProviderResponseError(RuntimeError):
    """Provider returned a transient or schema-invalid response."""


class _PermanentProviderError(RuntimeError):
    """Provider or response provenance failed without a safe retry path."""


class _DuplicateJsonKeyError(ValueError):
    """Structured provider output contains an ambiguous duplicate key."""


def _finite_numeric_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    match = _NUMERIC_SCALAR_RE.fullmatch(str(value or ""))
    if match is None:
        return None
    try:
        result = float(match.group("number").replace(",", ""))
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _monetary_unit_factor(value: Any) -> float | None:
    """Return a dollar multiplier only for an explicitly dollar-denominated unit."""

    text = str(value or "")
    if _DOLLAR_UNIT_RE.search(text) is None:
        return None
    scales = {
        match.group(1).casefold().removesuffix("s")
        for match in _MONETARY_SCALE_RE.finditer(text)
    }
    if len(scales) > 1:
        return None
    return _MONETARY_SCALE_FACTORS[next(iter(scales))] if scales else 1.0


def _canonical_monetary_amount(value: Any, units: Any = None) -> float | None:
    factor = _monetary_unit_factor(units)
    number = _finite_numeric_scalar(value)
    if factor is None or number is None:
        if not isinstance(value, str):
            return None
        combined = _COMBINED_MONETARY_VALUE_RE.fullmatch(value)
        if combined is None:
            return None
        combined_units = f"{combined.group('prefix') or ''}{combined.group('units')}"
        factor = _monetary_unit_factor(combined_units)
        number = _finite_numeric_scalar(combined.group("number"))
    if factor is None or number is None:
        return None
    canonical = number * factor
    return canonical if math.isfinite(canonical) else None


def _structured_monetary_support(evidence: str) -> set[float]:
    """Extract only values explicitly paired with dollar-denominated units."""

    try:
        payload = json.loads(str(evidence))
    except (json.JSONDecodeError, TypeError):
        return set()

    supported: set[float] = set()

    def visit(value: Any, inherited_units: Any = None) -> None:
        if isinstance(value, Mapping):
            fields = {str(key).strip().casefold(): child for key, child in value.items()}
            local_units = fields.get("units", fields.get("unit", inherited_units))
            if "value" in fields:
                amount = _canonical_monetary_amount(fields["value"], local_units)
                if amount is not None:
                    supported.add(amount)
            for child in value.values():
                if isinstance(child, (Mapping, list)):
                    visit(child, local_units)
            return
        if isinstance(value, list):
            for child in value:
                visit(child, inherited_units)

    visit(payload)
    return supported


def _candidate_monetary_claims(
    candidate: str, evidence: str
) -> list[tuple[float, float]]:
    """Return ``(displayed value, canonical dollars)`` for explicit scale claims."""

    candidate_text = str(candidate or "")
    for literal in sorted(_opaque_evidence_literals(evidence), key=len, reverse=True):
        candidate_text = candidate_text.replace(literal, " ")

    claims: list[tuple[float, float]] = []
    for match in _CANDIDATE_MONETARY_SCALE_RE.finditer(candidate_text):
        displayed = _finite_numeric_scalar(match.group("number"))
        scale = match.group("scale").casefold().removesuffix("s")
        if displayed is None:
            continue
        canonical = displayed * _MONETARY_SCALE_FACTORS[scale]
        if math.isfinite(canonical):
            claims.append((displayed, canonical))
    return claims


def _monetary_values_close(candidate: float, source: float) -> bool:
    return abs(candidate - source) <= max(0.01, abs(source) * 0.001)


def numeric_grounding_v3_deepseek_high(
    candidate: str, evidence: str
) -> tuple[float, list[float]]:
    """Extend v3 numeric grounding for explicit monetary scale conversions.

    All numbers without a scale word retain the exact v3 result. Scaled claims
    are compared in canonical dollars, which both admits equivalent unit changes
    and prevents an equal displayed number with the wrong scale from passing.
    """

    baseline_score, baseline_unsupported = numeric_grounding_v3(candidate, evidence)
    claims = _candidate_monetary_claims(candidate, evidence)
    if not claims:
        return baseline_score, baseline_unsupported
    sources = _structured_monetary_support(evidence)
    if not sources:
        return baseline_score, baseline_unsupported

    candidate_values = _candidate_numbers(candidate, evidence)
    available = Counter(candidate_values)
    claim_results: dict[float, list[bool]] = {}
    for displayed, canonical in claims:
        results = claim_results.setdefault(displayed, [])
        if len(results) >= available[displayed]:
            continue
        results.append(
            any(_monetary_values_close(canonical, source) for source in sources)
        )

    unsupported_counts = Counter(baseline_unsupported)
    for displayed, results in claim_results.items():
        # v3 gives every occurrence of the same raw value the same verdict.
        # Replace that verdict only for occurrences carrying an explicit scale.
        baseline_scaled_count = min(unsupported_counts[displayed], len(results))
        unsupported_counts[displayed] -= baseline_scaled_count
        unsupported_counts[displayed] += sum(not supported for supported in results)

    unsupported: list[float] = []
    for value in candidate_values:
        if unsupported_counts[value] > 0:
            unsupported.append(value)
            unsupported_counts[value] -= 1
    score = 1.0 - len(unsupported) / len(candidate_values)
    return score, unsupported


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _batch_json_schema() -> dict[str, Any]:
    evaluation_schema = _judge_schema_v3()["schema"]
    properties = {
        candidate_id: copy.deepcopy(evaluation_schema)
        for candidate_id in _CANDIDATE_IDS
    }
    return {
        "type": "json_schema",
        "name": "fomc_analysis_rubric_v3_deepseek_high_batch4",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": list(_CANDIDATE_IDS),
            "additionalProperties": False,
        },
    }


def build_deepseek_batch_request_v3(
    *,
    evidence: str,
    candidates: Sequence[str],
    model: str = DEEPSEEK_MODEL,
    max_output_tokens: int = DEEPSEEK_MAX_OUTPUT_TOKENS,
) -> dict[str, Any]:
    """Build the locked four-candidate DeepSeek Responses request."""

    if model != DEEPSEEK_MODEL:
        raise ValueError(f"DeepSeek reward model must remain {DEEPSEEK_MODEL!r}")
    if max_output_tokens != DEEPSEEK_MAX_OUTPUT_TOKENS:
        raise ValueError(
            "DeepSeek reward max_output_tokens must remain fixed at "
            f"{DEEPSEEK_MAX_OUTPUT_TOKENS}"
        )
    if len(candidates) != DEEPSEEK_GROUP_SIZE:
        raise ValueError(
            f"DeepSeek reward requires exactly {DEEPSEEK_GROUP_SIZE} candidates"
        )
    if not all(isinstance(candidate, str) for candidate in candidates):
        raise TypeError("DeepSeek reward candidates must all be strings")

    evidence_text = validate_judge_evidence(evidence)
    payload = _canonical_json(
        {
            "schema_version": _WIRE_SCHEMA_VERSION,
            "evidence": evidence_text,
            "candidates": {
                candidate_id: candidate
                for candidate_id, candidate in zip(
                    _CANDIDATE_IDS, candidates, strict=True
                )
            },
        }
    )
    if len(payload.encode("utf-8")) > DEEPSEEK_REQUEST_MAX_BYTES:
        raise ValueError(
            "DeepSeek four-candidate request exceeds the fixed 512-KiB safety limit"
        )
    return {
        "model": model,
        "instructions": _DEEPSEEK_BATCH_SYSTEM_PROMPT,
        "input": payload,
        "reasoning": {"effort": DEEPSEEK_REASONING_EFFORT},
        "max_output_tokens": max_output_tokens,
        "text": {"format": _batch_json_schema()},
    }


def _request_contract(
    *,
    url: str,
    model: str,
    max_output_tokens: int,
    timeout: float,
    max_attempts: int,
    backoff_seconds: float,
) -> dict[str, Any]:
    schema = _batch_json_schema()
    immutable = {
        "schema_version": _CONTRACT_SCHEMA_VERSION,
        "provider": "deepseek",
        "api": "responses",
        "base_url": url,
        "model": model,
        "reasoning_effort": DEEPSEEK_REASONING_EFFORT,
        "max_output_tokens": max_output_tokens,
        "group_size": DEEPSEEK_GROUP_SIZE,
        "candidate_ids": list(_CANDIDATE_IDS),
        "request_schema_sha256": _sha256_text(_canonical_json(schema)),
        "instructions_sha256": _sha256_text(_DEEPSEEK_BATCH_SYSTEM_PROMPT),
        "request_max_bytes": DEEPSEEK_REQUEST_MAX_BYTES,
        "timeout_seconds": timeout,
        "max_attempts": max_attempts,
        "backoff_seconds": backoff_seconds,
    }
    return {**immutable, "request_contract_sha256": _sha256_text(_canonical_json(immutable))}


def _safe_package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _write_json_exclusive_or_verify(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise JudgeInfrastructureError(f"refusing symlinked reward directory: {path.parent}")
    with _FILE_LOCK:
        if path.exists():
            if path.is_symlink() or path.read_text(encoding="utf-8") != serialized:
                raise JudgeInfrastructureError(f"DeepSeek reward receipt mismatch: {path}")
            return
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = -1
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary.exists():
                temporary.unlink()


def _ensure_contract_receipt(save_path: str | None, contract: Mapping[str, Any]) -> None:
    if not save_path:
        return
    receipt = {
        **dict(contract),
        "openai_version": _safe_package_version("openai"),
        "httpx_version": _safe_package_version("httpx"),
    }
    _write_json_exclusive_or_verify(Path(save_path).parent / _CONTRACT_FILENAME, receipt)


def _cache_binding(
    *, contract: Mapping[str, Any], evidence: str, candidates: Sequence[str]
) -> dict[str, Any]:
    binding = {
        "request_contract_sha256": contract["request_contract_sha256"],
        "evidence_sha256": _sha256_text(evidence),
        "candidate_sha256": [_sha256_text(candidate) for candidate in candidates],
    }
    return {**binding, "group_sha256": _sha256_text(_canonical_json(binding))}


def _cache_path(save_path: str | None, binding: Mapping[str, Any]) -> Path | None:
    if not save_path:
        return None
    return (
        Path(save_path).parent
        / _CACHE_DIRECTORY
        / f"{binding['group_sha256']}.json"
    )


def _validate_sanitized_evaluations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != DEEPSEEK_GROUP_SIZE:
        raise JudgeInfrastructureError("DeepSeek reward cache evaluation count mismatch")
    checked: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise JudgeInfrastructureError("DeepSeek reward cache evaluation is invalid")
        if set(item) != {
            "candidate_id",
            "rubric",
            "validated_violations",
            "invalid_violations",
            "penalties",
        }:
            raise JudgeInfrastructureError("DeepSeek reward cache evaluation keys mismatch")
        rubric = item.get("rubric")
        if not isinstance(rubric, dict) or set(rubric) != set(_RUBRIC_KEYS):
            raise JudgeInfrastructureError("DeepSeek reward cache rubric mismatch")
        for key in _RUBRIC_KEYS:
            score = rubric[key]
            if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4:
                raise JudgeInfrastructureError("DeepSeek reward cache score is invalid")
        penalties = item.get("penalties")
        if not isinstance(penalties, dict):
            raise JudgeInfrastructureError("DeepSeek reward cache penalties are invalid")
        required_penalties = {
            "major_answer_error",
            "minor_answer_error",
            "major_think_error",
            "minor_think_error",
            "unbounded",
            "applied",
            "target_leakage",
        }
        if set(penalties) != required_penalties:
            raise JudgeInfrastructureError("DeepSeek reward cache penalty keys mismatch")
        if not isinstance(penalties["target_leakage"], bool):
            raise JudgeInfrastructureError("DeepSeek reward cache leakage flag is invalid")
        for key in required_penalties - {"target_leakage"}:
            number = penalties[key]
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise JudgeInfrastructureError("DeepSeek reward cache penalty is invalid")
            if not math.isfinite(float(number)) or float(number) < 0:
                raise JudgeInfrastructureError("DeepSeek reward cache penalty is non-finite")
        for list_key in ("validated_violations", "invalid_violations"):
            audits = item.get(list_key)
            if not isinstance(audits, list):
                raise JudgeInfrastructureError("DeepSeek reward cache violation audit is invalid")
            expected_keys = {
                "section",
                "kind",
                "severity",
                "candidate_quote_sha256",
                "candidate_quote_chars",
            }
            if list_key == "invalid_violations":
                expected_keys.add("invalid_reason")
            for audit in audits:
                if not isinstance(audit, dict) or set(audit) != expected_keys:
                    raise JudgeInfrastructureError(
                        "DeepSeek reward cache violation audit keys mismatch"
                    )
                if audit["section"] not in {"think", "answer"}:
                    raise JudgeInfrastructureError(
                        "DeepSeek reward cache violation section is invalid"
                    )
                if audit["kind"] not in {
                    "factual",
                    "numerical",
                    "causal",
                    "format",
                    "omission",
                    "style",
                    "target_leakage",
                }:
                    raise JudgeInfrastructureError(
                        "DeepSeek reward cache violation kind is invalid"
                    )
                if audit["severity"] not in {"minor", "major"}:
                    raise JudgeInfrastructureError(
                        "DeepSeek reward cache violation severity is invalid"
                    )
                quote_digest = audit["candidate_quote_sha256"]
                if (
                    not isinstance(quote_digest, str)
                    or len(quote_digest) != 64
                    or any(character not in "0123456789abcdef" for character in quote_digest)
                ):
                    raise JudgeInfrastructureError(
                        "DeepSeek reward cache quote hash is invalid"
                    )
                quote_chars = audit["candidate_quote_chars"]
                if (
                    isinstance(quote_chars, bool)
                    or not isinstance(quote_chars, int)
                    or quote_chars < 0
                ):
                    raise JudgeInfrastructureError(
                        "DeepSeek reward cache quote length is invalid"
                    )
                if list_key == "invalid_violations" and not isinstance(
                    audit["invalid_reason"], str
                ):
                    raise JudgeInfrastructureError(
                        "DeepSeek reward cache invalid-violation reason is invalid"
                    )
        recomputed_penalties = _penalty_breakdown(
            [
                {
                    "section": str(audit["section"]),
                    "kind": str(audit["kind"]),
                    "severity": str(audit["severity"]),
                }
                for audit in item["validated_violations"]
            ]
        )
        if penalties != recomputed_penalties:
            raise JudgeInfrastructureError(
                "DeepSeek reward cache penalty does not match violation audits"
            )
        checked.append(dict(item))
        if checked[-1].get("candidate_id") != _CANDIDATE_IDS[index]:
            raise JudgeInfrastructureError("DeepSeek reward cache candidate binding mismatch")
    return checked


def _validate_cached_provider(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "attempts",
        "attempt_metrics",
        "latency_seconds",
        "returned_model",
        "response_id_sha256",
        "visible_response_sha256",
        "hidden_reasoning_sha256",
        "usage",
    }:
        raise JudgeInfrastructureError("DeepSeek reward cache provider keys mismatch")
    attempts = value["attempts"]
    if (
        isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or not 1 <= attempts <= DEEPSEEK_MAX_ATTEMPTS
    ):
        raise JudgeInfrastructureError("DeepSeek reward cache attempt count is invalid")
    metrics = value["attempt_metrics"]
    if not isinstance(metrics, list) or len(metrics) != attempts:
        raise JudgeInfrastructureError("DeepSeek reward cache attempt metrics mismatch")
    for index, metric in enumerate(metrics, start=1):
        if not isinstance(metric, dict):
            raise JudgeInfrastructureError("DeepSeek reward cache attempt metric is invalid")
        status = metric.get("status")
        expected_metric_keys = {
            "attempt",
            "status",
            "latency_seconds",
            "http_status",
        }
        if status != "completed":
            expected_metric_keys.add("error_class")
        if set(metric) != expected_metric_keys or status not in {
            "completed",
            "retryable_error",
        }:
            raise JudgeInfrastructureError("DeepSeek reward cache attempt metric is invalid")
        if metric["attempt"] != index:
            raise JudgeInfrastructureError("DeepSeek reward cache attempt order is invalid")
        metric_latency = metric["latency_seconds"]
        if (
            isinstance(metric_latency, bool)
            or not isinstance(metric_latency, (int, float))
            or not math.isfinite(float(metric_latency))
            or float(metric_latency) < 0
        ):
            raise JudgeInfrastructureError("DeepSeek reward cache attempt latency is invalid")
        http_status = metric["http_status"]
        if http_status is not None and (
            isinstance(http_status, bool)
            or not isinstance(http_status, int)
            or not 100 <= http_status <= 599
        ):
            raise JudgeInfrastructureError("DeepSeek reward cache HTTP status is invalid")
        if status == "completed" and http_status != 200:
            raise JudgeInfrastructureError("DeepSeek reward cache completed status is invalid")
        if status == "retryable_error" and not isinstance(metric["error_class"], str):
            raise JudgeInfrastructureError("DeepSeek reward cache error class is invalid")
    if metrics[-1].get("status") != "completed":
        raise JudgeInfrastructureError("DeepSeek reward cache has no completed attempt")
    if value["returned_model"] != DEEPSEEK_MODEL:
        raise JudgeInfrastructureError("DeepSeek reward cache model mismatch")
    for key in ("visible_response_sha256", "hidden_reasoning_sha256"):
        digest = value[key]
        if not isinstance(digest, str) or len(digest) != 64:
            raise JudgeInfrastructureError("DeepSeek reward cache response hash is invalid")
    response_id = value["response_id_sha256"]
    if response_id is not None and (
        not isinstance(response_id, str) or len(response_id) != 64
    ):
        raise JudgeInfrastructureError("DeepSeek reward cache response ID hash is invalid")
    latency = value["latency_seconds"]
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or float(latency) < 0
    ):
        raise JudgeInfrastructureError("DeepSeek reward cache latency is invalid")
    usage = value["usage"]
    expected_usage = {
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
    }
    if not isinstance(usage, dict) or set(usage) != expected_usage:
        raise JudgeInfrastructureError("DeepSeek reward cache usage keys mismatch")
    for key, count in usage.items():
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count < 0
        ):
            raise JudgeInfrastructureError(f"DeepSeek reward cache {key} is invalid")
    if not isinstance(usage["reasoning_tokens"], int) or usage["reasoning_tokens"] <= 0:
        raise JudgeInfrastructureError("DeepSeek reward cache has no reasoning tokens")
    return dict(value)


def _load_cache(path: Path | None, binding: Mapping[str, Any]) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    if path.is_symlink():
        raise JudgeInfrastructureError(f"refusing symlinked DeepSeek reward cache: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JudgeInfrastructureError("unable to read DeepSeek reward cache") from exc
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema_version",
            "binding",
            "provider",
            "evaluations",
            "payload_sha256",
        }
        or payload.get("schema_version") != _CACHE_SCHEMA_VERSION
    ):
        raise JudgeInfrastructureError("DeepSeek reward cache schema mismatch")
    cache_core = {
        key: payload[key]
        for key in ("schema_version", "binding", "provider", "evaluations")
    }
    if payload.get("payload_sha256") != _sha256_text(_canonical_json(cache_core)):
        raise JudgeInfrastructureError("DeepSeek reward cache payload hash mismatch")
    if payload.get("binding") != dict(binding):
        raise JudgeInfrastructureError("DeepSeek reward cache hash binding mismatch")
    payload["provider"] = _validate_cached_provider(payload.get("provider"))
    payload["evaluations"] = _validate_sanitized_evaluations(payload.get("evaluations"))
    return payload


def _write_cache(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError as exc:
        raise JudgeInfrastructureError("unable to protect DeepSeek reward cache directory") from exc
    _write_json_exclusive_or_verify(path, payload)


def _save_private_records(path: str | None, records: list[dict[str, Any]]) -> None:
    if not path:
        return
    output_path = Path(path)
    if output_path.is_symlink() or output_path.parent.is_symlink():
        raise JudgeInfrastructureError("refusing symlinked DeepSeek reward log")
    _save_records(path, records)
    try:
        output_path.chmod(0o600)
    except OSError as exc:
        raise JudgeInfrastructureError("unable to protect DeepSeek reward log") from exc


def _get_client(*, api_key: str, url: str, timeout: float) -> Any:
    key_digest = _sha256_text(api_key)
    cache_key = (url, timeout, key_digest)
    with _CLIENT_LOCK:
        existing = _CLIENTS.get(cache_key)
        if existing is not None:
            return existing
        try:
            import httpx
            from openai import OpenAI
        except ImportError as exc:
            raise JudgeInfrastructureError(
                "fomc_trainer must provide openai with Responses API support"
            ) from exc
        if not hasattr(OpenAI, "responses"):
            raise JudgeInfrastructureError("installed OpenAI SDK has no Responses API")
        http_timeout = httpx.Timeout(
            timeout,
            connect=min(15.0, timeout),
            read=timeout,
            write=min(30.0, timeout),
            pool=min(15.0, timeout),
        )
        client = OpenAI(
            api_key=api_key,
            base_url=url,
            timeout=http_timeout,
            max_retries=0,
        )
        _CLIENTS[cache_key] = client
        return client


def _extract_response_output(response: Any) -> tuple[str, str]:
    visible = str(_field(response, "output_text", "") or "")
    hidden_parts: list[str] = []
    visible_parts: list[str] = []
    for item in _field(response, "output", []) or []:
        item_type = _field(item, "type")
        for part in _field(item, "content", []) or []:
            part_type = _field(part, "type")
            text = _field(part, "text")
            if not isinstance(text, str) or not text:
                continue
            if item_type == "reasoning" and part_type == "reasoning_text":
                hidden_parts.append(text)
            elif item_type == "message" and part_type == "output_text":
                visible_parts.append(text)
    reconstructed = "".join(visible_parts)
    if visible and reconstructed and visible != reconstructed:
        raise _RetryableProviderResponseError(
            "Responses output_text disagrees with message output"
        )
    return visible or reconstructed, "".join(hidden_parts)


def _usage_payload(value: Any) -> dict[str, int | None]:
    def integer(source: Any, name: str) -> int | None:
        raw = _field(source, name) if source is not None else None
        return int(raw) if isinstance(raw, int) and not isinstance(raw, bool) else None

    output_details = _field(value, "output_tokens_details") if value is not None else None
    input_details = _field(value, "input_tokens_details") if value is not None else None
    return {
        "input_tokens": integer(value, "input_tokens"),
        "cached_input_tokens": integer(input_details, "cached_tokens"),
        "output_tokens": integer(value, "output_tokens"),
        "reasoning_tokens": integer(output_details, "reasoning_tokens"),
        "total_tokens": integer(value, "total_tokens"),
    }


def _parse_batch_output(visible: str) -> list[dict[str, Any]]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateJsonKeyError("DeepSeek JSON contains a duplicate key")
            result[key] = value
        return result

    try:
        payload = json.loads(
            str(visible or "").strip(),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, _DuplicateJsonKeyError) as exc:
        raise _RetryableProviderResponseError(
            "DeepSeek visible output is not one unambiguous JSON object"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != set(_CANDIDATE_IDS):
        raise _RetryableProviderResponseError(
            "DeepSeek batch response must contain exactly candidate_0..candidate_3"
        )
    evaluations: list[dict[str, Any]] = []
    for candidate_id in _CANDIDATE_IDS:
        try:
            evaluations.append(_validate_evaluation_v3(payload[candidate_id]))
        except (KeyError, TypeError, ValueError) as exc:
            raise _RetryableProviderResponseError(
                f"DeepSeek evaluation schema is invalid for {candidate_id}"
            ) from exc
    return evaluations


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, _RetryableProviderResponseError):
        return True
    if isinstance(exc, _PermanentProviderError):
        return False
    status = _status_code(exc)
    if status is not None:
        if status in {400, 401, 403, 404, 422}:
            return False
        return status in {408, 409, 429} or status >= 500
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__.casefold()
    return "timeout" in name or "connection" in name


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        value = float(headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
    return min(60.0, max(0.0, value)) if math.isfinite(value) else None


def _provider_batch(
    *,
    client: Any,
    request: Mapping[str, Any],
    model: str,
    max_attempts: int,
    backoff_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attempt_metrics: list[dict[str, Any]] = []
    last_error_class = "provider_not_called"
    group_started = time.perf_counter()
    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            response = client.responses.create(**dict(request))
            status = str(_field(response, "status", "") or "unknown")
            if status != "completed":
                raise _RetryableProviderResponseError(
                    f"DeepSeek response status was {status}"
                )
            returned_model = str(_field(response, "model", "") or "").strip()
            if not returned_model:
                raise _PermanentProviderError("DeepSeek response omitted model provenance")
            if returned_model != model:
                raise _PermanentProviderError("DeepSeek returned an unexpected model")
            visible, hidden = _extract_response_output(response)
            if not visible.strip():
                raise _RetryableProviderResponseError("DeepSeek returned no visible output")
            if not hidden.strip():
                raise _RetryableProviderResponseError("high-effort response has no reasoning")
            usage = _usage_payload(_field(response, "usage"))
            reasoning_tokens = usage.get("reasoning_tokens")
            if not isinstance(reasoning_tokens, int) or reasoning_tokens <= 0:
                raise _RetryableProviderResponseError(
                    "high-effort response reported no reasoning tokens"
                )
            evaluations = _parse_batch_output(visible)
            latency = time.perf_counter() - started
            attempt_metrics.append(
                {
                    "attempt": attempt,
                    "status": "completed",
                    "latency_seconds": latency,
                    "http_status": 200,
                }
            )
            response_id = str(_field(response, "id", "") or "")
            return evaluations, {
                "attempts": attempt,
                "attempt_metrics": attempt_metrics,
                "latency_seconds": time.perf_counter() - group_started,
                "returned_model": returned_model,
                "response_id_sha256": _sha256_text(response_id) if response_id else None,
                "visible_response_sha256": _sha256_text(visible),
                "hidden_reasoning_sha256": _sha256_text(hidden),
                "usage": usage,
            }
        except Exception as exc:  # provider SDK exceptions vary by version
            latency = time.perf_counter() - started
            status = _status_code(exc)
            retryable = _retryable_exception(exc)
            last_error_class = type(exc).__name__
            attempt_metrics.append(
                {
                    "attempt": attempt,
                    "status": "retryable_error" if retryable else "permanent_error",
                    "latency_seconds": latency,
                    "http_status": status,
                    "error_class": last_error_class,
                }
            )
            if not retryable:
                reason = (
                    "model_mismatch"
                    if isinstance(exc, _PermanentProviderError)
                    else "permanent_provider_error"
                )
                raise JudgeInfrastructureError(
                    "DeepSeek judge failed permanently: "
                    f"reason={reason}, error_class={last_error_class}, status={status}"
                ) from None
            if attempt < max_attempts:
                retry_after = _retry_after_seconds(exc)
                delay = (
                    retry_after
                    if retry_after is not None
                    else backoff_seconds * (2 ** (attempt - 1))
                )
                if delay > 0:
                    time.sleep(delay)
    raise JudgeInfrastructureError(
        "DeepSeek judge failed after "
        f"{max_attempts} attempts: error_class={last_error_class}"
    )


def _sanitize_provider_evaluations(
    evaluations: Sequence[Mapping[str, Any]], candidates: Sequence[str]
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for candidate_id, evaluation, candidate in zip(
        _CANDIDATE_IDS, evaluations, candidates, strict=True
    ):
        valid, invalid = _validated_violations(evaluation, candidate)
        sanitized.append(
            {
                "candidate_id": candidate_id,
                "rubric": {key: int(evaluation[key]) for key in _RUBRIC_KEYS},
                "validated_violations": [
                    _violation_audit(violation) for violation in valid
                ],
                "invalid_violations": [
                    _violation_audit(item["violation"], reason=item["reason"])
                    for item in invalid
                ],
                "penalties": _penalty_breakdown(valid),
            }
        )
    return sanitized


def _validate_base_url(value: str) -> str:
    normalized = str(value or "").rstrip("/")
    parsed = urlparse(normalized)
    if (
        normalized != DEEPSEEK_BASE_URL
        or parsed.scheme != "https"
        or parsed.netloc != "api.deepseek.com"
        or parsed.path
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"DeepSeek reward URL must remain {DEEPSEEK_BASE_URL}")
    return normalized


def _validate_runtime_numbers(
    *, timeout: float, max_attempts: int, backoff_seconds: float
) -> None:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("DeepSeek reward timeout must be numeric")
    if not math.isfinite(float(timeout)) or float(timeout) <= 0:
        raise ValueError("DeepSeek reward timeout must be positive and finite")
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or not 1 <= max_attempts <= DEEPSEEK_MAX_ATTEMPTS
    ):
        raise ValueError(
            f"DeepSeek reward max_retries means total attempts and must be in [1, {DEEPSEEK_MAX_ATTEMPTS}]"
        )
    if isinstance(backoff_seconds, bool) or not isinstance(backoff_seconds, (int, float)):
        raise ValueError("DeepSeek reward backoff must be numeric")
    if not math.isfinite(float(backoff_seconds)) or not 0 <= float(backoff_seconds) <= 60:
        raise ValueError("DeepSeek reward backoff must be finite and in [0, 60]")


def grounded_analysis_reward_v3_deepseek_high(
    completions: list[list[dict[str, str]]],
    provided_data: list[str],
    meeting_date: list[str] | None = None,
    save_path: str | None = None,
    url: str | None = None,
    model: str | None = None,
    tokenizer_path: str | None = None,
    max_model_len: int | None = None,
    max_completion_tokens: int | None = None,
    judge_tokenizer: Any | None = None,
    timeout: int | float | None = None,
    max_retries: int = DEEPSEEK_MAX_ATTEMPTS,
    backoff_seconds: float = 2.0,
    api_key_env: str | None = "DEEPSEEK_API_KEY",
    client: Any | None = None,
    **extra: Any,
) -> list[float]:
    """Score one four-generation GRPO group with one DeepSeek request."""

    del max_model_len  # DeepSeek has no local tokenizer for an exact context gate.
    if not completions:
        return []
    if len(completions) != DEEPSEEK_GROUP_SIZE:
        raise ValueError(
            f"DeepSeek reward requires exactly {DEEPSEEK_GROUP_SIZE} completions per call"
        )
    if len(provided_data) != DEEPSEEK_GROUP_SIZE:
        raise ValueError("provided_data must contain exactly four entries")
    if any(value != provided_data[0] for value in provided_data[1:]):
        raise ValueError("all four DeepSeek reward candidates must share exact evidence")
    prompts = extra.get("prompts")
    if prompts is not None:
        if not isinstance(prompts, list) or len(prompts) != DEEPSEEK_GROUP_SIZE:
            raise ValueError("prompts must contain exactly four entries when supplied")
        if any(value != prompts[0] for value in prompts[1:]):
            raise ValueError("all four DeepSeek reward candidates must share one prompt")

    dates = meeting_date or [""] * DEEPSEEK_GROUP_SIZE
    if len(dates) != DEEPSEEK_GROUP_SIZE:
        raise ValueError("meeting_date must contain exactly four entries")
    date_metadata = [_meeting_date_metadata(value) for value in dates]
    if any(value != date_metadata[0] for value in date_metadata[1:]):
        raise ValueError("all four DeepSeek reward candidates must share one meeting date")

    resolved_url = _validate_base_url(url or DEEPSEEK_BASE_URL)
    resolved_model = model or DEEPSEEK_MODEL
    if resolved_model != DEEPSEEK_MODEL:
        raise ValueError(f"DeepSeek reward model must remain {DEEPSEEK_MODEL}")
    resolved_max_output_tokens = (
        DEEPSEEK_MAX_OUTPUT_TOKENS
        if max_completion_tokens is None
        else max_completion_tokens
    )
    if resolved_max_output_tokens != DEEPSEEK_MAX_OUTPUT_TOKENS:
        raise ValueError(
            f"DeepSeek max output must remain {DEEPSEEK_MAX_OUTPUT_TOKENS} tokens"
        )
    resolved_timeout = DEEPSEEK_TIMEOUT_SECONDS if timeout is None else float(timeout)
    _validate_runtime_numbers(
        timeout=resolved_timeout,
        max_attempts=max_retries,
        backoff_seconds=backoff_seconds,
    )
    resolved_backoff = float(backoff_seconds)
    if client is None:
        if not save_path:
            raise ValueError(
                "live DeepSeek reward requires save_reward=true for receipts and cache"
            )
        for environment_name in ("OPENAI_LOG", "HTTPX_LOG_LEVEL"):
            if os.environ.get(environment_name, "").strip().casefold() in {
                "debug",
                "trace",
            }:
                raise ValueError(
                    f"live DeepSeek reward refuses unsafe {environment_name} logging"
                )
        if api_key_env != "DEEPSEEK_API_KEY":
            raise ValueError("live DeepSeek reward must read only DEEPSEEK_API_KEY")
        if resolved_timeout != DEEPSEEK_TIMEOUT_SECONDS:
            raise ValueError(
                f"live DeepSeek reward timeout must remain {DEEPSEEK_TIMEOUT_SECONDS:g} seconds"
            )
        if max_retries != DEEPSEEK_MAX_ATTEMPTS:
            raise ValueError(
                f"live DeepSeek reward must use {DEEPSEEK_MAX_ATTEMPTS} total attempts"
            )
        if resolved_backoff != 2.0:
            raise ValueError("live DeepSeek reward backoff must remain 2 seconds")
    evidence = validate_judge_evidence(provided_data[0])

    tokenizer = judge_tokenizer
    if tokenizer is None:
        resolved_tokenizer_path = _canonical_tokenizer_directory(
            tokenizer_path or DEFAULT_JUDGE_TOKENIZER_PATH
        )
        tokenizer = get_judge_tokenizer(resolved_tokenizer_path)

    prepared: list[dict[str, Any]] = []
    candidates: list[str] = []
    for completion in completions:
        content = _completion_text(completion)
        closing_think_count = content.count("</think>")
        first_boundary = content.find("</think>")
        answer_chars_after_boundary = (
            len(content[first_boundary + len("</think>") :].strip())
            if first_boundary >= 0
            else 0
        )
        parsed = parse_structured_response(content)
        contract_valid = bool(
            closing_think_count >= 1 and parsed.is_well_formed and parsed.answer.strip()
        )
        if contract_valid:
            reasoning = parsed.reasoning
            answer = parsed.answer
            rejection_reason = None
        else:
            reasoning = content.strip()
            answer = ""
            if closing_think_count == 0:
                rejection_reason = "missing_think_boundary"
            elif answer_chars_after_boundary == 0:
                rejection_reason = "empty_answer_after_boundary"
            else:
                rejection_reason = "malformed_response"
        candidate = _complete_candidate_response(parsed, content)
        candidates.append(candidate)
        answer_numeric, unsupported_answer = (
            numeric_grounding_v3_deepseek_high(answer, evidence)
            if contract_valid
            else (0.0, [])
        )
        think_numeric, unsupported_think = numeric_grounding_v3_deepseek_high(
            reasoning, evidence
        )
        if reasoning:
            efficiency, reasoning_tokens, repetition_rate = reasoning_efficiency(
                reasoning, tokenizer
            )
        else:
            efficiency, reasoning_tokens, repetition_rate = 0.0, 0, 0.0
        structure = float(
            contract_valid and bool(parsed.reasoning.strip()) and bool(parsed.answer.strip())
        )
        prepared.append(
            {
                "completion_chars": len(content),
                "closing_think_count": closing_think_count,
                "answer_chars_after_boundary": answer_chars_after_boundary,
                "parsed_format": parsed.format_name,
                "contract_valid": contract_valid,
                "rejection_reason": rejection_reason,
                "answer": answer,
                "reasoning": reasoning,
                "answer_numeric": answer_numeric,
                "think_numeric": think_numeric,
                "unsupported_answer": unsupported_answer,
                "unsupported_think": unsupported_think,
                "structure": structure,
                "answer_concision": _concision_score(answer),
                "reasoning_efficiency": efficiency,
                "reasoning_tokens": reasoning_tokens,
                "reasoning_repetition_rate": repetition_rate,
            }
        )

    request = build_deepseek_batch_request_v3(
        evidence=evidence,
        candidates=candidates,
        model=resolved_model,
        max_output_tokens=resolved_max_output_tokens,
    )
    contract = _request_contract(
        url=resolved_url,
        model=resolved_model,
        max_output_tokens=resolved_max_output_tokens,
        timeout=resolved_timeout,
        max_attempts=max_retries,
        backoff_seconds=resolved_backoff,
    )
    _ensure_contract_receipt(save_path, contract)
    binding = _cache_binding(contract=contract, evidence=evidence, candidates=candidates)
    cache_path = _cache_path(save_path, binding)
    cached = _load_cache(cache_path, binding)
    cache_hit = cached is not None
    if cached is None:
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if client is None:
            if not api_key:
                raise JudgeInfrastructureError(
                    f"DeepSeek API key is missing from {api_key_env or '<disabled>'}"
                )
            client = _get_client(
                api_key=api_key, url=resolved_url, timeout=resolved_timeout
            )
        evaluations, provider = _provider_batch(
            client=client,
            request=request,
            model=resolved_model,
            max_attempts=max_retries,
            backoff_seconds=resolved_backoff,
        )
        sanitized = _sanitize_provider_evaluations(evaluations, candidates)
        cache_core = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "binding": binding,
            "provider": provider,
            "evaluations": sanitized,
        }
        cached = {
            **cache_core,
            "payload_sha256": _sha256_text(_canonical_json(cache_core)),
        }
        _write_cache(cache_path, cached)
    sanitized = _validate_sanitized_evaluations(cached["evaluations"])
    provider = _validate_cached_provider(cached.get("provider"))

    rewards: list[float] = []
    records: list[dict[str, Any]] = []
    for index, (item, evaluation) in enumerate(
        zip(prepared, sanitized, strict=True)
    ):
        rubric = evaluation["rubric"]
        penalties = evaluation["penalties"]
        judge_score = _judge_score(rubric)
        base = (
            0.50 * judge_score
            + 0.25 * item["answer_numeric"]
            + 0.15 * item["structure"]
            + 0.05 * item["answer_concision"]
            + 0.05 * item["reasoning_efficiency"]
            if item["contract_valid"]
            else 0.0
        )
        empty_answer = not bool(item["answer"].strip())
        if (
            not item["contract_valid"]
            or empty_answer
            or penalties["target_leakage"]
        ):
            reward = 0.0
        else:
            reward = max(0.0, min(1.0, base - float(penalties["applied"])))
        if not math.isfinite(reward):
            raise JudgeInfrastructureError("DeepSeek reward produced a non-finite value")
        rewards.append(reward)
        records.append(
            {
                "type": "grounded_analysis_v3_deepseek_high",
                "candidate_id": _CANDIDATE_IDS[index],
                "group_sha256": binding["group_sha256"],
                "candidate_sha256": binding["candidate_sha256"][index],
                "evidence_sha256": binding["evidence_sha256"],
                "cache_hit": cache_hit,
                "meeting_date": date_metadata[index],
                "model": resolved_model,
                "judge_attempts": provider["attempts"],
                "judge_prompt_tokens": provider["usage"].get("input_tokens"),
                "judge_response_sha256": provider["visible_response_sha256"],
                "provider": {
                    "name": "deepseek",
                    "api": "responses",
                    "reasoning_effort": DEEPSEEK_REASONING_EFFORT,
                    "max_output_tokens": resolved_max_output_tokens,
                    "group_size": DEEPSEEK_GROUP_SIZE,
                    "request_contract_sha256": contract[
                        "request_contract_sha256"
                    ],
                    "cache_hit": cache_hit,
                    **provider,
                },
                "judge": dict(rubric),
                "validated_violations": evaluation["validated_violations"],
                "invalid_violations": evaluation["invalid_violations"],
                "components": {
                    "judge_score": judge_score,
                    "answer_numeric_score": item["answer_numeric"],
                    "think_numeric_score_audit": item["think_numeric"],
                    "structure_score": item["structure"],
                    "answer_concision": item["answer_concision"],
                    "reasoning_efficiency": item["reasoning_efficiency"],
                },
                "unsupported_answer_number_count": len(item["unsupported_answer"]),
                "unsupported_think_number_count_audit": len(
                    item["unsupported_think"]
                ),
                "contract": {
                    "well_formed": bool(item["structure"]),
                    "answer_nonempty": not empty_answer,
                    "reasoning_nonempty": bool(item["reasoning"].strip()),
                    "parsed_format": item["parsed_format"],
                    "completion_chars": item["completion_chars"],
                    "closing_think_count": item["closing_think_count"],
                    "answer_chars_after_boundary": item[
                        "answer_chars_after_boundary"
                    ],
                    "plain_answer_fallback": False,
                    "accepted_for_judge": item["contract_valid"],
                    "rejection_reason": item["rejection_reason"],
                },
                "reasoning_tokens": item["reasoning_tokens"],
                "reasoning_repetition_rate": item["reasoning_repetition_rate"],
                "penalties": penalties,
                "reward_before_penalty": base,
                "reward": reward,
            }
        )
    _save_private_records(save_path, records)
    return rewards


__all__ = [
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MAX_OUTPUT_TOKENS",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_REASONING_EFFORT",
    "build_deepseek_batch_request_v3",
    "grounded_analysis_reward_v3_deepseek_high",
    "numeric_grounding_v3_deepseek_high",
]
