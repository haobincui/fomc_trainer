"""Deterministic candidate verification for the chk1 local teacher.

Semantic groundedness still requires the independent local chk0 critic, but a
candidate cannot reach that critic until this module proves its structure,
numeric evidence, cutoff dates, special-token hygiene, and Minutes-copy guard.
All checks fail closed and return stable error codes suitable for one repair.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping, Sequence

from .contracts import canonical_json, sha256_text
from .critic import critic_accepts, validate_critic


_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?P<sign>[+-])?(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?P<suffix>\s*(?:%|percent(?:age)?(?:\s+points?)?|basis\s+points?|bps))?",
    flags=re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_MODEL_TOKEN_RE = re.compile(
    r"<\|(?:im_start|im_end|endoftext|assistant|user|system)[^>]*\>|"
    r"</?think>|```",
    flags=re.IGNORECASE,
)
_POLICY_OR_EVENT_TERMS = (
    "asset purchase",
    "balance sheet runoff",
    "forward guidance",
    "fiscal stimulus",
    "pandemic",
    "recession",
    "trade war",
    "bank failure",
    "policy rate was raised",
    "policy rate was cut",
    "committee decided",
    "fomc decided",
)
_CAUSAL_MARKERS = (
    " because ",
    " caused by ",
    " due to ",
    " driven by ",
    " reflecting ",
    " as a result of ",
)
_ACTUAL_RELEASE_BASIS = "actual_release_ts"
_SEALED_ALFRED_BASIS = "sealed_alfred_d1_upper_bound"


@dataclass(frozen=True)
class CandidateContent:
    reasoning: str
    final_analysis: str
    evidence_ids: tuple[str, ...]

    @property
    def combined_text(self) -> str:
        return f"{self.reasoning}\n{self.final_analysis}"

    @property
    def response(self) -> str:
        return f"{self.reasoning.strip()}\n</think>\n{self.final_analysis.strip()}"


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    error_codes: tuple[str, ...]
    details: Mapping[str, Any]
    candidate: CandidateContent | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "error_codes": list(self.error_codes),
            "details": dict(self.details),
            "candidate": (
                {
                    "reasoning": self.candidate.reasoning,
                    "final_analysis": self.candidate.final_analysis,
                    "evidence_ids": list(self.candidate.evidence_ids),
                    "response_sha256": sha256_text(self.candidate.response),
                }
                if self.candidate is not None
                else None
            ),
        }


def _as_utc(value: object, *, label: str) -> datetime:
    text = str(value or "").strip()
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO-8601, got {text!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _parse_candidate(value: str | Mapping[str, Any]) -> CandidateContent:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```") or text.endswith("```"):
            raise ValueError("candidate_json_markdown_fence")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("candidate_invalid_json") from exc
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise ValueError("candidate_not_object")
    if not isinstance(payload, dict):
        raise ValueError("candidate_not_object")
    expected = {"reasoning", "final_analysis", "evidence_ids"}
    if set(payload) != expected:
        raise ValueError("candidate_schema_keys")
    reasoning = payload["reasoning"]
    final = payload["final_analysis"]
    evidence_ids = payload["evidence_ids"]
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("candidate_empty_reasoning")
    if not isinstance(final, str) or not final.strip():
        raise ValueError("candidate_empty_final")
    if (
        not isinstance(evidence_ids, list)
        or not evidence_ids
        or any(not isinstance(item, str) or not item.strip() for item in evidence_ids)
    ):
        raise ValueError("candidate_invalid_evidence_ids")
    cleaned_ids = tuple(item.strip() for item in evidence_ids)
    if len(cleaned_ids) != len(set(cleaned_ids)):
        raise ValueError("candidate_duplicate_evidence_ids")
    return CandidateContent(reasoning.strip(), final.strip(), cleaned_ids)


def _decimal(value: object) -> Decimal | None:
    try:
        parsed = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _normalise_numeric_token(number: str, sign: str = "") -> Decimal | None:
    return _decimal(f"{sign}{number}")


def _numeric_claims(text: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(0).strip()
        value = _normalise_numeric_token(match.group("number"), match.group("sign") or "")
        if value is None:
            continue
        claims.append(
            {
                "raw": raw,
                "value": value,
                "suffix": (match.group("suffix") or "").strip().casefold(),
                "start": match.start(),
            }
        )
    return claims


def _evidence_numeric_values(evidence: Mapping[str, Any]) -> set[Decimal]:
    values: set[Decimal] = set()
    for field in ("value", "display_value", "result"):
        raw = evidence.get(field)
        if raw is None:
            continue
        if isinstance(raw, (int, float, Decimal, str)):
            value = _decimal(raw)
            if value is not None:
                values.add(value)
    formula = evidence.get("formula")
    if isinstance(formula, Mapping):
        result = _decimal(formula.get("result"))
        if result is not None:
            values.add(result)
    for raw in evidence.get("allowed_numeric_forms", []) or []:
        value = _decimal(raw)
        if value is not None:
            values.add(value)
    return values


def _numeric_supported(claim: Decimal, allowed: Iterable[Decimal]) -> bool:
    for value in allowed:
        # Exact decimal equality is preferred.  The small tolerance admits a
        # declared rounded display form without accepting unrelated values.
        tolerance = max(Decimal("0.000001"), abs(value) * Decimal("0.000001"))
        if abs(claim - value) <= tolerance:
            return True
    return False


def token_words(text: str) -> list[str]:
    return [match.group(0).casefold().replace("’", "'") for match in _WORD_RE.finditer(text)]


def ngram_overlap(
    candidate_text: str,
    reference_text: str,
    *,
    n: int = 8,
) -> list[str]:
    if n < 1:
        raise ValueError("n must be positive")
    candidate = token_words(candidate_text)
    reference = token_words(reference_text)
    if len(candidate) < n or len(reference) < n:
        return []
    reference_ngrams = {
        tuple(reference[index : index + n])
        for index in range(len(reference) - n + 1)
    }
    overlaps: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for index in range(len(candidate) - n + 1):
        gram = tuple(candidate[index : index + n])
        if gram in reference_ngrams and gram not in seen:
            overlaps.append(" ".join(gram))
            seen.add(gram)
    return overlaps


def _availability_errors(
    evidence: Mapping[str, Any],
    *,
    index: int,
    cutoff: datetime,
) -> list[str]:
    """Validate the availability proof already normalized by ``source_data``.

    Canonical ALFRED fact cards may deliberately omit an exact release time.
    In that case the sealed D-1 snapshot supplies a conservative upper bound;
    the bound proves that the value existed no later than that timestamp but
    must never be relabelled as an actual release time.

    ``availability_ts`` remains supported for older, already-normalized fact
    cards.  Canonical upper-bound fields, however, require an explicit basis so
    a partially copied provenance object fails closed.
    """

    errors: list[str] = []
    release = evidence.get("release_ts")
    upper_bound = evidence.get("availability_upper_bound_ts")
    legacy = evidence.get("availability_ts")
    release_present = release not in (None, "")
    upper_bound_present = upper_bound not in (None, "")
    legacy_present = legacy not in (None, "")
    basis = str(evidence.get("availability_basis") or "").strip()

    if basis == _ACTUAL_RELEASE_BASIS:
        if not release_present:
            errors.append("fact_card_missing_release_or_availability")
    elif basis == _SEALED_ALFRED_BASIS:
        if not upper_bound_present:
            errors.append("fact_card_missing_release_or_availability")
        if release_present or legacy_present:
            errors.append("fact_card_inconsistent_availability")
    elif basis:
        errors.append("fact_card_invalid_availability_basis")
    else:
        if upper_bound_present:
            errors.append("fact_card_inconsistent_availability")
        if not release_present and not legacy_present:
            errors.append("fact_card_missing_release_or_availability")

    for field, value in (
        ("release_ts", release),
        ("availability_upper_bound_ts", upper_bound),
        ("availability_ts", legacy),
    ):
        if value in (None, ""):
            continue
        try:
            available_at = _as_utc(value, label=f"evidence[{index}].{field}")
        except ValueError:
            # Preserve compatibility with legacy date-granularity attestations.
            # Canonical source_data timestamps always take the timezone-aware
            # branch above.
            try:
                available_date = date.fromisoformat(str(value))
            except ValueError:
                errors.append("fact_card_invalid_availability")
            else:
                if available_date > cutoff.date():
                    errors.append("fact_card_post_cutoff_availability")
        else:
            if available_at > cutoff:
                errors.append("fact_card_post_cutoff_availability")
    return errors


def validate_fact_card(fact_card: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    cutoff_raw = fact_card.get("cutoff_ts")
    try:
        cutoff = _as_utc(cutoff_raw, label="cutoff_ts")
    except ValueError:
        return ["fact_card_invalid_cutoff"]
    evidence_rows = fact_card.get("evidence")
    if not isinstance(evidence_rows, list) or not evidence_rows:
        return ["fact_card_empty_evidence"]
    seen_ids: set[str] = set()
    for index, evidence in enumerate(evidence_rows):
        if not isinstance(evidence, Mapping):
            errors.append("fact_card_invalid_evidence")
            continue
        evidence_id = str(evidence.get("evidence_id") or "").strip()
        if not evidence_id or evidence_id in seen_ids:
            errors.append("fact_card_duplicate_or_empty_evidence_id")
        seen_ids.add(evidence_id)
        source_hash = str(evidence.get("source_sha256") or "").strip().casefold()
        if len(source_hash) != 64 or any(ch not in "0123456789abcdef" for ch in source_hash):
            errors.append("fact_card_invalid_source_hash")
        errors.extend(_availability_errors(evidence, index=index, cutoff=cutoff))
        observation = evidence.get("observation_date")
        try:
            if observation and date.fromisoformat(str(observation)) > cutoff.date():
                errors.append("fact_card_post_cutoff_observation")
        except ValueError:
            errors.append("fact_card_invalid_observation_date")
        if not _evidence_numeric_values(evidence):
            errors.append("fact_card_evidence_without_numeric_value")
    return list(dict.fromkeys(errors))


def verify_candidate(
    raw_candidate: str | Mapping[str, Any],
    *,
    fact_card: Mapping[str, Any],
    same_sample_minutes: str = "",
    max_reasoning_tokens: int = 1024,
    max_final_tokens: int = 512,
    token_counter: Any | None = None,
) -> VerificationResult:
    errors = validate_fact_card(fact_card)
    details: dict[str, Any] = {}
    try:
        candidate = _parse_candidate(raw_candidate)
    except ValueError as exc:
        code = str(exc)
        errors.append(code if code.startswith("candidate_") else "candidate_invalid")
        return VerificationResult(False, tuple(dict.fromkeys(errors)), details, None)

    combined = candidate.combined_text
    if _MODEL_TOKEN_RE.search(combined):
        errors.append("candidate_forbidden_model_token")
    if "\x00" in combined or "\ufffd" in combined:
        errors.append("candidate_encoding_error")

    if token_counter is None:
        reasoning_tokens = len(token_words(candidate.reasoning))
        final_tokens = len(token_words(candidate.final_analysis))
        details["token_count_method"] = "word_proxy"
    else:
        reasoning_tokens = int(token_counter(candidate.reasoning))
        final_tokens = int(token_counter(candidate.final_analysis))
        details["token_count_method"] = "tokenizer"
    details.update(
        reasoning_tokens=reasoning_tokens,
        final_tokens=final_tokens,
    )
    if reasoning_tokens > max_reasoning_tokens:
        errors.append("candidate_reasoning_too_long")
    if final_tokens > max_final_tokens:
        errors.append("candidate_final_too_long")

    evidence_rows = {
        str(row.get("evidence_id")): row
        for row in fact_card.get("evidence", [])
        if isinstance(row, Mapping) and row.get("evidence_id")
    }
    unknown_ids = sorted(set(candidate.evidence_ids) - set(evidence_rows))
    if unknown_ids:
        errors.append("candidate_unknown_evidence_id")
    details["unknown_evidence_ids"] = unknown_ids
    allowed_values: set[Decimal] = set()
    for evidence_id in candidate.evidence_ids:
        row = evidence_rows.get(evidence_id)
        if row is not None:
            allowed_values.update(_evidence_numeric_values(row))
    claims = _numeric_claims(combined)
    unsupported = [
        claim["raw"]
        for claim in claims
        if not _numeric_supported(claim["value"], allowed_values)
    ]
    details["numeric_claim_count"] = len(claims)
    details["unsupported_numeric_claims"] = unsupported
    if unsupported:
        errors.append("candidate_unsupported_numeric_claim")

    lowered = f" {combined.casefold()} "
    fact_text = canonical_json(fact_card).casefold()
    unsupported_terms = [
        term for term in _POLICY_OR_EVENT_TERMS if term in lowered and term not in fact_text
    ]
    details["unsupported_policy_or_event_terms"] = unsupported_terms
    if unsupported_terms:
        errors.append("candidate_external_policy_or_event")
    causal_markers = [marker.strip() for marker in _CAUSAL_MARKERS if marker in lowered]
    details["causal_markers"] = causal_markers
    if causal_markers and not bool(fact_card.get("allow_causal_claims", False)):
        errors.append("candidate_unlicensed_causal_claim")

    overlaps = ngram_overlap(combined, same_sample_minutes, n=8) if same_sample_minutes else []
    details["minutes_8token_overlaps"] = overlaps
    if overlaps:
        errors.append("candidate_minutes_8token_overlap")

    reasoning_sentences = {
        sentence.strip().casefold()
        for sentence in re.split(r"(?<=[.!?])\s+", candidate.reasoning)
        if len(token_words(sentence)) >= 8
    }
    final_sentences = {
        sentence.strip().casefold()
        for sentence in re.split(r"(?<=[.!?])\s+", candidate.final_analysis)
        if len(token_words(sentence)) >= 8
    }
    duplicated = sorted(reasoning_sentences & final_sentences)
    details["duplicated_reasoning_final_sentences"] = duplicated
    if duplicated:
        errors.append("candidate_reasoning_final_duplication")

    used = set(candidate.evidence_ids) & set(evidence_rows)
    coverage = len(used) / len(evidence_rows) if evidence_rows else 0.0
    details["evidence_coverage"] = coverage
    details["candidate_sha256"] = sha256_text(
        canonical_json(
            {
                "reasoning": candidate.reasoning,
                "final_analysis": candidate.final_analysis,
                "evidence_ids": list(candidate.evidence_ids),
            }
        )
    )
    deduped = tuple(dict.fromkeys(errors))
    return VerificationResult(not deduped, deduped, details, candidate)


def select_candidate(
    candidates: Sequence[tuple[VerificationResult, Mapping[str, Any]]],
) -> int | None:
    """Return the passing candidate index by coverage, then shorter output."""

    ranked: list[tuple[float, int, int]] = []
    for index, (verification, critic) in enumerate(candidates):
        if not verification.passed or verification.candidate is None:
            continue
        try:
            validated_critic = validate_critic(critic)
        except ValueError:
            continue
        if not critic_accepts(validated_critic):
            continue
        coverage = float(verification.details.get("evidence_coverage", 0.0))
        if not math.isfinite(coverage):
            continue
        length = len(token_words(verification.candidate.combined_text))
        ranked.append((-coverage, length, index))
    return min(ranked)[2] if ranked else None
