"""Deterministic whole-evidence prompt projection for chk1 and chk2.

The source fact card is already a point-in-time, lineage-bound projection.  A
training prompt may nevertheless be too large once the chk0 chat template and
stage system message are included.  This module removes *whole evidence
objects* (and their matching lineage rows) in a deterministic order.  It never
truncates strings or edits an evidence value.

Two properties are deliberately enforced at this boundary:

* a retained derived fact keeps the transitive closure of its
  ``operand_evidence_ids``; and
* the canonical JSON stored as ``provided_data`` occurs byte-for-byte exactly
  once in the rendered user prompt.

Consequently a caller can compact before teacher generation, or retain the
teacher's selected evidence IDs when projecting a completed SFT example,
without creating a prompt/evidence or evidence/lineage mismatch.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from typing import Any

from .contracts import (
    GENERATOR_SYSTEM_PROMPT,
    canonical_json,
    render_generator_payload,
    render_student_prompt,
    sha256_text,
)


PROJECTION_SCHEMA_VERSION = "analysis-prompt-projection-v1"
PROJECTION_POLICY = "whole-evidence-dependency-closure-priority-v1"
SFT_PROMPT_TOKEN_LIMIT = 3072
SFT_COMPLETION_TOKEN_LIMIT = 1024
SFT_TOTAL_TOKEN_LIMIT = 4096
GRPO_PROMPT_TOKEN_LIMIT = 2560
GENERATOR_PROMPT_TOKEN_LIMIT = 4096

# Keep these strings in a shared, importable module so generation-time
# projection and base-release auditing cannot silently count different chats.
ANALYSIS_SFT_SYSTEM_PROMPT = (
    "Analyze only the supplied point-in-time FOMC evidence. Do not infer facts from\n"
    "later releases or from the target meeting's Minutes. Use the model's native\n"
    "reasoning boundary, then provide a concise, evidence-grounded final analysis.\n"
)
ANALYSIS_GRPO_SYSTEM_PROMPT = (
    "Analyze only the supplied point-in-time FOMC evidence. Do not use later data\n"
    "or the target meeting's Minutes. Use the native reasoning boundary and make\n"
    "every factual or numerical statement traceable to the supplied evidence.\n"
)
GRPO_USER_PREAMBLE = (
    "Analyze the supplied point-in-time evidence for the atomic topic. Ground "
    "every factual and numerical statement in the JSON evidence.\n\n"
)


class PromptProjectionError(ValueError):
    """Raised when a budget cannot be met without weakening the contract."""


ChatTokenCounter = Callable[[str, str], int]
SftTokenAuditor = Callable[[str, str], Mapping[str, Any]]
PromptRenderer = Callable[[Mapping[str, Any], str], str]


@dataclass(frozen=True)
class PromptProjection:
    """One immutable prompt/provided-data/lineage projection."""

    prompt: str
    provided_data: str
    fact_card: Mapping[str, Any]
    evidence_lineage: tuple[Mapping[str, Any], ...]
    prompt_tokens: int
    attestation: Mapping[str, Any]


@dataclass(frozen=True)
class PreteacherProjection:
    """Safe model-facing chk1 teacher/SFT inputs produced before GPU work."""

    generator_prompt: str
    student_prompt: str
    provided_data: str
    fact_card: Mapping[str, Any]
    validation_fact_card: Mapping[str, Any]
    evidence_lineage: tuple[Mapping[str, Any], ...]
    attestation: Mapping[str, Any]


def projection_contract_sha256() -> str:
    """Bind policy, budgets, and exact stage system/user prompt contracts."""

    payload = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "policy": PROJECTION_POLICY,
        "budgets": {
            "chk1_generator": GENERATOR_PROMPT_TOKEN_LIMIT,
            "analysis_sft": {
                "prompt": SFT_PROMPT_TOKEN_LIMIT,
                "completion": SFT_COMPLETION_TOKEN_LIMIT,
                "total": SFT_TOTAL_TOKEN_LIMIT,
            },
            "analysis_grpo": GRPO_PROMPT_TOKEN_LIMIT,
        },
        "system_prompts": {
            "chk1_generator": GENERATOR_SYSTEM_PROMPT,
            "analysis_sft": ANALYSIS_SFT_SYSTEM_PROMPT,
            "analysis_grpo": ANALYSIS_GRPO_SYSTEM_PROMPT,
        },
        "grpo_user_preamble": GRPO_USER_PREAMBLE,
    }
    return sha256_text(canonical_json(payload))


def build_chat_token_counter(tokenizer: Any) -> ChatTokenCounter:
    """Return the exact non-truncating chk0 chat-template counter.

    Loading the tokenizer remains the caller's responsibility.  This keeps the
    pure projection layer free of model/GPU side effects and makes tests use
    the same explicit counting boundary as production.
    """

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise PromptProjectionError("tokenizer lacks apply_chat_template")

    def count(system_prompt: str, user_prompt: str) -> int:
        try:
            token_ids = apply_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                truncation=False,
            )
        except Exception as exc:  # pragma: no cover - tokenizer-specific detail
            raise PromptProjectionError(f"chat token counting failed: {exc}") from exc
        if isinstance(token_ids, (str, bytes)) or not isinstance(token_ids, Sequence):
            raise PromptProjectionError("chat tokenizer returned a non-sequence")
        return len(token_ids)

    return count


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer(text=text)
    except Exception as exc:  # pragma: no cover - tokenizer-specific detail
        raise PromptProjectionError(f"tokenizer encoding failed: {exc}") from exc
    ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else getattr(
        encoded, "input_ids", None
    )
    if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence):
        raise PromptProjectionError("tokenizer returned invalid input_ids")
    if ids and isinstance(ids[0], Sequence) and not isinstance(ids[0], (str, bytes)):
        if len(ids) != 1:
            raise PromptProjectionError("tokenizer returned an unexpected batch")
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def build_sft_token_auditor(tokenizer: Any) -> SftTokenAuditor:
    """Build the exact chk1 prompt/completion/total budget audit.

    The completion is counted as the continuation of the rendered system/user
    chat, including one tokenizer EOS exactly as the SFT release builder does.
    Prefix instability is rejected rather than approximated.
    """

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        raise PromptProjectionError("tokenizer lacks apply_chat_template")

    def audit(prompt: str, response: str) -> Mapping[str, Any]:
        try:
            rendered = apply_template(
                [
                    {"role": "system", "content": ANALYSIS_SFT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:  # pragma: no cover - tokenizer-specific detail
            raise PromptProjectionError(f"SFT chat rendering failed: {exc}") from exc
        if not isinstance(rendered, str) or not rendered:
            raise PromptProjectionError("SFT chat template did not return text")
        completion = response
        eos = getattr(tokenizer, "eos_token", None)
        if isinstance(eos, str) and eos and not completion.endswith(eos):
            completion += eos
        prompt_ids = _token_ids(tokenizer, rendered)
        full_ids = _token_ids(tokenizer, rendered + completion)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise PromptProjectionError("SFT tokenizer continuation prefix mismatch")
        completion_tokens = len(full_ids) - len(prompt_ids)
        audit_payload = {
            "schema_version": "chk1-sft-token-budget-v1",
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": completion_tokens,
            "total_tokens": len(full_ids),
            "max_prompt_tokens": SFT_PROMPT_TOKEN_LIMIT,
            "max_completion_tokens": SFT_COMPLETION_TOKEN_LIMIT,
            "max_total_tokens": SFT_TOTAL_TOKEN_LIMIT,
            "overflow_policy": "error",
            "truncated": False,
        }
        audit_payload["passed"] = (
            audit_payload["prompt_tokens"] <= SFT_PROMPT_TOKEN_LIMIT
            and completion_tokens <= SFT_COMPLETION_TOKEN_LIMIT
            and audit_payload["total_tokens"] <= SFT_TOTAL_TOKEN_LIMIT
        )
        return audit_payload

    return audit


def render_grpo_user_prompt(fact_card: Mapping[str, Any], provided_data: str) -> str:
    """Render chk2 input with one byte-identical copy of ``provided_data``."""

    if provided_data != canonical_json(dict(fact_card)):
        raise PromptProjectionError("GRPO provided_data does not bind its fact card")
    return GRPO_USER_PREAMBLE + provided_data


def safe_model_fact_card(fact_card: Mapping[str, Any]) -> dict[str, Any]:
    """Remove target-meeting identity fields before any model sees the card.

    ``sample_id`` is unsafe even when it looks opaque: production chk1 IDs
    contain the target meeting date.  The meeting and canonical-key fields are
    likewise local manifest metadata, never model input.
    """

    required = ("schema_version", "atomic_topic", "evidence")
    missing = [field for field in required if field not in fact_card]
    if missing:
        raise PromptProjectionError(f"model fact card lacks fields: {missing}")
    raw_cutoff = fact_card.get("cutoff_ts")
    cutoff = (
        _required_text(raw_cutoff, label="fact_card.cutoff_ts")
        if raw_cutoff is not None
        else None
    )
    safe = {field: deepcopy(fact_card[field]) for field in required}
    evidence = safe.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise PromptProjectionError("model fact card has no evidence")
    for index, item in enumerate(evidence):
        if not isinstance(item, dict):
            raise PromptProjectionError(
                f"model fact_card.evidence[{index}] is not an object"
            )
        if cutoff is None and (
            "cutoff_ts" in item or "availability_upper_bound_ts" in item
        ):
            raise PromptProjectionError(
                f"model fact_card.evidence[{index}] retains date-bearing bounds"
            )
        if cutoff is not None and item.get("cutoff_ts") != cutoff:
            raise PromptProjectionError(
                f"model fact_card.evidence[{index}] cutoff binding mismatch"
            )
        # Cutoff remains in the local manifest and lineage for validation.  It
        # is date-bearing metadata and is not useful model input.
        item.pop("cutoff_ts", None)
        # Sparse-v2 upper bounds are deterministically 18/19 hours before the
        # cutoff and therefore reveal the same target-meeting boundary.
        item.pop("availability_upper_bound_ts", None)
        # Production source IDs also embed the meeting date.  ``series_id`` is
        # the non-target model-facing identity; the exact source ID remains in
        # lineage for audit.
        item.pop("source_id", None)
    return safe


def safe_grpo_fact_card(fact_card: Mapping[str, Any]) -> dict[str, Any]:
    """Backward-compatible stage-specific name for the shared safe projection."""

    return safe_model_fact_card(fact_card)


def project_preteacher_inputs(
    *,
    fact_card: Mapping[str, Any],
    evidence_lineage: Sequence[Mapping[str, Any]],
    atomic_topic: str,
    style_guide: Mapping[str, Any],
    student_chat_token_counter: ChatTokenCounter,
    generator_prompt_token_counter: Callable[[str], int],
) -> PreteacherProjection:
    """Build safe teacher and SFT inputs before any generation model is called."""

    topic = _required_text(atomic_topic, label="atomic_topic").strip()
    if not isinstance(style_guide, Mapping):
        raise PromptProjectionError("style_guide must be an object")
    safe_fact = safe_model_fact_card(fact_card)

    def render_sft(card: Mapping[str, Any], provided_data: str) -> str:
        prompt = render_student_prompt(
            fact_card=card,
            atomic_topic=topic,
            style_guide=style_guide,
        )
        if prompt.count(provided_data) != 1:
            raise PromptProjectionError(
                "student prompt does not bind safe provided_data exactly once"
            )
        return prompt

    student = project_prompt_to_budget(
        projection_name="analysis_sft_preteacher",
        fact_card=safe_fact,
        evidence_lineage=evidence_lineage,
        system_prompt=ANALYSIS_SFT_SYSTEM_PROMPT,
        max_prompt_tokens=SFT_PROMPT_TOKEN_LIMIT,
        prompt_renderer=render_sft,
        chat_token_counter=student_chat_token_counter,
    )
    generator_prompt = render_generator_payload(
        fact_card=student.fact_card,
        atomic_topic=topic,
        style_guide=style_guide,
    )
    if generator_prompt.count(student.provided_data) != 1:
        raise PromptProjectionError(
            "generator prompt does not bind safe provided_data exactly once"
        )
    try:
        generator_tokens = generator_prompt_token_counter(generator_prompt)
    except PromptProjectionError:
        raise
    except Exception as exc:
        raise PromptProjectionError(
            f"generator prompt token counter failed: {exc}"
        ) from exc
    if (
        isinstance(generator_tokens, bool)
        or not isinstance(generator_tokens, int)
        or generator_tokens < 0
    ):
        raise PromptProjectionError("generator token counter returned an invalid count")
    if generator_tokens > GENERATOR_PROMPT_TOKEN_LIMIT:
        raise PromptProjectionError(
            "safe generator prompt exceeds the 4096-token hard budget"
        )

    source_evidence = fact_card.get("evidence")
    if not isinstance(source_evidence, list):
        raise PromptProjectionError("source fact card evidence is not a list")
    source_by_id = {
        str(item.get("evidence_id") or ""): item
        for item in source_evidence
        if isinstance(item, Mapping)
    }
    selected_ids = [
        str(item.get("evidence_id") or "") for item in student.fact_card["evidence"]
    ]
    if "" in selected_ids or not set(selected_ids) <= set(source_by_id):
        raise PromptProjectionError("safe projection lost its source evidence binding")
    validation_fact = deepcopy(dict(fact_card))
    validation_fact["evidence"] = [
        deepcopy(source_by_id[evidence_id]) for evidence_id in selected_ids
    ]

    attestation = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "projection_contract_sha256": projection_contract_sha256(),
        "safe_model_fact_card_sha256": sha256_text(student.provided_data),
        "validation_fact_card_sha256": sha256_text(canonical_json(validation_fact)),
        "student_projection": dict(student.attestation),
        "generator_prompt_sha256": sha256_text(generator_prompt),
        "generator_system_prompt_sha256": sha256_text(GENERATOR_SYSTEM_PROMPT),
        "generator_prompt_tokens": generator_tokens,
        "generator_max_prompt_tokens": GENERATOR_PROMPT_TOKEN_LIMIT,
        "model_fact_card_keys": ["atomic_topic", "evidence", "schema_version"],
        "removed_identity_fields": [
            "availability_upper_bound_ts",
            "canonical_key",
            "cutoff_ts",
            "meeting_date",
            "sample_id",
            "source_id",
        ],
        "input_truncated": False,
        "string_truncation": False,
    }
    attestation["attestation_sha256"] = sha256_text(canonical_json(attestation))
    return PreteacherProjection(
        generator_prompt=generator_prompt,
        student_prompt=student.prompt,
        provided_data=student.provided_data,
        fact_card=student.fact_card,
        validation_fact_card=validation_fact,
        evidence_lineage=student.evidence_lineage,
        attestation=attestation,
    )


def _required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromptProjectionError(f"{label} must be non-empty text")
    if "\x00" in value or "\ufffd" in value:
        raise PromptProjectionError(f"{label} contains invalid encoding")
    return value


def _evidence_id(item: Mapping[str, Any], *, label: str) -> str:
    return _required_text(item.get("evidence_id"), label=f"{label}.evidence_id")


def _canonical_evidence_key(item: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(item.get("series_id") or item.get("source_id") or "").casefold(),
        str(item.get("observation_date") or ""),
        str(item.get("fact_kind") or "").casefold(),
        str(item.get("evidence_id") or ""),
    )


def _observation_ordinal(item: Mapping[str, Any]) -> int:
    raw = str(item.get("observation_date") or "")
    try:
        return date.fromisoformat(raw).toordinal()
    except ValueError:
        return -1


def _fact_priority(item: Mapping[str, Any]) -> int:
    kind = str(item.get("fact_kind") or "").casefold()
    if kind == "latest":
        return 0
    if kind in {
        "change_3m",
        "change_6m",
        "change_12m",
        "change_1q",
        "change_2q",
        "change_4q",
        "mom",
        "qoq",
        "yoy",
        "trend",
        "turning_point",
        "extreme",
    }:
        return 1
    if kind == "recent_observation":
        return 2
    if kind == "prior_target_range":
        return 3
    return 4


def _normalise_id_sequence(
    values: Sequence[str], *, label: str, known: set[str]
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PromptProjectionError(f"{label} must be a sequence of evidence IDs")
    result: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        evidence_id = _required_text(value, label=f"{label}[{index}]").strip()
        if evidence_id in seen:
            raise PromptProjectionError(f"{label} contains duplicate evidence IDs")
        if evidence_id not in known:
            raise PromptProjectionError(f"{label} references unknown evidence ID")
        seen.add(evidence_id)
        result.append(evidence_id)
    return tuple(sorted(result))


def _validated_bindings(
    fact_card: Mapping[str, Any], evidence_lineage: Sequence[Mapping[str, Any]]
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, tuple[str, ...]],
]:
    raw_evidence = fact_card.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise PromptProjectionError("fact_card.evidence must be a non-empty list")
    if (
        isinstance(evidence_lineage, (str, bytes, Mapping))
        or not isinstance(evidence_lineage, Sequence)
        or not evidence_lineage
    ):
        raise PromptProjectionError("evidence_lineage must be a non-empty sequence")

    evidence_by_id: dict[str, dict[str, Any]] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    for index, raw in enumerate(raw_evidence):
        if not isinstance(raw, Mapping):
            raise PromptProjectionError(f"fact_card.evidence[{index}] is not an object")
        item = deepcopy(dict(raw))
        evidence_id = _evidence_id(item, label=f"fact_card.evidence[{index}]")
        if evidence_id in evidence_by_id:
            raise PromptProjectionError("fact_card has duplicate evidence IDs")
        operands = item.get("operand_evidence_ids", [])
        if operands in (None, ""):
            operands = []
        if isinstance(operands, (str, bytes, Mapping)) or not isinstance(
            operands, Sequence
        ):
            raise PromptProjectionError(
                f"evidence {evidence_id!r} has invalid operand_evidence_ids"
            )
        operand_ids: list[str] = []
        for operand in operands:
            operand_ids.append(
                _required_text(
                    operand, label=f"evidence {evidence_id!r} operand"
                ).strip()
            )
        if len(operand_ids) != len(set(operand_ids)):
            raise PromptProjectionError(
                f"evidence {evidence_id!r} has duplicate operands"
            )
        evidence_by_id[evidence_id] = item
        dependencies[evidence_id] = tuple(sorted(operand_ids))

    known = set(evidence_by_id)
    for evidence_id, operands in dependencies.items():
        missing = sorted(set(operands) - known)
        if missing:
            raise PromptProjectionError(
                f"evidence {evidence_id!r} has missing operand evidence"
            )

    lineage_by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(evidence_lineage):
        if not isinstance(raw, Mapping):
            raise PromptProjectionError(f"evidence_lineage[{index}] is not an object")
        item = deepcopy(dict(raw))
        evidence_id = _evidence_id(item, label=f"evidence_lineage[{index}]")
        if evidence_id in lineage_by_id:
            raise PromptProjectionError("evidence_lineage has duplicate evidence IDs")
        lineage_by_id[evidence_id] = item
    if set(lineage_by_id) != known:
        raise PromptProjectionError("fact-card evidence/lineage ID sets differ")

    raw_card_cutoff = fact_card.get("cutoff_ts")
    card_cutoff = (
        _required_text(raw_card_cutoff, label="fact_card.cutoff_ts")
        if raw_card_cutoff is not None
        else None
    )
    lineage_cutoffs: set[str] = set()
    for evidence_id, evidence in evidence_by_id.items():
        lineage = lineage_by_id[evidence_id]
        if (
            not isinstance(lineage.get("source_id"), str)
            or not str(lineage.get("source_id")).strip()
        ):
            raise PromptProjectionError(
                f"lineage source_id is missing for {evidence_id!r}"
            )
        if evidence.get("source_id") is not None and evidence.get(
            "source_id"
        ) != lineage.get("source_id"):
            raise PromptProjectionError(
                f"evidence/lineage source_id mismatch for {evidence_id!r}"
            )
        for field in ("source_sha256",):
            if evidence.get(field) != lineage.get(field):
                raise PromptProjectionError(
                    f"evidence/lineage {field} mismatch for {evidence_id!r}"
                )
        lineage_cutoff = _required_text(
            lineage.get("cutoff_ts"), label=f"lineage cutoff for {evidence_id!r}"
        )
        lineage_cutoffs.add(lineage_cutoff)
        evidence_cutoff = evidence.get("cutoff_ts")
        if evidence_cutoff is not None and evidence_cutoff != lineage_cutoff:
            raise PromptProjectionError(
                f"evidence/lineage cutoff_ts mismatch for {evidence_id!r}"
            )
        if card_cutoff is not None and lineage_cutoff != card_cutoff:
            raise PromptProjectionError(f"evidence cutoff mismatch for {evidence_id!r}")
    if len(lineage_cutoffs) != 1:
        raise PromptProjectionError("evidence lineage mixes point-in-time cutoffs")
    return evidence_by_id, lineage_by_id, dependencies


def _dependency_closure(
    roots: Sequence[str], dependencies: Mapping[str, Sequence[str]]
) -> set[str]:
    selected: set[str] = set()
    visiting: set[str] = set()

    def visit(evidence_id: str) -> None:
        if evidence_id in selected:
            return
        if evidence_id in visiting:
            raise PromptProjectionError("evidence dependency cycle detected")
        visiting.add(evidence_id)
        for operand in dependencies[evidence_id]:
            visit(operand)
        visiting.remove(evidence_id)
        selected.add(evidence_id)

    for root in roots:
        visit(root)
    return selected


def _ranked_ids(
    evidence_by_id: Mapping[str, Mapping[str, Any]], preferred: set[str]
) -> list[str]:
    by_series: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in evidence_by_id.values():
        series = str(item.get("series_id") or item.get("source_id") or "").casefold()
        by_series[series].append(item)

    within_series_rank: dict[str, int] = {}
    for rows in by_series.values():
        ordered = sorted(
            rows,
            key=lambda item: (
                _fact_priority(item),
                -_observation_ordinal(item),
                str(item.get("evidence_id") or ""),
            ),
        )
        for index, item in enumerate(ordered):
            within_series_rank[str(item["evidence_id"])] = index

    def key(evidence_id: str) -> tuple[int, int, int, int, str, str]:
        item = evidence_by_id[evidence_id]
        series = str(item.get("series_id") or item.get("source_id") or "").casefold()
        return (
            0 if evidence_id in preferred else 1,
            within_series_rank[evidence_id],
            _fact_priority(item),
            -_observation_ordinal(item),
            series,
            evidence_id,
        )

    return sorted(evidence_by_id, key=key)


def project_prompt_to_budget(
    *,
    projection_name: str,
    fact_card: Mapping[str, Any],
    evidence_lineage: Sequence[Mapping[str, Any]],
    system_prompt: str,
    max_prompt_tokens: int,
    prompt_renderer: PromptRenderer,
    chat_token_counter: ChatTokenCounter,
    required_evidence_ids: Sequence[str] = (),
    preferred_evidence_ids: Sequence[str] = (),
) -> PromptProjection:
    """Project a fact card under an exact system+chat token budget.

    ``required_evidence_ids`` and their dependency closures can never be
    removed.  ``preferred_evidence_ids`` are tried first but remain optional.
    If no complete evidence object fits, or the required closure itself does
    not fit, the function fails closed.
    """

    name = _required_text(projection_name, label="projection_name").strip()
    system = _required_text(system_prompt, label="system_prompt")
    if (
        isinstance(max_prompt_tokens, bool)
        or not isinstance(max_prompt_tokens, int)
        or max_prompt_tokens <= 0
    ):
        raise PromptProjectionError("max_prompt_tokens must be a positive integer")
    if not callable(prompt_renderer) or not callable(chat_token_counter):
        raise PromptProjectionError("prompt renderer and token counter are required")
    if not isinstance(fact_card, Mapping):
        raise PromptProjectionError("fact_card must be an object")

    source_fact = deepcopy(dict(fact_card))
    evidence_by_id, lineage_by_id, dependencies = _validated_bindings(
        source_fact, evidence_lineage
    )
    known = set(evidence_by_id)
    required = _normalise_id_sequence(
        required_evidence_ids, label="required_evidence_ids", known=known
    )
    preferred = _normalise_id_sequence(
        preferred_evidence_ids, label="preferred_evidence_ids", known=known
    )

    attempts = 0

    def render(
        selected_ids: set[str],
    ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], str, str, int]:
        nonlocal attempts
        if not selected_ids:
            raise PromptProjectionError("a projection must retain evidence")
        attempts += 1
        ordered_ids = [
            evidence_id
            for evidence_id, _item in sorted(
                evidence_by_id.items(),
                key=lambda pair: _canonical_evidence_key(pair[1]),
            )
            if evidence_id in selected_ids
        ]
        projected_fact = deepcopy(source_fact)
        projected_fact["evidence"] = [
            deepcopy(evidence_by_id[evidence_id]) for evidence_id in ordered_ids
        ]
        projected_lineage = tuple(
            deepcopy(lineage_by_id[evidence_id]) for evidence_id in ordered_ids
        )
        provided_data = canonical_json(projected_fact)
        prompt = prompt_renderer(projected_fact, provided_data)
        _required_text(prompt, label="rendered prompt")
        if prompt.count(provided_data) != 1:
            raise PromptProjectionError(
                "rendered prompt must contain provided_data byte-for-byte exactly once"
            )
        try:
            token_count = chat_token_counter(system, prompt)
        except PromptProjectionError:
            raise
        except Exception as exc:
            raise PromptProjectionError(f"chat token counter failed: {exc}") from exc
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count < 0
        ):
            raise PromptProjectionError(
                "chat token counter returned an invalid token count"
            )
        return projected_fact, projected_lineage, provided_data, prompt, token_count

    all_ids = set(evidence_by_id)
    full = render(all_ids)
    if full[-1] <= max_prompt_tokens:
        selected = all_ids
        final = full
    else:
        selected = _dependency_closure(required, dependencies)
        if selected:
            required_projection = render(selected)
            if required_projection[-1] > max_prompt_tokens:
                raise PromptProjectionError(
                    "required evidence dependency closure exceeds the prompt budget"
                )
            final = required_projection
        else:
            final = None

        ranked = _ranked_ids(evidence_by_id, set(preferred))
        for evidence_id in ranked:
            additions = _dependency_closure((evidence_id,), dependencies)
            trial_ids = selected | additions
            if trial_ids == selected:
                continue
            trial = render(trial_ids)
            if trial[-1] <= max_prompt_tokens:
                selected = trial_ids
                final = trial
        if final is None or not selected:
            raise PromptProjectionError(
                "no complete evidence dependency closure fits the prompt budget"
            )

    projected_fact, projected_lineage, provided_data, prompt, token_count = final
    required_closure = _dependency_closure(required, dependencies)
    if not required_closure <= selected:
        raise AssertionError("required evidence closure was not retained")
    for evidence_id in selected:
        if not set(dependencies[evidence_id]) <= selected:
            raise AssertionError("projected evidence lost a dependency")
    if token_count > max_prompt_tokens:
        raise AssertionError("projected prompt exceeds its hard budget")

    selected_ids = tuple(
        str(item["evidence_id"]) for item in projected_fact["evidence"]
    )
    dropped_ids = tuple(sorted(all_ids - selected))
    evidence_bindings: list[dict[str, str]] = []
    for evidence, lineage in zip(
        projected_fact["evidence"], projected_lineage, strict=True
    ):
        evidence_id = str(evidence["evidence_id"])
        if lineage.get("evidence_id") != evidence_id:
            raise AssertionError("projected evidence/lineage order is not 1:1")
        evidence_bindings.append(
            {
                "evidence_id": evidence_id,
                "evidence_sha256": sha256_text(canonical_json(evidence)),
                "lineage_sha256": sha256_text(canonical_json(dict(lineage))),
            }
        )
    attestation = {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "policy": PROJECTION_POLICY,
        "projection_name": name,
        "projection_contract_sha256": projection_contract_sha256(),
        "system_prompt_sha256": sha256_text(system),
        "source_fact_card_sha256": sha256_text(canonical_json(source_fact)),
        "projected_fact_card_sha256": sha256_text(provided_data),
        "source_lineage_sha256": sha256_text(
            canonical_json([dict(item) for item in evidence_lineage])
        ),
        "projected_lineage_sha256": sha256_text(
            canonical_json([dict(item) for item in projected_lineage])
        ),
        "prompt_sha256": sha256_text(prompt),
        "provided_data_sha256": sha256_text(provided_data),
        "max_prompt_tokens": max_prompt_tokens,
        "prompt_tokens": token_count,
        "source_evidence_count": len(all_ids),
        "projected_evidence_count": len(selected),
        "selected_evidence_ids_sha256": sha256_text(canonical_json(selected_ids)),
        "dropped_evidence_ids_sha256": sha256_text(canonical_json(dropped_ids)),
        "required_evidence_ids_sha256": sha256_text(canonical_json(required)),
        "preferred_evidence_ids_sha256": sha256_text(canonical_json(preferred)),
        "evidence_bindings": evidence_bindings,
        "evidence_bindings_sha256": sha256_text(canonical_json(evidence_bindings)),
        "selection_attempt_count": attempts,
        "input_truncated": False,
        "string_truncation": False,
        "compaction_unit": "whole_evidence_object_with_dependency_closure",
    }
    # Hash the complete content-free attestation so it can be bound into a
    # generation cache or base-release manifest without copying evidence text.
    attestation["attestation_sha256"] = sha256_text(canonical_json(attestation))
    return PromptProjection(
        prompt=prompt,
        provided_data=provided_data,
        fact_card=projected_fact,
        evidence_lineage=projected_lineage,
        prompt_tokens=token_count,
        attestation=attestation,
    )


__all__ = [
    "ANALYSIS_GRPO_SYSTEM_PROMPT",
    "ANALYSIS_SFT_SYSTEM_PROMPT",
    "GENERATOR_PROMPT_TOKEN_LIMIT",
    "GRPO_PROMPT_TOKEN_LIMIT",
    "GRPO_USER_PREAMBLE",
    "PROJECTION_POLICY",
    "PROJECTION_SCHEMA_VERSION",
    "PromptProjection",
    "PromptProjectionError",
    "PreteacherProjection",
    "SFT_PROMPT_TOKEN_LIMIT",
    "build_chat_token_counter",
    "project_prompt_to_budget",
    "project_preteacher_inputs",
    "projection_contract_sha256",
    "render_grpo_user_prompt",
    "safe_grpo_fact_card",
    "safe_model_fact_card",
]
