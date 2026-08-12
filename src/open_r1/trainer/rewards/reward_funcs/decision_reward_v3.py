"""Versioned chk4 decision reward with narrow fenced-JSON recovery.

``decision_dense_v2`` remains unchanged.  V3 preserves its full score for the
native contract::

    reasoning </think> {"direction":"hold","magnitude_bp":0}

It additionally recovers a prediction when, and only when, the answer is one
complete lowercase ``json`` Markdown fence containing a schema-exact decision
object.  A recovered fence is deliberately *not* format-valid and receives no
format component; its semantic components are multiplied by a fixed 0.25.  An
exact fenced prediction therefore receives 0.2375 rather than the strict 1.0.

No completion text is written to the optional audit log.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from open_r1.trainer.rewards.reward_funcs.decision_reward_v2 import (
    _completion_text,
    _save_records,
    _targets,
)


BOUNDARY = "</think>"
STRICT_JSON = "strict_json"
FENCED_JSON = "fenced_json"
INVALID = "invalid"
FENCED_SEMANTIC_DISCOUNT = 0.25
FORMAT_COMPONENT = 0.05
DIRECTION_COMPONENT = 0.45
MAGNITUDE_COMPONENT = 0.30
EXACT_COMPONENT = 0.20

_DIRECTIONS = frozenset({"cut", "hold", "hike"})
_ACTION_MAGNITUDES = frozenset({25, 50, 75, 100})
_DECISION_KEYS = frozenset({"direction", "magnitude_bp"})
_LEADING_THINK_RE = re.compile(r"\A\s*<think>", flags=re.IGNORECASE)
_FENCED_JSON_RE = re.compile(
    r"\A```json[ \t]*\r?\n(?P<body>.*?)\r?\n```[ \t]*\Z",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class DecisionParseV3:
    """Auditable result of the v3 response parser."""

    prediction: tuple[str, int] | None
    response_format: str
    boundary_count: int
    format_valid: bool
    fenced_recovered: bool
    rejection_reason: str | None


def _invalid(boundary_count: int, reason: str) -> DecisionParseV3:
    return DecisionParseV3(
        prediction=None,
        response_format=INVALID,
        boundary_count=boundary_count,
        format_valid=False,
        fenced_recovered=False,
        rejection_reason=reason,
    )


def _load_one_json(value: str) -> tuple[Any | None, str | None]:
    """Load one JSON value and reject all non-whitespace trailing bytes."""

    stripped = value.lstrip()
    if not stripped:
        return None, "empty_json"
    try:
        parsed, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return None, "invalid_or_incomplete_json"
    trailing = stripped[end:].strip()
    if trailing:
        try:
            json.JSONDecoder().raw_decode(trailing)
        except json.JSONDecodeError:
            return None, "extra_text_after_json"
        return None, "multiple_json_values"
    return parsed, None


def _validate_decision(value: Any) -> tuple[tuple[str, int] | None, str | None]:
    if not isinstance(value, dict):
        return None, "json_root_not_object"
    if set(value) != _DECISION_KEYS:
        return None, "decision_schema_keys_not_exact"
    direction = value.get("direction")
    magnitude = value.get("magnitude_bp")
    if not isinstance(direction, str):
        return None, "direction_type_invalid"
    if direction not in _DIRECTIONS:
        return None, "direction_domain_invalid"
    if isinstance(magnitude, bool) or not isinstance(magnitude, int):
        return None, "magnitude_type_invalid"
    if direction == "hold":
        if magnitude != 0:
            return None, "hold_magnitude_invalid"
    elif magnitude not in _ACTION_MAGNITUDES:
        return None, "action_magnitude_invalid"
    return (direction, magnitude), None


def parse_decision_response_v3(text: str) -> DecisionParseV3:
    """Parse only strict native JSON or one exact lowercase JSON fence."""

    if not isinstance(text, str) or not text.strip():
        return _invalid(0, "empty_completion")
    boundary_count = text.count(BOUNDARY)
    if boundary_count == 0:
        return _invalid(0, "missing_think_boundary")
    if boundary_count != 1:
        return _invalid(boundary_count, "multiple_think_boundaries")

    reasoning, answer = text.split(BOUNDARY, 1)
    reasoning = _LEADING_THINK_RE.sub("", reasoning, count=1).strip()
    if not reasoning:
        return _invalid(boundary_count, "empty_reasoning")
    answer = answer.strip()
    if not answer:
        return _invalid(boundary_count, "empty_answer")

    if answer.startswith("```json"):
        match = _FENCED_JSON_RE.fullmatch(answer)
        if match is None:
            return _invalid(boundary_count, "invalid_or_incomplete_json_fence")
        value, json_error = _load_one_json(match.group("body"))
        if json_error is not None:
            return _invalid(boundary_count, f"fenced_{json_error}")
        prediction, schema_error = _validate_decision(value)
        if schema_error is not None:
            return _invalid(boundary_count, schema_error)
        return DecisionParseV3(
            prediction=prediction,
            response_format=FENCED_JSON,
            boundary_count=boundary_count,
            format_valid=False,
            fenced_recovered=True,
            rejection_reason=None,
        )

    if not answer.startswith("{"):
        return _invalid(boundary_count, "answer_is_not_plain_or_fenced_json")
    value, json_error = _load_one_json(answer)
    if json_error is not None:
        return _invalid(boundary_count, json_error)
    prediction, schema_error = _validate_decision(value)
    if schema_error is not None:
        return _invalid(boundary_count, schema_error)
    return DecisionParseV3(
        prediction=prediction,
        response_format=STRICT_JSON,
        boundary_count=boundary_count,
        format_valid=True,
        fenced_recovered=False,
        rejection_reason=None,
    )


def _score_prediction(
    prediction: tuple[str, int],
    target: tuple[str, int],
    *,
    response_format: str,
) -> dict[str, Any]:
    predicted_direction, predicted_magnitude = prediction
    target_direction, target_magnitude = target
    direction_correct = predicted_direction == target_direction
    magnitude_score = 0.0
    if direction_correct:
        magnitude_score = max(
            0.0,
            1.0 - abs(predicted_magnitude - target_magnitude) / 100.0,
        )
    exact = prediction == target
    semantic_score = (
        DIRECTION_COMPONENT * float(direction_correct)
        + MAGNITUDE_COMPONENT * magnitude_score
        + EXACT_COMPONENT * float(exact)
    )
    if response_format == STRICT_JSON:
        format_score = FORMAT_COMPONENT
        format_discount = 1.0
        reward = format_score + semantic_score
    elif response_format == FENCED_JSON:
        format_score = 0.0
        format_discount = FENCED_SEMANTIC_DISCOUNT
        reward = semantic_score * format_discount
    else:  # pragma: no cover - internal caller invariant
        raise ValueError(f"unsupported response format: {response_format}")
    return {
        "direction_correct": direction_correct,
        "magnitude_score": magnitude_score,
        "exact": exact,
        "format_component": format_score,
        "semantic_score_before_discount": semantic_score,
        "semantic_discount": format_discount,
        "reward": max(0.0, min(1.0, reward)),
    }


def decision_dense_reward_v3(
    completions: list[list[dict[str, str]]],
    direction: list[str] | None = None,
    magnitude_bp: list[int] | None = None,
    rate_change: list[str] | None = None,
    meeting_date: list[str] | None = None,
    save_path: str | None = None,
    **_: Any,
) -> list[float]:
    """Return v2-compatible strict scores plus discounted fenced recovery."""

    targets = _targets(len(completions), direction, magnitude_bp, rate_change)
    dates = [""] * len(completions) if meeting_date is None else meeting_date
    if len(dates) != len(completions):
        raise ValueError("meeting_date must match completions")

    rewards: list[float] = []
    records: list[dict[str, Any]] = []
    for completion, target, date in zip(completions, targets, dates, strict=True):
        parsed = parse_decision_response_v3(_completion_text(completion))
        common = {
            "type": "decision_dense_v3",
            "meeting_date": date,
            "target": {"direction": target[0], "magnitude_bp": target[1]},
            "prediction": (
                {
                    "direction": parsed.prediction[0],
                    "magnitude_bp": parsed.prediction[1],
                }
                if parsed.prediction is not None
                else None
            ),
            "response_format": parsed.response_format,
            "boundary_count": parsed.boundary_count,
            "format_valid": parsed.format_valid,
            "strict_format_valid": parsed.response_format == STRICT_JSON,
            "fenced_recovered": parsed.fenced_recovered,
            "recovery_mode": (
                FENCED_JSON if parsed.fenced_recovered else "none"
            ),
            "rejection_reason": parsed.rejection_reason,
        }
        if parsed.prediction is None:
            reward = 0.0
            record = {
                **common,
                "direction_correct": False,
                "magnitude_score": 0.0,
                "exact": False,
                "format_component": 0.0,
                "semantic_score_before_discount": 0.0,
                "semantic_discount": 0.0,
                "reward": reward,
            }
        else:
            score = _score_prediction(
                parsed.prediction,
                target,
                response_format=parsed.response_format,
            )
            reward = float(score["reward"])
            record = {**common, **score}
        rewards.append(reward)
        records.append(record)
    _save_records(save_path, records)
    return rewards


__all__ = [
    "DecisionParseV3",
    "FENCED_SEMANTIC_DISCOUNT",
    "decision_dense_reward_v3",
    "parse_decision_response_v3",
]
