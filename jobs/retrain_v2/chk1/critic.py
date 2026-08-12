"""Schema validation and acceptance policy for the independent chk0 critic."""

from __future__ import annotations

import json
from typing import Any, Mapping


def validate_critic(raw: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            payload = json.loads(raw.strip())
        except json.JSONDecodeError as exc:
            raise ValueError("critic_invalid_json") from exc
    elif isinstance(raw, Mapping):
        payload = dict(raw)
    else:
        raise ValueError("critic_not_object")
    expected = {
        "grounded",
        "unsupported_claims",
        "style_score",
        "reasoning_consistency",
    }
    if set(payload) != expected:
        raise ValueError("critic_schema_keys")
    if not isinstance(payload["grounded"], bool):
        raise ValueError("critic_grounded_not_boolean")
    if (
        not isinstance(payload["unsupported_claims"], list)
        or any(not isinstance(item, str) for item in payload["unsupported_claims"])
    ):
        raise ValueError("critic_unsupported_claims_not_string_list")
    score = payload["style_score"]
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError("critic_style_score_out_of_range")
    if not isinstance(payload["reasoning_consistency"], bool):
        raise ValueError("critic_consistency_not_boolean")
    return payload


def critic_accepts(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("grounded") is True
        and payload.get("unsupported_claims") == []
        and isinstance(payload.get("style_score"), int)
        and not isinstance(payload.get("style_score"), bool)
        and int(payload["style_score"]) >= 4
        and payload.get("reasoning_consistency") is True
    )


__all__ = ["critic_accepts", "validate_critic"]
