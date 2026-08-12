"""Deterministic hierarchical decision reward for retrain-v2."""

from __future__ import annotations

import fcntl
import json
import re
import threading
from pathlib import Path
from typing import Any

from open_r1.structured_response import parse_structured_response


_DIRECTIONS = {"cut", "hold", "hike"}
_ACTION_MAGNITUDES = {25, 50, 75, 100}
_LOG_LOCK = threading.Lock()


def _completion_text(completion: list[dict[str, str]]) -> str:
    if not completion or not isinstance(completion[0], dict):
        return ""
    return str(completion[0].get("content") or "")


def parse_decision_json(text: str) -> tuple[str, int] | None:
    parsed = parse_structured_response(text)
    if not parsed.is_well_formed:
        return None
    try:
        value = json.loads(parsed.answer)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or set(value) != {"direction", "magnitude_bp"}:
        return None
    direction = value.get("direction")
    magnitude = value.get("magnitude_bp")
    if direction not in _DIRECTIONS or isinstance(magnitude, bool) or not isinstance(magnitude, int):
        return None
    if direction == "hold" and magnitude != 0:
        return None
    if direction != "hold" and magnitude not in _ACTION_MAGNITUDES:
        return None
    return direction, magnitude


def _parse_legacy_target(value: str) -> tuple[str, int]:
    normalized = re.sub(r"\s+", " ", str(value or "").strip().lower())
    if normalized in {"hold", "no change", "unchanged"}:
        return "hold", 0
    match = re.search(r"(raise|hike|cut)\D*(25|50|75|100)", normalized)
    if not match:
        raise ValueError(f"unsupported decision target: {value!r}")
    direction = "cut" if match.group(1) == "cut" else "hike"
    return direction, int(match.group(2))


def _targets(
    count: int,
    direction: list[str] | None,
    magnitude_bp: list[int] | None,
    rate_change: list[str] | None,
) -> list[tuple[str, int]]:
    if direction is not None and magnitude_bp is not None:
        if len(direction) != count or len(magnitude_bp) != count:
            raise ValueError("decision target columns must match completions")
        result = [(str(d).strip().lower(), int(m)) for d, m in zip(direction, magnitude_bp, strict=True)]
    elif rate_change is not None:
        if len(rate_change) != count:
            raise ValueError("rate_change must match completions")
        result = [_parse_legacy_target(value) for value in rate_change]
    else:
        raise ValueError("direction+magnitude_bp or rate_change targets are required")
    for item in result:
        valid_magnitude = (
            item[1] == 0 if item[0] == "hold" else item[1] in _ACTION_MAGNITUDES
        )
        if item[0] not in _DIRECTIONS or not valid_magnitude:
            raise ValueError(f"invalid decision target: {item!r}")
    return result


def _save_records(path: str | None, records: list[dict[str, Any]]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK, output_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def decision_dense_reward_v2(
    completions: list[list[dict[str, str]]],
    direction: list[str] | None = None,
    magnitude_bp: list[int] | None = None,
    rate_change: list[str] | None = None,
    meeting_date: list[str] | None = None,
    save_path: str | None = None,
    **_: Any,
) -> list[float]:
    targets = _targets(len(completions), direction, magnitude_bp, rate_change)
    dates = [""] * len(completions) if meeting_date is None else meeting_date
    if len(dates) != len(completions):
        raise ValueError("meeting_date must match completions")
    rewards: list[float] = []
    records: list[dict[str, Any]] = []
    for completion, target, date in zip(completions, targets, dates, strict=True):
        content = _completion_text(completion)
        prediction = parse_decision_json(content)
        if prediction is None:
            rewards.append(0.0)
            records.append(
                {
                    "type": "decision_dense_v2",
                    "meeting_date": date,
                    "target": {"direction": target[0], "magnitude_bp": target[1]},
                    "prediction": None,
                    "format_valid": False,
                    "direction_correct": False,
                    "magnitude_score": 0.0,
                    "exact": False,
                    "reward": 0.0,
                }
            )
            continue
        predicted_direction, predicted_magnitude = prediction
        target_direction, target_magnitude = target
        direction_correct = predicted_direction == target_direction
        magnitude_score = 0.0
        if direction_correct:
            magnitude_score = max(0.0, 1.0 - abs(predicted_magnitude - target_magnitude) / 100.0)
        exact = prediction == target
        reward = 0.05 + 0.45 * float(direction_correct) + 0.30 * magnitude_score + 0.20 * float(exact)
        reward = max(0.0, min(1.0, reward))
        rewards.append(reward)
        records.append(
            {
                "type": "decision_dense_v2",
                "meeting_date": date,
                "target": {"direction": target_direction, "magnitude_bp": target_magnitude},
                "prediction": {
                    "direction": predicted_direction,
                    "magnitude_bp": predicted_magnitude,
                },
                "format_valid": True,
                "direction_correct": direction_correct,
                "magnitude_score": magnitude_score,
                "exact": exact,
                "reward": reward,
            }
        )
    _save_records(save_path, records)
    return rewards
