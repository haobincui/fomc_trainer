"""Evaluate the fail-closed 12-step chk2 reward-v3 GPU smoke outputs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any


class SmokeGateError(ValueError):
    """Raised when smoke output is missing, malformed, or outside a gate."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeGateError(message)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), f"missing smoke artifact: {path}")
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SmokeGateError(f"{path}:{line_number}: invalid JSON") from exc
            _require(isinstance(row, dict), f"{path}:{line_number}: row must be an object")
            result.append(row)
    _require(result, f"smoke artifact is empty: {path}")
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def check_smoke(output_dir: str | Path, *, expected_steps: int = 12) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    _require(root.is_dir(), f"smoke output directory does not exist: {root}")
    safety = _jsonl(root / "runtime_safety.jsonl")
    rewards = _jsonl(root / "reward.jsonl")

    observed_steps = sorted(
        {
            int(row["step"])
            for row in safety
            if isinstance(row.get("step"), int) and int(row["step"]) > 0
        }
    )
    _require(observed_steps, "runtime safety log contains no positive training steps")
    max_step = max(observed_steps)

    peak_reserved = max(float(row.get("peak_reserved_gib", 0.0)) for row in safety)
    peak_allocated = max(float(row.get("peak_allocated_gib", 0.0)) for row in safety)
    _require(
        all(math.isfinite(value) for value in (peak_reserved, peak_allocated)),
        "memory log contains a non-finite value",
    )

    clip_by_step: dict[int, float] = {}
    grad_by_step: dict[int, float] = {}
    for row in safety:
        step = row.get("step")
        if not isinstance(step, int) or step <= 0:
            continue
        clip = row.get("completion_clipped_ratio")
        if clip is not None:
            clip_by_step[step] = float(clip)
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict) and metrics.get("grad_norm") is not None:
            grad_by_step[step] = float(metrics["grad_norm"])
    clip_values = list(clip_by_step.values())
    _require(clip_values, "smoke did not log completion clipping")
    _require(
        all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in clip_values),
        "completion clipping log is invalid",
    )
    zero_or_fully_clipped_steps = {
        step for step, value in clip_by_step.items() if value >= 1.0 - 1e-12
    } | {step for step, value in grad_by_step.items() if abs(value) <= 1e-12}

    reward_values = [float(row.get("reward")) for row in rewards]
    reward_finite_and_bounded = all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in reward_values
    )
    judge_error_free = all(int(row.get("judge_attempts", 0)) == 1 for row in rewards)
    schema_valid = all(
        row.get("type") == "grounded_analysis_v3"
        and "candidate" not in row
        and "evidence" not in row
        for row in rewards
    )
    plain_answer_fraction = sum(
        bool(row.get("contract", {}).get("well_formed"))
        and bool(row.get("contract", {}).get("answer_nonempty"))
        for row in rewards
    ) / len(rewards)
    clipped_ratio = statistics.fmean(clip_values)
    zero_or_fully_clipped_fraction = len(zero_or_fully_clipped_steps) / max_step

    gates = {
        "exact_12_steps": max_step == expected_steps,
        "peak_reserved_le_22_gib": peak_reserved <= 22.0,
        "completion_clipped_ratio_le_0_25": clipped_ratio <= 0.25,
        "zero_or_fully_clipped_steps_le_0_15": zero_or_fully_clipped_fraction <= 0.15,
        "reward_finite_and_bounded": reward_finite_and_bounded,
        "judge_requests_error_free": judge_error_free,
        "reward_log_schema_and_privacy": schema_valid,
    }
    summary = {
        "schema_version": 1,
        "output_dir": str(root),
        "expected_steps": expected_steps,
        "max_step": max_step,
        "reward_records": len(rewards),
        "reward_mean": statistics.fmean(reward_values),
        "reward_median": statistics.median(reward_values),
        "plain_answer_fraction": plain_answer_fraction,
        "completion_clipped_ratio": clipped_ratio,
        "zero_or_fully_clipped_step_fraction": zero_or_fully_clipped_fraction,
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _atomic_json(root / "smoke_gate_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-steps", type=int, default=12)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = check_smoke(args.output_dir, expected_steps=args.expected_steps)
        if args.enforce and not summary["passed"]:
            raise SmokeGateError("chk2 reward-v3 smoke gates failed")
    except (OSError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
