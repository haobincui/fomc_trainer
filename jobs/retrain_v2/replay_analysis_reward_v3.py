"""Replay grounded_analysis_v3 over saved rank-0 chk2 completion logs.

The replay consumes only the model-facing point-in-time evidence embedded in the
logged prompt and the saved completion.  It never loads Minutes, SFT targets, or
meeting outcomes.  Raw prompts and completions are not copied into its outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from open_r1.structured_response import parse_structured_response
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v3 import (
    grounded_analysis_reward_v3,
)


class RewardReplayError(ValueError):
    """Raised when completion provenance or replay output is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RewardReplayError(message)


def _extract_evidence(prompt: str) -> str:
    start = str(prompt).find("{")
    end = str(prompt).rfind("}")
    _require(start >= 0 and end > start, "completion prompt has no JSON evidence")
    text = str(prompt)[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RewardReplayError("completion prompt evidence is invalid JSON") from exc
    _require(isinstance(payload, dict), "completion prompt evidence must be an object")
    _require(set(payload) == {"atomic_topic", "evidence", "schema_version"}, "unsafe evidence schema")
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _distribution(values: list[float]) -> dict[str, Any]:
    _require(values, "cannot summarize an empty reward population")
    rounded = Counter(round(value, 12) for value in values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "population_std": statistics.pstdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "largest_exact_value_share": max(rounded.values()) / len(values),
        "largest_exact_value": rounded.most_common(1)[0][0],
    }


def replay_completion_directory(
    *,
    completion_dir: str | Path,
    output_dir: str | Path,
    judge_url: str,
    judge_model: str,
    judge_tokenizer_path: str,
    judge_max_model_len: int = 8192,
    judge_max_completion_tokens: int = 2048,
    judge_timeout: int = 180,
    judge_max_retries: int = 3,
    judge_backoff_seconds: float = 1.0,
    limit_files: int | None = None,
) -> dict[str, Any]:
    source = Path(completion_dir).resolve()
    destination = Path(output_dir).resolve()
    _require(source.is_dir(), f"completion directory does not exist: {source}")
    files = sorted(source.glob("completions_*.parquet"))
    _require(files, f"no completion parquet files found in {source}")
    if limit_files is not None:
        _require(limit_files > 0, "limit_files must be positive")
        files = files[:limit_files]
    destination.mkdir(parents=True, exist_ok=False)
    reward_log = destination / "reward_v3.jsonl"

    rows: list[dict[str, Any]] = []
    group_rewards: list[list[float]] = []
    for parquet_path in files:
        frame = pd.read_parquet(parquet_path)
        required = {
            "step",
            "prompt",
            "completion",
            "grounded_analysis_reward_v2",
        }
        _require(required <= set(frame.columns), f"{parquet_path} schema mismatch")
        _require(len(frame) == 4, f"{parquet_path} must contain one four-candidate group")
        evidence_rows = [_extract_evidence(prompt) for prompt in frame["prompt"]]
        _require(len(set(evidence_rows)) == 1, f"{parquet_path} mixes prompt groups")
        completions = [
            [{"content": str(completion)}] for completion in frame["completion"]
        ]
        rewards = grounded_analysis_reward_v3(
            completions,
            evidence_rows,
            meeting_date=[""] * len(completions),
            save_path=str(reward_log),
            url=judge_url,
            model=judge_model,
            tokenizer_path=judge_tokenizer_path,
            max_model_len=judge_max_model_len,
            max_completion_tokens=judge_max_completion_tokens,
            timeout=judge_timeout,
            max_retries=judge_max_retries,
            backoff_seconds=judge_backoff_seconds,
        )
        _require(len(rewards) == len(frame), "reward replay cardinality mismatch")
        group_rewards.append(rewards)
        for row_index, (_, row) in enumerate(frame.iterrows()):
            parsed = parse_structured_response(str(row["completion"]))
            rows.append(
                {
                    "source_file": parquet_path.name,
                    "row_index": row_index,
                    "step": int(row["step"]),
                    "v2_reward": float(row["grounded_analysis_reward_v2"]),
                    "v3_reward": float(rewards[row_index]),
                    "well_formed": bool(parsed.is_well_formed),
                    "answer_nonempty": bool(parsed.is_well_formed and parsed.answer.strip()),
                }
            )

    records = [json.loads(line) for line in reward_log.read_text(encoding="utf-8").splitlines() if line]
    _require(len(records) == len(rows), "v3 reward log cardinality mismatch")
    rewards = [row["v3_reward"] for row in rows]
    eligible = [row["v3_reward"] for row in rows if row["answer_nonempty"]]
    finite_and_bounded = all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in rewards)
    zero_variance_groups = sum(statistics.pstdev(group) <= 1e-12 for group in group_rewards)
    nonfactual = {
        violation["kind"]
        for record in records
        for violation in record.get("validated_violations", [])
        if violation.get("kind") in {"format", "omission", "style"}
    }
    # Non-factual categories have no coefficients in the persisted penalty schema.
    nonfactual_penalty_weight = 0.0
    report = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "status": "passed" if finite_and_bounded else "failed",
        "source": {
            "completion_dir": str(source),
            "files": len(files),
            "rows": len(rows),
            "contains_minutes_or_targets": False,
        },
        "judge": {
            "url": judge_url,
            "model": judge_model,
            "tokenizer_path": str(Path(judge_tokenizer_path).resolve()),
            "max_model_len": judge_max_model_len,
            "max_completion_tokens": judge_max_completion_tokens,
        },
        "all_completions": _distribution(rewards),
        "scoreable_completions": _distribution(eligible) if eligible else None,
        "well_formed_rate": sum(row["well_formed"] for row in rows) / len(rows),
        "zero_variance_group_rate": zero_variance_groups / len(group_rewards),
        "validated_nonfactual_kinds": sorted(nonfactual),
        "nonfactual_penalty_weight": nonfactual_penalty_weight,
        "gates": {
            "finite_and_bounded": finite_and_bounded,
            "scoreable_largest_exact_value_share_lt_0_30": bool(
                eligible and _distribution(eligible)["largest_exact_value_share"] < 0.30
            ),
            "zero_variance_group_rate_le_0_05": zero_variance_groups / len(group_rewards) <= 0.05,
            "nonfactual_penalty_weight_is_zero": nonfactual_penalty_weight == 0.0,
        },
    }
    (destination / "replay_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (destination / "replay_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--judge-model", default="Qwen3.5-9B")
    parser.add_argument("--judge-tokenizer-path", default="models/Qwen3.5-9B")
    parser.add_argument("--judge-max-model-len", type=int, default=8192)
    parser.add_argument("--judge-max-completion-tokens", type=int, default=2048)
    parser.add_argument("--judge-timeout", type=int, default=180)
    parser.add_argument("--judge-max-retries", type=int, default=3)
    parser.add_argument("--judge-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--limit-files", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = replay_completion_directory(
        completion_dir=args.completion_dir,
        output_dir=args.output_dir,
        judge_url=args.judge_url,
        judge_model=args.judge_model,
        judge_tokenizer_path=args.judge_tokenizer_path,
        judge_max_model_len=args.judge_max_model_len,
        judge_max_completion_tokens=args.judge_max_completion_tokens,
        judge_timeout=args.judge_timeout,
        judge_max_retries=args.judge_max_retries,
        judge_backoff_seconds=args.judge_backoff_seconds,
        limit_files=args.limit_files,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
