"""Replay grounded_analysis_v3 over immutable chk2 completion parquet files.

The replay is resumable at one completion parquet per record file.  It never
loads Minutes or teacher targets and never persists raw prompts/completions in
its own outputs; those remain in the immutable source completion artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from open_r1.trainer.rewards.reward_funcs.analysis_reward_v3 import (
    grounded_analysis_reward_v3,
)


class ReplayError(ValueError):
    """Raised when replay inputs, resumability state, or gates are invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def canonical_evidence(value: str) -> str:
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ReplayError("provided_data is not JSON") from exc
    _require(isinstance(payload, dict), "provided_data must be a JSON object")
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_evidence_from_rendered_prompt(prompt: str) -> str:
    start = str(prompt).find("{")
    end = str(prompt).rfind("}")
    _require(start >= 0 and end >= start, "completion prompt has no JSON fact card")
    return canonical_evidence(str(prompt)[start : end + 1])


def evidence_signature(value: str) -> tuple[str, ...]:
    """Return the stable evidence-ID binding preserved by tokenizer decode.

    Rank-0 completion logging decodes prompt token IDs.  The DeepSeek/Llama
    tokenizer can remove spaces inside JSON string values while doing that,
    so replay must not treat the decoded JSON as the authoritative evidence.
    Evidence IDs are opaque ASCII identifiers and survive the round trip; use
    their ordered tuple to recover the exact sealed ``provided_data`` row.
    """

    payload = json.loads(canonical_evidence(value))
    evidence = payload.get("evidence")
    _require(isinstance(evidence, list) and evidence, "fact card evidence must be non-empty")
    identifiers: list[str] = []
    for index, row in enumerate(evidence):
        _require(isinstance(row, dict), f"fact card evidence[{index}] must be an object")
        identifier = row.get("evidence_id")
        _require(
            isinstance(identifier, str) and identifier.startswith("ev-"),
            f"fact card evidence[{index}] has no stable evidence_id",
        )
        identifiers.append(identifier)
    _require(len(set(identifiers)) == len(identifiers), "fact card evidence IDs are duplicated")
    return tuple(identifiers)


def load_dataset_bindings(dataset_dir: str | Path) -> dict[tuple[str, ...], tuple[str, str]]:
    root = Path(dataset_dir).resolve()
    _require(root.is_dir(), f"dataset directory does not exist: {root}")
    result: dict[tuple[str, ...], tuple[str, str]] = {}
    for name in ("train.jsonl", "eval.jsonl", "validation.jsonl", "test.jsonl"):
        path = root / name
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReplayError(f"{path}:{line_number}: invalid JSON") from exc
                _require(isinstance(row, dict), f"{path}:{line_number}: row must be an object")
                evidence = row.get("provided_data")
                meeting_date = row.get("meeting_date")
                _require(
                    isinstance(evidence, str) and isinstance(meeting_date, str),
                    f"{path}:{line_number}: missing provided_data/meeting_date",
                )
                canonical = canonical_evidence(evidence)
                key = evidence_signature(canonical)
                previous = result.setdefault(key, (canonical, meeting_date))
                _require(
                    previous == (canonical, meeting_date),
                    f"{path}:{line_number}: evidence-ID signature maps to multiple rows",
                )
    _require(result, f"no replay dataset rows found under {root}")
    return result


def _read_reward_records(path: Path, expected: int) -> list[dict[str, Any]]:
    _require(path.is_file(), f"missing reward record file: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayError(f"{path}:{line_number}: invalid JSON") from exc
            _require(
                isinstance(record, dict) and record.get("type") == "grounded_analysis_v3",
                f"{path}:{line_number}: wrong reward record type",
            )
            records.append(record)
    _require(
        len(records) == expected,
        f"{path} contains {len(records)} records; expected {expected}",
    )
    return records


def summarize_replay(groups: Iterable[list[dict[str, Any]]]) -> dict[str, Any]:
    materialized = list(groups)
    records = [record for group in materialized for record in group]
    _require(records, "replay produced no reward records")
    rewards = [float(record["reward"]) for record in records]
    finite = all(math.isfinite(value) for value in rewards)
    bounded = all(0.0 <= value <= 1.0 for value in rewards)
    point_counts = Counter(round(value, 6) for value in rewards)
    max_point_value, max_point_count = point_counts.most_common(1)[0]
    zero_variance_groups = sum(
        len(group) > 0
        and statistics.pstdev(float(record["reward"]) for record in group) <= 1e-12
        for group in materialized
    )
    nonfactual_penalty_violations = 0
    invalid_quote_count = 0
    judge_retry_count = 0
    for record in records:
        judge_retry_count += int(record.get("judge_attempts", 1) > 1)
        invalid_quote_count += len(record.get("invalid_violations", []))
        penalties = record.get("penalties", {})
        expected_penalty = (
            0.20 * penalties.get("major_answer_error", 0)
            + 0.08 * penalties.get("minor_answer_error", 0)
            + 0.04 * penalties.get("major_think_error", 0)
            + 0.02 * penalties.get("minor_think_error", 0)
        )
        if abs(float(penalties.get("unbounded", 0.0)) - expected_penalty) > 1e-12:
            nonfactual_penalty_violations += 1
    summary = {
        "schema_version": 1,
        "groups": len(materialized),
        "records": len(records),
        "reward": {
            "mean": statistics.fmean(rewards),
            "median": statistics.median(rewards),
            "min": min(rewards),
            "max": max(rewards),
            "finite": finite,
            "bounded_0_1": bounded,
            "max_point_mass_value_rounded_6dp": max_point_value,
            "max_point_mass_fraction": max_point_count / len(rewards),
        },
        "zero_variance_group_fraction": zero_variance_groups / len(materialized),
        "nonfactual_penalty_violations": nonfactual_penalty_violations,
        "invalid_quote_fraction": invalid_quote_count / len(records),
        "judge_retry_fraction": judge_retry_count / len(records),
    }
    summary["gates"] = {
        "finite_and_bounded": finite and bounded,
        "max_point_mass_lt_0_30": summary["reward"]["max_point_mass_fraction"] < 0.30,
        "zero_variance_groups_le_0_05": summary["zero_variance_group_fraction"] <= 0.05,
        "nonfactual_penalty_violations_zero": nonfactual_penalty_violations == 0,
    }
    summary["passed"] = all(summary["gates"].values())
    return summary


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    completion_dir = Path(args.completion_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    _require(completion_dir.is_dir(), f"completion directory does not exist: {completion_dir}")
    files = sorted(completion_dir.glob("completions_[0-9][0-9][0-9][0-9][0-9].parquet"))
    _require(files, f"no completion parquet files found under {completion_dir}")
    if args.limit is not None:
        _require(args.limit > 0, "--limit must be positive")
        files = files[: args.limit]
    dataset_bindings = load_dataset_bindings(args.dataset_dir)

    manifest = {
        "schema_version": 1,
        "completion_dir": str(completion_dir),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "files": [path.name for path in files],
        "judge": {
            "url": args.judge_url,
            "model": args.judge_model,
            "tokenizer_path": str(Path(args.judge_tokenizer_path).resolve()),
            "max_model_len": args.judge_max_model_len,
            "max_completion_tokens": args.judge_max_completion_tokens,
        },
    }
    manifest_path = output_dir / "replay_manifest.json"
    if manifest_path.exists():
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(observed == manifest, "replay manifest disagrees with existing output")
    else:
        _atomic_json(manifest_path, manifest)

    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    groups: list[list[dict[str, Any]]] = []
    baseline_v2_rewards: list[float] = []
    for index, path in enumerate(files, start=1):
        frame = pd.read_parquet(path)
        _require(
            list(frame.columns) == [
                "step",
                "prompt",
                "completion",
                "grounded_analysis_reward_v2",
                "advantage",
            ],
            f"{path} completion schema drifted",
        )
        baseline_v2_rewards.extend(
            float(value) for value in frame["grounded_analysis_reward_v2"]
        )
        record_path = records_dir / f"{path.stem}.reward_v3.jsonl"
        if not record_path.exists():
            decoded_evidence = [
                extract_evidence_from_rendered_prompt(value) for value in frame["prompt"]
            ]
            evidence = []
            dates = []
            for value in decoded_evidence:
                signature = evidence_signature(value)
                _require(
                    signature in dataset_bindings,
                    f"{path}: evidence-ID signature is not in the sealed dataset",
                )
                sealed_evidence, meeting_date = dataset_bindings[signature]
                evidence.append(sealed_evidence)
                dates.append(meeting_date)
            completions = [[{"content": str(value)}] for value in frame["completion"]]
            rewards = grounded_analysis_reward_v3(
                completions,
                evidence,
                meeting_date=dates,
                save_path=str(record_path),
                url=args.judge_url,
                model=args.judge_model,
                tokenizer_path=args.judge_tokenizer_path,
                max_model_len=args.judge_max_model_len,
                max_completion_tokens=args.judge_max_completion_tokens,
                timeout=args.judge_timeout,
                max_retries=args.judge_max_retries,
                backoff_seconds=args.judge_backoff_seconds,
            )
            _require(len(rewards) == len(frame), f"{path}: reward cardinality mismatch")
        groups.append(_read_reward_records(record_path, len(frame)))
        print(f"replayed {index}/{len(files)} {path.name}", flush=True)

    summary = summarize_replay(groups)
    baseline_point_counts = Counter(round(value, 6) for value in baseline_v2_rewards)
    _, baseline_max_point_count = baseline_point_counts.most_common(1)[0]
    summary["baseline_v2_reward"] = {
        "mean": statistics.fmean(baseline_v2_rewards),
        "median": statistics.median(baseline_v2_rewards),
        "max_point_mass_fraction": baseline_max_point_count / len(baseline_v2_rewards),
    }
    summary.update(
        {
            "completion_dir": str(completion_dir),
            "dataset_dir": str(Path(args.dataset_dir).resolve()),
            "source_files": len(files),
        }
    )
    _atomic_json(output_dir / "summary.json", summary)
    if args.enforce_gates and not summary["passed"]:
        raise ReplayError("reward-v3 replay gates failed")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--judge-model", default="Qwen3.5-9B")
    parser.add_argument("--judge-tokenizer-path", default="models/Qwen3.5-9B")
    parser.add_argument("--judge-max-model-len", type=int, default=8192)
    parser.add_argument("--judge-max-completion-tokens", type=int, default=2048)
    parser.add_argument("--judge-timeout", type=int, default=180)
    parser.add_argument("--judge-max-retries", type=int, default=3)
    parser.add_argument("--judge-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--enforce-gates", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = run_replay(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
