"""Benchmark DeepSeek low-effort judging without changing a training reward.

The benchmark sends the same strict four-candidate v3 rubric used by the
DeepSeek-high reward, but fixes the Responses API reasoning effort to ``low``.
It evaluates one real step-1 GRPO completion group plus two deterministic
diagnostic groups.  Its resumable artifacts contain hashes, rubric results,
reward components, provider usage, and latency only; raw evidence, candidates,
provider output, hidden reasoning, and credentials are never persisted.

This is deliberately a standalone evaluation job.  It is not imported by the
reward registry and cannot alter a trainer configuration or a dataset release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import pandas as pd

from jobs.retrain_v2.replay_chk2_reward_v3 import (
    evidence_signature,
    extract_evidence_from_rendered_prompt,
    load_dataset_bindings,
)
from jobs.retrain_v2.canary_chk2_deepseek_high_reward import (
    _candidates as _high_canary_candidates,
)
from open_r1.structured_response import parse_structured_response
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v2 import (
    _complete_candidate_response,
    _concision_score,
    get_judge_tokenizer,
    validate_judge_evidence,
)
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v3 import (
    _JUDGE_SYSTEM_PROMPT_V3,
    _RUBRIC_KEYS,
    _judge_schema_v3,
    _judge_score,
    _penalty_breakdown,
    _validate_evaluation_v3,
    _validated_violations,
    _violation_audit,
    reasoning_efficiency,
)
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v3_deepseek_high import (
    numeric_grounding_v3_deepseek_high,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "output/training/retrain_v2"
    / "chk2_clean_v2_cp200_deepseek_high_totalsl_v1_20260810"
    / "adapters/chk2"
)
DEFAULT_COMPLETION_PARQUET = (
    DEFAULT_RUN_ROOT / "completions/completions_00001.parquet"
)
DEFAULT_HIGH_REWARD_LOG = DEFAULT_RUN_ROOT / "reward.jsonl"
DEFAULT_HIGH_CANARY_REWARD_LOG = (
    REPO_ROOT
    / "output/evaluation/retrain_v2/chk2_deepseek_high_canary_20260810_01"
    / "reward.jsonl"
)
DEFAULT_DATASET_DIR = (
    REPO_ROOT
    / "dataset/processed/retrain_v2"
    / "analysis_grpo_full_v7_totalsl_billions_v1_20260810"
)
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "models/Qwen3.5-9B"

MODEL = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 16_384
TIMEOUT_SECONDS = 420.0
MAX_ATTEMPTS = 2
BACKOFF_SECONDS = 2.0
GROUP_SIZE = 4
PERMUTATION = (2, 0, 3, 1)
API_KEY_ENV = "DEEPSEEK_API_KEY"
REQUEST_MAX_BYTES = 512 * 1024

SCHEMA_VERSION = "chk2-deepseek-low-judge-benchmark-v1"
GROUP_CACHE_SCHEMA = "chk2-deepseek-low-judge-group-cache-v1"
WIRE_SCHEMA_VERSION = "grounded-analysis-v3-deepseek-low-request-v1"
CANDIDATE_IDS = tuple(f"candidate_{index}" for index in range(GROUP_SIZE))

_BATCH_SYSTEM_PROMPT = f"""{_JUDGE_SYSTEM_PROMPT_V3}

The input contains one shared evidence object and exactly four candidates named
candidate_0 through candidate_3. Evaluate every candidate independently against
the shared evidence. Never compare candidates with each other, transfer a claim
from one candidate to another, or use one candidate as evidence for another.
Return exactly the four named evaluation objects required by the JSON schema."""


class LowJudgeBenchmarkError(RuntimeError):
    """Local contract, source binding, or cached-result validation failed."""


class LowJudgeProviderError(RuntimeError):
    """The provider failed without a valid, privacy-safe benchmark result."""


class _RetryableResponseError(RuntimeError):
    """A provider response can be retried safely."""


class _PermanentResponseError(RuntimeError):
    """A provider or provenance failure must not be retried."""


@dataclass(frozen=True)
class BenchmarkGroup:
    group_id: str
    base_group_id: str
    order_variant: str
    source_kind: str
    evidence: str
    meeting_date: str
    completions: tuple[str, ...]
    labels: tuple[str, ...]
    high_baseline: tuple[Mapping[str, Any] | None, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LowJudgeBenchmarkError(message)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.is_symlink():
        raise LowJudgeBenchmarkError(f"refusing symlinked output: {path}")
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def _batch_json_schema() -> dict[str, Any]:
    evaluation_schema = _judge_schema_v3()["schema"]
    return {
        "type": "json_schema",
        "name": "fomc_analysis_rubric_v3_deepseek_low_batch4",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                candidate_id: json.loads(_canonical_json(evaluation_schema))
                for candidate_id in CANDIDATE_IDS
            },
            "required": list(CANDIDATE_IDS),
            "additionalProperties": False,
        },
    }


def build_low_batch_request(
    *, evidence: str, candidates: Sequence[str]
) -> dict[str, Any]:
    """Build the locked low-effort four-candidate Responses request."""

    _require(len(candidates) == GROUP_SIZE, "request must contain four candidates")
    _require(
        all(isinstance(candidate, str) for candidate in candidates),
        "request candidates must be strings",
    )
    evidence_text = validate_judge_evidence(evidence)
    payload = _canonical_json(
        {
            "schema_version": WIRE_SCHEMA_VERSION,
            "evidence": evidence_text,
            "candidates": dict(zip(CANDIDATE_IDS, candidates, strict=True)),
        }
    )
    _require(
        len(payload.encode("utf-8")) <= REQUEST_MAX_BYTES,
        "four-candidate request exceeds 512 KiB",
    )
    return {
        "model": MODEL,
        "instructions": _BATCH_SYSTEM_PROMPT,
        "input": payload,
        "reasoning": {"effort": REASONING_EFFORT},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "text": {"format": _batch_json_schema()},
    }


def _candidate_for_judge(completion: str) -> str:
    parsed = parse_structured_response(completion)
    return _complete_candidate_response(parsed, completion)


def _group_binding(group: BenchmarkGroup) -> dict[str, Any]:
    judge_candidates = [_candidate_for_judge(value) for value in group.completions]
    core = {
        "schema_version": SCHEMA_VERSION,
        "group_id": group.group_id,
        "base_group_id": group.base_group_id,
        "order_variant": group.order_variant,
        "source_kind": group.source_kind,
        "evidence_sha256": _sha256_text(group.evidence),
        "completion_sha256": [_sha256_text(value) for value in group.completions],
        "candidate_sha256": [_sha256_text(value) for value in judge_candidates],
        "labels": list(group.labels),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "schema_sha256": _sha256_text(_canonical_json(_batch_json_schema())),
        "instructions_sha256": _sha256_text(_BATCH_SYSTEM_PROMPT),
    }
    return {**core, "binding_sha256": _sha256_text(_canonical_json(core))}


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing regular file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            _require(bool(line.strip()), f"blank JSONL row: {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LowJudgeBenchmarkError(
                    f"invalid JSONL row: {path}:{line_number}"
                ) from exc
            _require(isinstance(row, dict), f"JSONL row is not an object: {path}")
            rows.append(row)
    return rows


def _load_high_baselines(
    *,
    evidence: str,
    completions: Sequence[str],
    reward_log: Path,
    expected_rewards: Sequence[float] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    evidence_sha = _sha256_text(evidence)
    by_binding: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_json_lines(reward_log.expanduser().resolve()):
        if row.get("type") != "grounded_analysis_v3_deepseek_high":
            continue
        key = (str(row.get("evidence_sha256")), str(row.get("candidate_sha256")))
        _require(key not in by_binding, "high reward log has a duplicate binding")
        by_binding[key] = row
    if expected_rewards is not None:
        _require(
            len(expected_rewards) == len(completions),
            "high baseline expected-reward count drifted",
        )
    result: list[Mapping[str, Any]] = []
    for index, completion in enumerate(completions):
        candidate_sha = _sha256_text(_candidate_for_judge(completion))
        row = by_binding.get((evidence_sha, candidate_sha))
        _require(row is not None, f"high reward log is missing candidate {index}")
        reward = float(row.get("reward"))
        if expected_rewards is not None:
            _require(
                abs(reward - float(expected_rewards[index])) <= 5e-6,
                f"parquet/high reward mismatch for candidate {index}",
            )
        judge = row.get("judge")
        penalties = row.get("penalties")
        _require(
            isinstance(judge, dict) and set(judge) == set(_RUBRIC_KEYS),
            "high reward rubric is invalid",
        )
        _require(isinstance(penalties, dict), "high reward penalties are invalid")
        result.append(
            {
                "reward": reward,
                "judge": {key: int(judge[key]) for key in _RUBRIC_KEYS},
                "penalties": {
                    key: penalties[key]
                    for key in (
                        "major_answer_error",
                        "minor_answer_error",
                        "major_think_error",
                        "minor_think_error",
                        "applied",
                        "target_leakage",
                    )
                },
            }
        )
    return tuple(result)


def _load_real_group(
    *, completion_parquet: Path, dataset_dir: Path, high_reward_log: Path
) -> BenchmarkGroup:
    parquet = completion_parquet.expanduser().resolve()
    dataset = dataset_dir.expanduser().resolve()
    high_log = high_reward_log.expanduser().resolve()
    _require(parquet.is_file() and not parquet.is_symlink(), "completion parquet is missing")
    frame = pd.read_parquet(parquet)
    required = {
        "step",
        "prompt",
        "completion",
        "grounded_analysis_reward_v3_deepseek_high",
    }
    _require(required <= set(frame.columns), "completion parquet schema drifted")
    _require(len(frame) == GROUP_SIZE, "completion parquet must contain exactly four rows")
    _require(len(set(int(value) for value in frame["step"])) == 1, "parquet mixes steps")
    prompts = [str(value) for value in frame["prompt"]]
    _require(len(set(prompts)) == 1, "parquet mixes prompt groups")

    decoded = extract_evidence_from_rendered_prompt(prompts[0])
    signature = evidence_signature(decoded)
    bindings = load_dataset_bindings(dataset)
    _require(signature in bindings, "completion evidence is not in the sealed dataset")
    evidence, meeting_date = bindings[signature]
    completions = tuple(str(value) for value in frame["completion"])
    baseline_rewards = [
        float(value) for value in frame["grounded_analysis_reward_v3_deepseek_high"]
    ]
    _require(
        all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in baseline_rewards),
        "high reward baseline is invalid",
    )

    evidence_sha = _sha256_text(evidence)
    high_rows = _read_json_lines(high_log)
    by_binding: dict[tuple[str, str], dict[str, Any]] = {}
    for row in high_rows:
        if row.get("type") != "grounded_analysis_v3_deepseek_high":
            continue
        key = (str(row.get("evidence_sha256")), str(row.get("candidate_sha256")))
        _require(key not in by_binding, "high reward log has a duplicate candidate binding")
        by_binding[key] = row

    baselines: list[Mapping[str, Any]] = []
    for index, (completion, expected_reward) in enumerate(
        zip(completions, baseline_rewards, strict=True)
    ):
        candidate_sha = _sha256_text(_candidate_for_judge(completion))
        row = by_binding.get((evidence_sha, candidate_sha))
        _require(row is not None, f"high reward log is missing real candidate {index}")
        observed_reward = float(row.get("reward"))
        _require(
            abs(observed_reward - expected_reward) <= 5e-6,
            f"parquet/high reward mismatch for real candidate {index}",
        )
        judge = row.get("judge")
        penalties = row.get("penalties")
        _require(
            isinstance(judge, dict) and set(judge) == set(_RUBRIC_KEYS),
            "high reward rubric is invalid",
        )
        _require(isinstance(penalties, dict), "high reward penalties are invalid")
        baselines.append(
            {
                "reward": observed_reward,
                "judge": {key: int(judge[key]) for key in _RUBRIC_KEYS},
                "penalties": {
                    key: penalties[key]
                    for key in (
                        "major_answer_error",
                        "minor_answer_error",
                        "major_think_error",
                        "minor_think_error",
                        "applied",
                        "target_leakage",
                    )
                },
            }
        )

    step = int(frame["step"].iloc[0])
    return BenchmarkGroup(
        group_id=f"g2_real_step_{step:05d}",
        base_group_id=f"g2_real_step_{step:05d}",
        order_variant="base",
        source_kind="real_step1_completion",
        evidence=evidence,
        meeting_date=meeting_date,
        completions=completions,
        labels=tuple(f"real_candidate_{index}" for index in range(GROUP_SIZE)),
        high_baseline=tuple(baselines),
    )


def _sealed_evidence_row(
    dataset_dir: Path,
    *,
    split: str,
    line_number: int,
    expected_topic: str,
) -> str:
    path = dataset_dir.expanduser().resolve() / split
    _require(path.is_file() and not path.is_symlink(), f"sealed split is missing: {split}")
    lines = path.read_text(encoding="utf-8").splitlines()
    _require(1 <= line_number <= len(lines), f"sealed row is missing: {split}:{line_number}")
    try:
        row = json.loads(lines[line_number - 1])
    except json.JSONDecodeError as exc:
        raise LowJudgeBenchmarkError(f"invalid sealed row: {split}:{line_number}") from exc
    _require(isinstance(row, dict), f"sealed row is not an object: {split}:{line_number}")
    evidence = row.get("provided_data")
    _require(isinstance(evidence, str), f"sealed row has no evidence: {split}:{line_number}")
    try:
        payload = json.loads(evidence)
    except json.JSONDecodeError as exc:
        raise LowJudgeBenchmarkError(f"sealed evidence is invalid: {split}:{line_number}") from exc
    _require(payload.get("atomic_topic") == expected_topic, f"sealed topic drifted: {split}:{line_number}")
    canonical = _canonical_json(payload)
    _require(canonical == evidence, f"sealed evidence is not canonical: {split}:{line_number}")
    validate_judge_evidence(canonical)
    return canonical


def _diagnostic_groups(
    dataset_dir: Path,
    high_canary_reward_log: Path,
) -> tuple[BenchmarkGroup, BenchmarkGroup, BenchmarkGroup, BenchmarkGroup]:
    # These positions are immutable bindings in the TOTALSL-overlay release.
    # G1/G5 deliberately share the existing FFR canary evidence.
    funds_evidence = _sealed_evidence_row(
        dataset_dir,
        split="test.jsonl",
        line_number=2,
        expected_topic="Federal Funds Rate",
    )
    canary_pairs = _high_canary_candidates()
    canary_labels = tuple(label for label, _ in canary_pairs)
    canary_completions = tuple(completion for _, completion in canary_pairs)
    canary_baselines = _load_high_baselines(
        evidence=funds_evidence,
        completions=canary_completions,
        reward_log=high_canary_reward_log,
    )
    supported_reasoning = parse_structured_response(canary_completions[0]).reasoning
    supported_answer = parse_structured_response(canary_completions[0]).answer
    localization = BenchmarkGroup(
        group_id="g1_ffr_canary",
        base_group_id="g1_ffr_canary",
        order_variant="base",
        source_kind="deterministic_diagnostic",
        evidence=funds_evidence,
        meeting_date="",
        completions=canary_completions,
        labels=canary_labels,
        high_baseline=canary_baselines,
    )

    totalsl_evidence = _sealed_evidence_row(
        dataset_dir,
        split="train.jsonl",
        line_number=19,
        expected_topic="Bank Credit to Private Sector",
    )
    totalsl_reasoning = (
        "Total consumer credit declined from 2518.4 billion dollars in April to "
        "2513.0 billion in May and 2502.7 billion in June."
    )
    totalsl = BenchmarkGroup(
        group_id="g3_totalsl_scale",
        base_group_id="g3_totalsl_scale",
        order_variant="base",
        source_kind="deterministic_diagnostic",
        evidence=totalsl_evidence,
        meeting_date="",
        completions=(
            (
                f"{totalsl_reasoning}\n</think>\nTotal consumer credit was "
                "2502.7 billion dollars in June."
            ),
            (
                f"{totalsl_reasoning}\n</think>\nTotal consumer credit was "
                "approximately 2.5027 trillion dollars in June."
            ),
            (
                "The evidence reports 2502.7 billion dollars in June."
                "\n</think>\nTotal consumer credit was 2502.7 million dollars."
            ),
            (
                "The evidence reports 2502.7 billion dollars in June."
                "\n</think>\nTotal consumer credit was 2.5027 billion dollars."
            ),
        ),
        labels=(
            "totalsl_exact_billions",
            "totalsl_equivalent_trillions",
            "totalsl_wrong_millions",
            "totalsl_wrong_billions",
        ),
        high_baseline=(None, None, None, None),
    )

    unemployment_evidence = _sealed_evidence_row(
        dataset_dir,
        split="train.jsonl",
        line_number=65,
        expected_topic="Unemployment Rate",
    )
    unemployment_reasoning = (
        "The unemployment rate was 9.6 percent in September and October before "
        "rising to 9.8 percent in November."
    )
    unemployment_answer = (
        "Unemployment edged higher in the latest observation, reaching 9.8 percent in November."
    )
    unemployment = BenchmarkGroup(
        group_id="g4_unemployment_causal_meta",
        base_group_id="g4_unemployment_causal_meta",
        order_variant="base",
        source_kind="deterministic_diagnostic",
        evidence=unemployment_evidence,
        meeting_date="",
        completions=(
            f"{unemployment_reasoning}\n</think>\n{unemployment_answer}",
            (
                f"{unemployment_reasoning} The increase was caused by restrictive "
                f"monetary policy.\n</think>\n{unemployment_answer}"
            ),
            (
                "The provided data fact card shows the series values.\n</think>\n"
                "Based on the provided data, unemployment edged higher to 9.8 percent."
            ),
            (
                f"{unemployment_reasoning}\n</think>\nUnemployment reached 8.8 percent "
                "in November."
            ),
        ),
        labels=(
            "unemployment_supported",
            "unemployment_unsupported_cause",
            "unemployment_meta_style",
            "unemployment_wrong_number",
        ),
        high_baseline=(None, None, None, None),
    )

    leakage = BenchmarkGroup(
        group_id="g5_ffr_target_leakage",
        base_group_id="g5_ffr_target_leakage",
        order_variant="base",
        source_kind="deterministic_diagnostic",
        evidence=funds_evidence,
        meeting_date="",
        completions=(
            f"{supported_reasoning}\n</think>\n{supported_answer}",
            (
                "The target meeting Minutes show that the Committee voted to cut."
                f"\n</think>\n{supported_answer}"
            ),
            (
                f"{supported_reasoning}\n</think>\nThe target meeting Minutes show "
                "that the Committee voted unanimously to cut rates."
            ),
            (
                f"{supported_reasoning}\n</think>\nThe supplied rates alone do not "
                "establish the target meeting decision."
            ),
        ),
        labels=(
            "ffr_supported",
            "ffr_think_target_leakage",
            "ffr_answer_target_leakage",
            "ffr_no_leakage_caveat",
        ),
        high_baseline=(None, None, None, None),
    )
    return localization, totalsl, unemployment, leakage


def _ordered_variants(group: BenchmarkGroup) -> tuple[BenchmarkGroup, BenchmarkGroup]:
    def materialize(name: str, order: Sequence[int]) -> BenchmarkGroup:
        _require(sorted(order) == list(range(GROUP_SIZE)), "candidate permutation is invalid")
        return BenchmarkGroup(
            group_id=f"{group.base_group_id}_{name}",
            base_group_id=group.base_group_id,
            order_variant=name,
            source_kind=group.source_kind,
            evidence=group.evidence,
            meeting_date=group.meeting_date,
            completions=tuple(group.completions[index] for index in order),
            labels=tuple(group.labels[index] for index in order),
            high_baseline=tuple(group.high_baseline[index] for index in order),
        )

    return (
        materialize("original", tuple(range(GROUP_SIZE))),
        materialize("permuted_2031", PERMUTATION),
    )


def load_benchmark_groups(
    *,
    completion_parquet: Path,
    dataset_dir: Path,
    high_reward_log: Path,
    high_canary_reward_log: Path = DEFAULT_HIGH_CANARY_REWARD_LOG,
) -> list[BenchmarkGroup]:
    real = _load_real_group(
        completion_parquet=completion_parquet,
        dataset_dir=dataset_dir,
        high_reward_log=high_reward_log,
    )
    diagnostics = _diagnostic_groups(dataset_dir, high_canary_reward_log)
    base_groups = (diagnostics[0], real, *diagnostics[1:])
    return [variant for group in base_groups for variant in _ordered_variants(group)]


def _extract_response_text(response: Any) -> tuple[str, str]:
    visible = str(_field(response, "output_text", "") or "")
    hidden_parts: list[str] = []
    visible_parts: list[str] = []
    for item in _field(response, "output", []) or []:
        item_type = _field(item, "type")
        for part in _field(item, "content", []) or []:
            text = _field(part, "text")
            if not isinstance(text, str) or not text:
                continue
            part_type = _field(part, "type")
            if item_type == "reasoning" and part_type == "reasoning_text":
                hidden_parts.append(text)
            elif item_type == "message" and part_type == "output_text":
                visible_parts.append(text)
    reconstructed = "".join(visible_parts)
    if visible and reconstructed and visible != reconstructed:
        raise _RetryableResponseError("visible response representations disagree")
    return visible or reconstructed, "".join(hidden_parts)


def _usage_payload(value: Any) -> dict[str, int | None]:
    def integer(source: Any, name: str) -> int | None:
        raw = _field(source, name) if source is not None else None
        return int(raw) if isinstance(raw, int) and not isinstance(raw, bool) else None

    input_details = _field(value, "input_tokens_details") if value is not None else None
    output_details = _field(value, "output_tokens_details") if value is not None else None
    return {
        "input_tokens": integer(value, "input_tokens"),
        "cached_input_tokens": integer(input_details, "cached_tokens"),
        "output_tokens": integer(value, "output_tokens"),
        "reasoning_tokens": integer(output_details, "reasoning_tokens"),
        "total_tokens": integer(value, "total_tokens"),
    }


def _decode_batch_output(visible: str) -> list[dict[str, Any]]:
    def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    try:
        payload = json.loads(str(visible or "").strip(), object_pairs_hook=no_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise _RetryableResponseError("visible output is not unambiguous JSON") from exc
    if not isinstance(payload, dict) or set(payload) != set(CANDIDATE_IDS):
        raise _RetryableResponseError("batch output candidate mapping is invalid")
    result: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        try:
            result.append(_validate_evaluation_v3(payload[candidate_id]))
        except (KeyError, TypeError, ValueError) as exc:
            raise _RetryableResponseError(
                f"evaluation schema is invalid for {candidate_id}"
            ) from exc
    return result


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _RetryableResponseError):
        return True
    if isinstance(exc, _PermanentResponseError):
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


def _call_provider(
    *, client: Any, request: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    group_started = time.perf_counter()
    last_error = "provider_not_called"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        try:
            response = client.responses.create(**dict(request))
            if str(_field(response, "status", "")) != "completed":
                raise _RetryableResponseError("response was not completed")
            returned_model = str(_field(response, "model", "") or "")
            if returned_model != MODEL:
                raise _PermanentResponseError("returned model does not match request")
            visible, hidden = _extract_response_text(response)
            if not visible.strip():
                raise _RetryableResponseError("provider returned no visible output")
            usage = _usage_payload(_field(response, "usage"))
            if not isinstance(usage["reasoning_tokens"], int) or usage["reasoning_tokens"] <= 0:
                raise _RetryableResponseError("low-effort response has no reasoning usage")
            evaluations = _decode_batch_output(visible)
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "completed",
                    "latency_seconds": time.perf_counter() - started,
                    "http_status": 200,
                }
            )
            response_id = str(_field(response, "id", "") or "")
            return evaluations, {
                "attempts": attempt,
                "attempt_metrics": attempts,
                "latency_seconds": time.perf_counter() - group_started,
                "returned_model": returned_model,
                "response_id_sha256": _sha256_text(response_id) if response_id else None,
                "visible_response_sha256": _sha256_text(visible),
                "hidden_reasoning_sha256": _sha256_text(hidden) if hidden else None,
                "usage": usage,
            }
        except Exception as exc:  # provider SDK exception types vary
            retryable = _is_retryable(exc)
            last_error = type(exc).__name__
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "retryable_error" if retryable else "permanent_error",
                    "latency_seconds": time.perf_counter() - started,
                    "http_status": _status_code(exc),
                    "error_class": last_error,
                }
            )
            if not retryable:
                raise LowJudgeProviderError(
                    f"provider failed permanently: error_class={last_error}"
                ) from None
            if attempt < MAX_ATTEMPTS and BACKOFF_SECONDS:
                time.sleep(BACKOFF_SECONDS * (2 ** (attempt - 1)))
    raise LowJudgeProviderError(
        f"provider failed after {MAX_ATTEMPTS} attempts: error_class={last_error}"
    )


def _sanitize_evaluations(
    evaluations: Sequence[Mapping[str, Any]], candidates: Sequence[str]
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for candidate_id, evaluation, candidate in zip(
        CANDIDATE_IDS, evaluations, candidates, strict=True
    ):
        valid, invalid = _validated_violations(evaluation, candidate)
        sanitized.append(
            {
                "candidate_id": candidate_id,
                "rubric": {key: int(evaluation[key]) for key in _RUBRIC_KEYS},
                "validated_violations": [_violation_audit(value) for value in valid],
                "invalid_violations": [
                    _violation_audit(value["violation"], reason=value["reason"])
                    for value in invalid
                ],
                "penalties": _penalty_breakdown(valid),
            }
        )
    return sanitized


def _score_group(
    *,
    group: BenchmarkGroup,
    evaluations: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    judge_candidates = [_candidate_for_judge(value) for value in group.completions]
    sanitized = _sanitize_evaluations(evaluations, judge_candidates)
    rows: list[dict[str, Any]] = []
    for index, (completion, evaluation, baseline) in enumerate(
        zip(group.completions, sanitized, group.high_baseline, strict=True)
    ):
        parsed = parse_structured_response(completion)
        boundary_count = completion.count("</think>")
        contract_valid = bool(
            boundary_count >= 1 and parsed.is_well_formed and parsed.answer.strip()
        )
        reasoning = parsed.reasoning if contract_valid else completion.strip()
        answer = parsed.answer if contract_valid else ""
        answer_numeric, unsupported_answer = (
            numeric_grounding_v3_deepseek_high(answer, group.evidence)
            if contract_valid
            else (0.0, [])
        )
        think_numeric, unsupported_think = numeric_grounding_v3_deepseek_high(
            reasoning, group.evidence
        )
        efficiency, reasoning_tokens, repetition = reasoning_efficiency(
            reasoning, tokenizer
        )
        structure = float(
            contract_valid and bool(parsed.reasoning.strip()) and bool(parsed.answer.strip())
        )
        rubric = evaluation["rubric"]
        penalties = evaluation["penalties"]
        judge_score = _judge_score(rubric)
        base = (
            0.50 * judge_score
            + 0.25 * answer_numeric
            + 0.15 * structure
            + 0.05 * _concision_score(answer)
            + 0.05 * efficiency
            if contract_valid
            else 0.0
        )
        reward = (
            max(0.0, min(1.0, base - float(penalties["applied"])))
            if contract_valid and not penalties["target_leakage"]
            else 0.0
        )
        _require(math.isfinite(reward), "low judge produced a non-finite reward")
        high = dict(baseline) if baseline is not None else None
        rows.append(
            {
                "candidate_id": CANDIDATE_IDS[index],
                "label": group.labels[index],
                "completion_sha256": _sha256_text(completion),
                "candidate_sha256": _sha256_text(judge_candidates[index]),
                "rubric": rubric,
                "validated_violations": evaluation["validated_violations"],
                "invalid_violations": evaluation["invalid_violations"],
                "penalties": penalties,
                "components": {
                    "judge_score": judge_score,
                    "answer_numeric_score": answer_numeric,
                    "think_numeric_score_audit": think_numeric,
                    "structure_score": structure,
                    "answer_concision": _concision_score(answer),
                    "reasoning_efficiency": efficiency,
                },
                "contract": {
                    "well_formed": contract_valid,
                    "closing_think_count": boundary_count,
                    "answer_nonempty": bool(answer.strip()),
                },
                "reasoning_tokens": reasoning_tokens,
                "reasoning_repetition_rate": repetition,
                "unsupported_answer_number_count": len(unsupported_answer),
                "unsupported_think_number_count_audit": len(unsupported_think),
                "reward_before_penalty": base,
                "reward": reward,
                "high_baseline": high,
                "reward_delta_vs_high": (
                    reward - float(high["reward"]) if high is not None else None
                ),
            }
        )
    return rows


def _cache_core(
    *,
    group: BenchmarkGroup,
    binding: Mapping[str, Any],
    provider: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": GROUP_CACHE_SCHEMA,
        "group_id": group.group_id,
        "base_group_id": group.base_group_id,
        "order_variant": group.order_variant,
        "source_kind": group.source_kind,
        "binding": dict(binding),
        "provider": dict(provider),
        "candidates": [dict(value) for value in candidates],
    }


def _write_group_cache(path: Path, core: Mapping[str, Any]) -> None:
    payload = {**dict(core), "payload_sha256": _sha256_text(_canonical_json(core))}
    _atomic_json(path, payload)


def _load_group_cache(path: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"invalid group cache: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LowJudgeBenchmarkError(f"invalid group cache JSON: {path}") from exc
    _require(isinstance(payload, dict), "group cache must be an object")
    _require(payload.get("schema_version") == GROUP_CACHE_SCHEMA, "cache schema drifted")
    _require(payload.get("binding") == dict(binding), "cache binding drifted")
    core = {key: value for key, value in payload.items() if key != "payload_sha256"}
    _require(
        payload.get("payload_sha256") == _sha256_text(_canonical_json(core)),
        "cache payload checksum mismatch",
    )
    candidates = payload.get("candidates")
    _require(isinstance(candidates, list) and len(candidates) == GROUP_SIZE, "cache rows drifted")
    return payload


def _pairwise_order_agreement(left: Sequence[float], right: Sequence[float]) -> float:
    _require(len(left) == len(right), "ranking vectors differ in length")
    matches = 0
    comparisons = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            left_sign = (left[first] > left[second]) - (left[first] < left[second])
            right_sign = (right[first] > right[second]) - (right[first] < right[second])
            matches += int(left_sign == right_sign)
            comparisons += 1
    return matches / comparisons if comparisons else 1.0


def _summarize(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(groups) == 10, "benchmark must contain exactly ten request groups")
    _require(
        sum(len(group["candidates"]) for group in groups) == 40,
        "benchmark must contain exactly forty judgments",
    )
    bases = sorted({str(group["base_group_id"]) for group in groups})
    _require(len(bases) == 5, "benchmark must contain exactly five base groups")

    def variants(base: str) -> dict[str, Mapping[str, Any]]:
        result = {
            str(group["order_variant"]): group
            for group in groups
            if group["base_group_id"] == base
        }
        _require(
            set(result) == {"original", "permuted_2031"},
            f"request variants drifted for {base}",
        )
        original_labels = [row["label"] for row in result["original"]["candidates"]]
        permuted_labels = [row["label"] for row in result["permuted_2031"]["candidates"]]
        _require(
            permuted_labels == [original_labels[index] for index in PERMUTATION],
            f"candidate permutation drifted for {base}",
        )
        return result

    def original_rows(base: str) -> dict[str, Mapping[str, Any]]:
        return {
            str(row["label"]): row
            for row in variants(base)["original"]["candidates"]
        }

    real_group = next(
        group
        for group in groups
        if group["source_kind"] == "real_step1_completion"
        and group["order_variant"] == "original"
    )
    comparable = [
        candidate
        for group in groups
        if group["order_variant"] == "original"
        for candidate in group["candidates"]
        if candidate["high_baseline"] is not None
    ]
    _require(len(comparable) == 8, "high comparison must contain exactly eight candidates")
    low_rewards = [float(value["reward"]) for value in comparable]
    high_rewards = [float(value["high_baseline"]["reward"]) for value in comparable]
    deltas = [low - high for low, high in zip(low_rewards, high_rewards, strict=True)]
    rubric_cells = [
        int(candidate["rubric"][key])
        == int(candidate["high_baseline"]["judge"][key])
        for candidate in comparable
        for key in _RUBRIC_KEYS
    ]
    judge_deltas = [
        float(candidate["components"]["judge_score"])
        - _judge_score(candidate["high_baseline"]["judge"])
        for candidate in comparable
    ]

    totalsl_rows = original_rows("g3_totalsl_scale")
    unemployment_rows = original_rows("g4_unemployment_causal_meta")
    leakage_rows = original_rows("g5_ffr_target_leakage")
    known_checks = {
        "totalsl_equivalent_scale_supported": abs(
            float(totalsl_rows["totalsl_equivalent_trillions"]["penalties"]["applied"])
        )
        <= 1e-12,
        "totalsl_wrong_millions_detected": int(
            totalsl_rows["totalsl_wrong_millions"]["penalties"]["major_answer_error"]
        )
        >= 1,
        "totalsl_wrong_billions_detected": int(
            totalsl_rows["totalsl_wrong_billions"]["penalties"]["major_answer_error"]
        )
        >= 1,
        "unemployment_causal_error_detected": int(
            unemployment_rows["unemployment_unsupported_cause"]["penalties"][
                "major_think_error"
            ]
        )
        >= 1,
        "unemployment_meta_has_no_factual_penalty": abs(
            float(unemployment_rows["unemployment_meta_style"]["penalties"]["applied"])
        )
        <= 1e-12,
        "unemployment_wrong_number_detected": int(
            unemployment_rows["unemployment_wrong_number"]["penalties"][
                "major_answer_error"
            ]
        )
        >= 1,
        "think_target_leakage_detected": bool(
            leakage_rows["ffr_think_target_leakage"]["penalties"]["target_leakage"]
        ),
        "answer_target_leakage_detected": bool(
            leakage_rows["ffr_answer_target_leakage"]["penalties"]["target_leakage"]
        ),
        "target_leakage_rewards_zero": abs(
            float(leakage_rows["ffr_think_target_leakage"]["reward"])
        )
        <= 1e-12
        and abs(float(leakage_rows["ffr_answer_target_leakage"]["reward"])) <= 1e-12,
        "caveat_not_target_leakage": not bool(
            leakage_rows["ffr_no_leakage_caveat"]["penalties"]["target_leakage"]
        ),
    }

    permutation_reward_deltas: list[float] = []
    permutation_rubric_cells: list[bool] = []
    permutation_penalty_matches: list[bool] = []
    for base in bases:
        pair = variants(base)
        original = {row["label"]: row for row in pair["original"]["candidates"]}
        permuted = {row["label"]: row for row in pair["permuted_2031"]["candidates"]}
        _require(set(original) == set(permuted), f"permutation label set drifted for {base}")
        for label, first in original.items():
            second = permuted[label]
            permutation_reward_deltas.append(
                float(second["reward"]) - float(first["reward"])
            )
            permutation_rubric_cells.extend(
                int(second["rubric"][key]) == int(first["rubric"][key])
                for key in _RUBRIC_KEYS
            )
            permutation_penalty_matches.append(
                second["penalties"] == first["penalties"]
            )
    paired_rows = {
        base: {
            variant: {
                row["label"]: row for row in variants(base)[variant]["candidates"]
            }
            for variant in ("original", "permuted_2031")
        }
        for base in bases
    }
    missing_both_zero = all(
        abs(float(paired_rows["g1_ffr_canary"][variant]["missing_boundary"]["reward"]))
        <= 1e-12
        for variant in ("original", "permuted_2031")
    )
    major_localization_both = all(
        int(paired_rows["g1_ffr_canary"][variant]["think_error"]["penalties"]["major_think_error"])
        >= 1
        and int(paired_rows["g1_ffr_canary"][variant]["answer_error"]["penalties"]["major_answer_error"])
        >= 1
        for variant in ("original", "permuted_2031")
    )
    leakage_both_zero = all(
        bool(
            paired_rows["g5_ffr_target_leakage"][variant][label]["penalties"][
                "target_leakage"
            ]
        )
        and abs(
            float(paired_rows["g5_ffr_target_leakage"][variant][label]["reward"])
        )
        <= 1e-12
        for variant in ("original", "permuted_2031")
        for label in ("ffr_think_target_leakage", "ffr_answer_target_leakage")
    )
    providers = [group["provider"] for group in groups]
    provider_checks = {
        "all_groups_completed": all(
            provider.get("returned_model") == MODEL for provider in providers
        ),
        "all_have_reasoning_tokens": all(
            isinstance(provider.get("usage", {}).get("reasoning_tokens"), int)
            and provider["usage"]["reasoning_tokens"] > 0
            for provider in providers
        ),
        "all_within_attempt_budget": all(
            1 <= int(provider.get("attempts", 0)) <= MAX_ATTEMPTS for provider in providers
        ),
    }
    latencies = [float(provider["latency_seconds"]) for provider in providers]
    output_utilizations = [
        float(provider["usage"]["output_tokens"]) / MAX_OUTPUT_TOKENS
        for provider in providers
    ]

    def percentile(values: Sequence[float], fraction: float) -> float:
        ordered = sorted(float(value) for value in values)
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return ordered[index]

    all_rewards = [
        float(candidate["reward"])
        for group in groups
        for candidate in group["candidates"]
    ]
    group_std = {
        str(group["group_id"]): statistics.pstdev(
            float(candidate["reward"]) for candidate in group["candidates"]
        )
        for group in groups
    }
    real_rewards = [float(candidate["reward"]) for candidate in real_group["candidates"]]
    comparison_checks = {
        "high8_reward_mae_le_0_08": statistics.fmean(abs(value) for value in deltas) <= 0.08,
        "high8_reward_max_abs_delta_le_0_15": max(abs(value) for value in deltas) <= 0.15,
        "high8_weighted_judge_mae_le_0_10": statistics.fmean(
            abs(value) for value in judge_deltas
        )
        <= 0.10,
        "permutation_max_reward_delta_le_0_12": max(
            abs(value) for value in permutation_reward_deltas
        )
        <= 0.12,
        "retry_groups_le_1": sum(int(value["attempts"]) > 1 for value in providers) <= 1,
        "median_latency_le_45_seconds": statistics.median(latencies) <= 45.0,
        "p95_latency_le_90_seconds": percentile(latencies, 0.95) <= 90.0,
        "p95_output_utilization_le_0_50": percentile(output_utilizations, 0.95) <= 0.50,
        "max_output_utilization_le_0_75": max(output_utilizations) <= 0.75,
        "forty_rewards_finite_bounded": len(all_rewards) == 40
        and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in all_rewards),
        "missing_boundary_both_rounds_zero": missing_both_zero,
        "target_leakage_both_rounds_zero": leakage_both_zero,
        "major_localization_both_rounds": major_localization_both,
        "real_group_std_ge_0_05": statistics.pstdev(real_rewards) >= 0.05,
        "real_group_at_least_3_distinct": len({round(value, 6) for value in real_rewards}) >= 3,
        "every_base_round_nonzero_std": all(value > 1e-12 for value in group_std.values()),
    }
    gates = {**provider_checks, **comparison_checks, **known_checks}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all(gates.values()) else "failed_quality_gates",
        "created_at_utc": _utc_now(),
        "groups": len(groups),
        "candidates": len(all_rewards),
        "provider": {
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "logical_requests": len(groups),
            "attempts": [int(value["attempts"]) for value in providers],
            "latency_seconds": [float(value["latency_seconds"]) for value in providers],
            "reasoning_tokens": [value["usage"]["reasoning_tokens"] for value in providers],
            "output_tokens": [value["usage"]["output_tokens"] for value in providers],
        },
        "high8_comparison": {
            "candidates": len(comparable),
            "low_rewards": low_rewards,
            "high_rewards": high_rewards,
            "reward_deltas": deltas,
            "reward_mae": statistics.fmean(abs(value) for value in deltas),
            "reward_max_abs_delta": max(abs(value) for value in deltas),
            "pairwise_order_agreement": _pairwise_order_agreement(
                low_rewards, high_rewards
            ),
            "rubric_exact_cell_agreement": sum(rubric_cells) / len(rubric_cells),
            "weighted_judge_mae": statistics.fmean(
                abs(value) for value in judge_deltas
            ),
        },
        "permutation_consistency": {
            "fixed_permutation": list(PERMUTATION),
            "paired_candidates": len(permutation_reward_deltas),
            "reward_mae": statistics.fmean(
                abs(value) for value in permutation_reward_deltas
            ),
            "reward_max_abs_delta": max(
                abs(value) for value in permutation_reward_deltas
            ),
            "rubric_exact_cell_agreement": sum(permutation_rubric_cells)
            / len(permutation_rubric_cells),
            "penalty_exact_agreement": sum(permutation_penalty_matches)
            / len(permutation_penalty_matches),
        },
        "known_error_checks": known_checks,
        "group_reward_std": group_std,
        "latency": {
            "median_seconds": statistics.median(latencies),
            "p95_seconds": percentile(latencies, 0.95),
        },
        "output_utilization": {
            "p95": percentile(output_utilizations, 0.95),
            "maximum": max(output_utilizations),
        },
        "gates": gates,
        "suitable_for_chk2_trial": all(gates.values()),
        "privacy": {
            "raw_evidence_persisted": False,
            "raw_candidates_persisted": False,
            "provider_output_persisted": False,
            "hidden_reasoning_persisted": False,
            "api_key_persisted": False,
        },
    }


_FORBIDDEN_KEYS = {
    "api_key",
    "candidate",
    "candidate_quote",
    "completion",
    "evidence",
    "explanation",
    "hidden_reasoning",
    "input",
    "instructions",
    "prompt",
    "provided_data",
    "raw_response",
    "request_body",
    "response_body",
}


def _assert_sanitized(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require(str(key).casefold() not in _FORBIDDEN_KEYS, f"forbidden output key: {key}")
            _assert_sanitized(child)
    elif isinstance(value, list):
        for child in value:
            _assert_sanitized(child)


def _manifest(
    *,
    args: argparse.Namespace,
    groups: Sequence[BenchmarkGroup],
) -> dict[str, Any]:
    source_files = {
        "completion_parquet": Path(args.completion_parquet).expanduser().resolve(),
        "high_reward_log": Path(args.high_reward_log).expanduser().resolve(),
        "high_canary_reward_log": Path(args.high_canary_reward_log).expanduser().resolve(),
    }
    dataset = Path(args.dataset_dir).expanduser().resolve()
    dataset_files = [
        path for name in ("train.jsonl", "eval.jsonl", "validation.jsonl", "test.jsonl")
        if (path := dataset / name).is_file()
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            key: {"path": str(path), "sha256": _sha256_file(path)}
            for key, path in source_files.items()
        },
        "dataset": {
            "path": str(dataset),
            "files": [
                {"name": path.name, "sha256": _sha256_file(path)} for path in dataset_files
            ],
        },
        "request": {
            "api": "responses",
            "base_url": BASE_URL,
            "model": MODEL,
            "reasoning_effort": REASONING_EFFORT,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "timeout_seconds": TIMEOUT_SECONDS,
            "max_attempts": MAX_ATTEMPTS,
            "backoff_seconds": BACKOFF_SECONDS,
            "group_size": GROUP_SIZE,
            "schema_sha256": _sha256_text(_canonical_json(_batch_json_schema())),
            "instructions_sha256": _sha256_text(_BATCH_SYSTEM_PROMPT),
            "api_key_env": API_KEY_ENV,
        },
        "groups": [_group_binding(group) for group in groups],
        "privacy": {
            "raw_evidence_persisted": False,
            "raw_candidates_persisted": False,
            "provider_output_persisted": False,
            "hidden_reasoning_persisted": False,
            "api_key_persisted": False,
        },
    }


def _get_client(api_key: str) -> Any:
    try:
        import httpx
        from openai import OpenAI
    except ImportError as exc:
        raise LowJudgeBenchmarkError(
            "fomc_trainer must provide openai Responses API support"
        ) from exc
    parsed = urlparse(BASE_URL)
    _require(
        parsed.scheme == "https" and parsed.netloc == "api.deepseek.com" and not parsed.path,
        "DeepSeek base URL drifted",
    )
    timeout = httpx.Timeout(
        TIMEOUT_SECONDS,
        connect=15.0,
        read=TIMEOUT_SECONDS,
        write=30.0,
        pool=15.0,
    )
    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=timeout,
        max_retries=0,
    )


def run_benchmark(
    args: argparse.Namespace,
    *,
    client: Any | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    groups = load_benchmark_groups(
        completion_parquet=Path(args.completion_parquet),
        dataset_dir=Path(args.dataset_dir),
        high_reward_log=Path(args.high_reward_log),
        high_canary_reward_log=Path(args.high_canary_reward_log),
    )
    manifest = _manifest(args=args, groups=groups)
    _assert_sanitized(manifest)
    if bool(args.dry_run):
        return {**manifest, "status": "dry_run", "network_requests": 0}

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        _require(output_dir.is_dir() and not output_dir.is_symlink(), "invalid output directory")
    else:
        output_dir.mkdir(parents=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    manifest_path = output_dir / "benchmark_manifest.json"
    if manifest_path.exists():
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(observed == manifest, "existing benchmark manifest does not match inputs")
    else:
        _atomic_json(manifest_path, manifest)

    if client is None:
        for name in ("OPENAI_LOG", "HTTPX_LOG_LEVEL"):
            _require(
                os.environ.get(name, "").strip().casefold() not in {"debug", "trace"},
                f"unsafe provider logging is enabled: {name}",
            )
        api_key = str(os.environ.get(API_KEY_ENV) or "").strip()
        _require(bool(api_key), f"missing credential in {API_KEY_ENV}")
        client = _get_client(api_key)
    if tokenizer is None:
        tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
        _require(tokenizer_path.is_dir(), "judge tokenizer path is missing")
        tokenizer = get_judge_tokenizer(str(tokenizer_path))

    cache_dir = output_dir / "groups"
    cache_dir.mkdir(exist_ok=True, mode=0o700)
    materialized: list[dict[str, Any]] = []
    for group in groups:
        binding = _group_binding(group)
        cache_path = cache_dir / f"{group.group_id}.json"
        if cache_path.exists():
            cached = _load_group_cache(cache_path, binding)
            cached["cache_hit"] = True
            materialized.append(cached)
            continue
        judge_candidates = [_candidate_for_judge(value) for value in group.completions]
        request = build_low_batch_request(
            evidence=group.evidence, candidates=judge_candidates
        )
        evaluations, provider = _call_provider(client=client, request=request)
        scored = _score_group(group=group, evaluations=evaluations, tokenizer=tokenizer)
        core = _cache_core(
            group=group,
            binding=binding,
            provider=provider,
            candidates=scored,
        )
        _assert_sanitized(core)
        _write_group_cache(cache_path, core)
        materialized.append({**core, "payload_sha256": _sha256_text(_canonical_json(core)), "cache_hit": False})

    jsonl_rows: list[dict[str, Any]] = []
    for group in materialized:
        for candidate in group["candidates"]:
            jsonl_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "group_id": group["group_id"],
                    "source_kind": group["source_kind"],
                    "group_binding_sha256": group["binding"]["binding_sha256"],
                    "cache_hit": bool(group["cache_hit"]),
                    "provider": group["provider"],
                    **candidate,
                }
            )
    _assert_sanitized(jsonl_rows)
    _atomic_text(
        output_dir / "benchmark_records.jsonl",
        "".join(_canonical_json(row) + "\n" for row in jsonl_rows),
    )
    summary = _summarize(materialized)
    summary["output_dir"] = str(output_dir)
    summary["cache_hits"] = sum(bool(value["cache_hit"]) for value in materialized)
    _assert_sanitized(summary)
    _atomic_json(output_dir / "benchmark_summary.json", summary)

    protected_values = [
        *(group.evidence for group in groups),
        *(completion for group in groups for completion in group.completions),
    ]
    key = str(os.environ.get(API_KEY_ENV) or "")
    if key:
        protected_values.append(key)
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8")
        _require(
            all(not value or value not in raw for value in protected_values),
            f"privacy scan failed for {path.name}",
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completion-parquet", type=Path, default=DEFAULT_COMPLETION_PARQUET)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--high-reward-log", type=Path, default=DEFAULT_HIGH_REWARD_LOG)
    parser.add_argument(
        "--high-canary-reward-log",
        type=Path,
        default=DEFAULT_HIGH_CANARY_REWARD_LOG,
    )
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate sources/contracts and print hashes without writing or calling the API.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_benchmark(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except LowJudgeProviderError as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "error_class": type(exc).__name__,
            "error_message_persisted": False,
            "created_at_utc": _utc_now(),
        }
        output_dir = Path(args.output_dir).expanduser().resolve()
        if output_dir.is_dir() and not output_dir.is_symlink():
            _atomic_json(output_dir / "failure_receipt.json", failure)
        print(json.dumps(failure, sort_keys=True), file=os.sys.stderr)
        return 2
    except (LowJudgeBenchmarkError, OSError, UnicodeError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ),
            file=os.sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
