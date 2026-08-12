"""Run one fail-closed DeepSeek-high chk2 reward canary group.

The canary reads one pinned row from an existing ``analysis_grpo`` test split,
constructs four deterministic completions in memory, and invokes the isolated
DeepSeek Responses reward exactly once.  It never writes the source evidence or
candidate text to its receipt.  ``--dry-run`` performs every local binding and
contract check without importing the reward implementation, reading an API key,
creating the output directory, or making a network request.

Example::

    conda run -n fomc_trainer python -m \
      jobs.retrain_v2.canary_chk2_deepseek_high_reward \
      --output-dir output/evaluation/retrain_v2/chk2_deepseek_high_canary_001 \
      --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST_JSONL = (
    REPO_ROOT
    / "dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804"
    / "analysis_grpo/test.jsonl"
)
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "models/Qwen3.5-9B"
PINNED_SAMPLE_ID = "chk1-analysis-2023-07-26-10da987c674d440d"
EXPECTED_TEST_ROWS = 190
EXPECTED_TOPIC = "Federal Funds Rate"
MODEL = "deepseek-v4-flash"
URL = "https://api.deepseek.com"
API_KEY_ENV = "DEEPSEEK_API_KEY"
EFFORT = "high"
MAX_OUTPUT_TOKENS = 16_384
TIMEOUT_SECONDS = 420
MAX_RETRIES = 2
BACKOFF_SECONDS = 2.0
BOUNDARY = "\n</think>\n"
RECEIPT_SCHEMA = "chk2-deepseek-high-reward-canary-v1"
REWARD_RECORD_TYPE = "grounded_analysis_v3_deepseek_high"


class CanaryError(RuntimeError):
    """Raised when a canary input, output, or privacy contract fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryError(message)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
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


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _read_pinned_row(path: Path) -> tuple[dict[str, Any], str, str]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"test split is not a regular file: {resolved}")
    _require(not resolved.is_symlink(), f"test split must not be a symlink: {resolved}")

    selected: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    rows = 0
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            _require(bool(line.strip()), f"blank row at test split line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CanaryError(
                    f"invalid JSON at test split line {line_number}"
                ) from exc
            _require(isinstance(row, dict), f"test row {line_number} is not an object")
            _require(
                set(row) == {"meeting_date", "prompt", "provided_data", "sample_id"},
                f"test row {line_number} schema drifted",
            )
            sample_id = row.get("sample_id")
            _require(
                isinstance(sample_id, str) and sample_id,
                f"test row {line_number} has no sample_id",
            )
            _require(sample_id not in sample_ids, f"duplicate sample_id: {sample_id}")
            sample_ids.add(sample_id)
            rows += 1
            if sample_id == PINNED_SAMPLE_ID:
                selected.append(row)

    _require(rows == EXPECTED_TEST_ROWS, f"test row count drifted: {rows}")
    _require(len(selected) == 1, "pinned canary sample is missing or duplicated")
    row = selected[0]
    prompt = row["prompt"]
    provided_data = row["provided_data"]
    meeting_date = row["meeting_date"]
    _require(
        all(isinstance(value, str) and value for value in (prompt, provided_data, meeting_date)),
        "pinned row has an invalid text field",
    )
    _require(prompt.count(provided_data) == 1, "prompt/evidence binding drifted")

    try:
        evidence_payload = json.loads(provided_data)
    except json.JSONDecodeError as exc:
        raise CanaryError("pinned provided_data is invalid JSON") from exc
    _require(isinstance(evidence_payload, dict), "provided_data must be an object")
    _require(
        provided_data == _canonical_json(evidence_payload),
        "provided_data is not canonical JSON",
    )
    _require(evidence_payload.get("atomic_topic") == EXPECTED_TOPIC, "topic drifted")
    evidence = evidence_payload.get("evidence")
    _require(isinstance(evidence, list) and evidence, "evidence list is empty")

    observed = {
        (
            item.get("series_id"),
            item.get("observation_date"),
            item.get("value"),
            item.get("units"),
        )
        for item in evidence
        if isinstance(item, dict)
    }
    required_facts = {
        ("DFEDTARU", "2023-05-31", "5.25", "Percent"),
        ("DFEDTARU", "2023-06-30", "5.25", "Percent"),
        ("DFEDTARU", "2023-07-25", "5.25", "Percent"),
        ("DFF", "2023-07-24", "5.08", "Percent"),
    }
    _require(required_facts <= observed, "pinned evidence facts drifted")
    return row, _sha256_file(resolved), _sha256_text(_canonical_json(sorted(sample_ids)))


def _candidates() -> list[tuple[str, str]]:
    supported_reasoning = (
        "The target range upper limit is 5.25 percent on May 31, June 30, "
        "and July 25. The latest daily effective rate is 5.08 percent on July 24."
    )
    supported_answer = (
        "The upper limit of the federal funds target range held at 5.25 percent "
        "through July 25, while the effective federal funds rate was 5.08 percent "
        "on July 24."
    )
    return [
        (
            "supported",
            f"{supported_reasoning}{BOUNDARY}{supported_answer}",
        ),
        (
            "think_error",
            (
                "The target range upper limit is 7.25 percent on July 25. "
                "The latest daily effective rate is 5.08 percent on July 24."
                f"{BOUNDARY}{supported_answer}"
            ),
        ),
        (
            "answer_error",
            (
                f"{supported_reasoning}{BOUNDARY}The upper limit of the federal funds "
                "target range rose to 7.25 percent by July 25, while the effective "
                "federal funds rate was 5.08 percent on July 24."
            ),
        ),
        (
            "missing_boundary",
            (
                "The upper limit of the federal funds target range held at 5.25 "
                "percent through July 25, while the effective federal funds rate "
                "was 5.08 percent on July 24."
            ),
        ),
    ]


def _candidate_descriptors(candidates: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for label, text in candidates:
        boundary_count = text.count("</think>")
        descriptors.append(
            {
                "label": label,
                "sha256": _sha256_text(text),
                "chars": len(text),
                "closing_think_count": boundary_count,
                "answer_nonempty": bool(
                    boundary_count == 1 and text.split("</think>", 1)[1].strip()
                ),
            }
        )
    return descriptors


def _contract_receipt(
    *,
    test_jsonl: Path,
    output_dir: Path,
    row: Mapping[str, Any],
    test_sha256: str,
    test_sample_ids_sha256: str,
    candidates: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    descriptors = _candidate_descriptors(candidates)
    group_binding = {
        "sample_id": row["sample_id"],
        "meeting_date": row["meeting_date"],
        "provided_data_sha256": _sha256_text(str(row["provided_data"])),
        "candidate_sha256": [item["sha256"] for item in descriptors],
        "request_contract": {
            "api": "responses",
            "url": URL,
            "model": MODEL,
            "reasoning": {"effort": EFFORT},
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "group_size": 4,
        },
    }
    return {
        "schema_version": RECEIPT_SCHEMA,
        "status": "planned",
        "created_at_utc": _utc_now(),
        "source": {
            "test_jsonl": str(test_jsonl.expanduser().resolve()),
            "test_jsonl_sha256": test_sha256,
            "test_rows": EXPECTED_TEST_ROWS,
            "test_sample_ids_sha256": test_sample_ids_sha256,
            "sample_id": row["sample_id"],
            "meeting_date": row["meeting_date"],
            "prompt_sha256": _sha256_text(str(row["prompt"])),
            "provided_data_sha256": _sha256_text(str(row["provided_data"])),
            "contains_minutes_or_target": False,
        },
        "request": {
            "logical_calls": 1,
            "candidate_count": 4,
            "url": URL,
            "model": MODEL,
            "api": "responses",
            "reasoning_effort": EFFORT,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "timeout_seconds": TIMEOUT_SECONDS,
            "max_retries": MAX_RETRIES,
            "backoff_seconds": BACKOFF_SECONDS,
            "api_key_env": API_KEY_ENV,
            "api_key_present": None,
            "group_binding_sha256": _sha256_text(_canonical_json(group_binding)),
        },
        "candidates": descriptors,
        "expected_order": [
            "supported",
            "think_error",
            "answer_error",
            "missing_boundary",
        ],
        "output_dir": str(output_dir.expanduser().resolve()),
        "privacy": {
            "raw_prompt_persisted": False,
            "raw_evidence_persisted": False,
            "raw_candidates_persisted": False,
            "api_key_persisted": False,
        },
    }


_FORBIDDEN_RAW_KEYS = {
    "api_key",
    "candidate",
    "candidate_quote",
    "evidence",
    "explanation",
    "hidden_reasoning",
    "prompt",
    "provided_data",
    "raw_response",
    "request_body",
    "response_body",
}


def _assert_no_forbidden_raw_keys(value: Any, *, location: str = "reward log") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require(
                str(key).casefold() not in _FORBIDDEN_RAW_KEYS,
                f"{location} contains forbidden raw field: {key}",
            )
            _assert_no_forbidden_raw_keys(child, location=location)
    elif isinstance(value, list):
        for child in value:
            _assert_no_forbidden_raw_keys(child, location=location)


def _read_reward_log(
    path: Path,
    *,
    provided_data: str,
    candidates: Sequence[tuple[str, str]],
    api_key: str,
) -> tuple[list[dict[str, Any]], str]:
    _require(path.is_file(), "reward function did not create reward.jsonl")
    os.chmod(path, 0o600)
    raw = path.read_text(encoding="utf-8")
    _require(api_key not in raw, "reward log contains the API key")
    _require(provided_data not in raw, "reward log contains raw evidence")
    for _, candidate in candidates:
        _require(candidate not in raw, "reward log contains a raw candidate")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        _require(bool(line), f"blank reward log row at line {line_number}")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CanaryError(f"invalid reward JSON at line {line_number}") from exc
        _require(isinstance(record, dict), f"reward row {line_number} is not an object")
        _assert_no_forbidden_raw_keys(record)
        _require(
            record.get("type") == REWARD_RECORD_TYPE,
            f"reward row {line_number} has the wrong type",
        )
        records.append(record)
    _require(len(records) == 4, f"reward log cardinality drifted: {len(records)}")
    return records, _sha256_text(raw)


def _validate_provider_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(
        [record.get("candidate_id") for record in records]
        == [f"candidate_{index}" for index in range(4)],
        "reward candidate ordering drifted",
    )
    _require(
        all(record.get("cache_hit") is False for record in records),
        "live canary unexpectedly used a provider cache",
    )
    provider_rows = [record.get("provider") for record in records]
    _require(
        all(isinstance(provider, dict) for provider in provider_rows),
        "reward record has no provider metadata",
    )
    provider = dict(provider_rows[0])
    _require(
        all(candidate == provider for candidate in provider_rows[1:]),
        "four candidate records disagree on provider metadata",
    )
    _require(provider.get("name") == "deepseek", "provider name drifted")
    _require(provider.get("api") == "responses", "provider API drifted")
    _require(provider.get("returned_model") == MODEL, "returned model drifted")
    _require(
        provider.get("reasoning_effort") == EFFORT,
        "provider reasoning effort drifted",
    )
    _require(
        provider.get("max_output_tokens") == MAX_OUTPUT_TOKENS,
        "provider output budget drifted",
    )
    attempts = provider.get("attempts")
    _require(
        isinstance(attempts, int) and not isinstance(attempts, bool) and 1 <= attempts <= 2,
        "provider attempt count is invalid",
    )
    usage = provider.get("usage")
    _require(isinstance(usage, dict), "provider usage is missing")
    reasoning_tokens = usage.get("reasoning_tokens")
    _require(
        isinstance(reasoning_tokens, int)
        and not isinstance(reasoning_tokens, bool)
        and reasoning_tokens > 0,
        "high-effort response reported no reasoning tokens",
    )
    hidden_sha = provider.get("hidden_reasoning_sha256")
    visible_sha = provider.get("visible_response_sha256")
    _require(
        isinstance(hidden_sha, str) and len(hidden_sha) == 64,
        "hidden reasoning hash is missing",
    )
    _require(
        isinstance(visible_sha, str) and len(visible_sha) == 64,
        "visible response hash is missing",
    )
    attempt_metrics = provider.get("attempt_metrics")
    _require(
        isinstance(attempt_metrics, list)
        and len(attempt_metrics) == attempts
        and attempt_metrics[-1].get("status") == "completed",
        "provider attempt metrics are incomplete",
    )
    return {
        "attempts": attempts,
        "input_tokens": usage.get("input_tokens"),
        "cached_input_tokens": usage.get("cached_input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": usage.get("total_tokens"),
        "provider_latency_seconds": provider.get("latency_seconds"),
        "visible_response_sha256": visible_sha,
        "hidden_reasoning_sha256": hidden_sha,
    }


def _scan_output_privacy(
    output_dir: Path,
    *,
    provided_data: str,
    candidates: Sequence[tuple[str, str]],
    api_key: str,
) -> list[str]:
    audited: list[str] = []
    forbidden_values = [api_key, provided_data, *[text for _, text in candidates]]
    for path in sorted(output_dir.rglob("*")):
        _require(not path.is_symlink(), f"canary output contains a symlink: {path}")
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8")
        for value in forbidden_values:
            _require(value not in raw, f"canary artifact leaks protected text: {path}")
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise CanaryError(f"canary artifact is not JSON: {path}") from exc
            _assert_no_forbidden_raw_keys(
                value, location=f"{path.name}:{line_number}"
            )
            if raw.lstrip().startswith("{") and "\n{" not in raw:
                break
        audited.append(str(path.relative_to(output_dir)))
    return audited


def _run_real_canary(
    *,
    receipt: dict[str, Any],
    row: Mapping[str, Any],
    candidates: Sequence[tuple[str, str]],
    output_dir: Path,
) -> tuple[dict[str, Any], int]:
    api_key = str(os.environ.get(API_KEY_ENV) or "").strip()
    _require(api_key, f"missing credential in environment: {API_KEY_ENV}")
    receipt["request"]["api_key_present"] = True
    _require(DEFAULT_TOKENIZER_PATH.is_dir(), "local Qwen tokenizer path is missing")
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(output_dir, 0o700)
    reward_path = output_dir / "reward.jsonl"
    request_started: float | None = None

    try:
        from open_r1.trainer.rewards.reward_funcs.analysis_reward_v3_deepseek_high import (
            grounded_analysis_reward_v3_deepseek_high,
        )

        completions = [[{"content": text}] for _, text in candidates]
        evidence = [str(row["provided_data"])] * 4
        dates = [str(row["meeting_date"])] * 4
        request_started = time.perf_counter()
        rewards = grounded_analysis_reward_v3_deepseek_high(
            completions,
            evidence,
            meeting_date=dates,
            save_path=str(reward_path),
            url=URL,
            model=MODEL,
            tokenizer_path=str(DEFAULT_TOKENIZER_PATH),
            max_completion_tokens=MAX_OUTPUT_TOKENS,
            timeout=TIMEOUT_SECONDS,
            max_retries=MAX_RETRIES,
            backoff_seconds=BACKOFF_SECONDS,
            api_key_env=API_KEY_ENV,
        )
        logical_call_latency_seconds = time.perf_counter() - request_started
        _require(len(rewards) == 4, f"reward cardinality drifted: {len(rewards)}")
        numeric_rewards = [float(value) for value in rewards]
        _require(
            all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in numeric_rewards),
            "reward contains a non-finite or out-of-range value",
        )
        records, reward_log_sha256 = _read_reward_log(
            reward_path,
            provided_data=str(row["provided_data"]),
            candidates=candidates,
            api_key=api_key,
        )
        provider_summary = _validate_provider_records(records)
        audited_artifacts = _scan_output_privacy(
            output_dir,
            provided_data=str(row["provided_data"]),
            candidates=candidates,
            api_key=api_key,
        )
        ordering_passed = (
            numeric_rewards[0]
            > numeric_rewards[1]
            > numeric_rewards[2]
            > numeric_rewards[3]
        )
        missing_boundary_zero = abs(numeric_rewards[3]) <= 1e-12
        receipt.update(
            {
                "status": (
                    "passed" if ordering_passed and missing_boundary_zero else "failed"
                ),
                "result": {
                    "reward_function_invocations": 1,
                    "logical_call_latency_seconds": logical_call_latency_seconds,
                    "judge_attempts": provider_summary["attempts"],
                    "provider": provider_summary,
                    "reward_records": len(records),
                    "reward_log_sha256": reward_log_sha256,
                    "privacy_audited_artifacts": audited_artifacts,
                    "labels": [label for label, _ in candidates],
                    "rewards": numeric_rewards,
                    "finite_and_bounded": True,
                    "strict_expected_order_passed": ordering_passed,
                    "missing_boundary_reward_is_zero": missing_boundary_zero,
                },
            }
        )
        return receipt, 0 if receipt["status"] == "passed" else 1
    except Exception as exc:  # provider and SDK exception types vary
        if reward_path.exists():
            os.chmod(reward_path, 0o600)
        receipt.update(
            {
                "status": "failed_closed",
                "error": {
                    "type": type(exc).__name__,
                    "message_persisted": False,
                    "logical_call_latency_seconds": (
                        time.perf_counter() - request_started
                        if request_started is not None
                        else None
                    ),
                },
            }
        )
        return receipt, 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-jsonl",
        type=Path,
        default=DEFAULT_TEST_JSONL,
        help="Existing immutable analysis_grpo test split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New, non-existing directory for the canary receipt and reward log.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print hashes/contracts without network or filesystem writes.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    try:
        _require(not output_dir.exists(), f"output directory already exists: {output_dir}")
        row, test_sha256, test_sample_ids_sha256 = _read_pinned_row(args.test_jsonl)
        candidates = _candidates()
        _require(len(candidates) == 4, "canary must contain exactly four candidates")
        descriptors = _candidate_descriptors(candidates)
        _require(
            [item["closing_think_count"] for item in descriptors] == [1, 1, 1, 0],
            "candidate boundary contract drifted",
        )
        receipt = _contract_receipt(
            test_jsonl=args.test_jsonl,
            output_dir=output_dir,
            row=row,
            test_sha256=test_sha256,
            test_sample_ids_sha256=test_sample_ids_sha256,
            candidates=candidates,
        )
        if args.dry_run:
            receipt["status"] = "dry_run"
            receipt["request"]["executed"] = False
            print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        receipt["request"]["executed"] = True
        receipt, exit_code = _run_real_canary(
            receipt=receipt,
            row=row,
            candidates=candidates,
            output_dir=output_dir,
        )
        _atomic_json(output_dir / "canary_receipt.json", receipt)
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        return exit_code
    except (CanaryError, OSError, UnicodeError, ValueError) as exc:
        # Local preflight errors are safe to show because no provider payload is involved.
        print(
            json.dumps(
                {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
