"""Audit changed chk1 clean-SFT rows with the local source-only Qwen judge.

The audit never stores candidate or evidence text.  It binds each verdict to
the immutable row hashes written by ``repair_chk1_sft_release`` and fails closed
on infrastructure errors, context overflow, or validated factual violations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from jobs.retrain_v2.judge_health import check_judge
from open_r1.structured_response import parse_structured_response
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v2 import (
    _complete_candidate_response,
    get_judge_tokenizer,
    render_judge_prompt_token_ids,
)
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v3 import (
    _judge_one_v3,
    _validated_violations,
    build_judge_request_v3,
)


SCHEMA_VERSION = "chk1-clean-sft-source-audit-v1"
SPLITS = ("train", "eval", "test")
BLOCKING_KINDS = {"factual", "numerical", "causal", "target_leakage"}


class CleanSftAuditError(RuntimeError):
    """Raised when audit inputs are not complete and hash-bound."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CleanSftAuditError(f"unable to read {path}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise CleanSftAuditError(f"{path}:{line_number}: blank JSONL row")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CleanSftAuditError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise CleanSftAuditError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def _required_text(row: Mapping[str, Any], key: str, *, label: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise CleanSftAuditError(f"{label}: {key} must be non-empty text")
    return value


def _reconstruct_judge_candidate(response: str, *, sample_id: str) -> str:
    """Mirror reward-v3's explicit section rendering before judge evaluation."""

    parsed = parse_structured_response(response)
    if not parsed.is_well_formed or not parsed.reasoning.strip() or not parsed.answer.strip():
        raise CleanSftAuditError(
            f"{sample_id}: candidate is not a complete reasoning+answer response"
        )
    return _complete_candidate_response(parsed, response)


def _load_changed_rows(
    *, release_dir: Path, repair_manifest: Path, expected_changed_rows: int
) -> list[dict[str, str]]:
    records = _read_jsonl(repair_manifest)
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    for number, record in enumerate(records, 1):
        label = f"repair_manifest:{number}"
        split = _required_text(record, "split", label=label)
        sample_id = _required_text(record, "sample_id", label=label)
        old_response_sha = _required_text(record, "old_response_sha256", label=label)
        new_response_sha = _required_text(record, "new_response_sha256", label=label)
        if len(old_response_sha) != 64 or len(new_response_sha) != 64:
            raise CleanSftAuditError(f"{label}: response hashes must be SHA-256 values")
        key = (split, new_response_sha)
        if key in index:
            raise CleanSftAuditError(f"{label}: duplicate split/new response hash")
        index[key] = {**record, "sample_id": sample_id}

    joined: list[dict[str, str]] = []
    observed_keys: set[tuple[str, str]] = set()
    for split in SPLITS:
        path = release_dir / "analysis_sft" / f"{split}.jsonl"
        for line_number, row in enumerate(_read_jsonl(path), 1):
            if set(row) != {"prompt", "response", "provided_data"}:
                raise CleanSftAuditError(f"{split}:{line_number}: invalid SFT row schema")
            response = _required_text(row, "response", label=f"{split}:{line_number}")
            response_sha = _sha256_text(response)
            key = (split, response_sha)
            record = index.get(key)
            if record is None:
                raise CleanSftAuditError(
                    f"{split}:{line_number}: no repair manifest hash binding"
                )
            observed_keys.add(key)
            if record["old_response_sha256"] == response_sha:
                continue
            joined.append(
                {
                    "sample_id": str(record["sample_id"]),
                    "split": split,
                    "response": response,
                    "response_sha256": response_sha,
                    "provided_data": _required_text(
                        row, "provided_data", label=f"{split}:{line_number}"
                    ),
                }
            )
    if observed_keys != set(index):
        raise CleanSftAuditError("repair manifest contains rows absent from the release")
    if len(joined) != expected_changed_rows:
        raise CleanSftAuditError(
            f"changed row count mismatch: {len(joined)} != {expected_changed_rows}"
        )
    return joined


def audit_release(
    *,
    release_dir: Path,
    repair_manifest: Path,
    output_dir: Path,
    url: str,
    model: str,
    tokenizer_path: Path,
    max_model_len: int,
    max_completion_tokens: int,
    timeout: int,
    max_retries: int,
    workers: int,
    expected_changed_rows: int,
) -> dict[str, Any]:
    rows = _load_changed_rows(
        release_dir=release_dir,
        repair_manifest=repair_manifest,
        expected_changed_rows=expected_changed_rows,
    )
    tokenizer = get_judge_tokenizer(tokenizer_path)
    health = check_judge(
        url=url,
        model=model,
        timeout=timeout,
        expected_model_root=tokenizer_path,
        tokenizer_path=tokenizer_path,
        max_model_len=max_model_len,
        max_completion_tokens=max_completion_tokens,
        tokenizer=tokenizer,
    )
    api_key = os.environ.get("OPEN_R1_JUDGE_API_KEY")

    prepared: list[tuple[dict[str, str], str, dict[str, Any], int]] = []
    for row in rows:
        judge_candidate = _reconstruct_judge_candidate(
            row["response"], sample_id=row["sample_id"]
        )
        body = build_judge_request_v3(
            evidence=row["provided_data"],
            candidate=judge_candidate,
            model=model,
            max_completion_tokens=max_completion_tokens,
        )
        prompt_tokens = len(render_judge_prompt_token_ids(body, tokenizer))
        if prompt_tokens + max_completion_tokens > max_model_len:
            raise CleanSftAuditError(
                f"{row['sample_id']}: judge context overflow "
                f"({prompt_tokens}+{max_completion_tokens}>{max_model_len})"
            )
        prepared.append((row, judge_candidate, body, prompt_tokens))

    def run_one(
        item: tuple[dict[str, str], str, dict[str, Any], int]
    ) -> dict[str, Any]:
        row, judge_candidate, body, prompt_tokens = item
        evaluation, raw, attempts = _judge_one_v3(
            evidence=row["provided_data"],
            candidate=judge_candidate,
            url=url,
            model=model,
            timeout=timeout,
            api_key=api_key,
            max_retries=max_retries,
            backoff_seconds=1.0,
            max_completion_tokens=max_completion_tokens,
            body=body,
        )
        valid, invalid = _validated_violations(evaluation, judge_candidate)
        blocking = [item for item in valid if item["kind"] in BLOCKING_KINDS]
        return {
            "schema_version": SCHEMA_VERSION,
            "sample_id": row["sample_id"],
            "split": row["split"],
            "candidate_sha256": row["response_sha256"],
            "evidence_sha256": _sha256_text(row["provided_data"]),
            "judge_prompt_tokens": prompt_tokens,
            "judge_attempts": attempts,
            "judge_raw_sha256": _sha256_text(raw),
            "rubric": {key: value for key, value in evaluation.items() if key != "violations"},
            "validated_violations": valid,
            "invalid_violations": invalid,
            "blocking_violations": blocking,
            "status": "passed" if not blocking else "failed",
        }

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, item): item[0] for item in prepared}
        for future in as_completed(futures):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                errors.append(
                    {
                        "sample_id": row["sample_id"],
                        "split": row["split"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    order = {name: index for index, name in enumerate(SPLITS)}
    results.sort(key=lambda row: (order[row["split"]], row["sample_id"]))
    errors.sort(key=lambda row: (order[row["split"]], row["sample_id"]))
    failed = [row for row in results if row["status"] != "passed"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not errors and not failed and len(results) == len(rows) else "failed",
        "release_dir": str(release_dir),
        "repair_manifest_sha256": hashlib.sha256(repair_manifest.read_bytes()).hexdigest(),
        "judge": {
            "url": url,
            "model": model,
            "tokenizer_path": str(tokenizer_path),
            "max_model_len": max_model_len,
            "max_completion_tokens": max_completion_tokens,
            "health": health,
        },
        "counts": {
            "expected": len(rows),
            "completed": len(results),
            "passed": sum(row["status"] == "passed" for row in results),
            "failed": len(failed),
            "judge_errors": len(errors),
            "blocking_violations": sum(len(row["blocking_violations"]) for row in results),
        },
        "row_audit": {
            "path": "row_audits.jsonl",
            "rows": len(results),
        },
        "errors": errors,
    }
    row_text = "".join(_canonical_json(row) + "\n" for row in results)
    _atomic_write(output_dir / "row_audits.jsonl", row_text)
    summary["row_audit"]["sha256"] = hashlib.sha256(row_text.encode("utf-8")).hexdigest()
    _atomic_write(output_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--repair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8001/v1/chat/completions")
    parser.add_argument("--model", default="Qwen3.5-9B")
    parser.add_argument("--tokenizer-path", type=Path, default=Path("models/Qwen3.5-9B"))
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-completion-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-changed-rows", type=int, default=237)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise SystemExit("workers must be in [1, 4]")
    summary = audit_release(
        release_dir=args.release_dir.resolve(),
        repair_manifest=args.repair_manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        url=args.url,
        model=args.model,
        tokenizer_path=args.tokenizer_path.resolve(),
        max_model_len=args.max_model_len,
        max_completion_tokens=args.max_completion_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
        workers=args.workers,
        expected_changed_rows=args.expected_changed_rows,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
