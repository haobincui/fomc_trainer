"""Targeted retry of missing chk4 Decision-SFT teacher targets.

Accepted entries from the immutable target acquisition contract are preserved.
Only rows without an accepted cache entry are sent to DeepSeek, using a tighter
JSON-rationale contract that prevents numeric reformatting and completion overflow.
Every repaired response must still pass the original target validator.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jobs.generation.generate_chk4_sft_targets import (
    CACHE_SCHEMA,
    DEFAULT_BRIEFS_ROOT,
    DEFAULT_CONCURRENCY,
    DEFAULT_CORE_MANIFEST_ROOT,
    DEFAULT_FFR_HISTORY,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TOKENIZER,
    DEFAULT_WORKBOOK,
    SPLITS,
    Chk4TargetError,
    ModelDriftError,
    OpenAICompatibleDeepSeekBackend,
    OutputContractError,
    PreparedRow,
    ProviderIdentityGuard,
    TeacherConfig,
    TeacherResponse,
    _cache_key,
    _cache_path,
    _load_json,
    _load_resume_cache,
    _load_tokenizer,
    _store_immutable,
    build_prompt_contract,
    canonical_json,
    load_and_validate_labels,
    load_core_split_map,
    prepare_rows,
    sha256_file,
    sha256_text,
    validate_prepared_token_budgets,
    validate_response,
)


DEFAULT_MAX_ATTEMPTS = 3
REPAIR_SCHEMA = "chk4-decision-targeted-repair-v1"

TARGETED_REPAIR_SYSTEM_PROMPT = """\
You are generating a concise rationale for an FOMC policy Decision-SFT example.
Use only the supplied target-neutral pre-meeting analysis and weigh inflation,
employment and real activity, financial conditions, and risks under the dual
mandate. The request ends with the canonical decision JSON that the rationale
must support.

Keep the JSON reasoning field below 180 words. Do not include any digit, date,
percentage, basis-point amount, explicit numeric quantity, or numeric policy
magnitude in that field. Describe directions and tradeoffs qualitatively
instead. Do not use outside or remembered history, infer or mention meeting
identity, or claim that the Committee actually took an action. Do not mention
gold labels, teacher targets, answer keys, hidden fields, prompts, schemas,
validation, or teacher/student roles.

Return content as exactly one valid JSON object with only reasoning, direction,
and magnitude_bp. Copy direction and magnitude_bp exactly from the canonical
decision at the end of the request. Return no content outside that JSON object.
Prioritize producing this JSON before the token limit.
"""


class TargetedTargetRepairError(Chk4TargetError):
    """The targeted Decision-SFT repair could not complete safely."""


@dataclass(frozen=True)
class RepairResult:
    sample_id: str
    accepted: bool
    payload: Mapping[str, Any]
    requests: int


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_write(path, "".join(canonical_json(row) + "\n" for row in rows))


def _repair_contract(
    code_sha256: str,
    original_contract_sha256: str,
    max_attempts: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": REPAIR_SCHEMA,
        "purpose": "targeted_retry_of_missing_decision_sft_targets",
        "original_contract_sha256": original_contract_sha256,
        "system_prompt": TARGETED_REPAIR_SYSTEM_PROMPT,
        "teacher": TeacherConfig().contract(),
        "max_targeted_attempts": max_attempts,
        "acceptance_validator": "original_chk4_decision_target_v1",
        "code_sha256": code_sha256,
    }
    payload["repair_contract_sha256"] = sha256_text(canonical_json(payload))
    return payload


def _repair_prompt(row: PreparedRow, attempt: int) -> str:
    return row.teacher_prompt + "\n" + canonical_json(
        {
            "targeted_retry": attempt,
            "reasoning_rule": (
                "Use qualitative prose under 180 words with no digits, dates, "
                "percentages, or numeric quantities in the JSON reasoning field."
            ),
        }
    )


def _provider_payload(response: TeacherResponse) -> dict[str, Any]:
    return {
        "response_id": response.response_id,
        "returned_model": response.returned_model,
        "system_fingerprint": response.system_fingerprint,
        "finish_reason": response.finish_reason,
        "created": response.created,
        "usage": dict(response.usage),
    }


def _store_rejected(
    *,
    output_root: Path,
    row: PreparedRow,
    attempt: int,
    errors: Sequence[str],
    response: TeacherResponse | None,
    original_contract: Mapping[str, Any],
    repair_contract: Mapping[str, Any],
) -> None:
    cache_key = _cache_key(row, original_contract)
    payload = {
        "schema_version": REPAIR_SCHEMA,
        "status": "rejected",
        "sample_id": row.sample_id,
        "attempt": attempt,
        "errors": list(errors),
        "repair_contract_sha256": repair_contract["repair_contract_sha256"],
        "provider": None if response is None else _provider_payload(response),
        "provider_raw": None
        if response is None
        else {
            "reasoning_content": response.reasoning,
            "content": response.content,
        },
    }
    rejection_key = sha256_text(canonical_json(payload))
    _store_immutable(
        output_root
        / "cache"
        / "targeted_repair_rejected"
        / cache_key[:2]
        / cache_key
        / f"{rejection_key}.json",
        payload,
    )


def repair_one(
    row: PreparedRow,
    *,
    output_root: Path,
    tokenizer: Any,
    backend: Any,
    guard: ProviderIdentityGuard,
    original_contract: Mapping[str, Any],
    repair_contract: Mapping[str, Any],
    environment: Mapping[str, str] | None,
    max_attempts: int,
) -> RepairResult:
    errors: tuple[str, ...] = ()
    request_count = 0
    cache_key = _cache_key(row, original_contract)
    for attempt in range(1, max_attempts + 1):
        response: TeacherResponse | None = None
        try:
            request_count += 1
            response = backend.generate(
                config=TeacherConfig(),
                system_prompt=TARGETED_REPAIR_SYSTEM_PROMPT,
                user_prompt=_repair_prompt(row, attempt),
                environment=environment,
            )
            guard.bind(response)
            target = validate_response(
                row=row, response=response, tokenizer=tokenizer
            )
            payload = {
                "schema_version": CACHE_SCHEMA,
                "status": "accepted",
                "cache_key": cache_key,
                "sample_id": row.sample_id,
                "input_sha256": row.input_sha256,
                "prompt_sha256": row.prompt_sha256,
                "gold_sha256": row.gold_sha256,
                "contract_sha256": original_contract["contract_sha256"],
                "attempt": f"targeted_repair_{attempt}",
                "provider": _provider_payload(response),
                "provider_raw": {
                    "reasoning_content": response.reasoning,
                    "content": response.content,
                },
                "target": target,
                "targeted_repair": {
                    "schema_version": REPAIR_SCHEMA,
                    "repair_contract_sha256": repair_contract[
                        "repair_contract_sha256"
                    ],
                    "attempt": attempt,
                },
            }
            _store_immutable(_cache_path(output_root, cache_key), payload)
            return RepairResult(row.sample_id, True, payload, request_count)
        except ModelDriftError:
            raise
        except OutputContractError as exc:
            errors = exc.codes
            _store_rejected(
                output_root=output_root,
                row=row,
                attempt=attempt,
                errors=errors,
                response=response,
                original_contract=original_contract,
                repair_contract=repair_contract,
            )
        except Exception as exc:
            errors = (f"{type(exc).__name__}:{exc}",)
            _store_rejected(
                output_root=output_root,
                row=row,
                attempt=attempt,
                errors=errors,
                response=response,
                original_contract=original_contract,
                repair_contract=repair_contract,
            )
    return RepairResult(
        row.sample_id,
        False,
        {
            "status": "failed",
            "sample_id": row.sample_id,
            "error": ";".join(errors),
            "requests": request_count,
        },
        request_count,
    )


def run(
    *,
    briefs_root: Path,
    output_root: Path,
    workbook: Path,
    ffr_history: Path,
    core_manifest_root: Path,
    tokenizer_path: Path,
    dry_run: bool,
    concurrency: int,
    max_attempts: int,
    backend: Any | None = None,
    environment: Mapping[str, str] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    if concurrency < 1 or max_attempts < 1:
        raise TargetedTargetRepairError(
            "concurrency and max_attempts must be positive"
        )
    source_script = Path(__file__).with_name("generate_chk4_sft_targets.py")
    original_contract = _load_json(output_root / "prompt_contract.json")
    expected_contract = build_prompt_contract(sha256_file(source_script))
    if original_contract != expected_contract:
        raise TargetedTargetRepairError(
            "original target code/contract drift; refusing to rewrite immutable cache"
        )

    labels, _ = load_and_validate_labels(workbook, ffr_history)
    core_splits = load_core_split_map(core_manifest_root)
    prepared, _ = prepare_rows(
        briefs_root=briefs_root,
        labels=labels,
        core_splits=core_splits,
    )
    rows = [row for split in SPLITS for row in prepared[split]]
    tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(tokenizer_path)
    validate_prepared_token_budgets(prepared, tokenizer)

    guard = ProviderIdentityGuard()
    accepted = _load_resume_cache(
        rows,
        output_root=output_root,
        contract=original_contract,
        resume=True,
        guard=guard,
        tokenizer=tokenizer,
    )
    pending = [row for row in rows if row.sample_id not in accepted]
    repair_contract = _repair_contract(
        sha256_file(Path(__file__).resolve()),
        original_contract["contract_sha256"],
        max_attempts,
    )
    _write_json(output_root / "targeted_repair_contract.json", repair_contract)
    base_summary: dict[str, Any] = {
        "schema_version": REPAIR_SCHEMA,
        "status": "prepared" if dry_run else "running",
        "original_contract_sha256": original_contract["contract_sha256"],
        "repair_contract_sha256": repair_contract["repair_contract_sha256"],
        "total_rows": len(rows),
        "resumed_accepted_count": len(accepted),
        "pending_count": len(pending),
        "pending_sample_ids": [row.sample_id for row in pending],
        "max_attempts_per_row": max_attempts,
        "api_requests": 0,
        "provider_identity": guard.identity,
    }
    if dry_run or not pending:
        base_summary["status"] = "prepared" if dry_run else "complete"
        _write_json(output_root / "targeted_repair_summary.json", base_summary)
        return base_summary

    provider = backend or OpenAICompatibleDeepSeekBackend()
    repaired: dict[str, Mapping[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    requests = 0
    drift: BaseException | None = None
    print(
        f"[chk4-target-repair] resumed={len(accepted)} pending={len(pending)}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                repair_one,
                row,
                output_root=output_root,
                tokenizer=tokenizer,
                backend=provider,
                guard=guard,
                original_contract=original_contract,
                repair_contract=repair_contract,
                environment=environment,
                max_attempts=max_attempts,
            ): row
            for row in pending
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
                requests += result.requests
                if result.accepted:
                    repaired[row.sample_id] = result.payload
                    print(
                        f"[chk4-target-repair] accepted={len(repaired)}/{len(pending)} "
                        f"sample={row.sample_id}",
                        flush=True,
                    )
                else:
                    failures.append(dict(result.payload))
                    print(
                        f"[chk4-target-repair] failed sample={row.sample_id}",
                        flush=True,
                    )
            except ModelDriftError as exc:
                drift = exc
                for other in futures:
                    other.cancel()
                break
            except Exception as exc:
                failures.append(
                    {
                        "status": "failed",
                        "sample_id": row.sample_id,
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )
    if drift is not None:
        raise ModelDriftError(str(drift))

    summary = {
        **base_summary,
        "status": "complete" if len(repaired) == len(pending) and not failures else "incomplete",
        "repaired_count": len(repaired),
        "remaining_failure_count": len(failures),
        "api_requests": requests,
        "provider_identity": guard.identity,
    }
    _write_jsonl(output_root / "targeted_repair_failures.jsonl", failures)
    _write_json(output_root / "targeted_repair_summary.json", summary)
    if summary["status"] != "complete":
        raise TargetedTargetRepairError(
            f"targeted target repair incomplete: {len(repaired)}/{len(pending)}"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--briefs-root", type=Path, default=DEFAULT_BRIEFS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--decision-workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--ffr-history", type=Path, default=DEFAULT_FFR_HISTORY)
    parser.add_argument(
        "--core-manifest-root", type=Path, default=DEFAULT_CORE_MANIFEST_ROOT
    )
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(
            briefs_root=args.briefs_root.resolve(),
            output_root=args.output_root.resolve(),
            workbook=args.decision_workbook.resolve(),
            ffr_history=args.ffr_history.resolve(),
            core_manifest_root=args.core_manifest_root.resolve(),
            tokenizer_path=args.tokenizer_path.resolve(),
            dry_run=args.dry_run,
            concurrency=args.concurrency,
            max_attempts=args.max_attempts,
        )
    except (Chk4TargetError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2
    print(canonical_json(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
