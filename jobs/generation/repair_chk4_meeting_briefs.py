"""Targeted repair for failed chk4 meeting-brief acquisitions.

This helper deliberately does not revise the immutable v1 brief contract.  It
loads and revalidates every accepted cache entry, then sends only currently
missing rows to DeepSeek with tighter repair instructions.  A repaired result
is accepted only when it passes the original v1 validator, so the ordinary
``generate_chk4_meeting_briefs --resume`` command can consume it unchanged.
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

from jobs.generation.generate_chk4_meeting_briefs import (
    CACHE_SCHEMA,
    DEFAULT_CHK1_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TOKENIZER,
    BriefOutputError,
    BriefTeacherConfig,
    MeetingInput,
    _cache_key,
    _cache_path,
    _flatten,
    _load_resume,
    _load_tokenizer,
    _store_immutable,
    build_contract,
    load_meeting_inputs,
    validate_brief,
)
from jobs.generation.generate_chk4_sft_targets import (
    ModelDriftError,
    OpenAICompatibleDeepSeekBackend,
    ProviderIdentityGuard,
    TeacherResponse,
    canonical_json,
    sha256_file,
    sha256_text,
)


DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_ATTEMPTS = 3
REPAIR_SCHEMA = "chk4-deepseek-meeting-brief-targeted-repair-v1"

TARGETED_REPAIR_SYSTEM_PROMPT = """\
You are repairing a failed target-neutral pre-meeting FOMC synthesis. Use only
the supplied atomic topic analyses. Keep reasoning_content under 180 words and
prioritize returning the required JSON content well before the token limit.

Do not infer or mention the meeting identity. Do not use outside or remembered
historical information. Do not state, recommend, predict, or imply a policy
decision, vote, rate change, target range, or action actually taken. Do not
mention validation, gold labels, Minutes, prompts, schemas, or hidden fields.

NUMBERS AND DATES ARE COPY-ONLY: either omit them or copy each complete numeric
or date expression exactly as written in the supplied analyses. Never compute,
round, expand, contract, reformat, or convert a value or unit. In particular,
do not convert thousand/million/billion expressions to digits or digits to
scaled units. Introduce no new fact, cause, number, or date.

Return content as exactly one valid JSON object with the single key
meeting_decision_brief. The value must be one concise paragraph without
headings, lists, citations, internal IDs, recommendations, policy actions, or
embedded JSON. Return no content outside that JSON object.
"""


class TargetedRepairError(RuntimeError):
    """The targeted cache repair could not complete safely."""


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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TargetedRepairError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise TargetedRepairError(f"JSON root must be an object: {path}")
    return value


def _failure_codes(output_root: Path) -> dict[str, tuple[str, ...]]:
    path = output_root / "failures.jsonl"
    if not path.is_file():
        raise TargetedRepairError(
            "no incomplete brief failure ledger exists; run the normal generator first"
        )
    result: dict[str, tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TargetedRepairError(
                    f"invalid failure ledger JSONL at {path}:{line_number}"
                ) from exc
            sample_id = str(row.get("sample_id") or "").strip()
            error = str(row.get("error") or "").strip()
            if not sample_id or not error:
                raise TargetedRepairError(
                    f"incomplete failure ledger row at {path}:{line_number}"
                )
            result[sample_id] = tuple(
                item for item in error.split(";") if item
            )
    if not result:
        raise TargetedRepairError("failure ledger is empty")
    return result


def _repair_contract(code_sha256: str, original_contract_sha256: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": REPAIR_SCHEMA,
        "purpose": "targeted_retry_of_originally_failed_rows_only",
        "original_contract_sha256": original_contract_sha256,
        "system_prompt": TARGETED_REPAIR_SYSTEM_PROMPT,
        "teacher": BriefTeacherConfig().contract(),
        "max_targeted_attempts": DEFAULT_MAX_ATTEMPTS,
        "acceptance_validator": "original_chk4_deepseek_meeting_brief_v1",
        "code_sha256": code_sha256,
    }
    payload["repair_contract_sha256"] = sha256_text(canonical_json(payload))
    return payload


def _repair_prompt(row: MeetingInput, errors: Sequence[str], attempt: int) -> str:
    return row.user_prompt + "\n\n" + canonical_json(
        {
            "repair_directive": (
                "Silently regenerate the JSON. Copy numeric and date expressions "
                "verbatim or omit them; do not normalize units."
            ),
            "previous_contract_errors": list(errors),
            "targeted_attempt": attempt,
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
    row: MeetingInput,
    attempt: int,
    errors: Sequence[str],
    response: TeacherResponse | None,
    repair_contract: Mapping[str, Any],
) -> None:
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
    key = sha256_text(canonical_json(payload))
    _store_immutable(
        output_root
        / "cache"
        / "targeted_repair_rejected"
        / row.sample_id
        / f"{key}.json",
        payload,
    )


def repair_one(
    row: MeetingInput,
    *,
    initial_errors: Sequence[str],
    output_root: Path,
    tokenizer: Any,
    backend: Any,
    guard: ProviderIdentityGuard,
    original_contract: Mapping[str, Any],
    repair_contract: Mapping[str, Any],
    environment: Mapping[str, str] | None,
    max_attempts: int,
) -> RepairResult:
    errors = tuple(initial_errors)
    request_count = 0
    for attempt in range(1, max_attempts + 1):
        response: TeacherResponse | None = None
        try:
            request_count += 1
            response = backend.generate(
                config=BriefTeacherConfig(),
                system_prompt=TARGETED_REPAIR_SYSTEM_PROMPT,
                user_prompt=_repair_prompt(row, errors, attempt),
                environment=environment,
            )
            guard.bind(response)
            target = validate_brief(row, response, tokenizer)
            payload = {
                "schema_version": CACHE_SCHEMA,
                "status": "accepted",
                "cache_key": _cache_key(row, original_contract),
                "sample_id": row.sample_id,
                "input_sha256": row.input_sha256,
                "prompt_sha256": row.prompt_sha256,
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
                    "initial_errors": list(initial_errors),
                    "attempt": attempt,
                },
            }
            _store_immutable(
                _cache_path(output_root, payload["cache_key"]), payload
            )
            return RepairResult(row.sample_id, True, payload, request_count)
        except ModelDriftError:
            raise
        except BriefOutputError as exc:
            errors = exc.codes
            _store_rejected(
                output_root=output_root,
                row=row,
                attempt=attempt,
                errors=errors,
                response=response,
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
    chk1_root: Path,
    output_root: Path,
    tokenizer_path: Path,
    dry_run: bool,
    concurrency: int,
    max_attempts: int,
    backend: Any | None = None,
    environment: Mapping[str, str] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    if concurrency < 1 or max_attempts < 1:
        raise TargetedRepairError("concurrency and max_attempts must be positive")
    grouped = load_meeting_inputs(chk1_root)
    rows = _flatten(grouped)
    row_by_id = {row.sample_id: row for row in rows}

    original_contract_path = output_root / "prompt_contract.json"
    original_contract = _load_json(original_contract_path)
    source_script = Path(__file__).with_name("generate_chk4_meeting_briefs.py")
    expected_contract = build_contract(sha256_file(source_script))
    if original_contract != expected_contract:
        raise TargetedRepairError(
            "original brief code/contract drift; refusing to rewrite immutable cache"
        )

    failure_codes = _failure_codes(output_root)
    unknown_failures = sorted(set(failure_codes) - set(row_by_id))
    if unknown_failures:
        raise TargetedRepairError(
            f"failure ledger contains unknown sample IDs: {unknown_failures}"
        )
    tokenizer = tokenizer if tokenizer is not None else _load_tokenizer(tokenizer_path)
    guard = ProviderIdentityGuard()
    accepted = _load_resume(
        rows,
        output_root=output_root,
        contract=original_contract,
        tokenizer=tokenizer,
        resume=True,
        guard=guard,
    )
    pending = [row for row in rows if row.sample_id not in accepted]
    unledgered = sorted(row.sample_id for row in pending if row.sample_id not in failure_codes)
    if unledgered:
        raise TargetedRepairError(
            f"pending rows are absent from failure ledger: {unledgered}"
        )

    repair_contract = _repair_contract(
        sha256_file(Path(__file__).resolve()), original_contract["contract_sha256"]
    )
    repair_contract["max_targeted_attempts"] = max_attempts
    repair_contract["repair_contract_sha256"] = sha256_text(
        canonical_json(
            {
                key: value
                for key, value in repair_contract.items()
                if key != "repair_contract_sha256"
            }
        )
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
    failures: list[dict[str, Any]] = []
    repaired: dict[str, Mapping[str, Any]] = {}
    requests = 0
    drift: BaseException | None = None
    print(
        f"[chk4-brief-repair] resumed={len(accepted)} pending={len(pending)}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                repair_one,
                row,
                initial_errors=failure_codes[row.sample_id],
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
                        f"[chk4-brief-repair] accepted={len(repaired)}/{len(pending)} "
                        f"sample={row.sample_id}",
                        flush=True,
                    )
                else:
                    failures.append(dict(result.payload))
                    print(
                        f"[chk4-brief-repair] failed sample={row.sample_id}",
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
        raise TargetedRepairError(
            f"targeted repair incomplete: {len(repaired)}/{len(pending)}"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chk1-root", type=Path, default=DEFAULT_CHK1_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(
            chk1_root=args.chk1_root.resolve(),
            output_root=args.output_root.resolve(),
            tokenizer_path=args.tokenizer_path.resolve(),
            dry_run=args.dry_run,
            concurrency=args.concurrency,
            max_attempts=args.max_attempts,
        )
    except (TargetedRepairError, ModelDriftError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2
    print(canonical_json(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
