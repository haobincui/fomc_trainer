"""Targeted repair for residual chk3 Minutes-SFT teacher failures.

The main chk3 generator is intentionally not modified: its source hash binds
the accepted v2 cache.  This overlay sends only rows without a current-contract
accepted entry to DeepSeek with a shorter, inventory-driven prompt, validates
them with the original validator, and rematerializes the original output.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jobs.generation.generate_chk3_sft_targets import (
    DEFAULT_CHK1_HANDOFF,
    DEFAULT_CONCURRENCY,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TOKENIZER_PATH,
    MAX_TOKENS,
    Chk3DataError,
    ModelDriftError,
    OpenAIDeepSeekBackend,
    OutputContractError,
    PreparedRow,
    ProviderIdentityGuard,
    TeacherBackend,
    _MONTH_DAY_RE,
    _NUMBER_EXPRESSION_RE,
    _YEAR_RE,
    _accepted_cache_path,
    _accepted_payload,
    _cache_key,
    _date_values,
    _flatten_prepared,
    _load_json,
    _load_resume_cache,
    _load_tokenizer,
    _materialize_outputs,
    _rejection_payload,
    _store_immutable_json,
    _teacher_config,
    _write_json,
    _write_jsonl,
    canonical_json,
    prepare_chk1_release,
    prompt_contract,
    render_user_prompt,
    sha256_file,
    sha256_text,
    validate_teacher_target,
)


REPAIR_SCHEMA = "chk3-minutes-targeted-repair-v2"
DEFAULT_MAX_ATTEMPTS = 3
TARGET_REASONING_WORDS = "100-350"
TARGET_ANSWER_WORD_LIMIT = 500
_ANY_EVIDENCE_ID_RE = re.compile(r"\bev-[0-9a-f]+\b", flags=re.IGNORECASE)

TARGETED_REPAIR_SYSTEM_PROMPT = """\
You are repairing a difficult FOMC Minutes-rewrite SFT target using only the
supplied analysis. Native reasoning must begin directly with the economic
substance and use roughly three to six concise prose sentences. Identify the
claims, directions, comparisons, uncertainty, quantities, and dates that the
rewrite must preserve, but do not quote the full analysis or draft the final
paragraph in reasoning.

The request includes a silent fidelity ledger. Use it only to check the prose;
never name, quote, enumerate, or describe the ledger in native reasoning.
Preserve every listed quantity occurrence in the final paragraph, preferably
using its source surface form; duplicate entries require duplicate semantic occurrences.
Exact Decimal-equivalent unit conversions are permitted. Do not add, derive,
round, or approximate any quantity. Preserve every supplied date, direction,
comparison, and expression of uncertainty. Do not add facts, causes, people,
attributions, policy actions, decisions, or votes. Remove ev- citations.

Return content as exactly one JSON object with only the key answer. The answer
must be exactly one formal FOMC Minutes-style paragraph. Do not put reasoning,
headings, lists, citations, commentary, or control tags inside answer.

Native reasoning must contain only economic-content fidelity planning. Never
discuss or quote the task wording, messages, roles, instructions, prompts,
inventories, field names, identifiers, word or token limits, validation,
interfaces, transport formats, JSON, APIs, schemas, keys, tools, the answer,
the output, or the act of responding. Do not use first-person task narration.
"""


class TargetedChk3RepairError(Chk3DataError):
    """The residual chk3 repair could not complete safely."""


@dataclass(frozen=True)
class RepairResult:
    sample_id: str
    accepted: bool
    payload: Mapping[str, Any]
    requests: int


def source_quantity_occurrences(analysis: str) -> list[str]:
    """Return source-surface economic quantities, excluding calendar dates."""

    clean = _ANY_EVIDENCE_ID_RE.sub("", semantic_analysis(analysis))
    clean = _MONTH_DAY_RE.sub(lambda match: match.group("month"), clean)
    occurrences: list[str] = []
    for match in _NUMBER_EXPRESSION_RE.finditer(clean):
        raw_number = match.group("number")
        if (
            not match.group("scale")
            and not match.group("rate")
            and _YEAR_RE.fullmatch(raw_number)
        ):
            continue
        occurrences.append(match.group(0).strip())
    return occurrences


def semantic_analysis(analysis: str) -> str:
    """Project malformed chk1 transport wrappers to their actual final answer.

    A small legacy subset contains the chk1 provider JSON envelope instead of
    its final ``answer`` value.  Some envelopes are truncated only after a
    complete answer.  This projection never invents prose: it selects an
    existing answer string, removes internal evidence IDs, and, for an
    incomplete answer string, retains only complete source sentences.
    """

    raw = str(analysis).strip()
    candidate = raw
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        direct = payload.get("answer")
        content = payload.get("content")
        if isinstance(direct, str) and direct.strip():
            candidate = direct
        elif isinstance(content, dict) and isinstance(content.get("answer"), str):
            candidate = str(content["answer"])
        elif isinstance(content, str):
            try:
                nested = json.loads(content)
            except json.JSONDecodeError:
                nested = None
            if isinstance(nested, dict) and isinstance(nested.get("answer"), str):
                candidate = str(nested["answer"])
    elif re.match(r'^\s*\{\s*"answer"\s*:\s*"', raw):
        prefix = re.match(r'^\s*\{\s*"answer"\s*:\s*"', raw)
        assert prefix is not None
        tail = raw[prefix.end() :]
        match = re.search(r'"\s*,\s*"evidence_ids"\s*:', tail, flags=re.DOTALL)
        if match is not None:
            encoded = tail[: match.start()]
            try:
                candidate = json.loads(f'"{encoded}"')
            except json.JSONDecodeError:
                candidate = encoded
        else:
            # The provider transport was truncated inside answer.  Retain
            # only complete sentences already present in that answer string.
            last_period = tail.rfind(".")
            if last_period >= 0:
                candidate = tail[: last_period + 1]

    candidate = _ANY_EVIDENCE_ID_RE.sub("", candidate)
    candidate = re.sub(r"\(\s*(?:,\s*)*\)|\[\s*(?:,\s*)*\]", "", candidate)
    candidate = re.sub(r"\s+([,.;:!?])", r"\1", candidate)
    candidate = " ".join(candidate.split()).strip()
    return candidate or raw


def targeted_user_prompt(
    row: PreparedRow,
    *,
    attempt: int,
    prior_errors: Sequence[str],
) -> str:
    projected = semantic_analysis(row.analysis)
    quantities = canonical_json(source_quantity_occurrences(row.analysis))
    dates = canonical_json(sorted(_date_values(projected)))
    diagnostics = canonical_json(list(prior_errors))
    return (
        render_user_prompt(projected)
        + "\n\nSilent fidelity ledger (use only for checking; never mention it):\n"
        + f"Quantity occurrences: {quantities}\n"
        + f"Date markers: {dates}\n"
        + "Every occurrence must be preserved; no extra or derived quantity is allowed.\n"
        + f"Silent correction notes for attempt {attempt}: {diagnostics}"
    )


def repair_contract(
    *,
    code_sha256: str,
    original_contract_sha256: str,
    max_attempts: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": REPAIR_SCHEMA,
        "purpose": "inventory_driven_retry_of_residual_chk3_minutes_targets",
        "original_contract_sha256": original_contract_sha256,
        "system_prompt": TARGETED_REPAIR_SYSTEM_PROMPT,
        "teacher": _teacher_config().contract(),
        "max_targeted_attempts": max_attempts,
        "acceptance_validator": (
            "original_chk3_v2_validator_after_semantic_source_projection"
        ),
        "reasoning_word_target": TARGET_REASONING_WORDS,
        "answer_word_limit": TARGET_ANSWER_WORD_LIMIT,
        "code_sha256": code_sha256,
    }
    payload["repair_contract_sha256"] = sha256_text(canonical_json(payload))
    return payload


def _targeted_rejection_path(
    output_root: Path,
    *,
    cache_key: str,
    response_id: str,
    payload_sha256: str,
) -> Path:
    identity = response_id.strip() or payload_sha256[:24]
    safe = "".join(char if char.isalnum() or char in "_.-" else "_" for char in identity)
    return (
        output_root
        / "cache"
        / "targeted_repair_rejected"
        / cache_key[:2]
        / cache_key
        / f"{safe}-{payload_sha256[:12]}.json"
    )


def _store_targeted_rejection(
    *,
    output_root: Path,
    row: PreparedRow,
    cache_key: str,
    attempt: int,
    errors: Sequence[str],
    response: Any | None,
    repair_contract_sha256: str,
) -> None:
    base = _rejection_payload(
        row=row,
        cache_key=cache_key,
        attempt=f"targeted_repair_{attempt}",
        error=";".join(errors),
        response=response,
    )
    base["schema_version"] = REPAIR_SCHEMA
    base["repair_contract_sha256"] = repair_contract_sha256
    payload_sha = sha256_text(canonical_json(base))
    response_id = "" if response is None else str(response.response_id)
    _store_immutable_json(
        _targeted_rejection_path(
            output_root,
            cache_key=cache_key,
            response_id=response_id,
            payload_sha256=payload_sha,
        ),
        base,
    )


def repair_one(
    row: PreparedRow,
    *,
    output_root: Path,
    tokenizer: Any,
    backend: TeacherBackend,
    identity_guard: ProviderIdentityGuard,
    environment: Mapping[str, str] | None,
    original_code_sha256: str,
    repair_contract_payload: Mapping[str, Any],
    max_attempts: int,
) -> RepairResult:
    cache_key = _cache_key(row, code_sha256=original_code_sha256)
    prior_errors: tuple[str, ...] = ()
    requests = 0
    for attempt in range(1, max_attempts + 1):
        response = None
        try:
            requests += 1
            response = backend.generate(
                config=_teacher_config(),
                system_prompt=TARGETED_REPAIR_SYSTEM_PROMPT,
                user_prompt=targeted_user_prompt(
                    row, attempt=attempt, prior_errors=prior_errors
                ),
                environment=environment,
            )
            identity_guard.bind(
                returned_model=response.returned_model,
                system_fingerprint=response.system_fingerprint,
            )
            target = validate_teacher_target(
                response=response,
                analysis=semantic_analysis(row.analysis),
                user_prompt=row.user_prompt,
                tokenizer=tokenizer,
                max_length=MAX_TOKENS,
            )
            payload = _accepted_payload(
                row=row,
                response=response,
                target=target,
                cache_key=cache_key,
                code_sha256=original_code_sha256,
                attempt=f"targeted_repair_{attempt}",
            )
            payload["targeted_repair"] = {
                "schema_version": REPAIR_SCHEMA,
                "repair_contract_sha256": repair_contract_payload[
                    "repair_contract_sha256"
                ],
                "attempt": attempt,
                "quantity_inventory_sha256": sha256_text(
                    canonical_json(source_quantity_occurrences(row.analysis))
                ),
                "date_inventory_sha256": sha256_text(
                    canonical_json(sorted(_date_values(semantic_analysis(row.analysis))))
                ),
                "semantic_source_projection_sha256": sha256_text(
                    semantic_analysis(row.analysis)
                ),
            }
            _store_immutable_json(
                _accepted_cache_path(output_root, cache_key), payload
            )
            return RepairResult(row.sample_id, True, payload, requests)
        except ModelDriftError:
            raise
        except OutputContractError as exc:
            prior_errors = exc.codes
            _store_targeted_rejection(
                output_root=output_root,
                row=row,
                cache_key=cache_key,
                attempt=attempt,
                errors=prior_errors,
                response=response,
                repair_contract_sha256=str(
                    repair_contract_payload["repair_contract_sha256"]
                ),
            )
        except Exception as exc:  # noqa: BLE001 - provider exceptions vary
            prior_errors = (f"{type(exc).__name__}:{exc}",)
            _store_targeted_rejection(
                output_root=output_root,
                row=row,
                cache_key=cache_key,
                attempt=attempt,
                errors=prior_errors,
                response=response,
                repair_contract_sha256=str(
                    repair_contract_payload["repair_contract_sha256"]
                ),
            )
    return RepairResult(
        row.sample_id,
        False,
        {
            "status": "failed",
            "sample_id": row.sample_id,
            "split": row.split,
            "source_index": row.source_index,
            "error": ";".join(prior_errors),
            "requests": requests,
        },
        requests,
    )


def run(
    *,
    chk1_handoff: Path,
    output_root: Path,
    tokenizer_path: Path,
    dry_run: bool,
    concurrency: int,
    max_attempts: int,
    backend: TeacherBackend | None = None,
    environment: Mapping[str, str] | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    if concurrency < 1 or max_attempts < 1:
        raise TargetedChk3RepairError(
            "concurrency and max-attempts must both be positive"
        )
    main_script = Path(__file__).with_name("generate_chk3_sft_targets.py")
    original_code_sha = sha256_file(main_script)
    on_disk_contract = _load_json(output_root / "prompt_contract.json", label="prompt contract")
    expected_contract = prompt_contract(code_sha256=original_code_sha)
    if on_disk_contract != expected_contract:
        raise TargetedChk3RepairError(
            "main chk3 prompt contract/code drift; refusing to alter its cache"
        )

    prepared, preparation = prepare_chk1_release(
        chk1_handoff, output_root=output_root
    )
    rows = _flatten_prepared(prepared)
    active_tokenizer = tokenizer or _load_tokenizer(tokenizer_path)
    identity_guard = ProviderIdentityGuard()
    accepted = _load_resume_cache(
        rows,
        output_root=output_root,
        code_sha256=original_code_sha,
        resume=True,
        identity_guard=identity_guard,
    )
    pending = [row for row in rows if row.sample_id not in accepted]
    repair = repair_contract(
        code_sha256=sha256_file(Path(__file__).resolve()),
        original_contract_sha256=sha256_text(canonical_json(on_disk_contract)),
        max_attempts=max_attempts,
    )
    _write_json(output_root / "targeted_repair_contract.json", repair)
    base_summary: dict[str, Any] = {
        "schema_version": REPAIR_SCHEMA,
        "status": "prepared" if dry_run else "running",
        "original_code_sha256": original_code_sha,
        "original_contract_sha256": repair["original_contract_sha256"],
        "repair_contract_sha256": repair["repair_contract_sha256"],
        "total_rows": len(rows),
        "resumed_accepted_count": len(accepted),
        "pending_count": len(pending),
        "pending_sample_ids": [row.sample_id for row in pending],
        "max_attempts_per_row": max_attempts,
        "api_requests": 0,
        "provider_identity": identity_guard.identity,
    }
    if dry_run or not pending:
        base_summary["status"] = "prepared" if dry_run else "complete"
        _write_json(output_root / "targeted_repair_summary.json", base_summary)
        return base_summary

    provider = backend or OpenAIDeepSeekBackend()
    repaired: dict[str, Mapping[str, Any]] = {}
    failures: dict[str, Mapping[str, Any]] = {}
    requests = 0
    drift: BaseException | None = None
    print(
        f"[chk3-target-repair] resumed={len(accepted)} pending={len(pending)} "
        f"max_attempts={max_attempts}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                repair_one,
                row,
                output_root=output_root,
                tokenizer=active_tokenizer,
                backend=provider,
                identity_guard=identity_guard,
                environment=environment,
                original_code_sha256=original_code_sha,
                repair_contract_payload=repair,
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
                    repaired[result.sample_id] = result.payload
                    print(
                        f"[chk3-target-repair] accepted={len(repaired)}/{len(pending)} "
                        f"sample_id={result.sample_id}",
                        flush=True,
                    )
                else:
                    failures[result.sample_id] = result.payload
                    print(
                        f"[chk3-target-repair] failed sample_id={result.sample_id}",
                        flush=True,
                    )
            except ModelDriftError as exc:
                drift = exc
                for other in futures:
                    other.cancel()
                break
            except Exception as exc:  # noqa: BLE001 - preserve row failure
                failures[row.sample_id] = {
                    "status": "failed",
                    "sample_id": row.sample_id,
                    "split": row.split,
                    "source_index": row.source_index,
                    "error": f"{type(exc).__name__}:{exc}",
                }
    if drift is not None:
        raise ModelDriftError(str(drift))

    accepted.update(repaired)
    main_summary = _materialize_outputs(
        prepared=prepared,
        accepted=accepted,
        failures=failures,
        output_root=output_root,
        source_handoff_sha256=str(preparation["source_handoff_sha256"]),
        identity=identity_guard.identity,
    )
    summary = {
        **base_summary,
        "status": "complete" if not failures else "incomplete",
        "repaired_count": len(repaired),
        "remaining_failure_count": len(failures),
        "api_requests": requests,
        "provider_identity": identity_guard.identity,
        "main_generation_status": main_summary["status"],
        "main_total_accepted": main_summary["total_accepted"],
    }
    _write_jsonl(
        output_root / "targeted_repair_failures.jsonl",
        [failures[key] for key in sorted(failures)],
    )
    _write_json(output_root / "targeted_repair_summary.json", summary)
    if failures:
        raise TargetedChk3RepairError(
            f"targeted repair incomplete: repaired={len(repaired)} "
            f"remaining={len(failures)}"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chk1-handoff", type=Path, default=DEFAULT_CHK1_HANDOFF)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run(
            chk1_handoff=args.chk1_handoff.resolve(),
            output_root=args.output_root.resolve(),
            tokenizer_path=args.tokenizer_path.resolve(),
            dry_run=args.dry_run,
            concurrency=args.concurrency,
            max_attempts=args.max_attempts,
        )
    except (Chk3DataError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
