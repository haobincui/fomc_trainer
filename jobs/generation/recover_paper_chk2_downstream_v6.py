"""Seal and recover an interrupted paper chk-2 v6 downstream acquisition.

The original v6 acquisition and its implementation are never modified.  A
byte-for-byte baseline is copied into an independent recovery root.  Recovery
then invokes the unchanged v6 per-row worker only for rows without a terminal.

The compatibility verifier is deliberately narrow.  It recognizes two v6
serialization cases in which a deterministic repair failure is terminalized
after the record has retained the last valid candidate (or no candidate):

* a failed fidelity repair after the primary Validator-A call; and
* a failed style repair after the primary Validator-B call.

Every other terminal is replayed by the original v6 verifier without a
fallback.  Compatibility never changes a terminal verdict or admits a row.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jobs.generation import generate_paper_chk2_chk1_analysis_rewrite as v5
from jobs.generation import generate_paper_chk2_downstream_v6 as v6
from jobs.generation import paper_chk2_official_reference_v2 as official_v2
from jobs.generation import seal_paper_chk2_source_handoff_v1 as source_handoff_v1


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIGINAL_ROOT = v6.DEFAULT_OUTPUT_ROOT
DEFAULT_RECOVERY_ROOT = REPO_ROOT / (
    "output/data/retrain_v2/chk2/"
    "chk1_final_analysis_to_minutes_flash_official_reference_"
    "v6_downstream128_recovery_v1_20260831"
)
DEFAULT_HANDOFF_ROOT = source_handoff_v1.DEFAULT_HANDOFF_ROOT
DEFAULT_TOKENIZER_PATH = v6.DEFAULT_TOKENIZER_PATH
SCHEMA_VERSION = "paper-chk2-v6-recovery-v1"
PARTIAL_MANIFEST_SCHEMA = "paper-chk2-v6-partial-handoff-v1"
RECEIPT_SCHEMA = "paper-chk2-v6-recovery-receipt-v1"
RECOVERY_CONCURRENCY = 128
EXPECTED_BASELINE_COMPATIBILITY_COUNT = 18
EXPECTED_BASELINE_COMPATIBILITY_ID_DIGEST = (
    "5c8fdab1f96a2993a44e1b061f034f73330cf137d581b7f5b57c0d8ebbd08f4c"
)
EXPECTED_BASELINE_COMPATIBILITY_STAGE_COUNTS = {
    "fidelity_repair_deterministic_gate": 2,
    "style_repair_deterministic_fidelity": 12,
    "style_repair_deterministic_style": 4,
}
FIXED_ACQUISITION_FILES = (
    "prompt_contract.json",
    "official_pre_action_reference_bank.jsonl",
    "source_admission_receipt.json",
    "preparation_summary.json",
    "preflight_downstream.json",
    "failures.jsonl",
)


class RecoveryError(RuntimeError):
    """Raised when the partial acquisition or recovery evidence drifts."""


@dataclass(frozen=True)
class RecoveryState:
    acquisition_root: Path
    terminals: Mapping[str, Mapping[str, Any]]
    missing_ids: tuple[str, ...]
    compatibility_ids: tuple[str, ...]
    cache_counts: Mapping[str, int]
    provider_identities: Mapping[str, Any]
    implementation_sha256: str
    run_binding_sha256: str
    reference_sha256: str
    prompt_contract_sha256: str


@dataclass(frozen=True)
class VerifiedRecovery:
    root: Path
    manifest: Mapping[str, Any]
    receipt: Mapping[str, Any]
    state: RecoveryState
    tokenizer_audit_path: Path


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return v5.sha256_text(value)


def sha256_file(path: Path) -> str:
    return v5.sha256_file(path)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RecoveryError(f"missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RecoveryError(f"missing or unsafe {label}: {path}")
    result: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RecoveryError(f"invalid {label} line {number}: {path}") from exc
        if not isinstance(value, dict):
            raise RecoveryError(f"non-object {label} line {number}: {path}")
        result.append(value)
    return result


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(dict(payload)) + "\n"
    if path.is_symlink():
        raise RecoveryError(f"immutable recovery artifact is a symlink: {path}")
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            raise RecoveryError(f"immutable recovery artifact drift: {path}")
        return
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError:
        if path.is_symlink() or path.read_text(encoding="utf-8") != text:
            raise RecoveryError(f"immutable recovery artifact race: {path}")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(canonical_json(dict(row)) + "\n" for row in rows)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise RecoveryError(f"unsafe temporary recovery artifact: {temporary}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _artifact(path: Path, *, relative_to: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RecoveryError(f"unsafe recovery artifact: {path}")
    try:
        relative = path.relative_to(relative_to)
    except ValueError as exc:
        raise RecoveryError(f"recovery artifact escapes root: {path}") from exc
    return {
        "path": relative.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _compatibility_stage_counts(
    terminals: Mapping[str, Mapping[str, Any]], compatibility_ids: Sequence[str]
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(terminals[sample_id].get("rejection_stage"))
                for sample_id in compatibility_ids
            ).items()
        )
    )


def _validate_baseline_compatibility_scope(state: RecoveryState) -> None:
    ids = sorted(state.compatibility_ids)
    if (
        len(ids) != EXPECTED_BASELINE_COMPATIBILITY_COUNT
        or sha256_text(canonical_json(ids)) != EXPECTED_BASELINE_COMPATIBILITY_ID_DIGEST
        or _compatibility_stage_counts(state.terminals, ids)
        != EXPECTED_BASELINE_COMPATIBILITY_STAGE_COUNTS
    ):
        raise RecoveryError("baseline compatibility scope drift")


def _implementation_contract() -> dict[str, Any]:
    path = Path(__file__).resolve()
    artifacts = {
        "recovery_runner_and_verifier": {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(path),
        },
        "legacy_v6_semantics": v6._implementation_contract(),
    }
    return {
        "artifacts": artifacts,
        "composite_sha256": sha256_text(canonical_json(artifacts)),
    }


def _row_map(
    handoff: source_handoff_v1.SourceHandoff,
) -> dict[str, v5.PreparedRow]:
    rows = {
        row.sample_id: row for split in v6.SPLITS for row in handoff.prepared[split]
    }
    if len(rows) != sum(len(handoff.prepared[split]) for split in v6.SPLITS):
        raise RecoveryError("source handoff contains duplicate sample IDs")
    return rows


def _terminal_payload(
    root: Path, row: v5.PreparedRow
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = v5._terminal_path(root, row)
    payload = _read_json(path, "v6 terminal")
    record = payload.get("record")
    if not isinstance(record, dict):
        raise RecoveryError(f"terminal record is invalid: {row.sample_id}")
    return path, payload, record


def _compatibility_kind(record: Mapping[str, Any]) -> str | None:
    generation = record.get("generation")
    if not isinstance(generation, Mapping):
        return None
    repair_history = record.get("repair_history")
    if not isinstance(repair_history, list):
        return None
    fidelity_events = [
        event
        for event in repair_history
        if isinstance(event, Mapping) and event.get("repair_type") == "fidelity"
    ]
    style_events = [
        event
        for event in repair_history
        if isinstance(event, Mapping) and event.get("repair_type") == "style"
    ]
    deterministic = generation.get("deterministic_validation")
    if (
        not isinstance(deterministic, Mapping)
        or deterministic.get("machine_pass") is not False
    ):
        return None
    if (
        record.get("terminal_status") == v5.TERMINAL_GENERATION_REJECT
        and record.get("rejection_stage") == "fidelity_repair_deterministic_gate"
        and generation.get("selected_attempt") == "fidelity_repair"
        and generation.get("fidelity_repair_used") is True
        and generation.get("style_repair_used") is False
        and len(fidelity_events) == 1
        and fidelity_events[0].get("trigger_stage") == "validator_a_primary"
        and not style_events
    ):
        return "fidelity_repair_deterministic_reject"
    if (
        record.get("terminal_status")
        in {v5.TERMINAL_STYLE_REJECT, v5.TERMINAL_STYLE_FIDELITY_REJECT}
        and record.get("rejection_stage")
        in {
            "style_repair_deterministic_style",
            "style_repair_deterministic_fidelity",
        }
        and generation.get("selected_attempt") == "style_repair"
        and generation.get("style_repair_used") is True
        and len(style_events) == 1
        and style_events[0].get("trigger_stage") == "validator_b_primary"
    ):
        return "style_repair_deterministic_reject"
    return None


def _copy_cache_file(
    source_root: Path, target_root: Path, role: str, row: v5.PreparedRow
) -> None:
    source = v5._cache_path(source_root, role, row)
    if source.is_symlink() or not source.is_file():
        raise RecoveryError(f"compatibility sidecar missing: {row.sample_id}:{role}")
    target = v5._cache_path(target_root, role, row)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _validate_repair_cache_binding(
    *,
    cache: Mapping[str, Any],
    role: str,
    row: v5.PreparedRow,
    system_prompt: str,
    user_prompt: str,
    reference_sha256: str | None,
    code_sha256: str,
    config: v5.ProviderConfig,
) -> None:
    projection = v5._request_projection(role, user_prompt)
    expected = {
        "schema_version": v5.CACHE_SCHEMA_VERSION,
        "role": role,
        "sample_id": row.sample_id,
        "split": row.split,
        "source_analysis_sha256": row.source_analysis_sha256,
        "provided_data_sha256": row.provided_data_sha256,
        "official_reference_bank_sha256": reference_sha256,
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(user_prompt),
        "provider_contract_sha256": config.contract_sha256,
        "code_sha256": code_sha256,
        "request_projection_sha256": sha256_text(canonical_json(projection)),
    }
    if (
        cache.get("binding") != expected
        or cache.get("binding_sha256") != sha256_text(canonical_json(expected))
        or cache.get("request_projection") != projection
    ):
        raise RecoveryError(f"repair cache binding drift: {row.sample_id}:{role}")


def _normalized_pre_repair_record(
    *,
    record: Mapping[str, Any],
    prior: v5.Candidate,
    kind: str,
) -> dict[str, Any]:
    normalized = json.loads(canonical_json(record))
    generation = normalized["generation"]
    if kind == "fidelity_repair_deterministic_reject":
        generation["selected_attempt"] = "primary"
        generation["fidelity_repair_used"] = False
        generation["style_repair_used"] = False
        generation["deterministic_validation"] = dict(prior.deterministic_validation)
        generation["provider"].pop(v5.ROLE_REWRITE_FIDELITY_REPAIR, None)
        normalized["repair_history"] = [
            event
            for event in normalized["repair_history"]
            if event.get("repair_type") != "fidelity"
        ]
        normalized["teacher_response_analysis"] = prior.teacher_response_analysis
        normalized["rewritten_minutes"] = prior.rewritten_minutes
        normalized["teacher_response_analysis_sha256"] = sha256_text(
            prior.teacher_response_analysis
        )
        normalized["rewritten_minutes_sha256"] = sha256_text(prior.rewritten_minutes)
        normalized["terminal_status"] = v5.TERMINAL_FIDELITY_REJECT
        normalized["rejection_stage"] = "validator_a_input_fidelity"
        normalized["rejection_reasons"] = list(normalized["validator_a"]["reasons"])
    else:
        generation["selected_attempt"] = prior.attempt
        generation["style_repair_used"] = False
        generation["deterministic_validation"] = dict(prior.deterministic_validation)
        generation["provider"].pop(v5.ROLE_REWRITE_STYLE_REPAIR, None)
        normalized["repair_history"] = [
            event
            for event in normalized["repair_history"]
            if event.get("repair_type") != "style"
        ]
        normalized["terminal_status"] = v5.TERMINAL_STYLE_REJECT
        normalized["rejection_stage"] = "validator_b_style_gate"
        normalized["rejection_reasons"] = list(normalized["validator_b"]["reasons"])
    return normalized


def _load_compatibility_terminal(
    root: Path,
    row: v5.PreparedRow,
    *,
    record: Mapping[str, Any],
    kind: str,
    code_sha256: str,
    reference_sha256: str,
    identity: v5.ProviderIdentityRegistry,
    reference_bank: v5.OfficialReferenceBank | Mapping[str, Any],
    config: v5.ProviderConfig,
    environment: Mapping[str, str] | None,
    source_audit: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    # Validate the actual immutable record before constructing a temporary
    # pre-repair view for the original strict v6 verifier.
    v5._validate_terminal_record(row, record)
    if record.get("source_audit") != source_audit:
        raise RecoveryError(f"compatibility source snapshot drift: {row.sample_id}")
    sidecars = {
        role: provider
        for role, provider in v5._terminal_provider_sidecars(row, record).items()
        if role not in v6.SOURCE_PROVIDER_ROLES
    }
    primary_cache, primary_response = v5._resume_cache_payload(
        output=root,
        row=row,
        role=v5.ROLE_REWRITE_PRIMARY,
        provider=sidecars[v5.ROLE_REWRITE_PRIMARY],
        environment=environment,
    )
    del primary_cache
    try:
        primary: v5.Candidate | None = v5._candidate_from_response(
            row,
            primary_response,
            attempt="primary",
            tokenizer=tokenizer,
            provider_record=sidecars[v5.ROLE_REWRITE_PRIMARY],
        )
    except v5.ContractError:
        primary = None
    repair_history = record.get("repair_history")
    if not isinstance(repair_history, list):
        raise RecoveryError(f"compatibility repair history drift: {row.sample_id}")

    if kind == "fidelity_repair_deterministic_reject":
        if primary is None:
            raise RecoveryError(
                f"A-triggered fidelity repair lacks valid primary: {row.sample_id}"
            )
        events = [
            event for event in repair_history if event.get("repair_type") == "fidelity"
        ]
        if len(events) != 1 or any(
            event.get("repair_type") == "style" for event in repair_history
        ):
            raise RecoveryError(
                f"fidelity compatibility sequence drift: {row.sample_id}"
            )
        event = events[0]
        if (
            event.get("trigger_stage") != "validator_a_primary"
            or event.get("attempt") != "primary"
            or event.get("reason_codes") != record.get("validator_a", {}).get("reasons")
            or event.get("validator_a") != record.get("validator_a", {}).get("result")
            or event.get("provider") != record.get("validator_a", {}).get("provider")
            or record.get("teacher_response_analysis") != ""
            or record.get("rewritten_minutes") != ""
            or record.get("teacher_response_analysis_sha256") is not None
            or record.get("rewritten_minutes_sha256") is not None
        ):
            raise RecoveryError(
                f"fidelity compatibility evidence drift: {row.sample_id}"
            )
        repair_role = v5.ROLE_REWRITE_FIDELITY_REPAIR
        system_prompt = v5.FIDELITY_REPAIR_SYSTEM_PROMPT
        user_prompt = v5._fidelity_repair_user_prompt(
            row,
            event.get("reason_codes") or [],
            candidate=primary,
            validator_a_result=event.get("validator_a"),
        )
        prior = primary
    else:
        events = [
            event for event in repair_history if event.get("repair_type") == "style"
        ]
        if len(events) != 1:
            raise RecoveryError(f"style compatibility sequence drift: {row.sample_id}")
        event = events[0]
        prior_role = (
            v5.ROLE_REWRITE_FIDELITY_REPAIR
            if record.get("generation", {}).get("fidelity_repair_used") is True
            else v5.ROLE_REWRITE_PRIMARY
        )
        if prior_role == v5.ROLE_REWRITE_PRIMARY:
            if primary is None:
                raise RecoveryError(
                    f"style repair lacks valid prior primary: {row.sample_id}"
                )
            prior = primary
        else:
            _cache, response = v5._resume_cache_payload(
                output=root,
                row=row,
                role=prior_role,
                provider=sidecars[prior_role],
                environment=environment,
            )
            prior = v5._candidate_from_response(
                row,
                response,
                attempt="fidelity_repair",
                tokenizer=tokenizer,
                provider_record=sidecars[prior_role],
            )
        snapshot = event.get("validator_b")
        if (
            event.get("trigger_stage") != "validator_b_primary"
            or not isinstance(snapshot, Mapping)
            or not isinstance(snapshot.get("style_feedback"), Mapping)
            or record.get("teacher_response_analysis")
            != prior.teacher_response_analysis
            or record.get("rewritten_minutes") != prior.rewritten_minutes
            or record.get("teacher_response_analysis_sha256")
            != sha256_text(prior.teacher_response_analysis)
            or record.get("rewritten_minutes_sha256")
            != sha256_text(prior.rewritten_minutes)
            or event.get("provider") != record.get("validator_b", {}).get("provider")
        ):
            raise RecoveryError(f"style compatibility evidence drift: {row.sample_id}")
        terminal_a = record.get("validator_a", {})
        event_a = event.get("validator_a")
        terminal_a_providers = terminal_a.get("provider")
        selected_a_role = (
            v5.ROLE_VALIDATOR_A_FIDELITY_REPAIR
            if prior.attempt == "fidelity_repair"
            else v5.ROLE_VALIDATOR_A_PRIMARY
        )
        expected_event_a_providers: dict[str, Any] = {}
        if (
            isinstance(terminal_a_providers, Mapping)
            and selected_a_role in terminal_a_providers
        ):
            expected_event_a_providers[selected_a_role] = terminal_a_providers[
                selected_a_role
            ]
            contract_role = v5.VALIDATOR_A_CONTRACT_REPAIR_ROLES[selected_a_role]
            if contract_role in terminal_a_providers:
                expected_event_a_providers[contract_role] = terminal_a_providers[
                    contract_role
                ]
        if (
            not isinstance(event_a, Mapping)
            or event_a.get("complete") is not True
            or event_a.get("machine_pass") is not True
            or event_a.get("result") != terminal_a.get("result")
            or event_a.get("reasons") != terminal_a.get("reasons")
            or event_a.get("provider") != expected_event_a_providers
            or event.get("attempt") != prior.attempt
            or event.get("reason_codes") != record.get("validator_b", {}).get("reasons")
        ):
            raise RecoveryError(
                f"style compatibility A snapshot drift: {row.sample_id}"
            )
        terminal_b = record.get("validator_b", {})
        result_b = terminal_b.get("result")
        if (
            not isinstance(result_b, Mapping)
            or snapshot.get("machine_pass") != terminal_b.get("machine_pass")
            or snapshot.get("mean_score") != terminal_b.get("mean_score")
            or snapshot.get("min_score") != terminal_b.get("min_score")
            or snapshot.get("style_feedback") != v5._safe_style_feedback(result_b)
            or snapshot.get("contract_repair_used")
            != result_b.get("contract_repair_used")
            or snapshot.get("contract_repair") != result_b.get("contract_repair")
        ):
            raise RecoveryError(
                f"style compatibility B snapshot drift: {row.sample_id}"
            )
        repair_role = v5.ROLE_REWRITE_STYLE_REPAIR
        system_prompt = v5.STYLE_REPAIR_SYSTEM_PROMPT
        user_prompt = v5._prompt_payload(
            "Apply one style-only repair",
            {
                "source_analysis": row.source_analysis,
                "current_rewritten_minutes": prior.rewritten_minutes,
                "style_feedback": dict(snapshot["style_feedback"]),
            },
        )

    if repair_role not in sidecars:
        raise RecoveryError(
            f"repair provider sidecar missing: {row.sample_id}:{repair_role}"
        )
    forbidden_followups = {
        v5.ROLE_VALIDATOR_A_STYLE_REPAIR,
        v5.VALIDATOR_A_CONTRACT_REPAIR_ROLES[v5.ROLE_VALIDATOR_A_STYLE_REPAIR],
        v5.ROLE_VALIDATOR_B_STYLE_REPAIR,
        v5.VALIDATOR_B_CONTRACT_REPAIR_ROLES[v5.ROLE_VALIDATOR_B_STYLE_REPAIR],
    }
    if kind.startswith("style") and forbidden_followups.intersection(sidecars):
        raise RecoveryError(
            f"deterministic style reject has follow-up calls: {row.sample_id}"
        )

    repair_cache, repair_response = v5._resume_cache_payload(
        output=root,
        row=row,
        role=repair_role,
        provider=sidecars[repair_role],
        environment=environment,
    )
    _validate_repair_cache_binding(
        cache=repair_cache,
        role=repair_role,
        row=row,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        reference_sha256=None,
        code_sha256=code_sha256,
        config=config,
    )
    try:
        v5._candidate_from_response(
            row,
            repair_response,
            attempt="repair",
            tokenizer=tokenizer,
            provider_record=sidecars[repair_role],
        )
    except v5.ContractError:
        pass
    else:
        raise RecoveryError(f"compatibility repair became valid: {row.sample_id}")
    diagnostics = v5._candidate_failure_diagnostics(
        row, repair_response, tokenizer=tokenizer
    )
    generation = record["generation"]
    if diagnostics != generation.get("deterministic_validation") or record.get(
        "rejection_reasons"
    ) != diagnostics.get("reasons"):
        raise RecoveryError(f"compatibility repair diagnostics drift: {row.sample_id}")
    if kind == "style_repair_deterministic_reject":
        expected_status, expected_stage = v5._style_repair_deterministic_rejection(
            diagnostics
        )
        if (
            record.get("terminal_status") != expected_status
            or record.get("rejection_stage") != expected_stage
        ):
            raise RecoveryError(f"style compatibility verdict drift: {row.sample_id}")
    else:
        v5._raise_if_unresolved_rewrite_failure(diagnostics, role=repair_role)

    normalized = _normalized_pre_repair_record(record=record, prior=prior, kind=kind)
    v5._validate_terminal_record(row, normalized)
    normalized_sidecars = {
        role: provider
        for role, provider in v5._terminal_provider_sidecars(row, normalized).items()
        if role not in v6.SOURCE_PROVIDER_ROLES
    }
    if set(sidecars) != set(normalized_sidecars) | {repair_role}:
        raise RecoveryError(f"compatibility invocation sequence drift: {row.sample_id}")

    with tempfile.TemporaryDirectory(prefix="paper_chk2_v6_compat_") as temporary:
        temp_root = Path(temporary)
        for role in normalized_sidecars:
            _copy_cache_file(root, temp_root, role, row)
        terminal_path = v5._terminal_path(temp_root, row)
        terminal_path.parent.mkdir(parents=True, exist_ok=True)
        terminal_payload = {
            "binding": v5._terminal_binding(row, code_sha256, reference_sha256),
            "record": normalized,
            "record_sha256": sha256_text(canonical_json(normalized)),
        }
        terminal_path.write_text(
            canonical_json(terminal_payload) + "\n", encoding="utf-8"
        )
        replayed = v6._load_downstream_terminal(
            temp_root,
            row,
            code_sha256=code_sha256,
            official_reference_bank_sha256=reference_sha256,
            identity=identity,
            reference_bank=reference_bank,
            config=config,
            environment=environment,
            source_audit=source_audit,
            tokenizer=tokenizer,
        )
    if replayed != normalized:
        raise RecoveryError(f"normalized compatibility replay drift: {row.sample_id}")
    identity.bind(repair_role, repair_response)
    return dict(record)


def load_terminal_compat(
    root: str | Path,
    row: v5.PreparedRow,
    *,
    code_sha256: str,
    official_reference_bank_sha256: str,
    identity: v5.ProviderIdentityRegistry,
    reference_bank: v5.OfficialReferenceBank | Mapping[str, Any],
    config: v5.ProviderConfig,
    environment: Mapping[str, str] | None,
    source_audit: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any] | None:
    """Replay one terminal, using compatibility only for two exact v6 cases."""
    output = Path(root).resolve()
    path = v5._terminal_path(output, row)
    if not path.is_file():
        return None
    _path, payload, record = _terminal_payload(output, row)
    if payload.get("binding") != v5._terminal_binding(
        row, code_sha256, official_reference_bank_sha256
    ):
        raise RecoveryError(f"terminal compatibility binding drift: {row.sample_id}")
    if payload.get("record_sha256") != sha256_text(canonical_json(record)):
        raise RecoveryError(
            f"terminal compatibility record hash drift: {row.sample_id}"
        )
    kind = _compatibility_kind(record)
    if kind is None:
        # No catch-and-fallback: all ordinary records remain under the exact v6
        # verifier and retain its fail-closed behavior.
        return v6._load_downstream_terminal(
            output,
            row,
            code_sha256=code_sha256,
            official_reference_bank_sha256=official_reference_bank_sha256,
            identity=identity,
            reference_bank=reference_bank,
            config=config,
            environment=environment,
            source_audit=source_audit,
            tokenizer=tokenizer,
        )
    return _load_compatibility_terminal(
        output,
        row,
        record=record,
        kind=kind,
        code_sha256=code_sha256,
        reference_sha256=official_reference_bank_sha256,
        identity=identity,
        reference_bank=reference_bank,
        config=config,
        environment=environment,
        source_audit=source_audit,
        tokenizer=tokenizer,
    )


def _validate_loose_provider_cache(
    path: Path,
    *,
    role: str,
    row: v5.PreparedRow,
    code_sha256: str,
    reference_sha256: str,
    config: v5.ProviderConfig,
    identity: v5.ProviderIdentityRegistry,
) -> None:
    payload = _read_json(path, f"partial provider cache {role}")
    binding = payload.get("binding")
    projection = payload.get("request_projection")
    expected_reference = reference_sha256 if role.startswith("validator_b") else None
    if (
        not isinstance(binding, Mapping)
        or binding.get("schema_version") != v5.CACHE_SCHEMA_VERSION
        or binding.get("role") != role
        or binding.get("sample_id") != row.sample_id
        or binding.get("split") != row.split
        or binding.get("source_analysis_sha256") != row.source_analysis_sha256
        or binding.get("provided_data_sha256") != row.provided_data_sha256
        or binding.get("official_reference_bank_sha256") != expected_reference
        or binding.get("provider_contract_sha256") != config.contract_sha256
        or binding.get("code_sha256") != code_sha256
        or not isinstance(projection, Mapping)
        or binding.get("request_projection_sha256")
        != sha256_text(canonical_json(projection))
        or payload.get("binding_sha256") != sha256_text(canonical_json(binding))
    ):
        raise RecoveryError(
            f"partial provider cache binding drift: {row.sample_id}:{role}"
        )
    raw = payload.get("provider_response")
    if not isinstance(raw, dict):
        raise RecoveryError(
            f"partial provider response missing: {row.sample_id}:{role}"
        )
    response = v5.ProviderResponse.from_dict(raw)
    if payload.get("raw_reasoning_sha256") != sha256_text(
        response.raw_reasoning
    ) or payload.get("raw_content_sha256") != sha256_text(response.raw_content):
        raise RecoveryError(
            f"partial provider response hash drift: {row.sample_id}:{role}"
        )
    v5._validate_provider_response_has_no_credentials(response, {})
    identity.bind(role, response)


def _validate_historical_failure_ledger(
    failures: Sequence[Mapping[str, Any]],
    *,
    row_ids: Sequence[str],
    missing_ids: Sequence[str],
    expected_failure_ids: Sequence[str] | None,
) -> None:
    """Validate the sealed v6 failure ledger without treating it as mutable state.

    Before sealing, the ledger must equal the then-current missing population.
    After sealing, it remains immutable evidence of the baseline failures while
    recovery may reduce the current missing population to any subset, including
    the empty set.
    """
    failure_ids = [item.get("sample_id") for item in failures]
    expected = (
        set(missing_ids) if expected_failure_ids is None else set(expected_failure_ids)
    )
    if (
        len(failure_ids) != len(set(failure_ids))
        or set(failure_ids) != expected
        or not set(missing_ids).issubset(expected)
        or not expected.issubset(row_ids)
        or any(
            item.get("error_type") not in {"ProviderRequestError", "ContractError"}
            for item in failures
        )
    ):
        raise RecoveryError("partial v6 failure/missing partition drift")


def inspect_acquisition(
    acquisition_root: str | Path,
    *,
    handoff: source_handoff_v1.SourceHandoff,
    tokenizer: Any,
    identity: v5.ProviderIdentityRegistry | None = None,
    expected_failure_ids: Sequence[str] | None = None,
) -> RecoveryState:
    """Replay a partial or completed acquisition without provider calls."""
    root = Path(acquisition_root).resolve()
    rows = _row_map(handoff)
    implementation = v6._implementation_contract()
    handoff_sha = v6._handoff_sha(handoff)
    run_binding = v6._run_binding_sha(
        str(implementation["composite_sha256"]), handoff_sha
    )
    config = v5.ProviderConfig()
    prompt = _read_json(root / "prompt_contract.json", "v6 prompt contract")
    expected_prompt = v6._prompt_contract(
        implementation=implementation,
        handoff_sha256=handoff_sha,
        run_binding_sha256=run_binding,
        config=config,
    )
    if prompt != expected_prompt:
        raise RecoveryError("partial v6 prompt contract drift")
    reference_path = root / "official_pre_action_reference_bank.jsonl"
    sealed_reference = (
        handoff.root / "sealed_source/official_pre_action_reference_bank.jsonl"
    )
    if (
        reference_path.is_symlink()
        or not reference_path.is_file()
        or sealed_reference.is_symlink()
        or not sealed_reference.is_file()
        or sha256_file(reference_path) != sha256_file(sealed_reference)
    ):
        raise RecoveryError("partial v6 official reference drift")
    reference_sha = sha256_file(reference_path)
    reference_bank = official_v2.deserialize_official_reference_bank(
        reference_path.read_bytes()
    )
    source_receipt = _read_json(
        root / "source_admission_receipt.json", "source receipt"
    )
    expected_source_receipt = {
        "schema_version": "paper-chk2-downstream-imported-source-v1",
        "status": "source_admission_complete",
        "quality_status": "passed",
        "source_handoff_manifest_sha256": handoff_sha,
        "source_rows": len(handoff.source_results),
        "admitted_rows": sum(
            result.get("machine_pass") is True
            for result in handoff.source_results.values()
        ),
        "unresolved_rows": 0,
        "status_counts": {
            split: dict(Counter(row["status"] for row in handoff.records[split]))
            for split in v6.SPLITS
        },
    }
    if source_receipt != expected_source_receipt:
        raise RecoveryError("partial v6 source receipt drift")
    preparation = _read_json(root / "preparation_summary.json", "preparation summary")
    if preparation != {
        "schema_version": "paper-chk2-downstream-preparation-v1",
        "source": {
            "source_handoff_manifest_sha256": handoff_sha,
            "training_only": True,
        },
    }:
        raise RecoveryError("partial v6 preparation summary drift")

    identity = identity or v5.ProviderIdentityRegistry()
    terminals: dict[str, Mapping[str, Any]] = {}
    compatibility_ids: list[str] = []
    for split in v6.SPLITS:
        for row in handoff.prepared[split]:
            terminal = load_terminal_compat(
                root,
                row,
                code_sha256=run_binding,
                official_reference_bank_sha256=reference_sha,
                identity=identity,
                reference_bank=reference_bank,
                config=config,
                environment={},
                source_audit=handoff.source_results[row.sample_id],
                tokenizer=tokenizer,
            )
            if terminal is None:
                continue
            terminals[row.sample_id] = terminal
            if _compatibility_kind(terminal) is not None:
                compatibility_ids.append(row.sample_id)
    missing = tuple(sample_id for sample_id in rows if sample_id not in terminals)
    if any(
        handoff.source_results[sample_id].get("machine_pass") is not True
        for sample_id in missing
    ):
        raise RecoveryError("partial v6 is missing a source-reject terminal")

    terminal_root = root / "cache/terminal"
    observed_terminal_files = (
        {path for path in terminal_root.glob("*.json")}
        if terminal_root.is_dir()
        else set()
    )
    expected_terminal_files = {
        v5._terminal_path(root, rows[sample_id]) for sample_id in terminals
    }
    if observed_terminal_files != expected_terminal_files:
        raise RecoveryError("partial v6 terminal inventory drift")

    expected_provider_paths: set[Path] = set()
    for sample_id, terminal in terminals.items():
        row = rows[sample_id]
        for role in v5._terminal_provider_sidecars(row, terminal):
            if role not in v6.SOURCE_PROVIDER_ROLES:
                expected_provider_paths.add(v5._cache_path(root, role, row))
    allowed_roles = set(v5.PROVIDER_ROLES) - set(v6.SOURCE_PROVIDER_ROLES)
    observed_provider_paths: set[Path] = set()
    cache_counts: dict[str, int] = {"terminal": len(observed_terminal_files)}
    cache_root = root / "cache"
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise RecoveryError("partial v6 cache root is missing or unsafe")
    for role_root in cache_root.iterdir():
        if role_root.is_symlink() or not role_root.is_dir():
            raise RecoveryError(f"unsafe cache namespace: {role_root}")
        role = role_root.name
        if role == "terminal":
            continue
        if role not in allowed_roles:
            raise RecoveryError(f"forbidden cache namespace in recovery: {role}")
        files = list(role_root.iterdir())
        if any(
            path.is_symlink() or not path.is_file() or path.suffix != ".json"
            for path in files
        ):
            raise RecoveryError(f"unsafe cache artifact in namespace: {role}")
        cache_counts[role] = len(files)
        for path in files:
            payload = _read_json(path, f"provider cache {role}")
            binding = payload.get("binding")
            sample_id = (
                binding.get("sample_id") if isinstance(binding, Mapping) else None
            )
            if not isinstance(sample_id, str) or sample_id not in rows:
                raise RecoveryError(f"provider cache sample drift: {path}")
            row = rows[sample_id]
            if path.name != f"{sha256_text(sample_id)}.json":
                raise RecoveryError(f"provider cache filename drift: {path}")
            observed_provider_paths.add(path)
            if path not in expected_provider_paths:
                if sample_id not in missing:
                    raise RecoveryError(
                        f"orphan provider cache for terminal row: {path}"
                    )
                _validate_loose_provider_cache(
                    path,
                    role=role,
                    row=row,
                    code_sha256=run_binding,
                    reference_sha256=reference_sha,
                    config=config,
                    identity=identity,
                )
    if not expected_provider_paths.issubset(observed_provider_paths):
        raise RecoveryError("partial v6 is missing a terminal provider cache")

    failure_path = root / "failures.jsonl"
    failures = _read_jsonl(failure_path, "initial transport failures")
    _validate_historical_failure_ledger(
        failures,
        row_ids=tuple(rows),
        missing_ids=missing,
        expected_failure_ids=expected_failure_ids,
    )

    preflight = _read_json(root / "preflight_downstream.json", "v6 preflight")
    split_by_id = {sample_id: row.split for sample_id, row in rows.items()}
    v6._validate_preflight_header(
        preflight, handoff_sha256=handoff_sha, split_by_id=split_by_id
    )
    selected: list[str] = []
    attempted: list[str] = []
    for attempt in preflight["attempts"]:
        if not isinstance(attempt, Mapping):
            raise RecoveryError("preflight attempt shape drift")
        sample_id = attempt.get("sample_id")
        terminal = terminals.get(str(sample_id))
        if (
            not isinstance(sample_id, str)
            or sample_id in attempted
            or terminal is None
            or attempt.get("split") != rows[sample_id].split
            or attempt.get("terminal_status") != terminal.get("terminal_status")
        ):
            raise RecoveryError("preflight attempt replay drift")
        passed = (
            terminal.get("terminal_status") == v5.TERMINAL_PASS
            and terminal.get("training_pass") is True
        )
        if attempt.get("selected") is not passed:
            raise RecoveryError("preflight selection replay drift")
        attempted.append(sample_id)
        if passed:
            selected.append(sample_id)
    if selected != preflight.get("selected_ids"):
        raise RecoveryError("preflight selected-ID drift")

    return RecoveryState(
        acquisition_root=root,
        terminals=terminals,
        missing_ids=missing,
        compatibility_ids=tuple(compatibility_ids),
        cache_counts=dict(sorted(cache_counts.items())),
        provider_identities=identity.as_dict(),
        implementation_sha256=str(implementation["composite_sha256"]),
        run_binding_sha256=run_binding,
        reference_sha256=reference_sha,
        prompt_contract_sha256=sha256_file(root / "prompt_contract.json"),
    )


def _partial_file_paths(root: Path) -> list[Path]:
    paths = [root / name for name in FIXED_ACQUISITION_FILES]
    cache = root / "cache"
    if cache.is_symlink() or not cache.is_dir():
        raise RecoveryError("partial cache tree is missing or unsafe")
    for path in cache.rglob("*"):
        if path.is_symlink():
            raise RecoveryError(f"partial cache tree contains symlink: {path}")
        if path.is_file():
            if path.suffix != ".json":
                raise RecoveryError(
                    f"partial cache tree contains non-JSON file: {path}"
                )
            paths.append(path)
        elif not path.is_dir():
            raise RecoveryError(f"partial cache tree contains unsafe entry: {path}")
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise RecoveryError("partial acquisition has a missing or unsafe required file")
    return sorted(set(paths), key=lambda path: path.relative_to(root).as_posix())


def _signed_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["manifest_sha256"] = sha256_text(canonical_json(result))
    return result


def _validate_signed_manifest(manifest: Mapping[str, Any]) -> None:
    unsigned = dict(manifest)
    stored = unsigned.pop("manifest_sha256", None)
    if stored != sha256_text(canonical_json(unsigned)):
        raise RecoveryError("partial handoff manifest signature drift")


def seal_partial_acquisition(
    original_root: str | Path,
    recovery_root: str | Path,
    *,
    handoff: source_handoff_v1.SourceHandoff,
    tokenizer: Any,
) -> Mapping[str, Any]:
    """Verify and byte-copy the partial v6 acquisition into a fresh root."""
    original = Path(original_root).resolve()
    output = Path(recovery_root).resolve()
    if original == output or original in output.parents or output in original.parents:
        raise RecoveryError("original and recovery roots must be independent")
    if output.exists():
        return verify_partial_handoff(output, handoff=handoff, tokenizer=tokenizer)
    state = inspect_acquisition(original, handoff=handoff, tokenizer=tokenizer)
    _validate_baseline_compatibility_scope(state)
    files = _partial_file_paths(original)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    )
    try:
        acquisition = staging / "acquisition"
        for source in files:
            relative = source.relative_to(original)
            target = acquisition / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        inventory = [
            _artifact(acquisition / path.relative_to(original), relative_to=staging)
            for path in files
        ]
        manifest = _signed_manifest(
            {
                "schema_version": PARTIAL_MANIFEST_SCHEMA,
                "status": "sealed_partial_acquisition",
                "original_v6_root": _display_path(original),
                "recovery_implementation": _implementation_contract(),
                "legacy_v6_implementation_sha256": state.implementation_sha256,
                "legacy_v6_run_binding_sha256": state.run_binding_sha256,
                "source_handoff_manifest_sha256": v6._handoff_sha(handoff),
                "source_handoff_manifest_file_sha256": sha256_file(
                    handoff.root / "handoff_manifest.json"
                ),
                "official_reference_bank_sha256": state.reference_sha256,
                "prompt_contract_sha256": state.prompt_contract_sha256,
                "baseline_terminal_count": len(state.terminals),
                "baseline_terminal_ids": sorted(state.terminals),
                "baseline_missing_ids": list(state.missing_ids),
                "compatibility_ids": sorted(state.compatibility_ids),
                "cache_counts": dict(state.cache_counts),
                "provider_identities": dict(state.provider_identities),
                "inventory": inventory,
                "inventory_sha256": sha256_text(canonical_json(inventory)),
            }
        )
        _write_immutable_json(staging / "partial_handoff_manifest.json", manifest)
        # Validate the complete copied snapshot before making the target name
        # visible.  A bad copy can therefore never poison the recovery path.
        verify_partial_handoff(staging, handoff=handoff, tokenizer=tokenizer)
        if output.exists() or output.is_symlink():
            raise RecoveryError(f"recovery root appeared during seal: {output}")
        os.rename(staging, output)
    except Exception:
        if staging.exists() and staging.parent == output.parent:
            shutil.rmtree(staging)
        raise
    return verify_partial_handoff(output, handoff=handoff, tokenizer=tokenizer)


def verify_partial_handoff(
    recovery_root: str | Path,
    *,
    handoff: source_handoff_v1.SourceHandoff,
    tokenizer: Any,
) -> Mapping[str, Any]:
    root = Path(recovery_root).resolve()
    manifest = _read_json(root / "partial_handoff_manifest.json", "partial handoff")
    _validate_signed_manifest(manifest)
    if (
        manifest.get("schema_version") != PARTIAL_MANIFEST_SCHEMA
        or manifest.get("status") != "sealed_partial_acquisition"
        or manifest.get("recovery_implementation") != _implementation_contract()
        or manifest.get("source_handoff_manifest_sha256") != v6._handoff_sha(handoff)
        or manifest.get("source_handoff_manifest_file_sha256")
        != sha256_file(handoff.root / "handoff_manifest.json")
    ):
        raise RecoveryError("partial handoff header drift")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list) or manifest.get(
        "inventory_sha256"
    ) != sha256_text(canonical_json(inventory)):
        raise RecoveryError("partial handoff inventory signature drift")
    baseline_paths: set[Path] = set()
    for descriptor in inventory:
        if not isinstance(descriptor, Mapping):
            raise RecoveryError("partial handoff inventory shape drift")
        relative = descriptor.get("path")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise RecoveryError("partial handoff inventory path drift")
        path = root / relative
        baseline_paths.add(path)
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != descriptor.get("bytes")
            or sha256_file(path) != descriptor.get("sha256")
        ):
            raise RecoveryError(f"sealed partial artifact drift: {path}")
    acquisition = root / "acquisition"
    baseline_missing_for_replay = manifest.get("baseline_missing_ids")
    if not isinstance(baseline_missing_for_replay, list):
        raise RecoveryError("partial handoff baseline-missing ledger drift")
    state = inspect_acquisition(
        acquisition,
        handoff=handoff,
        tokenizer=tokenizer,
        expected_failure_ids=baseline_missing_for_replay,
    )
    baseline_ids = manifest.get("baseline_terminal_ids")
    baseline_missing = manifest.get("baseline_missing_ids")
    if (
        not isinstance(baseline_ids, list)
        or not isinstance(baseline_missing, list)
        or set(baseline_ids).intersection(baseline_missing)
        or set(baseline_ids).union(baseline_missing) != set(_row_map(handoff))
        or not set(state.missing_ids).issubset(set(baseline_missing))
        or manifest.get("compatibility_ids")
        != sorted(
            sample_id
            for sample_id in baseline_ids
            if sample_id in set(state.compatibility_ids)
        )
        or manifest.get("legacy_v6_implementation_sha256")
        != state.implementation_sha256
        or manifest.get("legacy_v6_run_binding_sha256") != state.run_binding_sha256
        or manifest.get("official_reference_bank_sha256") != state.reference_sha256
        or manifest.get("prompt_contract_sha256") != state.prompt_contract_sha256
    ):
        raise RecoveryError("partial handoff population/binding drift")
    baseline_compatibility_ids = manifest.get("compatibility_ids")
    if (
        not isinstance(baseline_compatibility_ids, list)
        or len(baseline_compatibility_ids) != EXPECTED_BASELINE_COMPATIBILITY_COUNT
        or sha256_text(canonical_json(sorted(baseline_compatibility_ids)))
        != EXPECTED_BASELINE_COMPATIBILITY_ID_DIGEST
        or _compatibility_stage_counts(state.terminals, baseline_compatibility_ids)
        != EXPECTED_BASELINE_COMPATIBILITY_STAGE_COUNTS
    ):
        raise RecoveryError("sealed baseline compatibility scope drift")
    current_files = set(_partial_file_paths(acquisition))
    new_files = current_files - baseline_paths
    rows = _row_map(handoff)
    for path in new_files:
        if path.parent.name not in set(v5.PROVIDER_ROLES) | {"terminal"}:
            raise RecoveryError(f"unexpected post-seal acquisition artifact: {path}")
        try:
            sample_id = next(
                sample_id
                for sample_id in baseline_missing
                if path.name == f"{sha256_text(sample_id)}.json"
            )
        except StopIteration as exc:
            raise RecoveryError(
                f"post-seal artifact is not a missing row: {path}"
            ) from exc
        if sample_id not in rows:
            raise RecoveryError(f"post-seal artifact has unknown sample: {path}")
    return manifest


def verify_original_v6_unchanged(
    original_root: str | Path, manifest: Mapping[str, Any]
) -> None:
    """Prove that every sealed protocol artifact still has identical bytes."""
    original = Path(original_root).resolve()
    if manifest.get("original_v6_root") != _display_path(original):
        raise RecoveryError("original v6 root identity drift")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, list):
        raise RecoveryError("original v6 inventory is missing")
    expected_relative: set[Path] = set()
    for descriptor in inventory:
        if not isinstance(descriptor, Mapping):
            raise RecoveryError("original v6 inventory shape drift")
        recovery_relative = Path(str(descriptor.get("path")))
        if not recovery_relative.parts or recovery_relative.parts[0] != "acquisition":
            raise RecoveryError("original v6 inventory prefix drift")
        relative = Path(*recovery_relative.parts[1:])
        expected_relative.add(relative)
        path = original / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != descriptor.get("bytes")
            or sha256_file(path) != descriptor.get("sha256")
        ):
            raise RecoveryError(f"original v6 protocol artifact drift: {path}")
    observed_relative = {
        path.relative_to(original) for path in _partial_file_paths(original)
    }
    if observed_relative != expected_relative:
        raise RecoveryError("original v6 protocol inventory set drift")


def _attempt_paths(root: Path) -> list[Path]:
    attempt_root = root / "attempts"
    if not attempt_root.exists():
        return []
    if attempt_root.is_symlink() or not attempt_root.is_dir():
        raise RecoveryError("unsafe recovery attempt directory")
    paths = sorted(attempt_root.iterdir())
    if any(
        path.is_symlink() or not path.is_file() or path.suffix != ".json"
        for path in paths
    ):
        raise RecoveryError("unsafe recovery attempt artifact")
    return paths


def _load_attempts(
    root: Path, manifest: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], set[str]]:
    baseline_missing = set(manifest["baseline_missing_ids"])
    unresolved = set(baseline_missing)
    attempts: list[dict[str, Any]] = []
    for index, path in enumerate(_attempt_paths(root), 1):
        if path.name != f"attempt-{index:04d}.json":
            raise RecoveryError("recovery attempt sequence drift")
        attempt = _read_json(path, "recovery attempt")
        unsigned = dict(attempt)
        stored = unsigned.pop("attempt_sha256", None)
        requested = attempt.get("requested_ids")
        completed = attempt.get("completed_ids")
        failures = attempt.get("failures")
        kind = attempt.get("kind")
        if (
            stored != sha256_text(canonical_json(unsigned))
            or attempt.get("schema_version") != SCHEMA_VERSION
            or attempt.get("attempt_index") != index
            or attempt.get("partial_handoff_manifest_sha256")
            != manifest["manifest_sha256"]
            or kind not in {"provider_attempt", "interruption_reconciliation"}
            or not isinstance(requested, list)
            or not isinstance(completed, list)
            or not isinstance(failures, list)
            or len(requested) != len(set(requested))
            or len(completed) != len(set(completed))
            or not set(completed).issubset(set(requested))
            or not set(requested).issubset(unresolved)
        ):
            raise RecoveryError(f"recovery attempt receipt drift: {path}")
        failure_ids = [
            item.get("sample_id") for item in failures if isinstance(item, Mapping)
        ]
        if (
            len(failure_ids) != len(failures)
            or len(failure_ids) != len(set(failure_ids))
            or set(completed).intersection(failure_ids)
            or set(completed).union(failure_ids) != set(requested)
            or (kind == "provider_attempt" and set(requested) != unresolved)
            or (kind == "interruption_reconciliation" and failures)
        ):
            raise RecoveryError(f"recovery attempt partition drift: {path}")
        unresolved.difference_update(completed)
        attempts.append(attempt)
    return attempts, baseline_missing - unresolved


def _record_attempt(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    kind: str,
    requested_ids: Sequence[str],
    completed_ids: Sequence[str],
    failures: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    existing, _completed = _load_attempts(root, manifest)
    index = len(existing) + 1
    attempt = {
        "schema_version": SCHEMA_VERSION,
        "attempt_index": index,
        "kind": kind,
        "configured_concurrency": (
            RECOVERY_CONCURRENCY if kind == "provider_attempt" else 0
        ),
        "partial_handoff_manifest_sha256": manifest["manifest_sha256"],
        "requested_ids": sorted(requested_ids),
        "completed_ids": sorted(completed_ids),
        "failures": [dict(item) for item in failures],
    }
    attempt["attempt_sha256"] = sha256_text(canonical_json(attempt))
    _write_immutable_json(root / f"attempts/attempt-{index:04d}.json", attempt)
    _load_attempts(root, manifest)
    return attempt


def _reconcile_interrupted_terminals(
    root: Path,
    manifest: Mapping[str, Any],
    state: RecoveryState,
) -> None:
    _attempts, ledger_completed = _load_attempts(root, manifest)
    baseline_terminals = set(manifest["baseline_terminal_ids"])
    observed_added = set(state.terminals) - baseline_terminals
    unledgered = sorted(observed_added - ledger_completed)
    if unledgered:
        _record_attempt(
            root,
            manifest,
            kind="interruption_reconciliation",
            requested_ids=unledgered,
            completed_ids=unledgered,
            failures=[],
        )


def _recovery_receipt(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    state: RecoveryState,
    tokenizer_audit: Path,
) -> dict[str, Any]:
    baseline_missing = manifest["baseline_missing_ids"]
    attempts, completed = _load_attempts(root, manifest)
    if completed != set(baseline_missing):
        raise RecoveryError("recovery attempt ledger is incomplete")
    attempt_artifacts = [
        _artifact(path, relative_to=root) for path in _attempt_paths(root)
    ]
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "complete",
        "configured_concurrency": RECOVERY_CONCURRENCY,
        "recovery_implementation": _implementation_contract(),
        "partial_handoff_manifest_sha256": manifest["manifest_sha256"],
        "partial_handoff_manifest_file_sha256": sha256_file(
            root / "partial_handoff_manifest.json"
        ),
        "legacy_v6_implementation_sha256": state.implementation_sha256,
        "legacy_v6_run_binding_sha256": state.run_binding_sha256,
        "source_handoff_manifest_sha256": manifest["source_handoff_manifest_sha256"],
        "official_reference_bank_sha256": state.reference_sha256,
        "prompt_contract_sha256": state.prompt_contract_sha256,
        "baseline_terminal_count": manifest["baseline_terminal_count"],
        "recovered_terminal_count": len(baseline_missing),
        "recovered_ids": list(baseline_missing),
        "attempt_count": len(attempts),
        "attempt_artifacts": attempt_artifacts,
        "final_terminal_count": len(state.terminals),
        "final_missing_ids": [],
        "compatibility_ids": list(state.compatibility_ids),
        "terminal_status_counts": dict(
            sorted(
                Counter(
                    row["terminal_status"] for row in state.terminals.values()
                ).items()
            )
        ),
        "cache_counts": dict(state.cache_counts),
        "provider_identities": dict(state.provider_identities),
        "tokenizer_replay": _artifact(tokenizer_audit, relative_to=root),
        "original_v6_protocol_inventory_sha256": manifest["inventory_sha256"],
        "original_v6_protocol_inventory_verified": True,
    }
    receipt["receipt_sha256"] = sha256_text(canonical_json(receipt))
    return receipt


def _verify_complete_receipt(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    state: RecoveryState,
    original_root: str | Path,
) -> Mapping[str, Any]:
    verify_original_v6_unchanged(original_root, manifest)
    receipt = _read_json(root / "recovery_receipt.json", "recovery receipt")
    unsigned = dict(receipt)
    stored = unsigned.pop("receipt_sha256", None)
    if stored != sha256_text(canonical_json(unsigned)):
        raise RecoveryError("recovery receipt signature drift")
    tokenizer_audit = root / "audits/tokenizer_replay.jsonl"
    expected = _recovery_receipt(
        root=root,
        manifest=manifest,
        state=state,
        tokenizer_audit=tokenizer_audit,
    )
    if receipt != expected:
        raise RecoveryError("recovery receipt semantic drift")
    return receipt


def _expected_tokenizer_rows(
    state: RecoveryState,
    *,
    handoff: source_handoff_v1.SourceHandoff,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in v6.SPLITS:
        for row in handoff.prepared[split]:
            terminal = state.terminals[row.sample_id]
            if terminal.get("terminal_status") != v5.TERMINAL_PASS:
                continue
            replay = v5._tokenizer_replay(
                row=row, response=terminal["sft_response"], tokenizer=tokenizer
            )
            if replay != v5._preflight_tokenizer_receipt(terminal):
                raise RecoveryError(f"recovery tokenizer replay drift: {row.sample_id}")
            rows.append({"sample_id": row.sample_id, "split": split, **replay})
    return rows


def _finalize_recovery(
    root: Path,
    *,
    original_root: str | Path,
    manifest: Mapping[str, Any],
    state: RecoveryState,
    handoff: source_handoff_v1.SourceHandoff,
    tokenizer: Any,
) -> Mapping[str, Any]:
    if state.missing_ids:
        raise RecoveryError("cannot finalize recovery with missing terminals")
    verify_original_v6_unchanged(original_root, manifest)
    expected_rows = _expected_tokenizer_rows(
        state, handoff=handoff, tokenizer=tokenizer
    )
    tokenizer_audit = root / "audits/tokenizer_replay.jsonl"
    if tokenizer_audit.is_file():
        if _read_jsonl(tokenizer_audit, "recovery tokenizer audit") != expected_rows:
            raise RecoveryError("recovery tokenizer audit drift")
    else:
        _write_jsonl(tokenizer_audit, expected_rows)
    receipt_path = root / "recovery_receipt.json"
    if receipt_path.is_file():
        return _verify_complete_receipt(
            root,
            manifest=manifest,
            state=state,
            original_root=original_root,
        )
    receipt = _recovery_receipt(
        root=root,
        manifest=manifest,
        state=state,
        tokenizer_audit=tokenizer_audit,
    )
    _write_immutable_json(receipt_path, receipt)
    return _verify_complete_receipt(
        root,
        manifest=manifest,
        state=state,
        original_root=original_root,
    )


def load_and_verify_recovery(
    recovery_root: str | Path,
    *,
    original_root: str | Path,
    handoff: source_handoff_v1.SourceHandoff,
    tokenizer: Any,
) -> VerifiedRecovery:
    """Independently replay a completed recovery for a future publisher."""
    root = Path(recovery_root).resolve()
    manifest = verify_partial_handoff(root, handoff=handoff, tokenizer=tokenizer)
    state = inspect_acquisition(
        root / "acquisition",
        handoff=handoff,
        tokenizer=tokenizer,
        expected_failure_ids=manifest["baseline_missing_ids"],
    )
    if state.missing_ids:
        raise RecoveryError("recovery is incomplete")
    tokenizer_path = root / "audits/tokenizer_replay.jsonl"
    observed = _read_jsonl(tokenizer_path, "recovery tokenizer audit")
    expected = _expected_tokenizer_rows(state, handoff=handoff, tokenizer=tokenizer)
    if observed != expected:
        raise RecoveryError("recovery tokenizer audit drift")
    receipt = _verify_complete_receipt(
        root,
        manifest=manifest,
        state=state,
        original_root=original_root,
    )
    return VerifiedRecovery(root, manifest, receipt, state, tokenizer_path)


def resume_recovery(
    recovery_root: str | Path,
    *,
    original_root: str | Path,
    handoff: source_handoff_v1.SourceHandoff,
    tokenizer: Any,
    backend: v5.ProviderBackend,
    environment: Mapping[str, str] | None,
    concurrency: int = RECOVERY_CONCURRENCY,
    resume: bool = False,
) -> Mapping[str, Any]:
    """Process only baseline-missing rows in the independent recovery copy."""
    if concurrency != RECOVERY_CONCURRENCY:
        raise RecoveryError("v6 recovery requires concurrency=128")
    if not resume:
        raise RecoveryError("v6 recovery requires explicit --resume")
    root = Path(recovery_root).resolve()
    manifest = verify_partial_handoff(root, handoff=handoff, tokenizer=tokenizer)
    verify_original_v6_unchanged(original_root, manifest)
    identity = v5.ProviderIdentityRegistry()
    state = inspect_acquisition(
        root / "acquisition",
        handoff=handoff,
        tokenizer=tokenizer,
        identity=identity,
        expected_failure_ids=manifest["baseline_missing_ids"],
    )
    if identity.as_dict() != state.provider_identities:
        raise RecoveryError("existing provider identity was not restored")
    _reconcile_interrupted_terminals(root, manifest, state)
    if not state.missing_ids:
        return _finalize_recovery(
            root,
            original_root=original_root,
            manifest=manifest,
            state=state,
            handoff=handoff,
            tokenizer=tokenizer,
        )
    baseline_missing = set(manifest["baseline_missing_ids"])
    if not set(state.missing_ids).issubset(baseline_missing):
        raise RecoveryError("recovery attempted a non-baseline missing row")
    rows = _row_map(handoff)
    reference_path = root / "acquisition/official_pre_action_reference_bank.jsonl"
    reference_bank = official_v2.deserialize_official_reference_bank(
        reference_path.read_bytes()
    )
    # The inspection above bound every existing response, including partial
    # caches, into ``identity`` before any new provider call is allowed.
    config = v5.ProviderConfig()
    failures: list[dict[str, Any]] = []
    completed_ids: list[str] = []

    def worker(row: v5.PreparedRow) -> Mapping[str, Any]:
        return v6._process_downstream_terminal(
            row,
            output=root / "acquisition",
            tokenizer=tokenizer,
            reference_bank=reference_bank,
            official_reference_bank_sha256=state.reference_sha256,
            backend=backend,
            identity=identity,
            environment=environment,
            config=config,
            code_sha256=state.run_binding_sha256,
            source_audit=handoff.source_results[row.sample_id],
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(worker, rows[sample_id]): sample_id
            for sample_id in state.missing_ids
        }
        for future in as_completed(futures):
            sample_id = futures[future]
            try:
                future.result()
                completed_ids.append(sample_id)
            except Exception as exc:
                failures.append(
                    {
                        "sample_id": sample_id,
                        "split": rows[sample_id].split,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    failures.sort(key=lambda item: (item["split"], item["sample_id"]))
    _record_attempt(
        root,
        manifest,
        kind="provider_attempt",
        requested_ids=state.missing_ids,
        completed_ids=completed_ids,
        failures=failures,
    )
    if failures:
        raise RecoveryError("v6 recovery has unresolved failures")
    final_state = inspect_acquisition(
        root / "acquisition",
        handoff=handoff,
        tokenizer=tokenizer,
        expected_failure_ids=manifest["baseline_missing_ids"],
    )
    if final_state.missing_ids:
        raise RecoveryError("v6 recovery did not terminalize every missing row")
    return _finalize_recovery(
        root,
        original_root=original_root,
        manifest=manifest,
        state=final_state,
        handoff=handoff,
        tokenizer=tokenizer,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("seal", "verify", "resume", "all"), default="all"
    )
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL_ROOT)
    parser.add_argument("--recovery-root", type=Path, default=DEFAULT_RECOVERY_ROOT)
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--concurrency", type=int, default=RECOVERY_CONCURRENCY)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handoff = source_handoff_v1.load_and_verify_source_handoff(args.handoff_root)
    tokenizer = v5._load_tokenizer(args.tokenizer_path)
    if args.phase in {"seal", "all"}:
        seal_partial_acquisition(
            args.original_root,
            args.recovery_root,
            handoff=handoff,
            tokenizer=tokenizer,
        )
    if args.phase == "verify":
        manifest = verify_partial_handoff(
            args.recovery_root, handoff=handoff, tokenizer=tokenizer
        )
        verify_original_v6_unchanged(args.original_root, manifest)
        state = inspect_acquisition(
            args.recovery_root / "acquisition",
            handoff=handoff,
            tokenizer=tokenizer,
            expected_failure_ids=manifest["baseline_missing_ids"],
        )
        receipt = args.recovery_root / "recovery_receipt.json"
        result: Mapping[str, Any]
        if receipt.is_file():
            result = load_and_verify_recovery(
                args.recovery_root,
                original_root=args.original_root,
                handoff=handoff,
                tokenizer=tokenizer,
            ).receipt
        else:
            result = {
                "schema_version": SCHEMA_VERSION,
                "status": "partial_verified",
                "terminal_count": len(state.terminals),
                "missing_count": len(state.missing_ids),
                "compatibility_count": len(state.compatibility_ids),
            }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.phase in {"resume", "all"}:
        if not os.environ.get(v5.API_KEY_ENV, "").strip():
            raise RecoveryError(f"missing provider credential: {v5.API_KEY_ENV}")
        result = resume_recovery(
            args.recovery_root,
            original_root=args.original_root,
            handoff=handoff,
            tokenizer=tokenizer,
            backend=v5.OpenAICompatibleBackend(),
            environment=os.environ,
            concurrency=args.concurrency,
            resume=args.resume,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    result = verify_partial_handoff(
        args.recovery_root, handoff=handoff, tokenizer=tokenizer
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
