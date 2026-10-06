"""Provenance-sealed v6 downstream-only paper chk-2 acquisition.

Source admission is imported from an immutable v1 handoff.  This module never
invokes a source-audit provider role and never adopts v5 downstream caches.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from jobs.generation import generate_paper_chk2_chk1_analysis_rewrite as v5
from jobs.generation import paper_chk2_official_reference_v2 as official_v2
from jobs.generation import seal_paper_chk2_source_handoff_v1 as handoff_v1

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HANDOFF_ROOT = handoff_v1.DEFAULT_HANDOFF_ROOT
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "output/data/retrain_v2/chk2/chk1_final_analysis_to_minutes_flash_official_reference_v6_downstream128_20260831"
)
DEFAULT_TOKENIZER_PATH = v5.DEFAULT_TOKENIZER_PATH
DEFAULT_CONCURRENCY = 128
MAX_CONCURRENCY = 128
DEFAULT_PREFLIGHT_ROWS = 8
PREFLIGHT_SPLIT_COUNTS = {"train": 3, "validation": 3, "test": 2}
SCHEMA_VERSION = "paper-chk2-downstream-v6"
EXECUTION_SCHEMA_VERSION = "paper-chk2-downstream-execution-v1"
PROMPT_SCHEMA_VERSION = "paper-chk2-downstream-prompts-v1"
EXECUTION_RECEIPT_FIELDS = {
    "schema_version",
    "phase",
    "status",
    "configured_concurrency",
    "maximum_concurrency",
    "runner_sha256",
    "implementation_composite_sha256",
    "source_handoff_manifest_sha256",
    "source_handoff_manifest_file_sha256",
    "run_binding_sha256",
    "prompt_contract_sha256",
    "official_reference_bank_sha256",
    "tokenizer_contract_sha256",
    "provider_identities",
    "source_audit_provider_calls",
    "cache_counts",
    "terminal_cache_count",
    "artifacts",
    "receipt_sha256",
}
API_KEY_ENV = v5.API_KEY_ENV
SPLITS = v5.SPLITS

# Explicitly reused v5 downstream semantics.  The v5 module itself is sealed as
# an implementation dependency below; source-audit orchestration is not reused.
PreparedRow = v5.PreparedRow
OfficialReferenceBank = official_v2.OfficialReferenceBank
ProviderBackend = v5.ProviderBackend
ProviderResponse = v5.ProviderResponse
ProviderConfig = v5.ProviderConfig
ProviderIdentityRegistry = v5.ProviderIdentityRegistry
OpenAICompatibleBackend = v5.OpenAICompatibleBackend
SyntheticRewriteError = v5.SyntheticRewriteError
GenerationOutcome = v5.GenerationOutcome
Candidate = v5.Candidate
ContractError = v5.ContractError
legacy = v5.legacy
sha256_text = v5.sha256_text
canonical_json = v5.canonical_json
_cache_path = v5._cache_path
_terminal_path = v5._terminal_path
_terminal_binding = v5._terminal_binding
_validate_terminal_record = v5._validate_terminal_record
_terminal_provider_sidecars = v5._terminal_provider_sidecars
_resume_cache_payload = v5._resume_cache_payload
_resume_candidate = v5._resume_candidate
_candidate_from_response = v5._candidate_from_response
_teacher_user_prompt = v5._teacher_user_prompt
_prompt_payload = v5._prompt_payload
_validator_a_user_prompt = v5._validator_a_user_prompt
_validator_b_user_prompt = v5._validator_b_user_prompt
_resume_validator_contract_codes = v5._resume_validator_contract_codes
_validator_a_contract_repair_user_prompt = v5._validator_a_contract_repair_user_prompt
_validator_b_contract_repair_user_prompt = v5._validator_b_contract_repair_user_prompt
_validate_validator_a = v5._validate_validator_a
_validate_validator_b = v5._validate_validator_b
_controlled_validator_contract_codes = v5._controlled_validator_contract_codes
_request_projection = v5._request_projection
_base_terminal = v5._base_terminal
_store_terminal = v5._store_terminal
_generation_primary = v5._generation_primary
_validator_a_call = v5._validator_a_call
_validator_b_call = v5._validator_b_call
_rewrite_call = v5._rewrite_call
_raise_if_unresolved_rewrite_failure = v5._raise_if_unresolved_rewrite_failure
_fidelity_repair_user_prompt = v5._fidelity_repair_user_prompt
_style_repair_user_prompt = v5._style_repair_user_prompt
_style_repair_deterministic_rejection = v5._style_repair_deterministic_rejection
_safe_style_feedback = v5._safe_style_feedback
_student_prompt = v5._student_prompt
TERMINAL_SOURCE_CONTRACT_REJECT = v5.TERMINAL_SOURCE_CONTRACT_REJECT
TERMINAL_SOURCE_REJECT = v5.TERMINAL_SOURCE_REJECT
TERMINAL_GENERATION_REJECT = v5.TERMINAL_GENERATION_REJECT
TERMINAL_VALIDATOR_A_CONTRACT_REJECT = v5.TERMINAL_VALIDATOR_A_CONTRACT_REJECT
TERMINAL_FIDELITY_REJECT = v5.TERMINAL_FIDELITY_REJECT
TERMINAL_VALIDATOR_B_CONTRACT_REJECT = v5.TERMINAL_VALIDATOR_B_CONTRACT_REJECT
TERMINAL_REFERENCE_UNAVAILABLE_REJECT = v5.TERMINAL_REFERENCE_UNAVAILABLE_REJECT
TERMINAL_STYLE_REJECT = v5.TERMINAL_STYLE_REJECT
TERMINAL_STYLE_FIDELITY_REJECT = v5.TERMINAL_STYLE_FIDELITY_REJECT
TERMINAL_PASS = v5.TERMINAL_PASS
ROLE_VALIDATOR_A_PRIMARY = v5.ROLE_VALIDATOR_A_PRIMARY
ROLE_VALIDATOR_A_FIDELITY_REPAIR = v5.ROLE_VALIDATOR_A_FIDELITY_REPAIR
ROLE_VALIDATOR_A_STYLE_REPAIR = v5.ROLE_VALIDATOR_A_STYLE_REPAIR
ROLE_VALIDATOR_B_PRIMARY = v5.ROLE_VALIDATOR_B_PRIMARY
ROLE_VALIDATOR_B_STYLE_REPAIR = v5.ROLE_VALIDATOR_B_STYLE_REPAIR
ROLE_REWRITE_FIDELITY_REPAIR = v5.ROLE_REWRITE_FIDELITY_REPAIR
ROLE_REWRITE_STYLE_REPAIR = v5.ROLE_REWRITE_STYLE_REPAIR
ROLE_REWRITE_PRIMARY = v5.ROLE_REWRITE_PRIMARY
SOURCE_PROVIDER_ROLES = (
    v5.ROLE_SOURCE_AUDIT_PRIMARY,
    v5.ROLE_SOURCE_AUDIT_ADJUDICATION,
    v5.ROLE_SOURCE_AUDIT_CONTRACT_REPAIR,
)
FIDELITY_REPAIR_SYSTEM_PROMPT = v5.FIDELITY_REPAIR_SYSTEM_PROMPT
STYLE_REPAIR_SYSTEM_PROMPT = v5.STYLE_REPAIR_SYSTEM_PROMPT
REWRITE_SYSTEM_PROMPT = v5.REWRITE_SYSTEM_PROMPT
VALIDATOR_A_SYSTEM_PROMPT = v5.VALIDATOR_A_SYSTEM_PROMPT
VALIDATOR_B_SYSTEM_PROMPT = v5.VALIDATOR_B_SYSTEM_PROMPT
VALIDATOR_A_CONTRACT_REPAIR_SYSTEM_PROMPT = v5.VALIDATOR_A_CONTRACT_REPAIR_SYSTEM_PROMPT
VALIDATOR_B_CONTRACT_REPAIR_SYSTEM_PROMPT = v5.VALIDATOR_B_CONTRACT_REPAIR_SYSTEM_PROMPT
VALIDATOR_A_CONTRACT_REPAIR_ROLES = v5.VALIDATOR_A_CONTRACT_REPAIR_ROLES
VALIDATOR_B_CONTRACT_REPAIR_ROLES = v5.VALIDATOR_B_CONTRACT_REPAIR_ROLES
TERMINAL_STATUSES = v5.TERMINAL_STATUSES
CACHE_SCHEMA_VERSION = v5.CACHE_SCHEMA_VERSION


class _NoProviderCalls:
    def generate(self, **kwargs: Any) -> ProviderResponse:
        raise SyntheticRewriteError(
            f"v6 replay attempted provider call: {kwargs.get('role')}"
        )


def _load_downstream_terminal(
    output: Path,
    row: PreparedRow,
    *,
    code_sha256: str,
    official_reference_bank_sha256: str,
    identity: ProviderIdentityRegistry,
    reference_bank: OfficialReferenceBank | Mapping[str, Any],
    config: ProviderConfig,
    environment: Mapping[str, str] | None,
    source_audit: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any] | None:
    for source_role in SOURCE_PROVIDER_ROLES:
        if _cache_path(output, source_role, row).exists():
            raise SyntheticRewriteError(
                f"v6 source-role cache is forbidden: {row.sample_id}:{source_role}"
            )
    path = _terminal_path(output, row)
    if not path.is_file():
        return None
    payload = legacy._read_json(path, label="terminal cache")
    if payload.get("binding") != _terminal_binding(
        row, code_sha256, official_reference_bank_sha256
    ):
        raise SyntheticRewriteError(f"terminal cache binding mismatch: {path}")
    record = payload.get("record")
    if (
        not isinstance(record, dict)
        or record.get("terminal_status") not in TERMINAL_STATUSES
    ):
        raise SyntheticRewriteError(f"terminal cache record invalid: {path}")
    if payload.get("record_sha256") != sha256_text(canonical_json(record)):
        raise SyntheticRewriteError(f"terminal cache record hash mismatch: {path}")
    _validate_terminal_record(row, record)
    if record.get("source_audit") != source_audit:
        raise SyntheticRewriteError(
            f"downstream/source handoff snapshot drift: {row.sample_id}"
        )
    sidecars = {
        role: provider
        for role, provider in _terminal_provider_sidecars(row, record).items()
        if role not in SOURCE_PROVIDER_ROLES
    }
    caches: dict[str, dict[str, Any]] = {}
    responses: dict[str, ProviderResponse] = {}
    for role, provider in sidecars.items():
        cache, response = _resume_cache_payload(
            output=output,
            row=row,
            role=role,
            provider=provider,
            environment=environment,
        )
        caches[role] = cache
        responses[role] = response

    candidates: dict[str, Candidate] = {}
    for role, attempt in (
        (ROLE_REWRITE_PRIMARY, "primary"),
        (ROLE_REWRITE_FIDELITY_REPAIR, "fidelity_repair"),
        (ROLE_REWRITE_STYLE_REPAIR, "style_repair"),
    ):
        if role in responses:
            try:
                candidates[role] = _candidate_from_response(
                    row,
                    responses[role],
                    attempt=attempt,
                    tokenizer=tokenizer,
                    provider_record=sidecars[role],
                )
            except ContractError:
                pass

    selected_attempt = record.get("generation", {}).get("selected_attempt")
    selected_role = {
        "primary": ROLE_REWRITE_PRIMARY,
        "fidelity_repair": ROLE_REWRITE_FIDELITY_REPAIR,
        "style_repair": ROLE_REWRITE_STYLE_REPAIR,
    }.get(selected_attempt)
    selected_candidate = candidates.get(selected_role) if selected_role else None
    if record.get("teacher_response_analysis") or record.get("rewritten_minutes"):
        if (
            selected_candidate is None
            or record.get("teacher_response_analysis")
            != selected_candidate.teacher_response_analysis
            or record.get("rewritten_minutes") != selected_candidate.rewritten_minutes
            or record.get("teacher_response_analysis_sha256")
            != sha256_text(selected_candidate.teacher_response_analysis)
            or record.get("rewritten_minutes_sha256")
            != sha256_text(selected_candidate.rewritten_minutes)
            or record.get("generation", {}).get("deterministic_validation")
            != selected_candidate.deterministic_validation
        ):
            raise SyntheticRewriteError(
                f"v6 selected rewrite candidate drift: {row.sample_id}"
            )
        if record.get("terminal_status") == TERMINAL_PASS and (
            record.get("sft_response") != selected_candidate.sft_response
            or record.get("response_sha256")
            != sha256_text(selected_candidate.sft_response)
        ):
            raise SyntheticRewriteError(
                f"v6 selected SFT response drift: {row.sample_id}"
            )

    expected_prompts: dict[str, tuple[str, str, str | None]] = {}
    if ROLE_REWRITE_PRIMARY in sidecars:
        expected_prompts[ROLE_REWRITE_PRIMARY] = (
            REWRITE_SYSTEM_PROMPT,
            _teacher_user_prompt(row),
            None,
        )

    repair_history = record["repair_history"]
    fidelity_event = next(
        (event for event in repair_history if event.get("repair_type") == "fidelity"),
        None,
    )
    if ROLE_REWRITE_FIDELITY_REPAIR in sidecars:
        if not isinstance(fidelity_event, Mapping):
            raise SyntheticRewriteError(
                f"fidelity repair history missing: {row.sample_id}"
            )
        validator_result = (
            fidelity_event.get("validator_a")
            if fidelity_event.get("trigger_stage") == "validator_a_primary"
            else None
        )
        expected_prompts[ROLE_REWRITE_FIDELITY_REPAIR] = (
            FIDELITY_REPAIR_SYSTEM_PROMPT,
            _fidelity_repair_user_prompt(
                row,
                fidelity_event.get("reason_codes") or [],
                candidate=(
                    candidates.get(ROLE_REWRITE_PRIMARY)
                    if validator_result is not None
                    else None
                ),
                validator_a_result=(
                    validator_result if isinstance(validator_result, Mapping) else None
                ),
            ),
            None,
        )
    style_event = next(
        (event for event in repair_history if event.get("repair_type") == "style"),
        None,
    )
    if ROLE_REWRITE_STYLE_REPAIR in sidecars:
        if not isinstance(style_event, Mapping):
            raise SyntheticRewriteError(
                f"style repair history missing: {row.sample_id}"
            )
        prior = candidates.get(
            ROLE_REWRITE_FIDELITY_REPAIR
            if record["generation"].get("fidelity_repair_used")
            else ROLE_REWRITE_PRIMARY
        )
        snapshot = style_event.get("validator_b")
        if (
            prior is None
            or not isinstance(snapshot, Mapping)
            or not isinstance(snapshot.get("style_feedback"), Mapping)
        ):
            raise SyntheticRewriteError(
                f"style repair replay metadata invalid: {row.sample_id}"
            )
        expected_prompts[ROLE_REWRITE_STYLE_REPAIR] = (
            STYLE_REPAIR_SYSTEM_PROMPT,
            _prompt_payload(
                "Apply one style-only repair",
                {
                    "source_analysis": row.source_analysis,
                    "current_rewritten_minutes": prior.rewritten_minutes,
                    "style_feedback": dict(snapshot["style_feedback"]),
                },
            ),
            None,
        )

    candidate_by_a_role = {
        ROLE_VALIDATOR_A_PRIMARY: candidates.get(ROLE_REWRITE_PRIMARY),
        ROLE_VALIDATOR_A_FIDELITY_REPAIR: candidates.get(ROLE_REWRITE_FIDELITY_REPAIR),
        ROLE_VALIDATOR_A_STYLE_REPAIR: candidates.get(ROLE_REWRITE_STYLE_REPAIR),
    }
    for role, candidate in candidate_by_a_role.items():
        if role not in sidecars:
            continue
        if candidate is None:
            raise SyntheticRewriteError(
                f"Validator-A candidate replay missing: {row.sample_id}:{role}"
            )
        expected_prompts[role] = (
            VALIDATOR_A_SYSTEM_PROMPT,
            _validator_a_user_prompt(row, candidate),
            None,
        )
        repair_role = VALIDATOR_A_CONTRACT_REPAIR_ROLES[role]
        if repair_role in sidecars:
            codes = _resume_validator_contract_codes(
                validator="validator_a",
                row=row,
                candidate=candidate,
                response=responses[role],
                reference_bank=reference_bank,
            )
            expected_prompts[repair_role] = (
                VALIDATOR_A_CONTRACT_REPAIR_SYSTEM_PROMPT,
                _validator_a_contract_repair_user_prompt(row, candidate, codes),
                None,
            )

    a_results_for_replay: list[Mapping[str, Any]] = []
    terminal_a = record.get("validator_a")
    if isinstance(terminal_a, Mapping) and isinstance(
        terminal_a.get("result"), Mapping
    ):
        a_results_for_replay.append(terminal_a["result"])
    for event in repair_history:
        snapshot = event.get("validator_a")
        if isinstance(snapshot, Mapping):
            nested = snapshot.get("result", snapshot)
            if isinstance(nested, Mapping):
                a_results_for_replay.append(nested)
    for stored_result in a_results_for_replay:
        exhaustion = stored_result.get("contract_exhaustion")
        if stored_result.get("contract_exhausted") is not True:
            continue
        if not isinstance(exhaustion, Mapping):
            raise SyntheticRewriteError(
                f"Validator-A exhaustion missing on resume: {row.sample_id}"
            )
        target_role = str(exhaustion.get("target_role"))
        repair_role = str(exhaustion.get("contract_repair_role"))
        candidate = candidate_by_a_role.get(target_role)
        repair_response = responses.get(repair_role)
        if candidate is None or repair_response is None:
            raise SyntheticRewriteError(
                f"Validator-A exhaustion replay missing: {row.sample_id}:{target_role}"
            )
        try:
            replay_result, _passed, replay_reasons = _validate_validator_a(
                row, candidate, repair_response
            )
            replay_contract = replay_result.get("contract_reasons")
            if not isinstance(replay_contract, list):
                replay_contract = ["validator_a_contract_state_missing"]
        except ContractError as exc:
            replay_contract = list(exc.reasons)
        replay_codes = _controlled_validator_contract_codes(
            "validator_a", replay_contract
        )
        if replay_codes != exhaustion.get("controlled_error_codes"):
            raise SyntheticRewriteError(
                f"Validator-A exhausted report code drift: {row.sample_id}"
            )

    prior_for_b = candidates.get(
        ROLE_REWRITE_FIDELITY_REPAIR
        if record["generation"].get("fidelity_repair_used")
        else ROLE_REWRITE_PRIMARY
    )
    candidate_by_b_role = {
        ROLE_VALIDATOR_B_PRIMARY: prior_for_b,
        ROLE_VALIDATOR_B_STYLE_REPAIR: candidates.get(ROLE_REWRITE_STYLE_REPAIR),
    }
    for role, candidate in candidate_by_b_role.items():
        if role not in sidecars:
            continue
        if candidate is None:
            raise SyntheticRewriteError(
                f"Validator-B candidate replay missing: {row.sample_id}:{role}"
            )
        expected_prompts[role] = (
            VALIDATOR_B_SYSTEM_PROMPT,
            _validator_b_user_prompt(row, candidate, reference_bank),
            official_reference_bank_sha256,
        )
        repair_role = VALIDATOR_B_CONTRACT_REPAIR_ROLES[role]
        if repair_role in sidecars:
            codes = _resume_validator_contract_codes(
                validator="validator_b",
                row=row,
                candidate=candidate,
                response=responses[role],
                reference_bank=reference_bank,
            )
            expected_prompts[repair_role] = (
                VALIDATOR_B_CONTRACT_REPAIR_SYSTEM_PROMPT,
                _validator_b_contract_repair_user_prompt(
                    row, candidate, reference_bank, codes
                ),
                official_reference_bank_sha256,
            )

    b_results_for_replay: list[Mapping[str, Any]] = []
    terminal_b = record.get("validator_b")
    if isinstance(terminal_b, Mapping) and isinstance(
        terminal_b.get("result"), Mapping
    ):
        b_results_for_replay.append(terminal_b["result"])
    for event in repair_history:
        snapshot = event.get("validator_b")
        if isinstance(snapshot, Mapping):
            b_results_for_replay.append(snapshot)
    for stored_result in b_results_for_replay:
        exhaustion = stored_result.get("contract_exhaustion")
        if stored_result.get("contract_exhausted") is not True:
            continue
        if not isinstance(exhaustion, Mapping):
            raise SyntheticRewriteError(
                f"Validator-B exhaustion missing on resume: {row.sample_id}"
            )
        target_role = str(exhaustion.get("target_role"))
        repair_role = str(exhaustion.get("contract_repair_role"))
        candidate = candidate_by_b_role.get(target_role)
        repair_response = responses.get(repair_role)
        if candidate is None or repair_response is None:
            raise SyntheticRewriteError(
                f"Validator-B exhaustion replay missing: {row.sample_id}:{target_role}"
            )
        try:
            replay_result, _passed, _reasons = _validate_validator_b(
                row, candidate, repair_response, reference_bank
            )
            replay_contract = replay_result.get("contract_reasons")
            if not isinstance(replay_contract, list):
                replay_contract = ["validator_b_contract_state_missing"]
        except ContractError as exc:
            replay_contract = list(exc.reasons)
        replay_codes = _controlled_validator_contract_codes(
            "validator_b", replay_contract
        )
        if replay_codes != exhaustion.get("controlled_error_codes"):
            raise SyntheticRewriteError(
                f"Validator-B exhausted report code drift: {row.sample_id}"
            )

    if set(expected_prompts) != set(sidecars):
        raise SyntheticRewriteError(
            f"terminal provider invocation sequence drift: {row.sample_id}"
        )
    for role, (system_prompt, user_prompt, reference_sha) in expected_prompts.items():
        projection = _request_projection(role, user_prompt)
        expected_binding = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "role": role,
            "sample_id": row.sample_id,
            "split": row.split,
            "source_analysis_sha256": row.source_analysis_sha256,
            "provided_data_sha256": row.provided_data_sha256,
            "official_reference_bank_sha256": reference_sha,
            "system_prompt_sha256": sha256_text(system_prompt),
            "user_prompt_sha256": sha256_text(user_prompt),
            "provider_contract_sha256": config.contract_sha256,
            "code_sha256": code_sha256,
            "request_projection_sha256": sha256_text(canonical_json(projection)),
        }
        cache = caches[role]
        if cache.get("binding") != expected_binding or cache.get(
            "binding_sha256"
        ) != sha256_text(canonical_json(expected_binding)):
            raise SyntheticRewriteError(
                f"terminal provider cache binding mismatch: {_cache_path(output, role, row)}"
            )
        if cache.get("request_projection") != projection:
            raise SyntheticRewriteError(
                f"terminal provider cache projection mismatch: {_cache_path(output, role, row)}"
            )
        identity.bind(role, responses[role])

    terminal_a = record.get("validator_a")
    if isinstance(terminal_a, Mapping) and terminal_a:
        a_provider = terminal_a.get("provider")
        a_role = {
            "primary": ROLE_VALIDATOR_A_PRIMARY,
            "fidelity_repair": ROLE_VALIDATOR_A_FIDELITY_REPAIR,
            "style_repair": ROLE_VALIDATOR_A_STYLE_REPAIR,
        }.get(selected_attempt)
        if not isinstance(a_provider, Mapping) or a_role not in a_provider:
            raise SyntheticRewriteError(
                f"v6 terminal Validator-A role drift: {row.sample_id}"
            )
        a_candidate = candidate_by_a_role[a_role]
        if a_candidate is None:
            raise SyntheticRewriteError(
                f"v6 terminal Validator-A candidate drift: {row.sample_id}"
            )
        replay_result, replay_pass, replay_reasons, replay_provider = _validator_a_call(
            row,
            a_candidate,
            role=a_role,
            output=output,
            backend=_NoProviderCalls(),
            identity=identity,
            environment=environment,
            config=config,
            code_sha256=code_sha256,
        )
        if (
            terminal_a.get("result") != replay_result
            or terminal_a.get("machine_pass") is not replay_pass
            or terminal_a.get("reasons") != replay_reasons
            or any(
                a_provider.get(role) != value for role, value in replay_provider.items()
            )
        ):
            raise SyntheticRewriteError(
                f"v6 terminal Validator-A normalized replay drift: {row.sample_id}"
            )

    terminal_b = record.get("validator_b")
    if isinstance(terminal_b, Mapping) and terminal_b:
        b_provider = terminal_b.get("provider")
        b_role = (
            ROLE_VALIDATOR_B_STYLE_REPAIR
            if record.get("generation", {}).get("style_repair_used") is True
            else ROLE_VALIDATOR_B_PRIMARY
        )
        if not isinstance(b_provider, Mapping) or b_role not in b_provider:
            raise SyntheticRewriteError(
                f"v6 terminal Validator-B role drift: {row.sample_id}"
            )
        b_candidate = candidate_by_b_role[b_role]
        if b_candidate is None:
            raise SyntheticRewriteError(
                f"v6 terminal Validator-B candidate drift: {row.sample_id}"
            )
        replay_result, replay_pass, replay_reasons, replay_provider = _validator_b_call(
            row,
            b_candidate,
            reference_bank,
            role=b_role,
            output=output,
            backend=_NoProviderCalls(),
            identity=identity,
            environment=environment,
            config=config,
            code_sha256=code_sha256,
            official_reference_bank_sha256=official_reference_bank_sha256,
        )
        if (
            terminal_b.get("result") != replay_result
            or terminal_b.get("machine_pass") is not replay_pass
            or terminal_b.get("reasons") != replay_reasons
            or any(
                b_provider.get(role) != value for role, value in replay_provider.items()
            )
        ):
            raise SyntheticRewriteError(
                f"v6 terminal Validator-B normalized replay drift: {row.sample_id}"
            )
    return dict(record)


def _process_downstream_terminal(
    row: PreparedRow,
    *,
    output: Path,
    tokenizer: Any,
    reference_bank: OfficialReferenceBank | Mapping[str, Any],
    official_reference_bank_sha256: str,
    backend: ProviderBackend,
    identity: ProviderIdentityRegistry,
    environment: Mapping[str, str] | None,
    config: ProviderConfig,
    code_sha256: str,
    source_audit: Mapping[str, Any],
) -> dict[str, Any]:
    existing = _load_downstream_terminal(
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
    if existing is not None:
        return existing
    record = _base_terminal(row, source_audit)
    if source_audit.get("machine_pass") is not True:
        source_contract_repair = source_audit.get("result", {}).get("contract_repair")
        source_contract_exhausted = bool(
            isinstance(source_contract_repair, Mapping)
            and source_contract_repair.get("contract_exhausted") is True
        )
        record.update(
            {
                "terminal_status": (
                    TERMINAL_SOURCE_CONTRACT_REJECT
                    if source_contract_exhausted
                    else TERMINAL_SOURCE_REJECT
                ),
                "rejection_stage": (
                    "source_audit_contract"
                    if source_contract_exhausted
                    else "source_admission"
                ),
                "rejection_reasons": list(
                    source_audit.get("reasons") or ["source_quality_reject"]
                ),
            }
        )
        return _store_terminal(
            output,
            row,
            record,
            code_sha256=code_sha256,
            official_reference_bank_sha256=official_reference_bank_sha256,
        )
    generation = _generation_primary(
        row,
        output=output,
        tokenizer=tokenizer,
        backend=backend,
        identity=identity,
        environment=environment,
        config=config,
        code_sha256=code_sha256,
    )
    record["repair_history"] = [dict(item) for item in generation.repair_history]
    if generation.candidate is None:
        record.update(
            {
                "terminal_status": TERMINAL_GENERATION_REJECT,
                "rejection_stage": "generation_deterministic_gate",
                "rejection_reasons": list(generation.reasons),
                "generation": {
                    "selected_attempt": "fidelity_repair"
                    if generation.fidelity_repair_used
                    else "primary",
                    "fidelity_repair_used": generation.fidelity_repair_used,
                    "style_repair_used": False,
                    "deterministic_validation": dict(
                        generation.deterministic_validation
                    ),
                    "provider": dict(generation.provider),
                },
            }
        )
        return _store_terminal(
            output,
            row,
            record,
            code_sha256=code_sha256,
            official_reference_bank_sha256=official_reference_bank_sha256,
        )
    candidate = generation.candidate
    a_role = (
        ROLE_VALIDATOR_A_FIDELITY_REPAIR
        if generation.fidelity_repair_used
        else ROLE_VALIDATOR_A_PRIMARY
    )
    a_result, a_pass, a_reasons, a_provider = _validator_a_call(
        row,
        candidate,
        role=a_role,
        output=output,
        backend=backend,
        identity=identity,
        environment=environment,
        config=config,
        code_sha256=code_sha256,
    )
    a_providers: dict[str, Any] = dict(a_provider)
    if a_result.get("contract_exhausted") is True:
        record.update(
            {
                "teacher_response_analysis": candidate.teacher_response_analysis,
                "rewritten_minutes": candidate.rewritten_minutes,
                "teacher_response_analysis_sha256": sha256_text(
                    candidate.teacher_response_analysis
                ),
                "rewritten_minutes_sha256": sha256_text(candidate.rewritten_minutes),
                "generation": {
                    "selected_attempt": candidate.attempt,
                    "fidelity_repair_used": generation.fidelity_repair_used,
                    "style_repair_used": False,
                    "deterministic_validation": dict(
                        candidate.deterministic_validation
                    ),
                    "provider": dict(generation.provider),
                },
                "validator_a": {
                    "complete": True,
                    "machine_pass": False,
                    "reasons": list(a_reasons),
                    "result": a_result,
                    "provider": dict(a_providers),
                },
                "terminal_status": TERMINAL_VALIDATOR_A_CONTRACT_REJECT,
                "rejection_stage": a_role,
                "rejection_reasons": list(a_reasons),
            }
        )
        return _store_terminal(
            output,
            row,
            record,
            code_sha256=code_sha256,
            official_reference_bank_sha256=official_reference_bank_sha256,
        )
    if not a_pass and not generation.fidelity_repair_used:
        a_repair_event = {
            "repair_type": "fidelity",
            "trigger_stage": "validator_a_primary",
            "attempt": candidate.attempt,
            "reason_codes": list(a_reasons),
            "validator_a": dict(a_result),
            "provider": dict(a_provider),
        }
        record["repair_history"].append(a_repair_event)
        repaired, diagnostics, repair_provider = _rewrite_call(
            row,
            role=ROLE_REWRITE_FIDELITY_REPAIR,
            system_prompt=FIDELITY_REPAIR_SYSTEM_PROMPT,
            user_prompt=_fidelity_repair_user_prompt(
                row, a_reasons, candidate=candidate, validator_a_result=a_result
            ),
            attempt="fidelity_repair",
            output=output,
            tokenizer=tokenizer,
            backend=backend,
            identity=identity,
            environment=environment,
            config=config,
            code_sha256=code_sha256,
        )
        generation = GenerationOutcome(
            repaired,
            True,
            tuple(diagnostics.get("reasons") or []),
            {
                **dict(generation.provider),
                ROLE_REWRITE_FIDELITY_REPAIR: repair_provider,
            },
            diagnostics,
            (*generation.repair_history, a_repair_event),
        )
        if repaired is None:
            _raise_if_unresolved_rewrite_failure(
                diagnostics, role=ROLE_REWRITE_FIDELITY_REPAIR
            )
            record.update(
                {
                    "terminal_status": TERMINAL_GENERATION_REJECT,
                    "rejection_stage": "fidelity_repair_deterministic_gate",
                    "rejection_reasons": list(diagnostics.get("reasons") or []),
                    "generation": {
                        "selected_attempt": "fidelity_repair",
                        "fidelity_repair_used": True,
                        "style_repair_used": False,
                        "deterministic_validation": dict(diagnostics),
                        "provider": dict(generation.provider),
                    },
                    "validator_a": {
                        "complete": True,
                        "machine_pass": False,
                        "reasons": list(a_reasons),
                        "result": a_result,
                        "provider": dict(a_providers),
                    },
                }
            )
            return _store_terminal(
                output,
                row,
                record,
                code_sha256=code_sha256,
                official_reference_bank_sha256=official_reference_bank_sha256,
            )
        if repaired is not None:
            candidate = repaired
            a_result, a_pass, a_reasons, a_provider = _validator_a_call(
                row,
                candidate,
                role=ROLE_VALIDATOR_A_FIDELITY_REPAIR,
                output=output,
                backend=backend,
                identity=identity,
                environment=environment,
                config=config,
                code_sha256=code_sha256,
            )
            a_providers.update(a_provider)
            if a_result.get("contract_exhausted") is True:
                record.update(
                    {
                        "teacher_response_analysis": candidate.teacher_response_analysis,
                        "rewritten_minutes": candidate.rewritten_minutes,
                        "teacher_response_analysis_sha256": sha256_text(
                            candidate.teacher_response_analysis
                        ),
                        "rewritten_minutes_sha256": sha256_text(
                            candidate.rewritten_minutes
                        ),
                        "generation": {
                            "selected_attempt": candidate.attempt,
                            "fidelity_repair_used": True,
                            "style_repair_used": False,
                            "deterministic_validation": dict(
                                candidate.deterministic_validation
                            ),
                            "provider": dict(generation.provider),
                        },
                        "validator_a": {
                            "complete": True,
                            "machine_pass": False,
                            "reasons": list(a_reasons),
                            "result": a_result,
                            "provider": dict(a_providers),
                        },
                        "terminal_status": TERMINAL_VALIDATOR_A_CONTRACT_REJECT,
                        "rejection_stage": ROLE_VALIDATOR_A_FIDELITY_REPAIR,
                        "rejection_reasons": list(a_reasons),
                    }
                )
                return _store_terminal(
                    output,
                    row,
                    record,
                    code_sha256=code_sha256,
                    official_reference_bank_sha256=official_reference_bank_sha256,
                )
    record.update(
        {
            "teacher_response_analysis": candidate.teacher_response_analysis,
            "rewritten_minutes": candidate.rewritten_minutes,
            "teacher_response_analysis_sha256": sha256_text(
                candidate.teacher_response_analysis
            ),
            "rewritten_minutes_sha256": sha256_text(candidate.rewritten_minutes),
            "generation": {
                "selected_attempt": candidate.attempt,
                "fidelity_repair_used": generation.fidelity_repair_used,
                "style_repair_used": False,
                "deterministic_validation": dict(candidate.deterministic_validation),
                "provider": dict(generation.provider),
            },
            "validator_a": {
                "complete": True,
                "machine_pass": a_pass,
                "reasons": list(a_reasons),
                "result": a_result,
                "provider": dict(a_providers),
            },
        }
    )
    if not a_pass:
        record.update(
            {
                "terminal_status": TERMINAL_FIDELITY_REJECT,
                "rejection_stage": "validator_a_input_fidelity",
                "rejection_reasons": list(a_reasons),
            }
        )
        return _store_terminal(
            output,
            row,
            record,
            code_sha256=code_sha256,
            official_reference_bank_sha256=official_reference_bank_sha256,
        )
    b_result, b_pass, b_reasons, b_provider = _validator_b_call(
        row,
        candidate,
        reference_bank,
        role=ROLE_VALIDATOR_B_PRIMARY,
        output=output,
        backend=backend,
        identity=identity,
        environment=environment,
        config=config,
        code_sha256=code_sha256,
        official_reference_bank_sha256=official_reference_bank_sha256,
    )
    b_providers: dict[str, Any] = dict(b_provider)
    if b_result.get("contract_exhausted") is True:
        record["validator_b"] = {
            "complete": True,
            "machine_pass": False,
            "mean_score": None,
            "min_score": None,
            "reasons": list(b_reasons),
            "result": b_result,
            "provider": dict(b_providers),
        }
        record.update(
            {
                "terminal_status": TERMINAL_VALIDATOR_B_CONTRACT_REJECT,
                "rejection_stage": ROLE_VALIDATOR_B_PRIMARY,
                "rejection_reasons": list(b_reasons),
            }
        )
        return _store_terminal(
            output,
            row,
            record,
            code_sha256=code_sha256,
            official_reference_bank_sha256=official_reference_bank_sha256,
        )
    if not b_pass and b_result.get("comparison_status") == "no_comparable_passage":
        record["validator_b"] = {
            "complete": True,
            "machine_pass": False,
            "mean_score": b_result["mean_score"],
            "min_score": b_result["min_score"],
            "reasons": list(b_reasons),
            "result": b_result,
            "provider": dict(b_providers),
        }
        record.update(
            {
                "terminal_status": TERMINAL_REFERENCE_UNAVAILABLE_REJECT,
                "rejection_stage": "validator_b_reference_comparison",
                "rejection_reasons": list(b_reasons),
            }
        )
        return _store_terminal(
            output,
            row,
            record,
            code_sha256=code_sha256,
            official_reference_bank_sha256=official_reference_bank_sha256,
        )
    if not b_pass and b_result.get("critical_style_errors"):
        record["validator_b"] = {
            "complete": True,
            "machine_pass": False,
            "mean_score": b_result["mean_score"],
            "min_score": b_result["min_score"],
            "reasons": list(b_reasons),
            "result": b_result,
            "provider": dict(b_providers),
        }
        record.update(
            {
                "terminal_status": TERMINAL_STYLE_REJECT,
                "rejection_stage": "validator_b_critical_style_error",
                "rejection_reasons": list(b_reasons),
            }
        )
        return _store_terminal(
            output,
            row,
            record,
            code_sha256=code_sha256,
            official_reference_bank_sha256=official_reference_bank_sha256,
        )
    if not b_pass:
        record["repair_history"].append(
            {
                "repair_type": "style",
                "trigger_stage": "validator_b_primary",
                "attempt": candidate.attempt,
                "reason_codes": list(b_reasons),
                # Persist the factual-admission invocation that authorized this
                # candidate to reach Validator B.  A later style repair invokes
                # Validator A again and replaces the terminal selected result;
                # without this immutable snapshot, the earlier PASS (including
                # any one-shot contract repair) could not be independently
                # replayed by the publisher.
                "validator_a": {
                    "complete": True,
                    "machine_pass": a_pass,
                    "reasons": list(a_reasons),
                    "result": dict(a_result),
                    "provider": dict(a_provider),
                },
                "validator_b": {
                    "machine_pass": b_pass,
                    "mean_score": b_result["mean_score"],
                    "min_score": b_result["min_score"],
                    "style_feedback": _safe_style_feedback(b_result),
                    "contract_repair_used": b_result["contract_repair_used"],
                    "contract_repair": b_result["contract_repair"],
                },
                "provider": dict(b_provider),
            }
        )
        repaired, diagnostics, repair_provider = _rewrite_call(
            row,
            role=ROLE_REWRITE_STYLE_REPAIR,
            system_prompt=STYLE_REPAIR_SYSTEM_PROMPT,
            user_prompt=_style_repair_user_prompt(
                row, candidate, b_result, reference_bank=reference_bank
            ),
            attempt="style_repair",
            output=output,
            tokenizer=tokenizer,
            backend=backend,
            identity=identity,
            environment=environment,
            config=config,
            code_sha256=code_sha256,
        )
        if repaired is None:
            _raise_if_unresolved_rewrite_failure(
                diagnostics, role=ROLE_REWRITE_STYLE_REPAIR
            )
            rejection_status, rejection_stage = _style_repair_deterministic_rejection(
                diagnostics
            )
            record.update(
                {
                    "terminal_status": rejection_status,
                    "rejection_stage": rejection_stage,
                    "rejection_reasons": list(diagnostics.get("reasons") or []),
                    "generation": {
                        **record["generation"],
                        "selected_attempt": "style_repair",
                        "style_repair_used": True,
                        "deterministic_validation": diagnostics,
                        "provider": {
                            **dict(generation.provider),
                            ROLE_REWRITE_STYLE_REPAIR: repair_provider,
                        },
                    },
                    "validator_b": {
                        "complete": True,
                        "machine_pass": False,
                        "mean_score": b_result["mean_score"],
                        "min_score": b_result["min_score"],
                        "reasons": list(b_reasons),
                        "result": b_result,
                        "provider": dict(b_providers),
                    },
                }
            )
            return _store_terminal(
                output,
                row,
                record,
                code_sha256=code_sha256,
                official_reference_bank_sha256=official_reference_bank_sha256,
            )
        candidate = repaired
        a_result, a_pass, a_reasons, a_provider = _validator_a_call(
            row,
            candidate,
            role=ROLE_VALIDATOR_A_STYLE_REPAIR,
            output=output,
            backend=backend,
            identity=identity,
            environment=environment,
            config=config,
            code_sha256=code_sha256,
        )
        a_providers.update(a_provider)
        record.update(
            {
                "teacher_response_analysis": candidate.teacher_response_analysis,
                "rewritten_minutes": candidate.rewritten_minutes,
                "teacher_response_analysis_sha256": sha256_text(
                    candidate.teacher_response_analysis
                ),
                "rewritten_minutes_sha256": sha256_text(candidate.rewritten_minutes),
                "generation": {
                    **record["generation"],
                    "selected_attempt": "style_repair",
                    "style_repair_used": True,
                    "deterministic_validation": dict(
                        candidate.deterministic_validation
                    ),
                    "provider": {
                        **dict(generation.provider),
                        ROLE_REWRITE_STYLE_REPAIR: repair_provider,
                    },
                },
                "validator_a": {
                    "complete": True,
                    "machine_pass": a_pass,
                    "reasons": list(a_reasons),
                    "result": a_result,
                    "provider": dict(a_providers),
                },
            }
        )
        if a_result.get("contract_exhausted") is True:
            record.update(
                {
                    "terminal_status": TERMINAL_VALIDATOR_A_CONTRACT_REJECT,
                    "rejection_stage": ROLE_VALIDATOR_A_STYLE_REPAIR,
                    "rejection_reasons": list(a_reasons),
                }
            )
            return _store_terminal(
                output,
                row,
                record,
                code_sha256=code_sha256,
                official_reference_bank_sha256=official_reference_bank_sha256,
            )
        if not a_pass:
            record.update(
                {
                    "terminal_status": TERMINAL_STYLE_FIDELITY_REJECT,
                    "rejection_stage": "style_repair_validator_a",
                    "rejection_reasons": list(a_reasons),
                }
            )
            return _store_terminal(
                output,
                row,
                record,
                code_sha256=code_sha256,
                official_reference_bank_sha256=official_reference_bank_sha256,
            )
        b_result, b_pass, b_reasons, b_provider = _validator_b_call(
            row,
            candidate,
            reference_bank,
            role=ROLE_VALIDATOR_B_STYLE_REPAIR,
            output=output,
            backend=backend,
            identity=identity,
            environment=environment,
            config=config,
            code_sha256=code_sha256,
            official_reference_bank_sha256=official_reference_bank_sha256,
        )
        b_providers.update(b_provider)
        if b_result.get("contract_exhausted") is True:
            record["validator_b"] = {
                "complete": True,
                "machine_pass": False,
                "mean_score": None,
                "min_score": None,
                "reasons": list(b_reasons),
                "result": b_result,
                "provider": dict(b_providers),
            }
            record.update(
                {
                    "terminal_status": TERMINAL_VALIDATOR_B_CONTRACT_REJECT,
                    "rejection_stage": ROLE_VALIDATOR_B_STYLE_REPAIR,
                    "rejection_reasons": list(b_reasons),
                }
            )
            return _store_terminal(
                output,
                row,
                record,
                code_sha256=code_sha256,
                official_reference_bank_sha256=official_reference_bank_sha256,
            )
    record["validator_b"] = {
        "complete": True,
        "machine_pass": b_pass,
        "mean_score": b_result["mean_score"],
        "min_score": b_result["min_score"],
        "reasons": list(b_reasons),
        "result": b_result,
        "provider": dict(b_providers),
    }
    if not b_pass:
        reference_unavailable = (
            b_result.get("comparison_status") == "no_comparable_passage"
        )
        record.update(
            {
                "terminal_status": (
                    TERMINAL_REFERENCE_UNAVAILABLE_REJECT
                    if reference_unavailable
                    else TERMINAL_STYLE_REJECT
                ),
                "rejection_stage": (
                    "validator_b_reference_comparison"
                    if reference_unavailable
                    else "validator_b_style_gate"
                ),
                "rejection_reasons": list(b_reasons),
            }
        )
    else:
        record.update(
            {
                "terminal_status": TERMINAL_PASS,
                "training_pass": True,
                "rejection_stage": None,
                "rejection_reasons": [],
                "student_prompt": _student_prompt(row),
                "sft_response": candidate.sft_response,
                "prompt_sha256": sha256_text(_student_prompt(row)),
                "response_sha256": sha256_text(candidate.sft_response),
            }
        )
    return _store_terminal(
        output,
        row,
        record,
        code_sha256=code_sha256,
        official_reference_bank_sha256=official_reference_bank_sha256,
    )


def _implementation_contract() -> dict[str, Any]:
    artifacts = {
        "downstream_v6": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": v5.sha256_file(Path(__file__).resolve()),
        },
        "source_handoff_verifier": {
            "path": str(Path(handoff_v1.__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": v5.sha256_file(Path(handoff_v1.__file__).resolve()),
        },
        "v5_downstream_semantics": v5._implementation_contract(),
    }
    return {
        "artifacts": artifacts,
        "composite_sha256": sha256_text(canonical_json(artifacts)),
    }


def _handoff_sha(handoff: handoff_v1.SourceHandoff) -> str:
    value = handoff.manifest.get("manifest_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise SyntheticRewriteError("source handoff manifest SHA is invalid")
    return value


def _run_binding_sha(implementation_sha: str, handoff_sha: str) -> str:
    return sha256_text(
        canonical_json(
            {
                "implementation_composite_sha256": implementation_sha,
                "source_handoff_manifest_sha256": handoff_sha,
            }
        )
    )


def _store_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = canonical_json(dict(payload)) + "\n"
    if path.is_symlink():
        raise SyntheticRewriteError(f"immutable v6 artifact is a symlink: {path}")
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise SyntheticRewriteError(f"immutable v6 artifact drift: {path}")
        return
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except FileExistsError:
        if path.is_symlink() or path.read_text(encoding="utf-8") != serialized:
            raise SyntheticRewriteError(f"immutable v6 artifact race: {path}")


def _copy_immutable(source: Path, target: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise SyntheticRewriteError(f"handoff reference artifact is unsafe: {source}")
    content = source.read_text(encoding="utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise SyntheticRewriteError(f"immutable reference copy is a symlink: {target}")
    if target.exists():
        if target.read_text(encoding="utf-8") != content:
            raise SyntheticRewriteError(f"immutable reference copy drift: {target}")
        return
    try:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(content)
    except FileExistsError:
        if target.is_symlink() or target.read_text(encoding="utf-8") != content:
            raise SyntheticRewriteError(f"immutable reference copy race: {target}")


def _prompt_contract(
    *,
    implementation: Mapping[str, Any],
    handoff_sha256: str,
    run_binding_sha256: str,
    config: ProviderConfig,
) -> dict[str, Any]:
    v5_implementation = v5._implementation_contract()
    inherited = v5._prompt_contract(
        code_sha256=str(v5_implementation["composite_sha256"]), config=config
    )
    return {
        "schema_version": PROMPT_SCHEMA_VERSION,
        "implementation": dict(implementation),
        "implementation_composite_sha256": implementation["composite_sha256"],
        "source_handoff_manifest_sha256": handoff_sha256,
        "run_binding_sha256": run_binding_sha256,
        "source_audit_provider_roles_forbidden": True,
        "inherited_v5_downstream_prompt_contract": inherited,
    }


def _run_wave(
    rows: Sequence[PreparedRow], *, worker: Any, concurrency: int
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    results: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(worker, row): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                results[row.sample_id] = future.result()
            except Exception as exc:
                failures.append(
                    {
                        "sample_id": row.sample_id,
                        "split": row.split,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    failures.sort(key=lambda item: (item["split"], item["sample_id"]))
    return results, failures


def _select_downstream_preflight(
    prepared: Mapping[str, Sequence[PreparedRow]],
    source_results: Mapping[str, Mapping[str, Any]],
    *,
    worker: Any,
    concurrency: int,
) -> tuple[list[PreparedRow], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    selected: list[PreparedRow] = []
    terminals: dict[str, dict[str, Any]] = {}
    attempts: list[dict[str, Any]] = []
    for split in SPLITS:
        candidates = [
            row
            for row in prepared[split]
            if source_results[row.sample_id].get("machine_pass") is True
        ]
        cursor = 0
        while (
            sum(row.split == split for row in selected) < PREFLIGHT_SPLIT_COUNTS[split]
        ):
            need = PREFLIGHT_SPLIT_COUNTS[split] - sum(
                row.split == split for row in selected
            )
            wave = candidates[cursor : cursor + need]
            cursor += len(wave)
            if not wave:
                raise SyntheticRewriteError(
                    f"v6 preflight cannot fill terminal-PASS quota: {split}"
                )
            values, failures = _run_wave(wave, worker=worker, concurrency=concurrency)
            if failures:
                raise SyntheticRewriteError(
                    f"v6 preflight has unresolved provider failure: {failures[0]}"
                )
            terminals.update(values)
            for row in wave:
                terminal = values[row.sample_id]
                passed = (
                    terminal.get("terminal_status") == TERMINAL_PASS
                    and terminal.get("training_pass") is True
                    and v5._preflight_tokenizer_receipt(terminal) is not None
                )
                attempts.append(
                    {
                        "sample_id": row.sample_id,
                        "split": split,
                        "terminal_status": terminal.get("terminal_status"),
                        "selected": passed,
                    }
                )
                if passed:
                    selected.append(row)
    return selected, terminals, attempts


def _execution_receipt(
    *,
    phase: str,
    concurrency: int,
    implementation: Mapping[str, Any],
    handoff_sha256: str,
    run_binding_sha256: str,
    identities: Mapping[str, Any],
    status: str,
    output: Path,
    artifacts: Mapping[str, Any],
    handoff_manifest_file_sha256: str,
) -> dict[str, Any]:
    cache_counts = {
        role: len(list((output / "cache" / role).glob("*.json")))
        for role in (*v5.PROVIDER_ROLES, "terminal")
        if (output / "cache" / role).is_dir()
    }
    receipt = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "phase": phase,
        "status": status,
        "configured_concurrency": concurrency,
        "maximum_concurrency": MAX_CONCURRENCY,
        "runner_sha256": implementation["artifacts"]["downstream_v6"]["sha256"],
        "implementation_composite_sha256": implementation["composite_sha256"],
        "source_handoff_manifest_sha256": handoff_sha256,
        "source_handoff_manifest_file_sha256": handoff_manifest_file_sha256,
        "run_binding_sha256": run_binding_sha256,
        "prompt_contract_sha256": v5.sha256_file(output / "prompt_contract.json"),
        "official_reference_bank_sha256": v5.sha256_file(
            output / "official_pre_action_reference_bank.jsonl"
        ),
        "tokenizer_contract_sha256": sha256_text(
            canonical_json(v5._expected_tokenizer_runtime_contract())
        ),
        "provider_identities": dict(identities),
        "source_audit_provider_calls": 0,
        "cache_counts": cache_counts,
        "terminal_cache_count": cache_counts.get("terminal", 0),
        "artifacts": dict(artifacts),
    }
    receipt["receipt_sha256"] = sha256_text(canonical_json(receipt))
    return receipt


def _validate_execution_receipt_header(
    receipt: Mapping[str, Any],
    *,
    phase: str,
    implementation: Mapping[str, Any],
) -> None:
    unsigned = dict(receipt)
    stored_receipt_sha = unsigned.pop("receipt_sha256", None)
    expected_status = {
        "prepare": "prepared",
        "generate": "generation_complete",
        "verify": "complete",
        "all": "complete",
    }.get(phase)
    if (
        set(receipt) != EXECUTION_RECEIPT_FIELDS
        or receipt.get("schema_version") != EXECUTION_SCHEMA_VERSION
        or receipt.get("phase") != phase
        or receipt.get("status") != expected_status
        or receipt.get("maximum_concurrency") != MAX_CONCURRENCY
        or not isinstance(receipt.get("configured_concurrency"), int)
        or not 1 <= receipt["configured_concurrency"] <= MAX_CONCURRENCY
        or (phase != "prepare" and receipt["configured_concurrency"] != 128)
        or receipt.get("runner_sha256")
        != implementation["artifacts"]["downstream_v6"]["sha256"]
        or stored_receipt_sha != sha256_text(canonical_json(unsigned))
    ):
        raise SyntheticRewriteError("v6 execution receipt header drift")


def _validate_preflight_header(
    preflight: Mapping[str, Any],
    *,
    handoff_sha256: str,
    split_by_id: Mapping[str, str],
) -> None:
    selected = preflight.get("selected_ids")
    if (
        set(preflight)
        != {
            "schema_version",
            "source_handoff_manifest_sha256",
            "target_split_counts",
            "selected_ids",
            "attempts",
            "passed",
        }
        or preflight.get("schema_version") != SCHEMA_VERSION
        or preflight.get("source_handoff_manifest_sha256") != handoff_sha256
        or preflight.get("target_split_counts") != PREFLIGHT_SPLIT_COUNTS
        or preflight.get("passed") is not True
        or not isinstance(preflight.get("attempts"), list)
        or not isinstance(selected, list)
        or len(selected) != len(set(selected))
        or any(sample_id not in split_by_id for sample_id in selected)
        or Counter(split_by_id[sample_id] for sample_id in selected)
        != Counter(PREFLIGHT_SPLIT_COUNTS)
    ):
        raise SyntheticRewriteError("v6 preflight contract drift")


def _materialize_imported_source_receipt(
    output: Path,
    handoff: handoff_v1.SourceHandoff,
    handoff_sha256: str,
) -> None:
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        counts[split] = dict(Counter(row["status"] for row in handoff.records[split]))
    _store_immutable(
        output / "source_admission_receipt.json",
        {
            "schema_version": "paper-chk2-downstream-imported-source-v1",
            "status": "source_admission_complete",
            "quality_status": "passed",
            "source_handoff_manifest_sha256": handoff_sha256,
            "source_rows": len(handoff.source_results),
            "admitted_rows": sum(
                result.get("machine_pass") is True
                for result in handoff.source_results.values()
            ),
            "unresolved_rows": 0,
            "status_counts": counts,
        },
    )


def run_pipeline(
    handoff: handoff_v1.SourceHandoff,
    *,
    output_root: str | Path,
    tokenizer: Any,
    backend: ProviderBackend,
    environment: Mapping[str, str] | None,
    phase: str = "all",
    concurrency: int = DEFAULT_CONCURRENCY,
    resume: bool = False,
) -> dict[str, Any]:
    if phase not in {"prepare", "generate", "verify", "all"}:
        raise SyntheticRewriteError(f"invalid v6 downstream phase: {phase}")
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise SyntheticRewriteError(f"concurrency must be in [1,{MAX_CONCURRENCY}]")
    if phase != "prepare" and concurrency != 128:
        raise SyntheticRewriteError(
            "v6 downstream acquisition requires concurrency=128"
        )
    output = Path(output_root).resolve()
    implementation = _implementation_contract()
    handoff_sha256 = _handoff_sha(handoff)
    handoff_file_sha256 = v5.sha256_file(handoff.root / "handoff_manifest.json")
    binding_sha256 = _run_binding_sha(
        str(implementation["composite_sha256"]), handoff_sha256
    )
    config = ProviderConfig()
    _store_immutable(
        output / "prompt_contract.json",
        _prompt_contract(
            implementation=implementation,
            handoff_sha256=handoff_sha256,
            run_binding_sha256=binding_sha256,
            config=config,
        ),
    )
    reference_source = (
        handoff.root / "sealed_source/official_pre_action_reference_bank.jsonl"
    )
    reference_target = output / "official_pre_action_reference_bank.jsonl"
    _copy_immutable(reference_source, reference_target)
    reference_bank = official_v2.deserialize_official_reference_bank(
        reference_target.read_bytes()
    )
    reference_sha256 = v5.sha256_file(reference_target)
    _materialize_imported_source_receipt(output, handoff, handoff_sha256)
    _store_immutable(
        output / "preparation_summary.json",
        {
            "schema_version": "paper-chk2-downstream-preparation-v1",
            "source": {
                "source_handoff_manifest_sha256": handoff_sha256,
                "training_only": True,
            },
        },
    )
    if phase == "prepare":
        receipt = _execution_receipt(
            phase=phase,
            concurrency=concurrency,
            implementation=implementation,
            handoff_sha256=handoff_sha256,
            run_binding_sha256=binding_sha256,
            identities={},
            status="prepared",
            output=output,
            artifacts={},
            handoff_manifest_file_sha256=handoff_file_sha256,
        )
        _store_immutable(output / "execution_receipts/prepare.json", receipt)
        return receipt
    if tokenizer is None:
        raise SyntheticRewriteError("tokenizer is required for downstream acquisition")
    v5._verify_tokenizer_runtime_contract(tokenizer)
    for source_role in SOURCE_PROVIDER_ROLES:
        source_cache_root = output / "cache" / source_role
        if (
            source_cache_root.is_dir()
            and next(source_cache_root.glob("*.json"), None) is not None
        ):
            raise SyntheticRewriteError(
                f"v6 source-role cache namespace is forbidden: {source_role}"
            )
    if not resume:
        for role in (*v5.PROVIDER_ROLES, "terminal"):
            root = output / "cache" / role
            if root.is_dir() and next(root.glob("*.json"), None) is not None:
                raise SyntheticRewriteError(
                    f"v6 downstream cache exists; use --resume: {role}"
                )
    identity = ProviderIdentityRegistry()

    def terminal_worker(row: PreparedRow) -> dict[str, Any]:
        return _process_downstream_terminal(
            row,
            output=output,
            tokenizer=tokenizer,
            reference_bank=reference_bank,
            official_reference_bank_sha256=reference_sha256,
            backend=backend,
            identity=identity,
            environment=environment,
            config=config,
            code_sha256=binding_sha256,
            source_audit=handoff.source_results[row.sample_id],
        )

    selected, preflight_terminals, attempts = _select_downstream_preflight(
        handoff.prepared,
        handoff.source_results,
        worker=terminal_worker,
        concurrency=concurrency,
    )
    _store_immutable(
        output / "preflight_downstream.json",
        {
            "schema_version": SCHEMA_VERSION,
            "source_handoff_manifest_sha256": handoff_sha256,
            "target_split_counts": PREFLIGHT_SPLIT_COUNTS,
            "selected_ids": [row.sample_id for row in selected],
            "attempts": attempts,
            "passed": len(selected) == sum(PREFLIGHT_SPLIT_COUNTS.values()),
        },
    )
    selected_ids = {row.sample_id for row in selected}
    admitted = [
        row
        for split in SPLITS
        for row in handoff.prepared[split]
        if handoff.source_results[row.sample_id].get("machine_pass") is True
    ]
    if phase == "generate":

        def generation_worker(row: PreparedRow) -> dict[str, Any]:
            outcome = _generation_primary(
                row,
                output=output,
                tokenizer=tokenizer,
                backend=backend,
                identity=identity,
                environment=environment,
                config=config,
                code_sha256=binding_sha256,
            )
            return {
                "sample_id": row.sample_id,
                "deterministic_pass": outcome.candidate is not None,
                "reasons": list(outcome.reasons),
            }

        generated, failures = _run_wave(
            [row for row in admitted if row.sample_id not in selected_ids],
            worker=generation_worker,
            concurrency=concurrency,
        )
        if failures:
            v5.legacy._write_jsonl(output / "failures.jsonl", failures)
            raise SyntheticRewriteError("v6 generation has unresolved failures")
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "generation_complete_verification_pending",
            "source_handoff_manifest_sha256": handoff_sha256,
            "source_admitted": len(admitted),
            "generation_classified": len(generated) + len(selected),
            "training_ready": False,
        }
        _store_immutable(output / "generation_summary.json", summary)
        v5.legacy._write_jsonl(output / "failures.jsonl", [])
        receipt_status = "generation_complete"
        result: dict[str, Any] = summary
    else:
        all_rows = [row for split in SPLITS for row in handoff.prepared[split]]
        remaining = [
            row for row in all_rows if row.sample_id not in preflight_terminals
        ]
        rest, failures = _run_wave(
            remaining, worker=terminal_worker, concurrency=concurrency
        )
        if failures:
            v5.legacy._write_jsonl(output / "failures.jsonl", failures)
            raise SyntheticRewriteError("v6 verification has unresolved failures")
        terminals = {**preflight_terminals, **rest}
        tokenizer_rows: list[dict[str, Any]] = []
        for split in SPLITS:
            for row in handoff.prepared[split]:
                terminal = terminals[row.sample_id]
                if terminal.get("terminal_status") != TERMINAL_PASS:
                    continue
                replay = v5._tokenizer_replay(
                    row=row, response=terminal["sft_response"], tokenizer=tokenizer
                )
                stored_replay = v5._preflight_tokenizer_receipt(terminal)
                if stored_replay is None or replay != stored_replay:
                    raise SyntheticRewriteError(
                        f"v6 full tokenizer replay failed: {row.sample_id}"
                    )
                tokenizer_rows.append(
                    {"sample_id": row.sample_id, "split": split, **replay}
                )
        tokenizer_path = output / "audits/tokenizer_replay.jsonl"
        v5.legacy._write_jsonl(tokenizer_path, tokenizer_rows)
        result = v5._materialize_final(
            handoff.prepared,
            terminals,
            output=output,
            preparation={"source": {"source_handoff_manifest_sha256": handoff_sha256}},
            prompt_contract_sha256=v5.sha256_file(output / "prompt_contract.json"),
            official_reference_bank_sha256=reference_sha256,
            identities=identity.as_dict(),
        )
        result["schema_version"] = SCHEMA_VERSION
        result["source_handoff_manifest_sha256"] = handoff_sha256
        result["implementation_composite_sha256"] = implementation["composite_sha256"]
        result["run_binding_sha256"] = binding_sha256
        result.setdefault("artifacts", {})["tokenizer_replay"] = v5._artifact(
            tokenizer_path, rows=len(tokenizer_rows)
        )
        result["tokenizer_replay_pass_rows"] = len(tokenizer_rows)
        v5.legacy._write_json(output / "final_summary.json", result)
        v5.legacy._write_jsonl(output / "failures.jsonl", [])
        receipt_status = "complete"
    receipt_artifacts = dict(result.get("artifacts", {}))
    summary_name = (
        "generation_summary.json" if phase == "generate" else "final_summary.json"
    )
    summary_path = output / summary_name
    if summary_path.is_file():
        receipt_artifacts["phase_summary"] = v5._artifact(summary_path)
    preflight_path = output / "preflight_downstream.json"
    if preflight_path.is_file():
        receipt_artifacts["preflight_downstream"] = v5._artifact(preflight_path)
    receipt = _execution_receipt(
        phase=phase,
        concurrency=concurrency,
        implementation=implementation,
        handoff_sha256=handoff_sha256,
        run_binding_sha256=binding_sha256,
        identities=identity.as_dict(),
        status=receipt_status,
        output=output,
        artifacts=receipt_artifacts,
        handoff_manifest_file_sha256=handoff_file_sha256,
    )
    _store_immutable(output / f"execution_receipts/{phase}.json", receipt)
    return result


def load_and_verify_downstream(
    downstream_root: str | Path,
    *,
    source_handoff: handoff_v1.SourceHandoff,
    tokenizer: Any,
    phase: str = "all",
) -> dict[str, Any]:
    """Replay a completed v6 phase without making provider requests."""
    root = Path(downstream_root).resolve()
    implementation = _implementation_contract()
    handoff_sha256 = _handoff_sha(source_handoff)
    binding_sha256 = _run_binding_sha(
        str(implementation["composite_sha256"]), handoff_sha256
    )
    config = ProviderConfig()
    expected_prompt = _prompt_contract(
        implementation=implementation,
        handoff_sha256=handoff_sha256,
        run_binding_sha256=binding_sha256,
        config=config,
    )
    prompt = v5.legacy._read_json(root / "prompt_contract.json", label="v6 prompt")
    if prompt != expected_prompt:
        raise SyntheticRewriteError("v6 prompt contract drift")
    reference_path = root / "official_pre_action_reference_bank.jsonl"
    sealed_reference = (
        source_handoff.root / "sealed_source/official_pre_action_reference_bank.jsonl"
    )
    if (
        not reference_path.is_file()
        or not sealed_reference.is_file()
        or v5.sha256_file(reference_path) != v5.sha256_file(sealed_reference)
    ):
        raise SyntheticRewriteError("v6 sealed reference drift")
    receipt_path = root / f"execution_receipts/{phase}.json"
    receipt = v5.legacy._read_json(receipt_path, label="v6 execution receipt")
    _validate_execution_receipt_header(
        receipt, phase=phase, implementation=implementation
    )
    if (
        receipt.get("implementation_composite_sha256")
        != implementation["composite_sha256"]
        or receipt.get("source_handoff_manifest_sha256") != handoff_sha256
        or receipt.get("source_handoff_manifest_file_sha256")
        != v5.sha256_file(source_handoff.root / "handoff_manifest.json")
        or receipt.get("run_binding_sha256") != binding_sha256
        or receipt.get("prompt_contract_sha256")
        != v5.sha256_file(root / "prompt_contract.json")
        or receipt.get("official_reference_bank_sha256")
        != v5.sha256_file(reference_path)
        or receipt.get("tokenizer_contract_sha256")
        != sha256_text(canonical_json(v5._expected_tokenizer_runtime_contract()))
        or receipt.get("source_audit_provider_calls") != 0
    ):
        raise SyntheticRewriteError("v6 execution receipt drift")
    observed_cache_counts = {
        role: len(list((root / "cache" / role).glob("*.json")))
        for role in (*v5.PROVIDER_ROLES, "terminal")
        if (root / "cache" / role).is_dir()
    }
    if receipt.get("cache_counts") != observed_cache_counts or receipt.get(
        "terminal_cache_count"
    ) != observed_cache_counts.get("terminal", 0):
        raise SyntheticRewriteError("v6 execution cache-count drift")

    def verify_descriptors(value: Any, label: str) -> None:
        if not isinstance(value, Mapping):
            raise SyntheticRewriteError(f"v6 artifact descriptor drift: {label}")
        if {"path", "sha256", "bytes"}.issubset(value):
            raw_path = value.get("path")
            if not isinstance(raw_path, str):
                raise SyntheticRewriteError(f"v6 artifact path drift: {label}")
            artifact_path = Path(raw_path)
            if not artifact_path.is_absolute():
                artifact_path = REPO_ROOT / artifact_path
            artifact_path = artifact_path.resolve()
            try:
                artifact_path.relative_to(root)
            except ValueError as exc:
                raise SyntheticRewriteError(
                    f"v6 artifact escapes downstream root: {label}"
                ) from exc
            if (
                not artifact_path.is_file()
                or artifact_path.stat().st_size != value.get("bytes")
                or v5.sha256_file(artifact_path) != value.get("sha256")
            ):
                raise SyntheticRewriteError(f"v6 artifact content drift: {label}")
            if "rows" in value:
                rows = len(artifact_path.read_text(encoding="utf-8").splitlines())
                if rows != value.get("rows"):
                    raise SyntheticRewriteError(f"v6 artifact row drift: {label}")
            return
        for key, nested in value.items():
            verify_descriptors(nested, f"{label}.{key}")

    verify_descriptors(receipt.get("artifacts"), "artifacts")
    if phase == "prepare":
        return receipt
    v5._verify_tokenizer_runtime_contract(tokenizer)
    reference_bank = official_v2.deserialize_official_reference_bank(
        reference_path.read_bytes()
    )
    reference_sha256 = v5.sha256_file(reference_path)
    identity = ProviderIdentityRegistry()
    terminals: dict[str, dict[str, Any]] = {}
    expected_tokenizer_rows: list[dict[str, Any]] = []
    expected_cache_paths: dict[str, set[Path]] = {
        role: set() for role in (*v5.PROVIDER_ROLES, "terminal")
    }
    preflight = v5.legacy._read_json(
        root / "preflight_downstream.json", label="v6 downstream preflight"
    )
    split_by_id = {
        row.sample_id: split
        for split in SPLITS
        for row in source_handoff.prepared[split]
    }
    _validate_preflight_header(
        preflight, handoff_sha256=handoff_sha256, split_by_id=split_by_id
    )
    attempted_ids: list[str] = []
    selected_ids: list[str] = []
    preflight_terminals: dict[str, dict[str, Any]] = {}
    for attempt in preflight["attempts"]:
        if not isinstance(attempt, Mapping) or set(attempt) != {
            "sample_id",
            "split",
            "terminal_status",
            "selected",
        }:
            raise SyntheticRewriteError("v6 preflight attempt shape drift")
        sample_id = attempt.get("sample_id")
        split = attempt.get("split")
        if not isinstance(sample_id, str) or split not in SPLITS:
            raise SyntheticRewriteError("v6 preflight attempt identity drift")
        row = next(
            (
                item
                for item in source_handoff.prepared[split]
                if item.sample_id == sample_id
            ),
            None,
        )
        if (
            row is None
            or source_handoff.source_results[sample_id].get("machine_pass") is not True
            or sample_id in attempted_ids
        ):
            raise SyntheticRewriteError("v6 preflight admission/order drift")
        attempted_ids.append(sample_id)
        terminal = _load_downstream_terminal(
            root,
            row,
            code_sha256=binding_sha256,
            official_reference_bank_sha256=reference_sha256,
            identity=identity,
            reference_bank=reference_bank,
            config=config,
            environment={},
            source_audit=source_handoff.source_results[sample_id],
            tokenizer=tokenizer,
        )
        if terminal is None:
            raise SyntheticRewriteError(f"v6 preflight terminal missing: {sample_id}")
        preflight_terminals[sample_id] = terminal
        replay_pass = (
            terminal.get("terminal_status") == TERMINAL_PASS
            and terminal.get("training_pass") is True
            and v5._tokenizer_replay(
                row=row, response=terminal["sft_response"], tokenizer=tokenizer
            )
            == v5._preflight_tokenizer_receipt(terminal)
        )
        if (
            attempt.get("terminal_status") != terminal.get("terminal_status")
            or attempt.get("selected") is not replay_pass
        ):
            raise SyntheticRewriteError("v6 preflight terminal verdict drift")
        if replay_pass:
            selected_ids.append(sample_id)
    admitted_by_split = {
        split: [
            row.sample_id
            for row in source_handoff.prepared[split]
            if source_handoff.source_results[row.sample_id].get("machine_pass") is True
        ]
        for split in SPLITS
    }
    attempts_by_split = {
        split: [sid for sid in attempted_ids if sid in set(admitted_by_split[split])]
        for split in SPLITS
    }
    if (
        any(
            ids != admitted_by_split[split][: len(ids)]
            for split, ids in attempts_by_split.items()
        )
        or preflight["selected_ids"] != selected_ids
        or Counter(
            next(
                row.split
                for split in SPLITS
                for row in source_handoff.prepared[split]
                if row.sample_id == sid
            )
            for sid in selected_ids
        )
        != Counter(PREFLIGHT_SPLIT_COUNTS)
    ):
        raise SyntheticRewriteError("v6 preflight selection/top-up drift")
    row_by_id = {
        row.sample_id: row for split in SPLITS for row in source_handoff.prepared[split]
    }
    for sample_id, terminal in preflight_terminals.items():
        row = row_by_id[sample_id]
        expected_cache_paths["terminal"].add(v5._terminal_path(root, row))
        for role in v5._terminal_provider_sidecars(row, terminal):
            if role not in SOURCE_PROVIDER_ROLES:
                expected_cache_paths[role].add(v5._cache_path(root, role, row))
    if phase == "generate":
        for split in SPLITS:
            for row in source_handoff.prepared[split]:
                if (
                    source_handoff.source_results[row.sample_id].get("machine_pass")
                    is not True
                    or row.sample_id in selected_ids
                ):
                    continue
                outcome = _generation_primary(
                    row,
                    output=root,
                    tokenizer=tokenizer,
                    backend=_NoProviderCalls(),
                    identity=identity,
                    environment={},
                    config=config,
                    code_sha256=binding_sha256,
                )
                for role in outcome.provider:
                    expected_cache_paths[role].add(v5._cache_path(root, role, row))
    if phase in {"all", "verify"}:
        for split in SPLITS:
            for row in source_handoff.prepared[split]:
                terminal = _load_downstream_terminal(
                    root,
                    row,
                    code_sha256=binding_sha256,
                    official_reference_bank_sha256=reference_sha256,
                    identity=identity,
                    reference_bank=reference_bank,
                    config=config,
                    environment={},
                    source_audit=source_handoff.source_results[row.sample_id],
                    tokenizer=tokenizer,
                )
                if terminal is None:
                    raise SyntheticRewriteError(
                        f"v6 terminal cache missing: {row.sample_id}"
                    )
                terminals[row.sample_id] = terminal
                expected_cache_paths["terminal"].add(v5._terminal_path(root, row))
                for role in v5._terminal_provider_sidecars(row, terminal):
                    if not role.startswith("source_audit"):
                        expected_cache_paths[role].add(v5._cache_path(root, role, row))
                if terminal.get("terminal_status") == TERMINAL_PASS:
                    replay = v5._tokenizer_replay(
                        row=row,
                        response=terminal["sft_response"],
                        tokenizer=tokenizer,
                    )
                    if replay != v5._preflight_tokenizer_receipt(terminal):
                        raise SyntheticRewriteError(
                            f"v6 tokenizer replay drift: {row.sample_id}"
                        )
                    expected_tokenizer_rows.append(
                        {"sample_id": row.sample_id, "split": split, **replay}
                    )
    for role, expected in expected_cache_paths.items():
        role_root = root / "cache" / role
        observed = set(role_root.glob("*.json")) if role_root.is_dir() else set()
        if observed != expected:
            raise SyntheticRewriteError(f"v6 orphan/missing cache drift: {role}")
    if receipt.get("provider_identities") != identity.as_dict():
        raise SyntheticRewriteError("v6 provider identity receipt drift")
    if phase == "generate":
        generation_summary = v5.legacy._read_json(
            root / "generation_summary.json", label="v6 generation summary"
        )
        admitted_count = sum(
            result.get("machine_pass") is True
            for result in source_handoff.source_results.values()
        )
        if (
            generation_summary.get("schema_version") != SCHEMA_VERSION
            or generation_summary.get("status")
            != "generation_complete_verification_pending"
            or generation_summary.get("source_handoff_manifest_sha256")
            != handoff_sha256
            or generation_summary.get("source_admitted") != admitted_count
            or generation_summary.get("generation_classified") != admitted_count
            or generation_summary.get("training_ready") is not False
        ):
            raise SyntheticRewriteError("v6 generation summary drift")
    if phase in {"all", "verify"}:
        tokenizer_audit_path = root / "audits/tokenizer_replay.jsonl"
        observed_tokenizer_rows = v5.legacy._read_jsonl(
            tokenizer_audit_path, label="v6 tokenizer replay audit"
        )
        if observed_tokenizer_rows != expected_tokenizer_rows:
            raise SyntheticRewriteError("v6 tokenizer replay audit drift")
        summary_path = root / "final_summary.json"
        summary = v5.legacy._read_json(summary_path, label="v6 final summary")
        if (
            summary.get("schema_version") != SCHEMA_VERSION
            or summary.get("status") != "complete"
            or summary.get("quality_status") != "passed"
            or summary.get("total_source_rows") != len(row_by_id)
            or summary.get("terminal_classified") != len(terminals)
            or summary.get("unresolved_failure_count") != 0
            or summary.get("prompt_contract_sha256")
            != v5.sha256_file(root / "prompt_contract.json")
            or summary.get("official_reference_bank_sha256") != reference_sha256
            or summary.get("source_admission_receipt_sha256")
            != v5.sha256_file(root / "source_admission_receipt.json")
            or summary.get("preparation_summary_sha256")
            != v5.sha256_file(root / "preparation_summary.json")
            or summary.get("provider_identities") != identity.as_dict()
            or summary.get("training_ready_candidate") is not True
            or summary.get("evaluation_eligible") is not False
            or summary.get("source_handoff_manifest_sha256") != handoff_sha256
            or summary.get("run_binding_sha256") != binding_sha256
            or summary.get("implementation_composite_sha256")
            != implementation["composite_sha256"]
            or summary.get("tokenizer_replay_pass_rows") != len(expected_tokenizer_rows)
            or receipt.get("terminal_cache_count") != len(terminals)
        ):
            raise SyntheticRewriteError("v6 final summary/receipt drift")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, default=DEFAULT_HANDOFF_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument(
        "--phase", choices=("prepare", "generate", "verify", "all"), default="all"
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handoff = handoff_v1.load_and_verify_source_handoff(args.handoff_root)
    tokenizer = (
        None if args.phase == "prepare" else v5._load_tokenizer(args.tokenizer_path)
    )
    if args.phase != "prepare" and not os.environ.get(API_KEY_ENV, "").strip():
        raise SyntheticRewriteError(f"missing provider credential: {API_KEY_ENV}")
    result = run_pipeline(
        handoff,
        output_root=args.output_root,
        tokenizer=tokenizer,
        backend=OpenAICompatibleBackend(),
        environment=os.environ,
        phase=args.phase,
        concurrency=args.concurrency,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
