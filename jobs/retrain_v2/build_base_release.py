"""Build the immutable retrain-v2 analysis base release from canonical chk1 data.

This is deliberately a second, independent gate.  It accepts only the immutable
automatically validated handoff emitted by :mod:`jobs.retrain_v2.chk1.release`; draft and
generation handoffs are not valid inputs.  Publication is fail-closed and uses
a fully fsynced staging tree followed by one atomic rename.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

from jobs.retrain_v2.chk1 import canonical_workflow
from jobs.retrain_v2.chk1.contracts import (
    FACT_CARD_SCHEMA_VERSION,
    GENERATOR_SYSTEM_PROMPT,
    HANDOFF_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    RELEASE_SCHEMA_VERSION,
    canonical_json,
    sha256_text,
)
from jobs.retrain_v2.chk1.release import QUALITY_SCHEMA_VERSION
from jobs.retrain_v2.chk1.verifier import (
    critic_accepts,
    validate_critic,
    validate_fact_card,
)
from jobs.retrain_v2.chk1.prompt_projection import (
    ANALYSIS_GRPO_SYSTEM_PROMPT,
    ANALYSIS_SFT_SYSTEM_PROMPT,
    GRPO_PROMPT_TOKEN_LIMIT,
    PROJECTION_POLICY,
    PROJECTION_SCHEMA_VERSION,
    SFT_COMPLETION_TOKEN_LIMIT,
    SFT_PROMPT_TOKEN_LIMIT,
    SFT_TOTAL_TOKEN_LIMIT,
    PromptProjectionError,
    build_chat_token_counter,
    project_prompt_to_budget,
    projection_contract_sha256,
    render_grpo_user_prompt,
    safe_grpo_fact_card,
)
from jobs.main.checkpoint_provenance import (
    fingerprint_model_payload,
    fingerprint_tokenizer_payload,
)
from jobs.retrain_v2.tokenizer_provenance import (
    TOKENIZER_LOADER_CONTRACT,
    TokenizerBundleError,
    materialize_tokenizer_bundle,
    snapshot_tokenizer_bundle,
)
from jobs.retrain_v2.sft_training_budget import (
    SFT_TRAINING_COMPLETION_TOKEN_LIMIT,
    SFT_TRAINING_PROMPT_TOKEN_LIMIT,
    SFT_TRAINING_SCHEMA_VERSION,
    SFT_TRAINING_TOTAL_TOKEN_LIMIT,
    build_training_sft_token_auditor,
)
from open_r1.provenance import fingerprint_artifact_path, sha256_file, validate_sha256
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v2 import (
    validate_judge_evidence,
)


SPLITS = ("train", "eval", "test")
AUDITS = (
    "schema",
    "token_budget",
    "reference_leakage",
    "point_in_time",
    "split_integrity",
    "target_consistency",
    "encoding",
    "teacher_grounding",
)
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TARGET_KEYS = {
    "target_minutes",
    "target_meeting_minutes",
    "current_meeting_minutes",
    "reference_answer",
    "gold_label",
    "current_vote",
    "current_actual_decision",
    "actual_decision",
    "decision_label",
    "teacher_response",
    "archived_response",
    "response",
    "reasoning",
    "final_analysis",
}
TARGET_KEYS_COLLAPSED = {key.replace("_", "") for key in TARGET_KEYS}
TARGET_MARKER = re.compile(
    r"(?i)(?:target[_ -]*(?:meeting[_ -]*)?minutes|current[_ -]*meeting[_ -]*minutes|"
    r"reference[_ -]*answer|gold[_ -]*label|current[_ -]*vote|"
    r"actual[_ -]*decision|same-meeting minutes)"
)
SFT_SYSTEM = ANALYSIS_SFT_SYSTEM_PROMPT
GRPO_SYSTEM = ANALYSIS_GRPO_SYSTEM_PROMPT
TOKEN_BUDGETS = {
    "analysis_sft": {
        "prompt": SFT_TRAINING_PROMPT_TOKEN_LIMIT,
        "completion": SFT_TRAINING_COMPLETION_TOKEN_LIMIT,
        "total": SFT_TRAINING_TOTAL_TOKEN_LIMIT,
    },
    "analysis_grpo": {"prompt": 2560, "completion": 1536},
}
SAFE_MODEL_FACT_KEYS = {"schema_version", "atomic_topic", "evidence"}
SFT_TOKEN_AUDIT_KEYS = {
    "schema_version",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "max_prompt_tokens",
    "max_completion_tokens",
    "max_total_tokens",
    "overflow_policy",
    "truncated",
    "passed",
}
QUALITY_KEYS = {
    "schema_version",
    "status",
    "gates",
    "population_binding_sha256",
    "split_binding_sha256",
    "topic_binding_sha256",
    "population_count",
    "split_counts",
    "candidate_counts",
    "meeting_counts",
    "accepted_meeting_counts",
    "meeting_binding_modes",
    "excluded_counts",
    "exclusion_reasons",
    "train_acceptance_rate",
    "required_train_acceptance_rate",
    "per_topic_acceptance_rates",
    "topics_below_threshold",
    "zero_tolerance_exclusions",
    "manifest_provenance",
    "invariants",
}
QUALITY_GATE_KEYS = {
    "train_acceptance",
    "per_topic_acceptance",
    "zero_tolerance_exclusions",
}
QUALITY_PROVENANCE_KEYS = {
    "generation_provenance_sha256",
    "prompt_template_sha256",
    "generator_tokenizer_sha256",
    "style_guide_sha256",
    "teacher_model_sha256",
    "tokenizer_sha256",
}
LEGACY_QUALITY_PROVENANCE_KEYS = QUALITY_PROVENANCE_KEYS | {
    "critic_model_sha256"
}
QUALITY_INVARIANT_KEYS = {
    "canonical_key_unique",
    "sample_id_unique",
    "prompt_hash_unique",
    "response_hash_unique",
    "meeting_splits_disjoint",
    "expected_population_covered_once",
    "expected_topic_roster_exact",
    "input_truncation_count",
    "teacher_prompt_leakage_count",
}


class BaseReleaseError(ValueError):
    """Raised before publication when any source or output invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaseReleaseError(message)


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows).encode("utf-8")


def _read_bytes(path: Path, *, label: str) -> bytes:
    _require(path.is_file(), f"{label} does not exist: {path}")
    _require(not path.is_symlink(), f"{label} must not be a symlink: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise BaseReleaseError(f"unable to read {label}: {path}") from exc


def _decode_utf8(payload: bytes, *, label: str) -> str:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BaseReleaseError(f"{label} is not strict UTF-8") from exc
    _require("\x00" not in text and "\ufffd" not in text, f"{label} has invalid characters")
    return text


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    text = _decode_utf8(_read_bytes(path, label=label), label=label)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BaseReleaseError(f"{label} is invalid JSON: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    text = _decode_utf8(_read_bytes(path, label=label), label=label)
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), 1):
        _require(line.strip() != "", f"{label} has a blank line at {number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BaseReleaseError(f"{label} has invalid JSON at line {number}") from exc
        _require(isinstance(value, dict), f"{label} line {number} is not an object")
        _assert_finite(value, label=f"{label}[{number - 1}]")
        rows.append(value)
    return rows


def _assert_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"{label} contains a non-finite number")
    elif isinstance(value, dict):
        for key, child in value.items():
            _require(isinstance(key, str), f"{label} contains a non-string key")
            _assert_finite(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, label=f"{label}[{index}]")


def _resolve_inside(root: Path, relative: Any, *, label: str) -> Path:
    _require(isinstance(relative, str) and relative, f"{label}.path is invalid")
    lexical = Path(relative)
    _require(not lexical.is_absolute(), f"{label}.path must be relative")
    _require(".." not in lexical.parts, f"{label}.path escapes its release")
    path = (root / lexical).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise BaseReleaseError(f"{label}.path escapes its release") from exc
    _require(not (root / lexical).is_symlink(), f"{label}.path must not be a symlink")
    return path


def _tree_has_no_symlinks(root: Path, *, label: str) -> None:
    _require(root.is_dir() and not root.is_symlink(), f"{label} must be a real directory")
    for path in root.rglob("*"):
        _require(not path.is_symlink(), f"{label} contains symlink: {path}")


def _tree_is_immutable(root: Path) -> None:
    writable = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    for path in (root, *root.rglob("*")):
        _require(path.stat().st_mode & writable == 0, f"source release is writable: {path}")


def _require_no_symlink_components(root: Path, path: Path, *, label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise BaseReleaseError(f"{label} escapes repository") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            _require(not current.is_symlink(), f"{label} contains symlink component: {current}")


def _required_text(row: Mapping[str, Any], key: str, *, label: str) -> str:
    value = row.get(key)
    _require(isinstance(value, str) and bool(value.strip()), f"{label}.{key} must be text")
    _require("\x00" not in value and "\ufffd" not in value, f"{label}.{key} encoding error")
    return value


def _required_sha(value: Any, *, label: str) -> str:
    try:
        validate_sha256(value, label=label)
    except ValueError as exc:
        raise BaseReleaseError(str(exc)) from exc
    return str(value)


def _parse_utc(value: Any, *, label: str) -> datetime:
    text = str(value or "").strip()
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise BaseReleaseError(f"{label} must be ISO-8601") from exc
    _require(parsed.tzinfo is not None and parsed.utcoffset() is not None, f"{label} lacks timezone")
    return parsed.astimezone(timezone.utc)


def _parse_meeting(value: Any, *, label: str) -> date:
    _require(isinstance(value, str), f"{label} must be YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise BaseReleaseError(f"{label} must be YYYY-MM-DD") from exc
    _require(parsed.isoformat() == value, f"{label} must be canonical YYYY-MM-DD")
    return parsed


def stable_train_role(sample_id: str) -> str:
    """Return the immutable 70/20/10 assignment for a train sample."""

    bucket = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < 70:
        return "sft_only"
    if bucket < 90:
        return "grpo_only"
    return "shared"


def _load_bound_rows(
    source_root: Path, handoff: Mapping[str, Any], key: str
) -> dict[str, list[dict[str, Any]]]:
    records = handoff.get(key)
    _require(isinstance(records, dict) and set(records) == set(SPLITS), f"handoff.{key} invalid")
    result: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        record = records[split]
        _require(isinstance(record, dict), f"handoff.{key}.{split} invalid")
        path = _resolve_inside(source_root, record.get("path"), label=f"handoff.{key}.{split}")
        digest = _required_sha(record.get("sha256"), label=f"handoff.{key}.{split}.sha256")
        _require(sha256_file(path) == digest, f"handoff.{key}.{split} SHA mismatch")
        rows = _load_jsonl(path, label=f"handoff.{key}.{split}")
        _require(record.get("rows") == len(rows), f"handoff.{key}.{split} row count mismatch")
        result[split] = rows
    return result


def _parse_fact_card(text: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BaseReleaseError(f"{label}.provided_data must be JSON") from exc
    _require(isinstance(value, dict), f"{label}.provided_data must be an object")
    return value


def _validate_sft_token_budget_claim(
    value: Any, *, label: str, require_passed: bool = True
) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    claim = dict(value)
    _require(set(claim) == SFT_TOKEN_AUDIT_KEYS, f"{label} keys are not canonical")
    expected_contract = {
        "schema_version": "chk1-sft-token-budget-v1",
        "max_prompt_tokens": SFT_PROMPT_TOKEN_LIMIT,
        "max_completion_tokens": SFT_COMPLETION_TOKEN_LIMIT,
        "max_total_tokens": SFT_TOTAL_TOKEN_LIMIT,
        "overflow_policy": "error",
        "truncated": False,
    }
    for key, expected in expected_contract.items():
        _require(claim.get(key) == expected, f"{label}.{key} contract mismatch")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        observed = claim.get(key)
        _require(
            isinstance(observed, int) and not isinstance(observed, bool) and observed >= 0,
            f"{label}.{key} must be a non-negative integer",
        )
    _require(
        claim["total_tokens"]
        == claim["prompt_tokens"] + claim["completion_tokens"],
        f"{label} token counts do not reconcile",
    )
    expected_passed = (
        claim["prompt_tokens"] <= SFT_PROMPT_TOKEN_LIMIT
        and claim["completion_tokens"] <= SFT_COMPLETION_TOKEN_LIMIT
        and claim["total_tokens"] <= SFT_TOTAL_TOKEN_LIMIT
    )
    _require(claim.get("passed") is expected_passed, f"{label}.passed is false or stale")
    if require_passed:
        _require(expected_passed, f"{label} exceeds the canonical SFT token budget")
    return claim


def _validate_training_sft_token_budget_claim(
    value: Any, *, label: str
) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an object")
    claim = dict(value)
    _require(set(claim) == SFT_TOKEN_AUDIT_KEYS, f"{label} keys are not canonical")
    expected_contract = {
        "schema_version": SFT_TRAINING_SCHEMA_VERSION,
        "max_prompt_tokens": SFT_TRAINING_PROMPT_TOKEN_LIMIT,
        "max_completion_tokens": SFT_TRAINING_COMPLETION_TOKEN_LIMIT,
        "max_total_tokens": SFT_TRAINING_TOTAL_TOKEN_LIMIT,
        "overflow_policy": "error",
        "truncated": False,
    }
    for key, expected in expected_contract.items():
        _require(claim.get(key) == expected, f"{label}.{key} contract mismatch")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        observed = claim.get(key)
        _require(
            isinstance(observed, int) and not isinstance(observed, bool) and observed >= 0,
            f"{label}.{key} must be a non-negative integer",
        )
    _require(
        claim["total_tokens"]
        == claim["prompt_tokens"] + claim["completion_tokens"],
        f"{label} token counts do not reconcile",
    )
    expected_passed = (
        claim["prompt_tokens"] <= SFT_TRAINING_PROMPT_TOKEN_LIMIT
        and claim["total_tokens"] <= SFT_TRAINING_TOTAL_TOKEN_LIMIT
    )
    _require(claim.get("passed") is expected_passed, f"{label}.passed is stale")
    _require(expected_passed, f"{label} exceeds the canonical training token budget")
    return claim


def _validate_projection_attestation(
    *,
    clean: Mapping[str, str],
    manifest: Mapping[str, Any],
    fact: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    generation = manifest["generation"]
    projection = generation.get("model_input_projection")
    outer_keys = {
        "schema_version",
        "projection_contract_sha256",
        "safe_model_fact_card_sha256",
        "validation_fact_card_sha256",
        "student_projection",
        "generator_prompt_sha256",
        "generator_system_prompt_sha256",
        "generator_prompt_tokens",
        "generator_max_prompt_tokens",
        "model_fact_card_keys",
        "removed_identity_fields",
        "input_truncated",
        "string_truncation",
        "attestation_sha256",
    }
    _require(isinstance(projection, dict), f"{label} model-input projection missing")
    _require(set(projection) == outer_keys, f"{label} model-input projection keys drifted")
    projection_payload = {
        key: value for key, value in projection.items() if key != "attestation_sha256"
    }
    _require(
        _required_sha(
            projection.get("attestation_sha256"),
            label=f"{label}.generation.model_input_projection.attestation_sha256",
        )
        == sha256_text(canonical_json(projection_payload)),
        f"{label} model-input projection attestation SHA mismatch",
    )
    contract_sha = projection_contract_sha256()
    _require(
        projection.get("schema_version") == PROJECTION_SCHEMA_VERSION
        and projection.get("projection_contract_sha256") == contract_sha,
        f"{label} model-input projection contract mismatch",
    )
    provided_sha = sha256_text(clean["provided_data"])
    _require(
        projection.get("safe_model_fact_card_sha256") == provided_sha,
        f"{label} safe model fact-card SHA mismatch",
    )
    _required_sha(
        projection.get("validation_fact_card_sha256"),
        label=f"{label}.generation.model_input_projection.validation_fact_card_sha256",
    )
    _required_sha(
        projection.get("generator_prompt_sha256"),
        label=f"{label}.generation.model_input_projection.generator_prompt_sha256",
    )
    _require(
        projection.get("generator_system_prompt_sha256")
        == sha256_text(GENERATOR_SYSTEM_PROMPT),
        f"{label} generator system prompt mismatch",
    )
    generator_tokens = projection.get("generator_prompt_tokens")
    _require(
        isinstance(generator_tokens, int)
        and not isinstance(generator_tokens, bool)
        and 0 <= generator_tokens <= 4096
        and projection.get("generator_max_prompt_tokens") == 4096,
        f"{label} generator prompt token audit invalid",
    )
    _require(
        projection.get("model_fact_card_keys")
        == ["atomic_topic", "evidence", "schema_version"],
        f"{label} model fact-card key attestation mismatch",
    )
    _require(
        projection.get("removed_identity_fields")
        == [
            "availability_upper_bound_ts",
            "canonical_key",
            "cutoff_ts",
            "meeting_date",
            "sample_id",
            "source_id",
        ],
        f"{label} removed identity field attestation mismatch",
    )
    _require(
        projection.get("input_truncated") is False
        and projection.get("string_truncation") is False,
        f"{label} model-input projection was truncated",
    )

    student = projection.get("student_projection")
    student_keys = {
        "schema_version",
        "policy",
        "projection_name",
        "projection_contract_sha256",
        "system_prompt_sha256",
        "source_fact_card_sha256",
        "projected_fact_card_sha256",
        "source_lineage_sha256",
        "projected_lineage_sha256",
        "prompt_sha256",
        "provided_data_sha256",
        "max_prompt_tokens",
        "prompt_tokens",
        "source_evidence_count",
        "projected_evidence_count",
        "selected_evidence_ids_sha256",
        "dropped_evidence_ids_sha256",
        "required_evidence_ids_sha256",
        "preferred_evidence_ids_sha256",
        "evidence_bindings",
        "evidence_bindings_sha256",
        "selection_attempt_count",
        "input_truncated",
        "string_truncation",
        "compaction_unit",
        "attestation_sha256",
    }
    _require(isinstance(student, dict), f"{label} student projection missing")
    _require(set(student) == student_keys, f"{label} student projection keys drifted")
    student_payload = {
        key: value for key, value in student.items() if key != "attestation_sha256"
    }
    _require(
        _required_sha(
            student.get("attestation_sha256"),
            label=f"{label}.generation.model_input_projection.student_projection.attestation_sha256",
        )
        == sha256_text(canonical_json(student_payload)),
        f"{label} student projection attestation SHA mismatch",
    )
    _require(
        student.get("schema_version") == PROJECTION_SCHEMA_VERSION
        and student.get("policy") == PROJECTION_POLICY
        and student.get("projection_name") == "analysis_sft_preteacher"
        and student.get("projection_contract_sha256") == contract_sha,
        f"{label} student projection contract mismatch",
    )
    _require(
        student.get("system_prompt_sha256") == sha256_text(SFT_SYSTEM)
        and student.get("prompt_sha256") == sha256_text(clean["prompt"])
        and student.get("provided_data_sha256") == provided_sha
        and student.get("projected_fact_card_sha256") == provided_sha,
        f"{label} student prompt/fact projection binding mismatch",
    )
    _require(
        student.get("max_prompt_tokens") == SFT_PROMPT_TOKEN_LIMIT
        and isinstance(student.get("prompt_tokens"), int)
        and not isinstance(student.get("prompt_tokens"), bool)
        and 0 <= student["prompt_tokens"] <= SFT_PROMPT_TOKEN_LIMIT,
        f"{label} student projection token audit invalid",
    )
    evidence = fact["evidence"]
    lineage = manifest["evidence_lineage"]
    evidence_ids = tuple(str(item["evidence_id"]) for item in evidence)
    _require(
        student.get("projected_evidence_count") == len(evidence)
        and isinstance(student.get("source_evidence_count"), int)
        and student["source_evidence_count"] >= len(evidence),
        f"{label} student projection evidence counts invalid",
    )
    _require(
        student.get("selected_evidence_ids_sha256")
        == sha256_text(canonical_json(evidence_ids)),
        f"{label} selected evidence binding mismatch",
    )
    expected_bindings = [
        {
            "evidence_id": str(evidence_item["evidence_id"]),
            "evidence_sha256": sha256_text(canonical_json(evidence_item)),
            "lineage_sha256": sha256_text(canonical_json(lineage_item)),
        }
        for evidence_item, lineage_item in zip(evidence, lineage, strict=True)
    ]
    _require(
        student.get("evidence_bindings") == expected_bindings
        and student.get("evidence_bindings_sha256")
        == sha256_text(canonical_json(expected_bindings)),
        f"{label} projected evidence attestation mismatch",
    )
    _require(
        student.get("projected_lineage_sha256")
        == sha256_text(canonical_json(lineage)),
        f"{label} projected lineage attestation mismatch",
    )
    empty_ids_sha = sha256_text(canonical_json(()))
    _require(
        student.get("required_evidence_ids_sha256") == empty_ids_sha
        and student.get("preferred_evidence_ids_sha256") == empty_ids_sha,
        f"{label} unexpected required/preferred evidence projection",
    )
    _require(
        student.get("input_truncated") is False
        and student.get("string_truncation") is False
        and student.get("compaction_unit")
        == "whole_evidence_object_with_dependency_closure"
        and isinstance(student.get("selection_attempt_count"), int)
        and student["selection_attempt_count"] >= 1,
        f"{label} student projection policy mismatch",
    )
    return projection


def _validate_prepared_projection_replay(
    *,
    clean: Mapping[str, str],
    manifest: Mapping[str, Any],
    safe_fact: Mapping[str, Any],
    prepared: Mapping[str, Any],
    label: str,
) -> None:
    """Rebuild the full PIT validation fact behind one safe model input."""

    prepared_digest = sha256_text(canonical_json(dict(prepared)))
    _require(
        manifest["generation"].get("prepared_row_sha256") == prepared_digest,
        f"{label} prepared-row digest mismatch",
    )
    for field in ("sample_id", "split", "meeting_date", "atomic_topic"):
        _require(
            manifest.get(field) == prepared.get(field),
            f"{label} prepared {field} identity mismatch",
        )
    full_fact_value = prepared.get("fact_card")
    full_lineage_value = prepared.get("evidence_lineage")
    _require(isinstance(full_fact_value, dict), f"{label} prepared fact card missing")
    _require(
        isinstance(full_lineage_value, list) and full_lineage_value,
        f"{label} prepared lineage missing",
    )
    full_fact = deepcopy(full_fact_value)
    full_lineage = deepcopy(full_lineage_value)
    expected_full_keys = {
        "schema_version",
        "sample_id",
        "canonical_key",
        "meeting_date",
        "atomic_topic",
        "cutoff_ts",
        "evidence",
    }
    _require(
        set(full_fact) == expected_full_keys,
        f"{label} prepared full fact-card keys drifted",
    )
    _require(
        full_fact.get("schema_version") == FACT_CARD_SCHEMA_VERSION,
        f"{label} prepared fact-card schema mismatch",
    )
    _require(
        _required_sha(
            prepared.get("generator_fact_card_sha256"),
            label=f"{label}.prepared.generator_fact_card_sha256",
        )
        == sha256_text(canonical_json(full_fact)),
        f"{label} prepared generator fact-card hash mismatch",
    )
    for field in ("sample_id", "meeting_date", "atomic_topic", "cutoff_ts"):
        _require(
            full_fact.get(field) == manifest.get(field),
            f"{label} prepared fact-card {field} mismatch",
        )
    full_evidence = full_fact.get("evidence")
    _require(
        isinstance(full_evidence, list) and full_evidence,
        f"{label} prepared full evidence missing",
    )
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(full_evidence):
        _require(
            isinstance(item, dict),
            f"{label} prepared evidence[{index}] is invalid",
        )
        evidence_id = _required_text(
            item, "evidence_id", label=f"{label}.prepared.evidence[{index}]"
        )
        _require(
            evidence_id not in evidence_by_id,
            f"{label} prepared evidence IDs are duplicated",
        )
        evidence_by_id[evidence_id] = item
    lineage_by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(full_lineage):
        _require(
            isinstance(item, dict),
            f"{label} prepared lineage[{index}] is invalid",
        )
        evidence_id = _required_text(
            item, "evidence_id", label=f"{label}.prepared.lineage[{index}]"
        )
        _require(
            evidence_id not in lineage_by_id,
            f"{label} prepared lineage IDs are duplicated",
        )
        lineage_by_id[evidence_id] = item
    _require(
        set(evidence_by_id) == set(lineage_by_id),
        f"{label} prepared evidence/lineage sets differ",
    )

    selected_ids = [str(item["evidence_id"]) for item in safe_fact["evidence"]]
    _require(
        len(selected_ids) == len(set(selected_ids))
        and set(selected_ids) <= set(evidence_by_id),
        f"{label} safe fact does not select prepared evidence exactly",
    )
    expected_lineage = [deepcopy(lineage_by_id[item]) for item in selected_ids]
    _require(
        manifest.get("evidence_lineage") == expected_lineage,
        f"{label} projected manifest lineage does not replay from preparation",
    )
    validation_fact = deepcopy(full_fact)
    validation_fact["evidence"] = [
        deepcopy(evidence_by_id[item]) for item in selected_ids
    ]
    projection = manifest["generation"]["model_input_projection"]
    student_projection = projection["student_projection"]
    _require(
        sha256_text(canonical_json(validation_fact))
        == projection["validation_fact_card_sha256"],
        f"{label} validation fact-card attestation does not replay",
    )
    fact_errors = validate_fact_card(validation_fact)
    _require(
        not fact_errors,
        f"{label} replayed point-in-time fact card failed: {fact_errors}",
    )
    try:
        expected_safe_fact = safe_grpo_fact_card(validation_fact)
        full_safe_fact = safe_grpo_fact_card(full_fact)
    except PromptProjectionError as exc:
        raise BaseReleaseError(
            f"{label} prepared safe fact-card replay failed: {exc}"
        ) from exc
    _require(
        expected_safe_fact == dict(safe_fact)
        and canonical_json(expected_safe_fact) == clean["provided_data"],
        f"{label} safe model fact card does not replay from preparation",
    )
    _require(
        student_projection["source_fact_card_sha256"]
        == sha256_text(canonical_json(full_safe_fact))
        and student_projection["source_lineage_sha256"]
        == sha256_text(canonical_json(full_lineage)),
        f"{label} source projection attestation does not replay",
    )
    dropped_ids = tuple(sorted(set(evidence_by_id) - set(selected_ids)))
    _require(
        student_projection["source_evidence_count"] == len(evidence_by_id)
        and student_projection["dropped_evidence_ids_sha256"]
        == sha256_text(canonical_json(dropped_ids)),
        f"{label} dropped-evidence attestation does not replay",
    )


def _validate_row_pair(
    sft: Mapping[str, Any], manifest: Mapping[str, Any], *, split: str, index: int
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    label = f"{split}[{index}]"
    _require(set(sft) == {"prompt", "response", "provided_data"}, f"{label} SFT schema invalid")
    clean = {key: _required_text(sft, key, label=label) for key in ("prompt", "response", "provided_data")}
    required = {
        "schema_version", "sample_id", "meeting_date", "atomic_topic", "section_style_id",
        "split", "cutoff_ts", "evidence_lineage", "style_guide_sha256",
        "teacher_model_sha256", "tokenizer_sha256", "generation",
        "prompt_sha256", "reasoning_sha256", "final_analysis_sha256", "response_sha256",
        "provided_data_sha256", "input_truncated",
    }
    _require(required <= set(manifest), f"{label} manifest missing {sorted(required - set(manifest))}")
    _require(manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION, f"{label} manifest schema invalid")
    _require(manifest.get("split") == split, f"{label} split mismatch")
    _required_text(manifest, "sample_id", label=label)
    meeting = _parse_meeting(manifest.get("meeting_date"), label=f"{label}.meeting_date")
    _required_text(manifest, "atomic_topic", label=label)
    _required_text(manifest, "section_style_id", label=label)
    cutoff = _parse_utc(manifest.get("cutoff_ts"), label=f"{label}.cutoff_ts")
    _require(cutoff <= datetime.combine(meeting, time.max, tzinfo=timezone.utc), f"{label} cutoff is after meeting")
    expected_hashes = {
        "prompt_sha256": sha256_text(clean["prompt"]),
        "response_sha256": sha256_text(clean["response"]),
        "provided_data_sha256": sha256_text(clean["provided_data"]),
    }
    for field, expected in expected_hashes.items():
        _require(_required_sha(manifest.get(field), label=f"{label}.{field}") == expected, f"{label} {field} mismatch")
    parts = clean["response"].split("\n</think>\n")
    _require(len(parts) == 2 and all(part.strip() for part in parts), f"{label} malformed reasoning boundary")
    _require("<think>" not in clean["response"].casefold(), f"{label} contains opening think token")
    _require(_required_sha(manifest.get("reasoning_sha256"), label=f"{label}.reasoning_sha256") == sha256_text(parts[0].strip()), f"{label} reasoning mismatch")
    _require(_required_sha(manifest.get("final_analysis_sha256"), label=f"{label}.final_analysis_sha256") == sha256_text(parts[1].strip()), f"{label} final mismatch")
    for field in ("style_guide_sha256", "teacher_model_sha256", "tokenizer_sha256"):
        _required_sha(manifest.get(field), label=f"{label}.{field}")
    legacy_fields = {"critic_model_sha256", "verifier", "critic"}
    present_legacy_fields = legacy_fields & set(manifest)
    _require(
        not present_legacy_fields or present_legacy_fields == legacy_fields,
        f"{label} has an incomplete legacy local-critic contract",
    )
    if present_legacy_fields:
        _required_sha(
            manifest.get("critic_model_sha256"),
            label=f"{label}.critic_model_sha256",
        )
        verifier = manifest.get("verifier")
        _require(
            isinstance(verifier, dict)
            and verifier.get("passed") is True
            and verifier.get("error_codes") == [],
            f"{label} verifier did not pass cleanly",
        )
        try:
            critic = validate_critic(manifest.get("critic"))
        except ValueError as exc:
            raise BaseReleaseError(f"{label} critic invalid: {exc}") from exc
        _require(critic_accepts(critic), f"{label} critic did not accept")
    _require(manifest.get("input_truncated") is False, f"{label} input was truncated")
    generation = manifest.get("generation")
    _require(isinstance(generation, dict), f"{label} generation invalid")
    for field in ("cache_key", "generation_provenance_sha256", "prompt_template_sha256", "generator_tokenizer_sha256", "student_tokenizer_sha256"):
        _required_sha(generation.get(field), label=f"{label}.generation.{field}")
    _require(generation["student_tokenizer_sha256"] == manifest["tokenizer_sha256"], f"{label} student tokenizer mismatch")
    _required_sha(
        generation.get("prepared_row_sha256"),
        label=f"{label}.generation.prepared_row_sha256",
    )
    _validate_sft_token_budget_claim(
        generation.get("sft_token_budget"),
        label=f"{label}.generation.sft_token_budget",
        require_passed=False,
    )

    fact = _parse_fact_card(clean["provided_data"], label=label)
    _require(
        canonical_json(fact) == clean["provided_data"],
        f"{label} provided_data is not canonical JSON",
    )
    _require(set(fact) == SAFE_MODEL_FACT_KEYS, f"{label} safe fact card keys invalid")
    _require(
        fact.get("schema_version") == FACT_CARD_SCHEMA_VERSION,
        f"{label} fact card schema version invalid",
    )
    _require(
        fact.get("atomic_topic") == manifest["atomic_topic"],
        f"{label} safe fact card atomic_topic mismatch",
    )
    _require(
        clean["prompt"].count(clean["provided_data"]) == 1,
        f"{label} student prompt must contain provided_data exactly once",
    )
    try:
        validate_judge_evidence(clean["provided_data"])
    except ValueError as exc:
        raise BaseReleaseError(f"{label} unsafe model-facing fact card: {exc}") from exc
    evidence = fact.get("evidence")
    lineage = manifest.get("evidence_lineage")
    _require(isinstance(evidence, list) and evidence, f"{label} fact card evidence empty")
    _require(isinstance(lineage, list) and lineage, f"{label} lineage empty")
    _require(
        len(evidence) == len(lineage),
        f"{label} evidence/lineage cardinality mismatch",
    )
    seen_evidence_ids: set[str] = set()
    for index, item in enumerate(evidence):
        _require(isinstance(item, dict), f"{label} evidence entry invalid")
        evidence_id = _required_text(item, "evidence_id", label=label)
        _require(
            evidence_id not in seen_evidence_ids,
            f"{label} has duplicate evidence IDs",
        )
        seen_evidence_ids.add(evidence_id)
        _required_sha(item.get("source_sha256"), label=f"{label}.evidence.source_sha256")
        dangerous_keys = {
            re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
            for key in _walk_keys(item)
        }
        _require(
            not {
                "sample_id",
                "meeting_date",
                "canonical_key",
                "cutoff_ts",
                "availability_upper_bound_ts",
                "source_id",
            }
            & dangerous_keys,
            f"{label}.evidence[{index}] retains target identity metadata",
        )
        observation = item.get("observation_date")
        if observation not in (None, ""):
            _require(
                _parse_meeting(
                    observation, label=f"{label}.evidence[{index}].observation_date"
                )
                <= cutoff.date(),
                f"{label}.evidence[{index}] observation is post-cutoff",
            )
        release_ts = item.get("release_ts")
        if release_ts not in (None, ""):
            _require(
                _parse_utc(
                    release_ts, label=f"{label}.evidence[{index}].release_ts"
                )
                <= cutoff,
                f"{label}.evidence[{index}] release is post-cutoff",
            )
    seen_lineage_ids: set[str] = set()
    for item in lineage:
        _require(isinstance(item, dict), f"{label} lineage entry invalid")
        evidence_id = _required_text(item, "evidence_id", label=label)
        _require(
            evidence_id not in seen_lineage_ids,
            f"{label} has duplicate lineage IDs",
        )
        seen_lineage_ids.add(evidence_id)
        _required_sha(item.get("source_sha256"), label=f"{label}.lineage.source_sha256")
        _require(item.get("cutoff_ts") == manifest["cutoff_ts"], f"{label} lineage cutoff mismatch")
        for sha_key in (
            "evidence_sha256",
            "raw_sha256",
            "request_id",
            "snapshot_manifest_payload_sha256",
            "registry_sha256",
        ):
            _required_sha(item.get(sha_key), label=f"{label}.lineage.{sha_key}")
    _require(
        seen_evidence_ids == seen_lineage_ids,
        f"{label} evidence/lineage binding mismatch",
    )
    for evidence_item, lineage_item in zip(evidence, lineage, strict=True):
        _require(
            evidence_item.get("evidence_id") == lineage_item.get("evidence_id")
            and evidence_item.get("source_sha256")
            == lineage_item.get("source_sha256"),
            f"{label} evidence/lineage order is not exact 1:1",
        )
    _validate_projection_attestation(
        clean=clean,
        manifest=manifest,
        fact=fact,
        label=label,
    )
    return clean, dict(manifest), fact


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _audit_tokens(
    tokenizer: Any,
    outputs: Mapping[str, Mapping[str, Sequence[dict[str, Any]]]],
    *,
    sft_token_audits: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        grpo_counter = build_chat_token_counter(tokenizer)
    except PromptProjectionError as exc:
        raise BaseReleaseError(f"unable to build GRPO token counter: {exc}") from exc
    details: dict[str, Any] = {}
    total_checked = 0
    for dataset, splits in outputs.items():
        maxima = {"prompt": 0, "completion": 0, "total": 0}
        split_counts: dict[str, int] = {}
        for split, rows in splits.items():
            split_counts[split] = len(rows)
            for index, row in enumerate(rows):
                if dataset == "analysis_sft":
                    row_audit = sft_token_audits.get(
                        (row["prompt"], row["response"])
                    )
                    _require(
                        row_audit is not None,
                        f"{dataset}/{split}[{index}] lacks a recomputed token audit",
                    )
                    checked_audit = _validate_training_sft_token_budget_claim(
                        row_audit,
                        label=f"{dataset}/{split}[{index}].token_audit",
                    )
                    prompt_count = checked_audit["prompt_tokens"]
                    completion_count = checked_audit["completion_tokens"]
                    total = checked_audit["total_tokens"]
                else:
                    try:
                        prompt_count = grpo_counter(GRPO_SYSTEM, row["prompt"])
                    except PromptProjectionError as exc:
                        raise BaseReleaseError(
                            f"{dataset}/{split}[{index}] token audit failed: {exc}"
                        ) from exc
                    completion_count = 0
                    total = prompt_count
                budget = TOKEN_BUDGETS[dataset]
                _require(prompt_count <= budget["prompt"], f"{dataset}/{split}[{index}] exceeds prompt budget")
                if dataset == "analysis_sft":
                    if budget["completion"] is not None:
                        _require(completion_count <= budget["completion"], f"{dataset}/{split}[{index}] exceeds completion budget")
                    _require(total <= budget["total"], f"{dataset}/{split}[{index}] exceeds total budget")
                maxima["prompt"] = max(maxima["prompt"], prompt_count)
                maxima["completion"] = max(maxima["completion"], completion_count)
                maxima["total"] = max(maxima["total"], total)
                total_checked += 1
        details[dataset] = {"budgets": TOKEN_BUDGETS[dataset], "max_observed": maxima, "split_counts": split_counts}
    return {"checked_rows": total_checked, "datasets": details}


def _audit_payload(name: str, *, checked_rows: int, details: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "audit": name, "status": "passed", "checked_rows": checked_rows, "details": dict(details)}


def _write_fsynced(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _fsync_tree(root: Path) -> None:
    for path in sorted((item for item in root.rglob("*") if item.is_file())):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directories = sorted((item for item in root.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True)
    for path in (*directories, root):
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish without the overwrite semantics of ``os.replace``."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(renameat2 is not None, "atomic renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise BaseReleaseError(f"immutable destination already exists: {destination}")
    raise BaseReleaseError(
        f"atomic no-overwrite publication failed: {destination}: {os.strerror(error)}"
    )


def _load_tokenizer(model_dir: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise BaseReleaseError("transformers is required to load the chk0 tokenizer") from exc
    try:
        return AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
    except Exception as exc:
        raise BaseReleaseError(f"unable to load local chk0 tokenizer: {model_dir}") from exc


def _remove_stale_staging(path: Path) -> None:
    _require(path.is_dir() and not path.is_symlink(), f"unsafe stale staging: {path}")
    _tree_has_no_symlinks(path, label="stale base-release staging")
    for candidate in path.rglob("*"):
        candidate.chmod(0o700 if candidate.is_dir() else 0o600)
    path.chmod(0o700)
    shutil.rmtree(path)


def _recover_existing_release(
    destination: Path,
    *,
    repo_root: Path,
    base_release_id: str,
    source_handoff_sha256: str,
    source_release_sha256: str,
    builder_code_sha256: str,
    runtime_mode: str,
) -> Path:
    _require(
        runtime_mode == "auto_tokenizer_local_bundle",
        "test-injected releases are non-releasable and cannot be recovered",
    )
    _tree_has_no_symlinks(destination, label="existing base release")
    _tree_is_immutable(destination)
    manifest = _load_json(
        destination / "base_release_manifest.json", label="existing base manifest"
    )
    provenance = manifest.get("provenance")
    source = manifest.get("source_release")
    _require(
        manifest.get("release_id") == base_release_id
        and isinstance(provenance, dict)
        and isinstance(source, dict),
        "existing base release identity is invalid",
    )
    _require(
        source.get("handoff_sha256") == source_handoff_sha256
        and provenance.get("source_handoff_sha256") == source_handoff_sha256
        and provenance.get("source_release_artifact_sha256")
        == source_release_sha256
        and provenance.get("code_sha256") == builder_code_sha256,
        "existing base release was built from different immutable inputs",
    )
    try:
        from jobs.retrain_v2.dag import validate_base_release

        validate_base_release(
            repo_root=repo_root,
            base_release_id=base_release_id,
        )
    except Exception as exc:
        raise BaseReleaseError(
            f"existing base release failed idempotent revalidation: {exc}"
        ) from exc
    return destination


def _validate_canonical_record(
    source_root: Path,
    record: Any,
    *,
    label: str,
    expected_path: str,
    expected_keys: set[str],
) -> Path:
    _require(isinstance(record, dict), f"{label} record is missing")
    _require(set(record) == expected_keys, f"{label} record keys are not canonical")
    _require(record.get("path") == expected_path, f"{label} path is not canonical")
    path = _resolve_inside(source_root, record["path"], label=label)
    _require(
        sha256_file(path)
        == _required_sha(record.get("sha256"), label=f"{label}.sha256"),
        f"{label} SHA mismatch",
    )
    if "rows" in expected_keys:
        rows = record.get("rows")
        _require(
            isinstance(rows, int) and not isinstance(rows, bool) and rows >= 0,
            f"{label}.rows is invalid",
        )
        _require(
            len(_load_jsonl(path, label=label)) == rows,
            f"{label} row count mismatch",
        )
    if "sample_ids_sha256" in expected_keys:
        _required_sha(
            record.get("sample_ids_sha256"),
            label=f"{label}.sample_ids_sha256",
        )
    return path


def _replay_canonical_source(
    *, repo_root: Path, source_root: Path, handoff: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Replay the copied preparation and generation source graph.

    The canonical publisher deliberately copies the complete source graph into
    the immutable release.  Replaying those copied handoffs here prevents the
    base builder from trusting the publisher's derived SFT files or quality
    booleans as an authority.
    """

    source = handoff.get("source_generation")
    source_keys = {
        "schema_version",
        "generation_handoff",
        "prepare_handoff",
        "selection_manifest",
        "cache_manifest",
        "generation_bundle_manifest",
        "preparation_bundle_manifest",
        "generation_provenance_sha256",
        "generation_code_payload_sha256",
        "preparation_binding_sha256",
        "selected_sample_ids_sha256",
    }
    _require(isinstance(source, dict), "handoff.source_generation is missing")
    _require(set(source) == source_keys, "handoff.source_generation keys drifted")
    _require(
        source.get("schema_version")
        == canonical_workflow.SOURCE_CONTRACT_SCHEMA_VERSION,
        "handoff.source_generation schema mismatch",
    )
    simple_record_keys = {"path", "sha256"}
    generation_handoff_path = _validate_canonical_record(
        source_root,
        source.get("generation_handoff"),
        label="source generation handoff",
        expected_path="source/generation/generation_handoff.json",
        expected_keys=simple_record_keys,
    )
    prepare_handoff_path = _validate_canonical_record(
        source_root,
        source.get("prepare_handoff"),
        label="source preparation handoff",
        expected_path="source/preparation/prepare_handoff.json",
        expected_keys=simple_record_keys,
    )
    _validate_canonical_record(
        source_root,
        source.get("selection_manifest"),
        label="source selection manifest",
        expected_path="source/generation/selection_manifest.json",
        expected_keys=simple_record_keys,
    )
    _validate_canonical_record(
        source_root,
        source.get("cache_manifest"),
        label="source cache manifest",
        expected_path="source/generation/cache/cache_manifest.json",
        expected_keys=simple_record_keys,
    )
    for field in (
        "generation_provenance_sha256",
        "generation_code_payload_sha256",
        "preparation_binding_sha256",
        "selected_sample_ids_sha256",
    ):
        _required_sha(source.get(field), label=f"source_generation.{field}")

    try:
        generation_bundle = canonical_workflow._bundle_manifest_record(
            source_root,
            source.get("generation_bundle_manifest"),
            schema_version=canonical_workflow.GENERATION_BUNDLE_SCHEMA_VERSION,
            label="generation bundle manifest",
        )
        preparation_bundle = canonical_workflow._bundle_manifest_record(
            source_root,
            source.get("preparation_bundle_manifest"),
            schema_version=canonical_workflow.PREPARATION_BUNDLE_SCHEMA_VERSION,
            label="preparation bundle manifest",
        )
        canonical_workflow._verify_bundle_entries(
            source_root, generation_bundle, label="generation bundle"
        )
        canonical_workflow._verify_bundle_entries(
            source_root, preparation_bundle, label="preparation bundle"
        )
        verified = canonical_workflow.verify_full_generation(
            repo_root=repo_root,
            generation_handoff=generation_handoff_path,
            prepare_handoff=prepare_handoff_path,
        )
        expected_generation_bundle, expected_preparation_bundle = (
            canonical_workflow._build_bundle_manifests(verified)
        )
    except Exception as exc:
        raise BaseReleaseError(
            f"copied preparation/generation replay failed: {exc}"
        ) from exc
    _require(
        generation_bundle == expected_generation_bundle,
        "generation source bundle does not replay exactly",
    )
    _require(
        preparation_bundle == expected_preparation_bundle,
        "preparation source bundle does not replay exactly",
    )
    _require(
        verified.generation_handoff_sha256
        == source["generation_handoff"]["sha256"],
        "source generation handoff SHA does not replay",
    )
    _require(
        verified.prepare_handoff_sha256 == source["prepare_handoff"]["sha256"],
        "source preparation handoff SHA does not replay",
    )
    _require(
        verified.generation_handoff["generation_provenance"]["payload_sha256"]
        == source["generation_provenance_sha256"]
        and verified.preparation_binding["binding_sha256"]
        == source["preparation_binding_sha256"]
        and verified.selection["selected_sample_ids_sha256"]
        == source["selected_sample_ids_sha256"],
        "source generation identity fields do not replay",
    )
    generation_code = verified.generation_handoff["generation_provenance"].get(
        "generation_code"
    )
    _require(
        isinstance(generation_code, dict)
        and set(generation_code) == {"schema_version", "files", "payload_sha256"},
        "source generation code bundle is missing or malformed",
    )
    generation_code_payload = {
        key: value for key, value in generation_code.items() if key != "payload_sha256"
    }
    _require(
        _required_sha(
            generation_code.get("payload_sha256"),
            label="source generation code payload_sha256",
        )
        == sha256_text(canonical_json(generation_code_payload)),
        "source generation code bundle hash mismatch",
    )
    _require(
        source["generation_code_payload_sha256"]
        == generation_code["payload_sha256"]
        == generation_bundle["generation_code_payload_sha256"],
        "source generation code payload binding mismatch",
    )
    source_audit = {
        "generation_handoff_sha256": verified.generation_handoff_sha256,
        "prepare_handoff_sha256": verified.prepare_handoff_sha256,
        "generation_bundle_payload_sha256": generation_bundle["payload_sha256"],
        "preparation_bundle_payload_sha256": preparation_bundle["payload_sha256"],
        "generation_code_payload_sha256": generation_code["payload_sha256"],
        "preparation_binding_sha256": verified.preparation_binding["binding_sha256"],
        "selected_sample_ids_sha256": verified.selection[
            "selected_sample_ids_sha256"
        ],
    }
    return verified, source_audit


def build_base_release(
    *,
    repo_root: str | Path,
    canonical_handoff: str | Path,
    base_release_id: str,
    test_tokenizer: Any | None = None,
    generated_at_utc: str | None = None,
    recover_stale_staging: bool = False,
) -> Path:
    """Validate, stage, fsync, seal, and atomically publish one base release."""

    _require(SAFE_ID.fullmatch(base_release_id) is not None and base_release_id not in {".", ".."}, "unsafe base_release_id")
    root = Path(repo_root).resolve()
    handoff_argument = Path(canonical_handoff)
    handoff_lexical = handoff_argument if handoff_argument.is_absolute() else root / handoff_argument
    _require_no_symlink_components(root, handoff_lexical, label="canonical handoff")
    handoff_path = handoff_lexical.resolve()
    try:
        handoff_path.relative_to(root)
    except ValueError as exc:
        raise BaseReleaseError("canonical handoff escapes repository") from exc
    _require(handoff_path.name == "handoff.json", "canonical handoff must be a real handoff.json")
    source_root = handoff_path.parent
    _tree_has_no_symlinks(source_root, label="source release")
    _tree_is_immutable(source_root)
    source_artifact_before = fingerprint_artifact_path(source_root)
    builder_code_bytes = Path(__file__).read_bytes()
    handoff = _load_json(handoff_path, label="canonical handoff")
    _require(handoff.get("schema_version") == HANDOFF_SCHEMA_VERSION, "input is not a canonical chk1 release handoff")
    _require(handoff.get("release_schema_version") == RELEASE_SCHEMA_VERSION, "input release schema is not canonical")
    _require(handoff.get("immutable") is True and handoff.get("quality_status") == "passed", "canonical source is not immutable and passed")
    _require(isinstance(handoff.get("release_id"), str) and handoff["release_id"], "source release_id missing")
    verified_generation, source_replay_audit = _replay_canonical_source(
        repo_root=root,
        source_root=source_root,
        handoff=handoff,
    )

    expected_sft, expected_manifests, expected_exclusions = (
        canonical_workflow._release_terminals(verified_generation)
    )
    source_sft = _load_bound_rows(source_root, handoff, "split_files")
    source_manifests = _load_bound_rows(source_root, handoff, "manifest_files")
    for split in SPLITS:
        _require(
            source_sft[split]
            == [dict(row) for row in expected_sft[split]],
            f"canonical SFT {split} differs from replayed admission",
        )
        _require(
            source_manifests[split]
            == [dict(row) for row in expected_manifests[split]],
            f"canonical manifests {split} differ from replayed admission",
        )
    _require(handoff.get("split_counts") == {split: len(source_sft[split]) for split in SPLITS}, "handoff split_counts mismatch")
    exclusion_rows = _load_jsonl(
        source_root / "audit/exclusions.jsonl", label="source exclusions"
    )
    _require(
        exclusion_rows == [dict(row) for row in expected_exclusions],
        "canonical exclusions differ from replayed admission",
    )
    quality_record = handoff.get("quality_report")
    _require(isinstance(quality_record, dict), "handoff quality_report missing")
    quality_path = _resolve_inside(source_root, quality_record.get("path"), label="quality_report")
    _require(sha256_file(quality_path) == _required_sha(quality_record.get("sha256"), label="quality_report.sha256"), "quality report SHA mismatch")
    quality = _load_json(quality_path, label="quality report")
    _require(quality.get("schema_version") == QUALITY_SCHEMA_VERSION, "quality report schema mismatch")
    _require(set(quality) == QUALITY_KEYS, "quality report keys are not canonical")
    _require(quality.get("status") == "passed", "quality report did not pass")
    gates = quality.get("gates")
    _require(isinstance(gates, dict) and set(gates) == QUALITY_GATE_KEYS and all(value is True for value in gates.values()), "quality gates are not canonical and all true")
    quality_provenance = quality.get("manifest_provenance")
    _require(
        isinstance(quality_provenance, dict)
        and set(quality_provenance)
        in {frozenset(QUALITY_PROVENANCE_KEYS), frozenset(LEGACY_QUALITY_PROVENANCE_KEYS)},
        "quality manifest_provenance schema is not canonical",
    )
    quality_invariants = quality.get("invariants")
    _require(
        isinstance(quality_invariants, dict)
        and set(quality_invariants) == QUALITY_INVARIANT_KEYS
        and all(
            quality_invariants[key] is True
            for key in QUALITY_INVARIANT_KEYS
            if not key.endswith("_count")
        )
        and all(
            quality_invariants[key] == 0
            for key in QUALITY_INVARIANT_KEYS
            if key.endswith("_count")
        ),
        "quality invariants are not canonical and passing",
    )
    for field in ("population_binding_sha256", "split_binding_sha256", "topic_binding_sha256"):
        _require(quality.get(field) == handoff.get(field), f"quality/handoff {field} mismatch")
    _require(quality.get("split_counts") in (None, handoff["split_counts"]), "quality split_counts mismatch")
    _require(quality.get("population_count") in (None, handoff["population_count"]), "quality population_count mismatch")


    validated: dict[str, list[tuple[dict[str, str], dict[str, Any], dict[str, Any]]]] = {}
    prepared_by_digest: dict[str, dict[str, Any]] = {}
    prepared_by_sample_id: dict[str, dict[str, Any]] = {}
    for index, raw_prepared in enumerate(verified_generation.population):
        prepared = dict(raw_prepared)
        prepared_digest = sha256_text(canonical_json(prepared))
        sample_id = _required_text(
            prepared, "sample_id", label=f"prepared_population[{index}]"
        )
        _require(
            prepared_digest not in prepared_by_digest
            and sample_id not in prepared_by_sample_id,
            "replayed prepared population contains duplicate identities",
        )
        prepared_by_digest[prepared_digest] = prepared
        prepared_by_sample_id[sample_id] = prepared
    sample_owner: dict[str, str] = {}
    meeting_owner: dict[str, str] = {}
    provenance_sets = {key: set() for key in ("teacher_model_sha256", "tokenizer_sha256", "style_guide_sha256", "generation_provenance_sha256", "generator_tokenizer_sha256", "prompt_template_sha256")}
    legacy_critic_shas: set[str] = set()
    total_rows = 0
    for split in SPLITS:
        _require(len(source_sft[split]) == len(source_manifests[split]), f"{split} SFT/manifest count mismatch")
        pairs = []
        for index, (sft, manifest) in enumerate(zip(source_sft[split], source_manifests[split], strict=True)):
            pair = _validate_row_pair(sft, manifest, split=split, index=index)
            clean, checked_manifest, safe_fact = pair
            sample_id = checked_manifest["sample_id"]
            prepared_digest = checked_manifest["generation"]["prepared_row_sha256"]
            prepared = prepared_by_digest.get(prepared_digest)
            _require(
                prepared is not None
                and prepared_by_sample_id.get(sample_id) is prepared,
                f"{split}[{index}] cannot locate its exact prepared source row",
            )
            _validate_prepared_projection_replay(
                clean=clean,
                manifest=checked_manifest,
                safe_fact=safe_fact,
                prepared=prepared,
                label=f"{split}[{index}]",
            )
            _require(sample_id not in sample_owner, f"duplicate sample_id: {sample_id}")
            sample_owner[sample_id] = split
            meeting = checked_manifest["meeting_date"]
            prior = meeting_owner.setdefault(meeting, split)
            _require(prior == split, f"meeting split leakage: {meeting}")
            for key in ("teacher_model_sha256", "tokenizer_sha256", "style_guide_sha256"):
                provenance_sets[key].add(checked_manifest[key])
            if "critic_model_sha256" in checked_manifest:
                legacy_critic_shas.add(checked_manifest["critic_model_sha256"])
            for key in ("generation_provenance_sha256", "generator_tokenizer_sha256", "prompt_template_sha256"):
                provenance_sets[key].add(checked_manifest["generation"][key])
            pairs.append(pair)
            total_rows += 1
        validated[split] = pairs
    for key, values in provenance_sets.items():
        _require(len(values) == 1, f"source manifests mix {key}")
    _require(
        not legacy_critic_shas or len(legacy_critic_shas) == 1,
        "source manifests mix critic_model_sha256",
    )

    excluded_counts = {split: 0 for split in SPLITS}
    excluded_ids: set[str] = set()
    for index, exclusion in enumerate(exclusion_rows):
        label = f"exclusions[{index}]"
        sample_id = _required_text(exclusion, "sample_id", label=label)
        split = _required_text(exclusion, "split", label=label)
        _require(split in SPLITS, f"{label}.split is invalid")
        _required_text(exclusion, "atomic_topic", label=label)
        _required_text(exclusion, "reason_code", label=label)
        meeting = _parse_meeting(
            exclusion.get("meeting_date"), label=f"{label}.meeting_date"
        ).isoformat()
        _require(
            sample_id not in sample_owner and sample_id not in excluded_ids,
            f"{label} sample_id is duplicated or accepted",
        )
        excluded_ids.add(sample_id)
        prior_split = meeting_owner.setdefault(meeting, split)
        _require(
            prior_split == split,
            f"excluded meeting crosses split boundary: {meeting}",
        )
        _require(
            "prepared_row_sha256" in exclusion,
            f"{label} lacks its prepared source binding",
        )
        prepared_digest = _required_sha(
            exclusion["prepared_row_sha256"],
            label=f"{label}.prepared_row_sha256",
        )
        prepared = prepared_by_digest.get(prepared_digest)
        _require(
            prepared is not None
            and prepared_by_sample_id.get(sample_id) is prepared
            and all(
                exclusion.get(field) == prepared.get(field)
                for field in ("sample_id", "split", "meeting_date", "atomic_topic")
            ),
            f"{label} does not bind its exact prepared source row",
        )
        excluded_counts[split] += 1
    accepted_counts = {split: len(source_sft[split]) for split in SPLITS}
    candidate_counts = {
        split: accepted_counts[split] + excluded_counts[split] for split in SPLITS
    }
    quality_candidate_counts = quality.get("candidate_counts")
    _require(
        isinstance(quality_candidate_counts, dict)
        and quality_candidate_counts == candidate_counts,
        "quality candidate_counts do not equal accepted + excluded",
    )
    quality_excluded_counts = quality.get("excluded_counts")
    _require(
        isinstance(quality_excluded_counts, dict)
        and set(quality_excluded_counts) <= set(SPLITS)
        and {
            split: int(quality_excluded_counts.get(split, 0)) for split in SPLITS
        }
        == excluded_counts,
        "quality excluded_counts do not match exclusions.jsonl",
    )
    population_count = sum(candidate_counts.values())
    _require(
        population_count == len(prepared_by_digest)
        and set(sample_owner) | excluded_ids == set(prepared_by_sample_id),
        "accepted/excluded terminals do not cover the replayed prepared population",
    )
    _require(
        handoff.get("population_count") == population_count
        and quality.get("population_count") == population_count,
        "population_count does not equal accepted + excluded",
    )
    handoff_bindings = {
        "teacher_model_sha256": "teacher_model_sha256",
        "tokenizer_sha256": "tokenizer_sha256",
        "style_guide_sha256": "style_guide_sha256",
        "generation_provenance_sha256": "generation_provenance_sha256",
        "generator_tokenizer_sha256": "generator_tokenizer_sha256",
        "prompt_template_sha256": "prompt_template_sha256",
    }
    for source_key, handoff_key in handoff_bindings.items():
        _require(next(iter(provenance_sets[source_key])) == handoff.get(handoff_key), f"handoff {handoff_key} mismatch")
    if legacy_critic_shas:
        _require(
            next(iter(legacy_critic_shas)) == handoff.get("critic_model_sha256"),
            "handoff critic_model_sha256 mismatch",
        )


    outputs: dict[str, dict[str, list[dict[str, Any]]]] = {
        "analysis_sft": {split: [] for split in SPLITS},
        "analysis_grpo": {split: [] for split in SPLITS},
    }
    grpo_sources: dict[
        str, list[tuple[dict[str, str], dict[str, Any], dict[str, Any]]]
    ] = {split: [] for split in SPLITS}
    role_counts = {"sft_only": 0, "grpo_only": 0, "shared": 0}
    role_counts_by_topic: dict[str, dict[str, int]] = {}
    role_counts_by_year: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        for clean, manifest, fact in validated[split]:
            role = stable_train_role(manifest["sample_id"]) if split == "train" else "both_full"
            if split == "train":
                role_counts[role] += 1
                topic = str(manifest["atomic_topic"])
                year = str(manifest["meeting_date"])[:4]
                role_counts_by_topic.setdefault(topic, {}).setdefault(role, 0)
                role_counts_by_topic[topic][role] += 1
                role_counts_by_year.setdefault(year, {}).setdefault(role, 0)
                role_counts_by_year[year][role] += 1
            if split != "train" or role in {"sft_only", "shared"}:
                outputs["analysis_sft"][split].append(clean)
            if split != "train" or role in {"grpo_only", "shared"}:
                grpo_sources[split].append((clean, manifest, fact))

    for split, rows in outputs["analysis_sft"].items():
        _require(rows, f"analysis_sft/{split} would be empty")
    for split, rows in grpo_sources.items():
        _require(rows, f"analysis_grpo/{split} would be empty")

    # Independent leakage and target consistency checks over the exact output.
    leakage_checked = 0
    for split in SPLITS:
        for clean, manifest, _fact in validated[split]:
            for field in ("prompt", "provided_data"):
                candidate = clean[field]
                _require(TARGET_MARKER.search(candidate) is None, f"{split}/{manifest['sample_id']} {field} has target marker")
                for forbidden in clean["response"].split("\n</think>\n"):
                    _require(len(forbidden.strip()) < 32 or forbidden.strip() not in candidate, f"{split}/{manifest['sample_id']} target copied into input")
            fact_keys = {
                re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
                for key in _walk_keys(json.loads(clean["provided_data"]))
            }
            dangerous_fact_keys = {
                key
                for key in fact_keys
                if key in TARGET_KEYS or key.replace("_", "") in TARGET_KEYS_COLLAPSED
            }
            _require(not dangerous_fact_keys, f"{split}/{manifest['sample_id']} fact card target keys: {sorted(dangerous_fact_keys)}")
            leakage_checked += 1
    destination_parent = root / "dataset/processed/retrain_v2"
    _require_no_symlink_components(root, destination_parent, label="base release root")
    destination_parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_components(root, destination_parent, label="base release root")
    _require(destination_parent.resolve() == destination_parent, "base release root is not canonical")
    _require(
        source_root != destination_parent
        and destination_parent not in source_root.parents
        and source_root not in destination_parent.parents,
        "source release overlaps output root",
    )
    destination = destination_parent / base_release_id
    runtime_mode = (
        "test_injected_non_releasable"
        if test_tokenizer is not None
        else "auto_tokenizer_local_bundle"
    )
    stale_staging = sorted(destination_parent.glob(f".{base_release_id}.build-*"))
    if recover_stale_staging:
        for stale in stale_staging:
            _remove_stale_staging(stale)
        stale_staging = []
    if destination.exists() or destination.is_symlink():
        _require(not destination.is_symlink(), f"immutable destination is a symlink: {destination}")
        return _recover_existing_release(
            destination,
            repo_root=root,
            base_release_id=base_release_id,
            source_handoff_sha256=sha256_file(handoff_path),
            source_release_sha256=source_artifact_before["sha256"],
            builder_code_sha256=_sha_bytes(builder_code_bytes),
            runtime_mode=runtime_mode,
        )
    _require(
        not stale_staging,
        "stale staging exists; inspect it, then rerun with recover_stale_staging=True",
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{base_release_id}.build-", dir=destination_parent))
    try:
        chk0_model = root / "models/DeepSeek-R1-Distill-Llama-8B"
        _require_no_symlink_components(root, chk0_model, label="chk0 tokenizer bundle")
        try:
            live_student_tokenizer = fingerprint_tokenizer_payload(chk0_model)
        except (FileNotFoundError, ValueError) as exc:
            raise BaseReleaseError(f"unable to fingerprint the live chk0 tokenizer: {exc}") from exc
        _require(
            live_student_tokenizer["sha256"] == handoff["tokenizer_sha256"],
            "canonical student tokenizer payload does not match live chk0",
        )
        generation_provenance = verified_generation.generation_handoff.get(
            "generation_provenance"
        )
        _require(
            isinstance(generation_provenance, Mapping),
            "replayed generation provenance is missing",
        )
        teacher_model_provenance = generation_provenance.get("teacher_model")
        legacy_local_models = "critic_model_sha256" in handoff
        if not legacy_local_models:
            _require(
                isinstance(teacher_model_provenance, Mapping)
                and teacher_model_provenance.get("sha256")
                == handoff["teacher_model_sha256"],
                "canonical teacher model does not match replayed DeepSeek provenance",
            )
        live_critic_model: dict[str, Any] | None = None
        live_teacher_model: dict[str, Any] | None = None
        if legacy_local_models:
            teacher_model = root / "models/Qwen3.5-9B"
            _require_no_symlink_components(
                root, teacher_model, label="legacy local teacher model"
            )
            try:
                live_critic_model = fingerprint_model_payload(chk0_model)
                live_teacher_model = fingerprint_model_payload(teacher_model)
            except (FileNotFoundError, ValueError) as exc:
                raise BaseReleaseError(
                    f"unable to fingerprint legacy local generation models: {exc}"
                ) from exc
            _require(
                live_critic_model["sha256"] == handoff["critic_model_sha256"],
                "canonical critic model payload does not match live chk0",
            )
            _require(
                live_teacher_model["sha256"] == handoff["teacher_model_sha256"],
                "canonical teacher model payload does not match live Qwen",
            )
        try:
            tokenizer_source_before = snapshot_tokenizer_bundle(chk0_model)
            tokenizer_bundle_dir = staging / "provenance/tokenizer_bundle"
            materialize_tokenizer_bundle(
                chk0_model,
                tokenizer_bundle_dir,
                expected_snapshot=tokenizer_source_before,
            )
        except TokenizerBundleError as exc:
            raise BaseReleaseError(f"invalid chk0 tokenizer bundle: {exc}") from exc
        _seal_tree(tokenizer_bundle_dir)
        tokenizer_bundle_before = snapshot_tokenizer_bundle(tokenizer_bundle_dir)
        tokenizer_object = (
            test_tokenizer
            if test_tokenizer is not None
            else _load_tokenizer(tokenizer_bundle_dir)
        )
        try:
            sft_token_auditor = build_training_sft_token_auditor(tokenizer_object)
        except PromptProjectionError as exc:
            raise BaseReleaseError(
                f"unable to construct the SFT token auditor: {exc}"
            ) from exc
        recomputed_sft_token_audits: dict[
            tuple[str, str], dict[str, Any]
        ] = {}
        sft_token_attestations: list[dict[str, Any]] = []
        for split in SPLITS:
            for clean, manifest, _fact in validated[split]:
                label = f"{split}/{manifest['sample_id']}"
                try:
                    observed_audit = dict(
                        sft_token_auditor(clean["prompt"], clean["response"])
                    )
                except PromptProjectionError as exc:
                    raise BaseReleaseError(
                        f"{label} shared SFT token audit failed: {exc}"
                    ) from exc
                checked_audit = _validate_training_sft_token_budget_claim(
                    observed_audit,
                    label=f"{label}.recomputed_sft_token_budget",
                )
                claimed_audit = _validate_sft_token_budget_claim(
                    manifest["generation"].get("sft_token_budget"),
                    label=f"{label}.manifest_sft_token_budget",
                    require_passed=False,
                )
                _require(
                    all(
                        checked_audit[key] == claimed_audit[key]
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    ),
                    f"{label} acquisition/training token counts do not match the live chk0 tokenizer",
                )
                student_projection = manifest["generation"][
                    "model_input_projection"
                ]["student_projection"]
                _require(
                    student_projection["prompt_tokens"]
                    == checked_audit["prompt_tokens"],
                    f"{label} projection/token-budget prompt counts differ",
                )
                token_key = (clean["prompt"], clean["response"])
                _require(
                    token_key not in recomputed_sft_token_audits,
                    f"{label} duplicates an accepted prompt/response token binding",
                )
                recomputed_sft_token_audits[token_key] = checked_audit
                sft_token_attestations.append(
                    {
                        "split": split,
                        "sample_id_sha256": sha256_text(manifest["sample_id"]),
                        "token_audit_sha256": sha256_text(
                            canonical_json(checked_audit)
                        ),
                    }
                )
        try:
            grpo_chat_token_counter = build_chat_token_counter(tokenizer_object)
        except PromptProjectionError as exc:
            raise BaseReleaseError(
                f"unable to construct the GRPO chat token counter: {exc}"
            ) from exc
        grpo_projection_attestations: list[dict[str, Any]] = []
        for split in SPLITS:
            for _clean, manifest, fact in grpo_sources[split]:
                try:
                    safe_fact = safe_grpo_fact_card(fact)
                    projection = project_prompt_to_budget(
                        projection_name="analysis_grpo_base_release",
                        fact_card=safe_fact,
                        evidence_lineage=manifest["evidence_lineage"],
                        system_prompt=GRPO_SYSTEM,
                        max_prompt_tokens=GRPO_PROMPT_TOKEN_LIMIT,
                        prompt_renderer=render_grpo_user_prompt,
                        chat_token_counter=grpo_chat_token_counter,
                    )
                except PromptProjectionError as exc:
                    raise BaseReleaseError(
                        f"{split}/{manifest['sample_id']} GRPO prompt projection "
                        f"exceeds prompt budget or is invalid: {exc}"
                    ) from exc
                _require(
                    projection.prompt.count(projection.provided_data) == 1,
                    f"{split}/{manifest['sample_id']} GRPO prompt does not bind "
                    "provided_data exactly once",
                )
                _require(
                    canonical_json(dict(projection.fact_card))
                    == projection.provided_data,
                    f"{split}/{manifest['sample_id']} GRPO fact-card binding mismatch",
                )
                try:
                    validate_judge_evidence(projection.provided_data)
                except ValueError as exc:
                    raise BaseReleaseError(
                        f"{split}/{manifest['sample_id']} unsafe GRPO judge evidence: "
                        f"{exc}"
                    ) from exc
                outputs["analysis_grpo"][split].append(
                    {
                        "prompt": projection.prompt,
                        "provided_data": projection.provided_data,
                        "sample_id": manifest["sample_id"],
                        "meeting_date": manifest["meeting_date"],
                    }
                )
                grpo_projection_attestations.append(
                    {
                        "split": split,
                        "sample_id_sha256": sha256_text(manifest["sample_id"]),
                        "attestation": dict(projection.attestation),
                    }
                )

        for split, rows in outputs["analysis_grpo"].items():
            for index, row in enumerate(rows):
                label = f"analysis_grpo/{split}[{index}]"
                _require(
                    set(row)
                    == {"meeting_date", "prompt", "provided_data", "sample_id"},
                    f"{label} keys are not canonical",
                )
                _require(
                    row["prompt"].count(row["provided_data"]) == 1,
                    f"{label} does not contain provided_data exactly once",
                )
                try:
                    parsed_evidence = json.loads(row["provided_data"])
                except json.JSONDecodeError as exc:
                    raise BaseReleaseError(
                        f"{label} provided_data is not valid JSON: {exc}"
                    ) from exc
                _require(
                    isinstance(parsed_evidence, dict),
                    f"{label} provided_data is not an object",
                )
                _require(
                    canonical_json(parsed_evidence) == row["provided_data"],
                    f"{label} provided_data is not canonical JSON",
                )
                fact_keys = {
                    re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
                    for key in _walk_keys(parsed_evidence)
                }
                dangerous_fact_keys = {
                    key
                    for key in fact_keys
                    if key in TARGET_KEYS
                    or key.replace("_", "") in TARGET_KEYS_COLLAPSED
                }
                _require(
                    not dangerous_fact_keys,
                    f"{label} fact card target keys: {sorted(dangerous_fact_keys)}",
                )
                try:
                    validate_judge_evidence(row["provided_data"])
                except ValueError as exc:
                    raise BaseReleaseError(
                        f"{label} unsafe judge evidence: {exc}"
                    ) from exc
        token_details = _audit_tokens(
            tokenizer_object,
            outputs,
            sft_token_audits=recomputed_sft_token_audits,
        )
        _require(
            snapshot_tokenizer_bundle(tokenizer_bundle_dir)
            == tokenizer_bundle_before,
            "consumed tokenizer bundle changed during token audit",
        )
        _require(
            snapshot_tokenizer_bundle(chk0_model) == tokenizer_source_before,
            "chk0 tokenizer bundle changed during token audit",
        )
        for dataset, splits in outputs.items():
            for split, rows in splits.items():
                _write_fsynced(staging / dataset / f"{split}.jsonl", _jsonl_bytes(rows))

        audits = {
            "schema": _audit_payload("schema", checked_rows=population_count, details={"source_handoff_schema": HANDOFF_SCHEMA_VERSION, "source_release_schema": RELEASE_SCHEMA_VERSION, "accepted_rows": total_rows, "excluded_rows": len(exclusion_rows), "sft_keys": ["prompt", "provided_data", "response"], "grpo_keys": ["meeting_date", "prompt", "provided_data", "sample_id"]}),
            "token_budget": _audit_payload("token_budget", checked_rows=token_details["checked_rows"], details={**token_details, "tokenizer_runtime_mode": runtime_mode, "tokenizer_bundle_artifact_sha256": tokenizer_bundle_before["artifact_sha256"], "tokenizer_bundle_payload_sha256": tokenizer_bundle_before["payload_sha256"], "tokenizer_bundle_files": tokenizer_bundle_before["files"], "accepted_sft_rows_recomputed": len(sft_token_attestations), "accepted_sft_token_attestations": sft_token_attestations, "shared_sft_token_auditor": "jobs.retrain_v2.sft_training_budget.build_training_sft_token_auditor", "projection_contract_sha256": projection_contract_sha256()}),
            "reference_leakage": _audit_payload("reference_leakage", checked_rows=leakage_checked, details={"target_markers_found": 0, "target_keys_found": 0, "judge_evidence_rejections": 0}),
            "point_in_time": _audit_payload("point_in_time", checked_rows=total_rows, details={"cutoff_bound_rows": total_rows, "evidence_lineage_bound_rows": total_rows, "prepared_projection_replayed_rows": total_rows, "meeting_cutoff_violations": 0, "source_graph_replay": source_replay_audit}),
            "split_integrity": _audit_payload("split_integrity", checked_rows=population_count, details={"accepted_counts": accepted_counts, "excluded_counts": excluded_counts, "candidate_counts": candidate_counts, "population_count": population_count, "train_assignment": {"algorithm": "int(sha256(sample_id),16) % 100", "semantics": "deterministic per-sample buckets; 70/20/10 are expected aggregate proportions, not exact quotas", "ranges": {"sft_only": "0-69", "grpo_only": "70-89", "shared": "90-99"}, "actual_counts": role_counts, "actual_counts_by_atomic_topic": dict(sorted(role_counts_by_topic.items())), "actual_counts_by_meeting_year": dict(sorted(role_counts_by_year.items()))}, "eval_test_policy": "full_in_both_datasets", "sample_overlap_between_source_splits": 0, "meeting_overlap_between_source_splits": 0}),
            "target_consistency": _audit_payload("target_consistency", checked_rows=total_rows, details={"sft_rows_byte_semantics_preserved": True, "source_hash_bindings_recomputed": total_rows, "grpo_target_fields": 0, "grpo_projection_contract_sha256": projection_contract_sha256(), "grpo_projection_count": len(grpo_projection_attestations), "grpo_projection_attestations": grpo_projection_attestations}),
            "encoding": _audit_payload("encoding", checked_rows=total_rows, details={"strict_utf8": True, "nul_count": 0, "replacement_character_count": 0, "non_finite_numbers": 0, "output_jsonl": "canonical-json-per-line"}),
            "teacher_grounding": _audit_payload("teacher_grounding", checked_rows=total_rows, details={"deepseek_teacher_provenance_bound": total_rows, "teacher_response_provenance_bound": total_rows, "teacher_output_mapping_bound": total_rows, "evidence_lineage_bound": total_rows, "automated_source_graph_replay": True}),
        }
        audit_records: dict[str, dict[str, str]] = {}
        for name in AUDITS:
            path = staging / "audits" / f"{name}.json"
            _write_fsynced(path, _json_bytes(audits[name]))
            audit_records[name] = {"status": "passed", "path": f"audits/{name}.json", "sha256": sha256_file(path)}

        prompt_bundle = {
            "schema_version": 1,
            "source_prompt_template_sha256": handoff["prompt_template_sha256"],
            "training_system_prompts": {"analysis_sft": SFT_SYSTEM, "analysis_grpo": GRPO_SYSTEM},
            "training_system_prompt_sha256": {
                "analysis_sft": sha256_text(SFT_SYSTEM),
                "analysis_grpo": sha256_text(GRPO_SYSTEM),
            },
            "stage_config_paths": {
                "analysis_sft": "configs/retrain_v2/chk1_analysis_sft.yaml",
                "analysis_grpo": "configs/retrain_v2/chk2_analysis_grpo.yaml",
            },
            "chat_rendering_contract": {
                "messages": ["system", "user"],
                "add_generation_prompt": True,
                "grpo_truncation": False,
            },
            "grpo_projection_contract_sha256": projection_contract_sha256(),
            "grpo_projection_max_prompt_tokens": GRPO_PROMPT_TOKEN_LIMIT,
        }
        generation_config = {
            "schema_version": 1,
            "source_release_id": handoff["release_id"],
            "source_handoff_sha256": sha256_file(handoff_path),
            "train_role_assignment": {"hash": "sha256(sample_id)", "bucket": "int(digest,16)%100", "semantics": "deterministic per-sample buckets; aggregate 70/20/10 ratios are expectations, not exact quotas", "sft_only": [0, 69], "grpo_only": [70, 89], "shared": [90, 99], "actual_counts": role_counts, "actual_counts_by_atomic_topic": dict(sorted(role_counts_by_topic.items())), "actual_counts_by_meeting_year": dict(sorted(role_counts_by_year.items()))},
            "evaluation_policy": "eval and test copied in full to both datasets",
            "grpo_evidence_policy": "safe model fact card projected by whole evidence objects with dependency closure; no string truncation",
            "grpo_projection_contract_sha256": projection_contract_sha256(),
            "grpo_projection_max_prompt_tokens": GRPO_PROMPT_TOKEN_LIMIT,
            "source_generation_code_payload_sha256": source_replay_audit[
                "generation_code_payload_sha256"
            ],
        }
        provenance_dir = staging / "provenance"
        artifacts_payload = {
            "code": builder_code_bytes,
            "prompt": _json_bytes(prompt_bundle),
            "generation_config": _json_bytes(generation_config),
        }
        artifact_names = {"code": "build_base_release.py", "prompt": "prompt_contract.json", "generation_config": "generation_config.json"}
        artifact_records: dict[str, dict[str, str]] = {}
        for name, payload in artifacts_payload.items():
            path = provenance_dir / artifact_names[name]
            _write_fsynced(path, payload)
            artifact_records[name] = {"path": f"provenance/{artifact_names[name]}", "sha256": sha256_file(path)}
        tokenizer_bundle_manifest_path = provenance_dir / "tokenizer_bundle_manifest.json"
        _write_fsynced(
            tokenizer_bundle_manifest_path,
            _json_bytes(tokenizer_bundle_before),
        )
        tokenizer_bundle_manifest_record = {
            "path": "provenance/tokenizer_bundle_manifest.json",
            "sha256": sha256_file(tokenizer_bundle_manifest_path),
        }
        artifact_records["tokenizer"] = {
            "path": "provenance/tokenizer_bundle",
            "sha256": fingerprint_artifact_path(tokenizer_bundle_dir)["sha256"],
        }

        datasets: dict[str, Any] = {}
        for dataset in ("analysis_sft", "analysis_grpo"):
            dataset_dir = staging / dataset
            datasets[dataset] = {
                "path": dataset,
                "artifact_sha256": fingerprint_artifact_path(dataset_dir)["sha256"],
                "split_files": {
                    "train": {"path": f"{dataset}/train.jsonl", "sha256": sha256_file(dataset_dir / "train.jsonl")},
                    "validation": {"path": f"{dataset}/eval.jsonl", "sha256": sha256_file(dataset_dir / "eval.jsonl")},
                    "test": {"path": f"{dataset}/test.jsonl", "sha256": sha256_file(dataset_dir / "test.jsonl")},
                },
            }
        provenance = {
            "teacher_model": (
                "local Qwen3.5-9B"
                if legacy_local_models
                else str(teacher_model_provenance.get("model"))
            ),
            "teacher_model_version": handoff["teacher_model_sha256"],
            "teacher_provider": (
                "local"
                if legacy_local_models
                else str(teacher_model_provenance.get("provider"))
            ),
            "teacher_revision": (
                "local-payload"
                if legacy_local_models
                else str(teacher_model_provenance.get("revision"))
            ),
            "tokenizer_sha256": artifact_records["tokenizer"]["sha256"],
            "code_sha256": artifact_records["code"]["sha256"],
            "prompt_sha256": artifact_records["prompt"]["sha256"],
            "generation_config_sha256": artifact_records["generation_config"]["sha256"],
            "source_handoff_sha256": sha256_file(handoff_path),
            "source_release_id": handoff["release_id"],
            "source_release_artifact_sha256": source_artifact_before["sha256"],
            "source_generation_provenance_sha256": handoff["generation_provenance_sha256"],
            "source_generation_code": verified_generation.generation_handoff[
                "generation_provenance"
            ]["generation_code"],
            "source_generation_code_payload_sha256": source_replay_audit[
                "generation_code_payload_sha256"
            ],
            "source_graph_replay": source_replay_audit,
            "source_generator_tokenizer_payload_sha256": handoff["generator_tokenizer_sha256"],
            "source_student_tokenizer_payload_sha256": handoff["tokenizer_sha256"],
            "live_generation_payloads": {"teacher_model_sha256": handoff["teacher_model_sha256"], "teacher_model_binding": "replayed_remote_generation_provenance", "student_tokenizer_sha256": live_student_tokenizer["sha256"]},
            "tokenizer_runtime": {"mode": runtime_mode, "loader_contract": dict(TOKENIZER_LOADER_CONTRACT), "bundle_artifact_sha256": artifact_records["tokenizer"]["sha256"], "bundle_payload_sha256": tokenizer_bundle_before["payload_sha256"]},
            "tokenizer_bundle_manifest": tokenizer_bundle_manifest_record,
            "tokenizer_semantics": {"training_tokenizer_bundle": "models/DeepSeek-R1-Distill-Llama-8B", "training_tokenizer_bundle_sha256": tokenizer_bundle_before["artifact_sha256"], "source_student_tokenizer_payload_sha256": handoff["tokenizer_sha256"], "source_generator_tokenizer_payload_sha256": handoff["generator_tokenizer_sha256"]},
            "artifacts": artifact_records,
        }
        if legacy_local_models:
            _require(
                live_teacher_model is not None and live_critic_model is not None,
                "legacy live model fingerprints are missing",
            )
            provenance.update(
                {
                    "critic_model": "chk0 DeepSeek-R1-Distill-Llama-8B",
                    "critic_model_version": handoff["critic_model_sha256"],
                    "live_generation_payloads": {
                        "teacher_model_sha256": live_teacher_model["sha256"],
                        "critic_model_sha256": live_critic_model["sha256"],
                        "student_tokenizer_sha256": live_student_tokenizer[
                            "sha256"
                        ],
                    },
                }
            )
        generated = generated_at_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        _parse_utc(generated, label="generated_at_utc")
        manifest = {"schema_version": 1, "release_type": "analysis_base", "release_id": base_release_id, "generated_at_utc": generated, "source_release": {"release_id": handoff["release_id"], "handoff_sha256": sha256_file(handoff_path)}, "datasets": datasets, "audits": audit_records, "provenance": provenance}
        _write_fsynced(staging / "base_release_manifest.json", _json_bytes(manifest))
        _tree_has_no_symlinks(staging, label="staged base release")
        _require(snapshot_tokenizer_bundle(tokenizer_bundle_dir) == tokenizer_bundle_before, "staged tokenizer bundle changed before publication")
        _require(snapshot_tokenizer_bundle(chk0_model) == tokenizer_source_before, "chk0 tokenizer bundle changed before publication")
        _require(fingerprint_artifact_path(source_root)["sha256"] == source_artifact_before["sha256"], "source release changed during build")
        _require(Path(__file__).read_bytes() == builder_code_bytes, "builder code changed during build")
        _seal_tree(staging)
        _fsync_tree(staging)
        _require(not destination.exists(), f"immutable destination already exists: {destination}")
        _rename_noreplace(staging, destination)
        parent_fd = os.open(destination_parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return destination
    except BaseException:
        if staging.exists():
            for path in staging.rglob("*"):
                if path.is_dir():
                    path.chmod(0o700)
                elif path.is_file():
                    path.chmod(0o600)
            staging.chmod(0o700)
            shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--canonical-handoff", required=True)
    parser.add_argument("--base-release-id", required=True)
    parser.add_argument("--generated-at-utc")
    parser.add_argument("--recover-stale-staging", action="store_true")
    args = parser.parse_args(argv)
    destination = build_base_release(
        repo_root=args.repo_root,
        canonical_handoff=args.canonical_handoff,
        base_release_id=args.base_release_id,
        generated_at_utc=args.generated_at_utc,
        recover_stale_staging=args.recover_stale_staging,
    )
    print(json.dumps({"status": "published", "path": str(destination)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
