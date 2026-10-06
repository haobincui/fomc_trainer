"""Publish the paper Model chk-2 synthetic Minutes SFT release.

The acquisition generator owns API calls and terminal quality classification.
This module is deliberately API-free: it joins every terminal record back to
the pinned 2,083-row source population, recomputes PASS eligibility from the
deterministic and input-fidelity validators, replays the exact student
tokenizer contract, and atomically publishes a PASS-only training release.

The official-Minutes validator is retained as a required diagnostic for every
PASS row.  Its score or verdict never affects PASS selection and is never fed
back to the rewrite teacher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs.generation import generate_paper_chk2_synthetic_rewrite as generator
from open_r1.trainer.sft_prompt_renderer import (
    render_sft_prompt,
    tokenize_sft_text,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT
    / "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_machine_ready_v2_20260829"
)
DEFAULT_ACQUISITION_ROOT = (
    REPO_ROOT
    / "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_synthetic_rewrite_v1_20260830"
)
DEFAULT_RELEASE_ROOT = (
    REPO_ROOT
    / "dataset/processed/retrain_v2/"
    "chk2_target_derived_synthetic_rewrite_flash_v1_20260830"
)
DEFAULT_TOKENIZER_PATH = REPO_ROOT / "models/DeepSeek-R1-Distill-Llama-8B"

SOURCE_SUMMARY_SHA256 = (
    "7c8e9626ac6aaea435f2ca0a3e40e52dc3f9fc1b46e70f5f2f86acd53bb76c71"
)
SOURCE_HANDOFF_SHA256 = (
    "dd0ed476709a4c2e74c80f84ec175ce606e868297d84ac1d46520951fdb6d554"
)
SOURCE_SAMPLE_ID_SHA256 = (
    "9703dc5964bdb21ca6609bcf1cda19df64227c61a3e7c78df3e0a4df0227186a"
)

SPLITS = ("train", "validation", "test")
SOURCE_SPLIT_COUNTS = {"train": 1686, "validation": 221, "test": 176}
RELEASE_SCHEMA_VERSION = "paper-chk2-synthetic-rewrite-release-v1"
TERMINAL_SCHEMA_VERSION = "paper-chk2-synthetic-rewrite-terminal-v1"
DATASET_ROLE = "paper_chk2_minutes_synthetic_rewrite_sft"
TRAINING_SCOPE = "paper-chk2-chk1-cp200-minutes-sft-v1"
BOUNDARY = "\n</think>\n"

_CONTROL_RE = re.compile(r"<think>|</think>|<answer>|</answer>", re.IGNORECASE)
_SECRET_VALUE_RE = re.compile(r"(?:Bearer\s+[A-Za-z0-9._-]{12,}|\bsk-[A-Za-z0-9_-]{12,})")
_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "accesstoken",
    "secret",
}
_CREDENTIAL_METADATA_KEYS = {"api_key_env", "credential_env", "api_key_source"}

TERMINAL_FIELDS = {
    "schema_version",
    "sample_id",
    "split",
    "source_index",
    "meeting_date",
    "terminal_status",
    "training_pass",
    "rejection_stage",
    "rejection_reasons",
    "source_analysis",
    "teacher_response_analysis",
    "rewritten_minutes",
    "student_prompt",
    "sft_response",
    "source_analysis_sha256",
    "official_minutes_sha256",
    "teacher_response_analysis_sha256",
    "rewritten_minutes_sha256",
    "prompt_sha256",
    "response_sha256",
    "generation",
    "validator_a",
    "reference_diagnostic",
    "lineage",
}
GENERATION_FIELDS = {
    "selected_attempt",
    "repair_used",
    "deterministic_validation",
    "provider",
}
DETERMINISTIC_FIELDS = {"machine_pass", "reasons", "diagnostics"}
VALIDATOR_A_FIELDS = {"complete", "machine_pass", "reasons", "result", "provider"}
REFERENCE_DIAGNOSTIC_FIELDS = {
    "complete",
    "diagnostic_pass",
    "overall_score",
    "warning",
    "reasons",
    "result",
    "provider",
}
REQUEST_PROJECTION_FIELDS = {
    "role",
    "allowed_fields",
    "wire_fields",
    "field_value_sha256",
    "canonical_payload_sha256",
    "official_target_policy",
    "repair_trigger_source",
    "reference_diagnostic_can_trigger_repair",
    "api_key_env",
    "plaintext_credential_persisted",
}


class PublicationError(ValueError):
    """Raised when an acquisition cannot be safely published."""


@dataclass(frozen=True)
class SourceRow:
    sample_id: str
    split: str
    source_index: int
    candidate: Mapping[str, Any]
    manifest: Mapping[str, Any]

    @property
    def source_analysis(self) -> str:
        return str(self.manifest["analysis"])

    @property
    def prompt(self) -> str:
        return str(self.candidate["prompt"])

    @property
    def meeting_date(self) -> str:
        return str(self.manifest["meeting_date"])

    @property
    def official_minutes_sha256(self) -> str:
        return str(self.manifest["official_minutes_sha256"])


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PublicationError(f"cannot hash file: {path}: {exc}") from exc
    return digest.hexdigest()


def _id_digest(values: Sequence[str]) -> str:
    return sha256_text("".join(f"{value}\n" for value in sorted(values)))


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PublicationError(message)


def _required_string(value: Any, *, label: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be a non-empty string")
    return str(value)


def _required_sha(value: Any, *, label: str) -> str:
    text = _required_string(value, label=label)
    _require(bool(re.fullmatch(r"[0-9a-f]{64}", text)), f"{label} must be lowercase SHA-256")
    return text


def _required_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicationError(f"invalid {label}: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PublicationError(f"missing {label}: {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        _require(bool(line.strip()), f"blank row in {label}: {path}:{line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PublicationError(
                f"invalid JSON in {label}: {path}:{line_number}: {exc}"
            ) from exc
        _require(isinstance(row, dict), f"non-object row in {label}: {path}:{line_number}")
        rows.append(row)
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(dict(row)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _resolve_member(root: Path, value: Any, *, label: str) -> Path:
    text = _required_string(value, label=label)
    candidate = Path(text)
    if candidate.is_absolute():
        path = candidate.resolve()
    else:
        local_candidate = (root / candidate).resolve()
        repository_candidate = (REPO_ROOT / candidate).resolve()
        existing = [
            item
            for item in dict.fromkeys((local_candidate, repository_candidate))
            if item.is_file()
        ]
        _require(
            len(existing) == 1,
            f"{label} must resolve uniquely inside its release: {text}",
        )
        path = existing[0]
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise PublicationError(f"{label} escapes its release root: {text}") from exc
    _require(path.is_file() and not path.is_symlink(), f"missing or unsafe {label}: {path}")
    return path


def _verify_descriptor(root: Path, descriptor: Any, *, label: str) -> tuple[Path, int | None]:
    record = _required_mapping(descriptor, label=label)
    path = _resolve_member(root, record.get("path"), label=f"{label}.path")
    expected_sha = _required_sha(record.get("sha256"), label=f"{label}.sha256")
    _require(sha256_file(path) == expected_sha, f"{label} SHA-256 mismatch")
    rows = record.get("rows")
    if rows is not None:
        _require(isinstance(rows, int) and rows >= 0, f"{label}.rows must be non-negative")
        physical = len(path.read_text(encoding="utf-8").splitlines())
        _require(physical == rows, f"{label} physical row count mismatch")
    return path, rows


def _contains_secret(value: Any, *, path: str = "root") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            normalized_key = key_text.lower().replace("-", "_")
            if normalized_key in _CREDENTIAL_METADATA_KEYS:
                if not isinstance(item, str) or not re.fullmatch(
                    r"[A-Z][A-Z0-9_]{2,127}", item
                ):
                    return f"{path}.{key_text}"
                continue
            if normalized_key in _SECRET_KEYS:
                return f"{path}.{key_text}"
            found = _contains_secret(item, path=f"{path}.{key_text}")
            if found:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            found = _contains_secret(item, path=f"{path}[{index}]")
            if found:
                return found
    elif isinstance(value, str) and _SECRET_VALUE_RE.search(value):
        return path
    return None


def _machine_pass(value: Any, *, label: str) -> bool:
    mapping = _required_mapping(value, label=label)
    result = mapping.get("machine_pass")
    _require(isinstance(result, bool), f"{label}.machine_pass must be boolean")
    return result


def _reference_complete(value: Any, *, label: str) -> bool:
    mapping = _required_mapping(value, label=label)
    complete = mapping.get("complete")
    _require(isinstance(complete, bool), f"{label}.complete must be boolean")
    _require(complete, f"{label} is incomplete")
    score = mapping.get("overall_score")
    _require(
        isinstance(score, (int, float)) and not isinstance(score, bool) and 0 <= score <= 100,
        f"{label}.overall_score must be between 0 and 100",
    )
    return True


def _deterministic_result(row: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    generation = _required_mapping(row.get("generation"), label=f"{label}.generation")
    return _required_mapping(
        generation.get("deterministic_validation"),
        label=f"{label}.generation.deterministic_validation",
    )


def _student_system_prompt(contract: Mapping[str, Any]) -> str:
    return _required_string(
        contract.get("student_system_prompt"),
        label="prompt_contract.student_system_prompt",
    )


def _validate_terminal_shape(row: Mapping[str, Any], *, label: str) -> None:
    _require(set(row) == TERMINAL_FIELDS, f"{label} top-level field set mismatch")
    generation = _required_mapping(row.get("generation"), label=f"{label}.generation")
    _require(
        set(generation) == GENERATION_FIELDS,
        f"{label}.generation field set mismatch",
    )
    deterministic = _required_mapping(
        generation.get("deterministic_validation"),
        label=f"{label}.generation.deterministic_validation",
    )
    _require(
        set(deterministic) == DETERMINISTIC_FIELDS,
        f"{label}.generation.deterministic_validation field set mismatch",
    )
    validator_a = _required_mapping(
        row.get("validator_a"), label=f"{label}.validator_a"
    )
    _require(
        set(validator_a) == VALIDATOR_A_FIELDS,
        f"{label}.validator_a field set mismatch",
    )
    reference = _required_mapping(
        row.get("reference_diagnostic"),
        label=f"{label}.reference_diagnostic",
    )
    _require(
        set(reference) == REFERENCE_DIAGNOSTIC_FIELDS,
        f"{label}.reference_diagnostic field set mismatch",
    )
    for reasons_label, reasons in (
        ("generation.deterministic_validation.reasons", deterministic.get("reasons")),
        ("validator_a.reasons", validator_a.get("reasons")),
        ("reference_diagnostic.reasons", reference.get("reasons")),
    ):
        _require(
            isinstance(reasons, list)
            and all(isinstance(item, str) and item for item in reasons),
            f"{label}.{reasons_label} must be a list of non-empty codes",
        )
    _required_mapping(
        deterministic.get("diagnostics"),
        label=f"{label}.generation.deterministic_validation.diagnostics",
    )
    status = row.get("terminal_status")
    if status == "GENERATION_QUALITY_REJECT":
        _require(
            validator_a.get("complete") is False
            and validator_a.get("result") is None
            and validator_a.get("provider") == {},
            f"{label}.validator_a must be explicitly not run",
        )
        _require(
            reference.get("complete") is False
            and reference.get("diagnostic_pass") is None
            and reference.get("overall_score") is None
            and reference.get("result") is None
            and reference.get("provider") == {},
            f"{label}.reference_diagnostic must be explicitly not run",
        )
    elif status == "INPUT_FIDELITY_REJECT":
        _required_mapping(
            generation.get("provider"), label=f"{label}.generation.provider"
        )
        _require(
            validator_a.get("complete") is True,
            f"{label}.validator_a must be complete for fidelity rejection",
        )
        _required_mapping(
            validator_a.get("result"), label=f"{label}.validator_a.result"
        )
        _required_mapping(
            validator_a.get("provider"), label=f"{label}.validator_a.provider"
        )
        _require(
            reference.get("complete") is False
            and reference.get("diagnostic_pass") is None
            and reference.get("overall_score") is None
            and reference.get("result") is None
            and reference.get("provider") == {},
            f"{label}.reference_diagnostic must be explicitly not run",
        )
    elif status == "PASS":
        for nested_label, nested in (
            ("generation.provider", generation.get("provider")),
            ("validator_a.result", validator_a.get("result")),
            ("validator_a.provider", validator_a.get("provider")),
            ("reference_diagnostic.result", reference.get("result")),
            ("reference_diagnostic.provider", reference.get("provider")),
        ):
            _required_mapping(nested, label=f"{label}.{nested_label}")
    repair_used = generation.get("repair_used")
    _require(
        isinstance(repair_used, bool), f"{label}.generation.repair_used must be boolean"
    )
    selected_attempt = generation.get("selected_attempt")
    _require(
        selected_attempt in {None, "primary", "repair"},
        f"{label}.generation.selected_attempt is invalid",
    )
    if selected_attempt is not None:
        _require(
            repair_used == (selected_attempt == "repair"),
            f"{label}.generation selected_attempt/repair_used mismatch",
        )
    _require(
        reference.get("warning") in {None, "REFERENCE_DIAGNOSTIC_WARNING"},
        f"{label}.reference_diagnostic.warning enum drift",
    )


def _validate_provider_projection(
    provider_value: Any,
    *,
    label: str,
    role: str,
    allowed_fields: Sequence[str],
    wire_fields: Sequence[str],
    official_target_policy: str,
    allowed_repair_triggers: set[str | None],
    require_normal_stop: bool = True,
) -> None:
    provider = _required_mapping(provider_value, label=label)
    _require(
        provider.get("returned_model") == "deepseek-v4-flash",
        f"{label} model drift",
    )
    _required_string(provider.get("system_fingerprint"), label=f"{label}.system_fingerprint")
    if require_normal_stop:
        _require(
            provider.get("finish_reason") == "stop", f"{label} did not stop normally"
        )
    else:
        _required_string(provider.get("finish_reason"), label=f"{label}.finish_reason")
    projection = _required_mapping(
        provider.get("request_projection"), label=f"{label}.request_projection"
    )
    _require(
        set(projection) == REQUEST_PROJECTION_FIELDS,
        f"{label}.request_projection field set mismatch",
    )
    _require(projection.get("role") == role, f"{label} role drift")
    _require(
        projection.get("allowed_fields") == list(allowed_fields),
        f"{label} semantic field boundary drift",
    )
    _require(
        projection.get("wire_fields") == list(wire_fields),
        f"{label} wire field boundary drift",
    )
    hashes = _required_mapping(
        projection.get("field_value_sha256"),
        label=f"{label}.request_projection.field_value_sha256",
    )
    _require(
        set(hashes) == set(allowed_fields),
        f"{label} projected value-hash fields drift",
    )
    for field, digest in hashes.items():
        _required_sha(digest, label=f"{label}.request_projection.{field}")
    _required_sha(
        projection.get("canonical_payload_sha256"),
        label=f"{label}.request_projection.canonical_payload_sha256",
    )
    _require(
        projection.get("official_target_policy") == official_target_policy,
        f"{label} official-target policy drift",
    )
    _require(
        projection.get("repair_trigger_source") in allowed_repair_triggers,
        f"{label} repair trigger drift",
    )
    _require(
        projection.get("reference_diagnostic_can_trigger_repair") is False,
        f"{label} allows reference-triggered repair",
    )
    _require(
        projection.get("api_key_env") == "DEEPSEEK_API_KEY"
        and projection.get("plaintext_credential_persisted") is False,
        f"{label} credential policy drift",
    )


def _validate_terminal_provider_boundaries(
    row: Mapping[str, Any], *, sample_id: str
) -> None:
    generation = _required_mapping(
        row.get("generation"), label=f"generation: {sample_id}"
    )
    repair_used = generation.get("repair_used")
    _require(
        isinstance(repair_used, bool),
        f"generation.repair_used must be boolean: {sample_id}",
    )
    if repair_used:
        rewrite_role = "rewrite_repair"
        rewrite_fields = ("source_analysis", "repair_feedback")
        rewrite_wire_fields = ("analysis", "repair_feedback")
        rewrite_triggers: set[str | None] = {
            "deterministic_validation",
            "validator_a",
        }
        validator_a_role = "validator_a_repair"
    else:
        rewrite_role = "rewrite_primary"
        rewrite_fields = ("source_analysis",)
        rewrite_wire_fields = ("analysis",)
        rewrite_triggers = {None}
        validator_a_role = "validator_a_primary"
    _validate_provider_projection(
        generation.get("provider"),
        label=f"generation provider: {sample_id}",
        role=rewrite_role,
        allowed_fields=rewrite_fields,
        wire_fields=rewrite_wire_fields,
        official_target_policy="forbidden",
        allowed_repair_triggers=rewrite_triggers,
        require_normal_stop=row.get("terminal_status")
        != "GENERATION_QUALITY_REJECT",
    )

    validator_a = _required_mapping(
        row.get("validator_a"), label=f"validator A: {sample_id}"
    )
    if validator_a.get("complete") is True:
        _validate_provider_projection(
            validator_a.get("provider"),
            label=f"validator A provider: {sample_id}",
            role=validator_a_role,
            allowed_fields=(
                "source_analysis",
                "teacher_response_analysis",
                "rewritten_minutes",
            ),
            wire_fields=(
                "source_analysis",
                "teacher_response_analysis",
                "rewritten_minutes",
            ),
            official_target_policy="forbidden",
            allowed_repair_triggers={None},
            require_normal_stop=row.get("terminal_status") == "PASS",
        )

    reference = _required_mapping(
        row.get("reference_diagnostic"),
        label=f"reference diagnostic: {sample_id}",
    )
    if reference.get("complete") is True:
        _validate_provider_projection(
            reference.get("provider"),
            label=f"reference provider: {sample_id}",
            role="validator_b",
            allowed_fields=("official_minutes", "rewritten_minutes"),
            wire_fields=("official_minutes", "rewritten_minutes"),
            official_target_policy="required_exact_normalized_official_minutes",
            allowed_repair_triggers={None},
        )


def _load_source(
    source_root: Path,
    *,
    expected_summary_sha256: str,
    expected_handoff_sha256: str,
    expected_sample_id_sha256: str,
    expected_split_counts: Mapping[str, int],
) -> tuple[dict[str, list[SourceRow]], dict[str, Any], dict[str, Any]]:
    root = source_root.resolve()
    _require(root.is_dir() and not root.is_symlink(), f"invalid source root: {root}")
    summary_path = root / "summary.json"
    handoff_path = root / "handoff.json"
    _require(sha256_file(summary_path) == expected_summary_sha256, "source summary SHA-256 mismatch")
    _require(sha256_file(handoff_path) == expected_handoff_sha256, "source handoff SHA-256 mismatch")
    summary = _read_json(summary_path, label="source summary")
    handoff = _read_json(handoff_path, label="source handoff")
    _require(handoff.get("summary", {}).get("sha256") == expected_summary_sha256, "source handoff does not bind summary")
    artifacts = _required_mapping(summary.get("artifacts"), label="source artifacts")
    result: dict[str, list[SourceRow]] = {}
    all_ids: list[str] = []
    meeting_splits: dict[str, str] = {}
    for split in SPLITS:
        expected_count = expected_split_counts.get(split)
        _require(isinstance(expected_count, int) and expected_count > 0, f"invalid expected source count: {split}")
        split_record = _required_mapping(artifacts.get(split), label=f"source artifacts.{split}")
        candidate_path, candidate_rows = _verify_descriptor(
            root, split_record.get("sft_candidate"), label=f"source {split} candidate"
        )
        manifest_path, manifest_rows = _verify_descriptor(
            root, split_record.get("manifest"), label=f"source {split} manifest"
        )
        _require(
            split_record.get("rows") == expected_count,
            f"source {split} split-level count mismatch",
        )
        _require(
            candidate_rows in (None, expected_count)
            and manifest_rows in (None, expected_count),
            f"source {split} descriptor count mismatch",
        )
        candidates = _read_jsonl(candidate_path, label=f"source {split} candidate")
        manifests = _read_jsonl(manifest_path, label=f"source {split} manifest")
        _require(len(candidates) == len(manifests) == expected_count, f"source {split} count mismatch")
        rows: list[SourceRow] = []
        for source_index, (candidate, manifest) in enumerate(zip(candidates, manifests)):
            _require(set(candidate) == {"prompt", "response"}, f"source {split}:{source_index} candidate schema drift")
            sample_id = _required_string(manifest.get("sample_id"), label=f"source {split}:{source_index}.sample_id")
            _require(manifest.get("split") == split, f"source split mismatch: {sample_id}")
            analysis = _required_string(manifest.get("analysis"), label=f"source analysis: {sample_id}")
            prompt = _required_string(candidate.get("prompt"), label=f"source prompt: {sample_id}")
            _required_sha(
                manifest.get("official_minutes_sha256"),
                label=f"source official SHA: {sample_id}",
            )
            meeting = _required_string(manifest.get("meeting_date"), label=f"source meeting: {sample_id}")
            _require(manifest.get("prompt_sha256") == sha256_text(prompt), f"source prompt hash mismatch: {sample_id}")
            _require(manifest.get("analysis_sha256", sha256_text(analysis)) == sha256_text(analysis), f"source analysis hash mismatch: {sample_id}")
            previous = meeting_splits.setdefault(meeting, split)
            _require(previous == split, f"source meeting appears in multiple splits: {meeting}")
            rows.append(SourceRow(sample_id, split, source_index, candidate, manifest))
            all_ids.append(sample_id)
        result[split] = rows
    _require(len(all_ids) == len(set(all_ids)), "source sample IDs are not unique")
    _require(_id_digest(all_ids) == expected_sample_id_sha256, "source sample-ID digest mismatch")
    return result, summary, handoff


def _load_final_acquisition(acquisition_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Path]]:
    root = acquisition_root.resolve()
    _require(root.is_dir() and not root.is_symlink(), f"invalid acquisition root: {root}")
    summary = _read_json(root / "summary.json", label="acquisition summary")
    handoff = _read_json(root / "handoff.json", label="acquisition handoff")
    contract = _read_json(root / "prompt_contract.json", label="acquisition prompt contract")
    _require(_contains_secret((summary, handoff, contract)) is None, "acquisition metadata contains credential material")
    _require(
        summary.get("unresolved_failure_count") == 0,
        "acquisition contains unresolved failures",
    )
    _require(
        summary.get("quality_status") == "passed",
        "acquisition is not a final verified release",
    )
    _require(
        summary.get("prompt_contract_sha256")
        == sha256_file(root / "prompt_contract.json"),
        "acquisition prompt-contract binding drift",
    )
    summary_sha = sha256_file(root / "summary.json")
    handoff_summary = handoff.get("summary")
    _require(isinstance(handoff_summary, Mapping) and handoff_summary.get("sha256") == summary_sha, "acquisition handoff does not bind final summary")
    artifacts = _required_mapping(summary.get("artifacts"), label="acquisition artifacts")
    terminal: dict[str, list[dict[str, Any]]] = {}
    artifact_paths: dict[str, Path] = {}
    for split in SPLITS:
        record = _required_mapping(artifacts.get(split), label=f"acquisition artifacts.{split}")
        for name in ("terminal", "sft_candidate", "manifest"):
            path, _ = _verify_descriptor(root, record.get(name), label=f"acquisition {split} {name}")
            artifact_paths[f"{split}.{name}"] = path
        terminal[split] = _read_jsonl(artifact_paths[f"{split}.terminal"], label=f"acquisition {split} terminal")
    return summary, contract, terminal, artifact_paths


def _percentile(values: Sequence[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _token_stats(values: Sequence[int]) -> dict[str, int]:
    return {
        "min": min(values, default=0),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values, default=0),
    }


def _token_replay(
    *, tokenizer: Any, system_prompt: str, prompt: str, response: str, sample_id: str
) -> dict[str, int]:
    _require(response.count(BOUNDARY) == 1, f"invalid response boundary: {sample_id}")
    _require("<think>" not in response.lower(), f"response contains opening think tag: {sample_id}")
    reasoning, answer = response.split(BOUNDARY, 1)
    _require(bool(reasoning.strip()), f"reasoning content invalid: {sample_id}")
    _require(answer == answer.strip() and bool(answer), f"answer whitespace/content invalid: {sample_id}")
    _require("\n" not in answer and not _CONTROL_RE.search(answer), f"answer must be one paragraph without control tags: {sample_id}")
    rendered = render_sft_prompt(
        tokenizer,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    _require(rendered.count("<think>") == 1 and "</think>" not in rendered, f"chat template reasoning prefix drift: {sample_id}")
    eos = _required_string(getattr(tokenizer, "eos_token", None), label="tokenizer.eos_token")
    completion = response if response.endswith(eos) else response + eos
    prompt_ids = tokenize_sft_text(tokenizer, rendered)
    full_ids = tokenize_sft_text(tokenizer, rendered + completion)
    _require(full_ids[: len(prompt_ids)] == prompt_ids, f"completion mask prefix mismatch: {sample_id}")
    bos_id = getattr(tokenizer, "bos_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    _require(bos_id is not None and full_ids.count(int(bos_id)) == 1, f"single-BOS gate failed: {sample_id}")
    _require(eos_id is not None and full_ids[-1] == int(eos_id) and full_ids.count(int(eos_id)) == 1, f"single-EOS gate failed: {sample_id}")
    completion_tokens = len(full_ids) - len(prompt_ids)
    _require(completion_tokens > 0, f"empty completion token span: {sample_id}")
    return {
        "prompt_tokens": len(prompt_ids),
        "reasoning_tokens": len(tokenize_sft_text(tokenizer, reasoning)),
        "answer_tokens": len(tokenize_sft_text(tokenizer, answer)),
        "completion_tokens": completion_tokens,
        "total_tokens": len(full_ids),
        "masked_prompt_tokens": len(prompt_ids),
        "unmasked_completion_tokens": completion_tokens,
    }


def _source_file_bindings(source_root: Path, source_summary: Mapping[str, Any]) -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    artifacts = _required_mapping(source_summary.get("artifacts"), label="source artifacts")
    for split in SPLITS:
        record = _required_mapping(artifacts.get(split), label=f"source artifacts.{split}")
        bindings[split] = {
            name: dict(_required_mapping(record.get(name), label=f"source {split}.{name}"))
            for name in ("sft_candidate", "manifest")
        }
    return bindings


def _file_records(root: Path, relative_paths: Sequence[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative in sorted(relative_paths):
        path = root / relative
        record: dict[str, Any] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".jsonl":
            record["rows"] = len(path.read_text(encoding="utf-8").splitlines())
        result[relative] = record
    return result


def publish_release(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    acquisition_root: Path = DEFAULT_ACQUISITION_ROOT,
    release_root: Path = DEFAULT_RELEASE_ROOT,
    tokenizer: Any,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    expected_source_summary_sha256: str = SOURCE_SUMMARY_SHA256,
    expected_source_handoff_sha256: str = SOURCE_HANDOFF_SHA256,
    expected_source_sample_id_sha256: str = SOURCE_SAMPLE_ID_SHA256,
    expected_split_counts: Mapping[str, int] = SOURCE_SPLIT_COUNTS,
) -> dict[str, Any]:
    """Validate and atomically publish the PASS-only paper chk-2 release."""

    source_by_split, source_summary, _ = _load_source(
        source_root,
        expected_summary_sha256=expected_source_summary_sha256,
        expected_handoff_sha256=expected_source_handoff_sha256,
        expected_sample_id_sha256=expected_source_sample_id_sha256,
        expected_split_counts=expected_split_counts,
    )
    acquisition_summary, prompt_contract, terminal_by_split, acquisition_paths = _load_final_acquisition(acquisition_root)
    canonical_prompt_contract = generator._prompt_contract(
        code_sha256=sha256_file(Path(generator.__file__).resolve()),
        config=generator.ProviderConfig(),
    )
    _require(
        prompt_contract == canonical_prompt_contract,
        "acquisition prompt contract does not exactly match generator code",
    )
    acquisition_source = _required_mapping(
        acquisition_summary.get("source"), label="acquisition source binding"
    )
    _require(
        acquisition_source.get("summary_sha256") == expected_source_summary_sha256,
        "acquisition source-summary binding drift",
    )
    _require(
        acquisition_source.get("handoff_sha256") == expected_source_handoff_sha256,
        "acquisition source-handoff binding drift",
    )
    _require(
        acquisition_source.get("sample_id_sha256")
        == expected_source_sample_id_sha256,
        "acquisition source-ID binding drift",
    )
    _require(
        acquisition_source.get("split_counts") == dict(expected_split_counts),
        "acquisition source split-count binding drift",
    )
    _require(
        acquisition_source.get("total_rows") == sum(expected_split_counts.values()),
        "acquisition source total-row binding drift",
    )
    source_root_value = _required_string(
        acquisition_source.get("root"), label="acquisition source.root"
    )
    source_root_binding = Path(source_root_value)
    if not source_root_binding.is_absolute():
        source_root_binding = REPO_ROOT / source_root_binding
    _require(
        source_root_binding.resolve() == source_root.resolve(),
        "acquisition source-root binding drift",
    )
    _require(
        acquisition_source.get("merge_receipt_sha256")
        == sha256_file(source_root / "merge_receipt.json"),
        "acquisition source merge-receipt binding drift",
    )
    acquisition_source_artifacts = _required_mapping(
        acquisition_source.get("artifacts"),
        label="acquisition source artifacts",
    )
    canonical_source_artifacts = _required_mapping(
        source_summary.get("artifacts"), label="canonical source artifacts"
    )
    _require(
        set(acquisition_source_artifacts) == set(SPLITS),
        "acquisition source artifact split drift",
    )
    for split in SPLITS:
        observed_split = _required_mapping(
            acquisition_source_artifacts.get(split),
            label=f"acquisition source artifacts.{split}",
        )
        canonical_split = _required_mapping(
            canonical_source_artifacts.get(split),
            label=f"canonical source artifacts.{split}",
        )
        _require(
            observed_split.get("rows") == expected_split_counts[split],
            f"acquisition source artifact row-count drift: {split}",
        )
        for name in ("manifest", "sft_candidate"):
            observed_descriptor = _required_mapping(
                observed_split.get(name),
                label=f"acquisition source artifacts.{split}.{name}",
            )
            canonical_descriptor = _required_mapping(
                canonical_split.get(name),
                label=f"canonical source artifacts.{split}.{name}",
            )
            _required_string(
                observed_descriptor.get("path"),
                label=f"acquisition source artifacts.{split}.{name}.path",
            )
            _require(
                observed_descriptor.get("sha256")
                == canonical_descriptor.get("sha256"),
                f"acquisition source artifact SHA drift: {split}.{name}",
            )
    system_prompt = _student_system_prompt(prompt_contract)
    _require(
        prompt_contract.get("student_system_prompt_sha256")
        == sha256_text(system_prompt),
        "prompt contract student-system hash mismatch",
    )
    for key in (
        "student_user_prompt_template",
        "student_response_contract",
    ):
        _required_string(prompt_contract.get(key), label=f"prompt_contract.{key}")
    user_prompt_contract = _required_mapping(
        prompt_contract.get("student_user_prompt_contract"),
        label="prompt_contract.student_user_prompt_contract",
    )
    _require(
        user_prompt_contract
        == {
            "prefix": "Rewrite the following analysis as formal FOMC Minutes prose:\n\n",
            "json_keys": ["analysis"],
            "source_field": "source_analysis",
        },
        "student user-prompt contract drift",
    )
    _require(
        prompt_contract.get("request_field_allowlists")
        == {
            "rewrite_primary": ["analysis"],
            "rewrite_repair": ["analysis", "repair_feedback"],
            "validator_a_primary": [
                "source_analysis",
                "teacher_response_analysis",
                "rewritten_minutes",
            ],
            "validator_a_repair": [
                "source_analysis",
                "teacher_response_analysis",
                "rewritten_minutes",
            ],
            "validator_b": ["official_minutes", "rewritten_minutes"],
        },
        "prompt-contract request field boundaries drift",
    )
    selection_policy = _required_mapping(
        prompt_contract.get("selection_policy"),
        label="prompt_contract.selection_policy",
    )
    expected_selection_policy = {
        "deterministic_gate_required": True,
        "validator_a_required": True,
        "validator_b_diagnostic_only": True,
        "validator_b_never_repairs": True,
        "validator_b_never_changes_training_pass": True,
        "maximum_rewrite_repairs": 1,
        "complete_native_cot_preserved": True,
        "reasoning_operational_meta_allowed": True,
        "reasoning_meta_used_for_rejection": False,
        "reasoning_sanitization": "none",
        "source_phrase_sentence_reuse_allowed": True,
        "near_copy_used_for_rejection": False,
        "exact_full_source_copy_used_for_rejection": True,
        "validator_b_lexical_overlap_warning": False,
        "total_token_limit": None,
        "total_token_used_for_rejection": False,
        "tokenizer_length_recorded": True,
        "truncation": False,
        "human_review": False,
    }
    _require(
        dict(selection_policy) == expected_selection_policy,
        "prompt-contract selection policy drift",
    )
    provider_contract = _required_mapping(
        prompt_contract.get("provider_contract"),
        label="prompt_contract.provider_contract",
    )
    _require(
        provider_contract.get("model") == "deepseek-v4-flash"
        and provider_contract.get("api_key_env") == "DEEPSEEK_API_KEY"
        and provider_contract.get("fallback") == "forbidden"
        and provider_contract.get("temperature") is None
        and provider_contract.get("top_p") is None,
        "prompt-contract provider policy drift",
    )
    lineage = {
        "analysis_source_lineage": "target_derived_from_official_minutes",
        "analysis_teacher_saw_official_target": True,
        "source_population_target_derived": True,
        "rewrite_teacher_input_fields": ["source_analysis"],
        "rewrite_teacher_received_official_validator_feedback": False,
        "target_is_teacher_synthetic_rewrite": True,
        "student_prompt_has_direct_target_field": False,
        "rewrite_teacher_saw_official_target": False,
        "input_fidelity_validator_saw_official_target": False,
        "official_reference_validator_saw_official_target": True,
        "official_reference_feedback_returned_to_rewrite_teacher": False,
        "official_reference_score_used_for_selection": False,
        "rewrite_stage_target_assisted_selection": False,
        "reference_validator_saw_official_target": True,
        "reference_validator_used_for_training_selection": False,
        "reference_audit_only": True,
        "official_minutes_used_as_student_target": False,
        "teacher_response_analysis_was_sanitized": False,
        "teacher_response_analysis_is_complete_native_cot": True,
        "teacher_response_analysis_operational_meta_allowed": True,
        "teacher_response_analysis_draft_deliberation_allowed": True,
        "source_wording_reuse_allowed": True,
        "near_copy_used_for_training_rejection": False,
        "exact_full_source_copy_used_for_training_rejection": True,
        "reference_lexical_overlap_used_for_warning": False,
        "token_length_used_for_training_rejection": False,
        "maximum_total_tokens": None,
        "training_only": True,
        "evaluation_eligible": False,
        "suitable_for_leakage_safe_evaluation": False,
        "human_review_required": False,
    }
    _require(
        prompt_contract.get("lineage") == lineage,
        "prompt-contract lineage drift",
    )

    release = release_root.resolve()
    if release.exists():
        return verify_release(release, expected_manifest_sha256=None)
    release.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{release.name}.", dir=release.parent))
    try:
        pass_ids: list[str] = []
        reject_ids: list[str] = []
        pass_ids_by_split: dict[str, list[str]] = {}
        reject_ids_by_split: dict[str, list[str]] = {}
        split_counts: dict[str, int] = {}
        rejection_counts: dict[str, int] = {}
        primary_repair_counts: Counter[str] = Counter()
        rejection_reason_counts: Counter[str] = Counter()
        reference_score_values: list[int] = []
        reference_verdict_counts: Counter[str] = Counter()
        prompt_token_values: list[int] = []
        reasoning_token_values: list[int] = []
        answer_token_values: list[int] = []
        completion_token_values: list[int] = []
        total_token_values: list[int] = []
        all_release_meetings: dict[str, str] = {}
        all_rejections: list[dict[str, Any]] = []
        all_reference: list[dict[str, Any]] = []

        for split in SPLITS:
            source_rows = source_by_split[split]
            terminal_rows = terminal_by_split[split]
            _require(len(terminal_rows) == len(source_rows), f"terminal {split} does not cover source population")
            acquisition_candidates = _read_jsonl(acquisition_paths[f"{split}.sft_candidate"], label=f"acquisition {split} candidate")
            acquisition_manifests = _read_jsonl(acquisition_paths[f"{split}.manifest"], label=f"acquisition {split} manifest")
            output_rows: list[dict[str, str]] = []
            sidecars: list[dict[str, Any]] = []
            split_pass_ids: list[str] = []
            split_reject_ids: list[str] = []
            candidate_index = 0
            for source, terminal in zip(source_rows, terminal_rows):
                _validate_terminal_shape(
                    terminal, label=f"terminal {split}:{source.source_index}"
                )
                secret_path = _contains_secret(terminal)
                _require(secret_path is None, f"terminal row contains credential material at {secret_path}")
                sample_id = _required_string(terminal.get("sample_id"), label="terminal.sample_id")
                _require(sample_id == source.sample_id, f"terminal source order/identity mismatch: {sample_id}")
                _require(terminal.get("split") == split, f"terminal split mismatch: {sample_id}")
                _require(terminal.get("source_index") == source.source_index, f"terminal source_index mismatch: {sample_id}")
                source_analysis = _required_string(terminal.get("source_analysis"), label=f"terminal source_analysis: {sample_id}")
                _require(source_analysis == source.source_analysis, f"terminal source analysis drift: {sample_id}")
                _require(terminal.get("source_analysis_sha256") == sha256_text(source_analysis), f"terminal source analysis hash mismatch: {sample_id}")
                _require(terminal.get("official_minutes_sha256") == source.official_minutes_sha256, f"terminal official target hash mismatch: {sample_id}")
                terminal_lineage = _required_mapping(terminal.get("lineage"), label=f"terminal lineage: {sample_id}")
                _require(
                    dict(terminal_lineage) == lineage,
                    f"terminal lineage drift: {sample_id}",
                )

                _require(
                    terminal.get("schema_version") == TERMINAL_SCHEMA_VERSION,
                    f"terminal schema mismatch: {sample_id}",
                )
                _require(
                    terminal.get("meeting_date") == source.meeting_date,
                    f"terminal meeting mismatch: {sample_id}",
                )
                status = terminal.get("terminal_status")
                _require(
                    status
                    in {
                        "PASS",
                        "GENERATION_QUALITY_REJECT",
                        "INPUT_FIDELITY_REJECT",
                    },
                    f"unresolved terminal status: {sample_id}: {status!r}",
                )
                training_pass = terminal.get("training_pass")
                _require(
                    isinstance(training_pass, bool),
                    f"terminal training_pass must be boolean: {sample_id}",
                )
                deterministic = _deterministic_result(
                    terminal, label=f"terminal {sample_id}"
                )
                validator_a = _required_mapping(
                    terminal.get("validator_a"), label=f"validator A: {sample_id}"
                )
                det_pass = _machine_pass(
                    deterministic, label=f"deterministic validation: {sample_id}"
                )
                a_complete = validator_a.get("complete")
                _require(
                    isinstance(a_complete, bool),
                    f"validator A complete must be boolean: {sample_id}",
                )
                a_pass = _machine_pass(validator_a, label=f"validator A: {sample_id}")
                eligible = det_pass and a_pass
                _require(
                    training_pass == eligible == (status == "PASS"),
                    f"terminal PASS is not deterministic+A eligibility: {sample_id}",
                )
                _require(
                    not a_pass or a_complete,
                    f"validator A pass is incomplete: {sample_id}",
                )
                _validate_terminal_provider_boundaries(
                    terminal, sample_id=sample_id
                )

                if status != "PASS":
                    reasons = terminal.get("rejection_reasons")
                    _require(isinstance(reasons, list) and reasons and all(isinstance(item, str) and item for item in reasons), f"quality reject has no reason codes: {sample_id}")
                    stage = _required_string(terminal.get("rejection_stage"), label=f"rejection_stage: {sample_id}")
                    split_reject_ids.append(sample_id)
                    reject_ids.append(sample_id)
                    for reason in reasons:
                        rejection_reason_counts[str(reason)] += 1
                    all_rejections.append(
                        {
                            "schema_version": RELEASE_SCHEMA_VERSION,
                            "sample_id": sample_id,
                            "split": split,
                            "source_index": source.source_index,
                            "meeting_date": source.meeting_date,
                            "rejection_stage": stage,
                            "rejection_reasons": list(reasons),
                            "repair_used": bool(
                                _required_mapping(
                                    terminal.get("generation"),
                                    label=f"generation: {sample_id}",
                                ).get("repair_used", False)
                            ),
                            "generation_record_sha256": sha256_text(
                                canonical_json(terminal.get("generation"))
                            ),
                            "validator_a_record_sha256": sha256_text(
                                canonical_json(terminal.get("validator_a"))
                            ),
                            "reference_diagnostic_record_sha256": sha256_text(
                                canonical_json(terminal.get("reference_diagnostic"))
                            ),
                        }
                    )
                    continue

                reasoning = _required_string(terminal.get("teacher_response_analysis"), label=f"teacher reasoning: {sample_id}")
                answer = _required_string(terminal.get("rewritten_minutes"), label=f"rewritten Minutes: {sample_id}")
                prompt = _required_string(terminal.get("student_prompt"), label=f"student prompt: {sample_id}")
                response = _required_string(terminal.get("sft_response"), label=f"SFT response: {sample_id}")
                _require(prompt == source.prompt, f"student prompt is not source prompt: {sample_id}")
                _require(response == reasoning + BOUNDARY + answer, f"SFT response assembly mismatch: {sample_id}")
                hash_expectations = {
                    "source_analysis_sha256": sha256_text(source_analysis),
                    "teacher_response_analysis_sha256": sha256_text(reasoning),
                    "rewritten_minutes_sha256": sha256_text(answer),
                    "prompt_sha256": sha256_text(prompt),
                    "response_sha256": sha256_text(response),
                }
                for key, expected in hash_expectations.items():
                    _require(terminal.get(key) == expected, f"terminal {key} mismatch: {sample_id}")
                reference = _required_mapping(
                    terminal.get("reference_diagnostic"),
                    label=f"reference diagnostic: {sample_id}",
                )
                _reference_complete(
                    reference, label=f"reference diagnostic: {sample_id}"
                )
                reference_score = reference.get("overall_score")
                score_int = int(round(float(reference_score)))
                reference_score_values.append(score_int)
                verdict = reference.get("diagnostic_pass")
                _require(
                    isinstance(verdict, bool),
                    f"reference diagnostic_pass must be boolean: {sample_id}",
                )
                reference_verdict_counts[str(verdict).lower()] += 1
                all_reference.append(
                    {
                        "schema_version": RELEASE_SCHEMA_VERSION,
                        "sample_id": sample_id,
                        "split": split,
                        "source_index": source.source_index,
                        "overall_score": reference_score,
                        "diagnostic_pass": verdict,
                        "warning": reference.get("warning"),
                        "used_for_selection": False,
                        "used_for_repair": False,
                        "diagnostic": dict(reference),
                    }
                )
                tokens = _token_replay(
                    tokenizer=tokenizer,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    response=response,
                    sample_id=sample_id,
                )
                prompt_token_values.append(tokens["prompt_tokens"])
                reasoning_token_values.append(tokens["reasoning_tokens"])
                answer_token_values.append(tokens["answer_tokens"])
                completion_token_values.append(tokens["completion_tokens"])
                total_token_values.append(tokens["total_tokens"])
                previous_split = all_release_meetings.setdefault(source.meeting_date, split)
                _require(previous_split == split, f"published meeting crosses splits: {source.meeting_date}")
                release_index = len(output_rows)
                output = {"prompt": prompt, "response": response}
                _require(set(output) == {"prompt", "response"}, "internal training-row schema error")

                _require(candidate_index < len(acquisition_candidates), f"missing acquisition candidate: {sample_id}")
                acq_candidate = acquisition_candidates[candidate_index]
                acq_manifest = acquisition_manifests[candidate_index]
                _require(acq_candidate == output, f"acquisition candidate drift: {sample_id}")
                _require(acq_manifest.get("sample_id") == sample_id, f"acquisition manifest order drift: {sample_id}")
                candidate_index += 1

                output_rows.append(output)
                split_pass_ids.append(sample_id)
                pass_ids.append(sample_id)
                generation = _required_mapping(
                    terminal.get("generation"), label=f"generation: {sample_id}"
                )
                repair_used = generation.get("repair_used")
                _require(
                    isinstance(repair_used, bool),
                    f"generation.repair_used must be boolean: {sample_id}",
                )
                primary_repair_counts["repair" if repair_used else "primary"] += 1
                sidecars.append(
                    {
                        "schema_version": RELEASE_SCHEMA_VERSION,
                        "sample_id": sample_id,
                        "split": split,
                        "release_index": release_index,
                        "source_index": source.source_index,
                        "meeting_date": source.meeting_date,
                        "source_analysis_sha256": sha256_text(source_analysis),
                        "official_minutes_sha256": source.official_minutes_sha256,
                        "prompt_sha256": sha256_text(prompt),
                        "teacher_response_analysis_sha256": sha256_text(reasoning),
                        "rewritten_minutes_sha256": sha256_text(answer),
                        "response_sha256": sha256_text(response),
                        "reasoning_transformation": "none",
                        "repair_used": repair_used,
                        "generation": dict(generation),
                        "deterministic_validation": dict(_required_mapping(deterministic, label=f"deterministic: {sample_id}")),
                        "validator_a": dict(_required_mapping(validator_a, label=f"validator A: {sample_id}")),
                        "reference_diagnostic": {
                            "complete": True,
                            "overall_score": reference_score,
                            "diagnostic_pass": verdict,
                            "warning": reference.get("warning"),
                            "used_for_selection": False,
                            "used_for_repair": False,
                            "record_sha256": sha256_text(canonical_json(reference)),
                        },
                        "tokens": tokens,
                        "lineage": dict(lineage),
                    }
                )
            _require(candidate_index == len(acquisition_candidates) == len(acquisition_manifests), f"acquisition PASS artifacts do not match terminal PASS set: {split}")
            _require(bool(output_rows), f"PASS-only release split is empty: {split}")
            _write_jsonl(staging / f"minutes_alignment/{split}.jsonl", output_rows)
            _write_jsonl(staging / f"minutes_alignment/manifests/{split}.jsonl", sidecars)
            pass_ids_by_split[split] = split_pass_ids
            reject_ids_by_split[split] = split_reject_ids
            split_counts[split] = len(split_pass_ids)
            rejection_counts[split] = len(split_reject_ids)

        _require(set(pass_ids).isdisjoint(reject_ids), "PASS and REJECT populations overlap")
        source_ids = [row.sample_id for split in SPLITS for row in source_by_split[split]]
        _require(set(pass_ids) | set(reject_ids) == set(source_ids), "PASS/REJECT do not partition the source population")
        _require(len(pass_ids) + len(reject_ids) == sum(expected_split_counts.values()), "terminal population count mismatch")

        _write_jsonl(staging / "audits/rejections.jsonl", all_rejections)
        _write_jsonl(staging / "audits/reference_diagnostics.jsonl", all_reference)
        shutil.copyfile(acquisition_root / "prompt_contract.json", staging / "prompt_contract.json")

        tokenizer_binding = {
            "path": str(tokenizer_path.resolve()),
            "class": type(tokenizer).__name__,
            "bos_token": getattr(tokenizer, "bos_token", None),
            "bos_token_id": getattr(tokenizer, "bos_token_id", None),
            "eos_token": getattr(tokenizer, "eos_token", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        tokenizer_replay = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "status": "passed",
            "tokenizer": tokenizer_binding,
            "row_count": len(pass_ids),
            "contract": {
                "single_bos": True,
                "single_eos": True,
                "single_literal_close_think_boundary": True,
                "chat_template_supplies_opening_think": True,
                "prompt_is_full_token_prefix": True,
                "completion_only_prompt_masked": True,
                "reasoning_boundary_answer_eos_unmasked": True,
                "truncation": False,
                "total_length_gate": False,
                "total_max": None,
                "observed_lengths_recorded": True,
            },
            "token_stats": {
                "prompt": _token_stats(prompt_token_values),
                "reasoning": _token_stats(reasoning_token_values),
                "answer": _token_stats(answer_token_values),
                "completion": _token_stats(completion_token_values),
                "total": _token_stats(total_token_values),
            },
        }
        _write_json(staging / "audits/tokenizer_replay.json", tokenizer_replay)

        coverage = {
            split: {
                "source": expected_split_counts[split],
                "pass": split_counts[split],
                "reject": rejection_counts[split],
                "pass_rate": round(split_counts[split] / expected_split_counts[split], 8),
            }
            for split in SPLITS
        }
        data_quality = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "status": "passed",
            "grain": "one source analysis and teacher-synthetic Minutes rewrite per PASS row",
            "intended_use": "paper Model chk-2 SFT: de-styled analysis to synthetic FOMC Minutes prose",
            "source_rows": len(source_ids),
            "pass_rows": len(pass_ids),
            "reject_rows": len(reject_ids),
            "split_counts": split_counts,
            "rejection_counts": rejection_counts,
            "coverage": coverage,
            "coverage_threshold": None,
            "all_splits_nonempty": True,
            "unresolved_rows": 0,
            "selection_rule": "generation.deterministic_validation.machine_pass AND validator_a.machine_pass; reference_diagnostic never filters",
            "reference_score_used_for_selection": False,
            "reference_verdict_counts": dict(sorted(reference_verdict_counts.items())),
            "reference_score_stats": _token_stats(reference_score_values),
            "primary_repair_counts": dict(sorted(primary_repair_counts.items())),
            "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
            "selection_bias_disclosure": (
                "PASS-only row counts vary because deterministic input-fidelity gates may reject rows; "
                "the official-reference score does not filter rows."
            ),
            "integrity": {
                "source_sample_id_sha256": _id_digest(source_ids),
                "pass_sample_id_sha256": _id_digest(pass_ids),
                "reject_sample_id_sha256": _id_digest(reject_ids),
                "split_pass_sample_id_sha256": {
                    split: _id_digest(pass_ids_by_split[split]) for split in SPLITS
                },
                "split_reject_sample_id_sha256": {
                    split: _id_digest(reject_ids_by_split[split]) for split in SPLITS
                },
                "pass_reject_disjoint": True,
                "pass_reject_exact_source_partition": True,
                "meeting_split_conflicts": 0,
            },
            "checks": {
                "source_binding": "passed",
                "terminal_completeness": "passed",
                "pass_rule_recomputed": "passed",
                "split_preservation": "passed",
                "meeting_isolation": "passed",
                "training_row_schema": "passed",
                "response_assembly": "passed",
                "hashes": "passed",
                "lineage": "passed",
                "reference_diagnostics_complete": "passed",
                "credential_absence": "passed",
                "tokenizer_replay": "passed",
            },
            "lineage": lineage,
        }
        _write_json(staging / "audits/data_quality.json", data_quality)

        payload_paths = [
            "prompt_contract.json",
            "audits/data_quality.json",
            "audits/rejections.jsonl",
            "audits/reference_diagnostics.jsonl",
            "audits/tokenizer_replay.json",
            *(f"minutes_alignment/{split}.jsonl" for split in SPLITS),
            *(f"minutes_alignment/manifests/{split}.jsonl" for split in SPLITS),
        ]
        files = _file_records(staging, payload_paths)
        release_manifest = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": release.name,
            "created_at_utc": _utc_now(),
            "immutable": True,
            "quality_status": "passed",
            "dataset_role": DATASET_ROLE,
            "training_scope": TRAINING_SCOPE,
            "paper_model_label": "Model chk-2",
            "canonical_stage_id": None,
            "dag_bindable": False,
            "promotable_as_canonical_chk2": False,
            "canonical_naming_note": "repository canonical chk2 is analysis GRPO; this is an independent paper_chk2 Minutes-SFT branch",
            "training_mapping": "target-derived de-styled analysis -> teacher reasoning -> teacher-synthetic Minutes rewrite",
            "source": {
                "path": str(source_root.resolve()),
                "summary_sha256": expected_source_summary_sha256,
                "handoff_sha256": expected_source_handoff_sha256,
                "sample_id_sha256": expected_source_sample_id_sha256,
                "split_counts": dict(expected_split_counts),
                "files": _source_file_bindings(source_root, source_summary),
            },
            "acquisition": {
                "path": str(acquisition_root.resolve()),
                "summary_sha256": sha256_file(acquisition_root / "summary.json"),
                "handoff_sha256": sha256_file(acquisition_root / "handoff.json"),
                "prompt_contract_sha256": sha256_file(acquisition_root / "prompt_contract.json"),
            },
            "lineage": lineage,
            "split_counts": split_counts,
            "rejection_counts": rejection_counts,
            "total_rows": len(pass_ids),
            "source_rows": len(source_ids),
            "files": files,
        }
        _write_json(staging / "release_manifest.json", release_manifest)
        handoff = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": release.name,
            "created_at_utc": release_manifest["created_at_utc"],
            "immutable": True,
            "quality_status": "passed",
            "dataset_role": DATASET_ROLE,
            "training_scope": TRAINING_SCOPE,
            "dataset_path": str(release / "minutes_alignment"),
            "release_manifest": {
                "path": "release_manifest.json",
                "sha256": sha256_file(staging / "release_manifest.json"),
            },
            "data_quality_audit": {
                "path": "audits/data_quality.json",
                "sha256": sha256_file(staging / "audits/data_quality.json"),
            },
            "tokenizer_replay": {
                "path": "audits/tokenizer_replay.json",
                "sha256": sha256_file(staging / "audits/tokenizer_replay.json"),
            },
            "split_counts": split_counts,
            "rejection_counts": rejection_counts,
            "total_rows": len(pass_ids),
            "lineage": lineage,
        }
        _write_json(staging / "handoff.json", handoff)
        os.replace(staging, release)
        return handoff
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_release(
    release_root: Path, *, expected_manifest_sha256: str | None
) -> dict[str, Any]:
    """Verify sealed release bytes and the non-canonical paper-chk2 identity."""

    root = release_root.resolve()
    _require(root.is_dir() and not root.is_symlink(), f"invalid release root: {root}")
    manifest_path = root / "release_manifest.json"
    handoff_path = root / "handoff.json"
    if expected_manifest_sha256 is not None:
        _require(sha256_file(manifest_path) == expected_manifest_sha256, "release manifest SHA-256 mismatch")
    manifest = _read_json(manifest_path, label="release manifest")
    handoff = _read_json(handoff_path, label="release handoff")
    _require(manifest.get("schema_version") == RELEASE_SCHEMA_VERSION, "release schema mismatch")
    _require(manifest.get("dataset_role") == DATASET_ROLE, "release role mismatch")
    _require(manifest.get("training_scope") == TRAINING_SCOPE, "release scope mismatch")
    _require(manifest.get("quality_status") == "passed" and manifest.get("immutable") is True, "release is not immutable/pass")
    _require(manifest.get("dag_bindable") is False and manifest.get("promotable_as_canonical_chk2") is False, "release makes an unauthorized canonical chk2 claim")
    _require(handoff.get("release_manifest", {}).get("sha256") == sha256_file(manifest_path), "handoff manifest binding mismatch")
    expected_files = {
        "prompt_contract.json",
        "audits/data_quality.json",
        "audits/rejections.jsonl",
        "audits/reference_diagnostics.jsonl",
        "audits/tokenizer_replay.json",
        *(f"minutes_alignment/{split}.jsonl" for split in SPLITS),
        *(f"minutes_alignment/manifests/{split}.jsonl" for split in SPLITS),
    }
    files = _required_mapping(manifest.get("files"), label="release files")
    _require(set(files) == expected_files, "release sealed file set mismatch")
    physical_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    }
    _require(
        physical_files == expected_files | {"release_manifest.json", "handoff.json"},
        "release contains unbound or missing physical files",
    )
    for relative, descriptor in files.items():
        path, rows = _verify_descriptor(root, descriptor, label=f"release file {relative}")
        _require(str(path.relative_to(root)) == relative, f"release file key/path mismatch: {relative}")
        if relative.startswith("minutes_alignment/") and "/manifests/" not in relative:
            for line_number, row in enumerate(_read_jsonl(path, label=relative), start=1):
                _require(set(row) == {"prompt", "response"}, f"training row schema mismatch: {relative}:{line_number}")
                response = row["response"]
                _require(isinstance(response, str) and response.count(BOUNDARY) == 1 and "<think>" not in response.lower(), f"training response boundary mismatch: {relative}:{line_number}")
        if rows is not None:
            _require(rows >= 0, f"invalid row descriptor: {relative}")
    counts = _required_mapping(manifest.get("split_counts"), label="split_counts")
    _require(set(counts) == set(SPLITS) and all(isinstance(counts[s], int) and counts[s] > 0 for s in SPLITS), "release requires nonempty train/validation/test")
    _require(sum(counts.values()) == manifest.get("total_rows"), "release total_rows mismatch")
    sample_ids: list[str] = []
    for split in SPLITS:
        data_rows = _read_jsonl(
            root / f"minutes_alignment/{split}.jsonl",
            label=f"release {split} data",
        )
        sidecars = _read_jsonl(
            root / f"minutes_alignment/manifests/{split}.jsonl",
            label=f"release {split} sidecars",
        )
        _require(
            len(data_rows) == len(sidecars) == counts[split],
            f"release {split} data/sidecar population mismatch",
        )
        previous_source_index = -1
        for release_index, (data, sidecar) in enumerate(zip(data_rows, sidecars)):
            sample_id = _required_string(
                sidecar.get("sample_id"), label=f"{split}:{release_index}.sample_id"
            )
            _require(
                sidecar.get("schema_version") == RELEASE_SCHEMA_VERSION
                and sidecar.get("split") == split
                and sidecar.get("release_index") == release_index,
                f"release sidecar identity drift: {sample_id}",
            )
            source_index = sidecar.get("source_index")
            _require(
                isinstance(source_index, int) and source_index > previous_source_index,
                f"release source order drift: {sample_id}",
            )
            previous_source_index = source_index
            prompt = data["prompt"]
            response = data["response"]
            reasoning, answer = response.split(BOUNDARY, 1)
            expected_hashes = {
                "prompt_sha256": sha256_text(prompt),
                "response_sha256": sha256_text(response),
                "teacher_response_analysis_sha256": sha256_text(reasoning),
                "rewritten_minutes_sha256": sha256_text(answer),
            }
            for key, expected in expected_hashes.items():
                _require(
                    sidecar.get(key) == expected,
                    f"release sidecar {key} drift: {sample_id}",
                )
            _require(
                sidecar.get("reasoning_transformation") == "none",
                f"release reasoning transformation drift: {sample_id}",
            )
            reference = _required_mapping(
                sidecar.get("reference_diagnostic"),
                label=f"release reference diagnostic: {sample_id}",
            )
            _require(
                reference.get("complete") is True
                and reference.get("used_for_selection") is False
                and reference.get("used_for_repair") is False,
                f"release reference diagnostic policy drift: {sample_id}",
            )
            tokens = _required_mapping(
                sidecar.get("tokens"), label=f"release tokens: {sample_id}"
            )
            _require(
                isinstance(tokens.get("total_tokens"), int)
                and 0 < tokens["total_tokens"]
                and tokens.get("masked_prompt_tokens") == tokens.get("prompt_tokens")
                and tokens.get("unmasked_completion_tokens")
                == tokens.get("completion_tokens"),
                f"release token/mask record drift: {sample_id}",
            )
            sample_ids.append(sample_id)
    _require(
        len(sample_ids) == len(set(sample_ids)),
        "release contains duplicate sample IDs",
    )
    audit = _read_json(root / "audits/data_quality.json", label="data-quality audit")
    _require(audit.get("status") == "passed" and audit.get("split_counts") == counts, "data-quality population mismatch")
    _require(audit.get("unresolved_rows") == 0 and audit.get("reference_score_used_for_selection") is False, "data-quality selection contract mismatch")
    rejections = _read_jsonl(
        root / "audits/rejections.jsonl", label="release rejection ledger"
    )
    reference_rows = _read_jsonl(
        root / "audits/reference_diagnostics.jsonl",
        label="release reference diagnostics",
    )
    rejection_ids = [
        _required_string(row.get("sample_id"), label="rejection.sample_id")
        for row in rejections
    ]
    reference_ids = [
        _required_string(row.get("sample_id"), label="reference.sample_id")
        for row in reference_rows
    ]
    _require(
        len(rejection_ids) == audit.get("reject_rows")
        and set(reference_ids) == set(sample_ids)
        and set(rejection_ids).isdisjoint(sample_ids),
        "release PASS/REJECT/reference populations mismatch",
    )
    integrity = _required_mapping(audit.get("integrity"), label="audit.integrity")
    _require(
        integrity.get("pass_sample_id_sha256") == _id_digest(sample_ids)
        and integrity.get("reject_sample_id_sha256") == _id_digest(rejection_ids)
        and integrity.get("pass_reject_disjoint") is True
        and integrity.get("pass_reject_exact_source_partition") is True,
        "release population digest mismatch",
    )
    replay = _read_json(
        root / "audits/tokenizer_replay.json", label="tokenizer replay"
    )
    _require(
        replay.get("status") == "passed"
        and replay.get("row_count") == len(sample_ids)
        and replay.get("contract", {}).get("truncation") is False
        and replay.get("contract", {}).get("total_length_gate") is False
        and replay.get("contract", {}).get("total_max") is None
        and replay.get("contract", {}).get("observed_lengths_recorded") is True
        and replay.get("token_stats", {}).get("total", {}).get("max", 0) > 0,
        "tokenizer replay receipt mismatch",
    )
    _require(
        _contains_secret((manifest, handoff, audit, rejections, reference_rows))
        is None,
        "published release contains credential material",
    )
    return handoff


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment error
        raise PublicationError("transformers is required to load the tokenizer") from exc
    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception as exc:  # pragma: no cover - tokenizer-specific error
        raise PublicationError(f"cannot load local tokenizer: {path}: {exc}") from exc


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--acquisition-root", type=Path, default=DEFAULT_ACQUISITION_ROOT)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--expected-manifest-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.verify_only:
        result = verify_release(
            args.release_root,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    else:
        tokenizer = _load_tokenizer(args.tokenizer_path)
        result = publish_release(
            source_root=args.source_root,
            acquisition_root=args.acquisition_root,
            release_root=args.release_root,
            tokenizer=tokenizer,
            tokenizer_path=args.tokenizer_path,
        )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BOUNDARY",
    "DATASET_ROLE",
    "PublicationError",
    "RELEASE_SCHEMA_VERSION",
    "TRAINING_SCOPE",
    "publish_release",
    "sha256_file",
    "sha256_text",
    "verify_release",
]
