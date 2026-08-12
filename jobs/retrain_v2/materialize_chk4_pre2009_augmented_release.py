"""Publish a sealed, train-only pre-2009 chk4 Decision augmentation.

This module is intentionally independent of the existing Decision-v3 and
hierarchical-balanced-v1 publishers.  It never mutates either parent.  The
new release:

* reads unique core rows from the externally hash-pinned, sealed Decision-v3
  release;
* admits only completed 1993--2008 supplement teacher rows from ``train``;
* builds one deterministic 4-hold/2-hike/2-cut exposure schedule directly
  from unique rows (there is no intermediate direction-repeat layer);
* copies both SFT and GRPO validation/test bytes, plus their manifests,
  byte-for-byte from Decision-v3; and
* is published create-only with a complete file table, sealed modes, a
  manifest-bound handoff, and a verifier that replays every derived train row.

No command in this module starts training or uses a GPU.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import platform
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jobs.generation.generate_chk4_sft_targets import (
    STUDENT_SYSTEM_PROMPT,
    render_student_prompt,
)
from jobs.generation.generate_chk4_supplement import (
    BLIND_SCHEMA as SUPPLEMENT_BLIND_SCHEMA,
    CACHE_SCHEMA as SUPPLEMENT_PROVIDER_CACHE_SCHEMA,
    CONTRACT_SCHEMA as SUPPLEMENT_PROVIDER_CONTRACT_SCHEMA,
    SUMMARY_SCHEMA as SUPPLEMENT_SUMMARY_SCHEMA,
    TARGET_SCHEMA as SUPPLEMENT_TARGET_SCHEMA,
    qualitative_reasoning_has_number_or_date,
)
from jobs.generation.materialize_chk4_training_data import (
    MaterializationError,
    load_unique_rows,
)
from jobs.generation.prepare_chk4_supplement import (
    ADMISSION_PROFILE as SUPPLEMENT_ADMISSION_PROFILE,
    EXPECTED_CANDIDATES as SUPPLEMENT_EXPECTED_CANDIDATES,
    MIN_ADMITTED as SUPPLEMENT_PIPELINE_MIN_ADMITTED,
    MIN_VALID_ATOMIC_TOPICS as SUPPLEMENT_MIN_VALID_ATOMIC_TOPICS,
    POPULATION as SUPPLEMENT_POPULATION,
    REQUIRED_CATEGORIES as SUPPLEMENT_REQUIRED_CATEGORIES,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ID = "chk4_decision_pre2009_train_balanced_v1_20260811"
DEFAULT_CORE_RELEASE = (
    REPO_ROOT / "dataset/processed/retrain_v2/"
    "chk4_decision_warmstart_grpo_core_v3_20260810"
)
DEFAULT_CORE_MANIFEST_SHA256 = (
    "8b05dce09bcb0a0b27ee240da1e5d730be93921ce19e650a3b3e8b2689adb893"
)
DEFAULT_SUPPLEMENT_TEACHER = (
    REPO_ROOT / "output/data/retrain_v2/chk4/decision_supplement_1993_2008_v1/"
    "teacher_targets/supplement"
)
DEFAULT_OUTPUT = REPO_ROOT / "dataset/processed/retrain_v2" / RELEASE_ID

RELEASE_SCHEMA = "chk4-decision-pre2009-train-balanced-release-v1"
HANDOFF_SCHEMA = "chk4-decision-pre2009-train-balanced-handoff-v1"
SCHEDULE_SCHEMA = "chk4-decision-pre2009-fixed-schedule-row-v1"
SOURCE_ROW_SCHEMA = "chk4-decision-pre2009-source-row-v1"
REPEAT_SCHEMA = "chk4-decision-pre2009-repeat-row-v1"
AUDIT_SCHEMA = "chk4-decision-pre2009-train-balanced-audit-v1"
INPUT_CONTRACT_SCHEMA = "chk4-target-decision-blind-input-contract-v2"
# Pin the completed provider contract used by this immutable release.  Do not
# follow a mutable producer default here: a later grounding revision must get a
# new release/materializer contract instead of silently changing replay rules.
SUPPLEMENT_SUMMARY_PROMPT_CONTRACT_FILE = "prompt_contract.grounding-v6.json"
SUPPLEMENT_PRIOR_SUMMARY_PROMPT_CONTRACT_FILE = "prompt_contract.grounding-v5.json"
SUPPLEMENT_BLIND_PROMPT_CONTRACT_FILE = "prompt_contract.qualitative-v2.json"
SUPPLEMENT_BLIND_WRAPPER_FILE = "run_chk4_supplement_blind_v2.py"
SUPPLEMENT_BLIND_WRAPPER = REPO_ROOT / "jobs/generation" / SUPPLEMENT_BLIND_WRAPPER_FILE
SUPPLEMENT_TEACHER_PROMPT_CONTRACT_FILE = "prompt_contract.qualitative-v2.json"
SUPPLEMENT_TEACHER_WRAPPER_FILE = "run_chk4_supplement_teacher_v2.py"
SUPPLEMENT_TEACHER_WRAPPER = (
    REPO_ROOT / "jobs/generation" / SUPPLEMENT_TEACHER_WRAPPER_FILE
)
SUPPLEMENT_WRAPPER_SNAPSHOT_DIR = "execution_wrappers"
SUPPLEMENT_QUALITATIVE_REASONING_CONTRACT = "qualitative_no_numeric_quantities_v2"
EXPECTED_SUMMARY_REVALIDATED_ROWS = 97
EXPECTED_SUMMARY_PROVIDER_ROWS = 12
SAMPLER_TYPE = "manifest_fixed_schedule_v2"
SFT_ROLE = "decision_sft_pre2009_balanced"
GRPO_ROLE = "decision_grpo_pre2009_balanced"

SPLITS = ("train", "validation", "test")
EVALUATION_SPLITS = ("validation", "test")
ROLES = ("decision_sft", "decision_grpo")
WINDOW_PATTERN = ("hold", "hike", "hold", "cut", "hold", "hike", "hold", "cut")
PER_WINDOW_COUNTS = {"hold": 4, "hike": 2, "cut": 2}
EFFECTIVE_BATCH_SIZE = 8
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
WORLD_SIZE = 1
SAMPLER_SEED = 20260811
MAX_SOURCE_REPEAT = 4
MIN_SUPPLEMENT_ROWS = SUPPLEMENT_PIPELINE_MIN_ADMITTED
SUPPLEMENT_ACTION_FLOORS = {
    "hold:0": 60,
    "hike:25": 18,
    "hike:50": 3,
    "cut:25": 8,
    "cut:50": 7,
    "cut:75": 2,
}
SUPPLEMENT_DATE_MIN = "1993-03-23"
SUPPLEMENT_DATE_MAX = "2008-12-31"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAMPLE_ID_RE = re.compile(r"dec-[0-9a-f]{24}")
_TRAINING_ROW_ID_RE = re.compile(r"row-[0-9a-f]{24}")
_SUMMARY_CACHE_PATH_RE = re.compile(
    r"cache/summaries/accepted/(?P<prefix>[0-9a-f]{2})/"
    r"(?P<cache_key>[0-9a-f]{64})\.json"
)
_BOUNDARY = "\n</think>\n"


class Pre2009ReleaseError(RuntimeError):
    """A parent, supplement, schedule, or immutable release is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Pre2009ReleaseError(message)


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
    _require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Pre2009ReleaseError(f"invalid {label}: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} root must be an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise Pre2009ReleaseError(f"cannot read {label}: {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_lines, 1):
        _require(line != "", f"{label}:{line_number}: blank line")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise Pre2009ReleaseError(
                f"{label}:{line_number}: invalid JSON: {exc}"
            ) from exc
        _require(isinstance(value, dict), f"{label}:{line_number}: row must be object")
        rows.append(value)
    return rows


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_bytes(
        path,
        (
            json.dumps(
                dict(value),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_bytes(
        path,
        b"".join((_canonical_json(dict(row)) + "\n").encode("utf-8") for row in rows),
    )


def _file_record(path: Path, *, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    record: dict[str, Any] = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if path.suffix == ".jsonl":
        record["rows"] = len(path.read_text(encoding="utf-8").splitlines())
    return record


def _file_records(root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), f"tree contains symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative not in {"release_manifest.json", "handoff.json"}:
                records[relative] = _file_record(path, root=root)
    return records


def _verify_file_table(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    files = manifest.get("files")
    _require(isinstance(files, dict) and files, "release file table missing")
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    _require(
        actual == set(files) | {"release_manifest.json", "handoff.json"},
        "release contains missing or unlisted files",
    )
    for relative, raw_record in files.items():
        _require(isinstance(relative, str), "non-string file table key")
        _require(isinstance(raw_record, dict), f"invalid file record: {relative}")
        expected_keys = {"path", "bytes", "sha256"}
        if relative.endswith(".jsonl"):
            expected_keys.add("rows")
        _require(
            set(raw_record) == expected_keys and raw_record.get("path") == relative,
            f"file descriptor schema drift: {relative}",
        )
        path = root / relative
        _require(path.is_file() and not path.is_symlink(), f"missing file: {relative}")
        _require(path.stat().st_size == raw_record["bytes"], f"size drift: {relative}")
        _require(_sha256_file(path) == raw_record["sha256"], f"hash drift: {relative}")
        if relative.endswith(".jsonl"):
            _require(
                len(path.read_text(encoding="utf-8").splitlines())
                == raw_record["rows"],
                f"row-count drift: {relative}",
            )
    return files


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _unseal_tree(root: Path) -> None:
    if not root.exists():
        return
    root.chmod(0o700)
    for path in sorted(root.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)


def _assert_sealed_tree(root: Path) -> None:
    _require((root.stat().st_mode & 0o777) == 0o555, "release root is mutable")
    for path in root.rglob("*"):
        _require(not path.is_symlink(), f"release contains symlink: {path}")
        expected = 0o555 if path.is_dir() else 0o444
        _require(
            (path.stat().st_mode & 0o777) == expected,
            f"release member is mutable: {path}",
        )


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(renameat2 is not None, "renameat2(RENAME_NOREPLACE) unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise Pre2009ReleaseError(
            f"immutable destination already exists: {destination}"
        )
    raise Pre2009ReleaseError(f"atomic publish failed: {os.strerror(error)}")


def _resolve_recorded_path(value: Any, *, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} path missing")
    raw = Path(value)
    candidate = raw if raw.is_absolute() else REPO_ROOT / raw
    _require(not candidate.is_symlink(), f"{label} must not be symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise Pre2009ReleaseError(f"{label} is missing: {candidate}") from exc
    return resolved


def _record_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _gold_text(direction: str, magnitude_bp: int) -> str:
    _require(direction in {"hold", "hike", "cut"}, "invalid direction")
    allowed = {0} if direction == "hold" else {25, 50, 75, 100}
    _require(
        not isinstance(magnitude_bp, bool)
        and isinstance(magnitude_bp, int)
        and magnitude_bp in allowed,
        "invalid magnitude_bp",
    )
    return _canonical_json({"direction": direction, "magnitude_bp": magnitude_bp})


def _parse_response(response: str, *, label: str) -> tuple[str, int]:
    _require(response.count(_BOUNDARY) == 1, f"{label}: reasoning boundary drift")
    reasoning, answer = response.split(_BOUNDARY, 1)
    _require(bool(reasoning.strip()), f"{label}: empty reasoning")
    try:
        value = json.loads(answer)
    except json.JSONDecodeError as exc:
        raise Pre2009ReleaseError(f"{label}: invalid decision JSON") from exc
    _require(isinstance(value, dict), f"{label}: decision must be object")
    _require(set(value) == {"direction", "magnitude_bp"}, f"{label}: decision schema")
    direction = value.get("direction")
    magnitude = value.get("magnitude_bp")
    _gold_text(direction, magnitude)
    return str(direction), int(magnitude)


def _verify_sealed_parent_files(root: Path, manifest: Mapping[str, Any]) -> None:
    _assert_sealed_tree(root)
    _verify_file_table(root, manifest)


def _verify_core_release(
    core: Path, *, expected_manifest_sha256: str
) -> dict[str, Any]:
    core = core.resolve()
    _require(core.is_dir() and not core.is_symlink(), f"core release missing: {core}")
    _require(
        _SHA256_RE.fullmatch(expected_manifest_sha256) is not None,
        "invalid expected core manifest SHA",
    )
    manifest_path = core / "release_manifest.json"
    _require(
        _sha256_file(manifest_path) == expected_manifest_sha256,
        "core manifest SHA drift",
    )
    manifest = _read_json(manifest_path, label="core release manifest")
    _require(
        manifest.get("schema_version") == "chk4-decision-training-release-v1"
        and manifest.get("release_id") == core.name
        and manifest.get("quality_status") == "passed"
        and manifest.get("immutable") is True
        and manifest.get("training_ready") is True
        and manifest.get("population_scope") == "core-only",
        "core release identity/readiness drift",
    )
    _verify_sealed_parent_files(core, manifest)
    unique_rows = _read_jsonl(
        core / "manifests/unique/train.jsonl", label="core unique train"
    )
    sft_rows = _read_jsonl(core / "decision_sft/train.jsonl", label="core SFT train")
    sources: list[dict[str, Any]] = []
    offset = 0
    for index, unique in enumerate(unique_rows):
        factor = unique.get("repeat_factor")
        _require(
            not isinstance(factor, bool) and isinstance(factor, int) and factor > 0,
            f"core unique:{index}: invalid repeat factor",
        )
        _require(offset + factor <= len(sft_rows), "core physical train underflow")
        group = sft_rows[offset : offset + factor]
        _require(
            all(row == group[0] for row in group), "core repeated SFT payload drift"
        )
        payload = group[0]
        _require(set(payload) == {"prompt", "response"}, "core SFT schema drift")
        prompt = payload.get("prompt")
        response = payload.get("response")
        _require(
            isinstance(prompt, str) and isinstance(response, str),
            "core SFT text type drift",
        )
        decision = _parse_response(response, label=f"core unique:{index}")
        _require(
            decision == (unique.get("direction"), unique.get("magnitude_bp")),
            f"core unique:{index}: response/label drift",
        )
        sample_id = unique.get("sample_id")
        _require(
            isinstance(sample_id, str)
            and _SAMPLE_ID_RE.fullmatch(sample_id) is not None,
            f"core unique:{index}: invalid sample_id",
        )
        for name, observed in (
            ("prompt_sha256", _sha256_text(prompt)),
            ("response_sha256", _sha256_text(response)),
            ("gold_sha256", _sha256_text(_gold_text(*decision))),
        ):
            _require(unique.get(name) == observed, f"core unique:{index}: {name} drift")
        sources.append(
            {
                "schema_version": SOURCE_ROW_SCHEMA,
                "sample_id": sample_id,
                "meeting_date": unique.get("meeting_date"),
                "population": unique.get("population"),
                "population_role": "core",
                "source_ids": unique.get("source_ids"),
                "direction": decision[0],
                "magnitude_bp": decision[1],
                "prompt": prompt,
                "response": response,
                "prompt_sha256": unique["prompt_sha256"],
                "response_sha256": unique["response_sha256"],
                "gold_sha256": unique["gold_sha256"],
            }
        )
        offset += factor
    _require(offset == len(sft_rows), "core physical train overflow")
    _validate_source_rows(sources, label="core")
    return {
        "root": core,
        "manifest": manifest,
        "manifest_sha256": expected_manifest_sha256,
        "sources": sources,
        "unique_train_sha256": _sha256_file(core / "manifests/unique/train.jsonl"),
        "input_contract_sha256": _sha256_file(
            core / "contracts/decision_input_contract.json"
        ),
    }


def _teacher_required_files(root: Path) -> tuple[str, ...]:
    _supplement_pipeline_root(root)
    return (
        "summary.json",
        "provider_identity.json",
        "failures.jsonl",
        "prepared/summary_requests.jsonl",
        "manifests/admitted.jsonl",
        "reports/evidence_audit.json",
        "reports/blind_prediction_metrics.json",
        "reports/release_qa.json",
        "sources/materialized/source_handoff.json",
        f"summaries/{SUPPLEMENT_SUMMARY_PROMPT_CONTRACT_FILE}",
        f"summaries/{SUPPLEMENT_PRIOR_SUMMARY_PROMPT_CONTRACT_FILE}",
        "summaries/summary.json",
        "summaries/failures.jsonl",
        "summaries/meeting_decision_briefs.jsonl",
        "summaries/teacher_responses.jsonl",
        f"blind_predictions/{SUPPLEMENT_BLIND_PROMPT_CONTRACT_FILE}",
        "blind_predictions/summary.json",
        "blind_predictions/failures.jsonl",
        "blind_predictions/predictions.jsonl",
        "blind_predictions/teacher_responses.jsonl",
        "training/combined_v1/summary.json",
        f"teacher_targets/supplement/{SUPPLEMENT_TEACHER_PROMPT_CONTRACT_FILE}",
        "teacher_targets/supplement/summary.json",
        "teacher_targets/supplement/failures.jsonl",
        "teacher_targets/supplement/prepared/train.jsonl",
        "teacher_targets/supplement/prepared/validation.jsonl",
        "teacher_targets/supplement/prepared/test.jsonl",
        "teacher_targets/supplement/manifests/train.jsonl",
        "teacher_targets/supplement/manifests/validation.jsonl",
        "teacher_targets/supplement/manifests/test.jsonl",
        "teacher_targets/supplement/sft/train.jsonl",
        "teacher_targets/supplement/sft/validation.jsonl",
        "teacher_targets/supplement/sft/test.jsonl",
        "teacher_targets/supplement/teacher_responses/train.jsonl",
        "teacher_targets/supplement/teacher_responses/validation.jsonl",
        "teacher_targets/supplement/teacher_responses/test.jsonl",
    )


def _summary_revalidation_source_paths(root: Path) -> tuple[str, ...]:
    """Resolve only the bounded legacy-cache paths named by summary provenance."""

    pipeline = _supplement_pipeline_root(root)
    rows = _read_jsonl(
        pipeline / "summaries/teacher_responses.jsonl",
        label="summary provider responses",
    )
    paths: set[str] = set()
    for index, row in enumerate(rows):
        provenance = row.get("revalidated_from")
        if provenance is None:
            continue
        _require(
            isinstance(provenance, dict)
            and set(provenance)
            == {"path", "sha256", "contract_path", "contract_sha256"},
            f"summary:{index}: invalid revalidated_from schema",
        )
        relative = provenance.get("path")
        contract_relative = provenance.get("contract_path")
        match = (
            _SUMMARY_CACHE_PATH_RE.fullmatch(relative)
            if isinstance(relative, str)
            else None
        )
        _require(
            match is not None
            and match.group("prefix") == match.group("cache_key")[:2]
            and contract_relative
            == f"summaries/{SUPPLEMENT_PRIOR_SUMMARY_PROMPT_CONTRACT_FILE}",
            f"summary:{index}: unsafe revalidated_from path",
        )
        _require(relative not in paths, f"summary:{index}: reused legacy cache path")
        paths.add(relative)
    return tuple(sorted(paths))


def _supplement_pipeline_root(teacher_root: Path) -> Path:
    root = teacher_root.resolve()
    _require(
        root.name == "supplement" and root.parent.name == "teacher_targets",
        "supplement teacher must be teacher_targets/supplement",
    )
    pipeline = root.parents[1]
    _require(
        pipeline.is_dir() and not pipeline.is_symlink(),
        f"supplement pipeline root missing: {pipeline}",
    )
    return pipeline


def _teacher_bundle(root: Path) -> dict[str, dict[str, Any]]:
    pipeline = _supplement_pipeline_root(root)
    records: dict[str, dict[str, Any]] = {}
    relatives = (
        *_teacher_required_files(root),
        *_summary_revalidation_source_paths(root),
    )
    _require(len(relatives) == len(set(relatives)), "duplicate teacher bundle path")
    for relative in relatives:
        path = pipeline / relative
        _require(
            path.is_file() and not path.is_symlink(), f"missing teacher file: {path}"
        )
        record: dict[str, Any] = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        if relative.endswith(".jsonl"):
            record["rows"] = len(path.read_text(encoding="utf-8").splitlines())
        records[relative] = record
    return records


def _bundle_sha256(records: Mapping[str, Any]) -> str:
    return _sha256_text(_canonical_json(records))


def _rows_by_sample_id(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        _require(
            isinstance(sample_id, str)
            and _SAMPLE_ID_RE.fullmatch(sample_id) is not None
            and sample_id not in result,
            f"{label}:{index}: duplicate/invalid sample_id",
        )
        result[sample_id] = row
    return result


def _verify_prompt_contract(
    path: Path, *, stage: str, repair_attempts: int = 1
) -> tuple[dict[str, Any], str]:
    contract = _read_json(path, label=f"{stage} prompt contract")
    unsigned = dict(contract)
    contract_sha = unsigned.pop("contract_sha256", None)
    _require(
        contract.get("schema_version") == SUPPLEMENT_PROVIDER_CONTRACT_SCHEMA
        and contract.get("stage") == stage
        and isinstance(contract.get("system_prompt"), str)
        and bool(contract["system_prompt"])
        and isinstance(contract.get("repair_system_prompt"), str)
        and bool(contract["repair_system_prompt"])
        and isinstance(contract.get("provider"), dict)
        and contract.get("model_fallback") == "forbidden"
        and contract.get("repair_attempts") == repair_attempts
        and _SHA256_RE.fullmatch(str(contract.get("code_sha256") or "")) is not None
        and contract_sha == _sha256_text(_canonical_json(unsigned)),
        f"{stage} prompt contract drift",
    )
    return contract, str(contract_sha)


def _summary_contracts_are_revalidation_compatible(
    prior: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    """Replay the producer's bounded v5-to-v6 compatibility decision."""

    prior_payload = dict(prior)
    prior_digest = prior_payload.pop("contract_sha256", None)
    if prior_digest != _sha256_text(_canonical_json(prior_payload)):
        return False
    current_payload = dict(current)
    current_digest = current_payload.pop("contract_sha256", None)
    if current_digest != _sha256_text(_canonical_json(current_payload)):
        return False
    prior_repairs = prior_payload.pop("repair_attempts", None)
    current_repairs = current_payload.pop("repair_attempts", None)
    prior_payload.pop("code_sha256", None)
    current_payload.pop("code_sha256", None)
    return (
        prior_repairs == 1 and current_repairs == 2 and prior_payload == current_payload
    )


def _summary_cache_key(
    *,
    sample_id: str,
    input_sha256: str,
    prompt_sha256: str,
    contract_sha256: str,
) -> str:
    return _sha256_text(
        _canonical_json(
            {
                "schema_version": SUPPLEMENT_PROVIDER_CACHE_SCHEMA,
                "stage": "summaries",
                "sample_id": sample_id,
                "input_sha256": input_sha256,
                "prompt_sha256": prompt_sha256,
                "gold": None,
                "contract_sha256": contract_sha256,
            }
        )
    )


def _verify_summary_revalidation(
    *,
    pipeline: Path,
    rows: Sequence[Mapping[str, Any]],
    request_by_id: Mapping[str, Mapping[str, Any]],
    briefs_by_id: Mapping[str, Mapping[str, Any]],
    prior_contract: Mapping[str, Any],
    prior_contract_sha256: str,
    current_contract: Mapping[str, Any],
) -> int:
    _require(
        _summary_contracts_are_revalidation_compatible(
            prior_contract, current_contract
        ),
        "summary v5/v6 revalidation contract incompatibility",
    )
    revalidated = 0
    source_paths: set[str] = set()
    for index, row in enumerate(rows):
        provenance = row.get("revalidated_from")
        if provenance is None:
            continue
        _require(
            isinstance(provenance, dict)
            and set(provenance)
            == {"path", "sha256", "contract_path", "contract_sha256"},
            f"summary:{index}: invalid revalidated_from schema",
        )
        sample_id = str(row.get("sample_id") or "")
        request = request_by_id.get(sample_id)
        brief = briefs_by_id.get(sample_id)
        _require(
            isinstance(request, Mapping) and isinstance(brief, Mapping),
            f"summary:{index}: revalidation sample is not admitted",
        )
        expected_key = _summary_cache_key(
            sample_id=sample_id,
            input_sha256=str(request.get("input_sha256") or ""),
            prompt_sha256=str(request.get("prompt_sha256") or ""),
            contract_sha256=prior_contract_sha256,
        )
        expected_relative = (
            f"cache/summaries/accepted/{expected_key[:2]}/{expected_key}.json"
        )
        relative = provenance.get("path")
        _require(
            relative == expected_relative
            and relative not in source_paths
            and provenance.get("contract_path")
            == f"summaries/{SUPPLEMENT_PRIOR_SUMMARY_PROMPT_CONTRACT_FILE}"
            and provenance.get("contract_sha256") == prior_contract_sha256
            and _SHA256_RE.fullmatch(str(provenance.get("sha256") or "")) is not None,
            f"summary:{index}: revalidation binding drift",
        )
        source_paths.add(str(relative))
        cache_path = pipeline / expected_relative
        _require(
            provenance.get("sha256") == _sha256_file(cache_path),
            f"summary:{index}: legacy cache SHA drift",
        )
        cache = _read_json(cache_path, label=f"summary:{sample_id} legacy cache")
        expected_target = {
            key: brief.get(key)
            for key in (
                "meeting_decision_brief",
                "brief_sha256",
                "brief_tokens",
                "student_prompt_tokens",
            )
        }
        _require(
            cache.get("schema_version") == SUPPLEMENT_PROVIDER_CACHE_SCHEMA
            and cache.get("status") == "accepted"
            and cache.get("stage") == "summaries"
            and cache.get("cache_key") == expected_key
            and cache.get("sample_id") == sample_id
            and cache.get("input_sha256") == request.get("input_sha256")
            and cache.get("prompt_sha256") == request.get("prompt_sha256")
            and cache.get("gold_sha256") is None
            and cache.get("contract_sha256") == prior_contract_sha256
            and cache.get("attempt") in {"primary", "repair"}
            and cache.get("attempt") == row.get("attempt")
            and cache.get("provider") == row.get("provider")
            and cache.get("provider_raw")
            == {
                "reasoning_content": row.get("reasoning_content"),
                "content": row.get("content"),
            }
            and cache.get("target") == expected_target,
            f"summary:{index}: legacy cache replay drift",
        )
        revalidated += 1
    return revalidated


def _provider_identity_tuple(value: Any, *, label: str) -> tuple[str, str]:
    _require(
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, str) and item for item in value),
        f"{label} is missing",
    )
    return str(value[0]), str(value[1])


def _verify_stage_summary(
    summary: Mapping[str, Any],
    *,
    stage: str,
    contract_sha256: str,
    expected_rows: int,
    identity: tuple[str, str],
    prompt_contract_file: str = "prompt_contract.json",
) -> None:
    _require(
        summary.get("schema_version") == f"chk4-decision-supplement-{stage}-summary-v1"
        and summary.get("status") == "complete"
        and summary.get("stage") == stage
        and summary.get("prepared_count") == expected_rows
        and summary.get("accepted_count") == expected_rows
        and summary.get("failure_count") == 0
        and summary.get("contract_sha256") == contract_sha256
        and summary.get("prompt_contract_file") == prompt_contract_file
        and _provider_identity_tuple(
            summary.get("provider_identity"), label=f"{stage} provider identity"
        )
        == identity,
        f"{stage} completion summary drift",
    )


def _verify_execution_wrapper(
    summary: Mapping[str, Any],
    *,
    stage: str,
    expected_source: Path,
    snapshot_root: Path | None,
) -> dict[str, Any]:
    binding = summary.get("execution_wrapper")
    _require(
        isinstance(binding, dict)
        and set(binding) == {"path", "sha256"}
        and isinstance(binding.get("path"), str)
        and Path(binding["path"]).is_absolute()
        and Path(binding["path"]).resolve() == expected_source.resolve()
        and _SHA256_RE.fullmatch(str(binding.get("sha256") or "")) is not None,
        f"{stage} execution wrapper binding drift",
    )
    verified_path = (
        snapshot_root / expected_source.name
        if snapshot_root is not None
        else expected_source
    )
    _require(
        verified_path.is_file()
        and not verified_path.is_symlink()
        and _sha256_file(verified_path) == binding.get("sha256"),
        f"{stage} execution wrapper SHA drift",
    )
    return {
        "source_path": str(binding["path"]),
        "sha256": str(binding["sha256"]),
        "verified_path": verified_path,
    }


def _verify_stage_evidence_binding(
    summary: Mapping[str, Any],
    *,
    pipeline: Path,
    admitted_rows: int,
    source_handoff_payload_sha256: str,
    label: str,
) -> None:
    evidence = summary.get("evidence_audit")
    handoff = summary.get("source_handoff")
    admitted = summary.get("admitted_manifest")
    _require(
        summary.get("admission_profile") == SUPPLEMENT_ADMISSION_PROFILE
        and isinstance(evidence, dict)
        and isinstance(evidence.get("path"), str)
        and bool(evidence["path"])
        and evidence.get("sha256")
        == _sha256_file(pipeline / "reports/evidence_audit.json")
        and evidence.get("schema_version")
        == "chk4-decision-supplement-evidence-audit-v2"
        and isinstance(handoff, dict)
        and isinstance(handoff.get("path"), str)
        and bool(handoff["path"])
        and handoff.get("sha256")
        == _sha256_file(pipeline / "sources/materialized/source_handoff.json")
        and handoff.get("payload_sha256") == source_handoff_payload_sha256
        and isinstance(admitted, dict)
        and isinstance(admitted.get("path"), str)
        and bool(admitted["path"])
        and admitted.get("sha256")
        == _sha256_file(pipeline / "manifests/admitted.jsonl")
        and admitted.get("rows") == admitted_rows,
        f"{label} evidence binding drift",
    )


def _verify_provider_rows(
    rows_by_stage: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_ids: set[str],
    identity: tuple[str, str],
    repair_attempts_by_stage: Mapping[str, int],
) -> int:
    response_ids: set[str] = set()
    for stage, rows in rows_by_stage.items():
        repair_attempts = repair_attempts_by_stage.get(stage)
        _require(
            isinstance(repair_attempts, int) and repair_attempts >= 0,
            f"{stage}: missing repair-attempt contract",
        )
        allowed_attempts = {"primary"} | {
            "repair" if index == 1 else f"repair_{index}"
            for index in range(1, repair_attempts + 1)
        }
        indexed = _rows_by_sample_id(rows, label=f"{stage} teacher responses")
        _require(set(indexed) == expected_ids, f"{stage} provider row closure drift")
        for sample_id, row in indexed.items():
            provider = row.get("provider")
            _require(
                isinstance(provider, dict), f"{stage}:{sample_id}: provider missing"
            )
            response_id = provider.get("response_id")
            observed_identity = (
                provider.get("returned_model"),
                provider.get("system_fingerprint"),
            )
            _require(
                isinstance(response_id, str)
                and bool(response_id)
                and response_id not in response_ids
                and observed_identity == identity
                and provider.get("finish_reason") == "stop"
                and row.get("attempt") in allowed_attempts
                and isinstance(row.get("reasoning_content"), str)
                and isinstance(row.get("content"), str),
                f"{stage}:{sample_id}: provider provenance drift",
            )
            response_ids.add(response_id)
    return len(response_ids)


def _verify_supplement_teacher(
    root: Path, *, wrapper_snapshot_root: Path | None = None
) -> dict[str, Any]:
    root = root.resolve()
    _require(
        root.is_dir() and not root.is_symlink(), f"supplement teacher missing: {root}"
    )
    pipeline = _supplement_pipeline_root(root)
    bundle = _teacher_bundle(root)
    provider_identity = _read_json(
        pipeline / "provider_identity.json", label="supplement provider identity"
    )
    identity = (
        provider_identity.get("returned_model"),
        provider_identity.get("system_fingerprint"),
    )
    _require(
        all(isinstance(value, str) and value for value in identity),
        "supplement provider identity is incomplete",
    )
    identity = (str(identity[0]), str(identity[1]))

    summary_contract, summary_contract_sha = _verify_prompt_contract(
        pipeline / "summaries" / SUPPLEMENT_SUMMARY_PROMPT_CONTRACT_FILE,
        stage="summaries",
        repair_attempts=2,
    )
    prior_summary_contract, prior_summary_contract_sha = _verify_prompt_contract(
        pipeline / "summaries" / SUPPLEMENT_PRIOR_SUMMARY_PROMPT_CONTRACT_FILE,
        stage="summaries",
    )
    _require(
        _summary_contracts_are_revalidation_compatible(
            prior_summary_contract, summary_contract
        ),
        "summary v5/v6 revalidation contract incompatibility",
    )
    blind_contract, blind_contract_sha = _verify_prompt_contract(
        pipeline / "blind_predictions" / SUPPLEMENT_BLIND_PROMPT_CONTRACT_FILE,
        stage="blind_predictions",
        repair_attempts=2,
    )
    del blind_contract
    target_contract, contract_sha = _verify_prompt_contract(
        root / SUPPLEMENT_TEACHER_PROMPT_CONTRACT_FILE,
        stage="teacher_targets",
        repair_attempts=2,
    )
    del target_contract

    admitted_rows = _read_jsonl(
        pipeline / "manifests/admitted.jsonl", label="pipeline admitted rows"
    )
    admitted_by_id = _rows_by_sample_id(admitted_rows, label="pipeline admitted")
    _require(
        len(admitted_rows) == MIN_SUPPLEMENT_ROWS == SUPPLEMENT_EXPECTED_CANDIDATES,
        "completed supplement pipeline must admit every candidate",
    )
    admitted_ids = set(admitted_by_id)

    source_handoff = _read_json(
        pipeline / "sources/materialized/source_handoff.json",
        label="supplement source handoff",
    )
    unsigned_handoff = dict(source_handoff)
    handoff_sha = unsigned_handoff.pop("payload_sha256", None)
    ledgers = source_handoff.get("ledgers")
    _require(
        source_handoff.get("schema_version") == "chk1-source-handoff-v1"
        and handoff_sha == _sha256_text(_canonical_json(unsigned_handoff))
        and isinstance(ledgers, list)
        and bool(ledgers),
        "supplement source handoff integrity drift",
    )
    population_ids: set[str] = set()
    meeting_total = 0
    for index, record in enumerate(ledgers):
        _require(isinstance(record, dict), f"source handoff ledger:{index}: schema")
        population_id = record.get("population_id")
        meeting_count = record.get("meeting_count")
        _require(
            isinstance(population_id, str)
            and bool(population_id)
            and population_id not in population_ids
            and record.get("split") == "train"
            and record.get("coverage_mode") == "sparse"
            and not isinstance(meeting_count, bool)
            and isinstance(meeting_count, int)
            and meeting_count > 0,
            f"source handoff ledger:{index}: sparse train contract drift",
        )
        population_ids.add(population_id)
        meeting_total += meeting_count
    _require(
        meeting_total == len(admitted_rows),
        "source handoff/admitted meeting closure drift",
    )

    action_counts: Counter[str] = Counter()
    for sample_id, admitted in admitted_by_id.items():
        meeting_date = admitted.get("meeting_date")
        gold = admitted.get("gold")
        source_ids = admitted.get("source_ids")
        coverage = set(admitted.get("category_coverage") or ())
        _require(
            admitted.get("status") == "admitted"
            and admitted.get("split") == "train"
            and admitted.get("population") == SUPPLEMENT_POPULATION
            and admitted.get("population_role") == "supplement"
            and admitted.get("admission_profile") == SUPPLEMENT_ADMISSION_PROFILE
            and isinstance(admitted.get("valid_atomic_topic_count"), int)
            and admitted.get("valid_atomic_topic_count")
            >= SUPPLEMENT_MIN_VALID_ATOMIC_TOPICS
            and isinstance(meeting_date, str)
            and SUPPLEMENT_DATE_MIN <= meeting_date <= SUPPLEMENT_DATE_MAX
            and isinstance(gold, dict)
            and set(gold) == {"direction", "magnitude_bp"}
            and admitted.get("gold_sha256")
            == _sha256_text(_gold_text(gold["direction"], gold["magnitude_bp"]))
            and coverage >= set(SUPPLEMENT_REQUIRED_CATEGORIES)
            and isinstance(source_ids, list)
            and bool(source_ids)
            and len(source_ids) == len(set(source_ids)),
            f"pipeline admitted:{sample_id}: contract drift",
        )
        for source_id in source_ids:
            parts = str(source_id).split(":", 3)
            _require(
                len(parts) == 4
                and parts[0] == "canonical-loo-d1"
                and parts[1] in population_ids
                and parts[2] == meeting_date
                and bool(parts[3]),
                f"pipeline admitted:{sample_id}: source handoff reference drift",
            )
        action_counts[f"{gold['direction']}:{gold['magnitude_bp']}"] += 1

    evidence = _read_json(
        pipeline / "reports/evidence_audit.json", label="supplement evidence audit"
    )
    _require(
        evidence.get("schema_version") == "chk4-decision-supplement-evidence-audit-v2"
        and evidence.get("status") == "complete"
        and evidence.get("admission_profile") == SUPPLEMENT_ADMISSION_PROFILE
        and evidence.get("candidate_count") == SUPPLEMENT_EXPECTED_CANDIDATES
        and evidence.get("admitted_count") == len(admitted_rows)
        and evidence.get("rejected_count") == 0
        and evidence.get("minimum_admitted") == MIN_SUPPLEMENT_ROWS
        and evidence.get("minimum_valid_atomic_topics")
        == SUPPLEMENT_MIN_VALID_ATOMIC_TOPICS
        and set(evidence.get("required_categories") or ())
        == set(SUPPLEMENT_REQUIRED_CATEGORIES)
        and set(evidence.get("action_classes") or ()) == set(action_counts)
        and evidence.get("source_handoff_payload_sha256") == handoff_sha
        and evidence.get("forbidden_current_vintage_inputs") is True
        and evidence.get("meeting_identity_removed_from_model_input") is True,
        "supplement evidence audit replay drift",
    )

    summary_requests = _read_jsonl(
        pipeline / "prepared/summary_requests.jsonl",
        label="supplement summary requests",
    )
    summary_request_by_id = _rows_by_sample_id(
        summary_requests, label="supplement summary requests"
    )
    briefs = _read_jsonl(
        pipeline / "summaries/meeting_decision_briefs.jsonl",
        label="supplement meeting briefs",
    )
    briefs_by_id = _rows_by_sample_id(briefs, label="supplement meeting briefs")
    _require(
        set(summary_request_by_id) == admitted_ids == set(briefs_by_id),
        "summary/admitted/brief row closure drift",
    )
    for sample_id in sorted(admitted_ids):
        admitted = admitted_by_id[sample_id]
        request = summary_request_by_id[sample_id]
        brief_row = briefs_by_id[sample_id]
        prompt = request.get("prompt")
        brief = brief_row.get("meeting_decision_brief")
        _require(
            isinstance(prompt, str)
            and request.get("input_sha256") == admitted.get("input_sha256")
            and request.get("prompt_sha256") == admitted.get("prompt_sha256")
            and request.get("prompt_sha256") == _sha256_text(prompt)
            and brief_row.get("schema_version") == SUPPLEMENT_SUMMARY_SCHEMA
            and isinstance(brief, str)
            and bool(brief)
            and brief_row.get("brief_sha256") == _sha256_text(brief)
            and brief_row.get("input_sha256") == admitted.get("input_sha256")
            and brief_row.get("generation_contract_sha256") == summary_contract_sha,
            f"supplement brief:{sample_id}: replay drift",
        )

    summary = _read_json(root / "summary.json", label="supplement teacher summary")
    _verify_stage_summary(
        summary,
        stage="teacher_targets",
        contract_sha256=contract_sha,
        expected_rows=len(admitted_rows),
        identity=identity,
        prompt_contract_file=SUPPLEMENT_TEACHER_PROMPT_CONTRACT_FILE,
    )
    _require(
        summary.get("reasoning_contract") == SUPPLEMENT_QUALITATIVE_REASONING_CONTRACT
        and summary.get("teacher_output_mapping")
        == {
            "reasoning": "json.loads(message.content)['reasoning']",
            "decision": "locally_serialized_canonical_gold",
            "native_reasoning_content": "provenance_only",
        },
        "supplement teacher output mapping drift",
    )
    teacher_wrapper = _verify_execution_wrapper(
        summary,
        stage="teacher_targets",
        expected_source=SUPPLEMENT_TEACHER_WRAPPER,
        snapshot_root=wrapper_snapshot_root,
    )
    summary_stage = _read_json(
        pipeline / "summaries/summary.json", label="supplement summary stage"
    )
    blind_stage = _read_json(
        pipeline / "blind_predictions/summary.json", label="supplement blind stage"
    )
    _verify_stage_summary(
        summary_stage,
        stage="summaries",
        contract_sha256=summary_contract_sha,
        expected_rows=len(admitted_rows),
        identity=identity,
        prompt_contract_file=SUPPLEMENT_SUMMARY_PROMPT_CONTRACT_FILE,
    )
    _require(
        summary_stage.get("resumed_count") == EXPECTED_SUMMARY_REVALIDATED_ROWS
        and summary_stage.get("revalidated_count") == EXPECTED_SUMMARY_REVALIDATED_ROWS
        and summary_stage.get("api_requests") == EXPECTED_SUMMARY_PROVIDER_ROWS
        and EXPECTED_SUMMARY_REVALIDATED_ROWS + EXPECTED_SUMMARY_PROVIDER_ROWS
        == len(admitted_rows),
        "summary v6 revalidation/provider counts drift",
    )
    _verify_stage_summary(
        blind_stage,
        stage="blind_predictions",
        contract_sha256=blind_contract_sha,
        expected_rows=len(admitted_rows),
        identity=identity,
        prompt_contract_file=SUPPLEMENT_BLIND_PROMPT_CONTRACT_FILE,
    )
    _require(
        blind_stage.get("reasoning_contract")
        == SUPPLEMENT_QUALITATIVE_REASONING_CONTRACT,
        "blind_predictions qualitative reasoning contract drift",
    )
    blind_wrapper = _verify_execution_wrapper(
        blind_stage,
        stage="blind_predictions",
        expected_source=SUPPLEMENT_BLIND_WRAPPER,
        snapshot_root=wrapper_snapshot_root,
    )
    for label, stage_summary in (
        ("summaries", summary_stage),
        ("blind_predictions", blind_stage),
        ("teacher_targets", summary),
    ):
        _verify_stage_evidence_binding(
            stage_summary,
            pipeline=pipeline,
            admitted_rows=len(admitted_rows),
            source_handoff_payload_sha256=str(handoff_sha),
            label=label,
        )

    for relative in (
        "failures.jsonl",
        "summaries/failures.jsonl",
        "blind_predictions/failures.jsonl",
        "teacher_targets/supplement/failures.jsonl",
    ):
        _require(
            not _read_jsonl(pipeline / relative, label=relative),
            f"completed supplement contains failures: {relative}",
        )
    for split in ("validation", "test"):
        for family in ("prepared", "manifests", "sft", "teacher_responses"):
            _require(
                not _read_jsonl(
                    root / family / f"{split}.jsonl",
                    label=f"supplement {family}/{split}",
                ),
                "supplement leaked outside train",
            )

    prepared_rows = _read_jsonl(
        root / "prepared/train.jsonl", label="supplement prepared train"
    )
    manifest_rows = _read_jsonl(
        root / "manifests/train.jsonl", label="supplement manifest train"
    )
    sft_rows = _read_jsonl(root / "sft/train.jsonl", label="supplement SFT train")
    target_teacher_rows = _read_jsonl(
        root / "teacher_responses/train.jsonl",
        label="supplement teacher responses",
    )
    _require(
        len(prepared_rows)
        == len(manifest_rows)
        == len(sft_rows)
        == len(target_teacher_rows)
        == len(admitted_rows),
        "supplement teacher row-family count drift",
    )
    prepared_by_id = _rows_by_sample_id(prepared_rows, label="teacher prepared")
    manifests_by_id = _rows_by_sample_id(manifest_rows, label="teacher manifests")
    teacher_by_id = _rows_by_sample_id(target_teacher_rows, label="teacher responses")
    _require(
        set(prepared_by_id)
        == admitted_ids
        == set(manifests_by_id)
        == set(teacher_by_id),
        "supplement teacher row-family closure drift",
    )
    _require(
        [row["sample_id"] for row in prepared_rows]
        == [row["sample_id"] for row in manifest_rows]
        == [row["sample_id"] for row in target_teacher_rows],
        "supplement teacher row order drift",
    )
    for index, (prepared, manifest, training, teacher) in enumerate(
        zip(prepared_rows, manifest_rows, sft_rows, target_teacher_rows, strict=True)
    ):
        sample_id = str(prepared["sample_id"])
        admitted = admitted_by_id[sample_id]
        brief = str(briefs_by_id[sample_id]["meeting_decision_brief"])
        expected_prompt = render_student_prompt(brief)
        expected_gold = _gold_text(
            admitted["gold"]["direction"], admitted["gold"]["magnitude_bp"]
        )
        try:
            teacher_content = json.loads(str(teacher.get("content") or ""))
        except json.JSONDecodeError as exc:
            raise Pre2009ReleaseError(
                f"supplement:{index}: invalid teacher content"
            ) from exc
        _require(
            isinstance(teacher_content, dict)
            and set(teacher_content) == {"reasoning", "direction", "magnitude_bp"},
            f"supplement:{index}: teacher content schema drift",
        )
        reasoning = teacher_content.get("reasoning")
        _require(
            isinstance(reasoning, str)
            and bool(reasoning.strip())
            and reasoning == reasoning.strip()
            and not qualitative_reasoning_has_number_or_date(reasoning)
            and len(reasoning.split()) <= 180
            and (teacher_content.get("direction"), teacher_content.get("magnitude_bp"))
            == (admitted["gold"]["direction"], admitted["gold"]["magnitude_bp"]),
            f"supplement:{index}: teacher reasoning/decision drift",
        )
        expected_response = reasoning + _BOUNDARY + expected_gold
        _require(
            prepared.get("schema_version") == SUPPLEMENT_TARGET_SCHEMA
            and prepared.get("prompt") == expected_prompt
            and prepared.get("input_sha256") == _sha256_text(brief)
            and prepared.get("prompt_sha256") == _sha256_text(expected_prompt)
            and prepared.get("gold_sha256") == _sha256_text(expected_gold)
            and manifest.get("sample_id") == sample_id
            and manifest.get("meeting_date") == admitted.get("meeting_date")
            and manifest.get("split") == "train"
            and manifest.get("population") == SUPPLEMENT_POPULATION
            and manifest.get("population_role") == "supplement"
            and manifest.get("admission_profile") == SUPPLEMENT_ADMISSION_PROFILE
            and manifest.get("evidence_audit_sha256")
            == _sha256_file(pipeline / "reports/evidence_audit.json")
            and manifest.get("source_handoff_payload_sha256") == handoff_sha
            and manifest.get("source_ids") == admitted.get("source_ids")
            and manifest.get("gold") == admitted.get("gold")
            and manifest.get("input_sha256") == _sha256_text(brief)
            and manifest.get("prompt_sha256") == _sha256_text(expected_prompt)
            and manifest.get("gold_sha256") == _sha256_text(expected_gold)
            and manifest.get("contract_sha256") == contract_sha
            and training == {"prompt": expected_prompt, "response": expected_response},
            f"supplement:{index}: teacher-to-SFT reconstruction drift",
        )

    summary_provider_rows = _read_jsonl(
        pipeline / "summaries/teacher_responses.jsonl",
        label="summary provider responses",
    )
    blind_provider_rows = _read_jsonl(
        pipeline / "blind_predictions/teacher_responses.jsonl",
        label="blind provider responses",
    )
    revalidated_count = _verify_summary_revalidation(
        pipeline=pipeline,
        rows=summary_provider_rows,
        request_by_id=summary_request_by_id,
        briefs_by_id=briefs_by_id,
        prior_contract=prior_summary_contract,
        prior_contract_sha256=prior_summary_contract_sha,
        current_contract=summary_contract,
    )
    _require(
        revalidated_count == EXPECTED_SUMMARY_REVALIDATED_ROWS
        and len(summary_provider_rows) - revalidated_count
        == EXPECTED_SUMMARY_PROVIDER_ROWS,
        "summary response revalidation closure drift",
    )
    provider_response_count = _verify_provider_rows(
        {
            "summaries": summary_provider_rows,
            "blind_predictions": blind_provider_rows,
            "teacher_targets": target_teacher_rows,
        },
        expected_ids=admitted_ids,
        identity=identity,
        repair_attempts_by_stage={
            "summaries": 2,
            "blind_predictions": 2,
            "teacher_targets": 2,
        },
    )
    summary_response_by_id = _rows_by_sample_id(
        summary_provider_rows, label="summary provider responses"
    )
    for sample_id, row in summary_response_by_id.items():
        try:
            content = json.loads(str(row["content"]))
        except json.JSONDecodeError as exc:
            raise Pre2009ReleaseError(
                f"summary:{sample_id}: invalid provider content"
            ) from exc
        _require(
            content
            == {
                "meeting_decision_brief": briefs_by_id[sample_id][
                    "meeting_decision_brief"
                ]
            },
            f"summary:{sample_id}: provider-to-brief reconstruction drift",
        )

    blind_predictions = _read_jsonl(
        pipeline / "blind_predictions/predictions.jsonl",
        label="blind v2 predictions",
    )
    blind_prediction_by_id = _rows_by_sample_id(
        blind_predictions, label="blind v2 predictions"
    )
    blind_response_by_id = _rows_by_sample_id(
        blind_provider_rows, label="blind provider responses"
    )
    _require(
        set(blind_prediction_by_id) == admitted_ids == set(blind_response_by_id),
        "blind v2 prediction/response closure drift",
    )
    for sample_id in sorted(admitted_ids):
        response_row = blind_response_by_id[sample_id]
        prediction = blind_prediction_by_id[sample_id]
        try:
            content = json.loads(str(response_row.get("content") or ""))
        except json.JSONDecodeError as exc:
            raise Pre2009ReleaseError(
                f"blind:{sample_id}: invalid provider content"
            ) from exc
        _require(
            isinstance(content, dict)
            and set(content) == {"reasoning", "direction", "magnitude_bp"},
            f"blind:{sample_id}: provider content schema drift",
        )
        reasoning = content.get("reasoning")
        direction = content.get("direction")
        magnitude = content.get("magnitude_bp")
        try:
            _gold_text(direction, magnitude)
        except Pre2009ReleaseError as exc:
            raise Pre2009ReleaseError(
                f"blind:{sample_id}: invalid predicted decision"
            ) from exc
        token_fields = (
            prediction.get("prompt_tokens"),
            prediction.get("completion_tokens"),
            prediction.get("total_tokens"),
        )
        _require(
            isinstance(reasoning, str)
            and bool(reasoning.strip())
            and reasoning == reasoning.strip()
            and not qualitative_reasoning_has_number_or_date(reasoning)
            and len(reasoning.split()) <= 180
            and set(prediction)
            == {
                "schema_version",
                "sample_id",
                "reasoning",
                "direction",
                "magnitude_bp",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "analysis_sha256",
                "contract_sha256",
            }
            and prediction.get("schema_version") == SUPPLEMENT_BLIND_SCHEMA
            and (
                prediction.get("reasoning"),
                prediction.get("direction"),
                prediction.get("magnitude_bp"),
            )
            == (reasoning, direction, magnitude)
            and all(
                not isinstance(value, bool) and isinstance(value, int) and value >= 0
                for value in token_fields
            )
            and prediction.get("analysis_sha256")
            == _sha256_text(str(briefs_by_id[sample_id]["meeting_decision_brief"]))
            and prediction.get("contract_sha256") == blind_contract_sha,
            f"blind:{sample_id}: v2 provider-to-prediction replay drift",
        )

    blind_metrics = _read_json(
        pipeline / "reports/blind_prediction_metrics.json",
        label="supplement blind metrics",
    )
    training_summary = _read_json(
        pipeline / "training/combined_v1/summary.json",
        label="supplement combined training summary",
    )
    qa = _read_json(pipeline / "reports/release_qa.json", label="supplement QA")
    expected_unique_counts = {
        "train": 102 + len(admitted_rows),
        "validation": 13,
        "test": 13,
    }
    _require(
        blind_metrics.get("status") == "complete"
        and blind_metrics.get("sample_count") == len(admitted_rows)
        and blind_metrics.get("selection_policy")
        == "audit_only_never_filter_training_rows"
        and training_summary.get("unique_counts") == expected_unique_counts
        and qa.get("schema_version") == "chk4-decision-supplement-release-qa-v2"
        and qa.get("status") == "complete"
        and qa.get("admission_profile") == SUPPLEMENT_ADMISSION_PROFILE
        and qa.get("evidence_audit_sha256")
        == _sha256_file(pipeline / "reports/evidence_audit.json")
        and qa.get("source_handoff_payload_sha256") == handoff_sha
        and qa.get("errors") == []
        and qa.get("candidate_count") == SUPPLEMENT_EXPECTED_CANDIDATES
        and qa.get("admitted_count") == len(admitted_rows)
        and qa.get("rejected_count") == 0
        and set(qa.get("action_classes") or ()) == set(action_counts)
        and qa.get("blind_metrics_are_audit_only") is True
        and qa.get("combined_unique_counts") == expected_unique_counts
        and qa.get("combined_physical_counts")
        == training_summary.get("physical_counts")
        and qa.get("provider_response_count") == provider_response_count
        and _provider_identity_tuple(
            qa.get("provider_identity"), label="QA provider identity"
        )
        == identity,
        "supplement release QA replay drift",
    )
    pipeline_summary = _read_json(
        pipeline / "summary.json", label="supplement pipeline summary"
    )
    stages = pipeline_summary.get("stages")
    root_summary_stage = {
        key: value
        for key, value in summary_stage.items()
        if key
        not in {
            "admission_profile",
            "evidence_audit",
            "source_handoff",
            "admitted_manifest",
        }
    }
    _require(
        pipeline_summary.get("schema_version")
        == "chk4-decision-supplement-pipeline-summary-v1"
        and pipeline_summary.get("population") == SUPPLEMENT_POPULATION
        and pipeline_summary.get("status") == "complete"
        and isinstance(stages, dict)
        and stages.get("evidence") == evidence
        and stages.get("summaries") == root_summary_stage
        and stages.get("blind_predictions") == blind_stage
        and stages.get("teacher_targets") == summary
        and stages.get("qa") == qa,
        "supplement pipeline completion binding drift",
    )

    try:
        split_rows = load_unique_rows(root)
    except (MaterializationError, OSError, ValueError) as exc:
        raise Pre2009ReleaseError(f"invalid supplement teacher release: {exc}") from exc
    _require(
        not split_rows["validation"] and not split_rows["test"],
        "supplement leaked outside train",
    )
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(split_rows["train"]):
        meeting_date = source.get("meeting_date")
        _require(
            isinstance(meeting_date, str)
            and SUPPLEMENT_DATE_MIN <= meeting_date <= SUPPLEMENT_DATE_MAX,
            f"supplement:{index}: date outside 1993--2008",
        )
        _require(
            source.get("population_role") == "supplement",
            f"supplement:{index}: population role drift",
        )
        source_ids = source.get("source_ids")
        _require(
            isinstance(source_ids, list)
            and source_ids
            and len(source_ids) == len(set(source_ids))
            and all(isinstance(value, str) and value for value in source_ids),
            f"supplement:{index}: invalid source IDs",
        )
        reasoning = str(source["response"]).split(_BOUNDARY, 1)[0]
        _require(
            not qualitative_reasoning_has_number_or_date(reasoning),
            f"supplement:{index}: reasoning is not qualitative/date-free",
        )
        rows.append({"schema_version": SOURCE_ROW_SCHEMA, **source})
    _validate_source_rows(rows, label="supplement")
    _require(len(rows) >= MIN_SUPPLEMENT_ROWS, "supplement below minimum admission")
    actions = Counter(f"{row['direction']}:{row['magnitude_bp']}" for row in rows)
    for action, floor in SUPPLEMENT_ACTION_FLOORS.items():
        _require(
            actions[action] >= floor,
            f"supplement action floor failed: {action}={actions[action]} < {floor}",
        )
    accepted_count = summary.get("accepted_count")
    _require(
        not isinstance(accepted_count, bool)
        and isinstance(accepted_count, int)
        and accepted_count == len(rows),
        "supplement summary accepted_count drift",
    )
    return {
        "root": root,
        "pipeline_root": pipeline,
        "summary": summary,
        "summary_sha256": _sha256_file(root / "summary.json"),
        "contract_sha256": contract_sha,
        "bundle": bundle,
        "bundle_sha256": _bundle_sha256(bundle),
        "source_handoff_payload_sha256": handoff_sha,
        "release_qa_sha256": _sha256_file(pipeline / "reports/release_qa.json"),
        "provider_identity": list(identity),
        "execution_wrappers": {
            "blind_predictions": blind_wrapper,
            "teacher_targets": teacher_wrapper,
        },
        "sources": rows,
        "action_counts": dict(sorted(actions.items())),
    }


def _validate_source_rows(rows: Sequence[Mapping[str, Any]], *, label: str) -> None:
    sample_ids: set[str] = set()
    meeting_dates: set[str] = set()
    prompt_hashes: set[str] = set()
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id")
        meeting_date = row.get("meeting_date")
        prompt = row.get("prompt")
        response = row.get("response")
        _require(
            isinstance(sample_id, str)
            and _SAMPLE_ID_RE.fullmatch(sample_id) is not None
            and sample_id not in sample_ids,
            f"{label}:{index}: duplicate/invalid sample_id",
        )
        _require(
            isinstance(meeting_date, str)
            and meeting_date
            and meeting_date not in meeting_dates,
            f"{label}:{index}: duplicate/invalid meeting_date",
        )
        _require(
            isinstance(prompt, str) and isinstance(response, str),
            f"{label}:{index}: invalid payload",
        )
        decision = _parse_response(response, label=f"{label}:{index}")
        _require(
            decision == (row.get("direction"), row.get("magnitude_bp")),
            f"{label}:{index}: response/label drift",
        )
        prompt_sha = _sha256_text(prompt)
        _require(
            row.get("prompt_sha256") == prompt_sha and prompt_sha not in prompt_hashes,
            f"{label}:{index}: duplicate/prompt hash drift",
        )
        _require(
            row.get("response_sha256") == _sha256_text(response),
            f"{label}:{index}: response hash drift",
        )
        _require(
            row.get("gold_sha256") == _sha256_text(_gold_text(*decision)),
            f"{label}:{index}: gold hash drift",
        )
        sample_ids.add(sample_id)
        meeting_dates.add(meeting_date)
        prompt_hashes.add(prompt_sha)


def _validate_cross_population(
    core_rows: Sequence[Mapping[str, Any]], supplement_rows: Sequence[Mapping[str, Any]]
) -> None:
    for field in ("sample_id", "meeting_date", "prompt_sha256"):
        core_values = {row.get(field) for row in core_rows}
        supplement_values = {row.get(field) for row in supplement_rows}
        _require(
            not (core_values & supplement_values),
            f"core/supplement {field} overlap",
        )
    core_source_ids = {
        value
        for row in core_rows
        for value in (row.get("source_ids") or [])
        if isinstance(value, str)
    }
    supplement_source_ids = {
        value
        for row in supplement_rows
        for value in (row.get("source_ids") or [])
        if isinstance(value, str)
    }
    _require(
        not (core_source_ids & supplement_source_ids),
        "core/supplement source ID overlap",
    )


def _cycle_order(
    rows: Sequence[Mapping[str, Any]], *, direction: str, cycle: int
) -> list[str]:
    def score(row: Mapping[str, Any]) -> str:
        return _sha256_text(
            f"{SCHEDULE_SCHEMA}\0{SAMPLER_SEED}\0{direction}\0{cycle}\0"
            f"{row['sample_id']}"
        )

    return [str(row["sample_id"]) for row in sorted(rows, key=score)]


def _class_sequence(
    rows: Sequence[Mapping[str, Any]], *, direction: str, count: int, per_window: int
) -> list[str]:
    _require(len(rows) >= per_window, f"{direction}: insufficient unique rows")
    sequence: list[str] = []
    cycle = 0
    while len(sequence) < count:
        sequence.extend(_cycle_order(rows, direction=direction, cycle=cycle))
        cycle += 1
    del sequence[count:]
    for start in range(0, count, per_window):
        end = start + per_window
        for position in range(start, end):
            used = set(sequence[start:position])
            if sequence[position] not in used:
                continue
            replacement = next(
                (
                    candidate
                    for candidate in range(end, count)
                    if sequence[candidate] not in used
                    and sequence[candidate] not in sequence[position + 1 : end]
                ),
                None,
            )
            _require(
                replacement is not None, f"{direction}: cannot de-duplicate window"
            )
            sequence[position], sequence[replacement] = (
                sequence[replacement],
                sequence[position],
            )
        _require(
            len(set(sequence[start:end])) == per_window,
            f"{direction}: duplicate source within class window",
        )
    return sequence


def _training_row_id(
    *, role: str, source_sample_id: str, source_repeat_index: int, schedule_index: int
) -> str:
    payload = (
        f"{RELEASE_ID}\0{SCHEDULE_SCHEMA}\0{role}\0{source_sample_id}\0"
        f"{source_repeat_index}\0{schedule_index}"
    )
    return "row-" + _sha256_text(payload)[:24]


def build_schedule(
    unique_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the minimal complete dynamic 4/2/2 schedule from unique rows."""

    _validate_source_rows(unique_rows, label="combined unique")
    pools = {
        direction: [row for row in unique_rows if row.get("direction") == direction]
        for direction in PER_WINDOW_COUNTS
    }
    _require(all(pools.values()), "combined train is missing a direction")
    optimizer_steps = max(
        math.ceil(len(pools[direction]) / per_window)
        for direction, per_window in PER_WINDOW_COUNTS.items()
    )
    direction_counts = {
        direction: optimizer_steps * per_window
        for direction, per_window in PER_WINDOW_COUNTS.items()
    }
    sequences = {
        direction: _class_sequence(
            rows,
            direction=direction,
            count=direction_counts[direction],
            per_window=PER_WINDOW_COUNTS[direction],
        )
        for direction, rows in pools.items()
    }
    cursors: Counter[str] = Counter()
    ordered_ids: list[str] = []
    for _step in range(optimizer_steps):
        for direction in WINDOW_PATTERN:
            ordered_ids.append(sequences[direction][cursors[direction]])
            cursors[direction] += 1
    source_by_id = {str(row["sample_id"]): row for row in unique_rows}
    totals = Counter(ordered_ids)
    _require(
        set(totals) == set(source_by_id), "schedule does not cover every unique row"
    )
    _require(max(totals.values()) <= MAX_SOURCE_REPEAT, "schedule repeat cap exceeded")
    seen: Counter[str] = Counter()
    schedule: list[dict[str, Any]] = []
    for schedule_index, sample_id in enumerate(ordered_ids):
        source = source_by_id[sample_id]
        repeat_index = seen[sample_id]
        schedule.append(
            {
                "schema_version": SCHEDULE_SCHEMA,
                "schedule_index": schedule_index,
                "optimizer_step": schedule_index // EFFECTIVE_BATCH_SIZE,
                "microbatch_slot": schedule_index % EFFECTIVE_BATCH_SIZE,
                "training_row_id": _training_row_id(
                    role="decision_sft",
                    source_sample_id=sample_id,
                    source_repeat_index=repeat_index,
                    schedule_index=schedule_index,
                ),
                "source_sample_id": sample_id,
                "source_repeat_index": repeat_index,
                "source_repeat_total": totals[sample_id],
                "population_role": source["population_role"],
                "direction": source["direction"],
                "magnitude_bp": source["magnitude_bp"],
                "prompt_sha256": source["prompt_sha256"],
                "response_sha256": source["response_sha256"],
                "gold_sha256": source["gold_sha256"],
            }
        )
        seen[sample_id] += 1
    _validate_schedule(schedule, unique_rows=unique_rows)
    histograms: dict[str, Counter[int]] = {
        direction: Counter() for direction in PER_WINDOW_COUNTS
    }
    for sample_id, count in totals.items():
        histograms[str(source_by_id[sample_id]["direction"])][count] += 1
    contract = {
        "type": SAMPLER_TYPE,
        "seed": SAMPLER_SEED,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "world_size": WORLD_SIZE,
        "optimizer_steps": optimizer_steps,
        "schedule_rows": len(schedule),
        "train_rows": len(schedule),
        "shuffle_dataset": False,
        "direction_counts": direction_counts,
        "per_optimizer_window": PER_WINDOW_COUNTS,
        "repeat_histograms": {
            direction: {str(repeat): count for repeat, count in sorted(hist.items())}
            for direction, hist in histograms.items()
        },
        "max_source_repeat": max(totals.values()),
        "all_unique_sources_covered": True,
        "optimizer_windows_with_duplicate_source": 0,
        "source_population": "combined_unique_train_only",
        "oversampling_layers": 1,
        "order_is_authoritative": True,
        "required_sampler": "fixed_sequential_schedule_index_v2",
        "secondary_shuffle_forbidden": True,
    }
    return schedule, contract


def _validate_schedule(
    schedule: Sequence[Mapping[str, Any]], *, unique_rows: Sequence[Mapping[str, Any]]
) -> None:
    _require(
        len(schedule) > 0 and len(schedule) % EFFECTIVE_BATCH_SIZE == 0,
        "schedule size is not a positive optimizer-window multiple",
    )
    source_by_id = {str(row["sample_id"]): row for row in unique_rows}
    repeat_counts: Counter[str] = Counter()
    training_ids: set[str] = set()
    direction_counts: Counter[str] = Counter()
    for index, row in enumerate(schedule):
        expected_keys = {
            "schema_version",
            "schedule_index",
            "optimizer_step",
            "microbatch_slot",
            "training_row_id",
            "source_sample_id",
            "source_repeat_index",
            "source_repeat_total",
            "population_role",
            "direction",
            "magnitude_bp",
            "prompt_sha256",
            "response_sha256",
            "gold_sha256",
        }
        _require(set(row) == expected_keys, f"schedule:{index}: schema drift")
        _require(row.get("schema_version") == SCHEDULE_SCHEMA, "schedule schema drift")
        _require(row.get("schedule_index") == index, f"schedule:{index}: index drift")
        _require(
            row.get("optimizer_step") == index // EFFECTIVE_BATCH_SIZE
            and row.get("microbatch_slot") == index % EFFECTIVE_BATCH_SIZE,
            f"schedule:{index}: optimizer coordinates drift",
        )
        sample_id = row.get("source_sample_id")
        _require(
            isinstance(sample_id, str) and sample_id in source_by_id,
            f"schedule:{index}: unknown source",
        )
        source = source_by_id[sample_id]
        _require(
            row.get("source_repeat_index") == repeat_counts[sample_id],
            f"schedule:{index}: repeat index drift",
        )
        expected_id = _training_row_id(
            role="decision_sft",
            source_sample_id=sample_id,
            source_repeat_index=repeat_counts[sample_id],
            schedule_index=index,
        )
        training_id = row.get("training_row_id")
        _require(
            isinstance(training_id, str)
            and _TRAINING_ROW_ID_RE.fullmatch(training_id) is not None
            and training_id == expected_id
            and training_id not in training_ids,
            f"schedule:{index}: training row ID drift",
        )
        for field in (
            "population_role",
            "direction",
            "magnitude_bp",
            "prompt_sha256",
            "response_sha256",
            "gold_sha256",
        ):
            _require(
                row.get(field) == source.get(field),
                f"schedule:{index}: source {field} drift",
            )
        training_ids.add(training_id)
        repeat_counts[sample_id] += 1
        direction_counts[str(row["direction"])] += 1
    _require(set(repeat_counts) == set(source_by_id), "schedule unique coverage drift")
    _require(
        max(repeat_counts.values()) <= MAX_SOURCE_REPEAT, "schedule repeat cap drift"
    )
    for row in schedule:
        _require(
            row.get("source_repeat_total") == repeat_counts[row["source_sample_id"]],
            "schedule repeat total drift",
        )
    for start in range(0, len(schedule), EFFECTIVE_BATCH_SIZE):
        window = schedule[start : start + EFFECTIVE_BATCH_SIZE]
        _require(
            Counter(str(row["direction"]) for row in window) == PER_WINDOW_COUNTS,
            f"optimizer window {start // EFFECTIVE_BATCH_SIZE}: direction drift",
        )
        _require(
            len({str(row["source_sample_id"]) for row in window})
            == EFFECTIVE_BATCH_SIZE,
            f"optimizer window {start // EFFECTIVE_BATCH_SIZE}: duplicate source",
        )


def _materialize_train(
    schedule: Sequence[Mapping[str, Any]],
    *,
    source_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sft_rows: list[dict[str, Any]] = []
    grpo_rows: list[dict[str, Any]] = []
    repeats: list[dict[str, Any]] = []
    for row in schedule:
        sample_id = str(row["source_sample_id"])
        source = source_by_id[sample_id]
        repeat_index = int(row["source_repeat_index"])
        schedule_index = int(row["schedule_index"])
        sft_rows.append(
            {
                "prompt": source["prompt"],
                "response": source["response"],
                "training_row_id": row["training_row_id"],
                "source_sample_id": sample_id,
                "schedule_index": schedule_index,
            }
        )
        grpo_rows.append(
            {
                "sample_id": sample_id,
                "prompt": source["prompt"],
                "direction": source["direction"],
                "magnitude_bp": source["magnitude_bp"],
            }
        )
        for role in ROLES:
            repeats.append(
                {
                    "schema_version": REPEAT_SCHEMA,
                    "stage": role,
                    "training_row_id": _training_row_id(
                        role=role,
                        source_sample_id=sample_id,
                        source_repeat_index=repeat_index,
                        schedule_index=schedule_index,
                    ),
                    "source_sample_id": sample_id,
                    "split": "train",
                    "schedule_index": schedule_index,
                    "repeat_index": repeat_index,
                    "repeat_factor": row["source_repeat_total"],
                }
            )
    return sft_rows, grpo_rows, repeats


def _unique_manifest(
    sources: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    totals = Counter(str(row["source_sample_id"]) for row in schedule)
    output: list[dict[str, Any]] = []
    for source in sorted(sources, key=lambda row: str(row["sample_id"])):
        output.append(
            {
                key: source[key]
                for key in (
                    "sample_id",
                    "meeting_date",
                    "population",
                    "population_role",
                    "source_ids",
                    "direction",
                    "magnitude_bp",
                    "prompt_sha256",
                    "response_sha256",
                    "gold_sha256",
                )
            }
            | {"split": "train", "repeat_factor": totals[str(source["sample_id"])]}
        )
    return output


def _build_input_contract(
    *, core_info: Mapping[str, Any], supplement_info: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": INPUT_CONTRACT_SCHEMA,
        "status": "active",
        "population_scope": "core-plus-pre2009-train-only",
        "student_system_prompt": STUDENT_SYSTEM_PROMPT,
        "student_system_prompt_sha256": _sha256_text(STUDENT_SYSTEM_PROMPT),
        "target_blindness": "target-decision-blind",
        "split_policy": {
            "supplement_train_only": True,
            "validation_and_test_inherited_from_core": True,
        },
        "composed_contracts": {
            "core_decision_input_contract_sha256": core_info["input_contract_sha256"],
            "pre2009_teacher_contract_sha256": supplement_info["contract_sha256"],
        },
        "supplement_date_bounds": {
            "min": SUPPLEMENT_DATE_MIN,
            "max": SUPPLEMENT_DATE_MAX,
        },
        "forbidden_target_fields": [
            "sample_or_meeting_identity",
            "target_meeting_outcome",
            "gold_label",
        ],
    }
    return {**payload, "contract_sha256": _sha256_text(_canonical_json(payload))}


def _build_audit(
    *,
    core_info: Mapping[str, Any],
    supplement_info: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    sampler_contract: Mapping[str, Any],
) -> dict[str, Any]:
    unique_directions = Counter(str(row["direction"]) for row in sources)
    physical_directions = Counter(str(row["direction"]) for row in schedule)
    unique_populations = Counter(str(row["population_role"]) for row in sources)
    physical_populations = Counter(str(row["population_role"]) for row in schedule)
    return {
        "schema_version": AUDIT_SCHEMA,
        "status": "passed",
        "lineage_rows_checked": len(schedule),
        "unique_train_rows": len(sources),
        "supplement_train_rows": len(supplement_info["sources"]),
        "new_validation_rows": 0,
        "new_test_rows": 0,
        "cross_population_sample_id_overlaps": 0,
        "cross_population_meeting_date_overlaps": 0,
        "cross_population_prompt_overlaps": 0,
        "cross_population_source_id_overlaps": 0,
        "train_unique_direction_counts": dict(sorted(unique_directions.items())),
        "train_physical_direction_counts": dict(sorted(physical_directions.items())),
        "train_unique_population_counts": dict(sorted(unique_populations.items())),
        "train_physical_population_counts": dict(sorted(physical_populations.items())),
        "supplement_action_counts": supplement_info["action_counts"],
        "supplement_action_floors": SUPPLEMENT_ACTION_FLOORS,
        "supplement_date_bounds": {
            "min": min(row["meeting_date"] for row in supplement_info["sources"]),
            "max": max(row["meeting_date"] for row in supplement_info["sources"]),
        },
        "validation_byte_inherited": True,
        "test_byte_inherited": True,
        "test_is_sealed_evaluation_only": True,
        "unique_source_only_schedule": True,
        "oversampling_layers": 1,
        "max_source_repeat": sampler_contract["max_source_repeat"],
        "optimizer_windows_with_duplicate_source": 0,
        "core_manifest_sha256": core_info["manifest_sha256"],
        "supplement_teacher_bundle_sha256": supplement_info["bundle_sha256"],
    }


def _copy_teacher_bundle(source: Path, destination: Path) -> None:
    pipeline = _supplement_pipeline_root(source)
    for relative in _teacher_bundle(source):
        _write_bytes(destination / relative, (pipeline / relative).read_bytes())


def _copy_execution_wrappers(
    wrappers: Mapping[str, Mapping[str, Any]], destination: Path
) -> None:
    for stage, record in sorted(wrappers.items()):
        source = record.get("verified_path")
        _require(
            isinstance(source, Path)
            and source.is_file()
            and not source.is_symlink()
            and _sha256_file(source) == record.get("sha256"),
            f"{stage} execution wrapper changed before snapshot",
        )
        _write_bytes(destination / source.name, source.read_bytes())


def _public_execution_wrapper_records(
    wrappers: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    return {
        stage: {
            "source_path": str(record["source_path"]),
            "sha256": str(record["sha256"]),
            "snapshot_path": (
                "provenance/supplement_pipeline/"
                f"{SUPPLEMENT_WRAPPER_SNAPSHOT_DIR}/"
                f"{Path(record['verified_path']).name}"
            ),
        }
        for stage, record in sorted(wrappers.items())
    }


def _evaluation_records(core: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for role in ROLES:
        result[role] = {}
        for split in EVALUATION_SPLITS:
            path = core / role / f"{split}.jsonl"
            result[role][split] = {
                "source_path": _record_path(path),
                "bytes": path.stat().st_size,
                "rows": len(path.read_text(encoding="utf-8").splitlines()),
                "sha256": _sha256_file(path),
                "copied_byte_for_byte": True,
            }
    return result


def verify_release(
    root: Path,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Independently replay and verify a sealed augmented release."""

    root = root.resolve()
    _require(root.is_dir() and not root.is_symlink(), f"release missing: {root}")
    _assert_sealed_tree(root)
    manifest_path = root / "release_manifest.json"
    manifest_sha = _sha256_file(manifest_path)
    _require(
        isinstance(expected_manifest_sha256, str)
        and _SHA256_RE.fullmatch(expected_manifest_sha256) is not None,
        "invalid expected manifest SHA",
    )
    _require(manifest_sha == expected_manifest_sha256, "manifest SHA drift")
    manifest = _read_json(manifest_path, label="augmented release manifest")
    is_staging = root.name.startswith(f".{RELEASE_ID}.staging.")
    _require(
        manifest.get("schema_version") == RELEASE_SCHEMA
        and manifest.get("release_id") == RELEASE_ID
        and (root.name == RELEASE_ID or is_staging)
        and manifest.get("release_type") == "train_only_augmentation"
        and manifest.get("quality_status") == "passed"
        and manifest.get("immutable") is True
        and manifest.get("training_ready") is True
        and manifest.get("canonical_dag_bindable") is False
        and manifest.get("test_is_sealed_evaluation_only") is True,
        "augmented release identity/readiness drift",
    )
    files = _verify_file_table(root, manifest)

    parents = manifest.get("parent_releases")
    _require(isinstance(parents, dict), "parent release bindings missing")
    core_record = parents.get("core_v3")
    supplement_record = parents.get("pre2009_supplement_teacher")
    _require(isinstance(core_record, dict), "core parent binding missing")
    _require(isinstance(supplement_record, dict), "supplement parent binding missing")
    core_root = _resolve_recorded_path(core_record.get("path"), label="core parent")
    core_info = _verify_core_release(
        core_root,
        expected_manifest_sha256=str(core_record.get("manifest_sha256")),
    )
    _require(
        core_record
        == {
            "path": _record_path(core_root),
            "release_id": core_info["manifest"]["release_id"],
            "manifest_sha256": core_info["manifest_sha256"],
            "unique_train_sha256": core_info["unique_train_sha256"],
            "unique_train_rows": len(core_info["sources"]),
        },
        "core parent record drift",
    )
    supplement_pipeline_snapshot = root / "provenance/supplement_pipeline"
    supplement_snapshot = supplement_pipeline_snapshot / "teacher_targets/supplement"
    supplement_info = _verify_supplement_teacher(
        supplement_snapshot,
        wrapper_snapshot_root=(
            supplement_pipeline_snapshot / SUPPLEMENT_WRAPPER_SNAPSHOT_DIR
        ),
    )
    _require(
        isinstance(supplement_record.get("source_path"), str)
        and bool(supplement_record["source_path"]),
        "supplement source path binding missing",
    )
    expected_supplement_record = {
        "source_path": supplement_record.get("source_path"),
        "pipeline_source_path": supplement_record.get("pipeline_source_path"),
        "required_bundle_sha256": supplement_info["bundle_sha256"],
        "summary_sha256": supplement_info["summary_sha256"],
        "contract_sha256": supplement_info["contract_sha256"],
        "source_handoff_payload_sha256": supplement_info[
            "source_handoff_payload_sha256"
        ],
        "release_qa_sha256": supplement_info["release_qa_sha256"],
        "provider_identity": supplement_info["provider_identity"],
        "execution_wrappers": _public_execution_wrapper_records(
            supplement_info["execution_wrappers"]
        ),
        "train_rows": len(supplement_info["sources"]),
        "pipeline_snapshot_path": "provenance/supplement_pipeline",
        "snapshot_path": ("provenance/supplement_pipeline/teacher_targets/supplement"),
    }
    _require(
        isinstance(supplement_record.get("pipeline_source_path"), str)
        and bool(supplement_record["pipeline_source_path"])
        and supplement_record == expected_supplement_record,
        "supplement binding drift",
    )

    _validate_cross_population(core_info["sources"], supplement_info["sources"])
    sources = sorted(
        [*core_info["sources"], *supplement_info["sources"]],
        key=lambda row: str(row["sample_id"]),
    )
    persisted_sources = _read_jsonl(
        root / "manifests/source_unique_train.jsonl",
        label="augmented source unique train",
    )
    _require(persisted_sources == sources, "source unique train replay drift")
    schedule, sampler_contract = build_schedule(sources)
    persisted_schedule = _read_jsonl(
        root / "manifests/sampler_schedule.jsonl", label="augmented sampler schedule"
    )
    _require(persisted_schedule == schedule, "sampler schedule replay drift")
    source_by_id = {str(row["sample_id"]): row for row in sources}
    expected_sft, expected_grpo, expected_repeats = _materialize_train(
        schedule, source_by_id=source_by_id
    )
    _require(
        _read_jsonl(root / "decision_sft/train.jsonl", label="augmented SFT train")
        == expected_sft,
        "SFT train replay drift",
    )
    _require(
        _read_jsonl(root / "decision_grpo/train.jsonl", label="augmented GRPO train")
        == expected_grpo,
        "GRPO train replay drift",
    )
    _require(
        _read_jsonl(root / "manifests/repeats/train.jsonl", label="repeat manifest")
        == expected_repeats,
        "repeat manifest replay drift",
    )
    expected_unique = _unique_manifest(sources, schedule)
    _require(
        _read_jsonl(root / "manifests/unique/train.jsonl", label="combined unique")
        == expected_unique,
        "combined unique manifest replay drift",
    )

    evaluation = manifest.get("evaluation_inheritance")
    _require(isinstance(evaluation, dict), "evaluation inheritance binding missing")
    expected_evaluation = _evaluation_records(core_root)
    _require(evaluation == expected_evaluation, "evaluation inheritance record drift")
    for role in ROLES:
        for split in EVALUATION_SPLITS:
            _require(
                (root / role / f"{split}.jsonl").read_bytes()
                == (core_root / role / f"{split}.jsonl").read_bytes(),
                f"{role}/{split} is not byte-inherited",
            )
    for family in ("unique", "repeats"):
        for split in EVALUATION_SPLITS:
            _require(
                (root / "manifests" / family / f"{split}.jsonl").read_bytes()
                == (core_root / "manifests" / family / f"{split}.jsonl").read_bytes(),
                f"{family}/{split} is not byte-inherited",
            )

    expected_unique_counts = {
        "train": len(sources),
        "validation": files["decision_sft/validation.jsonl"]["rows"],
        "test": files["decision_sft/test.jsonl"]["rows"],
    }
    _require(
        manifest.get("unique_split_counts") == expected_unique_counts,
        "unique split counts drift",
    )
    expected_physical_counts = {
        role: {
            "train": len(schedule),
            "validation": files[f"{role}/validation.jsonl"]["rows"],
            "test": files[f"{role}/test.jsonl"]["rows"],
        }
        for role in ROLES
    }
    _require(
        manifest.get("physical_split_counts") == expected_physical_counts,
        "physical split counts drift",
    )
    expected_training_roles = {
        SFT_ROLE: {
            "dataset_path": "decision_sft",
            "parent_role": "selected_chk1_merged",
            "max_length": 3072,
            "completion_only_loss": True,
            "sampler_contract": "sampler_contract",
        },
        GRPO_ROLE: {
            "dataset_path": "decision_grpo",
            "parent_role": "merged_pre2009_decision_sft_warm_start",
            "reward": "decision_dense_v3",
            "max_prompt_length": 2560,
            "max_completion_length": 1024,
            "dataset_shuffle_allowed": True,
        },
    }
    _require(
        manifest.get("training_roles") == expected_training_roles,
        "training role contract drift",
    )
    expected_augmentation = {
        "split_policy": "supplement_train_only",
        "supplement_date_min": min(
            row["meeting_date"] for row in supplement_info["sources"]
        ),
        "supplement_date_max": max(
            row["meeting_date"] for row in supplement_info["sources"]
        ),
        "supplement_rows": len(supplement_info["sources"]),
        "supplement_action_counts": supplement_info["action_counts"],
        "supplement_action_floors": SUPPLEMENT_ACTION_FLOORS,
        "cross_population_overlaps": {
            "sample_id": 0,
            "meeting_date": 0,
            "prompt_sha256": 0,
            "source_id": 0,
        },
        "validation_new_rows": 0,
        "test_new_rows": 0,
    }
    _require(
        manifest.get("augmentation_contract") == expected_augmentation,
        "augmentation contract drift",
    )

    sampler = manifest.get("sampler_contract")
    _require(isinstance(sampler, dict), "sampler contract missing")
    expected_sampler = {
        **sampler_contract,
        "schedule_path": "manifests/sampler_schedule.jsonl",
        "schedule_sha256": files["manifests/sampler_schedule.jsonl"]["sha256"],
        "train_path": "decision_sft/train.jsonl",
        "train_sha256": files["decision_sft/train.jsonl"]["sha256"],
    }
    _require(sampler == expected_sampler, "sampler contract drift")
    expected_input = _build_input_contract(
        core_info=core_info, supplement_info=supplement_info
    )
    input_record = manifest.get("input_contract")
    _require(
        input_record
        == {
            "path": "contracts/decision_input_contract.json",
            "schema_version": INPUT_CONTRACT_SCHEMA,
            "contract_sha256": expected_input["contract_sha256"],
            "file_sha256": files["contracts/decision_input_contract.json"]["sha256"],
        },
        "input contract manifest binding drift",
    )
    _require(
        _read_json(
            root / "contracts/decision_input_contract.json", label="input contract"
        )
        == expected_input,
        "input contract replay drift",
    )
    _require(
        (root / "contracts/core_decision_input_contract_v1.json").read_bytes()
        == (core_root / "contracts/decision_input_contract.json").read_bytes(),
        "core input contract snapshot is not byte-identical",
    )
    expected_audit = _build_audit(
        core_info=core_info,
        supplement_info=supplement_info,
        sources=sources,
        schedule=schedule,
        sampler_contract=sampler_contract,
    )
    _require(
        _read_json(root / "audits/data_quality.json", label="data audit")
        == expected_audit,
        "data audit replay drift",
    )

    implementation = manifest.get("implementation")
    _require(isinstance(implementation, dict), "implementation binding missing")
    materializer_snapshot = root / "provenance/materializer_snapshot.py"
    _require(
        implementation.get("materializer_snapshot")
        == "provenance/materializer_snapshot.py"
        and implementation.get("materializer_snapshot_sha256")
        == _sha256_file(materializer_snapshot)
        and isinstance(implementation.get("python"), str)
        and bool(implementation["python"]),
        "implementation snapshot binding drift",
    )

    handoff = _read_json(root / "handoff.json", label="augmented handoff")
    unsigned_handoff = dict(handoff)
    unsigned_handoff.pop("release_manifest_sha256", None)
    _require(
        manifest.get("handoff")
        == {
            "path": "handoff.json",
            "schema_version": HANDOFF_SCHEMA,
            "unsigned_payload_sha256": _sha256_text(_canonical_json(unsigned_handoff)),
        },
        "handoff manifest binding drift",
    )
    _require(
        handoff.get("schema_version") == HANDOFF_SCHEMA
        and handoff.get("release_id") == RELEASE_ID
        and handoff.get("quality_status") == "passed"
        and handoff.get("immutable") is True
        and handoff.get("training_ready") is True
        and handoff.get("release_manifest") == "release_manifest.json"
        and handoff.get("release_manifest_sha256") == manifest_sha
        and handoff.get("test_is_sealed_evaluation_only") is True,
        "handoff contract drift",
    )
    return {
        **manifest,
        "verified_manifest_sha256": manifest_sha,
        "verified_split_files": {
            "decision_sft": {
                "train": root / "decision_sft/train.jsonl",
                "validation": root / "decision_sft/validation.jsonl",
            },
            "decision_grpo": {
                "train": root / "decision_grpo/train.jsonl",
                "validation": root / "decision_grpo/validation.jsonl",
            },
        },
    }


def publish(
    *,
    core_release: Path = DEFAULT_CORE_RELEASE,
    supplement_teacher: Path = DEFAULT_SUPPLEMENT_TEACHER,
    destination: Path = DEFAULT_OUTPUT,
    expected_core_manifest_sha256: str = DEFAULT_CORE_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Materialize, deeply verify, seal, and atomically publish the release."""

    raw_destination = destination.absolute()
    _require(
        not os.path.lexists(raw_destination),
        f"immutable destination already exists: {raw_destination}",
    )
    destination = raw_destination.parent.resolve() / raw_destination.name
    _require(
        destination.name == RELEASE_ID, "destination must use versioned release_id"
    )
    _require(
        not os.path.lexists(destination),
        f"immutable destination already exists: {destination}",
    )
    core_info = _verify_core_release(
        core_release.resolve(),
        expected_manifest_sha256=expected_core_manifest_sha256,
    )
    supplement_info = _verify_supplement_teacher(supplement_teacher.resolve())
    _validate_cross_population(core_info["sources"], supplement_info["sources"])
    sources = sorted(
        [*core_info["sources"], *supplement_info["sources"]],
        key=lambda row: str(row["sample_id"]),
    )
    schedule, sampler_base = build_schedule(sources)
    source_by_id = {str(row["sample_id"]): row for row in sources}
    sft_rows, grpo_rows, repeat_rows = _materialize_train(
        schedule, source_by_id=source_by_id
    )
    unique_rows = _unique_manifest(sources, schedule)
    input_contract = _build_input_contract(
        core_info=core_info, supplement_info=supplement_info
    )
    audit = _build_audit(
        core_info=core_info,
        supplement_info=supplement_info,
        sources=sources,
        schedule=schedule,
        sampler_contract=sampler_base,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{RELEASE_ID}.staging.", dir=destination.parent)
    )
    renamed = False
    try:
        _write_jsonl(staging / "decision_sft/train.jsonl", sft_rows)
        _write_jsonl(staging / "decision_grpo/train.jsonl", grpo_rows)
        for role in ROLES:
            for split in EVALUATION_SPLITS:
                _write_bytes(
                    staging / role / f"{split}.jsonl",
                    (core_info["root"] / role / f"{split}.jsonl").read_bytes(),
                )
        _write_jsonl(staging / "manifests/source_unique_train.jsonl", sources)
        _write_jsonl(staging / "manifests/sampler_schedule.jsonl", schedule)
        _write_jsonl(staging / "manifests/unique/train.jsonl", unique_rows)
        _write_jsonl(staging / "manifests/repeats/train.jsonl", repeat_rows)
        for family in ("unique", "repeats"):
            for split in EVALUATION_SPLITS:
                _write_bytes(
                    staging / "manifests" / family / f"{split}.jsonl",
                    (
                        core_info["root"] / "manifests" / family / f"{split}.jsonl"
                    ).read_bytes(),
                )
        _write_json(staging / "contracts/decision_input_contract.json", input_contract)
        _write_bytes(
            staging / "contracts/core_decision_input_contract_v1.json",
            (core_info["root"] / "contracts/decision_input_contract.json").read_bytes(),
        )
        _write_json(staging / "audits/data_quality.json", audit)
        _copy_teacher_bundle(
            supplement_info["root"], staging / "provenance/supplement_pipeline"
        )
        _copy_execution_wrappers(
            supplement_info["execution_wrappers"],
            staging
            / "provenance/supplement_pipeline"
            / SUPPLEMENT_WRAPPER_SNAPSHOT_DIR,
        )
        materializer_bytes = Path(__file__).resolve().read_bytes()
        _write_bytes(
            staging / "provenance/materializer_snapshot.py", materializer_bytes
        )
        files = _file_records(staging)
        sampler_contract = {
            **sampler_base,
            "schedule_path": "manifests/sampler_schedule.jsonl",
            "schedule_sha256": files["manifests/sampler_schedule.jsonl"]["sha256"],
            "train_path": "decision_sft/train.jsonl",
            "train_sha256": files["decision_sft/train.jsonl"]["sha256"],
        }
        evaluation = _evaluation_records(core_info["root"])
        created_at = _utc_now()
        handoff_unsigned = {
            "schema_version": HANDOFF_SCHEMA,
            "release_id": RELEASE_ID,
            "created_at_utc": created_at,
            "quality_status": "passed",
            "immutable": True,
            "training_ready": True,
            "release_manifest": "release_manifest.json",
            "decision_sft_path": "decision_sft",
            "decision_grpo_path": "decision_grpo",
            "sampler_schedule": "manifests/sampler_schedule.jsonl",
            "physical_train_rows": len(schedule),
            "optimizer_steps": sampler_base["optimizer_steps"],
            "test_is_sealed_evaluation_only": True,
        }
        manifest = {
            "schema_version": RELEASE_SCHEMA,
            "release_id": RELEASE_ID,
            "release_type": "train_only_augmentation",
            "created_at_utc": created_at,
            "quality_status": "passed",
            "immutable": True,
            "training_ready": True,
            "canonical_dag_bindable": False,
            "population_scope": "core-plus-pre2009-train-only",
            "grain": (
                "one unique target-decision-blind meeting brief; physical train "
                "exposure follows one manifest-bound 4/2/2 schedule"
            ),
            "parent_releases": {
                "core_v3": {
                    "path": _record_path(core_info["root"]),
                    "release_id": core_info["manifest"]["release_id"],
                    "manifest_sha256": core_info["manifest_sha256"],
                    "unique_train_sha256": core_info["unique_train_sha256"],
                    "unique_train_rows": len(core_info["sources"]),
                },
                "pre2009_supplement_teacher": {
                    "source_path": _record_path(supplement_info["root"]),
                    "pipeline_source_path": _record_path(
                        supplement_info["pipeline_root"]
                    ),
                    "required_bundle_sha256": supplement_info["bundle_sha256"],
                    "summary_sha256": supplement_info["summary_sha256"],
                    "contract_sha256": supplement_info["contract_sha256"],
                    "source_handoff_payload_sha256": supplement_info[
                        "source_handoff_payload_sha256"
                    ],
                    "release_qa_sha256": supplement_info["release_qa_sha256"],
                    "provider_identity": supplement_info["provider_identity"],
                    "execution_wrappers": _public_execution_wrapper_records(
                        supplement_info["execution_wrappers"]
                    ),
                    "train_rows": len(supplement_info["sources"]),
                    "pipeline_snapshot_path": "provenance/supplement_pipeline",
                    "snapshot_path": (
                        "provenance/supplement_pipeline/teacher_targets/supplement"
                    ),
                },
            },
            "augmentation_contract": {
                "split_policy": "supplement_train_only",
                "supplement_date_min": min(
                    row["meeting_date"] for row in supplement_info["sources"]
                ),
                "supplement_date_max": max(
                    row["meeting_date"] for row in supplement_info["sources"]
                ),
                "supplement_rows": len(supplement_info["sources"]),
                "supplement_action_counts": supplement_info["action_counts"],
                "supplement_action_floors": SUPPLEMENT_ACTION_FLOORS,
                "cross_population_overlaps": {
                    "sample_id": 0,
                    "meeting_date": 0,
                    "prompt_sha256": 0,
                    "source_id": 0,
                },
                "validation_new_rows": 0,
                "test_new_rows": 0,
            },
            "unique_split_counts": {
                "train": len(sources),
                "validation": files["decision_sft/validation.jsonl"]["rows"],
                "test": files["decision_sft/test.jsonl"]["rows"],
            },
            "physical_split_counts": {
                role: {
                    "train": len(schedule),
                    "validation": files[f"{role}/validation.jsonl"]["rows"],
                    "test": files[f"{role}/test.jsonl"]["rows"],
                }
                for role in ROLES
            },
            "training_roles": {
                SFT_ROLE: {
                    "dataset_path": "decision_sft",
                    "parent_role": "selected_chk1_merged",
                    "max_length": 3072,
                    "completion_only_loss": True,
                    "sampler_contract": "sampler_contract",
                },
                GRPO_ROLE: {
                    "dataset_path": "decision_grpo",
                    "parent_role": "merged_pre2009_decision_sft_warm_start",
                    "reward": "decision_dense_v3",
                    "max_prompt_length": 2560,
                    "max_completion_length": 1024,
                    "dataset_shuffle_allowed": True,
                },
            },
            "sampler_contract": sampler_contract,
            "evaluation_inheritance": evaluation,
            "input_contract": {
                "path": "contracts/decision_input_contract.json",
                "schema_version": INPUT_CONTRACT_SCHEMA,
                "contract_sha256": input_contract["contract_sha256"],
                "file_sha256": files["contracts/decision_input_contract.json"][
                    "sha256"
                ],
            },
            "implementation": {
                "materializer_snapshot": "provenance/materializer_snapshot.py",
                "materializer_snapshot_sha256": _sha256_bytes(materializer_bytes),
                "python": platform.python_version(),
            },
            "handoff": {
                "path": "handoff.json",
                "schema_version": HANDOFF_SCHEMA,
                "unsigned_payload_sha256": _sha256_text(
                    _canonical_json(handoff_unsigned)
                ),
            },
            "test_is_sealed_evaluation_only": True,
            "files": files,
        }
        _write_json(staging / "release_manifest.json", manifest)
        manifest_sha = _sha256_file(staging / "release_manifest.json")
        _write_json(
            staging / "handoff.json",
            {**handoff_unsigned, "release_manifest_sha256": manifest_sha},
        )
        _seal_tree(staging)
        verify_release(staging, expected_manifest_sha256=manifest_sha)
        _rename_noreplace(staging, destination)
        renamed = True
        verified = verify_release(destination, expected_manifest_sha256=manifest_sha)
        return {
            "status": "published",
            "release_id": RELEASE_ID,
            "destination": str(destination),
            "release_manifest_sha256": manifest_sha,
            "unique_train_rows": verified["unique_split_counts"]["train"],
            "physical_train_rows": verified["sampler_contract"]["train_rows"],
            "optimizer_steps": verified["sampler_contract"]["optimizer_steps"],
        }
    except Exception:
        if not renamed and staging.exists():
            _unseal_tree(staging)
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    publish_parser = commands.add_parser("publish")
    publish_parser.add_argument(
        "--core-release", type=Path, default=DEFAULT_CORE_RELEASE
    )
    publish_parser.add_argument(
        "--core-manifest-sha256", default=DEFAULT_CORE_MANIFEST_SHA256
    )
    publish_parser.add_argument(
        "--supplement-teacher", type=Path, default=DEFAULT_SUPPLEMENT_TEACHER
    )
    publish_parser.add_argument("--destination", type=Path, default=DEFAULT_OUTPUT)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--release", type=Path, default=DEFAULT_OUTPUT)
    verify_parser.add_argument("--expected-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "publish":
            result = publish(
                core_release=args.core_release,
                supplement_teacher=args.supplement_teacher,
                destination=args.destination,
                expected_core_manifest_sha256=args.core_manifest_sha256,
            )
        else:
            verified = verify_release(
                args.release,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
            result = {
                "status": "verified",
                "release_id": verified["release_id"],
                "release_manifest_sha256": verified["verified_manifest_sha256"],
                "unique_train_rows": verified["unique_split_counts"]["train"],
                "physical_train_rows": verified["sampler_contract"]["train_rows"],
                "optimizer_steps": verified["sampler_contract"]["optimizer_steps"],
            }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (Pre2009ReleaseError, MaterializationError, OSError, ValueError) as exc:
        print(
            json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False),
            file=os.sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CORE_MANIFEST_SHA256",
    "DEFAULT_CORE_RELEASE",
    "DEFAULT_OUTPUT",
    "DEFAULT_SUPPLEMENT_TEACHER",
    "EFFECTIVE_BATCH_SIZE",
    "GRPO_ROLE",
    "MAX_SOURCE_REPEAT",
    "PER_WINDOW_COUNTS",
    "Pre2009ReleaseError",
    "RELEASE_ID",
    "RELEASE_SCHEMA",
    "SFT_ROLE",
    "SUPPLEMENT_ACTION_FLOORS",
    "_canonical_json",
    "_gold_text",
    "_seal_tree",
    "_sha256_file",
    "_sha256_text",
    "_unseal_tree",
    "build_schedule",
    "publish",
    "verify_release",
]
