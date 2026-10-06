"""Safely supersede the live merged Core8 NF4/K10 run with vLLM BF16/K5.

This utility is deliberately tied to the 2026-08-16 production migration.  It
audits the durable K10 prefix, verifies the exact tmux/GPU0 process, sends one
terminal interrupt, audits the stopped prefix again, and creates a sealed,
read-only migration receipt.  It never edits or deletes the old prefix.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from open_r1.validator.loo_generation_spec import (
    derive_row_seed,
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
OLD_RUN_ROOT = (
    ROOT / "output/evaluation/main/chk3_beta_core8_merged_n2048_k10_20260815_v1"
)
OLD_SUITE_ROOT = OLD_RUN_ROOT / "generation_formal_n2048_k10_three_models"
OLD_MODEL_ROOT = OLD_SUITE_ROOT / "chk1"
OLD_STATE = OLD_MODEL_ROOT / "state.progress.v1.json"
OLD_WAL = OLD_MODEL_ROOT / ".partial/generations.progress.v1.jsonl"
OLD_SAMPLE_MANIFEST = OLD_RUN_ROOT / "preparation/samples_n2048_k10.json"
OLD_PIPELINE_LOG = OLD_RUN_ROOT / "pipeline_20260816.log"
OLD_SESSION = "chk3_beta_core8_prepost_k10_gpu0_20260816"
OLD_LOCK = Path("/tmp/fomc_trainer_chk3_beta_core8_merged_n2048_k10_gpu0.lock")
OLD_PYTHON = Path("/home/haobin_cui/.conda/envs/fomc_trainer/bin/python")
OLD_MODULE = "jobs.eval.eval_chk3_beta_core8_merged_stochastic_k10"
OLD_EVALUATION_ID = "chk3-beta-core8-merged-1993-2025-n2048-k10-v1"
OLD_SAMPLE_SHA256 = "e8d602e9ed1da8d5bd3d2e8dae11368667dfd2bc6192003b78a2b7d31887b866"
PRE_RELEASE_SHA256 = "82b045866d9ed6ccbc0d4f00014bffc97694859c2827215309dc8eabe5406937"
POST_RELEASE_SHA256 = "d7e6d9ea534039f60343dcab317588b45c1de5aa2814d2e6fff0d8ca0f1feb21"
REPLICATE_SEEDS = (
    20260811,
    21260811,
    22260811,
    23260811,
    24260811,
    25260811,
    26260811,
    27260811,
    28260811,
    29260811,
)
EXPECTED_OLD_CASES = 20480
EXPECTED_NEW_TOTAL_CASES = 30720
TARGET_VLLM_CONTRACT = {
    "backend": "vLLM continuous batching",
    "vllm_version": "0.8.5.post1",
    "weight_precision": "bfloat16",
    "dtype": "bfloat16",
    "quantization": None,
    "data_parallel_size": 2,
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "physical_gpu_indices": [0, 1],
    "full_model_replicas": 2,
    "gpu_memory_utilization": 0.95,
    "enforce_eager": True,
    "cuda_graphs": False,
    "max_num_seqs_candidates": [8, 12, 16],
    "max_num_seqs_selection": "freeze_fastest_stable_dual_gpu_smoke_candidate",
    "max_num_batched_tokens": 4096,
    "absolute_chunk_size": 40,
}
NEW_RUN_ROOT = (
    ROOT / "output/evaluation/main/chk3_beta_core8_merged_n2048_vllm_k5_20260816_v1"
)
DEFAULT_RECEIPT = NEW_RUN_ROOT / "migration/nf4_k10_superseded_receipt.json"
CONFIRMATION = "STOP-NF4-K10-AND-SEAL-VLLM-BF16-K5-MIGRATION"
RECEIPT_SCHEMA = "chk3-beta-core8-k10-to-vllm-k5-migration-receipt-v1"


class MigrationError(RuntimeError):
    """Raised when any migration invariant is not proven."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if path.is_symlink() or not resolved.is_file():
        raise MigrationError(f"unsafe or missing bound file: {path}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _read_json(path: Path, *, sealed: bool) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if path.is_symlink() or not resolved.is_file():
        raise MigrationError(f"unsafe or missing JSON file: {path}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MigrationError(f"cannot read JSON file {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise MigrationError(f"JSON root is not an object: {resolved}")
    if sealed:
        try:
            validate_manifest_integrity(value)
        except Exception as exc:
            raise MigrationError(
                f"sealed JSON integrity failed: {resolved}: {exc}"
            ) from exc
    return value


def _samples(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _read_json(path, sealed=True)
    observed_sha = _sha256_file(path)
    if (
        path.resolve() == OLD_SAMPLE_MANIFEST.resolve()
        and observed_sha != OLD_SAMPLE_SHA256
    ):
        raise MigrationError("production K10 sample-manifest SHA drift")
    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise MigrationError("sample manifest has no samples")
    samples: list[dict[str, Any]] = []
    for index, sample in enumerate(raw_samples):
        if not isinstance(sample, dict) or not isinstance(sample.get("sample_id"), str):
            raise MigrationError(f"invalid sample at index {index}")
        samples.append(sample)
    return manifest, samples


def _expected_tuple(samples: Sequence[Mapping[str, Any]], index: int) -> dict[str, Any]:
    sample_index, replicate_id = divmod(index, len(REPLICATE_SEEDS))
    if sample_index >= len(samples):
        raise MigrationError("WAL extends beyond the frozen sample/replicate matrix")
    sample_id = str(samples[sample_index]["sample_id"])
    replicate_seed = REPLICATE_SEEDS[replicate_id]
    return {
        "model_id": "chk1",
        "sample_id": sample_id,
        "replicate_id": replicate_id,
        "replicate_seed": replicate_seed,
        "row_seed": derive_row_seed(replicate_seed, sample_id),
    }


def _validate_row(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    sample_manifest_sha256: str,
    index: int,
) -> None:
    for key, value in expected.items():
        if row.get(key) != value:
            raise MigrationError(
                f"noncanonical WAL tuple/value at row {index}, {key}: "
                f"expected={value!r}, observed={row.get(key)!r}"
            )
    if row.get("sample_manifest_sha256") != sample_manifest_sha256:
        raise MigrationError(f"sample-manifest binding drift at WAL row {index}")
    if row.get("evaluation_id") != OLD_EVALUATION_ID:
        raise MigrationError(f"evaluation ID drift at WAL row {index}")


def _parse_canonical_lines(
    payload: bytes,
    *,
    samples: Sequence[Mapping[str, Any]],
    sample_manifest_sha256: str,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    if payload and not payload.endswith(b"\n"):
        raise MigrationError("WAL has an incomplete non-newline tail")
    rows: list[dict[str, Any]] = []
    for offset, raw_line in enumerate(payload.splitlines(keepends=True)):
        index = start_index + offset
        try:
            text = raw_line[:-1].decode("utf-8")
            row = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MigrationError(f"invalid WAL JSON at row {index}: {exc}") from exc
        if not isinstance(row, dict):
            raise MigrationError(f"WAL row {index} is not an object")
        if (_canonical(row) + "\n").encode("utf-8") != raw_line:
            raise MigrationError(f"WAL row {index} is not canonical JSON")
        _validate_row(
            row,
            expected=_expected_tuple(samples, index),
            sample_manifest_sha256=sample_manifest_sha256,
            index=index,
        )
        rows.append(row)
    return rows


def audit_state_bound_prefix(
    *, state_path: Path, wal_path: Path, sample_manifest_path: Path
) -> dict[str, Any]:
    """Validate the immutable state-bound WAL prefix while appends may continue."""

    state = _read_json(state_path, sealed=True)
    _, samples = _samples(sample_manifest_path)
    sample_sha = _sha256_file(sample_manifest_path)
    if (
        state.get("schema_version") != "chk3-beta-core8-merged-generation-state-v1"
        or state.get("evaluation_id") != OLD_EVALUATION_ID
        or state.get("model_id") != "chk1"
        or state.get("expected_cases") != EXPECTED_OLD_CASES
    ):
        raise MigrationError("K10 state contract drift")
    completed = state.get("completed_cases")
    if isinstance(completed, bool) or not isinstance(completed, int) or completed < 0:
        raise MigrationError("invalid completed_cases in K10 state")
    partial = state.get("partial_results")
    if not isinstance(partial, Mapping):
        raise MigrationError("K10 state has no partial-results binding")
    if Path(str(partial.get("path"))).resolve() != wal_path.resolve():
        raise MigrationError("K10 state WAL path drift")
    bound_bytes = partial.get("bytes")
    bound_sha = partial.get("sha256")
    if (
        isinstance(bound_bytes, bool)
        or not isinstance(bound_bytes, int)
        or bound_bytes < 0
        or not isinstance(bound_sha, str)
    ):
        raise MigrationError("invalid K10 state WAL binding")
    if wal_path.is_symlink() or not wal_path.is_file():
        raise MigrationError("unsafe or missing K10 WAL")
    with wal_path.open("rb") as handle:
        prefix = handle.read(bound_bytes)
    if len(prefix) != bound_bytes or hashlib.sha256(prefix).hexdigest() != bound_sha:
        raise MigrationError("state-bound K10 WAL prefix SHA/size mismatch")
    rows = _parse_canonical_lines(
        prefix,
        samples=samples,
        sample_manifest_sha256=sample_sha,
    )
    if len(rows) != completed:
        raise MigrationError("state-bound K10 WAL row-count mismatch")
    last_tuple = (
        None
        if not rows
        else {
            "model_id": rows[-1]["model_id"],
            "sample_id": rows[-1]["sample_id"],
            "replicate_id": rows[-1]["replicate_id"],
        }
    )
    if state.get("last_tuple") != last_tuple:
        raise MigrationError("K10 state last_tuple drift")
    matrix = state.get("completion_matrix")
    if not isinstance(matrix, Mapping) or matrix.get("last_tuple") != last_tuple:
        raise MigrationError("K10 compact completion matrix drift")
    if matrix.get("completed_cases") != completed:
        raise MigrationError("K10 compact completed-case count drift")
    return {
        "state": _binding(state_path),
        "state_payload_sha256": validate_manifest_integrity(state),
        "state_status": state.get("status"),
        "completed_cases": completed,
        "expected_cases": EXPECTED_OLD_CASES,
        "bound_wal": {
            "path": str(wal_path.resolve()),
            "bytes": bound_bytes,
            "sha256": bound_sha,
            "rows": completed,
        },
        "last_tuple": last_tuple,
    }


def audit_stopped_progress(
    *, state_path: Path, wal_path: Path, sample_manifest_path: Path
) -> dict[str, Any]:
    """Validate an exact stopped prefix, including the allowed one-row window."""

    prefix_audit = audit_state_bound_prefix(
        state_path=state_path,
        wal_path=wal_path,
        sample_manifest_path=sample_manifest_path,
    )
    _, samples = _samples(sample_manifest_path)
    sample_sha = _sha256_file(sample_manifest_path)
    bound_bytes = int(prefix_audit["bound_wal"]["bytes"])
    with wal_path.open("rb") as handle:
        handle.seek(bound_bytes)
        tail = handle.read()
    tail_rows = _parse_canonical_lines(
        tail,
        samples=samples,
        sample_manifest_sha256=sample_sha,
        start_index=int(prefix_audit["completed_cases"]),
    )
    if len(tail_rows) > 1:
        raise MigrationError("stopped WAL is more than one row ahead of state")
    actual = _binding(wal_path)
    actual_rows = int(prefix_audit["completed_cases"]) + len(tail_rows)
    actual_last = (
        prefix_audit["last_tuple"]
        if not tail_rows
        else {
            "model_id": tail_rows[-1]["model_id"],
            "sample_id": tail_rows[-1]["sample_id"],
            "replicate_id": tail_rows[-1]["replicate_id"],
        }
    )
    return {
        **prefix_audit,
        "actual_wal": {
            **actual,
            "rows": actual_rows,
            "last_tuple": actual_last,
        },
        "fsynced_one_row_ahead_of_state": len(tail_rows) == 1,
        "resume_contract_valid": True,
    }


def _production_pids() -> list[int]:
    result: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            parts = (entry / "cmdline").read_bytes().split(b"\0")
            argv = [part.decode("utf-8") for part in parts if part]
        except (OSError, UnicodeDecodeError):
            continue
        if (
            len(argv) >= 4
            and Path(argv[0]).resolve() == OLD_PYTHON.resolve()
            and argv[1:4] == ["-u", "-m", OLD_MODULE]
        ):
            result.append(int(entry.name))
    return sorted(result)


def _command(
    args: Sequence[str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(args),
            check=check,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MigrationError(f"command failed: {list(args)!r}: {exc}") from exc


def _gpu(index: int) -> dict[str, Any]:
    if index not in (0, 1):
        raise MigrationError(f"target GPU index must be 0 or 1, observed={index!r}")
    identity = _command(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    ).stdout.strip()
    apps = _command(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    ).stdout.splitlines()
    try:
        pids = sorted(int(value.strip()) for value in apps if value.strip())
    except ValueError as exc:
        raise MigrationError(f"cannot parse GPU{index} compute PIDs") from exc
    return {"identity": identity, "compute_pids": pids}


def _gpu0() -> dict[str, Any]:
    """Compatibility helper for the old single-GPU K10 process checks."""

    return _gpu(0)


def _tmux_pane_pid() -> int:
    result = _command(
        ["tmux", "list-panes", "-t", OLD_SESSION, "-F", "#{pane_pid}"],
    )
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(values) != 1 or not values[0].isdigit():
        raise MigrationError("the exact K10 tmux session does not have one pane")
    return int(values[0])


def validate_live_process() -> dict[str, Any]:
    pids = _production_pids()
    if len(pids) != 1:
        raise MigrationError(
            f"expected exactly one production K10 Python, observed={pids}"
        )
    pid = pids[0]
    pane_pid = _tmux_pane_pid()
    process_group = os.getpgid(pid)
    if pane_pid != process_group:
        raise MigrationError(
            f"tmux pane/process-group mismatch: pane={pane_pid}, pgid={process_group}"
        )
    gpu = _gpu0()
    if gpu["compute_pids"] != [pid]:
        raise MigrationError(
            f"GPU0 is not exclusively held by the exact K10 Python: {gpu['compute_pids']}"
        )
    return {
        "session": OLD_SESSION,
        "pane_pid": pane_pid,
        "python_pid": pid,
        "process_group": process_group,
        "gpu0": gpu,
    }


def _lock_is_free(path: Path) -> bool:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return True
    finally:
        os.close(descriptor)


def _wait_for_next_commit(initial_count: int, timeout_seconds: int) -> int:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _production_pids():
            raise MigrationError("K10 process exited before the controlled stop")
        state = _read_json(OLD_STATE, sealed=True)
        count = state.get("completed_cases")
        if (
            isinstance(count, int)
            and not isinstance(count, bool)
            and count > initial_count
        ):
            return count
        time.sleep(0.25)
    raise MigrationError("timed out waiting for the next durable K10 row")


def controlled_stop(
    *, boundary_wait_seconds: int, exit_wait_seconds: int
) -> dict[str, Any]:
    live = validate_live_process()
    before = audit_state_bound_prefix(
        state_path=OLD_STATE,
        wal_path=OLD_WAL,
        sample_manifest_path=OLD_SAMPLE_MANIFEST,
    )
    next_count = _wait_for_next_commit(
        int(before["completed_cases"]), boundary_wait_seconds
    )
    # The shortest observed completion is materially longer than two seconds;
    # this places SIGINT inside the next model.generate call, not the fsync window.
    time.sleep(2.0)
    stop_requested_at_utc = _utc_now()
    _command(["tmux", "send-keys", "-t", f"{OLD_SESSION}:0.0", "C-c"])
    deadline = time.monotonic() + exit_wait_seconds
    while time.monotonic() < deadline and _production_pids():
        time.sleep(0.25)
    remaining = _production_pids()
    if remaining:
        raise MigrationError(
            f"controlled SIGINT did not stop the exact K10 process: {remaining}"
        )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        session = _command(["tmux", "has-session", "-t", OLD_SESSION], check=False)
        if session.returncode != 0:
            break
        time.sleep(0.25)
    if (
        _command(["tmux", "has-session", "-t", OLD_SESSION], check=False).returncode
        == 0
    ):
        raise MigrationError("K10 tmux session survived controlled SIGINT")
    gpu_after = _gpu0()
    if gpu_after["compute_pids"]:
        raise MigrationError(
            f"GPU0 still has compute processes: {gpu_after['compute_pids']}"
        )
    if not _lock_is_free(OLD_LOCK):
        raise MigrationError("K10 GPU0 lock is still held")
    gpu1_after = _gpu(1)
    if gpu1_after["compute_pids"]:
        raise MigrationError(
            f"GPU1 is not idle for the target DP=2 run: {gpu1_after['compute_pids']}"
        )
    stopped = audit_stopped_progress(
        state_path=OLD_STATE,
        wal_path=OLD_WAL,
        sample_manifest_path=OLD_SAMPLE_MANIFEST,
    )
    return {
        "signal": "SIGINT_via_tmux_C-c",
        "stop_requested_at_utc": stop_requested_at_utc,
        "stopped_and_audited_at_utc": _utc_now(),
        "live_process_before_stop": live,
        "initial_completed_cases": before["completed_cases"],
        "commit_observed_before_signal": next_count,
        "gpu0_after_stop": gpu_after,
        "gpu1_target_before_start": gpu1_after,
        "lock_released": True,
        "stopped_prefix": stopped,
    }


def _implementation_bindings(paths: Sequence[Path]) -> list[dict[str, Any]]:
    unique = sorted({path.expanduser().resolve() for path in paths}, key=str)
    if not unique:
        raise MigrationError("at least one new K5 implementation file must be bound")
    return [_binding(path) for path in unique]


def build_receipt(
    *, stop_evidence: Mapping[str, Any], new_implementation_paths: Sequence[Path]
) -> dict[str, Any]:
    observed_stopped_prefix = audit_stopped_progress(
        state_path=OLD_STATE,
        wal_path=OLD_WAL,
        sample_manifest_path=OLD_SAMPLE_MANIFEST,
    )
    if observed_stopped_prefix != stop_evidence.get("stopped_prefix"):
        raise MigrationError("stopped K10 prefix changed before receipt sealing")
    old_implementations = [
        ROOT / "jobs/eval/eval_chk3_beta_core8_merged_stochastic_k10.py",
        ROOT / "jobs/eval/eval_chk3_stochastic_bootstrap_generation.py",
        ROOT / "run/eval_chk3_beta_core8_merged_n2048_k10.sh",
    ]
    rollback_log = OLD_RUN_ROOT / "pipeline_resume_20260816.log"
    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "passed",
        "created_at_utc": _utc_now(),
        "migration": {
            "user_request": "使用vLLM continuous batching，并且采用k=5",
            "requested_change": (
                "replace Transformers NF4 K10 on GPU0 with vLLM BF16 K5 "
                "replicated data parallelism on GPU0 and GPU1"
            ),
            "stop_policy": "one controlled SIGINT after a durable-row boundary",
            "old_prefix_preserved": True,
            "old_rows_reused_in_new_suite": False,
            "old_suite_superseded_for_primary_claim": True,
            "reason_old_rows_are_not_reusable": (
                "backend, precision, scheduler, and K changed; even shared first-five "
                "seeds are not exchangeable with the new common-backend cohort"
            ),
        },
        "old_suite": {
            "evaluation_id": OLD_EVALUATION_ID,
            "run_root": str(OLD_RUN_ROOT.resolve()),
            "backend": "transformers.generate",
            "weight_precision": "NF4",
            "compute_dtype": "bfloat16",
            "replicates": 10,
            "expected_total_rows": 61440,
            "preserved": True,
            "reusable_in_new_suite": False,
            "sample_manifest": _binding(OLD_SAMPLE_MANIFEST),
            "pre_release_sha256": PRE_RELEASE_SHA256,
            "post_release_sha256": POST_RELEASE_SHA256,
            "pipeline_log": _binding(OLD_PIPELINE_LOG),
            "implementations": [_binding(path) for path in old_implementations],
            "stop_evidence": dict(stop_evidence),
        },
        "new_suite": {
            "run_root": str(NEW_RUN_ROOT.resolve()),
            **TARGET_VLLM_CONTRACT,
            "replicates": 5,
            "replicate_seeds": list(REPLICATE_SEEDS[:5]),
            "prompts": 2048,
            "models": ["chk1", "chk3", "chk0"],
            "expected_total_rows": EXPECTED_NEW_TOTAL_CASES,
            "old_rows_reused": 0,
            "implementations": _implementation_bindings(new_implementation_paths),
        },
        "rollback": {
            "allowed_only_after_new_vllm_processes_are_gone_and_both_gpus_are_idle": True,
            "session": "chk3_beta_core8_prepost_k10_gpu0_resume_20260816",
            "working_directory": str(ROOT),
            "environment": {
                "FOMC_PYTHON_BIN": str(OLD_PYTHON),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "0",
            },
            "launcher_argv": [
                str(ROOT / "run/eval_chk3_beta_core8_merged_n2048_k10.sh"),
                POST_RELEASE_SHA256,
                "formal",
            ],
            "log": str(rollback_log.resolve()),
            "delete_new_failed_root": False,
        },
        "receipt_implementation": _binding(Path(__file__)),
    }
    return seal_manifest(payload)


def write_readonly_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise MigrationError(
            f"receipt is create-only and already exists: {destination}"
        )
    encoded = (_canonical(receipt) + "\n").encode("utf-8")
    staging = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(staging, 0o444)
        try:
            os.link(staging, destination)
        except FileExistsError as exc:
            raise MigrationError(f"receipt publication race: {destination}") from exc
        with destination.open("rb") as handle:
            os.fsync(handle.fileno())
    finally:
        staging.unlink(missing_ok=True)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def validate_receipt(path: Path) -> dict[str, Any]:
    receipt = _read_json(path, sealed=True)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o222:
        raise MigrationError(f"migration receipt is writable: mode={oct(mode)}")
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("migration", {}).get("old_prefix_preserved") is not True
        or receipt.get("migration", {}).get("old_rows_reused_in_new_suite") is not False
        or any(
            receipt.get("new_suite", {}).get(key) != expected
            for key, expected in TARGET_VLLM_CONTRACT.items()
        )
        or receipt.get("new_suite", {}).get("replicates") != 5
        or receipt.get("new_suite", {}).get("expected_total_rows")
        != EXPECTED_NEW_TOTAL_CASES
    ):
        raise MigrationError("migration receipt contract drift")
    return {
        "status": "passed",
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "payload_sha256": validate_manifest_integrity(receipt),
        "mode": oct(mode),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect-live-prefix")
    inspect.add_argument("--state", type=Path, default=OLD_STATE)
    inspect.add_argument("--wal", type=Path, default=OLD_WAL)
    inspect.add_argument("--sample-manifest", type=Path, default=OLD_SAMPLE_MANIFEST)
    stop = sub.add_parser("stop-and-seal")
    stop.add_argument("--confirm", required=True)
    stop.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    stop.add_argument("--new-implementation", action="append", type=Path, required=True)
    stop.add_argument("--boundary-wait-seconds", type=int, default=180)
    stop.add_argument("--exit-wait-seconds", type=int, default=120)
    validate = sub.add_parser("validate-receipt")
    validate.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect-live-prefix":
            result = audit_state_bound_prefix(
                state_path=args.state,
                wal_path=args.wal,
                sample_manifest_path=args.sample_manifest,
            )
        elif args.command == "stop-and-seal":
            if args.confirm != CONFIRMATION:
                raise MigrationError(
                    "exact production migration confirmation is required"
                )
            if args.receipt.resolve() != DEFAULT_RECEIPT.resolve():
                raise MigrationError(
                    "production migration receipt path cannot be overridden"
                )
            if args.receipt.exists() or args.receipt.is_symlink():
                raise MigrationError("production migration receipt already exists")
            # Resolve and hash every implementation before interrupting the old
            # run.  A typo must never leave a stopped run without a receipt.
            pre_stop_new_bindings = _implementation_bindings(args.new_implementation)
            for required in (
                OLD_SAMPLE_MANIFEST,
                OLD_PIPELINE_LOG,
                ROOT / "jobs/eval/eval_chk3_beta_core8_merged_stochastic_k10.py",
                ROOT / "jobs/eval/eval_chk3_stochastic_bootstrap_generation.py",
                ROOT / "run/eval_chk3_beta_core8_merged_n2048_k10.sh",
            ):
                _binding(required)
            stop_evidence = controlled_stop(
                boundary_wait_seconds=args.boundary_wait_seconds,
                exit_wait_seconds=args.exit_wait_seconds,
            )
            if (
                _implementation_bindings(args.new_implementation)
                != pre_stop_new_bindings
            ):
                raise MigrationError(
                    "new K5 implementation changed during controlled stop"
                )
            receipt = build_receipt(
                stop_evidence=stop_evidence,
                new_implementation_paths=args.new_implementation,
            )
            write_readonly_receipt(args.receipt, receipt)
            result = validate_receipt(args.receipt)
        else:
            result = validate_receipt(args.receipt)
    except MigrationError as exc:
        print(_canonical({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 1
    print(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
