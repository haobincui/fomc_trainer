"""Seal the zero-row vLLM V1 async-output compatibility remediation.

The first DP2/max_num_seqs=8 infrastructure benchmark failed before model
load because vLLM 0.8.5 V1 rejects an explicitly supplied
``disable_async_output_proc=True`` engine argument.  This production-scoped
bridge preserves that failed directory, binds its empty WAL evidence and the
original migration receipt, proves both GPUs are idle, and binds the patched
runner and launcher before publishing a create-only read-only receipt.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    ROOT / "output/evaluation/main/chk3_beta_core8_merged_n2048_vllm_k5_20260816_v1"
)
ORIGINAL_MIGRATION_RECEIPT = RUN_ROOT / "migration/nf4_k10_superseded_receipt.json"
FAILED_RUN_ROOT = RUN_ROOT / "benchmark_max_num_seqs_chk1_core8_k5/max_num_seqs_8/chk1"
PATCHED_BENCHMARK_ROOT = RUN_ROOT / "benchmark_max_num_seqs_chk1_core8_k5_v2"
PATCHED_RUNNER = ROOT / "jobs/eval/eval_chk3_beta_core8_merged_vllm_k5.py"
PATCHED_LAUNCHER = ROOT / "run/eval_chk3_beta_core8_merged_vllm_k5.sh"
DEFAULT_RECEIPT = RUN_ROOT / "migration/vllm_v1_async_output_remediation_receipt.json"
GPU_LOCKS = (
    Path("/tmp/fomc_trainer_chk3_beta_core8_vllm_k5_gpu0.lock"),
    Path("/tmp/fomc_trainer_chk3_beta_core8_vllm_k5_gpu1.lock"),
)

SCHEMA_VERSION = "chk3-beta-core8-vllm-k5-v1-async-output-remediation-receipt-v1"
EVALUATION_ID = "chk3-beta-core8-merged-1993-2025-n2048-vllm-k5-v1"
CONFIRMATION = "SEAL-VLLM-V1-ASYNC-OUTPUT-REMEDIATION"
FAILURE_REASON = (
    "vLLM 0.8.5 V1 rejects explicit "
    "AsyncEngineArgs(disable_async_output_proc=True) before model load"
)
EXPECTED_ERROR_TYPE = "NotImplementedError"
EXPECTED_ERROR = "VLLM_USE_V1=1 is not supported with --disable-async-output-proc."
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

ORIGINAL_MIGRATION_SHA256 = (
    "81b41fbcb9456a25e886377b23f22783c873235ccae4b794c3f954949247e07f"
)
ORIGINAL_MIGRATION_PAYLOAD_SHA256 = (
    "7c75660f0d0d9d39b618486c806346f597edcfeda2843fba4216b5b3ca494947"
)
OLD_RUNNER_SHA256 = "3e270d40ffc5e7be1989d3db79e6f26892974022d4a07601742fa1106d82dd37"
OLD_LAUNCHER_SHA256 = "d3d03aa0218458954a7bfcaebd43a0486da102e52b2ee3ff5d7e45a8834bb1c9"
FAILED_LAUNCH_SHA256 = (
    "e622533625c6a5eff282e34cee93529c57a5981a7de2cb85c0b5ff3299187554"
)
FAILED_LAUNCH_PAYLOAD_SHA256 = (
    "8f3f94ad5c678cf68c505feb1e284a941572972d6c0b4ed597cc840c8f6b057d"
)
FAILED_STATE_SHA256 = "8dc4da8f4b368fe343c31facb528698e1119b0a637c8558e3f747b5797768352"
FAILED_STATE_PAYLOAD_SHA256 = (
    "ce94a4e8f77f4e08738cdb4a298e06fa80f01abb5fbc9bc24233f68000c2b592"
)


class RemediationError(RuntimeError):
    """A production remediation invariant was not proven."""


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
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise RemediationError(f"bound artifact is a symlink: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise RemediationError(f"bound artifact is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _read_sealed_json(path: Path) -> tuple[dict[str, Any], str]:
    binding = _binding(path)
    try:
        value = json.loads(Path(binding["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemediationError(f"cannot read sealed JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RemediationError(f"sealed JSON is not an object: {path}")
    try:
        payload_sha256 = validate_manifest_integrity(value)
    except Exception as exc:
        raise RemediationError(f"sealed JSON integrity failed: {path}") from exc
    return value, payload_sha256


def _implementation_from_migration(
    migration: Mapping[str, Any], path: Path
) -> dict[str, Any]:
    implementations = migration.get("new_suite", {}).get("implementations")
    if not isinstance(implementations, list):
        raise RemediationError("original migration implementation inventory missing")
    matches = [
        item
        for item in implementations
        if isinstance(item, Mapping)
        and Path(str(item.get("path") or "")).resolve() == path.resolve()
    ]
    if len(matches) != 1:
        raise RemediationError(
            f"original migration has ambiguous implementation binding: {path}"
        )
    return dict(matches[0])


def audit_original_migration() -> dict[str, Any]:
    receipt, payload_sha256 = _read_sealed_json(ORIGINAL_MIGRATION_RECEIPT)
    binding = _binding(ORIGINAL_MIGRATION_RECEIPT)
    mode = stat.S_IMODE(ORIGINAL_MIGRATION_RECEIPT.stat().st_mode)
    runner = _implementation_from_migration(receipt, PATCHED_RUNNER)
    launcher = _implementation_from_migration(receipt, PATCHED_LAUNCHER)
    if (
        binding["sha256"] != ORIGINAL_MIGRATION_SHA256
        or payload_sha256 != ORIGINAL_MIGRATION_PAYLOAD_SHA256
        or mode != 0o444
        or receipt.get("status") != "passed"
        or runner.get("sha256") != OLD_RUNNER_SHA256
        or launcher.get("sha256") != OLD_LAUNCHER_SHA256
    ):
        raise RemediationError("original migration receipt drift")
    return {
        **binding,
        "payload_sha256": payload_sha256,
        "mode": oct(mode),
        "historical_runner": runner,
        "historical_launcher": launcher,
    }


def _assert_empty_jsonl(path: Path) -> dict[str, Any]:
    binding = _binding(path)
    if binding["bytes"] != 0 or binding["sha256"] != EMPTY_SHA256:
        raise RemediationError(f"failed-run JSONL is not empty: {path}")
    return {**binding, "rows": 0}


def audit_failed_attempt() -> dict[str, Any]:
    expected_inventory = {
        FAILED_RUN_ROOT / "launch.json",
        FAILED_RUN_ROOT / "state.progress.v1.json",
        FAILED_RUN_ROOT / ".partial/generations.completion_order.progress.v1.jsonl",
        FAILED_RUN_ROOT / ".partial/absolute_chunks.progress.v1.jsonl",
    }
    observed_inventory = {
        path.resolve() for path in FAILED_RUN_ROOT.rglob("*") if path.is_file()
    }
    if observed_inventory != {path.resolve() for path in expected_inventory}:
        raise RemediationError("failed max_num_seqs=8 directory inventory drift")
    for forbidden in (
        FAILED_RUN_ROOT / "manifest.json",
        FAILED_RUN_ROOT / "generations.canonical.v1.jsonl",
    ):
        if forbidden.exists() or forbidden.is_symlink():
            raise RemediationError(f"failed run unexpectedly produced {forbidden.name}")

    launch_path = FAILED_RUN_ROOT / "launch.json"
    state_path = FAILED_RUN_ROOT / "state.progress.v1.json"
    wal_path = (
        FAILED_RUN_ROOT / ".partial/generations.completion_order.progress.v1.jsonl"
    )
    receipt_path = FAILED_RUN_ROOT / ".partial/absolute_chunks.progress.v1.jsonl"
    launch, launch_payload_sha = _read_sealed_json(launch_path)
    state, state_payload_sha = _read_sealed_json(state_path)
    launch_binding = _binding(launch_path)
    state_binding = _binding(state_path)
    wal_binding = _assert_empty_jsonl(wal_path)
    chunk_binding = _assert_empty_jsonl(receipt_path)

    source_runner = (
        launch.get("source_artifact_sha256s", {})
        .get("implementation_sources", {})
        .get("vllm_k5_runner")
    )
    if (
        launch_binding["sha256"] != FAILED_LAUNCH_SHA256
        or launch_payload_sha != FAILED_LAUNCH_PAYLOAD_SHA256
        or state_binding["sha256"] != FAILED_STATE_SHA256
        or state_payload_sha != FAILED_STATE_PAYLOAD_SHA256
        or launch.get("schema_version")
        != "chk3-beta-core8-merged-vllm-k5-run-manifest-v1"
        or launch.get("status") != "initializing"
        or launch.get("evaluation_id") != EVALUATION_ID
        or launch.get("evaluation_scope") != "infrastructure_smoke"
        or launch.get("model_id") != "chk1"
        or launch.get("generation", {}).get("max_num_seqs") != 8
        or launch.get("generation", {}).get("expected_cases") != 40
        or not isinstance(source_runner, Mapping)
        or source_runner.get("sha256") != OLD_RUNNER_SHA256
        or state.get("schema_version") != "chk3-beta-core8-merged-vllm-k5-state-v1"
        or state.get("status") != "failed"
        or state.get("evaluation_id") != EVALUATION_ID
        or state.get("model_id") != "chk1"
        or state.get("expected_cases") != 40
        or state.get("completed_cases") != 0
        or state.get("completed_absolute_chunks") != 0
        or state.get("active_chunk_completed_indexes") != []
        or state.get("resume_count") != 0
        or state.get("engine_runtime") is not None
        or state.get("error_type") != EXPECTED_ERROR_TYPE
        or state.get("error") != EXPECTED_ERROR
        or state.get("completion_wal") != wal_binding
        or state.get("chunk_receipts") != chunk_binding
    ):
        raise RemediationError("failed max_num_seqs=8 launch/state evidence drift")
    return {
        "run_root": str(FAILED_RUN_ROOT.resolve()),
        "preserved_unchanged": True,
        "launch": {**launch_binding, "payload_sha256": launch_payload_sha},
        "state": {**state_binding, "payload_sha256": state_payload_sha},
        "completion_wal": wal_binding,
        "absolute_chunk_receipts": chunk_binding,
        "canonical_generations_absent": True,
        "sealed_run_manifest_absent": True,
        "model_load_started": False,
        "generated_rows": 0,
        "completed_chunks": 0,
        "error_type": EXPECTED_ERROR_TYPE,
        "error": EXPECTED_ERROR,
        "historical_runner": dict(source_runner),
    }


def _validate_patched_runner_source(path: Path) -> dict[str, Any]:
    binding = _binding(path)
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise RemediationError("patched runner is not valid Python") from exc
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "AsyncEngineArgs"
    ]
    if len(calls) != 1:
        raise RemediationError("patched runner AsyncEngineArgs call is ambiguous")
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    enforce_eager = keywords.get("enforce_eager")
    if (
        "disable_async_output_proc" in keywords
        or not isinstance(enforce_eager, ast.Constant)
        or enforce_eager.value is not True
        or "model_config.use_async_output_proc is not False" not in source
        or '"async_output_processing": False' not in source
        or binding["sha256"] == OLD_RUNNER_SHA256
    ):
        raise RemediationError("patched runner async-output remediation drift")
    return binding


def _validate_patched_launcher_source(path: Path) -> dict[str, Any]:
    binding = _binding(path)
    source = path.read_text(encoding="utf-8")
    required = (
        "vllm_v1_async_output_remediation_receipt.json",
        "jobs.eval.remediate_chk3_beta_core8_vllm_k5_v1_async_output",
        "benchmark_max_num_seqs_chk1_core8_k5_v2",
    )
    if binding["sha256"] == OLD_LAUNCHER_SHA256 or any(
        value not in source for value in required
    ):
        raise RemediationError("patched launcher remediation gate/root drift")
    return binding


def _gpu_identity_rows() -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id,name,memory.total",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RemediationError(f"cannot inspect GPU identities: {exc}") from exc
    rows = []
    for line in completed.stdout.splitlines():
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 5 or not columns[0].isdigit():
            raise RemediationError(f"invalid GPU identity row: {line!r}")
        rows.append(
            {
                "physical_gpu_index": int(columns[0]),
                "uuid": columns[1],
                "pci_bus_id": columns[2].lower(),
                "name": columns[3],
                "memory_total_mib": int(columns[4]),
            }
        )
    return rows


def _gpu_compute_processes(index: int) -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                f"--id={index}",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RemediationError(f"cannot inspect GPU{index} processes: {exc}") from exc
    rows = []
    for line in completed.stdout.splitlines():
        columns = [column.strip() for column in line.split(",", 1)]
        if not columns or not columns[0].isdigit():
            if not line.strip() or "No running processes" in line:
                continue
            raise RemediationError(f"invalid GPU{index} process row: {line!r}")
        rows.append(
            {
                "pid": int(columns[0]),
                "process_name": columns[1] if len(columns) == 2 else "",
            }
        )
    return rows


def _prove_lock_pair_available() -> list[str]:
    handles = []
    acquired = []
    try:
        for path in GPU_LOCKS:
            if path.is_symlink() or not path.is_file():
                raise RemediationError(f"GPU lock is missing or unsafe: {path}")
            handle = path.open("r+", encoding="utf-8")
            handles.append(handle)
        for handle in handles:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RemediationError("GPU0/GPU1 lock pair is not available") from exc
            acquired.append(handle)
        return [str(path.resolve()) for path in GPU_LOCKS]
    finally:
        for handle in reversed(acquired):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        for handle in handles:
            handle.close()


def audit_idle_gpus(failed_attempt: Mapping[str, Any]) -> dict[str, Any]:
    launch_path = Path(str(failed_attempt["launch"]["path"]))
    launch, _ = _read_sealed_json(launch_path)
    expected = launch.get("runtime", {}).get("physical_gpu_identities")
    observed = _gpu_identity_rows()
    if not isinstance(expected, list) or len(expected) != 2 or len(observed) != 2:
        raise RemediationError("dual-GPU identity evidence is incomplete")
    for index in range(2):
        if (
            observed[index]["physical_gpu_index"] != index
            or observed[index]["uuid"] != expected[index].get("uuid")
            or observed[index]["pci_bus_id"]
            != str(expected[index].get("pci_bus_id")).lower()
        ):
            raise RemediationError(f"physical GPU{index} identity drift")
    process_rows = {str(index): _gpu_compute_processes(index) for index in range(2)}
    if any(process_rows.values()):
        raise RemediationError(
            f"both GPUs must be idle before remediation seal: {process_rows}"
        )
    return {
        "audited_at_utc": _utc_now(),
        "physical_gpu_identities": observed,
        "compute_processes": process_rows,
        "both_gpus_idle": True,
        "lock_pair_available": True,
        "gpu_lock_paths": _prove_lock_pair_available(),
    }


def build_receipt(
    *,
    original_migration: Mapping[str, Any],
    failed_attempt: Mapping[str, Any],
    gpu_idle: Mapping[str, Any],
    patched_runner: Mapping[str, Any],
    patched_launcher: Mapping[str, Any],
) -> dict[str, Any]:
    return seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_at_utc": _utc_now(),
            "evaluation_id": EVALUATION_ID,
            "reason": FAILURE_REASON,
            "original_migration_receipt": dict(original_migration),
            "failed_attempt": dict(failed_attempt),
            "remediation": {
                "unsupported_explicit_engine_argument_removed": True,
                "resolved_model_config_async_output_assertion_retained": True,
                "failed_v1_root_preserved": True,
                "new_benchmark_root": str(PATCHED_BENCHMARK_ROOT.resolve()),
                "new_benchmark_root_must_be_fresh": True,
                "old_runner_sha256": OLD_RUNNER_SHA256,
                "old_launcher_sha256": OLD_LAUNCHER_SHA256,
                "patched_runner": dict(patched_runner),
                "patched_launcher": dict(patched_launcher),
                "bridge_implementation": _binding(Path(__file__)),
            },
            "gpu_idle_evidence": dict(gpu_idle),
            "safety": {
                "generated_rows_reused": 0,
                "failed_directory_deleted_or_modified": False,
                "gpu_work_launched_by_remediation": False,
                "receipt_create_only": True,
                "receipt_mode": "0o444",
            },
        }
    )


def write_readonly_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise RemediationError(f"receipt is create-only: {destination}")
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
            raise RemediationError(f"receipt publication race: {destination}") from exc
        with destination.open("rb") as handle:
            os.fsync(handle.fileno())
    finally:
        staging.unlink(missing_ok=True)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def validate_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    receipt, payload_sha256 = _read_sealed_json(path)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o444:
        raise RemediationError(f"remediation receipt mode drift: {oct(mode)}")
    original = audit_original_migration()
    failed = audit_failed_attempt()
    patched_runner = _validate_patched_runner_source(PATCHED_RUNNER)
    patched_launcher = _validate_patched_launcher_source(PATCHED_LAUNCHER)
    remediation = receipt.get("remediation")
    idle = receipt.get("gpu_idle_evidence")
    safety = receipt.get("safety")
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("status") != "passed"
        or receipt.get("evaluation_id") != EVALUATION_ID
        or receipt.get("reason") != FAILURE_REASON
        or receipt.get("original_migration_receipt") != original
        or receipt.get("failed_attempt") != failed
        or not isinstance(remediation, Mapping)
        or remediation.get("unsupported_explicit_engine_argument_removed") is not True
        or remediation.get("resolved_model_config_async_output_assertion_retained")
        is not True
        or remediation.get("failed_v1_root_preserved") is not True
        or remediation.get("new_benchmark_root")
        != str(PATCHED_BENCHMARK_ROOT.resolve())
        or remediation.get("new_benchmark_root_must_be_fresh") is not True
        or remediation.get("old_runner_sha256") != OLD_RUNNER_SHA256
        or remediation.get("old_launcher_sha256") != OLD_LAUNCHER_SHA256
        or remediation.get("patched_runner") != patched_runner
        or remediation.get("patched_launcher") != patched_launcher
        or remediation.get("bridge_implementation") != _binding(Path(__file__))
        or not isinstance(idle, Mapping)
        or idle.get("both_gpus_idle") is not True
        or idle.get("lock_pair_available") is not True
        or idle.get("compute_processes") != {"0": [], "1": []}
        or not isinstance(safety, Mapping)
        or safety.get("generated_rows_reused") != 0
        or safety.get("failed_directory_deleted_or_modified") is not False
        or safety.get("gpu_work_launched_by_remediation") is not False
        or safety.get("receipt_create_only") is not True
        or safety.get("receipt_mode") != "0o444"
    ):
        raise RemediationError("sealed remediation receipt contract drift")
    return {
        "status": "passed",
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "payload_sha256": payload_sha256,
        "mode": oct(mode),
        "patched_runner_sha256": patched_runner["sha256"],
        "patched_launcher_sha256": patched_launcher["sha256"],
        "failed_generated_rows": failed["generated_rows"],
    }


def create_receipt() -> dict[str, Any]:
    if DEFAULT_RECEIPT.exists() or DEFAULT_RECEIPT.is_symlink():
        raise RemediationError(
            f"production remediation already exists: {DEFAULT_RECEIPT}"
        )
    original = audit_original_migration()
    failed = audit_failed_attempt()
    patched_runner = _validate_patched_runner_source(PATCHED_RUNNER)
    patched_launcher = _validate_patched_launcher_source(PATCHED_LAUNCHER)
    gpu_idle = audit_idle_gpus(failed)
    receipt = build_receipt(
        original_migration=original,
        failed_attempt=failed,
        gpu_idle=gpu_idle,
        patched_runner=patched_runner,
        patched_launcher=patched_launcher,
    )
    write_readonly_receipt(DEFAULT_RECEIPT, receipt)
    return validate_receipt(DEFAULT_RECEIPT)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--confirm", required=True)
    sub.add_parser("validate")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            if args.confirm != CONFIRMATION:
                raise RemediationError("exact remediation confirmation is required")
            result = create_receipt()
        else:
            result = validate_receipt()
    except (RemediationError, OSError, ValueError) as exc:
        print(
            _canonical(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
