"""Supervise two independent single-GPU vLLM K5 shard workers.

This process deliberately imports neither torch nor vLLM.  It atomically
leases both canonical GPU lock files before either child is created, starts
one process group per physical GPU, waits until both model engines publish
sealed readiness evidence, releases one shared GO barrier, and measures the
parallel wall clock through both sealed DONE records.  Only after both workers
exit successfully does it validate the shards and ask the generation runner
to create the canonical merged run.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import secrets
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, IO

from jobs.eval import seal_chk3_beta_core8_vllm_k5_dual_dp1_suite as suite
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)
from jobs.eval import remediate_chk3_beta_core8_vllm_k5_dp2_to_dual_dp1 as remediation


ROOT = Path(__file__).resolve().parents[2]
RUNNER_MODULE = "jobs.eval.eval_chk3_beta_core8_merged_vllm_k5_dual_dp1"
EVALUATION_ID = "chk3-beta-core8-merged-1993-2025-n2048-vllm-k5-dual-independent-dp1-v1"
GO_SCHEMA = "chk3-beta-core8-vllm-k5-dual-dp1-go-v1"
TIMING_SCHEMA = "chk3-beta-core8-vllm-k5-dual-dp1-orchestrator-timing-v1"
MODEL_ORDER = ("chk1", "chk3", "chk0")
SCOPES = ("infrastructure_smoke", "formal_merged_panel")
MAX_NUM_SEQS = (8, 12, 16)
GPU_LOCKS = (
    Path("/tmp/fomc_trainer_chk3_beta_core8_vllm_k5_gpu0.lock"),
    Path("/tmp/fomc_trainer_chk3_beta_core8_vllm_k5_gpu1.lock"),
)
GPU_IDENTITIES = {
    0: {
        "uuid": "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
        "pci_bus_id": "00000000:21:00.0",
    },
    1: {
        "uuid": "GPU-0d2b981c-7a91-ad4c-7015-89445a0283a1",
        "pci_bus_id": "00000000:e1:00.0",
    },
}
READY_TIMEOUT_SECONDS = 1800
POLL_SECONDS = 0.25
TERMINATION_GRACE_SECONDS = 30.0
POST_EXIT_GPU_PASSED_STATUS = "passed_while_canonical_gpu_locks_held"
POST_EXIT_GPU_UNAVAILABLE_STATUS = "unavailable_reconstructed_without_new_gpu_work"
POST_EXIT_GPU_IDLE_TIMEOUT_SECONDS = 60
POST_EXIT_GPU_IDLE_POLL_SECONDS = 5


class DualDp1OrchestrationError(RuntimeError):
    """The two-worker launch or shared timing contract failed."""


class GpuBusyError(DualDp1OrchestrationError):
    """The canonical GPU pair is temporarily unavailable."""


def _formal_authorization_evidence(
    *,
    formal_authorization: Path | None,
    cohort: Path,
    cohort_sha256: str,
    scope: str,
    max_num_seqs: int,
) -> dict[str, Any] | None:
    if scope != "formal_merged_panel":
        if formal_authorization is not None:
            raise DualDp1OrchestrationError(
                "formal authorization is forbidden outside formal generation"
            )
        return None
    if formal_authorization is None:
        raise DualDp1OrchestrationError(
            "formal generation requires the official sealed smoke-suite authorization"
        )
    try:
        loaded = suite.load_and_validate_suite(
            formal_authorization,
            cohort_path=cohort,
            cohort_sha256=cohort_sha256,
            expected_scope="infrastructure_smoke",
            max_num_seqs=max_num_seqs,
        )
    except Exception as exc:
        raise DualDp1OrchestrationError(
            f"official smoke-suite authorization failed deep validation: {exc}"
        ) from exc
    manifest = loaded["manifest"]
    gates = manifest.get("generation_gates")
    benchmark_selection = manifest.get("benchmark_selection")
    if (
        not isinstance(gates, Mapping)
        or gates.get("benchmark_candidate_exact_token_parity_8_12_16") is not True
        or gates.get("selected_chk1_smoke_exact_replay") is not True
        or gates.get("formal_generation_unblocked") is not True
        or not isinstance(benchmark_selection, Mapping)
    ):
        raise DualDp1OrchestrationError(
            "official smoke suite did not seal every formal-generation gate"
        )
    return {
        "official_smoke_suite": copy.deepcopy(loaded["manifest_binding"]),
        "benchmark_selection": copy.deepcopy(dict(benchmark_selection)),
        "generation_gates": copy.deepcopy(dict(gates)),
    }


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


def _binding(path: Path, *, payload_sha256: str | None = None) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise DualDp1OrchestrationError(f"bound path is a symlink: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise DualDp1OrchestrationError(f"bound file is missing: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }
    if payload_sha256 is not None:
        result["payload_sha256"] = payload_sha256
    return result


def _read_sealed_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DualDp1OrchestrationError(
            f"cannot read sealed JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise DualDp1OrchestrationError(f"sealed JSON is not an object: {path}")
    try:
        payload_sha = validate_manifest_integrity(value)
    except Exception as exc:
        raise DualDp1OrchestrationError(
            f"sealed JSON integrity failed: {path}"
        ) from exc
    return value, _binding(path, payload_sha256=payload_sha)


def _write_new_sealed_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    sealed = seal_manifest(dict(payload))
    unresolved = path.expanduser()
    if os.path.lexists(unresolved):
        raise DualDp1OrchestrationError(f"refusing to overwrite {unresolved}")
    unresolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(unresolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_canonical(sealed) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(unresolved.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if unresolved.exists():
            unresolved.unlink()
        raise
    return sealed


def _open_lock(path: Path) -> IO[str]:
    if path.is_symlink():
        raise DualDp1OrchestrationError(f"GPU lock is a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    handle = os.fdopen(descriptor, "a+", encoding="utf-8")
    path_stat = os.stat(path, follow_symlinks=False)
    fd_stat = os.fstat(handle.fileno())
    if (
        not stat.S_ISREG(path_stat.st_mode)
        or not stat.S_ISREG(fd_stat.st_mode)
        or path_stat.st_nlink != 1
        or fd_stat.st_nlink != 1
        or (path_stat.st_dev, path_stat.st_ino) != (fd_stat.st_dev, fd_stat.st_ino)
    ):
        handle.close()
        raise DualDp1OrchestrationError(f"GPU lock is not regular: {path}")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise DualDp1OrchestrationError(f"GPU lock is already held: {path}") from exc
    return handle


def _acquire_gpu_lock_pair(*, timeout_seconds: int, poll_seconds: int) -> list[IO[str]]:
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise DualDp1OrchestrationError("GPU wait/poll intervals must be positive")
    deadline = time.monotonic() + timeout_seconds
    next_notice = time.monotonic()
    while True:
        handles: list[IO[str]] = []
        try:
            for path in GPU_LOCKS:
                handles.append(_open_lock(path))
            return handles
        except (DualDp1OrchestrationError, OSError) as exc:
            for handle in reversed(handles):
                handle.close()
            now = time.monotonic()
            if now >= deadline:
                raise DualDp1OrchestrationError(
                    f"timed out acquiring the atomic GPU0/GPU1 lock pair: {exc}"
                ) from exc
            if now >= next_notice:
                print(
                    f"[{_utc_now()}] waiting for atomic GPU0/GPU1 lock pair: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                next_notice = now + 60.0
            time.sleep(min(float(poll_seconds), max(0.05, deadline - now)))


def _run_checked(
    command: Sequence[str], *, env: Mapping[str, str] | None = None
) -> str:
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
        env=None if env is None else dict(env),
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise DualDp1OrchestrationError(
            f"command failed ({completed.returncode}): {' '.join(command)}: {message}"
        )
    return completed.stdout


def _audit_idle_physical_gpus() -> list[dict[str, Any]]:
    gpu_output = _run_checked(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        )
    )
    rows: list[dict[str, Any]] = []
    for line in gpu_output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise DualDp1OrchestrationError("unexpected nvidia-smi GPU row")
        index = int(fields[0])
        row = {
            "physical_gpu_index": index,
            "uuid": fields[1],
            "pci_bus_id": fields[2].lower(),
            "memory_used_mib": int(fields[3]),
            "memory_total_mib": int(fields[4]),
            "utilization_gpu_percent": int(fields[5]),
        }
        rows.append(row)
    identities = {
        row["physical_gpu_index"]: {
            "uuid": row["uuid"],
            "pci_bus_id": row["pci_bus_id"],
        }
        for row in rows
    }
    if identities != GPU_IDENTITIES:
        raise DualDp1OrchestrationError("physical GPU identity drift")
    compute_output = _run_checked(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        )
    )
    if compute_output.strip():
        raise GpuBusyError(
            "both leased GPUs must have no foreign compute process before spawn"
        )
    if any(
        row["memory_used_mib"] > 64 or row["utilization_gpu_percent"] != 0
        for row in rows
    ):
        raise GpuBusyError("both leased GPUs must be idle before spawn")
    return rows


def _wait_for_two_idle_sample_records(
    *, timeout_seconds: int, poll_seconds: int
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout_seconds
    next_notice = time.monotonic()
    consecutive: list[dict[str, Any]] = []
    while len(consecutive) < 2:
        try:
            gpus = _audit_idle_physical_gpus()
            consecutive.append(
                {
                    "sampled_at_utc": _utc_now(),
                    "compute_processes": [],
                    "gpus": copy.deepcopy(gpus),
                }
            )
        except GpuBusyError as exc:
            consecutive.clear()
            now = time.monotonic()
            if now >= deadline:
                raise DualDp1OrchestrationError(
                    f"timed out waiting for two idle GPU-pair samples: {exc}"
                ) from exc
            if now >= next_notice:
                print(
                    f"[{_utc_now()}] GPU pair locked; waiting for idle state: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                next_notice = now + 60.0
        if len(consecutive) < 2:
            now = time.monotonic()
            if now >= deadline:
                raise DualDp1OrchestrationError(
                    "timed out waiting for two consecutive idle GPU-pair samples"
                )
            time.sleep(min(float(poll_seconds), max(0.05, deadline - now)))
    return consecutive


def _wait_for_two_idle_samples(
    *, timeout_seconds: int, poll_seconds: int
) -> list[dict[str, Any]]:
    records = _wait_for_two_idle_sample_records(
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )
    return copy.deepcopy(records[-1]["gpus"])


def _post_exit_gpu_evidence(
    sample_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    evidence = {
        "status": "passed",
        "claimed": True,
        "checked_while_canonical_gpu_locks_held": True,
        "two_consecutive_idle_samples": True,
        "no_compute_processes": True,
        "timeout_seconds": POST_EXIT_GPU_IDLE_TIMEOUT_SECONDS,
        "poll_seconds": POST_EXIT_GPU_IDLE_POLL_SECONDS,
        "samples": copy.deepcopy(list(sample_records)),
    }
    _validate_post_exit_gpu_evidence(evidence, reconstructed=False)
    return evidence


def _validate_post_exit_gpu_evidence(evidence: Any, *, reconstructed: bool) -> None:
    if reconstructed:
        if evidence is not None:
            raise DualDp1OrchestrationError(
                "reconstructed timing falsely claims a post-exit GPU audit"
            )
        return
    if (
        not isinstance(evidence, Mapping)
        or set(evidence)
        != {
            "status",
            "claimed",
            "checked_while_canonical_gpu_locks_held",
            "two_consecutive_idle_samples",
            "no_compute_processes",
            "timeout_seconds",
            "poll_seconds",
            "samples",
        }
        or evidence.get("status") != "passed"
        or evidence.get("claimed") is not True
        or evidence.get("checked_while_canonical_gpu_locks_held") is not True
        or evidence.get("two_consecutive_idle_samples") is not True
        or evidence.get("no_compute_processes") is not True
        or evidence.get("timeout_seconds") != POST_EXIT_GPU_IDLE_TIMEOUT_SECONDS
        or evidence.get("poll_seconds") != POST_EXIT_GPU_IDLE_POLL_SECONDS
        or not isinstance(evidence.get("samples"), list)
        or len(evidence["samples"]) != 2
    ):
        raise DualDp1OrchestrationError("post-exit GPU idle evidence is incomplete")
    for sample in evidence["samples"]:
        if (
            not isinstance(sample, Mapping)
            or set(sample) != {"sampled_at_utc", "compute_processes", "gpus"}
            or not isinstance(sample.get("sampled_at_utc"), str)
            or sample.get("compute_processes") != []
            or not isinstance(sample.get("gpus"), list)
            or len(sample["gpus"]) != 2
        ):
            raise DualDp1OrchestrationError("post-exit GPU idle sample drift")
        rows = sample["gpus"]
        identities = {
            row.get("physical_gpu_index"): {
                "uuid": row.get("uuid"),
                "pci_bus_id": str(row.get("pci_bus_id", "")).lower(),
            }
            for row in rows
            if isinstance(row, Mapping)
        }
        if identities != GPU_IDENTITIES or any(
            not isinstance(row, Mapping)
            or set(row)
            != {
                "physical_gpu_index",
                "uuid",
                "pci_bus_id",
                "memory_used_mib",
                "memory_total_mib",
                "utilization_gpu_percent",
            }
            or not isinstance(row.get("memory_used_mib"), int)
            or row["memory_used_mib"] > 64
            or row.get("utilization_gpu_percent") != 0
            for row in rows
        ):
            raise DualDp1OrchestrationError(
                "post-exit GPU identity/idleness evidence drift"
            )


def _worker_environment(
    *, physical_gpu_index: int, lock_fd: int, run_token: str
) -> dict[str, str]:
    environment = dict(os.environ)
    for key in tuple(environment):
        if (
            key.startswith("VLLM_DP_")
            or key.startswith("NCCL_")
            or key
            in {
                "CUDA_VISIBLE_DEVICES",
                "MASTER_ADDR",
                "MASTER_PORT",
                "WORLD_SIZE",
                "RANK",
                "LOCAL_RANK",
                "LOCAL_WORLD_SIZE",
                "GROUP_RANK",
                "ROLE_RANK",
            }
        ):
            environment.pop(key, None)
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(physical_gpu_index),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}",
            "VLLM_USE_V1": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "FOMC_PREACQUIRED_GPU_LOCK_FD": str(lock_fd),
            "FOMC_DUAL_DP1_RUN_TOKEN": run_token,
        }
    )
    return environment


def _runner_command(
    *,
    python_bin: Path,
    model_id: str,
    shard_id: int,
    cohort: Path,
    cohort_sha256: str,
    output_dir: Path,
    scope: str,
    max_num_seqs: int,
    control_dir: Path,
    run_token: str,
    lock_fd: int,
    resume: bool,
    gpu_wait_timeout_seconds: int,
    gpu_poll_seconds: int,
    formal_authorization: Path | None = None,
) -> list[str]:
    command = [
        str(python_bin),
        "-u",
        "-m",
        RUNNER_MODULE,
        "run-shard",
        "--model-id",
        model_id,
        "--shard-id",
        str(shard_id),
        "--physical-gpu-index",
        str(shard_id),
        "--cohort",
        str(cohort),
        "--cohort-sha256",
        cohort_sha256,
        "--output-dir",
        str(output_dir),
        "--scope",
        scope,
        "--max-num-seqs",
        str(max_num_seqs),
        "--control-dir",
        str(control_dir),
        "--run-token",
        run_token,
        "--inherited-lock-fd",
        str(lock_fd),
        "--gpu-wait-timeout-seconds",
        str(gpu_wait_timeout_seconds),
        "--gpu-poll-seconds",
        str(gpu_poll_seconds),
    ]
    if scope == "infrastructure_smoke":
        command.append("--smoke")
    if resume:
        command.append("--resume")
    if formal_authorization is not None:
        command.extend(("--formal-authorization", str(formal_authorization)))
    return command


def _terminate_process_groups(processes: Mapping[int, subprocess.Popen[str]]) -> None:
    groups = [process.pid for process in processes.values()]

    def group_exists(pgid: int) -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False

    for pgid in groups:
        try:
            os.killpg(pgid, signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    live_groups = [pgid for pgid in groups if group_exists(pgid)]
    while live_groups and time.monotonic() < deadline:
        live_groups = [pgid for pgid in groups if group_exists(pgid)]
        if live_groups:
            time.sleep(POLL_SECONDS)
    for pgid in live_groups:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    term_deadline = time.monotonic() + 5.0
    while time.monotonic() < term_deadline:
        live_groups = [pgid for pgid in groups if group_exists(pgid)]
        if not live_groups:
            break
        time.sleep(POLL_SECONDS)
    for pgid in live_groups:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes.values():
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    gpu_deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    while True:
        compute = _run_checked(
            (
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            )
        ).strip()
        if not compute:
            break
        if time.monotonic() >= gpu_deadline:
            raise DualDp1OrchestrationError(
                f"spawned worker cleanup left GPU compute processes: {compute}"
            )
        time.sleep(POLL_SECONDS)


def _validate_control_record(
    path: Path,
    *,
    status: str,
    run_token: str,
    model_id: str,
    shard_id: int,
    scope: str,
    max_num_seqs: int,
    formal_authorization: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value, binding = _read_sealed_json(path)
    expected_schema = (
        "chk3-beta-core8-vllm-k5-dual-dp1-ready-v1"
        if status == "ready"
        else "chk3-beta-core8-vllm-k5-dual-dp1-done-v1"
    )
    if (
        value.get("schema_version") != expected_schema
        or value.get("status") != status
        or value.get("evaluation_id") != EVALUATION_ID
        or value.get("run_token") != run_token
        or value.get("model_id") != model_id
        or value.get("shard_id") != shard_id
        or value.get("physical_gpu_index") != shard_id
        or value.get("evaluation_scope") != scope
        or value.get("max_num_seqs") != max_num_seqs
        or value.get("formal_authorization")
        != (
            None
            if formal_authorization is None
            else copy.deepcopy(dict(formal_authorization))
        )
    ):
        raise DualDp1OrchestrationError(f"control record contract drift: {path}")
    return value, binding


def _wait_for_ready(
    *,
    processes: Mapping[int, subprocess.Popen[str]],
    control_dir: Path,
    run_token: str,
    model_id: str,
    scope: str,
    max_num_seqs: int,
    timeout_seconds: int,
    formal_authorization: Mapping[str, Any] | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    deadline = time.monotonic() + timeout_seconds
    values: dict[int, dict[str, Any]] = {}
    bindings: dict[int, dict[str, Any]] = {}
    while len(values) < len(processes):
        for shard_id, process in processes.items():
            returncode = process.poll()
            if returncode is not None:
                raise DualDp1OrchestrationError(
                    f"shard{shard_id} exited before GO with status {returncode}"
                )
            path = control_dir / f"shard{shard_id}.ready.json"
            if shard_id not in values and path.is_file():
                value, binding = _validate_control_record(
                    path,
                    status="ready",
                    run_token=run_token,
                    model_id=model_id,
                    shard_id=shard_id,
                    scope=scope,
                    max_num_seqs=max_num_seqs,
                    formal_authorization=formal_authorization,
                )
                if (
                    value.get("worker_pid") != process.pid
                    or not isinstance(value.get("engine_pid"), int)
                    or value.get("gpu_uuid") != GPU_IDENTITIES[shard_id]["uuid"]
                    or str(value.get("gpu_pci_bus_id", "")).lower()
                    != GPU_IDENTITIES[shard_id]["pci_bus_id"]
                    or value.get("cuda_visible_devices") != str(shard_id)
                ):
                    raise DualDp1OrchestrationError(
                        f"shard{shard_id} readiness PID/GPU binding drift"
                    )
                values[shard_id] = value
                bindings[shard_id] = binding
        if len(values) == len(processes):
            break
        if time.monotonic() >= deadline:
            raise DualDp1OrchestrationError("timed out waiting for both engines ready")
        time.sleep(POLL_SECONDS)
    return values, bindings


def _release_go(
    *,
    control_dir: Path,
    run_token: str,
    model_id: str,
    scope: str,
    max_num_seqs: int,
    ready_bindings: Mapping[int, Mapping[str, Any]],
    formal_authorization: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    go_epoch_ns = time.time_ns()
    go_monotonic_ns = time.monotonic_ns()
    path = control_dir / "go.json"
    payload = _write_new_sealed_json(
        path,
        {
            "schema_version": GO_SCHEMA,
            "status": "released",
            "evaluation_id": EVALUATION_ID,
            "run_token": run_token,
            "model_id": model_id,
            "evaluation_scope": scope,
            "max_num_seqs": max_num_seqs,
            "physical_gpu_indexes": [0, 1],
            "formal_authorization": (
                None
                if formal_authorization is None
                else copy.deepcopy(dict(formal_authorization))
            ),
            "go_epoch_ns": go_epoch_ns,
            "go_monotonic_ns": go_monotonic_ns,
            "go_at_utc": _utc_now(),
            "ready_manifests": {
                f"shard{shard_id}": dict(ready_bindings[shard_id])
                for shard_id in sorted(ready_bindings)
            },
        },
    )
    return payload, _binding(
        path, payload_sha256=payload["integrity"]["payload_sha256"]
    )


def _wait_for_done(
    *,
    processes: Mapping[int, subprocess.Popen[str]],
    control_dir: Path,
    run_token: str,
    model_id: str,
    scope: str,
    max_num_seqs: int,
    formal_authorization: Mapping[str, Any] | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    while any(process.poll() is None for process in processes.values()):
        failures = {
            shard_id: process.returncode
            for shard_id, process in processes.items()
            if process.poll() not in (None, 0)
        }
        if failures:
            raise DualDp1OrchestrationError(f"worker failure after GO: {failures}")
        time.sleep(POLL_SECONDS)
    failures = {
        shard_id: process.returncode
        for shard_id, process in processes.items()
        if process.returncode != 0
    }
    if failures:
        raise DualDp1OrchestrationError(f"worker failure after GO: {failures}")
    values: dict[int, dict[str, Any]] = {}
    bindings: dict[int, dict[str, Any]] = {}
    for shard_id in processes:
        value, binding = _validate_control_record(
            control_dir / f"shard{shard_id}.done.json",
            status="done",
            run_token=run_token,
            model_id=model_id,
            shard_id=shard_id,
            scope=scope,
            max_num_seqs=max_num_seqs,
            formal_authorization=formal_authorization,
        )
        values[shard_id] = value
        bindings[shard_id] = binding
    return values, bindings


def _runner_validate_shard_command(
    *,
    python_bin: Path,
    model_root: Path,
    cohort: Path,
    cohort_sha256: str,
    model_id: str,
    shard_id: int,
    scope: str,
    max_num_seqs: int,
) -> list[str]:
    return [
        str(python_bin),
        "-u",
        "-m",
        RUNNER_MODULE,
        "validate-shard",
        "--manifest",
        str(model_root / f"shard{shard_id}/manifest.json"),
        "--cohort",
        str(cohort),
        "--cohort-sha256",
        cohort_sha256,
        "--model-id",
        model_id,
        "--shard-id",
        str(shard_id),
        "--scope",
        scope,
        "--max-num-seqs",
        str(max_num_seqs),
    ]


def _validate_shards(
    *,
    python_bin: Path,
    model_root: Path,
    cohort: Path,
    cohort_sha256: str,
    model_id: str,
    scope: str,
    max_num_seqs: int,
) -> None:
    for shard_id in (0, 1):
        _run_checked(
            _runner_validate_shard_command(
                python_bin=python_bin,
                model_root=model_root,
                cohort=cohort,
                cohort_sha256=cohort_sha256,
                model_id=model_id,
                shard_id=shard_id,
                scope=scope,
                max_num_seqs=max_num_seqs,
            )
        )


def _seal_timing(
    *,
    path: Path,
    model_id: str,
    scope: str,
    max_num_seqs: int,
    run_token: str,
    go: Mapping[str, Any],
    go_binding: Mapping[str, Any],
    ready_bindings: Mapping[int, Mapping[str, Any]],
    done_values: Mapping[int, Mapping[str, Any]],
    done_bindings: Mapping[int, Mapping[str, Any]],
    logs: Mapping[int, Path],
    gpu_idle_evidence: Sequence[Mapping[str, Any]],
    processes: Mapping[int, subprocess.Popen[str]],
    remediation_receipt: Mapping[str, Any],
    formal_authorization: Mapping[str, Any] | None = None,
    gpu_idle_after_worker_exit: Mapping[str, Any] | None = None,
    reconstruction_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    go_epoch_ns = int(go["go_epoch_ns"])
    go_monotonic_ns = int(go["go_monotonic_ns"])
    finish_epoch_ns = max(
        int(done_values[shard_id]["generation_finished_epoch_ns"])
        for shard_id in done_values
    )
    finish_monotonic_ns = max(
        int(done_values[shard_id]["generation_finished_monotonic_ns"])
        for shard_id in done_values
    )
    wall_seconds = (finish_monotonic_ns - go_monotonic_ns) / 1_000_000_000
    output_tokens = sum(
        int(done_values[shard_id]["generated_output_tokens"])
        for shard_id in done_values
    )
    if wall_seconds <= 0 or output_tokens <= 0:
        raise DualDp1OrchestrationError("invalid shared-GO timing")
    reconstructed = reconstruction_evidence is not None
    post_exit_evidence = (
        None if reconstructed else copy.deepcopy(gpu_idle_after_worker_exit)
    )
    _validate_post_exit_gpu_evidence(
        post_exit_evidence,
        reconstructed=reconstructed,
    )
    if any(
        int(done_values[shard_id].get("observed_go_epoch_ns", -1)) != go_epoch_ns
        or int(done_values[shard_id].get("observed_go_monotonic_ns", -1))
        != go_monotonic_ns
        or int(done_values[shard_id].get("generation_started_monotonic_ns", -1))
        < go_monotonic_ns
        or int(done_values[shard_id].get("resume_count", -1)) != 0
        for shard_id in done_values
    ):
        raise DualDp1OrchestrationError(
            "fresh paired run did not preserve the exact shared GO/resume=0 contract"
        )
    payload = {
        "schema_version": TIMING_SCHEMA,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "evaluation_id": EVALUATION_ID,
        "model_id": model_id,
        "evaluation_scope": scope,
        "max_num_seqs_per_engine": max_num_seqs,
        "run_token": run_token,
        "remediation_receipt": dict(remediation_receipt),
        "formal_authorization": (
            None
            if formal_authorization is None
            else copy.deepcopy(dict(formal_authorization))
        ),
        "parallel_topology": {
            "independent_engine_count": 2,
            "data_parallel_size_per_engine": 1,
            "tensor_parallel_size_per_engine": 1,
            "physical_gpu_indices": [0, 1],
            "cross_gpu_collectives": False,
            "shared_go_barrier": True,
        },
        "gpu_idle_before_spawn": (
            [dict(row) for row in gpu_idle_evidence]
            if reconstruction_evidence is None
            else None
        ),
        "gpu_idle_after_worker_exit": post_exit_evidence,
        "gpu_idle_after_worker_exit_claimed": not reconstructed,
        "gpu_idle_after_worker_exit_status": (
            POST_EXIT_GPU_UNAVAILABLE_STATUS
            if reconstructed
            else POST_EXIT_GPU_PASSED_STATUS
        ),
        "timing_reconstruction": (
            None
            if reconstruction_evidence is None
            else copy.deepcopy(dict(reconstruction_evidence))
        ),
        "worker_processes": {
            f"shard{shard_id}": {
                "supervised_process_group_id": processes[shard_id].pid,
                "physical_gpu_index": shard_id,
            }
            for shard_id in (0, 1)
        },
        "control_artifacts": {
            "go": dict(go_binding),
            "ready": {
                f"shard{shard_id}": dict(ready_bindings[shard_id])
                for shard_id in (0, 1)
            },
            "done": {
                f"shard{shard_id}": dict(done_bindings[shard_id]) for shard_id in (0, 1)
            },
            "worker_console_logs": {
                f"shard{shard_id}": _binding(logs[shard_id]) for shard_id in (0, 1)
            },
        },
        "shared_timing": {
            "start_definition": "parent_released_go_after_both_engines_ready",
            "finish_definition": "maximum_of_two_worker_generation_done_timestamps",
            "go_epoch_ns": go_epoch_ns,
            "finish_epoch_ns": finish_epoch_ns,
            "go_monotonic_ns": go_monotonic_ns,
            "finish_monotonic_ns": finish_monotonic_ns,
            "parallel_generation_wall_seconds": wall_seconds,
            "aggregate_generated_output_tokens": output_tokens,
            "aggregate_output_tokens_per_second": output_tokens / wall_seconds,
            "speed_measurement_valid_for_candidate_selection": True,
            "both_workers_resume_count": [0, 0],
        },
    }
    return _write_new_sealed_json(path, payload)


def _completed_shard_evidence(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest, manifest_binding = _read_sealed_json(manifest_path)
    runtime = manifest.get("runtime")
    engine = runtime.get("engine_runtime") if isinstance(runtime, Mapping) else None
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(engine, Mapping)
        or not isinstance(runtime.get("resume_count"), int)
        or not isinstance(engine.get("ready_manifest"), Mapping)
        or not isinstance(engine.get("go_manifest"), Mapping)
        or not isinstance(engine.get("done_manifest"), Mapping)
        or not isinstance(engine.get("canonical_generated_output_tokens"), int)
        or not isinstance(engine.get("generated_output_tokens"), int)
        or not isinstance(engine.get("completed_requests_in_timed_session"), int)
        or not isinstance(engine.get("engine_session_resume_count"), int)
    ):
        raise DualDp1OrchestrationError("sealed shard recovery evidence is missing")
    return {
        "resume_count": int(runtime["resume_count"]),
        "ready": dict(engine["ready_manifest"]),
        "go": dict(engine["go_manifest"]),
        "done": dict(engine["done_manifest"]),
        "canonical_generated_output_tokens": int(
            engine["canonical_generated_output_tokens"]
        ),
        "timed_generated_output_tokens": int(engine["generated_output_tokens"]),
        "timed_completed_requests": int(engine["completed_requests_in_timed_session"]),
        "engine_session_resume_count": int(engine["engine_session_resume_count"]),
    }, manifest_binding


def _validate_completed_control_evidence(
    *,
    model_id: str,
    scope: str,
    max_num_seqs: int,
    shard_id: int,
    evidence: Mapping[str, Any],
    ready: Mapping[str, Any],
    go: Mapping[str, Any],
    done: Mapping[str, Any],
    formal_authorization: Mapping[str, Any] | None,
) -> None:
    common = {
        "evaluation_id": EVALUATION_ID,
        "model_id": model_id,
        "evaluation_scope": scope,
        "max_num_seqs": max_num_seqs,
    }
    if any(ready.get(key) != expected for key, expected in common.items()) or any(
        done.get(key) != expected for key, expected in common.items()
    ):
        raise DualDp1OrchestrationError("reconstructed shard control identity drift")
    if any(go.get(key) != expected for key, expected in common.items()):
        raise DualDp1OrchestrationError("reconstructed GO identity drift")
    ready_binding = evidence["ready"]
    ready_manifests = go.get("ready_manifests")
    expected_authorization = (
        None
        if formal_authorization is None
        else copy.deepcopy(dict(formal_authorization))
    )
    if (
        ready.get("schema_version") != "chk3-beta-core8-vllm-k5-dual-dp1-ready-v1"
        or ready.get("status") != "ready"
        or ready.get("shard_id") != shard_id
        or ready.get("physical_gpu_index") != shard_id
        or ready.get("gpu_uuid") != GPU_IDENTITIES[shard_id]["uuid"]
        or str(ready.get("gpu_pci_bus_id", "")).lower()
        != GPU_IDENTITIES[shard_id]["pci_bus_id"]
        or ready.get("cuda_visible_devices") != str(shard_id)
        or ready.get("formal_authorization") != expected_authorization
        or not isinstance(ready.get("worker_pid"), int)
        or not isinstance(ready.get("engine_pid"), int)
        or go.get("schema_version") != GO_SCHEMA
        or go.get("status") != "released"
        or go.get("physical_gpu_indexes") != [0, 1]
        or go.get("formal_authorization") != expected_authorization
        or not isinstance(ready_manifests, Mapping)
        or set(ready_manifests) != {"shard0", "shard1"}
        or ready_manifests.get(f"shard{shard_id}") != ready_binding
        or done.get("schema_version") != "chk3-beta-core8-vllm-k5-dual-dp1-done-v1"
        or done.get("status") != "done"
        or done.get("shard_id") != shard_id
        or done.get("physical_gpu_index") != shard_id
        or done.get("formal_authorization") != expected_authorization
        or ready.get("run_token") != go.get("run_token")
        or done.get("run_token") != go.get("run_token")
        or int(done.get("observed_go_epoch_ns", -1)) != int(go.get("go_epoch_ns", -2))
        or int(done.get("observed_go_monotonic_ns", -1))
        != int(go.get("go_monotonic_ns", -2))
        or int(done.get("generation_started_monotonic_ns", -1))
        < int(go.get("go_monotonic_ns", 0))
        or int(done.get("generation_finished_monotonic_ns", -1))
        <= int(done.get("generation_started_monotonic_ns", -1))
        or done.get("generated_output_tokens")
        != evidence["timed_generated_output_tokens"]
        or done.get("completed_requests") != evidence["timed_completed_requests"]
        or done.get("resume_count") != evidence["engine_session_resume_count"]
        or evidence["resume_count"] != evidence["engine_session_resume_count"]
        or int(evidence["canonical_generated_output_tokens"]) <= 0
    ):
        raise DualDp1OrchestrationError(
            "reconstructed shard READY/GO/DONE accounting drift"
        )


def _seal_recovery_timing(
    *,
    path: Path,
    model_id: str,
    max_num_seqs: int,
    run_token: str,
    go: Mapping[str, Any],
    go_binding: Mapping[str, Any],
    launched_shards: Sequence[int],
    preexisting_complete_shards: Sequence[int],
    done_values: Mapping[int, Mapping[str, Any]],
    logs: Mapping[int, Path],
    gpu_idle_evidence: Sequence[Mapping[str, Any]],
    processes: Mapping[int, subprocess.Popen[str]],
    final_evidence: Mapping[int, Mapping[str, Any]],
    shard_manifest_bindings: Mapping[int, Mapping[str, Any]],
    remediation_receipt: Mapping[str, Any],
    formal_authorization: Mapping[str, Any],
    gpu_idle_after_worker_exit: Mapping[str, Any] | None = None,
    reconstruction_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    launched = sorted(launched_shards)
    preexisting = sorted(preexisting_complete_shards)
    if not launched or set(launched) | set(preexisting) != {0, 1}:
        raise DualDp1OrchestrationError("formal recovery shard partition is invalid")
    go_monotonic_ns = int(go["go_monotonic_ns"])
    finish_monotonic_ns = max(
        int(done_values[shard_id]["generation_finished_monotonic_ns"])
        for shard_id in launched
    )
    wall_seconds = (finish_monotonic_ns - go_monotonic_ns) / 1_000_000_000
    full_tokens = sum(
        int(final_evidence[shard_id]["canonical_generated_output_tokens"])
        for shard_id in (0, 1)
    )
    attempt_tokens = sum(
        int(done_values[shard_id]["generated_output_tokens"]) for shard_id in launched
    )
    if wall_seconds <= 0 or full_tokens <= 0 or attempt_tokens <= 0:
        raise DualDp1OrchestrationError("formal recovery timing is invalid")
    reconstructed = reconstruction_evidence is not None
    post_exit_evidence = (
        None if reconstructed else copy.deepcopy(gpu_idle_after_worker_exit)
    )
    _validate_post_exit_gpu_evidence(
        post_exit_evidence,
        reconstructed=reconstructed,
    )
    resume_counts = [int(final_evidence[s]["resume_count"]) for s in (0, 1)]
    if any(
        done_values[shard_id].get("run_token") != run_token
        or int(done_values[shard_id].get("observed_go_epoch_ns", -1))
        != int(go["go_epoch_ns"])
        or int(done_values[shard_id].get("observed_go_monotonic_ns", -1))
        != go_monotonic_ns
        or int(done_values[shard_id].get("generation_started_monotonic_ns", -1))
        < go_monotonic_ns
        or int(done_values[shard_id].get("generation_finished_monotonic_ns", -1))
        <= int(done_values[shard_id].get("generation_started_monotonic_ns", -1))
        or int(done_values[shard_id].get("resume_count", -1))
        != int(final_evidence[shard_id]["resume_count"])
        for shard_id in launched
    ):
        raise DualDp1OrchestrationError("formal recovery DONE/GO contract drift")
    mode = "paired_resume" if launched == [0, 1] else "asymmetric_resume"
    payload = {
        "schema_version": TIMING_SCHEMA,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "evaluation_id": EVALUATION_ID,
        "model_id": model_id,
        "evaluation_scope": "formal_merged_panel",
        "max_num_seqs_per_engine": max_num_seqs,
        "run_token": run_token,
        "remediation_receipt": dict(remediation_receipt),
        "formal_authorization": copy.deepcopy(dict(formal_authorization)),
        "parallel_topology": {
            "independent_engine_count": 2,
            "data_parallel_size_per_engine": 1,
            "tensor_parallel_size_per_engine": 1,
            "physical_gpu_indices": [0, 1],
            "cross_gpu_collectives": False,
            "shared_go_barrier": True,
        },
        "gpu_idle_before_spawn": (
            [dict(row) for row in gpu_idle_evidence]
            if reconstruction_evidence is None
            else None
        ),
        "gpu_idle_after_worker_exit": post_exit_evidence,
        "gpu_idle_after_worker_exit_claimed": not reconstructed,
        "gpu_idle_after_worker_exit_status": (
            POST_EXIT_GPU_UNAVAILABLE_STATUS
            if reconstructed
            else POST_EXIT_GPU_PASSED_STATUS
        ),
        "timing_reconstruction": (
            None
            if reconstruction_evidence is None
            else copy.deepcopy(dict(reconstruction_evidence))
        ),
        "worker_processes": {
            f"shard{shard_id}": {
                "supervised_process_group_id": processes[shard_id].pid,
                "physical_gpu_index": shard_id,
            }
            for shard_id in launched
        },
        "control_artifacts": {
            "go": dict(go_binding),
            "ready": {f"shard{s}": dict(final_evidence[s]["ready"]) for s in (0, 1)},
            "done": {f"shard{s}": dict(final_evidence[s]["done"]) for s in (0, 1)},
            "shard_manifests": {
                f"shard{s}": dict(shard_manifest_bindings[s]) for s in (0, 1)
            },
            "worker_console_logs": {f"shard{s}": _binding(logs[s]) for s in launched},
        },
        "recovery": {
            "mode": mode,
            "launched_shards": launched,
            "preexisting_complete_shards": preexisting,
            "selection_eligible": False,
        },
        "shared_timing": {
            "start_definition": "parent_released_current_recovery_go",
            "finish_definition": "maximum_current_launched_worker_done_monotonic_time",
            "go_epoch_ns": int(go["go_epoch_ns"]),
            "go_monotonic_ns": go_monotonic_ns,
            "finish_monotonic_ns": finish_monotonic_ns,
            "recovery_attempt_wall_seconds": wall_seconds,
            "aggregate_generated_output_tokens": full_tokens,
            "attempt_generated_output_tokens": attempt_tokens,
            "aggregate_output_tokens_per_second": None,
            "speed_measurement_valid_for_candidate_selection": False,
            "both_workers_resume_count": resume_counts,
        },
    }
    return _write_new_sealed_json(path, payload)


def _validate_timing(
    path: Path,
    *,
    formal_authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value, _ = _read_sealed_json(path)
    topology = value.get("parallel_topology")
    timing = value.get("shared_timing")
    loaded_receipt = remediation.load_and_validate_receipt(remediation.DEFAULT_RECEIPT)
    receipt_binding = loaded_receipt["receipt_binding"]
    if (
        value.get("schema_version") != TIMING_SCHEMA
        or value.get("status") != "complete"
        or not isinstance(topology, Mapping)
        or topology.get("independent_engine_count") != 2
        or topology.get("data_parallel_size_per_engine") != 1
        or topology.get("cross_gpu_collectives") is not False
        or topology.get("shared_go_barrier") is not True
        or not isinstance(timing, Mapping)
        or value.get("remediation_receipt") != receipt_binding
        or value.get("formal_authorization")
        != (
            None
            if formal_authorization is None
            else copy.deepcopy(dict(formal_authorization))
        )
    ):
        raise DualDp1OrchestrationError("orchestrator timing contract drift")
    reconstructed = value.get("timing_reconstruction") is not None
    _validate_post_exit_gpu_evidence(
        value.get("gpu_idle_after_worker_exit"),
        reconstructed=reconstructed,
    )
    expected_post_exit_status = (
        POST_EXIT_GPU_UNAVAILABLE_STATUS
        if reconstructed
        else POST_EXIT_GPU_PASSED_STATUS
    )
    if value.get("gpu_idle_after_worker_exit_status") != expected_post_exit_status:
        raise DualDp1OrchestrationError("post-exit GPU audit status drift")
    if value.get("gpu_idle_after_worker_exit_claimed") is not (not reconstructed):
        raise DualDp1OrchestrationError("post-exit GPU audit claim drift")
    if reconstructed != (value.get("gpu_idle_before_spawn") is None):
        raise DualDp1OrchestrationError(
            "orchestrator reconstruction GPU-evidence contract drift"
        )
    eligible = timing.get("speed_measurement_valid_for_candidate_selection")
    if eligible is True:
        if (
            timing.get("both_workers_resume_count") != [0, 0]
            or float(timing.get("parallel_generation_wall_seconds", 0)) <= 0
            or float(timing.get("aggregate_output_tokens_per_second", 0)) <= 0
            or value.get("recovery") is not None
        ):
            raise DualDp1OrchestrationError("fresh orchestrator timing drift")
    elif (
        eligible is not False
        or value.get("evaluation_scope") != "formal_merged_panel"
        or float(timing.get("recovery_attempt_wall_seconds", 0)) <= 0
        or timing.get("aggregate_output_tokens_per_second") is not None
        or not isinstance(value.get("recovery"), Mapping)
    ):
        raise DualDp1OrchestrationError("recovery orchestrator timing drift")
    go_binding = value.get("control_artifacts", {}).get("go")
    if not isinstance(go_binding, Mapping):
        raise DualDp1OrchestrationError("orchestrator GO binding is missing")
    observed_go = _binding(Path(str(go_binding.get("path"))))
    if any(
        go_binding.get(key) != observed_go.get(key)
        for key in ("path", "bytes", "sha256")
    ):
        raise DualDp1OrchestrationError("orchestrator GO artifact drift")
    for section in ("ready", "done", "worker_console_logs"):
        for binding in value.get("control_artifacts", {}).get(section, {}).values():
            observed = _binding(Path(str(binding.get("path"))))
            if any(
                binding.get(key) != observed.get(key)
                for key in ("path", "bytes", "sha256")
            ):
                raise DualDp1OrchestrationError("orchestrator timing artifact drift")
    return value


def _read_control_binding(
    binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(str(binding.get("path")))
    value, observed = _read_sealed_json(path)
    if dict(binding) != observed:
        raise DualDp1OrchestrationError("sealed shard control binding drift")
    return value, observed


def _reconstruct_completed_pair_timing(
    *,
    timing_path: Path,
    model_id: str,
    scope: str,
    max_num_seqs: int,
    shard_manifests: Sequence[Path],
    remediation_receipt: Mapping[str, Any],
    formal_authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if (scope == "formal_merged_panel") != (formal_authorization is not None):
        raise DualDp1OrchestrationError(
            "timing reconstruction formal-authorization scope drift"
        )
    final_evidence: dict[int, dict[str, Any]] = {}
    manifest_bindings: dict[int, dict[str, Any]] = {}
    ready_values: dict[int, dict[str, Any]] = {}
    go_values: dict[int, dict[str, Any]] = {}
    done_values: dict[int, dict[str, Any]] = {}
    for shard_id in (0, 1):
        evidence, manifest_binding = _completed_shard_evidence(
            shard_manifests[shard_id]
        )
        final_evidence[shard_id] = evidence
        manifest_bindings[shard_id] = manifest_binding
        ready_values[shard_id], _ = _read_control_binding(evidence["ready"])
        go_values[shard_id], _ = _read_control_binding(evidence["go"])
        done_values[shard_id], _ = _read_control_binding(evidence["done"])
        _validate_completed_control_evidence(
            model_id=model_id,
            scope=scope,
            max_num_seqs=max_num_seqs,
            shard_id=shard_id,
            evidence=evidence,
            ready=ready_values[shard_id],
            go=go_values[shard_id],
            done=done_values[shard_id],
            formal_authorization=formal_authorization,
        )
    go_keys = {
        shard_id: _canonical(final_evidence[shard_id]["go"]) for shard_id in (0, 1)
    }
    unique_go = set(go_keys.values())
    if len(unique_go) == 1:
        launched = [0, 1]
    else:
        latest_shard = max(
            (0, 1), key=lambda shard_id: int(go_values[shard_id]["go_epoch_ns"])
        )
        launched = [latest_shard]
    go_binding = dict(final_evidence[launched[0]]["go"])
    go, _ = _read_control_binding(go_binding)
    control_dir = Path(str(go_binding["path"])).parent
    logs = {
        shard_id: control_dir / f"shard{shard_id}.console.log" for shard_id in launched
    }
    if any(not path.is_file() or path.is_symlink() for path in logs.values()):
        raise DualDp1OrchestrationError(
            "cannot reconstruct timing without sealed-at-source worker logs"
        )
    processes = {
        shard_id: SimpleNamespace(pid=int(ready_values[shard_id]["worker_pid"]))
        for shard_id in launched
    }
    fresh = launched == [0, 1] and [
        final_evidence[s]["resume_count"] for s in (0, 1)
    ] == [0, 0]
    if fresh:
        return _seal_timing(
            path=timing_path,
            model_id=model_id,
            scope=scope,
            max_num_seqs=max_num_seqs,
            run_token=str(ready_values[0]["run_token"]),
            go=go,
            go_binding=go_binding,
            ready_bindings={s: final_evidence[s]["ready"] for s in (0, 1)},
            done_values=done_values,
            done_bindings={s: final_evidence[s]["done"] for s in (0, 1)},
            logs=logs,
            gpu_idle_evidence=[],
            processes=processes,
            remediation_receipt=remediation_receipt,
            formal_authorization=formal_authorization,
            reconstruction_evidence={
                "reconstructed_from_two_deeply_validated_sealed_shards": True,
                "no_new_gpu_work": True,
            },
        )
    preexisting = sorted({0, 1} - set(launched))
    return _seal_recovery_timing(
        path=timing_path,
        model_id=model_id,
        max_num_seqs=max_num_seqs,
        run_token=str(ready_values[launched[0]]["run_token"]),
        go=go,
        go_binding=go_binding,
        launched_shards=launched,
        preexisting_complete_shards=preexisting,
        done_values={s: done_values[s] for s in launched},
        logs=logs,
        gpu_idle_evidence=[],
        processes=processes,
        final_evidence=final_evidence,
        shard_manifest_bindings=manifest_bindings,
        remediation_receipt=remediation_receipt,
        formal_authorization=formal_authorization or {},
        reconstruction_evidence={
            "reconstructed_from_two_deeply_validated_sealed_shards": True,
            "no_new_gpu_work": True,
        },
    )


def _validate_run_command(
    *,
    python_bin: Path,
    model_root: Path,
    cohort: Path,
    cohort_sha256: str,
    model_id: str,
    scope: str,
    max_num_seqs: int,
) -> list[str]:
    return [
        str(python_bin),
        "-u",
        "-m",
        RUNNER_MODULE,
        "validate-run",
        "--manifest",
        str(model_root / "manifest.json"),
        "--cohort",
        str(cohort),
        "--cohort-sha256",
        cohort_sha256,
        "--model-id",
        model_id,
        "--scope",
        scope,
        "--max-num-seqs",
        str(max_num_seqs),
    ]


def run_model(
    *,
    python_bin: Path,
    model_id: str,
    cohort: Path,
    cohort_sha256: str,
    model_root: Path,
    scope: str,
    max_num_seqs: int,
    allow_resume: bool,
    gpu_wait_timeout_seconds: int,
    gpu_poll_seconds: int,
    ready_timeout_seconds: int,
    formal_authorization: Path | None = None,
) -> dict[str, Any]:
    authorization_evidence = _formal_authorization_evidence(
        formal_authorization=formal_authorization,
        cohort=cohort,
        cohort_sha256=cohort_sha256,
        scope=scope,
        max_num_seqs=max_num_seqs,
    )
    loaded_remediation = remediation.load_and_validate_receipt(
        remediation.DEFAULT_RECEIPT
    )
    remediation_binding = loaded_remediation["receipt_binding"]
    unresolved_model_root = model_root.expanduser()
    if unresolved_model_root.is_symlink():
        raise DualDp1OrchestrationError("model output root is a symlink")
    model_root = unresolved_model_root.resolve()
    merged_manifest = model_root / "manifest.json"
    timing_path = model_root / "orchestrator.parallel_timing.v1.json"
    if merged_manifest.is_file():
        _run_checked(
            _validate_run_command(
                python_bin=python_bin,
                model_root=model_root,
                cohort=cohort,
                cohort_sha256=cohort_sha256,
                model_id=model_id,
                scope=scope,
                max_num_seqs=max_num_seqs,
            )
        )
        timing = _validate_timing(
            timing_path, formal_authorization=authorization_evidence
        )
        return {"status": "valid", "timing": timing}
    if merged_manifest.exists() or merged_manifest.is_symlink():
        raise DualDp1OrchestrationError("unsafe merged manifest path")

    shard_dirs = [model_root / "shard0", model_root / "shard1"]
    shard_manifests = [directory / "manifest.json" for directory in shard_dirs]
    if any(directory.is_symlink() for directory in shard_dirs):
        raise DualDp1OrchestrationError("shard output directory is a symlink")
    any_shard_state = any(os.path.lexists(path) for path in shard_dirs)
    complete_shards = {
        shard_id for shard_id, path in enumerate(shard_manifests) if path.is_file()
    }
    all_shards_complete = complete_shards == {0, 1}
    if scope == "infrastructure_smoke" and any_shard_state and not all_shards_complete:
        raise DualDp1OrchestrationError(
            "benchmark/smoke requires a fresh shard pair or an exact complete pair"
        )
    if all_shards_complete:
        _validate_shards(
            python_bin=python_bin,
            model_root=model_root,
            cohort=cohort,
            cohort_sha256=cohort_sha256,
            model_id=model_id,
            scope=scope,
            max_num_seqs=max_num_seqs,
        )
        if timing_path.is_file():
            _validate_timing(timing_path, formal_authorization=authorization_evidence)
        elif timing_path.exists() or timing_path.is_symlink():
            raise DualDp1OrchestrationError("unsafe orchestrator timing path")
        else:
            _reconstruct_completed_pair_timing(
                timing_path=timing_path,
                model_id=model_id,
                scope=scope,
                max_num_seqs=max_num_seqs,
                shard_manifests=shard_manifests,
                remediation_receipt=remediation_binding,
                formal_authorization=authorization_evidence,
            )
    else:
        if timing_path.exists() or timing_path.is_symlink():
            raise DualDp1OrchestrationError(
                "orchestrator timing exists before both shards are complete"
            )
        if any_shard_state and not allow_resume:
            raise DualDp1OrchestrationError("partial shards cannot enter a fresh run")
        if any_shard_state and scope != "formal_merged_panel":
            raise DualDp1OrchestrationError(
                "only formal generation may resume a partial dual-DP1 run"
            )
        model_root.mkdir(parents=True, exist_ok=True)
        for shard_id in complete_shards:
            _run_checked(
                _runner_validate_shard_command(
                    python_bin=python_bin,
                    model_root=model_root,
                    cohort=cohort,
                    cohort_sha256=cohort_sha256,
                    model_id=model_id,
                    shard_id=shard_id,
                    scope=scope,
                    max_num_seqs=max_num_seqs,
                )
            )
        pending_shards = sorted({0, 1} - complete_shards)
        resume_shards = {
            shard_id
            for shard_id in pending_shards
            if os.path.lexists(shard_dirs[shard_id])
        }
        if resume_shards and not allow_resume:
            raise DualDp1OrchestrationError("formal shard resume was not authorized")
        run_token = secrets.token_hex(32)
        control_root = model_root / ".orchestrator_control"
        if control_root.is_symlink():
            raise DualDp1OrchestrationError("orchestrator control root is a symlink")
        existing_attempts = (
            list(control_root.iterdir()) if control_root.is_dir() else []
        )
        if any(path.is_symlink() or not path.is_dir() for path in existing_attempts):
            raise DualDp1OrchestrationError("unsafe prior orchestrator control attempt")
        control_dir = control_root / (
            f"attempt_{len(existing_attempts):04d}_{run_token}"
        )
        control_dir.mkdir(parents=True, exist_ok=False)
        logs = {
            shard_id: control_dir / f"shard{shard_id}.console.log"
            for shard_id in pending_shards
        }
        processes: dict[int, subprocess.Popen[str]] = {}
        log_handles: dict[int, IO[str]] = {}
        with ExitStack() as stack:
            acquired = _acquire_gpu_lock_pair(
                timeout_seconds=gpu_wait_timeout_seconds,
                poll_seconds=gpu_poll_seconds,
            )
            lock_handles = [stack.enter_context(handle) for handle in acquired]
            gpu_idle = _wait_for_two_idle_samples(
                timeout_seconds=gpu_wait_timeout_seconds,
                poll_seconds=gpu_poll_seconds,
            )
            try:
                for shard_id in pending_shards:
                    log_handle = stack.enter_context(
                        logs[shard_id].open("x", encoding="utf-8")
                    )
                    log_handles[shard_id] = log_handle
                    lock_fd = lock_handles[shard_id].fileno()
                    processes[shard_id] = subprocess.Popen(
                        _runner_command(
                            python_bin=python_bin,
                            model_id=model_id,
                            shard_id=shard_id,
                            cohort=cohort,
                            cohort_sha256=cohort_sha256,
                            output_dir=shard_dirs[shard_id],
                            scope=scope,
                            max_num_seqs=max_num_seqs,
                            control_dir=control_dir,
                            run_token=run_token,
                            lock_fd=lock_fd,
                            resume=shard_id in resume_shards,
                            gpu_wait_timeout_seconds=gpu_wait_timeout_seconds,
                            gpu_poll_seconds=gpu_poll_seconds,
                            formal_authorization=formal_authorization,
                        ),
                        stdin=subprocess.DEVNULL,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=_worker_environment(
                            physical_gpu_index=shard_id,
                            lock_fd=lock_fd,
                            run_token=run_token,
                        ),
                        pass_fds=(lock_fd,),
                        start_new_session=True,
                    )
                ready_values, current_ready_bindings = _wait_for_ready(
                    processes=processes,
                    control_dir=control_dir,
                    run_token=run_token,
                    model_id=model_id,
                    scope=scope,
                    max_num_seqs=max_num_seqs,
                    timeout_seconds=ready_timeout_seconds,
                    formal_authorization=authorization_evidence,
                )
                del ready_values
                ready_bindings: dict[int, dict[str, Any]] = {
                    shard_id: dict(binding)
                    for shard_id, binding in current_ready_bindings.items()
                }
                for shard_id in complete_shards:
                    evidence, _ = _completed_shard_evidence(shard_manifests[shard_id])
                    ready_bindings[shard_id] = dict(evidence["ready"])
                go, go_binding = _release_go(
                    control_dir=control_dir,
                    run_token=run_token,
                    model_id=model_id,
                    scope=scope,
                    max_num_seqs=max_num_seqs,
                    ready_bindings=ready_bindings,
                    formal_authorization=authorization_evidence,
                )
                done_values, current_done_bindings = _wait_for_done(
                    processes=processes,
                    control_dir=control_dir,
                    run_token=run_token,
                    model_id=model_id,
                    scope=scope,
                    max_num_seqs=max_num_seqs,
                    formal_authorization=authorization_evidence,
                )
                for handle in log_handles.values():
                    handle.flush()
                    os.fsync(handle.fileno())
                _validate_shards(
                    python_bin=python_bin,
                    model_root=model_root,
                    cohort=cohort,
                    cohort_sha256=cohort_sha256,
                    model_id=model_id,
                    scope=scope,
                    max_num_seqs=max_num_seqs,
                )
                gpu_idle_after_worker_exit = _post_exit_gpu_evidence(
                    _wait_for_two_idle_sample_records(
                        timeout_seconds=POST_EXIT_GPU_IDLE_TIMEOUT_SECONDS,
                        poll_seconds=POST_EXIT_GPU_IDLE_POLL_SECONDS,
                    )
                )
                final_evidence: dict[int, dict[str, Any]] = {}
                shard_manifest_bindings: dict[int, dict[str, Any]] = {}
                for shard_id in (0, 1):
                    evidence, manifest_binding = _completed_shard_evidence(
                        shard_manifests[shard_id]
                    )
                    final_evidence[shard_id] = evidence
                    shard_manifest_bindings[shard_id] = manifest_binding
                fresh_pair = (
                    pending_shards == [0, 1]
                    and not resume_shards
                    and not complete_shards
                )
                if fresh_pair:
                    _seal_timing(
                        path=timing_path,
                        model_id=model_id,
                        scope=scope,
                        max_num_seqs=max_num_seqs,
                        run_token=run_token,
                        go=go,
                        go_binding=go_binding,
                        ready_bindings=ready_bindings,
                        done_values=done_values,
                        done_bindings=current_done_bindings,
                        logs=logs,
                        gpu_idle_evidence=gpu_idle,
                        processes=processes,
                        remediation_receipt=remediation_binding,
                        formal_authorization=authorization_evidence,
                        gpu_idle_after_worker_exit=gpu_idle_after_worker_exit,
                    )
                else:
                    _seal_recovery_timing(
                        path=timing_path,
                        model_id=model_id,
                        max_num_seqs=max_num_seqs,
                        run_token=run_token,
                        go=go,
                        go_binding=go_binding,
                        launched_shards=pending_shards,
                        preexisting_complete_shards=sorted(complete_shards),
                        done_values=done_values,
                        logs=logs,
                        gpu_idle_evidence=gpu_idle,
                        processes=processes,
                        final_evidence=final_evidence,
                        shard_manifest_bindings=shard_manifest_bindings,
                        remediation_receipt=remediation_binding,
                        formal_authorization=authorization_evidence or {},
                        gpu_idle_after_worker_exit=gpu_idle_after_worker_exit,
                    )
            except BaseException:
                _terminate_process_groups(processes)
                raise

    _run_checked(
        [
            str(python_bin),
            "-u",
            "-m",
            RUNNER_MODULE,
            "merge-run",
            "--cohort",
            str(cohort),
            "--cohort-sha256",
            cohort_sha256,
            "--output-dir",
            str(model_root),
            "--model-id",
            model_id,
            "--scope",
            scope,
            "--max-num-seqs",
            str(max_num_seqs),
            "--orchestrator-timing",
            str(timing_path),
        ]
    )
    _run_checked(
        _validate_run_command(
            python_bin=python_bin,
            model_root=model_root,
            cohort=cohort,
            cohort_sha256=cohort_sha256,
            model_id=model_id,
            scope=scope,
            max_num_seqs=max_num_seqs,
        )
    )
    return {
        "status": "complete",
        "timing": _validate_timing(
            timing_path, formal_authorization=authorization_evidence
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", required=True, type=Path)
    parser.add_argument("--model-id", required=True, choices=MODEL_ORDER)
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--cohort-sha256", required=True)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--scope", required=True, choices=SCOPES)
    parser.add_argument("--max-num-seqs", required=True, type=int, choices=MAX_NUM_SEQS)
    parser.add_argument("--allow-resume", action="store_true")
    parser.add_argument("--formal-authorization", type=Path)
    parser.add_argument("--gpu-wait-timeout-seconds", type=int, default=172800)
    parser.add_argument("--gpu-poll-seconds", type=int, default=30)
    parser.add_argument(
        "--ready-timeout-seconds", type=int, default=READY_TIMEOUT_SECONDS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_model(
            python_bin=args.python_bin,
            model_id=args.model_id,
            cohort=args.cohort,
            cohort_sha256=args.cohort_sha256,
            model_root=args.model_root,
            scope=args.scope,
            max_num_seqs=args.max_num_seqs,
            allow_resume=args.allow_resume,
            formal_authorization=args.formal_authorization,
            gpu_wait_timeout_seconds=args.gpu_wait_timeout_seconds,
            gpu_poll_seconds=args.gpu_poll_seconds,
            ready_timeout_seconds=args.ready_timeout_seconds,
        )
        print(
            _canonical(
                {
                    "status": result["status"],
                    "model_id": args.model_id,
                    "max_num_seqs": args.max_num_seqs,
                    "aggregate_output_tokens_per_second": result["timing"][
                        "shared_timing"
                    ]["aggregate_output_tokens_per_second"],
                }
            )
        )
        return 0
    except (
        DualDp1OrchestrationError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        print(
            _canonical(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DualDp1OrchestrationError",
    "GO_SCHEMA",
    "TIMING_SCHEMA",
    "_worker_environment",
    "main",
    "run_model",
]
