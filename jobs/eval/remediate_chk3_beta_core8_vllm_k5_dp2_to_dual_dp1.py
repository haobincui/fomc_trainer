"""Seal the zero-row DP2 hang remediation to two independent DP1 engines.

The vLLM 0.8.5 V1 DP=2 benchmark attempt initialized two EngineCore
processes, entered the data-parallel NCCL setup, and then made no observable
progress.  It was interrupted before model load completed and before any
generation row or chunk receipt was committed.  This CPU-only bridge keeps
that attempt immutable and authorizes a fresh v3 topology: two independent
single-GPU vLLM engines, with an explicit and deterministic 20/20 split of
each 40-request meeting chunk.

Creation is deliberately fail closed.  It binds the complete historical
receipt chain, the exact failed launch/state/empty WALs, the relevant pipeline
log, the recorded EngineCore PIDs, the sealed cohort, the old and replacement
implementations, idle physical GPU identities, acquirable GPU locks, and the
lexical absence of every v3 output root.  It never launches GPU work.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    ROOT / "output/evaluation/main/chk3_beta_core8_merged_n2048_vllm_k5_20260816_v1"
)
COHORT = RUN_ROOT / "preparation/cohort_n2048_k5.v1.json"
FAILED_DP2_ROOT = (
    RUN_ROOT / "benchmark_max_num_seqs_chk1_core8_k5_v2/max_num_seqs_8/chk1"
)
DP2_PIPELINE_LOG = RUN_ROOT / "pipeline_vllm_k5_20260816.log"
HISTORICAL_RECEIPTS = {
    "nf4_k10_superseded": RUN_ROOT / "migration/nf4_k10_superseded_receipt.json",
    "vllm_v1_async_output_remediation": (
        RUN_ROOT / "migration/vllm_v1_async_output_remediation_receipt.json"
    ),
    "vllm_v1_fresh_root_addendum": (
        RUN_ROOT / "migration/vllm_v1_async_output_fresh_root_addendum_receipt.json"
    ),
}

OLD_RUNNER = ROOT / "jobs/eval/eval_chk3_beta_core8_merged_vllm_k5.py"
OLD_LAUNCHER = ROOT / "run/eval_chk3_beta_core8_merged_vllm_k5.sh"
OLD_SELECTOR = ROOT / "jobs/eval/select_chk3_beta_core8_vllm_k5_benchmark.py"
NEW_RUNNER = ROOT / "jobs/eval/eval_chk3_beta_core8_merged_vllm_k5_dual_dp1.py"
NEW_LAUNCHER = ROOT / "run/eval_chk3_beta_core8_merged_vllm_k5_dual_dp1.sh"
NEW_SELECTOR = ROOT / "jobs/eval/select_chk3_beta_core8_vllm_k5_dual_dp1_benchmark.py"
NEW_ASSEMBLER = (
    ROOT / "jobs/eval/assemble_chk3_beta_core8_meeting_documents_vllm_k5_dual_dp1.py"
)
NEW_SUITE_SEALER = ROOT / "jobs/eval/seal_chk3_beta_core8_vllm_k5_dual_dp1_suite.py"
NEW_ORCHESTRATOR = (
    ROOT / "jobs/eval/orchestrate_chk3_beta_core8_merged_vllm_k5_dual_dp1.py"
)
VLLM_PYTHON = Path("/home/haobin_cui/.conda/envs/vllm_env/bin/python3.10")
VLLM_SITE_PACKAGES = Path(
    "/home/haobin_cui/.conda/envs/vllm_env/lib/python3.10/site-packages"
)
VLLM_RUNTIME_SOURCES = {
    "vllm_v1_async_llm": VLLM_SITE_PACKAGES / "vllm/v1/engine/async_llm.py",
    "vllm_v1_core_client": VLLM_SITE_PACKAGES / "vllm/v1/engine/core_client.py",
    "vllm_engine_arg_utils": VLLM_SITE_PACKAGES / "vllm/engine/arg_utils.py",
    "vllm_config": VLLM_SITE_PACKAGES / "vllm/config.py",
    "vllm_cuda_platform": VLLM_SITE_PACKAGES / "vllm/platforms/cuda.py",
    "vllm_v1_sampler": VLLM_SITE_PACKAGES / "vllm/v1/sample/sampler.py",
    "vllm_v1_gpu_model_runner": (
        VLLM_SITE_PACKAGES / "vllm/v1/worker/gpu_model_runner.py"
    ),
    "vllm_v1_topk_topp_sampler": (
        VLLM_SITE_PACKAGES / "vllm/v1/sample/ops/topk_topp_sampler.py"
    ),
    "torch_version": VLLM_SITE_PACKAGES / "torch/version.py",
}
VLLM_RUNTIME_SOURCE_HASHES = {
    "vllm_v1_async_llm": "e83f905bd879e4e9c87fb4ce458aa453f31d06bdde945822282ca251d7c54a74",
    "vllm_v1_core_client": "5e6d623e14fc4571e3ddb38ef44d4b88c0521874086abb925743efc129802805",
    "vllm_engine_arg_utils": "65dfb922f57b5b242ac226024ae6a8041cc141b5b003831ea0d372d37d9e1caa",
    "vllm_config": "016f1281a0519c636bace104012ff8d1c199db89aba190d6d9cd810d67c91a48",
    "vllm_cuda_platform": "f2a80d4df5733f5d03a5ca2861545eb8ada75df75260d9b6a665b699e2a91681",
    "vllm_v1_sampler": "9d1c04c6f44adbaabb96388ca0766791351a6c84fa95672be70a1116498972b1",
    "vllm_v1_gpu_model_runner": "cca495d4d86bcc389bd2167404d076d95c4d4b6ec473eb45012e43ebfcc9a41a",
    "vllm_v1_topk_topp_sampler": "eeddf6500a588e73dc726b57a4646b6f4b82763f0311a017af45979ecf91c56e",
    "torch_version": "fb37823f3a4c3c29cd008fac7d3ce5aeb8e7da6f883b17aa9ea6d73ce6eecbfd",
}

V3_ROOTS = (
    RUN_ROOT / "benchmark_max_num_seqs_chk1_core8_k5_v3_dual_dp1",
    RUN_ROOT / "generation_smoke_core8_k5_three_models_v3_dual_dp1",
    RUN_ROOT / "generation_formal_n2048_k5_three_models_v3_dual_dp1",
    RUN_ROOT / "meeting_documents_n3840_v3_dual_dp1",
)
GPU_LOCKS = (
    Path("/tmp/fomc_trainer_chk3_beta_core8_vllm_k5_gpu0.lock"),
    Path("/tmp/fomc_trainer_chk3_beta_core8_vllm_k5_gpu1.lock"),
)

DEFAULT_RECEIPT = (
    RUN_ROOT / "migration/vllm_dp2_hang_to_dual_independent_dp1_receipt.json"
)
SCHEMA_VERSION = "chk3-beta-core8-vllm-k5-dp2-to-dual-dp1-remediation-v1"
EVALUATION_ID = "chk3-beta-core8-merged-1993-2025-n2048-vllm-k5-dual-independent-dp1-v1"
CONFIRMATION = "SEAL-DP2-HANG-TO-DUAL-INDEPENDENT-DP1-V3"

COHORT_SHA256 = "a815b5af8e6b33e3d1a2b211e346393f155d1d45acaafae80b822ee08128ab8e"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
FAILED_LAUNCH_SHA256 = (
    "d0c49a7fa3befebcc5bab3a1d755b089ac23554ba5794b4132585a5a9c3f3260"
)
FAILED_LAUNCH_PAYLOAD_SHA256 = (
    "1c55d114e1102bf1b812a440bb57dc0e0eabb7861c7d964a64f93fc55cc38add"
)
FAILED_STATE_SHA256 = "1a6cf5f73e9a8d4fa2e09dc10c83ff2d95b6e92b051f00b4aa652a3d2d8e5fda"
FAILED_STATE_PAYLOAD_SHA256 = (
    "fd1e490d055249d2ea8465d8271cfe2919f7b80620a4128f878066348bcdb2db"
)
DP2_PIPELINE_LOG_SHA256 = (
    "46c27e1d2542e21ee69f87a3996634e426b3f1a0430279b5d4be34793dad4a52"
)
DP2_ENGINE_CORE_PIDS = (872830, 872831)
EXPECTED_GPU_IDENTITIES = (
    {
        "physical_gpu_index": 0,
        "uuid": "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
        "pci_bus_id": "00000000:21:00.0",
    },
    {
        "physical_gpu_index": 1,
        "uuid": "GPU-0d2b981c-7a91-ad4c-7015-89445a0283a1",
        "pci_bus_id": "00000000:e1:00.0",
    },
)
HISTORICAL_RECEIPT_HASHES = {
    "nf4_k10_superseded": (
        "81b41fbcb9456a25e886377b23f22783c873235ccae4b794c3f954949247e07f",
        "7c75660f0d0d9d39b618486c806346f597edcfeda2843fba4216b5b3ca494947",
    ),
    "vllm_v1_async_output_remediation": (
        "c271e8c396d6526cd8f0b06147638bbf69ab3497ed03367359695ae53daa9a5c",
        "35e4c3a8398e583f8d708c66ae01ce5e230708e755d0fbad95c78e8a46c4e04f",
    ),
    "vllm_v1_fresh_root_addendum": (
        "81fd4376cbacdcc1b119a00a27d47609b86de3a7f21e4000e176f993d6c4cfa6",
        "0c0cad14d8cf477f84262fb411ba650bf9a95dc56859399a0c161ddd8b9f1a49",
    ),
}


class DualDp1RemediationError(RuntimeError):
    """The topology remediation could not be proven safely."""


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
        raise DualDp1RemediationError(f"bound artifact is a symlink: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise DualDp1RemediationError(f"bound artifact is missing: {resolved}")
    stat_result = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": stat_result.st_size,
        "sha256": _sha256_file(resolved),
        "mtime_ns": stat_result.st_mtime_ns,
    }


def _read_sealed_json(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    binding = _binding(path)
    try:
        value = json.loads(Path(binding["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DualDp1RemediationError(f"cannot read sealed JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DualDp1RemediationError(f"sealed JSON is not an object: {path}")
    try:
        payload_sha256 = validate_manifest_integrity(value)
    except Exception as exc:
        raise DualDp1RemediationError(f"sealed JSON integrity failed: {path}") from exc
    return value, binding, payload_sha256


def audit_historical_receipts() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path in HISTORICAL_RECEIPTS.items():
        value, binding, payload_sha = _read_sealed_json(path)
        expected_file_sha, expected_payload_sha = HISTORICAL_RECEIPT_HASHES[name]
        mode = stat.S_IMODE(path.stat().st_mode)
        if (
            binding["sha256"] != expected_file_sha
            or payload_sha != expected_payload_sha
            or mode != 0o444
            or value.get("status") != "passed"
        ):
            raise DualDp1RemediationError(f"historical receipt drift: {name}")
        result[name] = {
            **binding,
            "payload_sha256": payload_sha,
            "mode": oct(mode),
        }
    return result


def _empty_jsonl_binding(path: Path) -> dict[str, Any]:
    binding = _binding(path)
    if binding["bytes"] != 0 or binding["sha256"] != EMPTY_SHA256:
        raise DualDp1RemediationError(f"failed DP2 WAL is not empty: {path}")
    return {**binding, "rows": 0}


def _audit_log() -> dict[str, Any]:
    binding = _binding(DP2_PIPELINE_LOG)
    if binding["sha256"] != DP2_PIPELINE_LOG_SHA256:
        raise DualDp1RemediationError("historical DP2 pipeline log drift")
    text = DP2_PIPELINE_LOG.read_text(encoding="utf-8")
    required_fragments = (
        "[2026-08-16T21:55:51Z] pipeline_start mode=benchmark",
        "[2026-08-16T21:55:57Z] running infrastructure_smoke chk1",
        "(EngineCore_0 pid=872830)",
        "(EngineCore_1 pid=872831)",
        "Adjusting world_size=2 rank=0",
        "Adjusting world_size=2 rank=1",
        "vLLM is using nccl==2.21.5",
        "NCCL version 2.21.5+cuda12.4",
    )
    if any(fragment not in text for fragment in required_fragments):
        raise DualDp1RemediationError("DP2 log no longer proves the recorded hang")
    observed_pids = tuple(
        sorted({int(match) for match in re.findall(r"EngineCore_[01] pid=(\d+)", text)})
    )
    if observed_pids != DP2_ENGINE_CORE_PIDS:
        raise DualDp1RemediationError("DP2 EngineCore PID evidence drift")
    return {
        **binding,
        "attempt_start_at_utc": "2026-08-16T21:55:51Z",
        "runner_start_at_utc": "2026-08-16T21:55:57Z",
        "engine_core_initialization_log_at_utc": "2026-08-16T21:56:55Z",
        "last_collective_progress_log_at_utc": "2026-08-16T21:56:56Z",
        "engine_core_pids": list(observed_pids),
        "last_observed_stage": "dp_world_size_2_nccl_initialization",
    }


def audit_failed_dp2_attempt() -> dict[str, Any]:
    launch_path = FAILED_DP2_ROOT / "launch.json"
    state_path = FAILED_DP2_ROOT / "state.progress.v1.json"
    wal_path = (
        FAILED_DP2_ROOT / ".partial/generations.completion_order.progress.v1.jsonl"
    )
    chunks_path = FAILED_DP2_ROOT / ".partial/absolute_chunks.progress.v1.jsonl"
    expected = {launch_path, state_path, wal_path, chunks_path}
    observed = {path.resolve() for path in FAILED_DP2_ROOT.rglob("*") if path.is_file()}
    if observed != {path.resolve() for path in expected}:
        raise DualDp1RemediationError("failed DP2 directory inventory drift")
    for forbidden in (
        FAILED_DP2_ROOT / "manifest.json",
        FAILED_DP2_ROOT / "generations.canonical.v1.jsonl",
    ):
        if forbidden.exists() or forbidden.is_symlink():
            raise DualDp1RemediationError(
                f"failed DP2 attempt unexpectedly contains {forbidden.name}"
            )

    launch, launch_binding, launch_payload = _read_sealed_json(launch_path)
    state, state_binding, state_payload = _read_sealed_json(state_path)
    wal = _empty_jsonl_binding(wal_path)
    chunks = _empty_jsonl_binding(chunks_path)
    generation = launch.get("generation")
    runtime = launch.get("runtime")
    if (
        launch_binding["sha256"] != FAILED_LAUNCH_SHA256
        or launch_payload != FAILED_LAUNCH_PAYLOAD_SHA256
        or state_binding["sha256"] != FAILED_STATE_SHA256
        or state_payload != FAILED_STATE_PAYLOAD_SHA256
        or launch.get("status") != "initializing"
        or launch.get("model_id") != "chk1"
        or launch.get("evaluation_scope") != "infrastructure_smoke"
        or not isinstance(generation, Mapping)
        or generation.get("data_parallel_size") != 2
        or generation.get("tensor_parallel_size") != 1
        or generation.get("physical_gpu_indexes") != [0, 1]
        or generation.get("max_num_seqs") != 8
        or generation.get("expected_cases") != 40
        or not isinstance(runtime, Mapping)
        or runtime.get("cuda_visible_devices") != "0,1"
        or state.get("status") != "failed"
        or state.get("model_id") != "chk1"
        or state.get("error_type") != "KeyboardInterrupt"
        or state.get("error") != ""
        or state.get("completed_cases") != 0
        or state.get("completed_absolute_chunks") != 0
        or state.get("active_chunk_completed_indexes") != []
        or state.get("resume_count") != 0
        or state.get("engine_runtime") is not None
        or state.get("completion_wal")
        != {key: wal[key] for key in ("path", "bytes", "sha256", "rows")}
        or state.get("chunk_receipts")
        != {key: chunks[key] for key in ("path", "bytes", "sha256", "rows")}
    ):
        raise DualDp1RemediationError("failed DP2 launch/state evidence drift")

    log = _audit_log()
    if any(Path(f"/proc/{pid}").exists() for pid in DP2_ENGINE_CORE_PIDS):
        raise DualDp1RemediationError("a historical DP2 EngineCore PID is still live")
    return {
        "root": str(FAILED_DP2_ROOT.resolve()),
        "launch": {**launch_binding, "payload_sha256": launch_payload},
        "state": {**state_binding, "payload_sha256": state_payload},
        "completion_wal": wal,
        "absolute_chunk_receipts": chunks,
        "pipeline_log": log,
        "timeline": {
            "launch_created_at_utc": launch.get("created_at_utc"),
            "last_collective_progress_log_at_utc": (
                log["last_collective_progress_log_at_utc"]
            ),
            "operator_interrupt_recorded_at_utc": state.get("failed_at_utc"),
            "no_progress_interval_seconds_at_least": 244,
        },
        "historical_engine_core_pids": list(DP2_ENGINE_CORE_PIDS),
        "historical_engine_core_pids_absent_at_seal": True,
        "model_load_completed": False,
        "generated_rows": 0,
        "completed_chunks": 0,
        "canonical_generations_absent": True,
        "sealed_run_manifest_absent": True,
        "preserved_unchanged": True,
    }


def _run_command(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise DualDp1RemediationError(
            f"command failed ({completed.returncode}): {' '.join(command)}: {message}"
        )
    return completed.stdout


def _gpu_rows() -> list[dict[str, Any]]:
    output = _run_command(
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        )
    )
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise DualDp1RemediationError("unexpected nvidia-smi GPU row")
        rows.append(
            {
                "physical_gpu_index": int(fields[0]),
                "uuid": fields[1],
                "pci_bus_id": fields[2].lower(),
                "memory_used_mib": int(fields[3]),
                "memory_total_mib": int(fields[4]),
                "utilization_gpu_percent": int(fields[5]),
            }
        )
    return rows


def _compute_process_rows() -> list[dict[str, Any]]:
    output = _run_command(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        )
    )
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", maxsplit=3)]
        if len(fields) != 4:
            raise DualDp1RemediationError("unexpected nvidia-smi process row")
        rows.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "used_memory_mib": int(fields[2]),
                "process_name": fields[3],
            }
        )
    return rows


def _acquire_gpu_locks(stack: ExitStack) -> list[TextIO]:
    handles: list[TextIO] = []
    for lock_path in GPU_LOCKS:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if lock_path.is_symlink():
            raise DualDp1RemediationError(f"GPU lock is a symlink: {lock_path}")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags, 0o600)
        handle = stack.enter_context(os.fdopen(descriptor, "a+", encoding="utf-8"))
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise DualDp1RemediationError(f"GPU lock is not regular: {lock_path}")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DualDp1RemediationError(f"GPU lock is held: {lock_path}") from exc
        handles.append(handle)
    return handles


def audit_idle_gpus_and_locks() -> dict[str, Any]:
    with ExitStack() as stack:
        handles = _acquire_gpu_locks(stack)
        gpu_rows = _gpu_rows()
        compute_rows = _compute_process_rows()
        identities = [
            {key: row[key] for key in ("physical_gpu_index", "uuid", "pci_bus_id")}
            for row in gpu_rows
        ]
        if identities != [dict(row) for row in EXPECTED_GPU_IDENTITIES]:
            raise DualDp1RemediationError("physical GPU identities drifted")
        if compute_rows:
            raise DualDp1RemediationError("both GPUs must have no compute processes")
        if any(
            row["memory_used_mib"] > 64 or row["utilization_gpu_percent"] != 0
            for row in gpu_rows
        ):
            raise DualDp1RemediationError("both GPUs must be idle at receipt seal")
        return {
            "observed_at_utc": _utc_now(),
            "physical_gpus": gpu_rows,
            "compute_processes": compute_rows,
            "both_gpus_idle": True,
            "ordered_lock_paths": [str(path.resolve()) for path in GPU_LOCKS],
            "ordered_locks_acquired": len(handles) == 2,
        }


def audit_v3_roots_absent() -> dict[str, Any]:
    rows = []
    for path in V3_ROOTS:
        lexists = os.path.lexists(path)
        if lexists:
            raise DualDp1RemediationError(f"v3 root must be fresh: {path}")
        rows.append({"path": str(path.resolve()), "lexists_at_seal": False})
    return {"all_absent_at_seal": True, "roots": rows}


def audit_implementations() -> dict[str, Any]:
    paths = {
        "historical_dp2_runner": OLD_RUNNER,
        "historical_dp2_launcher": OLD_LAUNCHER,
        "historical_dp2_benchmark_selector": OLD_SELECTOR,
        "dual_independent_dp1_runner": NEW_RUNNER,
        "dual_independent_dp1_launcher": NEW_LAUNCHER,
        "dual_independent_dp1_benchmark_selector": NEW_SELECTOR,
        "dual_independent_dp1_meeting_document_assembler": NEW_ASSEMBLER,
        "dual_independent_dp1_suite_sealer": NEW_SUITE_SEALER,
        "dual_independent_dp1_orchestrator": NEW_ORCHESTRATOR,
        "topology_remediation_tool": Path(__file__).resolve(),
    }
    result = {name: _binding(path) for name, path in paths.items()}
    if (
        result["historical_dp2_runner"]["sha256"]
        != "1bc77b941c278a53c88189f32f1a158c4c607600bd795b5f86af9474fffd2632"
        or result["historical_dp2_launcher"]["sha256"]
        != "9a44dc8cc48e956721f73d570ebe82c19111d20fdf49a41b7d4812ceaee7c421"
        or result["historical_dp2_benchmark_selector"]["sha256"]
        != "2298b8d6d19051897af338a883a8dba2568af288ebbb97f99f3871a6e932b8cf"
    ):
        raise DualDp1RemediationError("historical DP2 implementation drift")
    return result


def audit_vllm_runtime() -> dict[str, Any]:
    sources = {name: _binding(path) for name, path in VLLM_RUNTIME_SOURCES.items()}
    if any(
        sources[name]["sha256"] != expected
        for name, expected in VLLM_RUNTIME_SOURCE_HASHES.items()
    ):
        raise DualDp1RemediationError("installed vLLM/torch runtime source drift")
    executable = _binding(VLLM_PYTHON)
    if executable["sha256"] != (
        "a4d0418fa5e01928fca833b6ba7ba84a60b36b9334281ef83fbb9de790561929"
    ):
        raise DualDp1RemediationError("vLLM Python executable drift")
    metadata_script = (
        "import importlib.metadata as m, json, sys;"
        "print(json.dumps({'python':sys.version.split()[0],"
        "'vllm':m.version('vllm'),'torch_distribution':m.version('torch'),"
        "'transformers':m.version('transformers'),'tokenizers':m.version('tokenizers')}))"
    )
    completed = subprocess.run(
        [str(VLLM_PYTHON), "-c", metadata_script],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise DualDp1RemediationError("cannot inspect vLLM environment metadata")
    metadata = json.loads(completed.stdout)
    expected_metadata = {
        "python": "3.10.16",
        "vllm": "0.8.5.post1",
        "torch_distribution": "2.6.0",
        "transformers": "4.51.3",
        "tokenizers": "0.21.1",
    }
    if metadata != expected_metadata:
        raise DualDp1RemediationError("vLLM environment version metadata drift")
    return {
        "python_executable": executable,
        "package_versions": metadata,
        "torch_runtime_version": "2.6.0+cu124",
        "cuda_build_version": "12.4",
        "source_files": sources,
    }


def _cohort_binding() -> dict[str, Any]:
    binding = _binding(COHORT)
    if binding["sha256"] != COHORT_SHA256:
        raise DualDp1RemediationError("sealed K5 cohort drift")
    value, _, payload_sha = _read_sealed_json(COHORT)
    if (
        value.get("status") != "complete"
        or value.get("generation_design", {}).get("meetings") != 256
        or value.get("generation_design", {}).get("prompts") != 2048
        or value.get("generation_design", {}).get("replicates") != 5
        or value.get("generation_design", {}).get("total_rows") != 30720
    ):
        raise DualDp1RemediationError("sealed K5 cohort coverage drift")
    return {**binding, "payload_sha256": payload_sha}


def _fixed_shard_assignment() -> dict[str, Any]:
    assignments = [
        {
            "local_meeting_case_offset": offset,
            "shard_id": offset % 2,
            "physical_gpu_index": offset % 2,
        }
        for offset in range(40)
    ]
    return {
        "formula": "shard_id=absolute_case_index_mod_2",
        "equivalent_local_formula": "shard_id=local_meeting_case_offset_mod_2",
        "rows_per_shard_per_chunk": 20,
        "shard0_local_offsets": list(range(0, 40, 2)),
        "shard1_local_offsets": list(range(1, 40, 2)),
        "assignments": assignments,
        "assignment_sha256": hashlib.sha256(
            _canonical(assignments).encode("utf-8")
        ).hexdigest(),
        "stable_across_models_candidates_and_resume": True,
    }


def build_receipt() -> dict[str, Any]:
    return seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "created_at_utc": _utc_now(),
            "evaluation_id": EVALUATION_ID,
            "cohort_manifest": _cohort_binding(),
            "historical_receipts": audit_historical_receipts(),
            "failed_dp2_attempt": audit_failed_dp2_attempt(),
            "gpu_idle_evidence": audit_idle_gpus_and_locks(),
            "fresh_v3_roots": audit_v3_roots_absent(),
            "implementation_sources": audit_implementations(),
            "vllm_runtime": audit_vllm_runtime(),
            "topology_supersession": {
                "superseded_backend": "vllm-v1-data-parallel-async-engine",
                "superseded_data_parallel_size": 2,
                "superseded_failure_stage": "dp_world_size_2_nccl_initialization",
                "replacement_backend": (
                    "vllm-v1-dual-independent-dp1-continuous-batching"
                ),
                "independent_engine_count": 2,
                "data_parallel_size_per_engine": 1,
                "tensor_parallel_size_per_engine": 1,
                "pipeline_parallel_size_per_engine": 1,
                "physical_gpu_indices": [0, 1],
                "cuda_visible_devices_per_worker": {"shard0": "0", "shard1": "1"},
                "data_parallel_master_environment": "unset",
                "cross_gpu_collectives": False,
                "nccl_process_group_required": False,
                "weight_precision": "bfloat16",
                "quantization": None,
                "gpu_memory_utilization_per_engine": 0.95,
                "enforce_eager": False,
                "cuda_graphs": True,
                "async_output_processing": True,
                "success_path_gpu_release_postcondition": {
                    "worker_leaders_exited_before_audit": True,
                    "shards_deep_validated_before_audit": True,
                    "canonical_gpu_locks_held_during_audit": True,
                    "exact_gpu_uuid_and_pci_identity_required": True,
                    "no_compute_processes_required": True,
                    "consecutive_idle_samples_required": 2,
                    "teardown_timeout_seconds": 60,
                    "teardown_poll_seconds": 5,
                    "reconstructed_timing_must_record_null_evidence": True,
                },
                "max_num_seqs_candidates_per_engine": [8, 12, 16],
                "absolute_meeting_chunk_size": 40,
                "fixed_shard_split": _fixed_shard_assignment(),
                "replicates": 5,
                "rows_per_model": 10240,
                "expected_total_rows": 30720,
                "model_order": ["chk1", "chk3", "chk0"],
                "formal_generation_gates": {
                    "benchmark_candidate_exact_token_parity_8_12_16": True,
                    "selected_candidate_exact_chk1_smoke_replay": True,
                    "formal_forbidden_until_both_gates_sealed": True,
                    "official_smoke_suite_authorization_required_pre_gpu": True,
                    "authorization_bound_in_worker_controls_rows_and_manifests": True,
                },
            },
            "safety": {
                "historical_dp2_directories_deleted_or_modified": False,
                "historical_rows_reused": 0,
                "gpu_work_launched_by_remediation": False,
                "receipt_create_only": True,
                "receipt_mode": "0o444",
            },
        }
    )


def write_readonly_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    unresolved = path.expanduser()
    if os.path.lexists(unresolved):
        raise DualDp1RemediationError(f"receipt is create-only: {unresolved}")
    unresolved.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(unresolved, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_canonical(dict(receipt)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        unresolved.chmod(0o444)
        directory_fd = os.open(unresolved.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if unresolved.exists():
            unresolved.chmod(0o600)
            unresolved.unlink()
        raise


def _validate_binding(binding: Any) -> None:
    if not isinstance(binding, Mapping):
        raise DualDp1RemediationError("receipt artifact binding is missing")
    path = Path(str(binding.get("path")))
    observed = _binding(path)
    for field in ("path", "bytes", "sha256", "mtime_ns"):
        if binding.get(field) != observed.get(field):
            raise DualDp1RemediationError(f"bound artifact drift: {path}")


def load_and_validate_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    value, binding, payload_sha = _read_sealed_json(path)
    mode = stat.S_IMODE(path.stat().st_mode)
    history = value.get("historical_receipts")
    sources = value.get("implementation_sources")
    topology = value.get("topology_supersession")
    split = topology.get("fixed_shard_split") if isinstance(topology, Mapping) else None
    expected_source_keys = {
        "historical_dp2_runner",
        "historical_dp2_launcher",
        "historical_dp2_benchmark_selector",
        "dual_independent_dp1_runner",
        "dual_independent_dp1_launcher",
        "dual_independent_dp1_benchmark_selector",
        "dual_independent_dp1_meeting_document_assembler",
        "dual_independent_dp1_suite_sealer",
        "dual_independent_dp1_orchestrator",
        "topology_remediation_tool",
    }
    if (
        mode != 0o444
        or value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "passed"
        or value.get("evaluation_id") != EVALUATION_ID
        or value.get("safety", {}).get("receipt_create_only") is not True
        or value.get("safety", {}).get("receipt_mode") != "0o444"
        or value.get("safety", {}).get("gpu_work_launched_by_remediation") is not False
        or value.get("failed_dp2_attempt", {}).get("generated_rows") != 0
        or value.get("failed_dp2_attempt", {}).get("completed_chunks") != 0
        or value.get("failed_dp2_attempt", {}).get("preserved_unchanged") is not True
        or value.get("gpu_idle_evidence", {}).get("both_gpus_idle") is not True
        or value.get("gpu_idle_evidence", {}).get("compute_processes") != []
        or value.get("fresh_v3_roots", {}).get("all_absent_at_seal") is not True
        or not isinstance(history, Mapping)
        or set(history) != set(HISTORICAL_RECEIPTS)
        or not isinstance(sources, Mapping)
        or set(sources) != expected_source_keys
        or not isinstance(topology, Mapping)
        or topology.get("independent_engine_count") != 2
        or topology.get("data_parallel_size_per_engine") != 1
        or topology.get("tensor_parallel_size_per_engine") != 1
        or topology.get("pipeline_parallel_size_per_engine") != 1
        or topology.get("physical_gpu_indices") != [0, 1]
        or topology.get("cuda_visible_devices_per_worker")
        != {"shard0": "0", "shard1": "1"}
        or topology.get("data_parallel_master_environment") != "unset"
        or topology.get("cross_gpu_collectives") is not False
        or topology.get("nccl_process_group_required") is not False
        or topology.get("gpu_memory_utilization_per_engine") != 0.95
        or topology.get("enforce_eager") is not False
        or topology.get("cuda_graphs") is not True
        or topology.get("async_output_processing") is not True
        or topology.get("success_path_gpu_release_postcondition")
        != {
            "worker_leaders_exited_before_audit": True,
            "shards_deep_validated_before_audit": True,
            "canonical_gpu_locks_held_during_audit": True,
            "exact_gpu_uuid_and_pci_identity_required": True,
            "no_compute_processes_required": True,
            "consecutive_idle_samples_required": 2,
            "teardown_timeout_seconds": 60,
            "teardown_poll_seconds": 5,
            "reconstructed_timing_must_record_null_evidence": True,
        }
        or topology.get("max_num_seqs_candidates_per_engine") != [8, 12, 16]
        or topology.get("absolute_meeting_chunk_size") != 40
        or topology.get("replicates") != 5
        or topology.get("rows_per_model") != 10240
        or topology.get("expected_total_rows") != 30720
        or topology.get("model_order") != ["chk1", "chk3", "chk0"]
        or not isinstance(split, Mapping)
        or split.get("formula") != "shard_id=absolute_case_index_mod_2"
        or split.get("rows_per_shard_per_chunk") != 20
        or split.get("shard0_local_offsets") != list(range(0, 40, 2))
        or split.get("shard1_local_offsets") != list(range(1, 40, 2))
        or split.get("assignment_sha256")
        != _fixed_shard_assignment()["assignment_sha256"]
        or topology.get("formal_generation_gates")
        != {
            "benchmark_candidate_exact_token_parity_8_12_16": True,
            "selected_candidate_exact_chk1_smoke_replay": True,
            "formal_forbidden_until_both_gates_sealed": True,
            "official_smoke_suite_authorization_required_pre_gpu": True,
            "authorization_bound_in_worker_controls_rows_and_manifests": True,
        }
    ):
        raise DualDp1RemediationError("sealed topology remediation contract drift")
    _validate_binding(value.get("cohort_manifest"))
    for historical in value.get("historical_receipts", {}).values():
        _validate_binding(historical)
    failed = value.get("failed_dp2_attempt", {})
    for key in (
        "launch",
        "state",
        "completion_wal",
        "absolute_chunk_receipts",
        "pipeline_log",
    ):
        _validate_binding(failed.get(key))
    for source in value.get("implementation_sources", {}).values():
        _validate_binding(source)
    runtime = value.get("vllm_runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("package_versions")
        != {
            "python": "3.10.16",
            "vllm": "0.8.5.post1",
            "torch_distribution": "2.6.0",
            "transformers": "4.51.3",
            "tokenizers": "0.21.1",
        }
        or runtime.get("torch_runtime_version") != "2.6.0+cu124"
        or runtime.get("cuda_build_version") != "12.4"
        or set(runtime.get("source_files", {})) != set(VLLM_RUNTIME_SOURCES)
    ):
        raise DualDp1RemediationError("sealed vLLM runtime contract drift")
    _validate_binding(runtime.get("python_executable"))
    for name, source in runtime["source_files"].items():
        if source.get("sha256") != VLLM_RUNTIME_SOURCE_HASHES[name]:
            raise DualDp1RemediationError(
                f"sealed vLLM runtime source hash drift: {name}"
            )
        _validate_binding(source)
    python_binding = runtime["python_executable"]
    if (
        python_binding.get("path") != str(VLLM_PYTHON.resolve())
        or python_binding.get("sha256")
        != "a4d0418fa5e01928fca833b6ba7ba84a60b36b9334281ef83fbb9de790561929"
    ):
        raise DualDp1RemediationError("sealed vLLM Python executable drift")
    return {
        **value,
        "receipt_binding": {
            **binding,
            "payload_sha256": payload_sha,
            "mode": oct(mode),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--confirm", required=True)
    create.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            if args.confirm != CONFIRMATION:
                raise DualDp1RemediationError("explicit confirmation string mismatch")
            receipt = build_receipt()
            write_readonly_receipt(args.receipt, receipt)
            loaded = load_and_validate_receipt(args.receipt)
        else:
            loaded = load_and_validate_receipt(args.receipt)
        print(
            _canonical(
                {
                    "status": "passed",
                    "path": loaded["receipt_binding"]["path"],
                    "sha256": loaded["receipt_binding"]["sha256"],
                    "payload_sha256": loaded["receipt_binding"]["payload_sha256"],
                    "mode": loaded["receipt_binding"]["mode"],
                    "failed_dp2_rows": loaded["failed_dp2_attempt"]["generated_rows"],
                    "replacement_backend": loaded["topology_supersession"][
                        "replacement_backend"
                    ],
                }
            )
        )
        return 0
    except (
        DualDp1RemediationError,
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
    "CONFIRMATION",
    "DEFAULT_RECEIPT",
    "DualDp1RemediationError",
    "audit_failed_dp2_attempt",
    "audit_historical_receipts",
    "audit_idle_gpus_and_locks",
    "audit_implementations",
    "audit_v3_roots_absent",
    "build_receipt",
    "load_and_validate_receipt",
    "main",
    "write_readonly_receipt",
]
