"""Generate Core8 K=5 with two independently scheduled vLLM DP=1 workers.

This is a versioned alternative to ``eval_chk3_beta_core8_merged_vllm_k5``.
It never opens or mutates an output produced by that DP=2 runner.  A worker
sees exactly one physical GPU and owns one deterministic parity shard:
``absolute_case_index % 2 == shard_id``.  Consequently every immutable
40-request meeting window contributes exactly 20 requests to each worker.

Each worker has its own completion-order WAL, state, chunk receipts, and
sealed manifest.  ``merge-run`` deep-validates both shards, proves an exact
disjoint union, and creates the canonical absolute-index ordered model run.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import fcntl
import gc
import math
import os
import re
import stat
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.eval import eval_chk3_beta_core8_merged_vllm_k5 as dp2
from jobs.eval import remediate_chk3_beta_core8_vllm_k5_dp2_to_dual_dp1 as remediation
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ID = "chk3-beta-core8-merged-1993-2025-n2048-vllm-k5-dual-independent-dp1-v1"
MODEL_ORDER = dp2.MODEL_ORDER
MODEL_LABELS = dp2.MODEL_LABELS
REPLICATE_SEEDS = dp2.REPLICATE_SEEDS
CASES_PER_MODEL = dp2.CASES_PER_MODEL
ABSOLUTE_CHUNK_SIZE = dp2.ABSOLUTE_CHUNK_SIZE
CHUNKS_PER_MODEL = dp2.CHUNKS_PER_MODEL
SMOKE_CASES_PER_MODEL = dp2.SMOKE_CASES_PER_MODEL
SHARD_COUNT = 2
CASES_PER_SHARD_CHUNK = ABSOLUTE_CHUNK_SIZE // SHARD_COUNT
FORMAL_CASES_PER_SHARD = CASES_PER_MODEL // SHARD_COUNT
SMOKE_CASES_PER_SHARD = SMOKE_CASES_PER_MODEL // SHARD_COUNT

ROW_SCHEMA = "chk3-beta-core8-merged-vllm-k5-dual-dp1-generation-row-v1"
SHARD_MANIFEST_SCHEMA = "chk3-beta-core8-merged-vllm-k5-dual-dp1-shard-manifest-v1"
MERGED_MANIFEST_SCHEMA = "chk3-beta-core8-merged-vllm-k5-dual-dp1-merged-manifest-v1"
STATE_SCHEMA = "chk3-beta-core8-merged-vllm-k5-dual-dp1-state-v1"
CHUNK_RECEIPT_SCHEMA = "chk3-beta-core8-merged-vllm-k5-dual-dp1-chunk-receipt-v1"

TEMPERATURE = dp2.TEMPERATURE
TOP_P = dp2.TOP_P
TOP_K = dp2.TOP_K
REPETITION_PENALTY = dp2.REPETITION_PENALTY
MAX_NEW_TOKENS = dp2.MAX_NEW_TOKENS
TAIL_TOKENS = dp2.TAIL_TOKENS
MAX_MODEL_LEN = dp2.MAX_MODEL_LEN
ALLOWED_MAX_NUM_SEQS = dp2.ALLOWED_MAX_NUM_SEQS
MAX_NUM_BATCHED_TOKENS = dp2.MAX_NUM_BATCHED_TOKENS
GPU_MEMORY_UTILIZATION = dp2.GPU_MEMORY_UTILIZATION
ALLOWED_SHARD_IDS = (0, 1)
ALLOWED_PHYSICAL_GPU_INDEXES = (0, 1)
EXPECTED_GPU_IDENTITIES = {
    0: {
        "uuid": "GPU-ba3a7f4c-74c8-4922-364d-031c5187624c",
        "pci_bus_id": "00000000:21:00.0",
    },
    1: {
        "uuid": "GPU-0d2b981c-7a91-ad4c-7015-89445a0283a1",
        "pci_bus_id": "00000000:e1:00.0",
    },
}
DATA_PARALLEL_SIZE = 1
TENSOR_PARALLEL_SIZE = 1
PIPELINE_PARALLEL_SIZE = 1
EXPECTED_VLLM_VERSION = dp2.EXPECTED_VLLM_VERSION
# Reuse the canonical lock namespace so the historical DP=2 runner and this
# runner can never allocate the same card concurrently.
GPU_LOCK_PATH_TEMPLATE = dp2.GPU_LOCK_PATH_TEMPLATE
REQUIRED_VLLM_ENV = {
    "VLLM_USE_V1": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
}
FORBIDDEN_DP_ENV = (
    "VLLM_DP_MASTER_IP",
    "VLLM_DP_MASTER_PORT",
    "VLLM_DP_RANK",
    "VLLM_DP_RANK_LOCAL",
    "VLLM_DP_SIZE",
)
RUN_TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")
READY_SCHEMA = "chk3-beta-core8-vllm-k5-dual-dp1-ready-v1"
GO_SCHEMA = "chk3-beta-core8-vllm-k5-dual-dp1-go-v1"
DONE_SCHEMA = "chk3-beta-core8-vllm-k5-dual-dp1-done-v1"
ORCHESTRATOR_TIMING_SCHEMA = "chk3-beta-core8-vllm-k5-dual-dp1-orchestrator-timing-v1"


class DualDp1GenerationError(RuntimeError):
    """A frozen independent-worker generation invariant was violated."""


WalTracker = dp2.WalTracker
_canonical = dp2._canonical
_encoded_row = dp2._encoded_row
_sha256_text = dp2._sha256_text
_read_canonical_jsonl = dp2._read_canonical_jsonl
_file_binding = dp2._file_binding
_verify_file_binding = dp2._verify_file_binding
load_cohort = dp2.load_cohort
canonical_cases = dp2.canonical_cases
core = dp2.core


def _validate_max_num_seqs(value: int) -> int:
    try:
        return dp2._validate_max_num_seqs(value)
    except dp2.VllmK5GenerationError as exc:
        raise DualDp1GenerationError(str(exc)) from exc


def _validate_shard_id(value: int) -> int:
    if isinstance(value, bool) or value not in ALLOWED_SHARD_IDS:
        raise DualDp1GenerationError("shard_id must be exactly 0 or 1")
    return value


def _validate_gpu_index(value: int) -> int:
    if isinstance(value, bool) or value not in ALLOWED_PHYSICAL_GPU_INDEXES:
        raise DualDp1GenerationError("physical_gpu_index must be exactly 0 or 1")
    return value


def _validate_worker_mapping(shard_id: int, physical_gpu_index: int) -> tuple[int, int]:
    shard_id = _validate_shard_id(shard_id)
    physical_gpu_index = _validate_gpu_index(physical_gpu_index)
    if shard_id != physical_gpu_index:
        raise DualDp1GenerationError(
            "the frozen worker mapping requires shard_id == physical_gpu_index"
        )
    return shard_id, physical_gpu_index


def assigned_shard(absolute_case_index: int) -> int:
    if (
        not isinstance(absolute_case_index, int)
        or isinstance(absolute_case_index, bool)
        or absolute_case_index < 0
    ):
        raise DualDp1GenerationError("absolute_case_index must be non-negative")
    return absolute_case_index % SHARD_COUNT


def shard_cases(
    cases: Sequence[Mapping[str, Any]], shard_id: int
) -> list[Mapping[str, Any]]:
    shard_id = _validate_shard_id(shard_id)
    selected = [
        case
        for case in cases
        if assigned_shard(int(case["absolute_case_index"])) == shard_id
    ]
    for chunk_id in range(math.ceil(len(cases) / ABSOLUTE_CHUNK_SIZE)):
        chunk = [
            case for case in selected if int(case["absolute_chunk_id"]) == chunk_id
        ]
        if len(chunk) != CASES_PER_SHARD_CHUNK:
            raise DualDp1GenerationError(
                f"absolute chunk {chunk_id} does not contain exactly 20 shard cases"
            )
    return selected


def sampling_contract(*, max_num_seqs: int) -> dict[str, Any]:
    max_num_seqs = _validate_max_num_seqs(max_num_seqs)
    return {
        "backend": "vllm-async-engine-v1-two-independent-dp1-workers",
        "vllm_version": EXPECTED_VLLM_VERSION,
        "generation_mode": "stochastic_sampling",
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "max_new_tokens": MAX_NEW_TOKENS,
        "tail_tokens": TAIL_TOKENS,
        "dtype": "bfloat16",
        "quantization": None,
        "load_format": "safetensors",
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs_per_worker": max_num_seqs,
        "max_num_seqs_selection": "benchmark_candidates_8_12_16_then_freeze_one_formal_value",
        "max_num_batched_tokens_per_worker": MAX_NUM_BATCHED_TOKENS,
        "gpu_memory_utilization_per_worker": GPU_MEMORY_UTILIZATION,
        "swap_space_gib": 0,
        "cpu_offload_gib": 0,
        "workers": SHARD_COUNT,
        "data_parallel_size_per_worker": DATA_PARALLEL_SIZE,
        "tensor_parallel_size_per_worker": TENSOR_PARALLEL_SIZE,
        "pipeline_parallel_size_per_worker": PIPELINE_PARALLEL_SIZE,
        "parallel_topology": "two_external_independent_full_bf16_dp1_replicas",
        "physical_gpu_indexes": list(ALLOWED_PHYSICAL_GPU_INDEXES),
        "worker_mapping": "shard_id_equals_physical_gpu_index",
        "shard_function": "absolute_case_index_mod_2",
        "cases_per_absolute_chunk_per_worker": CASES_PER_SHARD_CHUNK,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "enforce_eager": False,
        "cuda_graphs": True,
        "async_output_processing": True,
        "engine_seed": 0,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "row_seed": "derive_row_seed(replicate_seed,sample_id)",
        "input_contract": "consume_exact_fomc_prompt_token_ledger_no_runtime_chat_template",
        "absolute_chunk_size_across_workers": ABSOLUTE_CHUNK_SIZE,
        "worker_wal_order": "request_completion_order",
        "canonical_output_order": "absolute_case_index",
        "required_environment": dict(REQUIRED_VLLM_ENV),
        "forbidden_data_parallel_environment": list(FORBIDDEN_DP_ENV),
        "recommended_model_order": list(MODEL_ORDER),
    }


def _sampling_contract_sha256(*, max_num_seqs: int) -> str:
    return _sha256_text(_canonical(sampling_contract(max_num_seqs=max_num_seqs)))


def _assert_required_environment(
    *, physical_gpu_index: int, require_cuda: bool
) -> None:
    physical_gpu_index = _validate_gpu_index(physical_gpu_index)
    for key, expected in REQUIRED_VLLM_ENV.items():
        if os.environ.get(key) != expected:
            raise DualDp1GenerationError(
                f"{key} must be exactly {expected!r}; observed={os.environ.get(key)!r}"
            )
    present = {
        key: value for key, value in os.environ.items() if key.startswith("VLLM_DP_")
    }
    if present:
        raise DualDp1GenerationError(
            f"data-parallel environment must be absent: {present}"
        )
    if require_cuda and (
        os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu_index)
        or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
    ):
        raise DualDp1GenerationError(
            f"worker {physical_gpu_index} requires CUDA_VISIBLE_DEVICES="
            f"{physical_gpu_index} and CUDA_DEVICE_ORDER=PCI_BUS_ID"
        )


def gpu_lock_path(physical_gpu_index: int) -> Path:
    return Path(
        GPU_LOCK_PATH_TEMPLATE.format(index=_validate_gpu_index(physical_gpu_index))
    )


def _validate_run_token(value: str) -> str:
    if not isinstance(value, str) or RUN_TOKEN_RE.fullmatch(value) is None:
        raise DualDp1GenerationError(
            "run_token must be exactly 64 lowercase hex characters"
        )
    return value


def _control_paths(control_dir: Path, shard_id: int) -> dict[str, Path]:
    shard_id = _validate_shard_id(shard_id)
    unresolved = control_dir.expanduser()
    if unresolved.is_symlink():
        raise DualDp1GenerationError("control directory is a symlink")
    control_dir = unresolved.resolve()
    if not control_dir.is_dir():
        raise DualDp1GenerationError("control directory does not exist")
    return {
        "control_dir": control_dir,
        "ready": control_dir / f"shard{shard_id}.ready.json",
        "go": control_dir / "go.json",
        "done": control_dir / f"shard{shard_id}.done.json",
    }


def _binding_with_payload(path: Path) -> dict[str, Any]:
    value = core._read_json(path)
    payload = validate_manifest_integrity(value)
    return {**_file_binding(path), "payload_sha256": payload}


@contextlib.contextmanager
def inherited_gpu_lease(*, physical_gpu_index: int, inherited_lock_fd: int):
    """Validate, but never reacquire, a supervisor-owned canonical GPU lock."""

    index = _validate_gpu_index(physical_gpu_index)
    if (
        not isinstance(inherited_lock_fd, int)
        or isinstance(inherited_lock_fd, bool)
        or inherited_lock_fd < 3
    ):
        raise DualDp1GenerationError(
            "inherited_lock_fd must be an open descriptor >= 3"
        )
    unresolved = gpu_lock_path(index).expanduser()
    if unresolved.is_symlink():
        raise DualDp1GenerationError("canonical GPU lock path is a symlink")
    try:
        path_stat = os.stat(unresolved, follow_symlinks=False)
        fd_stat = os.fstat(inherited_lock_fd)
    except OSError as exc:
        raise DualDp1GenerationError(
            f"cannot inspect inherited GPU lock: {exc}"
        ) from exc
    if (
        not stat.S_ISREG(path_stat.st_mode)
        or not stat.S_ISREG(fd_stat.st_mode)
        or path_stat.st_nlink != 1
        or fd_stat.st_nlink != 1
        or (path_stat.st_dev, path_stat.st_ino) != (fd_stat.st_dev, fd_stat.st_ino)
    ):
        raise DualDp1GenerationError(
            "inherited FD is not the canonical regular GPU lock"
        )
    proc_link = Path(f"/proc/self/fd/{inherited_lock_fd}")
    try:
        if proc_link.resolve() != unresolved.resolve():
            raise DualDp1GenerationError(
                "inherited FD target path does not match GPU lock"
            )
    except OSError as exc:
        raise DualDp1GenerationError(
            f"cannot resolve inherited GPU lock FD: {exc}"
        ) from exc
    # A separately opened file description must be excluded by the supervisor's
    # exclusive flock.  Acquiring it would prove that no supervising lock exists.
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    probe_fd = os.open(unresolved, flags)
    try:
        try:
            fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe_fd, fcntl.LOCK_UN)
            raise DualDp1GenerationError(
                "supervisor does not hold the inherited GPU lock"
            )
    finally:
        os.close(probe_fd)
    external = _external_gpu_processes(index)
    if external:
        raise DualDp1GenerationError(
            f"GPU{index} is not idle under the inherited supervisor lease: {external}"
        )
    yield {
        "lease_mode": "supervisor_preacquired_inherited_fd",
        "physical_gpu_index": index,
        "lock_path": str(unresolved.resolve()),
        "lock_fd": inherited_lock_fd,
        "lock_device": fd_stat.st_dev,
        "lock_inode": fd_stat.st_ino,
        "worker_pid": os.getpid(),
        "validated_at_utc": core._utc_now(),
        "external_compute_pids_at_acquire": external,
    }


def _publish_ready_and_wait_go(
    *,
    control_dir: Path,
    run_token: str,
    model_id: str,
    shard_id: int,
    physical_gpu_index: int,
    evaluation_scope: str,
    max_num_seqs: int,
    formal_authorization: Mapping[str, Any] | None,
    engine_runtime: Mapping[str, Any],
    timeout_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Seal readiness after model load, then block until the shared GO file."""

    run_token = _validate_run_token(run_token)
    paths = _control_paths(control_dir, shard_id)
    if paths["ready"].exists() or paths["done"].exists():
        raise DualDp1GenerationError("worker control artifacts already exist")
    binding = engine_runtime.get("engine_core_gpu_binding")
    identity = engine_runtime.get("physical_gpu_identity")
    if not isinstance(binding, Mapping) or not isinstance(identity, Mapping):
        raise DualDp1GenerationError("engine/GPU evidence missing before ready barrier")
    ready = seal_manifest(
        {
            "schema_version": READY_SCHEMA,
            "status": "ready",
            "created_at_utc": core._utc_now(),
            "evaluation_id": EVALUATION_ID,
            "run_token": run_token,
            "model_id": model_id,
            "shard_id": shard_id,
            "physical_gpu_index": physical_gpu_index,
            "evaluation_scope": evaluation_scope,
            "max_num_seqs": max_num_seqs,
            "formal_authorization": copy.deepcopy(formal_authorization),
            "required_vllm_environment": {
                key: os.environ.get(key) for key in REQUIRED_VLLM_ENV
            },
            "present_vllm_dp_environment": sorted(
                key for key in os.environ if key.startswith("VLLM_DP_")
            ),
            "worker_pid": binding["worker_pid"],
            "engine_pid": binding["engine_core_pid"],
            "engine_core_pid": binding["engine_core_pid"],
            "gpu_uuid": identity["uuid"],
            "gpu_pci_bus_id": identity["pci_bus_id"],
            "cuda_visible_devices": str(physical_gpu_index),
        }
    )
    core._write_new_json(paths["ready"], ready)
    ready_binding = _binding_with_payload(paths["ready"])
    started = time.monotonic()
    while not paths["go"].is_file():
        if time.monotonic() - started >= timeout_seconds:
            raise DualDp1GenerationError("timed out waiting for shared GO barrier")
        time.sleep(0.05)
    go = core._read_json(paths["go"])
    validate_manifest_integrity(go)
    ready_manifests = go.get("ready_manifests")
    expected_identity = {
        "schema_version": GO_SCHEMA,
        "status": "released",
        "evaluation_id": EVALUATION_ID,
        "run_token": run_token,
        "model_id": model_id,
        "evaluation_scope": evaluation_scope,
        "max_num_seqs": max_num_seqs,
        "formal_authorization": formal_authorization,
    }
    for key, expected in expected_identity.items():
        if go.get(key) != expected:
            raise DualDp1GenerationError(f"shared GO contract drift at {key}")
    if (
        not isinstance(go.get("go_epoch_ns"), int)
        or isinstance(go.get("go_epoch_ns"), bool)
        or go["go_epoch_ns"] <= 0
        or not isinstance(go.get("go_monotonic_ns"), int)
        or isinstance(go.get("go_monotonic_ns"), bool)
        or go["go_monotonic_ns"] <= 0
        or not isinstance(go.get("go_at_utc"), str)
        or go.get("physical_gpu_indexes") != [0, 1]
        or not isinstance(ready_manifests, Mapping)
        or ready_manifests.get(f"shard{shard_id}") != ready_binding
        or set(ready_manifests) != {"shard0", "shard1"}
    ):
        raise DualDp1GenerationError("shared GO readiness/timing binding drift")
    return ready_binding, {
        "manifest": go,
        "binding": _binding_with_payload(paths["go"]),
    }


def _publish_done(
    *,
    control_dir: Path,
    run_token: str,
    model_id: str,
    shard_id: int,
    physical_gpu_index: int,
    evaluation_scope: str,
    max_num_seqs: int,
    formal_authorization: Mapping[str, Any] | None,
    generation_started_epoch_ns: int,
    generation_finished_epoch_ns: int,
    observed_go_epoch_ns: int,
    observed_go_monotonic_ns: int,
    generation_started_monotonic_ns: int,
    generation_finished_monotonic_ns: int,
    generated_output_tokens: int,
    completed_requests: int,
    resume_count: int,
) -> dict[str, Any]:
    paths = _control_paths(control_dir, shard_id)
    payload = seal_manifest(
        {
            "schema_version": DONE_SCHEMA,
            "status": "done",
            "created_at_utc": core._utc_now(),
            "evaluation_id": EVALUATION_ID,
            "run_token": _validate_run_token(run_token),
            "model_id": model_id,
            "shard_id": shard_id,
            "physical_gpu_index": physical_gpu_index,
            "evaluation_scope": evaluation_scope,
            "max_num_seqs": max_num_seqs,
            "formal_authorization": copy.deepcopy(formal_authorization),
            "generation_started_epoch_ns": generation_started_epoch_ns,
            "generation_finished_epoch_ns": generation_finished_epoch_ns,
            "observed_go_epoch_ns": observed_go_epoch_ns,
            "observed_go_monotonic_ns": observed_go_monotonic_ns,
            "generation_started_monotonic_ns": generation_started_monotonic_ns,
            "generation_finished_monotonic_ns": generation_finished_monotonic_ns,
            "generated_output_tokens": generated_output_tokens,
            "completed_requests": completed_requests,
            "resume_count": resume_count,
        }
    )
    core._write_new_json(paths["done"], payload)
    return _binding_with_payload(paths["done"])


def _recover_one_record_ahead_done(
    *,
    engine_runtime: Mapping[str, Any],
    model_id: str,
    shard_id: int,
    physical_gpu_index: int,
    evaluation_scope: str,
    max_num_seqs: int,
    formal_authorization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Recover DONE from the current attempt, including repeated resumes."""

    ready_binding = engine_runtime.get("ready_manifest")
    go_binding = engine_runtime.get("go_manifest")
    control_attempt = engine_runtime.get("control_attempt")
    if (
        not isinstance(ready_binding, Mapping)
        or not isinstance(go_binding, Mapping)
        or not isinstance(control_attempt, Mapping)
    ):
        raise DualDp1GenerationError(
            "completed shard lacks current control evidence for closure resume"
        )
    ready_path = _verify_arbitrary_binding(ready_binding)
    go_path = _verify_arbitrary_binding(go_binding)
    control_dir = ready_path.parent
    done_path = control_dir / f"shard{shard_id}.done.json"
    if (
        ready_path.name != f"shard{shard_id}.ready.json"
        or go_path != control_dir / "go.json"
        or control_attempt
        != {
            "control_dir": str(control_dir),
            "run_token_sha256": control_attempt.get("run_token_sha256"),
            "ready_path": str(ready_path),
            "go_path": str(go_path),
            "done_path": str(done_path),
        }
        or done_path.is_symlink()
        or not done_path.is_file()
    ):
        raise DualDp1GenerationError("closure-resume control path binding drift")
    ready = core._read_json(ready_path)
    go = core._read_json(go_path)
    done = core._read_json(done_path)
    for control_payload in (ready, go, done):
        validate_manifest_integrity(control_payload)
    if (
        done.get("schema_version") != DONE_SCHEMA
        or done.get("status") != "done"
        or done.get("evaluation_id") != EVALUATION_ID
        or done.get("model_id") != model_id
        or done.get("shard_id") != shard_id
        or done.get("physical_gpu_index") != physical_gpu_index
        or done.get("evaluation_scope") != evaluation_scope
        or done.get("max_num_seqs") != max_num_seqs
        or ready.get("formal_authorization") != formal_authorization
        or go.get("formal_authorization") != formal_authorization
        or done.get("formal_authorization") != formal_authorization
        or done.get("run_token") != ready.get("run_token")
        or done.get("run_token") != go.get("run_token")
        or control_attempt.get("run_token_sha256")
        != _sha256_text(str(done.get("run_token")))
        or done.get("generated_output_tokens")
        != engine_runtime.get("generated_output_tokens")
        or done.get("completed_requests")
        != engine_runtime.get("completed_requests_in_timed_session")
        or done.get("resume_count") != engine_runtime.get("engine_session_resume_count")
    ):
        raise DualDp1GenerationError("one-record-ahead DONE evidence drift")
    recovered = copy.deepcopy(dict(engine_runtime))
    for key in (
        "observed_go_epoch_ns",
        "observed_go_monotonic_ns",
        "generation_started_epoch_ns",
        "generation_started_monotonic_ns",
        "generation_finished_epoch_ns",
        "generation_finished_monotonic_ns",
    ):
        if key in done:
            recovered[key] = done[key]
    recovered["done_manifest"] = _binding_with_payload(done_path)
    recovered["engine_session_resume_count"] = done.get("resume_count")
    return recovered


def _source_hashes(
    cohort_path: Path,
    cohort_sha256: str,
    cohort: Mapping[str, Any],
    sample_manifest: Mapping[str, Any],
    max_num_seqs: int,
    formal_authorization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_sample = cohort["source_sample_manifest"]
    hashes = core._source_hashes(sample_manifest, str(source_sample["sha256"]))
    hashes.update(
        {
            "vllm_k5_cohort_sha256": cohort_sha256,
            "vllm_k5_token_ledger_sha256": cohort["token_ledger"]["sha256"],
            "vllm_k5_dual_dp1_sampling_contract_sha256": _sampling_contract_sha256(
                max_num_seqs=max_num_seqs
            ),
            "vllm_k5_dp2_to_dual_dp1_remediation_receipt": (
                _remediation_receipt_binding()
            ),
            "vllm_k5_formal_generation_authorization": copy.deepcopy(
                formal_authorization
            ),
        }
    )
    hashes["implementation_sources"].update(
        {
            "vllm_k5_dual_dp1_runner": _file_binding(Path(__file__).resolve()),
            "vllm_k5_dp2_shared_validation": _file_binding(
                Path(dp2.__file__).resolve()
            ),
            "vllm_k5_preparer": _file_binding(Path(dp2.preparation.__file__).resolve()),
        }
    )
    return hashes


def _remediation_receipt_binding() -> dict[str, Any]:
    """Deep-validate and bind the topology authorization at artifact level."""

    try:
        loaded = remediation.load_and_validate_receipt(remediation.DEFAULT_RECEIPT)
    except remediation.DualDp1RemediationError as exc:
        raise DualDp1GenerationError(str(exc)) from exc
    if loaded.get("evaluation_id") != EVALUATION_ID:
        raise DualDp1GenerationError("topology remediation evaluation ID drift")
    binding = loaded.get("receipt_binding")
    if not isinstance(binding, Mapping):
        raise DualDp1GenerationError("topology remediation binding is missing")
    return copy.deepcopy(dict(binding))


def _load_formal_authorization(
    authorization_path: Path,
    *,
    cohort_path: Path,
    cohort_sha256: str,
    max_num_seqs: int,
) -> dict[str, Any]:
    """Deep-validate the official smoke replay gate without importing vLLM.

    The suite module imports this runner, so keep the import local: at runtime
    this module is fully initialized, while import-time coupling would create a
    cycle.  The returned object is the one canonical authorization record used
    by the supervisor, workers, rows, timing record, and merged manifest.
    """

    from jobs.eval import seal_chk3_beta_core8_vllm_k5_dual_dp1_suite as suite

    try:
        loaded = suite.load_and_validate_suite(
            authorization_path,
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            expected_scope="infrastructure_smoke",
            max_num_seqs=max_num_seqs,
        )
    except Exception as exc:
        raise DualDp1GenerationError(
            f"formal authorization deep validation failed: {exc}"
        ) from exc
    manifest = loaded.get("manifest")
    manifest_binding = loaded.get("manifest_binding")
    if not isinstance(manifest, Mapping) or not isinstance(manifest_binding, Mapping):
        raise DualDp1GenerationError("formal authorization suite evidence is missing")
    gates = manifest.get("generation_gates")
    selection = manifest.get("benchmark_selection")
    required_gates = {
        "benchmark_candidate_exact_token_parity_8_12_16": True,
        "selected_chk1_smoke_exact_replay": True,
        "formal_generation_unblocked": True,
    }
    if (
        not isinstance(gates, Mapping)
        or any(gates.get(key) is not value for key, value in required_gates.items())
        or not isinstance(selection, Mapping)
        or manifest.get("remediation_receipt") != _remediation_receipt_binding()
    ):
        raise DualDp1GenerationError("official smoke replay gate is not authorized")
    return {
        "official_smoke_suite": copy.deepcopy(dict(manifest_binding)),
        "benchmark_selection": copy.deepcopy(dict(selection)),
        "generation_gates": copy.deepcopy(dict(gates)),
    }


def _formal_authorization_for_scope(
    *,
    scope: str,
    authorization_path: Path | None,
    cohort_path: Path,
    cohort_sha256: str,
    max_num_seqs: int,
) -> dict[str, Any] | None:
    if scope == "infrastructure_smoke":
        if authorization_path is not None:
            raise DualDp1GenerationError(
                "formal authorization is forbidden for infrastructure smoke"
            )
        return None
    if scope != "formal_merged_panel":
        raise DualDp1GenerationError("invalid generation scope")
    if authorization_path is None:
        raise DualDp1GenerationError(
            "formal generation requires a deeply validated official smoke authorization"
        )
    return _load_formal_authorization(
        authorization_path,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
        max_num_seqs=max_num_seqs,
    )


def _revalidate_formal_authorization(
    authorization: Any,
    *,
    scope: str,
    cohort_path: Path,
    cohort_sha256: str,
    max_num_seqs: int,
) -> dict[str, Any] | None:
    if scope == "infrastructure_smoke":
        if authorization is not None:
            raise DualDp1GenerationError(
                "infrastructure smoke must not carry formal authorization"
            )
        return None
    if not isinstance(authorization, Mapping):
        raise DualDp1GenerationError("sealed formal authorization is missing")
    smoke_binding = authorization.get("official_smoke_suite")
    if not isinstance(smoke_binding, Mapping):
        raise DualDp1GenerationError("official smoke suite binding is missing")
    path = _verify_arbitrary_binding(smoke_binding)
    observed = _load_formal_authorization(
        path,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
        max_num_seqs=max_num_seqs,
    )
    if dict(authorization) != observed:
        raise DualDp1GenerationError("sealed formal authorization binding drift")
    return observed


def build_result(
    *,
    model_id: str,
    model_label: str,
    case: Mapping[str, Any],
    source_row: Mapping[str, Any],
    sample_manifest_sha256: str,
    source_artifact_sha256s: Mapping[str, Any],
    generated_text: str,
    generated_token_ids: Sequence[int],
    eos_token_ids: Sequence[int],
    pad_token_id: int,
    raw_finish_reason: Any,
    raw_stop_reason: Any,
    max_num_seqs: int,
    shard_id: int,
    physical_gpu_index: int,
) -> dict[str, Any]:
    shard_id, physical_gpu_index = _validate_worker_mapping(
        shard_id, physical_gpu_index
    )
    if assigned_shard(int(case["absolute_case_index"])) != shard_id:
        raise DualDp1GenerationError("case was submitted to the wrong parity shard")
    try:
        row = dp2.build_result(
            model_id=model_id,
            model_label=model_label,
            case=case,
            source_row=source_row,
            sample_manifest_sha256=sample_manifest_sha256,
            source_artifact_sha256s=source_artifact_sha256s,
            generated_text=generated_text,
            generated_token_ids=generated_token_ids,
            eos_token_ids=eos_token_ids,
            pad_token_id=pad_token_id,
            raw_finish_reason=raw_finish_reason,
            raw_stop_reason=raw_stop_reason,
            max_num_seqs=max_num_seqs,
        )
    except dp2.VllmK5GenerationError as exc:
        raise DualDp1GenerationError(str(exc)) from exc
    row.update(
        {
            "schema_version": ROW_SCHEMA,
            "evaluation_id": EVALUATION_ID,
            "inference_backend": "vllm-async-engine-v1-independent-dp1",
            "data_parallel_size": DATA_PARALLEL_SIZE,
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "physical_gpu_indexes": [physical_gpu_index],
            "physical_gpu_index": physical_gpu_index,
            "shard_id": shard_id,
            "shard_function": "absolute_case_index_mod_2",
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_num_seqs": _validate_max_num_seqs(max_num_seqs),
            "sampling_contract_sha256": _sampling_contract_sha256(
                max_num_seqs=max_num_seqs
            ),
        }
    )
    return row


def validate_result(row: Mapping[str, Any], **kwargs: Any) -> None:
    expected = build_result(
        **kwargs,
        generated_text=str(row.get("generated_text") or ""),
        generated_token_ids=row.get("generated_token_ids") or [],
        eos_token_ids=row.get("eos_token_ids") or [],
        pad_token_id=row.get("pad_token_id"),
        raw_finish_reason=row.get("vllm_raw_finish_reason"),
        raw_stop_reason=row.get("vllm_raw_stop_reason"),
    )
    if dict(row) != expected:
        changed = sorted(
            key for key in set(row) | set(expected) if row.get(key) != expected.get(key)
        )
        raise DualDp1GenerationError(
            f"persisted shard row drift at fields: {changed[:12]}"
        )


def _expected_total_cases(*, smoke: bool) -> int:
    return SMOKE_CASES_PER_MODEL if smoke else CASES_PER_MODEL


def _expected_shard_cases(*, smoke: bool) -> int:
    return SMOKE_CASES_PER_SHARD if smoke else FORMAL_CASES_PER_SHARD


def _chunk_indexes(chunk_id: int, shard_id: int, *, total_cases: int) -> list[int]:
    start = chunk_id * ABSOLUTE_CHUNK_SIZE
    end = min(start + ABSOLUTE_CHUNK_SIZE, total_cases)
    return [index for index in range(start, end) if assigned_shard(index) == shard_id]


def _completion_status(
    indexes: set[int], *, shard_id: int, total_cases: int
) -> tuple[int, set[int]]:
    shard_id = _validate_shard_id(shard_id)
    allowed_all = {
        index for index in range(total_cases) if assigned_shard(index) == shard_id
    }
    if not indexes.issubset(allowed_all):
        raise DualDp1GenerationError("shard WAL has wrong-parity/out-of-range indexes")
    chunks = math.ceil(total_cases / ABSOLUTE_CHUNK_SIZE)
    complete = 0
    while complete < chunks:
        required = set(_chunk_indexes(complete, shard_id, total_cases=total_cases))
        if required.issubset(indexes):
            complete += 1
        else:
            break
    previous = {
        index
        for chunk_id in range(complete)
        for index in _chunk_indexes(chunk_id, shard_id, total_cases=total_cases)
    }
    active = (
        set(_chunk_indexes(complete, shard_id, total_cases=total_cases))
        if complete < chunks
        else set()
    )
    extras = indexes - previous
    if not extras.issubset(active):
        raise DualDp1GenerationError("shard WAL skips an incomplete absolute chunk")
    return complete, extras


def validate_wal_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    cases: Sequence[Mapping[str, Any]],
    model_id: str,
    model_label: str,
    bound_rows: Mapping[str, Mapping[str, Any]],
    sample_manifest_sha256: str,
    source_artifact_sha256s: Mapping[str, Any],
    max_num_seqs: int,
    shard_id: int,
    physical_gpu_index: int,
) -> dict[int, dict[str, Any]]:
    shard_id, physical_gpu_index = _validate_worker_mapping(
        shard_id, physical_gpu_index
    )
    observed_indexes = [row.get("absolute_case_index") for row in rows]
    if any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= len(cases)
        or assigned_shard(index) != shard_id
        for index in observed_indexes
    ) or len(set(observed_indexes)) != len(observed_indexes):
        raise DualDp1GenerationError("duplicate/invalid/wrong-parity shard WAL index")
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = row.get("absolute_case_index")
        assert isinstance(index, int)
        case = cases[index]
        validate_result(
            row,
            model_id=model_id,
            model_label=model_label,
            case=case,
            source_row=bound_rows[str(case["sample"]["sample_id"])],
            sample_manifest_sha256=sample_manifest_sha256,
            source_artifact_sha256s=source_artifact_sha256s,
            max_num_seqs=max_num_seqs,
            shard_id=shard_id,
            physical_gpu_index=physical_gpu_index,
        )
        by_index[index] = dict(row)
    _completion_status(set(by_index), shard_id=shard_id, total_cases=len(cases))
    return by_index


def _chunk_receipt(
    *,
    model_id: str,
    shard_id: int,
    chunk_id: int,
    rows_by_index: Mapping[int, Mapping[str, Any]],
    total_cases: int,
) -> dict[str, Any]:
    indexes = _chunk_indexes(chunk_id, shard_id, total_cases=total_cases)
    if len(indexes) != CASES_PER_SHARD_CHUNK:
        raise DualDp1GenerationError("shard chunk does not contain exactly 20 cases")
    bindings = []
    for index in indexes:
        row = rows_by_index.get(index)
        if row is None:
            raise DualDp1GenerationError("cannot receipt an incomplete shard chunk")
        bindings.append(
            {
                "absolute_case_index": index,
                "sample_id": row["sample_id"],
                "replicate_id": row["replicate_id"],
                "row_seed": row["row_seed"],
                "completion_sha256": row["completion_sha256"],
                "generated_token_ids_sha256": row["generated_token_ids_sha256"],
            }
        )
    return {
        "schema_version": CHUNK_RECEIPT_SCHEMA,
        "evaluation_id": EVALUATION_ID,
        "model_id": model_id,
        "shard_id": shard_id,
        "absolute_chunk_id": chunk_id,
        "cases": len(indexes),
        "absolute_case_indexes": indexes,
        "canonical_row_bindings_sha256": _sha256_text(_canonical(bindings)),
    }


def _validate_receipts(
    receipts: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    shard_id: int,
    rows_by_index: Mapping[int, Mapping[str, Any]],
    total_cases: int,
) -> None:
    complete, _ = _completion_status(
        set(rows_by_index), shard_id=shard_id, total_cases=total_cases
    )
    if len(receipts) > complete or len(receipts) < complete - 1:
        raise DualDp1GenerationError("shard receipt/WAL closure drift")
    for chunk_id, receipt in enumerate(receipts):
        expected = _chunk_receipt(
            model_id=model_id,
            shard_id=shard_id,
            chunk_id=chunk_id,
            rows_by_index=rows_by_index,
            total_cases=total_cases,
        )
        if dict(receipt) != expected:
            raise DualDp1GenerationError(f"shard chunk receipt drift at {chunk_id}")


def _state_payload(
    *,
    status: str,
    model_id: str,
    shard_id: int,
    expected_cases: int,
    total_cases: int,
    resume_count: int,
    wal_binding: Mapping[str, Any],
    receipt_binding: Mapping[str, Any],
    completed_indexes: set[int],
    formal_authorization: Mapping[str, Any] | None,
    engine_runtime: Mapping[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    complete, partial = _completion_status(
        completed_indexes, shard_id=shard_id, total_cases=total_cases
    )
    return {
        "schema_version": STATE_SCHEMA,
        "status": status,
        "updated_at_utc": core._utc_now(),
        "evaluation_id": EVALUATION_ID,
        "model_id": model_id,
        "shard_id": shard_id,
        "formal_authorization": copy.deepcopy(formal_authorization),
        "expected_cases": expected_cases,
        "completed_cases": len(completed_indexes),
        "completed_absolute_chunks": complete,
        "wal_complete_absolute_chunks": complete,
        "committed_receipted_chunks": receipt_binding.get("rows"),
        "active_chunk_completed_indexes": sorted(partial),
        "resume_count": resume_count,
        "completion_wal": dict(wal_binding),
        "chunk_receipts": dict(receipt_binding),
        "engine_runtime": copy.deepcopy(engine_runtime),
        **extra,
    }


def _write_state(path: Path, payload: Mapping[str, Any]) -> None:
    core._atomic_write_json(path, seal_manifest(payload))


def _append_fsync(handle: Any, tracker: WalTracker, value: Mapping[str, Any]) -> None:
    dp2._append_fsync(handle, tracker, value)


def _write_or_validate_canonical_jsonl(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    """Create a canonical JSONL once, or exact-validate a crash-left artifact."""

    if path.exists() or path.is_symlink():
        existing, _, _ = _read_canonical_jsonl(path)
        if existing != [dict(row) for row in rows]:
            raise DualDp1GenerationError(f"existing canonical output drift: {path}")
        return
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical(dict(row)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    core._fsync_directory(path.parent)


def _exact_shard_union(
    shard_results: Mapping[int, Sequence[Mapping[str, Any]]], *, total_cases: int
) -> dict[int, dict[str, Any]]:
    if set(shard_results) != set(ALLOWED_SHARD_IDS):
        raise DualDp1GenerationError("exact union requires shard0 and shard1")
    by_index: dict[int, dict[str, Any]] = {}
    for shard_id in ALLOWED_SHARD_IDS:
        for row in shard_results[shard_id]:
            index = row.get("absolute_case_index")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or index in by_index
                or assigned_shard(index) != shard_id
            ):
                raise DualDp1GenerationError("shard union overlaps or violates parity")
            by_index[index] = dict(row)
    if set(by_index) != set(range(total_cases)):
        raise DualDp1GenerationError("two shards do not form an exact case union")
    return by_index


def _runtime_versions() -> dict[str, Any]:
    return dp2._runtime_versions()


def _physical_gpu_identity(index: int) -> dict[str, Any]:
    try:
        identity = dp2._physical_gpu_identity(index)
    except dp2.VllmK5GenerationError as exc:
        raise DualDp1GenerationError(str(exc)) from exc
    expected = EXPECTED_GPU_IDENTITIES[_validate_gpu_index(index)]
    if (
        identity.get("uuid") != expected["uuid"]
        or str(identity.get("pci_bus_id", "")).lower() != expected["pci_bus_id"]
    ):
        raise DualDp1GenerationError(f"physical GPU{index} frozen identity drift")
    return identity


def _verify_visible_cuda_device(
    torch_module: Any, identity: Mapping[str, Any]
) -> dict[str, Any]:
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 1:
        raise DualDp1GenerationError("exactly one logical CUDA device is required")
    properties = torch_module.cuda.get_device_properties(0)
    if core._normalise_gpu_uuid(
        getattr(properties, "uuid", None)
    ) != core._normalise_gpu_uuid(identity["uuid"]):
        raise DualDp1GenerationError(
            "logical cuda:0 UUID does not match physical target"
        )
    observed_pci = getattr(properties, "pci_bus_id", None)
    if (
        observed_pci is not None
        and str(observed_pci).lower() != str(identity["pci_bus_id"]).lower()
    ):
        raise DualDp1GenerationError(
            "logical cuda:0 PCI bus does not match physical target"
        )
    return {"logical_cuda_index": 0, **dict(identity)}


def _external_gpu_processes(index: int) -> list[dict[str, Any]]:
    try:
        return dp2._external_gpu_compute_processes(index)
    except dp2.VllmK5GenerationError as exc:
        raise DualDp1GenerationError(str(exc)) from exc


def _descendant_pids() -> set[int]:
    try:
        return dp2._descendant_pids()
    except dp2.VllmK5GenerationError as exc:
        raise DualDp1GenerationError(str(exc)) from exc


@contextlib.contextmanager
def exclusive_gpu_lease(
    *,
    physical_gpu_index: int,
    timeout_seconds: int,
    poll_seconds: int,
    on_wait: Any = None,
):
    index = _validate_gpu_index(physical_gpu_index)
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise DualDp1GenerationError("GPU wait timeout/poll must be positive")
    path = gpu_lock_path(index).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise DualDp1GenerationError(f"GPU lock is a symlink: {path}")
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    handle = os.fdopen(descriptor, "a+", encoding="utf-8")
    acquired = False
    started = time.monotonic()
    last_notice = 0.0
    try:
        mode = os.fstat(handle.fileno()).st_mode
        if not stat.S_ISREG(mode) or os.fstat(handle.fileno()).st_nlink != 1:
            raise DualDp1GenerationError(
                "GPU lock must be one regular hard-link-free file"
            )
        while not acquired:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                elapsed = time.monotonic() - started
                if elapsed >= timeout_seconds:
                    raise DualDp1GenerationError(
                        f"timed out waiting for GPU{index} lock"
                    )
                if on_wait is not None and elapsed - last_notice >= 60:
                    on_wait("waiting_for_gpu_lock", [])
                    last_notice = elapsed
                time.sleep(min(poll_seconds, max(0.1, timeout_seconds - elapsed)))
        consecutive_idle = 0
        while consecutive_idle < 2:
            processes = _external_gpu_processes(index)
            elapsed = time.monotonic() - started
            consecutive_idle = consecutive_idle + 1 if not processes else 0
            if processes and on_wait is not None and elapsed - last_notice >= 60:
                on_wait("waiting_for_external_compute_pids", processes)
                last_notice = elapsed
            if consecutive_idle < 2:
                if elapsed >= timeout_seconds:
                    raise DualDp1GenerationError(
                        f"timed out waiting for GPU{index} idle"
                    )
                time.sleep(min(poll_seconds, max(0.1, timeout_seconds - elapsed)))
        yield {
            "physical_gpu_index": index,
            "lock_path": str(path.resolve()),
            "acquired_at_utc": core._utc_now(),
            "wait_seconds": round(time.monotonic() - started, 3),
            "external_compute_pids_at_acquire": [],
        }
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _engine_core_pid(engine: Any) -> int:
    client = getattr(engine, "engine_core", None)
    core_engine = getattr(client, "core_engine", None)
    if core_engine is None:
        engines = getattr(client, "core_engines", None)
        if isinstance(engines, list) and len(engines) == 1:
            core_engine = engines[0]
    if core_engine is None:
        resources = getattr(client, "resources", None)
        engines = getattr(resources, "core_engines", None)
        if isinstance(engines, list) and len(engines) == 1:
            core_engine = engines[0]
    process = getattr(getattr(core_engine, "proc_handle", None), "proc", None)
    pid = getattr(process, "pid", None)
    if (
        getattr(core_engine, "index", None) != 0
        or not isinstance(pid, int)
        or isinstance(pid, bool)
        or pid <= 0
        or not bool(process.is_alive())
    ):
        raise DualDp1GenerationError("vLLM did not create one live DP=1 EngineCore")
    return pid


def _verify_engine_gpu_process(
    engine: Any,
    *,
    physical_gpu_index: int,
    timeout_seconds: float = 60.0,
    poll_seconds: float = 0.5,
) -> dict[str, Any]:
    engine_pid = _engine_core_pid(engine)
    deadline = time.monotonic() + timeout_seconds
    observed: list[dict[str, Any]] = []
    while True:
        allowed = _descendant_pids()
        if engine_pid not in allowed:
            raise DualDp1GenerationError("EngineCore escaped the worker process tree")
        observed = _external_gpu_processes(physical_gpu_index)
        foreign = [p for p in observed if int(p["pid"]) not in allowed]
        if foreign:
            raise DualDp1GenerationError(
                f"foreign GPU process after model load: {foreign}"
            )
        observed_pids = {int(p["pid"]) for p in observed}
        if observed_pids == {engine_pid}:
            return {
                "worker_pid": os.getpid(),
                "engine_core_pid": engine_pid,
                "physical_gpu_index": physical_gpu_index,
                "descendant_compute_processes": copy.deepcopy(observed),
            }
        if time.monotonic() >= deadline:
            raise DualDp1GenerationError(
                f"could not prove EngineCore {engine_pid} on GPU{physical_gpu_index}: {observed}"
            )
        time.sleep(poll_seconds)


def _assert_no_foreign_processes(index: int) -> None:
    allowed = _descendant_pids()
    foreign = [
        p for p in _external_gpu_processes(index) if int(p["pid"]) not in allowed
    ]
    if foreign:
        raise DualDp1GenerationError(f"foreign GPU{index} process appeared: {foreign}")


async def _consume_request(
    engine: Any, sampling_params_cls: Any, case: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Any]:
    return await dp2._consume_request(engine, sampling_params_cls, case)


async def _run_async_generation(
    *,
    anchor: Mapping[str, Any],
    tokenizer_path: Path,
    pending_chunks: Sequence[tuple[int, Sequence[Mapping[str, Any]]]],
    on_completed: Any,
    on_chunk_completed: Any,
    max_num_seqs: int,
    physical_gpu_identity: Mapping[str, Any],
    gpu_lease: Mapping[str, Any],
    control_dir: Path,
    run_token: str,
    model_id: str,
    shard_id: int,
    evaluation_scope: str,
    formal_authorization: Mapping[str, Any] | None,
    control_timeout_seconds: int,
    resume_count: int,
) -> dict[str, Any]:
    import torch
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    if vllm.__version__ != EXPECTED_VLLM_VERSION:
        raise DualDp1GenerationError(
            f"expected vLLM {EXPECTED_VLLM_VERSION}, got {vllm.__version__}"
        )
    max_num_seqs = _validate_max_num_seqs(max_num_seqs)
    physical_index = int(physical_gpu_identity["physical_gpu_index"])
    visible = _verify_visible_cuda_device(torch, physical_gpu_identity)
    args = AsyncEngineArgs(
        model=str(anchor["model_path"]),
        tokenizer=str(tokenizer_path),
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        quantization=None,
        load_format="safetensors",
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        data_parallel_size=1,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        swap_space=0,
        cpu_offload_gb=0,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        enforce_eager=False,
        disable_custom_all_reduce=True,
        generation_config="vllm",
        seed=0,
        disable_log_requests=True,
        disable_log_stats=True,
    )
    load_started = time.perf_counter()
    engine = AsyncLLMEngine.from_engine_args(args)
    try:
        parallel = engine.vllm_config.parallel_config
        model_config = engine.vllm_config.model_config
        compilation = engine.vllm_config.compilation_config
        if (
            parallel.data_parallel_size != 1
            or parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or model_config.enforce_eager is not False
            or compilation.use_cudagraph is not True
            or model_config.use_async_output_proc is not True
        ):
            raise DualDp1GenerationError(
                "resolved vLLM independent DP=1 topology drift"
            )
        binding = _verify_engine_gpu_process(engine, physical_gpu_index=physical_index)
        runtime: dict[str, Any] = {
            **_runtime_versions(),
            "worker_pid": os.getpid(),
            "visible_cuda_device": visible,
            "engine_core_gpu_binding": binding,
            "physical_gpu_identity": dict(physical_gpu_identity),
            "gpu_lease": copy.deepcopy(dict(gpu_lease)),
            "model_load_wall_seconds": time.perf_counter() - load_started,
            "generation_wall_seconds": None,
            "generated_output_tokens": 0,
            "completed_requests_in_timed_session": 0,
            "output_tokens_per_second": None,
            "enforce_eager": False,
            "cuda_graphs": True,
            "async_output_processing": bool(model_config.use_async_output_proc),
            "engine": sampling_contract(max_num_seqs=max_num_seqs),
            "engine_session_resume_count": resume_count,
            "control_attempt": {
                "control_dir": str(control_dir.resolve()),
                "run_token_sha256": _sha256_text(run_token),
                "ready_path": str(
                    (control_dir / f"shard{shard_id}.ready.json").resolve()
                ),
                "go_path": str((control_dir / "go.json").resolve()),
                "done_path": str(
                    (control_dir / f"shard{shard_id}.done.json").resolve()
                ),
            },
        }
        tokenizer = await engine.get_tokenizer()
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if not isinstance(eos_id, int) or isinstance(eos_id, bool):
            raise DualDp1GenerationError("tokenizer has no scalar EOS token")
        pad_id = getattr(tokenizer, "pad_token_id", None)
        pad_id = eos_id if pad_id is None else pad_id
        ready_binding, go_evidence = _publish_ready_and_wait_go(
            control_dir=control_dir,
            run_token=run_token,
            model_id=model_id,
            shard_id=shard_id,
            physical_gpu_index=physical_index,
            evaluation_scope=evaluation_scope,
            max_num_seqs=max_num_seqs,
            formal_authorization=formal_authorization,
            engine_runtime=runtime,
            timeout_seconds=control_timeout_seconds,
        )
        go_file_observed_epoch_ns = time.time_ns()
        go_file_observed_monotonic_ns = time.monotonic_ns()
        # These echoed fields let the supervisor prove that both workers
        # consumed the same sealed GO record.  Separate fields below preserve
        # the actual file-observation timestamps.
        observed_go_epoch_ns = int(go_evidence["manifest"]["go_epoch_ns"])
        observed_go_monotonic_ns = int(go_evidence["manifest"]["go_monotonic_ns"])
        generation_started_epoch_ns = time.time_ns()
        generation_started_monotonic_ns = time.monotonic_ns()
        generation_started = time.perf_counter()
        runtime["ready_manifest"] = ready_binding
        runtime["go_manifest"] = go_evidence["binding"]
        runtime["supervisor_go_epoch_ns"] = go_evidence["manifest"]["go_epoch_ns"]
        runtime["supervisor_go_monotonic_ns"] = go_evidence["manifest"][
            "go_monotonic_ns"
        ]
        runtime["observed_go_epoch_ns"] = observed_go_epoch_ns
        runtime["observed_go_monotonic_ns"] = observed_go_monotonic_ns
        runtime["go_file_observed_epoch_ns"] = go_file_observed_epoch_ns
        runtime["go_file_observed_monotonic_ns"] = go_file_observed_monotonic_ns
        runtime["generation_started_epoch_ns"] = generation_started_epoch_ns
        runtime["generation_started_monotonic_ns"] = generation_started_monotonic_ns
        timed_tokens = 0
        timed_requests = 0

        def update_timing() -> None:
            elapsed = time.perf_counter() - generation_started
            runtime["generation_wall_seconds"] = elapsed
            runtime["generated_output_tokens"] = timed_tokens
            runtime["completed_requests_in_timed_session"] = timed_requests
            runtime["output_tokens_per_second"] = (
                timed_tokens / elapsed if elapsed > 0 else None
            )

        for chunk_id, cases in pending_chunks:
            _assert_no_foreign_processes(physical_index)
            tasks = [
                asyncio.create_task(_consume_request(engine, SamplingParams, case))
                for case in cases
            ]
            try:
                for future in asyncio.as_completed(tasks):
                    case, output = await future
                    if (
                        list(getattr(output, "prompt_token_ids", []) or [])
                        != list(case["token_entry"]["prompt_token_ids"])
                        or len(getattr(output, "outputs", []) or []) != 1
                    ):
                        raise DualDp1GenerationError("vLLM prompt-token ledger drift")
                    completion = output.outputs[0]
                    token_ids = list(completion.token_ids)
                    try:
                        text = core.native_eval.native_probe.common_probe.decode_completion_preserving_boundary(
                            tokenizer, token_ids, [eos_id]
                        )
                    except core.native_eval.native_probe.common_probe.ProbeError as exc:
                        raise DualDp1GenerationError(str(exc)) from exc
                    if completion.text != text:
                        raise DualDp1GenerationError("vLLM completion text/token drift")
                    timed_tokens += len(token_ids)
                    timed_requests += 1
                    update_timing()
                    await on_completed(
                        chunk_id,
                        case,
                        text,
                        token_ids,
                        [eos_id],
                        int(pad_id),
                        completion.finish_reason,
                        completion.stop_reason,
                        runtime,
                    )
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            await on_chunk_completed(chunk_id, runtime)
        update_timing()
        _assert_no_foreign_processes(physical_index)
        generation_finished_epoch_ns = time.time_ns()
        generation_finished_monotonic_ns = time.monotonic_ns()
        runtime["generation_finished_epoch_ns"] = generation_finished_epoch_ns
        runtime["generation_finished_monotonic_ns"] = generation_finished_monotonic_ns
        runtime["done_manifest"] = _publish_done(
            control_dir=control_dir,
            run_token=run_token,
            model_id=model_id,
            shard_id=shard_id,
            physical_gpu_index=physical_index,
            evaluation_scope=evaluation_scope,
            max_num_seqs=max_num_seqs,
            formal_authorization=formal_authorization,
            generation_started_epoch_ns=generation_started_epoch_ns,
            generation_finished_epoch_ns=generation_finished_epoch_ns,
            observed_go_epoch_ns=observed_go_epoch_ns,
            observed_go_monotonic_ns=observed_go_monotonic_ns,
            generation_started_monotonic_ns=generation_started_monotonic_ns,
            generation_finished_monotonic_ns=generation_finished_monotonic_ns,
            generated_output_tokens=timed_tokens,
            completed_requests=timed_requests,
            resume_count=resume_count,
        )
        return runtime
    finally:
        if hasattr(engine, "shutdown"):
            engine.shutdown()
        elif hasattr(engine, "shutdown_background_loop"):
            engine.shutdown_background_loop()


def _summary(
    rows_by_index: Mapping[int, Mapping[str, Any]], *, smoke: bool
) -> dict[str, Any]:
    expected = _expected_shard_cases(smoke=smoke)
    if len(rows_by_index) != expected:
        raise DualDp1GenerationError("cannot summarize an incomplete shard")
    rows = list(rows_by_index.values())
    failures: Counter[str] = Counter()
    for row in rows:
        failures.update(row["preregistered_core_failures"])
    return {
        "status": "complete",
        "cases": len(rows),
        "unique_samples": len({row["sample_id"] for row in rows}),
        "unique_meetings": len({row["meeting_id"] for row in rows}),
        "input_truncation_cases": sum(bool(row["input_truncated"]) for row in rows),
        "eos_cases": sum(row["finish_reason"] == "eos" for row in rows),
        "length_cases": sum(row["finish_reason"] == "length" for row in rows),
        "core_valid_cases": sum(bool(row["preregistered_core_valid"]) for row in rows),
        "core_failure_counts": dict(sorted(failures.items())),
    }


def _persistence_contract() -> dict[str, Any]:
    return {
        "single_writer_shard_wal": True,
        "append_flush_fsync_per_completed_request": True,
        "active_20_case_shard_chunk_crash_replay": True,
        "full_generated_text": True,
        "full_answer": True,
        "full_generated_token_ids": True,
        "canonical_shard_sort": "absolute_case_index",
    }


def _classify_resume_layout(output_dir: Path) -> str:
    """Fail-closed classification of the only resumable initialization layouts."""

    unresolved = output_dir.expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise DualDp1GenerationError("resume output must be one real directory")
    output_dir = unresolved.resolve()
    entries = {path.name for path in output_dir.iterdir()}
    if not entries:
        return "empty_root_before_launch"
    if "manifest.json" in entries:
        raise DualDp1GenerationError("sealed shard cannot be resumed")
    launch = output_dir / "launch.json"
    state = output_dir / "state.progress.v1.json"
    partial = output_dir / ".partial"
    wal = partial / "generations.completion_order.progress.v1.jsonl"
    receipts = partial / "absolute_chunks.progress.v1.jsonl"
    if (
        not launch.is_file()
        or launch.is_symlink()
        or not partial.is_dir()
        or partial.is_symlink()
    ):
        raise DualDp1GenerationError(
            "resume layout lacks sealed launch/partial artifacts"
        )
    if not state.exists() and not state.is_symlink():
        if (
            entries != {launch.name, partial.name}
            or {path.name for path in partial.iterdir()} != {wal.name, receipts.name}
            or any(
                path.is_symlink() or not path.is_file() or path.stat().st_size != 0
                for path in (wal, receipts)
            )
        ):
            raise DualDp1GenerationError(
                "pre-state crash recovery requires an exact empty WAL/receipt prefix"
            )
        return "launch_and_empty_wals_before_state"
    if state.is_file() and not state.is_symlink():
        allowed_entries = {
            launch.name,
            partial.name,
            state.name,
            "generations.canonical.v1.jsonl",
        }
        canonical = output_dir / "generations.canonical.v1.jsonl"
        if (
            not entries.issubset(allowed_entries)
            or {path.name for path in partial.iterdir()} != {wal.name, receipts.name}
            or any(path.is_symlink() or not path.is_file() for path in (wal, receipts))
            or (
                (canonical.exists() or canonical.is_symlink())
                and (canonical.is_symlink() or not canonical.is_file())
            )
        ):
            raise DualDp1GenerationError(
                "stateful resume layout contains unsafe artifacts"
            )
        return "stateful_resume"
    raise DualDp1GenerationError("resume state path is unsafe")


def _initialization_recovery_resume_count(layout: str) -> int:
    if layout in {
        "empty_root_before_launch",
        "launch_and_empty_wals_before_state",
    }:
        return 1
    raise DualDp1GenerationError("layout is not an initialization recovery")


def run_shard(
    *,
    model_id: str,
    shard_id: int,
    physical_gpu_index: int,
    cohort_path: Path,
    cohort_sha256: str,
    output_dir: Path,
    resume: bool,
    smoke: bool,
    gpu_wait_timeout_seconds: int,
    gpu_poll_seconds: int,
    max_num_seqs: int,
    control_dir: Path,
    run_token: str,
    inherited_lock_fd: int,
    control_timeout_seconds: int,
    formal_authorization_path: Path | None = None,
    expected_scope: str | None = None,
) -> dict[str, Any]:
    if model_id not in MODEL_ORDER:
        raise DualDp1GenerationError(f"unsupported model ID: {model_id}")
    shard_id, physical_gpu_index = _validate_worker_mapping(
        shard_id, physical_gpu_index
    )
    max_num_seqs = _validate_max_num_seqs(max_num_seqs)
    run_token = _validate_run_token(run_token)
    scope = "infrastructure_smoke" if smoke else "formal_merged_panel"
    if expected_scope is not None and expected_scope != scope:
        raise DualDp1GenerationError("--scope and --smoke select different scopes")
    # This authorization gate deliberately precedes CUDA environment checks,
    # physical-GPU inspection, output-root mutation, and vLLM engine creation.
    formal_authorization = _formal_authorization_for_scope(
        scope=scope,
        authorization_path=formal_authorization_path,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
        max_num_seqs=max_num_seqs,
    )
    control_paths = _control_paths(control_dir, shard_id)
    if control_timeout_seconds <= 0:
        raise DualDp1GenerationError("control timeout must be positive")
    _assert_required_environment(
        physical_gpu_index=physical_gpu_index, require_cuda=True
    )
    physical_identity = _physical_gpu_identity(physical_gpu_index)
    cohort, ledger, sample_manifest, bound_rows = load_cohort(
        cohort_path, cohort_sha256
    )
    all_cases = canonical_cases(
        samples=sample_manifest["samples"], ledger=ledger, smoke=smoke
    )
    selected_cases = shard_cases(all_cases, shard_id)
    expected_cases = len(selected_cases)
    total_cases = len(all_cases)
    source_sample_sha = str(cohort["source_sample_manifest"]["sha256"])
    source_hashes = _source_hashes(
        cohort_path,
        cohort_sha256,
        cohort,
        sample_manifest,
        max_num_seqs,
        formal_authorization,
    )
    anchor = core.load_frozen_anchor(model_id)
    try:
        dp2._validate_bf16_anchor(anchor)
    except dp2.VllmK5GenerationError as exc:
        raise DualDp1GenerationError(str(exc)) from exc
    model_label = str(anchor["model_label"])

    unresolved = output_dir.expanduser()
    if unresolved.is_symlink():
        raise DualDp1GenerationError("shard output directory is a symlink")
    output_dir = unresolved.resolve()
    partial_dir = output_dir / ".partial"
    wal_path = partial_dir / "generations.completion_order.progress.v1.jsonl"
    receipt_path = partial_dir / "absolute_chunks.progress.v1.jsonl"
    canonical_path = output_dir / "generations.canonical.v1.jsonl"
    launch_path = output_dir / "launch.json"
    state_path = output_dir / "state.progress.v1.json"
    manifest_path = output_dir / "manifest.json"
    persistence = _persistence_contract()
    launch_payload = {
        "schema_version": SHARD_MANIFEST_SCHEMA,
        "status": "initializing",
        "created_at_utc": core._utc_now(),
        "evaluation_id": EVALUATION_ID,
        "task_contract_id": core.TASK_CONTRACT_ID,
        "evaluation_scope": scope,
        "model_id": model_id,
        "model_label": model_label,
        "model": anchor["model"],
        "sealed_anchor_manifest": anchor["anchor"],
        "shard_id": shard_id,
        "physical_gpu_index": physical_gpu_index,
        "cohort": {"path": str(cohort_path.resolve()), "sha256": cohort_sha256},
        "formal_authorization": copy.deepcopy(formal_authorization),
        "source_artifact_sha256s": source_hashes,
        "generation": {
            **sampling_contract(max_num_seqs=max_num_seqs),
            "expected_shard_cases": expected_cases,
            "expected_total_cases": total_cases,
        },
        "runtime": {
            **_runtime_versions(),
            "worker_pid": os.getpid(),
            "cuda_visible_devices": str(physical_gpu_index),
            "cuda_device_order": "PCI_BUS_ID",
            "physical_gpu_identity": physical_identity,
            "gpu_lock_path": str(gpu_lock_path(physical_gpu_index).resolve()),
            "max_num_seqs": max_num_seqs,
            "required_vllm_environment": {
                key: os.environ.get(key) for key in REQUIRED_VLLM_ENV
            },
            "present_vllm_dp_environment": sorted(
                key for key in os.environ if key.startswith("VLLM_DP_")
            ),
            "supervisor_control": {
                "control_dir": str(control_paths["control_dir"]),
                "run_token_sha256": _sha256_text(run_token),
                "inherited_lock_fd": inherited_lock_fd,
                "ready_path": str(control_paths["ready"]),
                "go_path": str(control_paths["go"]),
                "done_path": str(control_paths["done"]),
            },
        },
        "persistence": persistence,
    }
    compare_keys = tuple(
        key for key in launch_payload if key not in {"created_at_utc", "runtime"}
    )
    resume_count = 0
    engine_runtime: Mapping[str, Any] | None = None
    prior_launch: Mapping[str, Any] | None = None
    resume_layout = _classify_resume_layout(output_dir) if resume else None
    empty_root_recovery = resume_layout == "empty_root_before_launch"
    if empty_root_recovery:
        resume_count = _initialization_recovery_resume_count(str(resume_layout))
        partial_dir.mkdir()
        core._write_new_json(launch_path, seal_manifest(launch_payload))
        for path in (wal_path, receipt_path):
            with path.open("x", encoding="utf-8") as handle:
                handle.flush()
                os.fsync(handle.fileno())
        core._fsync_directory(partial_dir)
        rows, receipts = [], []
        wal_tracker, receipt_tracker = WalTracker.empty(), WalTracker.empty()
        _write_state(
            state_path,
            _state_payload(
                status="recovered_empty_prelaunch_root",
                model_id=model_id,
                shard_id=shard_id,
                expected_cases=expected_cases,
                total_cases=total_cases,
                resume_count=resume_count,
                wal_binding=wal_tracker.binding(wal_path),
                receipt_binding=receipt_tracker.binding(receipt_path),
                completed_indexes=set(),
                formal_authorization=formal_authorization,
                recovery={
                    "kind": "empty_output_directory_before_launch_commit",
                    "generation_rows_recovered": 0,
                },
            ),
        )
    elif resume:
        if resume_layout not in {
            "launch_and_empty_wals_before_state",
            "stateful_resume",
        }:
            raise DualDp1GenerationError("unsupported resume layout")
        prior_launch = core._read_json(launch_path)
        validate_manifest_integrity(prior_launch)
        for key in compare_keys:
            if prior_launch.get(key) != launch_payload.get(key):
                raise DualDp1GenerationError(f"resume launch drift at {key}")
        prior_runtime = prior_launch.get("runtime")
        if not isinstance(prior_runtime, Mapping):
            raise DualDp1GenerationError("resume launch runtime is missing")
        for key in (
            "vllm",
            "torch",
            "transformers",
            "tokenizers",
            "cuda_visible_devices",
            "cuda_device_order",
            "physical_gpu_identity",
            "gpu_lock_path",
            "max_num_seqs",
        ):
            if prior_runtime.get(key) != launch_payload["runtime"].get(key):
                raise DualDp1GenerationError(f"resume runtime drift at {key}")
        if resume_layout == "launch_and_empty_wals_before_state":
            rows, receipts = [], []
            wal_tracker, receipt_tracker = WalTracker.empty(), WalTracker.empty()
            resume_count = _initialization_recovery_resume_count(str(resume_layout))
            engine_runtime = None
            _write_state(
                state_path,
                _state_payload(
                    status="recovered_launch_before_initial_state",
                    model_id=model_id,
                    shard_id=shard_id,
                    expected_cases=expected_cases,
                    total_cases=total_cases,
                    resume_count=resume_count,
                    wal_binding=wal_tracker.binding(wal_path),
                    receipt_binding=receipt_tracker.binding(receipt_path),
                    completed_indexes=set(),
                    formal_authorization=formal_authorization,
                    recovery={
                        "kind": "sealed_launch_and_empty_wals_before_initial_state",
                        "generation_rows_recovered": 0,
                    },
                ),
            )
        else:
            prior_state = core._read_json(state_path)
            validate_manifest_integrity(prior_state)
            if (
                prior_state.get("schema_version") != STATE_SCHEMA
                or prior_state.get("model_id") != model_id
                or prior_state.get("shard_id") != shard_id
                or prior_state.get("expected_cases") != expected_cases
                or prior_state.get("formal_authorization") != formal_authorization
            ):
                raise DualDp1GenerationError("resume shard state drift")
            recorded = prior_state.get("completed_cases")
            if not isinstance(recorded, int) or isinstance(recorded, bool):
                raise DualDp1GenerationError("resume completed count is invalid")
            rows, wal_tracker, prefix = _read_canonical_jsonl(
                wal_path, recorded_prefix_rows=recorded
            )
            if len(rows) not in {recorded, recorded + 1}:
                raise DualDp1GenerationError("WAL/state differs by more than one row")
            observed_binding = (
                wal_tracker.binding(wal_path) if len(rows) == recorded else prefix
            )
            if observed_binding != prior_state.get("completion_wal"):
                raise DualDp1GenerationError("resume WAL prefix binding drift")
            prior_receipt_binding = prior_state.get("chunk_receipts")
            if not isinstance(prior_receipt_binding, Mapping):
                raise DualDp1GenerationError("resume receipt binding is missing")
            recorded_receipts = prior_receipt_binding.get("rows")
            if not isinstance(recorded_receipts, int) or isinstance(
                recorded_receipts, bool
            ):
                raise DualDp1GenerationError("resume receipt count is invalid")
            receipts, receipt_tracker, receipt_prefix = _read_canonical_jsonl(
                receipt_path, recorded_prefix_rows=recorded_receipts
            )
            if len(receipts) not in {recorded_receipts, recorded_receipts + 1}:
                raise DualDp1GenerationError(
                    "receipt WAL/state differs by more than one row"
                )
            observed_receipts = (
                receipt_tracker.binding(receipt_path)
                if len(receipts) == recorded_receipts
                else receipt_prefix
            )
            if observed_receipts != prior_receipt_binding:
                raise DualDp1GenerationError("resume receipt WAL prefix binding drift")
            resume_count = int(prior_state.get("resume_count", 0)) + 1
            engine_runtime = prior_state.get("engine_runtime")
    else:
        if output_dir.exists():
            raise DualDp1GenerationError(f"output already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        partial_dir.mkdir()
        core._fsync_directory(output_dir.parent)
        core._write_new_json(launch_path, seal_manifest(launch_payload))
        for path in (wal_path, receipt_path):
            with path.open("x", encoding="utf-8") as handle:
                handle.flush()
                os.fsync(handle.fileno())
        core._fsync_directory(partial_dir)
        rows, receipts = [], []
        wal_tracker, receipt_tracker = WalTracker.empty(), WalTracker.empty()
        _write_state(
            state_path,
            _state_payload(
                status="initializing",
                model_id=model_id,
                shard_id=shard_id,
                expected_cases=expected_cases,
                total_cases=total_cases,
                resume_count=0,
                wal_binding=wal_tracker.binding(wal_path),
                receipt_binding=receipt_tracker.binding(receipt_path),
                completed_indexes=set(),
                formal_authorization=formal_authorization,
            ),
        )

    rows_by_index = validate_wal_rows(
        rows,
        cases=all_cases,
        model_id=model_id,
        model_label=model_label,
        bound_rows=bound_rows,
        sample_manifest_sha256=source_sample_sha,
        source_artifact_sha256s=source_hashes,
        max_num_seqs=max_num_seqs,
        shard_id=shard_id,
        physical_gpu_index=physical_gpu_index,
    )
    _validate_receipts(
        receipts,
        model_id=model_id,
        shard_id=shard_id,
        rows_by_index=rows_by_index,
        total_cases=total_cases,
    )
    total_chunks = 1 if smoke else CHUNKS_PER_MODEL
    pending_chunks = [
        (
            chunk_id,
            [
                all_cases[index]
                for index in _chunk_indexes(chunk_id, shard_id, total_cases=total_cases)
            ],
        )
        for chunk_id in range(len(receipts), total_chunks)
    ]
    if (
        not pending_chunks
        and isinstance(engine_runtime, Mapping)
        and "done_manifest" not in engine_runtime
    ):
        # Power loss can occur after DONE was fsynced but before the final
        # engine-session state update.  DONE is permitted to be one durable
        # control record ahead, just as the row/receipt WALs are.  Rebind it
        # without regenerating any completed request.
        engine_runtime = _recover_one_record_ahead_done(
            engine_runtime=engine_runtime,
            model_id=model_id,
            shard_id=shard_id,
            physical_gpu_index=physical_gpu_index,
            evaluation_scope=scope,
            max_num_seqs=max_num_seqs,
            formal_authorization=formal_authorization,
        )

    async def on_completed(
        chunk_id: int,
        case: Mapping[str, Any],
        text: str,
        token_ids: Sequence[int],
        eos_ids: Sequence[int],
        pad_id: int,
        raw_finish_reason: Any,
        raw_stop_reason: Any,
        runtime: Mapping[str, Any],
    ) -> None:
        nonlocal engine_runtime
        engine_runtime = runtime
        index = int(case["absolute_case_index"])
        result = build_result(
            model_id=model_id,
            model_label=model_label,
            case=case,
            source_row=bound_rows[str(case["sample"]["sample_id"])],
            sample_manifest_sha256=source_sample_sha,
            source_artifact_sha256s=source_hashes,
            generated_text=text,
            generated_token_ids=token_ids,
            eos_token_ids=eos_ids,
            pad_token_id=pad_id,
            raw_finish_reason=raw_finish_reason,
            raw_stop_reason=raw_stop_reason,
            max_num_seqs=max_num_seqs,
            shard_id=shard_id,
            physical_gpu_index=physical_gpu_index,
        )
        prior = rows_by_index.get(index)
        if prior is not None:
            if prior != result:
                raise DualDp1GenerationError(f"crash replay changed case {index}")
            return
        with wal_path.open("a", encoding="utf-8") as handle:
            _append_fsync(handle, wal_tracker, result)
        rows_by_index[index] = result
        _write_state(
            state_path,
            _state_payload(
                status="generating",
                model_id=model_id,
                shard_id=shard_id,
                expected_cases=expected_cases,
                total_cases=total_cases,
                resume_count=resume_count,
                wal_binding=wal_tracker.binding(wal_path),
                receipt_binding=receipt_tracker.binding(receipt_path),
                completed_indexes=set(rows_by_index),
                formal_authorization=formal_authorization,
                engine_runtime=engine_runtime,
                last_completed_request={
                    "absolute_case_index": index,
                    "absolute_chunk_id": chunk_id,
                },
            ),
        )

    async def on_chunk_completed(chunk_id: int, runtime: Mapping[str, Any]) -> None:
        nonlocal engine_runtime
        engine_runtime = runtime
        if len(receipts) != chunk_id:
            raise DualDp1GenerationError("shard receipts are not a contiguous prefix")
        receipt = _chunk_receipt(
            model_id=model_id,
            shard_id=shard_id,
            chunk_id=chunk_id,
            rows_by_index=rows_by_index,
            total_cases=total_cases,
        )
        with receipt_path.open("a", encoding="utf-8") as handle:
            _append_fsync(handle, receipt_tracker, receipt)
        receipts.append(receipt)
        _write_state(
            state_path,
            _state_payload(
                status="chunk_complete",
                model_id=model_id,
                shard_id=shard_id,
                expected_cases=expected_cases,
                total_cases=total_cases,
                resume_count=resume_count,
                wal_binding=wal_tracker.binding(wal_path),
                receipt_binding=receipt_tracker.binding(receipt_path),
                completed_indexes=set(rows_by_index),
                formal_authorization=formal_authorization,
                engine_runtime=engine_runtime,
                last_completed_chunk=chunk_id,
            ),
        )

    if pending_chunks:
        # The external supervisor atomically acquires both canonical card
        # locks before spawning either worker and retains them until both
        # children exit.  The worker only validates its inherited descriptor.
        with inherited_gpu_lease(
            physical_gpu_index=physical_gpu_index,
            inherited_lock_fd=inherited_lock_fd,
        ) as lease:
            try:
                engine_runtime = asyncio.run(
                    _run_async_generation(
                        anchor=anchor,
                        tokenizer_path=Path(str(cohort["fomc_tokenizer"]["path"])),
                        pending_chunks=pending_chunks,
                        on_completed=on_completed,
                        on_chunk_completed=on_chunk_completed,
                        max_num_seqs=max_num_seqs,
                        physical_gpu_identity=physical_identity,
                        gpu_lease=lease,
                        control_dir=control_paths["control_dir"],
                        run_token=run_token,
                        model_id=model_id,
                        shard_id=shard_id,
                        evaluation_scope=scope,
                        formal_authorization=formal_authorization,
                        control_timeout_seconds=control_timeout_seconds,
                        resume_count=resume_count,
                    )
                )
                # Close the narrow crash window between the sealed DONE record
                # and later canonicalization.  A resume with no pending chunks
                # can now finish entirely from durable engine/barrier evidence.
                _write_state(
                    state_path,
                    _state_payload(
                        status="engine_session_complete",
                        model_id=model_id,
                        shard_id=shard_id,
                        expected_cases=expected_cases,
                        total_cases=total_cases,
                        resume_count=resume_count,
                        wal_binding=wal_tracker.binding(wal_path),
                        receipt_binding=receipt_tracker.binding(receipt_path),
                        completed_indexes=set(rows_by_index),
                        formal_authorization=formal_authorization,
                        engine_runtime=engine_runtime,
                    ),
                )
            except BaseException as exc:
                _write_state(
                    state_path,
                    _state_payload(
                        status="failed",
                        model_id=model_id,
                        shard_id=shard_id,
                        expected_cases=expected_cases,
                        total_cases=total_cases,
                        resume_count=resume_count,
                        wal_binding=wal_tracker.binding(wal_path),
                        receipt_binding=receipt_tracker.binding(receipt_path),
                        completed_indexes=set(rows_by_index),
                        formal_authorization=formal_authorization,
                        engine_runtime=engine_runtime,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        failed_at_utc=core._utc_now(),
                    ),
                )
                raise

    if len(rows_by_index) != expected_cases or not isinstance(engine_runtime, Mapping):
        raise DualDp1GenerationError("generation ended before shard completion")
    engine_runtime = copy.deepcopy(dict(engine_runtime))
    timed_tokens = engine_runtime.get("generated_output_tokens")
    canonical_tokens = sum(
        len(row["generated_token_ids"]) for row in rows_by_index.values()
    )
    seconds = engine_runtime.get("generation_wall_seconds")
    if (
        not isinstance(timed_tokens, int)
        or isinstance(timed_tokens, bool)
        or timed_tokens <= 0
        or not isinstance(seconds, (int, float))
        or isinstance(seconds, bool)
        or float(seconds) <= 0
    ):
        raise DualDp1GenerationError("shard timing evidence is incomplete")
    engine_runtime["timed_session_generated_output_tokens"] = timed_tokens
    engine_runtime["canonical_generated_output_tokens"] = canonical_tokens
    engine_runtime["output_tokens_per_second"] = timed_tokens / float(seconds)
    engine_runtime["speed_measurement_valid_for_candidate_selection"] = bool(
        resume_count == 0
        and engine_runtime.get("completed_requests_in_timed_session") == expected_cases
        and timed_tokens == canonical_tokens
    )
    with receipt_path.open("a", encoding="utf-8") as handle:
        while len(receipts) < total_chunks:
            receipt = _chunk_receipt(
                model_id=model_id,
                shard_id=shard_id,
                chunk_id=len(receipts),
                rows_by_index=rows_by_index,
                total_cases=total_cases,
            )
            _append_fsync(handle, receipt_tracker, receipt)
            receipts.append(receipt)
    _validate_receipts(
        receipts,
        model_id=model_id,
        shard_id=shard_id,
        rows_by_index=rows_by_index,
        total_cases=total_cases,
    )
    canonical_rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    _write_or_validate_canonical_jsonl(canonical_path, canonical_rows)
    summary = _summary(rows_by_index, smoke=smoke)
    _write_state(
        state_path,
        _state_payload(
            status="generation_complete",
            model_id=model_id,
            shard_id=shard_id,
            expected_cases=expected_cases,
            total_cases=total_cases,
            resume_count=resume_count,
            wal_binding=wal_tracker.binding(wal_path),
            receipt_binding=receipt_tracker.binding(receipt_path),
            completed_indexes=set(rows_by_index),
            formal_authorization=formal_authorization,
            engine_runtime=engine_runtime,
            canonical_output=_file_binding(canonical_path, rows=expected_cases),
            summary=summary,
        ),
    )
    state = core._read_json(state_path)
    state_payload_sha = validate_manifest_integrity(state)
    payload = seal_manifest(
        {
            "schema_version": SHARD_MANIFEST_SCHEMA,
            "status": "complete",
            "created_at_utc": core._utc_now(),
            "evaluation_id": EVALUATION_ID,
            "task_contract_id": core.TASK_CONTRACT_ID,
            "evaluation_scope": scope,
            "model_id": model_id,
            "model_label": model_label,
            "model": anchor["model"],
            "sealed_anchor_manifest": anchor["anchor"],
            "shard_id": shard_id,
            "physical_gpu_index": physical_gpu_index,
            "cohort": launch_payload["cohort"],
            "formal_authorization": copy.deepcopy(formal_authorization),
            "source_artifact_sha256s": source_hashes,
            "generation_contract": {
                **sampling_contract(max_num_seqs=max_num_seqs),
                "expected_shard_cases": expected_cases,
                "expected_total_cases": total_cases,
                "expected_absolute_chunks": total_chunks,
            },
            "runtime": {"resume_count": resume_count, "engine_runtime": engine_runtime},
            "persistence": persistence,
            "artifacts": {
                "launch": _file_binding(launch_path),
                "completion_order_wal": _file_binding(wal_path, rows=expected_cases),
                "absolute_chunk_receipts": _file_binding(
                    receipt_path, rows=total_chunks
                ),
                "canonical_generations": _file_binding(
                    canonical_path, rows=expected_cases
                ),
                "generation_complete_state": {
                    **_file_binding(state_path),
                    "payload_sha256": state_payload_sha,
                },
            },
            "summary": summary,
        }
    )
    core._write_new_json(manifest_path, payload)
    return payload


def _verify_arbitrary_binding(binding: Mapping[str, Any]) -> Path:
    unresolved = Path(str(binding.get("path") or "")).expanduser()
    if unresolved.is_symlink():
        raise DualDp1GenerationError("bound control artifact is a symlink")
    path = unresolved.resolve()
    if (
        not path.is_file()
        or binding.get("sha256") != core._sha256_file(path)
        or binding.get("bytes") != path.stat().st_size
    ):
        raise DualDp1GenerationError(f"bound control artifact drift: {path}")
    if "payload_sha256" in binding:
        payload = core._read_json(path)
        if validate_manifest_integrity(payload) != binding["payload_sha256"]:
            raise DualDp1GenerationError(f"control payload drift: {path}")
    return path


def load_and_validate_shard(
    manifest_path: Path,
    *,
    cohort_path: Path,
    cohort_sha256: str,
    expected_model_id: str,
    expected_shard_id: int,
    expected_scope: str,
    max_num_seqs: int,
) -> dict[str, Any]:
    shard_id = _validate_shard_id(expected_shard_id)
    max_num_seqs = _validate_max_num_seqs(max_num_seqs)
    if expected_scope not in {"infrastructure_smoke", "formal_merged_panel"}:
        raise DualDp1GenerationError("invalid shard scope")
    unresolved = manifest_path.expanduser()
    if unresolved.is_symlink():
        raise DualDp1GenerationError("shard manifest is a symlink")
    manifest_path = unresolved.resolve()
    manifest = core._read_json(manifest_path)
    payload_sha = validate_manifest_integrity(manifest)
    smoke = expected_scope == "infrastructure_smoke"
    expected_cases = _expected_shard_cases(smoke=smoke)
    total_cases = _expected_total_cases(smoke=smoke)
    if (
        manifest.get("schema_version") != SHARD_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("evaluation_id") != EVALUATION_ID
        or manifest.get("evaluation_scope") != expected_scope
        or manifest.get("model_id") != expected_model_id
        or manifest.get("shard_id") != shard_id
        or manifest.get("physical_gpu_index") != shard_id
        or manifest.get("generation_contract")
        != {
            **sampling_contract(max_num_seqs=max_num_seqs),
            "expected_shard_cases": expected_cases,
            "expected_total_cases": total_cases,
            "expected_absolute_chunks": 1 if smoke else CHUNKS_PER_MODEL,
        }
    ):
        raise DualDp1GenerationError("sealed shard identity/generation drift")
    cohort, ledger, sample_manifest, bound_rows = load_cohort(
        cohort_path, cohort_sha256
    )
    formal_authorization = _revalidate_formal_authorization(
        manifest.get("formal_authorization"),
        scope=expected_scope,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
        max_num_seqs=max_num_seqs,
    )
    all_cases = canonical_cases(
        samples=sample_manifest["samples"], ledger=ledger, smoke=smoke
    )
    source_hashes = _source_hashes(
        cohort_path,
        cohort_sha256,
        cohort,
        sample_manifest,
        max_num_seqs,
        formal_authorization,
    )
    anchor = core.load_frozen_anchor(expected_model_id)
    try:
        dp2._validate_bf16_anchor(anchor)
    except dp2.VllmK5GenerationError as exc:
        raise DualDp1GenerationError(str(exc)) from exc
    if (
        manifest.get("model") != anchor["model"]
        or manifest.get("sealed_anchor_manifest") != anchor["anchor"]
        or manifest.get("source_artifact_sha256s") != source_hashes
        or manifest.get("persistence") != _persistence_contract()
        or manifest.get("cohort")
        != {"path": str(cohort_path.resolve()), "sha256": cohort_sha256}
    ):
        raise DualDp1GenerationError("sealed shard model/source drift")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise DualDp1GenerationError("sealed shard artifacts are missing")
    launch_path = _verify_file_binding(artifacts["launch"], parent=manifest_path.parent)
    wal_path = _verify_file_binding(
        artifacts["completion_order_wal"], parent=manifest_path.parent / ".partial"
    )
    receipt_path = _verify_file_binding(
        artifacts["absolute_chunk_receipts"], parent=manifest_path.parent / ".partial"
    )
    canonical_path = _verify_file_binding(
        artifacts["canonical_generations"], parent=manifest_path.parent
    )
    state_path = _verify_file_binding(
        artifacts["generation_complete_state"], parent=manifest_path.parent
    )
    launch = core._read_json(launch_path)
    validate_manifest_integrity(launch)
    if (
        launch.get("schema_version") != SHARD_MANIFEST_SCHEMA
        or launch.get("evaluation_id") != EVALUATION_ID
        or launch.get("model_id") != expected_model_id
        or launch.get("shard_id") != shard_id
        or launch.get("physical_gpu_index") != shard_id
        or launch.get("formal_authorization") != formal_authorization
        or launch.get("source_artifact_sha256s") != source_hashes
        or launch.get("persistence") != _persistence_contract()
        or launch.get("generation")
        != {
            **sampling_contract(max_num_seqs=max_num_seqs),
            "expected_shard_cases": expected_cases,
            "expected_total_cases": total_cases,
        }
    ):
        raise DualDp1GenerationError("sealed shard launch drift")
    launch_runtime = launch.get("runtime")
    if (
        not isinstance(launch_runtime, Mapping)
        or launch_runtime.get("cuda_visible_devices") != str(shard_id)
        or launch_runtime.get("cuda_device_order") != "PCI_BUS_ID"
        or launch_runtime.get("physical_gpu_identity")
        != {"physical_gpu_index": shard_id, **EXPECTED_GPU_IDENTITIES[shard_id]}
        or launch_runtime.get("gpu_lock_path") != str(gpu_lock_path(shard_id).resolve())
        or launch_runtime.get("max_num_seqs") != max_num_seqs
        or launch_runtime.get("required_vllm_environment") != REQUIRED_VLLM_ENV
        or launch_runtime.get("present_vllm_dp_environment") != []
    ):
        raise DualDp1GenerationError("sealed shard launch GPU runtime drift")
    wal_rows, wal_tracker, _ = _read_canonical_jsonl(wal_path)
    rows_by_index = validate_wal_rows(
        wal_rows,
        cases=all_cases,
        model_id=expected_model_id,
        model_label=str(anchor["model_label"]),
        bound_rows=bound_rows,
        sample_manifest_sha256=str(cohort["source_sample_manifest"]["sha256"]),
        source_artifact_sha256s=source_hashes,
        max_num_seqs=max_num_seqs,
        shard_id=shard_id,
        physical_gpu_index=shard_id,
    )
    expected_indexes = {
        index for index in range(total_cases) if assigned_shard(index) == shard_id
    }
    if set(rows_by_index) != expected_indexes:
        raise DualDp1GenerationError("sealed shard coverage is not exact")
    canonical_rows, _, _ = _read_canonical_jsonl(canonical_path)
    if canonical_rows != [rows_by_index[index] for index in sorted(expected_indexes)]:
        raise DualDp1GenerationError("sealed shard canonical output drift")
    receipts, receipt_tracker, _ = _read_canonical_jsonl(receipt_path)
    _validate_receipts(
        receipts,
        model_id=expected_model_id,
        shard_id=shard_id,
        rows_by_index=rows_by_index,
        total_cases=total_cases,
    )
    if len(receipts) != (1 if smoke else CHUNKS_PER_MODEL):
        raise DualDp1GenerationError("sealed shard receipt coverage drift")
    state = core._read_json(state_path)
    state_payload_sha = validate_manifest_integrity(state)
    runtime = manifest.get("runtime")
    engine_runtime = (
        runtime.get("engine_runtime") if isinstance(runtime, Mapping) else None
    )
    resume_count = runtime.get("resume_count") if isinstance(runtime, Mapping) else None
    if (
        not isinstance(resume_count, int)
        or isinstance(resume_count, bool)
        or resume_count < 0
        or not isinstance(engine_runtime, Mapping)
    ):
        raise DualDp1GenerationError("sealed shard runtime is missing")
    identity = engine_runtime.get("physical_gpu_identity")
    binding = engine_runtime.get("engine_core_gpu_binding")
    lease = engine_runtime.get("gpu_lease")
    seconds = engine_runtime.get("generation_wall_seconds")
    tokens = engine_runtime.get("canonical_generated_output_tokens")
    timed_tokens = engine_runtime.get("timed_session_generated_output_tokens")
    timed_requests = engine_runtime.get("completed_requests_in_timed_session")
    engine_session_resume_count = engine_runtime.get("engine_session_resume_count")
    started_ns = engine_runtime.get("generation_started_epoch_ns")
    finished_ns = engine_runtime.get("generation_finished_epoch_ns")
    started_mono = engine_runtime.get("generation_started_monotonic_ns")
    finished_mono = engine_runtime.get("generation_finished_monotonic_ns")
    if (
        not isinstance(identity, Mapping)
        or identity.get("physical_gpu_index") != shard_id
        or identity
        != {"physical_gpu_index": shard_id, **EXPECTED_GPU_IDENTITIES[shard_id]}
        or not isinstance(binding, Mapping)
        or binding.get("physical_gpu_index") != shard_id
        or not isinstance(binding.get("worker_pid"), int)
        or not isinstance(binding.get("engine_core_pid"), int)
        or binding.get("worker_pid") == binding.get("engine_core_pid")
        or not isinstance(binding.get("descendant_compute_processes"), list)
        or {
            process.get("pid")
            for process in binding.get("descendant_compute_processes", [])
            if isinstance(process, Mapping)
        }
        != {binding.get("engine_core_pid")}
        or not isinstance(lease, Mapping)
        or lease.get("lease_mode") != "supervisor_preacquired_inherited_fd"
        or lease.get("physical_gpu_index") != shard_id
        or engine_runtime.get("enforce_eager") is not False
        or engine_runtime.get("cuda_graphs") is not True
        or engine_runtime.get("async_output_processing") is not True
        or engine_runtime.get("engine") != sampling_contract(max_num_seqs=max_num_seqs)
        or not isinstance(seconds, (int, float))
        or isinstance(seconds, bool)
        or float(seconds) <= 0
        or not isinstance(tokens, int)
        or isinstance(tokens, bool)
        or tokens <= 0
        or not isinstance(timed_tokens, int)
        or isinstance(timed_tokens, bool)
        or timed_tokens <= 0
        or not isinstance(timed_requests, int)
        or isinstance(timed_requests, bool)
        or timed_requests <= 0
        or timed_requests > expected_cases
        or not isinstance(engine_session_resume_count, int)
        or isinstance(engine_session_resume_count, bool)
        or engine_session_resume_count < 0
        or (engine_session_resume_count == 0 and timed_requests != expected_cases)
        or not isinstance(started_ns, int)
        or isinstance(started_ns, bool)
        or not isinstance(finished_ns, int)
        or isinstance(finished_ns, bool)
        or finished_ns <= started_ns
        or not isinstance(started_mono, int)
        or isinstance(started_mono, bool)
        or not isinstance(finished_mono, int)
        or isinstance(finished_mono, bool)
        or finished_mono <= started_mono
    ):
        raise DualDp1GenerationError("sealed shard GPU/timing runtime drift")
    expected_tokens = sum(
        len(row["generated_token_ids"]) for row in rows_by_index.values()
    )
    expected_speed_valid = bool(resume_count == 0 and timed_tokens == expected_tokens)
    if (
        tokens != expected_tokens
        or engine_runtime.get("speed_measurement_valid_for_candidate_selection")
        is not expected_speed_valid
        or not math.isclose(
            float(engine_runtime.get("output_tokens_per_second")),
            timed_tokens / float(seconds),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise DualDp1GenerationError("sealed shard token/timing accounting drift")
    ready_path = _verify_arbitrary_binding(engine_runtime["ready_manifest"])
    go_path = _verify_arbitrary_binding(engine_runtime["go_manifest"])
    done_path = _verify_arbitrary_binding(engine_runtime["done_manifest"])
    control_attempt = engine_runtime.get("control_attempt")
    if (
        not isinstance(control_attempt, Mapping)
        or ready_path.parent != go_path.parent
        or ready_path.parent != done_path.parent
        or control_attempt
        != {
            "control_dir": str(ready_path.parent),
            "run_token_sha256": control_attempt.get("run_token_sha256"),
            "ready_path": str(ready_path),
            "go_path": str(go_path),
            "done_path": str(done_path),
        }
        or ready_path.name != f"shard{shard_id}.ready.json"
        or go_path.name != "go.json"
        or done_path.name != f"shard{shard_id}.done.json"
    ):
        raise DualDp1GenerationError("sealed shard current control paths drift")
    ready, go, done = map(core._read_json, (ready_path, go_path, done_path))
    for payload in (ready, go, done):
        validate_manifest_integrity(payload)
    if (
        ready.get("schema_version") != READY_SCHEMA
        or ready.get("status") != "ready"
        or ready.get("evaluation_id") != EVALUATION_ID
        or ready.get("model_id") != expected_model_id
        or ready.get("shard_id") != shard_id
        or ready.get("physical_gpu_index") != shard_id
        or ready.get("evaluation_scope") != expected_scope
        or ready.get("max_num_seqs") != max_num_seqs
        or ready.get("formal_authorization") != formal_authorization
        or ready.get("cuda_visible_devices") != str(shard_id)
        or ready.get("required_vllm_environment") != REQUIRED_VLLM_ENV
        or ready.get("present_vllm_dp_environment") != []
        or ready.get("worker_pid") != binding.get("worker_pid")
        or ready.get("engine_core_pid") != binding.get("engine_core_pid")
        or ready.get("gpu_uuid") != identity.get("uuid")
        or str(ready.get("gpu_pci_bus_id", "")).lower()
        != EXPECTED_GPU_IDENTITIES[shard_id]["pci_bus_id"]
        or go.get("schema_version") != GO_SCHEMA
        or go.get("status") != "released"
        or go.get("evaluation_id") != EVALUATION_ID
        or go.get("model_id") != expected_model_id
        or go.get("evaluation_scope") != expected_scope
        or go.get("max_num_seqs") != max_num_seqs
        or go.get("formal_authorization") != formal_authorization
        or go.get("physical_gpu_indexes") != [0, 1]
        or not isinstance(go.get("ready_manifests"), Mapping)
        or go.get("ready_manifests", {}).get(f"shard{shard_id}")
        != engine_runtime.get("ready_manifest")
        or done.get("schema_version") != DONE_SCHEMA
        or done.get("status") != "done"
        or done.get("evaluation_id") != EVALUATION_ID
        or done.get("model_id") != expected_model_id
        or done.get("shard_id") != shard_id
        or done.get("physical_gpu_index") != shard_id
        or done.get("evaluation_scope") != expected_scope
        or done.get("max_num_seqs") != max_num_seqs
        or done.get("formal_authorization") != formal_authorization
        or done.get("generation_started_epoch_ns") != started_ns
        or done.get("generation_finished_epoch_ns") != finished_ns
        or done.get("observed_go_epoch_ns")
        != engine_runtime.get("observed_go_epoch_ns")
        or done.get("observed_go_monotonic_ns")
        != engine_runtime.get("observed_go_monotonic_ns")
        or done.get("generation_started_monotonic_ns") != started_mono
        or done.get("generation_finished_monotonic_ns") != finished_mono
        or done.get("generated_output_tokens") != timed_tokens
        or done.get("completed_requests")
        != engine_runtime.get("completed_requests_in_timed_session")
        or done.get("resume_count") != engine_session_resume_count
        or ready.get("run_token") != go.get("run_token")
        or ready.get("run_token") != done.get("run_token")
        or control_attempt.get("run_token_sha256")
        != _sha256_text(str(ready.get("run_token")))
    ):
        raise DualDp1GenerationError("sealed shard barrier evidence drift")
    if (
        state.get("schema_version") != STATE_SCHEMA
        or state.get("status") != "generation_complete"
        or state.get("evaluation_id") != EVALUATION_ID
        or state.get("model_id") != expected_model_id
        or state.get("shard_id") != shard_id
        or state.get("formal_authorization") != formal_authorization
        or state.get("expected_cases") != expected_cases
        or state.get("completed_cases") != expected_cases
        or state.get("completed_absolute_chunks") != (1 if smoke else CHUNKS_PER_MODEL)
        or state.get("wal_complete_absolute_chunks")
        != (1 if smoke else CHUNKS_PER_MODEL)
        or state.get("committed_receipted_chunks") != (1 if smoke else CHUNKS_PER_MODEL)
        or state.get("active_chunk_completed_indexes") != []
        or state.get("completion_wal") != wal_tracker.binding(wal_path)
        or state.get("chunk_receipts") != receipt_tracker.binding(receipt_path)
        or state.get("engine_runtime") != engine_runtime
        or state.get("resume_count") != resume_count
        or state.get("canonical_output")
        != _file_binding(canonical_path, rows=expected_cases)
        or artifacts["generation_complete_state"].get("payload_sha256")
        != state_payload_sha
        or state.get("summary") != _summary(rows_by_index, smoke=smoke)
        or manifest.get("summary") != state.get("summary")
    ):
        raise DualDp1GenerationError("sealed shard state/summary drift")
    return {
        "manifest": manifest,
        "manifest_binding": {
            **_file_binding(manifest_path),
            "payload_sha256": payload_sha,
        },
        "results": [rows_by_index[index] for index in sorted(expected_indexes)],
        "engine_runtime": dict(engine_runtime),
        "formal_authorization": copy.deepcopy(formal_authorization),
    }


def _validate_orchestrator_post_exit_gpu_evidence(payload: Mapping[str, Any]) -> None:
    evidence = payload.get("gpu_idle_after_worker_exit")
    reconstructed = payload.get("timing_reconstruction") is not None
    if reconstructed:
        if (
            evidence is not None
            or payload.get("gpu_idle_before_spawn") is not None
            or payload.get("gpu_idle_after_worker_exit_claimed") is not False
            or payload.get("gpu_idle_after_worker_exit_status")
            != "unavailable_reconstructed_without_new_gpu_work"
        ):
            raise DualDp1GenerationError(
                "reconstructed timing falsely claims a post-exit GPU audit"
            )
        return
    if (
        payload.get("gpu_idle_before_spawn") is None
        or payload.get("gpu_idle_after_worker_exit_claimed") is not True
        or payload.get("gpu_idle_after_worker_exit_status")
        != "passed_while_canonical_gpu_locks_held"
        or not isinstance(evidence, Mapping)
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
        or evidence.get("timeout_seconds") != 60
        or evidence.get("poll_seconds") != 5
        or not isinstance(evidence.get("samples"), list)
        or len(evidence["samples"]) != 2
    ):
        raise DualDp1GenerationError("post-exit GPU idle evidence is incomplete")
    for sample in evidence["samples"]:
        if (
            not isinstance(sample, Mapping)
            or set(sample) != {"sampled_at_utc", "compute_processes", "gpus"}
            or not isinstance(sample.get("sampled_at_utc"), str)
            or sample.get("compute_processes") != []
            or not isinstance(sample.get("gpus"), list)
            or len(sample["gpus"]) != 2
        ):
            raise DualDp1GenerationError("post-exit GPU idle sample drift")
        rows = sample["gpus"]
        identities = {
            row.get("physical_gpu_index"): {
                "uuid": row.get("uuid"),
                "pci_bus_id": str(row.get("pci_bus_id", "")).lower(),
            }
            for row in rows
            if isinstance(row, Mapping)
        }
        if identities != EXPECTED_GPU_IDENTITIES or any(
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
            raise DualDp1GenerationError(
                "post-exit GPU identity/idleness evidence drift"
            )


def _load_orchestrator_timing(
    path: Path,
    *,
    model_id: str,
    scope: str,
    max_num_seqs: int,
    expected_tokens: int,
    formal_authorization: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise DualDp1GenerationError("orchestrator timing is a symlink")
    path = unresolved.resolve()
    payload = core._read_json(path)
    payload_sha = validate_manifest_integrity(payload)
    required = {
        "status": "complete",
        "evaluation_id": EVALUATION_ID,
        "model_id": model_id,
        "evaluation_scope": scope,
        "max_num_seqs_per_engine": max_num_seqs,
    }
    if payload.get("schema_version") != ORCHESTRATOR_TIMING_SCHEMA:
        raise DualDp1GenerationError("orchestrator timing schema drift")
    if payload.get("formal_authorization") != formal_authorization:
        raise DualDp1GenerationError("orchestrator formal authorization drift")
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise DualDp1GenerationError(f"orchestrator timing drift at {key}")
    _validate_orchestrator_post_exit_gpu_evidence(payload)
    _validate_run_token(str(payload.get("run_token") or ""))
    topology = payload.get("parallel_topology")
    control = payload.get("control_artifacts")
    timing = payload.get("shared_timing")
    if (
        topology
        != {
            "independent_engine_count": 2,
            "data_parallel_size_per_engine": 1,
            "tensor_parallel_size_per_engine": 1,
            "physical_gpu_indices": [0, 1],
            "cross_gpu_collectives": False,
            "shared_go_barrier": True,
        }
        or not isinstance(control, Mapping)
        or not isinstance(timing, Mapping)
    ):
        raise DualDp1GenerationError("orchestrator topology/control timing drift")
    eligible = timing.get("speed_measurement_valid_for_candidate_selection")
    recovery = payload.get("recovery")
    expected_control_keys = {
        "go",
        "ready",
        "done",
        "worker_console_logs",
    }
    if eligible is False:
        expected_control_keys.add("shard_manifests")
    if set(control) != expected_control_keys:
        raise DualDp1GenerationError("orchestrator control artifact keys drift")
    wall_key = (
        "parallel_generation_wall_seconds"
        if eligible is True
        else "recovery_attempt_wall_seconds"
    )
    wall = timing.get(wall_key)
    tokens = timing.get("aggregate_generated_output_tokens")
    throughput = timing.get("aggregate_output_tokens_per_second")
    if (
        not isinstance(wall, (int, float))
        or isinstance(wall, bool)
        or float(wall) <= 0
        or tokens != expected_tokens
        or not isinstance(eligible, bool)
        or (scope == "infrastructure_smoke" and eligible is not True)
    ):
        raise DualDp1GenerationError("orchestrator parallel timing accounting drift")
    if eligible:
        if (
            not isinstance(throughput, (int, float))
            or isinstance(throughput, bool)
            or not math.isclose(
                float(throughput),
                expected_tokens / float(wall),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or recovery is not None
        ):
            raise DualDp1GenerationError("fresh orchestrator throughput drift")
    elif (
        scope != "formal_merged_panel"
        or throughput is not None
        or not isinstance(recovery, Mapping)
        or recovery.get("mode") not in {"paired_resume", "asymmetric_resume"}
        or recovery.get("selection_eligible") is not False
        or not isinstance(recovery.get("launched_shards"), list)
        or not recovery["launched_shards"]
        or not set(recovery["launched_shards"]).issubset({0, 1})
        or not isinstance(recovery.get("preexisting_complete_shards"), list)
        or set(recovery["preexisting_complete_shards"])
        & set(recovery["launched_shards"])
        or set(recovery["preexisting_complete_shards"])
        | set(recovery["launched_shards"])
        != {0, 1}
        or not isinstance(timing.get("attempt_generated_output_tokens"), int)
        or timing["attempt_generated_output_tokens"] <= 0
        or timing["attempt_generated_output_tokens"] > expected_tokens
    ):
        raise DualDp1GenerationError("formal recovery timing contract drift")
    return payload, {**_file_binding(path), "payload_sha256": payload_sha}


def merge_run(
    *,
    cohort_path: Path,
    cohort_sha256: str,
    output_dir: Path,
    model_id: str,
    scope: str,
    max_num_seqs: int,
    orchestrator_timing_path: Path,
) -> dict[str, Any]:
    if model_id not in MODEL_ORDER:
        raise DualDp1GenerationError(f"unsupported model ID: {model_id}")
    max_num_seqs = _validate_max_num_seqs(max_num_seqs)
    unresolved_output = output_dir.expanduser()
    if unresolved_output.is_symlink() or not unresolved_output.is_dir():
        raise DualDp1GenerationError("merged output directory is missing or a symlink")
    output_dir = unresolved_output.resolve()
    manifest_path = output_dir / "manifest.json"
    canonical_path = output_dir / "generations.canonical.v1.jsonl"
    if manifest_path.exists() or manifest_path.is_symlink():
        raise DualDp1GenerationError("merged run is already sealed")
    shards = {
        shard_id: load_and_validate_shard(
            output_dir / f"shard{shard_id}" / "manifest.json",
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            expected_model_id=model_id,
            expected_shard_id=shard_id,
            expected_scope=scope,
            max_num_seqs=max_num_seqs,
        )
        for shard_id in ALLOWED_SHARD_IDS
    }
    remediation_binding = _remediation_receipt_binding()
    if any(
        shard["manifest"]["source_artifact_sha256s"].get(
            "vllm_k5_dp2_to_dual_dp1_remediation_receipt"
        )
        != remediation_binding
        for shard in shards.values()
    ):
        raise DualDp1GenerationError("shard remediation receipt bindings disagree")
    formal_authorization = shards[0]["formal_authorization"]
    if any(
        shard["formal_authorization"] != formal_authorization
        for shard in shards.values()
    ):
        raise DualDp1GenerationError("shard formal authorization bindings disagree")
    expected_cases = (
        SMOKE_CASES_PER_MODEL if scope == "infrastructure_smoke" else CASES_PER_MODEL
    )
    by_index = _exact_shard_union(
        {s: shards[s]["results"] for s in ALLOWED_SHARD_IDS},
        total_cases=expected_cases,
    )
    rows = [by_index[index] for index in range(expected_cases)]
    run_tokens = {
        core._read_json(
            _verify_arbitrary_binding(shards[s]["engine_runtime"]["ready_manifest"])
        )["run_token"]
        for s in ALLOWED_SHARD_IDS
    }
    pids = [
        shards[s]["engine_runtime"]["engine_core_gpu_binding"]["engine_core_pid"]
        for s in ALLOWED_SHARD_IDS
    ]
    uuids = [
        shards[s]["engine_runtime"]["physical_gpu_identity"]["uuid"]
        for s in ALLOWED_SHARD_IDS
    ]
    if len(set(pids)) != 2 or len(set(uuids)) != 2:
        raise DualDp1GenerationError("shards do not bind two distinct engines/GPUs")
    total_tokens = sum(len(row["generated_token_ids"]) for row in rows)
    timing, timing_binding = _load_orchestrator_timing(
        orchestrator_timing_path,
        model_id=model_id,
        scope=scope,
        max_num_seqs=max_num_seqs,
        expected_tokens=total_tokens,
        formal_authorization=formal_authorization,
    )
    timing_control = timing["control_artifacts"]
    expected_ready = {
        f"shard{s}": shards[s]["engine_runtime"]["ready_manifest"]
        for s in ALLOWED_SHARD_IDS
    }
    expected_done = {
        f"shard{s}": shards[s]["engine_runtime"]["done_manifest"]
        for s in ALLOWED_SHARD_IDS
    }
    timing_eligible = timing["shared_timing"][
        "speed_measurement_valid_for_candidate_selection"
    ]
    launched_shards = (
        list(ALLOWED_SHARD_IDS)
        if timing_eligible
        else list(timing["recovery"]["launched_shards"])
    )
    go_bindings = {
        _canonical(shards[s]["engine_runtime"]["go_manifest"]) for s in launched_shards
    }
    if (
        timing_control.get("ready") != expected_ready
        or timing_control.get("done") != expected_done
        or len(go_bindings) != 1
        or timing_control.get("go")
        != shards[launched_shards[0]]["engine_runtime"]["go_manifest"]
        or (timing_eligible and len(run_tokens) != 1)
        or (timing_eligible and "shard_manifests" in timing_control)
        or (
            not timing_eligible
            and timing_control.get("shard_manifests")
            != {f"shard{s}": shards[s]["manifest_binding"] for s in ALLOWED_SHARD_IDS}
        )
        or timing["shared_timing"].get("both_workers_resume_count")
        != [shards[s]["manifest"]["runtime"]["resume_count"] for s in ALLOWED_SHARD_IDS]
    ):
        raise DualDp1GenerationError("orchestrator timing does not bind shard barriers")
    _write_or_validate_canonical_jsonl(canonical_path, rows)
    anchor = core.load_frozen_anchor(model_id)
    payload = seal_manifest(
        {
            "schema_version": MERGED_MANIFEST_SCHEMA,
            "status": "complete",
            "created_at_utc": core._utc_now(),
            "evaluation_id": EVALUATION_ID,
            "task_contract_id": core.TASK_CONTRACT_ID,
            "evaluation_scope": scope,
            "model_id": model_id,
            "model_label": anchor["model_label"],
            "model": anchor["model"],
            "sealed_anchor_manifest": anchor["anchor"],
            "cohort": {"path": str(cohort_path.resolve()), "sha256": cohort_sha256},
            "remediation_receipt": remediation_binding,
            "formal_authorization": copy.deepcopy(formal_authorization),
            "generation_contract": {
                **sampling_contract(max_num_seqs=max_num_seqs),
                "expected_cases": expected_cases,
            },
            "coverage": {
                "cases": expected_cases,
                "shard_cases": {
                    f"shard{s}": len(shards[s]["results"]) for s in ALLOWED_SHARD_IDS
                },
                "absolute_indexes_exact": True,
                "overlaps": 0,
                "input_truncation_cases": sum(
                    bool(row["input_truncated"]) for row in rows
                ),
            },
            "runtime": {
                "shared_run_token_sha256": (
                    _sha256_text(next(iter(run_tokens)))
                    if len(run_tokens) == 1
                    else None
                ),
                "run_token_sha256_by_shard": {
                    f"shard{s}": _sha256_text(
                        core._read_json(
                            _verify_arbitrary_binding(
                                shards[s]["engine_runtime"]["ready_manifest"]
                            )
                        )["run_token"]
                    )
                    for s in ALLOWED_SHARD_IDS
                },
                "worker_pids": [
                    shards[s]["engine_runtime"]["worker_pid"] for s in ALLOWED_SHARD_IDS
                ],
                "engine_core_pids": pids,
                "physical_gpu_uuids": uuids,
                "resume_counts": {
                    f"shard{s}": shards[s]["manifest"]["runtime"]["resume_count"]
                    for s in ALLOWED_SHARD_IDS
                },
                "aggregate_generated_output_tokens": total_tokens,
                "parallel_generation_wall_seconds": timing["shared_timing"].get(
                    "parallel_generation_wall_seconds"
                ),
                "recovery_attempt_wall_seconds": timing["shared_timing"].get(
                    "recovery_attempt_wall_seconds"
                ),
                "aggregate_output_tokens_per_second": timing["shared_timing"][
                    "aggregate_output_tokens_per_second"
                ],
                "speed_measurement_valid_for_candidate_selection": timing[
                    "shared_timing"
                ]["speed_measurement_valid_for_candidate_selection"],
                "orchestrator_timing": timing_binding,
            },
            "shard_manifests": {
                f"shard{s}": shards[s]["manifest_binding"] for s in ALLOWED_SHARD_IDS
            },
            "artifacts": {
                "canonical_generations": _file_binding(
                    canonical_path, rows=expected_cases
                )
            },
        }
    )
    core._write_new_json(manifest_path, payload)
    return payload


def load_and_validate_run(
    manifest_path: Path,
    *,
    cohort_path: Path,
    cohort_sha256: str,
    expected_model_id: str,
    expected_scope: str,
    max_num_seqs: int,
) -> dict[str, Any]:
    max_num_seqs = _validate_max_num_seqs(max_num_seqs)
    unresolved = manifest_path.expanduser()
    if unresolved.is_symlink():
        raise DualDp1GenerationError("merged manifest is a symlink")
    manifest_path = unresolved.resolve()
    manifest = core._read_json(manifest_path)
    payload_sha = validate_manifest_integrity(manifest)
    expected_cases = (
        SMOKE_CASES_PER_MODEL
        if expected_scope == "infrastructure_smoke"
        else CASES_PER_MODEL
    )
    if (
        manifest.get("schema_version") != MERGED_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("evaluation_id") != EVALUATION_ID
        or manifest.get("model_id") != expected_model_id
        or manifest.get("evaluation_scope") != expected_scope
        or manifest.get("generation_contract")
        != {
            **sampling_contract(max_num_seqs=max_num_seqs),
            "expected_cases": expected_cases,
        }
    ):
        raise DualDp1GenerationError("merged manifest identity/generation drift")
    remediation_binding = _remediation_receipt_binding()
    if manifest.get("remediation_receipt") != remediation_binding:
        raise DualDp1GenerationError("merged remediation receipt binding drift")
    shards = {
        s: load_and_validate_shard(
            manifest_path.parent / f"shard{s}" / "manifest.json",
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            expected_model_id=expected_model_id,
            expected_shard_id=s,
            expected_scope=expected_scope,
            max_num_seqs=max_num_seqs,
        )
        for s in ALLOWED_SHARD_IDS
    }
    if manifest.get("shard_manifests") != {
        f"shard{s}": shards[s]["manifest_binding"] for s in ALLOWED_SHARD_IDS
    }:
        raise DualDp1GenerationError("merged child manifest bindings drift")
    if any(
        shard["manifest"]["source_artifact_sha256s"].get(
            "vllm_k5_dp2_to_dual_dp1_remediation_receipt"
        )
        != remediation_binding
        for shard in shards.values()
    ):
        raise DualDp1GenerationError("merged child remediation receipt drift")
    formal_authorization = shards[0]["formal_authorization"]
    if manifest.get("formal_authorization") != formal_authorization or any(
        shard["formal_authorization"] != formal_authorization
        for shard in shards.values()
    ):
        raise DualDp1GenerationError("merged formal authorization binding drift")
    rows_by_index = _exact_shard_union(
        {s: shards[s]["results"] for s in ALLOWED_SHARD_IDS},
        total_cases=expected_cases,
    )
    canonical_binding = manifest.get("artifacts", {}).get("canonical_generations")
    canonical_path = _verify_file_binding(
        canonical_binding, parent=manifest_path.parent
    )
    canonical_rows, _, _ = _read_canonical_jsonl(canonical_path)
    results = [rows_by_index[index] for index in range(expected_cases)]
    if canonical_rows != results:
        raise DualDp1GenerationError("merged canonical rows drift")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise DualDp1GenerationError("merged runtime is missing")
    total_tokens = sum(len(row["generated_token_ids"]) for row in results)
    timing_path = _verify_arbitrary_binding(runtime["orchestrator_timing"])
    timing, timing_binding = _load_orchestrator_timing(
        timing_path,
        model_id=expected_model_id,
        scope=expected_scope,
        max_num_seqs=max_num_seqs,
        expected_tokens=total_tokens,
        formal_authorization=formal_authorization,
    )
    timing_control = timing["control_artifacts"]
    timing_eligible = timing["shared_timing"][
        "speed_measurement_valid_for_candidate_selection"
    ]
    launched_shards = (
        list(ALLOWED_SHARD_IDS)
        if timing_eligible
        else list(timing["recovery"]["launched_shards"])
    )
    if (
        timing_control.get("ready")
        != {
            f"shard{s}": shards[s]["engine_runtime"]["ready_manifest"]
            for s in ALLOWED_SHARD_IDS
        }
        or timing_control.get("done")
        != {
            f"shard{s}": shards[s]["engine_runtime"]["done_manifest"]
            for s in ALLOWED_SHARD_IDS
        }
        or timing_control.get("go")
        != shards[launched_shards[0]]["engine_runtime"]["go_manifest"]
        or any(
            shards[s]["engine_runtime"]["go_manifest"]
            != shards[launched_shards[0]]["engine_runtime"]["go_manifest"]
            for s in launched_shards
        )
        or (
            not timing_eligible
            and timing_control.get("shard_manifests")
            != {f"shard{s}": shards[s]["manifest_binding"] for s in ALLOWED_SHARD_IDS}
        )
        or timing["shared_timing"].get("both_workers_resume_count")
        != [shards[s]["manifest"]["runtime"]["resume_count"] for s in ALLOWED_SHARD_IDS]
    ):
        raise DualDp1GenerationError("merged orchestrator barrier bindings drift")
    pids = [
        shards[s]["engine_runtime"]["engine_core_gpu_binding"]["engine_core_pid"]
        for s in ALLOWED_SHARD_IDS
    ]
    uuids = [
        shards[s]["engine_runtime"]["physical_gpu_identity"]["uuid"]
        for s in ALLOWED_SHARD_IDS
    ]
    expected_coverage = {
        "cases": expected_cases,
        "shard_cases": {
            f"shard{s}": len(shards[s]["results"]) for s in ALLOWED_SHARD_IDS
        },
        "absolute_indexes_exact": True,
        "overlaps": 0,
        "input_truncation_cases": sum(bool(row["input_truncated"]) for row in results),
    }
    if (
        manifest.get("coverage") != expected_coverage
        or runtime.get("engine_core_pids") != pids
        or runtime.get("physical_gpu_uuids") != uuids
        or len(set(pids)) != 2
        or len(set(uuids)) != 2
        or runtime.get("aggregate_generated_output_tokens") != total_tokens
        or runtime.get("parallel_generation_wall_seconds")
        != timing["shared_timing"].get("parallel_generation_wall_seconds")
        or runtime.get("recovery_attempt_wall_seconds")
        != timing["shared_timing"].get("recovery_attempt_wall_seconds")
        or runtime.get("aggregate_output_tokens_per_second")
        != timing["shared_timing"]["aggregate_output_tokens_per_second"]
        or runtime.get("speed_measurement_valid_for_candidate_selection")
        != timing["shared_timing"]["speed_measurement_valid_for_candidate_selection"]
        or runtime.get("orchestrator_timing") != timing_binding
    ):
        raise DualDp1GenerationError("merged coverage/runtime drift")
    return {
        "manifest": manifest,
        "manifest_binding": {
            **_file_binding(manifest_path),
            "payload_sha256": payload_sha,
        },
        "results": results,
        "shards": shards,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run-shard")
    run.add_argument("--model-id", choices=MODEL_ORDER, required=True)
    run.add_argument("--shard-id", choices=ALLOWED_SHARD_IDS, type=int, required=True)
    run.add_argument(
        "--physical-gpu-index",
        choices=ALLOWED_PHYSICAL_GPU_INDEXES,
        type=int,
        required=True,
    )
    run.add_argument("--cohort", type=Path, required=True)
    run.add_argument("--cohort-sha256", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--smoke", action="store_true")
    run.add_argument(
        "--scope",
        choices=("infrastructure_smoke", "formal_merged_panel"),
        required=True,
    )
    run.add_argument("--gpu-wait-timeout-seconds", type=int, default=172800)
    run.add_argument("--gpu-poll-seconds", type=int, default=30)
    run.add_argument(
        "--max-num-seqs", choices=ALLOWED_MAX_NUM_SEQS, type=int, required=True
    )
    run.add_argument("--control-dir", type=Path, required=True)
    run.add_argument("--run-token", required=True)
    run.add_argument("--inherited-lock-fd", type=int, required=True)
    run.add_argument("--control-timeout-seconds", type=int, default=172800)
    run.add_argument("--formal-authorization", type=Path)

    validate_shard_parser = sub.add_parser("validate-shard")
    validate_shard_parser.add_argument("--manifest", type=Path, required=True)
    validate_shard_parser.add_argument("--cohort", type=Path, required=True)
    validate_shard_parser.add_argument("--cohort-sha256", required=True)
    validate_shard_parser.add_argument("--model-id", choices=MODEL_ORDER, required=True)
    validate_shard_parser.add_argument(
        "--shard-id", choices=ALLOWED_SHARD_IDS, type=int, required=True
    )
    validate_shard_parser.add_argument("--scope", required=True)
    validate_shard_parser.add_argument(
        "--max-num-seqs", choices=ALLOWED_MAX_NUM_SEQS, type=int, required=True
    )

    merge = sub.add_parser("merge-run")
    merge.add_argument("--cohort", type=Path, required=True)
    merge.add_argument("--cohort-sha256", required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--model-id", choices=MODEL_ORDER, required=True)
    merge.add_argument("--scope", required=True)
    merge.add_argument(
        "--max-num-seqs", choices=ALLOWED_MAX_NUM_SEQS, type=int, required=True
    )
    merge.add_argument("--orchestrator-timing", type=Path, required=True)

    validate_run_parser = sub.add_parser("validate-run")
    validate_run_parser.add_argument("--manifest", type=Path, required=True)
    validate_run_parser.add_argument("--cohort", type=Path, required=True)
    validate_run_parser.add_argument("--cohort-sha256", required=True)
    validate_run_parser.add_argument("--model-id", choices=MODEL_ORDER, required=True)
    validate_run_parser.add_argument("--scope", required=True)
    validate_run_parser.add_argument(
        "--max-num-seqs", choices=ALLOWED_MAX_NUM_SEQS, type=int, required=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run-shard":
            result = run_shard(
                model_id=args.model_id,
                shard_id=args.shard_id,
                physical_gpu_index=args.physical_gpu_index,
                cohort_path=args.cohort,
                cohort_sha256=args.cohort_sha256,
                output_dir=args.output_dir,
                resume=args.resume,
                smoke=args.smoke,
                gpu_wait_timeout_seconds=args.gpu_wait_timeout_seconds,
                gpu_poll_seconds=args.gpu_poll_seconds,
                max_num_seqs=args.max_num_seqs,
                control_dir=args.control_dir,
                run_token=args.run_token,
                inherited_lock_fd=args.inherited_lock_fd,
                control_timeout_seconds=args.control_timeout_seconds,
                formal_authorization_path=args.formal_authorization,
                expected_scope=args.scope,
            )
        elif args.command == "validate-shard":
            result = load_and_validate_shard(
                args.manifest,
                cohort_path=args.cohort,
                cohort_sha256=args.cohort_sha256,
                expected_model_id=args.model_id,
                expected_shard_id=args.shard_id,
                expected_scope=args.scope,
                max_num_seqs=args.max_num_seqs,
            )["manifest"]
        elif args.command == "merge-run":
            result = merge_run(
                cohort_path=args.cohort,
                cohort_sha256=args.cohort_sha256,
                output_dir=args.output_dir,
                model_id=args.model_id,
                scope=args.scope,
                max_num_seqs=args.max_num_seqs,
                orchestrator_timing_path=args.orchestrator_timing,
            )
        else:
            result = load_and_validate_run(
                args.manifest,
                cohort_path=args.cohort,
                cohort_sha256=args.cohort_sha256,
                expected_model_id=args.model_id,
                expected_scope=args.scope,
                max_num_seqs=args.max_num_seqs,
            )["manifest"]
        print(
            _canonical(
                {
                    "status": "valid"
                    if args.command.startswith("validate")
                    else result["status"],
                    "schema_version": result["schema_version"],
                    "payload_sha256": result["integrity"]["payload_sha256"],
                }
            )
        )
        return 0
    except (
        DualDp1GenerationError,
        dp2.VllmK5GenerationError,
        core.StochasticBootstrapGenerationError,
        OSError,
        ValueError,
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
    finally:
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ABSOLUTE_CHUNK_SIZE",
    "CASES_PER_MODEL",
    "CHUNKS_PER_MODEL",
    "DualDp1GenerationError",
    "EVALUATION_ID",
    "MODEL_ORDER",
    "ROW_SCHEMA",
    "assigned_shard",
    "build_result",
    "canonical_cases",
    "load_and_validate_run",
    "load_and_validate_shard",
    "load_cohort",
    "merge_run",
    "run_shard",
    "sampling_contract",
    "shard_cases",
    "validate_result",
    "validate_wal_rows",
]
