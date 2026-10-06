"""Generate the merged Core8 K=5 cohort with asynchronous vLLM batching.

This runner is independent of the historical Transformers/NF4 K=10 runner.
It consumes the exact prompt-token ledger produced by
``prepare_chk3_beta_core8_merged_vllm_k5`` and loads the three frozen model
artifacts as unquantized BF16 safetensors.  Forty requests (one meeting's
Core8 rows times five replicates) form an immutable absolute submission
chunk.  Two data-parallel engine ranks each hold one complete BF16 model
replica on physical GPU0/GPU1.  A preregistered 8/12/16 ``max_num_seqs``
candidate is frozen into every row and manifest, and vLLM continuously
refills that per-rank capacity.  Each request is appended, flushed, and
fsynced as soon as it finishes; the WAL therefore records completion order.
A separate canonical file is emitted only after all absolute tuple indices
are present.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import contextlib
import fcntl
import gc
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from jobs.eval import chk3_beta_core8_merged_contract as data_contract
from jobs.eval import eval_chk3_beta_core8_merged_stochastic_k10 as k10_profile
from jobs.eval import eval_chk3_stochastic_bootstrap_generation as core
from jobs.eval import prepare_chk3_beta_core8_merged_vllm_k5 as preparation
from open_r1.validator.loo_generation_spec import (
    derive_row_seed,
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ID = preparation.EVALUATION_ID
ROW_SCHEMA = "chk3-beta-core8-merged-vllm-k5-generation-row-v1"
RUN_MANIFEST_SCHEMA = "chk3-beta-core8-merged-vllm-k5-run-manifest-v1"
SUITE_MANIFEST_SCHEMA = "chk3-beta-core8-merged-vllm-k5-suite-manifest-v1"
STATE_SCHEMA = "chk3-beta-core8-merged-vllm-k5-state-v1"
CHUNK_RECEIPT_SCHEMA = "chk3-beta-core8-merged-vllm-k5-chunk-receipt-v1"
MODEL_ORDER = preparation.MODEL_ORDER
MODEL_LABELS = core.MODEL_LABELS
REPLICATE_SEEDS = preparation.REPLICATE_SEEDS
PROMPTS = preparation.EXPECTED_PROMPTS
CASES_PER_MODEL = preparation.EXPECTED_CASES_PER_MODEL
ABSOLUTE_CHUNK_SIZE = preparation.ABSOLUTE_CHUNK_SIZE
CHUNKS_PER_MODEL = CASES_PER_MODEL // ABSOLUTE_CHUNK_SIZE
SMOKE_PROMPTS = len(data_contract.CORE_TOPICS)
SMOKE_CASES_PER_MODEL = SMOKE_PROMPTS * len(REPLICATE_SEEDS)

TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 50
REPETITION_PENALTY = 1.0
MAX_NEW_TOKENS = 3072
TAIL_TOKENS = 1024
MAX_MODEL_LEN = 4096
ALLOWED_MAX_NUM_SEQS = (8, 12, 16)
MAX_NUM_BATCHED_TOKENS = 4096
QUEUED_REQUESTS_PER_CHUNK = ABSOLUTE_CHUNK_SIZE
GPU_MEMORY_UTILIZATION = 0.95
ALLOWED_PHYSICAL_GPU_INDEXES = (0, 1)
DATA_PARALLEL_SIZE = 2
TENSOR_PARALLEL_SIZE = 1
PIPELINE_PARALLEL_SIZE = 1
DATA_PARALLEL_MASTER_IP = "127.0.0.1"
DATA_PARALLEL_MASTER_PORT_POLICY = "vllm_ephemeral_open_port_per_engine"
GPU_LOCK_PATH_TEMPLATE = "/tmp/fomc_trainer_chk3_beta_core8_vllm_k5_gpu{index}.lock"
EXPECTED_VLLM_VERSION = "0.8.5.post1"
REQUIRED_VLLM_ENV = {
    "VLLM_USE_V1": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "VLLM_DP_MASTER_IP": DATA_PARALLEL_MASTER_IP,
    # ParallelConfig allocates the actual engine port itself for DP>1.  Zero
    # keeps the ambient SPMD fallback inert and the policy explicit.
    "VLLM_DP_MASTER_PORT": "0",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class VllmK5GenerationError(RuntimeError):
    """The frozen asynchronous K=5 generation contract was violated."""


@dataclass
class WalTracker:
    """Incremental SHA/byte/row accounting for a canonical JSONL WAL."""

    digest: Any
    bytes: int = 0
    rows: int = 0

    @classmethod
    def empty(cls) -> "WalTracker":
        return cls(hashlib.sha256())

    def update(self, encoded: bytes) -> None:
        self.digest.update(encoded)
        self.bytes += len(encoded)
        self.rows += 1

    def binding(self, path: Path) -> dict[str, Any]:
        return {
            "path": str(path.expanduser().resolve()),
            "sha256": self.digest.hexdigest(),
            "bytes": self.bytes,
            "rows": self.rows,
        }


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _encoded_row(value: Mapping[str, Any]) -> bytes:
    return (_canonical(dict(value)) + "\n").encode("utf-8")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_binding(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    binding = core._file_binding(path)
    if rows is not None:
        binding["rows"] = rows
    return binding


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _validate_max_num_seqs(value: int) -> int:
    if isinstance(value, bool) or value not in ALLOWED_MAX_NUM_SEQS:
        raise VllmK5GenerationError(
            f"max_num_seqs must be one of {ALLOWED_MAX_NUM_SEQS}; observed={value!r}"
        )
    return value


def _validate_physical_gpu_index(value: int) -> int:
    if isinstance(value, bool) or value not in ALLOWED_PHYSICAL_GPU_INDEXES:
        raise VllmK5GenerationError("physical_gpu_index must be exactly 0 or 1")
    return value


def gpu_lock_path(physical_gpu_index: int) -> Path:
    index = _validate_physical_gpu_index(physical_gpu_index)
    return Path(GPU_LOCK_PATH_TEMPLATE.format(index=index))


def sampling_contract(*, max_num_seqs: int) -> dict[str, Any]:
    max_num_seqs = _validate_max_num_seqs(max_num_seqs)
    return {
        "backend": "vllm-async-engine-v1-continuous-batching",
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
        "max_num_seqs": max_num_seqs,
        "max_num_seqs_selection": "benchmark_candidates_8_12_16_then_freeze_one_formal_value",
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "swap_space_gib": 0,
        "cpu_offload_gib": 0,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "data_parallel_size": DATA_PARALLEL_SIZE,
        "pipeline_parallel_size": PIPELINE_PARALLEL_SIZE,
        "parallel_topology": "two_full_bf16_model_replicas_one_per_physical_gpu",
        "data_parallel_request_routing": "least_inflight_requests",
        "distributed_world_backend": "nccl",
        "data_parallel_control_backend": "gloo_cpu",
        "dense_model_per_layer_tensor_collectives": False,
        "data_parallel_master_ip": DATA_PARALLEL_MASTER_IP,
        "data_parallel_master_port_policy": DATA_PARALLEL_MASTER_PORT_POLICY,
        "physical_gpu_indexes": list(ALLOWED_PHYSICAL_GPU_INDEXES),
        "enforce_eager": True,
        "cuda_graphs": False,
        "async_output_processing": False,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "engine_seed": 0,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "row_seed": "derive_row_seed(replicate_seed,sample_id)",
        "input_contract": "consume_exact_fomc_prompt_token_ledger_no_runtime_chat_template",
        "absolute_chunk_size": ABSOLUTE_CHUNK_SIZE,
        "absolute_chunk_semantics": "one_meeting_core8_times_five_replicates",
        "queued_requests_per_chunk": QUEUED_REQUESTS_PER_CHUNK,
        "wal_order": "request_completion_order",
        "canonical_output_order": "absolute_case_index",
        "required_environment": dict(REQUIRED_VLLM_ENV),
        "recommended_model_order": list(MODEL_ORDER),
    }


def _sampling_contract_sha256(*, max_num_seqs: int) -> str:
    return _sha256_text(_canonical(sampling_contract(max_num_seqs=max_num_seqs)))


def _assert_required_environment(*, require_cuda: bool) -> None:
    for key, expected in REQUIRED_VLLM_ENV.items():
        if os.environ.get(key) != expected:
            raise VllmK5GenerationError(
                f"{key} must be exactly {expected!r}; observed={os.environ.get(key)!r}"
            )
    if require_cuda:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible != "0,1" or os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
            raise VllmK5GenerationError(
                "physical GPU0 and GPU1 must be the only visible CUDA devices "
                "in the exact order CUDA_VISIBLE_DEVICES=0,1"
            )


def _read_canonical_jsonl(
    path: Path, *, recorded_prefix_rows: int | None = None
) -> tuple[list[dict[str, Any]], WalTracker, dict[str, Any] | None]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise VllmK5GenerationError(f"JSONL is a symlink: {unresolved}")
    path = unresolved.resolve()
    if not path.is_file():
        raise VllmK5GenerationError(f"JSONL is missing: {path}")
    tracker = WalTracker.empty()
    rows: list[dict[str, Any]] = []
    prefix: dict[str, Any] | None = (
        tracker.binding(path) if recorded_prefix_rows == 0 else None
    )
    with path.open("rb") as handle:
        while raw := handle.readline():
            if not raw.endswith(b"\n"):
                raise VllmK5GenerationError(f"JSONL has an incomplete tail: {path}")
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise VllmK5GenerationError(f"invalid JSONL row in {path}") from exc
            if not isinstance(value, dict) or raw != _encoded_row(value):
                raise VllmK5GenerationError(f"JSONL row is not canonical: {path}")
            rows.append(value)
            tracker.update(raw)
            if recorded_prefix_rows == tracker.rows:
                prefix = tracker.binding(path)
    if tracker.bytes != path.stat().st_size:
        raise VllmK5GenerationError(f"JSONL changed while scanning: {path}")
    return rows, tracker, prefix


def _verify_file_binding(binding: Mapping[str, Any], *, parent: Path) -> Path:
    unresolved = Path(str(binding.get("path") or "")).expanduser()
    if unresolved.is_symlink():
        raise VllmK5GenerationError("bound artifact is a symlink")
    path = unresolved.resolve()
    if not path.is_file():
        raise VllmK5GenerationError(f"bound artifact is missing: {path}")
    if path.parent != parent.resolve():
        raise VllmK5GenerationError("bound artifact escaped its sealed directory")
    if (
        binding.get("sha256") != core._sha256_file(path)
        or binding.get("bytes") != path.stat().st_size
    ):
        raise VllmK5GenerationError(f"bound artifact hash/size drift: {path}")
    return path


def _configure_source_profile(cohort: Mapping[str, Any]) -> None:
    releases = cohort.get("harmonized_source_releases")
    if not isinstance(releases, Mapping):
        raise VllmK5GenerationError("cohort source release bindings are missing")
    try:
        pre = releases["pre2009_external"]
        post = releases["post2008_chk3_release"]
        k10_profile.configure_profile(
            pre_release_manifest=Path(str(pre["path"])),
            post_release_manifest=Path(str(post["path"])),
            pre_release_sha256=str(pre["sha256"]),
            post_release_sha256=str(post["sha256"]),
        )
    except (KeyError, TypeError, data_contract.MergedCore8ContractError) as exc:
        raise VllmK5GenerationError(str(exc)) from exc


def load_cohort(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], list[dict[str, Any]], Mapping[str, Any], dict[str, Any]]:
    """Deep-load the cohort, source manifest, exact token ledger, and rows."""

    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise VllmK5GenerationError("cohort manifest is a symlink")
    path = unresolved.resolve()
    if not path.is_file() or core._sha256_file(path) != expected_sha256:
        raise VllmK5GenerationError("cohort manifest SHA256 mismatch")
    cohort = core._read_json(path)
    try:
        validate_manifest_integrity(cohort)
    except Exception as exc:
        raise VllmK5GenerationError(f"cohort integrity failed: {exc}") from exc
    if (
        cohort.get("schema_version") != preparation.COHORT_SCHEMA
        or cohort.get("status") != "complete"
        or cohort.get("immutable") is not True
        or cohort.get("evaluation_id") != EVALUATION_ID
        or cohort.get("task_contract_id") != core.TASK_CONTRACT_ID
        or cohort.get("generation_design")
        != {
            "backend": "vllm-async-engine-v1-continuous-batching",
            "models": list(MODEL_ORDER),
            "prompts": PROMPTS,
            "meetings": data_contract.EXPECTED_MEETINGS,
            "topics_per_meeting": len(data_contract.CORE_TOPICS),
            "replicate_seeds": list(REPLICATE_SEEDS),
            "replicates": len(REPLICATE_SEEDS),
            "rows_per_model": CASES_PER_MODEL,
            "total_rows": CASES_PER_MODEL * len(MODEL_ORDER),
            "canonical_case_order": "sample_manifest_order_then_replicate_id",
            "absolute_chunk_size": ABSOLUTE_CHUNK_SIZE,
            "absolute_chunk_count_per_model": CHUNKS_PER_MODEL,
            "partial_chunk_count_per_model": 0,
            "row_seed": "derive_row_seed(replicate_seed,sample_id)",
        }
    ):
        raise VllmK5GenerationError("cohort identity/design drift")
    _configure_source_profile(cohort)
    sample_binding = cohort.get("source_sample_manifest")
    if not isinstance(sample_binding, Mapping):
        raise VllmK5GenerationError("source sample manifest binding is missing")
    sample_path = Path(str(sample_binding.get("path"))).expanduser().resolve()
    try:
        sample_manifest, sample_sha = core.load_full_test_sample_manifest(
            sample_path, str(sample_binding.get("sha256"))
        )
        bound_rows = core._load_bound_rows(sample_manifest)
    except core.StochasticBootstrapGenerationError as exc:
        raise VllmK5GenerationError(str(exc)) from exc
    if (
        sample_sha != sample_binding.get("sha256")
        or sample_manifest["integrity"]["payload_sha256"]
        != sample_binding.get("payload_sha256")
        or sample_binding.get("transport_generation_design_ignored") is not True
    ):
        raise VllmK5GenerationError("source sample binding drift")

    tokenizer_binding = cohort.get("fomc_tokenizer")
    if not isinstance(tokenizer_binding, Mapping):
        raise VllmK5GenerationError("FOMC tokenizer binding is missing")
    tokenizer_path = Path(str(tokenizer_binding.get("path"))).expanduser().resolve()
    files = tokenizer_binding.get("files")
    if not isinstance(files, Mapping) or not files:
        raise VllmK5GenerationError("FOMC tokenizer file inventory is missing")
    for name, expected in files.items():
        file_path = tokenizer_path / str(name)
        if not file_path.is_file() or core._sha256_file(file_path) != expected:
            raise VllmK5GenerationError(f"FOMC tokenizer drift: {name}")
    if dict(sample_manifest["tokenizer"]) != {
        "path": tokenizer_binding["path"],
        "files": dict(files),
    }:
        raise VllmK5GenerationError("cohort/source FOMC tokenizer mismatch")
    prompt_contract = cohort.get("prompt_contract")
    if not isinstance(prompt_contract, Mapping):
        raise VllmK5GenerationError("cohort prompt contract is missing")
    config_path = (
        Path(str(prompt_contract.get("training_config"))).expanduser().resolve()
    )
    try:
        system_prompt, suffix, config_sha = (
            core.native_eval.native_probe._load_prompt_contract(config_path)
        )
    except core.native_eval.native_probe.Chk3ProbeError as exc:
        raise VllmK5GenerationError(str(exc)) from exc
    if (
        config_sha != prompt_contract.get("training_config_sha256")
        or _sha256_text(system_prompt) != prompt_contract.get("system_prompt_sha256")
        or (_sha256_text(suffix) if suffix is not None else None)
        != prompt_contract.get("user_prompt_suffix_sha256")
        or dict(sample_manifest["prompt_contract"]) != dict(prompt_contract)
    ):
        raise VllmK5GenerationError("cohort/source prompt contract drift")

    ledger_binding = cohort.get("token_ledger")
    if not isinstance(ledger_binding, Mapping):
        raise VllmK5GenerationError("token ledger binding is missing")
    ledger_path = _verify_file_binding(ledger_binding, parent=path.parent)
    ledger, ledger_tracker, _ = _read_canonical_jsonl(ledger_path)
    if (
        len(ledger) != PROMPTS
        or ledger_binding.get("rows") != PROMPTS
        or ledger_tracker.rows != PROMPTS
        or ledger_binding.get("row_schema_version") != preparation.LEDGER_ROW_SCHEMA
        or ledger_binding.get("full_prompt_token_ids_persisted") is not True
        or ledger_binding.get("vllm_runtime_chat_templating") is not False
    ):
        raise VllmK5GenerationError("token ledger coverage/contract drift")
    samples = sample_manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != PROMPTS:
        raise VllmK5GenerationError("source sample inventory is not N2048")
    for index, (entry, sample) in enumerate(zip(ledger, samples, strict=True)):
        token_ids = entry.get("prompt_token_ids")
        source = bound_rows[str(sample["sample_id"])]
        try:
            messages = core.native_eval.native_probe._messages(
                source,
                system_prompt=system_prompt,
                user_prompt_suffix=suffix,
            )
        except core.native_eval.native_probe.Chk3ProbeError as exc:
            raise VllmK5GenerationError(str(exc)) from exc
        if (
            entry.get("schema_version") != preparation.LEDGER_ROW_SCHEMA
            or entry.get("absolute_prompt_index") != index
            or entry.get("source_sample_line_number") != index + 1
            or entry.get("sample_id") != sample.get("sample_id")
            or entry.get("meeting_id") != sample.get("meeting_id")
            or entry.get("prompt_sha256") != sample.get("prompt_sha256")
            or entry.get("messages_sha256") != _sha256_text(_canonical(messages))
            or not isinstance(token_ids, list)
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in token_ids
            )
            or entry.get("prompt_token_count") != len(token_ids)
            or entry.get("prompt_token_count") != sample.get("prompt_token_count")
            or entry.get("prompt_token_ids_sha256")
            != _sha256_text(_canonical(token_ids))
            or len(token_ids) + MAX_NEW_TOKENS > MAX_MODEL_LEN
        ):
            raise VllmK5GenerationError(f"token ledger drift at index {index}")
    return cohort, ledger, sample_manifest, dict(bound_rows)


def canonical_cases(
    *,
    samples: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    smoke: bool,
) -> list[dict[str, Any]]:
    # One complete Core8 meeting at K=5 fills one immutable 40-case window.
    # This is large enough to exercise every 8/12/16 scheduler candidate,
    # DP load balancing, paired seeds, the per-completion WAL, and receipting.
    selected_samples = list(samples[:SMOKE_PROMPTS] if smoke else samples)
    selected_ledger = list(ledger[:SMOKE_PROMPTS] if smoke else ledger)
    seeds = REPLICATE_SEEDS
    cases: list[dict[str, Any]] = []
    for sample, token_entry in zip(selected_samples, selected_ledger, strict=True):
        for replicate_id, replicate_seed in enumerate(seeds):
            absolute_case_index = len(cases)
            cases.append(
                {
                    "absolute_case_index": absolute_case_index,
                    "absolute_chunk_id": absolute_case_index // ABSOLUTE_CHUNK_SIZE,
                    "sample": sample,
                    "token_entry": token_entry,
                    "replicate_id": replicate_id,
                    "replicate_seed": replicate_seed,
                    "row_seed": derive_row_seed(
                        replicate_seed, str(sample["sample_id"])
                    ),
                }
            )
    return cases


def _normalize_finish_reason(
    *,
    raw_finish_reason: Any,
    raw_stop_reason: Any,
    generated_token_ids: Sequence[int],
    eos_token_ids: Sequence[int],
) -> str:
    if raw_stop_reason is not None:
        raise VllmK5GenerationError(
            "vLLM reported an explicit stop reason although no explicit stop is configured"
        )
    eos = set(eos_token_ids)
    positions = [
        index for index, token_id in enumerate(generated_token_ids) if token_id in eos
    ]
    if raw_finish_reason == "stop":
        if positions != [len(generated_token_ids) - 1]:
            raise VllmK5GenerationError(
                "vLLM stop completion did not retain exactly one final EOS token"
            )
        return "eos"
    if raw_finish_reason == "length":
        if positions or len(generated_token_ids) != MAX_NEW_TOKENS:
            raise VllmK5GenerationError("vLLM length finish/token contract drift")
        return "length"
    raise VllmK5GenerationError(
        f"unsupported vLLM finish reason: {raw_finish_reason!r}"
    )


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
) -> dict[str, Any]:
    max_num_seqs = _validate_max_num_seqs(max_num_seqs)
    sample = case["sample"]
    token_entry = case["token_entry"]
    source_prompt = str(source_row["prompt"])
    source_analysis = core.native_eval.native_probe.extract_source_analysis(
        source_prompt
    )
    normalized_finish = _normalize_finish_reason(
        raw_finish_reason=raw_finish_reason,
        raw_stop_reason=raw_stop_reason,
        generated_token_ids=generated_token_ids,
        eos_token_ids=eos_token_ids,
    )
    try:
        base = core.native_eval.build_full_result(
            text=generated_text,
            generated_token_ids=generated_token_ids,
            eos_token_ids=eos_token_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            tail_tokens=TAIL_TOKENS,
            source_prompt=source_prompt,
            source_analysis=source_analysis,
            reference_response=str(source_row["response"]),
            load_in_4bit=False,
            attn_implementation="vllm-engine-managed",
            pad_token_id=pad_token_id,
            stage_id=model_id,
            model_label=model_label,
            sample_manifest_sha256=sample_manifest_sha256,
            sample=sample,
            seed=int(case["row_seed"]),
        )
    except core.native_eval.NativeThreeModelEvalError as exc:
        raise VllmK5GenerationError(str(exc)) from exc
    row: dict[str, Any] = {
        **base,
        "schema_version": ROW_SCHEMA,
        "evaluation_id": EVALUATION_ID,
        "model_id": model_id,
        "meeting_id": sample["meeting_id"],
        "replicate_id": case["replicate_id"],
        "replicate_seed": case["replicate_seed"],
        "row_seed": case["row_seed"],
        "absolute_case_index": case["absolute_case_index"],
        "absolute_chunk_id": case["absolute_chunk_id"],
        "absolute_chunk_size": ABSOLUTE_CHUNK_SIZE,
        "request_id": (
            f"{model_id}-case-{int(case['absolute_case_index']):05d}-"
            f"seed-{int(case['row_seed'])}"
        ),
        "generation_mode": "stochastic_sampling",
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "inference_backend": "vllm-async-engine-v1",
        "model_dtype": "bfloat16",
        "quantization": None,
        "data_parallel_size": DATA_PARALLEL_SIZE,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "physical_gpu_indexes": list(ALLOWED_PHYSICAL_GPU_INDEXES),
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_num_seqs": max_num_seqs,
        "input_token_count": token_entry["prompt_token_count"],
        "prompt_token_ledger_line_number": (
            int(token_entry["absolute_prompt_index"]) + 1
        ),
        "prompt_token_ids_sha256": token_entry["prompt_token_ids_sha256"],
        "input_truncated": False,
        "finish_reason": normalized_finish,
        "vllm_raw_finish_reason": raw_finish_reason,
        "vllm_raw_stop_reason": raw_stop_reason,
        "sampling_contract_sha256": _sampling_contract_sha256(
            max_num_seqs=max_num_seqs
        ),
        "source_artifact_sha256s": copy.deepcopy(dict(source_artifact_sha256s)),
        "semantic_status": "pending_offline",
    }
    row.update(k10_profile._sample_metadata(sample))
    generation_metrics, audit = core._generation_metrics(row)
    row["generation_metrics"] = generation_metrics
    row["preregistered_core_valid"] = bool(audit["preregistered_core_valid"])
    row["preregistered_core_failures"] = list(audit["preregistered_core_failures"])
    row["six_metrics"] = {
        **generation_metrics,
        "mpnet_cosine": None,
        "bertscore_f1": None,
    }
    return row


def validate_result(
    row: Mapping[str, Any],
    *,
    model_id: str,
    model_label: str,
    case: Mapping[str, Any],
    source_row: Mapping[str, Any],
    sample_manifest_sha256: str,
    source_artifact_sha256s: Mapping[str, Any],
    max_num_seqs: int,
) -> None:
    expected = build_result(
        model_id=model_id,
        model_label=model_label,
        case=case,
        source_row=source_row,
        sample_manifest_sha256=sample_manifest_sha256,
        source_artifact_sha256s=source_artifact_sha256s,
        generated_text=str(row.get("generated_text") or ""),
        generated_token_ids=row.get("generated_token_ids") or [],
        eos_token_ids=row.get("eos_token_ids") or [],
        pad_token_id=row.get("pad_token_id"),
        raw_finish_reason=row.get("vllm_raw_finish_reason"),
        raw_stop_reason=row.get("vllm_raw_stop_reason"),
        max_num_seqs=max_num_seqs,
    )
    if dict(row) != expected:
        changed = sorted(
            key for key in set(row) | set(expected) if row.get(key) != expected.get(key)
        )
        raise VllmK5GenerationError(
            f"persisted generation row drift at fields: {changed[:12]}"
        )


def _source_hashes(
    cohort_path: Path,
    cohort_sha256: str,
    cohort: Mapping[str, Any],
    sample_manifest: Mapping[str, Any],
    max_num_seqs: int,
) -> dict[str, Any]:
    source_sample = cohort["source_sample_manifest"]
    hashes = core._source_hashes(sample_manifest, str(source_sample["sha256"]))
    hashes.update(
        {
            "vllm_k5_cohort_sha256": cohort_sha256,
            "vllm_k5_token_ledger_sha256": cohort["token_ledger"]["sha256"],
            "vllm_k5_sampling_contract_sha256": _sampling_contract_sha256(
                max_num_seqs=max_num_seqs
            ),
        }
    )
    hashes["implementation_sources"].update(
        {
            "vllm_k5_runner": _file_binding(Path(__file__).resolve()),
            "vllm_k5_preparer": _file_binding(
                Path(str(preparation.__file__)).resolve()
            ),
        }
    )
    return hashes


def _validate_bf16_anchor(anchor: Mapping[str, Any]) -> None:
    model_path = Path(str(anchor["model_path"])).resolve()
    config = core._read_json(model_path / "config.json")
    declared = config.get("dtype", config.get("torch_dtype"))
    if declared != "bfloat16" or config.get("vocab_size") != 128256:
        raise VllmK5GenerationError(
            f"{anchor['model_id']} is not the frozen BF16/vocab model contract"
        )


def _expected_cases(*, smoke: bool) -> int:
    return SMOKE_CASES_PER_MODEL if smoke else CASES_PER_MODEL


def _limitations(*, smoke: bool) -> dict[str, bool]:
    return {
        "semantic_metrics_pending_offline": True,
        "smoke_not_formal_evidence": smoke,
        "checkpoint_318_post_selection_robustness": not smoke,
        "data_parallel_world_initializes_nccl": True,
        "data_parallel_control_uses_cpu_gloo": True,
        "dense_model_has_no_per_layer_dp_tensor_collectives": True,
        "enforce_eager_disables_cuda_graphs": True,
        "max_num_seqs_requires_smoke_benchmark_selection": True,
    }


def _completion_status(
    indexes: set[int], *, expected_cases: int
) -> tuple[int, set[int]]:
    if any(index < 0 or index >= expected_cases for index in indexes):
        raise VllmK5GenerationError("WAL has an out-of-range absolute case index")
    full_chunks = 0
    while True:
        start = full_chunks * ABSOLUTE_CHUNK_SIZE
        if start >= expected_cases:
            break
        end = min(start + ABSOLUTE_CHUNK_SIZE, expected_cases)
        if set(range(start, end)).issubset(indexes):
            full_chunks += 1
        else:
            break
    next_start = full_chunks * ABSOLUTE_CHUNK_SIZE
    allowed_partial = set(
        range(next_start, min(next_start + ABSOLUTE_CHUNK_SIZE, expected_cases))
    )
    extras = indexes - set(range(next_start))
    if not extras.issubset(allowed_partial):
        raise VllmK5GenerationError("WAL skips an incomplete absolute chunk")
    return full_chunks, extras


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
) -> dict[int, dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = row.get("absolute_case_index")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index in by_index
            or index < 0
            or index >= len(cases)
        ):
            raise VllmK5GenerationError("duplicate/invalid WAL absolute case index")
        case = cases[index]
        sample_id = str(case["sample"]["sample_id"])
        validate_result(
            row,
            model_id=model_id,
            model_label=model_label,
            case=case,
            source_row=bound_rows[sample_id],
            sample_manifest_sha256=sample_manifest_sha256,
            source_artifact_sha256s=source_artifact_sha256s,
            max_num_seqs=max_num_seqs,
        )
        by_index[index] = dict(row)
    _completion_status(set(by_index), expected_cases=len(cases))
    return by_index


def _chunk_receipt(
    *,
    model_id: str,
    chunk_id: int,
    rows_by_index: Mapping[int, Mapping[str, Any]],
    expected_cases: int,
) -> dict[str, Any]:
    start = chunk_id * ABSOLUTE_CHUNK_SIZE
    end = min(start + ABSOLUTE_CHUNK_SIZE, expected_cases)
    bindings = []
    for index in range(start, end):
        row = rows_by_index.get(index)
        if row is None:
            raise VllmK5GenerationError("cannot receipt an incomplete absolute chunk")
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
        "absolute_chunk_id": chunk_id,
        "absolute_case_start": start,
        "absolute_case_end_exclusive": end,
        "cases": end - start,
        "canonical_row_bindings_sha256": _sha256_text(_canonical(bindings)),
    }


def _validate_receipts(
    receipts: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
    rows_by_index: Mapping[int, Mapping[str, Any]],
    expected_cases: int,
) -> None:
    full_chunks, _ = _completion_status(
        set(rows_by_index), expected_cases=expected_cases
    )
    if len(receipts) > full_chunks:
        raise VllmK5GenerationError("chunk receipt count exceeds completed chunks")
    for chunk_id, receipt in enumerate(receipts):
        expected = _chunk_receipt(
            model_id=model_id,
            chunk_id=chunk_id,
            rows_by_index=rows_by_index,
            expected_cases=expected_cases,
        )
        if dict(receipt) != expected:
            raise VllmK5GenerationError(f"chunk receipt drift at {chunk_id}")
    if len(receipts) < full_chunks - 1:
        raise VllmK5GenerationError("more than one completed chunk lacks a receipt")


def _state_payload(
    *,
    status: str,
    model_id: str,
    expected_cases: int,
    resume_count: int,
    wal_binding: Mapping[str, Any],
    receipt_binding: Mapping[str, Any],
    completed_indexes: set[int],
    engine_runtime: Mapping[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    full_chunks, partial = _completion_status(
        completed_indexes, expected_cases=expected_cases
    )
    return {
        "schema_version": STATE_SCHEMA,
        "status": status,
        "updated_at_utc": core._utc_now(),
        "evaluation_id": EVALUATION_ID,
        "model_id": model_id,
        "expected_cases": expected_cases,
        "completed_cases": len(completed_indexes),
        "completed_absolute_chunks": full_chunks,
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
    encoded = _encoded_row(value)
    handle.write(encoded.decode("utf-8"))
    handle.flush()
    os.fsync(handle.fileno())
    tracker.update(encoded)


def _runtime_versions() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "vllm": _package_version("vllm"),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "tokenizers": _package_version("tokenizers"),
    }


def _validate_sealed_runtime(
    *,
    launch_runtime: Any,
    engine_runtime: Any,
    max_num_seqs: int,
    expected_cases: int,
    expected_generated_output_tokens: int,
    resume_count: int,
) -> None:
    """Deep-check the sealed dual-GPU launch and live-engine evidence."""

    max_num_seqs = _validate_max_num_seqs(max_num_seqs)
    if not isinstance(launch_runtime, Mapping):
        raise VllmK5GenerationError("sealed launch runtime is missing")
    identities = launch_runtime.get("physical_gpu_identities")
    if not isinstance(identities, list) or len(identities) != DATA_PARALLEL_SIZE:
        raise VllmK5GenerationError("sealed launch does not bind two GPU identities")
    expected_indexes = list(ALLOWED_PHYSICAL_GPU_INDEXES)
    uuids: list[str] = []
    for index, identity in enumerate(identities):
        if (
            not isinstance(identity, Mapping)
            or identity.get("physical_gpu_index") != index
            or not isinstance(identity.get("uuid"), str)
            or not str(identity["uuid"]).startswith("GPU-")
            or not isinstance(identity.get("pci_bus_id"), str)
            or not str(identity["pci_bus_id"])
        ):
            raise VllmK5GenerationError("sealed physical GPU identity drift")
        uuids.append(str(identity["uuid"]))
    if len(set(uuids)) != DATA_PARALLEL_SIZE:
        raise VllmK5GenerationError("sealed physical GPU UUIDs are not distinct")
    if (
        launch_runtime.get("vllm") != EXPECTED_VLLM_VERSION
        or launch_runtime.get("cuda_visible_devices") != "0,1"
        or launch_runtime.get("cuda_device_order") != "PCI_BUS_ID"
        or launch_runtime.get("physical_gpu_indexes") != expected_indexes
        or launch_runtime.get("gpu_lock_paths")
        != [
            str(gpu_lock_path(index).resolve())
            for index in ALLOWED_PHYSICAL_GPU_INDEXES
        ]
        or launch_runtime.get("max_num_seqs") != max_num_seqs
    ):
        raise VllmK5GenerationError("sealed dual-GPU launch runtime drift")

    if not isinstance(engine_runtime, Mapping):
        raise VllmK5GenerationError("sealed engine runtime is missing")
    port = engine_runtime.get("data_parallel_master_port")
    expected_visible = [
        {"logical_cuda_index": index, **dict(identities[index])}
        for index in ALLOWED_PHYSICAL_GPU_INDEXES
    ]
    if (
        engine_runtime.get("vllm") != EXPECTED_VLLM_VERSION
        or engine_runtime.get("visible_cuda_devices") != expected_visible
        or engine_runtime.get("data_parallel_master_ip") != DATA_PARALLEL_MASTER_IP
        or not isinstance(port, int)
        or isinstance(port, bool)
        or port <= 0
        or engine_runtime.get("data_parallel_master_port_policy")
        != DATA_PARALLEL_MASTER_PORT_POLICY
        or engine_runtime.get("distributed_world_backend") != "nccl"
        or engine_runtime.get("data_parallel_control_backend") != "gloo_cpu"
        or engine_runtime.get("dense_model_per_layer_tensor_collectives") is not False
        or engine_runtime.get("data_parallel_request_routing")
        != "least_inflight_requests"
        or engine_runtime.get("enforce_eager") is not True
        or engine_runtime.get("cuda_graphs") is not False
        or engine_runtime.get("async_output_processing") is not False
        or engine_runtime.get("engine") != sampling_contract(max_num_seqs=max_num_seqs)
    ):
        raise VllmK5GenerationError("sealed vLLM engine runtime drift")

    core_bindings = engine_runtime.get("engine_core_gpu_bindings")
    if not isinstance(core_bindings, list) or len(core_bindings) != DATA_PARALLEL_SIZE:
        raise VllmK5GenerationError("sealed EngineCore GPU evidence is missing")
    core_pids: list[int] = []
    for index, binding in enumerate(core_bindings):
        if not isinstance(binding, Mapping):
            raise VllmK5GenerationError("sealed EngineCore GPU evidence drift")
        pid = binding.get("engine_core_pid")
        processes = binding.get("descendant_compute_processes")
        if (
            binding.get("physical_gpu_index") != index
            or binding.get("data_parallel_rank") != index
            or not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 0
            or not isinstance(processes, list)
            or not any(
                isinstance(process, Mapping)
                and process.get("physical_gpu_index") == index
                and process.get("pid") == pid
                for process in processes
            )
        ):
            raise VllmK5GenerationError("sealed EngineCore GPU evidence drift")
        core_pids.append(pid)
    if len(set(core_pids)) != DATA_PARALLEL_SIZE:
        raise VllmK5GenerationError("sealed EngineCore PIDs are not distinct")

    model_load_seconds = engine_runtime.get("model_load_wall_seconds")
    generation_seconds = engine_runtime.get("generation_wall_seconds")
    generated_tokens = engine_runtime.get("generated_output_tokens")
    timed_session_tokens = engine_runtime.get("timed_session_generated_output_tokens")
    timed_requests = engine_runtime.get("completed_requests_in_timed_session")
    tokens_per_second = engine_runtime.get("output_tokens_per_second")
    if (
        not isinstance(model_load_seconds, (int, float))
        or isinstance(model_load_seconds, bool)
        or float(model_load_seconds) <= 0
        or not isinstance(generation_seconds, (int, float))
        or isinstance(generation_seconds, bool)
        or float(generation_seconds) <= 0
        or generated_tokens != expected_generated_output_tokens
        or not isinstance(timed_session_tokens, int)
        or isinstance(timed_session_tokens, bool)
        or timed_session_tokens <= 0
        or timed_session_tokens > expected_generated_output_tokens
        or not isinstance(timed_requests, int)
        or isinstance(timed_requests, bool)
        or timed_requests <= 0
        or timed_requests > expected_cases
        or not isinstance(tokens_per_second, (int, float))
        or isinstance(tokens_per_second, bool)
        or float(tokens_per_second) <= 0
        or not math.isclose(
            float(tokens_per_second),
            expected_generated_output_tokens / float(generation_seconds),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or engine_runtime.get("timing_scope")
        != "current_engine_session_after_dp_pid_validation"
        or engine_runtime.get("preemption_count") is not None
        or engine_runtime.get("preemption_count_status")
        != "unavailable_vllm_async_public_api"
    ):
        raise VllmK5GenerationError("sealed generation timing evidence drift")
    expected_speed_valid = bool(
        resume_count == 0
        and timed_requests == expected_cases
        and timed_session_tokens == expected_generated_output_tokens
    )
    if (
        engine_runtime.get("speed_measurement_valid_for_candidate_selection")
        is not expected_speed_valid
    ):
        raise VllmK5GenerationError("sealed benchmark eligibility drift")

    lease = engine_runtime.get("gpu_lease")
    if (
        not isinstance(lease, Mapping)
        or lease.get("physical_gpu_indexes") != expected_indexes
        or lease.get("lock_paths") != launch_runtime.get("gpu_lock_paths")
        or lease.get("external_compute_pids_at_acquire") != []
        or not isinstance(lease.get("acquired_at_utc"), str)
        or not isinstance(lease.get("wait_seconds"), (int, float))
        or isinstance(lease.get("wait_seconds"), bool)
        or float(lease["wait_seconds"]) < 0
    ):
        raise VllmK5GenerationError("sealed dual-GPU lease evidence drift")


def _summary(
    rows_by_index: Mapping[int, Mapping[str, Any]], *, smoke: bool
) -> dict[str, Any]:
    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    expected = _expected_cases(smoke=smoke)
    if len(rows) != expected:
        raise VllmK5GenerationError("cannot summarize an incomplete run")
    failures: Counter[str] = Counter()
    for row in rows:
        failures.update(row["preregistered_core_failures"])
    return {
        "status": "complete",
        "evaluation_scope": "infrastructure_smoke" if smoke else "formal_merged_panel",
        "cases": len(rows),
        "unique_samples": len({row["sample_id"] for row in rows}),
        "unique_meetings": len({row["meeting_id"] for row in rows}),
        "replicates": len({row["replicate_id"] for row in rows}),
        "input_truncation_cases": sum(bool(row["input_truncated"]) for row in rows),
        "eos_cases": sum(row["finish_reason"] == "eos" for row in rows),
        "length_cases": sum(row["finish_reason"] == "length" for row in rows),
        "core_valid_cases": sum(bool(row["preregistered_core_valid"]) for row in rows),
        "core_failure_counts": dict(sorted(failures.items())),
    }


def _descendant_pids() -> set[int]:
    """Return this process and descendants so vLLM workers are not foreign."""

    try:
        completed = core.subprocess.run(
            ("ps", "-eo", "pid=,ppid="),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, core.subprocess.SubprocessError) as exc:
        raise VllmK5GenerationError(f"cannot inspect process tree: {exc}") from exc
    children: dict[int, set[int]] = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            pid, parent = map(int, parts)
            children.setdefault(parent, set()).add(pid)
    allowed = {os.getpid()}
    frontier = [os.getpid()]
    while frontier:
        parent = frontier.pop()
        for child in children.get(parent, set()):
            if child not in allowed:
                allowed.add(child)
                frontier.append(child)
    return allowed


def _external_gpu_compute_processes(
    physical_gpu_index: int,
) -> list[dict[str, Any]]:
    index = _validate_physical_gpu_index(physical_gpu_index)
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
        raise VllmK5GenerationError(
            f"cannot inspect physical GPU{index}: {exc}"
        ) from exc
    processes: list[dict[str, Any]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line or "No running processes" in line or line.startswith("N/A"):
            continue
        columns = [value.strip() for value in line.split(",", 1)]
        if not columns or not columns[0].isdigit():
            raise VllmK5GenerationError(f"unexpected GPU{index} compute row: {line!r}")
        pid = int(columns[0])
        if pid == os.getpid():
            continue
        processes.append(
            {
                "physical_gpu_index": index,
                "pid": pid,
                "process_name": columns[1] if len(columns) == 2 else "",
            }
        )
    return processes


def _physical_gpu_identity(physical_gpu_index: int) -> dict[str, Any]:
    index = _validate_physical_gpu_index(physical_gpu_index)
    try:
        completed = subprocess.run(
            (
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=uuid,pci.bus_id",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise VllmK5GenerationError(
            f"cannot bind physical GPU{index} identity: {exc}"
        ) from exc
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise VllmK5GenerationError(f"physical GPU{index} identity is ambiguous")
    columns = [value.strip() for value in rows[0].split(",")]
    if len(columns) != 2 or not columns[0].startswith("GPU-"):
        raise VllmK5GenerationError(f"invalid physical GPU{index} identity row")
    return {
        "physical_gpu_index": index,
        "uuid": columns[0],
        "pci_bus_id": columns[1].lower(),
    }


def _physical_gpu_identities() -> list[dict[str, Any]]:
    return [_physical_gpu_identity(index) for index in ALLOWED_PHYSICAL_GPU_INDEXES]


def _verify_visible_cuda_devices(
    torch_module: Any, physical_identities: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 2:
        raise VllmK5GenerationError(
            "exactly two visible CUDA devices are required for DP=2"
        )
    observed: list[dict[str, Any]] = []
    for logical_index, expected in enumerate(physical_identities):
        logical_uuid = getattr(
            torch_module.cuda.get_device_properties(logical_index), "uuid", None
        )
        if core._normalise_gpu_uuid(logical_uuid) != core._normalise_gpu_uuid(
            expected["uuid"]
        ):
            raise VllmK5GenerationError(
                f"logical cuda:{logical_index} does not match physical "
                f"GPU{expected['physical_gpu_index']}"
            )
        observed.append(
            {
                "logical_cuda_index": logical_index,
                **dict(expected),
            }
        )
    return observed


@contextlib.contextmanager
def exclusive_dual_gpu_lease(
    *, timeout_seconds: int, poll_seconds: int, on_wait: Any = None
):
    """Hold both per-GPU locks and wait for two consecutive dual-idle polls."""

    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise VllmK5GenerationError("GPU wait timeout/poll must be positive")
    handles: list[Any] = []
    acquired: list[Any] = []
    started = time.monotonic()
    last_notice = 0.0
    try:
        for index in ALLOWED_PHYSICAL_GPU_INDEXES:
            path = gpu_lock_path(index).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_symlink():
                raise VllmK5GenerationError(f"GPU lock is a symlink: {path}")
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags, 0o600)
            handle = os.fdopen(descriptor, "a+", encoding="utf-8")
            handles.append(handle)
        # Acquire the pair atomically from the scheduler's perspective.  If
        # GPU1 is unavailable, immediately release GPU0 before retrying so a
        # single-GPU job is never starved behind a partial dual-card lease.
        while len(acquired) != len(handles):
            acquired.clear()
            for handle in handles:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired.append(handle)
                except BlockingIOError:
                    break
            if len(acquired) == len(handles):
                break
            for handle in reversed(acquired):
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            acquired.clear()
            elapsed = time.monotonic() - started
            if elapsed >= timeout_seconds:
                raise VllmK5GenerationError(
                    "timed out waiting for the physical GPU0/GPU1 lock pair"
                )
            if on_wait is not None and elapsed - last_notice >= 60:
                on_wait("waiting_for_dual_gpu_locks", [])
                last_notice = elapsed
            time.sleep(min(poll_seconds, max(0.1, timeout_seconds - elapsed)))
        consecutive_idle = 0
        while consecutive_idle < 2:
            processes = [
                process
                for index in ALLOWED_PHYSICAL_GPU_INDEXES
                for process in _external_gpu_compute_processes(index)
            ]
            elapsed = time.monotonic() - started
            if processes:
                consecutive_idle = 0
                if on_wait is not None and elapsed - last_notice >= 60:
                    on_wait("waiting_for_external_compute_pids", processes)
                    last_notice = elapsed
            else:
                consecutive_idle += 1
            if consecutive_idle < 2:
                if elapsed >= timeout_seconds:
                    raise VllmK5GenerationError(
                        "timed out waiting for physical GPU0 and GPU1 to become idle"
                    )
                time.sleep(min(poll_seconds, max(0.1, timeout_seconds - elapsed)))
        yield {
            "lock_paths": [
                str(gpu_lock_path(index).resolve())
                for index in ALLOWED_PHYSICAL_GPU_INDEXES
            ],
            "acquired_at_utc": core._utc_now(),
            "wait_seconds": round(time.monotonic() - started, 3),
            "physical_gpu_indexes": list(ALLOWED_PHYSICAL_GPU_INDEXES),
            "external_compute_pids_at_acquire": [],
        }
    finally:
        for handle in reversed(acquired):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        for handle in handles:
            handle.close()


def _assert_no_foreign_gpu_processes() -> None:
    allowed = _descendant_pids()
    foreign = [
        process
        for index in ALLOWED_PHYSICAL_GPU_INDEXES
        for process in _external_gpu_compute_processes(index)
        if int(process["pid"]) not in allowed
    ]
    if foreign:
        raise VllmK5GenerationError(
            f"foreign GPU0/GPU1 compute processes appeared during generation: {foreign}"
        )


def _engine_core_pids(engine: Any) -> list[int]:
    """Read the two V1 DP EngineCore children from the live async client."""

    engine_core = getattr(engine, "engine_core", None)
    core_engines = getattr(engine_core, "core_engines", None)
    if not isinstance(core_engines, list) or len(core_engines) != DATA_PARALLEL_SIZE:
        raise VllmK5GenerationError(
            "vLLM did not create exactly two DP EngineCore children"
        )
    pids: list[int] = []
    for rank, core_engine in enumerate(core_engines):
        if getattr(core_engine, "index", None) != rank:
            raise VllmK5GenerationError("vLLM DP EngineCore rank ordering drift")
        process = getattr(getattr(core_engine, "proc_handle", None), "proc", None)
        pid = getattr(process, "pid", None)
        if (
            not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 0
            or not bool(process.is_alive())
        ):
            raise VllmK5GenerationError(f"vLLM DP EngineCore rank {rank} is not alive")
        pids.append(pid)
    if len(set(pids)) != DATA_PARALLEL_SIZE:
        raise VllmK5GenerationError("vLLM DP EngineCore PIDs are not distinct")
    return pids


def _verify_dual_engine_gpu_processes(
    engine: Any, *, timeout_seconds: float = 60.0, poll_seconds: float = 0.5
) -> list[dict[str, Any]]:
    """Prove one distinct descendant EngineCore owns each physical GPU."""

    engine_core_pids = _engine_core_pids(engine)
    deadline = time.monotonic() + timeout_seconds
    last_observed: list[list[dict[str, Any]]] = [[], []]
    while True:
        allowed = _descendant_pids()
        if not set(engine_core_pids).issubset(allowed):
            raise VllmK5GenerationError("vLLM EngineCore escaped the process tree")
        per_gpu = [
            _external_gpu_compute_processes(index)
            for index in ALLOWED_PHYSICAL_GPU_INDEXES
        ]
        last_observed = per_gpu
        foreign = [
            process
            for processes in per_gpu
            for process in processes
            if int(process["pid"]) not in allowed
        ]
        if foreign:
            raise VllmK5GenerationError(
                "foreign GPU0/GPU1 compute processes appeared after model load: "
                f"{foreign}"
            )
        pid_sets = [
            {int(process["pid"]) for process in processes} for processes in per_gpu
        ]
        rank_mapping_ok = all(
            engine_core_pids[rank] in pid_sets[rank]
            and engine_core_pids[1 - rank] not in pid_sets[rank]
            for rank in range(DATA_PARALLEL_SIZE)
        )
        if rank_mapping_ok:
            return [
                {
                    "physical_gpu_index": index,
                    "data_parallel_rank": index,
                    "engine_core_pid": engine_core_pids[index],
                    "descendant_compute_processes": copy.deepcopy(per_gpu[index]),
                }
                for index in ALLOWED_PHYSICAL_GPU_INDEXES
            ]
        if time.monotonic() >= deadline:
            raise VllmK5GenerationError(
                "could not prove one distinct DP EngineCore PID per physical GPU; "
                f"engine_core_pids={engine_core_pids}, observed={last_observed}"
            )
        time.sleep(poll_seconds)


async def _consume_request(
    engine: Any, sampling_params_cls: Any, case: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Any]:
    params = sampling_params_cls(
        n=1,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        max_tokens=MAX_NEW_TOKENS,
        seed=int(case["row_seed"]),
        detokenize=True,
        # DeepSeek's </think> is token 128014 and must remain in the text.
        skip_special_tokens=False,
        spaces_between_special_tokens=True,
    )
    request_id = (
        f"case-{int(case['absolute_case_index']):05d}-seed-{int(case['row_seed'])}"
    )
    final = None
    async for output in engine.generate(
        {"prompt_token_ids": list(case["token_entry"]["prompt_token_ids"])},
        params,
        request_id,
    ):
        final = output
    if final is None or not getattr(final, "finished", True):
        raise VllmK5GenerationError(f"vLLM request did not finish: {request_id}")
    return case, final


async def _run_async_generation(
    *,
    anchor: Mapping[str, Any],
    tokenizer_path: Path,
    pending_chunks: Sequence[tuple[int, Sequence[Mapping[str, Any]]]],
    on_completed: Any,
    on_chunk_completed: Any,
    max_num_seqs: int,
    physical_gpu_identities: Sequence[Mapping[str, Any]],
    gpu_lease: Mapping[str, Any],
) -> dict[str, Any]:
    import torch
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    if vllm.__version__ != EXPECTED_VLLM_VERSION:
        raise VllmK5GenerationError(
            f"expected vLLM {EXPECTED_VLLM_VERSION}, got {vllm.__version__}"
        )
    max_num_seqs = _validate_max_num_seqs(max_num_seqs)
    visible_devices = _verify_visible_cuda_devices(torch, physical_gpu_identities)
    engine_args = AsyncEngineArgs(
        model=str(anchor["model_path"]),
        tokenizer=str(tokenizer_path),
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        quantization=None,
        load_format="safetensors",
        pipeline_parallel_size=PIPELINE_PARALLEL_SIZE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        data_parallel_size=DATA_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        swap_space=0,
        cpu_offload_gb=0,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        generation_config="vllm",
        seed=0,
        disable_log_requests=True,
        disable_log_stats=True,
    )
    engine_init_started = time.perf_counter()
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    try:
        parallel_config = engine.vllm_config.parallel_config
        model_config = engine.vllm_config.model_config
        compilation_config = engine.vllm_config.compilation_config
        if (
            parallel_config.data_parallel_size != DATA_PARALLEL_SIZE
            or parallel_config.tensor_parallel_size != TENSOR_PARALLEL_SIZE
            or parallel_config.pipeline_parallel_size != PIPELINE_PARALLEL_SIZE
            or parallel_config.data_parallel_master_ip != DATA_PARALLEL_MASTER_IP
            or not isinstance(parallel_config.data_parallel_master_port, int)
            or isinstance(parallel_config.data_parallel_master_port, bool)
            or parallel_config.data_parallel_master_port <= 0
            or model_config.enforce_eager is not True
            or compilation_config.use_cudagraph is not False
            or model_config.use_async_output_proc is not False
        ):
            raise VllmK5GenerationError("vLLM DP=2 engine topology drift")
        engine_core_gpu_bindings = _verify_dual_engine_gpu_processes(engine)
        model_load_wall_seconds = time.perf_counter() - engine_init_started
        if model_load_wall_seconds <= 0:
            raise VllmK5GenerationError("non-positive vLLM model-load timing")
        runtime = {
            **_runtime_versions(),
            "visible_cuda_devices": visible_devices,
            "data_parallel_master_ip": parallel_config.data_parallel_master_ip,
            "data_parallel_master_port": parallel_config.data_parallel_master_port,
            "data_parallel_master_port_policy": DATA_PARALLEL_MASTER_PORT_POLICY,
            "distributed_world_backend": "nccl",
            "data_parallel_control_backend": "gloo_cpu",
            "dense_model_per_layer_tensor_collectives": False,
            "data_parallel_request_routing": "least_inflight_requests",
            "engine_core_gpu_bindings": engine_core_gpu_bindings,
            "enforce_eager": True,
            "cuda_graphs": False,
            "async_output_processing": False,
            "gpu_lease": copy.deepcopy(dict(gpu_lease)),
            "model_load_wall_seconds": model_load_wall_seconds,
            "generation_wall_seconds": None,
            "generated_output_tokens": 0,
            "completed_requests_in_timed_session": 0,
            "output_tokens_per_second": None,
            "timing_scope": "current_engine_session_after_dp_pid_validation",
            "preemption_count": None,
            "preemption_count_status": "unavailable_vllm_async_public_api",
            "engine": sampling_contract(max_num_seqs=max_num_seqs),
        }
    except BaseException:
        if hasattr(engine, "shutdown"):
            engine.shutdown()
        elif hasattr(engine, "shutdown_background_loop"):
            engine.shutdown_background_loop()
        raise
    try:
        tokenizer = await engine.get_tokenizer()
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if not isinstance(eos_id, int) or isinstance(eos_id, bool):
            raise VllmK5GenerationError("vLLM FOMC tokenizer has no scalar EOS")
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_id
        generation_started = time.perf_counter()
        timed_output_tokens = 0
        timed_completed_requests = 0

        def update_timing() -> None:
            elapsed = time.perf_counter() - generation_started
            runtime["generation_wall_seconds"] = elapsed
            runtime["generated_output_tokens"] = timed_output_tokens
            runtime["completed_requests_in_timed_session"] = timed_completed_requests
            runtime["output_tokens_per_second"] = (
                timed_output_tokens / elapsed if elapsed > 0 else None
            )

        for chunk_id, chunk_cases in pending_chunks:
            _assert_no_foreign_gpu_processes()
            tasks = [
                asyncio.create_task(_consume_request(engine, SamplingParams, case))
                for case in chunk_cases
            ]
            try:
                for completed in asyncio.as_completed(tasks):
                    case, output = await completed
                    if (
                        list(getattr(output, "prompt_token_ids", []) or [])
                        != list(case["token_entry"]["prompt_token_ids"])
                        or len(getattr(output, "outputs", []) or []) != 1
                    ):
                        raise VllmK5GenerationError(
                            "vLLM consumed a different prompt-token ledger row"
                        )
                    completion = output.outputs[0]
                    token_ids = list(completion.token_ids)
                    try:
                        text = core.native_eval.native_probe.common_probe.decode_completion_preserving_boundary(
                            tokenizer, token_ids, [eos_id]
                        )
                    except core.native_eval.native_probe.common_probe.ProbeError as exc:
                        raise VllmK5GenerationError(str(exc)) from exc
                    if completion.text != text:
                        raise VllmK5GenerationError(
                            "vLLM text differs from exact token-ledger tokenizer decode"
                        )
                    timed_output_tokens += len(token_ids)
                    timed_completed_requests += 1
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
        _assert_no_foreign_gpu_processes()
        return runtime
    finally:
        if hasattr(engine, "shutdown"):
            engine.shutdown()
        elif hasattr(engine, "shutdown_background_loop"):
            engine.shutdown_background_loop()


def run_model(
    *,
    model_id: str,
    cohort_path: Path,
    cohort_sha256: str,
    output_dir: Path,
    resume: bool,
    smoke: bool,
    gpu_wait_timeout_seconds: int,
    gpu_poll_seconds: int,
    max_num_seqs: int,
) -> dict[str, Any]:
    """Run one model with completion-order durability and absolute chunks."""

    if model_id not in MODEL_ORDER:
        raise VllmK5GenerationError(f"unsupported model ID: {model_id}")
    max_num_seqs = _validate_max_num_seqs(max_num_seqs)
    _assert_required_environment(require_cuda=True)
    physical_gpu_identities = _physical_gpu_identities()
    cohort, ledger, sample_manifest, bound_rows = load_cohort(
        cohort_path, cohort_sha256
    )
    source_sample_sha = str(cohort["source_sample_manifest"]["sha256"])
    source_hashes = _source_hashes(
        cohort_path,
        cohort_sha256,
        cohort,
        sample_manifest,
        max_num_seqs,
    )
    anchor = core.load_frozen_anchor(model_id)
    _validate_bf16_anchor(anchor)
    model_label = str(anchor["model_label"])
    cases = canonical_cases(
        samples=sample_manifest["samples"], ledger=ledger, smoke=smoke
    )
    expected_cases = len(cases)

    unresolved_output = output_dir.expanduser()
    if unresolved_output.is_symlink():
        raise VllmK5GenerationError("output directory is a symlink")
    output_dir = unresolved_output.resolve()
    partial_dir = output_dir / ".partial"
    wal_path = partial_dir / "generations.completion_order.progress.v1.jsonl"
    receipt_path = partial_dir / "absolute_chunks.progress.v1.jsonl"
    canonical_path = output_dir / "generations.canonical.v1.jsonl"
    launch_path = output_dir / "launch.json"
    state_path = output_dir / "state.progress.v1.json"
    manifest_path = output_dir / "manifest.json"
    scope = "infrastructure_smoke" if smoke else "formal_merged_panel"
    launch_payload = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "status": "initializing",
        "created_at_utc": core._utc_now(),
        "evaluation_id": EVALUATION_ID,
        "task_contract_id": core.TASK_CONTRACT_ID,
        "evaluation_scope": scope,
        "model_id": model_id,
        "model_label": model_label,
        "model": anchor["model"],
        "adapter": None,
        "sealed_anchor_manifest": anchor["anchor"],
        "cohort": {"path": str(cohort_path.resolve()), "sha256": cohort_sha256},
        "source_artifact_sha256s": source_hashes,
        "generation": {
            **sampling_contract(max_num_seqs=max_num_seqs),
            "expected_cases": expected_cases,
        },
        "runtime": {
            **_runtime_versions(),
            "cuda_visible_devices": "0,1",
            "cuda_device_order": "PCI_BUS_ID",
            "physical_gpu_indexes": list(ALLOWED_PHYSICAL_GPU_INDEXES),
            "physical_gpu_identities": physical_gpu_identities,
            "gpu_lock_paths": [
                str(gpu_lock_path(index).resolve())
                for index in ALLOWED_PHYSICAL_GPU_INDEXES
            ],
            "max_num_seqs": max_num_seqs,
        },
        "persistence": {
            "completion_order_wal": True,
            "append_flush_fsync_per_completed_request": True,
            "absolute_chunk_crash_replay": True,
            "full_generated_text": True,
            "full_answer": True,
            "full_generated_token_ids": True,
            "full_prompt_token_ids_in_sealed_ledger": True,
            "source_text_and_hashes_per_row": True,
            "canonical_final_sort": "absolute_case_index",
        },
    }
    compare_keys = tuple(key for key in launch_payload if key != "created_at_utc")
    resume_count = 0
    engine_runtime: Mapping[str, Any] | None = None
    if resume:
        if manifest_path.exists():
            raise VllmK5GenerationError("refusing to resume a sealed run")
        if not launch_path.is_file() or not state_path.is_file():
            raise VllmK5GenerationError("resume launch/state is missing")
        prior_launch = core._read_json(launch_path)
        validate_manifest_integrity(prior_launch)
        for key in compare_keys:
            if prior_launch.get(key) != launch_payload.get(key):
                raise VllmK5GenerationError(f"resume launch drift at {key}")
        prior_state = core._read_json(state_path)
        validate_manifest_integrity(prior_state)
        if (
            prior_state.get("schema_version") != STATE_SCHEMA
            or prior_state.get("evaluation_id") != EVALUATION_ID
            or prior_state.get("model_id") != model_id
            or prior_state.get("expected_cases") != expected_cases
        ):
            raise VllmK5GenerationError("resume state contract drift")
        recorded = prior_state.get("completed_cases")
        if not isinstance(recorded, int) or isinstance(recorded, bool):
            raise VllmK5GenerationError("resume state count is invalid")
        rows, wal_tracker, prefix = _read_canonical_jsonl(
            wal_path, recorded_prefix_rows=recorded
        )
        if len(rows) not in {recorded, recorded + 1}:
            raise VllmK5GenerationError("WAL/state differs by more than one row")
        expected_binding = prior_state.get("completion_wal")
        observed_binding = (
            wal_tracker.binding(wal_path) if len(rows) == recorded else prefix
        )
        if expected_binding != observed_binding:
            raise VllmK5GenerationError("resume WAL prefix binding drift")
        receipts, receipt_tracker, _ = _read_canonical_jsonl(receipt_path)
        resume_count = int(prior_state.get("resume_count", 0)) + 1
        engine_runtime = prior_state.get("engine_runtime")
    else:
        if output_dir.exists():
            raise VllmK5GenerationError(f"output already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        partial_dir.mkdir()
        core._fsync_directory(output_dir.parent)
        core._write_new_json(launch_path, seal_manifest(launch_payload))
        for path in (wal_path, receipt_path):
            with path.open("x", encoding="utf-8") as handle:
                handle.flush()
                os.fsync(handle.fileno())
        core._fsync_directory(partial_dir)
        rows = []
        receipts = []
        wal_tracker = WalTracker.empty()
        receipt_tracker = WalTracker.empty()
        _write_state(
            state_path,
            _state_payload(
                status="initializing",
                model_id=model_id,
                expected_cases=expected_cases,
                resume_count=resume_count,
                wal_binding=wal_tracker.binding(wal_path),
                receipt_binding=receipt_tracker.binding(receipt_path),
                completed_indexes=set(),
            ),
        )

    rows_by_index = validate_wal_rows(
        rows,
        cases=cases,
        model_id=model_id,
        model_label=model_label,
        bound_rows=bound_rows,
        sample_manifest_sha256=source_sample_sha,
        source_artifact_sha256s=source_hashes,
        max_num_seqs=max_num_seqs,
    )
    _validate_receipts(
        receipts,
        model_id=model_id,
        rows_by_index=rows_by_index,
        expected_cases=expected_cases,
    )
    total_chunks = 1 if smoke else CHUNKS_PER_MODEL
    # A chunk is committed only by its receipt.  If all request rows reached
    # the WAL but the process died before the receipt, replay the entire fixed
    # window and exact-compare every persisted row before committing it.
    start_chunk = len(receipts)
    pending_chunks = [
        (
            chunk_id,
            cases[
                chunk_id * ABSOLUTE_CHUNK_SIZE : min(
                    (chunk_id + 1) * ABSOLUTE_CHUNK_SIZE, expected_cases
                )
            ],
        )
        for chunk_id in range(start_chunk, total_chunks)
    ]

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
        )
        prior = rows_by_index.get(index)
        if prior is not None:
            if prior != result:
                raise VllmK5GenerationError(
                    f"crash replay changed persisted case {index}"
                )
        else:
            with wal_path.open("a", encoding="utf-8") as handle:
                _append_fsync(handle, wal_tracker, result)
            rows.append(result)
            rows_by_index[index] = result
            _write_state(
                state_path,
                _state_payload(
                    status="generating",
                    model_id=model_id,
                    expected_cases=expected_cases,
                    resume_count=resume_count,
                    wal_binding=wal_tracker.binding(wal_path),
                    receipt_binding=receipt_tracker.binding(receipt_path),
                    completed_indexes=set(rows_by_index),
                    engine_runtime=engine_runtime,
                    last_completed_request={
                        "absolute_case_index": index,
                        "absolute_chunk_id": chunk_id,
                        "sample_id": result["sample_id"],
                        "replicate_id": result["replicate_id"],
                    },
                ),
            )

    async def on_chunk_completed(chunk_id: int, runtime: Mapping[str, Any]) -> None:
        nonlocal engine_runtime
        engine_runtime = runtime
        if len(receipts) != chunk_id:
            raise VllmK5GenerationError(
                "absolute chunk receipts are not a contiguous prefix"
            )
        receipt = _chunk_receipt(
            model_id=model_id,
            chunk_id=chunk_id,
            rows_by_index=rows_by_index,
            expected_cases=expected_cases,
        )
        with receipt_path.open("a", encoding="utf-8") as handle:
            _append_fsync(handle, receipt_tracker, receipt)
        receipts.append(receipt)
        _write_state(
            state_path,
            _state_payload(
                status="chunk_complete",
                model_id=model_id,
                expected_cases=expected_cases,
                resume_count=resume_count,
                wal_binding=wal_tracker.binding(wal_path),
                receipt_binding=receipt_tracker.binding(receipt_path),
                completed_indexes=set(rows_by_index),
                engine_runtime=engine_runtime,
                last_completed_chunk=chunk_id,
            ),
        )

    if pending_chunks:

        def on_wait(reason: str, processes: Sequence[Mapping[str, Any]]) -> None:
            print(
                _canonical(
                    {"status": reason, "model_id": model_id, "processes": processes}
                ),
                file=sys.stderr,
                flush=True,
            )

        with exclusive_dual_gpu_lease(
            timeout_seconds=gpu_wait_timeout_seconds,
            poll_seconds=gpu_poll_seconds,
            on_wait=on_wait,
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
                        physical_gpu_identities=physical_gpu_identities,
                        gpu_lease=lease,
                    )
                )
            except BaseException as exc:
                _write_state(
                    state_path,
                    _state_payload(
                        status="failed",
                        model_id=model_id,
                        expected_cases=expected_cases,
                        resume_count=resume_count,
                        wal_binding=wal_tracker.binding(wal_path),
                        receipt_binding=receipt_tracker.binding(receipt_path),
                        completed_indexes=set(rows_by_index),
                        engine_runtime=engine_runtime,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        failed_at_utc=core._utc_now(),
                    ),
                )
                raise
            # Receipt every newly complete absolute chunk after request WAL fsyncs.
            full_chunks, _ = _completion_status(
                set(rows_by_index), expected_cases=expected_cases
            )
            with receipt_path.open("a", encoding="utf-8") as handle:
                while len(receipts) < full_chunks:
                    receipt = _chunk_receipt(
                        model_id=model_id,
                        chunk_id=len(receipts),
                        rows_by_index=rows_by_index,
                        expected_cases=expected_cases,
                    )
                    _append_fsync(handle, receipt_tracker, receipt)
                    receipts.append(receipt)
    if len(rows_by_index) != expected_cases:
        raise VllmK5GenerationError("generation ended before all cases completed")
    if not isinstance(engine_runtime, Mapping):
        raise VllmK5GenerationError("generation has no sealed engine runtime")
    engine_runtime = copy.deepcopy(dict(engine_runtime))
    timed_session_tokens = engine_runtime.get("generated_output_tokens")
    canonical_generated_output_tokens = sum(
        len(row["generated_token_ids"]) for row in rows_by_index.values()
    )
    generation_wall_seconds = engine_runtime.get("generation_wall_seconds")
    if (
        not isinstance(timed_session_tokens, int)
        or isinstance(timed_session_tokens, bool)
        or timed_session_tokens <= 0
        or not isinstance(generation_wall_seconds, (int, float))
        or isinstance(generation_wall_seconds, bool)
        or float(generation_wall_seconds) <= 0
    ):
        raise VllmK5GenerationError("generation timing evidence is incomplete")
    engine_runtime["timed_session_generated_output_tokens"] = timed_session_tokens
    engine_runtime["generated_output_tokens"] = canonical_generated_output_tokens
    engine_runtime["output_tokens_per_second"] = (
        canonical_generated_output_tokens / float(generation_wall_seconds)
    )
    engine_runtime["speed_measurement_valid_for_candidate_selection"] = bool(
        resume_count == 0
        and engine_runtime.get("completed_requests_in_timed_session") == expected_cases
        and timed_session_tokens == canonical_generated_output_tokens
    )
    _validate_receipts(
        receipts,
        model_id=model_id,
        rows_by_index=rows_by_index,
        expected_cases=expected_cases,
    )
    # Normal completion can leave all receipts unwritten until the engine exits.
    with receipt_path.open("a", encoding="utf-8") as handle:
        while len(receipts) < total_chunks:
            receipt = _chunk_receipt(
                model_id=model_id,
                chunk_id=len(receipts),
                rows_by_index=rows_by_index,
                expected_cases=expected_cases,
            )
            _append_fsync(handle, receipt_tracker, receipt)
            receipts.append(receipt)
    canonical_rows = [rows_by_index[index] for index in range(expected_cases)]
    if canonical_path.exists() or canonical_path.is_symlink():
        existing, _, _ = _read_canonical_jsonl(canonical_path)
        if existing != canonical_rows:
            raise VllmK5GenerationError("existing canonical output drift")
    else:
        with canonical_path.open("x", encoding="utf-8") as handle:
            for row in canonical_rows:
                handle.write(_canonical(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        core._fsync_directory(output_dir)
    summary = _summary(rows_by_index, smoke=smoke)
    _write_state(
        state_path,
        _state_payload(
            status="generation_complete",
            model_id=model_id,
            expected_cases=expected_cases,
            resume_count=resume_count,
            wal_binding=wal_tracker.binding(wal_path),
            receipt_binding=receipt_tracker.binding(receipt_path),
            completed_indexes=set(rows_by_index),
            engine_runtime=engine_runtime,
            canonical_output=_file_binding(canonical_path, rows=expected_cases),
            summary=summary,
        ),
    )
    frozen_state = core._read_json(state_path)
    state_payload_sha = validate_manifest_integrity(frozen_state)
    run_payload = {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "status": "complete",
        "created_at_utc": core._utc_now(),
        "evaluation_id": EVALUATION_ID,
        "task_contract_id": core.TASK_CONTRACT_ID,
        "evaluation_scope": scope,
        "model_id": model_id,
        "model_label": model_label,
        "model": anchor["model"],
        "adapter": None,
        "sealed_anchor_manifest": anchor["anchor"],
        "cohort": launch_payload["cohort"],
        "source_artifact_sha256s": source_hashes,
        "generation_contract": {
            **sampling_contract(max_num_seqs=max_num_seqs),
            "expected_cases": expected_cases,
            "expected_absolute_chunks": total_chunks,
        },
        "runtime": {
            "resume_count": resume_count,
            "engine_runtime": copy.deepcopy(engine_runtime),
        },
        "persistence": launch_payload["persistence"],
        "artifacts": {
            "launch": _file_binding(
                launch_path,
                rows=None,
            ),
            "completion_order_wal": _file_binding(wal_path, rows=expected_cases),
            "absolute_chunk_receipts": _file_binding(receipt_path, rows=total_chunks),
            "canonical_generations": _file_binding(canonical_path, rows=expected_cases),
            "generation_complete_state": {
                **_file_binding(state_path),
                "payload_sha256": state_payload_sha,
            },
        },
        "summary": summary,
        "limitations": _limitations(smoke=smoke),
    }
    sealed = seal_manifest(run_payload)
    core._write_new_json(manifest_path, sealed)
    return sealed


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
        raise VllmK5GenerationError("run manifest is a symlink")
    manifest_path = unresolved.resolve()
    manifest = core._read_json(manifest_path)
    payload_sha = validate_manifest_integrity(manifest)
    if (
        manifest.get("schema_version") != RUN_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("evaluation_id") != EVALUATION_ID
        or manifest.get("model_id") != expected_model_id
        or manifest.get("evaluation_scope") != expected_scope
    ):
        raise VllmK5GenerationError("sealed run identity drift")
    cohort, ledger, samples_manifest, bound_rows = load_cohort(
        cohort_path, cohort_sha256
    )
    smoke = expected_scope == "infrastructure_smoke"
    cases = canonical_cases(
        samples=samples_manifest["samples"], ledger=ledger, smoke=smoke
    )
    anchor = core.load_frozen_anchor(expected_model_id)
    _validate_bf16_anchor(anchor)
    source_hashes = _source_hashes(
        cohort_path,
        cohort_sha256,
        cohort,
        samples_manifest,
        max_num_seqs,
    )
    if (
        manifest.get("model") != anchor["model"]
        or manifest.get("sealed_anchor_manifest") != anchor["anchor"]
        or manifest.get("source_artifact_sha256s") != source_hashes
        or manifest.get("limitations") != _limitations(smoke=smoke)
        or manifest.get("generation_contract")
        != {
            **sampling_contract(max_num_seqs=max_num_seqs),
            "expected_cases": len(cases),
            "expected_absolute_chunks": 1 if smoke else CHUNKS_PER_MODEL,
        }
    ):
        raise VllmK5GenerationError("sealed run model/source/generation drift")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise VllmK5GenerationError("sealed run artifact bindings are missing")
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
        launch.get("schema_version") != RUN_MANIFEST_SCHEMA
        or launch.get("evaluation_id") != EVALUATION_ID
        or launch.get("evaluation_scope") != expected_scope
        or launch.get("model_id") != expected_model_id
        or launch.get("model") != manifest.get("model")
        or launch.get("cohort") != manifest.get("cohort")
        or launch.get("source_artifact_sha256s") != source_hashes
        or launch.get("generation")
        != {
            **sampling_contract(max_num_seqs=max_num_seqs),
            "expected_cases": len(cases),
        }
        or launch.get("persistence") != manifest.get("persistence")
    ):
        raise VllmK5GenerationError("sealed launch contract drift")
    manifest_runtime = manifest.get("runtime")
    if not isinstance(manifest_runtime, Mapping):
        raise VllmK5GenerationError("sealed run runtime is missing")
    resume_count = manifest_runtime.get("resume_count")
    if (
        not isinstance(resume_count, int)
        or isinstance(resume_count, bool)
        or resume_count < 0
    ):
        raise VllmK5GenerationError("sealed run resume count drift")
    engine_runtime = manifest_runtime.get("engine_runtime")
    wal_rows, wal_tracker, _ = _read_canonical_jsonl(wal_path)
    rows_by_index = validate_wal_rows(
        wal_rows,
        cases=cases,
        model_id=expected_model_id,
        model_label=str(anchor["model_label"]),
        bound_rows=bound_rows,
        sample_manifest_sha256=str(cohort["source_sample_manifest"]["sha256"]),
        source_artifact_sha256s=source_hashes,
        max_num_seqs=max_num_seqs,
    )
    _validate_sealed_runtime(
        launch_runtime=launch.get("runtime"),
        engine_runtime=engine_runtime,
        max_num_seqs=max_num_seqs,
        expected_cases=len(cases),
        expected_generated_output_tokens=sum(
            len(row["generated_token_ids"]) for row in rows_by_index.values()
        ),
        resume_count=resume_count,
    )
    canonical_rows, _, _ = _read_canonical_jsonl(canonical_path)
    if canonical_rows != [rows_by_index[index] for index in range(len(cases))]:
        raise VllmK5GenerationError("canonical generation sort/binding drift")
    receipts, receipt_tracker, _ = _read_canonical_jsonl(receipt_path)
    _validate_receipts(
        receipts,
        model_id=expected_model_id,
        rows_by_index=rows_by_index,
        expected_cases=len(cases),
    )
    if len(receipts) != (1 if smoke else CHUNKS_PER_MODEL):
        raise VllmK5GenerationError("sealed run chunk receipt closure drift")
    state = core._read_json(state_path)
    state_payload_sha = validate_manifest_integrity(state)
    if (
        state.get("schema_version") != STATE_SCHEMA
        or state.get("status") != "generation_complete"
        or state.get("evaluation_id") != EVALUATION_ID
        or state.get("model_id") != expected_model_id
        or state.get("expected_cases") != len(cases)
        or state.get("completed_cases") != len(cases)
        or state.get("completion_wal") != wal_tracker.binding(wal_path)
        or state.get("chunk_receipts") != receipt_tracker.binding(receipt_path)
        or state.get("canonical_output")
        != _file_binding(canonical_path, rows=len(cases))
        or state.get("engine_runtime") != engine_runtime
        or state.get("resume_count") != resume_count
        or artifacts["generation_complete_state"].get("payload_sha256")
        != state_payload_sha
        or manifest.get("summary") != _summary(rows_by_index, smoke=smoke)
    ):
        raise VllmK5GenerationError("sealed run state/summary drift")
    return {
        "manifest": manifest,
        "manifest_binding": {
            **_file_binding(manifest_path),
            "payload_sha256": payload_sha,
        },
        "results": [rows_by_index[index] for index in range(len(cases))],
    }


def seal_suite(
    *,
    cohort_path: Path,
    cohort_sha256: str,
    output_dir: Path,
    scope: str,
    max_num_seqs: int,
) -> dict[str, Any]:
    max_num_seqs = _validate_max_num_seqs(max_num_seqs)
    if scope not in {"infrastructure_smoke", "formal_merged_panel"}:
        raise VllmK5GenerationError("invalid suite scope")
    output_dir = output_dir.expanduser().resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise VllmK5GenerationError("suite is already sealed")
    runs = {
        model_id: load_and_validate_run(
            output_dir / model_id / "manifest.json",
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            expected_model_id=model_id,
            expected_scope=scope,
            max_num_seqs=max_num_seqs,
        )
        for model_id in MODEL_ORDER
    }
    expected_per_model = (
        SMOKE_CASES_PER_MODEL if scope == "infrastructure_smoke" else CASES_PER_MODEL
    )
    payload = seal_manifest(
        {
            "schema_version": SUITE_MANIFEST_SCHEMA,
            "status": "complete",
            "created_at_utc": core._utc_now(),
            "evaluation_id": EVALUATION_ID,
            "task_contract_id": core.TASK_CONTRACT_ID,
            "evaluation_scope": scope,
            "model_order": list(MODEL_ORDER),
            "cohort": {"path": str(cohort_path.resolve()), "sha256": cohort_sha256},
            "generation_contract": sampling_contract(max_num_seqs=max_num_seqs),
            "coverage": {
                "models": len(MODEL_ORDER),
                "rows_per_model": expected_per_model,
                "total_rows": expected_per_model * len(MODEL_ORDER),
            },
            "run_manifests": {
                model_id: runs[model_id]["manifest_binding"] for model_id in MODEL_ORDER
            },
        }
    )
    core._write_new_json(manifest_path, payload)
    return payload


def load_and_validate_suite(
    manifest_path: Path,
    *,
    cohort_path: Path,
    cohort_sha256: str,
    expected_scope: str,
    max_num_seqs: int,
) -> dict[str, Any]:
    max_num_seqs = _validate_max_num_seqs(max_num_seqs)
    manifest_path = manifest_path.expanduser().resolve()
    suite = core._read_json(manifest_path)
    payload_sha = validate_manifest_integrity(suite)
    expected_per_model = (
        SMOKE_CASES_PER_MODEL
        if expected_scope == "infrastructure_smoke"
        else CASES_PER_MODEL
    )
    if (
        suite.get("schema_version") != SUITE_MANIFEST_SCHEMA
        or suite.get("status") != "complete"
        or suite.get("evaluation_id") != EVALUATION_ID
        or suite.get("evaluation_scope") != expected_scope
        or suite.get("model_order") != list(MODEL_ORDER)
        or suite.get("generation_contract")
        != sampling_contract(max_num_seqs=max_num_seqs)
        or suite.get("coverage")
        != {
            "models": len(MODEL_ORDER),
            "rows_per_model": expected_per_model,
            "total_rows": expected_per_model * len(MODEL_ORDER),
        }
    ):
        raise VllmK5GenerationError("sealed suite contract drift")
    runs = {
        model_id: load_and_validate_run(
            manifest_path.parent / model_id / "manifest.json",
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            expected_model_id=model_id,
            expected_scope=expected_scope,
            max_num_seqs=max_num_seqs,
        )
        for model_id in MODEL_ORDER
    }
    if suite.get("run_manifests") != {
        model_id: runs[model_id]["manifest_binding"] for model_id in MODEL_ORDER
    }:
        raise VllmK5GenerationError("suite child manifest bindings drift")
    return {
        "manifest": suite,
        "manifest_binding": {
            **_file_binding(manifest_path),
            "payload_sha256": payload_sha,
        },
        "runs": runs,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-model")
    run.add_argument("--model-id", choices=MODEL_ORDER, required=True)
    run.add_argument("--cohort", required=True, type=Path)
    run.add_argument("--cohort-sha256", required=True)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--gpu-wait-timeout-seconds", type=int, default=172800)
    run.add_argument("--gpu-poll-seconds", type=int, default=30)
    run.add_argument(
        "--max-num-seqs",
        type=int,
        choices=ALLOWED_MAX_NUM_SEQS,
        required=True,
    )
    validate_run_parser = sub.add_parser("validate-run")
    validate_run_parser.add_argument("--manifest", required=True, type=Path)
    validate_run_parser.add_argument("--cohort", required=True, type=Path)
    validate_run_parser.add_argument("--cohort-sha256", required=True)
    validate_run_parser.add_argument("--model-id", choices=MODEL_ORDER, required=True)
    validate_run_parser.add_argument("--scope", required=True)
    validate_run_parser.add_argument(
        "--max-num-seqs",
        type=int,
        choices=ALLOWED_MAX_NUM_SEQS,
        required=True,
    )
    seal = sub.add_parser("seal-suite")
    seal.add_argument("--cohort", required=True, type=Path)
    seal.add_argument("--cohort-sha256", required=True)
    seal.add_argument("--output-dir", required=True, type=Path)
    seal.add_argument("--scope", required=True)
    seal.add_argument(
        "--max-num-seqs",
        type=int,
        choices=ALLOWED_MAX_NUM_SEQS,
        required=True,
    )
    validate_suite_parser = sub.add_parser("validate-suite")
    validate_suite_parser.add_argument("--manifest", required=True, type=Path)
    validate_suite_parser.add_argument("--cohort", required=True, type=Path)
    validate_suite_parser.add_argument("--cohort-sha256", required=True)
    validate_suite_parser.add_argument("--scope", required=True)
    validate_suite_parser.add_argument(
        "--max-num-seqs",
        type=int,
        choices=ALLOWED_MAX_NUM_SEQS,
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run-model":
            result = run_model(
                model_id=args.model_id,
                cohort_path=args.cohort,
                cohort_sha256=args.cohort_sha256,
                output_dir=args.output_dir,
                resume=args.resume,
                smoke=args.smoke,
                gpu_wait_timeout_seconds=args.gpu_wait_timeout_seconds,
                gpu_poll_seconds=args.gpu_poll_seconds,
                max_num_seqs=args.max_num_seqs,
            )
        elif args.command == "validate-run":
            result = load_and_validate_run(
                args.manifest,
                cohort_path=args.cohort,
                cohort_sha256=args.cohort_sha256,
                expected_model_id=args.model_id,
                expected_scope=args.scope,
                max_num_seqs=args.max_num_seqs,
            )["manifest"]
        elif args.command == "seal-suite":
            result = seal_suite(
                cohort_path=args.cohort,
                cohort_sha256=args.cohort_sha256,
                output_dir=args.output_dir,
                scope=args.scope,
                max_num_seqs=args.max_num_seqs,
            )
        else:
            result = load_and_validate_suite(
                args.manifest,
                cohort_path=args.cohort,
                cohort_sha256=args.cohort_sha256,
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
        VllmK5GenerationError,
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
    "EVALUATION_ID",
    "MODEL_ORDER",
    "REPLICATE_SEEDS",
    "ROW_SCHEMA",
    "VllmK5GenerationError",
    "WalTracker",
    "build_result",
    "canonical_cases",
    "load_and_validate_run",
    "load_and_validate_suite",
    "load_cohort",
    "run_model",
    "sampling_contract",
    "seal_suite",
    "validate_result",
    "validate_wal_rows",
]
