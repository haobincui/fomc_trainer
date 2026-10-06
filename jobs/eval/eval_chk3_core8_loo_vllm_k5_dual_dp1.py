"""Run CHK3 cp318 Core8 LOO K=5 on two independent vLLM DP=1 workers.

This module is a new, versioned LOO profile over the deeply-tested dual-DP1
runtime.  It intentionally leaves every historical runner and output root
unchanged.  The profile replaces all beta-panel data, row, seed, sharding,
authorization, and policy hooks before delegating process/GPU/barrier/WAL
mechanics to the shared implementation.

Five fresh stochastic replicates are generated for every one of the 2,176
LOO prompts.  A paired block is one ``(meeting, replicate)`` combination, so
all seventeen arms in that block share both the row seed and physical GPU.
The historical greedy K=1 output is source evidence only and is never reused
as a stochastic replicate.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.eval import (
    eval_chk3_beta_core8_merged_vllm_k5_dual_dp1_stochastic_schedule_v2 as shared,
)
from jobs.eval import eval_chk3_core8_loo_stochastic_k10 as k10
from jobs.eval import eval_chk3_native_three_model as native_eval
from jobs.eval import eval_chk3_stochastic_bootstrap_generation as stochastic
from jobs.eval import prepare_chk3_core8_loo_vllm_k5 as preparation
from open_r1.validator.loo_generation_spec import (
    derive_row_seed,
    validate_manifest_integrity,
)
from open_r1.validator import loo_generation_spec as loo_spec


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ID = "chk3-cp318-core8-loo-n128-vllm-k5-dual-dp1-v1"
MODEL_ORDER = ("chk3",)
MODEL_LABELS = {"chk3": "chk3-cp318-exact-merged"}
REPLICATE_SEEDS = (
    20260811,
    21260811,
    22260811,
    23260811,
    24260811,
)
EXPECTED_PROMPTS = 2176
EXPECTED_MEETINGS = 128
VARIANTS_PER_MEETING = 17
REPLICATES = len(REPLICATE_SEEDS)
CASES_PER_MEETING = VARIANTS_PER_MEETING * REPLICATES
CASES_PER_MODEL = EXPECTED_PROMPTS * REPLICATES
ABSOLUTE_CHUNK_SIZE = CASES_PER_MEETING * 2
CHUNKS_PER_MODEL = EXPECTED_MEETINGS // 2
SHARD_COUNT = 2
CASES_PER_SHARD_CHUNK = CASES_PER_MEETING
FORMAL_CASES_PER_SHARD = CASES_PER_MODEL // SHARD_COUNT
SMOKE_MEETINGS = 2
SMOKE_PROMPTS = SMOKE_MEETINGS * VARIANTS_PER_MEETING
SMOKE_CASES_PER_MODEL = SMOKE_PROMPTS * REPLICATES
SMOKE_CASES_PER_SHARD = SMOKE_CASES_PER_MODEL // SHARD_COUNT

ROW_SCHEMA = "chk3-core8-loo-vllm-k5-dual-dp1-generation-row-v1"
SHARD_MANIFEST_SCHEMA = "chk3-core8-loo-vllm-k5-dual-dp1-shard-manifest-v1"
MERGED_MANIFEST_SCHEMA = "chk3-core8-loo-vllm-k5-dual-dp1-merged-manifest-v1"
STATE_SCHEMA = "chk3-core8-loo-vllm-k5-dual-dp1-state-v1"
CHUNK_RECEIPT_SCHEMA = "chk3-core8-loo-vllm-k5-dual-dp1-chunk-receipt-v1"
READY_SCHEMA = "chk3-core8-loo-vllm-k5-dual-dp1-ready-v1"
GO_SCHEMA = "chk3-core8-loo-vllm-k5-dual-dp1-go-v1"
DONE_SCHEMA = "chk3-core8-loo-vllm-k5-dual-dp1-done-v1"
ORCHESTRATOR_TIMING_SCHEMA = "chk3-core8-loo-vllm-k5-dual-dp1-timing-v1"

TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 50
REPETITION_PENALTY = 1.0
MAX_NEW_TOKENS = 2560
TAIL_TOKENS = 1024
MAX_MODEL_LEN = 4096
SELECTED_MAX_NUM_SEQS = 16
ALLOWED_MAX_NUM_SEQS = (SELECTED_MAX_NUM_SEQS,)
MAX_NUM_BATCHED_TOKENS = 4096
GPU_MEMORY_UTILIZATION = 0.95
DATA_PARALLEL_SIZE = 1
TENSOR_PARALLEL_SIZE = 1
PIPELINE_PARALLEL_SIZE = 1
EXPECTED_VLLM_VERSION = "0.8.5.post1"
DEFAULT_RUN_ROOT = ROOT / (
    "output/evaluation/main/"
    "chk3_cp318_core8_loo_vllm_k5_n128_1993_2008_20260817_v1"
)
DEFAULT_COHORT = DEFAULT_RUN_ROOT / "preparation_v2/cohort_n2176_k5.v2.json"
DEFAULT_EXECUTION_POLICY = (
    DEFAULT_RUN_ROOT / "preparation_v2/execution_policy.v2.json"
)
HISTORICAL_GREEDY_K1_ROOT = ROOT / (
    "output/evaluation/main/"
    "chk3_cp318_core8_loo_full_n128_1993_2008_20260814_v1"
)
HISTORICAL_GREEDY_K1_FILES = {
    "generation_manifest": (
        HISTORICAL_GREEDY_K1_ROOT / "generation_v1/manifest.json",
        "b610716e5f35a4ea8012e9a4b5c2a41b0d557e2a08bac79d709839f87ec969b6",
    ),
    "generations": (
        HISTORICAL_GREEDY_K1_ROOT / "generation_v1/generations.jsonl",
        "9901a90210d81ae7f97a08965e201d267b9360dcc449d8813d2d0348550608af",
    ),
    "score_manifest": (
        HISTORICAL_GREEDY_K1_ROOT / "score_v1/manifest.json",
        "4226551fd40ea9b02678624756295d868929fecec8664352ff5282134dde8df8",
    ),
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Core8LooVllmK5Error(shared.DualDp1GenerationError):
    """The frozen LOO vLLM K=5 contract was violated."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_text(value: str) -> str:
    return shared._sha256_text(value)


def _binding(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    return shared._file_binding(path, rows=rows)


def _validate_max_num_seqs(value: int) -> int:
    if isinstance(value, bool) or value != SELECTED_MAX_NUM_SEQS:
        raise Core8LooVllmK5Error("LOO K5 requires max_num_seqs=16")
    return value


def sampling_contract(*, max_num_seqs: int) -> dict[str, Any]:
    """Return the immutable stochastic inference contract."""

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
        "max_num_seqs_selection": "fixed_16_from_sealed_dual_dp1_speed_evidence",
        "max_num_batched_tokens_per_worker": MAX_NUM_BATCHED_TOKENS,
        "gpu_memory_utilization_per_worker": GPU_MEMORY_UTILIZATION,
        "swap_space_gib": 0,
        "cpu_offload_gib": 0,
        "workers": SHARD_COUNT,
        "data_parallel_size_per_worker": DATA_PARALLEL_SIZE,
        "tensor_parallel_size_per_worker": TENSOR_PARALLEL_SIZE,
        "pipeline_parallel_size_per_worker": PIPELINE_PARALLEL_SIZE,
        "parallel_topology": "two_external_independent_full_bf16_dp1_replicas",
        "physical_gpu_indexes": [0, 1],
        "worker_mapping": "shard_id_equals_physical_gpu_index",
        "paired_block": "meeting_rank_times_5_plus_replicate_id",
        "shard_function": "paired_block_mod_2",
        "paired_block_never_crosses_workers": True,
        "cases_per_absolute_chunk_per_worker": CASES_PER_SHARD_CHUNK,
        "absolute_chunk_size_across_workers": ABSOLUTE_CHUNK_SIZE,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "enforce_eager": False,
        "cuda_graphs": True,
        "async_output_processing": True,
        "engine_seed": 0,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "row_seed": "derive_row_seed(replicate_seed,meeting_full_sample_id)",
        "common_random_numbers": "same_meeting_replicate_across_all_17_arms",
        "canonical_case_order": (
            "meeting_order_then_variant_rank_then_replicate_id"
        ),
        "historical_greedy_k1_reused_as_replicate": False,
        "five_fresh_same_distribution_stochastic_replicates": True,
        "stochastic_reproducibility_semantics": (
            "schedule_sensitive_seeded_sampling_token_identity_not_required_"
            "across_engine_launches_or_batch_concurrency"
        ),
        "token_identity_replay_gate": False,
        "resume_dispatch": "only_missing_tuple_keys_never_fsynced_wal_keys",
        "durable_wal_tuple_policy": "immutable_validate_only_never_redispatch",
        "chunk_receipt_commit": "only_after_exact_full_chunk_key_union",
        "input_contract": "consume_exact_fomc_prompt_token_ledger_no_runtime_chat_template",
        "worker_wal_order": "request_completion_order",
        "canonical_output_order": "absolute_case_index",
        "required_environment": dict(shared.REQUIRED_VLLM_ENV),
        "forbidden_data_parallel_environment": list(shared.FORBIDDEN_DP_ENV),
        "recommended_model_order": ["chk3"],
    }


def _sampling_contract_sha256(*, max_num_seqs: int) -> str:
    return _sha256_text(_canonical(sampling_contract(max_num_seqs=max_num_seqs)))


def paired_block_id(meeting_rank: int, replicate_id: int) -> int:
    if (
        isinstance(meeting_rank, bool)
        or not isinstance(meeting_rank, int)
        or meeting_rank < 0
        or meeting_rank >= EXPECTED_MEETINGS
        or isinstance(replicate_id, bool)
        or not isinstance(replicate_id, int)
        or replicate_id < 0
        or replicate_id >= REPLICATES
    ):
        raise Core8LooVllmK5Error("invalid paired-block identity")
    return meeting_rank * REPLICATES + replicate_id


def assigned_shard(absolute_case_index: int) -> int:
    if (
        isinstance(absolute_case_index, bool)
        or not isinstance(absolute_case_index, int)
        or absolute_case_index < 0
    ):
        raise Core8LooVllmK5Error("absolute_case_index must be non-negative")
    meeting_rank = absolute_case_index // CASES_PER_MEETING
    replicate_id = absolute_case_index % REPLICATES
    return paired_block_id(meeting_rank, replicate_id) % SHARD_COUNT


def shard_cases(
    cases: Sequence[Mapping[str, Any]], shard_id: int
) -> list[Mapping[str, Any]]:
    if shard_id not in (0, 1) or isinstance(shard_id, bool):
        raise Core8LooVllmK5Error("shard_id must be exactly 0 or 1")
    selected = [
        case
        for case in cases
        if assigned_shard(int(case["absolute_case_index"])) == shard_id
    ]
    for chunk_id in range(len(cases) // ABSOLUTE_CHUNK_SIZE):
        chunk = [
            case for case in selected if int(case["absolute_chunk_id"]) == chunk_id
        ]
        if len(chunk) != CASES_PER_SHARD_CHUNK:
            raise Core8LooVllmK5Error(
                f"absolute chunk {chunk_id} does not contain 85 shard cases"
            )
    return selected


def _read_json_binding(binding: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(str(binding.get("path") or "")).expanduser().resolve()
    if path.is_symlink() or not path.is_file():
        raise Core8LooVllmK5Error(f"bound JSON is missing or unsafe: {path}")
    if (
        binding.get("sha256") != shared.core._sha256_file(path)
        or binding.get("bytes") != path.stat().st_size
    ):
        raise Core8LooVllmK5Error(f"bound JSON drift: {path}")
    value = shared.core._read_json(path)
    payload = validate_manifest_integrity(value)
    if binding.get("payload_sha256") not in {None, payload}:
        raise Core8LooVllmK5Error(f"bound JSON payload drift: {path}")
    return path, value


def load_execution_policy(
    path: Path = DEFAULT_EXECUTION_POLICY,
) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise Core8LooVllmK5Error("execution policy is a symlink")
    path = unresolved.resolve()
    if not path.is_file():
        raise Core8LooVllmK5Error(f"execution policy is missing: {path}")
    value = shared.core._read_json(path)
    payload = validate_manifest_integrity(value)
    topology = value.get("topology")
    engine = value.get("engine")
    sampling = value.get("sampling")
    scheduling = value.get("scheduling_and_replay")
    coverage = value.get("coverage")
    if (
        value.get("status") != "complete"
        or value.get("evaluation_id") != EVALUATION_ID
        or not isinstance(topology, Mapping)
        or topology.get("workers") != 2
        or topology.get("physical_gpu_indexes") != [0, 1]
        or topology.get("data_parallel_size_per_worker") != 1
        or topology.get("tensor_parallel_size_per_worker") != 1
        or topology.get("pipeline_parallel_size_per_worker") != 1
        or topology.get("dtype") != "bfloat16"
        or topology.get("quantization") is not None
        or not isinstance(engine, Mapping)
        or engine.get("max_num_seqs_per_worker") != 16
        or engine.get("max_model_len") != 4096
        or engine.get("gpu_memory_utilization_per_worker") != 0.95
        or not isinstance(sampling, Mapping)
        or sampling.get("replicate_seeds") != list(REPLICATE_SEEDS)
        or sampling.get("replicates") != 5
        or sampling.get("historical_greedy_k1_excluded") is not True
        or sampling.get("all_replicates_fresh_vllm_generations") is not True
        or not isinstance(scheduling, Mapping)
        or scheduling.get("token_identity_across_launches_or_batch_schedules_required")
        is not False
        or scheduling.get("schedule_sensitive_token_identity_replay_gate")
        != "non_blocking"
        or scheduling.get("canonical_case_order")
        != "meeting_order_then_variant_rank_then_replicate_id"
        or scheduling.get("shard_unit") != "meeting_replicate_17_arm_block"
        or not isinstance(coverage, Mapping)
        or coverage.get("generation_rows") != CASES_PER_MODEL
        or coverage.get("meetings") != EXPECTED_MEETINGS
        or coverage.get("prompts") != EXPECTED_PROMPTS
    ):
        raise Core8LooVllmK5Error("execution policy contract drift")
    return {
        "manifest": value,
        "receipt_binding": {
            **_binding(path),
            "payload_sha256": payload,
        },
    }


def load_cohort(
    cohort_path: Path, cohort_sha256: str
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    if SHA256_RE.fullmatch(cohort_sha256) is None:
        raise Core8LooVllmK5Error("cohort SHA-256 is invalid")
    cohort_path = cohort_path.expanduser().resolve()
    if (
        cohort_path.is_symlink()
        or not cohort_path.is_file()
        or shared.core._sha256_file(cohort_path) != cohort_sha256
    ):
        raise Core8LooVllmK5Error("cohort binding drift")
    cohort = shared.core._read_json(cohort_path)
    validate_manifest_integrity(cohort)
    if (
        cohort.get("schema_version") != preparation.COHORT_SCHEMA
        or cohort.get("status") != "complete"
        or cohort.get("evaluation_id") != EVALUATION_ID
    ):
        raise Core8LooVllmK5Error("cohort identity drift")
    policy_binding = load_execution_policy()["receipt_binding"]
    if cohort.get("execution_policy") != policy_binding:
        raise Core8LooVllmK5Error("cohort execution-policy binding drift")
    design = cohort.get("generation_design")
    coverage = cohort.get("coverage")
    if (
        not isinstance(design, Mapping)
        or design.get("prompts") != EXPECTED_PROMPTS
        or design.get("meetings") != EXPECTED_MEETINGS
        or design.get("variants_per_meeting") != VARIANTS_PER_MEETING
        or design.get("replicate_seeds") != list(REPLICATE_SEEDS)
        or design.get("rows") != CASES_PER_MODEL
        or design.get("absolute_chunk_size") != ABSOLUTE_CHUNK_SIZE
        or design.get("canonical_case_order")
        != "meeting_order_then_variant_rank_then_replicate_id"
        or design.get("greedy_rows_reused") is not False
        or not isinstance(coverage, Mapping)
        or coverage.get("prompts") != EXPECTED_PROMPTS
        or coverage.get("meetings") != EXPECTED_MEETINGS
        or coverage.get("generation_rows") != CASES_PER_MODEL
    ):
        raise Core8LooVllmK5Error("cohort K5 design/coverage drift")
    source_binding = cohort.get("source_sample_manifest")
    ledger_binding = cohort.get("token_ledger")
    if not isinstance(source_binding, Mapping) or not isinstance(
        ledger_binding, Mapping
    ):
        raise Core8LooVllmK5Error("cohort source/ledger binding is missing")
    sample_path = Path(str(source_binding.get("path") or "")).resolve()
    try:
        sample_manifest, input_rows, _tokenizer, observed = k10._load_inputs(
            sample_path,
            str(source_binding.get("sha256") or ""),
            load_tokenizer=False,
        )
    except k10.Core8StochasticK10Error as exc:
        raise Core8LooVllmK5Error(str(exc)) from exc
    if observed != source_binding.get("sha256"):
        raise Core8LooVllmK5Error("source sample manifest SHA drift")
    try:
        runtime_inventory = preparation.load_model_runtime_inventory(sample_manifest)
    except preparation.LooVllmK5PreparationError as exc:
        raise Core8LooVllmK5Error(str(exc)) from exc
    if cohort.get("model_runtime_inventory") != runtime_inventory:
        raise Core8LooVllmK5Error("CHK3 vLLM runtime model/tokenizer inventory drift")
    ledger_path = Path(str(ledger_binding.get("path") or "")).resolve()
    ledger_rows, encoded = k10._read_canonical_jsonl(ledger_path)
    if (
        ledger_binding.get("sha256") != shared.core._sha256_file(ledger_path)
        or ledger_binding.get("bytes") != ledger_path.stat().st_size
        or ledger_binding.get("rows") != EXPECTED_PROMPTS
        or len(ledger_rows) != EXPECTED_PROMPTS
        or sum(len(line) for line in encoded) != ledger_path.stat().st_size
    ):
        raise Core8LooVllmK5Error("exact token ledger binding drift")
    samples = sample_manifest["samples"]
    for index, (sample, entry) in enumerate(zip(samples, ledger_rows, strict=True)):
        token_ids = entry.get("prompt_token_ids")
        if (
            entry.get("schema_version") != preparation.LEDGER_ROW_SCHEMA
            or entry.get("absolute_prompt_index") != index
            or entry.get("source_sample_line_number") != index + 1
            or entry.get("sample_id") != sample.get("sample_id")
            or entry.get("meeting_id") != sample.get("meeting_id")
            or entry.get("prompt_sha256") != sample.get("prompt_sha256")
            or SHA256_RE.fullmatch(str(entry.get("messages_sha256") or "")) is None
            or not isinstance(token_ids, list)
            or not token_ids
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in token_ids
            )
            or entry.get("prompt_token_count") != len(token_ids)
            or len(token_ids) != sample.get("prompt_token_count")
            or entry.get("prompt_token_ids_sha256")
            != _sha256_text(_canonical(token_ids))
            or len(token_ids) + MAX_NEW_TOKENS > MAX_MODEL_LEN
        ):
            raise Core8LooVllmK5Error(f"token ledger drift at row {index}")
    bound_rows = {str(row["sample_id"]): dict(row) for row in input_rows}
    if len(bound_rows) != EXPECTED_PROMPTS:
        raise Core8LooVllmK5Error("input row identities are not unique")
    return cohort, ledger_rows, sample_manifest, bound_rows


def _paired_seed_keys(samples: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    try:
        return k10._paired_seed_keys(samples)
    except k10.Core8StochasticK10Error as exc:
        raise Core8LooVllmK5Error(str(exc)) from exc


def canonical_cases(
    *,
    samples: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    smoke: bool,
) -> list[dict[str, Any]]:
    if len(samples) != EXPECTED_PROMPTS or len(ledger) != EXPECTED_PROMPTS:
        raise Core8LooVllmK5Error("LOO sample/ledger coverage drift")
    count = SMOKE_PROMPTS if smoke else EXPECTED_PROMPTS
    paired = _paired_seed_keys(samples)
    cases: list[dict[str, Any]] = []
    for sample, token_entry in zip(samples[:count], ledger[:count], strict=True):
        meeting_rank = int(sample["meeting_rank"])
        for replicate_id, replicate_seed in enumerate(REPLICATE_SEEDS):
            absolute_case_index = len(cases)
            block = paired_block_id(meeting_rank, replicate_id)
            paired_seed_key = paired[str(sample["meeting_id"])]
            cases.append(
                {
                    "absolute_case_index": absolute_case_index,
                    "absolute_chunk_id": absolute_case_index // ABSOLUTE_CHUNK_SIZE,
                    "sample": sample,
                    "token_entry": token_entry,
                    "meeting_rank": meeting_rank,
                    "variant_rank": int(sample["variant_rank"]),
                    "replicate_id": replicate_id,
                    "replicate_seed": replicate_seed,
                    "paired_seed_key": paired_seed_key,
                    "paired_block_id": block,
                    "row_seed": derive_row_seed(replicate_seed, paired_seed_key),
                }
            )
    expected = SMOKE_CASES_PER_MODEL if smoke else CASES_PER_MODEL
    if len(cases) != expected:
        raise Core8LooVllmK5Error("canonical case coverage drift")
    for case in cases:
        if assigned_shard(int(case["absolute_case_index"])) != int(
            case["paired_block_id"]
        ) % 2:
            raise Core8LooVllmK5Error("paired-block shard drift")
    return cases


def _normalize_finish_reason(
    *,
    raw_finish_reason: Any,
    raw_stop_reason: Any,
    generated_token_ids: Sequence[int],
    eos_token_ids: Sequence[int],
) -> str:
    if raw_stop_reason is not None:
        raise Core8LooVllmK5Error("unexpected explicit vLLM stop reason")
    positions = [
        index
        for index, token in enumerate(generated_token_ids)
        if token in set(eos_token_ids)
    ]
    if raw_finish_reason == "stop" and positions == [len(generated_token_ids) - 1]:
        return "eos"
    if (
        raw_finish_reason == "length"
        and not positions
        and len(generated_token_ids) == MAX_NEW_TOKENS
    ):
        return "length"
    raise Core8LooVllmK5Error("vLLM finish/token contract drift")


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
    _validate_max_num_seqs(max_num_seqs)
    if model_id != "chk3" or model_label != MODEL_LABELS["chk3"]:
        raise Core8LooVllmK5Error("model identity drift")
    if shard_id != physical_gpu_index or assigned_shard(
        int(case["absolute_case_index"])
    ) != shard_id:
        raise Core8LooVllmK5Error("case submitted to wrong paired-block shard")
    sample = case["sample"]
    normalized_finish = _normalize_finish_reason(
        raw_finish_reason=raw_finish_reason,
        raw_stop_reason=raw_stop_reason,
        generated_token_ids=generated_token_ids,
        eos_token_ids=eos_token_ids,
    )
    try:
        base = native_eval.build_full_result(
            text=generated_text,
            generated_token_ids=generated_token_ids,
            eos_token_ids=eos_token_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            tail_tokens=TAIL_TOKENS,
            source_prompt=str(source_row["prompt"]),
            source_analysis=str(source_row["source_analysis"]),
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
    except native_eval.NativeThreeModelEvalError as exc:
        raise Core8LooVllmK5Error(str(exc)) from exc
    row: dict[str, Any] = {
        **base,
        "schema_version": ROW_SCHEMA,
        "evaluation_id": EVALUATION_ID,
        "model_id": model_id,
        "variant_sample_id": sample["sample_id"],
        "meeting_id": sample["meeting_id"],
        "meeting_rank": case["meeting_rank"],
        "variant_rank": case["variant_rank"],
        "arm": sample["arm"],
        "intervention_topic": sample["intervention_topic"],
        "replicate_id": case["replicate_id"],
        "replicate_seed": case["replicate_seed"],
        "paired_seed_key": case["paired_seed_key"],
        "paired_block_id": case["paired_block_id"],
        "row_seed": case["row_seed"],
        "absolute_case_index": case["absolute_case_index"],
        "absolute_chunk_id": case["absolute_chunk_id"],
        "absolute_chunk_size": ABSOLUTE_CHUNK_SIZE,
        "request_id": (
            f"chk3-case-{int(case['absolute_case_index']):05d}-"
            f"seed-{int(case['row_seed'])}"
        ),
        "generation_mode": "stochastic_sampling",
        "do_sample": True,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "inference_backend": "vllm-async-engine-v1-independent-dp1",
        "model_dtype": "bfloat16",
        "quantization": None,
        "data_parallel_size": 1,
        "tensor_parallel_size": 1,
        "physical_gpu_indexes": [physical_gpu_index],
        "physical_gpu_index": physical_gpu_index,
        "shard_id": shard_id,
        "shard_function": "paired_block_mod_2",
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_num_seqs": max_num_seqs,
        "input_token_count": case["token_entry"]["prompt_token_count"],
        "prompt_token_ledger_line_number": (
            int(case["token_entry"]["absolute_prompt_index"]) + 1
        ),
        "prompt_token_ids_sha256": case["token_entry"]["prompt_token_ids_sha256"],
        "input_truncated": False,
        "finish_reason": normalized_finish,
        "vllm_raw_finish_reason": raw_finish_reason,
        "vllm_raw_stop_reason": raw_stop_reason,
        "sampling_contract_sha256": _sampling_contract_sha256(
            max_num_seqs=max_num_seqs
        ),
        "source_artifact_sha256s": copy.deepcopy(dict(source_artifact_sha256s)),
        "full_source_analysis_sha256": source_row["full_source_analysis_sha256"],
        "full_reference_sha256": source_row["full_reference_sha256"],
        "full_prompt_token_count": source_row["full_prompt_token_count"],
        "semantic_status": "pending_offline",
    }
    try:
        metrics, audit = stochastic._generation_metrics(row)
    except stochastic.StochasticBootstrapGenerationError as exc:
        raise Core8LooVllmK5Error(str(exc)) from exc
    row["generation_metrics"] = metrics
    row["preregistered_core_valid"] = bool(audit["preregistered_core_valid"])
    row["preregistered_core_failures"] = list(audit["preregistered_core_failures"])
    row["six_metrics"] = {
        **metrics,
        "mpnet_cosine": None,
        "bertscore_f1": None,
    }
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
        raise Core8LooVllmK5Error(
            f"persisted LOO row drift at fields: {changed[:12]}"
        )


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
        raise Core8LooVllmK5Error(f"vLLM request did not finish: {request_id}")
    return case, final


def _persistence_contract() -> dict[str, Any]:
    return {
        "single_writer_shard_wal": True,
        "append_flush_fsync_per_completed_request": True,
        "fsynced_wal_tuple_keys_are_immutable_validate_only": True,
        "resume_dispatch": "only_missing_tuple_keys_never_fsynced_wal_keys",
        "active_85_case_paired_block_chunk_missing_key_recovery": True,
        "chunk_receipt_after_exact_85_key_union_only": True,
        "token_identity_replay_required": False,
        "full_generated_text": True,
        "full_answer": True,
        "full_generated_token_ids": True,
        "canonical_shard_sort": "absolute_case_index",
    }


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
        raise Core8LooVllmK5Error("shard chunk does not contain exactly 85 cases")
    start = chunk_id * ABSOLUTE_CHUNK_SIZE
    end = start + ABSOLUTE_CHUNK_SIZE
    observed = {
        index
        for index in rows_by_index
        if start <= index < end and assigned_shard(index) == shard_id
    }
    if observed != set(indexes):
        raise Core8LooVllmK5Error(
            "cannot receipt without the exact 85-key paired-block union"
        )
    bindings = []
    for index in indexes:
        row = rows_by_index.get(index)
        if row is None:
            raise Core8LooVllmK5Error("cannot receipt an incomplete LOO chunk")
        bindings.append(
            {
                "absolute_case_index": index,
                "sample_id": row["sample_id"],
                "replicate_id": row["replicate_id"],
                "row_seed": row["row_seed"],
                "completion_sha256": row["completion_sha256"],
                "generated_token_ids_sha256": row[
                    "generated_token_ids_sha256"
                ],
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


def _historical_greedy_k1_evidence() -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for name, (path, expected_sha) in HISTORICAL_GREEDY_K1_FILES.items():
        binding = _binding(path, rows=2176 if name == "generations" else None)
        if binding["sha256"] != expected_sha:
            raise Core8LooVllmK5Error(f"historical greedy K1 {name} SHA drift")
        if name.endswith("manifest"):
            value = shared.core._read_json(path)
            binding["payload_sha256"] = validate_manifest_integrity(value)
        evidence[name] = binding
    return {
        "role": "deterministic_anchor_only_excluded_from_five_stochastic_replicates",
        "reuse_as_stochastic_replicate": False,
        "artifacts": evidence,
    }


def _source_hashes(
    cohort_path: Path,
    cohort_sha256: str,
    cohort: Mapping[str, Any],
    sample_manifest: Mapping[str, Any],
    max_num_seqs: int,
    formal_authorization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source_sha = str(cohort["source_sample_manifest"]["sha256"])
    hashes = k10._source_hashes(sample_manifest, source_sha)
    hashes.update(
        {
            "loo_vllm_k5_cohort_sha256": cohort_sha256,
            "loo_vllm_k5_token_ledger_sha256": cohort["token_ledger"]["sha256"],
            "loo_vllm_k5_sampling_contract_sha256": _sampling_contract_sha256(
                max_num_seqs=max_num_seqs
            ),
            # The shared merger currently calls this historical field name;
            # its value is the new LOO execution policy, never the beta receipt.
            "vllm_k5_dual_dp1_stochastic_schedule_remediation_receipt": (
                load_execution_policy()["receipt_binding"]
            ),
            "loo_formal_generation_authorization": copy.deepcopy(
                formal_authorization
            ),
            "historical_greedy_k1": _historical_greedy_k1_evidence(),
        }
    )
    hashes["implementation_sources"] = {
        "loo_vllm_k5_runner": _binding(Path(__file__).resolve()),
        "loo_vllm_k5_preparer": _binding(Path(preparation.__file__).resolve()),
        "shared_dual_dp1_runtime": _binding(Path(shared.__file__).resolve()),
        "shared_vllm_dp2_validation": _binding(Path(shared.dp2.__file__).resolve()),
        "shared_generation_contract": _binding(Path(shared.core.__file__).resolve()),
        "loo_input_contract": _binding(Path(k10.__file__).resolve()),
        "native_result_contract": _binding(Path(native_eval.__file__).resolve()),
        "row_seed_contract": _binding(Path(loo_spec.__file__).resolve()),
        "loo_dual_dp1_orchestrator": _binding(
            ROOT / "jobs/eval/orchestrate_chk3_core8_loo_vllm_k5_dual_dp1.py"
        ),
        "shared_dual_dp1_orchestrator": _binding(
            ROOT
            / "jobs/eval/"
            "orchestrate_chk3_beta_core8_merged_vllm_k5_dual_dp1_"
            "stochastic_schedule_v2.py"
        ),
        "pipeline_launcher": _binding(
            ROOT / "run/eval_chk3_core8_loo_vllm_k5_dual_dp1.sh"
        ),
    }
    return hashes


def _load_formal_authorization(
    authorization_path: Path,
    *,
    cohort_path: Path,
    cohort_sha256: str,
    max_num_seqs: int,
) -> dict[str, Any]:
    loaded = load_and_validate_run(
        authorization_path,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
        expected_model_id="chk3",
        expected_scope="infrastructure_smoke",
        max_num_seqs=max_num_seqs,
    )
    manifest = loaded["manifest"]
    coverage = manifest.get("coverage")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("cases") != SMOKE_CASES_PER_MODEL
        or coverage.get("shard_cases") != {"shard0": 85, "shard1": 85}
        or coverage.get("input_truncation_cases") != 0
        or coverage.get("absolute_indexes_exact") is not True
        or coverage.get("overlaps") != 0
    ):
        raise Core8LooVllmK5Error("LOO smoke authorization gates failed")
    return {
        "official_smoke_run": copy.deepcopy(loaded["manifest_binding"]),
        "execution_policy": load_execution_policy()["receipt_binding"],
        "generation_gates": {
            "exact_inputs_seeds_config_topology_coverage": True,
            "two_meetings_170_rows_85_per_worker": True,
            "zero_input_truncation": True,
            "schedule_sensitive_token_identity_nonblocking": True,
            "formal_generation_unblocked": True,
        },
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
            raise Core8LooVllmK5Error("smoke must not carry formal authorization")
        return None
    if scope != "formal_merged_panel" or authorization_path is None:
        raise Core8LooVllmK5Error(
            "formal generation requires the deeply validated LOO smoke run"
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
            raise Core8LooVllmK5Error("smoke authorization must be null")
        return None
    if not isinstance(authorization, Mapping):
        raise Core8LooVllmK5Error("formal authorization is missing")
    binding = authorization.get("official_smoke_run")
    if not isinstance(binding, Mapping):
        raise Core8LooVllmK5Error("formal smoke binding is missing")
    path = shared._verify_arbitrary_binding(binding)
    observed = _load_formal_authorization(
        path,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
        max_num_seqs=max_num_seqs,
    )
    if dict(authorization) != observed:
        raise Core8LooVllmK5Error("formal authorization drift")
    return observed


def _shared_replacements() -> dict[str, Any]:
    return {
        "__doc__": __doc__,
        "EVALUATION_ID": EVALUATION_ID,
        "MODEL_ORDER": MODEL_ORDER,
        "MODEL_LABELS": MODEL_LABELS,
        "REPLICATE_SEEDS": REPLICATE_SEEDS,
        "CASES_PER_MODEL": CASES_PER_MODEL,
        "ABSOLUTE_CHUNK_SIZE": ABSOLUTE_CHUNK_SIZE,
        "CHUNKS_PER_MODEL": CHUNKS_PER_MODEL,
        "SMOKE_CASES_PER_MODEL": SMOKE_CASES_PER_MODEL,
        "SHARD_COUNT": SHARD_COUNT,
        "CASES_PER_SHARD_CHUNK": CASES_PER_SHARD_CHUNK,
        "FORMAL_CASES_PER_SHARD": FORMAL_CASES_PER_SHARD,
        "SMOKE_CASES_PER_SHARD": SMOKE_CASES_PER_SHARD,
        "ROW_SCHEMA": ROW_SCHEMA,
        "SHARD_MANIFEST_SCHEMA": SHARD_MANIFEST_SCHEMA,
        "MERGED_MANIFEST_SCHEMA": MERGED_MANIFEST_SCHEMA,
        "STATE_SCHEMA": STATE_SCHEMA,
        "CHUNK_RECEIPT_SCHEMA": CHUNK_RECEIPT_SCHEMA,
        "READY_SCHEMA": READY_SCHEMA,
        "GO_SCHEMA": GO_SCHEMA,
        "DONE_SCHEMA": DONE_SCHEMA,
        "ORCHESTRATOR_TIMING_SCHEMA": ORCHESTRATOR_TIMING_SCHEMA,
        "TEMPERATURE": TEMPERATURE,
        "TOP_P": TOP_P,
        "TOP_K": TOP_K,
        "REPETITION_PENALTY": REPETITION_PENALTY,
        "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
        "TAIL_TOKENS": TAIL_TOKENS,
        "MAX_MODEL_LEN": MAX_MODEL_LEN,
        "SELECTED_MAX_NUM_SEQS": SELECTED_MAX_NUM_SEQS,
        "ALLOWED_MAX_NUM_SEQS": ALLOWED_MAX_NUM_SEQS,
        "MAX_NUM_BATCHED_TOKENS": MAX_NUM_BATCHED_TOKENS,
        "GPU_MEMORY_UTILIZATION": GPU_MEMORY_UTILIZATION,
        "DATA_PARALLEL_SIZE": DATA_PARALLEL_SIZE,
        "TENSOR_PARALLEL_SIZE": TENSOR_PARALLEL_SIZE,
        "PIPELINE_PARALLEL_SIZE": PIPELINE_PARALLEL_SIZE,
        "load_cohort": load_cohort,
        "canonical_cases": canonical_cases,
        "assigned_shard": assigned_shard,
        "shard_cases": shard_cases,
        "sampling_contract": sampling_contract,
        "_sampling_contract_sha256": _sampling_contract_sha256,
        "_validate_max_num_seqs": _validate_max_num_seqs,
        "_consume_request": _consume_request,
        "_persistence_contract": _persistence_contract,
        "_chunk_receipt": _chunk_receipt,
        "build_result": build_result,
        "validate_result": validate_result,
        "_source_hashes": _source_hashes,
        "_remediation_receipt_binding": lambda: load_execution_policy()[
            "receipt_binding"
        ],
        "_load_formal_authorization": _load_formal_authorization,
        "_formal_authorization_for_scope": _formal_authorization_for_scope,
        "_revalidate_formal_authorization": _revalidate_formal_authorization,
    }


def configure_shared_implementation() -> dict[str, Any]:
    """Install this profile and return the exact values needed to restore it."""

    replacements = _shared_replacements()
    previous = {name: getattr(shared, name) for name in replacements}
    for name, value in replacements.items():
        setattr(shared, name, value)
    return previous


def restore_shared_implementation(previous: Mapping[str, Any]) -> None:
    for name, value in previous.items():
        setattr(shared, name, value)


@contextlib.contextmanager
def configured_shared_implementation() -> Any:
    previous = configure_shared_implementation()
    try:
        yield shared
    finally:
        restore_shared_implementation(previous)


def _delegate(name: str, *args: Any, **kwargs: Any) -> Any:
    with configured_shared_implementation():
        return getattr(shared, name)(*args, **kwargs)


def run_shard(**kwargs: Any) -> dict[str, Any]:
    return _delegate("run_shard", **kwargs)


def load_and_validate_shard(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _delegate("load_and_validate_shard", *args, **kwargs)


def merge_run(**kwargs: Any) -> dict[str, Any]:
    return _delegate("merge_run", **kwargs)


def load_and_validate_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return _delegate("load_and_validate_run", *args, **kwargs)


def _resume_dispatch_plan(**kwargs: Any) -> Any:
    return _delegate("_resume_dispatch_plan", **kwargs)


def _chunk_indexes(chunk_id: int, shard_id: int, *, total_cases: int) -> list[int]:
    with configured_shared_implementation():
        return shared._chunk_indexes(chunk_id, shard_id, total_cases=total_cases)


def _parser() -> argparse.ArgumentParser:
    with configured_shared_implementation():
        return shared._parser()


def main(argv: Sequence[str] | None = None) -> int:
    with configured_shared_implementation():
        return shared.main(argv)


def __getattr__(name: str) -> Any:
    value = getattr(shared, name)
    if callable(value):
        def scoped(*args: Any, **kwargs: Any) -> Any:
            return _delegate(name, *args, **kwargs)

        return scoped
    return value


if __name__ == "__main__":
    raise SystemExit(main())
