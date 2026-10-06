"""Generate the paper chk-2 checkpoint-50 Core8 LOO K=10 panel with vLLM.

The preparation contract is owned by :mod:`paper_chk2_core8_loo_common`.
This module consumes its sealed 2,176-row prompt-token ledger and runs one
independent DP=1 vLLM engine per physical GPU.  A complete
``(meeting, replicate)`` block (all 17 prompt variants) is assigned to one
worker, so common-random-number comparisons never cross GPUs.

Every completed request is appended, flushed, and fsynced to a shard WAL.
Resume accepts only deeply revalidated rows and dispatches only absent tuple
keys.  A formal worker is fail-closed until the two-meeting K=2 smoke run has
been validated.  This module does not launch both workers itself; the caller
pins each worker with ``CUDA_VISIBLE_DEVICES=0`` or ``1``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import gc
import hashlib
import os
import platform
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.eval import eval_chk3_native_three_model as native_eval
from jobs.eval import eval_chk3_stochastic_bootstrap_generation as stochastic
from jobs.eval import paper_chk2_core8_loo_common as common
from jobs.eval import paper_chk2_text_similarity_vllm as durable
from jobs.retrain_v2 import probe_chk3_sft_degeneration as native_probe
from open_r1.validator.loo_generation_spec import derive_row_seed


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = common.DEFAULT_OUTPUT_ROOT

EVALUATION_ID = "paper-chk2-cp50-core8-loo-vllm-k10-v1"
MODEL_ID = "paper_chk2"
MODEL_LABEL = "paper-chk2-cp50-lora-over-chk1-cp200"
SHARD_IDS = (0, 1)
EXPECTED_MEETINGS = 128
SMOKE_MEETINGS = 2
SMOKE_REPLICATES = 2
REPLICATES = len(common.REPLICATE_SEEDS)
EXPECTED_PROMPTS = common.EXPECTED_PROMPTS
EXPECTED_GENERATIONS = common.EXPECTED_GENERATIONS
VARIANTS_PER_MEETING = common.VARIANTS_PER_MEETING
TAIL_TOKENS = 1024
MAX_NUM_SEQS = 16
MAX_NUM_BATCHED_TOKENS = 4096
EXPECTED_VLLM_VERSION = "0.8.5.post1"
EXPECTED_VLLM_TRANSFORMERS_VERSION = "4.51.3"
EXPECTED_VLLM_TOKENIZERS_VERSION = "0.21.1"

ROW_SCHEMA = "paper-chk2-core8-loo-vllm-k10-generation-row-v1"
LAUNCH_SCHEMA = "paper-chk2-core8-loo-vllm-k10-launch-v1"
SHARD_SCHEMA = "paper-chk2-core8-loo-vllm-k10-shard-v1"
VALIDATION_SCHEMA = "paper-chk2-core8-loo-vllm-k10-validation-v1"
TORN_TAIL_SCHEMA = "paper-chk2-core8-loo-vllm-k10-wal-tail-recovery-v1"
LOCK_TEMPLATE = "/tmp/fomc_trainer_paper_chk2_core8_loo_gpu{shard}.lock"


class PaperChk2Core8GenerationError(RuntimeError):
    """The prepared data, model route, WAL, or paired-panel contract drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperChk2Core8GenerationError(message)


def _canonical(value: Any) -> str:
    return durable.canonical_json(value)


def _sha256_text(value: str) -> str:
    return durable.sha256_text(value)


def _sha256_file(path: Path) -> str:
    return durable.sha256_file(path)


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    return _sha256_text(_canonical(list(token_ids)))


def _mode_root(output_root: Path, *, smoke: bool) -> Path:
    output_root = output_root.expanduser().resolve()
    return output_root / ("smoke" if smoke else "generation")


def _manifest_path(output_root: Path) -> Path:
    return output_root.expanduser().resolve() / "preparation/evaluation_manifest.json"


def generation_contract() -> dict[str, Any]:
    """Return the immutable cp50 stochastic inference contract."""

    return {
        "backend": "vllm-async-engine-two-independent-dp1-workers",
        "vllm_version": EXPECTED_VLLM_VERSION,
        "vllm_worker_transformers_version": EXPECTED_VLLM_TRANSFORMERS_VERSION,
        "vllm_worker_tokenizers_version": EXPECTED_VLLM_TOKENIZERS_VERSION,
        "model_id": MODEL_ID,
        "model_label": MODEL_LABEL,
        "base_model": "exact-merged-chk1-checkpoint-200",
        "adapter": "paper-chk2-checkpoint-50-dynamic-lora",
        "lora_request": {
            "name": "paper-chk2-cp50",
            "integer_id": 50,
            "max_lora_rank": 32,
            "max_loras": 1,
            "max_cpu_loras": 1,
            "lora_dtype": "bfloat16",
        },
        "generation_mode": "stochastic_sampling",
        "do_sample": True,
        "temperature": common.TEMPERATURE,
        "top_p": common.TOP_P,
        "top_k": common.TOP_K,
        "repetition_penalty": common.REPETITION_PENALTY,
        "max_new_tokens": common.MAX_NEW_TOKENS,
        "max_model_len": common.MAX_MODEL_LEN,
        "dtype": "bfloat16",
        "quantization": None,
        "load_format": "safetensors",
        "max_num_seqs_per_worker": MAX_NUM_SEQS,
        "max_num_batched_tokens_per_worker": MAX_NUM_BATCHED_TOKENS,
        "gpu_memory_policy": "floor((free_mib-2048)/total_mib,0.01); minimum 0.78",
        "workers": 2,
        "tensor_parallel_size_per_worker": 1,
        "data_parallel_size_per_worker": 1,
        "pipeline_parallel_size_per_worker": 1,
        "paired_block": "meeting_rank_times_10_plus_replicate_id",
        "shard_function": "paired_block_mod_2",
        "paired_block_never_crosses_workers": True,
        "row_seed": "derive_row_seed(replicate_seed,meeting_full_sample_id)",
        "common_random_numbers": "same_meeting_replicate_across_all_17_arms",
        "replicate_seeds": list(common.REPLICATE_SEEDS),
        "input_contract": "sealed_prompt_token_ids_no_runtime_chat_template",
        "resume_dispatch": "only_missing_fsynced_wal_tuple_keys",
        "wal_commit": "append_flush_fsync_per_completed_request",
    }


def prepare_evaluation(
    output_root: Path = DEFAULT_OUTPUT_ROOT, *, resume: bool = False
) -> Mapping[str, Any]:
    """Build preparation once; an existing preparation is verify-only."""

    output_root = output_root.expanduser().resolve()
    manifest_path = _manifest_path(output_root)
    if manifest_path.exists():
        _require(
            manifest_path.is_file() and not manifest_path.is_symlink(),
            "unsafe existing preparation manifest",
        )
        # ``--resume`` deliberately performs no writes in this branch.
        result = common.validate_preparation_artifacts(
            output_root=output_root, verify_static_bindings=True
        )
        return {
            "status": "complete",
            "resume_verify_only": bool(resume),
            "evaluation_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256_file(manifest_path),
            },
            "validation": {
                "prepared_prompts": len(result.ledger_rows),
                "neutral_token_match_proofs": len(result.neutral_proofs),
                "preparation_dir": str(result.preparation_dir),
            },
        }
    _require(
        not output_root.is_symlink(), "output root must not be a symlink"
    )
    common.build_preparation_artifacts(output_root=output_root)
    result = common.validate_preparation_artifacts(
        output_root=output_root, verify_static_bindings=True
    )
    return {
        "status": "complete",
        "resume_verify_only": False,
        "evaluation_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
        },
        "validation": {
            "prepared_prompts": len(result.ledger_rows),
            "neutral_token_match_proofs": len(result.neutral_proofs),
            "preparation_dir": str(result.preparation_dir),
        },
    }


def _load_context(
    *, output_root: Path, manifest_path: Path
) -> tuple[Any, Any, dict[str, Any], tuple[dict[str, Any], ...], str]:
    output_root = output_root.expanduser().resolve()
    expected_manifest = _manifest_path(output_root)
    manifest_path = manifest_path.expanduser().resolve()
    _require(manifest_path == expected_manifest, "evaluation manifest path drift")
    common.validate_preparation_artifacts(
        output_root=output_root, verify_static_bindings=True
    )
    sealed_manifest, ledger = common.load_prepared_ledger(output_root=output_root)
    source = common.load_and_validate_frozen_source()
    bindings = common.verify_paper_chk2_bindings(load_tokenizer=False)
    _require(
        bindings.runtime_versions.get("transformers")
        == EXPECTED_VLLM_TRANSFORMERS_VERSION
        and bindings.runtime_versions.get("tokenizers")
        == EXPECTED_VLLM_TOKENIZERS_VERSION,
        "vLLM worker tokenizer runtime version drift",
    )
    _require(len(ledger) == EXPECTED_PROMPTS, "prepared ledger cardinality drift")
    manifest_sha = _sha256_file(manifest_path)
    return source, bindings, sealed_manifest, tuple(ledger), manifest_sha


def paired_block_id(meeting_rank: int, replicate_id: int) -> int:
    _require(
        isinstance(meeting_rank, int)
        and not isinstance(meeting_rank, bool)
        and 0 <= meeting_rank < EXPECTED_MEETINGS,
        "invalid meeting rank",
    )
    _require(
        isinstance(replicate_id, int)
        and not isinstance(replicate_id, bool)
        and 0 <= replicate_id < REPLICATES,
        "invalid replicate ID",
    )
    return meeting_rank * REPLICATES + replicate_id


def assigned_shard(meeting_rank: int, replicate_id: int) -> int:
    return paired_block_id(meeting_rank, replicate_id) % len(SHARD_IDS)


def generation_key(sample_id: str, replicate_id: int) -> str:
    _require(isinstance(sample_id, str) and bool(sample_id), "invalid sample ID")
    _require(0 <= replicate_id < REPLICATES, "invalid replicate ID")
    return f"{MODEL_ID}|{sample_id}|replicate-{replicate_id:02d}"


def build_case_matrix(
    source: Any,
    ledger: Sequence[Mapping[str, Any]],
    *,
    smoke: bool = False,
    smoke_meetings: int = SMOKE_MEETINGS,
) -> list[dict[str, Any]]:
    """Build the canonical sample-major, replicate-inner generation matrix."""

    _require(len(ledger) == EXPECTED_PROMPTS, "ledger row count drift")
    _require(
        1 <= smoke_meetings <= EXPECTED_MEETINGS,
        "invalid smoke meeting count",
    )
    ledger_by_id: dict[str, Mapping[str, Any]] = {}
    for row in ledger:
        sample_id = row.get("sample_id")
        _require(isinstance(sample_id, str), "ledger sample ID missing")
        _require(sample_id not in ledger_by_id, "duplicate ledger sample ID")
        ledger_by_id[sample_id] = row

    rows = tuple(source.ordered_rows)
    _require(len(rows) == EXPECTED_PROMPTS, "frozen source row count drift")
    cases: list[dict[str, Any]] = []
    for prompt_index, source_row in enumerate(rows):
        sample_id = str(source_row["sample_id"])
        token_entry = ledger_by_id.get(sample_id)
        _require(token_entry is not None, f"missing token ledger row: {sample_id}")
        meeting_id = str(source_row["meeting_id"])
        meeting_rank = int(source_row["meeting_rank"])
        variant_rank = int(source_row["variant_rank"])
        _require(
            prompt_index == meeting_rank * VARIANTS_PER_MEETING + variant_rank,
            "frozen source order/rank drift",
        )
        if smoke and meeting_rank >= smoke_meetings:
            continue
        for field in ("meeting_rank", "variant_rank", "prompt_sha256"):
            if field in token_entry:
                _require(
                    token_entry[field] == source_row[field],
                    f"ledger/source drift at {sample_id}:{field}",
                )
        prompt_ids = list(token_entry.get("prompt_token_ids") or [])
        _require(
            prompt_ids
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in prompt_ids
            ),
            f"invalid prompt token IDs: {sample_id}",
        )
        prompt_count = int(token_entry.get("prompt_token_count", len(prompt_ids)))
        _require(prompt_count == len(prompt_ids), "prompt token count drift")
        prompt_ids_sha = str(token_entry.get("prompt_token_ids_sha256"))
        _require(
            prompt_ids_sha == _token_ids_sha256(prompt_ids),
            f"prompt-token SHA drift: {sample_id}",
        )
        paired_seed_key = str(
            token_entry.get("paired_seed_key")
            or source.full_sample_id_by_meeting[meeting_id]
        )
        _require(
            paired_seed_key == source.full_sample_id_by_meeting[meeting_id],
            "paired seed key must be the meeting Full sample ID",
        )
        replicate_seeds = (
            common.REPLICATE_SEEDS[:SMOKE_REPLICATES]
            if smoke
            else common.REPLICATE_SEEDS
        )
        for replicate_id, replicate_seed in enumerate(replicate_seeds):
            block = paired_block_id(meeting_rank, replicate_id)
            absolute_case_index = prompt_index * REPLICATES + replicate_id
            cases.append(
                {
                    "absolute_case_index": absolute_case_index,
                    "generation_key": generation_key(sample_id, replicate_id),
                    "sample_id": sample_id,
                    "meeting_id": meeting_id,
                    "meeting_rank": meeting_rank,
                    "variant_rank": variant_rank,
                    "arm": source_row["arm"],
                    "intervention_topic": source_row.get("intervention_topic"),
                    "replicate_id": replicate_id,
                    "replicate_seed": int(replicate_seed),
                    "paired_seed_key": paired_seed_key,
                    "paired_block_id": block,
                    "row_seed": derive_row_seed(int(replicate_seed), paired_seed_key),
                    "shard_id": block % len(SHARD_IDS),
                    "prompt": source_row["prompt"],
                    "prompt_sha256": source_row["prompt_sha256"],
                    "prompt_token_ids": prompt_ids,
                    "prompt_token_ids_sha256": prompt_ids_sha,
                    "prompt_token_count": prompt_count,
                    "source_analysis": source_row["source_analysis"],
                    "source_analysis_sha256": source_row["source_analysis_sha256"],
                    "reference_minutes": source_row["reference_minutes"],
                    "reference_minutes_sha256": source_row[
                        "reference_minutes_sha256"
                    ],
                    "reference_response": source_row["response"],
                    "full_source_analysis_sha256": source_row[
                        "full_source_analysis_sha256"
                    ],
                    "full_reference_sha256": source_row["full_reference_sha256"],
                    "full_prompt_token_count": source_row["full_prompt_token_count"],
                }
            )
    expected = (
        smoke_meetings * VARIANTS_PER_MEETING * SMOKE_REPLICATES
        if smoke
        else EXPECTED_GENERATIONS
    )
    _require(len(cases) == expected, "case matrix cardinality drift")
    _require(
        len({str(case["generation_key"]) for case in cases}) == expected,
        "duplicate generation key",
    )
    return cases


def _load_tokenizer(bindings: Any) -> tuple[Any, list[int], int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(bindings.tokenizer_path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
        fix_mistral_regex=False,
    )
    class_name = f"{type(tokenizer).__module__}.{type(tokenizer).__name__}"
    _require(class_name == common.EXPECTED_TOKENIZER_CLASS, "tokenizer class drift")
    _require(
        isinstance(tokenizer.chat_template, str)
        and _sha256_text(tokenizer.chat_template)
        == common.CHK2_RUNTIME_FILES["chat_template.jinja"],
        "worker tokenizer chat-template drift",
    )
    eos = tokenizer.eos_token_id
    eos_ids = [int(value) for value in eos] if isinstance(eos, list) else [int(eos)]
    _require(eos_ids and all(value >= 0 for value in eos_ids), "invalid EOS IDs")
    pad = tokenizer.pad_token_id
    pad_id = int(pad if pad is not None else eos_ids[0])
    return tokenizer, eos_ids, pad_id


def _build_result(
    *,
    case: Mapping[str, Any],
    generated_text: str,
    generated_token_ids: Sequence[int],
    raw_finish_reason: Any,
    raw_stop_reason: Any,
    eos_token_ids: Sequence[int],
    pad_token_id: int,
    manifest_sha: str,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    token_ids = [int(value) for value in generated_token_ids]
    _require(
        all(value >= 0 for value in token_ids), "generated token IDs are invalid"
    )
    if tokenizer is not None:
        try:
            decoded = native_probe.common_probe.decode_completion_preserving_boundary(
                tokenizer, token_ids, eos_token_ids
            )
        except native_probe.common_probe.ProbeError as exc:
            raise PaperChk2Core8GenerationError(str(exc)) from exc
        _require(decoded == generated_text, "vLLM completion text/token drift")
    sample = {
        "sample_id": case["sample_id"],
        "length_bucket": "core8_combined",
        "prompt_token_count": case["prompt_token_count"],
    }
    try:
        base = dict(
            native_probe.build_redacted_result(
                text=generated_text,
                generated_token_ids=token_ids,
                eos_token_ids=eos_token_ids,
                max_new_tokens=common.MAX_NEW_TOKENS,
                tail_tokens=TAIL_TOKENS,
                source_analysis=str(case["source_analysis"]),
                model_label=MODEL_LABEL,
                sample_manifest_sha256=manifest_sha,
                sample=sample,
                seed=int(case["row_seed"]),
            )
        )
    except native_probe.Chk3ProbeError as exc:
        raise PaperChk2Core8GenerationError(str(exc)) from exc
    answer = native_eval._final_answer(generated_text)
    signed = native_eval._signed_numeric_metrics(str(case["source_analysis"]), answer)
    structure = native_eval._native_structure_metrics(generated_text)
    quality_failures = list(base.get("quality_failures") or [])
    if not signed["signed_numeric_surface_preserved"]:
        quality_failures.append("signed_numeric_surface_not_preserved")
    quality_failures.extend(structure["native_structure_failures"])
    quality_failures = sorted(set(quality_failures))
    normalized_finish = str(base["finish_reason"])
    row: dict[str, Any] = {
        **base,
        **signed,
        **structure,
        "schema_version": ROW_SCHEMA,
        "evaluation_id": EVALUATION_ID,
        "evaluation_manifest_sha256": manifest_sha,
        "model_id": MODEL_ID,
        "stage_id": MODEL_ID,
        "model_label": MODEL_LABEL,
        "variant_sample_id": case["sample_id"],
        "generation_key": case["generation_key"],
        "absolute_case_index": case["absolute_case_index"],
        "meeting_id": case["meeting_id"],
        "meeting_rank": case["meeting_rank"],
        "variant_rank": case["variant_rank"],
        "arm": case["arm"],
        "intervention_topic": case["intervention_topic"],
        "replicate_id": case["replicate_id"],
        "replicate_seed": case["replicate_seed"],
        "paired_seed_key": case["paired_seed_key"],
        "paired_block_id": case["paired_block_id"],
        "row_seed": case["row_seed"],
        "shard_id": case["shard_id"],
        "physical_gpu_index": case["shard_id"],
        "request_id": (
            f"paper-chk2-core8-case-{int(case['absolute_case_index']):05d}-"
            f"seed-{int(case['row_seed'])}"
        ),
        "generation_mode": "stochastic_sampling",
        "do_sample": True,
        "temperature": common.TEMPERATURE,
        "top_p": common.TOP_P,
        "top_k": common.TOP_K,
        "repetition_penalty": common.REPETITION_PENALTY,
        "max_new_tokens": common.MAX_NEW_TOKENS,
        "input_token_count": case["prompt_token_count"],
        "input_truncated": False,
        "source_prompt_sha256": case["prompt_sha256"],
        "prompt_sha256": case["prompt_sha256"],
        "prompt_token_ids_sha256": case["prompt_token_ids_sha256"],
        "prompt_token_count": case["prompt_token_count"],
        "source_analysis": case["source_analysis"],
        "source_analysis_sha256": case["source_analysis_sha256"],
        "reference_minutes": case["reference_minutes"],
        "reference_minutes_sha256": case["reference_minutes_sha256"],
        "full_source_analysis_sha256": case["full_source_analysis_sha256"],
        "full_reference_sha256": case["full_reference_sha256"],
        "full_prompt_token_count": case["full_prompt_token_count"],
        "completion": generated_text,
        "generated_text": generated_text,
        "completion_sha256": _sha256_text(generated_text),
        "generated_token_ids": token_ids,
        "generated_token_ids_sha256": _token_ids_sha256(token_ids),
        "answer": answer,
        "answer_sha256": _sha256_text(answer),
        "eos_token_ids": list(eos_token_ids),
        "pad_token_id": pad_token_id,
        "finish_reason": normalized_finish,
        "vllm_raw_finish_reason": raw_finish_reason,
        "vllm_raw_stop_reason": raw_stop_reason,
        "quality_valid": not quality_failures,
        "quality_failures": quality_failures,
        "inference_backend": "vllm-async-engine-v1-independent-dp1-dynamic-lora",
        "model_dtype": "bfloat16",
        "quantization": None,
        "load_in_4bit": False,
        "attn_implementation": "vllm-engine-managed",
        "semantic_status": "pending_offline",
        "generation_contract": generation_contract(),
        "lora_request": generation_contract()["lora_request"],
    }
    try:
        metrics, audit = stochastic._generation_metrics(row)
    except stochastic.StochasticBootstrapGenerationError as exc:
        raise PaperChk2Core8GenerationError(str(exc)) from exc
    row["generation_metrics"] = metrics
    row["preregistered_core_valid"] = bool(audit["preregistered_core_valid"])
    row["preregistered_core_failures"] = list(
        audit["preregistered_core_failures"]
    )
    row["hard_gate_audit"] = audit
    row["six_metrics"] = {
        **metrics,
        "mpnet_cosine": None,
        "bertscore_f1": None,
    }
    return row


def _case_index(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        key = str(case["generation_key"])
        _require(key not in indexed, f"duplicate expected key: {key}")
        indexed[key] = case
    return indexed


def _validate_wal_row(
    row: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    manifest_sha: str,
    tokenizer: Any,
    eos_token_ids: Sequence[int],
    pad_token_id: int,
) -> None:
    _require(row.get("schema_version") == ROW_SCHEMA, "WAL row schema drift")
    expected = _build_result(
        case=case,
        generated_text=str(row.get("generated_text") or ""),
        generated_token_ids=row.get("generated_token_ids") or [],
        raw_finish_reason=row.get("vllm_raw_finish_reason"),
        raw_stop_reason=row.get("vllm_raw_stop_reason"),
        eos_token_ids=eos_token_ids,
        pad_token_id=pad_token_id,
        manifest_sha=manifest_sha,
        tokenizer=tokenizer,
    )
    _require(dict(row) == expected, "persisted WAL row deep-replay drift")


def load_wal(
    path: Path,
    *,
    cases_by_key: Mapping[str, Mapping[str, Any]],
    manifest_sha: str,
    tokenizer: Any,
    eos_token_ids: Sequence[int],
    pad_token_id: int,
) -> dict[str, dict[str, Any]]:
    """Load a durable WAL, repairing only one final unterminated fragment."""

    if not path.exists():
        return {}
    _require(path.is_file() and not path.is_symlink(), "unsafe generation WAL")
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        recovered_bytes = payload.rfind(b"\n") + 1
        discarded = payload[recovered_bytes:]
        _require(bool(discarded), "WAL torn-tail detection drift")
        with path.open("r+b") as handle:
            handle.truncate(recovered_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        ledger_path = path.with_name(f"{path.name}.torn-tail-recoveries.jsonl")
        _require(not ledger_path.is_symlink(), "unsafe torn-tail ledger")
        with ledger_path.open("a", encoding="utf-8") as handle:
            durable._append_fsync(
                handle,
                {
                    "schema_version": TORN_TAIL_SCHEMA,
                    "wal_path": str(path.resolve()),
                    "original_bytes": len(payload),
                    "recovered_bytes": recovered_bytes,
                    "discarded_bytes": len(discarded),
                    "discarded_sha256": hashlib.sha256(discarded).hexdigest(),
                    "policy": "discard_only_final_unterminated_jsonl_fragment",
                },
            )
    rows = durable._read_jsonl(path, label="paper chk2 Core8 generation WAL")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("generation_key")
        _require(isinstance(key, str) and key in cases_by_key, f"unknown WAL key: {key}")
        _require(key not in indexed, f"duplicate WAL key: {key}")
        _validate_wal_row(
            row,
            cases_by_key[key],
            manifest_sha=manifest_sha,
            tokenizer=tokenizer,
            eos_token_ids=eos_token_ids,
            pad_token_id=pad_token_id,
        )
        indexed[key] = row
    return indexed


def _validate_visible_gpu(shard_id: int) -> None:
    _require(
        os.environ.get("CUDA_DEVICE_ORDER") == "PCI_BUS_ID",
        "CUDA_DEVICE_ORDER must be PCI_BUS_ID",
    )
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == str(shard_id),
        f"CUDA_VISIBLE_DEVICES must be exactly {shard_id}",
    )


@contextlib.contextmanager
def _worker_lock(shard_id: int) -> Iterable[Mapping[str, Any]]:
    path = Path(LOCK_TEMPLATE.format(shard=shard_id))
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    handle = os.fdopen(descriptor, "a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PaperChk2Core8GenerationError(
                f"paper chk2 Core8 GPU{shard_id} lock is held"
            ) from exc
        yield {"path": str(path), "pid": os.getpid()}
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


async def _run_cp50_engine(
    *,
    parent_model_path: Path,
    adapter_path: Path,
    tokenizer_path: Path,
    tokenizer: Any,
    eos_token_ids: Sequence[int],
    pad_token_id: int,
    cases: Sequence[Mapping[str, Any]],
    manifest_sha: str,
    gpu_memory_utilization: float,
    on_result: Any,
) -> Mapping[str, Any]:
    import torch
    import vllm
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.lora.request import LoRARequest

    _require(vllm.__version__ == EXPECTED_VLLM_VERSION, "vLLM version drift")
    _require(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "worker requires exactly one visible CUDA GPU",
    )
    args = AsyncEngineArgs(
        model=str(parent_model_path),
        tokenizer=str(tokenizer_path),
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        quantization=None,
        load_format="safetensors",
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        data_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        swap_space=0,
        cpu_offload_gb=0,
        max_model_len=common.MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        generation_config="vllm",
        seed=0,
        disable_log_requests=True,
        disable_log_stats=True,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        max_cpu_loras=1,
        lora_dtype="bfloat16",
    )
    started_load = time.perf_counter()
    engine = AsyncLLMEngine.from_engine_args(args)
    load_seconds = time.perf_counter() - started_load
    lora_request = LoRARequest(
        "paper-chk2-cp50", 50, lora_path=str(adapter_path)
    )

    async def consume(case: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
        params = SamplingParams(
            n=1,
            temperature=common.TEMPERATURE,
            top_p=common.TOP_P,
            top_k=common.TOP_K,
            repetition_penalty=common.REPETITION_PENALTY,
            max_tokens=common.MAX_NEW_TOKENS,
            seed=int(case["row_seed"]),
            detokenize=True,
            skip_special_tokens=False,
            spaces_between_special_tokens=True,
        )
        request_id = (
            f"paper-chk2-core8-{int(case['absolute_case_index']):05d}-"
            f"{int(case['row_seed'])}"
        )
        final = None
        async for output in engine.generate(
            {"prompt_token_ids": list(case["prompt_token_ids"])},
            params,
            request_id,
            lora_request=lora_request,
        ):
            final = output
        _require(final is not None, f"vLLM returned no output: {request_id}")
        _require(len(getattr(final, "outputs", []) or []) == 1, "vLLM n drift")
        _require(
            list(getattr(final, "prompt_token_ids", []) or [])
            == list(case["prompt_token_ids"]),
            "vLLM prompt-token drift",
        )
        return case, final.outputs[0]

    started_generation = time.perf_counter()
    generated_rows = 0
    try:
        for start in range(0, len(cases), 64):
            tasks = [
                asyncio.create_task(consume(case))
                for case in cases[start : start + 64]
            ]
            try:
                for future in asyncio.as_completed(tasks):
                    case, completion = await future
                    token_ids = list(getattr(completion, "token_ids", []) or [])
                    try:
                        text = native_probe.common_probe.decode_completion_preserving_boundary(
                            tokenizer, token_ids, eos_token_ids
                        )
                    except native_probe.common_probe.ProbeError as exc:
                        raise PaperChk2Core8GenerationError(str(exc)) from exc
                    _require(
                        getattr(completion, "text", None) == text,
                        "vLLM completion text/token drift",
                    )
                    row = _build_result(
                        case=case,
                        generated_text=text,
                        generated_token_ids=token_ids,
                        raw_finish_reason=getattr(completion, "finish_reason", None),
                        raw_stop_reason=getattr(completion, "stop_reason", None),
                        eos_token_ids=eos_token_ids,
                        pad_token_id=pad_token_id,
                        manifest_sha=manifest_sha,
                        tokenizer=tokenizer,
                    )
                    await on_result(row)
                    generated_rows += 1
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        durable._shutdown_engine(engine)
    return {
        "base_model_path": str(parent_model_path),
        "adapter_path": str(adapter_path),
        "enable_lora": True,
        "lora_request": generation_contract()["lora_request"],
        "model_load_wall_seconds": load_seconds,
        "generation_wall_seconds": time.perf_counter() - started_generation,
        "new_rows": generated_rows,
    }


def _file_binding(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result = {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        result["rows"] = rows
    return result


def _write_canonical(
    path: Path,
    *,
    rows_by_key: Mapping[str, Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    encoded = "".join(
        _canonical(rows_by_key[str(case["generation_key"])]) + "\n"
        for case in cases
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.is_file() and not path.is_symlink(), "unsafe canonical JSONL")
        _require(path.read_bytes() == encoded, "canonical JSONL content drift")
    else:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    return _file_binding(path, rows=len(cases))


def _smoke_authorization(
    *, output_root: Path, manifest_sha: str, explicit_path: Path | None
) -> Mapping[str, Any]:
    path = (
        explicit_path.expanduser().resolve()
        if explicit_path is not None
        else _mode_root(output_root, smoke=True) / "validation.json"
    )
    value = durable._read_json(path, label="paper chk2 Core8 smoke authorization")
    _require(value.get("schema_version") == VALIDATION_SCHEMA, "smoke schema drift")
    _require(value.get("status") == "complete", "smoke is not complete")
    _require(value.get("mode") == "smoke", "formal authorization is not smoke")
    _require(value.get("evaluation_manifest_sha256") == manifest_sha, "smoke manifest drift")
    _require(value.get("generation_rows") == SMOKE_MEETINGS * VARIANTS_PER_MEETING * SMOKE_REPLICATES, "smoke row count drift")
    gates = value.get("gates")
    _require(isinstance(gates, Mapping) and all(bool(item) for item in gates.values()), "smoke authorization gates failed")
    return {"artifact": _file_binding(path), "gates": dict(gates)}


def _launch_contract(
    *,
    manifest_path: Path,
    manifest_sha: str,
    bindings: Any,
    shard_id: int,
    smoke: bool,
    smoke_meetings: int,
    formal_authorization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": LAUNCH_SCHEMA,
        "evaluation_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha,
        },
        "model_files": dict(bindings.file_bindings),
        "mode": "smoke" if smoke else "formal",
        "smoke_meetings": smoke_meetings if smoke else None,
        "shard_id": shard_id,
        "physical_gpu_index": shard_id,
        "generation_contract": generation_contract(),
        "formal_authorization": formal_authorization,
    }


def run_worker(
    *,
    manifest_path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    shard_id: int,
    smoke: bool = False,
    smoke_meetings: int = SMOKE_MEETINGS,
    resume: bool = False,
    formal_authorization: Path | None = None,
) -> Mapping[str, Any]:
    """Run one GPU-pinned shard; formal mode requires validated smoke evidence."""

    _require(shard_id in SHARD_IDS, "shard ID must be 0 or 1")
    _validate_visible_gpu(shard_id)
    output_root = output_root.expanduser().resolve()
    source, bindings, _manifest, ledger, manifest_sha = _load_context(
        output_root=output_root, manifest_path=manifest_path
    )
    tokenizer, eos_ids, pad_id = _load_tokenizer(bindings)
    cases = build_case_matrix(
        source, ledger, smoke=smoke, smoke_meetings=smoke_meetings
    )
    cases = [case for case in cases if int(case["shard_id"]) == shard_id]
    expected_rows = (
        smoke_meetings * VARIANTS_PER_MEETING * SMOKE_REPLICATES // 2
        if smoke
        else EXPECTED_GENERATIONS // 2
    )
    _require(len(cases) == expected_rows, "per-shard case count drift")
    case_index = _case_index(cases)
    authorization = (
        None
        if smoke
        else _smoke_authorization(
            output_root=output_root,
            manifest_sha=manifest_sha,
            explicit_path=formal_authorization,
        )
    )
    _require(
        smoke or authorization is not None,
        "formal generation requires smoke authorization",
    )
    mode_root = _mode_root(output_root, smoke=smoke)
    shard_root = mode_root / f"shard-{shard_id}"
    launch_path = shard_root / "launch.json"
    wal_path = shard_root / "generations.wal.jsonl"
    canonical_path = shard_root / "generations.canonical.jsonl"
    state_path = shard_root / "state.json"
    shard_manifest_path = shard_root / "manifest.json"
    expected_launch = _launch_contract(
        manifest_path=manifest_path,
        manifest_sha=manifest_sha,
        bindings=bindings,
        shard_id=shard_id,
        smoke=smoke,
        smoke_meetings=smoke_meetings,
        formal_authorization=authorization,
    )
    if shard_root.exists():
        _require(resume, f"shard output exists; pass --resume: {shard_root}")
        _require(shard_root.is_dir() and not shard_root.is_symlink(), "unsafe shard root")
        observed_launch = durable._read_json(launch_path, label="Core8 worker launch")
        _require(observed_launch.get("contract") == expected_launch, "resume launch contract drift")
    else:
        shard_root.mkdir(parents=True, exist_ok=False)
        durable._write_new_json(
            launch_path,
            {
                "contract": expected_launch,
                "runtime": {
                    "python": sys.executable,
                    "python_version": platform.python_version(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
            },
        )
    rows = load_wal(
        wal_path,
        cases_by_key=case_index,
        manifest_sha=manifest_sha,
        tokenizer=tokenizer,
        eos_token_ids=eos_ids,
        pad_token_id=pad_id,
    )
    if len(rows) == len(cases):
        canonical = _write_canonical(
            canonical_path, rows_by_key=rows, cases=cases
        )
        result = {
            "schema_version": SHARD_SCHEMA,
            "status": "complete",
            "mode": "smoke" if smoke else "formal",
            "shard_id": shard_id,
            "expected_rows": len(cases),
            "completed_rows": len(rows),
            "resume_noop": True,
            "wal": _file_binding(wal_path, rows=len(rows)),
            "canonical": canonical,
        }
        durable._atomic_json(shard_manifest_path, result)
        return result

    state: dict[str, Any] = {
        "schema_version": SHARD_SCHEMA,
        "status": "running",
        "shard_id": shard_id,
        "expected_rows": len(cases),
        "resumed_rows": len(rows),
        "completed_rows": len(rows),
    }
    durable._atomic_json(state_path, state)
    pending = [
        case for case in cases if str(case["generation_key"]) not in rows
    ]
    with _worker_lock(shard_id) as lock:
        state["worker_lock"] = dict(lock)
        snapshot = durable.gpu_snapshot(shard_id)
        _require(not wal_path.is_symlink(), "unsafe WAL symlink")
        wal_path.parent.mkdir(parents=True, exist_ok=True)
        with wal_path.open("a", encoding="utf-8") as handle:

            async def on_result(row: Mapping[str, Any]) -> None:
                key = str(row["generation_key"])
                _require(key not in rows, f"duplicate newly generated key: {key}")
                durable._append_fsync(handle, row)
                rows[key] = dict(row)
                state["completed_rows"] = len(rows)
                if len(rows) % 16 == 0 or len(rows) == len(cases):
                    durable._atomic_json(state_path, state)

            report = asyncio.run(
                _run_cp50_engine(
                    parent_model_path=bindings.parent_model_path,
                    adapter_path=bindings.adapter_path,
                    tokenizer_path=bindings.tokenizer_path,
                    tokenizer=tokenizer,
                    eos_token_ids=eos_ids,
                    pad_token_id=pad_id,
                    cases=pending,
                    manifest_sha=manifest_sha,
                    gpu_memory_utilization=float(
                        snapshot["vllm_gpu_memory_utilization"]
                    ),
                    on_result=on_result,
                )
            )
    _require(len(rows) == len(cases), "worker ended without shard closure")
    canonical = _write_canonical(canonical_path, rows_by_key=rows, cases=cases)
    result = {
        "schema_version": SHARD_SCHEMA,
        "status": "complete",
        "mode": "smoke" if smoke else "formal",
        "shard_id": shard_id,
        "expected_rows": len(cases),
        "completed_rows": len(rows),
        "resumed_rows": state["resumed_rows"],
        "new_rows": len(rows) - int(state["resumed_rows"]),
        "gpu_before": snapshot,
        "engine": report,
        "wal": _file_binding(wal_path, rows=len(rows)),
        "canonical": canonical,
    }
    durable._atomic_json(shard_manifest_path, result)
    state["status"] = "complete"
    state["engine"] = report
    durable._atomic_json(state_path, state)
    gc.collect()
    return result


def _write_or_verify_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        _require(path.is_file() and not path.is_symlink(), "unsafe validation artifact")
        _require(
            durable._read_json(path, label="Core8 validation") == dict(value),
            "existing validation artifact drift",
        )
    else:
        durable._write_new_json(path, value)


def validate_run(
    *,
    manifest_path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    smoke: bool = False,
    smoke_meetings: int = SMOKE_MEETINGS,
) -> Mapping[str, Any]:
    """Deep-replay both WALs and publish canonical two-shard closure evidence."""

    output_root = output_root.expanduser().resolve()
    source, bindings, _manifest, ledger, manifest_sha = _load_context(
        output_root=output_root, manifest_path=manifest_path
    )
    tokenizer, eos_ids, pad_id = _load_tokenizer(bindings)
    cases = build_case_matrix(
        source, ledger, smoke=smoke, smoke_meetings=smoke_meetings
    )
    expected = _case_index(cases)
    combined: dict[str, Mapping[str, Any]] = {}
    shard_bindings: dict[str, Any] = {}
    mode_root = _mode_root(output_root, smoke=smoke)
    for shard_id in SHARD_IDS:
        selected = [case for case in cases if int(case["shard_id"]) == shard_id]
        selected_index = _case_index(selected)
        shard_root = mode_root / f"shard-{shard_id}"
        wal_path = shard_root / "generations.wal.jsonl"
        rows = load_wal(
            wal_path,
            cases_by_key=selected_index,
            manifest_sha=manifest_sha,
            tokenizer=tokenizer,
            eos_token_ids=eos_ids,
            pad_token_id=pad_id,
        )
        _require(rows.keys() == selected_index.keys(), f"shard-{shard_id} incomplete")
        for key, row in rows.items():
            _require(key not in combined, f"cross-shard duplicate: {key}")
            combined[key] = row
        canonical = _write_canonical(
            shard_root / "generations.canonical.jsonl",
            rows_by_key=rows,
            cases=selected,
        )
        shard_bindings[f"shard-{shard_id}"] = {
            "rows": len(rows),
            "wal": _file_binding(wal_path, rows=len(rows)),
            "canonical": canonical,
        }
    _require(combined.keys() == expected.keys(), "two-shard union is incomplete")

    blocks: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in combined.values():
        key = (str(row["meeting_id"]), int(row["replicate_id"]))
        blocks.setdefault(key, []).append(row)
    for key, rows in blocks.items():
        _require(len(rows) == VARIANTS_PER_MEETING, f"paired block size drift: {key}")
        _require(len({int(row["shard_id"]) for row in rows}) == 1, f"paired block crossed GPUs: {key}")
        _require(len({int(row["row_seed"]) for row in rows}) == 1, f"paired seed drift: {key}")
        _require({int(row["variant_rank"]) for row in rows} == set(range(VARIANTS_PER_MEETING)), f"variant closure drift: {key}")

    combined_path = mode_root / "generation_rows.jsonl"
    combined_binding = _write_canonical(
        combined_path, rows_by_key=combined, cases=cases
    )
    metric_counts = {
        name: sum(bool(row["generation_metrics"][name]) for row in combined.values())
        for name in (
            "structure_delivery",
            "numeric_fidelity",
            "date_fidelity",
            "degeneration_free",
        )
    }
    core_pass = sum(
        bool(row["preregistered_core_valid"]) for row in combined.values()
    )
    finish_counts = Counter(str(row["finish_reason"]) for row in combined.values())
    result = {
        "schema_version": VALIDATION_SCHEMA,
        "status": "complete",
        "mode": "smoke" if smoke else "formal",
        "evaluation_id": EVALUATION_ID,
        "evaluation_manifest": str(manifest_path.expanduser().resolve()),
        "evaluation_manifest_sha256": manifest_sha,
        "model_id": MODEL_ID,
        "model_label": MODEL_LABEL,
        "generation_rows": len(combined),
        "meetings": smoke_meetings if smoke else EXPECTED_MEETINGS,
        "prompts": (smoke_meetings * VARIANTS_PER_MEETING if smoke else EXPECTED_PROMPTS),
        "replicates": SMOKE_REPLICATES if smoke else REPLICATES,
        "paired_blocks": len(blocks),
        "shards": shard_bindings,
        "combined_generation_rows": combined_binding,
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "hard_gate_funnel": {
            **metric_counts,
            "joint_core_pass": core_pass,
            "joint_core_fail": len(combined) - core_pass,
        },
        "generation_contract": generation_contract(),
        "gates": {
            "all_expected_keys_present": True,
            "no_duplicate_or_cross_shard_keys": True,
            "all_17_arms_colocated_per_meeting_replicate": True,
            "common_row_seed_within_each_paired_block": True,
            "all_rows_bound_to_cp50_dynamic_lora": all(
                row.get("lora_request") == generation_contract()["lora_request"]
                for row in combined.values()
            ),
            "all_rows_deep_replayed_from_text_and_token_ids": True,
            "wal_rows_fsynced_by_worker": True,
        },
    }
    _require(all(result["gates"].values()), "run validation gate failed")
    validation_path = mode_root / "validation.json"
    _write_or_verify_json(validation_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    prepare.add_argument("--resume", action="store_true")
    for name in ("worker", "smoke", "formal"):
        child = commands.add_parser(name)
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        child.add_argument("--shard-id", type=int, choices=SHARD_IDS, required=True)
        child.add_argument("--resume", action="store_true")
        child.add_argument("--smoke-meetings", type=int, default=SMOKE_MEETINGS)
        child.add_argument("--formal-authorization", type=Path)
        if name == "worker":
            child.add_argument("--smoke", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    validate.add_argument("--smoke", action="store_true")
    validate.add_argument("--smoke-meetings", type=int, default=SMOKE_MEETINGS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_evaluation(args.output_root, resume=args.resume)
    elif args.command == "validate":
        result = validate_run(
            manifest_path=args.manifest,
            output_root=args.output_root,
            smoke=args.smoke,
            smoke_meetings=args.smoke_meetings,
        )
    else:
        smoke = args.command == "smoke" or (
            args.command == "worker" and bool(args.smoke)
        )
        if args.command == "formal":
            smoke = False
        result = run_worker(
            manifest_path=args.manifest,
            output_root=args.output_root,
            shard_id=args.shard_id,
            smoke=smoke,
            smoke_meetings=args.smoke_meetings,
            resume=args.resume,
            formal_authorization=args.formal_authorization,
        )
    print(_canonical(result), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
