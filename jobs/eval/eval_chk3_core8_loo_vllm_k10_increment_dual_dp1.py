"""Run the create-only CHK3 Core8 LOO K=6--K=10 vLLM increment.

This is a thin profile over the frozen K=5 dual-DP1 implementation.  It
generates only five fresh replicates.  Local replicate IDs 0--4 retain the
audited sharding mechanics; global IDs 5--9 make the rows safe to combine
with the immutable predecessor release.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.eval import eval_chk3_core8_loo_vllm_k5_dual_dp1 as frozen
from jobs.eval import prepare_chk3_core8_loo_vllm_k10_increment as preparation


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ID = preparation.EVALUATION_ID
MODEL_ORDER = frozen.MODEL_ORDER
MODEL_LABELS = frozen.MODEL_LABELS
REPLICATE_SEEDS = preparation.REPLICATE_SEEDS
LOCAL_REPLICATE_IDS = preparation.LOCAL_REPLICATE_IDS
GLOBAL_REPLICATE_OFFSET = preparation.GLOBAL_REPLICATE_OFFSET
GLOBAL_REPLICATE_IDS = preparation.GLOBAL_REPLICATE_IDS

EXPECTED_PROMPTS = preparation.EXPECTED_PROMPTS
EXPECTED_MEETINGS = preparation.EXPECTED_MEETINGS
VARIANTS_PER_MEETING = preparation.VARIANTS_PER_MEETING
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

_SCHEMA_PREFIX = "chk3-core8-loo-vllm-k10-increment-k6-k10-dual-dp1"
ROW_SCHEMA = f"{_SCHEMA_PREFIX}-generation-row-v1"
SHARD_MANIFEST_SCHEMA = f"{_SCHEMA_PREFIX}-shard-manifest-v1"
MERGED_MANIFEST_SCHEMA = f"{_SCHEMA_PREFIX}-merged-manifest-v1"
STATE_SCHEMA = f"{_SCHEMA_PREFIX}-state-v1"
CHUNK_RECEIPT_SCHEMA = f"{_SCHEMA_PREFIX}-chunk-receipt-v1"
READY_SCHEMA = f"{_SCHEMA_PREFIX}-ready-v1"
GO_SCHEMA = f"{_SCHEMA_PREFIX}-go-v1"
DONE_SCHEMA = f"{_SCHEMA_PREFIX}-done-v1"
ORCHESTRATOR_TIMING_SCHEMA = f"{_SCHEMA_PREFIX}-timing-v1"

TEMPERATURE = frozen.TEMPERATURE
TOP_P = frozen.TOP_P
TOP_K = frozen.TOP_K
REPETITION_PENALTY = frozen.REPETITION_PENALTY
MAX_NEW_TOKENS = frozen.MAX_NEW_TOKENS
TAIL_TOKENS = frozen.TAIL_TOKENS
MAX_MODEL_LEN = frozen.MAX_MODEL_LEN
SELECTED_MAX_NUM_SEQS = frozen.SELECTED_MAX_NUM_SEQS
ALLOWED_MAX_NUM_SEQS = frozen.ALLOWED_MAX_NUM_SEQS
MAX_NUM_BATCHED_TOKENS = frozen.MAX_NUM_BATCHED_TOKENS
GPU_MEMORY_UTILIZATION = frozen.GPU_MEMORY_UTILIZATION
DATA_PARALLEL_SIZE = frozen.DATA_PARALLEL_SIZE
TENSOR_PARALLEL_SIZE = frozen.TENSOR_PARALLEL_SIZE
PIPELINE_PARALLEL_SIZE = frozen.PIPELINE_PARALLEL_SIZE
EXPECTED_VLLM_VERSION = frozen.EXPECTED_VLLM_VERSION
# A physical GPU has one canonical cross-pipeline lease.  Reuse the existing
# K5/semantic/DP2 namespace so a separate create-only result root cannot run
# concurrently on the same cards under a different lock filename.
GPU_LOCK_PATH_TEMPLATE = frozen.shared.GPU_LOCK_PATH_TEMPLATE

DEFAULT_RUN_ROOT = preparation.DEFAULT_RUN_ROOT
DEFAULT_COHORT = preparation.DEFAULT_OUTPUT_DIR / preparation.COHORT_FILENAME
DEFAULT_EXECUTION_POLICY = (
    preparation.DEFAULT_OUTPUT_DIR / preparation.EXECUTION_POLICY_FILENAME
)


class Core8LooVllmK10IncrementError(frozen.Core8LooVllmK5Error):
    """The frozen five-replicate incremental contract was violated."""


_FROZEN_SAMPLING_CONTRACT = frozen.sampling_contract
_FROZEN_LOAD_EXECUTION_POLICY = frozen.load_execution_policy
_FROZEN_LOAD_COHORT = frozen.load_cohort
_FROZEN_CANONICAL_CASES = frozen.canonical_cases
_FROZEN_BUILD_RESULT = frozen.build_result
_FROZEN_CONSUME_REQUEST = frozen._consume_request
_FROZEN_PERSISTENCE_CONTRACT = frozen._persistence_contract
_FROZEN_CHUNK_RECEIPT = frozen._chunk_receipt
_FROZEN_SOURCE_HASHES = frozen._source_hashes
_FROZEN_SHARED_REPLACEMENTS = frozen._shared_replacements
_FROZEN_ASSIGNED_SHARD = frozen.assigned_shard
_FROZEN_SHARD_CASES = frozen.shard_cases
_FROZEN_PAIRED_BLOCK_ID = frozen.paired_block_id
_FROZEN_VALIDATE_MAX_NUM_SEQS = frozen._validate_max_num_seqs


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_text(value: str) -> str:
    return frozen._sha256_text(value)


def _binding(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    return frozen._binding(path, rows=rows)


def _load_execution_policy_for_frozen_base(
    path: Path = DEFAULT_EXECUTION_POLICY,
) -> dict[str, Any]:
    """Call the frozen validator with the increment path bound explicitly.

    The K5 function's original default argument points at the immutable K5
    policy even when its module globals are scoped.  Frozen functions that
    call ``load_execution_policy()`` without arguments must therefore resolve
    to this adapter, whose default was bound to the increment at definition.
    """

    return _FROZEN_LOAD_EXECUTION_POLICY(path)


def _frozen_replacements() -> dict[str, Any]:
    return {
        "__doc__": __doc__,
        "EVALUATION_ID": EVALUATION_ID,
        "MODEL_ORDER": MODEL_ORDER,
        "MODEL_LABELS": MODEL_LABELS,
        "REPLICATE_SEEDS": REPLICATE_SEEDS,
        "EXPECTED_PROMPTS": EXPECTED_PROMPTS,
        "EXPECTED_MEETINGS": EXPECTED_MEETINGS,
        "VARIANTS_PER_MEETING": VARIANTS_PER_MEETING,
        "REPLICATES": REPLICATES,
        "CASES_PER_MEETING": CASES_PER_MEETING,
        "CASES_PER_MODEL": CASES_PER_MODEL,
        "ABSOLUTE_CHUNK_SIZE": ABSOLUTE_CHUNK_SIZE,
        "CHUNKS_PER_MODEL": CHUNKS_PER_MODEL,
        "SHARD_COUNT": SHARD_COUNT,
        "CASES_PER_SHARD_CHUNK": CASES_PER_SHARD_CHUNK,
        "FORMAL_CASES_PER_SHARD": FORMAL_CASES_PER_SHARD,
        "SMOKE_MEETINGS": SMOKE_MEETINGS,
        "SMOKE_PROMPTS": SMOKE_PROMPTS,
        "SMOKE_CASES_PER_MODEL": SMOKE_CASES_PER_MODEL,
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
        "DEFAULT_RUN_ROOT": DEFAULT_RUN_ROOT,
        "DEFAULT_COHORT": DEFAULT_COHORT,
        "DEFAULT_EXECUTION_POLICY": DEFAULT_EXECUTION_POLICY,
        "preparation": preparation,
        "load_execution_policy": _load_execution_policy_for_frozen_base,
    }


@contextlib.contextmanager
def configured_frozen_runner() -> Any:
    replacements = _frozen_replacements()
    previous = {name: getattr(frozen, name) for name in replacements}
    for name, value in replacements.items():
        setattr(frozen, name, value)
    try:
        yield frozen
    finally:
        for name, value in previous.items():
            setattr(frozen, name, value)


def _validate_max_num_seqs(value: int) -> int:
    with configured_frozen_runner():
        return _FROZEN_VALIDATE_MAX_NUM_SEQS(value)


def sampling_contract(*, max_num_seqs: int) -> dict[str, Any]:
    with configured_frozen_runner():
        contract = copy.deepcopy(
            _FROZEN_SAMPLING_CONTRACT(max_num_seqs=max_num_seqs)
        )
    contract["replicate_indexing"] = preparation.replicate_indexing_contract()
    contract["increment_authorization"] = (
        "2026-08-24:user-requested-five-fresh-replicates-to-extend-k5-to-k10"
    )
    return contract


def _sampling_contract_sha256(*, max_num_seqs: int) -> str:
    return _sha256_text(_canonical(sampling_contract(max_num_seqs=max_num_seqs)))


def paired_block_id(meeting_rank: int, replicate_id: int) -> int:
    with configured_frozen_runner():
        return _FROZEN_PAIRED_BLOCK_ID(meeting_rank, replicate_id)


def assigned_shard(absolute_case_index: int) -> int:
    with configured_frozen_runner():
        return _FROZEN_ASSIGNED_SHARD(absolute_case_index)


def shard_cases(
    cases: Sequence[Mapping[str, Any]], shard_id: int
) -> list[Mapping[str, Any]]:
    with configured_frozen_runner():
        return _FROZEN_SHARD_CASES(cases, shard_id)


def _validate_increment_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("replicate_indexing") != preparation.replicate_indexing_contract():
        raise Core8LooVllmK10IncrementError("replicate-indexing contract drift")
    if manifest.get("increment_authorization") != preparation.increment_authorization():
        raise Core8LooVllmK10IncrementError("increment authorization drift")
    prior = manifest.get("prior_k5_release")
    if not isinstance(prior, Mapping):
        raise Core8LooVllmK10IncrementError("prior K5 binding is missing")
    expected = preparation.prior_k5_bindings()
    if dict(prior) != expected:
        raise Core8LooVllmK10IncrementError("prior K5 binding drift")
    excluded = manifest.get("excluded_incomplete_nf4_k10")
    if (
        not isinstance(excluded, Mapping)
        or dict(excluded) != preparation.excluded_incomplete_nf4_k10_bindings()
    ):
        raise Core8LooVllmK10IncrementError(
            "excluded incomplete NF4 K10 binding drift"
        )


def load_execution_policy(
    path: Path = DEFAULT_EXECUTION_POLICY,
) -> dict[str, Any]:
    with configured_frozen_runner():
        result = _FROZEN_LOAD_EXECUTION_POLICY(path)
    _validate_increment_manifest(result["manifest"])
    return result


def load_cohort(
    cohort_path: Path, cohort_sha256: str
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    with configured_frozen_runner():
        result = _FROZEN_LOAD_COHORT(cohort_path, cohort_sha256)
    _validate_increment_manifest(result[0])
    return result


def canonical_cases(
    *,
    samples: Sequence[Mapping[str, Any]],
    ledger: Sequence[Mapping[str, Any]],
    smoke: bool,
) -> list[dict[str, Any]]:
    with configured_frozen_runner():
        cases = _FROZEN_CANONICAL_CASES(
            samples=samples,
            ledger=ledger,
            smoke=smoke,
        )
    for case in cases:
        local_id = int(case["replicate_id"])
        case["local_replicate_id"] = local_id
        case["global_replicate_id"] = GLOBAL_REPLICATE_OFFSET + local_id
    return cases


def build_result(**kwargs: Any) -> dict[str, Any]:
    with configured_frozen_runner():
        row = _FROZEN_BUILD_RESULT(**kwargs)
    case = kwargs["case"]
    local_id = int(case["replicate_id"])
    row["local_replicate_id"] = local_id
    row["global_replicate_id"] = GLOBAL_REPLICATE_OFFSET + local_id
    row["global_replicate_offset"] = GLOBAL_REPLICATE_OFFSET
    row["sampling_contract_sha256"] = _sampling_contract_sha256(
        max_num_seqs=int(kwargs["max_num_seqs"])
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
            key
            for key in set(row) | set(expected)
            if row.get(key) != expected.get(key)
        )
        raise Core8LooVllmK10IncrementError(
            f"persisted increment row drift at fields: {changed[:12]}"
        )


async def _consume_request(engine: Any, sampling_params_cls: Any, case: Mapping[str, Any]) -> tuple[Mapping[str, Any], Any]:
    # This frozen request primitive only reads decoding constants that are
    # identical in K5 and in the increment.  Do not install process-global
    # module substitutions around an ``await``: up to 16 requests overlap in
    # one event loop, and non-LIFO context exits could otherwise restore stale
    # globals into the worker process.
    return await _FROZEN_CONSUME_REQUEST(engine, sampling_params_cls, case)


def _persistence_contract() -> dict[str, Any]:
    with configured_frozen_runner():
        contract = copy.deepcopy(_FROZEN_PERSISTENCE_CONTRACT())
    contract["replicate_indexing"] = preparation.replicate_indexing_contract()
    return contract


def _chunk_receipt(**kwargs: Any) -> dict[str, Any]:
    with configured_frozen_runner():
        receipt = _FROZEN_CHUNK_RECEIPT(**kwargs)
    receipt["global_replicate_offset"] = GLOBAL_REPLICATE_OFFSET
    receipt["global_replicate_ids"] = list(GLOBAL_REPLICATE_IDS)
    return receipt


def _source_hashes(
    cohort_path: Path,
    cohort_sha256: str,
    cohort: Mapping[str, Any],
    sample_manifest: Mapping[str, Any],
    max_num_seqs: int,
    formal_authorization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    with configured_frozen_runner():
        result = _FROZEN_SOURCE_HASHES(
            cohort_path,
            cohort_sha256,
            cohort,
            sample_manifest,
            max_num_seqs,
            formal_authorization,
        )
    result.pop("loo_vllm_k5_cohort_sha256", None)
    result.pop("loo_vllm_k5_token_ledger_sha256", None)
    result.pop("loo_vllm_k5_sampling_contract_sha256", None)
    result["loo_vllm_k10_increment_cohort_sha256"] = cohort_sha256
    result["loo_vllm_k10_increment_token_ledger_sha256"] = cohort[
        "token_ledger"
    ]["sha256"]
    result["loo_vllm_k10_increment_sampling_contract_sha256"] = (
        _sampling_contract_sha256(max_num_seqs=max_num_seqs)
    )
    result["replicate_indexing"] = preparation.replicate_indexing_contract()
    sources = dict(result.get("implementation_sources") or {})
    sources.pop("loo_vllm_k5_runner", None)
    sources.pop("loo_vllm_k5_preparer", None)
    sources.pop("loo_dual_dp1_orchestrator", None)
    sources.pop("pipeline_launcher", None)
    sources["loo_vllm_k10_increment_runner_adapter"] = _binding(
        Path(__file__).resolve()
    )
    sources["frozen_loo_vllm_k5_runner_base"] = _binding(
        Path(frozen.__file__).resolve()
    )
    sources["loo_vllm_k10_increment_preparer_adapter"] = _binding(
        Path(preparation.__file__).resolve()
    )
    sources["frozen_loo_vllm_k5_preparer_base"] = _binding(
        Path(preparation.frozen.__file__).resolve()
    )
    sources["loo_vllm_k10_increment_orchestrator"] = _binding(
        ROOT / "jobs/eval/orchestrate_chk3_core8_loo_vllm_k10_increment_dual_dp1.py"
    )
    sources["loo_vllm_k10_increment_pipeline_launcher"] = _binding(
        ROOT / "run/eval_chk3_core8_loo_vllm_k10_increment_dual_dp1.sh"
    )
    result["implementation_sources"] = sources
    return result


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
        raise Core8LooVllmK10IncrementError(
            "increment smoke authorization gates failed"
        )
    return {
        "official_smoke_run": copy.deepcopy(loaded["manifest_binding"]),
        "execution_policy": load_execution_policy()["receipt_binding"],
        "replicate_indexing": preparation.replicate_indexing_contract(),
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
            raise Core8LooVllmK10IncrementError(
                "smoke must not carry formal authorization"
            )
        return None
    if scope != "formal_merged_panel" or authorization_path is None:
        raise Core8LooVllmK10IncrementError(
            "formal generation requires the validated increment smoke run"
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
            raise Core8LooVllmK10IncrementError(
                "smoke authorization must be null"
            )
        return None
    if not isinstance(authorization, Mapping):
        raise Core8LooVllmK10IncrementError("formal authorization is missing")
    binding = authorization.get("official_smoke_run")
    if not isinstance(binding, Mapping):
        raise Core8LooVllmK10IncrementError("formal smoke binding is missing")
    path = frozen.shared._verify_arbitrary_binding(binding)
    observed = _load_formal_authorization(
        path,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
        max_num_seqs=max_num_seqs,
    )
    if dict(authorization) != observed:
        raise Core8LooVllmK10IncrementError("formal authorization drift")
    return observed


def _shared_replacements() -> dict[str, Any]:
    with configured_frozen_runner():
        replacements = _FROZEN_SHARED_REPLACEMENTS()
    replacements.update(
        {
            "__doc__": __doc__,
            "EVALUATION_ID": EVALUATION_ID,
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
            "GPU_LOCK_PATH_TEMPLATE": GPU_LOCK_PATH_TEMPLATE,
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
            "_revalidate_formal_authorization": (
                _revalidate_formal_authorization
            ),
        }
    )
    return replacements


def configure_shared_implementation() -> dict[str, Any]:
    replacements = _shared_replacements()
    shared = frozen.shared
    previous = {name: getattr(shared, name) for name in replacements}
    for name, value in replacements.items():
        setattr(shared, name, value)
    return previous


def restore_shared_implementation(previous: Mapping[str, Any]) -> None:
    for name, value in previous.items():
        setattr(frozen.shared, name, value)


@contextlib.contextmanager
def configured_shared_implementation() -> Any:
    previous = configure_shared_implementation()
    try:
        yield frozen.shared
    finally:
        restore_shared_implementation(previous)


def _delegate(name: str, *args: Any, **kwargs: Any) -> Any:
    with configured_shared_implementation():
        return getattr(frozen.shared, name)(*args, **kwargs)


@contextlib.contextmanager
def inherited_gpu_lease(
    *, physical_gpu_index: int, inherited_lock_fd: int
) -> Any:
    """Keep the K10 lock template installed for the lazy context lifetime."""

    with configured_shared_implementation():
        with frozen.shared.inherited_gpu_lease(
            physical_gpu_index=physical_gpu_index,
            inherited_lock_fd=inherited_lock_fd,
        ) as evidence:
            yield evidence


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
        return frozen.shared._chunk_indexes(
            chunk_id, shard_id, total_cases=total_cases
        )


def _parser() -> argparse.ArgumentParser:
    with configured_shared_implementation():
        return frozen.shared._parser()


def main(argv: Sequence[str] | None = None) -> int:
    with configured_shared_implementation():
        return frozen.shared.main(argv)


def __getattr__(name: str) -> Any:
    value = getattr(frozen.shared, name)
    if isinstance(value, type):
        return value
    if callable(value):
        def scoped(*args: Any, **kwargs: Any) -> Any:
            return _delegate(name, *args, **kwargs)

        return scoped
    return value


if __name__ == "__main__":
    raise SystemExit(main())
