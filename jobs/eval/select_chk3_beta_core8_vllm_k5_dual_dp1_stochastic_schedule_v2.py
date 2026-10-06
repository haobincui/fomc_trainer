"""Seal the speed-only fixed-16 policy for schedule-sensitive vLLM sampling.

The historical v3 benchmark deliberately required token-identical stochastic
outputs across ``max_num_seqs`` candidates.  That gate correctly stopped: the
input tuples and row seeds were identical, while the sampled continuations
were not.  A fresh max_num_seqs=16 replay then showed the same phenomenon
across otherwise identical engine launches.  This amendment therefore treats
token identity as a diagnostic, never as an acceptance gate.

Acceptance is deliberately narrower: exact tuple/input/seed/model/common
configuration, exact 40-row 20/20 coverage, no truncation or infrastructure
error, normal EOS completion, and no more than five percent aggregate output
token-volume drift.  Among the original 8/12/16 candidates, selection is the
strict argmax of measured aggregate output-token throughput.  The replay is
reported but is not allowed to influence that selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.eval import eval_chk3_beta_core8_merged_vllm_k5_dual_dp1 as v3_runner
from jobs.eval import (
    orchestrate_chk3_beta_core8_merged_vllm_k5_dual_dp1 as v3_orchestrator,
)
from jobs.eval import (
    remediate_chk3_beta_core8_vllm_k5_dual_dp1_stochastic_schedule_v2 as amendment,
)
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


EVALUATION_ID = (
    "chk3-beta-core8-merged-1993-2025-n2048-vllm-k5-"
    "dual-independent-dp1-stochastic-schedule-v2"
)
SCHEMA_VERSION = (
    "chk3-beta-core8-vllm-k5-dual-dp1-stochastic-schedule-speed-selection-v2"
)
SELECTION_SCOPE = "chk1_speed_only_fixed16_after_schedule_sensitivity_audit"
CANDIDATES = (8, 12, 16)
SELECTED_MAX_NUM_SEQS = 16
EXPECTED_CASES = 40
EXPECTED_SHARD_CASES = 20
MAX_TOKEN_VOLUME_RATIO = 1.05

IDENTITY_INPUT_SEED_FIELDS = (
    "model_id",
    "sample_id",
    "replicate_id",
    "replicate_seed",
    "row_seed",
    "seed",
    "absolute_case_index",
    "absolute_chunk_id",
    "shard_id",
    "prompt_sha256",
    "prompt_token_ids_sha256",
    "prompt_token_count",
    "input_token_count",
    "source_prompt_sha256",
    "source_analysis_sha256",
    "reference_minutes_sha256",
)
OUTPUT_DIAGNOSTIC_FIELDS = (
    "generated_token_ids",
    "generated_text",
    "answer",
    "finish_reason",
    "completion_sha256",
    "answer_sha256",
    "generated_token_ids_sha256",
)


class StochasticScheduleSelectionError(RuntimeError):
    """Historical evidence cannot authorize the fixed-16 v2 policy."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_value(value: Any) -> str:
    return v3_runner._sha256_text(_canonical(value))


def _binding(path: Path, *, payload_sha256: str | None = None) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise StochasticScheduleSelectionError(f"bound path is a symlink: {path}")
    path = unresolved.resolve()
    if not path.is_file():
        raise StochasticScheduleSelectionError(f"bound file is missing: {path}")
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": v3_runner.core._sha256_file(path),
    }
    if payload_sha256 is not None:
        result["payload_sha256"] = payload_sha256
    return result


def _projection(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> list[dict[str, Any]]:
    return [{field: copy.deepcopy(row.get(field)) for field in fields} for row in rows]


def _common_generation_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(contract))
    result.pop("max_num_seqs_per_worker", None)
    return result


def _normal_finish_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        row.get("status") == "ok"
        and row.get("finish_reason") == "eos"
        and row.get("vllm_raw_finish_reason") == "stop"
        and row.get("hit_eos") is True
        and row.get("cap_reached") is False
        for row in rows
    )


def _token_volume_ratio(values: Sequence[int]) -> float:
    if not values or min(values) <= 0:
        raise StochasticScheduleSelectionError("aggregate token volume is invalid")
    return max(values) / min(values)


def _speed_argmax(rates: Mapping[int, float]) -> int:
    if set(rates) != set(CANDIDATES) or any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in rates.values()
    ):
        raise StochasticScheduleSelectionError("candidate speed map is invalid")
    return max(CANDIDATES, key=lambda candidate: (float(rates[candidate]), candidate))


def _require_historical_run(
    *,
    label: str,
    max_num_seqs: int,
    manifest_path: Path,
    cohort_path: Path,
    cohort_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loaded = v3_runner.load_and_validate_run(
        manifest_path,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
        expected_model_id="chk1",
        expected_scope="infrastructure_smoke",
        max_num_seqs=max_num_seqs,
    )
    manifest = loaded.get("manifest")
    rows = loaded.get("results")
    if (
        not isinstance(manifest, Mapping)
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_CASES
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise StochasticScheduleSelectionError(f"{label} is not an exact 40-row run")
    indexes = [row.get("absolute_case_index") for row in rows]
    coverage = manifest.get("coverage")
    runtime = manifest.get("runtime")
    if (
        indexes != list(range(EXPECTED_CASES))
        or not isinstance(coverage, Mapping)
        or coverage.get("cases") != EXPECTED_CASES
        or coverage.get("shard_cases")
        != {"shard0": EXPECTED_SHARD_CASES, "shard1": EXPECTED_SHARD_CASES}
        or coverage.get("input_truncation_cases") != 0
        or any(row.get("input_truncated") is not False for row in rows)
        or not isinstance(runtime, Mapping)
        or runtime.get("resume_counts") != {"shard0": 0, "shard1": 0}
        or runtime.get("speed_measurement_valid_for_candidate_selection") is not True
        or _normal_finish_count(rows) != EXPECTED_CASES
    ):
        raise StochasticScheduleSelectionError(
            f"{label} coverage, freshness, truncation, or normal-finish gate failed"
        )
    timing_binding = runtime.get("orchestrator_timing")
    if not isinstance(timing_binding, Mapping):
        raise StochasticScheduleSelectionError(f"{label} timing binding is missing")
    timing_path = Path(str(timing_binding.get("path")))
    timing = v3_orchestrator._validate_timing(timing_path)
    shared = timing.get("shared_timing")
    topology = timing.get("parallel_topology")
    if (
        timing.get("evaluation_scope") != "infrastructure_smoke"
        or timing.get("max_num_seqs_per_engine") != max_num_seqs
        or timing.get("recovery") is not None
        or not isinstance(shared, Mapping)
        or not isinstance(topology, Mapping)
        or topology.get("independent_engine_count") != 2
        or topology.get("data_parallel_size_per_engine") != 1
        or topology.get("tensor_parallel_size_per_engine") != 1
        or topology.get("shared_go_barrier") is not True
        or shared.get("both_workers_resume_count") != [0, 0]
    ):
        raise StochasticScheduleSelectionError(f"{label} topology gate failed")
    total_tokens = sum(len(row.get("generated_token_ids", [])) for row in rows)
    wall = shared.get("parallel_generation_wall_seconds")
    rate = shared.get("aggregate_output_tokens_per_second")
    if (
        shared.get("aggregate_generated_output_tokens") != total_tokens
        or not isinstance(wall, (int, float))
        or isinstance(wall, bool)
        or float(wall) <= 0
        or not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or not math.isclose(
            float(rate), total_tokens / float(wall), rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise StochasticScheduleSelectionError(f"{label} timing accounting failed")
    contract = manifest.get("generation_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("max_num_seqs_per_worker") != max_num_seqs
        or contract.get("max_num_batched_tokens_per_worker") != 4096
        or contract.get("workers") != 2
        or contract.get("data_parallel_size_per_worker") != 1
        or contract.get("tensor_parallel_size_per_worker") != 1
        or contract.get("shard_function") != "absolute_case_index_mod_2"
    ):
        raise StochasticScheduleSelectionError(f"{label} generation contract failed")
    record = {
        "label": label,
        "max_num_seqs": max_num_seqs,
        "run_manifest": copy.deepcopy(dict(loaded["manifest_binding"])),
        "canonical_generations": copy.deepcopy(
            dict(manifest["artifacts"]["canonical_generations"])
        ),
        "orchestrator_timing": copy.deepcopy(dict(timing_binding)),
        "model": copy.deepcopy(manifest.get("model")),
        "sealed_anchor_manifest": copy.deepcopy(manifest.get("sealed_anchor_manifest")),
        "cohort": copy.deepcopy(manifest.get("cohort")),
        "generation_contract": copy.deepcopy(dict(contract)),
        "common_generation_contract": _common_generation_contract(contract),
        "input_seed_projection_sha256": _sha256_value(
            _projection(rows, IDENTITY_INPUT_SEED_FIELDS)
        ),
        "exact_output_projection_sha256": _sha256_value(
            _projection(rows, OUTPUT_DIAGNOSTIC_FIELDS)
        ),
        "normal_eos_finishes": EXPECTED_CASES,
        "input_truncation_cases": 0,
        "aggregate_generated_output_tokens": total_tokens,
        "parallel_generation_wall_seconds": float(wall),
        "aggregate_output_tokens_per_second": float(rate),
        "cases_per_second": EXPECTED_CASES / float(wall),
        "fresh_resume_counts": {"shard0": 0, "shard1": 0},
    }
    return record, [dict(row) for row in rows]


def _exact_output_diagnostic(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    per_row = [
        all(
            left_row.get(field) == right_row.get(field)
            for field in OUTPUT_DIAGNOSTIC_FIELDS
        )
        for left_row, right_row in zip(left, right, strict=True)
    ]
    by_shard = {
        f"shard{shard_id}": {
            "rows": sum(int(row["shard_id"]) == shard_id for row in left),
            "exact_output_rows": sum(
                exact and int(row["shard_id"]) == shard_id
                for exact, row in zip(per_row, left, strict=True)
            ),
        }
        for shard_id in (0, 1)
    }
    return {
        "rows": len(left),
        "exact_output_rows": sum(per_row),
        "different_output_rows": len(left) - sum(per_row),
        "exact_output_identity_required": False,
        "by_shard": by_shard,
    }


def _build(
    *,
    candidate_paths: Mapping[int, Path],
    replay_path: Path,
    cohort_path: Path,
    cohort_sha256: str,
    created_at_utc: str,
) -> dict[str, Any]:
    if set(candidate_paths) != set(CANDIDATES):
        raise StochasticScheduleSelectionError("candidate set must be exactly 8,12,16")
    if _binding(cohort_path)["sha256"] != cohort_sha256:
        raise StochasticScheduleSelectionError("cohort SHA drift")
    records: dict[int, dict[str, Any]] = {}
    rows: dict[int, list[dict[str, Any]]] = {}
    for candidate in CANDIDATES:
        records[candidate], rows[candidate] = _require_historical_run(
            label=f"max_num_seqs_{candidate}",
            max_num_seqs=candidate,
            manifest_path=candidate_paths[candidate],
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
        )
    replay_record, replay_rows = _require_historical_run(
        label="max_num_seqs_16_replay",
        max_num_seqs=SELECTED_MAX_NUM_SEQS,
        manifest_path=replay_path,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
    )
    reference_inputs = _projection(rows[8], IDENTITY_INPUT_SEED_FIELDS)
    reference_record = records[8]
    for candidate in CANDIDATES[1:]:
        if _projection(rows[candidate], IDENTITY_INPUT_SEED_FIELDS) != reference_inputs:
            raise StochasticScheduleSelectionError(
                f"max_num_seqs_{candidate} input/seed tuple drift"
            )
        for key in (
            "common_generation_contract",
            "model",
            "sealed_anchor_manifest",
            "cohort",
        ):
            if records[candidate][key] != reference_record[key]:
                raise StochasticScheduleSelectionError(
                    f"max_num_seqs_{candidate} common contract drift: {key}"
                )
    if _projection(replay_rows, IDENTITY_INPUT_SEED_FIELDS) != reference_inputs:
        raise StochasticScheduleSelectionError(
            "same-config replay input/seed tuple drift"
        )
    for key in ("generation_contract", "model", "sealed_anchor_manifest", "cohort"):
        if replay_record[key] != records[SELECTED_MAX_NUM_SEQS][key]:
            raise StochasticScheduleSelectionError(f"same-config replay drift: {key}")
    candidate_token_ratio = _token_volume_ratio(
        [records[c]["aggregate_generated_output_tokens"] for c in CANDIDATES]
    )
    replay_token_ratio = _token_volume_ratio(
        [
            records[SELECTED_MAX_NUM_SEQS]["aggregate_generated_output_tokens"],
            replay_record["aggregate_generated_output_tokens"],
        ]
    )
    if (
        candidate_token_ratio > MAX_TOKEN_VOLUME_RATIO
        or replay_token_ratio > MAX_TOKEN_VOLUME_RATIO
    ):
        raise StochasticScheduleSelectionError(
            "aggregate output token-volume gate failed"
        )
    rates = {c: records[c]["aggregate_output_tokens_per_second"] for c in CANDIDATES}
    selected = _speed_argmax(rates)
    if selected != SELECTED_MAX_NUM_SEQS:
        raise StochasticScheduleSelectionError(
            f"speed-only argmax is {selected}, expected fixed contract 16"
        )
    cross_diagnostics = {
        f"max_num_seqs_{left}_vs_{right}": _exact_output_diagnostic(
            rows[left], rows[right]
        )
        for left, right in ((8, 12), (8, 16), (12, 16))
    }
    same_diagnostic = _exact_output_diagnostic(rows[16], replay_rows)
    amendment_value = amendment.load_and_validate_receipt(amendment.DEFAULT_RECEIPT)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": created_at_utc,
        "evaluation_id": EVALUATION_ID,
        "selection_scope": SELECTION_SCOPE,
        "model_id": "chk1",
        "cohort_manifest": _binding(cohort_path),
        "amendment_receipt": copy.deepcopy(amendment_value["receipt_binding"]),
        "acceptance_contract": {
            "candidate_max_num_seqs": list(CANDIDATES),
            "selected_max_num_seqs": SELECTED_MAX_NUM_SEQS,
            "cases_per_run": EXPECTED_CASES,
            "cases_per_shard": EXPECTED_SHARD_CASES,
            "fresh_resume_count_required": 0,
            "shared_go_required": True,
            "exact_input_seed_config_topology_coverage_required": True,
            "normal_eos_finish_required": True,
            "input_truncation_or_infrastructure_error_allowed": False,
            "max_aggregate_output_token_volume_ratio": MAX_TOKEN_VOLUME_RATIO,
            "exact_generated_token_or_text_identity_required": False,
            "selection_metric": "aggregate_output_tokens_per_second",
            "selection_rule": "strict_argmax_speed_no_tie_preference",
            "replay_influences_selection": False,
        },
        "stochastic_semantics": {
            "classification": "schedule_sensitive_seeded_sampling",
            "row_seed_pairing_preserved": True,
            "token_identity_across_batch_concurrency_or_engine_launches_expected": False,
            "token_identity_differences_are_nonblocking": True,
            "scientific_unit": "each persisted stochastic generation row",
            "resume_rule": "never_resubmit_fsynced_wal_tuple_only_missing_keys",
        },
        "candidate_runs": [records[c] for c in CANDIDATES],
        "cross_candidate_diagnostics": cross_diagnostics,
        "same_config_replay": {
            "run": replay_record,
            "diagnostic": same_diagnostic,
            "aggregate_output_token_volume_ratio": replay_token_ratio,
            "within_five_percent_token_volume": True,
            "diagnostic_only_nonblocking": True,
        },
        "volume_gates": {
            "candidate_aggregate_output_token_volume_ratio": candidate_token_ratio,
            "candidate_within_five_percent": True,
            "same_config_replay_within_five_percent": True,
        },
        "selection": {
            "selected_max_num_seqs": selected,
            "selected_output_tokens_per_second": rates[selected],
            "selected_cases_per_second": records[selected]["cases_per_second"],
            "selected_run_manifest": copy.deepcopy(records[selected]["run_manifest"]),
            "candidate_rates": {str(c): rates[c] for c in CANDIDATES},
            "rationale": "max_num_seqs_16_is_strict_speed_argmax",
        },
        "generation_gates": {
            "speed_only_fixed_max_num_seqs_16": True,
            "schedule_sensitive_stochastic_token_identity_nonblocking": True,
            "exact_inputs_seeds_config_topology_coverage": True,
            "normal_finish_and_token_volume_gate": True,
            "formal_generation_unblocked": True,
        },
        "implementation_sources": {
            "selector": _binding(Path(__file__).resolve()),
            "historical_runner": _binding(Path(v3_runner.__file__).resolve()),
            "historical_orchestrator": _binding(
                Path(v3_orchestrator.__file__).resolve()
            ),
        },
    }


def _write_readonly(path: Path, value: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise StochasticScheduleSelectionError("selection artifact is create-only")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(_canonical(dict(value)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def select_and_seal(
    *,
    candidate_paths: Mapping[int, Path],
    replay_path: Path,
    cohort_path: Path,
    cohort_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    value = seal_manifest(
        _build(
            candidate_paths=candidate_paths,
            replay_path=replay_path,
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            created_at_utc=v3_runner.core._utc_now(),
        )
    )
    _write_readonly(output_path, value)
    return value


def load_and_validate_selection(
    path: Path, *, cohort_path: Path, cohort_sha256: str
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StochasticScheduleSelectionError(
            "selection artifact is missing or symlink"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StochasticScheduleSelectionError("selection artifact is not an object")
    validate_manifest_integrity(value)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise StochasticScheduleSelectionError("selection artifact is not read-only")
    runs = value.get("candidate_runs")
    replay = value.get("same_config_replay", {}).get("run")
    if not isinstance(runs, list) or len(runs) != 3 or not isinstance(replay, Mapping):
        raise StochasticScheduleSelectionError(
            "selection evidence bindings are missing"
        )
    candidate_paths = {
        int(record["max_num_seqs"]): Path(record["run_manifest"]["path"])
        for record in runs
    }
    rebuilt = seal_manifest(
        _build(
            candidate_paths=candidate_paths,
            replay_path=Path(str(replay["run_manifest"]["path"])),
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            created_at_utc=str(value.get("created_at_utc")),
        )
    )
    if value != rebuilt:
        raise StochasticScheduleSelectionError("selection artifact drift")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--cohort", type=Path, required=True)
    select.add_argument("--cohort-sha256", required=True)
    for candidate in CANDIDATES:
        select.add_argument(f"--manifest-{candidate}", type=Path, required=True)
    select.add_argument("--replay-manifest-16", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--cohort", type=Path, required=True)
    validate.add_argument("--cohort-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "select":
            value = select_and_seal(
                candidate_paths={
                    candidate: getattr(args, f"manifest_{candidate}")
                    for candidate in CANDIDATES
                },
                replay_path=args.replay_manifest_16,
                cohort_path=args.cohort,
                cohort_sha256=args.cohort_sha256,
                output_path=args.output,
            )
        else:
            value = load_and_validate_selection(
                args.manifest,
                cohort_path=args.cohort,
                cohort_sha256=args.cohort_sha256,
            )
        print(
            _canonical(
                {
                    "status": "complete" if args.command == "select" else "valid",
                    "selected_max_num_seqs": value["selection"][
                        "selected_max_num_seqs"
                    ],
                    "same_config_exact_rows": value["same_config_replay"]["diagnostic"][
                        "exact_output_rows"
                    ],
                    "payload_sha256": value["integrity"]["payload_sha256"],
                }
            )
        )
        return 0
    except (
        StochasticScheduleSelectionError,
        amendment.StochasticScheduleAmendmentError,
        v3_runner.DualDp1GenerationError,
        v3_orchestrator.DualDp1OrchestrationError,
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


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVALUATION_ID",
    "MAX_TOKEN_VOLUME_RATIO",
    "SELECTED_MAX_NUM_SEQS",
    "StochasticScheduleSelectionError",
    "load_and_validate_selection",
    "main",
    "select_and_seal",
]
