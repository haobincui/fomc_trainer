"""Select and validate the frozen DP=2 vLLM K=5 concurrency setting.

The selector is deliberately CPU-only.  It deep-validates three independently
sealed CHK1 infrastructure-smoke runs (``max_num_seqs`` 8, 12, and 16) with
the generation runner, proves that all forty seeded requests produced exactly
the same token IDs and decoded outputs, and then selects a concurrency value
using only the sealed generation throughput.  If 12 is within five percent of
the fastest stable candidate, 12 is preferred; otherwise the exact fastest
candidate is selected.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.eval import eval_chk3_beta_core8_merged_vllm_k5 as runner
from jobs.eval import eval_chk3_stochastic_bootstrap_generation as core
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


SCHEMA_VERSION = "chk3-beta-core8-merged-vllm-k5-benchmark-selection-v1"
SELECTION_SCOPE = "chk1_dp2_infrastructure_smoke_max_num_seqs_benchmark"
CANDIDATES = runner.ALLOWED_MAX_NUM_SEQS
PREFERRED_CANDIDATE = 12
TIE_RELATIVE_TOLERANCE = 0.05
EXPECTED_MODEL_ID = "chk1"
EXPECTED_SCOPE = "infrastructure_smoke"
EXPECTED_CASES = runner.SMOKE_CASES_PER_MODEL
IDENTITY_FIELDS = (
    "model_id",
    "sample_id",
    "replicate_id",
    "row_seed",
    "absolute_case_index",
)
OUTPUT_EQUIVALENCE_FIELDS = (
    "generated_token_ids",
    "generated_text",
    "answer",
    "finish_reason",
    "completion_sha256",
    "answer_sha256",
    "generated_token_ids_sha256",
)
TIMING_FIELDS = (
    "model_load_wall_seconds",
    "generation_wall_seconds",
    "generated_output_tokens",
    "timed_session_generated_output_tokens",
    "completed_requests_in_timed_session",
    "output_tokens_per_second",
    "timing_scope",
    "preemption_count",
    "preemption_count_status",
    "speed_measurement_valid_for_candidate_selection",
)


class BenchmarkSelectionError(RuntimeError):
    """A benchmark candidate or sealed selection violated its contract."""


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise BenchmarkSelectionError(
            f"value is not finite canonical JSON: {exc}"
        ) from exc


def _sha256_value(value: Any) -> str:
    return core._sha256_text(_canonical(value))


def _file_binding(path: Path, *, payload_sha256: str | None = None) -> dict[str, Any]:
    binding = core._file_binding(path, payload_sha256=payload_sha256)
    binding["path"] = str(path.resolve())
    return binding


def _require_bound_file(binding: Any, *, expected_path: Path) -> None:
    if not isinstance(binding, Mapping):
        raise BenchmarkSelectionError("run manifest binding is missing")
    expected_path = expected_path.resolve()
    if (
        binding.get("path") != str(expected_path)
        or binding.get("sha256") != core._sha256_file(expected_path)
        or binding.get("bytes") != expected_path.stat().st_size
    ):
        raise BenchmarkSelectionError("run manifest binding drift")
    payload_sha = binding.get("payload_sha256")
    if not isinstance(payload_sha, str) or len(payload_sha) != 64:
        raise BenchmarkSelectionError("run manifest payload binding is missing")


def _common_source_hashes(source_hashes: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the one candidate-specific sampling-contract digest."""

    normalized = copy.deepcopy(dict(source_hashes))
    if "vllm_k5_sampling_contract_sha256" not in normalized:
        raise BenchmarkSelectionError(
            "candidate source hashes omit the vLLM sampling contract"
        )
    del normalized["vllm_k5_sampling_contract_sha256"]
    return normalized


def _identity_projection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{field: row.get(field) for field in IDENTITY_FIELDS} for row in rows]


def _output_projection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "identity": {field: row.get(field) for field in IDENTITY_FIELDS},
            "output": {field: row.get(field) for field in OUTPUT_EQUIVALENCE_FIELDS},
        }
        for row in rows
    ]


def _require_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != EXPECTED_CASES:
        raise BenchmarkSelectionError(
            f"each benchmark candidate must contain exactly {EXPECTED_CASES} rows"
        )
    copied = [dict(row) if isinstance(row, Mapping) else None for row in rows]
    if any(row is None for row in copied):
        raise BenchmarkSelectionError("benchmark results contain a non-object row")
    result = [row for row in copied if row is not None]
    identities = [tuple(row.get(field) for field in IDENTITY_FIELDS) for row in result]
    if len(set(identities)) != EXPECTED_CASES:
        raise BenchmarkSelectionError("benchmark tuple identities are not unique")
    if any(
        row.get("model_id") != EXPECTED_MODEL_ID
        or row.get("input_truncated") is not False
        for row in result
    ):
        raise BenchmarkSelectionError(
            "benchmark has a model mismatch or input truncation"
        )
    for row in result:
        if not isinstance(row.get("generated_token_ids"), list):
            raise BenchmarkSelectionError("generated token IDs are missing")
        if not isinstance(row.get("generated_text"), str):
            raise BenchmarkSelectionError("generated text is missing")
        for field in (
            "completion_sha256",
            "answer_sha256",
            "generated_token_ids_sha256",
        ):
            value = row.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise BenchmarkSelectionError(f"invalid output hash: {field}")
    return result


def _require_summary(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    summary = manifest.get("summary")
    if not isinstance(summary, Mapping):
        raise BenchmarkSelectionError("sealed run summary is missing")
    if (
        manifest.get("status") != "complete"
        or manifest.get("model_id") != EXPECTED_MODEL_ID
        or manifest.get("evaluation_scope") != EXPECTED_SCOPE
        or summary.get("status") != "complete"
        or summary.get("evaluation_scope") != EXPECTED_SCOPE
        or summary.get("cases") != EXPECTED_CASES
        or summary.get("input_truncation_cases") != 0
        or sum(bool(row.get("input_truncated")) for row in rows) != 0
    ):
        raise BenchmarkSelectionError(
            "candidate is incomplete, failed, or contains input truncation"
        )


def _require_runtime(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("resume_count") != 0:
        raise BenchmarkSelectionError(
            "benchmark speed runs must be fresh, non-resumed engine sessions"
        )
    engine = runtime.get("engine_runtime")
    if not isinstance(engine, Mapping):
        raise BenchmarkSelectionError("sealed engine runtime is missing")
    seconds = engine.get("generation_wall_seconds")
    output_tokens = engine.get("generated_output_tokens")
    timed_session_tokens = engine.get("timed_session_generated_output_tokens")
    requests = engine.get("completed_requests_in_timed_session")
    rate = engine.get("output_tokens_per_second")
    model_load_seconds = engine.get("model_load_wall_seconds")
    if (
        not isinstance(model_load_seconds, (int, float))
        or isinstance(model_load_seconds, bool)
        or not math.isfinite(float(model_load_seconds))
        or float(model_load_seconds) <= 0
        or not isinstance(seconds, (int, float))
        or isinstance(seconds, bool)
        or not math.isfinite(float(seconds))
        or float(seconds) <= 0
        or not isinstance(output_tokens, int)
        or isinstance(output_tokens, bool)
        or output_tokens <= 0
        or timed_session_tokens != output_tokens
        or requests != EXPECTED_CASES
        or not isinstance(rate, (int, float))
        or isinstance(rate, bool)
        or not math.isfinite(float(rate))
        or float(rate) <= 0
        or engine.get("speed_measurement_valid_for_candidate_selection") is not True
    ):
        raise BenchmarkSelectionError("sealed benchmark timing is invalid")
    observed_tokens = sum(len(row["generated_token_ids"]) for row in rows)
    if output_tokens != observed_tokens:
        raise BenchmarkSelectionError(
            "sealed output-token timing count differs from canonical results"
        )
    expected_rate = output_tokens / float(seconds)
    if not math.isclose(float(rate), expected_rate, rel_tol=1e-9, abs_tol=1e-12):
        raise BenchmarkSelectionError(
            "sealed output throughput differs from tokens divided by seconds"
        )

    identities = engine.get("visible_cuda_devices")
    bindings = engine.get("engine_core_gpu_bindings")
    engine_contract = engine.get("engine")
    lease = engine.get("gpu_lease")
    if (
        not isinstance(identities, list)
        or len(identities) != 2
        or not all(isinstance(entry, Mapping) for entry in identities)
        or [entry.get("logical_cuda_index") for entry in identities] != [0, 1]
        or not isinstance(bindings, list)
        or len(bindings) != 2
        or not isinstance(lease, Mapping)
        or lease.get("physical_gpu_indexes") != [0, 1]
        or not isinstance(engine_contract, Mapping)
        or engine_contract.get("data_parallel_size") != 2
        or engine_contract.get("tensor_parallel_size") != 1
        or engine_contract.get("parallel_topology")
        != "two_full_bf16_model_replicas_one_per_physical_gpu"
    ):
        raise BenchmarkSelectionError("dual-GPU runtime evidence is missing")
    pids: list[int] = []
    for index, binding in enumerate(bindings):
        if not isinstance(binding, Mapping):
            raise BenchmarkSelectionError("dual-GPU runtime evidence is invalid")
        pid = binding.get("engine_core_pid")
        if (
            binding.get("physical_gpu_index") != index
            or binding.get("data_parallel_rank") != index
            or not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 0
        ):
            raise BenchmarkSelectionError("dual-GPU runtime evidence is invalid")
        pids.append(pid)
    if len(set(pids)) != 2:
        raise BenchmarkSelectionError("dual-GPU EngineCore PIDs are not distinct")

    timing = {field: copy.deepcopy(engine.get(field)) for field in TIMING_FIELDS}
    dual_gpu = {
        "visible_cuda_devices": copy.deepcopy(identities),
        "engine_core_gpu_bindings": copy.deepcopy(bindings),
        "gpu_lease": copy.deepcopy(lease),
        "parallel_topology": engine_contract["parallel_topology"],
        "data_parallel_size": engine_contract["data_parallel_size"],
        "tensor_parallel_size": engine_contract["tensor_parallel_size"],
    }
    return timing, dual_gpu


def _candidate_record(
    *,
    max_num_seqs: int,
    manifest_path: Path,
    loaded: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = loaded.get("manifest")
    if not isinstance(manifest, Mapping):
        raise BenchmarkSelectionError("deep validator returned no run manifest")
    binding = loaded.get("manifest_binding")
    _require_bound_file(binding, expected_path=manifest_path)
    rows = _require_rows(loaded.get("results"))
    _require_summary(manifest, rows)
    timing, dual_gpu = _require_runtime(manifest, rows)
    source_hashes = manifest.get("source_artifact_sha256s")
    artifacts = manifest.get("artifacts")
    if not isinstance(source_hashes, Mapping) or not isinstance(artifacts, Mapping):
        raise BenchmarkSelectionError("candidate source/artifact bindings are missing")
    canonical_binding = artifacts.get("canonical_generations")
    if not isinstance(canonical_binding, Mapping):
        raise BenchmarkSelectionError("canonical generation binding is missing")
    sampling_sha = source_hashes.get("vllm_k5_sampling_contract_sha256")
    if not isinstance(sampling_sha, str) or len(sampling_sha) != 64:
        raise BenchmarkSelectionError("candidate sampling-contract hash is invalid")
    record = {
        "max_num_seqs": max_num_seqs,
        "stable": True,
        "infrastructure_failure_cases": 0,
        "input_truncation_cases": 0,
        "run_manifest": copy.deepcopy(dict(binding)),
        "canonical_generations": copy.deepcopy(dict(canonical_binding)),
        "source_artifact_sha256s": copy.deepcopy(dict(source_hashes)),
        "sampling_contract_sha256": sampling_sha,
        "timing": timing,
        "dual_gpu_runtime_evidence": dual_gpu,
        "tuple_identity_sha256": _sha256_value(_identity_projection(rows)),
        "exact_output_equivalence_sha256": _sha256_value(_output_projection(rows)),
    }
    return record, rows


def _select_candidate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rates = {
        int(record["max_num_seqs"]): float(record["timing"]["output_tokens_per_second"])
        for record in records
    }
    fastest_rate = max(rates.values())
    fastest = min(
        candidate for candidate, rate in rates.items() if rate == fastest_rate
    )
    threshold_rate = fastest_rate * (1.0 - TIE_RELATIVE_TOLERANCE)
    within_band = sorted(
        candidate for candidate, rate in rates.items() if rate >= threshold_rate
    )
    if PREFERRED_CANDIDATE in within_band:
        selected = PREFERRED_CANDIDATE
        rationale = "max_num_seqs_12_is_within_5_percent_of_fastest"
    else:
        selected = fastest
        rationale = "fastest_stable_candidate_outside_12_preference_band"
    return {
        "selected_max_num_seqs": selected,
        "selected_output_tokens_per_second": rates[selected],
        "fastest_max_num_seqs": fastest,
        "fastest_output_tokens_per_second": fastest_rate,
        "five_percent_floor_output_tokens_per_second": threshold_rate,
        "candidates_within_five_percent_of_fastest": within_band,
        "selected_relative_slowdown_from_fastest": (fastest_rate - rates[selected])
        / fastest_rate,
        "rationale": rationale,
    }


def _build_payload(
    *,
    candidate_manifest_paths: Mapping[int, Path],
    cohort_path: Path,
    cohort_sha256: str,
    created_at_utc: str,
) -> dict[str, Any]:
    if set(candidate_manifest_paths) != set(CANDIDATES):
        raise BenchmarkSelectionError(
            f"candidate set must be exactly {list(CANDIDATES)}"
        )
    unresolved_cohort = cohort_path.expanduser()
    if unresolved_cohort.is_symlink():
        raise BenchmarkSelectionError("cohort manifest is a symlink")
    cohort_path = unresolved_cohort.resolve()
    if not cohort_path.is_file():
        raise BenchmarkSelectionError("cohort manifest is missing or a symlink")
    cohort_binding = _file_binding(cohort_path)
    if cohort_binding["sha256"] != cohort_sha256:
        raise BenchmarkSelectionError("cohort manifest SHA256 mismatch")

    records: list[dict[str, Any]] = []
    candidate_rows: dict[int, list[dict[str, Any]]] = {}
    for candidate in CANDIDATES:
        unresolved_path = candidate_manifest_paths[candidate].expanduser()
        if unresolved_path.is_symlink():
            raise BenchmarkSelectionError("run manifest is a symlink")
        path = unresolved_path.resolve()
        loaded = runner.load_and_validate_run(
            path,
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            expected_model_id=EXPECTED_MODEL_ID,
            expected_scope=EXPECTED_SCOPE,
            max_num_seqs=candidate,
        )
        record, rows = _candidate_record(
            max_num_seqs=candidate,
            manifest_path=path,
            loaded=loaded,
        )
        records.append(record)
        candidate_rows[candidate] = rows

    reference_identities = _identity_projection(candidate_rows[CANDIDATES[0]])
    reference_outputs = _output_projection(candidate_rows[CANDIDATES[0]])
    reference_common_sources = _common_source_hashes(
        records[0]["source_artifact_sha256s"]
    )
    for candidate, record in zip(CANDIDATES[1:], records[1:], strict=True):
        if _identity_projection(candidate_rows[candidate]) != reference_identities:
            raise BenchmarkSelectionError(
                "the three candidates do not have the same forty tuple identities"
            )
        if _output_projection(candidate_rows[candidate]) != reference_outputs:
            raise BenchmarkSelectionError(
                "max_num_seqs changed generated token IDs, text, or output hashes"
            )
        if (
            _common_source_hashes(record["source_artifact_sha256s"])
            != reference_common_sources
        ):
            raise BenchmarkSelectionError(
                "candidate source artifacts differ outside the sampling contract"
            )

    selection = _select_candidate(records)
    fastest_rate = float(selection["fastest_output_tokens_per_second"])
    for rank, record in enumerate(
        sorted(
            records,
            key=lambda item: (
                -float(item["timing"]["output_tokens_per_second"]),
                int(item["max_num_seqs"]),
            ),
        ),
        start=1,
    ):
        record["throughput_rank"] = rank
        record["within_five_percent_of_fastest"] = float(
            record["timing"]["output_tokens_per_second"]
        ) >= fastest_rate * (1.0 - TIE_RELATIVE_TOLERANCE)
    records.sort(key=lambda item: int(item["max_num_seqs"]))

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": created_at_utc,
        "evaluation_id": runner.EVALUATION_ID,
        "task_contract_id": core.TASK_CONTRACT_ID,
        "selection_scope": SELECTION_SCOPE,
        "model_id": EXPECTED_MODEL_ID,
        "evaluation_scope": EXPECTED_SCOPE,
        "cohort_manifest": cohort_binding,
        "benchmark_contract": {
            "candidate_max_num_seqs": list(CANDIDATES),
            "required_cases_per_candidate": EXPECTED_CASES,
            "required_tuple_identity_fields": list(IDENTITY_FIELDS),
            "required_exact_output_fields": list(OUTPUT_EQUIVALENCE_FIELDS),
            "timing_fields": list(TIMING_FIELDS),
            "fresh_engine_session_required": True,
            "resume_count_required": 0,
            "dual_gpu_runtime_evidence_required": True,
            "all_candidates_must_be_stable": True,
            "input_truncation_cases_required": 0,
            "preference_candidate": PREFERRED_CANDIDATE,
            "tie_relative_tolerance": TIE_RELATIVE_TOLERANCE,
            "selection_rule": (
                "choose_12_if_rate_gte_95_percent_of_fastest_else_choose_fastest"
            ),
        },
        "implementation_sources": {
            "benchmark_selector": _file_binding(Path(__file__).resolve()),
            "generation_runner": _file_binding(Path(runner.__file__).resolve()),
        },
        "common_source_artifact_sha256s": reference_common_sources,
        "cross_candidate_equivalence": {
            "tuple_identities": EXPECTED_CASES,
            "tuple_identity_sha256": _sha256_value(reference_identities),
            "generated_token_ids_exact": True,
            "generated_text_exact": True,
            "answer_exact": True,
            "finish_reason_exact": True,
            "output_hashes_exact": True,
            "exact_output_equivalence_sha256": _sha256_value(reference_outputs),
        },
        "candidate_runs": records,
        "selection": selection,
    }


def select_and_seal(
    *,
    candidate_manifest_paths: Mapping[int, Path],
    cohort_path: Path,
    cohort_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    unresolved_output = output_path.expanduser()
    if unresolved_output.is_symlink():
        raise BenchmarkSelectionError("selection output is a symlink")
    output_path = unresolved_output.resolve()
    if output_path.exists():
        raise BenchmarkSelectionError(
            f"refusing to overwrite sealed selection: {output_path}"
        )
    payload = _build_payload(
        candidate_manifest_paths=candidate_manifest_paths,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
        created_at_utc=core._utc_now(),
    )
    sealed = seal_manifest(payload)
    core._write_new_json(output_path, sealed)
    os.chmod(output_path, 0o444)
    core._fsync_directory(output_path.parent)
    return sealed


def load_and_validate_selection(
    manifest_path: Path, *, cohort_path: Path, cohort_sha256: str
) -> dict[str, Any]:
    unresolved = manifest_path.expanduser()
    if unresolved.is_symlink():
        raise BenchmarkSelectionError("selection manifest is a symlink")
    manifest_path = unresolved.resolve()
    if not manifest_path.is_file():
        raise BenchmarkSelectionError("selection manifest is missing")
    if stat.S_IMODE(manifest_path.stat().st_mode) & 0o222:
        raise BenchmarkSelectionError("selection manifest is not sealed read-only")
    manifest = core._read_json(manifest_path)
    validate_manifest_integrity(manifest)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("evaluation_id") != runner.EVALUATION_ID
        or manifest.get("selection_scope") != SELECTION_SCOPE
        or not isinstance(manifest.get("created_at_utc"), str)
    ):
        raise BenchmarkSelectionError("sealed selection identity drift")
    records = manifest.get("candidate_runs")
    if not isinstance(records, list) or len(records) != len(CANDIDATES):
        raise BenchmarkSelectionError("sealed candidate bindings are missing")
    paths: dict[int, Path] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise BenchmarkSelectionError("sealed candidate binding is invalid")
        candidate = record.get("max_num_seqs")
        binding = record.get("run_manifest")
        if candidate not in CANDIDATES or not isinstance(binding, Mapping):
            raise BenchmarkSelectionError("sealed candidate binding is invalid")
        paths[int(candidate)] = Path(str(binding.get("path")))
    rebuilt = _build_payload(
        candidate_manifest_paths=paths,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
        created_at_utc=str(manifest["created_at_utc"]),
    )
    expected = seal_manifest(rebuilt)
    if manifest != expected:
        raise BenchmarkSelectionError("sealed benchmark selection drift")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--cohort", required=True, type=Path)
    select.add_argument("--cohort-sha256", required=True)
    for candidate in CANDIDATES:
        select.add_argument(f"--manifest-{candidate}", required=True, type=Path)
    select.add_argument("--output", required=True, type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--cohort", required=True, type=Path)
    validate.add_argument("--cohort-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "select":
            paths = {
                candidate: getattr(args, f"manifest_{candidate}")
                for candidate in CANDIDATES
            }
            result = select_and_seal(
                candidate_manifest_paths=paths,
                cohort_path=args.cohort,
                cohort_sha256=args.cohort_sha256,
                output_path=args.output,
            )
            status = result["status"]
        else:
            result = load_and_validate_selection(
                args.manifest,
                cohort_path=args.cohort,
                cohort_sha256=args.cohort_sha256,
            )
            status = "valid"
        print(
            _canonical(
                {
                    "status": status,
                    "schema_version": result["schema_version"],
                    "selected_max_num_seqs": result["selection"][
                        "selected_max_num_seqs"
                    ],
                    "payload_sha256": result["integrity"]["payload_sha256"],
                }
            )
        )
        return 0
    except (
        BenchmarkSelectionError,
        runner.VllmK5GenerationError,
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
            file=os.sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BenchmarkSelectionError",
    "CANDIDATES",
    "SCHEMA_VERSION",
    "load_and_validate_selection",
    "main",
    "select_and_seal",
]
