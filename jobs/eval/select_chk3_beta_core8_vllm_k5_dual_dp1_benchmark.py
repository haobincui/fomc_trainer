"""Select max_num_seqs for the dual-independent-DP1 vLLM K5 runner.

All three CHK1 candidates must be fresh 40-row paired runs with a shared GO,
20 rows per parity shard, resume_count=0, and exact token/text/output equality.
Selection uses only aggregate dual-engine tokens divided by the supervisor's
monotonic GO-to-both-DONE wall clock.  A separate helper proves that the
selected candidate exactly replays in the later fresh CHK1 official smoke.
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

from jobs.eval import eval_chk3_beta_core8_merged_vllm_k5_dual_dp1 as runner
from jobs.eval import (
    orchestrate_chk3_beta_core8_merged_vllm_k5_dual_dp1 as orchestrator,
)
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


SCHEMA_VERSION = "chk3-beta-core8-vllm-k5-dual-dp1-benchmark-selection-v1"
SELECTION_SCOPE = "chk1_dual_independent_dp1_max_num_seqs_benchmark"
CANDIDATES = (8, 12, 16)
PREFERRED_CANDIDATE = 12
TIE_RELATIVE_TOLERANCE = 0.05
EXPECTED_CASES = 40
EXPECTED_SHARD_CASES = 20
IDENTITY_FIELDS = (
    "model_id",
    "sample_id",
    "replicate_id",
    "row_seed",
    "absolute_case_index",
)
OUTPUT_FIELDS = (
    "generated_token_ids",
    "generated_text",
    "answer",
    "finish_reason",
    "completion_sha256",
    "answer_sha256",
    "generated_token_ids_sha256",
)


class DualDp1BenchmarkError(RuntimeError):
    """A benchmark candidate, selection, or smoke replay failed."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_value(value: Any) -> str:
    return runner._sha256_text(_canonical(value))


def _binding(path: Path, *, payload_sha256: str | None = None) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise DualDp1BenchmarkError(f"bound path is a symlink: {unresolved}")
    path = unresolved.resolve()
    if not path.is_file():
        raise DualDp1BenchmarkError(f"bound file is missing: {path}")
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": runner.core._sha256_file(path),
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
    result.pop("expected_cases", None)
    return result


def _require_candidate(
    *,
    candidate: int,
    manifest_path: Path,
    cohort_path: Path,
    cohort_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loaded = runner.load_and_validate_run(
        manifest_path,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
        expected_model_id="chk1",
        expected_scope="infrastructure_smoke",
        max_num_seqs=candidate,
    )
    manifest = loaded.get("manifest")
    rows = loaded.get("results")
    if (
        not isinstance(manifest, Mapping)
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_CASES
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise DualDp1BenchmarkError("candidate is not an exact 40-row run")
    indexes = [row.get("absolute_case_index") for row in rows]
    if (
        indexes != list(range(EXPECTED_CASES))
        or any(row.get("input_truncated") is not False for row in rows)
        or manifest.get("coverage", {}).get("shard_cases")
        != {"shard0": EXPECTED_SHARD_CASES, "shard1": EXPECTED_SHARD_CASES}
    ):
        raise DualDp1BenchmarkError("candidate parity coverage/truncation drift")
    runtime = manifest.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("resume_counts") != {"shard0": 0, "shard1": 0}
        or runtime.get("speed_measurement_valid_for_candidate_selection") is not True
        or not isinstance(runtime.get("orchestrator_timing"), Mapping)
    ):
        raise DualDp1BenchmarkError("candidate is not a fresh shared-GO run")
    timing_path = Path(str(runtime["orchestrator_timing"].get("path")))
    timing = orchestrator._validate_timing(timing_path)
    shared = timing.get("shared_timing")
    topology = timing.get("parallel_topology")
    if (
        timing.get("evaluation_scope") != "infrastructure_smoke"
        or timing.get("max_num_seqs_per_engine") != candidate
        or timing.get("recovery") is not None
        or not isinstance(shared, Mapping)
        or not isinstance(topology, Mapping)
        or topology.get("independent_engine_count") != 2
        or topology.get("data_parallel_size_per_engine") != 1
        or topology.get("shared_go_barrier") is not True
        or shared.get("both_workers_resume_count") != [0, 0]
    ):
        raise DualDp1BenchmarkError("candidate orchestrator topology drift")
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
        raise DualDp1BenchmarkError("candidate aggregate timing accounting drift")
    contract = manifest.get("generation_contract")
    if not isinstance(contract, Mapping):
        raise DualDp1BenchmarkError("candidate generation contract is missing")
    record = {
        "max_num_seqs": candidate,
        "stable": True,
        "run_manifest": copy.deepcopy(dict(loaded["manifest_binding"])),
        "canonical_generations": copy.deepcopy(
            dict(manifest["artifacts"]["canonical_generations"])
        ),
        "orchestrator_timing": copy.deepcopy(dict(runtime["orchestrator_timing"])),
        "generation_contract": copy.deepcopy(dict(contract)),
        "common_generation_contract": _common_generation_contract(contract),
        "model": copy.deepcopy(manifest.get("model")),
        "sealed_anchor_manifest": copy.deepcopy(manifest.get("sealed_anchor_manifest")),
        "cohort": copy.deepcopy(manifest.get("cohort")),
        "timing": {
            "parallel_generation_wall_seconds": float(wall),
            "aggregate_generated_output_tokens": total_tokens,
            "aggregate_output_tokens_per_second": float(rate),
            "timing_clock": "time.monotonic_ns_shared_host_clock",
            "start": "parent_shared_go_after_both_engines_ready",
            "finish": "maximum_two_worker_done_monotonic_time",
        },
        "identity_sha256": _sha256_value(_projection(rows, IDENTITY_FIELDS)),
        "exact_output_sha256": _sha256_value(_projection(rows, OUTPUT_FIELDS)),
    }
    return record, [dict(row) for row in rows]


def _select(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rates = {
        int(record["max_num_seqs"]): float(
            record["timing"]["aggregate_output_tokens_per_second"]
        )
        for record in records
    }
    fastest_rate = max(rates.values())
    fastest = min(
        candidate for candidate, rate in rates.items() if rate == fastest_rate
    )
    floor = fastest_rate * (1.0 - TIE_RELATIVE_TOLERANCE)
    within = sorted(candidate for candidate, rate in rates.items() if rate >= floor)
    if PREFERRED_CANDIDATE in within:
        selected = PREFERRED_CANDIDATE
        rationale = "max_num_seqs_12_is_within_5_percent_of_fastest"
    else:
        selected = fastest
        rationale = "fastest_stable_candidate_outside_12_preference_band"
    selected_record = next(
        record for record in records if record["max_num_seqs"] == selected
    )
    return {
        "selected_max_num_seqs": selected,
        "selected_output_tokens_per_second": rates[selected],
        "selected_run_manifest": copy.deepcopy(selected_record["run_manifest"]),
        "fastest_max_num_seqs": fastest,
        "fastest_output_tokens_per_second": fastest_rate,
        "five_percent_floor_output_tokens_per_second": floor,
        "candidates_within_five_percent_of_fastest": within,
        "selected_relative_slowdown_from_fastest": (fastest_rate - rates[selected])
        / fastest_rate,
        "rationale": rationale,
    }


def _build(
    *,
    paths: Mapping[int, Path],
    cohort_path: Path,
    cohort_sha256: str,
    created_at_utc: str,
) -> dict[str, Any]:
    if set(paths) != set(CANDIDATES):
        raise DualDp1BenchmarkError("candidate set must be exactly 8, 12, 16")
    if _binding(cohort_path)["sha256"] != cohort_sha256:
        raise DualDp1BenchmarkError("cohort SHA drift")
    records: list[dict[str, Any]] = []
    rows: dict[int, list[dict[str, Any]]] = {}
    for candidate in CANDIDATES:
        record, candidate_rows = _require_candidate(
            candidate=candidate,
            manifest_path=paths[candidate],
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
        )
        records.append(record)
        rows[candidate] = candidate_rows
    reference_identity = _projection(rows[8], IDENTITY_FIELDS)
    reference_output = _projection(rows[8], OUTPUT_FIELDS)
    reference_record = records[0]
    for candidate, record in zip(CANDIDATES[1:], records[1:], strict=True):
        if _projection(rows[candidate], IDENTITY_FIELDS) != reference_identity:
            raise DualDp1BenchmarkError("candidate tuple identity drift")
        if _projection(rows[candidate], OUTPUT_FIELDS) != reference_output:
            raise DualDp1BenchmarkError(
                "max_num_seqs candidates changed exact generated token/output rows"
            )
        for key in (
            "common_generation_contract",
            "model",
            "sealed_anchor_manifest",
            "cohort",
        ):
            if record[key] != reference_record[key]:
                raise DualDp1BenchmarkError(f"candidate common contract drift: {key}")
    fastest_rate = max(
        float(record["timing"]["aggregate_output_tokens_per_second"])
        for record in records
    )
    ranked = sorted(
        records,
        key=lambda row: (
            -float(row["timing"]["aggregate_output_tokens_per_second"]),
            int(row["max_num_seqs"]),
        ),
    )
    ranks = {int(record["max_num_seqs"]): rank for rank, record in enumerate(ranked, 1)}
    for record in records:
        record["throughput_rank"] = ranks[int(record["max_num_seqs"])]
        record["within_five_percent_of_fastest"] = (
            float(record["timing"]["aggregate_output_tokens_per_second"])
            >= fastest_rate * 0.95
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": created_at_utc,
        "evaluation_id": runner.EVALUATION_ID,
        "selection_scope": SELECTION_SCOPE,
        "model_id": "chk1",
        "cohort_manifest": _binding(cohort_path),
        "benchmark_contract": {
            "candidate_max_num_seqs": list(CANDIDATES),
            "cases_per_candidate": EXPECTED_CASES,
            "cases_per_shard": EXPECTED_SHARD_CASES,
            "fresh_resume_count_required": 0,
            "shared_go_required": True,
            "exact_output_fields": list(OUTPUT_FIELDS),
            "candidate_exact_output_parity_required": True,
            "timing_clock": "time.monotonic_ns_shared_host_clock",
            "selection_rule": (
                "choose_12_if_rate_gte_95_percent_of_fastest_else_choose_fastest"
            ),
            "selected_chk1_smoke_exact_replay_required_before_formal": True,
        },
        "implementation_sources": {
            "selector": _binding(Path(__file__).resolve()),
            "runner": _binding(Path(runner.__file__).resolve()),
            "orchestrator": _binding(Path(orchestrator.__file__).resolve()),
        },
        "cross_candidate_equivalence": {
            "tuple_identities": EXPECTED_CASES,
            "tuple_identity_sha256": _sha256_value(reference_identity),
            "exact_outputs": True,
            "exact_output_sha256": _sha256_value(reference_output),
        },
        "candidate_runs": records,
        "selection": _select(records),
    }


def _write_readonly(path: Path, value: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise DualDp1BenchmarkError("benchmark selection is create-only")
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
    paths: Mapping[int, Path],
    cohort_path: Path,
    cohort_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    value = seal_manifest(
        _build(
            paths=paths,
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            created_at_utc=runner.core._utc_now(),
        )
    )
    _write_readonly(output_path, value)
    return value


def load_and_validate_selection(
    path: Path, *, cohort_path: Path, cohort_sha256: str
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DualDp1BenchmarkError("selection manifest is missing or a symlink")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DualDp1BenchmarkError("selection manifest is not an object")
    validate_manifest_integrity(value)
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise DualDp1BenchmarkError("selection manifest is not read-only")
    records = value.get("candidate_runs")
    if not isinstance(records, list) or len(records) != 3:
        raise DualDp1BenchmarkError("selection candidate bindings are missing")
    paths = {
        int(record["max_num_seqs"]): Path(record["run_manifest"]["path"])
        for record in records
    }
    rebuilt = seal_manifest(
        _build(
            paths=paths,
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            created_at_utc=str(value.get("created_at_utc")),
        )
    )
    if value != rebuilt:
        raise DualDp1BenchmarkError("selection manifest drift")
    return value


def validate_selected_smoke_replay(
    *,
    selection: Mapping[str, Any],
    smoke_loaded: Mapping[str, Any],
) -> dict[str, Any]:
    selected = int(selection.get("selection", {}).get("selected_max_num_seqs"))
    record = next(
        (
            row
            for row in selection.get("candidate_runs", [])
            if row.get("max_num_seqs") == selected
        ),
        None,
    )
    smoke_rows = smoke_loaded.get("results")
    smoke_manifest = smoke_loaded.get("manifest")
    if (
        not isinstance(record, Mapping)
        or not isinstance(smoke_rows, list)
        or len(smoke_rows) != EXPECTED_CASES
        or not isinstance(smoke_manifest, Mapping)
        or smoke_manifest.get("model_id") != "chk1"
        or smoke_manifest.get("evaluation_scope") != "infrastructure_smoke"
        or smoke_manifest.get("generation_contract", {}).get("max_num_seqs_per_worker")
        != selected
        or smoke_manifest.get("cohort")
        != {
            "path": str(selection["cohort_manifest"]["path"]),
            "sha256": str(selection["cohort_manifest"]["sha256"]),
        }
        or smoke_manifest.get("runtime", {}).get("resume_counts")
        != {"shard0": 0, "shard1": 0}
        or smoke_manifest.get("runtime", {}).get(
            "speed_measurement_valid_for_candidate_selection"
        )
        is not True
    ):
        raise DualDp1BenchmarkError("CHK1 smoke is not a fresh 40-row replay")
    selected_manifest = Path(str(record["run_manifest"]["path"]))
    selected_rows = runner.load_and_validate_run(
        selected_manifest,
        cohort_path=Path(str(selection["cohort_manifest"]["path"])),
        cohort_sha256=str(selection["cohort_manifest"]["sha256"]),
        expected_model_id="chk1",
        expected_scope="infrastructure_smoke",
        max_num_seqs=selected,
    )["results"]
    selected_projection = _projection(selected_rows, OUTPUT_FIELDS)
    smoke_projection = _projection(smoke_rows, OUTPUT_FIELDS)
    if (
        _projection(selected_rows, IDENTITY_FIELDS)
        != _projection(smoke_rows, IDENTITY_FIELDS)
        or selected_projection != smoke_projection
    ):
        raise DualDp1BenchmarkError(
            "selected benchmark does not exactly replay in fresh CHK1 smoke"
        )
    return {
        "status": "passed",
        "selected_max_num_seqs": selected,
        "rows": EXPECTED_CASES,
        "exact_generated_token_ids": True,
        "exact_generated_text_and_hashes": True,
        "exact_output_sha256": _sha256_value(smoke_projection),
        "selected_benchmark_run_manifest": copy.deepcopy(record["run_manifest"]),
        "smoke_run_manifest": copy.deepcopy(smoke_loaded["manifest_binding"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--cohort", type=Path, required=True)
    select.add_argument("--cohort-sha256", required=True)
    for candidate in CANDIDATES:
        select.add_argument(f"--manifest-{candidate}", type=Path, required=True)
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
                paths={
                    candidate: getattr(args, f"manifest_{candidate}")
                    for candidate in CANDIDATES
                },
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
                    "payload_sha256": value["integrity"]["payload_sha256"],
                }
            )
        )
        return 0
    except (
        DualDp1BenchmarkError,
        runner.DualDp1GenerationError,
        orchestrator.DualDp1OrchestrationError,
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
    "CANDIDATES",
    "DualDp1BenchmarkError",
    "OUTPUT_FIELDS",
    "load_and_validate_selection",
    "main",
    "select_and_seal",
    "validate_selected_smoke_replay",
]
