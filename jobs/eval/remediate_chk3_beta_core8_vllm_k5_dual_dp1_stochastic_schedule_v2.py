"""Authorize the schedule-sensitive, speed-only fixed-16 vLLM amendment.

This CPU-only receipt preserves the failed v3 exact-token selector and all
four historical CHK1 runs (8, 12, 16, and a fresh 16 replay).  It authorizes a
new evaluation ID and fresh v4 roots in which stochastic token identity across
batch concurrency or independent engine launches is diagnostic rather than a
gate.  No historical generation row is copied into the new formal matrix.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any, IO

from jobs.eval import eval_chk3_beta_core8_merged_vllm_k5_dual_dp1 as v3_runner
from jobs.eval import (
    remediate_chk3_beta_core8_vllm_k5_dp2_to_dual_dp1 as v3_remediation,
)
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = (
    ROOT / "output/evaluation/main/chk3_beta_core8_merged_n2048_vllm_k5_20260816_v1"
)
COHORT = RUN_ROOT / "preparation/cohort_n2048_k5.v1.json"
COHORT_SHA256 = "a815b5af8e6b33e3d1a2b211e346393f155d1d45acaafae80b822ee08128ab8e"
V3_BENCHMARK_ROOT = RUN_ROOT / "benchmark_max_num_seqs_chk1_core8_k5_v3_dual_dp1"
V3_SELECTION = V3_BENCHMARK_ROOT / "selection.v1.json"
V3_RUNS = {
    "max_num_seqs_8": V3_BENCHMARK_ROOT / "max_num_seqs_8/chk1/manifest.json",
    "max_num_seqs_12": V3_BENCHMARK_ROOT / "max_num_seqs_12/chk1/manifest.json",
    "max_num_seqs_16": V3_BENCHMARK_ROOT / "max_num_seqs_16/chk1/manifest.json",
    "max_num_seqs_16_replay": (
        V3_BENCHMARK_ROOT / "max_num_seqs_16_replay/chk1/manifest.json"
    ),
}
EXPECTED_V3_RUN_SHA256 = {
    "max_num_seqs_8": "1a5ccfa30d1b6ec48739cdd981d53f6d1c66b9757c23770609edd47dcb1352ac",
    "max_num_seqs_12": "2beb3a2bf0ddcc3b34e6a54d7052d99481ef108a79e83a37e908603c064a7a9c",
    "max_num_seqs_16": "8d76e5283a895a3dd0db8a0d3926900772ae745e29a3275b88e5c1da88cf1cf2",
    "max_num_seqs_16_replay": "2940eb801900b0c966c28762f898733927f3bfff6887af62b061c8e4ffbbbec1",
}
EXPECTED_V3_SOURCE_SHA256 = {
    ROOT / "jobs/eval/eval_chk3_beta_core8_merged_vllm_k5_dual_dp1.py": (
        "8128a6e4a928d695c8ca2a530899189f1e691c5c339f6c9d2c1c4a04d9fc872a"
    ),
    ROOT / "jobs/eval/orchestrate_chk3_beta_core8_merged_vllm_k5_dual_dp1.py": (
        "87078f8328bc1cdd4fffcaa9f4a3286117da5fa1553d745f39ffe1efea5f3ac3"
    ),
    ROOT / "jobs/eval/select_chk3_beta_core8_vllm_k5_dual_dp1_benchmark.py": (
        "a3b0f3ccf5f78ce2b5722dc5d2c305c3fad05107a4073e4008a4105f261c05e1"
    ),
    ROOT / "jobs/eval/seal_chk3_beta_core8_vllm_k5_dual_dp1_suite.py": (
        "be43ab2e4f098948cc921d17a15814d719ff316432004724a95a685b02099a5b"
    ),
    ROOT / "run/eval_chk3_beta_core8_merged_vllm_k5_dual_dp1.sh": (
        "5020eba70d96a9371bde5841877d28b95391eaae0f4f5280a3e4ff05f29b5a11"
    ),
}
V4_POLICY_ROOT = RUN_ROOT / "benchmark_stochastic_schedule_v2_fixed16"
V4_SMOKE_ROOT = (
    RUN_ROOT / "generation_smoke_core8_k5_three_models_v4_stochastic_schedule_v2"
)
V4_FORMAL_ROOT = (
    RUN_ROOT / "generation_formal_n2048_k5_three_models_v4_stochastic_schedule_v2"
)
V4_DOCUMENT_ROOT = RUN_ROOT / "meeting_documents_n3840_v4_stochastic_schedule_v2"
FRESH_V4_ROOTS = (V4_POLICY_ROOT, V4_SMOKE_ROOT, V4_FORMAL_ROOT, V4_DOCUMENT_ROOT)
DEFAULT_RECEIPT = (
    RUN_ROOT / "migration/vllm_dual_dp1_stochastic_schedule_v2_amendment_receipt.json"
)
V4_SOURCES = {
    "amendment": Path(__file__).resolve(),
    "runner": ROOT
    / "jobs/eval/eval_chk3_beta_core8_merged_vllm_k5_dual_dp1_stochastic_schedule_v2.py",
    "orchestrator": ROOT
    / "jobs/eval/orchestrate_chk3_beta_core8_merged_vllm_k5_dual_dp1_stochastic_schedule_v2.py",
    "selector": ROOT
    / "jobs/eval/select_chk3_beta_core8_vllm_k5_dual_dp1_stochastic_schedule_v2.py",
    "suite_sealer": ROOT
    / "jobs/eval/seal_chk3_beta_core8_vllm_k5_dual_dp1_stochastic_schedule_v2_suite.py",
    "launcher": ROOT
    / "run/eval_chk3_beta_core8_merged_vllm_k5_dual_dp1_stochastic_schedule_v2.sh",
}
GPU_LOCKS = (
    Path("/tmp/fomc_trainer_chk3_beta_core8_vllm_k5_gpu0.lock"),
    Path("/tmp/fomc_trainer_chk3_beta_core8_vllm_k5_gpu1.lock"),
)
EVALUATION_ID = (
    "chk3-beta-core8-merged-1993-2025-n2048-vllm-k5-"
    "dual-independent-dp1-stochastic-schedule-v2"
)
SCHEMA_VERSION = "chk3-beta-core8-vllm-k5-dual-dp1-stochastic-schedule-amendment-v2"


class StochasticScheduleAmendmentError(RuntimeError):
    """The v3 evidence cannot authorize a fresh v4 evaluation."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_file(path: Path) -> str:
    return v3_runner.core._sha256_file(path)


def _binding(path: Path, *, payload_sha256: str | None = None) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise StochasticScheduleAmendmentError(f"bound path is symlink: {path}")
    path = unresolved.resolve()
    if not path.is_file():
        raise StochasticScheduleAmendmentError(f"bound file is missing: {path}")
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if payload_sha256 is not None:
        result["payload_sha256"] = payload_sha256
    return result


def _read_sealed(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StochasticScheduleAmendmentError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise StochasticScheduleAmendmentError(f"sealed artifact is not object: {path}")
    try:
        payload_sha = validate_manifest_integrity(value)
    except Exception as exc:
        raise StochasticScheduleAmendmentError(
            f"sealed artifact integrity failed: {path}"
        ) from exc
    return value, _binding(path, payload_sha256=payload_sha)


def _audit_v3_sources() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path, expected_sha in EXPECTED_V3_SOURCE_SHA256.items():
        bound = _binding(path)
        if bound["sha256"] != expected_sha:
            raise StochasticScheduleAmendmentError(f"v3 source drift: {path}")
        result[path.name] = bound
    return result


def _run_stats(
    name: str, max_num_seqs: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = V3_RUNS[name]
    if _binding(path)["sha256"] != EXPECTED_V3_RUN_SHA256[name]:
        raise StochasticScheduleAmendmentError(f"historical run SHA drift: {name}")
    loaded = v3_runner.load_and_validate_run(
        path,
        cohort_path=COHORT,
        cohort_sha256=COHORT_SHA256,
        expected_model_id="chk1",
        expected_scope="infrastructure_smoke",
        max_num_seqs=max_num_seqs,
    )
    manifest = loaded["manifest"]
    rows = loaded["results"]
    normal = sum(
        row.get("status") == "ok"
        and row.get("finish_reason") == "eos"
        and row.get("hit_eos") is True
        and row.get("cap_reached") is False
        for row in rows
    )
    tokens = sum(len(row["generated_token_ids"]) for row in rows)
    runtime = manifest["runtime"]
    if (
        len(rows) != 40
        or manifest["coverage"].get("shard_cases") != {"shard0": 20, "shard1": 20}
        or manifest["coverage"].get("input_truncation_cases") != 0
        or normal != 40
        or runtime.get("resume_counts") != {"shard0": 0, "shard1": 0}
        or runtime.get("aggregate_generated_output_tokens") != tokens
    ):
        raise StochasticScheduleAmendmentError(f"historical run gates failed: {name}")
    return (
        {
            "manifest": loaded["manifest_binding"],
            "canonical_generations": manifest["artifacts"]["canonical_generations"],
            "max_num_seqs": max_num_seqs,
            "rows": 40,
            "shard_rows": {"shard0": 20, "shard1": 20},
            "normal_eos_finishes": normal,
            "input_truncation_cases": 0,
            "aggregate_generated_output_tokens": tokens,
            "aggregate_output_tokens_per_second": runtime[
                "aggregate_output_tokens_per_second"
            ],
            "parallel_generation_wall_seconds": runtime[
                "parallel_generation_wall_seconds"
            ],
        },
        rows,
    )


def _identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("sample_id"),
        row.get("replicate_id"),
        row.get("row_seed"),
        row.get("absolute_case_index"),
        row.get("shard_id"),
        row.get("prompt_sha256"),
        row.get("prompt_token_ids_sha256"),
    )


def _output(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("generated_token_ids_sha256"),
        row.get("completion_sha256"),
        row.get("answer_sha256"),
        row.get("finish_reason"),
    )


def _diagnostic(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if [_identity(row) for row in left] != [_identity(row) for row in right]:
        raise StochasticScheduleAmendmentError("historical input/seed tuple drift")
    exact = [_output(a) == _output(b) for a, b in zip(left, right, strict=True)]
    return {
        "rows": len(left),
        "exact_output_rows": sum(exact),
        "different_output_rows": len(left) - sum(exact),
        "by_shard": {
            f"shard{shard}": {
                "rows": sum(int(row["shard_id"]) == shard for row in left),
                "exact_output_rows": sum(
                    same and int(row["shard_id"]) == shard
                    for same, row in zip(exact, left, strict=True)
                ),
            }
            for shard in (0, 1)
        },
        "blocking": False,
    }


def _audit_historical_runs() -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for name, max_num_seqs in (
        ("max_num_seqs_8", 8),
        ("max_num_seqs_12", 12),
        ("max_num_seqs_16", 16),
        ("max_num_seqs_16_replay", 16),
    ):
        stats[name], rows[name] = _run_stats(name, max_num_seqs)
    reference = [_identity(row) for row in rows["max_num_seqs_8"]]
    if any([_identity(row) for row in rows[name]] != reference for name in rows):
        raise StochasticScheduleAmendmentError("historical tuple identities disagree")
    candidate_tokens = [
        stats[f"max_num_seqs_{value}"]["aggregate_generated_output_tokens"]
        for value in (8, 12, 16)
    ]
    replay_tokens = [
        stats["max_num_seqs_16"]["aggregate_generated_output_tokens"],
        stats["max_num_seqs_16_replay"]["aggregate_generated_output_tokens"],
    ]
    candidate_ratio = max(candidate_tokens) / min(candidate_tokens)
    replay_ratio = max(replay_tokens) / min(replay_tokens)
    if candidate_ratio > 1.05 or replay_ratio > 1.05:
        raise StochasticScheduleAmendmentError("historical token-volume gate failed")
    rates = {
        value: stats[f"max_num_seqs_{value}"]["aggregate_output_tokens_per_second"]
        for value in (8, 12, 16)
    }
    if max(rates, key=lambda value: (rates[value], value)) != 16:
        raise StochasticScheduleAmendmentError("historical speed argmax is not 16")
    return {
        "runs": stats,
        "exact_input_seed_tuple_identity": True,
        "cross_candidate_diagnostics": {
            "8_vs_12": _diagnostic(rows["max_num_seqs_8"], rows["max_num_seqs_12"]),
            "8_vs_16": _diagnostic(rows["max_num_seqs_8"], rows["max_num_seqs_16"]),
            "12_vs_16": _diagnostic(rows["max_num_seqs_12"], rows["max_num_seqs_16"]),
        },
        "same_config_replay_diagnostic": _diagnostic(
            rows["max_num_seqs_16"], rows["max_num_seqs_16_replay"]
        ),
        "candidate_token_volume_ratio": candidate_ratio,
        "same_config_replay_token_volume_ratio": replay_ratio,
        "all_token_volume_ratios_lte_1_05": True,
        "candidate_rates": {str(key): value for key, value in rates.items()},
        "speed_only_argmax": 16,
    }


def _audit_v4_sources() -> dict[str, Any]:
    return {name: _binding(path) for name, path in V4_SOURCES.items()}


def _audit_fresh_roots() -> dict[str, Any]:
    rows = [
        {"path": str(path.resolve()), "lexists_at_seal": os.path.lexists(path)}
        for path in FRESH_V4_ROOTS
    ]
    if any(row["lexists_at_seal"] for row in rows):
        raise StochasticScheduleAmendmentError("every v4 output root must be fresh")
    return {"all_absent_at_seal": True, "roots": rows}


def _open_lock(path: Path) -> IO[str]:
    if path.is_symlink():
        raise StochasticScheduleAmendmentError(f"GPU lock is symlink: {path}")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    handle = os.fdopen(descriptor, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise StochasticScheduleAmendmentError(f"GPU lock held: {path}") from exc
    return handle


def _audit_gpu_idle_and_locks() -> dict[str, Any]:
    with ExitStack() as stack:
        handles = [stack.enter_context(_open_lock(path)) for path in GPU_LOCKS]
        del handles
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode != 0:
            raise StochasticScheduleAmendmentError("cannot audit GPU compute processes")
        processes = [
            line.strip() for line in completed.stdout.splitlines() if line.strip()
        ]
        if processes:
            raise StochasticScheduleAmendmentError(
                f"GPUs must be idle while sealing amendment: {processes}"
            )
        return {
            "canonical_gpu_locks_acquired_atomically": True,
            "compute_processes": [],
            "gpu_work_launched": False,
        }


def build_receipt() -> dict[str, Any]:
    if _binding(COHORT)["sha256"] != COHORT_SHA256:
        raise StochasticScheduleAmendmentError("cohort SHA drift")
    if os.path.lexists(V3_SELECTION):
        raise StochasticScheduleAmendmentError(
            "failed v3 selector must not have a selection artifact"
        )
    prior = v3_remediation.load_and_validate_receipt(v3_remediation.DEFAULT_RECEIPT)
    return seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "created_at_utc": v3_runner.core._utc_now(),
            "evaluation_id": EVALUATION_ID,
            "cohort": _binding(COHORT),
            "prior_v3_remediation_receipt": prior["receipt_binding"],
            "v3_failed_selector": {
                "selection_path": str(V3_SELECTION.resolve()),
                "selection_absent_at_seal": True,
                "historical_sources": _audit_v3_sources(),
            },
            "historical_evidence": _audit_historical_runs(),
            "amendment_contract": {
                "selected_max_num_seqs": 16,
                "selection_rule": "strict_argmax_aggregate_output_tokens_per_second",
                "tie_preference": None,
                "batch_configuration_is_generation_contract": True,
                "classification": "schedule_sensitive_seeded_sampling",
                "token_identity_across_batch_concurrency_or_engine_launches_required": False,
                "token_identity_differences_are_diagnostic_and_nonblocking": True,
                "hard_gates": [
                    "exact_inputs_seeds_model_common_config_topology_coverage",
                    "zero_input_truncation_and_infrastructure_error",
                    "normal_eos_finish_40_of_40",
                    "aggregate_output_token_volume_ratio_lte_1_05",
                ],
                "resume_rule": "never_resubmit_fsynced_wal_tuple_only_missing_keys",
                "historical_rows_reused_in_v4_formal": 0,
            },
            "implementation_sources": _audit_v4_sources(),
            "fresh_v4_roots": _audit_fresh_roots(),
            "gpu_and_lock_gate": _audit_gpu_idle_and_locks(),
            "safety": {
                "receipt_create_only": True,
                "receipt_mode": "0o444",
                "v3_directories_deleted_or_modified": False,
                "v3_failed_selector_overwritten": False,
                "gpu_work_launched_by_amendment": False,
            },
        }
    )


def _write_readonly(path: Path, value: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise StochasticScheduleAmendmentError(f"receipt is create-only: {path}")
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


def create_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    value = build_receipt()
    _write_readonly(path, value)
    return load_and_validate_receipt(path)


def load_and_validate_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StochasticScheduleAmendmentError("amendment receipt missing or symlink")
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise StochasticScheduleAmendmentError("amendment receipt is not read-only")
    value, binding = _read_sealed(path)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "complete"
        or value.get("evaluation_id") != EVALUATION_ID
        or value.get("cohort") != _binding(COHORT)
        or value.get("safety", {}).get("receipt_create_only") is not True
        or value.get("safety", {}).get("receipt_mode") != "0o444"
        or value.get("fresh_v4_roots", {}).get("all_absent_at_seal") is not True
        or value.get("amendment_contract", {}).get("selected_max_num_seqs") != 16
        or value.get("amendment_contract", {}).get(
            "token_identity_differences_are_diagnostic_and_nonblocking"
        )
        is not True
        or value.get("amendment_contract", {}).get("resume_rule")
        != "never_resubmit_fsynced_wal_tuple_only_missing_keys"
    ):
        raise StochasticScheduleAmendmentError("amendment receipt contract drift")
    prior = v3_remediation.load_and_validate_receipt(v3_remediation.DEFAULT_RECEIPT)
    if value.get("prior_v3_remediation_receipt") != prior["receipt_binding"]:
        raise StochasticScheduleAmendmentError("prior remediation binding drift")
    if (
        value.get("v3_failed_selector", {}).get("historical_sources")
        != _audit_v3_sources()
    ):
        raise StochasticScheduleAmendmentError("historical source binding drift")
    historical = value.get("historical_evidence", {}).get("runs")
    if not isinstance(historical, Mapping):
        raise StochasticScheduleAmendmentError("historical run bindings missing")
    for name, expected_sha in EXPECTED_V3_RUN_SHA256.items():
        if (
            historical.get(name, {}).get("manifest", {}).get("sha256") != expected_sha
            or _binding(V3_RUNS[name])["sha256"] != expected_sha
        ):
            raise StochasticScheduleAmendmentError(f"historical run drift: {name}")
    if value.get("implementation_sources") != _audit_v4_sources():
        raise StochasticScheduleAmendmentError("v4 implementation source drift")
    result = dict(value)
    result["receipt_binding"] = binding
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    validate = sub.add_parser("validate")
    validate.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        value = (
            create_receipt(args.receipt)
            if args.command == "create"
            else load_and_validate_receipt(args.receipt)
        )
        print(
            _canonical(
                {
                    "status": "complete" if args.command == "create" else "valid",
                    "path": value["receipt_binding"]["path"],
                    "sha256": value["receipt_binding"]["sha256"],
                    "payload_sha256": value["receipt_binding"]["payload_sha256"],
                    "selected_max_num_seqs": 16,
                    "same_config_exact_rows": value["historical_evidence"][
                        "same_config_replay_diagnostic"
                    ]["exact_output_rows"],
                }
            )
        )
        return 0
    except (StochasticScheduleAmendmentError, OSError, ValueError) as exc:
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
    "DEFAULT_RECEIPT",
    "EVALUATION_ID",
    "StochasticScheduleAmendmentError",
    "build_receipt",
    "create_receipt",
    "load_and_validate_receipt",
    "main",
]
