"""Seal and deep-validate dual-independent-DP1 three-model K5 suites."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.eval import eval_chk3_beta_core8_merged_vllm_k5_dual_dp1 as runner
from jobs.eval import select_chk3_beta_core8_vllm_k5_dual_dp1_benchmark as selector
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


SCHEMA_VERSION = "chk3-beta-core8-vllm-k5-dual-dp1-suite-v1"
MODEL_ORDER = ("chk1", "chk3", "chk0")
SCOPES = ("infrastructure_smoke", "formal_merged_panel")


class DualDp1SuiteError(RuntimeError):
    """A suite or its benchmark/replay gate is incomplete or inconsistent."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _binding(path: Path, *, payload_sha256: str | None = None) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise DualDp1SuiteError(f"bound path is a symlink: {unresolved}")
    path = unresolved.resolve()
    if not path.is_file():
        raise DualDp1SuiteError(f"bound file is missing: {path}")
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": runner.core._sha256_file(path),
    }
    if payload_sha256 is not None:
        result["payload_sha256"] = payload_sha256
    return result


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise DualDp1SuiteError(f"suite manifest is create-only: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(_canonical(dict(value)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_runs(
    *,
    suite_root: Path,
    cohort_path: Path,
    cohort_sha256: str,
    scope: str,
    max_num_seqs: int,
) -> dict[str, dict[str, Any]]:
    return {
        model_id: runner.load_and_validate_run(
            suite_root / model_id / "manifest.json",
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            expected_model_id=model_id,
            expected_scope=scope,
            max_num_seqs=max_num_seqs,
        )
        for model_id in MODEL_ORDER
    }


def _load_selection(
    path: Path, *, cohort_path: Path, cohort_sha256: str
) -> dict[str, Any]:
    return selector.load_and_validate_selection(
        path,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
    )


def _smoke_gate(
    *,
    selection: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    cross = selection.get("cross_candidate_equivalence")
    if (
        not isinstance(cross, Mapping)
        or cross.get("exact_outputs") is not True
        or cross.get("tuple_identities") != 40
    ):
        raise DualDp1SuiteError("8/12/16 exact benchmark parity is not sealed")
    try:
        replay = selector.validate_selected_smoke_replay(
            selection=selection,
            smoke_loaded=runs["chk1"],
        )
    except selector.DualDp1BenchmarkError as exc:
        raise DualDp1SuiteError(str(exc)) from exc
    return {
        "benchmark_candidate_exact_token_parity_8_12_16": True,
        "selected_chk1_smoke_exact_replay": True,
        "formal_generation_unblocked": True,
        "replay_evidence": replay,
    }


def _formal_authorization_from_smoke(
    smoke: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = smoke.get("manifest")
    manifest_binding = smoke.get("manifest_binding")
    if not isinstance(manifest, Mapping) or not isinstance(manifest_binding, Mapping):
        raise DualDp1SuiteError("official smoke authorization evidence is missing")
    selection = manifest.get("benchmark_selection")
    gates = manifest.get("generation_gates")
    if not isinstance(selection, Mapping) or not isinstance(gates, Mapping):
        raise DualDp1SuiteError("official smoke authorization contract is incomplete")
    return {
        "official_smoke_suite": copy.deepcopy(dict(manifest_binding)),
        "benchmark_selection": copy.deepcopy(dict(selection)),
        "generation_gates": copy.deepcopy(dict(gates)),
    }


def seal_suite(
    *,
    cohort_path: Path,
    cohort_sha256: str,
    output_dir: Path,
    scope: str,
    max_num_seqs: int,
    benchmark_selection: Path,
    smoke_suite_manifest: Path | None = None,
) -> dict[str, Any]:
    unresolved = output_dir.expanduser()
    if unresolved.is_symlink():
        raise DualDp1SuiteError("suite output root is a symlink")
    output_dir = unresolved.resolve()
    if scope not in SCOPES:
        raise DualDp1SuiteError("invalid suite scope")
    selection = _load_selection(
        benchmark_selection,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
    )
    selected = int(selection["selection"]["selected_max_num_seqs"])
    if max_num_seqs != selected:
        raise DualDp1SuiteError("suite max_num_seqs differs from sealed selection")
    runs = _load_runs(
        suite_root=output_dir,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
        scope=scope,
        max_num_seqs=max_num_seqs,
    )
    remediation_receipt = runs["chk1"]["manifest"].get("remediation_receipt")
    if not isinstance(remediation_receipt, Mapping) or any(
        runs[model_id]["manifest"].get("remediation_receipt") != remediation_receipt
        for model_id in MODEL_ORDER
    ):
        raise DualDp1SuiteError("model runs disagree on remediation receipt")
    if scope == "infrastructure_smoke":
        gate = _smoke_gate(selection=selection, runs=runs)
        smoke_binding = None
        formal_authorization = None
        if any(
            runs[model_id]["manifest"].get("formal_authorization") is not None
            for model_id in MODEL_ORDER
        ):
            raise DualDp1SuiteError("smoke runs must not carry formal authorization")
        rows_per_model = 40
        rows_per_shard = 20
    else:
        if smoke_suite_manifest is None:
            raise DualDp1SuiteError("formal suite requires the official smoke suite")
        smoke = load_and_validate_suite(
            smoke_suite_manifest,
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            expected_scope="infrastructure_smoke",
            max_num_seqs=max_num_seqs,
        )
        smoke_gate = smoke["manifest"].get("generation_gates")
        if (
            not isinstance(smoke_gate, Mapping)
            or smoke_gate.get("formal_generation_unblocked") is not True
            or smoke_gate.get("selected_chk1_smoke_exact_replay") is not True
        ):
            raise DualDp1SuiteError("official smoke did not seal the replay gate")
        gate = copy.deepcopy(dict(smoke_gate))
        smoke_binding = copy.deepcopy(smoke["manifest_binding"])
        formal_authorization = _formal_authorization_from_smoke(smoke)
        if any(
            runs[model_id]["manifest"].get("formal_authorization")
            != formal_authorization
            for model_id in MODEL_ORDER
        ):
            raise DualDp1SuiteError(
                "formal runs do not bind the exact official smoke authorization"
            )
        rows_per_model = 10240
        rows_per_shard = 5120
    payload = seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "created_at_utc": runner.core._utc_now(),
            "evaluation_id": runner.EVALUATION_ID,
            "evaluation_scope": scope,
            "model_order": list(MODEL_ORDER),
            "cohort": {
                "path": str(cohort_path.resolve()),
                "sha256": cohort_sha256,
            },
            "generation_contract": runner.sampling_contract(max_num_seqs=max_num_seqs),
            "remediation_receipt": copy.deepcopy(dict(remediation_receipt)),
            "generation_gates": gate,
            "benchmark_selection": _binding(
                benchmark_selection,
                payload_sha256=selection["integrity"]["payload_sha256"],
            ),
            "official_smoke_suite": smoke_binding,
            "formal_authorization": copy.deepcopy(formal_authorization),
            "coverage": {
                "models": 3,
                "rows_per_model": rows_per_model,
                "rows_per_shard": rows_per_shard,
                "total_rows": rows_per_model * 3,
                "input_truncation_rows": sum(
                    bool(row["input_truncated"])
                    for run in runs.values()
                    for row in run["results"]
                ),
            },
            "run_manifests": {
                model_id: copy.deepcopy(runs[model_id]["manifest_binding"])
                for model_id in MODEL_ORDER
            },
            "implementation_sources": {
                "suite_sealer": _binding(Path(__file__).resolve()),
                "runner": _binding(Path(runner.__file__).resolve()),
                "benchmark_selector": _binding(Path(selector.__file__).resolve()),
            },
        }
    )
    if payload["coverage"]["input_truncation_rows"] != 0:
        raise DualDp1SuiteError("suite contains input truncation")
    _write_new(output_dir / "manifest.json", payload)
    return payload


def load_and_validate_suite(
    manifest_path: Path,
    *,
    cohort_path: Path,
    cohort_sha256: str,
    expected_scope: str,
    max_num_seqs: int,
) -> dict[str, Any]:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DualDp1SuiteError("suite manifest is missing or a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise DualDp1SuiteError("suite manifest is not an object")
    payload_sha = validate_manifest_integrity(manifest)
    selection_binding = manifest.get("benchmark_selection")
    if not isinstance(selection_binding, Mapping):
        raise DualDp1SuiteError("suite benchmark selection binding is missing")
    selection_path = Path(str(selection_binding.get("path")))
    selection = _load_selection(
        selection_path,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
    )
    selected = int(selection["selection"]["selected_max_num_seqs"])
    if max_num_seqs != selected:
        raise DualDp1SuiteError("suite selection/max_num_seqs drift")
    runs = _load_runs(
        suite_root=manifest_path.resolve().parent,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
        scope=expected_scope,
        max_num_seqs=max_num_seqs,
    )
    remediation_receipt = runs["chk1"]["manifest"].get("remediation_receipt")
    rows_per_model = 40 if expected_scope == "infrastructure_smoke" else 10240
    rows_per_shard = rows_per_model // 2
    expected_smoke: dict[str, Any] | None = None
    if expected_scope == "infrastructure_smoke":
        expected_gate = _smoke_gate(selection=selection, runs=runs)
        expected_formal_authorization = None
    else:
        smoke_binding = manifest.get("official_smoke_suite")
        if not isinstance(smoke_binding, Mapping):
            raise DualDp1SuiteError("formal suite smoke binding is missing")
        expected_smoke = load_and_validate_suite(
            Path(str(smoke_binding.get("path"))),
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            expected_scope="infrastructure_smoke",
            max_num_seqs=max_num_seqs,
        )
        expected_gate = copy.deepcopy(
            dict(expected_smoke["manifest"]["generation_gates"])
        )
        expected_formal_authorization = _formal_authorization_from_smoke(expected_smoke)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("evaluation_id") != runner.EVALUATION_ID
        or manifest.get("evaluation_scope") != expected_scope
        or manifest.get("model_order") != list(MODEL_ORDER)
        or manifest.get("cohort")
        != {"path": str(cohort_path.resolve()), "sha256": cohort_sha256}
        or manifest.get("generation_contract")
        != runner.sampling_contract(max_num_seqs=max_num_seqs)
        or not isinstance(remediation_receipt, Mapping)
        or manifest.get("remediation_receipt") != remediation_receipt
        or any(
            runs[model_id]["manifest"].get("remediation_receipt") != remediation_receipt
            for model_id in MODEL_ORDER
        )
        or manifest.get("generation_gates") != expected_gate
        or manifest.get("formal_authorization") != expected_formal_authorization
        or any(
            runs[model_id]["manifest"].get("formal_authorization")
            != expected_formal_authorization
            for model_id in MODEL_ORDER
        )
        or manifest.get("coverage")
        != {
            "models": 3,
            "rows_per_model": rows_per_model,
            "rows_per_shard": rows_per_shard,
            "total_rows": rows_per_model * 3,
            "input_truncation_rows": 0,
        }
        or manifest.get("run_manifests")
        != {model_id: runs[model_id]["manifest_binding"] for model_id in MODEL_ORDER}
        or manifest.get("benchmark_selection")
        != _binding(
            selection_path,
            payload_sha256=selection["integrity"]["payload_sha256"],
        )
        or (
            expected_scope == "formal_merged_panel"
            and manifest.get("official_smoke_suite")
            != expected_smoke["manifest_binding"]
        )
        or (
            expected_scope == "infrastructure_smoke"
            and manifest.get("official_smoke_suite") is not None
        )
        or manifest.get("implementation_sources")
        != {
            "suite_sealer": _binding(Path(__file__).resolve()),
            "runner": _binding(Path(runner.__file__).resolve()),
            "benchmark_selector": _binding(Path(selector.__file__).resolve()),
        }
    ):
        raise DualDp1SuiteError("sealed suite contract drift")
    return {
        "manifest": manifest,
        "manifest_binding": _binding(manifest_path, payload_sha256=payload_sha),
        "runs": runs,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--cohort", type=Path, required=True)
    seal.add_argument("--cohort-sha256", required=True)
    seal.add_argument("--output-dir", type=Path, required=True)
    seal.add_argument("--scope", choices=SCOPES, required=True)
    seal.add_argument("--max-num-seqs", type=int, choices=(8, 12, 16), required=True)
    seal.add_argument("--benchmark-selection", type=Path, required=True)
    seal.add_argument("--smoke-suite-manifest", type=Path)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--cohort", type=Path, required=True)
    validate.add_argument("--cohort-sha256", required=True)
    validate.add_argument("--scope", choices=SCOPES, required=True)
    validate.add_argument(
        "--max-num-seqs", type=int, choices=(8, 12, 16), required=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "seal":
            result = seal_suite(
                cohort_path=args.cohort,
                cohort_sha256=args.cohort_sha256,
                output_dir=args.output_dir,
                scope=args.scope,
                max_num_seqs=args.max_num_seqs,
                benchmark_selection=args.benchmark_selection,
                smoke_suite_manifest=args.smoke_suite_manifest,
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
                    "status": "complete" if args.command == "seal" else "valid",
                    "scope": result["evaluation_scope"],
                    "rows": result["coverage"]["total_rows"],
                    "formal_generation_unblocked": result["generation_gates"][
                        "formal_generation_unblocked"
                    ],
                }
            )
        )
        return 0
    except (
        DualDp1SuiteError,
        selector.DualDp1BenchmarkError,
        runner.DualDp1GenerationError,
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
    "DualDp1SuiteError",
    "SCHEMA_VERSION",
    "load_and_validate_suite",
    "main",
    "seal_suite",
]
