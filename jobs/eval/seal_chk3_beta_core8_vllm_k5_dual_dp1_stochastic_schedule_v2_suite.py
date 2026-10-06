"""Seal and deep-validate fixed-16 schedule-sensitive three-model suites."""

from __future__ import annotations

import argparse
import copy
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.eval import (
    eval_chk3_beta_core8_merged_vllm_k5_dual_dp1_stochastic_schedule_v2 as runner,
)
from jobs.eval import (
    remediate_chk3_beta_core8_vllm_k5_dual_dp1_stochastic_schedule_v2 as amendment,
)
from jobs.eval import (
    select_chk3_beta_core8_vllm_k5_dual_dp1_stochastic_schedule_v2 as selector,
)
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


SCHEMA_VERSION = "chk3-beta-core8-vllm-k5-dual-dp1-stochastic-schedule-suite-v2"
MODEL_ORDER = ("chk1", "chk3", "chk0")
SCOPES = ("infrastructure_smoke", "formal_merged_panel")
FIXED_MAX_NUM_SEQS = 16
GENERATION_GATES = {
    "speed_only_fixed_max_num_seqs_16": True,
    "schedule_sensitive_stochastic_token_identity_nonblocking": True,
    "exact_inputs_seeds_config_topology_coverage": True,
    "normal_finish_and_token_volume_gate": True,
    "formal_generation_unblocked": True,
}


class StochasticScheduleSuiteError(RuntimeError):
    """A fixed-16 suite or its authorization chain is invalid."""


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
        raise StochasticScheduleSuiteError(f"bound path is symlink: {path}")
    path = unresolved.resolve()
    if not path.is_file():
        raise StochasticScheduleSuiteError(f"bound file missing: {path}")
    result: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": runner.core._sha256_file(path),
    }
    if payload_sha256 is not None:
        result["payload_sha256"] = payload_sha256
    return result


def _write_readonly(path: Path, value: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise StochasticScheduleSuiteError(f"suite manifest is create-only: {path}")
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


def _load_selection(
    path: Path, *, cohort_path: Path, cohort_sha256: str
) -> dict[str, Any]:
    value = selector.load_and_validate_selection(
        path,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
    )
    if (
        value.get("evaluation_id") != runner.EVALUATION_ID
        or value.get("selection", {}).get("selected_max_num_seqs") != FIXED_MAX_NUM_SEQS
        or value.get("generation_gates") != GENERATION_GATES
    ):
        raise StochasticScheduleSuiteError("speed selection authorization drift")
    return value


def _load_runs(
    *,
    suite_root: Path,
    cohort_path: Path,
    cohort_sha256: str,
    scope: str,
) -> dict[str, dict[str, Any]]:
    return {
        model_id: runner.load_and_validate_run(
            suite_root / model_id / "manifest.json",
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            expected_model_id=model_id,
            expected_scope=scope,
            max_num_seqs=FIXED_MAX_NUM_SEQS,
        )
        for model_id in MODEL_ORDER
    }


def _schedule_diagnostic(selection: Mapping[str, Any]) -> dict[str, Any]:
    replay = selection.get("same_config_replay")
    cross = selection.get("cross_candidate_diagnostics")
    semantics = selection.get("stochastic_semantics")
    if (
        not isinstance(replay, Mapping)
        or not isinstance(cross, Mapping)
        or not isinstance(semantics, Mapping)
        or replay.get("diagnostic_only_nonblocking") is not True
        or replay.get("within_five_percent_token_volume") is not True
        or replay.get("diagnostic", {}).get("exact_output_identity_required")
        is not False
        or semantics.get("token_identity_differences_are_nonblocking") is not True
    ):
        raise StochasticScheduleSuiteError("schedule-sensitivity diagnostic drift")
    return {
        "classification": "schedule_sensitive_seeded_sampling",
        "same_config_replay": copy.deepcopy(dict(replay)),
        "cross_candidate_diagnostics": copy.deepcopy(dict(cross)),
        "exact_token_identity_is_acceptance_gate": False,
        "normal_finish_and_token_volume_gate_passed": True,
    }


def _formal_authorization_from_smoke_loaded(
    smoke: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = smoke.get("manifest")
    binding = smoke.get("manifest_binding")
    if not isinstance(manifest, Mapping) or not isinstance(binding, Mapping):
        raise StochasticScheduleSuiteError("official smoke evidence missing")
    speed = manifest.get("speed_selection")
    diagnostic = manifest.get("schedule_sensitivity_diagnostic")
    gates = manifest.get("generation_gates")
    if (
        not isinstance(speed, Mapping)
        or not isinstance(diagnostic, Mapping)
        or gates != GENERATION_GATES
    ):
        raise StochasticScheduleSuiteError("official smoke authorization incomplete")
    return {
        "official_smoke_suite": copy.deepcopy(dict(binding)),
        "speed_selection": copy.deepcopy(dict(speed)),
        "schedule_sensitivity_diagnostic": copy.deepcopy(dict(diagnostic)),
        "generation_gates": copy.deepcopy(dict(gates)),
    }


def _expected_payload(
    *,
    created_at_utc: str,
    output_dir: Path,
    cohort_path: Path,
    cohort_sha256: str,
    scope: str,
    selection_path: Path,
    selection: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
    smoke_binding: Mapping[str, Any] | None,
    formal_authorization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    amendment_value = amendment.load_and_validate_receipt(amendment.DEFAULT_RECEIPT)
    remediation_binding = amendment_value["receipt_binding"]
    if selection.get("amendment_receipt") != remediation_binding:
        raise StochasticScheduleSuiteError("selection amendment binding drift")
    if any(
        runs[model_id]["manifest"].get("remediation_receipt") != remediation_binding
        for model_id in MODEL_ORDER
    ):
        raise StochasticScheduleSuiteError("model runs disagree on amendment receipt")
    expected_rows = 40 if scope == "infrastructure_smoke" else 10240
    expected_shard_rows = 20 if scope == "infrastructure_smoke" else 5120
    expected_auth = None if scope == "infrastructure_smoke" else formal_authorization
    if any(
        runs[model_id]["manifest"].get("formal_authorization") != expected_auth
        for model_id in MODEL_ORDER
    ):
        raise StochasticScheduleSuiteError("run formal authorization drift")
    speed_binding = _binding(
        selection_path,
        payload_sha256=str(selection["integrity"]["payload_sha256"]),
    )
    diagnostic = _schedule_diagnostic(selection)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": created_at_utc,
        "evaluation_id": runner.EVALUATION_ID,
        "evaluation_scope": scope,
        "model_order": list(MODEL_ORDER),
        "cohort": {
            "path": str(cohort_path.resolve()),
            "sha256": cohort_sha256,
        },
        "generation_contract": runner.sampling_contract(
            max_num_seqs=FIXED_MAX_NUM_SEQS
        ),
        "remediation_receipt": copy.deepcopy(remediation_binding),
        "speed_selection": speed_binding,
        "schedule_sensitivity_diagnostic": diagnostic,
        "generation_gates": copy.deepcopy(GENERATION_GATES),
        "official_smoke_suite": (
            None if smoke_binding is None else copy.deepcopy(dict(smoke_binding))
        ),
        "formal_authorization": (
            None
            if formal_authorization is None
            else copy.deepcopy(dict(formal_authorization))
        ),
        "coverage": {
            "models": 3,
            "rows_per_model": expected_rows,
            "rows_per_shard": expected_shard_rows,
            "total_rows": expected_rows * 3,
            "input_truncation_rows": sum(
                bool(row["input_truncated"])
                for run in runs.values()
                for row in run["results"]
            ),
            "normal_finish_rows": sum(
                row.get("status") == "ok"
                and row.get("finish_reason") in {"eos", "length"}
                and (
                    (
                        row.get("finish_reason") == "eos"
                        and row.get("hit_eos") is True
                        and row.get("cap_reached") is False
                    )
                    or (
                        row.get("finish_reason") == "length"
                        and row.get("hit_eos") is False
                        and row.get("cap_reached") is True
                    )
                )
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
            "selector": _binding(Path(selector.__file__).resolve()),
            "amendment": _binding(Path(amendment.__file__).resolve()),
        },
    }
    if (
        payload["coverage"]["input_truncation_rows"] != 0
        or payload["coverage"]["normal_finish_rows"] != expected_rows * 3
    ):
        raise StochasticScheduleSuiteError("suite truncation/normal-finish gate failed")
    return payload


def seal_suite(
    *,
    cohort_path: Path,
    cohort_sha256: str,
    output_dir: Path,
    scope: str,
    max_num_seqs: int,
    speed_selection: Path,
    smoke_suite_manifest: Path | None = None,
) -> dict[str, Any]:
    if max_num_seqs != FIXED_MAX_NUM_SEQS:
        raise StochasticScheduleSuiteError("v2 suite requires max_num_seqs=16")
    unresolved = output_dir.expanduser()
    if unresolved.is_symlink():
        raise StochasticScheduleSuiteError("suite output root is symlink")
    output_dir = unresolved.resolve()
    if scope not in SCOPES:
        raise StochasticScheduleSuiteError("invalid suite scope")
    selection = _load_selection(
        speed_selection,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
    )
    runs = _load_runs(
        suite_root=output_dir,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
        scope=scope,
    )
    smoke_binding: Mapping[str, Any] | None = None
    formal_authorization: Mapping[str, Any] | None = None
    if scope == "formal_merged_panel":
        if smoke_suite_manifest is None:
            raise StochasticScheduleSuiteError("formal suite requires official smoke")
        smoke = load_and_validate_suite(
            smoke_suite_manifest,
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            expected_scope="infrastructure_smoke",
            max_num_seqs=FIXED_MAX_NUM_SEQS,
        )
        smoke_binding = smoke["manifest_binding"]
        formal_authorization = _formal_authorization_from_smoke_loaded(smoke)
    elif smoke_suite_manifest is not None:
        raise StochasticScheduleSuiteError("smoke cannot bind another smoke suite")
    value = seal_manifest(
        _expected_payload(
            created_at_utc=runner.core._utc_now(),
            output_dir=output_dir,
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            scope=scope,
            selection_path=speed_selection,
            selection=selection,
            runs=runs,
            smoke_binding=smoke_binding,
            formal_authorization=formal_authorization,
        )
    )
    _write_readonly(output_dir / "manifest.json", value)
    return value


def load_and_validate_suite(
    manifest_path: Path,
    *,
    cohort_path: Path,
    cohort_sha256: str,
    expected_scope: str,
    max_num_seqs: int,
) -> dict[str, Any]:
    if max_num_seqs != FIXED_MAX_NUM_SEQS:
        raise StochasticScheduleSuiteError("v2 suite requires max_num_seqs=16")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise StochasticScheduleSuiteError("suite manifest missing or symlink")
    if stat.S_IMODE(manifest_path.stat().st_mode) != 0o444:
        raise StochasticScheduleSuiteError("suite manifest is not read-only")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise StochasticScheduleSuiteError("suite manifest is not object")
    payload_sha = validate_manifest_integrity(manifest)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("evaluation_id") != runner.EVALUATION_ID
        or manifest.get("evaluation_scope") != expected_scope
        or manifest.get("generation_gates") != GENERATION_GATES
    ):
        raise StochasticScheduleSuiteError("suite header/gates drift")
    speed_binding = manifest.get("speed_selection")
    if not isinstance(speed_binding, Mapping):
        raise StochasticScheduleSuiteError("speed selection binding missing")
    speed_path = Path(str(speed_binding.get("path")))
    selection = _load_selection(
        speed_path,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
    )
    if speed_binding != _binding(
        speed_path, payload_sha256=str(selection["integrity"]["payload_sha256"])
    ):
        raise StochasticScheduleSuiteError("speed selection binding drift")
    root = manifest_path.resolve().parent
    runs = _load_runs(
        suite_root=root,
        cohort_path=cohort_path,
        cohort_sha256=cohort_sha256,
        scope=expected_scope,
    )
    smoke_binding: Mapping[str, Any] | None = None
    formal_authorization: Mapping[str, Any] | None = None
    if expected_scope == "formal_merged_panel":
        smoke_binding = manifest.get("official_smoke_suite")
        if not isinstance(smoke_binding, Mapping):
            raise StochasticScheduleSuiteError("formal smoke binding missing")
        smoke = load_and_validate_suite(
            Path(str(smoke_binding.get("path"))),
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            expected_scope="infrastructure_smoke",
            max_num_seqs=FIXED_MAX_NUM_SEQS,
        )
        if smoke_binding != smoke["manifest_binding"]:
            raise StochasticScheduleSuiteError("formal smoke binding drift")
        formal_authorization = _formal_authorization_from_smoke_loaded(smoke)
    rebuilt = seal_manifest(
        _expected_payload(
            created_at_utc=str(manifest.get("created_at_utc")),
            output_dir=root,
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            scope=expected_scope,
            selection_path=speed_path,
            selection=selection,
            runs=runs,
            smoke_binding=smoke_binding,
            formal_authorization=formal_authorization,
        )
    )
    if manifest != rebuilt:
        raise StochasticScheduleSuiteError("suite manifest drift")
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
    seal.add_argument("--max-num-seqs", type=int, required=True)
    seal.add_argument("--speed-selection", type=Path, required=True)
    seal.add_argument("--smoke-suite-manifest", type=Path)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--cohort", type=Path, required=True)
    validate.add_argument("--cohort-sha256", required=True)
    validate.add_argument("--scope", choices=SCOPES, required=True)
    validate.add_argument("--max-num-seqs", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "seal":
            value = seal_suite(
                cohort_path=args.cohort,
                cohort_sha256=args.cohort_sha256,
                output_dir=args.output_dir,
                scope=args.scope,
                max_num_seqs=args.max_num_seqs,
                speed_selection=args.speed_selection,
                smoke_suite_manifest=args.smoke_suite_manifest,
            )
            status = "complete"
        else:
            loaded = load_and_validate_suite(
                args.manifest,
                cohort_path=args.cohort,
                cohort_sha256=args.cohort_sha256,
                expected_scope=args.scope,
                max_num_seqs=args.max_num_seqs,
            )
            value = loaded["manifest"]
            status = "valid"
        print(
            _canonical(
                {
                    "status": status,
                    "scope": value["evaluation_scope"],
                    "rows": value["coverage"]["total_rows"],
                    "selected_max_num_seqs": FIXED_MAX_NUM_SEQS,
                    "payload_sha256": value["integrity"]["payload_sha256"],
                }
            )
        )
        return 0
    except (
        StochasticScheduleSuiteError,
        selector.StochasticScheduleSelectionError,
        amendment.StochasticScheduleAmendmentError,
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
    "FIXED_MAX_NUM_SEQS",
    "GENERATION_GATES",
    "StochasticScheduleSuiteError",
    "load_and_validate_suite",
    "main",
    "seal_suite",
]
