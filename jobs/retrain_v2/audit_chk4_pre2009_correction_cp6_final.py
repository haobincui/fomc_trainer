"""Replay checkpoint-6 and seal the final failed correction selection.

This CPU-only auditor is the terminal state for the preregistered 2 -> 4 -> 6
static screening sequence when checkpoint-6 does not pass every unchanged
gate.  It never selects or merges a model and explicitly blocks blind
confirmation and GRPO.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.retrain_v2 import audit_chk4_pre2009_correction_cp4_fixed as cp4_audit
from jobs.retrain_v2 import audit_chk4_pre2009_correction_selection as base_audit
from jobs.retrain_v2 import probe_chk4_pre2009_correction as probe
from jobs.retrain_v2 import run_chk4_pre2009_correction_cp6_fixed as cp6_runner
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


SCREEN_ROOT = base_audit.SCREEN_ROOT
OUTPUT = cp6_runner.OUTPUT
AUTHORIZATION = cp6_runner.AUTHORIZATION
FINAL_RECEIPT = (
    SCREEN_ROOT
    / "receipts/checkpoint-selection-final-failure-cap1536-v1.json"
)
AUTHORIZATION_SHA256 = (
    "011822f7d17a8755941fb7010b53daf8c53beae3ef5d57ffe1188a7d0d28a36a"
)
AUTHORIZATION_PAYLOAD_SHA256 = (
    "aa9dc95f6f1e3748f03a56136f030a62fd27cc1d9e05b035f20423b869a1019d"
)
LAUNCH_SHA256 = "803f9a1acc1ed5e175e19317b16f3e531668b5776049d0ea25a9309a5b83a318"
RESULTS_SHA256 = "49764e0e60bcf0e55f0ccb0d7b7242421e53d699ae9699e3986c52fd4ae13dbd"
SUMMARY_SHA256 = "ecc1f58ff6d988fe6d36b97c2edf6150dcff64e5b89baac03d8fafb398171604"
RUNNER_SHA256 = "d97c6bef9daa0fef46376c6d49c48f6f513d5932f37881b9872c21f6eb2f3633"
BASE_MODEL_SHA256 = "715bdb763043e7fda49e4a189b601d5e070fefbcecd14388bde41dbf7c165dd0"
EFFECTIVE_MODEL_SHA256 = (
    "76d34b7180c6c5dd188c9efe9524a1c41de09f5e6680bae690059d8904a96eeb"
)
EXPECTED_GATE_REASONS = (
    "step1:hike_correct_nonzero_lt_1",
    "step1:hike_reward_zero_std",
)
SCHEMA = "chk4-pre2009-correction-final-selection-failure-v1"


class FinalSelectionError(RuntimeError):
    """The final failed selection cannot be sealed from current evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalSelectionError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalSelectionError(f"invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _descriptor(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing artifact: {path}")
    observed = sha256_file(path)
    if expected_sha256 is not None:
        _require(observed == expected_sha256, f"artifact hash drift: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": observed,
        "bytes": path.stat().st_size,
    }


def _sealed_binding(
    path: Path, *, file_sha256: str, payload_sha256: str, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _read_json(path, label=label)
    validate_manifest_integrity(value)
    _require(
        sha256_file(path) == file_sha256
        and value.get("integrity", {}).get("payload_sha256") == payload_sha256,
        f"{label} binding drift",
    )
    return value, {
        **_descriptor(path, expected_sha256=file_sha256),
        "payload_sha256": payload_sha256,
    }


def audit(repo_root: Path) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    receipts = base_audit._receipts(root)
    manifest_path, manifest = base_audit._manifest(root)
    candidate = base_audit._candidate_binding(root, 6)

    cp2, cp2_binding = _sealed_binding(
        root / cp6_runner.cp4_runner.CP2_DECISION,
        file_sha256=cp6_runner.cp4_runner.CP2_DECISION_SHA256,
        payload_sha256=cp6_runner.cp4_runner.CP2_DECISION_PAYLOAD_SHA256,
        label="checkpoint-2 decision",
    )
    cp4, cp4_binding = _sealed_binding(
        root / cp4_audit.DECISION,
        file_sha256=cp6_runner.CP4_DECISION_SHA256,
        payload_sha256=cp6_runner.CP4_DECISION_PAYLOAD_SHA256,
        label="checkpoint-4 decision",
    )
    _require(
        cp2.get("advance_allowed") is True
        and cp4.get("outcome") == "failed_not_selected_advance_to_final_cp6",
        "prior decision sequence drift",
    )
    authorization, authorization_binding = _sealed_binding(
        root / AUTHORIZATION,
        file_sha256=AUTHORIZATION_SHA256,
        payload_sha256=AUTHORIZATION_PAYLOAD_SHA256,
        label="checkpoint-6 authorization",
    )
    _require(
        authorization.get("candidate_checkpoint", {}).get("sha256")
        == cp6_runner.CP6_CHECKPOINT_SHA256
        and authorization.get("candidate_checkpoint", {}).get("adapter_model", {}).get(
            "sha256"
        )
        == cp6_runner.CP6_ADAPTER_WEIGHTS_SHA256,
        "checkpoint-6 authorization candidate drift",
    )
    _require(
        authorization.get("generation")
        == {
            "seed": 31416,
            "seed_once_before_all_batches": True,
            "batches": [
                [
                    "dec-bb7d9c61358a339a9a1f4aa5",
                    "dec-4dfab939b0a910c949931475",
                ],
                [
                    "dec-29de02cb43945c20f838fcf7",
                    "dec-8b8d55ea065b662a19cefe88",
                ],
            ],
            "num_return_sequences": 4,
            "max_new_tokens": 1536,
            "temperature": 0.7,
            "top_p": 0.9,
            "padding_postprocessor": (
                "truncate_at_first_bound_eos_inclusive_before_decode"
            ),
        },
        "checkpoint-6 generation contract drift",
    )

    output = (root / OUTPUT).resolve()
    _require(output.is_dir() and not output.is_symlink(), "checkpoint-6 output missing")
    inventory = sorted(
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    _require(
        inventory == ["launch.json", "results.jsonl", "summary.json"],
        f"checkpoint-6 output inventory drift: {inventory}",
    )
    launch_path = output / "launch.json"
    results_path = output / "results.jsonl"
    summary_path = output / "summary.json"
    launch = _read_json(launch_path, label="checkpoint-6 launch")
    results = base_audit._read_jsonl(results_path, label="checkpoint-6 results")
    summary = _read_json(summary_path, label="checkpoint-6 summary")
    _descriptor(launch_path, expected_sha256=LAUNCH_SHA256)
    _descriptor(results_path, expected_sha256=RESULTS_SHA256)
    _descriptor(summary_path, expected_sha256=SUMMARY_SHA256)
    _require(len(results) == 16, "checkpoint-6 screen must contain 16 rows")
    observed_order = [
        (
            int(row.get("batch_index", -1)),
            str(row.get("sample_id")),
            int(row.get("generation_index", -1)),
        )
        for row in results
    ]
    _require(observed_order == base_audit._expected_order(), "result order drift")
    provenance = summary.get("provenance")
    _require(isinstance(provenance, Mapping), "summary provenance missing")
    eos_values = provenance.get("eos_token_ids")
    _require(
        isinstance(eos_values, list)
        and eos_values
        and all(isinstance(value, int) for value in eos_values),
        "EOS binding missing",
    )
    eos_token_ids = set(eos_values)
    for row in results:
        _require(
            row.get("purpose") == probe.SELECTION_PURPOSE
            and row.get("panel") == "selection"
            and row.get("generation_mode") == "sampled"
            and row.get("seed") == probe.SELECTION_SEED,
            "selection generation contract drift",
        )
        cp4_audit._validate_padding_row(row, eos_token_ids)
        base_audit._replay_row(
            row, manifest=manifest, eos_token_ids=eos_token_ids
        )
    recomputed = probe.summarize_results(
        results, manifest=manifest, provenance=provenance
    )
    for key, expected in recomputed.items():
        _require(summary.get(key) == expected, f"summary replay drift: {key}")
    reasons = summary.get("quality_gate", {}).get("reasons")
    _require(
        summary.get("quality_status") == "failed"
        and tuple(reasons or ()) == EXPECTED_GATE_REASONS,
        "checkpoint-6 final failure contract drift",
    )
    postprocessor = (
        provenance.get("model_source", {}).get("batched_padding_postprocessor", {})
    )
    _require(
        postprocessor.get("implementation", {}).get("sha256") == RUNNER_SHA256
        and postprocessor.get("predecessor_implementation", {}).get("sha256")
        == cp4_audit.RUNNER_SHA256,
        "checkpoint-6 fixed-padding implementation provenance drift",
    )
    _require(
        provenance.get("model_source", {})
        .get("adapter", {})
        .get("directory")
        == candidate,
        "checkpoint-6 candidate composition drift",
    )
    model_source = provenance.get("model_source", {})
    _require(
        provenance.get("model_sha256") == BASE_MODEL_SHA256
        and provenance.get("effective_model_sha256") == EFFECTIVE_MODEL_SHA256
        and model_source.get("base_model", {})
        .get("directory", {})
        .get("sha256")
        == BASE_MODEL_SHA256
        and model_source.get("adapter", {})
        .get("files", {})
        .get("adapter_model.safetensors", {})
        .get("sha256")
        == cp6_runner.CP6_ADAPTER_WEIGHTS_SHA256,
        "checkpoint-6 base/effective/adapter model binding drift",
    )
    _require(
        summary.get("results", {}).get("sha256") == RESULTS_SHA256
        and summary.get("results", {}).get("rows") == 16
        and summary.get("results", {}).get(
            "raw_completions_and_token_ids_landed"
        )
        is True,
        "summary result binding drift",
    )
    _require(
        launch.get("status") == "initializing"
        and launch.get("model_label") == "correction-sft-checkpoint-6",
        "checkpoint-6 launch contract drift",
    )
    for path in (launch_path, results_path, summary_path):
        _require(path.stat().st_mode & 0o777 == 0o444, f"unsealed artifact: {path}")
    _require(output.stat().st_mode & 0o777 == 0o555, "unsealed cp6 output root")

    run_root = (root / base_audit.RUN_ROOT).resolve()
    promotion_entries = sorted(
        str(path.relative_to(run_root))
        for path in run_root.rglob("*")
        if any(
            marker in str(path.relative_to(run_root)).lower()
            for marker in ("merged", "blind", "grpo")
        )
        and not str(path.relative_to(run_root)).startswith(
            "static_screening/manifests/"
        )
    )
    _require(
        promotion_entries == [],
        f"unauthorized promotion output exists: {promotion_entries}",
    )

    implementation_relatives = (
        "jobs/retrain_v2/audit_chk4_pre2009_correction_cp6_final.py",
        "jobs/retrain_v2/run_chk4_pre2009_correction_cp6_fixed.py",
        "jobs/retrain_v2/audit_chk4_pre2009_correction_cp4_fixed.py",
        "jobs/retrain_v2/run_chk4_pre2009_correction_screen_fixed.py",
        "jobs/retrain_v2/audit_chk4_pre2009_correction_selection.py",
        "jobs/retrain_v2/probe_chk4_pre2009_correction.py",
        "jobs/retrain_v2/replay_chk4_pre2009_correction_padding_v3.py",
        "jobs/retrain_v2/probe_chk4_decision_grpo_stratified.py",
        "jobs/retrain_v2/probe_chk4_decision_sft_generation.py",
        "src/open_r1/trainer/rewards/reward_funcs/decision_reward_v3.py",
        "src/open_r1/structured_response.py",
    )
    implementation = {
        relative: _descriptor(root / relative) for relative in implementation_relatives
    }
    _require(
        implementation["jobs/retrain_v2/run_chk4_pre2009_correction_cp6_fixed.py"]
        ["sha256"]
        == RUNNER_SHA256,
        "checkpoint-6 runner source drift",
    )
    authorized_implementation = authorization.get("implementation")
    _require(
        isinstance(authorized_implementation, Mapping)
        and all(
            implementation.get(relative) == descriptor
            for relative, descriptor in authorized_implementation.items()
        ),
        "checkpoint-6 authorized implementation inventory drift",
    )
    return {
        "schema_version": SCHEMA,
        "status": "selection_failed",
        "outcome": "no_preregistered_checkpoint_passed_full_gate",
        "selected_checkpoint": None,
        "candidate_order_completed": [2, 4, 6],
        "candidate_outcomes": {
            "2": {
                "selected": False,
                "outcome": cp2["outcome"],
            },
            "4": {
                "selected": False,
                "outcome": cp4["outcome"],
            },
            "6": {
                "selected": False,
                "outcome": "failed_full_gate_final_candidate",
                "gate_reasons": list(EXPECTED_GATE_REASONS),
            },
        },
        "advance_allowed": False,
        "terminal": True,
        "prohibited": [
            "checkpoint_selection",
            "adapter_merge",
            "blind_confirmation",
            "grpo_smoke",
            "grpo_full",
            "additional_candidate_or_seed_without_new_authorization",
        ],
        "methodology": {
            "name": "static_generation_screening",
            "not_a_bitwise_grpo_replay": True,
            "selection_rule": "earliest_all_pass_in_preregistered_order_2_4_6",
            "full_gate_unchanged": True,
            "checkpoint_6_failed_full_gate": True,
            "failure_reason": "step1_hike_signal_absent",
            "no_candidate_after_checkpoint_6": True,
        },
        "receipts": receipts,
        "checkpoint_2_decision": cp2_binding,
        "checkpoint_4_decision": cp4_binding,
        "checkpoint_6_authorization": authorization_binding,
        "manifest": {**_descriptor(manifest_path), "sha256": base_audit.MANIFEST_SHA256},
        "candidate_checkpoint": candidate,
        "checkpoint_6_screen_artifacts": {
            "launch": _descriptor(launch_path, expected_sha256=LAUNCH_SHA256),
            "results": {
                **_descriptor(results_path, expected_sha256=RESULTS_SHA256),
                "rows": 16,
            },
            "summary": _descriptor(summary_path, expected_sha256=SUMMARY_SHA256),
            "inventory": inventory,
        },
        "promotion_output_absence_at_seal_time": {
            "run_root": str(run_root),
            "ignored_preregistered_manifest": str(
                root
                / base_audit.SCREEN_ROOT
                / "manifests/locked_checkpoint_blind_confirmation_cap1536_v1.json"
            ),
            "matching_non_manifest_entries": promotion_entries,
            "merged_absent": True,
            "blind_results_absent": True,
            "grpo_absent": True,
        },
        "quality_gate": summary["quality_gate"],
        "panels": summary["panels"],
        "implementation": implementation,
    }


def seal(repo_root: Path, *, execute: bool) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    value = seal_manifest(audit(root))
    path = (root / FINAL_RECEIPT).resolve()
    if not execute:
        return {**value, "status": "ready_to_seal", "path": str(path)}
    _require(not path.exists() and not path.is_symlink(), f"receipt exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "status": value["status"],
        "outcome": value["outcome"],
        "path": str(path),
        "sha256": sha256_file(path),
        "payload_sha256": value["integrity"]["payload_sha256"],
        "advance_allowed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = seal(args.repo_root, execute=bool(args.execute))
    except (FinalSelectionError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
