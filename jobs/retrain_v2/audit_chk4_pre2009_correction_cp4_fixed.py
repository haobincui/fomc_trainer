"""Seal the failed checkpoint-4 static screen and authorize final cp6 screening.

This is additive because the checkpoint-2 decision and checkpoint-4 launch
receipts bind the older audit/runner sources byte-for-byte.  It never generates,
trains, merges, or launches GRPO.  The sealed outcome is deliberately *not* a
checkpoint selection: checkpoint-4 failed the full preregistered gate, while
its delivery/hold safety evidence permits evaluating the final checkpoint-6
candidate in the fixed 2 -> 4 -> 6 order.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.retrain_v2 import audit_chk4_pre2009_correction_selection as base_audit
from jobs.retrain_v2 import probe_chk4_pre2009_correction as probe
from jobs.retrain_v2 import run_chk4_pre2009_correction_screen_fixed as fixed_runner
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


SCREEN_ROOT = base_audit.SCREEN_ROOT
STEP = 4
OUTPUT = SCREEN_ROOT / "checkpoint-4-selection-cap1536-v2"
AUTHORIZATION = (
    SCREEN_ROOT
    / "receipts/checkpoint-4-selection-cap1536-v2_authorization.json"
)
DECISION = (
    SCREEN_ROOT
    / "receipts/checkpoint-4-selection-decision-cap1536-v2.json"
)
AUTHORIZATION_SHA256 = (
    "6a50b17ba8f85b7b004cb7c4d801a2a3ecf55e2f3dcf2cc14f2915cf0e837321"
)
AUTHORIZATION_PAYLOAD_SHA256 = (
    "af05d1dfc4338d3915d3c01c4af3e8bb657e26282923027240bfc9229979a056"
)
LAUNCH_SHA256 = "c3e5965b5e46ffeeb2718fb5c4471acb9a6b85a384a390eccf4a8f2d58b921fb"
RESULTS_SHA256 = "7d4d2c2ae8bbc229101dae2ff9ee47e37aa0c73ccc4b512bd22ba3e3b038abac"
SUMMARY_SHA256 = "4031a31a3fd5737adb6c677f03310f8d01a84dc866bb0215d5746b00190391df"
RUNNER_SHA256 = "9aa95639994e209adcd4c091a19eb368a8308248e7d0d57f2d76bc923fdda5ff"
CP4_ADAPTER_WEIGHTS_SHA256 = (
    "02ed6ce7382bd4f80739085e2a799de4bd6041d26198460ecc13623053a69493"
)
CP2_DECISION_SHA256 = fixed_runner.CP2_DECISION_SHA256
CP2_DECISION_PAYLOAD_SHA256 = fixed_runner.CP2_DECISION_PAYLOAD_SHA256
EXPECTED_GATE_REASONS = (
    "step1:hike_correct_nonzero_lt_1",
    "step1:hold_reward_zero_std",
    "step2:cut_correct_nonzero_lt_1",
    "step2:cut_reward_zero_std",
)
SCHEMA = "chk4-pre2009-correction-cp4-fixed-screen-decision-v1"


class Cp4DecisionError(RuntimeError):
    """The checkpoint-4 result cannot safely advance to checkpoint-6."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Cp4DecisionError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Cp4DecisionError(f"invalid {label}: {path}") from exc
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


def _validate_padding_row(row: Mapping[str, Any], eos_token_ids: set[int]) -> None:
    token_ids = row.get("generated_token_ids")
    evidence = row.get("batched_padding_normalization")
    _require(isinstance(token_ids, list) and bool(token_ids), "trimmed token IDs missing")
    _require(isinstance(evidence, Mapping), "padding-normalization evidence missing")
    _require(
        evidence.get("schema_version") == "batched-first-eos-normalization-v1"
        and evidence.get("method") == "truncate_at_first_eos_inclusive"
        and evidence.get("generation_not_changed") is True,
        "padding-normalization contract drift",
    )
    trimmed_count = evidence.get("trimmed_token_count")
    original_count = evidence.get("original_token_count")
    discarded_count = evidence.get("discarded_after_first_eos_count")
    first_eos = evidence.get("first_eos_index")
    _require(
        isinstance(trimmed_count, int)
        and isinstance(original_count, int)
        and isinstance(discarded_count, int)
        and trimmed_count == len(token_ids)
        and original_count == trimmed_count + discarded_count,
        "padding-normalization counts drift",
    )
    if first_eos is None:
        _require(
            discarded_count == 0
            and not any(token in eos_token_ids for token in token_ids),
            "no-EOS normalization drift",
        )
    else:
        _require(
            isinstance(first_eos, int)
            and first_eos == len(token_ids) - 1
            and token_ids[first_eos] in eos_token_ids
            and not any(token in eos_token_ids for token in token_ids[:first_eos]),
            "first-EOS inclusive truncation drift",
        )
    discarded_unique = evidence.get("discarded_unique_token_ids")
    _require(isinstance(discarded_unique, list), "discarded-token audit missing")
    if discarded_count:
        _require(
            evidence.get("all_discarded_tokens_are_bound_eos") is True
            and set(discarded_unique).issubset(eos_token_ids),
            "discarded tokens were not exclusively bound EOS padding",
        )
    else:
        _require(discarded_unique == [], "zero-discard token audit drift")


def audit(repo_root: Path) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    receipts = base_audit._receipts(root)
    manifest_path, manifest = base_audit._manifest(root)
    candidate = base_audit._candidate_binding(root, STEP)

    cp2_path = root / fixed_runner.CP2_DECISION
    cp2, cp2_binding = _sealed_binding(
        cp2_path,
        file_sha256=CP2_DECISION_SHA256,
        payload_sha256=CP2_DECISION_PAYLOAD_SHA256,
        label="checkpoint-2 decision",
    )
    _require(
        cp2.get("outcome") == "advance_allowed_action_direction_insufficiency"
        and cp2.get("advance_allowed") is True,
        "checkpoint-2 did not permit checkpoint-4",
    )
    authorization_path = root / AUTHORIZATION
    authorization, authorization_binding = _sealed_binding(
        authorization_path,
        file_sha256=AUTHORIZATION_SHA256,
        payload_sha256=AUTHORIZATION_PAYLOAD_SHA256,
        label="checkpoint-4 authorization",
    )
    _require(
        authorization.get("scope", {}).get("operation")
        == "checkpoint_4_static_generation_screening"
        and authorization.get("candidate_checkpoint") == candidate,
        "checkpoint-4 authorization scope/candidate drift",
    )
    _require(
        authorization.get("lineage", {})
        .get("checkpoint_2_advance_decision", {})
        .get("sha256")
        == CP2_DECISION_SHA256,
        "checkpoint-4 authorization lost checkpoint-2 lineage",
    )

    output = (root / OUTPUT).resolve()
    _require(output.is_dir() and not output.is_symlink(), "checkpoint-4 output missing")
    inventory = sorted(
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    _require(
        inventory == ["launch.json", "results.jsonl", "summary.json"],
        f"checkpoint-4 output inventory drift: {inventory}",
    )
    launch_path = output / "launch.json"
    results_path = output / "results.jsonl"
    summary_path = output / "summary.json"
    launch = _read_json(launch_path, label="checkpoint-4 launch")
    results = base_audit._read_jsonl(results_path, label="checkpoint-4 results")
    summary = _read_json(summary_path, label="checkpoint-4 summary")
    _descriptor(launch_path, expected_sha256=LAUNCH_SHA256)
    _descriptor(results_path, expected_sha256=RESULTS_SHA256)
    _descriptor(summary_path, expected_sha256=SUMMARY_SHA256)
    _require(len(results) == 16, "checkpoint-4 screen must contain 16 rows")
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
        _validate_padding_row(row, eos_token_ids)
        base_audit._replay_row(
            row, manifest=manifest, eos_token_ids=eos_token_ids
        )
    recomputed = probe.summarize_results(
        results, manifest=manifest, provenance=provenance
    )
    for key, expected in recomputed.items():
        _require(summary.get(key) == expected, f"summary replay drift: {key}")
    _require(
        summary.get("results", {}).get("sha256") == RESULTS_SHA256
        and summary.get("results", {}).get("rows") == 16
        and summary.get("results", {}).get(
            "raw_completions_and_token_ids_landed"
        )
        is True,
        "summary result binding drift",
    )
    postprocessor = (
        provenance.get("model_source", {}).get("batched_padding_postprocessor", {})
    )
    _require(
        postprocessor.get("implementation", {}).get("sha256") == RUNNER_SHA256
        and postprocessor.get("method")
        == "truncate_at_first_bound_eos_inclusive_before_decode",
        "fixed padding implementation provenance drift",
    )
    _require(
        launch.get("status") == "initializing"
        and launch.get("model_label") == "correction-sft-checkpoint-4",
        "checkpoint-4 launch contract drift",
    )
    reasons = summary.get("quality_gate", {}).get("reasons")
    _require(
        summary.get("quality_status") == "failed"
        and tuple(reasons or ()) == EXPECTED_GATE_REASONS,
        "checkpoint-4 failure contract drift",
    )
    step1 = summary.get("panels", {}).get("step1", {}).get("directions", {})
    step2 = (
        summary.get("panels", {})
        .get("static_step2_prior", {})
        .get("directions", {})
    )
    deliveries = [
        summary.get("panels", {}).get(panel, {}).get("delivery", {})
        for panel in ("step1", "static_step2_prior")
    ]
    _require(
        step1.get("hold", {}).get("correct_nonzero_count", 0) >= 2
        and step2.get("hold", {}).get("correct_nonzero_count", 0) >= 1
        and all(
            delivery.get("cap_count", 99) <= 2
            and delivery.get("boundary_count", 0) >= 6
            and delivery.get("periodic_tail_count", 99) == 0
            for delivery in deliveries
        ),
        "checkpoint-4 hold/delivery safety regression",
    )

    implementation_relatives = (
        "jobs/retrain_v2/audit_chk4_pre2009_correction_cp4_fixed.py",
        "jobs/retrain_v2/audit_chk4_pre2009_correction_selection.py",
        "jobs/retrain_v2/run_chk4_pre2009_correction_screen_fixed.py",
        "jobs/retrain_v2/probe_chk4_pre2009_correction.py",
        "jobs/retrain_v2/probe_chk4_decision_grpo_stratified.py",
        "jobs/retrain_v2/probe_chk4_decision_sft_generation.py",
        "src/open_r1/trainer/rewards/reward_funcs/decision_reward_v3.py",
        "src/open_r1/structured_response.py",
    )
    implementation = {
        relative: _descriptor(root / relative) for relative in implementation_relatives
    }
    _require(
        implementation[
            "jobs/retrain_v2/run_chk4_pre2009_correction_screen_fixed.py"
        ]["sha256"]
        == RUNNER_SHA256,
        "checkpoint-4 runner source drift",
    )
    return {
        "schema_version": SCHEMA,
        "status": "sealed_decision",
        "checkpoint_step": STEP,
        "outcome": "failed_not_selected_advance_to_final_cp6",
        "selected": False,
        "advance_allowed": True,
        "next_and_final_candidate": 6,
        "full_gate_unchanged": True,
        "failure_is_not_a_pass": True,
        "methodology": {
            "name": "static_generation_screening",
            "not_a_bitwise_grpo_replay": True,
            "selection_rule": "earliest_all_pass_in_preregistered_order_2_4_6",
            "checkpoint_4_failed_full_gate": True,
            "checkpoint_6_must_all_pass_or_selection_fails": True,
            "no_merge_blind_confirmation_or_grpo_before_all_pass": True,
        },
        "receipts": receipts,
        "prior_checkpoint_2_decision": cp2_binding,
        "checkpoint_4_authorization": authorization_binding,
        "manifest": {**_descriptor(manifest_path), "sha256": base_audit.MANIFEST_SHA256},
        "candidate_checkpoint": {
            **candidate,
            "adapter_model": _descriptor(
                Path(str(candidate["path"])) / "adapter_model.safetensors",
                expected_sha256=CP4_ADAPTER_WEIGHTS_SHA256,
            ),
        },
        "screen_artifacts": {
            "launch": _descriptor(launch_path, expected_sha256=LAUNCH_SHA256),
            "results": {
                **_descriptor(results_path, expected_sha256=RESULTS_SHA256),
                "rows": 16,
            },
            "summary": _descriptor(summary_path, expected_sha256=SUMMARY_SHA256),
            "inventory": inventory,
        },
        "quality_gate": summary["quality_gate"],
        "panels": summary["panels"],
        "implementation": implementation,
    }


def seal(repo_root: Path, *, execute: bool) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    value = seal_manifest(audit(root))
    path = (root / DECISION).resolve()
    if not execute:
        return {**value, "status": "ready_to_seal", "path": str(path)}
    _require(not path.exists() and not path.is_symlink(), f"decision exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "status": value["outcome"],
        "path": str(path),
        "sha256": sha256_file(path),
        "payload_sha256": value["integrity"]["payload_sha256"],
        "advance_allowed": True,
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
    except (Cp4DecisionError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
