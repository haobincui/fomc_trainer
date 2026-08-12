"""Replay and seal sequential chk4 correction checkpoint-screen decisions.

This module never generates, trains, merges, or starts GRPO.  It validates the
raw static-screen completions and deterministic reward replay, then permits the
pre-registered order 2 -> 4 -> 6.  A later checkpoint is eligible only when
every earlier checkpoint was safe and failed solely for insufficient step-1
hike signal.  The first passing checkpoint is locked; any safety regression
invalidates the correction run.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.retrain_v2 import attest_chk4_pre2009_correction_sft_gap as gap
from jobs.retrain_v2 import probe_chk4_decision_grpo_stratified as reward_probe
from jobs.retrain_v2 import probe_chk4_decision_sft_generation as completion_probe
from jobs.retrain_v2 import probe_chk4_pre2009_correction as probe
from jobs.retrain_v2 import replay_chk4_pre2009_correction_padding_v3 as padding_replay
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


RUN_ROOT = gap.RUN_ROOT
TRAINING_OUTPUT = gap.TRAINING_OUTPUT
SCREEN_ROOT = RUN_ROOT / "static_screening"
MANIFEST = SCREEN_ROOT / "manifests/checkpoint_selection_cap1536_v1.json"
MANIFEST_SHA256 = "e234c3bdb3021b231454c46e05d6350839d9013a40e2d9dc4afca4130c0bb9c8"
ORIGINAL_AUTH_SHA256 = gap.AUTHORIZATION_SHA256
ORIGINAL_AUTH_PAYLOAD_SHA256 = gap.AUTHORIZATION_PAYLOAD_SHA256
SUPPLEMENTAL_SHA256 = (
    "2770fac9ffc58bb1d8a68f018ee3802fd9ad94b9d1a7a51ea45a58f4b2878cbe"
)
SUPPLEMENTAL_PAYLOAD_SHA256 = (
    "3a344de0237c38be9c79dd875ef88e6e9554b318b0d54cfa901fec033518cd72"
)
CANDIDATE_STEPS = (2, 4, 6)
OUTPUT_VERSION = {2: 3, 4: 1, 6: 1}
SCHEMA = "chk4-pre2009-correction-static-selection-decision-v1"
ACTION_ONLY_REASONS = frozenset(
    {
        "step1:hike_correct_nonzero_lt_1",
        "step1:hike_reward_zero_std",
        "step2:cut_correct_nonzero_lt_1",
        "step2:cut_reward_zero_std",
    }
)


class SelectionAuditError(RuntimeError):
    """A static checkpoint screen or its sequence failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectionAuditError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectionAuditError(f"invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _require(bool(raw), f"{label}:{line_number}: blank row")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SelectionAuditError(f"{label}:{line_number}: invalid JSON") from exc
        _require(isinstance(value, dict), f"{label}:{line_number}: not an object")
        rows.append(value)
    return rows


def _descriptor(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing artifact: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _receipts(repo_root: Path) -> dict[str, Any]:
    authorization_path = repo_root / gap.AUTHORIZATION
    authorization = _read_json(authorization_path, label="original authorization")
    validate_manifest_integrity(authorization)
    _require(
        sha256_file(authorization_path) == ORIGINAL_AUTH_SHA256
        and authorization.get("integrity", {}).get("payload_sha256")
        == ORIGINAL_AUTH_PAYLOAD_SHA256,
        "original SFT authorization drift",
    )
    supplemental_result = gap.verify(repo_root)
    _require(
        supplemental_result["sha256"] == SUPPLEMENTAL_SHA256
        and supplemental_result["payload_sha256"] == SUPPLEMENTAL_PAYLOAD_SHA256,
        "supplemental provenance-gap attestation drift",
    )
    return {
        "original_pre_run_authorization": {
            **_descriptor(authorization_path),
            "payload_sha256": ORIGINAL_AUTH_PAYLOAD_SHA256,
        },
        "supplemental_post_run_attestation": {
            **_descriptor(repo_root / gap.ATTESTATION),
            "payload_sha256": SUPPLEMENTAL_PAYLOAD_SHA256,
            "status": gap.STATUS,
            "is_not_pre_authorization": True,
        },
    }


def _screen_output(repo_root: Path, step: int) -> Path:
    _require(step in CANDIDATE_STEPS, "checkpoint step must be 2, 4, or 6")
    version = OUTPUT_VERSION[step]
    return (
        repo_root
        / SCREEN_ROOT
        / f"checkpoint-{step}-selection-cap1536-v{version}"
    ).resolve()


def _candidate_binding(repo_root: Path, step: int) -> dict[str, Any]:
    supplemental = _read_json(
        repo_root / gap.ATTESTATION, label="supplemental attestation"
    )
    expected = supplemental.get("completed_run", {}).get(
        "candidate_checkpoint_fingerprints", {}
    ).get(str(step))
    _require(isinstance(expected, Mapping), f"checkpoint-{step} gap binding missing")
    candidate = (repo_root / TRAINING_OUTPUT / f"checkpoint-{step}").resolve()
    observed = fingerprint_artifact_path(candidate)
    _require(observed == expected, f"checkpoint-{step} fingerprint drift")
    return dict(observed)


def _manifest(repo_root: Path) -> tuple[Path, dict[str, Any]]:
    path = (repo_root / MANIFEST).resolve()
    _require(sha256_file(path) == MANIFEST_SHA256, "selection manifest hash drift")
    try:
        manifest = probe._validate_manifest(path, MANIFEST_SHA256)
    except Exception as exc:
        raise SelectionAuditError(f"selection manifest replay failed: {exc}") from exc
    return path, manifest


def _expected_order() -> list[tuple[int, str, int]]:
    expected: list[tuple[int, str, int]] = []
    for batch_index, batch in enumerate(probe.SELECTION_BATCH_IDS):
        for sample_id in batch:
            for generation_index in range(1, probe.NUM_RETURN_SEQUENCES + 1):
                expected.append((batch_index, sample_id, generation_index))
    return expected


def _replay_row(
    row: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    eos_token_ids: set[int],
) -> None:
    sample_id = str(row.get("sample_id"))
    sample = manifest["samples"].get(sample_id)
    _require(isinstance(sample, Mapping), f"unexpected sample: {sample_id}")
    completion = row.get("completion")
    token_ids = row.get("generated_token_ids")
    _require(isinstance(completion, str), "raw completion is missing")
    _require(
        isinstance(token_ids, list)
        and token_ids
        and all(isinstance(value, int) and not isinstance(value, bool) for value in token_ids),
        "raw generated token IDs are missing or invalid",
    )
    _require(
        row.get("completion_sha256") == probe._sha256_text(completion),
        "completion SHA drift",
    )
    metrics = completion_probe.analyze_completion(
        text=completion,
        generated_token_ids=token_ids,
        eos_token_ids=eos_token_ids,
        max_new_tokens=probe.MAX_NEW_TOKENS,
        tail_tokens=probe.TAIL_TOKENS,
    )
    for key, expected in metrics.items():
        _require(row.get(key) == expected, f"completion metric drift: {key}")
    replay = reward_probe.replay_decision_dense_v3(
        completion,
        {
            "direction": sample["direction"],
            "magnitude_bp": sample["magnitude_bp"],
        },
        hit_eos=bool(metrics["hit_eos"]),
        cap_reached=bool(metrics["cap_reached"]),
    )
    replay_fields = {
        "decision_dense_v3_reward": "reward",
        "decision_dense_v3_nonzero": "nonzero",
        "decision_prediction": "prediction",
        "decision_direction_correct": "direction_correct",
        "decision_exact": "exact",
        "decision_rejection_reason": "rejection_reason",
        "decision_forced_zero_reason": "forced_zero_reason",
        "response_format": "response_format",
        "strict_json": "strict_json",
        "fenced_json": "fenced_json",
    }
    for result_key, replay_key in replay_fields.items():
        _require(
            row.get(result_key) == replay[replay_key],
            f"reward replay drift: {result_key}",
        )


def audit_step(repo_root: Path, step: int) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    receipts = _receipts(root)
    manifest_path, manifest = _manifest(root)
    candidate = _candidate_binding(root, step)
    output = _screen_output(root, step)
    _require(output.is_dir() and not output.is_symlink(), "screen output missing")
    inventory = sorted(
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    _require(
        inventory == ["launch.json", "results.jsonl", "summary.json"],
        f"completed screen inventory drift: {inventory}",
    )
    launch_path = output / "launch.json"
    results_path = output / "results.jsonl"
    summary_path = output / "summary.json"
    launch = _read_json(launch_path, label="screen launch")
    results = _read_jsonl(results_path, label="screen results")
    summary = _read_json(summary_path, label="screen summary")
    _require(len(results) == 16, "selection screen must land 16 result rows")
    expected_order = _expected_order()
    observed_order = [
        (
            int(row.get("batch_index", -1)),
            str(row.get("sample_id")),
            int(row.get("generation_index", -1)),
        )
        for row in results
    ]
    _require(observed_order == expected_order, "selection result order drift")
    provenance = summary.get("provenance")
    _require(isinstance(provenance, Mapping), "summary provenance missing")
    eos_ids = provenance.get("eos_token_ids")
    _require(
        isinstance(eos_ids, list)
        and eos_ids
        and all(isinstance(value, int) for value in eos_ids),
        "summary EOS binding missing",
    )
    for row in results:
        _require(
            row.get("purpose") == probe.SELECTION_PURPOSE
            and row.get("panel") == "selection"
            and row.get("generation_mode") == "sampled"
            and row.get("seed") == probe.SELECTION_SEED
            and row.get("generation_parameters")
            == {
                "max_new_tokens": 1536,
                "temperature": 0.7,
                "top_p": 0.9,
            },
            "selection generation contract drift",
        )
        _replay_row(row, manifest=manifest, eos_token_ids=set(eos_ids))
    recomputed = probe.summarize_results(
        results, manifest=manifest, provenance=provenance
    )
    for key, expected in recomputed.items():
        if key == "schema_version" and step == 2:
            _require(
                summary.get(key) == padding_replay.REPLAY_SUMMARY_SCHEMA,
                "offline replay summary schema drift",
            )
            continue
        _require(summary.get(key) == expected, f"summary replay drift: {key}")
    result_record = summary.get("results")
    _require(
        isinstance(result_record, Mapping)
        and result_record.get("sha256") == sha256_file(results_path)
        and result_record.get("rows") == 16
        and result_record.get("raw_completions_and_token_ids_landed") is True,
        "summary raw result binding drift",
    )
    model_source = provenance.get("model_source")
    adapter = model_source.get("adapter") if isinstance(model_source, Mapping) else None
    _require(
        provenance.get("model_sha256") == probe.SELECTED_CP38_MODEL_SHA256
        and isinstance(adapter, Mapping)
        and adapter.get("directory") == candidate,
        "screen candidate composition drift",
    )
    _require(
        launch.get("status") == "initializing"
        and launch.get("model_label") == f"correction-sft-checkpoint-{step}"
        and launch.get("purpose") == probe.SELECTION_PURPOSE,
        "screen launch contract drift",
    )
    instrumentation_replay = None
    if step == 2:
        diagnosis = padding_replay.verify_diagnosis(root)
        _require(
            summary.get("generation_not_rerun") is True
            and summary.get("instrumentation_fix")
            == "truncate_at_first_eos_padding"
            and summary.get("source_artifacts", {})
            .get("results", {})
            .get("sha256")
            == padding_replay.SOURCE_RESULTS_SHA256
            and summary.get("source_artifacts", {})
            .get("summary", {})
            .get("sha256")
            == padding_replay.SOURCE_SUMMARY_SHA256,
            "checkpoint-2 offline padding replay binding drift",
        )
        instrumentation_replay = {
            "generation_not_rerun": True,
            "instrumentation_fix": "truncate_at_first_eos_padding",
            "invalid_v2_diagnosis": {
                **_descriptor(root / padding_replay.DIAGNOSIS),
                "payload_sha256": diagnosis["integrity"]["payload_sha256"],
                "status": diagnosis["status"],
            },
            "invalid_v2_source_artifacts": summary["source_artifacts"],
        }
    reasons = summary.get("quality_gate", {}).get("reasons")
    _require(
        isinstance(reasons, list) and all(isinstance(reason, str) for reason in reasons),
        "screen gate reasons missing",
    )
    reason_set = frozenset(reasons)
    quality = summary.get("quality_status")
    _require(
        (quality == "passed" and not reasons)
        or (quality == "failed" and bool(reasons)),
        "screen quality status/reasons drift",
    )
    if quality == "passed":
        outcome = "selected_earliest_passing_checkpoint"
        advance_allowed = False
    elif reason_set and reason_set.issubset(ACTION_ONLY_REASONS):
        outcome = "advance_allowed_action_direction_insufficiency"
        advance_allowed = step != CANDIDATE_STEPS[-1]
    else:
        outcome = "correction_run_invalid_safety_or_direction_regression"
        advance_allowed = False
    return {
        "schema_version": "chk4-pre2009-correction-static-selection-audit-v1",
        "status": "audited",
        "checkpoint_step": step,
        "outcome": outcome,
        "advance_allowed": advance_allowed,
        "quality_status": quality,
        "gate_reasons": reasons,
        "methodology": {
            "name": "static_generation_screening",
            "not_a_bitwise_grpo_replay": True,
            "authoritative_gate": (
                "fresh_two_step_cap1536_grpo_smoke_after_selected_exact_merge"
            ),
            "advance_rule": (
                "a failed checkpoint may advance only when hold and all delivery "
                "gates pass and every failure is an action-direction signal gate; "
                "the full original gate remains required for checkpoint selection"
            ),
            "advance_rule_confirmation": (
                "explicitly confirmed after deterministic EOS-padding replay; "
                "does not convert checkpoint-2 into a passing candidate"
            ),
        },
        "receipts": receipts,
        "manifest": {**_descriptor(manifest_path), "sha256": MANIFEST_SHA256},
        "candidate_checkpoint": candidate,
        "screen_artifacts": {
            "launch": _descriptor(launch_path),
            "results": {**_descriptor(results_path), "rows": 16},
            "summary": _descriptor(summary_path),
            "inventory": inventory,
        },
        "instrumentation_replay": instrumentation_replay,
        "summary": {
            "cases": summary["cases"],
            "panels": summary["panels"],
            "quality_gate": summary["quality_gate"],
            "provenance": provenance,
        },
        "implementation": {
            relative: _descriptor(root / relative)
            for relative in (
                "jobs/retrain_v2/audit_chk4_pre2009_correction_selection.py",
                "jobs/retrain_v2/probe_chk4_pre2009_correction.py",
                "jobs/retrain_v2/replay_chk4_pre2009_correction_padding_v3.py",
                "jobs/retrain_v2/probe_chk4_decision_grpo_stratified.py",
                "jobs/retrain_v2/probe_chk4_decision_sft_generation.py",
                "src/open_r1/trainer/rewards/reward_funcs/decision_reward_v3.py",
                "src/open_r1/structured_response.py",
            )
        },
    }


def _decision_path(repo_root: Path, step: int) -> Path:
    return (
        repo_root
        / SCREEN_ROOT
        / "receipts"
        / f"checkpoint-{step}-selection-decision-v1.json"
    ).resolve()


def _prior_decisions(repo_root: Path, step: int) -> list[dict[str, Any]]:
    index = CANDIDATE_STEPS.index(step)
    prior: list[dict[str, Any]] = []
    for previous in CANDIDATE_STEPS[:index]:
        path = _decision_path(repo_root, previous)
        value = _read_json(path, label=f"checkpoint-{previous} decision")
        validate_manifest_integrity(value)
        expected = seal_manifest(
            {
                **audit_step(repo_root, previous),
                "schema_version": SCHEMA,
                "prior_checkpoint_decisions": _prior_decisions(
                    repo_root, previous
                ),
            }
        )
        _require(value == expected, f"checkpoint-{previous} decision replay drift")
        _require(
            value.get("outcome")
            == "advance_allowed_action_direction_insufficiency"
            and value.get("advance_allowed") is True,
            f"checkpoint-{previous} did not authorize advancing",
        )
        prior.append(
            {
                "checkpoint_step": previous,
                "path": str(path),
                "sha256": sha256_file(path),
                "payload_sha256": value["integrity"]["payload_sha256"],
                "outcome": value["outcome"],
            }
        )
    return prior


def seal_decision(repo_root: Path, step: int, *, execute: bool) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    prior = _prior_decisions(root, step)
    audit = audit_step(root, step)
    value = seal_manifest(
        {
            **audit,
            "schema_version": SCHEMA,
            "prior_checkpoint_decisions": prior,
        }
    )
    path = _decision_path(root, step)
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
        "advance_allowed": value["advance_allowed"],
    }


def status(repo_root: Path) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    return {
        "candidate_steps": list(CANDIDATE_STEPS),
        "screens": {
            str(step): {
                "output": str(_screen_output(root, step)),
                "present": _screen_output(root, step).exists(),
                "decision_present": _decision_path(root, step).is_file(),
            }
            for step in CANDIDATE_STEPS
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    for name in ("audit", "seal"):
        command = sub.add_parser(name)
        command.add_argument("--step", type=int, choices=CANDIDATE_STEPS, required=True)
        if name == "seal":
            command.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "status":
            result = status(args.repo_root)
        elif args.command == "audit":
            result = audit_step(args.repo_root, args.step)
        else:
            result = seal_decision(
                args.repo_root, args.step, execute=bool(args.execute)
            )
    except (SelectionAuditError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
