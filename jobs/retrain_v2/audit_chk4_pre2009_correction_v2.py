"""CPU replay and create-only decisions for correction-v2 static screens.

The auditor independently reconstructs completion text, delivery metrics, and
``decision_dense_v3`` reward from each output's raw padded token IDs.  It then
applies the frozen absolute and exact-cp38-relative gates.  Selection considers
only the prefix of the pre-registered checkpoint order 2 -> 4 -> 6 and locks
the earliest all-pass checkpoint.  Blind and retention confirmations accept
only that locked checkpoint and fail terminally; neither can select a fallback.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.retrain_v2 import probe_chk4_decision_grpo_stratified as reward_probe
from jobs.retrain_v2 import probe_chk4_decision_sft_generation as generation_probe
from jobs.retrain_v2 import probe_chk4_pre2009_correction_v2 as probe
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


SELECTION_RECEIPT_SCHEMA = "chk4-pre2009-correction-v2-selection-decision-v1"
CONFIRMATION_RECEIPT_SCHEMA = (
    "chk4-pre2009-correction-v2-locked-confirmation-decision-v1"
)
RUN_ROOT = probe.RUN_ROOT
RECEIPT_ROOT = probe.SCREEN_ROOT / "receipts"


def selection_receipt_path(last_step: int) -> Path:
    _require(last_step in probe.CANDIDATE_ORDER, "invalid selection receipt step")
    return RECEIPT_ROOT / f"selection-after-checkpoint-{last_step}.json"


def confirmation_receipt_path(stage: str) -> Path:
    _require(
        stage in {probe.BLIND_STAGE, probe.RETENTION_STAGE},
        "invalid confirmation stage",
    )
    return RECEIPT_ROOT / f"{stage}-confirmation.json"


class CorrectionV2AuditError(RuntimeError):
    """A static screen could not be replayed or did not satisfy its lineage."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorrectionV2AuditError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorrectionV2AuditError(f"invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must contain an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CorrectionV2AuditError(f"cannot read {label}: {path}") from exc
    for line_number, line in enumerate(lines, 1):
        _require(bool(line), f"{label}:{line_number}: blank row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CorrectionV2AuditError(
                f"{label}:{line_number}: invalid JSON"
            ) from exc
        _require(isinstance(row, dict), f"{label}:{line_number}: not an object")
        rows.append(row)
    return rows


def _descriptor(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing artifact: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _expected_order(
    manifest: Mapping[str, Any], stage: str
) -> list[tuple[int, str, int]]:
    return [
        (batch_index, str(sample_id), generation_index)
        for batch_index, batch in enumerate(manifest["stages"][stage]["batches"])
        for sample_id in batch
        for generation_index in range(1, probe.NUM_RETURN_SEQUENCES + 1)
    ]


def _load_tokenizer(manifest: Mapping[str, Any]) -> Any:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            str(manifest["tokenizer"]["path"]),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise CorrectionV2AuditError(f"cannot load bound tokenizer: {exc}") from exc


def _replay_row(
    row: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    stage: str,
    role: str,
    eos_token_ids: set[int],
    tokenizer: Any,
) -> None:
    sample_id = str(row.get("sample_id"))
    _require(sample_id in manifest["stages"][stage]["sample_ids"], "unexpected sample")
    target = manifest["samples"][sample_id]
    _require(
        row.get("schema_version") == probe.RESULT_SCHEMA
        and row.get("stage") == stage
        and row.get("model_role") == role
        and row.get("target_direction") == target["direction"]
        and row.get("target_magnitude_bp") == target["magnitude_bp"]
        and row.get("population_role") == target["population_role"],
        f"row contract drift: {sample_id}",
    )
    stage_contract = manifest["stages"][stage]
    _require(
        row.get("seed") == stage_contract["seed"]["seed"]
        and row.get("generation_parameters") == stage_contract["generation"],
        f"generation contract drift: {sample_id}",
    )
    raw_ids = row.get("generated_token_ids_raw_padded")
    normalized_ids = row.get("generated_token_ids_first_eos_inclusive")
    _require(
        isinstance(raw_ids, list)
        and isinstance(normalized_ids, list)
        and raw_ids
        and normalized_ids
        and all(
            isinstance(value, int) and not isinstance(value, bool) for value in raw_ids
        )
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in normalized_ids
        ),
        f"raw/normalized token IDs invalid: {sample_id}",
    )
    rebuilt_ids, evidence = probe.normalize_at_first_bound_eos(raw_ids, eos_token_ids)
    _require(rebuilt_ids == normalized_ids, f"normalized IDs drift: {sample_id}")
    _require(
        row.get("batched_padding_normalization") == evidence,
        f"normalization evidence drift: {sample_id}",
    )
    completion = generation_probe.decode_completion_preserving_boundary(
        tokenizer, normalized_ids, eos_token_ids
    )
    _require(
        row.get("completion") == completion
        and row.get("completion_sha256") == probe.sha256_text(completion),
        f"completion replay drift: {sample_id}",
    )
    metrics = generation_probe.analyze_completion(
        text=completion,
        generated_token_ids=normalized_ids,
        eos_token_ids=eos_token_ids,
        max_new_tokens=probe.MAX_NEW_TOKENS,
        tail_tokens=probe.TAIL_TOKENS,
    )
    for key, value in metrics.items():
        _require(row.get(key) == value, f"metric replay drift {key}: {sample_id}")
    replay = reward_probe.replay_decision_dense_v3(
        completion,
        {"direction": target["direction"], "magnitude_bp": target["magnitude_bp"]},
        hit_eos=bool(metrics["hit_eos"]),
        cap_reached=bool(metrics["cap_reached"]),
    )
    mapping = {
        "decision_dense_v3_reward": "reward",
        "decision_prediction": "prediction",
        "decision_direction_correct": "direction_correct",
        "decision_exact": "exact",
        "decision_rejection_reason": "rejection_reason",
        "decision_forced_zero_reason": "forced_zero_reason",
        "response_format": "response_format",
        "strict_json": "strict_json",
        "fenced_json": "fenced_json",
    }
    for row_key, replay_key in mapping.items():
        _require(
            row.get(row_key) == replay[replay_key],
            f"reward replay drift {row_key}: {sample_id}",
        )
    provenance = row.get("provenance")
    _require(
        isinstance(provenance, Mapping)
        and provenance.get("manifest_sha256") == manifest_sha256,
        f"row manifest binding drift: {sample_id}",
    )


def replay_output(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    stage: str,
    role: str,
    output_dir: Path,
    checkpoint_step: int | None = None,
) -> dict[str, Any]:
    """Independently replay one complete baseline or candidate output."""

    output = output_dir.expanduser().resolve()
    _require(output.is_dir() and not output.is_symlink(), f"missing output: {output}")
    inventory = sorted(
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    _require(
        inventory == ["launch.json", "results.jsonl", "summary.json"],
        f"output inventory drift: {inventory}",
    )
    launch_path = output / "launch.json"
    results_path = output / "results.jsonl"
    summary_path = output / "summary.json"
    launch = _read_json(launch_path, label="launch")
    rows = _read_jsonl(results_path, label="results")
    summary = _read_json(summary_path, label="summary")
    _require(
        launch.get("stage") == stage
        and summary.get("stage") == stage
        and launch.get("model_role") == role
        and summary.get("model_role") == role,
        "output stage/role drift",
    )
    _require(
        summary.get("status") == "generation_complete"
        and summary.get("quality_status")
        == "not_evaluated_without_paired_baseline_or_candidate",
        "output did not complete generation-only stage",
    )
    _require(
        summary.get("results", {}).get("sha256") == sha256_file(results_path)
        and summary.get("results", {}).get("rows") == len(rows)
        and summary.get("results", {}).get("raw_padded_and_normalized_ids_landed")
        is True,
        "summary/results binding drift",
    )
    observed_order = [
        (
            int(row.get("batch_index", -1)),
            str(row.get("sample_id")),
            int(row.get("generation_index", -1)),
        )
        for row in rows
    ]
    _require(observed_order == _expected_order(manifest, stage), "result order drift")
    provenance = summary.get("provenance")
    _require(isinstance(provenance, Mapping), "summary provenance missing")
    eos_values = provenance.get("eos_token_ids")
    _require(
        isinstance(eos_values, list)
        and eos_values
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in eos_values
        ),
        "EOS binding invalid",
    )
    authorization_binding = provenance.get("authorization")
    _require(
        isinstance(authorization_binding, Mapping)
        and set(authorization_binding) == {"path", "sha256", "payload_sha256"},
        "output authorization binding missing",
    )
    authorization_path = Path(str(authorization_binding["path"])).resolve()
    authorization = _read_json(authorization_path, label="output authorization")
    suffix = "baseline" if role == "baseline" else f"checkpoint-{checkpoint_step}"
    expected_authorization_path = (
        probe.SCREEN_ROOT / "authorizations" / f"{stage}-{suffix}.json"
    ).resolve()
    _require(
        authorization_path == expected_authorization_path
        and authorization_path.stat().st_mode & 0o777 == 0o400,
        "output authorization path/mode drift",
    )
    validate_manifest_integrity(authorization)
    _require(
        authorization.get("schema_version")
        == "chk4-pre2009-correction-v2-static-screen-authorization-v1"
        and authorization.get("status") == "authorized"
        and sha256_file(authorization_path) == authorization_binding["sha256"]
        and authorization.get("integrity", {}).get("payload_sha256")
        == authorization_binding["payload_sha256"],
        "output authorization hash drift",
    )
    _require(
        authorization.get("scope", {}).get("stage") == stage
        and authorization.get("scope", {}).get("model_role") == role
        and authorization.get("scope", {}).get("checkpoint_step") == checkpoint_step
        and authorization.get("scope", {}).get("gpu") == 1
        and authorization.get("output") == str(output)
        and authorization.get("manifest", {}).get("sha256") == manifest_sha256,
        "output authorization scope drift",
    )
    _require(
        authorization.get("stage_contract") == manifest["stages"][stage],
        "output authorization stage contract drift",
    )
    authorized_implementation = authorization.get("implementation")
    _require(
        isinstance(authorized_implementation, Mapping)
        and set(authorized_implementation) == set(probe.AUTH_IMPLEMENTATION_RELATIVES),
        "authorization implementation inventory drift",
    )
    for relative, descriptor in authorized_implementation.items():
        _require(isinstance(descriptor, Mapping), "invalid implementation descriptor")
        implementation_path = Path(str(descriptor.get("path")))
        _require(
            implementation_path == (probe.REPO_ROOT / relative).resolve()
            and implementation_path.is_file()
            and not implementation_path.is_symlink()
            and sha256_file(implementation_path) == descriptor.get("sha256"),
            f"authorized implementation drift: {relative}",
        )
    tokenizer = _load_tokenizer(manifest)
    for row in rows:
        _require(
            row.get("provenance", {}).get("authorization_sha256")
            == authorization_binding["sha256"],
            "row authorization binding drift",
        )
        _replay_row(
            row,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            stage=stage,
            role=role,
            eos_token_ids=set(eos_values),
            tokenizer=tokenizer,
        )
    metrics = probe.summarize_rows(rows, manifest, stage)
    _require(summary.get("metrics") == metrics, "summary metric replay drift")
    _require(
        provenance.get("manifest", {}).get("sha256") == manifest_sha256,
        "summary manifest binding drift",
    )
    model_source = provenance.get("model_source")
    _require(isinstance(model_source, Mapping), "model source provenance missing")
    if role == "baseline":
        _require(
            provenance.get("model_sha256") == probe.SELECTED_CP38_MODEL_SHA256
            and provenance.get("effective_model_sha256")
            == probe.SELECTED_CP38_MODEL_SHA256
            and model_source.get("mode") == reward_probe.MERGED_MODEL_MODE
            and model_source.get("model_path") == str(probe.SELECTED_CP38_MODEL),
            "baseline is not exact cp38",
        )
    else:
        _require(checkpoint_step in probe.CANDIDATE_ORDER, "candidate step invalid")
        adapter = model_source.get("adapter")
        _require(isinstance(adapter, Mapping), "candidate adapter binding missing")
        directory = adapter.get("directory")
        _require(
            isinstance(directory, Mapping)
            and directory.get("path")
            == str(probe.SFT_OUTPUT / f"checkpoint-{checkpoint_step}"),
            "candidate checkpoint path drift",
        )
        base = model_source.get("base_model")
        _require(
            isinstance(base, Mapping)
            and base.get("directory", {}).get("path") == str(probe.SELECTED_CP38_MODEL)
            and base.get("directory", {}).get("sha256")
            == probe.SELECTED_CP38_MODEL_SHA256,
            "candidate base is not exact cp38",
        )
    _require(
        authorization.get("model_sha256") == provenance.get("model_sha256")
        and authorization.get("effective_model_sha256")
        == provenance.get("effective_model_sha256")
        and authorization.get("model_source") == model_source,
        "authorization/model output binding drift",
    )
    for path in (launch_path, results_path, summary_path):
        _require(path.stat().st_mode & 0o777 == 0o444, f"unsealed file: {path}")
    _require(output.stat().st_mode & 0o777 == 0o555, "unsealed output directory")
    return {
        "output": str(output),
        "artifacts": {
            "launch": _descriptor(launch_path),
            "results": {**_descriptor(results_path), "rows": len(rows)},
            "summary": _descriptor(summary_path),
        },
        "model_label": summary.get("model_label"),
        "model_source": model_source,
        "effective_model_sha256": provenance.get("effective_model_sha256"),
        "metrics": metrics,
    }


def _prompt_absolute_reasons(
    *, stage: str, sample_id: str, block: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    direction = str(block["target_direction"])
    minimum = 2 if direction == "hold" else 1
    if int(block["correct_nonzero_count"]) < minimum:
        reasons.append(f"{stage}:{sample_id}:correct_nonzero_lt_{minimum}")
    if int(block["strict_json_exact_count"]) < 1:
        reasons.append(f"{stage}:{sample_id}:strict_json_exact_lt_1")
    if not (
        float(block["reward_std"]) > 0
        or int(block["strict_json_exact_count"]) == probe.NUM_RETURN_SEQUENCES
    ):
        reasons.append(f"{stage}:{sample_id}:reward_zero_std_without_4_strict_exact")
    if int(block["boundary_count"]) < 3:
        reasons.append(f"{stage}:{sample_id}:boundary_count_lt_3")
    if int(block["cap_count"]) > 1:
        reasons.append(f"{stage}:{sample_id}:cap_count_gt_1")
    if int(block["periodic_tail_count"]) > 0:
        reasons.append(f"{stage}:{sample_id}:periodic_tail_present")
    if block.get("rewards_finite") is not True:
        reasons.append(f"{stage}:{sample_id}:nonfinite_reward")
    return reasons


def evaluate_selection_or_blind(
    *,
    stage: str,
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    _require(stage in {probe.SELECTION_STAGE, probe.BLIND_STAGE}, "wrong gate stage")
    _require(
        set(baseline_metrics["prompts"]) == set(candidate_metrics["prompts"]),
        "baseline/candidate prompt mismatch",
    )
    reasons: list[str] = []
    for sample_id, block in candidate_metrics["prompts"].items():
        reasons.extend(
            _prompt_absolute_reasons(stage=stage, sample_id=sample_id, block=block)
        )
    baseline = baseline_metrics["aggregate_correct_nonzero"]
    candidate = candidate_metrics["aggregate_correct_nonzero"]
    thresholds = {
        "hold": max(4, int(baseline["hold"]) - 1),
        "hike": min(8, max(2, int(baseline["hike"]) + 1)),
        "cut": max(2, int(baseline["cut"])),
    }
    for direction, threshold in thresholds.items():
        if int(candidate[direction]) < threshold:
            reasons.append(
                f"{stage}:aggregate_{direction}_correct_nonzero_lt_{threshold}"
            )
    return {
        "quality_status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "aggregate_thresholds": thresholds,
        "baseline": dict(baseline_metrics),
        "candidate": dict(candidate_metrics),
    }


def evaluate_retention(
    *, baseline_metrics: Mapping[str, Any], candidate_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    _require(
        set(baseline_metrics["prompts"]) == set(candidate_metrics["prompts"]),
        "retention baseline/candidate prompt mismatch",
    )
    reasons: list[str] = []
    thresholds: dict[str, Any] = {}
    for sample_id, candidate in candidate_metrics["prompts"].items():
        baseline = baseline_metrics["prompts"][sample_id]
        correct_min = max(1, int(baseline["correct_nonzero_count"]))
        strict_min = int(baseline["strict_json_exact_count"])
        thresholds[sample_id] = {
            "correct_nonzero_min": correct_min,
            "strict_json_exact_min": strict_min,
        }
        if int(candidate["correct_nonzero_count"]) < correct_min:
            reasons.append(
                f"retention:{sample_id}:correct_nonzero_lt_baseline_floor_{correct_min}"
            )
        if int(candidate["strict_json_exact_count"]) < strict_min:
            reasons.append(
                f"retention:{sample_id}:strict_json_exact_lt_baseline_{strict_min}"
            )
        if not (
            float(candidate["reward_std"]) > 0
            or int(candidate["strict_json_exact_count"]) == probe.NUM_RETURN_SEQUENCES
        ):
            reasons.append(
                f"retention:{sample_id}:reward_zero_std_without_4_strict_exact"
            )
        if int(candidate["boundary_count"]) < 3:
            reasons.append(f"retention:{sample_id}:boundary_count_lt_3")
        if int(candidate["cap_count"]) > 1:
            reasons.append(f"retention:{sample_id}:cap_count_gt_1")
        if int(candidate["periodic_tail_count"]) > 0:
            reasons.append(f"retention:{sample_id}:periodic_tail_present")
        if candidate.get("rewards_finite") is not True:
            reasons.append(f"retention:{sample_id}:nonfinite_reward")
    return {
        "quality_status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "per_prompt_thresholds": thresholds,
        "baseline": dict(baseline_metrics),
        "candidate": dict(candidate_metrics),
    }


def _implementation(repo_root: Path) -> dict[str, Any]:
    return {
        relative: _descriptor(repo_root / relative)
        for relative in probe.AUTH_IMPLEMENTATION_RELATIVES
    }


def selection_decision(
    *,
    repo_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    baseline_output: Path,
    candidates: Sequence[tuple[int, Path]],
) -> dict[str, Any]:
    manifest = probe.validate_manifest(manifest_path.resolve(), manifest_sha256)
    steps = [step for step, _ in candidates]
    _require(
        steps == list(probe.CANDIDATE_ORDER[: len(steps)]) and bool(steps),
        "candidate outputs must be a non-empty 2->4->6 prefix",
    )
    baseline = replay_output(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        stage=probe.SELECTION_STAGE,
        role="baseline",
        output_dir=baseline_output,
    )
    outcomes: dict[str, Any] = {}
    selected: int | None = None
    for index, (step, output) in enumerate(candidates):
        candidate = replay_output(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            stage=probe.SELECTION_STAGE,
            role="candidate",
            output_dir=output,
            checkpoint_step=step,
        )
        gate = evaluate_selection_or_blind(
            stage=probe.SELECTION_STAGE,
            baseline_metrics=baseline["metrics"],
            candidate_metrics=candidate["metrics"],
        )
        outcomes[str(step)] = {"screen": candidate, "gate": gate}
        if gate["quality_status"] == "passed":
            _require(
                index == len(candidates) - 1, "later candidate ran after earlier pass"
            )
            selected = step
            break
    if selected is not None:
        status = "selection_passed"
        outcome = "earliest_all_pass_checkpoint_locked"
        terminal = False
        next_step = None
    elif len(candidates) < len(probe.CANDIDATE_ORDER):
        status = "selection_in_progress"
        outcome = "candidate_failed_next_preregistered_candidate_allowed"
        terminal = False
        next_step = probe.CANDIDATE_ORDER[len(candidates)]
    else:
        status = "selection_failed"
        outcome = "no_preregistered_checkpoint_passed_full_gate"
        terminal = True
        next_step = None
    return {
        "schema_version": SELECTION_RECEIPT_SCHEMA,
        "status": status,
        "outcome": outcome,
        "selected_checkpoint": selected,
        "next_checkpoint_allowed": next_step,
        "terminal": terminal,
        "candidate_order": list(probe.CANDIDATE_ORDER),
        "evaluated_candidate_prefix": steps,
        "baseline": baseline,
        "candidate_outcomes": outcomes,
        "manifest": {**_descriptor(manifest_path), "sha256": manifest_sha256},
        "methodology": manifest["methodology"],
        "implementation": _implementation(repo_root.resolve()),
        "prohibited": (
            ["blind", "retention", "merge", "grpo"]
            if selected is None
            else [
                "fallback_checkpoint",
                "seed_change",
                "gate_change",
                "merge_before_blind_and_retention",
                "grpo_before_blind_and_retention",
            ]
        ),
    }


def _verified_selection_receipt(path: Path) -> dict[str, Any]:
    receipt = _read_json(path, label="selection receipt")
    validate_manifest_integrity(receipt)
    _require(
        receipt.get("schema_version") == SELECTION_RECEIPT_SCHEMA
        and receipt.get("status") == "selection_passed"
        and receipt.get("selected_checkpoint") in probe.CANDIDATE_ORDER,
        "selection receipt has no locked checkpoint",
    )
    return receipt


def confirmation_decision(
    *,
    repo_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    stage: str,
    selection_receipt_path: Path,
    predecessor_receipt_path: Path | None,
    baseline_output: Path,
    candidate_output: Path,
) -> dict[str, Any]:
    _require(
        stage in {probe.BLIND_STAGE, probe.RETENTION_STAGE},
        "invalid confirmation stage",
    )
    manifest = probe.validate_manifest(manifest_path.resolve(), manifest_sha256)
    selection = verify_selection_receipt(
        path=selection_receipt_path.resolve(),
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    _require(
        selection.get("manifest", {}).get("sha256") == manifest_sha256,
        "selection receipt belongs to another suite manifest",
    )
    locked_step = int(selection["selected_checkpoint"])
    predecessor = None
    if stage == probe.RETENTION_STAGE:
        _require(
            predecessor_receipt_path is not None, "retention requires blind receipt"
        )
        predecessor = _read_json(
            predecessor_receipt_path.resolve(), label="blind receipt"
        )
        validate_manifest_integrity(predecessor)
        verify_confirmation_receipt(
            path=predecessor_receipt_path.resolve(),
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
        )
        _require(
            predecessor.get("schema_version") == CONFIRMATION_RECEIPT_SCHEMA
            and predecessor.get("stage") == probe.BLIND_STAGE
            and predecessor.get("status") == "confirmation_passed"
            and predecessor.get("locked_checkpoint") == locked_step,
            "blind predecessor did not pass for locked checkpoint",
        )
        _require(
            predecessor.get("manifest", {}).get("sha256") == manifest_sha256,
            "blind predecessor belongs to another suite manifest",
        )
        bound_selection = predecessor.get("selection_receipt")
        _require(
            isinstance(bound_selection, Mapping)
            and bound_selection.get("path") == str(selection_receipt_path.resolve())
            and bound_selection.get("sha256")
            == sha256_file(selection_receipt_path.resolve()),
            "blind predecessor does not bind the exact selection receipt",
        )
    baseline = replay_output(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        stage=stage,
        role="baseline",
        output_dir=baseline_output,
    )
    candidate = replay_output(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        stage=stage,
        role="candidate",
        output_dir=candidate_output,
        checkpoint_step=locked_step,
    )
    if stage == probe.BLIND_STAGE:
        gate = evaluate_selection_or_blind(
            stage=stage,
            baseline_metrics=baseline["metrics"],
            candidate_metrics=candidate["metrics"],
        )
    else:
        gate = evaluate_retention(
            baseline_metrics=baseline["metrics"],
            candidate_metrics=candidate["metrics"],
        )
    passed = gate["quality_status"] == "passed"
    return {
        "schema_version": CONFIRMATION_RECEIPT_SCHEMA,
        "status": "confirmation_passed" if passed else "confirmation_failed",
        "stage": stage,
        "outcome": (
            "locked_checkpoint_confirmed"
            if passed
            else "locked_checkpoint_failed_terminal_no_fallback"
        ),
        "locked_checkpoint": locked_step,
        "terminal": not passed,
        "advance_allowed": passed,
        "gate": gate,
        "baseline": baseline,
        "candidate": candidate,
        "selection_receipt": _descriptor(selection_receipt_path.resolve()),
        "predecessor_receipt": (
            _descriptor(predecessor_receipt_path.resolve())
            if predecessor_receipt_path is not None
            else None
        ),
        "manifest": {**_descriptor(manifest_path), "sha256": manifest_sha256},
        "implementation": _implementation(repo_root.resolve()),
        "prohibited": (
            ["fallback_checkpoint", "rerun_seed", "gate_change", "merge", "grpo"]
            if not passed
            else (
                [
                    "fallback_checkpoint",
                    "rerun_seed",
                    "gate_change",
                    "merge_before_retention",
                    "grpo_before_retention",
                ]
                if stage == probe.BLIND_STAGE
                else [
                    "fallback_checkpoint",
                    "rerun_seed",
                    "gate_change",
                    "full_grpo_before_authoritative_smoke",
                ]
            )
        ),
    }


def verify_selection_receipt(
    *, path: Path, manifest_path: Path, manifest_sha256: str
) -> dict[str, Any]:
    """Replay every bound output and require exact sealed selection payload."""

    receipt_path = path.expanduser().resolve()
    observed = _read_json(receipt_path, label="selection receipt")
    _require(
        receipt_path.stat().st_mode & 0o777 == 0o400,
        "selection receipt mode drift",
    )
    validate_manifest_integrity(observed)
    _require(
        observed.get("schema_version") == SELECTION_RECEIPT_SCHEMA,
        "selection receipt schema drift",
    )
    steps = observed.get("evaluated_candidate_prefix")
    _require(
        isinstance(steps, list)
        and steps
        and all(isinstance(step, int) for step in steps),
        "selection receipt candidate prefix invalid",
    )
    _require(
        receipt_path == selection_receipt_path(int(steps[-1])).resolve(),
        "selection receipt path is not canonical",
    )
    baseline_output = Path(str(observed.get("baseline", {}).get("output")))
    outcomes = observed.get("candidate_outcomes")
    _require(isinstance(outcomes, Mapping), "selection candidate outcomes missing")
    candidates: list[tuple[int, Path]] = []
    for step in steps:
        outcome = outcomes.get(str(step))
        _require(isinstance(outcome, Mapping), f"selection outcome missing: {step}")
        candidates.append((step, Path(str(outcome.get("screen", {}).get("output")))))
    rebuilt = selection_decision(
        repo_root=probe.REPO_ROOT,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        baseline_output=baseline_output,
        candidates=candidates,
    )
    _require(
        seal_manifest(rebuilt) == observed, "selection receipt semantic replay drift"
    )
    return observed


def verify_confirmation_receipt(
    *, path: Path, manifest_path: Path, manifest_sha256: str
) -> dict[str, Any]:
    """Replay a locked blind/retention receipt and all predecessor semantics."""

    receipt_path = path.expanduser().resolve()
    observed = _read_json(receipt_path, label="confirmation receipt")
    _require(
        receipt_path.stat().st_mode & 0o777 == 0o400,
        "confirmation receipt mode drift",
    )
    validate_manifest_integrity(observed)
    stage = str(observed.get("stage"))
    _require(
        observed.get("schema_version") == CONFIRMATION_RECEIPT_SCHEMA
        and stage in {probe.BLIND_STAGE, probe.RETENTION_STAGE},
        "confirmation receipt identity drift",
    )
    _require(
        receipt_path == confirmation_receipt_path(stage).resolve(),
        "confirmation receipt path is not canonical",
    )
    selection_path = Path(str(observed.get("selection_receipt", {}).get("path")))
    verify_selection_receipt(
        path=selection_path,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    predecessor_record = observed.get("predecessor_receipt")
    predecessor_path = None
    if stage == probe.RETENTION_STAGE:
        _require(
            isinstance(predecessor_record, Mapping),
            "retention blind predecessor missing",
        )
        predecessor_path = Path(str(predecessor_record.get("path")))
        verify_confirmation_receipt(
            path=predecessor_path,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
        )
    else:
        _require(
            predecessor_record is None, "blind cannot have confirmation predecessor"
        )
    rebuilt = confirmation_decision(
        repo_root=probe.REPO_ROOT,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        stage=stage,
        selection_receipt_path=selection_path,
        predecessor_receipt_path=predecessor_path,
        baseline_output=Path(str(observed.get("baseline", {}).get("output"))),
        candidate_output=Path(str(observed.get("candidate", {}).get("output"))),
    )
    _require(
        seal_manifest(rebuilt) == observed,
        "confirmation receipt semantic replay drift",
    )
    return observed


def _write_receipt(
    path: Path, value: Mapping[str, Any], *, execute: bool
) -> dict[str, Any]:
    sealed = seal_manifest(value)
    if not execute:
        return {
            **sealed,
            "dry_run_status": "ready_to_seal",
            "path": str(path.resolve()),
        }
    _require(not path.exists() and not path.is_symlink(), f"receipt exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(sealed, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "status": sealed["status"],
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "payload_sha256": sealed["integrity"]["payload_sha256"],
    }


def _candidate_argument(value: str) -> tuple[int, Path]:
    try:
        raw_step, raw_path = value.split("=", 1)
        step = int(raw_step)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("candidate must be STEP=PATH") from exc
    if step not in probe.CANDIDATE_ORDER or not raw_path:
        raise argparse.ArgumentTypeError("candidate step must be 2, 4, or 6")
    return step, Path(raw_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    selection = sub.add_parser("selection")
    selection.add_argument("--repo-root", type=Path, default=Path.cwd())
    selection.add_argument("--manifest", type=Path, required=True)
    selection.add_argument("--manifest-sha256", required=True)
    selection.add_argument("--baseline-output", type=Path, required=True)
    selection.add_argument(
        "--candidate", type=_candidate_argument, action="append", required=True
    )
    selection.add_argument("--output", type=Path, required=True)
    selection.add_argument("--execute", action="store_true")
    confirmation = sub.add_parser("confirmation")
    confirmation.add_argument("--repo-root", type=Path, default=Path.cwd())
    confirmation.add_argument(
        "--stage", choices=(probe.BLIND_STAGE, probe.RETENTION_STAGE), required=True
    )
    confirmation.add_argument("--manifest", type=Path, required=True)
    confirmation.add_argument("--manifest-sha256", required=True)
    confirmation.add_argument("--selection-receipt", type=Path, required=True)
    confirmation.add_argument("--predecessor-receipt", type=Path)
    confirmation.add_argument("--baseline-output", type=Path, required=True)
    confirmation.add_argument("--candidate-output", type=Path, required=True)
    confirmation.add_argument("--output", type=Path, required=True)
    confirmation.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "selection":
            value = selection_decision(
                repo_root=args.repo_root,
                manifest_path=args.manifest,
                manifest_sha256=args.manifest_sha256,
                baseline_output=args.baseline_output,
                candidates=args.candidate,
            )
            expected_output = selection_receipt_path(args.candidate[-1][0]).resolve()
        else:
            value = confirmation_decision(
                repo_root=args.repo_root,
                manifest_path=args.manifest,
                manifest_sha256=args.manifest_sha256,
                stage=args.stage,
                selection_receipt_path=args.selection_receipt,
                predecessor_receipt_path=args.predecessor_receipt,
                baseline_output=args.baseline_output,
                candidate_output=args.candidate_output,
            )
            expected_output = confirmation_receipt_path(args.stage).resolve()
        _require(
            args.output.resolve() == expected_output,
            f"receipt output must use canonical path: {expected_output}",
        )
        result = _write_receipt(args.output, value, execute=bool(args.execute))
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (CorrectionV2AuditError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
