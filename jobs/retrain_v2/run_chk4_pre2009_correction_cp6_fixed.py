"""Authorize and run the final checkpoint-6 correction static screen on GPU1.

Checkpoint-6 is the last preregistered candidate in the fixed 2 -> 4 -> 6
order.  This additive runner binds the immutable checkpoint-4 failed/not-
selected decision and repeats the exact same manifest, seed, batches, sampling
parameters, and first-EOS padding normalization.  Checkpoint-6 must satisfy the
unchanged full gate; this command does not authorize selection, merge, blind
confirmation, or GRPO.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from jobs.retrain_v2 import attest_chk4_pre2009_correction_sft_gap as gap
from jobs.retrain_v2 import audit_chk4_pre2009_correction_cp4_fixed as cp4_audit
from jobs.retrain_v2 import probe_chk4_decision_sft_generation as base_probe
from jobs.retrain_v2 import probe_chk4_pre2009_correction as probe
from jobs.retrain_v2 import run_chk4_pre2009_correction_screen_fixed as cp4_runner
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


RUN_ROOT = gap.RUN_ROOT
TRAINING_OUTPUT = gap.TRAINING_OUTPUT
SCREEN_ROOT = RUN_ROOT / "static_screening"
MANIFEST = cp4_runner.MANIFEST
MANIFEST_SHA256 = cp4_runner.MANIFEST_SHA256
STEP = 6
OUTPUT = SCREEN_ROOT / "checkpoint-6-selection-cap1536-v2"
AUTHORIZATION = (
    SCREEN_ROOT
    / "receipts/checkpoint-6-selection-cap1536-v2_authorization.json"
)
CP4_DECISION = cp4_audit.DECISION
CP4_DECISION_SHA256 = (
    "9d14c98990ecb93efe2961109927e5f62ce5db5290cc61d79154d08fc057c9e8"
)
CP4_DECISION_PAYLOAD_SHA256 = (
    "71039c972a804e0d4017b3766c6cdedd55bfb34deb8f54209b3b573c1fd6521c"
)
CP4_AUDIT_SOURCE_SHA256 = (
    "875af105e12c7fe508ad33a0622616602b980c609ee43223d7c37e14a6448020"
)
CP6_CHECKPOINT_SHA256 = (
    "57dd40d5be277ca32a1bceb92b34b6243cc47e2f5188651aa96d5d77e6336260"
)
CP6_ADAPTER_WEIGHTS_SHA256 = (
    "f00bc9aeea12adba00ffeda47f55262209d961e516e03f00e9b2484ed23d6f79"
)
SCHEMA = "chk4-pre2009-correction-cp6-fixed-screen-authorization-v1"


class Cp6ScreenError(RuntimeError):
    """The final checkpoint-6 screen is not safely launchable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Cp6ScreenError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Cp6ScreenError(f"invalid {label}: {path}") from exc
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


def output_path(repo_root: Path) -> Path:
    return (repo_root.expanduser().resolve() / OUTPUT).resolve()


def authorization_path(repo_root: Path) -> Path:
    return (repo_root.expanduser().resolve() / AUTHORIZATION).resolve()


def _implementation(repo_root: Path) -> dict[str, Any]:
    relatives = (
        "jobs/retrain_v2/run_chk4_pre2009_correction_cp6_fixed.py",
        "jobs/retrain_v2/audit_chk4_pre2009_correction_cp4_fixed.py",
        "jobs/retrain_v2/run_chk4_pre2009_correction_screen_fixed.py",
        "jobs/retrain_v2/audit_chk4_pre2009_correction_selection.py",
        "jobs/retrain_v2/probe_chk4_pre2009_correction.py",
        "jobs/retrain_v2/replay_chk4_pre2009_correction_padding_v3.py",
        "jobs/retrain_v2/probe_chk4_decision_sft_generation.py",
        "jobs/retrain_v2/probe_chk4_decision_grpo_stratified.py",
        "src/open_r1/trainer/rewards/reward_funcs/decision_reward_v3.py",
        "src/open_r1/structured_response.py",
    )
    result = {relative: _descriptor(repo_root / relative) for relative in relatives}
    _require(
        result["jobs/retrain_v2/audit_chk4_pre2009_correction_cp4_fixed.py"][
            "sha256"
        ]
        == CP4_AUDIT_SOURCE_SHA256,
        "sealed checkpoint-4 audit source drift",
    )
    _require(
        result["jobs/retrain_v2/run_chk4_pre2009_correction_screen_fixed.py"][
            "sha256"
        ]
        == cp4_audit.RUNNER_SHA256,
        "sealed checkpoint-4 runner source drift",
    )
    _require(
        result["jobs/retrain_v2/probe_chk4_pre2009_correction.py"]["sha256"]
        == cp4_runner.SOURCE_PROBE_SHA256,
        "sealed generation implementation drift",
    )
    return result


def authorization_payload(repo_root: Path) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    original, original_binding = _sealed_binding(
        root / gap.AUTHORIZATION,
        file_sha256=gap.AUTHORIZATION_SHA256,
        payload_sha256=gap.AUTHORIZATION_PAYLOAD_SHA256,
        label="original SFT authorization",
    )
    _require(original.get("status") == "authorized", "original SFT auth invalid")
    supplemental_result = gap.verify(root)
    _require(
        supplemental_result["sha256"] == cp4_runner.SUPPLEMENTAL_SHA256
        and supplemental_result["payload_sha256"]
        == cp4_runner.SUPPLEMENTAL_PAYLOAD_SHA256,
        "supplemental SFT attestation drift",
    )
    supplemental_value = _read_json(root / gap.ATTESTATION, label="supplemental")
    supplemental_binding = {
        **_descriptor(
            root / gap.ATTESTATION,
            expected_sha256=cp4_runner.SUPPLEMENTAL_SHA256,
        ),
        "payload_sha256": cp4_runner.SUPPLEMENTAL_PAYLOAD_SHA256,
        "status": gap.STATUS,
        "is_not_pre_authorization": True,
    }
    cp2, cp2_binding = _sealed_binding(
        root / cp4_runner.CP2_DECISION,
        file_sha256=cp4_runner.CP2_DECISION_SHA256,
        payload_sha256=cp4_runner.CP2_DECISION_PAYLOAD_SHA256,
        label="checkpoint-2 decision",
    )
    _require(
        cp2.get("outcome") == "advance_allowed_action_direction_insufficiency",
        "checkpoint-2 decision drift",
    )
    cp4, cp4_binding = _sealed_binding(
        root / CP4_DECISION,
        file_sha256=CP4_DECISION_SHA256,
        payload_sha256=CP4_DECISION_PAYLOAD_SHA256,
        label="checkpoint-4 decision",
    )
    _require(
        cp4.get("outcome") == "failed_not_selected_advance_to_final_cp6"
        and cp4.get("selected") is False
        and cp4.get("advance_allowed") is True
        and cp4.get("next_and_final_candidate") == STEP
        and cp4.get("full_gate_unchanged") is True,
        "checkpoint-4 decision does not permit final checkpoint-6 screen",
    )
    manifest_path = (root / MANIFEST).resolve()
    _require(sha256_file(manifest_path) == MANIFEST_SHA256, "manifest hash drift")
    try:
        manifest = probe._validate_manifest(manifest_path, MANIFEST_SHA256)
    except Exception as exc:
        raise Cp6ScreenError(f"selection manifest replay failed: {exc}") from exc
    _require(
        manifest.get("purpose") == probe.SELECTION_PURPOSE,
        "selection manifest purpose drift",
    )
    expected_candidate = supplemental_value.get("completed_run", {}).get(
        "candidate_checkpoint_fingerprints", {}
    ).get(str(STEP))
    _require(isinstance(expected_candidate, Mapping), "cp6 supplemental binding missing")
    candidate_path = (root / TRAINING_OUTPUT / f"checkpoint-{STEP}").resolve()
    candidate = fingerprint_artifact_path(candidate_path)
    _require(
        candidate == expected_candidate
        and candidate.get("sha256") == CP6_CHECKPOINT_SHA256
        and candidate.get("file_count") == 12
        and candidate.get("total_bytes") == 356059569,
        "checkpoint-6 fingerprint drift",
    )
    adapter_weights = _descriptor(
        candidate_path / "adapter_model.safetensors",
        expected_sha256=CP6_ADAPTER_WEIGHTS_SHA256,
    )
    output = output_path(root)
    _require(not output.exists() and not output.is_symlink(), "cp6 output exists")
    implementation = _implementation(root)
    return seal_manifest(
        {
            "schema_version": SCHEMA,
            "status": "authorized",
            "scope": {
                "operation": "checkpoint_6_final_static_generation_screening",
                "gpu": 1,
                "fresh_output": True,
                "not_authorized": [
                    "gpu0",
                    "checkpoint_selection",
                    "merge",
                    "blind_confirmation",
                    "grpo",
                    "generation_parameter_change",
                    "overwrite",
                    "candidate_after_checkpoint6",
                ],
            },
            "lineage": {
                "original_sft_authorization": original_binding,
                "supplemental_post_run_attestation": supplemental_binding,
                "checkpoint_2_decision": cp2_binding,
                "checkpoint_4_failed_not_selected_decision": cp4_binding,
            },
            "candidate_checkpoint": {
                **candidate,
                "adapter_model": adapter_weights,
            },
            "manifest": {
                **_descriptor(manifest_path, expected_sha256=MANIFEST_SHA256),
                "purpose": probe.SELECTION_PURPOSE,
            },
            "generation": {
                "seed": probe.SELECTION_SEED,
                "seed_once_before_all_batches": True,
                "batches": [list(batch) for batch in probe.SELECTION_BATCH_IDS],
                "num_return_sequences": probe.NUM_RETURN_SEQUENCES,
                "max_new_tokens": probe.MAX_NEW_TOKENS,
                "temperature": probe.TEMPERATURE,
                "top_p": probe.TOP_P,
                "padding_postprocessor": (
                    "truncate_at_first_bound_eos_inclusive_before_decode"
                ),
            },
            "output": str(output),
            "model_label": "correction-sft-checkpoint-6",
            "implementation": implementation,
            "methodology": {
                "static_screen_not_bitwise_grpo_replay": True,
                "preregistered_order": [2, 4, 6],
                "checkpoint_6_is_final_candidate": True,
                "full_original_gate_required_for_selection": True,
                "failure_blocks_selection_merge_blind_and_grpo": True,
                "authoritative_gate_after_selection_and_exact_merge": (
                    "fresh_two_step_cap1536_grpo_smoke"
                ),
            },
        }
    )


def authorize(repo_root: Path, *, execute: bool) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    value = authorization_payload(root)
    path = authorization_path(root)
    if not execute:
        return {**value, "status": "ready_to_authorize", "path": str(path)}
    _require(not path.exists() and not path.is_symlink(), "cp6 authorization exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "status": "authorized",
        "path": str(path),
        "sha256": sha256_file(path),
        "payload_sha256": value["integrity"]["payload_sha256"],
    }


def _verify_authorization(repo_root: Path) -> dict[str, Any]:
    path = authorization_path(repo_root)
    observed = _read_json(path, label="checkpoint-6 authorization")
    validate_manifest_integrity(observed)
    _require(observed == authorization_payload(repo_root), "cp6 authorization drift")
    return observed


@contextmanager
def _fixed_padding_postprocessor(repo_root: Path) -> Iterator[None]:
    original_decode = base_probe.decode_completion_preserving_boundary
    original_result = probe._result_row
    original_model_source = probe._model_source
    implementation = _implementation(repo_root)

    def fixed_decode(tokenizer: Any, token_ids: Sequence[int], eos_ids: Any) -> str:
        normalized_eos = set(base_probe._normalize_eos_ids(eos_ids))
        trimmed, _evidence = cp4_runner.normalize_generated_ids(
            token_ids, normalized_eos
        )
        return original_decode(tokenizer, trimmed, normalized_eos)

    def fixed_result(**kwargs: Any) -> dict[str, Any]:
        eos_ids = set(kwargs["provenance"]["eos_token_ids"])
        trimmed, evidence = cp4_runner.normalize_generated_ids(
            kwargs["generated_ids"], eos_ids
        )
        kwargs["generated_ids"] = trimmed
        row = original_result(**kwargs)
        row["batched_padding_normalization"] = evidence
        return row

    def fixed_model_source(**kwargs: Any) -> Mapping[str, Any]:
        source = dict(original_model_source(**kwargs))
        provenance = dict(source["provenance"])
        provenance["batched_padding_postprocessor"] = {
            "schema_version": "batched-first-eos-normalization-v1",
            "method": "truncate_at_first_bound_eos_inclusive_before_decode",
            "generation_sampling_unchanged": True,
            "implementation": implementation[
                "jobs/retrain_v2/run_chk4_pre2009_correction_cp6_fixed.py"
            ],
            "predecessor_implementation": implementation[
                "jobs/retrain_v2/run_chk4_pre2009_correction_screen_fixed.py"
            ],
        }
        source["provenance"] = provenance
        return source

    base_probe.decode_completion_preserving_boundary = fixed_decode
    probe._result_row = fixed_result
    probe._model_source = fixed_model_source
    try:
        yield
    finally:
        base_probe.decode_completion_preserving_boundary = original_decode
        probe._result_row = original_result
        probe._model_source = original_model_source


def launch(repo_root: Path, *, execute: bool) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    authorization = _verify_authorization(root)
    output = output_path(root)
    result: dict[str, Any] = {
        "status": "ready",
        "gpu": 1,
        "step": STEP,
        "output": str(output),
        "authorization": {
            "path": str(authorization_path(root)),
            "sha256": sha256_file(authorization_path(root)),
            "payload_sha256": authorization["integrity"]["payload_sha256"],
        },
    }
    if not execute:
        return result
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "launch requires GPU1")
    with _fixed_padding_postprocessor(root):
        generated = probe.run_screen(
            manifest_path=(root / MANIFEST).resolve(),
            manifest_sha256=MANIFEST_SHA256,
            output_dir=output,
            model_label="correction-sft-checkpoint-6",
            base_model_path=probe.SELECTED_CP38_MODEL,
            adapter_path=(root / TRAINING_OUTPUT / "checkpoint-6").resolve(),
            load_in_4bit=True,
        )
    return dict(generated)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("authorize", "launch"):
        command = sub.add_parser(name)
        command.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "authorize":
            result = authorize(args.repo_root, execute=bool(args.execute))
        else:
            result = launch(args.repo_root, execute=bool(args.execute))
    except (Cp6ScreenError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "launch" and result.get("quality_status") == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
