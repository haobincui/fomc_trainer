"""Authorize and run fixed-padding correction checkpoint screens on GPU1.

The original static-screen implementation remains byte-identical because the
sealed checkpoint-2 decision binds its source hash.  This additive runner
installs a process-local postprocessor: it truncates batched token tensors at
the first bound EOS before decode, metrics, reward, and persistence.  Sampling,
seed, batch order, model, and manifest are unchanged.
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
from jobs.retrain_v2 import probe_chk4_decision_sft_generation as base_probe
from jobs.retrain_v2 import probe_chk4_pre2009_correction as probe
from jobs.retrain_v2 import replay_chk4_pre2009_correction_padding_v3 as padding
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


RUN_ROOT = gap.RUN_ROOT
TRAINING_OUTPUT = gap.TRAINING_OUTPUT
SCREEN_ROOT = RUN_ROOT / "static_screening"
MANIFEST = SCREEN_ROOT / "manifests/checkpoint_selection_cap1536_v1.json"
MANIFEST_SHA256 = padding.MANIFEST_SHA256
ORIGINAL_AUTH_SHA256 = gap.AUTHORIZATION_SHA256
ORIGINAL_AUTH_PAYLOAD_SHA256 = gap.AUTHORIZATION_PAYLOAD_SHA256
SUPPLEMENTAL_SHA256 = padding.SUPPLEMENTAL_ATTESTATION_SHA256
SUPPLEMENTAL_PAYLOAD_SHA256 = (
    "3a344de0237c38be9c79dd875ef88e6e9554b318b0d54cfa901fec033518cd72"
)
CP2_DECISION = SCREEN_ROOT / "receipts/checkpoint-2-selection-decision-v1.json"
CP2_DECISION_SHA256 = (
    "5fee3bce94006dbb38fd739273c2f2e92ea891a023aff63242f2b0b79dd2469f"
)
CP2_DECISION_PAYLOAD_SHA256 = (
    "a90b1654ab7caa43d980e0263fd48e6f15d276f7c29bbfa9495888123c29bdd7"
)
SOURCE_PROBE_SHA256 = padding.SOURCE_PROBE_SHA256
SUPPORTED_STEPS = (4,)
SCHEMA = "chk4-pre2009-correction-fixed-padding-screen-authorization-v1"


class FixedScreenError(RuntimeError):
    """A fixed-padding checkpoint screen is not safely launchable."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FixedScreenError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FixedScreenError(f"invalid {label}: {path}") from exc
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


def output_path(repo_root: Path, step: int) -> Path:
    _require(step in SUPPORTED_STEPS, "only checkpoint-4 is currently eligible")
    return (
        repo_root.expanduser().resolve()
        / SCREEN_ROOT
        / f"checkpoint-{step}-selection-cap1536-v2"
    ).resolve()


def authorization_path(repo_root: Path, step: int) -> Path:
    return (
        repo_root.expanduser().resolve()
        / SCREEN_ROOT
        / "receipts"
        / f"checkpoint-{step}-selection-cap1536-v2_authorization.json"
    ).resolve()


def normalize_generated_ids(
    generated_token_ids: Sequence[int], eos_token_ids: set[int]
) -> tuple[list[int], dict[str, Any]]:
    trimmed, evidence = padding.truncate_at_first_eos(
        generated_token_ids, eos_token_ids
    )
    return trimmed, {
        "schema_version": "batched-first-eos-normalization-v1",
        "method": "truncate_at_first_eos_inclusive",
        "generation_not_changed": True,
        **evidence,
    }


def _implementation(repo_root: Path) -> dict[str, Any]:
    relatives = (
        "jobs/retrain_v2/run_chk4_pre2009_correction_screen_fixed.py",
        "jobs/retrain_v2/probe_chk4_pre2009_correction.py",
        "jobs/retrain_v2/replay_chk4_pre2009_correction_padding_v3.py",
        "jobs/retrain_v2/probe_chk4_decision_sft_generation.py",
        "jobs/retrain_v2/probe_chk4_decision_grpo_stratified.py",
        "src/open_r1/trainer/rewards/reward_funcs/decision_reward_v3.py",
        "src/open_r1/structured_response.py",
    )
    result = {relative: _descriptor(repo_root / relative) for relative in relatives}
    _require(
        result["jobs/retrain_v2/probe_chk4_pre2009_correction.py"]["sha256"]
        == SOURCE_PROBE_SHA256,
        "sealed source probe implementation drift",
    )
    return result


def _receipt_binding(
    path: Path, *, file_sha256: str, payload_sha256: str, label: str
) -> dict[str, Any]:
    value = _read_json(path, label=label)
    validate_manifest_integrity(value)
    _require(
        sha256_file(path) == file_sha256
        and value.get("integrity", {}).get("payload_sha256") == payload_sha256,
        f"{label} binding drift",
    )
    return {
        **_descriptor(path, expected_sha256=file_sha256),
        "payload_sha256": payload_sha256,
    }


def authorization_payload(repo_root: Path, step: int) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    _require(step == 4, "checkpoint-4 is the only next pre-registered candidate")
    original = _receipt_binding(
        root / gap.AUTHORIZATION,
        file_sha256=ORIGINAL_AUTH_SHA256,
        payload_sha256=ORIGINAL_AUTH_PAYLOAD_SHA256,
        label="original SFT authorization",
    )
    supplemental_result = gap.verify(root)
    _require(
        supplemental_result["sha256"] == SUPPLEMENTAL_SHA256
        and supplemental_result["payload_sha256"] == SUPPLEMENTAL_PAYLOAD_SHA256,
        "supplemental SFT attestation drift",
    )
    supplemental = {
        **_descriptor(root / gap.ATTESTATION, expected_sha256=SUPPLEMENTAL_SHA256),
        "payload_sha256": SUPPLEMENTAL_PAYLOAD_SHA256,
        "status": gap.STATUS,
        "is_not_pre_authorization": True,
    }
    cp2 = _receipt_binding(
        root / CP2_DECISION,
        file_sha256=CP2_DECISION_SHA256,
        payload_sha256=CP2_DECISION_PAYLOAD_SHA256,
        label="checkpoint-2 selection decision",
    )
    cp2_value = _read_json(root / CP2_DECISION, label="checkpoint-2 decision")
    _require(
        cp2_value.get("outcome")
        == "advance_allowed_action_direction_insufficiency"
        and cp2_value.get("advance_allowed") is True,
        "checkpoint-2 does not permit checkpoint-4 screening",
    )
    manifest_path = (root / MANIFEST).resolve()
    _require(sha256_file(manifest_path) == MANIFEST_SHA256, "manifest hash drift")
    try:
        manifest = probe._validate_manifest(manifest_path, MANIFEST_SHA256)
    except Exception as exc:
        raise FixedScreenError(f"selection manifest replay failed: {exc}") from exc
    _require(
        manifest.get("purpose") == probe.SELECTION_PURPOSE,
        "selection manifest purpose drift",
    )
    supplemental_value = _read_json(root / gap.ATTESTATION, label="supplemental")
    expected_candidate = supplemental_value.get("completed_run", {}).get(
        "candidate_checkpoint_fingerprints", {}
    ).get(str(step))
    candidate = (root / TRAINING_OUTPUT / f"checkpoint-{step}").resolve()
    observed_candidate = fingerprint_artifact_path(candidate)
    _require(observed_candidate == expected_candidate, "checkpoint-4 fingerprint drift")
    output = output_path(root, step)
    _require(not output.exists() and not output.is_symlink(), "screen output exists")
    return seal_manifest(
        {
            "schema_version": SCHEMA,
            "status": "authorized",
            "scope": {
                "operation": "checkpoint_4_static_generation_screening",
                "gpu": 1,
                "fresh_output": True,
                "not_authorized": [
                    "gpu0",
                    "checkpoint6",
                    "checkpoint_selection",
                    "merge",
                    "grpo",
                    "generation_parameter_change",
                    "overwrite",
                ],
            },
            "lineage": {
                "original_sft_authorization": original,
                "supplemental_post_run_attestation": supplemental,
                "checkpoint_2_advance_decision": cp2,
            },
            "candidate_checkpoint": observed_candidate,
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
            "model_label": f"correction-sft-checkpoint-{step}",
            "implementation": _implementation(root),
            "methodology": {
                "static_screen_not_bitwise_grpo_replay": True,
                "full_original_gate_required_for_selection": True,
                "authoritative_gate_after_merge": "fresh_two_step_cap1536_grpo_smoke",
            },
        }
    )


def authorize(repo_root: Path, step: int, *, execute: bool) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    value = authorization_payload(root, step)
    path = authorization_path(root, step)
    if not execute:
        return {**value, "status": "ready_to_authorize", "path": str(path)}
    _require(not path.exists() and not path.is_symlink(), "authorization exists")
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


def _verify_authorization(repo_root: Path, step: int) -> dict[str, Any]:
    path = authorization_path(repo_root, step)
    observed = _read_json(path, label="screen authorization")
    validate_manifest_integrity(observed)
    _require(observed == authorization_payload(repo_root, step), "authorization drift")
    return observed


@contextmanager
def _fixed_padding_postprocessor(repo_root: Path) -> Iterator[None]:
    original_decode = base_probe.decode_completion_preserving_boundary
    original_result = probe._result_row
    original_model_source = probe._model_source
    implementation = _implementation(repo_root)

    def fixed_decode(tokenizer: Any, token_ids: Sequence[int], eos_ids: Any) -> str:
        normalized_eos = set(base_probe._normalize_eos_ids(eos_ids))
        trimmed, _evidence = normalize_generated_ids(token_ids, normalized_eos)
        return original_decode(tokenizer, trimmed, normalized_eos)

    def fixed_result(**kwargs: Any) -> dict[str, Any]:
        eos_ids = set(kwargs["provenance"]["eos_token_ids"])
        trimmed, evidence = normalize_generated_ids(
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


def launch(repo_root: Path, step: int, *, execute: bool) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    authorization = _verify_authorization(root, step)
    output = output_path(root, step)
    command = {
        "status": "ready",
        "gpu": 1,
        "step": step,
        "output": str(output),
        "authorization": {
            "path": str(authorization_path(root, step)),
            "sha256": sha256_file(authorization_path(root, step)),
            "payload_sha256": authorization["integrity"]["payload_sha256"],
        },
    }
    if not execute:
        return command
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "launch requires GPU1")
    with _fixed_padding_postprocessor(root):
        result = probe.run_screen(
            manifest_path=(root / MANIFEST).resolve(),
            manifest_sha256=MANIFEST_SHA256,
            output_dir=output,
            model_label=f"correction-sft-checkpoint-{step}",
            base_model_path=probe.SELECTED_CP38_MODEL,
            adapter_path=(root / TRAINING_OUTPUT / f"checkpoint-{step}").resolve(),
            load_in_4bit=True,
        )
    return dict(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("authorize", "launch"):
        command = sub.add_parser(name)
        command.add_argument("--step", type=int, choices=SUPPORTED_STEPS, required=True)
        command.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "authorize":
            result = authorize(args.repo_root, args.step, execute=args.execute)
        else:
            result = launch(args.repo_root, args.step, execute=args.execute)
    except (FixedScreenError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "launch" and result.get("quality_status") == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
