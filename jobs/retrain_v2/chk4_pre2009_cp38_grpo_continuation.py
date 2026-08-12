"""GPU1-only chk4 GRPO continuation from the selected pre-2009 SFT cp38.

This module applies a versioned contract to the reusable v5 continuation
engine.  State changes are isolated to a temporary engine context, so the
historical v5 cp24 workflow retains its original hashes, seed, paths, and
callback.  A fresh two-step smoke must pass before the full run can be
authorized.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from jobs.retrain_v2 import chk4_v5_cp24_grpo_continuation as engine
from jobs.retrain_v2 import merge_chk4_selected_sft_checkpoint as selected


SOURCE_PROFILE = selected.PRE2009_CP38_PROFILE_NAME
SOURCE_BRANCH_ID = (
    "chk4_from_chk1_cp200_sft_grpo_pre2009_balanced_lr1e5_steps39_v1_20260811"
)
CHECKPOINT_STEP = 38
CHECKPOINT_SHA256 = "339671e06053f0dc80d4dfc6d006500e9a9fccba1945aa5ff770f05f755a2a4b"
ADAPTER_WEIGHTS_SHA256 = (
    "513cb7bc91136f9a9b5c7d83c93d8808cdd583dc0fb293ad3af452428354b09b"
)
PILOT_MANIFEST_SHA256 = (
    "99853a20f246684b0e03a90087af5dcb3ca23732bf5db06fd5433b09516611a7"
)
PILOT_SUMMARY_SHA256 = (
    "2b33278c9b96b27954e710c0e705b3dc929ade2ecf906d5296e572cc0ba75a0b"
)
PILOT_RESULTS_SHA256 = (
    "86b0085e27f21d3fb1ba3b88893309b89f8e2363148c5a576d20fe0178b590a9"
)
PILOT_LAUNCH_SHA256 = "6f193bd91a50c0f4eeaf57c3b1492d1e6ee108085eae33a9195e69107f14ed3f"
GRPO_RELEASE_SHA256 = "5141e00569b94e4af96f030f46176e9cd409199a6f26ac1c63a2ce335fe1e2d6"
GRPO_TRAIN_SHA256 = "e8c9e17892fb31a02f811c6aee222b1bd23486e60164c9426228444743eab046"
SMOKE_SOURCE_INDICES = (238, 133, 104, 87)
SMOKE_SOURCE_ROWS = (
    ("dec-bb7d9c61358a339a9a1f4aa5", "hold", 0),
    ("dec-4dfab939b0a910c949931475", "hike", 25),
    ("dec-29de02cb43945c20f838fcf7", "hold", 0),
    ("dec-8b8d55ea065b662a19cefe88", "cut", 50),
)
SMOKE_SEED = 31416
SMOKE_SCHEMA = "chk4-pre2009-cp38-grpo-smoke-authorization-v1"
FULL_SCHEMA = "chk4-pre2009-cp38-grpo-full-authorization-v1"
GATE_SCHEMA = "chk4-pre2009-cp38-grpo-smoke-gate-v1"
STEP_GATE_SCHEMA = "chk4-pre2009-cp38-grpo-smoke-step-gate-v1"

SOURCE_RUN_ROOT = Path(f"output/training/retrain_v2/{SOURCE_BRANCH_ID}")
PILOT_MANIFEST = (
    SOURCE_RUN_ROOT
    / "selected_sft_checkpoints/pre2009_train_stratified_manifest_v1.json"
)
PILOT_SUMMARY = (
    SOURCE_RUN_ROOT
    / "selected_sft_checkpoints/pre2009_cp38_train_stratified_probe_v1/summary.json"
)
SELECTED_ROOT = SOURCE_RUN_ROOT / "selected_sft_checkpoints/checkpoint-38"
SELECTED_MODEL = SELECTED_ROOT / "merged/chk4_sft"
SELECTION_RECEIPT = SELECTED_ROOT / "receipts/selection_receipt.json"

SMOKE = engine.Phase(
    name="smoke",
    branch_id="chk4_from_pre2009_cp38_selected_grpo_smoke_v1_20260811",
    config=Path(
        "configs/retrain_v2/"
        "chk4_decision_grpo_from_pre2009_cp38_selected_smoke_v1_20260811.yaml"
    ),
    config_sha256="50573b8fe446dc19121bc9e1f6941b5ac073e3fcbad49574cd4a824140d64bf5",
    authorization_schema=SMOKE_SCHEMA,
    max_steps=2,
)
FULL = engine.Phase(
    name="full",
    branch_id="chk4_from_pre2009_cp38_selected_grpo_full_v1_20260811",
    config=Path(
        "configs/retrain_v2/"
        "chk4_decision_grpo_from_pre2009_cp38_selected_full_v1_20260811.yaml"
    ),
    config_sha256="f5a773d44e81990cf1843c96bf6c9572269868ff7b64436117c30ca1961211d8",
    authorization_schema=FULL_SCHEMA,
    max_steps=-1,
)

ContinuationError = engine.ContinuationError


def _engine_updates() -> dict[str, Any]:
    return {
        "SOURCE_PROFILE": SOURCE_PROFILE,
        "SOURCE_BRANCH_ID": SOURCE_BRANCH_ID,
        "CHECKPOINT_STEP": CHECKPOINT_STEP,
        "CHECKPOINT_SHA256": CHECKPOINT_SHA256,
        "PILOT_MANIFEST_SHA256": PILOT_MANIFEST_SHA256,
        "PILOT_SUMMARY_SHA256": PILOT_SUMMARY_SHA256,
        "PILOT_RESULTS_SHA256": PILOT_RESULTS_SHA256,
        "GRPO_RELEASE_SHA256": GRPO_RELEASE_SHA256,
        "GRPO_TRAIN_SHA256": GRPO_TRAIN_SHA256,
        "SMOKE_SOURCE_INDICES": SMOKE_SOURCE_INDICES,
        "SMOKE_SOURCE_ROWS": SMOKE_SOURCE_ROWS,
        "SMOKE_SCHEMA": SMOKE_SCHEMA,
        "FULL_SCHEMA": FULL_SCHEMA,
        "GATE_SCHEMA": GATE_SCHEMA,
        "STEP_GATE_SCHEMA": STEP_GATE_SCHEMA,
        "SOURCE_RUN_ROOT": SOURCE_RUN_ROOT,
        "PILOT_MANIFEST": PILOT_MANIFEST,
        "PILOT_SUMMARY": PILOT_SUMMARY,
        "SELECTED_ROOT": SELECTED_ROOT,
        "SELECTED_MODEL": SELECTED_MODEL,
        "SELECTION_RECEIPT": SELECTION_RECEIPT,
        "SMOKE": SMOKE,
        "FULL": FULL,
        "SMOKE_SEED": SMOKE_SEED,
        "SMOKE_DATA_SEED": SMOKE_SEED,
        "SMOKE_SHUFFLE_DATASET": True,
        "SMOKE_TRAIN_ROWS": 312,
        "SMOKE_CALLBACK_NAME": "chk4_pre2009_cp38_smoke_gate",
        "GRPO_DATASET_ROLE": "decision_grpo_pre2009_balanced",
        "RUNTIME_PROFILE_PREFIX": "pre2009_cp38_selected",
        "CONTINUATION_SCHEMA_PREFIX": "chk4-pre2009-cp38-grpo",
        "FORBIDDEN_CANONICAL_SCOPE": "historical_pre2009_canonical_grpo_output",
        "ADDITIONAL_FORBIDDEN_SCOPES": (
            "source_checkpoint_39",
            "source_root_adapter_checkpoint_39",
        ),
        "CONTINUATION_IMPLEMENTATION_FILES": (
            "jobs/retrain_v2/chk4_pre2009_cp38_grpo_continuation.py",
            "jobs/retrain_v2/merge_chk4_selected_sft_checkpoint.py",
        ),
    }


@contextmanager
def _engine_contract() -> Iterator[None]:
    updates = _engine_updates()
    previous = {name: getattr(engine, name) for name in updates}
    previous_profile = selected.ACTIVE_SELECTION_PROFILE.name
    try:
        for name, value in updates.items():
            setattr(engine, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(engine, name, value)
        selected._activate_selection_profile(previous_profile)


def _call(name: str, *args: Any, **kwargs: Any) -> Any:
    with _engine_contract():
        return getattr(engine, name)(*args, **kwargs)


def _selection_kwargs(repo_root: Path) -> dict[str, Any]:
    return {
        "repo_root": repo_root,
        "checkpoint_step": CHECKPOINT_STEP,
        "pilot_manifest": repo_root / PILOT_MANIFEST,
        "pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
        "pilot_summary": repo_root / PILOT_SUMMARY,
        "pilot_summary_sha256": PILOT_SUMMARY_SHA256,
    }


def _validate_config(repo_root: Path, phase: engine.Phase) -> dict[str, Any]:
    return _call("_validate_config", repo_root, phase)


def _smoke_sampling_contract(
    repo_root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    return _call("_smoke_sampling_contract", repo_root, config)


def _published_selection(repo_root: Path):
    plan, receipt = _call("_published_selection", repo_root)
    if plan.checkpoint_fingerprint.get("sha256") != CHECKPOINT_SHA256:
        raise ContinuationError("selected checkpoint-38 fingerprint drift")
    if plan.pilot_results.get("sha256") != PILOT_RESULTS_SHA256:
        raise ContinuationError("selected cp38 pilot results hash drift")
    if not plan.pilot_launch or plan.pilot_launch.get("sha256") != PILOT_LAUNCH_SHA256:
        raise ContinuationError("selected cp38 pilot launch hash drift")
    if plan.checkpoint.joinpath("adapter_model.safetensors").is_symlink():
        raise ContinuationError("selected cp38 adapter weights are unsafe")
    from open_r1.provenance import sha256_file

    if (
        sha256_file(plan.checkpoint / "adapter_model.safetensors")
        != ADAPTER_WEIGHTS_SHA256
    ):
        raise ContinuationError("selected cp38 adapter weights hash drift")
    return plan, receipt


def preflight(repo_root: Path) -> dict[str, Any]:
    result = _call("preflight", repo_root)
    result["source_adapter_weights_sha256"] = ADAPTER_WEIGHTS_SHA256
    result["pilot_launch_sha256"] = PILOT_LAUNCH_SHA256
    return result


def status(repo_root: Path) -> dict[str, Any]:
    return _call("status", repo_root)


def authorize(repo_root: Path, phase: engine.Phase, *, execute: bool) -> dict[str, Any]:
    return _call("authorize", repo_root, phase, execute=execute)


def training_command(repo_root: Path, phase: engine.Phase) -> dict[str, Any]:
    return _call("training_command", repo_root, phase)


def execute_training(repo_root: Path, phase: engine.Phase) -> None:
    # os.execvpe replaces this process, so the context intentionally never
    # restores on a successful launch.
    with _engine_contract():
        engine.execute_training(repo_root, phase)


def gate_smoke(repo_root: Path, *, execute: bool) -> dict[str, Any]:
    return _call("gate_smoke", repo_root, execute=execute)


def _authorization_payload(repo_root: Path, phase: engine.Phase) -> dict[str, Any]:
    return _call("_authorization_payload", repo_root, phase)


def _verify_authorization(repo_root: Path, phase: engine.Phase) -> dict[str, Any]:
    return _call("_verify_authorization", repo_root, phase)


def _verify_smoke_gate(repo_root: Path) -> dict[str, Any]:
    return _call("_verify_smoke_gate", repo_root)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "status"):
        subparsers.add_parser(command)
    for command in (
        "authorize-smoke",
        "launch-smoke",
        "gate-smoke",
        "authorize-full",
        "launch-full",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    try:
        if args.command == "preflight":
            result = preflight(repo_root)
        elif args.command == "status":
            result = status(repo_root)
        elif args.command == "authorize-smoke":
            result = authorize(repo_root, SMOKE, execute=args.execute)
        elif args.command == "authorize-full":
            result = authorize(repo_root, FULL, execute=args.execute)
        elif args.command == "gate-smoke":
            result = gate_smoke(repo_root, execute=args.execute)
        elif args.command in {"launch-smoke", "launch-full"}:
            phase = SMOKE if args.command == "launch-smoke" else FULL
            if args.execute:
                execute_training(repo_root, phase)
                raise AssertionError("os.execvpe unexpectedly returned")
            result = training_command(repo_root, phase)
        else:  # pragma: no cover
            raise ContinuationError(f"unsupported command: {args.command}")
    except (
        ContinuationError,
        selected.SelectionMergeError,
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
