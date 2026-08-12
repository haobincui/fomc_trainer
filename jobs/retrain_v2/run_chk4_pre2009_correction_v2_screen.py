"""Create-only authorization and GPU1 launch for correction-v2 screens.

The underlying probe deliberately exposes no direct generation CLI.  Every
baseline/candidate run must pass through this state machine so blind and
retention panels cannot be observed before checkpoint selection is locked.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.retrain_v2 import audit_chk4_pre2009_correction_v2 as auditor
from jobs.retrain_v2 import probe_chk4_pre2009_correction_v2 as probe
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = probe.RUN_ROOT
SFT_OUTPUT = probe.SFT_OUTPUT
SCREEN_ROOT = probe.SCREEN_ROOT
AUTH_SCHEMA = "chk4-pre2009-correction-v2-static-screen-authorization-v1"


class CorrectionV2ScreenLaunchError(RuntimeError):
    """A correction-v2 generation run is not authorized in the current state."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorrectionV2ScreenLaunchError(message)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorrectionV2ScreenLaunchError(f"invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _descriptor(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing artifact: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def output_path(stage: str, role: str, checkpoint_step: int | None) -> Path:
    _require(stage in probe.STATIC_STAGES, "invalid screen stage")
    _require(role in {"baseline", "candidate"}, "invalid model role")
    if role == "baseline":
        _require(checkpoint_step is None, "baseline cannot declare checkpoint")
        name = "baseline-exact-cp38"
    else:
        _require(checkpoint_step in probe.CANDIDATE_ORDER, "candidate step invalid")
        name = f"checkpoint-{checkpoint_step}"
    return SCREEN_ROOT / stage / name


def authorization_path(stage: str, role: str, checkpoint_step: int | None) -> Path:
    suffix = "baseline" if role == "baseline" else f"checkpoint-{checkpoint_step}"
    return SCREEN_ROOT / "authorizations" / f"{stage}-{suffix}.json"


def _verify_bound_implementation(receipt: Mapping[str, Any]) -> None:
    implementation = receipt.get("implementation")
    _require(
        isinstance(implementation, Mapping)
        and set(implementation) == set(probe.AUTH_IMPLEMENTATION_RELATIVES),
        "receipt implementation inventory drift",
    )
    for relative, descriptor in implementation.items():
        _require(isinstance(descriptor, Mapping), "invalid implementation descriptor")
        path = Path(str(descriptor.get("path")))
        _require(
            path == (REPO_ROOT / relative).resolve()
            and path.is_file()
            and not path.is_symlink()
            and sha256_file(path) == descriptor.get("sha256"),
            f"receipt implementation drift: {relative}",
        )


def _sealed_receipt(
    path: Path,
    *,
    schema: str,
    statuses: set[str],
    manifest_path: Path,
    manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = _read_json(path, label="predecessor receipt")
    _require(path.stat().st_mode & 0o777 == 0o400, "predecessor receipt mode drift")
    validate_manifest_integrity(value)
    _require(
        value.get("schema_version") == schema and value.get("status") in statuses,
        "predecessor receipt state drift",
    )
    _require(
        value.get("manifest", {}).get("sha256") == manifest_sha256,
        "predecessor belongs to another suite manifest",
    )
    _verify_bound_implementation(value)
    if schema == auditor.SELECTION_RECEIPT_SCHEMA:
        auditor.verify_selection_receipt(
            path=path,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
        )
    else:
        auditor.verify_confirmation_receipt(
            path=path,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
        )
    return value, {
        **_descriptor(path),
        "payload_sha256": value["integrity"]["payload_sha256"],
        "status": value["status"],
    }


def _selection_predecessor(
    *,
    stage: str,
    role: str,
    checkpoint_step: int | None,
    path: Path | None,
    manifest_path: Path,
    manifest_sha256: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if stage == probe.SELECTION_STAGE and role == "baseline":
        _require(path is None, "selection baseline cannot consume a decision")
        return None, None
    if stage == probe.SELECTION_STAGE and checkpoint_step == 2:
        _require(path is None, "checkpoint-2 cannot consume a selection decision")
        return None, None
    _require(path is not None, "stage requires selection receipt")
    statuses = (
        {"selection_in_progress"}
        if stage == probe.SELECTION_STAGE
        else {"selection_passed"}
    )
    value, binding = _sealed_receipt(
        path.resolve(),
        schema=auditor.SELECTION_RECEIPT_SCHEMA,
        statuses=statuses,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    if stage == probe.SELECTION_STAGE:
        _require(
            value.get("selected_checkpoint") is None
            and value.get("next_checkpoint_allowed") == checkpoint_step,
            "selection receipt does not authorize this next checkpoint",
        )
    else:
        _require(
            value.get("selected_checkpoint") == checkpoint_step
            if role == "candidate"
            else value.get("selected_checkpoint") in probe.CANDIDATE_ORDER,
            "locked checkpoint drift",
        )
    return value, binding


def _blind_predecessor(
    *,
    stage: str,
    path: Path | None,
    selection_path: Path | None,
    manifest_path: Path,
    manifest_sha256: str,
    locked_step: int | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if stage != probe.RETENTION_STAGE:
        _require(path is None, "blind receipt is only valid for retention")
        return None, None
    _require(
        path is not None and selection_path is not None,
        "retention predecessors missing",
    )
    value, binding = _sealed_receipt(
        path.resolve(),
        schema=auditor.CONFIRMATION_RECEIPT_SCHEMA,
        statuses={"confirmation_passed"},
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    _require(
        value.get("stage") == probe.BLIND_STAGE
        and value.get("locked_checkpoint") == locked_step
        and value.get("selection_receipt", {}).get("path")
        == str(selection_path.resolve())
        and value.get("selection_receipt", {}).get("sha256")
        == sha256_file(selection_path.resolve()),
        "blind receipt does not bind exact locked selection",
    )
    return value, binding


def _implementation(repo_root: Path) -> dict[str, Any]:
    return {
        relative: _descriptor(repo_root / relative)
        for relative in probe.AUTH_IMPLEMENTATION_RELATIVES
    }


def authorization_payload(
    *,
    repo_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    stage: str,
    role: str,
    checkpoint_step: int | None,
    selection_receipt_path: Path | None,
    blind_receipt_path: Path | None,
) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    _require(root == REPO_ROOT, "repo root drift")
    manifest = probe.validate_manifest(manifest_path.resolve(), manifest_sha256)
    _require(stage in probe.STATIC_STAGES, "invalid static stage")
    selection, selection_binding = _selection_predecessor(
        stage=stage,
        role=role,
        checkpoint_step=checkpoint_step,
        path=selection_receipt_path,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    locked_step = (
        int(selection["selected_checkpoint"])
        if selection is not None and selection.get("selected_checkpoint") is not None
        else checkpoint_step
    )
    _, blind_binding = _blind_predecessor(
        stage=stage,
        path=blind_receipt_path,
        selection_path=selection_receipt_path,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        locked_step=locked_step,
    )
    output = output_path(stage, role, checkpoint_step)
    _require(not output.exists() and not output.is_symlink(), "screen output exists")

    if role == "baseline":
        source = probe._model_source(
            role="baseline",
            model_path=probe.SELECTED_CP38_MODEL,
            base_model_path=None,
            adapter_path=None,
        )
    else:
        _require(checkpoint_step in probe.CANDIDATE_ORDER, "candidate step missing")
        adapter = SFT_OUTPUT / f"checkpoint-{checkpoint_step}"
        source = probe._model_source(
            role="candidate",
            model_path=None,
            base_model_path=probe.SELECTED_CP38_MODEL,
            adapter_path=adapter,
        )
        # Every candidate run requires the corresponding exact-cp38 baseline
        # for the same stage to be complete and independently replayable.
        auditor.replay_output(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            stage=stage,
            role="baseline",
            output_dir=output_path(stage, "baseline", None),
        )

    return seal_manifest(
        {
            "schema_version": AUTH_SCHEMA,
            "status": "authorized",
            "scope": {
                "operation": "correction_v2_static_generation_screen",
                "stage": stage,
                "model_role": role,
                "checkpoint_step": checkpoint_step,
                "gpu": 1,
                "fresh_output": True,
                "not_authorized": [
                    "gpu0",
                    "overwrite",
                    "seed_change",
                    "gate_change",
                    "unregistered_checkpoint",
                    "merge",
                    "grpo",
                ],
            },
            "manifest": {
                **_descriptor(manifest_path.resolve()),
                "sha256": manifest_sha256,
            },
            "stage_contract": manifest["stages"][stage],
            "model_source": source["provenance"],
            "model_sha256": source["model_fingerprint"]["sha256"],
            "effective_model_sha256": source["effective_model_fingerprint"]["sha256"],
            "output": str(output.resolve()),
            "predecessors": {
                "selection": selection_binding,
                "blind": blind_binding,
            },
            "implementation": _implementation(root),
        }
    )


def authorize(path: Path, value: Mapping[str, Any], *, execute: bool) -> dict[str, Any]:
    if not execute:
        return {**value, "dry_run_status": "ready_to_authorize", "path": str(path)}
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
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "payload_sha256": value["integrity"]["payload_sha256"],
    }


def launch(
    *,
    repo_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    stage: str,
    role: str,
    checkpoint_step: int | None,
    selection_receipt_path: Path | None,
    blind_receipt_path: Path | None,
    execute: bool,
) -> dict[str, Any]:
    auth_path = authorization_path(stage, role, checkpoint_step)
    observed = _read_json(auth_path, label="screen authorization")
    validate_manifest_integrity(observed)
    expected = authorization_payload(
        repo_root=repo_root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        stage=stage,
        role=role,
        checkpoint_step=checkpoint_step,
        selection_receipt_path=selection_receipt_path,
        blind_receipt_path=blind_receipt_path,
    )
    _require(observed == expected, "screen authorization drift")
    command = {
        "status": "ready_to_launch",
        "authorization": _descriptor(auth_path),
        "output": str(output_path(stage, role, checkpoint_step)),
        "gpu": 1,
    }
    if not execute:
        return command
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "1", "launch requires GPU1")
    authorization_descriptor = _descriptor(auth_path)
    # ``run_panel`` and the independent replay auditor intentionally accept
    # only this minimal three-field binding.  The general file descriptor also
    # contains ``bytes`` and must not be forwarded as part of the schema.
    authorization_binding = {
        "path": authorization_descriptor["path"],
        "sha256": authorization_descriptor["sha256"],
        "payload_sha256": observed["integrity"]["payload_sha256"],
    }
    if role == "baseline":
        return probe.run_panel(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            stage=stage,
            role=role,
            output_dir=output_path(stage, role, checkpoint_step),
            model_label=f"exact-cp38-{stage}-baseline",
            model_path=probe.SELECTED_CP38_MODEL,
            load_in_4bit=True,
            authorization_binding=authorization_binding,
        )
    return probe.run_panel(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        stage=stage,
        role=role,
        output_dir=output_path(stage, role, checkpoint_step),
        model_label=f"correction-v2-checkpoint-{checkpoint_step}-{stage}",
        base_model_path=probe.SELECTED_CP38_MODEL,
        adapter_path=SFT_OUTPUT / f"checkpoint-{checkpoint_step}",
        load_in_4bit=True,
        authorization_binding=authorization_binding,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("authorize", "launch"))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--stage", choices=probe.STATIC_STAGES, required=True)
    parser.add_argument("--role", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--checkpoint-step", type=int, choices=probe.CANDIDATE_ORDER)
    parser.add_argument("--selection-receipt", type=Path)
    parser.add_argument("--blind-receipt", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "authorize":
            value = authorization_payload(
                repo_root=args.repo_root,
                manifest_path=args.manifest,
                manifest_sha256=args.manifest_sha256,
                stage=args.stage,
                role=args.role,
                checkpoint_step=args.checkpoint_step,
                selection_receipt_path=args.selection_receipt,
                blind_receipt_path=args.blind_receipt,
            )
            result = authorize(
                authorization_path(args.stage, args.role, args.checkpoint_step),
                value,
                execute=bool(args.execute),
            )
        else:
            result = launch(
                repo_root=args.repo_root,
                manifest_path=args.manifest,
                manifest_sha256=args.manifest_sha256,
                stage=args.stage,
                role=args.role,
                checkpoint_step=args.checkpoint_step,
                selection_receipt_path=args.selection_receipt,
                blind_receipt_path=args.blind_receipt,
                execute=bool(args.execute),
            )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (CorrectionV2ScreenLaunchError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
