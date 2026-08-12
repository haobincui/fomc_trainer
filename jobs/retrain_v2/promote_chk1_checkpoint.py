"""Safely promote one selected chk1 checkpoint into a merged candidate model.

This entry point is intentionally independent of the canonical retrain-v2 DAG.
It validates immutable checkpoint-selection evidence and an explicit chk2-only
authorization receipt, then delegates the non-destructive CPU merge to
``jobs.retrain_v2.merge_adapter.merge_from_config``.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from jobs.retrain_v2.merge_adapter import _rename_noreplace, merge_from_config
from jobs.retrain_v2.merge_attestation import verify_merge_attestation
from open_r1.provenance import (
    fingerprint_artifact_path,
    sha256_file,
    sha256_text,
    validate_sha256,
)


SCHEMA_VERSION = 1
MERGE_POLICY = {
    "executor": "jobs.retrain_v2.merge_adapter.merge_from_config",
    "model_dtype": "bfloat16",
    "device": "cpu",
    "source_adapter_copy": True,
    "publication": "renameat2(RENAME_NOREPLACE)",
    "overwrite": False,
}
CONFIG_KEYS = {
    "promotion_schema_version",
    "promotion_id",
    "stage",
    "checkpoint_step",
    "model_name_or_path",
    "output_dir",
    "peft_merged_model_path",
    "expected_base_model_sha256",
    "expected_source_adapter_sha256",
    "expected_adapter_model_sha256",
    "expected_adapter_config_sha256",
    "selection_evidence",
    "authorization_receipt",
    "promotion_receipt",
    "downstream_stages_allowed",
    "canonical_dag_integration",
    "merge_policy",
}


class CheckpointPromotionError(ValueError):
    """Raised when promotion inputs drift from the immutable contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointPromotionError(message)


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointPromotionError(
            "Promotion receipt is not canonical JSON"
        ) from exc


def _resolve_repo_path(root: Path, value: object, *, label: str) -> Path:
    text = str(value or "").strip()
    _require(bool(text), f"{label} is missing")
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise CheckpointPromotionError(
            f"{label} must remain inside the repository: {path}"
        ) from exc
    return path


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _portable_fingerprint(root: Path, path: Path) -> dict[str, Any]:
    fingerprint = fingerprint_artifact_path(path)
    fingerprint["path"] = _relative(root, path)
    return fingerprint


def _load_config(config_path: Path) -> dict[str, Any]:
    _require(not config_path.is_symlink(), "Promotion config must not be a symlink")
    _require(config_path.is_file(), f"Promotion config is missing: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CheckpointPromotionError("Unable to parse promotion config") from exc
    _require(isinstance(payload, dict), "Promotion config must contain a mapping")
    _require(set(payload) == CONFIG_KEYS, "Promotion config schema mismatch")
    _require(
        payload["promotion_schema_version"] == SCHEMA_VERSION,
        "Unsupported promotion config schema",
    )
    _require(
        isinstance(payload["promotion_id"], str)
        and bool(payload["promotion_id"].strip()),
        "promotion_id must be a non-empty string",
    )
    _require(payload["stage"] == "chk1", "Promotion stage must be chk1")
    step = payload["checkpoint_step"]
    _require(
        isinstance(step, int) and not isinstance(step, bool) and step > 0,
        "checkpoint_step must be a positive integer",
    )
    _require(
        payload["downstream_stages_allowed"] == ["chk2"],
        "Promotion scope must allow chk2 only",
    )
    _require(
        payload["canonical_dag_integration"] is False,
        "Candidate promotion must remain outside the canonical DAG",
    )
    _require(payload["merge_policy"] == MERGE_POLICY, "Merge policy drift detected")
    for key in (
        "expected_base_model_sha256",
        "expected_source_adapter_sha256",
        "expected_adapter_model_sha256",
        "expected_adapter_config_sha256",
    ):
        payload[key] = validate_sha256(payload[key], label=key)
    return payload


def _reject_symlinks(root: Path, *, label: str) -> None:
    _require(not root.is_symlink(), f"{label} root must not be a symlink")
    try:
        symlink = next((path for path in root.rglob("*") if path.is_symlink()), None)
    except OSError as exc:
        raise CheckpointPromotionError(f"Unable to inspect {label}") from exc
    _require(symlink is None, f"{label} contains a symlink: {symlink}")


def _selection_evidence(
    root: Path, value: object
) -> list[dict[str, Any]]:
    _require(isinstance(value, list) and bool(value), "selection_evidence is empty")
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for index, item in enumerate(value):
        label = f"selection_evidence[{index}]"
        _require(
            isinstance(item, dict) and set(item) == {"path", "sha256"},
            f"{label} schema mismatch",
        )
        path = _resolve_repo_path(root, item["path"], label=f"{label}.path")
        _require(path not in seen, f"Duplicate selection evidence path: {path}")
        seen.add(path)
        _require(not path.is_symlink() and path.is_file(), f"{label} is missing")
        expected = validate_sha256(item["sha256"], label=f"{label}.sha256")
        actual = sha256_file(path)
        _require(actual == expected, f"{label} SHA-256 mismatch")
        records.append(
            {
                "path": _relative(root, path),
                "sha256": actual,
                "total_bytes": path.stat().st_size,
            }
        )
    return records


def _authorization(
    root: Path, path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(not path.is_symlink() and path.is_file(), "Authorization receipt is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointPromotionError("Unable to parse authorization receipt") from exc
    _require(isinstance(payload, dict), "Authorization receipt must contain an object")
    _require(
        payload.get("downstream_stages_allowed") == ["chk2"],
        "Authorization receipt must allow chk2 only",
    )
    _require(
        payload.get("canonical_dag_integration") is False,
        "Authorization receipt must reject canonical DAG integration",
    )
    _require(
        "chk3" not in payload["downstream_stages_allowed"]
        and "chk4" not in payload["downstream_stages_allowed"],
        "Authorization receipt must not authorize chk3 or chk4",
    )
    return payload, {
        "path": _relative(root, path),
        "sha256": sha256_file(path),
        "total_bytes": path.stat().st_size,
    }


def preflight_promotion(
    config_path: str | Path, *, repo_root: str | Path
) -> dict[str, Any]:
    """Validate and bind every immutable input without creating merge output."""

    root = Path(repo_root).resolve()
    _require(root.is_dir(), f"Repository root is missing: {root}")
    config = _resolve_repo_path(root, config_path, label="config")
    payload = _load_config(config)

    base = _resolve_repo_path(root, payload["model_name_or_path"], label="base model")
    adapter = _resolve_repo_path(root, payload["output_dir"], label="source adapter")
    destination = _resolve_repo_path(
        root, payload["peft_merged_model_path"], label="merged destination"
    )
    authorization_path = _resolve_repo_path(
        root, payload["authorization_receipt"], label="authorization receipt"
    )
    receipt_path = _resolve_repo_path(
        root, payload["promotion_receipt"], label="promotion receipt"
    )
    _require(base.is_dir(), f"Base model is missing: {base}")
    _require(adapter.is_dir(), f"Source adapter is missing: {adapter}")
    _require(not destination.exists(), f"Merged destination already exists: {destination}")
    _require(not receipt_path.exists(), f"Promotion receipt already exists: {receipt_path}")
    _require(
        len({base, adapter, destination, authorization_path, receipt_path}) == 5,
        "Promotion paths must be distinct",
    )
    _reject_symlinks(base, label="Base model")
    _reject_symlinks(adapter, label="Source adapter")

    step = payload["checkpoint_step"]
    _require(adapter.name == f"checkpoint-{step}", "Adapter checkpoint step mismatch")
    trainer_state = adapter / "trainer_state.json"
    _require(trainer_state.is_file(), "Source checkpoint trainer_state.json is missing")
    try:
        trainer_payload = json.loads(trainer_state.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointPromotionError("Unable to parse checkpoint trainer state") from exc
    _require(
        isinstance(trainer_payload, dict)
        and trainer_payload.get("global_step") == step,
        "Source checkpoint global_step mismatch",
    )

    base_fingerprint = _portable_fingerprint(root, base)
    adapter_fingerprint = _portable_fingerprint(root, adapter)
    _require(
        base_fingerprint["sha256"] == payload["expected_base_model_sha256"],
        "Base model SHA-256 mismatch",
    )
    _require(
        adapter_fingerprint["sha256"] == payload["expected_source_adapter_sha256"],
        "Source adapter SHA-256 mismatch",
    )
    adapter_model = adapter / "adapter_model.safetensors"
    adapter_config = adapter / "adapter_config.json"
    _require(adapter_model.is_file(), "adapter_model.safetensors is missing")
    _require(adapter_config.is_file(), "adapter_config.json is missing")
    _require(
        sha256_file(adapter_model) == payload["expected_adapter_model_sha256"],
        "Adapter model SHA-256 mismatch",
    )
    _require(
        sha256_file(adapter_config) == payload["expected_adapter_config_sha256"],
        "Adapter config SHA-256 mismatch",
    )

    evidence = _selection_evidence(root, payload["selection_evidence"])
    _, authorization_fingerprint = _authorization(root, authorization_path)
    config_fingerprint = _portable_fingerprint(root, config)
    binding = {
        "schema_version": SCHEMA_VERSION,
        "promotion_type": "selected_chk1_checkpoint_candidate",
        "promotion_id": payload["promotion_id"],
        "scope": {
            "stage": "chk1",
            "downstream_stages_allowed": ["chk2"],
            "canonical_dag_integration": False,
        },
        "checkpoint": {
            "global_step": step,
            "source_adapter": adapter_fingerprint,
            "adapter_model_sha256": payload["expected_adapter_model_sha256"],
            "adapter_config_sha256": payload["expected_adapter_config_sha256"],
        },
        "base_model": base_fingerprint,
        "merged_destination": _relative(root, destination),
        "merge_config": config_fingerprint,
        "selection_evidence": evidence,
        "authorization_receipt": authorization_fingerprint,
        "merge_policy": dict(MERGE_POLICY),
    }
    return {
        "root": root,
        "config": config,
        "payload": payload,
        "base": base,
        "adapter": adapter,
        "destination": destination,
        "authorization_path": authorization_path,
        "receipt_path": receipt_path,
        "binding": binding,
        "source_adapter_before": adapter_fingerprint,
    }


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"Promotion receipt already exists: {path}")
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _rename_noreplace(temporary, path)
        parent_descriptor = os.open(
            path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def promote_checkpoint(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    merge_executor: Callable[[Path, Path, Path], Mapping[str, Any] | None]
    | None = None,
) -> dict[str, Any]:
    """Merge, attest, verify, and record the selected checkpoint exactly once."""

    context = preflight_promotion(config_path, repo_root=repo_root)
    merge_kwargs: dict[str, Any] = {
        "merge_attestation_binding": context["binding"],
        "repo_root": context["root"],
    }
    if merge_executor is not None:
        merge_kwargs["merge_executor"] = merge_executor
    merge_result = merge_from_config(context["config"], **merge_kwargs)

    source_after = _portable_fingerprint(context["root"], context["adapter"])
    _require(
        source_after == context["source_adapter_before"],
        "Source adapter fingerprint changed during promotion",
    )
    attestation = verify_merge_attestation(
        context["destination"], expected_binding=context["binding"]
    )
    merged_fingerprint = _portable_fingerprint(
        context["root"], context["destination"]
    )
    receipt_without_sha = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": "chk1_checkpoint_promotion_merge",
        "status": "merged_and_verified",
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "promotion_id": context["payload"]["promotion_id"],
        "binding": context["binding"],
        "source_adapter_before": context["source_adapter_before"],
        "source_adapter_after": source_after,
        "source_adapter_unchanged": True,
        "merged_artifact": merged_fingerprint,
        "merge_attestation": {
            "canonical_payload_sha256": attestation["canonical_payload_sha256"],
            "semantic_evidence": attestation["semantic_evidence"],
        },
        "merge_result_status": merge_result["status"],
    }
    receipt = {
        **receipt_without_sha,
        "canonical_payload_sha256": sha256_text(_canonical_json(receipt_without_sha)),
    }
    _write_receipt(context["receipt_path"], receipt)
    return {
        "status": "merged_and_verified",
        "promotion_id": context["payload"]["promotion_id"],
        "merged_artifact": merged_fingerprint,
        "promotion_receipt": {
            "path": _relative(context["root"], context["receipt_path"]),
            "sha256": sha256_file(context["receipt_path"]),
        },
        "source_adapter_unchanged": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate and fingerprint immutable inputs without running the merge.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.preflight_only:
            context = preflight_promotion(args.config, repo_root=args.repo_root)
            result = {
                "status": "preflight_passed",
                "promotion_id": context["payload"]["promotion_id"],
                "binding": context["binding"],
                "merged_destination": _relative(
                    context["root"], context["destination"]
                ),
                "promotion_receipt": _relative(
                    context["root"], context["receipt_path"]
                ),
            }
        else:
            result = promote_checkpoint(args.config, repo_root=args.repo_root)
    except (
        CheckpointPromotionError,
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CheckpointPromotionError",
    "MERGE_POLICY",
    "preflight_promotion",
    "promote_checkpoint",
]
