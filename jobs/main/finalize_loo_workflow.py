"""Seal the provenance graph for a complete D-1 canonical LOO workflow."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


SCHEMA_VERSION = "canonical-loo-workflow-release-v1"
REQUIRED_ARTIFACTS = {
    "pilot": {
        "source_registry",
        "snapshot_manifest",
        "pilot_ledger_manifest",
        "pilot_release_manifest",
    },
    "formal": {
        "source_registry",
        "snapshot_manifest",
        "pilot_release_manifest",
        "formal_ledger_manifest",
        "formal_release_manifest",
    },
    "all": {
        "source_registry",
        "snapshot_manifest",
        "pilot_ledger_manifest",
        "pilot_release_manifest",
        "formal_ledger_manifest",
        "formal_release_manifest",
    },
}
ARTIFACT_SCHEMAS = {
    "source_registry": "loo-indicator-source-registry-v1",
    "snapshot_manifest": "loo-source-snapshot-manifest-v1",
    "pilot_ledger_manifest": "loo-indicator-ledger-manifest-v1",
    "formal_ledger_manifest": "loo-indicator-ledger-manifest-v1",
    "pilot_release_manifest": "canonical-loo-generation-release-v1",
    "formal_release_manifest": "canonical-loo-generation-release-v1",
}
POPULATION_CONTRACTS = {
    "pilot_ledger_manifest": ("pilot", "pilot_eval_13"),
    "formal_ledger_manifest": ("formal", "formal_test_13"),
    "pilot_release_manifest": ("pilot", "pilot_eval_13"),
    "formal_release_manifest": ("formal", "formal_test_13"),
}


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("Both NAME and PATH are required")
    return name, Path(raw_path)


def _read_json_artifact(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Workflow artifact {name!r} is not a regular file: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Workflow artifact {name!r} is not valid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Workflow artifact {name!r} must be a JSON object")
    expected_schema = ARTIFACT_SCHEMAS[name]
    if payload.get("schema_version") != expected_schema:
        raise ValueError(
            f"Workflow artifact {name!r} has schema "
            f"{payload.get('schema_version')!r}; expected {expected_schema!r}"
        )
    if name != "source_registry":
        if payload.get("status") != "complete":
            raise ValueError(f"Workflow artifact {name!r} must have status='complete'")
        validate_manifest_integrity(payload)
    return payload


def _validate_artifact_contracts(
    *,
    mode: str,
    named: Mapping[str, Path],
) -> None:
    payloads = {
        name: _read_json_artifact(path, name=name) for name, path in named.items()
    }
    registry = payloads["source_registry"]
    policy = registry.get("policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("network_mode") != "keyless"
        or policy.get("access_interface") != "alfred-graph-csv-v1"
    ):
        raise ValueError(
            "Source registry must declare keyless alfred-graph-csv-v1 access"
        )

    registry_sha256 = sha256_file(named["source_registry"])
    snapshot = payloads["snapshot_manifest"]
    registry_binding = snapshot.get("registry")
    if (
        snapshot.get("source_interface") != "alfred-graph-csv-v1"
        or not isinstance(registry_binding, Mapping)
        or registry_binding.get("sha256") != registry_sha256
    ):
        raise ValueError(
            "Snapshot manifest does not bind the active keyless source registry"
        )
    expected_vintage_count = 26 if mode == "all" else 13
    meetings = snapshot.get("meetings")
    if (
        snapshot.get("vintage_count") != expected_vintage_count
        or not isinstance(meetings, list)
        or len(meetings) != expected_vintage_count
    ):
        raise ValueError(
            f"Snapshot manifest meeting inventory does not match workflow mode {mode!r}"
        )
    snapshot_sha256 = sha256_file(named["snapshot_manifest"])

    for name, (phase, population_id) in POPULATION_CONTRACTS.items():
        payload = payloads.get(name)
        if payload is None:
            continue
        if payload.get("population_id") != population_id:
            raise ValueError(
                f"Workflow artifact {name!r} has population_id "
                f"{payload.get('population_id')!r}; expected {population_id!r}"
            )
        if name.endswith("_release_manifest"):
            if (
                payload.get("phase") != phase
                or payload.get("generation_only") is not True
                or payload.get("training_performed") is not False
            ):
                raise ValueError(
                    f"Workflow artifact {name!r} is not a complete "
                    "generation-only release for the expected phase"
                )

    for prefix in ("pilot", "formal"):
        ledger_name = f"{prefix}_ledger_manifest"
        release_name = f"{prefix}_release_manifest"
        ledger = payloads.get(ledger_name)
        release = payloads.get(release_name)
        if ledger is not None:
            inputs = ledger.get("inputs")
            if not isinstance(inputs, Mapping):
                raise ValueError(
                    f"Workflow artifact {ledger_name!r} has no input bindings"
                )
            ledger_registry = inputs.get("registry")
            ledger_snapshot = inputs.get("snapshot_manifest")
            if (
                not isinstance(ledger_registry, Mapping)
                or ledger_registry.get("sha256") != registry_sha256
                or not isinstance(ledger_snapshot, Mapping)
                or ledger_snapshot.get("sha256") != snapshot_sha256
            ):
                raise ValueError(
                    f"Workflow artifact {ledger_name!r} does not bind the "
                    "workflow registry and snapshot"
                )
        if release is not None:
            analysis = release.get("analysis")
            provenance = (
                analysis.get("ledger_provenance")
                if isinstance(analysis, Mapping)
                else None
            )
            if (
                not isinstance(provenance, Mapping)
                or provenance.get("source_registry_sha256") != registry_sha256
            ):
                raise ValueError(
                    f"Workflow artifact {release_name!r} does not bind the "
                    "workflow source registry"
                )
            if ledger is not None and (
                provenance.get("ledger_manifest_sha256")
                != sha256_file(named[ledger_name])
                or provenance.get("snapshot_manifest_sha256") != snapshot_sha256
            ):
                raise ValueError(
                    f"Workflow artifact {release_name!r} does not bind its "
                    "workflow ledger and snapshot"
                )


def build_workflow_release(
    *,
    run_id: str,
    mode: str,
    workflow_root: str | Path,
    artifacts: Sequence[tuple[str, Path]],
) -> dict[str, Any]:
    if mode not in REQUIRED_ARTIFACTS:
        raise ValueError(f"Unsupported workflow mode: {mode!r}")
    if not run_id or not run_id.strip():
        raise ValueError("run_id must be non-empty")
    root = Path(workflow_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"workflow_root does not exist: {root}")

    named: dict[str, Path] = {}
    for raw_name, raw_path in artifacts:
        name = str(raw_name).strip()
        if name in named:
            raise ValueError(f"Duplicate workflow artifact name: {name!r}")
        path = raw_path.expanduser().resolve()
        if path != root and root not in path.parents:
            raise ValueError(
                f"Workflow artifact {name!r} is outside workflow_root: {path}"
            )
        if not path.is_file():
            raise FileNotFoundError(
                f"Workflow artifact {name!r} is not a regular file: {path}"
            )
        named[name] = path

    missing = sorted(REQUIRED_ARTIFACTS[mode] - set(named))
    unexpected = sorted(set(named) - REQUIRED_ARTIFACTS[mode])
    if missing or unexpected:
        raise ValueError(
            f"Workflow artifact inventory mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )

    _validate_artifact_contracts(mode=mode, named=named)

    inventory = {
        name: {
            **fingerprint_artifact_path(path),
            "relative_path": path.relative_to(root).as_posix(),
        }
        for name, path in sorted(named.items())
    }
    for record in inventory.values():
        record.pop("path", None)

    return seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "run_id": run_id.strip(),
            "mode": mode,
            "generation_only": True,
            "training_performed": False,
            "information_cutoff_policy": "previous-calendar-day-v1",
            "meeting_day_data": "forbidden",
            "authentication_required": False,
            "workflow_root": str(root),
            "artifacts": inventory,
        }
    )


def write_immutable(path: str | Path, payload: dict[str, Any]) -> str:
    output = Path(path).expanduser().resolve()
    serialised = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if output.is_file():
        if output.read_text(encoding="utf-8") != serialised:
            raise ValueError(
                f"Refusing to overwrite incompatible workflow release: {output}"
            )
        return sha256_file(output)
    if output.exists():
        raise ValueError(f"Workflow release destination is not a file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialised)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            if not output.is_file() or output.read_text(encoding="utf-8") != serialised:
                raise ValueError(
                    f"Refusing to overwrite incompatible workflow release: {output}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal a generation-only D-1 canonical LOO workflow."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=sorted(REQUIRED_ARTIFACTS), required=True)
    parser.add_argument("--workflow-root", required=True, type=Path)
    parser.add_argument(
        "--artifact",
        action="append",
        type=parse_named_path,
        required=True,
        metavar="NAME=PATH",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        release = build_workflow_release(
            run_id=args.run_id,
            mode=args.mode,
            workflow_root=args.workflow_root,
            artifacts=args.artifact,
        )
        file_sha256 = write_immutable(args.output, release)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"output={args.output.expanduser().resolve()}")
    print(f"file_sha256={file_sha256}")
    print(f"payload_sha256={release['integrity']['payload_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
