"""Repair batched EOS-padding instrumentation without rerunning generation.

The checkpoint-2 v2 screen generated all 16 cases correctly, but Hugging Face
right-padded early-stopped rows with the EOS-valued pad token to the longest
sequence in each batch.  The v2 postprocessor removed only the final EOS and
therefore decoded hundreds of padding tokens as model output.  This utility
seals that diagnosis and creates a new v3 result set by truncating each stored
token sequence at its first bound EOS (inclusive), then deterministically
replaying decode, completion metrics, Decision-v3 reward, and the fixed gate.
It never loads the model or generates a token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.retrain_v2 import probe_chk4_decision_sft_generation as completion_probe
from jobs.retrain_v2 import probe_chk4_pre2009_correction as probe
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


RUN_ROOT = Path(
    "output/training/retrain_v2/"
    "chk4_from_pre2009_cp38_selected_correction_sft_lr2e6_steps6_v1_20260811"
)
SCREEN_ROOT = RUN_ROOT / "static_screening"
MANIFEST = SCREEN_ROOT / "manifests/checkpoint_selection_cap1536_v1.json"
MANIFEST_SHA256 = "e234c3bdb3021b231454c46e05d6350839d9013a40e2d9dc4afca4130c0bb9c8"
SOURCE_V2 = SCREEN_ROOT / "checkpoint-2-selection-cap1536-v2"
OUTPUT_V3 = SCREEN_ROOT / "checkpoint-2-selection-cap1536-v3"
DIAGNOSIS = SCREEN_ROOT / (
    "receipts/checkpoint-2-selection-cap1536-v2_invalid_instrumentation.json"
)

SOURCE_LAUNCH_SHA256 = (
    "5061a5020d41edab43f9c8973d98d85d5a9655b2634f69766b3bf4e353ac82be"
)
SOURCE_RESULTS_SHA256 = (
    "259191cb2a478068ab4be2c7a03da810a8534190f2c992fc788d77b6b596b4e9"
)
SOURCE_SUMMARY_SHA256 = (
    "59c3a4a10f6aa71c84aed5adc063b25a9b48363d0d05bbbdc2b6173a240578b2"
)
SOURCE_PROBE_SHA256 = (
    "3843f840d4013a20714f480d43fac7159b399234432e44efb51892356edff5cc"
)
SUPPLEMENTAL_ATTESTATION_SHA256 = (
    "2770fac9ffc58bb1d8a68f018ee3802fd9ad94b9d1a7a51ea45a58f4b2878cbe"
)
EOS_TOKEN_ID = 128001
DIAGNOSIS_SCHEMA = "chk4-pre2009-correction-padding-instrumentation-diagnosis-v1"
REPLAY_RESULT_SCHEMA = "chk4-pre2009-correction-static-screen-padding-replay-v3"
REPLAY_SUMMARY_SCHEMA = "chk4-pre2009-correction-padding-replay-summary-v3"


class PaddingReplayError(RuntimeError):
    """The v2 diagnosis or deterministic v3 replay failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PaddingReplayError(message)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PaddingReplayError(f"invalid {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be an object")
    return value


def _read_jsonl_with_raw(
    path: Path, *, label: str
) -> list[tuple[str, dict[str, Any]]]:
    _require(path.is_file() and not path.is_symlink(), f"missing {label}: {path}")
    rows: list[tuple[str, dict[str, Any]]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _require(bool(raw), f"{label}:{line_number}: blank row")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PaddingReplayError(f"{label}:{line_number}: invalid JSON") from exc
        _require(isinstance(value, dict), f"{label}:{line_number}: not an object")
        rows.append((raw, value))
    return rows


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


def truncate_at_first_eos(
    token_ids: Sequence[int], eos_token_ids: set[int]
) -> tuple[list[int], dict[str, Any]]:
    """Keep the first EOS and discard every later batched padding token."""

    _require(bool(eos_token_ids), "EOS set must not be empty")
    try:
        values = [int(value) for value in token_ids]
    except (TypeError, ValueError) as exc:
        raise PaddingReplayError("generated token IDs must be integers") from exc
    first_eos_index = next(
        (index for index, token_id in enumerate(values) if token_id in eos_token_ids),
        None,
    )
    if first_eos_index is None:
        trimmed = values
    else:
        trimmed = values[: first_eos_index + 1]
    discarded = values[len(trimmed) :]
    return trimmed, {
        "original_token_count": len(values),
        "first_eos_index": first_eos_index,
        "trimmed_token_count": len(trimmed),
        "discarded_after_first_eos_count": len(discarded),
        "discarded_unique_token_ids": sorted(set(discarded)),
        "all_discarded_tokens_are_bound_eos": bool(discarded)
        and all(value in eos_token_ids for value in discarded),
    }


def _source(repo_root: Path) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    source = root / SOURCE_V2
    inventory = sorted(
        str(path.relative_to(source))
        for path in source.rglob("*")
        if path.is_file() or path.is_symlink()
    )
    _require(
        inventory == ["launch.json", "results.jsonl", "summary.json"],
        "v2 source inventory drift",
    )
    manifest_path = root / MANIFEST
    _require(sha256_file(manifest_path) == MANIFEST_SHA256, "manifest hash drift")
    _require(
        sha256_file(root / "jobs/retrain_v2/probe_chk4_pre2009_correction.py")
        == SOURCE_PROBE_SHA256,
        "source probe implementation drift",
    )
    launch_path = source / "launch.json"
    results_path = source / "results.jsonl"
    summary_path = source / "summary.json"
    launch = _read_json(launch_path, label="v2 launch")
    summary = _read_json(summary_path, label="v2 summary")
    raw_rows = _read_jsonl_with_raw(results_path, label="v2 results")
    _require(len(raw_rows) == 16, "v2 result row count drift")
    _require(
        launch.get("purpose") == probe.SELECTION_PURPOSE
        and launch.get("model_label") == "correction-sft-checkpoint-2",
        "v2 launch scope drift",
    )
    _require(
        summary.get("results", {}).get("sha256") == SOURCE_RESULTS_SHA256
        and summary.get("results", {}).get("rows") == 16
        and summary.get("provenance", {}).get("manifest_sha256")
        == MANIFEST_SHA256,
        "v2 summary source binding drift",
    )
    eos_ids = summary.get("provenance", {}).get("eos_token_ids")
    _require(eos_ids == [EOS_TOKEN_ID], "v2 EOS binding drift")
    return {
        "root": source,
        "launch": launch,
        "summary": summary,
        "rows": raw_rows,
        "artifacts": {
            "launch": _descriptor(
                launch_path, expected_sha256=SOURCE_LAUNCH_SHA256
            ),
            "results": {
                **_descriptor(results_path, expected_sha256=SOURCE_RESULTS_SHA256),
                "rows": 16,
            },
            "summary": _descriptor(
                summary_path, expected_sha256=SOURCE_SUMMARY_SHA256
            ),
            "inventory": inventory,
        },
    }


def build_diagnosis(repo_root: Path) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    source = _source(root)
    rows: list[dict[str, Any]] = []
    affected = 0
    for index, (raw, row) in enumerate(source["rows"], 1):
        tokens = row.get("generated_token_ids")
        _require(isinstance(tokens, list), f"v2 row {index} token IDs missing")
        _trimmed, trim = truncate_at_first_eos(tokens, {EOS_TOKEN_ID})
        if trim["discarded_after_first_eos_count"]:
            affected += 1
        rows.append(
            {
                "row": index,
                "sample_id": row.get("sample_id"),
                "batch_index": row.get("batch_index"),
                "generation_index": row.get("generation_index"),
                "source_line_sha256": _sha256_text(raw),
                **trim,
            }
        )
    _require(affected == 13, "unexpected affected EOS-padding row count")
    _require(
        all(
            row["all_discarded_tokens_are_bound_eos"]
            for row in rows
            if row["discarded_after_first_eos_count"]
        ),
        "discarded suffix contains a non-EOS token",
    )
    supplemental_path = root / (
        RUN_ROOT
        / "receipts/supplemental_post_run_provenance_gap_attestation_v1.json"
    )
    return seal_manifest(
        {
            "schema_version": DIAGNOSIS_SCHEMA,
            "status": "invalid_instrumentation_repeated_eos_padding",
            "scope": {
                "checkpoint_step": 2,
                "source_version": "checkpoint-2-selection-cap1536-v2",
                "invalid_for_checkpoint_selection": True,
                "model_failure_claim_prohibited": True,
                "generation_rerun_required": False,
                "offline_replay_required": True,
            },
            "cause": {
                "component": "batched_generation_postprocessing",
                "pad_token_id_equals_bound_eos": True,
                "bound_eos_token_ids": [EOS_TOKEN_ID],
                "bug": "only_final_eos_removed_after_batch_padding",
                "fix": "truncate_at_first_eos_inclusive_before_decode_metrics_reward",
                "generation_sampling_was_not_affected": True,
            },
            "source_artifacts": source["artifacts"],
            "manifest": _descriptor(root / MANIFEST, expected_sha256=MANIFEST_SHA256),
            "source_probe_implementation": _descriptor(
                root / "jobs/retrain_v2/probe_chk4_pre2009_correction.py",
                expected_sha256=SOURCE_PROBE_SHA256,
            ),
            "supplemental_sft_attestation": _descriptor(
                supplemental_path,
                expected_sha256=SUPPLEMENTAL_ATTESTATION_SHA256,
            ),
            "diagnosis": {
                "rows": rows,
                "affected_rows": affected,
                "unaffected_rows": 16 - affected,
                "all_discarded_tokens_are_bound_eos": True,
            },
            "generator": _descriptor(Path(__file__).resolve()),
        }
    )


def _write_exclusive_json(path: Path, value: Mapping[str, Any], *, mode: int) -> None:
    _require(not path.exists() and not path.is_symlink(), f"artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def create_diagnosis(repo_root: Path) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    payload = build_diagnosis(root)
    path = root / DIAGNOSIS
    _write_exclusive_json(path, payload, mode=0o400)
    return {
        "status": payload["status"],
        "path": str(path),
        "sha256": sha256_file(path),
        "payload_sha256": payload["integrity"]["payload_sha256"],
    }


def verify_diagnosis(repo_root: Path) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    path = root / DIAGNOSIS
    observed = _read_json(path, label="padding diagnosis")
    validate_manifest_integrity(observed)
    _require(observed == build_diagnosis(root), "padding diagnosis replay drift")
    return observed


def _replay_row(
    *,
    old_raw: str,
    old: Mapping[str, Any],
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    old_ids = old.get("generated_token_ids")
    _require(isinstance(old_ids, list), "source token IDs missing")
    trimmed, trim = truncate_at_first_eos(old_ids, {EOS_TOKEN_ID})
    completion = completion_probe.decode_completion_preserving_boundary(
        tokenizer, trimmed, {EOS_TOKEN_ID}
    )
    row = probe._result_row(
        manifest=manifest,
        provenance=provenance,
        sample_id=str(old["sample_id"]),
        completion=completion,
        generated_ids=trimmed,
        panel=str(old["panel"]),
        generation_mode=str(old["generation_mode"]),
        generation_index=int(old["generation_index"]),
        seed=int(old["seed"]),
        batch_index=int(old["batch_index"]),
    )
    for key in (
        "purpose",
        "panel",
        "batch_index",
        "sample_id",
        "population_role",
        "target_direction",
        "target_magnitude_bp",
        "generation_mode",
        "generation_index",
        "seed",
        "generation_parameters",
    ):
        _require(row.get(key) == old.get(key), f"immutable row identity drift: {key}")
    row["schema_version"] = REPLAY_RESULT_SCHEMA
    row["offline_replay"] = {
        "generation_not_rerun": True,
        "source_line_sha256": _sha256_text(old_raw),
        "instrumentation_fix": "truncate_at_first_eos_padding",
        **trim,
    }
    return row


def replay(repo_root: Path) -> dict[str, Any]:
    root = repo_root.expanduser().resolve()
    diagnosis = verify_diagnosis(root)
    output = root / OUTPUT_V3
    _require(not output.exists() and not output.is_symlink(), "v3 output exists")
    source = _source(root)
    manifest_path = root / MANIFEST
    try:
        manifest = probe._validate_manifest(manifest_path, MANIFEST_SHA256)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(manifest["tokenizer"]["path"]),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise PaddingReplayError(f"manifest/tokenizer replay failed: {exc}") from exc
    provenance = dict(source["summary"]["provenance"])
    provenance["offline_padding_replay"] = {
        "generation_not_rerun": True,
        "instrumentation_fix": "truncate_at_first_eos_padding",
        "source_results_sha256": SOURCE_RESULTS_SHA256,
        "source_summary_sha256": SOURCE_SUMMARY_SHA256,
        "diagnosis_sha256": sha256_file(root / DIAGNOSIS),
        "diagnosis_payload_sha256": diagnosis["integrity"]["payload_sha256"],
    }
    output.mkdir(parents=True, exist_ok=False)
    launch = {
        "schema_version": REPLAY_SUMMARY_SCHEMA,
        "status": "initializing",
        "model_label": "correction-sft-checkpoint-2",
        "purpose": probe.SELECTION_PURPOSE,
        "generation_not_rerun": True,
        "runtime": {
            "python": platform.python_version(),
            "cuda_used": False,
        },
        "provenance": provenance,
        "source_artifacts": source["artifacts"],
        "diagnosis": _descriptor(root / DIAGNOSIS),
    }
    _write_exclusive_json(output / "launch.json", launch, mode=0o444)
    rows: list[dict[str, Any]] = []
    results_path = output / "results.jsonl"
    descriptor = os.open(results_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for old_raw, old in source["rows"]:
                row = _replay_row(
                    old_raw=old_raw,
                    old=old,
                    manifest=manifest,
                    provenance=provenance,
                    tokenizer=tokenizer,
                )
                handle.write(_canonical_json(row) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                rows.append(row)
    except Exception:
        results_path.chmod(0o444)
        raise
    summary = probe.summarize_results(rows, manifest=manifest, provenance=provenance)
    summary.update(
        {
            "schema_version": REPLAY_SUMMARY_SCHEMA,
            "model_label": "correction-sft-checkpoint-2",
            "generation_not_rerun": True,
            "instrumentation_fix": "truncate_at_first_eos_padding",
            "source_artifacts": source["artifacts"],
            "diagnosis": _descriptor(root / DIAGNOSIS),
            "implementation": {
                "replay": _descriptor(Path(__file__).resolve()),
                "source_probe": _descriptor(
                    root / "jobs/retrain_v2/probe_chk4_pre2009_correction.py",
                    expected_sha256=SOURCE_PROBE_SHA256,
                ),
                "decoder": _descriptor(
                    root
                    / "jobs/retrain_v2/probe_chk4_decision_sft_generation.py"
                ),
            },
            "results": {
                "path": str(results_path.resolve()),
                "sha256": sha256_file(results_path),
                "rows": len(rows),
                "raw_completions_and_token_ids_landed": True,
            },
        }
    )
    _write_exclusive_json(output / "summary.json", summary, mode=0o444)
    output.chmod(0o555)
    return {
        "status": "complete",
        "quality_status": summary["quality_status"],
        "gate_reasons": summary["quality_gate"]["reasons"],
        "path": str(output),
        "results_sha256": summary["results"]["sha256"],
        "summary_sha256": sha256_file(output / "summary.json"),
        "generation_not_rerun": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--execute", action="store_true")
    sub.add_parser("verify-diagnosis")
    replay_parser = sub.add_parser("replay")
    replay_parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "diagnose":
            if args.execute:
                result = create_diagnosis(args.repo_root)
            else:
                value = build_diagnosis(args.repo_root)
                result = {
                    "status": "ready_to_seal_invalid_instrumentation",
                    "payload_sha256": value["integrity"]["payload_sha256"],
                    "path": str((args.repo_root.resolve() / DIAGNOSIS)),
                }
        elif args.command == "verify-diagnosis":
            value = verify_diagnosis(args.repo_root)
            result = {
                "status": value["status"],
                "path": str((args.repo_root.resolve() / DIAGNOSIS)),
                "sha256": sha256_file(args.repo_root.resolve() / DIAGNOSIS),
                "verified": True,
            }
        elif args.execute:
            result = replay(args.repo_root)
        else:
            _require(
                not (args.repo_root.resolve() / OUTPUT_V3).exists(),
                "v3 output exists",
            )
            result = {
                "status": "ready_to_replay_offline",
                "generation_not_rerun": True,
                "path": str(args.repo_root.resolve() / OUTPUT_V3),
            }
    except (PaddingReplayError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
