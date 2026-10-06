"""Fresh Model chk-0 extension to the sealed stochastic Decision evaluation.

This create-only workflow evaluates paper Model chk-0 on the exact N19/N12
prompt population used by the sealed three-model K=10 release.  It reuses no
generated text.  Prompt token IDs, replicate seeds, and meeting--replicate row
seeds are byte-identical to the existing Model chk-1 / Model chk-3 evaluation,
so the eventual four-model comparison changes model weights only.

The implementation deliberately delegates raw-token persistence, vLLM V1
generation, replay, parsing, scoring, and bootstrap mechanics to the frozen
three-model evaluator whose hash is bound by the source release.  This module
only supplies the chk-0 model binding, one-model population closure, a fresh
output root, and worker-launch identity.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jobs.retrain_v2 import (
    evaluate_chk4_external_core8_panels as source,
)
from jobs.retrain_v2 import (
    evaluate_chk4_external_core8_stochastic_vllm_k10 as base,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_K10_ROOT = REPO_ROOT / (
    "output/evaluation/retrain_v2/"
    "chk4_external_stochastic_core8_n19_n12_vllm_t06_p09_k10_v3_20260824"
)
SOURCE_K10_MANIFEST = SOURCE_K10_ROOT / "evaluation_manifest.json"
SOURCE_K10_MANIFEST_SHA256 = (
    "8a39f948c2ba1ce69ad82c821d3c2a8602bc4a5285157adcbf3eed58c451a09e"
)
SOURCE_EVALUATION_ID = (
    "chk4-external-core8-n19-n12-vllm-t06-p09-k10-dual-dp1-v3"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / (
    "output/evaluation/retrain_v2/"
    "chk0_external_stochastic_core8_n19_n12_vllm_t06_p09_k10_v1_20260824"
)
EVALUATION_ID = "chk0-external-core8-n19-n12-vllm-t06-p09-k10-dual-dp1-v1"

CHK0_PATH = (REPO_ROOT / "models/DeepSeek-R1-Distill-Llama-8B").resolve()
CHK0_SHA256 = "bfb086cb87e9616e60805fc6f6f95826294b1de70b158cfea2d3e213df38ca11"
CHK0_SOURCE_SPEC: dict[str, Any] = {
    "label": "model_chk0",
    "display_name": "Model chk-0",
    "paper_model": "Model chk-0",
    "checkpoint_step": None,
    "training_state": "pretrained_base_model",
    "path": CHK0_PATH,
    "sha256": CHK0_SHA256,
}


def _paired_row_seed(sample_id: str, replicate_id: int) -> int:
    """Replay the source release's model-shared request seed exactly."""

    base._require(0 <= replicate_id < base.REPLICATES, "invalid replicate ID")
    value = (
        f"{SOURCE_EVALUATION_ID}:"
        f"{base.REPLICATE_SEEDS[replicate_id]}:{sample_id}"
    )
    return int(base.hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16) % (
        2**31
    )


def _configure() -> None:
    """Set the frozen evaluator's population constants for one chk-0 arm."""

    base.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    base.EVALUATION_ID = EVALUATION_ID
    base.MODEL_COUNT = 1
    base.EXPECTED_RAW_ROWS = base.PAIRED_BLOCKS
    base.EXPECTED_PANEL_ROWS = {
        panel: count * base.REPLICATES
        for panel, count in base.PANEL_COUNTS.items()
    }
    base.EXPECTED_ROWS_PER_SHARD = base.EXPECTED_BLOCKS_PER_SHARD
    base.SMOKE_MODEL_LABEL = "model_chk0"
    base.GPU_LOCK_TEMPLATE = "/tmp/fomc_trainer_chk0_decision_k10_gpu{index}.lock"
    base.MANIFEST_SCHEMA = "chk0-external-core8-stochastic-vllm-k10-manifest-v1"
    base.AUTHORIZATION_SCHEMA = (
        "chk0-external-core8-stochastic-vllm-k10-authorization-v1"
    )
    base.SMOKE_RECEIPT_SCHEMA = (
        "chk0-external-core8-stochastic-vllm-k10-smoke-receipt-v1"
    )
    base.RAW_ROW_SCHEMA = "chk0-external-core8-stochastic-vllm-k10-raw-row-v1"
    base.RESULT_ROW_SCHEMA = (
        "chk0-external-core8-stochastic-vllm-k10-result-row-v1"
    )
    base.CHUNK_RECEIPT_SCHEMA = (
        "chk0-external-core8-stochastic-vllm-k10-chunk-receipt-v1"
    )
    base.WORKER_RECEIPT_SCHEMA = (
        "chk0-external-core8-stochastic-vllm-k10-worker-receipt-v1"
    )
    base.RUN_RECEIPT_SCHEMA = (
        "chk0-external-core8-stochastic-vllm-k10-run-receipt-v1"
    )
    base.SUMMARY_SCHEMA = "chk0-external-core8-stochastic-vllm-k10-summary-v1"
    base.BOOTSTRAP_SCHEMA = (
        "chk0-external-core8-stochastic-vllm-k10-bootstrap-draw-v1"
    )
    base.IMPLEMENTATION_FILES = {
        "chk0_extension": Path(__file__).resolve(),
        "frozen_three_model_evaluator": Path(base.__file__).resolve(),
        "source_v4_evaluator": Path(source.__file__).resolve(),
        **{
            key: path
            for key, path in base.IMPLEMENTATION_FILES.items()
            if key
            not in {
                "stochastic_vllm_evaluator",
                "source_v4_evaluator",
            }
        },
    }
    base._row_seed = _paired_row_seed
    base._launch_worker_pair = _launch_worker_pair
    source.MODELS = (CHK0_SOURCE_SPEC,)


def _source_k10_manifest() -> dict[str, Any]:
    base._require(
        base._sha256_file(SOURCE_K10_MANIFEST) == SOURCE_K10_MANIFEST_SHA256,
        "source K10 manifest drift",
    )
    value = base._read_json(SOURCE_K10_MANIFEST, label="source K10 manifest")
    base._validate_integrity(value, label="source K10 manifest")
    base._require(
        value.get("evaluation_id") == SOURCE_EVALUATION_ID
        and value.get("population", {}).get("expected_rows") == 930,
        "source K10 identity/population drift",
    )
    for name in (
        "samples",
        "prompt_token_ledger",
        "case_schedule",
        "official_sources",
    ):
        record = value["inputs"][name]
        path = SOURCE_K10_ROOT / str(record["path"])
        base._require(
            base._file_record(path, relative_to=SOURCE_K10_ROOT) == record,
            f"source K10 input drift: {name}",
        )
    return value


def prepare(output_root: Path) -> dict[str, Any]:
    """Create and seal a fresh one-model chk-0 evaluation root."""

    from open_r1.provenance import fingerprint_artifact_path

    root = output_root.resolve()
    base._require(
        not root.exists() and not root.is_symlink(), f"output root exists: {root}"
    )
    source_manifest = _source_k10_manifest()
    artifact = fingerprint_artifact_path(CHK0_PATH)
    base._require(
        artifact.get("sha256") == CHK0_SHA256,
        "Model chk-0 artifact fingerprint drift",
    )
    model = {
        key: copy.deepcopy(value)
        for key, value in CHK0_SOURCE_SPEC.items()
        if key != "path"
    }
    model["artifact"] = artifact

    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.", dir=root.parent))
    try:
        copied_records: dict[str, dict[str, Any]] = {}
        for name, destination in (
            ("samples", "inputs/panel_samples.jsonl"),
            ("prompt_token_ledger", "inputs/prompt_token_ledger.jsonl"),
        ):
            source_path = SOURCE_K10_ROOT / source_manifest["inputs"][name]["path"]
            destination_path = staging / destination
            base._copy_regular_file(source_path, destination_path)
            copied_records[name] = base._file_record(
                destination_path, relative_to=staging
            )

        samples = base._read_jsonl(
            staging / copied_records["samples"]["path"], label="copied samples"
        )
        tokens = base._read_jsonl(
            staging / copied_records["prompt_token_ledger"]["path"],
            label="copied prompt-token ledger",
        )
        base._require(
            len(samples) == len(tokens) == base.MEETING_COUNT
            and Counter(row["panel"] for row in samples)
            == Counter(base.PANEL_COUNTS),
            "copied sample/token population drift",
        )

        # The local chk-0 tokenizer files are not byte-identical to the merged
        # checkpoints.  Fair comparison therefore requires a semantic check:
        # all 31 rendered prompts must nevertheless reproduce the exact sealed
        # token-ID ledger used by the three-model release.
        system_prompt, _ = source._training_contract()
        tokenizer = source._load_tokenizer(CHK0_SOURCE_SPEC)
        token_by_id = {str(row["sample_id"]): row for row in tokens}
        for sample in samples:
            observed_ids = source._prompt_ids(
                tokenizer,
                source._messages(system_prompt, str(sample["prompt"])),
            )
            expected = token_by_id[str(sample["sample_id"])]
            base._require(
                observed_ids == expected["prompt_token_ids"],
                f"chk0/source prompt-token mismatch: {sample['sample_id']}",
            )
        global_preimage = [
            [row["sample_id"], row["prompt_token_ids"]] for row in tokens
        ]
        base._require(
            base._sha256_text(
                json.dumps(
                    global_preimage,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
            == base.GLOBAL_PROMPT_TOKEN_IDS_SHA256,
            "chk0 global prompt-token ledger drift",
        )

        source_official = (
            SOURCE_K10_ROOT / source_manifest["inputs"]["official_sources"]["path"]
        )
        official_rows = base._read_jsonl(
            source_official, label="source official manifest"
        )
        for row in official_rows:
            base._copy_regular_file(
                SOURCE_K10_ROOT / str(row["path"]),
                staging / str(row["path"]),
            )
        official_path = staging / "sources/official_source_manifest.jsonl"
        base._write_exclusive_jsonl(official_path, official_rows)

        cases = base.canonical_cases(samples, [model])
        source_cases = base._read_jsonl(
            SOURCE_K10_ROOT / source_manifest["inputs"]["case_schedule"]["path"],
            label="source K10 case schedule",
        )
        source_seed_by_block = {
            int(row["paired_block_id"]): int(row["row_seed"])
            for row in source_cases
        }
        base._require(
            len(source_seed_by_block) == base.PAIRED_BLOCKS
            and all(
                int(row["row_seed"])
                == source_seed_by_block[int(row["paired_block_id"])]
                for row in cases
            ),
            "chk0/source meeting-replicate seed pairing drift",
        )
        case_path = staging / "inputs/case_schedule.jsonl"
        base._write_exclusive_jsonl(case_path, cases)

        runtime = base._vllm_runtime_probe()
        gpu_inventory = base._physical_gpu_inventory()
        manifest = base._sealed(
            {
                "schema_version": base.MANIFEST_SCHEMA,
                "status": "prepared_pending_smoke_and_authorization",
                "created_at_utc": base._utc_now(),
                "evaluation_id": EVALUATION_ID,
                "purpose": "fresh_chk0_extension_to_sealed_stochastic_decision_evaluation",
                "output_root": str(root),
                "source_v4": copy.deepcopy(source_manifest["source_v4"]),
                "source_stochastic_k10": {
                    "role": "sealed_inputs_and_pairing_source_only",
                    "manifest": base._file_record(SOURCE_K10_MANIFEST),
                    "manifest_expected_sha256": SOURCE_K10_MANIFEST_SHA256,
                    "generated_rows_reused": 0,
                    "prompt_token_ids_reused_as_frozen_inputs": True,
                    "meeting_replicate_row_seeds_replayed": True,
                },
                "supersedes_failed_attempts": [
                    base._failed_smoke_v1_supersession_record(),
                    base._failed_formal_v2_supersession_record(),
                ],
                "implementation": base._implementation_records(),
                "models": [model],
                "checkpoint_governance": {},
                "inputs": {
                    "samples": copied_records["samples"],
                    "prompt_token_ledger": copied_records["prompt_token_ledger"],
                    "case_schedule": base._file_record(
                        case_path, relative_to=staging
                    ),
                    "official_sources": base._file_record(
                        official_path, relative_to=staging
                    ),
                },
                "prompt_contract": {
                    **copy.deepcopy(source_manifest["prompt_contract"]),
                    "chk0_tokenizer_reproduces_source_prompt_ids": True,
                    "comparison_changes_model_weights_not_prompt_token_ids": True,
                },
                "input_contract": copy.deepcopy(source_manifest["input_contract"]),
                "population": {
                    "panels": copy.deepcopy(base.PANEL_COUNTS),
                    "class_counts": copy.deepcopy(
                        source_manifest["population"]["class_counts"]
                    ),
                    "meetings": base.MEETING_COUNT,
                    "models": 1,
                    "replicates_per_meeting_model": base.REPLICATES,
                    "expected_rows": base.EXPECTED_RAW_ROWS,
                    "expected_panel_rows": copy.deepcopy(
                        base.EXPECTED_PANEL_ROWS
                    ),
                    "training_performed": False,
                },
                "generation": {
                    **copy.deepcopy(source_manifest["generation"]),
                    "expected_rows_per_shard": base.EXPECTED_ROWS_PER_SHARD,
                    "models_processed_sequentially": False,
                    "paired_block": "meeting_x_replicate_chk0_extension",
                    "row_seed": (
                        "sha256(source_evaluation_id:replicate_seed:sample_id)_"
                        "first64_mod_2^31"
                    ),
                    "source_evaluation_id_for_seed": SOURCE_EVALUATION_ID,
                    "row_seed_shared_with_source_three_models": True,
                    "fresh_generations_only": True,
                    "source_generated_rows_reused": 0,
                },
                "runtime_prebinding": {
                    "vllm_python": runtime,
                    "physical_gpu_inventory": gpu_inventory,
                    "required_environment": copy.deepcopy(base.REQUIRED_VLLM_ENV),
                    "gpu_lock_template": base.GPU_LOCK_TEMPLATE,
                },
                "statistics": {
                    "independent_unit": "meeting",
                    "replicates_nested_within_meeting": True,
                    "absolute_chk0_intervals_only_in_extension": True,
                    "four_model_contrasts_deferred_until_merge_with_source_summary": True,
                    "bootstrap_draw_plan_shared_with_source_release": True,
                    "panels_reported_separately": True,
                    "pooled_n31_prohibited": True,
                },
                "limitations": [
                    "K=10 characterizes decoding variability and does not increase independent meeting N",
                    "postcutoff_n12 contains no hike meeting",
                    "historical_n19 may be represented in base-model pretraining",
                    "BF16 vLLM results are not directly interchangeable with earlier NF4 Transformers results",
                ],
            }
        )
        manifest_path = staging / "evaluation_manifest.json"
        base._write_exclusive_json(manifest_path, manifest)
        os.rename(staging, root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "status": "prepared_pending_smoke_and_authorization",
        "manifest": base._file_record(root / "evaluation_manifest.json"),
        "expected_rows": base.EXPECTED_RAW_ROWS,
        "source_generated_rows_reused": 0,
    }


def _launch_worker_pair(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    authorization_path: Path | None,
    authorization_sha256: str | None,
    model_label: str,
    resume: bool,
    scope: str = "formal",
) -> None:
    """Launch both GPU workers through this module so chk-0 constants persist."""

    root = manifest_path.parent
    base._require(scope in {"formal", "smoke"}, "invalid launch scope")
    logs = root / ("run/logs" if scope == "formal" else "smoke/logs")
    logs.mkdir(parents=True, exist_ok=True)
    processes: list[subprocess.Popen[str]] = []
    handles = []
    try:
        for shard_id in range(base.SHARD_COUNT):
            log_path = logs / f"{model_label}.shard{shard_id}.log"
            handle = log_path.open("a", encoding="utf-8")
            handles.append(handle)
            environment = {
                **os.environ,
                **base.REQUIRED_VLLM_ENV,
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": str(shard_id),
                "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT}",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
            command = [
                str(base.VLLM_PYTHON),
                "-u",
                "-m",
                "jobs.retrain_v2.evaluate_chk0_external_core8_stochastic_vllm_k10",
                "worker",
                "--manifest",
                str(manifest_path),
                "--manifest-sha256",
                manifest_sha256,
                "--model-label",
                model_label,
                "--shard-id",
                str(shard_id),
                "--physical-gpu-index",
                str(shard_id),
                "--scope",
                scope,
            ]
            if authorization_path is not None:
                command.extend(
                    [
                        "--authorization",
                        str(authorization_path),
                        "--authorization-sha256",
                        str(authorization_sha256),
                    ]
                )
            if resume:
                command.append("--resume")
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        failures = []
        for shard_id, process in enumerate(processes):
            code = process.wait()
            if code:
                failures.append((shard_id, code))
        base._require(
            not failures, f"chk0 vLLM worker failures: {failures}"
        )
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for handle in handles:
            handle.close()


def authorize(
    manifest_path: Path,
    manifest_sha256: str,
    *,
    audit_statement: str,
) -> dict[str, Any]:
    manifest = base.validate_manifest(
        manifest_path, manifest_sha256, deep_models=True
    )
    base._require(
        len(audit_statement.strip()) >= 20,
        "independent audit statement is too short",
    )
    smoke_path = manifest_path.parent / "smoke/smoke_receipt.json"
    smoke = base._validate_smoke_receipt(
        smoke_path,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
    )
    path = manifest_path.parent / "formal_generation_authorization.json"
    base._require(
        not path.exists() and not path.is_symlink(), "authorization already exists"
    )
    value = base._sealed(
        {
            "schema_version": base.AUTHORIZATION_SCHEMA,
            # Keep the frozen validator's status enum; the narrower 310-row
            # scope is recorded separately in ``authorized_scope``.
            "status": "authorized_after_independent_audit",
            "created_at_utc": base._utc_now(),
            "evaluation_id": EVALUATION_ID,
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": manifest_sha256,
            },
            "audit_statement": audit_statement.strip(),
            "authorized_scope": "formal_310_row_chk0_two_gpu_vllm_generation_only",
            "validated_infrastructure_smoke": {
                "receipt": base._file_record(smoke_path),
                "rows": smoke["rows"],
                "shards": smoke["shards"],
                "gates": copy.deepcopy(smoke["gates"]),
            },
            "generation_contract": copy.deepcopy(manifest["generation"]),
        }
    )
    base._write_exclusive_json(path, value)
    return {"status": value["status"], "authorization": base._file_record(path)}


def _status(output_root: Path) -> dict[str, Any]:
    root = output_root.resolve()
    workers = {"model_chk0": {}}
    receipts = 0
    rows = 0
    for shard_id in range(base.SHARD_COUNT):
        paths = base._worker_paths(root, "model_chk0", shard_id)
        count = 0
        if paths["wal"].is_file():
            with paths["wal"].open("rb") as handle:
                count = sum(bool(line.strip()) for line in handle)
        complete = paths["receipt"].is_file()
        rows += count
        receipts += int(complete)
        workers["model_chk0"][f"shard{shard_id}"] = {
            "wal_rows": count,
            "expected_rows": base.EXPECTED_BLOCKS_PER_SHARD,
            "complete": complete,
        }
    return {
        "output_root": str(root),
        "prepared": (root / "evaluation_manifest.json").is_file(),
        "authorized": (root / "formal_generation_authorization.json").is_file(),
        "generation_complete": (root / "run/run_receipt.json").is_file(),
        "scored": (root / "report/report_receipt.json").is_file(),
        "wal_rows": rows,
        "expected_rows": base.EXPECTED_RAW_ROWS,
        "worker_receipts": receipts,
        "expected_worker_receipts": 2,
        "workers": workers,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "smoke", "authorize", "run", "worker", "score", "status"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--audit-statement")
    parser.add_argument("--model-label")
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--scope", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    _configure()
    args = _parser().parse_args()
    root = args.output_root.resolve()
    if args.stage == "prepare":
        result = prepare(root)
    elif args.stage == "status":
        result = _status(root)
    else:
        manifest_path = (
            args.manifest.resolve()
            if args.manifest is not None
            else root / "evaluation_manifest.json"
        )
        base._require(bool(args.manifest_sha256), "--manifest-sha256 is required")
        if args.stage == "smoke":
            result = base.smoke(
                manifest_path, args.manifest_sha256, resume=args.resume
            )
        elif args.stage == "authorize":
            base._require(
                bool(args.audit_statement), "--audit-statement is required"
            )
            result = authorize(
                manifest_path,
                args.manifest_sha256,
                audit_statement=args.audit_statement,
            )
        elif args.stage == "worker":
            base._require(bool(args.model_label), "--model-label is required")
            base._require(args.shard_id is not None, "--shard-id is required")
            base._require(
                args.physical_gpu_index is not None,
                "--physical-gpu-index is required",
            )
            result = base.worker(
                manifest_path=manifest_path,
                manifest_sha256=args.manifest_sha256,
                authorization_path=args.authorization,
                authorization_sha256=args.authorization_sha256,
                model_label=args.model_label,
                shard_id=args.shard_id,
                physical_gpu_index=args.physical_gpu_index,
                resume=args.resume,
                scope=args.scope,
            )
        elif args.stage == "run":
            base._require(args.authorization is not None, "--authorization required")
            base._require(
                bool(args.authorization_sha256),
                "--authorization-sha256 required",
            )
            result = base.run(
                manifest_path,
                args.manifest_sha256,
                authorization_path=args.authorization,
                authorization_sha256=args.authorization_sha256,
                resume=args.resume,
            )
        else:
            base._require(args.authorization is not None, "--authorization required")
            base._require(
                bool(args.authorization_sha256),
                "--authorization-sha256 required",
            )
            result = base.score(
                manifest_path,
                args.manifest_sha256,
                authorization_path=args.authorization,
                authorization_sha256=args.authorization_sha256,
            )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
