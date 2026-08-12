"""Recompute the bounded facts used by the chk1 completion report.

The audit is intentionally read-only and uses only Python's standard library so it can
be rerun without loading model weights or contacting DeepSeek.  It validates the final
data release, the completed SFT run, merge evidence, and downstream import records.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


GENERATION = Path("output/data/retrain_v2/chk1/generation_full_v7/generation_handoff.json")
CANONICAL = Path(
    "output/data/retrain_v2/chk1/canonical_releases/"
    "chk1_full_v7_automated_v2_20260804"
)
BASE = Path(
    "dataset/processed/retrain_v2/"
    "analysis_base_full_v7_automated_v3_20260804"
)
RUN = Path(
    "output/training/retrain_v2/"
    "retrain_v2_full_v7_automated_v5_20260804"
)


def _json(root: Path, relative: Path | str) -> dict[str, Any]:
    with (root / relative).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {relative}")
    return value


def _jsonl(root: Path, relative: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (root / relative).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected object at {relative}:{line_number}")
            rows.append(value)
    return rows


def run_audit(root: Path | str = Path.cwd()) -> dict[str, Any]:
    root = Path(root).resolve()
    generation = _json(root, GENERATION)
    quality = _json(root, CANONICAL / "audit/quality_report.json")
    base_manifest = _json(root, BASE / "base_release_manifest.json")
    split_audit = _json(root, BASE / "audits/split_integrity.json")
    token_audit = _json(root, BASE / "audits/token_budget.json")
    run_manifest = _json(root, RUN / "run_manifest.json")
    all_results = _json(root, RUN / "adapters/chk1/all_results.json")
    trainer_state = _json(root, RUN / "adapters/chk1/trainer_state.json")
    runtime = _json(root, RUN / "adapters/chk1/resolved_runtime_config.json")
    merge = _json(root, RUN / "merged/chk1/merge_attestation.json")

    canonical_counts = {
        split: len(_jsonl(root, CANONICAL / f"sft/{split}.jsonl"))
        for split in ("train", "eval", "test")
    }
    sft_rows = {
        split: _jsonl(root, BASE / f"analysis_sft/{split}.jsonl")
        for split in ("train", "eval", "test")
    }
    grpo_counts = {
        split: len(_jsonl(root, BASE / f"analysis_grpo/{split}.jsonl"))
        for split in ("train", "eval", "test")
    }
    sft_counts = {split: len(rows) for split, rows in sft_rows.items()}
    empty_final_responses = sum(
        not str(row.get("response", "")).strip()
        for rows in sft_rows.values()
        for row in rows
    )
    missing_reasoning_boundary = sum(
        "\n</think>\n" not in str(row.get("response", ""))
        for rows in sft_rows.values()
        for row in rows
    )

    audit_statuses = {
        name: _json(root, BASE / f"audits/{name}.json").get("status")
        for name in (
            "schema",
            "token_budget",
            "reference_leakage",
            "point_in_time",
            "split_integrity",
            "target_consistency",
            "encoding",
            "teacher_grounding",
        )
    }
    imports = []
    for version in range(6, 10):
        relative = Path(
            "output/training/retrain_v2/"
            f"retrain_v2_full_v7_automated_v{version}_20260804/imports/chk1.json"
        )
        record = _json(root, relative)
        imports.append(
            {
                "target_run_id": record["target_run_id"],
                "source_run_id": record["source_run_id"],
                "source_artifact_sha256": record["source_artifact"]["sha256"],
            }
        )

    stage = run_manifest["stages"]["chk1"]
    teacher = generation["generation_provenance"]["teacher_contract"]
    roles = split_audit["details"]["train_assignment"]["actual_counts"]
    sft_budget = token_audit["details"]["datasets"]["analysis_sft"]
    eval_history = [
        {
            "step": row["step"],
            "epoch": row["epoch"],
            "eval_loss": row["eval_loss"],
            "eval_mean_token_accuracy": row["eval_mean_token_accuracy"],
        }
        for row in trainer_state["log_history"]
        if "eval_loss" in row
    ]

    results = {
        "generation": {
            "status": generation["status"],
            "selected": generation["selected_count"],
            "accepted": generation["accepted_count"],
            "excluded": generation["excluded_count"],
            "retrieval_rate": generation["accepted_count"] / generation["selected_count"],
            "teacher_provider": teacher["provider"],
            "teacher_model": teacher["model"],
            "concurrency": generation["generation_provenance"]["deepseek_concurrency"],
            "analysis_source": teacher["analysis_source"],
            "answer_source": teacher["answer_source"],
            "candidate_processing": generation["generation_provenance"]["candidate_processing"],
        },
        "canonical_release": {
            "release_id": base_manifest["source_release"]["release_id"],
            "population": quality["population_count"],
            "accepted": sum(canonical_counts.values()),
            "excluded": sum(quality["excluded_counts"].values()),
            "split_counts": canonical_counts,
            "status": quality["status"],
        },
        "base_release": {
            "release_id": base_manifest["release_id"],
            "artifact_sha256": "b8128b394420f90d8e7cb2e0e422a7dc097b5d6c68adb30f73ded29f764c040f",
            "train_roles": roles,
            "sft_counts": sft_counts,
            "grpo_counts": grpo_counts,
            "empty_final_responses": empty_final_responses,
            "missing_reasoning_boundary": missing_reasoning_boundary,
            "audit_statuses": audit_statuses,
            "token_budget": sft_budget,
        },
        "training": {
            "run_id": run_manifest["run_id"],
            "status": stage["status"],
            "sealed_at_utc": stage["sealed_at_utc"],
            "base_model": runtime["model"]["model_name_or_path"],
            "world_size": runtime["environment"]["world_size"],
            "gpus": runtime["environment"]["cuda_device_names"],
            "epochs": runtime["training"]["num_train_epochs"],
            "global_step": trainer_state["global_step"],
            "train_samples": all_results["train_samples"],
            "eval_samples": all_results["eval_samples"],
            "train_loss": all_results["train_loss"],
            "eval_loss": all_results["eval_loss"],
            "train_runtime_seconds": all_results["train_runtime"],
            "eval_history": eval_history,
            "adapter_sha256": stage["adapter"]["sha256"],
            "merged_sha256": stage["artifact"]["sha256"],
            "merged_bytes": stage["artifact"]["total_bytes"],
            "training_receipt_sha256": stage["training_receipt_sha256"],
            "merge_receipt_sha256": stage["merge_receipt_sha256"],
            "merge_semantic_evidence": merge["semantic_evidence"],
        },
        "downstream_reuse": imports,
    }

    # Fail loudly if a source changes underneath the report.
    assert results["generation"]["status"] == "complete"
    assert (results["generation"]["selected"], results["generation"]["accepted"]) == (2117, 2115)
    assert results["canonical_release"]["accepted"] == 2072
    assert results["canonical_release"]["excluded"] == 45
    assert roles == {"sft_only": 1190, "grpo_only": 328, "shared": 165}
    assert sft_counts == {"train": 1355, "eval": 199, "test": 190}
    assert grpo_counts == {"train": 493, "eval": 199, "test": 190}
    assert empty_final_responses == 0
    assert missing_reasoning_boundary == 0
    assert set(audit_statuses.values()) == {"passed"}
    assert results["training"]["status"] == "sealed"
    assert results["training"]["global_step"] == 170
    assert results["training"]["merged_sha256"] == "9b355f903b7722f274bca1bc226bf75c1da4ebe449de726174bd1d054ff8948c"
    assert results["training"]["merge_semantic_evidence"]["residual_lora_modules_after"] == 0
    assert len(imports) == 4
    assert {item["source_run_id"] for item in imports} == {run_manifest["run_id"]}
    assert {item["source_artifact_sha256"] for item in imports} == {results["training"]["merged_sha256"]}
    return results


if __name__ == "__main__":
    print(json.dumps(run_audit(), ensure_ascii=False, indent=2, sort_keys=True))
