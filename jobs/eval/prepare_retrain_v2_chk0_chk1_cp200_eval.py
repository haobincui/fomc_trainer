"""Prepare immutable manifests for the chk0 -> selected chk1 Chapter 2 test."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


BASE_ARTIFACT_ID = "eval-chk0-base"
CHK1_ARTIFACT_ID = "eval-chk1-clean-v2-lr1e6-cp200"


def _read_json(path: Path, *, sealed: bool = False) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    if sealed:
        validate_manifest_integrity(value)
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"Refusing to replace immutable artifact: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def prepare(*, repo_root: Path, output_dir: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    output = output_dir.resolve()
    checkpoint_path = output / "evaluation_checkpoint_manifest.json"
    lineage_path = output / "evaluation_lineage_manifest.json"
    if checkpoint_path.exists() or lineage_path.exists():
        if not checkpoint_path.is_file() or not lineage_path.is_file():
            raise FileExistsError("Evaluation manifest set is only partially present")
        checkpoint_manifest = _read_json(checkpoint_path, sealed=True)
        lineage_manifest = _read_json(lineage_path, sealed=True)
        observed_ids = [
            str(row.get("artifact_id") or "")
            for row in checkpoint_manifest.get("artifacts", [])
        ]
        if observed_ids != [BASE_ARTIFACT_ID, CHK1_ARTIFACT_ID]:
            raise ValueError("Existing evaluation checkpoint inventory changed")
        if set(lineage_manifest.get("checkpoints", {})) != set(observed_ids):
            raise ValueError("Existing evaluation lineage inventory changed")
        return {
            "status": "reused",
            "checkpoint_manifest": str(checkpoint_path),
            "checkpoint_manifest_sha256": sha256_file(checkpoint_path),
            "lineage_manifest": str(lineage_path),
            "lineage_manifest_sha256": sha256_file(lineage_path),
            "artifact_ids": observed_ids,
            "test_rows": 33,
        }
    base = (root / "models/DeepSeek-R1-Distill-Llama-8B").resolve()
    merged = (
        root
        / "output/training/retrain_v2/"
        "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
    ).resolve()
    adapter = (
        root
        / "output/training/retrain_v2/"
        "chk1_clean_v2_override_full3ep_lr1e6_maxlen7168_20260810/"
        "adapters/chk1/checkpoint-200"
    ).resolve()
    exact_path = output / "chk1_cp200_exact_merge_lineage.json"
    promotion_path = output / "promotion_manifest.json"
    config_path = root / "configs/main/checkpoint_generation_eval_11.json"
    test_manifest_path = (
        root
        / "output/evaluation/main/checkpoint_generation/checkpoint_eval_11_v1/"
        "dataset/test_manifest.json"
    )
    prompts_path = test_manifest_path.parent / "prompts.jsonl"
    references_path = test_manifest_path.parent / "references.jsonl"

    exact = _read_json(exact_path, sealed=True)
    promotion = _read_json(promotion_path, sealed=True)
    test_manifest = _read_json(test_manifest_path, sealed=True)
    config = _read_json(config_path)
    base_fingerprint = fingerprint_artifact_path(base)
    merged_fingerprint = fingerprint_artifact_path(merged)
    adapter_fingerprint = fingerprint_artifact_path(adapter)

    if exact.get("conclusion") != "exact_base_plus_adapter_merge_verified":
        raise ValueError("Exact merge evidence is not verified")
    if exact.get("subject_artifact_id") != "chk1-clean-v2-lr1e6-cp200":
        raise ValueError("Exact merge evidence subject changed")
    sources = exact.get("sources", {})
    for label, observed, expected in (
        ("base", base_fingerprint, sources.get("base_model", {})),
        ("merged", merged_fingerprint, sources.get("merged_model", {})),
        ("adapter", adapter_fingerprint, sources.get("adapter", {})),
    ):
        if observed.get("sha256") != expected.get("sha256"):
            raise ValueError(f"{label} fingerprint differs from exact merge evidence")
    if promotion.get("checkpoint_step") != 200:
        raise ValueError("Promotion manifest does not select checkpoint-200")
    if test_manifest.get("reference_in_prompt") is not False:
        raise ValueError("Frozen test does not attest reference-free prompts")
    if config.get("evaluation_id") != test_manifest.get("evaluation_id"):
        raise ValueError("Evaluation config/test manifest ID mismatch")
    if _row_count(prompts_path) != 33 or _row_count(references_path) != 33:
        raise ValueError("Frozen Chapter 2 test must contain exactly 33 rows")
    for key, path in (("prompts", prompts_path), ("references", references_path)):
        binding = test_manifest.get("outputs", {}).get(key, {})
        if binding.get("sha256") != sha256_file(path):
            raise ValueError(f"Frozen {key} hash changed")

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    checkpoint_manifest = seal_manifest(
        {
            "schema_version": "retrain-v2-checkpoint-eval-manifest-v1",
            "generated_at_utc": generated_at,
            "training_performed": False,
            "evaluation_scope": "current chk0 -> clean-v2 LR1e-6 selected chk1 checkpoint-200",
            "claim_boundary": (
                "Adjacent chk0-to-chk1 common-test comparison on the frozen Chapter 2 "
                "raw D-1 evidence-to-Minutes stress benchmark. It estimates only the "
                "first post-training edge and is out-of-task for an atomic-analysis chk1."
            ),
            "artifacts": [
                {
                    "artifact_id": BASE_ARTIFACT_ID,
                    "design_checkpoint_id": "chk-0",
                    "intended_parent_id": None,
                    "verified_parent_artifact_id": None,
                    "lineage_status": "verified_root_artifact",
                    "model_path": str(base),
                    "model_sha256": base_fingerprint["sha256"],
                    "tokenizer_path": str(base),
                    "tokenizer_sha256": base_fingerprint["sha256"],
                    "adapter_path": None,
                    "adapter_sha256": None,
                    "usable_for_evaluation": True,
                },
                {
                    "artifact_id": CHK1_ARTIFACT_ID,
                    "design_checkpoint_id": "chk-1",
                    "intended_parent_id": "chk-0",
                    "verified_parent_artifact_id": BASE_ARTIFACT_ID,
                    "lineage_status": "verified_parent_matches_intended_parent",
                    "model_path": str(merged),
                    "model_sha256": merged_fingerprint["sha256"],
                    "tokenizer_path": str(merged),
                    "tokenizer_sha256": merged_fingerprint["sha256"],
                    "adapter_path": str(adapter),
                    "adapter_sha256": adapter_fingerprint["sha256"],
                    "usable_for_evaluation": True,
                },
            ],
            "lineage_evidence": {
                CHK1_ARTIFACT_ID: str(exact_path),
            },
            "bindings": {
                "promotion_manifest": {
                    "path": str(promotion_path),
                    "sha256": sha256_file(promotion_path),
                    "payload_sha256": promotion["integrity"]["payload_sha256"],
                },
                "exact_merge_evidence": {
                    "path": str(exact_path),
                    "sha256": sha256_file(exact_path),
                    "payload_sha256": exact["integrity"]["payload_sha256"],
                },
                "evaluation_config": {
                    "path": str(config_path.resolve()),
                    "sha256": sha256_file(config_path),
                },
                "test_manifest": {
                    "path": str(test_manifest_path.resolve()),
                    "sha256": sha256_file(test_manifest_path),
                    "payload_sha256": test_manifest["integrity"]["payload_sha256"],
                },
            },
        }
    )
    lineage_manifest = seal_manifest(
        {
            "schema_version": "retrain-v2-eval-lineage-v1",
            "checkpoints": {
                BASE_ARTIFACT_ID: {
                    "artifact_id": BASE_ARTIFACT_ID,
                    "parent_artifact_id": None,
                },
                CHK1_ARTIFACT_ID: {
                    "artifact_id": CHK1_ARTIFACT_ID,
                    "parent_artifact_id": BASE_ARTIFACT_ID,
                    "exact_merge_evidence": str(exact_path),
                },
            },
        }
    )
    _atomic_json(checkpoint_path, checkpoint_manifest)
    _atomic_json(lineage_path, lineage_manifest)
    return {
        "status": "prepared",
        "checkpoint_manifest": str(checkpoint_path),
        "checkpoint_manifest_sha256": sha256_file(checkpoint_path),
        "lineage_manifest": str(lineage_path),
        "lineage_manifest_sha256": sha256_file(lineage_path),
        "artifact_ids": [BASE_ARTIFACT_ID, CHK1_ARTIFACT_ID],
        "test_rows": 33,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "docs/summary/20260810T124500Z/chk1_cp200_merge_for_chk2"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(prepare(repo_root=args.repo_root, output_dir=args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
