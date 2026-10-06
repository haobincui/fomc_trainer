"""Score the sealed 1993--2008 N128 x K10 stochastic CHK3 evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from jobs.eval import eval_chk3_external_holdout_stochastic_k10 as profile
from jobs.eval import eval_chk3_stochastic_bootstrap_generation as generation
from jobs.eval import score_chk3_stochastic_bootstrap as core
from open_r1.validator.loo_generation_spec import seal_manifest, validate_manifest_integrity


SCHEMA_VERSION = "chk3-external-holdout-stochastic-n128-k10-scorecard-v1"
RESULT_SCHEMA_VERSION = "chk3-external-holdout-stochastic-n128-k10-results-v1"
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20_260_813


def configure_profile() -> None:
    meetings = profile.configure_profile()
    core.REPLICATE_SEEDS = profile.REPLICATE_SEEDS
    core.REPLICATE_IDS = tuple(range(len(profile.REPLICATE_SEEDS)))
    core.EXPECTED_PROMPTS = 128
    core.EXPECTED_MEETINGS = len(meetings)
    core.EXPECTED_ROWS_PER_MODEL = 128 * len(profile.REPLICATE_SEEDS)
    core.EXPECTED_TOTAL_ROWS = core.EXPECTED_ROWS_PER_MODEL * len(core.MODEL_ORDER)
    core.BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
    core.BOOTSTRAP_SEED = BOOTSTRAP_SEED


def _wait_notice(reason: str, processes: list[dict[str, Any]]) -> None:
    print(
        core._canonical_json(
            {
                "status": "waiting_for_exclusive_gpu0_semantic_scoring",
                "reason": reason,
                "external_processes": processes,
            }
        ),
        file=sys.stderr,
        flush=True,
    )


def score(
    *,
    suite_manifest: Path,
    suite_sha256: str,
    sample_manifest: Path,
    sample_sha256: str,
    semantic_manifest: Path,
    output_dir: Path,
    semantic_batch_size: int,
) -> dict[str, Any]:
    configure_profile()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise core.StochasticBootstrapError(f"refusing to overwrite: {output_dir}")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID" or os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ) != "0":
        raise core.StochasticBootstrapError(
            "external K10 scoring requires CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES=0"
        )
    sample = core._load_sample_manifest(sample_manifest, sample_sha256)
    if (
        len(sample.get("samples", [])) != 128
        or sample.get("selection", {}).get("meeting_clusters") != 128
        or sample.get("selection", {}).get("normalized_identity_rows") != 0
    ):
        raise core.StochasticBootstrapError("external sample manifest denominator drift")
    suite_path = suite_manifest.expanduser().resolve()
    suite_binding = core._file_binding(suite_path, sealed=True)
    if suite_binding["sha256"] != suite_sha256:
        raise core.StochasticBootstrapError("suite SHA-256 differs from CLI binding")
    try:
        suite = generation.load_and_validate_suite(
            suite_path,
            sample_manifest_path=sample_manifest,
            sample_manifest_sha256=sample_sha256,
            expected_scope="formal_full_test",
        )
    except Exception as exc:
        raise core.StochasticBootstrapError(f"suite deep validation failed: {exc}") from exc
    raw_runs = suite.get("runs")
    if not isinstance(raw_runs, dict) or set(raw_runs) != set(core.MODEL_ORDER):
        raise core.StochasticBootstrapError("suite child inventory drift")
    runs = {model: dict(raw_runs[model]) for model in core.MODEL_ORDER}
    core.validate_run_matrix(runs)

    semantic_binding = core._file_binding(semantic_manifest, sealed=True)
    if (
        semantic_binding["sha256"] != core.SEMANTIC_MANIFEST_FILE_SHA256
        or semantic_binding["payload_sha256"]
        != core.SEMANTIC_MANIFEST_PAYLOAD_SHA256
    ):
        raise core.StochasticBootstrapError("semantic model manifest drift")
    with generation.exclusive_gpu0_lease(
        timeout_seconds=172_800,
        poll_seconds=30,
        on_wait=_wait_notice,
    ) as lease:
        try:
            bert, mpnet, semantic_provenance = (
                core.semantic_eval.load_formal_semantic_backends(
                    semantic_manifest_path=semantic_manifest,
                    batch_size=semantic_batch_size,
                    device="cuda:0",
                )
            )
        except Exception as exc:
            raise core.StochasticBootstrapError(
                f"cannot load pinned semantic backends: {exc}"
            ) from exc
        row_scores, failures, semantic_audit = core.build_scored_rows(
            runs=runs, bert=bert, mpnet=mpnet, formal=True
        )
        if generation._external_gpu0_compute_processes():
            raise core.StochasticBootstrapError(
                "external GPU0 process appeared during semantic scoring"
            )
        gpu_lease = dict(lease)

    full_records = [
        {
            "sample_id": str(item["sample_id"]),
            "meeting_id": str(item["meeting_id"]),
            "normalized_identity": bool(item["normalized_identity"]),
        }
        for item in sample["samples"]
    ]
    meeting_ids = sorted({item["meeting_id"] for item in full_records})
    view = {
        "view_id": "full_external_1993_2008",
        "role": "historical_external_post_selection_robustness",
        "sample_ids": [item["sample_id"] for item in full_records],
        "generation_prompts": 128,
        "semantic_prompts": 128,
        "meeting_ids": meeting_ids,
        "meetings": 128,
        "selection_anchor_rows": 0,
        "excluded_prompts": 0,
        "selection_meetings_excluded": 0,
        "inferential_conclusion_authorized": True,
    }
    view_result, draw_rows = core.bootstrap_view(
        view=view,
        full_sample_records=full_records,
        row_scores=row_scores,
        draws=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED,
    )
    results = seal_manifest(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "complete",
            "model_order": list(core.MODEL_ORDER),
            "metric_order": list(core.SIX_METRICS),
            "bootstrap_contract": {
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "meeting_equal_weighting": True,
                "replicate_resampling": "K10 with replacement within prompt",
                "meeting_resampling": "N128 with replacement",
                "paired_shared_indices": True,
                "confidence": core.CONFIDENCE,
                "sign_flip": "exact through N13; deterministic 100000-draw Monte Carlo for N128",
            },
            "view": view_result,
            "limitations": [
                "The meetings are historically external to checkpoint selection but may be present in base-model pretraining.",
                "References are deterministic source-grounded Minutes-style targets, not official Minutes excerpts.",
                "BERTScore and MPNet cannot override a hard-gate regression.",
                "One balanced Core8 topic is selected per meeting; conclusions do not cover every topic within every meeting.",
            ],
        }
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    core._fsync_directory(output_dir.parent)
    row_path = output_dir / "row_scores.jsonl"
    failure_path = output_dir / "failure_samples.jsonl"
    draw_path = output_dir / "bootstrap_draws.jsonl"
    results_path = output_dir / "bootstrap_results.json"
    core._write_new_jsonl(row_path, row_scores)
    core._write_new_jsonl(failure_path, failures)
    core._write_new_jsonl(draw_path, draw_rows)
    core._write_new_json(results_path, results)
    artifacts = {
        "row_scores": {**core._file_binding(row_path), "rows": len(row_scores)},
        "failure_samples": {**core._file_binding(failure_path), "rows": len(failures)},
        "bootstrap_draws": {**core._file_binding(draw_path), "rows": len(draw_rows)},
        "bootstrap_results": core._file_binding(results_path, sealed=True),
    }
    failure_counts: Counter[str] = Counter()
    for item in failures:
        failure_counts.update(item["preregistered_core_failures"])
    manifest = seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "operation": "external_n128_k10_six_metric_scoring_and_paired_bootstrap",
            "inputs": {
                "generation_suite": suite_binding,
                "sample_manifest": core._file_binding(sample_manifest, sealed=True),
                "semantic_manifest": semantic_binding,
                "run_manifests": {
                    model: dict(runs[model]["manifest_binding"])
                    for model in core.MODEL_ORDER
                },
            },
            "coverage": {
                "meetings": 128,
                "prompts": 128,
                "replicates": 10,
                "rows_per_model": 1280,
                "total_rows": len(row_scores),
                "input_truncation_rows": sum(
                    row.get("input_truncated") is not False
                    for model in core.MODEL_ORDER
                    for row in runs[model]["results"]
                ),
                "hard_gate_failure_rows": len(failures),
                "hard_gate_failure_counts": dict(sorted(failure_counts.items())),
            },
            "execution": {
                "semantic_device": "cuda:0",
                "semantic_batch_size": semantic_batch_size,
                "bootstrap_device": "cpu",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "python_executable": sys.executable,
                "network": False,
                "gpu0_lease": gpu_lease,
            },
            "semantic_models": semantic_provenance,
            "semantic_backend_audit": semantic_audit,
            "artifacts": artifacts,
            "sources": {
                "scorer": core._file_binding(Path(__file__)),
                "profile": core._file_binding(Path(profile.__file__)),
                "durable_runner": core._file_binding(Path(generation.__file__)),
                "bootstrap_core": core._file_binding(Path(core.__file__)),
            },
        }
    )
    manifest_path = output_dir / "manifest.json"
    core._write_new_json(manifest_path, manifest)
    validate_manifest_integrity(manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-manifest", required=True, type=Path)
    parser.add_argument("--suite-sha256", required=True)
    parser.add_argument("--sample-manifest", required=True, type=Path)
    parser.add_argument("--sample-sha256", required=True)
    parser.add_argument("--semantic-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--semantic-batch-size", type=int, default=8)
    args = parser.parse_args()
    try:
        result = score(
            suite_manifest=args.suite_manifest,
            suite_sha256=args.suite_sha256,
            sample_manifest=args.sample_manifest,
            sample_sha256=args.sample_sha256,
            semantic_manifest=args.semantic_manifest,
            output_dir=args.output_dir,
            semantic_batch_size=args.semantic_batch_size,
        )
    except core.StochasticBootstrapError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "payload_sha256": result["integrity"]["payload_sha256"],
                "rows": result["coverage"]["total_rows"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
