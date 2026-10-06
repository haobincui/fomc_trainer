"""Parallel CPU executor for the paper-CHK2 bootstrap.

The canonical statistical contract remains implemented in
``paper_chk2_text_similarity_scoring``.  This module accelerates only the
independent draw reductions and sign-flip reductions with a shared thread
pool, then lets the canonical implementation assemble and seal the artifacts.
The parallel and sequential paths are required to be byte-for-byte
deterministic.

Keeping the accelerator separate is intentional: completed score bundles bind
the canonical implementation by SHA, so adding an execution optimization must
not invalidate already sealed results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

import numpy as np

from jobs.eval import paper_chk2_text_similarity_scoring as canonical


MAX_BOOTSTRAP_WORKERS = 12
DEFAULT_BOOTSTRAP_WORKERS = min(6, max(1, os.cpu_count() or 1))
_PATCH_LOCK = RLock()
_CANONICAL_BUILD_STATISTICS = canonical.build_statistics


def _validated_workers(workers: int) -> int:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise canonical.PaperChk2SimilarityError(
            "bootstrap workers must be a positive integer"
        )
    if workers > MAX_BOOTSTRAP_WORKERS:
        raise canonical.PaperChk2SimilarityError(
            f"bootstrap workers must not exceed {MAX_BOOTSTRAP_WORKERS}"
        )
    return workers


def build_statistics_parallel(
    *,
    row_scores: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    workers: int = DEFAULT_BOOTSTRAP_WORKERS,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Run the canonical statistic builder with deterministic parallel kernels."""

    worker_count = _validated_workers(workers)
    if worker_count == 1:
        return canonical.build_statistics(row_scores=row_scores, samples=samples)

    # ``build_statistics`` resolves these three helpers from its module globals.
    # Patch them only while holding a process-local lock, and restore them even
    # when a worker fails.  The CLI is single-shot; the lock also makes direct
    # library use fail-safe rather than racy.
    with _PATCH_LOCK:
        original_draw_matrix = canonical._draw_matrix
        original_draw_rows = canonical._draw_rows
        original_sign_flip = canonical._sign_flip_p_value

        arrays = canonical._score_arrays(row_scores, samples)
        meeting_ids, by_meeting, meeting_positions_by_split = (
            canonical._meeting_layout(samples)
        )
        sample_positions_by_split = {
            split: [
                index
                for index, sample in enumerate(samples)
                if sample["split"] == split
            ]
            for split in canonical.SPLIT_ORDER
        }
        meeting_values = canonical._meeting_matrices(
            arrays, meeting_ids=meeting_ids, by_meeting=by_meeting
        )
        meeting_plan, row_plan, _replicate_plan, *_metadata = (
            canonical._bootstrap_plans(samples)
        )

        cell_keys = [
            (policy, model, metric)
            for policy in canonical.POLICY_ORDER
            for model in canonical.MODEL_ORDER
            for metric in canonical.METRIC_ORDER
        ]

        def compute_cell(
            key: tuple[str, str, str],
        ) -> tuple[np.ndarray, np.ndarray]:
            policy, model, metric = key
            meeting_draws = original_draw_matrix(
                meeting_values[policy][model][metric],
                plan=meeting_plan,
                positions_by_split=meeting_positions_by_split,
            )
            row_draws = original_draw_rows(
                arrays[policy][model][metric],
                plan=row_plan,
                sample_positions_by_split=sample_positions_by_split,
            )
            return meeting_draws, row_draws

        sign_keys: list[tuple[str, str]] = []
        sign_jobs: list[tuple[np.ndarray, str, str]] = []
        for contrast_id, before, after in canonical.CONTRASTS:
            for metric in canonical.METRIC_ORDER:
                meeting_delta = (
                    meeting_values[canonical.PRIMARY_POLICY][after][metric].mean(
                        axis=1
                    )
                    - meeting_values[canonical.PRIMARY_POLICY][before][metric].mean(
                        axis=1
                    )
                )
                sign_keys.append((contrast_id, metric))
                sign_jobs.append((meeting_delta, contrast_id, metric))

        def compute_sign_flip(
            job: tuple[np.ndarray, str, str],
        ) -> tuple[float, int, int]:
            differences, contrast_id, metric = job
            return original_sign_flip(
                differences, contrast_id=contrast_id, metric=metric
            )

        # ``executor.map`` preserves the predefined canonical job order.  RNG
        # plans are generated once above and every sign-flip owns a content-
        # addressed seed, so scheduling cannot affect any output value.
        with ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="paper-chk2-bootstrap"
        ) as executor:
            cell_results = list(executor.map(compute_cell, cell_keys))
            sign_results = list(executor.map(compute_sign_flip, sign_jobs))

        meeting_cursor = 0
        row_cursor = 0
        sign_calls: list[tuple[str, str]] = []
        sign_result_by_key = dict(zip(sign_keys, sign_results, strict=True))

        def draw_matrix(
            matrix: np.ndarray,
            *,
            plan: Mapping[str, np.ndarray],
            positions_by_split: Mapping[str, Sequence[int]],
        ) -> np.ndarray:
            del matrix, plan, positions_by_split
            nonlocal meeting_cursor
            result = cell_results[meeting_cursor][0]
            meeting_cursor += 1
            return result

        def draw_rows(
            matrix: np.ndarray,
            *,
            plan: Mapping[str, np.ndarray],
            sample_positions_by_split: Mapping[str, Sequence[int]],
        ) -> np.ndarray:
            del matrix, plan, sample_positions_by_split
            nonlocal row_cursor
            result = cell_results[row_cursor][1]
            row_cursor += 1
            return result

        def sign_flip(
            differences: Sequence[float], *, contrast_id: str, metric: str
        ) -> tuple[float, int, int]:
            del differences
            key = (contrast_id, metric)
            sign_calls.append(key)
            return sign_result_by_key[key]

        canonical._draw_matrix = draw_matrix
        canonical._draw_rows = draw_rows
        canonical._sign_flip_p_value = sign_flip
        try:
            statistics = _CANONICAL_BUILD_STATISTICS(
                row_scores=row_scores, samples=samples
            )
            if meeting_cursor != len(cell_keys) or row_cursor != len(cell_keys):
                raise canonical.PaperChk2SimilarityError(
                    "canonical bootstrap cell-call order drift"
                )
            if sign_calls != sign_keys:
                raise canonical.PaperChk2SimilarityError(
                    "canonical sign-flip call order drift"
                )
            return statistics
        finally:
            canonical._draw_matrix = original_draw_matrix
            canonical._draw_rows = original_draw_rows
            canonical._sign_flip_p_value = original_sign_flip


def validate_command(*, output_dir: Path, workers: int) -> dict[str, Any]:
    """Deep-replay a sealed bundle using the parallel deterministic executor."""

    worker_count = _validated_workers(workers)
    if worker_count == 1:
        return canonical.validate_command(output_dir=output_dir)
    with _PATCH_LOCK:
        original_builder = canonical.build_statistics

        def parallel_builder(
            *,
            row_scores: Sequence[Mapping[str, Any]],
            samples: Sequence[Mapping[str, Any]],
        ):
            return build_statistics_parallel(
                row_scores=row_scores, samples=samples, workers=worker_count
            )

        canonical.build_statistics = parallel_builder
        try:
            return canonical.validate_command(output_dir=output_dir)
        finally:
            canonical.build_statistics = original_builder


def bootstrap_command(
    *,
    generation_rows: Path,
    samples: Path,
    evaluation_manifest: Path,
    semantic_manifest: Path,
    output_dir: Path,
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    """Drop-in bootstrap command with parallel deterministic kernels."""

    worker_count = _validated_workers(workers)
    inputs = canonical._load_inputs(
        generation_rows=generation_rows,
        samples=samples,
        evaluation_manifest=evaluation_manifest,
        semantic_manifest=semantic_manifest,
    )
    output = output_dir.expanduser().resolve()
    if output.is_symlink() or not output.is_dir():
        raise canonical.PaperChk2SimilarityError(
            "score output directory does not exist"
        )
    row_path = output / "row_scores.jsonl"
    row_scores = canonical._read_jsonl(row_path, label="row scores")
    canonical.validate_row_scores(
        row_scores,
        samples=inputs["samples"],
        generations=inputs["generations"],
        score_contract_sha256=inputs["score_contract"]["sha256"],
    )
    manifest_path = output / "score_manifest.json"
    if manifest_path.exists() or manifest_path.is_symlink():
        if not resume:
            raise canonical.PaperChk2SimilarityError(
                "score manifest already exists; pass --resume to validate it"
            )
        result = validate_command(output_dir=output, workers=worker_count)
        return {**result, "bootstrap_workers_requested": worker_count}

    started = time.perf_counter()
    statistics = build_statistics_parallel(
        row_scores=row_scores,
        samples=inputs["samples"],
        workers=worker_count,
    )
    for name, (text, _rows, _sealed) in canonical._artifact_texts(statistics).items():
        canonical._write_or_verify_text(output / name, text, resume=resume)
    manifest = canonical._manifest_payload(
        output_dir=output, inputs=inputs, row_scores=row_scores
    )
    canonical._write_or_verify_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        resume=False,
    )
    return {
        **manifest,
        "parallel_execution": {
            "workers": worker_count,
            "elapsed_seconds": time.perf_counter() - started,
            "canonical_byte_equivalence_validated_on_resume": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    canonical._common_arguments(parser)
    parser.add_argument("--workers", type=int, default=DEFAULT_BOOTSTRAP_WORKERS)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = bootstrap_command(
            generation_rows=args.generation_rows,
            samples=args.samples,
            evaluation_manifest=args.evaluation_manifest,
            semantic_manifest=args.semantic_manifest,
            output_dir=args.output_dir,
            workers=args.workers,
            resume=args.resume,
        )
    except canonical.PaperChk2SimilarityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(canonical._canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
