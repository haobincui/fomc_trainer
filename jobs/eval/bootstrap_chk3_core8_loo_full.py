"""Meeting-paired bootstrap for the sealed CHK3 Core8 LOO N128 score matrix.

This command is generation-free and semantic-scoring-free.  It validates the
sealed Core8 LOO score bundle, reconstructs the 128 paired meeting deltas for
all eight topics, both intervention arms, and both raw semantic metrics, then
applies one shared meeting-resampling plan to every one of the 32 cells.

The formal artifact uses 10,000 percentile-bootstrap draws and seed 20260815.
It quantifies sensitivity to the observed meeting panel only.  The source run
contains one greedy completion per meeting/arm (K=1), so this analysis cannot
estimate within-prompt stochastic-decoding variation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ID = "chk3-cp318-core8-loo-full-n128-1993-2008-v1"
SOURCE_SCORE_SCHEMA = "chk3-core8-loo-full-score-manifest-v1"
SOURCE_ROW_SCHEMA = "chk3-core8-loo-full-score-row-v1"
RESULT_SCHEMA = "chk3-core8-loo-full-meeting-bootstrap-results-v1"
MANIFEST_SCHEMA = "chk3-core8-loo-full-meeting-bootstrap-manifest-v1"
DRAW_SCHEMA = "chk3-core8-loo-full-meeting-bootstrap-draw-v1"
INDEX_PLAN_SCHEMA = "chk3-core8-loo-full-shared-meeting-index-plan-v1"

EXPECTED_MEETINGS = 128
EXPECTED_TOPICS = 8
EXPECTED_ROWS = 2_176
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_815
CONFIDENCE = 0.95
ARMS = ("exact_deletion", "token_matched_neutral")
METRICS = {
    "mpnet_cosine": "mpnet_cosine_raw",
    "bertscore_f1": "bertscore_f1_raw",
}

DEFAULT_SCORE_MANIFEST = ROOT / (
    "output/evaluation/main/"
    "chk3_cp318_core8_loo_full_n128_1993_2008_20260814_v1/"
    "score_v1/manifest.json"
)


class Core8LooBootstrapError(RuntimeError):
    """An immutable input or bootstrap contract failed validation."""


@dataclass(frozen=True)
class ScoreBundle:
    score_manifest: dict[str, Any]
    score_manifest_binding: dict[str, Any]
    row_scores: list[dict[str, Any]]
    row_scores_binding: dict[str, Any]
    generation_manifest: dict[str, Any]
    generation_manifest_binding: dict[str, Any]


@dataclass(frozen=True)
class BootstrapComputation:
    meeting_ids: tuple[str, ...]
    topics: tuple[str, ...]
    meeting_indices: np.ndarray
    index_plan_sha256: str
    index_plan_metadata: dict[str, Any]
    delta_vectors: dict[tuple[str, str, str], np.ndarray]
    draw_means: dict[tuple[str, str, str], np.ndarray]
    results: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise Core8LooBootstrapError(f"non-canonical JSON value: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise Core8LooBootstrapError(f"missing regular JSON file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Core8LooBootstrapError(f"cannot read JSON {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise Core8LooBootstrapError(f"JSON root is not an object: {resolved}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise Core8LooBootstrapError(f"missing regular JSONL file: {resolved}")
    result: list[dict[str, Any]] = []
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise Core8LooBootstrapError(
                        f"blank JSONL row at {resolved}:{line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise Core8LooBootstrapError(
                        f"JSONL row is not an object at {resolved}:{line_number}"
                    )
                result.append(value)
    except json.JSONDecodeError as exc:
        raise Core8LooBootstrapError(f"invalid JSONL {resolved}: {exc}") from exc
    return result


def _file_binding(
    path: Path, *, sealed: bool = False, rows: int | None = None
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise Core8LooBootstrapError(f"artifact is not a regular file: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if rows is not None:
        observed_rows = 0
        with resolved.open("rb") as handle:
            observed_rows = sum(1 for _ in handle)
        if observed_rows != rows:
            raise Core8LooBootstrapError(
                f"row count drift for {resolved}: expected={rows}, observed={observed_rows}"
            )
        result["rows"] = observed_rows
    if sealed:
        try:
            result["payload_sha256"] = validate_manifest_integrity(
                _read_json(resolved)
            )
        except Exception as exc:
            raise Core8LooBootstrapError(
                f"sealed manifest integrity failed for {resolved}: {exc}"
            ) from exc
    return result


def _binding_matches(
    recorded: Mapping[str, Any], observed: Mapping[str, Any], *, label: str
) -> None:
    for key in ("path", "sha256", "bytes", "rows", "payload_sha256"):
        if key in recorded and recorded.get(key) != observed.get(key):
            raise Core8LooBootstrapError(
                f"{label} binding drift for {key}: "
                f"recorded={recorded.get(key)!r}, observed={observed.get(key)!r}"
            )


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        with resolved.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise Core8LooBootstrapError(f"refusing to overwrite: {resolved}") from exc


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def load_score_bundle(
    *,
    score_manifest_path: Path,
    score_manifest_sha256: str,
    expected_meetings: int = EXPECTED_MEETINGS,
    expected_topics: int = EXPECTED_TOPICS,
) -> ScoreBundle:
    """Deep-validate the sealed score manifest and its immutable source rows."""

    score_path = score_manifest_path.expanduser().resolve()
    score_binding = _file_binding(score_path, sealed=True)
    if score_binding["sha256"] != score_manifest_sha256:
        raise Core8LooBootstrapError(
            "score manifest SHA-256 differs from the explicit CLI binding"
        )
    score_manifest = _read_json(score_path)
    if (
        score_manifest.get("schema_version") != SOURCE_SCORE_SCHEMA
        or score_manifest.get("status") != "complete"
        or score_manifest.get("evaluation_id") != EVALUATION_ID
    ):
        raise Core8LooBootstrapError("source score manifest contract drift")

    artifacts = score_manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(
        artifacts.get("row_scores"), Mapping
    ):
        raise Core8LooBootstrapError("source score manifest has no row_scores binding")
    recorded_rows = dict(artifacts["row_scores"])
    row_path = Path(str(recorded_rows.get("path", ""))).expanduser().resolve()
    expected_rows = expected_meetings * (1 + 2 * expected_topics)
    observed_rows = _file_binding(row_path, rows=expected_rows)
    _binding_matches(recorded_rows, observed_rows, label="row_scores")
    rows = _read_jsonl(row_path)

    recorded_run = score_manifest.get("run_manifest")
    if not isinstance(recorded_run, Mapping):
        raise Core8LooBootstrapError("source score manifest has no run binding")
    run_path = Path(str(recorded_run.get("path", ""))).expanduser().resolve()
    observed_run = _file_binding(run_path, sealed=True)
    _binding_matches(recorded_run, observed_run, label="generation run manifest")
    run_manifest = _read_json(run_path)
    if run_manifest.get("status") != "complete":
        raise Core8LooBootstrapError("generation run is not complete")
    generation_contract = run_manifest.get("generation_contract")
    if (
        not isinstance(generation_contract, Mapping)
        or generation_contract.get("mode") != "greedy"
        or generation_contract.get("do_sample") is not False
        or generation_contract.get("batch_size") != 1
    ):
        raise Core8LooBootstrapError("source generation is not the sealed K=1 greedy run")

    return ScoreBundle(
        score_manifest=score_manifest,
        score_manifest_binding=score_binding,
        row_scores=rows,
        row_scores_binding=observed_rows,
        generation_manifest=run_manifest,
        generation_manifest_binding=observed_run,
    )


def _finite_metric(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool):
        raise Core8LooBootstrapError(f"{key} must be numeric, not boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise Core8LooBootstrapError(f"{key} is not numeric") from exc
    if not math.isfinite(numeric):
        raise Core8LooBootstrapError(f"{key} is not finite")
    return numeric


def _prepare_delta_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_meetings: int,
    expected_topics: int,
) -> tuple[
    tuple[str, ...], tuple[str, ...], dict[tuple[str, str, str], np.ndarray]
]:
    if len(rows) != expected_meetings * (1 + 2 * expected_topics):
        raise Core8LooBootstrapError("score row count does not match the LOO matrix")
    by_key: dict[tuple[str, str, str | None], Mapping[str, Any]] = {}
    sample_ids: set[str] = set()
    meeting_references: dict[str, str] = {}
    topics: set[str] = set()
    meetings: set[str] = set()
    for row in rows:
        if row.get("schema_version") != SOURCE_ROW_SCHEMA:
            raise Core8LooBootstrapError("score-row schema drift")
        sample_id = row.get("sample_id")
        meeting_id = row.get("meeting_id")
        arm = row.get("arm")
        topic = row.get("intervention_topic")
        reference_sha = row.get("reference_minutes_sha256")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise Core8LooBootstrapError("score sample IDs are invalid or duplicated")
        if not isinstance(meeting_id, str) or not meeting_id:
            raise Core8LooBootstrapError("score row has no meeting ID")
        if not isinstance(reference_sha, str) or len(reference_sha) != 64:
            raise Core8LooBootstrapError("score row has no reference binding")
        prior_reference = meeting_references.setdefault(meeting_id, reference_sha)
        if prior_reference != reference_sha:
            raise Core8LooBootstrapError("reference changed within a meeting")
        if arm == "full":
            if topic is not None:
                raise Core8LooBootstrapError("full row unexpectedly names a topic")
        elif arm in ARMS:
            if not isinstance(topic, str) or not topic:
                raise Core8LooBootstrapError("intervention row has no topic")
            topics.add(topic)
        else:
            raise Core8LooBootstrapError(f"unknown LOO arm: {arm!r}")
        for metric_key in METRICS.values():
            _finite_metric(row, metric_key)
        key = (meeting_id, str(arm), topic if isinstance(topic, str) else None)
        if key in by_key:
            raise Core8LooBootstrapError(f"duplicate LOO cell: {key}")
        by_key[key] = row
        meetings.add(meeting_id)
        sample_ids.add(sample_id)

    meeting_ids = tuple(sorted(meetings))
    topic_ids = tuple(sorted(topics))
    if len(meeting_ids) != expected_meetings or len(topic_ids) != expected_topics:
        raise Core8LooBootstrapError(
            "meeting/topic denominator drift: "
            f"meetings={len(meeting_ids)}, topics={len(topic_ids)}"
        )
    expected_keys = {
        (meeting_id, "full", None) for meeting_id in meeting_ids
    } | {
        (meeting_id, arm, topic)
        for meeting_id in meeting_ids
        for arm in ARMS
        for topic in topic_ids
    }
    if set(by_key) != expected_keys:
        missing = sorted(expected_keys - set(by_key))[:5]
        extra = sorted(set(by_key) - expected_keys)[:5]
        raise Core8LooBootstrapError(
            f"LOO matrix coverage drift; missing={missing}, extra={extra}"
        )

    deltas: dict[tuple[str, str, str], np.ndarray] = {}
    for topic in topic_ids:
        for arm in ARMS:
            for metric, row_key in METRICS.items():
                vector = np.asarray(
                    [
                        _finite_metric(by_key[(meeting_id, "full", None)], row_key)
                        - _finite_metric(by_key[(meeting_id, arm, topic)], row_key)
                        for meeting_id in meeting_ids
                    ],
                    dtype=np.float64,
                )
                if vector.shape != (expected_meetings,) or not np.all(
                    np.isfinite(vector)
                ):
                    raise Core8LooBootstrapError("paired delta vector is incomplete")
                deltas[(topic, arm, metric)] = vector
    return meeting_ids, topic_ids, deltas


def _index_plan(
    *, meeting_ids: Sequence[str], draws: int, seed: int
) -> tuple[np.ndarray, str, dict[str, Any]]:
    if draws < 1 or not meeting_ids:
        raise Core8LooBootstrapError("bootstrap dimensions must be positive")
    if not 0 <= seed < 2**64:
        raise Core8LooBootstrapError("bootstrap seed must be an unsigned 64-bit integer")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(meeting_ids),
        size=(draws, len(meeting_ids)),
        dtype=np.uint16,
    )
    raw_indices = indices.astype("<u2", copy=False).tobytes(order="C")
    metadata = {
        "schema_version": INDEX_PLAN_SCHEMA,
        "draws": draws,
        "seed": seed,
        "meeting_ids": list(meeting_ids),
        "meeting_count": len(meeting_ids),
        "shape": list(indices.shape),
        "dtype": "uint16-le",
        "numpy_version": np.__version__,
        "numpy_bit_generator": type(rng.bit_generator).__name__,
        "raw_indices_sha256": hashlib.sha256(raw_indices).hexdigest(),
        "sampling": "meetings_with_replacement",
        "shared_across_all_cells": True,
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json(metadata).encode("utf-8"))
    digest.update(raw_indices)
    return indices, digest.hexdigest(), metadata


def _nested_cell_container() -> dict[str, Any]:
    return {}


def compute_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    expected_meetings: int = EXPECTED_MEETINGS,
    expected_topics: int = EXPECTED_TOPICS,
) -> BootstrapComputation:
    """Compute the shared-index, meeting-paired raw-delta bootstrap."""

    meeting_ids, topics, deltas = _prepare_delta_panel(
        rows,
        expected_meetings=expected_meetings,
        expected_topics=expected_topics,
    )
    indices, plan_sha, plan_metadata = _index_plan(
        meeting_ids=meeting_ids, draws=draws, seed=seed
    )
    draw_means = {
        key: vector[indices].mean(axis=1, dtype=np.float64)
        for key, vector in deltas.items()
    }
    alpha = 1.0 - CONFIDENCE
    cells = _nested_cell_container()
    for topic in topics:
        topic_result: dict[str, Any] = {}
        for arm in ARMS:
            arm_result: dict[str, Any] = {}
            for metric in METRICS:
                vector = deltas[(topic, arm, metric)]
                distribution = draw_means[(topic, arm, metric)]
                positive = int(np.count_nonzero(vector > 0.0))
                zero = int(np.count_nonzero(vector == 0.0))
                negative = int(np.count_nonzero(vector < 0.0))
                draw_positive = int(np.count_nonzero(distribution > 0.0))
                draw_zero = int(np.count_nonzero(distribution == 0.0))
                draw_share = draw_positive / draws
                arm_result[metric] = {
                    "point_estimate_mean_delta": float(vector.mean()),
                    "ci_lower": float(
                        np.quantile(distribution, alpha / 2.0, method="linear")
                    ),
                    "ci_upper": float(
                        np.quantile(
                            distribution, 1.0 - alpha / 2.0, method="linear"
                        )
                    ),
                    "confidence": CONFIDENCE,
                    "bootstrap_draws": draws,
                    "meeting_count": len(meeting_ids),
                    "meeting_delta_sample_std": float(vector.std(ddof=1)),
                    "meeting_delta_standard_error": float(
                        vector.std(ddof=1) / math.sqrt(len(vector))
                    ),
                    "positive_meeting_count": positive,
                    "zero_meeting_count": zero,
                    "negative_meeting_count": negative,
                    "positive_meeting_sign_share": positive / len(vector),
                    "bootstrap_draw_count_above_zero": draw_positive,
                    "bootstrap_draw_count_equal_zero": draw_zero,
                    "bootstrap_draw_count_below_zero": draws
                    - draw_positive
                    - draw_zero,
                    "bootstrap_draw_share_above_zero": draw_share,
                    "bootstrap_draw_share_monte_carlo_se": math.sqrt(
                        draw_share * (1.0 - draw_share) / draws
                    ),
                    "delta_direction": "full_minus_intervention",
                }
            topic_result[arm] = arm_result
        cells[topic] = topic_result

    k_adequacy = {
        "generation_replicates_per_meeting_arm": 1,
        "generation_mode": "greedy",
        "do_sample": False,
        "replicate_resampling_performed": False,
        "bootstrap_resampling_dimension": "meeting_only",
        "within_prompt_decoding_variance_estimable": False,
        "meeting_panel_resampling_estimable": True,
        "can_infer": (
            "Finite-panel sensitivity of paired mean deltas to resampling the "
            "128 observed meetings."
        ),
        "cannot_infer": (
            "Within-prompt stochastic-decoding variance, replicate stability, "
            "or uncertainty under a sampling decoder; K=1 greedy supplies no "
            "within-cell replicate variance."
        ),
        "bootstrap_draw_note": (
            "Increasing B controls Monte Carlo precision of the meeting bootstrap; "
            "it does not increase generation K or the effective meeting sample."
        ),
        "worst_case_monte_carlo_se_for_a_draw_share": 0.5 / math.sqrt(draws),
    }
    results = seal_manifest(
        {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "evaluation_id": EVALUATION_ID,
            "metric_order": list(METRICS),
            "arm_order": list(ARMS),
            "topic_order": list(topics),
            "delta_definition": "full_raw_semantic_score_minus_intervention_raw_semantic_score",
            "bootstrap_contract": {
                **plan_metadata,
                "index_plan_sha256": plan_sha,
                "confidence": CONFIDENCE,
                "interval": "two_sided_percentile",
                "quantile_method": "numpy_linear",
                "point_estimator": "arithmetic_mean_of_128_paired_meeting_deltas",
                "joint_cells": len(deltas),
                "multiple_testing_correction": None,
                "bootstrap_draw_share_above_zero_is_p_value": False,
            },
            "coverage": {
                "meetings": len(meeting_ids),
                "topics": len(topics),
                "intervention_arms": len(ARMS),
                "raw_semantic_metrics": len(METRICS),
                "joint_cells": len(deltas),
                "source_score_rows": len(rows),
            },
            "k_adequacy": k_adequacy,
            "cells": cells,
            "limitations": [
                "The bootstrap resamples only the 128 observed meetings and does not establish temporal or causal generalization.",
                "The source has K=1 greedy generation, so within-prompt decoding uncertainty is not represented.",
                "The 32 cellwise percentile intervals are not multiplicity-adjusted.",
                "Bootstrap draw share above zero is a resampling diagnostic, not a posterior probability or a p-value.",
                "Generation quality gates remain diagnostic-only; the bootstrap uses raw MPNet and BERTScore deltas.",
            ],
        }
    )
    return BootstrapComputation(
        meeting_ids=meeting_ids,
        topics=topics,
        meeting_indices=indices,
        index_plan_sha256=plan_sha,
        index_plan_metadata=plan_metadata,
        delta_vectors=deltas,
        draw_means=draw_means,
        results=results,
    )


def _validate_points_against_source_summary(
    computation: BootstrapComputation, score_manifest: Mapping[str, Any]
) -> None:
    summary = score_manifest.get("summary")
    if not isinstance(summary, Mapping) or not isinstance(
        summary.get("topic_deltas"), Mapping
    ):
        raise Core8LooBootstrapError("source score summary has no topic deltas")
    source = summary["topic_deltas"]
    for topic in computation.topics:
        for arm in ARMS:
            for metric, source_key in (
                ("mpnet_cosine", "mpnet_delta_raw_mean"),
                ("bertscore_f1", "bertscore_f1_delta_raw_mean"),
            ):
                try:
                    expected = float(source[topic][arm][source_key])
                except (KeyError, TypeError, ValueError) as exc:
                    raise Core8LooBootstrapError(
                        f"source summary is missing {topic}/{arm}/{source_key}"
                    ) from exc
                observed = float(
                    computation.results["cells"][topic][arm][metric][
                        "point_estimate_mean_delta"
                    ]
                )
                if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
                    raise Core8LooBootstrapError(
                        f"source point estimate mismatch for {topic}/{arm}/{metric}: "
                        f"source={expected}, recomputed={observed}"
                    )


def _write_draw_rows(path: Path, computation: BootstrapComputation) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        with resolved.open("x", encoding="utf-8") as handle:
            for draw_id in range(len(computation.meeting_indices)):
                cells: dict[str, Any] = {}
                for topic in computation.topics:
                    cells[topic] = {
                        arm: {
                            metric: float(
                                computation.draw_means[(topic, arm, metric)][draw_id]
                            )
                            for metric in METRICS
                        }
                        for arm in ARMS
                    }
                row = {
                    "schema_version": DRAW_SCHEMA,
                    "draw_id": draw_id,
                    "index_plan_sha256": computation.index_plan_sha256,
                    "meeting_indices": [
                        int(value) for value in computation.meeting_indices[draw_id]
                    ],
                    "cell_mean_deltas": cells,
                }
                handle.write(_canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise Core8LooBootstrapError(f"refusing to overwrite: {resolved}") from exc


def bootstrap_score_bundle(
    *,
    score_manifest_path: Path,
    score_manifest_sha256: str,
    output_dir: Path,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    expected_meetings: int = EXPECTED_MEETINGS,
    expected_topics: int = EXPECTED_TOPICS,
) -> dict[str, Any]:
    """Validate, bootstrap, persist, and seal a versioned result bundle."""

    resolved_output = output_dir.expanduser().resolve()
    if resolved_output.exists() or resolved_output.is_symlink():
        raise Core8LooBootstrapError(f"refusing to overwrite: {resolved_output}")
    bundle = load_score_bundle(
        score_manifest_path=score_manifest_path,
        score_manifest_sha256=score_manifest_sha256,
        expected_meetings=expected_meetings,
        expected_topics=expected_topics,
    )
    computation = compute_bootstrap(
        bundle.row_scores,
        draws=draws,
        seed=seed,
        expected_meetings=expected_meetings,
        expected_topics=expected_topics,
    )
    _validate_points_against_source_summary(computation, bundle.score_manifest)

    resolved_output.mkdir(parents=True, exist_ok=False)
    _fsync_directory(resolved_output.parent)
    results_path = resolved_output / "bootstrap_results.json"
    draws_path = resolved_output / "bootstrap_draws.jsonl"
    _write_new_json(results_path, computation.results)
    _write_draw_rows(draws_path, computation)
    artifacts = {
        "bootstrap_results": _file_binding(results_path, sealed=True),
        "bootstrap_draws": _file_binding(draws_path, rows=draws),
    }

    # Refuse publication if any immutable source changed while computing.
    if _file_binding(score_manifest_path, sealed=True) != bundle.score_manifest_binding:
        raise Core8LooBootstrapError("score manifest changed during bootstrap")
    if _file_binding(
        Path(bundle.row_scores_binding["path"]), rows=len(bundle.row_scores)
    ) != bundle.row_scores_binding:
        raise Core8LooBootstrapError("row_scores changed during bootstrap")
    if _file_binding(
        Path(bundle.generation_manifest_binding["path"]), sealed=True
    ) != bundle.generation_manifest_binding:
        raise Core8LooBootstrapError("generation manifest changed during bootstrap")

    manifest = seal_manifest(
        {
            "schema_version": MANIFEST_SCHEMA,
            "status": "complete",
            "created_at_utc": _utc_now(),
            "evaluation_id": EVALUATION_ID,
            "operation": "cpu_only_joint_meeting_paired_bootstrap",
            "inputs": {
                "score_manifest": bundle.score_manifest_binding,
                "row_scores": bundle.row_scores_binding,
                "generation_manifest": bundle.generation_manifest_binding,
            },
            "execution": {
                "device": "cpu",
                "generation_performed": False,
                "semantic_scoring_performed": False,
                "python_executable": sys.executable,
                "numpy_version": np.__version__,
                "network": False,
            },
            "bootstrap_contract": computation.results["bootstrap_contract"],
            "coverage": computation.results["coverage"],
            "k_adequacy": computation.results["k_adequacy"],
            "artifacts": artifacts,
            "sources": {"scorer": _file_binding(Path(__file__))},
            "limitations": computation.results["limitations"],
        }
    )
    manifest_path = resolved_output / "manifest.json"
    _write_new_json(manifest_path, manifest)
    validate_manifest_integrity(manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--score-manifest", type=Path, default=DEFAULT_SCORE_MANIFEST
    )
    parser.add_argument("--score-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        manifest = bootstrap_score_bundle(
            score_manifest_path=args.score_manifest,
            score_manifest_sha256=args.score_manifest_sha256,
            output_dir=args.output_dir,
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed,
        )
    except Core8LooBootstrapError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "draws": manifest["bootstrap_contract"]["draws"],
                "payload_sha256": manifest["integrity"]["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
