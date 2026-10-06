"""Extend the sealed CHK3 Core8 LOO analysis from stochastic K=5 to K=10.

This is a create-only, incremental scorer.  It reuses the sealed raw semantic
scores for global replicates 0--4, scores only the five new BF16-vLLM
generations, maps their local replicate IDs 0--4 to global IDs 5--9, and then
recomputes the meeting-paired hierarchical bootstrap on the combined panel.

The three stages are intentionally separated:

``prepare``
    Bind and validate the old K=5 generation/score release, the new K6--K10
    generation release, the pinned semantic model inventory, the local-to-
    global replicate map, the bootstrap contract, and the excluded incomplete
    historical NF4 K=10 attempt before reading any new semantic score.
``score-metric``
    Run one pinned semantic backend on only the 10,880 incremental rows.
``finalize``
    Align the old and new scores by validated keys, publish 21,760 combined
    rows, and run the shared-index K=10 hierarchical paired bootstrap.

For each bootstrap draw, 128 meetings are sampled with replacement.  Within
every sampled meeting occurrence, ten paired replicate indices are sampled
with replacement and shared across all topics, arms, and semantic metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
import statistics
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from jobs.eval import score_chk3_core8_loo_vllm_k5 as k5
from open_r1.validator.loo_generation_spec import seal_manifest, validate_manifest_integrity

try:  # The increment profile is developed independently but is mandatory at run time.
    from jobs.eval import eval_chk3_core8_loo_vllm_k10_increment_dual_dp1 as increment
except ImportError:  # pragma: no cover - permits pure unit tests during staged development.
    increment = None


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ID = "chk3-cp318-core8-loo-n128-vllm-k10-combined-v1"
SCORE_EVALUATION_ID = f"{EVALUATION_ID}-raw-semantic-b10000-v1"

PLAN_SCHEMA = "chk3-core8-loo-vllm-k10-incremental-score-plan-v1"
METRIC_ROW_SCHEMA = "chk3-core8-loo-vllm-k10-increment-semantic-metric-row-v1"
METRIC_MANIFEST_SCHEMA = "chk3-core8-loo-vllm-k10-increment-semantic-metric-manifest-v1"
SCORE_ROW_SCHEMA = "chk3-core8-loo-vllm-k10-combined-score-row-v1"
FAILURE_ROW_SCHEMA = "chk3-core8-loo-vllm-k10-generation-diagnostic-failure-v1"
MEETING_CELL_SCHEMA = "chk3-core8-loo-vllm-k10-meeting-cell-delta-v1"
INDEX_PLAN_SCHEMA = "chk3-core8-loo-vllm-k10-shared-hierarchical-index-plan-v1"
BOOTSTRAP_DRAW_SCHEMA = "chk3-core8-loo-vllm-k10-bootstrap-draw-v1"
RESULT_SCHEMA = "chk3-core8-loo-vllm-k10-bootstrap-results-v1"
MANIFEST_SCHEMA = "chk3-core8-loo-vllm-k10-score-manifest-v1"

EXPECTED_MEETINGS = 128
EXPECTED_VARIANTS = 17
OLD_REPLICATES = 5
INCREMENT_REPLICATES = 5
EXPECTED_REPLICATES = 10
EXPECTED_INCREMENT_ROWS = EXPECTED_MEETINGS * EXPECTED_VARIANTS * INCREMENT_REPLICATES
EXPECTED_ROWS = EXPECTED_MEETINGS * EXPECTED_VARIANTS * EXPECTED_REPLICATES
EXPECTED_PAIRED_BLOCKS = EXPECTED_MEETINGS * EXPECTED_REPLICATES
GLOBAL_REPLICATE_OFFSET = 5
OLD_REPLICATE_SEEDS = tuple(k5.REPLICATE_SEEDS)
INCREMENT_REPLICATE_SEEDS = (25260811, 26260811, 27260811, 28260811, 29260811)
REPLICATE_SEEDS = OLD_REPLICATE_SEEDS + INCREMENT_REPLICATE_SEEDS
LOCAL_REPLICATE_IDS = tuple(range(INCREMENT_REPLICATES))
GLOBAL_REPLICATE_IDS = tuple(range(GLOBAL_REPLICATE_OFFSET, EXPECTED_REPLICATES))

TOPICS = tuple(k5.TOPICS)
ARMS = tuple(k5.ARMS)
METRICS = tuple(k5.METRICS)
METRIC_WORKERS = tuple(k5.METRIC_WORKERS)

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_824
CONFIDENCE = 0.95
K10_K9_POINT_DRIFT_THRESHOLD = float(k5.K5_K4_POINT_DRIFT_THRESHOLD)
K10_K9_CI_ENDPOINT_DRIFT_THRESHOLD = float(k5.K5_K4_CI_ENDPOINT_DRIFT_THRESHOLD)
LORO_MAX_DRIFT_THRESHOLD = float(k5.LORO_MAX_DRIFT_THRESHOLD)
DECODING_VARIANCE_SHARE_THRESHOLD = float(k5.DECODING_VARIANCE_SHARE_THRESHOLD)
K_ADEQUACY_FREEZE_STATUS = "frozen_before_any_incremental_K6_to_K10_semantic_scores_were_computed_or_read"
K_INCREASE_RULES = (
    "any cell has absolute K10-minus-K9 point drift above 0.005",
    "any cell has either absolute K10-minus-K9 CI endpoint drift above 0.010",
    "any cell has leave-one-replicate max absolute point drift above 0.010",
    "any cell has estimated decoding variance share above 0.20",
    "any K10-zero-excluding cell is not same-direction zero-excluding at K8 and K9",
    "any K10-zero-excluding cell reverses point sign at K8, K9, K10, or leave-one-replicate",
    "replicate coverage is incomplete",
)

OLD_ROOT = ROOT / "output/evaluation/main/chk3_cp318_core8_loo_vllm_k5_n128_1993_2008_20260817_v1"
DEFAULT_OLD_GENERATION_MANIFEST = OLD_ROOT / "generation_formal_n2176_k5_v1/chk3/manifest.json"
DEFAULT_OLD_GENERATION_MANIFEST_SHA256 = "23af367ba52ddebbc76e47b3f119f73fcbace052d10f77d117a8741b9fdd40da"
DEFAULT_OLD_SCORE_MANIFEST = OLD_ROOT / "score_raw_semantic_dual_gpu_b10000_v2/manifest.json"
DEFAULT_OLD_SCORE_MANIFEST_SHA256 = "4df87c36a7215e86fc22c1a33b96b700344f523254c035e5efaf5d00383c5fee"

DEFAULT_ROOT = ROOT / "output/evaluation/main/chk3_cp318_core8_loo_vllm_k10_n128_1993_2008_20260824_v1"
DEFAULT_INCREMENT_MANIFEST = DEFAULT_ROOT / "generation_formal_increment_k6_k10_n2176_v1/chk3/manifest.json"
DEFAULT_INCREMENT_COHORT = DEFAULT_ROOT / "preparation_increment_k6_k10_v1/cohort_n2176_increment_k6_k10.v1.json"
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "score_incremental_reuse_k5_raw_semantic_b10000_v1"
DEFAULT_SEMANTIC_MANIFEST = k5.DEFAULT_SEMANTIC_MANIFEST
DEFAULT_SEMANTIC_MANIFEST_SHA256 = k5.DEFAULT_SEMANTIC_MANIFEST_SHA256
DEFAULT_LEGACY_NF4_ROOT = ROOT / "output/evaluation/main/chk3_cp318_core8_loo_stochastic_k10_n128_1993_2008_20260815_v1"
SCORE_LAUNCHER = ROOT / "run/score_chk3_core8_loo_vllm_k10_dual_gpu.sh"

OLD_K5_IMPLEMENTATION_COMPATIBILITY_ROLES = (
    "semantic_wrapper",
    "semantic_long_text_engine",
    "native_generation_and_gate_contract",
    "native_probe_gate_engine",
)


class LooK10ScoreError(RuntimeError):
    """A sealed input, mapping, semantic result, or statistic failed closed."""


@dataclass(frozen=True)
class BootstrapPlan:
    meeting_indices: np.ndarray
    replicate_indices: np.ndarray
    prefix_replicate_indices: dict[int, np.ndarray]
    sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BootstrapComputation:
    plan: BootstrapPlan
    cell_draws: dict[tuple[str, str, str], np.ndarray]
    prefix_cell_draws: dict[tuple[str, str, str, int], np.ndarray]
    meeting_only_draws: dict[tuple[str, str, str], np.ndarray]
    results: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


_canonical = k5._canonical
_sha256_file = k5._sha256_file
_read_json = k5._read_json
_iter_jsonl = k5._iter_jsonl
_file_binding = k5._file_binding
_assert_binding = k5._assert_binding
_write_new_json = k5._write_new_json
_write_new_jsonl = k5._write_new_jsonl
_publish_new_file = k5._publish_new_file
_fsync_directory = k5._fsync_directory


def _require_increment_profile() -> Any:
    if increment is None:
        raise LooK10ScoreError("increment generation profile is not importable")
    expected = {
        "GLOBAL_REPLICATE_OFFSET": GLOBAL_REPLICATE_OFFSET,
        "REPLICATE_SEEDS": INCREMENT_REPLICATE_SEEDS,
        "EXPECTED_PROMPTS": EXPECTED_MEETINGS * EXPECTED_VARIANTS,
        "CASES_PER_MODEL": EXPECTED_INCREMENT_ROWS,
    }
    for name, value in expected.items():
        observed = getattr(increment, name, None)
        if name == "REPLICATE_SEEDS" and observed is not None:
            observed = tuple(observed)
        if observed != value:
            raise LooK10ScoreError(
                f"increment profile constant drift at {name}: expected={value!r}, observed={observed!r}"
            )
    return increment


def _source_bundle() -> dict[str, dict[str, Any]]:
    profile = _require_increment_profile()
    return {
        "k10_incremental_scorer": _file_binding(Path(__file__)),
        "sealed_k5_scorer": _file_binding(Path(k5.__file__)),
        "increment_generation_profile": _file_binding(Path(profile.__file__)),
        "semantic_wrapper": _file_binding(Path(k5.semantic.__file__)),
        "semantic_long_text_engine": _file_binding(Path(k5.semantic.semantic_eval.__file__)),
        "native_generation_and_gate_contract": _file_binding(
            Path(k5.semantic.native_eval.__file__)
        ),
        "native_probe_gate_engine": _file_binding(
            Path(k5.semantic.native_eval.native_probe.__file__)
        ),
        "gpu_lease_runtime": _file_binding(Path(k5.gpu_runtime.__file__)),
        "score_launcher": _file_binding(SCORE_LAUNCHER),
    }


def _assert_source_bundle(value: Any) -> dict[str, dict[str, Any]]:
    current = _source_bundle()
    if not isinstance(value, Mapping) or set(value) != set(current):
        raise LooK10ScoreError("implementation source inventory drift")
    for role, binding in value.items():
        if not isinstance(binding, Mapping):
            raise LooK10ScoreError(f"invalid source binding: {role}")
        _assert_binding(binding)
    if dict(value) != current:
        raise LooK10ScoreError("implementation source hashes changed after planning")
    return {key: dict(value[key]) for key in sorted(value)}


def _validate_old_k5_source_compatibility(
    old_score_manifest: Mapping[str, Any],
    current_k10_sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Revalidate the complete K5 source inventory and the reused code paths.

    The first check deliberately delegates to the frozen K5 validator so a
    changed K5 scorer, generation validator, GPU runtime, semantic engine, or
    gate implementation blocks reuse.  The second check makes the narrower
    cross-release requirement explicit: the semantic and hard-gate code used
    on new rows must be byte-identical to the implementations sealed in K5.
    """

    old_sources = old_score_manifest.get("sources")
    if not isinstance(old_sources, Mapping):
        raise LooK10ScoreError("sealed K5 score manifest lacks its source inventory")
    try:
        validated = k5._assert_source_bundle(old_sources)
    except Exception as exc:
        raise LooK10ScoreError(
            f"sealed K5 implementation source revalidation failed: {exc}"
        ) from exc
    for role in OLD_K5_IMPLEMENTATION_COMPATIBILITY_ROLES:
        if old_sources.get(role) != current_k10_sources.get(role):
            raise LooK10ScoreError(
                f"current K10 implementation is not byte-identical to sealed K5 role: {role}"
            )
    return validated


def _expected_variant(variant_rank: int) -> tuple[str, str | None]:
    return k5._expected_variant(variant_rank)


def _combined_absolute_index(meeting_rank: int, variant_rank: int, global_replicate_id: int) -> int:
    return ((meeting_rank * EXPECTED_VARIANTS + variant_rank) * EXPECTED_REPLICATES + global_replicate_id)


def _combined_tuple_key(row: Mapping[str, Any], global_replicate_id: int) -> dict[str, Any]:
    meeting_rank = int(row["meeting_rank"])
    variant_rank = int(row["variant_rank"])
    return {
        "absolute_case_index": _combined_absolute_index(meeting_rank, variant_rank, global_replicate_id),
        "meeting_id": row.get("meeting_id"),
        "meeting_rank": meeting_rank,
        "variant_rank": variant_rank,
        "arm": row.get("arm"),
        "intervention_topic": row.get("intervention_topic"),
        "replicate_id": global_replicate_id,
        "replicate_seed": row.get("replicate_seed"),
        "row_seed": row.get("row_seed"),
    }


def _old_score_inventory(manifest_path: Path, manifest_sha256: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    binding = _file_binding(manifest_path, sealed=True)
    if binding["sha256"] != manifest_sha256 or binding["sha256"] != DEFAULT_OLD_SCORE_MANIFEST_SHA256:
        raise LooK10ScoreError("old K5 score manifest binding drift")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema_version") != k5.MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("coverage", {}).get("rows") != k5.EXPECTED_ROWS
        or manifest.get("coverage", {}).get("semantic_scored_rows") != k5.EXPECTED_ROWS
        or manifest.get("coverage", {}).get("replicates") != OLD_REPLICATES
    ):
        raise LooK10ScoreError("old K5 score release contract drift")
    artifact = manifest.get("artifacts", {}).get("row_scores")
    if not isinstance(artifact, Mapping):
        raise LooK10ScoreError("old K5 score manifest lacks row_scores")
    path = _assert_binding(artifact, rows=k5.EXPECTED_ROWS)
    if path.parent != manifest_path.expanduser().resolve().parent:
        raise LooK10ScoreError("old K5 row scores escaped the sealed score root")
    rows = list(_iter_jsonl(path))
    return manifest, binding, rows


def validate_old_score_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != EXPECTED_INCREMENT_ROWS:
        raise LooK10ScoreError("old K5 row-score denominator drift")
    digest = hashlib.sha256()
    sample_map: dict[tuple[int, int], tuple[str, str, str]] = {}
    paired: dict[tuple[int, int], tuple[int, int]] = {}
    for position, row in enumerate(rows):
        if row.get("schema_version") != k5.SCORE_ROW_SCHEMA:
            raise LooK10ScoreError(f"old K5 score-row schema drift at {position}")
        meeting = row.get("meeting_rank")
        variant = row.get("variant_rank")
        replicate = row.get("replicate_id")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (meeting, variant, replicate)):
            raise LooK10ScoreError(f"old K5 coordinate type drift at {position}")
        expected_position = ((meeting * EXPECTED_VARIANTS + variant) * OLD_REPLICATES + replicate)
        arm, topic = _expected_variant(variant)
        if (
            position != expected_position
            or not 0 <= meeting < EXPECTED_MEETINGS
            or not 0 <= variant < EXPECTED_VARIANTS
            or not 0 <= replicate < OLD_REPLICATES
            or row.get("replicate_seed") != OLD_REPLICATE_SEEDS[replicate]
            or row.get("arm") != arm
            or row.get("intervention_topic") != topic
            or row.get("tuple_key", {}).get("absolute_case_index") != position
        ):
            raise LooK10ScoreError(f"old K5 canonical coverage drift at {position}")
        key = _combined_tuple_key(row, replicate)
        digest.update((_canonical(key) + "\n").encode())
        sample = (str(row.get("meeting_id")), str(row.get("sample_id")), str(row.get("reference_minutes_sha256")))
        if not all(sample):
            raise LooK10ScoreError(f"old K5 sample binding missing at {position}")
        prior = sample_map.setdefault((meeting, variant), sample)
        if prior != sample:
            raise LooK10ScoreError(f"old K5 sample identity changed across replicates at {position}")
        pair = (int(row.get("row_seed")), int(row.get("replicate_seed")))
        prior_pair = paired.setdefault((meeting, replicate), pair)
        if prior_pair != pair:
            raise LooK10ScoreError(f"old K5 paired seed drift at {position}")
        scores = row.get("raw_semantic_scores")
        if not isinstance(scores, Mapping) or any(
            not math.isfinite(float(scores[name])) for name in ("mpnet_cosine", "bertscore_f1")
        ):
            raise LooK10ScoreError(f"old K5 semantic score drift at {position}")
    return {
        "rows": len(rows),
        "replicates": OLD_REPLICATES,
        "replicate_ids": list(range(OLD_REPLICATES)),
        "replicate_seeds": list(OLD_REPLICATE_SEEDS),
        "paired_blocks": len(paired),
        "combined_tuple_key_sha256_in_source_order": digest.hexdigest(),
        "sample_map": sample_map,
    }


def _increment_local_tuple_key(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "absolute_case_index": row.get("absolute_case_index"),
        "meeting_id": row.get("meeting_id"),
        "meeting_rank": row.get("meeting_rank"),
        "variant_rank": row.get("variant_rank"),
        "arm": row.get("arm"),
        "intervention_topic": row.get("intervention_topic"),
        "replicate_id": row.get("replicate_id"),
        "replicate_seed": row.get("replicate_seed"),
        "row_seed": row.get("row_seed"),
    }


def validate_increment_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    old_sample_map: Mapping[tuple[int, int], tuple[str, str, str]],
) -> dict[str, Any]:
    profile = _require_increment_profile()
    if len(rows) != EXPECTED_INCREMENT_ROWS:
        raise LooK10ScoreError("increment generation denominator drift")
    local_digest = hashlib.sha256()
    global_digest = hashlib.sha256()
    paired: dict[tuple[int, int], tuple[int, str]] = {}
    sample_map: dict[tuple[int, int], tuple[str, str, str]] = {}
    empty = 0
    finish: Counter[str] = Counter()
    for position, row in enumerate(rows):
        if row.get("schema_version") != profile.ROW_SCHEMA:
            raise LooK10ScoreError(f"increment generation schema drift at {position}")
        if row.get("evaluation_id") != profile.EVALUATION_ID or row.get("model_id") != "chk3":
            raise LooK10ScoreError(f"increment generation identity drift at {position}")
        meeting = row.get("meeting_rank")
        variant = row.get("variant_rank")
        local = row.get("replicate_id")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (meeting, variant, local)):
            raise LooK10ScoreError(f"increment coordinate type drift at {position}")
        global_id = GLOBAL_REPLICATE_OFFSET + local
        expected_position = ((meeting * EXPECTED_VARIANTS + variant) * INCREMENT_REPLICATES + local)
        arm, topic = _expected_variant(variant)
        if (
            position != expected_position
            or row.get("absolute_case_index") != position
            or not 0 <= meeting < EXPECTED_MEETINGS
            or not 0 <= variant < EXPECTED_VARIANTS
            or local not in LOCAL_REPLICATE_IDS
            or row.get("local_replicate_id") != local
            or row.get("global_replicate_id") != global_id
            or row.get("replicate_seed") != INCREMENT_REPLICATE_SEEDS[local]
            or row.get("paired_block_id") != meeting * INCREMENT_REPLICATES + local
            or row.get("arm") != arm
            or row.get("intervention_topic") != topic
            or row.get("input_truncated") is not False
        ):
            raise LooK10ScoreError(f"increment canonical/mapping drift at {position}")
        answer = row.get("answer")
        reference = row.get("reference_minutes")
        if not isinstance(answer, str) or not isinstance(reference, str) or not reference:
            raise LooK10ScoreError(f"increment semantic text missing at {position}")
        sample = (
            str(row.get("meeting_id")),
            str(row.get("variant_sample_id", row.get("sample_id"))),
            str(row.get("reference_minutes_sha256")),
        )
        if sample != old_sample_map.get((meeting, variant)):
            raise LooK10ScoreError(f"old/new sample or reference mismatch at {position}")
        sample_map[(meeting, variant)] = sample
        paired_value = (int(row.get("row_seed")), str(row.get("paired_seed_key")))
        prior = paired.setdefault((meeting, local), paired_value)
        if prior != paired_value:
            raise LooK10ScoreError(f"increment common-random-number drift at {position}")
        local_key = _increment_local_tuple_key(row)
        global_key = _combined_tuple_key(row, global_id)
        local_digest.update((_canonical(local_key) + "\n").encode())
        global_digest.update((_canonical(global_key) + "\n").encode())
        empty += not answer.strip()
        finish[str(row.get("finish_reason"))] += 1
    if len(sample_map) != EXPECTED_MEETINGS * EXPECTED_VARIANTS or len(paired) != EXPECTED_MEETINGS * INCREMENT_REPLICATES:
        raise LooK10ScoreError("increment prompt/pair matrix is incomplete")
    return {
        "rows": len(rows),
        "replicates": INCREMENT_REPLICATES,
        "local_replicate_ids": list(LOCAL_REPLICATE_IDS),
        "global_replicate_offset": GLOBAL_REPLICATE_OFFSET,
        "global_replicate_ids": list(GLOBAL_REPLICATE_IDS),
        "replicate_seeds": list(INCREMENT_REPLICATE_SEEDS),
        "paired_blocks": len(paired),
        "empty_answer_rows": empty,
        "finish_reason_counts": dict(sorted(finish.items())),
        "local_tuple_key_sha256": local_digest.hexdigest(),
        "mapped_global_tuple_key_sha256_in_source_order": global_digest.hexdigest(),
    }


def _legacy_nf4_binding(root: Path) -> dict[str, Any]:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise LooK10ScoreError(f"legacy NF4 root is missing/not a regular directory: {resolved}")
    files = [path for path in sorted(resolved.rglob("*")) if path.is_file() and not path.is_symlink()]
    if not files:
        raise LooK10ScoreError("legacy NF4 inventory is empty")
    bindings = {str(path.relative_to(resolved)): _file_binding(path) for path in files}
    launch_path = resolved / "generation_v1/launch.json"
    state_path = resolved / "generation_v1/state.progress.v1.json"
    launch = _read_json(launch_path)
    state = _read_json(state_path)
    if (
        launch.get("generation", {}).get("load_in_4bit") is not True
        or launch.get("generation", {}).get("bnb_4bit_quant_type") != "nf4"
        or state.get("completed_cases", EXPECTED_ROWS) >= EXPECTED_ROWS
        or state.get("status") == "complete"
    ):
        raise LooK10ScoreError("legacy root is not the expected incomplete NF4 K10 attempt")
    return {
        "role": "excluded_incomplete_historical_nf4_k10_attempt",
        "included_in_generation_panel": False,
        "included_in_semantic_scoring": False,
        "included_in_point_estimates_or_intervals": False,
        "reason": "incomplete Transformers NF4 run; decoding and numerical distribution differs from the BF16-vLLM release",
        "root": str(resolved),
        "observed_completed_cases": state.get("completed_cases"),
        "expected_cases": state.get("expected_cases"),
        "files": bindings,
    }


def _assert_legacy_nf4_binding(record: Any) -> None:
    if not isinstance(record, Mapping) or record.get("included_in_point_estimates_or_intervals") is not False:
        raise LooK10ScoreError("legacy NF4 exclusion record drift")
    files = record.get("files")
    if not isinstance(files, Mapping) or not files:
        raise LooK10ScoreError("legacy NF4 file inventory missing")
    for binding in files.values():
        _assert_binding(binding)


def _combined_key_digest(old_rows: Sequence[Mapping[str, Any]], increment_rows: Sequence[Mapping[str, Any]]) -> str:
    keys = [(_combined_absolute_index(int(row["meeting_rank"]), int(row["variant_rank"]), int(row["replicate_id"])), _combined_tuple_key(row, int(row["replicate_id"]))) for row in old_rows]
    keys += [(_combined_absolute_index(int(row["meeting_rank"]), int(row["variant_rank"]), int(row["global_replicate_id"])), _combined_tuple_key(row, int(row["global_replicate_id"]))) for row in increment_rows]
    keys.sort(key=lambda item: item[0])
    if [index for index, _ in keys] != list(range(EXPECTED_ROWS)):
        raise LooK10ScoreError("combined absolute indexes are not exactly 0..21759")
    digest = hashlib.sha256()
    for _, key in keys:
        digest.update((_canonical(key) + "\n").encode())
    return digest.hexdigest()


def validate_increment_manifest_contract(manifest: Mapping[str, Any]) -> None:
    """Validate the merged-manifest shape emitted by the increment runner."""

    profile = _require_increment_profile()
    indexing = manifest.get("generation_contract", {}).get("replicate_indexing")
    required_indexing = {
        "local_replicate_ids": list(LOCAL_REPLICATE_IDS),
        "global_replicate_offset": GLOBAL_REPLICATE_OFFSET,
        "global_replicate_ids": list(GLOBAL_REPLICATE_IDS),
        "prior_completed_replicates": OLD_REPLICATES,
        "incremental_replicates": INCREMENT_REPLICATES,
        "combined_replicates": EXPECTED_REPLICATES,
    }
    if (
        manifest.get("status") != "complete"
        or manifest.get("evaluation_id") != profile.EVALUATION_ID
        or manifest.get("model_id") != "chk3"
        or manifest.get("coverage", {}).get("cases") != EXPECTED_INCREMENT_ROWS
        or not isinstance(indexing, Mapping)
        or any(indexing.get(key) != value for key, value in required_indexing.items())
        or indexing.get("replicate_id_scope") != "increment_local"
        or indexing.get("prior_k5_rows_reused_in_increment") is not False
        or indexing.get("replicate_seed_by_global_id")
        != {
            str(global_id): REPLICATE_SEEDS[global_id]
            for global_id in GLOBAL_REPLICATE_IDS
        }
    ):
        raise LooK10ScoreError("increment generation manifest/indexing contract drift")


def prepare_score_plan(
    *,
    old_generation_manifest_path: Path,
    old_generation_manifest_sha256: str,
    old_score_manifest_path: Path,
    old_score_manifest_sha256: str,
    increment_manifest_path: Path,
    increment_manifest_sha256: str,
    increment_cohort_path: Path,
    increment_cohort_sha256: str,
    semantic_manifest_path: Path,
    semantic_manifest_sha256: str,
    legacy_nf4_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Deep-validate all inputs and publish the pre-semantic K10 plan."""

    _require_increment_profile()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise LooK10ScoreError(f"score output root already exists: {output_dir}")

    old_generation_binding = _file_binding(old_generation_manifest_path, sealed=True)
    if (
        old_generation_binding["sha256"] != old_generation_manifest_sha256
        or old_generation_binding["sha256"] != DEFAULT_OLD_GENERATION_MANIFEST_SHA256
    ):
        raise LooK10ScoreError("sealed old K5 generation manifest binding drift")
    old_generation_manifest = _read_json(old_generation_manifest_path)
    old_generation_artifact = old_generation_manifest.get("artifacts", {}).get(
        "canonical_generations"
    )
    if not isinstance(old_generation_artifact, Mapping):
        raise LooK10ScoreError("old K5 generation artifact missing")
    _assert_binding(old_generation_artifact, rows=EXPECTED_INCREMENT_ROWS)

    old_score_manifest, old_score_binding, old_rows = _old_score_inventory(
        old_score_manifest_path, old_score_manifest_sha256
    )
    if (
        old_score_manifest.get("inputs", {}).get("generation_manifest")
        != old_generation_binding
        or old_score_manifest.get("inputs", {}).get("canonical_generations")
        != old_generation_artifact
    ):
        raise LooK10ScoreError("old score release is not bound to the requested K5 generation")
    old_coverage = validate_old_score_rows(old_rows)

    increment_binding = _file_binding(increment_manifest_path, sealed=True)
    if increment_binding["sha256"] != increment_manifest_sha256:
        raise LooK10ScoreError("increment generation manifest SHA-256 differs from CLI binding")
    increment_manifest = _read_json(increment_manifest_path)
    validate_increment_manifest_contract(increment_manifest)
    increment_artifact = increment_manifest.get("artifacts", {}).get(
        "canonical_generations"
    )
    if not isinstance(increment_artifact, Mapping):
        raise LooK10ScoreError("increment generation artifact missing")
    increment_path = _assert_binding(increment_artifact, rows=EXPECTED_INCREMENT_ROWS)
    if increment_path.parent != increment_manifest_path.expanduser().resolve().parent:
        raise LooK10ScoreError("increment generations escaped the generation root")

    cohort_binding = _file_binding(increment_cohort_path, sealed=True)
    if cohort_binding["sha256"] != increment_cohort_sha256:
        raise LooK10ScoreError("increment cohort SHA-256 differs from CLI binding")
    manifest_cohort = increment_manifest.get("cohort")
    if not isinstance(manifest_cohort, Mapping):
        raise LooK10ScoreError("increment manifest lacks cohort binding")
    if (
        manifest_cohort.get("path") != cohort_binding["path"]
        or manifest_cohort.get("sha256") != cohort_binding["sha256"]
    ):
        raise LooK10ScoreError("increment manifest/cohort binding mismatch")

    semantic_binding = _file_binding(semantic_manifest_path, sealed=True)
    if (
        semantic_binding["sha256"] != semantic_manifest_sha256
        or semantic_binding["sha256"] != DEFAULT_SEMANTIC_MANIFEST_SHA256
        or old_score_manifest.get("inputs", {}).get("semantic_manifest")
        != semantic_binding
    ):
        raise LooK10ScoreError("semantic inventory is not identical to the sealed K5 scorer")

    increment_rows = list(_iter_jsonl(increment_path))
    increment_coverage = validate_increment_rows(
        increment_rows, old_sample_map=old_coverage.pop("sample_map")
    )
    combined_digest = _combined_key_digest(old_rows, increment_rows)
    legacy = _legacy_nf4_binding(legacy_nf4_root)
    sources = _source_bundle()
    _validate_old_k5_source_compatibility(old_score_manifest, sources)

    output_dir.mkdir(parents=True, exist_ok=False)
    _fsync_directory(output_dir.parent)
    plan = seal_manifest(
        {
            "schema_version": PLAN_SCHEMA,
            "status": "complete",
            "created_at_utc": _utc_now(),
            "evaluation_id": SCORE_EVALUATION_ID,
            "operation": "freeze_incremental_K6_to_K10_semantic_and_combination_plan",
            "inputs": {
                "old_k5_generation_manifest": old_generation_binding,
                "old_k5_canonical_generations": dict(old_generation_artifact),
                "old_k5_score_manifest": old_score_binding,
                "old_k5_row_scores": old_score_manifest["artifacts"]["row_scores"],
                "increment_generation_manifest": increment_binding,
                "increment_canonical_generations": dict(increment_artifact),
                "increment_cohort": cohort_binding,
                "semantic_manifest": semantic_binding,
                "excluded_legacy_nf4_k10": legacy,
            },
            "replicate_mapping": {
                "old_global_replicate_ids": list(range(OLD_REPLICATES)),
                "increment_local_replicate_ids": list(LOCAL_REPLICATE_IDS),
                "increment_global_replicate_offset": GLOBAL_REPLICATE_OFFSET,
                "increment_global_replicate_ids": list(GLOBAL_REPLICATE_IDS),
                "combined_global_replicate_ids": list(range(EXPECTED_REPLICATES)),
                "combined_replicate_seeds": list(REPLICATE_SEEDS),
                "mapping_rule": "global_replicate_id=local_replicate_id+5_for_increment_only",
            },
            "coverage": {
                "old_k5": old_coverage,
                "increment_k6_to_k10": increment_coverage,
                "combined": {
                    "rows": EXPECTED_ROWS,
                    "meetings": EXPECTED_MEETINGS,
                    "variants_per_meeting": EXPECTED_VARIANTS,
                    "replicates": EXPECTED_REPLICATES,
                    "global_replicate_ids": list(range(EXPECTED_REPLICATES)),
                    "paired_blocks": EXPECTED_PAIRED_BLOCKS,
                    "combined_tuple_key_sha256": combined_digest,
                },
            },
            "semantic_policy": {
                "old_rows": "reuse_all_10880_sealed_raw_semantic_scores_without_rescoring",
                "increment_rows": "score_all_10880_rows_with_identical_pinned_backends",
                "generation_gates": "diagnostic_only_no_filter_no_zero_penalty",
                "metrics": list(METRICS),
                "network": False,
            },
            "bootstrap_contract": {
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "confidence": CONFIDENCE,
                "meeting_resampling": "128_with_replacement",
                "replicate_resampling": "10_with_replacement_per_sampled_meeting_occurrence",
                "paired_within_global_replicate": True,
                "shared_indices_across_all_32_cells": True,
                "primary_interval": "hierarchical_percentile",
                "meeting_only_interval": "fixed_K10_meeting_resampling_sensitivity",
            },
            "k_adequacy_rule": {
                "scope": "aggregate_128_meeting_estimand_not_per_meeting_inference",
                "status": K_ADEQUACY_FREEZE_STATUS,
                "threshold_provenance": "unchanged_numeric_thresholds_from_sealed_K5_presemantic_plan",
                "k10_minus_k9_point_drift_threshold": K10_K9_POINT_DRIFT_THRESHOLD,
                "k10_minus_k9_ci_endpoint_drift_threshold": K10_K9_CI_ENDPOINT_DRIFT_THRESHOLD,
                "leave_one_replicate_max_drift_threshold": LORO_MAX_DRIFT_THRESHOLD,
                "decoding_variance_share_threshold": DECODING_VARIANCE_SHARE_THRESHOLD,
                "increase_k_if": list(K_INCREASE_RULES),
            },
            "sources": sources,
        }
    )
    _write_new_json(output_dir / "score_plan.json", plan)
    return plan


def load_score_plan(plan_path: Path, plan_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _file_binding(plan_path, sealed=True)
    if binding["sha256"] != plan_sha256:
        raise LooK10ScoreError("score-plan SHA-256 differs from CLI binding")
    plan = _read_json(plan_path)
    expected_mapping = {
        "old_global_replicate_ids": list(range(OLD_REPLICATES)),
        "increment_local_replicate_ids": list(LOCAL_REPLICATE_IDS),
        "increment_global_replicate_offset": GLOBAL_REPLICATE_OFFSET,
        "increment_global_replicate_ids": list(GLOBAL_REPLICATE_IDS),
        "combined_global_replicate_ids": list(range(EXPECTED_REPLICATES)),
        "combined_replicate_seeds": list(REPLICATE_SEEDS),
        "mapping_rule": "global_replicate_id=local_replicate_id+5_for_increment_only",
    }
    expected_bootstrap = {
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "confidence": CONFIDENCE,
        "meeting_resampling": "128_with_replacement",
        "replicate_resampling": "10_with_replacement_per_sampled_meeting_occurrence",
        "paired_within_global_replicate": True,
        "shared_indices_across_all_32_cells": True,
        "primary_interval": "hierarchical_percentile",
        "meeting_only_interval": "fixed_K10_meeting_resampling_sensitivity",
    }
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "complete"
        or plan.get("evaluation_id") != SCORE_EVALUATION_ID
        or plan.get("replicate_mapping") != expected_mapping
        or plan.get("bootstrap_contract") != expected_bootstrap
        or plan.get("coverage", {}).get("combined", {}).get("rows") != EXPECTED_ROWS
        or plan.get("coverage", {}).get("combined", {}).get("paired_blocks")
        != EXPECTED_PAIRED_BLOCKS
        or plan.get("k_adequacy_rule", {}).get("status") != K_ADEQUACY_FREEZE_STATUS
        or plan.get("k_adequacy_rule", {}).get("increase_k_if") != list(K_INCREASE_RULES)
    ):
        raise LooK10ScoreError("score-plan contract drift")
    inputs = plan.get("inputs")
    if not isinstance(inputs, Mapping):
        raise LooK10ScoreError("score-plan inputs missing")
    for name in (
        "old_k5_generation_manifest",
        "old_k5_score_manifest",
        "increment_generation_manifest",
        "increment_cohort",
        "semantic_manifest",
    ):
        _assert_binding(inputs[name], sealed=True)
    _assert_binding(inputs["old_k5_canonical_generations"], rows=EXPECTED_INCREMENT_ROWS)
    _assert_binding(inputs["old_k5_row_scores"], rows=EXPECTED_INCREMENT_ROWS)
    _assert_binding(inputs["increment_canonical_generations"], rows=EXPECTED_INCREMENT_ROWS)
    _assert_legacy_nf4_binding(inputs.get("excluded_legacy_nf4_k10"))
    _assert_source_bundle(plan.get("sources"))
    return plan, binding


def _load_old_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = _assert_binding(plan["inputs"]["old_k5_row_scores"], rows=EXPECTED_INCREMENT_ROWS)
    rows = list(_iter_jsonl(path))
    inventory = validate_old_score_rows(rows)
    inventory.pop("sample_map")
    if inventory != plan["coverage"]["old_k5"]:
        raise LooK10ScoreError("old K5 score coverage changed after planning")
    return rows


def _load_increment_projection(
    plan: Mapping[str, Any], *, diagnostics: bool
) -> list[dict[str, Any]]:
    old_rows = _load_old_rows(plan)
    old_inventory = validate_old_score_rows(old_rows)
    path = _assert_binding(
        plan["inputs"]["increment_canonical_generations"],
        rows=EXPECTED_INCREMENT_ROWS,
    )
    raw_rows = list(_iter_jsonl(path))
    coverage = validate_increment_rows(raw_rows, old_sample_map=old_inventory["sample_map"])
    if coverage != plan["coverage"]["increment_k6_to_k10"]:
        raise LooK10ScoreError("increment generation coverage changed after planning")
    projection: list[dict[str, Any]] = []
    for position, row in enumerate(raw_rows):
        local = int(row["replicate_id"])
        global_id = GLOBAL_REPLICATE_OFFSET + local
        item = {
            **_combined_tuple_key(row, global_id),
            "source_local_tuple_key": _increment_local_tuple_key(row),
            "source_generation_row": position + 1,
            "sample_id": row.get("variant_sample_id", row.get("sample_id")),
            "answer": row["answer"],
            "reference_minutes": row["reference_minutes"],
            "completion_sha256": row.get("completion_sha256"),
            "answer_sha256": row.get("answer_sha256"),
            "generated_token_ids_sha256": row.get("generated_token_ids_sha256"),
            "reference_minutes_sha256": row.get("reference_minutes_sha256"),
            "finish_reason": row.get("finish_reason"),
            "input_truncated": row.get("input_truncated"),
            "content_tokens": row.get("content_tokens"),
            "full_token_4gram_repetition": row.get("full_token_4gram_repetition"),
            "tail_token_4gram_repetition": row.get("tail_token_4gram_repetition"),
            "local_replicate_id": local,
            "global_replicate_id": global_id,
        }
        if diagnostics:
            try:
                gate = k5.semantic._recompute_core_hard_gate(row)
            except Exception as exc:
                raise LooK10ScoreError(
                    f"increment generation gate diagnostic failed at {position}: {exc}"
                ) from exc
            stored = sorted(row.get("preregistered_core_failures") or [])
            if (
                bool(row.get("preregistered_core_valid"))
                != bool(gate["preregistered_core_valid"])
                or stored != sorted(gate["preregistered_core_failures"])
            ):
                raise LooK10ScoreError(
                    f"stored/recomputed increment gate drift at {position}"
                )
            item["gate"] = gate
        projection.append(item)
    return projection


def _backend_chunk_audit(backend: Any, *, label: str) -> dict[str, Any]:
    metadata = backend.semantic_metadata() if hasattr(backend, "semantic_metadata") else None
    audit = metadata.get("chunk_audit") if isinstance(metadata, Mapping) else None
    if (
        not isinstance(audit, Mapping)
        or audit.get("silent_truncation") is not False
        or audit.get("document_count") != EXPECTED_INCREMENT_ROWS
    ):
        raise LooK10ScoreError(f"{label} did not prove complete no-truncation scoring")
    return dict(metadata)


def _assert_empty_candidate_scores(
    *, metric: str, rows: Sequence[Mapping[str, Any]], vectors: Mapping[str, Sequence[float]]
) -> int:
    empty = [index for index, row in enumerate(rows) if not str(row["answer"]).strip()]
    required = (
        ("bertscore_precision", "bertscore_recall", "bertscore_f1")
        if metric == "bertscore"
        else ("mpnet_cosine",)
    )
    if set(vectors) != set(required):
        raise LooK10ScoreError(f"{metric} score-vector inventory drift")
    for position in empty:
        if any(float(vectors[name][position]) != 0.0 for name in required):
            raise LooK10ScoreError(f"empty increment answer has nonzero {metric} score")
    return len(empty)


def score_metric(
    *,
    metric: str,
    plan_path: Path,
    plan_sha256: str,
    output_dir: Path,
    physical_gpu_index: int,
    batch_size: int,
    gpu_wait_timeout_seconds: int,
    gpu_poll_seconds: int,
) -> dict[str, Any]:
    """Score only the new K6--K10 rows for one semantic metric."""

    if metric not in METRIC_WORKERS or batch_size < 1:
        raise LooK10ScoreError("invalid semantic metric worker or batch size")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise LooK10ScoreError(f"metric output already exists: {output_dir}")
    plan, plan_binding = load_score_plan(plan_path, plan_sha256)
    rows = _load_increment_projection(plan, diagnostics=False)
    try:
        torch, logical_gpu = k5._validate_metric_environment(physical_gpu_index)
    except Exception as exc:
        raise LooK10ScoreError(f"semantic GPU identity validation failed: {exc}") from exc

    def wait_notice(reason: str, processes: list[dict[str, Any]]) -> None:
        print(
            _canonical(
                {
                    "status": "waiting_for_exclusive_increment_semantic_gpu",
                    "metric": metric,
                    "physical_gpu_index": physical_gpu_index,
                    "reason": reason,
                    "external_processes": processes,
                }
            ),
            file=sys.stderr,
            flush=True,
        )

    started = time.monotonic()
    with k5.gpu_runtime.exclusive_gpu_lease(
        physical_gpu_index=physical_gpu_index,
        timeout_seconds=gpu_wait_timeout_seconds,
        poll_seconds=gpu_poll_seconds,
        on_wait=wait_notice,
    ) as lease:
        try:
            bert, mpnet, provenance = k5.semantic.load_formal_semantic_backends(
                semantic_manifest_path=Path(plan["inputs"]["semantic_manifest"]["path"]),
                batch_size=batch_size,
                device="cuda:0",
            )
            candidates = [str(row["answer"]) for row in rows]
            references = [str(row["reference_minutes"]) for row in rows]
            if metric == "bertscore":
                raw = bert.score(candidates, references)
                expected_names = {
                    "bertscore_precision",
                    "bertscore_recall",
                    "bertscore_f1",
                }
                if not isinstance(raw, Mapping) or set(raw) != expected_names:
                    raise LooK10ScoreError("BERTScore metric inventory drift")
                vectors = {
                    name: k5._metric_vector(
                        raw[name], expected=EXPECTED_INCREMENT_ROWS, label=name
                    )
                    for name in sorted(raw)
                }
                backend_audit = _backend_chunk_audit(bert, label="BERTScore")
            else:
                raw = mpnet.score(candidates, references)
                vectors = {
                    "mpnet_cosine": k5._metric_vector(
                        raw, expected=EXPECTED_INCREMENT_ROWS, label="mpnet_cosine"
                    )
                }
                backend_audit = _backend_chunk_audit(mpnet, label="MPNet")
            empty_rows = _assert_empty_candidate_scores(
                metric=metric, rows=rows, vectors=vectors
            )
        except LooK10ScoreError:
            raise
        except Exception as exc:
            raise LooK10ScoreError(f"{metric} semantic inference failed: {exc}") from exc
        torch.cuda.synchronize()
        k5._assert_no_foreign_gpu_processes(physical_gpu_index)
        lease_evidence = dict(lease)
    wall_seconds = time.monotonic() - started

    refreshed, refreshed_binding = load_score_plan(plan_path, plan_sha256)
    if refreshed != plan or refreshed_binding != plan_binding:
        raise LooK10ScoreError("score plan changed during semantic inference")
    sources = _assert_source_bundle(plan.get("sources"))
    metric_rows = []
    for position, source in enumerate(rows):
        metric_rows.append(
            {
                "schema_version": METRIC_ROW_SCHEMA,
                "metric_worker": metric,
                "source_increment_generation_row": position + 1,
                "source_local_tuple_key": source["source_local_tuple_key"],
                "mapped_global_tuple_key": _combined_tuple_key(
                    source, int(source["global_replicate_id"])
                ),
                "sample_id": source["sample_id"],
                "answer_sha256": source["answer_sha256"],
                "reference_minutes_sha256": source["reference_minutes_sha256"],
                "scores": {name: values[position] for name, values in vectors.items()},
            }
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    _fsync_directory(output_dir.parent)
    row_path = output_dir / "row_scores.jsonl"
    _write_new_jsonl(row_path, metric_rows)
    manifest = seal_manifest(
        {
            "schema_version": METRIC_MANIFEST_SCHEMA,
            "status": "complete",
            "created_at_utc": _utc_now(),
            "evaluation_id": SCORE_EVALUATION_ID,
            "metric_worker": metric,
            "operation": "score_increment_rows_only_global_replicates_5_to_9",
            "inputs": {
                "score_plan": plan_binding,
                "increment_canonical_generations": plan["inputs"][
                    "increment_canonical_generations"
                ],
                "semantic_manifest": plan["inputs"]["semantic_manifest"],
            },
            "coverage": {
                "rows": len(metric_rows),
                "local_replicate_ids": list(LOCAL_REPLICATE_IDS),
                "global_replicate_ids": list(GLOBAL_REPLICATE_IDS),
                "mapped_global_tuple_key_sha256_in_source_order": plan["coverage"]
                ["increment_k6_to_k10"]
                ["mapped_global_tuple_key_sha256_in_source_order"],
                "raw_all_increment_rows_scored": True,
                "old_k5_rows_rescored": 0,
                "gate_filtered_rows": 0,
                "gate_zero_penalized_rows": 0,
                "empty_answer_rows": empty_rows,
                "empty_answer_all_raw_metrics_exact_zero": True,
            },
            "execution": {
                "device": "cuda:0",
                "physical_gpu": logical_gpu,
                "batch_size": batch_size,
                "wall_seconds": wall_seconds,
                "python": sys.executable,
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "network": False,
                "gpu_lease": lease_evidence,
            },
            "semantic_models": provenance,
            "backend_audit": backend_audit,
            "artifacts": {
                "row_scores": _file_binding(row_path, rows=EXPECTED_INCREMENT_ROWS)
            },
            "sources": sources,
        }
    )
    load_score_plan(plan_path, plan_sha256)
    _write_new_json(output_dir / "manifest.json", manifest)
    return manifest


def load_metric_bundle(
    *,
    metric: str,
    manifest_path: Path,
    manifest_sha256: str,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    binding = _file_binding(manifest_path, sealed=True)
    if binding["sha256"] != manifest_sha256:
        raise LooK10ScoreError(f"{metric} metric manifest SHA-256 differs from CLI")
    manifest = _read_json(manifest_path)
    coverage = manifest.get("coverage", {})
    if (
        manifest.get("schema_version") != METRIC_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("evaluation_id") != SCORE_EVALUATION_ID
        or manifest.get("metric_worker") != metric
        or manifest.get("inputs", {}).get("score_plan") != plan_binding
        or manifest.get("inputs", {}).get("increment_canonical_generations")
        != plan["inputs"]["increment_canonical_generations"]
        or manifest.get("inputs", {}).get("semantic_manifest")
        != plan["inputs"]["semantic_manifest"]
        or coverage.get("rows") != EXPECTED_INCREMENT_ROWS
        or coverage.get("global_replicate_ids") != list(GLOBAL_REPLICATE_IDS)
        or coverage.get("mapped_global_tuple_key_sha256_in_source_order")
        != plan["coverage"]["increment_k6_to_k10"][
            "mapped_global_tuple_key_sha256_in_source_order"
        ]
        or coverage.get("raw_all_increment_rows_scored") is not True
        or coverage.get("old_k5_rows_rescored") != 0
        or coverage.get("gate_filtered_rows") != 0
        or coverage.get("gate_zero_penalized_rows") != 0
        or coverage.get("empty_answer_rows")
        != plan["coverage"]["increment_k6_to_k10"]["empty_answer_rows"]
        or coverage.get("empty_answer_all_raw_metrics_exact_zero") is not True
        or manifest.get("sources") != plan.get("sources")
    ):
        raise LooK10ScoreError(f"{metric} incremental metric contract drift")
    artifact = manifest.get("artifacts", {}).get("row_scores")
    if not isinstance(artifact, Mapping):
        raise LooK10ScoreError(f"{metric} incremental row-score artifact missing")
    row_path = _assert_binding(artifact, rows=EXPECTED_INCREMENT_ROWS)
    if row_path.parent != manifest_path.expanduser().resolve().parent:
        raise LooK10ScoreError(f"{metric} rows escaped metric root")
    rows = list(_iter_jsonl(row_path))
    return manifest, rows, binding


def _normalise_old_scored_row(row: Mapping[str, Any], position: int) -> dict[str, Any]:
    global_id = int(row["replicate_id"])
    key = _combined_tuple_key(row, global_id)
    if not isinstance(row.get("generation_diagnostics"), Mapping):
        raise LooK10ScoreError(f"old K5 generation diagnostic missing at {position}")
    return {
        "schema_version": SCORE_ROW_SCHEMA,
        "source_release": "sealed_k5_score_release",
        "source_score_row": position + 1,
        "combined_score_row": int(key["absolute_case_index"]) + 1,
        "tuple_key": key,
        "source_tuple_key": row.get("tuple_key"),
        "sample_id": row.get("sample_id"),
        "meeting_id": row.get("meeting_id"),
        "meeting_rank": row.get("meeting_rank"),
        "variant_rank": row.get("variant_rank"),
        "arm": row.get("arm"),
        "intervention_topic": row.get("intervention_topic"),
        "replicate_id": global_id,
        "replicate_seed": row.get("replicate_seed"),
        "row_seed": row.get("row_seed"),
        "completion_sha256": row.get("completion_sha256"),
        "answer_sha256": row.get("answer_sha256"),
        "generated_token_ids_sha256": row.get("generated_token_ids_sha256"),
        "reference_minutes_sha256": row.get("reference_minutes_sha256"),
        "finish_reason": row.get("finish_reason"),
        "input_truncated": row.get("input_truncated"),
        "generated_tokens": row.get("generated_tokens"),
        "empty_answer": bool(row.get("empty_answer")),
        "full_token_4gram_repetition": row.get("full_token_4gram_repetition"),
        "tail_token_4gram_repetition": row.get("tail_token_4gram_repetition"),
        "raw_semantic_scores": dict(row["raw_semantic_scores"]),
        "generation_diagnostics": dict(row["generation_diagnostics"]),
        "primary_loo_policy": dict(row["primary_loo_policy"]),
        "semantic_score_provenance": "reused_without_rescoring",
    }


def build_increment_scored_rows(
    *,
    generation_rows: Sequence[Mapping[str, Any]],
    bert_rows: Sequence[Mapping[str, Any]],
    mpnet_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not (
        len(generation_rows)
        == len(bert_rows)
        == len(mpnet_rows)
        == EXPECTED_INCREMENT_ROWS
    ):
        raise LooK10ScoreError("increment semantic/generation denominator drift")
    scored: list[dict[str, Any]] = []
    for position, (source, bert, mpnet) in enumerate(
        zip(generation_rows, bert_rows, mpnet_rows, strict=True)
    ):
        global_key = _combined_tuple_key(source, int(source["global_replicate_id"]))
        for worker, metric_row in (("bertscore", bert), ("mpnet", mpnet)):
            if (
                metric_row.get("schema_version") != METRIC_ROW_SCHEMA
                or metric_row.get("metric_worker") != worker
                or metric_row.get("source_increment_generation_row") != position + 1
                or metric_row.get("source_local_tuple_key")
                != source["source_local_tuple_key"]
                or metric_row.get("mapped_global_tuple_key") != global_key
                or metric_row.get("sample_id") != source["sample_id"]
                or metric_row.get("answer_sha256") != source["answer_sha256"]
                or metric_row.get("reference_minutes_sha256")
                != source["reference_minutes_sha256"]
            ):
                raise LooK10ScoreError(f"increment semantic row alignment drift at {position}")
        bert_scores = bert.get("scores")
        mpnet_scores = mpnet.get("scores")
        if not isinstance(bert_scores, Mapping) or not isinstance(mpnet_scores, Mapping):
            raise LooK10ScoreError(f"increment semantic score payload missing at {position}")
        semantic_scores = {
            "mpnet_cosine": k5._finite_score(
                mpnet_scores.get("mpnet_cosine"), label=f"MPNet[{position}]"
            ),
            "bertscore_precision": k5._finite_score(
                bert_scores.get("bertscore_precision"), label=f"BERTScore-P[{position}]"
            ),
            "bertscore_recall": k5._finite_score(
                bert_scores.get("bertscore_recall"), label=f"BERTScore-R[{position}]"
            ),
            "bertscore_f1": k5._finite_score(
                bert_scores.get("bertscore_f1"), label=f"BERTScore-F1[{position}]"
            ),
        }
        empty = not str(source.get("answer") or "").strip()
        if empty and any(value != 0.0 for value in semantic_scores.values()):
            raise LooK10ScoreError(f"empty increment answer has nonzero scores at {position}")
        gate = source.get("gate")
        if not isinstance(gate, Mapping):
            raise LooK10ScoreError(f"increment diagnostic gate missing at {position}")
        diagnostics = {
            "metrics": {
                "structure_delivery": bool(
                    gate["delivery_valid"] and gate["native_structure_valid"]
                ),
                "numeric_fidelity": bool(gate["numeric_multiset_preserved"]),
                "date_fidelity": bool(gate["date_set_preserved"]),
                "degeneration_free": bool(gate["degeneration_free"]),
            },
            "preregistered_core_valid": bool(gate["preregistered_core_valid"]),
            "preregistered_core_failures": list(
                gate["preregistered_core_failures"]
            ),
        }
        scored.append(
            {
                "schema_version": SCORE_ROW_SCHEMA,
                "source_release": "increment_generation_k6_to_k10",
                "source_generation_row": position + 1,
                "combined_score_row": int(global_key["absolute_case_index"]) + 1,
                "tuple_key": global_key,
                "source_tuple_key": source["source_local_tuple_key"],
                "sample_id": source["sample_id"],
                "meeting_id": source["meeting_id"],
                "meeting_rank": source["meeting_rank"],
                "variant_rank": source["variant_rank"],
                "arm": source["arm"],
                "intervention_topic": source["intervention_topic"],
                "replicate_id": source["global_replicate_id"],
                "local_replicate_id": source["local_replicate_id"],
                "replicate_seed": source["replicate_seed"],
                "row_seed": source["row_seed"],
                "completion_sha256": source["completion_sha256"],
                "answer_sha256": source["answer_sha256"],
                "generated_token_ids_sha256": source["generated_token_ids_sha256"],
                "reference_minutes_sha256": source["reference_minutes_sha256"],
                "finish_reason": source["finish_reason"],
                "input_truncated": source["input_truncated"],
                "generated_tokens": source["content_tokens"],
                "empty_answer": empty,
                "full_token_4gram_repetition": source[
                    "full_token_4gram_repetition"
                ],
                "tail_token_4gram_repetition": source[
                    "tail_token_4gram_repetition"
                ],
                "raw_semantic_scores": semantic_scores,
                "generation_diagnostics": diagnostics,
                "primary_loo_policy": {
                    "included": True,
                    "gate_filter_applied": False,
                    "gate_zero_penalty_applied": False,
                },
                "semantic_score_provenance": "new_increment_only_scoring",
            }
        )
    return scored


def combine_scored_rows(
    old_rows: Sequence[Mapping[str, Any]],
    increment_rows: Sequence[Mapping[str, Any]],
    *,
    expected_key_sha256: str | None = None,
) -> list[dict[str, Any]]:
    old = [_normalise_old_scored_row(row, index) for index, row in enumerate(old_rows)]
    combined = old + [dict(row) for row in increment_rows]
    combined.sort(key=lambda row: int(row["tuple_key"]["absolute_case_index"]))
    if len(combined) != EXPECTED_ROWS:
        raise LooK10ScoreError("combined score denominator drift")
    digest = hashlib.sha256()
    coordinates: set[tuple[int, int, int]] = set()
    sample_hashes: dict[tuple[int, int], tuple[str, str]] = {}
    paired: dict[tuple[int, int], tuple[int, int]] = {}
    for position, row in enumerate(combined):
        key = row["tuple_key"]
        coordinate = (
            int(row["meeting_rank"]),
            int(row["variant_rank"]),
            int(row["replicate_id"]),
        )
        if (
            key.get("absolute_case_index") != position
            or row.get("combined_score_row") != position + 1
            or coordinate in coordinates
            or row.get("replicate_seed") != REPLICATE_SEEDS[coordinate[2]]
        ):
            raise LooK10ScoreError(f"combined score coordinate/mapping drift at {position}")
        coordinates.add(coordinate)
        digest.update((_canonical(key) + "\n").encode())
        identity = (str(row["sample_id"]), str(row["reference_minutes_sha256"]))
        prior = sample_hashes.setdefault(coordinate[:2], identity)
        if prior != identity:
            raise LooK10ScoreError(f"combined sample/reference drift at {position}")
        pair_value = (int(row["row_seed"]), int(row["replicate_seed"]))
        prior_pair = paired.setdefault((coordinate[0], coordinate[2]), pair_value)
        if prior_pair != pair_value:
            raise LooK10ScoreError(f"combined common-random-number drift at {position}")
    expected_coordinates = {
        (meeting, variant, replicate)
        for meeting in range(EXPECTED_MEETINGS)
        for variant in range(EXPECTED_VARIANTS)
        for replicate in range(EXPECTED_REPLICATES)
    }
    if coordinates != expected_coordinates or len(paired) != EXPECTED_PAIRED_BLOCKS:
        raise LooK10ScoreError("combined 128x17x10 matrix is incomplete")
    if expected_key_sha256 is not None and digest.hexdigest() != expected_key_sha256:
        raise LooK10ScoreError("combined tuple-key digest differs from pre-semantic plan")
    return combined


def build_meeting_cells(
    score_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], np.ndarray]]:
    if len(score_rows) != EXPECTED_ROWS:
        raise LooK10ScoreError("score matrix must contain 21,760 rows")
    by_coordinate: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    meeting_ids: dict[int, str] = {}
    for row in score_rows:
        key = (
            int(row["meeting_rank"]),
            int(row["variant_rank"]),
            int(row["replicate_id"]),
        )
        if key in by_coordinate:
            raise LooK10ScoreError(f"duplicate combined score coordinate: {key}")
        by_coordinate[key] = row
        meeting = int(row["meeting_rank"])
        prior = meeting_ids.setdefault(meeting, str(row["meeting_id"]))
        if prior != str(row["meeting_id"]):
            raise LooK10ScoreError(f"meeting identity drift for rank {meeting}")
    expected = {
        (meeting, variant, replicate)
        for meeting in range(EXPECTED_MEETINGS)
        for variant in range(EXPECTED_VARIANTS)
        for replicate in range(EXPECTED_REPLICATES)
    }
    if set(by_coordinate) != expected:
        raise LooK10ScoreError("combined score coordinate coverage is not exact")

    matrices = {
        (topic, arm, metric): np.empty(
            (EXPECTED_MEETINGS, EXPECTED_REPLICATES), dtype=np.float64
        )
        for topic in TOPICS
        for arm in ARMS
        for metric in METRICS
    }
    cells: list[dict[str, Any]] = []
    for meeting in range(EXPECTED_MEETINGS):
        for topic_index, topic in enumerate(TOPICS):
            for arm_index, arm in enumerate(ARMS):
                variant = 1 + 2 * topic_index + arm_index
                full_scores = {metric: [] for metric in METRICS}
                intervention_scores = {metric: [] for metric in METRICS}
                deltas = {metric: [] for metric in METRICS}
                full_valid: list[bool] = []
                intervention_valid: list[bool] = []
                for replicate in range(EXPECTED_REPLICATES):
                    full = by_coordinate[(meeting, 0, replicate)]
                    intervention = by_coordinate[(meeting, variant, replicate)]
                    if (
                        full["replicate_seed"] != intervention["replicate_seed"]
                        or full["row_seed"] != intervention["row_seed"]
                        or full["reference_minutes_sha256"]
                        != intervention["reference_minutes_sha256"]
                    ):
                        raise LooK10ScoreError(
                            f"paired Full/intervention binding drift at {meeting}/{variant}/{replicate}"
                        )
                    for metric in METRICS:
                        full_value = float(full["raw_semantic_scores"][metric])
                        intervention_value = float(
                            intervention["raw_semantic_scores"][metric]
                        )
                        delta = full_value - intervention_value
                        full_scores[metric].append(full_value)
                        intervention_scores[metric].append(intervention_value)
                        deltas[metric].append(delta)
                        matrices[(topic, arm, metric)][meeting, replicate] = delta
                    full_valid.append(
                        bool(
                            full["generation_diagnostics"][
                                "preregistered_core_valid"
                            ]
                        )
                    )
                    intervention_valid.append(
                        bool(
                            intervention["generation_diagnostics"][
                                "preregistered_core_valid"
                            ]
                        )
                    )
                cells.append(
                    {
                        "schema_version": MEETING_CELL_SCHEMA,
                        "meeting_id": meeting_ids[meeting],
                        "meeting_rank": meeting,
                        "topic": topic,
                        "arm": arm,
                        "replicate_ids": list(range(EXPECTED_REPLICATES)),
                        "replicate_seeds": list(REPLICATE_SEEDS),
                        "full_scores": full_scores,
                        "intervention_scores": intervention_scores,
                        "paired_deltas": deltas,
                        "meeting_mean_delta": {
                            metric: statistics.fmean(deltas[metric])
                            for metric in METRICS
                        },
                        "meeting_replicate_sd": {
                            metric: statistics.stdev(deltas[metric])
                            for metric in METRICS
                        },
                        "positive_replicates": {
                            metric: sum(value > 0 for value in deltas[metric])
                            for metric in METRICS
                        },
                        "generation_diagnostics": {
                            "full_core_valid_replicates": sum(full_valid),
                            "intervention_core_valid_replicates": sum(
                                intervention_valid
                            ),
                            "used_to_filter_primary": False,
                        },
                    }
                )
    if len(cells) != EXPECTED_MEETINGS * len(TOPICS) * len(ARMS):
        raise LooK10ScoreError("meeting-cell denominator drift")
    return cells, matrices


def make_bootstrap_plan(
    *,
    meeting_ids: Sequence[str],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = EXPECTED_REPLICATES,
) -> BootstrapPlan:
    if (
        draws < 1
        or replicates != EXPECTED_REPLICATES
        or len(meeting_ids) != EXPECTED_MEETINGS
        or len(set(meeting_ids)) != EXPECTED_MEETINGS
        or seed < 0
        or seed >= 2**64
    ):
        raise LooK10ScoreError("invalid K10 hierarchical bootstrap dimensions")
    rng = np.random.default_rng(seed)
    meetings = rng.integers(
        0, EXPECTED_MEETINGS, size=(draws, EXPECTED_MEETINGS), dtype=np.uint16
    )
    uniforms = rng.random(size=(draws, EXPECTED_MEETINGS, replicates))
    prefixes = {
        prefix: np.floor(uniforms[:, :, :prefix] * prefix).astype(np.uint8)
        for prefix in range(1, replicates + 1)
    }
    replicate_indices = prefixes[replicates]
    metadata = {
        "schema_version": INDEX_PLAN_SCHEMA,
        "draws": draws,
        "seed": seed,
        "meeting_ids": list(meeting_ids),
        "replicates": replicates,
        "meeting_index_shape": list(meetings.shape),
        "replicate_index_shape": list(replicate_indices.shape),
        "meeting_index_dtype": "uint16-le",
        "replicate_index_dtype": "uint8",
        "prefix_coupling": "shared_uniforms_floor_times_K_for_cumulative_K1_to_K10",
        "prefix_inventory": list(range(1, replicates + 1)),
        "prefix_replicate_index_shapes": {
            f"k{prefix}": list(prefixes[prefix].shape)
            for prefix in range(1, replicates + 1)
        },
        "numpy_version": np.__version__,
        "numpy_bit_generator": type(rng.bit_generator).__name__,
        "paired_indices": True,
        "shared_across_topics": list(TOPICS),
        "shared_across_arms": list(ARMS),
        "shared_across_metrics": list(METRICS),
    }
    digest = hashlib.sha256()
    digest.update(_canonical(metadata).encode())
    digest.update(meetings.astype("<u2", copy=False).tobytes(order="C"))
    for prefix in range(1, replicates + 1):
        digest.update(bytes([prefix]))
        digest.update(prefixes[prefix].tobytes(order="C"))
    return BootstrapPlan(
        meeting_indices=meetings,
        replicate_indices=replicate_indices,
        prefix_replicate_indices=prefixes,
        sha256=digest.hexdigest(),
        metadata=metadata,
    )


def _ci(values: np.ndarray) -> tuple[float, float]:
    alpha = (1.0 - CONFIDENCE) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha], method="linear")
    return float(lower), float(upper)


def _interval_classification(lower: float, upper: float) -> str:
    if lower > 0:
        return "positive_excludes_zero"
    if upper < 0:
        return "negative_excludes_zero"
    return "crosses_zero"


def _same_nonzero_sign(left: float, right: float) -> bool:
    return left != 0 and right != 0 and math.copysign(1, left) == math.copysign(1, right)


def compute_bootstrap(
    *,
    matrices: Mapping[tuple[str, str, str], np.ndarray],
    meeting_ids: Sequence[str],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    adequacy_thresholds: Mapping[str, Any] | None = None,
) -> BootstrapComputation:
    thresholds = dict(
        adequacy_thresholds
        or {
            "k10_minus_k9_point_drift_threshold": K10_K9_POINT_DRIFT_THRESHOLD,
            "k10_minus_k9_ci_endpoint_drift_threshold": K10_K9_CI_ENDPOINT_DRIFT_THRESHOLD,
            "leave_one_replicate_max_drift_threshold": LORO_MAX_DRIFT_THRESHOLD,
            "decoding_variance_share_threshold": DECODING_VARIANCE_SHARE_THRESHOLD,
        }
    )
    required = {
        "k10_minus_k9_point_drift_threshold",
        "k10_minus_k9_ci_endpoint_drift_threshold",
        "leave_one_replicate_max_drift_threshold",
        "decoding_variance_share_threshold",
    }
    if set(thresholds) != required or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in thresholds.values()
    ):
        raise LooK10ScoreError("invalid frozen K10-adequacy thresholds")
    expected_keys = {
        (topic, arm, metric)
        for topic in TOPICS
        for arm in ARMS
        for metric in METRICS
    }
    if set(matrices) != expected_keys or len(meeting_ids) != EXPECTED_MEETINGS:
        raise LooK10ScoreError("bootstrap matrix/meeting inventory drift")
    for key, matrix in matrices.items():
        if (
            not isinstance(matrix, np.ndarray)
            or matrix.shape != (EXPECTED_MEETINGS, EXPECTED_REPLICATES)
            or not np.all(np.isfinite(matrix))
        ):
            raise LooK10ScoreError(f"invalid finite K10 matrix for {key}")

    point_threshold = float(thresholds["k10_minus_k9_point_drift_threshold"])
    endpoint_threshold = float(
        thresholds["k10_minus_k9_ci_endpoint_drift_threshold"]
    )
    loro_threshold = float(thresholds["leave_one_replicate_max_drift_threshold"])
    variance_threshold = float(thresholds["decoding_variance_share_threshold"])
    plan = make_bootstrap_plan(meeting_ids=meeting_ids, draws=draws, seed=seed)
    cell_draws: dict[tuple[str, str, str], np.ndarray] = {}
    prefix_draws: dict[tuple[str, str, str, int], np.ndarray] = {}
    meeting_only_draws: dict[tuple[str, str, str], np.ndarray] = {}
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    adequacy_cells: list[dict[str, Any]] = []
    increase_cells: list[dict[str, Any]] = []

    for topic in TOPICS:
        cells[topic] = {}
        for arm in ARMS:
            cells[topic][arm] = {}
            for metric in METRICS:
                key = (topic, arm, metric)
                matrix = matrices[key]
                meeting_means = matrix.mean(axis=1)
                nested = matrix[
                    plan.meeting_indices[:, :, None], plan.replicate_indices
                ].mean(axis=(1, 2))
                meeting_only = meeting_means[plan.meeting_indices].mean(axis=1)
                cell_draws[key] = nested
                meeting_only_draws[key] = meeting_only
                point = float(matrix.mean())
                nested_low, nested_high = _ci(nested)
                meeting_low, meeting_high = _ci(meeting_only)
                nested_variance = float(np.var(nested, ddof=1)) if draws > 1 else 0.0
                within_variances = np.var(matrix, axis=1, ddof=1)
                decoding_variance = float(
                    within_variances.sum()
                    / (EXPECTED_MEETINGS**2 * EXPECTED_REPLICATES)
                )
                # ``np.var`` can leave sub-ulp residue for an exactly constant
                # floating-point matrix.  Such a panel has zero decoding
                # variance by definition and must not fail closed as ``None``.
                if float(np.ptp(matrix)) == 0.0:
                    decoding_variance = 0.0
                variance_share = (
                    decoding_variance / nested_variance
                    if nested_variance > 0
                    else (0.0 if decoding_variance == 0 else None)
                )
                by_replicate = matrix.mean(axis=0)
                leave_one = [
                    float(np.delete(matrix, replicate, axis=1).mean())
                    for replicate in range(EXPECTED_REPLICATES)
                ]
                prefixes: dict[str, Any] = {}
                for prefix in range(1, EXPECTED_REPLICATES + 1):
                    prefix_matrix = matrix[:, :prefix]
                    distribution = prefix_matrix[
                        plan.meeting_indices[:, :, None],
                        plan.prefix_replicate_indices[prefix],
                    ].mean(axis=(1, 2))
                    if not np.all(np.isfinite(distribution)):
                        raise LooK10ScoreError(f"non-finite K{prefix} draw for {key}")
                    prefix_draws[(*key, prefix)] = distribution
                    lower, upper = _ci(distribution)
                    prefixes[f"k{prefix}"] = {
                        "fresh_stochastic_replicates": prefix,
                        "replicate_seeds": list(REPLICATE_SEEDS[:prefix]),
                        "point_estimate_mean_delta": float(prefix_matrix.mean()),
                        "hierarchical_ci_lower": lower,
                        "hierarchical_ci_upper": upper,
                        "hierarchical_ci_width": upper - lower,
                        "interval_classification": _interval_classification(
                            lower, upper
                        ),
                        "historical_greedy_nf4_included": False,
                    }
                k9 = prefixes["k9"]
                k10 = prefixes["k10"]
                point_drift = abs(
                    float(k10["point_estimate_mean_delta"])
                    - float(k9["point_estimate_mean_delta"])
                )
                lower_drift = abs(
                    float(k10["hierarchical_ci_lower"])
                    - float(k9["hierarchical_ci_lower"])
                )
                upper_drift = abs(
                    float(k10["hierarchical_ci_upper"])
                    - float(k9["hierarchical_ci_upper"])
                )
                endpoint_drift = max(lower_drift, upper_drift)
                loro_max = max(abs(value - point) for value in leave_one)
                decisive = nested_low > 0 or nested_high < 0
                classification = _interval_classification(nested_low, nested_high)
                prefix_sign_stable = all(
                    _same_nonzero_sign(
                        point, float(prefixes[f"k{prefix}"]["point_estimate_mean_delta"])
                    )
                    for prefix in (8, 9, 10)
                )
                prefix_ci_stable = not decisive or all(
                    prefixes[f"k{prefix}"]["interval_classification"] == classification
                    for prefix in (8, 9, 10)
                )
                loro_sign_stable = all(_same_nonzero_sign(point, value) for value in leave_one)
                reasons: list[str] = []
                if point_drift > point_threshold:
                    reasons.append("k10_minus_k9_point_drift_above_threshold")
                if endpoint_drift > endpoint_threshold:
                    reasons.append("k10_minus_k9_ci_endpoint_drift_above_threshold")
                if loro_max > loro_threshold:
                    reasons.append("leave_one_replicate_max_drift_above_threshold")
                if variance_share is None or variance_share > variance_threshold:
                    reasons.append("decoding_variance_share_above_threshold")
                if decisive and not prefix_ci_stable:
                    reasons.append("critical_cell_k8_k9_k10_ci_classification_unstable")
                if decisive and (not prefix_sign_stable or not loro_sign_stable):
                    reasons.append("critical_cell_prefix_or_loro_point_sign_unstable")
                if reasons:
                    increase_cells.append(
                        {"topic": topic, "arm": arm, "metric": metric, "reasons": reasons}
                    )
                diagnostic = {
                    "topic": topic,
                    "arm": arm,
                    "metric": metric,
                    "conditional_decoding_mcse": math.sqrt(decoding_variance),
                    "nested_bootstrap_variance": nested_variance,
                    "estimated_decoding_variance_share": variance_share,
                    "thresholds": {
                        "k10_minus_k9_point_drift": point_threshold,
                        "k10_minus_k9_ci_endpoint_drift": endpoint_threshold,
                        "leave_one_replicate_max_drift": loro_threshold,
                        "decoding_variance_share": variance_threshold,
                    },
                    "cumulative_fresh_stochastic_prefixes": prefixes,
                    "k10_minus_k9_point_drift": point_drift,
                    "k10_minus_k9_ci_lower_drift": lower_drift,
                    "k10_minus_k9_ci_upper_drift": upper_drift,
                    "k10_minus_k9_ci_endpoint_max_drift": endpoint_drift,
                    "global_replicate_mean_deltas": [float(value) for value in by_replicate],
                    "leave_one_replicate_estimates": leave_one,
                    "leave_one_replicate_max_abs_shift": loro_max,
                    "decisive_nested_ci": decisive,
                    "decisive_cell_leave_one_sign_stable": loro_sign_stable,
                    "decisive_cell_k8_k9_k10_point_sign_stable": prefix_sign_stable,
                    "decisive_cell_k8_k9_k10_ci_classification_stable": prefix_ci_stable,
                    "increase_k_reasons": reasons,
                }
                adequacy_cells.append(diagnostic)
                cells[topic][arm][metric] = {
                    "full_minus_intervention": True,
                    "point_estimate_mean_delta": point,
                    "replicate_delta_mean_by_seed": [float(value) for value in by_replicate],
                    "meeting_mean_sd": float(np.std(meeting_means, ddof=1)),
                    "positive_meetings": int(np.count_nonzero(meeting_means > 0)),
                    "negative_meetings": int(np.count_nonzero(meeting_means < 0)),
                    "zero_meetings": int(np.count_nonzero(meeting_means == 0)),
                    "hierarchical_bootstrap": {
                        "draws": draws,
                        "failed_draws": 0,
                        "mean": float(nested.mean()),
                        "standard_error": float(np.std(nested, ddof=1)) if draws > 1 else 0.0,
                        "ci_lower": nested_low,
                        "ci_upper": nested_high,
                        "draw_share_above_zero": float(np.mean(nested > 0)),
                    },
                    "meeting_only_bootstrap_sensitivity": {
                        "draws": draws,
                        "failed_draws": 0,
                        "mean": float(meeting_only.mean()),
                        "standard_error": float(np.std(meeting_only, ddof=1)) if draws > 1 else 0.0,
                        "ci_lower": meeting_low,
                        "ci_upper": meeting_high,
                        "draw_share_above_zero": float(np.mean(meeting_only > 0)),
                    },
                    "k_adequacy": diagnostic,
                }

    results = seal_manifest(
        {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "evaluation_id": SCORE_EVALUATION_ID,
            "estimand": {
                "replicate_delta": "raw_score_full_minus_raw_score_intervention",
                "meeting_aggregation": "arithmetic_mean_of_ten_paired_replicates",
                "panel_aggregation": "equal_weight_arithmetic_mean_of_128_meetings",
                "generation_gates": "diagnostic_only_no_filter_no_zero_penalty",
                "positive_delta": "intervention_reduced_target_similarity",
            },
            "coverage": {
                "meetings": EXPECTED_MEETINGS,
                "topics": len(TOPICS),
                "arms": len(ARMS),
                "metrics": len(METRICS),
                "primary_cells": len(expected_keys),
                "replicates": EXPECTED_REPLICATES,
                "paired_blocks": EXPECTED_PAIRED_BLOCKS,
            },
            "bootstrap_contract": {
                **plan.metadata,
                "index_plan_sha256": plan.sha256,
                "confidence": CONFIDENCE,
                "primary_interval": "hierarchical_paired_percentile",
                "meeting_only_interval": "fixed_K10_meeting_resampling_sensitivity",
            },
            "cells": cells,
            "k_adequacy": {
                "generation_replicates_per_meeting_arm": EXPECTED_REPLICATES,
                "within_meeting_decoding_variance_estimable": True,
                "replicate_resampling_performed": True,
                "scope": "aggregate_128_meeting_estimand_not_per_meeting_inference",
                "thresholds_frozen_before_increment_semantic_scoring": True,
                "stochastic_prefix_policy": "cumulative fresh BF16-vLLM replicates K1..K10; historical NF4 excluded",
                "k10_minus_k9_point_drift_threshold": point_threshold,
                "k10_minus_k9_ci_endpoint_drift_threshold": endpoint_threshold,
                "leave_one_replicate_max_drift_threshold": loro_threshold,
                "decoding_variance_share_threshold": variance_threshold,
                "increase_k_recommended": bool(increase_cells),
                "increase_k_reason_cells": increase_cells,
                "aggregate_status": (
                    "increase_k_recommended"
                    if increase_cells
                    else "k10_adequate_for_aggregate_estimand"
                ),
                "cell_diagnostics": adequacy_cells,
                "per_meeting_inference": "not_authorized_by_aggregate_K_adequacy_gate",
            },
            "multiplicity": {
                "primary_cells": 32,
                "confidence_intervals_are_unadjusted": True,
                "familywise_significance_claim_authorized": False,
            },
            "limitations": [
                "This is post-selection within-Model-chk-2 sensitivity, not causal attribution.",
                "The frozen target is a deterministic source-grounded Minutes-style reference, not official Minutes.",
                "The 32 percentile intervals are unadjusted for multiple comparisons.",
                "K-adequacy is evaluated for aggregate topic-arm-metric estimands, not per-meeting inference.",
                "The incomplete historical Transformers NF4 K10 attempt is bound for audit but excluded from all estimates.",
            ],
        }
    )
    return BootstrapComputation(
        plan=plan,
        cell_draws=cell_draws,
        prefix_cell_draws=prefix_draws,
        meeting_only_draws=meeting_only_draws,
        results=results,
    )


def _write_bootstrap_draws(path: Path, computation: BootstrapComputation) -> None:
    def writer(handle: Any) -> None:
        for draw_id in range(len(computation.plan.meeting_indices)):
            cells: dict[str, Any] = {}
            for topic in TOPICS:
                cells[topic] = {}
                for arm in ARMS:
                    cells[topic][arm] = {
                        metric: {
                            "hierarchical_mean_delta": float(
                                computation.cell_draws[(topic, arm, metric)][draw_id]
                            ),
                            "meeting_only_mean_delta": float(
                                computation.meeting_only_draws[(topic, arm, metric)][
                                    draw_id
                                ]
                            ),
                            "stochastic_prefix_hierarchical_mean_delta": {
                                f"k{prefix}": float(
                                    computation.prefix_cell_draws[
                                        (topic, arm, metric, prefix)
                                    ][draw_id]
                                )
                                for prefix in range(1, EXPECTED_REPLICATES + 1)
                            },
                        }
                        for metric in METRICS
                    }
            row = {
                "schema_version": BOOTSTRAP_DRAW_SCHEMA,
                "draw_id": draw_id,
                "index_plan_sha256": computation.plan.sha256,
                "meeting_indices": [
                    int(value)
                    for value in computation.plan.meeting_indices[draw_id]
                ],
                "replicate_indices": [
                    [int(value) for value in values]
                    for values in computation.plan.replicate_indices[draw_id]
                ],
                "replicate_indices_by_stochastic_prefix": {
                    f"k{prefix}": [
                        [int(value) for value in values]
                        for values in computation.plan.prefix_replicate_indices[
                            prefix
                        ][draw_id]
                    ]
                    for prefix in range(1, EXPECTED_REPLICATES + 1)
                },
                "cell_mean_deltas": cells,
            }
            handle.write(_canonical(row) + "\n")

    _publish_new_file(path, writer)


def _diagnostic_failure_rows(
    score_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in score_rows:
        diagnostics = row["generation_diagnostics"]
        if bool(diagnostics["preregistered_core_valid"]):
            continue
        failures.append(
            {
                "schema_version": FAILURE_ROW_SCHEMA,
                "combined_score_row": row["combined_score_row"],
                "source_release": row["source_release"],
                "tuple_key": row["tuple_key"],
                "sample_id": row["sample_id"],
                "meeting_id": row["meeting_id"],
                "arm": row["arm"],
                "intervention_topic": row["intervention_topic"],
                "replicate_id": row["replicate_id"],
                "completion_sha256": row["completion_sha256"],
                "answer_sha256": row["answer_sha256"],
                "diagnostic_failures": list(
                    diagnostics["preregistered_core_failures"]
                ),
                "raw_semantic_scores": {
                    "mpnet_cosine": row["raw_semantic_scores"]["mpnet_cosine"],
                    "bertscore_f1": row["raw_semantic_scores"]["bertscore_f1"],
                },
                "included_in_primary_loo": True,
                "effect_on_primary_score": "none_diagnostic_only",
            }
        )
    return failures


def finalize_score(
    *,
    plan_path: Path,
    plan_sha256: str,
    bert_manifest_path: Path,
    bert_manifest_sha256: str,
    mpnet_manifest_path: Path,
    mpnet_manifest_sha256: str,
    output_dir: Path,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise LooK10ScoreError("score root must already contain the immutable plan")
    final_path = output_dir / "manifest.json"
    if final_path.exists() or final_path.is_symlink():
        raise LooK10ScoreError(f"refusing to overwrite final score manifest: {final_path}")
    plan, plan_binding = load_score_plan(plan_path, plan_sha256)
    if (
        draws != plan["bootstrap_contract"]["draws"]
        or seed != plan["bootstrap_contract"]["seed"]
    ):
        raise LooK10ScoreError("finalize bootstrap parameters differ from frozen plan")
    bert_manifest, bert_rows, bert_binding = load_metric_bundle(
        metric="bertscore",
        manifest_path=bert_manifest_path,
        manifest_sha256=bert_manifest_sha256,
        plan=plan,
        plan_binding=plan_binding,
    )
    mpnet_manifest, mpnet_rows, mpnet_binding = load_metric_bundle(
        metric="mpnet",
        manifest_path=mpnet_manifest_path,
        manifest_sha256=mpnet_manifest_sha256,
        plan=plan,
        plan_binding=plan_binding,
    )
    old_rows = _load_old_rows(plan)
    increment_generation = _load_increment_projection(plan, diagnostics=True)
    increment_scores = build_increment_scored_rows(
        generation_rows=increment_generation,
        bert_rows=bert_rows,
        mpnet_rows=mpnet_rows,
    )
    score_rows = combine_scored_rows(
        old_rows,
        increment_scores,
        expected_key_sha256=plan["coverage"]["combined"][
            "combined_tuple_key_sha256"
        ],
    )
    failures = _diagnostic_failure_rows(score_rows)
    meeting_cells, matrices = build_meeting_cells(score_rows)
    meeting_ids = [
        str(score_rows[meeting * EXPECTED_VARIANTS * EXPECTED_REPLICATES]["meeting_id"])
        for meeting in range(EXPECTED_MEETINGS)
    ]
    rule = plan["k_adequacy_rule"]
    thresholds = {
        name: rule[name]
        for name in (
            "k10_minus_k9_point_drift_threshold",
            "k10_minus_k9_ci_endpoint_drift_threshold",
            "leave_one_replicate_max_drift_threshold",
            "decoding_variance_share_threshold",
        )
    }
    computation = compute_bootstrap(
        matrices=matrices,
        meeting_ids=meeting_ids,
        draws=draws,
        seed=seed,
        adequacy_thresholds=thresholds,
    )

    row_path = output_dir / "row_scores.jsonl"
    failure_path = output_dir / "generation_diagnostic_failures.jsonl"
    meeting_path = output_dir / "meeting_cell_deltas.jsonl"
    draw_path = output_dir / "bootstrap_draws.jsonl"
    results_path = output_dir / "bootstrap_results.json"
    _write_new_jsonl(row_path, score_rows)
    _write_new_jsonl(failure_path, failures)
    _write_new_jsonl(meeting_path, meeting_cells)
    _write_bootstrap_draws(draw_path, computation)
    _write_new_json(results_path, computation.results)

    failure_counts: Counter[str] = Counter()
    for row in failures:
        failure_counts.update(row["diagnostic_failures"])
    semantic_summary = {
        metric: {
            "mean_all_rows": statistics.fmean(
                float(row["raw_semantic_scores"][metric]) for row in score_rows
            ),
            "zero_score_rows": sum(
                float(row["raw_semantic_scores"][metric]) == 0.0
                for row in score_rows
            ),
        }
        for metric in METRICS
    }
    source_counts = Counter(str(row["source_release"]) for row in score_rows)
    artifacts = {
        "row_scores": _file_binding(row_path, rows=EXPECTED_ROWS),
        "generation_diagnostic_failures": _file_binding(
            failure_path, rows=len(failures)
        ),
        "meeting_cell_deltas": _file_binding(
            meeting_path, rows=EXPECTED_MEETINGS * len(TOPICS) * len(ARMS)
        ),
        "bootstrap_draws": _file_binding(draw_path, rows=draws),
        "bootstrap_results": _file_binding(results_path, sealed=True),
    }

    if _file_binding(plan_path, sealed=True) != plan_binding:
        raise LooK10ScoreError("score plan changed during finalization")
    if _file_binding(bert_manifest_path, sealed=True) != bert_binding:
        raise LooK10ScoreError("BERTScore increment manifest changed during finalization")
    if _file_binding(mpnet_manifest_path, sealed=True) != mpnet_binding:
        raise LooK10ScoreError("MPNet increment manifest changed during finalization")
    _assert_binding(plan["inputs"]["old_k5_row_scores"], rows=EXPECTED_INCREMENT_ROWS)
    _assert_binding(
        plan["inputs"]["increment_canonical_generations"],
        rows=EXPECTED_INCREMENT_ROWS,
    )
    sources = _assert_source_bundle(plan.get("sources"))

    manifest = seal_manifest(
        {
            "schema_version": MANIFEST_SCHEMA,
            "status": "complete",
            "created_at_utc": _utc_now(),
            "evaluation_id": SCORE_EVALUATION_ID,
            "operation": "reuse_sealed_K5_scores_score_increment_only_combine_K10_and_bootstrap",
            "inputs": {
                "score_plan": plan_binding,
                "old_k5_generation_manifest": plan["inputs"][
                    "old_k5_generation_manifest"
                ],
                "old_k5_score_manifest": plan["inputs"]["old_k5_score_manifest"],
                "old_k5_row_scores": plan["inputs"]["old_k5_row_scores"],
                "increment_generation_manifest": plan["inputs"][
                    "increment_generation_manifest"
                ],
                "increment_canonical_generations": plan["inputs"][
                    "increment_canonical_generations"
                ],
                "semantic_manifest": plan["inputs"]["semantic_manifest"],
                "excluded_legacy_nf4_k10": plan["inputs"][
                    "excluded_legacy_nf4_k10"
                ],
                "bertscore_increment_metric_manifest": bert_binding,
                "mpnet_increment_metric_manifest": mpnet_binding,
            },
            "replicate_mapping": plan["replicate_mapping"],
            "coverage": {
                **plan["coverage"]["combined"],
                "semantic_scored_rows_combined": len(score_rows),
                "semantic_rows_reused_from_sealed_k5": source_counts[
                    "sealed_k5_score_release"
                ],
                "semantic_rows_newly_scored": source_counts[
                    "increment_generation_k6_to_k10"
                ],
                "old_k5_rows_rescored": 0,
                "primary_included_rows": len(score_rows),
                "primary_excluded_rows": 0,
                "gate_filtered_rows": 0,
                "gate_zero_penalized_rows": 0,
                "generation_diagnostic_failure_rows": len(failures),
                "generation_diagnostic_failure_counts": dict(
                    sorted(failure_counts.items())
                ),
                "meeting_cells": len(meeting_cells),
                "bootstrap_draws": draws,
            },
            "semantic_summary": semantic_summary,
            "semantic_policy": plan["semantic_policy"],
            "estimand": computation.results["estimand"],
            "bootstrap_contract": computation.results["bootstrap_contract"],
            "k_adequacy": computation.results["k_adequacy"],
            "execution": {
                "semantic_topology_increment_only": {
                    "bertscore": bert_manifest["execution"],
                    "mpnet": mpnet_manifest["execution"],
                },
                "bootstrap_device": "cpu",
                "python": sys.executable,
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "network": False,
            },
            "artifacts": artifacts,
            "sources": sources,
            "limitations": computation.results["limitations"],
        }
    )
    _write_new_json(final_path, manifest)
    validate_manifest_integrity(manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument(
        "--old-generation-manifest",
        type=Path,
        default=DEFAULT_OLD_GENERATION_MANIFEST,
    )
    prepare.add_argument(
        "--old-generation-manifest-sha256",
        default=DEFAULT_OLD_GENERATION_MANIFEST_SHA256,
    )
    prepare.add_argument(
        "--old-score-manifest", type=Path, default=DEFAULT_OLD_SCORE_MANIFEST
    )
    prepare.add_argument(
        "--old-score-manifest-sha256", default=DEFAULT_OLD_SCORE_MANIFEST_SHA256
    )
    prepare.add_argument(
        "--increment-manifest", type=Path, default=DEFAULT_INCREMENT_MANIFEST
    )
    prepare.add_argument("--increment-manifest-sha256", required=True)
    prepare.add_argument(
        "--increment-cohort", type=Path, default=DEFAULT_INCREMENT_COHORT
    )
    prepare.add_argument("--increment-cohort-sha256", required=True)
    prepare.add_argument(
        "--semantic-manifest", type=Path, default=DEFAULT_SEMANTIC_MANIFEST
    )
    prepare.add_argument(
        "--semantic-manifest-sha256", default=DEFAULT_SEMANTIC_MANIFEST_SHA256
    )
    prepare.add_argument(
        "--legacy-nf4-root", type=Path, default=DEFAULT_LEGACY_NF4_ROOT
    )
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    metric = sub.add_parser("score-metric")
    metric.add_argument("--metric", choices=METRIC_WORKERS, required=True)
    metric.add_argument("--plan", type=Path, required=True)
    metric.add_argument("--plan-sha256", required=True)
    metric.add_argument("--output-dir", type=Path, required=True)
    metric.add_argument(
        "--physical-gpu-index", choices=(0, 1), type=int, required=True
    )
    metric.add_argument("--batch-size", type=int, required=True)
    metric.add_argument("--gpu-wait-timeout-seconds", type=int, default=172_800)
    metric.add_argument("--gpu-poll-seconds", type=int, default=30)

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--plan", type=Path, required=True)
    finalize.add_argument("--plan-sha256", required=True)
    finalize.add_argument("--bertscore-manifest", type=Path, required=True)
    finalize.add_argument("--bertscore-manifest-sha256", required=True)
    finalize.add_argument("--mpnet-manifest", type=Path, required=True)
    finalize.add_argument("--mpnet-manifest-sha256", required=True)
    finalize.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    finalize.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    finalize.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_score_plan(
                old_generation_manifest_path=args.old_generation_manifest,
                old_generation_manifest_sha256=args.old_generation_manifest_sha256,
                old_score_manifest_path=args.old_score_manifest,
                old_score_manifest_sha256=args.old_score_manifest_sha256,
                increment_manifest_path=args.increment_manifest,
                increment_manifest_sha256=args.increment_manifest_sha256,
                increment_cohort_path=args.increment_cohort,
                increment_cohort_sha256=args.increment_cohort_sha256,
                semantic_manifest_path=args.semantic_manifest,
                semantic_manifest_sha256=args.semantic_manifest_sha256,
                legacy_nf4_root=args.legacy_nf4_root,
                output_dir=args.output_dir,
            )
            summary = {
                "status": result["status"],
                "operation": "prepare",
                "combined_rows": result["coverage"]["combined"]["rows"],
                "increment_rows_to_score": result["coverage"][
                    "increment_k6_to_k10"
                ]["rows"],
                "payload_sha256": result["integrity"]["payload_sha256"],
            }
        elif args.command == "score-metric":
            result = score_metric(
                metric=args.metric,
                plan_path=args.plan,
                plan_sha256=args.plan_sha256,
                output_dir=args.output_dir,
                physical_gpu_index=args.physical_gpu_index,
                batch_size=args.batch_size,
                gpu_wait_timeout_seconds=args.gpu_wait_timeout_seconds,
                gpu_poll_seconds=args.gpu_poll_seconds,
            )
            summary = {
                "status": result["status"],
                "operation": "score-metric",
                "metric": result["metric_worker"],
                "increment_rows": result["coverage"]["rows"],
                "payload_sha256": result["integrity"]["payload_sha256"],
            }
        else:
            result = finalize_score(
                plan_path=args.plan,
                plan_sha256=args.plan_sha256,
                bert_manifest_path=args.bertscore_manifest,
                bert_manifest_sha256=args.bertscore_manifest_sha256,
                mpnet_manifest_path=args.mpnet_manifest,
                mpnet_manifest_sha256=args.mpnet_manifest_sha256,
                output_dir=args.output_dir,
                draws=args.bootstrap_draws,
                seed=args.bootstrap_seed,
            )
            summary = {
                "status": result["status"],
                "operation": "finalize",
                "combined_rows": result["coverage"]["rows"],
                "newly_scored_rows": result["coverage"][
                    "semantic_rows_newly_scored"
                ],
                "bootstrap_draws": result["coverage"]["bootstrap_draws"],
                "k_adequacy": result["k_adequacy"]["aggregate_status"],
                "payload_sha256": result["integrity"]["payload_sha256"],
            }
    except (LooK10ScoreError, k5.LooK5ScoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(_canonical(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
