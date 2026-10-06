"""Score and bootstrap the sealed CHK3 Core8 LOO vLLM K=5 panel.

The primary analysis is deliberately raw-only: every one of the 10,880
generated answers is sent to the pinned MPNet and BERTScore backends.  The
historical generation/fidelity gates are recomputed and retained as diagnostics
but never filter, zero, or reweight the primary LOO estimand.

The command is split into three write-once stages so the two semantic metrics
can run concurrently on independent GPUs without changing the statistical
contract:

``prepare``
    Deep-validates the sealed generation run and writes an immutable score plan.
``score-metric``
    Scores all rows for exactly one metric and seals the row-level intermediate.
``finalize``
    Aligns both metric intermediates, writes complete row-level scores and
    diagnostics, and performs a shared-index hierarchical paired bootstrap.

For meeting h, topic t, intervention arm a, semantic metric m, and replicate r,
the paired effect is ``delta[h,a,t,m,r] = score(full)-score(intervention)``.
The point estimate first averages the five paired replicates within meeting and
then gives each of the 128 meetings equal weight.  Each bootstrap draw resamples
128 meetings with replacement and, for every sampled meeting occurrence,
resamples five paired replicate indices with replacement.  The same meeting and
replicate index arrays are shared by both metrics and all 16 intervention cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import statistics
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from jobs.eval import eval_chk3_core8_loo_vllm_k5_dual_dp1 as generation
from jobs.eval import (
    eval_chk3_beta_core8_merged_vllm_k5_dual_dp1_stochastic_schedule_v2
    as gpu_runtime,
)
from jobs.eval import score_chk3_native_checkpoint_sweep_semantic as semantic
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ID = "chk3-cp318-core8-loo-n128-vllm-k5-dual-dp1-v1"
SCORE_EVALUATION_ID = f"{EVALUATION_ID}-raw-semantic-b10000-v2"

PLAN_SCHEMA = "chk3-core8-loo-vllm-k5-score-plan-v2"
METRIC_ROW_SCHEMA = "chk3-core8-loo-vllm-k5-semantic-metric-row-v2"
METRIC_MANIFEST_SCHEMA = "chk3-core8-loo-vllm-k5-semantic-metric-manifest-v2"
SCORE_ROW_SCHEMA = "chk3-core8-loo-vllm-k5-score-row-v2"
FAILURE_ROW_SCHEMA = "chk3-core8-loo-vllm-k5-generation-diagnostic-failure-v2"
MEETING_CELL_SCHEMA = "chk3-core8-loo-vllm-k5-meeting-cell-delta-v2"
BOOTSTRAP_DRAW_SCHEMA = "chk3-core8-loo-vllm-k5-bootstrap-draw-v2"
RESULT_SCHEMA = "chk3-core8-loo-vllm-k5-bootstrap-results-v2"
MANIFEST_SCHEMA = "chk3-core8-loo-vllm-k5-score-manifest-v2"
INDEX_PLAN_SCHEMA = "chk3-core8-loo-vllm-k5-shared-hierarchical-index-plan-v2"

EXPECTED_ROWS = 10_880
EXPECTED_MEETINGS = 128
EXPECTED_VARIANTS = 17
EXPECTED_REPLICATES = 5
EXPECTED_FULL_ROWS = EXPECTED_MEETINGS * EXPECTED_REPLICATES
EXPECTED_INTERVENTION_ROWS = EXPECTED_ROWS - EXPECTED_FULL_ROWS
EXPECTED_EMPTY_ANSWER_ROWS = 8
REPLICATE_SEEDS = generation.REPLICATE_SEEDS
ARMS = ("exact_deletion", "token_matched_neutral")
TOPICS = (
    "Consumer-Price-Index-(CPI)",
    "GDP-Growth",
    "Government-Purchases",
    "Housing-Starts",
    "Industrial-Production",
    "Labour-Market",
    "Money-Supply",
    "Unemployment-Rate",
)
METRICS = ("mpnet_cosine", "bertscore_f1")
METRIC_WORKERS = ("mpnet", "bertscore")

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_817
CONFIDENCE = 0.95
K5_K4_POINT_DRIFT_THRESHOLD = 0.005
K5_K4_CI_ENDPOINT_DRIFT_THRESHOLD = 0.010
LORO_MAX_DRIFT_THRESHOLD = 0.010
DECODING_VARIANCE_SHARE_THRESHOLD = 0.20
K_PREFIX_POLICY = (
    "cumulative fresh vLLM replicates K=1..5 in frozen seed order; "
    "historical greedy K1 excluded"
)
K_ADEQUACY_FREEZE_STATUS = (
    "frozen_before_any_K5_semantic_scores_were_computed_or_read"
)
K_INCREASE_RULES = (
    "any cell has absolute K5-minus-K4 point drift above 0.005",
    "any cell has either absolute K5-minus-K4 CI endpoint drift above 0.010",
    "any cell has leave-one-replicate max absolute point drift above 0.010",
    "any cell has estimated decoding variance share above 0.20",
    "any K5-zero-excluding cell is not same-direction zero-excluding at K3 and K4",
    "any K5-zero-excluding cell reverses point sign in K3, K4, K5, or a leave-one-replicate estimate",
    "replicate coverage is incomplete",
)

DEFAULT_ROOT = ROOT / (
    "output/evaluation/main/"
    "chk3_cp318_core8_loo_vllm_k5_n128_1993_2008_20260817_v1"
)
DEFAULT_RUN_MANIFEST = (
    DEFAULT_ROOT / "generation_formal_n2176_k5_v1/chk3/manifest.json"
)
DEFAULT_COHORT = DEFAULT_ROOT / "preparation_v2/cohort_n2176_k5.v2.json"
DEFAULT_COHORT_SHA256 = (
    "26c1a06231fd8b10a0ccf2a9de22a59b7d657696da88a18981bad6fb82762e1c"
)
DEFAULT_SEMANTIC_MANIFEST = ROOT / "configs/main/checkpoint_eval_semantic_models.json"
DEFAULT_SEMANTIC_MANIFEST_SHA256 = (
    "639ca25cf4f5e695b0c6cfeeb47aba75d3e6dfeda1ad87510c23bb4f4a378bf7"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "score_raw_semantic_dual_gpu_b10000_v2"
HISTORICAL_K1_SCORE_MANIFEST = ROOT / (
    "output/evaluation/main/"
    "chk3_cp318_core8_loo_full_n128_1993_2008_20260814_v1/"
    "score_v1/manifest.json"
)
HISTORICAL_K1_SCORE_MANIFEST_SHA256 = (
    "4226551fd40ea9b02678624756295d868929fecec8664352ff5282134dde8df8"
)
HISTORICAL_K1_BOOTSTRAP_MANIFEST = ROOT / (
    "output/evaluation/main/"
    "chk3_cp318_core8_loo_full_n128_1993_2008_20260814_v1/"
    "meeting_paired_bootstrap_b10000_v1/manifest.json"
)
SOURCE_ROLES = (
    "scorer",
    "semantic_wrapper",
    "semantic_long_text_engine",
    "native_generation_and_gate_contract",
    "native_probe_gate_engine",
    "generation_validator",
    "gpu_lease_runtime",
)


class LooK5ScoreError(RuntimeError):
    """A frozen input, semantic result, or statistical invariant failed."""


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
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise LooK5ScoreError(f"value is not canonical JSON: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise LooK5ScoreError(f"missing regular JSON file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LooK5ScoreError(f"cannot read JSON {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise LooK5ScoreError(f"JSON root is not an object: {resolved}")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise LooK5ScoreError(f"missing regular JSONL file: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise LooK5ScoreError(
                        f"blank JSONL line at {resolved}:{line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise LooK5ScoreError(
                        f"JSONL row is not an object at {resolved}:{line_number}"
                    )
                yield value
    except json.JSONDecodeError as exc:
        raise LooK5ScoreError(f"invalid JSONL {resolved}: {exc}") from exc


def _file_binding(
    path: Path, *, rows: int | None = None, sealed: bool = False
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise LooK5ScoreError(f"cannot bind missing regular file: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if rows is not None:
        with resolved.open("rb") as handle:
            observed = sum(1 for _ in handle)
        if observed != rows:
            raise LooK5ScoreError(
                f"row count drift for {resolved}: expected={rows}, observed={observed}"
            )
        result["rows"] = observed
    if sealed:
        try:
            result["payload_sha256"] = validate_manifest_integrity(
                _read_json(resolved)
            )
        except Exception as exc:
            raise LooK5ScoreError(
                f"sealed manifest integrity failed for {resolved}: {exc}"
            ) from exc
    return result


def _assert_binding(
    recorded: Mapping[str, Any], *, rows: int | None = None, sealed: bool = False
) -> Path:
    path = Path(str(recorded.get("path") or "")).expanduser().resolve()
    observed = _file_binding(path, rows=rows, sealed=sealed)
    for field in ("path", "sha256", "bytes", "rows", "payload_sha256"):
        if field in recorded and recorded.get(field) != observed.get(field):
            raise LooK5ScoreError(
                f"artifact binding drift for {path} at {field}: "
                f"recorded={recorded.get(field)!r}, observed={observed.get(field)!r}"
            )
    return path


def _source_bundle() -> dict[str, dict[str, Any]]:
    return {
        "scorer": _file_binding(Path(__file__)),
        "semantic_wrapper": _file_binding(Path(semantic.__file__)),
        "semantic_long_text_engine": _file_binding(
            Path(semantic.semantic_eval.__file__)
        ),
        "native_generation_and_gate_contract": _file_binding(
            Path(semantic.native_eval.__file__)
        ),
        "native_probe_gate_engine": _file_binding(
            Path(semantic.native_eval.native_probe.__file__)
        ),
        "generation_validator": _file_binding(Path(generation.__file__)),
        "gpu_lease_runtime": _file_binding(Path(gpu_runtime.__file__)),
    }


def _assert_source_bundle(sources: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(sources, Mapping) or set(sources) != set(SOURCE_ROLES):
        raise LooK5ScoreError("score implementation source inventory drift")
    result: dict[str, dict[str, Any]] = {}
    for role in SOURCE_ROLES:
        binding = sources.get(role)
        if not isinstance(binding, Mapping):
            raise LooK5ScoreError(f"score source binding missing: {role}")
        _assert_binding(binding)
        result[role] = dict(binding)
    return result


def _publish_new_file(path: Path, writer: Any) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists() or resolved.is_symlink():
        raise LooK5ScoreError(f"refusing to overwrite artifact: {resolved}")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".staging", dir=str(resolved.parent)
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, resolved, follow_symlinks=False)
        except FileExistsError as exc:
            raise LooK5ScoreError(f"refusing to overwrite artifact: {resolved}") from exc
        _fsync_directory(resolved.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    def writer(handle: Any) -> None:
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

    _publish_new_file(path, writer)


def _write_new_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    def writer(handle: Any) -> None:
        for row in rows:
            handle.write(_canonical(row) + "\n")

    _publish_new_file(path, writer)


def _tuple_key(row: Mapping[str, Any]) -> dict[str, Any]:
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


def _expected_variant(variant_rank: int) -> tuple[str, str | None]:
    if variant_rank == 0:
        return "full", None
    topic = TOPICS[(variant_rank - 1) // 2]
    arm = ARMS[(variant_rank - 1) % 2]
    return arm, topic


def validate_generation_panel(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate exact K5 pairing and return a compact immutable inventory."""

    if len(rows) != EXPECTED_ROWS:
        raise LooK5ScoreError(
            f"generation panel must contain {EXPECTED_ROWS} rows, observed={len(rows)}"
        )
    digest = hashlib.sha256()
    meetings: dict[int, str] = {}
    tuple_keys: set[tuple[Any, ...]] = set()
    sample_ids: dict[tuple[int, int], str] = {}
    paired_blocks: dict[tuple[int, int], tuple[int, str]] = {}
    arm_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    finish_counts: Counter[str] = Counter()
    gate_failure_counts: Counter[str] = Counter()
    gate_failure_rows = 0
    empty_answer_rows = 0
    input_truncation_rows = 0

    for position, row in enumerate(rows):
        if row.get("schema_version") != generation.ROW_SCHEMA:
            raise LooK5ScoreError(f"generation row schema drift at {position}")
        if row.get("evaluation_id") != EVALUATION_ID or row.get("model_id") != "chk3":
            raise LooK5ScoreError(f"generation identity drift at {position}")
        meeting_rank = row.get("meeting_rank")
        variant_rank = row.get("variant_rank")
        replicate_id = row.get("replicate_id")
        absolute_index = row.get("absolute_case_index")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (
            meeting_rank,
            variant_rank,
            replicate_id,
            absolute_index,
        )):
            raise LooK5ScoreError(f"non-integer panel coordinate at {position}")
        assert isinstance(meeting_rank, int)
        assert isinstance(variant_rank, int)
        assert isinstance(replicate_id, int)
        assert isinstance(absolute_index, int)
        expected_index = (
            (meeting_rank * EXPECTED_VARIANTS + variant_rank) * EXPECTED_REPLICATES
            + replicate_id
        )
        expected_arm, expected_topic = _expected_variant(variant_rank)
        if (
            not 0 <= meeting_rank < EXPECTED_MEETINGS
            or not 0 <= variant_rank < EXPECTED_VARIANTS
            or not 0 <= replicate_id < EXPECTED_REPLICATES
            or absolute_index != position
            or absolute_index != expected_index
            or row.get("replicate_seed") != REPLICATE_SEEDS[replicate_id]
            or row.get("arm") != expected_arm
            or row.get("intervention_topic") != expected_topic
            or row.get("input_truncated") is not False
        ):
            raise LooK5ScoreError(f"canonical K5 panel drift at row {position}")
        meeting_id = row.get("meeting_id")
        sample_id = row.get("variant_sample_id", row.get("sample_id"))
        if not isinstance(meeting_id, str) or not meeting_id:
            raise LooK5ScoreError(f"missing meeting ID at row {position}")
        if not isinstance(sample_id, str) or not sample_id:
            raise LooK5ScoreError(f"missing variant sample ID at row {position}")
        prior_meeting = meetings.setdefault(meeting_rank, meeting_id)
        if prior_meeting != meeting_id:
            raise LooK5ScoreError(f"meeting-rank identity drift at row {position}")
        sample_key = (meeting_rank, variant_rank)
        prior_sample = sample_ids.setdefault(sample_key, sample_id)
        if prior_sample != sample_id:
            raise LooK5ScoreError(f"sample identity changed across replicates at {position}")
        paired_key = (meeting_rank, replicate_id)
        paired_value = (int(row.get("row_seed")), str(row.get("paired_seed_key")))
        prior_paired = paired_blocks.setdefault(paired_key, paired_value)
        if (
            prior_paired != paired_value
            or row.get("paired_block_id")
            != meeting_rank * EXPECTED_REPLICATES + replicate_id
        ):
            raise LooK5ScoreError(f"common-random-number pairing drift at {position}")
        key = tuple(_tuple_key(row).values())
        if key in tuple_keys:
            raise LooK5ScoreError(f"duplicated generation tuple at {position}")
        tuple_keys.add(key)
        digest.update((_canonical(_tuple_key(row)) + "\n").encode("utf-8"))
        arm_counts[expected_arm] += 1
        if expected_topic is not None:
            topic_counts[expected_topic] += 1
        finish_counts[str(row.get("finish_reason"))] += 1
        if not str(row.get("answer") or "").strip():
            empty_answer_rows += 1
        if bool(row.get("input_truncated")):
            input_truncation_rows += 1
        failures = row.get("preregistered_core_failures")
        if not isinstance(failures, list) or any(
            not isinstance(value, str) for value in failures
        ):
            raise LooK5ScoreError(f"generation diagnostic failures drift at {position}")
        if failures:
            gate_failure_rows += 1
            gate_failure_counts.update(failures)

    if (
        set(meetings) != set(range(EXPECTED_MEETINGS))
        or len(sample_ids) != EXPECTED_MEETINGS * EXPECTED_VARIANTS
        or len(paired_blocks) != EXPECTED_MEETINGS * EXPECTED_REPLICATES
        or any(
            sum(
                1
                for row in rows
                if row["meeting_rank"] == meeting_rank
                and row["replicate_id"] == replicate_id
            )
            != EXPECTED_VARIANTS
            for meeting_rank, replicate_id in paired_blocks
        )
        or arm_counts
        != Counter(
            {
                "full": EXPECTED_FULL_ROWS,
                "exact_deletion": EXPECTED_MEETINGS * len(TOPICS) * EXPECTED_REPLICATES,
                "token_matched_neutral": EXPECTED_MEETINGS
                * len(TOPICS)
                * EXPECTED_REPLICATES,
            }
        )
        or topic_counts
        != Counter(
            {topic: EXPECTED_MEETINGS * len(ARMS) * EXPECTED_REPLICATES for topic in TOPICS}
        )
        or empty_answer_rows != EXPECTED_EMPTY_ANSWER_ROWS
    ):
        raise LooK5ScoreError("generation K5 matrix coverage is not exact")

    return {
        "rows": len(rows),
        "meetings": len(meetings),
        "variant_prompts": len(sample_ids),
        "variants_per_meeting": EXPECTED_VARIANTS,
        "replicates": EXPECTED_REPLICATES,
        "replicate_seeds": list(REPLICATE_SEEDS),
        "full_rows": arm_counts["full"],
        "intervention_rows": len(rows) - arm_counts["full"],
        "arm_rows": dict(sorted(arm_counts.items())),
        "topic_rows": {topic: topic_counts[topic] for topic in TOPICS},
        "paired_blocks": len(paired_blocks),
        "tuple_key_sha256": digest.hexdigest(),
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "empty_answer_rows": empty_answer_rows,
        "input_truncation_rows": input_truncation_rows,
        "generation_diagnostic_failure_rows": gate_failure_rows,
        "generation_diagnostic_failure_counts": dict(sorted(gate_failure_counts.items())),
    }


def _historical_k1_binding() -> dict[str, Any]:
    score_binding = _file_binding(HISTORICAL_K1_SCORE_MANIFEST, sealed=True)
    if score_binding["sha256"] != HISTORICAL_K1_SCORE_MANIFEST_SHA256:
        raise LooK5ScoreError("historical K1 score manifest binding drift")
    bootstrap_binding = _file_binding(HISTORICAL_K1_BOOTSTRAP_MANIFEST, sealed=True)
    return {
        "role": "deterministic_greedy_nf4_anchor_only",
        "included_as_stochastic_replicate": False,
        "included_in_k5_point_estimates_or_intervals": False,
        "formal_contrast_with_k5_authorized": False,
        "reason": (
            "K1 used greedy Transformers NF4 inference whereas K5 uses stochastic "
            "vLLM BF16 inference; the decoding and numerical distributions differ"
        ),
        "score_manifest": score_binding,
        "meeting_bootstrap_manifest": bootstrap_binding,
    }


def prepare_score_plan(
    *,
    run_manifest_path: Path,
    run_manifest_sha256: str,
    cohort_path: Path,
    cohort_sha256: str,
    semantic_manifest_path: Path,
    semantic_manifest_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Deep-validate generation once and freeze the downstream score plan."""

    output_dir = output_dir.expanduser().resolve()
    plan_path = output_dir / "score_plan.json"
    if output_dir.exists() or output_dir.is_symlink():
        raise LooK5ScoreError(f"score output root already exists: {output_dir}")
    run_binding = _file_binding(run_manifest_path, sealed=True)
    if run_binding["sha256"] != run_manifest_sha256:
        raise LooK5ScoreError("generation manifest SHA-256 differs from CLI binding")
    cohort_binding = _file_binding(cohort_path, sealed=True)
    if cohort_binding["sha256"] != cohort_sha256:
        raise LooK5ScoreError("cohort SHA-256 differs from CLI binding")
    semantic_binding = _file_binding(semantic_manifest_path, sealed=True)
    if semantic_binding["sha256"] != semantic_manifest_sha256:
        raise LooK5ScoreError("semantic manifest SHA-256 differs from CLI binding")
    if semantic_binding["sha256"] != DEFAULT_SEMANTIC_MANIFEST_SHA256:
        raise LooK5ScoreError("semantic manifest is not the pinned formal inventory")

    try:
        validated = generation.load_and_validate_run(
            run_manifest_path,
            cohort_path=cohort_path,
            cohort_sha256=cohort_sha256,
            expected_model_id="chk3",
            expected_scope="formal_merged_panel",
            max_num_seqs=generation.SELECTED_MAX_NUM_SEQS,
        )
    except Exception as exc:
        raise LooK5ScoreError(f"generation deep validation failed: {exc}") from exc
    if validated["manifest_binding"] != run_binding:
        raise LooK5ScoreError("generation deep validator returned a different binding")
    coverage = validate_generation_panel(validated["results"])
    artifact = validated["manifest"].get("artifacts", {}).get("canonical_generations")
    if not isinstance(artifact, Mapping):
        raise LooK5ScoreError("generation manifest lacks canonical generations")
    generation_path = _assert_binding(artifact, rows=EXPECTED_ROWS)
    if generation_path.parent != run_manifest_path.expanduser().resolve().parent:
        raise LooK5ScoreError("canonical generations escaped the generation root")

    output_dir.mkdir(parents=True, exist_ok=False)
    _fsync_directory(output_dir.parent)
    plan = seal_manifest(
        {
            "schema_version": PLAN_SCHEMA,
            "status": "complete",
            "created_at_utc": _utc_now(),
            "evaluation_id": SCORE_EVALUATION_ID,
            "operation": "freeze_raw_semantic_k5_scoring_plan",
            "inputs": {
                "generation_manifest": run_binding,
                "canonical_generations": dict(artifact),
                "cohort": cohort_binding,
                "semantic_manifest": semantic_binding,
                "historical_k1_anchor": _historical_k1_binding(),
            },
            "coverage": coverage,
            "semantic_policy": {
                "primary": "raw_scores_all_10880_rows",
                "generation_gates": "diagnostic_only_no_filter_no_zero_penalty",
                "empty_answers": "scored_by_frozen_long_text_backend_missing_side_policy",
                "metrics": list(METRICS),
                "network": False,
            },
            "estimand": {
                "replicate_delta": "score(full_h_r)-score(intervention_h_a_t_r)",
                "meeting_value": "mean_over_five_paired_replicate_deltas",
                "point_estimate": "equal_weight_mean_over_128_meetings",
                "positive_sign": "intervention_reduced_similarity_to_full_reference",
            },
            "bootstrap_contract": {
                "draws": BOOTSTRAP_DRAWS,
                "seed": BOOTSTRAP_SEED,
                "confidence": CONFIDENCE,
                "meeting_resampling": "128_with_replacement",
                "replicate_resampling": "5_with_replacement_per_sampled_meeting_occurrence",
                "paired_within_replicate": True,
                "shared_indices_across_all_32_cells": True,
                "primary_interval": "hierarchical_percentile",
                "meeting_only_interval": "fixed_K5_meeting_resampling_sensitivity",
            },
            "k_adequacy_rule": {
                "scope": "aggregate_128_meeting_estimand_not_per_meeting_inference",
                "status": K_ADEQUACY_FREEZE_STATUS,
                "stochastic_prefixes": K_PREFIX_POLICY,
                "k5_minus_k4_point_drift_threshold": K5_K4_POINT_DRIFT_THRESHOLD,
                "k5_minus_k4_ci_endpoint_drift_threshold": (
                    K5_K4_CI_ENDPOINT_DRIFT_THRESHOLD
                ),
                "leave_one_replicate_max_drift_threshold": LORO_MAX_DRIFT_THRESHOLD,
                "decoding_variance_share_threshold": DECODING_VARIANCE_SHARE_THRESHOLD,
                "increase_k_if": list(K_INCREASE_RULES),
            },
            "sources": _source_bundle(),
        }
    )
    _write_new_json(plan_path, plan)
    return plan


def load_score_plan(
    plan_path: Path, plan_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _file_binding(plan_path, sealed=True)
    if binding["sha256"] != plan_sha256:
        raise LooK5ScoreError("score plan SHA-256 differs from CLI binding")
    plan = _read_json(plan_path)
    if (
        plan.get("schema_version") != PLAN_SCHEMA
        or plan.get("status") != "complete"
        or plan.get("evaluation_id") != SCORE_EVALUATION_ID
        or plan.get("coverage", {}).get("rows") != EXPECTED_ROWS
        or plan.get("coverage", {}).get("empty_answer_rows")
        != EXPECTED_EMPTY_ANSWER_ROWS
        or plan.get("coverage", {}).get("tuple_key_sha256") is None
    ):
        raise LooK5ScoreError("score plan contract drift")
    inputs = plan.get("inputs")
    if not isinstance(inputs, Mapping):
        raise LooK5ScoreError("score plan input inventory missing")
    expected_bootstrap = {
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "confidence": CONFIDENCE,
        "meeting_resampling": "128_with_replacement",
        "replicate_resampling": "5_with_replacement_per_sampled_meeting_occurrence",
        "paired_within_replicate": True,
        "shared_indices_across_all_32_cells": True,
        "primary_interval": "hierarchical_percentile",
        "meeting_only_interval": "fixed_K5_meeting_resampling_sensitivity",
    }
    rule = plan.get("k_adequacy_rule")
    threshold_fields = (
        "k5_minus_k4_point_drift_threshold",
        "k5_minus_k4_ci_endpoint_drift_threshold",
        "leave_one_replicate_max_drift_threshold",
        "decoding_variance_share_threshold",
    )
    if (
        plan.get("bootstrap_contract") != expected_bootstrap
        or not isinstance(rule, Mapping)
        or rule.get("status") != K_ADEQUACY_FREEZE_STATUS
        or rule.get("stochastic_prefixes") != K_PREFIX_POLICY
        or rule.get("increase_k_if") != list(K_INCREASE_RULES)
        or rule.get("k5_minus_k4_point_drift_threshold")
        != K5_K4_POINT_DRIFT_THRESHOLD
        or rule.get("k5_minus_k4_ci_endpoint_drift_threshold")
        != K5_K4_CI_ENDPOINT_DRIFT_THRESHOLD
        or rule.get("leave_one_replicate_max_drift_threshold")
        != LORO_MAX_DRIFT_THRESHOLD
        or rule.get("decoding_variance_share_threshold")
        != DECODING_VARIANCE_SHARE_THRESHOLD
        or any(
            isinstance(rule.get(field), bool)
            or not isinstance(rule.get(field), (int, float))
            or not math.isfinite(float(rule[field]))
            or float(rule[field]) <= 0
            for field in threshold_fields
        )
    ):
        raise LooK5ScoreError("score plan bootstrap/K-adequacy contract drift")
    _assert_binding(inputs["generation_manifest"], sealed=True)
    _assert_binding(inputs["cohort"], sealed=True)
    _assert_binding(inputs["semantic_manifest"], sealed=True)
    _assert_binding(inputs["canonical_generations"], rows=EXPECTED_ROWS)
    historical = inputs.get("historical_k1_anchor")
    if not isinstance(historical, Mapping):
        raise LooK5ScoreError("score plan historical K1 anchor binding missing")
    _assert_binding(historical["score_manifest"], sealed=True)
    _assert_binding(historical["meeting_bootstrap_manifest"], sealed=True)
    _assert_source_bundle(plan.get("sources"))
    return plan, binding


def _load_generation_projection(
    plan: Mapping[str, Any], *, diagnostics: bool
) -> list[dict[str, Any]]:
    path = _assert_binding(
        plan["inputs"]["canonical_generations"], rows=EXPECTED_ROWS
    )
    projection: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for position, row in enumerate(_iter_jsonl(path)):
        if row.get("absolute_case_index") != position:
            raise LooK5ScoreError(f"generation canonical order drift at {position}")
        key = _tuple_key(row)
        digest.update((_canonical(key) + "\n").encode("utf-8"))
        answer = row.get("answer")
        reference = row.get("reference_minutes")
        if not isinstance(answer, str) or not isinstance(reference, str) or not reference:
            raise LooK5ScoreError(f"semantic text missing at row {position}")
        item = {
            **key,
            "sample_id": row.get("variant_sample_id", row.get("sample_id")),
            "answer": answer,
            "reference_minutes": reference,
            "completion_sha256": row.get("completion_sha256"),
            "answer_sha256": row.get("answer_sha256"),
            "generated_token_ids_sha256": row.get("generated_token_ids_sha256"),
            "reference_minutes_sha256": row.get("reference_minutes_sha256"),
            "finish_reason": row.get("finish_reason"),
            "input_truncated": row.get("input_truncated"),
            "content_tokens": row.get("content_tokens"),
            "full_token_4gram_repetition": row.get("full_token_4gram_repetition"),
            "tail_token_4gram_repetition": row.get("tail_token_4gram_repetition"),
        }
        if diagnostics:
            try:
                gate = semantic._recompute_core_hard_gate(row)
            except Exception as exc:
                raise LooK5ScoreError(
                    f"generation gate diagnostic failed at row {position}: {exc}"
                ) from exc
            stored_failures = sorted(row.get("preregistered_core_failures") or [])
            if (
                bool(row.get("preregistered_core_valid"))
                != bool(gate["preregistered_core_valid"])
                or stored_failures
                != sorted(gate["preregistered_core_failures"])
            ):
                raise LooK5ScoreError(
                    f"stored/recomputed generation gate drift at row {position}"
                )
            item["gate"] = gate
        projection.append(item)
    if len(projection) != EXPECTED_ROWS:
        raise LooK5ScoreError("generation projection row count drift")
    if digest.hexdigest() != plan["coverage"]["tuple_key_sha256"]:
        raise LooK5ScoreError("generation tuple-key digest changed after planning")
    return projection


def _metric_vector(values: Any, *, expected: int, label: str) -> list[float]:
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise LooK5ScoreError(f"{label} vector is not numeric") from exc
    if len(result) != expected:
        raise LooK5ScoreError(
            f"{label} vector length drift: expected={expected}, observed={len(result)}"
        )
    if any(not math.isfinite(value) or not -1.000001 <= value <= 1.000001 for value in result):
        raise LooK5ScoreError(f"{label} contains a non-finite/out-of-range score")
    return result


def _normalise_gpu_uuid(value: Any) -> str:
    """Return the 32 hexadecimal UUID digits used by both CUDA and nvidia-smi."""

    if isinstance(value, bytes):
        if len(value) != 16:
            raise LooK5ScoreError(
                f"CUDA device UUID is unavailable/invalid: {value!r}"
            )
        return value.hex()
    text = str(value).strip().casefold()
    if text.startswith("gpu-"):
        text = text[4:]
    if re.fullmatch(r"[0-9a-f]{32}", text):
        return text
    if re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
        r"[0-9a-f]{4}-[0-9a-f]{12}",
        text,
    ):
        return text.replace("-", "")
    raise LooK5ScoreError(f"CUDA device UUID is unavailable/invalid: {value!r}")


def _normalise_full_pci_bus_id(value: Any) -> tuple[int, int, int, int]:
    """Parse a full PCI domain:bus:device.function string fail-closed."""

    if not isinstance(value, str):
        raise LooK5ScoreError(f"full PCI bus ID must be a string: {value!r}")
    match = re.fullmatch(
        r"(?P<domain>[0-9a-fA-F]{4}|[0-9a-fA-F]{8}):"
        r"(?P<bus>[0-9a-fA-F]{2}):"
        r"(?P<device>[0-9a-fA-F]{2})\."
        r"(?P<function>[0-7])",
        value.strip(),
    )
    if match is None:
        raise LooK5ScoreError(f"PCI bus ID is unavailable/invalid: {value!r}")
    domain = int(match.group("domain"), 16)
    bus = int(match.group("bus"), 16)
    device = int(match.group("device"), 16)
    function = int(match.group("function"), 16)
    if domain > 0xFFFF or bus > 0xFF or device > 0x1F:
        raise LooK5ScoreError(f"PCI bus ID component is out of range: {value!r}")
    return domain, bus, device, function


def _verify_visible_cuda_device(
    torch_module: Any, identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind logical cuda:0 to the nvidia-smi target by UUID and PCI address.

    PyTorch 2.6 exposes ``pci_bus_id`` as the integer PCI bus number on this
    host (33/225), whereas nvidia-smi exposes the complete hexadecimal BDF
    (21/e1).  Integer observations are therefore compared to the parsed bus
    component; string observations must match the complete normalized BDF.
    UUID equality remains mandatory in both cases, so an index alone can never
    satisfy this identity check.
    """

    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() != 1:
        raise LooK5ScoreError("exactly one logical CUDA device is required")
    properties = torch_module.cuda.get_device_properties(0)
    observed_uuid = _normalise_gpu_uuid(getattr(properties, "uuid", None))
    expected_uuid = _normalise_gpu_uuid(identity.get("uuid"))
    if observed_uuid != expected_uuid:
        raise LooK5ScoreError(
            "logical cuda:0 UUID does not match physical target"
        )

    expected_pci = _normalise_full_pci_bus_id(identity.get("pci_bus_id"))
    if expected_pci[3] != 0:
        raise LooK5ScoreError(
            "physical target PCI function is not independently verifiable"
        )
    observed_pci = getattr(properties, "pci_bus_id", None)
    if isinstance(observed_pci, bool):
        raise LooK5ScoreError("logical cuda:0 PCI bus is unavailable/invalid")
    if isinstance(observed_pci, int):
        observed_domain = getattr(properties, "pci_domain_id", None)
        observed_device = getattr(properties, "pci_device_id", None)
        observed_components = (observed_domain, observed_pci, observed_device)
        if any(
            isinstance(component, bool) or not isinstance(component, int)
            for component in observed_components
        ):
            raise LooK5ScoreError(
                "logical cuda:0 PCI domain/bus/device is unavailable/invalid"
            )
        if (
            not 0 <= observed_domain <= 0xFFFF
            or not 0 <= observed_pci <= 0xFF
            or not 0 <= observed_device <= 0x1F
            or observed_components != expected_pci[:3]
        ):
            raise LooK5ScoreError(
                "logical cuda:0 PCI domain/bus/device does not match physical target"
            )
    else:
        if _normalise_full_pci_bus_id(observed_pci) != expected_pci:
            raise LooK5ScoreError(
                "logical cuda:0 PCI bus does not match physical target"
            )
    return {"logical_cuda_index": 0, **dict(identity)}


def _validate_metric_environment(physical_gpu_index: int) -> tuple[Any, dict[str, Any]]:
    if physical_gpu_index not in (0, 1):
        raise LooK5ScoreError("physical GPU index must be 0 or 1")
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise LooK5ScoreError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu_index):
        raise LooK5ScoreError(
            "CUDA_VISIBLE_DEVICES must expose exactly the requested physical GPU"
        )
    try:
        import torch

        identity = gpu_runtime._physical_gpu_identity(physical_gpu_index)
        logical = _verify_visible_cuda_device(torch, identity)
    except Exception as exc:
        raise LooK5ScoreError(f"semantic GPU identity validation failed: {exc}") from exc
    return torch, logical


def _assert_no_foreign_gpu_processes(physical_gpu_index: int) -> None:
    """Allow this scorer/descendants on CUDA and reject only foreign PIDs."""

    try:
        gpu_runtime._assert_no_foreign_processes(physical_gpu_index)
    except Exception as exc:
        raise LooK5ScoreError(
            f"foreign GPU process appeared during semantic scoring: {exc}"
        ) from exc


def _assert_empty_candidate_scores(
    *,
    metric: str,
    rows: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, Sequence[float]],
) -> int:
    empty_positions = [
        position for position, row in enumerate(rows) if not str(row["answer"]).strip()
    ]
    if len(empty_positions) != EXPECTED_EMPTY_ANSWER_ROWS:
        raise LooK5ScoreError(
            "empty-answer denominator differs from the sealed generation plan"
        )
    required = (
        ("bertscore_precision", "bertscore_recall", "bertscore_f1")
        if metric == "bertscore"
        else ("mpnet_cosine",)
    )
    if set(vectors) != set(required):
        raise LooK5ScoreError(f"{metric} vector inventory drift")
    for position in empty_positions:
        for name in required:
            if float(vectors[name][position]) != 0.0:
                raise LooK5ScoreError(
                    f"empty candidate must have raw {name}=0.0 at row {position}"
                )
    return len(empty_positions)


def _backend_chunk_audit(backend: Any, *, label: str) -> dict[str, Any]:
    metadata = backend.semantic_metadata() if hasattr(backend, "semantic_metadata") else None
    audit = metadata.get("chunk_audit") if isinstance(metadata, Mapping) else None
    if not isinstance(audit, Mapping) or audit.get("silent_truncation") is not False:
        raise LooK5ScoreError(f"{label} did not prove zero silent truncation")
    if audit.get("document_count") != EXPECTED_ROWS:
        raise LooK5ScoreError(f"{label} chunk-audit document denominator drift")
    return dict(metadata)


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
    """Score all rows for one semantic metric on one exclusively leased GPU."""

    if metric not in METRIC_WORKERS:
        raise LooK5ScoreError(f"unsupported metric worker: {metric}")
    if batch_size < 1:
        raise LooK5ScoreError("semantic batch size must be positive")
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise LooK5ScoreError(f"metric output already exists: {output_dir}")
    plan, plan_binding = load_score_plan(plan_path, plan_sha256)
    rows = _load_generation_projection(plan, diagnostics=False)
    torch, logical_gpu = _validate_metric_environment(physical_gpu_index)

    def wait_notice(reason: str, processes: list[dict[str, Any]]) -> None:
        print(
            _canonical(
                {
                    "status": "waiting_for_exclusive_semantic_gpu",
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
    with gpu_runtime.exclusive_gpu_lease(
        physical_gpu_index=physical_gpu_index,
        timeout_seconds=gpu_wait_timeout_seconds,
        poll_seconds=gpu_poll_seconds,
        on_wait=wait_notice,
    ) as lease:
        try:
            bert, mpnet, provenance = semantic.load_formal_semantic_backends(
                semantic_manifest_path=Path(plan["inputs"]["semantic_manifest"]["path"]),
                batch_size=batch_size,
                device="cuda:0",
            )
            candidates = [str(row["answer"]) for row in rows]
            references = [str(row["reference_minutes"]) for row in rows]
            if metric == "bertscore":
                raw = bert.score(candidates, references)
                if not isinstance(raw, Mapping) or set(raw) != {
                    "bertscore_precision",
                    "bertscore_recall",
                    "bertscore_f1",
                }:
                    raise LooK5ScoreError("BERTScore metric inventory drift")
                vectors = {
                    name: _metric_vector(raw[name], expected=EXPECTED_ROWS, label=name)
                    for name in sorted(raw)
                }
                backend_audit = _backend_chunk_audit(bert, label="BERTScore")
            else:
                raw = mpnet.score(candidates, references)
                vectors = {
                    "mpnet_cosine": _metric_vector(
                        raw, expected=EXPECTED_ROWS, label="mpnet_cosine"
                    )
                }
                backend_audit = _backend_chunk_audit(mpnet, label="MPNet")
            empty_score_rows = _assert_empty_candidate_scores(
                metric=metric, rows=rows, vectors=vectors
            )
        except LooK5ScoreError:
            raise
        except Exception as exc:
            raise LooK5ScoreError(f"{metric} semantic inference failed: {exc}") from exc
        torch.cuda.synchronize()
        _assert_no_foreign_gpu_processes(physical_gpu_index)
        lease_evidence = dict(lease)
    wall_seconds = time.monotonic() - started
    refreshed_plan, refreshed_binding = load_score_plan(plan_path, plan_sha256)
    if refreshed_plan != plan or refreshed_binding != plan_binding:
        raise LooK5ScoreError("score plan changed during semantic inference")
    source_bundle = _assert_source_bundle(refreshed_plan.get("sources"))

    metric_rows: list[dict[str, Any]] = []
    for position, source in enumerate(rows):
        scores = {name: values[position] for name, values in vectors.items()}
        metric_rows.append(
            {
                "schema_version": METRIC_ROW_SCHEMA,
                "metric_worker": metric,
                "source_generation_row": position + 1,
                "tuple_key": _tuple_key(source),
                "sample_id": source["sample_id"],
                "answer_sha256": source["answer_sha256"],
                "reference_minutes_sha256": source["reference_minutes_sha256"],
                "scores": scores,
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
            "inputs": {
                "score_plan": plan_binding,
                "canonical_generations": plan["inputs"]["canonical_generations"],
                "semantic_manifest": plan["inputs"]["semantic_manifest"],
            },
            "coverage": {
                "rows": len(metric_rows),
                "tuple_key_sha256": plan["coverage"]["tuple_key_sha256"],
                "raw_all_rows_scored": True,
                "gate_filtered_rows": 0,
                "gate_zero_penalized_rows": 0,
                "empty_answer_rows": empty_score_rows,
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
            "artifacts": {"row_scores": _file_binding(row_path, rows=EXPECTED_ROWS)},
            "sources": source_bundle,
        }
    )
    final_plan, final_plan_binding = load_score_plan(plan_path, plan_sha256)
    if final_plan != plan or final_plan_binding != plan_binding:
        raise LooK5ScoreError("score plan changed before metric publication")
    _assert_source_bundle(source_bundle)
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
        raise LooK5ScoreError(f"{metric} manifest SHA-256 differs from CLI binding")
    manifest = _read_json(manifest_path)
    metric_inputs = manifest.get("inputs")
    if (
        manifest.get("schema_version") != METRIC_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("evaluation_id") != SCORE_EVALUATION_ID
        or manifest.get("metric_worker") != metric
        or manifest.get("inputs", {}).get("score_plan") != plan_binding
        or manifest.get("coverage", {}).get("rows") != EXPECTED_ROWS
        or manifest.get("coverage", {}).get("raw_all_rows_scored") is not True
        or manifest.get("coverage", {}).get("gate_filtered_rows") != 0
        or manifest.get("coverage", {}).get("gate_zero_penalized_rows") != 0
        or manifest.get("coverage", {}).get("empty_answer_rows")
        != EXPECTED_EMPTY_ANSWER_ROWS
        or manifest.get("coverage", {}).get(
            "empty_answer_all_raw_metrics_exact_zero"
        )
        is not True
        or not isinstance(metric_inputs, Mapping)
        or metric_inputs.get("canonical_generations")
        != plan["inputs"]["canonical_generations"]
        or metric_inputs.get("semantic_manifest")
        != plan["inputs"]["semantic_manifest"]
        or manifest.get("sources") != plan.get("sources")
    ):
        raise LooK5ScoreError(f"{metric} metric manifest contract drift")
    row_binding = manifest.get("artifacts", {}).get("row_scores")
    if not isinstance(row_binding, Mapping):
        raise LooK5ScoreError(f"{metric} manifest lacks row scores")
    row_path = _assert_binding(row_binding, rows=EXPECTED_ROWS)
    if row_path.parent != manifest_path.expanduser().resolve().parent:
        raise LooK5ScoreError(f"{metric} row scores escaped metric directory")
    rows = list(_iter_jsonl(row_path))
    if len(rows) != EXPECTED_ROWS:
        raise LooK5ScoreError(f"{metric} row count drift")
    _assert_source_bundle(manifest.get("sources"))
    return manifest, rows, binding


def _finite_score(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise LooK5ScoreError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LooK5ScoreError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or not -1.000001 <= number <= 1.000001:
        raise LooK5ScoreError(f"{label} is non-finite/outside similarity range")
    return number


def build_scored_rows(
    *,
    generation_rows: Sequence[Mapping[str, Any]],
    bert_rows: Sequence[Mapping[str, Any]],
    mpnet_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not (
        len(generation_rows) == len(bert_rows) == len(mpnet_rows) == EXPECTED_ROWS
    ):
        raise LooK5ScoreError("semantic/generation alignment denominator drift")
    scored: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, (source, bert, mpnet) in enumerate(
        zip(generation_rows, bert_rows, mpnet_rows, strict=True)
    ):
        expected_key = _tuple_key(source)
        for worker, row in (("bertscore", bert), ("mpnet", mpnet)):
            if (
                row.get("schema_version") != METRIC_ROW_SCHEMA
                or row.get("metric_worker") != worker
                or row.get("source_generation_row") != position + 1
                or row.get("tuple_key") != expected_key
                or row.get("sample_id") != source["sample_id"]
                or row.get("answer_sha256") != source["answer_sha256"]
                or row.get("reference_minutes_sha256")
                != source["reference_minutes_sha256"]
            ):
                raise LooK5ScoreError(f"semantic row alignment drift at {position}")
        bert_scores = bert.get("scores")
        mpnet_scores = mpnet.get("scores")
        if not isinstance(bert_scores, Mapping) or not isinstance(mpnet_scores, Mapping):
            raise LooK5ScoreError(f"semantic score payload missing at {position}")
        bert_f1 = _finite_score(
            bert_scores.get("bertscore_f1"), label=f"BERTScore-F1[{position}]"
        )
        mpnet_cosine = _finite_score(
            mpnet_scores.get("mpnet_cosine"), label=f"MPNet[{position}]"
        )
        empty_answer = not str(source.get("answer") or "").strip()
        if empty_answer and any(
            value != 0.0
            for value in (
                mpnet_cosine,
                _finite_score(
                    bert_scores.get("bertscore_precision"),
                    label=f"BERTScore-P[{position}]",
                ),
                _finite_score(
                    bert_scores.get("bertscore_recall"),
                    label=f"BERTScore-R[{position}]",
                ),
                bert_f1,
            )
        ):
            raise LooK5ScoreError(
                f"empty candidate raw semantic scores are not all zero at {position}"
            )
        gate = source.get("gate")
        if not isinstance(gate, Mapping):
            raise LooK5ScoreError(f"recomputed diagnostic gate missing at {position}")
        generation_metrics = {
            "structure_delivery": bool(
                gate["delivery_valid"] and gate["native_structure_valid"]
            ),
            "numeric_fidelity": bool(gate["numeric_multiset_preserved"]),
            "date_fidelity": bool(gate["date_set_preserved"]),
            "degeneration_free": bool(gate["degeneration_free"]),
        }
        core_valid = bool(gate["preregistered_core_valid"])
        item = {
            "schema_version": SCORE_ROW_SCHEMA,
            "source_generation_row": position + 1,
            "tuple_key": expected_key,
            "sample_id": source["sample_id"],
            "meeting_id": source["meeting_id"],
            "meeting_rank": source["meeting_rank"],
            "variant_rank": source["variant_rank"],
            "arm": source["arm"],
            "intervention_topic": source["intervention_topic"],
            "replicate_id": source["replicate_id"],
            "replicate_seed": source["replicate_seed"],
            "row_seed": source["row_seed"],
            "completion_sha256": source["completion_sha256"],
            "answer_sha256": source["answer_sha256"],
            "generated_token_ids_sha256": source["generated_token_ids_sha256"],
            "reference_minutes_sha256": source["reference_minutes_sha256"],
            "finish_reason": source["finish_reason"],
            "input_truncated": source["input_truncated"],
            "generated_tokens": source["content_tokens"],
            "empty_answer": empty_answer,
            "full_token_4gram_repetition": source["full_token_4gram_repetition"],
            "tail_token_4gram_repetition": source["tail_token_4gram_repetition"],
            "raw_semantic_scores": {
                "mpnet_cosine": mpnet_cosine,
                "bertscore_precision": _finite_score(
                    bert_scores.get("bertscore_precision"),
                    label=f"BERTScore-P[{position}]",
                ),
                "bertscore_recall": _finite_score(
                    bert_scores.get("bertscore_recall"),
                    label=f"BERTScore-R[{position}]",
                ),
                "bertscore_f1": bert_f1,
            },
            "generation_diagnostics": {
                "metrics": generation_metrics,
                "preregistered_core_valid": core_valid,
                "preregistered_core_failures": list(
                    gate["preregistered_core_failures"]
                ),
            },
            "primary_loo_policy": {
                "included": True,
                "gate_filter_applied": False,
                "gate_zero_penalty_applied": False,
            },
        }
        scored.append(item)
        if not core_valid:
            failures.append(
                {
                    "schema_version": FAILURE_ROW_SCHEMA,
                    "source_generation_row": position + 1,
                    "tuple_key": expected_key,
                    "sample_id": source["sample_id"],
                    "meeting_id": source["meeting_id"],
                    "arm": source["arm"],
                    "intervention_topic": source["intervention_topic"],
                    "replicate_id": source["replicate_id"],
                    "completion_sha256": source["completion_sha256"],
                    "answer_sha256": source["answer_sha256"],
                    "diagnostic_failures": list(
                        gate["preregistered_core_failures"]
                    ),
                    "raw_semantic_scores": {
                        "mpnet_cosine": mpnet_cosine,
                        "bertscore_f1": bert_f1,
                    },
                    "included_in_primary_loo": True,
                    "effect_on_primary_score": "none_diagnostic_only",
                }
            )
    if sum(bool(row["empty_answer"]) for row in scored) != EXPECTED_EMPTY_ANSWER_ROWS:
        raise LooK5ScoreError("final empty-answer denominator drift")
    return scored, failures


def build_meeting_cells(
    score_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], np.ndarray]]:
    if len(score_rows) != EXPECTED_ROWS:
        raise LooK5ScoreError("score matrix must contain 10,880 rows")
    by_coordinate: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    meeting_ids: dict[int, str] = {}
    for row in score_rows:
        key = (
            int(row["meeting_rank"]),
            int(row["variant_rank"]),
            int(row["replicate_id"]),
        )
        if key in by_coordinate:
            raise LooK5ScoreError(f"duplicate score coordinate: {key}")
        by_coordinate[key] = row
        meeting_ids.setdefault(int(row["meeting_rank"]), str(row["meeting_id"]))
    expected_coordinates = {
        (meeting, variant, replicate)
        for meeting in range(EXPECTED_MEETINGS)
        for variant in range(EXPECTED_VARIANTS)
        for replicate in range(EXPECTED_REPLICATES)
    }
    if set(by_coordinate) != expected_coordinates:
        raise LooK5ScoreError("score coordinate coverage is not exact")

    matrices = {
        (topic, arm, metric): np.empty(
            (EXPECTED_MEETINGS, EXPECTED_REPLICATES), dtype=np.float64
        )
        for topic in TOPICS
        for arm in ARMS
        for metric in METRICS
    }
    meeting_cells: list[dict[str, Any]] = []
    for meeting_rank in range(EXPECTED_MEETINGS):
        for topic_index, topic in enumerate(TOPICS):
            for arm_index, arm in enumerate(ARMS):
                variant_rank = 1 + 2 * topic_index + arm_index
                full_scores = {metric: [] for metric in METRICS}
                intervention_scores = {metric: [] for metric in METRICS}
                deltas = {metric: [] for metric in METRICS}
                full_gate_valid: list[bool] = []
                intervention_gate_valid: list[bool] = []
                for replicate_id in range(EXPECTED_REPLICATES):
                    full = by_coordinate[(meeting_rank, 0, replicate_id)]
                    intervention = by_coordinate[
                        (meeting_rank, variant_rank, replicate_id)
                    ]
                    if (
                        full["replicate_seed"] != intervention["replicate_seed"]
                        or full["row_seed"] != intervention["row_seed"]
                        or full["reference_minutes_sha256"]
                        != intervention["reference_minutes_sha256"]
                    ):
                        raise LooK5ScoreError(
                            f"paired Full/intervention binding drift at "
                            f"meeting={meeting_rank}, variant={variant_rank}, "
                            f"replicate={replicate_id}"
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
                        matrices[(topic, arm, metric)][meeting_rank, replicate_id] = delta
                    full_gate_valid.append(
                        bool(
                            full["generation_diagnostics"][
                                "preregistered_core_valid"
                            ]
                        )
                    )
                    intervention_gate_valid.append(
                        bool(
                            intervention["generation_diagnostics"][
                                "preregistered_core_valid"
                            ]
                        )
                    )
                meeting_cells.append(
                    {
                        "schema_version": MEETING_CELL_SCHEMA,
                        "meeting_id": meeting_ids[meeting_rank],
                        "meeting_rank": meeting_rank,
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
                            "full_core_valid_replicates": sum(full_gate_valid),
                            "intervention_core_valid_replicates": sum(
                                intervention_gate_valid
                            ),
                            "used_to_filter_primary": False,
                        },
                    }
                )
    if len(meeting_cells) != EXPECTED_MEETINGS * len(TOPICS) * len(ARMS):
        raise LooK5ScoreError("meeting-cell denominator drift")
    return meeting_cells, matrices


def make_bootstrap_plan(
    *,
    meeting_ids: Sequence[str],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = EXPECTED_REPLICATES,
) -> BootstrapPlan:
    if (
        draws < 1
        or replicates < 1
        or not meeting_ids
        or len(set(meeting_ids)) != len(meeting_ids)
        or seed < 0
        or seed >= 2**64
    ):
        raise LooK5ScoreError("invalid hierarchical bootstrap dimensions")
    rng = np.random.default_rng(seed)
    meetings = rng.integers(
        0, len(meeting_ids), size=(draws, len(meeting_ids)), dtype=np.uint16
    )
    # One shared U(0,1) plan couples the cumulative stochastic K=1..5
    # diagnostics.  floor(U*K) is discrete-uniform on 0..K-1, while using the
    # same uniforms and meeting indices materially reduces simulation noise in
    # K4-versus-K5 CI endpoint comparisons.
    uniforms = rng.random(size=(draws, len(meeting_ids), replicates))
    prefix_replicate_indices = {
        prefix: np.floor(uniforms[:, :, :prefix] * prefix).astype(np.uint8)
        for prefix in range(1, replicates + 1)
    }
    replicate_indices = prefix_replicate_indices[replicates]
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
        "prefix_coupling": "shared_uniforms_floor_times_K_for_cumulative_K1_to_K5",
        "prefix_inventory": list(range(1, replicates + 1)),
        "prefix_replicate_index_shapes": {
            f"k{prefix}": list(prefix_replicate_indices[prefix].shape)
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
    digest.update(_canonical(metadata).encode("utf-8"))
    digest.update(meetings.astype("<u2", copy=False).tobytes(order="C"))
    for prefix in range(1, replicates + 1):
        digest.update(bytes([prefix]))
        digest.update(prefix_replicate_indices[prefix].tobytes(order="C"))
    return BootstrapPlan(
        meeting_indices=meetings,
        replicate_indices=replicate_indices,
        prefix_replicate_indices=prefix_replicate_indices,
        sha256=digest.hexdigest(),
        metadata=metadata,
    )


def _ci(distribution: np.ndarray, confidence: float = CONFIDENCE) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(distribution, [alpha, 1.0 - alpha], method="linear")
    return float(lower), float(upper)


def _same_nonzero_sign(left: float, right: float) -> bool:
    return (
        left != 0.0
        and right != 0.0
        and math.copysign(1.0, left) == math.copysign(1.0, right)
    )


def _interval_classification(lower: float, upper: float) -> str:
    if lower > 0:
        return "positive_excludes_zero"
    if upper < 0:
        return "negative_excludes_zero"
    return "crosses_zero"


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
            "k5_minus_k4_point_drift_threshold": K5_K4_POINT_DRIFT_THRESHOLD,
            "k5_minus_k4_ci_endpoint_drift_threshold": (
                K5_K4_CI_ENDPOINT_DRIFT_THRESHOLD
            ),
            "leave_one_replicate_max_drift_threshold": LORO_MAX_DRIFT_THRESHOLD,
            "decoding_variance_share_threshold": (
                DECODING_VARIANCE_SHARE_THRESHOLD
            ),
        }
    )
    required_thresholds = {
        "k5_minus_k4_point_drift_threshold",
        "k5_minus_k4_ci_endpoint_drift_threshold",
        "leave_one_replicate_max_drift_threshold",
        "decoding_variance_share_threshold",
    }
    if set(thresholds) != required_thresholds or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in thresholds.values()
    ):
        raise LooK5ScoreError("invalid frozen K-adequacy threshold inventory")
    point_threshold = float(thresholds["k5_minus_k4_point_drift_threshold"])
    endpoint_threshold = float(
        thresholds["k5_minus_k4_ci_endpoint_drift_threshold"]
    )
    loro_threshold = float(thresholds["leave_one_replicate_max_drift_threshold"])
    variance_threshold = float(thresholds["decoding_variance_share_threshold"])
    expected_keys = {
        (topic, arm, metric)
        for topic in TOPICS
        for arm in ARMS
        for metric in METRICS
    }
    if set(matrices) != expected_keys:
        raise LooK5ScoreError("bootstrap cell matrix inventory drift")
    if len(meeting_ids) != EXPECTED_MEETINGS:
        raise LooK5ScoreError("bootstrap meeting denominator drift")
    for key, matrix in matrices.items():
        if (
            not isinstance(matrix, np.ndarray)
            or matrix.shape != (EXPECTED_MEETINGS, EXPECTED_REPLICATES)
            or not np.all(np.isfinite(matrix))
        ):
            raise LooK5ScoreError(f"invalid finite K5 matrix for {key}")
    plan = make_bootstrap_plan(
        meeting_ids=meeting_ids, draws=draws, seed=seed, replicates=EXPECTED_REPLICATES
    )
    cell_draws: dict[tuple[str, str, str], np.ndarray] = {}
    prefix_cell_draws: dict[tuple[str, str, str, int], np.ndarray] = {}
    meeting_only_draws: dict[tuple[str, str, str], np.ndarray] = {}
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    adequacy_cells: list[dict[str, Any]] = []
    increase_reasons: list[dict[str, Any]] = []

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
                if not np.all(np.isfinite(nested)) or not np.all(
                    np.isfinite(meeting_only)
                ):
                    raise LooK5ScoreError(f"non-finite bootstrap distribution for {key}")
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
                if nested_variance > 0:
                    variance_share: float | None = decoding_variance / nested_variance
                else:
                    variance_share = 0.0 if decoding_variance == 0 else None
                by_replicate = matrix.mean(axis=0)
                leave_one = [
                    float(np.delete(matrix, replicate_id, axis=1).mean())
                    for replicate_id in range(EXPECTED_REPLICATES)
                ]
                prefix_diagnostics: dict[str, Any] = {}
                for prefix in range(1, EXPECTED_REPLICATES + 1):
                    prefix_matrix = matrix[:, :prefix]
                    prefix_distribution = prefix_matrix[
                        plan.meeting_indices[:, :, None],
                        plan.prefix_replicate_indices[prefix],
                    ].mean(axis=(1, 2))
                    if not np.all(np.isfinite(prefix_distribution)):
                        raise LooK5ScoreError(
                            f"non-finite K{prefix} prefix distribution for {key}"
                        )
                    prefix_cell_draws[(*key, prefix)] = prefix_distribution
                    prefix_low, prefix_high = _ci(prefix_distribution)
                    prefix_point = float(prefix_matrix.mean())
                    prefix_diagnostics[f"k{prefix}"] = {
                        "fresh_stochastic_replicates": prefix,
                        "replicate_seeds": list(REPLICATE_SEEDS[:prefix]),
                        "point_estimate_mean_delta": prefix_point,
                        "hierarchical_ci_lower": prefix_low,
                        "hierarchical_ci_upper": prefix_high,
                        "hierarchical_ci_width": prefix_high - prefix_low,
                        "interval_classification": _interval_classification(
                            prefix_low, prefix_high
                        ),
                        "historical_greedy_k1_included": False,
                    }
                k4 = prefix_diagnostics["k4"]
                k5 = prefix_diagnostics["k5"]
                point_drift = abs(
                    float(k5["point_estimate_mean_delta"])
                    - float(k4["point_estimate_mean_delta"])
                )
                lower_drift = abs(
                    float(k5["hierarchical_ci_lower"])
                    - float(k4["hierarchical_ci_lower"])
                )
                upper_drift = abs(
                    float(k5["hierarchical_ci_upper"])
                    - float(k4["hierarchical_ci_upper"])
                )
                endpoint_drift = max(lower_drift, upper_drift)
                decisive = nested_low > 0 or nested_high < 0
                sign_stable = all(
                    _same_nonzero_sign(point, estimate)
                    for estimate in leave_one
                )
                prefix_point_sign_stable = all(
                    _same_nonzero_sign(
                        point,
                        float(prefix_diagnostics[f"k{prefix}"]["point_estimate_mean_delta"]),
                    )
                    for prefix in (3, 4, 5)
                )
                decisive_classification = _interval_classification(
                    nested_low, nested_high
                )
                critical_prefix_ci_stable = (
                    not decisive
                    or all(
                        prefix_diagnostics[f"k{prefix}"]["interval_classification"]
                        == decisive_classification
                        for prefix in (3, 4, 5)
                    )
                )
                high_decode = variance_share is None or variance_share > variance_threshold
                reasons: list[str] = []
                if point_drift > point_threshold:
                    reasons.append("k5_minus_k4_point_drift_above_threshold")
                if endpoint_drift > endpoint_threshold:
                    reasons.append("k5_minus_k4_ci_endpoint_drift_above_threshold")
                loro_max_shift = max(abs(value - point) for value in leave_one)
                if loro_max_shift > loro_threshold:
                    reasons.append("leave_one_replicate_max_drift_above_threshold")
                if high_decode:
                    reasons.append("decoding_variance_share_above_threshold")
                if decisive and not critical_prefix_ci_stable:
                    reasons.append("critical_cell_k3_k4_k5_ci_classification_unstable")
                if decisive and (not sign_stable or not prefix_point_sign_stable):
                    reasons.append("critical_cell_prefix_or_loro_point_sign_unstable")
                if reasons:
                    increase_reasons.append(
                        {
                            "topic": topic,
                            "arm": arm,
                            "metric": metric,
                            "reasons": reasons,
                        }
                    )
                diagnostic = {
                    "topic": topic,
                    "arm": arm,
                    "metric": metric,
                    "conditional_decoding_mcse": math.sqrt(decoding_variance),
                    "nested_bootstrap_variance": nested_variance,
                    "estimated_decoding_variance_share": variance_share,
                    "thresholds": {
                        "k5_minus_k4_point_drift": point_threshold,
                        "k5_minus_k4_ci_endpoint_drift": endpoint_threshold,
                        "leave_one_replicate_max_drift": loro_threshold,
                        "decoding_variance_share": variance_threshold,
                    },
                    "cumulative_fresh_stochastic_prefixes": prefix_diagnostics,
                    "k5_minus_k4_point_drift": point_drift,
                    "k5_minus_k4_ci_lower_drift": lower_drift,
                    "k5_minus_k4_ci_upper_drift": upper_drift,
                    "k5_minus_k4_ci_endpoint_max_drift": endpoint_drift,
                    "k5_minus_k4_ci_width_drift": abs(
                        float(k5["hierarchical_ci_width"])
                        - float(k4["hierarchical_ci_width"])
                    ),
                    "global_replicate_mean_deltas": [
                        float(value) for value in by_replicate
                    ],
                    "leave_one_replicate_estimates": leave_one,
                    "leave_one_replicate_max_abs_shift": loro_max_shift,
                    "decisive_nested_ci": decisive,
                    "decisive_cell_leave_one_sign_stable": sign_stable,
                    "decisive_cell_k3_k4_k5_point_sign_stable": (
                        prefix_point_sign_stable
                    ),
                    "decisive_cell_k3_k4_k5_ci_classification_stable": (
                        critical_prefix_ci_stable
                    ),
                    "increase_k_reasons": reasons,
                }
                adequacy_cells.append(diagnostic)
                cells[topic][arm][metric] = {
                    "full_minus_intervention": True,
                    "point_estimate_mean_delta": point,
                    "replicate_delta_mean_by_seed": [
                        float(value) for value in by_replicate
                    ],
                    "meeting_mean_sd": float(np.std(meeting_means, ddof=1)),
                    "positive_meetings": int(np.count_nonzero(meeting_means > 0)),
                    "negative_meetings": int(np.count_nonzero(meeting_means < 0)),
                    "zero_meetings": int(np.count_nonzero(meeting_means == 0)),
                    "positive_meeting_share": float(np.mean(meeting_means > 0)),
                    "hierarchical_bootstrap": {
                        "draws": draws,
                        "failed_draws": 0,
                        "mean": float(nested.mean()),
                        "standard_error": float(np.std(nested, ddof=1))
                        if draws > 1
                        else 0.0,
                        "ci_lower": nested_low,
                        "ci_upper": nested_high,
                        "draw_share_above_zero": float(np.mean(nested > 0)),
                    },
                    "meeting_only_bootstrap_sensitivity": {
                        "draws": draws,
                        "failed_draws": 0,
                        "mean": float(meeting_only.mean()),
                        "standard_error": float(np.std(meeting_only, ddof=1))
                        if draws > 1
                        else 0.0,
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
                "meeting_aggregation": "arithmetic_mean_of_five_paired_replicates",
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
            },
            "bootstrap_contract": {
                **plan.metadata,
                "index_plan_sha256": plan.sha256,
                "confidence": CONFIDENCE,
                "primary_interval": "hierarchical_paired_percentile",
                "meeting_only_interval": "fixed_K5_meeting_resampling_sensitivity",
            },
            "cells": cells,
            "k_adequacy": {
                "generation_replicates_per_meeting_arm": EXPECTED_REPLICATES,
                "within_meeting_decoding_variance_estimable": True,
                "replicate_resampling_performed": True,
                "scope": "aggregate_128_meeting_estimand_not_per_meeting_inference",
                "thresholds_frozen_before_semantic_scoring": True,
                "stochastic_prefix_policy": (
                    "cumulative fresh vLLM replicates K1..K5; historical greedy K1 excluded"
                ),
                "k5_minus_k4_point_drift_threshold": point_threshold,
                "k5_minus_k4_ci_endpoint_drift_threshold": endpoint_threshold,
                "leave_one_replicate_max_drift_threshold": loro_threshold,
                "decoding_variance_share_threshold": variance_threshold,
                "increase_k_recommended": bool(increase_reasons),
                "increase_k_reason_cells": increase_reasons,
                "aggregate_status": (
                    "increase_k_recommended" if increase_reasons else "k5_adequate_for_aggregate_estimand"
                ),
                "cell_diagnostics": adequacy_cells,
                "per_meeting_inference": "not_authorized_at_K5",
            },
            "multiplicity": {
                "primary_cells": 32,
                "confidence_intervals_are_unadjusted": True,
                "familywise_significance_claim_authorized": False,
            },
            "limitations": [
                "This is post-selection within-CHK3 sensitivity, not causal attribution.",
                "The frozen target is a deterministic source-grounded Minutes-style reference, not an official Minutes excerpt.",
                "The 32 percentile intervals are unadjusted for multiple comparisons.",
                "K5 supports aggregate stochastic-decoding diagnostics but not per-meeting inference.",
                "The historical greedy K1 run is a different decoding and numerical distribution and is not pooled or formally contrasted.",
            ],
        }
    )
    return BootstrapComputation(
        plan=plan,
        cell_draws=cell_draws,
        prefix_cell_draws=prefix_cell_draws,
        meeting_only_draws=meeting_only_draws,
        results=results,
    )


def _write_bootstrap_draws(path: Path, computation: BootstrapComputation) -> None:
    keys = [
        (topic, arm, metric)
        for topic in TOPICS
        for arm in ARMS
        for metric in METRICS
    ]

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
                                computation.meeting_only_draws[
                                    (topic, arm, metric)
                                ][draw_id]
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

    del keys
    _publish_new_file(path, writer)


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
        raise LooK5ScoreError("score root must already contain the immutable plan")
    final_path = output_dir / "manifest.json"
    if final_path.exists() or final_path.is_symlink():
        raise LooK5ScoreError(f"refusing to overwrite final score manifest: {final_path}")
    plan, plan_binding = load_score_plan(plan_path, plan_sha256)
    if (
        draws != plan["bootstrap_contract"]["draws"]
        or seed != plan["bootstrap_contract"]["seed"]
    ):
        raise LooK5ScoreError(
            "finalize draws/seed differ from the frozen score-plan contract"
        )
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
    generation_rows = _load_generation_projection(plan, diagnostics=True)
    score_rows, failures = build_scored_rows(
        generation_rows=generation_rows,
        bert_rows=bert_rows,
        mpnet_rows=mpnet_rows,
    )
    meeting_cells, matrices = build_meeting_cells(score_rows)
    meeting_ids = [
        str(score_rows[meeting_rank * EXPECTED_VARIANTS * EXPECTED_REPLICATES]["meeting_id"])
        for meeting_rank in range(EXPECTED_MEETINGS)
    ]
    frozen_rule = plan["k_adequacy_rule"]
    frozen_thresholds = {
        field: frozen_rule[field]
        for field in (
            "k5_minus_k4_point_drift_threshold",
            "k5_minus_k4_ci_endpoint_drift_threshold",
            "leave_one_replicate_max_drift_threshold",
            "decoding_variance_share_threshold",
        )
    }
    computation = compute_bootstrap(
        matrices=matrices,
        meeting_ids=meeting_ids,
        draws=draws,
        seed=seed,
        adequacy_thresholds=frozen_thresholds,
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

    # Refuse publication if any immutable input changed during CPU finalization.
    if _file_binding(plan_path, sealed=True) != plan_binding:
        raise LooK5ScoreError("score plan changed during finalization")
    if _file_binding(bert_manifest_path, sealed=True) != bert_binding:
        raise LooK5ScoreError("BERTScore manifest changed during finalization")
    if _file_binding(mpnet_manifest_path, sealed=True) != mpnet_binding:
        raise LooK5ScoreError("MPNet manifest changed during finalization")
    _assert_binding(
        bert_manifest["artifacts"]["row_scores"], rows=EXPECTED_ROWS
    )
    _assert_binding(
        mpnet_manifest["artifacts"]["row_scores"], rows=EXPECTED_ROWS
    )
    _assert_binding(plan["inputs"]["canonical_generations"], rows=EXPECTED_ROWS)
    source_bundle = _assert_source_bundle(plan.get("sources"))

    manifest = seal_manifest(
        {
            "schema_version": MANIFEST_SCHEMA,
            "status": "complete",
            "created_at_utc": _utc_now(),
            "evaluation_id": SCORE_EVALUATION_ID,
            "operation": "raw_all_row_dual_gpu_semantic_scoring_and_hierarchical_paired_bootstrap",
            "inputs": {
                "score_plan": plan_binding,
                "generation_manifest": plan["inputs"]["generation_manifest"],
                "canonical_generations": plan["inputs"]["canonical_generations"],
                "cohort": plan["inputs"]["cohort"],
                "semantic_manifest": plan["inputs"]["semantic_manifest"],
                "historical_k1_anchor": plan["inputs"]["historical_k1_anchor"],
                "bertscore_metric_manifest": bert_binding,
                "mpnet_metric_manifest": mpnet_binding,
            },
            "coverage": {
                **plan["coverage"],
                "semantic_scored_rows": len(score_rows),
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
                "semantic_topology": {
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
            "sources": source_bundle,
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
    prepare.add_argument("--run-manifest", type=Path, default=DEFAULT_RUN_MANIFEST)
    prepare.add_argument("--run-manifest-sha256", required=True)
    prepare.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    prepare.add_argument("--cohort-sha256", default=DEFAULT_COHORT_SHA256)
    prepare.add_argument(
        "--semantic-manifest", type=Path, default=DEFAULT_SEMANTIC_MANIFEST
    )
    prepare.add_argument(
        "--semantic-manifest-sha256", default=DEFAULT_SEMANTIC_MANIFEST_SHA256
    )
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    metric = sub.add_parser("score-metric")
    metric.add_argument("--metric", choices=METRIC_WORKERS, required=True)
    metric.add_argument("--plan", type=Path, required=True)
    metric.add_argument("--plan-sha256", required=True)
    metric.add_argument("--output-dir", type=Path, required=True)
    metric.add_argument("--physical-gpu-index", choices=(0, 1), type=int, required=True)
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
                run_manifest_path=args.run_manifest,
                run_manifest_sha256=args.run_manifest_sha256,
                cohort_path=args.cohort,
                cohort_sha256=args.cohort_sha256,
                semantic_manifest_path=args.semantic_manifest,
                semantic_manifest_sha256=args.semantic_manifest_sha256,
                output_dir=args.output_dir,
            )
            summary = {
                "status": result["status"],
                "operation": "prepare",
                "rows": result["coverage"]["rows"],
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
                "rows": result["coverage"]["rows"],
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
                "rows": result["coverage"]["semantic_scored_rows"],
                "bootstrap_draws": result["coverage"]["bootstrap_draws"],
                "k_adequacy": result["k_adequacy"]["aggregate_status"],
                "payload_sha256": result["integrity"]["payload_sha256"],
            }
    except LooK5ScoreError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(_canonical(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
