"""Score and bootstrap the sealed CHK0/CHK1/CHK3 stochastic evaluation.

This command is deliberately generation-free.  It deep-validates
the three completed stochastic generation runs, independently recomputes the
four hard-gate metrics, scores eligible non-identity rows with the pinned local
MPNet and BERTScore backends (on CPU or an exclusively leased GPU0), and performs
the paired two-level bootstrap on CPU.  The same meeting and replicate indices
are used for every model and metric.

The formal contract is fixed at 190 prompts, five paired replicates, three
models, and 2,000 bootstrap draws.  No weighted composite is calculated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jobs.eval import score_chk0_chk1_chk3_six_metrics as gate_eval
from jobs.eval import score_chk3_native_checkpoint_sweep_semantic as semantic_eval
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


SCHEMA_VERSION = "chk3-stochastic-bootstrap-scorecard-v1"
ROW_SCHEMA_VERSION = "chk3-stochastic-bootstrap-six-metric-row-v1"
RESULT_SCHEMA_VERSION = "chk3-stochastic-bootstrap-results-v1"
DRAW_SCHEMA_VERSION = "chk3-stochastic-bootstrap-draw-v1"
FAILURE_SCHEMA_VERSION = "chk3-stochastic-bootstrap-failure-v1"

MODEL_ORDER = ("chk0", "chk1", "chk3")
GENERATION_ORDER = ("chk1", "chk3", "chk0")
REPLICATE_SEEDS = (20260811, 21260811, 22260811, 23260811, 24260811)
REPLICATE_IDS = tuple(range(len(REPLICATE_SEEDS)))
GENERATION_METRICS = (
    "structure_delivery",
    "numeric_fidelity",
    "date_fidelity",
    "degeneration_free",
)
SEMANTIC_METRICS = ("mpnet_cosine", "bertscore_f1")
SIX_METRICS = (*GENERATION_METRICS, *SEMANTIC_METRICS)

EXPECTED_PROMPTS = 190
EXPECTED_MEETINGS = 13
EXPECTED_ROWS_PER_MODEL = EXPECTED_PROMPTS * len(REPLICATE_IDS)
EXPECTED_TOTAL_ROWS = EXPECTED_ROWS_PER_MODEL * len(MODEL_ORDER)
BOOTSTRAP_DRAWS = 2_000
BOOTSTRAP_SEED = 20_260_812
CONFIDENCE = 0.95
GREEDY_ANCHOR_SCHEMA_VERSION = "chk0-chk1-chk3-six-metrics-v1"
GREEDY_ANCHOR_PAYLOAD_SHA256 = (
    "2828a7c6d94ec586b66c7edae12f9f31674cc528bcd346198c448d45b1322e72"
)
SELECTION_N12_FILE_SHA256 = (
    "371d29601e343acf98cac6173663d842ead9acaa4a454991d512b3f00bf5ee77"
)
SELECTION_N12_PAYLOAD_SHA256 = (
    "691edd595422cb0c384728d88c86d361628da7914c8e22c32a9f6967b1e88ee6"
)
SELECTION_N12_SCHEMA_VERSION = "chk3-native-test-samples-v1"
TASK_CONTRACT_ID = "chk3-native-analysis-to-minutes-v1"
SEMANTIC_MANIFEST_FILE_SHA256 = (
    "639ca25cf4f5e695b0c6cfeeb47aba75d3e6dfeda1ad87510c23bb4f4a378bf7"
)
SEMANTIC_MANIFEST_PAYLOAD_SHA256 = (
    "37a6896c388f965a572d3a3e99c2668a913513f2ac0883a0660634c807130445"
)

VIEW_ORDER = ("full", "row_disjoint", "strict_meeting_disjoint")
VIEW_EXPECTATIONS: dict[str, dict[str, int]] = {
    "full": {"generation_prompts": 190, "semantic_prompts": 171, "meetings": 13},
    "row_disjoint": {
        "generation_prompts": 178,
        "semantic_prompts": 160,
        "meetings": 13,
    },
    "strict_meeting_disjoint": {
        "generation_prompts": 59,
        "semantic_prompts": 52,
        "meetings": 4,
    },
}

CONTRASTS = (
    ("chk3_minus_chk1", "chk1", "chk3", "primary"),
    ("chk1_minus_chk0", "chk0", "chk1", "background"),
    ("chk3_minus_chk0", "chk0", "chk3", "background"),
)
_MEETING_RE = re.compile(r"(?:^|[^0-9])((?:19|20)\d{2}-\d{2}-\d{2})(?:[^0-9]|$)")


class StochasticBootstrapError(RuntimeError):
    """An input, scoring, or statistical contract failed closed."""


@dataclass(frozen=True)
class BootstrapPlan:
    """Shared resampling indices for one analysis view."""

    meeting_indices: np.ndarray
    replicate_indices: np.ndarray
    sha256: str
    metadata: dict[str, Any]


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
        raise StochasticBootstrapError(f"non-canonical JSON payload: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise StochasticBootstrapError(f"missing regular JSON file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StochasticBootstrapError(f"cannot read JSON {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise StochasticBootstrapError(f"JSON root is not an object: {resolved}")
    return value


def _file_binding(path: Path, *, sealed: bool = False) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise StochasticBootstrapError(f"artifact is not a regular file: {resolved}")
    result: dict[str, Any] = {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if sealed:
        try:
            result["payload_sha256"] = validate_manifest_integrity(
                _read_json(resolved)
            )
        except Exception as exc:
            raise StochasticBootstrapError(
                f"sealed artifact integrity failed for {resolved}: {exc}"
            ) from exc
    return result


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        raise StochasticBootstrapError(f"refusing to overwrite: {resolved}") from exc
    _fsync_directory(resolved.parent)


def _write_new_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        with resolved.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise StochasticBootstrapError(f"refusing to overwrite: {resolved}") from exc
    _fsync_directory(resolved.parent)


def _meeting_id(sample: Mapping[str, Any]) -> str:
    explicit = sample.get("meeting_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    sample_id = sample.get("sample_id")
    if not isinstance(sample_id, str):
        raise StochasticBootstrapError("sample has no valid sample_id/meeting_id")
    match = _MEETING_RE.search(sample_id)
    if match is None:
        raise StochasticBootstrapError(
            f"cannot derive meeting date from sample_id={sample_id!r}"
        )
    return match.group(1)


def _load_sample_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    binding = _file_binding(path, sealed=True)
    if binding["sha256"] != expected_sha256:
        raise StochasticBootstrapError(
            f"sample manifest file SHA-256 drift: {binding['sha256']}"
        )
    payload = _read_json(path)
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise StochasticBootstrapError("sample manifest has no samples")
    return payload


def _load_greedy_anchor(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deep-bind the deterministic N12 anchor without pooling it into inference."""

    binding = _file_binding(path, sealed=True)
    if binding["sha256"] != expected_sha256:
        raise StochasticBootstrapError(
            f"greedy anchor file SHA-256 drift: {binding['sha256']}"
        )
    if binding["payload_sha256"] != GREEDY_ANCHOR_PAYLOAD_SHA256:
        raise StochasticBootstrapError(
            "greedy anchor payload is not the frozen cp318 N12 anchor"
        )
    payload = _read_json(path)
    if (
        payload.get("schema_version") != GREEDY_ANCHOR_SCHEMA_VERSION
        or payload.get("status") != "complete"
        or payload.get("run_order") != list(MODEL_ORDER)
        or payload.get("scoring_contract", {}).get("metric_order")
        != list(gate_eval.SIX_METRICS)
        or len(payload.get("row_scores", [])) != 36
        or payload.get("selected_checkpoint", {}).get("checkpoint_step") != 318
    ):
        raise StochasticBootstrapError("greedy anchor scorecard contract drift")
    return payload, binding


def load_runs(
    *,
    chk0_manifest: Path,
    chk1_manifest: Path,
    chk3_manifest: Path,
    sample_manifest: Path,
    sample_manifest_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Use the stochastic runner's deep validator for all three sealed runs."""

    from jobs.eval import eval_chk3_stochastic_bootstrap_generation as stochastic

    paths = {
        "chk0": chk0_manifest,
        "chk1": chk1_manifest,
        "chk3": chk3_manifest,
    }
    if len({path.expanduser().resolve() for path in paths.values()}) != len(paths):
        raise StochasticBootstrapError("the three run manifests must be distinct")
    runs: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_ORDER:
        try:
            run = stochastic.load_and_validate_run(
                paths[model_id],
                expected_model_id=model_id,
                sample_manifest_path=sample_manifest,
                sample_manifest_sha256=sample_manifest_sha256,
            )
        except Exception as exc:
            raise StochasticBootstrapError(
                f"{model_id} stochastic run deep validation failed: {exc}"
            ) from exc
        runs[model_id] = dict(run)
    validate_run_matrix(runs)
    return runs


def load_generation_suite(
    *,
    generation_suite_manifest: Path,
    generation_suite_manifest_sha256: str,
    chk0_manifest: Path,
    chk1_manifest: Path,
    chk3_manifest: Path,
    sample_manifest: Path,
    sample_manifest_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Deep-validate the ordered suite and exact CLI child-manifest bindings."""

    from jobs.eval import eval_chk3_stochastic_bootstrap_generation as stochastic

    suite_binding = _file_binding(generation_suite_manifest, sealed=True)
    if suite_binding["sha256"] != generation_suite_manifest_sha256:
        raise StochasticBootstrapError("generation suite external SHA-256 drift")
    try:
        suite = stochastic.load_and_validate_suite(
            generation_suite_manifest,
            sample_manifest_path=sample_manifest,
            sample_manifest_sha256=sample_manifest_sha256,
            expected_scope="formal_full_test",
        )
    except Exception as exc:
        raise StochasticBootstrapError(
            f"ordered generation suite deep validation failed: {exc}"
        ) from exc
    if suite.get("manifest_binding") != suite_binding:
        raise StochasticBootstrapError("generation suite binding drift")
    raw_runs = suite.get("runs")
    if not isinstance(raw_runs, Mapping) or set(raw_runs) != set(MODEL_ORDER):
        raise StochasticBootstrapError("generation suite child run inventory drift")
    runs = {model_id: dict(raw_runs[model_id]) for model_id in MODEL_ORDER}
    validate_run_matrix(runs)
    cli_paths = {
        "chk0": chk0_manifest,
        "chk1": chk1_manifest,
        "chk3": chk3_manifest,
    }
    for model_id in MODEL_ORDER:
        cli_binding = _file_binding(cli_paths[model_id], sealed=True)
        if runs[model_id].get("manifest_binding") != cli_binding:
            raise StochasticBootstrapError(
                f"CLI {model_id} manifest is not the suite-bound child"
            )
    return runs, dict(suite), suite_binding


def _tuple_key(row: Mapping[str, Any]) -> tuple[str, int]:
    sample_id = row.get("sample_id")
    replicate_id = row.get("replicate_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise StochasticBootstrapError("generation row has no sample_id")
    if isinstance(replicate_id, bool) or not isinstance(replicate_id, int):
        raise StochasticBootstrapError("generation row has invalid replicate_id")
    return sample_id, replicate_id


def validate_run_matrix(
    runs: Mapping[str, Mapping[str, Any]], *, formal: bool = True
) -> list[tuple[str, int]]:
    """Reject incomplete, duplicated, or differently ordered paired coverage."""

    if tuple(runs) != MODEL_ORDER:
        raise StochasticBootstrapError(
            f"run inventory/order must be exactly {list(MODEL_ORDER)}"
        )
    canonical: list[tuple[str, int]] | None = None
    canonical_pairing: list[tuple[Any, ...]] | None = None
    reference_contract: Any = None
    reference_sample_binding: Any = None
    reference_source_hashes: Any = None
    for model_id in MODEL_ORDER:
        run = runs[model_id]
        rows = run.get("results")
        if not isinstance(rows, list) or not rows:
            raise StochasticBootstrapError(f"{model_id} has no generation rows")
        if formal and len(rows) != EXPECTED_ROWS_PER_MODEL:
            raise StochasticBootstrapError(
                f"{model_id} requires {EXPECTED_ROWS_PER_MODEL} rows, got {len(rows)}"
            )
        if formal and any(row.get("input_truncated") is not False for row in rows):
            raise StochasticBootstrapError(
                f"{model_id} does not prove zero input truncation for every row"
            )
        keys = [_tuple_key(row) for row in rows]
        if len(keys) != len(set(keys)):
            raise StochasticBootstrapError(f"{model_id} contains duplicate tuple keys")
        if any(row.get("model_id") != model_id for row in rows):
            raise StochasticBootstrapError(f"{model_id} row model_id drift")
        observed_replicates: dict[str, list[int]] = defaultdict(list)
        for sample_id, replicate_id in keys:
            observed_replicates[sample_id].append(replicate_id)
        for sample_id, values in observed_replicates.items():
            if tuple(values) != REPLICATE_IDS:
                raise StochasticBootstrapError(
                    f"{model_id}/{sample_id} replicate order/coverage drift: {values}"
                )
        pairing = [
            (
                row.get("sample_id"),
                row.get("meeting_id"),
                row.get("replicate_id"),
                row.get("replicate_seed"),
                row.get("seed"),
                row.get("row_seed"),
                bool(row.get("normalized_identity")),
                row.get("sample_manifest_sha256"),
                row.get("source_prompt_sha256"),
                row.get("source_analysis_sha256"),
                row.get("reference_minutes_sha256"),
                row.get("source_artifact_sha256s"),
            )
            for row in rows
        ]
        manifest = run.get("manifest")
        if not isinstance(manifest, Mapping):
            raise StochasticBootstrapError(f"{model_id} has no run manifest payload")
        sample_binding = manifest.get("sample_manifest")
        source_hashes = manifest.get("source_artifact_sha256s")
        if not isinstance(sample_binding, Mapping) or not isinstance(
            source_hashes, Mapping
        ):
            raise StochasticBootstrapError(
                f"{model_id} lost sample/source artifact bindings"
            )
        if canonical is None:
            canonical = keys
            canonical_pairing = pairing
            reference_contract = run.get("generation_contract")
            reference_sample_binding = sample_binding
            reference_source_hashes = source_hashes
        elif keys != canonical:
            raise StochasticBootstrapError(f"paired tuple order drift in {model_id}")
        elif pairing != canonical_pairing:
            raise StochasticBootstrapError(
                f"paired sample/seed/prompt/reference/source drift in {model_id}"
            )
        if run.get("generation_contract") != reference_contract:
            raise StochasticBootstrapError(f"generation contract drift in {model_id}")
        if sample_binding != reference_sample_binding:
            raise StochasticBootstrapError(
                f"sample/prompt/tokenizer binding drift in {model_id}"
            )
        if source_hashes != reference_source_hashes:
            raise StochasticBootstrapError(
                f"runner/gate/seed source artifact drift in {model_id}"
            )
    assert canonical is not None
    return canonical


def _generation_gate(row: Mapping[str, Any]) -> dict[str, Any]:
    try:
        gate = gate_eval._gate(row)
    except Exception as exc:
        raise StochasticBootstrapError(
            f"hard-gate recomputation failed for {_tuple_key(row)}: {exc}"
        ) from exc
    stored = row.get("generation_metrics")
    if not isinstance(stored, Mapping):
        raise StochasticBootstrapError("generation row has no generation_metrics")
    expected = {metric: bool(gate["metrics"][metric]) for metric in GENERATION_METRICS}
    observed = {metric: stored.get(metric) for metric in GENERATION_METRICS}
    if any(not isinstance(value, bool) for value in observed.values()):
        raise StochasticBootstrapError("stored generation metrics are malformed")
    if observed != expected:
        raise StochasticBootstrapError(
            f"stored generation metrics disagree with recomputation for {_tuple_key(row)}"
        )
    return gate


def build_scored_rows(
    *,
    runs: Mapping[str, Mapping[str, Any]],
    bert: Any,
    mpnet: Any,
    formal: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Recompute gates and score all eligible, non-identity semantic pairs."""

    validate_run_matrix(runs, formal=formal)
    gated: dict[tuple[str, str, int], dict[str, Any]] = {}
    eligible_work: list[tuple[str, Mapping[str, Any]]] = []
    eligible_keys: list[tuple[str, str, int]] = []
    for model_id in MODEL_ORDER:
        for row in runs[model_id]["results"]:
            sample_id, replicate_id = _tuple_key(row)
            key = (model_id, sample_id, replicate_id)
            gate = _generation_gate(row)
            gated[key] = gate
            if not bool(row.get("normalized_identity")) and gate[
                "preregistered_core_valid"
            ]:
                eligible_keys.append(key)
                eligible_work.append(("|".join(map(str, key)), row))

    try:
        semantic_values, semantic_audit = semantic_eval._score_semantic_pairs(
            work_items=eligible_work, bert=bert, mpnet=mpnet
        )
    except Exception as exc:
        raise StochasticBootstrapError(f"semantic scoring failed: {exc}") from exc
    if len(semantic_values) != len(eligible_keys):
        raise StochasticBootstrapError("semantic result length drift")
    semantic_by_key = {
        key: {
            "mpnet_cosine": float(value["mpnet_cosine"]),
            "bertscore_f1": float(value["bertscore_f1"]),
        }
        for key, value in zip(eligible_keys, semantic_values, strict=True)
    }

    row_scores: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        for source_row_number, row in enumerate(runs[model_id]["results"], start=1):
            sample_id, replicate_id = _tuple_key(row)
            key = (model_id, sample_id, replicate_id)
            gate = gated[key]
            normalized_identity = bool(row.get("normalized_identity"))
            core_valid = bool(gate["preregistered_core_valid"])
            if normalized_identity:
                semantic = {metric: None for metric in SEMANTIC_METRICS}
                semantic_status = "excluded_normalized_identity"
            elif not core_valid:
                semantic = {metric: 0.0 for metric in SEMANTIC_METRICS}
                semantic_status = "hard_gate_zero_penalty"
            else:
                semantic = semantic_by_key.get(key)
                if semantic is None:
                    raise StochasticBootstrapError(
                        f"eligible semantic result is missing for {key}"
                    )
                semantic_status = "scored"
            six_metrics: dict[str, float | None] = {
                metric: float(bool(gate["metrics"][metric]))
                for metric in GENERATION_METRICS
            }
            six_metrics.update(semantic)
            score_row = {
                "schema_version": ROW_SCHEMA_VERSION,
                "model_id": model_id,
                "model_label": row.get("model_label"),
                "tuple_key": {
                    "model_id": model_id,
                    "sample_id": sample_id,
                    "replicate_id": replicate_id,
                },
                "source_generation_row": source_row_number,
                "source_generation_manifest_sha256": runs[model_id]
                .get("manifest_binding", {})
                .get("sha256"),
                "source_generation_manifest_payload_sha256": runs[model_id]
                .get("manifest_binding", {})
                .get("payload_sha256"),
                "source_generations_sha256": runs[model_id]
                .get("manifest", {})
                .get("artifacts", {})
                .get("generations", {})
                .get("sha256"),
                "sample_id": sample_id,
                "meeting_id": _meeting_id(row),
                "replicate_id": replicate_id,
                "replicate_seed": row.get("replicate_seed"),
                "row_seed": row.get("seed", row.get("row_seed")),
                "normalized_identity": normalized_identity,
                "completion_sha256": row.get("completion_sha256"),
                "answer_sha256": row.get("answer_sha256"),
                "source_prompt_sha256": row.get("source_prompt_sha256"),
                "source_analysis_sha256": row.get("source_analysis_sha256"),
                "reference_minutes_sha256": row.get("reference_minutes_sha256"),
                "generated_token_ids_sha256": row.get(
                    "generated_token_ids_sha256"
                ),
                "finish_reason": row.get("finish_reason"),
                "input_truncated": row.get("input_truncated"),
                "generation_metrics": {
                    metric: bool(gate["metrics"][metric])
                    for metric in GENERATION_METRICS
                },
                "preregistered_core_valid": core_valid,
                "preregistered_core_failures": list(
                    gate["preregistered_core_failures"]
                ),
                "semantic_status": semantic_status,
                "semantic_denominator_eligible": not normalized_identity,
                "semantic_encoder_eligible": not normalized_identity and core_valid,
                "semantic_zero_penalty_applied": (
                    not normalized_identity and not core_valid
                ),
                "six_metrics": six_metrics,
            }
            row_scores.append(score_row)
            if not core_valid:
                failures.append(
                    {
                        "schema_version": FAILURE_SCHEMA_VERSION,
                        "model_id": model_id,
                        "tuple_key": score_row["tuple_key"],
                        "sample_id": sample_id,
                        "meeting_id": score_row["meeting_id"],
                        "replicate_id": replicate_id,
                        "replicate_seed": score_row["replicate_seed"],
                        "row_seed": score_row["row_seed"],
                        "completion_sha256": score_row["completion_sha256"],
                        "answer_sha256": score_row["answer_sha256"],
                        "failed_generation_metrics": [
                            metric
                            for metric in GENERATION_METRICS
                            if not score_row["generation_metrics"][metric]
                        ],
                        "preregistered_core_failures": score_row[
                            "preregistered_core_failures"
                        ],
                        "semantic_policy": (
                            "identity_excluded"
                            if normalized_identity
                            else "both_semantic_metrics_fixed_to_zero"
                        ),
                    }
                )
    expected = sum(len(runs[model_id]["results"]) for model_id in MODEL_ORDER)
    if len(row_scores) != expected:
        raise StochasticBootstrapError("scored row count drift")
    semantic_audit = dict(semantic_audit)
    semantic_audit["eligible_pair_order"] = [
        {
            "model_id": key[0],
            "sample_id": key[1],
            "replicate_id": key[2],
        }
        for key in eligible_keys
    ]
    semantic_audit["normalized_identity_policy"] = "excluded_from_semantic_denominator"
    semantic_audit["hard_gate_failure_policy"] = "both_semantic_metrics_fixed_to_zero"
    return row_scores, failures, semantic_audit


def build_views(
    *,
    full_sample_manifest: Mapping[str, Any],
    selection_sample_manifest: Mapping[str, Any],
    formal: bool = True,
) -> dict[str, dict[str, Any]]:
    """Construct Full, exact-row-disjoint, and meeting-disjoint cohorts."""

    full = full_sample_manifest.get("samples")
    selected = selection_sample_manifest.get("samples")
    if not isinstance(full, list) or not isinstance(selected, list):
        raise StochasticBootstrapError("sample manifests must contain sample lists")
    full_by_id: dict[str, dict[str, Any]] = {}
    for sample in full:
        if not isinstance(sample, Mapping):
            raise StochasticBootstrapError("full sample row is not an object")
        sample_id = sample.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in full_by_id:
            raise StochasticBootstrapError("full sample IDs are invalid or duplicated")
        full_by_id[sample_id] = {
            "sample_id": sample_id,
            "meeting_id": _meeting_id(sample),
            "normalized_identity": bool(sample.get("normalized_identity")),
        }
    selected_ids: set[str] = set()
    selected_meetings: set[str] = set()
    for sample in selected:
        if not isinstance(sample, Mapping) or not isinstance(sample.get("sample_id"), str):
            raise StochasticBootstrapError("selection sample row is malformed")
        sample_id = str(sample["sample_id"])
        if sample_id in selected_ids or sample_id not in full_by_id:
            raise StochasticBootstrapError(
                f"selection sample is duplicated or absent from full test: {sample_id}"
            )
        selected_ids.add(sample_id)
        selected_meetings.add(full_by_id[sample_id]["meeting_id"])

    view_ids = {
        "full": set(full_by_id),
        "row_disjoint": set(full_by_id) - selected_ids,
        "strict_meeting_disjoint": {
            sample_id
            for sample_id, sample in full_by_id.items()
            if sample["meeting_id"] not in selected_meetings
        },
    }
    result: dict[str, dict[str, Any]] = {}
    for view_id in VIEW_ORDER:
        ids = view_ids[view_id]
        records = [sample for sample in full_by_id.values() if sample["sample_id"] in ids]
        semantic_records = [
            sample for sample in records if not sample["normalized_identity"]
        ]
        record = {
            "view_id": view_id,
            "role": (
                "descriptive_full"
                if view_id == "full"
                else (
                    "primary_post_selection_row_disjoint_robustness"
                    if view_id == "row_disjoint"
                    else "meeting_disjoint_sensitivity_only"
                )
            ),
            "sample_ids": [sample["sample_id"] for sample in records],
            "generation_prompts": len(records),
            "semantic_prompts": len(semantic_records),
            "meeting_ids": sorted({sample["meeting_id"] for sample in records}),
            "meetings": len({sample["meeting_id"] for sample in records}),
            "selection_anchor_rows": len(selected_ids),
            "excluded_prompts": len(full_by_id) - len(records),
            "selection_meetings_excluded": (
                len(selected_meetings) if view_id == "strict_meeting_disjoint" else 0
            ),
            "inferential_conclusion_authorized": view_id == "row_disjoint",
        }
        if formal:
            expected = VIEW_EXPECTATIONS[view_id]
            observed = {
                key: record[key]
                for key in ("generation_prompts", "semantic_prompts", "meetings")
            }
            if observed != expected:
                raise StochasticBootstrapError(
                    f"{view_id} denominator drift: expected={expected}, observed={observed}"
                )
        result[view_id] = record
    return result


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return monotone Holm family-wise adjusted p-values."""

    indexed: list[tuple[int, float]] = []
    for index, value in enumerate(p_values):
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise StochasticBootstrapError("Holm p-values must be finite in [0,1]")
        indexed.append((index, numeric))
    indexed.sort(key=lambda item: item[1])
    result = [0.0] * len(indexed)
    running = 0.0
    total = len(indexed)
    for rank, (original, value) in enumerate(indexed):
        running = max(running, min(1.0, (total - rank) * value))
        result[original] = running
    return result


def exact_sign_flip_p_value(differences: Sequence[float]) -> tuple[float, int]:
    """Two-sided exact sign-flip test, capped by the 13-meeting contract."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or len(values) > EXPECTED_MEETINGS:
        raise StochasticBootstrapError("exact sign-flip requires 1..13 meetings")
    if not np.all(np.isfinite(values)):
        raise StochasticBootstrapError("sign-flip differences are not finite")
    combinations = 1 << len(values)
    observed = abs(float(values.mean()))
    indexes = np.arange(combinations, dtype=np.uint64)[:, None]
    bits = (indexes >> np.arange(len(values), dtype=np.uint64)) & 1
    signs = np.where(bits == 1, 1.0, -1.0)
    null_values = np.abs((signs * values).mean(axis=1))
    p_value = float(np.mean(null_values >= observed - 1e-15))
    return p_value, combinations


def paired_sign_flip_p_value(
    differences: Sequence[float],
    *,
    monte_carlo_draws: int = 100_000,
    seed: int = BOOTSTRAP_SEED + 1,
) -> tuple[float, int, str]:
    """Use exact sign flips when feasible, otherwise deterministic Monte Carlo.

    The original 13-meeting contract remains bit-for-bit exact.  Larger external
    meeting panels cannot enumerate ``2**M`` assignments, so they use a seeded
    plus-one-corrected Monte Carlo randomization test.
    """

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.all(np.isfinite(values)):
        raise StochasticBootstrapError("sign-flip differences must be finite 1-D")
    if len(values) <= 13:
        p_value, combinations = exact_sign_flip_p_value(values)
        return p_value, combinations, "two_sided_exact_sign_flip"
    if monte_carlo_draws < 1:
        raise StochasticBootstrapError("Monte Carlo sign-flip draws must be positive")
    rng = np.random.default_rng(seed)
    observed = abs(float(values.mean()))
    extreme = 0
    remaining = monte_carlo_draws
    chunk = 10_000
    while remaining:
        size = min(chunk, remaining)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, len(values)))
        null_values = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(null_values >= observed - 1e-15))
        remaining -= size
    p_value = (extreme + 1.0) / (monte_carlo_draws + 1.0)
    return float(p_value), monte_carlo_draws, "two_sided_monte_carlo_sign_flip_plus_one"


def make_bootstrap_plan(
    *,
    view_id: str,
    meeting_ids: Sequence[str],
    sample_ids: Sequence[str],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = len(REPLICATE_IDS),
) -> BootstrapPlan:
    """Create one model/metric-shared hierarchical resampling plan."""

    if draws < 1 or replicates < 1 or not meeting_ids or not sample_ids:
        raise StochasticBootstrapError("bootstrap dimensions must be positive")
    if len(set(meeting_ids)) != len(meeting_ids):
        raise StochasticBootstrapError("bootstrap meeting inventory is duplicated")
    if len(set(sample_ids)) != len(sample_ids):
        raise StochasticBootstrapError("bootstrap sample inventory is duplicated")
    rng = np.random.default_rng(seed)
    meeting_indices = rng.integers(
        0,
        len(meeting_ids),
        size=(draws, len(meeting_ids)),
        dtype=np.uint16,
    )
    replicate_indices = rng.integers(
        0,
        replicates,
        size=(draws, len(meeting_ids), len(sample_ids), replicates),
        dtype=np.uint8,
    )
    metadata = {
        "schema_version": "chk3-hierarchical-paired-bootstrap-index-plan-v1",
        "view_id": view_id,
        "draws": draws,
        "seed": seed,
        "meeting_ids": list(meeting_ids),
        "sample_ids": list(sample_ids),
        "replicates": replicates,
        "meeting_index_shape": list(meeting_indices.shape),
        "replicate_index_shape": list(replicate_indices.shape),
        "meeting_index_dtype": "uint16-le",
        "replicate_index_dtype": "uint8",
        "numpy_version": np.__version__,
        "numpy_bit_generator": type(rng.bit_generator).__name__,
        "shared_across_models": list(MODEL_ORDER),
        "shared_across_metrics": list(SIX_METRICS),
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json(metadata).encode("utf-8"))
    digest.update(meeting_indices.astype("<u2", copy=False).tobytes(order="C"))
    digest.update(replicate_indices.tobytes(order="C"))
    return BootstrapPlan(
        meeting_indices=meeting_indices,
        replicate_indices=replicate_indices,
        sha256=digest.hexdigest(),
        metadata=metadata,
    )


def _prepare_values(
    *,
    row_scores: Sequence[Mapping[str, Any]],
    sample_records: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, list[int]],
    dict[str, bool],
]:
    sample_ids = [str(sample["sample_id"]) for sample in sample_records]
    sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    identity = {
        str(sample["sample_id"]): bool(sample["normalized_identity"])
        for sample in sample_records
    }
    meeting_rows: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(sample_records):
        meeting_rows[str(sample["meeting_id"])].append(index)
    values = {
        model_id: {
            metric: np.full(
                (len(sample_records), len(REPLICATE_IDS)),
                np.nan,
                dtype=np.float64,
            )
            for metric in SIX_METRICS
        }
        for model_id in MODEL_ORDER
    }
    seen: set[tuple[str, str, int]] = set()
    for row in row_scores:
        model_id = str(row.get("model_id"))
        sample_id = str(row.get("sample_id"))
        replicate_id = row.get("replicate_id")
        if model_id not in values or sample_id not in sample_index:
            continue
        if isinstance(replicate_id, bool) or replicate_id not in REPLICATE_IDS:
            raise StochasticBootstrapError("scored row replicate_id drift")
        key = (model_id, sample_id, int(replicate_id))
        if key in seen:
            raise StochasticBootstrapError(f"duplicate scored row: {key}")
        seen.add(key)
        metrics = row.get("six_metrics")
        if not isinstance(metrics, Mapping):
            raise StochasticBootstrapError("scored row has no six_metrics")
        for metric in SIX_METRICS:
            value = metrics.get(metric)
            if metric in SEMANTIC_METRICS and identity[sample_id]:
                if value is not None:
                    raise StochasticBootstrapError(
                        "identity semantic metric must be excluded with null"
                    )
                continue
            if isinstance(value, bool):
                raise StochasticBootstrapError("scored metric must be numeric")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise StochasticBootstrapError("scored metric is not numeric") from exc
            if not math.isfinite(numeric):
                raise StochasticBootstrapError("scored metric is non-finite")
            if metric in GENERATION_METRICS and numeric not in (0.0, 1.0):
                raise StochasticBootstrapError("hard-gate metric is not binary")
            values[model_id][metric][sample_index[sample_id], int(replicate_id)] = numeric
    expected_keys = {
        (model_id, sample_id, replicate_id)
        for model_id in MODEL_ORDER
        for sample_id in sample_ids
        for replicate_id in REPLICATE_IDS
    }
    if seen != expected_keys:
        missing = sorted(expected_keys - seen)[:5]
        extra = sorted(seen - expected_keys)[:5]
        raise StochasticBootstrapError(
            f"view scored-row coverage drift; missing={missing}, extra={extra}"
        )
    for model_id in MODEL_ORDER:
        for metric in SIX_METRICS:
            matrix = values[model_id][metric]
            eligible_rows = np.asarray(
                [metric not in SEMANTIC_METRICS or not identity[sid] for sid in sample_ids]
            )
            if not np.all(np.isfinite(matrix[eligible_rows])):
                raise StochasticBootstrapError(
                    f"incomplete values for {model_id}/{metric}"
                )
    return values, dict(meeting_rows), identity


def _point_and_meeting_estimates(
    *,
    values: Mapping[str, Mapping[str, np.ndarray]],
    meeting_rows: Mapping[str, Sequence[int]],
    identity: Mapping[str, bool],
    sample_ids: Sequence[str],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    points = {model_id: {} for model_id in MODEL_ORDER}
    meeting_estimates = {model_id: {} for model_id in MODEL_ORDER}
    identity_mask = np.asarray([identity[sample_id] for sample_id in sample_ids])
    for model_id in MODEL_ORDER:
        for metric in SIX_METRICS:
            per_row = values[model_id][metric].mean(axis=1)
            by_meeting: dict[str, float] = {}
            for meeting_id, raw_indices in meeting_rows.items():
                indices = np.asarray(raw_indices, dtype=np.int64)
                if metric in SEMANTIC_METRICS:
                    indices = indices[~identity_mask[indices]]
                if len(indices) == 0:
                    raise StochasticBootstrapError(
                        f"meeting {meeting_id} has no eligible rows for {metric}"
                    )
                by_meeting[meeting_id] = float(per_row[indices].mean())
            meeting_estimates[model_id][metric] = by_meeting
            points[model_id][metric] = float(np.mean(list(by_meeting.values())))
    return points, meeting_estimates


def bootstrap_view(
    *,
    view: Mapping[str, Any],
    full_sample_records: Sequence[Mapping[str, Any]],
    row_scores: Sequence[Mapping[str, Any]],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute one view with paired meeting+replicate resampling."""

    wanted = set(view["sample_ids"])
    sample_records = [
        sample for sample in full_sample_records if str(sample["sample_id"]) in wanted
    ]
    sample_ids = [str(sample["sample_id"]) for sample in sample_records]
    if len(sample_ids) != len(wanted):
        raise StochasticBootstrapError("view sample universe is incomplete")
    values, meeting_rows, identity = _prepare_values(
        row_scores=row_scores, sample_records=sample_records
    )
    meeting_ids = sorted(meeting_rows)
    points, meeting_estimates = _point_and_meeting_estimates(
        values=values,
        meeting_rows=meeting_rows,
        identity=identity,
        sample_ids=sample_ids,
    )
    plan = make_bootstrap_plan(
        view_id=str(view["view_id"]),
        meeting_ids=meeting_ids,
        sample_ids=sample_ids,
        draws=draws,
        seed=seed,
        # The external evaluation profile expands the formal contract from K=5
        # to K=10 at runtime.  Do not rely on make_bootstrap_plan's definition-
        # time default, which was captured while REPLICATE_IDS still described
        # the original K=5 profile.
        replicates=len(REPLICATE_IDS),
    )
    boot = {
        model_id: {
            metric: np.empty(draws, dtype=np.float64) for metric in SIX_METRICS
        }
        for model_id in MODEL_ORDER
    }
    identity_mask = np.asarray([identity[sample_id] for sample_id in sample_ids])
    for draw_id in range(draws):
        for model_id in MODEL_ORDER:
            for metric in SIX_METRICS:
                occurrence_means: list[float] = []
                matrix = values[model_id][metric]
                for occurrence, meeting_index in enumerate(
                    plan.meeting_indices[draw_id]
                ):
                    meeting_id = meeting_ids[int(meeting_index)]
                    indices = np.asarray(meeting_rows[meeting_id], dtype=np.int64)
                    if metric in SEMANTIC_METRICS:
                        indices = indices[~identity_mask[indices]]
                    replicate_draws = plan.replicate_indices[
                        draw_id, occurrence, indices, :
                    ]
                    selected = np.take_along_axis(
                        matrix[indices], replicate_draws.astype(np.int64), axis=1
                    )
                    occurrence_means.append(float(selected.mean(axis=1).mean()))
                boot[model_id][metric][draw_id] = float(
                    np.mean(occurrence_means)
                )

    alpha = 1.0 - CONFIDENCE
    estimates: dict[str, Any] = {}
    for model_id in MODEL_ORDER:
        estimates[model_id] = {}
        for metric in SIX_METRICS:
            distribution = boot[model_id][metric]
            finite = np.isfinite(distribution)
            valid = distribution[finite]
            diagnostic_rows = [
                row
                for row in row_scores
                if row.get("model_id") == model_id
                and row.get("sample_id") in wanted
                and (
                    metric in GENERATION_METRICS
                    or not bool(row.get("normalized_identity"))
                )
            ]
            if metric in GENERATION_METRICS:
                failed_rows = [
                    row
                    for row in diagnostic_rows
                    if not bool(row["generation_metrics"][metric])
                ]
                semantic_diagnostics = {
                    "hard_gate_zero_penalty_generation_count": None,
                    "semantic_scored_generation_count": None,
                    "normalized_identity_excluded_generation_count": None,
                }
            else:
                failed_rows = [
                    row
                    for row in diagnostic_rows
                    if row.get("semantic_status") == "hard_gate_zero_penalty"
                ]
                semantic_diagnostics = {
                    "hard_gate_zero_penalty_generation_count": len(failed_rows),
                    "semantic_scored_generation_count": sum(
                        row.get("semantic_status") == "scored"
                        for row in diagnostic_rows
                    ),
                    "normalized_identity_excluded_generation_count": (
                        int(view["generation_prompts"] - view["semantic_prompts"])
                        * len(REPLICATE_IDS)
                    ),
                }
            estimates[model_id][metric] = {
                "point_estimate": points[model_id][metric],
                "ci_lower": float(np.quantile(valid, alpha / 2.0)),
                "ci_upper": float(np.quantile(valid, 1.0 - alpha / 2.0)),
                "confidence": CONFIDENCE,
                "bootstrap_draws": draws,
                "failed_draws": int(draws - len(valid)),
                "aggregation": "replicate_mean_then_meeting_prompt_mean_then_meeting_equal_mean",
                "generation_prompts": int(view["generation_prompts"]),
                "semantic_prompts": (
                    int(view["semantic_prompts"])
                    if metric in SEMANTIC_METRICS
                    else None
                ),
                "meetings": len(meeting_ids),
                "observations": len(diagnostic_rows),
                "zero_or_fail_generation_count": len(failed_rows),
                "prompts_with_any_failure": len(
                    {str(row["sample_id"]) for row in failed_rows}
                ),
                "meetings_with_any_failure": len(
                    {str(row["meeting_id"]) for row in failed_rows}
                ),
                **semantic_diagnostics,
            }

    contrast_records: list[dict[str, Any]] = []
    for contrast_id, before, after, family in CONTRASTS:
        for metric in SIX_METRICS:
            distribution = boot[after][metric] - boot[before][metric]
            finite = np.isfinite(distribution)
            valid = distribution[finite]
            meeting_differences = [
                meeting_estimates[after][metric][meeting_id]
                - meeting_estimates[before][metric][meeting_id]
                for meeting_id in meeting_ids
            ]
            p_value, combinations, p_value_method = paired_sign_flip_p_value(
                meeting_differences
            )
            contrast_records.append(
                {
                    "contrast_id": contrast_id,
                    "before_model": before,
                    "after_model": after,
                    "family": family,
                    "metric": metric,
                    "point_difference": points[after][metric]
                    - points[before][metric],
                    "ci_lower": float(np.quantile(valid, alpha / 2.0)),
                    "ci_upper": float(np.quantile(valid, 1.0 - alpha / 2.0)),
                    "confidence": CONFIDENCE,
                    "failed_draws": int(draws - len(valid)),
                    "meeting_ids": meeting_ids,
                    "meeting_differences": meeting_differences,
                    "p_value": p_value,
                    "p_value_method": p_value_method,
                    "sign_flip_combinations": combinations,
                    "sign_flip_assignments_evaluated": combinations,
                    "sign_flip_exact": p_value_method == "two_sided_exact_sign_flip",
                    "holm_adjusted_p": None,
                }
            )
    for family in ("primary", "background"):
        selected = [record for record in contrast_records if record["family"] == family]
        adjusted = holm_adjust([float(record["p_value"]) for record in selected])
        for record, adjusted_p in zip(selected, adjusted, strict=True):
            record["holm_adjusted_p"] = adjusted_p
            record["holm_family_size"] = len(selected)

    primary = {
        record["metric"]: record
        for record in contrast_records
        if record["contrast_id"] == "chk3_minus_chk1"
    }
    regression_tolerance = 1e-12
    regressed = [
        metric
        for metric in GENERATION_METRICS
        if float(primary[metric]["point_difference"]) < -regression_tolerance
    ]
    semantic_supported = all(
        float(primary[metric]["ci_lower"]) > 0.0
        and float(primary[metric]["holm_adjusted_p"]) < 0.05
        for metric in SEMANTIC_METRICS
    )
    if regressed:
        verdict = "random_decoding_regression"
    elif semantic_supported:
        verdict = "semantic_increment_under_random_decoding"
    else:
        verdict = "robust_but_increment_uncertain"
    interpretation = {
        "verdict": verdict,
        "hard_gate_regressions": regressed,
        "hard_gate_regression_tolerance": regression_tolerance,
        "semantic_increment_supported": semantic_supported,
        "conclusion_eligible": bool(view["inferential_conclusion_authorized"]),
        "strict_view_override": (
            "sensitivity_only_no_significance_conclusion"
            if not bool(view["inferential_conclusion_authorized"])
            else None
        ),
    }

    draw_rows = []
    for draw_id in range(draws):
        draw_rows.append(
            {
                "schema_version": DRAW_SCHEMA_VERSION,
                "view_id": view["view_id"],
                "draw_id": draw_id,
                "index_plan_sha256": plan.sha256,
                "model_metrics": {
                    model_id: {
                        metric: float(boot[model_id][metric][draw_id])
                        for metric in SIX_METRICS
                    }
                    for model_id in MODEL_ORDER
                },
            }
        )
    result = {
        "view": dict(view),
        "bootstrap_index_plan": {
            **plan.metadata,
            "sha256": plan.sha256,
            "meeting_resampling": "sample M meetings with replacement",
            "row_policy": "retain all rows in each sampled meeting occurrence",
            "replicate_resampling": "sample K replicates with replacement per row occurrence",
        },
        "estimates": estimates,
        "contrasts": contrast_records,
        "interpretation": interpretation,
    }
    return result, draw_rows


def build_bootstrap_results(
    *,
    row_scores: Sequence[Mapping[str, Any]],
    full_sample_manifest: Mapping[str, Any],
    selection_sample_manifest: Mapping[str, Any],
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    formal: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    views = build_views(
        full_sample_manifest=full_sample_manifest,
        selection_sample_manifest=selection_sample_manifest,
        formal=formal,
    )
    full_records = [
        {
            "sample_id": str(sample["sample_id"]),
            "meeting_id": _meeting_id(sample),
            "normalized_identity": bool(sample.get("normalized_identity")),
        }
        for sample in full_sample_manifest["samples"]
    ]
    view_results: dict[str, Any] = {}
    draw_rows: list[dict[str, Any]] = []
    for view_id in VIEW_ORDER:
        result, view_draws = bootstrap_view(
            view=views[view_id],
            full_sample_records=full_records,
            row_scores=row_scores,
            draws=draws,
            seed=seed,
        )
        view_results[view_id] = result
        draw_rows.extend(view_draws)
    result = seal_manifest(
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "complete",
            "model_order": list(MODEL_ORDER),
            "metric_order": list(SIX_METRICS),
            "view_order": list(VIEW_ORDER),
            "bootstrap_contract": {
                "draws": draws,
                "seed": seed,
                "confidence": CONFIDENCE,
                "interval": "two_sided_percentile",
                "quantile_method": "numpy_linear",
                "meeting_equal_weighting": True,
                "shared_indices_across_models_and_metrics": True,
                "paired": True,
                "weighted_composite_calculated": False,
            },
            "multiple_testing": {
                "primary_family": "CHK3-minus-CHK1 across six metrics; Holm size 6",
                "background_family": (
                    "CHK1-minus-CHK0 and CHK3-minus-CHK0 across six metrics; Holm size 12"
                ),
            },
            "views": view_results,
            "limitations": [
                "This is post-selection robustness: cp318 used the earlier N12 for selection.",
                "Bootstrap resampling cannot remove checkpoint-selection bias.",
                "The strict four-meeting view is sensitivity-only and is not used for a significance conclusion.",
                "BERTScore and MPNet similarity cannot offset a hard-gate regression.",
                "An independent confirmation set with new meetings remains necessary.",
            ],
        }
    )
    return result, draw_rows


def score_and_bootstrap(
    *,
    generation_suite_manifest: Path,
    generation_suite_manifest_sha256: str,
    chk0_manifest: Path,
    chk1_manifest: Path,
    chk3_manifest: Path,
    sample_manifest: Path,
    sample_manifest_sha256: str,
    selection_sample_manifest: Path,
    selection_sample_manifest_sha256: str,
    greedy_anchor_scorecard: Path,
    greedy_anchor_scorecard_sha256: str,
    semantic_manifest: Path,
    output_dir: Path,
    semantic_device: str,
    semantic_batch_size: int,
    gpu_wait_timeout_seconds: int = 172_800,
    gpu_poll_seconds: int = 30,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run formal offline scoring plus CPU bootstrap and seal every output."""

    if draws != BOOTSTRAP_DRAWS or seed != BOOTSTRAP_SEED:
        raise StochasticBootstrapError(
            f"formal bootstrap is fixed at draws={BOOTSTRAP_DRAWS}, seed={BOOTSTRAP_SEED}"
        )
    if semantic_batch_size < 1:
        raise StochasticBootstrapError("semantic batch size must be positive")
    if semantic_device not in {"cpu", "cuda:0"}:
        raise StochasticBootstrapError("semantic device must be cpu or cuda:0")
    visible = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if semantic_device == "cuda:0" and visible != ["0"]:
        raise StochasticBootstrapError(
            "CUDA semantic scoring requires CUDA_VISIBLE_DEVICES=0 and --semantic-device cuda:0"
        )
    if semantic_device == "cuda:0" and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise StochasticBootstrapError(
            "CUDA semantic scoring requires CUDA_DEVICE_ORDER=PCI_BUS_ID"
        )
    resolved_output = output_dir.expanduser().resolve()
    if resolved_output.exists() or resolved_output.is_symlink():
        raise StochasticBootstrapError(
            f"refusing to reuse or overwrite output directory: {resolved_output}"
        )

    full_manifest = _load_sample_manifest(sample_manifest, sample_manifest_sha256)
    if selection_sample_manifest_sha256 != SELECTION_N12_FILE_SHA256:
        raise StochasticBootstrapError(
            "formal row-disjoint view requires the frozen cp318-selection N12 file SHA"
        )
    selection_manifest = _load_sample_manifest(
        selection_sample_manifest, selection_sample_manifest_sha256
    )
    if (
        selection_manifest.get("schema_version") != SELECTION_N12_SCHEMA_VERSION
        or selection_manifest.get("task_contract_id") != TASK_CONTRACT_ID
        or selection_manifest.get("integrity", {}).get("payload_sha256")
        != SELECTION_N12_PAYLOAD_SHA256
        or len(selection_manifest.get("samples", [])) != 12
    ):
        raise StochasticBootstrapError(
            "formal row-disjoint selection anchor schema/task/payload drift"
        )
    greedy_anchor, greedy_anchor_binding = _load_greedy_anchor(
        greedy_anchor_scorecard, greedy_anchor_scorecard_sha256
    )
    semantic_manifest_binding = _file_binding(semantic_manifest, sealed=True)
    if (
        semantic_manifest_binding["sha256"] != SEMANTIC_MANIFEST_FILE_SHA256
        or semantic_manifest_binding["payload_sha256"]
        != SEMANTIC_MANIFEST_PAYLOAD_SHA256
    ):
        raise StochasticBootstrapError(
            "semantic manifest is not the frozen MPNet/BERTScore contract"
        )
    runs, generation_suite, generation_suite_binding = load_generation_suite(
        generation_suite_manifest=generation_suite_manifest,
        generation_suite_manifest_sha256=generation_suite_manifest_sha256,
        chk0_manifest=chk0_manifest,
        chk1_manifest=chk1_manifest,
        chk3_manifest=chk3_manifest,
        sample_manifest=sample_manifest,
        sample_manifest_sha256=sample_manifest_sha256,
    )
    from jobs.eval import eval_chk3_stochastic_bootstrap_generation as stochastic

    source_bindings = {
        "scorer": _file_binding(Path(__file__)),
        "stochastic_runner_and_deep_validator": _file_binding(Path(stochastic.__file__)),
        "hard_gate": _file_binding(Path(gate_eval.__file__)),
        "semantic_backend": _file_binding(Path(semantic_eval.__file__)),
    }
    input_bindings = {
        "sample_manifest": _file_binding(sample_manifest, sealed=True),
        "generation_suite_manifest": generation_suite_binding,
        "selection_n12_sample_manifest": _file_binding(
            selection_sample_manifest, sealed=True
        ),
        "greedy_n12_six_metric_anchor": greedy_anchor_binding,
        "semantic_model_manifest": semantic_manifest_binding,
        "run_manifests": {
            model_id: dict(runs[model_id]["manifest_binding"])
            for model_id in MODEL_ORDER
        },
        "generation_artifacts": {
            model_id: dict(runs[model_id]["manifest"]["artifacts"]["generations"])
            for model_id in MODEL_ORDER
        },
    }
    def load_and_score_semantics() -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, Any],
        dict[str, Any],
    ]:
        try:
            bert, mpnet, provenance = semantic_eval.load_formal_semantic_backends(
                semantic_manifest_path=semantic_manifest,
                batch_size=semantic_batch_size,
                device=semantic_device,
            )
        except Exception as exc:
            raise StochasticBootstrapError(
                f"cannot load pinned semantic backends on {semantic_device}: {exc}"
            ) from exc
        scored, failed, audit = build_scored_rows(
            runs=runs, bert=bert, mpnet=mpnet, formal=True
        )
        return scored, failed, audit, provenance

    semantic_gpu_lease: dict[str, Any] | None = None
    if semantic_device == "cuda:0":
        physical_gpu_identity = stochastic._physical_gpu0_identity()

        def wait_notice(
            reason: str, processes: Sequence[Mapping[str, Any]]
        ) -> None:
            print(
                _canonical_json(
                    {
                        "status": "waiting_for_exclusive_gpu0_semantic_scoring",
                        "reason": reason,
                        "external_processes": list(processes),
                    }
                ),
                file=sys.stderr,
                flush=True,
            )

        try:
            with stochastic.exclusive_gpu0_lease(
                timeout_seconds=gpu_wait_timeout_seconds,
                poll_seconds=gpu_poll_seconds,
                on_wait=wait_notice,
            ) as lease:
                semantic_gpu_lease = dict(lease)
                import torch

                if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                    raise StochasticBootstrapError(
                        "semantic scoring requires exactly one visible CUDA GPU"
                    )
                logical_gpu_uuid = stochastic._verify_logical_cuda0_is_physical_gpu0(
                    torch, physical_gpu_identity
                )
                row_scores, failures, semantic_audit, semantic_provenance = (
                    load_and_score_semantics()
                )
                external_at_release = stochastic._external_gpu0_compute_processes()
                if external_at_release:
                    raise StochasticBootstrapError(
                        "external GPU0 process detected during semantic scoring: "
                        f"{external_at_release}"
                    )
                verified_after_load = stochastic._verify_logical_cuda0_is_physical_gpu0(
                    torch, physical_gpu_identity
                )
                if verified_after_load != logical_gpu_uuid:
                    raise StochasticBootstrapError(
                        "logical CUDA UUID changed during semantic backend scoring"
                    )
                semantic_gpu_lease.update(
                    {
                        "physical_gpu_identity": physical_gpu_identity,
                        "logical_cuda0_uuid": logical_gpu_uuid,
                        "cuda_device_order": "PCI_BUS_ID",
                        "external_compute_pids_at_release_check": [],
                    }
                )
        except StochasticBootstrapError:
            raise
        except Exception as exc:
            raise StochasticBootstrapError(
                f"exclusive GPU0 semantic scoring failed: {exc}"
            ) from exc
    else:
        row_scores, failures, semantic_audit, semantic_provenance = (
            load_and_score_semantics()
        )
    if len(row_scores) != EXPECTED_TOTAL_ROWS:
        raise StochasticBootstrapError(
            f"formal scored matrix must contain {EXPECTED_TOTAL_ROWS} rows"
        )
    results, draw_rows = build_bootstrap_results(
        row_scores=row_scores,
        full_sample_manifest=full_manifest,
        selection_sample_manifest=selection_manifest,
        draws=draws,
        seed=seed,
        formal=True,
    )

    # Publish only after all expensive inference/statistics have succeeded.  A
    # failed scorer therefore cannot leave the requested final directory looking
    # like a resumable or complete result.
    if resolved_output.exists() or resolved_output.is_symlink():
        raise StochasticBootstrapError(
            f"output directory appeared during scoring: {resolved_output}"
        )
    resolved_output.mkdir(parents=True, exist_ok=False)
    _fsync_directory(resolved_output.parent)
    row_path = resolved_output / "row_scores.jsonl"
    failure_path = resolved_output / "failure_samples.jsonl"
    draws_path = resolved_output / "bootstrap_draws.jsonl"
    results_path = resolved_output / "bootstrap_results.json"
    _write_new_jsonl(row_path, row_scores)
    _write_new_jsonl(failure_path, failures)
    _write_new_jsonl(draws_path, draw_rows)
    _write_new_json(results_path, results)
    artifacts = {
        "row_scores": {**_file_binding(row_path), "rows": len(row_scores)},
        "failure_samples": {**_file_binding(failure_path), "rows": len(failures)},
        "bootstrap_draws": {**_file_binding(draws_path), "rows": len(draw_rows)},
        "bootstrap_results": _file_binding(results_path, sealed=True),
    }

    # Recheck all immutable inputs after expensive inference and before any
    # top-level status=complete manifest can become visible.
    for label in (
        "sample_manifest",
        "generation_suite_manifest",
        "selection_n12_sample_manifest",
        "semantic_model_manifest",
        "greedy_n12_six_metric_anchor",
    ):
        if _file_binding(
            Path(input_bindings[label]["path"]), sealed=True
        ) != input_bindings[label]:
            raise StochasticBootstrapError(f"{label} changed during scoring")
    for model_id in MODEL_ORDER:
        if _file_binding(
            Path(input_bindings["run_manifests"][model_id]["path"]), sealed=True
        ) != input_bindings["run_manifests"][model_id]:
            raise StochasticBootstrapError(f"{model_id} manifest changed during scoring")
        generation_binding = input_bindings["generation_artifacts"][model_id]
        observed = _file_binding(Path(generation_binding["path"]))
        for key in ("sha256", "bytes"):
            if observed[key] != generation_binding[key]:
                raise StochasticBootstrapError(
                    f"{model_id} generations changed during scoring"
                )
    for label, binding in artifacts.items():
        observed = _file_binding(
            Path(binding["path"]), sealed=label == "bootstrap_results"
        )
        for key in ("sha256", "bytes"):
            if observed[key] != binding[key]:
                raise StochasticBootstrapError(
                    f"new output artifact changed before publication: {label}"
                )
    for label, binding in source_bindings.items():
        if _file_binding(Path(binding["path"])) != binding:
            raise StochasticBootstrapError(f"source code changed during scoring: {label}")
    failure_counts = Counter()
    for failure in failures:
        failure_counts.update(failure["preregistered_core_failures"])
    manifest = seal_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "operation": "offline_six_metric_scoring_and_cpu_hierarchical_paired_bootstrap",
            "model_order": list(MODEL_ORDER),
            "generation_order": list(GENERATION_ORDER),
            "metric_order": list(SIX_METRICS),
            "execution": {
                "semantic_device": semantic_device,
                "semantic_batch_size": semantic_batch_size,
                "bootstrap_device": "cpu",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "python_executable": sys.executable,
                "network_at_scoring_time": False,
                "semantic_gpu0_exclusive_lease": semantic_gpu_lease,
            },
            "inputs": input_bindings,
            "semantic_models": semantic_provenance,
            "semantic_backend_audit": semantic_audit,
            "artifacts": artifacts,
            "coverage": {
                "generation_rows": len(row_scores),
                "expected_generation_rows": EXPECTED_TOTAL_ROWS,
                "rows_per_model": {
                    model_id: len(runs[model_id]["results"])
                    for model_id in MODEL_ORDER
                },
                "paired_tuple_coverage_identical": True,
                "input_truncation_rows": sum(
                    row.get("input_truncated") is not False
                    for model_id in MODEL_ORDER
                    for row in runs[model_id]["results"]
                ),
                "hard_gate_failure_rows": len(failures),
                "hard_gate_failure_counts": dict(sorted(failure_counts.items())),
                "bootstrap_draw_rows": len(draw_rows),
                "generation_suite_coverage": generation_suite["coverage"],
            },
            "scoring_contract": {
                "identity_semantic_policy": "excluded_from_semantic_denominator",
                "hard_gate_failure_semantic_policy": "both_metrics_fixed_to_zero",
                "meeting_equal_weighting": True,
                "weighted_composite_calculated": False,
                "bootstrap_results_payload_sha256": results["integrity"][
                    "payload_sha256"
                ],
                "bootstrap_draws_sha256": artifacts["bootstrap_draws"]["sha256"],
                "greedy_anchor_role": (
                    "deterministic descriptive anchor only; excluded from stochastic bootstrap"
                ),
                "greedy_anchor_payload_sha256": greedy_anchor["integrity"][
                    "payload_sha256"
                ],
            },
            "sources": source_bindings,
            "limitations": [
                *results["limitations"],
                "The retained greedy N12 scorecard is descriptive and is not pooled into stochastic inference.",
            ],
        }
    )
    manifest_path = resolved_output / "manifest.json"
    _write_new_json(manifest_path, manifest)
    validate_manifest_integrity(manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-suite-manifest", required=True, type=Path)
    parser.add_argument("--generation-suite-manifest-sha256", required=True)
    parser.add_argument("--chk0-manifest", required=True, type=Path)
    parser.add_argument("--chk1-manifest", required=True, type=Path)
    parser.add_argument("--chk3-manifest", required=True, type=Path)
    parser.add_argument("--sample-manifest", required=True, type=Path)
    parser.add_argument("--sample-manifest-sha256", required=True)
    parser.add_argument("--selection-sample-manifest", required=True, type=Path)
    parser.add_argument("--selection-sample-manifest-sha256", required=True)
    parser.add_argument("--greedy-anchor-scorecard", required=True, type=Path)
    parser.add_argument("--greedy-anchor-scorecard-sha256", required=True)
    parser.add_argument("--semantic-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--semantic-device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--semantic-batch-size", type=int, default=8)
    parser.add_argument("--gpu-wait-timeout-seconds", type=int, default=172_800)
    parser.add_argument("--gpu-poll-seconds", type=int, default=30)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        manifest = score_and_bootstrap(
            generation_suite_manifest=args.generation_suite_manifest,
            generation_suite_manifest_sha256=args.generation_suite_manifest_sha256,
            chk0_manifest=args.chk0_manifest,
            chk1_manifest=args.chk1_manifest,
            chk3_manifest=args.chk3_manifest,
            sample_manifest=args.sample_manifest,
            sample_manifest_sha256=args.sample_manifest_sha256,
            selection_sample_manifest=args.selection_sample_manifest,
            selection_sample_manifest_sha256=args.selection_sample_manifest_sha256,
            greedy_anchor_scorecard=args.greedy_anchor_scorecard,
            greedy_anchor_scorecard_sha256=args.greedy_anchor_scorecard_sha256,
            semantic_manifest=args.semantic_manifest,
            output_dir=args.output_dir,
            semantic_device=args.semantic_device,
            semantic_batch_size=args.semantic_batch_size,
            gpu_wait_timeout_seconds=args.gpu_wait_timeout_seconds,
            gpu_poll_seconds=args.gpu_poll_seconds,
            draws=args.bootstrap_draws,
            seed=args.bootstrap_seed,
        )
    except (StochasticBootstrapError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output": str(args.output_dir.expanduser().resolve()),
                "rows": manifest["coverage"]["generation_rows"],
                "draws_sha256": manifest["scoring_contract"][
                    "bootstrap_draws_sha256"
                ],
                "payload_sha256": manifest["integrity"]["payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
