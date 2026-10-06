"""Estimate meeting-level synthetic-to-reference sentiment-score closeness.

This CPU-only, versioned estimator consumes the sealed regression-v2
sentiment suite and the sealed official Minutes-release ledger.  It evaluates
the artifact arms ``chk0``, ``chk1``, and ``chk3`` against the deterministic
synthetic reference at the meeting grain.  The paper's CHK2 label maps to the
artifact arm ``chk3`` (checkpoint 318); no ``chk2`` artifact arm exists.

Uncertainty uses one paired plan per backend/view/score-policy cell: calendar
years of official Minutes releases are sampled as blocks and the finite K=5
generation replicates are sampled within meeting.  The same indices are used
for all synthetic arms.  Results describe agreement with a synthetic teacher
target under fixed sentiment classifiers; they are neither model-quality
scores nor causal market evidence.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import math
import os
import secrets
import stat
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np

if sys.version_info < (3, 11):
    raise RuntimeError("the sealed release-ledger loader requires Python 3.11+")

from jobs.eval import build_chk3_beta_minutes_release_market_panel as market
from jobs.eval import chk3_beta_statistics as statistics
from jobs.eval import eval_chk3_beta_core8_sentiment_stochastic_schedule_v1 as sentiment
from jobs.eval import seal_chk3_beta_sentiment_regression_suite_v2 as regression_suite
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / (
    "output/evaluation/main/chk3_beta_core8_merged_n2048_vllm_k5_20260816_v1"
)
DEFAULT_SENTIMENT_SUITE = (
    RUN_ROOT
    / "sentiment_core8_stochastic_schedule_v1/suite_manifest.regression_v2.json"
)
DEFAULT_RELEASE_LEDGER = (
    RUN_ROOT
    / "market_panel_fed_minutes_release_daily_v1/official_minutes_release_dates.v1.jsonl"
)
DEFAULT_OUTPUT = RUN_ROOT / "sentiment_score_closeness_bootstrap_v1"

EVALUATION_ID = "chk3-sentiment-score-closeness-bootstrap-v1"
MANIFEST_SCHEMA = "chk3-sentiment-score-closeness-bootstrap-manifest-v1"
MEETING_SCHEMA = "chk3-sentiment-score-closeness-meeting-aggregate-v1"
ESTIMATE_SCHEMA = "chk3-sentiment-score-closeness-estimate-v1"
CONTRAST_SCHEMA = "chk3-sentiment-score-closeness-contrast-v1"
DRAW_SCHEMA = "chk3-sentiment-score-closeness-bootstrap-draw-v1"
DIAGNOSTICS_SCHEMA = "chk3-sentiment-score-closeness-diagnostics-v1"
REPORT_MODEL_SCHEMA = "chk3-sentiment-score-closeness-report-model-v1"

MEETING_FILENAME = "meeting_aggregates.v1.jsonl"
ESTIMATES_FILENAME = "estimates.v1.jsonl"
CONTRASTS_FILENAME = "contrasts.v1.jsonl"
DRAWS_FILENAME = "bootstrap_draws.v1.jsonl"
DIAGNOSTICS_FILENAME = "diagnostics.v1.json"

BACKEND_ORDER = (sentiment.DISTIL_BACKEND, sentiment.FINBERT_BACKEND)
SYNTHETIC_ARMS = statistics.SYNTHETIC_ARMS
ARM_ORDER = ("reference", *SYNTHETIC_ARMS)
VIEW_ORDER = (
    "pre_external",
    "full",
    "post",
    "full_excl_current",
    "post_excl_current",
)
SCORE_POLICY_ORDER = (
    "shared_complete_core8",
    "neutral_imputed_fixed_k5",
)
SCALE_ORDER = ("raw_score", "pre_external_reference_sd")
METRIC_ORDER = (
    "pearson_correlation",
    "spearman_correlation",
    "signed_bias",
    "mean_absolute_error",
    "root_mean_squared_error",
)
CONTRAST_ORDER = (
    "chk3_closeness_gain_vs_chk1",
    "chk3_closeness_gain_vs_chk0",
)

PRIMARY_BACKEND = sentiment.DISTIL_BACKEND
PRIMARY_VIEW = "pre_external"
PRIMARY_SCORE_POLICY = "shared_complete_core8"
PRIMARY_SCALE = "pre_external_reference_sd"
PAPER_EXPERIMENTAL_STAGE_ID = "chk2"
ARTIFACT_EXPERIMENTAL_ARM_ID = "chk3"
REPLICATES_PER_MEETING = 5
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20_260_817
REQUIRED_RUNTIME_ENVIRONMENT = {
    "PYTHONNOUSERSITE": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

EXPECTED_VIEW_MEETINGS = {
    "pre_external": 128,
    "full": 256,
    "post": 128,
    "full_excl_current": 247,
    "post_excl_current": 119,
}
EXPECTED_VIEW_BLOCKS = {
    "pre_external": 17,
    "full": 33,
    "post": 17,
    "full_excl_current": 32,
    "post_excl_current": 16,
}
VIEW_LABELS = {
    "pre_external": "Pre-2009 external meetings (primary)",
    "full": "Full 1993-2025 sentiment panel (descriptive)",
    "post": "Post-2008 sentiment panel (descriptive)",
    "full_excl_current": "Full panel excluding current cp318-selection exposure",
    "post_excl_current": "Post panel excluding current cp318-selection exposure",
}
BACKEND_LABELS = {
    sentiment.DISTIL_BACKEND: "DistilBERT FOMC stance",
    sentiment.FINBERT_BACKEND: "ProsusAI FinBERT financial valence",
}
ARM_LABELS = {
    "reference": "Deterministic synthetic reference",
    "chk0": "CHK0",
    "chk1": "CHK1",
    "chk3": "CHK2 (artifact CHK3/cp318)",
}
SCALE_UNITS = {
    "raw_score": "backend_signed_score_units",
    "pre_external_reference_sd": "pre_external_reference_standard_deviations",
}
METRIC_LABELS = {
    "pearson_correlation": "Pearson correlation",
    "spearman_correlation": "Spearman rank correlation",
    "signed_bias": "Signed mean error (model minus reference)",
    "mean_absolute_error": "Mean absolute error",
    "root_mean_squared_error": "Root mean squared error",
}
METRIC_DIRECTIONS = {
    "pearson_correlation": "higher_is_closer_in_linear_association_only",
    "spearman_correlation": "higher_is_closer_in_rank_association_only",
    "signed_bias": "zero_is_closer",
    "mean_absolute_error": "lower_is_closer",
    "root_mean_squared_error": "lower_is_closer",
}
GAIN_METRIC_IDS = {
    "pearson_correlation": "pearson_correlation_gain",
    "spearman_correlation": "spearman_correlation_gain",
    "signed_bias": "absolute_bias_closeness_gain",
    "mean_absolute_error": "mean_absolute_error_closeness_gain",
    "root_mean_squared_error": "root_mean_squared_error_closeness_gain",
}

AT_FDCWD = -100
RENAME_NOREPLACE = 1


class ClosenessEstimatorError(RuntimeError):
    """An input, numerical, replay, or publication contract failed closed."""


@dataclass(frozen=True)
class ReferenceScale:
    backend_id: str
    mean: float
    standard_deviation: float
    meeting_ids: tuple[str, ...]
    ddof: int = 1


@dataclass(frozen=True)
class BackendScores:
    backend_id: str
    construct: str
    reference_by_meeting: Mapping[str, float]
    synthetic_by_meeting: Mapping[str, Mapping[str, Mapping[int, float]]]
    neutral_reference_by_meeting: Mapping[str, float]
    neutral_synthetic_by_meeting: Mapping[str, Mapping[str, Mapping[int, float]]]
    paired_replicates_by_meeting: Mapping[str, tuple[int, ...]]
    exposure_by_meeting: Mapping[str, bool]


@dataclass(frozen=True)
class SentimentView:
    view_id: str
    meeting_ids: tuple[str, ...]
    release_dates: tuple[str, ...]
    release_years: tuple[str, ...]
    block_ids: tuple[str, ...]

    @property
    def n_meetings(self) -> int:
        return len(self.meeting_ids)


@dataclass(frozen=True)
class AnalysisResult:
    draw_rows: int
    estimates: tuple[dict[str, Any], ...]
    contrasts: tuple[dict[str, Any], ...]
    plan_diagnostic: dict[str, Any]


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
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
        raise ClosenessEstimatorError(f"non-canonical payload: {exc}") from exc


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClosenessEstimatorError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ClosenessEstimatorError(f"{name} must be finite")
    return result


def _binding(
    path: Path, *, rows: int | None = None, payload_sha256: str | None = None
) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise ClosenessEstimatorError(f"bound path is a symlink: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise ClosenessEstimatorError(f"bound file is missing: {resolved}")
    value: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if rows is not None:
        value["rows"] = rows
    if payload_sha256 is not None:
        value["payload_sha256"] = payload_sha256
    return value


_IMPLEMENTATION_IMPORT_BINDINGS = {
    "estimator": _binding(Path(__file__).resolve()),
    "statistical_primitives": _binding(Path(statistics.__file__).resolve()),
    "sentiment_loader": _binding(Path(sentiment.__file__).resolve()),
    "sentiment_regression_suite_loader": _binding(
        Path(regression_suite.__file__).resolve()
    ),
    "market_release_loader": _binding(Path(market.__file__).resolve()),
}


def _implementation_source_bindings() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source_id, imported in _IMPLEMENTATION_IMPORT_BINDINGS.items():
        current = _binding(Path(str(imported["path"])))
        if current != imported:
            raise ClosenessEstimatorError(
                f"implementation source changed after import: {source_id}"
            )
        result[source_id] = current
    return result


def _require_runtime_environment() -> dict[str, str]:
    observed = {name: os.environ.get(name, "") for name in REQUIRED_RUNTIME_ENVIRONMENT}
    if observed != REQUIRED_RUNTIME_ENVIRONMENT:
        raise ClosenessEstimatorError(
            "runtime environment must exactly set PYTHONNOUSERSITE and all "
            "OMP/OPENBLAS/MKL/NUMEXPR thread counts to 1"
        )
    return observed


def _record_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(dict(value)).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ClosenessEstimatorError(f"JSON file missing/symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClosenessEstimatorError(str(exc)) from exc
    if not isinstance(value, dict):
        raise ClosenessEstimatorError(f"JSON file is not an object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ClosenessEstimatorError(f"JSONL file missing/symlink: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ClosenessEstimatorError(
                        f"invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ClosenessEstimatorError(
                        f"JSONL row is not an object: {path}:{line_number}"
                    )
                yield value
    except (OSError, UnicodeDecodeError) as exc:
        raise ClosenessEstimatorError(str(exc)) from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _load_inputs(
    *, sentiment_suite_manifest: Path, release_ledger: Path
) -> dict[str, Any]:
    suite_path = sentiment_suite_manifest.expanduser().resolve()
    ledger_path = release_ledger.expanduser().resolve()
    if ledger_path.name != "official_minutes_release_dates.v1.jsonl":
        raise ClosenessEstimatorError("official release-ledger filename drift")
    market_manifest_path = ledger_path.parent / "manifest.json"
    try:
        suite_loaded = regression_suite.load_and_validate_regression_suite(suite_path)
        market_loaded = market.validate_panel(market_manifest_path)
    except Exception as exc:
        raise ClosenessEstimatorError(str(exc)) from exc
    market_manifest = market_loaded["manifest"]
    release_binding = market_manifest.get("artifacts", {}).get("release_dates")
    if not isinstance(release_binding, Mapping) or dict(release_binding) != _binding(
        ledger_path, rows=256
    ):
        raise ClosenessEstimatorError("release ledger is not the sealed market binding")
    release_rows = market_loaded["release_rows"]
    if len(release_rows) != 256:
        raise ClosenessEstimatorError("release ledger is not N=256")
    market_payload = validate_manifest_integrity(market_manifest)
    return {
        "sentiment": suite_loaded,
        "release_rows": release_rows,
        "sentiment_manifest_binding": suite_loaded["manifest_binding"],
        "release_ledger_binding": dict(release_binding),
        "market_manifest_binding": _binding(
            market_manifest_path, payload_sha256=market_payload
        ),
    }


def _backend_scores(loaded: Mapping[str, Any]) -> BackendScores:
    manifest = loaded.get("manifest")
    rows = loaded.get("meeting_scores")
    if not isinstance(manifest, Mapping) or not isinstance(rows, Sequence):
        raise ClosenessEstimatorError("sentiment backend payload missing")
    backend_id = str(manifest.get("backend"))
    construct = str(manifest.get("construct"))
    reference: dict[str, float] = {}
    neutral_reference: dict[str, float] = {}
    synthetic: dict[str, dict[str, dict[int, float]]] = {
        arm: defaultdict(dict) for arm in SYNTHETIC_ARMS
    }
    neutral_synthetic: dict[str, dict[str, dict[int, float]]] = {
        arm: defaultdict(dict) for arm in SYNTHETIC_ARMS
    }
    exposure: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ClosenessEstimatorError("sentiment meeting row is not an object")
        meeting_id = str(row.get("meeting_id"))
        arm = str(row.get("arm"))
        exposed = row.get("cp318_selection_exposed")
        if not isinstance(exposed, bool):
            raise ClosenessEstimatorError("sentiment exposure is not boolean")
        if meeting_id in exposure and exposure[meeting_id] is not exposed:
            raise ClosenessEstimatorError(f"exposure drift: {meeting_id}")
        exposure[meeting_id] = exposed
        neutral_score = _finite(
            f"{backend_id}:{meeting_id}:{arm}:neutral", row.get("neutral_imputed_score")
        )
        replicate_id = row.get("replicate_id")
        if arm == "reference":
            if replicate_id is not None or meeting_id in neutral_reference:
                raise ClosenessEstimatorError("reference identity drift")
            neutral_reference[meeting_id] = neutral_score
        else:
            if (
                arm not in SYNTHETIC_ARMS
                or isinstance(replicate_id, bool)
                or not isinstance(replicate_id, int)
                or replicate_id not in range(REPLICATES_PER_MEETING)
                or replicate_id in neutral_synthetic[arm][meeting_id]
            ):
                raise ClosenessEstimatorError("synthetic neutral identity drift")
            neutral_synthetic[arm][meeting_id][replicate_id] = neutral_score
        if row.get("complete_core8") is not True or row.get("score") is None:
            continue
        score = _finite(f"{backend_id}:{meeting_id}:{arm}:score", row.get("score"))
        if arm == "reference":
            if meeting_id in reference:
                raise ClosenessEstimatorError("duplicate reference score")
            reference[meeting_id] = score
        else:
            assert isinstance(replicate_id, int)
            if replicate_id in synthetic[arm][meeting_id]:
                raise ClosenessEstimatorError("duplicate synthetic score")
            synthetic[arm][meeting_id][replicate_id] = score
    if len(exposure) != 256 or set(reference) != set(exposure):
        raise ClosenessEstimatorError("sentiment backend meeting coverage is not N=256")
    paired: dict[str, tuple[int, ...]] = {}
    for meeting_id in sorted(exposure):
        if meeting_id not in neutral_reference or any(
            set(neutral_synthetic[arm].get(meeting_id, {}))
            != set(range(REPLICATES_PER_MEETING))
            for arm in SYNTHETIC_ARMS
        ):
            raise ClosenessEstimatorError(f"neutral K5 closure drift: {meeting_id}")
        shared = set(range(REPLICATES_PER_MEETING))
        for arm in SYNTHETIC_ARMS:
            shared &= set(synthetic[arm].get(meeting_id, {}))
        if not shared:
            raise ClosenessEstimatorError(f"no paired complete replicate: {meeting_id}")
        paired[meeting_id] = tuple(sorted(shared))
    counts = sorted(len(values) for values in paired.values())
    if counts.count(5) != 250 or counts.count(4) != 6 or set(counts) != {4, 5}:
        raise ClosenessEstimatorError("paired-complete K5 inventory drift")
    return BackendScores(
        backend_id=backend_id,
        construct=construct,
        reference_by_meeting=reference,
        synthetic_by_meeting=synthetic,
        neutral_reference_by_meeting=neutral_reference,
        neutral_synthetic_by_meeting=neutral_synthetic,
        paired_replicates_by_meeting=paired,
        exposure_by_meeting=exposure,
    )


def _ordered_release_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = [dict(row) for row in rows]
    if len(result) != 256:
        raise ClosenessEstimatorError("release row coverage is not N=256")
    identities = [str(row.get("generation_meeting_id")) for row in result]
    meeting_end_dates = [str(row.get("meeting_end_date")) for row in result]
    dates = [str(row.get("release_date")) for row in result]
    if len(set(identities)) != 256 or len(set(dates)) != 256:
        raise ClosenessEstimatorError("release identities/dates are not unique")
    if dates != sorted(dates):
        raise ClosenessEstimatorError("release ledger is not chronological")
    if identities != meeting_end_dates:
        raise ClosenessEstimatorError(
            "generation meeting IDs must equal ISO meeting-end dates"
        )
    if (
        sum(row.get("era") == "pre_external" for row in result) != 128
        or sum(row.get("era") == "post" for row in result) != 128
    ):
        raise ClosenessEstimatorError("release era coverage drift")
    if any(not isinstance(row.get("cp318_selection_exposed"), bool) for row in result):
        raise ClosenessEstimatorError("release exposure flag drift")
    return result


def _views(release_rows: Sequence[Mapping[str, Any]]) -> dict[str, SentimentView]:
    def include(view_id: str, row: Mapping[str, Any]) -> bool:
        era = row.get("era")
        exposed = row.get("cp318_selection_exposed") is True
        if view_id == "pre_external":
            return era == "pre_external"
        if view_id == "full":
            return True
        if view_id == "post":
            return era == "post"
        if view_id == "full_excl_current":
            return not exposed
        if view_id == "post_excl_current":
            return era == "post" and not exposed
        raise ClosenessEstimatorError(f"unknown view: {view_id}")

    result: dict[str, SentimentView] = {}
    for view_id in VIEW_ORDER:
        selected = [row for row in release_rows if include(view_id, row)]
        meeting_ids = tuple(str(row["generation_meeting_id"]) for row in selected)
        release_dates = tuple(str(row["release_date"]) for row in selected)
        years = tuple(value[:4] for value in release_dates)
        blocks = tuple(dict.fromkeys(years))
        if len(meeting_ids) != EXPECTED_VIEW_MEETINGS[view_id]:
            raise ClosenessEstimatorError(f"view N drift: {view_id}")
        if len(blocks) != EXPECTED_VIEW_BLOCKS[view_id]:
            raise ClosenessEstimatorError(f"view release-year count drift: {view_id}")
        result[view_id] = SentimentView(
            view_id=view_id,
            meeting_ids=meeting_ids,
            release_dates=release_dates,
            release_years=years,
            block_ids=blocks,
        )
    return result


def _validate_release_score_alignment(
    release_rows: Sequence[Mapping[str, Any]], scores: BackendScores
) -> None:
    meeting_ids = {str(row["generation_meeting_id"]) for row in release_rows}
    if meeting_ids != set(scores.reference_by_meeting):
        raise ClosenessEstimatorError(
            f"release/sentiment meeting mismatch: {scores.backend_id}"
        )
    for row in release_rows:
        meeting_id = str(row["generation_meeting_id"])
        if (
            row.get("cp318_selection_exposed")
            is not scores.exposure_by_meeting[meeting_id]
        ):
            raise ClosenessEstimatorError(
                f"release/sentiment exposure drift: {meeting_id}"
            )


def _reference_scale(
    scores: BackendScores, release_rows: Sequence[Mapping[str, Any]]
) -> ReferenceScale:
    meeting_ids = tuple(
        str(row["generation_meeting_id"])
        for row in release_rows
        if row.get("era") == "pre_external"
    )
    if len(meeting_ids) != 128:
        raise ClosenessEstimatorError("reference scale population is not N=128")
    values = np.asarray(
        [scores.reference_by_meeting[value] for value in meeting_ids],
        dtype=np.float64,
    )
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ClosenessEstimatorError(
            "reference scale standard deviation is not positive"
        )
    return ReferenceScale(
        backend_id=scores.backend_id,
        mean=mean,
        standard_deviation=standard_deviation,
        meeting_ids=meeting_ids,
    )


def _policy_sources(
    scores: BackendScores, score_policy: str
) -> tuple[
    Mapping[str, float],
    Mapping[str, Mapping[str, Mapping[int, float]]],
    Mapping[str, tuple[int, ...]],
]:
    if score_policy == "shared_complete_core8":
        return (
            scores.reference_by_meeting,
            scores.synthetic_by_meeting,
            scores.paired_replicates_by_meeting,
        )
    if score_policy == "neutral_imputed_fixed_k5":
        inventory = {
            meeting_id: tuple(range(REPLICATES_PER_MEETING))
            for meeting_id in scores.reference_by_meeting
        }
        return (
            scores.neutral_reference_by_meeting,
            scores.neutral_synthetic_by_meeting,
            inventory,
        )
    raise ClosenessEstimatorError(f"unknown score policy: {score_policy}")


def _score_arrays(
    view: SentimentView, scores: BackendScores, score_policy: str
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, tuple[int, ...]]]:
    reference_source, synthetic_source, inventory = _policy_sources(
        scores, score_policy
    )
    reference = np.asarray(
        [reference_source[meeting_id] for meeting_id in view.meeting_ids],
        dtype=np.float64,
    )
    matrices: dict[str, np.ndarray] = {}
    for arm in SYNTHETIC_ARMS:
        matrix = np.full(
            (view.n_meetings, REPLICATES_PER_MEETING), np.nan, dtype=np.float64
        )
        for meeting_index, meeting_id in enumerate(view.meeting_ids):
            for replicate_id, value in synthetic_source[arm][meeting_id].items():
                matrix[meeting_index, int(replicate_id)] = float(value)
        matrices[arm] = matrix
    return (
        reference,
        matrices,
        {meeting_id: inventory[meeting_id] for meeting_id in view.meeting_ids},
    )


def _point_means(
    view: SentimentView,
    matrices: Mapping[str, np.ndarray],
    inventory: Mapping[str, tuple[int, ...]],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for arm in SYNTHETIC_ARMS:
        result[arm] = np.asarray(
            [
                float(matrices[arm][meeting_index, list(inventory[meeting_id])].mean())
                for meeting_index, meeting_id in enumerate(view.meeting_ids)
            ],
            dtype=np.float64,
        )
    return result


def _midranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.all(np.isfinite(array)):
        raise ClosenessEstimatorError("midranks require a finite vector of length >=2")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _pearson(reference: np.ndarray, model: np.ndarray) -> float:
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(model, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ClosenessEstimatorError("correlation vectors are incompatible")
    centered_x = x - float(x.mean())
    centered_y = y - float(y.mean())
    denominator = math.sqrt(
        float(centered_x @ centered_x) * float(centered_y @ centered_y)
    )
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ClosenessEstimatorError("correlation is undefined for zero variance")
    value = float((centered_x @ centered_y) / denominator)
    if not math.isfinite(value):
        raise ClosenessEstimatorError("correlation is non-finite")
    return max(-1.0, min(1.0, value))


def _raw_metrics(reference: np.ndarray, model: np.ndarray) -> dict[str, float]:
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(model, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        raise ClosenessEstimatorError("metric vectors are incompatible")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ClosenessEstimatorError("metric vectors contain non-finite values")
    difference = y - x
    result = {
        "pearson_correlation": _pearson(x, y),
        # Ranking happens after block expansion, so repeated meetings create ties.
        "spearman_correlation": _pearson(_midranks(x), _midranks(y)),
        "signed_bias": float(difference.mean()),
        "mean_absolute_error": float(np.abs(difference).mean()),
        "root_mean_squared_error": float(
            math.sqrt(float(difference @ difference) / len(difference))
        ),
    }
    if set(result) != set(METRIC_ORDER) or any(
        not math.isfinite(value) for value in result.values()
    ):
        raise ClosenessEstimatorError("metric bundle is incomplete/non-finite")
    return result


def _scale_metrics(
    raw_metrics: Mapping[str, float], standard_deviation: float
) -> dict[str, float]:
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ClosenessEstimatorError("metric scaling requires positive sigma")
    return {
        "pearson_correlation": float(raw_metrics["pearson_correlation"]),
        "spearman_correlation": float(raw_metrics["spearman_correlation"]),
        "signed_bias": float(raw_metrics["signed_bias"] / standard_deviation),
        "mean_absolute_error": float(
            raw_metrics["mean_absolute_error"] / standard_deviation
        ),
        "root_mean_squared_error": float(
            raw_metrics["root_mean_squared_error"] / standard_deviation
        ),
    }


def _gain_values(
    metrics: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    if set(metrics) != set(SYNTHETIC_ARMS):
        raise ClosenessEstimatorError("gain arm inventory drift")
    result: dict[str, dict[str, float]] = {}
    for contrast_id, baseline in (
        ("chk3_closeness_gain_vs_chk1", "chk1"),
        ("chk3_closeness_gain_vs_chk0", "chk0"),
    ):
        values: dict[str, float] = {}
        for metric_id in METRIC_ORDER:
            after = float(metrics["chk3"][metric_id])
            before = float(metrics[baseline][metric_id])
            if metric_id in ("pearson_correlation", "spearman_correlation"):
                value = after - before
            elif metric_id == "signed_bias":
                value = abs(before) - abs(after)
            else:
                value = before - after
            values[metric_id] = float(value)
        result[contrast_id] = values
    return result


def _analysis_id(backend_id: str, view_id: str, score_policy: str) -> str:
    return f"{backend_id}::{view_id}::{score_policy}"


def _meeting_aggregate_rows(
    *,
    release_rows: Sequence[Mapping[str, Any]],
    scores_by_backend: Mapping[str, BackendScores],
    scales: Mapping[str, ReferenceScale],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backend_id in BACKEND_ORDER:
        scores = scores_by_backend[backend_id]
        scale = scales[backend_id]
        for score_policy in SCORE_POLICY_ORDER:
            reference_source, synthetic_source, inventory = _policy_sources(
                scores, score_policy
            )
            for release in release_rows:
                meeting_id = str(release["generation_meeting_id"])
                for arm in ARM_ORDER:
                    if arm == "reference":
                        replicate_ids: tuple[int, ...] = ()
                        raw_score = float(reference_source[meeting_id])
                    else:
                        replicate_ids = inventory[meeting_id]
                        raw_score = float(
                            np.mean(
                                [
                                    synthetic_source[arm][meeting_id][replicate_id]
                                    for replicate_id in replicate_ids
                                ]
                            )
                        )
                    row = {
                        "schema_version": MEETING_SCHEMA,
                        "row_id": (
                            f"{backend_id}::{score_policy}::{meeting_id}::{arm}"
                        ),
                        "backend_id": backend_id,
                        "backend_label": BACKEND_LABELS[backend_id],
                        "construct": scores.construct,
                        "score_policy": score_policy,
                        "meeting_id": meeting_id,
                        "meeting_date": str(release["meeting_end_date"]),
                        "release_date": str(release["release_date"]),
                        "release_year": str(release["release_date"])[:4],
                        "era": str(release["era"]),
                        "cp318_selection_exposed": bool(
                            release["cp318_selection_exposed"]
                        ),
                        "arm": arm,
                        "arm_label": ARM_LABELS[arm],
                        "paper_stage_id": (
                            PAPER_EXPERIMENTAL_STAGE_ID if arm == "chk3" else None
                        ),
                        "deterministic_reference": arm == "reference",
                        "replicate_ids": list(replicate_ids),
                        "replicate_count": len(replicate_ids),
                        "raw_score": raw_score,
                        "pre_external_reference_sd_score": float(
                            (raw_score - scale.mean) / scale.standard_deviation
                        ),
                    }
                    row["row_sha256"] = _record_sha256(row)
                    rows.append(row)
    expected = (
        len(BACKEND_ORDER)
        * len(SCORE_POLICY_ORDER)
        * EXPECTED_VIEW_MEETINGS["full"]
        * len(ARM_ORDER)
    )
    if len(rows) != expected or len({row["row_id"] for row in rows}) != expected:
        raise ClosenessEstimatorError("meeting aggregate matrix closure drift")
    return rows


def _bootstrap_plan(
    *,
    view: SentimentView,
    inventory: Mapping[str, tuple[int, ...]],
    score_policy: str,
    draws: int,
    seed: int,
) -> statistics.PairedBlockBootstrapPlan:
    draw_counts = {
        meeting_id: (
            REPLICATES_PER_MEETING
            if score_policy == "neutral_imputed_fixed_k5"
            else len(replicate_ids)
        )
        for meeting_id, replicate_ids in inventory.items()
    }
    try:
        return statistics.make_paired_block_bootstrap_plan(
            block_ids=view.block_ids,
            meeting_ids=view.meeting_ids,
            meeting_block_ids=view.release_years,
            replicate_ids_by_meeting=inventory,
            replicate_draw_counts_by_meeting=draw_counts,
            draws=draws,
            seed=seed,
            replicates_per_draw=REPLICATES_PER_MEETING,
        )
    except statistics.BetaStatisticsError as exc:
        raise ClosenessEstimatorError(str(exc)) from exc


def _metric_bundles(
    *,
    reference: np.ndarray,
    synthetic: Mapping[str, np.ndarray],
    standard_deviation: float,
) -> dict[str, dict[str, dict[str, float]]]:
    raw = {arm: _raw_metrics(reference, synthetic[arm]) for arm in SYNTHETIC_ARMS}
    standardized = {
        arm: _scale_metrics(raw[arm], standard_deviation) for arm in SYNTHETIC_ARMS
    }
    # Correlations must be bit-for-bit identical across a positive affine scale.
    for arm in SYNTHETIC_ARMS:
        for metric_id in ("pearson_correlation", "spearman_correlation"):
            if raw[arm][metric_id] != standardized[arm][metric_id]:
                raise ClosenessEstimatorError("correlation scale invariance drift")
        for metric_id in (
            "signed_bias",
            "mean_absolute_error",
            "root_mean_squared_error",
        ):
            if standardized[arm][metric_id] != (
                raw[arm][metric_id] / standard_deviation
            ):
                raise ClosenessEstimatorError("distance metric scale identity drift")
    return {"raw_score": raw, "pre_external_reference_sd": standardized}


def _gain_formula(metric_id: str) -> str:
    if metric_id in ("pearson_correlation", "spearman_correlation"):
        return "metric_chk3_minus_metric_baseline"
    if metric_id == "signed_bias":
        return "absolute_bias_baseline_minus_absolute_bias_chk3"
    return "error_metric_baseline_minus_error_metric_chk3"


def _percentile(values: Sequence[float]) -> tuple[float, float]:
    try:
        return statistics.percentile_interval(values, confidence=0.95)
    except statistics.BetaStatisticsError as exc:
        raise ClosenessEstimatorError(str(exc)) from exc


def _estimate_rows(
    *,
    view: SentimentView,
    scores: BackendScores,
    score_policy: str,
    point_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
    metric_draws: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
    plan_sha256: str,
    successful_draws: int,
    failed_draws: int,
) -> list[dict[str, Any]]:
    analysis_id = _analysis_id(scores.backend_id, view.view_id, score_policy)
    rows: list[dict[str, Any]] = []
    for scale_id in SCALE_ORDER:
        for arm in SYNTHETIC_ARMS:
            for metric_id in METRIC_ORDER:
                lower, upper = _percentile(metric_draws[scale_id][arm][metric_id])
                row = {
                    "schema_version": ESTIMATE_SCHEMA,
                    "estimate_id": (f"{analysis_id}::{scale_id}::{arm}::{metric_id}"),
                    "analysis_id": analysis_id,
                    "backend_id": scores.backend_id,
                    "backend_label": BACKEND_LABELS[scores.backend_id],
                    "construct": scores.construct,
                    "view_id": view.view_id,
                    "view_label": VIEW_LABELS[view.view_id],
                    "score_policy": score_policy,
                    "scale_id": scale_id,
                    "scale_units": SCALE_UNITS[scale_id],
                    "arm": arm,
                    "arm_label": ARM_LABELS[arm],
                    "paper_stage_id": (
                        PAPER_EXPERIMENTAL_STAGE_ID if arm == "chk3" else None
                    ),
                    "metric_id": metric_id,
                    "metric_label": METRIC_LABELS[metric_id],
                    "closeness_direction": METRIC_DIRECTIONS[metric_id],
                    "point_estimate": float(point_metrics[scale_id][arm][metric_id]),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "bootstrap_interval": "percentile_95_linear",
                    "n_meetings": view.n_meetings,
                    "successful_bootstrap_draws": successful_draws,
                    "failed_bootstrap_draws": failed_draws,
                    "bootstrap_plan_sha256": plan_sha256,
                }
                row["estimate_row_sha256"] = _record_sha256(row)
                rows.append(row)
    return rows


def _contrast_rows(
    *,
    view: SentimentView,
    scores: BackendScores,
    score_policy: str,
    point_gains: Mapping[str, Mapping[str, Mapping[str, float]]],
    gain_draws: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]],
    plan_sha256: str,
    successful_draws: int,
    failed_draws: int,
) -> list[dict[str, Any]]:
    analysis_id = _analysis_id(scores.backend_id, view.view_id, score_policy)
    rows: list[dict[str, Any]] = []
    for scale_id in SCALE_ORDER:
        for contrast_id in CONTRAST_ORDER:
            baseline = "chk1" if contrast_id.endswith("chk1") else "chk0"
            for metric_id in METRIC_ORDER:
                lower, upper = _percentile(gain_draws[scale_id][contrast_id][metric_id])
                row = {
                    "schema_version": CONTRAST_SCHEMA,
                    "contrast_row_id": (
                        f"{analysis_id}::{scale_id}::{contrast_id}::{metric_id}"
                    ),
                    "analysis_id": analysis_id,
                    "backend_id": scores.backend_id,
                    "backend_label": BACKEND_LABELS[scores.backend_id],
                    "construct": scores.construct,
                    "view_id": view.view_id,
                    "view_label": VIEW_LABELS[view.view_id],
                    "score_policy": score_policy,
                    "scale_id": scale_id,
                    "scale_units": SCALE_UNITS[scale_id],
                    "contrast_id": contrast_id,
                    "baseline_arm": baseline,
                    "after_arm": "chk3",
                    "paper_after_stage_id": PAPER_EXPERIMENTAL_STAGE_ID,
                    "metric_id": metric_id,
                    "metric_label": METRIC_LABELS[metric_id],
                    "gain_metric_id": GAIN_METRIC_IDS[metric_id],
                    "gain_formula": _gain_formula(metric_id),
                    "positive_interpretation": (
                        "positive means artifact CHK3/cp318 (paper CHK2) is closer "
                        "to the deterministic synthetic reference than the baseline"
                    ),
                    "point_estimate": float(
                        point_gains[scale_id][contrast_id][metric_id]
                    ),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "bootstrap_interval": "paired_percentile_95_linear",
                    "n_meetings": view.n_meetings,
                    "successful_bootstrap_draws": successful_draws,
                    "failed_bootstrap_draws": failed_draws,
                    "bootstrap_plan_sha256": plan_sha256,
                }
                row["contrast_row_sha256"] = _record_sha256(row)
                rows.append(row)
    return rows


def _run_analysis(
    *,
    view: SentimentView,
    scores: BackendScores,
    scale: ReferenceScale,
    score_policy: str,
    draws: int,
    seed: int,
    consume_draw: Callable[[dict[str, Any]], None],
) -> AnalysisResult:
    reference, matrices, inventory = _score_arrays(view, scores, score_policy)
    point_means = _point_means(view, matrices, inventory)
    point_metrics = _metric_bundles(
        reference=reference,
        synthetic=point_means,
        standard_deviation=scale.standard_deviation,
    )
    point_gains = {
        scale_id: _gain_values(point_metrics[scale_id]) for scale_id in SCALE_ORDER
    }
    plan = _bootstrap_plan(
        view=view,
        inventory=inventory,
        score_policy=score_policy,
        draws=draws,
        seed=seed,
    )
    metric_draws: dict[str, dict[str, dict[str, list[float]]]] = {
        scale_id: {
            arm: {metric_id: [] for metric_id in METRIC_ORDER} for arm in SYNTHETIC_ARMS
        }
        for scale_id in SCALE_ORDER
    }
    gain_draws: dict[str, dict[str, dict[str, list[float]]]] = {
        scale_id: {
            contrast_id: {metric_id: [] for metric_id in METRIC_ORDER}
            for contrast_id in CONTRAST_ORDER
        }
        for scale_id in SCALE_ORDER
    }
    successful = 0
    failed = 0
    analysis_id = _analysis_id(scores.backend_id, view.view_id, score_policy)
    for draw_index in range(draws):
        sampled_rows = statistics.sampled_meeting_row_indices(
            plan, view.release_years, draw_index=draw_index
        )
        try:
            sampled_means = statistics.resampled_meeting_means(
                matrices, plan, draw_index=draw_index
            )
            bundles = _metric_bundles(
                reference=reference[sampled_rows],
                synthetic={
                    arm: sampled_means[arm][sampled_rows] for arm in SYNTHETIC_ARMS
                },
                standard_deviation=scale.standard_deviation,
            )
            gains = {
                scale_id: _gain_values(bundles[scale_id]) for scale_id in SCALE_ORDER
            }
            row = {
                "schema_version": DRAW_SCHEMA,
                "analysis_id": analysis_id,
                "backend_id": scores.backend_id,
                "view_id": view.view_id,
                "score_policy": score_policy,
                "draw_index": draw_index,
                "plan_sha256": plan.sha256,
                "status": "ok",
                "failure_reason": None,
                "sampled_meeting_rows": int(len(sampled_rows)),
                "metrics": bundles,
                "closeness_gains": gains,
            }
            for scale_id in SCALE_ORDER:
                for arm in SYNTHETIC_ARMS:
                    for metric_id in METRIC_ORDER:
                        metric_draws[scale_id][arm][metric_id].append(
                            bundles[scale_id][arm][metric_id]
                        )
                for contrast_id in CONTRAST_ORDER:
                    for metric_id in METRIC_ORDER:
                        gain_draws[scale_id][contrast_id][metric_id].append(
                            gains[scale_id][contrast_id][metric_id]
                        )
            successful += 1
        except (ClosenessEstimatorError, statistics.BetaStatisticsError) as exc:
            row = {
                "schema_version": DRAW_SCHEMA,
                "analysis_id": analysis_id,
                "backend_id": scores.backend_id,
                "view_id": view.view_id,
                "score_policy": score_policy,
                "draw_index": draw_index,
                "plan_sha256": plan.sha256,
                "status": "failed",
                "failure_reason": str(exc),
                "sampled_meeting_rows": int(len(sampled_rows)),
                "metrics": None,
                "closeness_gains": None,
            }
            failed += 1
        row["draw_row_sha256"] = _record_sha256(row)
        consume_draw(row)
    if successful == 0 or successful + failed != draws:
        raise ClosenessEstimatorError("bootstrap success/failure closure drift")
    estimates = _estimate_rows(
        view=view,
        scores=scores,
        score_policy=score_policy,
        point_metrics=point_metrics,
        metric_draws=metric_draws,
        plan_sha256=plan.sha256,
        successful_draws=successful,
        failed_draws=failed,
    )
    contrasts = _contrast_rows(
        view=view,
        scores=scores,
        score_policy=score_policy,
        point_gains=point_gains,
        gain_draws=gain_draws,
        plan_sha256=plan.sha256,
        successful_draws=successful,
        failed_draws=failed,
    )
    plan_diagnostic = {
        "analysis_id": analysis_id,
        "backend_id": scores.backend_id,
        "view_id": view.view_id,
        "score_policy": score_policy,
        "n_meetings": view.n_meetings,
        "block_ids": list(view.block_ids),
        "calendar_year_blocks": len(view.block_ids),
        "ordered_meeting_ids_sha256": statistics.sha256_json(list(view.meeting_ids)),
        "ordered_release_year_vector_sha256": statistics.sha256_json(
            list(view.release_years)
        ),
        "replicate_inventories_sha256": statistics.sha256_json(
            {
                meeting_id: list(plan.replicate_ids_by_meeting[index])
                for index, meeting_id in enumerate(plan.meeting_ids)
            }
        ),
        "replicate_draw_counts_sha256": statistics.sha256_json(
            {
                meeting_id: plan.replicate_draw_counts_by_meeting[index]
                for index, meeting_id in enumerate(plan.meeting_ids)
            }
        ),
        "meeting_date_start": view.meeting_ids[0],
        "meeting_date_end": view.meeting_ids[-1],
        "release_date_start": view.release_dates[0],
        "release_date_end": view.release_dates[-1],
        "plan_sha256": plan.sha256,
        "draws": draws,
        "seed": seed,
        "numpy_version": np.__version__,
        "numpy_bit_generator": type(np.random.default_rng(seed).bit_generator).__name__,
        "successful_draws": successful,
        "failed_draws": failed,
        "replicate_draw_count_distribution": {
            str(value): plan.replicate_draw_counts_by_meeting.count(value)
            for value in sorted(set(plan.replicate_draw_counts_by_meeting))
        },
        "same_indices_across_synthetic_arms": True,
        "duplicate_block_reuses_meeting_resample": True,
        "spearman_reranked_after_block_expansion": True,
    }
    return AnalysisResult(
        draw_rows=draws,
        estimates=tuple(estimates),
        contrasts=tuple(contrasts),
        plan_diagnostic=plan_diagnostic,
    )


def _reference_scale_payload(scale: ReferenceScale) -> dict[str, Any]:
    return {
        "backend_id": scale.backend_id,
        "population": "all_128_pre_external_deterministic_reference_meetings",
        "rows": len(scale.meeting_ids),
        "mean": scale.mean,
        "standard_deviation": scale.standard_deviation,
        "ddof": scale.ddof,
        "meeting_ids": list(scale.meeting_ids),
        "meeting_ids_sha256": statistics.sha256_json(list(scale.meeting_ids)),
        "frozen_before_bootstrap": True,
    }


def _view_specs(views: Mapping[str, SentimentView]) -> list[dict[str, Any]]:
    return [
        {
            "view_id": view_id,
            "view_label": VIEW_LABELS[view_id],
            "role": ("primary_external" if view_id == PRIMARY_VIEW else "descriptive"),
            "selection_exclusion": (
                "current_meeting_cp318_selection_exposure_only"
                if view_id.endswith("excl_current")
                else "none"
            ),
            "n_meetings": views[view_id].n_meetings,
            "calendar_year_blocks": len(views[view_id].block_ids),
            "meeting_date_start": views[view_id].meeting_ids[0],
            "meeting_date_end": views[view_id].meeting_ids[-1],
            "release_date_start": views[view_id].release_dates[0],
            "release_date_end": views[view_id].release_dates[-1],
            "release_years": list(views[view_id].block_ids),
        }
        for view_id in VIEW_ORDER
    ]


def _support_diagnostics(
    *,
    views: Mapping[str, SentimentView],
    scores_by_backend: Mapping[str, BackendScores],
    scales: Mapping[str, ReferenceScale],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for backend_id in BACKEND_ORDER:
        scores = scores_by_backend[backend_id]
        scale = scales[backend_id]
        for view_id in VIEW_ORDER:
            view = views[view_id]
            for score_policy in SCORE_POLICY_ORDER:
                reference, matrices, inventory = _score_arrays(
                    view, scores, score_policy
                )
                point = _point_means(view, matrices, inventory)
                reference_sd = float(reference.std(ddof=1))
                values_by_arm = {"reference": reference, **point}
                for arm in ARM_ORDER:
                    values = values_by_arm[arm]
                    arm_sd = float(values.std(ddof=1))
                    rows.append(
                        {
                            "backend_id": backend_id,
                            "view_id": view_id,
                            "score_policy": score_policy,
                            "arm": arm,
                            "n_meetings": len(values),
                            "raw_mean": float(values.mean()),
                            "raw_standard_deviation_ddof1": arm_sd,
                            "raw_minimum": float(values.min()),
                            "raw_maximum": float(values.max()),
                            "pre_external_reference_sd_mean": float(
                                (float(values.mean()) - scale.mean)
                                / scale.standard_deviation
                            ),
                            "pre_external_reference_sd_standard_deviation_ddof1": float(
                                arm_sd / scale.standard_deviation
                            ),
                            "variance_ratio_to_same_view_reference": float(
                                (arm_sd / reference_sd) ** 2
                            ),
                        }
                    )
    return rows


def _diagnostics_payload(
    *,
    views: Mapping[str, SentimentView],
    scores_by_backend: Mapping[str, BackendScores],
    scales: Mapping[str, ReferenceScale],
    plan_diagnostics: Sequence[Mapping[str, Any]],
    draws: int,
    estimate_rows: int,
    contrast_rows: int,
    draw_rows: int,
    meeting_rows: int,
) -> dict[str, Any]:
    incomplete = {
        backend_id: [
            {
                "meeting_id": meeting_id,
                "paired_complete_replicate_ids": list(replicate_ids),
                "n_i": len(replicate_ids),
            }
            for meeting_id, replicate_ids in scores_by_backend[
                backend_id
            ].paired_replicates_by_meeting.items()
            if len(replicate_ids) != REPLICATES_PER_MEETING
        ]
        for backend_id in BACKEND_ORDER
    }
    if any(len(rows) != 6 for rows in incomplete.values()):
        raise ClosenessEstimatorError("incomplete replicate diagnostic drift")
    return seal_manifest(
        {
            "schema_version": DIAGNOSTICS_SCHEMA,
            "status": "complete",
            "immutable": True,
            "reference_scales": {
                backend_id: _reference_scale_payload(scales[backend_id])
                for backend_id in BACKEND_ORDER
            },
            "view_specs": _view_specs(views),
            "paired_complete_replicate_exceptions": incomplete,
            "support_diagnostics": _support_diagnostics(
                views=views,
                scores_by_backend=scores_by_backend,
                scales=scales,
            ),
            "bootstrap_plans": [dict(value) for value in plan_diagnostics],
            "coverage": {
                "backends": len(BACKEND_ORDER),
                "views": len(VIEW_ORDER),
                "score_policies": len(SCORE_POLICY_ORDER),
                "scales": len(SCALE_ORDER),
                "metrics": len(METRIC_ORDER),
                "synthetic_arms": len(SYNTHETIC_ARMS),
                "analysis_cells": len(BACKEND_ORDER)
                * len(VIEW_ORDER)
                * len(SCORE_POLICY_ORDER),
                "meeting_aggregate_rows": meeting_rows,
                "estimate_rows": estimate_rows,
                "contrast_rows": contrast_rows,
                "bootstrap_draw_rows": draw_rows,
                "bootstrap_draws_per_analysis": draws,
            },
            "all_metrics_are_meeting_grain": True,
            "correlations_are_agreement_metrics": False,
            "generation_replicates_are_outer_independent_units": False,
        }
    )


def _artifact_binding(
    *, staging_path: Path, final_path: Path, rows: int | None = None
) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "path": str(final_path.resolve()),
        "bytes": staging_path.stat().st_size,
        "sha256": sha256_file(staging_path),
    }
    if rows is not None:
        binding["rows"] = rows
    return binding


def _headline_estimate_ids() -> list[str]:
    analysis_id = _analysis_id(PRIMARY_BACKEND, PRIMARY_VIEW, PRIMARY_SCORE_POLICY)
    return [
        f"{analysis_id}::{PRIMARY_SCALE}::chk3::{metric_id}"
        for metric_id in METRIC_ORDER
    ]


def _headline_contrast_ids() -> list[str]:
    analysis_id = _analysis_id(PRIMARY_BACKEND, PRIMARY_VIEW, PRIMARY_SCORE_POLICY)
    return [
        f"{analysis_id}::{PRIMARY_SCALE}::{contrast_id}::{metric_id}"
        for contrast_id in CONTRAST_ORDER
        for metric_id in METRIC_ORDER
    ]


def _manifest_payload(
    *,
    created_at_utc: str,
    final_root: Path,
    staging_root: Path,
    inputs: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    meeting_rows: int,
    estimate_rows: int,
    contrast_rows: int,
    draw_rows: int,
    draws: int,
    seed: int,
    publish_attempt_id: str,
    staging_basename: str,
) -> dict[str, Any]:
    coverage = copy.deepcopy(dict(diagnostics["coverage"]))
    generation_binding = inputs["sentiment"]["inventory"]["manifest"].get(
        "source_generation_suite"
    )
    if not isinstance(generation_binding, Mapping):
        raise ClosenessEstimatorError("generation-suite provenance binding missing")
    artifacts = {
        "meeting_aggregates": _artifact_binding(
            staging_path=staging_root / MEETING_FILENAME,
            final_path=final_root / MEETING_FILENAME,
            rows=meeting_rows,
        ),
        "estimates": _artifact_binding(
            staging_path=staging_root / ESTIMATES_FILENAME,
            final_path=final_root / ESTIMATES_FILENAME,
            rows=estimate_rows,
        ),
        "contrasts": _artifact_binding(
            staging_path=staging_root / CONTRASTS_FILENAME,
            final_path=final_root / CONTRASTS_FILENAME,
            rows=contrast_rows,
        ),
        "bootstrap_draws": _artifact_binding(
            staging_path=staging_root / DRAWS_FILENAME,
            final_path=final_root / DRAWS_FILENAME,
            rows=draw_rows,
        ),
        "diagnostics": _artifact_binding(
            staging_path=staging_root / DIAGNOSTICS_FILENAME,
            final_path=final_root / DIAGNOSTICS_FILENAME,
        ),
    }
    return {
        "schema_version": MANIFEST_SCHEMA,
        "evaluation_id": EVALUATION_ID,
        "created_at_utc": created_at_utc,
        "status": "complete",
        "immutable": True,
        "causal_claim_permitted": False,
        "paper_to_artifact_mapping": {
            "paper_stage_id": PAPER_EXPERIMENTAL_STAGE_ID,
            "artifact_arm_id": ARTIFACT_EXPERIMENTAL_ARM_ID,
            "artifact_checkpoint": 318,
            "chk2_artifact_arm_exists": False,
        },
        "backend_order": list(BACKEND_ORDER),
        "arm_order": list(ARM_ORDER),
        "view_order": list(VIEW_ORDER),
        "score_policy_order": list(SCORE_POLICY_ORDER),
        "scale_order": list(SCALE_ORDER),
        "metric_order": list(METRIC_ORDER),
        "contrast_order": list(CONTRAST_ORDER),
        "primary_selectors": {
            "backend_id": PRIMARY_BACKEND,
            "view_id": PRIMARY_VIEW,
            "score_policy": PRIMARY_SCORE_POLICY,
            "scale_id": PRIMARY_SCALE,
            "arm": ARTIFACT_EXPERIMENTAL_ARM_ID,
            "paper_stage_id": PAPER_EXPERIMENTAL_STAGE_ID,
        },
        "analysis_contract": {
            "grain": "meeting",
            "reference": "deterministic synthetic teacher target; not official FOMC Minutes",
            "metric_formulas": {
                "pearson_correlation": "sample Pearson correlation(reference, model)",
                "spearman_correlation": "sample Pearson correlation(midrank(reference), midrank(model)); reranked after block expansion",
                "signed_bias": "mean(model_minus_reference)",
                "mean_absolute_error": "mean(abs(model_minus_reference))",
                "root_mean_squared_error": "sqrt(mean((model_minus_reference)^2))",
            },
            "scale_transform": (
                "correlations are exact-identical across scales; raw bias/MAE/RMSE "
                "are divided by the positive frozen pre-external reference sample SD"
            ),
            "primary_missing_policy": (
                "per-meeting intersection of complete replicate IDs shared by chk0/chk1/chk3; "
                "observed-size n_i mean and bootstrap"
            ),
            "neutral_sensitivity": (
                "missing topic signed score zero; fixed replicate IDs 0..4 and K=5"
            ),
            "bootstrap_unit": "calendar year of official current Minutes release",
            "bootstrap_draws": draws,
            "bootstrap_seed": seed,
            "within_meeting_resampling": "with replacement using shared positions across chk0/chk1/chk3",
            "duplicate_calendar_block_replicate_semantics": (
                "a repeated year within one draw reuses that draw's single meeting-level replicate resample"
            ),
            "reference_resampled_within_meeting": False,
            "bootstrap_interval": "95% linear percentile",
            "undefined_correlation_draw_policy": "reject_and_count_without_replacement",
            "positive_gain_formulas": {
                "correlations": "metric_chk3_minus_metric_baseline",
                "signed_bias": "abs(bias_baseline)_minus_abs(bias_chk3)",
                "mae_rmse": "metric_baseline_minus_metric_chk3",
            },
            "multiplicity_adjustment": None,
            "inference_role": "exploratory paired-bootstrap uncertainty",
        },
        "coverage": coverage,
        "reference_scales": copy.deepcopy(dict(diagnostics["reference_scales"])),
        "view_specs": copy.deepcopy(list(diagnostics["view_specs"])),
        "inputs": {
            "sentiment_suite_manifest": copy.deepcopy(
                inputs["sentiment_manifest_binding"]
            ),
            "official_release_ledger": copy.deepcopy(inputs["release_ledger_binding"]),
            "market_manifest_seal_anchor": copy.deepcopy(
                inputs["market_manifest_binding"]
            ),
            "generation_suite_manifest": copy.deepcopy(dict(generation_binding)),
        },
        "implementation_sources": _implementation_source_bindings(),
        "artifacts": artifacts,
        "reporting": {
            "headline_estimate_row_ids": _headline_estimate_ids(),
            "headline_contrast_row_ids": _headline_contrast_ids(),
            "meeting_aggregate_scatter_source": (
                "primary backend/view/policy rows pivoted by meeting_id and arm"
            ),
            "arm_display_alias": "CHK2 (artifact CHK3/cp318)",
        },
        "limitations": [
            "The reference is a deterministic synthetic teacher target, not official FOMC Minutes or ground truth.",
            "Fixed classifier outputs can reflect calibration, support, and domain shift rather than textual quality.",
            "Pearson and Spearman measure association, not agreement; bias, MAE, and RMSE measure score-space distance only.",
            "The shared-complete policy has n_i=5 for 250 and n_i=4 for six full-panel meetings; the primary pre_external view has 124 and four, respectively, so within-meeting uncertainty remains coarse.",
            "CHK3/cp318 is post-selected; bootstrap uncertainty does not remove checkpoint-selection bias.",
            "Only pre_external DistilBERT shared-complete reference-SD results are primary; other views, policies, scales, and FinBERT are descriptive robustness analyses.",
            "No result is a causal market-impact estimate or a general model-equivalence claim.",
        ],
        "publication": {
            "method": "staging_fsync_then_renameat2_RENAME_NOREPLACE",
            "create_only": True,
            "failure_staging_preserved": True,
            "final_root": str(final_root),
            "publish_attempt_id": publish_attempt_id,
            "staging_basename": staging_basename,
            "file_mode": "0444",
            "directory_mode": "0555",
            "gpu_work": False,
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "required_environment": _require_runtime_environment(),
            "gpu_work": False,
        },
    }


class _JsonlWriter:
    def __init__(self, path: Path, *, fsync_every: int = 1_000) -> None:
        if os.path.lexists(path):
            raise ClosenessEstimatorError(f"refusing overwrite: {path}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        self.path = path
        self.handle: TextIO = os.fdopen(descriptor, "w", encoding="utf-8")
        self.rows = 0
        self.fsync_every = fsync_every

    def write(self, value: Mapping[str, Any]) -> None:
        self.handle.write(_canonical(dict(value)) + "\n")
        self.rows += 1
        if self.rows % self.fsync_every == 0:
            self.handle.flush()
            os.fsync(self.handle.fileno())

    def close(self) -> None:
        if self.handle.closed:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.path.chmod(0o444)

    def __enter__(self) -> _JsonlWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise ClosenessEstimatorError(f"refusing overwrite: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(_canonical(dict(value)) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise ClosenessEstimatorError("Linux renameat2 is required")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(target),
        RENAME_NOREPLACE,
    )
    if result != 0:
        observed = ctypes.get_errno()
        if observed == errno.EEXIST:
            raise ClosenessEstimatorError(f"output already exists: {target}")
        raise ClosenessEstimatorError(
            f"renameat2 failed: errno={observed} {os.strerror(observed)}"
        )
    _fsync_directory(target.parent)


def _expected_counts(draws: int) -> dict[str, int]:
    analyses = len(BACKEND_ORDER) * len(VIEW_ORDER) * len(SCORE_POLICY_ORDER)
    return {
        "analyses": analyses,
        "meetings": len(BACKEND_ORDER) * len(SCORE_POLICY_ORDER) * 256 * len(ARM_ORDER),
        "estimates": analyses
        * len(SCALE_ORDER)
        * len(SYNTHETIC_ARMS)
        * len(METRIC_ORDER),
        "contrasts": analyses
        * len(SCALE_ORDER)
        * len(CONTRAST_ORDER)
        * len(METRIC_ORDER),
        "draws": analyses * draws,
    }


def estimate(
    *,
    sentiment_suite_manifest: Path = DEFAULT_SENTIMENT_SUITE,
    release_ledger: Path = DEFAULT_RELEASE_LEDGER,
    output_dir: Path = DEFAULT_OUTPUT,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    _require_runtime_environment()
    if (
        isinstance(bootstrap_draws, bool)
        or not isinstance(bootstrap_draws, int)
        or bootstrap_draws < 1
        or isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ClosenessEstimatorError("bootstrap draws/seed are invalid")
    unresolved_output = output_dir.expanduser()
    if unresolved_output.is_symlink() or os.path.lexists(unresolved_output):
        raise ClosenessEstimatorError("output must be a fresh create-only root")
    final_root = unresolved_output.resolve()
    if final_root == DEFAULT_OUTPUT.resolve() and (
        bootstrap_draws != BOOTSTRAP_DRAWS or bootstrap_seed != BOOTSTRAP_SEED
    ):
        raise ClosenessEstimatorError(
            "the formal default output requires exactly B=10000 and seed=20260817"
        )
    inputs = _load_inputs(
        sentiment_suite_manifest=sentiment_suite_manifest,
        release_ledger=release_ledger,
    )
    release_rows = _ordered_release_rows(inputs["release_rows"])
    scores_by_backend = {
        backend_id: _backend_scores(inputs["sentiment"]["backends"][backend_id])
        for backend_id in BACKEND_ORDER
    }
    for scores in scores_by_backend.values():
        _validate_release_score_alignment(release_rows, scores)
    views = _views(release_rows)
    scales = {
        backend_id: _reference_scale(scores_by_backend[backend_id], release_rows)
        for backend_id in BACKEND_ORDER
    }
    meeting_aggregates = _meeting_aggregate_rows(
        release_rows=release_rows,
        scores_by_backend=scores_by_backend,
        scales=scales,
    )
    expected = _expected_counts(bootstrap_draws)
    if len(meeting_aggregates) != expected["meetings"]:
        raise ClosenessEstimatorError("meeting aggregate count drift")

    final_root.parent.mkdir(parents=True, exist_ok=True)
    attempt = secrets.token_hex(16)
    staging = final_root.with_name(f"{final_root.name}.staging.{os.getpid()}.{attempt}")
    if os.path.lexists(staging):
        raise ClosenessEstimatorError("unexpected staging collision")
    staging.mkdir(mode=0o700)
    _fsync_directory(staging.parent)

    estimates: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    plan_diagnostics: list[dict[str, Any]] = []
    with _JsonlWriter(staging / MEETING_FILENAME) as writer:
        for row in meeting_aggregates:
            writer.write(row)
    draw_count = 0
    with _JsonlWriter(staging / DRAWS_FILENAME) as draw_writer:
        for backend_id in BACKEND_ORDER:
            scores = scores_by_backend[backend_id]
            scale = scales[backend_id]
            for view_id in VIEW_ORDER:
                view = views[view_id]
                for score_policy in SCORE_POLICY_ORDER:
                    result = _run_analysis(
                        view=view,
                        scores=scores,
                        scale=scale,
                        score_policy=score_policy,
                        draws=bootstrap_draws,
                        seed=bootstrap_seed,
                        consume_draw=draw_writer.write,
                    )
                    draw_count += result.draw_rows
                    estimates.extend(result.estimates)
                    contrasts.extend(result.contrasts)
                    plan_diagnostics.append(result.plan_diagnostic)
    if (
        draw_count != expected["draws"]
        or len(estimates) != expected["estimates"]
        or len(contrasts) != expected["contrasts"]
        or len(plan_diagnostics) != expected["analyses"]
    ):
        raise ClosenessEstimatorError("output matrix closure drift")
    with _JsonlWriter(staging / ESTIMATES_FILENAME) as writer:
        for row in estimates:
            writer.write(row)
    with _JsonlWriter(staging / CONTRASTS_FILENAME) as writer:
        for row in contrasts:
            writer.write(row)
    diagnostics = _diagnostics_payload(
        views=views,
        scores_by_backend=scores_by_backend,
        scales=scales,
        plan_diagnostics=plan_diagnostics,
        draws=bootstrap_draws,
        estimate_rows=len(estimates),
        contrast_rows=len(contrasts),
        draw_rows=draw_count,
        meeting_rows=len(meeting_aggregates),
    )
    _write_new_json(staging / DIAGNOSTICS_FILENAME, diagnostics)
    manifest = seal_manifest(
        _manifest_payload(
            created_at_utc=_utc_now(),
            final_root=final_root,
            staging_root=staging,
            inputs=inputs,
            diagnostics=diagnostics,
            meeting_rows=len(meeting_aggregates),
            estimate_rows=len(estimates),
            contrast_rows=len(contrasts),
            draw_rows=draw_count,
            draws=bootstrap_draws,
            seed=bootstrap_seed,
            publish_attempt_id=attempt,
            staging_basename=staging.name,
        )
    )
    _write_new_json(staging / "manifest.json", manifest)
    staging.chmod(0o555)
    _fsync_directory(staging)
    _fsync_directory(staging.parent)
    _rename_noreplace(staging, final_root)
    return load_and_validate_closeness(final_root / "manifest.json")["manifest"]


def _validate_artifact_binding(
    binding: Mapping[str, Any], *, root: Path, filename: str, rows: int | None
) -> Path:
    path = root / filename
    observed = _binding(path, rows=rows)
    if dict(binding) != observed:
        raise ClosenessEstimatorError(f"artifact binding drift: {filename}")
    if stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise ClosenessEstimatorError(f"artifact mode drift: {filename}")
    return path


def _validate_row_hash(
    row: Mapping[str, Any], *, hash_field: str, row_label: str
) -> None:
    observed = row.get(hash_field)
    if not isinstance(observed, str) or len(observed) != 64:
        raise ClosenessEstimatorError(f"{row_label} hash missing")
    payload = dict(row)
    del payload[hash_field]
    if _record_sha256(payload) != observed:
        raise ClosenessEstimatorError(f"{row_label} hash drift")


def _validate_small_rows(
    *,
    meetings: Sequence[Mapping[str, Any]],
    estimates: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
) -> None:
    for index, row in enumerate(meetings):
        if row.get("schema_version") != MEETING_SCHEMA:
            raise ClosenessEstimatorError(f"meeting schema drift: {index}")
        _validate_row_hash(row, hash_field="row_sha256", row_label=f"meeting:{index}")
    for index, row in enumerate(estimates):
        if row.get("schema_version") != ESTIMATE_SCHEMA:
            raise ClosenessEstimatorError(f"estimate schema drift: {index}")
        _validate_row_hash(
            row, hash_field="estimate_row_sha256", row_label=f"estimate:{index}"
        )
    for index, row in enumerate(contrasts):
        if row.get("schema_version") != CONTRAST_SCHEMA:
            raise ClosenessEstimatorError(f"contrast schema drift: {index}")
        _validate_row_hash(
            row, hash_field="contrast_row_sha256", row_label=f"contrast:{index}"
        )


def _report_model(
    *,
    manifest: Mapping[str, Any],
    manifest_binding: Mapping[str, Any],
    meeting_aggregates: Sequence[Mapping[str, Any]],
    estimates: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    selectors = manifest["primary_selectors"]
    headline_estimate_ids = set(manifest["reporting"]["headline_estimate_row_ids"])
    headline_contrast_ids = set(manifest["reporting"]["headline_contrast_row_ids"])
    primary_rows = [
        row
        for row in meeting_aggregates
        if row["backend_id"] == selectors["backend_id"]
        and row["score_policy"] == selectors["score_policy"]
        and row["era"] == "pre_external"
    ]
    by_meeting: dict[str, dict[str, Any]] = {}
    for row in primary_rows:
        value = by_meeting.setdefault(
            str(row["meeting_id"]),
            {
                "meeting_id": row["meeting_id"],
                "meeting_date": row["meeting_date"],
                "release_date": row["release_date"],
            },
        )
        value[str(row["arm"])] = {
            "raw_score": row["raw_score"],
            "pre_external_reference_sd_score": row["pre_external_reference_sd_score"],
        }
    scatter_rows = [by_meeting[key] for key in sorted(by_meeting)]
    if len(scatter_rows) != 128 or any(
        set(row)
        != {
            "meeting_id",
            "meeting_date",
            "release_date",
            *ARM_ORDER,
        }
        for row in scatter_rows
    ):
        raise ClosenessEstimatorError("primary scatter-row closure drift")
    return {
        "schema_version": REPORT_MODEL_SCHEMA,
        "source_manifest_binding": copy.deepcopy(dict(manifest_binding)),
        "source_bindings": {
            "manifest": copy.deepcopy(dict(manifest_binding)),
            **copy.deepcopy(dict(manifest["artifacts"])),
            **copy.deepcopy(dict(manifest["inputs"])),
        },
        "primary_backend_id": selectors["backend_id"],
        "primary_view_id": selectors["view_id"],
        "primary_score_policy": selectors["score_policy"],
        "primary_scale_id": selectors["scale_id"],
        "primary_artifact_arm": selectors["arm"],
        "primary_paper_stage_id": selectors["paper_stage_id"],
        "arm_display_alias": manifest["reporting"]["arm_display_alias"],
        "methodology": copy.deepcopy(dict(manifest["analysis_contract"])),
        "limitations": copy.deepcopy(list(manifest["limitations"])),
        "coverage": copy.deepcopy(dict(manifest["coverage"])),
        "reference_scales": copy.deepcopy(dict(manifest["reference_scales"])),
        "view_specs": copy.deepcopy(list(manifest["view_specs"])),
        "headline_estimates": [
            copy.deepcopy(dict(row))
            for row in estimates
            if row["estimate_id"] in headline_estimate_ids
        ],
        "headline_contrasts": [
            copy.deepcopy(dict(row))
            for row in contrasts
            if row["contrast_row_id"] in headline_contrast_ids
        ],
        "scatter_rows": scatter_rows,
        "meeting_aggregates": [copy.deepcopy(dict(row)) for row in meeting_aggregates],
        "estimates": [copy.deepcopy(dict(row)) for row in estimates],
        "contrasts": [copy.deepcopy(dict(row)) for row in contrasts],
        "diagnostics": copy.deepcopy(dict(diagnostics)),
    }


def load_and_validate_closeness(manifest_path: Path) -> dict[str, Any]:
    _require_runtime_environment()
    unresolved = manifest_path.expanduser()
    if (
        unresolved.is_symlink()
        or unresolved.parent.is_symlink()
        or not unresolved.is_file()
    ):
        raise ClosenessEstimatorError("manifest missing/symlink")
    manifest_path = unresolved.resolve()
    root = manifest_path.parent
    if manifest_path != root / "manifest.json":
        raise ClosenessEstimatorError("manifest filename drift")
    if (
        stat.S_IMODE(root.stat().st_mode) != 0o555
        or stat.S_IMODE(manifest_path.stat().st_mode) != 0o444
    ):
        raise ClosenessEstimatorError("output modes are not sealed")
    expected_files = {
        "manifest.json",
        MEETING_FILENAME,
        ESTIMATES_FILENAME,
        CONTRASTS_FILENAME,
        DRAWS_FILENAME,
        DIAGNOSTICS_FILENAME,
    }
    if {path.name for path in root.iterdir()} != expected_files or any(
        path.is_symlink() for path in root.iterdir()
    ):
        raise ClosenessEstimatorError("sealed file inventory drift")
    manifest = _read_json(manifest_path)
    payload_sha = validate_manifest_integrity(manifest)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("evaluation_id") != EVALUATION_ID
        or manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or manifest.get("causal_claim_permitted") is not False
        or manifest.get("backend_order") != list(BACKEND_ORDER)
        or manifest.get("arm_order") != list(ARM_ORDER)
        or manifest.get("view_order") != list(VIEW_ORDER)
        or manifest.get("score_policy_order") != list(SCORE_POLICY_ORDER)
        or manifest.get("scale_order") != list(SCALE_ORDER)
        or manifest.get("metric_order") != list(METRIC_ORDER)
        or manifest.get("contrast_order") != list(CONTRAST_ORDER)
        or manifest.get("paper_to_artifact_mapping")
        != {
            "paper_stage_id": "chk2",
            "artifact_arm_id": "chk3",
            "artifact_checkpoint": 318,
            "chk2_artifact_arm_exists": False,
        }
    ):
        raise ClosenessEstimatorError("manifest header drift")
    contract = manifest.get("analysis_contract")
    coverage = manifest.get("coverage")
    artifacts = manifest.get("artifacts")
    input_bindings = manifest.get("inputs")
    publication = manifest.get("publication")
    if not all(
        isinstance(value, Mapping)
        for value in (contract, coverage, artifacts, input_bindings, publication)
    ):
        raise ClosenessEstimatorError("manifest objects missing")
    draws = contract.get("bootstrap_draws")
    seed = contract.get("bootstrap_seed")
    if (
        isinstance(draws, bool)
        or not isinstance(draws, int)
        or draws < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ClosenessEstimatorError("manifest bootstrap contract drift")
    expected = _expected_counts(draws)
    expected_coverage = {
        "backends": len(BACKEND_ORDER),
        "views": len(VIEW_ORDER),
        "score_policies": len(SCORE_POLICY_ORDER),
        "scales": len(SCALE_ORDER),
        "metrics": len(METRIC_ORDER),
        "synthetic_arms": len(SYNTHETIC_ARMS),
        "analysis_cells": expected["analyses"],
        "meeting_aggregate_rows": expected["meetings"],
        "estimate_rows": expected["estimates"],
        "contrast_rows": expected["contrasts"],
        "bootstrap_draw_rows": expected["draws"],
        "bootstrap_draws_per_analysis": draws,
    }
    if dict(coverage) != expected_coverage:
        raise ClosenessEstimatorError("coverage drift")
    if (
        publication.get("method") != "staging_fsync_then_renameat2_RENAME_NOREPLACE"
        or publication.get("create_only") is not True
        or publication.get("failure_staging_preserved") is not True
        or publication.get("final_root") != str(root)
        or publication.get("file_mode") != "0444"
        or publication.get("directory_mode") != "0555"
        or publication.get("gpu_work") is not False
    ):
        raise ClosenessEstimatorError("publication contract drift")
    attempt = publication.get("publish_attempt_id")
    staging_basename = publication.get("staging_basename")
    if (
        not isinstance(attempt, str)
        or len(attempt) != 32
        or any(value not in "0123456789abcdef" for value in attempt)
        or not isinstance(staging_basename, str)
        or not staging_basename.startswith(f"{root.name}.staging.")
        or not staging_basename.endswith(f".{attempt}")
    ):
        raise ClosenessEstimatorError("publication identity drift")
    pid_text = staging_basename.removeprefix(f"{root.name}.staging.").removesuffix(
        f".{attempt}"
    )
    if not pid_text.isdigit() or int(pid_text) <= 0:
        raise ClosenessEstimatorError("publication staging PID drift")

    meeting_path = _validate_artifact_binding(
        artifacts["meeting_aggregates"],
        root=root,
        filename=MEETING_FILENAME,
        rows=expected["meetings"],
    )
    estimate_path = _validate_artifact_binding(
        artifacts["estimates"],
        root=root,
        filename=ESTIMATES_FILENAME,
        rows=expected["estimates"],
    )
    contrast_path = _validate_artifact_binding(
        artifacts["contrasts"],
        root=root,
        filename=CONTRASTS_FILENAME,
        rows=expected["contrasts"],
    )
    draw_path = _validate_artifact_binding(
        artifacts["bootstrap_draws"],
        root=root,
        filename=DRAWS_FILENAME,
        rows=expected["draws"],
    )
    diagnostic_path = _validate_artifact_binding(
        artifacts["diagnostics"],
        root=root,
        filename=DIAGNOSTICS_FILENAME,
        rows=None,
    )
    meeting_aggregates = _read_jsonl(meeting_path)
    estimates = _read_jsonl(estimate_path)
    contrasts = _read_jsonl(contrast_path)
    diagnostics = _read_json(diagnostic_path)
    validate_manifest_integrity(diagnostics)
    if (
        len(meeting_aggregates) != expected["meetings"]
        or len(estimates) != expected["estimates"]
        or len(contrasts) != expected["contrasts"]
    ):
        raise ClosenessEstimatorError("small artifact row count drift")
    _validate_small_rows(
        meetings=meeting_aggregates, estimates=estimates, contrasts=contrasts
    )

    sentiment_binding = input_bindings.get("sentiment_suite_manifest")
    release_binding = input_bindings.get("official_release_ledger")
    market_binding = input_bindings.get("market_manifest_seal_anchor")
    if not all(
        isinstance(value, Mapping)
        for value in (sentiment_binding, release_binding, market_binding)
    ):
        raise ClosenessEstimatorError("input bindings missing")
    inputs = _load_inputs(
        sentiment_suite_manifest=Path(str(sentiment_binding["path"])),
        release_ledger=Path(str(release_binding["path"])),
    )
    if (
        dict(sentiment_binding) != inputs["sentiment_manifest_binding"]
        or dict(release_binding) != inputs["release_ledger_binding"]
        or dict(market_binding) != inputs["market_manifest_binding"]
        or manifest.get("implementation_sources") != _implementation_source_bindings()
    ):
        raise ClosenessEstimatorError("input/implementation binding drift")
    release_rows = _ordered_release_rows(inputs["release_rows"])
    scores_by_backend = {
        backend_id: _backend_scores(inputs["sentiment"]["backends"][backend_id])
        for backend_id in BACKEND_ORDER
    }
    for scores in scores_by_backend.values():
        _validate_release_score_alignment(release_rows, scores)
    views = _views(release_rows)
    scales = {
        backend_id: _reference_scale(scores_by_backend[backend_id], release_rows)
        for backend_id in BACKEND_ORDER
    }
    rebuilt_meetings = _meeting_aggregate_rows(
        release_rows=release_rows,
        scores_by_backend=scores_by_backend,
        scales=scales,
    )
    if meeting_aggregates != rebuilt_meetings:
        raise ClosenessEstimatorError("meeting aggregate reconstruction drift")

    actual_draws = _iter_jsonl(draw_path)
    replayed_draws = 0

    def require_draw(expected_row: dict[str, Any]) -> None:
        nonlocal replayed_draws
        try:
            actual = next(actual_draws)
        except StopIteration as exc:
            raise ClosenessEstimatorError(
                f"bootstrap draw ledger ended early: {replayed_draws}"
            ) from exc
        _validate_row_hash(
            actual,
            hash_field="draw_row_sha256",
            row_label=f"draw:{replayed_draws}",
        )
        if actual != expected_row:
            raise ClosenessEstimatorError(
                f"bootstrap deterministic replay drift: {replayed_draws}"
            )
        replayed_draws += 1

    rebuilt_estimates: list[dict[str, Any]] = []
    rebuilt_contrasts: list[dict[str, Any]] = []
    plan_diagnostics: list[dict[str, Any]] = []
    for backend_id in BACKEND_ORDER:
        for view_id in VIEW_ORDER:
            for score_policy in SCORE_POLICY_ORDER:
                result = _run_analysis(
                    view=views[view_id],
                    scores=scores_by_backend[backend_id],
                    scale=scales[backend_id],
                    score_policy=score_policy,
                    draws=draws,
                    seed=seed,
                    consume_draw=require_draw,
                )
                rebuilt_estimates.extend(result.estimates)
                rebuilt_contrasts.extend(result.contrasts)
                plan_diagnostics.append(result.plan_diagnostic)
    try:
        next(actual_draws)
    except StopIteration:
        pass
    else:
        raise ClosenessEstimatorError("bootstrap draw ledger has extra rows")
    if replayed_draws != expected["draws"]:
        raise ClosenessEstimatorError("bootstrap replay coverage drift")
    if estimates != rebuilt_estimates:
        raise ClosenessEstimatorError("estimate reconstruction drift")
    if contrasts != rebuilt_contrasts:
        raise ClosenessEstimatorError("contrast reconstruction drift")
    rebuilt_diagnostics = _diagnostics_payload(
        views=views,
        scores_by_backend=scores_by_backend,
        scales=scales,
        plan_diagnostics=plan_diagnostics,
        draws=draws,
        estimate_rows=len(estimates),
        contrast_rows=len(contrasts),
        draw_rows=replayed_draws,
        meeting_rows=len(meeting_aggregates),
    )
    if diagnostics != rebuilt_diagnostics:
        raise ClosenessEstimatorError("diagnostics reconstruction drift")
    rebuilt_manifest = seal_manifest(
        _manifest_payload(
            created_at_utc=str(manifest.get("created_at_utc")),
            final_root=root,
            staging_root=root,
            inputs=inputs,
            diagnostics=diagnostics,
            meeting_rows=len(meeting_aggregates),
            estimate_rows=len(estimates),
            contrast_rows=len(contrasts),
            draw_rows=replayed_draws,
            draws=draws,
            seed=seed,
            publish_attempt_id=str(attempt),
            staging_basename=str(staging_basename),
        )
    )
    if manifest != rebuilt_manifest:
        raise ClosenessEstimatorError("manifest reconstruction drift")
    manifest_binding = _binding(manifest_path, payload_sha256=payload_sha)
    report_model = _report_model(
        manifest=manifest,
        manifest_binding=manifest_binding,
        meeting_aggregates=meeting_aggregates,
        estimates=estimates,
        contrasts=contrasts,
        diagnostics=diagnostics,
    )
    return {
        "manifest": manifest,
        "manifest_binding": manifest_binding,
        "meeting_aggregates": meeting_aggregates,
        "estimates": estimates,
        "contrasts": contrasts,
        "bootstrap_draw_rows": replayed_draws,
        "diagnostics": diagnostics,
        "report_model": report_model,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("estimate")
    run.add_argument(
        "--sentiment-suite-manifest", type=Path, default=DEFAULT_SENTIMENT_SUITE
    )
    run.add_argument("--release-ledger", type=Path, default=DEFAULT_RELEASE_LEDGER)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    run.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "estimate":
            manifest = estimate(
                sentiment_suite_manifest=args.sentiment_suite_manifest,
                release_ledger=args.release_ledger,
                output_dir=args.output_dir,
                bootstrap_draws=args.bootstrap_draws,
                bootstrap_seed=args.bootstrap_seed,
            )
            summary = {
                "status": manifest["status"],
                "manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
                "coverage": manifest["coverage"],
            }
        else:
            loaded = load_and_validate_closeness(args.manifest)
            summary = {
                "status": "valid",
                "manifest_binding": loaded["manifest_binding"],
                "coverage": loaded["manifest"]["coverage"],
            }
        print(_canonical(summary))
        return 0
    except (
        ClosenessEstimatorError,
        statistics.BetaStatisticsError,
        OSError,
        ValueError,
    ) as exc:
        print(
            _canonical(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SEED",
    "ClosenessEstimatorError",
    "DEFAULT_OUTPUT",
    "DEFAULT_RELEASE_LEDGER",
    "DEFAULT_SENTIMENT_SUITE",
    "EVALUATION_ID",
    "MANIFEST_SCHEMA",
    "estimate",
    "load_and_validate_closeness",
    "main",
]
