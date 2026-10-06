"""Estimate the CHK3 sentiment--Treasury association with paired resampling.

This is the versioned, CPU-only estimator for the sealed Core8 sentiment and
Minutes-release market panels.  The estimand is deliberately reduced-form and
exploratory: synthetic Minutes were never released at the historical event
time, so no coefficient is a causal market-impact estimate.

The primary view uses pre-external meetings, DGS2 release-minus-previous yield
changes, and the frozen DistilBERT FOMC stance classifier.  FinBERT financial
valence is a non-pooled robustness construct.  The hawk/dove lexicon is kept
only as a zero-reference-variance diagnostic and never enters a regression.
Current and lagged sentiment are scaled once by the complete pre-external
*reference* distribution for each eligible neural backend.  Calendar-year
blocks and within-meeting generation replicates are resampled with one paired
index plan shared by all text arms.  Lags are fixed on the original chronology
before any block resampling.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import json
import math
import os
import secrets
import stat
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if sys.version_info < (3, 11):
    raise RuntimeError("the sealed market-panel loader requires Python 3.11 or newer")

from jobs.eval import build_chk3_beta_minutes_release_market_panel as market
from jobs.eval import chk3_beta_statistics as statistics
from jobs.eval import eval_chk3_beta_core8_sentiment_stochastic_schedule_v1 as sentiment
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
DEFAULT_MARKET_PANEL = (
    RUN_ROOT / "market_panel_fed_minutes_release_daily_v1/manifest.json"
)
DEFAULT_OUTPUT = RUN_ROOT / "beta_estimator_time_block_bootstrap_v1"

EVALUATION_ID = "chk3-beta-core8-econometric-estimator-time-block-bootstrap-v1"
MANIFEST_SCHEMA = (
    "chk3-beta-core8-econometric-estimator-time-block-bootstrap-manifest-v1"
)
ESTIMATE_SCHEMA = "chk3-beta-core8-econometric-estimate-v1"
CONTRAST_SCHEMA = "chk3-beta-core8-econometric-contrast-v1"
DRAW_SCHEMA = "chk3-beta-core8-econometric-bootstrap-draw-v1"
DIAGNOSTICS_SCHEMA = "chk3-beta-core8-econometric-diagnostics-v1"

ESTIMATES_FILENAME = "estimates.v1.jsonl"
CONTRASTS_FILENAME = "contrasts.v1.jsonl"
DRAWS_FILENAME = "bootstrap_draws.v1.jsonl"
DIAGNOSTICS_FILENAME = "diagnostics.v1.json"

ARM_ORDER = sentiment.ARM_ORDER
SYNTHETIC_ARMS = statistics.SYNTHETIC_ARMS
REGRESSION_BACKEND_ORDER = (sentiment.DISTIL_BACKEND, sentiment.FINBERT_BACKEND)
VIEW_ORDER = (
    "pre_external",
    "full",
    "post",
    "full_excluding_cp318_selection",
    "post_excluding_cp318_selection",
)
PRIMARY_VIEW = "pre_external"
PRIMARY_BACKEND = sentiment.DISTIL_BACKEND
OUTCOME_ID = "dgs2_release_minus_previous_bp"
VIX_CONTROL_ID = "vix_release_minus_previous_log_pct"
SCALE_ORDER = ("pre_external_reference_sd", "raw_score")
PRIMARY_SCALE = SCALE_ORDER[0]
SCORE_POLICY_ORDER = (
    "shared_complete_core8",
    "neutral_imputed_fixed_k5",
)
PRIMARY_SCORE_POLICY = SCORE_POLICY_ORDER[0]
CONTRASTS = {
    "chk3_minus_chk1": ("chk1", "chk3", "primary"),
    "chk1_minus_chk0": ("chk0", "chk1", "background"),
    "chk3_minus_chk0": ("chk0", "chk3", "background"),
    "chk0_minus_reference": ("reference", "chk0", "reference_context"),
    "chk1_minus_reference": ("reference", "chk1", "reference_context"),
    "chk3_minus_reference": ("reference", "chk3", "reference_context"),
}
DERIVED_CONTRAST_ID = "reference_distance_gain_chk3_vs_chk1"
EXPECTED_MARKET_ROWS = 255
EXPECTED_VIEW_NOBS = {
    "pre_external": 126,
    "full": 254,
    "post": 127,
    "full_excluding_cp318_selection": 243,
    "post_excluding_cp318_selection": 116,
}
EXPECTED_REFERENCE_SCALE_ROWS = 128
BOOTSTRAP_DRAWS = statistics.BOOTSTRAP_DRAWS
BOOTSTRAP_SEED = statistics.BOOTSTRAP_SEED
HAC_LAG = statistics.DEFAULT_HAC_LAG
REPLICATES_PER_DRAW = len(sentiment.preparation.REPLICATE_SEEDS)
NORMAL_95 = 1.959963984540054
AT_FDCWD = -100
RENAME_NOREPLACE = 1

BACKEND_LABELS = {
    sentiment.LEXICON_BACKEND: "Lucca–Trebbi-inspired hawk/dove lexicon",
    sentiment.DISTIL_BACKEND: "DistilBERT FOMC stance",
    sentiment.FINBERT_BACKEND: "ProsusAI FinBERT financial valence",
}
OUTCOME_LABELS = {
    "dgs2_release_minus_previous_bp": "DGS2 release close minus previous close (bp)",
    "dgs5_release_minus_previous_bp": "DGS5 release close minus previous close (bp)",
    "dgs10_release_minus_previous_bp": "DGS10 release close minus previous close (bp)",
    "dgs2_next_minus_release_bp": "DGS2 next close minus release close (bp)",
    "dgs2_placebo_previous_minus_previous_2_bp": (
        "DGS2 previous close minus previous-2 close placebo (bp)"
    ),
}
OUTCOME_VIX_CONTROL = {
    "dgs2_release_minus_previous_bp": VIX_CONTROL_ID,
    "dgs5_release_minus_previous_bp": VIX_CONTROL_ID,
    "dgs10_release_minus_previous_bp": VIX_CONTROL_ID,
    "dgs2_next_minus_release_bp": "vix_next_minus_release_log_pct",
    "dgs2_placebo_previous_minus_previous_2_bp": (
        "vix_placebo_previous_minus_previous_2_log_pct"
    ),
}
VIEW_LABELS = {
    "pre_external": "Pre-2009 external meetings (primary)",
    "full": "Full 1993–2025 panel (descriptive)",
    "post": "Post-2008 panel (descriptive)",
    "full_excluding_cp318_selection": "Full panel excluding cp318-selection exposure",
    "post_excluding_cp318_selection": "Post panel excluding cp318-selection exposure",
}


class BetaEstimatorError(RuntimeError):
    """The estimator input, numerical, or publication contract failed closed."""


@dataclass(frozen=True)
class ReferenceScale:
    backend_id: str
    mean: float
    standard_deviation: float
    rows: int
    ddof: int
    meeting_ids: tuple[str, ...]


@dataclass(frozen=True)
class DesignView:
    view_id: str
    outcome_id: str
    vix_control_id: str
    lag_meeting_ids: tuple[str, ...]
    current_meeting_ids: tuple[str, ...]
    union_meeting_ids: tuple[str, ...]
    union_block_ids: tuple[str, ...]
    block_ids: tuple[str, ...]
    design_block_ids: tuple[str, ...]
    outcome: np.ndarray
    vix: np.ndarray
    months: np.ndarray
    cp318_selection_exposed: tuple[bool, ...]
    meeting_date_start: str
    meeting_date_end: str
    release_date_start: str
    release_date_end: str

    @property
    def nobs(self) -> int:
        return len(self.current_meeting_ids)


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
        raise BetaEstimatorError(f"non-canonical payload: {exc}") from exc


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BetaEstimatorError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BetaEstimatorError(f"{name} must be finite")
    return result


def _binding(
    path: Path, *, rows: int | None = None, payload_sha256: str | None = None
) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise BetaEstimatorError(f"bound path is a symlink: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise BetaEstimatorError(f"bound file is missing: {resolved}")
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
    "market_loader": _binding(Path(market.__file__).resolve()),
}
_REGRESSION_SUITE_IMPORT_BINDING: dict[str, Any] | None = None


def _require_import_binding_unchanged(
    source_id: str, import_binding: Mapping[str, Any]
) -> dict[str, Any]:
    current = _binding(Path(str(import_binding.get("path"))))
    if dict(import_binding) != current:
        raise BetaEstimatorError(
            f"implementation source changed after import: {source_id}"
        )
    return current


def _implementation_source_bindings() -> dict[str, Any]:
    if _REGRESSION_SUITE_IMPORT_BINDING is None:
        raise BetaEstimatorError("regression-suite-v2 loader import was not captured")
    frozen = {
        **_IMPLEMENTATION_IMPORT_BINDINGS,
        "sentiment_regression_suite_loader": _REGRESSION_SUITE_IMPORT_BINDING,
    }
    return {
        source_id: _require_import_binding_unchanged(source_id, binding)
        for source_id, binding in frozen.items()
    }


def _record_sha256(value: Mapping[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(_canonical(dict(value)).encode("utf-8")).hexdigest()


def _require_finite_array(name: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise BetaEstimatorError(f"{name} is not a finite non-empty vector")
    return array


def _load_inputs(
    *, sentiment_suite_manifest: Path, market_panel_manifest: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    global _REGRESSION_SUITE_IMPORT_BINDING
    try:
        from jobs.eval import seal_chk3_beta_sentiment_regression_suite_v2

        suite_binding = _binding(
            Path(seal_chk3_beta_sentiment_regression_suite_v2.__file__).resolve()
        )
        if _REGRESSION_SUITE_IMPORT_BINDING is None:
            _REGRESSION_SUITE_IMPORT_BINDING = suite_binding
        elif _REGRESSION_SUITE_IMPORT_BINDING != suite_binding:
            raise BetaEstimatorError(
                "regression-suite-v2 loader changed after first import"
            )
        sentiment_loaded = seal_chk3_beta_sentiment_regression_suite_v2.load_and_validate_regression_suite(
            sentiment_suite_manifest
        )
        market_loaded = market.validate_panel(market_panel_manifest)
    except Exception as exc:
        raise BetaEstimatorError(str(exc)) from exc
    return sentiment_loaded, market_loaded


def _backend_scores(loaded: Mapping[str, Any]) -> BackendScores:
    manifest = loaded["manifest"]
    backend_id = str(manifest["backend"])
    rows = loaded["meeting_scores"]
    reference: dict[str, float] = {}
    synthetic: dict[str, dict[str, dict[int, float]]] = {
        arm: defaultdict(dict) for arm in SYNTHETIC_ARMS
    }
    neutral_reference: dict[str, float] = {}
    neutral_synthetic: dict[str, dict[str, dict[int, float]]] = {
        arm: defaultdict(dict) for arm in SYNTHETIC_ARMS
    }
    exposure: dict[str, bool] = {}
    construct = str(manifest["construct"])
    for row in rows:
        meeting_id = str(row["meeting_id"])
        row_exposure = row.get("cp318_selection_exposed")
        if not isinstance(row_exposure, bool):
            raise BetaEstimatorError("sentiment exposure flag is not boolean")
        if meeting_id in exposure and exposure[meeting_id] is not row_exposure:
            raise BetaEstimatorError(f"sentiment exposure drift: {meeting_id}")
        exposure[meeting_id] = row_exposure
        arm = str(row["arm"])
        score = row.get("score")
        neutral_score = _finite(
            f"{backend_id}:{meeting_id}:{arm}:neutral_imputed_score",
            row.get("neutral_imputed_score"),
        )
        replicate_id = row.get("replicate_id")
        if arm == "reference":
            if replicate_id is not None or meeting_id in neutral_reference:
                raise BetaEstimatorError("neutral reference sentiment identity drift")
            neutral_reference[meeting_id] = neutral_score
        elif (
            arm not in SYNTHETIC_ARMS
            or isinstance(replicate_id, bool)
            or not isinstance(replicate_id, int)
            or replicate_id not in range(REPLICATES_PER_DRAW)
            or replicate_id in neutral_synthetic[arm][meeting_id]
        ):
            raise BetaEstimatorError("neutral synthetic sentiment identity drift")
        else:
            neutral_synthetic[arm][meeting_id][replicate_id] = neutral_score
        if row.get("complete_core8") is not True or score is None:
            continue
        numeric = _finite(f"{backend_id}:{meeting_id}:{arm}:score", score)
        if arm == "reference":
            if replicate_id is not None or meeting_id in reference:
                raise BetaEstimatorError("reference sentiment identity drift")
            reference[meeting_id] = numeric
            continue
        if (
            arm not in SYNTHETIC_ARMS
            or isinstance(replicate_id, bool)
            or not isinstance(replicate_id, int)
            or replicate_id not in range(REPLICATES_PER_DRAW)
            or replicate_id in synthetic[arm][meeting_id]
        ):
            raise BetaEstimatorError("synthetic sentiment identity drift")
        synthetic[arm][meeting_id][replicate_id] = numeric
    paired: dict[str, tuple[int, ...]] = {}
    all_meetings = sorted(exposure)
    for meeting_id in all_meetings:
        if meeting_id not in neutral_reference or any(
            set(neutral_synthetic[arm].get(meeting_id, {}))
            != set(range(REPLICATES_PER_DRAW))
            for arm in SYNTHETIC_ARMS
        ):
            raise BetaEstimatorError(
                f"neutral-imputation K5 closure drift: {backend_id}:{meeting_id}"
            )
        if meeting_id not in reference:
            raise BetaEstimatorError(
                f"complete reference score missing: {backend_id}:{meeting_id}"
            )
        shared = set(range(REPLICATES_PER_DRAW))
        for arm in SYNTHETIC_ARMS:
            shared &= set(synthetic[arm].get(meeting_id, {}))
        if not shared:
            raise BetaEstimatorError(
                f"no shared complete replicate: {backend_id}:{meeting_id}"
            )
        paired[meeting_id] = tuple(sorted(shared))
    if len(all_meetings) != sentiment.data_contract.EXPECTED_MEETINGS:
        raise BetaEstimatorError("sentiment meeting coverage is not N=256")
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


def _reference_scale(
    scores: BackendScores, meeting_rows: Sequence[Mapping[str, Any]]
) -> ReferenceScale:
    pre_ids = tuple(
        sorted(
            {
                str(row["meeting_id"])
                for row in meeting_rows
                if row.get("arm") == "reference"
                and row.get("role") == "external_holdout"
                and row.get("complete_core8") is True
                and row.get("score") is not None
            }
        )
    )
    if len(pre_ids) != EXPECTED_REFERENCE_SCALE_ROWS:
        raise BetaEstimatorError(
            "reference scale is not all 128 complete pre-external meetings"
        )
    values = _require_finite_array(
        "pre-external reference scale",
        [scores.reference_by_meeting[meeting_id] for meeting_id in pre_ids],
    )
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise BetaEstimatorError("pre-external reference standard deviation is <= 0")
    return ReferenceScale(
        backend_id=scores.backend_id,
        mean=mean,
        standard_deviation=standard_deviation,
        rows=len(values),
        ddof=1,
        meeting_ids=pre_ids,
    )


def _ordered_market_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != EXPECTED_MARKET_ROWS:
        raise BetaEstimatorError("market panel is not N=255")
    ordered = sorted((dict(row) for row in rows), key=lambda row: row["release_date"])
    if len({row["generation_meeting_id"] for row in ordered}) != len(ordered):
        raise BetaEstimatorError("market generation meeting IDs are not unique")
    if len({row["release_date"] for row in ordered}) != len(ordered):
        raise BetaEstimatorError("market release dates are not unique")
    if any(row.get("in_full_sample") is not True for row in ordered):
        raise BetaEstimatorError("market full-sample flag drift")
    return ordered


def _base_view_rows(
    market_rows: Sequence[Mapping[str, Any]], view_id: str
) -> list[Mapping[str, Any]]:
    if view_id in {"full", "full_excluding_cp318_selection"}:
        return list(market_rows)
    if view_id == "pre_external":
        return [row for row in market_rows if row.get("in_pre_sample") is True]
    if view_id in {"post", "post_excluding_cp318_selection"}:
        return [row for row in market_rows if row.get("in_post_sample") is True]
    raise BetaEstimatorError(f"unknown view: {view_id}")


def _base_release_rows(
    release_rows: Sequence[Mapping[str, Any]], view_id: str
) -> list[Mapping[str, Any]]:
    ordered = sorted(release_rows, key=lambda row: row["meeting_end_date"])
    if len(ordered) != sentiment.data_contract.EXPECTED_MEETINGS:
        raise BetaEstimatorError("official release ledger is not N=256")
    if view_id in {"full", "full_excluding_cp318_selection"}:
        return ordered
    if view_id == "pre_external":
        return [row for row in ordered if row.get("era") == "pre_external"]
    if view_id in {"post", "post_excluding_cp318_selection"}:
        return [row for row in ordered if row.get("era") == "post"]
    raise BetaEstimatorError(f"unknown view: {view_id}")


def _make_design_view(
    *,
    market_rows: Sequence[Mapping[str, Any]],
    release_rows: Sequence[Mapping[str, Any]],
    view_id: str,
    outcome_id: str = OUTCOME_ID,
) -> DesignView:
    vix_control_id = OUTCOME_VIX_CONTROL.get(outcome_id)
    if vix_control_id is None:
        raise BetaEstimatorError(f"outcome has no frozen VIX control: {outcome_id}")
    current_rows = _base_view_rows(market_rows, view_id)
    chronology = _base_release_rows(release_rows, view_id)
    previous_by_meeting = {
        str(current["generation_meeting_id"]): previous
        for previous, current in zip(chronology[:-1], chronology[1:], strict=True)
    }
    # The lag is the immediately prior FOMC meeting in the sealed N=256
    # official-release chronology.  It is intentionally not the prior
    # *market-complete* row: 2004-09-21 lacks a release-day Treasury close but
    # remains the sentiment lag for 2004-11-10.
    pairs = [
        (previous_by_meeting[str(current["generation_meeting_id"])], current)
        for current in current_rows
        if str(current["generation_meeting_id"]) in previous_by_meeting
    ]
    if view_id.endswith("excluding_cp318_selection"):
        pairs = [
            (lag, current)
            for lag, current in pairs
            if lag.get("cp318_selection_exposed") is False
            and current.get("cp318_selection_exposed") is False
        ]
    expected_nobs = EXPECTED_VIEW_NOBS[view_id]
    if len(pairs) != expected_nobs:
        raise BetaEstimatorError(
            f"{view_id} design N drift: {len(pairs)} != {expected_nobs}"
        )
    lag_ids = tuple(str(lag["generation_meeting_id"]) for lag, _ in pairs)
    current_ids = tuple(str(current["generation_meeting_id"]) for _, current in pairs)
    union_ids = tuple(dict.fromkeys([*lag_ids, *current_ids]))
    release_year_by_meeting = {
        str(row["generation_meeting_id"]): str(row["release_date"])[:4]
        for row in chronology
    }
    union_blocks = tuple(release_year_by_meeting[value] for value in union_ids)
    design_blocks = tuple(str(current["release_date"])[:4] for _, current in pairs)
    block_ids = tuple(dict.fromkeys(design_blocks))
    months = np.asarray(
        [int(str(current["release_date"])[5:7]) for _, current in pairs],
        dtype=np.int16,
    )
    month_levels = tuple(sorted(int(value) for value in np.unique(months)))
    if len(month_levels) < 2:
        raise BetaEstimatorError(f"{view_id} has fewer than two release months")
    outcome = _require_finite_array(
        f"{view_id}/{outcome_id} outcome",
        [current[outcome_id] for _, current in pairs],
    )
    vix = _require_finite_array(
        f"{view_id}/{outcome_id} VIX control",
        [current[vix_control_id] for _, current in pairs],
    )
    return DesignView(
        view_id=view_id,
        outcome_id=outcome_id,
        vix_control_id=vix_control_id,
        lag_meeting_ids=lag_ids,
        current_meeting_ids=current_ids,
        union_meeting_ids=union_ids,
        union_block_ids=union_blocks,
        block_ids=block_ids,
        design_block_ids=design_blocks,
        outcome=outcome,
        vix=vix,
        months=months,
        cp318_selection_exposed=tuple(
            bool(current["cp318_selection_exposed"]) for _, current in pairs
        ),
        meeting_date_start=str(pairs[0][1]["meeting_end_date"]),
        meeting_date_end=str(pairs[-1][1]["meeting_end_date"]),
        release_date_start=str(pairs[0][1]["release_date"]),
        release_date_end=str(pairs[-1][1]["release_date"]),
    )


def _validate_view_against_scores(view: DesignView, scores: BackendScores) -> None:
    for meeting_id in view.union_meeting_ids:
        if (
            meeting_id not in scores.reference_by_meeting
            or meeting_id not in scores.paired_replicates_by_meeting
        ):
            raise BetaEstimatorError(
                f"sentiment/market meeting mismatch: {scores.backend_id}:{meeting_id}"
            )
    if view.view_id.endswith("excluding_cp318_selection") and any(
        scores.exposure_by_meeting[meeting_id]
        for meeting_id in [*view.lag_meeting_ids, *view.current_meeting_ids]
    ):
        raise BetaEstimatorError("selection-exposure view retained an exposed meeting")


def _score_vectors(
    view: DesignView,
    scores: BackendScores,
    *,
    score_policy: str,
    synthetic_means: Mapping[str, np.ndarray] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    union_index = {
        meeting_id: index for index, meeting_id in enumerate(view.union_meeting_ids)
    }
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if score_policy not in SCORE_POLICY_ORDER:
        raise BetaEstimatorError(f"unknown score policy: {score_policy}")
    neutral = score_policy == "neutral_imputed_fixed_k5"
    reference_source = (
        scores.neutral_reference_by_meeting if neutral else scores.reference_by_meeting
    )
    synthetic_source = (
        scores.neutral_synthetic_by_meeting if neutral else scores.synthetic_by_meeting
    )
    reference = np.asarray(
        [reference_source[value] for value in view.union_meeting_ids], dtype=np.float64
    )
    all_means: dict[str, np.ndarray] = {"reference": reference}
    if synthetic_means is None:
        for arm in SYNTHETIC_ARMS:
            all_means[arm] = np.asarray(
                [
                    np.mean(
                        [
                            synthetic_source[arm][meeting_id][replicate]
                            for replicate in (
                                range(REPLICATES_PER_DRAW)
                                if neutral
                                else scores.paired_replicates_by_meeting[meeting_id]
                            )
                        ]
                    )
                    for meeting_id in view.union_meeting_ids
                ],
                dtype=np.float64,
            )
    else:
        for arm in SYNTHETIC_ARMS:
            all_means[arm] = np.asarray(synthetic_means[arm], dtype=np.float64)
    lag_indexes = np.asarray(
        [union_index[value] for value in view.lag_meeting_ids], dtype=np.int64
    )
    current_indexes = np.asarray(
        [union_index[value] for value in view.current_meeting_ids], dtype=np.int64
    )
    for arm in ARM_ORDER:
        values = all_means[arm]
        if values.shape != (len(view.union_meeting_ids),) or not np.all(
            np.isfinite(values)
        ):
            raise BetaEstimatorError(f"invalid score vector: {scores.backend_id}:{arm}")
        result[arm] = (values[current_indexes], values[lag_indexes])
    return result


def _design_matrix(
    view: DesignView,
    current_score: np.ndarray,
    lag_score: np.ndarray,
    *,
    scale: ReferenceScale,
    row_indexes: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    indexes = (
        np.arange(view.nobs, dtype=np.int64)
        if row_indexes is None
        else np.asarray(row_indexes, dtype=np.int64)
    )
    current = (
        np.asarray(current_score)[indexes] - scale.mean
    ) / scale.standard_deviation
    lagged = (np.asarray(lag_score)[indexes] - scale.mean) / scale.standard_deviation
    columns = [
        np.ones(len(indexes), dtype=np.float64),
        current,
        lagged,
        view.vix[indexes],
    ]
    names = [
        "intercept",
        "current_sentiment_z",
        "lag_sentiment_z",
        view.vix_control_id,
    ]
    phase = 2.0 * np.pi * view.months[indexes].astype(np.float64) / 12.0
    columns.extend((np.sin(phase), np.cos(phase)))
    names.extend(("release_month_sin", "release_month_cos"))
    matrix = np.column_stack(columns)
    if not np.all(np.isfinite(matrix)):
        raise BetaEstimatorError("design matrix contains non-finite values")
    return matrix, tuple(names)


def _analysis_inventory() -> tuple[tuple[str, str, str], ...]:
    """Return (view, outcome, score-policy) in canonical execution order."""

    main = [
        (view_id, OUTCOME_ID, score_policy)
        for view_id in VIEW_ORDER
        for score_policy in SCORE_POLICY_ORDER
    ]
    robustness = [
        (PRIMARY_VIEW, outcome_id, PRIMARY_SCORE_POLICY)
        for outcome_id in (
            "dgs5_release_minus_previous_bp",
            "dgs10_release_minus_previous_bp",
            "dgs2_next_minus_release_bp",
            "dgs2_placebo_previous_minus_previous_2_bp",
        )
    ]
    return tuple([*main, *robustness])


def _analysis_id(view: DesignView, score_policy: str) -> str:
    return f"{view.view_id}__{view.outcome_id}__{score_policy}"


def _specification_id(view: DesignView, score_policy: str, scale_id: str) -> str:
    return (
        f"daily_release_association__{view.outcome_id}__current_lag_sentiment__"
        f"vix__cyclic_month__hac4__{score_policy}__{scale_id}"
    )


def _replicate_inventory(
    view: DesignView, scores: BackendScores, score_policy: str
) -> dict[str, tuple[int, ...]]:
    if score_policy == "neutral_imputed_fixed_k5":
        return {
            meeting_id: tuple(range(REPLICATES_PER_DRAW))
            for meeting_id in view.union_meeting_ids
        }
    if score_policy != PRIMARY_SCORE_POLICY:
        raise BetaEstimatorError(f"unknown score policy: {score_policy}")
    return {
        meeting_id: scores.paired_replicates_by_meeting[meeting_id]
        for meeting_id in view.union_meeting_ids
    }


def _synthetic_score_matrices(
    view: DesignView, scores: BackendScores, score_policy: str
) -> dict[str, np.ndarray]:
    source = (
        scores.neutral_synthetic_by_meeting
        if score_policy == "neutral_imputed_fixed_k5"
        else scores.synthetic_by_meeting
    )
    result: dict[str, np.ndarray] = {}
    for arm in SYNTHETIC_ARMS:
        values = np.full(
            (len(view.union_meeting_ids), REPLICATES_PER_DRAW),
            np.nan,
            dtype=np.float64,
        )
        for meeting_index, meeting_id in enumerate(view.union_meeting_ids):
            for replicate_id, score in source[arm][meeting_id].items():
                values[meeting_index, replicate_id] = score
        result[arm] = values
    return result


def _bootstrap_plan(
    view: DesignView, scores: BackendScores, score_policy: str
) -> statistics.PairedBlockBootstrapPlan:
    inventory = _replicate_inventory(view, scores, score_policy)
    draw_counts = {
        meeting_id: (
            REPLICATES_PER_DRAW
            if score_policy == "neutral_imputed_fixed_k5"
            else len(replicate_ids)
        )
        for meeting_id, replicate_ids in inventory.items()
    }
    try:
        return statistics.make_paired_block_bootstrap_plan(
            block_ids=view.block_ids,
            meeting_ids=view.union_meeting_ids,
            meeting_block_ids=view.union_block_ids,
            replicate_ids_by_meeting=inventory,
            replicate_draw_counts_by_meeting=draw_counts,
            draws=BOOTSTRAP_DRAWS,
            seed=BOOTSTRAP_SEED,
            replicates_per_draw=REPLICATES_PER_DRAW,
        )
    except statistics.BetaStatisticsError as exc:
        raise BetaEstimatorError(str(exc)) from exc


def _sampled_design_row_indexes(
    view: DesignView,
    plan: statistics.PairedBlockBootstrapPlan,
    draw_index: int,
) -> np.ndarray:
    rows_by_block = {
        block_id: np.asarray(
            [
                index
                for index, observed in enumerate(view.design_block_ids)
                if observed == block_id
            ],
            dtype=np.int64,
        )
        for block_id in view.block_ids
    }
    if any(not len(indexes) for indexes in rows_by_block.values()):
        raise BetaEstimatorError("declared design block has no rows")
    return np.concatenate(
        [
            rows_by_block[plan.block_ids[int(block_index)]]
            for block_index in plan.sampled_block_indices[draw_index]
        ]
    )


def _point_fits(
    view: DesignView,
    scores: BackendScores,
    scale: ReferenceScale,
    score_policy: str,
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    vectors = _score_vectors(view, scores, score_policy=score_policy)
    fitted: dict[str, dict[str, Any]] = {}
    column_names: tuple[str, ...] | None = None
    for arm in ARM_ORDER:
        matrix, names = _design_matrix(
            view,
            vectors[arm][0],
            vectors[arm][1],
            scale=scale,
        )
        if column_names is None:
            column_names = names
        elif names != column_names:
            raise BetaEstimatorError("arm design columns drift")
        try:
            result = statistics.ols_hac(view.outcome, matrix, hac_lag=HAC_LAG)
        except statistics.BetaStatisticsError as exc:
            raise BetaEstimatorError(
                f"point fit failed: {scores.backend_id}:{view.view_id}:{arm}:{exc}"
            ) from exc
        fitted[arm] = {
            "standardized": {
                "beta_current": float(result.coefficients[1]),
                "beta_lag": float(result.coefficients[2]),
                "hac_se_current": float(result.standard_errors[1]),
                "hac_se_lag": float(result.standard_errors[2]),
            },
            "raw_score": {
                "beta_current": float(
                    result.coefficients[1] / scale.standard_deviation
                ),
                "beta_lag": float(result.coefficients[2] / scale.standard_deviation),
                "hac_se_current": float(
                    result.standard_errors[1] / scale.standard_deviation
                ),
                "hac_se_lag": float(
                    result.standard_errors[2] / scale.standard_deviation
                ),
            },
            "nobs": result.nobs,
            "rank": result.rank,
            "dof_resid": result.dof_resid,
            "r_squared": float(result.r_squared),
            "adjusted_r_squared": float(result.adjusted_r_squared),
        }
    assert column_names is not None
    return fitted, column_names


def _strict_bootstrap_betas(
    y: np.ndarray, matrices: Mapping[str, np.ndarray]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for arm in ARM_ORDER:
        matrix = matrices[arm]
        if len(y) <= matrix.shape[1]:
            raise statistics.BetaStatisticsError("bootstrap has insufficient rows")
        coefficients, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
        if int(rank) != matrix.shape[1]:
            raise statistics.BetaStatisticsError(
                f"rank-deficient bootstrap design: {arm}:rank={rank}:p={matrix.shape[1]}"
            )
        result[arm] = float(coefficients[1])
    return result


def _contrast_values(betas: Mapping[str, float]) -> dict[str, float]:
    result = {
        contrast_id: float(betas[after] - betas[before])
        for contrast_id, (before, after, _) in CONTRASTS.items()
    }
    result[DERIVED_CONTRAST_ID] = float(
        abs(betas["chk1"] - betas["reference"])
        - abs(betas["chk3"] - betas["reference"])
    )
    return result


def _bootstrap_analysis(
    view: DesignView,
    scores: BackendScores,
    scale: ReferenceScale,
    score_policy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    plan = _bootstrap_plan(view, scores, score_policy)
    matrices_by_arm = _synthetic_score_matrices(view, scores, score_policy)
    fixed_vectors = _score_vectors(view, scores, score_policy=score_policy)
    draw_rows: list[dict[str, Any]] = []
    failures = Counter()
    for draw_index in range(BOOTSTRAP_DRAWS):
        try:
            synthetic_means = statistics.resampled_meeting_means(
                matrices_by_arm, plan, draw_index=draw_index
            )
            vectors = _score_vectors(
                view,
                scores,
                score_policy=score_policy,
                synthetic_means=synthetic_means,
            )
            # Reference is deterministic; _score_vectors reconstructs it identically.
            if not np.array_equal(
                vectors["reference"][0], fixed_vectors["reference"][0]
            ):
                raise BetaEstimatorError("reference score changed under replicate draw")
            row_indexes = _sampled_design_row_indexes(view, plan, draw_index)
            y = view.outcome[row_indexes]
            design_matrices = {
                arm: _design_matrix(
                    view,
                    vectors[arm][0],
                    vectors[arm][1],
                    scale=scale,
                    row_indexes=row_indexes,
                )[0]
                for arm in ARM_ORDER
            }
            standardized = _strict_bootstrap_betas(y, design_matrices)
            raw = {
                arm: value / scale.standard_deviation
                for arm, value in standardized.items()
            }
            draw_rows.append(
                {
                    "schema_version": DRAW_SCHEMA,
                    "analysis_id": _analysis_id(view, score_policy),
                    "backend_id": scores.backend_id,
                    "view_id": view.view_id,
                    "outcome_id": view.outcome_id,
                    "score_policy": score_policy,
                    "draw_index": draw_index,
                    "plan_sha256": plan.sha256,
                    "status": "ok",
                    "failure_reason": None,
                    "sampled_regression_rows": len(row_indexes),
                    "betas": {
                        PRIMARY_SCALE: standardized,
                        "raw_score": raw,
                    },
                    "contrasts": {
                        PRIMARY_SCALE: _contrast_values(standardized),
                        "raw_score": _contrast_values(raw),
                    },
                }
            )
        except (statistics.BetaStatisticsError, np.linalg.LinAlgError) as exc:
            reason = str(exc)
            failures[reason] += 1
            draw_rows.append(
                {
                    "schema_version": DRAW_SCHEMA,
                    "analysis_id": _analysis_id(view, score_policy),
                    "backend_id": scores.backend_id,
                    "view_id": view.view_id,
                    "outcome_id": view.outcome_id,
                    "score_policy": score_policy,
                    "draw_index": draw_index,
                    "plan_sha256": plan.sha256,
                    "status": "rank_deficient_rejected",
                    "failure_reason": reason,
                    "sampled_regression_rows": None,
                    "betas": None,
                    "contrasts": None,
                }
            )
    successful = sum(row["status"] == "ok" for row in draw_rows)
    if not successful:
        raise BetaEstimatorError("all bootstrap draws were rejected")
    return draw_rows, {
        "plan_sha256": plan.sha256,
        "draws": BOOTSTRAP_DRAWS,
        "successful_draws": successful,
        "failed_draws": BOOTSTRAP_DRAWS - successful,
        "failure_reasons": dict(sorted(failures.items())),
        "block_ids": list(plan.block_ids),
        "meeting_ids": list(plan.meeting_ids),
        "replicate_ids_by_meeting": {
            meeting_id: list(plan.replicate_ids_by_meeting[index])
            for index, meeting_id in enumerate(plan.meeting_ids)
        },
        "replicate_draw_counts_by_meeting": {
            meeting_id: plan.replicate_draw_counts_by_meeting[index]
            for index, meeting_id in enumerate(plan.meeting_ids)
        },
        "sampled_block_indices_shape": list(plan.sampled_block_indices.shape),
        "sampled_replicate_positions_shape": list(
            plan.sampled_replicate_positions.shape
        ),
    }


def _require_exact_bootstrap_replay(
    observed_rows: Sequence[Mapping[str, Any]],
    replayed_rows: Sequence[Mapping[str, Any]],
    *,
    analysis_key: tuple[str, str],
) -> None:
    """Reject any stored draw that differs from a fresh deterministic replay."""
    if len(observed_rows) != len(replayed_rows):
        raise BetaEstimatorError(
            f"bootstrap deterministic replay coverage drift: {analysis_key}"
        )
    for draw_index, (observed, replayed) in enumerate(
        zip(observed_rows, replayed_rows, strict=True)
    ):
        if dict(observed) != dict(replayed):
            raise BetaEstimatorError(
                "bootstrap deterministic replay drift: "
                f"{analysis_key}:draw_index={draw_index}:"
                f"observed_sha256={_record_sha256(observed)}:"
                f"replayed_sha256={_record_sha256(replayed)}"
            )


def _scale_point_key(scale_id: str) -> str:
    if scale_id == PRIMARY_SCALE:
        return "standardized"
    if scale_id == "raw_score":
        return "raw_score"
    raise BetaEstimatorError(f"unknown score scale: {scale_id}")


def _estimate_id(
    *,
    backend_id: str,
    view: DesignView,
    score_policy: str,
    scale_id: str,
    arm: str,
) -> str:
    return "::".join((backend_id, _analysis_id(view, score_policy), scale_id, arm))


def _contrast_row_id(
    *,
    backend_id: str,
    view: DesignView,
    score_policy: str,
    scale_id: str,
    contrast_id: str,
) -> str:
    return "::".join(
        (backend_id, _analysis_id(view, score_policy), scale_id, contrast_id)
    )


def _summarize_analysis(
    *,
    view: DesignView,
    scores: BackendScores,
    scale: ReferenceScale,
    score_policy: str,
    point_fits: Mapping[str, Mapping[str, Any]],
    draw_rows: Sequence[Mapping[str, Any]],
    bootstrap_diagnostic: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    successful = [row for row in draw_rows if row["status"] == "ok"]
    successful_count = len(successful)
    failed_count = BOOTSTRAP_DRAWS - successful_count
    estimates: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    for scale_id in SCALE_ORDER:
        point_key = _scale_point_key(scale_id)
        for arm in ARM_ORDER:
            point = point_fits[arm][point_key]
            draws = np.asarray(
                [row["betas"][scale_id][arm] for row in successful],
                dtype=np.float64,
            )
            low, high = statistics.percentile_interval(draws)
            beta = float(point["beta_current"])
            se = float(point["hac_se_current"])
            estimates.append(
                {
                    "schema_version": ESTIMATE_SCHEMA,
                    "estimate_id": _estimate_id(
                        backend_id=scores.backend_id,
                        view=view,
                        score_policy=score_policy,
                        scale_id=scale_id,
                        arm=arm,
                    ),
                    "analysis_id": _analysis_id(view, score_policy),
                    "specification_id": _specification_id(view, score_policy, scale_id),
                    "view_id": view.view_id,
                    "view_label": VIEW_LABELS[view.view_id],
                    "outcome_id": view.outcome_id,
                    "outcome_label": OUTCOME_LABELS[view.outcome_id],
                    "backend_id": scores.backend_id,
                    "backend_label": BACKEND_LABELS[scores.backend_id],
                    "construct": scores.construct,
                    "score_policy": score_policy,
                    "scale_id": scale_id,
                    "scale_units": (
                        "basis_points_per_one_pre_external_reference_sd"
                        if scale_id == PRIMARY_SCALE
                        else "basis_points_per_one_raw_sentiment_score_unit"
                    ),
                    "arm": arm,
                    "point_estimate": beta,
                    "ci_lower": low,
                    "ci_upper": high,
                    "bootstrap_interval": "percentile_95",
                    "beta_lag": float(point["beta_lag"]),
                    "hac_standard_error_current": se,
                    "hac_ci_lower": beta - NORMAL_95 * se,
                    "hac_ci_upper": beta + NORMAL_95 * se,
                    "hac_standard_error_lag": float(point["hac_se_lag"]),
                    "nobs": int(point_fits[arm]["nobs"]),
                    "rank": int(point_fits[arm]["rank"]),
                    "dof_resid": int(point_fits[arm]["dof_resid"]),
                    "r_squared": float(point_fits[arm]["r_squared"]),
                    "adjusted_r_squared": float(point_fits[arm]["adjusted_r_squared"]),
                    "hac_lag": HAC_LAG,
                    "successful_bootstrap_draws": successful_count,
                    "failed_bootstrap_draws": failed_count,
                    "bootstrap_plan_sha256": bootstrap_diagnostic["plan_sha256"],
                }
            )

        point_betas = {
            arm: float(point_fits[arm][point_key]["beta_current"]) for arm in ARM_ORDER
        }
        point_contrasts = _contrast_values(point_betas)
        p_values_by_family: dict[str, list[tuple[int, float]]] = defaultdict(list)
        contrast_ids = [*CONTRASTS, DERIVED_CONTRAST_ID]
        for contrast_id in contrast_ids:
            if contrast_id == DERIVED_CONTRAST_ID:
                before, after, family = "chk1", "chk3", "reference_context"
                formula = "abs(beta_chk1-beta_reference)-abs(beta_chk3-beta_reference)"
            else:
                before, after, family = CONTRASTS[contrast_id]
                formula = "beta_after_minus_beta_before"
            values = np.asarray(
                [row["contrasts"][scale_id][contrast_id] for row in successful],
                dtype=np.float64,
            )
            low, high = statistics.percentile_interval(values)
            p_value = statistics.two_sided_bootstrap_p_value(values)
            family_size = {
                "primary": 1,
                "background": 2,
                "reference_context": 4,
            }[family]
            multiplicity_family_id = "::".join(
                (
                    scores.backend_id,
                    _analysis_id(view, score_policy),
                    scale_id,
                    family,
                )
            )
            row = {
                "schema_version": CONTRAST_SCHEMA,
                "contrast_row_id": _contrast_row_id(
                    backend_id=scores.backend_id,
                    view=view,
                    score_policy=score_policy,
                    scale_id=scale_id,
                    contrast_id=contrast_id,
                ),
                "contrast_id": contrast_id,
                "family": family,
                "multiplicity_family_id": multiplicity_family_id,
                "family_size": family_size,
                "before_arm": before,
                "after_arm": after,
                "formula": formula,
                "analysis_id": _analysis_id(view, score_policy),
                "specification_id": _specification_id(view, score_policy, scale_id),
                "view_id": view.view_id,
                "view_label": VIEW_LABELS[view.view_id],
                "outcome_id": view.outcome_id,
                "outcome_label": OUTCOME_LABELS[view.outcome_id],
                "backend_id": scores.backend_id,
                "backend_label": BACKEND_LABELS[scores.backend_id],
                "construct": scores.construct,
                "score_policy": score_policy,
                "scale_id": scale_id,
                "point_estimate": point_contrasts[contrast_id],
                "ci_lower": low,
                "ci_upper": high,
                "bootstrap_interval": "percentile_95",
                "p_value": p_value,
                "holm_adjusted_p": None,
                "nobs": view.nobs,
                "successful_bootstrap_draws": successful_count,
                "failed_bootstrap_draws": failed_count,
                "bootstrap_plan_sha256": bootstrap_diagnostic["plan_sha256"],
                "positive_interpretation": (
                    "CHK3 is closer to the deterministic synthetic reference beta"
                    if contrast_id == DERIVED_CONTRAST_ID
                    else "the after-arm beta exceeds the before-arm beta"
                ),
                "reference_is_official_minutes": False,
            }
            p_values_by_family[family].append((len(contrasts), p_value))
            contrasts.append(row)
        for family_values in p_values_by_family.values():
            adjusted = statistics.holm_adjust([value for _, value in family_values])
            for (row_index, _), corrected in zip(family_values, adjusted, strict=True):
                contrasts[row_index]["holm_adjusted_p"] = corrected
    return estimates, contrasts


def _descriptive_diagnostics(
    view: DesignView,
    scores: BackendScores,
    score_policy: str,
) -> dict[str, Any]:
    vectors = _score_vectors(view, scores, score_policy=score_policy)
    current = {arm: values[0] for arm, values in vectors.items()}
    summaries = {
        arm: {
            "mean": float(values.mean()),
            "standard_deviation_ddof1": float(values.std(ddof=1)),
            "rows": len(values),
        }
        for arm, values in current.items()
    }
    reference_correlations = {
        arm: float(np.corrcoef(current[arm], current["reference"])[0, 1])
        for arm in SYNTHETIC_ARMS
    }
    model_correlations = {
        f"{left}_with_{right}": float(np.corrcoef(current[left], current[right])[0, 1])
        for left, right in (("chk0", "chk1"), ("chk0", "chk3"), ("chk1", "chk3"))
    }
    if any(
        not math.isfinite(value)
        for value in [*reference_correlations.values(), *model_correlations.values()]
    ):
        raise BetaEstimatorError("descriptive score correlation is non-finite")
    return {
        "backend_id": scores.backend_id,
        "view_id": view.view_id,
        "score_policy": score_policy,
        "arm_score_summary": summaries,
        "model_with_reference_pearson": reference_correlations,
        "pairwise_model_pearson": model_correlations,
        "descriptive_only": True,
        "construct_noncomparability_warning": (
            "FinBERT measures financial valence, not hawkish-minus-dovish stance"
            if scores.backend_id == sentiment.FINBERT_BACKEND
            else None
        ),
    }


class _JsonlWriter:
    def __init__(self, path: Path, *, fsync_every: int = 1_000) -> None:
        if os.path.lexists(path):
            raise BetaEstimatorError(f"refusing overwrite: {path}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        self.path = path
        self.handle = os.fdopen(descriptor, "w", encoding="utf-8")
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
        raise BetaEstimatorError(f"refusing overwrite: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_canonical(dict(value)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
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
        raise BetaEstimatorError("Linux renameat2 is required")
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
            raise BetaEstimatorError(f"output already exists: {target}")
        raise BetaEstimatorError(
            f"renameat2 failed: errno={observed} {os.strerror(observed)}"
        )
    _fsync_directory(target.parent)


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise BetaEstimatorError(f"JSON file missing/symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BetaEstimatorError(str(exc)) from exc
    if not isinstance(value, dict):
        raise BetaEstimatorError(f"JSON file is not an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise BetaEstimatorError(f"JSONL file missing/symlink: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n") or not line.strip():
                    raise BetaEstimatorError(
                        f"noncanonical JSONL framing: {path}:{line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict) or line != _canonical(value) + "\n":
                    raise BetaEstimatorError(
                        f"noncanonical JSONL row: {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BetaEstimatorError(str(exc)) from exc
    return rows


def _intended_binding(
    physical_path: Path, final_path: Path, *, rows: int | None = None
) -> dict[str, Any]:
    value = _binding(physical_path, rows=rows)
    value["path"] = str(final_path.resolve())
    return value


def _scale_payload(scale: ReferenceScale) -> dict[str, Any]:
    return {
        "backend_id": scale.backend_id,
        "mean": scale.mean,
        "standard_deviation": scale.standard_deviation,
        "ddof": scale.ddof,
        "rows": scale.rows,
        "meeting_ids": list(scale.meeting_ids),
        "meeting_ids_sha256": statistics.sha256_json(list(scale.meeting_ids)),
        "population": (
            "all_128_complete_pre_external_reference_sentiment_meetings;"
            " includes meetings unavailable in the market-complete panel"
        ),
        "frozen_before_bootstrap": True,
    }


def _lexicon_exclusion(loaded: Mapping[str, Any]) -> dict[str, Any]:
    excluded = loaded.get("excluded_backends")
    value = (
        excluded.get(sentiment.LEXICON_BACKEND)
        if isinstance(excluded, Mapping)
        else None
    )
    if (
        not isinstance(value, Mapping)
        or value.get("backend_id") != sentiment.LEXICON_BACKEND
        or value.get("used_for_regression") is not False
        or value.get("reason") != "zero_reference_variance"
        or value.get("reference_topic_rows") != 2_048
        or value.get("reference_meeting_rows") != 256
        or value.get("pre_external_reference_meeting_rows") != 128
        or value.get("reference_matched_topic_rows") != 0
        or value.get("reference_total_matches") != 0
        or value.get("pre_external_reference_score_standard_deviation_ddof1") != 0.0
    ):
        raise BetaEstimatorError("lexicon zero-variance exclusion contract drift")
    return copy.deepcopy(dict(value))


def _sample_specs(
    market_rows: Sequence[Mapping[str, Any]], release_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for view_id in VIEW_ORDER:
        view = _make_design_view(
            market_rows=market_rows,
            release_rows=release_rows,
            view_id=view_id,
        )
        base_nobs = (
            EXPECTED_VIEW_NOBS["full"]
            if view_id.startswith("full")
            else (
                EXPECTED_VIEW_NOBS["post"]
                if view_id.startswith("post")
                else EXPECTED_VIEW_NOBS["pre_external"]
            )
        )
        result.append(
            {
                "sample_id": view_id,
                "label": VIEW_LABELS[view_id],
                "role": (
                    "primary_external"
                    if view_id == PRIMARY_VIEW
                    else "descriptive_or_sensitivity"
                ),
                "market_outcome_rows": len(_base_view_rows(market_rows, view_id)),
                "regression_nobs": view.nobs,
                "meeting_date_start": view.meeting_date_start,
                "meeting_date_end": view.meeting_date_end,
                "release_date_start": view.release_date_start,
                "release_date_end": view.release_date_end,
                "exclusion_count": base_nobs - view.nobs,
                "lag_source": "immediately_prior_meeting_in_sealed_n256_release_chronology",
            }
        )
    return result


def _formal_reporting_selectors(
    estimates: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    def primary_estimate(row: Mapping[str, Any]) -> bool:
        return (
            row["backend_id"] == PRIMARY_BACKEND
            and row["view_id"] == PRIMARY_VIEW
            and row["outcome_id"] == OUTCOME_ID
            and row["score_policy"] == PRIMARY_SCORE_POLICY
            and row["scale_id"] == PRIMARY_SCALE
        )

    primary_estimates = [
        str(row["estimate_id"]) for row in estimates if primary_estimate(row)
    ]
    primary_contrasts = [
        str(row["contrast_row_id"])
        for row in contrasts
        if row["backend_id"] == PRIMARY_BACKEND
        and row["view_id"] == PRIMARY_VIEW
        and row["outcome_id"] == OUTCOME_ID
        and row["score_policy"] == PRIMARY_SCORE_POLICY
        and row["scale_id"] == PRIMARY_SCALE
        and row["contrast_id"] == "chk3_minus_chk1"
    ]
    background = [
        str(row["contrast_row_id"])
        for row in contrasts
        if row["backend_id"] == PRIMARY_BACKEND
        and row["view_id"] == PRIMARY_VIEW
        and row["outcome_id"] == OUTCOME_ID
        and row["score_policy"] == PRIMARY_SCORE_POLICY
        and row["scale_id"] == PRIMARY_SCALE
        and row["contrast_id"] in {"chk1_minus_chk0", "chk3_minus_chk0"}
    ]
    reference_context = [
        str(row["contrast_row_id"])
        for row in contrasts
        if row["backend_id"] == PRIMARY_BACKEND
        and row["view_id"] == PRIMARY_VIEW
        and row["outcome_id"] == OUTCOME_ID
        and row["score_policy"] == PRIMARY_SCORE_POLICY
        and row["scale_id"] == PRIMARY_SCALE
        and row["family"] == "reference_context"
    ]
    headline_estimate = next(
        row["estimate_id"]
        for row in estimates
        if primary_estimate(row) and row["arm"] == "chk3"
    )
    if not (
        len(primary_estimates) == len(ARM_ORDER)
        and len(primary_contrasts) == 1
        and len(background) == 2
        and len(reference_context) == 4
    ):
        raise BetaEstimatorError("formal report selector closure drift")
    return {
        "headline_estimate_id": headline_estimate,
        "headline_contrast_row_id": primary_contrasts[0],
        "primary_estimate_ids": primary_estimates,
        "primary_contrast_row_ids": primary_contrasts,
        "background_contrast_row_ids": background,
        "reference_contrast_row_ids": reference_context,
    }


def _verdict(contrasts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary = next(
        row
        for row in contrasts
        if row["backend_id"] == PRIMARY_BACKEND
        and row["view_id"] == PRIMARY_VIEW
        and row["outcome_id"] == OUTCOME_ID
        and row["score_policy"] == PRIMARY_SCORE_POLICY
        and row["scale_id"] == PRIMARY_SCALE
        and row["contrast_id"] == "chk3_minus_chk1"
    )
    low, high = float(primary["ci_lower"]), float(primary["ci_upper"])
    relation = "above_zero" if low > 0 else "below_zero" if high < 0 else "crosses_zero"
    return {
        "primary_contrast_id": "chk3_minus_chk1",
        "paired_bootstrap_ci_relation_to_zero": relation,
        "interpretation": (
            "difference in reduced-form sentiment association coefficients;"
            " not a model-quality score and not a causal market-impact estimate"
        ),
        "larger_raw_beta_is_automatically_better": False,
    }


def _manifest_payload(
    *,
    created_at_utc: str,
    final_root: Path,
    staging_root: Path,
    sentiment_suite_path: Path,
    sentiment_loaded: Mapping[str, Any],
    market_panel_path: Path,
    market_loaded: Mapping[str, Any],
    estimates: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    publish_attempt_id: str,
    staging_basename: str,
) -> dict[str, Any]:
    inventory_manifest = sentiment_loaded["inventory"]["manifest"]
    generation_binding = inventory_manifest.get("source_generation_suite")
    if not isinstance(generation_binding, Mapping):
        raise BetaEstimatorError("sentiment inventory lost generation-suite binding")
    output_bindings = {
        "estimates": _intended_binding(
            staging_root / ESTIMATES_FILENAME,
            final_root / ESTIMATES_FILENAME,
            rows=len(estimates),
        ),
        "contrasts": _intended_binding(
            staging_root / CONTRASTS_FILENAME,
            final_root / CONTRASTS_FILENAME,
            rows=len(contrasts),
        ),
        "bootstrap_draws": _intended_binding(
            staging_root / DRAWS_FILENAME,
            final_root / DRAWS_FILENAME,
            rows=int(diagnostics["coverage"]["bootstrap_draw_rows"]),
        ),
        "diagnostics": _intended_binding(
            staging_root / DIAGNOSTICS_FILENAME,
            final_root / DIAGNOSTICS_FILENAME,
        ),
    }
    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": "complete",
        "immutable": True,
        "created_at_utc": created_at_utc,
        "evaluation_id": EVALUATION_ID,
        "research_scope": "exploratory_daily_minutes_release_association",
        "causal_claim_permitted": False,
        "arm_order": list(ARM_ORDER),
        "backend_order": list(REGRESSION_BACKEND_ORDER),
        "view_order": list(VIEW_ORDER),
        "score_policy_order": list(SCORE_POLICY_ORDER),
        "scale_order": list(SCALE_ORDER),
        "inputs": {
            "sentiment_suite_manifest": _binding(
                sentiment_suite_path,
                payload_sha256=sentiment_loaded["manifest"]["integrity"][
                    "payload_sha256"
                ],
            ),
            "market_panel_manifest": _binding(
                market_panel_path,
                payload_sha256=market_loaded["manifest"]["integrity"]["payload_sha256"],
            ),
            "generation_suite_manifest": copy.deepcopy(dict(generation_binding)),
        },
        "analysis_contract": {
            "primary_view": PRIMARY_VIEW,
            "primary_backend": PRIMARY_BACKEND,
            "primary_outcome": OUTCOME_ID,
            "primary_score_policy": PRIMARY_SCORE_POLICY,
            "primary_scale": PRIMARY_SCALE,
            "outcome_vix_control_map": copy.deepcopy(OUTCOME_VIX_CONTROL),
            "regression_formula": (
                "outcome = intercept + beta_current*current_sentiment + "
                "beta_lag*lag_sentiment + gamma*vix_matching_window + "
                "release_month_sin + release_month_cos + error"
            ),
            "lag_y_included": False,
            "lag_sentiment_source": (
                "immediately prior meeting in each view's original sealed N=256 "
                "official-release chronology; fixed before block resampling"
            ),
            "market_missing_lag_policy": (
                "a meeting excluded as a market outcome remains eligible as the "
                "sentiment lag of the next market-complete meeting"
            ),
            "seasonality_controls": [
                "sin(2*pi*release_month/12)",
                "cos(2*pi*release_month/12)",
            ],
            "covariance": "Newey-West HAC with Bartlett kernel and finite-sample correction",
            "hac_lag": HAC_LAG,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "calendar_block": "current Minutes release calendar year",
            "within_meeting_resampling": (
                "observed-size n_i draws with replacement from each meeting's "
                "paired-complete replicate intersection for the primary policy; "
                "neutral sensitivity uses five draws from fixed replicate IDs 0..4"
            ),
            "duplicate_calendar_block_replicate_semantics": (
                "when a calendar-year block is sampled more than once in one draw, "
                "each repeated occurrence reuses that draw's single meeting-level "
                "paired replicate resample"
            ),
            "shared_indices_across_arms": True,
            "lags_recomputed_at_sampled_block_boundaries": False,
            "rank_deficient_draw_policy": "reject_and_count_without_replacement",
            "holm_multiplicity_scope": (
                "separate within each backend x analysis cell x score scale x "
                "contrast family; no correction is pooled across backends, views, "
                "outcomes, score policies, or scales"
            ),
            "robustness_inference_role": "exploratory_sensitivity",
            "reference_scaling": (
                "backend-specific mean and sample SD (ddof=1) from all 128 complete "
                "pre-external reference sentiment meetings; fixed across views/draws"
            ),
            "raw_score_sensitivity": True,
            "robustness_outcomes": [
                outcome for outcome in OUTCOME_LABELS if outcome != OUTCOME_ID
            ],
            "reference_semantics": (
                "deterministic synthetic teacher target; not official FOMC Minutes"
            ),
            "excluded_lexicon": _lexicon_exclusion(sentiment_loaded),
            "lexicon_used_for_regression": False,
        },
        "sample_specs": copy.deepcopy(diagnostics["sample_specs"]),
        "reference_scales": copy.deepcopy(diagnostics["reference_scales"]),
        "reporting": _formal_reporting_selectors(estimates, contrasts),
        "artifacts": output_bindings,
        "coverage": copy.deepcopy(diagnostics["coverage"]),
        "verdict": _verdict(contrasts),
        "limitations": [
            "Synthetic Minutes were never historically released and cannot have caused observed market changes.",
            "The daily-close window contains all same-day information and is not a narrow announcement shock.",
            "The reference arm is a deterministic synthetic teacher target, not official FOMC Minutes.",
            "CHK3 cp318 is post-selection; bootstrap uncertainty does not remove checkpoint-selection bias.",
            "Full and post views are descriptive/sensitivity analyses; only pre_external is the primary external view.",
            "FinBERT financial valence is a different construct and is never pooled with hawkish-minus-dovish stance.",
            "This reduced-form daily-change design replaces an unrecoverable legacy futures-price equation.",
            "The frozen hawk/dove lexicon has zero pre-external reference variance and is diagnostic only, with no beta estimate.",
            "Bootstrap sign-tail probabilities are descriptive tail areas around zero, not null-centered hypothesis-test p-values.",
        ],
        "publication": {
            "method": "staging_fsync_then_renameat2_RENAME_NOREPLACE",
            "create_only": True,
            "failure_staging_preserved": True,
            "publish_attempt_id": publish_attempt_id,
            "staging_basename": staging_basename,
            "final_root": str(final_root),
            "gpu_work": False,
        },
        "implementation_sources": _implementation_source_bindings(),
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "gpu_work": False,
        },
    }


def estimate(
    *,
    sentiment_suite_manifest: Path,
    market_panel_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    unresolved_output = output_dir.expanduser()
    if unresolved_output.is_symlink() or os.path.lexists(unresolved_output):
        raise BetaEstimatorError("estimator output must be a fresh create-only root")
    final_root = unresolved_output.resolve()
    sentiment_path = sentiment_suite_manifest.expanduser().resolve()
    market_path = market_panel_manifest.expanduser().resolve()
    sentiment_loaded, market_loaded = _load_inputs(
        sentiment_suite_manifest=sentiment_path,
        market_panel_manifest=market_path,
    )
    market_rows = _ordered_market_rows(market_loaded["panel_rows"])
    release_rows = market_loaded["release_rows"]
    scores_by_backend = {
        backend_id: _backend_scores(sentiment_loaded["backends"][backend_id])
        for backend_id in REGRESSION_BACKEND_ORDER
    }
    scales = {
        backend_id: _reference_scale(
            scores_by_backend[backend_id],
            sentiment_loaded["backends"][backend_id]["meeting_scores"],
        )
        for backend_id in REGRESSION_BACKEND_ORDER
    }

    final_root.parent.mkdir(parents=True, exist_ok=True)
    attempt = secrets.token_hex(16)
    staging = final_root.with_name(f"{final_root.name}.staging.{os.getpid()}.{attempt}")
    if os.path.lexists(staging):
        raise BetaEstimatorError("unexpected estimator staging collision")
    staging.mkdir(mode=0o700)
    _fsync_directory(staging.parent)

    estimates: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    plan_diagnostics: list[dict[str, Any]] = []
    descriptives: list[dict[str, Any]] = []
    draw_count = 0
    with _JsonlWriter(staging / DRAWS_FILENAME) as draw_writer:
        for backend_id in REGRESSION_BACKEND_ORDER:
            scores = scores_by_backend[backend_id]
            scale = scales[backend_id]
            for view_id, outcome_id, score_policy in _analysis_inventory():
                view = _make_design_view(
                    market_rows=market_rows,
                    release_rows=release_rows,
                    view_id=view_id,
                    outcome_id=outcome_id,
                )
                _validate_view_against_scores(view, scores)
                point_fits, columns = _point_fits(view, scores, scale, score_policy)
                draw_rows, plan = _bootstrap_analysis(view, scores, scale, score_policy)
                for row in draw_rows:
                    draw_writer.write(row)
                draw_count += len(draw_rows)
                estimate_rows, contrast_rows = _summarize_analysis(
                    view=view,
                    scores=scores,
                    scale=scale,
                    score_policy=score_policy,
                    point_fits=point_fits,
                    draw_rows=draw_rows,
                    bootstrap_diagnostic=plan,
                )
                estimates.extend(estimate_rows)
                contrasts.extend(contrast_rows)
                plan_diagnostics.append(
                    {
                        "analysis_id": _analysis_id(view, score_policy),
                        "backend_id": backend_id,
                        "view_id": view_id,
                        "outcome_id": outcome_id,
                        "vix_control_id": view.vix_control_id,
                        "score_policy": score_policy,
                        "nobs": view.nobs,
                        "design_columns": list(columns),
                        "original_chronology_lags_precomputed": True,
                        **plan,
                    }
                )
                if outcome_id == OUTCOME_ID:
                    descriptives.append(
                        _descriptive_diagnostics(view, scores, score_policy)
                    )

    expected_analyses = len(REGRESSION_BACKEND_ORDER) * len(_analysis_inventory())
    expected_estimates = expected_analyses * len(SCALE_ORDER) * len(ARM_ORDER)
    expected_contrasts = expected_analyses * len(SCALE_ORDER) * (len(CONTRASTS) + 1)
    expected_draw_rows = expected_analyses * BOOTSTRAP_DRAWS
    if (
        len(plan_diagnostics) != expected_analyses
        or len(estimates) != expected_estimates
        or len(contrasts) != expected_contrasts
        or draw_count != expected_draw_rows
    ):
        raise BetaEstimatorError("estimator output matrix closure drift")
    with _JsonlWriter(staging / ESTIMATES_FILENAME) as writer:
        for row in estimates:
            writer.write(row)
    with _JsonlWriter(staging / CONTRASTS_FILENAME) as writer:
        for row in contrasts:
            writer.write(row)
    sample_specs = _sample_specs(market_rows, release_rows)
    diagnostics = seal_manifest(
        {
            "schema_version": DIAGNOSTICS_SCHEMA,
            "status": "complete",
            "immutable": True,
            "sample_specs": sample_specs,
            "reference_scales": {
                backend_id: _scale_payload(scales[backend_id])
                for backend_id in REGRESSION_BACKEND_ORDER
            },
            "excluded_backends": {
                sentiment.LEXICON_BACKEND: _lexicon_exclusion(sentiment_loaded)
            },
            "bootstrap_plans": plan_diagnostics,
            "descriptive_score_diagnostics": descriptives,
            "coverage": {
                "backends": len(REGRESSION_BACKEND_ORDER),
                "views": len(VIEW_ORDER),
                "analysis_cells": expected_analyses,
                "estimate_rows": len(estimates),
                "contrast_rows": len(contrasts),
                "bootstrap_draw_rows": draw_count,
                "bootstrap_draws_per_analysis": BOOTSTRAP_DRAWS,
                "score_policies": len(SCORE_POLICY_ORDER),
                "scales": len(SCALE_ORDER),
            },
            "non_regression_diagnostics_are_inferential": False,
        }
    )
    _write_new_json(staging / DIAGNOSTICS_FILENAME, diagnostics)
    manifest = seal_manifest(
        _manifest_payload(
            created_at_utc=_utc_now(),
            final_root=final_root,
            staging_root=staging,
            sentiment_suite_path=sentiment_path,
            sentiment_loaded=sentiment_loaded,
            market_panel_path=market_path,
            market_loaded=market_loaded,
            estimates=estimates,
            contrasts=contrasts,
            diagnostics=diagnostics,
            publish_attempt_id=attempt,
            staging_basename=staging.name,
        )
    )
    _write_new_json(staging / "manifest.json", manifest)
    staging.chmod(0o555)
    _fsync_directory(staging)
    _fsync_directory(staging.parent)
    _rename_noreplace(staging, final_root)
    return load_and_validate_estimator(final_root / "manifest.json")["manifest"]


def _validate_artifact_binding(
    binding: Mapping[str, Any], *, root: Path, filename: str, rows: int | None = None
) -> Path:
    expected_path = root / filename
    observed = _binding(expected_path, rows=rows)
    if dict(binding) != observed:
        raise BetaEstimatorError(f"artifact binding drift: {filename}")
    if stat.S_IMODE(expected_path.stat().st_mode) != 0o444:
        raise BetaEstimatorError(f"artifact is not read-only: {filename}")
    return expected_path


def _rebuild_results(
    *,
    sentiment_loaded: Mapping[str, Any],
    market_loaded: Mapping[str, Any],
    draw_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    market_rows = _ordered_market_rows(market_loaded["panel_rows"])
    release_rows = market_loaded["release_rows"]
    grouped_draws: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(draw_rows):
        if row.get("schema_version") != DRAW_SCHEMA:
            raise BetaEstimatorError(f"bootstrap draw schema drift: {row_index}")
        key = (str(row.get("backend_id")), str(row.get("analysis_id")))
        grouped_draws[key].append(row)

    scores_by_backend = {
        backend_id: _backend_scores(sentiment_loaded["backends"][backend_id])
        for backend_id in REGRESSION_BACKEND_ORDER
    }
    scales = {
        backend_id: _reference_scale(
            scores_by_backend[backend_id],
            sentiment_loaded["backends"][backend_id]["meeting_scores"],
        )
        for backend_id in REGRESSION_BACKEND_ORDER
    }
    estimates: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    descriptives: list[dict[str, Any]] = []
    expected_order: list[tuple[str, str]] = []
    for backend_id in REGRESSION_BACKEND_ORDER:
        scores = scores_by_backend[backend_id]
        scale = scales[backend_id]
        for view_id, outcome_id, score_policy in _analysis_inventory():
            view = _make_design_view(
                market_rows=market_rows,
                release_rows=release_rows,
                view_id=view_id,
                outcome_id=outcome_id,
            )
            _validate_view_against_scores(view, scores)
            analysis_id = _analysis_id(view, score_policy)
            key = (backend_id, analysis_id)
            expected_order.append(key)
            rows = grouped_draws.get(key)
            if rows is None or len(rows) != BOOTSTRAP_DRAWS:
                raise BetaEstimatorError(f"bootstrap draw coverage drift: {key}")
            if [row.get("draw_index") for row in rows] != list(range(BOOTSTRAP_DRAWS)):
                raise BetaEstimatorError(f"bootstrap draw ordering drift: {key}")
            plan = _bootstrap_plan(view, scores, score_policy)
            if any(
                row.get("plan_sha256") != plan.sha256
                or row.get("view_id") != view_id
                or row.get("outcome_id") != outcome_id
                or row.get("score_policy") != score_policy
                or row.get("status") not in {"ok", "rank_deficient_rejected"}
                for row in rows
            ):
                raise BetaEstimatorError(f"bootstrap draw identity drift: {key}")
            for row in rows:
                if row["status"] == "ok":
                    if (
                        row.get("failure_reason") is not None
                        or not isinstance(row.get("betas"), Mapping)
                        or not isinstance(row.get("contrasts"), Mapping)
                        or row.get("sampled_regression_rows") is None
                    ):
                        raise BetaEstimatorError(
                            f"successful draw payload drift: {key}"
                        )
                    for scale_id in SCALE_ORDER:
                        betas = row["betas"].get(scale_id)
                        contrasts_value = row["contrasts"].get(scale_id)
                        if (
                            not isinstance(betas, Mapping)
                            or set(betas) != set(ARM_ORDER)
                            or not isinstance(contrasts_value, Mapping)
                            or set(contrasts_value) != {*CONTRASTS, DERIVED_CONTRAST_ID}
                        ):
                            raise BetaEstimatorError(f"draw beta/contrast drift: {key}")
                        for value in [*betas.values(), *contrasts_value.values()]:
                            _finite("bootstrap draw value", value)
                        if contrasts_value != _contrast_values(
                            {arm: float(betas[arm]) for arm in ARM_ORDER}
                        ):
                            raise BetaEstimatorError(
                                f"bootstrap contrast arithmetic drift: {key}"
                            )
                elif (
                    not isinstance(row.get("failure_reason"), str)
                    or row.get("betas") is not None
                    or row.get("contrasts") is not None
                    or row.get("sampled_regression_rows") is not None
                ):
                    raise BetaEstimatorError(f"rejected draw payload drift: {key}")
            point_fits, columns = _point_fits(view, scores, scale, score_policy)
            replayed_rows, plan_diagnostic = _bootstrap_analysis(
                view, scores, scale, score_policy
            )
            _require_exact_bootstrap_replay(rows, replayed_rows, analysis_key=key)
            if plan_diagnostic["plan_sha256"] != plan.sha256:
                raise BetaEstimatorError(f"bootstrap replay plan drift: {key}")
            estimate_rows, contrast_rows = _summarize_analysis(
                view=view,
                scores=scores,
                scale=scale,
                score_policy=score_policy,
                point_fits=point_fits,
                draw_rows=rows,
                bootstrap_diagnostic=plan_diagnostic,
            )
            estimates.extend(estimate_rows)
            contrasts.extend(contrast_rows)
            plans.append(
                {
                    "analysis_id": analysis_id,
                    "backend_id": backend_id,
                    "view_id": view_id,
                    "outcome_id": outcome_id,
                    "vix_control_id": view.vix_control_id,
                    "score_policy": score_policy,
                    "nobs": view.nobs,
                    "design_columns": list(columns),
                    "original_chronology_lags_precomputed": True,
                    **plan_diagnostic,
                }
            )
            if outcome_id == OUTCOME_ID:
                descriptives.append(
                    _descriptive_diagnostics(view, scores, score_policy)
                )
    if set(grouped_draws) != set(expected_order):
        raise BetaEstimatorError("unexpected bootstrap analysis cell")
    expected_row_order = [key for key in expected_order for _ in range(BOOTSTRAP_DRAWS)]
    if [
        (str(row["backend_id"]), str(row["analysis_id"])) for row in draw_rows
    ] != expected_row_order:
        raise BetaEstimatorError("global bootstrap draw ordering drift")
    diagnostics = seal_manifest(
        {
            "schema_version": DIAGNOSTICS_SCHEMA,
            "status": "complete",
            "immutable": True,
            "sample_specs": _sample_specs(market_rows, release_rows),
            "reference_scales": {
                backend_id: _scale_payload(scales[backend_id])
                for backend_id in REGRESSION_BACKEND_ORDER
            },
            "excluded_backends": {
                sentiment.LEXICON_BACKEND: _lexicon_exclusion(sentiment_loaded)
            },
            "bootstrap_plans": plans,
            "descriptive_score_diagnostics": descriptives,
            "coverage": {
                "backends": len(REGRESSION_BACKEND_ORDER),
                "views": len(VIEW_ORDER),
                "analysis_cells": len(expected_order),
                "estimate_rows": len(estimates),
                "contrast_rows": len(contrasts),
                "bootstrap_draw_rows": len(draw_rows),
                "bootstrap_draws_per_analysis": BOOTSTRAP_DRAWS,
                "score_policies": len(SCORE_POLICY_ORDER),
                "scales": len(SCALE_ORDER),
            },
            "non_regression_diagnostics_are_inferential": False,
        }
    )
    return estimates, contrasts, diagnostics


def _report_model(
    *,
    manifest: Mapping[str, Any],
    manifest_binding: Mapping[str, Any],
    estimates: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    sample_specs = manifest["sample_specs"]
    full = next(row for row in sample_specs if row["sample_id"] == "full")
    sample_rows = [
        {
            "sample_id": row["sample_id"],
            "label": row["label"],
            "role": row["role"],
            "meeting_rows": row["market_outcome_rows"],
            "regression_nobs": row["regression_nobs"],
            "meeting_date_start": row["meeting_date_start"],
            "meeting_date_end": row["meeting_date_end"],
            "exclusion_count": row["exclusion_count"],
        }
        for row in sample_specs
    ]
    normalized_estimates = [
        {
            key: row[key]
            for key in (
                "estimate_id",
                "specification_id",
                "outcome_id",
                "outcome_label",
                "backend_id",
                "backend_label",
                "scale_id",
                "arm",
                "point_estimate",
                "ci_lower",
                "ci_upper",
                "nobs",
                "successful_bootstrap_draws",
                "failed_bootstrap_draws",
            )
        }
        | {"sample_id": row["view_id"]}
        for row in estimates
    ]
    normalized_contrasts = [
        {
            key: row[key]
            for key in (
                "contrast_row_id",
                "contrast_id",
                "family",
                "multiplicity_family_id",
                "family_size",
                "before_arm",
                "after_arm",
                "specification_id",
                "outcome_id",
                "outcome_label",
                "backend_id",
                "backend_label",
                "scale_id",
                "point_estimate",
                "ci_lower",
                "ci_upper",
                "p_value",
                "holm_adjusted_p",
                "nobs",
                "successful_bootstrap_draws",
                "failed_bootstrap_draws",
            )
        }
        | {"sample_id": row["view_id"]}
        for row in contrasts
    ]
    artifacts = manifest["artifacts"]
    inputs = manifest["inputs"]
    reference_scales = manifest["reference_scales"]
    lexicon_exclusion = manifest["analysis_contract"]["excluded_lexicon"]
    return {
        "generated_at_utc": manifest["created_at_utc"],
        "study": {
            "study_id": EVALUATION_ID,
            "title": "CHK3 sentiment–Treasury beta preservation study",
            "meeting_date_start": full["meeting_date_start"],
            "meeting_date_end": full["meeting_date_end"],
            "release_date_start": full["release_date_start"],
            "release_date_end": full["release_date_end"],
            "arm_order": list(ARM_ORDER),
            "primary_outcome_id": OUTCOME_ID,
            "primary_backend_id": PRIMARY_BACKEND,
            "primary_sample_id": PRIMARY_VIEW,
            "primary_scale_id": PRIMARY_SCALE,
        },
        "backend_rows": [
            {
                "backend_id": sentiment.DISTIL_BACKEND,
                "backend_label": BACKEND_LABELS[sentiment.DISTIL_BACKEND],
                "role": "primary",
                "regression_eligible": True,
                "reference_scale_sd": reference_scales[sentiment.DISTIL_BACKEND][
                    "standard_deviation"
                ],
                "exclusion_reason": None,
            },
            {
                "backend_id": sentiment.FINBERT_BACKEND,
                "backend_label": BACKEND_LABELS[sentiment.FINBERT_BACKEND],
                "role": "robustness_nonpooled_construct",
                "regression_eligible": True,
                "reference_scale_sd": reference_scales[sentiment.FINBERT_BACKEND][
                    "standard_deviation"
                ],
                "exclusion_reason": None,
            },
            {
                "backend_id": sentiment.LEXICON_BACKEND,
                "backend_label": BACKEND_LABELS[sentiment.LEXICON_BACKEND],
                "role": "diagnostic_only",
                "regression_eligible": False,
                "reference_scale_sd": lexicon_exclusion[
                    "pre_external_reference_score_standard_deviation_ddof1"
                ],
                "exclusion_reason": "zero_reference_variance",
            },
        ],
        "reporting": copy.deepcopy(manifest["reporting"]),
        "methodology": {
            "estimand": (
                "reduced-form difference in current sentiment association "
                "coefficients under a common backend-specific reference scale"
            ),
            "regression_formula": manifest["analysis_contract"]["regression_formula"],
            "outcome_definition": OUTCOME_LABELS[OUTCOME_ID],
            "sentiment_definition": (
                "Core8 meeting-equal score; paired-complete stochastic replicate mean"
            ),
            "covariance_estimator": "Newey-West HAC (Bartlett, finite-sample corrected)",
            "hac_lag": HAC_LAG,
            "bootstrap_unit": "calendar year of current Minutes release",
            "within_meeting_resampling": (
                "observed-size n_i paired replicate draws with replacement for the "
                "shared-complete policy; fixed K=5 for neutral imputation; shared "
                "across CHK0/CHK1/CHK3"
            ),
            "duplicate_calendar_block_replicate_semantics": (
                "duplicate occurrences of one sampled calendar-year block reuse the "
                "same meeting-level replicate resample within that bootstrap draw"
            ),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "same_sample_design": True,
            "lag_construction": manifest["analysis_contract"]["lag_sentiment_source"],
            "seasonality_controls": copy.deepcopy(
                manifest["analysis_contract"]["seasonality_controls"]
            ),
            "bootstrap_p_value": (
                "plus-one-corrected two-sided descriptive bootstrap sign-tail "
                "probability around zero, not a null-centered hypothesis test; Holm "
                "adjustment separately within backend x analysis cell x score scale "
                "x frozen contrast family; robustness inference is exploratory"
            ),
            "sensitivity_designs": [
                "full and post descriptive views",
                "selection-exposure-excluded full and post views",
                "fixed-denominator neutral-imputation K5 scores",
                "raw sentiment-score coefficient units",
                "DGS5 and DGS10 same-window outcomes",
                "DGS2 next-day and placebo windows with matching VIX controls",
                "DistilBERT FOMC stance and non-pooled FinBERT financial valence",
            ],
        },
        "sample_rows": sample_rows,
        "estimates": normalized_estimates,
        "contrasts": normalized_contrasts,
        "limitations": copy.deepcopy(manifest["limitations"]),
        "source_bindings": {
            "estimator_manifest": copy.deepcopy(dict(manifest_binding)),
            "estimates": copy.deepcopy(artifacts["estimates"]),
            "contrasts": copy.deepcopy(artifacts["contrasts"]),
            "bootstrap_draws": copy.deepcopy(artifacts["bootstrap_draws"]),
            "diagnostics": copy.deepcopy(artifacts["diagnostics"]),
            "market_panel_manifest": copy.deepcopy(inputs["market_panel_manifest"]),
            "sentiment_suite_manifest": copy.deepcopy(
                inputs["sentiment_suite_manifest"]
            ),
            "generation_suite_manifest": copy.deepcopy(
                inputs["generation_suite_manifest"]
            ),
        },
    }


def load_and_validate_estimator(manifest_path: Path) -> dict[str, Any]:
    unresolved = manifest_path.expanduser()
    if (
        unresolved.is_symlink()
        or unresolved.parent.is_symlink()
        or not unresolved.is_file()
    ):
        raise BetaEstimatorError("estimator manifest missing/symlink")
    manifest_path = unresolved.resolve()
    root = manifest_path.parent
    if manifest_path != root / "manifest.json":
        raise BetaEstimatorError("estimator manifest filename drift")
    if (
        stat.S_IMODE(root.stat().st_mode) != 0o555
        or stat.S_IMODE(manifest_path.stat().st_mode) != 0o444
    ):
        raise BetaEstimatorError("estimator output modes are not sealed")
    manifest = _read_json(manifest_path)
    payload_sha = validate_manifest_integrity(manifest)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or manifest.get("evaluation_id") != EVALUATION_ID
        or manifest.get("causal_claim_permitted") is not False
        or manifest.get("arm_order") != list(ARM_ORDER)
        or manifest.get("backend_order") != list(REGRESSION_BACKEND_ORDER)
        or manifest.get("view_order") != list(VIEW_ORDER)
        or manifest.get("score_policy_order") != list(SCORE_POLICY_ORDER)
        or manifest.get("scale_order") != list(SCALE_ORDER)
    ):
        raise BetaEstimatorError("estimator manifest header drift")
    coverage = manifest.get("coverage")
    artifacts = manifest.get("artifacts")
    inputs = manifest.get("inputs")
    publication = manifest.get("publication")
    if not all(
        isinstance(value, Mapping)
        for value in (coverage, artifacts, inputs, publication)
    ):
        raise BetaEstimatorError("estimator manifest objects missing")
    expected_analyses = len(REGRESSION_BACKEND_ORDER) * len(_analysis_inventory())
    expected_estimates = expected_analyses * len(SCALE_ORDER) * len(ARM_ORDER)
    expected_contrasts = expected_analyses * len(SCALE_ORDER) * (len(CONTRASTS) + 1)
    expected_draws = expected_analyses * BOOTSTRAP_DRAWS
    if coverage != {
        "backends": len(REGRESSION_BACKEND_ORDER),
        "views": len(VIEW_ORDER),
        "analysis_cells": expected_analyses,
        "estimate_rows": expected_estimates,
        "contrast_rows": expected_contrasts,
        "bootstrap_draw_rows": expected_draws,
        "bootstrap_draws_per_analysis": BOOTSTRAP_DRAWS,
        "score_policies": len(SCORE_POLICY_ORDER),
        "scales": len(SCALE_ORDER),
    }:
        raise BetaEstimatorError("estimator coverage drift")
    if (
        publication.get("method") != "staging_fsync_then_renameat2_RENAME_NOREPLACE"
        or publication.get("create_only") is not True
        or publication.get("failure_staging_preserved") is not True
        or publication.get("gpu_work") is not False
        or publication.get("final_root") != str(root)
    ):
        raise BetaEstimatorError("estimator publication drift")
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
        raise BetaEstimatorError("estimator publication identity drift")
    pid_text = staging_basename.removeprefix(f"{root.name}.staging.").removesuffix(
        f".{attempt}"
    )
    if not pid_text.isdigit() or int(pid_text) <= 0:
        raise BetaEstimatorError("estimator staging PID drift")
    if set(path.name for path in root.iterdir()) != {
        "manifest.json",
        ESTIMATES_FILENAME,
        CONTRASTS_FILENAME,
        DRAWS_FILENAME,
        DIAGNOSTICS_FILENAME,
    }:
        raise BetaEstimatorError("estimator file inventory drift")

    estimate_path = _validate_artifact_binding(
        artifacts["estimates"],
        root=root,
        filename=ESTIMATES_FILENAME,
        rows=expected_estimates,
    )
    contrast_path = _validate_artifact_binding(
        artifacts["contrasts"],
        root=root,
        filename=CONTRASTS_FILENAME,
        rows=expected_contrasts,
    )
    draw_path = _validate_artifact_binding(
        artifacts["bootstrap_draws"],
        root=root,
        filename=DRAWS_FILENAME,
        rows=expected_draws,
    )
    diagnostic_path = _validate_artifact_binding(
        artifacts["diagnostics"],
        root=root,
        filename=DIAGNOSTICS_FILENAME,
    )
    estimates = _read_jsonl(estimate_path)
    contrasts = _read_jsonl(contrast_path)
    draw_rows = _read_jsonl(draw_path)
    diagnostics = _read_json(diagnostic_path)
    validate_manifest_integrity(diagnostics)
    input_sentiment = inputs.get("sentiment_suite_manifest")
    input_market = inputs.get("market_panel_manifest")
    if not isinstance(input_sentiment, Mapping) or not isinstance(
        input_market, Mapping
    ):
        raise BetaEstimatorError("estimator input bindings missing")
    sentiment_path = Path(str(input_sentiment.get("path")))
    market_path = Path(str(input_market.get("path")))
    sentiment_loaded, market_loaded = _load_inputs(
        sentiment_suite_manifest=sentiment_path,
        market_panel_manifest=market_path,
    )
    if input_sentiment != _binding(
        sentiment_path,
        payload_sha256=sentiment_loaded["manifest"]["integrity"]["payload_sha256"],
    ) or input_market != _binding(
        market_path,
        payload_sha256=market_loaded["manifest"]["integrity"]["payload_sha256"],
    ):
        raise BetaEstimatorError("estimator input binding drift")
    rebuilt_estimates, rebuilt_contrasts, rebuilt_diagnostics = _rebuild_results(
        sentiment_loaded=sentiment_loaded,
        market_loaded=market_loaded,
        draw_rows=draw_rows,
    )
    if estimates != rebuilt_estimates:
        raise BetaEstimatorError("estimate reconstruction drift")
    if contrasts != rebuilt_contrasts:
        raise BetaEstimatorError("contrast reconstruction drift")
    if diagnostics != rebuilt_diagnostics:
        raise BetaEstimatorError("diagnostics reconstruction drift")
    manifest_binding = _binding(manifest_path, payload_sha256=payload_sha)
    rebuilt_manifest = seal_manifest(
        _manifest_payload(
            created_at_utc=str(manifest.get("created_at_utc")),
            final_root=root,
            staging_root=root,
            sentiment_suite_path=sentiment_path,
            sentiment_loaded=sentiment_loaded,
            market_panel_path=market_path,
            market_loaded=market_loaded,
            estimates=estimates,
            contrasts=contrasts,
            diagnostics=diagnostics,
            publish_attempt_id=attempt,
            staging_basename=staging_basename,
        )
    )
    if manifest != rebuilt_manifest:
        raise BetaEstimatorError("estimator manifest reconstruction drift")
    report_model = _report_model(
        manifest=manifest,
        manifest_binding=manifest_binding,
        estimates=estimates,
        contrasts=contrasts,
    )
    return {
        "manifest": manifest,
        "manifest_binding": manifest_binding,
        "estimates": estimates,
        "contrasts": contrasts,
        "bootstrap_draws": draw_rows,
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
    run.add_argument("--market-panel-manifest", type=Path, default=DEFAULT_MARKET_PANEL)
    run.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "estimate":
            manifest = estimate(
                sentiment_suite_manifest=args.sentiment_suite_manifest,
                market_panel_manifest=args.market_panel_manifest,
                output_dir=args.output_dir,
            )
            summary = {
                "status": manifest["status"],
                "manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
                "coverage": manifest["coverage"],
                "verdict": manifest["verdict"],
            }
        else:
            loaded = load_and_validate_estimator(args.manifest)
            summary = {
                "status": "valid",
                "manifest_binding": loaded["manifest_binding"],
                "coverage": loaded["manifest"]["coverage"],
                "verdict": loaded["manifest"]["verdict"],
            }
        print(_canonical(summary))
        return 0
    except (
        BetaEstimatorError,
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
    "BetaEstimatorError",
    "DEFAULT_OUTPUT",
    "DEFAULT_SENTIMENT_SUITE",
    "EVALUATION_ID",
    "MANIFEST_SCHEMA",
    "estimate",
    "load_and_validate_estimator",
    "main",
]
