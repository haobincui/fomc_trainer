"""Build the sealed CHK3 beta technical report from a complete estimator.

This module is a presentation adapter, not an estimator.  It delegates source
integrity and statistical reconstruction to the authoritative econometric
loader, then converts those validated rows into the canonical Data Analytics
``artifact.json`` contract.  The packaged report builder is the only HTML
renderer.  Incomplete, partial, or schema-drifting estimator inputs fail before
the final report directory is created.

The study is an exploratory association-preservation exercise.  Synthetic
Minutes were not released at the historical event time and cannot have caused
the observed Treasury-yield movements.
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
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / (
    "output/evaluation/main/chk3_beta_core8_merged_n2048_vllm_k5_20260816_v1"
)
DEFAULT_ESTIMATOR_MANIFEST = (
    RUN_ROOT / "beta_estimator_time_block_bootstrap_v1/manifest.json"
)
DEFAULT_OUTPUT_ROOT = RUN_ROOT / "technical_report_v1"

ESTIMATOR_MANIFEST_SCHEMA = (
    "chk3-beta-core8-econometric-estimator-time-block-bootstrap-manifest-v1"
)
REPORT_MANIFEST_SCHEMA = "chk3-beta-core8-technical-report-manifest-v1"
REPORT_ARTIFACT_CONTRACT = "chk3-beta-core8-technical-report-artifact-v1"
REPORT_ID = "chk3-beta-core8-technical-report-v1"
EXPECTED_BOOTSTRAP_DRAWS = 10_000
APPENDIX_PAGE_SIZE = 15
ARM_ORDER = ("reference", "chk0", "chk1", "chk3")
PRIMARY_CONTRAST_ID = "chk3_minus_chk1"
BACKGROUND_CONTRAST_IDS = ("chk1_minus_chk0", "chk3_minus_chk0")
REFERENCE_CONTRAST_IDS = (
    "chk0_minus_reference",
    "chk1_minus_reference",
    "chk3_minus_reference",
    "reference_distance_gain_chk3_vs_chk1",
)
EXPECTED_VIEW_NOBS = {
    "pre_external": 126,
    "full": 254,
    "post": 127,
    "full_excluding_cp318_selection": 243,
    "post_excluding_cp318_selection": 116,
}
REQUIRED_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
PRIMARY_OUTCOME_ID = "dgs2_release_minus_previous_bp"
PRIMARY_SCALE_ID = "pre_external_reference_sd"
PRIMARY_BACKEND_ID = "distilbert_fomc_9c061b4_v1"
ROBUSTNESS_BACKEND_ID = "prosus_finbert_4556d13_v1"
EXCLUDED_LEXICON_BACKEND_ID = "lucca_trebbi_hawk_dove_lexicon_v1"
REGRESSION_BACKEND_IDS = (PRIMARY_BACKEND_ID, ROBUSTNESS_BACKEND_ID)

PACKAGER_RELATIVE = Path("skills/build-report/scripts/deliver_portable_artifact.mjs")
DEFAULT_PLUGIN_ROOT = Path(
    "/home/haobin_cui/.codex/plugins/cache/openai-curated-remote/"
    "data-analytics/0.2.8-13ceeea1f599"
)
DEFAULT_NODE_CANDIDATES = (
    ROOT / ".cache/report_node/bin/node",
    ROOT / ".cache/node_official/node-v22.12.0-linux-x64/bin/node",
    Path("/home/haobin_cui/.conda/envs/fomc_trainer/bin/node"),
)

NON_CAUSAL_DISCLOSURE = (
    "This is an exploratory association analysis, not a causal estimate. "
    "Synthetic Minutes were never released at the historical event time and "
    "therefore could not have caused the observed Treasury-yield changes."
)
REFERENCE_DISCLOSURE = (
    "The reference arm is a deterministic synthetic teacher target, not "
    "official FOMC Minutes."
)
EVENT_WINDOW_DISCLOSURE = (
    "Daily-close event windows contain all information and trading during the "
    "window, not only news about the Minutes release."
)
POST_SELECTION_DISCLOSURE = (
    "CHK3 cp318 is post-selection: checkpoint selection used an earlier N12 "
    "diagnostic, so bootstrap uncertainty does not remove checkpoint-selection "
    "bias."
)
LEXICON_DISCLOSURE = (
    "The frozen hawk/dove lexicon is diagnostic only: its pre-external "
    "reference-score standard deviation is 0, so it is excluded from every "
    "regression and no lexicon beta is reported."
)


class BetaTechnicalReportError(RuntimeError):
    """The estimator-to-report contract or publication process failed closed."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _canonical(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise BetaTechnicalReportError(f"non-canonical JSON payload: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path, *, payload_sha256: str | None = None) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise BetaTechnicalReportError(f"bound path is a symlink: {unresolved}")
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise BetaTechnicalReportError(f"bound file is missing: {resolved}")
    value: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }
    if payload_sha256 is not None:
        value["payload_sha256"] = payload_sha256
    return value


def _current_runtime_versions() -> dict[str, str]:
    # Import NumPy through the same interpreter environment that will import the
    # estimator.  Bootstrap plan hashes include NumPy RNG output, so accepting a
    # different NumPy build would make an otherwise sealed plan unreplayable.
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - deployment guard
        raise BetaTechnicalReportError(
            "NumPy is unavailable for estimator replay"
        ) from exc
    return {
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
    }


def _current_thread_runtime() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in REQUIRED_THREAD_ENV}


def _report_runtime() -> dict[str, Any]:
    return {
        **_current_runtime_versions(),
        "python_executable": str(Path(sys.executable).resolve()),
        "thread_environment": _current_thread_runtime(),
        "gpu_work": False,
    }


def _validate_estimator_runtime(manifest_path: Path) -> dict[str, Any]:
    """Fail before the authoritative loader when replay versions drift."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BetaTechnicalReportError(
            f"estimator manifest cannot be read for runtime preflight: {exc}"
        ) from exc
    runtime = manifest.get("runtime") if isinstance(manifest, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise BetaTechnicalReportError("estimator runtime contract is missing")
    expected = {
        "python_version": runtime.get("python_version"),
        "numpy_version": runtime.get("numpy_version"),
    }
    if any(not isinstance(value, str) or not value for value in expected.values()):
        raise BetaTechnicalReportError("estimator runtime version contract drift")
    observed = _current_runtime_versions()
    for name in ("python_version", "numpy_version"):
        if observed[name] != expected[name]:
            raise BetaTechnicalReportError(
                f"estimator {name} runtime drift: "
                f"expected {expected[name]}, observed {observed[name]}"
            )
    thread_runtime = _current_thread_runtime()
    if thread_runtime != REQUIRED_THREAD_ENV:
        raise BetaTechnicalReportError(
            "estimator linear-algebra thread runtime drift: "
            f"expected {REQUIRED_THREAD_ENV}, observed {thread_runtime}"
        )
    return {
        **observed,
        "python_executable": str(Path(sys.executable).resolve()),
        "thread_environment": thread_runtime,
        "gpu_work": False,
    }


def _intended_binding(staged_path: Path, final_path: Path) -> dict[str, Any]:
    value = _binding(staged_path)
    value["path"] = str(final_path.expanduser().resolve())
    return value


def _repo_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise BetaTechnicalReportError(
            f"portable report source is outside repository root: {resolved}"
        ) from exc


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BetaTechnicalReportError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise BetaTechnicalReportError(f"{name} must be finite")
    return numeric


def _nonnegative_int(name: str, value: Any, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BetaTechnicalReportError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise BetaTechnicalReportError(f"{name} must be {qualifier}")
    return value


def _validate_interval(prefix: str, row: Mapping[str, Any]) -> None:
    point = _finite(f"{prefix}.point_estimate", row.get("point_estimate"))
    low = _finite(f"{prefix}.ci_lower", row.get("ci_lower"))
    high = _finite(f"{prefix}.ci_upper", row.get("ci_upper"))
    if low > point or point > high:
        raise BetaTechnicalReportError(
            f"{prefix} has unordered CI/point: {low}, {point}, {high}"
        )


def _ci_text(row: Mapping[str, Any]) -> str:
    return f"[{float(row['ci_lower']):+.6f}, {float(row['ci_upper']):+.6f}]"


def _signed_text(value: float) -> str:
    return f"{value:+.6f}"


def _beta_display(value: float, *, difference: bool = False) -> str:
    symbol = "Δβ" if difference else "β"
    return f"{symbol}={float(value):+.6f}"


def _probability_display(value: float, *, adjusted: bool = False) -> str:
    label = "Holm p" if adjusted else "p"
    return f"{label}={float(value):.6f}"


def _reference_sd_display(value: float) -> str:
    numeric = float(value)
    if numeric == 0.0:
        return "SD=0 (exact zero; excluded)"
    return f"SD={numeric:.12g}"


def _score_policy_from_specification(specification_id: str) -> str:
    matches = [
        policy
        for policy in ("shared_complete_core8", "neutral_imputed_fixed_k5")
        if f"__{policy}__" in specification_id
    ]
    if len(matches) != 1:
        raise BetaTechnicalReportError(
            f"specification score-policy identity drift: {specification_id}"
        )
    return matches[0]


def _paged(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [
        rows[offset : offset + APPENDIX_PAGE_SIZE]
        for offset in range(0, len(rows), APPENDIX_PAGE_SIZE)
    ]


def _deduplicated_limitations(items: list[str]) -> list[str]:
    canonical = [
        NON_CAUSAL_DISCLOSURE,
        REFERENCE_DISCLOSURE,
        EVENT_WINDOW_DISCLOSURE,
        POST_SELECTION_DISCLOSURE,
        LEXICON_DISCLOSURE,
    ]

    def duplicate_category(value: str) -> bool:
        lowered = value.lower()
        return any(
            (
                "synthetic minutes" in lowered and "caus" in lowered,
                "daily-close" in lowered,
                "daily data" in lowered and "intraday" in lowered,
                "reference arm" in lowered and "official" in lowered,
                "cp318" in lowered and "post-selection" in lowered,
                "lexicon" in lowered and "zero" in lowered,
            )
        )

    seen = set(canonical)
    for item in items:
        if item not in seen and not duplicate_category(item):
            canonical.append(item)
            seen.add(item)
    return canonical


def _zero_relation(row: Mapping[str, Any]) -> str:
    low = float(row["ci_lower"])
    high = float(row["ci_upper"])
    if low > 0:
        return "lies entirely above zero"
    if high < 0:
        return "lies entirely below zero"
    return "includes zero"


def _normalized_contract() -> dict[str, Any]:
    """Describe the stable internal adapter required by the artifact builder."""

    return {
        "schema_version": REPORT_ARTIFACT_CONTRACT,
        "required_top_level": [
            "generated_at_utc",
            "study",
            "backend_rows",
            "reporting",
            "methodology",
            "sample_rows",
            "estimates",
            "contrasts",
            "limitations",
            "source_bindings",
        ],
        "estimate_fields": [
            "estimate_id",
            "specification_id",
            "sample_id",
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
        ],
        "contrast_fields": [
            "contrast_row_id",
            "contrast_id",
            "family",
            "multiplicity_family_id",
            "family_size",
            "before_arm",
            "after_arm",
            "specification_id",
            "sample_id",
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
        ],
        "reporting_fields": [
            "headline_estimate_id",
            "headline_contrast_row_id",
            "primary_estimate_ids",
            "primary_contrast_row_ids",
            "background_contrast_row_ids",
            "reference_contrast_row_ids",
        ],
        "formal_bootstrap_draws": EXPECTED_BOOTSTRAP_DRAWS,
    }


def validate_report_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact normalized data model used by report rendering."""

    required = set(_normalized_contract()["required_top_level"])
    if set(model) != required:
        raise BetaTechnicalReportError(
            f"normalized report-model inventory drift: {sorted(set(model) ^ required)}"
        )
    generated_at = model.get("generated_at_utc")
    study = model.get("study")
    backend_rows = model.get("backend_rows")
    reporting = model.get("reporting")
    methodology = model.get("methodology")
    sample_rows = model.get("sample_rows")
    estimates = model.get("estimates")
    contrasts = model.get("contrasts")
    limitations = model.get("limitations")
    bindings = model.get("source_bindings")
    if not isinstance(generated_at, str) or not generated_at.endswith("Z"):
        raise BetaTechnicalReportError("generated_at_utc is not a UTC timestamp")
    if not isinstance(study, Mapping) or set(study) != {
        "study_id",
        "title",
        "meeting_date_start",
        "meeting_date_end",
        "release_date_start",
        "release_date_end",
        "arm_order",
        "primary_outcome_id",
        "primary_backend_id",
        "primary_sample_id",
        "primary_scale_id",
    }:
        raise BetaTechnicalReportError("study metadata contract drift")
    if study.get("arm_order") != list(ARM_ORDER):
        raise BetaTechnicalReportError("study arm order drift")
    if (
        study.get("primary_outcome_id") != PRIMARY_OUTCOME_ID
        or study.get("primary_backend_id") != PRIMARY_BACKEND_ID
        or study.get("primary_sample_id") != "pre_external"
        or study.get("primary_scale_id") != PRIMARY_SCALE_ID
    ):
        raise BetaTechnicalReportError(
            "frozen primary view/outcome/backend/scale drift"
        )
    required_backend_fields = {
        "backend_id",
        "backend_label",
        "role",
        "regression_eligible",
        "reference_scale_sd",
        "exclusion_reason",
    }
    if not isinstance(backend_rows, list) or len(backend_rows) != 3:
        raise BetaTechnicalReportError("backend eligibility inventory drift")
    backend_by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(backend_rows):
        if not isinstance(row, Mapping) or set(row) != required_backend_fields:
            raise BetaTechnicalReportError(f"backend row {index} contract drift")
        backend_id = row.get("backend_id")
        if (
            not isinstance(backend_id, str)
            or not backend_id
            or backend_id in backend_by_id
            or not isinstance(row.get("backend_label"), str)
            or not row.get("backend_label")
        ):
            raise BetaTechnicalReportError("backend identities must be unique")
        backend_by_id[backend_id] = row
    if set(backend_by_id) != {
        PRIMARY_BACKEND_ID,
        ROBUSTNESS_BACKEND_ID,
        EXCLUDED_LEXICON_BACKEND_ID,
    }:
        raise BetaTechnicalReportError("backend eligibility inventory drift")
    expected_backend_roles = {
        PRIMARY_BACKEND_ID: ("primary", True, None),
        ROBUSTNESS_BACKEND_ID: ("robustness_nonpooled_construct", True, None),
        EXCLUDED_LEXICON_BACKEND_ID: (
            "diagnostic_only",
            False,
            "zero_reference_variance",
        ),
    }
    for backend_id, (
        role,
        eligible,
        exclusion_reason,
    ) in expected_backend_roles.items():
        row = backend_by_id[backend_id]
        scale_sd = _finite(
            f"backend[{backend_id}].reference_scale_sd",
            row.get("reference_scale_sd"),
        )
        if (
            row.get("role") != role
            or row.get("regression_eligible") is not eligible
            or row.get("exclusion_reason") != exclusion_reason
            or (scale_sd > 0.0) is not eligible
        ):
            raise BetaTechnicalReportError(
                f"backend eligibility/scale contract drift: {backend_id}"
            )
    if not isinstance(methodology, Mapping) or set(methodology) != {
        "estimand",
        "regression_formula",
        "outcome_definition",
        "sentiment_definition",
        "covariance_estimator",
        "hac_lag",
        "bootstrap_unit",
        "within_meeting_resampling",
        "duplicate_calendar_block_replicate_semantics",
        "bootstrap_draws",
        "bootstrap_seed",
        "same_sample_design",
        "lag_construction",
        "seasonality_controls",
        "bootstrap_p_value",
        "sensitivity_designs",
    }:
        raise BetaTechnicalReportError("methodology contract drift")
    if methodology.get("bootstrap_draws") != EXPECTED_BOOTSTRAP_DRAWS:
        raise BetaTechnicalReportError("formal report requires exactly 10,000 draws")
    _nonnegative_int("methodology.hac_lag", methodology.get("hac_lag"))
    _nonnegative_int(
        "methodology.bootstrap_seed", methodology.get("bootstrap_seed"), positive=True
    )
    if methodology.get("same_sample_design") is not True:
        raise BetaTechnicalReportError("same-sample design is not affirmed")
    if not isinstance(methodology.get("sensitivity_designs"), list):
        raise BetaTechnicalReportError("sensitivity designs must be a list")
    if methodology.get("seasonality_controls") != [
        "sin(2*pi*release_month/12)",
        "cos(2*pi*release_month/12)",
    ]:
        raise BetaTechnicalReportError("cyclic release-month controls drift")
    if methodology.get("bootstrap_p_value") != (
        "plus-one-corrected two-sided descriptive bootstrap sign-tail "
        "probability around zero, not a null-centered hypothesis test; Holm "
        "adjustment separately within backend x analysis cell x score scale "
        "x frozen contrast family; robustness inference is exploratory"
    ):
        raise BetaTechnicalReportError("bootstrap p-value method drift")
    if not isinstance(sample_rows, list) or not sample_rows:
        raise BetaTechnicalReportError("sample audit rows are missing")
    required_sample_fields = {
        "sample_id",
        "label",
        "role",
        "meeting_rows",
        "regression_nobs",
        "meeting_date_start",
        "meeting_date_end",
        "exclusion_count",
    }
    sample_ids: set[str] = set()
    for index, row in enumerate(sample_rows):
        if not isinstance(row, Mapping) or set(row) != required_sample_fields:
            raise BetaTechnicalReportError(f"sample row {index} contract drift")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise BetaTechnicalReportError("sample IDs must be non-empty and unique")
        sample_ids.add(sample_id)
        _nonnegative_int(f"sample[{sample_id}].meeting_rows", row.get("meeting_rows"))
        _nonnegative_int(
            f"sample[{sample_id}].regression_nobs",
            row.get("regression_nobs"),
        )
        _nonnegative_int(
            f"sample[{sample_id}].exclusion_count", row.get("exclusion_count")
        )
    if sample_ids != set(EXPECTED_VIEW_NOBS):
        raise BetaTechnicalReportError("sample view inventory drift")
    for row in sample_rows:
        expected_nobs = EXPECTED_VIEW_NOBS[str(row["sample_id"])]
        if int(row["regression_nobs"]) != expected_nobs:
            raise BetaTechnicalReportError(
                f"{row['sample_id']} regression N drift: {row['regression_nobs']}"
            )

    if not isinstance(estimates, list) or not estimates:
        raise BetaTechnicalReportError("estimate rows are missing")
    estimate_fields = set(_normalized_contract()["estimate_fields"])
    estimate_ids: set[str] = set()
    for index, row in enumerate(estimates):
        if not isinstance(row, Mapping) or set(row) != estimate_fields:
            raise BetaTechnicalReportError(f"estimate row {index} contract drift")
        estimate_id = row.get("estimate_id")
        if (
            not isinstance(estimate_id, str)
            or not estimate_id
            or estimate_id in estimate_ids
        ):
            raise BetaTechnicalReportError("estimate IDs must be non-empty and unique")
        estimate_ids.add(estimate_id)
        if (
            row.get("arm") not in ARM_ORDER
            or row.get("sample_id") not in sample_ids
            or row.get("backend_id") not in REGRESSION_BACKEND_IDS
        ):
            raise BetaTechnicalReportError(f"estimate identity drift: {estimate_id}")
        _validate_interval(f"estimate[{estimate_id}]", row)
        _nonnegative_int(
            f"estimate[{estimate_id}].nobs", row.get("nobs"), positive=True
        )
        successful = _nonnegative_int(
            f"estimate[{estimate_id}].successful_bootstrap_draws",
            row.get("successful_bootstrap_draws"),
        )
        failed = _nonnegative_int(
            f"estimate[{estimate_id}].failed_bootstrap_draws",
            row.get("failed_bootstrap_draws"),
        )
        if successful + failed != EXPECTED_BOOTSTRAP_DRAWS:
            raise BetaTechnicalReportError(
                f"estimate {estimate_id} bootstrap accounting is not 10,000"
            )

    if not isinstance(contrasts, list) or not contrasts:
        raise BetaTechnicalReportError("contrast rows are missing")
    contrast_fields = set(_normalized_contract()["contrast_fields"])
    contrast_row_ids: set[str] = set()
    estimate_pairs = {
        (
            str(row["specification_id"]),
            str(row["sample_id"]),
            str(row["outcome_id"]),
            str(row["backend_id"]),
            str(row["scale_id"]),
            str(row["arm"]),
        ): row
        for row in estimates
    }
    for index, row in enumerate(contrasts):
        if not isinstance(row, Mapping) or set(row) != contrast_fields:
            raise BetaTechnicalReportError(f"contrast row {index} contract drift")
        row_id = row.get("contrast_row_id")
        if not isinstance(row_id, str) or not row_id or row_id in contrast_row_ids:
            raise BetaTechnicalReportError(
                "contrast row IDs must be non-empty and unique"
            )
        contrast_row_ids.add(row_id)
        if (
            row.get("family") not in {"primary", "background", "reference_context"}
            or row.get("before_arm") not in ARM_ORDER
            or row.get("after_arm") not in ARM_ORDER
            or row.get("sample_id") not in sample_ids
            or row.get("backend_id") not in REGRESSION_BACKEND_IDS
        ):
            raise BetaTechnicalReportError(f"contrast identity drift: {row_id}")
        expected_family_size = {
            "primary": 1,
            "background": 2,
            "reference_context": 4,
        }[str(row["family"])]
        if (
            row.get("family_size") != expected_family_size
            or not isinstance(row.get("multiplicity_family_id"), str)
            or not str(row["multiplicity_family_id"]).endswith(
                f"::{row['scale_id']}::{row['family']}"
            )
        ):
            raise BetaTechnicalReportError(
                f"contrast multiplicity boundary drift: {row_id}"
            )
        expected_arm_pair = {
            PRIMARY_CONTRAST_ID: ("chk1", "chk3"),
            "chk1_minus_chk0": ("chk0", "chk1"),
            "chk3_minus_chk0": ("chk0", "chk3"),
            "chk0_minus_reference": ("reference", "chk0"),
            "chk1_minus_reference": ("reference", "chk1"),
            "chk3_minus_reference": ("reference", "chk3"),
            "reference_distance_gain_chk3_vs_chk1": ("chk1", "chk3"),
        }
        contrast_id = str(row.get("contrast_id"))
        allowed_ids = {
            "primary": {PRIMARY_CONTRAST_ID},
            "background": set(BACKGROUND_CONTRAST_IDS),
            "reference_context": set(REFERENCE_CONTRAST_IDS),
        }[str(row["family"])]
        if contrast_id not in allowed_ids or (
            row.get("before_arm"),
            row.get("after_arm"),
        ) != expected_arm_pair.get(contrast_id):
            raise BetaTechnicalReportError(
                f"contrast family/arm definition drift: {row_id}"
            )
        _validate_interval(f"contrast[{row_id}]", row)
        for field in ("p_value", "holm_adjusted_p"):
            probability = _finite(f"contrast[{row_id}].{field}", row.get(field))
            if not 0.0 <= probability <= 1.0:
                raise BetaTechnicalReportError(
                    f"contrast {row_id} {field} outside [0,1]"
                )
        if float(row["holm_adjusted_p"]) + 1e-15 < float(row["p_value"]):
            raise BetaTechnicalReportError(
                f"contrast {row_id} Holm p is below its raw p-value"
            )
        _nonnegative_int(f"contrast[{row_id}].nobs", row.get("nobs"), positive=True)
        successful = _nonnegative_int(
            f"contrast[{row_id}].successful_bootstrap_draws",
            row.get("successful_bootstrap_draws"),
        )
        failed = _nonnegative_int(
            f"contrast[{row_id}].failed_bootstrap_draws",
            row.get("failed_bootstrap_draws"),
        )
        if successful + failed != EXPECTED_BOOTSTRAP_DRAWS:
            raise BetaTechnicalReportError(
                f"contrast {row_id} bootstrap accounting is not 10,000"
            )
        common_key = (
            str(row["specification_id"]),
            str(row["sample_id"]),
            str(row["outcome_id"]),
            str(row["backend_id"]),
            str(row["scale_id"]),
        )
        before = estimate_pairs.get((*common_key, str(row["before_arm"])))
        after = estimate_pairs.get((*common_key, str(row["after_arm"])))
        if before is None or after is None:
            raise BetaTechnicalReportError(
                f"contrast {row_id} has no same-specification arm estimates"
            )
        reference = estimate_pairs.get((*common_key, "reference"))
        if contrast_id == "reference_distance_gain_chk3_vs_chk1":
            if reference is None:
                raise BetaTechnicalReportError(
                    f"contrast {row_id} has no same-specification reference estimate"
                )
            recomputed = abs(
                float(before["point_estimate"]) - float(reference["point_estimate"])
            ) - abs(float(after["point_estimate"]) - float(reference["point_estimate"]))
        else:
            recomputed = float(after["point_estimate"]) - float(
                before["point_estimate"]
            )
        if not math.isclose(
            recomputed,
            float(row["point_estimate"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise BetaTechnicalReportError(
                f"contrast {row_id} point difference does not reconcile"
            )
        same_sample_nobs = {
            int(before["nobs"]),
            int(after["nobs"]),
            int(row["nobs"]),
        }
        if (
            reference is not None
            and contrast_id == "reference_distance_gain_chk3_vs_chk1"
        ):
            same_sample_nobs.add(int(reference["nobs"]))
        if len(same_sample_nobs) != 1:
            raise BetaTechnicalReportError(
                f"contrast {row_id} violates the same-sample denominator"
            )

    reporting_fields = set(_normalized_contract()["reporting_fields"])
    if not isinstance(reporting, Mapping) or set(reporting) != reporting_fields:
        raise BetaTechnicalReportError("reporting selector contract drift")
    for field in (
        "primary_estimate_ids",
        "primary_contrast_row_ids",
        "background_contrast_row_ids",
        "reference_contrast_row_ids",
    ):
        values = reporting.get(field)
        if (
            not isinstance(values, list)
            or not values
            or len(set(values)) != len(values)
        ):
            raise BetaTechnicalReportError(f"{field} must be a non-empty unique list")
    if not set(reporting["primary_estimate_ids"]).issubset(estimate_ids):
        raise BetaTechnicalReportError(
            "primary estimate selector references missing rows"
        )
    if reporting.get("headline_estimate_id") not in set(
        reporting["primary_estimate_ids"]
    ):
        raise BetaTechnicalReportError("headline estimate is not primary")
    if not set(reporting["primary_contrast_row_ids"]).issubset(contrast_row_ids):
        raise BetaTechnicalReportError(
            "primary contrast selector references missing rows"
        )
    if reporting.get("headline_contrast_row_id") not in set(
        reporting["primary_contrast_row_ids"]
    ):
        raise BetaTechnicalReportError("headline contrast is not primary")
    if not set(reporting["background_contrast_row_ids"]).issubset(contrast_row_ids):
        raise BetaTechnicalReportError(
            "background contrast selector references missing rows"
        )
    if not set(reporting["reference_contrast_row_ids"]).issubset(contrast_row_ids):
        raise BetaTechnicalReportError(
            "reference-context selector references missing rows"
        )
    selected_primary = {
        str(row["contrast_row_id"]): row
        for row in contrasts
        if row["contrast_row_id"] in reporting["primary_contrast_row_ids"]
    }
    selected_background = {
        str(row["contrast_row_id"]): row
        for row in contrasts
        if row["contrast_row_id"] in reporting["background_contrast_row_ids"]
    }
    selected_reference = {
        str(row["contrast_row_id"]): row
        for row in contrasts
        if row["contrast_row_id"] in reporting["reference_contrast_row_ids"]
    }
    selected_primary_estimates = {
        str(row["estimate_id"]): row
        for row in estimates
        if row["estimate_id"] in reporting["primary_estimate_ids"]
    }
    if {str(row["arm"]) for row in selected_primary_estimates.values()} != set(
        ARM_ORDER
    ):
        raise BetaTechnicalReportError(
            "primary estimate selector must contain all four text arms exactly once"
        )
    primary_keys = {
        (
            str(row["specification_id"]),
            str(row["sample_id"]),
            str(row["outcome_id"]),
            str(row["backend_id"]),
            str(row["scale_id"]),
            int(row["nobs"]),
        )
        for row in selected_primary_estimates.values()
    }
    expected_primary_key = next(iter(primary_keys)) if len(primary_keys) == 1 else None
    if (
        expected_primary_key is None
        or expected_primary_key[1] != "pre_external"
        or expected_primary_key[2] != PRIMARY_OUTCOME_ID
        or expected_primary_key[3] != str(study["primary_backend_id"])
        or expected_primary_key[4] != PRIMARY_SCALE_ID
        or expected_primary_key[5] != EXPECTED_VIEW_NOBS["pre_external"]
    ):
        raise BetaTechnicalReportError(
            "primary estimate selector is not the frozen same-specification design"
        )
    if {str(row["contrast_id"]) for row in selected_primary.values()} != {
        PRIMARY_CONTRAST_ID
    }:
        raise BetaTechnicalReportError(
            "primary selector must contain the CHK3-minus-CHK1 contrast exactly"
        )
    if {str(row["contrast_id"]) for row in selected_background.values()} != set(
        BACKGROUND_CONTRAST_IDS
    ):
        raise BetaTechnicalReportError(
            "background selector must contain both frozen CHK0 contrasts"
        )
    if {str(row["contrast_id"]) for row in selected_reference.values()} != set(
        REFERENCE_CONTRAST_IDS
    ):
        raise BetaTechnicalReportError(
            "reference selector must contain all frozen contextual contrasts"
        )
    primary_contrast_rows = [
        *selected_primary.values(),
        *selected_background.values(),
        *selected_reference.values(),
    ]
    if any(
        (
            str(row["specification_id"]),
            str(row["sample_id"]),
            str(row["outcome_id"]),
            str(row["backend_id"]),
            str(row["scale_id"]),
            int(row["nobs"]),
        )
        != expected_primary_key
        for row in primary_contrast_rows
    ):
        raise BetaTechnicalReportError(
            "reported primary/background/reference contrasts do not share the "
            "frozen primary specification"
        )
    if any(row.get("family") != "primary" for row in selected_primary.values()):
        raise BetaTechnicalReportError("primary selector includes non-primary contrast")
    if any(row.get("family") != "background" for row in selected_background.values()):
        raise BetaTechnicalReportError(
            "background selector includes non-background contrast"
        )
    if any(
        row.get("family") != "reference_context" for row in selected_reference.values()
    ):
        raise BetaTechnicalReportError(
            "reference-context selector includes a different contrast family"
        )
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(not isinstance(item, str) or not item.strip() for item in limitations)
    ):
        raise BetaTechnicalReportError("limitations must be non-empty strings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "estimator_manifest",
        "estimates",
        "contrasts",
        "bootstrap_draws",
        "diagnostics",
        "market_panel_manifest",
        "sentiment_suite_manifest",
        "generation_suite_manifest",
    }:
        raise BetaTechnicalReportError("source binding inventory drift")
    for name, value in bindings.items():
        if not isinstance(value, Mapping):
            raise BetaTechnicalReportError(f"{name} source binding is not an object")
        observed = _binding(Path(str(value.get("path"))))
        for field in ("path", "bytes", "sha256"):
            if observed[field] != value.get(field):
                raise BetaTechnicalReportError(f"{name} source binding drift")

    return copy.deepcopy(dict(model))


def _source(
    *,
    source_id: str,
    label: str,
    binding: Mapping[str, Any],
    executed_at: str,
) -> dict[str, Any]:
    relative_path = _repo_relative(Path(str(binding["path"])))
    escaped_path = relative_path.replace("'", "''")
    return {
        "id": source_id,
        "label": label,
        "path": relative_path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": f"SELECT * FROM read_json_auto('{escaped_path}')",
            "description": (
                "Reads the sealed source artifact. Report datasets are bounded, "
                "deterministic projections of rows validated by the authoritative "
                "estimator loader."
            ),
            "executed_at": executed_at,
            "tables_used": [relative_path],
        },
    }


def _estimate_display(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "estimate_id": row["estimate_id"],
        "specification": row["specification_id"],
        "sample": row["sample_id"],
        "outcome": row["outcome_label"],
        "backend_id": row["backend_id"],
        "sentiment_backend": row["backend_label"],
        "scale": row["scale_id"],
        "arm": str(row["arm"]).upper(),
        "beta": float(row["point_estimate"]),
        "beta_display": _beta_display(float(row["point_estimate"])),
        "ci_lower": float(row["ci_lower"]),
        "ci_upper": float(row["ci_upper"]),
        "ci_95": _ci_text(row),
        "nobs": int(row["nobs"]),
        "successful_draws": int(row["successful_bootstrap_draws"]),
        "failed_draws": int(row["failed_bootstrap_draws"]),
        "series_label": f"{str(row['arm']).upper()} · {row['backend_label']}",
    }


def _contrast_display(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contrast_row_id": row["contrast_row_id"],
        "contrast": row["contrast_id"],
        "family": row["family"],
        "specification": row["specification_id"],
        "sample": row["sample_id"],
        "outcome": row["outcome_label"],
        "backend_id": row["backend_id"],
        "sentiment_backend": row["backend_label"],
        "scale": row["scale_id"],
        "score_policy": _score_policy_from_specification(str(row["specification_id"])),
        "difference": float(row["point_estimate"]),
        "difference_display": _beta_display(
            float(row["point_estimate"]), difference=True
        ),
        "ci_lower": float(row["ci_lower"]),
        "ci_upper": float(row["ci_upper"]),
        "ci_95": _ci_text(row),
        "zero_relation": _zero_relation(row),
        "p_value": float(row["p_value"]),
        "p_value_display": _probability_display(float(row["p_value"])),
        "holm_adjusted_p": float(row["holm_adjusted_p"]),
        "holm_adjusted_p_display": _probability_display(
            float(row["holm_adjusted_p"]), adjusted=True
        ),
        "nobs": int(row["nobs"]),
        "successful_draws": int(row["successful_bootstrap_draws"]),
        "failed_draws": int(row["failed_bootstrap_draws"]),
        "series_label": f"{row['contrast_id']} · {row['backend_label']}",
    }


def build_canonical_artifact(model: Mapping[str, Any]) -> dict[str, Any]:
    """Build the complete canonical report payload without rendering it."""

    validated = validate_report_model(model)
    estimates_by_id = {str(row["estimate_id"]): row for row in validated["estimates"]}
    contrasts_by_id = {
        str(row["contrast_row_id"]): row for row in validated["contrasts"]
    }
    reporting = validated["reporting"]
    headline_estimate = estimates_by_id[str(reporting["headline_estimate_id"])]
    headline_contrast = contrasts_by_id[str(reporting["headline_contrast_row_id"])]
    primary_estimates = [
        _estimate_display(estimates_by_id[str(row_id)])
        for row_id in reporting["primary_estimate_ids"]
    ]
    primary_contrasts = [
        _contrast_display(contrasts_by_id[str(row_id)])
        for row_id in reporting["primary_contrast_row_ids"]
    ]
    background_contrasts = [
        _contrast_display(contrasts_by_id[str(row_id)])
        for row_id in reporting["background_contrast_row_ids"]
    ]
    reference_contrasts = [
        _contrast_display(contrasts_by_id[str(row_id)])
        for row_id in reporting["reference_contrast_row_ids"]
    ]
    backend_audit = [
        {
            "backend_id": row["backend_id"],
            "backend": row["backend_label"],
            "role": row["role"],
            "regression_eligible": row["regression_eligible"],
            "reference_scale_sd": float(row["reference_scale_sd"]),
            "reference_scale_sd_display": _reference_sd_display(
                float(row["reference_scale_sd"])
            ),
            "exclusion_reason": row["exclusion_reason"] or "not_excluded",
        }
        for row in validated["backend_rows"]
    ]
    all_estimates = [_estimate_display(row) for row in validated["estimates"]]
    all_contrasts = [_contrast_display(row) for row in validated["contrasts"]]
    view_order = list(EXPECTED_VIEW_NOBS)
    five_view_primary_contrasts = [
        _contrast_display(row)
        for backend_id in REGRESSION_BACKEND_IDS
        for view_id in view_order
        for row in validated["contrasts"]
        if row["backend_id"] == backend_id
        and row["sample_id"] == view_id
        and row["outcome_id"] == PRIMARY_OUTCOME_ID
        and row["scale_id"] == PRIMARY_SCALE_ID
        and row["contrast_id"] == PRIMARY_CONTRAST_ID
        and _score_policy_from_specification(str(row["specification_id"]))
        == "shared_complete_core8"
    ]
    expected_five_view_keys = {
        (backend_id, view_id)
        for backend_id in REGRESSION_BACKEND_IDS
        for view_id in view_order
    }
    observed_five_view_keys = {
        (str(row["backend_id"]), str(row["sample"]))
        for row in five_view_primary_contrasts
    }
    if (
        len(five_view_primary_contrasts) != len(expected_five_view_keys)
        or observed_five_view_keys != expected_five_view_keys
    ):
        raise BetaTechnicalReportError(
            "five-view primary CHK3-minus-CHK1 summary coverage drift"
        )
    estimate_pages = _paged(all_estimates)
    contrast_pages = _paged(all_contrasts)

    source_bindings = validated["source_bindings"]
    sources = [
        _source(
            source_id="estimator_manifest_source",
            label="Sealed CHK3 beta estimator manifest",
            binding=source_bindings["estimator_manifest"],
            executed_at=validated["generated_at_utc"],
        ),
        _source(
            source_id="estimates_source",
            label="Sealed coefficient estimates",
            binding=source_bindings["estimates"],
            executed_at=validated["generated_at_utc"],
        ),
        _source(
            source_id="contrasts_source",
            label="Sealed paired coefficient contrasts",
            binding=source_bindings["contrasts"],
            executed_at=validated["generated_at_utc"],
        ),
        _source(
            source_id="diagnostics_source",
            label="Sealed estimator diagnostics and sample audit",
            binding=source_bindings["diagnostics"],
            executed_at=validated["generated_at_utc"],
        ),
        _source(
            source_id="market_panel_source",
            label="Sealed Fed Minutes release and Treasury market panel",
            binding=source_bindings["market_panel_manifest"],
            executed_at=validated["generated_at_utc"],
        ),
        _source(
            source_id="sentiment_suite_source",
            label="Sealed Core8 meeting-level sentiment suite",
            binding=source_bindings["sentiment_suite_manifest"],
            executed_at=validated["generated_at_utc"],
        ),
        _source(
            source_id="generation_suite_source",
            label="Sealed CHK0/CHK1/CHK3 stochastic generation suite",
            binding=source_bindings["generation_suite_manifest"],
            executed_at=validated["generated_at_utc"],
        ),
    ]

    headline = {
        "headline_beta": float(headline_estimate["point_estimate"]),
        "headline_beta_display": _beta_display(
            float(headline_estimate["point_estimate"])
        ),
        "headline_beta_ci_lower": float(headline_estimate["ci_lower"]),
        "headline_beta_ci_upper": float(headline_estimate["ci_upper"]),
        "headline_beta_ci_display": _ci_text(headline_estimate),
        "headline_difference": float(headline_contrast["point_estimate"]),
        "headline_difference_display": _beta_display(
            float(headline_contrast["point_estimate"]), difference=True
        ),
        "headline_difference_ci_lower": float(headline_contrast["ci_lower"]),
        "headline_difference_ci_upper": float(headline_contrast["ci_upper"]),
        "headline_difference_ci_display": _ci_text(headline_contrast),
        "headline_nobs": int(headline_estimate["nobs"]),
        "bootstrap_draws": EXPECTED_BOOTSTRAP_DRAWS,
    }
    study = validated["study"]
    methodology = validated["methodology"]
    title = str(study["title"])
    summary_context = (
        "## The primary estimate is descriptive association evidence, not market impact\n\n"
        f"{NON_CAUSAL_DISCLOSURE}"
    )
    summary_estimate = (
        f"For **{headline_estimate['arm'].upper()}** under the frozen primary "
        f"{headline_estimate['outcome_label']} / {headline_estimate['backend_label']} "
        f"specification, the sentiment coefficient is "
        f"**{_signed_text(float(headline_estimate['point_estimate']))}** "
        f"with a 95% paired-bootstrap interval of **{_ci_text(headline_estimate)}** "
        f"(N={int(headline_estimate['nobs'])})."
    )
    summary_contrast = (
        f"The primary **{headline_contrast['contrast_id']}** coefficient difference "
        f"is **{_signed_text(float(headline_contrast['point_estimate']))}**, 95% CI "
        f"**{_ci_text(headline_contrast)}**; that interval "
        f"{_zero_relation(headline_contrast)}."
    )
    deduplicated_limitations = _deduplicated_limitations(list(validated["limitations"]))
    estimator_limitations = [
        item
        for item in deduplicated_limitations
        if item
        not in {REFERENCE_DISCLOSURE, EVENT_WINDOW_DISCLOSURE, LEXICON_DISCLOSURE}
    ]
    sensitivity_text = (
        "\n".join(f"- {item}" for item in methodology["sensitivity_designs"])
        or "- No sensitivity design was reported."
    )

    cards = [
        {
            "id": "headline_beta_card",
            "description": (
                f"{headline_estimate['arm'].upper()} beta; 95% CI endpoints are "
                "shown as contextual values and in the exact estimate table."
            ),
            "dataset": "headline",
            "sourceId": "estimates_source",
            "metrics": [
                {
                    "label": "Primary beta",
                    "field": "headline_beta_display",
                },
                {
                    "label": "95% CI",
                    "field": "headline_beta_ci_display",
                },
            ],
        },
        {
            "id": "headline_contrast_card",
            "description": (
                f"Primary paired coefficient difference: "
                f"{headline_contrast['contrast_id']}."
            ),
            "dataset": "headline",
            "sourceId": "contrasts_source",
            "metrics": [
                {
                    "label": "Primary beta difference",
                    "field": "headline_difference_display",
                },
                {
                    "label": "95% CI",
                    "field": "headline_difference_ci_display",
                },
            ],
        },
        {
            "id": "headline_sample_card",
            "description": "Same-sample regression observations for the headline estimate.",
            "dataset": "headline",
            "sourceId": "diagnostics_source",
            "metrics": [
                {"label": "Regression N", "field": "headline_nobs", "format": "number"}
            ],
        },
        {
            "id": "bootstrap_draws_card",
            "description": "Shared time-aware paired bootstrap draws.",
            "dataset": "headline",
            "sourceId": "estimator_manifest_source",
            "metrics": [
                {
                    "label": "Bootstrap draws",
                    "field": "bootstrap_draws",
                    "format": "number",
                }
            ],
        },
    ]
    charts = [
        {
            "id": "primary_beta_chart",
            "title": "Primary coefficient estimates by text arm",
            "subtitle": (
                "Point estimates under the frozen primary sample, outcome, and "
                "sentiment backend; exact 95% CIs are in the following table."
            ),
            "intent": "comparison",
            "question": "How do primary sentiment coefficients compare across text arms?",
            "rationale": (
                "A categorical bar chart provides a compact comparison of the "
                "model-arm point estimates; the adjacent audit table preserves "
                "the paired-bootstrap confidence intervals exactly."
            ),
            "type": "bar",
            "dataset": "primary_estimates",
            "sourceId": "estimates_source",
            "encodings": {
                "x": {"field": "series_label", "type": "nominal", "label": "Text arm"},
                "y": {
                    "field": "beta",
                    "type": "quantitative",
                    "label": "Beta estimate",
                },
                "tooltip": [
                    {
                        "field": "beta_display",
                        "type": "nominal",
                        "label": "Exact beta",
                    },
                    {
                        "field": "ci_95",
                        "type": "nominal",
                        "label": "Exact 95% CI",
                    },
                    {"field": "nobs", "type": "quantitative", "label": "Regression N"},
                ],
            },
            "referenceLines": [
                {
                    "axis": "y",
                    "value": 0,
                    "label": "Zero",
                    "color": "neutral",
                    "lineStyle": "dashed",
                }
            ],
            "palette": {"kind": "categorical", "name": "blue"},
            "labels": {"values": "auto"},
            "valueFormat": "number",
            "layout": "full",
        },
        {
            "id": "primary_contrast_chart",
            "title": "Primary paired coefficient contrasts",
            "subtitle": (
                "Point differences with a zero reference; exact 95% CIs and "
                "Holm-adjusted p-values are in the following table."
            ),
            "intent": "comparison",
            "question": "What are the paired primary beta differences between arms?",
            "rationale": (
                "A zero-centered categorical comparison makes direction visible "
                "without treating interval inclusion as a causal result."
            ),
            "type": "bar",
            "dataset": "primary_contrasts",
            "sourceId": "contrasts_source",
            "encodings": {
                "x": {
                    "field": "series_label",
                    "type": "nominal",
                    "label": "Paired contrast",
                },
                "y": {
                    "field": "difference",
                    "type": "quantitative",
                    "label": "Beta difference",
                },
                "tooltip": [
                    {
                        "field": "difference_display",
                        "type": "nominal",
                        "label": "Exact beta difference",
                    },
                    {
                        "field": "ci_95",
                        "type": "nominal",
                        "label": "Exact 95% CI",
                    },
                    {
                        "field": "holm_adjusted_p_display",
                        "type": "nominal",
                        "label": "Exact Holm-adjusted descriptive p",
                    },
                ],
            },
            "referenceLines": [
                {
                    "axis": "y",
                    "value": 0,
                    "label": "Zero",
                    "color": "neutral",
                    "lineStyle": "dashed",
                }
            ],
            "palette": {"kind": "categorical", "name": "purple"},
            "labels": {"values": "auto"},
            "valueFormat": "number",
            "layout": "full",
        },
    ]

    estimate_columns = [
        {"field": "specification", "label": "Specification", "type": "text"},
        {"field": "sample", "label": "Sample", "type": "text"},
        {"field": "outcome", "label": "Outcome", "type": "text"},
        {"field": "sentiment_backend", "label": "Sentiment backend", "type": "text"},
        {"field": "scale", "label": "Sentiment scale", "type": "text"},
        {"field": "arm", "label": "Text arm", "type": "text"},
        {"field": "beta_display", "label": "Beta", "type": "text"},
        {"field": "ci_95", "label": "95% CI", "type": "text"},
        {"field": "nobs", "label": "N", "format": "number"},
        {"field": "successful_draws", "label": "Successful draws", "format": "number"},
        {"field": "failed_draws", "label": "Failed draws", "format": "number"},
    ]
    contrast_columns = [
        {"field": "contrast", "label": "Contrast", "type": "text"},
        {"field": "specification", "label": "Specification", "type": "text"},
        {"field": "sample", "label": "Sample", "type": "text"},
        {"field": "outcome", "label": "Outcome", "type": "text"},
        {"field": "sentiment_backend", "label": "Sentiment backend", "type": "text"},
        {"field": "scale", "label": "Sentiment scale", "type": "text"},
        {
            "field": "difference_display",
            "label": "Beta difference",
            "type": "text",
        },
        {"field": "ci_95", "label": "95% CI", "type": "text"},
        {"field": "zero_relation", "label": "CI vs zero", "type": "text"},
        {
            "field": "p_value_display",
            "label": "Descriptive two-sided sign-tail probability",
            "type": "text",
        },
        {
            "field": "holm_adjusted_p_display",
            "label": "Holm-adjusted descriptive p",
            "type": "text",
        },
        {"field": "nobs", "label": "N", "format": "number"},
        {"field": "failed_draws", "label": "Failed draws", "format": "number"},
    ]
    five_view_columns = [
        {"field": "sample", "label": "Analysis view", "type": "text"},
        {
            "field": "sentiment_backend",
            "label": "Sentiment backend",
            "type": "text",
        },
        {"field": "score_policy", "label": "Score policy", "type": "text"},
        {"field": "scale", "label": "Sentiment scale", "type": "text"},
        {
            "field": "difference_display",
            "label": "CHK3−CHK1 beta difference",
            "type": "text",
        },
        {"field": "ci_95", "label": "95% CI", "type": "text"},
        {"field": "zero_relation", "label": "CI vs zero", "type": "text"},
        {
            "field": "p_value_display",
            "label": "Descriptive sign-tail p",
            "type": "text",
        },
        {
            "field": "holm_adjusted_p_display",
            "label": "Holm-adjusted descriptive p",
            "type": "text",
        },
        {"field": "nobs", "label": "N", "format": "number"},
        {"field": "failed_draws", "label": "Failed draws", "format": "number"},
    ]
    tables = [
        {
            "id": "primary_estimates_table",
            "title": "Primary coefficient estimates and 95% intervals",
            "subtitle": "Same sample, outcome, and regression design across text arms.",
            "dataset": "primary_estimates",
            "sourceId": "estimates_source",
            "defaultSort": {"field": "arm", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": estimate_columns,
        },
        {
            "id": "primary_contrasts_table",
            "title": "Primary paired contrasts",
            "subtitle": "Paired coefficient differences; family-wise inference uses Holm adjustment.",
            "dataset": "primary_contrasts",
            "sourceId": "contrasts_source",
            "defaultSort": {"field": "contrast", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": contrast_columns,
        },
        {
            "id": "background_contrasts_table",
            "title": "Background paired contrasts",
            "subtitle": "CHK1−CHK0 and CHK3−CHK0 comparisons are reported separately from the primary family.",
            "dataset": "background_contrasts",
            "sourceId": "contrasts_source",
            "defaultSort": {"field": "contrast", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": contrast_columns,
        },
        {
            "id": "reference_contrasts_table",
            "title": "Contextual contrasts with the synthetic reference arm",
            "subtitle": (
                "The reference is a deterministic synthetic teacher target, not "
                "official FOMC Minutes or an observed market benchmark."
            ),
            "dataset": "reference_contrasts",
            "sourceId": "contrasts_source",
            "defaultSort": {"field": "contrast", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": contrast_columns,
        },
        {
            "id": "backend_eligibility_table",
            "title": "Sentiment backend eligibility",
            "subtitle": (
                "Only positive-variance reference scales enter regression; "
                "FinBERT remains a non-pooled robustness construct."
            ),
            "dataset": "backend_audit",
            "sourceId": "sentiment_suite_source",
            "defaultSort": {"field": "role", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "backend", "label": "Backend", "type": "text"},
                {"field": "role", "label": "Role", "type": "text"},
                {
                    "field": "regression_eligible",
                    "label": "Regression eligible",
                    "type": "boolean",
                },
                {
                    "field": "reference_scale_sd_display",
                    "label": "Pre-external reference SD (exact)",
                    "type": "text",
                },
                {
                    "field": "exclusion_reason",
                    "label": "Exclusion reason",
                    "type": "text",
                },
            ],
        },
        {
            "id": "sample_audit_table",
            "title": "Analysis samples and exclusions",
            "subtitle": "Meeting coverage and regression denominator after frozen market-data exclusions and lags.",
            "dataset": "sample_rows",
            "sourceId": "diagnostics_source",
            "defaultSort": {"field": "meeting_date_start", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "label", "label": "Sample", "type": "text"},
                {"field": "role", "label": "Role", "type": "text"},
                {"field": "meeting_rows", "label": "Market rows", "format": "number"},
                {
                    "field": "regression_nobs",
                    "label": "Regression N",
                    "format": "number",
                },
                {
                    "field": "meeting_date_start",
                    "label": "Start meeting",
                    "type": "text",
                },
                {"field": "meeting_date_end", "label": "End meeting", "type": "text"},
                {"field": "exclusion_count", "label": "Excluded", "format": "number"},
            ],
        },
        {
            "id": "five_view_primary_contrasts_table",
            "title": "CHK3−CHK1 primary-outcome contrast across all five views",
            "subtitle": (
                "Ten directly reviewable rows: five views × DistilBERT primary "
                "and FinBERT robustness, using shared-complete Core8 scores and "
                "the frozen pre-external reference-SD scale."
            ),
            "dataset": "five_view_primary_contrasts",
            "sourceId": "contrasts_source",
            "defaultSort": {"field": "sample", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": five_view_columns,
        },
    ]
    for page_index, page in enumerate(estimate_pages, start=1):
        tables.append(
            {
                "id": f"all_estimates_page_{page_index:02d}_table",
                "title": (
                    f"All coefficient estimates — page {page_index} of "
                    f"{len(estimate_pages)}"
                ),
                "subtitle": (
                    "Complete sealed estimate inventory, paged at no more than "
                    f"{APPENDIX_PAGE_SIZE} rows so the portable report displays "
                    "every result."
                ),
                "dataset": f"all_estimates_page_{page_index:02d}",
                "sourceId": "estimates_source",
                "density": "dense",
                "layout": "full",
                "columns": estimate_columns,
            }
        )
    for page_index, page in enumerate(contrast_pages, start=1):
        tables.append(
            {
                "id": f"all_contrasts_page_{page_index:02d}_table",
                "title": (
                    f"All paired coefficient contrasts — page {page_index} of "
                    f"{len(contrast_pages)}"
                ),
                "subtitle": (
                    "Complete sealed contrast inventory, paged at no more than "
                    f"{APPENDIX_PAGE_SIZE} rows so the portable report displays "
                    "every result."
                ),
                "dataset": f"all_contrasts_page_{page_index:02d}",
                "sourceId": "contrasts_source",
                "density": "dense",
                "layout": "full",
                "columns": contrast_columns,
            }
        )

    appendix_blocks = [
        *[
            {
                "id": f"all_estimates_page_{page_index:02d}_table_block",
                "type": "table",
                "tableId": f"all_estimates_page_{page_index:02d}_table",
                "layout": "full",
            }
            for page_index in range(1, len(estimate_pages) + 1)
        ],
        *[
            {
                "id": f"all_contrasts_page_{page_index:02d}_table_block",
                "type": "table",
                "tableId": f"all_contrasts_page_{page_index:02d}_table",
                "layout": "full",
            }
            for page_index in range(1, len(contrast_pages) + 1)
        ],
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary_context",
            "type": "markdown",
            "sourceId": "estimator_manifest_source",
            "body": summary_context,
        },
        {
            "id": "technical_summary_estimate",
            "type": "markdown",
            "sourceId": "estimates_source",
            "body": summary_estimate,
        },
        {
            "id": "technical_summary_contrast",
            "type": "markdown",
            "sourceId": "contrasts_source",
            "body": summary_contrast,
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [card["id"] for card in cards],
        },
        {
            "id": "primary_estimate_finding",
            "type": "markdown",
            "sourceId": "estimates_source",
            "body": (
                "## Primary beta estimates compare association preservation on one common sample\n\n"
                "The chart compares point estimates only. Read the exact 95% "
                "paired-bootstrap intervals and draw failures in the table "
                "immediately below; a larger or more positive coefficient is not, "
                "by itself, evidence that one synthetic text caused a market move."
            ),
        },
        {
            "id": "primary_beta_chart_block",
            "type": "chart",
            "chartId": "primary_beta_chart",
            "layout": "full",
        },
        {
            "id": "primary_estimates_table_block",
            "type": "table",
            "tableId": "primary_estimates_table",
            "layout": "full",
        },
        {
            "id": "primary_contrast_finding",
            "type": "markdown",
            "sourceId": "contrasts_source",
            "body": (
                "## Paired contrasts are the relevant model comparison\n\n"
                "The primary family compares CHK3 with its CHK1 experimental "
                "baseline using shared time-block and replicate bootstrap indices. "
                "The zero line marks no coefficient difference. The exact interval "
                "and descriptive sign-tail probability characterize uncertainty; "
                "they are not a null-centered formal hypothesis test."
            ),
        },
        {
            "id": "primary_contrast_chart_block",
            "type": "chart",
            "chartId": "primary_contrast_chart",
            "layout": "full",
        },
        {
            "id": "primary_contrasts_table_block",
            "type": "table",
            "tableId": "primary_contrasts_table",
            "layout": "full",
        },
        {
            "id": "background_finding",
            "type": "markdown",
            "sourceId": "contrasts_source",
            "body": (
                "## CHK0 comparisons are background context, not the experimental estimand\n\n"
                "CHK1−CHK0 and CHK3−CHK0 locate the two trained systems relative "
                "to the base model. They remain a separate multiplicity family and "
                "do not replace the CHK3−CHK1 primary comparison."
            ),
        },
        {
            "id": "background_contrasts_table_block",
            "type": "table",
            "tableId": "background_contrasts_table",
            "layout": "full",
        },
        {
            "id": "reference_context_finding",
            "type": "markdown",
            "sourceId": "contrasts_source",
            "body": (
                "## Synthetic-reference contrasts are contextual only\n\n"
                "These rows compare each model arm with the deterministic synthetic "
                "teacher target. A smaller absolute coefficient gap may be described "
                "as closer to that synthetic reference association, but it is not "
                "agreement with official FOMC Minutes, not factual validation, and "
                "not causal evidence. For `reference_distance_gain_chk3_vs_chk1`, "
                "a positive value means CHK3 is closer than CHK1 to the synthetic "
                "reference coefficient; that derived contrast is calculated within "
                "each shared bootstrap draw."
            ),
        },
        {
            "id": "reference_contrasts_table_block",
            "type": "table",
            "tableId": "reference_contrasts_table",
            "layout": "full",
        },
        {
            "id": "scope_market_definitions",
            "type": "markdown",
            "sourceId": "market_panel_source",
            "body": (
                "## Scope, data, and metric definitions\n\n"
                f"Meetings span **{study['meeting_date_start']} to "
                f"{study['meeting_date_end']}**; observed Minutes releases span "
                f"**{study['release_date_start']} to {study['release_date_end']}**. "
                f"The primary outcome is {methodology['outcome_definition']}. "
                f"{EVENT_WINDOW_DISCLOSURE}"
            ),
        },
        {
            "id": "scope_sentiment_definition",
            "type": "markdown",
            "sourceId": "sentiment_suite_source",
            "body": (
                f"**Sentiment construct.** {methodology['sentiment_definition']}."
            ),
        },
        {
            "id": "scope_reference_definition",
            "type": "markdown",
            "sourceId": "generation_suite_source",
            "body": f"**Reference arm.** {REFERENCE_DISCLOSURE}",
        },
        {
            "id": "backend_eligibility",
            "type": "markdown",
            "sourceId": "sentiment_suite_source",
            "body": (
                "## Backend eligibility is determined before regression\n\n"
                "The primary backend is **DistilBERT FOMC stance**; FinBERT "
                "financial valence is shown only as a non-pooled robustness "
                f"construct. {LEXICON_DISCLOSURE}"
            ),
        },
        {
            "id": "backend_eligibility_table_block",
            "type": "table",
            "tableId": "backend_eligibility_table",
            "layout": "full",
        },
        {
            "id": "sample_audit_table_block",
            "type": "table",
            "tableId": "sample_audit_table",
            "layout": "full",
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "estimator_manifest_source",
            "body": (
                "## Model specification and time-aware uncertainty\n\n"
                f"**Estimand.** {methodology['estimand']}\n\n"
                f"**Regression.** `{methodology['regression_formula']}`\n\n"
                f"**Covariance.** {methodology['covariance_estimator']} with HAC "
                f"lag {methodology['hac_lag']}.\n\n"
                f"**Lag construction.** {methodology['lag_construction']}\n\n"
                f"**Seasonality.** {', '.join(methodology['seasonality_controls'])}.\n\n"
                f"**Bootstrap.** {EXPECTED_BOOTSTRAP_DRAWS:,} draws (seed "
                f"{methodology['bootstrap_seed']}); resampling unit: "
                f"{methodology['bootstrap_unit']}; within-meeting rule: "
                f"{methodology['within_meeting_resampling']}. All text arms use "
                "the same estimation rows and the same resampling indices. "
                f"**P-values.** {methodology['bootstrap_p_value']}."
            ),
        },
        {
            "id": "robustness",
            "type": "markdown",
            "sourceId": "estimator_manifest_source",
            "body": f"## Robustness checks remain separate from the primary result\n\n{sensitivity_text}",
        },
        {
            "id": "five_view_primary_contrasts_finding",
            "type": "markdown",
            "sourceId": "contrasts_source",
            "body": (
                "## The five analysis views are directly comparable in one table\n\n"
                "The following ten rows hold the outcome, score policy, and scale "
                "fixed while showing CHK3−CHK1 for all five views under both the "
                "DistilBERT primary construct and the separate FinBERT robustness "
                "construct. Intervals and sign-tail probabilities remain descriptive."
            ),
        },
        {
            "id": "five_view_primary_contrasts_table_block",
            "type": "table",
            "tableId": "five_view_primary_contrasts_table",
            "layout": "full",
        },
        {
            "id": "complete_numeric_appendix",
            "type": "markdown",
            "sourceId": "estimator_manifest_source",
            "body": (
                "## Complete numeric appendix\n\n"
                "Every sealed estimate and contrast is shown below. Pages contain "
                f"at most {APPENDIX_PAGE_SIZE} rows to avoid the portable renderer's "
                "first-15 display limit; no result is omitted from the HTML."
            ),
        },
        *appendix_blocks,
        {
            "id": "limitations_heading",
            "type": "markdown",
            "body": "## Interpretation is deliberately non-causal",
        },
        {
            "id": "limitations_estimator",
            "type": "markdown",
            "sourceId": "estimator_manifest_source",
            "body": "\n".join(f"- {item}" for item in estimator_limitations),
        },
        {
            "id": "limitations_reference",
            "type": "markdown",
            "sourceId": "generation_suite_source",
            "body": f"- {REFERENCE_DISCLOSURE}",
        },
        {
            "id": "limitations_market_window",
            "type": "markdown",
            "sourceId": "market_panel_source",
            "body": f"- {EVENT_WINDOW_DISCLOSURE}",
        },
        {
            "id": "limitations_lexicon",
            "type": "markdown",
            "sourceId": "sentiment_suite_source",
            "body": f"- {LEXICON_DISCLOSURE}",
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Recommended next steps\n\n"
                "1. Treat the result as an exploratory preservation diagnostic, "
                "not evidence of market impact by generated text.\n"
                "2. Review outcome-window and sentiment-backend sensitivities "
                "before using any coefficient direction in the thesis narrative.\n"
                "3. Seek a checkpoint-independent confirmation sample or rerun the "
                "full selection-and-evaluation pipeline under nested resampling."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "Do the primary coefficient and CHK3−CHK1 difference persist "
                "across Treasury maturities, delayed windows, placebo windows, and "
                "independent sentiment constructs? How much of any difference is "
                "concentrated in checkpoint-selection-exposed meetings?"
            ),
        },
        {
            "id": "provenance",
            "type": "markdown",
            "sourceId": "estimator_manifest_source",
            "body": (
                "## Provenance and reproducibility\n\n"
                "Every displayed coefficient, interval, contrast, denominator, and "
                "diagnostic is copied from the sealed estimator bundle. The report "
                "does not refit regressions or rerun bootstrap draws. Source actions "
                "on the cards, charts, and tables expose the exact bound artifacts."
            ),
        },
        {
            "id": "generation_lineage",
            "type": "markdown",
            "sourceId": "generation_suite_source",
            "body": (
                "### Generation lineage\n\n"
                "The CHK0, CHK1, and CHK3 text arms trace to the sealed stochastic "
                "generation suite bound by the estimator. This lineage establishes "
                "input identity and coverage; it does not turn synthetic text into "
                "a historical market exposure."
            ),
        },
    ]

    snapshot_datasets = {
        "headline": [headline],
        "primary_estimates": primary_estimates,
        "primary_contrasts": primary_contrasts,
        "background_contrasts": background_contrasts,
        "reference_contrasts": reference_contrasts,
        "five_view_primary_contrasts": five_view_primary_contrasts,
        "backend_audit": backend_audit,
        "sample_rows": copy.deepcopy(validated["sample_rows"]),
    }
    snapshot_datasets.update(
        {
            f"all_estimates_page_{page_index:02d}": page
            for page_index, page in enumerate(estimate_pages, start=1)
        }
    )
    snapshot_datasets.update(
        {
            f"all_contrasts_page_{page_index:02d}": page
            for page_index, page in enumerate(contrast_pages, start=1)
        }
    )

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": (
                "Technical report of CHK0/CHK1/CHK3 and synthetic-reference "
                "sentiment--Treasury association estimates."
            ),
            "generatedAt": validated["generated_at_utc"],
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": validated["generated_at_utc"],
            "status": "ready",
            "datasets": snapshot_datasets,
        },
        "sources": copy.deepcopy(sources),
    }


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _resolve_node(node_bin: Path | None) -> Path:
    if node_bin is not None:
        candidate = node_bin.expanduser().resolve()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        raise BetaTechnicalReportError(f"Node executable is unavailable: {candidate}")
    discovered = shutil.which("node")
    candidates = ([Path(discovered)] if discovered else []) + list(
        DEFAULT_NODE_CANDIDATES
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    raise BetaTechnicalReportError(
        "no executable Node runtime found for report packaging"
    )


def _resolve_packager(plugin_root: Path | None) -> tuple[Path, Path]:
    root = (plugin_root or DEFAULT_PLUGIN_ROOT).expanduser().resolve()
    script = root / PACKAGER_RELATIVE
    if not script.is_file() or script.is_symlink():
        raise BetaTechnicalReportError(
            f"packaged report builder is unavailable: {script}"
        )
    package = root / "package.json"
    if not package.is_file() or package.is_symlink():
        raise BetaTechnicalReportError(
            f"report plugin package is unavailable: {package}"
        )
    return root, script


def _package_report(
    *,
    artifact_path: Path,
    report_path: Path,
    node_bin: Path | None,
    plugin_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    node = _resolve_node(node_bin)
    plugin, script = _resolve_packager(plugin_root)
    completed = subprocess.run(
        [
            str(node),
            str(script),
            "--input",
            str(artifact_path),
            "--output",
            str(report_path),
            "--timeout-ms",
            "30000",
        ],
        cwd=plugin,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        message = detail[-1] if detail else f"exit {completed.returncode}"
        raise BetaTechnicalReportError(f"portable report packaging failed: {message}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise BetaTechnicalReportError(
            "portable report builder emitted an ambiguous receipt"
        )
    try:
        receipt = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise BetaTechnicalReportError(
            "portable report receipt is invalid JSON"
        ) from exc
    stages = receipt.get("stages") if isinstance(receipt, Mapping) else None
    if (
        receipt.get("ok") is not True
        or not isinstance(stages, Mapping)
        or stages.get("validation") != "passed"
        or stages.get("package") != "passed"
        or stages.get("verification") not in {"passed", "structural_only"}
        or not report_path.is_file()
    ):
        raise BetaTechnicalReportError("portable report receipt did not pass delivery")
    return copy.deepcopy(dict(receipt)), {
        "node": _binding(node),
        "packager_script": _binding(script),
        "plugin_package": _binding(plugin / "package.json"),
    }


def _rename_noreplace(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise BetaTechnicalReportError(
            f"report directory already exists: {destination}"
        )
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError):
        renameat2 = None
    if renameat2 is not None:
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise BetaTechnicalReportError(
                f"report directory already exists: {destination}"
            )
        if error_number not in {errno.ENOSYS, errno.EINVAL}:
            raise OSError(error_number, os.strerror(error_number), str(destination))
    if destination.exists() or destination.is_symlink():
        raise BetaTechnicalReportError(
            f"report directory already exists: {destination}"
        )
    os.rename(source, destination)


def _seal_permissions(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise BetaTechnicalReportError(f"report output contains a symlink: {path}")
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _normalize_estimator(loaded: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the authoritative estimator loader result to the report model.

    The exact mapping is intentionally kept in one function.  It is completed
    only against the frozen estimator row contract; presentation code never
    guesses alternate field names.
    """

    adapter = loaded.get("report_model")
    if not isinstance(adapter, Mapping):
        raise BetaTechnicalReportError(
            "authoritative estimator loader did not provide its frozen report_model"
        )
    return validate_report_model(adapter)


def load_report_model(
    estimator_manifest: Path, estimator_manifest_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use the estimator's deep loader, then validate the report adapter model."""

    manifest_path = estimator_manifest.expanduser().resolve()
    observed = _binding(manifest_path)
    if observed["sha256"] != estimator_manifest_sha256:
        raise BetaTechnicalReportError("estimator manifest external SHA-256 drift")
    _validate_estimator_runtime(manifest_path)
    try:
        from jobs.eval import estimate_chk3_beta_time_block_bootstrap_v1 as estimator
    except ImportError as exc:
        raise BetaTechnicalReportError(
            "authoritative estimator module is unavailable"
        ) from exc
    try:
        loaded = estimator.load_and_validate_estimator(manifest_path)
    except Exception as exc:
        raise BetaTechnicalReportError(
            f"authoritative estimator validation failed: {exc}"
        ) from exc
    if not isinstance(loaded, Mapping):
        raise BetaTechnicalReportError(
            "authoritative estimator loader returned no bundle"
        )
    manifest = loaded.get("manifest")
    binding = loaded.get("manifest_binding")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != ESTIMATOR_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or not isinstance(binding, Mapping)
        or binding.get("sha256") != estimator_manifest_sha256
    ):
        raise BetaTechnicalReportError("estimator manifest/report identity drift")
    model = _normalize_estimator(loaded)
    if model["source_bindings"]["estimator_manifest"] != binding:
        raise BetaTechnicalReportError("normalized report model lost estimator binding")
    return model, copy.deepcopy(dict(binding))


def render_and_seal_report(
    *,
    estimator_manifest: Path,
    estimator_manifest_sha256: str,
    output_dir: Path,
    node_bin: Path | None = None,
    plugin_root: Path | None = None,
) -> dict[str, Any]:
    """Publish create-only artifact.json, packaged report.html, and manifest."""

    output = output_dir.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise BetaTechnicalReportError(f"report directory already exists: {output}")
    report_model, estimator_binding = load_report_model(
        estimator_manifest, estimator_manifest_sha256
    )
    artifact = build_canonical_artifact(report_model)
    artifact_text = _canonical(artifact)
    for required in (
        NON_CAUSAL_DISCLOSURE,
        REFERENCE_DISCLOSURE,
        EVENT_WINDOW_DISCLOSURE,
        POST_SELECTION_DISCLOSURE,
        LEXICON_DISCLOSURE,
    ):
        if required not in artifact_text:
            raise BetaTechnicalReportError(
                "canonical artifact lost a required disclosure"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    ).resolve()
    published = False
    try:
        artifact_path = staging / "artifact.json"
        report_path = staging / "report.html"
        _write_new_json(artifact_path, artifact)
        receipt, builder_bindings = _package_report(
            artifact_path=artifact_path,
            report_path=report_path,
            node_bin=node_bin,
            plugin_root=plugin_root,
        )
        refreshed_model, refreshed_binding = load_report_model(
            estimator_manifest, estimator_manifest_sha256
        )
        if refreshed_binding != estimator_binding or refreshed_model != report_model:
            raise BetaTechnicalReportError("estimator bundle changed during reporting")
        if build_canonical_artifact(refreshed_model) != artifact:
            raise BetaTechnicalReportError(
                "canonical report artifact is not reproducible"
            )

        artifact_final = output / "artifact.json"
        report_final = output / "report.html"
        manifest_final = output / "manifest.json"
        renderer_binding = _binding(Path(__file__))
        manifest = seal_manifest(
            {
                "schema_version": REPORT_MANIFEST_SCHEMA,
                "status": "complete",
                "immutable": True,
                "report_id": REPORT_ID,
                "created_at_utc": report_model["generated_at_utc"],
                "operation": "render_only_no_regression_or_bootstrap_recomputation",
                "estimator_manifest": copy.deepcopy(estimator_binding),
                "artifacts": {
                    "canonical_artifact": {
                        **_intended_binding(artifact_path, artifact_final),
                        "media_type": "application/json",
                        "contract": REPORT_ARTIFACT_CONTRACT,
                    },
                    "portable_report": {
                        **_intended_binding(report_path, report_final),
                        "media_type": "text/html; charset=utf-8",
                        "self_contained": True,
                        "network_dependencies": False,
                    },
                },
                "report_builder": builder_bindings,
                "delivery_receipt": receipt,
                "renderer": renderer_binding,
                "runtime": _report_runtime(),
                "report_contract": {
                    "audience": "technical",
                    "delivery_mode": "html",
                    "canonical_artifact_packaged_once": True,
                    "snapshot_status": "ready",
                    "bootstrap_draws": EXPECTED_BOOTSTRAP_DRAWS,
                    "beta_estimates_and_95pct_ci_displayed": True,
                    "primary_background_and_reference_contrasts_separated": True,
                    "sample_and_provenance_displayed": True,
                    "non_causal_language_displayed": True,
                    "reference_not_official_language_displayed": True,
                    "lexicon_zero_variance_exclusion_displayed": True,
                    "chart_contract": {
                        "primary_beta": "categorical point-estimate bar; exact 95% CI in adjacent table",
                        "primary_contrasts": "zero-referenced point-difference bar; exact 95% CI in adjacent table",
                        "interval_mark_omission_reason": (
                            "The packaged canonical chart contract has no native "
                            "error-bar mark; intervals are displayed exactly in "
                            "the adjacent source-backed tables."
                        ),
                    },
                },
                "limitations": [
                    NON_CAUSAL_DISCLOSURE,
                    REFERENCE_DISCLOSURE,
                    EVENT_WINDOW_DISCLOSURE,
                    POST_SELECTION_DISCLOSURE,
                    LEXICON_DISCLOSURE,
                    (
                        "Browser-level chart QA was unavailable; semantic chart "
                        "tables remain present."
                        if receipt["stages"]["verification"] == "structural_only"
                        else "Browser-level desktop and narrow-width QA passed."
                    ),
                ],
            }
        )
        _write_new_json(staging / "manifest.json", manifest)
        validate_manifest_integrity(manifest)
        _seal_permissions(staging)
        _fsync_directory(staging)
        _rename_noreplace(staging, output)
        published = True
        _fsync_directory(output.parent)

        final_manifest = json.loads(manifest_final.read_text(encoding="utf-8"))
        validate_manifest_integrity(final_manifest)
        for name, binding in final_manifest["artifacts"].items():
            observed = _binding(Path(str(binding["path"])))
            if any(
                observed[field] != binding[field]
                for field in ("path", "bytes", "sha256")
            ):
                raise BetaTechnicalReportError(f"published {name} binding drift")
        if output.stat().st_mode & 0o222:
            raise BetaTechnicalReportError("published report root is writable")
        return final_manifest
    finally:
        if not published and staging.exists():
            for path in [*staging.rglob("*"), staging]:
                try:
                    path.chmod(0o755 if path.is_dir() else 0o600)
                except OSError:
                    pass
            shutil.rmtree(staging, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    describe = subparsers.add_parser(
        "describe-contract", help="print the result-free report adapter contract"
    )
    describe.set_defaults(command="describe-contract")
    render = subparsers.add_parser(
        "render", help="deep-validate a complete estimator and publish the report"
    )
    render.add_argument(
        "--estimator-manifest", type=Path, default=DEFAULT_ESTIMATOR_MANIFEST
    )
    render.add_argument("--estimator-manifest-sha256", required=True)
    render.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    render.add_argument("--node-bin", type=Path)
    render.add_argument("--plugin-root", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "describe-contract":
        print(json.dumps(_normalized_contract(), ensure_ascii=False, indent=2))
        return 0
    try:
        manifest = render_and_seal_report(
            estimator_manifest=args.estimator_manifest,
            estimator_manifest_sha256=args.estimator_manifest_sha256,
            output_dir=args.output_dir,
            node_bin=args.node_bin,
            plugin_root=args.plugin_root,
        )
    except (
        BetaTechnicalReportError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        _canonical(
            {
                "status": manifest["status"],
                "report": manifest["artifacts"]["portable_report"]["path"],
                "report_sha256": manifest["artifacts"]["portable_report"]["sha256"],
                "manifest_payload_sha256": manifest["integrity"]["payload_sha256"],
                "verification": manifest["delivery_receipt"]["stages"]["verification"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
