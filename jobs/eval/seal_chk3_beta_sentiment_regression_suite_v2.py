"""Seal the regression-facing DistilBERT/FinBERT sentiment suite v2."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from jobs.eval import (
    eval_chk3_beta_core8_sentiment_neural_stochastic_schedule_v2 as neural_v2,
)
from jobs.eval import (
    eval_chk3_beta_core8_sentiment_stochastic_schedule_v1 as sentiment_v1,
)
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


IMPLEMENTATION_PATH = Path(__file__).resolve()
IMPLEMENTATION_SHA256_AT_IMPORT = sha256_file(IMPLEMENTATION_PATH)
SCHEMA_VERSION = "chk3-beta-core8-sentiment-regression-suite-v2"
DEFAULT_OUTPUT = sentiment_v1.DEFAULT_OUTPUT_ROOT / "suite_manifest.regression_v2.json"
LEGACY_SUITE_ABSENCE_REASON = (
    "v1_neural_scoring_failed_before_first_row_due_transformers_5_5_"
    "tokenizer_api_incompatibility"
)
V1_FAILURE_DIAGNOSTIC = sentiment_v1.DEFAULT_OUTPUT_ROOT / (
    "non_artifact_diagnostics/neural_v1_transformers_5_5_failure.json"
)


class SentimentRegressionSuiteV2Error(RuntimeError):
    """The regression-facing sentiment suite v2 is invalid."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _binding(path: Path, *, payload_sha256: str | None = None) -> dict[str, Any]:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise SentimentRegressionSuiteV2Error(f"bound path is symlink: {unresolved}")
    path = unresolved.resolve()
    if not path.is_file():
        raise SentimentRegressionSuiteV2Error(f"bound file is missing: {path}")
    result = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if payload_sha256 is not None:
        result["payload_sha256"] = payload_sha256
    return result


def _implementation_binding() -> dict[str, Any]:
    binding = _binding(IMPLEMENTATION_PATH)
    if binding["sha256"] != IMPLEMENTATION_SHA256_AT_IMPORT:
        raise SentimentRegressionSuiteV2Error(
            "suite implementation changed after this process imported it"
        )
    return binding


def _lexicon_exclusion(loaded: Mapping[str, Any]) -> dict[str, Any]:
    topic_rows = loaded["topic_scores"]
    meeting_rows = loaded["meeting_scores"]
    reference_topics = [row for row in topic_rows if row.get("arm") == "reference"]
    reference_meetings = [row for row in meeting_rows if row.get("arm") == "reference"]
    pre_external = [
        row for row in reference_meetings if row.get("role") == "external_holdout"
    ]
    matched = [
        row
        for row in reference_topics
        if int(row.get("hawkish_count", 0)) + int(row.get("dovish_count", 0)) > 0
    ]
    total_matches = sum(
        int(row.get("hawkish_count", 0)) + int(row.get("dovish_count", 0))
        for row in reference_topics
    )
    values = [float(row["score"]) for row in pre_external]
    if len(values) < 2:
        raise SentimentRegressionSuiteV2Error(
            "lexicon diagnostic lacks pre-external reference meetings"
        )
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    result = {
        "backend_id": sentiment_v1.LEXICON_BACKEND,
        "used_for_regression": False,
        "reason": "zero_reference_variance",
        "reference_topic_rows": len(reference_topics),
        "reference_meeting_rows": len(reference_meetings),
        "pre_external_reference_meeting_rows": len(pre_external),
        "reference_matched_topic_rows": len(matched),
        "reference_total_matches": total_matches,
        "reference_score_mean": sum(float(row["score"]) for row in reference_meetings)
        / len(reference_meetings),
        "pre_external_reference_score_standard_deviation_ddof1": math.sqrt(variance),
        "diagnostic_only": True,
        "backend_manifest": copy.deepcopy(dict(loaded["manifest_binding"])),
    }
    expected = {
        "reference_topic_rows": 2048,
        "reference_meeting_rows": 256,
        "pre_external_reference_meeting_rows": 128,
        "reference_matched_topic_rows": 0,
        "reference_total_matches": 0,
        "reference_score_mean": 0.0,
        "pre_external_reference_score_standard_deviation_ddof1": 0.0,
    }
    if any(result[key] != value for key, value in expected.items()):
        raise SentimentRegressionSuiteV2Error(
            "lexicon is not the frozen zero-reference-variance diagnostic"
        )
    return result


def _payload(
    *,
    inventory: Mapping[str, Any],
    lexicon: Mapping[str, Any],
    backends: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    exclusion = _lexicon_exclusion(lexicon)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "immutable": True,
        "regression_primary_backend": sentiment_v1.DISTIL_BACKEND,
        "regression_robustness_backends": [sentiment_v1.FINBERT_BACKEND],
        "construct_nonpooling": {
            sentiment_v1.DISTIL_BACKEND: (
                "monetary_policy_stance_hawkish_minus_dovish"
            ),
            sentiment_v1.FINBERT_BACKEND: ("financial_valence_positive_minus_negative"),
            "cross_construct_composite_score": False,
        },
        "legacy_v1_sentiment_suite": None,
        "legacy_v1_sentiment_suite_absence_reason": LEGACY_SUITE_ABSENCE_REASON,
        "v1_failure_diagnostic": _binding(V1_FAILURE_DIAGNOSTIC),
        "inventory_manifest": copy.deepcopy(dict(inventory["manifest_binding"])),
        "backend_manifests": {
            backend_id: copy.deepcopy(dict(backends[backend_id]["manifest_binding"]))
            for backend_id in neural_v2.NEURAL_BACKENDS
        },
        "excluded_backends": {sentiment_v1.LEXICON_BACKEND: exclusion},
        "implementation": _implementation_binding(),
    }


def seal_regression_suite(
    *,
    inventory_manifest: Path,
    lexicon_manifest: Path,
    distil_manifest: Path,
    finbert_manifest: Path,
    output_path: Path,
) -> dict[str, Any]:
    try:
        inventory = sentiment_v1.load_and_validate_inventory(inventory_manifest)
        lexicon = sentiment_v1.load_and_validate_meeting_scores(lexicon_manifest)
        backends = {
            sentiment_v1.DISTIL_BACKEND: neural_v2.load_and_validate_neural_backend(
                distil_manifest
            ),
            sentiment_v1.FINBERT_BACKEND: neural_v2.load_and_validate_neural_backend(
                finbert_manifest
            ),
        }
    except Exception as exc:
        raise SentimentRegressionSuiteV2Error(str(exc)) from exc
    if lexicon["manifest"]["backend"] != sentiment_v1.LEXICON_BACKEND:
        raise SentimentRegressionSuiteV2Error("excluded backend is not lexicon")
    if lexicon["manifest"]["inventory_manifest"] != inventory["manifest_binding"]:
        raise SentimentRegressionSuiteV2Error("lexicon inventory binding drift")
    if any(
        loaded["manifest"]["inventory_manifest"] != inventory["manifest_binding"]
        for loaded in backends.values()
    ):
        raise SentimentRegressionSuiteV2Error("neural inventory bindings disagree")
    value = seal_manifest(
        _payload(inventory=inventory, lexicon=lexicon, backends=backends)
    )
    try:
        sentiment_v1._write_readonly_json(output_path, value)
    except Exception as exc:
        raise SentimentRegressionSuiteV2Error(str(exc)) from exc
    return value


def load_and_validate_regression_suite(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = sentiment_v1._read_json(manifest_path)
    except Exception as exc:
        raise SentimentRegressionSuiteV2Error(str(exc)) from exc
    payload_sha = validate_manifest_integrity(manifest)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("immutable") is not True
        or manifest.get("regression_primary_backend") != sentiment_v1.DISTIL_BACKEND
        or manifest.get("regression_robustness_backends")
        != [sentiment_v1.FINBERT_BACKEND]
        or manifest.get("legacy_v1_sentiment_suite") is not None
        or manifest.get("legacy_v1_sentiment_suite_absence_reason")
        != LEGACY_SUITE_ABSENCE_REASON
        or manifest.get("construct_nonpooling")
        != {
            sentiment_v1.DISTIL_BACKEND: (
                "monetary_policy_stance_hawkish_minus_dovish"
            ),
            sentiment_v1.FINBERT_BACKEND: ("financial_valence_positive_minus_negative"),
            "cross_construct_composite_score": False,
        }
        or manifest.get("v1_failure_diagnostic") != _binding(V1_FAILURE_DIAGNOSTIC)
    ):
        raise SentimentRegressionSuiteV2Error("regression suite header drift")
    if manifest.get("implementation") != _implementation_binding():
        raise SentimentRegressionSuiteV2Error("regression suite implementation drift")
    inventory_binding = manifest.get("inventory_manifest")
    backend_bindings = manifest.get("backend_manifests")
    excluded = manifest.get("excluded_backends")
    if (
        not isinstance(inventory_binding, Mapping)
        or not isinstance(backend_bindings, Mapping)
        or not isinstance(excluded, Mapping)
        or set(excluded) != {sentiment_v1.LEXICON_BACKEND}
        or not isinstance(excluded.get(sentiment_v1.LEXICON_BACKEND), Mapping)
    ):
        raise SentimentRegressionSuiteV2Error("regression suite bindings missing")
    try:
        inventory = sentiment_v1.load_and_validate_inventory(
            Path(str(inventory_binding.get("path")))
        )
        backends = {
            backend_id: neural_v2.load_and_validate_neural_backend(
                Path(str(backend_bindings[backend_id]["path"]))
            )
            for backend_id in neural_v2.NEURAL_BACKENDS
        }
        lexicon_binding = excluded[sentiment_v1.LEXICON_BACKEND]["backend_manifest"]
        lexicon = sentiment_v1.load_and_validate_meeting_scores(
            Path(str(lexicon_binding["path"]))
        )
    except Exception as exc:
        raise SentimentRegressionSuiteV2Error(str(exc)) from exc
    if inventory_binding != inventory["manifest_binding"] or any(
        backend_bindings[backend_id] != backends[backend_id]["manifest_binding"]
        for backend_id in neural_v2.NEURAL_BACKENDS
    ):
        raise SentimentRegressionSuiteV2Error("regression suite source binding drift")
    rebuilt = seal_manifest(
        _payload(inventory=inventory, lexicon=lexicon, backends=backends)
    )
    if manifest != rebuilt:
        raise SentimentRegressionSuiteV2Error("regression suite payload drift")
    return {
        "manifest": manifest,
        "manifest_binding": _binding(manifest_path, payload_sha256=payload_sha),
        "inventory": inventory,
        "backends": backends,
        "excluded_backends": {
            sentiment_v1.LEXICON_BACKEND: copy.deepcopy(
                excluded[sentiment_v1.LEXICON_BACKEND]
            )
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--inventory-manifest", type=Path, required=True)
    seal.add_argument("--lexicon-manifest", type=Path, required=True)
    seal.add_argument("--distil-manifest", type=Path, required=True)
    seal.add_argument("--finbert-manifest", type=Path, required=True)
    seal.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "seal":
            value = seal_regression_suite(
                inventory_manifest=args.inventory_manifest,
                lexicon_manifest=args.lexicon_manifest,
                distil_manifest=args.distil_manifest,
                finbert_manifest=args.finbert_manifest,
                output_path=args.output,
            )
        else:
            value = load_and_validate_regression_suite(args.manifest)["manifest"]
    except SentimentRegressionSuiteV2Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(_canonical(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "SCHEMA_VERSION",
    "SentimentRegressionSuiteV2Error",
    "load_and_validate_regression_suite",
    "seal_regression_suite",
]
